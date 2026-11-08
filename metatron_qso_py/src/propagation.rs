//! Expectation values of large noisy circuits by Pauli propagation.

use metatron_qso::noise::learning::SparsePauliLindblad;
use metatron_qso::noise::{Channel, DeviceModel};
use metatron_qso::propagation::{
    DEFAULT_MAX_TERMS, Layer, PauliOperator, PropagationOptions, device_from_json, device_layers,
    program_circuit, propagate as propagate_core,
};
use metatron_qso::quantum::{Circuit, Pauli, PauliSum, Program};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

use crate::circuit::{PyCircuit, PyProgram};
use crate::learning::PySparsePauliLindblad;
use crate::noise::{PyChannel, PyDeviceModel};
use crate::state::PyPauliSum;
use crate::{Raise, RealArg, RealMatrixArg, check_finite, invalid, pydict, real_vec};

/// The default of `max_terms` as written in the signature and the stub.
const MAX_TERMS_DEFAULT: usize = 16_777_216;
const _: () = assert!(MAX_TERMS_DEFAULT == DEFAULT_MAX_TERMS);

/// A sum of Pauli strings as given in Python: text or `(coefficient,
/// label)` pairs, or a `PauliSum`.
enum Terms {
    Text(String),
    List(Vec<(f64, String)>),
    Sum(PauliSum),
}

/// One layer item as given in Python.
enum Item {
    /// A `Circuit`, a `Program` (as its circuit) or OpenQASM text.
    Circuit(Circuit),
    Qasm(String),
    Channel(Channel, Vec<usize>),
    Lindblad(SparsePauliLindblad, Vec<usize>, f64),
    Rates(Terms, f64),
}

/// The device as given in Python.
enum Device {
    Model(DeviceModel),
    Json(String),
}

/// The input state as given in Python.
enum State {
    Bits(String),
    Bloch(Vec<[f64; 3]>),
}

const LAYER_KINDS: &str = "a Circuit, a Program, OpenQASM text, (Channel, qubits), \
     (SparsePauliLindblad, qubits, factor) or (rates, factor)";

fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "object".into())
}

/// Rates or an observable that is not a `PauliSum`.
fn extract_terms(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Terms> {
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(Terms::Text(s.extract::<String>()?));
    }
    obj.extract::<Vec<(f64, String)>>()
        .map(Terms::List)
        .map_err(|_| {
            invalid(format!(
                "{what} must be a string such as '0.5*Z0 Z1 + X2' or a list of \
                 (coefficient, Pauli string) pairs, got {}",
                type_name(obj)
            ))
        })
}

fn extract_observable(obj: &Bound<'_, PyAny>) -> PyResult<Terms> {
    if let Ok(s) = obj.cast::<PyPauliSum>() {
        return Ok(Terms::Sum(s.get().inner.clone()));
    }
    extract_terms(obj, "the observable")
}

fn extract_item(obj: &Bound<'_, PyAny>) -> PyResult<Item> {
    if let Ok(c) = obj.cast::<PyCircuit>() {
        return Ok(Item::Circuit(c.try_borrow()?.inner.clone()));
    }
    if let Ok(p) = obj.cast::<PyProgram>() {
        return Ok(Item::Circuit(program_circuit(&p.get().inner).raise()?));
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(Item::Qasm(s.extract::<String>()?));
    }
    let Ok(t) = obj.cast::<PyTuple>() else {
        return Err(invalid(format!(
            "a layer must be {LAYER_KINDS}, got {}",
            type_name(obj)
        )));
    };
    let first = t.get_item(0).ok();
    // A tuple of circuits, programs or OpenQASM texts is meant as several
    // layers (no layer tuple is made of these alone).
    let is_circuit = |o: &Bound<'_, PyAny>| {
        o.cast::<PyCircuit>().is_ok()
            || o.cast::<PyProgram>().is_ok()
            || o.cast::<PyString>().is_ok()
    };
    if t.len() >= 2 && t.iter().all(|o| is_circuit(&o)) {
        return Err(invalid(
            "several layers must be given as a list, not a tuple (a tuple is one \
             (Channel, qubits), (SparsePauliLindblad, qubits, factor) or (rates, factor) layer)",
        ));
    }
    match (t.len(), first) {
        (3, Some(model)) => {
            let Ok(model) = model.cast::<PySparsePauliLindblad>() else {
                return Err(invalid(format!(
                    "a layer of three items must be (SparsePauliLindblad, qubits, factor), \
                     got a {} first",
                    type_name(&model)
                )));
            };
            let qubits: Vec<usize> = t.get_item(1)?.extract()?;
            let factor: f64 = t.get_item(2)?.extract()?;
            Ok(Item::Lindblad(model.get().inner.clone(), qubits, factor))
        }
        (2, Some(head)) => {
            if let Ok(channel) = head.cast::<PyChannel>() {
                let qubits: Vec<usize> = t.get_item(1)?.extract()?;
                return Ok(Item::Channel(channel.get().inner.clone(), qubits));
            }

            let rates = extract_terms(&head, "the rates of a (rates, factor) layer")?;
            let factor: f64 = t.get_item(1)?.extract()?;
            Ok(Item::Rates(rates, factor))
        }
        (len, _) => Err(invalid(format!(
            "a layer must be {LAYER_KINDS}, got a tuple of {len} items"
        ))),
    }
}

fn extract_state(obj: &Bound<'_, PyAny>) -> PyResult<State> {
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(State::Bits(s.extract::<String>()?));
    }
    let arr: RealMatrixArg<'_> = obj.extract().map_err(|_| {
        invalid(format!(
            "`state` must be a bit string or an (n, 3) array of Bloch vectors, got {}",
            type_name(obj)
        ))
    })?;
    let view = arr.as_array();
    if view.ncols() != 3 {
        return Err(invalid(format!(
            "`state` must have shape (n, 3), got ({}, {})",
            view.nrows(),
            view.ncols()
        )));
    }
    Ok(State::Bloch(
        view.rows()
            .into_iter()
            .map(|r| [r[0], r[1], r[2]])
            .collect(),
    ))
}

/// One more than the highest qubit an `X`, `Y` or `Z` factor names in
/// `text` (the grammar of `PauliSum`); 0 if none.
fn named_qubits(text: &str) -> usize {
    let bytes = text.as_bytes();
    let mut n = 0usize;
    let mut i = 0;
    while i < bytes.len() {
        let is_factor = matches!(bytes[i].to_ascii_uppercase(), b'X' | b'Y' | b'Z');
        i += 1;
        if is_factor {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            // An index too large for `usize` is reported by the parser.
            if let Ok(q) = text[start..i].parse::<usize>() {
                n = n.max(q.saturating_add(1));
            }
        }
    }
    n
}

fn terms_width(terms: &Terms) -> usize {
    match terms {
        Terms::Text(t) => named_qubits(t),
        Terms::List(l) => l.iter().map(|(_, s)| named_qubits(s)).max().unwrap_or(0),
        Terms::Sum(s) => s.min_qubits(),
    }
}

/// The operator of `terms` on `n` qubits.
fn operator(terms: &Terms, n: usize, what: &str) -> PyResult<PauliOperator> {
    match terms {
        Terms::Text(t) => PauliOperator::parse(n, t).raise(),
        Terms::Sum(s) => PauliOperator::from_sum(n, s).raise(),
        Terms::List(list) => {
            // Collected and merged at once (`add_term` is a sorted insert,
            // quadratic over many terms).
            let mut terms = Vec::with_capacity(list.len());
            for (c, label) in list {
                if !c.is_finite() {
                    return Err(invalid(format!(
                        "the coefficients of {what} must be finite"
                    )));
                }
                for (tc, factors) in PauliOperator::parse(n, label).raise()?.terms() {
                    terms.push((c * tc, factors));
                }
            }
            PauliOperator::from_terms(n, terms).raise()
        }
    }
}

fn label(factors: &[(usize, Pauli)]) -> String {
    if factors.is_empty() {
        return "I".into();
    }
    factors
        .iter()
        .map(|(q, p)| format!("{p:?}{q}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn max_qubit(qubits: &[usize]) -> usize {
    qubits
        .iter()
        .map(|&q| q.saturating_add(1))
        .max()
        .unwrap_or(0)
}

/// The circuit of an item (OpenQASM parsed and unrolled here).
fn item_circuit(item: Item) -> PyResult<Item> {
    match item {
        Item::Qasm(text) => Ok(Item::Circuit(
            Program::from_qasm(&text)
                .and_then(|p| program_circuit(&p))
                .raise()?,
        )),
        other => Ok(other),
    }
}

struct Request {
    items: Vec<Item>,
    observable: Terms,
    n_qubits: Option<usize>,
    device: Option<Device>,
    theta: Option<Vec<f64>>,
    state: Option<State>,
    options: PropagationOptions,
}

/// Everything after the arguments were read (runs without the GIL).
fn run(req: Request) -> PyResult<metatron_qso::propagation::PropagationResult> {
    let items = req
        .items
        .into_iter()
        .map(item_circuit)
        .collect::<PyResult<Vec<_>>>()?;
    let n = match req.n_qubits {
        Some(n) => n,
        None => {
            let mut n = terms_width(&req.observable);
            for item in &items {
                n = n.max(match item {
                    Item::Circuit(c) => c.n_qubits(),
                    Item::Channel(_, q) | Item::Lindblad(_, q, _) => max_qubit(q),
                    Item::Rates(r, _) => terms_width(r),
                    Item::Qasm(_) => 0,
                });
            }
            if let Some(State::Bits(b)) = &req.state {
                n = n.max(b.chars().count());
            }
            if let Some(State::Bloch(b)) = &req.state {
                n = n.max(b.len());
            }
            n.max(1)
        }
    };
    let state = match req.state {
        None => None,
        Some(State::Bloch(b)) => Some(b),
        Some(State::Bits(bits)) => Some(
            bits.chars()
                .rev()
                .map(|c| match c {
                    '0' => Ok([0.0, 0.0, 1.0]),
                    '1' => Ok([0.0, 0.0, -1.0]),
                    _ => Err(invalid(format!(
                        "`state` must be a string of '0' and '1', got '{bits}'"
                    ))),
                })
                .collect::<PyResult<Vec<_>>>()?,
        ),
    };
    if let Some(s) = &state
        && s.len() != n
    {
        return Err(invalid(format!(
            "`state` has {} qubits, the register {n}",
            s.len()
        )));
    }
    let device = match req.device {
        None => None,
        Some(Device::Model(d)) => Some(d),
        Some(Device::Json(text)) => Some(device_from_json(&text).raise()?),
    };
    let theta = req.theta.unwrap_or_default();
    let mut layers = Vec::with_capacity(items.len());
    for item in items {
        match item {
            Item::Circuit(c) => {
                let theta = if c.num_parameters() > 0 {
                    theta.clone()
                } else {
                    Vec::new()
                };
                match &device {
                    Some(d) => layers.extend(device_layers(&c, &theta, d).raise()?),
                    None => layers.push(Layer::Gates { circuit: c, theta }),
                }
            }
            Item::Channel(channel, qubits) => layers.push(Layer::channel(channel, &qubits)),
            Item::Lindblad(model, qubits, factor) => layers.push(Layer::pauli_lindblad(
                PauliOperator::from_lindblad(&model, &qubits, n).raise()?,
                factor,
            )),
            Item::Rates(rates, factor) => layers.push(Layer::pauli_lindblad(
                operator(&rates, n, "the rates")?,
                factor,
            )),
            Item::Qasm(_) => unreachable!("OpenQASM items are parsed above"),
        }
    }
    let observable = operator(&req.observable, n, "the observable")?;
    propagate_core(&layers, &observable, state.as_deref(), &req.options).raise()
}

/// `⟨observable⟩` after `layers`, by Pauli propagation: the observable is
/// carried backwards through the circuit as a table of Pauli strings on up
/// to 16384 qubits and read in the input state.
///
/// `layers` is one item or a list of items in time order, each a
/// `Circuit`, a `Program`, OpenQASM 2 or 3 text (final measurements are
/// left out, other measurements, resets and conditions refused),
/// `(Channel, qubits)`,
/// `(SparsePauliLindblad, qubits, factor)` or `(rates, factor)` with
/// `rates` a string such as `"0.01*X0 + 0.02*Z0 Z1"` or a list of
/// `(rate, Pauli string)` pairs; a Pauli–Lindblad layer multiplies every
/// rate by `factor` (`factor < 0` inverts the noise). With `device` (a
/// `DeviceModel` or its JSON text, which may describe up to 16384 qubits)
/// every circuit item runs with the device's noise. `theta` binds the
/// parameters of every circuit item that has parameters.
///
/// `observable` is a string, a `PauliSum` or a list of `(coefficient,
/// Pauli string)` pairs; `state` is `None` (`|0…0⟩`), a bit string (qubit
/// 0 on the right) or an `(n, 3)` array of Bloch vectors. `n_qubits`
/// defaults to the largest of the widest circuit, one more than the
/// highest qubit an item or the observable names, and the length of
/// `state`.
///
/// Strings with `|c| < min_coefficient` or more than `max_weight`
/// non-identity factors are dropped; holding more than `max_terms` strings
/// is an error. Returns a dict with `value`, `truncation_bound` (a bound
/// on the error of the dropped strings), `terms`, `peak_terms`,
/// `dropped_terms` and `operator` (the Heisenberg observable as
/// `(coefficient, Pauli string)` pairs if `return_operator`, else `None`).
#[pyfunction]
#[pyo3(signature = (
    layers, observable, *, n_qubits = None, device = None, theta = None, state = None,
    min_coefficient = 0.0, max_weight = None, max_terms = 16777216, return_operator = false
))]
#[allow(clippy::too_many_arguments)]
fn propagate<'py>(
    py: Python<'py>,
    layers: &Bound<'py, PyAny>,
    observable: &Bound<'py, PyAny>,
    n_qubits: Option<usize>,
    device: Option<&Bound<'py, PyAny>>,
    theta: Option<RealArg<'py>>,
    state: Option<&Bound<'py, PyAny>>,
    min_coefficient: f64,
    max_weight: Option<usize>,
    max_terms: usize,
    return_operator: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let items = match layers.cast::<PyList>() {
        Ok(list) => list
            .iter()
            .map(|item| extract_item(&item))
            .collect::<PyResult<Vec<_>>>()?,
        Err(_) => vec![extract_item(layers)?],
    };
    let device = match device {
        None => None,
        Some(d) => {
            if let Ok(model) = d.cast::<PyDeviceModel>() {
                Some(Device::Model(model.get().inner.clone()))
            } else if let Ok(text) = d.cast::<PyString>() {
                Some(Device::Json(text.extract::<String>()?))
            } else {
                return Err(invalid(format!(
                    "`device` must be a DeviceModel or its JSON text, got {}",
                    type_name(d)
                )));
            }
        }
    };
    let theta = theta.map(|t| real_vec(&t));
    if let Some(t) = &theta {
        check_finite("theta", t)?;
        let parametrised = items
            .iter()
            .any(|i| matches!(i, Item::Circuit(c) if c.num_parameters() > 0));
        if !parametrised {
            return Err(invalid("`theta` was given but no circuit has parameters"));
        }
    }
    let mut options = PropagationOptions::default()
        .with_min_coefficient(min_coefficient)
        .with_max_terms(max_terms)
        .with_operator(return_operator);
    if let Some(w) = max_weight {
        options = options.with_max_weight(w);
    }
    let req = Request {
        items,
        observable: extract_observable(observable)?,
        n_qubits,
        device,
        theta,
        state: state.map(extract_state).transpose()?,
        options,
    };
    let r = py.detach(|| run(req))?;
    let operator = r
        .operator
        .as_ref()
        .map(|op| op.terms().map(|(c, f)| (c, label(&f))).collect::<Vec<_>>());
    Ok(pydict!(py, {
        "value" => r.value,
        "truncation_bound" => r.truncation_bound,
        "terms" => r.terms,
        "peak_terms" => r.peak_terms,
        "dropped_terms" => r.dropped_terms,
        "operator" => operator,
    }))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(propagate, m)?)?;
    Ok(())
}
