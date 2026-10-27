//! Error mitigation: readout calibration, zero-noise extrapolation,
//! probabilistic error cancellation and Pauli twirling.

use std::collections::BTreeMap;

use metatron_qso::Rng;
use metatron_qso::mitigation::pec::{PecResult, pec_expectation as pec_core};
use metatron_qso::mitigation::twirl::{twirl as twirl_core, twirled_average as twirled_core};
use metatron_qso::mitigation::zne::{
    Extrapolation, Folding, extrapolate as extrapolate_core, fold as fold_core, zne as zne_core,
};
use metatron_qso::mitigation::{ReadoutCalibration, nearest_probability_distribution as nearest};
use metatron_qso::quantum::{Counts, bitstring};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::circuit::{PyCircuit, theta_or_empty};
use crate::estimation::{PyEstimate, backend, extract_estimate, wrap_estimate};
use crate::noise::{PyDeviceModel, PyNoiseModel};
use crate::state::PyPauliSum;
use crate::{Raise, RealArg, RealMatrixArg, check_finite, invalid, pydict};

/// Parse `{bitstring: value}` into basis indices (qubit `n−1` first).
fn parse_bits(key: &str) -> PyResult<usize> {
    if key.is_empty() || key.len() > 64 || !key.bytes().all(|b| b == b'0' || b == b'1') {
        return Err(invalid(format!("'{key}' is not a bit string")));
    }
    Ok(u64::from_str_radix(key, 2).expect("checked digits") as usize)
}

/// Counts keyed by bit strings of exactly `width` bits (if given).
fn parse_counts(counts: &Bound<'_, PyDict>, width: Option<usize>) -> PyResult<Counts> {
    let mut out = Counts::new();
    for (k, v) in counts.iter() {
        let key: String = k.extract()?;
        if let Some(w) = width
            && key.len() != w
        {
            return Err(invalid(format!(
                "bit string '{key}' does not have {w} bits"
            )));
        }
        let n: usize = v
            .extract()
            .map_err(|_| invalid("counts must be non-negative integers"))?;
        *out.entry(parse_bits(&key)?).or_insert(0) += n;
    }
    Ok(out)
}

fn quasi_dict<'py>(
    py: Python<'py>,
    q: &BTreeMap<usize, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (&b, &p) in q {
        d.set_item(bitstring(b, n), p)?;
    }
    Ok(d)
}

/// Readout calibration: per-qubit (tensored) or full confusion matrix.
#[pyclass(name = "ReadoutCalibration", module = "metatron_qso", frozen)]
#[derive(Clone)]
pub struct PyReadoutCalibration {
    pub inner: ReadoutCalibration,
}

fn calibration(r: metatron_qso::Result<ReadoutCalibration>) -> PyResult<PyReadoutCalibration> {
    r.raise().map(|inner| PyReadoutCalibration { inner })
}

#[pymethods]
impl PyReadoutCalibration {
    /// Tensored calibration from known per-qubit errors
    /// `[(P(1|0), P(0|1)), …]` (qubit 0 first).
    #[staticmethod]
    fn from_errors(errors: Vec<(f64, f64)>) -> PyResult<Self> {
        calibration(ReadoutCalibration::from_errors(&errors))
    }

    /// Full calibration from the confusion matrix `A[measured, prepared]`
    /// (columns are probability distributions).
    #[staticmethod]
    fn from_matrix(n_qubits: usize, matrix: RealMatrixArg<'_>) -> PyResult<Self> {
        let view = matrix.as_array();
        let (rows, cols) = view.dim();
        if rows != cols {
            return Err(invalid("the confusion matrix must be square"));
        }
        // Column-major A[measured][prepared].
        let data: Vec<f64> = (0..cols)
            .flat_map(|c| (0..rows).map(move |r| (r, c)))
            .map(|(r, c)| view[(r, c)])
            .collect();
        calibration(ReadoutCalibration::from_matrix(n_qubits, data))
    }

    /// Measure a tensored calibration (|0…0⟩ and |1…1⟩, `shots` each).
    #[staticmethod]
    #[pyo3(signature = (n_qubits, shots, seed = 0, noise = None, device = None, trajectories = None))]
    fn tensored(
        py: Python<'_>,
        n_qubits: usize,
        shots: usize,
        seed: u64,
        noise: Option<&PyNoiseModel>,
        device: Option<&PyDeviceModel>,
        trajectories: Option<usize>,
    ) -> PyResult<Self> {
        crate::check_qubits(n_qubits)?;
        let b = backend(noise, device, trajectories)?;
        calibration(py.detach(|| ReadoutCalibration::tensored(b.as_ref(), n_qubits, shots, seed)))
    }

    /// Measure a full calibration (all 2ⁿ basis states, `shots` each).
    #[staticmethod]
    #[pyo3(signature = (n_qubits, shots, seed = 0, noise = None, device = None, trajectories = None))]
    fn full(
        py: Python<'_>,
        n_qubits: usize,
        shots: usize,
        seed: u64,
        noise: Option<&PyNoiseModel>,
        device: Option<&PyDeviceModel>,
        trajectories: Option<usize>,
    ) -> PyResult<Self> {
        crate::check_qubits(n_qubits)?;
        let b = backend(noise, device, trajectories)?;
        calibration(py.detach(|| ReadoutCalibration::full(b.as_ref(), n_qubits, shots, seed)))
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    /// Per-qubit errors `(P(1|0), P(0|1))`.
    fn qubit_errors(&self) -> Vec<(f64, f64)> {
        self.inner.qubit_errors()
    }

    /// Quasi-probabilities `{bitstring: q}` from measured counts.
    fn mitigate<'py>(
        &self,
        py: Python<'py>,
        counts: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let n = self.inner.n_qubits();
        let counts = parse_counts(counts, Some(n))?;
        let cal = &self.inner;
        let q = py.detach(|| cal.mitigate(&counts)).raise()?;
        quasi_dict(py, &q, n)
    }

    fn __repr__(&self) -> String {
        format!("ReadoutCalibration(n_qubits={})", self.inner.n_qubits())
    }
}

/// The probability distribution nearest to quasi-probabilities (bit strings
/// of one length, summing to one), Smolin, Gambetta and Smith 2012.
#[pyfunction]
fn nearest_probability_distribution<'py>(
    py: Python<'py>,
    quasi: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut map = BTreeMap::new();
    let mut width = None;
    for (k, v) in quasi.iter() {
        let key: String = k.extract()?;
        if *width.get_or_insert(key.len()) != key.len() {
            return Err(invalid("all bit strings must have the same length"));
        }
        let p: f64 = v.extract()?;
        if !p.is_finite() {
            return Err(invalid("quasi-probabilities must be finite"));
        }
        if map.insert(parse_bits(&key)?, p).is_some() {
            return Err(invalid(format!("bit string '{key}' appears twice")));
        }
    }
    let total: f64 = map.values().sum();
    if (total - 1.0).abs() > 1e-6 {
        return Err(invalid(format!(
            "quasi-probabilities must sum to 1, got {total}"
        )));
    }
    quasi_dict(py, &nearest(&map), width.unwrap_or(0))
}

fn folding(name: &str, seed: u64) -> PyResult<Folding> {
    match name {
        "global" => Ok(Folding::Global),
        "random" => Ok(Folding::Random { seed }),
        other => Err(invalid(format!(
            "folding must be 'global' or 'random', got '{other}'"
        ))),
    }
}

pub(crate) fn extrapolation(
    method: &str,
    degree: Option<usize>,
    asymptote: f64,
) -> PyResult<Extrapolation> {
    if degree.is_some() && method != "polynomial" {
        return Err(invalid("`degree` applies to polynomial extrapolation only"));
    }
    match (method, degree) {
        ("linear", _) => Ok(Extrapolation::Linear),
        ("richardson", _) => Ok(Extrapolation::Richardson),
        ("polynomial", Some(degree)) => Ok(Extrapolation::Polynomial { degree }),
        ("polynomial", None) => Err(invalid("polynomial extrapolation needs `degree`")),
        ("exponential", _) => {
            if !asymptote.is_finite() {
                return Err(invalid("asymptote must be finite"));
            }
            Ok(Extrapolation::Exponential { asymptote })
        }
        (other, _) => Err(invalid(format!(
            "method must be 'linear', 'richardson', 'polynomial' or 'exponential', got '{other}'"
        ))),
    }
}

/// Fold `circuit` to amplify its noise by about `scale`; returns the folded
/// circuit and the realised scale. `folding`: "global" or "random".
#[pyfunction]
#[pyo3(signature = (circuit, scale, folding = "global", seed = 0))]
fn fold(
    py: Python<'_>,
    circuit: &PyCircuit,
    scale: f64,
    folding: &str,
    seed: u64,
) -> PyResult<(PyCircuit, f64)> {
    let folding = self::folding(folding, seed)?;
    let c = &circuit.inner;
    let (c, s) = py.detach(|| fold_core(c, scale, folding)).raise()?;
    Ok((PyCircuit::wrap(c)?, s))
}

fn estimates(values: &Bound<'_, PyAny>) -> PyResult<Vec<metatron_qso::estimation::Estimate>> {
    values
        .try_iter()?
        .map(|v| extract_estimate(&v?, "each value"))
        .collect()
}

/// Extrapolate values (floats or Estimates) at `scales` to zero noise.
/// `method`: "linear", "richardson", "polynomial" (with `degree`) or
/// "exponential" (with `asymptote`).
#[pyfunction]
#[pyo3(signature = (scales, values, method = "richardson", degree = None, asymptote = 0.0))]
fn extrapolate(
    scales: Vec<f64>,
    values: &Bound<'_, PyAny>,
    method: &str,
    degree: Option<usize>,
    asymptote: f64,
) -> PyResult<PyEstimate> {
    check_finite("scales", &scales)?;
    let values = estimates(values)?;
    extrapolate_core(&scales, &values, extrapolation(method, degree, asymptote)?)
        .raise()
        .map(wrap_estimate)
}

/// Zero-noise extrapolation: fold `circuit` to each scale, evaluate it with
/// `execute(folded) -> float | Estimate` and extrapolate to zero noise.
/// Returns a dict with `estimate`, `scales` (realised) and `values`.
#[pyfunction]
#[pyo3(signature = (
    circuit, scales, execute, folding = "global", seed = 0, method = "richardson",
    degree = None, asymptote = 0.0
))]
#[allow(clippy::too_many_arguments)]
fn zne<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    scales: Vec<f64>,
    execute: &Bound<'py, PyAny>,
    folding: &str,
    seed: u64,
    method: &str,
    degree: Option<usize>,
    asymptote: f64,
) -> PyResult<Bound<'py, PyDict>> {
    check_finite("scales", &scales)?;
    let folding = self::folding(folding, seed)?;
    let method = extrapolation(method, degree, asymptote)?;
    let mut error: Option<PyErr> = None;
    let result = zne_core(&circuit.inner, &scales, folding, method, |folded| {
        let call = PyCircuit::wrap(folded.clone())
            .and_then(|c| execute.call1((c,)))
            .and_then(|v| extract_estimate(&v, "the value returned by `execute`"));
        call.map_err(|e| {
            error = Some(e);
            metatron_qso::QsoError::invalid("execute", "the callable raised an exception")
        })
    });
    if let Some(e) = error {
        return Err(e);
    }
    let r = result.raise()?;
    Ok(pydict!(py, {
        "estimate" => wrap_estimate(r.estimate),
        "scales" => r.scales,
        "values" => r.values.into_iter().map(wrap_estimate).collect::<Vec<_>>(),
    }))
}

/// Probabilistic error cancellation of the Pauli noise of `device`:
/// returns a dict with `estimate`, `overhead` (Γ) and `noisy_operations`.
#[pyfunction]
#[pyo3(signature = (circuit, observable, device, samples, theta = None, seed = 0))]
fn pec_expectation<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    observable: &PyPauliSum,
    device: &PyDeviceModel,
    samples: usize,
    theta: Option<RealArg<'_>>,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let theta = theta_or_empty(theta);
    check_finite("theta", &theta)?;
    let (c, h, d) = (&circuit.inner, &observable.inner, &device.inner);
    let r: PecResult = py
        .detach(|| pec_core(c, &theta, d, h, samples, seed))
        .raise()?;
    Ok(pydict!(py, {
        "estimate" => wrap_estimate(r.estimate),
        "overhead" => r.overhead,
        "noisy_operations" => r.noisy_operations,
    }))
}

/// `circuit` with random Pauli frames around every cx, cy, cz and swap
/// (the ideal circuit is unchanged up to a global phase).
#[pyfunction]
#[pyo3(signature = (circuit, seed = 0))]
fn twirl(circuit: &PyCircuit, seed: u64) -> PyResult<PyCircuit> {
    PyCircuit::wrap(twirl_core(&circuit.inner, &mut Rng::seed_from_u64(seed)))
}

/// Average of `execute(twirled) -> float | Estimate` over `count` twirled
/// copies of `circuit`.
#[pyfunction]
#[pyo3(signature = (circuit, count, execute, seed = 0))]
fn twirled_average(
    circuit: &PyCircuit,
    count: usize,
    execute: &Bound<'_, PyAny>,
    seed: u64,
) -> PyResult<PyEstimate> {
    let mut error: Option<PyErr> = None;
    let result = twirled_core(&circuit.inner, count, seed, |t| {
        let call = PyCircuit::wrap(t.clone())
            .and_then(|c| execute.call1((c,)))
            .and_then(|v| extract_estimate(&v, "the value returned by `execute`"));
        call.map_err(|e| {
            error = Some(e);
            metatron_qso::QsoError::invalid("execute", "the callable raised an exception")
        })
    });
    if let Some(e) = error {
        return Err(e);
    }
    result.raise().map(wrap_estimate)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyReadoutCalibration>()?;
    m.add_function(wrap_pyfunction!(nearest_probability_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(fold, m)?)?;
    m.add_function(wrap_pyfunction!(extrapolate, m)?)?;
    m.add_function(wrap_pyfunction!(zne, m)?)?;
    m.add_function(wrap_pyfunction!(pec_expectation, m)?)?;
    m.add_function(wrap_pyfunction!(twirl, m)?)?;
    m.add_function(wrap_pyfunction!(twirled_average, m)?)?;
    Ok(())
}
