//! Noise learning: sparse Pauli–Lindblad models of a layer of two-qubit
//! Clifford gates, and error cancellation and amplification with them.

use std::collections::BTreeMap;

use metatron_qso::noise::learning::{
    LearningOptions, SparsePauliLindblad, learn_layer as learn_core, pea_layered as pea_core,
    pec_layered as pec_core,
};
use metatron_qso::quantum::{Circuit, PauliString, PauliSum};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::circuit::PyCircuit;
use crate::estimation::wrap_estimate;
use crate::mitigation::extrapolation;
use crate::noise::PyDeviceModel;
use crate::state::PyPauliSum;
use crate::{Raise, check_finite, invalid, pydict};

/// A single Pauli string such as `"X0 Z1"`.
fn parse_string(text: &str) -> PyResult<PauliString> {
    let sum: PauliSum = text.parse().raise()?;
    match sum.terms() {
        [(c, s)] if *c == 1.0 => Ok(*s),
        _ => Err(invalid(format!("'{text}' is not a single Pauli string"))),
    }
}

/// A sparse Pauli–Lindblad noise model: generators `P_k` with rates
/// `λ_k ≥ 0`, fidelities `f_b = exp(−2 Σ_{P_k anticommutes with b} λ_k)`.
#[pyclass(name = "SparsePauliLindblad", module = "metatron_qso", frozen)]
#[derive(Clone)]
pub struct PySparsePauliLindblad {
    pub inner: SparsePauliLindblad,
}

#[pymethods]
impl PySparsePauliLindblad {
    /// A model on `n_qubits` from generators such as `"Z0 Z1"` and their
    /// rates.
    #[new]
    fn new(n_qubits: usize, generators: Vec<String>, rates: Vec<f64>) -> PyResult<Self> {
        let generators = generators
            .iter()
            .map(|g| parse_string(g))
            .collect::<PyResult<Vec<_>>>()?;
        SparsePauliLindblad::new(n_qubits, generators, rates)
            .raise()
            .map(|inner| Self { inner })
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits
    }

    #[getter]
    fn generators(&self) -> Vec<String> {
        self.inner
            .generators
            .iter()
            .map(|g| g.to_string())
            .collect()
    }

    #[getter]
    fn rates(&self) -> Vec<f64> {
        self.inner.rates.clone()
    }

    /// Pauli fidelity of `pauli` (e.g. `"X0 Y2"`).
    fn fidelity(&self, pauli: &str) -> PyResult<f64> {
        Ok(self.inner.fidelity(&parse_string(pauli)?))
    }

    /// Sampling overhead `γ = exp(2 Σ λ_k)` of cancelling the noise once.
    fn overhead(&self) -> f64 {
        self.inner.overhead()
    }

    /// The model with every rate multiplied by `factor`.
    fn scaled(&self, factor: f64) -> PyResult<Self> {
        self.inner
            .scaled(factor)
            .raise()
            .map(|inner| Self { inner })
    }

    /// Pauli error probabilities `{string: p}`, the identity (`"I"`)
    /// included.
    fn error_probabilities(&self) -> PyResult<BTreeMap<String, f64>> {
        Ok(self
            .inner
            .error_probabilities()
            .raise()?
            .into_iter()
            .map(|(p, prob)| (p.to_string(), prob))
            .collect())
    }

    fn __repr__(&self) -> String {
        format!(
            "SparsePauliLindblad(n_qubits={}, generators={})",
            self.inner.n_qubits,
            self.inner.generators.len()
        )
    }
}

fn blocks_of(blocks: &[PyRef<'_, PyCircuit>]) -> Vec<Circuit> {
    blocks.iter().map(|b| b.inner.clone()).collect()
}

/// Learn the noise of the twirled `layer` (cx, cy, cz and swap gates on
/// disjoint qubits) on `device` (`depths` default to 2, 4, 8, 16, 32). Returns a dict with `model`,
/// `std_errors`, `products` (`(string, fitted f_b·f_UbU, standard error)`)
/// and `symmetrised` (strings whose fidelity was taken equal to that of
/// their image under the layer).
#[pyfunction]
#[pyo3(signature = (
    layer, device, depths = None, twirls = 16, shots = 256, edges = None,
    seed = 0
))]
#[allow(clippy::too_many_arguments)]
fn learn_layer<'py>(
    py: Python<'py>,
    layer: &PyCircuit,
    device: &PyDeviceModel,
    depths: Option<Vec<usize>>,
    twirls: usize,
    shots: usize,
    edges: Option<Vec<(usize, usize)>>,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let options = LearningOptions {
        depths: depths.unwrap_or_else(|| LearningOptions::default().depths),
        twirls,
        shots,
        edges,
        seed,
    };
    let (l, d) = (&layer.inner, &device.inner);
    let r = py.detach(|| learn_core(l, d, &options)).raise()?;
    Ok(pydict!(py, {
        "model" => PySparsePauliLindblad { inner: r.model },
        "std_errors" => r.std_errors,
        "products" => r
            .products
            .into_iter()
            .map(|(p, f, se)| (p.to_string(), f, se))
            .collect::<Vec<_>>(),
        "symmetrised" => r.symmetrised.iter().map(|p| p.to_string()).collect::<Vec<_>>(),
    }))
}

/// Error-cancelled `⟨observable⟩` of `B₀ L B₁ L … L B_m` (`blocks`, `L`
/// the twirled `layer`) on `device`, cancelling the layer noise with
/// `model`. Returns a dict with `estimate`, `overhead` and
/// `noisy_operations` (the number of layers).
#[pyfunction]
#[pyo3(signature = (blocks, layer, model, device, observable, samples, seed = 0))]
#[allow(clippy::too_many_arguments)]
fn pec_layered<'py>(
    py: Python<'py>,
    blocks: Vec<PyRef<'py, PyCircuit>>,
    layer: &PyCircuit,
    model: &PySparsePauliLindblad,
    device: &PyDeviceModel,
    observable: &PyPauliSum,
    samples: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let blocks = blocks_of(&blocks);
    let (l, m, d, h) = (&layer.inner, &model.inner, &device.inner, &observable.inner);
    let r = py
        .detach(|| pec_core(&blocks, l, m, d, h, samples, seed))
        .raise()?;
    Ok(pydict!(py, {
        "estimate" => wrap_estimate(r.estimate),
        "overhead" => r.overhead,
        "noisy_operations" => r.noisy_operations,
    }))
}

/// Zero-noise extrapolation by probabilistic error amplification: every
/// rate of `model` is multiplied by each of `scales` (≥ 1) by inserting
/// Paulis after the layers, and the values are extrapolated to zero
/// noise. Returns a dict with `estimate`, `scales` and `values`.
#[pyfunction]
#[pyo3(signature = (
    blocks, layer, model, device, observable, scales, samples, method = "richardson",
    degree = None, asymptote = 0.0, seed = 0
))]
#[allow(clippy::too_many_arguments)]
fn pea_layered<'py>(
    py: Python<'py>,
    blocks: Vec<PyRef<'py, PyCircuit>>,
    layer: &PyCircuit,
    model: &PySparsePauliLindblad,
    device: &PyDeviceModel,
    observable: &PyPauliSum,
    scales: Vec<f64>,
    samples: usize,
    method: &str,
    degree: Option<usize>,
    asymptote: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    check_finite("scales", &scales)?;
    let method = extrapolation(method, degree, asymptote)?;
    let blocks = blocks_of(&blocks);
    let (l, m, d, h) = (&layer.inner, &model.inner, &device.inner, &observable.inner);
    let r = py
        .detach(|| pea_core(&blocks, l, m, d, h, &scales, samples, method, seed))
        .raise()?;
    Ok(pydict!(py, {
        "estimate" => wrap_estimate(r.estimate),
        "scales" => r.scales,
        "values" => r.values.into_iter().map(wrap_estimate).collect::<Vec<_>>(),
    }))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySparsePauliLindblad>()?;
    m.add_function(wrap_pyfunction!(learn_layer, m)?)?;
    m.add_function(wrap_pyfunction!(pec_layered, m)?)?;
    m.add_function(wrap_pyfunction!(pea_layered, m)?)?;
    Ok(())
}
