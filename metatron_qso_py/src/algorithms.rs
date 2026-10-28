//! Models, ansätze, gradients, VQE, QAOA, Grover, QFT and the classifier.

use metatron_qso::algorithms::ansatz::{self, Entanglement, Rotation};
use metatron_qso::algorithms::classifier::VariationalClassifier;
use metatron_qso::algorithms::gradients::{self, GradientMethod};
use metatron_qso::algorithms::graph_classifier::GraphClassifier;
use metatron_qso::algorithms::qaoa::{self, MaxCutQaoa, QaoaConfig, QaoaResult};
use metatron_qso::algorithms::{Vqe, grover, models, qft};
use metatron_qso::graph::Graph;
use metatron_qso::optimize::{Method, Options};
use metatron_qso::quantum::{Precision, StateVector32, bitstring};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::circuit::PyCircuit;
use crate::graph::PyGraph;
use crate::state::{PyPauliSum, PyStateVector, wrap_sum};
use crate::{
    ComplexArg, ComplexMatrixArg, Raise, RealArg, RealMatrixArg, check_finite, check_qubits,
    complex_vec, invalid, pydict, real_array, real_vec, square_complex,
};

/// One index or a list of indices.
#[derive(FromPyObject)]
pub enum Indices {
    One(usize),
    Many(Vec<usize>),
}

impl Indices {
    pub fn into_vec(self) -> Vec<usize> {
        match self {
            Indices::One(i) => vec![i],
            Indices::Many(v) => v,
        }
    }
}

/// One name or a list of names.
#[derive(FromPyObject)]
enum Names {
    One(String),
    Many(Vec<String>),
}

fn entanglement(name: &str) -> PyResult<Entanglement> {
    match name.to_ascii_lowercase().as_str() {
        "linear" => Ok(Entanglement::Linear),
        "circular" => Ok(Entanglement::Circular),
        "full" => Ok(Entanglement::Full),
        _ => Err(invalid(format!(
            "unknown entanglement '{name}' (expected linear, circular or full)"
        ))),
    }
}

fn rotation(name: &str) -> PyResult<Rotation> {
    match name.to_ascii_lowercase().as_str() {
        "rx" => Ok(Rotation::RX),
        "ry" => Ok(Rotation::RY),
        "rz" => Ok(Rotation::RZ),
        _ => Err(invalid(format!(
            "unknown rotation '{name}' (expected rx, ry or rz)"
        ))),
    }
}

pub(crate) fn gradient_method(name: &str, step: f64) -> PyResult<GradientMethod> {
    match name.to_ascii_lowercase().replace('-', "_").as_str() {
        "adjoint" => Ok(GradientMethod::Adjoint),
        "parameter_shift" => Ok(GradientMethod::ParameterShift),
        "finite_difference" => {
            if !(step.is_finite() && step > 0.0) {
                return Err(invalid("finite-difference step must be positive"));
            }
            Ok(GradientMethod::FiniteDifference { step })
        }
        _ => Err(invalid(format!(
            "unknown gradient method '{name}' (expected adjoint, parameter_shift or \
             finite_difference)"
        ))),
    }
}

fn method(name: &str) -> PyResult<Method> {
    Method::from_name(name).raise()
}

fn options(max_iterations: usize) -> Options {
    Options {
        max_iterations,
        ..Options::default()
    }
}

// ---------------------------------------------------------------------------
// Model Hamiltonians

fn check_chain(n: usize, couplings: &[f64]) -> PyResult<()> {
    if n == 0 || n > 64 {
        return Err(invalid("chain length must lie in 1..=64"));
    }
    check_finite("couplings", couplings)
}

/// H₂ (STO-3G, 0.735 Å, parity mapping, 2 qubits) with reference data.
#[pyfunction]
fn h2_sto3g(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let h2 = models::h2_sto3g();
    Ok(pydict!(py, {
        "hamiltonian" => wrap_sum(h2.hamiltonian),
        "nuclear_repulsion" => h2.nuclear_repulsion,
        "reference_total_energy" => h2.reference_total_energy,
        "n_qubits" => h2.n_qubits,
        "name" => h2.name,
    }))
}

/// Transverse-field Ising chain `H = −J Σ Z_i Z_{i+1} − h Σ X_i`.
#[pyfunction]
#[pyo3(signature = (n, j, h, periodic = false))]
fn transverse_field_ising(n: usize, j: f64, h: f64, periodic: bool) -> PyResult<PyPauliSum> {
    check_chain(n, &[j, h])?;
    Ok(wrap_sum(models::transverse_field_ising(n, j, h, periodic)))
}

/// XXZ chain `H = J Σ (X_i X_{i+1} + Y_i Y_{i+1} + Δ Z_i Z_{i+1})`.
#[pyfunction]
#[pyo3(signature = (n, j, delta, periodic = false))]
fn heisenberg_xxz(n: usize, j: f64, delta: f64, periodic: bool) -> PyResult<PyPauliSum> {
    check_chain(n, &[j, delta])?;
    Ok(wrap_sum(models::heisenberg_xxz(n, j, delta, periodic)))
}

/// MaxCut cost `C = Σ w_uv (1 − Z_u Z_v)/2` (eigenvalue = cut weight).
#[pyfunction]
fn maxcut_cost(graph: &PyGraph) -> PyResult<PyPauliSum> {
    models::maxcut_cost(&graph.inner).raise().map(wrap_sum)
}

// ---------------------------------------------------------------------------
// Ansätze and gradients

/// RY layers with CX entanglers (`n·(layers+1)` parameters).
#[pyfunction]
#[pyo3(signature = (n, layers, entanglement = "linear"))]
fn real_amplitudes(n: usize, layers: usize, entanglement: &str) -> PyResult<PyCircuit> {
    check_qubits(n)?;
    let e = self::entanglement(entanglement)?;
    PyCircuit::wrap(ansatz::real_amplitudes(n, layers, e))
}

/// RY·RZ layers with CX entanglers (`2n·(layers+1)` parameters).
#[pyfunction]
#[pyo3(signature = (n, layers, entanglement = "linear"))]
fn efficient_su2(n: usize, layers: usize, entanglement: &str) -> PyResult<PyCircuit> {
    check_qubits(n)?;
    let e = self::entanglement(entanglement)?;
    PyCircuit::wrap(ansatz::efficient_su2(n, layers, e))
}

/// `layers` × (rotations on every qubit, CX entanglers) plus a final
/// rotation layer. `rotations`: any of `rx`, `ry`, `rz` (default
/// `("ry", "rz")`); `entanglement`: `linear`, `circular` or `full`.
#[pyfunction]
#[pyo3(signature = (n, layers, rotations = None, entanglement = "linear"))]
fn hardware_efficient(
    n: usize,
    layers: usize,
    rotations: Option<Names>,
    entanglement: &str,
) -> PyResult<PyCircuit> {
    check_qubits(n)?;
    let names = match rotations {
        None => vec!["ry".to_string(), "rz".to_string()],
        Some(Names::One(s)) => vec![s],
        Some(Names::Many(v)) => v,
    };
    let rotations = names
        .iter()
        .map(|s| rotation(s))
        .collect::<PyResult<Vec<_>>>()?;
    if rotations.is_empty() {
        return Err(invalid("at least one rotation axis required"));
    }
    let e = self::entanglement(entanglement)?;
    PyCircuit::wrap(ansatz::hardware_efficient(n, layers, &rotations, e))
}

/// Gate-level QAOA circuit for MaxCut with parameters `[γ…, β…]`.
#[pyfunction]
fn qaoa_maxcut_circuit(graph: &PyGraph, p: usize) -> PyResult<PyCircuit> {
    check_qubits(graph.inner.num_nodes())?;
    PyCircuit::wrap(ansatz::qaoa_maxcut_circuit(&graph.inner, p).raise()?)
}

/// `"double"` or `"single"`.
fn precision(name: &str) -> PyResult<Precision> {
    match name {
        "double" => Ok(Precision::Double),
        "single" => Ok(Precision::Single),
        other => Err(invalid(format!(
            "unknown precision {other:?}; use \"double\" or \"single\""
        ))),
    }
}

/// `⟨ψ(θ)|H|ψ(θ)⟩` and its gradient (`adjoint`, `parameter_shift` or
/// `finite_difference` with step `step`). With `precision="single"` the
/// state is held in single precision (half the memory) and the gradient
/// is adjoint; sums are taken in double precision.
#[pyfunction]
#[pyo3(signature = (circuit, observable, theta, method = "adjoint", step = 1e-5, initial_state = None, precision = "double"))]
#[allow(clippy::too_many_arguments)]
fn expectation_and_gradient<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    observable: &PyPauliSum,
    theta: RealArg<'_>,
    method: &str,
    step: f64,
    initial_state: Option<&PyStateVector>,
    precision: &str,
) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
    let theta = real_vec(&theta);
    check_finite("theta", &theta)?;
    let m = gradient_method(method, step)?;
    let precision = self::precision(precision)?;
    if precision == Precision::Single && m != GradientMethod::Adjoint {
        return Err(invalid("single precision supports adjoint gradients only"));
    }
    let (c, h) = (&circuit.inner, &observable.inner);
    let init = initial_state.map(|s| &s.inner);
    let (value, grad) = py
        .detach(|| match precision {
            Precision::Double => gradients::expectation_and_gradient(c, h, &theta, init, m),
            _ => {
                let init = init.map(StateVector32::from_state);
                gradients::expectation_and_gradient_single(c, h, &theta, init.as_ref())
            }
        })
        .raise()?;
    Ok((value, real_array(py, grad)))
}

/// Values `(k,)` and gradients `(k, p)` of a batch.
type ValuesAndGradients<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>);

/// Parameter vectors as the rows of a 2-D array.
fn rows(thetas: &RealMatrixArg<'_>) -> PyResult<Vec<Vec<f64>>> {
    let view = thetas.as_array();
    let out: Vec<Vec<f64>> = view.rows().into_iter().map(|r| r.to_vec()).collect();
    for row in &out {
        check_finite("thetas", row)?;
    }
    Ok(out)
}

/// `⟨H⟩` for every row of `thetas` (shape `(k, num_parameters)`); below 16
/// qubits the rows are evaluated in parallel, one per thread.
#[pyfunction]
#[pyo3(signature = (circuit, observable, thetas, initial_state = None))]
fn expectation_batch<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    observable: &PyPauliSum,
    thetas: RealMatrixArg<'_>,
    initial_state: Option<&PyStateVector>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let thetas = rows(&thetas)?;
    let (c, h) = (&circuit.inner, &observable.inner);
    let init = initial_state.map(|s| &s.inner);
    let values = py
        .detach(|| gradients::expectation_batch(c, h, &thetas, init))
        .raise()?;
    Ok(real_array(py, values))
}

/// Values `(k,)` and gradients `(k, num_parameters)` for every row of
/// `thetas`, in parallel as for `expectation_batch`.
#[pyfunction]
#[pyo3(signature = (circuit, observable, thetas, method = "adjoint", step = 1e-5, initial_state = None))]
fn expectation_and_gradient_batch<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    observable: &PyPauliSum,
    thetas: RealMatrixArg<'_>,
    method: &str,
    step: f64,
    initial_state: Option<&PyStateVector>,
) -> PyResult<ValuesAndGradients<'py>> {
    let thetas = rows(&thetas)?;
    let m = gradient_method(method, step)?;
    let (c, h) = (&circuit.inner, &observable.inner);
    let init = initial_state.map(|s| &s.inner);
    let results = py
        .detach(|| gradients::expectation_and_gradient_batch(c, h, &thetas, init, m))
        .raise()?;
    let p = c.num_parameters();
    let values: Vec<f64> = results.iter().map(|r| r.0).collect();
    let grads = numpy::ndarray::Array2::from_shape_fn((results.len(), p), |(k, j)| results[k].1[j]);
    Ok((real_array(py, values), grads.into_pyarray(py)))
}

// ---------------------------------------------------------------------------
// VQE

/// Variational eigensolver: minimise ⟨H⟩ over the ansatz parameters from
/// `restarts` seeded random starts (run in parallel).
///
/// `method`: `lbfgs`, `adam`, `gradient_descent`, `nelder_mead` or `spsa`;
/// `gradient`: `adjoint`, `parameter_shift` or `finite_difference`.
#[pyfunction]
#[pyo3(signature = (
    hamiltonian,
    ansatz,
    method = "lbfgs",
    restarts = 4,
    seed = 0,
    max_iterations = 1000,
    gradient = "adjoint",
    initial_parameters = None,
    initial_state = None,
    precision = "double",
))]
#[allow(clippy::too_many_arguments)]
fn run_vqe<'py>(
    py: Python<'py>,
    hamiltonian: &PyPauliSum,
    ansatz: &PyCircuit,
    method: &str,
    restarts: usize,
    seed: u64,
    max_iterations: usize,
    gradient: &str,
    initial_parameters: Option<RealArg<'_>>,
    initial_state: Option<&PyStateVector>,
    precision: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let precision = self::precision(precision)?;
    let gradient = gradient_method(gradient, 1e-5)?;
    if precision == Precision::Single && gradient != GradientMethod::Adjoint {
        return Err(invalid("single precision supports adjoint gradients only"));
    }
    let mut vqe = Vqe::new(hamiltonian.inner.clone(), ansatz.inner.clone())
        .raise()?
        .with_precision(precision)
        .with_method(self::method(method)?)
        .with_options(options(max_iterations))
        .with_gradient(gradient)
        .with_restarts(restarts)
        .with_seed(seed);
    if let Some(theta) = initial_parameters {
        let theta = real_vec(&theta);
        check_finite("initial_parameters", &theta)?;
        vqe = vqe.with_initial_parameters(theta).raise()?;
    }
    if let Some(state) = initial_state {
        vqe = vqe.with_initial_state(state.inner.clone()).raise()?;
    }
    let r = py.detach(|| vqe.run()).raise()?;
    Ok(pydict!(py, {
        "energy" => r.energy,
        "parameters" => r.parameters.into_pyarray(py),
        "iterations" => r.iterations,
        "evaluations" => r.evaluations,
        "converged" => r.converged,
        "history" => r.history.into_pyarray(py),
        "restart_energies" => r.restart_energies.into_pyarray(py),
    }))
}

// ---------------------------------------------------------------------------
// QAOA

fn qaoa_dict<'py>(py: Python<'py>, r: QaoaResult, n: usize) -> PyResult<Bound<'py, PyDict>> {
    Ok(pydict!(py, {
        "p" => r.p,
        "gammas" => r.gammas.into_pyarray(py),
        "betas" => r.betas.into_pyarray(py),
        "expectation" => r.expectation,
        "max_cut" => r.max_cut,
        "approximation_ratio" => r.approximation_ratio,
        "optimal_probability" => r.optimal_probability,
        "most_likely" => r.most_likely,
        "most_likely_bitstring" => bitstring(r.most_likely, n),
        "most_likely_cut" => r.most_likely_cut,
        "iterations" => r.iterations,
        "evaluations" => r.evaluations,
        "converged" => r.converged,
    }))
}

fn qaoa_config(
    restarts: usize,
    seed: u64,
    method: &str,
    max_iterations: usize,
    initial: Option<Vec<f64>>,
) -> PyResult<QaoaConfig> {
    if let Some(x) = &initial {
        check_finite("initial", x)?;
    }
    Ok(QaoaConfig {
        method: self::method(method)?,
        options: options(max_iterations),
        restarts,
        seed,
        initial,
    })
}

/// QAOA for weighted MaxCut at depth `p`, simulated exactly on 2ⁿ
/// amplitudes. Parameters are `[γ₁…γ_p, β₁…β_p]`.
#[pyclass(name = "MaxCutQaoa", module = "metatron_qso", frozen)]
pub struct PyMaxCutQaoa {
    inner: MaxCutQaoa,
}

impl PyMaxCutQaoa {
    fn params(&self, params: RealArg<'_>) -> PyResult<Vec<f64>> {
        let x = real_vec(&params);
        check_finite("params", &x)?;
        if x.len() != self.inner.num_parameters() {
            return Err(invalid(format!(
                "expected {} parameters, got {}",
                self.inner.num_parameters(),
                x.len()
            )));
        }
        Ok(x)
    }
}

#[pymethods]
impl PyMaxCutQaoa {
    #[new]
    #[pyo3(signature = (graph, p, precision = "double"))]
    fn new(py: Python<'_>, graph: &PyGraph, p: usize, precision: &str) -> PyResult<Self> {
        check_qubits(graph.inner.num_nodes())?;
        let precision = self::precision(precision)?;
        let g = &graph.inner;
        let inner = py
            .detach(|| MaxCutQaoa::new(g, p).map(|q| q.with_precision(precision)))
            .raise()?;
        Ok(Self { inner })
    }

    /// Expected cut weight ⟨C⟩.
    fn expectation(&self, py: Python<'_>, params: RealArg<'_>) -> PyResult<f64> {
        let x = self.params(params)?;
        let q = &self.inner;
        py.detach(|| q.expectation(&x)).raise()
    }

    /// ⟨C⟩ and its exact gradient.
    fn expectation_and_gradient<'py>(
        &self,
        py: Python<'py>,
        params: RealArg<'_>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
        let x = self.params(params)?;
        let q = &self.inner;
        let (v, g) = py.detach(|| q.expectation_and_gradient(&x)).raise()?;
        Ok((v, real_array(py, g)))
    }

    /// Statistics of the QAOA state at `params`.
    fn analyze<'py>(&self, py: Python<'py>, params: RealArg<'_>) -> PyResult<Bound<'py, PyDict>> {
        let x = self.params(params)?;
        let q = &self.inner;
        let r = py.detach(|| q.analyze(&x)).raise()?;
        qaoa_dict(py, r, q.n_qubits())
    }

    /// QAOA state at `params`.
    fn state(&self, py: Python<'_>, params: RealArg<'_>) -> PyResult<PyStateVector> {
        let x = self.params(params)?;
        let q = &self.inner;
        let inner = py.detach(|| q.state(&x)).raise()?;
        Ok(PyStateVector { inner })
    }

    /// Maximise ⟨C⟩ from `restarts` seeded starts (`initial` is used for
    /// the first one).
    #[pyo3(signature = (restarts = 8, seed = 0, method = "lbfgs", max_iterations = 1000, initial = None))]
    fn optimize<'py>(
        &self,
        py: Python<'py>,
        restarts: usize,
        seed: u64,
        method: &str,
        max_iterations: usize,
        initial: Option<RealArg<'_>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let initial = initial.map(|x| real_vec(&x));
        let config = qaoa_config(restarts, seed, method, max_iterations, initial)?;
        let q = &self.inner;
        let r = py.detach(|| q.optimize(&config)).raise()?;
        qaoa_dict(py, r, q.n_qubits())
    }

    /// Exact maximum cut weight.
    #[getter]
    fn max_cut(&self) -> f64 {
        self.inner.max_cut()
    }

    #[getter]
    fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }

    #[getter]
    fn p(&self) -> usize {
        self.inner.depth()
    }

    /// `"double"` or `"single"`: the precision of the simulated state.
    #[getter]
    fn precision(&self) -> &'static str {
        self.inner.precision().name()
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    /// All optimal cuts as basis indices (bit q = side of node q).
    fn optimal_cuts(&self) -> Vec<usize> {
        self.inner.optimal_cuts()
    }

    /// Weight of the cut encoded by basis index `b`.
    fn cut_value(&self, b: usize) -> PyResult<f64> {
        if b >> self.inner.n_qubits() != 0 {
            return Err(invalid(format!(
                "cut index {b} out of range for {} nodes",
                self.inner.n_qubits()
            )));
        }
        Ok(self.inner.cut_value(b))
    }

    fn __repr__(&self) -> String {
        format!(
            "MaxCutQaoa(n_qubits={}, p={}, max_cut={}, precision={:?})",
            self.inner.n_qubits(),
            self.inner.depth(),
            self.inner.max_cut(),
            self.inner.precision().name()
        )
    }
}

/// Optimise depths 1..=p_max, seeding each with the INTERP extrapolation of
/// the previous optimum.
#[pyfunction]
#[pyo3(signature = (graph, p_max, restarts = 4, seed = 0, method = "lbfgs", max_iterations = 1000, precision = "double"))]
#[allow(clippy::too_many_arguments)]
fn optimize_qaoa_layerwise<'py>(
    py: Python<'py>,
    graph: &PyGraph,
    p_max: usize,
    restarts: usize,
    seed: u64,
    method: &str,
    max_iterations: usize,
    precision: &str,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let n = graph.inner.num_nodes();
    check_qubits(n)?;
    let config = qaoa_config(restarts, seed, method, max_iterations, None)?;
    let precision = self::precision(precision)?;
    let g = &graph.inner;
    let results = py
        .detach(|| qaoa::optimize_layerwise_at(g, p_max, &config, precision))
        .raise()?;
    results.into_iter().map(|r| qaoa_dict(py, r, n)).collect()
}

// ---------------------------------------------------------------------------
// Grover, QFT, phase estimation

fn grover_marked(n_qubits: usize, marked: Indices) -> PyResult<Vec<usize>> {
    if n_qubits == 0 || n_qubits > 30 {
        return Err(invalid("Grover search supports 1..=30 qubits"));
    }
    let mut m = marked.into_vec();
    m.sort_unstable();
    m.dedup();
    Ok(m)
}

/// Simulate Grover search for `marked` items (`iterations=None` → optimal).
#[pyfunction]
#[pyo3(signature = (n_qubits, marked, iterations = None))]
fn grover_search<'py>(
    py: Python<'py>,
    n_qubits: usize,
    marked: Indices,
    iterations: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let marked = grover_marked(n_qubits, marked)?;
    let r = py
        .detach(|| grover::grover_search(n_qubits, &marked, iterations))
        .raise()?;
    Ok(pydict!(py, {
        "n_qubits" => r.n_qubits,
        "marked" => r.marked,
        "iterations" => r.iterations,
        "success_probability" => r.success_probability,
        "theoretical_success_probability" => r.theoretical_success_probability,
        "gates" => r.gates,
        "multi_qubit_gates" => r.multi_qubit_gates,
    }))
}

/// Grover circuit: Hadamards then `iterations` × (oracle, diffusion).
#[pyfunction]
#[pyo3(signature = (n_qubits, marked, iterations = None))]
fn grover_circuit(
    n_qubits: usize,
    marked: Indices,
    iterations: Option<usize>,
) -> PyResult<PyCircuit> {
    let marked = grover_marked(n_qubits, marked)?;
    let k = iterations.unwrap_or_else(|| grover::optimal_iterations(n_qubits, marked.len()));
    PyCircuit::wrap(grover::grover_circuit(n_qubits, &marked, k).raise()?)
}

/// Quantum Fourier transform on `n` qubits (bit-reversed output without
/// `swaps`).
#[pyfunction(name = "qft")]
#[pyo3(signature = (n, swaps = true))]
fn py_qft(n: usize, swaps: bool) -> PyResult<PyCircuit> {
    check_qubits(n)?;
    PyCircuit::wrap(qft::qft(n, swaps))
}

/// Inverse quantum Fourier transform.
#[pyfunction]
#[pyo3(signature = (n, swaps = true))]
fn inverse_qft(n: usize, swaps: bool) -> PyResult<PyCircuit> {
    check_qubits(n)?;
    PyCircuit::wrap(qft::inverse_qft(n, swaps))
}

/// Phase estimation of a single-qubit `unitary` (2×2) with `precision`
/// counting qubits on its `eigenstate` (length 2).
#[pyfunction]
fn estimate_phase<'py>(
    py: Python<'py>,
    precision: usize,
    unitary: ComplexMatrixArg<'_>,
    eigenstate: ComplexArg<'_>,
) -> PyResult<Bound<'py, PyDict>> {
    let (dim, u) = square_complex("unitary", &unitary)?;
    let v = complex_vec(&eigenstate);
    if dim != 2 || v.len() != 2 {
        return Err(invalid("unitary must be 2x2 and eigenstate of length 2"));
    }
    let u: [metatron_qso::C64; 4] = [u[0], u[1], u[2], u[3]];
    let v = [v[0], v[1]];
    let r = py.detach(|| qft::estimate_phase(precision, u, v)).raise()?;
    Ok(pydict!(py, {
        "phase" => r.phase,
        "probability" => r.probability,
        "distribution" => r.distribution.into_pyarray(py),
    }))
}

// ---------------------------------------------------------------------------
// Variational classifier

/// Variational quantum classifier with data re-uploading;
/// `p(y=1|x) = (1 − ⟨Z₀⟩)/2`.
#[pyclass(name = "VariationalClassifier", module = "metatron_qso")]
pub struct PyClassifier {
    inner: VariationalClassifier,
}

fn samples(xs: &RealMatrixArg<'_>) -> PyResult<Vec<Vec<f64>>> {
    let rows: Vec<Vec<f64>> = xs
        .as_array()
        .rows()
        .into_iter()
        .map(|r| r.to_vec())
        .collect();
    for r in &rows {
        check_finite("xs", r)?;
    }
    Ok(rows)
}

fn labels(ys: &RealArg<'_>) -> PyResult<Vec<bool>> {
    real_vec(ys)
        .into_iter()
        .map(|y| {
            if y == 0.0 || y == 1.0 {
                Ok(y == 1.0)
            } else {
                Err(invalid("labels must be 0/1 or booleans"))
            }
        })
        .collect()
}

impl PyClassifier {
    fn check_shape(&self, xs: &[Vec<f64>], ys: Option<&[bool]>) -> PyResult<()> {
        if let Some(ys) = ys
            && ys.len() != xs.len()
        {
            return Err(invalid(format!(
                "got {} samples but {} labels",
                xs.len(),
                ys.len()
            )));
        }
        if let Some(x) = xs.iter().find(|x| x.len() != self.inner.n_features) {
            return Err(invalid(format!(
                "expected {} features per sample, got {}",
                self.inner.n_features,
                x.len()
            )));
        }
        Ok(())
    }

    /// Probabilities for a batch.
    fn probabilities(&self, py: Python<'_>, xs: &[Vec<f64>]) -> PyResult<Vec<f64>> {
        self.check_shape(xs, None)?;
        let model = &self.inner;
        py.detach(|| {
            xs.iter()
                .map(|x| model.predict_proba(x))
                .collect::<metatron_qso::Result<Vec<_>>>()
        })
        .raise()
    }

    /// Batch (2-D input) or single sample (1-D input).
    fn batch(x: &Bound<'_, PyAny>) -> PyResult<(Vec<Vec<f64>>, bool)> {
        if let Ok(m) = x.extract::<RealMatrixArg<'_>>() {
            return Ok((samples(&m)?, true));
        }
        let v: RealArg<'_> = x.extract()?;
        let v = real_vec(&v);
        check_finite("x", &v)?;
        Ok((vec![v], false))
    }
}

#[pymethods]
impl PyClassifier {
    #[new]
    fn new(n_qubits: usize, n_features: usize, layers: usize) -> PyResult<Self> {
        check_qubits(n_qubits)?;
        let inner = VariationalClassifier::new(n_qubits, n_features, layers).raise()?;
        Ok(Self { inner })
    }

    /// Train on samples `xs` (shape `(m, n_features)`) and 0/1 labels `ys`
    /// with L-BFGS from a seeded start.
    #[pyo3(signature = (xs, ys, max_iterations = 200, seed = 0))]
    fn fit<'py>(
        &mut self,
        py: Python<'py>,
        xs: RealMatrixArg<'_>,
        ys: RealArg<'_>,
        max_iterations: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let xs = samples(&xs)?;
        let ys = labels(&ys)?;
        self.check_shape(&xs, Some(&ys))?;
        let model = &mut self.inner;
        let r = py
            .detach(|| model.fit(&xs, &ys, &options(max_iterations), seed))
            .raise()?;
        Ok(pydict!(py, {
            "loss" => r.loss,
            "accuracy" => r.accuracy,
            "iterations" => r.iterations,
            "evaluations" => r.evaluations,
        }))
    }

    /// `p(y=1|x)`: a float for one sample, an array for a 2-D batch.
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (xs, is_batch) = Self::batch(x)?;
        let p = self.probabilities(py, &xs)?;
        if is_batch {
            Ok(real_array(py, p).into_any())
        } else {
            Ok(p[0].into_pyobject(py)?.into_any())
        }
    }

    /// Predicted class (`p ≥ ½`): a bool for one sample, an array for a
    /// batch.
    fn predict<'py>(&self, py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let (xs, is_batch) = Self::batch(x)?;
        let labels: Vec<bool> = self
            .probabilities(py, &xs)?
            .into_iter()
            .map(|p| p >= 0.5)
            .collect();
        if is_batch {
            Ok(PyArray1::from_vec(py, labels).into_any())
        } else {
            Ok(labels[0].into_pyobject(py)?.to_owned().into_any())
        }
    }

    /// Fraction of correctly classified samples.
    fn accuracy(&self, py: Python<'_>, xs: RealMatrixArg<'_>, ys: RealArg<'_>) -> PyResult<f64> {
        let xs = samples(&xs)?;
        let ys = labels(&ys)?;
        self.check_shape(&xs, Some(&ys))?;
        let model = &self.inner;
        py.detach(|| model.accuracy(&xs, &ys)).raise()
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits
    }

    #[getter]
    fn n_features(&self) -> usize {
        self.inner.n_features
    }

    #[getter]
    fn layers(&self) -> usize {
        self.inner.layers
    }

    #[getter]
    fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }

    /// Trained parameters (empty before `fit`).
    #[getter]
    fn parameters<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        real_array(py, self.inner.parameters.clone())
    }

    #[setter]
    fn set_parameters(&mut self, theta: RealArg<'_>) -> PyResult<()> {
        let theta = real_vec(&theta);
        check_finite("parameters", &theta)?;
        if theta.len() != self.inner.num_parameters() {
            return Err(invalid(format!(
                "expected {} parameters, got {}",
                self.inner.num_parameters(),
                theta.len()
            )));
        }
        self.inner.parameters = theta;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "VariationalClassifier(n_qubits={}, n_features={}, layers={}, trained={})",
            self.inner.n_qubits,
            self.inner.n_features,
            self.inner.layers,
            !self.inner.parameters.is_empty()
        )
    }
}

/// Graph classifier with QAOA-type quantum layers.
///
/// Each graph `G` defines its cut operator `C_G`; `layers` trainable blocks
/// `exp(−iβ Σ X)·exp(−iγ C_G)` act on `|+⟩ⁿ` with the same parameters for
/// every graph, and `p(y = 1 | G) = σ(a·(⟨C_G⟩/W − ½) + b)`. Graphs are
/// simulated exactly, so each needs at most about 20 nodes.
#[pyclass(name = "GraphClassifier", module = "metatron_qso")]
pub struct PyGraphClassifier {
    inner: GraphClassifier,
}

fn graph_list(graphs: &[PyRef<'_, PyGraph>]) -> Vec<Graph> {
    graphs.iter().map(|g| g.inner.clone()).collect()
}

#[pymethods]
impl PyGraphClassifier {
    #[new]
    fn new(layers: usize) -> PyResult<Self> {
        Ok(Self {
            inner: GraphClassifier::new(layers).raise()?,
        })
    }

    /// Train on `graphs` with 0/1 `labels` (L-BFGS, `restarts` seeded
    /// starts, best loss kept).
    #[pyo3(signature = (graphs, labels, max_iterations = 300, restarts = 4, seed = 0))]
    fn fit<'py>(
        &mut self,
        py: Python<'py>,
        graphs: Vec<PyRef<'py, PyGraph>>,
        labels: RealArg<'_>,
        max_iterations: usize,
        restarts: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let graphs = graph_list(&graphs);
        let ys = self::labels(&labels)?;
        let model = &mut self.inner;
        let r = py
            .detach(|| model.fit(&graphs, &ys, &options(max_iterations), restarts, seed))
            .raise()?;
        Ok(pydict!(py, {
            "loss" => r.loss,
            "accuracy" => r.accuracy,
            "iterations" => r.iterations,
            "evaluations" => r.evaluations,
        }))
    }

    /// Normalised expected cut `⟨C_G⟩ / W(G)`, with the trained angles or
    /// with `angles = [γ…, β…]`.
    #[pyo3(signature = (graph, angles = None))]
    fn feature(
        &self,
        py: Python<'_>,
        graph: &PyGraph,
        angles: Option<RealArg<'_>>,
    ) -> PyResult<f64> {
        let model = &self.inner;
        let g = &graph.inner;
        match angles {
            Some(a) => {
                let a = real_vec(&a);
                check_finite("angles", &a)?;
                py.detach(|| model.feature_with(g, &a)).raise()
            }
            None => py.detach(|| model.feature(g)).raise(),
        }
    }

    /// `p(y = 1 | G)`.
    fn predict_proba(&self, py: Python<'_>, graph: &PyGraph) -> PyResult<f64> {
        let (model, g) = (&self.inner, &graph.inner);
        py.detach(|| model.predict_proba(g)).raise()
    }

    /// Predicted class (`p ≥ ½`).
    fn predict(&self, py: Python<'_>, graph: &PyGraph) -> PyResult<bool> {
        let (model, g) = (&self.inner, &graph.inner);
        py.detach(|| model.predict(g)).raise()
    }

    /// Fraction of correctly classified graphs.
    fn accuracy(
        &self,
        py: Python<'_>,
        graphs: Vec<PyRef<'_, PyGraph>>,
        labels: RealArg<'_>,
    ) -> PyResult<f64> {
        let graphs = graph_list(&graphs);
        let ys = self::labels(&labels)?;
        let model = &self.inner;
        py.detach(|| model.accuracy(&graphs, &ys)).raise()
    }

    #[getter]
    fn layers(&self) -> usize {
        self.inner.layers
    }

    #[getter]
    fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }

    /// Trained parameters `[γ…, β…, a, b]` (empty before `fit`).
    #[getter]
    fn parameters<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        real_array(py, self.inner.parameters.clone())
    }

    #[setter]
    fn set_parameters(&mut self, theta: RealArg<'_>) -> PyResult<()> {
        let theta = real_vec(&theta);
        check_finite("parameters", &theta)?;
        if theta.len() != self.inner.num_parameters() {
            return Err(invalid(format!(
                "expected {} parameters, got {}",
                self.inner.num_parameters(),
                theta.len()
            )));
        }
        self.inner.parameters = theta;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "GraphClassifier(layers={}, trained={})",
            self.inner.layers,
            !self.inner.parameters.is_empty()
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(h2_sto3g, m)?)?;
    m.add_function(wrap_pyfunction!(transverse_field_ising, m)?)?;
    m.add_function(wrap_pyfunction!(heisenberg_xxz, m)?)?;
    m.add_function(wrap_pyfunction!(maxcut_cost, m)?)?;
    m.add_function(wrap_pyfunction!(real_amplitudes, m)?)?;
    m.add_function(wrap_pyfunction!(efficient_su2, m)?)?;
    m.add_function(wrap_pyfunction!(hardware_efficient, m)?)?;
    m.add_function(wrap_pyfunction!(qaoa_maxcut_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(expectation_and_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(expectation_batch, m)?)?;
    m.add_function(wrap_pyfunction!(expectation_and_gradient_batch, m)?)?;
    m.add_function(wrap_pyfunction!(run_vqe, m)?)?;
    m.add_class::<PyMaxCutQaoa>()?;
    m.add_class::<PyGraphClassifier>()?;
    m.add_function(wrap_pyfunction!(optimize_qaoa_layerwise, m)?)?;
    m.add_function(wrap_pyfunction!(grover_search, m)?)?;
    m.add_function(wrap_pyfunction!(grover_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(py_qft, m)?)?;
    m.add_function(wrap_pyfunction!(inverse_qft, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_phase, m)?)?;
    m.add_class::<PyClassifier>()?;
    Ok(())
}
