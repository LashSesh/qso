//! QUBO and Ising problems, QAOA with problem-specific mixers, and
//! amplitude estimation.

use metatron_qso::algorithms::amplitude_estimation::{
    AmplitudeEstimate, AmplitudeProblem, canonical, iterative, maximum_likelihood,
};
use metatron_qso::algorithms::combinatorial::{Mixer, Qaoa, QaoaAnalysis, QaoaOptions, Qubo};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::circuit::PyCircuit;
use crate::state::{PyPauliSum, PyStateVector, wrap, wrap_sum};
use crate::{Raise, RealArg, RealMatrixArg, check_finite, invalid, pydict, real_vec};

/// Bit mask from an int or a sequence of 0/1 values (index i = variable i).
fn bits(x: &Bound<'_, PyAny>, n: usize) -> PyResult<u64> {
    if let Ok(v) = x.extract::<u64>() {
        if n < 64 && v >> n != 0 {
            return Err(invalid(format!("{v} does not fit into {n} bits")));
        }
        return Ok(v);
    }
    let list: Vec<u8> = x
        .extract()
        .map_err(|_| invalid("x must be an int bit mask or a sequence of 0/1"))?;
    if list.len() != n || list.iter().any(|&b| b > 1) {
        return Err(invalid(format!("x must hold {n} values 0 or 1")));
    }
    Ok(list
        .iter()
        .enumerate()
        .fold(0u64, |m, (i, &b)| m | (u64::from(b) << i)))
}

/// Quadratic unconstrained binary optimisation `min Σ Q_ij x_i x_j + offset`.
#[pyclass(name = "Qubo", module = "metatron_qso", frozen)]
#[derive(Clone)]
pub struct PyQubo {
    inner: Qubo,
}

#[pymethods]
impl PyQubo {
    /// From a square matrix (`Q_ij` and `Q_ji` add up).
    #[new]
    #[pyo3(signature = (matrix, offset = 0.0))]
    fn new(matrix: RealMatrixArg<'_>, offset: f64) -> PyResult<Self> {
        let view = matrix.as_array();
        let (r, c) = view.dim();
        if r != c {
            return Err(invalid(format!("matrix must be square, got ({r}, {c})")));
        }
        let data: Vec<f64> = view.iter().copied().collect();
        Qubo::new(r, &data, offset)
            .raise()
            .map(|inner| Self { inner })
    }

    /// From `(i, j, value)` entries.
    #[staticmethod]
    #[pyo3(signature = (n, entries, offset = 0.0))]
    fn from_entries(n: usize, entries: Vec<(usize, usize, f64)>, offset: f64) -> PyResult<Self> {
        Qubo::from_entries(n, &entries, offset)
            .raise()
            .map(|inner| Self { inner })
    }

    #[getter]
    fn num_variables(&self) -> usize {
        self.inner.num_variables()
    }

    #[getter]
    fn offset(&self) -> f64 {
        self.inner.offset()
    }

    /// `f(x)` for a bit mask or a 0/1 sequence.
    fn value(&self, x: &Bound<'_, PyAny>) -> PyResult<f64> {
        Ok(self.inner.value(bits(x, self.inner.num_variables())?))
    }

    /// The Ising form as a dict with `h`, `j` (list of `(i, j, J)`) and
    /// `offset`.
    fn to_ising<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let ising = self.inner.to_ising();
        Ok(pydict!(py, {
            "h" => ising.h.into_pyarray(py),
            "j" => ising.j,
            "offset" => ising.offset,
        }))
    }

    /// The diagonal cost Hamiltonian.
    fn to_pauli_sum(&self) -> PyPauliSum {
        wrap_sum(self.inner.to_pauli_sum())
    }

    /// `(minimum, [bit masks])` by enumeration (at most 30 variables).
    fn brute_force(&self, py: Python<'_>) -> PyResult<(f64, Vec<u64>)> {
        let q = &self.inner;
        py.detach(|| q.brute_force()).raise()
    }

    fn __repr__(&self) -> String {
        format!("Qubo(num_variables={})", self.inner.num_variables())
    }
}

fn analysis<'py>(py: Python<'py>, r: QaoaAnalysis) -> PyResult<Bound<'py, PyDict>> {
    Ok(pydict!(py, {
        "gammas" => r.gammas.into_pyarray(py),
        "betas" => r.betas.into_pyarray(py),
        "expectation" => r.expectation,
        "optimum" => r.optimum,
        "worst" => r.worst,
        "approximation_ratio" => r.approximation_ratio,
        "optimal_probability" => r.optimal_probability,
        "feasible_probability" => r.feasible_probability,
        "most_likely" => r.most_likely,
        "most_likely_value" => r.most_likely_value,
        "iterations" => r.iterations,
        "evaluations" => r.evaluations,
        "converged" => r.converged,
    }))
}

/// QAOA for a diagonal cost with mixer "x", "xy_ring", "xy_complete"
/// (from the Dicke state of Hamming weight `weight`) or "warm_start"
/// (from `warm_start` values in [0, 1], clipped to [ε, 1 − ε]).
#[pyclass(name = "Qaoa", module = "metatron_qso", frozen)]
pub struct PyQaoa {
    inner: Qaoa,
}

#[pymethods]
impl PyQaoa {
    #[new]
    #[pyo3(signature = (cost, n_qubits, p, mixer = "x", weight = None, warm_start = None, epsilon = 0.25))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        cost: &PyPauliSum,
        n_qubits: usize,
        p: usize,
        mixer: &str,
        weight: Option<usize>,
        warm_start: Option<RealArg<'_>>,
        epsilon: f64,
    ) -> PyResult<Self> {
        let need_weight = || weight.ok_or_else(|| invalid("XY mixers need `weight`"));
        if weight.is_some() && !mixer.starts_with("xy_") {
            return Err(invalid("`weight` applies only to the XY mixers"));
        }
        if warm_start.is_some() && mixer != "warm_start" {
            return Err(invalid("`warm_start` applies only to mixer='warm_start'"));
        }
        let mixer = match mixer {
            "x" => Mixer::X,
            "xy_ring" => Mixer::XyRing {
                weight: need_weight()?,
            },
            "xy_complete" => Mixer::XyComplete {
                weight: need_weight()?,
            },
            "warm_start" => {
                let c = warm_start
                    .map(|c| real_vec(&c))
                    .ok_or_else(|| invalid("the warm-start mixer needs `warm_start`"))?;
                check_finite("warm_start", &c)?;
                Mixer::WarmStart {
                    solution: c,
                    epsilon,
                }
            }
            other => {
                return Err(invalid(format!(
                    "mixer must be 'x', 'xy_ring', 'xy_complete' or 'warm_start', got '{other}'"
                )));
            }
        };
        if p > 1000 {
            return Err(invalid("p must be at most 1000"));
        }
        let c = &cost.inner;
        py.detach(|| Qaoa::new(c, n_qubits, p, mixer))
            .raise()
            .map(|inner| Self { inner })
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    #[getter]
    fn p(&self) -> usize {
        self.inner.depth()
    }

    /// The parametrised circuit `[γ…, β…]` (apply to `initial_state`).
    fn circuit(&self) -> PyResult<PyCircuit> {
        PyCircuit::wrap(self.inner.circuit().clone())
    }

    fn initial_state(&self) -> PyStateVector {
        wrap(self.inner.initial_state().clone())
    }

    /// `(⟨C⟩, gradient)` at `[γ…, β…]`.
    fn expectation_and_gradient<'py>(
        &self,
        py: Python<'py>,
        params: RealArg<'_>,
    ) -> PyResult<(f64, Bound<'py, numpy::PyArray1<f64>>)> {
        let x = real_vec(&params);
        check_finite("params", &x)?;
        let q = &self.inner;
        let (v, g) = py.detach(|| q.expectation_and_gradient(&x)).raise()?;
        Ok((v, g.into_pyarray(py)))
    }

    fn analyze<'py>(&self, py: Python<'py>, params: RealArg<'_>) -> PyResult<Bound<'py, PyDict>> {
        let x = real_vec(&params);
        check_finite("params", &x)?;
        if x.len() != 2 * self.inner.depth() {
            return Err(invalid(format!(
                "need {} parameters",
                2 * self.inner.depth()
            )));
        }
        let q = &self.inner;
        analysis(py, py.detach(|| q.analyze(&x)).raise()?)
    }

    /// Minimise ⟨C⟩ from seeded random starts.
    #[pyo3(signature = (restarts = 8, seed = 0, initial = None))]
    fn optimize<'py>(
        &self,
        py: Python<'py>,
        restarts: usize,
        seed: u64,
        initial: Option<RealArg<'_>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut opts = QaoaOptions::default()
            .with_restarts(restarts)
            .with_seed(seed);
        if let Some(x) = initial {
            let x = real_vec(&x);
            check_finite("initial", &x)?;
            opts = opts.with_initial(x);
        }
        let q = &self.inner;
        let r = py.detach(|| q.optimize(&opts)).raise()?;
        analysis(py, r)
    }

    fn __repr__(&self) -> String {
        format!(
            "Qaoa(n_qubits={}, p={})",
            self.inner.n_qubits(),
            self.inner.depth()
        )
    }
}

/// Amplitude `a = P(objective = 1)` of `circuit|0⟩`.
#[pyclass(name = "AmplitudeProblem", module = "metatron_qso", frozen)]
pub struct PyAmplitudeProblem {
    inner: AmplitudeProblem,
}

#[pymethods]
impl PyAmplitudeProblem {
    #[new]
    fn new(circuit: &PyCircuit, objective: usize) -> PyResult<Self> {
        AmplitudeProblem::new(circuit.inner.clone(), objective)
            .raise()
            .map(|inner| Self { inner })
    }

    /// One qubit with P(1) = p.
    #[staticmethod]
    fn bernoulli(p: f64) -> PyResult<Self> {
        AmplitudeProblem::bernoulli(p)
            .raise()
            .map(|inner| Self { inner })
    }

    fn exact(&self) -> PyResult<f64> {
        self.inner.exact().raise()
    }

    fn grover_operator(&self) -> PyResult<PyCircuit> {
        PyCircuit::wrap(self.inner.grover_operator())
    }

    /// P(objective = 1) in Q^k A|0⟩ for ascending powers k.
    fn probabilities(&self, py: Python<'_>, powers: Vec<usize>) -> PyResult<Vec<f64>> {
        let p = &self.inner;
        py.detach(|| p.probabilities(&powers)).raise()
    }

    fn __repr__(&self) -> String {
        format!(
            "AmplitudeProblem(n_qubits={}, objective={})",
            self.inner.n_qubits(),
            self.inner.objective()
        )
    }
}

fn estimate_dict<'py>(py: Python<'py>, r: AmplitudeEstimate) -> PyResult<Bound<'py, PyDict>> {
    Ok(pydict!(py, {
        "estimate" => r.estimate,
        "interval" => r.interval,
        "oracle_calls" => r.oracle_calls,
        "shots" => r.shots,
        "schedule" => r.schedule,
    }))
}

/// Canonical (phase-estimation) amplitude estimation with `m` evaluation
/// qubits: exact outcome distribution.
#[pyfunction]
fn canonical_amplitude_estimation<'py>(
    py: Python<'py>,
    problem: &PyAmplitudeProblem,
    m: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let p = &problem.inner;
    let r = py.detach(|| canonical(p, m)).raise()?;
    Ok(pydict!(py, {
        "estimate" => r.estimate,
        "distribution" => r.distribution,
        "oracle_calls" => r.oracle_calls,
    }))
}

/// Iterative amplitude estimation to half-width `epsilon` at confidence
/// `1 − alpha`.
#[pyfunction]
#[pyo3(signature = (problem, epsilon = 0.01, alpha = 0.05, shots = 100, seed = 0))]
fn iterative_amplitude_estimation<'py>(
    py: Python<'py>,
    problem: &PyAmplitudeProblem,
    epsilon: f64,
    alpha: f64,
    shots: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    if shots > 10_000_000 {
        return Err(invalid("shots must be at most 10^7"));
    }
    let p = &problem.inner;
    let r = py
        .detach(|| iterative(p, epsilon, alpha, shots, seed))
        .raise()?;
    estimate_dict(py, r)
}

/// Maximum-likelihood amplitude estimation with powers 0, 1, 2, 4, ….
#[pyfunction]
#[pyo3(signature = (problem, levels = 6, shots = 100, confidence = 0.95, seed = 0))]
fn ml_amplitude_estimation<'py>(
    py: Python<'py>,
    problem: &PyAmplitudeProblem,
    levels: usize,
    shots: usize,
    confidence: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    if shots > 10_000_000 {
        return Err(invalid("shots must be at most 10^7"));
    }
    let p = &problem.inner;
    let r = py
        .detach(|| maximum_likelihood(p, levels, shots, confidence, seed))
        .raise()?;
    estimate_dict(py, r)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyQubo>()?;
    m.add_class::<PyQaoa>()?;
    m.add_class::<PyAmplitudeProblem>()?;
    m.add_function(wrap_pyfunction!(canonical_amplitude_estimation, m)?)?;
    m.add_function(wrap_pyfunction!(iterative_amplitude_estimation, m)?)?;
    m.add_function(wrap_pyfunction!(ml_amplitude_estimation, m)?)?;
    Ok(())
}
