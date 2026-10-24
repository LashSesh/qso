//! `StateVector`, `PauliSum` and `simulate`.

use metatron_qso::Rng;
use metatron_qso::linalg::KrylovOptions;
use metatron_qso::quantum::{MAX_QUBITS, PauliSum, StateVector};
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::circuit::{PyCircuit, theta_or_empty};
use crate::{
    ComplexArg, Raise, RealArg, check_finite, check_qubits, cmatrix, complex_array, complex_vec,
    counts_dict, invalid, real_array,
};

/// Pure state of `n` qubits (2ⁿ complex amplitudes, qubit 0 least
/// significant).
#[pyclass(name = "StateVector", module = "metatron_qso")]
#[derive(Clone)]
pub struct PyStateVector {
    pub inner: StateVector,
}

fn wrap(inner: StateVector) -> PyStateVector {
    PyStateVector { inner }
}

/// Run `circuit(theta)` on |0…0⟩ without the GIL.
fn run_circuit(
    py: Python<'_>,
    circuit: &PyCircuit,
    theta: Option<RealArg<'_>>,
) -> PyResult<StateVector> {
    let theta = theta_or_empty(theta);
    check_finite("theta", &theta)?;
    let c = &circuit.inner;
    py.detach(|| StateVector::from_circuit(c, &theta)).raise()
}

#[pymethods]
impl PyStateVector {
    /// |0…0⟩.
    #[staticmethod]
    fn zero(n_qubits: usize) -> PyResult<Self> {
        check_qubits(n_qubits)?;
        Ok(wrap(StateVector::zero(n_qubits)))
    }

    /// Uniform superposition |+⟩^⊗n.
    #[staticmethod]
    fn plus(n_qubits: usize) -> PyResult<Self> {
        check_qubits(n_qubits)?;
        Ok(wrap(StateVector::plus(n_qubits)))
    }

    /// Computational basis state |index⟩.
    #[staticmethod]
    fn basis(n_qubits: usize, index: usize) -> PyResult<Self> {
        check_qubits(n_qubits)?;
        StateVector::basis(n_qubits, index).raise().map(wrap)
    }

    /// State from amplitudes (length a power of two; normalised).
    #[staticmethod]
    fn from_amplitudes(amplitudes: ComplexArg<'_>) -> PyResult<Self> {
        let amps = complex_vec(&amplitudes);
        if amps.iter().any(|a| !(a.re.is_finite() && a.im.is_finite())) {
            return Err(invalid("amplitudes must be finite"));
        }
        StateVector::from_amplitudes(amps).raise().map(wrap)
    }

    /// Haar-random state.
    #[staticmethod]
    #[pyo3(signature = (n_qubits, seed = 0))]
    fn random(n_qubits: usize, seed: u64) -> PyResult<Self> {
        check_qubits(n_qubits)?;
        Ok(wrap(StateVector::random(
            n_qubits,
            &mut Rng::seed_from_u64(seed),
        )))
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    /// Amplitudes as a `complex128` array (a copy).
    #[getter]
    fn amplitudes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<metatron_qso::C64>> {
        complex_array(py, self.inner.amplitudes().to_vec())
    }

    /// Probabilities of all basis states.
    fn probabilities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        real_array(py, self.inner.probabilities())
    }

    /// Measure all qubits `shots` times: `{bitstring: count}` with qubit
    /// n−1 first.
    #[pyo3(signature = (shots, seed = 0))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        shots: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let s = &self.inner;
        let counts = py.detach(|| s.sample(shots, &mut Rng::seed_from_u64(seed)));
        counts_dict(py, &counts, s.n_qubits())
    }

    /// ⟨ψ|H|ψ⟩.
    fn expectation(&self, py: Python<'_>, observable: &PyPauliSum) -> PyResult<f64> {
        let (s, h) = (&self.inner, &observable.inner);
        py.detach(|| h.expectation(s)).raise()
    }

    /// ⟨self|other⟩.
    fn inner(&self, other: &PyStateVector) -> PyResult<metatron_qso::C64> {
        self.inner.inner(&other.inner).raise()
    }

    /// |⟨self|other⟩|².
    fn fidelity(&self, other: &PyStateVector) -> PyResult<f64> {
        self.inner.fidelity(&other.inner).raise()
    }

    /// Von Neumann entropy (bits) of `qubits` against the rest.
    fn entanglement_entropy(&self, qubits: Vec<usize>) -> PyResult<f64> {
        self.inner.entanglement_entropy(&qubits).raise()
    }

    /// Reduced density matrix of `qubits` (`qubits[j]` is bit j of the
    /// row/column index).
    fn reduced_density_matrix<'py>(
        &self,
        py: Python<'py>,
        qubits: Vec<usize>,
    ) -> PyResult<Bound<'py, PyArray2<metatron_qso::C64>>> {
        Ok(cmatrix(
            py,
            &self.inner.reduced_density_matrix(&qubits).raise()?,
        ))
    }

    /// Apply `circuit(theta)` in place; returns the state.
    #[pyo3(signature = (circuit, theta = None))]
    fn apply<'py>(
        mut slf: PyRefMut<'py, Self>,
        circuit: &PyCircuit,
        theta: Option<RealArg<'_>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let theta = theta_or_empty(theta);
        check_finite("theta", &theta)?;
        let py = slf.py();
        let s = &mut slf.inner;
        let c = &circuit.inner;
        py.detach(|| s.apply_circuit(c, &theta)).raise()?;
        Ok(slf)
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!("StateVector(n_qubits={})", self.inner.n_qubits())
    }
}

/// Final state of `circuit(theta)` applied to |0…0⟩.
#[pyfunction]
#[pyo3(signature = (circuit, theta = None))]
fn simulate(
    py: Python<'_>,
    circuit: &PyCircuit,
    theta: Option<RealArg<'_>>,
) -> PyResult<PyStateVector> {
    run_circuit(py, circuit, theta).map(wrap)
}

/// Real linear combination of Pauli strings (a Hermitian observable).
///
/// Parsed from text such as `"0.5*Z0 Z1 - 1.2 X0 + 0.3"`.
#[pyclass(name = "PauliSum", module = "metatron_qso", frozen)]
#[derive(Clone)]
pub struct PyPauliSum {
    pub inner: PauliSum,
}

pub(crate) fn wrap_sum(inner: PauliSum) -> PyPauliSum {
    PyPauliSum { inner }
}

/// Right-hand operand of `+`/`-`: another sum or a constant.
pub enum Operand {
    Sum(PauliSum),
    Scalar(f64),
}

impl<'a, 'py> FromPyObject<'a, 'py> for Operand {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(s) = obj.cast::<PyPauliSum>() {
            return Ok(Operand::Sum(s.get().inner.clone()));
        }
        Ok(Operand::Scalar(obj.extract::<f64>()?))
    }
}

impl Operand {
    fn into_sum(self) -> PyResult<PauliSum> {
        match self {
            Operand::Sum(s) => Ok(s),
            Operand::Scalar(c) => constant(c),
        }
    }
}

fn constant(c: f64) -> PyResult<PauliSum> {
    if !c.is_finite() {
        return Err(invalid("coefficients must be finite"));
    }
    Ok(PauliSum::from_terms([(
        c,
        metatron_qso::quantum::PauliString::identity(),
    )]))
}

fn register_size(sum: &PauliSum, n_qubits: Option<usize>) -> PyResult<usize> {
    let n = n_qubits.unwrap_or_else(|| sum.min_qubits().max(1));
    if n > MAX_QUBITS {
        return Err(invalid(format!(
            "at most {MAX_QUBITS} qubits are supported, got {n}"
        )));
    }
    Ok(n)
}

#[pymethods]
impl PyPauliSum {
    /// Parse e.g. `"0.5*Z0 Z1 - 1.2 X0 + 0.3*Y2*Y3 - 1.05"` (`""` is zero).
    #[new]
    #[pyo3(signature = (text = ""))]
    fn new(text: &str) -> PyResult<Self> {
        text.parse::<PauliSum>().raise().map(wrap_sum)
    }

    /// Sum of `(coefficient, "Pauli string")` terms.
    #[staticmethod]
    fn from_terms(terms: Vec<(f64, String)>) -> PyResult<Self> {
        let mut total = PauliSum::new();
        for (c, s) in terms {
            if !c.is_finite() {
                return Err(invalid("coefficients must be finite"));
            }
            let string: PauliSum = s.parse().raise()?;
            total = total + string * c;
        }
        Ok(wrap_sum(total))
    }

    #[getter]
    fn num_terms(&self) -> usize {
        self.inner.num_terms()
    }

    /// Smallest register containing every term.
    #[getter]
    fn min_qubits(&self) -> usize {
        self.inner.min_qubits()
    }

    /// Coefficient of the identity term.
    #[getter]
    fn constant(&self) -> f64 {
        self.inner.constant()
    }

    /// Terms as `(coefficient, string)` pairs (`"I"` for the identity).
    fn terms(&self) -> Vec<(f64, String)> {
        self.inner
            .terms()
            .iter()
            .map(|(c, s)| (*c, s.to_string()))
            .collect()
    }

    /// ⟨ψ|H|ψ⟩.
    fn expectation(&self, py: Python<'_>, state: &PyStateVector) -> PyResult<f64> {
        let (s, h) = (&state.inner, &self.inner);
        py.detach(|| h.expectation(s)).raise()
    }

    /// Exact ground-state energy on `n_qubits` (default: `min_qubits`).
    #[pyo3(signature = (n_qubits = None))]
    fn ground_state_energy(&self, py: Python<'_>, n_qubits: Option<usize>) -> PyResult<f64> {
        let n = register_size(&self.inner, n_qubits)?;
        let h = &self.inner;
        py.detach(|| h.ground_state_energy(n)).raise()
    }

    /// Ground-state energy and state (Lanczos) on `n_qubits`.
    #[pyo3(signature = (n_qubits = None))]
    fn ground_state(
        &self,
        py: Python<'_>,
        n_qubits: Option<usize>,
    ) -> PyResult<(f64, PyStateVector)> {
        let n = register_size(&self.inner, n_qubits)?;
        let h = &self.inner;
        let (e, s) = py
            .detach(|| h.ground_state(n, &KrylovOptions::default()))
            .raise()?;
        Ok((e, wrap(s)))
    }

    /// Dense `2ⁿ × 2ⁿ` matrix (n ≤ 14; default `min_qubits`).
    #[pyo3(signature = (n_qubits = None))]
    fn to_matrix<'py>(
        &self,
        py: Python<'py>,
        n_qubits: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<metatron_qso::C64>>> {
        let n = register_size(&self.inner, n_qubits)?;
        Ok(cmatrix(py, &self.inner.to_dense(n).raise()?))
    }

    fn __add__(&self, other: Operand) -> PyResult<Self> {
        Ok(wrap_sum(self.inner.clone() + other.into_sum()?))
    }

    fn __radd__(&self, other: Operand) -> PyResult<Self> {
        self.__add__(other)
    }

    fn __sub__(&self, other: Operand) -> PyResult<Self> {
        Ok(wrap_sum(self.inner.clone() - other.into_sum()?))
    }

    fn __rsub__(&self, other: Operand) -> PyResult<Self> {
        Ok(wrap_sum(other.into_sum()? - self.inner.clone()))
    }

    fn __mul__(&self, other: f64) -> PyResult<Self> {
        if !other.is_finite() {
            return Err(invalid("coefficients must be finite"));
        }
        Ok(wrap_sum(self.inner.clone() * other))
    }

    fn __rmul__(&self, other: f64) -> PyResult<Self> {
        self.__mul__(other)
    }

    fn __truediv__(&self, other: f64) -> PyResult<Self> {
        if other == 0.0 {
            return Err(invalid("division of a PauliSum by zero"));
        }
        self.__mul__(1.0 / other)
    }

    fn __neg__(&self) -> Self {
        wrap_sum(-self.inner.clone())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PyPauliSum>()
            .is_ok_and(|o| o.get().inner == self.inner)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("PauliSum({:?})", self.inner.to_string())
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStateVector>()?;
    m.add_class::<PyPauliSum>()?;
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    Ok(())
}
