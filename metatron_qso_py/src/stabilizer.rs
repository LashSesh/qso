//! `StabilizerState`, local complementation and Clifford checks.

use metatron_qso::quantum::PauliSum;
use metatron_qso::rng::Rng;
use metatron_qso::stabilizer::program::is_clifford_circuit;
use metatron_qso::stabilizer::{StabilizerState, local_complementation, parse_pauli};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::circuit::{PyCircuit, theta_or_empty};
use crate::graph::PyGraph;
use crate::state::{PyPauliSum, PyStateVector};
use crate::{Raise, RealArg, invalid};

/// Most shots one call samples.
const MAX_SHOTS: usize = 10_000_000;

/// A stabilizer state of up to 16384 qubits (Aaronson–Gottesman tableau).
///
/// Clifford gates, Pauli measurements and expectation values; graph
/// states of any unweighted graph. Measurements draw from a generator
/// seeded by `seed`.
#[pyclass(name = "StabilizerState", module = "metatron_qso")]
#[derive(Clone)]
pub struct PyStabilizerState {
    inner: StabilizerState,
    rng: Rng,
}

type Slf<'py> = PyRefMut<'py, PyStabilizerState>;

/// Observable argument: a Pauli string `"X0 Z1"` or a `PauliSum`.
enum Observable {
    Text(String),
    Sum(PauliSum),
}

impl<'a, 'py> FromPyObject<'a, 'py> for Observable {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(s) = obj.cast::<PyPauliSum>() {
            return Ok(Observable::Sum(s.get().inner.clone()));
        }
        Ok(Observable::Text(obj.extract::<String>()?))
    }
}

#[pymethods]
impl PyStabilizerState {
    /// `|0…0⟩` on `n_qubits` qubits.
    #[new]
    #[pyo3(signature = (n_qubits, seed = 0))]
    fn new(n_qubits: usize, seed: u64) -> PyResult<Self> {
        Ok(Self {
            inner: StabilizerState::zero(n_qubits).raise()?,
            rng: Rng::seed_from_u64(seed),
        })
    }

    /// Run a Clifford circuit (parameters bound to `theta`) on `|0…0⟩`.
    #[staticmethod]
    #[pyo3(signature = (circuit, theta = None, seed = 0))]
    fn from_circuit(circuit: &PyCircuit, theta: Option<RealArg<'_>>, seed: u64) -> PyResult<Self> {
        Ok(Self {
            inner: StabilizerState::from_circuit(&circuit.inner, &theta_or_empty(theta)).raise()?,
            rng: Rng::seed_from_u64(seed),
        })
    }

    /// The graph state of an unweighted graph, stabilized by
    /// `X_v Π_{w∈N(v)} Z_w`.
    #[staticmethod]
    #[pyo3(signature = (graph, seed = 0))]
    fn graph_state(graph: &PyGraph, seed: u64) -> PyResult<Self> {
        Ok(Self {
            inner: StabilizerState::graph_state(&graph.inner).raise()?,
            rng: Rng::seed_from_u64(seed),
        })
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    fn h(mut slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        slf.inner.h(qubit).raise()?;
        Ok(slf)
    }

    fn s(mut slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        slf.inner.s(qubit).raise()?;
        Ok(slf)
    }

    fn sdg(mut slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        slf.inner.sdg(qubit).raise()?;
        Ok(slf)
    }

    fn x(mut slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        slf.inner.x(qubit).raise()?;
        Ok(slf)
    }

    fn y(mut slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        slf.inner.y(qubit).raise()?;
        Ok(slf)
    }

    fn z(mut slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        slf.inner.z(qubit).raise()?;
        Ok(slf)
    }

    fn sx(mut slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        slf.inner.sx(qubit).raise()?;
        Ok(slf)
    }

    fn sxdg(mut slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        slf.inner.sxdg(qubit).raise()?;
        Ok(slf)
    }

    fn cx(mut slf: Slf<'_>, control: usize, target: usize) -> PyResult<Slf<'_>> {
        slf.inner.cx(control, target).raise()?;
        Ok(slf)
    }

    fn cy(mut slf: Slf<'_>, control: usize, target: usize) -> PyResult<Slf<'_>> {
        slf.inner.cy(control, target).raise()?;
        Ok(slf)
    }

    fn cz(mut slf: Slf<'_>, a: usize, b: usize) -> PyResult<Slf<'_>> {
        slf.inner.cz(a, b).raise()?;
        Ok(slf)
    }

    fn swap(mut slf: Slf<'_>, a: usize, b: usize) -> PyResult<Slf<'_>> {
        slf.inner.swap(a, b).raise()?;
        Ok(slf)
    }

    /// Apply a Clifford circuit (parameters bound to `theta`).
    #[pyo3(signature = (circuit, theta = None))]
    fn apply<'py>(
        mut slf: Slf<'py>,
        circuit: &PyCircuit,
        theta: Option<RealArg<'_>>,
    ) -> PyResult<Slf<'py>> {
        slf.inner
            .apply_circuit(&circuit.inner, &theta_or_empty(theta))
            .raise()?;
        Ok(slf)
    }

    /// Measure qubit `qubit` in the computational basis: 0 or 1.
    fn measure(&mut self, py: Python<'_>, qubit: usize) -> PyResult<u8> {
        let Self { inner, rng } = self;
        Ok(u8::from(py.detach(|| inner.measure(qubit, rng)).raise()?))
    }

    /// Measure a Pauli operator such as `"X0 X1"`: 0 for the eigenvalue
    /// +1, 1 for −1.
    fn measure_pauli(&mut self, py: Python<'_>, pauli: &str) -> PyResult<u8> {
        let (sign, ops) = parse_pauli(pauli).raise()?;
        let Self { inner, rng } = self;
        let v = py.detach(|| inner.measure_pauli(&ops, rng)).raise()?;
        Ok(u8::from(v != (sign < 0.0)))
    }

    /// Reset `qubit` to `|0⟩`.
    fn reset(mut slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        let this = &mut *slf;
        this.inner.reset(qubit, &mut this.rng).raise()?;
        Ok(slf)
    }

    /// ⟨P⟩ of a Pauli string (+1, −1 or 0) or of a `PauliSum`.
    fn expectation(&self, py: Python<'_>, observable: Observable) -> PyResult<f64> {
        let s = &self.inner;
        match observable {
            Observable::Text(t) => py.detach(|| s.expectation_str(&t)).raise(),
            Observable::Sum(h) => py.detach(|| s.expectation_sum(&h)).raise(),
        }
    }

    /// Measure every qubit `shots` times (at most 10⁷), leaving the state
    /// unchanged: `{bitstring: count}` with qubit n−1 first.
    #[pyo3(signature = (shots, seed = 0))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        shots: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        if shots > MAX_SHOTS {
            return Err(invalid("shots must be at most 10⁷"));
        }
        let s = &self.inner;
        let counts = py.detach(|| s.sample_counts(shots, &mut Rng::seed_from_u64(seed)));
        let d = PyDict::new(py);
        for (k, v) in counts {
            d.set_item(k, v)?;
        }
        Ok(d)
    }

    /// Stabilizer generators as signed Pauli strings, `"+X0 Z1"`.
    fn stabilizers(&self) -> Vec<String> {
        self.inner.stabilizers()
    }

    /// Entanglement entropy in bits between `qubits` and the rest.
    fn entanglement_entropy(&self, py: Python<'_>, qubits: Vec<usize>) -> PyResult<usize> {
        let s = &self.inner;
        py.detach(|| s.entanglement_entropy(&qubits)).raise()
    }

    /// Whether `other` is the same state up to a global phase.
    fn same_state(&self, py: Python<'_>, other: &PyStabilizerState) -> bool {
        let (a, b) = (&self.inner, &other.inner);
        py.detach(|| a.same_state(b))
    }

    /// Apply the local Clifford that maps the graph state `|G⟩` to
    /// `|G * v⟩` (local complementation at `v`).
    fn local_complement<'py>(mut slf: Slf<'py>, graph: &PyGraph, v: usize) -> PyResult<Slf<'py>> {
        slf.inner.local_complement(&graph.inner, v).raise()?;
        Ok(slf)
    }

    /// The state vector (up to a global phase), for at most 20 qubits.
    fn to_statevector(&self) -> PyResult<PyStateVector> {
        Ok(PyStateVector {
            inner: self.inner.to_statevector().raise()?,
        })
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!("StabilizerState(n_qubits={})", self.inner.n_qubits())
    }
}

/// The local complement `G * v` of an unweighted graph: edges between
/// neighbours of `v` are toggled.
#[pyfunction(name = "local_complementation")]
fn py_local_complementation(graph: &PyGraph, v: usize) -> PyResult<PyGraph> {
    Ok(PyGraph {
        inner: local_complementation(&graph.inner, v).raise()?,
    })
}

/// Whether every gate of `circuit` (with `theta`) is a Clifford gate the
/// stabilizer simulator supports.
#[pyfunction(name = "is_clifford")]
#[pyo3(signature = (circuit, theta = None))]
fn py_is_clifford(circuit: &PyCircuit, theta: Option<RealArg<'_>>) -> bool {
    is_clifford_circuit(&circuit.inner, &theta_or_empty(theta))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStabilizerState>()?;
    m.add_function(wrap_pyfunction!(py_local_complementation, m)?)?;
    m.add_function(wrap_pyfunction!(py_is_clifford, m)?)?;
    m.add(
        "MAX_STABILIZER_QUBITS",
        metatron_qso::stabilizer::MAX_STABILIZER_QUBITS,
    )?;
    Ok(())
}
