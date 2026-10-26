//! `MatrixProductState`: circuits with little entanglement on many qubits.

use metatron_qso::C64;
use metatron_qso::mps::{Mps, MpsOptions};
use metatron_qso::quantum::PauliSum;
use metatron_qso::rng::Rng;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::circuit::{PyCircuit, theta_or_empty};
use crate::state::{PyPauliSum, PyStateVector};
use crate::{ComplexMatrixArg, Raise, RealArg, invalid};

/// Most shots one call samples.
const MAX_SHOTS: usize = 10_000_000;

/// A pure state of up to 65 536 qubits as a matrix-product state.
///
/// Bonds keep at most `max_bond` Schmidt values, and a split drops the
/// smallest ones whose total weight is at most `cutoff`. Every truncation
/// is measured: `fidelity_lower_bound` bounds the fidelity with the state
/// the gates prepare exactly. Measurements draw from a generator seeded by
/// `seed`.
#[pyclass(name = "MatrixProductState", module = "metatron_qso")]
#[derive(Clone)]
pub struct PyMps {
    inner: Mps,
    rng: Rng,
}

type Slf<'py> = PyRefMut<'py, PyMps>;

/// Observable argument: a Pauli string, a `PauliSum` or a list of
/// `(coefficient, string)` terms (for registers beyond 64 qubits).
enum Observable {
    Text(String),
    Sum(PauliSum),
    Terms(Vec<(f64, String)>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for Observable {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(s) = obj.cast::<PyPauliSum>() {
            return Ok(Observable::Sum(s.get().inner.clone()));
        }
        if let Ok(t) = obj.extract::<String>() {
            return Ok(Observable::Text(t));
        }
        Ok(Observable::Terms(obj.extract::<Vec<(f64, String)>>()?))
    }
}

fn options(max_bond: usize, cutoff: f64) -> MpsOptions {
    MpsOptions { max_bond, cutoff }
}

#[pymethods]
impl PyMps {
    /// `|0…0⟩` on `n_qubits` qubits.
    #[new]
    #[pyo3(signature = (n_qubits, max_bond = 256, cutoff = 1e-14, seed = 0))]
    fn new(n_qubits: usize, max_bond: usize, cutoff: f64, seed: u64) -> PyResult<Self> {
        Ok(Self {
            inner: Mps::zero(n_qubits, options(max_bond, cutoff)).raise()?,
            rng: Rng::seed_from_u64(seed),
        })
    }

    /// Run `circuit` (parameters bound to `theta`) on `|0…0⟩`.
    #[staticmethod]
    #[pyo3(signature = (circuit, theta = None, max_bond = 256, cutoff = 1e-14, seed = 0))]
    fn from_circuit(
        py: Python<'_>,
        circuit: &PyCircuit,
        theta: Option<RealArg<'_>>,
        max_bond: usize,
        cutoff: f64,
        seed: u64,
    ) -> PyResult<Self> {
        let theta = theta_or_empty(theta);
        let c = &circuit.inner;
        Ok(Self {
            inner: py
                .detach(|| Mps::from_circuit(c, &theta, options(max_bond, cutoff)))
                .raise()?,
            rng: Rng::seed_from_u64(seed),
        })
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    #[getter]
    fn max_bond(&self) -> usize {
        self.inner.options().max_bond
    }

    #[getter]
    fn cutoff(&self) -> f64 {
        self.inner.options().cutoff
    }

    fn h(mut slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        slf.inner.h(qubit).raise()?;
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

    fn s(mut slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        slf.inner.s(qubit).raise()?;
        Ok(slf)
    }

    fn t(mut slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        slf.inner.t(qubit).raise()?;
        Ok(slf)
    }

    fn rx(mut slf: Slf<'_>, qubit: usize, angle: f64) -> PyResult<Slf<'_>> {
        slf.inner.rx(qubit, angle).raise()?;
        Ok(slf)
    }

    fn ry(mut slf: Slf<'_>, qubit: usize, angle: f64) -> PyResult<Slf<'_>> {
        slf.inner.ry(qubit, angle).raise()?;
        Ok(slf)
    }

    fn rz(mut slf: Slf<'_>, qubit: usize, angle: f64) -> PyResult<Slf<'_>> {
        slf.inner.rz(qubit, angle).raise()?;
        Ok(slf)
    }

    fn cx(mut slf: Slf<'_>, control: usize, target: usize) -> PyResult<Slf<'_>> {
        slf.inner.cx(control, target).raise()?;
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

    fn rzz(mut slf: Slf<'_>, a: usize, b: usize, angle: f64) -> PyResult<Slf<'_>> {
        slf.inner.rzz(a, b, angle).raise()?;
        Ok(slf)
    }

    /// Apply a unitary matrix (`qubits[0]` most significant) to up to 8
    /// distinct qubits.
    fn apply_matrix<'py>(
        mut slf: Slf<'py>,
        matrix: ComplexMatrixArg<'_>,
        qubits: Vec<usize>,
    ) -> PyResult<Slf<'py>> {
        let m = matrix.as_array();
        if m.nrows() != m.ncols() {
            return Err(invalid("the matrix must be square"));
        }
        let flat: Vec<C64> = m.iter().copied().collect();
        slf.inner.apply_matrix(&flat, &qubits).raise()?;
        Ok(slf)
    }

    /// Apply a circuit (parameters bound to `theta`).
    #[pyo3(signature = (circuit, theta = None))]
    fn apply<'py>(
        mut slf: Slf<'py>,
        circuit: &PyCircuit,
        theta: Option<RealArg<'_>>,
    ) -> PyResult<Slf<'py>> {
        let theta = theta_or_empty(theta);
        let py = slf.py();
        let s = &mut slf.inner;
        py.detach(|| s.apply_circuit(&circuit.inner, &theta))
            .raise()?;
        Ok(slf)
    }

    /// Measure `qubit` in the computational basis: 0 or 1.
    fn measure(&mut self, qubit: usize) -> PyResult<u8> {
        let Self { inner, rng } = self;
        Ok(u8::from(inner.measure(qubit, rng).raise()?))
    }

    /// Reset `qubit` to `|0⟩`.
    fn reset(mut slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        let this = &mut *slf;
        this.inner.reset(qubit, &mut this.rng).raise()?;
        Ok(slf)
    }

    /// `⟨O⟩` of a Pauli string (`"Z0 Z99"`), a `PauliSum` (qubits below
    /// 64) or a list of `(coefficient, string)` terms.
    fn expectation(&mut self, py: Python<'_>, observable: Observable) -> PyResult<f64> {
        let s = &mut self.inner;
        match observable {
            Observable::Text(t) => py.detach(|| s.expectation_str(&t)).raise(),
            Observable::Sum(h) => py.detach(|| s.expectation_sum(&h)).raise(),
            Observable::Terms(terms) => py
                .detach(|| {
                    terms.iter().try_fold(0.0, |acc, (c, t)| {
                        Ok::<_, metatron_qso::QsoError>(acc + c * s.expectation_str(t)?)
                    })
                })
                .raise(),
        }
    }

    /// The amplitude of a basis state given as a bit string, qubit n − 1
    /// first.
    fn amplitude(&self, bitstring: &str) -> PyResult<C64> {
        let n = self.inner.n_qubits();
        if bitstring.len() != n || !bitstring.bytes().all(|b| b == b'0' || b == b'1') {
            return Err(invalid(format!(
                "expected a bit string of {n} zeros and ones"
            )));
        }
        let bits: Vec<bool> = bitstring.bytes().rev().map(|b| b == b'1').collect();
        self.inner.amplitude(&bits).raise()
    }

    /// Measure every qubit `shots` times, leaving the state unchanged:
    /// `{bitstring: count}` with qubit n − 1 first.
    #[pyo3(signature = (shots, seed = 0))]
    fn sample<'py>(
        &mut self,
        py: Python<'py>,
        shots: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        if shots > MAX_SHOTS {
            return Err(invalid("shots must be at most 10⁷"));
        }
        let s = &mut self.inner;
        let counts = py
            .detach(|| s.sample_counts(shots, &mut Rng::seed_from_u64(seed)))
            .raise()?;
        let d = PyDict::new(py);
        for (k, v) in counts {
            d.set_item(k, v)?;
        }
        Ok(d)
    }

    /// Bond dimensions between consecutive qubits.
    fn bond_dimensions(&self) -> Vec<usize> {
        self.inner.bond_dimensions()
    }

    /// The largest bond dimension.
    #[getter]
    fn max_bond_dimension(&self) -> usize {
        self.inner.max_bond_dimension()
    }

    /// Entanglement entropy in bits between qubits `0..=b` and the rest,
    /// for every bond `b`.
    fn bond_entropies(&mut self, py: Python<'_>) -> PyResult<Vec<f64>> {
        let s = &mut self.inner;
        py.detach(|| s.bond_entropies()).raise()
    }

    /// Total weight discarded by truncations.
    #[getter]
    fn discarded_weight(&self) -> f64 {
        self.inner.discarded_weight()
    }

    /// Number of splits that discarded weight beyond rounding.
    #[getter]
    fn truncations(&self) -> usize {
        self.inner.truncations()
    }

    /// A lower bound on the fidelity with the exactly prepared state, or
    /// `None` after a measurement that followed a truncation.
    #[getter]
    fn fidelity_lower_bound(&self) -> Option<f64> {
        self.inner.fidelity_lower_bound()
    }

    /// The state vector, for at most 24 qubits.
    fn to_statevector(&self) -> PyResult<PyStateVector> {
        Ok(PyStateVector {
            inner: self.inner.to_statevector().raise()?,
        })
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "MatrixProductState(n_qubits={}, max_bond_dimension={}, fidelity_lower_bound={:?})",
            self.inner.n_qubits(),
            self.inner.max_bond_dimension(),
            self.inner.fidelity_lower_bound()
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMps>()?;
    m.add("MAX_MPS_QUBITS", metatron_qso::mps::MAX_MPS_QUBITS)?;
    Ok(())
}
