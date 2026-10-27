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

/// Run `f` on the state without holding the GIL: one distant gate on a
/// long chain takes seconds.
fn run<'py>(
    mut slf: Slf<'py>,
    f: impl FnOnce(&mut Mps) -> metatron_qso::Result<()> + Send,
) -> PyResult<Slf<'py>> {
    let py = slf.py();
    let s = &mut slf.inner;
    py.detach(|| f(s)).raise()?;
    Ok(slf)
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

    fn h(slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        run(slf, |s| s.h(qubit).map(drop))
    }

    fn x(slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        run(slf, |s| s.x(qubit).map(drop))
    }

    fn y(slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        run(slf, |s| s.y(qubit).map(drop))
    }

    fn z(slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        run(slf, |s| s.z(qubit).map(drop))
    }

    fn s(slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        run(slf, |s| s.s(qubit).map(drop))
    }

    fn t(slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        run(slf, |s| s.t(qubit).map(drop))
    }

    fn rx(slf: Slf<'_>, qubit: usize, angle: f64) -> PyResult<Slf<'_>> {
        run(slf, |s| s.rx(qubit, angle).map(drop))
    }

    fn ry(slf: Slf<'_>, qubit: usize, angle: f64) -> PyResult<Slf<'_>> {
        run(slf, |s| s.ry(qubit, angle).map(drop))
    }

    fn rz(slf: Slf<'_>, qubit: usize, angle: f64) -> PyResult<Slf<'_>> {
        run(slf, |s| s.rz(qubit, angle).map(drop))
    }

    fn cx(slf: Slf<'_>, control: usize, target: usize) -> PyResult<Slf<'_>> {
        run(slf, |s| s.cx(control, target).map(drop))
    }

    fn cz(slf: Slf<'_>, a: usize, b: usize) -> PyResult<Slf<'_>> {
        run(slf, |s| s.cz(a, b).map(drop))
    }

    fn swap(slf: Slf<'_>, a: usize, b: usize) -> PyResult<Slf<'_>> {
        run(slf, |s| s.swap(a, b).map(drop))
    }

    fn rzz(slf: Slf<'_>, a: usize, b: usize, angle: f64) -> PyResult<Slf<'_>> {
        run(slf, |s| s.rzz(a, b, angle).map(drop))
    }

    /// Apply a unitary matrix (`qubits[0]` most significant) to up to 8
    /// distinct qubits.
    fn apply_matrix<'py>(
        slf: Slf<'py>,
        matrix: ComplexMatrixArg<'_>,
        qubits: Vec<usize>,
    ) -> PyResult<Slf<'py>> {
        let m = matrix.as_array();
        if m.nrows() != m.ncols() {
            return Err(invalid("the matrix must be square"));
        }
        let flat: Vec<C64> = m.iter().copied().collect();
        run(slf, |s| s.apply_matrix(&flat, &qubits))
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
    fn measure(&mut self, py: Python<'_>, qubit: usize) -> PyResult<u8> {
        let Self { inner, rng } = self;
        Ok(u8::from(py.detach(|| inner.measure(qubit, rng)).raise()?))
    }

    /// Reset `qubit` to `|0⟩`.
    fn reset(mut slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        let py = slf.py();
        let PyMps { inner, rng } = &mut *slf;
        py.detach(|| inner.reset(qubit, rng)).raise()?;
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
        let bound = self
            .inner
            .fidelity_lower_bound()
            .map_or_else(|| "None".to_string(), |b| format!("{b:?}"));
        format!(
            "MatrixProductState(n_qubits={}, max_bond_dimension={}, fidelity_lower_bound={bound})",
            self.inner.n_qubits(),
            self.inner.max_bond_dimension(),
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMps>()?;
    m.add("MAX_MPS_QUBITS", metatron_qso::mps::MAX_MPS_QUBITS)?;
    Ok(())
}
