//! Channels, noise models, density matrices and shot-based estimation.

use metatron_qso::Rng;
use metatron_qso::backend::{
    DensityMatrixBackend, StatevectorBackend, estimate_expectation as estimate,
};
use metatron_qso::noise::{Channel, DensityMatrix, NoiseModel};
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::circuit::{PyCircuit, theta_or_empty};
use crate::state::{PyPauliSum, PyStateVector};
use crate::{
    ComplexMatrixArg, Raise, RealArg, check_finite, cmatrix, counts_dict, invalid, real_array,
    square_array, square_complex,
};

/// Completely positive trace-preserving map on one or two qubits, given by
/// Kraus operators.
#[pyclass(name = "Channel", module = "metatron_qso", frozen)]
#[derive(Clone)]
pub struct PyChannel {
    inner: Channel,
}

fn channel(c: metatron_qso::Result<Channel>) -> PyResult<PyChannel> {
    c.raise().map(|inner| PyChannel { inner })
}

#[pymethods]
impl PyChannel {
    /// `ρ ↦ (1−p)ρ + p/3 (XρX + YρY + ZρZ)`.
    #[staticmethod]
    fn depolarizing(p: f64) -> PyResult<Self> {
        channel(Channel::depolarizing(p))
    }

    /// Two-qubit depolarising channel `ρ ↦ (1−p)ρ + p/15 Σ_{P≠II} PρP`.
    #[staticmethod]
    fn depolarizing_2q(p: f64) -> PyResult<Self> {
        channel(Channel::depolarizing_2q(p))
    }

    /// Amplitude damping with decay probability `gamma`.
    #[staticmethod]
    fn amplitude_damping(gamma: f64) -> PyResult<Self> {
        channel(Channel::amplitude_damping(gamma))
    }

    /// Phase damping: coherences shrink by `√(1−lam)`.
    #[staticmethod]
    fn phase_damping(lam: f64) -> PyResult<Self> {
        channel(Channel::phase_damping(lam))
    }

    /// Bit flip with probability `p`.
    #[staticmethod]
    fn bit_flip(p: f64) -> PyResult<Self> {
        channel(Channel::bit_flip(p))
    }

    /// Phase flip with probability `p`.
    #[staticmethod]
    fn phase_flip(p: f64) -> PyResult<Self> {
        channel(Channel::phase_flip(p))
    }

    /// Relaxation towards |0⟩ for duration `time` (`t2 ≤ 2·t1`).
    #[staticmethod]
    fn thermal_relaxation(t1: f64, t2: f64, time: f64) -> PyResult<Self> {
        channel(Channel::thermal_relaxation(t1, t2, time))
    }

    /// Channel from explicit 2×2 or 4×4 Kraus operators (checked for
    /// `Σ K†K = I`).
    #[staticmethod]
    #[pyo3(signature = (kraus, name = "custom"))]
    fn from_kraus(kraus: Vec<ComplexMatrixArg<'_>>, name: &str) -> PyResult<Self> {
        let ops = kraus
            .iter()
            .map(|k| square_complex("kraus", k).map(|(_, data)| data))
            .collect::<PyResult<Vec<_>>>()?;
        channel(Channel::from_kraus(name, ops))
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    /// Kraus operators as complex matrices.
    fn kraus_operators<'py>(
        &self,
        py: Python<'py>,
    ) -> Vec<Bound<'py, PyArray2<metatron_qso::C64>>> {
        let d = 1usize << self.inner.n_qubits();
        self.inner
            .kraus_operators()
            .iter()
            .map(|k| square_array(py, d, k))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("Channel({})", self.inner.name())
    }
}

/// Noise inserted after every gate: `single_qubit` channels after one-qubit
/// gates, `multi_qubit` channels after multi-qubit gates (one-qubit channels
/// on each qubit, two-qubit channels on consecutive pairs), plus classical
/// readout flips `(P(1|0), P(0|1))`.
#[pyclass(name = "NoiseModel", module = "metatron_qso", frozen)]
#[derive(Clone)]
pub struct PyNoiseModel {
    inner: NoiseModel,
}

#[pymethods]
impl PyNoiseModel {
    #[new]
    #[pyo3(signature = (single_qubit = Vec::new(), multi_qubit = Vec::new(), readout = None))]
    fn new(
        single_qubit: Vec<PyRef<'_, PyChannel>>,
        multi_qubit: Vec<PyRef<'_, PyChannel>>,
        readout: Option<(f64, f64)>,
    ) -> PyResult<Self> {
        if let Some((p01, p10)) = readout
            && !((0.0..=1.0).contains(&p01) && (0.0..=1.0).contains(&p10))
        {
            return Err(invalid("readout error probabilities must lie in [0, 1]"));
        }
        Ok(Self {
            inner: NoiseModel {
                single_qubit: single_qubit.iter().map(|c| c.inner.clone()).collect(),
                multi_qubit: multi_qubit.iter().map(|c| c.inner.clone()).collect(),
                readout,
            },
        })
    }

    /// Depolarising noise: `p1` after one-qubit gates, two-qubit
    /// depolarising `p2` after two-qubit gates.
    #[staticmethod]
    fn depolarizing(p1: f64, p2: f64) -> PyResult<Self> {
        NoiseModel::depolarizing(p1, p2)
            .raise()
            .map(|inner| Self { inner })
    }

    #[getter]
    fn single_qubit(&self) -> Vec<PyChannel> {
        self.inner
            .single_qubit
            .iter()
            .map(|c| PyChannel { inner: c.clone() })
            .collect()
    }

    #[getter]
    fn multi_qubit(&self) -> Vec<PyChannel> {
        self.inner
            .multi_qubit
            .iter()
            .map(|c| PyChannel { inner: c.clone() })
            .collect()
    }

    #[getter]
    fn readout(&self) -> Option<(f64, f64)> {
        self.inner.readout
    }

    fn __repr__(&self) -> String {
        let names = |cs: &[Channel]| {
            cs.iter()
                .map(|c| c.name().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!(
            "NoiseModel(single_qubit=[{}], multi_qubit=[{}], readout={:?})",
            names(&self.inner.single_qubit),
            names(&self.inner.multi_qubit),
            self.inner.readout
        )
    }
}

/// Density matrix of up to 14 qubits.
#[pyclass(name = "DensityMatrix", module = "metatron_qso")]
#[derive(Clone)]
pub struct PyDensityMatrix {
    inner: DensityMatrix,
}

fn dm(inner: DensityMatrix) -> PyDensityMatrix {
    PyDensityMatrix { inner }
}

#[pymethods]
impl PyDensityMatrix {
    /// |0…0⟩⟨0…0|.
    #[staticmethod]
    fn zero(n_qubits: usize) -> PyResult<Self> {
        DensityMatrix::zero(n_qubits).raise().map(dm)
    }

    /// Maximally mixed state I/2ⁿ.
    #[staticmethod]
    fn maximally_mixed(n_qubits: usize) -> PyResult<Self> {
        DensityMatrix::maximally_mixed(n_qubits).raise().map(dm)
    }

    /// |ψ⟩⟨ψ|.
    #[staticmethod]
    fn from_pure(state: &PyStateVector) -> PyResult<Self> {
        DensityMatrix::from_pure(&state.inner).raise().map(dm)
    }

    /// Run `circuit(theta)` on |0…0⟩⟨0…0|, inserting `noise` after every
    /// gate if given.
    #[staticmethod]
    #[pyo3(signature = (circuit, theta = None, noise = None))]
    fn simulate(
        py: Python<'_>,
        circuit: &PyCircuit,
        theta: Option<RealArg<'_>>,
        noise: Option<&PyNoiseModel>,
    ) -> PyResult<Self> {
        let theta = theta_or_empty(theta);
        check_finite("theta", &theta)?;
        let c = &circuit.inner;
        let noise = noise.map(|n| &n.inner);
        py.detach(|| {
            let mut rho = DensityMatrix::zero(c.n_qubits())?;
            rho.apply_circuit(c, &theta, noise)?;
            Ok(rho)
        })
        .raise()
        .map(dm)
    }

    /// Apply `circuit(theta)` (with optional `noise`) in place.
    #[pyo3(signature = (circuit, theta = None, noise = None))]
    fn apply<'py>(
        mut slf: PyRefMut<'py, Self>,
        circuit: &PyCircuit,
        theta: Option<RealArg<'_>>,
        noise: Option<&PyNoiseModel>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let theta = theta_or_empty(theta);
        check_finite("theta", &theta)?;
        let py = slf.py();
        let rho = &mut slf.inner;
        let c = &circuit.inner;
        let noise = noise.map(|n| &n.inner);
        py.detach(|| rho.apply_circuit(c, &theta, noise)).raise()?;
        Ok(slf)
    }

    /// Apply `channel` to `qubits` (one or two) in place.
    fn apply_channel<'py>(
        mut slf: PyRefMut<'py, Self>,
        channel: &PyChannel,
        qubits: Vec<usize>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner.apply_channel(&channel.inner, &qubits).raise()?;
        Ok(slf)
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    /// Diagonal of ρ.
    fn probabilities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        real_array(py, self.inner.probabilities())
    }

    /// Tr ρ² (1 for pure states, 2⁻ⁿ for the maximally mixed state).
    fn purity(&self) -> f64 {
        self.inner.purity()
    }

    /// Tr ρ.
    fn trace(&self) -> f64 {
        self.inner.trace()
    }

    /// Tr(ρH).
    fn expectation(&self, py: Python<'_>, observable: &PyPauliSum) -> PyResult<f64> {
        let (rho, h) = (&self.inner, &observable.inner);
        py.detach(|| rho.expectation(h)).raise()
    }

    /// ⟨ψ|ρ|ψ⟩.
    fn fidelity_with_pure(&self, state: &PyStateVector) -> PyResult<f64> {
        self.inner.fidelity_with_pure(&state.inner).raise()
    }

    /// Dense copy of ρ.
    fn to_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<metatron_qso::C64>> {
        cmatrix(py, &self.inner.to_matrix())
    }

    /// Measure all qubits `shots` times with optional readout flips
    /// `(P(1|0), P(0|1))`: `{bitstring: count}`.
    #[pyo3(signature = (shots, seed = 0, readout = None))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        shots: usize,
        seed: u64,
        readout: Option<(f64, f64)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if let Some((a, b)) = readout
            && !((0.0..=1.0).contains(&a) && (0.0..=1.0).contains(&b))
        {
            return Err(invalid("readout error probabilities must lie in [0, 1]"));
        }
        let rho = &self.inner;
        let counts = py.detach(|| rho.sample(shots, readout, &mut Rng::seed_from_u64(seed)));
        counts_dict(py, &counts, rho.n_qubits())
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "DensityMatrix(n_qubits={}, purity={:.6})",
            self.inner.n_qubits(),
            self.inner.purity()
        )
    }
}

/// Estimate ⟨H⟩ of `circuit(theta)` from `shots` measurements per group of
/// qubit-wise commuting terms — on the exact state-vector backend, or on
/// the density-matrix backend when `noise` is given.
#[pyfunction]
#[pyo3(signature = (circuit, observable, theta = None, shots = 10000, seed = 0, noise = None))]
fn estimate_expectation(
    py: Python<'_>,
    circuit: &PyCircuit,
    observable: &PyPauliSum,
    theta: Option<RealArg<'_>>,
    shots: usize,
    seed: u64,
    noise: Option<&PyNoiseModel>,
) -> PyResult<f64> {
    let theta = theta_or_empty(theta);
    check_finite("theta", &theta)?;
    let (c, h) = (&circuit.inner, &observable.inner);
    match noise {
        None => py.detach(|| estimate(&StatevectorBackend, c, &theta, h, shots, seed)),
        Some(n) => {
            let backend = DensityMatrixBackend::new(n.inner.clone());
            py.detach(|| estimate(&backend, c, &theta, h, shots, seed))
        }
    }
    .raise()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyChannel>()?;
    m.add_class::<PyNoiseModel>()?;
    m.add_class::<PyDensityMatrix>()?;
    m.add_function(wrap_pyfunction!(estimate_expectation, m)?)?;
    Ok(())
}
