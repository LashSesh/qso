//! Channels, noise models, density matrices and shot-based estimation.

use metatron_qso::Rng;
use metatron_qso::backend::estimate_expectation as estimate;
use metatron_qso::noise::{
    Channel, ChannelSpec, DensityMatrix, DeviceModel, GateRule, NoiseModel, QubitProperties,
    Spread, Telegraph, TrajectorySimulator,
};
use metatron_qso::quantum::MAX_QUBITS;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::circuit::{PyCircuit, theta_or_empty};
use crate::estimation::{PyEstimate, backend, wrap_estimate};
use crate::state::{PyPauliSum, PyStateVector};
use crate::{
    ComplexMatrixArg, Raise, RealArg, check_finite, cmatrix, counts_dict, invalid, real_array,
    square_array, square_complex,
};

/// Completely positive trace-preserving map on one to four qubits, given by
/// Kraus operators.
#[pyclass(name = "Channel", module = "metatron_qso", frozen)]
#[derive(Clone)]
pub struct PyChannel {
    pub inner: Channel,
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

    /// Channel from explicit Kraus operators on one to four qubits (2×2 to
    /// 16×16, checked for `Σ K†K = I`).
    #[staticmethod]
    #[pyo3(signature = (kraus, name = "custom"))]
    fn from_kraus(kraus: Vec<ComplexMatrixArg<'_>>, name: &str) -> PyResult<Self> {
        let ops = kraus
            .iter()
            .map(|k| square_complex("kraus", k).map(|(_, data)| data))
            .collect::<PyResult<Vec<_>>>()?;
        channel(Channel::from_kraus(name, ops))
    }

    /// Pauli channel from probabilities of non-identity Pauli strings of one
    /// to four letters, e.g. `{"ZZ": 0.02, "XI": 0.001}` (first letter on
    /// the first qubit).
    #[staticmethod]
    fn pauli(probabilities: std::collections::BTreeMap<String, f64>) -> PyResult<Self> {
        let list: Vec<(&str, f64)> = probabilities
            .iter()
            .map(|(k, &v)| (k.as_str(), v))
            .collect();
        channel(Channel::pauli(&list))
    }

    /// Unitary channel `ρ ↦ UρU†` (coherent error) from a unitary on one to
    /// four qubits.
    #[staticmethod]
    #[pyo3(signature = (matrix, name = "unitary"))]
    fn unitary(matrix: ComplexMatrixArg<'_>, name: &str) -> PyResult<Self> {
        let (_, data) = square_complex("matrix", &matrix)?;
        channel(Channel::unitary(name, data))
    }

    /// Coherent rotation `exp(−iθP/2)` about a Pauli string of one to four
    /// letters ("X", "ZZ", "ZZZ", …).
    #[staticmethod]
    fn rotation(pauli: &str, angle: f64) -> PyResult<Self> {
        channel(Channel::rotation(pauli, angle))
    }

    /// Probabilities `p_P` (base-4 index, first qubit most significant) if
    /// this is a Pauli channel, else `None`.
    fn pauli_probabilities<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.pauli_probabilities().map(|p| real_array(py, p))
    }

    /// The Pauli-twirled channel.
    fn twirled(&self) -> Self {
        Self {
            inner: self.inner.twirled(),
        }
    }

    /// `then ∘ self`: this channel followed by `then` on the same qubits.
    fn compose(&self, then: &PyChannel) -> PyResult<Self> {
        channel(self.inner.compose(&then.inner))
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
    pub inner: NoiseModel,
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

/// A device: per-qubit T1, T2 and readout errors, and gate rules (gate
/// name, qubits, size) with durations, depolarising errors, over-rotations
/// and further channels. The first matching rule applies to a gate. Builder
/// methods return a new model.
#[pyclass(name = "DeviceModel", module = "metatron_qso", frozen)]
#[derive(Clone)]
pub struct PyDeviceModel {
    pub inner: DeviceModel,
}

fn device(inner: DeviceModel) -> PyResult<PyDeviceModel> {
    inner.validate().raise()?;
    Ok(PyDeviceModel { inner })
}

fn spread(arg: Option<(String, f64)>) -> PyResult<Option<Spread>> {
    arg.map(|(kind, width)| match kind.to_ascii_lowercase().as_str() {
        "gaussian" => Ok(Spread::Gaussian { sigma: width }),
        "uniform" => Ok(Spread::Uniform { half_width: width }),
        _ => Err(invalid(format!(
            "unknown spread '{kind}' (expected gaussian or uniform)"
        ))),
    })
    .transpose()
}

fn qubit_properties(
    t1: Option<f64>,
    t2: Option<f64>,
    readout: Option<(f64, f64)>,
    detuning: Option<(String, f64)>,
    telegraph: Option<(f64, f64)>,
) -> PyResult<QubitProperties> {
    let mut q = QubitProperties::new();
    q.t1 = t1;
    q.t2 = t2;
    q.readout = readout;
    q.detuning = spread(detuning)?;
    q.telegraph = telegraph.map(|(amplitude, rate)| Telegraph::new(amplitude, rate));
    Ok(q)
}

#[pymethods]
impl PyDeviceModel {
    #[new]
    #[pyo3(signature = (name = "device", num_qubits = None, idle_noise = true))]
    fn new(name: &str, num_qubits: Option<usize>, idle_noise: bool) -> PyResult<Self> {
        let mut d = DeviceModel::new(name).with_idle_noise(idle_noise);
        if let Some(n) = num_qubits {
            d = d.with_num_qubits(n);
        }
        device(d)
    }

    /// Parse a JSON device description.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        DeviceModel::from_json(text)
            .raise()
            .map(|inner| Self { inner })
    }

    /// The same noise as a `NoiseModel`.
    #[staticmethod]
    fn from_noise_model(model: &PyNoiseModel) -> PyResult<Self> {
        device(DeviceModel::from(&model.inner))
    }

    /// The description as JSON.
    fn to_json(&self) -> String {
        self.inner.to_json()
    }

    /// Set T1, T2 (≤ 2·T1) and readout errors `(P(1|0), P(0|1))` of qubit
    /// `q`, a quasi-static frequency offset `detuning=("gaussian", sigma)`
    /// or `("uniform", half_width)` and telegraph noise
    /// `telegraph=(amplitude, rate)`; values left out follow the default
    /// qubit. Detunings and telegraph noise are correlated in time: only
    /// trajectories simulate them.
    #[pyo3(signature = (q, t1 = None, t2 = None, readout = None, detuning = None, telegraph = None))]
    #[allow(clippy::too_many_arguments)]
    fn with_qubit(
        &self,
        q: usize,
        t1: Option<f64>,
        t2: Option<f64>,
        readout: Option<(f64, f64)>,
        detuning: Option<(String, f64)>,
        telegraph: Option<(f64, f64)>,
    ) -> PyResult<Self> {
        let limit = self.inner.num_qubits.unwrap_or(MAX_QUBITS);
        if q >= limit {
            return Err(invalid(format!(
                "qubit {q} outside a device of {limit} qubits"
            )));
        }
        let properties = qubit_properties(t1, t2, readout, detuning, telegraph)?;
        device(self.inner.clone().with_qubit(q, properties))
    }

    /// Set the properties of every qubit not set with `with_qubit` (the
    /// same arguments).
    #[pyo3(signature = (t1 = None, t2 = None, readout = None, detuning = None, telegraph = None))]
    fn with_default_qubit(
        &self,
        t1: Option<f64>,
        t2: Option<f64>,
        readout: Option<(f64, f64)>,
        detuning: Option<(String, f64)>,
        telegraph: Option<(f64, f64)>,
    ) -> PyResult<Self> {
        let properties = qubit_properties(t1, t2, readout, detuning, telegraph)?;
        device(self.inner.clone().with_default_qubit(properties))
    }

    /// Append a gate rule. `gate` ("cx", "rz", "id" for `Circuit.id`, …),
    /// `qubits` (controls first) and `num_qubits` select gates (`None`:
    /// any); `channels` act after the gate (a k-qubit channel on every k
    /// consecutive qubits of the gate); `spectators` are `(qubits, channel)`
    /// pairs on fixed qubits, which may lie outside the gate (crosstalk);
    /// `over_rotation_spread=("gaussian", sigma)` adds a quasi-static error
    /// drawn once per trajectory to `over_rotation`.
    #[pyo3(signature = (
        gate = None, qubits = None, num_qubits = None, duration = 0.0, depolarizing = None,
        over_rotation = 0.0, channels = Vec::new(), spectators = Vec::new(),
        over_rotation_spread = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn with_rule(
        &self,
        gate: Option<&str>,
        qubits: Option<Vec<usize>>,
        num_qubits: Option<usize>,
        duration: f64,
        depolarizing: Option<f64>,
        over_rotation: f64,
        channels: Vec<PyRef<'_, PyChannel>>,
        spectators: Vec<(Vec<usize>, PyRef<'_, PyChannel>)>,
        over_rotation_spread: Option<(String, f64)>,
    ) -> PyResult<Self> {
        let mut rule = match gate {
            Some(g) => GateRule::for_gate(g),
            None => GateRule::any(),
        };
        rule.qubits = qubits;
        rule.num_qubits = num_qubits;
        rule.duration = duration;
        rule.depolarizing = depolarizing;
        rule.over_rotation = over_rotation;
        rule.channels = channels
            .iter()
            .map(|c| ChannelSpec::from(&c.inner))
            .collect();
        rule.over_rotation_spread = spread(over_rotation_spread)?;
        for (qs, c) in &spectators {
            rule = rule.with_spectator(qs, ChannelSpec::from(&c.inner));
        }
        device(self.inner.clone().with_rule(rule))
    }

    /// Whether all noise is channels at points of the circuit (no
    /// detunings, telegraph noise or over-rotation spreads, which only
    /// `TrajectorySimulator` simulates).
    #[getter]
    fn is_markovian(&self) -> bool {
        self.inner.is_markovian()
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn num_qubits(&self) -> Option<usize> {
        self.inner.num_qubits
    }

    #[getter]
    fn idle_noise(&self) -> bool {
        self.inner.idle_noise
    }

    /// Readout errors `(P(1|0), P(0|1))` of qubits `0..n`.
    fn readout_errors(&self, n: usize) -> PyResult<Vec<(f64, f64)>> {
        if n > MAX_QUBITS {
            return Err(invalid(format!("at most {MAX_QUBITS} qubits")));
        }
        Ok(self.inner.readout_errors(n))
    }

    fn __repr__(&self) -> String {
        format!(
            "DeviceModel(name={:?}, num_qubits={}, qubit_entries={}, rules={})",
            self.inner.name,
            self.inner
                .num_qubits
                .map_or_else(|| "None".to_string(), |n| n.to_string()),
            self.inner.qubits.len(),
            self.inner.gates.len()
        )
    }
}

/// Density matrix of up to 14 qubits.
#[pyclass(name = "DensityMatrix", module = "metatron_qso")]
#[derive(Clone)]
pub struct PyDensityMatrix {
    pub inner: DensityMatrix,
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
    /// gate if given, or with the noise of `device`.
    #[staticmethod]
    #[pyo3(signature = (circuit, theta = None, noise = None, device = None))]
    fn simulate(
        py: Python<'_>,
        circuit: &PyCircuit,
        theta: Option<RealArg<'_>>,
        noise: Option<&PyNoiseModel>,
        device: Option<&PyDeviceModel>,
    ) -> PyResult<Self> {
        let theta = theta_or_empty(theta);
        check_finite("theta", &theta)?;
        if noise.is_some() && device.is_some() {
            return Err(invalid("give either `noise` or `device`, not both"));
        }
        let c = &circuit.inner;
        let noise = noise.map(|n| &n.inner);
        let device = device.map(|d| &d.inner);
        py.detach(|| {
            let mut rho = DensityMatrix::zero(c.n_qubits())?;
            match device {
                Some(d) => rho.apply_device(c, &theta, d)?,
                None => rho.apply_circuit(c, &theta, noise)?,
            }
            Ok(rho)
        })
        .raise()
        .map(dm)
    }

    /// Apply `circuit(theta)` (with optional `noise` or `device`) in place.
    #[pyo3(signature = (circuit, theta = None, noise = None, device = None))]
    fn apply<'py>(
        mut slf: PyRefMut<'py, Self>,
        circuit: &PyCircuit,
        theta: Option<RealArg<'_>>,
        noise: Option<&PyNoiseModel>,
        device: Option<&PyDeviceModel>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let theta = theta_or_empty(theta);
        check_finite("theta", &theta)?;
        if noise.is_some() && device.is_some() {
            return Err(invalid("give either `noise` or `device`, not both"));
        }
        let py = slf.py();
        let rho = &mut slf.inner;
        let c = &circuit.inner;
        let noise = noise.map(|n| &n.inner);
        let device = device.map(|d| &d.inner);
        py.detach(|| match device {
            Some(d) => rho.apply_device(c, &theta, d),
            None => rho.apply_circuit(c, &theta, noise),
        })
        .raise()?;
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

    /// Measure all qubits with per-qubit readout errors
    /// `[(P(1|0), P(0|1)), …]` (qubit 0 first).
    #[pyo3(signature = (shots, readout, seed = 0))]
    fn sample_with_readout<'py>(
        &self,
        py: Python<'py>,
        shots: usize,
        readout: Vec<(f64, f64)>,
        seed: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        if readout.len() != self.inner.n_qubits() {
            return Err(invalid(format!(
                "need one (P(1|0), P(0|1)) pair per qubit ({}), got {}",
                self.inner.n_qubits(),
                readout.len()
            )));
        }
        if readout
            .iter()
            .any(|&(a, b)| !((0.0..=1.0).contains(&a) && (0.0..=1.0).contains(&b)))
        {
            return Err(invalid("readout error probabilities must lie in [0, 1]"));
        }
        let rho = &self.inner;
        let counts =
            py.detach(|| rho.sample_with_readout(shots, &readout, &mut Rng::seed_from_u64(seed)));
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

/// Simulates noisy circuits by averaging quantum trajectories (Monte-Carlo
/// wave functions) on state vectors: any register size the state-vector
/// simulator supports.
#[pyclass(name = "TrajectorySimulator", module = "metatron_qso", frozen)]
pub struct PyTrajectorySimulator {
    inner: TrajectorySimulator,
}

#[pymethods]
impl PyTrajectorySimulator {
    #[new]
    #[pyo3(signature = (device, trajectories, seed = 0))]
    fn new(device: &PyDeviceModel, trajectories: usize, seed: u64) -> PyResult<Self> {
        TrajectorySimulator::new(device.inner.clone(), trajectories, seed)
            .raise()
            .map(|inner| Self { inner })
    }

    #[getter]
    fn trajectories(&self) -> usize {
        self.inner.trajectories()
    }

    /// Average of ⟨ψ|H|ψ⟩ over the trajectories, with its standard error.
    #[pyo3(signature = (circuit, observable, theta = None))]
    fn expectation(
        &self,
        py: Python<'_>,
        circuit: &PyCircuit,
        observable: &PyPauliSum,
        theta: Option<RealArg<'_>>,
    ) -> PyResult<PyEstimate> {
        let theta = theta_or_empty(theta);
        check_finite("theta", &theta)?;
        let (s, c, h) = (&self.inner, &circuit.inner, &observable.inner);
        py.detach(|| s.expectation(c, &theta, h))
            .raise()
            .map(wrap_estimate)
    }

    /// Averages of several observables over the same trajectories.
    #[pyo3(signature = (circuit, observables, theta = None))]
    fn expectations(
        &self,
        py: Python<'_>,
        circuit: &PyCircuit,
        observables: Vec<PyRef<'_, PyPauliSum>>,
        theta: Option<RealArg<'_>>,
    ) -> PyResult<Vec<PyEstimate>> {
        let theta = theta_or_empty(theta);
        check_finite("theta", &theta)?;
        let hs: Vec<_> = observables.iter().map(|h| h.inner.clone()).collect();
        let (s, c) = (&self.inner, &circuit.inner);
        let es = py.detach(|| s.expectations(c, &theta, &hs)).raise()?;
        Ok(es.into_iter().map(wrap_estimate).collect())
    }

    /// Measure all qubits `shots` times with the device's readout errors.
    #[pyo3(signature = (circuit, shots, theta = None))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        circuit: &PyCircuit,
        shots: usize,
        theta: Option<RealArg<'_>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let theta = theta_or_empty(theta);
        check_finite("theta", &theta)?;
        let (s, c) = (&self.inner, &circuit.inner);
        let counts = py.detach(|| s.sample(c, &theta, shots)).raise()?;
        counts_dict(py, &counts, c.n_qubits())
    }

    /// Average of |ψ⟩⟨ψ| over the trajectories (up to 14 qubits).
    #[pyo3(signature = (circuit, theta = None))]
    fn density_matrix(
        &self,
        py: Python<'_>,
        circuit: &PyCircuit,
        theta: Option<RealArg<'_>>,
    ) -> PyResult<PyDensityMatrix> {
        let theta = theta_or_empty(theta);
        check_finite("theta", &theta)?;
        let (s, c) = (&self.inner, &circuit.inner);
        py.detach(|| s.density_matrix(c, &theta)).raise().map(dm)
    }

    /// Final state of trajectory `k`.
    #[pyo3(signature = (circuit, k, theta = None))]
    fn trajectory_state(
        &self,
        py: Python<'_>,
        circuit: &PyCircuit,
        k: usize,
        theta: Option<RealArg<'_>>,
    ) -> PyResult<PyStateVector> {
        let theta = theta_or_empty(theta);
        check_finite("theta", &theta)?;
        let (s, c) = (&self.inner, &circuit.inner);
        py.detach(|| s.trajectory_state(c, &theta, k))
            .raise()
            .map(crate::state::wrap)
    }

    fn __repr__(&self) -> String {
        format!(
            "TrajectorySimulator(device={:?}, trajectories={}, seed={})",
            self.inner.device().name,
            self.inner.trajectories(),
            self.inner.seed()
        )
    }
}

/// Estimate ⟨H⟩ of `circuit(theta)` from `shots` measurements *per group*
/// of qubit-wise commuting terms — on the exact state-vector backend, with
/// a `noise` model, or on a `device` (density matrices, or at most
/// `trajectories` trajectories per run). See `estimate` for a total shot
/// budget, allocation and standard errors.
#[pyfunction]
#[pyo3(signature = (
    circuit, observable, theta = None, shots = 10000, seed = 0, noise = None, device = None,
    trajectories = None
))]
#[allow(clippy::too_many_arguments)]
fn estimate_expectation(
    py: Python<'_>,
    circuit: &PyCircuit,
    observable: &PyPauliSum,
    theta: Option<RealArg<'_>>,
    shots: usize,
    seed: u64,
    noise: Option<&PyNoiseModel>,
    device: Option<&PyDeviceModel>,
    trajectories: Option<usize>,
) -> PyResult<f64> {
    let theta = theta_or_empty(theta);
    check_finite("theta", &theta)?;
    let backend = backend(noise, device, trajectories)?;
    let (c, h) = (&circuit.inner, &observable.inner);
    py.detach(|| estimate(backend.as_ref(), c, &theta, h, shots, seed))
        .raise()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyChannel>()?;
    m.add_class::<PyNoiseModel>()?;
    m.add_class::<PyDensityMatrix>()?;
    m.add_class::<PyDeviceModel>()?;
    m.add_class::<PyTrajectorySimulator>()?;
    m.add_function(wrap_pyfunction!(estimate_expectation, m)?)?;
    Ok(())
}
