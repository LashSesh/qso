//! Detector circuits, Pauli-frame sampling, detector error models, the
//! union-find, matching and BP+OSD decoders, CSS codes and memory
//! experiments.

use metatron_qso::qec::{
    self, BpMethod, BpOsdDecoder, BpOsdOptions, BpOsdOutcome, CircuitDistance, CircuitNoise,
    ConditionedDecoder, CorrelatedError, CorrelatedErrorModel, CorrelatedMatchingDecoder, CssCode,
    CssSchedule, Decoder, DecoderChoice, DetectorCircuit, DetectorErrorModel, DistanceMethod,
    DistanceOptions, ErasureNoise, ErrorMechanism, ExactDistanceOptions, HeraldedErrorModel,
    HeraldedMethod, MatchingDecoder, MatchingGraph, MemoryBasis, Noise, OsdMethod,
    ParallelWindowOptions, RepeatedCircuit, SparseMatchingDecoder, StreamState, UnionFindDecoder,
    WindowDecoder, WindowMethod, WindowOptions,
};
use numpy::ndarray::Array2;
use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArrayLike2};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::circuit::{PyCircuit, theta_or_empty};
use crate::{Raise, RealArg, invalid, pydict};

/// Most entries (shots × (detectors + observables)) one call returns as
/// arrays.
const MAX_ARRAY_ENTRIES: usize = 1 << 30;

/// A Clifford circuit with Pauli noise, a measurement record, detectors
/// and logical observables, in the manner of Stim.
///
/// Measurements return their index in the record; detectors are parities
/// of records that are deterministic without noise, observables are
/// parities whose flips a decoder predicts.
#[pyclass(name = "DetectorCircuit", module = "metatron_qso")]
#[derive(Clone)]
pub struct PyDetectorCircuit {
    pub inner: DetectorCircuit,
}

type Slf<'py> = PyRefMut<'py, PyDetectorCircuit>;

/// Noise argument of the memory experiments: a `CircuitNoise`, a
/// probability for every kind of noise, or an `ErasureNoise`.
enum NoiseArg {
    Model(CircuitNoise),
    Uniform(f64),
    Erasure(ErasureNoise),
}

impl<'a, 'py> FromPyObject<'a, 'py> for NoiseArg {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(n) = obj.cast::<PyCircuitNoise>() {
            return Ok(NoiseArg::Model(n.get().inner));
        }
        if let Ok(n) = obj.cast::<PyErasureNoise>() {
            return Ok(NoiseArg::Erasure(n.get().inner));
        }
        Ok(NoiseArg::Uniform(obj.extract::<f64>()?))
    }
}

/// The noise of a memory experiment: circuit noise, or erasure-biased
/// noise (built by the `*_erasure` generators).
enum MemoryNoise {
    Plain(CircuitNoise),
    Erasure(ErasureNoise),
}

impl MemoryNoise {
    /// No noise when `noise` is None.
    fn of(noise: Option<NoiseArg>) -> Self {
        match noise {
            None => MemoryNoise::Plain(CircuitNoise::default()),
            Some(NoiseArg::Model(m)) => MemoryNoise::Plain(m),
            Some(NoiseArg::Uniform(p)) => MemoryNoise::Plain(CircuitNoise::uniform(p)),
            Some(NoiseArg::Erasure(e)) => MemoryNoise::Erasure(e),
        }
    }
}

impl PyDetectorCircuit {
    fn add_noise<'py>(mut slf: Slf<'py>, noise: Noise, qubits: Vec<usize>) -> PyResult<Slf<'py>> {
        slf.inner.noise(noise, &qubits).raise()?;
        Ok(slf)
    }
}

#[pymethods]
impl PyDetectorCircuit {
    /// An empty circuit on `n_qubits` qubits, all starting in `|0⟩`.
    #[new]
    fn new(n_qubits: usize) -> PyResult<Self> {
        Ok(Self {
            inner: DetectorCircuit::new(n_qubits).raise()?,
        })
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    #[getter]
    fn n_measurements(&self) -> usize {
        self.inner.n_measurements()
    }

    #[getter]
    fn n_detectors(&self) -> usize {
        self.inner.n_detectors()
    }

    #[getter]
    fn n_observables(&self) -> usize {
        self.inner.n_observables()
    }

    /// The layer (round) of every detector: detectors declared one after
    /// another form one layer, as each round of a memory experiment
    /// declares its detectors after its measurements.
    fn detector_layers(&self) -> Vec<usize> {
        self.inner.detector_layers()
    }

    /// Operations, heralded ones included.
    fn __len__(&self) -> usize {
        self.inner.n_steps()
    }

    fn h(mut slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        slf.inner.h(qubit).raise()?;
        Ok(slf)
    }

    fn s(mut slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        slf.inner.s(qubit).raise()?;
        Ok(slf)
    }

    fn x(mut slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        slf.inner.x(qubit).raise()?;
        Ok(slf)
    }

    fn z(mut slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        slf.inner.z(qubit).raise()?;
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

    /// Append the gates of a Clifford circuit (parameters bound to
    /// `theta`) on the first qubits.
    #[pyo3(signature = (circuit, theta = None))]
    fn append<'py>(
        mut slf: Slf<'py>,
        circuit: &PyCircuit,
        theta: Option<RealArg<'_>>,
    ) -> PyResult<Slf<'py>> {
        slf.inner
            .append_circuit(&circuit.inner, &theta_or_empty(theta))
            .raise()?;
        Ok(slf)
    }

    /// Measure `qubit` in the Z basis, the result flipped with
    /// probability `flip`; returns its index in the measurement record.
    #[pyo3(signature = (qubit, flip = 0.0))]
    fn measure(&mut self, qubit: usize, flip: f64) -> PyResult<usize> {
        self.inner.measure(qubit, flip).raise()
    }

    /// Reset `qubit` to `|0⟩`.
    fn reset(mut slf: Slf<'_>, qubit: usize) -> PyResult<Slf<'_>> {
        slf.inner.reset(qubit).raise()?;
        Ok(slf)
    }

    /// X with probability `p` on each qubit.
    fn x_error(slf: Slf<'_>, p: f64, qubits: Vec<usize>) -> PyResult<Slf<'_>> {
        Self::add_noise(slf, Noise::XError(p), qubits)
    }

    /// Y with probability `p` on each qubit.
    fn y_error(slf: Slf<'_>, p: f64, qubits: Vec<usize>) -> PyResult<Slf<'_>> {
        Self::add_noise(slf, Noise::YError(p), qubits)
    }

    /// Z with probability `p` on each qubit.
    fn z_error(slf: Slf<'_>, p: f64, qubits: Vec<usize>) -> PyResult<Slf<'_>> {
        Self::add_noise(slf, Noise::ZError(p), qubits)
    }

    /// X, Y or Z, each with probability p/3, on each qubit (p ≤ 3/4).
    fn depolarize1(slf: Slf<'_>, p: f64, qubits: Vec<usize>) -> PyResult<Slf<'_>> {
        Self::add_noise(slf, Noise::Depolarize1(p), qubits)
    }

    /// One of the 15 non-identity two-qubit Paulis, each with probability
    /// p/15, on consecutive pairs of `qubits` (p ≤ 15/16).
    fn depolarize2(slf: Slf<'_>, p: f64, qubits: Vec<usize>) -> PyResult<Slf<'_>> {
        Self::add_noise(slf, Noise::Depolarize2(p), qubits)
    }

    /// Stim's `HERALDED_ERASE(p)`: for each qubit in order, with
    /// probability `p` the qubit is replaced by a maximally mixed one (I, X,
    /// Y or Z with p/4 each) and its herald, a new measurement record, reads
    /// 1; otherwise the herald reads 0. Returns the herald's record of each
    /// qubit (a qubit listed twice is erased twice).
    fn heralded_erase(&mut self, p: f64, qubits: Vec<usize>) -> PyResult<Vec<usize>> {
        self.inner.heralded_erase(p, &qubits).raise()
    }

    /// Stim's `HERALDED_PAULI_CHANNEL_1(p_i, p_x, p_y, p_z)`: for each
    /// qubit, disjointly, the herald reads 1 with I, X, Y or Z with these
    /// probabilities, and 0 with no error otherwise (the sum at most 1).
    /// Returns the herald's record of each qubit.
    fn heralded_pauli_channel_1(
        &mut self,
        p_i: f64,
        p_x: f64,
        p_y: f64,
        p_z: f64,
        qubits: Vec<usize>,
    ) -> PyResult<Vec<usize>> {
        self.inner
            .heralded_pauli_channel_1([p_i, p_x, p_y, p_z], &qubits)
            .raise()
    }

    /// Declare a detector on these measurement records; returns its index.
    fn detector(&mut self, records: Vec<usize>) -> PyResult<usize> {
        self.inner.detector(&records).raise()
    }

    /// Add these measurement records to logical observable `index` (< 64).
    fn observable(mut slf: Slf<'_>, index: usize, records: Vec<usize>) -> PyResult<Slf<'_>> {
        slf.inner.observable(index, &records).raise()?;
        Ok(slf)
    }

    /// The circuit without noise.
    fn without_noise(&self) -> Self {
        Self {
            inner: self.inner.without_noise(),
        }
    }

    /// The circuit in Stim's text format.
    fn to_stim(&self) -> PyResult<String> {
        self.inner.to_stim().raise()
    }

    /// A circuit in Stim's text format: Clifford gates, measurements and
    /// resets in the X, Y and Z bases, X/Y/Z errors, one- and two-qubit
    /// depolarising noise, heralded erasures and heralded Pauli channels
    /// (`HERALDED_ERASE`, `HERALDED_PAULI_CHANNEL_1`), detectors,
    /// observables and REPEAT blocks.
    #[staticmethod]
    fn from_stim(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: DetectorCircuit::from_stim(text).raise()?,
        })
    }

    /// Sample `shots` shots: `(detectors, observables)`, boolean arrays of
    /// shape (shots, n_detectors) and (shots, n_observables) that say
    /// whether noise flipped each detector and observable.
    #[pyo3(signature = (shots, seed = 0))]
    #[allow(clippy::type_complexity)]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        shots: usize,
        seed: u64,
    ) -> PyResult<(Bound<'py, PyArray2<bool>>, Bound<'py, PyArray2<bool>>)> {
        let (nd, no) = (self.inner.n_detectors(), self.inner.n_observables());
        if shots.saturating_mul(nd + no) > MAX_ARRAY_ENTRIES {
            return Err(invalid(
                "shots × (detectors + observables) must be at most 2³⁰",
            ));
        }
        let c = &self.inner;
        sample_arrays(py, shots, nd, no, || qec::sample_detectors(c, shots, seed))
    }

    /// The independent error mechanisms of the circuit. With `decompose`,
    /// errors that flip more than two detectors are split into their
    /// single-qubit X and Z parts, so that the model is a matching graph.
    #[pyo3(signature = (decompose = false))]
    fn detector_error_model(
        &self,
        py: Python<'_>,
        decompose: bool,
    ) -> PyResult<PyDetectorErrorModel> {
        let c = &self.inner;
        Ok(PyDetectorErrorModel {
            inner: py
                .detach(|| qec::detector_error_model(c, decompose))
                .raise()?,
        })
    }

    /// The decomposed model with, for every error, the graph-like parts it
    /// was split into (for `CorrelatedMatchingDecoder`).
    fn correlated_error_model(&self, py: Python<'_>) -> PyResult<PyCorrelatedErrorModel> {
        let c = &self.inner;
        Ok(PyCorrelatedErrorModel {
            inner: py.detach(|| qec::correlated_error_model(c)).raise()?,
        })
    }

    /// The model a `ConditionedDecoder` conditions on the heralds that
    /// fired: its herald detectors (detectors whose only record is a herald
    /// nothing else reads), for each the probability that it fires and its
    /// channel's parts given that it fired, and the errors of every other
    /// channel (the base, by Stim's approximate rule; decomposed with
    /// `decompose`, the parts never). The decomposed model (the default) is
    /// for union-find and matching; BP+OSD takes `decompose=False`, which
    /// is also the only model of circuits whose errors flip three
    /// detectors after splitting, such as colour codes.
    #[pyo3(signature = (decompose = true))]
    fn heralded_error_model(
        &self,
        py: Python<'_>,
        decompose: bool,
    ) -> PyResult<PyHeraldedErrorModel> {
        let c = &self.inner;
        Ok(PyHeraldedErrorModel {
            inner: py
                .detach(|| qec::heralded_error_model(c, decompose))
                .raise()?,
        })
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "DetectorCircuit(n_qubits={}, operations={}, detectors={}, observables={})",
            self.inner.n_qubits(),
            self.inner.n_steps(),
            self.inner.n_detectors(),
            self.inner.n_observables()
        )
    }
}

/// Detection events and observable flips as boolean arrays of shape
/// (shots, nd) and (shots, no).
type SampleArrays<'py> = (Bound<'py, PyArray2<bool>>, Bound<'py, PyArray2<bool>>);

fn sample_arrays<'py>(
    py: Python<'py>,
    shots: usize,
    nd: usize,
    no: usize,
    sample: impl FnOnce() -> metatron_qso::Result<qec::DetectorSamples> + Send,
) -> PyResult<SampleArrays<'py>> {
    let (det, obs) = py
        .detach(|| {
            let s = sample()?;
            let det = Array2::from_shape_fn((shots, nd), |(k, d)| s.fired(d, k));
            let obs =
                Array2::from_shape_fn((shots, no), |(k, i)| (s.observable_flips(k) >> i) & 1 == 1);
            Ok::<_, metatron_qso::QsoError>((det, obs))
        })
        .raise()?;
    Ok((det.into_pyarray(py), obs.into_pyarray(py)))
}

/// A detector circuit that keeps its repeated blocks (Stim's `REPEAT`),
/// from `from_stim`, `from_circuit` or the `*_memory_repeated`
/// experiments: each block is stored once, with its measurement records
/// counted back. It samples as it runs, keeping only the records a
/// detector can still reach, and its detector error model is folded: one
/// pass backwards that, once an iteration's start is its end shifted by
/// one iteration, adds the earlier iterations' errors shifted.
#[pyclass(name = "RepeatedCircuit", module = "metatron_qso", frozen)]
#[derive(Clone)]
pub struct PyRepeatedCircuit {
    pub inner: RepeatedCircuit,
}

#[pymethods]
impl PyRepeatedCircuit {
    /// A circuit in Stim's text format with its REPEAT blocks kept (the
    /// instructions `DetectorCircuit.from_stim` reads).
    #[staticmethod]
    fn from_stim(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: RepeatedCircuit::from_stim(text).raise()?,
        })
    }

    /// A flat circuit as a repeated one without blocks.
    #[staticmethod]
    fn from_circuit(circuit: &PyDetectorCircuit) -> PyResult<Self> {
        Ok(Self {
            inner: RepeatedCircuit::from_circuit(&circuit.inner).raise()?,
        })
    }

    /// The circuit in Stim's text format, its blocks as REPEAT.
    fn to_stim(&self) -> PyResult<String> {
        self.inner.to_stim().raise()
    }

    /// Every block expanded, if it fits a `DetectorCircuit`.
    fn flatten(&self) -> PyResult<PyDetectorCircuit> {
        Ok(PyDetectorCircuit {
            inner: self.inner.flatten().raise()?,
        })
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    /// Measurements, expanded.
    #[getter]
    fn n_measurements(&self) -> usize {
        self.inner.n_measurements()
    }

    /// Detectors, expanded.
    #[getter]
    fn n_detectors(&self) -> usize {
        self.inner.n_detectors()
    }

    #[getter]
    fn n_observables(&self) -> usize {
        self.inner.n_observables()
    }

    /// Operations, expanded.
    #[getter]
    fn n_operations(&self) -> u64 {
        self.inner.n_operations()
    }

    /// Operations stored (each block once).
    #[getter]
    fn n_stored_operations(&self) -> usize {
        self.inner.n_stored_operations()
    }

    /// The layer (round) of every detector, as for the expanded circuit.
    fn detector_layers(&self) -> Vec<usize> {
        self.inner.detector_layers()
    }

    /// Sample `shots` shots as `DetectorCircuit.sample` samples the
    /// expanded circuit (the same shots for the same seed).
    #[pyo3(signature = (shots, seed = 0))]
    fn sample<'py>(&self, py: Python<'py>, shots: usize, seed: u64) -> PyResult<SampleArrays<'py>> {
        let (nd, no) = (self.inner.n_detectors(), self.inner.n_observables());
        if shots.saturating_mul(nd + no) > MAX_ARRAY_ENTRIES {
            return Err(invalid(
                "shots × (detectors + observables) must be at most 2³⁰",
            ));
        }
        let c = &self.inner;
        sample_arrays(py, shots, nd, no, || c.sample(shots, seed))
    }

    /// The detector error model of the expanded circuit (as
    /// `DetectorCircuit.detector_error_model` gives it, up to the rounding
    /// of merged probabilities), folded.
    #[pyo3(signature = (decompose = false))]
    fn detector_error_model(
        &self,
        py: Python<'_>,
        decompose: bool,
    ) -> PyResult<PyDetectorErrorModel> {
        let c = &self.inner;
        Ok(PyDetectorErrorModel {
            inner: py.detach(|| c.detector_error_model(decompose)).raise()?,
        })
    }

    /// The correlated model of the expanded circuit, folded.
    fn correlated_error_model(&self, py: Python<'_>) -> PyResult<PyCorrelatedErrorModel> {
        let c = &self.inner;
        Ok(PyCorrelatedErrorModel {
            inner: py.detach(|| c.correlated_error_model()).raise()?,
        })
    }

    /// The heralded model of the expanded circuit
    /// (`DetectorCircuit.heralded_error_model`), folded.
    #[pyo3(signature = (decompose = true))]
    fn heralded_error_model(
        &self,
        py: Python<'_>,
        decompose: bool,
    ) -> PyResult<PyHeraldedErrorModel> {
        let c = &self.inner;
        Ok(PyHeraldedErrorModel {
            inner: py.detach(|| c.heralded_error_model(decompose)).raise()?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "RepeatedCircuit(n_qubits={}, operations={}, stored={}, detectors={}, observables={})",
            self.inner.n_qubits(),
            self.inner.n_operations(),
            self.inner.n_stored_operations(),
            self.inner.n_detectors(),
            self.inner.n_observables()
        )
    }
}

/// Circuit-level noise of the memory experiments; every probability
/// defaults to 0.
#[pyclass(name = "CircuitNoise", module = "metatron_qso", frozen, eq)]
#[derive(Clone, PartialEq)]
pub struct PyCircuitNoise {
    inner: CircuitNoise,
}

#[pymethods]
impl PyCircuitNoise {
    #[new]
    #[pyo3(signature = (*, after_gate1 = 0.0, after_gate2 = 0.0, measurement = 0.0, after_reset = 0.0, before_round = 0.0))]
    fn new(
        after_gate1: f64,
        after_gate2: f64,
        measurement: f64,
        after_reset: f64,
        before_round: f64,
    ) -> Self {
        Self {
            inner: CircuitNoise {
                after_gate1,
                after_gate2,
                measurement,
                after_reset,
                before_round,
            },
        }
    }

    /// Every kind of noise with probability `p`.
    #[staticmethod]
    fn uniform(p: f64) -> Self {
        Self {
            inner: CircuitNoise::uniform(p),
        }
    }

    /// Only depolarizing noise on the data qubits before each round.
    #[staticmethod]
    fn data_only(p: f64) -> Self {
        Self {
            inner: CircuitNoise::data_only(p),
        }
    }

    /// Depolarizing noise after each single-qubit gate.
    #[getter]
    fn after_gate1(&self) -> f64 {
        self.inner.after_gate1
    }

    /// Two-qubit depolarizing noise after each CNOT.
    #[getter]
    fn after_gate2(&self) -> f64 {
        self.inner.after_gate2
    }

    /// Flip probability of each measurement result.
    #[getter]
    fn measurement(&self) -> f64 {
        self.inner.measurement
    }

    /// X error after each reset.
    #[getter]
    fn after_reset(&self) -> f64 {
        self.inner.after_reset
    }

    /// Depolarizing noise on the data qubits before each round.
    #[getter]
    fn before_round(&self) -> f64 {
        self.inner.before_round
    }

    fn __repr__(&self) -> String {
        let n = &self.inner;
        format!(
            "CircuitNoise(after_gate1={}, after_gate2={}, measurement={}, after_reset={}, before_round={})",
            n.after_gate1, n.after_gate2, n.measurement, n.after_reset, n.before_round
        )
    }
}

/// Erasure-biased circuit noise for the memory experiments: a share
/// `erasure_share` (in [0, 1]) of every probability of `noise` (a
/// `CircuitNoise`, or one probability for every kind of noise) is a
/// heralded erasure (Stim's `HERALDED_ERASE`) of each qubit the operation
/// touches, its herald a detector of the round; the rest acts as `noise`.
/// A CNOT erases each of its qubits independently, so its total error is
/// about (1 + e)·p.
#[pyclass(name = "ErasureNoise", module = "metatron_qso", frozen, eq)]
#[derive(Clone, PartialEq)]
pub struct PyErasureNoise {
    inner: ErasureNoise,
}

#[pymethods]
impl PyErasureNoise {
    #[new]
    fn new(noise: NoiseArg, erasure_share: f64) -> PyResult<Self> {
        let noise = match noise {
            NoiseArg::Model(m) => m,
            NoiseArg::Uniform(p) => CircuitNoise::uniform(p),
            NoiseArg::Erasure(_) => {
                return Err(PyTypeError::new_err(
                    "noise must be a CircuitNoise or a probability",
                ));
            }
        };
        Ok(Self {
            inner: ErasureNoise::new(noise, erasure_share).raise()?,
        })
    }

    /// The noise whose probabilities are split.
    #[getter]
    fn noise(&self) -> PyCircuitNoise {
        PyCircuitNoise {
            inner: self.inner.noise,
        }
    }

    /// The heralded share of each probability.
    #[getter]
    fn erasure_share(&self) -> f64 {
        self.inner.erasure_share
    }

    fn __repr__(&self) -> String {
        let n = &self.inner.noise;
        format!(
            "ErasureNoise(noise=CircuitNoise(after_gate1={}, after_gate2={}, measurement={}, after_reset={}, before_round={}), erasure_share={})",
            n.after_gate1,
            n.after_gate2,
            n.measurement,
            n.after_reset,
            n.before_round,
            self.inner.erasure_share
        )
    }
}

/// Independent error mechanisms of a circuit: `errors` lists
/// `(probability, detectors, observables)` with `observables` a bit mask.
/// Built by `DetectorCircuit.detector_error_model`, or directly from such
/// a list (for instance the columns of a parity-check matrix).
#[pyclass(name = "DetectorErrorModel", module = "metatron_qso", frozen)]
pub struct PyDetectorErrorModel {
    inner: DetectorErrorModel,
}

#[pymethods]
impl PyDetectorErrorModel {
    /// `n_observables` defaults to one more than the highest observable
    /// bit used.
    #[new]
    #[pyo3(signature = (n_detectors, errors, n_observables = None))]
    fn new(
        n_detectors: usize,
        errors: Vec<(f64, Vec<usize>, u64)>,
        n_observables: Option<usize>,
    ) -> PyResult<Self> {
        if n_detectors > qec::MAX_MODEL_DETECTORS {
            return Err(invalid(format!(
                "at most {} detectors",
                qec::MAX_MODEL_DETECTORS
            )));
        }
        if n_observables.is_some_and(|o| o > qec::MAX_OBSERVABLES) {
            return Err(invalid(format!(
                "at most {} observables",
                qec::MAX_OBSERVABLES
            )));
        }
        let mut out = Vec::with_capacity(errors.len());
        for (k, (probability, mut detectors, observables)) in errors.into_iter().enumerate() {
            if !(0.0..=1.0).contains(&probability) {
                return Err(invalid(format!(
                    "error {k}: {probability} is not a probability"
                )));
            }
            detectors.sort_unstable();
            if detectors.windows(2).any(|w| w[0] == w[1]) {
                return Err(invalid(format!("error {k} lists a detector twice")));
            }
            if let Some(&d) = detectors.last().filter(|&&d| d >= n_detectors) {
                return Err(invalid(format!(
                    "error {k} flips detector {d} of {n_detectors}"
                )));
            }
            out.push(ErrorMechanism {
                probability,
                detectors,
                observables,
            });
        }
        let used = out
            .iter()
            .map(|e| 64 - e.observables.leading_zeros() as usize)
            .max()
            .unwrap_or(0);
        let n_observables = n_observables.unwrap_or(used);
        if n_observables < used {
            return Err(invalid(format!(
                "observable {} is used but n_observables is {n_observables}",
                used - 1
            )));
        }
        Ok(Self {
            inner: DetectorErrorModel {
                n_detectors,
                n_observables,
                errors: out,
            },
        })
    }

    #[getter]
    fn n_detectors(&self) -> usize {
        self.inner.n_detectors
    }

    #[getter]
    fn n_observables(&self) -> usize {
        self.inner.n_observables
    }

    #[getter]
    fn errors(&self) -> Vec<PyMechanism> {
        mechanisms(&self.inner.errors)
    }

    fn __len__(&self) -> usize {
        self.inner.errors.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "DetectorErrorModel(detectors={}, observables={}, errors={})",
            self.inner.n_detectors,
            self.inner.n_observables,
            self.inner.errors.len()
        )
    }
}

/// An error as Python lists it: (probability, detectors, observables).
type PyMechanism = (f64, Vec<usize>, u64);

fn mechanisms(errors: &[ErrorMechanism]) -> Vec<PyMechanism> {
    errors
        .iter()
        .map(|e| (e.probability, e.detectors.clone(), e.observables))
        .collect()
}

/// The model a `ConditionedDecoder` conditions on the heralds that fired
/// (from `DetectorCircuit.heralded_error_model`, or built from its parts):
/// `base`, the errors of every channel no herald detector owns;
/// `herald_detectors`, ascending; for each herald `h` (an index into
/// `herald_detectors`) the probability that it fires and its `outcomes`,
/// the parts of its channel given that it fired, as
/// `(probability, detectors, observables)` (½ each for an erasure).
#[pyclass(name = "HeraldedErrorModel", module = "metatron_qso", frozen)]
pub struct PyHeraldedErrorModel {
    inner: HeraldedErrorModel,
}

impl PyHeraldedErrorModel {
    fn check_herald(&self, h: usize) -> PyResult<()> {
        let n = self.inner.herald_detectors().len();
        if h >= n {
            return Err(invalid(format!("herald {h} of {n}")));
        }
        Ok(())
    }
}

#[pymethods]
impl PyHeraldedErrorModel {
    /// Whether the base model was decomposed into graph-like parts
    /// (`heralded_error_model(decompose=True)`); BP+OSD refuses it.
    #[getter]
    fn decomposed(&self) -> bool {
        self.inner.is_decomposed()
    }

    /// Checked: herald detectors strictly ascending and below
    /// `base.n_detectors`; one probability (in [0, 1]) and one list of
    /// parts per herald; no base error or part flips a herald detector.
    /// The parts of each herald are sorted and equal parts merged.
    #[new]
    fn new(
        base: &PyDetectorErrorModel,
        herald_detectors: Vec<usize>,
        herald_probabilities: Vec<f64>,
        outcomes: Vec<Vec<PyMechanism>>,
    ) -> PyResult<Self> {
        let outcomes = outcomes
            .into_iter()
            .map(|parts| {
                parts
                    .into_iter()
                    .map(|(probability, mut detectors, observables)| {
                        detectors.sort_unstable();
                        ErrorMechanism {
                            probability,
                            detectors,
                            observables,
                        }
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            inner: HeraldedErrorModel::new(
                base.inner.clone(),
                herald_detectors,
                herald_probabilities,
                outcomes,
            )
            .raise()?,
        })
    }

    /// Detectors of the circuit, herald detectors included.
    #[getter]
    fn n_detectors(&self) -> usize {
        self.inner.n_detectors()
    }

    #[getter]
    fn n_observables(&self) -> usize {
        self.inner.n_observables()
    }

    /// The errors of every channel no herald detector owns.
    #[getter]
    fn base(&self) -> PyDetectorErrorModel {
        PyDetectorErrorModel {
            inner: self.inner.base().clone(),
        }
    }

    /// The herald detectors, ascending.
    #[getter]
    fn herald_detectors(&self) -> Vec<usize> {
        self.inner.herald_detectors().to_vec()
    }

    /// The probability that herald `h` fires.
    fn herald_probability(&self, h: usize) -> PyResult<f64> {
        self.check_herald(h)?;
        Ok(self.inner.herald_probability(h))
    }

    /// The parts of herald `h`'s channel given that it fired, ascending.
    fn outcomes(&self, h: usize) -> PyResult<Vec<PyMechanism>> {
        self.check_herald(h)?;
        Ok(mechanisms(self.inner.outcomes(h)))
    }

    /// The model of one shot: `base` with the parts of the fired herald
    /// detectors `heralds` (a detector listed twice cancels) merged in as
    /// independent errors.
    fn conditioned(&self, heralds: Vec<usize>) -> PyResult<PyDetectorErrorModel> {
        Ok(PyDetectorErrorModel {
            inner: self.inner.conditioned(&heralds).raise()?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "HeraldedErrorModel(detectors={}, observables={}, heralds={}, base_errors={})",
            self.inner.n_detectors(),
            self.inner.n_observables(),
            self.inner.herald_detectors().len(),
            self.inner.base().errors.len()
        )
    }
}

/// The weighted union-find decoder (Delfosse–Nickerson) of a detector
/// error model whose errors flip at most two detectors each.
#[pyclass(name = "UnionFindDecoder", module = "metatron_qso")]
pub struct PyUnionFindDecoder {
    inner: UnionFindDecoder,
}

#[pymethods]
impl PyUnionFindDecoder {
    #[new]
    fn new(model: &PyDetectorErrorModel) -> PyResult<Self> {
        Ok(Self {
            inner: UnionFindDecoder::new(MatchingGraph::from_dem(&model.inner).raise()?),
        })
    }

    #[getter]
    fn n_detectors(&self) -> usize {
        self.inner.graph().n_detectors()
    }

    /// Predicted observable flips (a bit mask) for the detectors that
    /// fired.
    fn decode(&mut self, fired: Vec<usize>) -> PyResult<u64> {
        self.inner.decode(&fired).raise()
    }

    /// Decode each row of a (shots, n_detectors) boolean array; returns
    /// the predicted observable masks.
    fn decode_batch<'py>(
        &mut self,
        py: Python<'py>,
        detectors: PyArrayLike2<'py, bool, AllowTypeChange>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let n = self.inner.graph().n_detectors();
        decode_batch(py, &mut self.inner, n, detectors)
    }

    fn __repr__(&self) -> String {
        let g = self.inner.graph();
        format!(
            "UnionFindDecoder(detectors={}, edges={})",
            g.n_detectors(),
            g.edges().len()
        )
    }
}

/// Decode each row of a (shots, n) boolean array of detection events.
fn decode_batch<'py>(
    py: Python<'py>,
    decoder: &mut (impl Decoder + Send),
    n: usize,
    detectors: PyArrayLike2<'py, bool, AllowTypeChange>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let a = detectors.as_array();
    if a.ncols() != n {
        return Err(invalid(format!(
            "expected {n} columns (one per detector), got {}",
            a.ncols()
        )));
    }
    let rows: Vec<Vec<usize>> = a
        .rows()
        .into_iter()
        .map(|r| (0..n).filter(|&d| r[d]).collect())
        .collect();
    let out = py
        .detach(|| {
            rows.iter()
                .map(|f| {
                    if f.is_empty() {
                        Ok(0)
                    } else {
                        decoder.decode(f)
                    }
                })
                .collect::<metatron_qso::Result<Vec<u64>>>()
        })
        .raise()?;
    Ok(out.into_pyarray(py))
}

/// The exact minimum-weight perfect matching decoder of a detector error
/// model whose errors flip at most two detectors each. Distances from
/// each detector are cached when it first fires (up to 32 MiB).
#[pyclass(name = "MatchingDecoder", module = "metatron_qso")]
pub struct PyMatchingDecoder {
    inner: MatchingDecoder,
}

#[pymethods]
impl PyMatchingDecoder {
    #[new]
    fn new(py: Python<'_>, model: &PyDetectorErrorModel) -> PyResult<Self> {
        let graph = MatchingGraph::from_dem(&model.inner).raise()?;
        Ok(Self {
            inner: py.detach(|| MatchingDecoder::new(graph)),
        })
    }

    #[getter]
    fn n_detectors(&self) -> usize {
        self.inner.graph().n_detectors()
    }

    /// Predicted observable flips (a bit mask) for the detectors that
    /// fired.
    fn decode(&mut self, fired: Vec<usize>) -> PyResult<u64> {
        self.inner.decode(&fired).raise()
    }

    /// Predicted observable flips and the weight of the correction,
    /// the sum of ln((1 − p)/p) over its errors.
    fn decode_with_weight(&mut self, fired: Vec<usize>) -> PyResult<(u64, f64)> {
        self.inner.decode_with_weight(&fired).raise()
    }

    /// Decode each row of a (shots, n_detectors) boolean array; returns
    /// the predicted observable masks.
    fn decode_batch<'py>(
        &mut self,
        py: Python<'py>,
        detectors: PyArrayLike2<'py, bool, AllowTypeChange>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let n = self.inner.graph().n_detectors();
        decode_batch(py, &mut self.inner, n, detectors)
    }

    fn __repr__(&self) -> String {
        let g = self.inner.graph();
        format!(
            "MatchingDecoder(detectors={}, edges={})",
            g.n_detectors(),
            g.edges().len()
        )
    }
}

/// Minimum-weight perfect matching by growing regions on the detector
/// graph (sparse blossom): the same weights as `MatchingDecoder`, faster
/// for large codes.
#[pyclass(name = "SparseMatchingDecoder", module = "metatron_qso")]
pub struct PySparseMatchingDecoder {
    inner: SparseMatchingDecoder,
}

#[pymethods]
impl PySparseMatchingDecoder {
    #[new]
    fn new(py: Python<'_>, model: &PyDetectorErrorModel) -> PyResult<Self> {
        let graph = MatchingGraph::from_dem(&model.inner).raise()?;
        Ok(Self {
            inner: py.detach(|| SparseMatchingDecoder::new(graph)),
        })
    }

    #[getter]
    fn n_detectors(&self) -> usize {
        self.inner.graph().n_detectors()
    }

    /// Predicted observable flips (a bit mask) for the detectors that
    /// fired.
    fn decode(&mut self, fired: Vec<usize>) -> PyResult<u64> {
        self.inner.decode(&fired).raise()
    }

    /// Predicted observable flips and the weight of the correction,
    /// the sum of ln((1 − p)/p) over its errors.
    fn decode_with_weight(&mut self, fired: Vec<usize>) -> PyResult<(u64, f64)> {
        self.inner.decode_with_weight(&fired).raise()
    }

    /// Decode each row of a (shots, n_detectors) boolean array; returns
    /// the predicted observable masks.
    fn decode_batch<'py>(
        &mut self,
        py: Python<'py>,
        detectors: PyArrayLike2<'py, bool, AllowTypeChange>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let n = self.inner.graph().n_detectors();
        decode_batch(py, &mut self.inner, n, detectors)
    }

    fn __repr__(&self) -> String {
        let g = self.inner.graph();
        format!(
            "SparseMatchingDecoder(detectors={}, edges={})",
            g.n_detectors(),
            g.edges().len()
        )
    }
}

/// A decomposed detector error model that keeps the parts of each error:
/// `errors` lists `(probability, [(detectors, observables), ...])`, every
/// part flipping one or two detectors (or none).
#[pyclass(name = "CorrelatedErrorModel", module = "metatron_qso", frozen)]
pub struct PyCorrelatedErrorModel {
    inner: CorrelatedErrorModel,
}

type PyCorrelatedError = (f64, Vec<(Vec<usize>, u64)>);

#[pymethods]
impl PyCorrelatedErrorModel {
    /// `n_observables` defaults to one more than the highest observable
    /// bit used.
    #[new]
    #[pyo3(signature = (n_detectors, errors, n_observables = None))]
    fn new(
        n_detectors: usize,
        errors: Vec<PyCorrelatedError>,
        n_observables: Option<usize>,
    ) -> PyResult<Self> {
        if n_detectors > qec::MAX_MODEL_DETECTORS {
            return Err(invalid(format!(
                "at most {} detectors",
                qec::MAX_MODEL_DETECTORS
            )));
        }
        if n_observables.is_some_and(|o| o > qec::MAX_OBSERVABLES) {
            return Err(invalid(format!(
                "at most {} observables",
                qec::MAX_OBSERVABLES
            )));
        }
        let mut out = Vec::with_capacity(errors.len());
        let mut used = 0;
        for (k, (probability, parts)) in errors.into_iter().enumerate() {
            if !(0.0..=1.0).contains(&probability) {
                return Err(invalid(format!(
                    "error {k}: {probability} is not a probability"
                )));
            }
            let mut checked = Vec::with_capacity(parts.len());
            for (mut detectors, observables) in parts {
                detectors.sort_unstable();
                if detectors.len() > 2 || detectors.windows(2).any(|w| w[0] == w[1]) {
                    return Err(invalid(format!(
                        "a part of error {k} must flip at most two distinct detectors"
                    )));
                }
                if let Some(&d) = detectors.last().filter(|&&d| d >= n_detectors) {
                    return Err(invalid(format!(
                        "error {k} flips detector {d} of {n_detectors}"
                    )));
                }
                used = used.max(64 - observables.leading_zeros() as usize);
                checked.push((detectors, observables));
            }
            out.push(CorrelatedError {
                probability,
                parts: checked,
            });
        }
        let n_observables = n_observables.unwrap_or(used);
        if n_observables < used {
            return Err(invalid(format!(
                "observable {} is used but n_observables is {n_observables}",
                used - 1
            )));
        }
        Ok(Self {
            inner: CorrelatedErrorModel {
                n_detectors,
                n_observables,
                errors: out,
            },
        })
    }

    #[getter]
    fn n_detectors(&self) -> usize {
        self.inner.n_detectors
    }

    #[getter]
    fn n_observables(&self) -> usize {
        self.inner.n_observables
    }

    #[getter]
    fn errors(&self) -> Vec<PyCorrelatedError> {
        self.inner
            .errors
            .iter()
            .map(|e| (e.probability, e.parts.clone()))
            .collect()
    }

    /// The decomposed model, every part an independent error.
    fn to_dem(&self) -> PyDetectorErrorModel {
        PyDetectorErrorModel {
            inner: self.inner.to_dem(),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.errors.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "CorrelatedErrorModel(detectors={}, observables={}, errors={})",
            self.inner.n_detectors,
            self.inner.n_observables,
            self.inner.errors.len()
        )
    }
}

/// Two-pass (correlated) matching: match, lower the weights of the edges
/// that the correction's errors make likely, match again (PyMatching's
/// `enable_correlations=True`).
#[pyclass(name = "CorrelatedMatchingDecoder", module = "metatron_qso")]
pub struct PyCorrelatedMatchingDecoder {
    inner: CorrelatedMatchingDecoder,
}

#[pymethods]
impl PyCorrelatedMatchingDecoder {
    #[new]
    fn new(py: Python<'_>, model: &PyCorrelatedErrorModel) -> PyResult<Self> {
        let m = &model.inner;
        Ok(Self {
            inner: py.detach(|| CorrelatedMatchingDecoder::new(m)).raise()?,
        })
    }

    #[getter]
    fn n_detectors(&self) -> usize {
        self.inner.graph().n_detectors()
    }

    /// Predicted observable flips (a bit mask) for the detectors that
    /// fired.
    fn decode(&mut self, fired: Vec<usize>) -> PyResult<u64> {
        self.inner.decode(&fired).raise()
    }

    /// Decode each row of a (shots, n_detectors) boolean array; returns
    /// the predicted observable masks.
    fn decode_batch<'py>(
        &mut self,
        py: Python<'py>,
        detectors: PyArrayLike2<'py, bool, AllowTypeChange>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let n = self.inner.graph().n_detectors();
        decode_batch(py, &mut self.inner, n, detectors)
    }

    fn __repr__(&self) -> String {
        let g = self.inner.graph();
        format!(
            "CorrelatedMatchingDecoder(detectors={}, edges={})",
            g.n_detectors(),
            g.edges().len()
        )
    }
}

/// Bounds on the circuit distance: `lower_bound` (no undetectable logical
/// error is lighter; `math.inf` if none exists), the lightest found
/// `witness` (indices into the model's errors, or None), the observables
/// it flips, and `distance` when the bounds meet.
#[pyclass(name = "CircuitDistance", module = "metatron_qso", frozen)]
pub struct PyCircuitDistance {
    inner: CircuitDistance,
}

#[pymethods]
impl PyCircuitDistance {
    /// `math.inf` when no undetectable logical error exists.
    #[getter]
    fn lower_bound(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(if self.inner.lower_bound == usize::MAX {
            f64::INFINITY.into_pyobject(py)?.into_any().unbind()
        } else {
            self.inner
                .lower_bound
                .into_pyobject(py)?
                .into_any()
                .unbind()
        })
    }

    #[getter]
    fn upper_bound(&self) -> Option<usize> {
        self.inner.upper_bound()
    }

    #[getter]
    fn distance(&self) -> Option<usize> {
        self.inner.distance()
    }

    #[getter]
    fn witness(&self) -> Option<Vec<usize>> {
        self.inner.witness.clone()
    }

    #[getter]
    fn observables(&self) -> u64 {
        self.inner.observables
    }

    fn __repr__(&self) -> String {
        let bound = |b: Option<usize>| b.map_or("None".to_string(), |b| b.to_string());
        let lower = if self.inner.lower_bound == usize::MAX {
            "inf".to_string()
        } else {
            self.inner.lower_bound.to_string()
        };
        format!(
            "CircuitDistance(lower_bound={lower}, upper_bound={}, distance={})",
            bound(self.inner.upper_bound()),
            bound(self.inner.distance())
        )
    }
}

/// Bounds on the circuit distance of a detector error model (exact when
/// every error flips at most two detectors): an exhaustive search proves
/// weights up to `max_weight` within `max_work` visited sets, and
/// `bposd_trials` BP+OSD searches per observable look for light logical
/// errors.
#[pyfunction]
#[pyo3(signature = (model, max_weight = 12, max_work = 16_777_216, bposd_trials = 4, seed = 0))]
fn circuit_distance(
    py: Python<'_>,
    model: &PyDetectorErrorModel,
    max_weight: usize,
    max_work: u64,
    bposd_trials: usize,
    seed: u64,
) -> PyResult<PyCircuitDistance> {
    let options = DistanceOptions {
        max_weight,
        max_work,
        bposd_trials,
        seed,
    };
    let m = &model.inner;
    Ok(PyCircuitDistance {
        inner: py.detach(|| qec::circuit_distance(m, &options)).raise()?,
    })
}

/// The circuit distance of a model, exactly when the time allows: the
/// lower bound rises by the exhaustive search or by levels of the
/// Brouwer–Zimmermann enumeration (`method`: "auto" takes whichever is
/// estimated to cost less at each step, or "search", or
/// "brouwer_zimmermann") until it meets the lightest logical error found;
/// after `max_seconds` (None: no limit) the bounds reached so far are
/// returned. With `symmetry` the search starts only from one error of each
/// orbit of the model's symmetries, those `error_model_symmetries` finds
/// (up to 4096 detectors; larger models are searched without them).
/// Detector permutations given as `symmetries` are used with or without
/// `symmetry`: each is checked and refused if it is not a symmetry.
#[pyfunction]
#[pyo3(
    signature = (model, max_seconds = Some(60.0), method = "auto", bposd_trials = 4, seed = 0, symmetry = false, symmetries = None),
    text_signature = "(model, max_seconds=60.0, method='auto', bposd_trials=4, seed=0, symmetry=False, symmetries=None)"
)]
#[allow(clippy::too_many_arguments)]
fn exact_circuit_distance(
    py: Python<'_>,
    model: &PyDetectorErrorModel,
    max_seconds: Option<f64>,
    method: &str,
    bposd_trials: usize,
    seed: u64,
    symmetry: bool,
    symmetries: Option<Vec<Vec<usize>>>,
) -> PyResult<PyCircuitDistance> {
    let method = match method {
        "auto" => DistanceMethod::Auto,
        "search" => DistanceMethod::Search,
        "brouwer_zimmermann" => DistanceMethod::BrouwerZimmermann,
        other => {
            return Err(invalid(format!(
                "method must be 'auto', 'search' or 'brouwer_zimmermann', not {other:?}"
            )));
        }
    };
    let options = ExactDistanceOptions::default()
        .with_max_seconds(max_seconds.unwrap_or(f64::INFINITY))
        .with_method(method)
        .with_bposd_trials(bposd_trials)
        .with_seed(seed)
        .with_symmetry(symmetry);
    let m = &model.inner;
    let declared = symmetries.unwrap_or_default();
    Ok(PyCircuitDistance {
        inner: py
            .detach(|| qec::exact_circuit_distance_with_symmetries(m, &options, &declared))
            .raise()?,
    })
}

/// Symmetries of a detector error model, as permutations of its detectors
/// (detector `d` goes to `p[d]`): the generators that QSO's automorphism
/// search finds on the graph joining each error to its detectors and that
/// map logical errors onto logical errors. They generate a group of
/// symmetries, though not always all of them (a product of generators
/// that fail is lost with them; it can be declared). Empty when two errors
/// flip the same detectors and different observables (the distance is then
/// at most 2); otherwise refused beyond 4096 detectors.
#[pyfunction]
fn error_model_symmetries(
    py: Python<'_>,
    model: &PyDetectorErrorModel,
) -> PyResult<Vec<Vec<usize>>> {
    let m = &model.inner;
    py.detach(|| qec::error_model_symmetries(m)).raise()
}

/// Where errors of the circuit's (undecomposed) detector error model come
/// from: for each error (an index into `model.errors`; all when `indices`
/// is None), the circuit faults with exactly its effect, in circuit order,
/// as dicts `{"operation": i, "paulis": [(qubit, "X"), ...],
/// "probability": p}` (`"M"` for a flipped measurement). With a distance's
/// `witness` this places the lightest logical error in the circuit.
#[pyfunction]
#[pyo3(signature = (circuit, model, indices = None))]
fn explain_errors<'py>(
    py: Python<'py>,
    circuit: &PyDetectorCircuit,
    model: &PyDetectorErrorModel,
    indices: Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyList>> {
    let all = &model.inner.errors;
    let indices = indices.unwrap_or_else(|| (0..all.len()).collect());
    if let Some(&k) = indices.iter().find(|&&k| k >= all.len()) {
        return Err(invalid(format!(
            "error {k} out of range ({} errors)",
            all.len()
        )));
    }
    let chosen: Vec<ErrorMechanism> = indices.iter().map(|&k| all[k].clone()).collect();
    let c = &circuit.inner;
    let explained = py.detach(|| qec::explain_errors(c, &chosen)).raise()?;
    let out = PyList::empty(py);
    for faults in explained {
        let list = PyList::empty(py);
        for f in faults {
            let d = PyDict::new(py);
            d.set_item("operation", f.operation)?;
            let paulis: Vec<(usize, String)> =
                f.paulis.iter().map(|&(q, c)| (q, c.to_string())).collect();
            d.set_item("paulis", paulis)?;
            d.set_item("probability", f.probability)?;
            list.append(d)?;
        }
        out.append(list)?;
    }
    Ok(out)
}

/// The lightest undetectable logical error among the model's graph-like
/// errors (those flipping one or two detectors), as Stim's
/// `shortest_graphlike_error`; exact.
#[pyfunction]
fn graphlike_distance(py: Python<'_>, model: &PyDetectorErrorModel) -> PyResult<PyCircuitDistance> {
    let m = &model.inner;
    Ok(PyCircuitDistance {
        inner: py.detach(|| qec::graphlike_distance(m)).raise()?,
    })
}

/// What one BP+OSD decoding found: the predicted observable mask, the
/// correction (indices into the model's errors), whether belief
/// propagation alone converged, its iterations, and the correction's
/// weight Σ ln((1 − p)/p).
#[pyclass(name = "BpOsdOutcome", module = "metatron_qso", frozen)]
pub struct PyBpOsdOutcome {
    #[pyo3(get)]
    observables: u64,
    #[pyo3(get)]
    errors: Vec<usize>,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    weight: f64,
}

impl From<BpOsdOutcome> for PyBpOsdOutcome {
    fn from(o: BpOsdOutcome) -> Self {
        PyBpOsdOutcome {
            observables: o.observables,
            errors: o.errors,
            converged: o.converged,
            iterations: o.iterations,
            weight: o.weight,
        }
    }
}

#[pymethods]
impl PyBpOsdOutcome {
    fn __repr__(&self) -> String {
        format!(
            "BpOsdOutcome(observables={}, errors={:?}, converged={}, iterations={}, weight={})",
            self.observables,
            self.errors,
            if self.converged { "True" } else { "False" },
            self.iterations,
            self.weight
        )
    }
}

/// Belief propagation with ordered-statistics post-processing on a
/// detector error model whose errors may flip any number of detectors.
///
/// `method` is `"min_sum"` (messages scaled by `scaling`; 0 means
/// 1 − 2^−t in iteration t) or `"product_sum"`; `max_iterations=0`
/// skips belief propagation. `osd` is `"osd0"`, `"cs"` (combination
/// sweep: every single error outside the information set and the pairs
/// among the first `osd_order`) or `"exhaustive"` (every subset of the
/// first `osd_order`, at most 20).
#[pyclass(name = "BpOsdDecoder", module = "metatron_qso")]
pub struct PyBpOsdDecoder {
    inner: BpOsdDecoder,
}

#[pymethods]
impl PyBpOsdDecoder {
    #[new]
    #[pyo3(signature = (model, max_iterations = 50, method = "min_sum", scaling = 0.625, osd = "cs", osd_order = 7))]
    fn new(
        py: Python<'_>,
        model: &PyDetectorErrorModel,
        max_iterations: usize,
        method: &str,
        scaling: f64,
        osd: &str,
        osd_order: usize,
    ) -> PyResult<Self> {
        let options = bposd_options(max_iterations, method, scaling, osd, osd_order)?;
        let dem = &model.inner;
        Ok(Self {
            inner: py.detach(|| BpOsdDecoder::new(dem, options)).raise()?,
        })
    }

    #[getter]
    fn n_detectors(&self) -> usize {
        self.inner.n_detectors()
    }

    /// Predicted observable flips (a bit mask) for the detectors that
    /// fired.
    fn decode(&mut self, py: Python<'_>, fired: Vec<usize>) -> PyResult<u64> {
        let d = &mut self.inner;
        py.detach(|| d.decode(&fired)).raise()
    }

    /// The correction and how it was found (a `BpOsdOutcome`).
    fn decode_detailed(&mut self, py: Python<'_>, fired: Vec<usize>) -> PyResult<PyBpOsdOutcome> {
        let d = &mut self.inner;
        Ok(py.detach(|| d.decode_detailed(&fired)).raise()?.into())
    }

    /// Decode each row of a (shots, n_detectors) boolean array; returns
    /// the predicted observable masks.
    fn decode_batch<'py>(
        &mut self,
        py: Python<'py>,
        detectors: PyArrayLike2<'py, bool, AllowTypeChange>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let n = self.inner.n_detectors();
        decode_batch(py, &mut self.inner, n, detectors)
    }

    fn __repr__(&self) -> String {
        let o = self.inner.options();
        format!(
            "BpOsdDecoder(detectors={}, max_iterations={}, method={:?}, osd={:?}, osd_order={})",
            self.inner.n_detectors(),
            o.max_iterations,
            o.method,
            o.osd,
            o.osd_order
        )
    }
}

/// BP+OSD options from their Python names.
fn bposd_options(
    max_iterations: usize,
    method: &str,
    scaling: f64,
    osd: &str,
    osd_order: usize,
) -> PyResult<BpOsdOptions> {
    let method = match method {
        "min_sum" => BpMethod::MinSum,
        "product_sum" => BpMethod::ProductSum,
        other => {
            return Err(invalid(format!(
                "unknown method {other:?}; use \"min_sum\" or \"product_sum\""
            )));
        }
    };
    let osd = match osd {
        "osd0" => OsdMethod::Zero,
        "cs" => OsdMethod::CombinationSweep,
        "exhaustive" => OsdMethod::Exhaustive,
        other => {
            return Err(invalid(format!(
                "unknown osd {other:?}; use \"osd0\", \"cs\" or \"exhaustive\""
            )));
        }
    };
    Ok(BpOsdOptions {
        max_iterations,
        method,
        scaling,
        osd,
        osd_order,
    })
}

/// A decoder that conditions a `HeraldedErrorModel` on the heralds that
/// fired, shot by shot: the fired heralds' parts are merged into the
/// model's errors as independent errors (an erased part has probability
/// ½), the ordinary detectors are decoded with those probabilities, and
/// the model is restored. `decoder` is `"union_find"` (erased edges joined
/// before growing, after Delfosse and Nickerson), `"matching"` (sparse
/// minimum-weight matching) or `"bposd"` (BP+OSD with `max_iterations`,
/// `bp_method`, `scaling`, `osd` and `osd_order` as for `BpOsdDecoder`;
/// the other decoders refuse values other than the defaults). Union-find
/// and matching need every error and part to flip at most two ordinary
/// detectors (the decomposed model, `heralded_error_model()`); give
/// BP+OSD the model without decomposition,
/// `heralded_error_model(decompose=False)`, which colour codes need.
///
/// `decode(fired, heralds)` takes the fired ordinary detectors and the
/// fired herald detectors; with `heralds=None`, `fired` lists every fired
/// detector and is split. An error the shot makes more likely than not
/// (p > ½, such as a Pauli channel's X part (p_x + p_y)/t) is applied
/// before decoding and the rest decoded with 1 − p, as PyMatching does
/// with negative weights; a shot with no ordinary detector left to explain
/// predicts the flips of the applied errors.
#[pyclass(name = "ConditionedDecoder", module = "metatron_qso")]
pub struct PyConditionedDecoder {
    inner: ConditionedDecoder,
}

impl PyConditionedDecoder {
    /// The fired ordinary and herald detectors: as given, or `fired`
    /// split when `heralds` is None.
    fn shot(&self, fired: Vec<usize>, heralds: Option<Vec<usize>>) -> (Vec<usize>, Vec<usize>) {
        match heralds {
            Some(h) => (fired, h),
            None => {
                let hd = self.inner.herald_detectors();
                fired
                    .into_iter()
                    .partition(|d| hd.binary_search(d).is_err())
            }
        }
    }
}

/// The heralded decoding method named by `decoder`, with the BP+OSD
/// options (refused for the other methods unless they are the defaults).
fn heralded_method(
    decoder: &str,
    max_iterations: usize,
    bp_method: &str,
    scaling: f64,
    osd: &str,
    osd_order: usize,
) -> PyResult<HeraldedMethod> {
    let options = bposd_options(max_iterations, bp_method, scaling, osd, osd_order)?;
    let method = match decoder {
        "union_find" | "matching" => {
            if options != BpOsdOptions::default() {
                return Err(invalid(format!(
                    "the BP+OSD options apply to decoder=\"bposd\", not {decoder:?}"
                )));
            }
            if decoder == "matching" {
                HeraldedMethod::Matching
            } else {
                HeraldedMethod::UnionFind
            }
        }
        "bposd" => HeraldedMethod::BpOsd(options),
        other => {
            return Err(invalid(format!(
                "unknown decoder {other:?}; use \"union_find\", \"matching\" or \"bposd\""
            )));
        }
    };
    Ok(method)
}

#[pymethods]
impl PyConditionedDecoder {
    #[new]
    #[pyo3(signature = (model, decoder = "union_find", max_iterations = 50, bp_method = "min_sum", scaling = 0.625, osd = "cs", osd_order = 7))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        model: &PyHeraldedErrorModel,
        decoder: &str,
        max_iterations: usize,
        bp_method: &str,
        scaling: f64,
        osd: &str,
        osd_order: usize,
    ) -> PyResult<Self> {
        let method = heralded_method(decoder, max_iterations, bp_method, scaling, osd, osd_order)?;
        let m = &model.inner;
        Ok(Self {
            inner: py.detach(|| ConditionedDecoder::new(m, method)).raise()?,
        })
    }

    /// The decoder of a `DetectorCircuit` or `RepeatedCircuit` on the
    /// heralded model its method needs: decomposed for `"union_find"` and
    /// `"matching"`, without decomposition for `"bposd"`.
    #[staticmethod]
    #[pyo3(signature = (circuit, decoder = "union_find", max_iterations = 50, bp_method = "min_sum", scaling = 0.625, osd = "cs", osd_order = 7))]
    #[allow(clippy::too_many_arguments)]
    fn for_circuit(
        py: Python<'_>,
        circuit: &Bound<'_, PyAny>,
        decoder: &str,
        max_iterations: usize,
        bp_method: &str,
        scaling: f64,
        osd: &str,
        osd_order: usize,
    ) -> PyResult<Self> {
        let method = heralded_method(decoder, max_iterations, bp_method, scaling, osd, osd_order)?;
        if let Ok(c) = circuit.cast::<PyDetectorCircuit>() {
            let c = c.borrow().inner.clone();
            return Ok(Self {
                inner: py
                    .detach(|| ConditionedDecoder::for_circuit(&c, method))
                    .raise()?,
            });
        }
        if let Ok(c) = circuit.cast::<PyRepeatedCircuit>() {
            let c = c.borrow().inner.clone();
            return Ok(Self {
                inner: py
                    .detach(|| ConditionedDecoder::for_repeated(&c, method))
                    .raise()?,
            });
        }
        Err(invalid(
            "circuit must be a DetectorCircuit or a RepeatedCircuit",
        ))
    }
    /// Detectors of the circuit, herald detectors included.
    #[getter]
    fn n_detectors(&self) -> usize {
        self.inner.n_detectors()
    }

    /// The herald detectors, ascending.
    #[getter]
    fn herald_detectors(&self) -> Vec<usize> {
        self.inner.herald_detectors().to_vec()
    }

    /// The slots (edges or columns) in decoding order as
    /// `(probability, detectors, observables)`, with their base
    /// probabilities (0 for a slot only heralds turn on).
    fn slots(&self) -> Vec<PyMechanism> {
        mechanisms(self.inner.slots())
    }

    /// Predicted observable flips (a bit mask) for the fired ordinary
    /// detectors `fired` given the fired herald detectors `heralds`, or,
    /// with `heralds=None`, for every fired detector `fired`.
    #[pyo3(signature = (fired, heralds = None))]
    fn decode(
        &mut self,
        py: Python<'_>,
        fired: Vec<usize>,
        heralds: Option<Vec<usize>>,
    ) -> PyResult<u64> {
        let (fired, heralds) = self.shot(fired, heralds);
        let d = &mut self.inner;
        py.detach(|| d.decode_heralded(&fired, &heralds)).raise()
    }

    /// Matching only: the prediction and the weight of the correction,
    /// the sum of ln((1 − p)/p) over its errors at the shot's
    /// probabilities (errors at p = ½ count 0, applied errors above ½
    /// negatively).
    #[pyo3(signature = (fired, heralds = None))]
    fn decode_with_weight(
        &mut self,
        py: Python<'_>,
        fired: Vec<usize>,
        heralds: Option<Vec<usize>>,
    ) -> PyResult<(u64, f64)> {
        let (fired, heralds) = self.shot(fired, heralds);
        let d = &mut self.inner;
        py.detach(|| d.decode_with_weight(&fired, &heralds)).raise()
    }

    /// BP+OSD only: the correction and how it was found; its `errors`
    /// index `slots()`, the applied errors (p > ½) included unless BP+OSD
    /// undid them.
    #[pyo3(signature = (fired, heralds = None))]
    fn decode_detailed(
        &mut self,
        py: Python<'_>,
        fired: Vec<usize>,
        heralds: Option<Vec<usize>>,
    ) -> PyResult<PyBpOsdOutcome> {
        let (fired, heralds) = self.shot(fired, heralds);
        let d = &mut self.inner;
        Ok(py
            .detach(|| d.decode_detailed(&fired, &heralds))
            .raise()?
            .into())
    }

    /// Decode each row of a (shots, n_detectors) boolean array of every
    /// detector, herald detectors included; returns the predicted
    /// observable masks.
    fn decode_batch<'py>(
        &mut self,
        py: Python<'py>,
        detectors: PyArrayLike2<'py, bool, AllowTypeChange>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let n = self.inner.n_detectors();
        decode_batch(py, &mut self.inner, n, detectors)
    }

    fn __repr__(&self) -> String {
        let method = match self.inner.method() {
            HeraldedMethod::UnionFind => "union_find",
            HeraldedMethod::Matching => "matching",
            HeraldedMethod::BpOsd(_) => "bposd",
            _ => "other",
        };
        format!(
            "ConditionedDecoder(detectors={}, heralds={}, slots={}, decoder={method:?})",
            self.inner.n_detectors(),
            self.inner.herald_detectors().len(),
            self.inner.slots().len()
        )
    }
}

/// Decoding in sliding windows of layers (rounds): a window of `window`
/// layers commits the corrections of its first `commit` (default: half,
/// rounded up) and carries the rest of the syndrome forward.
/// `layers[d]` is the layer of detector d (`DetectorCircuit.detector_layers`),
/// non-decreasing. `decoder` is `"matching"` (sparse minimum-weight
/// matching, on a decomposed model) or `"bposd"` (default options).
/// Windows of the same shape share one decoder. `push_layer` gives the
/// detection events layer by layer and returns the observables committed
/// so far; `finish` returns the prediction and starts a new stream,
/// `reset` starts one without it. A window that fails to decode voids the
/// stream until `reset`.
#[pyclass(name = "WindowDecoder", module = "metatron_qso")]
pub struct PyWindowDecoder {
    inner: WindowDecoder,
    state: StreamState,
}

#[pymethods]
impl PyWindowDecoder {
    #[new]
    #[pyo3(signature = (model, layers, window, commit = None, decoder = "matching"))]
    fn new(
        py: Python<'_>,
        model: &PyDetectorErrorModel,
        layers: Vec<usize>,
        window: usize,
        commit: Option<usize>,
        decoder: &str,
    ) -> PyResult<Self> {
        let method = window_method(decoder)?;
        let options = WindowOptions {
            window,
            commit: commit.unwrap_or(WindowOptions::new(window).commit),
        };
        let dem = &model.inner;
        let inner = py
            .detach(|| WindowDecoder::new(dem, &layers, options, method))
            .raise()?;
        let state = inner.new_stream();
        Ok(Self { inner, state })
    }

    /// Parallel windows (Skoric et al.): a first pass of windows of
    /// `buffer + commit + buffer` layers (`buffer` defaults to `commit`),
    /// each decoded on the detection events as given with both ends as
    /// boundaries and committing its middle `commit` layers, then the seams
    /// between them; each pass runs on threads. A stream (`push_layer`)
    /// decodes the same windows in order with the same prediction.
    #[staticmethod]
    #[pyo3(signature = (model, layers, commit, buffer = None, decoder = "matching"))]
    fn parallel(
        py: Python<'_>,
        model: &PyDetectorErrorModel,
        layers: Vec<usize>,
        commit: usize,
        buffer: Option<usize>,
        decoder: &str,
    ) -> PyResult<Self> {
        let method = window_method(decoder)?;
        let options = ParallelWindowOptions::new(commit, buffer.unwrap_or(commit));
        let dem = &model.inner;
        let inner = py
            .detach(|| WindowDecoder::parallel(dem, &layers, options, method))
            .raise()?;
        let state = inner.new_stream();
        Ok(Self { inner, state })
    }

    #[getter]
    fn n_detectors(&self) -> usize {
        self.inner.n_detectors()
    }

    #[getter]
    fn n_layers(&self) -> usize {
        self.inner.n_layers()
    }

    #[getter]
    fn n_windows(&self) -> usize {
        self.inner.n_windows()
    }

    /// Distinct decoders the windows share.
    #[getter]
    fn n_decoders(&self) -> usize {
        self.inner.n_decoders()
    }

    /// Predicted observable flips (a bit mask) for the detectors that
    /// fired, window by window.
    fn decode(&mut self, py: Python<'_>, fired: Vec<usize>) -> PyResult<u64> {
        let d = &mut self.inner;
        py.detach(|| d.decode(&fired)).raise()
    }

    /// Decode each row of a (shots, n_detectors) boolean array; returns
    /// the predicted observable masks.
    fn decode_batch<'py>(
        &mut self,
        py: Python<'py>,
        detectors: PyArrayLike2<'py, bool, AllowTypeChange>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let n = self.inner.n_detectors();
        decode_batch(py, &mut self.inner, n, detectors)
    }

    /// The detection events of the stream's next layer; returns the
    /// observable flips committed so far.
    fn push_layer(&mut self, py: Python<'_>, fired: Vec<usize>) -> PyResult<u64> {
        let (d, s) = (&mut self.inner, &mut self.state);
        py.detach(|| d.push_layer(s, &fired)).raise()
    }

    /// The observable flips the stream committed so far.
    #[getter]
    fn committed(&self) -> u64 {
        self.state.committed()
    }

    /// The prediction of the stream once every layer was given; the next
    /// `push_layer` starts a new stream.
    fn finish(&mut self) -> PyResult<u64> {
        let r = self.inner.finish(&self.state).raise()?;
        self.state = self.inner.new_stream();
        Ok(r)
    }

    /// Drop the stream (after an error, or to start again).
    fn reset(&mut self) {
        self.state = self.inner.new_stream();
    }

    fn __repr__(&self) -> String {
        if let Some(p) = self.inner.parallel_options() {
            return format!(
                "WindowDecoder.parallel(detectors={}, layers={}, commit={}, buffer={}, windows={}, decoders={})",
                self.inner.n_detectors(),
                self.inner.n_layers(),
                p.commit,
                p.buffer,
                self.inner.n_windows(),
                self.inner.n_decoders()
            );
        }
        let o = self.inner.options();
        format!(
            "WindowDecoder(detectors={}, layers={}, window={}, commit={}, windows={}, decoders={})",
            self.inner.n_detectors(),
            self.inner.n_layers(),
            o.window,
            o.commit,
            self.inner.n_windows(),
            self.inner.n_decoders()
        )
    }
}

fn window_method(decoder: &str) -> PyResult<WindowMethod> {
    match decoder {
        "matching" => Ok(WindowMethod::Matching),
        "bposd" => Ok(WindowMethod::BpOsd(BpOsdOptions::default())),
        other => Err(invalid(format!(
            "unknown decoder {other:?}; use \"matching\" or \"bposd\""
        ))),
    }
}

/// A CSS code: X-type and Z-type checks (lists of qubits) on `n` qubits,
/// with a basis of logical operators paired so that X logical i and Z
/// logical j overlap oddly exactly when i = j.
#[pyclass(name = "CssCode", module = "metatron_qso", frozen)]
pub struct PyCssCode {
    inner: CssCode,
}

#[pymethods]
impl PyCssCode {
    #[new]
    fn new(n: usize, x_checks: Vec<Vec<usize>>, z_checks: Vec<Vec<usize>>) -> PyResult<Self> {
        Ok(Self {
            inner: CssCode::new(n, x_checks, z_checks).raise()?,
        })
    }

    /// The triangular 6.6.6 colour code of odd `distance`: (3d² + 1)/4
    /// qubits, one logical qubit, the same faces as X and Z checks.
    #[staticmethod]
    fn color_code(distance: usize) -> PyResult<Self> {
        Ok(Self {
            inner: CssCode::color_code(distance).raise()?,
        })
    }

    /// The bivariate bicycle code with `x = S_l ⊗ I_m`, `y = I_l ⊗ S_m`,
    /// `A = Σ x^a y^b` over the pairs `a`, `B` over `b`:
    /// `H_X = [A | B]`, `H_Z = [Bᵀ | Aᵀ]`.
    #[staticmethod]
    fn bivariate_bicycle(
        l: usize,
        m: usize,
        a: Vec<(usize, usize)>,
        b: Vec<(usize, usize)>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CssCode::bivariate_bicycle(l, m, &a, &b).raise()?,
        })
    }

    /// One of the bivariate bicycle codes of Bravyi et al. (2024) by its
    /// qubits (72, 90, 108, 144 or 288); returns the code and its distance
    /// from the paper.
    #[staticmethod]
    fn bivariate_bicycle_code(n: usize) -> PyResult<(Self, usize)> {
        let (inner, d) = CssCode::bivariate_bicycle_code(n).raise()?;
        Ok((Self { inner }, d))
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    #[getter]
    fn x_checks(&self) -> Vec<Vec<usize>> {
        self.inner.x_checks().to_vec()
    }

    #[getter]
    fn z_checks(&self) -> Vec<Vec<usize>> {
        self.inner.z_checks().to_vec()
    }

    #[getter]
    fn x_logicals(&self) -> Vec<Vec<usize>> {
        self.inner.x_logicals().to_vec()
    }

    #[getter]
    fn z_logicals(&self) -> Vec<Vec<usize>> {
        self.inner.z_logicals().to_vec()
    }

    fn __repr__(&self) -> String {
        format!(
            "CssCode(n={}, k={}, x_checks={}, z_checks={})",
            self.inner.n(),
            self.inner.k(),
            self.inner.x_checks().len(),
            self.inner.z_checks().len()
        )
    }
}

fn parse_basis(basis: &str) -> PyResult<MemoryBasis> {
    match basis {
        "z" | "Z" => Ok(MemoryBasis::Z),
        "x" | "X" => Ok(MemoryBasis::X),
        other => Err(invalid(format!(
            "unknown basis {other:?}; use \"z\" or \"x\""
        ))),
    }
}

/// The order of CNOTs: a name, or a colour code's order of directions.
#[derive(FromPyObject)]
enum ScheduleArg {
    Name(String),
    Directions(Vec<u8>),
}

/// A memory experiment for a CSS code; detectors are the checks of `basis`
/// compared between rounds, observable i is logical i of that basis.
/// `schedule`: `"greedy"` (every X check, then every Z check, in greedy
/// CNOT layers), `"bicycle"` (the depth-8 cycle of Bravyi et al. for a
/// bivariate bicycle code with three monomials per polynomial), or, for a
/// colour code, the order in which every face visits its six directions
/// (a permutation of 0..6, as a list or as "0,1,2,3,4,5"); `noise` as for
/// `repetition_code_memory`.
#[pyfunction]
#[pyo3(signature = (code, rounds, noise = None, basis = "z", schedule = None))]
fn css_code_memory(
    code: &PyCssCode,
    rounds: usize,
    noise: Option<NoiseArg>,
    basis: &str,
    schedule: Option<ScheduleArg>,
) -> PyResult<PyDetectorCircuit> {
    let basis = parse_basis(basis)?;
    let schedule = parse_schedule(schedule)?;
    let c = &code.inner;
    Ok(PyDetectorCircuit {
        inner: match MemoryNoise::of(noise) {
            MemoryNoise::Plain(n) => qec::css_code_memory_with(c, rounds, &n, basis, schedule),
            MemoryNoise::Erasure(e) => qec::css_code_memory_erasure(c, rounds, &e, basis, schedule),
        }
        .raise()?,
    })
}

/// `css_code_memory` with rounds 2 to `rounds` as one repeated block, for
/// up to 10⁷ rounds; expanded it is the same circuit.
#[pyfunction]
#[pyo3(signature = (code, rounds, noise = None, basis = "z", schedule = None))]
fn css_code_memory_repeated(
    code: &PyCssCode,
    rounds: usize,
    noise: Option<NoiseArg>,
    basis: &str,
    schedule: Option<ScheduleArg>,
) -> PyResult<PyRepeatedCircuit> {
    let basis = parse_basis(basis)?;
    let schedule = parse_schedule(schedule)?;
    let c = &code.inner;
    Ok(PyRepeatedCircuit {
        inner: match MemoryNoise::of(noise) {
            MemoryNoise::Plain(n) => qec::css_code_memory_repeated(c, rounds, &n, basis, schedule),
            MemoryNoise::Erasure(e) => {
                qec::css_code_memory_erasure_repeated(c, rounds, &e, basis, schedule)
            }
        }
        .raise()?,
    })
}

fn parse_schedule(schedule: Option<ScheduleArg>) -> PyResult<CssSchedule> {
    Ok(match schedule {
        None => CssSchedule::Greedy,
        Some(ScheduleArg::Name(name)) => match name.as_str() {
            "greedy" => CssSchedule::Greedy,
            "bicycle" => CssSchedule::Bicycle,
            other => {
                // Six directions as the command line writes them: "0,1,2,3,4,5".
                let order: Option<Vec<u8>> =
                    other.split(',').map(|x| x.trim().parse().ok()).collect();
                match order.map(<[u8; 6]>::try_from) {
                    Some(Ok(order)) => CssSchedule::Directions(order),
                    _ => {
                        return Err(invalid(format!(
                            "unknown schedule {other:?}; use \"greedy\", \"bicycle\" or six directions"
                        )));
                    }
                }
            }
        },
        Some(ScheduleArg::Directions(order)) => {
            let order: [u8; 6] = order
                .try_into()
                .map_err(|_| invalid("a schedule by directions lists six directions"))?;
            CssSchedule::Directions(order)
        }
    })
}

/// Logical failures of a decoder on sampled shots.
#[pyclass(name = "LogicalErrorRate", module = "metatron_qso", frozen, eq)]
#[derive(Clone, PartialEq)]
pub struct PyLogicalErrorRate {
    #[pyo3(get)]
    shots: usize,
    /// Shots whose observable flips the decoder predicted wrongly.
    #[pyo3(get)]
    errors: usize,
    #[pyo3(get)]
    rate: f64,
    /// Binomial standard error of `rate`.
    #[pyo3(get)]
    standard_error: f64,
}

#[pymethods]
impl PyLogicalErrorRate {
    fn __repr__(&self) -> String {
        format!(
            "LogicalErrorRate(shots={}, errors={}, rate={}, standard_error={})",
            self.shots, self.errors, self.rate, self.standard_error
        )
    }
}

/// A repetition-code memory experiment: `distance` data qubits and
/// `distance − 1` ancillas measuring neighbouring ZZ parities for
/// `rounds` rounds; `noise` is a `CircuitNoise`, one probability for
/// every kind of noise, or an `ErasureNoise` (whose heralds become
/// detectors of their rounds).
#[pyfunction]
#[pyo3(signature = (distance, rounds, noise = None))]
fn repetition_code_memory(
    distance: usize,
    rounds: usize,
    noise: Option<NoiseArg>,
) -> PyResult<PyDetectorCircuit> {
    Ok(PyDetectorCircuit {
        inner: match MemoryNoise::of(noise) {
            MemoryNoise::Plain(n) => qec::repetition_code_memory(distance, rounds, &n),
            MemoryNoise::Erasure(e) => qec::repetition_code_memory_erasure(distance, rounds, &e),
        }
        .raise()?,
    })
}

/// `repetition_code_memory` with rounds 2 to `rounds` as one repeated
/// block, for up to 10⁷ rounds; expanded it is the same circuit.
#[pyfunction]
#[pyo3(signature = (distance, rounds, noise = None))]
fn repetition_code_memory_repeated(
    distance: usize,
    rounds: usize,
    noise: Option<NoiseArg>,
) -> PyResult<PyRepeatedCircuit> {
    Ok(PyRepeatedCircuit {
        inner: match MemoryNoise::of(noise) {
            MemoryNoise::Plain(n) => qec::repetition_code_memory_repeated(distance, rounds, &n),
            MemoryNoise::Erasure(e) => {
                qec::repetition_code_memory_erasure_repeated(distance, rounds, &e)
            }
        }
        .raise()?,
    })
}

/// `surface_code_memory` with rounds 2 to `rounds` as one repeated block,
/// for up to 10⁷ rounds; expanded it is the same circuit.
#[pyfunction]
#[pyo3(signature = (distance, rounds, noise = None, basis = "z"))]
fn surface_code_memory_repeated(
    distance: usize,
    rounds: usize,
    noise: Option<NoiseArg>,
    basis: &str,
) -> PyResult<PyRepeatedCircuit> {
    let basis = parse_basis(basis)?;
    Ok(PyRepeatedCircuit {
        inner: match MemoryNoise::of(noise) {
            MemoryNoise::Plain(n) => qec::surface_code_memory_repeated(distance, rounds, &n, basis),
            MemoryNoise::Erasure(e) => {
                qec::surface_code_memory_erasure_repeated(distance, rounds, &e, basis)
            }
        }
        .raise()?,
    })
}

/// A rotated surface-code memory experiment of odd `distance` protecting
/// `|0⟩_L` (`basis="z"`: Z-type detectors, the logical Z as observable 0)
/// or `|+⟩_L` (`basis="x"`: X-type detectors, the logical X); `noise` as
/// for `repetition_code_memory`.
#[pyfunction]
#[pyo3(signature = (distance, rounds, noise = None, basis = "z"))]
fn surface_code_memory(
    distance: usize,
    rounds: usize,
    noise: Option<NoiseArg>,
    basis: &str,
) -> PyResult<PyDetectorCircuit> {
    let basis = parse_basis(basis)?;
    Ok(PyDetectorCircuit {
        inner: match MemoryNoise::of(noise) {
            MemoryNoise::Plain(n) => qec::surface_code_memory_with(distance, rounds, &n, basis),
            MemoryNoise::Erasure(e) => {
                qec::surface_code_memory_erasure(distance, rounds, &e, basis)
            }
        }
        .raise()?,
    })
}

/// Sample `shots` shots of `circuit`, decode them with the decoder of its
/// decomposed detector error model (`"union_find"`, `"matching"`, or
/// `"correlated_matching"` with the parts of each error) or of its whole
/// model (`"bposd"`, default options) and count the logical failures.
/// `"window_matching"` and `"window_bposd"` decode in sliding windows of
/// `window` rounds committing `commit` (default: half, rounded up);
/// `"parallel_window_matching"` and `"parallel_window_bposd"` in parallel
/// windows committing `commit` rounds with `buffer` (default: `commit`)
/// on each side. Every decoder sees the same shots. A `RepeatedCircuit`
/// is sampled as it runs and decoded on its folded model. A circuit with
/// herald detectors (`heralded_error_model`) is decoded by a
/// `ConditionedDecoder` (`"union_find"`, `"matching"` or `"bposd"`); the
/// other decoders refuse it.
#[pyfunction]
#[pyo3(signature = (
    circuit, shots, seed = 0, decoder = "union_find", window = None, commit = None, buffer = None
))]
#[allow(clippy::too_many_arguments)]
fn logical_error_rate(
    py: Python<'_>,
    circuit: AnyCircuit<'_>,
    shots: usize,
    seed: u64,
    decoder: &str,
    window: Option<usize>,
    commit: Option<usize>,
    buffer: Option<usize>,
) -> PyResult<PyLogicalErrorRate> {
    let kind = decoder_choice(decoder, window, commit, buffer)?;
    let r = match &circuit {
        AnyCircuit::Flat(c) => {
            let c = &c.inner;
            py.detach(|| qec::logical_error_rate_using(c, shots, seed, kind))
        }
        AnyCircuit::Repeated(c) => {
            let c = &c.get().inner;
            py.detach(|| qec::logical_error_rate_repeated(c, shots, seed, kind))
        }
    }
    .raise()?;
    Ok(PyLogicalErrorRate {
        shots: r.shots,
        errors: r.errors,
        rate: r.rate,
        standard_error: r.standard_error,
    })
}

/// A flat or a repeated circuit.
#[derive(FromPyObject)]
enum AnyCircuit<'py> {
    Flat(PyRef<'py, PyDetectorCircuit>),
    Repeated(Bound<'py, PyRepeatedCircuit>),
}

/// A decoder by name.
fn decoder_choice(
    decoder: &str,
    window: Option<usize>,
    commit: Option<usize>,
    buffer: Option<usize>,
) -> PyResult<DecoderChoice> {
    let windowed = decoder.starts_with("window_");
    let parallel = decoder.starts_with("parallel_window_");
    if !windowed && !parallel && (window.is_some() || commit.is_some() || buffer.is_some()) {
        return Err(invalid(format!(
            "window, commit and buffer apply to the window decoders, not {decoder:?}"
        )));
    }
    if windowed && buffer.is_some() {
        return Err(invalid(format!(
            "{decoder:?} slides windows of `window` rounds; buffer applies to parallel windows"
        )));
    }
    if parallel && window.is_some() {
        return Err(invalid(format!(
            "{decoder:?} takes commit and buffer, not a window"
        )));
    }
    let halves = || -> PyResult<(usize, usize)> {
        let c = commit.ok_or_else(|| invalid(format!("{decoder:?} needs commit")))?;
        Ok((c, buffer.unwrap_or(c)))
    };
    let sizes = || -> PyResult<(usize, usize)> {
        let w = window.ok_or_else(|| invalid(format!("{decoder:?} needs a window")))?;
        Ok((w, commit.unwrap_or(WindowOptions::new(w).commit)))
    };
    Ok(match decoder {
        "union_find" => DecoderChoice::UnionFind,
        "matching" => DecoderChoice::Matching,
        "correlated_matching" => DecoderChoice::CorrelatedMatching,
        "bposd" => DecoderChoice::BpOsd,
        "window_matching" => {
            let (window, commit) = sizes()?;
            DecoderChoice::WindowMatching { window, commit }
        }
        "window_bposd" => {
            let (window, commit) = sizes()?;
            DecoderChoice::WindowBpOsd { window, commit }
        }
        "parallel_window_matching" => {
            let (commit, buffer) = halves()?;
            DecoderChoice::ParallelWindowMatching { commit, buffer }
        }
        "parallel_window_bposd" => {
            let (commit, buffer) = halves()?;
            DecoderChoice::ParallelWindowBpOsd { commit, buffer }
        }
        other => {
            return Err(invalid(format!(
                "unknown decoder {other:?}; use \"union_find\", \"matching\", \"correlated_matching\", \"bposd\", \"window_matching\", \"window_bposd\", \"parallel_window_matching\" or \"parallel_window_bposd\""
            )));
        }
    })
}

/// The logical failure probability per shot of the circuits
/// `circuit_at(p)` at `target`, by the splitting method of Bravyi and
/// Vargo from `start` (where it is sampled directly): Metropolis chains
/// over the failing error configurations at falling rates, the ratios of
/// consecutive failure probabilities by Bennett's acceptance ratio. A dict
/// with `rate`, `standard_error`, `log_standard_error`, `start` (the
/// direct sampling) and `rates` (per rate: `p`, `ratio`,
/// `log_ratio_error`, `acceptance`, `mean_weight`, `weight_spread`), and
/// `decodes`. `circuit_at` must give circuits with the same errors at
/// every rate.
#[pyfunction]
#[pyo3(signature = (
    circuit_at, start, target, decoder = "matching", start_shots = 100_000, chains = 16,
    steps = 20_000, burn_in = 5000, seed = 0
))]
#[allow(clippy::too_many_arguments)]
fn splitting_failure_rate<'py>(
    py: Python<'py>,
    circuit_at: Py<PyAny>,
    start: f64,
    target: f64,
    decoder: &str,
    start_shots: usize,
    chains: usize,
    steps: usize,
    burn_in: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let kind = decoder_choice(decoder, None, None, None)?;
    let options = qec::SplittingOptions {
        start_shots,
        chains,
        steps,
        burn_in,
        seed,
    };
    let at = |p: f64| -> metatron_qso::Result<DetectorCircuit> {
        Python::attach(|py| {
            py.check_signals()?;
            circuit_at
                .call1(py, (p,))
                .and_then(|c| Ok(c.extract::<PyRef<'_, PyDetectorCircuit>>(py)?.inner.clone()))
        })
        .map_err(|e| metatron_qso::QsoError::invalid("circuit_at", e.to_string()))
    };
    let r = py
        .detach(|| qec::splitting_failure_rate(&at, kind, start, target, &options))
        .raise()?;
    let rates = PyList::empty(py);
    for x in &r.rates {
        rates.append(pydict!(py, {
            "p" => x.p,
            "ratio" => x.ratio,
            "log_ratio_error" => x.log_ratio_error,
            "acceptance" => x.acceptance,
            "mean_weight" => x.mean_weight,
            "weight_spread" => x.weight_spread,
        }))?;
    }
    Ok(pydict!(py, {
        "p" => r.p,
        "rate" => r.rate,
        "standard_error" => r.standard_error,
        "log_standard_error" => r.log_standard_error,
        "start" => PyLogicalErrorRate {
            shots: r.start.shots,
            errors: r.start.errors,
            rate: r.start.rate,
            standard_error: r.start.standard_error,
        },
        "rates" => rates,
        "decodes" => r.decodes,
    }))
}

/// Every error configuration of at most `max_weight` errors of the
/// circuit's detector error model, decoded: a dict with `rate` (their
/// failure probability), `remainder` (the probability of all heavier
/// configurations, so the exact rate lies in `[rate, rate + remainder]`),
/// `failing` and `by_weight` (per weight) and `configurations`.
#[pyfunction]
#[pyo3(signature = (circuit, max_weight, decoder = "matching"))]
fn enumerated_failure_rate<'py>(
    py: Python<'py>,
    circuit: &PyDetectorCircuit,
    max_weight: usize,
    decoder: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let kind = decoder_choice(decoder, None, None, None)?;
    let c = &circuit.inner;
    let r = py
        .detach(|| qec::enumerated_failure_rate(c, kind, max_weight))
        .raise()?;
    Ok(pydict!(py, {
        "rate" => r.rate,
        "remainder" => r.remainder,
        "failing" => r.failing,
        "by_weight" => r.by_weight,
        "configurations" => r.configurations,
    }))
}

/// The leading term of the failure probability: the distance, the fewest
/// errors that fail (`weight`), the logical errors of at most `2⌈d/2⌉`
/// errors, the failing configurations of that weight (all of them when the
/// configurations of at most `⌈d/2⌉` errors number at most
/// `max_configurations`, `exhaustive`; otherwise among the subsets of the
/// logical errors, a lower bound for BP+OSD and unequal weights) and their
/// summed probability `leading_term` (`A·p^w`) and `probability` (with
/// the factors `1 − p` of the other errors: their exact share of the
/// failure probability, a lower bound), as a dict (with `work`, the sets
/// the enumeration visited, at most `max_work`).
#[pyfunction]
#[pyo3(signature = (circuit, decoder = "matching", max_work = 1_073_741_824, max_configurations = 16_777_216))]
fn leading_failures<'py>(
    py: Python<'py>,
    circuit: &PyDetectorCircuit,
    decoder: &str,
    max_work: u64,
    max_configurations: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let kind = decoder_choice(decoder, None, None, None)?;
    let c = &circuit.inner;
    let r = py
        .detach(|| qec::leading_failures(c, kind, max_work, max_configurations))
        .raise()?;
    Ok(pydict!(py, {
        "distance" => r.distance,
        "weight" => r.weight,
        "exhaustive" => r.exhaustive,
        "logicals" => r.logicals,
        "configurations" => r.configurations,
        "leading_term" => r.leading_term,
        "probability" => r.probability,
        "work" => r.work,
    }))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(splitting_failure_rate, m)?)?;
    m.add_function(wrap_pyfunction!(enumerated_failure_rate, m)?)?;
    m.add_function(wrap_pyfunction!(leading_failures, m)?)?;
    m.add_class::<PyDetectorCircuit>()?;
    m.add_class::<PyCircuitNoise>()?;
    m.add_class::<PyErasureNoise>()?;
    m.add_class::<PyHeraldedErrorModel>()?;
    m.add_class::<PyConditionedDecoder>()?;
    m.add_class::<PyDetectorErrorModel>()?;
    m.add_class::<PyUnionFindDecoder>()?;
    m.add_class::<PyMatchingDecoder>()?;
    m.add_class::<PySparseMatchingDecoder>()?;
    m.add_class::<PyCorrelatedErrorModel>()?;
    m.add_class::<PyCorrelatedMatchingDecoder>()?;
    m.add_class::<PyCircuitDistance>()?;
    m.add_function(wrap_pyfunction!(circuit_distance, m)?)?;
    m.add_function(wrap_pyfunction!(exact_circuit_distance, m)?)?;
    m.add_function(wrap_pyfunction!(error_model_symmetries, m)?)?;
    m.add_function(wrap_pyfunction!(explain_errors, m)?)?;
    m.add_function(wrap_pyfunction!(graphlike_distance, m)?)?;
    m.add_class::<PyBpOsdDecoder>()?;
    m.add_class::<PyWindowDecoder>()?;
    m.add_class::<PyBpOsdOutcome>()?;
    m.add_class::<PyCssCode>()?;
    m.add_class::<PyLogicalErrorRate>()?;
    m.add_class::<PyRepeatedCircuit>()?;
    m.add_function(wrap_pyfunction!(repetition_code_memory, m)?)?;
    m.add_function(wrap_pyfunction!(surface_code_memory, m)?)?;
    m.add_function(wrap_pyfunction!(css_code_memory, m)?)?;
    m.add_function(wrap_pyfunction!(repetition_code_memory_repeated, m)?)?;
    m.add_function(wrap_pyfunction!(surface_code_memory_repeated, m)?)?;
    m.add_function(wrap_pyfunction!(css_code_memory_repeated, m)?)?;
    m.add_function(wrap_pyfunction!(logical_error_rate, m)?)?;
    Ok(())
}
