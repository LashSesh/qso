//! Detector circuits, Pauli-frame sampling, detector error models, the
//! union-find, matching and BP+OSD decoders, CSS codes and memory
//! experiments.

use metatron_qso::qec::{
    self, BpMethod, BpOsdDecoder, BpOsdOptions, BpOsdOutcome, CircuitDistance, CircuitNoise,
    CorrelatedError, CorrelatedErrorModel, CorrelatedMatchingDecoder, CssCode, CssSchedule,
    Decoder, DecoderChoice, DetectorCircuit, DetectorErrorModel, DistanceOptions, ErrorMechanism,
    MatchingDecoder, MatchingGraph, MemoryBasis, Noise, OsdMethod, SparseMatchingDecoder,
    UnionFindDecoder,
};
use numpy::ndarray::Array2;
use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArrayLike2};
use pyo3::prelude::*;

use crate::circuit::{PyCircuit, theta_or_empty};
use crate::{Raise, RealArg, invalid};

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

/// Noise argument of the memory experiments: a `CircuitNoise` or a
/// probability for every kind of noise.
enum NoiseArg {
    Model(CircuitNoise),
    Uniform(f64),
}

impl<'a, 'py> FromPyObject<'a, 'py> for NoiseArg {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(n) = obj.cast::<PyCircuitNoise>() {
            return Ok(NoiseArg::Model(n.get().inner));
        }
        Ok(NoiseArg::Uniform(obj.extract::<f64>()?))
    }
}

impl NoiseArg {
    fn model(self) -> CircuitNoise {
        match self {
            NoiseArg::Model(m) => m,
            NoiseArg::Uniform(p) => CircuitNoise::uniform(p),
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

    fn __len__(&self) -> usize {
        self.inner.operations().len()
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
    /// depolarizing noise, detectors, observables and REPEAT blocks.
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
        let (det, obs) = py
            .detach(|| {
                let s = qec::sample_detectors(c, shots, seed)?;
                let det = Array2::from_shape_fn((shots, nd), |(k, d)| s.fired(d, k));
                let obs = Array2::from_shape_fn((shots, no), |(k, i)| {
                    (s.observable_flips(k) >> i) & 1 == 1
                });
                Ok::<_, metatron_qso::QsoError>((det, obs))
            })
            .raise()?;
        Ok((det.into_pyarray(py), obs.into_pyarray(py)))
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

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "DetectorCircuit(n_qubits={}, operations={}, detectors={}, observables={})",
            self.inner.n_qubits(),
            self.inner.operations().len(),
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
    fn errors(&self) -> Vec<(f64, Vec<usize>, u64)> {
        self.inner
            .errors
            .iter()
            .map(|e| (e.probability, e.detectors.clone(), e.observables))
            .collect()
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
        let options = BpOsdOptions {
            max_iterations,
            method,
            scaling,
            osd,
            osd_order,
        };
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
/// (a permutation of 0..6, as a list or as "0,1,2,3,4,5").
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
    let noise = noise.map(NoiseArg::model).unwrap_or_default();
    let schedule = match schedule {
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
    };
    Ok(PyDetectorCircuit {
        inner: qec::css_code_memory_with(&code.inner, rounds, &noise, basis, schedule).raise()?,
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
/// `rounds` rounds; `noise` is a `CircuitNoise` or one probability for
/// every kind of noise.
#[pyfunction]
#[pyo3(signature = (distance, rounds, noise = None))]
fn repetition_code_memory(
    distance: usize,
    rounds: usize,
    noise: Option<NoiseArg>,
) -> PyResult<PyDetectorCircuit> {
    let noise = noise.map(NoiseArg::model).unwrap_or_default();
    Ok(PyDetectorCircuit {
        inner: qec::repetition_code_memory(distance, rounds, &noise).raise()?,
    })
}

/// A rotated surface-code memory experiment of odd `distance` protecting
/// `|0⟩_L` (`basis="z"`: Z-type detectors, the logical Z as observable 0)
/// or `|+⟩_L` (`basis="x"`: X-type detectors, the logical X).
#[pyfunction]
#[pyo3(signature = (distance, rounds, noise = None, basis = "z"))]
fn surface_code_memory(
    distance: usize,
    rounds: usize,
    noise: Option<NoiseArg>,
    basis: &str,
) -> PyResult<PyDetectorCircuit> {
    let basis = parse_basis(basis)?;
    let noise = noise.map(NoiseArg::model).unwrap_or_default();
    Ok(PyDetectorCircuit {
        inner: qec::surface_code_memory_with(distance, rounds, &noise, basis).raise()?,
    })
}

/// Sample `shots` shots of `circuit`, decode them with the decoder of its
/// decomposed detector error model (`"union_find"`, `"matching"`, or
/// `"correlated_matching"` with the parts of each error) or of its whole
/// model (`"bposd"`, default options) and count the logical failures.
/// Every decoder sees the same shots.
#[pyfunction]
#[pyo3(signature = (circuit, shots, seed = 0, decoder = "union_find"))]
fn logical_error_rate(
    py: Python<'_>,
    circuit: &PyDetectorCircuit,
    shots: usize,
    seed: u64,
    decoder: &str,
) -> PyResult<PyLogicalErrorRate> {
    let kind = match decoder {
        "union_find" => DecoderChoice::UnionFind,
        "matching" => DecoderChoice::Matching,
        "correlated_matching" => DecoderChoice::CorrelatedMatching,
        "bposd" => DecoderChoice::BpOsd,
        other => {
            return Err(invalid(format!(
                "unknown decoder {other:?}; use \"union_find\", \"matching\", \"correlated_matching\" or \"bposd\""
            )));
        }
    };
    let c = &circuit.inner;
    let r = py
        .detach(|| qec::logical_error_rate_using(c, shots, seed, kind))
        .raise()?;
    Ok(PyLogicalErrorRate {
        shots: r.shots,
        errors: r.errors,
        rate: r.rate,
        standard_error: r.standard_error,
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDetectorCircuit>()?;
    m.add_class::<PyCircuitNoise>()?;
    m.add_class::<PyDetectorErrorModel>()?;
    m.add_class::<PyUnionFindDecoder>()?;
    m.add_class::<PyMatchingDecoder>()?;
    m.add_class::<PySparseMatchingDecoder>()?;
    m.add_class::<PyCorrelatedErrorModel>()?;
    m.add_class::<PyCorrelatedMatchingDecoder>()?;
    m.add_class::<PyCircuitDistance>()?;
    m.add_function(wrap_pyfunction!(circuit_distance, m)?)?;
    m.add_function(wrap_pyfunction!(graphlike_distance, m)?)?;
    m.add_class::<PyBpOsdDecoder>()?;
    m.add_class::<PyBpOsdOutcome>()?;
    m.add_class::<PyCssCode>()?;
    m.add_class::<PyLogicalErrorRate>()?;
    m.add_function(wrap_pyfunction!(repetition_code_memory, m)?)?;
    m.add_function(wrap_pyfunction!(surface_code_memory, m)?)?;
    m.add_function(wrap_pyfunction!(css_code_memory, m)?)?;
    m.add_function(wrap_pyfunction!(logical_error_rate, m)?)?;
    Ok(())
}
