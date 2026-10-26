//! Detector circuits, Pauli-frame sampling, detector error models, the
//! union-find decoder and memory experiments.

use metatron_qso::qec::{
    self, CircuitNoise, DetectorCircuit, DetectorErrorModel, MatchingGraph, Noise, UnionFindDecoder,
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
#[pyclass(name = "DetectorErrorModel", module = "metatron_qso", frozen)]
pub struct PyDetectorErrorModel {
    inner: DetectorErrorModel,
}

#[pymethods]
impl PyDetectorErrorModel {
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
        let a = detectors.as_array();
        let n = self.inner.graph().n_detectors();
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
        let decoder = &mut self.inner;
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

    fn __repr__(&self) -> String {
        let g = self.inner.graph();
        format!(
            "UnionFindDecoder(detectors={}, edges={})",
            g.n_detectors(),
            g.edges().len()
        )
    }
}

/// Logical failures of the union-find decoder on sampled shots.
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
/// `|0⟩_L`, with Z-type detectors and the logical Z as observable 0.
#[pyfunction]
#[pyo3(signature = (distance, rounds, noise = None))]
fn surface_code_memory(
    distance: usize,
    rounds: usize,
    noise: Option<NoiseArg>,
) -> PyResult<PyDetectorCircuit> {
    let noise = noise.map(NoiseArg::model).unwrap_or_default();
    Ok(PyDetectorCircuit {
        inner: qec::surface_code_memory(distance, rounds, &noise).raise()?,
    })
}

/// Sample `shots` shots of `circuit`, decode them with the union-find
/// decoder of its decomposed detector error model and count the logical
/// failures.
#[pyfunction]
#[pyo3(signature = (circuit, shots, seed = 0))]
fn logical_error_rate(
    py: Python<'_>,
    circuit: &PyDetectorCircuit,
    shots: usize,
    seed: u64,
) -> PyResult<PyLogicalErrorRate> {
    let c = &circuit.inner;
    let r = py
        .detach(|| qec::logical_error_rate(c, shots, seed))
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
    m.add_class::<PyLogicalErrorRate>()?;
    m.add_function(wrap_pyfunction!(repetition_code_memory, m)?)?;
    m.add_function(wrap_pyfunction!(surface_code_memory, m)?)?;
    m.add_function(wrap_pyfunction!(logical_error_rate, m)?)?;
    Ok(())
}
