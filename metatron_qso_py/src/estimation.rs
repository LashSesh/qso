//! Estimates with standard errors, grouped shot-based estimation and
//! classical shadows.

use metatron_qso::backend::{Backend, DensityMatrixBackend, DeviceBackend, StatevectorBackend};
use metatron_qso::estimation::{
    ClassicalShadow, Estimate, EstimatorOptions, ShotAllocation, estimate as estimate_core,
};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::circuit::{PyCircuit, theta_or_empty};
use crate::mitigation::PyReadoutCalibration;
use crate::noise::{PyDeviceModel, PyNoiseModel};
use crate::state::{PyPauliSum, PyStateVector};
use crate::{Raise, RealArg, check_finite, invalid, pydict};

/// A value with its standard error and the number of samples behind it.
#[pyclass(name = "Estimate", module = "metatron_qso", frozen, eq)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyEstimate {
    pub inner: Estimate,
}

pub(crate) fn wrap_estimate(inner: Estimate) -> PyEstimate {
    PyEstimate { inner }
}

#[pymethods]
impl PyEstimate {
    #[new]
    #[pyo3(signature = (value, std_error = 0.0, samples = 0))]
    fn new(value: f64, std_error: f64, samples: usize) -> PyResult<Self> {
        if !value.is_finite() || !(std_error >= 0.0 && std_error.is_finite()) {
            return Err(invalid(
                "value must be finite and std_error finite and non-negative",
            ));
        }
        Ok(wrap_estimate(Estimate {
            value,
            std_error,
            samples,
        }))
    }

    #[getter]
    fn value(&self) -> f64 {
        self.inner.value
    }

    #[getter]
    fn std_error(&self) -> f64 {
        self.inner.std_error
    }

    #[getter]
    fn samples(&self) -> usize {
        self.inner.samples
    }

    /// Two-sided normal confidence interval `(low, high)` at `level`.
    #[pyo3(signature = (level = 0.95))]
    fn confidence_interval(&self, level: f64) -> PyResult<(f64, f64)> {
        if !(level > 0.0 && level < 1.0) {
            return Err(invalid("level must lie in (0, 1)"));
        }
        Ok(self.inner.confidence_interval(level))
    }

    fn __float__(&self) -> f64 {
        self.inner.value
    }

    fn __repr__(&self) -> String {
        format!(
            "Estimate(value={}, std_error={}, samples={})",
            self.inner.value, self.inner.std_error, self.inner.samples
        )
    }
}

/// The backend selected by the common `noise` / `device` / `trajectories`
/// arguments: exact state vectors, a noise model, or a device simulated
/// with density matrices (or with at most `trajectories` trajectories per
/// run, the shots spread over them).
pub(crate) fn backend(
    noise: Option<&PyNoiseModel>,
    device: Option<&PyDeviceModel>,
    trajectories: Option<usize>,
) -> PyResult<Box<dyn Backend>> {
    match (noise, device, trajectories) {
        (Some(_), Some(_), _) => Err(invalid("give either `noise` or `device`, not both")),
        (_, None, Some(_)) => Err(invalid("`trajectories` needs a `device`")),
        (Some(n), None, None) => Ok(Box::new(DensityMatrixBackend::new(n.inner.clone()))),
        (None, Some(d), None) => Ok(Box::new(
            DeviceBackend::density_matrix(d.inner.clone()).raise()?,
        )),
        (None, Some(d), Some(t)) => Ok(Box::new(
            DeviceBackend::trajectories(d.inner.clone(), t).raise()?,
        )),
        (None, None, _) => Ok(Box::new(StatevectorBackend)),
    }
}

fn allocation(name: &str, pilot: f64) -> PyResult<ShotAllocation> {
    match name {
        "uniform" => Ok(ShotAllocation::Uniform),
        "weighted" => Ok(ShotAllocation::Weighted),
        "adaptive" => {
            if !(pilot > 0.0 && pilot < 1.0) {
                return Err(invalid("pilot must lie in (0, 1)"));
            }
            Ok(ShotAllocation::Adaptive { pilot })
        }
        other => Err(invalid(format!(
            "allocation must be 'uniform', 'weighted' or 'adaptive', got '{other}'"
        ))),
    }
}

/// Estimate ⟨H⟩ of `circuit(theta)` from `shots` shots in total (at least
/// two per group), spread over the groups of qubit-wise commuting terms
/// (`allocation`: "uniform", "weighted" by Σ|c|, or "adaptive" Neyman
/// allocation chosen with a `pilot` fraction of the shots, which do not
/// enter the estimate). With `readout_calibration` the measured parities are
/// mitigated. Returns `(estimate, groups)` where each group is a dict with
/// `basis`, `terms`, `shots` and `estimate`.
#[pyfunction]
#[pyo3(signature = (
    circuit, observable, shots, theta = None, allocation = "weighted", pilot = 0.1,
    readout_calibration = None, seed = 0, noise = None, device = None, trajectories = None
))]
#[allow(clippy::too_many_arguments)]
fn estimate<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    observable: &PyPauliSum,
    shots: usize,
    theta: Option<RealArg<'_>>,
    allocation: &str,
    pilot: f64,
    readout_calibration: Option<&PyReadoutCalibration>,
    seed: u64,
    noise: Option<&PyNoiseModel>,
    device: Option<&PyDeviceModel>,
    trajectories: Option<usize>,
) -> PyResult<(PyEstimate, Bound<'py, PyList>)> {
    let theta = theta_or_empty(theta);
    check_finite("theta", &theta)?;
    let mut options = EstimatorOptions::new(shots)
        .with_allocation(self::allocation(allocation, pilot)?)
        .with_seed(seed);
    if let Some(cal) = readout_calibration {
        options = options.with_readout_mitigation(cal.inner.clone());
    }
    let backend = backend(noise, device, trajectories)?;
    let (c, h) = (&circuit.inner, &observable.inner);
    let result = py
        .detach(|| estimate_core(backend.as_ref(), c, &theta, h, &options))
        .raise()?;
    let groups = PyList::empty(py);
    for g in &result.groups {
        let basis = g
            .basis
            .iter()
            .map(|(q, p)| format!("{p:?}{q}"))
            .collect::<Vec<_>>()
            .join(" ");
        groups.append(pydict!(py, {
            "basis" => basis,
            "terms" => g.terms,
            "shots" => g.shots,
            "estimate" => wrap_estimate(g.estimate),
        }))?;
    }
    Ok((wrap_estimate(result.estimate), groups))
}

/// Classical shadow: snapshots of a state measured in random Pauli bases.
#[pyclass(name = "ClassicalShadow", module = "metatron_qso", frozen)]
pub struct PyClassicalShadow {
    inner: ClassicalShadow,
}

#[pymethods]
impl PyClassicalShadow {
    /// `snapshots` random-basis snapshots of a state vector.
    #[staticmethod]
    #[pyo3(signature = (state, snapshots, seed = 0))]
    fn from_state(
        py: Python<'_>,
        state: &PyStateVector,
        snapshots: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let s = &state.inner;
        py.detach(|| ClassicalShadow::from_state(s, snapshots, seed))
            .raise()
            .map(|inner| Self { inner })
    }

    /// `snapshots` snapshots of `circuit(theta)`, measured exactly, with a
    /// noise model or on a device (see `estimate`).
    #[staticmethod]
    #[pyo3(signature = (circuit, snapshots, theta = None, seed = 0, noise = None, device = None, trajectories = None))]
    #[allow(clippy::too_many_arguments)]
    fn collect(
        py: Python<'_>,
        circuit: &PyCircuit,
        snapshots: usize,
        theta: Option<RealArg<'_>>,
        seed: u64,
        noise: Option<&PyNoiseModel>,
        device: Option<&PyDeviceModel>,
        trajectories: Option<usize>,
    ) -> PyResult<Self> {
        let theta = theta_or_empty(theta);
        check_finite("theta", &theta)?;
        let backend = backend(noise, device, trajectories)?;
        let c = &circuit.inner;
        py.detach(|| ClassicalShadow::collect(backend.as_ref(), c, &theta, snapshots, seed))
            .raise()
            .map(|inner| Self { inner })
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Mean over the snapshots with its standard error.
    fn estimate(&self, py: Python<'_>, observable: &PyPauliSum) -> PyResult<PyEstimate> {
        let (s, h) = (&self.inner, &observable.inner);
        py.detach(|| s.estimate(h)).raise().map(wrap_estimate)
    }

    /// Median of the means of `groups` blocks of snapshots.
    fn median_of_means(
        &self,
        py: Python<'_>,
        observable: &PyPauliSum,
        groups: usize,
    ) -> PyResult<f64> {
        let (s, h) = (&self.inner, &observable.inner);
        py.detach(|| s.median_of_means(h, groups)).raise()
    }

    fn __repr__(&self) -> String {
        format!(
            "ClassicalShadow(n_qubits={}, snapshots={})",
            self.inner.n_qubits(),
            self.inner.len()
        )
    }
}

/// Read an `Estimate` or a float (`what` names its source in errors).
pub(crate) fn extract_estimate(value: &Bound<'_, PyAny>, what: &str) -> PyResult<Estimate> {
    if let Ok(e) = value.cast::<PyEstimate>() {
        return Ok(e.get().inner);
    }
    let v: f64 = value
        .extract()
        .map_err(|_| invalid(format!("{what} must be a float or an Estimate")))?;
    if !v.is_finite() {
        return Err(invalid(format!("{what} must be finite")));
    }
    Ok(Estimate::exact(v))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEstimate>()?;
    m.add_class::<PyClassicalShadow>()?;
    m.add_function(wrap_pyfunction!(estimate, m)?)?;
    Ok(())
}
