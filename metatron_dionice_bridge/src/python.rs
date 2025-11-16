//! PyO3 Python bindings for DioniceKernel
//!
//! This module provides Python bindings for the Rust DioniceKernel,
//! enabling direct integration with the SCS Python code.

use crate::{DioniceKernel, QDashCalibrationState, QDashCalibrationSuggestion};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Python wrapper for DioniceKernel
#[pyclass(name = "DioniceKernel")]
pub struct PyDioniceKernel {
    kernel: DioniceKernel,
}

#[pymethods]
impl PyDioniceKernel {
    /// Create a new DioniceKernel
    #[new]
    fn new() -> Self {
        Self {
            kernel: DioniceKernel::new(),
        }
    }

    /// Ingest calibration state and get suggestion
    ///
    /// Args:
    ///     psi (float): Quality metric (0.0 - 1.0)
    ///     rho (float): Stability metric (0.0 - 1.0)
    ///     omega (float): Efficiency metric (0.0 - 1.0)
    ///     algorithm (str): Algorithm family ("VQE", "QAOA", "VQC")
    ///
    /// Returns:
    ///     dict: Calibration suggestion with keys:
    ///         - new_config (dict): Suggested configuration updates
    ///         - notes (str): Human-readable suggestions
    ///         - resonance_score (float): Quality score
    ///         - regime_change_suggested (bool): Whether to switch algorithms
    fn step(
        &mut self,
        psi: f64,
        rho: f64,
        omega: f64,
        algorithm: String,
    ) -> PyResult<PyDioniceResult> {
        let state = QDashCalibrationState {
            psi,
            rho,
            omega,
            algorithm,
            extra_params: HashMap::new(),
        };

        self.kernel
            .ingest_state(state)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let suggestion = self
            .kernel
            .step()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(PyDioniceResult::from(suggestion))
    }

    /// Switch to Explore policy
    fn switch_to_explore(&mut self) {
        self.kernel.switch_to_explore();
    }

    /// Switch to Exploit policy
    fn switch_to_exploit(&mut self) {
        self.kernel.switch_to_exploit();
    }

    /// Switch to Homeostasis policy
    fn switch_to_homeostasis(&mut self) {
        self.kernel.switch_to_homeostasis();
    }

    /// Get current funnel density
    fn funnel_density(&self) -> f64 {
        self.kernel.funnel_density()
    }

    /// Get number of nodes in funnel
    fn funnel_node_count(&self) -> usize {
        self.kernel.funnel_node_count()
    }
}

/// Python-compatible result structure
#[pyclass(name = "DioniceResult")]
#[derive(Clone)]
pub struct PyDioniceResult {
    #[pyo3(get)]
    pub notes: String,
    #[pyo3(get)]
    pub resonance_score: f64,
    #[pyo3(get)]
    pub regime_change_suggested: bool,
    pub new_config: serde_json::Value,
}

#[pymethods]
impl PyDioniceResult {
    /// Get configuration updates as dict
    fn get_config(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let config_str = serde_json::to_string(&self.new_config)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

            let json_module = py.import("json")?;
            let config_dict = json_module.call_method1("loads", (config_str,))?;

            Ok(config_dict.to_object(py))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "DioniceResult(resonance_score={:.4}, regime_change={})",
            self.resonance_score, self.regime_change_suggested
        )
    }
}

impl From<QDashCalibrationSuggestion> for PyDioniceResult {
    fn from(suggestion: QDashCalibrationSuggestion) -> Self {
        Self {
            notes: suggestion.notes,
            resonance_score: suggestion.resonance_score,
            regime_change_suggested: suggestion.regime_change_suggested,
            new_config: suggestion.new_config,
        }
    }
}

/// Python module definition
#[pymodule]
fn metatron_dionice_bridge(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDioniceKernel>()?;
    m.add_class::<PyDioniceResult>()?;

    m.add(
        "__doc__",
        "dioniceOS backend integration for Q⊗DASH Seraphic Calibration Shell",
    )?;

    Ok(())
}
