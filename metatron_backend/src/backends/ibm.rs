//! IBM Quantum backend
//!
//! Provides integration with IBM Quantum Platform via Qiskit Runtime.
//!
//! ## Safety Features
//!
//! - **Default Mode: Disabled** - No QPU access unless explicitly enabled
//! - **Dry-Run Mode** - Test circuits without consuming QPU time
//! - **Explicit Configuration** - Must set environment variables or config file
//!
//! ## Configuration
//!
//! Set these environment variables:
//! - `IBM_QUANTUM_TOKEN` - Your IBM Quantum API token (required for Enabled mode)
//! - `IBM_BACKEND_NAME` - Backend name (e.g., "ibm_kyoto", "ibm_osaka")
//! - `IBM_BACKEND_MODE` - "disabled", "dry-run", or "enabled" (default: "disabled")
//!
//! ## Example
//!
//! ```rust,no_run
//! use metatron_backend::IbmQuantumBackend;
//!
//! // Create IBM backend (reads from environment)
//! let backend = IbmQuantumBackend::from_env().unwrap();
//!
//! // Check mode
//! println!("IBM mode: {:?}", backend.mode());
//! ```

use super::{BackendCapabilities, QuantumBackend};
use crate::circuit::{MeasurementResult, MetatronCircuit};
use anyhow::{bail, Result};
use figment::{providers::Env, Figment};
use serde::Deserialize;
use std::collections::HashMap;
use std::time::Instant;

/// IBM Quantum backend execution mode
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum IbmMode {
    /// Backend is disabled - will error on circuit execution
    ///
    /// This is the SAFE DEFAULT. No QPU access possible.
    #[default]
    Disabled,

    /// Dry-run mode - log circuits without executing
    ///
    /// Circuits are validated and logged, but not sent to IBM.
    /// Returns stubbed measurement results for testing.
    DryRun,

    /// Enabled - execute circuits on real IBM hardware
    ///
    /// **WARNING**: This mode consumes QPU time and may incur costs.
    /// Only use when explicitly authorized.
    Enabled,
}

/// Configuration for IBM Quantum backend
#[derive(Debug, Clone, Deserialize)]
pub struct IbmConfig {
    /// IBM Quantum API token
    #[serde(default)]
    pub token: Option<String>,

    /// Backend name (e.g., "ibm_kyoto", "ibm_osaka")
    #[serde(default = "default_backend_name")]
    pub backend_name: String,

    /// Execution mode
    #[serde(default)]
    pub mode: IbmMode,

    /// Maximum number of shots per job
    #[serde(default = "default_max_shots")]
    pub max_shots: u32,
}

fn default_backend_name() -> String {
    "ibm_kyoto".to_string()
}

fn default_max_shots() -> u32 {
    8192
}

impl Default for IbmConfig {
    fn default() -> Self {
        Self {
            token: None,
            backend_name: default_backend_name(),
            mode: IbmMode::default(),
            max_shots: default_max_shots(),
        }
    }
}

impl IbmConfig {
    /// Load configuration from environment variables
    ///
    /// Environment variables:
    /// - `IBM_QUANTUM_TOKEN`
    /// - `IBM_BACKEND_NAME`
    /// - `IBM_BACKEND_MODE`
    /// - `IBM_MAX_SHOTS`
    pub fn from_env() -> Result<Self> {
        let config: IbmConfig = Figment::new()
            .merge(Env::prefixed("IBM_").map(|key| {
                // Map IBM_QUANTUM_TOKEN -> token
                // Map IBM_BACKEND_NAME -> backend_name
                // Map IBM_BACKEND_MODE -> mode
                key.as_str()
                    .strip_prefix("QUANTUM_")
                    .or(key.as_str().strip_prefix("BACKEND_"))
                    .or(key.as_str().strip_prefix("MAX_"))
                    .unwrap_or(key.as_str())
                    .to_lowercase()
                    .into()
            }))
            .extract()?;

        // Validate configuration
        if config.mode == IbmMode::Enabled && config.token.is_none() {
            bail!("IBM_QUANTUM_TOKEN is required when mode is 'enabled'");
        }

        Ok(config)
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        match self.mode {
            IbmMode::Disabled => {
                // No validation needed
                Ok(())
            }
            IbmMode::DryRun => {
                // Backend name should be set
                if self.backend_name.is_empty() {
                    bail!("backend_name must be set for dry-run mode");
                }
                Ok(())
            }
            IbmMode::Enabled => {
                // Token and backend name must be set
                if self.token.is_none() {
                    bail!("token is required for enabled mode");
                }
                if self.backend_name.is_empty() {
                    bail!("backend_name must be set for enabled mode");
                }
                Ok(())
            }
        }
    }
}

/// IBM Quantum backend adapter
///
/// Provides integration with IBM Quantum Platform. Supports three modes:
/// - Disabled (default): No execution possible
/// - Dry-run: Validate and log circuits without execution
/// - Enabled: Execute on real IBM hardware
pub struct IbmQuantumBackend {
    config: IbmConfig,
}

impl IbmQuantumBackend {
    /// Create IBM backend with custom configuration
    pub fn new(config: IbmConfig) -> Result<Self> {
        config.validate()?;

        tracing::info!(
            "IBM Quantum backend initialized: {} (mode: {:?})",
            config.backend_name,
            config.mode
        );

        if config.mode == IbmMode::Enabled {
            tracing::warn!("IBM backend is in ENABLED mode - will consume QPU time!");
        }

        Ok(Self { config })
    }

    /// Create IBM backend from environment variables
    pub fn from_env() -> Result<Self> {
        let config = IbmConfig::from_env()?;
        Self::new(config)
    }

    /// Get the current execution mode
    pub fn mode(&self) -> &IbmMode {
        &self.config.mode
    }

    /// Execute in dry-run mode (log and return stubbed result)
    fn execute_dry_run(&self, circuit: &MetatronCircuit, shots: u32) -> Result<MeasurementResult> {
        tracing::info!(
            "[DRY-RUN] Would execute circuit on '{}': {} qubits, {} gates, {} shots",
            self.config.backend_name,
            circuit.num_qubits,
            circuit.gates.len(),
            shots
        );

        // Log circuit structure
        tracing::debug!("[DRY-RUN] Circuit depth: {}", circuit.depth());

        // Return stubbed result (equal superposition)
        let mut counts = HashMap::new();
        let num_outcomes = 2_usize.pow(circuit.num_qubits.min(10) as u32);

        for i in 0..num_outcomes.min(100) {
            let bitstring = format!("{:0width$b}", i, width = circuit.num_qubits);
            counts.insert(bitstring, (shots / num_outcomes.min(100) as u32) as u64);
        }

        let mut result = MeasurementResult::new(
            counts,
            shots,
            format!("{}_dry_run", self.config.backend_name),
        );
        result.execution_time_ms = Some(0.0);

        Ok(result)
    }

    /// Execute on real IBM hardware
    #[cfg(feature = "ibm")]
    fn execute_real(&self, _circuit: &MetatronCircuit, _shots: u32) -> Result<MeasurementResult> {
        use tokio::runtime::Runtime;

        tracing::warn!(
            "Executing circuit on REAL IBM HARDWARE: {}",
            self.config.backend_name
        );

        let _token = self.config.token.as_ref().unwrap();

        // Create async runtime for IBM API calls
        let rt = Runtime::new()?;

        rt.block_on(async {
            // TODO: Implement real IBM Qiskit Runtime API calls
            // This would involve:
            // 1. Convert MetatronCircuit to Qiskit circuit JSON
            // 2. Submit job via IBM Quantum REST API
            // 3. Poll for job completion
            // 4. Retrieve and parse results

            bail!("Real IBM execution not yet implemented - use Python sidecar or REST API")
        })
    }

    /// Stub implementation for when ibm feature is disabled
    #[cfg(not(feature = "ibm"))]
    fn execute_real(&self, _circuit: &MetatronCircuit, _shots: u32) -> Result<MeasurementResult> {
        bail!("IBM backend feature not enabled. Recompile with --features ibm")
    }
}

impl QuantumBackend for IbmQuantumBackend {
    fn info(&self) -> BackendCapabilities {
        let available = match self.config.mode {
            IbmMode::Disabled => false,
            IbmMode::DryRun => true,
            IbmMode::Enabled => self.config.token.is_some(),
        };

        BackendCapabilities {
            provider: "ibm".to_string(),
            name: self.config.backend_name.clone(),
            num_qubits: 127, // IBM Quantum System Two
            is_simulator: false,
            max_shots: Some(self.config.max_shots),
            available,
            metadata: serde_json::json!({
                "mode": format!("{:?}", self.config.mode),
                "max_shots": self.config.max_shots,
            }),
        }
    }

    fn run_circuit(&self, circuit: &MetatronCircuit, shots: u32) -> Result<MeasurementResult> {
        let start = Instant::now();

        // Validate shot count
        if shots > self.config.max_shots {
            bail!(
                "Requested {} shots exceeds maximum {} for backend '{}'",
                shots,
                self.config.max_shots,
                self.config.backend_name
            );
        }

        let result = match self.config.mode {
            IbmMode::Disabled => {
                bail!(
                    "IBM backend '{}' is disabled. Set IBM_BACKEND_MODE=dry-run or enabled",
                    self.config.backend_name
                );
            }

            IbmMode::DryRun => self.execute_dry_run(circuit, shots)?,

            IbmMode::Enabled => self.execute_real(circuit, shots)?,
        };

        let execution_time = start.elapsed().as_millis() as f64;
        tracing::info!(
            "IBM backend '{}' completed in {:.2}ms (mode: {:?})",
            self.config.backend_name,
            execution_time,
            self.config.mode
        );

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = IbmConfig::default();
        assert_eq!(config.mode, IbmMode::Disabled);
        assert!(config.token.is_none());
        assert_eq!(config.backend_name, "ibm_kyoto");
    }

    #[test]
    fn test_disabled_mode() {
        let config = IbmConfig {
            mode: IbmMode::Disabled,
            ..Default::default()
        };

        let backend = IbmQuantumBackend::new(config).unwrap();
        let circuit = MetatronCircuit::new(2);

        let result = backend.run_circuit(&circuit, 100);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("disabled"));
    }

    #[test]
    fn test_dry_run_mode() {
        let config = IbmConfig {
            mode: IbmMode::DryRun,
            backend_name: "ibm_test".to_string(),
            ..Default::default()
        };

        let backend = IbmQuantumBackend::new(config).unwrap();
        let circuit = MetatronCircuit::new(2).h(0).cnot(0, 1).measure_all();

        let result = backend.run_circuit(&circuit, 100).unwrap();
        assert_eq!(result.shots, 100);
        assert!(!result.counts.is_empty());
        assert!(result.backend_name.contains("dry_run"));
    }

    #[test]
    fn test_mode_validation() {
        // Enabled mode requires token
        let config = IbmConfig {
            mode: IbmMode::Enabled,
            token: None,
            ..Default::default()
        };

        let result = IbmQuantumBackend::new(config);
        assert!(result.is_err());
    }
}
