//! Quantum backend implementations

pub mod local;

#[cfg(feature = "ibm")]
pub mod ibm;

use crate::circuit::{MetatronCircuit, MeasurementResult};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Capabilities and metadata for a quantum backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendCapabilities {
    /// Provider name (e.g., "local", "ibm", "azure")
    pub provider: String,
    /// Backend name (e.g., "local_sim", "ibm_osaka", "ionq_aria")
    pub name: String,
    /// Number of qubits available
    pub num_qubits: u32,
    /// Whether this is a simulator (true) or real QPU (false)
    pub is_simulator: bool,
    /// Maximum number of shots per job
    pub max_shots: Option<u32>,
    /// Whether the backend is currently available
    pub available: bool,
    /// Additional metadata
    pub metadata: serde_json::Value,
}

impl BackendCapabilities {
    /// Create basic capabilities for a simulator
    pub fn simulator(provider: &str, name: &str, num_qubits: u32) -> Self {
        Self {
            provider: provider.to_string(),
            name: name.to_string(),
            num_qubits,
            is_simulator: true,
            max_shots: Some(1_000_000),
            available: true,
            metadata: serde_json::json!({}),
        }
    }

    /// Create basic capabilities for a QPU
    pub fn qpu(provider: &str, name: &str, num_qubits: u32) -> Self {
        Self {
            provider: provider.to_string(),
            name: name.to_string(),
            num_qubits,
            is_simulator: false,
            max_shots: Some(100_000),
            available: false, // Must be explicitly enabled
            metadata: serde_json::json!({}),
        }
    }
}

/// Trait for quantum backend implementations
///
/// All quantum backends must implement this trait to be used with Q⊗DASH.
/// This provides a unified interface for circuit execution across different
/// providers (local simulator, IBM, Azure, IonQ, etc.).
pub trait QuantumBackend: Send + Sync {
    /// Get backend capabilities and metadata
    fn info(&self) -> BackendCapabilities;

    /// Execute a quantum circuit and return measurement results
    ///
    /// # Arguments
    /// * `circuit` - The circuit to execute
    /// * `shots` - Number of measurement shots
    ///
    /// # Returns
    /// Measurement results with counts and metadata
    fn run_circuit(&self, circuit: &MetatronCircuit, shots: u32) -> Result<MeasurementResult>;

    /// Check if this backend can handle a circuit with given requirements
    fn can_run(&self, num_qubits: usize) -> bool {
        let caps = self.info();
        caps.available && (num_qubits as u32) <= caps.num_qubits
    }

    /// Get a human-readable description of this backend
    fn description(&self) -> String {
        let caps = self.info();
        format!(
            "{} ({}) - {} qubits, {}",
            caps.name,
            caps.provider,
            caps.num_qubits,
            if caps.is_simulator { "simulator" } else { "QPU" }
        )
    }
}

/// Helper to create a boxed backend trait object
pub type BoxedBackend = Box<dyn QuantumBackend>;
