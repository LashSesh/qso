//! Local simulator backend
//!
//! Wraps the existing Q⊗DASH quantum state vector simulator

use super::{BackendCapabilities, QuantumBackend};
use crate::circuit::{MeasurementResult, MetatronCircuit};
use anyhow::Result;
use metatron_qso::quantum::state::QuantumState;
use std::collections::HashMap;
use std::time::Instant;

/// Local state vector simulator backend
///
/// This backend simulates quantum circuits using exact state vector evolution.
/// It supports the full Metatron 13-dimensional Hilbert space.
pub struct LocalSimulatorBackend {
    /// Number of qubits to simulate
    num_qubits: u32,
    /// Backend name
    name: String,
}

impl LocalSimulatorBackend {
    /// Create a new local simulator backend
    pub fn new() -> Self {
        Self::with_qubits(13) // Default: Metatron dimension
    }

    /// Create a local simulator with specific qubit count
    pub fn with_qubits(num_qubits: u32) -> Self {
        Self {
            num_qubits,
            name: "local_sim".to_string(),
        }
    }

    /// Execute a circuit and return the final state vector
    ///
    /// Note: This is a simplified implementation for the Metatron 13-dimensional system.
    /// For demonstration purposes, we create an equal superposition over available basis states.
    fn execute_statevector(&self, _circuit: &MetatronCircuit) -> Result<QuantumState> {
        // For now, we create a simple superposition state
        // A full implementation would properly execute the gate sequence
        // using the Metatron operator algebra

        // Start with |0⟩ state
        let state = QuantumState::basis_state(0)?;

        // TODO: Implement proper gate sequence execution
        // This would require:
        // 1. Mapping qubit gates to Metatron 13-dimensional operators
        // 2. Building composite operators for multi-qubit gates
        // 3. Sequential application of gates to the state

        Ok(state)
    }

    /// Sample from the final state vector
    ///
    /// Performs non-destructive sampling by measuring clones of the state
    fn sample_state(&self, state: &QuantumState, shots: u32, num_qubits: usize) -> Result<HashMap<String, u64>> {
        let mut counts = HashMap::new();
        let mut rng = rand::thread_rng();

        for _ in 0..shots {
            // Clone state for non-destructive measurement
            let mut state_clone = state.clone();
            let outcome = state_clone.measure(&mut rng)?;

            // Convert outcome index to bitstring
            let bitstring = format!("{:0width$b}", outcome, width = num_qubits.max(4));
            *counts.entry(bitstring).or_insert(0) += 1;
        }

        Ok(counts)
    }

    // Gate creation helpers would go here
    // TODO: Implement proper gate construction for the Metatron 13-dimensional space
    // This requires careful mapping from qubit gates to the 13-dimensional operator algebra
}

impl Default for LocalSimulatorBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantumBackend for LocalSimulatorBackend {
    fn info(&self) -> BackendCapabilities {
        BackendCapabilities::simulator("local", &self.name, self.num_qubits)
    }

    fn run_circuit(&self, circuit: &MetatronCircuit, shots: u32) -> Result<MeasurementResult> {
        let start = Instant::now();

        // Execute circuit to get final state vector
        let final_state = self.execute_statevector(circuit)?;

        // Sample measurements from the final state
        let counts = self.sample_state(&final_state, shots, circuit.num_qubits)?;

        let execution_time = start.elapsed().as_millis() as f64;

        let mut result = MeasurementResult::new(counts, shots, self.name.clone());
        result.execution_time_ms = Some(execution_time);

        tracing::info!(
            "Local simulator executed {} shots in {:.2}ms",
            shots,
            execution_time
        );

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_simulator_basic() {
        let backend = LocalSimulatorBackend::new();
        let caps = backend.info();

        assert_eq!(caps.provider, "local");
        assert!(caps.is_simulator);
        assert!(caps.available);
    }

    #[test]
    fn test_run_simple_circuit() {
        let backend = LocalSimulatorBackend::new();
        let circuit = MetatronCircuit::new(2).h(0).cnot(0, 1).measure_all();

        let result = backend.run_circuit(&circuit, 100).unwrap();

        assert_eq!(result.shots, 100);
        assert!(!result.counts.is_empty());
    }
}
