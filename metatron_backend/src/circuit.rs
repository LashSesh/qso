//! Circuit representation for quantum backends
//!
//! Provides a backend-agnostic circuit representation that can be executed
//! on any quantum backend.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Gate types supported by the backend abstraction
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GateType {
    // Single-qubit gates
    /// Hadamard gate
    H,
    /// Pauli X (NOT)
    X,
    /// Pauli Y
    Y,
    /// Pauli Z
    Z,
    /// S gate (phase)
    S,
    /// S† gate
    Sdg,
    /// T gate
    T,
    /// T† gate
    Tdg,
    /// Rotation around X axis
    RX(f64),
    /// Rotation around Y axis
    RY(f64),
    /// Rotation around Z axis
    RZ(f64),
    /// Arbitrary single-qubit unitary (theta, phi, lambda)
    U(f64, f64, f64),

    // Two-qubit gates
    /// Controlled-NOT
    CNOT,
    /// Controlled-Z
    CZ,
    /// SWAP
    SWAP,
    /// Controlled phase
    CPhase(f64),

    // Multi-qubit gates
    /// Toffoli (CCX)
    Toffoli,

    // Measurement
    /// Measurement in computational basis
    Measure,
}

/// A single quantum gate with target qubits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gate {
    pub gate_type: GateType,
    pub qubits: Vec<usize>,
}

/// Backend-agnostic quantum circuit
///
/// This circuit representation can be executed on any backend that implements
/// the `QuantumBackend` trait.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetatronCircuit {
    /// Number of qubits
    pub num_qubits: usize,
    /// Sequence of gates
    pub gates: Vec<Gate>,
    /// Classical register size (for measurements)
    pub num_clbits: usize,
}

impl MetatronCircuit {
    /// Create a new circuit with specified number of qubits
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            gates: Vec::new(),
            num_clbits: num_qubits, // Default: one classical bit per qubit
        }
    }

    /// Add a gate to the circuit
    pub fn add_gate(&mut self, gate_type: GateType, qubits: Vec<usize>) {
        // Validate qubit indices
        for &q in &qubits {
            assert!(q < self.num_qubits, "Qubit index {} out of bounds", q);
        }

        self.gates.push(Gate { gate_type, qubits });
    }

    /// Builder pattern for gates
    pub fn h(mut self, qubit: usize) -> Self {
        self.add_gate(GateType::H, vec![qubit]);
        self
    }

    pub fn x(mut self, qubit: usize) -> Self {
        self.add_gate(GateType::X, vec![qubit]);
        self
    }

    pub fn y(mut self, qubit: usize) -> Self {
        self.add_gate(GateType::Y, vec![qubit]);
        self
    }

    pub fn z(mut self, qubit: usize) -> Self {
        self.add_gate(GateType::Z, vec![qubit]);
        self
    }

    pub fn s(mut self, qubit: usize) -> Self {
        self.add_gate(GateType::S, vec![qubit]);
        self
    }

    pub fn t(mut self, qubit: usize) -> Self {
        self.add_gate(GateType::T, vec![qubit]);
        self
    }

    pub fn rx(mut self, qubit: usize, theta: f64) -> Self {
        self.add_gate(GateType::RX(theta), vec![qubit]);
        self
    }

    pub fn ry(mut self, qubit: usize, theta: f64) -> Self {
        self.add_gate(GateType::RY(theta), vec![qubit]);
        self
    }

    pub fn rz(mut self, qubit: usize, theta: f64) -> Self {
        self.add_gate(GateType::RZ(theta), vec![qubit]);
        self
    }

    pub fn u(mut self, qubit: usize, theta: f64, phi: f64, lambda: f64) -> Self {
        self.add_gate(GateType::U(theta, phi, lambda), vec![qubit]);
        self
    }

    pub fn cnot(mut self, control: usize, target: usize) -> Self {
        self.add_gate(GateType::CNOT, vec![control, target]);
        self
    }

    pub fn cz(mut self, control: usize, target: usize) -> Self {
        self.add_gate(GateType::CZ, vec![control, target]);
        self
    }

    pub fn swap(mut self, qubit1: usize, qubit2: usize) -> Self {
        self.add_gate(GateType::SWAP, vec![qubit1, qubit2]);
        self
    }

    pub fn measure(mut self, qubit: usize) -> Self {
        self.add_gate(GateType::Measure, vec![qubit]);
        self
    }

    pub fn measure_all(mut self) -> Self {
        for q in 0..self.num_qubits {
            self.add_gate(GateType::Measure, vec![q]);
        }
        self
    }

    /// Get depth of the circuit (number of layers)
    pub fn depth(&self) -> usize {
        // Simple depth calculation: just count gates
        // (A more sophisticated version would compute parallel gate layers)
        self.gates.len()
    }

    /// Count number of specific gate type
    pub fn count_gates(&self, gate_type: &GateType) -> usize {
        self.gates
            .iter()
            .filter(|g| &g.gate_type == gate_type)
            .count()
    }
}

/// Result of measuring a quantum circuit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementResult {
    /// Measurement outcomes and their counts
    /// Keys are bitstrings like "00", "01", "10", "11"
    pub counts: HashMap<String, u64>,
    /// Total number of shots
    pub shots: u32,
    /// Execution time (milliseconds)
    pub execution_time_ms: Option<f64>,
    /// Backend that executed the circuit
    pub backend_name: String,
}

impl MeasurementResult {
    /// Create a new measurement result
    pub fn new(counts: HashMap<String, u64>, shots: u32, backend_name: String) -> Self {
        Self {
            counts,
            shots,
            execution_time_ms: None,
            backend_name,
        }
    }

    /// Get probability of a specific outcome
    pub fn probability(&self, outcome: &str) -> f64 {
        let count = self.counts.get(outcome).copied().unwrap_or(0);
        count as f64 / self.shots as f64
    }

    /// Get the most likely outcome
    pub fn most_likely_outcome(&self) -> Option<(String, u64)> {
        self.counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(outcome, &count)| (outcome.clone(), count))
    }

    /// Compute expectation value of Z on a specific qubit
    pub fn expectation_z(&self, qubit: usize) -> f64 {
        let mut expectation = 0.0;
        for (outcome, &count) in &self.counts {
            let prob = count as f64 / self.shots as f64;
            // Check if qubit is |1⟩ (contributes -1) or |0⟩ (contributes +1)
            let bit = outcome.chars().rev().nth(qubit).unwrap_or('0');
            let sign = if bit == '1' { -1.0 } else { 1.0 };
            expectation += sign * prob;
        }
        expectation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_builder() {
        let circuit = MetatronCircuit::new(2).h(0).cnot(0, 1).measure_all();

        assert_eq!(circuit.num_qubits, 2);
        assert_eq!(circuit.gates.len(), 4); // H + CNOT + 2 measurements
    }

    #[test]
    fn test_measurement_result() {
        let mut counts = HashMap::new();
        counts.insert("00".to_string(), 500);
        counts.insert("11".to_string(), 500);

        let result = MeasurementResult::new(counts, 1000, "test".to_string());

        assert_eq!(result.probability("00"), 0.5);
        assert_eq!(result.probability("11"), 0.5);
        assert_eq!(result.expectation_z(0), 0.0);
    }
}
