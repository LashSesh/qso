//! Parametrized Quantum Circuit Ansätze
//!
//! Provides various ansatz templates for variational quantum algorithms:
//! - Hardware-Efficient: Alternating rotations and entanglers for NISQ devices
//! - EfficientSU2: Qiskit-inspired structure with full SU(2) rotations
//! - Metatron: Optimized for 13-dimensional Metatron Cube structure

use crate::quantum::operator::{OperatorMatrix, QuantumOperator};
use crate::quantum::state::{METATRON_DIMENSION, QuantumState};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Ansatz type variants
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnsatzType {
    HardwareEfficient,
    EfficientSU2,
    Metatron,
}

/// Trait for parametrized quantum circuits
pub trait Ansatz: Send + Sync {
    /// Apply the ansatz to a quantum state with given parameters
    fn apply(&self, state: &QuantumState, parameters: &[f64]) -> QuantumState;

    /// Get the total number of parameters
    fn num_parameters(&self) -> usize;

    /// Get the ansatz type
    fn ansatz_type(&self) -> AnsatzType;

    /// Get the number of qubits (for Metatron: effectively log2(13) ≈ 3.7, but we work in 13-dim)
    fn num_qubits(&self) -> usize {
        METATRON_DIMENSION
    }

    /// Get the circuit depth (number of layers)
    fn depth(&self) -> usize;

    /// Validate parameter vector length
    fn validate_parameters(&self, parameters: &[f64]) -> Result<(), String> {
        if parameters.len() != self.num_parameters() {
            Err(format!(
                "Expected {} parameters, got {}",
                self.num_parameters(),
                parameters.len()
            ))
        } else {
            Ok(())
        }
    }
}

/// Hardware-Efficient Ansatz
///
/// Alternating layers of single-qubit rotations (Ry) and nearest-neighbor entangling gates.
/// This is designed for NISQ devices where gate count must be minimized.
///
/// Structure per layer:
/// 1. Ry(θ) rotations on all qubits
/// 2. Rz(θ) rotations on all qubits
/// 3. Circular entangling pattern using controlled rotations
///
/// Parameters: 2 * num_qubits * depth
#[derive(Clone, Debug)]
pub struct HardwareEfficientAnsatz {
    num_qubits: usize,
    depth: usize,
}

impl HardwareEfficientAnsatz {
    pub fn new(depth: usize) -> Self {
        Self {
            num_qubits: METATRON_DIMENSION,
            depth,
        }
    }

    /// Create Ry rotation matrix for the full Hilbert space
    fn ry_rotation_matrix(&self, qubit: usize, angle: f64) -> OperatorMatrix {
        // For 13-dimensional space, we create a rotation that acts on specific dimensions
        // This is a simplified rotation acting on pairs of basis states
        let mut matrix = OperatorMatrix::identity();

        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        // Apply rotation on the qubit subspace
        if qubit < METATRON_DIMENSION - 1 {
            matrix[(qubit, qubit)] = Complex64::new(cos_half, 0.0);
            matrix[(qubit, qubit + 1)] = Complex64::new(-sin_half, 0.0);
            matrix[(qubit + 1, qubit)] = Complex64::new(sin_half, 0.0);
            matrix[(qubit + 1, qubit + 1)] = Complex64::new(cos_half, 0.0);
        }

        matrix
    }

    /// Create Rz rotation matrix
    fn rz_rotation_matrix(&self, qubit: usize, angle: f64) -> OperatorMatrix {
        let mut matrix = OperatorMatrix::identity();

        if qubit < METATRON_DIMENSION {
            let phase_plus = Complex64::from_polar(1.0, angle / 2.0);
            let phase_minus = Complex64::from_polar(1.0, -angle / 2.0);

            matrix[(qubit, qubit)] = phase_minus;
            if qubit + 1 < METATRON_DIMENSION {
                matrix[(qubit + 1, qubit + 1)] = phase_plus;
            }
        }

        matrix
    }

    /// Create entangling gate between neighboring qubits
    fn entangling_gate(&self, qubit1: usize, qubit2: usize, angle: f64) -> OperatorMatrix {
        let mut matrix = OperatorMatrix::identity();

        if qubit1 < METATRON_DIMENSION && qubit2 < METATRON_DIMENSION {
            // Controlled rotation
            let cos_val = angle.cos();
            let sin_val = angle.sin();
            let i_sin = Complex64::new(0.0, sin_val);

            matrix[(qubit1, qubit1)] = Complex64::new(cos_val, 0.0);
            matrix[(qubit1, qubit2)] = -i_sin;
            matrix[(qubit2, qubit1)] = -i_sin;
            matrix[(qubit2, qubit2)] = Complex64::new(cos_val, 0.0);
        }

        matrix
    }
}

impl Ansatz for HardwareEfficientAnsatz {
    fn apply(&self, state: &QuantumState, parameters: &[f64]) -> QuantumState {
        self.validate_parameters(parameters)
            .expect("Invalid parameters");

        let mut current_state = state.clone();
        let params_per_layer = 2 * self.num_qubits;

        for layer in 0..self.depth {
            let layer_offset = layer * params_per_layer;

            // Apply Ry rotations
            for qubit in 0..self.num_qubits {
                let param_idx = layer_offset + qubit;
                if param_idx < parameters.len() {
                    let rotation = self.ry_rotation_matrix(qubit, parameters[param_idx]);
                    let operator = QuantumOperator::from_matrix(rotation);
                    current_state = current_state.apply(&operator);
                }
            }

            // Apply Rz rotations
            for qubit in 0..self.num_qubits {
                let param_idx = layer_offset + self.num_qubits + qubit;
                if param_idx < parameters.len() {
                    let rotation = self.rz_rotation_matrix(qubit, parameters[param_idx]);
                    let operator = QuantumOperator::from_matrix(rotation);
                    current_state = current_state.apply(&operator);
                }
            }

            // Apply entangling gates (circular pattern)
            for qubit in 0..self.num_qubits - 1 {
                let entangle_angle = parameters[layer_offset + qubit % params_per_layer] * 0.5;
                let gate = self.entangling_gate(qubit, qubit + 1, entangle_angle);
                let operator = QuantumOperator::from_matrix(gate);
                current_state = current_state.apply(&operator);
            }
        }

        current_state
    }

    fn num_parameters(&self) -> usize {
        2 * self.num_qubits * self.depth
    }

    fn ansatz_type(&self) -> AnsatzType {
        AnsatzType::HardwareEfficient
    }

    fn depth(&self) -> usize {
        self.depth
    }
}

/// EfficientSU2 Ansatz
///
/// Inspired by Qiskit's EfficientSU2 circuit. Uses full SU(2) rotations
/// (Ry-Rz combinations) for expressive power.
///
/// Structure per layer:
/// 1. Full SU(2) rotation on each qubit (Rz-Ry-Rz)
/// 2. Entangling layer with controlled operations
///
/// Parameters: 3 * num_qubits * depth
#[derive(Clone, Debug)]
pub struct EfficientSU2Ansatz {
    num_qubits: usize,
    depth: usize,
}

impl EfficientSU2Ansatz {
    pub fn new(depth: usize) -> Self {
        Self {
            num_qubits: METATRON_DIMENSION,
            depth,
        }
    }

    fn su2_rotation(&self, qubit: usize, theta1: f64, theta2: f64, theta3: f64) -> OperatorMatrix {
        // SU(2) = Rz(θ1) · Ry(θ2) · Rz(θ3)
        let mut matrix = OperatorMatrix::identity();

        if qubit < METATRON_DIMENSION - 1 {
            // Compose the three rotations
            let cos_half = (theta2 / 2.0).cos();
            let sin_half = (theta2 / 2.0).sin();
            let phase1 = Complex64::from_polar(1.0, theta1 / 2.0);
            let phase3 = Complex64::from_polar(1.0, theta3 / 2.0);

            matrix[(qubit, qubit)] = phase1 * Complex64::new(cos_half, 0.0) * phase3;
            matrix[(qubit, qubit + 1)] = phase1 * Complex64::new(-sin_half, 0.0) * phase3.conj();
            matrix[(qubit + 1, qubit)] = phase1.conj() * Complex64::new(sin_half, 0.0) * phase3;
            matrix[(qubit + 1, qubit + 1)] =
                phase1.conj() * Complex64::new(cos_half, 0.0) * phase3.conj();
        }

        matrix
    }
}

impl Ansatz for EfficientSU2Ansatz {
    fn apply(&self, state: &QuantumState, parameters: &[f64]) -> QuantumState {
        self.validate_parameters(parameters)
            .expect("Invalid parameters");

        let mut current_state = state.clone();
        let params_per_layer = 3 * self.num_qubits;

        for layer in 0..self.depth {
            let layer_offset = layer * params_per_layer;

            // Apply SU(2) rotations
            for qubit in 0..self.num_qubits {
                let idx = layer_offset + qubit * 3;
                if idx + 2 < parameters.len() {
                    let rotation = self.su2_rotation(
                        qubit,
                        parameters[idx],
                        parameters[idx + 1],
                        parameters[idx + 2],
                    );
                    let operator = QuantumOperator::from_matrix(rotation);
                    current_state = current_state.apply(&operator);
                }
            }

            // Entangling layer (simplified for 13-dim space)
            for qubit in 0..self.num_qubits - 1 {
                let angle = PI / 4.0; // Fixed entangling angle
                let mut gate = OperatorMatrix::identity();

                let cos_val = angle.cos();
                let sin_val = angle.sin();
                gate[(qubit, qubit)] = Complex64::new(cos_val, 0.0);
                gate[(qubit, qubit + 1)] = Complex64::new(0.0, -sin_val);
                gate[(qubit + 1, qubit)] = Complex64::new(0.0, -sin_val);
                gate[(qubit + 1, qubit + 1)] = Complex64::new(cos_val, 0.0);

                let operator = QuantumOperator::from_matrix(gate);
                current_state = current_state.apply(&operator);
            }
        }

        current_state
    }

    fn num_parameters(&self) -> usize {
        3 * self.num_qubits * self.depth
    }

    fn ansatz_type(&self) -> AnsatzType {
        AnsatzType::EfficientSU2
    }

    fn depth(&self) -> usize {
        self.depth
    }
}

/// Metatron-Optimized Ansatz
///
/// Leverages the 13-sphere geometry and Metatron Cube symmetries for
/// efficient parametrization. Uses the natural graph structure for entanglement.
///
/// Entanglement strategies:
/// - Ring: Circular nearest-neighbor connections
/// - Full: All-to-all entanglement for maximum expressiveness
///
/// Parameters depend on entanglement strategy:
/// - Ring: num_qubits + num_qubits per layer
/// - Full: num_qubits + num_qubits*(num_qubits-1)/2 per layer
#[derive(Clone, Debug)]
pub struct MetatronAnsatz {
    depth: usize,
    entanglement_strategy: EntanglementStrategy,
}

/// Entanglement strategy for Metatron ansatz
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EntanglementStrategy {
    /// Ring/circular entanglement pattern
    Ring,
    /// Full all-to-all entanglement
    Full,
}

impl MetatronAnsatz {
    /// Create Metatron ansatz with default Ring entanglement
    pub fn new(depth: usize) -> Self {
        Self {
            depth,
            entanglement_strategy: EntanglementStrategy::Ring,
        }
    }

    /// Create Metatron ansatz with specified entanglement strategy
    pub fn new_with_entanglement(depth: usize, entanglement: EntanglementStrategy) -> Self {
        Self {
            depth,
            entanglement_strategy: entanglement,
        }
    }

    /// Get number of entangling gates based on strategy
    fn num_entangling_gates(&self) -> usize {
        match self.entanglement_strategy {
            EntanglementStrategy::Ring => METATRON_DIMENSION,
            EntanglementStrategy::Full => METATRON_DIMENSION * (METATRON_DIMENSION - 1) / 2,
        }
    }

    /// Create rotation leveraging Metatron geometry
    fn metatron_rotation(&self, node: usize, angle: f64) -> OperatorMatrix {
        let mut matrix = OperatorMatrix::identity();

        if node < METATRON_DIMENSION {
            // Rotation in the Bloch sphere representation
            let cos_half = (angle / 2.0).cos();
            let sin_half = (angle / 2.0).sin();

            // Central node (0) gets special treatment
            if node == 0 {
                // Rotate in higher-dimensional subspace
                for i in 0..METATRON_DIMENSION {
                    let phase = 2.0 * PI * i as f64 / METATRON_DIMENSION as f64;
                    matrix[(i, i)] = Complex64::from_polar(1.0, angle * phase);
                }
            } else {
                matrix[(node, node)] = Complex64::new(cos_half, 0.0);
                let next = (node + 1) % METATRON_DIMENSION;
                matrix[(node, next)] = Complex64::new(0.0, -sin_half);
                matrix[(next, node)] = Complex64::new(0.0, -sin_half);
                matrix[(next, next)] = Complex64::new(cos_half, 0.0);
            }
        }

        matrix
    }
}

impl Ansatz for MetatronAnsatz {
    fn apply(&self, state: &QuantumState, parameters: &[f64]) -> QuantumState {
        self.validate_parameters(parameters)
            .expect("Invalid parameters");

        let mut current_state = state.clone();
        let params_per_layer = METATRON_DIMENSION + self.num_entangling_gates();

        for layer in 0..self.depth {
            let layer_offset = layer * params_per_layer;

            // Node rotations
            for node in 0..METATRON_DIMENSION {
                let param_idx = layer_offset + node;
                if param_idx < parameters.len() {
                    let rotation = self.metatron_rotation(node, parameters[param_idx]);
                    let operator = QuantumOperator::from_matrix(rotation);
                    current_state = current_state.apply(&operator);
                }
            }

            // Entanglement layer based on strategy
            match self.entanglement_strategy {
                EntanglementStrategy::Ring => {
                    // Ring entanglement: each qubit connected to next in circular pattern
                    for i in 0..METATRON_DIMENSION {
                        let param_idx = layer_offset + METATRON_DIMENSION + i;
                        if param_idx < parameters.len() {
                            let source = i;
                            let target = (i + 1) % METATRON_DIMENSION;

                            let mut gate = OperatorMatrix::identity();
                            let angle = parameters[param_idx];
                            let cos_val = angle.cos();
                            let sin_val = angle.sin();

                            gate[(source, source)] = Complex64::new(cos_val, 0.0);
                            gate[(source, target)] = Complex64::new(0.0, -sin_val);
                            gate[(target, source)] = Complex64::new(0.0, -sin_val);
                            gate[(target, target)] = Complex64::new(cos_val, 0.0);

                            let operator = QuantumOperator::from_matrix(gate);
                            current_state = current_state.apply(&operator);
                        }
                    }
                }
                EntanglementStrategy::Full => {
                    // Full entanglement: all-to-all connections
                    let mut gate_idx = 0;
                    for i in 0..METATRON_DIMENSION {
                        for j in (i + 1)..METATRON_DIMENSION {
                            let param_idx = layer_offset + METATRON_DIMENSION + gate_idx;
                            if param_idx < parameters.len() {
                                let mut gate = OperatorMatrix::identity();
                                let angle = parameters[param_idx];
                                let cos_val = angle.cos();
                                let sin_val = angle.sin();

                                gate[(i, i)] = Complex64::new(cos_val, 0.0);
                                gate[(i, j)] = Complex64::new(0.0, -sin_val);
                                gate[(j, i)] = Complex64::new(0.0, -sin_val);
                                gate[(j, j)] = Complex64::new(cos_val, 0.0);

                                let operator = QuantumOperator::from_matrix(gate);
                                current_state = current_state.apply(&operator);
                            }
                            gate_idx += 1;
                        }
                    }
                }
            }
        }

        current_state
    }

    fn num_parameters(&self) -> usize {
        (METATRON_DIMENSION + self.num_entangling_gates()) * self.depth
    }

    fn ansatz_type(&self) -> AnsatzType {
        AnsatzType::Metatron
    }

    fn depth(&self) -> usize {
        self.depth
    }
}

// Implement Ansatz for Box<dyn Ansatz> to allow polymorphic usage
impl Ansatz for Box<dyn Ansatz> {
    fn apply(&self, state: &QuantumState, parameters: &[f64]) -> QuantumState {
        (**self).apply(state, parameters)
    }

    fn num_parameters(&self) -> usize {
        (**self).num_parameters()
    }

    fn ansatz_type(&self) -> AnsatzType {
        (**self).ansatz_type()
    }

    fn depth(&self) -> usize {
        (**self).depth()
    }
}

/// Factory function to create ansatz instances
pub fn create_ansatz(ansatz_type: AnsatzType, depth: usize) -> Box<dyn Ansatz> {
    match ansatz_type {
        AnsatzType::HardwareEfficient => Box::new(HardwareEfficientAnsatz::new(depth)),
        AnsatzType::EfficientSU2 => Box::new(EfficientSU2Ansatz::new(depth)),
        AnsatzType::Metatron => Box::new(MetatronAnsatz::new(depth)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_efficient_parameter_count() {
        let ansatz = HardwareEfficientAnsatz::new(3);
        assert_eq!(ansatz.num_parameters(), 2 * METATRON_DIMENSION * 3);
    }

    #[test]
    fn test_efficient_su2_parameter_count() {
        let ansatz = EfficientSU2Ansatz::new(2);
        assert_eq!(ansatz.num_parameters(), 3 * METATRON_DIMENSION * 2);
    }

    #[test]
    fn test_ansatz_preserves_normalization() {
        let ansatz = HardwareEfficientAnsatz::new(2);
        let state = QuantumState::uniform_superposition();
        let params = vec![0.1; ansatz.num_parameters()];

        let new_state = ansatz.apply(&state, &params);
        assert!(new_state.is_normalized(1e-10));
    }
}
