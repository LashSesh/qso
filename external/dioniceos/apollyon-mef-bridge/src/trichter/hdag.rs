//! HDAG Field - Hyperdimensional Acyclic Resonance Grid
//!
//! As per Section 2 of the specification:
//! - Nodes: 5D resonance tensors Tᵢ ∈ ℝ⁵
//! - Edges: Phase-gradient transitions Φᵢⱼ(t)
//! - Acyclicity emerges from phase disalignment

use super::types::{HyperbionFields, State5D};
use std::collections::HashMap;

/// Resonance tensor node in the HDAG
#[derive(Debug, Clone)]
pub struct ResonanceTensor {
    pub id: usize,
    pub tensor: State5D,
    pub resonance: f64, // Accumulated resonance strength
}

impl ResonanceTensor {
    pub fn new(id: usize, tensor: State5D) -> Self {
        Self {
            id,
            tensor,
            resonance: 0.0,
        }
    }
}

/// Phase transition between tensors
#[derive(Debug, Clone)]
pub struct PhaseTransition {
    pub from: usize,
    pub to: usize,
    pub phase_gradient: f64,
    pub coherence: f64, // ∈ [0,1]
}

impl PhaseTransition {
    pub fn new(from: usize, to: usize) -> Self {
        Self {
            from,
            to,
            phase_gradient: 0.0,
            coherence: 0.0,
        }
    }
}

/// HDAG Field - manages the resonance grid
#[derive(Debug)]
pub struct HDAGField {
    pub tensors: HashMap<usize, ResonanceTensor>,
    pub transitions: Vec<PhaseTransition>,
    next_id: usize,
}

impl HDAGField {
    /// Create new empty HDAG field
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            transitions: Vec::new(),
            next_id: 0,
        }
    }

    /// Add a resonance tensor to the field
    pub fn add_tensor(&mut self, tensor: State5D) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.tensors.insert(id, ResonanceTensor::new(id, tensor));
        id
    }

    /// Add a phase transition between tensors
    pub fn add_transition(&mut self, from: usize, to: usize) {
        if self.tensors.contains_key(&from) && self.tensors.contains_key(&to) {
            self.transitions.push(PhaseTransition::new(from, to));
        }
    }

    /// Relax the HDAG field based on Hyperbion fields
    ///
    /// Updates tensor resonances and transition gradients
    pub fn relax(&mut self, fields: HyperbionFields) {
        // Update tensor resonances based on phi (resonance field)
        for tensor in self.tensors.values_mut() {
            // Resonance accumulates based on alignment with global field
            let alignment = (tensor.tensor.omega - fields.phi).cos();
            tensor.resonance += fields.phi * alignment;
        }

        // Update phase gradients based on tensor differences
        for transition in self.transitions.iter_mut() {
            if let (Some(from_t), Some(to_t)) = 
                (self.tensors.get(&transition.from), self.tensors.get(&transition.to)) {
                
                // Phase gradient is difference in omega
                transition.phase_gradient = to_t.tensor.omega - from_t.tensor.omega;
                
                // Coherence based on resonance alignment
                let res_diff = (to_t.resonance - from_t.resonance).abs();
                transition.coherence = (-res_diff).exp();
            }
        }

        // Apply morphodynamic damping
        self.apply_morphodynamic_damping(fields.mu);
    }

    /// Apply morphodynamic damping to transitions
    fn apply_morphodynamic_damping(&mut self, mu: f64) {
        let damping_factor = (-mu.abs() * 0.1).exp();
        
        for transition in self.transitions.iter_mut() {
            transition.coherence *= damping_factor;
        }

        // Prune low-coherence transitions (simplified acyclicity enforcement)
        self.transitions.retain(|t| t.coherence > 0.01);
    }

    /// Compute gradient field ∇Φ at a given position
    ///
    /// Returns a 5D gradient vector for guidance
    pub fn gradient(&self, position: State5D) -> State5D {
        if self.tensors.is_empty() {
            return State5D::zero();
        }

        let mut gradient = State5D::zero();
        let mut total_influence = 0.0;

        // Compute gradient as weighted sum of tensor influences
        for tensor in self.tensors.values() {
            let diff = State5D::new(
                tensor.tensor.x - position.x,
                tensor.tensor.y - position.y,
                tensor.tensor.z - position.z,
                tensor.tensor.psi - position.psi,
                tensor.tensor.omega - position.omega,
            );

            let distance = diff.norm() + 1e-10; // avoid division by zero
            let influence = tensor.resonance / distance.powi(2);

            gradient.x += diff.x * influence;
            gradient.y += diff.y * influence;
            gradient.z += diff.z * influence;
            gradient.psi += diff.psi * influence;
            gradient.omega += diff.omega * influence;

            total_influence += influence;
        }

        // Normalize gradient
        if total_influence > 1e-10 {
            gradient.x /= total_influence;
            gradient.y /= total_influence;
            gradient.z /= total_influence;
            gradient.psi /= total_influence;
            gradient.omega /= total_influence;
        }

        gradient
    }

    /// Get number of tensors
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Get number of active transitions
    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }
}

impl Default for HDAGField {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdag_creation() {
        let hdag = HDAGField::new();
        assert_eq!(hdag.tensor_count(), 0);
        assert_eq!(hdag.transition_count(), 0);
    }

    #[test]
    fn test_add_tensor() {
        let mut hdag = HDAGField::new();
        let tensor = State5D::new(1.0, 2.0, 3.0, 4.0, 5.0);
        let id = hdag.add_tensor(tensor);
        
        assert_eq!(id, 0);
        assert_eq!(hdag.tensor_count(), 1);
    }

    #[test]
    fn test_add_multiple_tensors() {
        let mut hdag = HDAGField::new();
        let id1 = hdag.add_tensor(State5D::zero());
        let id2 = hdag.add_tensor(State5D::zero());
        
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(hdag.tensor_count(), 2);
    }

    #[test]
    fn test_add_transition() {
        let mut hdag = HDAGField::new();
        let id1 = hdag.add_tensor(State5D::zero());
        let id2 = hdag.add_tensor(State5D::zero());
        
        hdag.add_transition(id1, id2);
        assert_eq!(hdag.transition_count(), 1);
    }

    #[test]
    fn test_relax_updates_resonance() {
        let mut hdag = HDAGField::new();
        let tensor = State5D::new(0.0, 0.0, 0.0, 0.0, 10.0);
        hdag.add_tensor(tensor);
        
        let fields = HyperbionFields::new(5.0, 0.1);
        hdag.relax(fields);
        
        // Resonance should be updated
        let t = hdag.tensors.get(&0).unwrap();
        assert!(t.resonance != 0.0);
    }

    #[test]
    fn test_gradient_empty_field() {
        let hdag = HDAGField::new();
        let position = State5D::zero();
        let gradient = hdag.gradient(position);
        
        assert_eq!(gradient, State5D::zero());
    }

    #[test]
    fn test_gradient_single_tensor() {
        let mut hdag = HDAGField::new();
        let tensor = State5D::new(10.0, 0.0, 0.0, 0.0, 0.0);
        let id = hdag.add_tensor(tensor);
        
        // Set some resonance
        hdag.tensors.get_mut(&id).unwrap().resonance = 1.0;
        
        let position = State5D::zero();
        let gradient = hdag.gradient(position);
        
        // Gradient should point toward the tensor
        assert!(gradient.x > 0.0);
    }

    #[test]
    fn test_morphodynamic_damping() {
        let mut hdag = HDAGField::new();
        let id1 = hdag.add_tensor(State5D::zero());
        let id2 = hdag.add_tensor(State5D::new(1.0, 0.0, 0.0, 0.0, 0.0));
        hdag.add_transition(id1, id2);
        
        // Set initial coherence
        hdag.transitions[0].coherence = 0.5;
        
        // Apply strong damping
        let fields = HyperbionFields::new(0.0, 10.0);
        hdag.relax(fields);
        
        // Coherence should decrease
        assert!(hdag.transitions.first().map(|t| t.coherence).unwrap_or(1.0) < 0.5);
    }
}
