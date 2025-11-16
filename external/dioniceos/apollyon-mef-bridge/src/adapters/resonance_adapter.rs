//! Resonance Bridge: Connect APOLLYON ResonanceField with MEF Proof-of-Resonance
//!
//! This bridge connects APOLLYON's ResonanceField dynamics with MEF's
//! Proof-of-Resonance validation system and Merkaba Gate evaluation logic.
//!
//! # PoR Components
//! - delta_pi: Path invariance (Euclidean distance in 5D space)
//! - phi: Alignment (resonance field modulation)
//! - delta_v: Lyapunov delta (energy/norm change)
//! - por_valid: Overall validity flag
//!
//! # Gate Logic
//! FIRE ⟺ (PoR = valid) ∧ (ΔPI ≤ ε) ∧ (Φ ≥ φ_threshold) ∧ (ΔV < 0)

use bridge::ResonanceField;
use core_5d::State5D;
use mef_schemas::GateDecision;
use serde::{Deserialize, Serialize};

/// Simplified Proof-of-Resonance data structure for bridge integration
///
/// This structure captures the essential PoR metrics needed for gate evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProofOfResonanceData {
    /// Path invariance (Euclidean distance in 5D)
    pub delta_pi: f64,

    /// Alignment (resonance field modulation value)
    pub phi: f64,

    /// Lyapunov delta (energy change)
    pub delta_v: f64,

    /// Overall proof validity
    pub por_valid: bool,
}

impl ProofOfResonanceData {
    /// Create a new PoR data structure
    pub fn new(delta_pi: f64, phi: f64, delta_v: f64, por_valid: bool) -> Self {
        Self {
            delta_pi,
            phi,
            delta_v,
            por_valid,
        }
    }
}

impl Default for ProofOfResonanceData {
    fn default() -> Self {
        Self {
            delta_pi: 0.0,
            phi: 1.0,
            delta_v: 0.0,
            por_valid: false,
        }
    }
}

/// Bridge between APOLLYON ResonanceField and MEF Proof-of-Resonance
///
/// This bridge computes PoR metrics from state transitions and evaluates
/// the Merkaba Gate decision logic.
pub struct ResonanceBridge;

impl ResonanceBridge {
    /// Compute Proof-of-Resonance from APOLLYON state transition
    ///
    /// # Arguments
    /// * `field` - The resonance field providing modulation
    /// * `state_prev` - Previous 5D state
    /// * `state_curr` - Current 5D state
    /// * `t` - Current time for field evaluation
    ///
    /// # Returns
    /// ProofOfResonanceData containing all PoR metrics
    ///
    /// # PoR Components
    /// 1. **delta_pi** (Path Invariance): Euclidean distance in 5D space
    ///    - Measures how far the state has moved
    ///    - Lower values indicate stable transitions
    ///
    /// 2. **phi** (Alignment): Average resonance field modulation
    ///    - Computed by averaging modulation across all node pairs
    ///    - Higher values indicate strong resonance
    ///
    /// 3. **delta_v** (Lyapunov Delta): Change in state norm (energy)
    ///    - Negative values indicate energy dissipation (stability)
    ///    - Positive values indicate energy injection (instability)
    ///
    /// 4. **por_valid**: Basic validity checks
    ///    - Checks for finite values and reasonable ranges
    pub fn compute_proof(
        field: &dyn ResonanceField,
        state_prev: &State5D,
        state_curr: &State5D,
        t: f64,
    ) -> ProofOfResonanceData {
        // 1. Path Invariance: Euclidean distance in 5D
        let delta_pi = Self::compute_path_invariance(state_prev, state_curr);

        // 2. Alignment: Average resonance field modulation
        // Sample modulation across all 5D node pairs
        let mut phi_sum = 0.0;
        let mut count = 0.0;
        for i in 0..5 {
            for j in 0..5 {
                if i != j {
                    phi_sum += field.modulation(t, i, j);
                    count += 1.0;
                }
            }
        }
        let phi = if count > 0.0 {
            phi_sum / count
        } else {
            1.0 // Default neutral value
        };

        // 3. Lyapunov Delta: Change in state norm (energy)
        let delta_v = Self::compute_lyapunov_delta(state_prev, state_curr);

        // 4. PoR Validity: Check basic constraints
        let por_valid = phi.is_finite()
            && phi > 0.0
            && delta_pi.is_finite()
            && delta_pi < 100.0  // Reasonable bound
            && delta_v.is_finite();

        ProofOfResonanceData {
            delta_pi,
            phi,
            delta_v,
            por_valid,
        }
    }

    /// Evaluate Merkaba Gate decision based on PoR metrics
    ///
    /// # Gate Logic
    /// FIRE ⟺ (PoR = valid) ∧ (ΔPI ≤ ε) ∧ (Φ ≥ φ_threshold) ∧ (ΔV < 0)
    ///
    /// Where:
    /// - ε (epsilon) = 0.1 (default path invariance threshold)
    /// - φ_threshold = 0.5 (default alignment threshold)
    ///
    /// # Arguments
    /// * `field` - Resonance field
    /// * `state_prev` - Previous state
    /// * `state_curr` - Current state
    /// * `t` - Current time
    ///
    /// # Returns
    /// GateDecision (FIRE or HOLD)
    pub fn evaluate_gate(
        field: &dyn ResonanceField,
        state_prev: &State5D,
        state_curr: &State5D,
        t: f64,
    ) -> GateDecision {
        Self::evaluate_gate_with_thresholds(field, state_prev, state_curr, t, 0.1, 0.5)
    }

    /// Evaluate gate with custom thresholds
    ///
    /// # Arguments
    /// * `epsilon` - Path invariance threshold
    /// * `phi_threshold` - Alignment threshold
    pub fn evaluate_gate_with_thresholds(
        field: &dyn ResonanceField,
        state_prev: &State5D,
        state_curr: &State5D,
        t: f64,
        epsilon: f64,
        phi_threshold: f64,
    ) -> GateDecision {
        let proof = Self::compute_proof(field, state_prev, state_curr, t);

        if proof.por_valid
            && proof.delta_pi <= epsilon
            && proof.phi >= phi_threshold
            && proof.delta_v < 0.0
        {
            GateDecision::FIRE
        } else {
            GateDecision::HOLD
        }
    }

    /// Compute path invariance (Euclidean distance in 5D)
    fn compute_path_invariance(prev: &State5D, curr: &State5D) -> f64 {
        let mut sum_sq = 0.0;
        for i in 0..5 {
            let diff = curr.get(i) - prev.get(i);
            sum_sq += diff * diff;
        }
        sum_sq.sqrt()
    }

    /// Compute Lyapunov delta (change in state norm)
    fn compute_lyapunov_delta(prev: &State5D, curr: &State5D) -> f64 {
        let v_curr = curr.norm();
        let v_prev = prev.norm();
        v_curr - v_prev
    }
}

impl Default for ResonanceBridge {
    fn default() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bridge::ConstantResonanceField;

    #[test]
    fn test_proof_computation_basic() {
        let field = ConstantResonanceField::new(0.8);
        let prev = State5D::new(1.0, 0.0, 0.0, 0.0, 0.0);
        let curr = State5D::new(1.01, 0.0, 0.0, 0.0, 0.0);

        let proof = ResonanceBridge::compute_proof(&field, &prev, &curr, 0.0);

        assert!(proof.por_valid);
        assert!(proof.delta_pi < 0.1); // Small change
        assert!((proof.phi - 0.8).abs() < 1e-10); // Constant field (with floating point tolerance)
        assert!(proof.delta_v.abs() < 0.1); // Small energy change
    }

    #[test]
    fn test_proof_path_invariance() {
        let field = ConstantResonanceField::new(1.0);
        let prev = State5D::new(0.0, 0.0, 0.0, 0.0, 0.0);
        let curr = State5D::new(1.0, 0.0, 0.0, 0.0, 0.0);

        let proof = ResonanceBridge::compute_proof(&field, &prev, &curr, 0.0);

        // Distance should be 1.0 (moved along x-axis only)
        assert!((proof.delta_pi - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_proof_lyapunov_delta() {
        let field = ConstantResonanceField::new(1.0);

        // Test decreasing norm (energy dissipation)
        let prev = State5D::new(2.0, 0.0, 0.0, 0.0, 0.0);
        let curr = State5D::new(1.0, 0.0, 0.0, 0.0, 0.0);

        let proof = ResonanceBridge::compute_proof(&field, &prev, &curr, 0.0);

        // Norm decreased from 2.0 to 1.0, so delta_v should be -1.0
        assert!((proof.delta_v + 1.0).abs() < 1e-6);
        assert!(proof.delta_v < 0.0);
    }

    #[test]
    fn test_gate_fires_on_valid_transition() {
        let field = ConstantResonanceField::new(0.8);

        // Small change with decreasing norm
        let prev = State5D::new(1.0, 0.0, 0.0, 0.0, 0.0);
        let curr = State5D::new(0.99, 0.0, 0.0, 0.0, 0.0);

        let decision = ResonanceBridge::evaluate_gate(&field, &prev, &curr, 0.0);

        // Should FIRE: valid, small delta_pi, good phi, negative delta_v
        assert_eq!(decision, GateDecision::FIRE);
    }

    #[test]
    fn test_gate_holds_on_large_change() {
        let field = ConstantResonanceField::new(0.8);

        // Large change (violates path invariance threshold)
        let prev = State5D::new(0.0, 0.0, 0.0, 0.0, 0.0);
        let curr = State5D::new(1.0, 1.0, 1.0, 1.0, 1.0);

        let decision = ResonanceBridge::evaluate_gate(&field, &prev, &curr, 0.0);

        // Should HOLD: delta_pi too large
        assert_eq!(decision, GateDecision::HOLD);
    }

    #[test]
    fn test_gate_holds_on_energy_increase() {
        let field = ConstantResonanceField::new(0.8);

        // Small change but increasing norm
        let prev = State5D::new(1.0, 0.0, 0.0, 0.0, 0.0);
        let curr = State5D::new(1.01, 0.0, 0.0, 0.0, 0.0);

        let decision = ResonanceBridge::evaluate_gate(&field, &prev, &curr, 0.0);

        // Should HOLD: delta_v positive (energy increasing)
        assert_eq!(decision, GateDecision::HOLD);
    }

    #[test]
    fn test_gate_holds_on_low_alignment() {
        let field = ConstantResonanceField::new(0.3); // Low phi

        // Good transition otherwise
        let prev = State5D::new(1.0, 0.0, 0.0, 0.0, 0.0);
        let curr = State5D::new(0.99, 0.0, 0.0, 0.0, 0.0);

        let decision = ResonanceBridge::evaluate_gate(&field, &prev, &curr, 0.0);

        // Should HOLD: phi below threshold (0.5)
        assert_eq!(decision, GateDecision::HOLD);
    }

    #[test]
    fn test_gate_with_custom_thresholds() {
        let field = ConstantResonanceField::new(0.4);
        let prev = State5D::new(1.0, 0.0, 0.0, 0.0, 0.0);
        let curr = State5D::new(0.99, 0.0, 0.0, 0.0, 0.0);

        // With default thresholds (phi_threshold=0.5), should HOLD
        let decision1 = ResonanceBridge::evaluate_gate(&field, &prev, &curr, 0.0);
        assert_eq!(decision1, GateDecision::HOLD);

        // With lower threshold (phi_threshold=0.3), should FIRE
        let decision2 =
            ResonanceBridge::evaluate_gate_with_thresholds(&field, &prev, &curr, 0.0, 0.1, 0.3);
        assert_eq!(decision2, GateDecision::FIRE);
    }

    #[test]
    fn test_por_data_default() {
        let por = ProofOfResonanceData::default();
        assert_eq!(por.delta_pi, 0.0);
        assert_eq!(por.phi, 1.0);
        assert_eq!(por.delta_v, 0.0);
        assert!(!por.por_valid);
    }

    #[test]
    fn test_por_data_serialization() {
        let por = ProofOfResonanceData::new(0.05, 0.8, -0.01, true);

        // Test serialization roundtrip
        let json = serde_json::to_string(&por).unwrap();
        let por_back: ProofOfResonanceData = serde_json::from_str(&json).unwrap();

        assert_eq!(por, por_back);
    }

    #[test]
    fn test_zero_state_transition() {
        let field = ConstantResonanceField::new(1.0);
        let zero = State5D::zero();

        let proof = ResonanceBridge::compute_proof(&field, &zero, &zero, 0.0);

        // No change: delta_pi and delta_v should be 0
        assert_eq!(proof.delta_pi, 0.0);
        assert_eq!(proof.delta_v, 0.0);
    }
}
