//! # Merkaba Gate Event Schema
//!
//! Defines the structure for gate decision events in the Solve-Coagula pipeline.
//!
//! ## SPEC-006 Reference
//!
//! From Part 2, Section 1.8:
//! - FIRE ⟺ PoR = valid ∧ (ΔPI ≤ ε) ∧ (Φ ≥ φ) ∧ (ΔV < 0)
//! - On HOLD: skip commit and emit event

use serde::{Deserialize, Serialize};

/// Gate decision checks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateChecks {
    /// Proof of Resonance status
    pub por: String, // "valid" or "invalid"

    /// Path invariance delta: ΔPI = ||Π(vₜ₊₁) - Π(vₜ)||₂
    pub delta_pi: f64,

    /// Alignment metric: Φ = ⟨vₜ₊₁, T(vₜ)⟩ / (||vₜ₊₁||₂ ||T(vₜ)||₂)
    pub phi: f64,

    /// Lyapunov functional change: ΔV = V(vₜ₊₁) - V(vₜ)
    pub delta_v: f64,

    /// Mean curvature index (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mci: Option<f64>,
}

/// Gate decision outcome
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateDecision {
    /// Whether to commit the TIC to the ledger
    pub commit: bool,

    /// Human-readable reason for the decision
    pub reason: String,
}

/// Merkaba Gate event
///
/// Records gate evaluation results for a TIC candidate. On FIRE, the TIC is
/// committed to the ledger. On HOLD, the event is logged but no commit occurs.
///
/// ## JSON Schema
///
/// ```json
/// {
///   "gate_id": "gate-uuid-123",
///   "snapshot_id": "SNAP-xyz",
///   "tic_candidate_id": "TIC-candidate-abc",
///   "checks": {
///     "por": "valid",
///     "delta_pi": 0.001,
///     "phi": 0.95,
///     "delta_v": -0.02,
///     "mci": 0.5
///   },
///   "decision": {
///     "commit": true,
///     "reason": "All gate conditions satisfied (FIRE)"
///   },
///   "timestamp": "2025-10-16T22:00:00Z"
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MerkabaGateEvent {
    /// Unique gate event identifier
    pub gate_id: String,

    /// Associated snapshot ID
    pub snapshot_id: String,

    /// TIC candidate being evaluated
    pub tic_candidate_id: String,

    /// Gate validation checks
    pub checks: GateChecks,

    /// Gate decision (FIRE or HOLD)
    pub decision: GateDecision,

    /// Timestamp of gate evaluation
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl MerkabaGateEvent {
    /// Create a new gate event
    pub fn new(
        gate_id: String,
        snapshot_id: String,
        tic_candidate_id: String,
        checks: GateChecks,
    ) -> Self {
        // Evaluate FIRE condition: PoR=valid ∧ ΔPI≤ε ∧ Φ≥φ ∧ ΔV<0
        // TODO: These thresholds should come from config
        let epsilon_pi = 0.01;
        let phi_threshold = 0.85;

        let por_valid = checks.por == "valid";
        let delta_pi_ok = checks.delta_pi <= epsilon_pi;
        let phi_ok = checks.phi >= phi_threshold;
        let delta_v_ok = checks.delta_v < 0.0;

        let commit = por_valid && delta_pi_ok && phi_ok && delta_v_ok;

        let reason = if commit {
            "All gate conditions satisfied (FIRE)".to_string()
        } else {
            let mut reasons = Vec::new();
            if !por_valid {
                reasons.push("PoR invalid".to_string());
            }
            if !delta_pi_ok {
                reasons.push(format!("ΔPI = {:.6} > {:.6}", checks.delta_pi, epsilon_pi));
            }
            if !phi_ok {
                reasons.push(format!("Φ = {:.6} < {:.6}", checks.phi, phi_threshold));
            }
            if !delta_v_ok {
                reasons.push(format!("ΔV = {:.6} ≥ 0", checks.delta_v));
            }
            format!("HOLD: {}", reasons.join(", "))
        };

        Self {
            gate_id,
            snapshot_id,
            tic_candidate_id,
            checks,
            decision: GateDecision { commit, reason },
            timestamp: chrono::Utc::now(),
        }
    }

    /// Check if gate fired (committed)
    pub fn is_fire(&self) -> bool {
        self.decision.commit
    }

    /// Check if gate held (not committed)
    pub fn is_hold(&self) -> bool {
        !self.decision.commit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_fire() {
        let checks = GateChecks {
            por: "valid".to_string(),
            delta_pi: 0.001,
            phi: 0.95,
            delta_v: -0.02,
            mci: Some(0.5),
        };

        let event = MerkabaGateEvent::new(
            "gate-1".to_string(),
            "snap-1".to_string(),
            "tic-1".to_string(),
            checks,
        );

        assert!(event.is_fire());
        assert!(!event.is_hold());
        assert!(event.decision.reason.contains("FIRE"));
    }

    #[test]
    fn test_gate_hold_por_invalid() {
        let checks = GateChecks {
            por: "invalid".to_string(),
            delta_pi: 0.001,
            phi: 0.95,
            delta_v: -0.02,
            mci: None,
        };

        let event = MerkabaGateEvent::new(
            "gate-2".to_string(),
            "snap-2".to_string(),
            "tic-2".to_string(),
            checks,
        );

        assert!(event.is_hold());
        assert!(event.decision.reason.contains("HOLD"));
        assert!(event.decision.reason.contains("PoR invalid"));
    }

    #[test]
    fn test_gate_hold_delta_v() {
        let checks = GateChecks {
            por: "valid".to_string(),
            delta_pi: 0.001,
            phi: 0.95,
            delta_v: 0.01, // Positive, should HOLD
            mci: None,
        };

        let event = MerkabaGateEvent::new(
            "gate-3".to_string(),
            "snap-3".to_string(),
            "tic-3".to_string(),
            checks,
        );

        assert!(event.is_hold());
        assert!(event.decision.reason.contains("ΔV"));
    }
}
