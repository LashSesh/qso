//! MerkabaGateEvent - Gate decision events (FIRE/HOLD)

use serde::{Deserialize, Serialize};

/// Gate decision enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateDecision {
    /// Gate fires - knowledge propagates
    FIRE,

    /// Gate holds - knowledge does not propagate
    HOLD,
}

/// MerkabaGateEvent represents a gate decision event
/// Gate condition: FIRE ⟺ (PoR = valid) ∧ (ΔPI ≤ ε) ∧ (Φ ≥ φ) ∧ (ΔV < 0)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MerkabaGateEvent {
    /// Event identifier
    pub event_id: String,

    /// Knowledge object ID
    pub mef_id: String,

    /// Gate decision
    pub decision: GateDecision,

    /// Path invariance metric (ΔPI)
    pub path_invariance: f64,

    /// Alignment metric (Φ)
    pub alignment: f64,

    /// Lyapunov metric (ΔV)
    pub lyapunov_delta: f64,

    /// Proof of Resonance validity
    pub por_valid: bool,

    /// Timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

impl MerkabaGateEvent {
    /// Create a new MerkabaGateEvent
    pub fn new(
        event_id: String,
        mef_id: String,
        path_invariance: f64,
        alignment: f64,
        lyapunov_delta: f64,
        por_valid: bool,
        epsilon: f64,
        phi_threshold: f64,
    ) -> Self {
        // Evaluate gate condition
        let decision = if por_valid
            && path_invariance <= epsilon
            && alignment >= phi_threshold
            && lyapunov_delta < 0.0
        {
            GateDecision::FIRE
        } else {
            GateDecision::HOLD
        };

        Self {
            event_id,
            mef_id,
            decision,
            path_invariance,
            alignment,
            lyapunov_delta,
            por_valid,
            timestamp: None,
        }
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, timestamp: String) -> Self {
        self.timestamp = Some(timestamp);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_fires() {
        let event = MerkabaGateEvent::new(
            "event_001".to_string(),
            "mef_001".to_string(),
            0.01, // ΔPI ≤ ε (0.05)
            0.8,  // Φ ≥ φ (0.7)
            -0.1, // ΔV < 0
            true, // PoR valid
            0.05, // ε threshold
            0.7,  // φ threshold
        );

        assert_eq!(event.decision, GateDecision::FIRE);
    }

    #[test]
    fn test_gate_holds_por_invalid() {
        let event = MerkabaGateEvent::new(
            "event_002".to_string(),
            "mef_002".to_string(),
            0.01,
            0.8,
            -0.1,
            false, // PoR invalid
            0.05,
            0.7,
        );

        assert_eq!(event.decision, GateDecision::HOLD);
    }

    #[test]
    fn test_gate_holds_high_path_invariance() {
        let event = MerkabaGateEvent::new(
            "event_003".to_string(),
            "mef_003".to_string(),
            0.1, // ΔPI > ε
            0.8,
            -0.1,
            true,
            0.05,
            0.7,
        );

        assert_eq!(event.decision, GateDecision::HOLD);
    }
}
