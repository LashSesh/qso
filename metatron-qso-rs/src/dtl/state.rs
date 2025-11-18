use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Enumeration of the fundamental tripolar state classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TripolarStateKind {
    L0,
    L1,
    Ld,
}

/// Dynamic Tripolar Logic state representation.
#[derive(Clone)]
pub struct DTLState {
    kind: TripolarStateKind,
    value: Option<f64>,
    trajectory: Option<Arc<dyn Fn(f64) -> f64 + Send + Sync>>, // x: ℝ → [0,1]
}

impl fmt::Debug for DTLState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            TripolarStateKind::L0 | TripolarStateKind::L1 => f
                .debug_struct("DTLState")
                .field("kind", &self.kind)
                .field("value", &self.value)
                .finish(),
            TripolarStateKind::Ld => f
                .debug_struct("DTLState")
                .field("kind", &self.kind)
                .field("has_trajectory", &self.trajectory.is_some())
                .finish(),
        }
    }
}

impl DTLState {
    /// Static null-pole state L0.
    pub fn l0() -> Self {
        Self {
            kind: TripolarStateKind::L0,
            value: Some(0.0),
            trajectory: None,
        }
    }

    /// Static one-pole state L1.
    pub fn l1() -> Self {
        Self {
            kind: TripolarStateKind::L1,
            value: Some(1.0),
            trajectory: None,
        }
    }

    /// Dynamic oscillatory LD state defined by sine trajectory.
    pub fn ld_oscillatory(frequency: f64, phase: f64, amplitude: f64, offset: f64) -> Self {
        Self::ld_from_function(move |t: f64| {
            let value =
                offset + amplitude * (2.0 * std::f64::consts::PI * frequency * t + phase).sin();
            value.clamp(0.0, 1.0)
        })
    }

    /// Dynamic LD state derived from a phase trajectory φ(t).
    pub fn ld_from_phase<F>(phase: F) -> Self
    where
        F: Fn(f64) -> f64 + Send + Sync + 'static,
    {
        Self::ld_from_function(move |t: f64| {
            let phi = phase(t);
            (1.0 + phi.cos()) / 2.0
        })
    }

    /// Create LD state from an arbitrary trajectory x: ℝ → \[0,1\].
    pub fn ld_from_function<F>(trajectory: F) -> Self
    where
        F: Fn(f64) -> f64 + Send + Sync + 'static,
    {
        Self {
            kind: TripolarStateKind::Ld,
            value: None,
            trajectory: Some(Arc::new(trajectory)),
        }
    }

    /// Evaluate state at time t.
    pub fn evaluate(&self, t: f64) -> f64 {
        match self.kind {
            TripolarStateKind::L0 | TripolarStateKind::L1 => self.value.unwrap(),
            TripolarStateKind::Ld => self
                .trajectory
                .as_ref()
                .map(|f| f(t).clamp(0.0, 1.0))
                .unwrap_or(0.5),
        }
    }

    /// Access state classification.
    pub fn kind(&self) -> TripolarStateKind {
        self.kind
    }
}

/// Information-theoretic properties of tripolar logic.
pub struct TripolarInformationTheory;

impl TripolarInformationTheory {
    /// Tripolar channel capacity C_tri = log₂(3).
    pub fn channel_capacity_tripolar() -> f64 {
        3f64.log2()
    }

    /// Binary channel capacity C_bin = 1.
    pub fn channel_capacity_binary() -> f64 {
        1.0
    }

    /// Relative advantage (C_tri - C_bin) / C_bin.
    pub fn relative_advantage() -> f64 {
        (Self::channel_capacity_tripolar() - Self::channel_capacity_binary())
            / Self::channel_capacity_binary()
    }

    /// Shannon entropy of a discrete distribution.
    pub fn entropy(probabilities: &[f64]) -> f64 {
        probabilities
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.log2())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oscillatory_state_clamps_values() {
        let state = DTLState::ld_oscillatory(1.0, 0.0, 2.0, 0.5);
        let value = state.evaluate(0.25);
        assert!((0.0..=1.0).contains(&value));
    }

    #[test]
    fn tripolar_capacity_advantage() {
        let advantage = TripolarInformationTheory::relative_advantage();
        assert!(advantage > 0.5);
    }
}
