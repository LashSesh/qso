//! Policy definitions for the 4D-Trichter
//!
//! As per Section 5 of the specification:
//! - Explore: High hebbian, medium decay, low merge/prune
//! - Exploit: Medium hebbian, low decay, high merge, strict phase_lock
//! - Homeostasis: Adaptive regulation targeting density ρ̄

use serde::{Deserialize, Serialize};

/// Policy type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Policy {
    Explore,
    Exploit,
    Homeostasis,
}

/// Policy parameters controlling Funnel dynamics
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PolicyParams {
    pub alpha_hebb: f64,  // Hebbian learning rate
    pub decay: f64,       // Weight decay rate
    pub theta_split: f64, // Split threshold for mass
    pub theta_merge: f64, // Merge threshold for mass
    pub theta_prune: f64, // Prune threshold for weights
    pub phase_lock_strict: bool, // Strict phase locking
    pub target_density: Option<f64>, // Target density for Homeostasis
}

impl PolicyParams {
    /// Create Explore policy parameters
    ///
    /// Characteristics:
    /// - High Hebbian learning (promotes new connections)
    /// - Medium decay (allows exploration)
    /// - Low merge threshold (preserves diversity)
    /// - Very low prune (keeps alternatives)
    pub fn explore() -> Self {
        Self {
            alpha_hebb: 0.5,
            decay: 0.05,
            theta_split: 10.0,
            theta_merge: 0.5,
            theta_prune: 0.01,
            phase_lock_strict: false,
            target_density: None,
        }
    }

    /// Create Exploit policy parameters
    ///
    /// Characteristics:
    /// - Medium Hebbian (consolidates known paths)
    /// - Low decay (preserves strong connections)
    /// - High merge (consolidates similar states)
    /// - Medium prune (removes weak alternatives)
    /// - Strict phase lock (enforces coherence)
    pub fn exploit() -> Self {
        Self {
            alpha_hebb: 0.2,
            decay: 0.01,
            theta_split: 20.0,
            theta_merge: 2.0,
            theta_prune: 0.1,
            phase_lock_strict: true,
            target_density: None,
        }
    }

    /// Create Homeostasis policy parameters
    ///
    /// Characteristics:
    /// - Adaptive Hebbian and decay
    /// - Targets specific node density
    /// - Uses hysteresis for stability
    pub fn homeostasis(target_density: f64) -> Self {
        Self {
            alpha_hebb: 0.3,
            decay: 0.03,
            theta_split: 15.0,
            theta_merge: 1.0,
            theta_prune: 0.05,
            phase_lock_strict: false,
            target_density: Some(target_density),
        }
    }

    /// Adapt parameters based on current density (for Homeostasis)
    ///
    /// Adjusts alpha_hebb and decay to maintain target density
    pub fn adapt_to_density(&mut self, current_density: f64) {
        if let Some(target) = self.target_density {
            let density_ratio = current_density / target;

            if density_ratio > 1.2 {
                // Too dense - increase decay, decrease hebbian
                self.decay *= 1.1;
                self.alpha_hebb *= 0.9;
                self.theta_prune *= 1.1;
            } else if density_ratio < 0.8 {
                // Too sparse - decrease decay, increase hebbian
                self.decay *= 0.9;
                self.alpha_hebb *= 1.1;
                self.theta_prune *= 0.9;
            }

            // Clamp values to reasonable ranges
            self.decay = self.decay.clamp(0.001, 0.2);
            self.alpha_hebb = self.alpha_hebb.clamp(0.05, 0.8);
            self.theta_prune = self.theta_prune.clamp(0.001, 0.3);
        }
    }
}

impl Policy {
    /// Get default parameters for this policy
    pub fn params(&self) -> PolicyParams {
        match self {
            Policy::Explore => PolicyParams::explore(),
            Policy::Exploit => PolicyParams::exploit(),
            Policy::Homeostasis => PolicyParams::homeostasis(1.0),
        }
    }

    /// Get parameters with custom target density (for Homeostasis)
    pub fn params_with_density(&self, density: f64) -> PolicyParams {
        match self {
            Policy::Homeostasis => PolicyParams::homeostasis(density),
            _ => self.params(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explore_params() {
        let params = PolicyParams::explore();
        assert_eq!(params.alpha_hebb, 0.5);
        assert!(!params.phase_lock_strict);
        assert!(params.target_density.is_none());
    }

    #[test]
    fn test_exploit_params() {
        let params = PolicyParams::exploit();
        assert_eq!(params.alpha_hebb, 0.2);
        assert!(params.phase_lock_strict);
        assert!(params.decay < PolicyParams::explore().decay);
    }

    #[test]
    fn test_homeostasis_params() {
        let params = PolicyParams::homeostasis(1.5);
        assert_eq!(params.target_density, Some(1.5));
        assert!(!params.phase_lock_strict);
    }

    #[test]
    fn test_policy_enum() {
        let policy = Policy::Explore;
        let params = policy.params();
        assert_eq!(params.alpha_hebb, 0.5);
    }

    #[test]
    fn test_density_adaptation_high() {
        let mut params = PolicyParams::homeostasis(1.0);
        let original_decay = params.decay;
        let original_hebb = params.alpha_hebb;

        // Current density too high
        params.adapt_to_density(1.5);

        // Should increase decay and decrease hebbian
        assert!(params.decay > original_decay);
        assert!(params.alpha_hebb < original_hebb);
    }

    #[test]
    fn test_density_adaptation_low() {
        let mut params = PolicyParams::homeostasis(1.0);
        let original_decay = params.decay;
        let original_hebb = params.alpha_hebb;

        // Current density too low
        params.adapt_to_density(0.5);

        // Should decrease decay and increase hebbian
        assert!(params.decay < original_decay);
        assert!(params.alpha_hebb > original_hebb);
    }

    #[test]
    fn test_density_adaptation_in_range() {
        let mut params = PolicyParams::homeostasis(1.0);
        let original_decay = params.decay;
        let original_hebb = params.alpha_hebb;

        // Current density within acceptable range
        params.adapt_to_density(1.0);

        // Should remain unchanged
        assert_eq!(params.decay, original_decay);
        assert_eq!(params.alpha_hebb, original_hebb);
    }

    #[test]
    fn test_clamping() {
        let mut params = PolicyParams::homeostasis(1.0);
        
        // Force extreme adaptation
        for _ in 0..100 {
            params.adapt_to_density(0.1);
        }

        // Should be clamped
        assert!(params.alpha_hebb <= 0.8);
        assert!(params.decay >= 0.001);
    }
}
