//! Hyperbion Layer - Morphodynamic coupling between 4D flow and 5D field
//!
//! As per Section 2 of the specification:
//! H(x,t) = α·Φ(x,t) + β·μ(x,t)
//!
//! Where:
//! - Φ: Phase/Resonance field
//! - μ: Morphodynamic growth/damping function
//! - α, β: Modulation constants

use super::types::{HyperbionFields, State5D};

/// Hyperbion layer configuration
#[derive(Debug, Clone, Copy)]
pub struct Hyperbion {
    pub alpha: f64, // Phase/resonance modulation
    pub beta: f64,  // Morphodynamic modulation
}

impl Hyperbion {
    /// Create new Hyperbion with default parameters
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }

    /// Create with custom modulation constants
    pub fn with_params(alpha: f64, beta: f64) -> Self {
        Self { alpha, beta }
    }

    /// Absorb 5D states and compute Hyperbion fields
    ///
    /// Computes:
    /// - Φ: Phase/Resonance field from state oscillations
    /// - μ: Morphodynamic field from state variance
    ///
    /// # Arguments
    /// * `states` - Batch of 5D states
    ///
    /// # Returns
    /// Combined Hyperbion fields (Φ, μ)
    pub fn absorption(&self, states: &[State5D]) -> HyperbionFields {
        if states.is_empty() {
            return HyperbionFields::zero();
        }

        // Compute phase/resonance field Φ from omega oscillations
        let phi = self.compute_resonance_field(states);

        // Compute morphodynamic field μ from spatial variance
        let mu = self.compute_morphodynamic_field(states);

        HyperbionFields::new(phi, mu)
    }

    /// Compute resonance field Φ from state oscillations
    fn compute_resonance_field(&self, states: &[State5D]) -> f64 {
        if states.is_empty() {
            return 0.0;
        }

        // Average omega (temporal phase) weighted by psi (semantic weight)
        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;

        for state in states {
            let weight = state.psi.abs() + 1e-10; // avoid division by zero
            weighted_sum += state.omega * weight;
            weight_total += weight;
        }

        weighted_sum / weight_total
    }

    /// Compute morphodynamic field μ from spatial variance
    fn compute_morphodynamic_field(&self, states: &[State5D]) -> f64 {
        if states.is_empty() {
            return 0.0;
        }

        // Compute variance in spatial coordinates (x, y, z)
        let n = states.len() as f64;
        
        let mean_x = states.iter().map(|s| s.x).sum::<f64>() / n;
        let mean_y = states.iter().map(|s| s.y).sum::<f64>() / n;
        let mean_z = states.iter().map(|s| s.z).sum::<f64>() / n;

        let var_x = states.iter().map(|s| (s.x - mean_x).powi(2)).sum::<f64>() / n;
        let var_y = states.iter().map(|s| (s.y - mean_y).powi(2)).sum::<f64>() / n;
        let var_z = states.iter().map(|s| (s.z - mean_z).powi(2)).sum::<f64>() / n;

        // Return average variance (simplified morphodynamic measure)
        (var_x + var_y + var_z) / 3.0
    }

    /// Evaluate combined Hyperbion function: H(x,t) = α·Φ + β·μ
    pub fn evaluate(&self, fields: HyperbionFields) -> f64 {
        self.alpha * fields.phi + self.beta * fields.mu
    }
}

impl Default for Hyperbion {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperbion_creation() {
        let h = Hyperbion::new();
        assert_eq!(h.alpha, 1.0);
        assert_eq!(h.beta, 1.0);
    }

    #[test]
    fn test_custom_params() {
        let h = Hyperbion::with_params(2.0, 0.5);
        assert_eq!(h.alpha, 2.0);
        assert_eq!(h.beta, 0.5);
    }

    #[test]
    fn test_absorption_empty() {
        let h = Hyperbion::new();
        let fields = h.absorption(&[]);
        assert_eq!(fields.phi, 0.0);
        assert_eq!(fields.mu, 0.0);
    }

    #[test]
    fn test_absorption_single_state() {
        let h = Hyperbion::new();
        let state = State5D::new(1.0, 2.0, 3.0, 0.5, 10.0);
        let fields = h.absorption(&[state]);
        
        // With single state, phi should be omega, mu should be 0
        assert_eq!(fields.phi, 10.0);
        assert_eq!(fields.mu, 0.0);
    }

    #[test]
    fn test_resonance_field_computation() {
        let h = Hyperbion::new();
        let states = vec![
            State5D::new(0.0, 0.0, 0.0, 1.0, 10.0),
            State5D::new(0.0, 0.0, 0.0, 1.0, 20.0),
        ];
        let fields = h.absorption(&states);
        
        // Average omega with equal weights
        assert!((fields.phi - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_morphodynamic_field_computation() {
        let h = Hyperbion::new();
        let states = vec![
            State5D::new(0.0, 0.0, 0.0, 1.0, 0.0),
            State5D::new(2.0, 0.0, 0.0, 1.0, 0.0),
        ];
        let fields = h.absorption(&states);
        
        // Variance in x dimension
        assert!(fields.mu > 0.0);
    }

    #[test]
    fn test_evaluate_function() {
        let h = Hyperbion::with_params(2.0, 3.0);
        let fields = HyperbionFields::new(5.0, 7.0);
        let result = h.evaluate(fields);
        
        // H = 2.0*5.0 + 3.0*7.0 = 10 + 21 = 31
        assert_eq!(result, 31.0);
    }
}
