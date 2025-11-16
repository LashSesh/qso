//! Stability Analysis and Lyapunov Exponents
//!
//! Section 5 of the specification - fixed point detection, stability, and chaos analysis.

use crate::dynamics::VectorField;
use crate::state::State5D;
use nalgebra::DMatrix;

/// Fixed point finder
///
/// Searches for states where F(σ*) = 0
pub struct FixedPointFinder {
    pub vector_field: VectorField,
    pub tolerance: f64,
}

impl FixedPointFinder {
    pub fn new(vector_field: VectorField, tolerance: f64) -> Self {
        FixedPointFinder {
            vector_field,
            tolerance,
        }
    }

    /// Check if a state is a fixed point
    pub fn is_fixed_point(&self, state: &State5D) -> bool {
        let f = self.vector_field.evaluate(state);
        f.norm() < self.tolerance
    }
}

/// Stability analyzer
///
/// Analyzes stability of fixed points using eigenvalues of Jacobian
pub struct StabilityAnalyzer;

impl StabilityAnalyzer {
    /// Compute eigenvalues of Jacobian matrix
    ///
    /// Returns real parts of eigenvalues (sorted descending)
    pub fn eigenvalues(jacobian: &[[f64; 5]; 5]) -> Vec<f64> {
        // Convert to DMatrix for eigenvalue computation
        let mut matrix = DMatrix::zeros(5, 5);
        for i in 0..5 {
            for j in 0..5 {
                matrix[(i, j)] = jacobian[i][j];
            }
        }

        // Compute eigenvalues
        let eigenvalues = matrix.complex_eigenvalues();
        let mut real_parts: Vec<f64> = eigenvalues.iter().map(|c| c.re).collect();
        real_parts.sort_by(|a, b| b.partial_cmp(a).unwrap());
        real_parts
    }

    /// Determine stability from eigenvalues
    ///
    /// Returns:
    /// - Stable if all ℜ(λᵢ) < 0
    /// - Unstable if any ℜ(λᵢ) > 0
    /// - Marginal otherwise
    pub fn classify_stability(eigenvalues: &[f64]) -> StabilityType {
        let max_real = eigenvalues.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if max_real < -1e-6 {
            StabilityType::Stable
        } else if max_real > 1e-6 {
            StabilityType::Unstable
        } else {
            StabilityType::Marginal
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilityType {
    Stable,
    Unstable,
    Marginal,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stability_classification() {
        let stable_eigs = vec![-1.0, -2.0, -3.0, -4.0, -5.0];
        assert_eq!(
            StabilityAnalyzer::classify_stability(&stable_eigs),
            StabilityType::Stable
        );

        let unstable_eigs = vec![1.0, -2.0, -3.0, -4.0, -5.0];
        assert_eq!(
            StabilityAnalyzer::classify_stability(&unstable_eigs),
            StabilityType::Unstable
        );
    }
}
