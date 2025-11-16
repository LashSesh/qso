//! Spectral Signature (ψ, ρ, ω)
//!
//! The spectral signature captures three fundamental quality metrics:
//! - ψ (psi): Quality / Accuracy
//! - ρ (rho): Stability / Consistency
//! - ω (omega): Efficiency / Performance
//!
//! The **resonance** is the geometric mean: ψ × ρ × ω

use serde::{Deserialize, Serialize};

/// Spectral signature capturing three quality dimensions
///
/// This is the core metric used by TRITON search to evaluate parameter configurations.
/// All values should ideally be in [0, 1] range, though the algorithm handles arbitrary values.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpectralSignature {
    /// ψ (psi): Quality / Accuracy metric
    pub psi: f64,

    /// ρ (rho): Stability / Consistency metric
    pub rho: f64,

    /// ω (omega): Efficiency / Performance metric
    pub omega: f64,
}

impl SpectralSignature {
    /// Create a new spectral signature
    ///
    /// # Arguments
    /// * `psi` - Quality/Accuracy (typically 0.0-1.0)
    /// * `rho` - Stability/Consistency (typically 0.0-1.0)
    /// * `omega` - Efficiency/Performance (typically 0.0-1.0)
    ///
    /// # Example
    /// ```
    /// use metatron_triton::SpectralSignature;
    ///
    /// let sig = SpectralSignature::new(0.9, 0.85, 0.92);
    /// assert!((sig.resonance() - 0.70362).abs() < 0.001);
    /// ```
    pub fn new(psi: f64, rho: f64, omega: f64) -> Self {
        Self { psi, rho, omega }
    }

    /// Compute the resonance: ψ × ρ × ω
    ///
    /// The resonance represents the combined quality across all three dimensions.
    /// Higher resonance indicates better overall performance.
    ///
    /// # Example
    /// ```
    /// use metatron_triton::SpectralSignature;
    ///
    /// let sig = SpectralSignature::new(0.8, 0.9, 0.7);
    /// assert_eq!(sig.resonance(), 0.8 * 0.9 * 0.7);
    /// ```
    pub fn resonance(&self) -> f64 {
        self.psi * self.rho * self.omega
    }

    /// Compute the harmonic mean of the three metrics
    ///
    /// This provides an alternative aggregation that penalizes imbalance
    /// more strongly than the geometric mean.
    pub fn harmonic_mean(&self) -> f64 {
        let sum_inv = 1.0 / self.psi + 1.0 / self.rho + 1.0 / self.omega;
        if sum_inv.is_finite() && sum_inv > 0.0 {
            3.0 / sum_inv
        } else {
            0.0
        }
    }

    /// Check if this signature dominates another (all components >= other)
    pub fn dominates(&self, other: &Self) -> bool {
        self.psi >= other.psi && self.rho >= other.rho && self.omega >= other.omega
    }

    /// Compute the L2 distance to another signature
    pub fn distance(&self, other: &Self) -> f64 {
        let dpsi = self.psi - other.psi;
        let drho = self.rho - other.rho;
        let domega = self.omega - other.omega;
        (dpsi * dpsi + drho * drho + domega * domega).sqrt()
    }

    /// Create a signature with all zeros
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Create a signature with all ones (perfect score)
    pub fn perfect() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }
}

impl Default for SpectralSignature {
    fn default() -> Self {
        Self::zero()
    }
}

impl std::fmt::Display for SpectralSignature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "σ(ψ={:.4}, ρ={:.4}, ω={:.4}) → {:.6}",
            self.psi,
            self.rho,
            self.omega,
            self.resonance()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_resonance() {
        let sig = SpectralSignature::new(0.8, 0.9, 0.7);
        assert_relative_eq!(sig.resonance(), 0.504, epsilon = 1e-10);
    }

    #[test]
    fn test_perfect_signature() {
        let sig = SpectralSignature::perfect();
        assert_eq!(sig.resonance(), 1.0);
    }

    #[test]
    fn test_zero_signature() {
        let sig = SpectralSignature::zero();
        assert_eq!(sig.resonance(), 0.0);
    }

    #[test]
    fn test_dominance() {
        let sig1 = SpectralSignature::new(0.8, 0.9, 0.7);
        let sig2 = SpectralSignature::new(0.7, 0.8, 0.6);
        assert!(sig1.dominates(&sig2));
        assert!(!sig2.dominates(&sig1));
    }

    #[test]
    fn test_distance() {
        let sig1 = SpectralSignature::new(1.0, 0.0, 0.0);
        let sig2 = SpectralSignature::new(0.0, 1.0, 0.0);
        assert_relative_eq!(sig1.distance(&sig2), 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_harmonic_mean() {
        let sig = SpectralSignature::new(0.6, 0.8, 0.9);
        let hm = sig.harmonic_mean();
        // Harmonic mean should be less than arithmetic mean
        let am = (0.6 + 0.8 + 0.9) / 3.0;
        assert!(hm < am);
    }
}
