//! Spectral Adapter: Convert spectral analysis between APOLLYON and MEF
//!
//! This adapter provides conversion between:
//! - APOLLYON's SpectralAnalyzer output (entropy, centroids, frequencies)
//! - MEF's SpectralSignature (psi, rho, omega)
//!
//! The mapping strategy:
//! - psi (ψ): Phase alignment from spectral centroids
//! - rho (ρ): Resonance from inverse entropy (1 - normalized_entropy)
//! - omega (ω): Oscillation frequency from dominant frequency

use mef_schemas::SpectralSignature;

/// Adapter for converting spectral analysis between systems
pub struct SpectralAdapter;

impl SpectralAdapter {
    /// Convert entropy, centroids, and frequency to MEF SpectralSignature
    ///
    /// # Arguments
    /// * `entropy` - Spectral entropy from APOLLYON analysis
    /// * `centroids` - Spectral centroids (typically one per component)
    /// * `dominant_freq` - Dominant frequency from spectral analysis
    ///
    /// # Mapping
    /// - `psi`: First centroid value (or 0.5 if empty)
    /// - `rho`: 1.0 - clamped_entropy (inverse relationship)
    /// - `omega`: Dominant frequency
    ///
    /// # Example
    /// ```
    /// use apollyon_mef_bridge::SpectralAdapter;
    ///
    /// let sig = SpectralAdapter::features_to_signature(0.3, &[0.5], 2.1);
    /// assert_eq!(sig.psi, 0.5);   // From centroids
    /// assert_eq!(sig.rho, 0.7);   // 1 - 0.3
    /// assert_eq!(sig.omega, 2.1); // Frequency
    /// ```
    pub fn features_to_signature(
        entropy: f64,
        centroids: &[f64],
        dominant_freq: f64,
    ) -> SpectralSignature {
        // Phase alignment from first spectral centroid
        let psi = centroids.first().copied().unwrap_or(0.5);

        // Resonance from inverse entropy (clamped to [0,1])
        // Higher entropy = lower resonance
        let clamped_entropy = entropy.min(1.0).max(0.0);
        let rho = 1.0 - clamped_entropy;

        // Oscillation frequency (use as-is)
        let omega = dominant_freq;

        SpectralSignature { psi, rho, omega }
    }

    /// Create a simple spectral signature from basic trajectory statistics
    ///
    /// This is a simplified version for when full spectral analysis isn't available.
    /// Uses simple statistical measures from the trajectory.
    ///
    /// # Arguments
    /// * `mean_value` - Mean value across trajectory (used for psi)
    /// * `variance` - Variance (low variance = high resonance)
    /// * `oscillation_count` - Number of oscillations (for omega)
    /// * `trajectory_length` - Total number of points
    pub fn from_trajectory_stats(
        mean_value: f64,
        variance: f64,
        oscillation_count: f64,
        trajectory_length: f64,
    ) -> SpectralSignature {
        // Normalize mean to [0, 1] range using tanh
        let psi = (mean_value.tanh() + 1.0) / 2.0;

        // Low variance = high resonance
        // Use exponential decay: rho = exp(-variance)
        let rho = (-variance).exp().min(1.0).max(0.0);

        // Oscillation frequency = oscillations per unit time
        let omega = if trajectory_length > 0.0 {
            oscillation_count / trajectory_length
        } else {
            0.0
        };

        SpectralSignature { psi, rho, omega }
    }

    /// Convert spectral signature to a readable description
    pub fn describe(sig: &SpectralSignature) -> String {
        format!(
            "SpectralSignature {{ psi: {:.4} (phase), rho: {:.4} (resonance), omega: {:.4} (frequency) }}",
            sig.psi, sig.rho, sig.omega
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_features_to_signature_basic() {
        let sig = SpectralAdapter::features_to_signature(
            0.3,    // entropy
            &[0.5], // centroids
            2.1,    // dominant_freq
        );

        assert_eq!(sig.psi, 0.5);
        assert_eq!(sig.rho, 0.7); // 1 - 0.3
        assert_eq!(sig.omega, 2.1);
    }

    #[test]
    fn test_entropy_clamping_high() {
        // High entropy (>1.0) should be clamped to 1.0
        let sig = SpectralAdapter::features_to_signature(1.5, &[0.0], 0.0);
        assert_eq!(sig.rho, 0.0); // 1 - 1.0 = 0
    }

    #[test]
    fn test_entropy_clamping_low() {
        // Low entropy should give high resonance
        let sig = SpectralAdapter::features_to_signature(0.1, &[0.0], 0.0);
        assert_eq!(sig.rho, 0.9); // 1 - 0.1
    }

    #[test]
    fn test_entropy_negative() {
        // Negative entropy should be clamped to 0
        let sig = SpectralAdapter::features_to_signature(-0.5, &[0.0], 0.0);
        assert_eq!(sig.rho, 1.0); // 1 - 0.0
    }

    #[test]
    fn test_empty_centroids() {
        // Empty centroids should default to 0.5
        let sig = SpectralAdapter::features_to_signature(0.0, &[], 0.0);
        assert_eq!(sig.psi, 0.5);
    }

    #[test]
    fn test_multiple_centroids() {
        // Should use first centroid
        let sig = SpectralAdapter::features_to_signature(0.0, &[0.3, 0.7, 0.9], 0.0);
        assert_eq!(sig.psi, 0.3);
    }

    #[test]
    fn test_from_trajectory_stats() {
        let sig = SpectralAdapter::from_trajectory_stats(
            0.0,   // mean_value
            0.1,   // variance
            5.0,   // oscillation_count
            100.0, // trajectory_length
        );

        assert_eq!(sig.psi, 0.5); // tanh(0) = 0, normalized to 0.5
        assert!(sig.rho > 0.9); // exp(-0.1) ≈ 0.905
        assert_eq!(sig.omega, 0.05); // 5/100
    }

    #[test]
    fn test_from_trajectory_stats_high_variance() {
        let sig = SpectralAdapter::from_trajectory_stats(
            1.0,   // mean
            5.0,   // high variance
            10.0,  // oscillations
            100.0, // length
        );

        assert!(sig.psi > 0.5); // Positive mean
        assert!(sig.rho < 0.1); // exp(-5) ≈ 0.0067
        assert_eq!(sig.omega, 0.1);
    }

    #[test]
    fn test_describe() {
        let sig = SpectralSignature {
            psi: 0.5,
            rho: 0.7,
            omega: 2.1,
        };

        let desc = SpectralAdapter::describe(&sig);
        assert!(desc.contains("0.5000"));
        assert!(desc.contains("0.7000"));
        assert!(desc.contains("2.1000"));
    }

    #[test]
    fn test_inverse_entropy_resonance_relationship() {
        // Low entropy = high resonance
        let sig_low_entropy = SpectralAdapter::features_to_signature(0.2, &[0.5], 0.0);
        let sig_high_entropy = SpectralAdapter::features_to_signature(0.8, &[0.5], 0.0);

        assert!(sig_low_entropy.rho > sig_high_entropy.rho);
        assert!((sig_low_entropy.rho - 0.8).abs() < 1e-10);
        assert!((sig_high_entropy.rho - 0.2).abs() < 1e-10);
    }
}
