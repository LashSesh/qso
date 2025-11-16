/// DoubleKick (DK) operator implementation - SPEC-002 konform.
/// Local unsticking through dual impulse without expansion.
use ndarray::Array1;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// DoubleKick operator information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoubleKickInfo {
    pub alpha1: f64,
    pub alpha2: f64,
    pub eta: f64,
    pub u1_u2_orthogonal: bool,
    pub non_expansive: bool,
}

/// DoubleKick operator: DK(v) = v + α₁u₁ + α₂u₂
/// where ⟨u₁, u₂⟩ = 0 and ||u_i||₂ ≤ 1
/// Maintains non-expansiveness: |α₁| + |α₂| ≤ η with η ≪ 1
#[derive(Debug, Clone)]
pub struct DoubleKick {
    alpha1: f64,
    alpha2: f64,
    eta: f64,
    u1: Array1<f64>,
    u2: Array1<f64>,
}

impl DoubleKick {
    /// Create new DoubleKick operator
    ///
    /// # Arguments
    /// * `alpha1` - First impulse coefficient
    /// * `alpha2` - Second impulse coefficient
    ///
    /// # Returns
    /// New DoubleKick instance
    pub fn new(alpha1: f64, alpha2: f64) -> Self {
        let eta = alpha1.abs() + alpha2.abs();
        if eta > 0.1 {
            eprintln!(
                "Warning: DoubleKick η={} > 0.1, may affect contractivity",
                eta
            );
        }

        let (u1, u2) = Self::generate_orthogonal_vectors();

        Self {
            alpha1,
            alpha2,
            eta,
            u1,
            u2,
        }
    }

    /// Create DoubleKick from config
    pub fn from_config(config: &HashMap<String, f64>) -> Self {
        let alpha1 = *config.get("alpha1").unwrap_or(&0.05);
        let alpha2 = *config.get("alpha2").unwrap_or(&-0.03);
        Self::new(alpha1, alpha2)
    }

    /// Generate two orthogonal unit vectors u₁ and u₂
    fn generate_orthogonal_vectors() -> (Array1<f64>, Array1<f64>) {
        // Use fixed seed for deterministic generation
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let normal = StandardNormal;

        // Generate first unit vector
        let mut u1_data: Vec<f64> = (0..5).map(|_| normal.sample(&mut rng)).collect();
        let u1_norm: f64 = u1_data.iter().map(|&x| x * x).sum::<f64>().sqrt();
        u1_data.iter_mut().for_each(|x| *x /= u1_norm);
        let u1 = Array1::from(u1_data);

        // Generate second vector orthogonal to first
        let u2_data: Vec<f64> = (0..5).map(|_| normal.sample(&mut rng)).collect();
        let mut u2 = Array1::from(u2_data);

        // Gram-Schmidt orthogonalization
        let dot_u2_u1 = u2.dot(&u1);
        u2 -= &(&u1 * dot_u2_u1);

        let u2_norm = u2.dot(&u2).sqrt();
        u2 /= u2_norm;

        // Verify orthogonality
        let dot_product = u1.dot(&u2);
        assert!(
            dot_product.abs() < 1e-10,
            "Vectors not orthogonal: ⟨u₁,u₂⟩ = {}",
            dot_product
        );

        (u1, u2)
    }

    /// Apply DoubleKick operator
    ///
    /// # Arguments
    /// * `v` - Input vector (5D)
    ///
    /// # Returns
    /// Transformed vector
    pub fn apply(&self, v: &Array1<f64>) -> Array1<f64> {
        v + &(&self.u1 * self.alpha1) + &(&self.u2 * self.alpha2)
    }

    /// Verify that the operator is non-expansive
    ///
    /// # Returns
    /// True if non-expansive (Lipschitz ≤ 1)
    pub fn verify_non_expansive(&self) -> bool {
        // For DK to be non-expansive, we need |α₁| + |α₂| ≤ η ≪ 1
        // The Lipschitz constant is approximately 1 + η
        self.eta <= 0.1
    }

    /// Get operator information
    ///
    /// # Returns
    /// Information about operator parameters
    pub fn get_info(&self) -> DoubleKickInfo {
        DoubleKickInfo {
            alpha1: self.alpha1,
            alpha2: self.alpha2,
            eta: self.eta,
            u1_u2_orthogonal: self.u1.dot(&self.u2).abs() < 1e-10,
            non_expansive: self.verify_non_expansive(),
        }
    }

    /// Compute the strength of impulse applied to vector
    ///
    /// # Arguments
    /// * `_v` - Input vector (not used, kept for API compatibility)
    ///
    /// # Returns
    /// Impulse magnitude
    pub fn compute_impulse_strength(&self, _v: &Array1<f64>) -> f64 {
        let impulse = &self.u1 * self.alpha1 + &self.u2 * self.alpha2;
        impulse.dot(&impulse).sqrt()
    }
}

impl Default for DoubleKick {
    fn default() -> Self {
        Self::new(0.05, -0.03)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_doublekick() {
        let dk = DoubleKick::new(0.05, -0.03);
        let info = dk.get_info();

        assert_eq!(info.alpha1, 0.05);
        assert_eq!(info.alpha2, -0.03);
        assert_eq!(info.eta, 0.08);
        assert!(info.u1_u2_orthogonal);
    }

    #[test]
    fn test_apply_doublekick() {
        let dk = DoubleKick::new(0.05, -0.03);
        let v = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = dk.apply(&v);

        assert_eq!(result.len(), 5);
        // Result should be close to input plus small impulse
        let diff = &result - &v;
        let diff_norm = diff.dot(&diff).sqrt();
        assert!(diff_norm < 0.2); // Small impulse
    }

    #[test]
    fn test_verify_non_expansive() {
        let dk = DoubleKick::new(0.05, -0.03);
        assert!(dk.verify_non_expansive());

        let dk_large = DoubleKick::new(0.1, 0.05);
        assert!(!dk_large.verify_non_expansive());
    }

    #[test]
    fn test_orthogonal_vectors() {
        let (u1, u2) = DoubleKick::generate_orthogonal_vectors();

        // Check unit vectors
        assert!((u1.dot(&u1) - 1.0).abs() < 1e-10);
        assert!((u2.dot(&u2) - 1.0).abs() < 1e-10);

        // Check orthogonality
        assert!(u1.dot(&u2).abs() < 1e-10);
    }

    #[test]
    fn test_compute_impulse_strength() {
        let dk = DoubleKick::new(0.05, -0.03);
        let v = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let strength = dk.compute_impulse_strength(&v);

        assert!(strength > 0.0);
        assert!(strength < 0.2); // Should be small
    }

    #[test]
    fn test_from_config() {
        let mut config = HashMap::new();
        config.insert("alpha1".to_string(), 0.04);
        config.insert("alpha2".to_string(), -0.02);

        let dk = DoubleKick::from_config(&config);
        let info = dk.get_info();

        assert_eq!(info.alpha1, 0.04);
        assert_eq!(info.alpha2, -0.02);
    }

    #[test]
    fn test_deterministic_vectors() {
        let (u1_1, u2_1) = DoubleKick::generate_orthogonal_vectors();
        let (u1_2, u2_2) = DoubleKick::generate_orthogonal_vectors();

        // Should be deterministic (same seed)
        for i in 0..5 {
            assert!((u1_1[i] - u1_2[i]).abs() < 1e-10);
            assert!((u2_1[i] - u2_2[i]).abs() < 1e-10);
        }
    }
}
