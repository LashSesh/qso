//! # Vector8 Metric Builder
//!
//! Constructs 8D weighted vectors from 5D spiral coordinates and 3D spectral signatures.
//!
//! ## SPEC-006 Reference
//!
//! From Part 2, Section 2.1 and Part 4, Section 4:
//! - z' = [w1*x1, ..., w5*x5, wψ*ψ, wρ*ρ, wω*ω] ∈ R^8
//! - ẑ = z' / ||z'||₂ (L2 normalization)
//!
//! ## Properties
//!
//! - Deterministic: same inputs → same output
//! - Normalized: ||ẑ||₂ = 1
//! - Weighted: configurable weights for spatial and spectral components

use ndarray::Array1;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MetricError {
    #[error("Invalid dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },

    #[error("Zero norm: cannot normalize zero vector")]
    ZeroNorm,

    #[error("Invalid spectral signature: {0}")]
    InvalidSignature(String),
}

/// Weights for 8D vector construction
///
/// Default weights from SPEC-006 (can be overridden via config)
#[derive(Debug, Clone, PartialEq)]
pub struct Vector8Weights {
    /// Weights for 5D spatial coordinates
    pub spatial: [f64; 5],

    /// Weight for ψ (mid-band ratio)
    pub psi: f64,

    /// Weight for ρ (low/mid ratio)
    pub rho: f64,

    /// Weight for ω (high-band ratio)
    pub omega: f64,
}

impl Default for Vector8Weights {
    fn default() -> Self {
        Self {
            spatial: [1.0, 1.0, 1.0, 1.0, 1.0],
            psi: 1.0,
            rho: 1.0,
            omega: 1.0,
        }
    }
}

/// Builder for 8D weighted vectors
///
/// ## Usage
///
/// ```rust,ignore
/// use mef_knowledge::Vector8Builder;
///
/// let builder = Vector8Builder::default();
/// let x5 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
/// let sigma = (0.3, 0.3, 0.4); // (psi, rho, omega)
///
/// let z_hat = builder.build(&x5, sigma)?;
/// assert_eq!(z_hat.len(), 8);
/// ```
#[derive(Debug, Clone)]
pub struct Vector8Builder {
    weights: Vector8Weights,
}

impl Default for Vector8Builder {
    fn default() -> Self {
        Self::new(Vector8Weights::default())
    }
}

impl Vector8Builder {
    /// Create a new builder with custom weights
    pub fn new(weights: Vector8Weights) -> Self {
        Self { weights }
    }

    /// Build 8D weighted vector from 5D coordinates and 3D spectral signature
    ///
    /// ## Arguments
    ///
    /// * `x5` - 5D spatial coordinates from spiral embedding
    /// * `sigma` - Spectral signature (psi, rho, omega)
    ///
    /// ## Returns
    ///
    /// Normalized 8D vector ẑ with ||ẑ||₂ = 1
    pub fn build(&self, x5: &[f64], sigma: (f64, f64, f64)) -> Result<Vec<f64>, MetricError> {
        if x5.len() != 5 {
            return Err(MetricError::InvalidDimension {
                expected: 5,
                actual: x5.len(),
            });
        }

        let (psi, rho, omega) = sigma;

        // Validate spectral signature ranges
        if !(0.0..=1.0).contains(&psi)
            || !(0.0..=1.0).contains(&rho)
            || !(0.0..=1.0).contains(&omega)
        {
            return Err(MetricError::InvalidSignature(format!(
                "Spectral components must be in [0, 1]: psi={}, rho={}, omega={}",
                psi, rho, omega
            )));
        }

        // Build weighted vector z'
        let mut z_prime = Vec::with_capacity(8);
        for i in 0..5 {
            z_prime.push(self.weights.spatial[i] * x5[i]);
        }
        z_prime.push(self.weights.psi * psi);
        z_prime.push(self.weights.rho * rho);
        z_prime.push(self.weights.omega * omega);

        // Normalize to unit length
        let norm: f64 = z_prime.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm < 1e-10 {
            return Err(MetricError::ZeroNorm);
        }

        let z_hat: Vec<f64> = z_prime.iter().map(|x| x / norm).collect();

        Ok(z_hat)
    }

    /// Build 8D vector using ndarray (for compatibility with numerical code)
    pub fn build_array(
        &self,
        x5: &[f64],
        sigma: (f64, f64, f64),
    ) -> Result<Array1<f64>, MetricError> {
        let vec = self.build(x5, sigma)?;
        Ok(Array1::from_vec(vec))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector8_builder_default() {
        let builder = Vector8Builder::default();
        let x5 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let sigma = (0.3, 0.3, 0.4);

        let result = builder.build(&x5, sigma);
        assert!(result.is_ok());

        let z_hat = result.unwrap();
        assert_eq!(z_hat.len(), 8);

        // Check normalization
        let norm: f64 = z_hat.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_vector8_builder_custom_weights() {
        let weights = Vector8Weights {
            spatial: [2.0, 2.0, 2.0, 2.0, 2.0],
            psi: 1.0,
            rho: 1.0,
            omega: 1.0,
        };

        let builder = Vector8Builder::new(weights);
        let x5 = vec![0.1, 0.1, 0.1, 0.1, 0.1];
        let sigma = (0.3, 0.3, 0.4);

        let z_hat = builder.build(&x5, sigma).unwrap();

        // After normalization, the relationship depends on the relative magnitudes
        // Spatial components are weighted 2x, but there are 5 of them vs 3 spectral
        // Just verify the vector is valid
        assert_eq!(z_hat.len(), 8);
        let norm: f64 = z_hat.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_vector8_builder_invalid_dimension() {
        let builder = Vector8Builder::default();
        let x4 = vec![0.1, 0.2, 0.3, 0.4]; // Wrong size
        let sigma = (0.3, 0.3, 0.4);

        let result = builder.build(&x4, sigma);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector8_builder_deterministic() {
        let builder = Vector8Builder::default();
        let x5 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let sigma = (0.3, 0.3, 0.4);

        let z1 = builder.build(&x5, sigma).unwrap();
        let z2 = builder.build(&x5, sigma).unwrap();

        assert_eq!(z1, z2);
    }

    #[test]
    fn test_vector8_builder_zero_vector() {
        let builder = Vector8Builder::default();
        let x5 = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let sigma = (0.0, 0.0, 0.0);

        let result = builder.build(&x5, sigma);
        assert!(matches!(result, Err(MetricError::ZeroNorm)));
    }
}
