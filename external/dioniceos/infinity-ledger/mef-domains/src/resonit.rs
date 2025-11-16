/*!
 * Resonit - Elementary information atom with tripolar signature
 *
 * Resonits are fundamental units of domain-specific information,
 * characterized by their resonance signature σ = (ψ, ρ, ω).
 */

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Elementary information atom with tripolar signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resonit {
    /// Unique identifier
    pub id: String,
    /// Tripolar signature: psi (activation), rho (coherence), omega (rhythm)
    pub sigma: Sigma,
    /// Source domain adapter
    pub src: String,
    /// Unix timestamp
    pub ts: i64,
    /// Optional position in domain space
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coordinates: Option<Vec<f64>>,
    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Tripolar resonance signature
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Sigma {
    /// Activation level
    pub psi: f64,
    /// Coherence measure
    pub rho: f64,
    /// Rhythmic frequency
    pub omega: f64,
}

impl Resonit {
    /// Create a new Resonit
    pub fn new(sigma: Sigma, src: String, ts: i64) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            sigma,
            src,
            ts,
            coordinates: None,
            metadata: HashMap::new(),
        }
    }

    /// Convert Resonit to vector representation
    pub fn to_vector(&self) -> Vec<f64> {
        vec![self.sigma.psi, self.sigma.rho, self.sigma.omega]
    }

    /// Calculate resonance between two Resonits (cosine similarity)
    pub fn resonance_with(&self, other: &Resonit) -> f64 {
        let v1 = self.to_vector();
        let v2 = other.to_vector();

        let dot_product: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();

        dot_product / (norm1 * norm2 + 1e-10)
    }
}

impl Sigma {
    /// Create a new Sigma from components
    pub fn new(psi: f64, rho: f64, omega: f64) -> Self {
        Self { psi, rho, omega }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resonit_creation() {
        let sigma = Sigma::new(0.5, 0.7, 0.3);
        let resonit = Resonit::new(sigma, "test".to_string(), 1234567890);

        assert_eq!(resonit.sigma.psi, 0.5);
        assert_eq!(resonit.sigma.rho, 0.7);
        assert_eq!(resonit.sigma.omega, 0.3);
        assert_eq!(resonit.src, "test");
        assert_eq!(resonit.ts, 1234567890);
    }

    #[test]
    fn test_to_vector() {
        let sigma = Sigma::new(0.5, 0.7, 0.3);
        let resonit = Resonit::new(sigma, "test".to_string(), 0);

        let vec = resonit.to_vector();
        assert_eq!(vec, vec![0.5, 0.7, 0.3]);
    }

    #[test]
    fn test_resonance_with() {
        let sigma1 = Sigma::new(0.5, 0.5, 0.5);
        let sigma2 = Sigma::new(0.5, 0.5, 0.5);

        let r1 = Resonit::new(sigma1, "test".to_string(), 0);
        let r2 = Resonit::new(sigma2, "test".to_string(), 0);

        let resonance = r1.resonance_with(&r2);
        assert!((resonance - 1.0).abs() < 1e-6); // Perfect resonance with identical vectors
    }

    #[test]
    fn test_resonance_orthogonal() {
        let sigma1 = Sigma::new(1.0, 0.0, 0.0);
        let sigma2 = Sigma::new(0.0, 1.0, 0.0);

        let r1 = Resonit::new(sigma1, "test".to_string(), 0);
        let r2 = Resonit::new(sigma2, "test".to_string(), 0);

        let resonance = r1.resonance_with(&r2);
        assert!(resonance.abs() < 1e-6); // Zero resonance for orthogonal vectors
    }

    #[test]
    fn test_serialization() {
        let sigma = Sigma::new(0.5, 0.7, 0.3);
        let resonit = Resonit::new(sigma, "test".to_string(), 1234567890);

        let json = serde_json::to_string(&resonit).unwrap();
        let deserialized: Resonit = serde_json::from_str(&json).unwrap();

        assert_eq!(resonit.sigma.psi, deserialized.sigma.psi);
        assert_eq!(resonit.src, deserialized.src);
        assert_eq!(resonit.ts, deserialized.ts);
    }
}
