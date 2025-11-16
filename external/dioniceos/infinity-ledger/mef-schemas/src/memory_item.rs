//! MemoryItem - 8D normalized vector with spectral signature

use serde::{Deserialize, Serialize};

/// Spectral signature components (ψ, ρ, ω)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpectralSignature {
    /// Phase alignment (ψ)
    pub psi: f64,

    /// Resonance (ρ)
    pub rho: f64,

    /// Oscillation (ω)
    pub omega: f64,
}

/// Proof of Resonance status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PorStatus {
    /// PoR validation passed
    Valid,
    /// PoR validation failed
    Invalid,
    /// PoR not yet computed
    Pending,
}

/// MemoryItem represents an 8D normalized vector with spectral signature
/// Constructed from 5D spiral coordinates + 3D spectral features
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryItem {
    /// Unique memory identifier
    pub id: String,

    /// The 8D normalized vector (||z||₂ = 1)
    pub vector: Vec<f64>,

    /// Alias for vector to support both naming conventions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector8: Option<Vec<f64>>,

    /// Spectral signature
    pub spectral: SpectralSignature,

    /// Proof of Resonance status
    #[serde(skip_serializing_if = "Option::is_none")]
    pub por_status: Option<PorStatus>,

    /// Associated TIC identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tic_id: Option<String>,

    /// Optional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl MemoryItem {
    /// Create a new MemoryItem with validation
    pub fn new(
        id: String,
        vector: Vec<f64>,
        spectral: SpectralSignature,
        metadata: Option<serde_json::Value>,
    ) -> crate::Result<Self> {
        // Validate vector dimension
        if vector.len() != 8 {
            return Err(crate::SchemaError::InvalidDimension {
                expected: 8,
                got: vector.len(),
            });
        }

        // Validate normalization (with tolerance)
        let norm: f64 = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        let tolerance = 1e-6;
        if (norm - 1.0).abs() > tolerance {
            return Err(crate::SchemaError::InvalidSpectral(format!(
                "Vector not normalized: ||z|| = {:.8}",
                norm
            )));
        }

        Ok(Self {
            id,
            vector: vector.clone(),
            vector8: Some(vector),
            spectral,
            por_status: None,
            tic_id: None,
            metadata,
        })
    }

    /// Create a new MemoryItem with extended fields
    pub fn new_extended(
        id: String,
        vector: Vec<f64>,
        spectral: SpectralSignature,
        por_status: PorStatus,
        tic_id: String,
    ) -> Self {
        Self {
            id,
            vector: vector.clone(),
            vector8: Some(vector),
            spectral,
            por_status: Some(por_status),
            tic_id: Some(tic_id),
            metadata: None,
        }
    }

    /// Get the vector (supports both vector and vector8 fields)
    pub fn get_vector(&self) -> &[f64] {
        self.vector8.as_deref().unwrap_or(&self.vector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_memory_item() {
        // Create a normalized 8D vector
        let val = 1.0 / (8.0_f64).sqrt();
        let vector = vec![val; 8];

        let spectral = SpectralSignature {
            psi: 0.3,
            rho: 0.3,
            omega: 0.4,
        };

        let item = MemoryItem::new("mem_001".to_string(), vector, spectral, None);
        assert!(item.is_ok());
    }

    #[test]
    fn test_invalid_dimension() {
        let vector = vec![0.5; 5];
        let spectral = SpectralSignature {
            psi: 0.3,
            rho: 0.3,
            omega: 0.4,
        };

        let item = MemoryItem::new("mem_002".to_string(), vector, spectral, None);
        assert!(item.is_err());
    }

    #[test]
    fn test_not_normalized() {
        let vector = vec![1.0; 8];
        let spectral = SpectralSignature {
            psi: 0.3,
            rho: 0.3,
            omega: 0.4,
        };

        let item = MemoryItem::new("mem_003".to_string(), vector, spectral, None);
        assert!(item.is_err());
    }
}
