//! 8D vector construction from 5D spiral + 3D spectral features

/// Configuration for 8D vector construction
#[derive(Debug, Clone)]
pub struct Vector8Config {
    /// Weights for 5D spiral coordinates
    pub spiral_weights: [f64; 5],

    /// Weights for spectral components (ψ, ρ, ω)
    pub spectral_weights: [f64; 3],
}

impl Default for Vector8Config {
    fn default() -> Self {
        Self {
            spiral_weights: [1.0; 5],
            spectral_weights: [1.0; 3],
        }
    }
}

/// Builder for 8D normalized vectors
pub struct Vector8Builder {
    config: Vector8Config,
}

impl Default for Vector8Builder {
    fn default() -> Self {
        Self {
            config: Vector8Config::default(),
        }
    }
}

impl Vector8Builder {
    /// Create a new Vector8Builder with custom configuration
    pub fn new(config: Vector8Config) -> Self {
        Self { config }
    }

    /// Build an 8D normalized vector from 5D spiral + 3D spectral
    ///
    /// Input:
    /// - x5: 5D spiral coordinates
    /// - sigma: (ψ, ρ, ω) spectral signature
    ///
    /// Output: ẑ = z' / ||z'||₂ where z' = [w₁·x₁, ..., w₅·x₅, wψ·ψ, wρ·ρ, wω·ω]
    pub fn build(&self, x5: &[f64], sigma: (f64, f64, f64)) -> crate::Result<Vec<f64>> {
        if x5.len() != 5 {
            return Err(crate::KnowledgeError::VectorConstruction(format!(
                "Expected 5D spiral input, got {}",
                x5.len()
            )));
        }

        // Construct weighted 8D vector
        let mut z = Vec::with_capacity(8);

        // Add weighted spiral components
        for i in 0..5 {
            z.push(self.config.spiral_weights[i] * x5[i]);
        }

        // Add weighted spectral components
        z.push(self.config.spectral_weights[0] * sigma.0); // ψ
        z.push(self.config.spectral_weights[1] * sigma.1); // ρ
        z.push(self.config.spectral_weights[2] * sigma.2); // ω

        // Normalize
        let norm: f64 = z.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-10 {
            return Err(crate::KnowledgeError::VectorConstruction(
                "Cannot normalize zero vector".to_string(),
            ));
        }

        let z_hat: Vec<f64> = z.iter().map(|x| x / norm).collect();
        Ok(z_hat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_8d_vector() {
        let builder = Vector8Builder::default();
        let x5 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let sigma = (0.3, 0.3, 0.4);

        let z_hat = builder.build(&x5, sigma);
        assert!(z_hat.is_ok());

        let vec = z_hat.unwrap();
        assert_eq!(vec.len(), 8);

        // Check normalization
        let norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_invalid_dimension() {
        let builder = Vector8Builder::default();
        let x3 = vec![0.1, 0.2, 0.3];
        let sigma = (0.3, 0.3, 0.4);

        let result = builder.build(&x3, sigma);
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_weights() {
        let config = Vector8Config {
            spiral_weights: [2.0, 2.0, 2.0, 2.0, 2.0],
            spectral_weights: [1.0, 1.0, 1.0],
        };
        let builder = Vector8Builder::new(config);

        let x5 = vec![0.1, 0.1, 0.1, 0.1, 0.1];
        let sigma = (0.1, 0.1, 0.1);

        let z_hat = builder.build(&x5, sigma).unwrap();
        assert_eq!(z_hat.len(), 8);

        // Check normalization
        let norm: f64 = z_hat.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_zero_vector_error() {
        let builder = Vector8Builder::default();
        let x5 = vec![0.0; 5];
        let sigma = (0.0, 0.0, 0.0);

        let result = builder.build(&x5, sigma);
        assert!(result.is_err());
    }
}
