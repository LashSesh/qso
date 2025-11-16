/*!
 * Field Vector Module - Universal n-dimensional Vector and Resonance Utilities
 *
 * Provides:
 * - Vector arithmetic
 * - Normalization
 * - Cosine similarity
 * - Projection/addition/scaling
 * - Multipolar TRM2-Resonance/Decision-Update (emergent, dynamic)
 */

use ndarray::Array1;
use std::f64::consts::PI;

/// Universal 5D (or n-dim) vector and resonance utility
#[derive(Debug, Clone)]
pub struct FieldVector {
    /// The vector data
    pub vec: Array1<f64>,
    /// Number of dimensions
    pub n: usize,
    /// Fundamental frequency (e.g., for oscillator)
    pub omega: f64,
    /// Current phase (for TRM)
    pub phi: f64,
    /// History of phase values
    pub history: Vec<f64>,
}

impl FieldVector {
    /// Create a new FieldVector from data
    pub fn new(data: Vec<f64>, omega: f64) -> Self {
        let n = data.len();
        Self {
            vec: Array1::from_vec(data),
            n,
            omega,
            phi: 0.0,
            history: Vec::new(),
        }
    }

    /// Compute the L2 norm of the vector
    pub fn norm(&self) -> f64 {
        self.vec.dot(&self.vec).sqrt()
    }

    /// Normalize the vector in-place
    pub fn normalize(&mut self) -> Array1<f64> {
        let nrm = self.norm();
        if nrm == 0.0 {
            return self.vec.clone();
        }
        self.vec = &self.vec / nrm;
        self.vec.clone()
    }

    /// Compute cosine similarity with another vector
    pub fn similarity(&self, other: &[f64]) -> f64 {
        let other_arr = Array1::from_vec(other.to_vec());
        let dot = self.vec.dot(&other_arr);
        let norm_self = self.norm();
        let norm_other = other_arr.dot(&other_arr).sqrt();
        dot / ((norm_self * norm_other) + 1e-12)
    }

    /// Add another vector and return a new FieldVector
    pub fn add(&self, other: &[f64]) -> FieldVector {
        let other_arr = Array1::from_vec(other.to_vec());
        let result = &self.vec + &other_arr;
        FieldVector {
            vec: result,
            n: self.n,
            omega: self.omega,
            phi: self.phi,
            history: self.history.clone(),
        }
    }

    /// Scale the vector and return a new FieldVector
    pub fn scale(&self, s: f64) -> FieldVector {
        let result = &self.vec * s;
        FieldVector {
            vec: result,
            n: self.n,
            omega: self.omega,
            phi: self.phi,
            history: self.history.clone(),
        }
    }

    /// TRM2 update - multipolar resonance model
    ///
    /// # Arguments
    /// * `inputs` - Input signals
    /// * `kappas` - Coupling strengths (optional, defaults to ones)
    /// * `thetas` - Phase offsets (optional, defaults to evenly spaced)
    /// * `dt` - Time step (default 1.0)
    pub fn trm2_update(
        &mut self,
        inputs: &[f64],
        kappas: Option<Vec<f64>>,
        thetas: Option<Vec<f64>>,
        dt: f64,
    ) -> f64 {
        let kappas = kappas.unwrap_or_else(|| vec![1.0; self.n]);
        let thetas = thetas.unwrap_or_else(|| {
            (0..self.n)
                .map(|i| 2.0 * PI * i as f64 / self.n as f64)
                .collect()
        });

        let mut dphi = self.omega;
        for i in 0..self.n {
            if i < inputs.len() && i < kappas.len() && i < thetas.len() {
                dphi += kappas[i] * inputs[i] * (thetas[i] - self.phi).sin();
            }
        }

        self.phi += dphi * dt;
        self.history.push(self.phi);
        self.phi.sin()
    }

    /// Return the vector as an Array1
    pub fn as_array(&self) -> Array1<f64> {
        self.vec.clone()
    }

    /// Return the vector as a Vec
    pub fn as_vec(&self) -> Vec<f64> {
        self.vec.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_field_vector() {
        let fv = FieldVector::new(vec![1.0, 0.0, 0.0, 0.0, 0.0], 0.0);
        assert_eq!(fv.n, 5);
        assert_eq!(fv.omega, 0.0);
        assert_eq!(fv.phi, 0.0);
    }

    #[test]
    fn test_norm() {
        let fv = FieldVector::new(vec![3.0, 4.0], 0.0);
        let norm = fv.norm();
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let mut fv = FieldVector::new(vec![3.0, 4.0], 0.0);
        fv.normalize();
        let norm = fv.norm();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let mut fv = FieldVector::new(vec![0.0, 0.0], 0.0);
        let result = fv.normalize();
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.0);
    }

    #[test]
    fn test_similarity() {
        let fv = FieldVector::new(vec![1.0, 0.0, 0.0], 0.0);
        let other = vec![1.0, 0.0, 0.0];
        let sim = fv.similarity(&other);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_similarity_orthogonal() {
        let fv = FieldVector::new(vec![1.0, 0.0, 0.0], 0.0);
        let other = vec![0.0, 1.0, 0.0];
        let sim = fv.similarity(&other);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_add() {
        let fv = FieldVector::new(vec![1.0, 2.0], 0.0);
        let result = fv.add(&[3.0, 4.0]);
        assert_eq!(result.as_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_scale() {
        let fv = FieldVector::new(vec![1.0, 2.0, 3.0], 0.0);
        let result = fv.scale(2.0);
        assert_eq!(result.as_vec(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_trm2_update() {
        let mut fv = FieldVector::new(vec![1.0, 0.0, 0.0], 0.5);
        let inputs = vec![0.1, 0.2, 0.3];
        let result = fv.trm2_update(&inputs, None, None, 1.0);

        // Check that phase was updated
        assert!(fv.phi != 0.0);
        // Check that history was recorded
        assert_eq!(fv.history.len(), 1);
        // Result should be sin(phi)
        assert!((result - fv.phi.sin()).abs() < 1e-10);
    }

    #[test]
    fn test_trm2_update_with_params() {
        let mut fv = FieldVector::new(vec![1.0, 0.0], 0.5);
        let inputs = vec![0.1, 0.2];
        let kappas = vec![1.0, 2.0];
        let thetas = vec![0.0, PI / 2.0];

        let result = fv.trm2_update(&inputs, Some(kappas), Some(thetas), 0.5);

        // Check that phase was updated
        assert!(fv.phi != 0.0);
        // Check that history was recorded
        assert_eq!(fv.history.len(), 1);
        // Result should be sin(phi)
        assert!((result - fv.phi.sin()).abs() < 1e-10);
    }

    #[test]
    fn test_as_array() {
        let fv = FieldVector::new(vec![1.0, 2.0, 3.0], 0.0);
        let arr = fv.as_array();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0], 1.0);
        assert_eq!(arr[1], 2.0);
        assert_eq!(arr[2], 3.0);
    }

    #[test]
    fn test_as_vec() {
        let fv = FieldVector::new(vec![1.0, 2.0, 3.0], 0.0);
        let v = fv.as_vec();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }
}
