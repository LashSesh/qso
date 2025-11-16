/*!
 * Quantum Module - Quantum States and Operators for Metatron Cube
 *
 * This module provides basic classes and utilities for representing quantum-
 * mechanical states and operators on the Metatron Cube. A `QuantumState` is a
 * 13-dimensional complex vector corresponding to the amplitudes of being at each
 * of the 13 canonical nodes. A `QuantumOperator` is a 13×13 matrix (typically
 * unitary) that acts on these states. Together they enable a rudimentary
 * Hilbert-space formalism for post-symbolic cognition as envisioned in the
 * Theory of Everything document.
 *
 * The initial implementation focuses on basic superposition, inner products,
 * and permutation-based unitaries derived from the symmetry groups of the cube.
 * Future extensions might include entanglement across multiple cubes, higher-
 * order tensor representations, and non-permutation gates.
 */

use anyhow::{anyhow, Result};
use ndarray::Array1;
use num_complex::Complex64;
use rand::prelude::*;

use crate::symmetries::permutation_matrix;

/// A quantum state on the 13-dimensional Hilbert space of the cube
///
/// The state is represented internally as an ndarray of complex amplitudes
/// (column vector). Upon initialization, the state is normalised to unit
/// length. Basic operations such as applying operators and computing inner
/// products are provided.
#[derive(Debug, Clone)]
pub struct QuantumState {
    /// Complex amplitudes for the 13 nodes
    pub amplitudes: Array1<Complex64>,
}

impl QuantumState {
    /// Create a new quantum state from complex amplitudes
    ///
    /// # Arguments
    ///
    /// * `amplitudes` - A vector of 13 complex numbers representing the amplitudes
    ///   for nodes 1–13. If fewer than 13 entries are provided, the vector will
    ///   be padded with zeros; if more entries are provided, an error is returned.
    /// * `normalize` - If true (default), the state vector is normalised to have
    ///   Euclidean norm 1. If false, no normalisation is performed.
    ///
    /// # Returns
    ///
    /// A new QuantumState instance
    pub fn new(amplitudes: Vec<Complex64>, normalize: bool) -> Result<Self> {
        let mut amps = amplitudes;

        if amps.len() > 13 {
            return Err(anyhow!("QuantumState expects at most 13 amplitudes"));
        }

        // Pad with zeros if necessary
        while amps.len() < 13 {
            amps.push(Complex64::new(0.0, 0.0));
        }

        let mut state = QuantumState {
            amplitudes: Array1::from(amps),
        };

        if normalize {
            state.normalise();
        }

        Ok(state)
    }

    /// Normalise the state to unit norm (L2)
    pub fn normalise(&mut self) {
        let norm = self.norm();
        if norm == 0.0 {
            // Avoid division by zero: define |0⟩
            self.amplitudes.fill(Complex64::new(0.0, 0.0));
        } else {
            self.amplitudes.mapv_inplace(|x| x / norm);
        }
    }

    /// Compute the L2 norm of the state
    fn norm(&self) -> f64 {
        self.amplitudes
            .iter()
            .map(|&c| c.norm_sqr())
            .sum::<f64>()
            .sqrt()
    }

    /// Return the inner product ⟨ψ|ϕ⟩ between this state and another
    ///
    /// The inner product is conjugate linear in the first argument and linear
    /// in the second. The result is a complex number.
    pub fn inner_product(&self, other: &QuantumState) -> Complex64 {
        self.amplitudes
            .iter()
            .zip(other.amplitudes.iter())
            .map(|(&a, &b)| a.conj() * b)
            .sum()
    }

    /// Apply a quantum operator to this state and return the new state
    pub fn apply(&self, operator: &QuantumOperator) -> Result<Self> {
        if operator.matrix.shape() != [13, 13] {
            return Err(anyhow!("Operator must be 13×13 to act on a QuantumState"));
        }

        let new_amplitudes = operator.matrix.dot(&self.amplitudes);

        Ok(QuantumState {
            amplitudes: new_amplitudes,
        })
    }

    /// Return the probability distribution |ψ_i|² over the 13 nodes
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|c| c.norm_sqr()).collect()
    }

    /// Perform a projective measurement in the computational basis
    ///
    /// Returns the index (1-based) of the measured node. Measurement collapses
    /// the state; subsequent calls will collapse relative to the post-measurement
    /// state.
    pub fn measure(&mut self) -> usize {
        let probs = self.probabilities();
        let mut rng = thread_rng();

        // Sample from the probability distribution
        let mut cumulative = 0.0;
        let random_value: f64 = rng.gen();

        for (idx, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if random_value < cumulative {
                // Collapse to the basis state |idx⟩
                let mut collapsed = Array1::zeros(13);
                collapsed[idx] = Complex64::new(1.0, 0.0);
                self.amplitudes = collapsed;
                return idx + 1; // 1-based indexing
            }
        }

        // Fallback (should not reach here if probabilities sum to 1)
        self.amplitudes = Array1::zeros(13);
        self.amplitudes[12] = Complex64::new(1.0, 0.0);
        13
    }

    /// Return a copy of the underlying amplitude vector
    pub fn as_array(&self) -> Array1<Complex64> {
        self.amplitudes.clone()
    }
}

/// A linear operator acting on the 13-dimensional state space
///
/// The operator is represented as a 13×13 complex matrix. For permutation
/// operators, the matrix is unitary (binary entries), but the class can hold
/// arbitrary linear operators. Composition and unitarity checks are provided.
#[derive(Debug, Clone)]
pub struct QuantumOperator {
    /// The 13×13 complex matrix
    pub matrix: ndarray::Array2<Complex64>,
}

impl QuantumOperator {
    /// Create a new quantum operator from a 13×13 complex matrix
    pub fn new(matrix: ndarray::Array2<Complex64>) -> Result<Self> {
        if matrix.shape() != [13, 13] {
            return Err(anyhow!("QuantumOperator matrix must be 13×13"));
        }

        Ok(QuantumOperator { matrix })
    }

    /// Construct a permutation operator from a 13-length permutation vector
    ///
    /// # Arguments
    ///
    /// * `sigma` - A permutation of (1..13) describing how basis vectors map to
    ///   new positions. This is typically produced by the symmetries module.
    ///
    /// # Returns
    ///
    /// The corresponding permutation operator as a QuantumOperator
    pub fn from_permutation(sigma: &[usize]) -> Self {
        let p = permutation_matrix(sigma, 13);
        // Convert to complex
        let matrix = p.mapv(|x| Complex64::new(x, 0.0));
        QuantumOperator { matrix }
    }

    /// Return the composition (matrix multiplication) of this operator with another
    pub fn compose(&self, other: &QuantumOperator) -> Result<Self> {
        if self.matrix.shape() != [13, 13] || other.matrix.shape() != [13, 13] {
            return Err(anyhow!("Both operators must be 13×13"));
        }

        let result = self.matrix.dot(&other.matrix);
        Ok(QuantumOperator { matrix: result })
    }

    /// Check whether the operator is unitary (O⋅O† = I)
    pub fn is_unitary(&self, atol: f64) -> bool {
        // Compute O @ O^†
        let conjugate_transpose = self.matrix.t().mapv(|x| x.conj());
        let product1 = self.matrix.dot(&conjugate_transpose);
        let product2 = conjugate_transpose.dot(&self.matrix);

        // Check if close to identity
        let identity = ndarray::Array2::eye(13).mapv(|x| Complex64::new(x, 0.0));

        let close1 = product1
            .iter()
            .zip(identity.iter())
            .all(|(&a, &b)| (a - b).norm() < atol);

        let close2 = product2
            .iter()
            .zip(identity.iter())
            .all(|(&a, &b)| (a - b).norm() < atol);

        close1 && close2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_quantum_state() {
        let amps = vec![Complex64::new(1.0, 0.0)];
        let state = QuantumState::new(amps, true).unwrap();
        assert_eq!(state.amplitudes.len(), 13);
    }

    #[test]
    fn test_normalise() {
        let amps = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let state = QuantumState::new(amps, true).unwrap();

        // Check that the norm is 1
        let norm = state
            .amplitudes
            .iter()
            .map(|c| c.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inner_product() {
        let amps1 = vec![Complex64::new(1.0, 0.0)];
        let amps2 = vec![Complex64::new(1.0, 0.0)];

        let state1 = QuantumState::new(amps1, true).unwrap();
        let state2 = QuantumState::new(amps2, true).unwrap();

        let inner = state1.inner_product(&state2);
        assert!((inner - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_inner_product_orthogonal() {
        let mut amps1 = vec![Complex64::new(0.0, 0.0); 13];
        amps1[0] = Complex64::new(1.0, 0.0);

        let mut amps2 = vec![Complex64::new(0.0, 0.0); 13];
        amps2[1] = Complex64::new(1.0, 0.0);

        let state1 = QuantumState::new(amps1, false).unwrap();
        let state2 = QuantumState::new(amps2, false).unwrap();

        let inner = state1.inner_product(&state2);
        assert!(inner.norm() < 1e-10);
    }

    #[test]
    fn test_probabilities() {
        let amps = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let state = QuantumState::new(amps, true).unwrap();
        let probs = state.probabilities();

        assert_eq!(probs.len(), 13);
        assert!((probs[0] - 1.0).abs() < 1e-10);
        assert!(probs[1].abs() < 1e-10);
    }

    #[test]
    fn test_create_quantum_operator() {
        let matrix = ndarray::Array2::eye(13).mapv(|x| Complex64::new(x, 0.0));
        let op = QuantumOperator::new(matrix).unwrap();
        assert_eq!(op.matrix.shape(), &[13, 13]);
    }

    #[test]
    fn test_operator_from_permutation() {
        let sigma: Vec<usize> = (1..=13).collect();
        let op = QuantumOperator::from_permutation(&sigma);

        // Identity permutation should give identity matrix
        let identity = ndarray::Array2::eye(13).mapv(|x| Complex64::new(x, 0.0));

        for i in 0..13 {
            for j in 0..13 {
                assert!((op.matrix[[i, j]] - identity[[i, j]]).norm() < 1e-10);
            }
        }
    }

    #[test]
    fn test_apply_operator() {
        let amps = vec![Complex64::new(1.0, 0.0)];
        let state = QuantumState::new(amps, true).unwrap();

        // Apply identity operator
        let sigma: Vec<usize> = (1..=13).collect();
        let op = QuantumOperator::from_permutation(&sigma);

        let new_state = state.apply(&op).unwrap();

        // State should be unchanged
        for i in 0..13 {
            assert!((state.amplitudes[i] - new_state.amplitudes[i]).norm() < 1e-10);
        }
    }

    #[test]
    fn test_compose_operators() {
        let sigma: Vec<usize> = (1..=13).collect();
        let op1 = QuantumOperator::from_permutation(&sigma);
        let op2 = QuantumOperator::from_permutation(&sigma);

        let composed = op1.compose(&op2).unwrap();

        // Identity composed with identity should be identity
        let identity = ndarray::Array2::eye(13).mapv(|x| Complex64::new(x, 0.0));

        for i in 0..13 {
            for j in 0..13 {
                assert!((composed.matrix[[i, j]] - identity[[i, j]]).norm() < 1e-10);
            }
        }
    }

    #[test]
    fn test_is_unitary() {
        let sigma: Vec<usize> = (1..=13).collect();
        let op = QuantumOperator::from_permutation(&sigma);

        assert!(op.is_unitary(1e-8));
    }

    #[test]
    fn test_too_many_amplitudes() {
        let amps = vec![Complex64::new(1.0, 0.0); 14];
        let result = QuantumState::new(amps, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_measure() {
        // Create a state that's definitely in node 1
        let mut amps = vec![Complex64::new(0.0, 0.0); 13];
        amps[0] = Complex64::new(1.0, 0.0);
        let mut state = QuantumState::new(amps, false).unwrap();

        let measurement = state.measure();
        assert_eq!(measurement, 1); // Should always measure node 1
    }
}
