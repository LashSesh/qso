use nalgebra::SMatrix;
use num_complex::Complex64;
use thiserror::Error;

use super::state::{METATRON_DIMENSION, StateVector};

/// Static 13×13 matrix type alias.
pub type OperatorMatrix = SMatrix<Complex64, 13, 13>;

/// Errors related to quantum operator construction.
#[derive(Debug, Error, PartialEq)]
pub enum QuantumOperatorError {
    /// Permutation input had the wrong length.
    #[error("permutation must have length {expected}, got {actual}")]
    InvalidPermutationLength { expected: usize, actual: usize },

    /// Permutation indices exceeded the supported dimension.
    #[error("permutation index {index} exceeds dimension {dimension}")]
    PermutationIndexOutOfRange { index: usize, dimension: usize },
}

/// Unitary or general linear operator on the Metatron Hilbert space.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantumOperator {
    matrix: OperatorMatrix,
}

impl QuantumOperator {
    /// Construct from a raw matrix (not validated for unitarity).
    pub fn from_matrix(matrix: OperatorMatrix) -> Self {
        Self { matrix }
    }

    /// Identity operator.
    pub fn identity() -> Self {
        Self {
            matrix: OperatorMatrix::identity(),
        }
    }

    /// Build unitary operator from a permutation (0-based indices).
    pub fn from_permutation(permutation: &[usize]) -> Result<Self, QuantumOperatorError> {
        if permutation.len() != METATRON_DIMENSION {
            return Err(QuantumOperatorError::InvalidPermutationLength {
                expected: METATRON_DIMENSION,
                actual: permutation.len(),
            });
        }

        let mut matrix = OperatorMatrix::zeros();
        for (i, &target) in permutation.iter().enumerate() {
            if target >= METATRON_DIMENSION {
                return Err(QuantumOperatorError::PermutationIndexOutOfRange {
                    index: target,
                    dimension: METATRON_DIMENSION,
                });
            }
            matrix[(target, i)] = Complex64::new(1.0, 0.0);
        }

        Ok(Self { matrix })
    }

    /// Check unitarity within tolerance.
    pub fn is_unitary(&self, tol: f64) -> bool {
        let adjoint = self.matrix.adjoint();
        let identity = adjoint * self.matrix;
        identity.iter().enumerate().all(|(idx, value)| {
            let (row, col) = (idx / METATRON_DIMENSION, idx % METATRON_DIMENSION);
            if row == col {
                (value - Complex64::new(1.0, 0.0)).norm() < tol
            } else {
                value.norm() < tol
            }
        })
    }

    /// Compose operators: `self` ∘ `other`.
    pub fn compose(&self, other: &Self) -> Self {
        Self {
            matrix: self.matrix * other.matrix,
        }
    }

    /// Adjoint (Hermitian conjugate).
    pub fn adjoint(&self) -> Self {
        Self {
            matrix: self.matrix.adjoint(),
        }
    }

    /// Apply operator to state vector.
    pub fn apply(&self, state: &StateVector) -> StateVector {
        self.matrix * state
    }

    /// Trace of the matrix.
    pub fn trace(&self) -> Complex64 {
        (0..METATRON_DIMENSION).map(|i| self.matrix[(i, i)]).sum()
    }

    /// Dense matrix access.
    pub fn matrix(&self) -> &OperatorMatrix {
        &self.matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_operator_is_unitary() {
        let perm: Vec<_> = (0..METATRON_DIMENSION).collect();
        let op = QuantumOperator::from_permutation(&perm).unwrap();
        assert!(op.is_unitary(1e-12));
    }

    #[test]
    fn compose_identity() {
        let op = QuantumOperator::identity();
        let composed = op.compose(&op);
        assert!(composed.is_unitary(1e-12));
    }
}
