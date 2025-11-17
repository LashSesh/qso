use std::fmt;

use nalgebra::SVector;
use num_complex::Complex64;
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use serde::de::{Error as SerdeError, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

/// Dimension of the Metatron Cube Hilbert space.
pub const METATRON_DIMENSION: usize = 13;

/// Static state vector type alias.
pub type StateVector = SVector<Complex64, 13>;

/// Errors that can occur when manipulating quantum states.
#[derive(Debug, Error, PartialEq)]
pub enum QuantumStateError {
    /// Provided amplitudes did not match the required dimension.
    #[error("expected {expected} amplitudes, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Attempted to sample from a state with zero norm.
    #[error("state has zero norm and cannot be measured")]
    ZeroNorm,
}

/// Quantum state on the 13-dimensional Metatron Hilbert space.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantumState {
    amplitudes: StateVector,
}

impl QuantumState {
    /// Construct a new state from raw amplitudes.
    pub fn try_new(amplitudes: &[Complex64], normalize: bool) -> Result<Self, QuantumStateError> {
        if amplitudes.len() != METATRON_DIMENSION {
            return Err(QuantumStateError::DimensionMismatch {
                expected: METATRON_DIMENSION,
                actual: amplitudes.len(),
            });
        }

        let mut vector = StateVector::from_column_slice(amplitudes);
        if normalize {
            normalize_vector(&mut vector);
        }

        Ok(Self { amplitudes: vector })
    }

    /// Construct from an owned vector.
    pub fn from_vector(mut vector: StateVector, normalize: bool) -> Self {
        if normalize {
            normalize_vector(&mut vector);
        }
        Self { amplitudes: vector }
    }

    /// Basis state |i⟩.
    pub fn basis_state(index: usize) -> Result<Self, QuantumStateError> {
        if index >= METATRON_DIMENSION {
            return Err(QuantumStateError::DimensionMismatch {
                expected: METATRON_DIMENSION,
                actual: index + 1,
            });
        }
        let mut vec = StateVector::zeros();
        vec[index] = Complex64::new(1.0, 0.0);
        Ok(Self { amplitudes: vec })
    }

    /// Uniform superposition over all basis states.
    pub fn uniform_superposition() -> Self {
        let amp = 1.0 / (METATRON_DIMENSION as f64).sqrt();
        let vec = StateVector::from_element(Complex64::new(amp, 0.0));
        Self { amplitudes: vec }
    }

    /// Uniform superposition with explicit dimension check
    pub fn uniform_superposition_checked(dimension: usize) -> Self {
        assert_eq!(
            dimension, METATRON_DIMENSION,
            "Dimension must be 13 for Metatron Cube"
        );
        Self::uniform_superposition()
    }

    /// Haar-random state (using Gaussian sampling with optional seed).
    pub fn random(seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_entropy(),
        };

        let mut vector = StateVector::zeros();
        for amp in vector.iter_mut() {
            let re = StandardNormal.sample(&mut rng);
            let im = StandardNormal.sample(&mut rng);
            *amp = Complex64::new(re, im);
        }

        normalize_vector(&mut vector);
        Self { amplitudes: vector }
    }

    /// Compute ⟨ψ|ψ⟩.
    pub fn norm(&self) -> f64 {
        self.amplitudes
            .iter()
            .map(|c| c.norm_sqr())
            .sum::<f64>()
            .sqrt()
    }

    /// Check if normalized within tolerance.
    pub fn is_normalized(&self, tol: f64) -> bool {
        (self.norm() - 1.0).abs() <= tol
    }

    /// Return probabilities |αᵢ|².
    pub fn probabilities(&self) -> [f64; METATRON_DIMENSION] {
        let mut probs = [0.0; METATRON_DIMENSION];
        for (idx, amp) in self.amplitudes.iter().enumerate() {
            probs[idx] = amp.norm_sqr();
        }
        probs
    }

    /// Get probability at specific node
    pub fn probability_at_node(&self, node: usize) -> f64 {
        if node >= METATRON_DIMENSION {
            0.0
        } else {
            self.amplitudes[node].norm_sqr()
        }
    }

    /// Create state from amplitudes (convenience wrapper for try_new)
    pub fn from_amplitudes(amps: Vec<Complex64>) -> Result<Self, QuantumStateError> {
        Self::try_new(&amps, true)
    }

    /// Compute inner product ⟨self|other⟩
    pub fn inner_product(&self, other: &Self) -> Complex64 {
        self.amplitudes.dotc(&other.amplitudes)
    }

    /// Perform projective measurement, returning 0-based index.
    pub fn measure<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<usize, QuantumStateError> {
        let probs = self.probabilities();
        let total: f64 = probs.iter().sum();
        if total == 0.0 {
            return Err(QuantumStateError::ZeroNorm);
        }
        let dist =
            WeightedIndex::new(probs.iter().cloned()).map_err(|_| QuantumStateError::ZeroNorm)?;
        let idx = dist.sample(rng);
        self.amplitudes.fill(Complex64::new(0.0, 0.0));
        self.amplitudes[idx] = Complex64::new(1.0, 0.0);
        Ok(idx)
    }

    /// Apply a quantum operator to this state.
    pub fn apply(&self, operator: &crate::quantum::operator::QuantumOperator) -> Self {
        let new_vec = operator.matrix() * self.amplitudes;
        Self {
            amplitudes: new_vec,
        }
    }

    /// Expectation value ⟨ψ|O|ψ⟩.
    pub fn expectation_value(
        &self,
        operator: &crate::quantum::operator::QuantumOperator,
    ) -> Complex64 {
        let temp = operator.matrix() * self.amplitudes;
        self.amplitudes.dotc(&temp)
    }

    /// Access raw amplitudes.
    pub fn amplitudes(&self) -> &StateVector {
        &self.amplitudes
    }

    /// Convert to owned vector.
    pub fn into_vector(self) -> StateVector {
        self.amplitudes
    }
}

fn normalize_vector(vector: &mut StateVector) {
    let norm = vector.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    if norm == 0.0 {
        vector.fill(Complex64::new(0.0, 0.0));
        vector[0] = Complex64::new(1.0, 0.0);
    } else {
        for amp in vector.iter_mut() {
            *amp /= norm;
        }
    }
}

impl Serialize for QuantumState {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.amplitudes.as_slice().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for QuantumState {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct QuantumStateVisitor;

        impl<'de> Visitor<'de> for QuantumStateVisitor {
            type Value = QuantumState;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a sequence of 13 complex amplitudes")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut data = Vec::with_capacity(METATRON_DIMENSION);
                while let Some(value) = seq.next_element::<Complex64>()? {
                    data.push(value);
                }
                QuantumState::try_new(&data, true)
                    .map_err(|err| SerdeError::custom(err.to_string()))
            }
        }

        deserializer.deserialize_seq(QuantumStateVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn basis_state_is_normalized() {
        let state = QuantumState::basis_state(0).unwrap();
        assert!(state.is_normalized(1e-12));
        assert_eq!(state.probabilities()[0], 1.0);
    }

    #[test]
    fn uniform_superposition_has_equal_probabilities() {
        let state = QuantumState::uniform_superposition();
        let probs = state.probabilities();
        for p in probs.iter() {
            assert_relative_eq!(*p, 1.0 / METATRON_DIMENSION as f64, epsilon = 1e-12);
        }
    }
}
