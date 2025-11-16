//! 5-Dimensional State Vector
//!
//! Defines the state space σ ∈ ℝ⁵ for the dynamical system.

use nalgebra::SVector;
use serde::{Deserialize, Serialize};

/// 5-dimensional state vector σ = (σ₁, σ₂, σ₃, σ₄, σ₅)
///
/// Each component σᵢ represents a dynamical variable in the system.
/// All components must be finite (no NaN or Inf).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct State5D {
    pub data: SVector<f64, 5>,
}

impl State5D {
    /// Create a new state vector from five components
    ///
    /// # Arguments
    /// * `s1` - First component σ₁
    /// * `s2` - Second component σ₂
    /// * `s3` - Third component σ₃
    /// * `s4` - Fourth component σ₄
    /// * `s5` - Fifth component σ₅
    ///
    /// # Panics
    /// Panics if any component is NaN or infinite
    pub fn new(s1: f64, s2: f64, s3: f64, s4: f64, s5: f64) -> Self {
        let state = State5D {
            data: SVector::from([s1, s2, s3, s4, s5]),
        };
        state.validate();
        state
    }

    /// Create a state vector from an array
    pub fn from_array(arr: [f64; 5]) -> Self {
        Self::new(arr[0], arr[1], arr[2], arr[3], arr[4])
    }

    /// Create a zero state vector
    pub fn zero() -> Self {
        State5D {
            data: SVector::zeros(),
        }
    }

    /// Access component by index (0-4)
    pub fn get(&self, i: usize) -> f64 {
        self.data[i]
    }

    /// Set component by index (0-4)
    ///
    /// Returns false if the value is not finite
    pub fn set(&mut self, i: usize, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        self.data[i] = value;
        true
    }

    /// Validate that all components are finite
    ///
    /// # Panics
    /// Panics if any component is NaN or infinite
    pub fn validate(&self) {
        for i in 0..5 {
            let val = self.data[i];
            if !val.is_finite() {
                panic!("State component {} is not finite: {}", i, val);
            }
        }
    }

    /// Check if state is valid (all finite)
    pub fn is_valid(&self) -> bool {
        self.data.iter().all(|&x| x.is_finite())
    }

    /// Convert to array
    pub fn to_array(&self) -> [f64; 5] {
        [
            self.data[0],
            self.data[1],
            self.data[2],
            self.data[3],
            self.data[4],
        ]
    }

    /// Compute Euclidean norm ‖σ‖₂
    pub fn norm(&self) -> f64 {
        self.data.norm()
    }

    /// Compute inner product ⟨σ, other⟩
    pub fn dot(&self, other: &State5D) -> f64 {
        self.data.dot(&other.data)
    }

    /// Element-wise addition
    pub fn add(&self, other: &State5D) -> State5D {
        State5D {
            data: self.data + other.data,
        }
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &State5D) -> State5D {
        State5D {
            data: self.data - other.data,
        }
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: f64) -> State5D {
        State5D {
            data: self.data * scalar,
        }
    }
}

impl std::ops::Add for State5D {
    type Output = State5D;

    fn add(self, other: State5D) -> State5D {
        State5D {
            data: self.data + other.data,
        }
    }
}

impl std::ops::Sub for State5D {
    type Output = State5D;

    fn sub(self, other: State5D) -> State5D {
        State5D {
            data: self.data - other.data,
        }
    }
}

impl std::ops::Mul<f64> for State5D {
    type Output = State5D;

    fn mul(self, scalar: f64) -> State5D {
        self.scale(scalar)
    }
}

impl std::ops::Mul<State5D> for f64 {
    type Output = State5D;

    fn mul(self, state: State5D) -> State5D {
        state.scale(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_creation() {
        let s = State5D::new(1.0, 2.0, 3.0, 4.0, 5.0);
        assert_eq!(s.get(0), 1.0);
        assert_eq!(s.get(1), 2.0);
        assert_eq!(s.get(2), 3.0);
        assert_eq!(s.get(3), 4.0);
        assert_eq!(s.get(4), 5.0);
    }

    #[test]
    fn test_state_zero() {
        let s = State5D::zero();
        for i in 0..5 {
            assert_eq!(s.get(i), 0.0);
        }
    }

    #[test]
    fn test_state_operations() {
        let s1 = State5D::new(1.0, 2.0, 3.0, 4.0, 5.0);
        let s2 = State5D::new(5.0, 4.0, 3.0, 2.0, 1.0);

        let sum = s1.add(&s2);
        assert_eq!(sum.get(0), 6.0);
        assert_eq!(sum.get(4), 6.0);

        let diff = s1.sub(&s2);
        assert_eq!(diff.get(0), -4.0);
        assert_eq!(diff.get(4), 4.0);

        let scaled = s1.scale(2.0);
        assert_eq!(scaled.get(0), 2.0);
        assert_eq!(scaled.get(4), 10.0);
    }

    #[test]
    fn test_state_norm() {
        let s = State5D::new(1.0, 0.0, 0.0, 0.0, 0.0);
        assert!((s.norm() - 1.0).abs() < 1e-10);

        let s = State5D::new(3.0, 4.0, 0.0, 0.0, 0.0);
        assert!((s.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "not finite")]
    fn test_state_nan_validation() {
        State5D::new(1.0, f64::NAN, 3.0, 4.0, 5.0);
    }

    #[test]
    #[should_panic(expected = "not finite")]
    fn test_state_inf_validation() {
        State5D::new(1.0, 2.0, f64::INFINITY, 4.0, 5.0);
    }
}
