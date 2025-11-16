//! System Dynamics and Vector Field
//!
//! Defines the vector field F(σ) that governs system evolution.
//! dσ/dt = F(σ)

use crate::coupling::CouplingMatrix;
use crate::state::State5D;
use serde::{Deserialize, Serialize};

/// Parameters for the system dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemParameters {
    /// Intrinsic rates αᵢ for each variable
    pub intrinsic_rates: [f64; 5],
    /// External forcing fᵢ(t) - for now constant
    pub external_forcing: [f64; 5],
}

impl SystemParameters {
    /// Create new system parameters
    pub fn new(intrinsic_rates: [f64; 5], external_forcing: [f64; 5]) -> Self {
        SystemParameters {
            intrinsic_rates,
            external_forcing,
        }
    }

    /// Create zero parameters (no intrinsic dynamics or forcing)
    pub fn zero() -> Self {
        SystemParameters {
            intrinsic_rates: [0.0; 5],
            external_forcing: [0.0; 5],
        }
    }
}

impl Default for SystemParameters {
    fn default() -> Self {
        Self::zero()
    }
}

/// Vector field F(σ) defining system dynamics
///
/// Implements Equation (9) from the specification:
/// Fᵢ(σ) = αᵢσᵢ + Σⱼ τᵢⱼ(σᵢ, σⱼ, Cᵢⱼ) + fᵢ(t)
///
/// where:
/// - αᵢ: intrinsic rate for variable i
/// - τᵢⱼ: coupling function of type specified in coupling matrix
/// - fᵢ(t): external forcing (currently time-independent)
#[derive(Debug, Clone)]
pub struct VectorField {
    pub coupling: CouplingMatrix,
    pub parameters: SystemParameters,
}

impl VectorField {
    /// Create a new vector field
    pub fn new(coupling: CouplingMatrix, parameters: SystemParameters) -> Self {
        VectorField {
            coupling,
            parameters,
        }
    }

    /// Create a simple vector field with only coupling
    pub fn from_coupling(coupling: CouplingMatrix) -> Self {
        VectorField {
            coupling,
            parameters: SystemParameters::zero(),
        }
    }

    /// Evaluate the vector field at state σ: F(σ)
    ///
    /// Returns dσ/dt as a State5D
    pub fn evaluate(&self, state: &State5D) -> State5D {
        let mut result = State5D::zero();

        for i in 0..5 {
            // Intrinsic dynamics: αᵢσᵢ
            let intrinsic = self.parameters.intrinsic_rates[i] * state.get(i);

            // Coupling contributions: Σⱼ τᵢⱼ(σᵢ, σⱼ, Cᵢⱼ)
            let coupling = self.coupling.apply_to_variable(i, state);

            // External forcing: fᵢ(t)
            let forcing = self.parameters.external_forcing[i];

            // Total rate of change
            let rate = intrinsic + coupling + forcing;

            // Only set if finite
            if !result.set(i, rate) {
                // If any component is non-finite, return zero vector
                return State5D::zero();
            }
        }

        result
    }

    /// Compute Jacobian matrix J at state σ
    ///
    /// Jᵢⱼ = ∂Fᵢ/∂σⱼ
    ///
    /// Used for stability analysis (Section 5.2)
    pub fn jacobian(&self, state: &State5D) -> [[f64; 5]; 5] {
        let mut jac = [[0.0; 5]; 5];

        for i in 0..5 {
            for j in 0..5 {
                // Diagonal term: ∂(αᵢσᵢ)/∂σⱼ = αᵢ if i==j, else 0
                let diagonal_term = if i == j {
                    self.parameters.intrinsic_rates[i]
                } else {
                    0.0
                };

                // Coupling term: ∂(Σₖ τᵢₖ(σᵢ, σₖ, Cᵢₖ))/∂σⱼ
                let coupling_term = self.coupling_derivative(i, j, state);

                jac[i][j] = diagonal_term + coupling_term;
            }
        }

        jac
    }

    /// Compute ∂/∂σⱼ of coupling terms for variable i
    fn coupling_derivative(&self, i: usize, j: usize, state: &State5D) -> f64 {
        let si = state.get(i);
        let mut deriv = 0.0;

        // Sum over all coupling terms that depend on σⱼ
        for k in 0..5 {
            let sk = state.get(k);
            let cik = self.coupling.get_strength(i, k);
            let tau_ik = self.coupling.get_type(i, k);

            if k == j {
                // Direct coupling: ∂τᵢⱼ/∂σⱼ
                deriv += tau_ik.derivative_wrt_sj(si, sk, cik);
            }

            // For Product coupling, there's also dependence on σᵢ
            if i == j && k != i {
                deriv += tau_ik.derivative_wrt_si(si, sk, cik);
            }
        }

        deriv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coupling::{CouplingMatrix, CouplingType};

    #[test]
    fn test_vector_field_zero() {
        let coupling = CouplingMatrix::zero();
        let params = SystemParameters::zero();
        let vf = VectorField::new(coupling, params);

        let state = State5D::new(1.0, 2.0, 3.0, 4.0, 5.0);
        let result = vf.evaluate(&state);

        for i in 0..5 {
            assert_eq!(result.get(i), 0.0);
        }
    }

    #[test]
    fn test_vector_field_intrinsic() {
        let coupling = CouplingMatrix::zero();
        let mut params = SystemParameters::zero();
        params.intrinsic_rates = [-1.0, -2.0, -3.0, -4.0, -5.0];

        let vf = VectorField::new(coupling, params);
        let state = State5D::new(1.0, 1.0, 1.0, 1.0, 1.0);
        let result = vf.evaluate(&state);

        assert_eq!(result.get(0), -1.0);
        assert_eq!(result.get(1), -2.0);
        assert_eq!(result.get(2), -3.0);
        assert_eq!(result.get(3), -4.0);
        assert_eq!(result.get(4), -5.0);
    }

    #[test]
    fn test_vector_field_coupling() {
        let mut coupling = CouplingMatrix::zero();
        coupling.set(0, 1, 2.0, CouplingType::Linear);

        let vf = VectorField::from_coupling(coupling);
        let state = State5D::new(1.0, 3.0, 0.0, 0.0, 0.0);
        let result = vf.evaluate(&state);

        // F₀ = 2.0 * σ₁ = 2.0 * 3.0 = 6.0
        assert_eq!(result.get(0), 6.0);
    }

    #[test]
    fn test_jacobian_identity() {
        let coupling = CouplingMatrix::identity();
        let vf = VectorField::from_coupling(coupling);
        let state = State5D::zero();

        let jac = vf.jacobian(&state);

        // For identity coupling (linear), Jacobian should be identity
        for i in 0..5 {
            for j in 0..5 {
                if i == j {
                    assert_eq!(jac[i][j], 1.0);
                } else {
                    assert_eq!(jac[i][j], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_jacobian_linear_coupling() {
        let mut coupling = CouplingMatrix::zero();
        coupling.set(0, 1, 2.0, CouplingType::Linear);

        let vf = VectorField::from_coupling(coupling);
        let state = State5D::zero();

        let jac = vf.jacobian(&state);

        // J₀₁ should be 2.0 (derivative of linear coupling)
        assert_eq!(jac[0][1], 2.0);
        assert_eq!(jac[0][0], 0.0);
    }
}
