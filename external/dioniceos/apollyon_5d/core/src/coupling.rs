//! Coupling Matrix and Interaction Types
//!
//! Defines the coupling matrix C and coupling types τᵢⱼ for variable interactions.

use serde::{Deserialize, Serialize};

/// Coupling type τᵢⱼ defining how variables interact
///
/// Each type corresponds to a specific mathematical form:
/// - Linear: Cᵢⱼ · σⱼ
/// - Quadratic: Cᵢⱼ · σⱼ²
/// - Product: Cᵢⱼ · σᵢ · σⱼ (cross-product coupling)
/// - Sigmoid: Cᵢⱼ · tanh(σⱼ) (bounded nonlinear)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CouplingType {
    /// Linear coupling: Cᵢⱼ · σⱼ
    Linear,
    /// Quadratic coupling: Cᵢⱼ · σⱼ²
    Quadratic,
    /// Product coupling: Cᵢⱼ · σᵢ · σⱼ
    Product,
    /// Sigmoid coupling: Cᵢⱼ · tanh(σⱼ)
    Sigmoid,
}

impl CouplingType {
    /// Apply coupling function to values
    ///
    /// # Arguments
    /// * `si` - Source variable value σᵢ
    /// * `sj` - Target variable value σⱼ
    /// * `cij` - Coupling strength Cᵢⱼ
    ///
    /// # Returns
    /// The coupling contribution based on the type
    pub fn apply(&self, si: f64, sj: f64, cij: f64) -> f64 {
        match self {
            CouplingType::Linear => cij * sj,
            CouplingType::Quadratic => cij * sj * sj,
            CouplingType::Product => cij * si * sj,
            CouplingType::Sigmoid => cij * sj.tanh(),
        }
    }

    /// Compute partial derivative ∂/∂σⱼ of coupling term
    ///
    /// Used for Jacobian computation in stability analysis
    pub fn derivative_wrt_sj(&self, si: f64, sj: f64, cij: f64) -> f64 {
        match self {
            CouplingType::Linear => cij,
            CouplingType::Quadratic => 2.0 * cij * sj,
            CouplingType::Product => cij * si,
            CouplingType::Sigmoid => {
                let tanh_sj = sj.tanh();
                cij * (1.0 - tanh_sj * tanh_sj)
            }
        }
    }

    /// Compute partial derivative ∂/∂σᵢ of coupling term
    ///
    /// Used for Jacobian computation (relevant for Product type)
    pub fn derivative_wrt_si(&self, _si: f64, sj: f64, cij: f64) -> f64 {
        match self {
            CouplingType::Linear => 0.0,
            CouplingType::Quadratic => 0.0,
            CouplingType::Product => cij * sj,
            CouplingType::Sigmoid => 0.0,
        }
    }
}

/// Coupling matrix C and types τ
///
/// Represents a 5×5 matrix of coupling strengths and their types.
/// The element Cᵢⱼ with type τᵢⱼ defines how variable j influences variable i.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingMatrix {
    /// Coupling strength matrix C ∈ ℝ⁵ˣ⁵
    pub strengths: [[f64; 5]; 5],
    /// Coupling type matrix τ ∈ {Linear, Quadratic, Product, Sigmoid}⁵ˣ⁵
    pub types: [[CouplingType; 5]; 5],
}

impl CouplingMatrix {
    /// Create a new coupling matrix with all linear couplings
    pub fn new(strengths: [[f64; 5]; 5]) -> Self {
        CouplingMatrix {
            strengths,
            types: [[CouplingType::Linear; 5]; 5],
        }
    }

    /// Create a zero coupling matrix (no interactions)
    pub fn zero() -> Self {
        CouplingMatrix {
            strengths: [[0.0; 5]; 5],
            types: [[CouplingType::Linear; 5]; 5],
        }
    }

    /// Create an identity coupling matrix
    pub fn identity() -> Self {
        let mut strengths = [[0.0; 5]; 5];
        for i in 0..5 {
            strengths[i][i] = 1.0;
        }
        CouplingMatrix {
            strengths,
            types: [[CouplingType::Linear; 5]; 5],
        }
    }

    /// Set coupling strength and type
    pub fn set(&mut self, i: usize, j: usize, strength: f64, coupling_type: CouplingType) {
        self.strengths[i][j] = strength;
        self.types[i][j] = coupling_type;
    }

    /// Get coupling strength
    pub fn get_strength(&self, i: usize, j: usize) -> f64 {
        self.strengths[i][j]
    }

    /// Get coupling type
    pub fn get_type(&self, i: usize, j: usize) -> CouplingType {
        self.types[i][j]
    }

    /// Apply coupling from all variables to compute contribution to dσᵢ/dt
    ///
    /// Computes: Σⱼ τᵢⱼ(σᵢ, σⱼ, Cᵢⱼ)
    pub fn apply_to_variable(&self, i: usize, state: &crate::state::State5D) -> f64 {
        let si = state.get(i);
        let mut sum = 0.0;

        for j in 0..5 {
            let sj = state.get(j);
            let cij = self.strengths[i][j];
            let tau_ij = self.types[i][j];
            sum += tau_ij.apply(si, sj, cij);
        }

        sum
    }
}

impl Default for CouplingMatrix {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::State5D;

    #[test]
    fn test_coupling_type_linear() {
        let ct = CouplingType::Linear;
        assert_eq!(ct.apply(1.0, 2.0, 3.0), 6.0); // 3.0 * 2.0
        assert_eq!(ct.derivative_wrt_sj(1.0, 2.0, 3.0), 3.0);
    }

    #[test]
    fn test_coupling_type_quadratic() {
        let ct = CouplingType::Quadratic;
        assert_eq!(ct.apply(1.0, 2.0, 3.0), 12.0); // 3.0 * 2.0^2
        assert_eq!(ct.derivative_wrt_sj(1.0, 2.0, 3.0), 12.0); // 2 * 3.0 * 2.0
    }

    #[test]
    fn test_coupling_type_product() {
        let ct = CouplingType::Product;
        assert_eq!(ct.apply(2.0, 3.0, 4.0), 24.0); // 4.0 * 2.0 * 3.0
        assert_eq!(ct.derivative_wrt_sj(2.0, 3.0, 4.0), 8.0); // 4.0 * 2.0
        assert_eq!(ct.derivative_wrt_si(2.0, 3.0, 4.0), 12.0); // 4.0 * 3.0
    }

    #[test]
    fn test_coupling_type_sigmoid() {
        let ct = CouplingType::Sigmoid;
        let result = ct.apply(1.0, 0.0, 2.0);
        assert!((result - 0.0).abs() < 1e-10); // tanh(0) = 0

        let result = ct.apply(1.0, 1.0, 2.0);
        assert!((result - 2.0 * 1.0_f64.tanh()).abs() < 1e-10);
    }

    #[test]
    fn test_coupling_matrix_zero() {
        let cm = CouplingMatrix::zero();
        for i in 0..5 {
            for j in 0..5 {
                assert_eq!(cm.get_strength(i, j), 0.0);
            }
        }
    }

    #[test]
    fn test_coupling_matrix_identity() {
        let cm = CouplingMatrix::identity();
        for i in 0..5 {
            for j in 0..5 {
                if i == j {
                    assert_eq!(cm.get_strength(i, j), 1.0);
                } else {
                    assert_eq!(cm.get_strength(i, j), 0.0);
                }
            }
        }
    }

    #[test]
    fn test_coupling_matrix_apply() {
        let mut cm = CouplingMatrix::identity();
        cm.set(0, 1, 0.5, CouplingType::Linear);

        let state = State5D::new(1.0, 2.0, 3.0, 4.0, 5.0);
        let contrib = cm.apply_to_variable(0, &state);

        // 1.0 * 1.0 (diagonal) + 0.5 * 2.0 (coupling from var 1)
        assert_eq!(contrib, 2.0);
    }
}
