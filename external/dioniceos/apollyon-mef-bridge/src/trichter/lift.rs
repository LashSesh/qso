//! Lift and Projection operations between 4D and 5D spaces
//!
//! As per Section 1.2 of the 4D-Trichter specification:
//! - lift: ℝ⁴ → ℝ⁵ (adds temporal phase ω)
//! - proj_4d: ℝ⁵ → ℝ⁴ (projects to process space)

use super::types::{GuidanceVector, State4D, State5D};

/// Lift a 4D state to 5D space by adding temporal phase ω
///
/// lift((x, y, z, ψ), ω) = (x, y, z, ψ, ω)
///
/// # Arguments
/// * `s4d` - 4D state in process space
/// * `omega` - Temporal phase/oscillation (typically current tick/time)
///
/// # Returns
/// 5D state with added temporal dimension
pub fn lift(s4d: State4D, omega: f64) -> State5D {
    State5D::new(s4d.x, s4d.y, s4d.z, s4d.psi, omega)
}

/// Project 5D guidance field back to 4D process space
///
/// proj_4d(vₓ, vᵧ, vᵧ, vᵩ, vᵪ) = (vₓ, vᵧ, vᵧ, vᵩ)
///
/// # Arguments
/// * `gradient` - 5D gradient field ∇Φ
///
/// # Returns
/// 4D guidance vector for Funnel advection
pub fn proj_4d(gradient: State5D) -> GuidanceVector {
    GuidanceVector::new(gradient.x, gradient.y, gradient.z, gradient.psi)
}

/// Project 5D state to 4D (drops omega component)
pub fn project_state(s5d: State5D) -> State4D {
    State4D::new(s5d.x, s5d.y, s5d.z, s5d.psi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lift_basic() {
        let s4 = State4D::new(1.0, 2.0, 3.0, 4.0);
        let s5 = lift(s4, 5.0);
        
        assert_eq!(s5.x, 1.0);
        assert_eq!(s5.y, 2.0);
        assert_eq!(s5.z, 3.0);
        assert_eq!(s5.psi, 4.0);
        assert_eq!(s5.omega, 5.0);
    }

    #[test]
    fn test_proj_4d_basic() {
        let gradient = State5D::new(1.0, 2.0, 3.0, 4.0, 5.0);
        let guide = proj_4d(gradient);
        
        assert_eq!(guide.vx, 1.0);
        assert_eq!(guide.vy, 2.0);
        assert_eq!(guide.vz, 3.0);
        assert_eq!(guide.vpsi, 4.0);
    }

    #[test]
    fn test_roundtrip_preserves_4d() {
        let s4_orig = State4D::new(1.5, -2.3, 3.7, 0.5);
        let s5 = lift(s4_orig, 100.0);
        let s4_back = project_state(s5);
        
        assert_eq!(s4_orig, s4_back);
    }

    #[test]
    fn test_lift_with_zero_omega() {
        let s4 = State4D::new(1.0, 2.0, 3.0, 4.0);
        let s5 = lift(s4, 0.0);
        assert_eq!(s5.omega, 0.0);
    }

    #[test]
    fn test_lift_with_negative_omega() {
        let s4 = State4D::new(1.0, 2.0, 3.0, 4.0);
        let s5 = lift(s4, -10.0);
        assert_eq!(s5.omega, -10.0);
    }
}
