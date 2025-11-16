//! Validation Tests
//!
//! Section 11 - reference solutions for testing correctness.

use crate::coupling::{CouplingMatrix, CouplingType};
use crate::dynamics::VectorField;
use crate::integration::{Integrator, TimeConfig};
use crate::state::State5D;

/// Test 1: Linear Decoupled System (Section 11.1.1)
///
/// dσᵢ/dt = -λᵢ σᵢ with λᵢ > 0
/// Analytical solution: σᵢ(t) = σᵢ(0) exp(-λᵢ t)
pub fn test_linear_decoupled() -> bool {
    let mut coupling = CouplingMatrix::zero();

    // Set diagonal to negative values (decay rates)
    let lambdas = [1.0, 2.0, 3.0, 4.0, 5.0];
    for i in 0..5 {
        coupling.set(i, i, -lambdas[i], CouplingType::Linear);
    }

    let vf = VectorField::from_coupling(coupling);
    let tc = TimeConfig::new(0.001, 0.0, 1.0);
    let integrator = Integrator::new(vf, tc);

    let initial = State5D::new(1.0, 1.0, 1.0, 1.0, 1.0);
    let final_state = integrator.integrate_final(initial);

    // Check against analytical solution
    let mut success = true;
    for i in 0..5 {
        let analytical = (-lambdas[i]).exp();
        let numerical = final_state.get(i);
        let error = (analytical - numerical).abs();

        if error > 0.01 {
            eprintln!(
                "Test 1 failed for variable {}: expected {}, got {}, error {}",
                i, analytical, numerical, error
            );
            success = false;
        }
    }

    success
}

/// Test 2: Harmonic Oscillator (Section 11.1.2)
///
/// System: dσ₁/dt = σ₂, dσ₂/dt = -σ₁ (other variables zero)
/// Analytical solution: σ₁(t) = cos(t), σ₂(t) = -sin(t) for σ₁(0)=1, σ₂(0)=0
pub fn test_harmonic_oscillator() -> bool {
    let mut coupling = CouplingMatrix::zero();

    // σ₁' = σ₂
    coupling.set(0, 1, 1.0, CouplingType::Linear);

    // σ₂' = -σ₁
    coupling.set(1, 0, -1.0, CouplingType::Linear);

    let vf = VectorField::from_coupling(coupling);
    let tc = TimeConfig::new(0.001, 0.0, std::f64::consts::PI / 2.0);
    let integrator = Integrator::new(vf, tc);

    let initial = State5D::new(1.0, 0.0, 0.0, 0.0, 0.0);
    let final_state = integrator.integrate_final(initial);

    // At t = π/2: σ₁ should be ~0, σ₂ should be ~-1
    let t = std::f64::consts::PI / 2.0;
    let expected_s1 = t.cos(); // ≈ 0
    let expected_s2 = -t.sin(); // ≈ -1

    let error_s1 = (expected_s1 - final_state.get(0)).abs();
    let error_s2 = (expected_s2 - final_state.get(1)).abs();

    let success = error_s1 < 0.01 && error_s2 < 0.01;

    if !success {
        eprintln!(
            "Test 2 failed: σ₁ error = {}, σ₂ error = {}",
            error_s1, error_s2
        );
    }

    success
}

/// Test 3: Fixed Point Convergence (Section 11.1.3)
///
/// System with stable fixed point at origin
pub fn test_fixed_point_convergence() -> bool {
    let mut coupling = CouplingMatrix::zero();

    // All variables decay to zero
    for i in 0..5 {
        coupling.set(i, i, -1.0, CouplingType::Linear);
    }

    let vf = VectorField::from_coupling(coupling);
    let tc = TimeConfig::new(0.01, 0.0, 5.0);
    let integrator = Integrator::new(vf, tc);

    let initial = State5D::new(1.0, 2.0, 3.0, 4.0, 5.0);
    let final_state = integrator.integrate_final(initial);

    // After t=5 with decay rate 1, all values should be near zero
    let expected = (-5.0_f64).exp();
    let mut success = true;

    for i in 0..5 {
        let expected_val = (i + 1) as f64 * expected;
        let error = (final_state.get(i) - expected_val).abs();

        if error > 0.1 {
            eprintln!(
                "Test 3 failed for variable {}: expected ~{}, got {}, error {}",
                i,
                expected_val,
                final_state.get(i),
                error
            );
            success = false;
        }
    }

    success
}

/// Run all validation tests
pub fn run_all_tests() -> bool {
    println!("Running validation tests...");

    let test1 = test_linear_decoupled();
    println!(
        "Test 1 (Linear Decoupled): {}",
        if test1 { "PASS" } else { "FAIL" }
    );

    let test2 = test_harmonic_oscillator();
    println!(
        "Test 2 (Harmonic Oscillator): {}",
        if test2 { "PASS" } else { "FAIL" }
    );

    let test3 = test_fixed_point_convergence();
    println!(
        "Test 3 (Fixed Point): {}",
        if test3 { "PASS" } else { "FAIL" }
    );

    test1 && test2 && test3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_test_1() {
        assert!(test_linear_decoupled());
    }

    #[test]
    fn validation_test_2() {
        assert!(test_harmonic_oscillator());
    }

    #[test]
    fn validation_test_3() {
        assert!(test_fixed_point_convergence());
    }
}
