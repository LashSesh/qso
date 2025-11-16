//! Example: Custom 5D System
//!
//! This example shows how to create a custom 5D system from scratch.

use core_5d::*;

fn main() {
    println!("=== Custom 5D System Example ===\n");

    // Define a custom coupling structure
    // Variable 0 and 1 form an oscillator
    // Variable 2 decays exponentially
    // Variable 3 grows logistically (sigmoid coupling)
    // Variable 4 is constant

    let mut coupling = CouplingMatrix::zero();

    // Oscillator: dσ₀/dt = σ₁, dσ₁/dt = -σ₀
    coupling.set(0, 1, 1.0, CouplingType::Linear);
    coupling.set(1, 0, -1.0, CouplingType::Linear);

    // Decay: dσ₂/dt = -0.5σ₂
    coupling.set(2, 2, -0.5, CouplingType::Linear);

    // Logistic growth with saturation
    coupling.set(3, 3, 0.3, CouplingType::Linear);
    coupling.set(3, 3, -0.1, CouplingType::Sigmoid); // Self-limiting

    // Variable 4 constant (no dynamics)

    // Add small coupling from oscillator to variable 3
    coupling.set(3, 0, 0.05, CouplingType::Product);

    // Create vector field
    let vf = VectorField::from_coupling(coupling);

    // Initial condition
    let initial = State5D::new(1.0, 0.0, 2.0, 0.1, 1.0);

    println!("Initial state: {:?}", initial.to_array());

    // Set up integration
    let time_config = integration::TimeConfig::new(0.01, 0.0, 20.0);
    let integrator = integration::Integrator::new(vf, time_config);

    // Run simulation
    println!("Integrating for {} time units...", time_config.t_final);
    let trajectory = integrator.integrate(initial);

    // Analyze results
    let final_state = trajectory.last().unwrap().1;
    println!("Final state: {:?}", final_state.to_array());

    // Check oscillation amplitude
    let states: Vec<State5D> = trajectory.iter().map(|(_, s)| *s).collect();
    let var0_values: Vec<f64> = states.iter().map(|s| s.get(0)).collect();
    let max_var0 = var0_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_var0 = var0_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    println!(
        "\nVariable 0 (oscillator) range: [{:.3}, {:.3}]",
        min_var0, max_var0
    );
    println!(
        "Variable 2 (decay): {:.3} → {:.3}",
        initial.get(2),
        final_state.get(2)
    );
    println!(
        "Variable 3 (logistic): {:.3} → {:.3}",
        initial.get(3),
        final_state.get(3)
    );

    // Stability analysis at initial point
    println!("\n=== Stability Analysis ===");
    let jacobian = integrator.vector_field.jacobian(&initial);
    let eigenvalues = stability::StabilityAnalyzer::eigenvalues(&jacobian);

    println!("Eigenvalues (real parts):");
    for (i, &eig) in eigenvalues.iter().enumerate() {
        println!("  λ{} = {:.4}", i, eig);
    }

    let stability_type = stability::StabilityAnalyzer::classify_stability(&eigenvalues);
    println!("Stability at initial state: {:?}", stability_type);

    // Create 2D projection
    println!("\n=== Projection ===");
    let projector = projection::Projector::orthogonal(0, 1);
    let points = projector.project_many(&states);

    println!("Projected {} points to 2D (variables 0 vs 1)", points.len());
    println!("First point: ({:.3}, {:.3})", points[0].x, points[0].y);
    println!(
        "Last point: ({:.3}, {:.3})",
        points.last().unwrap().x,
        points.last().unwrap().y
    );

    // Export trajectory
    println!("\n=== Data Export ===");
    let traj_export = export::Trajectory::from_pairs(trajectory);

    match traj_export.export_csv("/tmp/custom_system.csv") {
        Ok(_) => println!("Exported to: /tmp/custom_system.csv"),
        Err(e) => eprintln!("Export failed: {}", e),
    }

    match traj_export.export_json("/tmp/custom_system.json") {
        Ok(_) => println!("Exported to: /tmp/custom_system.json"),
        Err(e) => eprintln!("Export failed: {}", e),
    }

    println!("\n=== Example Complete ===");
}
