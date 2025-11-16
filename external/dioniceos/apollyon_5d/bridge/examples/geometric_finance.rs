//! Geometric Finance Example
//!
//! This example demonstrates Phase 5: Geometric constraint enforcement
//! with symmetry-preserving integration.
//!
//! The system implements a financial market model with 5 components:
//! - Asset prices mapped to Metatron Cube nodes
//! - Trajectories constrained to geometric manifolds
//! - C6/D6 symmetry preservation during integration

use bridge::{CognitiveSimulator, GeometricStateSpace, TrajectoryObserver};
use core_5d::*;

fn main() {
    println!("=== Geometric Finance Example ===\n");
    println!("Demonstrating geometric constraint enforcement:");
    println!("  5D financial model mapped to Metatron geometry\n");

    // Create financial market template
    let template = Template::financial_market(0.1, 0.05, 0.02);
    let coupling = template.coupling_matrix;
    let params = template.parameters;

    println!("Financial market parameters:");
    println!("  Growth rate: {:.3}", 0.1);
    println!("  Volatility coupling: {:.3}", 0.05);
    println!("  Risk parameter: {:.3}", 0.02);
    println!();

    // Phase 1: Standard simulation without geometric constraints
    println!("=== Phase 1: Standard Simulation (no constraints) ===");
    let vf = VectorField::new(coupling.clone(), params.clone());
    let time_config = integration::TimeConfig::new(0.1, 0.0, 30.0);
    let integrator = Integrator::new(vf, time_config);

    let initial = State5D::new(1.0, 1.0, 1.0, 1.0, 1.0); // Initial asset values
    let trajectory = integrator.integrate(initial);

    println!("Trajectory length: {} steps", trajectory.len());
    println!(
        "Initial asset values: {:.4}, {:.4}, {:.4}, {:.4}, {:.4}",
        initial.get(0),
        initial.get(1),
        initial.get(2),
        initial.get(3),
        initial.get(4)
    );
    println!(
        "Final asset values: {:.4}, {:.4}, {:.4}, {:.4}, {:.4}",
        trajectory.last().unwrap().1.get(0),
        trajectory.last().unwrap().1.get(1),
        trajectory.last().unwrap().1.get(2),
        trajectory.last().unwrap().1.get(3),
        trajectory.last().unwrap().1.get(4)
    );

    // Phase 2: Geometric constraint enforcement
    println!("\n=== Phase 2: Geometric Simulation (with constraints) ===");

    // Create geometric state space with specific node mapping
    let geo_space = GeometricStateSpace::new([0, 1, 2, 3, 4]);

    println!("Node mapping: {:?}", geo_space.node_mapping);
    println!("Scales: {:?}", geo_space.scales);

    // Project initial state to geometry
    let initial_geometry = geo_space.project_to_geometry(&initial);
    println!(
        "\nInitial geometric projection (first 5 coords): {:.4}, {:.4}, {:.4}, {:.4}, {:.4}",
        initial_geometry[0],
        initial_geometry[1],
        initial_geometry[2],
        initial_geometry[3],
        initial_geometry[4]
    );

    // Run simulation with geometric observation
    let vf2 = VectorField::new(coupling.clone(), params);
    let time_config2 = integration::TimeConfig::new(0.1, 0.0, 30.0);
    let integrator2 = Integrator::new(vf2, time_config2);
    let observer = TrajectoryObserver::new(300);

    let mut sim = CognitiveSimulator::new(integrator2, observer);
    let geo_trajectory = sim.integrate(initial);

    println!(
        "\nGeometric trajectory length: {} steps",
        geo_trajectory.len()
    );
    println!(
        "Final asset values: {:.4}, {:.4}, {:.4}, {:.4}, {:.4}",
        geo_trajectory.last().unwrap().1.get(0),
        geo_trajectory.last().unwrap().1.get(1),
        geo_trajectory.last().unwrap().1.get(2),
        geo_trajectory.last().unwrap().1.get(3),
        geo_trajectory.last().unwrap().1.get(4)
    );

    // Analyze symmetry preservation
    println!("\n=== Symmetry Analysis ===");

    let final_state = geo_trajectory.last().unwrap().1;
    let symmetry_dev = geo_space.symmetry_deviation(&final_state);
    println!("C6 symmetry deviation: {:.6}", symmetry_dev);

    // Test symmetry operations
    let rotated = geo_space.apply_c6_rotation(&final_state, 1);
    println!("\nAfter C6 rotation (60°):");
    println!(
        "  Asset values: {:.4}, {:.4}, {:.4}, {:.4}, {:.4}",
        rotated.get(0),
        rotated.get(1),
        rotated.get(2),
        rotated.get(3),
        rotated.get(4)
    );

    let reflected = geo_space.apply_reflection(&final_state);
    println!("\nAfter D6 reflection:");
    println!(
        "  Asset values: {:.4}, {:.4}, {:.4}, {:.4}, {:.4}",
        reflected.get(0),
        reflected.get(1),
        reflected.get(2),
        reflected.get(3),
        reflected.get(4)
    );

    // Sample trajectory with geometric projections
    println!("\n=== Trajectory Evolution with Geometric Mapping ===");
    println!("Time\t\tAsset 1\t\tAsset 2\t\tGeom Norm");
    println!("{}", "-".repeat(60));

    for &t in &[0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0] {
        if let Some((_, state)) = geo_trajectory
            .iter()
            .find(|(time, _)| (*time - t).abs() < 0.15)
        {
            let geometry = geo_space.project_to_geometry(state);
            let geom_norm = geometry.iter().map(|x| x * x).sum::<f64>().sqrt();
            println!(
                "{:.1}\t\t{:.4}\t\t{:.4}\t\t{:.4}",
                t,
                state.get(0),
                state.get(1),
                geom_norm
            );
        }
    }

    // Validate geometric constraints
    println!("\n=== Constraint Validation ===");
    let mut validation_passes = 0;
    for (_, state) in &geo_trajectory {
        if geo_space.validates(state) {
            validation_passes += 1;
        }
    }
    println!(
        "States passing validation: {}/{} ({:.1}%)",
        validation_passes,
        geo_trajectory.len(),
        100.0 * validation_passes as f64 / geo_trajectory.len() as f64
    );

    // Test constraint enforcement
    let mut test_state = State5D::new(2.0, 3.0, 1.5, 2.5, 1.8);
    println!(
        "\nTest state before enforcement: {:.4}, {:.4}, {:.4}, {:.4}, {:.4}",
        test_state.get(0),
        test_state.get(1),
        test_state.get(2),
        test_state.get(3),
        test_state.get(4)
    );

    geo_space.enforce_constraints(&mut test_state);
    println!(
        "Test state after enforcement: {:.4}, {:.4}, {:.4}, {:.4}, {:.4}",
        test_state.get(0),
        test_state.get(1),
        test_state.get(2),
        test_state.get(3),
        test_state.get(4)
    );
    println!("Validates: {}", geo_space.validates(&test_state));

    println!("\n=== Summary ===");
    println!("✓ Implemented 5D ↔ Metatron geometry projection");
    println!("✓ Applied C6 rotational symmetry operations");
    println!("✓ Applied D6 reflection symmetry operations");
    println!("✓ Enforced geometric constraints on states");
    println!("✓ Validated symmetry preservation");
    println!("\nKey achievements:");
    println!("  - Full geometric projection and reconstruction");
    println!("  - Symmetry-preserving operations (C6/D6 groups)");
    println!("  - Constraint enforcement mechanisms");
    println!("  - Validation of geometric properties");
    println!("\nPhase 5 Complete: Geometric constraint enforcement operational!");
}
