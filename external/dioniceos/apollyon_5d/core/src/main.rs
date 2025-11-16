use core_5d::*;

fn main() {
    println!("=== 5D System Framework Demo ===\n");

    // Run validation tests
    println!("Running validation tests...");
    let all_passed = validation::run_all_tests();
    println!(
        "\nValidation: {}\n",
        if all_passed { "✓ PASS" } else { "✗ FAIL" }
    );

    // Demo 1: SIR Model
    println!("=== Demo 1: SIR Epidemiological Model ===");
    demo_sir_model();

    // Demo 2: Financial Market
    println!("\n=== Demo 2: Financial Market Model ===");
    demo_financial_model();

    // Demo 3: Predator-Prey
    println!("\n=== Demo 3: Predator-Prey Ecosystem ===");
    demo_predator_prey();

    // Demo 4: Ensemble Simulation
    println!("\n=== Demo 4: Ensemble Simulation ===");
    demo_ensemble();

    // Demo 5: Parameter Sweep
    println!("\n=== Demo 5: Parameter Sweep ===");
    demo_parameter_sweep();

    println!("\n=== Framework Demo Complete ===");
}

fn demo_sir_model() {
    // Create SIR model: β=0.3 (transmission), γ=0.1 (recovery), μ=0.01 (death)
    let template = template::Template::sir_model(0.3, 0.1, 0.01);
    let vf = template.to_vector_field();

    // Initial conditions: 99% susceptible, 1% infected
    let initial = State5D::new(0.99, 0.01, 0.0, 0.0, 0.0);

    // Integrate for 100 time units with dt=0.1
    let time_config = integration::TimeConfig::new(0.1, 0.0, 100.0);
    let integrator = integration::Integrator::new(vf, time_config);

    let trajectory = integrator.integrate(initial);

    // Print key statistics
    let final_state = trajectory.last().unwrap().1;
    println!(
        "Initial: S={:.3}, I={:.3}, R={:.3}",
        initial.get(0),
        initial.get(1),
        initial.get(2)
    );
    println!(
        "Final:   S={:.3}, I={:.3}, R={:.3}, D={:.3}",
        final_state.get(0),
        final_state.get(1),
        final_state.get(2),
        final_state.get(4)
    );

    // Export to CSV
    let traj_export = export::Trajectory::from_pairs(trajectory);
    if let Err(e) = traj_export.export_csv("/tmp/sir_trajectory.csv") {
        eprintln!("Warning: Could not export CSV: {}", e);
    } else {
        println!("Exported to: /tmp/sir_trajectory.csv");
    }
}

fn demo_financial_model() {
    // Create financial model with moderate volatility and momentum
    let template = template::Template::financial_market(0.2, 0.1, 0.05);
    let vf = template.to_vector_field();

    // Initial conditions: normalized prices and volumes
    let initial = State5D::new(1.0, 0.5, 0.0, 1.0, 0.1);

    // Integrate for 50 time units
    let time_config = integration::TimeConfig::new(0.05, 0.0, 50.0);
    let integrator = integration::Integrator::new(vf, time_config);

    let states = integrator.integrate_states(initial);

    println!("Simulated {} time steps", states.len());
    println!("Initial Price: {:.3}", initial.get(0));
    println!("Final Price: {:.3}", states.last().unwrap().get(0));
    println!("Final Risk: {:.3}", states.last().unwrap().get(4));

    // Analyze stability at initial state
    let jacobian = integrator.vector_field.jacobian(&initial);
    let eigenvalues = stability::StabilityAnalyzer::eigenvalues(&jacobian);
    let stability_type = stability::StabilityAnalyzer::classify_stability(&eigenvalues);

    println!("Stability at initial state: {:?}", stability_type);
    println!("Largest eigenvalue (real): {:.3}", eigenvalues[0]);
}

fn demo_predator_prey() {
    // Create predator-prey model with very conservative parameters
    let template = template::Template::predator_prey(0.1, 0.05, 0.05);
    let vf = template.to_vector_field();

    // Initial conditions: small populations
    let initial = State5D::new(0.3, 0.2, 0.1, 1.0, 1.0);

    // Integrate for 50 time units with small timestep
    let time_config = integration::TimeConfig::new(0.001, 0.0, 50.0);
    let integrator = integration::Integrator::new(vf, time_config);

    let states = integrator.integrate_states(initial);

    // Check if integration was successful
    if states.is_empty() || !states.last().unwrap().is_valid() {
        println!("Warning: Integration became unstable");
        return;
    }

    // Find extrema (for oscillations) - only use valid states
    let valid_states: Vec<&State5D> = states.iter().filter(|s| s.is_valid()).collect();

    if valid_states.is_empty() {
        println!("Warning: No valid states found");
        return;
    }

    let prey1_values: Vec<f64> = valid_states.iter().map(|s| s.get(0)).collect();
    let max_prey1 = prey1_values
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_prey1 = prey1_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    println!(
        "Prey1 oscillates between {:.3} and {:.3}",
        min_prey1, max_prey1
    );

    // Demonstrate 2D projection
    let projector = projection::Projector::orthogonal(0, 2); // Prey1 vs Predator
    let points = projector.project_many(&valid_states.iter().map(|&&s| s).collect::<Vec<_>>());

    println!("Generated {} 2D projection points", points.len());

    if !points.is_empty() {
        println!(
            "Phase space range: x=[{:.3}, {:.3}]",
            points
                .iter()
                .map(|p| p.x)
                .fold(f64::INFINITY, |a, b| a.min(b)),
            points
                .iter()
                .map(|p| p.x)
                .fold(f64::NEG_INFINITY, |a, b| a.max(b))
        );
    }
}

fn demo_ensemble() {
    // Create SIR model
    let template = template::Template::sir_model(0.3, 0.1, 0.01);
    let vf = template.to_vector_field();

    // Configure ensemble with slight variation in initial conditions
    let mean_state = State5D::new(0.99, 0.01, 0.0, 0.0, 0.0);
    let config = ensemble::EnsembleConfig::new(20, mean_state, 0.005);

    // Time configuration
    let time_config = integration::TimeConfig::new(0.1, 0.0, 50.0);

    // Run ensemble
    println!(
        "Running {} simulations with randomized initial conditions...",
        config.num_runs
    );
    let result = ensemble::run_ensemble(&config, &vf, &time_config);

    // Report statistics
    println!("Generated {} trajectories", result.trajectories.len());

    if !result.mean_trajectory.is_empty() {
        let final_mean = result.mean_trajectory.last().unwrap();
        let final_std = result.std_trajectory.last().unwrap();

        println!("\nFinal state statistics:");
        println!(
            "  Susceptible: {:.3} ± {:.3}",
            final_mean.get(0),
            final_std.get(0)
        );
        println!(
            "  Infected:    {:.3} ± {:.3}",
            final_mean.get(1),
            final_std.get(1)
        );
        println!(
            "  Recovered:   {:.3} ± {:.3}",
            final_mean.get(2),
            final_std.get(2)
        );
        println!(
            "  Deceased:    {:.3} ± {:.3}",
            final_mean.get(4),
            final_std.get(4)
        );
    }
}

fn demo_parameter_sweep() {
    // Create base SIR template
    let template = template::Template::sir_model(0.3, 0.1, 0.01);

    // Sweep coupling strength from 0.5 to 1.5
    let sweep = ensemble::ParameterSweep::new("coupling".to_string(), 0.5, 1.5, 5);

    // Initial state
    let initial = State5D::new(0.99, 0.01, 0.0, 0.0, 0.0);

    // Time configuration
    let time_config = integration::TimeConfig::new(0.1, 0.0, 30.0);

    // Run sweep
    println!(
        "Running parameter sweep with {} values...",
        sweep.values.len()
    );
    let results = ensemble::run_parameter_sweep(&sweep, &template, initial, &time_config);

    // Report final states for each parameter value
    println!("\nParameter sweep results:");
    for (i, traj) in results.iter().enumerate() {
        if let Some(final_state) = traj.last() {
            println!(
                "  {}: {:.2} -> Final I={:.4}, R={:.4}",
                sweep.parameter_name,
                sweep.values[i],
                final_state.get(1),
                final_state.get(2)
            );
        }
    }
}
