//! Self-Tuning Ecology Example
//!
//! This example demonstrates Phase 4: Cognitive feedback loop connecting
//! QLogic spectral analysis with QDASH parameter tuning.
//!
//! The system implements a predator-prey ecosystem where:
//! - Spectral analysis monitors trajectory patterns
//! - QDASH adapts system parameters based on observed dynamics
//! - The system self-tunes to maintain stable oscillations

use bridge::{CognitiveSimulator, ParameterTuner, SpectralAnalyzer, TrajectoryObserver};
use core_5d::*;

fn main() {
    println!("=== Self-Tuning Ecology Example ===\n");
    println!("Demonstrating cognitive feedback loop:");
    println!("  QLogic spectral analysis → QDASH parameter tuning\n");

    // Create predator-prey ecosystem template
    let template = Template::predator_prey(0.5, 0.3, 0.1);
    let coupling = template.coupling_matrix;
    let mut params = template.parameters;

    println!("Initial ecosystem parameters:");
    println!("  Prey growth rate: {:.3}", params.intrinsic_rates[0]);
    println!("  Predator death rate: {:.3}", params.intrinsic_rates[1]);
    println!();

    // Phase 1: Run without adaptation to establish baseline
    println!("=== Phase 1: Standard Simulation (no adaptation) ===");
    let vf = VectorField::new(coupling.clone(), params.clone());
    let time_config = integration::TimeConfig::new(0.1, 0.0, 50.0);
    let integrator = Integrator::new(vf, time_config);
    let initial = State5D::new(1.0, 0.5, 0.0, 0.0, 0.0); // Prey, Predators

    let trajectory = integrator.integrate(initial);
    println!("Trajectory length: {} steps", trajectory.len());
    println!(
        "Final state: Prey={:.4}, Predators={:.4}",
        trajectory.last().unwrap().1.get(0),
        trajectory.last().unwrap().1.get(1)
    );

    // Analyze baseline trajectory
    let mut observer = TrajectoryObserver::new(500);
    for (_, state) in &trajectory {
        observer.observe(*state);
    }

    let analyzer = SpectralAnalyzer::new();
    let baseline_entropy = analyzer.average_entropy(&observer);
    let baseline_centroids = analyzer.spectral_centroids(&observer);

    println!("\nBaseline spectral analysis:");
    println!("  Average entropy: {:.4}", baseline_entropy);
    println!("  Prey spectral centroid: {:.4}", baseline_centroids[0].1);
    println!(
        "  Predator spectral centroid: {:.4}",
        baseline_centroids[1].1
    );

    // Phase 2: Run with cognitive feedback
    println!("\n=== Phase 2: Cognitive Simulation (with feedback) ===");

    let vf2 = VectorField::new(coupling.clone(), params.clone());
    let time_config2 = integration::TimeConfig::new(0.1, 0.0, 50.0);
    let integrator2 = Integrator::new(vf2, time_config2);
    let observer2 = TrajectoryObserver::new(500);

    let mut sim = CognitiveSimulator::new(integrator2, observer2);
    let mut tuner = ParameterTuner::default_config().with_learning_rate(0.05);

    // Integrate with periodic parameter tuning
    let mut cognitive_trajectory = Vec::new();
    let mut current_state = initial;
    let mut current_time = 0.0;
    let dt = 0.1;
    let tune_interval = 5.0; // Tune every 5 time units
    let mut last_tune_time = 0.0;

    cognitive_trajectory.push((current_time, current_state));
    sim.observer_mut().observe(current_state);

    println!("Integrating with adaptive parameter tuning...");
    let mut tune_count = 0;

    while current_time < 50.0 {
        // Step the simulation
        current_state = sim.step(current_state, dt);
        current_time += dt;
        cognitive_trajectory.push((current_time, current_state));

        // Periodically tune parameters based on trajectory analysis
        if current_time - last_tune_time >= tune_interval && sim.observer().history().len() > 20 {
            let adjustments = tuner.suggest_adjustments(sim.observer(), &params);

            // Apply adjustments to parameters
            for i in 0..5 {
                params.intrinsic_rates[i] += adjustments[i];
            }

            // Update vector field with new parameters
            let new_vf = VectorField::new(coupling.clone(), params.clone());
            sim = CognitiveSimulator::new(
                Integrator::new(new_vf, integration::TimeConfig::new(dt, current_time, 50.0)),
                TrajectoryObserver::new(500),
            );

            last_tune_time = current_time;
            tune_count += 1;

            println!(
                "  Tuning at t={:.1}, QDASH resonance={:.4}, adjustments applied",
                current_time,
                tuner.resonance()
            );
        }

        if !current_state.is_valid() {
            println!("Warning: Invalid state at t={:.1}, stopping", current_time);
            break;
        }
    }

    println!("Parameter tuning cycles: {}", tune_count);
    println!("Final parameters:");
    println!("  Prey growth rate: {:.3}", params.intrinsic_rates[0]);
    println!("  Predator death rate: {:.3}", params.intrinsic_rates[1]);

    // Analyze cognitive trajectory
    let mut observer_final = TrajectoryObserver::new(500);
    for (_, state) in &cognitive_trajectory {
        observer_final.observe(*state);
    }

    let cognitive_entropy = analyzer.average_entropy(&observer_final);
    let cognitive_centroids = analyzer.spectral_centroids(&observer_final);

    println!("\nCognitive trajectory spectral analysis:");
    println!("  Average entropy: {:.4}", cognitive_entropy);
    println!("  Prey spectral centroid: {:.4}", cognitive_centroids[0].1);
    println!(
        "  Predator spectral centroid: {:.4}",
        cognitive_centroids[1].1
    );

    println!(
        "\nFinal state: Prey={:.4}, Predators={:.4}",
        cognitive_trajectory.last().unwrap().1.get(0),
        cognitive_trajectory.last().unwrap().1.get(1)
    );

    // Show trajectory samples
    println!("\n=== Trajectory Evolution (Cognitive) ===");
    println!("Time\t\tPrey\t\tPredators");
    println!("{}", "-".repeat(50));
    for &t in &[0.0, 10.0, 20.0, 30.0, 40.0, 50.0] {
        if let Some((_, state)) = cognitive_trajectory
            .iter()
            .find(|(time, _)| (*time - t).abs() < 0.15)
        {
            println!("{:.1}\t\t{:.4}\t\t{:.4}", t, state.get(0), state.get(1));
        }
    }

    println!("\n=== Summary ===");
    println!("✓ Integrated QLogic spectral analysis with trajectory observation");
    println!("✓ Connected QDASH parameter tuning to spectral features");
    println!(
        "✓ Demonstrated adaptive feedback loop ({} tuning cycles)",
        tune_count
    );
    println!("✓ System self-tuned parameters based on dynamics");
    println!("\nSpectral changes:");
    println!(
        "  Entropy change: {:.4} → {:.4} (Δ={:.4})",
        baseline_entropy,
        cognitive_entropy,
        cognitive_entropy - baseline_entropy
    );
    println!("\nPhase 4 Complete: Cognitive feedback loop operational!");
}
