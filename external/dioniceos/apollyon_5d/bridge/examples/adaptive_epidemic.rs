//! Adaptive Epidemic Model Example
//!
//! This example demonstrates the integration of the 5D framework with Metatron-R
//! cognition by implementing an SIR epidemic model where transmission rates
//! adapt based on Mandorla resonance fields.
//!
//! The system models:
//! - Susceptible (S)
//! - Infected (I)
//! - Recovered (R)
//! - Exposed (E)
//! - Deceased (D)
//!
//! With adaptive coupling that responds to resonance patterns.

use bridge::{AdaptiveCoupling, CognitiveSimulator, TrajectoryObserver};
use core_5d::*;

fn main() {
    println!("=== Adaptive Epidemic Model ===\n");
    println!("Demonstrating 5D + Metatron-R integration");
    println!("SIR model with resonance-based coupling adaptation\n");

    // Create base SIR model template
    let template = Template::sir_model(0.3, 0.1, 0.01);
    let base_coupling = template.coupling_matrix;
    let params = template.parameters;

    // Create Mandorla resonance field for adaptive coupling
    let resonance = bridge::OscillatoryResonanceField::new(0.3, 0.5, 0.0);

    println!("Using oscillatory resonance field (frequency=0.5 Hz, amplitude=0.3)\n");

    // Create adaptive coupling system
    let adaptive = AdaptiveCoupling::new(base_coupling.clone(), Box::new(resonance));

    // Show how coupling changes over time
    println!("Coupling strength adaptation over time:");
    println!("Time\t\tS→I Coupling\tModulation");
    println!("{}", "-".repeat(50));

    for t in (0..=10).map(|i| i as f64) {
        let modulated = adaptive.compute_coupling(t);
        let s_to_i = modulated.get_strength(0, 1);
        let base_s_to_i = base_coupling.get_strength(0, 1);
        let modulation = if base_s_to_i != 0.0 {
            s_to_i / base_s_to_i
        } else {
            0.0
        };
        println!("{:.1}\t\t{:.6}\t{:.4}x", t, s_to_i, modulation);
    }

    // Run standard simulation for comparison
    println!("\n=== Standard 5D Simulation (no adaptation) ===");
    let vf_standard = VectorField::new(base_coupling.clone(), params.clone());
    let time_config = integration::TimeConfig::new(0.1, 0.0, 50.0);
    let integrator_standard = Integrator::new(vf_standard, time_config);

    let initial = State5D::new(0.99, 0.01, 0.0, 0.0, 0.0); // 99% susceptible, 1% infected
    let trajectory_standard = integrator_standard.integrate(initial);

    let final_standard = trajectory_standard.last().unwrap().1;
    println!(
        "Initial state: S={:.4}, I={:.4}, R={:.4}",
        initial.get(0),
        initial.get(1),
        initial.get(2)
    );
    println!(
        "Final state (t=50): S={:.4}, I={:.4}, R={:.4}",
        final_standard.get(0),
        final_standard.get(1),
        final_standard.get(2)
    );
    println!(
        "Peak infected: {:.4}",
        trajectory_standard
            .iter()
            .map(|(_, s)| s.get(1))
            .fold(0.0f64, f64::max)
    );

    // Create cognitive simulator with adaptive coupling
    println!("\n=== Cognitive Simulation (with adaptive coupling) ===");
    println!("Full adaptive integration with time-varying coupling...\n");

    let vf_cognitive = VectorField::new(base_coupling.clone(), params);
    let time_config_cog = integration::TimeConfig::new(0.1, 0.0, 50.0);
    let integrator_cog = Integrator::new(vf_cognitive, time_config_cog);
    let observer = TrajectoryObserver::new(100);

    // Create adaptive coupling with a different resonance for comparison
    let resonance2 = bridge::OscillatoryResonanceField::new(0.5, 1.0, 0.0);
    let adaptive2 = AdaptiveCoupling::new(base_coupling, Box::new(resonance2));

    let mut sim = CognitiveSimulator::with_adaptive_coupling(integrator_cog, observer, adaptive2);
    let trajectory_cognitive = sim.integrate_adaptive(initial);

    let final_cognitive = trajectory_cognitive.last().unwrap().1;
    println!(
        "Initial state: S={:.4}, I={:.4}, R={:.4}",
        initial.get(0),
        initial.get(1),
        initial.get(2)
    );
    println!(
        "Final state (t=50): S={:.4}, I={:.4}, R={:.4}",
        final_cognitive.get(0),
        final_cognitive.get(1),
        final_cognitive.get(2)
    );
    println!(
        "Peak infected: {:.4}",
        trajectory_cognitive
            .iter()
            .map(|(_, s)| s.get(1))
            .fold(0.0f64, f64::max)
    );

    // Show trajectory analysis
    if let Some(velocity) = sim.observer().velocity() {
        println!("\nFinal velocity norm: {:.6}", velocity.norm());
    }

    println!("Trajectory energy: {:.6}", sim.observer().energy());
    println!("System converging: {}", sim.observer().is_converging(0.001));

    // Compare trajectories
    println!("\n=== Comparison ===");
    println!("Standard vs Adaptive:");
    println!(
        "  Peak infected reduction: {:.2}%",
        100.0
            * (1.0
                - trajectory_cognitive
                    .iter()
                    .map(|(_, s)| s.get(1))
                    .fold(0.0f64, f64::max)
                    / trajectory_standard
                        .iter()
                        .map(|(_, s)| s.get(1))
                        .fold(0.0f64, f64::max))
    );
    println!(
        "  Final recovered difference: {:.4}",
        final_cognitive.get(2) - final_standard.get(2)
    );

    // Sample trajectory at key time points
    println!("\n=== Trajectory Evolution (Adaptive) ===");
    println!("Time\t\tS\t\tI\t\tR");
    println!("{}", "-".repeat(60));
    for &t in &[0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0] {
        if let Some((_, state)) = trajectory_cognitive
            .iter()
            .find(|(time, _)| (*time - t).abs() < 0.15)
        {
            println!(
                "{:.1}\t\t{:.4}\t\t{:.4}\t\t{:.4}",
                t,
                state.get(0),
                state.get(1),
                state.get(2)
            );
        }
    }

    println!("\n=== Summary ===");
    println!("✓ Successfully integrated 5D dynamics with Metatron resonance");
    println!("✓ Demonstrated full adaptive coupling modulation during integration");
    println!("✓ Showed trajectory observation and analysis");
    println!("✓ Compared standard vs cognitive simulations");
    println!("\nKey achievements:");
    println!("  - Real-time coupling adaptation based on oscillatory resonance");
    println!("  - Trajectory monitoring with velocity and energy calculation");
    println!("  - Convergence detection");
    println!("\nNext steps:");
    println!("  - Add QLogic spectral analysis");
    println!("  - Connect QDASH for parameter tuning");
    println!("  - Implement geometric constraint enforcement");
}
