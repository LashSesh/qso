//! Multiple Scenarios Example
//!
//! Demonstrates the unified pipeline with different configurations to show
//! various gate decisions (FIRE vs HOLD) and different trajectory behaviors.
//!
//! Run with: cargo run --example multiple_scenarios

use apollyon_mef_bridge::{CognitiveInput, UnifiedCognitiveEngine};
use core_5d::{State5D, SystemParameters};

fn main() {
    println!("=== APOLLYON-MEF Multiple Scenarios Demo ===\n");

    let mut engine = UnifiedCognitiveEngine::new();

    // Define multiple scenarios with different characteristics
    let scenarios = vec![
        // Scenario 1: Small initial state, weak damping
        (
            "Weak Damping",
            State5D::new(0.5, 0.3, 0.2, 0.1, 0.05),
            SystemParameters::new(
                [-0.05, -0.05, -0.05, -0.05, -0.05],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            0.5,
        ),
        // Scenario 2: Larger initial state, stronger damping
        (
            "Strong Damping",
            State5D::new(2.0, 1.5, 1.0, 0.8, 0.5),
            SystemParameters::new(
                [-0.5, -0.5, -0.5, -0.5, -0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            1.0,
        ),
        // Scenario 3: Mixed dynamics (some positive, some negative rates)
        (
            "Mixed Dynamics",
            State5D::new(1.0, 1.0, 1.0, 1.0, 1.0),
            SystemParameters::new(
                [-0.2, 0.1, -0.15, 0.05, -0.1],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            1.5,
        ),
        // Scenario 4: Very small state, very weak dynamics
        (
            "Minimal Dynamics",
            State5D::new(0.1, 0.1, 0.1, 0.1, 0.1),
            SystemParameters::new(
                [-0.01, -0.01, -0.01, -0.01, -0.01],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            2.0,
        ),
    ];

    println!("Processing {} scenarios...\n", scenarios.len());

    for (idx, (name, initial_state, parameters, t_final)) in scenarios.iter().enumerate() {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Scenario {}: {}", idx + 1, name);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let input = CognitiveInput {
            initial_state: *initial_state,
            parameters: parameters.clone(),
            t_final: *t_final,
            tic_id: format!("TIC-SCENARIO-{:03}", idx + 1),
            seed: format!("scenario_seed_{}", idx + 1),
            seed_path: format!("MEF/scenarios/{:04}", idx + 1),
        };

        println!("Initial state: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
            initial_state.get(0), initial_state.get(1), initial_state.get(2),
            initial_state.get(3), initial_state.get(4));
        println!("Integration time: {:.1}s\n", t_final);

        match engine.process(input) {
            Ok(output) => {
                let final_state = output.trajectory.last().unwrap();

                println!("Results:");
                println!("  Trajectory: {} states", output.trajectory.len());
                println!("  Final state: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
                    final_state.get(0), final_state.get(1), final_state.get(2),
                    final_state.get(3), final_state.get(4));

                // Compute state change
                let initial_norm = (0..5).map(|i| initial_state.get(i).powi(2)).sum::<f64>().sqrt();
                let final_norm = final_state.norm();
                let norm_change = final_norm - initial_norm;

                println!("\n  Spectral:");
                println!("    ρ (resonance): {:.4}", output.spectral_signature.rho);
                println!("    ω (frequency): {:.4}", output.spectral_signature.omega);

                println!("\n  Proof-of-Resonance:");
                println!("    Valid: {}", output.proof.por_valid);
                println!("    ΔPI:   {:.6}", output.proof.delta_pi);
                println!("    Φ:     {:.6}", output.proof.phi);
                println!("    ΔV:    {:.6}", output.proof.delta_v);

                println!("\n  Energy:");
                println!("    Initial norm: {:.6}", initial_norm);
                println!("    Final norm:   {:.6}", final_norm);
                println!("    Change:       {:.6}", norm_change);

                println!("\n  Gate Decision:");
                match output.gate_decision {
                    mef_schemas::GateDecision::FIRE => {
                        println!("    🔥 FIRE - Knowledge stored");
                        println!("       All conditions met!");
                    }
                    mef_schemas::GateDecision::HOLD => {
                        println!("    ⏸️  HOLD - Knowledge not stored");
                        if !output.proof.por_valid {
                            println!("       ✗ PoR invalid");
                        }
                        if output.proof.delta_pi > 0.1 {
                            println!("       ✗ Path invariance too large");
                        }
                        if output.proof.phi < 0.5 {
                            println!("       ✗ Alignment too low");
                        }
                        if output.proof.delta_v >= 0.0 {
                            println!("       ✗ Energy not decreasing");
                        }
                    }
                }

                println!("\n  Route: {}", output.route.route_id);
                println!("  Mesh score: {:.6}", output.route.mesh_score);
            }
            Err(e) => {
                println!("  ✗ Error: {}", e);
            }
        }

        println!();
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Summary");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Re-run to collect statistics
    let mut fire_count = 0;
    let mut hold_count = 0;

    for (idx, (_, initial_state, parameters, t_final)) in scenarios.iter().enumerate() {
        let input = CognitiveInput {
            initial_state: *initial_state,
            parameters: parameters.clone(),
            t_final: *t_final,
            tic_id: format!("TIC-SCENARIO-{:03}", idx + 1),
            seed: format!("scenario_seed_{}", idx + 1),
            seed_path: format!("MEF/scenarios/{:04}", idx + 1),
        };

        if let Ok(output) = engine.process(input) {
            match output.gate_decision {
                mef_schemas::GateDecision::FIRE => fire_count += 1,
                mef_schemas::GateDecision::HOLD => hold_count += 1,
            }
        }
    }

    println!("Processed {} scenarios", scenarios.len());
    println!("  🔥 FIRE: {} ({:.1}%)", fire_count, 
        100.0 * fire_count as f64 / scenarios.len() as f64);
    println!("  ⏸️  HOLD: {} ({:.1}%)", hold_count,
        100.0 * hold_count as f64 / scenarios.len() as f64);

    println!("\n=== Demo Complete! ===");
}
