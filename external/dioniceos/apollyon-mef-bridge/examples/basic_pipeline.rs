//! Basic Pipeline Example
//!
//! Demonstrates the complete APOLLYON-5D → MEF-Core processing pipeline.
//!
//! # Pipeline Stages
//! 1. APOLLYON-5D: 5D integration
//! 2. APOLLYON-5D: Spectral analysis
//! 3. Bridge: State conversion
//! 4. MEF: Route selection
//! 5. MEF: Knowledge derivation
//! 6. Bridge: Proof-of-Resonance
//! 7. MEF: Gate evaluation
//! 8. MEF: Storage (if FIRE)
//!
//! Run with: cargo run --example basic_pipeline

use apollyon_mef_bridge::{CognitiveInput, UnifiedCognitiveEngine};
use core_5d::{State5D, SystemParameters};

fn main() {
    println!("=== APOLLYON-MEF Unified Cognitive Engine Demo ===\n");

    // Create the unified cognitive engine
    let mut engine = UnifiedCognitiveEngine::new();
    println!("✓ Unified Cognitive Engine initialized\n");

    // Define initial 5D state
    let initial_state = State5D::new(1.0, 0.5, 0.3, 0.2, 0.1);
    println!("Initial 5D State:");
    println!("  x={:.3}, y={:.3}, z={:.3}, ψ={:.3}, ω={:.3}\n",
        initial_state.get(0), initial_state.get(1), initial_state.get(2),
        initial_state.get(3), initial_state.get(4));

    // Define system parameters (weak coupling)
    let parameters = SystemParameters::new(
        [-0.1, -0.15, 0.1, 0.0, -0.05], // intrinsic rates
        [0.0, 0.0, 0.0, 0.0, 0.0],       // no external forcing
    );

    // Create cognitive input
    let input = CognitiveInput {
        initial_state,
        parameters,
        t_final: 2.0,
        tic_id: "TIC-DEMO-001".to_string(),
        seed: "demo_seed_12345".to_string(),
        seed_path: "MEF/demo/example/0001".to_string(),
    };

    println!("Processing Configuration:");
    println!("  Integration time: {:.1}s", input.t_final);
    println!("  TIC ID: {}", input.tic_id);
    println!("  Seed: {}", input.seed);
    println!("  Seed path: {}\n", input.seed_path);

    // Process through the complete pipeline
    println!("Processing through unified pipeline...\n");
    let result = engine.process(input);

    match result {
        Ok(output) => {
            println!("✓ Pipeline completed successfully!\n");

            // Display trajectory information
            println!("=== APOLLYON Integration Results ===");
            println!("Trajectory length: {} states", output.trajectory.len());
            let final_state = output.trajectory.last().unwrap();
            println!("Final 5D state:");
            println!("  x={:.6}, y={:.6}, z={:.6}, ψ={:.6}, ω={:.6}\n",
                final_state.get(0), final_state.get(1), final_state.get(2),
                final_state.get(3), final_state.get(4));

            // Display spectral analysis results
            println!("=== Spectral Analysis ===");
            println!("Spectral Signature:");
            println!("  ψ (psi):   {:.6} - Phase alignment", output.spectral_signature.psi);
            println!("  ρ (rho):   {:.6} - Resonance (1-entropy)", output.spectral_signature.rho);
            println!("  ω (omega): {:.6} - Oscillation frequency\n", output.spectral_signature.omega);

            // Display route selection results
            println!("=== MEF Route Selection ===");
            println!("Route ID: {}", output.route.route_id);
            println!("Permutation: {:?}", output.route.permutation);
            println!("Mesh Score: {:.6}\n", output.route.mesh_score);

            // Display proof-of-resonance
            println!("=== Proof-of-Resonance ===");
            println!("PoR Valid: {}", output.proof.por_valid);
            println!("ΔPI (Path Invariance): {:.6}", output.proof.delta_pi);
            println!("Φ (Alignment):         {:.6}", output.proof.phi);
            println!("ΔV (Lyapunov Delta):   {:.6}\n", output.proof.delta_v);

            // Display gate decision
            println!("=== Gate Evaluation ===");
            match output.gate_decision {
                mef_schemas::GateDecision::FIRE => {
                    println!("Gate Decision: 🔥 FIRE");
                    println!("  ✓ PoR is valid");
                    println!("  ✓ ΔPI ≤ ε (path invariance threshold)");
                    println!("  ✓ Φ ≥ φ (alignment threshold)");
                    println!("  ✓ ΔV < 0 (energy decreasing)");
                    println!("  → Knowledge will be stored in ledger\n");
                }
                mef_schemas::GateDecision::HOLD => {
                    println!("Gate Decision: ⏸️  HOLD");
                    println!("  One or more conditions not met:");
                    if !output.proof.por_valid {
                        println!("  ✗ PoR is invalid");
                    }
                    if output.proof.delta_pi > 0.1 {
                        println!("  ✗ ΔPI > ε (path invariance too large)");
                    }
                    if output.proof.phi < 0.5 {
                        println!("  ✗ Φ < φ (alignment too low)");
                    }
                    if output.proof.delta_v >= 0.0 {
                        println!("  ✗ ΔV ≥ 0 (energy not decreasing)");
                    }
                    println!("  → Knowledge will NOT be stored\n");
                }
            }

            // Display knowledge object
            if let Some(knowledge) = output.knowledge {
                println!("=== Knowledge Object ===");
                println!("MEF ID:    {}", knowledge.mef_id);
                println!("TIC ID:    {}", knowledge.tic_id);
                println!("Route ID:  {}", knowledge.route_id);
                println!("Seed Path: {}", knowledge.seed_path);
                println!("Payload:   {} bytes\n", 
                    knowledge.payload.as_ref()
                        .map(|p| serde_json::to_string(p).unwrap().len())
                        .unwrap_or(0)
                );
            }

            println!("=== Pipeline Summary ===");
            println!("✓ 5D Integration: {} states computed", output.trajectory.len());
            println!("✓ Spectral Analysis: Signature extracted");
            println!("✓ State Conversion: 5D → Spiral coordinates");
            println!("✓ Route Selection: S7 route determined");
            println!("✓ Knowledge Derivation: Object created");
            println!("✓ Proof-of-Resonance: Metrics computed");
            println!("✓ Gate Evaluation: Decision made");

            match output.gate_decision {
                mef_schemas::GateDecision::FIRE => {
                    println!("✓ Storage: Knowledge stored in ledger");
                }
                mef_schemas::GateDecision::HOLD => {
                    println!("○ Storage: Knowledge held (not stored)");
                }
            }

            println!("\n=== Complete! ===");
        }
        Err(e) => {
            eprintln!("✗ Pipeline failed: {}", e);
            std::process::exit(1);
        }
    }
}
