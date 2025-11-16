//! Simple example demonstrating basic 5D Cube overlay usage

use unified_5d_cube::{InterlockConfig, InterlockAdapter, tick_5d_cube};
use core_5d::State5D;

fn main() {
    println!("=== Unified 5D Cube - Simple Example ===\n");
    
    // Create configuration with default settings
    let config = InterlockConfig::default();
    println!("Configuration:");
    println!("  Seed: {}", config.seed);
    println!("  Phi threshold: {}", config.gate_phi_threshold);
    println!("  Delta PI max: {}", config.gate_delta_pi_max);
    println!("  Shadow mode: {}\n", config.shadow_mode);
    
    // Create interlock adapter
    let mut adapter = InterlockAdapter::new(config);
    
    // Initial state
    let state0 = State5D::from_array([1.0, 0.5, 0.3, 0.7, 0.4]);
    println!("Initial state: {:?}\n", state0.to_array());
    
    // Execute first tick (no previous state)
    println!("--- Tick 0 ---");
    let result0 = tick_5d_cube(&mut adapter, &state0, None, 0.0, 0);
    println!("Gate decision: {:?}", result0.gate_decision);
    println!("PoR valid: {}", result0.proof.por_valid);
    println!("Metrics:");
    println!("  BI: {:.4}", result0.metrics.bi);
    println!("  ΔF: {:.4}", result0.metrics.delta_f);
    println!("  W2_step: {:.4}", result0.metrics.w2_step);
    println!("  λ_gap: {:.4}", result0.metrics.lambda_gap);
    println!("  S_mand: {:.4}", result0.metrics.s_mand);
    println!("  Duty/PoR: {:.4}", result0.metrics.duty_por);
    println!();
    
    // Execute second tick (with previous state)
    let state1_arr = result0.state_condensed.as_array();
    let state1 = State5D::from_array(state1_arr);
    
    println!("--- Tick 1 ---");
    let result1 = tick_5d_cube(&mut adapter, &state1, Some(&state0), 0.1, 1);
    println!("Gate decision: {:?}", result1.gate_decision);
    println!("PoR valid: {}", result1.proof.por_valid);
    println!("Proof:");
    println!("  ΔPI: {:.4}", result1.proof.delta_pi);
    println!("  Φ: {:.4}", result1.proof.phi);
    println!("  ΔV: {:.4}", result1.proof.delta_v);
    println!("Metrics:");
    println!("  BI: {:.4}", result1.metrics.bi);
    println!("  ΔF: {:.4}", result1.metrics.delta_f);
    println!("  W2_step: {:.4}", result1.metrics.w2_step);
    println!();
    
    // Check if commit would be made
    if let Some(commit) = &result1.commit {
        println!("Commit prepared:");
        println!("  Hash: {}", commit.commit_hash);
        println!("  Timestamp: {}", commit.timestamp);
    } else {
        println!("No commit (gate HOLD)");
    }
    
    println!("\n=== Example Complete ===");
}
