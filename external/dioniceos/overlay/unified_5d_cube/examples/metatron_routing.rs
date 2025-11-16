//! Example demonstrating Metatron S7 router integration
//!
//! Run with: cargo run --example metatron_routing

use unified_5d_cube::{InterlockAdapter, InterlockConfig, tick_5d_cube};
use core_5d::State5D;
use std::path::PathBuf;
use tempfile::TempDir;

fn main() {
    // Create a temporary directory for router storage
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let router_path = temp_dir.path().join("metatron_router");
    
    println!("=== Metatron S7 Router Example ===\n");
    println!("Router path: {:?}\n", router_path);
    
    // Enable Metatron routing
    let mut config = InterlockConfig::default();
    config.enable_metatron_routing = true;
    config.ledger_path = Some(router_path.clone());
    config.enable_logging = true;
    
    println!("Configuration:");
    println!("  - Metatron Routing: {}", config.enable_metatron_routing);
    println!("  - Seed: {}\n", config.seed);
    
    let mut adapter = InterlockAdapter::new(config);
    
    // Run ticks with different state trajectories
    let mut state = State5D::from_array([1.5, 1.0, 0.5, 0.8, 0.6]);
    
    println!("Running 8 ticks with Metatron route selection...\n");
    
    for i in 0..8 {
        let prev_state = if i > 0 { Some(&state) } else { None };
        
        // Different dynamics for different phases
        let phase = i / 3;
        let factor = match phase {
            0 => 0.95, // Convergence
            1 => 1.02, // Divergence
            _ => 0.98, // Stabilization
        };
        
        let next_state = State5D::from_array([
            state.to_array()[0] * factor,
            state.to_array()[1] * (factor - 0.01),
            state.to_array()[2] * (factor - 0.02),
            state.to_array()[3] * (factor - 0.03),
            state.to_array()[4] * (factor - 0.04),
        ]);
        
        let result = tick_5d_cube(&mut adapter, &next_state, prev_state, i as f64 * 0.1, i);
        
        println!("Tick {} (Phase {}):", i, phase);
        println!("  State norm: {:.6}", next_state.norm());
        
        if let Some(ref route) = result.selected_route {
            println!("  Selected route: {:?}", route);
            println!("  Operators: {}", route.join(" → "));
        } else {
            println!("  Selected route: None (routing disabled)");
        }
        
        println!("  Metrics:");
        println!("    ΔF: {:.6}", result.metrics.delta_f);
        println!("    λ_gap: {:.6}", result.metrics.lambda_gap);
        println!("    S_mand: {:.6}", result.metrics.s_mand);
        println!("  Gate: {:?}", result.gate_decision);
        println!();
        
        state = next_state;
    }
    
    println!("=== Metatron Routing Complete ===");
    println!("\nNote: Metatron router uses S7 permutations over the 13-node");
    println!("Metatron Cube topology to select optimal transformation routes.");
    println!("Routes are selected based on resonance metrics and system state.");
}
