//! Example demonstrating full HDAG relaxation with Hyperbion fields
//!
//! Run with: cargo run --example full_hdag

use unified_5d_cube::{InterlockAdapter, InterlockConfig, tick_5d_cube};
use core_5d::State5D;

fn main() {
    // Enable full HDAG relaxation
    let mut config = InterlockConfig::default();
    config.enable_full_hdag = true;
    config.enable_logging = true;
    
    println!("=== Full HDAG Relaxation Example ===\n");
    println!("Configuration:");
    println!("  - Full HDAG: {}", config.enable_full_hdag);
    println!("  - Seed: {}\n", config.seed);
    
    let mut adapter = InterlockAdapter::new(config);
    
    // Run a sequence of ticks
    let mut state = State5D::from_array([1.0, 0.5, 0.3, 0.7, 0.4]);
    
    println!("Running 5 ticks with HDAG relaxation...\n");
    
    for i in 0..5 {
        let prev_state = if i > 0 { Some(&state) } else { None };
        
        // Small perturbation for next state
        let next_state = State5D::from_array([
            state.to_array()[0] * 0.98,
            state.to_array()[1] * 0.97,
            state.to_array()[2] * 0.96,
            state.to_array()[3] * 0.99,
            state.to_array()[4] * 0.95,
        ]);
        
        let result = tick_5d_cube(&mut adapter, &next_state, prev_state, i as f64 * 0.1, i);
        
        println!("Tick {}: ", i);
        println!("  State norm: {:.6}", next_state.norm());
        println!("  Guidance magnitude: {:.6}", 
                 result.guidance.iter().map(|x| x*x).sum::<f64>().sqrt());
        println!("  ΔF: {:.6}", result.metrics.delta_f);
        println!("  Gate: {:?}", result.gate_decision);
        println!();
        
        state = next_state;
    }
    
    println!("=== HDAG Relaxation Complete ===");
    println!("\nNote: Full HDAG uses Hyperbion field computation and phase gradients");
    println!("to compute guidance vectors, providing more accurate field dynamics.");
}
