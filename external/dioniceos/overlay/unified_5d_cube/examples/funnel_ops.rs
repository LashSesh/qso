//! Example demonstrating Funnel operations (split/merge/prune)
//!
//! Run with: cargo run --example funnel_ops

use unified_5d_cube::{InterlockAdapter, InterlockConfig, tick_5d_cube};
use core_5d::State5D;
use apollyon_mef_bridge::trichter::Policy;

fn main() {
    // Enable full Funnel operations
    let mut config = InterlockConfig::default();
    config.enable_funnel_ops = true;
    config.funnel_policy = Policy::Explore; // High hebbian, preserves diversity
    config.enable_logging = true;
    
    println!("=== Funnel Operations Example ===\n");
    println!("Configuration:");
    println!("  - Funnel Ops: {}", config.enable_funnel_ops);
    println!("  - Policy: {:?}", config.funnel_policy);
    println!("  - Seed: {}\n", config.seed);
    
    let mut adapter = InterlockAdapter::new(config);
    
    // Run a sequence of ticks with varying states
    let mut state = State5D::from_array([2.0, 1.5, 1.0, 0.8, 0.6]);
    
    println!("Running 10 ticks with Funnel split/merge/prune...\n");
    
    for i in 0..10 {
        let prev_state = if i > 0 { Some(&state) } else { None };
        
        // Larger variations to trigger funnel operations
        let variation = (i as f64 * 0.1).sin() * 0.2;
        let next_state = State5D::from_array([
            state.to_array()[0] * (1.0 - variation),
            state.to_array()[1] * (1.0 - variation * 0.8),
            state.to_array()[2] * (1.0 - variation * 0.6),
            state.to_array()[3] * (1.0 - variation * 0.4),
            state.to_array()[4] * (1.0 - variation * 0.2),
        ]);
        
        let result = tick_5d_cube(&mut adapter, &next_state, prev_state, i as f64 * 0.1, i);
        
        println!("Tick {}: ", i);
        println!("  State norm: {:.6}", next_state.norm());
        println!("  Condensed norm: {:.6}", result.state_condensed.norm());
        println!("  W2_step: {:.6}", result.metrics.w2_step);
        println!("  S_mand (coherence): {:.6}", result.metrics.s_mand);
        println!("  Gate: {:?}", result.gate_decision);
        println!();
        
        state = next_state;
    }
    
    println!("=== Funnel Operations Complete ===");
    println!("\nNote: Funnel operations apply Hebbian learning, decay, and");
    println!("structural operations (split/merge/prune) to optimize the graph topology.");
}
