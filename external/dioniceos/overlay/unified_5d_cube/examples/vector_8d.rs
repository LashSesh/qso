//! Example demonstrating 8D vector pipeline for knowledge derivation
//!
//! Run with: cargo run --example vector_8d

use unified_5d_cube::{InterlockAdapter, InterlockConfig, tick_5d_cube};
use core_5d::State5D;

fn main() {
    // Enable 8D vector derivation
    let mut config = InterlockConfig::default();
    config.enable_8d_vectors = true;
    config.enable_logging = true;
    
    println!("=== 8D Vector Pipeline Example ===\n");
    println!("Configuration:");
    println!("  - 8D Vectors: {}", config.enable_8d_vectors);
    println!("  - Seed: {}\n", config.seed);
    
    let mut adapter = InterlockAdapter::new(config);
    
    // Run ticks with various states
    let test_states = vec![
        State5D::from_array([1.0, 0.0, 0.0, 0.5, 0.3]),
        State5D::from_array([0.8, 0.6, 0.0, 0.7, 0.4]),
        State5D::from_array([0.5, 0.5, 0.5, 0.8, 0.5]),
        State5D::from_array([0.3, 0.4, 0.5, 0.9, 0.6]),
        State5D::from_array([0.1, 0.2, 0.3, 0.95, 0.7]),
    ];
    
    println!("Running {} ticks with 8D vector derivation...\n", test_states.len());
    
    for (i, state) in test_states.iter().enumerate() {
        let prev_state = if i > 0 { Some(&test_states[i - 1]) } else { None };
        
        let result = tick_5d_cube(&mut adapter, state, prev_state, i as f64 * 0.1, i as u64);
        
        println!("Tick {}: ", i);
        println!("  5D State: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
                 state.to_array()[0],
                 state.to_array()[1],
                 state.to_array()[2],
                 state.to_array()[3],
                 state.to_array()[4]);
        
        if let Some(ref vec8) = result.vector_8d {
            println!("  8D Vector (normalized):");
            println!("    Spiral: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                     vec8[0], vec8[1], vec8[2], vec8[3], vec8[4]);
            println!("    Spectral: [{:.4}, {:.4}, {:.4}]",
                     vec8[5], vec8[6], vec8[7]);
            
            // Verify normalization
            let norm: f64 = vec8.iter().map(|x| x * x).sum::<f64>().sqrt();
            println!("    Norm: {:.6} (should be ~1.0)", norm);
        } else {
            println!("  8D Vector: None");
        }
        
        println!("  Metrics:");
        println!("    ΔF: {:.6}", result.metrics.delta_f);
        println!("    S_mand: {:.6}", result.metrics.s_mand);
        println!();
    }
    
    println!("=== 8D Vector Pipeline Complete ===");
    println!("\nNote: 8D vectors combine 5D spiral coordinates with spectral");
    println!("signature (ψ, ρ, ω) for knowledge derivation. They are normalized");
    println!("to unit length for consistent downstream processing.");
}
