//! Example demonstrating MEF Ledger writes
//!
//! Run with: cargo run --example ledger_writes

use unified_5d_cube::{InterlockAdapter, InterlockConfig, tick_5d_cube};
use core_5d::State5D;
use std::path::PathBuf;
use tempfile::TempDir;

fn main() {
    // Create a temporary directory for ledger
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let ledger_path = temp_dir.path().join("test_ledger");
    
    println!("=== MEF Ledger Writes Example ===\n");
    println!("Ledger path: {:?}\n", ledger_path);
    
    // Enable ledger writes (but keep shadow mode for safety)
    let mut config = InterlockConfig::default();
    config.enable_ledger_writes = true;
    config.ledger_path = Some(ledger_path.clone());
    config.shadow_mode = false; // Enable actual writes
    config.enable_logging = true;
    
    // Lower thresholds to get some FIRE decisions
    config.gate_phi_threshold = 0.3;
    config.gate_delta_pi_max = 0.5;
    
    println!("Configuration:");
    println!("  - Ledger Writes: {}", config.enable_ledger_writes);
    println!("  - Shadow Mode: {}", config.shadow_mode);
    println!("  - Gate φ threshold: {}", config.gate_phi_threshold);
    println!("  - Seed: {}\n", config.seed);
    
    let mut adapter = InterlockAdapter::new(config);
    
    // Run a sequence of ticks
    let mut state = State5D::from_array([1.0, 0.8, 0.6, 0.9, 0.7]);
    let mut fire_count = 0;
    let mut write_count = 0;
    
    println!("Running 10 ticks with ledger writes...\n");
    
    for i in 0..10 {
        let prev_state = if i > 0 { Some(&state) } else { None };
        
        // Converging states to trigger FIRE
        let next_state = State5D::from_array([
            state.to_array()[0] * 0.95,
            state.to_array()[1] * 0.96,
            state.to_array()[2] * 0.97,
            state.to_array()[3] * 0.98,
            state.to_array()[4] * 0.99,
        ]);
        
        let result = tick_5d_cube(&mut adapter, &next_state, prev_state, i as f64 * 0.1, i);
        
        println!("Tick {}: ", i);
        println!("  Gate: {:?}", result.gate_decision);
        println!("  ΔF: {:.6}", result.metrics.delta_f);
        println!("  φ: {:.6}", result.proof.phi);
        println!("  Ledger written: {}", result.ledger_written);
        
        if result.commit.is_some() {
            fire_count += 1;
            if result.ledger_written {
                write_count += 1;
                println!("  ✓ Commit hash: {}", result.commit.as_ref().unwrap().commit_hash);
            }
        }
        println!();
        
        state = next_state;
    }
    
    println!("=== Ledger Writes Complete ===");
    println!("\nSummary:");
    println!("  - FIRE decisions: {}", fire_count);
    println!("  - Ledger writes: {}", write_count);
    
    // Check if ledger index exists
    let index_path = ledger_path.join("ledger_index.json");
    if index_path.exists() {
        println!("  - Ledger index created: ✓");
        
        // Try to read the index
        if let Ok(contents) = std::fs::read_to_string(&index_path) {
            if let Ok(index) = serde_json::from_str::<serde_json::Value>(&contents) {
                if let Some(blocks) = index.get("blocks").and_then(|b| b.as_array()) {
                    println!("  - Blocks in ledger: {}", blocks.len());
                }
            }
        }
    }
    
    println!("\nNote: Ledger writes only occur when gate decision is FIRE");
    println!("and shadow mode is disabled. Commits are hash-chained for integrity.");
}
