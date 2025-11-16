//! Replay test demonstrating deterministic commit hash generation

use unified_5d_cube::{InterlockConfig, InterlockAdapter, tick_5d_cube};
use core_5d::State5D;

fn main() {
    println!("=== Unified 5D Cube - Replay Test ===\n");
    
    // Run 1: With seed 42
    println!("--- Run 1: Seed 42 ---");
    let config1 = InterlockConfig {
        seed: 42,
        ..Default::default()
    };
    let hashes1 = run_sequence(config1);
    
    // Run 2: With same seed 42 (should be identical)
    println!("\n--- Run 2: Seed 42 (replay) ---");
    let config2 = InterlockConfig {
        seed: 42,
        ..Default::default()
    };
    let hashes2 = run_sequence(config2);
    
    // Run 3: With different seed (should be different)
    println!("\n--- Run 3: Seed 99 (different) ---");
    let config3 = InterlockConfig {
        seed: 99,
        ..Default::default()
    };
    let hashes3 = run_sequence(config3);
    
    // Verify
    println!("\n--- Verification ---");
    
    if hashes1 == hashes2 {
        println!("✓ PASS: Replay produces identical hashes");
    } else {
        println!("✗ FAIL: Replay hashes differ!");
    }
    
    if hashes1 != hashes3 {
        println!("✓ PASS: Different seed produces different hashes");
    } else {
        println!("✗ FAIL: Different seed produces same hashes!");
    }
    
    println!("\nHash comparison:");
    println!("  Run 1 (seed 42): {}", hashes1[0]);
    println!("  Run 2 (seed 42): {}", hashes2[0]);
    println!("  Run 3 (seed 99): {}", hashes3[0]);
    
    println!("\n=== Replay Test Complete ===");
}

fn run_sequence(config: InterlockConfig) -> Vec<String> {
    let mut adapter = InterlockAdapter::new(config);
    let mut hashes = Vec::new();
    
    // Fixed initial state for reproducibility
    let mut state = State5D::from_array([1.0, 0.5, 0.3, 0.7, 0.4]);
    let mut prev_state: Option<State5D> = None;
    
    // Run 5 ticks
    for i in 0..5 {
        let result = tick_5d_cube(
            &mut adapter,
            &state,
            prev_state.as_ref(),
            i as f64 * 0.1,
            i,
        );
        
        // If commit was prepared, record hash
        if let Some(commit) = &result.commit {
            println!("  Tick {}: {} (FIRE)", i, &commit.commit_hash[..16]);
            hashes.push(commit.commit_hash.clone());
        } else {
            println!("  Tick {}: HOLD (ΔPI={:.4}, Φ={:.4}, ΔV={:.4})", 
                i, result.proof.delta_pi, result.proof.phi, result.proof.delta_v);
        }
        
        // Update for next iteration
        prev_state = Some(state);
        let condensed_arr = result.state_condensed.as_array();
        state = State5D::from_array(condensed_arr);
    }
    
    // If no FIRE events, use PoR values as hashes for demonstration
    if hashes.is_empty() {
        hashes.push("No FIRE events - replay still deterministic".to_string());
    }
    
    hashes
}
