//! Comprehensive example demonstrating all Future Extensions integrated
//!
//! Run with: cargo run --example all_extensions

use unified_5d_cube::{InterlockAdapter, InterlockConfig, tick_5d_cube};
use core_5d::State5D;
use apollyon_mef_bridge::trichter::Policy;
use std::path::PathBuf;
use tempfile::TempDir;

fn main() {
    // Create a temporary directory for storage
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let storage_path = temp_dir.path().join("5d_cube_storage");
    
    println!("=== All Future Extensions Integrated ===\n");
    println!("Storage path: {:?}\n", storage_path);
    
    // Enable ALL extensions
    let mut config = InterlockConfig::default();
    config.enable_full_hdag = true;
    config.enable_funnel_ops = true;
    config.enable_ledger_writes = true;
    config.enable_8d_vectors = true;
    config.enable_metatron_routing = true;
    config.funnel_policy = Policy::Homeostasis;
    config.ledger_path = Some(storage_path.clone());
    config.shadow_mode = false; // Enable actual operations
    config.enable_logging = true;
    
    // Lower thresholds for demonstration
    config.gate_phi_threshold = 0.4;
    config.gate_delta_pi_max = 0.3;
    
    println!("Configuration:");
    println!("  ✓ Full HDAG relaxation: {}", config.enable_full_hdag);
    println!("  ✓ Funnel operations: {} (Policy: {:?})", config.enable_funnel_ops, config.funnel_policy);
    println!("  ✓ MEF Ledger writes: {}", config.enable_ledger_writes);
    println!("  ✓ 8D vector pipeline: {}", config.enable_8d_vectors);
    println!("  ✓ Metatron routing: {}", config.enable_metatron_routing);
    println!("  - Shadow mode: {}", config.shadow_mode);
    println!("  - Seed: {}\n", config.seed);
    
    let mut adapter = InterlockAdapter::new(config);
    
    // Run a realistic convergence sequence
    let mut state = State5D::from_array([2.0, 1.5, 1.0, 0.9, 0.8]);
    let mut fire_count = 0;
    let mut ledger_write_count = 0;
    
    println!("Running 12 ticks with all extensions enabled...\n");
    println!("{:-<80}", "");
    
    for i in 0..12 {
        let prev_state = if i > 0 { Some(&state) } else { None };
        
        // Converging trajectory with noise
        let decay = 0.94 + (i as f64 * 0.002);
        let noise = ((i as f64 * 1.7).sin() * 0.02).abs();
        
        let next_state = State5D::from_array([
            state.to_array()[0] * (decay + noise),
            state.to_array()[1] * (decay + noise * 0.8),
            state.to_array()[2] * (decay + noise * 0.6),
            state.to_array()[3] * (decay + noise * 0.4),
            state.to_array()[4] * (decay + noise * 0.2),
        ]);
        
        let result = tick_5d_cube(&mut adapter, &next_state, prev_state, i as f64 * 0.1, i);
        
        println!("\nTick {:2}: ", i);
        println!("  State:     norm={:.4}  [x={:.3}, y={:.3}, z={:.3}, ψ={:.3}, ω={:.3}]",
                 next_state.norm(),
                 next_state.to_array()[0],
                 next_state.to_array()[1],
                 next_state.to_array()[2],
                 next_state.to_array()[3],
                 next_state.to_array()[4]);
        
        println!("  Guidance:  magnitude={:.6}", 
                 result.guidance.iter().map(|x| x*x).sum::<f64>().sqrt());
        
        if let Some(ref route) = result.selected_route {
            println!("  Route:     {}", route.join(" → "));
        }
        
        if let Some(ref vec8) = result.vector_8d {
            let norm: f64 = vec8.iter().map(|x| x * x).sum::<f64>().sqrt();
            println!("  8D Vector: norm={:.6}  spiral=[{:.3}..{:.3}]  spectral=[{:.3}..{:.3}]",
                     norm, vec8[0], vec8[4], vec8[5], vec8[7]);
        }
        
        println!("  Metrics:   ΔF={:.6}  W2={:.6}  S_mand={:.4}  BI={:.4}",
                 result.metrics.delta_f,
                 result.metrics.w2_step,
                 result.metrics.s_mand,
                 result.metrics.bi);
        
        println!("  Gate:      {:?}  (φ={:.4}, ΔPI={:.4}, PoR={})",
                 result.gate_decision,
                 result.proof.phi,
                 result.proof.delta_pi,
                 if result.proof.por_valid { "✓" } else { "✗" });
        
        if result.commit.is_some() {
            fire_count += 1;
            print!("  Commit:    FIRE");
            if result.ledger_written {
                ledger_write_count += 1;
                println!(" → Ledger ✓ [hash={}]", 
                         &result.commit.as_ref().unwrap().commit_hash[..16]);
            } else {
                println!(" (not written)");
            }
        }
        
        state = next_state;
    }
    
    println!("\n{:-<80}", "");
    println!("\n=== Integration Complete ===");
    println!("\nSummary:");
    println!("  - Total ticks: 12");
    println!("  - FIRE decisions: {}", fire_count);
    println!("  - Ledger writes: {}", ledger_write_count);
    
    // Check ledger
    let ledger_index = storage_path.join("ledger_index.json");
    if ledger_index.exists() {
        if let Ok(contents) = std::fs::read_to_string(&ledger_index) {
            if let Ok(index) = serde_json::from_str::<serde_json::Value>(&contents) {
                if let Some(blocks) = index.get("blocks").and_then(|b| b.as_array()) {
                    println!("  - Blocks in ledger: {}", blocks.len());
                }
            }
        }
    }
    
    println!("\nIntegration Features:");
    println!("  1. HDAG Relaxation:   Hyperbion fields + phase gradients for guidance");
    println!("  2. Funnel Operations: Hebbian learning + split/merge/prune on graph");
    println!("  3. Ledger Writes:     Hash-chained commits to MEF Ledger on FIRE");
    println!("  4. 8D Vectors:        Normalized vectors from 5D + spectral signature");
    println!("  5. Metatron Routing:  S7 permutations over 13-node topology");
    
    println!("\nAll future extensions are now fully integrated and functional!");
}
