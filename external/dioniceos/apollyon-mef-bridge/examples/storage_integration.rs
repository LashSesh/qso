//! Example: Storage Integration
//!
//! Demonstrates how to use different storage backends
//! to persist cognitive processing results.

use apollyon_mef_bridge::storage::{LedgerStorage, MemoryStorage, StorageBackend};
use apollyon_mef_bridge::unified::{CognitiveInput, UnifiedCognitiveEngine};
use core_5d::{State5D, SystemParameters};
use std::path::Path;
use tempfile::TempDir;

#[tokio::main]
async fn main() {
    println!("=== Storage Integration Example ===\n");

    // Example 1: Memory Storage
    println!("1. Memory Storage (Development/Testing)");
    example_memory_storage().await;
    println!();

    // Example 2: Ledger Storage
    println!("2. Ledger Storage (Production)");
    example_ledger_storage().await;
    println!();

    // Example 3: Storage Statistics
    println!("3. Storage Statistics");
    example_storage_stats().await;
    println!();

    // Example 4: Batch Processing with Storage
    println!("4. Batch Processing with Storage");
    example_batch_with_storage().await;

    println!("\n=== Complete! ===");
}

async fn example_memory_storage() {
    let mut storage = MemoryStorage::new();

    println!("   Creating engine and processing input...");

    let mut engine = UnifiedCognitiveEngine::new();
    let input = CognitiveInput {
        initial_state: State5D::new(1.0, 0.5, 0.3, 0.2, 0.1),
        parameters: SystemParameters::default(),
        t_final: 0.5,
        tic_id: "TIC-MEM-001".to_string(),
        seed: "memory_seed".to_string(),
        seed_path: "MEF/examples/memory/0001".to_string(),
    };

    match engine.process(input) {
        Ok(output) => {
            println!("   ✓ Processing completed");
            println!("   - Gate decision: {:?}", output.gate_decision);

            // Store the output
            match storage.store(&output).await {
                Ok(id) => {
                    println!("   ✓ Stored in memory with ID: {}", id);

                    // Retrieve it back
                    match storage.retrieve(&id).await {
                        Ok(retrieved) => {
                            println!("   ✓ Retrieved from memory");
                            println!(
                                "   - Trajectory length matches: {}",
                                retrieved.trajectory.len() == output.trajectory.len()
                            );
                        }
                        Err(e) => println!("   ✗ Retrieval error: {}", e),
                    }
                }
                Err(e) => println!("   ✗ Storage error: {}", e),
            }
        }
        Err(e) => println!("   ✗ Processing error: {}", e),
    }

    // Check storage stats
    if let Ok(stats) = storage.stats().await {
        println!("   - Total items in memory: {}", stats.total_items);
    }
}

async fn example_ledger_storage() {
    // Create temporary directory for ledger
    let temp_dir = TempDir::new().unwrap();
    let ledger_path = temp_dir.path();

    println!("   Ledger path: {:?}", ledger_path);

    let mut storage = LedgerStorage::new(ledger_path).unwrap();

    println!("   Creating engine and processing input...");

    let mut engine = UnifiedCognitiveEngine::new();
    let input = CognitiveInput {
        initial_state: State5D::new(1.0, 0.5, 0.3, 0.2, 0.1),
        parameters: SystemParameters::default(),
        t_final: 0.5,
        tic_id: "TIC-LEDGER-001".to_string(),
        seed: "ledger_seed".to_string(),
        seed_path: "MEF/examples/ledger/0001".to_string(),
    };

    match engine.process(input) {
        Ok(output) => {
            println!("   ✓ Processing completed");
            println!("   - Gate decision: {:?}", output.gate_decision);

            // Only FIRE decisions are stored in ledger
            if matches!(output.gate_decision, mef_schemas::GateDecision::FIRE) {
                match storage.store(&output).await {
                    Ok(id) => {
                        println!("   ✓ Stored in ledger with ID: {}", id);

                        // Verify file exists
                        let block_file = ledger_path.join(format!("{}.json", id));
                        if block_file.exists() {
                            println!("   ✓ Ledger file created: {}", block_file.display());

                            // Read and display a snippet
                            if let Ok(content) = std::fs::read_to_string(&block_file) {
                                let json: serde_json::Value =
                                    serde_json::from_str(&content).unwrap();
                                println!("   - Ledger entry preview:");
                                println!(
                                    "     TIC ID: {}",
                                    json["tic_id"].as_str().unwrap_or("N/A")
                                );
                                println!(
                                    "     Route ID: {}",
                                    json["route_id"].as_str().unwrap_or("N/A")
                                );
                                println!(
                                    "     Gate: {}",
                                    json["gate_decision"].as_str().unwrap_or("N/A")
                                );
                            }
                        }
                    }
                    Err(e) => println!("   ✗ Storage error: {}", e),
                }
            } else {
                println!("   ⓘ Gate decision is HOLD, not storing to ledger");
            }
        }
        Err(e) => println!("   ✗ Processing error: {}", e),
    }
}

async fn example_storage_stats() {
    let mut storage = MemoryStorage::new();

    println!("   Processing multiple inputs...");

    let mut engine = UnifiedCognitiveEngine::new();

    for i in 1..=5 {
        let input = CognitiveInput {
            initial_state: State5D::new(1.0, 0.5, 0.3, 0.2, 0.1),
            parameters: SystemParameters::default(),
            t_final: 0.3,
            tic_id: format!("TIC-STATS-{:03}", i),
            seed: format!("stats_seed_{}", i),
            seed_path: format!("MEF/examples/stats/{:04}", i),
        };

        if let Ok(output) = engine.process(input) {
            let _ = storage.store(&output).await;
        }
    }

    // Get statistics
    match storage.stats().await {
        Ok(stats) => {
            println!("   ✓ Storage Statistics:");
            println!("     - Backend type: {}", stats.backend_type);
            println!("     - Total items: {}", stats.total_items);
            println!("     - Total size: {} bytes", stats.total_size_bytes);
            println!("     - Successful writes: {}", stats.successful_writes);
            println!("     - Failed writes: {}", stats.failed_writes);
        }
        Err(e) => println!("   ✗ Stats error: {}", e),
    }

    // Health check
    match storage.health_check().await {
        Ok(healthy) => println!("   - Storage health: {}", if healthy { "✓" } else { "✗" }),
        Err(e) => println!("   ✗ Health check error: {}", e),
    }
}

async fn example_batch_with_storage() {
    let temp_dir = TempDir::new().unwrap();
    let mut storage = LedgerStorage::new(temp_dir.path()).unwrap();

    println!("   Processing batch...");

    let mut engine = UnifiedCognitiveEngine::new();

    let inputs: Vec<_> = (1..=3)
        .map(|i| CognitiveInput {
            initial_state: State5D::new(1.0, 0.5, 0.3, 0.2, 0.1),
            parameters: SystemParameters::default(),
            t_final: 0.5,
            tic_id: format!("TIC-BATCH-{:03}", i),
            seed: format!("batch_seed_{}", i),
            seed_path: format!("MEF/examples/batch/{:04}", i),
        })
        .collect();

    let batch_result = engine.process_batch(inputs);

    println!("   ✓ Batch completed:");
    println!("     - Total: {}", batch_result.total_count());
    println!("     - Successes: {}", batch_result.success_count());
    println!("     - Failures: {}", batch_result.failure_count());
    println!("     - Success rate: {:.1}%", batch_result.success_rate());
    println!("     - Total time: {:.3}s", batch_result.total_time);
    println!("     - Avg time per item: {:.3}s", batch_result.avg_time);

    // Store successful outputs
    let mut stored_count = 0;
    for output in &batch_result.successes {
        if matches!(output.gate_decision, mef_schemas::GateDecision::FIRE) {
            if storage.store(output).await.is_ok() {
                stored_count += 1;
            }
        }
    }

    println!("   - Stored to ledger: {} outputs", stored_count);

    // Final stats
    if let Ok(stats) = storage.stats().await {
        println!("   - Ledger total items: {}", stats.total_items);
    }
}
