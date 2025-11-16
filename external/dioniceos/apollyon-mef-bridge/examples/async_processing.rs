//! Example: Async Processing
//!
//! Demonstrates asynchronous processing with AsyncUnifiedCognitiveEngine,
//! including parallel batch processing and performance comparisons.

use apollyon_mef_bridge::unified::{AsyncUnifiedCognitiveEngine, CognitiveInput};
use core_5d::{State5D, SystemParameters};
use std::time::Instant;

#[tokio::main]
async fn main() {
    println!("=== Async Processing Example ===\n");

    // Example 1: Single async processing
    println!("1. Single Async Processing");
    example_single_async().await;
    println!();

    // Example 2: Sequential async batch
    println!("2. Sequential Async Batch");
    example_sequential_batch().await;
    println!();

    // Example 3: Parallel batch processing
    println!("3. Parallel Batch Processing");
    example_parallel_batch().await;
    println!();

    // Example 4: Performance comparison
    println!("4. Performance Comparison");
    example_performance_comparison().await;

    println!("\n=== Complete! ===");
}

async fn example_single_async() {
    let mut engine = AsyncUnifiedCognitiveEngine::new();

    println!("   Processing single input asynchronously...");

    let input = CognitiveInput {
        initial_state: State5D::new(1.0, 0.5, 0.3, 0.2, 0.1),
        parameters: SystemParameters::default(),
        t_final: 0.5,
        tic_id: "TIC-ASYNC-001".to_string(),
        seed: "async_seed".to_string(),
        seed_path: "MEF/examples/async/0001".to_string(),
    };

    let start = Instant::now();

    match engine.process_async(input).await {
        Ok(output) => {
            let duration = start.elapsed();
            println!("   ✓ Processing completed in {:?}", duration);
            println!("   - Trajectory length: {}", output.trajectory.len());
            println!("   - Gate decision: {:?}", output.gate_decision);
            println!("   - PoR valid: {}", output.proof.por_valid);
        }
        Err(e) => println!("   ✗ Error: {}", e),
    }
}

async fn example_sequential_batch() {
    let mut engine = AsyncUnifiedCognitiveEngine::new();

    println!("   Creating 5 inputs...");

    let inputs: Vec<_> = (1..=5)
        .map(|i| CognitiveInput {
            initial_state: State5D::new(1.0, 0.5, 0.3, 0.2, 0.1),
            parameters: SystemParameters::default(),
            t_final: 0.3,
            tic_id: format!("TIC-SEQ-{:03}", i),
            seed: format!("seq_seed_{}", i),
            seed_path: format!("MEF/examples/seq/{:04}", i),
        })
        .collect();

    println!("   Processing sequentially (async)...");

    let start = Instant::now();
    let batch_result = engine.process_batch_async(inputs).await;
    let duration = start.elapsed();

    println!("   ✓ Batch completed in {:?}", duration);
    println!("   - Total: {}", batch_result.total_count());
    println!("   - Successes: {}", batch_result.success_count());
    println!("   - Failures: {}", batch_result.failure_count());
    println!("   - Success rate: {:.1}%", batch_result.success_rate());
    println!("   - Avg time per item: {:.3}s", batch_result.avg_time);
}

async fn example_parallel_batch() {
    let engine = AsyncUnifiedCognitiveEngine::new();

    println!("   Creating 10 inputs...");

    let inputs: Vec<_> = (1..=10)
        .map(|i| CognitiveInput {
            initial_state: State5D::new(1.0, 0.5, 0.3, 0.2, 0.1),
            parameters: SystemParameters::default(),
            t_final: 0.3,
            tic_id: format!("TIC-PAR-{:03}", i),
            seed: format!("par_seed_{}", i),
            seed_path: format!("MEF/examples/par/{:04}", i),
        })
        .collect();

    // Test different parallelism levels
    for parallelism in [2, 4, 8] {
        println!("   Processing with parallelism = {}...", parallelism);

        let start = Instant::now();
        let batch_result = engine
            .process_batch_parallel(inputs.clone(), Some(parallelism))
            .await;
        let duration = start.elapsed();

        println!("     ✓ Completed in {:?}", duration);
        println!("     - Successes: {}", batch_result.success_count());
        println!("     - Throughput: {:.1} items/sec",
                 batch_result.total_count() as f64 / duration.as_secs_f64());
    }
}

async fn example_performance_comparison() {
    println!("   Comparing sequential vs parallel processing...");
    println!("   (10 inputs, medium trajectories)");
    println!();

    let inputs: Vec<_> = (1..=10)
        .map(|i| CognitiveInput {
            initial_state: State5D::new(1.0, 0.5, 0.3, 0.2, 0.1),
            parameters: SystemParameters::default(),
            t_final: 0.5,
            tic_id: format!("TIC-COMP-{:03}", i),
            seed: format!("comp_seed_{}", i),
            seed_path: format!("MEF/examples/comp/{:04}", i),
        })
        .collect();

    // Sequential
    println!("   Sequential processing:");
    let mut seq_engine = AsyncUnifiedCognitiveEngine::new();
    let start = Instant::now();
    let seq_result = seq_engine.process_batch_async(inputs.clone()).await;
    let seq_duration = start.elapsed();

    println!("     - Time: {:?}", seq_duration);
    println!("     - Success rate: {:.1}%", seq_result.success_rate());
    println!("     - Throughput: {:.1} items/sec",
             seq_result.total_count() as f64 / seq_duration.as_secs_f64());
    println!();

    // Parallel
    println!("   Parallel processing (4 workers):");
    let par_engine = AsyncUnifiedCognitiveEngine::new();
    let start = Instant::now();
    let par_result = par_engine
        .process_batch_parallel(inputs, Some(4))
        .await;
    let par_duration = start.elapsed();

    println!("     - Time: {:?}", par_duration);
    println!("     - Success rate: {:.1}%", par_result.success_rate());
    println!("     - Throughput: {:.1} items/sec",
             par_result.total_count() as f64 / par_duration.as_secs_f64());
    println!();

    // Speedup
    let speedup = seq_duration.as_secs_f64() / par_duration.as_secs_f64();
    println!("   Speedup: {:.2}x", speedup);

    if speedup > 1.5 {
        println!("   ✓ Significant performance improvement with parallelism");
    } else if speedup > 1.0 {
        println!("   ⓘ Modest performance improvement with parallelism");
    } else {
        println!("   ⓘ Parallelism overhead may be affecting performance");
        println!("   (Try larger batches or longer processing times)");
    }
}
