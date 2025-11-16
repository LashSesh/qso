//! Example demonstrating metrics collection and export

use unified_5d_cube::{InterlockConfig, InterlockAdapter, tick_5d_cube, MetricsCollector, MetricsFormat};
use core_5d::State5D;
use std::path::Path;

fn main() {
    println!("=== Unified 5D Cube - Metrics Example ===\n");
    
    // Create configuration
    let config = InterlockConfig::default();
    let mut adapter = InterlockAdapter::new(config);
    
    // Create metrics collector
    let mut collector = MetricsCollector::new(100);
    
    // Initial state
    let mut state = State5D::from_array([1.0, 0.5, 0.3, 0.7, 0.4]);
    let mut prev_state: Option<State5D> = None;
    
    println!("Running 10 ticks...\n");
    
    // Run multiple ticks
    for i in 0..10 {
        let result = tick_5d_cube(
            &mut adapter,
            &state,
            prev_state.as_ref(),
            i as f64 * 0.1,
            i,
        );
        
        // Collect metrics
        collector.add(result.metrics.clone());
        
        println!("Tick {}: ΔF={:.4}, W2={:.4}, S_mand={:.4}",
            i, result.metrics.delta_f, result.metrics.w2_step, result.metrics.s_mand);
        
        // Update for next iteration
        prev_state = Some(state);
        let condensed_arr = result.state_condensed.as_array();
        state = State5D::from_array(condensed_arr);
    }
    
    println!("\n--- Metrics Summary ---");
    
    // Check trends
    if collector.is_improving_trend() {
        println!("✓ System showing improving trend");
    } else {
        println!("⚠ System not consistently improving");
    }
    
    if collector.delta_f_decreasing(3) {
        println!("✓ ΔF decreasing over last 3 ticks");
    }
    
    if collector.w2_step_decreasing(3) {
        println!("✓ W2_step decreasing over last 3 ticks");
    }
    
    // Compute averages
    if let Some(avg) = collector.average() {
        println!("\nAverage metrics:");
        println!("  BI: {:.4}", avg.bi);
        println!("  ΔF: {:.4}", avg.delta_f);
        println!("  W2_step: {:.4}", avg.w2_step);
        println!("  λ_gap: {:.4}", avg.lambda_gap);
        println!("  S_mand: {:.4}", avg.s_mand);
        println!("  Duty/PoR: {:.4}", avg.duty_por);
        println!("  Avg time: {:.2}ms", avg.elapsed_ms);
    }
    
    // Export to files
    println!("\nExporting metrics...");
    
    if let Err(e) = collector.export(Path::new("/tmp/metrics.csv"), MetricsFormat::CSV) {
        println!("CSV export failed: {}", e);
    } else {
        println!("✓ Exported to /tmp/metrics.csv");
    }
    
    if let Err(e) = collector.export(Path::new("/tmp/metrics.json"), MetricsFormat::JSON) {
        println!("JSON export failed: {}", e);
    } else {
        println!("✓ Exported to /tmp/metrics.json");
    }
    
    println!("\n=== Example Complete ===");
}
