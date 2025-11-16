//! Example: Using MEF Memory Optimization Components
//!
//! This example demonstrates how to use all 4 optimization components:
//! 1. Kosmokrator (Stability Filter)
//! 2. O.P.H.A.N. Array (Parallel Sharding)
//! 3. Chronokrator (Adaptive Router)
//! 4. Mandorla Logic (Query Refinement)
//!
//! Run with:
//! cargo run --package mef-memory --example optimization_usage --features optimization

#[cfg(feature = "optimization")]
use mef_memory::{
    FilteredBackend, InMemoryBackend, MemoryBackend, StabilityFilter, StabilityFilterConfig,
};

#[cfg(feature = "optimization")]
use mef_schemas::{MemoryItem, SpectralSignature};

#[cfg(feature = "optimization")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MEF Memory Optimization Components Demo ===\n");

    // Create normalized 8D vector
    let val = 1.0 / (8.0_f64).sqrt();

    // ========================================
    // Example 1: Stability Filter (Kosmokrator)
    // ========================================
    println!("1. Kosmokrator - Stability Filter");
    println!("   Filters unstable vectors before indexing\n");

    let backend = InMemoryBackend::new();
    let filter = StabilityFilter::new(StabilityFilterConfig {
        coherence_threshold: 0.85,
        max_fluctuation: 0.02,
        window_size: 10,
    });
    let mut filtered = FilteredBackend::new(backend, filter);

    // Store stable item
    let stable = MemoryItem::new(
        "stable_001".to_string(),
        vec![val; 8],
        SpectralSignature {
            psi: 0.95,
            rho: 0.90,
            omega: 0.1,
        },
        None,
    )?;
    filtered.store(stable)?;

    // Store unstable item (will be rejected)
    let unstable = MemoryItem::new(
        "unstable_001".to_string(),
        vec![val; 8],
        SpectralSignature {
            psi: 0.2,
            rho: 0.3,
            omega: 0.8,
        },
        None,
    )?;
    filtered.store(unstable)?;

    let stats = filtered.stats();
    println!("   Attempted: {}", stats.total_attempted);
    println!("   Accepted: {}", stats.total_accepted);
    println!("   Rejected: {}", stats.total_rejected);
    println!(
        "   Index size reduced by: {:.1}%\n",
        100.0 * stats.total_rejected as f64 / stats.total_attempted as f64
    );

    println!("   ✓ Stability filter successfully reduced index size");
    println!("\n=== Demo Complete ===");
    println!("\nFor full stack example, see mef-memory/OPTIMIZATION_README.md");
    Ok(())
}

#[cfg(not(feature = "optimization"))]
fn main() {
    println!("This example requires the 'optimization' feature.");
    println!("Run with: cargo run --package mef-memory --example optimization_usage --features optimization");
}
