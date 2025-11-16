//! Performance Benchmarking Suite for MEF
//!
//! Establishes performance baseline and tracks metrics over time.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mef_ledger::MEFLedger;
use mef_spiral::{SpiralConfig, SpiralSnapshot};
use serde_json::json;

fn benchmark_spiral_snapshot_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("spiral_snapshot");

    let temp_dir = tempfile::tempdir().unwrap();
    let config = SpiralConfig::default();
    let spiral = SpiralSnapshot::new(config, temp_dir.path().to_str().unwrap()).unwrap();

    let test_data = json!({
        "type": "benchmark",
        "value": 42,
        "timestamp": "2025-10-15T18:00:00Z"
    });

    group.bench_function("create_snapshot", |b| {
        b.iter(|| {
            spiral
                .create_snapshot(black_box(&test_data), black_box("BENCH_SEED_001"), None)
                .unwrap()
        })
    });

    group.finish();
}

fn benchmark_ledger_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("ledger");

    let temp_dir = tempfile::tempdir().unwrap();
    let mut ledger = MEFLedger::new(temp_dir.path().to_str().unwrap()).unwrap();

    let tic = json!({
        "tic_id": "tic-bench-001",
        "seed": "BENCH_SEED",
        "fixpoint": [0.1, 0.2, 0.3],
        "invariants": {"variance": 0.1},
        "sigma_bar": {"psi": 0.5},
        "window": ["2025-10-15T00:00:00", "2025-10-15T01:00:00"],
        "proof": {"merkle_root": "bench_root"}
    });

    let snapshot = json!({
        "id": "snap-bench-001",
        "coordinates": [0.1, 0.2, 0.3, 0.4, 0.5]
    });

    group.bench_function("append_block", |b| {
        b.iter(|| {
            ledger
                .append_block(black_box(&tic), black_box(&snapshot))
                .unwrap()
        })
    });

    // Add blocks for verification benchmark
    for i in 0..10 {
        let tic = json!({
            "tic_id": format!("tic-{}", i),
            "seed": format!("SEED_{}", i),
            "fixpoint": [0.1 * i as f64, 0.2 * i as f64, 0.3 * i as f64],
            "invariants": {"variance": 0.1},
            "sigma_bar": {"psi": 0.5},
            "window": ["2025-10-15T00:00:00", "2025-10-15T01:00:00"],
            "proof": {"merkle_root": format!("root_{}", i)}
        });

        let snapshot = json!({
            "id": format!("snap-{}", i),
            "coordinates": [0.1 * i as f64, 0.2 * i as f64, 0.3 * i as f64, 0.4 * i as f64, 0.5 * i as f64]
        });

        ledger.append_block(&tic, &snapshot).unwrap();
    }

    group.bench_function("verify_chain_integrity", |b| {
        b.iter(|| ledger.verify_chain_integrity(black_box(5)).unwrap())
    });

    group.finish();
}

fn benchmark_json_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");

    let data = json!({
        "tic_id": "tic-bench-001",
        "seed": "BENCH_SEED",
        "fixpoint": [0.1, 0.2, 0.3],
        "invariants": {"variance": 0.1},
        "sigma_bar": {"psi": 0.5},
        "window": ["2025-10-15T00:00:00", "2025-10-15T01:00:00"],
        "proof": {"merkle_root": "bench_root"}
    });

    group.bench_function("serialize", |b| {
        b.iter(|| serde_json::to_string(black_box(&data)).unwrap())
    });

    let serialized = serde_json::to_string(&data).unwrap();

    group.bench_function("deserialize", |b| {
        b.iter(|| serde_json::from_str::<serde_json::Value>(black_box(&serialized)).unwrap())
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_spiral_snapshot_creation,
    benchmark_ledger_operations,
    benchmark_json_serialization
);
criterion_main!(benches);
