//! Benchmarks for UnifiedCognitiveEngine
//!
//! Measures performance of the complete APOLLYON → MEF pipeline

use apollyon_mef_bridge::unified::{CognitiveInput, GateConfig, UnifiedCognitiveEngine};
use core_5d::{State5D, SystemParameters};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn create_test_input(t_final: f64, tic_id: &str) -> CognitiveInput {
    CognitiveInput {
        initial_state: State5D::new(1.0, 0.5, 0.3, 0.2, 0.1),
        parameters: SystemParameters::default(),
        t_final,
        tic_id: tic_id.to_string(),
        seed: "benchmark_seed".to_string(),
        seed_path: "MEF/bench/0001".to_string(),
    }
}

fn bench_single_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_processing");

    // Benchmark different integration times
    for t_final in [0.1, 0.5, 1.0, 2.0].iter() {
        group.bench_with_input(
            BenchmarkId::new("process", format!("t_{}", t_final)),
            t_final,
            |b, &t_final| {
                let mut engine = UnifiedCognitiveEngine::new();
                let input = create_test_input(t_final, "BENCH-001");

                b.iter(|| {
                    let result = engine.process(black_box(input.clone()));
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");
    group.sample_size(20); // Reduce sample size for batch operations

    // Benchmark different batch sizes
    for batch_size in [1, 5, 10, 20].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            batch_size,
            |b, &batch_size| {
                let mut engine = UnifiedCognitiveEngine::new();
                let inputs: Vec<_> = (0..batch_size)
                    .map(|i| create_test_input(0.5, &format!("BENCH-{:03}", i)))
                    .collect();

                b.iter(|| {
                    let result = engine.process_batch(black_box(inputs.clone()));
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn bench_gate_configurations(c: &mut Criterion) {
    let mut group = c.benchmark_group("gate_configurations");

    let configs = vec![
        ("default", GateConfig::default()),
        ("strict", GateConfig::strict()),
        ("relaxed", GateConfig::relaxed()),
        (
            "custom",
            GateConfig::new(0.15, 0.6, 0.85),
        ),
    ];

    for (name, config) in configs {
        group.bench_with_input(BenchmarkId::new("config", name), &config, |b, config| {
            let mut engine = UnifiedCognitiveEngine::new_with_config(config.clone());
            let input = create_test_input(0.5, "BENCH-CONFIG");

            b.iter(|| {
                let result = engine.process(black_box(input.clone()));
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_component_stages(c: &mut Criterion) {
    let mut group = c.benchmark_group("component_stages");

    // Benchmark individual pipeline stages
    let mut engine = UnifiedCognitiveEngine::new();
    let input = create_test_input(0.5, "BENCH-STAGES");

    // Full pipeline
    group.bench_function("full_pipeline", |b| {
        b.iter(|| {
            let result = engine.process(black_box(input.clone()));
            black_box(result)
        });
    });

    // Note: Individual stages are private, so we can only benchmark the full pipeline
    // In a real scenario, you might want to expose internal methods for benchmarking

    group.finish();
}

fn bench_trajectory_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("trajectory_sizes");

    // Benchmark different trajectory lengths (controlled by t_final and dt)
    // dt is fixed at 0.01 in the engine, so trajectory length ≈ t_final / 0.01
    for (name, t_final) in [
        ("small", 0.1),   // ~10 states
        ("medium", 0.5),  // ~50 states
        ("large", 1.0),   // ~100 states
        ("xlarge", 2.0),  // ~200 states
    ]
    .iter()
    {
        group.bench_with_input(
            BenchmarkId::new("trajectory", name),
            t_final,
            |b, &t_final| {
                let mut engine = UnifiedCognitiveEngine::new();
                let input = create_test_input(t_final, "BENCH-TRAJ");

                b.iter(|| {
                    let result = engine.process(black_box(input.clone()));
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_processing,
    bench_batch_processing,
    bench_gate_configurations,
    bench_component_stages,
    bench_trajectory_sizes
);
criterion_main!(benches);
