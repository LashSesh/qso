//! Criterion benchmarks for MEF optimization components

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mef_schemas::{MemoryItem, SpectralSignature};

#[cfg(feature = "optimization")]
use mef_memory::{
    AdaptiveRouter, FilteredBackend, InMemoryBackend, MandorlaBackend, MandorlaConfig,
    MandorlaRefiner, MemoryBackend, OphanBackend, RouterConfig, StabilityFilter,
    StabilityFilterConfig,
};

#[cfg(not(feature = "optimization"))]
use mef_memory::{InMemoryBackend, MemoryBackend};

fn create_test_item(id: &str, val: f64) -> MemoryItem {
    MemoryItem::new(
        id.to_string(),
        vec![val; 8],
        SpectralSignature {
            psi: 0.9,
            rho: 0.9,
            omega: 0.1,
        },
        None,
    )
    .unwrap()
}

fn bench_baseline(c: &mut Criterion) {
    let val = 1.0 / (8.0_f64).sqrt();
    
    c.bench_function("baseline_inmemory_store", |b| {
        let mut backend = InMemoryBackend::new();
        let mut counter = 0;
        
        b.iter(|| {
            let item = create_test_item(&format!("item_{}", counter), val);
            counter += 1;
            backend.store(black_box(item)).unwrap()
        });
    });

    c.bench_function("baseline_inmemory_search", |b| {
        let mut backend = InMemoryBackend::new();
        
        // Populate backend
        for i in 0..1000 {
            let item = create_test_item(&format!("item_{}", i), val);
            backend.store(item).unwrap();
        }
        
        let query = vec![val; 8];
        
        b.iter(|| {
            backend.search(black_box(&query), black_box(10)).unwrap()
        });
    });
}

#[cfg(feature = "optimization")]
fn bench_stability_filter(c: &mut Criterion) {
    let val = 1.0 / (8.0_f64).sqrt();
    
    c.bench_function("stability_filter_store", |b| {
        let inner = InMemoryBackend::new();
        let filter = StabilityFilter::new(StabilityFilterConfig::default());
        let mut backend = FilteredBackend::new(inner, filter);
        let mut counter = 0;
        
        b.iter(|| {
            let item = create_test_item(&format!("item_{}", counter), val);
            counter += 1;
            backend.store(black_box(item)).unwrap()
        });
    });
}

#[cfg(feature = "optimization")]
fn bench_ophan_sharding(c: &mut Criterion) {
    let val = 1.0 / (8.0_f64).sqrt();
    
    c.bench_function("ophan_sharding_store", |b| {
        let inner = InMemoryBackend::new();
        let mut backend = OphanBackend::new(inner);
        let mut counter = 0;
        
        b.iter(|| {
            let item = create_test_item(&format!("item_{}", counter), val);
            counter += 1;
            backend.store(black_box(item)).unwrap()
        });
    });

    c.bench_function("ophan_sharding_search", |b| {
        let inner = InMemoryBackend::new();
        let mut backend = OphanBackend::new(inner);
        
        // Populate backend
        for i in 0..1000 {
            let item = create_test_item(&format!("item_{}", i), val);
            backend.store(item).unwrap();
        }
        
        let query = vec![val; 8];
        
        b.iter(|| {
            backend.search(black_box(&query), black_box(10)).unwrap()
        });
    });
}

#[cfg(feature = "optimization")]
fn bench_adaptive_router(c: &mut Criterion) {
    let val = 1.0 / (8.0_f64).sqrt();
    
    let mut group = c.benchmark_group("adaptive_router_search");
    
    for k in [5, 10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(k), k, |b, &k| {
            let inner = InMemoryBackend::new();
            let mut backend = AdaptiveRouter::new(inner, RouterConfig::default());
            
            // Populate backend
            for i in 0..1000 {
                let item = create_test_item(&format!("item_{}", i), val);
                backend.store(item).unwrap();
            }
            
            let query = vec![val; 8];
            
            b.iter(|| {
                backend.search(black_box(&query), black_box(k)).unwrap()
            });
        });
    }
    
    group.finish();
}

#[cfg(feature = "optimization")]
fn bench_mandorla_refiner(c: &mut Criterion) {
    let val = 1.0 / (8.0_f64).sqrt();
    
    c.bench_function("mandorla_search", |b| {
        let inner = InMemoryBackend::new();
        let refiner = MandorlaRefiner::new(MandorlaConfig::default());
        let mut backend = MandorlaBackend::new(inner, refiner);
        
        // Populate backend
        for i in 0..1000 {
            let item = create_test_item(&format!("item_{}", i), val);
            backend.store(item).unwrap();
        }
        
        let query = vec![val; 8];
        
        b.iter(|| {
            backend.search(black_box(&query), black_box(10)).unwrap()
        });
    });
}

#[cfg(feature = "optimization")]
fn bench_full_stack(c: &mut Criterion) {
    let val = 1.0 / (8.0_f64).sqrt();
    
    c.bench_function("full_stack_search", |b| {
        // Layer all optimizations in a valid order
        // Note: OphanBackend requires Clone, so it must come before FilteredBackend
        let base = InMemoryBackend::new();
        let sharded = OphanBackend::new(base);
        let filter = StabilityFilter::new(StabilityFilterConfig::default());
        let filtered = FilteredBackend::new(sharded, filter);
        let routed = AdaptiveRouter::new(filtered, RouterConfig::default());
        let refiner = MandorlaRefiner::new(MandorlaConfig::default());
        let mut optimized = MandorlaBackend::new(routed, refiner);
        
        // Populate backend
        for i in 0..1000 {
            let item = create_test_item(&format!("item_{}", i), val);
            optimized.store(item).unwrap();
        }
        
        let query = vec![val; 8];
        
        b.iter(|| {
            optimized.search(black_box(&query), black_box(10)).unwrap()
        });
    });
}

criterion_group!(
    benches,
    bench_baseline,
);

#[cfg(feature = "optimization")]
criterion_group!(
    optimization_benches,
    bench_stability_filter,
    bench_ophan_sharding,
    bench_adaptive_router,
    bench_mandorla_refiner,
    bench_full_stack,
);

#[cfg(feature = "optimization")]
criterion_main!(benches, optimization_benches);

#[cfg(not(feature = "optimization"))]
criterion_main!(benches);
