//! Criterion benchmarks: `cargo bench -p metatron-qso-rs`.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use metatron_qso::algorithms::gradients::{GradientMethod, expectation_and_gradient};
use metatron_qso::prelude::*;

fn gate_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/layer");
    group.sample_size(10);
    for n in [12usize, 16, 20, 22] {
        let mut layer = Circuit::new(n);
        for q in 0..n {
            layer.h(q).rz(q, 0.3);
        }
        for q in 0..n - 1 {
            layer.cx(q, q + 1);
        }
        let mut psi = StateVector::zero(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| psi.apply_circuit(black_box(&layer), &[]).unwrap())
        });
    }
    group.finish();
}

fn qaoa_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("qaoa/value_and_gradient_p3");
    group.sample_size(10);
    for n in [10usize, 14, 18] {
        let g = generators::random_regular(n, 3, &mut Rng::seed_from_u64(1)).unwrap();
        let qaoa = MaxCutQaoa::new(&g, 3).unwrap();
        let x = [0.4, 0.6, 0.8, 0.5, 0.3, 0.1];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| qaoa.expectation_and_gradient(black_box(&x)).unwrap())
        });
    }
    group.finish();
}

fn vqe_gradients(c: &mut Criterion) {
    let mut group = c.benchmark_group("vqe/gradient_tfim10");
    group.sample_size(10);
    let h = models::transverse_field_ising(10, 1.0, 0.7, false);
    let circuit = ansatz::efficient_su2(10, 4, Entanglement::Linear);
    let theta: Vec<f64> = (0..circuit.num_parameters())
        .map(|k| 0.01 * k as f64)
        .collect();
    for (name, method) in [
        ("adjoint", GradientMethod::Adjoint),
        ("parameter-shift", GradientMethod::ParameterShift),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| {
                expectation_and_gradient(&circuit, &h, black_box(&theta), None, method).unwrap()
            })
        });
    }
    group.finish();
}

fn walks(c: &mut Criterion) {
    let mut group = c.benchmark_group("walks");
    group.sample_size(10);
    let cube = ContinuousWalk::new(&metatron::metatron_cube(), Generator::Laplacian, 1.0).unwrap();
    group.bench_function("metatron_probabilities", |b| {
        b.iter(|| cube.probabilities(0, black_box(1.3)).unwrap())
    });
    let torus = generators::grid_2d(100, 100, true);
    let big = ContinuousWalk::new(&torus, Generator::Laplacian, 1.0).unwrap();
    group.bench_function("torus_10k_krylov_t5", |b| {
        b.iter(|| big.probabilities(0, black_box(5.0)).unwrap())
    });
    group.finish();
}

fn symmetry(c: &mut Criterion) {
    let mut group = c.benchmark_group("automorphisms");
    group.sample_size(10);
    for (name, g) in [
        ("metatron_cube", metatron::metatron_cube()),
        (
            "dodecahedron",
            generators::platonic(generators::PlatonicSolid::Dodecahedron),
        ),
        ("hypercube5", generators::hypercube(5)),
    ] {
        group.bench_function(name, |b| b.iter(|| black_box(&g).automorphism_group()));
    }
    group.finish();
}

criterion_group!(
    benches,
    gate_kernels,
    qaoa_gradient,
    vqe_gradients,
    walks,
    symmetry
);
criterion_main!(benches);
