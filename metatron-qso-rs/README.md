# Metatron QSO (Rust Edition)

An idiomatic Rust reimagination of the Metatron Quantum State Operator, evolving
the original Python system into a modular, high-performance hybrid quantum-classical
framework. The library couples a 13-dimensional Hilbert space, the Metatron Cube
graph, Dynamic Tripolar Logic (DTL) resonator networks, and symmetry-aware
Hamiltonian dynamics.

## Highlights

- **Static typing & safety:** compile-time guarantees around dimensions,
  normalization, and operator compatibility.
- **High-performance numerics:** powered by [`nalgebra`](https://nalgebra.org/)
  for dense linear algebra and [`rayon`](https://docs.rs/rayon) ready for future
  parallel extensions.
- **Graph-native design:** explicit Metatron Cube construction using `petgraph`
  interoperability.
- **DTL dynamics:** modular Tripolar Logic primitives with both single
  resonators and network-level Kuramoto synchronisation.
- **Symmetry awareness:** discrete symmetry group utilities validate Hamiltonian
  invariants and prepare the ground for group-theoretic orbital analysis.
- **Quantum walk analytics:** high-end benchmarking harness with mixing time,
  hitting time, scattering, and Krylov subspace solvers tailored to the Metatron
  Cube topology.

## Crate Layout

```
src/
├── dtl/              # Dynamic Tripolar Logic (states, operations, resonators)
├── graph/            # Metatron Cube structure and analytics
├── hamiltonian.rs    # Tight-binding Hamiltonian and spectral tooling
├── params.rs         # Global configuration (J, ε, ω, κ)
├── qso.rs            # High-level orchestrator and reporting
├── quantum/          # Quantum states and operators on H₁₃
└── quantum_walk/     # Continuous-time walks, benchmarks, Krylov & scattering
```

Supplementary documentation lives in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Getting Started

```bash
cargo test
```

For exploratory usage, run the provided example:

```bash
cargo run --example basic
cargo run --example quantum_walk
```

The basic example constructs the full QSO stack, evolves a basis state, and
prints an analysis report. The `quantum_walk` example exercises the high-end
benchmarking harness (mixing time, hitting time, Krylov, scattering).

To generate the consolidated benchmark suite used in CI, run:

```bash
# Option 1: Write to file directly (recommended)
cargo run --release --bin quantum_walk_bench target/quantum_walk_bench.json

# Option 2: Write to stdout and redirect
cargo run --release --bin quantum_walk_bench > target/quantum_walk_bench.json

# Compare against baseline
cargo run --release --bin quantum_walk_bench_compare \
    ../ci/quantum_walk_baseline.json target/quantum_walk_bench.json
```

The first command emits a JSON report capturing mixing-time and hitting-time
statistics; the second ensures the results stay within the expected tolerance
window captured in `ci/quantum_walk_baseline.json`.

## Minimum Rust Version

Rust 1.78 (edition 2024) or later is recommended.

## Roadmap

- [ ] Expand symmetry group enumeration with automated graph automorphism
      discovery (`petgraph` + Schreier-Sims).
- [ ] Integrate optional GPU backends for Hamiltonian exponentiation.
- [ ] Extend benchmarking harnesses in `benches/` for quantum-walk performance
      studies.
- [ ] Add serialization stories for dynamic DTL trajectories.
- [ ] Provide bindings to external hardware control layers.
