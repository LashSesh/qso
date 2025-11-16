# Architecture & Mapping Guide

This document captures the conceptual migration from the original Python
implementation to the new Rust-based Metatron QSO. The goal is not a
line-by-line port, but an evolution that leverages Rust's strengths around
safety, modularity, and performance.

## Module Mapping

| Python Module              | Rust Counterpart                            | Notes |
|---------------------------|----------------------------------------------|-------|
| `quantum_state.py`        | `src/quantum/state.rs`, `src/quantum/operator.rs` | Strongly-typed Hilbert space primitives built on `nalgebra`.
| `metatron_graph.py`       | `src/graph/metatron.rs`                      | Explicit graph construction with adjacency/laplacian extraction.
| `dtl.py`                  | `src/dtl/state.rs`, `src/dtl/operations.rs`, `src/dtl/resonator.rs`, `src/dtl/network.rs` | DTL states & Kuramoto dynamics with closure-based trajectories.
| `qso.py`                  | `src/qso.rs`, `src/hamiltonian.rs`, `src/params.rs` | High-level orchestration, Hamiltonian spectrum, symmetry tooling.
| `quantum_walk.py`         | `src/quantum_walk/`                          | Continuous-time walk engine, benchmarks, Krylov & scattering utilities.
| `examples.py` / tests     | `examples/basic.rs`, inline Rust unit tests  | Rust examples + `cargo test` for verification.

## Key Data Structures

- **Quantum State (`QuantumState`)** – `SVector<Complex64, 13>` storing
  amplitudes with normalization helpers and measurement primitives.
- **Quantum Operator (`QuantumOperator`)** – `SMatrix<Complex64, 13, 13>`
  supporting permutation-derived unitaries and eigen decompositions.
- **Metatron Graph (`MetatronGraph`)** – coordinates, edge lists, adjacency,
  Laplacian, and structural statistics.
- **DTL State (`DTLState`)** – enum-backed dynamic logic states with
  closure-based trajectories for oscillatory behaviour.
- **Resonator Network (`DTLResonatorNetwork`)** – explicit Euler integrator for
  Kuramoto dynamics on the Metatron topology.
- **Hamiltonian (`MetatronHamiltonian`)** – tight-binding operator with cached
  eigen decomposition and time evolution utilities.
- **Symmetry Group (`SymmetryGroup`)** – permutation closure over hexagonal
  generators, validating Hamiltonian invariance.
- **Quantum Walk Benchmarker (`QuantumWalkBenchmarker`)** – orchestrates mixing
  time, hitting time, scattering, and Krylov diagnostics over the Metatron
  topology.

## Quantum ↔ DTL Correspondence

- **Amplitude probabilities** map to tripolar logic intensities via
  `QuantumStateOperator::quantum_to_dtl`.
- **Resonator phases** map to DTL dynamic states through
  `DTLResonator::to_dtl_state`, which samples and interpolates phase
  trajectories before projecting them onto `[0,1]`.

## Extensibility Hooks

- **Hamiltonian backends:** `MetatronHamiltonian::time_evolution_operator`
  isolates exponentiation logic, making it straightforward to swap in GPU or
  sparse solvers.
- **Symmetry growth:** `SymmetryGroup` exposes permutation storage; a TODO
  tracks integrating full automorphism discovery via `petgraph`.
- **Hardware integration:** `QuantumStateOperator` centralises parameters and
  exposes a `QSOReport` summary, preparing the crate for control-plane bindings.

## Outstanding TODOs

- Implement automated Metatron automorphism enumeration (Schreier-Sims or
  `nauty`-style integration).
- Provide serialization helpers for dynamic `DTLState::Ld` trajectories.
- Extend `examples/` with Kuramoto synchronisation visualisation and spectral
  plots (potentially leveraging `plotters`).
- Formalise benchmarking harnesses in `benches/` for Hamiltonian exponentiation,
  resonator integration, and quantum-walk performance sweeps.
