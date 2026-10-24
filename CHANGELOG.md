# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-10-24

First stable release: a ground-up rewrite of the 0.1 prototype. The audit in [docs/AUDIT_v0.1.md](docs/AUDIT_v0.1.md)
showed that v0.1 simulated a single 13-level system (not qubits), that its
headline results were trivial or incorrect, and that several subsystems
returned placeholder data. 1.0 replaces it with a validated simulator. The
API is not compatible with 0.1.

### Added

- n-qubit statevector simulator with multi-threaded kernels, arbitrary
  controls, custom unitaries, sampling, measurement, entanglement entropy.
- Parametrised circuits with affine symbolic parameters; OpenQASM 2.0 export
  and import.
- Pauli strings and sums with parser, matrix-free expectation values,
  Lanczos ground states.
- Density-matrix simulator with Kraus channels, noise models and readout
  error.
- Adjoint, parameter-shift and finite-difference gradients.
- VQE, QAOA-MaxCut (exact optimum, INTERP), Grover, QFT, phase estimation,
  variational classifier, model Hamiltonians (TFIM, Heisenberg, MaxCut,
  H₂/STO-3G).
- Optimisers: L-BFGS, Adam, momentum gradient descent, Nelder–Mead, SPSA.
- Graph library with Platonic solids, the Metatron's Cube / Fruit-of-Life
  family and exact automorphism groups.
- Quantum walks: exact and Krylov CTQW, limiting/time-averaged
  distributions, mixing times, spatial search with optimised hopping rate,
  Lindblad dephasing, classical baselines.
- Statevector and density-matrix backends with shot-based estimation.
- `qso` command-line tool, criterion benchmarks, benchmark report example,
  validation test-suite, CI for Linux/macOS/Windows, MSRV and Python.
- Python package rewritten with PyO3 0.30 and numpy; importable and tested.

### Removed

- DioniceOS (`external/`), `metatron_dionice_bridge`, the Seraphic
  Calibration Shell, TRITON, the telemetry dashboard and the imitation IBM
  backend. They remain in the git history; see the audit for the reasons.
- Dynamic Tripolar Logic (its "58.5 % information advantage" is log₂3 − 1 by
  definition).

### Changed

- License metadata now matches the Apache-2.0 `LICENSE` file (v0.1 metadata
  said MIT).
- Minimum supported Rust version 1.89, edition 2024.

## [0.1.0] — 2025-11-16

Initial prototype. Superseded by 1.0.0; see the audit for an assessment.
