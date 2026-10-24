# QSO 2.0 — Plan

## Goal

Turn QSO into a small but first-rate quantum simulation framework in pure
Rust whose every claim is backed by a test against an analytic or literature
result. Metatron's Cube stays as the project's signature geometry, analysed
honestly (see [AUDIT_v0.1.md](AUDIT_v0.1.md) for why v0.1 could not be
evolved incrementally).

## Principles

1. **Validated, not asserted.** Each algorithm ships with tests against
   closed-form or published results (Grover amplitudes, QAOA p = 1 on rings
   and 3-regular graphs, H₂/STO-3G ground energy, Childs–Goldstone search,
   automorphism group orders, Lindblad steady states).
2. **Reproducible.** All randomness flows through an explicit, portable,
   seeded RNG; parallel reductions use fixed chunking so results are
   bit-identical for any thread count.
3. **Correct conventions, documented.** Little-endian qubit order (qubit 0 =
   least-significant bit, as in Qiskit), `U(t) = exp(−iHt)`, rotation gates
   `R_P(θ) = exp(−iθP/2)`.
4. **Fast where it matters.** Bit-twiddling kernels with rayon parallelism,
   specialised diagonal paths, matrix-free Lanczos/Krylov methods, adjoint
   differentiation (O(1) state copies for any number of parameters).
5. **Small surface, no dead weight.** Remove subsystems that are not quantum
   simulation or that cannot be made honest.

## Scope decisions

| v0.1 component | 2.0 decision |
|---|---|
| `metatron-qso-rs` (13-dim toy core) | **Rewritten** as the n-qubit simulation core (same crate name, lib `metatron_qso`). |
| `metatron_backend` (fake simulator, IBM stub) | **Replaced** by the core `backend` module: a real statevector backend plus OpenQASM 2.0 export for running on any vendor's hardware with their own tooling. |
| `metatron_qso_py` (broken packaging) | **Rewritten** with PyO3 + numpy, fixed packaging, pytest suite. |
| DTL / Kuramoto "58.5 % advantage" | **Removed** (definitional, not quantum). |
| SCS, TRITON, auto-tuning | **Removed.** Hyper-parameter search for 2.0 is done with real metrics (seeded multi-start, layer-wise QAOA initialisation); an external tuner such as Optuna can drive the Python API. |
| DioniceOS (`external/`), `metatron_dionice_bridge` | **Removed** from QSO; unrelated to quantum simulation. History preserves it; the ledger already has its own repository. |
| `metatron_telemetry` (synthetic data) | **Removed.** Results are emitted as JSON by the CLI and the API. |
| Concept PDFs | Moved to `docs/archive/` unchanged. |

## Architecture (crate `metatron-qso-rs`, lib `metatron_qso`)

```
error, rng, parallel          shared infrastructure
linalg   dense Hermitian eigh/expm, CSR sparse, Lanczos ground state,
         Krylov exp(−iHt)v
quantum  StateVector, gate kernels, Gate set, Circuit (+ parameters),
         PauliString/PauliSum, OpenQASM 2.0 export
noise    DensityMatrix (vectorised, reuses kernels), Kraus channels
graph    Graph, generators, Metatron's Cube family, automorphism groups
         (individualisation–refinement search, exact group order)
walks    CTQW (exact + Krylov), limiting distribution with degeneracies,
         spatial search, site-dephasing Lindblad walk, classical walks
optimize Adam, L-BFGS (strong Wolfe), Nelder–Mead, SPSA, unified minimize()
algorithms
         gradients (adjoint, parameter-shift), ansätze, VQE, QAOA-MaxCut
         (fast diagonal path, analytic gradients, exact optimum), Grover,
         QFT/QPE, Hamiltonian models (TFIM, Heisenberg, H₂, MaxCut)
backend  Backend trait, StatevectorSimulator, shot sampling
bin/qso  command-line interface with JSON output (feature `cli`)
```

Python package `metatron_qso` (crate `metatron_qso_py`) wraps graphs,
circuits, states, Pauli sums, VQE, QAOA, walks and search with numpy I/O.

## Validation targets (all automated)

| Area | Reference |
|---|---|
| Gates | Unitarity; algebraic identities (HZH = X, CX via H·CZ·H, …); random circuits vs. dense Kronecker reference |
| Pauli algebra | Expectation values vs. dense matrices; Hermiticity |
| Lanczos | vs. dense eigensolver; H₂ (parity, 2 qubits, R = 0.735 Å) → −1.857275 Ha electronic (−1.137306 Ha total) |
| Gradients | Adjoint = parameter-shift = finite differences |
| VQE | Reaches exact ground energy (Lanczos) of H₂, TFIM, Heisenberg within 1e-6 |
| QAOA | p = 1 on even ring → ratio 3/4; p = 1 on cube graph Q₃ (3-regular, triangle-free) → 0.6925; monotone in p; exact optimum by enumeration |
| Grover | P(success) = sin²((2k+1)θ) exactly |
| QFT/QPE | Exact phases k/2ᵗ recovered with probability 1 |
| Automorphisms | |Aut(Kₙ)| = n!, |Aut(Cₙ)| = 2n, Petersen 120, Q₃ 48, Q₄ 384, Fruit-of-Life contact graph 12 |
| CTQW | Unitarity; K₁₃ limiting distribution 145/169 vs 2/169; Krylov = exact |
| Spatial search | Complete graph, γ = 1/N: success → 1 at t = π√N/2 |
| Dephasing | Trace preserved; steady state uniform on connected graphs |
| Optimisers | Rosenbrock minimum, convex quadratics |
| Noise | Depolarising/amplitude-damping channels vs. closed forms |

## Phases

1. Foundation: infrastructure, linear algebra, statevector, gates, circuits,
   Pauli sums.
2. Graphs, Metatron geometry, automorphisms; quantum walks.
3. Optimisers, gradients, VQE, QAOA, Grover, QFT/QPE; density-matrix noise.
4. Backend, CLI, Python bindings, criterion benchmarks with measured numbers.
5. CI (fmt, clippy `-D warnings`, tests, docs, Python tests), README and
   documentation rewrite, removal of legacy components.
6. Independent review and fixes.
