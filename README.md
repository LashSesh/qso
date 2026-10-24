# QSO 2.0 — Metatron Quantum State Operator

[![CI](https://github.com/LashSesh/qso/actions/workflows/ci.yml/badge.svg)](https://github.com/LashSesh/qso/actions/workflows/ci.yml)
[![Rust 1.89+](https://img.shields.io/badge/rust-1.89%2B-orange.svg)](https://www.rust-lang.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Validated quantum simulation in pure Rust, with Python bindings.**

QSO simulates quantum circuits and continuous-time quantum walks exactly and
runs the standard variational and textbook algorithms on top of them. Its
signature geometry is **Metatron's Cube**, analysed honestly: it is the
complete graph K₁₃ drawn on the 13 centres of the Fruit of Life.

Every quantitative statement in this README is reproduced by an automated
test ([`tests/validation.rs`](metatron-qso-rs/tests/validation.rs)) or by a
benchmark you can run yourself. Version 2.0 is a ground-up rewrite; the
[audit of v0.1](docs/AUDIT_v0.1.md) explains why.

## Features

| Area | What you get |
|---|---|
| **Statevector simulator** | n-qubit states (up to 34 qubits), multi-threaded bit-twiddling kernels, every standard gate plus arbitrary controls and custom unitaries, sampling, mid-circuit measurement, reduced density matrices, entanglement entropy |
| **Circuits** | symbolic, differentiable parameters (`θ = a·p[i] + b`), inverse, depth, gate counts, OpenQASM 2.0 export *and* import |
| **Observables** | Pauli strings and sums with a parser (`"0.5 Z0 Z1 - X2"`), matrix-free expectation values and `H|ψ⟩`, exact ground states by Lanczos |
| **Noise** | density-matrix simulator reusing the same kernels; depolarising, amplitude/phase damping, bit/phase flip, thermal relaxation (T₁/T₂), readout error |
| **Gradients** | adjoint differentiation (all gradients for ≈ 3 circuit runs), parameter-shift rules (2- and 4-term), finite differences |
| **Algorithms** | VQE, QAOA-MaxCut (exact optimum, INTERP layer-wise initialisation), Grover, QFT, phase estimation, variational classifier with data re-uploading |
| **Optimisers** | L-BFGS (strong Wolfe), Adam, momentum gradient descent, Nelder–Mead, SPSA |
| **Graphs** | weighted graphs, standard families, the five Platonic solids, the Metatron / Fruit-of-Life family, **exact automorphism groups** |
| **Quantum walks** | CTQW on any graph (exact or Krylov for 10⁴+ nodes), limiting and time-averaged distributions, mixing times, Childs–Goldstone spatial search with optimal hopping rate, Lindblad dephasing, classical baselines |
| **Backends** | statevector and noisy density-matrix backends, shot-based estimation with qubit-wise-commuting measurement groups |
| **Tooling** | `qso` command-line tool (JSON output), Python package `metatron_qso` (numpy I/O), criterion benchmarks, deterministic seeded RNG |

## Validated results

| Quantity | QSO | Reference |
|---|---:|---:|
| H₂ / STO-3G total energy at 0.735 Å (VQE) | −1.137306 Ha | −1.137306 Ha (full CI) |
| QAOA ratio on a ring, depth p = 1…4 | 0.75, 0.8333, 0.875, 0.9 | (2p+1)/(2p+2) — Mbeng et al. 2019 |
| QAOA p = 1 on the cube graph Q₃ | 0.692450 | 0.6925 (3-regular, triangle-free; Farhi et al. 2014) |
| QAOA p = 3 on Metatron's Cube (max cut 42) | 0.99997 | — |
| Grover, 10 qubits, 25 iterations | 0.999461 | sin²((2k+1)θ) = 0.999461 |
| Quantum walk on Metatron's Cube, limiting P(centre) | 0.857988 | 145/169 |
| Spatial search on Metatron's Cube, γ = 1/13, t = π√13/2 | 1.000000 | 1 |
| Automorphism group orders: Metatron's Cube / Fruit of Life / dodecahedron | 6 227 020 800 / 12 / 120 | 13! / D₆ / Iₕ |

Further checks in the test-suite: every gate against an independent dense
reference, QFT against the DFT matrix, exact phase estimation, adjoint =
parameter-shift = finite-difference gradients, Lanczos and Krylov against
dense diagonalisation, Platonic solid symmetry groups (24, 48, 48, 120, 120),
T₁/T₂ decay of thermal relaxation, the Zeno limit of dephased walks, and
classical hitting times against closed forms.

## Metatron's Cube, precisely

The Fruit of Life consists of 13 equal circles: one in the centre, six
touching it, six more touching those along the same rays. Joining every
pair of the 13 centres gives the 78 lines of Metatron's Cube.

* As a graph it is **K₁₃**: every node is adjacent to every other, the
  symmetry group is the full S₁₃ (order 13!), and the Laplacian spectrum is
  {0, 13¹²}. All geometric structure lives in the embedding: seven distinct
  line lengths 1, √3, 2, √7, 3, 2√3, 4 with multiplicities 18, 18, 15, 12, 6,
  6, 3.
* Keeping only lines up to a given length recovers the geometry. The
  **Fruit-of-Life contact graph** (tangent circles, 18 edges) has exactly
  the hexagonal symmetry D₆ of order 12.
* K₁₃ contains every graph on ≤ 13 vertices, so it trivially "contains" the
  tetrahedron, octahedron, cube and icosahedron graphs, but not the
  20-vertex dodecahedron. The drawing shows planar *projections* of
  Platonic solids. QSO provides the solids themselves as graphs.

```text
$ qso graph fruit-of-life
Fruit of Life (contact graph)
  nodes 13, edges 18, degree 1..6, connected true, bipartite false
  diameter Some(4), triangles 6, algebraic connectivity 0.585786
  |Aut| = 12 (log10 1.079), orbits 3
```

## Quick start

### Rust

```toml
[dependencies]
metatron-qso-rs = { git = "https://github.com/LashSesh/qso" }
```

```rust
use metatron_qso::prelude::*;

fn main() -> Result<()> {
    // VQE for H₂.
    let h2 = models::h2_sto3g();
    let ansatz = ansatz::real_amplitudes(2, 1, Entanglement::Linear);
    let vqe = Vqe::new(h2.hamiltonian.clone(), ansatz)?.with_restarts(4).run()?;
    println!("E = {:.6} Ha", vqe.energy + h2.nuclear_repulsion); // −1.137306

    // QAOA for MaxCut on Metatron's Cube.
    let qaoa = MaxCutQaoa::new(&metatron::metatron_cube(), 2)?;
    let r = qaoa.optimize(&QaoaConfig::default())?;
    println!("ratio = {:.4}", r.approximation_ratio.unwrap());

    // A parametrised circuit with exact gradients.
    let mut c = Circuit::new(2);
    let theta = c.new_parameter();
    c.h(0).cx(0, 1).ry(1, theta);
    let zz: PauliSum = "Z0 Z1".parse()?;
    let (e, grad) = metatron_qso::algorithms::gradients::expectation_and_gradient(
        &c, &zz, &[0.3], None, GradientMethod::Adjoint)?;
    println!("<ZZ> = {e:.4}, d<ZZ>/dθ = {:.4}", grad[0]);

    // Quantum walk and spatial search.
    let walk = ContinuousWalk::new(&metatron::fruit_of_life(), Generator::Laplacian, 1.0)?;
    println!("P(t=2) = {:?}", walk.probabilities(0, 2.0)?);
    Ok(())
}
```

Runnable examples: `cargo run --release --example <name>` with
`quickstart` (the snippet above), `metatron_walks`, `qaoa_maxcut`, `vqe_h2`,
`circuits`, `benchmark_report`.

### Command line

```bash
cargo install --path metatron-qso-rs --features cli
qso graph platonic:dodecahedron       # structure, spectrum, |Aut| = 120
qso walk --graph metatron --time 0.5  # node probabilities + limiting distribution
qso search --graph hypercube:8 --marked 17 --generator adjacency
qso qaoa --graph petersen --p 4 --layerwise
qso vqe --model tfim:6,1,0.8 --layers 6 --restarts 8
qso grover --qubits 10 --marked 123
qso qasm circuit.qasm --shots 4096 --noise 0.001,0.01
qso vqe --model h2 --json             # every command supports --json
```

Graph specifications: `metatron`, `metatron:<max line length>`,
`fruit-of-life`, `complete:N`, `cycle:N`, `path:N`, `star:N`, `hypercube:D`,
`petersen`, `grid:RxC`, `torus:RxC`, `bipartite:A,B`, `platonic:NAME`,
`random-regular:N,D[,SEED]`, `erdos-renyi:N,P[,SEED]`.

### Python

```bash
pip install maturin
maturin develop --release -m metatron_qso_py/Cargo.toml   # inside a virtualenv
```

```python
import numpy as np
import metatron_qso as qso

h2 = qso.h2_sto3g()
result = qso.run_vqe(h2["hamiltonian"], qso.real_amplitudes(2, 1), restarts=4)
print(result["energy"] + h2["nuclear_repulsion"])          # -1.137306

qaoa = qso.MaxCutQaoa(qso.Graph.metatron_cube(), 2)
print(qaoa.optimize()["approximation_ratio"])

walk = qso.ContinuousWalk(qso.Graph.fruit_of_life())
print(walk.limiting_distribution(0))                       # numpy array
```

See [`metatron_qso_py/README.md`](metatron_qso_py/README.md) for the full
Python API.

## Performance

Measured with `cargo run --release --example benchmark_report` on a
4-thread cloud container (full table and method in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md)):

| Workload | Time |
|---|---:|
| 24-qubit layer (71 gates: H, RZ, CX chain) | 0.94 s (0.79 ns per gate·amplitude) |
| Full gradient, 10 qubits, 100 parameters, adjoint | 1.6 ms (43× faster than parameter shift) |
| QAOA value + gradient, 20 nodes, p = 3 | 0.31 s |
| Quantum walk on a 100 × 100 torus (10 000 nodes), Krylov | 0.13 s |
| Automorphism group of Metatron's Cube (13!) | 3.2 ms |

Timings varied between runs on this shared VM (up to 2× for sub-millisecond
workloads, up to 1.5× for the larger ones); see the benchmark page.

## Conventions

* Qubit 0 is the least significant bit of a basis index (as in Qiskit); bit
  strings print qubit n−1 first.
* Time evolution is `exp(−iHt)`; rotations are `R_P(θ) = exp(−iθP/2)`;
  `P(λ) = diag(1, e^{iλ})`; `U(θ, φ, λ)` follows OpenQASM.
* All randomness is explicit and seeded (portable xoshiro256++). Parallel
  reductions use fixed chunking, so results are bit-identical for any number
  of threads.

## Repository layout

```
metatron-qso-rs/     Rust library + `qso` CLI
  src/quantum/       states, gates, circuits, Pauli sums, OpenQASM
  src/noise/         density matrices and channels
  src/algorithms/    gradients, VQE, QAOA, Grover, QFT/QPE, classifier, models
  src/optimize/      L-BFGS, Adam, Nelder–Mead, SPSA
  src/graph/         graphs, Platonic solids, Metatron family, automorphisms
  src/walks/         quantum walks, search, dephasing, classical walks
  src/linalg/        eigensolvers, Lanczos, Krylov, sparse matrices
  tests/validation.rs  the claims in this README, as tests
  examples/, benches/
metatron_qso_py/     Python bindings (PyO3 + numpy)
docs/                audit of v0.1, 2.0 plan, architecture, benchmarks
```

## Development

```bash
cargo test -p metatron-qso-rs --all-features      # unit, validation and doc tests
cargo clippy --workspace --all-targets -- -D warnings
cargo bench -p metatron-qso-rs                    # criterion benchmarks
```

See [CONTRIBUTING.md](CONTRIBUTING.md). The one rule: a new claim needs a
test against an independent reference.

## License

Apache License 2.0, see [LICENSE](LICENSE). © 2025–2026 Sebastian Klemm.
