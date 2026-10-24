# QSO — Metatron Quantum State Operator

[![CI](https://github.com/LashSesh/qso/actions/workflows/ci.yml/badge.svg)](https://github.com/LashSesh/qso/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/metatron-qso-rs.svg)](https://crates.io/crates/metatron-qso-rs)
[![PyPI](https://img.shields.io/pypi/v/metatron-qso.svg)](https://pypi.org/project/metatron-qso/)
[![docs.rs](https://img.shields.io/docsrs/metatron-qso-rs)](https://docs.rs/metatron-qso-rs)
[![Rust 1.89+](https://img.shields.io/badge/rust-1.89%2B-orange.svg)](https://www.rust-lang.org/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](metatron_qso_py/README.md)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Validated quantum simulation in pure Rust, with Python bindings and a
command-line tool.**

QSO simulates quantum circuits exactly — as pure states or as density
matrices with noise — and runs variational and textbook algorithms on top of
them. It also simulates continuous-time quantum walks on arbitrary graphs.
Its signature geometry is **Metatron's Cube**, treated with mathematical
precision: the complete graph K₁₃ drawn on the 13 centres of the Fruit of
Life.

The rule of the project: **no claim without a test.** Every number in this
README is reproduced by
[`metatron-qso-rs/tests/validation.rs`](metatron-qso-rs/tests/validation.rs)
or by the benchmark report you can run yourself.

> **Version 1.0.0** is the first stable release and a complete rewrite of
> the 0.1 prototype. Version 0.1 simulated a single 13-level system instead
> of qubits, and several of its results and subsystems did not hold up; the
> [audit](docs/AUDIT_v0.1.md) documents why. The 0.1 code remains available
> in the git history.

## Contents

- [What is in the box](#what-is-in-the-box)
- [Validated results](#validated-results)
- [Metatron's Cube, precisely](#metatrons-cube-precisely)
- [Installation](#installation)
- [Quick start](#quick-start) — Rust, Python, command line
- [Examples](#examples)
- [Command-line reference](#command-line-reference)
- [Performance](#performance)
- [Scope and limits](#scope-and-limits)
- [Conventions](#conventions)
- [Repository layout](#repository-layout)
- [Development](#development)
- [Documentation](#documentation)
- [License](#license)

## What is in the box

| Area | Contents |
|---|---|
| **Statevector simulator** | n-qubit pure states (up to 34 qubits) with multi-threaded kernels. Gates: H, X, Y, Z, S, S†, T, T†, √X, √X†, RX, RY, RZ, P, U, SWAP, RXX, RYY, RZZ and custom unitaries on up to 6 qubits, each with any number of control qubits. Sampling, mid-circuit measurement, reduced density matrices, entanglement entropy |
| **Circuits** | Builder API with symbolic, differentiable parameters (`θ = a·p[i] + b`), inverse, binding, depth and gate counts. OpenQASM 2.0 export *and* import |
| **Observables** | Pauli strings and sums with a parser (`"0.5*Z0 Z1 - X2"`), matrix-free expectation values and `H\|ψ⟩`, exact ground states by Lanczos |
| **Noise** | Density-matrix simulator (up to 14 qubits) on the same kernels. Depolarising (1 and 2 qubits), amplitude damping, phase damping, bit flip, phase flip, thermal relaxation (T₁/T₂), custom Kraus channels, readout error, noise models applied after every gate |
| **Gradients** | Adjoint differentiation (all partial derivatives for about three circuit runs), parameter-shift rules (2-term, and 4-term for controlled rotations), finite differences |
| **Algorithms** | VQE with seeded parallel restarts; QAOA for weighted MaxCut (fast diagonal simulation, exact optimum by enumeration, INTERP layer-wise initialisation); Grover search with any number of marked items; QFT; phase estimation; variational classifier with data re-uploading; graph classifier with trainable QAOA-type layers |
| **Models** | Transverse-field Ising, XXZ Heisenberg, MaxCut cost operator, H₂ / STO-3G (2 qubits, with full-CI reference) |
| **Optimisers** | L-BFGS with strong-Wolfe line search, Adam, gradient descent with momentum, Nelder–Mead, SPSA |
| **Graphs** | Weighted graphs; complete, cycle, path, star, bipartite, hypercube, grid/torus, Petersen and random graphs; the five Platonic solids; the Metatron's Cube / Fruit-of-Life family; **exact automorphism groups** |
| **Quantum walks** | Continuous-time walks on any graph — exact on up to 2048 nodes, Krylov propagation beyond — with limiting and time-averaged distributions and mixing times. Childs–Goldstone spatial search with optimised hopping rate, Lindblad dephasing, classical random-walk baselines |
| **Boson sampling** | Permanents (Glynn's formula, `O(n·2ⁿ)`), Haar-random interferometers, exact output probabilities for indistinguishable and distinguishable photons, exact sampling (Clifford & Clifford) |
| **Backends** | Statevector and noisy density-matrix backends, shot-based estimation with qubit-wise-commuting measurement groups |
| **Interfaces** | Rust library `metatron_qso`, Python package `metatron_qso` (NumPy I/O, type stubs), `qso` command-line tool with JSON output |
| **Reproducibility** | One explicit, portable seeded RNG (xoshiro256++) for all randomness; parallel sums give bit-identical results for any thread count |

## Validated results

| Quantity | QSO | Reference |
|---|---:|---:|
| H₂ / STO-3G total energy at 0.735 Å (VQE) | −1.137306 Ha | −1.137306 Ha (full CI) |
| QAOA on a ring, depth p = 1, 2, 3, 4 | 0.75, 0.8333, 0.875, 0.9 | (2p+1)/(2p+2), Mbeng et al. 2019 |
| QAOA p = 1 on the cube graph Q₃ | 0.692450 | 0.6925, 3-regular triangle-free (Farhi et al. 2014) |
| QAOA p = 2 on the Heawood graph (girth 6) | 0.7559 | 0.7559, Wurtz & Love 2021 |
| QAOA p = 3 on Metatron's Cube (maximum cut 42) | 0.99997 | — |
| Grover, 10 qubits, 25 iterations | 0.999461 | sin²((2k+1)θ) = 0.999461 |
| Quantum walk on Metatron's Cube, limiting P(centre) | 0.857988 | 145/169 |
| Spatial search on Metatron's Cube, γ = 1/13, t = π√13/2 | 1.000000 | 1 |
| Automorphism group orders: Metatron's Cube / Fruit of Life / dodecahedron | 6 227 020 800 / 12 / 120 | 13! / D₆ / I_h |
| Hong–Ou–Mandel: two photons on a 50:50 beam splitter, coincidence probability | 0 (distinguishable: 0.5) | 0, Hong, Ou & Mandel 1987 |
| Full bunching of n photons, boson / distinguishable probability | n! for n = 1 … 6 | n! |
| State vectors of random circuits, largest difference to Qiskit Aer, Qulacs, Cirq, PennyLane | ≤ 3 × 10⁻¹⁵ | — |

The test-suite (183 Rust tests and 196 Python tests, run in CI on Linux,
macOS and Windows) also checks:

- every gate against an independent dense-matrix reference, and the QFT
  against the DFT matrix;
- exact phase estimation, and adjoint vs. parameter-shift vs.
  finite-difference gradients;
- Lanczos and Krylov propagation against dense diagonalisation;
- the symmetry groups of all Platonic solids (24, 48, 48, 120, 120);
- T₁/T₂ decay, the Zeno limit of dephased walks, and classical hitting
  times against closed forms;
- property tests on random circuits (norm, inverse, QASM round trips,
  statevector = density matrix) and on random graphs (automorphism orders
  against brute force);
- robustness: invalid input yields an error, never a panic or a wrong
  result.

## Metatron's Cube, precisely

The Fruit of Life consists of 13 equal circles: one in the centre, six
touching it, and six more touching those along the same rays. Joining every
pair of the 13 centres gives the 78 lines of Metatron's Cube.

* **As a graph it is K₁₃.** Every node is adjacent to every other, the
  symmetry group is the full S₁₃ (order 13! = 6 227 020 800), and the
  Laplacian spectrum is {0, 13¹²}.
* **The geometry lives in the embedding.** There are seven distinct line
  lengths: 1, √3, 2, √7, 3, 2√3 and 4, with multiplicities 18, 18, 15, 12, 6, 6
  and 3.
* **Short lines recover the structure.** Keeping only lines up to a given
  length gives a family of graphs (`metatron:<length>`). The Fruit-of-Life
  **contact graph** (tangent circles, 18 edges) has exactly the hexagonal
  symmetry D₆ of order 12.
* **On the Platonic solids:** K₁₃ contains every graph on at most 13
  vertices, so it trivially contains the tetrahedron, octahedron, cube and
  icosahedron graphs, but not the 20-vertex dodecahedron. The drawing shows
  planar *projections* of the solids. QSO provides the solids themselves as
  graphs (`platonic:<name>`).

What quantum dynamics on these graphs looks like:

* A quantum walk started at the centre of the Cube stays there with
  time-averaged probability exactly 145/169.
* Spatial search with γ = 1/13 finds a marked node with probability 1 at
  t = π√13/2 ≈ 5.66. A classical random walk needs 12 steps on average.
* QAOA reaches 99.997 % of the maximum cut at depth 3.

```text
$ qso graph fruit-of-life
Fruit of Life (contact graph)
  nodes 13, edges 18, degree 1..6, connected true, bipartite false
  diameter Some(4), triangles 6, algebraic connectivity 0.585786
  |Aut| = 12 (log10 1.079), orbits 3
```

## Installation

**Python package** ([PyPI](https://pypi.org/project/metatron-qso/))

```bash
pip install metatron-qso
```

Prebuilt wheels for Linux (x86_64, aarch64), macOS (Intel and Apple silicon)
and Windows (x86_64); one `abi3` wheel covers CPython 3.9 and newer.

**Rust library** ([crates.io](https://crates.io/crates/metatron-qso-rs),
Rust ≥ 1.89)

```bash
cargo add metatron-qso-rs
```

The library is imported as `metatron_qso`. Features: `parallel` (default;
multi-threading via rayon) and `cli`.

**Command-line tool**

```bash
cargo install metatron-qso-rs --features cli
```

**From source.** No C/C++ toolchain or system libraries are needed, only Rust
(and maturin for the Python package):

```bash
git clone https://github.com/LashSesh/qso && cd qso
cargo install --path metatron-qso-rs --features cli
pip install maturin && maturin develop --release -m metatron_qso_py/Cargo.toml
```

## Quick start

### Rust

```rust
use metatron_qso::prelude::*;

fn main() -> Result<()> {
    // VQE for H₂.
    let h2 = models::h2_sto3g();
    let ansatz = ansatz::real_amplitudes(2, 1, Entanglement::Linear);
    let vqe = Vqe::new(h2.hamiltonian.clone(), ansatz)?
        .with_restarts(4)
        .run()?;
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
        &c,
        &zz,
        &[0.3],
        None,
        GradientMethod::Adjoint,
    )?;
    println!("<ZZ> = {e:.4}, d<ZZ>/dθ = {:.4}", grad[0]);

    // Quantum walk and spatial search.
    let walk = ContinuousWalk::new(&metatron::fruit_of_life(), Generator::Laplacian, 1.0)?;
    println!("P(t=2) = {:?}", walk.probabilities(0, 2.0)?);
    Ok(())
}
```

This is [`examples/quickstart.rs`](metatron-qso-rs/examples/quickstart.rs)
(`cargo run --release --example quickstart`). It prints `E = -1.137306 Ha`,
`ratio = 0.9991`, `<ZZ> = 0.9553, d<ZZ>/dθ = -0.2955` (that is cos 0.3 and
−sin 0.3) and the walk probabilities.

### Python

```python
import metatron_qso as qso

h2 = qso.h2_sto3g()
result = qso.run_vqe(h2["hamiltonian"], qso.real_amplitudes(2, 1), restarts=4)
print(result["energy"] + h2["nuclear_repulsion"])          # -1.137306

qaoa = qso.MaxCutQaoa(qso.Graph.metatron_cube(), 2)
print(qaoa.optimize()["approximation_ratio"])              # 0.9991

walk = qso.ContinuousWalk(qso.Graph.fruit_of_life())
print(walk.limiting_distribution(0))                       # numpy array

best = qso.SpatialSearch.optimize(qso.Graph.from_spec("metatron"), [3])
print(best["gamma"], best["success_probability"])          # 0.0769 (= 1/13), 1.0
```

The full Python API is described in
[`metatron_qso_py/README.md`](metatron_qso_py/README.md).

### Command line

```text
$ qso vqe --model tfim:6,1,0.8 --layers 6 --restarts 8
VQE (lbfgs, 6 qubits, 6 layers): E = -6.4397095536  exact = -6.4397097308  error = 1.771e-7

$ qso qaoa --graph petersen --p 4 --layerwise
p=1  <C>=10.386751  C_max=12  ratio=0.865563  P(optimal)=0.1682  most likely 1110001011 (cut 12)
p=2  <C>=11.105320  C_max=12  ratio=0.925443  P(optimal)=0.4490  most likely 1110001011 (cut 12)
p=3  <C>=11.648892  C_max=12  ratio=0.970741  P(optimal)=0.7626  most likely 1100110110 (cut 12)
p=4  <C>=11.936517  C_max=12  ratio=0.994710  P(optimal)=0.9767  most likely 1001101101 (cut 12)

$ qso search --graph hypercube:8 --marked 17 --generator adjacency
search on Q8 for [17]: gamma 0.146868, best time 27.522674, success 0.754627 (expected time with repetition 36.4719)
```

## Examples

| Rust (`cargo run --release --example …`) | Shows |
|---|---|
| [`quickstart`](metatron-qso-rs/examples/quickstart.rs) | The snippet above |
| [`metatron_walks`](metatron-qso-rs/examples/metatron_walks.rs) | Symmetry groups, line lengths, coherent and dephased walks, spatial search vs. classical hitting times on Metatron's Cube and the Fruit of Life |
| [`qaoa_maxcut`](metatron-qso-rs/examples/qaoa_maxcut.rs) | QAOA depth 1–4 with INTERP on rings, Q₃, Petersen, random 3-regular and Fruit-of-Life graphs |
| [`vqe_h2`](metatron-qso-rs/examples/vqe_h2.rs) | VQE for H₂ with every optimiser, shot-based and noisy energy estimates |
| [`circuits`](metatron-qso-rs/examples/circuits.rs) | GHZ state, OpenQASM export, Grover's sine law, phase estimation |
| [`benchmark_report`](metatron-qso-rs/examples/benchmark_report.rs) | The performance and validation tables of [BENCHMARKS.md](docs/BENCHMARKS.md) |
| [`technical_note`](metatron-qso-rs/examples/technical_note.rs) | Every number of the [technical note](docs/technical-note/), with its reference, as JSON |

Python scripts are in [`metatron_qso_py/examples/`](metatron_qso_py/examples/):
`01_metatron_quantum_walk.py`, `02_qaoa_maxcut.py`, `03_vqe_h2.py`,
`04_grover_and_qft.py`, `05_noise.py`, `06_boson_sampling.py` and
`07_graph_classifier.py`.

## Command-line reference

`qso <command> [options]`. Every command accepts `--json` for
machine-readable output; `qso <command> --help` lists all options.

| Command | Purpose | Main options |
|---|---|---|
| `graph [SPEC]` | Structure, Laplacian spectrum, automorphism group and orbits | — |
| `walk` | Node probabilities at time t and limiting distribution | `--graph`, `--start`, `--time`, `--gamma`, `--generator laplacian\|adjacency` |
| `search` | Quantum-walk spatial search (rate and time optimised unless `--gamma` is given) | `--graph`, `--marked 3,7`, `--gamma`, `--generator`, `--t-max` |
| `qaoa` | QAOA for MaxCut with exact optimum and approximation ratio | `--graph`, `--p`, `--restarts`, `--seed`, `--layerwise` |
| `vqe` | VQE with an EfficientSU2 ansatz, compared with the exact ground energy | `--model h2\|tfim:N,J,H\|heisenberg:N,J,Δ`, `--hamiltonian "<Pauli sum>"`, `--qubits`, `--layers`, `--method`, `--restarts`, `--seed` |
| `grover` | Grover search, simulated vs. theoretical success | `--qubits`, `--marked`, `--iterations` |
| `qasm FILE` | Simulate an OpenQASM 2.0 file and sample counts | `--shots`, `--seed`, `--noise p1,p2` |

Graph specifications (also `Graph.from_spec` in Python):

| Spec | Graph |
|---|---|
| `metatron` | Metatron's Cube (K₁₃ with the Fruit-of-Life embedding) |
| `metatron:<length>` | Lines of Metatron's Cube up to a length (`metatron:1` = contact graph) |
| `fruit-of-life` | Fruit-of-Life contact graph |
| `complete:N`, `cycle:N`, `path:N`, `star:N` | Standard families |
| `hypercube:D`, `petersen`, `heawood`, `bipartite:A,B` | |
| `grid:RxC`, `torus:RxC` | Rectangular grid, periodic grid |
| `platonic:NAME` | `tetrahedron`, `cube`, `octahedron`, `dodecahedron`, `icosahedron` |
| `random-regular:N,D[,SEED]`, `erdos-renyi:N,P[,SEED]` | Random graphs |

## Performance

Measured with `cargo run --release --example benchmark_report` on a
4-thread cloud VM (full tables, method and run-to-run variance in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md)):

| Workload | Time |
|---|---:|
| 24-qubit layer (71 gates: H, RZ, CX chain) | 0.46 s (0.39 ns per gate·amplitude) |
| Full gradient, 10 qubits, 100 parameters, adjoint | 0.53 ms (52× faster than parameter shift) |
| QAOA value + gradient, 20 nodes, p = 3 | 0.17 s |
| Quantum walk on a 100 × 100 torus (10 000 nodes), Krylov | 84 ms |
| Automorphism group of Metatron's Cube (13!) | 3.0 ms |
| Permanent of a 24 × 24 complex matrix | 0.14 s |

Timings varied between runs on this shared VM (up to 2× for
sub-millisecond workloads, up to 1.5× for the larger ones).

On the same machine, with every framework at its default settings, QSO
simulates 20–24-qubit circuits about as fast as Qulacs or up to 1.4× faster
and 1.7–3.3× faster than Qiskit Aer and PennyLane Lightning; Qulacs is 2×
faster on 16 qubits. Its adjoint gradient is the fastest of those compared,
and its permanents are 1.1–6.5× faster than The Walrus. Method and full
tables:
[technical note](docs/technical-note/), Section 3.8.

## Scope and limits

QSO is a classical simulator running on one machine (multi-threaded CPU).
Know its boundaries:

| Limit | Value | Reason |
|---|---|---|
| Pure-state register | 34 qubits | 2³⁴ amplitudes = 256 GiB; practical limit is your RAM (16 bytes per amplitude) |
| Density-matrix register | 14 qubits | 4¹⁴ entries = 4 GiB |
| Pauli strings | qubits 0–63 | Stored as two 64-bit masks |
| Custom unitary gates | up to 6 target qubits | Dense kernel |
| Exact walk analysis | up to 2048 nodes | Dense eigendecomposition. Beyond that, `evolve`/`probabilities` use Krylov; limiting distributions and spatial search need the dense path |
| Automorphism groups | intended for up to a few hundred nodes | Exact search, worst-case exponential like every known algorithm |
| OpenQASM | version 2.0, `qelib1.inc` gates | No custom `gate` definitions, classical control or `reset`; version 3 is not supported |

Not included: GPU acceleration, tensor-network or stabiliser simulation,
fermion-to-qubit mappings beyond the bundled H₂ Hamiltonian, and
submission to quantum hardware. To run on hardware, export a circuit with
`to_qasm2` and submit it with the vendor's SDK.

## Conventions

* Qubit 0 is the least significant bit of a basis index (as in Qiskit); bit
  strings print qubit n−1 first.
* Time evolution is `exp(−iHt)`; rotations are `R_P(θ) = exp(−iθP/2)`;
  `P(λ) = diag(1, e^{iλ})`; `U(θ, φ, λ)` follows OpenQASM.
* Quantum walks use `H = γL` (Laplacian) or `H = −γA` (adjacency).
* Invalid input returns `QsoError` in Rust and raises `ValueError` in
  Python.

## Repository layout

```
metatron-qso-rs/            Rust library `metatron_qso` and the `qso` CLI
  src/quantum/              states, gates, kernels, circuits, Pauli sums, OpenQASM
  src/noise/                density matrices, channels, noise models
  src/algorithms/           gradients, ansätze, models, VQE, QAOA, Grover, QFT/QPE, classifiers
  src/optimize/             L-BFGS, Adam, gradient descent, Nelder–Mead, SPSA
  src/graph/                graphs, generators, Metatron family, automorphisms
  src/walks/                quantum walks, spatial search, dephasing, classical walks
  src/linalg/               Hermitian eigensolvers, Lanczos, Krylov, sparse matrices
  src/photonics/            permanents, interferometers, boson sampling
  src/backend.rs            execution backends and shot-based estimation
  src/bin/qso.rs            command-line tool
  tests/                    validation, robustness and property tests
  examples/, benches/       runnable examples, criterion benchmarks
metatron_qso_py/            Python package (PyO3 + NumPy): src/, python/, tests/, examples/
docs/                       audit of v0.1, 1.0 plan, roadmap, architecture, benchmarks
docs/technical-note/        technical note: source, data and build script
benchmarks/cross_framework/ comparison with Qiskit Aer, Qulacs, Cirq, PennyLane, The Walrus
docs/archive/               concept documents of v0.1 (PDF)
.github/workflows/          CI, release and benchmark workflows
```

## Development

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p metatron-qso-rs --all-features      # unit, validation, robustness, property and doc tests
cargo bench -p metatron-qso-rs                    # criterion benchmarks

# Python (inside a virtualenv)
pip install maturin numpy pytest
maturin develop --release -m metatron_qso_py/Cargo.toml
pytest metatron_qso_py/tests
```

CI runs formatting, Clippy and rustdoc; tests on Linux, macOS and Windows; a
build without multi-threading; the minimum Rust version; all examples and
CLI smoke tests; and the Python tests on 3.9 and 3.13. Pushing a tag
`vX.Y.Z` builds and tests the wheels and the source distribution and
publishes the release to crates.io, PyPI and GitHub
([CONTRIBUTING.md](CONTRIBUTING.md#releasing)).

New features follow one rule ([CONTRIBUTING.md](CONTRIBUTING.md)): a claim
needs a test against an independent reference.

## Documentation

| Document | Content |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the simulator works: kernels, determinism, gradients, Krylov methods, automorphism search, walks |
| [Technical note](docs/technical-note/QSO_Technical_Note_v2.pdf) | Validated benchmarks of every algorithm family, comparison with Qiskit Aer, Qulacs, Cirq and PennyLane, correction of the 2025 note ([source and data](docs/technical-note/); DOI [10.5281/zenodo.23623226](https://doi.org/10.5281/zenodo.23623226)) |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | Measured performance with method and variance |
| [benchmarks/cross_framework/](benchmarks/cross_framework/) | Reproducible comparison with other frameworks |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Plans for the 1.x series |
| [docs/AUDIT_v0.1.md](docs/AUDIT_v0.1.md) | What was wrong with version 0.1, with evidence |
| [docs/QSO_1.0_PLAN.md](docs/QSO_1.0_PLAN.md) | Goals, scope decisions and validation targets of the rewrite |
| [metatron_qso_py/README.md](metatron_qso_py/README.md) | Python API |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [docs.rs/metatron-qso-rs](https://docs.rs/metatron-qso-rs) | Rust API reference (locally: `cargo doc -p metatron-qso-rs --open`) |

## License

Apache License 2.0, see [LICENSE](LICENSE). © 2025–2026 Sebastian Klemm.
