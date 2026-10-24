# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.3.0] — 2026-10-24

Milestone 1.3 of the [roadmap](docs/ROADMAP.md), which combined the two
milestones planned as "quantum walks: exact at scale" and "simulation
performance". Every new feature is tested against a closed form, a
published result or an independent computation (named below). No
validated result of earlier releases changed.

### Added

- **Graph library.** Johnson, Kneser, Hamming, rook's, Paley, Shrikhande,
  circulant and Cayley graphs of Z₂ⁿ (constructors and spec strings such as
  `johnson:7,3`), Cartesian products and strongly regular parameters
  (`(10, 6, 3, 4)` for J(5, 2), `(13, 6, 2, 3)` for Paley(13)).
- **Canonical labelling and isomorphism** (`graph::canonical`,
  `Graph::canonical_form`, `is_isomorphic`, `isomorphism`): individualisation–
  refinement with Hopcroft-style colour refinement, automorphism and twin
  pruning and backjumping. Agrees with brute force on 300 random graphs, is
  invariant under relabelling and separates the cospectral Shrikhande and
  4×4 rook's graphs; K₄₀₀ and stars need one leaf, 8000-node cycles take
  well under a second.
- **Graph files** (`graph::io`): edge lists and GraphML (common subset,
  without an XML dependency), with size limits and a new fuzz target;
  `Graph.from_networkx` in Python by duck typing, without importing NetworkX.
- **Symmetry-reduced walks** (`walks::quotient`): `Quotient` (coarsest
  equitable partition by colour refinement, checked user partitions, and
  the distance partitions of hypercubes, Hamming, Johnson, complete and
  complete multipartite graphs without building the graph), `QuotientWalk`
  and `QuotientSearch`. Reduced and full walks agree to 1e-12; the walk on
  the 2³⁰-node hypercube Q₃₀ runs in 31 dimensions and matches
  C(30,k)·sin²ᵏ(γt)·cos²⁽³⁰⁻ᵏ⁾(γt).
- **Walks beyond 2048 nodes:** limiting and time-averaged distributions and
  mixing times through the start node's cyclic subspace
  (`linalg::cyclic_spectrum`, `ContinuousWalk::cyclic_dimension`).
- **State transfer** (`walks::transfer::StateTransfer`): decides perfect
  state transfer (strong cospectrality and the eigenvalue condition,
  confirmed numerically) and finds the best transfer in a time window.
  Paths P₂ and P₃ have perfect transfer and P₄…P₁₂ do not (Christandl et
  al. 2004); hypercube antipodes at γt = π/2 up to Q₃₀; the Krawtchouk
  chain for every length.
- **Discrete-time walks** (`walks::discrete`): `CoinedWalk` (Grover,
  Fourier, Hadamard or custom coins, flip-flop shift; marked nodes give the
  Shenvi–Kempe–Whaley search, checked against numpy on Q₁₀), `LineWalk`
  (Hadamard walk: σ/t → √(1 − 1/√2), Konno) and `SzegedyWalk` with quantum
  hitting times (n/4 on cycles; √-scaling against the classical hitting
  times on complete graphs and cycles).
- **Walks as circuits** (`walks::circuits`): exact circuits for circulant
  graphs on 2ⁿ nodes (QFT, Walsh-decomposed diagonal, inverse QFT) and for
  Cayley graphs of Z₂ⁿ, product formulas of order 1 and 2 for any graph
  (error ∝ steps⁻ᵒʳᵈᵉʳ, checked), exact diagonal unitaries.
- OpenQASM 2 export of **multi-controlled gates** (X, Z, single-qubit gates
  and swaps with any number of controls) by an exact, ancilla-free
  decomposition into singly controlled gates and Toffolis (Barenco et al.
  1995).
- **Batched evaluation:** `expectation_batch` and
  `expectation_and_gradient_batch` (Rust and Python) evaluate many
  parameter vectors in parallel; 3–4× faster than a loop at 10 qubits
  on 4 cores.
- CLI commands `iso`, `reduce`, `transfer`, `coined`, `hitting` and
  `walk-circuit`; graph arguments accept edge-list and GraphML files;
  `qso graph` reports strongly regular parameters and can save the graph.
- Example `metatron_atlas` and tutorial 05 (symmetry, state transfer and
  discrete walks); guide chapters on symmetry and state transfer and on
  discrete-time walks and circuits.
- Python: all of the above, with type stubs.

### Changed

- **Faster state-vector simulation** (`quantum::plan`): gates whose targets
  lie below qubit 14, and diagonal gates anywhere, are applied block by
  block to cache-sized blocks of 2¹⁴ amplitudes, the blocks in parallel;
  runs of single-qubit gates on higher qubits are fused. Against 1.2.0 on
  the same machine (medians of three alternating runs, see
  [BENCHMARKS.md](docs/BENCHMARKS.md)), layers of H, RZ and a CX chain run
  4.7× faster at 16 qubits and 1.6–2.6× faster at 18–24 qubits, single
  gate types 1.7–2.3× faster at 22 qubits; 10- and 14-qubit registers are
  unchanged. Planned and gate-by-gate simulation agree to 1e-12 on random
  circuits over the whole gate set, with controls and symbolic angles
  (also under Miri).
- QAOA cost layers use a table of the distinct cut values instead of a
  complex exponential per amplitude, and share a blocked pass with the
  mixer: value and gradient 1.15–1.45× faster.
- `Graph::from_weighted_edges` builds adjacency lists in O(m log m)
  whatever the edge order (was quadratic for reversed stars);
  `strongly_regular_parameters` counts 2-paths in O(n·k²) time and O(n)
  memory.

### Fixed

- Limiting and time-averaged distributions grouped eigenvalues closer than
  1e-9·max(1, ‖H‖) into one eigenspace. For Hamiltonians of small norm
  (small hopping rates) this merged distinct eigenvalues and returned
  wrong distributions. Eigenvalues are now grouped within 1e-11 of the
  spectral radius, with no absolute floor; results no longer depend on the
  scale of H (checked for γ from 1e-10 to 1e10).
- Spatial-search time optimisation refined only the best grid point and
  could return the lower of two peaks whose heights differ by less than
  the grid error. Every near-maximal peak is now refined, NaN values are
  skipped, and the earliest of equal maxima is reported.
- Lanczos breakdown detection scales with the off-diagonal entries as
  well (it could miss breakdowns for large hopping rates).

## [1.2.0] — 2026-10-24

Milestone 1.2 of the [roadmap](docs/ROADMAP.md): trust, supply chain and
documentation. Fuzzing and the review of the parsers that came with it
found six robustness bugs, all fixed (below); no validated result changed.

### Added

- Documentation site (<https://lashsesh.github.io/qso/>): a user guide
  (mdBook, `docs/guide`), four tutorials as notebooks (`docs/tutorials`)
  and the Python API reference (pdoc). The notebooks are executed and every
  code block of the guide is run on each build (`docs/build_site.py`, the
  *Docs* workflow), so none of them can go stale.
- Fuzz targets (`metatron-qso-rs/fuzz`) for the OpenQASM and Pauli-sum
  parsers, graph specifications and the JSON form of `Circuit`,
  `StateVector`, `Graph` and `PauliSum`; 60 s per target on every pull
  request, 10 min per target weekly.
- Miri checks of the `unsafe` statevector kernels on multi-threaded paths,
  and a test of every kernel against a textbook reference.
- API checks: `cargo-semver-checks` against the last release, `stubtest`
  for the Python type stubs.
- Weekly cross-validation, now also of noisy circuits: density matrices
  with depolarising and amplitude-damping noise agree with Qiskit Aer to
  3.5·10⁻¹⁵. `benchmarks/cross_framework/run.py --check` fails on any
  deviation above rounding level.
- Weekly benchmark tracking (`benchmark_report --json`,
  `benchmarks/track.py`): slowdowns of more than 25 % against the median of
  the last four runs fail the job.
- Wheels for musllinux (x86_64, aarch64) and Windows on Arm; Linux aarch64
  wheels are now built and tested natively; CI runs on Linux arm64 and
  Python 3.14.
- `generators::check_generated_size`, `MAX_GENERATED_NODES`,
  `MAX_GENERATED_EDGES`, `qasm::MAX_QASM_QUBITS`, `MAX_QASM_INSTRUCTIONS`.

### Changed

- Releases are published with Trusted Publishing (no stored registry
  tokens) from the protected environment `release`; PyPI uploads carry
  attestations.
- The Python type stubs declare constructors as `__new__`, mark classes
  `@final` and result dictionaries `@type_check_only`, and match the
  compiled module.
- JSON numbers round-trip exactly (`serde_json` with `float_roundtrip`).

### Fixed

Found by fuzzing (the unvalidated graph JSON by reviewing the targets):

- `from_qasm2` panicked on malformed brackets (`qreg q]3[;`, `h q]0[;`),
  overflowed on huge register sizes and could overflow the stack on deeply
  nested parameter expressions; it now rejects them, as well as duplicate
  registers and parameters that are not finite (`1/0`, `ln(0)`).
- The `U`/`u3` gate lost unitarity for very large angles (|λ| ≳ 10¹²),
  because `e^{i(φ+λ)}` was computed from the rounded sum `φ + λ`.
- `PauliSum` parsing accepted coefficients that overflow to infinity
  (`5e555 Z0`), which then printed as `inf` and could not be read back.
- Graph specifications such as `complete:204204` built graphs with up to
  10¹⁸ edges; specifications and the Python graph constructors are now
  limited to 2²⁰ nodes and 2²² edges (`hypercube:20` stays available), and
  `Graph(10**12)` raises `ValueError` instead of aborting the interpreter.
- Graphs read from JSON were not validated; inconsistent adjacency lists or
  invalid edges now fail to deserialise.
- `random_regular` bounds its work for parameters where the pairing method
  almost never succeeds (large `d`) instead of running for minutes.

## [1.1.0] — 2026-10-24

Delivers the scope of the 2025 technical note with validated
implementations, and the second version of that note, which withdraws the
results of the first.

### Added

- `photonics` module: permanents (Glynn's formula in Gray-code order,
  deterministic in parallel), Haar-random interferometers, beam splitters
  and `BosonSampler` with exact output probabilities for indistinguishable
  and distinguishable photons, full output distributions and exact
  sampling (Clifford & Clifford 2018). Python: `permanent`, `haar_unitary`,
  `beam_splitter`, `BosonSampler`.
- `GraphClassifier`: graph classification with trainable QAOA-type quantum
  layers and exact gradients (Python: `GraphClassifier`).
- `Graph::girth` and the Heawood graph (`heawood`); Python
  `Graph.girth`, `Graph.triangle_count`, `Graph.heawood`.
- `examples/technical_note.rs`, which computes every number of the
  technical note with its reference, and `benchmarks/cross_framework/`,
  which compares QSO with Qiskit Aer, Qulacs, Cirq, PennyLane Lightning and
  The Walrus; the manual *Benchmarks* workflow runs both.
- Technical note, version 2 ([docs/technical-note/](docs/technical-note/)).
- Validation against further published results: QAOA at p = 2 reaches
  0.7559 on the Heawood graph (Wurtz & Love 2021), the one-layer QAOA
  expectation matches the closed form of Wang et al. (2018), the
  Hong–Ou–Mandel dip, the n! enhancement of full bunching, and a χ² test
  of the exact boson sampler.
- `CITATION.cff`.

### Changed

- Faster statevector kernels, found by the cross-framework comparison:
  real 2×2 matrices (H, RY) use real arithmetic, diagonal matrices (RZ)
  and single-qubit phase gates scale contiguous halves, singly controlled X
  swaps contiguous runs, and new registers are written in parallel. At 22
  qubits H and RY are about twice as fast, RZ and CX about 1.4 times.
- `MaxCutQaoa` accepts depths up to `MAX_QAOA_DEPTH` = 10 000.

### Fixed

- `ContinuousWalk::mixing_time` returned the first time at which the
  time-averaged distribution came within ε of the limit. As documented
  (Aharonov et al. 2001) it now returns the time from which on the bound
  holds; the two differ when the average oscillates.
- `ContinuousWalk::mixing_time` accepted a NaN or negative ε and a
  non-positive or infinite `t_max`; a NaN bound counted as satisfied.

## [1.0.1] — 2026-10-24

Packaging fix; the code is identical to 1.0.0, whose release notes describe
the rewrite: <https://github.com/LashSesh/qso/blob/main/CHANGELOG.md#100--2026-10-24>.

### Fixed

- PyPI rejected the source distribution of 1.0.0 because its metadata named a
  `LICENSE` file at the archive root that the archive did not contain, so
  1.0.0 is on PyPI as wheels only. 1.0.1 ships the source distribution as
  well, for platforms without a prebuilt wheel.
- The release workflow now checks that every license file named in the
  source distribution's metadata is in the archive before anything is
  published.

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
