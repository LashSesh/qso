# QSO Roadmap (1.x)

Status: October 2026. 1.1, 1.2 and 1.3 delivered — 1.3 combined the
milestones planned as "quantum walks" and "simulation performance". Noise
at scale (1.4) is next. Milestone numbers are the release numbers; the
later milestones moved up by one when the two were merged.

QSO 1.0 delivers what it documents: every number in the README is reproduced
by a test against an independent reference, the packages are on crates.io
and PyPI, and releases are automated. This roadmap plans the next stage: from
a correct, small simulator to the reference toolkit for its niche.

Release 1.1 delivered what the 2025 technical note had promised but QSO 0.1
never implemented — exact boson sampling, multi-target Grover search,
machine learning on graphs and a real comparison with other frameworks —
together with version 2 of the note ([docs/technical-note/](technical-note/)),
which withdraws the results of version 1.

## 1. Where QSO stands

**Strengths**

* Correctness culture: closed-form and literature references for every
  algorithm, property tests, robustness tests, bit-reproducible parallelism.
* Pure Rust, no C/C++ dependencies: builds anywhere Rust builds, abi3 wheels
  for all major platforms, small binaries.
* Adjoint differentiation: a full gradient costs about three circuit runs
  regardless of the parameter count.
* Graph analysis that other simulators lack: exact automorphism groups,
  spectral walk analysis, Childs–Goldstone search with optimised rate.

**Gaps** (found by reviewing the code and measuring; details in each milestone)

| Area | Now | Gap |
|---|---|---|
| Statevector speed | specialised kernels (1.1); gate fusion and cache blocking (1.3.0): 16-qubit layers 3.8× and 18–24-qubit layers 1.4–2× faster than in 1.2; batched evaluation | no explicit SIMD, no single precision; the adjoint backward pass is not blocked — see [measurements](#measured-performance-against-established-simulators) |
| Boson sampling | exact probabilities and exact sampling (1.1) | no partial distinguishability, photon loss or Gaussian boson sampling |
| Noise | density matrices up to 14 qubits; one noise model for all gates and qubits | no trajectories for larger registers, no per-qubit/per-gate models, no error mitigation |
| Quantum walks | continuous- and discrete-time walks, symmetry reduction to exact low-dimensional problems (Q₃₀ in 31 dimensions), perfect state transfer, walk circuits (1.3.0) | no time-dependent Hamiltonians, no open-system discrete-time walks, no pretty-good-transfer decision (only best fidelity in a window) |
| Physics and chemistry | TFIM, XXZ, MaxCut, H₂ only | no fermion-to-qubit mappings, no product-formula (Trotter) circuits, no excited states |
| Interoperability | OpenQASM 2 without `gate` definitions; multi-controlled gates exported by exact decomposition (1.3.0); edge lists, GraphML and duck-typed NetworkX import for graphs | no OpenQASM 3, no browser build; no bridges to other frameworks, by decision (see section 2) |
| Documentation | user guide, executed tutorials and Python reference on GitHub Pages (1.2) | no Rust-specific tutorials; no hosted documentation per released version |
| Engineering | CI on Linux (x86_64, arm64), macOS and Windows; Miri, fuzzing, API checks, weekly cross-validation and benchmark tracking; Trusted Publishing (1.2) | no coverage measurement; fuzzing only of the parsers and deserialisers |

### Measured performance against established simulators

Measured for the technical note (Section 3.8) with
[`benchmarks/cross_framework`](../benchmarks/cross_framework/) on a 4-core
cloud VM, every framework with default threading, five interleaved rounds:

* **Correctness:** final states of random circuits agree with Qiskit Aer,
  Qulacs, Cirq and PennyLane Lightning to ≤ 3·10⁻¹⁵; gradients agree with
  PennyLane's and Qulacs' to ≤ 2·10⁻¹⁵; permanents agree with The Walrus to
  3·10⁻¹⁴.
* **State vectors:** a first comparison (QSO 1.0.1) put QSO 1.2–1.7× behind
  Qulacs from 20 to 24 qubits and traced this to the H and CX kernels
  (2× slower per amplitude) and to single-threaded allocation. After the
  1.1 kernels Qulacs takes 0.96–1.4 times as long as QSO on 20–24 qubits,
  Qiskit Aer and PennyLane Lightning 1.7–3.3 times as long. On 16 qubits
  Qulacs was 2× faster: a gate takes tens of microseconds there and the
  cost of handing work to sleeping worker threads dominated. Release 1.3.0
  removed that cost (16-qubit layers 3.8× faster, see
  [BENCHMARKS.md](BENCHMARKS.md)); the comparison with the other
  frameworks will be repeated before figures are quoted.
* **Gradients:** the adjoint gradient of a 100-parameter, 10-qubit ansatz
  takes about 0.5 ms, against 2–5 ms with Qulacs' `backprop` and about
  20 ms with PennyLane Lightning's adjoint method.
* **Gate fusion** did not help Qiskit Aer on four threads but gave it
  1.6–1.7× single-threaded, so fusion matters most for low thread counts.

## 2. Direction

QSO aims to be **the validated, embeddable simulator for graph-structured
quantum computing**:

1. **Quantum walks and graph algorithms, best in class.** This is QSO's
   identity (Metatron's Cube is a graph problem) and the place where it can
   offer what general-purpose simulators do not: exact, symmetry-aware
   analysis of continuous- and discrete-time walks on very large graphs,
   compiled to circuits when needed. The closest project, Hiperwalk
   ([arXiv:2406.08186](https://arxiv.org/abs/2406.08186)), describes itself
   as high-performance numerical simulation of coined and continuous-time
   walks (Python front end, C core, optional GPU). QSO's angle is different:
   exact, symmetry-aware analysis validated against theory, and compilation
   of walks to circuits.
2. **A fast, validated gate-model core** that holds its own against
   established single-machine simulators, because every walk, VQE or QAOA
   result rests on it.

Constant across all milestones: pure Rust, deterministic, and **no claim
without a test against an independent reference**.

**Non-goals for 1.x:** hardware vendor integrations beyond OpenQASM export;
bridges to Qiskit, PennyLane or Cirq (QSO stays free of framework
dependencies; OpenQASM is the exchange format); GPU or multi-node execution
before the CPU kernels are close to memory bandwidth; a general
tensor-network library; any claim about Metatron's Cube that is not graph
theory or physics.

## 3. Rules for the 1.x series

* **Semantic Versioning.** 1.x releases add, never break. `cargo-semver-checks`
  and a Python API diff run in CI. Deprecations come at least one minor
  release before removal; removals wait for 2.0.
* **MSRV.** Raised only in a minor release, never beyond the stable release
  of six months earlier, and noted in the CHANGELOG.
* **Validation.** Every milestone below lists its validation targets. A
  feature without one is not done.
* **Performance claims** come only from `benchmark_report` with the machine
  stated, as today.
* **One minor release per milestone**, published by the release workflow
  after a dry run; patch releases fix bugs in between. Work that finishes
  together ships together (1.3 combined two planned milestones).

## 4. Milestones

Sizes: S ≈ days, M ≈ 1–2 weeks, L ≈ 3–6 weeks of focused work.

### 1.1 — Scope of the 2025 technical note (delivered)

Boson sampling with exact sampling, the graph classifier, multi-target
Grover benchmarks, the cross-framework comparison, faster H/RY/RZ/CX
kernels, the `technical_note` benchmark program and version 2 of the note.

### 1.2 — Foundations: trust, supply chain, documentation (delivered)

Cheap, protects the 1.x promise, and gives users their first impression.
Everything below is in place. Fuzzing and the accompanying review found
six robustness bugs, all fixed with regression tests
([CHANGELOG](../CHANGELOG.md)). Miri found
nothing in the `unsafe` kernels. Density matrices of noisy circuits agree
with Qiskit Aer to 3.5·10⁻¹⁵.

| Deliverable | Detail |
|---|---|
| Trusted Publishing | PyPI and crates.io via GitHub OIDC; delete the `CRATES_IO`/`PYPI` secrets; crates.io set to trusted-only; publishing jobs in a `release` environment with required approval |
| Unsafe-code checks | Miri job for `kernels.rs` and `parallel.rs` on small registers |
| Fuzzing | `cargo-fuzz` targets for the OpenQASM, Pauli-string and graph-spec parsers and for `Circuit`/`StateVector` deserialisation; short run per PR, long run weekly |
| API stability | `cargo-semver-checks`; `stubtest` comparing `_core.pyi` with the compiled module |
| Cross-validation | run `benchmarks/cross_framework` on a schedule and extend it to noisy circuits (density matrices against Qiskit Aer) |
| Benchmark tracking | weekly `benchmark_report` run, results stored, regressions > 25 % flagged |
| Platforms | musllinux and Windows arm64 wheels; native aarch64 tests; Python 3.14 in the matrix; Node-24 action versions |
| Documentation | user guide (mdBook) and Python API reference on GitHub Pages; tutorials as notebooks executed in CI so they cannot go stale: circuits and gradients, VQE and QAOA, noise, quantum walks on Metatron's Cube |

**Done when** every job is green on `main`, the docs site is live, and a
release has been published without stored tokens.

### 1.3 — Quantum walks exact at scale, and simulation performance (delivered)

#### Quantum walks

| Deliverable | Delivered |
|---|---|
| Symmetry reduction | `Quotient` (coarsest equitable partition by colour refinement, checked user partitions), `QuotientWalk`, `QuotientSearch`; walks, search and limiting distributions in as many dimensions as cells |
| Implicit graph families | hypercube, Hamming, Johnson, complete and complete multipartite quotients without building the graph |
| Beyond 2048 nodes | limiting and averaged distributions and mixing times through the start node's cyclic (Krylov) subspace |
| Discrete-time walks | `CoinedWalk` (Grover, Fourier, Hadamard, custom coins; SKW search), `LineWalk`, `SzegedyWalk` with quantum hitting times |
| State transfer | `StateTransfer`: perfect-transfer decision and time (strong cospectrality and the eigenvalue condition, confirmed numerically), best fidelity in a window |
| Walks as circuits | exact circuits for circulant graphs (QFT) and Cayley graphs of Z₂ⁿ, first- and second-order product formulas for any graph, exact diagonal unitaries; OpenQASM export with ancilla-free decomposition of multi-controlled gates |
| Graph library | Johnson, Kneser, Hamming, rook, Paley, Shrikhande, circulant and Cayley graphs; Cartesian products; strongly regular parameters; canonical labelling and isomorphism; edge-list and GraphML I/O; NetworkX import by duck typing |
| Showcase | `metatron_atlas` example and tutorial 05 |

All validation targets of the plan are tests: reduced and full walks agree
to 1e-12; the Q₃₀ antipodal amplitude |sin γt|³⁰ and perfect transfer at
γt = π/2; no perfect transfer on Pₙ for n ≥ 4; SKW search on Q₁₀ (numpy
cross-check); σ/t → √(1 − 1/√2) for the Hadamard walk; Szegedy hitting
times n/4 on cycles and √-scaling against the classical hitting times;
circulant walk circuits equal to `exp(−iHt)` to 1e-10; canonical forms
invariant under relabelling and agreeing with brute force on 300 random
graphs; Shrikhande vs 4×4 rook's graph separated.

#### Simulation performance

| Deliverable | Delivered | Not done, and why |
|---|---|---|
| Small registers | registers of up to 2¹⁴ amplitudes run gate by gate on one thread; larger ones are parallel over cache blocks instead of per gate, so 15–18 qubits no longer pay thread wake-ups per gate | — |
| Gate fusion | runs of single-qubit gates on qubits outside the cache block fused into one matrix | fusion into dense k-qubit blocks: inside a cache block, the specialised real, diagonal and permutation kernels were faster than fused complex matrices |
| Cache blocking | gates on the 14 lowest qubits and diagonal gates anywhere applied block by block (blocks of 2¹⁴ amplitudes, parallel over blocks); QAOA cost and mixer layers likewise, with phase tables | gathering high qubits into blocks (as in qHiPSTER-style blocking) was implemented and measured; with a greedy scheduler it was not faster on the 4-core test machine and was dropped |
| Batched evaluation | `expectation_batch` and `expectation_and_gradient_batch`, parallel over parameter vectors below 16 qubits, in Rust and Python | — |
| SIMD kernels | — | deferred: the autovectorised kernels are memory-bound from 20 qubits; explicit SIMD needs per-architecture code and testing on hardware CI does not provide |
| Single precision | — | deferred: halving the memory conflicts with the 1e-12 validation targets; would need its own tolerance policy |

Validation: planned (fused and blocked) simulation equals gate-by-gate
simulation to 1e-12 on random circuits over the whole gate set with
controls, including blocks of 2–32 amplitudes (also under Miri); batched
results equal single calls exactly; QAOA phase tables equal direct phases
to 1e-12. Measurements: [BENCHMARKS.md](BENCHMARKS.md).

### 1.4 — Noise at scale and error mitigation (L, next)

| Deliverable | Detail |
|---|---|
| Quantum trajectories | Monte-Carlo wavefunction simulation for any register the statevector supports; one seeded stream per trajectory, parallel over trajectories |
| Device-like noise models | per-qubit and per-gate channels, gate durations with T₁/T₂ idling, coherent over-rotations, correlated two-qubit Pauli noise; JSON device description |
| Error mitigation | readout calibration (tensored and full), zero-noise extrapolation (unitary folding; Richardson and exponential fits), probabilistic error cancellation for Pauli noise with reported sampling overhead, Pauli twirling |
| Estimation | classical shadows; variance-optimal shot allocation across measurement groups; confidence intervals in all estimates |

Validation: trajectories match density-matrix results within statistical
error (χ² test) up to 10 qubits; mitigated estimates converge to the
noiseless value within their error bars; PEC is unbiased for known Pauli
channels; shadow estimates satisfy their variance bounds.

### 1.5 — Algorithms and physics (L)

| Deliverable | Detail |
|---|---|
| Hamiltonian simulation | 1st/2nd/4th-order Trotter–Suzuki circuits with commutator error bounds, qDRIFT |
| Fermions | fermionic operators; Jordan–Wigner, parity (with two-qubit reduction) and Bravyi–Kitaev mappings; molecular Hamiltonians from FCIDUMP files; Hartree–Fock states; UCCSD and k-UpCCGSD ansätze; ADAPT-VQE |
| Excited states and geometry | VQD, SSVQE; quantum Fisher information and natural gradient; variational imaginary-time evolution |
| Optimisation problems | QUBO/Ising input; constrained QAOA with XY mixers; warm-start QAOA |
| Amplitude estimation | canonical, iterative and maximum-likelihood variants |

Validation: Trotter errors stay below their bounds and converge with the
order of the formula; all three fermion mappings give identical spectra,
equal to full CI, for H₂, LiH and BeH₂ in STO-3G; QFI agrees with finite
differences; VQD energies match exact diagonalisation; amplitude-estimation
error scales as 1/M.

### 1.6 — Interoperability and reach (M)

| Deliverable | Detail |
|---|---|
| OpenQASM | `gate` definitions in QASM 2; an OpenQASM 3 subset (gate definitions, parameters, `measure`, `reset`, simple classical conditions) |
| Browser | `wasm32` build and npm package; interactive Metatron playground (walks, search, symmetry) on GitHub Pages |

Bridges to Qiskit, PennyLane and Cirq are not planned (decision of October
2026): they would add test dependencies and follow those projects' release
cycles. OpenQASM 2/3 is the exchange format.

Validation: QASM round trips on the official example files; the wasm build
passes the same test vectors as the native build.

### Research track (decide after 1.6)

Stabilizer simulation (Clifford tableaux for thousands of qubits, as the base
for error-correction experiments), matrix-product-state simulation for
low-entanglement circuits, and a GPU backend through `wgpu`. Each is a
project of its own; they start only with a clear use case and after the CPU
core is close to memory bandwidth.

## 5. Order and reasoning

1. **1.2 first:** small, made every later release safer, and documentation
   is what new users see first.
2. **1.3: walks and performance together.** Walks are QSO's distinctive
   contribution and build on the refinement, Lanczos and walk code; the
   performance work was independent and finished at the same time.
3. **1.4 (noise) and 1.5 (algorithms)** enter crowded fields and can swap
   places depending on whether noise or chemistry users show up first.
4. **1.6 (interoperability)** without framework bridges.

## 6. Measures of success

* Correctness: the cross-validation job is green on every scheduled run;
  every feature in the CHANGELOG names its reference test.
* Speed: throughput relative to Qulacs and Qiskit Aer on the same machine,
  tracked per release.
* Usability: docs site with executed tutorials; every user-facing Rust
  feature is available from Python.
* Releases: one per milestone, published without stored tokens.

## 7. Decisions

Taken (October 2026):

* Walks and performance first, shipped together as 1.3.
* No Qiskit, PennyLane or Cirq bridges.

Open:

* Whether the browser playground is worth its maintenance cost.
* Whether noise (1.4) or algorithms and physics (1.5) come first.
