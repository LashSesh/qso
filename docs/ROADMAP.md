# QSO Roadmap (1.x)

Status: October 2026. Milestones 1.1 to 1.6 delivered — 1.3 combined the
milestones planned as "quantum walks" and "simulation performance".
Milestones 1.4 (noise at scale and error mitigation), 1.5 (algorithms and
physics) and 1.6 (interoperability) shipped together as release 1.4.0.
Milestone 1.7 (stabilizer simulation and graph states) is delivered and
unreleased; next is 1.8, error-correction experiments. Up to 1.3 milestone and release numbers coincided; from 1.4 on the
milestone numbers name units of work and releases follow Semantic
Versioning on their own (the next release after 1.3.0 is 1.4.0, whatever
it contains).

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
| Statevector speed | specialised kernels (1.1); gate fusion and cache blocking (1.3.0): 16-qubit layers 4.7× and 18–24-qubit layers 1.6–2.6× faster than in 1.2; batched evaluation | no explicit SIMD, no single precision; the adjoint backward pass is not blocked; worker threads sleep when idle, so short isolated simulations on virtual machines run cold — see [measurements](#measured-performance-against-established-simulators) |
| Boson sampling | exact probabilities and exact sampling (1.1) | no partial distinguishability, photon loss or Gaussian boson sampling |
| Noise and mitigation | device models (per-qubit T₁/T₂ and readout, per-gate durations, errors and over-rotations, JSON), density matrices up to 14 qubits, quantum trajectories beyond; readout mitigation, ZNE, PEC, Pauli twirling, classical shadows, standard errors on every estimate (1.4) | no noise learning (PEC needs the Pauli noise given), no leakage or non-Markovian noise, channels on at most two qubits, no virtual distillation or probabilistic error amplification |
| Quantum walks | continuous- and discrete-time walks, symmetry reduction to exact low-dimensional problems (Q₃₀ in 31 dimensions), perfect state transfer, walk circuits (1.3.0) | no time-dependent Hamiltonians, no open-system discrete-time walks, no pretty-good-transfer decision (only best fidelity in a window) |
| Physics and chemistry | fermion-to-qubit mappings, FCIDUMP molecules with full CI, UCCSD, k-UpCCGSD, ADAPT-VQE; product formulas with error bounds, qDRIFT; VQD, SSVQE, natural gradient, imaginary-time evolution; QUBO/QAOA mixers; amplitude estimation (1.5) | no integrals of its own (needs FCIDUMP files from a chemistry code), no unrestricted (UHF) integrals, no tight commutator bound for product formulas of order 4 and higher, no time-dependent Hamiltonians |
| Interoperability | OpenQASM 2 without `gate` definitions; multi-controlled gates exported by exact decomposition (1.3.0); edge lists, GraphML and duck-typed NetworkX import for graphs | no OpenQASM 3, no browser build; no bridges to other frameworks, by decision (see section 2) |
| Documentation | user guide, executed tutorials and Python reference on GitHub Pages (1.2) | no Rust-specific tutorials; no hosted documentation per released version |
| Engineering | CI on Linux (x86_64, arm64), macOS and Windows; Miri, fuzzing, API checks, weekly cross-validation and benchmark tracking; Trusted Publishing (1.2) | no coverage measurement; fuzzing only of the parsers and deserialisers |

### Measured performance against established simulators

Measured with [`benchmarks/cross_framework`](../benchmarks/cross_framework/)
on a 4-core cloud VM, every framework with default threading, five
interleaved rounds (QSO 1.1 in the technical note, Section 3.8; QSO 1.3.0
in [BENCHMARKS.md](BENCHMARKS.md#comparison-with-other-simulators-130)):

* **Correctness:** final states of random circuits agree with Qiskit Aer,
  Qulacs, Cirq and PennyLane Lightning to ≤ 3·10⁻¹⁵; gradients agree with
  PennyLane's and Qulacs' to ≤ 2·10⁻¹⁵; permanents agree with The Walrus to
  ≤ 8·10⁻¹⁴.
* **State vectors:** a first comparison (QSO 1.0.1) put QSO 1.2–1.7× behind
  Qulacs from 20 to 24 qubits and traced this to the H and CX kernels
  (2× slower per amplitude) and to single-threaded allocation. After the
  1.1 kernels Qulacs took 0.96–1.4 times as long as QSO on 20–24 qubits and
  was 2× faster on 16 qubits, where handing every gate to sleeping worker
  threads dominated. With cache blocking (1.3.0) Qulacs takes 1.5–2.6
  times as long as QSO on 20–24 qubits, Qiskit Aer 3.3–4.7 times and
  PennyLane Lightning 3.7–9 times; on 16 qubits Qulacs is still 1.3×
  faster in the median.
* **Thread wake-up:** rayon workers sleep as soon as they are idle; on the
  test VM waking them is slow, and a burst of short simulations needs tens
  of milliseconds to reach full speed (a cold 16-qubit layer takes 1.5 ms,
  a warm one 0.55 ms). Qulacs' OpenMP workers spin. The comparison warms
  every framework up for 0.25 s since 1.3.0.
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
* **Minor releases bundle finished milestones**, published by the release
  workflow after a dry run; patch releases fix bugs in between. Work that
  finishes together ships together (release 1.3.0 combined two planned
  milestones; release 1.4.0 carries milestones 1.4 to 1.6). Release
  numbers never skip a version.

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
| Thread wake-up | — | found while measuring 1.3.0: rayon workers sleep as soon as they are idle, and on virtual machines short isolated simulations (around 1 ms) run at close to single-thread speed until the workers stay awake; keeping them spinning briefly needs a custom thread pool, open |

Validation: planned (fused and blocked) simulation equals gate-by-gate
simulation to 1e-12 on random circuits over the whole gate set with
controls, including blocks of 2–32 amplitudes (also under Miri); batched
results equal single calls exactly; QAOA phase tables equal direct phases
to 1e-12. Measurements: [BENCHMARKS.md](BENCHMARKS.md).

### 1.4 — Noise at scale and error mitigation (delivered)

| Deliverable | Delivered | Not done, and why |
|---|---|---|
| Quantum trajectories | `TrajectorySimulator` and a trajectory device backend for any register the state vector holds, one seeded stream per trajectory, parallel over trajectories. Waiting-time method: unitary mixtures sampled up front, jumps found by galloping search and bisection, stretches without jumps run as fused, cache-blocked circuits — 200 trajectories take 1.2–1.3 times as long as 200 noiseless runs at 16 qubits | — |
| Device-like noise models | `DeviceModel` with per-qubit T₁/T₂ and readout errors, ordered gate rules (name, qubits, size) with durations, depolarising errors, over-rotations and further channels; ASAP scheduling with relaxation of waiting qubits; Pauli channels (correlated two-qubit noise), coherent rotations; JSON descriptions with a fuzz target | channels on three or more qubits: no device data needs them yet |
| Error mitigation | readout calibration (tensored and full) with exact per-shot mitigated weights; zero-noise extrapolation (global and random folding; linear, Richardson, polynomial and exponential fits with propagated errors); probabilistic error cancellation of Pauli noise with its overhead; Pauli twirling of cx, cy, cz and swap | learning the Pauli noise of a device (cycle benchmarking): QSO simulates devices whose noise it is given |
| Estimation | `Estimate` (standard error, confidence interval) for every stochastic result; uniform, weighted and adaptive (Neyman) shot allocation; classical shadows with median of means | — |

Validation:

* trajectory samples pass χ² tests against density matrices on 3, 6 and
  10 qubits, also with several jumps per trajectory;
* readout mitigation plus ZNE recovers a GHZ correlator within its error
  bars (4.00 ± 0.01 against a raw 3.38);
* PEC equals the noiseless value to 10⁻¹² when summed over all
  corrections, and sampled PEC lies within its error bars;
* the Pauli twirl of a channel equals the exact average over all 16
  frames;
* the single-snapshot variance of shadows is 3ᵏ − ⟨P⟩²;
* 95 % intervals cover the exact value in 91–99 % of repeated estimates;
* adaptive allocation stays unbiased for nearly deterministic groups.

An independent review of the new code found two parser-reachable panics,
a biased adaptive allocation (pilot and main shots were pooled) and
quadratic trajectory cost in the number of jumps. All were fixed before
the release.

### 1.5 — Algorithms and physics (delivered)

| Deliverable | Delivered | Not done, and why |
|---|---|---|
| Hamiltonian simulation | `Circuit::pauli_rotation`; product formulas of order 1, 2, 4, 6, …; `trotter_error_bound` with the commutator bounds of Childs et al. (2021) for orders 1 and 2 and the smaller of a stage-wise and a Taylor-remainder bound above; qDRIFT with Campbell's bound | a tight commutator bound for order ≥ 4 (the paper's Appendix J is heuristic in its constants); the bound QSO gives there is rigorous but loose |
| Fermions | `FermionOperator` with normal ordering; Jordan–Wigner, parity and Bravyi–Kitaev mappings as binary transforms; two-qubit reduction; `MolecularIntegrals` from FCIDUMP files with Hartree–Fock states and energies and full CI; UCCSD, k-UpCCGSD, ADAPT-VQE; FCIDUMP files of H₂, LiH and BeH₂ with PySCF reference energies | computing integrals (that needs a quantum-chemistry code; FCIDUMP is the exchange format, as OpenQASM is for circuits); unrestricted orbitals |
| Excited states and geometry | VQD, weighted SSVQE; exact state derivatives, quantum Fisher information, quantum natural gradient; McLachlan imaginary-time evolution (RK4) | — |
| Optimisation problems | `Qubo`/`Ising` with exact enumeration; QAOA for any diagonal cost with X, ring and complete XY mixers from Dicke states, and the warm-start mixer | a built-in continuous relaxation for warm starts: the user supplies `c ∈ [0, 1]ⁿ` |
| Amplitude estimation | canonical (exact outcome distribution), iterative (Chernoff–Hoeffding intervals) and maximum-likelihood variants | Clopper–Pearson intervals for iterative estimation |

Validation:

* product-formula errors stay below their bounds and fall as `r^{-order}`
  (orders 1, 2 and 4 on Heisenberg, Ising and random 4-qubit
  Hamiltonians); the qDRIFT channel converges as `1/N` below its bound;
* all four encodings give identical determinant-space spectra, equal to
  PySCF's full CI to 10⁻⁸ Ha, for H₂, LiH and BeH₂ in STO-3G; Hartree–Fock
  energies agree to 10⁻⁹ Ha;
* UCCSD and ADAPT-VQE reach the full-CI energy of LiH within 1.1·10⁻⁵ Ha;
* the quantum Fisher information equals −2 times the Hessian of the
  fidelity (finite differences); imaginary-time evolution follows the
  exact normalised evolution to 10⁻⁶;
* VQD and SSVQE reproduce exact spectra (all levels of H₂, three levels of
  an Ising chain);
* XY mixers keep all probability on feasible strings;
* the RMSE of maximum-likelihood amplitude estimation falls with slope
  −0.99 in the oracle calls (−0.49 for plain sampling); canonical
  estimates meet their error bound with probability ≥ 8/π²; iterative
  intervals cover the true value (1 miss in 200 at α = 0.05).

Two independent reviews (core and Python bindings) found no wrong results
on the validated paths. They found panics on overflowing angles and
malformed inputs, unbounded allocations (circuit size, fermion products,
full-CI spaces, enumeration, restart counts), an error bound that collapsed
to 0 when its commutators overflowed, a VQD default penalty that equalled
the largest possible gap instead of exceeding it, and SSVQE accepting
weights that are not strictly decreasing. All were fixed with regression
tests before the release.

### 1.6 — Interoperability and reach (delivered)

| Deliverable | Delivered | Not done, and why |
|---|---|---|
| OpenQASM | one parser for OpenQASM 2.0 and an OpenQASM 3 subset: `gate` definitions, `ctrl`/`negctrl`/`inv`/integer `pow` modifiers, `gphase`, `measure`, `reset`, conditions with blocks and `else`; `Program` with exact outcome distributions over measurement branches and seeded sampling; export to both versions; Python `Program`, `qso qasm` for programs | OpenQASM 3's classical types and arithmetic, loops, subroutines, calibrations and timing: they turn programs into a classical language, and the exact branch-following simulation would need an interpreter. They are rejected with an error naming the feature |
| Browser | the core builds for `wasm32`; `metatron_qso_wasm` (a module without imports, JSON interface); npm package `metatron-qso` with types; playground for OpenQASM programs, walks, search and symmetry data at `/playground/` of the documentation site | multi-threading in the browser (needs cross-origin isolation and nightly Rust); the first npm publish is manual, because npm attaches trusted publishers only to existing packages |

Bridges to Qiskit, PennyLane and Cirq are not planned (decision of October
2026): they would add test dependencies and follow those projects' release
cycles. OpenQASM 2/3 is the exchange format.

Validation:

* the example programs of the OpenQASM project give their known results
  (adder, phase estimation, teleportation, W state, repetition code,
  QFT, randomised benchmarking); the six OpenQASM 3 examples in the
  subset match their OpenQASM 2 versions, the other fifteen are
  rejected; all round-trip through both exports;
* a fuzz target checks that the exports of random programs reproduce
  their outcome distributions;
* the wasm build passes the same test vectors as the native build: the
  library's unit, validation, OpenQASM and robustness tests run as
  `wasm32-wasip1` under Node.js in CI, and the npm package's tests check
  the OpenQASM examples and graph results.

An independent review confirmed the phase semantics against the
specification's `stdgates.inc` and found that 3000 random programs
survive both exports. It found inputs that crashed or hung the reader (a
50,000-term sum overflowed the stack; doubling definitions of `id` and
large `ctrl`/`pow` products escaped the operation budget), constructs that
were misread instead of rejected (`bool(c) == true`, slices with negative
ends, parameters named `euler`, integers above 2⁵³, odd version numbers),
register names that made exports unreadable, and records above 32 bits
truncated in WebAssembly. All were fixed with regression tests before the
release.

### 1.7 — Stabilizer simulation and graph states (delivered, unreleased)

| Deliverable | Delivered | Not done, and why |
|---|---|---|
| Tableau simulator | Aaronson–Gottesman tableaux for up to 16 384 qubits, stored by qubit (a gate is O(n/32) words); Clifford gates including rotations by multiples of π/2; Pauli measurements, reset, expectation values, sampling from the affine outcome space | measurements in column form without transposition (they cost O(n²), a second at the limit); noisy Clifford circuits at scale come with the frame sampler of 1.8 |
| Clifford programs | OpenQASM programs of Clifford gates, mid-circuit measurement and feed-forward included; programs that end in measurements are sampled at a cost per shot independent of n; `qso qasm`, Python (`method="auto"`) and the WebAssembly module choose the tableau for Clifford programs too large for a state vector | — |
| Graph states | graph states of any unweighted graph, local complementation as a local Clifford, entanglement entropy from GF(2) ranks | a test of local-Clifford equivalence (Bouchet's algorithm) |

Validation:

* 200 random Clifford circuits on 1–8 qubits match the state-vector
  simulator (states, Pauli expectation values, collapse, supports);
* a GHZ state on 10 000 qubits has its known correlations;
* graph-state entanglement equals the state-vector entropy and the
  cut-rank (Q₁₀: 512, C₁₀₀₀: 2);
* local complementation equals the local Clifford of Hein, Eisert and
  Briegel (2004);
* the graph state of Metatron's Cube is a GHZ state up to local Cliffords.

An independent review compared the tableau with the state vector in 4000
further random trials on registers of 1 to 257 qubits (word and block
boundaries included), 600 random programs and 300 random graphs, and
found no wrong result on these paths. It found:

* a panic for parametrized circuits without parameters;
* angles above about 10⁸ rad accepted as Clifford, because the
  tolerance was relative;
* unbounded time in program sampling and in sampling all qubits;
* cost figures in the docs that ignored the transposition of the tableau.

All were fixed with regression tests. Programs that end in measurements
are now sampled from their record bits alone.

### 1.8 — Error-correction experiments (M–L)

| Deliverable | Detail |
|---|---|
| Detector circuits | Clifford circuits with Pauli noise channels, measurement records, detectors and logical observables |
| Pauli-frame sampler | 64 shots per machine word, for millions of shots of noisy Clifford circuits |
| Codes | repetition code and rotated surface code memory experiments with circuit-level noise |
| Decoding | detector error models from circuits, matching graphs, a union-find decoder |

Validation: frame sampling against the tableau simulator; logical error
rates against exact enumeration for distance 3; error rates that fall with
distance below threshold.

### Research track (later)

Matrix-product-state simulation for low-entanglement circuits and a GPU
backend through `wgpu`. Each is a project of its own; they start only with
a clear use case, and a GPU backend only after the CPU core is close to
memory bandwidth.

## 5. Order and reasoning

1. **1.2 first:** small, made every later release safer, and documentation
   is what new users see first.
2. **1.3: walks and performance together.** Walks are QSO's distinctive
   contribution and build on the refinement, Lanczos and walk code; the
   performance work was independent and finished at the same time.
3. **1.4: noise before algorithms.** Device models, trajectories and
   mitigation reuse the kernels and execution plans directly and make
   every later algorithm testable under realistic noise.
4. **1.5: algorithms and physics**, on top of the noise and estimation
   tools.
5. **1.6: interoperability**, without framework bridges. Milestones 1.4
   to 1.6 ship as one release, 1.4.0, to keep releases (and CI runs) few.
6. **1.7 and 1.8: stabilizers before error correction.** Error-correction
   experiments need the stabilizer simulator; graph states connect it to
   QSO's graphs.

## 6. Measures of success

* Correctness: the cross-validation job is green on every scheduled run;
  every feature in the CHANGELOG names its reference test.
* Speed: throughput relative to Qulacs and Qiskit Aer on the same machine,
  tracked per release.
* Usability: docs site with executed tutorials; every user-facing Rust
  feature is available from Python.
* Releases: one per milestone, or fewer when milestones finish together;
  published without stored tokens.

## 7. Decisions

Taken (October 2026):

* Walks and performance first, shipped together as 1.3.
* No Qiskit, PennyLane or Cirq bridges.
* Noise and mitigation (1.4) before algorithms and physics (1.5).
* Milestones 1.4 to 1.6 ship as release 1.4.0; release numbers follow
  Semantic Versioning and are no longer tied to milestone numbers.
* No integral code: molecules come in as FCIDUMP files, written by any
  quantum-chemistry program (the repository's files by PySCF).
* From the research track, stabilizer simulation comes first (1.7, 1.8):
  it needs no dependency, it has exact references (the state-vector
  simulator, closed forms, enumeration), and graph states and
  error-correcting codes are graph-structured quantum computing.
  Matrix-product states and a GPU backend wait.
* The browser playground stays, because it costs little to maintain: it
  is plain HTML and JavaScript without dependencies or a build step, it
  uses only the npm package's JSON interface, which the package's tests
  cover, and the documentation build publishes it. If it ever needs
  more than that, it is removed from the site rather than given a
  framework.
