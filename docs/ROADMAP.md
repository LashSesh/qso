# QSO Roadmap (1.x)

Status: October 2026. Milestones 1.1 to 1.13 delivered — 1.3 combined the
milestones planned as "quantum walks" and "simulation performance".
Milestones 1.4 (noise at scale and error mitigation), 1.5 (algorithms and
physics) and 1.6 (interoperability) shipped together as release 1.4.0.
Milestones 1.7 (stabilizer simulation and graph states) and 1.8
(error-correction experiments) shipped together as release 1.5.0 (1.5.1
fixed its npm publication). Milestone 1.9 (matrix-product states) and the
second stage — 1.10 (exact decoding and Stim circuits), 1.11 (the
state-vector core at memory bandwidth), 1.12 (noise learning) and 1.13
(walks beyond static Hamiltonians) — ship together as release 1.6.0;
finished milestones are released together, not one by one. The third
stage (1.14 to 1.17) is delivered and not yet released; of the fourth
stage (1.18 to 1.21), 1.18 to 1.20 are delivered and 1.21 is in progress. Up to 1.3
milestone and release numbers coincided; from 1.4 on the milestone
numbers name units of work and releases follow Semantic Versioning on
their own (the next release after 1.3.0 is 1.4.0, whatever
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
| Statevector speed | specialised kernels (1.1); gate fusion and cache blocking (1.3.0): 16-qubit layers 4.7× and 18–24-qubit layers 1.6–2.6× faster than in 1.2; batched evaluation; AVX and NEON kernels close to memory bandwidth and a blocked adjoint pass (1.11); single precision, one more qubit and about twice the speed on large registers (1.20) | worker threads sleep when idle, so short isolated simulations on virtual machines run cold — see [measurements](#measured-performance-against-established-simulators) |
| Boson sampling | exact probabilities and exact sampling (1.1); partial distinguishability, loss and Gaussian boson sampling with hafnians and torontonians (1.14) | mixed Gaussian states with many photons spread over many modes are impractical (10¹² operations at 20 photons in 10 modes) |
| Noise and mitigation | device models (per-qubit T₁/T₂ and readout, per-gate durations, errors and over-rotations, JSON), density matrices up to 14 qubits, quantum trajectories beyond; readout mitigation, ZNE, PEC, Pauli twirling, classical shadows, standard errors on every estimate (1.4) | noise learning for layers of two-qubit Clifford gates only, on density matrices (1.12); no leakage or non-Markovian noise, channels on at most two qubits; virtual distillation (1.17) mitigates only its measured copies' preparation |
| Quantum walks | continuous- and discrete-time walks, symmetry reduction to exact low-dimensional problems (Q₃₀ in 31 dimensions), perfect state transfer, walk circuits (1.3.0); driven and decoherent walks (1.13); exact pretty-good-transfer decisions on paths, cycles and abelian Cayley graphs (1.18) | time-dependent Hamiltonians and decoherent coined walks without symmetry reduction; pretty good transfer on other graphs |
| Physics and chemistry | fermion-to-qubit mappings, FCIDUMP molecules with full CI, UCCSD, k-UpCCGSD, ADAPT-VQE; product formulas with error bounds, qDRIFT; VQD, SSVQE, natural gradient, imaginary-time evolution; QUBO/QAOA mixers; amplitude estimation (1.5) | Gaussian integrals (STO-3G, 6-31G, H to Ne), RHF and UHF, frozen-core active spaces and matrix-free full CI (1.15); no d functions (no larger bases, nothing beyond neon), no restricted open-shell Hamiltonians, no tight commutator bound for product formulas of order 4 and higher |
| Interoperability | OpenQASM 2 with `gate` definitions and an OpenQASM 3 subset (1.6); a WASM build, npm package and browser playground (1.6); Stim's circuit format, written (1.8) and read (1.10); edge lists, GraphML and duck-typed NetworkX import for graphs | no bridges to other frameworks, by decision (see section 2) |
| Error correction | stabilizer tableaux up to 16 384 qubits (1.7); detector circuits, frame sampling, error models and union-find decoding of repetition and surface codes (1.8); exact matching decoding and X-basis memory (1.10); sparse matching at 1.1–2.3 times PyMatching's time (1.16); BP+OSD with colour and bivariate bicycle codes (1.19) | no correlated decoding; syndrome circuits of CSS codes are not tuned against hook errors |
| Large registers | matrix-product states up to 65 536 qubits with a fidelity bound (1.9) | no MPS in the browser build |
| Documentation | user guide, executed tutorials and Python reference on GitHub Pages (1.2) | no Rust-specific tutorials; no hosted documentation per released version |
| Engineering | CI on Linux (x86_64, arm64), macOS and Windows; Miri, fuzzing, API checks, weekly cross-validation and benchmark tracking; Trusted Publishing (1.2) | no coverage measurement; fuzzing only of the parsers and deserialisers; the five simulators (state vector, density matrix, trajectories, MPS, tableau) are compared on fixed cases, not on random circuits |

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

### 1.7 — Stabilizer simulation and graph states (delivered, release 1.5.0)

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

### 1.8 — Error-correction experiments (delivered, release 1.5.0)

| Deliverable | Delivered | Left for later |
|---|---|---|
| Detector circuits | Clifford gates, X/Y/Z errors, one- and two-qubit depolarizing noise, noisy measurements, resets, detectors and up to 64 observables; any Clifford `Circuit` can be appended; export to Stim's text format | an importer for Stim circuits; leakage and erasure |
| Pauli-frame sampler | 64 shots per machine word, noise by geometric skipping, a determinism check of every detector and observable | multi-threaded sampling |
| Codes | repetition and rotated surface-code memory experiments with circuit-level noise (`CircuitNoise`) | X-basis memory, lattice surgery, colour codes |
| Decoding | detector error models (exact depolarizing decomposition, merged effects, graphlike decomposition), matching graphs, a weighted union-find decoder, logical error rates in bounded memory | minimum-weight perfect matching; correlated decoding |

Validation:

* frame sampling agrees with shot-by-shot tableau simulation, with
  explicitly sampled errors, on random noisy Clifford circuits and on
  memory experiments;
* the error model predicts every detector's rate;
* on the repetition code, the decoder fails on exactly the error patterns
  with more than d/2 flips (all 2ᵈ patterns, d = 3 … 9);
* every single fault (surface code d = 3 and 5, repetition code d = 5)
  and every pair of faults (surface code d = 5) is corrected;
* below threshold (p = 0.001), surface-code logical error rates fall with
  the distance, and above it (p = 0.01) they rise;
* the scheduled cross-validation compares the error models with Stim's
  term by term and the detection-event rates with Stim's sampler, and
  reports PyMatching's logical error rates beside QSO's.

Profiling during the milestone made error models of large circuits about
13 times faster: the error model now propagates up to 4096 errors per
pass, walks the operations with a cursor instead of hash lookups, and
reads signatures from set bits.

An independent review compared the error models with Stim on 1495 random
circuits and tested the sampler's joint outcome distribution on about 300
random circuits, and found no wrong result there. It found:

* union-find with weights rounded to integer steps misdecoded some
  syndromes of (d − 1)/2 errors, and its logical error rate at d = 7 was
  3.8 times PyMatching's; growth is now event-driven with exact weights,
  and on the cross-validation circuits the rate is at most 6% above
  PyMatching's;
* unbounded memory in `logical_error_rate` for circuits with many
  detectors, and in error models whose errors flip many detectors before
  merging, and unbounded time for error models of long circuits;
* error models of circuits with random detectors returned silently;
* `to_stim` wrote tiny probabilities that Stim cannot read;
* generators that allocated gigabytes before reaching the operation
  limit.

### 1.9 — Matrix-product states (delivered, release 1.6.0)

Use case: circuits with little entanglement on far more qubits than a
state vector holds — Trotterized dynamics of spin chains, shallow QAOA on
rings and other sparse graphs, walk and Hamiltonian-simulation circuits —
with a stated bound on the truncation error instead of a silent
approximation.

| Deliverable | Delivered | Left for later |
|---|---|---|
| MPS simulator | states of up to 65 536 qubits in mixed canonical form, with a bond-dimension limit and a truncation cutoff; any gate of the gate set and any unitary on up to 8 qubits, controls included, routed by swaps to adjacent sites; consecutive gates on the same qubits fused | faster splits at large bond dimension (randomized range finders); folding rings so that the closing edge stays local; time-evolving block decimation for Hamiltonians directly |
| Error accounting | the discarded weight of every split, measured exactly, and a rigorous lower bound on the fidelity with the exactly prepared state | — |
| Readout | Pauli strings, `PauliSum`s and term lists; exact sampling of all qubits; amplitudes; measurement and reset; entanglement entropy at every bond; conversion to a state vector up to 24 qubits | — |
| Interfaces | `metatron_qso::mps` in Rust, `MatrixProductState` and `Program.sample(method="mps")` in Python, `qso qasm --mps --max-bond` | the WASM package and the browser playground |

Splits use the Hermitian eigendecomposition of a Gram matrix and a
Householder QR instead of nalgebra's complex SVD, which reconstructs
rank-deficient matrices with errors of several percent.

Validation:

* 120 random circuits on 1 to 12 qubits, with every gate kind and random
  controlled unitaries, against the state-vector simulator: amplitudes,
  expectation values and bond entropies agree, and with truncation the
  true fidelity is never below the reported bound;
* a Trotterized XX chain of 100 qubits against the exact free-fermion
  evolution of the same circuit: ⟨Z_j⟩ and ⟨Z_i Z_j⟩ (Wick's theorem) to
  10⁻⁸;
* QAOA on a ring of 100 nodes with angles optimised on a ring of 10:
  (2p + 1)/(2p + 2) for p = 1, 2, 3 to 10⁻⁶;
* GHZ states on 2000 qubits with bond dimension 2;
* sampling, measurement and reset follow the Born rule.

An independent review checked 600 random circuits with every gate kind
against the state vector and 4200 truncated runs against their bounds,
and found no wrong amplitude and no violated bound there. It found:

* programs that measured after a truncation in some shots and not in
  others reported the bound of the others;
* the discarded weight, computed as 1 − ‖F‖²/‖M‖², charged every exact
  split about 10⁻⁸ in distance, so long exact circuits reported bounds
  below 1 (0.99998 for a star of CNOTs on 2000 qubits); it is now
  computed from the residual, and the distance without cancellation;
* angles that overflowed after scaling (θ · 10³⁰⁰) produced a NaN state
  with a bound of 1;
* sampling memory grew with shots × qubits beyond any limit, and the
  work limit of programs ignored the bodies of `if` statements;
* a gate's tensor was checked against the memory limit only after it was
  built, and a failing split left the state half written.

## 4b. Second stage (1.10 to 1.13)

The first stage made every area QSO covers correct and validated. The
second closes the gaps that the first stage measured but left open,
in the order of what they block.

### 1.10 — Exact decoding and Stim circuits (delivered, release 1.6.0)

Use case: logical error rates that are those of minimum-weight matching,
not an approximation of them, on circuits written by Stim or by QSO.

| Deliverable | Delivered | Left for later |
|---|---|---|
| Matching decoder | `MatchingDecoder`: Dijkstra distances (cached per detector up to 32 MiB, bounded searches beyond), the boundary through savings b_i + b_j − d_ij, Edmonds' blossom algorithm on integer weights; `Decoder` trait and `logical_error_rate_with`, Python and `qso qec --decoder matching` | correlated (hyperedge) decoding; a sparse-blossom implementation for very large shots (PyMatching is several hundred times faster at d = 17) |
| Stim circuits | `DetectorCircuit::from_stim`: Clifford gates, measurements and resets in three bases, X/Y/Z and depolarizing noise, detectors, observables, REPEAT blocks; unsupported instructions named in the error | PAULI_CHANNEL, correlated and heralded errors, MPP, feedback |
| Codes | X-basis memory of the rotated surface code (`MemoryBasis::X`) | lattice surgery, colour codes |

Validation:

* the blossom algorithm against dynamic programming on 4000 random
  graphs with frequent ties;
* corrections against enumeration of every error set on 300 random
  error models, for every syndrome;
* every single fault, every pair at d = 5 (both bases) and 20 000
  random triples at d = 7 corrected;
* predictions against PyMatching on the same error model and syndromes:
  one of 800 000 differed in the cross-validation;
* Stim-generated repetition and surface-code circuits reproduce Stim's
  error models (unit tests with fixtures; the cross-validation at d = 5);
  exported circuits read back to the same text; a fuzz target for the
  importer.

An independent review compared the decoder with its own exact reference
(Dijkstra over node and observable parity, then dynamic programming over
events) on 240 000 syndromes of 40 000 random models. It also compared it
with PyMatching up to d = 17, and imported circuits with Stim on 400
random circuits. It found no wrong result there. It found:

* an error of probability 0, or one so small that (1 − p)/p overflows,
  set every matching weight to 0; such errors are now left out or get
  finite weights;
* deeply nested REPEAT blocks overflowed the stack, and annotations
  (TICK) inside large blocks were not counted toward the size limit;
  nesting is now limited to 64 and every annotation counts;
* the decoder's memory grew with the square of the number of events
  without a limit; shots needing more than 2²² pairs are now an error;
* noise without a probability was read as noiseless, and gate arguments
  were ignored; both are errors now, as in Stim.

### 1.11 — The state-vector core at memory bandwidth (delivered, release 1.6.0)

Use case: the precondition the roadmap sets for a GPU backend, measured
instead of assumed.

| Deliverable | Delivered | Left for later |
|---|---|---|
| Bandwidth reference | `benchmark_report` times every gate on 24 qubits against the fastest in-place pass over the state (one, two or four memory streams per thread) and names the kernels' instruction set | per-kernel numbers on other machines |
| Kernels | the kernels work on contiguous runs through primitives in AVX (x86-64, detected at run time) and NEON (aarch64), without FMA so that every version rounds as the portable one; diagonal gates read two memory streams; `set_simd` and `simd_level` in Rust and Python | AVX-512; vector paths for two-qubit gates on qubits 0 and 1, which keep element loops |
| Gradients | the adjoint backward pass blocked like the forward pass, with per-block terms summed in block order | blocked parameter-shift and finite differences |

Measured on the 4-core test machine (BENCHMARKS.md):

* on high qubits, H, RZ, RZZ, the complex U and a dense 4×4 matrix take
  0.97–1.32 times the reference pass; on qubits 0 and 1, 1.2–1.8 times
  (the dense 4×4 matrix 2.9 times);
* circuits of H, RZ and CX layers run 1.2–1.5 times faster than with the
  1.10 code at 20–24 qubits, layers with RY, RZ, CX and RZZ 1.5–1.9
  times;
* an adjoint gradient is 2–3 times faster at 16–20 qubits.

The cores are therefore close to memory bandwidth on whole-register
gates. Inside cache blocks, the arithmetic now runs in vector registers.

Validation:

* the vector and portable kernels agree bit for bit on every kernel, on
  small registers and on 17 qubits (native, and for NEON under Miri's
  aarch64 interpretation; CI also runs the tests on arm64);
* phase gates leave the amplitudes they do not touch unchanged, signs of
  zero included;
* the blocked backward pass equals the plain one to 10⁻¹² on circuits with
  gates inside and across blocks and controls above them, and Miri checks
  its unsafe code.

An independent review compared the vector paths with the portable ones
on random circuits of 1 to 20 qubits over the whole gate set (no bit
differed), compared states with a dense reference simulator and with the
1.10 code (bit-identical on 72 random circuits), checked gradients
against finite differences at 15–17 qubits and determinism for 1 to 7
threads. It found no memory-safety bug or wrong result. It found that
phase gates on low qubits multiplied the other half by 1, flipping the
sign of some zeros (fixed), and descriptions that overstated what the
equality test covered (corrected, and the test extended).

### 1.12 — Noise learning (delivered, release 1.6.0)

Use case: error mitigation whose noise model comes from data, as on a
device, not from the simulator's own model.

| Deliverable | Delivered | Left for later |
|---|---|---|
| Pauli-channel learning | `learn_layer` fits the sparse Pauli–Lindblad model of a twirled layer of `cx`, `cy`, `cz` and `swap` gates (van den Berg et al., Nat. Phys. 19, 1116 (2023)): one rate per one-qubit Pauli and per two-qubit Pauli on coupled qubits, from the decay of product eigenstates over even depths, measured in the bases of an orthogonal array over a colouring of the coupling graph, with readout twirling, non-negative least squares and standard errors from a bootstrap over the twirled circuits | layers that are not self-inverse; learning beyond the density-matrix limit of 10 qubits; resolving the degeneracy of Paulis the layer exchanges (only their product is learnable; the fit splits it evenly) |
| Mitigation | `pec_layered` cancels and `pea_layered` amplifies the learned noise in circuits of blocks separated by the layer, the latter with zero-noise extrapolation | inserted Paulis merged into neighbouring gates (they are separate, noiseless when the device has no one-qubit gate noise) |

Validation:

* on a four-qubit device with two-qubit depolarising noise, a coherent
  phase error on the control qubits and readout errors, all 39 learned
  rates lie within three standard errors of the exact ones, and PEC with
  the learned model gives 0.534 ± 0.005 for an exact 0.533 (raw 0.492);
* unit tests recover known sparse models (also with exactly measured
  idle qubits), check that PEC with the learned model is unbiased and
  that amplification extrapolates towards the exact value, check the
  Pauli conjugation rules against state vectors and NNLS against brute
  force and rescaled rows.

An independent review confirmed the conjugation rules, the twirls, the
bases, the decay fit, the bootstrap indexing and the PEC and PEA
probabilities. It found that the NNLS thresholds were absolute, so that
exactly measured fidelities (idle qubits without readout errors, whose
rows have huge weights) could set almost every rate to zero; that
bootstrap replicas with non-positive means inflated standard errors;
that one twirl was accepted and gave zero standard errors; and that the
mitigation functions did not re-check deserialised models. All are
fixed with tests: relative thresholds with protection against cycling, a
relative weight floor, robust spreads, at least two twirls, model
validation. The greedy colouring was replaced by DSatur and the bases by
orthogonal arrays from GF(3)ᵏ (9 bases up to 4 colours, 27 up to 13).

### 1.13 — Walks beyond static Hamiltonians (delivered, release 1.6.0)

Use case: adiabatic and driven walks, and decoherent discrete-time walks.

| Deliverable | Delivered | Left for later |
|---|---|---|
| Time-dependent walks | `DrivenWalk`: H(t) = Σ_j c_j(t)·H_j with Laplacian, adjacency, projector, potential and uniform-projector terms, by the fourth-order commutator-free Magnus integrator with two Krylov exponentials per step, step-size control by step doubling and a step cap for pulsed drives | higher-order Magnus schemes; symmetry-reduced driven walks (the quotient walks of 1.3 with a time-dependent rate); detecting discontinuities of the drive (they must be split by the caller) |
| Open discrete-time walks | `DecoherentCoinedWalk`: coined walks on density matrices of up to 2048 arcs, with partial measurements of the coin register and of the node after every step | trajectories for larger graphs; dephasing of the line walk |

Validation:

* the local adiabatic search of Roland and Cerf on K₆₄, K₂₅₆ and K₁₀₂₄
  (ε = 0.1) reaches the marked node with probability 0.99942, 0.99987
  and 0.99996 at the closed-form running time, a linear schedule over the
  same time 0.765, 0.525 and 0.315; all six values agree with SciPy's
  DOP853 on the two-dimensional subspace to 10⁻⁶;
* halving a fixed step divides the error by 18.5, 16.6 and 16.1 (order
  four); static walks equal `ContinuousWalk`; step-size control meets
  tolerances from 10⁻⁴ to 10⁻¹⁰, finishes down to 10⁻¹⁶ and resolves a
  pulse with `max_step`;
* with the coin measured every step, the Fourier-coin walk equals the
  simple random walk (Petersen graph, cube, cycle, Metatron's Cube, to
  10⁻¹²) and the Grover walk the Markov chain on arcs; steps equal a
  dense UρU† and keep states positive for every dephasing strength.

An independent review confirmed the Magnus coefficients, nodes and factor
order (error ratios 16.1–16.4 against a DOP853 reference with strongly
non-commuting terms), the error estimate and controller exponent, the
term handling and the Roland–Cerf numbers, which it reproduced
independently. It found two errors in the decoherent walk: a step
returned the transpose of UρU†, wrong for complex coins such as the
Fourier coin of degree three or more, and coin dephasing as first
defined was not a channel when it was stronger than position dephasing
(negative probabilities). Both are fixed, with tests that fail on the
old code. It also found that the step control stalled once it reached
its minimum step (tolerances below 10⁻¹² or stiff drives ran until
`max_steps`) and that steps longer than a pulse or across a jump of the
drive passed unnoticed; the controller was reworked, `max_step` added
and the behaviour documented.

## 4c. Third stage (1.14 to 1.17)

Planned October 2026, after release 1.6.0, from the gaps in the table of
section 1. Each milestone closes a gap without a new dependency and has an
independent reference.

### 1.14 — Gaussian boson sampling and imperfect photons (M, delivered)

Use case: the photonic experiments that followed Aaronson–Arkhipov
boson sampling, and photons as they are in the laboratory.

| Deliverable | Detail |
|---|---|
| Hafnians | hafnian and loop hafnian (Björklund, Gupt and Quesada 2019, O(n³2^{n/2})), exact for real and complex matrices |
| Gaussian states | covariance matrices and displacements of squeezed, coherent and thermal states; interferometers, squeezers, displacements and photon loss |
| Gaussian boson sampling | photon-number probabilities (hafnians of the kernel), threshold-detector probabilities (torontonians), exact sampling for small systems |
| Imperfect photons | partial distinguishability of Fock-state boson sampling (Gram matrix of the internal states) and photon loss |

Validation: hafnians of matrices with closed forms ((2n−1)!! for the all-ones matrix,
perfect matchings of graphs counted by brute force); a single-mode squeezed
state against its closed-form photon statistics; probabilities summing to
one; partial distinguishability interpolating between the permanent and the
classical limit, with the Hong–Ou–Mandel visibility |⟨ψ₁|ψ₂⟩|²; hafnians,
torontonians and GBS probabilities against The Walrus in the
cross-validation.

Delivered: `photonics::hafnian` (hafnians, loop hafnians, Kan's formula
for repeated rows, torontonians), `GaussianState` and
`ImperfectBosonSampler`, with Python bindings. The perfect matchings of
the hypercubes Q₃ to Q₅ (9, 272, 589 185), the dodecahedron (36), the
icosahedron (125), the Petersen (6) and Heawood (24) graphs, closed forms
for squeezed, coherent, thermal and two-mode squeezed states and the
Hong–Ou–Mandel dip are in the validation suite. Against The Walrus,
hafnians agree to 7·10⁻¹² relative up to 20 × 20 and 20-photon
probabilities to 2·10⁻¹¹.

An independent review confirmed the hafnian terms, the symplectic updates
(to 4·10⁻¹⁵ against The Walrus), Q, A and the prefactor, the click
probabilities and the sampling (χ² tests). It found that input loss was
wrong when partially distinguishable photons shared a lossy mode
(dropping photons one by one loses the coherences of the survivor's
internal state); loss now goes through environment modes, tested against
a Fock-space simulation that fails on the old code. Probabilities of
strongly squeezed states failed a symmetry check, lost 10⁻⁹ in a
closed-form 4 × 4 inverse and overflowed in det Q for hundreds of modes;
all three are fixed. Kan's formula, called stable, loses digits with the
photon number and overflowed: pure states now use a loop hafnian over
half the rows (20-photon errors from 10⁻⁶ down to 2·10⁻¹¹), Kan's formula
is rescaled and limited to 40 rows, and the accuracy statements were
corrected. Smaller findings (repeated modes in `reduced`, a scale-free
uncertainty check, error messages) are fixed as well.

### 1.15 — Molecular integrals (M, delivered)

Use case: molecules from their geometry, without an external
quantum-chemistry code.

| Deliverable | Detail |
|---|---|
| Integrals | overlap, kinetic, nuclear-attraction and electron-repulsion integrals over contracted Cartesian Gaussians (Obara–Saika or McMurchie–Davidson), STO-3G and 6-31G for the first two rows |
| Self-consistent field | restricted and unrestricted Hartree–Fock with DIIS; FCIDUMP output and the existing qubit Hamiltonians |

Validation: integrals, Hartree–Fock and full-CI energies of H₂, LiH, BeH₂
and H₂O against PySCF (10⁻⁸ Ha), the existing FCIDUMP files reproduced
from geometries.

Delivered: `chemistry::integrals` (molecules, STO-3G and 6-31G for H to
Ne, McMurchie–Davidson integrals) and `chemistry::scf` (RHF and UHF with
DIIS from atomic densities, a level-shifted fallback, frozen-core active
spaces as `MolecularIntegrals`), with Python bindings and `qso scf`. Full
CI moved to the Slater–Condon rules and, above 1024 determinants, to
Davidson's method on `H c` without the matrix. On identical geometries,
integrals agree with PySCF to 1.4·10⁻¹³, Hartree–Fock energies to
10⁻¹⁰ Ha, full CI and CASCI to 10⁻⁹ Ha.

An independent review confirmed the integrals element by element on 20
systems (to 3·10⁻¹⁴), the Boys function against 40-digit values, RHF and
UHF energies on some 40 atoms and molecules (10⁻¹³ Ha where both codes
reach the same state), CASCI on 20 active spaces, the FCIDUMP output read
by PySCF and the Slater–Condon FCI in spin-polarised sectors. It found
that the core-Hamiltonian guess converged silently to excited
determinants (N₂ in STO-3G 0.73 Ha too high, also O₂, CH₂, NH₂ and
stretched bonds); the atomic guess now reaches PySCF's states, with tests
that fail on the old guess. DIIS failed for NO and CN; NO now converges
and CN through the level-shifted fallback. Full CI near its limit needed
6.6 GB and minutes and several roots were impractically slow; the
matrix-free Davidson needs 0.7 GB and 36 s for 245 025 determinants. A
claimed PySCF integral error was the difference of the two codes' bohr
constants; the comparisons now use identical geometries. Smaller findings
(unchecked deserialisation, integrals of another molecule, error
messages, slow integrals, copies of the integral tensor in Python) are
fixed; near-linearly-dependent bases (atoms within 0.01 Å) still leave
rounding noise above the default gradient tolerance, as documented.

### 1.16 — Fast matching (M, delivered)

Use case: decoding large surface-code experiments shot by shot.

| Deliverable | Detail |
|---|---|
| Sparse matching | a matching decoder that grows regions on the detector graph instead of building complete graphs of defects (the idea of sparse blossom, Higgott and Gidney 2023) |

Validation: identical correction weights to the current exact decoder and
to PyMatching on every shot; speed measured against PyMatching on the same
machine.

Delivered: `qec::SparseMatchingDecoder` (Python `SparseMatchingDecoder`),
Edmonds' primal-dual algorithm run on the detector graph: regions grow,
freeze and shrink, trees grow and augment, blossoms form and shatter, and
`decoder="matching"` uses it. Its correction weights equal the exact
decoder's on every tested shot and PyMatching's to the rounding of the
weights (the cross-validation checks the first 2000 shots of every circuit
and every shot whose prediction differs). On surface codes from d = 11 to
25 it takes 1.1 to 1.9 times PyMatching's time (the exact decoder: up to
1000 times).

An independent review found no wrong correction in about 2.4 million
comparisons with brute force, the exact decoder and PyMatching (random
graphs with and without boundary, zero-weight and parallel edges,
lattices, surface codes), no violated invariant with debug assertions, and
no overflow. It found a cubic worst case: deeply nested blossoms made
every node walk up the nesting (a chain of 2001 events took 3 s); the
top-level region is now cached per node (51 ms). It also corrected the
cost and speed claims, which came from single unwarmed runs, and a guide
example that compared the two decoders' tie-breaking.

### 1.17 — Mitigation by purification (S–M, delivered)

Use case: error mitigation without a noise model.

| Deliverable | Detail |
|---|---|
| Virtual distillation | ⟨O⟩ from Tr(ρᵏO)/Tr(ρᵏ) for k = 2, 3, exactly on density matrices and with the circuits of Huggins et al. (2021) on copies of the register |

Validation: exact purified values against the dominant eigenvector of ρ;
the circuit estimator against the exact one; error suppression measured
on noisy circuits.

Delivered: `mitigation::distillation` (`distilled_expectation`,
`dominant_expectation`, `DistillationCircuits`) with Python bindings. The
circuits diagonalise, per qubit, the k-qubit unitary of the cyclic shift
times the Pauli factor with two-level rotations, so they export to
OpenQASM. They reproduce `Tr(ρᵏO)/Tr(ρᵏ)` to 10⁻¹¹ for one to three
copies. On a depolarised GHZ-like state the error falls from 0.19 to
3.6·10⁻³ and 6.9·10⁻⁵; when the dominant eigenvector moves, it stops at
that eigenvector's floor (9·10⁻⁴ in the second example).

An independent review confirmed the factorisation over qubits, the
estimator, the bit and copy conventions (also with qiskit and Aer noisy
density matrices, to 10⁻¹⁵) and the coverage of the standard errors. It
found that the measurement circuits could not leave the library (custom
multi-qubit gates); they are now built from exportable controlled
rotations. Malformed probabilities and counts were accepted, the dense
functions were slow at their limit, and the documentation presented a
state without a noise floor as typical; all fixed, with an example and a
test that show the floor and one that shows a coherent error passing
through.

## 4d. Fourth stage (1.18 to 1.21)

Planned October 2026, after the third stage, from the updated gaps in the
table of section 1. As before, each milestone closes a gap without a new
dependency of the crate and has an independent reference; reference tools
(here the Python package `ldpc`) enter only the cross-validation.

### 1.18 — Pretty good state transfer, decided exactly (M, delivered)

Use case: quantum walks as transport. Perfect state transfer is rare;
pretty good state transfer — fidelity arbitrarily close to 1 at suitable
times — is common on paths and cycles and is decided by number theory,
not by searching a time window.

| Deliverable | Detail |
|---|---|
| Strong cospectrality | exact test from the eigenprojections, with the eigenvalue support of the pair |
| The decision | by Kronecker's theorem: PGST between strongly cospectral `u` and `v` holds exactly when every integer relation `Σ ℓ_r θ_r = 0` among the support eigenvalues has `Σ ℓ_r σ_r` even (`σ_r` the sign of `v` in eigenspace `r`; Banchi, Coutinho, Godsil and Severini 2017) |
| Exact eigenvalues | for graphs whose eigenvalues lie in cyclotomic fields (circulants and Cayley graphs of abelian groups, paths and their symmetric quotients): eigenvalues as rational vectors in a basis of `Q(ζ_N)`, integer relations by Hermite normal form; other graphs answer "undecided" |
| Times | for a decided pair, a time with fidelity above a requested `1 − ε` (simultaneous Diophantine approximation), checked by evolving the walk |

(The criterion in the table, stated precisely: every relation
`Σ_{r≥1} ℓ_r (θ_r − θ_0) = 0` has `Σ ℓ_r δ_r` even, with `δ_r` marking the
eigenspaces where the sign differs from that of `θ_0`.)

Validation: paths `P_n` have PGST between their ends exactly when `n + 1`
is a prime, twice a prime or a power of two (Godsil, Kirkland, Severini
and Smith 2012), checked for every `n` up to a few hundred; cycles `C_n`
have PGST between antipodal vertices exactly when `n` is a power of two
(Pal 2017); perfect state transfer (already decided since 1.3) implies
PGST; every returned time reaches its fidelity in the simulated walk.

| Deliverable | Delivered | Not done, and why |
|---|---|---|
| Strong cospectrality | from the closed-form eigenvectors of each family, exactly (sine and cosine signs by residues) | — |
| The decision | `walks::PrettyGoodTransfer` (Python `PrettyGoodTransfer`, `qso transfer`): Kronecker's criterion with the relations as the integer kernel of the eigenvalues' power-basis coordinates in `Q(ζ_N)`; obstructions as witnesses; PST and its time from the same integer data | a parity-only search (rational kernel and 2-adic saturation) that would avoid 128-bit overflow on large non-power-of-two cycles: the column reduction errs there instead of answering, and refuses relation searches above 2²⁴ integers |
| Exact eigenvalues | paths (adjacency and Laplacian), cycles, Cayley graphs of finite abelian groups | symmetric quotients of paths; other graphs give an error rather than "undecided" (the CLI says "undecided") |
| Times | `best_transfer`, a window search, with `fidelity(t)` exact for any pair | the Diophantine construction of a time for a requested `1 − ε`: PGST times grow quickly with the fidelity, and the window search with a propagated-walk check covers the cases a user can simulate |

Delivered as planned otherwise: the three theorems hold for every size
tested (paths n ≤ 300, Laplacian paths n ≤ 200, cycles n ≤ 512, both
generators), PST times and fidelities agree with `StateTransfer` on
hypercubes, circulants, paths, tori, `Z_3 × Z_5` and H(2, 3), and a P₄ time
reaches fidelity 0.999 in the walk.

An independent review checked the decisions against its own exact oracle
(big-integer elimination with 2-adic saturation) on 21 320 path pairs and
18 554 Cayley pairs, against van Bommel's characterisation of all mirror
pairs of paths and against Bernasconi–Godsil–Severini on cube-like graphs,
and found no wrong decision. It found that `fidelity` was wrong for pairs
that are not strongly cospectral (it summed only the cospectral
eigenspaces), that a large Cayley group exhausted memory and aborted the
process, that PST times were confirmed in floating point (and lost at
extreme hopping rates), that duplicate connection elements and empty sets
were accepted, that some errors named the wrong argument, and that the
guide's timings were untested and partly wrong. All fixed: the fidelity
comes from the closed-form `⟨v|E_θ|u⟩` and is tested for every pair, the
relation search is bounded before it allocates, PST is decided from the
integer data, and the guide states the measured costs.

### 1.19 — Decoding beyond matching (M–L, delivered)

Use case: codes whose errors flip more than two detectors — colour codes
and quantum LDPC codes such as the bivariate bicycle codes.

| Deliverable | Detail |
|---|---|
| BP+OSD | belief propagation (min-sum and product-sum) on the detector error model's check matrix, with ordered-statistics post-processing (OSD-0 and combination sweep; Roffe et al. 2020), GF(2) elimination without dependencies |
| Codes | colour-code memory experiments (triangular 4.8.8 or 6.6.6) and bivariate bicycle codes, including the [[144, 12, 12]] code (Bravyi et al. 2024), as detector circuits |
| Interfaces | Rust, Python and `qso qec --decoder bposd`; error models without the `decompose` restriction |

Validation: on small codes the decoder against exhaustive minimum-weight
decoding (agreement measured and stated); logical error rates and
predictions against the `ldpc` package on the same error models and
syndromes in the cross-validation; colour-code circuits against Stim's
generated colour-code circuits (error models term by term); the code
parameters `n` and `k` of the bivariate bicycle codes by GF(2) rank.

| Deliverable | Delivered | Not done, and why |
|---|---|---|
| BP+OSD | `qec::BpOsdDecoder` (Python `BpOsdDecoder`): min-sum (fixed or adaptive factor) and product-sum BP, OSD-0, combination sweep and exhaustive OSD on a reduced GF(2) basis | serial and layered schedules: flooding is what `ldpc` defaults to and what the comparison needs |
| Codes | `qec::CssCode` from any checks, triangular 6.6.6 colour codes, bivariate bicycle codes and the table of Bravyi et al.; `css_code_memory` in both bases | the depth-8 syndrome circuit of Bravyi et al. and hook-aware colour-code schedules: the generic schedule measures X then Z checks in greedy layers, so its circuit distance can be lower |
| Interfaces | `DecoderKind::BpOsd`, `qso qec --code color\|bb --decoder bposd`, Python `css_code_memory`, `logical_error_rate(..., decoder="bposd")`, `DetectorErrorModel` from a list | — |

Validation as planned, with one change: Stim's generated colour-code
circuit (`color_code:memory_xyz`) uses C_XYZ gates outside the importer's
gate set, so QSO's own CSS circuits are compared with Stim's error models
term by term instead (to 10⁻¹⁵). Exhaustive OSD equals enumeration on
every syndrome of 150 random models with hyperedges; every weight ≤ 2
error is corrected on the colour codes d = 5, 7 and the [[72, 12, 6]]
code; against `ldpc` 2.4, min-sum BP with OSD-CS and exhaustive OSD gives
the same correction on every shot with equal priors (product-sum: 0.3% on
[[144, 12, 12]]), and circuit-level predictions differ in at most 0.1% of
shots, where the OSD scores differ (Σ ln((1 − p)/p) against Σ ln(1/p)).
One finding is recorded rather than fixed, because `ldpc` shares it: on
the Steane code, scaled min-sum BP converges on four errors for the error
on the qubit of all three checks.

An independent review confirmed the BP messages line by line against
`ldpc`, the OSD solution and candidate mapping, the GF(2) basis, the code
constructions and the memory circuits, and fuzzed 300 models with
enumeration. It found that a directly built error model with an enormous
detector count panicked or aborted the matching decoders, that
`DecoderKind::BpOsd.build` could panic, that bicycle-code exponents beyond
2⁶³ wrapped into a wrong code, that `CssCode` had no size limit (hours at
65 536 qubits), that exhaustive OSD and order-0 sweeps broke ties
differently from `ldpc`, and that the README overstated the agreement and
the guide quoted untimed speeds. All fixed: models, decoders and codes are
bounded with errors, exponents are reduced, tie order and order 0 follow
`ldpc`, and the cross-validation now times both decoders and checks the
Steane case and exhaustive OSD.

### 1.20 — Single precision (M, delivered)

Use case: one more qubit in the same memory and faster gates where
10⁻⁶ accuracy suffices (sampling, QAOA landscapes, large walks).

| Deliverable | Detail |
|---|---|
| `f32` state vectors | the gate kernels, fusion, measurement and sampling in single precision (AVX and NEON paths as for `f64`), opt-in per simulation |
| Limits | the qubit limit raised by one for single precision; the documentation states the accuracy |

Validation: random circuits against the double-precision simulator
(fidelity loss measured and bounded in the documentation); speed and
memory against the `f64` path and against Qulacs and Qiskit Aer in single
precision on the same machine.

| Deliverable | Delivered | Not done, and why |
|---|---|---|
| `f32` state vectors | `quantum::StateVector32` (Python `StateVector32`, `simulate(..., precision="single")`, `qso qasm --single`): the kernels and the fused, blocked plan are generic over the amplitude type; AVX (four amplitudes per register) and NEON (two) primitives, bit for bit the portable arithmetic; sampling, measurement, probabilities, expectation values, conversion to and from double precision | single-precision density matrices, trajectories and mid-circuit programs: they keep double precision, and their registers are small or their cost is elsewhere |
| Limits | `MAX_QUBITS_SINGLE` = `MAX_QUBITS` + 1 | — |

Validation as planned: after 2140 gates on 18 qubits the direction stays
within 1 − F < 10⁻¹¹ of the double-precision state (the norm drifts by
8·10⁻⁶; probabilities and expectation values are divided by it), and
expectation values agree to 10⁻⁶. In the cross-validation QSO's and
Qiskit Aer's single precision both reach 1 − F ≈ 4·10⁻¹⁴ on 60-gate
circuits. Speed on 4 cores: 1.6 to 2.1 times faster than QSO's double
precision from 20 qubits on (24 qubits, one layer: 150 against 277 ms),
and 7 to 13 times faster than Qiskit Aer, whose single precision was not
faster than its double precision here. Qulacs offers no single precision on
the CPU; Cirq and PennyLane Lightning, which do, are timed in double
precision only. Memory halves by construction (8 bytes per amplitude
instead of 16).

An independent review found no soundness or rounding error in the AVX and
NEON code (shuffles, blend masks, tails, operation order; bit for bit
against the portable loops with signed zeros, subnormals and infinities)
and confirmed that the double-precision path is bit for bit unchanged:
state vectors, noisy density matrices, adjoint gradients and state
derivatives of 78 circuits on 1 to 17 qubits, before and after. It found
wrong numbers in the documentation (the gate count of the validation
circuit, the size of the norm drift, which frameworks offer single
precision, "two more qubits"), a Python test that could not fail, a
kernel test that compared with `==` rather than bit patterns, and NaN
from the readouts of an all-zero state; all fixed.

### 1.21 — Coverage and differential testing (S–M)

Use case: trust in the parts no reference covers directly.

| Deliverable | Detail |
|---|---|
| Coverage | line and branch coverage of the crate measured in CI and published with the documentation |
| Differential testing | random circuits run on all simulators that accept them — state vector, density matrix (no noise), trajectories, MPS (unbounded bond), tableau (Clifford circuits) — and through OpenQASM 2 and 3 round trips, in property tests and as fuzz targets |
| Fixes | every disagreement found, fixed with a regression test |

Validation: the coverage number per release; the differential targets run
in CI on every change and longer on the scheduled job.

### Not in this stage

Larger basis sets with d functions and restricted open-shell Hamiltonians
(chemistry), tight commutator bounds for product formulas of order four
and higher, MPS in the browser build, and documentation per released
version remain gaps; they are smaller or have fewer users than the four
above.

### Research track (later)

A GPU backend through `wgpu`: it starts only with a clear use case and
after the CPU core is close to memory bandwidth.

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
   QSO's graphs. Both ship as one release.
7. **1.9: matrix-product states** from the research track: they need no
   new dependency (splits use nalgebra's Hermitian eigensolver), they
   reach the large chains and sparse graphs that QSO's walks and QAOA
   work on, and free fermions and the QAOA ring formula give exact
   references far beyond 34 qubits.
8. **1.10: exact decoding first in the second stage.** The accuracy cost
   of union-find is measured and visible; matching removes it, and
   PyMatching gives a shot-by-shot reference. Importing Stim circuits
   lets the cross-validation use Stim's own generated circuits.
9. **1.11: performance** next, because a GPU backend waits for it.
10. **1.12 and 1.13: noise learning and driven walks**, each closing a gap
    of the first stage's table.
11. **Third stage: photons first.** Boson sampling is where QSO started
    (1.1), and The Walrus already serves as a reference in the
    cross-validation, so Gaussian boson sampling (1.14) extends a
    validated core. Molecular integrals (1.15) remove the last external
    input of the chemistry stack; PySCF is the reference. Fast matching
    (1.16) closes a measured speed gap, and purification (1.17) adds a
    mitigation that needs no noise model.
12. **Fourth stage: walks first again.** Pretty good state transfer (1.18)
    is the walk question QSO still answers only numerically, and number
    theory gives exact references. Decoding beyond matching (1.19) opens
    the codes current experiments use. Single precision (1.20) serves the
    gate-model core; coverage and differential testing (1.21) close the
    stage because they test everything before it.

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
  Matrix-product states follow in 1.9; a GPU backend waits.
* Decoding starts with union-find, not minimum-weight matching: it is
  short, has no dependency and runs in almost linear time. Its logical
  error rates are reported beside PyMatching's, so its accuracy cost is
  visible. Error models and circuits interoperate with Stim through its
  text format, not through a binding.
* Second stage (October 2026): exact decoding (1.10), the state-vector
  core at memory bandwidth (1.11), noise learning (1.12), driven and open
  walks (1.13). Finished milestones are released together.
* Third stage (October 2026): Gaussian boson sampling and imperfect
  photons (1.14), molecular integrals (1.15), fast matching (1.16),
  purification (1.17). The GPU research track still waits for a use case,
  and would need a dependency (`wgpu`).
* Fourth stage (October 2026): exact pretty-good-state-transfer decisions
  (1.18), BP+OSD decoding with colour and bivariate bicycle codes (1.19),
  single precision (1.20), coverage and differential testing (1.21). The
  `ldpc` package joins The Walrus, PySCF, Stim and PyMatching as a
  reference of the cross-validation only.
* The browser playground stays, because it costs little to maintain: it
  is plain HTML and JavaScript without dependencies or a build step, it
  uses only the npm package's JSON interface, which the package's tests
  cover, and the documentation build publishes it. If it ever needs
  more than that, it is removed from the site rather than given a
  framework.
