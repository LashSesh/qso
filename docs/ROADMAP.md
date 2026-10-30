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
(1.14 to 1.17), fourth (1.18 to 1.21), fifth (1.22 to 1.25) and sixth
(1.26 to 1.28) stages ship together as release 1.7.0. The seventh
(1.29 to 1.33, section 4g) and eighth (1.34 to 1.37, magnetic resonance
for medical research, section 4h) stages ship together as release
1.8.0. Up to 1.3
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
| Statevector speed | specialised kernels (1.1); gate fusion and cache blocking (1.3.0): 16-qubit layers 4.7× and 18–24-qubit layers 1.6–2.6× faster than in 1.2; batched evaluation; AVX and NEON kernels close to memory bandwidth and a blocked adjoint pass (1.11); single precision, one more qubit and about twice the speed on large registers (1.20); a worker pool that stays awake between short simulations (1.24) | spinning workers cost CPU time between simulations (`QSO_POOL_SPIN_US=0` turns it off) |
| Boson sampling | exact probabilities and exact sampling (1.1); partial distinguishability, loss and Gaussian boson sampling with hafnians and torontonians (1.14) | mixed Gaussian states with many photons spread over many modes are impractical (10¹² operations at 20 photons in 10 modes) |
| Noise and mitigation | device models (per-qubit T₁/T₂ and readout, per-gate durations, errors and over-rotations, JSON), density matrices up to 14 qubits, quantum trajectories beyond; readout mitigation, ZNE, PEC, Pauli twirling, classical shadows, standard errors on every estimate (1.4) | noise learning for layers of two-qubit Clifford gates only, on density matrices up to 10 qubits and Pauli frames up to 64 (1.12, 1.31), with non-Pauli channels twirled one at a time; mitigation with the learned model above 24 qubits only on matrix-product states, for channels that are mixtures of unitaries (1.31); no leakage; channels on at most four qubits, slow noise as quasi-static and telegraph processes only, and on trajectories only (1.30); virtual distillation (1.17) mitigates only its measured copies' preparation |
| Quantum walks | continuous- and discrete-time walks, symmetry reduction to exact low-dimensional problems (Q₃₀ in 31 dimensions), perfect state transfer, walk circuits (1.3.0); driven and decoherent walks (1.13); exact pretty-good-transfer decisions on paths, cycles and abelian Cayley graphs (1.18) | time-dependent Hamiltonians and decoherent coined walks without symmetry reduction; pretty good transfer on other graphs |
| Physics and chemistry | fermion-to-qubit mappings, FCIDUMP molecules with full CI, UCCSD, k-UpCCGSD, ADAPT-VQE; product formulas with error bounds, qDRIFT; VQD, SSVQE, natural gradient, imaginary-time evolution; QUBO/QAOA mixers; amplitude estimation (1.5); Gaussian integrals (STO-3G, 6-31G, H to Ne), RHF and UHF, frozen-core active spaces and matrix-free full CI (1.15); shells up to g, 6-31G* and cc-pVDZ to argon, ROHF (1.23); nuclear spin dynamics with pulses, relaxation, spectra and parahydrogen methods, up to eleven spins (1.33); chemical exchange between spin systems (SABRE build-up, lineshapes), long-lived singlet order and field-cycling transfer (1.35); optimal control (GRAPE) of rf pulses and field profiles with exact gradients (1.36); NV centres with their nitrogen and up to eight ¹³C, sensing sequences with ideal or finite pulses, classical field noise with filter functions and exact trajectories (1.37) | Schwarz screening helps extended molecules only (5 % of the quartets of butane); the nested-commutator bound of product formulas (1.32) is tight for small Hamiltonians only, its tail ignoring which terms commute; spin relaxation, phenomenological (T₁ and T₂) or from dipolar couplings and shielding anisotropy by Redfield theory for isotropic tumbling (1.34), is exact up to about six relaxing spins of one isotope; larger relaxing systems run for short times only (Redfield not at all), and propagations estimated above a minute are refused; chemical exchange needs steady chemistry (no irreversible hydrogenation) and one block of at most 2048 elements over all sites; optimal control with relaxation works in Liouville space of up to six spins ½, and field profiles have many local optima; NV centres are secular (field along the axis), with classical noise only (no spin bath beyond the explicit nuclei, no optical cycle, so pulsed ODMR only) |
| Interoperability | OpenQASM 2 with `gate` definitions and an OpenQASM 3 subset (1.6); a WASM build, npm package and browser playground (1.6); Stim's circuit format, written (1.8) and read (1.10); edge lists, GraphML and duck-typed NetworkX import for graphs | no bridges to other frameworks, by decision (see section 2) |
| Error correction | stabilizer tableaux up to 16 384 qubits (1.7); detector circuits, frame sampling, error models and union-find decoding of repetition and surface codes (1.8); exact matching decoding and X-basis memory (1.10); sparse matching at 1.1–2.3 times PyMatching's time (1.16); BP+OSD with colour and bivariate bicycle codes (1.19); circuit distance, distance-preserving schedules and correlated matching (1.22) | circuit distances of hypergraph models are bounds, not always exact |
| Large registers | matrix-product states up to 65 536 qubits with a fidelity bound (1.9), in the browser build too (1.25) | — |
| Documentation | user guide, executed tutorials and Python reference on GitHub Pages (1.2); Rust tutorials from the examples and the documentation of every release since 1.2.0 (1.25) | — |
| Engineering | CI on Linux (x86_64, arm64), macOS and Windows; Miri, fuzzing, API checks, weekly cross-validation and benchmark tracking; Trusted Publishing (1.2); line and branch coverage, and differential tests of all simulators on random circuits (1.21); coverage of the CLI and of the Python tests (1.25) | — |

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
  PennyLane Lightning 3.7–9 times; on 16 qubits Qulacs was still 1.3×
  faster in the median, until the worker pool of 1.24 (next item).
* **Thread wake-up:** rayon workers sleep as soon as they are idle; on the
  test VM waking them is slow, and a burst of short simulations needed tens
  of milliseconds to reach full speed (a cold 16-qubit layer took 1.5 ms,
  a warm one 0.55 ms). Qulacs' OpenMP workers spin. Since 1.24 the
  state-vector kernels run on their own pool, whose workers spin briefly
  before parking: a 16-qubit layer after a 20 ms pause takes about
  0.7 ms in Rust (1.2–1.3 ms before), and in Python QSO is ahead of
  Qulacs cold and warm from 10 to 18 qubits. The comparison still warms every framework up for 0.25 s and
  times cold starts separately.
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
| Thread wake-up | — | found while measuring 1.3.0: rayon workers sleep as soon as they are idle, and on virtual machines short isolated simulations (around 1 ms) run at close to single-thread speed until the workers stay awake; keeping them spinning briefly needs a custom thread pool; delivered in 1.24 |

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
| Hamiltonian simulation | `Circuit::pauli_rotation`; product formulas of order 1, 2, 4, 6, …; `trotter_error_bound` with the commutator bounds of Childs et al. (2021) for orders 1 and 2 and the smaller of a stage-wise and a Taylor-remainder bound above; qDRIFT with Campbell's bound | a tight commutator bound for order ≥ 4 (the paper's Appendix J is heuristic in its constants); the bound QSO gives there is rigorous but loose (1.32 adds a bound from the exact nested commutators, tight for small Hamiltonians) |
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

### 1.21 — Coverage and differential testing (S–M, delivered)

Use case: trust in the parts no reference covers directly.

| Deliverable | Detail |
|---|---|
| Coverage | line and branch coverage of the crate measured in CI and published with the documentation |
| Differential testing | random circuits run on all simulators that accept them — state vector, density matrix (no noise), trajectories, MPS (unbounded bond), tableau (Clifford circuits) — and through OpenQASM 2 and 3 round trips, in property tests and as fuzz targets |
| Fixes | every disagreement found, fixed with a regression test |

Validation: the coverage number per release; the differential targets run
in CI on every change and longer on the scheduled job.

| Deliverable | Delivered | Not done, and why |
|---|---|---|
| Coverage | a CI job (`cargo-llvm-cov`, nightly for branches) with totals and the least-covered files in the job summary; the HTML report published with the documentation under `coverage/`; at this milestone 90.5 % of lines, 92.8 % of functions, 90.6 % of regions and 79.4 % of branches | coverage of the Python bindings and the CLI: their tests run a separate build of the extension and the binary, so the CLI's source (`src/bin/qso.rs`, 872 lines) counts in the totals as uncovered |
| Differential testing | `differential::compare`: one circuit on the single-precision state vector, the noiseless density matrix and trajectory, an MPS of unlimited bond dimension, the tableau (all Pauli expectation values for Clifford circuits) and the OpenQASM 2 and 3 round trips, against the double-precision state vector; property tests on random circuits of every gate kind (up to three controls, dense unitaries on up to three qubits, symbolic and special angles, 1 to 10 qubits) and a `differential` fuzz target, both on every change and longer every week | mid-circuit programs across simulators: their outcomes are distributions, compared already by the OpenQASM fuzz target and the program tests |
| Fixes | three, each with a regression test: an MPS without cutoff kept singular values of 10⁻⁴⁰, whose squares broke the next eigendecomposition (NaN states; the fuzz target could crash on it); the tableau accepted any rotation by an angle above 4·10¹⁴ as Clifford; the single-precision kernel test compared NaN bit patterns, which Miri randomises | — |

An independent review found the three faults above with its own
adversarial circuits, and showed that the first version of the checks was
too weak: `1 − F` is quadratic in the error, so a tolerance of 10⁻⁹ let
angle errors of 5·10⁻⁵ through (a mutation that rounded every exported
angle to six digits passed), norm errors were clamped away, export
failures were taken for "no form", the tableau check was nearly vacuous
beyond five qubits, and the generators missed gate kinds, multiple
controls, three-qubit unitaries, parameters and larger registers. All
fixed: amplitudes are compared after aligning the phase (an angle off by
10⁻⁸ shows), norms are checked, only the exporters' explicit refusals
count as "no form", the tableau's generators are checked, and the
generators cover what was missing. 40 000 property-test circuits and
21 000 fuzzed ones then agreed.

### Not in this stage

Larger basis sets with d functions and restricted open-shell Hamiltonians
(chemistry), tight commutator bounds for product formulas of order four
and higher, MPS in the browser build, and documentation per released
version remain gaps; they are smaller or have fewer users than the four
above.

### Research track (later)

A GPU backend through `wgpu`: it starts only with a clear use case and
after the CPU core is close to memory bandwidth. The sixth stage names the
use case and serves it on the CPU first (section 4f).

## 4e. Fifth stage (1.22 to 1.25)

Planned October 2026, after the fourth stage, from the gaps that remain in
the table of section 1 and from what the fourth stage's reviews measured.
The rules stay: no new dependency of the crate, an independent reference
for every deliverable, reference tools (Stim, PySCF, Qulacs, Qiskit Aer)
in the cross-validation only.

### 1.22 — Error-correction circuits that keep their distance (M–L, delivered)

Use case: experiments with colour and bivariate bicycle codes. The generic
syndrome schedule of 1.19 measures X and then Z checks in greedy layers;
hook errors lower its circuit distance (the distance-3 colour code fails
linearly in p), and nothing in QSO reports the distance a circuit actually
has.

| Deliverable | Detail |
|---|---|
| Circuit distance | the fewest error mechanisms of a detector error model that flip an observable and fire no detector: exact for graph-like models (shortest paths through the matching graph), and for hypergraph models an exact search up to a weight limit with a BP+OSD upper bound beyond it |
| Schedules | CNOT orders for the triangular colour code that keep circuit distance d, and the syndrome cycle of Bravyi et al. (2024) for bivariate bicycle codes, as options of `css_code_memory`; the current greedy layering stays the default for arbitrary codes |
| Correlated decoding | two-pass matching for depolarising noise: after a first matching, the edges correlated with the chosen ones through Y-type errors are reweighted and the shot is decoded again (Fowler, arXiv:1310.0863) |

Validation: circuit distances of surface, colour and repetition circuits
against Stim's `shortest_graphlike_error` and
`search_for_undetectable_logical_errors` on the same circuits; the rotated
surface code keeps distance d with the hook-safe order and loses it with a
bad one; the colour-code schedules reach distance d; the bicycle-code
cycle reaches the circuit distance and logical error rates reported in the
paper (within sampling error); correlated matching against exhaustive
maximum-likelihood decoding on small codes and against plain matching on
surface codes (logical error rates measured side by side).

| Deliverable | Delivered | Not done, and why |
|---|---|---|
| Circuit distance | `graphlike_distance` (exact: shortest cycle with odd observable parity, by breadth-first search on the graph doubled by parity) and `circuit_distance` (exact for graph-like models; otherwise an exhaustive search by increasing weight gives a proven lower bound, and the graph-like errors and BP+OSD searches, with the parity of observable sets as an extra check, give an upper bound with its witness); Python, `qso qec --circuit-distance` | — |
| Schedules | `css_code_memory_with` with `CssSchedule::Bicycle`, the depth-8 cycle of Bravyi et al.: [[72, 12, 6]] keeps circuit distance 6, proven (greedy layers: 4); `CssSchedule::Directions` for colour codes | colour-code orders that keep distance d: none exists with one bare ancilla per face. All 720 orders of directions give (d + 1)/2 for d = 3 and 5 (in the Steane code every pair of qubits lies on a weight-3 logical, so every two-qubit hook loses distance); this needs flag qubits or other extraction circuits, not a better order |
| Correlated decoding | `CorrelatedMatchingDecoder` with PyMatching 2.4's rule (implied weights from joint and marginal probabilities of decomposed parts, two passes); `correlated_error_model`, and decomposition by Stim's rule within each noise channel, so that QSO's decomposed models equal Stim's | Fowler's original rule (arXiv:1310.0863) in its own form: PyMatching's is a version of it and a stronger reference, since its results can be compared shot by shot |

Validation:

- **Distances.** Exact results equal enumeration on 600 random models,
  graph-like and with hyperedges. On Stim's generated surface and
  repetition circuits (d = 3 to 7), the graph-like distance equals Stim's
  `shortest_graphlike_error`. The exact circuit distance is never larger
  than what `search_for_undetectable_logical_errors` finds, and equal on
  these circuits.
- **Surface and repetition codes.** QSO's memory experiments keep
  distance d.
- **Bicycle codes.** With three cycles of QSO's noise, the circuit
  distances lie in 7 to 8, 7 to 9 and 7 to 11 for [[90, 8, 10]],
  [[108, 8, 10]] and [[144, 12, 12]], against the paper's ≤ 8, ≤ 8 and
  ≤ 10. The paper's noise also strikes idle qubits; the upper bounds are
  the lightest errors 256 BP+OSD searches per observable found.
- **Correlated matching.** QSO's decomposition of Stim's circuits equals
  Stim's, error by error with its parts. On Stim's model and syndromes,
  QSO's two-pass decoder and PyMatching's disagree on 0.01–0.04% of
  shots. At d = 5 and p = 0.006, both lower the logical error rate from
  2.44·10⁻² to 1.93–1.94·10⁻².

Not done:

- **Exhaustive maximum-likelihood decoding.** Against PyMatching the
  comparison is shot by shot, which is stronger than comparing rates.
- **The paper's logical error rates.** Its noise model has idle errors,
  which QSO's circuit noise lacks, and 10⁻⁵ to 10⁻¹² per cycle needs far
  more shots than a test can afford.
- **A "bad" surface-code order losing distance.** QSO's surface code
  builds only the hook-safe order. The loss is shown on the
  [[72, 12, 6]] code instead (greedy 4 against the cycle's 6).

Also in this milestone:

- `DecoderKind` went back to its 1.6.0 form. 1.19 had added a variant to
  this exhaustive enum and made `build` fallible, both breaking for 1.x;
  `cargo-semver-checks` reported the first.
- The non-exhaustive `DecoderChoice` takes all decoders.

An independent review fuzzed the distance search against enumeration on
12 000 random models and checked the decomposition against Stim's source,
the two-pass decoder against PyMatching's and the bicycle cycle against
the paper's table. It found:

- **A false "none".** A search that stopped at `max_weight` without a
  witness reported that no logical error exists (true distance 6 on the
  [[72, 12, 6]] cycle with `max_weight = 5`). Existence is now decided
  exactly by GF(2) linear algebra.
- **A regression.** `Y_ERROR`s that flip more than two detectors no
  longer decomposed. A Y error is now split over its X and Z parts, and
  stays whole when graph-like, as in Stim.
- **No bound on BP+OSD searches.** `bposd_trials` is now at most 2¹⁶.
- **Unchecked parts.** Parts are now checked to add up to the error.
- **Smaller points, all fixed:**
  - `CssCode` equality compared the new structure;
  - `--schedule` was silently ignored with `--circuit`;
  - Python `lower_bound` was 2⁶⁴ − 1 for "none" (now `math.inf`);
  - the colour-code claims were stated for d = 7 and "three rounds";
  - missing tests: a stopped search, Y errors, and the gain of
    correlations in Python.

The review confirmed the rest: the search's pruning and bookkeeping, the
shortest-cycle search, the channel decomposition line by line, the
reweighting rule, the depth-8 cycle and the semver check against 1.6.0.

### 1.23 — Larger basis sets and open shells (M–L, delivered)

Use case: chemistry beyond the smallest basis sets and the first two rows.
STO-3G and 6-31G without polarisation functions give qualitatively wrong
geometries and reaction energies; most benchmark sets use cc-pVDZ or
6-31G*.

| Deliverable | Detail |
|---|---|
| Integrals | the McMurchie–Davidson recursion for any angular momentum (d and f functions), Cartesian and spherical shells, with the same Boys-function and screening code |
| Basis sets | 6-31G* and cc-pVDZ for H to Ar; STO-3G and 6-31G extended to Ar; basis data in the repository, as today |
| Open shells | restricted open-shell Hartree–Fock (ROHF) and the active-space Hamiltonian of its orbitals, beside RHF and UHF |

Validation: one- and two-electron integrals against PySCF to 10⁻¹² and
SCF energies to 10⁻⁹ (RHF, UHF, ROHF) on molecules with d functions
(H₂O and NH₃ in cc-pVDZ and 6-31G*, HCl and H₂S in cc-pVDZ, the triplet
O₂ in ROHF); CASCI energies in small active spaces against PySCF; the
existing STO-3G and 6-31G references unchanged.

| Deliverable | Delivered | Not done, and why |
|---|---|---|
| Integrals | McMurchie–Davidson for shells up to g (Hermite tables sized at run time); repulsion integrals per quartet of shells with the columns of a general contraction together; spherical shells (real solid harmonics in PySCF's order and signs, by a sparse transformation of the Cartesian integrals) or Cartesian ones (each component normalised to one); `Harmonics`, Python `harmonics=`, `qso scf --cartesian` | screening: every integral is still computed; at the 64-function limit that takes seconds |
| Basis sets | STO-3G, 6-31G, 6-31G* and cc-pVDZ for H to Ar, generated by `scripts/generate_basis_data.py` from PySCF's copy of the Basis Set Exchange | — |
| Open shells | `rohf` (Python `restricted_open=True`, `qso scf --restricted-open`) with PySCF's effective Fock matrix; its orbitals give the active-space Hamiltonian and FCIDUMP of open shells | — |

Validation as planned, against PySCF 2.14 with the geometries in bohr:

- **Integrals.** Every integral of H₂O, NH₃, CO and F₂ in all four bases,
  of HCl, H₂S, SiH₄ and NaF, and of two molecules with Cartesian shells,
  agrees to 2.6·10⁻¹³. f and g shells (a custom basis on OH⁻) agree in
  PySCF's fingerprints to 10⁻¹⁰ and in single integrals to 10⁻¹³.
- **SCF energies.** RHF of H₂O, NH₃, HCl and H₂S with d functions, UHF
  and ROHF of triplet O₂ and ROHF of NH₂ agree to 10⁻⁹ Ha in the tests,
  and to 2.4·10⁻¹⁰ Ha in the cross-validation.
- **CASCI.** In ROHF orbitals (O₂ with 12 electrons in 8 orbitals, NH₂,
  OH) and in RHF orbitals, CASCI agrees to 10⁻⁹ Ha.
- **Old references.** The STO-3G and 6-31G references are unchanged.

An independent review compared every integral of each element H to Ar
(paired with H at random low-symmetry geometries) in all four bases and
both harmonics, and of low-symmetry polyatomics, charged species and
extreme distances, with PySCF: 3.6·10⁻¹³ at most. It also compared f and
g shells elementwise and ROHF in 21 open shells (quartets to a septet, to
1.5·10⁻¹² Ha), checked CASCI with many frozen and active spaces, and
regenerated the basis data byte for byte. It found:

- **Memory.** Peak memory was 2.3 times what was needed (743 MB at 64
  functions): every quarter transformation was kept alive. It is now at
  most two tensors (335 MB).
- **Speed.** Third-row atoms were slow: 37 s at 64 functions, because each
  quartet of Cartesian components recomputed the Hermite table, and the
  columns of a general contraction repeated their primitives. Repulsion
  integrals now go per quartet of shells, with those columns together:
  3.3 s.
- **Unbacked figures.** The timings and the "f and g included" accuracy
  had nothing behind them; they now come from `benchmark_report` and from
  PySCF fingerprints in a test.
- **Citations.** 6-31G and cc-pVDZ (He, Na, Mg) were cited incompletely.
- **Other fixes:**
  - the 6-31G* Cartesian convention and stale RHF/UHF-only wording;
  - limits made crate-private and a debug assertion made a check;
  - the Boys function's asymptotic form kept to where it is exact for
    every order;
  - Python harmonics case-insensitive;
  - missing tests (closed-shell ROHF equals RHF, Cartesian third-row ROHF
    and level shift).

Cost (`benchmark_report`, development machine, 4 cores):

- **Integrals.** H₂O in cc-pVDZ (24 functions) takes about 0.03 s, H₂S
  (28) about 0.1 s, and the 64-function limit with three chlorines in
  cc-pVDZ about 3 s and 0.35 GB.

### 1.24 — Short simulations at full speed (S–M, delivered)

Use case: VQE, QAOA and sampling loops that run many short simulations,
each after a pause (an optimiser step, Python code). After a warm-up QSO
is ahead at 16 qubits in the latest cross-validation (one layer in 1.3 ms,
Qulacs 1.5 ms), but rayon's workers sleep as soon as they are idle, so on
virtual machines the first simulation after a pause pays for waking them
(section 1, thread wake-up); that cost is not measured yet.

| Deliverable | Detail |
|---|---|
| Worker pool | a pool for the state-vector kernels on the standard library's threads that spins briefly before parking, with the parallel thresholds measured again; rayon stays for the rest of the crate |
| Cold timings | the first simulation after an idle period timed in `benchmark_report` and in the cross-validation, beside the warmed-up timings |

Validation: cross-framework timings at 10 to 18 qubits, cold and warm,
against Qulacs and Qiskit Aer on the same machine; no slowdown at 20 to 26
qubits; results bit for bit unchanged (differential tests); the pool under
Miri for data races.

| Deliverable | Delivered | Not done, and why |
|---|---|---|
| Worker pool | `pool` (crate-private): standard-library threads that spin for 200 µs after a job before parking (`QSO_POOL_SPIN_US`), yielding the core regularly while they wait; one job at a time with the caller working too; calls from inside a job or while another job runs run serially, with the same panic contract; thread count from `QSO_NUM_THREADS` or `RAYON_NUM_THREADS`. Every state-vector helper of `parallel` (gate kernels, cache blocks, sums, and the maps behind `StateVector::zero` and `probabilities`) runs on it; the parallel threshold stays at 2¹⁶ amplitudes after measuring 2¹⁴ and 2¹⁵ again | rayon stays for batches, trajectories, restarts and the rest of the crate, as planned; inside rayon tasks the helpers keep using rayon |
| Cold timings | `benchmark_report` times one layer at 12–18 qubits warm and after pauses of 0.5 and 20 ms; the cross-validation has the same configurations (`1@500`, `1@20000`) for QSO, Qulacs and Aer at 10–18 qubits | — |

Validation as planned, on the development machine (4 cores, a VM):

- **Bit for bit.** Final states of random 16–22-qubit circuits hash the
  same with the old rayon build, with the pool, and with the pool on one
  thread; the differential and library tests pass unchanged.
- **Miri.** The pool's tests (many jobs, nesting, concurrent callers,
  panics), the plan tests and the unsafe kernels, run on the pool, pass
  under Miri with two workers (CI's filter, 16 tests).
- **Parameters.** Lower thresholds were not faster: 14 qubits in parallel
  took 380–400 µs a layer against 230–280 µs serial, and with 2¹⁵ a
  15-qubit layer took 0.36–0.44 ms warm and 0.52–0.54 ms after 20 ms,
  as with 2¹⁶ (0.36–0.47 and 0.53–0.54 ms). Without spinning a layer took
  1.3–1.7 times as long at 16 and 17 qubits, warm too (the workers
  parked between gates); 1 ms of spinning was not measurably better than
  200 µs.
- **Against the old build** (Rust, one layer, three interleaved runs,
  the range of their medians):

  | qubits | warm, old | warm, pool | after 20 ms, old | after 20 ms, pool |
  |---:|---:|---:|---:|---:|
  | 14 | 0.21–0.27 ms | 0.25–0.27 ms | 0.32–0.33 ms | 0.32–0.34 ms |
  | 15 | 0.43–0.54 ms | 0.36–0.47 ms | 0.78–0.83 ms | 0.53–0.54 ms |
  | 16 | 0.76–0.87 ms | 0.50–0.75 ms | 1.22–1.34 ms | 0.71–0.73 ms |
  | 17 | 1.43–1.62 ms | 0.87–1.04 ms | 1.90–1.98 ms | 1.21–1.28 ms |
  | 18 | 2.3–2.9 ms | 2.0–2.4 ms | 3.30–3.42 ms | 2.30–2.77 ms |

  At 14 qubits (one thread in both) they are equal. From 20 to 26
  qubits the pool is not slower (22 qubits: 36–39 ms old, 37–39 ms pool;
  26 qubits: 0.98–1.02 s and 0.95–1.02 s a layer).
- **Against Qulacs and Aer** (Python, a fresh process per measurement,
  three interleaved rounds of 21 runs, medians):

  | qubits | QSO warm | after 0.5 ms | after 20 ms | Qulacs warm | after 0.5 ms | after 20 ms | Aer after 20 ms |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 10 | 10 µs | 25 µs | 93 µs | 53 µs | 85 µs | 159 µs | 2.2 ms |
  | 12 | 55 µs | 80 µs | 170 µs | 139 µs | 187 µs | 464 µs | 2.9 ms |
  | 14 | 0.31 ms | 0.33 ms | 0.41 ms | 0.39 ms | 0.43 ms | 0.68 ms | 4.5 ms |
  | 16 | 0.51 ms | 0.57 ms | 0.77 ms | 1.19 ms | 0.90 ms | 1.42 ms | 7.8 ms |
  | 18 | 2.21 ms | 2.33 ms | 2.78 ms | 4.04 ms | 3.93 ms | 5.50 ms | 18.0 ms |

  The times after a pause include Python waking up from it, for every
  framework; QSO is ahead of Qulacs cold and warm at every size. The
  configurations are those of the cross-validation (`1`, `1@500`,
  `1@20000`).

An independent review checked the pool's memory ordering and lifetime
argument (sound), stress-tested it (6 threads × 3000 random jobs, calls
from rayon and plain threads at once, 5·10⁷ tasks, nesting), ran Miri
with 16 seeds, compared state hashes of the old build and of the pool on
1–4 threads at 15–22 qubits (identical, single precision, gradients,
sampling and trajectories included) and checked fork safety (the child
runs serially; the old build hung). It found:

- **A panic disabled the pool.** The lock of the running job was still
  held while the task's panic unwound, which poisoned it, and every later
  job of the process ran serially (under Miri, every test after the
  panic test too). The lock is now released first, and a poisoned lock
  is taken.
- **Panics in serial runs.** When a job ran serially (pool busy, one
  core), a panic stopped the remaining tasks, against the documented
  contract; the test failed with `RAYON_NUM_THREADS=1`. Serial runs now
  finish every task first, and a second payload is never dropped (a
  panicking `Drop` escaped and aborted).
- **Shared machines.** Two QSO processes at once lost 25–40 % of their
  throughput: the caller spun on workers that the scheduler had
  preempted. Waits now yield the core every 64 steps; two processes then
  ran 2000–2170 16-qubit layers per second, against 1540–1710 with rayon
  and about 1200 before the fix.
- **Rayon still woken.** `StateVector::zero`, `probabilities` and other
  maps still ran on rayon, so a simulation from scratch after a pause
  paid for waking it. They run on the pool now.
- **Smaller fixes:** workers that start after the first job join it;
  threads that cannot be spawned are done without; `QSO_NUM_THREADS=0`
  falls back to `RAYON_NUM_THREADS`; Miri's kernel test runs on the pool
  rather than inside rayon (whose fallback is safe code); the guide
  documents both variables and the CPU time spinning costs (about 40 %
  more with a layer every millisecond); stale 16-qubit statements; the
  warm Python column now comes from the cross-validation.

### 1.25 — Documentation and reach (S–M, delivered)

Use case: users of released versions, Rust users, and the browser.

| Deliverable | Detail |
|---|---|
| Versioned documentation | the guide and the Python reference of every release on the site, with a version selector, built from the release tags |
| Rust tutorials | the examples as tutorial pages of the guide, compiled and run in CI |
| MPS in the browser | matrix-product states in the WebAssembly build, its JSON interface and the playground |
| Coverage | the CLI and the Rust code run by the Python tests in the coverage report |

Validation: the documentation build produces every released version; the
tutorials compile and run on every change; the browser build's MPS gives
the native results on the npm package's tests; the coverage report lists
the CLI's lines.

| Deliverable | Delivered | Not done, and why |
|---|---|---|
| Versioned documentation | `docs/build_site.py --versions`: the guide and Python reference of every release since 1.2.0 at `/<version>/`, built from the release's tag with its wheel from PyPI (its notebooks executed against it), cached per release in the docs workflow, which also runs after a release; a version selector on every guide page; release pages link the playground and their tag's sources | 1.0 and 1.1 had no guide |
| Rust tutorials | eight guide pages from the ten examples: code between `// ANCHOR:` lines, explained, and the output of running the examples in the documentation build (about 140 s) | — |
| MPS in the browser | the WebAssembly module's `mps` method and npm `qso.mps()` (expectation values, bond dimensions and entropies, amplitudes, fidelity bound, samples); a playground tab with GHZ on 1000 qubits, a quench on 100 and a truncated brick wall | — |
| Coverage | `tests/cli.rs` runs every subcommand of `qso` as a program; `scripts/coverage.sh` merges the crate's tests, the CLI and the Python tests (on an instrumented wheel) into one report, with lines per component in the CI summary | — |

Validation as planned:

- **Every release.** 1.2.0, 1.4.0, 1.5.1 and 1.6.0 were built from their
  tags with today's mdBook; their notebooks ran against their own wheels
  (the output shows the version). The version selector was tested in
  Chromium on a site served under `/qso/`: the current version on every
  page and depth, the same page in another version, the start page where
  a page is missing, query and anchor kept, keyboard use.
- **Tutorials.** Every statement of the eight pages was checked against
  the code and the printed output (the review found one error, below).
- **Native results in the browser.** The npm tests replay four requests
  (40 qubits with truncation, 8 with amplitudes, a 300-qubit GHZ state
  with samples, a program with mid-circuit measurements) whose native
  responses are recorded in `js/test/mps_native.json`, and agree to
  10⁻¹⁰; a Rust test keeps the recording equal to the native build.
- **Coverage.** The merged report covers 92.0 % of lines and 81.5 % of
  branches: the library 93.9 % of its lines (the Python tests add
  0.9 points), the command line 78.0 % (before: counted as uncovered),
  the Python bindings 83.5 %.

An independent review built 1.4.0 from its tag, checked the eight
tutorials against their output, the selector in Chromium, the `mps`
method's limits and conventions, the workflows and the coverage flow. It
found:

- **Coverage build directory.** With cargo-llvm-cov 0.9 the script built
  into the normal `target/` (cleaning it, and racing with other builds
  of the CLI binary); it now uses its own directory, reports even when a
  test fails, and fails at the end. Running it also showed that maturin
  called stable cargo with nightly flags, and that `maturin develop` put
  the instrumented module into the source tree (300 times slower
  simulations afterwards); it uses nightly throughout and installs a
  wheel into its own environment.
- **`mps` conventions.** Counts were keyed by qubits even when the
  program declares classical bits; they now follow `qasm`. Samples had
  no output limit (16 384 qubits × 16 000 shots took a minute and 262 MB);
  they are capped at 2²⁶ bits like the stabilizer path. Programs without
  qubits, empty observables and measured programs without bits are
  refused.
- **Release documentation.** A tag pushed before its wheel reached PyPI
  failed the whole build; such a release is now skipped with a warning
  and tried again. Release guides linked a playground that did not exist
  under `/<version>/`, their 404 pages lacked the site URL, and their
  "edit" links pointed to `main`.
- **Selector.** The arrow keys switched versions at once; the keyboard
  now steps through them and switches on Enter. Query and anchor were
  dropped.
- **Smaller fixes:** the QAOA light cone in the MPS tutorial (2p + 2
  nodes, not 2p + 1); the validation page's coverage section; the
  introduction (Rust tutorials, version paths); stale playground results
  after an error; a misnamed npm test; CLI tests leaving directories in
  the temporary folder and a loose assertion; release caches keyed by a
  build number instead of the whole script.

### Not in this stage

Tight commutator bounds for product formulas of order four and higher,
leakage and non-Markovian noise, channels on more than two qubits, and
pretty good state transfer on graphs outside the cyclotomic families
remain gaps; they have fewer users than the four above or no exact
reference yet.

## 4f. Sixth stage (1.26 to 1.28)

Planned October 2026. The GPU research track asked for a use case; this
is it, and the stage serves it on the CPU first.

**Use case: variational training at 26 to 28 qubits on one machine.**
QAOA for MaxCut with p = 10 or a VQE ansatz of similar depth on 26–28
qubits, optimised with exact gradients for a few hundred steps, on a
workstation or cloud VM. Measured before the stage on the 4-core test VM
(double precision, one call each): QAOA on random 3-regular graphs
(20 parameters) as a generic circuit through
`gradients::expectation_and_gradient`, and through `MaxCutQaoa`, whose
adjoint pass is specialised to QAOA (phase tables, since 1.4):

| qubits | gates | generic: energy | energy and gradient | `MaxCutQaoa`: energy | energy and gradient | peak memory (generic) |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 520 | 0.16 s | 1.7 s | 0.07 s | 0.29 s | 54 MB |
| 22 | 572 | 0.68 s | 10.0 s | 0.36 s | 1.7 s | 0.20 GB |
| 24 | 624 | 4.4 s | 47.7 s | 1.8 s | 8.5 s | 0.79 GB |
| 26 | 676 | 18.4 s | 221 s | 12.7 s | 37.4 s | 3.15 GB |

A VQE ansatz (EfficientSU2, 4 layers, linear entanglement, transverse-field
Ising Hamiltonian) has no specialised path: at 24 and 26 qubits its energy
and gradient took 13.9 s and 73.4 s, 7–8.5 energies. A step at 26 qubits
takes between half a minute and four minutes, and 28 qubits need about
12.6 GB in double precision, more than such a machine has to spare. A GPU
would help, but none is available to test on (not on this machine, not
in headless Chromium's WebGPU, not on GitHub's standard runners), and the
rule "no claim without a test" holds for a backend too. The measurements
show where the CPU loses instead:

- **The generic gradient costs 7–13 energies.** The backward pass copies the
  whole state for every parametrised gate outside a cache block, applies
  the derivative to the copy and takes an inner product: about five
  passes over memory per gate instead of one. Almost every QAOA gate is
  parametrised. For a gate `exp(−iθG/2)` with a Pauli generator `G` the
  term is a matrix element of `G` between the two states, one reading
  pass without a copy.
- **The energy costs twice its gates.** The MaxCut cost is a sum of `ZZ`
  terms, which can be evaluated in one pass over the state.
- **Memory.** Three state vectors in double precision; single precision
  halves them, and 1.20 left gradients in double precision.

The rules stay: no new dependency, an independent reference for every
deliverable, results bit for bit independent of the number of threads.

### 1.26 — Gradients at the cost of a few simulations (M, delivered)

| Deliverable | Detail |
|---|---|
| Matrix elements | the backward pass computes the gradient terms of Pauli-rotation gates (controlled too) and phase gates as matrix elements of their generators between the two states, without copying a state; gates on high qubits undone and differentiated in one pass where the kernels allow |
| Diagonal observables | sums of `Z` strings (MaxCut, Ising, QUBO costs) evaluated and applied in one pass over the state, for expectation values and for the backward pass |
| Shared parameters | runs of commuting diagonal gates with the same parameter (a QAOA cost layer) undone and differentiated together |

Validation: gradients equal to parameter-shift gradients to 10⁻¹⁰ and to
the old adjoint pass to 10⁻¹² on random circuits over the whole gate set
(controls, shared and scaled parameters, every observable kind); results
independent of the thread count; the table above measured again, with the
gradient at no more than about four energies.

| Deliverable | Delivered | Not done, and why |
|---|---|---|
| Matrix elements | `kernels::derivative_element`: the gradient term of every parametrised gate as `⟨λ|D|φ⟩` in one reading pass, without the copy of the state (uncontrolled single-qubit gates over contiguous runs, bit for bit the pairwise sum); in cache blocks and over the whole register | undoing and differentiating a gate on a high qubit in one pass: it trades memory passes for scalar complex arithmetic and was not expected to pay without vector code |
| Diagonal observables | expectation values take all diagonal terms (Z strings, identity) in one pass; `H·ψ` of a diagonal sum is one multiplication per amplitude | — |
| Shared parameters | diagonal gates on qubits above the cache block are undone and differentiated block by block (per block a phase, or a diagonal gate on their one low target), so a QAOA cost layer no longer costs three passes per gate | one pass for a whole run of diagonal gates with one parameter: `MaxCutQaoa` already does it with phase tables, and in generic circuits the blocked run removed most of the cost |

Validation as planned:

- **References.** On random 7-qubit circuits over the gates with shift
  rules, with diagonal gates (controlled from inside and above the
  block) across the block boundary and shared and scaled parameters, the
  gradients equal the parameter-shift gradients to 10⁻¹⁰ and the
  unblocked pass to 10⁻¹², for diagonal and general observables; the
  existing adjoint, parameter-shift and finite-difference tests pass.
- **Threads.** Energies and gradients of a 20-qubit VQE and QAOA circuit
  are bit for bit the same on 1, 2, 3 and 4 threads and without spinning.
- **Speed and memory** (4-core VM, one call each):

  | qubits | VQE energy | VQE gradient, before | after | QAOA (generic) gradient, before | after | peak memory, before | after |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 20 | 0.10 s | — | 0.44 s | 1.7 s | 1.2 s | 54 MB | 36 MB |
  | 22 | 0.43 s | — | 1.9 s | 10.0 s | 4.9 s | 0.20 GB | 0.13 GB |
  | 24 | 2.1 s | 13.9 s | 9.2 s | 47.7 s | 21.1 s | 0.79 GB | 0.53 GB |
  | 26 | 9.4 s | 73.4 s | 41.6 s | 221 s | 89.1 s | 3.15 GB | 2.10 GB |

  The gradient now costs 4.4–4.6 energies for the VQE ansatz and 4.6–5.7
  for QAOA as a generic circuit (the energies themselves did not change
  much), short of the planned "about four": the forward simulation is
  compute-bound in the cache blocks (QAOA's RZZ phases), and gates on
  high qubits still take three passes in
  the backward pass. `MaxCutQaoa`, which 1.26 does not change, stays the
  faster way to QAOA (34.8–37.4 s at 26 qubits in two runs), and the
  memory is one state less (two instead of three).

An independent review found no error in the new code: against an
independent reference (a plain pass that copies the state for every
derivative) on 540 random circuits of 6–10 qubits with every block size
from 1 to 32 amplitudes, the whole gate set with up to two controls,
shared and scaled parameters and every kind of observable, the gradients
agree to 2.8·10⁻¹⁶, and against the previous build to 1.5·10⁻¹⁴ at 16–18
qubits; they are bit for bit the same on 1–4 threads, also in single
precision. Its tests are part of the suite now. It found:

- **Slower energies.** The one-pass diagonal expectation looped over all
  terms for every amplitude and was compute-bound: 1.5 times slower than
  the per-term passes at 20–22 qubits. It now goes chunk by chunk, every
  term over a chunk of probabilities in cache: for the QAOA cost 2–4 times
  faster than before 1.26 (24 qubits: 0.24 s instead of 0.97 s), for the
  VQE Hamiltonian as fast or faster.
- **Docs.** ARCHITECTURE.md still described three vectors and blocking by
  targets only; two `MaxCutQaoa` timings disagreed; the changelog did not
  say that energies change by rounding too.
- **Small savings:** phases that are one and derivatives that are zero
  (Z, S, T and P where the target bit is 0) are skipped in the blocks.

### 1.27 — Variational training in single precision (M, delivered)

| Deliverable | Detail |
|---|---|
| Single-precision gradients | energies and adjoint gradients on `StateVector32`, with sums accumulated in double precision; `precision="single"` for expectation values, gradients and the VQE and QAOA optimisers in Rust, Python and the CLI |
| Accuracy | the error of single-precision energies and gradients measured against double precision and stated in the documentation |

Validation: single against double precision on random circuits and on
the use case at 20–24 qubits (energy and gradient errors, optimisation
trajectories that reach the same optimum); memory measured at 26 and 28
qubits.

| Deliverable | Delivered | Not done, and why |
|---|---|---|
| Single-precision gradients | the adjoint pass, the matrix elements, the blocked diagonal gates, the diagonal observables and the QAOA layer generic over the amplitude type; `gradients::expectation_and_gradient_single` and `expectation_single`, `Vqe::with_precision`, `MaxCutQaoa::with_precision`, `qaoa::optimize_layerwise_at`; `precision="single"` for `expectation_and_gradient`, `run_vqe`, `MaxCutQaoa` and `optimize_qaoa_layerwise` in Python; `--single` for `qso vqe` and `qso qaoa`. Energies and gradients are divided by the state's norm², which rounding moves away from one, and the observable's constant term stays out of the co-state; the optimisers are told the energies' rounding (`optimize::minimize_at`): the L-BFGS line search treats values within it as equal and accepts Hager and Zhang's approximate Wolfe condition there, and L-BFGS and Adam stop when the gradient reaches its rounding | parameter-shift and finite-difference gradients in single precision: their differences of nearby energies lose the digits single precision has, and the adjoint pass is the one the use case needs; batches (`expectation_batch`) stay in double precision |
| Accuracy | measured below and stated in the guide (gradients page) | — |

Validation as planned:

- **Random circuits** (7, 9 and 16 qubits, the whole gate set, every
  observable kind): energies agree with double precision to 10⁻⁵ and
  gradients to 10⁻⁴ of their largest component (the tests' bounds; the
  measured errors are two orders smaller).
- **The use case** at 20–24 qubits (one call each, relative errors; the
  VQE ansatz of 1.26, `MaxCutQaoa` with p = 10 on random 3-regular
  graphs):

  | qubits | VQE energy | VQE gradient | QAOA energy | QAOA gradient |
  |---:|---:|---:|---:|---:|
  | 20 | 3.1·10⁻⁸ | 6.0·10⁻⁷ | 3.2·10⁻⁹ | 9.4·10⁻⁷ |
  | 22 | 2.7·10⁻⁸ | 7.2·10⁻⁷ | 1.9·10⁻⁹ | 1.0·10⁻⁶ |
  | 24 | 8.0·10⁻⁹ | 6.7·10⁻⁷ | 3.0·10⁻⁹ | 1.1·10⁻⁶ |

- **Optimisation.** QAOA layer by layer to p = 6 on a 16-qubit 3-regular
  graph reaches the same approximation ratios to seven digits (0.7480050
  … 0.9500122) with 591 evaluations instead of 423; QAOA at p = 4 on
  12–16 qubits reaches the same ratios to 4·10⁻⁹ with at most 1.2 times
  the evaluations. VQE (Ising chain, `EfficientSU2` with 2–3 layers, 6–10
  qubits, 12 runs from the same random starts): 6 end where double
  precision does (to 3·10⁻⁸ relative); 6 end in another local minimum, 2
  lower and 4 higher, by 6·10⁻⁵ to 4·10⁻² (rounding perturbs every step;
  double precision started 10⁻⁷ away does the same in 2 of the 12), with
  1.1 times the evaluations in all. UCCSD for LiH (10 qubits, 16 794
  gates) ends 2·10⁻⁸ from double precision.
- **Memory and time** (4-core VM, 15 GB, one call each, peak resident
  memory, VmHWM in GiB):

  | qubits | case | single | double |
  |---:|---|---:|---:|
  | 26 | VQE energy and gradient | 32.8 s, 1.00 GiB | 39 s, 2.00 GiB |
  | 26 | `MaxCutQaoa` energy and gradient | 28.5 s, 1.13 GiB | 35.8 s, 2.13 GiB |
  | 28 | VQE energy and gradient | 154 s, 4.01 GiB | 235 s, 8.01 GiB |
  | 28 | `MaxCutQaoa` energy and gradient | 122 s, 4.50 GiB | 160 s, 8.51 GiB |

  Single precision halves the memory (from |0…0⟩; with an initial state
  `Vqe` keeps a double-precision copy of it) and saves a fifth to a third
  of the time. On this VM 28 qubits fit in double precision as well; on a
  machine with 8 GB only single precision runs them.

The measurements found one fault: with the optimisers' default
tolerances (relative decrease 10⁻¹², gradient 10⁻⁸), which
single-precision energies cannot resolve, the last line searches followed
rounding noise: twice the evaluations and no convergence reported (QAOA
at 12 qubits: 318 evaluations instead of 161). A floor of 10⁻⁸ on the
decrease, the first fix, did not survive the review (below).

An independent review checked the division by the norm on every path,
the error handling, that no sum is accumulated in single precision (none
is), bit-identical results on 1–4 threads (18 qubits: VQE and QAOA
energies, gradients and a QAOA optimisation), the builder interactions,
every optimiser in single precision, and the stubs. It found:

- **VQE stopped early.** The decrease floor of 10⁻⁸ stopped L-BFGS on
  slow plateaus up to 1.3·10⁻² above double precision (the same
  tolerance in double precision stops at the same energies), and Adam's
  gradient tolerance of 10⁻⁸ was never reached (not converged after 3000
  steps). The optimisers are now told the energies' rounding (10⁻⁷ of
  the value) instead: values within it count as equal in the line search,
  where Hager and Zhang's approximate Wolfe condition decides by the
  slope, and L-BFGS and Adam stop when the gradient is below 4·10⁻⁷·|f|
  or, for L-BFGS, 3·10⁻⁶ of the run's first gradient (the gradient's
  rounding grows with its scale; without that, QAOA at depth 5–6 wandered
  at its noise for four times the evaluations). A stall test over ten
  iterations, tried in between, stopped VQE on plateaus again.
- **Large constants.** The identity term of the observable entered the
  single-precision co-state `H·φ` and its rounding the gradient: at an
  offset of 10⁴, 1.4·10⁻³ relative error. It is now added to the energy
  afterwards (1.9·10⁻⁷ at any offset). The error that grows with depth
  remains (1.7·10⁻⁴ for the LiH UCCSD ansatz) and is documented.
- **Memory with an initial state.** With an initial state the peak is 40
  bytes per amplitude against 48 in double precision, not half: `Vqe`
  keeps the double-precision initial state beside the rounded one (which
  it rounded at every evaluation; now once). Documented; the use case
  starts from |0…0⟩.
- **Small inconsistencies:** a zero or NaN state gave 0 from
  `expectation_single` but an error from the gradient (both are errors
  now); coefficients beyond the range of single precision gave `inf` and
  NaN (an error now); Python refused gradient-free methods with a
  parameter-shift gradient in single precision, which Rust runs; the CLI's
  text output did not name the precision; the guide promised one more
  qubit, which `Vqe` and `MaxCutQaoa` do not allow (removed).
- **Left as they are:** `MaxCutQaoa::state` in single precision holds both
  vectors while widening; reported single-precision energies carry their
  rounding and may lie a few 10⁻⁸ beyond an exact bound (documented);
  `qso qaoa --single` keeps its JSON a list of results without a
  precision key.

Tests for each: the plateau case (6 qubits, single within 10⁻⁶ of
double), Adam with an initial state (converged, energies within 10⁻⁵,
fewer than twice the evaluations), the offset of 10⁴, zero, NaN and
overflowing inputs, and the order of the builder calls.

### 1.28 — 28 qubits end to end (S–M, delivered)

| Deliverable | Detail |
|---|---|
| The use case | a QAOA optimisation (p = 10, a few hundred gradient steps) on 28 qubits on the 4-core test VM, timed and with its peak memory, as an example and in `benchmark_report` |
| Comparison | energy and gradient against Qulacs and Qiskit Aer at 22–26 qubits in the cross-validation (their gradient methods as they come) |
| GPU decision | with the CPU numbers in hand, the GPU track restated: what a GPU would add for this use case, and what it needs (a test runner with a GPU, a dependency) |

Validation: the 28-qubit optimum against the 20–24-qubit trend and the
energy of the best cut; the comparison on the same machine.

| Deliverable | Delivered | Not done, and why |
|---|---|---|
| The use case | `examples/qaoa_large.rs`: angles optimised layer by layer on a 16-node graph, carried over to 28 nodes (angles of random 3-regular graphs concentrate) and refined there with L-BFGS in single precision; `benchmark_report` times p = 10 at 22 and 24 qubits in both precisions. On the way, `MaxCutQaoa` became twice as fast (below) | a few hundred gradient steps at 28 qubits: 30 iterations took 36 minutes, a few hundred would take hours; with carried-over angles they are not needed. 28 qubits in `benchmark_report`: the weekly runner would spend minutes per run on it; 22 and 24 qubits track the same code |
| Comparison | `run.py --qaoa 22,24` (and 26 locally): energy and gradient of `MaxCutQaoa` and the generic circuit in both precisions, Qulacs' `backprop` and Aer's energy, each in its own process with time and peak memory, checked against tolerances | Aer's gradient: Aer has none, and a parameter-shift gradient of shared QAOA angles needs two energies per gate (1300 at 26 nodes) |
| GPU decision | below | — |

**Faster `MaxCutQaoa`.** Profiling the 24-qubit energy showed the mixer
compute-bound in the blocks (a complex 2×2 product per amplitude and
qubit) and the qubits above the block one pass each. Three changes, each
measured:

- **Y frame.** `e^{−iβX} = S†·e^{−iβY}·S` with the diagonal
  `S = diag(1, i)`, which commutes with the cost layers: the state evolves
  with real Y rotations from `S^{⊗n}|+⟩`, and only `state()` applies
  `S†^{⊗n}` (an exact phase). The real kernels need a quarter of the
  products.
- **Tiles** (`plan::Tiles`): the qubits above the block rotated in tiles of
  2⁵ runs of 2¹² amplitudes, five qubits per pass over memory. Runs of
  2⁷ in groups of seven (a block-sized tile) were slower than one pass
  per qubit (0.117 s against 0.046 s per layer at 24 qubits; prefetchers
  and the TLB do badly on 128 short runs), runs of 2¹² in groups of five
  took 0.029 s.
- **Fused backward pass.** `Σ Y` commutes with every rotation of the
  mixer, so its overlap is taken in the passes that undo them: per layer
  one tile pass per group and one block pass over φ and λ together, with
  the cost gradient and the phases in the block pass, and only imaginary
  parts summed (in double precision, four partial sums).

Energy and gradient at p = 10 (4-core VM, one call):

| qubits | double, before | after | single, before | after |
|---:|---:|---:|---:|---:|
| 24 | 8.55 s | 4.22 s | 6.58 s | 3.57 s |
| 26 | 35.8 s | 16.3 s | 28.5 s | 13.9 s |
| 28 | 160 s | 79.5 s | 122 s | 62.4 s |

The gradient now costs four to five and a half energies (26 qubits:
energy 4.3 s, energy and gradient 17.3 s in double precision; 2.5 s and
13.8 s in single); the rest is the scalar
overlap sums, which a kernel fused with the rotation could remove.
Tests: tiles against gate after gate bit for bit in both precisions,
their overlap against a direct sum, `MaxCutQaoa` above the block against
the explicit circuit (state, energy) and central differences (gradient)
in both precisions; the last runs under Miri too.

**The use case end to end** (`examples/qaoa_large.rs`, single precision,
random 3-regular graphs with seed 1, angles from 16 nodes, where layer by
layer to p = 10 reaches 0.97424 in 23 s; 30 L-BFGS iterations):

| nodes | maximum cut | ratio, carried over | refined | P(optimal cut) | evaluations | time | peak memory |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 26 | 0.97813 | 0.98046 | 0.62 | 37 | 6 s | 0.03 GiB |
| 22 | 29 | 0.97357 | 0.97498 | 0.56 | 38 | 32 s | 0.08 GiB |
| 24 | 32 | 0.96834 | 0.96917 | 0.35 | 40 | 128 s | 0.29 GiB |
| 26 | 35 | 0.97562 | 0.97602 | 0.49 | 38 | 526 s | 1.14 GiB |
| 28 | 38 | 0.96191 | 0.96635 | 0.61 | 36 | 2178 s | 4.51 GiB |

At 28 qubits ⟨C⟩ = 36.72 of the maximum cut 38 (found by enumeration of
the 2²⁸ cuts, 12 s), and a measurement gives one of the optimal cuts
with probability 0.61 (an optimal cut and its complement, 0.31 each).
The ratio is below those of the smaller graphs (0.969–0.980 refined,
0.968–0.978 carried over): the angles from 16 nodes fit this graph less
well, and 30 iterations recover only part of it.
None of the runs converged in 30 iterations. Optimising layer by layer on
the graph itself (two starts per depth) reached 0.97495 on 20 nodes
(440 s), below the carried-over and refined 0.98046, and 0.98086 on 22
nodes (1229 s), above its 0.97498: carried-over angles are a good start,
not the optimum, and both ways end in local optima.
The whole 28-qubit run took 37 minutes and
4.5 GiB; in double precision it would take 8.5 GiB and about 30 % longer.

**Comparison** (`run.py --qaoa`, same machine, one call each in a fresh
process):

| qubits | `MaxCutQaoa` double | single | generic circuit | Qulacs `backprop` | Aer, energy only |
|---:|---:|---:|---:|---:|---:|
| 22 | 0.83 s, 0.17 GiB | 0.68 s, 0.11 GiB | 4.5 s | 34.6 s, 0.35 GiB | 2.7 s |
| 24 | 3.65 s, 0.57 GiB | 3.13 s, 0.32 GiB | 19.2 s | 163 s, 1.29 GiB | 13.8 s |
| 26 | 16.4 s, 2.16 GiB | 13.3 s, 1.16 GiB | 80.9 s | 650 s, 5.04 GiB | 49.5 s |

In double precision, energies and gradients agree to 5.4·10⁻¹⁴ relative
(all four double-precision paths); QSO's single precision to 1.1·10⁻⁸
and 1.3·10⁻⁶. QSO's generic circuit is 5–6 times slower than its
specialised path and 8 times faster than Qulacs' `backprop` (which uses
all four cores); Aer takes
three times as long for the energy alone as `MaxCutQaoa` for energy and
gradient.

**The GPU decision, restated.** The use case runs on the CPU: 28 qubits
at one minute per gradient, an optimisation from carried-over angles in
under an hour, in 4.5 GiB. What a GPU would add: one QAOA layer at 26
qubits moves about 8.6 GB through memory (a block pass and three tile
passes, read and write), 0.14 s at the VM's 62 GB/s (the fastest pass
over the state on all four threads, BENCHMARKS.md), but takes 0.43 s:
the CPU is three times slower than its bandwidth, held back by the
arithmetic of the blocks. A GPU has both more bandwidth (300–1000 GB/s)
and far more arithmetic, so 5–15 times faster evaluations are plausible
and more possible (an estimate, not a measurement), and 28 qubits in
single precision fit in 8 GB of GPU memory. What it needs: hardware to test on (none here,
none on GitHub's standard runners) and a dependency, `wgpu`, whose
shading language has no double precision, so a WebGPU backend would be
single precision only, which 1.27 makes usable and measurable against
the CPU. The track stays parked until a test machine with a GPU is
available; the plan for it is then concrete: the QAOA layer (phase table,
Y rotations, tiles) and the fused backward pass as compute shaders, with
the CPU's single precision as the reference. Beyond 29 qubits memory
ends on a 16 GB machine in either precision, which a GPU does not change.

An independent review found no error in the code. It checked the Y
frame (conventions, initial state, the phases of `state()`, the gradient
in the frame and the overlap taken partway through the undone mixer)
against the explicit circuit at 3–21 qubits including odd sizes (states
to 8.4·10⁻¹⁶, gradients against central differences to 2·10⁻⁹),
bit-identical results on 1–4 threads at 15–22 qubits in both precisions,
the soundness of the tiles and the fused pass (disjoint tiles and runs,
no aliasing; Miri with tree borrows), and the conventions of the Qulacs
and Aer runners. It found claims the logs did not support, all
corrected: the 28-qubit memory was in units of 10⁶ KiB, not GiB (8.5 and
4.5 GiB, not 8.9 and 4.7; the same for the 1.27 table); the agreement of
the four double-precision paths is 5.4·10⁻¹⁴, not 2.4·10⁻¹⁴; the generic
circuit is 5–6 times slower than `MaxCutQaoa`, not 2.5–4; the 28-node
ratio lies below the smaller graphs', not within their spread; 0.61 is
the probability of all optimal cuts together, not of the most likely
string; Qulacs' `backprop` uses all cores; and the CPU runs at a third of
its bandwidth, not at it, which the GPU estimate now says. It also
suggested making the tiles' preconditions run-time and compile-time
assertions instead of debug assertions (done).

### Not in this stage

A GPU backend: it needs hardware to test on and a dependency (`wgpu`),
neither available now (1.28 restates the decision with the CPU numbers).
Multi-node simulation, which the use case does not need below 30 qubits.

## 4g. Seventh stage (1.29 to 1.33)

Planned October 2026, after release 1.7.0, from the gaps that remain in
the table of section 1. The stage starts with QSO's identity, walks on
graphs, then closes three gaps of the noise and physics rows, and ends
with spin dynamics for magnetic resonance (1.33, added on request). The
rules stay: no new dependency of the crate, an independent reference for
every deliverable, reference tools in the cross-validation only.

### 1.29 — Interacting walkers (M, delivered)

Use case: walks of several particles on a graph, the many-body side of
graph-structured quantum computing. This covers Bose– and Fermi–Hubbard
dynamics on lattices and graphs, two-particle interference (the
Hong–Ou–Mandel effect on a graph), and telling graphs apart where one
walker cannot. On a strongly regular graph one walker's propagator is a
combination of `I`, `A` and `J − I − A` with coefficients fixed by the
parameters, so strongly regular graphs with the same parameters look
alike to it. Gamble,
Friesen, Zhou, Joynt and Coppersmith (Phys. Rev. A 81, 052313 (2010))
found that two non-interacting particles still fail on strongly regular
graphs, while two interacting bosons distinguished every pair they
tested.

| Deliverable | Detail |
|---|---|
| Multi-particle walks | N bosons, fermions or distinguishable particles on a graph: hopping along the edges (adjacency or Laplacian generator, edge weights), on-site interaction U (Bose–Hubbard) and interaction V between neighbours; exact evolution in the occupation basis (dense below the dense limit, Krylov above), site densities, pair correlations `⟨n_i n_j⟩` and configuration probabilities |
| Graph certificates | the sorted two-particle transition probabilities at given times, as a graph invariant, with a comparison that tells graphs apart up to a stated tolerance |
| Graphs | Seidel switching, and the three Chang graphs (switchings of the triangular graph T(8)), so that four strongly regular graphs with the same parameters (28, 12, 6, 4) are at hand next to the Shrikhande and 4×4 rook graphs |
| Bindings | Python, CLI, guide page and tutorial |

Validation:

- **Non-interacting particles.** Amplitudes equal the permanents (bosons)
  and determinants (fermions) of submatrices of the single-walker
  propagator, using QSO's permanent code from boson sampling.
  Distinguishable particles equal the tensor product of single walkers.
- **Interacting particles.** The two-site Hubbard model against its
  closed-form spectrum. Small graphs against the full tensor-product
  space, with the interaction applied there and the result restricted to
  the symmetric or antisymmetric subspace.
- **Certificates.** Equal for isomorphic graphs under random
  relabellings. Without interaction they are equal for the Shrikhande
  and rook graphs and for T(8) and the Chang graphs (Gamble et al.).
  With interaction they tell all of them apart; the canonical labelling
  confirms that the graphs are not isomorphic.

Delivered as planned: `walks::InteractingWalk` (Python
`InteractingWalk`, CLI `qso particles`) with `certificate`,
`certificate_tolerance` and `certificate_distance`;
`generators::seidel_switching` and `generators::chang`; a section of the
walks chapter and the tutorial `08_interacting_walks`. Measured:

- **References.** Amplitudes equal the permanents and determinants of
  the single-walker propagator to 10⁻¹². With interaction, they equal the
  Hamiltonian of the tensor-product space restricted to the symmetric or
  antisymmetric subspace to 10⁻¹². That holds up to four particles on
  `K₅` with weights of both signs.
- **Krylov propagation** (2744 to 3003 configurations, with U and V)
  equals an independent Taylor integration to 10⁻¹⁰.
- **The Hubbard dimer** follows its closed-form spectrum.
- **Certificates.** Without interaction all pairs among `T(8)` and the
  Chang graphs, and the Shrikhande and rook's graphs, agree to 10⁻¹².
  With `U = 1` they differ by 1.7–3.4·10⁻⁴ and 5.4·10⁻³ respectively.
  The canonical labelling confirms that the graphs are not isomorphic,
  and their automorphism groups have orders 8!, 384, 96 and 360.

An independent review compared the code with its own references. It
built the Hamiltonian on the tensor-product space for 40 random graphs
of up to six nodes, with weights of both signs, up to four particles and
both generators. It ran a second-quantised Krylov builder against
SciPy, and checked the Chang graphs with networkx. It found no error in
the physics, the ranking or the documented numbers. It found:

- **No limits on the resources.** The particle count, the stored sites,
  the Hamiltonian's nonzeros and the length of a certificate had no
  bound, so some inputs aborted the process (100 000 bosons on an edge,
  two fermions on `K₁₅₀₀`). They are refused now, with checked
  arithmetic.
- **A fixed threshold for certificates.** 10⁻⁹ called a relabelled copy
  of a graph "not isomorphic" at `t = 10⁶`, because rounding grows with
  the time. The threshold is now `certificate_tolerance`,
  `10⁻¹² · max(1, |t|·‖H‖)`, and a test pins the relabelled rook's graph
  at `t = 10⁶` far below it.
- **Quadratic costs.** One search per rank when listing the
  configurations, now a successor function. One walker on 2²⁰ nodes
  takes 0.7 s, where the old code needed minutes. Two fermions on 2000
  nodes (2 million configurations) take 1.8 s with the Hamiltonian. Two eigendecompositions per walk (now one, on
  first use). The neighbour interaction summed over all edges for every
  configuration (now over pairs of occupied sites).
- **Seidel switching** ignored the edge limit of the generators and took
  `O(n²)` time. It now checks the limit first and keeps node positions.
- **Smaller faults.** NaN in a certificate distance read as "equal". The
  CLI printed `null` distances and "1 bosons", and failed by default with
  more bosons than nodes. The dense-limit error of walks spoke of qubits.
- **Docs.** The argument why one walker cannot tell strongly regular
  graphs apart was a non-sequitur; it is now that the propagator lies in
  the span of `I`, `A` and `J − I − A`. The Shrikhande pair was claimed
  for the validation suite, which now checks it.
- **Weak tests.** The tensor-product test now covers hops over three
  fermions and negative weights, and there is a Krylov test with
  interaction for fermions and distinguishable particles.

### 1.30 — Correlated and non-Markovian noise (M, delivered)

Use case: noise on devices that the channels of 1.4 cannot express.
That is crosstalk and correlated errors on more than two qubits, and
slow noise whose correlations last longer than a gate, such as 1/f flux
noise and charge noise switching between two values.

| Deliverable | Detail |
|---|---|
| Channels on more qubits | Kraus and Pauli channels on up to four qubits, in density matrices and trajectories, and in device models (correlated errors after a gate, crosstalk on spectator qubits) |
| Quasi-static noise | over-rotations and detunings drawn once per trajectory from a given distribution (Gaussian or uniform), constant during it |
| Telegraph noise | a frequency shift that switches between ±ν at a given rate, simulated exactly per trajectory (switching times drawn from the exponential distribution) |

Validation: channels on three and four qubits match the explicit
superoperator on density matrices and the average over trajectories.
Ramsey decay under Gaussian quasi-static noise equals `exp(−σ²t²/2)`,
and a Hahn echo removes it exactly. Telegraph noise matches the closed
forms of the decay of coherence for free evolution and the echo (Paladino,
Galperin, Falci and Altshuler, Rev. Mod. Phys. 86, 361 (2014)), in both
the fast- and the slow-switching regime.

Delivered as planned, with one addition: `Circuit::id`, which a rule for
the gate `"id"` turns into a wait of given duration. Spectator qubits
wait for the gate whose crosstalk they receive. Turns by slow noise
during a gate are split, half before and half after it. Density matrices
and PEC refuse slow noise instead of ignoring it. Measured:

- **Channels on three and four qubits.** On density matrices they equal
  `Σ K ρ K†` with explicitly embedded operators to 10⁻¹³, also for
  non-product, non-unitary channels. Trajectories (20 000) average to the
  density matrix within `4/√N` for general three- and four-qubit
  crosstalk channels.
- **Slow noise.** A Gaussian detuning follows `exp(−σ²t²/2)` and a uniform
  one `sin(at)/(at)`. The echo of a static detuning is exact in every
  trajectory. Telegraph noise follows the Kubo–Anderson decay and its
  echo for slow (γ = 0.2ν) and fast (γ = 5ν) switching. A quasi-static
  over-rotation gives `cos(θ(1+ε)) e^{−θ²σ²/2}`. All of these are within
  four standard errors at 8000 trajectories; the largest deviation of the
  20 validation estimates is 1.5.
- **Unchanged Markovian trajectories.** Devices without slow noise draw
  no extra random numbers, so existing trajectories and their tests are
  unchanged.

An independent review built its own hybrid master-equation model:
classical telegraph states times a density matrix, with T₁/T₂ Lindblad
terms, Gaussian detunings by Gauss–Hermite quadrature, and its own
scheduler. Against it, QSO's trajectories on two- and three-qubit
circuits (gate durations, waits, spectators, `idle_noise` on and off)
gave z-scores with mean about 0 and standard deviation about 1. The
same model reproduced QSO's Markovian density matrices to 5·10⁻¹⁶. The
review also compared every trajectory output of Markovian devices with
the previous build and found them identical bit for bit. It confirmed the
closed forms of the tests (also against the textbook form of the
telegraph echo) and found no error in the physics. It found:

- **A NoiseModel dropped channels silently.** A three-qubit channel after
  a two-qubit gate was ignored, and such a model could not be converted
  to a device. Both now report an error, and conversion works for
  circuits whose gates are large enough.
- **PEC refused three- and four-qubit Pauli noise.** It now inverts Pauli
  channels on up to four qubits; a test cancels ZZZ and XIX crosstalk.
- **`id` did not survive OpenQASM**, and the stabilizer simulator
  rejected it. `id q;` now reads and writes `Circuit::id`,
  `gate_counts` names it `"id"`, and the stabilizer simulator treats it as
  a Clifford no-op.
- **Composing four-qubit channels took 20 s** (all products of up to
  256 × 256 Kraus operators). On three and four qubits the composition
  now multiplies the superoperators and diagonalises the Choi matrix.
- **A pulse in the middle of an echo broke the echo.** The turn during a
  gate of finite duration was applied entirely after it, giving 0.948
  where the exact pulse gives 0.994. The turn is now split around the
  gate, and the echo refocuses exactly (test).
- **Smaller faults.** Rules without a known gate size refused large
  channels with a misleading message; they are now checked when a circuit
  is compiled. `DeviceBackend::density_matrix` accepted slow noise and
  failed only when run; it now fails when it is created. There were stale
  comments and a misplaced doc comment. The wide kernel duplicated the
  narrow one and is now one function with a buffer size as a constant
  parameter.
- **Docs.** Spectators also delay the gate; `is_markovian` looks at the
  whole device; telegraph noise costs time in proportion to its number
  of jumps. A claim in this section about an external check had no
  artifact in the repository and was removed.
- **Weak tests.** A test that only checked determinism now checks that
  Markovian programs draw no extra random numbers. The eight-target
  kernel test now covers several groups and a control. The four-qubit
  trajectory test now uses different damping per qubit. The Python test
  of channels on three and four qubits compares with explicit matrices.

Left as is: a four-qubit channel on a density matrix costs 16 times a
two-qubit one (applying few-Kraus channels as `K ρ K†` would be cheaper),
and slow-noise sampling scans the steps once per trajectory.

### 1.31 — Noise learning beyond ten qubits (M, delivered)

Use case: learning the sparse Pauli–Lindblad model of a layer on a
device with 20 to 100 qubits, as on hardware. 1.12 learns on density
matrices, which stop at ten qubits. Twirled noise on a Clifford layer is
a Pauli channel, and the learning circuits are Clifford circuits, so the
Pauli-frame sampler of 1.8 simulates them exactly at any width.

| Deliverable | Detail |
|---|---|
| Learning at scale | `learn_layer` on the frame sampler, for layers of `cx`, `cy`, `cz` and `swap` on any coupling graph, including the correlated two-qubit errors and crosstalk of 1.30 once twirled |
| Mitigation at scale | PEC and PEA with the learned model on circuits simulated by trajectories or MPS |

Validation: on a 4-qubit device the learned rates on the frame sampler
agree with those on density matrices within their standard errors. On a
heavy-hex layer of about 50 qubits all rates lie within three standard
errors of the exact rates of the twirled noise.

Delivered with two changes to the plan. First, the frames do not run on
the detector-circuit sampler of 1.8 but directly through the compiled
device program, so device rules, spectators and readout errors apply as
on density matrices. Second, the bound is four standard errors, not
three: with 681 rates, about two would lie beyond three by chance alone.
Measured:

- **Frames.** On four qubits, the outcome distribution of a learning
  circuit with depolarising, Pauli and crosstalk noise matches the
  density matrix within 1.5 standard errors per outcome (400 000 shots).
  Rates learned on frames and on density matrices differ by at most 2.0
  combined standard errors.
- **Heavy hex, 53 qubits.** 21 `cx` gates with two-qubit depolarising
  noise, a coherent phase error on the control and a ZZ crosstalk error
  with 14 bridge qubits, plus readout errors: all 681 rates lie within 4
  standard errors of the exact rates of the twirled noise. The largest
  deviation is 2.9; over six seeds the largest is 3.5. Over the 329
  nonzero rates the deviations have a spread of 0.94 standard errors, so
  the errors are honest. Most zero rates are fitted exactly as 0. Learning
  takes about 2 s.
- **Large fits.** Above 64 generators the non-negative least squares
  starts from coordinate descent on the normal equations and finishes
  with the active-set iteration on Cholesky factors. For the 681 rates
  it needs one factorisation; a collinear system falls back to the SVD
  path.
- **Mitigation at scale.** Samples run on density matrices (up to 10
  qubits), trajectories (up to 24) or matrix-product states (up to 64),
  each channel replaced by one of its unitaries. On 30 qubits, with a
  layer gate between qubits 2 and 9 and entangling blocks, PEC gives
  0.200 ± 0.004 for an exact value of 0.197 (raw 0.179), and PEA moves
  towards it. On 12 qubits PEC on trajectories gives the exact value
  within 4 standard errors; the raw value is 25 standard errors off.

An independent review checked the frame rules of every gate by hand,
the Pauli index conventions, the geometric sampling of errors and the
termination of the warm-started active set. It confirmed that results on
density matrices are unchanged bit for bit up to ten qubits, and that
learning matrices are well conditioned. It found:

- **Memory.** Frames of all shots of a task were held at once, up to
  0.8 GB per task at the shot limit; they now run in blocks of 4096 shots
  that update the parities directly. Trajectory samples of 20 to 24
  qubits ran on all threads at once (17 GB at 24 qubits on 64 threads);
  they now run one after another, as in the trajectory simulator.
- **Failures surfaced late.** A device that matrix-product samples cannot
  use, or a sample that `Auto` must refuse, failed only after every
  sample had run. The first failure now stops the others, and a strict
  matrix-product sample stops at the first truncation.
- **A test that missed its target.** The test of the fallback for
  collinear columns never reached it; it now uses a column the solution
  needs.
- **Edge cases of the error sampling.** The guard on the geometric gaps
  relied on a NaN that cannot occur; the gap now uses `ln(1 − p)` via
  `ln_1p` and stops on any non-finite gap, and a channel whose Pauli
  probabilities round to zero is skipped.
- **Docs and messages.** The condition for frames (the gates other than
  Paulis multiply to the identity) and for exact per-channel twirling
  (no overlapping non-Pauli channels between two twirls) are stated
  precisely; the truncation bias is stated per sample, before PEC's
  factor. Size errors of the mitigation named noise learning instead of
  the simulator, and the strict refusal advised a Rust-only option.
- **Python.** `max_bond` was silently ignored unless `simulator="mps"`,
  and a negative value raised `OverflowError`; both are now `ValueError`.
- **Tests.** New: every frame gate against its conjugation, PEC agreeing
  on density matrices, trajectories and matrix-product states, learning
  on 64 qubits, the choice of strict matrix-product samples above 24
  qubits, and entangling blocks with a distant layer gate at 30 qubits.
- **Smaller.** Matrix-product samples no longer re-check the unitarity
  of mixture terms that are unitary by construction, which could reject
  a term trajectories accept. The twirl is computed once per used
  channel.

Left as is: learned rates on 7 to 10 qubits now come from the new fit
and differ from 1.30 by about 10⁻¹⁷. A trajectory sample builds its
simulator for each sample. `LayeredSimulator` has no serde form, because
`MpsOptions` has none.

### 1.32 — Faster integrals and tighter bounds (S–M, delivered)

Use case: molecules at the size limit of 1.23, and step counts for
product formulas of order four and higher.

| Deliverable | Detail |
|---|---|
| Integral screening | Schwarz screening of the electron-repulsion integrals by shell quartets, with a stated threshold |
| Bounds of commutator type | the bound of Childs, Su, Tran, Wiebe and Zhu (Phys. Rev. X 11, 011020 (2021)) for orders 2k ≥ 4, from the nested commutators of order 2k + 1 computed in the Pauli algebra, alongside the bounds of 1.5 |

Validation: screened energies equal unscreened ones to the threshold and
PySCF's as in 1.23, with the time saved measured. The new bound exceeds
the exact error of small Heisenberg, Ising and random Hamiltonians at
every step count tested, falls as `r^{−2k}`, and is stated against the
old one.

Delivered with one change to the plan: instead of the explicit constant
of Childs et al., the bound uses their error representation,
`‖S(τ) − e^{−iHτ}‖ ≤ ∫₀^τ ‖T‖`, and computes the Taylor coefficients of
the error generator `T` exactly as Pauli operators: the nested
commutators of order p + 1 and higher, up to a degree that grows until a
rigorous multinomial tail is below 1 % of the rest. This is tighter than
a sum of commutator norms, and it applies to every order. Measured:

- **Screening.** Quartets of shells are skipped when their Schwarz bound
  is below 10⁻¹² Ha (by default). A chain of 32 hydrogens in 6-31G (64
  functions) skips 86 % of the quartets and takes 0.2 s instead of 1.5 s;
  its RHF energy moves by 5·10⁻¹³ Ha and agrees with PySCF to
  2·10⁻¹¹ Ha. Butane gains 5 %. Every skipped Cartesian integral lies
  below the threshold (test), spherical d functions within 0.98 of it
  (review), and all the PySCF comparisons of 1.15 and 1.23 pass with
  screening on.
- **Bounds.** On 4-qubit Heisenberg, Ising and random Hamiltonians, the
  new bound is never below the exact error, and from 8 to 32 steps it is
  7 to 11 times the error at order 4 and 8 to 24 times at order 6. It
  falls as `r^{−p}` (16 and 64 per doubling, within 25 %). The old
  stage-by-stage bound, which falls as `r⁻²`, is 780 to 10⁹ times the
  error there. `trotter_error_bound` returns the smallest bound, so no
  result got worse.

An independent review rederived the bound step by step (the error
representation, the Horner recursion, the factor 2 of the commutators,
the integration of the tail) and checked about 36 000 random cases
(2 to 5 qubits, coefficients up to 30, positive and negative times,
orders 1 to 8, up to 400 steps). None fell below the exact error beyond
the simulator's own rounding. It confirmed the Schwarz argument for
grouped contractions, the diagonal entries used, exact zeros and
bit-identical computed integrals. It found:

- **Units.** The series was built in the units of H and overflowed for
  coefficients beyond about 10⁷ to 10¹², losing the bound. It is now
  built for `H/λ` and evaluated at `λτ`; a test checks scales from 10⁻⁶
  to 10⁴⁰.
- **Cost.** When the series is out of budget, every call still spent
  most of the budget (0.15 to 0.35 s, a hundred times the old bound). The
  series does not depend on the steps or the time and is now cached per
  Hamiltonian and order (a second call takes no time, test).
- **Docs.** The agreement with the old bound at orders 1 and 2 holds
  only from about 16 steps; the factors (780, not 800), the cost, and the
  cases in which the nested bound is infinite are now stated as
  measured. The bound is tight for small Hamiltonians only: for 8 to 16
  spins its tail, which ignores which terms commute, dominates.
- **Smaller.** A test comment claimed that errors near rounding are left
  out; the error named `threshold` instead of Python's `screening`; a
  test asserted that a large series fails, which only held because of
  the current speed; reformatting of unrelated Python tests was undone.

Left as is: rounding in the vanishing coefficients sets a floor of about
10⁻¹⁵ per step, below which the Taylor bound takes over; a tail aware of
commutation would tighten the bound for larger Hamiltonians.

### 1.33 — Spin dynamics and magnetic resonance (M–L, delivered)

Use case: designing and understanding magnetic resonance experiments on
small molecules, above all hyperpolarisation. Hyperpolarised
[1-¹³C]pyruvate (dissolution DNP, Ardenkjær-Larsen et al., PNAS 100,
10158 (2003)) makes tumour metabolism visible in MRI, and was first used
in patients with prostate cancer by Nelson et al. (Sci. Transl. Med. 5,
198ra108 (2013)). Parahydrogen methods (PASADENA, Bowers and Weitekamp 1986–1987;
ALTADENA, Pravica and Weitekamp 1988; SABRE, Adams et al., Science 323,
1708 (2009))
polarise molecules chemically. Each method moves polarisation through a
network of coupled nuclear spins with pulses, fields and relaxation.
These dynamics are a quantum simulation in the literal sense: a spin-½
nucleus is a qubit, and a molecule's spin system has 2 to about 12 of
them.

QSO already has most of what this needs:

- density matrices up to 14 qubits and trajectories beyond;
- T₁/T₂ relaxation and Lindblad dynamics (1.4);
- time-dependent Hamiltonians with a fourth-order Magnus integrator
  (1.13);
- the Pauli algebra (1.5);
- the slow noise and echoes of 1.30.

The coupling network of a molecule is a weighted graph, so polarisation
transfer through it is state transfer on a graph (1.3, 1.18).

| Deliverable | Detail |
|---|---|
| Spin systems | nuclei by isotope (gyromagnetic ratios of ¹H, ²H, ¹³C, ¹⁵N, ¹⁹F, ³¹P), chemical shifts, scalar (J) couplings and the field; the Hamiltonian in the laboratory and in the rotating frame, with the coupling graph as a QSO graph |
| Pulses and sequences | hard and shaped pulses on chosen isotopes, delays and free evolution, phase cycles; free induction decay and spectra by Fourier transform |
| Relaxation | phenomenological T₁ and T₂ per spin, relaxation towards thermal equilibrium at a given field and temperature; cross-relaxation and Redfield theory left for later |
| Polarisation transfer | INEPT; parahydrogen singlet order turned into magnetisation (PASADENA, ALTADENA); SABRE transfer at level anti-crossings while the field is swept |
| Bindings | Python, CLI, a guide page and a tutorial on hyperpolarisation |

Validation:

- **Two spins in closed form.** Free induction decay, spin echo, and the
  INEPT gain of γ(¹H)/γ(¹³C) ≈ 3.98 over direct excitation.
- **Parahydrogen.** The PASADENA and ALTADENA line patterns of a two-spin
  product in closed form.
- **Level anti-crossings.** For the smallest SABRE models, at the fields
  given by the couplings and shifts.
- **Larger spin systems.** Cross-validated against QuTiP's master-equation
  solver, a reference in the cross-validation only, like PySCF and Stim.

Not part of the milestone: MRI image simulation (classical Bloch equations
per voxel, served by MRI simulators), quantum algorithms for CT, and any
clinical claim. QSO is a tool for physicists who design these
experiments, not medical software.

Delivered as planned, as the module `spin` (`SpinSystem`), with the
Python module `metatron_qso.spin`, `qso spins`, a guide page and the
tutorial `09_hyperpolarisation`. The Magnus integrator and the Pauli
algebra were not needed: `H` conserves `M` of every isotope in the
rotating frame and the total `M` in the laboratory frame, and the
Lindblad equation conserves the matching coherence orders, so free
evolution, spectra and time averages diagonalise each block of `H`, and
relaxation exponentiates each block of the superoperator exactly.
Measured:

- **Closed forms.** The FID of one spin `−(iP/2)e^{iΩt − t/T₂}` holds to
  10⁻¹⁴, the spin echo refocuses every offset and is modulated as
  `cos(2πJτ)` for coupled protons, inversion recovery follows
  `P(1 − 2e^{−t/T₁})`, and every state relaxes to the thermal one. The
  phase-cycled INEPT gain is `P_H/P_C` = 3.976 at 9.4 T to 10⁻⁸.
  PASADENA lines agree to 0.3 % and ALTADENA lines (a 20 s sweep to 1 T)
  to 0.2 %, the rest being strong coupling. SABRE transfer to a proton
  and to ¹⁵N (SABRE-SHEATH, laboratory frame) peaks within 2 % of the
  anti-crossing fields.
- **QuTiP.** Five systems built independently with QuTiP's spin
  operators, four of them at 9.4 T and 298 K, where polarisations are
  10⁻⁵: QSO agrees with the exact exponential of QuTiP's Liouvillian to
  7·10⁻¹² of the largest expectation in the rotating frame and 2·10⁻¹⁰ in
  the laboratory frame, and with `mesolve` to 4·10⁻⁶ and 2.5·10⁻⁴, the
  error of its adaptive steps.
- **Cost.** One second of five relaxing protons takes 0.2 s in either
  frame, of six about ten seconds; a free induction decay needs only the
  coherences of order −1. Eleven spins without relaxation take 1.8 s for
  a time average, 2.6 s for a spectrum and 2.7 s for a delay.

An independent review confirmed the gyromagnetic ratios, the operators
and Hamiltonians (against an independent construction), both frames,
the thermal state, pulses, shaped pulses and sweeps, the signal and
Fourier conventions, the parahydrogen state, the fixed points and the
exact T₁ and T₂ of the Lindblad operators for spin ½, and the closed
forms of INEPT, PASADENA, ALTADENA and the anti-crossings. It found:

- **Relaxation at real polarisations.** The first version integrated ρ
  by Dormand–Prince steps with an absolute error of 10⁻¹², while thermal
  states deviate from `1/d` by 10⁻⁶: four protons at 9.4 T after one
  second were 0.94 % off, and took 32 s. The laboratory frame and large
  systems were out of reach without a warning. Now the deviation is
  propagated, by block exponentials (10⁻¹¹ of it, against an independent
  full superoperator; 11 ms), or by Dormand–Prince steps with an error
  relative to it when those are cheaper; propagations estimated above
  1.5·10¹² operations are refused at once (test).
- **Time averages.** They kept coherences between different `M` of an
  isotope in the rotating frame, which precess at its Larmor frequency,
  so the result depended on the frame; and their tolerance, 10⁻⁹ of the
  largest energy, was 10 rad/s in the laboratory frame. Energies are now
  compared as in the laboratory frame, block by block, to the rounding
  of the eigendecomposition (tests in both frames).
- **Spectra.** Lines were merged within 10⁻⁶ Hz, finer than the
  precision of laboratory-frame energies, against the first line of a
  group, which split chains; and rounding produced lines of 10⁻¹⁷. Lines
  are now clustered by single linkage within the rounding of the
  energies, and those below 10⁻¹³ of the state's largest entry dropped.
- **Input.** `set_relaxation` stored invalid times before checking them,
  which made later assertions of a test vacuous; INEPT accepted J = 0,
  NaN and one isotope; events accepted unknown JSON fields; the thermal
  state overflowed at extreme fields and temperatures; Python took a bool
  for a spin index; `qso spins` showed lines at the starting field after
  a field change. All fixed, with tests; `field_after`,
  `with_temperature` and `carrier` were added to Python.
- **QuTiP.** The comparison used QSO's own Hamiltonian and operators,
  low temperatures and absolute differences. It now builds everything in
  QuTiP, runs at room temperature, and compares relative differences
  against both the exact Liouvillian exponential and `mesolve`.
- **Smaller.** `parahydrogen_state` multiplied dense matrices and every
  call diagonalised the whole `H` (2048 × 2048 at eleven spins); the FFT
  accumulated its twiddle factors; the Python bindings held the GIL. The
  docs now state the spin-1 relaxation correctly (T₁ and T₂ exact at high
  temperature, test), that `T₂ ≤ 2T₁` for every spin, the `(1 + δ)` of
  the Boltzmann factor, NaN shifts at zero field, the rotating-wave
  approximation of shaped pulses, and that laboratory-frame pulse phases
  refer to the laboratory axes.

Left as is: one evaluation of the Lindblad equation costs about 30 μs at
four spins, and Dormand–Prince needs 10⁵ steps of six per second at
9.4 T, so long shaped pulses with relaxation are slow; the laboratory frame at high field loses `ε·ωt` (2·10⁻¹⁰ after
a millisecond at 9.4 T), which is the conditioning of the problem.

### Not in this stage

Leakage out of the qubit space, which needs qutrits throughout the
simulators. Exact circuit distances of hypergraph error models. Pretty
good state transfer outside the cyclotomic families.

## 4h. Eighth stage (1.34 to 1.37): magnetic resonance for medical research

Planned October 2026, on request, after 1.33. QSO's strength in medicine
is the exact simulation of small quantum systems: nuclear spins in
hyperpolarised agents and the spins of quantum sensors. This stage takes
1.33 from textbook experiments to the questions of groups that develop
hyperpolarised MRI agents and quantum sensors for biology: how long
polarisation lives, how much of it reaches the molecule, which sequence
moves it best, and what a sensor near a cell detects. The rules of the
seventh stage stay. Every number has an independent reference, with
closed forms and published results first and QuTiP (including its
Bloch–Redfield solver) in the cross-validation only. The crate takes no
new dependency.

Not in this stage, as in 1.33: image simulation, the kinetics of
metabolism (pyruvate to lactate) and other classical models, quantum
algorithms for CT, which have no demonstrable advantage at the size of a
real image (loading and reading out 10⁶ to 10⁹ values undoes it), and
any clinical claim. QSO stays a tool for physicists and chemists who
design these experiments, not medical software.

### 1.34 — Relaxation from first principles (M, delivered)

Use case: predicting relaxation times instead of taking them as input.
For example, whether deuteration lengthens the T₁ of a ¹³C agent, or how
long a singlet state survives. 1.33 takes T₁ and T₂ as phenomenological
inputs.

| Deliverable | Detail |
|---|---|
| Interactions | dipolar couplings from the nuclear positions, chemical-shift anisotropy tensors, isotropic rotational diffusion with a correlation time (spectral densities `J(ω) = (2/5) τ_c/(1 + ω²τ_c²)`) |
| Redfield theory | the Bloch–Wangsness–Redfield relaxation superoperator (Redfield, IBM J. Res. Dev. 1, 19 (1957)), with cross-correlations between mechanisms, relaxing to thermal equilibrium at the given field and temperature |
| Use | relaxation of any state in `SpinSystem.evolve`, together with or instead of the phenomenological T₁ and T₂; relaxation rates of chosen observables (T₁, T₂, singlet order) as numbers |

Validation: two dipolar-coupled spins follow Solomon's equations (Phys.
Rev. 99, 559 (1955)). The steady-state nuclear Overhauser enhancement
of ¹³C under ¹H saturation is `1 + γ_H/(2γ_C) ≈ 2.99` in the extreme
narrowing limit. CSA relaxation grows with B². Singlet order of two
spins is immune to their mutual dipolar relaxation (Carravetta and Levitt,
J. Am. Chem. Soc. 126, 6228 (2004)). For three to five spins every rate
matches QuTiP's Bloch–Redfield solver (`brmesolve`) in the
cross-validation.

Delivered as planned in `spin/redfield.rs`: positions and shielding
tensors per spin, a rotational correlation time per system, the
non-secular Bloch–Wangsness–Redfield superoperator with all
cross-correlations (secular with respect to the carriers in the rotating
frame) relaxing towards the thermal state, in every propagation path
together with the T₁ and T₂ of 1.33, and `relaxation_rates` for any
observables (`qso spins --rates`). Writing the spatial tensors in an
orthonormal basis makes the isotropic average a sum of 30 independent
terms, built entry by entry for each coherence-order block. One change to
the plan: QuTiP's Bloch–Redfield tensor is compared directly (exactly
propagated) rather than through its `brmesolve` integrator, for two to
four spins. Measured:

- **Solomon and BPP.** Auto-, cross- and transverse rates of ¹H–¹H and
  ¹³C–¹H pairs at three correlation times, in both frames, to 10⁻⁹; the
  total rate of two equivalent protons to 10⁻⁹ of Bloembergen, Purcell and
  Pound.
- **Overhauser enhancement.** 2.988 for ¹³C under proton saturation in
  extreme narrowing (`1 + γ_H/(2γ_C)`); the transient after inverting the
  proton follows Solomon's equations to 10⁻⁷.
- **Shielding anisotropy.** `R₁` and `R₂` of a general tensor to 10⁻⁹,
  four times faster per doubling of the field; the interference with the
  dipolar coupling (the TROSY rate) to 10⁻⁹ of its closed form, with the
  sign of the shielding convention.
- **Singlet order** of two equivalent protons does not relax by their
  dipolar coupling (below 10⁻¹² of the magnetisation's rate) and lives on
  over twenty magnetisation lifetimes; a third proton or anisotropy
  relaxes it.
- **Equilibrium and deuteration.** Every state relaxes to the thermal one
  (exactly in the rotating frame, to `J/Δν` in the laboratory frame);
  deuteration slows the dipolar relaxation of carbon 15.9 times.
- **QuTiP.** With bath operators from an exact numerical average over
  orientations, QuTiP's `bloch_redfield_tensor` gives the same rates for a
  ¹³C–¹H₂, a ¹³CH₃ and an amide ¹⁵N–¹H with anisotropy to 4·10⁻¹⁵, and the
  same expectations after exact propagation to 3·10⁻⁹.
- **Cost.** Building the superoperator takes 0.04 s for five spins and
  0.5 s for six; one second of five relaxing protons with it costs
  0.2 s.

An independent review rederived the theory (the isotropic correlation,
the factor 1/5, the weights, the signs, the spectral densities at
laboratory frequencies) and wrote its own superoperator as a direct
Cartesian sum: QSO's agrees to 2·10⁻¹⁵ in six systems and both frames,
conserves the coherence orders exactly, and its dynamics agree in every
path (delays, FIDs, shaped pulses, Dormand–Prince steps, field jumps).
It found:

- **NaN after long delays.** Coherence blocks were shifted by their mean
  diagonal including the mean decay, so after about 10³ s (Redfield) or
  10⁴ s (laboratory frame, 1.33) `exp` overflowed while `e^{shift}`
  underflowed. The shift now carries only the mean precession (test: 20
  minutes and twice a singlet lifetime).
- **Validity and limits.** Any correlation time was accepted, so τ_c = 1 ms
  gave meaningless rates and extreme inputs NaN; the superoperator was
  rebuilt for every delay and shaped-pulse segment, outside the work
  limit (a 200-segment pulse on five spins took 13 s, against 0.6 s
  without Redfield relaxation). Now
  `|A|·τ_c > 1` and non-finite superoperators are refused, the build is
  counted, and `evolve` reuses it while the field stays (2.2 s).
- **Output.** Spins without relaxation printed `T1 -inf`; the rates are
  now never −0 and the CLI prints "none".
- **Tests.** Nothing pinned the sign of the shielding term relative to
  the dipolar one (the QuTiP comparison rebuilds the same Hamiltonian),
  and Redfield was not tested with FIDs, shaped pulses, Dormand–Prince
  steps, field jumps or the rotating-frame projection. All are tested
  now.
- **Docs.** The thermal correction was said to hold "at any
  temperature"; it is semiclassical, first order in `ħω/k_BT` away from
  equilibrium, and can give unphysical states near full polarisation.
  The guide now states this, the shielding convention (a chemical-shift
  tensor is `−σ`), the validity condition, the mechanisms not modelled
  (scalar, quadrupolar, spin-rotation, paramagnetic, intermolecular,
  exchange), the size limits, and that `--rates` prints auto-relaxation
  times; the tutorial's deuteration gain is stated for its field and
  model.

Left as is: the laboratory frame loses about `ε·Δω·t` over long delays
(3·10⁻⁵ after 1000 s at 9.4 T); each step of a field sweep rebuilds the
superoperator; Python accepts `True` as a number, as Python does.

### 1.35 — Hyperpolarisation in realistic models (M–L, delivered)

Use case: the chemistry and storage of hyperpolarisation. In SABRE,
substrate and parahydrogen bind and leave the catalyst all the time;
1.33 replaces this exchange by a time average. Parahydrogen-induced
polarisation by side-arm hydrogenation (PHIP-SAH, Reineri, Boi and Aime,
Nat. Commun. 6, 5858 (2015)) polarises [1-¹³C]pyruvate, the agent used
in patients. Long-lived states keep polarisation beyond T₁ on its way to
the patient.

| Deliverable | Detail |
|---|---|
| Chemical exchange | several chemical sites (free substrate, catalyst complex, free hydrogen) with rate constants, in Liouville space (Bloch–McConnell equations, McConnell, J. Chem. Phys. 28, 430 (1958)); SABRE polarisation as it builds up over time and with the field, temperature-dependent rates as input |
| Spin-order transfer | ¹H → ¹³C transfer for PHIP and PHIP-SAH by field cycling (Jóhannesson, Axelsson and Karlsson, C. R. Physique 5, 315 (2004)) and by pulse sequences (PH-INEPT and its relatives) |
| Long-lived states | singlet order of near-equivalent spins, M2S and S2M conversion (Pileio, Carravetta and Levitt, PNAS 107, 17135 (2010)), spin locking and field cycling for storage, lifetimes from 1.34 |

Validation: two-site exchange gives the Bloch–McConnell lineshapes, with
coalescence at `k = πΔν/√2`. The SABRE build-up of the smallest models
matches the published simulations of Lindale et al. (Nat. Commun. 10,
395 (2019)) and Pravdivtsev et al. (ChemPhysChem 14, 3327 (2013)) at
their fields. The field-cycling transfer of a three-spin PHIP system
matches its closed form. M2S and S2M reach the conversion efficiencies
of Pileio et al. The singlet lifetime of a near-equivalent pair exceeds
T₁ by the factor Redfield theory predicts.

Delivered in `spin/exchange.rs`: chemical sites (spin systems with
concentrations) and first-order channels that carry chosen spins with
their state, trace out the source's others and bring the target's others
in thermal or as parahydrogen (with a singlet fraction); the
Bloch–McConnell equations in Liouville space, propagated exactly block by
block of coherence order over all sites, with each site's `T₁`, `T₂` and
Redfield relaxation; pulses on isotopes, delays, field jumps and sweeps,
steady states and FIDs (`spin.Exchange` in Python). For storage and
transfer, `m2s`, `s2m` and `spin_lock`, and field cycling with the sweeps
of 1.33; tutorial 09 shows SABRE with exchange and field cycling. Changes
to the plan:

- **No published SABRE simulations reproduced.** Lindale et al. and
  Pravdivtsev et al. do not give every coupling and rate constant their
  figures need. Instead a Monte Carlo of exchange events (written with
  NumPy alone) and the position of the level anti-crossing test the
  exchange, and an independent dense implementation of the equations
  tests the bookkeeping. Lindale et al. sample event times uniformly and
  argue that ensemble-averaged equations fail in their regime; QSO's
  equations are exact for exponentially distributed events.
- **Not implemented:** PH-INEPT and its relatives as named sequences
  (they can be written from pulses and delays), the irreversible
  chemistry of PHIP-SAH (exchange needs steady amounts; the hydrogenated
  product is simulated as a spin system), helpers for
  temperature-dependent rates (rates are inputs), field cycling for
  singlet storage (shown with spin locks only).
- **References.** The field-cycling transfer is compared with
  sudden-adiabatic theory evaluated numerically, not a closed form; the
  singlet lifetime under a spin lock with the Redfield rate of `I₁·I₂`
  from 1.34 (validated against QuTiP there), not an independent theory.

Measured:

- **Bloch–McConnell.** One proton exchanging between two sites gives the
  closed-form FID to 10⁻¹⁰ and coalescence at `πΔν/√2` (two maxima at
  0.85 times that rate, one at 1.15). Renewal at rate `k` relaxes as
  `e^{−kt}`, slow renewal with parahydrogen tends to the time average
  of 1.33.
- **Dense reference.** An independent NumPy implementation of the
  homogeneous equations (partial traces by `einsum`, spin orders by axis
  permutations) agrees for three systems — three sites with different
  isotopes and channels that reorder spins, parahydrogen on spins that
  are not neighbours between carried ones, ²H in the laboratory frame —
  with delays, sweeps, FIDs and pulses on some of the sites to 10⁻¹⁰ and
  with steady states (rotating frame) to 10⁻⁹.
  SABRE's steady state in the laboratory frame from 1 mT to 9.4 T agrees
  with a 40-digit solution to 3·10⁻¹⁵.
- **Monte Carlo.** 4000 SABRE complexes with exponentially distributed
  hydrogen and ligand exchange agree with `Exchange` within their
  statistical error (z = 1.15, 0.27, −0.73).
- **SABRE.** The free ¹⁵N is polarised most within 3 % of the level
  anti-crossing (0.139 μT): 0.739 at `k_d = 2/s`, 0.832 at 10/s, 0.551
  at 40/s with the maximum moved to 0.172 μT; the build-up rises
  monotonically and reaches the steady state to 10⁻⁶ after 400 s.
- **Singlet order.** M2S and S2M return 99.5, 97.2 and 99.9 % of the
  2/3 bound for three near-equivalent pairs (inverted magnetisation).
  Under a 1 kHz spin lock the singlet order of a near-equivalent proton
  pair with a third proton decays at its Redfield rate to 4·10⁻⁴, a
  lifetime 17.5 times its magnetisation's; without the lock 0.515
  instead of 0.607 is left after half that lifetime.
- **Field cycling.** Parahydrogen order carried from zero field to
  50 μT puts −0.8127 polarisation on the ¹³C of an AA'X system, against
  −0.8128 from sudden-adiabatic theory; a sudden jump transfers none.
- **Cost.** For a SABRE complex of six spins with a free substrate of
  four in the laboratory frame (one block of 994 elements), the steady
  state takes 1.1 s and a one-second delay 16 s; a sweep of 20 steps is
  refused before it starts (7·10¹² operations).

An independent review rederived the model, wrote its own dense
Liouville-space implementation and compared four systems (with
permuting and self channels, channels at rate 0, parahydrogen pairs
reversed and interleaved, ²H, the laboratory frame and relaxation):
delays to 2·10⁻¹³, steady states and FIDs to rounding, sweeps to
2·10⁻¹². Redfield sites inside an exchange match single systems, the
M2S/S2M sequences match Pileio et al., and no levels of one total `M`
cross up to 50 μT, so the sudden-adiabatic mapping holds. It found:

- **Steady states refused.** The residual test was not scale-invariant:
  in the laboratory frame at 1 T and 9.4 T, where zero-quantum
  frequencies exceed the rates by 10⁸, valid and well-posed systems
  (smallest singular value 0.22) were refused, among them SABRE at 1 T.
  Now one population equation per site is replaced by its amount, rows
  and columns are scaled, and a fully pivoted LU solves the square
  system; singular systems show pivot ratios of 10⁻¹⁸, a `T₁` of 10⁶ s
  3·10⁻⁹ (threshold 10⁻¹²). It agrees with 40 digits and is eight times
  faster than the SVD before.
- **Cost.** Sweeps had no total limit (20 steps of the system above took
  255 s, Python's default of 1000 about 3.5 hours) and FIDs no check;
  both are now estimated before they start.
- **Tests.** No test exercised the parts of the index mapping where
  errors hide (strides, reordered and interleaved spins, parahydrogen on
  distant spins, isotopes of only some sites, spin 1, sweeps, pulses on
  some sites): the dense reference now does, and fails if the target's
  strides are replaced by the source's. Nothing tested steady states in
  the laboratory frame at high field (now from 0.1 to 9.4 T). The M2S
  test took the absolute value; S2M returns the magnetisation inverted,
  now asserted and documented. The spin-lock ratio was asserted only to
  exceed 3.
- **Docs.** "Without the lock, 15 % faster" meant 15 % less singlet
  order left; the guide claimed `Exchange` computes what Lindale et al.
  sample; "exchange keeps thermal states thermal" holds for equal shifts
  only; the README called numerical references closed forms; the
  tutorial's exchange times and fields were wrong. Missing caveats are
  now stated: steady chemistry, one source per channel (a partner's spins
  arrive fresh), fresh states as products of single-spin populations, the
  size limit, no shaped pulses during exchange, free-precession
  relaxation during a spin lock, the cost of sweeps and the precision
  next to parahydrogen order (10⁻¹⁵ absolute).
- **Python.** A singlet `fraction` without `parahydrogen` was ignored; it
  is an error now.

Left as is: the steady-chemistry requirement (irreversible chemistry
would make the amounts dynamic variables) and one source per channel;
every sweep step rebuilds the whole generator; long delays in the
laboratory frame lose `ε·ω·t` (10⁻⁹ after 600 s at 1 T); the Monte
Carlo test resolves errors above about 4 %, the dense reference the
rest.

### 1.36 — Optimal control of spin experiments (M, delivered)

Use case: designing the sequence, not only simulating it. Shaped pulses
and field profiles that move the most polarisation to ¹³C in the
shortest time, robust to errors in the radiofrequency field and the
offsets. 1.33 has shaped pulses, and QSO has exact gradients and
optimisers (1.26).

| Deliverable | Detail |
|---|---|
| GRAPE | gradient ascent in pulse engineering (Khaneja et al., J. Magn. Reson. 172, 296 (2005)): exact gradients of a transfer or of a target state with respect to piecewise-constant amplitudes and phases, with relaxation (Liouville space) and with ensembles of field and offset errors for robustness |
| Field profiles | the same for the field of SABRE and field-cycling experiments |
| Constraints | amplitude and power limits, a fixed duration, smooth shapes |

Validation: the gradients agree with finite differences. The optimised
transfer reaches the unitary bound (Glaser et al., Science 280, 421
(1998)): the largest `Tr(C U ρ U†)` over all unitaries, the sum of the
products of the ordered eigenvalues of `ρ` and `C`. For two heteronuclear
spins it follows the time-optimal limits of Khaneja, Brockett and Glaser
(Phys. Rev. A 63, 032308 (2001)): the antiphase transfer `2I_zS⁻ → I⁻`
reaches at most `sin(πJT/2)` in a time `T ≤ 1/J`, and the in-phase
transfer `S⁻ → I⁻` reaches one only at `3/(2J)`. Pulses optimised for a
range of field errors keep their transfer over that range, where INEPT
does not.

Delivered in `spin/control.rs`: GRAPE for piecewise-constant rf
amplitudes on several channels (rotating frame) or field profiles, with
the transfer `Re Tr(C† ρ(T))` averaged over an ensemble of rf scalings
and offsets, amplitude limits, power and smoothness penalties and a
fixed duration; exact gradients without relaxation (eigenvectors of each
step) and with it (Liouville space, the Van Loan block exponential
applied to vectors); `Event::Control` to replay the pulses in `evolve`;
`unitary_bound`; field profiles through the exchange model of 1.35
(`Exchange::optimize_profile`); `spin.Control`, `spin.Profile` and
`spin.rf_pulse` in Python; tutorial 10. Changes to the plan:

- **Time-optimal target corrected.** The plan asked for "no less than
  `1/(2J)`"; Khaneja, Brockett and Glaser give `sin(πJT/2)` for the
  antiphase transfer `2I_zS⁻ → I⁻` up to `T = 1/J`, and `3/(2J)` for the
  in-phase transfer `S⁻ → I⁻`. Those are the limits tested.
- **Field profiles.** The field derivative of relaxation and of thermal
  populations is taken by Richardson-extrapolated differences (the rest
  of the gradient is exact). For SABRE build-up in the exchange model the
  optimum is the constant anti-crossing field: modulating the field gains
  nothing there.
- **Not done:** rf pulses through chemical exchange (field profiles
  only), control with relaxation beyond six spins ½, and a comparison
  with QuTiP's optimal-control package (independent NumPy and SciPy
  builds instead).

Measured:

- **Gradients.** Against central differences for rf and field controls,
  with and without relaxation, Redfield relaxation, a spin 1, ensembles
  with rf errors and offsets and complex operators: to 10⁻⁷–10⁻⁸ (the
  differences' own error). Against an independent SciPy build (`expm`,
  `expm_frechet`) with relaxation: 10⁻¹⁰ (rf) and 10⁻⁹ (field). The
  transfer equals `evolve` to 10⁻¹⁰ and an independent NumPy propagation
  to 10⁻¹².
- **Unitary bound.** Thermal ¹H → ¹³C transfer reaches 3.976 times
  carbon's own polarisation, I₂S 4.476 times, both the bound to 10⁻⁹;
  with `T₂` = 20 ms the optimum is 0.74 of it.
- **Time-optimal limits.** In-phase `S⁻ → I⁻` reaches `η*(T)` to
  3·10⁻¹⁰ at `JT` = 0.8 and 1.2 and one at 1.7; antiphase `2I_zS⁻ → I⁻` 99.39 % of
  `sin(πJT/2)` at `JT` = 0.5 with 100 steps (the ideal sequence of
  Khaneja et al. reaches it exactly), 99.73 % at 0.8 and one at 1.2.
- **Robustness.** A pulse optimised for rf amplitudes of 80–120 %
  transfers more than 99.99 % at and between them and more than 99.9 %
  out to 75 % and 125 %, where nothing was optimised (99.95–99.98 % over
  twelve local optima; which one L-BFGS finds depends on rounding, which
  differs between machines: two Linux runners of the release's CI found
  99.97 %);
  refocused INEPT gives 0.089 and 0.291 at 80 % and 120 %.
- **Field cycling.** A profile of 2 s in 40 steps, started at zero field,
  gives −0.8128 on ¹³C: the unitary bound within the blocks of total `M`
  a field conserves, so no field profile does better; the adiabatic
  passage needs 50 s for it, a linear sweep of 2 s gives −0.249.
- **SABRE.** Over 5 s of build-up the best constant field is 0.138 μT
  (0.33294); optimised ten-step profiles from 0.1 μT or from zero field
  end there.
- **Cost** of one evaluation, 100 steps: 0.5 ms for two spins without
  relaxation (27 ms for four spins and 200 steps); with relaxation 50 ms
  for two spins and 0.7 s for three at rf up to 2 kHz, about four times
  that at 10 kHz.

An independent review wrote its own SciPy implementation and compared
values and gradients in both engines (rf with ensembles, spin-index
channels, non-Hermitian operators, `T₁`/`T₂`, Redfield relaxation, a spin
1, field drives in both frames, large norms, degenerate spectra): to
5·10⁻¹⁴, and 4·10⁻¹¹ for thermal starts at room temperature. It confirmed
the in-phase limit by a direct maximisation over the reachable
propagators, the unitary bound, the fairness of the INEPT comparison and
the SABRE optimum. It found:

- **Out of memory.** The dense Liouville generator was allocated before
  the work check (eight relaxing spins aborted the process), and the work
  checks did not bound memory: the propagators of every step, a cache of
  field-step generators and members running in parallel (a 5-spin field
  profile took 9.8 GB). Now memory (at most 2 GB) and work are estimated
  before anything is allocated, Liouville space is limited to 4096
  elements, no generators are cached and propagators are recomputed from
  the stored eigenvectors; the work estimates were 15 and 5 times too
  low and are calibrated now, for the whole ensemble.
- **Converged on a bound.** The `tanh` map of bounded controls saturated
  and the optimiser's first step (one unit: 1 T for a field) threw every
  field onto a bound, where the gradient vanished: from zero field the
  PHIP profile stopped at 0.40 and SABRE at 0.005, both reported as
  converged. Bounds are now kept by a sine (no plateau beyond them),
  field variables count twentieths of the range, and starts on a bound
  are moved inside: both now reach their optimum. The PHIP value the
  milestone first reported from random starts came from that lucky first
  step; field profiles are strongly multimodal (random starts stop at
  0.08–0.38), which the guide now says.
- **Accuracy.** The field derivative of relaxation, claimed to 10⁻⁹, was
  off by up to 1.7·10⁻³ at room temperature and millitesla fields: the
  drift towards equilibrium was the small difference of two large rates.
  It is now `tanh(x/2)/T₁` (in `evolve` as well), with Richardson
  extrapolation: 2·10⁻¹⁴ against the rates written out. The Liouville
  engine propagated `ρ` itself and lost digits of thermal polarisation;
  it now propagates the deviation as `evolve` does.
- **Smaller issues.** The cost of a field profile was estimated at its
  first field; a maximally mixed start gave a degenerate normalisation
  and stopped after one iteration; non-finite operators and overflowing
  weights were accepted; out-of-range starts were silently projected;
  Python docstrings were displaced, `spin.control` (now `spin.rf_pulse`)
  took only tuples and was easily confused with `spin.Control`; the
  guide's timings, the scope of `limit`, the offset sign, the scope of
  robustness and an exchange test's name were wrong or missing. All are
  fixed, with tests (bound and mixed starts, refusals, JSON, precision at
  low polarisation, Redfield and spin-1 gradients, the SciPy reference).

Left as is: Ctrl-C reaches Python only when an optimisation ends; with
relaxation and no amplitude limit, an evaluation the optimiser tries
beyond the work limit ends the run with an error; the cost with
relaxation grows with rf amplitude times duration; `evolve` resolves
polarisations below about 10⁻¹² only to 10⁻¹⁷ absolute.

### 1.37 — Quantum sensing with spins in diamond (M, delivered)

Use case: nitrogen-vacancy (NV) centres in diamond measure magnetic
fields at the nanoscale, inside and next to living cells (Le Sage et al.,
Nature 496, 486 (2013); Schirhagl et al., Annu. Rev. Phys. Chem. 65, 83
(2014)), and detect single nuclear spins. An NV centre is an electron
spin 1 coupled to its nitrogen nucleus and to nearby ¹³C nuclei. Its
coherence against noise decides what it can sense. This combines the
spins of 1.33 with the slow noise and echoes of 1.30.

| Deliverable | Detail |
|---|---|
| NV centres | electron spin 1 with zero-field splitting `D = 2.87 GHz`, Zeeman term, ¹⁴N or ¹⁵N with hyperfine and quadrupole couplings, ¹³C nuclei with hyperfine tensors, in the secular approximation |
| Sequences | optically detected magnetic resonance (ODMR) spectra, Ramsey, Hahn echo, CPMG and XY-8 dynamical decoupling with finite pulses |
| Noise | classical field noise of a given spectrum (telegraph, Ornstein–Uhlenbeck, 1/f), the filter-function description of decoupling (Cywiński et al., Phys. Rev. B 77, 174509 (2008)) against exact trajectories |

Validation: the ODMR lines lie at `D ± γ_e B` and split by the hyperfine
coupling. The CPMG signal of one weakly coupled ¹³C follows the closed
form of Taminiau et al. (Phys. Rev. Lett. 109, 137602 (2012)), with its
dips at the predicted pulse spacings. The coherence under Gaussian
Ornstein–Uhlenbeck noise follows the filter-function integral, and under
telegraph noise the exact Kubo–Anderson results of 1.30.

Delivered in `spin/nv.rs`: `NvCenter` (electron spin 1 with `D`, the
Zeeman term, ¹⁴N with hyperfine and quadrupole couplings or ¹⁵N, and
¹³C nuclei with `A_∥` and `A_⊥`, in the secular approximation and the
frame of the microwave), resonance `lines`, pulsed `odmr`, `signal` and
`evolve` of sequences on either transition with ideal pulses or finite
ones (rotating-wave approximation, both transitions driven with the
carrier's phase, free periods between pulse centres); `ramsey`, `hahn`,
`cpmg`, `xy8`; `FieldNoise` (telegraph, Ornstein–Uhlenbeck, 1/f as a sum
of fluctuators) with spectra, `filter_function`, `decay_exponent` and
`signal_with_noise`; `spin.NvCenter`, `spin.FieldNoise` and the sequences
in Python; tutorial 11. Changes to the plan:

- **Hyperfine tensors.** In the secular approximation only `S_z` couples
  to a nucleus, and the azimuth of each ¹³C can be rotated away (the
  nuclei do not couple to each other), so `A_∥` and `A_⊥` are the whole
  tensor.
- **ODMR spectra** are the resonance lines with their strengths and
  pulsed ODMR; continuous-wave ODMR needs the optical cycle, which is not
  modelled.
- **A block propagation** was added for speed: the secular Hamiltonian is
  block diagonal in the electron level and a sum of one term per nucleus
  within a block, so free evolution applies one 2 × 2 or 3 × 3 factor per
  nucleus. CPMG-16 with ¹⁴N and six ¹³C (dimension 576) took 36 s dense
  and takes 0.4 s.
- **Not done:** fields off the NV axis (the secular approximation needs
  the field along it), the optical cycle, nuclear spin baths beyond the
  explicit nuclei (cluster expansions), and pulse errors other than
  finite duration and off-resonance (amplitude errors can be modelled by
  the angle).

Measured:

- **Lines.** `D ± γ_e B ± A_N m_I` to 10⁻³ Hz for ¹⁴N and ¹⁵N; pulsed
  ODMR dips to 2/3 at each ¹⁴N line (within 0.02) and equals one finite
  pulse of the dense propagation to 10⁻¹⁰; at zero field the degenerate
  lines of one ¹³C merge at `D ± A/2` with strength ½ each; above 102 mT
  the line of `m = −1` is negative.
- **Nuclear-spin detection.** The CPMG-64 signal of a ¹³C (60 and 30 kHz,
  40 mT, with ¹⁴N) equals Taminiau's closed form to 1.5·10⁻¹³ over 1360
  spacings, with six dips within 1.5 % of `τ_k`. With 20 MHz pulses XY-8
  keeps the first three dips within 0.03 of their ideal depth (0.118
  against 0.131 for the first); CPMG loses more than 0.3 of it (0.64),
  because the ±2.14 MHz of ¹⁴N makes its pulses imperfect and it corrects
  pulse errors only along its axis.
- **Propagation.** An independent NumPy build (¹⁴N and two ¹³C, both
  transitions, ideal and finite pulses, random states) agrees to 10⁻¹⁰,
  the block propagation with the dense one (¹⁵N and three ¹³C, ¹⁴N and
  two, non-Hermitian operators) to 10⁻¹⁰. A linearly polarised carrier
  simulated in the laboratory frame without the rotating-wave
  approximation agrees to 8·10⁻⁴ at zero field, 0.5 mT and 2 mT (Hahn,
  XY-8, phase-shifted Ramsey, 5 MHz pulses); ideal pulses are the limit of
  selective finite ones to 3·10⁻⁴ (30 MHz at 1 T).
- **Noise.** Ornstein–Uhlenbeck decay exponents equal the closed forms of
  Klauder and Anderson to 10⁻¹⁰ (Ramsey) and 10⁻⁹ (Hahn); the
  filter-function integral of 1/f noise, by quadrature, to 10⁻⁵;
  trajectories match `e^{−χ}` within four standard errors (2·10⁴
  trajectories, Ramsey, Hahn, CPMG, XY-8, both transitions), telegraph
  noise Kubo and Anderson's free decay and echo, where `e^{−χ}` misses by
  more than 0.05. Noise during finite pulses agrees with 128 pieces per
  pulse within the statistical error.
- **Cost.** CPMG-64 with ¹⁴N and one ¹³C: 0.4 ms (ideal pulses), 1.2 ms
  (20 MHz); a 1360-point CPMG-64 sweep 0.9 s; CPMG-16 at dimension 576
  0.4 s and 8.5 s; 10⁴ noisy trajectories of CPMG-16 on a bare centre
  0.05 s (0.15–0.2 s with finite pulses); `lines` at dimension 768
  0.03 s; `decay_exponent` of XY-8×250 under 1/f noise of 1000
  fluctuators 0.17 s.

An independent review wrote its own NumPy model and a laboratory-frame
simulation, and confirmed the Hamiltonian's signs and couplings, the
block propagation for arbitrary states (to 7·10⁻¹³, up to dimension 768),
the decay exponents, the exact Ornstein–Uhlenbeck and telegraph sampling
(with up to 4·10⁶ trajectories) and the 16-piece treatment of noise
during pulses against 256 pieces. It found:

- **Wrong drive phase at low field.** The drive `cos φ S_x + sin φ S_y`
  gave the two transitions opposite phases; a carrier gives both the
  same. With one transition driven the signals are unchanged, but at
  zero field a Hahn echo gave 0.467 instead of 0.768. The drive is now
  the co-rotating part of a linearly polarised carrier, tested against
  the laboratory frame; the NumPy reference now compares the whole state
  (a population from the pumped state does not see the phase).
- **Crash.** The dimension of 64 or more carbons wrapped around to zero,
  passed the check and divided by zero (or tried to allocate 38 GB). It
  saturates now and is refused.
- **Wrong results.** `evolve` mirrored the blocks and so was wrong for
  non-Hermitian operators; it now propagates their Hermitian and
  anti-Hermitian parts. Filter functions did not check their periods (a
  negative one gave a value) and refused −90° or 540° pulses; they check
  and take angles modulo 360° now.
- **Cost.** Noise sampling was not counted in the work limit (1000
  fluctuators took 26 s), `decay_exponent` was quadratic in the number of
  periods (56 s for XY-8×250), every finite pulse under noise took 16
  full exponentials whatever the correlation time (105 s for 1000
  trajectories at dimension 36, now 8.5 s), `lines` diagonalised the full
  Hamiltonian (13.5 s at dimension 768, now 0.03 s), and ODMR, `lines`
  and noiseless finite pulses had no work limit. All are fixed or
  counted, with the costs measured.
- **Smaller issues.** The comment on `γ_e` named `g = 2.0028` for the
  free electron's value; negative frequencies above 102 mT were not
  explained; degenerate lines were listed separately with strengths that
  depended on the basis; one trajectory reported a standard error of
  zero; static telegraph noise had a NaN spectrum and tiny rates a NaN
  `χ`; adjacent free periods could be refused next to finite pulses;
  custom nitrogen couplings were reachable only through JSON, and a
  transition of 128 raised `OverflowError`; the trajectory tests allowed
  10⁻³ beyond four standard errors. All are fixed.

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
13. **Fifth stage: error correction first.** Distance-preserving circuits
    and correlated decoding (1.22) finish what 1.19 opened, and its review
    measured the cost of the generic schedule; Stim gives exact distance
    references. Larger basis sets and open shells (1.23) close the oldest
    chemistry gap with PySCF as reference. Short simulations (1.24) address
    the last speed gap of the table, the cold start after a pause, and
    documentation and reach (1.25) end the stage, as tests ended the last.
14. **Sixth stage: one use case, measured.** Variational training at 26–28
    qubits stands in for the GPU track's missing use case. The CPU loses
    a factor of four to five in the gradient and two in the energy before
    memory or a GPU matter, so cheaper gradients come first (1.26), then
    single precision for the memory (1.27), then the use case end to end,
    which also decides what a GPU would still add (1.28).
15. **Seventh stage: walks first again.** Interacting walkers (1.29)
    extend QSO's walks to many particles and to graphs that one walker
    cannot tell apart, with permanents and determinants as references.
    Correlated and slow noise (1.30) comes before learning at scale
    (1.31), because the larger learning experiments should include it.
    Screening and tighter bounds (1.32) close two small, long-standing
    gaps. Spin dynamics (1.33) comes last, because it builds on the echoes
    and slow noise of 1.30.
16. **Eighth stage: magnetic resonance for medical research.** Relaxation
    from first principles (1.34) comes first, because exchange models and
    long-lived states (1.35) need relaxation times that follow from the
    molecule rather than from input. Optimal control (1.36) comes after
    them, so that it optimises sequences on realistic models. Quantum
    sensing (1.37) comes last: it adds a new kind of spin (the electron
    of an NV centre) and builds on the noise of 1.30 and the decoupling
    sequences that optimal control can improve.

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
* Fifth stage (October 2026): error-correction circuits that keep their
  distance, with circuit-distance search and correlated decoding (1.22),
  larger basis sets and open shells (1.23), short simulations at full
  speed (1.24), documentation and reach (1.25).
* Sixth stage (October 2026): the GPU track's use case is variational
  training at 26–28 qubits; it is served on the CPU first (1.26–1.28),
  because no GPU is available to test on and the measurements show
  larger losses elsewhere.
* After the sixth stage (October 2026): the use case runs on the CPU (28
  qubits, one minute per gradient, 4.5 GiB); the GPU track stays parked
  until a machine with a GPU is available to test on, and would then be a
  single-precision `wgpu` backend for the QAOA layer, with the CPU's single
  precision as reference (1.28).
* Seventh stage (October 2026): interacting walkers (1.29), correlated
  and non-Markovian noise (1.30), noise learning beyond ten qubits on the
  Pauli-frame sampler (1.31), integral screening and commutator bounds
  for higher orders (1.32). Leakage stays out: it needs qutrits in every
  simulator.
* Spin dynamics and magnetic resonance (1.33, added October 2026): the
  quantum dynamics of nuclear spins behind hyperpolarised MRI, built on
  the density-matrix, relaxation and Magnus tools QSO already has. Image
  simulation, quantum algorithms for CT and clinical claims stay out of
  scope; QuTiP joins the cross-validation as a reference only.
* Eighth stage (1.34 to 1.37, added October 2026 on request): magnetic
  resonance for medical research. It covers relaxation from first
  principles, realistic hyperpolarisation (chemical exchange, PHIP-SAH,
  long-lived states), optimal control of spin experiments, and quantum
  sensing with NV centres. Quantum algorithms for CT were considered and
  left out: at the size of a real image, loading the data and reading
  out the result undo any speed-up, so a demonstration would mislead.
* The browser playground stays, because it costs little to maintain: it
  is plain HTML and JavaScript without dependencies or a build step, it
  uses only the npm package's JSON interface, which the package's tests
  cover, and the documentation build publishes it. If it ever needs
  more than that, it is removed from the site rather than given a
  framework.
