# QSO Roadmap (1.x)

Status: proposal, October 2026, after the 1.0.1 release.

QSO 1.0 delivers what it documents: every number in the README is reproduced
by a test against an independent reference, the packages are on crates.io
and PyPI, and releases are automated. This roadmap plans the next stage: from
a correct, small simulator to the reference toolkit for its niche.

## 1. Where 1.0 stands

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

| Area | 1.0 | Gap |
|---|---|---|
| Statevector speed | one pass over the state per gate | no gate fusion, no explicit SIMD — see [measurements](#measured-performance-against-established-simulators) |
| Noise | density matrices up to 14 qubits; one noise model for all gates and qubits | no trajectories for larger registers, no per-qubit/per-gate models, no error mitigation |
| Quantum walks | continuous-time only; exact analysis up to 2048 nodes | no discrete-time (coined, Szegedy) walks, no symmetry reduction, no state-transfer analysis, no circuits for walks |
| Physics and chemistry | TFIM, XXZ, MaxCut, H₂ only | no fermion-to-qubit mappings, no product-formula (Trotter) circuits, no excited states |
| Interoperability | OpenQASM 2 without `gate` definitions | no OpenQASM 3, no Qiskit/PennyLane/Cirq bridges, no browser build |
| Documentation | README, architecture notes, rustdoc | no user guide, no tutorials or notebooks, no hosted Python reference |
| Engineering | CI on 3 OSes, MSRV, release workflow | no fuzzing of the parsers, no Miri for the `unsafe` kernels, no API-compatibility check, no benchmark tracking, registry tokens instead of Trusted Publishing |

### Measured performance against established simulators

<!-- filled in from the comparison run -->

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

**Non-goals for 1.x:** hardware vendor integrations beyond OpenQASM export
and the Qiskit bridge; GPU or multi-node execution before the CPU kernels are
close to memory bandwidth; a general tensor-network library; any claim about
Metatron's Cube that is not graph theory or physics.

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
  after a dry run.

## 4. Milestones

Sizes: S ≈ days, M ≈ 1–2 weeks, L ≈ 3–6 weeks of focused work.

### 1.1 — Foundations: trust, supply chain, documentation (M)

Cheap, protects the 1.x promise, and gives users their first impression.

| Deliverable | Detail |
|---|---|
| Trusted Publishing | PyPI and crates.io via GitHub OIDC; delete the `CRATES_IO`/`PYPI` secrets; crates.io set to trusted-only; publishing jobs in a `release` environment with required approval |
| Unsafe-code checks | Miri job for `kernels.rs` and `parallel.rs` on small registers |
| Fuzzing | `cargo-fuzz` targets for the OpenQASM, Pauli-string and graph-spec parsers and for `Circuit`/`StateVector` deserialisation; short run per PR, long run weekly |
| API stability | `cargo-semver-checks`; `stubtest` comparing `_core.pyi` with the compiled module |
| Cross-validation | scheduled job comparing QSO with Qiskit Aer and Qulacs on random circuits (pure and noisy), using the harness from the comparison run |
| Benchmark tracking | weekly `benchmark_report` run, results stored, regressions > 25 % flagged |
| Platforms | musllinux and Windows arm64 wheels; native aarch64 tests; Python 3.14 in the matrix; Node-24 action versions |
| Documentation | user guide (mdBook) and Python API reference on GitHub Pages; tutorials as notebooks executed in CI so they cannot go stale: circuits and gradients, VQE and QAOA, noise, quantum walks on Metatron's Cube |

**Done when** every job is green on `main`, the docs site is live, and a
release has been published without stored tokens.

### 1.2 — Quantum walks: exact at scale (L)

| Deliverable | Detail |
|---|---|
| Symmetry reduction | Coarsest equitable partition with chosen vertices individualised (reuses the colour refinement of `graph::automorphism`). With characteristic matrix `P`, `HP = PB` holds exactly, so `exp(−iHt)P = P·exp(−iBt)`: walks, search and limiting distributions run in dimension = number of cells. Generic fallback: the minimal invariant (Krylov) subspace of the start vector via the existing Lanczos code |
| Implicit graph families | hypercubes, Hamming, Johnson and complete multipartite graphs built directly as quotients, so a hypercube with 2³⁰ vertices never materialises |
| Discrete-time walks | coined walks on arbitrary graphs (arc basis, flip-flop shift; Grover, Hadamard, Fourier coins), Shenvi–Kempe–Whaley search, Szegedy walks of reversible Markov chains with quantum hitting times |
| State transfer | perfect and pretty-good state-transfer analysis (spectral criteria, transfer times, fidelity optimisation) |
| Beyond 2048 nodes | limiting distributions and search through sparse spectral methods on the invariant subspace instead of a dense eigendecomposition |
| Walks as circuits | exact circuits for circulant graphs (diagonalised by the QFT: cycles, complete graphs), product-formula circuits for general graphs on binary-encoded vertices; QASM export |
| Graph library | Johnson, Kneser, Hamming, Paley, strongly regular and Cayley graphs; canonical labelling and isomorphism test from the existing search; edge-list and GraphML I/O; `networkx` conversion in Python |
| Showcase | "Metatron atlas": spectra, symmetry groups, search and transfer behaviour of every `metatron:<length>` graph, as a reproducible report and notebook |

Validation targets:

* Reduced and full walks agree to 1e-12 on all small test families.
* Hypercube Qₙ, adjacency walk: the antipodal amplitude is `(−i·sin γt)ⁿ`,
  perfect state transfer at `γt = π/2` (Christandl et al. 2004), computed for
  n = 30 in dimension 31.
* No perfect state transfer on paths Pₙ for n ≥ 4 (Christandl et al.
  2004/2005).
* SKW search on Qₙ: success probability 1/2 − O(1/n) after about
  (π/2)·√N steps (Shenvi, Kempe, Whaley 2003).
* Hadamard walk on the line with the symmetric initial coin state:
  σ(t)/t → √(1 − 1/√2) ≈ 0.541 (Konno's limit theorem).
* Szegedy walks: quantum hitting times scale as the square root of the
  classical hitting times already in `walks::classical` (Szegedy 2004), on
  complete graphs and cycles.
* Circulant walk circuits equal `exp(−iγLt)` to 1e-12.
* Canonical forms agree under random relabelling and separate known
  cospectral, non-isomorphic pairs (Shrikhande graph vs 4×4 rook's graph).

### 1.3 — Simulation performance (L)

| Deliverable | Detail |
|---|---|
| Gate fusion | fuse runs of gates into dense blocks on up to k qubits (k chosen by measurement, typically 2–5) before execution; on by default, can be switched off; used by simulation, sampling, expectation values and the adjoint pass |
| SIMD kernels | explicit AVX2/AVX-512/NEON paths for dense 1-, 2- and k-qubit kernels with runtime detection and the current code as fallback |
| Cache blocking | execute fused blocks on low qubits chunk-wise so the state passes through main memory fewer times |
| Batched evaluation | many parameter vectors or observables per call (landscapes, multi-start optimisation); vectorised in Python |
| Single precision | optional `f32` state: half the memory, one more qubit, with documented tolerances |

Validation: fused and unfused results agree to 1e-12 on random circuits; the
cross-validation job stays green; speed-ups are measured with
`benchmark_report` and the comparison harness. Targets are set from the
measurements in section 1.

### 1.4 — Noise at scale and error mitigation (L)

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

### 1.6 — Interoperability and reach (M–L)

| Deliverable | Detail |
|---|---|
| OpenQASM | `gate` definitions in QASM 2; an OpenQASM 3 subset (gate definitions, parameters, `measure`, `reset`, simple classical conditions) |
| Python bridges | optional extras: Qiskit (circuits, `SparsePauliOp`, Estimator/Sampler primitives backed by QSO), a PennyLane device with adjoint differentiation, a Cirq converter |
| Browser | `wasm32` build and npm package; interactive Metatron playground (walks, search, symmetry) on GitHub Pages |

Validation: converted circuits have identical unitaries on random inputs;
primitives agree with Qiskit Aer within shot noise; the PennyLane device
passes PennyLane's device test-suite; QASM round trips on the official
example files.

### Research track (decide after 1.6)

Stabilizer simulation (Clifford tableaux for thousands of qubits, as the base
for error-correction experiments), matrix-product-state simulation for
low-entanglement circuits, and a GPU backend through `wgpu`. Each is a
project of its own; they start only with a clear use case and after the CPU
core is close to memory bandwidth.

## 5. Order and reasoning

1. **1.1 first:** small, makes every later release safer, and documentation
   is what new users see first.
2. **1.2 next:** QSO's distinctive contribution; it builds directly on the
   refinement, Lanczos and walk code that already exists.
3. **1.3 before 1.6:** a Qiskit or PennyLane backend is only attractive if
   it is fast.
4. **1.4 and 1.5** enter crowded fields and can swap places depending on
   whether noise or chemistry users show up first.

## 6. Measures of success

* Correctness: the cross-validation job is green on every scheduled run;
  every feature in the CHANGELOG names its reference test.
* Speed: throughput relative to Qulacs and Qiskit Aer on the same machine,
  tracked per release.
* Usability: docs site with executed tutorials; every user-facing Rust
  feature is available from Python.
* Releases: one per milestone, published without stored tokens.

## 7. Decisions for the maintainer

* Order of 1.2 (walks) and 1.3 (performance).
* Whether the Qiskit and PennyLane bridges (1.6) are wanted, given that they
  add test dependencies and follow those projects' release cycles.
* Whether the browser playground is worth its maintenance cost.
