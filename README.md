# QSO — Metatron Quantum State Operator

[![CI](https://github.com/LashSesh/qso/actions/workflows/ci.yml/badge.svg)](https://github.com/LashSesh/qso/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/metatron-qso-rs.svg)](https://crates.io/crates/metatron-qso-rs)
[![PyPI](https://img.shields.io/pypi/v/metatron-qso.svg)](https://pypi.org/project/metatron-qso/)
[![npm](https://img.shields.io/npm/v/metatron-qso.svg)](https://www.npmjs.com/package/metatron-qso)
[![docs.rs](https://img.shields.io/docsrs/metatron-qso-rs)](https://docs.rs/metatron-qso-rs)
[![Rust 1.89+](https://img.shields.io/badge/rust-1.89%2B-orange.svg)](https://www.rust-lang.org/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](metatron_qso_py/README.md)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Validated quantum simulation in pure Rust, with Python bindings, a
command-line tool and a WebAssembly build for JavaScript and the
browser.**

QSO simulates quantum circuits exactly — as pure states or as density
matrices with noise — and runs variational and textbook algorithms on top of
them. It models noisy devices (relaxation, gate and readout errors) and
simulates them with quantum trajectories beyond the reach of density
matrices, and it mitigates their errors: readout mitigation, zero-noise
extrapolation, probabilistic error cancellation and Pauli twirling, with a
standard error on every estimate. For chemistry and physics it computes
Hartree–Fock from a molecule's geometry (STO-3G, 6-31G, 6-31G* and
cc-pVDZ, hydrogen to argon) or reads FCIDUMP files, maps the molecules to qubits, solves them with UCCSD and
ADAPT-VQE against full CI, finds excited states, builds Trotter–Suzuki
circuits with rigorous error bounds, and estimates amplitudes with
Heisenberg-limited scaling. It also simulates continuous- and discrete-time quantum walks on
arbitrary graphs, reduces them by symmetry to exact low-dimensional problems
where the graph allows it (a walk on the 2³⁰-node hypercube in 31
dimensions), decides perfect state transfer, and compiles walks to circuits.
Its signature geometry is **Metatron's Cube**, treated with mathematical
precision: the complete graph K₁₃ drawn on the 13 centres of the Fruit of
Life. It reads and writes OpenQASM 2 and 3, including mid-circuit
measurement and feed-forward, simulates Clifford circuits and the graph
states of its graphs on thousands of qubits, and runs in the browser: try the
[playground](https://lashsesh.github.io/qso/playground/).

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
| **Statevector simulator** | n-qubit pure states (up to 34 qubits; 35 in single precision, `StateVector32`) with multi-threaded kernels (AVX on x86-64, NEON on aarch64, bit-identical to the portable ones), gate fusion and cache blocking. Gates: H, X, Y, Z, S, S†, T, T†, √X, √X†, RX, RY, RZ, P, U, SWAP, RXX, RYY, RZZ and custom unitaries on up to 6 qubits, each with any number of control qubits. Sampling, mid-circuit measurement, reduced density matrices, entanglement entropy |
| **Circuits** | Builder API with symbolic, differentiable parameters (`θ = a·p[i] + b`), inverse, binding, depth and gate counts. OpenQASM 2.0 and 3 export *and* import; multi-controlled gates are exported by exact, ancilla-free decomposition |
| **Programs** | OpenQASM 2 and an OpenQASM 3 subset: gate definitions, `ctrl`/`negctrl`/`inv`/`pow` modifiers, mid-circuit measurement, reset and conditions. Exact outcome distributions by following every measurement branch, seeded sampling; checked against the OpenQASM project's example programs |
| **Stabilizer simulation** | Aaronson–Gottesman tableaux for up to 16 384 qubits: Clifford gates (rotations by multiples of π/2 included), Pauli measurements, reset, expectation values, sampling. Graph states of any graph, local complementation, entanglement entropy from GF(2) cut-ranks. Clifford OpenQASM programs, mid-circuit measurement included, run on it automatically when they are too large for a state vector |
| **Matrix-product states** | Circuits with little entanglement on up to 65 536 qubits: any gate of the gate set (controls included), routed by swaps; bond-dimension limit and cutoff with every truncation measured and a rigorous lower bound on the fidelity with the exact state; Pauli expectation values, exact sampling, amplitudes, measurement and reset, entanglement at every bond; OpenQASM programs |
| **Error correction** | Detector circuits (Clifford gates, Pauli noise, measurement records, detectors, logical observables) in the style of Stim; Pauli-frame sampling, 64 shots per machine word; detector error models, equal to Stim's term by term; a weighted union-find decoder and exact minimum-weight perfect matching (sparse blossom: regions grown on the detector graph, 1.1 to 2.3 times PyMatching's time; and a blossom algorithm on the complete graph of events); BP+OSD (min-sum or product-sum belief propagation with ordered-statistics decoding) for models whose errors flip more than two detectors, with the same corrections as `ldpc` for min-sum BP and equal priors; two-pass (correlated) matching with PyMatching's reweighting rule, on decomposed models equal to Stim's; CSS codes from their checks, triangular colour codes and bivariate bicycle codes (including the [[144, 12, 12]] code); repetition, rotated surface-code (Z and X basis) and CSS-code memory experiments with circuit-level noise, for bicycle codes also with the depth-8 syndrome cycle of Bravyi et al.; circuit distance (exact for graph-like models, proven bounds otherwise); logical error rates; export to and import from Stim's circuit format |
| **Observables** | Pauli strings and sums with a parser (`"0.5*Z0 Z1 - X2"`), matrix-free expectation values and `H\|ψ⟩`, exact ground states by Lanczos |
| **Noise** | Density-matrix simulator (up to 14 qubits) on the same kernels. Depolarising (1 and 2 qubits), amplitude damping, phase damping, bit flip, phase flip, thermal relaxation (T₁/T₂), Pauli channels (correlated two-qubit noise), coherent rotations, custom Kraus channels, Pauli twirls. **Device models** (JSON): per-qubit T₁/T₂ and readout errors, per-gate durations, depolarising errors, over-rotations and channels, relaxation while qubits wait. **Quantum trajectories** for any register a state vector holds: one trajectory, compilation included, costs about three noiseless runs at 16–20 qubits, and 200 trajectories take 1.2–1.3 times as long as 200 noiseless runs |
| **Mitigation and estimation** | Estimates with standard errors and confidence intervals; variance-optimal (Neyman) shot allocation over measurement groups; readout calibration (tensored and full) and mitigation; zero-noise extrapolation (global and random folding; linear, Richardson, polynomial and exponential fits); probabilistic error cancellation of Pauli noise with its sampling overhead; Pauli twirling; noise learning (sparse Pauli–Lindblad models of twirled two-qubit layers, with bootstrap standard errors) and error cancellation and amplification with the learned model; virtual distillation (exact, and the measurement circuits on 2 or 3 copies); classical shadows |
| **Gradients** | Adjoint differentiation (all partial derivatives for about three circuit runs), parameter-shift rules (2-term, and 4-term for controlled rotations), finite differences; batched evaluation of many parameter vectors in parallel |
| **Algorithms** | VQE with seeded parallel restarts; QAOA for weighted MaxCut (fast diagonal simulation, exact optimum by enumeration, INTERP layer-wise initialisation); Grover search with any number of marked items; QFT; phase estimation; variational classifier with data re-uploading; graph classifier with trainable QAOA-type layers |
| **Excited states and geometry** | VQD and SSVQE; exact state derivatives, quantum Fisher information, quantum natural gradient; variational imaginary-time evolution |
| **Hamiltonian simulation** | Pauli rotations for any string; Trotter–Suzuki circuits of order 1, 2, 4, 6, … with rigorous error bounds (commutator bounds for orders 1 and 2); qDRIFT with its error bound |
| **Electronic structure** | Fermionic operators with normal ordering; Jordan–Wigner, parity (with two-qubit reduction) and Bravyi–Kitaev mappings; Gaussian integrals over shells up to g, spherical or Cartesian (STO-3G, 6-31G, 6-31G*, cc-pVDZ; H to Ar), restricted, unrestricted and restricted open-shell Hartree–Fock with DIIS, frozen-core active spaces (also of open shells); molecular integrals from FCIDUMP files (H₂, LiH, BeH₂ bundled with PySCF reference energies); Hartree–Fock states; full CI; UCCSD, k-UpCCGSD and ADAPT-VQE |
| **Optimisation and estimation** | QUBO and Ising problems with exact enumeration; QAOA for any diagonal cost with X, XY (Hamming-weight conserving) and warm-start mixers; canonical, iterative and maximum-likelihood amplitude estimation |
| **Models** | Transverse-field Ising, XXZ Heisenberg, MaxCut cost operator, H₂ / STO-3G (2 qubits, with full-CI reference), any molecule given as an FCIDUMP file |
| **Optimisers** | L-BFGS with strong-Wolfe line search, Adam, gradient descent with momentum, Nelder–Mead, SPSA |
| **Graphs** | Weighted graphs; complete, cycle, path, star, bipartite, hypercube, grid/torus, Petersen, Heawood, Johnson, Kneser, Hamming, rook's, Paley, Shrikhande, circulant, Cayley (Z₂ⁿ) and random graphs; the five Platonic solids; the Metatron's Cube / Fruit-of-Life family; Cartesian products; **exact automorphism groups, canonical labelling and isomorphism**; edge-list and GraphML files, NetworkX import |
| **Quantum walks** | Continuous-time walks on any graph — exact on up to 2048 nodes, Krylov propagation and cyclic subspaces beyond — with limiting and time-averaged distributions and mixing times. **Symmetry reduction** by equitable partitions (hypercubes, Hamming and Johnson graphs of any size). Childs–Goldstone spatial search with optimised hopping rate, **perfect state transfer** and **pretty good state transfer decided exactly** (cyclotomic arithmetic and Kronecker's theorem, for paths, cycles and abelian Cayley graphs), Lindblad dephasing, classical baselines. **Driven walks** with time-dependent Hamiltonians (fourth-order commutator-free Magnus integrator with step-size control), such as adiabatic search. **Discrete-time walks**: coined walks (Grover, Fourier, Hadamard coins; Shenvi–Kempe–Whaley search), the Hadamard walk on the line, Szegedy walks with quantum hitting times; coined walks with coin and position dephasing on density matrices. **Walk circuits**: exact for circulant and Cayley graphs, product formulas for any graph |
| **Boson sampling** | Permanents (Glynn's formula, `O(n·2ⁿ)`), Haar-random interferometers, exact output probabilities for indistinguishable and distinguishable photons, exact sampling (Clifford & Clifford); partially distinguishable photons and losses; **Gaussian boson sampling**: Gaussian states (squeezing, displacement, interferometers, loss), photon-number and threshold-click probabilities by hafnians, loop hafnians and torontonians, exact sampling |
| **Backends** | Statevector, noisy density-matrix and device backends (density matrices or trajectories), shot-based estimation with qubit-wise-commuting measurement groups |
| **Interfaces** | Rust library `metatron_qso`, Python package `metatron_qso` (NumPy I/O, type stubs), `qso` command-line tool with JSON output, npm package `metatron-qso` (WebAssembly, no dependencies) and a browser playground |
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
| Walk on the hypercube Q₃₀ (2³⁰ nodes, 31 cells), P(antipode, γt = π/2) | 1 (to 2·10⁻¹⁵) | 1, Christandl et al. 2004 |
| Perfect state transfer between the ends of P₂, P₃ / P₄ … P₁₂ | π/2, π/√2 / none | Christandl et al. 2004 |
| Pretty good state transfer between the ends of P_n (n ≤ 300), of the Heisenberg chain (n ≤ 200), antipodes of C_n (n ≤ 512) | exactly when n + 1 is p, 2p or 2ᵏ / n = 2ᵏ / n = 2ᵏ | Godsil et al. 2012; Banchi et al. 2017; Pal 2017 |
| Hadamard walk on the line, σ(t)/t at t = 1000 | 0.54120 | √(1 − 1/√2) = 0.54120, Konno 2002 |
| Szegedy quantum hitting time on cycles Cₙ, n = 16 … 128 | n/4 | √ of the classical n²/6 up to a constant, Szegedy 2004 |
| Local adiabatic search on K_N (ε = 0.1), success at T = N/(ε√(N−1))·arctan√(N−1) for N = 64 / 256 / 1024; linear schedule, same T | 0.99942 / 0.99987 / 0.99996; 0.765 / 0.525 / 0.315 | ≥ 1 − ε², Roland & Cerf 2002; SciPy integration to 10⁻⁶ |
| Fourth-order Magnus integrator, error ratio when the step halves | 18.5 / 16.6 / 16.1 | 2⁴ = 16 |
| Coined walk on Metatron's Cube with the coin measured every step (Fourier coin) | equals the simple random walk to 10⁻¹³ | classical random walk |
| Shrikhande vs 4 × 4 rook's graph (same spectrum) | not isomorphic, \|Aut\| 192 vs 1152 | — |
| Automorphism group orders: Metatron's Cube / Fruit of Life / dodecahedron | 6 227 020 800 / 12 / 120 | 13! / D₆ / I_h |
| Hong–Ou–Mandel: two photons on a 50:50 beam splitter, coincidence probability | 0 (distinguishable: 0.5) | 0, Hong, Ou & Mandel 1987 |
| Full bunching of n photons, boson / distinguishable probability | n! for n = 1 … 6 | n! |
| Hong–Ou–Mandel with partially distinguishable photons, coincidence probability | (1 − \|⟨ψ₁\|ψ₂⟩\|²)/2 | Hong, Ou & Mandel 1987 |
| Perfect matchings by hafnians: hypercubes Q₃ / Q₄ / Q₅, dodecahedron, icosahedron | 9 / 272 / 589 185, 36, 125 | OEIS A005271; known counts |
| Squeezed vacuum (r = 0.8), photon-number distribution to 18 photons | closed form to 10⁻¹⁰ (relative) | tanh^{2k} r (2k)! / (4^k k!² cosh r) |
| Gaussian boson sampling: photon-number and click probabilities, 4 modes with displacement and loss; 20 photons from 10 squeezed modes | about 10⁻¹⁵ absolute; 2·10⁻¹¹ relative at 20 photons | The Walrus 0.22 |
| State vectors of random circuits, largest difference to Qiskit Aer, Qulacs, Cirq, PennyLane | ≤ 3 × 10⁻¹⁵ | — |
| Quantum trajectories vs density matrices, χ² test of 20 000 samples on 3, 6 and 10 qubits | passes at the 0.1 % level | — |
| GHZ₅ on the toy device, ⟨Σ Z_i Z_{i+1}⟩: raw / readout-mitigated + ZNE (1, 3, 5) | 3.38 / 4.00 ± 0.01 | 4 |
| Same with Pauli noise: probabilistic error cancellation (overhead 1.08) | 4.00 ± 0.01 | 4 |
| Noise learning, 4 qubits (depolarising + coherent phase error + readout errors): 39 learned rates / PEC with them on 3 layers (raw 0.492) | all within 3 standard errors / 0.534 ± 0.005 | exact rates / 0.533 |
| Virtual distillation of a depolarised GHZ-like state (no noise floor: its rotation comes last), error of ⟨Z0 Z1 + X0 X1 X2⟩ with 1 / 2 / 3 copies; circuits on copies vs. exact | 0.19 / 3.6·10⁻³ / 6.9·10⁻⁵; equal to 10⁻¹¹ | noiseless value; Tr(ρᵏO)/Tr(ρᵏ) |
| Zero-noise extrapolation (Richardson, scales 1, 3, 5), depolarised 3-qubit circuit | error ÷ 380 | — |
| Classical shadows, single-snapshot variance of a weight-k Pauli string | 3ᵏ − ⟨P⟩² | Huang, Kueng & Preskill 2020 |
| Full CI of H₂ / LiH / BeH₂ (STO-3G), identical under Jordan–Wigner, parity, Bravyi–Kitaev and the reduced parity encoding | −1.1373060 / −7.8824019 / −15.5951824 Ha | PySCF full CI, to 10⁻⁸ Ha |
| Hartree–Fock of H₂O from its geometry (6-31G), and full CI with a frozen core in 8 orbitals | −75.98397447 / −76.02472563 Ha | PySCF RHF and CASCI, to 10⁻¹⁰ / 10⁻⁹ Ha |
| Unrestricted Hartree–Fock of OH (6-31G) | −75.36316825 Ha, ⟨S²⟩ = 0.75377 | PySCF UHF, to 10⁻¹⁰ Ha |
| Hartree–Fock with d functions: H₂O / HCl / H₂S (cc-pVDZ); ROHF of triplet O₂ (cc-pVDZ) and CASCI in its orbitals (12 electrons, 8 orbitals) | −76.02677205 / −460.08944519 / −398.69453740; −149.60808447, −149.68777966 Ha | PySCF RHF, ROHF and CASCI, to 10⁻⁹ Ha |
| Gaussian integrals with d shells (6-31G*, cc-pVDZ, spherical and Cartesian) of H₂O, NH₃, CO, F₂, HCl, H₂S, SiH₄ and NaF; f and g shells in a custom basis | every integral to 2.6·10⁻¹³; f and g: PySCF's fingerprints to 10⁻¹⁰ | PySCF 2.14 |
| LiH on 10 qubits: UCCSD (92 parameters) / ADAPT-VQE (30 operators), error vs. full CI | 1.1·10⁻⁵ / 1.1·10⁻⁵ Ha | chemical accuracy 1.6·10⁻³ Ha |
| VQD, all four levels of H₂ (Sz = 0) | equal to full CI to 10⁻⁶ Ha | PySCF full CI |
| Trotter–Suzuki on a 4-qubit Heisenberg chain: error ratio when the steps double, orders 1 / 2 / 4 | 2.01 / 4.00 / 16.06, always below the error bound | 2, 4, 16 |
| Maximum-likelihood amplitude estimation: slope of log RMSE against log oracle calls | −0.99 (sampling: −0.49) | −1 (Heisenberg limit), −1/2 |
| GHZ state on 10 000 qubits (stabilizer simulator): ⟨Z₀Z₉₉₉₉⟩ / ⟨Z₄₃₂₁⟩ / ⟨X^⊗n⟩ | 1 / 0 / 1 | 1 / 0 / 1 |
| Graph-state entanglement across the two halves of the hypercube Q₁₀ / of the cycle C₁₀₀₀ | 512 / 2 ebits | cut-rank over GF(2): 512 / 2 |
| Random Clifford circuits (200, 1–8 qubits): stabilizer vs. state-vector simulator | fidelity 1 to 10⁻¹⁰ | — |
| Trotterized XX chain, 100 qubits, 6 steps (matrix-product state): ⟨Z_j⟩ and ⟨Z_i Z_j⟩ | equal to the free-fermion values to 10⁻⁸ | exact free-fermion evolution of the same circuit |
| QAOA on a ring of 100 nodes (matrix-product state), p = 1 / 2 / 3 | 0.75 / 0.8333 / 0.875 | (2p+1)/(2p+2), Mbeng et al. 2019 |
| Rotated surface code, d rounds, circuit-level noise p = 0.001, union-find decoding: logical error rate at d = 3 / 5 / 7 | 8.4·10⁻⁴ / 1.4·10⁻⁴ / 2.1·10⁻⁵ (10⁶ shots; falls with d; rises with d at p = 0.01) | below threshold ⇔ falls with d |
| Minimum-weight matching: correction weight on random error models, every syndrome | equal to the lightest error set found by enumeration | enumeration |
| Minimum-weight matching vs. PyMatching, same error model and syndromes, 8 circuits × 10⁵ shots | 1 of 800 000 predictions differs | PyMatching 2.4 |
| Sparse matching vs. the exact decoder: every syndrome of 400 random models, 6400 sampled and random shots, odd cycles | equal correction weights on every shot | `MatchingDecoder`, enumeration |
| BP+OSD with exhaustive OSD: every syndrome of 150 random models with hyperedges | the most likely correction | enumeration |
| BP+OSD vs. ldpc: colour codes d = 7, 9, [[72, 12, 6]], [[144, 12, 12]], bit flips p = 0.05, three BP rules; exhaustive OSD on random models | identical corrections (product-sum on [[144, 12, 12]]: 0.3% differ) | ldpc 2.4 |
| Colour and bivariate bicycle codes | [[(3d² + 1)/4, 1]] with distance d (searched for d = 3, 5); [[72, 12]], [[90, 8]], [[108, 8]], [[144, 12]], [[288, 12]]; every weight ≤ 2 error corrected on d = 5, 7 and [[72, 12, 6]] | Bravyi et al. 2024; enumeration |
| Stim-generated repetition and surface-code circuits, imported | error models equal to Stim's, probabilities to 10⁻¹⁵ relative | Stim 1.16 |
| Repetition code, one round of bit flips, union-find decoding, all 2ᵈ error patterns (d = 3 … 9) | fails exactly on the patterns with more than d/2 flips | majority vote |
| Detector error models of memory experiments vs. Stim | identical terms, probabilities to 10⁻¹⁵ relative | Stim 1.16 |
| Decomposed error models of Stim's surface and repetition circuits (d = 3 to 7) | identical errors and parts, probabilities to 10⁻¹⁵ relative | Stim 1.16 |
| Two-pass matching vs. PyMatching `enable_correlations=True`, Stim's surface codes, same model and syndromes | 0.01–0.04% of predictions differ; d = 5, p = 0.006: 1.93·10⁻² against PyMatching's 1.94·10⁻² (plain matching 2.44·10⁻²) | PyMatching 2.4 |
| Circuit distance: random models; Stim's surface and repetition circuits | equal to enumeration; equal to `shortest_graphlike_error` and no larger than `search_for_undetectable_logical_errors` | enumeration; Stim 1.16 |
| [[72, 12, 6]] memory, depth-8 cycle of Bravyi et al. vs. greedy CNOT layers, 3 rounds | circuit distance 6 vs. 4 (both exact) | Bravyi et al. 2024: ≤ 6 |

The test-suite (572 Rust tests and 296 Python tests, run in CI on Linux
(x86_64 and arm64), macOS and Windows; together with the command line
and the Python tests they cover 92.0 % of the Rust lines and 81.5 % of
the branches) also checks:

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
  result;
- noise: device models against hand-built channel sequences, Pauli twirls
  against the exact average over all 16 frames, PEC against the exact sum
  over all corrections (to 10⁻¹²), and 95 % confidence intervals that cover
  the exact value in 91–99 % of repeated estimates;
- error correction: Pauli-frame sampling against shot-by-shot tableau
  simulation on random noisy Clifford circuits, error models against
  sampled detector rates, every single fault (and, at distance 5, every
  pair of faults) of the surface code corrected in both bases, and the
  blossom algorithm against dynamic programming on 4000 random graphs.

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
* Its graph state is a 13-qubit GHZ state up to local Cliffords: local
  complementation at any node leaves the star, and every bipartition
  carries exactly one ebit.

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

Prebuilt wheels for Linux (x86_64, aarch64; glibc and musl), macOS (Intel
and Apple silicon) and Windows (x86_64, arm64); one `abi3` wheel covers
CPython 3.9 and newer.

**Rust library** ([crates.io](https://crates.io/crates/metatron-qso-rs),
Rust ≥ 1.89)

```bash
cargo add metatron-qso-rs
```

The library is imported as `metatron_qso`. Features: `parallel` (default;
multi-threading via rayon) and `cli`.

**JavaScript** ([npm](https://www.npmjs.com/package/metatron-qso); Node.js
18+, browsers)

```bash
npm install metatron-qso
```

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
| [`metatron_atlas`](metatron-qso-rs/examples/metatron_atlas.rs) | The Metatron family of 13-node graphs: symmetry, reduction from the centre, state transfer, and continuous-time, coined and Szegedy search |
| [`qaoa_maxcut`](metatron-qso-rs/examples/qaoa_maxcut.rs) | QAOA depth 1–4 with INTERP on rings, Q₃, Petersen, random 3-regular and Fruit-of-Life graphs |
| [`vqe_h2`](metatron-qso-rs/examples/vqe_h2.rs) | VQE for H₂ with every optimiser, shot-based and noisy energy estimates |
| [`circuits`](metatron-qso-rs/examples/circuits.rs) | GHZ state, OpenQASM export, Grover's sine law, phase estimation |
| [`noise_mitigation`](metatron-qso-rs/examples/noise_mitigation.rs) | A toy device: trajectories against density matrices, readout mitigation, ZNE, PEC and classical shadows on a GHZ state |
| [`stabilizers`](metatron-qso-rs/examples/stabilizers.rs) | A GHZ state on 10 000 qubits, graph-state entanglement of Metatron's Cube, the Fruit of Life, C₁₀₀₀ and Q₁₀, local complementation |
| [`mps`](metatron-qso-rs/examples/mps.rs) | An XX-chain quench on 200 qubits and QAOA on a ring of 200 nodes on matrix-product states, with the fidelity bound |
| [`surface_code`](metatron-qso-rs/examples/surface_code.rs) | Logical error rates of rotated surface codes (d = 3, 5, 7; Z and X basis) and repetition codes under circuit-level noise, below and above threshold, with union-find and matching decoders |
| [`molecules`](metatron-qso-rs/examples/molecules.rs) | LiH under four encodings against full CI, UCCSD, k-UpCCGSD and ADAPT-VQE; VQD excited states of H₂; Trotter errors against their bounds |
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
| `graph [SPEC]` | Structure, Laplacian spectrum, automorphism group and orbits, strongly regular parameters | `--save FILE` |
| `iso A B` | Isomorphism test with a mapping | — |
| `walk` | Node probabilities at time t and limiting distribution | `--graph`, `--start`, `--time`, `--gamma`, `--generator laplacian\|adjacency` |
| `search` | Quantum-walk spatial search (rate and time optimised unless `--gamma` is given) | `--graph`, `--marked 3,7`, `--gamma`, `--generator`, `--t-max` |
| `reduce` | Equitable partition from a start node and the reduced walk | `--graph`, `--start`, `--time` |
| `transfer` | Perfect state transfer or the best transfer in a window; pretty good state transfer decided exactly on paths and cycles | `--graph`, `--from`, `--to`, `--t-max` |
| `coined` | Coined-walk search | `--graph`, `--marked`, `--coin`, `--steps` |
| `hitting` | Szegedy and classical hitting times | `--graph`, `--marked` |
| `walk-circuit` | A walk as an OpenQASM circuit | `--graph`, `--time`, `--method exact\|trotter`, `--steps`, `--order` |
| `qaoa` | QAOA for MaxCut with exact optimum and approximation ratio | `--graph`, `--p`, `--restarts`, `--seed`, `--layerwise` |
| `vqe` | VQE with an EfficientSU2 ansatz, compared with the exact ground energy | `--model h2\|tfim:N,J,H\|heisenberg:N,J,Δ`, `--hamiltonian "<Pauli sum>"`, `--qubits`, `--layers`, `--method`, `--restarts`, `--seed` |
| `grover` | Grover search, simulated vs. theoretical success | `--qubits`, `--marked`, `--iterations` |
| `qasm FILE` | Run an OpenQASM 2 or 3 program and sample its bits, or convert it; Clifford programs beyond 34 qubits on the stabilizer simulator | `--shots`, `--seed`, `--export qasm2\|qasm3`, `--stabilizer`, `--mps`, `--max-bond N`, `--single`, `--noise p1,p2`, `--device FILE`, `--trajectories N` |
| `qec` | Repetition, surface, colour or bivariate bicycle code memory experiment, or a Stim circuit file: logical error rate with union-find, matching, correlated matching or BP+OSD decoding; optionally the circuit distance | `--code surface\|repetition\|color\|bb`, `--basis z\|x`, `--decoder union-find\|matching\|correlated-matching\|bposd`, `--schedule greedy\|bicycle\|0,1,2,3,4,5`, `--circuit-distance`, `--distance 3,5,7`, `--rounds`, `--p 0.001,0.01`, `--shots`, `--stim FILE`, `--circuit FILE` |
| `estimate FILE` | ⟨H⟩ with standard error, optionally mitigated | `--observable`, `--shots`, `--device`, `--mitigate-readout`, `--zne 1,3,5`, `--pec N` |
| `scf` | Hartree–Fock of a molecule from its geometry, full CI in an active space, FCIDUMP output | `--atoms`, `--basis`, `--charge`, `--multiplicity`, `--unrestricted`, `--frozen`, `--active`, `--fci`, `--fcidump` |
| `molecule FILE` | Qubit Hamiltonian, Hartree–Fock and full-CI energies of an FCIDUMP molecule, and a variational solution | `--mapping`, `--two-qubit-reduction`, `--method none\|uccsd\|upccgsd\|adapt`, `--roots` |
| `trotter` | Trotter–Suzuki circuit for exp(−iHt) with its error bound | `--hamiltonian`, `--time`, `--steps`, `--order`, `--output FILE` |

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
| `johnson:N,K`, `kneser:N,K`, `hamming:D,Q` | Distance-regular families |
| `paley:Q`, `shrikhande`, `rook:MxN` | Strongly regular graphs |
| `circulant:N,S1,S2,…` | Circulant graph with jumps ±Sᵢ |
| `random-regular:N,D[,SEED]`, `erdos-renyi:N,P[,SEED]` | Random graphs |

Every graph argument of the CLI also accepts a file: an edge list, or
GraphML for `.graphml` and `.xml`.

## Performance

Measured with `cargo run --release --example benchmark_report` on a
4-thread cloud VM (full tables, method and run-to-run variance in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md)):

| Workload | Time |
|---|---:|
| 16-qubit layer (47 gates: H, RZ, CX chain) | 0.55 ms (4.7× faster than 1.2.0) |
| 24-qubit layer (71 gates) | 0.21 s (0.18 ns per gate·amplitude; 2.3× faster than 1.2.0) |
| Full gradient, 10 qubits, 100 parameters, adjoint | 0.47 ms (53× faster than parameter shift) |
| 256 values and gradients of that ansatz, one batch call | 34 ms (3.7× faster than a loop) |
| QAOA value + gradient, 20 nodes, p = 3 | 0.13 s |
| Quantum walk on a 100 × 100 torus (10 000 nodes), Krylov | 78 ms |
| Automorphism group of Metatron's Cube (13!) | 2.7 ms |
| Permanent of a 24 × 24 complex matrix | 0.18 s (4.8× faster than The Walrus) |

Timings varied between runs on this shared VM (up to 2× for
sub-millisecond workloads, up to 1.5× for the larger ones).

On the same machine, with every framework at its default settings and
warmed up, QSO 1.3 simulates 20–24-qubit circuits 1.5–2.6× faster than
Qulacs, 3.3–4.7× faster than Qiskit Aer and 3.7–9× faster than PennyLane
Lightning; at 16 qubits Qulacs was 1.3× faster in the median. Its adjoint
gradient is the fastest of those compared (3.8× faster than Qulacs'
backprop), and its permanents are 1.8–8.7× faster than The Walrus. Since
1.24 the state-vector kernels run on a worker pool that stays awake
between short simulations: from 10 to 18 qubits QSO is ahead of Qulacs
both warmed up and after pauses of 0.5 and 20 ms (one layer at 16 qubits:
0.51 against 1.19 ms warm, 0.77 against 1.42 ms after 20 ms). Method and
tables: [docs/BENCHMARKS.md](docs/BENCHMARKS.md#comparison-with-other-simulators-130)
and the [roadmap](docs/ROADMAP.md) (1.24).

## Scope and limits

QSO is a classical simulator running on one machine (multi-threaded CPU).
Know its boundaries:

| Limit | Value | Reason |
|---|---|---|
| Pure-state register | 34 qubits; 35 in single precision | 2³⁴ amplitudes = 256 GiB; practical limit is your RAM (16 bytes per amplitude, 8 in single precision) |
| Stabilizer register | 16 384 qubits, Clifford gates only | n²/2 bytes, about twice that while measuring. A gate costs n/32 word operations, a measurement O(n²) (about a second at the limit), sampling every qubit O(n³/64) once (about two minutes at the limit) |
| Matrix-product state | 65 536 qubits, bond dimension up to 4096, 2²⁸ tensor entries | A gate between neighbours costs O(χ³): about 15 ms at χ = 64, 0.13 s at χ = 128, 0.9 s at χ = 256 on one core; distant gates add two swaps per site in between. Highly entangled circuits need χ = 2^(n/2) and belong on the state vector |
| Detector circuit | 65 536 qubits, 2²⁴ operations, 64 observables | Pauli-frame sampling: one word operation per operation and 64 shots. Surface code d = 21 with 21 rounds (881 qubits, 4840 detectors): error model 0.5 s, 100 000 decoded shots 15 s on one core |
| Density-matrix register | 14 qubits | 4¹⁴ entries = 4 GiB. Quantum trajectories reach the pure-state limit at the price of statistical error (and up to three state vectors in memory per running trajectory) |
| Noise channels | one and two qubits | Kraus operators of channels on more qubits are not supported; noise after larger gates acts on consecutive pairs |
| Probabilistic error cancellation | Pauli noise; overhead grows as Γ = Π γ | Twirling turns other noise into Pauli noise on average; the standard error grows with Γ, exponentially in the number of noisy gates |
| Classical shadows | qubits 0–63 | Random Pauli measurements; weight-k observables need ∝ 3ᵏ snapshots |
| Pauli strings | qubits 0–63 | Stored as two 64-bit masks |
| Custom unitary gates | up to 6 target qubits | Dense kernel |
| Exact walk analysis | up to 2048 nodes, or 2048 cells of an equitable partition | Dense eigendecomposition. Beyond that, `evolve`/`probabilities` use Krylov, and limiting distributions the start node's cyclic subspace when it is small; symmetric graphs of any size reduce to their quotients |
| Automorphism groups, canonical labelling | intended for up to a few thousand nodes | Exact search, worst-case exponential like every known algorithm |
| OpenQASM | version 2.0 and an OpenQASM 3 subset | OpenQASM 3's classical variables and arithmetic, loops, subroutines, calibrations and timing are rejected with an error naming the feature. Exact distributions follow up to 4096 measurement branches; beyond that, sampling simulates shot by shot |
| WebAssembly | 26 qubits; graphs and walks up to 2048 nodes | 4 GiB address space; single-threaded |
| Full CI | 2¹⁸ determinants, 32 roots (dense up to 1024) | Davidson on `H c` without the matrix above 1024 determinants; molecules up to 32 spatial orbitals can be mapped to qubits |
| Product-formula bound, order ≥ 4 | rigorous but loose | Falls only as `r⁻²` with the steps until they are short; the actual error falls as `r⁻⁴` |

Not included: GPU acceleration, tensor-network simulation, simulation of noisy Clifford circuits at scale (planned for 1.8),
molecular integrals in larger basis sets (triple zeta and beyond) or beyond argon (QSO reads such
integrals from FCIDUMP files written by a quantum-chemistry code), and
submission to quantum hardware. To run on hardware, export a circuit or
program to OpenQASM 2 or 3 and submit it with the vendor's SDK.

## Conventions

* Qubit 0 is the least significant bit of a basis index (as in Qiskit); bit
  strings print qubit n−1 first.
* Time evolution is `exp(−iHt)`; rotations are `R_P(θ) = exp(−iθP/2)`;
  `P(λ) = diag(1, e^{iλ})`; `U(θ, φ, λ)` is the standard U3 matrix, which
  OpenQASM 2's `U` equals up to a global phase; OpenQASM 3's `U` is
  `e^{iθ/2}` times it, a phase QSO keeps where it matters (under
  control).
* Quantum walks use `H = γL` (Laplacian) or `H = −γA` (adjacency).
* Invalid input returns `QsoError` in Rust and raises `ValueError` in
  Python.

## Repository layout

```
metatron-qso-rs/            Rust library `metatron_qso` and the `qso` CLI
  src/quantum/              states, gates, kernels, circuits, Pauli sums, OpenQASM
  src/noise/                density matrices, channels, noise and device models, quantum trajectories
  src/estimation/           estimates with standard errors, shot allocation, classical shadows
  src/mitigation/           readout calibration, zero-noise extrapolation, PEC, Pauli twirling
  src/algorithms/           gradients, ansätze, models, VQE, QAOA, Grover, QFT/QPE, classifiers,
                            excited states, state geometry, time evolution, QUBO/QAOA mixers,
                            amplitude estimation
  src/chemistry/            fermionic operators, qubit mappings, Gaussian integrals, Hartree–Fock, FCIDUMP molecules, UCC ansätze, ADAPT-VQE
  src/optimize/             L-BFGS, Adam, gradient descent, Nelder–Mead, SPSA
  src/graph/                graphs, generators, Metatron family, automorphisms, canonical forms, file I/O
  src/walks/                continuous- and discrete-time walks, quotients, state transfer, search, circuits
  src/linalg/               Hermitian eigensolvers, Lanczos, Krylov, sparse matrices
  src/stabilizer/           stabilizer tableaux, graph states, Clifford programs
  src/qec/                  detector circuits, frame sampling, error models, decoding, codes
  src/mps/                  matrix-product states, OpenQASM programs on them
  src/photonics/            permanents, hafnians, boson sampling, Gaussian states
  src/backend.rs            execution backends and shot-based estimation
  src/bin/qso.rs            command-line tool
  tests/                    validation, robustness and property tests
  examples/, benches/       runnable examples, criterion benchmarks
metatron_qso_py/            Python package (PyO3 + NumPy): src/, python/, tests/, examples/
metatron_qso_wasm/          WebAssembly module (JSON interface) and the npm package in js/
docs/                       roadmap, architecture, benchmarks, audit of v0.1, 1.0 plan
docs/guide/, docs/tutorials/ user guide (mdBook) and executed notebooks of the documentation site
docs/playground/            browser playground (part of the documentation site)
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

# WebAssembly and the npm package (Node.js 18+)
rustup target add wasm32-unknown-unknown
python scripts/build_wasm.py
(cd metatron_qso_wasm/js && npm test)
```

CI runs formatting, Clippy and rustdoc; tests on Linux, macOS and Windows; a
build without multi-threading; the minimum Rust version; all examples and
CLI smoke tests; the Python tests on 3.9 and 3.13; and the core tests
built for WebAssembly and run under Node.js, with the npm package's
tests. Pushing a tag `vX.Y.Z` builds and tests the wheels, the source
distribution and the npm package and publishes the release to crates.io,
PyPI, npm and GitHub
([CONTRIBUTING.md](CONTRIBUTING.md#releasing)).

New features follow one rule ([CONTRIBUTING.md](CONTRIBUTING.md)): a claim
needs a test against an independent reference.

## Documentation

| Document | Content |
|---|---|
| [User guide](https://lashsesh.github.io/qso/) | Guide, executed tutorials and Python API reference (sources in [docs/guide](docs/guide/) and [docs/tutorials](docs/tutorials/)) |
| [Playground](https://lashsesh.github.io/qso/playground/) | OpenQASM programs, quantum walks and spatial search in the browser ([source](docs/playground/)) |
| [metatron_qso_wasm/js/README.md](metatron_qso_wasm/js/README.md) | JavaScript API of the npm package |
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
