# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Interacting walkers** (milestone 1.29). `walks::InteractingWalk`
  simulates N bosons, fermions or distinguishable particles on a graph:
  hopping with the adjacency or Laplacian generator, on-site interaction
  U and neighbour interaction V. It gives exact evolution in the
  occupation basis, densities, pair correlations, propagators and graph
  certificates (`certificate`, `certificate_distance`, compared against
  `certificate_tolerance`, which grows with the time).
  - Validated against permanents and determinants of the single-walker
    propagator, the tensor-product space, and the Hubbard dimer.
  - Two free bosons cannot tell apart T(8) and the three Chang graphs, or
    the Shrikhande and 4 × 4 rook's graphs; two interacting bosons can
    (validation suite).
  - Walks beyond 2²² configurations, 2²⁵ stored sites or 2²⁵ nonzero
    Hamiltonian entries are refused instead of exhausting memory.
  - Python: `InteractingWalk`, `interacting_certificate_distance`. CLI:
    `qso particles`. Tutorial: `08_interacting_walks`.
- **Correlated and non-Markovian noise** (milestone 1.30).
  - Channels (Kraus, Pauli, unitary, rotations) act on up to four qubits,
    on density matrices and in trajectories. In device rules, a k-qubit
    channel acts on every k consecutive qubits of the gate.
  - `GateRule::with_spectator`: channels on fixed qubits after a gate,
    for crosstalk on qubits outside it, which wait for the gate.
  - `Circuit::id` and rules for the gate `"id"` make explicit waits.
  - Quasi-static detunings (`QubitProperties::with_detuning`, Gaussian or
    uniform), random telegraph noise (`with_telegraph`) and quasi-static
    over-rotations (`GateRule::with_over_rotation_spread`) are drawn per
    trajectory and simulated exactly by `TrajectorySimulator`. Density
    matrices and PEC refuse them (`DeviceModel::is_markovian`).
  - Validated against `exp(−σ²t²/2)`, the exact echo of a static
    detuning, and the Kubo–Anderson decay of telegraph noise and its echo.
  - PEC inverts Pauli channels on up to four qubits.
  - The same in Python, with JSON round trips.
  - A one-qubit identity unitary (`Circuit::id`) is named `"id"` in
    `gate_counts` and device rules (it was `"unitary"`); OpenQASM `id q;`
    now reads as `Circuit::id` instead of being dropped, and is written
    back as `id`; the stabilizer simulator accepts it.
  - A `NoiseModel` channel on more qubits than a gate has is an error at
    that gate instead of being ignored.
- **Graphs:** `generators::seidel_switching` (within the size limits of
  the generators, keeping node positions) and the three Chang graphs
  (`generators::chang`, spec `chang:K`), in Python as
  `Graph.seidel_switching` and `Graph.chang`.

### Fixed

- `ContinuousWalk` beyond 2048 nodes reported that dense analysis
  "supports at most 2048 qubits"; the message now speaks of nodes.
- Documentation site: the guide of 1.7.0 at `/1.7.0/` was missing. Its
  Rust tutorials are rendered from the tag's examples, which the build of
  a release did not do, so mdBook failed and the version was left out.
  The coverage report at `/coverage/` now has an index page (the report
  itself is at `/coverage/html/`). The documentation build a release
  starts no longer cancels the one of its merge on `main` but waits for
  it; the cancelled build had left the site without a deployment.

## [1.7.0] — 2026-10-28

Milestones 1.14 to 1.28 of the [roadmap](docs/ROADMAP.md), its third to
sixth stages, released together:

- **third stage**: 1.14 (Gaussian boson sampling and imperfect photons),
  1.15 (molecular integrals), 1.16 (fast matching), 1.17 (mitigation by
  purification);
- **fourth stage**: 1.18 (pretty good state transfer), 1.19 (decoding
  beyond matching), 1.20 (single precision), 1.21 (coverage and
  differential testing);
- **fifth stage**: 1.22 (error-correction circuits that keep their
  distance), 1.23 (larger basis sets and open shells), 1.24 (short
  simulations at full speed), 1.25 (documentation and reach);
- **sixth stage**: 1.26 (gradients at the cost of a few simulations),
  1.27 (variational training in single precision), 1.28 (28 qubits end to
  end).

New results are checked against The Walrus, PySCF, Stim, PyMatching,
`ldpc`, Qulacs and Qiskit Aer, closed forms and enumeration (named
below). Every validated result of 1.6.0 still holds (the validation tests
only grew); energies and gradients differ from 1.6.0 by rounding where
kernels changed. The public Rust API only grows (`cargo-semver-checks`
against 1.6.0).

### Added

- **QAOA at 22–26 qubits against Qulacs and Qiskit Aer** (1.28) in the
  cross-validation (`run.py --qaoa 22,24`): p = 10 on a random 3-regular
  graph, energy and gradient of QSO (`MaxCutQaoa` and the generic
  circuit, both precisions) and Qulacs (`backprop`), Aer's energy (it has
  no gradient), each in its own process with time and peak memory. In
  double precision values and gradients agree to 5.4·10⁻¹⁴ (relative); at
  26 qubits `MaxCutQaoa` takes 16.4 s for energy and gradient, the
  generic circuit 81 s, Qulacs 650 s and 5.0 GiB, Aer 49.5 s for the
  energy alone.
- **`examples/qaoa_large.rs`** (1.28): QAOA at depth 10 on 28 qubits on one
  machine, with angles optimised layer by layer on 16 nodes, carried
  over and refined with L-BFGS in single precision. `benchmark_report`
  times p = 10 at 22 and 24 qubits in both precisions.
- **Variational training in single precision** (1.27): energies and
  adjoint gradients on single-precision states, with sums in double
  precision and results divided by the state's norm²:
  `gradients::expectation_and_gradient_single`, `Vqe::with_precision`,
  `MaxCutQaoa::with_precision` and `qaoa::optimize_layerwise_at` in Rust;
  `precision="single"` for `expectation_and_gradient`, `run_vqe`,
  `MaxCutQaoa` and `optimize_qaoa_layerwise` in Python; `--single` for
  `qso vqe` and `qso qaoa`. Half the memory (28 qubits: 4.0 GiB instead of
  8.0 GiB for VQE, 4.5 GiB instead of 8.5 GiB for `MaxCutQaoa`) and a fifth
  to a third less time; energies agree with double precision to 3·10⁻⁸
  and gradients to 1.1·10⁻⁶ relative at 20–24 qubits. In single
  precision L-BFGS treats energies within their rounding (10⁻⁷ of the
  value) as equal and decides by the slope (Hager and Zhang's
  approximate Wolfe condition), and L-BFGS and Adam stop when the
  gradient reaches its rounding; QAOA reaches the same ratios as in
  double precision to 10⁻⁸. Parameter-shift and finite-difference
  gradients stay in double precision.

- **Matrix-product states in WebAssembly**: the module's `mps` method and
  npm `qso.mps(source, {shots, seed, maxBond, cutoff, observables})` run
  OpenQASM programs on matrix-product states, with Pauli expectation
  values, bond dimensions and entropies, amplitudes (up to 10 qubits),
  the discarded weight, the fidelity bound and samples. The npm tests
  replay requests whose native responses are recorded in
  `js/test/mps_native.json` and require the same results to 10⁻¹⁰. The
  playground has a tab for them (GHZ on 1000 qubits, a quench on 100).
- **Documentation of every release**: the site carries the guide and
  Python reference of each release since 1.2.0 at `/<version>/`, built
  from the release's tag with its wheel from PyPI, and a version selector
  on every guide page (`docs/build_site.py --versions`).
- **Rust tutorials**: eight guide pages built from the crate's examples
  (code between `// ANCHOR:` lines, output from running them in the
  documentation build).
- **CLI tests** (`tests/cli.rs`): every subcommand run as a program with
  its JSON output checked against known values.
- **Coverage of the CLI and the Python tests**: `scripts/coverage.sh`
  merges the crate's tests, the CLI run by `tests/cli.rs` and the Python
  tests on an instrumented extension into one report, with totals per
  component.

- **d and higher shells, 6-31G* and cc-pVDZ to argon, ROHF**
  (`chemistry::integrals`, `chemistry::scf::rohf`; Python
  `AoIntegrals(..., harmonics=...)`, `hartree_fock(...,
  restricted_open=True, harmonics=...)`; `qso scf --restricted-open
  --cartesian`).
  - McMurchie–Davidson integrals for shells up to g, with Hermite tables
    sized at run time; spherical shells (real solid harmonics in PySCF's
    order and signs) or Cartesian ones, each normalised to one.
  - STO-3G, 6-31G, 6-31G* and cc-pVDZ for H to Ar, generated by
    `scripts/generate_basis_data.py` from PySCF's copy of the Basis Set
    Exchange.
  - Restricted open-shell Hartree–Fock with PySCF's effective Fock
    matrix; its orbitals give active-space Hamiltonians of open shells.
  - Repulsion integrals per quartet of shells, the columns of a general
    contraction together (64 functions, three chlorines in cc-pVDZ: 3 s).
  - Checked against PySCF 2.14: every integral of the cross-validation
    molecules (d shells, H to Ar) to 2.6·10⁻¹³; f and g shells in a custom
    basis by PySCF's fingerprints and single integrals; RHF, UHF, ROHF and
    CASCI energies to 10⁻⁹ Ha.

- **Circuit distance, syndrome schedules and correlated matching**
  (`qec::distance`, `qec::correlated`, `css_code_memory_with`; Python
  `circuit_distance`, `graphlike_distance`, `CorrelatedMatchingDecoder`,
  `css_code_memory(..., schedule=...)`; `qso qec --decoder
  correlated-matching --schedule ... --circuit-distance`).
  - `graphlike_distance`: the lightest undetectable logical error among
    errors that flip at most two detectors, exactly (as Stim's
    `shortest_graphlike_error`). `circuit_distance`: exact for graph-like
    models; otherwise a proven lower bound from an exhaustive search and an
    upper bound with a witness from BP+OSD (whether any logical error
    exists at all is decided by GF(2) linear algebra).
  - `CssSchedule::Bicycle`: the depth-8 syndrome cycle of Bravyi et al.
    (2024) for bivariate bicycle codes; [[72, 12, 6]] keeps circuit
    distance 6 (greedy layers: 4). `CssSchedule::Directions` for colour
    codes: every order gives (d + 1)/2 for d = 3 and 5, as greedy layers
    do.
  - `CorrelatedMatchingDecoder` and `correlated_error_model`: two-pass
    matching with PyMatching 2.4's reweighting rule; on Stim's surface
    codes it agrees with PyMatching on all but 0.01–0.04% of shots.
  - `DecoderChoice` (non-exhaustive) and `logical_error_rate_using` for
    every decoder.

- **Differential testing and coverage** (`differential::compare`).
  - One circuit on every simulator that takes it — single-precision state
    vector, density matrix, trajectory, unbounded MPS, stabilizer tableau,
    OpenQASM 2 and 3 round trips — with each deviation from the
    double-precision state vector.
  - Amplitudes compared after aligning the global phase (linear in the
    error), norms and traces separately; property tests on random
    circuits of every gate kind, controls, dense unitaries, parameters,
    1 to 10 qubits (256 per test on every change, 20 000 in the weekly
    job) and a `differential` fuzz target.
  - Fixed on the way: an MPS without cutoff turned to NaN (singular
    values below the rounding of the Gram matrix are now always
    dropped); the tableau refuses rotations by angles above about 5.6·10⁸ rad,
    where any angle passed as Clifford from 4·10¹⁴ on.
  - Line and branch coverage measured in CI and published with the
    documentation (90.5 % of lines, 79.4 % of branches).

- **Single-precision state vectors** (`quantum::StateVector32`, Python
  `StateVector32` and `simulate(..., precision="single")`, `qso qasm
  --single`).
  - The gate kernels and the fused, blocked execution plan are generic
    over the amplitude type; single precision has AVX versions with four
    amplitudes per register (bit for bit the portable arithmetic).
  - Half the memory, one more qubit (`MAX_QUBITS_SINGLE`); gate matrices
    in double precision, rounded once; sums in double precision.
  - Tests: 1 − F below 10⁻¹¹ after 2140 gates on 18 qubits, expectation
    values to 10⁻⁶; in the cross-validation, against double precision
    beside Qiskit Aer's single precision, and timed against both.

- **BP+OSD decoding, colour and bivariate bicycle codes** (`qec::bposd`,
  `qec::css`; Python `BpOsdDecoder`, `CssCode`, `css_code_memory`; `qso qec
  --code color|bb --decoder bposd`).
  - Belief propagation on the Tanner graph of a detector error model
    (min-sum with a fixed or adaptive factor, or product-sum; flooding
    schedule), then ordered-statistics decoding (OSD-0, combination sweep,
    exhaustive) on a reduced GF(2) basis; `DecoderChoice::BpOsd` decodes the
    undecomposed model.
  - `CssCode` from X and Z checks with paired logical operators;
    triangular 6.6.6 colour codes, bivariate bicycle codes and the table
    of Bravyi et al. (2024); memory experiments in either basis.
  - `DetectorErrorModel` can be built from a list of errors in Python;
    `MatchingGraph::to_dem`.
  - Tests: exhaustive OSD against enumeration; every weight ≤ 2 error on
    colour codes d = 5, 7 and the [[72, 12, 6]] code; code parameters; in
    the cross-validation, shot-by-shot agreement with `ldpc` and Stim's
    error models of the CSS circuits.

- **Pretty good state transfer, decided exactly** (`walks::pretty_good`,
  Python `PrettyGoodTransfer`, in `qso transfer`).
  - Kronecker's theorem on the integer relations among the eigenvalues,
    computed exactly in cyclotomic fields (power-basis coordinates modulo
    the cyclotomic polynomial, integer kernels by column reduction) for
    paths (adjacency and Laplacian), cycles and Cayley graphs of abelian
    groups; obstructions as witnesses; PST times from the same data.
  - Tests: the theorems of Godsil, Kirkland, Severini and Smith (paths,
    n ≤ 300), Banchi, Coutinho, Godsil and Severini (Heisenberg chains,
    n ≤ 200) and Pal (cycles, n ≤ 512); agreement with `StateTransfer` on
    hypercubes, circulants and paths; a PGST time checked against the
    propagated walk.

- **Virtual distillation** (`mitigation::distillation`, Python
  `distilled_expectation`, `dominant_expectation`, `DistillationCircuits`).
  - `Tr(ρᵏO)/Tr(ρᵏ)` exactly for k = 1 to 3, and the limit of the dominant
    eigenvector.
  - Measurement circuits on k copies of the register: per qubit, the
    k-qubit unitary `S_q P_q` diagonalised by two-level rotations
    (controlled single-qubit gates, exportable to OpenQASM); estimates
    from exact probabilities or from counts with a standard error.
  - Tests: closed forms for random mixtures, the circuits on copies of ρ
    against the exact value to 10⁻¹¹, shots within 4 standard errors; a
    depolarised GHZ-like state whose rotation comes last, from an error of
    0.19 to 3.6·10⁻³ (two copies) and 6.9·10⁻⁵ (three), and one with the
    rotation first, where the error stops at the dominant eigenvector's
    floor of 9·10⁻⁴; an over-rotated pure state is unchanged.

- **Sparse matching** (`qec::SparseMatchingDecoder`, Python
  `SparseMatchingDecoder`): minimum-weight perfect matching by growing
  regions on the detector graph (the idea of sparse blossom), with
  `decode_with_weight` on both matching decoders.
  - The same correction weights as `MatchingDecoder` on every tested
    shot (every syndrome of random models, sampled memory experiments,
    dense random syndromes with nested and shattering blossoms).
  - At d = 25, p = 0.001: 95 µs per shot instead of 94 ms; 1.1 to 1.9
    times PyMatching's time on surface codes from d = 11 to 25 (up to 2.3
    at p = 0.01 to 0.05); a chain of 2001 events that all fire (deeply
    nested blossoms) 51 ms, PyMatching 1.3 s.

- **Molecules from their geometry** (`chemistry::integrals`,
  `chemistry::scf`; Python `Molecule`, `AoIntegrals`, `hartree_fock`;
  CLI `qso scf`).
  - Overlap, kinetic, nuclear-attraction and electron-repulsion integrals
    over contracted Cartesian Gaussians (McMurchie–Davidson), STO-3G and
    6-31G for hydrogen to neon.
  - Restricted and unrestricted Hartree–Fock with DIIS from a
    superposition of atomic densities (or the core Hamiltonian), a
    level-shifted fallback when DIIS does not converge, ⟨S²⟩, and the
    transformation to molecular orbitals with a frozen core and an active
    space, giving `MolecularIntegrals` (qubit Hamiltonians, full CI,
    FCIDUMP output).
  - Tests against PySCF 2.14 on the same geometries: integrals to
    1.4·10⁻¹³; restricted energies of H₂, LiH, BeH₂, H₂O and N₂ and
    unrestricted energies of OH, CH₂, H₂O⁺, Li, NH₂ and NO to 10⁻¹⁰ Ha;
    full CI and CASCI to 10⁻⁹ Ha; the bundled FCIDUMP files reproduced
    from geometries.
- **Hafnians** (`photonics::hafnian`, Python `hafnian`,
  `hafnian_repeated`, `torontonian`).
  - Hafnians and loop hafnians by the formula of Björklund, Gupt and
    Quesada (2019), Kan's formula for matrices with repeated rows, and
    torontonians.
  - Tests: brute force, (2n − 1)!! and the telephone numbers for the
    all-ones matrix, perm(B) for bipartite graphs, and the perfect
    matchings of the hypercubes Q₃ to Q₅ (9, 272, 589 185), the
    dodecahedron, icosahedron, Petersen and Heawood graphs.
- **Gaussian states** (`photonics::GaussianState`, Python
  `GaussianState`).
  - Covariance matrices and means with ħ = 2, as in The Walrus.
  - Squeezing, displacement, rotation, interferometers and loss, also
    into a thermal environment.
  - Photon-number probabilities (loop hafnians, over N rows instead of
    2N for pure states; Kan's formula for mixed states with many photons
    in few modes when it is cheaper), threshold-click probabilities
    (torontonians), and exact samples drawn mode by mode.
  - Tests: squeezed, coherent, thermal and two-mode squeezed states
    against their closed forms, loss as binomial thinning, sums to one,
    sampling frequencies.
- **Imperfect photons** (`photonics::ImperfectBosonSampler`, Python
  `ImperfectBosonSampler`).
  - Partial distinguishability from the Gram matrix of the internal
    states (Tichy; Shchesnovich).
  - Losses at the inputs (through environment modes, exact also when
    partially distinguishable photons share a lossy mode) and detector
    efficiencies.
  - Tests: a simulation with the internal states as extra modes of an
    interferometer with environment modes; the Hong–Ou–Mandel
    visibility.
- **Cross-validation** against PySCF 2.14 (integrals to 1.4·10⁻¹³,
  energies to 10⁻¹⁰ Ha) and against The Walrus 0.22: probabilities differ by
  about 10⁻¹⁵ (absolute, 4 modes) and by 2·10⁻¹¹ relative at 20 photons;
  hafnians by 7·10⁻¹² relative up to 20 × 20.

### Changed

- **`MaxCutQaoa` twice as fast at 22–28 qubits** (1.28). The mixer runs as
  real Y rotations in the frame `S^{⊗n}`, which commutes with the cost
  layers (only `state()` leaves it, with an exact phase); qubits above the
  cache block are rotated in tiles, up to five per pass over memory; the
  backward pass takes the mixer's overlap in the passes that undo it, with
  the cost gradient and phases in the same block pass. Energy and
  gradient at p = 10 on 4 cores: 26 qubits 16.3 s instead of 35.8 s
  (single precision 13.9 s instead of 28.5 s), 28 qubits 79 s instead of
  160 s (62 s instead of 122 s). Results change by rounding.
- **Cheaper adjoint gradients** (1.26). Each gradient term is the matrix
  element of the gate's derivative between the two states, in one
  reading pass without copying the state; diagonal gates on qubits above
  the cache block are undone and differentiated block by block; diagonal
  observables (Z strings) are evaluated in one pass. On the 4-core VM the
  energy and gradient of a 260-parameter VQE ansatz at 26 qubits take
  41.6 s instead of 73.4 s, of QAOA p = 10 as a generic circuit 89 s
  instead of 221 s, with two state vectors instead of three. Diagonal
  observables (MaxCut, Ising and QUBO costs) are evaluated chunk by chunk,
  2–4 times faster for a QAOA cost at 22–24 qubits. Energies and gradients
  are bit for bit independent of the thread count; they differ from the
  previous version by rounding.
- **State-vector kernels on their own worker pool.** Rayon's workers
  sleep as soon as they are idle, and on virtual machines the first
  simulation after a pause paid for waking them. The gate kernels, cache
  blocks and sums now run on a pool whose threads spin for 200 µs after
  each job before parking (`QSO_POOL_SPIN_US`; threads from
  `QSO_NUM_THREADS` or `RAYON_NUM_THREADS`) and yield the core while
  they wait; rayon stays for batches, trajectories and the rest. Results
  are bit for bit unchanged. On the 4-core development VM a 16-qubit
  layer after a 20 ms pause takes 0.71–0.73 ms instead of 1.2–1.3 ms,
  warm 0.50–0.75 ms instead of 0.76–0.87 ms; from 20 qubits on the times
  are unchanged, and two QSO processes at once run faster than with
  rayon. `benchmark_report` and the cross-validation time these cold
  starts. Spinning costs CPU time; `QSO_POOL_SPIN_US=0` turns it off.
- `detector_error_model(circuit, decompose=true)` now splits errors as
  Stim does within each noise channel: decomposed models of Stim's
  generated circuits equal Stim's error by error. Before, every error
  with more than two detectors was split into all of its single-qubit
  parts, which made correlated matching less effective.
- `CssCode` equality compares the checks and logical operators only.
- `DecoderKind` is back to its 1.6.0 form (union-find and matching;
  `build` cannot fail): adding `BpOsd` to it (1.19) broke the 1.x
  compatibility promise. BP+OSD and correlated matching are variants of
  the new `DecoderChoice`; `logical_error_rate_with` keeps its signature.
- The matching decoder of `logical_error_rate(..., decoder="matching")`
  and `qso qec --decoder matching` is now `SparseMatchingDecoder`
  (the same weights; where corrections weigh the same, it may pick
  another one).
- `MolecularIntegrals.fci_energies` builds the determinant matrix by the
  Slater–Condon rules instead of through the qubit Hamiltonian up to 1024
  determinants (`DENSE_FCI_LIMIT`, was 4096), and above that runs
  Davidson's method on `H c` computed from the integrals without the
  matrix, for up to 32 roots (`MAX_FCI_ROOTS`; before, one root by
  Lanczos). Full CI of LiH in 6-31G: 69 s before, now under a second.

## [1.6.0] — 2026-10-27

Milestones 1.9 (matrix-product states), 1.10 (exact decoding and Stim
circuits), 1.11 (the state-vector core at memory bandwidth), 1.12 (noise
learning) and 1.13 (walks beyond static Hamiltonians) of the
[roadmap](docs/ROADMAP.md), released together. Results are checked
against the state-vector simulator, exact free-fermion evolution, the
QAOA ring optimum, enumeration, Stim, PyMatching, known device models,
SciPy and the classical random walk (named below). No validated result
of earlier releases changed; the public Rust API only grows
(`cargo-semver-checks` against 1.5.1).

### Added

- **Matrix-product states** (`mps::Mps`, Python `MatrixProductState`).
  - States of up to 65 536 qubits in mixed canonical form.
  - Any gate of the gate set, controls included, on up to 8 qubits,
    routed by swaps to adjacent sites and swapped back. Consecutive gates
    on the same one or two qubits are merged into one matrix first.
  - Truncation by a bond-dimension limit and a weight cutoff
    (`MpsOptions`).
  - Every truncation is measured exactly, and `fidelity_lower_bound`
    gives a rigorous lower bound on the fidelity with the exact state.
  - Pauli expectation values (strings, `PauliSum`, term lists), exact
    sampling, amplitudes, measurement and reset, entanglement entropy at
    every bond, conversion to a state vector up to 24 qubits.
  - Validation:
    - random circuits on 1–12 qubits match the state vector, amplitudes
      included, within the bound;
    - truncated circuits never fall below their bound;
    - a GHZ state on 2000 qubits keeps bond dimension 2, and 20 gates
      between the ends of a 2000-qubit chain keep the bound at 1;
    - a program that measures after a truncation reports no bound;
    - a Trotterized XX chain on 100 qubits matches the exact
      free-fermion evolution of the same circuit (⟨Z⟩, ⟨ZZ⟩ to 10⁻⁸);
    - QAOA on a ring of 100 nodes reaches (2p + 1)/(2p + 2) for
      p = 1, 2, 3.
- **Programs:** `mps::program::sample_program`,
  `Program.sample(method="mps", max_bond=…)` and
  `qso qasm FILE --mps --max-bond N`, which reports the fidelity bound.
- **Docs:** example `mps` (200 qubits) and the guide chapter
  "Matrix-product states".
- **Minimum-weight matching decoder** (`qec::MatchingDecoder`, Python
  `MatchingDecoder`).
  - Each shot's detection events are matched in pairs or to the boundary
    at minimum total weight: Dijkstra distances, cached per detector
    (up to 32 MiB) or found by bounded searches, and Edmonds' blossom
    algorithm on integer weights.
  - `Decoder` trait, `DecoderKind` and `logical_error_rate_with`; Python
    `logical_error_rate(..., decoder="matching")`; `qso qec --decoder
    matching`.
  - Validation:
    - the blossom algorithm matches dynamic programming on 4000 random
      graphs;
    - on random error models, every syndrome's correction weighs as
      little as the lightest error set found by enumeration;
    - every single fault, every pair at d = 5 and 20 000 random triples
      at d = 7 are corrected;
    - on the same error model and syndromes, predictions agree with
      PyMatching's (one of 800 000 differed in the cross-validation).
- **Stim circuits:** `DetectorCircuit::from_stim` (Python
  `DetectorCircuit.from_stim`, `qso qec --circuit FILE`) reads Clifford
  gates, measurements and resets in all three bases, X/Y/Z and
  depolarizing noise, detectors, observables and REPEAT blocks; other
  instructions are an error naming them. Circuits generated by Stim 1.16
  reproduce Stim's error models (unit tests and the cross-validation),
  and exported circuits read back to the same text.
- **X-basis memory:** `surface_code_memory_with(…, MemoryBasis::X)`
  (Python `surface_code_memory(..., basis="x")`, `qso qec --basis x`)
  protects |+⟩_L; every single fault and every pair at d = 5 are
  corrected.
- **Cross-validation:** matching predictions against PyMatching shot by
  shot; Stim's generated circuits imported and compared with the
  originals; the X-basis surface code.
- **Fuzzing:** a target for the Stim importer; the detector-circuit
  target also checks the Stim round trip and the matching decoder.
- **Vector kernels** (`quantum::set_simd`, `quantum::simd_level`, and
  the same in Python). The state-vector kernels work on contiguous runs
  of amplitudes, with primitives in AVX on x86-64 (detected at run time)
  and NEON on aarch64. They round exactly as the portable code, so the
  results are the same bit for bit. A test compares the two on every
  kernel natively, and under Miri for aarch64.
- **Blocked adjoint gradients:** on registers beyond a cache block, runs
  of gates inside a block are undone and differentiated block by block.
  The gradient terms are summed in block order, independent of the thread
  count.
- **Noise learning** (`noise::learning`, Python `learn_layer`,
  `SparsePauliLindblad`, `pec_layered`, `pea_layered`).
  - `learn_layer` fits the sparse Pauli–Lindblad model of a twirled layer
    of `cx`, `cy`, `cz` and `swap` gates on disjoint qubits (van den Berg
    et al. 2023): one rate per one-qubit Pauli and per two-qubit Pauli on
    neighbouring qubits, from decays over even depths in product bases
    (an orthogonal array over a colouring of the coupling graph: 9 bases
    up to 4 colours, 27 up to 13), with readout twirling and non-negative
    least squares.
  - Standard errors from a bootstrap over the twirled circuits (robust
    spread; those of the unconstrained fit, where λ ≥ 0 is active).
  - `pec_layered` cancels the learned noise in circuits of blocks
    separated by the layer; `pea_layered` amplifies it by inserting Paulis
    and extrapolates to zero noise.
  - Validation: on a four-qubit device with depolarising noise, a coherent
    phase error and readout errors, all 39 learned rates lie within three
    standard errors of the exact ones, and PEC with the learned model is
    unbiased (0.534 ± 0.005 for 0.533, raw 0.492).
- **Driven walks** (`walks::driven::DrivenWalk`, Python `DrivenWalk`).
  - Walks under H(t) = Σ_j c_j(t)·H_j with Laplacian, adjacency,
    projector, potential and uniform-projector terms.
  - Fourth-order commutator-free Magnus integrator with two Krylov
    exponentials per step, and step-size control by step doubling
    (`max_step` for pulsed drives).
  - Validation: static walks equal `ContinuousWalk`; halving the step
    divides the error by 16.1–18.5; the local adiabatic search of Roland
    and Cerf reaches the marked node of K₆₄, K₂₅₆ and K₁₀₂₄ with
    probability ≥ 1 − ε², matching an independent SciPy integration to
    10⁻⁶.
- **Decoherent coined walks** (`walks::decoherent::DecoherentCoinedWalk`,
  Python `DecoherentCoinedWalk`): coin and position dephasing (partial
  measurements of the coin register and of the node) on density matrices
  of up to 2048 arcs. With the coin measured every step, the
  Fourier-coin walk is the simple random walk (checked exactly, also on
  Metatron's Cube) and the Grover walk a Markov chain on arcs.
- **Benchmark report:** every gate on 24 qubits against the fastest
  in-place pass over the state, the kernels' instruction set, and the
  adjoint gradient on 20 qubits.

### Changed

- Diagonal gates read the two halves of the register in step (two
  memory streams), and diagonal gates inside cache blocks are
  vectorised. A run of a single block-local gate goes straight to the
  whole-register kernel.

### Notes

- Splits use the Hermitian eigendecomposition of a Gram matrix and a
  Householder QR, not nalgebra's complex SVD: nalgebra 0.35's complex SVD
  reconstructs rank-deficient matrices with errors of up to 20%.
- Fuzzing: the OpenQASM 2 target also runs every imported circuit of up
  to 8 qubits on an MPS and compares it with the state vector.

## [1.5.1] — 2026-10-26

### Fixed

- **Release workflow:** the npm package was not published. `npm publish`
  read the tarball path `npm/metatron-qso-X.Y.Z.tgz` as a GitHub
  repository; the path is now `./npm/…`. The library is unchanged from
  1.5.0, which is on crates.io and PyPI but not on npm.

## [1.5.0] — 2026-10-26

Milestones 1.7 (stabilizer simulation and graph states) and 1.8
(error-correction experiments) of the [roadmap](docs/ROADMAP.md). Every
result is checked against an independent reference — the state-vector
simulator, the tableau simulator, a closed form, exact enumeration or
Stim (named below). No validated result of earlier releases changed.

### Added — error-correction experiments (milestone 1.8)

- **Detector circuits** (`qec::DetectorCircuit`): Clifford gates, X/Y/Z
  errors, one- and two-qubit depolarizing noise, noisy measurements,
  resets, detectors and up to 64 logical observables; `append_circuit`
  for any Clifford `Circuit`; `to_stim` writes Stim's text format.
- **Pauli-frame sampling** (`qec::sample_detectors`): 64 shots per word;
  rejects detectors and observables that are not deterministic without
  noise.
  - Detection events and observable flips match shot-by-shot simulation
    on the tableau, with explicitly sampled errors, on random noisy
    Clifford circuits and on memory experiments.
- **Detector error models** (`qec::detector_error_model`): the exact
  decomposition of depolarizing channels into independent Pauli errors,
  propagation of every error, merging of equal effects, and an optional
  decomposition into graphlike parts.
  - On four repetition and surface-code memory experiments, the models
    equal Stim 1.16's term by term (probabilities to 10⁻¹⁵ relative), and
    detection-event rates agree with Stim's sampler; this is part of the
    scheduled cross-validation.
- **Union-find decoder** (`qec::UnionFindDecoder`, `qec::MatchingGraph`):
  weighted growth and peeling (Delfosse and Nickerson 2021).
  - On the repetition code, it fails on exactly the error patterns with
    more than d/2 flips (all 2ᵈ patterns, d = 3 … 9).
  - Every single fault of the repetition code (d = 5) and of the surface
    code (d = 3, 5) is corrected, and every pair of faults at d = 5.
  - Growth is event-driven with exact weights.
  - Logical error rates are at most 6% above PyMatching's
    (minimum-weight matching) on the cross-validation circuits.
  - The weight-3 syndrome at d = 7 that a review found misdecoded by
    rounded weights, and 20 000 random weight-3 sets, are corrected.
- **Memory experiments** (`qec::repetition_code_memory`,
  `qec::surface_code_memory`) with circuit-level noise (`CircuitNoise`),
  and `qec::logical_error_rate`.
  - Rotated surface code under circuit-level noise p = 0.001: logical
    error rates of 8.4·10⁻⁴, 1.4·10⁻⁴ and 2.1·10⁻⁵ at d = 3, 5, 7 (10⁶
    shots). At p = 0.01, they rise with d. Both directions are validation
    tests.
- **Limits:** sampling, error models (signature entries, propagation
  work) and the memory-experiment generators fail with an error before
  they exhaust memory or run for hours.
- **CLI:** `qso qec`.
- **Python:** `DetectorCircuit`, `CircuitNoise`, `DetectorErrorModel`,
  `UnionFindDecoder` (with `decode_batch`), `LogicalErrorRate`,
  `repetition_code_memory`, `surface_code_memory`, `logical_error_rate`.
- **Tests:** fuzz target `qec`. It builds circuits from random bytes,
  samples them, builds their models, and checks that the decoder
  explains every syndrome of matching-graph edges.
- **Docs:** example `surface_code`, guide chapter "Error-correction
  experiments", and a Stim and PyMatching section in the cross-framework
  comparison.

### Added — stabilizer simulation (milestone 1.7)

- **Stabilizer simulation** (`stabilizer::StabilizerState`).
  - Aaronson–Gottesman tableaux, bit-packed and stored by qubit, for up to
    16 384 qubits.
  - Clifford gates: H, S, S†, X, Y, Z, √X, √X†, SWAP, CX, CY, CZ, and
    RX, RY, RZ, P, RXX, RYY, RZZ and U at multiples of π/2.
  - Pauli and computational-basis measurements, reset, expectation
    values of Pauli strings and sums, sampling every qubit from the
    affine outcome space, conversion to a state vector up to 20 qubits.
  - 200 random Clifford circuits on 1–8 qubits match the state-vector
    simulator: states to fidelity 1 − 10⁻¹⁰, Pauli expectation values,
    collapse after measurement, outcome supports. A GHZ state on 10 000
    qubits has its known correlations.
- **Graph states.** `graph_state` for any unweighted graph,
  `local_complement` and `local_complementation`, and
  `entanglement_entropy` from the GF(2) rank of the generators.
  - The entropy equals the state-vector entropy and the cut-rank on six
    graphs, and the cut-ranks of Q₁₀ (512) and C₁₀₀₀ (2).
  - Local complementation equals the local Clifford of Hein, Eisert and
    Briegel (2004).
  - The graph state of Metatron's Cube (K₁₃) is a GHZ state up to local
    Cliffords.
- **Clifford programs** (`stabilizer::program`): OpenQASM programs made
  of Clifford gates, mid-circuit measurement, reset and conditions
  included, sampled on the tableau.
  - `qso qasm --stabilizer`.
  - The CLI, the WebAssembly module and the playground choose the
    stabilizer simulator for Clifford programs too large for a state
    vector.
- **Python:** `StabilizerState`, `local_complementation`, `is_clifford`,
  `Program.is_clifford` and `Program.sample(method=…)`.
- **Tests:** the `openqasm` fuzz target also runs Clifford programs on
  the stabilizer simulator; every record must have non-zero exact
  probability.
- **Docs:** example `stabilizers` and the guide chapter "Stabilizer
  simulation and graph states".

### Changed

- `quantum::Condition::holds` is public.

## [1.4.0] — 2026-10-25

Milestones 1.4 (noise at scale and error mitigation), 1.5 (algorithms and
physics) and 1.6 (interoperability) of the [roadmap](docs/ROADMAP.md),
released together. Every new feature is tested against an
independent reference — exact diagonalisation, PySCF's full CI, density
matrices, exact sums, closed forms or the known results of the OpenQASM
project's example programs (named below). No validated result of earlier
releases changed.

### Added — interoperability (milestone 1.6)

- **OpenQASM programs** (`quantum::Program`, `quantum::openqasm`).
  - One parser for OpenQASM 2.0 and a subset of OpenQASM 3: `gate`
    definitions with parameters, `U`, `CX` and `gphase`; the modifiers
    `ctrl @`, `ctrl(n) @`, `negctrl @`, `inv @` and integer `pow(k) @`;
    `measure`, `reset`, `barrier`; conditions on a register, a bit or
    `int[n](c)` with `==`/`!=`, blocks and `else`; register broadcasting,
    slices and parameter expressions. Other OpenQASM 3 features are
    rejected with an error that names them.
  - Global phases follow each version's specification: OpenQASM 3's `U`
    is `e^{iθ/2}` times the standard U3 matrix, and `gphase` and the
    phases of `u2`/`u3` become relative phases under `ctrl @`.
  - `Program::probabilities` gives the exact distribution of the
    classical bits by following every measurement branch (up to 4096);
    `Program::sample` draws seeded shots, simulating shot by shot beyond
    that.
  - Export to OpenQASM 3 (`Program::to_qasm3`, `to_qasm3` for circuits,
    controlled gates as `ctrl @`) and to OpenQASM 2 where the program
    allows it.
  - The example programs of the OpenQASM project
    (`tests/data/openqasm`, Apache-2.0) give their known results: the
    adder computes 1 + 15 = 16, phase estimation of `e^{i·3π/8}` gives
    3/16 exactly, teleportation sends sin²(0.15), the W state has three
    outcomes of 1/3, the repetition code corrects its error. The six
    OpenQASM 3 examples in the subset match their OpenQASM 2 versions,
    the other fifteen are rejected, and all survive round trips through
    both exports.
- **WebAssembly and JavaScript.**
  - The core library builds for `wasm32` (without the `parallel`
    feature). Its unit, validation, OpenQASM and robustness tests run in
    CI as `wasm32-wasip1` under Node.js (`scripts/wasi-runner.mjs`).
  - `metatron_qso_wasm`: a module without imports and with one JSON entry
    point (`qasm`, `graph`, `walk`, `search`, `isomorphism`, `version`).
  - npm package `metatron-qso` (`metatron_qso_wasm/js`): a dependency-free
    wrapper with TypeScript types, tested with Node against the same
    OpenQASM examples and graph results as the native build. Built by
    `scripts/build_wasm.py`.
  - A browser playground (`docs/playground`, published with the
    documentation site at `/playground/`): OpenQASM programs with
    histograms and exports, animated quantum walks and spatial search on
    Metatron's Cube and any graph specification, run in a Web Worker.
- Python: `Program` (`from_qasm`, `from_circuit`, `probabilities`,
  `sample`, `to_circuit`, `to_qasm3`, `to_qasm2`) and `Circuit.to_qasm3`.
- CLI: `qso qasm` runs OpenQASM 2 and 3 programs exactly, mid-circuit
  measurements and conditions included, and converts them
  (`--export qasm2|qasm3`); `qso estimate` reads unitary OpenQASM 3
  programs.
- Fuzz target `openqasm`: the OpenQASM 2 and 3 exports of a random
  program reproduce its outcome distribution.
- Guide chapters "OpenQASM programs" and "WebAssembly and JavaScript".

### Added — algorithms and physics (milestone 1.5)

- **Hamiltonian simulation** (`algorithms::time_evolution`).
  - `Circuit::pauli_rotation` appends `exp(−iθP/2)` for any Pauli string.
  - Product-formula circuits of order 1, 2, 4, 6, … (Lie–Trotter, Strang,
    Suzuki) and `trotter_error_bound`: the commutator bounds of Childs et
    al. (2021) for orders 1 and 2, the smaller of a stage-wise and a
    Taylor-remainder bound for higher orders. In the tests the actual
    errors stay below the bounds (by factors 3.5–13 for orders 1 and 2;
    the order-4 bound is valid but loose) and fall as `r^{-order}`.
  - qDRIFT circuits with Campbell's bound; the exact average channel
    converges as `1/N` and stays below it.
  - `QubitOperator` for complex combinations of Pauli strings.
- **Electronic structure** (`chemistry`).
  - `FermionOperator` with parsing and normal ordering.
  - Jordan–Wigner, parity and Bravyi–Kitaev mappings as linear binary
    transforms, and the two-qubit reduction of the parity mapping
    (`QubitEncoding`).
  - `MolecularIntegrals`: FCIDUMP reading and writing, qubit
    Hamiltonians, Hartree–Fock states and energies, full CI in the
    determinant space (dense up to 4096 determinants, Lanczos above).
  - FCIDUMP files and PySCF reference energies for H₂, LiH and BeH₂ in
    STO-3G (`examples/molecules/`). All four encodings give identical
    determinant-space spectra equal to PySCF's full CI (10⁻⁸ Ha);
    Hartree–Fock energies agree to 10⁻⁹ Ha.
  - UCCSD and k-UpCCGSD ansätze and ADAPT-VQE. UCCSD and ADAPT-VQE reach
    LiH's full-CI energy within 1.1·10⁻⁵ Ha on 10 qubits.
- **Excited states and state geometry.**
  - VQD and weighted SSVQE (`algorithms::excited`), checked against exact
    diagonalisation (all levels of H₂, three levels of an Ising chain).
  - State derivatives, the quantum Fisher information (checked against
    finite differences of the fidelity), quantum natural gradient and
    McLachlan imaginary-time evolution, which follows the exact
    normalised evolution to 10⁻⁶ (`algorithms::geometry`).
- **Optimisation problems** (`algorithms::combinatorial`): QUBO and Ising
  problems with exact enumeration; QAOA for any diagonal cost with X, ring
  or complete XY (Hamming weight conserved) and warm-start mixers.
- **Amplitude estimation** (`algorithms::amplitude_estimation`):
  canonical (exact outcome distribution, error bound met with probability
  ≥ 8/π²), iterative (Grinko et al.; 1 miss in 200 at α = 0.05) and
  maximum-likelihood (Suzuki et al.; RMSE slope −0.99 in the oracle calls,
  against −0.49 for plain sampling).
- **CLI:** `qso molecule FILE` and `qso trotter`.
- Example `molecules`, tutorial 07, guide chapters on excited states,
  Hamiltonian simulation, electronic structure, and optimisation and
  amplitude estimation; a fuzz target for the new parsers.
- Python: all of the above, with type stubs.

### Added — noise at scale and error mitigation (milestone 1.4)


- **Device models** (`noise::DeviceModel`, JSON via `from_json` /
  `to_json`).
  - Per-qubit T₁/T₂ and readout errors. Unset fields follow
    `default_qubit`.
  - Ordered gate rules matched by name, qubits or size; the first match
    applies. A rule sets the duration, a depolarising error, a relative
    over-rotation and further channels.
  - Gates are scheduled as early as possible; qubits relax while they
    wait.
  - Results agree with hand-built channel sequences to 1e-14, and a
    `NoiseModel` compiles to the same channels.
  - New fuzz target `device`.
- **Channels:** `Channel::pauli` (labels such as `"ZZ"`), `rotation`
  (coherent errors), `unitary`, `pauli_probabilities`, `twirled` (Pauli
  twirl) and `compose`.
- **Quantum trajectories** (`noise::TrajectorySimulator`,
  `backend::DeviceBackend`).
  - Monte-Carlo wave functions for any register a state vector holds,
    seeded per trajectory and parallel over trajectories.
  - Waiting-time method: unitary mixtures are sampled up front; stretches
    without a jump run as one fused, cache-blocked circuit; jumps are
    found by galloping search and bisection.
  - One trajectory, compilation included, costs 3.0–3.5 noiseless runs at
    16–20 qubits; 200 trajectories take 1.2–1.3 times as long as 200 noiseless
    runs at 16 qubits.
  - Samples pass χ² tests against density matrices on 3, 6 and 10
    qubits, also with several jumps per trajectory.
- **Estimates with errors** (`estimation`).
  - `Estimate` (standard error, confidence interval).
  - `estimate()` with uniform, weighted and adaptive (Neyman) shot
    allocation and optional readout mitigation by exact per-shot weights.
    95 % intervals cover the exact value in 91–99 % of 300 repetitions.
  - `ClassicalShadow`: random Pauli snapshots, mean and median of means.
    The single-snapshot variance equals 3ᵏ − ⟨P⟩².
- **Error mitigation** (`mitigation`).
  - `ReadoutCalibration`: tensored or full calibration, quasi-probabilities
    and the nearest probability distribution. Checked against brute force.
  - Zero-noise extrapolation (`zne`): global and random folding; linear,
    Richardson, polynomial and exponential fits with propagated errors.
    Richardson on scales 1, 3, 5 reduces the error of a depolarised test
    circuit about 380×.
  - Probabilistic error cancellation of Pauli noise (`pec`) with its
    overhead. The exact sum over all corrections equals the noiseless
    value to 1e-12.
  - Pauli twirling of cx, cy, cz and swap (`twirl`). The average over all
    16 frames equals the twirled channel.
  - On a five-qubit toy device, readout mitigation plus ZNE recovers
    ⟨Σ Z_i Z_{i+1}⟩ = 4.00 ± 0.01 (raw 3.38), and PEC 4.00 ± 0.01.
- **CLI:** `qso qasm --device FILE [--trajectories N]`, and
  `qso estimate` with standard errors, readout mitigation, `--zne` and
  `--pec`.
- Example `noise_mitigation` with `examples/devices/toy_device.json`;
  tutorial 06 and the guide chapter "Estimation and error mitigation".
- Python: all of the above, with type stubs.

### Changed

- `from_qasm2` reads through the new parser: it now accepts `gate`
  definitions, and still drops measurements; `reset` and
  conditions remain errors there.
- `qso qasm` without `--noise`/`--device` samples the program's classical
  bits (all qubits for a program without bits), where it used to drop
  measurements and measure every qubit.
- `MAX_QUBITS` is 26 on 32-bit targets such as WebAssembly (34 on 64-bit
  targets).
- The release workflow also publishes the npm package.
- `DensityMatrix::apply_device`, `sample_with_readout`, `scale` and
  `add_scaled`. Consecutive channels on the same qubits are applied as
  one superoperator.

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
- The cross-framework comparison warms every framework up for at least
  0.25 s: rayon workers sleep when idle and need tens of milliseconds to
  reach full speed on virtual machines. Results for 1.3.0 (QSO fastest
  from 20 qubits on, Qulacs 1.3× faster at 16 qubits) are in
  [BENCHMARKS.md](docs/BENCHMARKS.md) with the raw data.

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
