# How QSO is validated

QSO makes no claim it cannot test. This page lists the kinds of evidence
behind its results and where to find them.

## References for every algorithm

Every algorithm is checked against an independent reference:

| Kind of reference | Examples |
|---|---|
| Closed forms | Grover's sin²((2k+1)θ); QAOA on rings, (2p+1)/(2p+2); the one-layer QAOA formula of Wang et al.; walk limits such as 145/169; T₁/T₂ decay; n! bunching; the running time of local adiabatic search; the classical random walk as the limit of fully dephased coined walks |
| Published values | 0.6925 and 0.7559 for QAOA on large-girth cubic graphs; the full-CI energy of H₂/STO-3G; automorphism group orders |
| Dense matrices | every gate, the QFT against the DFT matrix, Lanczos and Krylov against full diagonalisation |
| A second method | adjoint, parameter-shift and finite-difference gradients; statevector against density matrix; quantum trajectories against density matrices (χ² tests); the quantum Fisher information against finite differences of the fidelity; four fermion encodings against each other |
| Exact sums | PEC summed over all corrections, Pauli twirls over all 16 frames, the variance 3ᵏ of classical shadows |
| Statistics | coverage of confidence intervals over repeated runs; unbiased adaptive shot allocation over 2000 seeds; the 1/N error scaling of amplitude estimation over 200 seeds; learned noise rates of a known device within their bootstrap standard errors |
| Rigorous bounds | product-formula errors below their bounds; qDRIFT channels below Campbell's bound; canonical amplitude estimation within its error bound with probability ≥ 8/π² |
| Other codes | Qiskit Aer, Qulacs, Cirq, PennyLane Lightning and The Walrus (permanents, hafnians, Gaussian boson sampling); PySCF's integrals and its Hartree–Fock, full-CI and CASCI energies for H₂, LiH, BeH₂, H₂O, NH₃, OH and CH₂ |

These checks live in the test suites, which run on every change on Linux
(x86_64 and arm64), macOS and Windows:

- Rust: unit, validation, property and robustness tests;
- Python: tests of the bindings;
- WebAssembly: the unit, validation, OpenQASM and robustness tests built
  for `wasm32-wasip1` and run under Node.js, and the npm package's tests
  against the same OpenQASM examples and graph results.

The OpenQASM reader runs the example programs of the OpenQASM project and
checks their known results ([OpenQASM programs](programs.md)). The
stabilizer simulator is checked against the state-vector simulator on
random Clifford circuits ([Stabilizer simulation](stabilizers.md)). The
Pauli-frame sampler of the error-correction experiments is checked against
the stabilizer simulator, and the decoders against exact majority-vote
failure rates and against every single fault (and, at distance 5, every
pair of faults) of the surface code. The matching decoder's corrections
weigh as little as the lightest error sets found by enumeration, and its
blossom algorithm agrees with dynamic programming on random graphs
([Error correction](qec.md)).
Matrix-product states are checked against the state vector on random
circuits (within their fidelity bound), and on 100 qubits against the
exact free-fermion evolution of a Trotterized XX chain and the QAOA ring
optimum ([Matrix-product states](mps.md)).

## Cross-validation with other frameworks

[`benchmarks/cross_framework`](https://github.com/LashSesh/qso/tree/main/benchmarks/cross_framework)
compares QSO with established simulators. It runs every week, and a
deviation above rounding level fails the run.

| Comparison | Largest difference |
|---|---|
| Final states of random circuits (Qiskit Aer, Qulacs, Cirq, PennyLane Lightning) | 2.4·10⁻¹⁵ |
| Energies and gradients of a 100-parameter ansatz (PennyLane adjoint, Qulacs backprop) | 1.8·10⁻¹⁵ |
| Density matrices of noisy circuits (Qiskit Aer) | 3.5·10⁻¹⁵ |
| Permanents (The Walrus) | 2.8·10⁻¹⁴ relative |
| Detector error models of repetition and surface-code memory experiments (Stim), and of Stim's generated circuits imported | 1.0·10⁻¹⁵ relative; detection-event rates agree to sampling precision |
| Matching decoder against PyMatching, same error model and syndromes | 1 differing prediction in 800 000 (8 circuits × 10⁵ shots) |

## Robustness

| Check | What it covers |
|---|---|
| Fuzzing | Each parser and deserialiser runs under cargo-fuzz for one minute on every pull request and ten minutes every week. The parsers are OpenQASM 2 and 3, Pauli sums, graph specifications and Stim circuits; the JSON forms are those of circuits, states, graphs, Pauli sums and device models; fermion operators and FCIDUMP files are fuzzed as well, and so are detector circuits built from random bytes. Beyond "no crash", the targets check that exported QASM re-imports to an equivalent circuit that prepares the same state on a matrix-product state, that the OpenQASM 2 and 3 exports of a program reproduce its outcome distribution, that accepted data are well-formed, that the union-find and matching decoders explain every syndrome that some set of matching-graph edges produces, and that Stim circuits (imported, or exported and read back) reproduce their text. Every input that ever failed is replayed on each run. |
| Differential testing | `differential::compare` (Rust) runs one circuit on every simulator that takes it: the single-precision state vector, the noiseless density matrix, a noiseless trajectory, a matrix-product state of unlimited bond dimension, the stabilizer tableau (its generators and every Pauli expectation value, for Clifford circuits) and the OpenQASM 2 and 3 round trips. States are compared amplitude by amplitude after aligning the global phase, so an angle off by 10⁻⁸ shows as 10⁻⁸; norms and traces separately. The tolerance is 10⁻¹⁰ (single precision 10⁻⁷ per gate; the MPS within its own truncation bound, since even an unlimited bond drops singular values below about 10⁻⁸ of the largest). Property tests feed it random circuits of every gate kind, under up to three controls, dense unitaries on up to three qubits, symbolic and special angles, on 1 to 10 qubits: 256 per test on every change and 20 000 every week. A fuzz target decodes circuits from random bytes, for one minute on every pull request and ten minutes every week. The first runs and an independent review found three faults, now regression tests: the MPS turned to NaN when asked to keep everything, the tableau accepted rotations by huge angles as Clifford gates, and a test compared NaN bit patterns. |
| Miri | The `unsafe` statevector kernels run under Miri on multi-threaded paths, which checks them for undefined behaviour and data races; so do the buffer exports of the WebAssembly module. |
| Hostile input | Tests feed oversized, non-finite and inconsistent input to every public entry point and expect an error. |

## Coverage

CI measures line and branch coverage with `cargo-llvm-cov`
(`scripts/coverage.sh`), merged from three sources: the crate's tests
(unit, validation, property, differential and robustness tests), the
command-line tool, which `tests/cli.rs` runs as a program, and the Python
tests, run against an instrumented build of the extension module. The
job summary lists the totals, the coverage of the library, the command
line and the Python bindings separately, and the files with the most
uncovered lines. The documentation site publishes the report of each
build under [`coverage/`](coverage/html/index.html). At milestone 1.21 the
crate's tests alone covered 90.5 % of lines, 92.8 % of functions and
79.4 % of branches; until 1.25 the command line's source counted as
uncovered.

## API stability

QSO 1.x adds and never breaks:

- `cargo-semver-checks` compares the public Rust API with the last release;
- `stubtest` checks that the Python type stubs match the compiled module.

## Performance

The [benchmark report](https://github.com/LashSesh/qso/blob/main/docs/BENCHMARKS.md)
states the machine it ran on. A weekly run compares the timings with the
median of the previous four runs. A slowdown of more than 25 % is flagged.

## The technical note

[*Validated Quantum Algorithm Benchmarks of the Metatron QSO Simulation
Framework*](https://github.com/LashSesh/qso/tree/main/docs/technical-note)
(DOI [10.5281/zenodo.23623226](https://doi.org/10.5281/zenodo.23623226))
reports all of the above with the data and the programs that produce it.
