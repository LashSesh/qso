# How QSO is validated

QSO makes no claim it cannot test. This page lists the kinds of evidence
behind its results and where to find them.

## References for every algorithm

Every algorithm is checked against an independent reference:

| Kind of reference | Examples |
|---|---|
| Closed forms | Grover's sin²((2k+1)θ); QAOA on rings, (2p+1)/(2p+2); the one-layer QAOA formula of Wang et al.; walk limits such as 145/169; T₁/T₂ decay; n! bunching |
| Published values | 0.6925 and 0.7559 for QAOA on large-girth cubic graphs; the full-CI energy of H₂/STO-3G; automorphism group orders |
| Dense matrices | every gate, the QFT against the DFT matrix, Lanczos and Krylov against full diagonalisation |
| A second method | adjoint, parameter-shift and finite-difference gradients; statevector against density matrix; quantum trajectories against density matrices (χ² tests); the quantum Fisher information against finite differences of the fidelity; four fermion encodings against each other |
| Exact sums | PEC summed over all corrections, Pauli twirls over all 16 frames, the variance 3ᵏ of classical shadows |
| Statistics | coverage of confidence intervals over repeated runs; unbiased adaptive shot allocation over 2000 seeds; the 1/N error scaling of amplitude estimation over 200 seeds |
| Rigorous bounds | product-formula errors below their bounds; qDRIFT channels below Campbell's bound; canonical amplitude estimation within its error bound with probability ≥ 8/π² |
| Other codes | Qiskit Aer, Qulacs, Cirq, PennyLane Lightning and The Walrus; PySCF's Hartree–Fock and full-CI energies for H₂, LiH and BeH₂ |

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
the stabilizer simulator, and the decoder against exact majority-vote
failure rates and against every single fault (and, at distance 5, every
pair of faults) of the surface code ([Error correction](qec.md)).

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
| Detector error models of repetition and surface-code memory experiments (Stim) | 1.0·10⁻¹⁵ relative; detection-event rates agree to sampling precision |

## Robustness

| Check | What it covers |
|---|---|
| Fuzzing | Each parser and deserialiser runs under cargo-fuzz for one minute on every pull request and ten minutes every week. The parsers are OpenQASM 2 and 3, Pauli sums and graph specifications; the JSON forms are those of circuits, states, graphs, Pauli sums and device models; fermion operators and FCIDUMP files are fuzzed as well, and so are detector circuits built from random bytes. Beyond "no crash", the targets check that exported QASM re-imports to an equivalent circuit, that the OpenQASM 2 and 3 exports of a program reproduce its outcome distribution, that accepted data are well-formed, and that the union-find decoder explains every syndrome that some set of matching-graph edges produces. Every input that ever failed is replayed on each run. |
| Miri | The `unsafe` statevector kernels run under Miri on multi-threaded paths, which checks them for undefined behaviour and data races; so do the buffer exports of the WebAssembly module. |
| Hostile input | Tests feed oversized, non-finite and inconsistent input to every public entry point and expect an error. |

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
