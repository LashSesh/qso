# metatron-qso (Python)

Python bindings for **QSO**, a validated quantum simulator written in
pure Rust (`metatron-qso-rs`): exact state-vector and density-matrix
simulation, VQE and QAOA with exact gradients, Grover search, QFT and phase
estimation, and continuous-time quantum walks on arbitrary graphs —
including Metatron's Cube. Every algorithm is tested against closed-form or
literature results.

## Install

```bash
pip install metatron-qso
```

Wheels target the stable ABI (`abi3`, one wheel for CPython ≥ 3.9) and are
provided for Linux (x86_64, aarch64; glibc and musl), macOS (Intel and Apple
silicon) and Windows (x86_64, arm64). The only runtime dependency is NumPy. On other platforms
pip builds from the source distribution, which needs a Rust toolchain
≥ 1.89.

From a clone of the [repository](https://github.com/LashSesh/qso):

```bash
pip install maturin
maturin develop --release -m metatron_qso_py/Cargo.toml   # into the active venv
```

## Conventions

* Qubit 0 is the least significant bit of a basis index (as in Qiskit); bit
  strings print qubit n−1 first (`"001"` means qubit 0 is 1).
* Rotations are `R_P(θ) = exp(−iθP/2)`; time evolution is `exp(−iHt)`.
* Vectors and matrices are NumPy `float64` / `complex128` arrays; array
  arguments accept lists, tuples or arrays.
* Invalid input raises `ValueError` with a descriptive message.
* Long-running calls (VQE, QAOA, walks, simulation) release the GIL.

## Examples

Circuits, sampling and observables:

```python
import metatron_qso as q

bell = q.Circuit(2).h(0).cx(0, 1)
state = q.simulate(bell)
print(state.amplitudes)                       # complex128 array of length 4
print(state.sample(1000, seed=1))             # {'00': ..., '11': ...}
print(state.expectation(q.PauliSum("Z0 Z1"))) # 1.0
print(state.entanglement_entropy([0]))        # 1.0 (bits)
```

Symbolic parameters, exact gradients and OpenQASM:

```python
import numpy as np
import metatron_qso as q

c = q.Circuit(2)
a, b = c.new_parameters(2)
c.ry(0, a).cx(0, 1).rz(1, 2.0 * b + 0.5)
h = q.PauliSum("0.5*Z0 Z1 - X0")
value, grad = q.expectation_and_gradient(c, h, [0.3, 0.1])  # adjoint method
_, fd = q.expectation_and_gradient(c, h, [0.3, 0.1], method="finite_difference")
assert np.allclose(grad, fd, atol=1e-6)

text = c.to_qasm([0.3, 0.1], measure=True)      # OpenQASM 2.0
again = q.Circuit.from_qasm(text)
print(c.to_qasm3([0.3, 0.1]))                     # OpenQASM 3
```

Programs with mid-circuit measurement and feed-forward, from OpenQASM 2
or 3, with exact outcome probabilities:

```python
import metatron_qso as q

p = q.Program.from_qasm('''
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
c[0] = measure q[0];
if (c[0] == 1) x q[1];
c[1] = measure q[1];
''')
print(p.probabilities())        # ≈ {'00': 0.5, '11': 0.5}, bit 0 on the right
print(p.sample(1000, seed=1))   # {'00': 504, '11': 496}
print(p.to_qasm3())
```

Ground state of H₂ (STO-3G) with VQE:

```python
import metatron_qso as q

h2 = q.h2_sto3g()
result = q.run_vqe(h2["hamiltonian"], q.real_amplitudes(2, 1), restarts=4)
print(result["energy"] + h2["nuclear_repulsion"])  # -1.137306 Ha (full CI)
```

QAOA for MaxCut (depth 1 on a ring gives exactly 3/4):

```python
import metatron_qso as q

qaoa = q.MaxCutQaoa(q.Graph.cycle(8), p=1)
result = qaoa.optimize(restarts=8, seed=0)
print(result["approximation_ratio"])     # 0.75
print(result["most_likely_bitstring"])   # an optimal cut, e.g. '10101010'
```

Quantum walks on Metatron's Cube (the complete graph K13 on the 13
Fruit-of-Life centres):

```python
import math
import metatron_qso as q

cube = q.Graph.metatron_cube()
print(cube.automorphism_group_order())         # 6227020800 = 13!

walk = q.ContinuousWalk(cube)                   # H = γL, γ = 1
print(walk.limiting_distribution(0)[:2] * 169)  # [145. 2.]

search = q.SpatialSearch(cube, marked=[4], gamma=1 / 13)
print(search.success_probability(math.pi * math.sqrt(13) / 2))  # 1.0
```

Symmetry, state transfer and discrete-time walks:

```python
import math
import metatron_qso as q

walk = q.QuotientWalk(q.Quotient.hypercube(30))       # 2^30 nodes, 31 cells
print(walk.cell_probabilities(0, math.pi / 2)[-1])     # 1.0: the antipode

print(q.StateTransfer(q.Graph.path(3), 0, 2, "adjacency").perfect_transfer_time)  # pi/sqrt(2)
print(q.Graph.shrikhande().is_isomorphic(q.Graph.rook(4, 4)))                     # False

skw = q.CoinedWalk(q.Graph.hypercube(10), "grover", marked=[0]).search(60)
print(skw["steps"], skw["success_probability"])        # 38 0.435
print(q.SzegedyWalk(q.Graph.cycle(64), 0).hitting_time(1000))                    # 16
```

Noise:

```python
import metatron_qso as q

bell = q.Circuit(2).h(0).cx(0, 1)
rho = q.DensityMatrix.simulate(bell, noise=q.NoiseModel.depolarizing(0.01, 0.02))
print(rho.purity(), rho.fidelity_with_pure(q.simulate(bell)))
energy = q.estimate_expectation(bell, q.PauliSum("Z0 Z1"), shots=10_000,
                                noise=q.NoiseModel.depolarizing(0.01, 0.02))
```

Device models, trajectories and error mitigation (readout mitigation plus
zero-noise extrapolation):

```python
import metatron_qso as q

device = (q.DeviceModel("toy")
          .with_default_qubit(t1=100.0, t2=80.0, readout=(0.02, 0.04))
          .with_rule(gate="cx", duration=0.3, depolarizing=0.01)
          .with_rule(duration=0.035, depolarizing=0.001))
ghz = q.Circuit(3).h(0).cx(0, 1).cx(1, 2)
h = q.PauliSum("Z0 Z1 + Z1 Z2")                                  # noiseless value 2
cal = q.ReadoutCalibration.tensored(3, 100_000, seed=1, device=device)

def run(circuit):
    return q.estimate(circuit, h, 100_000, device=device, readout_calibration=cal, seed=2)[0]

raw, _ = q.estimate(ghz, h, 100_000, device=device, seed=2)
mitigated = q.zne(ghz, [1.0, 3.0, 5.0], run)["estimate"]
print(f"{raw.value:.3f} {mitigated.value:.3f} ± {mitigated.std_error:.3f}")  # 1.725 2.001 ± 0.007
print(q.TrajectorySimulator(device, 2000, seed=3).expectation(ghz, h).value)  # 1.969
```

Molecules, time evolution and amplitude estimation (run from the
repository root, which holds the FCIDUMP files):

```python
import metatron_qso as q

lih = q.MolecularIntegrals.from_file("metatron-qso-rs/examples/molecules/lih_sto3g.fcidump")
adapt = q.adapt_vqe(lih, "parity", two_qubit_reduction=True)          # 10 qubits
print(f"{adapt['energy'] - lih.fci_energies()[0]:.1e}", len(adapt["operators"]))  # 1.1e-05 30

h = q.heisenberg_xxz(4, 1.0, 0.5)
c = q.trotter_circuit(h, 4, 1.0, steps=20, order=2)
print(len(c), f"{q.trotter_error_bound(h, 1.0, steps=20, order=2):.4f}")  # 321 0.0229

r = q.ml_amplitude_estimation(q.AmplitudeProblem.bernoulli(0.3), levels=8, shots=100, seed=1)
print(f"{r['estimate']:.4f}", r["oracle_calls"])                        # 0.2998 26200
```

More complete scripts are in
[`examples/`](https://github.com/LashSesh/qso/tree/main/metatron_qso_py/examples);
the [user guide](https://lashsesh.github.io/qso/) has tutorials and the full
API reference.

## API overview

| Area | Names |
|---|---|
| Graphs | `Graph` (`from_spec`, `metatron_cube`, `fruit_of_life`, `metatron_distance_graph`, `complete`, `cycle`, `path`, `star`, `complete_bipartite`, `hypercube`, `petersen`, `grid`, `platonic`, `random_regular`, `erdos_renyi`, `heawood`, `johnson`, `kneser`, `hamming`, `rook`, `paley`, `circulant`, `cayley_z2`, `shrikhande`; `cartesian_product`, `strongly_regular_parameters`, `girth`, `triangle_count`, automorphisms, spectra; `canonical_labelling`, `canonical_form`, `is_isomorphic`, `isomorphism`, `relabeled`; `from_edge_list`/`to_edge_list`, `from_graphml`/`to_graphml`, `from_networkx`), `metatron_line_lengths` |
| Circuits | `Circuit`, `Parameter`, `qft`, `inverse_qft`, `grover_circuit` |
| States | `StateVector`, `simulate`, `PauliSum`, `bitstring` |
| Models | `h2_sto3g`, `transverse_field_ising`, `heisenberg_xxz`, `maxcut_cost` |
| Variational | `real_amplitudes`, `efficient_su2`, `hardware_efficient`, `qaoa_maxcut_circuit`, `expectation_and_gradient`, `expectation_batch`, `expectation_and_gradient_batch`, `run_vqe`, `MaxCutQaoa`, `optimize_qaoa_layerwise`, `VariationalClassifier`, `GraphClassifier` |
| Textbook | `grover_search`, `estimate_phase` |
| Walks | `ContinuousWalk`, `SpatialSearch` (`optimize` tunes rate and time jointly), `DephasingWalk`, `DrivenWalk`, `classical_walk_probabilities`, `classical_hitting_times`; `Quotient`, `QuotientWalk`, `QuotientSearch`, `StateTransfer`; `CoinedWalk`, `DecoherentCoinedWalk`, `LineWalk`, `SzegedyWalk`; `circulant_walk_circuit`, `cayley_walk_circuit`, `trotter_walk_circuit`, `diagonal_unitary_circuit` |
| Noise | `Channel`, `NoiseModel`, `DeviceModel`, `DensityMatrix`, `TrajectorySimulator`, `estimate_expectation` |
| Estimation | `Estimate`, `estimate` (uniform, weighted, adaptive shot allocation), `ClassicalShadow` |
| Mitigation | `ReadoutCalibration`, `nearest_probability_distribution`, `fold`, `extrapolate`, `zne`, `pec_expectation`, `twirl`, `twirled_average`; noise learning: `SparsePauliLindblad`, `learn_layer`, `pec_layered`, `pea_layered` |
| Hamiltonian simulation | `Circuit.pauli_rotation`, `trotter_circuit`, `trotter_error_bound`, `qdrift_circuit`, `qdrift_error_bound` |
| Chemistry | `FermionOperator`, `QubitOperator`, `MolecularIntegrals`, `uccsd`, `k_upccgsd`, `adapt_vqe` |
| Excited states and geometry | `vqd`, `ssvqe`, `quantum_fisher_information`, `natural_gradient`, `quantum_natural_gradient`, `imaginary_time_evolution` |
| Optimisation and estimation | `Qubo`, `Qaoa` (X, XY and warm-start mixers), `AmplitudeProblem`, `canonical_amplitude_estimation`, `iterative_amplitude_estimation`, `ml_amplitude_estimation` |
| Boson sampling | `BosonSampler`, `ImperfectBosonSampler`, `permanent`, `haar_unitary`, `beam_splitter`; Gaussian boson sampling: `GaussianState`, `hafnian`, `hafnian_repeated`, `torontonian` |

`Graph.from_spec` accepts the graph specifications of the `qso` command-line
tool, e.g. `"metatron"`, `"metatron:1"`, `"hypercube:4"`, `"platonic:cube"`
(full list in the
[main README](https://github.com/LashSesh/qso#command-line-reference)).

Type information ships with the package (`_core.pyi`, `py.typed`).

Optimisers for `run_vqe` / `MaxCutQaoa.optimize`: `lbfgs` (default), `adam`,
`gradient_descent`, `nelder_mead`, `spsa`. Gradient methods: `adjoint`
(default), `parameter_shift`, `finite_difference`.

## Tests

```bash
pip install pytest
pytest metatron_qso_py/tests -q
```

## License

Apache-2.0
