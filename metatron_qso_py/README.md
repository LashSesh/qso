# metatron-qso (Python)

Python bindings for **QSO 2.0**, a validated quantum simulator written in
pure Rust (`metatron-qso-rs`): exact state-vector and density-matrix
simulation, VQE and QAOA with exact gradients, Grover search, QFT and phase
estimation, and continuous-time quantum walks on arbitrary graphs —
including Metatron's Cube. Every algorithm is tested against closed-form or
literature results.

## Install

From the repository (requires a Rust toolchain):

```bash
pip install maturin
maturin develop --release -m metatron_qso_py/Cargo.toml   # into the active venv
# or: maturin build --release -m metatron_qso_py/Cargo.toml && pip install target/wheels/*.whl
```

The wheel targets the stable ABI (`abi3`, CPython ≥ 3.9) and needs only
NumPy at runtime.

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

Symbolic parameters, exact gradients and OpenQASM 2.0:

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

text = c.to_qasm([0.3, 0.1], measure=True)
again = q.Circuit.from_qasm(text)
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

Noise:

```python
import metatron_qso as q

bell = q.Circuit(2).h(0).cx(0, 1)
rho = q.DensityMatrix.simulate(bell, noise=q.NoiseModel.depolarizing(0.01, 0.02))
print(rho.purity(), rho.fidelity_with_pure(q.simulate(bell)))
energy = q.estimate_expectation(bell, q.PauliSum("Z0 Z1"), shots=10_000,
                                noise=q.NoiseModel.depolarizing(0.01, 0.02))
```

More complete scripts are in [`examples/`](examples/).

## API overview

| Area | Names |
|---|---|
| Graphs | `Graph` (`from_spec`, `metatron_cube`, `fruit_of_life`, `metatron_distance_graph`, `complete`, `cycle`, `path`, `star`, `complete_bipartite`, `hypercube`, `petersen`, `grid`, `platonic`, `random_regular`, `erdos_renyi`), `metatron_line_lengths` |
| Circuits | `Circuit`, `Parameter`, `qft`, `inverse_qft`, `grover_circuit` |
| States | `StateVector`, `simulate`, `PauliSum`, `bitstring` |
| Models | `h2_sto3g`, `transverse_field_ising`, `heisenberg_xxz`, `maxcut_cost` |
| Variational | `real_amplitudes`, `efficient_su2`, `hardware_efficient`, `qaoa_maxcut_circuit`, `expectation_and_gradient`, `run_vqe`, `MaxCutQaoa`, `optimize_qaoa_layerwise`, `VariationalClassifier` |
| Textbook | `grover_search`, `estimate_phase` |
| Walks | `ContinuousWalk`, `SpatialSearch` (`optimize` tunes rate and time jointly), `DephasingWalk`, `classical_walk_probabilities`, `classical_hitting_times` |
| Noise | `Channel`, `NoiseModel`, `DensityMatrix`, `estimate_expectation` |

`Graph.from_spec` accepts the graph specifications of the `qso` command-line
tool, e.g. `"metatron"`, `"metatron:1"`, `"hypercube:4"`, `"platonic:cube"`
(full list in the [main README](../README.md#command-line-reference)).

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
