# Noise

## Channels

A `Channel` is a completely positive, trace-preserving map given by Kraus
operators. QSO provides the standard channels:

| Channel | Action |
|---|---|
| `depolarizing(p)` | (1−p)ρ + p/3 (XρX + YρY + ZρZ) |
| `depolarizing_2q(p)` | (1−p)ρ + p/15 Σ_{P≠II} PρP |
| `amplitude_damping(γ)` | decay \|1⟩ → \|0⟩ with probability γ |
| `phase_damping(λ)` | loss of coherence without energy loss |
| `bit_flip(p)`, `phase_flip(p)` | X or Z with probability p |
| `thermal_relaxation(T₁, T₂, t)` | T₁/T₂ decay over a time t |
| `from_kraus([...])` | any channel; completeness is checked |

Qiskit parametrises depolarising noise differently. Its λ is λ = 4p/3 for
one qubit and λ = 16p/15 for two qubits.

## Noise models and density matrices

A `NoiseModel` applies channels after every gate:

- `single_qubit` channels follow each one-qubit gate;
- `multi_qubit` channels follow each multi-qubit gate, a one-qubit channel
  on each of its qubits, a two-qubit channel on the pair.

Optional readout errors flip measured bits. `DensityMatrix.simulate` runs a
circuit with such a model:

```python
import metatron_qso as q

bell = q.Circuit(2).h(0).cx(0, 1)
model = q.NoiseModel(
    single_qubit=[q.Channel.depolarizing(0.01), q.Channel.amplitude_damping(0.02)],
    multi_qubit=[q.Channel.depolarizing_2q(0.03)],
    readout=(0.02, 0.05),
)
rho = q.DensityMatrix.simulate(bell, noise=model)
print(rho.purity(), rho.fidelity_with_pure(q.simulate(bell)))
print(rho.sample(1000, seed=1, readout=model.readout))
```

Density matrices use the same kernels as pure states, applied to the
vectorised ρ, and support up to 14 qubits. The weekly cross-validation
compares them with Qiskit Aer's density-matrix method on random noisy
circuits. They agree to 3.5·10⁻¹⁵.

## Shot-based estimates

`estimate_expectation` measures an observable the way hardware would. It
groups qubit-wise commuting terms, measures each group with the given number
of shots, and can add noise:

```python
import metatron_qso as q

bell = q.Circuit(2).h(0).cx(0, 1)
h = q.PauliSum("Z0 Z1 + X0 X1")
exact = h.expectation(q.simulate(bell))                         # 2.0
noisy = q.estimate_expectation(bell, h, shots=20_000, seed=3,
                               noise=q.NoiseModel.depolarizing(0.01, 0.02))
print(exact, noisy)
```

## T₁ and T₂

`thermal_relaxation` composes amplitude damping and dephasing. The test
suite checks the decay against the closed forms ⟨1|ρ(t)|1⟩ = e^{−t/T₁} and
|⟨0|ρ(t)|1⟩| = ½ e^{−t/T₂}:

```python
import math
import metatron_qso as q

t1, t2, t = 50.0, 30.0, 20.0
rho = q.DensityMatrix.simulate(q.Circuit(1).h(0))
rho = rho.apply_channel(q.Channel.thermal_relaxation(t1, t2, t), [0])
m = rho.to_matrix()
assert abs(abs(m[0, 1]) - 0.5 * math.exp(-t / t2)) < 1e-12
```
