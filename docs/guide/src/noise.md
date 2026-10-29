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
| `pauli({"ZZ": p, "XI": q})` | Pauli channel: each listed Pauli string with its probability |
| `rotation("ZZ", θ)`, `unitary(U)` | coherent errors exp(−iθP/2) or any unitary |

Qiskit parametrises depolarising noise differently. Its λ is λ = 4p/3 for
one qubit and λ = 16p/15 for two qubits.

`pauli_probabilities()` returns the probabilities of a Pauli channel (and
`None` for other channels), `twirled()` the Pauli channel a channel becomes
under Pauli twirling, and `a.compose(b)` the channel "first `a`, then `b`":

```python
import metatron_qso as q

ad = q.Channel.amplitude_damping(0.3)
print(ad.pauli_probabilities())               # None: not a Pauli channel
print(ad.twirled().pauli_probabilities())     # [I, X, Y, Z] after twirling
zz = q.Channel.rotation("ZZ", 0.1)            # coherent ZZ crosstalk
print(zz.twirled().pauli_probabilities()[15]) # sin²(0.05)
```

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

## Device models

A `DeviceModel` describes a device the way calibration data does:

- every qubit has T₁, T₂ and readout errors;
- gates match an ordered list of rules (by name, qubits or size). The first
  matching rule gives the gate's duration, a depolarising error, a relative
  over-rotation of rotation angles and further channels.

Compiling a circuit schedules every gate as early as its qubits allow.
Qubits that wait for others relax with their T₁ and T₂. Descriptions are
JSON, so calibration data from any source can be written in this form.

```python
import metatron_qso as q

device = (
    q.DeviceModel("toy")
    .with_default_qubit(t1=100.0, t2=80.0, readout=(0.02, 0.04))
    .with_qubit(3, t1=60.0, t2=40.0, readout=(0.03, 0.08))
    .with_rule(gate="cx", qubits=[2, 3], duration=0.45, depolarizing=0.02)
    .with_rule(gate="cx", duration=0.3, depolarizing=0.008,
               channels=[q.Channel.rotation("ZZ", 0.02)])
    .with_rule(num_qubits=1, duration=0.035, depolarizing=0.0005, over_rotation=0.002)
)
same = q.DeviceModel.from_json(device.to_json())

ghz = q.Circuit(4).h(0).cx(0, 1).cx(1, 2).cx(2, 3)
rho = q.DensityMatrix.simulate(ghz, device=device)
print(rho.fidelity_with_pure(q.simulate(ghz)))
counts = rho.sample_with_readout(1000, device.readout_errors(4), seed=1)
```

A JSON description looks like this (units of time are whatever the
description uses consistently):

```json
{
  "name": "toy-5q",
  "num_qubits": 5,
  "qubits": [{"t1": 110.0, "t2": 85.0, "readout": [0.015, 0.04]}],
  "default_qubit": {"t1": 100.0, "t2": 90.0, "readout": [0.02, 0.045]},
  "gates": [
    {"gate": "cx", "qubits": [3, 4], "duration": 0.45, "depolarizing": 0.02,
     "channels": [{"rotation": {"pauli": "ZZ", "angle": 0.02}}]},
    {"gate": "cx", "duration": 0.3, "depolarizing": 0.008,
     "channels": [{"pauli": {"ZZ": 0.002}}]},
    {"num_qubits": 1, "duration": 0.035, "depolarizing": 0.0005, "over_rotation": 0.002}
  ]
}
```

`DeviceModel.from_noise_model` turns a `NoiseModel` into the equivalent
device.

## Quantum trajectories

Density matrices need 4ⁿ numbers and stop at 14 qubits. Quantum
trajectories (Monte-Carlo wave functions) need one state vector per
trajectory, so they reach any register a state vector holds. Each
trajectory applies one Kraus operator of every channel, chosen with its
Born probability. The average over trajectories converges to the
density-matrix result with a `1/√N` statistical error.

QSO samples unitary mixtures (Pauli, depolarising, dephasing and coherent
errors) up front. Damping is handled by the waiting-time method: runs of
gates without a jump execute as one fused, cache-blocked circuit. One
trajectory, compilation included, costs about three noiseless runs at
16–20 qubits; 200 trajectories share one compilation and take 1.2–1.3 times
as long as 200 noiseless runs (see
[benchmarks](https://github.com/LashSesh/qso/blob/main/docs/BENCHMARKS.md)).
Every trajectory has its own random stream, so results do not depend on the
number of threads.

```python
import metatron_qso as q

device = (
    q.DeviceModel("uniform")
    .with_default_qubit(t1=100.0, t2=80.0)
    .with_rule(gate="cx", duration=0.3, depolarizing=0.01)
    .with_rule(num_qubits=1, duration=0.035, depolarizing=0.001)
)
n = 16
ghz = q.Circuit(n).h(0)
for k in range(n - 1):
    ghz.cx(k, k + 1)
zz = q.PauliSum(" + ".join(f"Z{k} Z{k + 1}" for k in range(n - 1)))
sim = q.TrajectorySimulator(device, trajectories=200, seed=1)
e = sim.expectation(ghz, zz)
print(e.value, e.std_error, e.confidence_interval(0.95))
counts = sim.sample(ghz, shots=1000)        # with the device's readout errors
```

Trajectory samples pass a χ² test against density matrices on up to 10
qubits, including circuits with several jumps per trajectory.

## Correlated and slow noise

Real devices have noise that the channels above cannot express: errors
that act on several qubits at once, crosstalk on qubits a gate does not
touch, and noise whose correlations last longer than a gate. QSO models
these three as follows.

**Channels on up to four qubits.** `from_kraus`, `pauli`, `unitary` and
`rotation` accept operators and Pauli strings on one to four qubits, such
as `Channel.pauli({"ZZZ": 0.01})` or `Channel.rotation("ZZZZ", 0.02)`.
Density matrices and trajectories apply them like the others. In a gate
rule, a k-qubit channel acts on every k consecutive qubits of the gate.

**Crosstalk on spectators.** `with_rule(..., spectators=[(qubits,
channel)])` applies a channel on fixed qubits after every matching gate.
These qubits may lie outside the gate. They wait for the gate and relax
during it, like its own qubits.

**Waits.** `Circuit.id(q)` is an identity. A rule for the gate `"id"`
gives it a duration, so the qubit waits for that time, the way Ramsey and
echo experiments do.

**Noise correlated in time** is drawn once per trajectory:

| Noise | Description | Effect |
|---|---|---|
| quasi-static detuning | `detuning=("gaussian", σ)` or `("uniform", a)` per qubit | a frequency offset δ, constant during a trajectory: the qubit turns by `RZ(δ·t)` over every time `t` it spends in gates or waiting |
| telegraph noise | `telegraph=(ν, γ)` per qubit | a frequency shift that jumps between +ν and −ν at the times of a Poisson process with γ jumps per unit time; the qubit turns by `RZ(∫ν(t) dt)` |
| over-rotation spread | `over_rotation_spread=("gaussian", σ)` per rule | an error η added to the rule's relative over-rotation: θ ↦ θ(1 + ε + η) |

Density matrices describe a single step of noise at a time, so they
refuse these three kinds; `DeviceModel.is_markovian` tells whether a
device has none of them. Trajectories simulate them exactly: each
trajectory draws its detunings and spreads, and follows its telegraph
process through every gate and wait. A turn during a gate is applied
after the gate, like the relaxation.

A Ramsey experiment with a Gaussian detuning decays as `exp(−σ²t²/2)`. A
Hahn echo (an X halfway) refocuses a static detuning in every trajectory:

```python
import math
import metatron_qso as q

def device(wait, sigma=0.8):
    return (q.DeviceModel("quasi-static")
            .with_default_qubit(detuning=("gaussian", sigma))
            .with_rule(gate="id", duration=wait))

def z(device, circuit):
    sim = q.TrajectorySimulator(device, 4000, seed=1)
    return sim.expectation(circuit, q.PauliSum("Z0"))

t = 2.0
ramsey = q.Circuit(1).h(0).id(0).h(0)
echo = q.Circuit(1).h(0).id(0).x(0).id(0).h(0)
e = z(device(t), ramsey)
print(round(e.value, 3), round(e.std_error, 3), round(math.exp(-(0.8 * t) ** 2 / 2), 3))
# 0.264 0.01 0.278
print(z(device(t / 2), echo).value)
# 1.0
```

Telegraph noise gives the Kubo–Anderson decay
`e^{−γt} (cosh μt + (γ/μ) sinh μt)` with `μ = √(γ² − ν²)` (Paladino,
Galperin, Falci and Altshuler, Rev. Mod. Phys. 86, 361 (2014)). Slow
switching (γ < ν) makes the coherence oscillate, and at times short
compared with 1/γ the echo removes most of the decay (⟨Z⟩ = 0.92 instead
of 0.23 at t = 1.5 for ν = 1, γ = 0.2). Fast switching (γ ≫ ν) narrows the
decay to a rate of about ν²/(2γ) (motional narrowing).

The validation suite checks Ramsey and echo against these closed forms:
a Gaussian detuning, and telegraph noise with slow (ν = 1, γ = 0.2) and
fast (ν = 1, γ = 5) switching. It uses times from 0.5 to 6 and 8000
trajectories each. The echo of telegraph noise comes from the same
two-state equations, solved in closed form for each half. All 20
estimates lie within four standard errors (the largest deviation is 1.5),
and the echo of a static detuning is exact.

