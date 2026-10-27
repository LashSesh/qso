# Estimation and error mitigation

Every stochastic result in QSO comes as an `Estimate`. It holds the value,
its standard error and the number of samples behind it.
`confidence_interval(level)` gives the normal-approximation interval.
Mitigation methods propagate errors, so a mitigated value carries the
statistical price of mitigation in its error bar.

## Estimates from shots

`estimate` measures an observable as hardware would. It groups
qubit-wise commuting terms and gives each group a share of the shots. The
standard error comes from the spread of the per-shot values. The
allocation decides how the shots are spread:

| `allocation` | Shots per group |
|---|---|
| `"uniform"` | equal |
| `"weighted"` (default) | proportional to Σ\|c\| of the group's terms |
| `"adaptive"` | a `pilot` fraction uniformly to measure the standard deviations, the rest proportional to them (Neyman allocation, the variance-optimal choice). The pilot shots only choose the allocation and do not enter the estimate, so the estimate stays unbiased |

```python
import metatron_qso as q

c = q.Circuit(2).h(0).h(1)
h = q.PauliSum("5 Z0 Z1 + 0.1 X0 X1 + 0.1 Y0 Y1 + 0.1 X0 Y1")
for allocation in ["uniform", "weighted", "adaptive"]:
    e, groups = q.estimate(c, h, 4000, allocation=allocation, seed=1)
    print(allocation, e.value, e.std_error, [g["shots"] for g in groups])
```

Here only `5 Z0 Z1` fluctuates. Uniform allocation wastes three quarters
of the shots, and its standard error is about 1.9 times larger. In tests,
95 % intervals cover the exact value in 91–99 % of 300 repetitions.

Intervals use the normal approximation. When all shots of a group agree,
the sample variance is zero; QSO then reports the variance of a value
that spans the whole range with probability 1/(n + 2), so no estimate
claims to be exact. Near ±1, with few shots, the normal interval still
covers less than its nominal level.

The same arguments as everywhere select the backend: `noise=` for a
`NoiseModel`, `device=` for a `DeviceModel` (density matrices), and
`trajectories=` to simulate the device with trajectories.

## Readout errors

A `ReadoutCalibration` measures how bit strings are misread:

- `tensored` uses two circuits and assumes independent errors per qubit;
- `full` uses all 2ⁿ basis states and also captures correlated errors.

`mitigate(counts)` applies the inverse confusion matrix and returns
quasi-probabilities. `nearest_probability_distribution` turns them back
into a probability distribution. Passed to `estimate`, the calibration
mitigates every measured parity exactly:

```python
import metatron_qso as q

device = q.DeviceModel("readout").with_default_qubit(readout=(0.04, 0.09))
c = q.Circuit(2).ry(0, 0.5).cx(0, 1)
h = q.PauliSum("Z0 Z1 + 0.5 Z0")
exact = h.expectation(q.simulate(c))
cal = q.ReadoutCalibration.tensored(2, 100_000, seed=1, device=device)
raw, _ = q.estimate(c, h, 200_000, device=device, seed=2)
mitigated, _ = q.estimate(c, h, 200_000, device=device, seed=2, readout_calibration=cal)
print(exact, raw.value, mitigated.value, mitigated.std_error)
```

## Zero-noise extrapolation

`zne` runs the circuit at amplified noise and extrapolates to zero noise.
`fold` amplifies the noise by unitary folding: `G → G G† G` leaves the
ideal circuit unchanged and triples the noise of `G`. The executor is any
Python function from a circuit to a float or an `Estimate`. The
extrapolation (`"linear"`, `"richardson"`, `"polynomial"` with `degree`,
`"exponential"` with `asymptote`) propagates the standard errors.

Odd scale factors (1, 3, 5, …) fold every gate equally often and amplify
all noise by the same factor. Fractional factors fold only some gates and
work well only when all gates are about equally noisy.

```python
import metatron_qso as q

device = q.DeviceModel("dep").with_rule(num_qubits=2, depolarizing=0.02).with_rule(depolarizing=0.002)
c = q.Circuit(2).h(0).cx(0, 1).rz(1, 0.3).cx(0, 1).h(0)
z = q.PauliSum("Z0")

def run(folded):
    return q.DensityMatrix.simulate(folded, device=device).expectation(z)

r = q.zne(c, [1.0, 3.0, 5.0], run)
print(run(c), r["estimate"].value, z.expectation(q.simulate(c)))   # noisy, mitigated, ideal
```

ZNE amplifies gate noise only. Readout errors do not grow with folding,
so combine ZNE with readout mitigation.

## Probabilistic error cancellation

The inverse of a Pauli channel is a quasi-probabilistic mixture of Paulis.
Some of its weights are negative. `pec_expectation` inserts Paulis drawn
from that mixture after every noisy operation and reweights the results.
The average is the noiseless value exactly, and the tests check this
exactly by summing over all corrections. The price is a standard error
larger by the overhead Γ, which grows exponentially with the number of
noisy gates. PEC needs the device's noise as Pauli channels.

```python
import metatron_qso as q

device = q.DeviceModel("pauli").with_rule(gate="cx", depolarizing=0.05,
                                          channels=[q.Channel.pauli({"ZZ": 0.02})])
bell = q.Circuit(2).h(0).cx(0, 1)
r = q.pec_expectation(bell, q.PauliSum("Z0 Z1"), device, samples=4000, seed=1)
print(r["estimate"].value, r["estimate"].std_error, r["overhead"])
```

## Pauli twirling

`twirl` inserts a random Pauli before every `cx`, `cy`, `cz` and `swap`,
and the matching Pauli after it. The ideal circuit does not change. On
average, the noise after these gates becomes its Pauli twirl (see
`Channel.twirled`). Coherent errors and damping then act as stochastic
Pauli noise, which degrades results predictably and which PEC can invert.
`twirled_average` averages any executor over twirled copies:

```python
import metatron_qso as q

coherent = q.DeviceModel("zz").with_rule(gate="cx", channels=[q.Channel.rotation("ZZ", 0.5)])
c = q.Circuit(2).h(0).cx(0, 1).rz(1, 0.3).cx(0, 1).h(0)
z = q.PauliSum("Z0")
avg = q.twirled_average(
    c, 200, lambda t: q.DensityMatrix.simulate(t, device=coherent).expectation(z), seed=1
)
print(avg.value, avg.std_error)
```

## Learned noise

PEC above takes the noise from the device model. On hardware the noise
has to be measured first. `learn_layer` does this for a layer of
two-qubit Clifford gates (`cx`, `cy`, `cz`, `swap` on disjoint qubits),
following van den Berg et al. (Nat. Phys. 19, 1116 (2023)). Twirled, the
layer's noise is a Pauli channel, and the sparse Pauli–Lindblad model
writes it with one rate λ_k per generator P_k: every one-qubit Pauli and
every two-qubit Pauli on neighbouring qubits. A Pauli b then has fidelity
f_b = exp(−2 Σ λ_k), summed over the generators that anticommute with b.

The experiment prepares product eigenstates in a set of product bases
in which every pair of coupled qubits sees all nine pairs of letters
(nine bases when the coupling graph has at most four colours, 27 up to
13 colours). It repeats the twirled layer an even number of times d,
and measures. Readout is
twirled too, so readout errors only scale the signal. The decay of ⟨b⟩
with d gives the product f_b·f_{UbU} free of state preparation and
measurement errors. Non-negative least squares turns the products into
rates. Standard errors come from a bootstrap over the twirled circuits.

One limitation comes from the method itself: where the layer maps b to
another Pauli, only the product f_b·f_{UbU} is learnable, and the two
are taken equal. Noise that commutes with the layer (depolarising
noise, a phase error on the control of a cx) is recovered. Noise that
the layer moves, such as a ZZ error after a cx, is split evenly between
b and UbU. The result lists the Paulis this affects (`symmetrised`).

`pec_layered` cancels the learned noise in a circuit of blocks separated
by the layer, `B₀ L B₁ L … B_m`. `pea_layered` amplifies it by inserting
Paulis with positive probabilities, without the sign problem and
overhead of PEC, and extrapolates the values to zero noise. The inserted
Paulis are one-qubit gates. On hardware they are merged into the
neighbouring gates. In the simulator they are noiseless when the device
has no one-qubit gate noise.

```python
import metatron_qso as q

device = (q.DeviceModel("learnable", idle_noise=False)
          .with_rule(gate="cx", depolarizing=0.012,
                     channels=[q.Channel.rotation("ZI", 0.12)])
          .with_default_qubit(readout=(0.02, 0.04)))
layer = q.Circuit(2).cx(0, 1)
learned = q.learn_layer(layer, device, seed=1)
model = learned["model"]
for g, rate, se in zip(model.generators, model.rates, learned["std_errors"]):
    print(f"{g:8} {rate:.5f} ± {se:.5f}")

blocks = [q.Circuit(2).ry(0, 0.7).ry(1, -0.4), q.Circuit(2).ry(0, -1.1).rz(1, 2.0),
          q.Circuit(2).ry(1, 0.3)]
obs = q.PauliSum("Z0 Z1 + 0.5*X0")
pec = q.pec_layered(blocks, layer, model, device, obs, samples=4000, seed=2)
pea = q.pea_layered(blocks, layer, model, device, obs, [1.0, 2.0, 3.0], 4000, seed=3)
print(pec["estimate"].value, pea["estimate"].value)
```

In the validation suite, a four-qubit device with two-qubit depolarising
noise, a coherent phase error and readout errors is learned. All 39 rates
lie within three standard errors of their exact values, and PEC with the
learned model gives 0.534 ± 0.005 for a value of 0.533 (raw 0.492).

## Virtual distillation

Virtual distillation needs no noise model (Huggins et al., Phys. Rev. X 11,
041036 (2021); Koczor, Phys. Rev. X 11, 031057 (2021)). Noise mixes other
states into ρ, but the dominant eigenvector of ρ usually stays close to the
ideal state. With k copies of ρ,

    ⟨O⟩_k = Tr(ρᵏ O) / Tr(ρᵏ)

weights each eigenvector by the k-th power of its probability, so the
others fade and ⟨O⟩_k tends to the value of the dominant eigenvector.
`distilled_expectation(rho, observable, copies)` computes it exactly from a
density matrix (k = 1, 2, 3); `dominant_expectation` gives the limit.

On hardware, `Tr(ρᵏ P)` for a Pauli string P is the expectation of S·P on
the first copy in ρ^⊗k, where S shifts the copies cyclically. Both factor
over the qubits: on the k copies of qubit q, `S_q P_q` is a k-qubit unitary,
and the unitaries of different qubits commute. `DistillationCircuits`
rotates every copy so that P becomes a product of Zs, turns each unitary
into a diagonal one with a k-qubit gate (its eigenvectors) and measures in
the computational basis. Each shot contributes the real part of the product
of the eigenvalues it lands in. One setting per Pauli string estimates the
numerator, one more estimates `Tr(ρᵏ)`, and `estimate` returns their ratio
with a standard error from the delta method. Copy c of qubit q is qubit
c·n + q.

```python
import metatron_qso as q

prep = q.Circuit(3).h(0).cx(0, 1).cx(1, 2).ry(1, 0.3)
rho = q.DensityMatrix.simulate(prep, noise=q.NoiseModel.depolarizing(0.01, 0.04))
o = q.PauliSum("Z0 Z1 + X0 X1 X2")
ideal = o.expectation(q.simulate(prep))
for k in (1, 2, 3):
    print(k, abs(q.distilled_expectation(rho, o, k) - ideal))   # 0.19, 3.6e-3, 6.9e-5

vd = q.DistillationCircuits(prep, o, copies=2)       # 3 settings on 6 qubits
counts = [q.simulate(c).sample(20_000, seed=s) for s, c in enumerate(vd.circuits)]
print(vd.estimate(counts))                            # noiseless here: ≈ ideal
```

The tests check the exact values against `Σ_i p_iᵏ⟨v_i|O|v_i⟩ / Σ_i p_iᵏ`
for random mixtures, the circuits on copies of ρ against the exact value
(to 10⁻¹¹, for one, two and three copies and Pauli strings with X, Y and
Z), sampled shots against the exact value within their standard error,
and the suppression of depolarising noise above. Distillation does not
remove coherent errors: the dominant eigenvector of ρ differs from the
ideal state when the noise rotates it, and that difference remains.

## Classical shadows

A `ClassicalShadow` records snapshots of a state, each measured in random
X, Y or Z bases. One set of snapshots estimates many local observables at
once. A Pauli string on k qubits has single-snapshot variance 3ᵏ − ⟨P⟩²,
and the tests check this value. `median_of_means` is a robust
alternative to the mean.

```python
import metatron_qso as q

ghz = q.Circuit(3).h(0).cx(0, 1).cx(1, 2)
shadow = q.ClassicalShadow.collect(ghz, 20_000, seed=1,
                                   noise=q.NoiseModel.depolarizing(0.0, 0.02))
for label in ["Z0 Z1", "Z1 Z2", "X0 X1 X2"]:
    e = shadow.estimate(q.PauliSum(label))
    print(label, e.value, e.std_error)
```

## In Rust

The same functionality lives in `metatron_qso::estimation`,
`metatron_qso::mitigation` and `metatron_qso::noise`. The example
`noise_mitigation` runs everything on a five-qubit toy device and prints a
table: readout mitigation plus ZNE recovers ⟨H⟩ = 4.003 ± 0.008 for a GHZ
state (noiseless 4, raw 3.38).

```rust
use metatron_qso::backend::DeviceBackend;
use metatron_qso::estimation::{EstimatorOptions, ShotAllocation, estimate};
use metatron_qso::noise::{DeviceModel, GateRule, QubitProperties};
use metatron_qso::quantum::{Circuit, PauliSum};

fn main() -> metatron_qso::Result<()> {
    let device = DeviceModel::new("toy")
        .with_default_qubit(QubitProperties::new().with_t1_t2(100.0, 80.0).with_readout(0.02, 0.04))
        .with_rule(GateRule::for_gate("cx").with_duration(0.3).with_depolarizing(0.01));
    let backend = DeviceBackend::trajectories(device, 2000)?;
    let mut c = Circuit::new(3);
    c.h(0).cx(0, 1).cx(1, 2);
    let h: PauliSum = "Z0 Z1 + Z1 Z2".parse()?;
    let opts = EstimatorOptions::new(20_000).with_allocation(ShotAllocation::Adaptive { pilot: 0.1 });
    let e = estimate(&backend, &c, &[], &h, &opts)?.estimate;
    println!("{} ± {}", e.value, e.std_error);
    Ok(())
}
```
