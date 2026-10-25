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
