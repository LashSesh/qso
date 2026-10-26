# Error-correction experiments

A quantum error-correction experiment repeatedly measures the stabilizers
of a code. Without noise, these measurements are deterministic; noise
flips some of them, and a decoder infers from the flips which logical
operators the noise has changed. QSO runs such experiments in the style
of Stim (Gidney, Quantum 5, 497 (2021)):

- a `DetectorCircuit` holds Clifford gates, Pauli noise, measurements,
  *detectors* and *logical observables*;
- the Pauli-frame sampler simulates many shots of it, 64 per machine word;
- the detector error model lists each independent error and its effect;
- a union-find or an exact minimum-weight matching decoder predicts the
  observable flips of each shot;
- `logical_error_rate` combines all four.

## Detector circuits

A detector is a parity of measurement results that is fixed when there is
no noise. A *detection event* is a detector whose parity noise has
flipped. A logical observable is a parity of measurement results whose
flips the decoder must predict. `measure` returns the index of its
result in the measurement record, and detectors and observables refer to
these indices.

```python
import metatron_qso as q

# One round of a distance-3 bit-flip code: data qubits 0-2, ancillas 3-4.
c = q.DetectorCircuit(5)
c.x_error(0.1, [0, 1, 2])                    # each data qubit flips w.p. 0.1
c.cx(0, 3).cx(1, 3).cx(1, 4).cx(2, 4)        # ancillas collect Z0Z1, Z1Z2
m = [c.measure(3), c.measure(4)]
for r in m:
    c.detector([r])                           # parities are 0 without noise
data = [c.measure(k) for k in range(3)]
c.detector([m[0], data[0], data[1]])          # final data agree with the parities
c.detector([m[1], data[1], data[2]])
c.observable(0, [data[0]])                    # the logical bit

detectors, observables = c.sample(10_000, seed=1)
print(detectors.shape, observables.mean())    # (10000, 4) and about 0.1
```

The noise channels are:

| Method | Channel |
|---|---|
| `x_error(p, qubits)`, `y_error`, `z_error` | the Pauli with probability p on each qubit |
| `depolarize1(p, qubits)` | X, Y or Z, each with probability p/3 (p ≤ 3/4) |
| `depolarize2(p, qubits)` | each of the 15 non-identity two-qubit Paulis with probability p/15, on consecutive pairs (p ≤ 15/16) |
| `measure(q, flip=p)` | the result is reported flipped with probability p |

The gates are H, S, X, Z, CX and CZ. `append` adds any Clifford `Circuit`
(see [Stabilizer simulation](stabilizers.md)).

## Pauli-frame sampling

Every shot carries a Pauli frame: the Pauli error accumulated so far,
relative to a noiseless run. Clifford gates conjugate the frame, and
noise multiplies it. A measurement is flipped when the frame has an X or
Y on the measured qubit. Bit s of a machine word belongs to shot s, so
one word operation updates 64 shots.

Because detectors are deterministic, a detection event is just the
parity of the flips of its measurements, and no reference run is needed.
The Z part of the frame of a freshly reset or measured qubit is drawn at
random; this reproduces random measurement outcomes as random flips.
Before sampling, QSO runs 64 noiseless shots. If any detector or
observable is not deterministic, it raises an error; a random parity
escapes this check with probability 2⁻⁶⁴.

## Detector error models

`detector_error_model()` lists the independent error mechanisms of the
circuit as `(probability, detectors, observables)` triples, with
`observables` a bit mask:

- A depolarizing channel is written exactly as independent Pauli errors:
  X, Y and Z each with probability ½ − ½(1 − 4p/3)^(1/2) for one qubit,
  and the 15 two-qubit Paulis each with ½ − ½(1 − 16p/15)^(1/8). This is
  the conversion Stim uses.
- Each error is propagated through the noiseless circuit as a Pauli
  frame, 64 errors per word and up to 4096 per pass.
- Errors with the same effect are merged: p = p₁(1 − p₂) + p₂(1 − p₁).

With `decompose=True`, an error that flips more than two detectors (a Y
error, or a two-qubit error) is split into its single-qubit X and Z
parts. Each part must then flip at most two detectors, and the model
becomes a matching graph. Each part keeps the error's full probability,
so correlated errors are counted slightly too often; matching decoders do
the same.

The scheduled cross-validation compares QSO's error models with Stim
1.16 on repetition and surface-code memory experiments, term by term. All
terms agree to within 10⁻¹⁵ relative. The detection-event rates agree
with Stim's sampler to sampling precision
([`benchmarks/cross_framework`](https://github.com/LashSesh/qso/tree/main/benchmarks/cross_framework)).

## Decoding with union-find

`UnionFindDecoder` is the weighted union-find decoder of Delfosse and
Nickerson (Quantum 5, 595 (2021)). It works on the matching graph:

1. **Graph.** Detectors are nodes, plus one node for the boundary. Every
   error is an edge of weight ln((1 − p)/p).
2. **Growth.** Every cluster with an odd number of detection events grows
   at unit rate along the edges leaving it, and an edge is covered when
   the growth along it reaches its weight. A cluster stops when it holds
   an even number of events or touches the boundary. Growth is
   event-driven: every round advances all clusters to the next covered
   edge, so the weights are used exactly rather than rounded to steps.
3. **Peeling.** A spanning forest of each cluster is peeled from the
   leaves: an event at a leaf takes the edge to its parent into the
   correction.

The correction's observable flips are the prediction.

```python
import metatron_qso as q

c = q.surface_code_memory(3, 3, q.CircuitNoise.uniform(0.003))
decoder = q.UnionFindDecoder(c.detector_error_model(decompose=True))
detectors, observables = c.sample(50_000, seed=2)
predicted = decoder.decode_batch(detectors)          # one mask per shot
failures = (predicted != observables[:, 0]).mean()
r = q.logical_error_rate(c, 50_000, seed=2)          # the same in one call
assert r.errors == int((predicted != observables[:, 0]).sum())
print(f"{r.rate:.4f} ± {r.standard_error:.4f}")
```

Union-find runs in almost linear time. It is not a minimum-weight
decoder, and with unequal weights it is not guaranteed to correct every
error of weight below d/2. On the four circuits of the cross-validation,
its logical error rate is at most 6% above PyMatching's (minimum-weight
perfect matching). An earlier version rounded the weights to integer
steps, and an independent review found syndromes of three errors at
d = 7 that it misdecoded; exact growth fixed them, and a unit test keeps
them fixed. `logical_error_rate` samples in blocks of at most 65 536
shots and 1 GiB of results, so its memory does not grow with the number
of shots.

## Exact decoding: minimum-weight matching

`MatchingDecoder` finds, for each shot, the lightest set of errors that
explains the detection events: a minimum-weight perfect matching of the
events in pairs or to the boundary (Dennis, Kitaev, Landahl and Preskill,
J. Math. Phys. 43, 4452 (2002)). It takes three steps:

1. **Distances.** Dijkstra's algorithm gives the distance between events
   and the observables flipped along the path. The distances from a
   detector are cached when it first fires, up to 2²¹ entries (32 MiB).
   Beyond that, each search stops as soon as no farther event can be
   worth pairing.
2. **Savings.** Pairing events i and j instead of sending both to the
   boundary saves b_i + b_j − d_ij, where b is the distance to the
   boundary. Only positive savings become edges, so the boundary needs no
   copies of the events.
3. **Blossoms.** Edmonds' blossom algorithm (after van Rantwijk's
   implementation, the one NetworkX uses) finds the matching with the
   largest total saving: the lightest correction.

Weights are rounded to integers, with the largest edge at 2²⁰, so the
blossom algorithm computes exactly. Errors more likely than not have
weight 0.

```python
import metatron_qso as q

c = q.surface_code_memory(5, 5, 0.006)
detectors, observables = c.sample(20_000, seed=3)
dem = c.detector_error_model(decompose=True)
predicted = q.MatchingDecoder(dem).decode_batch(detectors)
mwpm = q.logical_error_rate(c, 20_000, seed=3, decoder="matching")
uf = q.logical_error_rate(c, 20_000, seed=3)        # the same shots
assert mwpm.errors == int((predicted != observables[:, 0]).sum())
assert mwpm.errors < uf.errors
```

The results are checked against independent references:

- **Blossom algorithm.** On 4000 random graphs of up to 14 vertices, with
  weights in narrow ranges so that ties and nested blossoms are common,
  it matches dynamic programming over subsets.
- **Correction weight.** On 300 random error models, for every syndrome,
  the correction weighs as little as the lightest set of errors with that
  syndrome found by enumeration. The prediction is the observable flip
  of one such set.
- **Faults.** It corrects every single fault, every pair of faults at
  distance 5 and 20 000 random triples at distance 7.
- **PyMatching.** In the scheduled cross-validation, on the same error
  model and the same syndromes, its predictions agree with PyMatching's.
  In a run with 100 000 shots for each of eight circuits, a single shot
  differed, on the X-basis surface code. Two exact decoders may differ
  where corrections weigh the same, or through their rounding of the
  weights.

## Memory experiments

Two families of memory experiments are built in. They protect |0⟩ (the
surface code also |+⟩) and take a `CircuitNoise` or a single probability
for every kind of noise.

| Noise | Where |
|---|---|
| `after_gate1` | depolarizing noise after each H |
| `after_gate2` | two-qubit depolarizing noise after each CNOT |
| `measurement` | flip of each measurement result |
| `after_reset` | X error after each reset |
| `before_round` | depolarizing noise on the data qubits before each round |

**`repetition_code_memory(d, rounds, noise)`.** d data qubits and d − 1
ancillas measure Z_iZ_{i+1}. Each detector compares a parity with the
previous round: the first round with 0, the last with the final data
measurements. Observable 0 is the final value of data qubit 0.

**`surface_code_memory(d, rounds, noise)`.** The rotated surface code of
odd distance d has d² data qubits and d² − 1 ancillas. Its CNOTs follow
the order that keeps hook errors perpendicular to the logical operators:
NW, NE, SW, SE for X ancillas, NW, SW, NE, SE for Z ancillas. The
detectors are the Z-type stabilizers, so the model is a matching graph
for bit flips, and observable 0 is the logical Z, Z on the first row of
data qubits. `surface_code_memory(d, rounds, noise, basis="x")` (in Rust
`surface_code_memory_with(…, MemoryBasis::X)`) protects |+⟩ instead. The
data qubits are prepared and measured in the X basis, through H gates
with one-qubit gate noise. The X-type stabilizers are the detectors, and
the logical X, X on the first column, is the observable.

Under uniform circuit-level noise p (d rounds, seed 7, 10⁶ shots at
p = 0.001 and 10⁵ otherwise; `cargo run --release --example surface_code`),
with union-find / matching on the same shots:

| p | d = 3 | d = 5 | d = 7 |
|---|---|---|---|
| 0.001 | 8.4·10⁻⁴ / 8.3·10⁻⁴ | 1.4·10⁻⁴ / 1.3·10⁻⁴ | 2.1·10⁻⁵ / 1.7·10⁻⁵ |
| 0.003 | 7.0·10⁻³ / 6.8·10⁻³ | 3.8·10⁻³ / 3.6·10⁻³ | 1.6·10⁻³ / 1.4·10⁻³ |
| 0.01 | 0.062 / 0.060 | 0.086 / 0.081 | 0.109 / 0.100 |

Below the threshold, larger codes fail less often. Above it, they fail
more often. Here the threshold lies between 0.3% and 1%. The test
`surface_codes_suppress_logical_errors_below_threshold` in
`tests/validation.rs` checks both directions.

```rust
use metatron_qso::qec::{CircuitNoise, logical_error_rate, surface_code_memory};

fn main() -> metatron_qso::Result<()> {
    for d in [3, 5] {
        let circuit = surface_code_memory(d, d, &CircuitNoise::uniform(0.001))?;
        let r = logical_error_rate(&circuit, 20_000, 7)?;
        println!("d = {d}: {:.2e} ± {:.1e}", r.rate, r.standard_error);
    }
    Ok(())
}
```

## Stim circuits and the command line

`to_stim()` writes a circuit in Stim's text format, with relative
measurement records (`rec[-k]`) and exact probabilities.
`DetectorCircuit.from_stim(text)` reads one. It understands the
following instructions:

- the Clifford gates H, S, S_DAG, X, Y, Z, SQRT_X(_DAG), SQRT_Y(_DAG), CX,
  CY, CZ, XCZ, YCZ and SWAP;
- measurements and resets in the X, Y and Z bases (M, MX, MY, R, RX, RY,
  MR, MRX, MRY), with an optional flip probability;
- the noise channels X_ERROR, Y_ERROR, Z_ERROR, DEPOLARIZE1 and
  DEPOLARIZE2;
- DETECTOR, OBSERVABLE_INCLUDE and REPEAT blocks.

TICK and coordinates are ignored. Inverted measurement targets (`!q`)
read the same as plain ones, since detection events are relative to the
noiseless circuit. Any other instruction is an error that names it and
its line: classical feedback, MPP, correlated, heralded and Pauli-channel
noise.

```python
import metatron_qso as q

text = q.repetition_code_memory(3, 2, 0.01).to_stim()
assert text.startswith("R 0\n")
assert "DETECTOR rec[-1] rec[-3]" in text
assert q.DetectorCircuit.from_stim(text).to_stim() == text
```

Circuits that Stim generates read back with Stim's error model. Unit
tests cover the repetition code and the rotated surface code in both
bases, and the scheduled cross-validation covers them at d = 5.

`qso qec` runs memory experiments from the shell:

```text
qso qec --distance 3,5,7 --p 0.001,0.01 --shots 100000
qso qec --basis x --decoder matching --distance 5 --p 0.003
qso qec --code repetition --distance 9 --p 0.05 --stim rep9.stim
qso qec --circuit surface.stim --decoder matching --shots 100000
```

## Limits and costs

| Quantity | Limit |
|---|---|
| Qubits of a detector circuit | 65 536 |
| Operations | 2²⁴ |
| Observables | 64 |
| Results of one `sample` call | shots × (detectors + 64) ≤ 2³³ bits (1 GiB); in Python, shots × (detectors + observables) ≤ 2³⁰ |
| Error components of a model | 2²²; their detector lists before merging, 2²⁷ entries (1 GiB); propagation, 2³⁴ word operations |
| Code distance, rounds | 101, 10 000, and at most 2²⁴ operations (checked before building) |
| Imported Stim circuits | 65 536 qubits, 2²⁴ operations after expanding REPEAT blocks (checked before expanding) |
| Matching decoder | cached distances 2²¹ entries (32 MiB); bounded searches beyond |

- **Sampling.** Each operation costs one word operation per 64 shots.
- **Error model.** The model takes one pass through the circuit per 4096
  error components, or more when the circuit has more than 4096
  measurements (each pass holds its records).
- **Matching.** A shot with k detection events costs k distance lookups
  (or bounded searches) and O(k³) for the blossom algorithm. At low noise
  matching is about as fast as union-find. At p = 0.01 and d = 7 it takes
  twice as long (7.4 s against 3.7 s for 10⁵ shots).
- **Example timing.** On one core, a distance-21 surface code with 21
  rounds (881 qubits, 4840 detectors) needs about 0.5 s for its error
  model and 15 s for 100 000 shots decoded by union-find at p = 0.001. The error model
  of d = 41 with 41 rounds (35 280 detectors) takes 30 s.
