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
  observable flips of each shot, or, where errors flip more than two
  detectors (colour codes, quantum LDPC codes), belief propagation with
  ordered-statistics decoding (BP+OSD);
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

With `decompose=True`, every error is split into graph-like parts, each
flipping at most two detectors, so that the model becomes a matching
graph. The split follows Stim's rule within each noise channel (the X
and Z errors of a depolarizing channel on its qubits, and their
combinations):

- A combination that flips one detector is a part.
- So is a combination that flips two detectors, at least one of which no
  single-detector combination flips (an "irreducible pair").
- Every other combination is written as at most two irreducible pairs,
  whose remainder single-detector combinations flip, and finished with
  single-detector combinations.
- A combination that this rule cannot write, or whose parts would not
  add up to it, is split into its single-qubit X and Z parts; each must
  then flip at most two detectors. Stim instead decomposes such an error
  against the graph-like errors of the whole model, which can give other
  (equally valid) parts.
- A Y error (`Y_ERROR`) is one error, not a family: it stays whole when
  it flips at most two detectors, and otherwise is split by the rule
  above over its X and Z parts.

Each part keeps the error's full probability, so correlated errors are
counted slightly too often; matching decoders do the same.
`correlated_error_model()` returns the same decomposition, but keeps the
parts of each error together (see [Correlated matching](#correlated-matching)).
On Stim's generated surface-code and repetition-code circuits, both give
Stim's decomposed model error by error, parts included, to 10⁻¹⁵
relative (cross-validation).

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
weight 0, and errors of probability 0 are left out. Where several
corrections weigh the same, which one is returned can depend on the
distances earlier shots have cached. Each of them has minimum weight,
and a seeded run is reproducible.

```python
import metatron_qso as q

c = q.surface_code_memory(5, 5, 0.006)
detectors, observables = c.sample(20_000, seed=3)
dem = c.detector_error_model(decompose=True)
predicted = q.SparseMatchingDecoder(dem).decode_batch(detectors)
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

## Fast matching: regions on the detector graph

`SparseMatchingDecoder` computes the same minimum-weight matching without
the distances between events. It runs the primal-dual blossom algorithm
on the detector graph itself, as sparse blossom does (Higgott and Gidney,
Quantum 9, 1600 (2025)):

- Every detection event owns a region of the graph. Its radius is the
  event's dual variable. Regions grow, freeze or shrink at unit speed in
  a common time.
- When two regions touch, the path between their events is tight. A
  growing region that touches a matched pair adds both to its
  alternating tree. Two trees that touch augment and become matched
  pairs. Two regions of one tree close an odd cycle, which becomes a
  blossom region. A growing region that reaches the boundary matches to
  it.
- A shrinking blossom at radius zero shatters into its children. A
  shrinking single event at radius zero closes a blossom with its tree
  neighbours.

A shot touches only the nodes the regions cover. At low noise, where
events lie apart, its cost is close to proportional to the events and
their area; it grows faster where blossoms nest deeply, because a region
that changes its growth reschedules its whole area. A chain of 2001
detectors that all fire (blossoms nested 1000 deep) takes 51 ms, PyMatching
1.3 s. Weights are those of `MatchingDecoder`, doubled so that two
growing regions meet at integer times. `logical_error_rate(...,
decoder="matching")` and `qso qec --decoder matching` use this decoder;
`decode_with_weight` returns the correction's weight Σ ln((1 − p)/p).

The checks:

- **The exact decoder.** On every syndrome of 400 random error models,
  on 4000 sampled shots of surface codes (d = 3 to 9) and a repetition
  code, on 2400 dense random syndromes of random graphs and on the even
  syndromes of odd cycles, the correction weighs exactly as much as
  `MatchingDecoder`'s. Together these shots form 11 000 blossoms and
  shatter 1600.
- **PyMatching.** In the cross-validation, on the same error model and
  syndromes, the weights agree to 10⁻⁵ relative (both decoders round
  ln((1 − p)/p) to integers) on the first 2000 shots of every circuit and
  on every shot whose prediction differs: predictions differ only where
  corrections weigh the same.
- **Speed.** Measured against PyMatching 2.4 on one core of the same
  machine, on surface-code memory experiments with d rounds (after a
  warm-up, medians of five interleaved runs of 20 000 shots, as the
  cross-validation measures them):

| d | p | detectors | QSO (µs/shot) | PyMatching (µs/shot) |
|---|---|---:|---:|---:|
| 11 | 0.003 | 720 | 23 | 12 |
| 17 | 0.001 | 2 592 | 22 | 14 |
| 17 | 0.005 | 2 592 | 220 | 121 |
| 25 | 0.001 | 8 112 | 95 | 88 |
| 25 | 0.005 | 8 112 | 865 | 520 |

PyMatching's own timing varies between runs (at d = 25, p = 0.001 by up
to a factor 2), so the ratios, 1.1 to 1.9 here and up to 2.3 at
p = 0.01 to 0.05, are approximate.
`MatchingDecoder`, on the complete graph of events, takes 94 000 µs per
shot at d = 25 and p = 0.001.

## Correlated matching

Decomposition forgets that the parts of an error happen together: a Y
error on a data qubit of a surface code flips two X-type and two Z-type
detectors, and matching sees two independent edges. Two-pass matching
(`CorrelatedMatchingDecoder`) uses this knowledge, with the rule of
PyMatching 2.4's `enable_correlations=True`:

1. For every two edges e ≠ f that are parts of the same errors, the
   probability of f given e is q = min(½, P(e and f)/P(e)). Here P(e)
   combines, as independent errors, every error with part e, and
   P(e and f) every error with both. This gives the rule "if e happened,
   f weighs ln((1 − q)/q)".
2. The first pass matches as usual. The correction is the set of edges
   on shortest paths between the matched detection events; edges used an
   even number of times cancel.
3. Every rule of a correction edge lowers the weight of its edge f, but
   never raises it. The second pass matches again on the reweighted
   graph, and its prediction is the result.

```python
import metatron_qso as q

circuit = q.surface_code_memory(3, 3, 0.004)
model = circuit.correlated_error_model()
decoder = q.CorrelatedMatchingDecoder(model)
det, obs = circuit.sample(1000, seed=1)
predicted = decoder.decode_batch(det)
assert (predicted != obs[:, 0]).mean() < 0.05
rate = q.logical_error_rate(circuit, 1000, seed=1, decoder="correlated_matching")
```

QSO's own memory experiments use only the checks of the memory basis as
detectors. The parts of a Y error then fall on one kind of detector, and
correlations change little. Circuits whose detectors include both kinds,
such as Stim's generated ones, gain the most. The cross-validation runs
Stim's rotated surface code (d = 5, 5 rounds, p = 0.006, 50 000 shots)
on Stim's model and syndromes:

| Decoder | Logical error rate |
|---|---:|
| PyMatching | 2.44·10⁻² |
| PyMatching, `enable_correlations=True` | 1.94·10⁻² |
| QSO, two-pass, on Stim's model | 1.93·10⁻² |
| QSO, two-pass, on QSO's own decomposition | 1.93·10⁻² |

The two two-pass decoders disagree on 0.01% of shots: when the first
pass finds shortest paths of equal weight, the choice can differ from
PyMatching's, and the second pass then starts from other weights. The
check allows 10⁻³; the unrotated surface code, with more ties, measures
4·10⁻⁴. `correlations_lower_the_logical_error_rate` in the unit tests
checks the gain on Stim's d = 5 circuit, read from
`tests/data/stim/surface_d5_memory_z.stim`.

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

## Beyond matching: BP+OSD, colour and bicycle codes

In a colour code or a quantum LDPC code, one error flips three or more
detectors. Such a model is not a matching graph. `BpOsdDecoder` decodes
it on its Tanner graph, with detectors as checks and errors as bits,
following Roffe et al., "Decoding across the quantum LDPC code landscape"
(2020), and the `ldpc` package:

- **Belief propagation.** Log-likelihood messages run on the flooding
  schedule. The min-sum rule is scaled by a factor (default 0.625; 0
  gives 1 − 2⁻ᵗ in iteration t). The product-sum rule is also
  available. BP stops as soon as its hard decision flips exactly the
  fired detectors.
- **Ordered statistics.** If BP does not converge, OSD sorts the errors
  by their posterior, most likely first. The first independent columns
  form an information set, and OSD-0 solves for it on a reduced GF(2)
  basis built column by column.
- **Combination sweep.** The default, OSD-CS of order 7, also tries every
  single error outside the information set and every pair among the
  first `osd_order` of them. It keeps the candidate with the largest
  prior likelihood, Σ ln((1 − p)/p) (`ldpc` sums ln(1/p)). The
  exhaustive variant tries every subset of the first `osd_order`, at
  most 20.

`max_iterations=0` skips BP (in `ldpc` it means as many iterations as
errors). OSD of order 0 is OSD-0 for every method, as in `ldpc`. Errors
that flip no detector are left out, as in the matching decoders.

Two families of codes are built in, and `CssCode(n, x_checks, z_checks)`
takes any others. The logical operators are computed and paired, so that
X logical i and Z logical j overlap oddly exactly when i = j.

- **Colour codes.** `CssCode.color_code(d)` is the triangular colour code
  on the 6.6.6 lattice. It has odd distance d, (3d² + 1)/4 qubits and
  one logical qubit. The same faces of weight 4 and 6 serve as X and Z
  checks.
- **Bivariate bicycle codes.** `CssCode.bivariate_bicycle(l, m, a, b)`
  builds the code of Bravyi et al., Nature 627, 778 (2024), with
  H_X = [A | B] and H_Z = [Bᵀ | Aᵀ]. A and B are sums of monomials
  x^i y^j in the cyclic shifts x = S_l ⊗ I_m and y = I_l ⊗ S_m.
  `CssCode.bivariate_bicycle_code(n)` returns one of the codes from the
  paper's table, with its distance: [[72, 12, 6]], [[90, 8, 10]],
  [[108, 8, 10]], [[144, 12, 12]] ("gross") or [[288, 12, 18]].

`css_code_memory(code, rounds, noise, basis="z")` measures every X check
and then every Z check in each round, with one ancilla each. The CNOTs of
each type are packed into layers that touch every qubit at most once.
The detectors and observables are those of the memory basis. This
schedule is not tuned against hook errors, so the circuit distance can
be below the code distance.

```python
import metatron_qso as q

code, distance = q.CssCode.bivariate_bicycle_code(72)
assert (code.n, code.k, distance) == (72, 12, 6)
circuit = q.css_code_memory(code, 6, 0.001)
rate = q.logical_error_rate(circuit, 200, seed=1, decoder="bposd")
assert rate.rate < 0.05
```

The checks:

- **Minimum weight.** Without BP and with every non-pivot column in the
  exhaustive search, OSD returns the most likely correction. This holds
  on every syndrome of 150 random error models with hyperedges, compared
  with enumeration. Every variant's correction flips exactly the fired
  detectors, and the sweeps are never heavier than OSD-0.
- **Low-weight errors.** With every BP variant, the decoder corrects
  every single and every pair of bit flips on the colour codes of
  distance 5 and 7 and on the [[72, 12, 6]] code.
- **The Steane-code exception.** On the distance-3 colour code (the
  Steane code), an error on the qubit shared by all three checks fires
  every detector. Scaled min-sum and product-sum BP then converge in
  their first iteration to four errors on the other qubits, which is a
  logical error. `ldpc` returns the same four errors with the same
  options (checked in the cross-validation). Plain min-sum
  (`scaling=1.0`) leaves this case to OSD, which corrects it.
- **ldpc.** The cross-validation compares the two decoders shot by shot
  with the same options:
  - On colour codes (d = 7, 9) and the [[72, 12, 6]] and
    [[144, 12, 12]] codes under bit flips with p = 0.05, scaled,
    adaptive and product-sum BP agree on every correction and every
    convergence. The one exception is product-sum on [[144, 12, 12]],
    which differs in 0.3% of shots, where tanh saturates.
  - Exhaustive OSD of order 4 after one BP iteration, on 100 random
    models with equal priors, gives the same correction on every shot.
  - On circuit-level models from Stim, which equal QSO's to 10⁻¹⁵, the
    predictions agree except in 0.1% of shots, where the different OSD
    scores pick different candidates.
- **Speed.** The same section of the cross-validation times both decoders
  from Python, one call per shot, on bit flips with p = 0.05 (scaled
  min-sum, OSD-CS of order 7):

  | Code | QSO (µs/shot) | ldpc (µs/shot) |
  |---|---:|---:|
  | Colour, d = 7 | 16 | 19 |
  | Colour, d = 9 | 38 | 42 |
  | [[72, 12, 6]] | 21 | 32 |
  | [[144, 12, 12]] | 66 | 85 |

Under uniform circuit-level noise with d rounds (20 000 shots,
`qso qec --code color --decoder bposd --distance 3,5,7 --p 0.001,0.002
--shots 20000`):

| p | d = 3 | d = 5 | d = 7 |
|---|---|---|---|
| 0.001 | 1.5·10⁻² | 1.0·10⁻³ | 3.5·10⁻⁴ |
| 0.002 | 3.2·10⁻² | 4.7·10⁻³ | 2.2·10⁻³ |

At d = 3 the rate grows linearly with p, because single faults already
fail: the Steane-code case above, and hook errors of the untuned
schedule. From d = 5 on, it grows faster.

`bposd_decodes_colour_and_bicycle_codes` in `tests/validation.rs` checks
the ordering of the colour codes (d = 3, 5, 7 at p = 0.002) and that the
[[72, 12, 6]] code fails in under 1% of shots at p = 0.001 with 6 rounds
(about 1.5·10⁻³).

## Circuit distance and CNOT schedules

The circuit distance is the fewest faults that flip a logical observable
without firing any detector. A poor order of CNOTs can make it smaller
than the code distance: a fault on an ancilla spreads to several data
qubits ("hook errors").

- **`graphlike_distance(model)`** considers only the errors that flip at
  most two detectors, as Stim's `shortest_graphlike_error` does. It is
  exact. Such errors form a graph with a boundary node, and an
  undetectable set of them has even degree at every detector. It flips an
  observable when it holds an odd number of edges that flip it. The
  lightest such set is a cycle, found by breadth-first search on the
  graph doubled by parity.
- **`circuit_distance(model)`** is exact when every error flips at most
  two detectors. Otherwise the question is that of a code's minimum
  distance, which is NP-hard, and it returns bounds:
  - The upper bound is the lightest logical error found, among the
    graph-like errors and by BP+OSD. BP+OSD runs on the checks with the
    parity of some observables as an extra check that must fire: first
    each observable alone, then random sets with perturbed priors
    (`bposd_trials` per observable).
  - The lower bound comes from an exhaustive search of the sets of
    errors that cancel every detector, by increasing weight. It
    branches on the errors that flip the lowest fired detector and stops
    at `max_weight` or after `max_work` sets.

  When the bounds meet, `distance` is set. The witness lists the errors,
  as indices into the model. When nothing is found, linear algebra over
  GF(2) decides (up to 4096 detectors) whether any undetectable logical
  error exists at all; only then is the lower bound "none" (`usize::MAX`
  in Rust, `math.inf` in Python). BP+OSD needs fewer than 16 384
  detectors and is skipped beyond.
- **`exact_circuit_distance(model, max_seconds=60)`** computes the
  distance itself, for any model, when the time allows. It starts from
  the same upper bound and raises the lower bound step by step, each
  step by whichever of two methods it estimates to cost less:
  - the exhaustive search above, for the next weight. It is quick when
    every error flips few detectors and every detector is flipped by few
    errors, as in circuits.
  - a level of the Brouwer–Zimmermann enumeration, as described by
    Grassl (2006). The undetectable sets of errors form a linear code. Its
    words are enumerated as sums of `w` rows of generator matrices that
    are systematic on disjoint sets of columns, and after level `w` no
    word lighter than a bound that grows with `w` remains unseen. It is
    quick when that code is small compared with the number of errors, as
    for classical codes.

  `method="search"` or `"brouwer_zimmermann"` forces one of them. After
  `max_seconds` (`None`: no limit) the bounds reached so far are
  returned, and `distance` is set only if they met.
- **`explain_errors(circuit, model, indices)`** names the circuit faults
  behind errors of the model: for each error, every noise operation and
  Pauli error (or flipped measurement) with exactly its effect. With the
  witness of a distance it places the lightest logical error in the
  circuit.

```python
import metatron_qso as q

model = q.surface_code_memory(5, 5, 0.001).detector_error_model()
assert q.circuit_distance(model).distance == 5
code, _ = q.CssCode.bivariate_bicycle_code(72)
for schedule, d in [("bicycle", 6), ("greedy", 4)]:
    c = q.css_code_memory(code, 3, 0.001, schedule=schedule)
    assert q.circuit_distance(c.detector_error_model()).distance == d

# Hook errors cut the colour code's distance from 5 to 3. Where are they?
c = q.css_code_memory(q.CssCode.color_code(5), 2, 0.001)
model = c.detector_error_model()
r = q.exact_circuit_distance(model, max_seconds=10)
assert r.distance == 3
for faults in q.explain_errors(c, model, r.witness):
    print(faults[0]["operation"], faults[0]["paulis"])
```

`css_code_memory(..., schedule=...)` chooses the order of the CNOTs:

- **`"greedy"`** (the default): every X check, then every Z check, in
  layers found greedily.
- **`"bicycle"`**: the depth-8 cycle of Bravyi et al. (table 5) for a
  bivariate bicycle code with three monomials in each polynomial. X and
  Z checks are interleaved over seven CNOT layers.
- **Six directions**, for a colour code: every face visits its qubits in
  this order of the directions around its centre, one direction per
  layer.

The results, with uniform circuit noise (three rounds for the bicycle
codes, two for the colour codes):

| Code | Schedule | Circuit distance | Bravyi et al. |
|---|---|---|---|
| [[72, 12, 6]] | depth-8 cycle | 6 (exact) | ≤ 6 |
| [[72, 12, 6]] | greedy | 4 (exact) | |
| [[90, 8, 10]] | depth-8 cycle | 8 (exact) | ≤ 8 |
| [[108, 8, 10]] | depth-8 cycle | 8 (exact) | ≤ 8 |
| [[144, 12, 12]] | depth-8 cycle | 9 to 11 | ≤ 10 |
| Colour code, d = 3, 5 | greedy, or any of the 720 orders of directions | (d + 1)/2 (exact) | |
| Colour code, d = 7 | greedy, or the order 0, 1, 2, 3, 4, 5 | 4 (exact) | |

- **Bicycle codes.** `exact_circuit_distance` gives [[90, 8, 10]] and
  [[108, 8, 10]] their circuit distances in 12 s and 97 s on four
  cores. For [[144, 12, 12]] it rules out every logical error of up to 8
  faults in 20 minutes (a measurement; the weekly test checks 7, with a
  fixed amount of work), and 256 BP+OSD searches per observable find one
  of 11. The paper's bounds hold under its own noise model, which also
  strikes idle qubits; QSO's has no idle errors.
- **Colour codes.** No order of directions keeps the distance: all 720
  give (d + 1)/2 for d = 3 and 5. With one bare ancilla per face, a
  fault halfway through a weight-6 check spreads to three data qubits. In
  the Steane code every pair of qubits lies on a logical operator of
  weight 3, so any hook of two qubits already loses one. Keeping the
  distance needs flag qubits or other extraction circuits.

The checks:

- Exact results equal brute-force enumeration on 600 random models, both
  graph-like and with hyperedges.
- `exact_circuit_distance` equals enumeration on 300 random models with
  each method. Level by level, Brouwer–Zimmermann meets every logical
  word lighter than its bound on over 1400 random codes of high rate,
  over 1300 of them with deficient information sets.
- Brouwer–Zimmermann gives the Golay codes [23, 12, 7] and [24, 12, 8]
  and the Reed–Muller codes [32, 6, 16], [32, 16, 8], [64, 22, 16] and
  [64, 42, 8] their minimum distances; so does the automatic choice for
  the Golay codes and [64, 22, 16], and the search alone for
  [23, 12, 7]. The automatic choice gives the code-capacity models of
  the d = 7 colour code, [[72, 12, 6]] and [[144, 12, 12]] theirs.
- Under a work limit, `circuit_distance` returns the same bounds and
  witness on one to four threads.
- The repetition and surface codes keep their distance d.
- `circuit_distances_of_memory_experiments` in `tests/validation.rs`
  checks the table above, by both functions.
- `bicycle_cycle_reaches_the_published_circuit_distances`, a long test
  run weekly, checks the bicycle distances and bounds and all 720
  colour-code orders.
- The cross-validation compares Stim's generated circuits with
  `shortest_graphlike_error` (equal) and with
  `search_for_undetectable_logical_errors` (QSO's exact distance is never
  larger). Without truncation Stim's search is exhaustive. Its distance
  then equals `exact_circuit_distance`'s on nine circuits, colour codes
  with hyperedges and [[72, 12, 6]] included.

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
noiseless circuit. As in Stim, noise needs its probability, gates take
no arguments, and a REPEAT block repeats at least once. Any other instruction is an error that names it and
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
qso qec --code color --distance 5,7 --decoder bposd --p 0.002
qso qec --code bb --distance 72,144 --decoder bposd --p 0.001 --shots 2000
qso qec --code bb --distance 72 --rounds 3 --decoder bposd --schedule bicycle --circuit-distance
qso qec --circuit surface.stim --decoder correlated-matching
```

For `--code bb`, `--distance` names the code by its qubits.
`--schedule` takes `greedy`, `bicycle` or six directions (`0,1,2,3,4,5`,
colour codes only); it is refused with `--circuit`. `--circuit-distance` adds the bounds on the circuit
distance to each line and to the JSON output.

## Limits and costs

| Quantity | Limit |
|---|---|
| Qubits of a detector circuit | 65 536 |
| Operations | 2²⁴ |
| Observables | 64 |
| Results of one `sample` call | shots × (detectors + 64) ≤ 2³³ bits (1 GiB); in Python, shots × (detectors + observables) ≤ 2³⁰ |
| Error components of a model | 2²²; their detector lists before merging, 2²⁷ entries (1 GiB); propagation, 2³⁴ word operations |
| Code distance, rounds | 101, 10 000, and at most 2²⁴ operations (checked before building) |
| Imported Stim circuits | 65 536 qubits; 2²⁴ operations and annotations after expanding REPEAT blocks (checked before expanding); blocks nested at most 64 deep |
| Matching decoder (`MatchingDecoder`) | cached distances 2²¹ entries (32 MiB), bounded searches beyond; at most 2²² pairs of events per shot |
| Sparse matching decoder | no limit on detection events; time grows faster than linearly where blossoms nest deeply (51 ms for 2001 events in one chain) |
| BP+OSD decoder | 16 384 detectors; OSD keeps a dense basis and its combinations, about 2 · (detectors)² bits (64 MiB at the limit), and takes time in proportion to errors × rank × detectors / 64 per shot that BP does not settle |
| CSS codes | 8192 qubits (the logical operators cost about n³/64 word operations); bivariate bicycle codes l · m ≤ 4096 |
| Detector error models built directly | 2²⁴ detectors, 64 observables |
| Circuit distance | exact and polynomial for graph-like models (a breadth-first search per start node, bounded by the best cycle found); otherwise BP+OSD searches (each like one decoding) and an exhaustive search of at most `max_work` sets (2²⁴ by default; some 10⁷ a second per thread). `exact_circuit_distance` runs until `max_seconds` (its polynomial preliminaries aside); the generator matrices of the Brouwer–Zimmermann enumeration take at most 2³¹ bits (256 MiB) together, on information sets whose rank falls at most 64 short of the dimension, and while the first is built the detectors' matrix takes up to as much again |

- **Sampling.** Each operation costs one word operation per 64 shots.
- **Error model.** The model takes one pass through the circuit per 4096
  error components, or more when the circuit has more than 4096
  measurements (each pass holds its records).
- **Matching.** With `MatchingDecoder`, a shot with k detection events
  costs k distance lookups (or bounded searches) and O(k³) for the
  blossom algorithm. `SparseMatchingDecoder` costs time in proportion to
  the events and the area their regions cover (see the table above).
- **Example timing.** On one core, a distance-21 surface code with 21
  rounds (881 qubits, 4840 detectors) needs about 0.5 s for its error
  model and 15 s for 100 000 shots decoded by union-find at p = 0.001. The error model
  of d = 41 with 41 rounds (35 280 detectors) takes 30 s.
