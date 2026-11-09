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

Erasures whose place a herald reveals (Stim's `HERALDED_ERASE`) join the
noise, and the decoders then condition on the heralds that fired (see
[Erasures with heralds](#erasures-with-heralds)).

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

## Decoding in windows

An experiment of thousands of rounds, or one that runs while it is
decoded, is decoded a window of rounds at a time. `WindowDecoder(model,
layers, window, commit)` takes a model and the layer (round) of every
detector, `DetectorCircuit.detector_layers()` for a circuit: detectors
declared one after another form a layer, as every round of the memory
experiments declares its detectors after its measurements. A circuit that
declares a round's detectors in several blocks (or one per operation)
has several layers per round; windows count layers.

A window covers `window` layers. It is decoded on the errors whose first
detector lies in it, with their detectors beyond the window dropped (the
window's future end acts as a boundary), and without the errors that
reach back before it. The part of the correction that touches its first
`commit` layers is applied: its observables are added to the prediction
and its detection events flipped, also those in the rest of the window
(the buffer) and beyond, which the next window sees. The next window
starts `commit` layers later; the last one commits everything. This is
the sliding window of overlapping recovery (Dennis et al., J. Math. Phys.
43, 4452 (2002)) in the commit-and-buffer form of Skoric et al. (Nat.
Commun. 14, 7040 (2023)). `decoder="matching"` takes a decomposed model
and commits matched pairs of events; `decoder="bposd"` commits errors of
the BP+OSD correction. Windows of the same shape, as in the bulk of a
memory experiment, share one decoder.

```python
import metatron_qso as q

c = q.surface_code_memory(5, 200, 5e-3)
model = c.detector_error_model(decompose=True)
w = q.WindowDecoder(model, c.detector_layers(), window=10, commit=5)
print(w.n_windows, w.n_decoders)          # 40 windows, 3 decoders
detectors, observables = c.sample(1000, seed=1)
wrong = (w.decode_batch(detectors) != observables[:, 0]).sum()

# Round by round, as on hardware: push_layer returns what is committed.
fired = [int(d) for d in detectors[0].nonzero()[0]]
layers = c.detector_layers()
for layer in range(w.n_layers):
    w.push_layer([d for d in fired if layers[d] == layer])
assert w.finish() == w.decode(fired)
```

`logical_error_rate(..., decoder="window_matching" | "window_bposd",
window=w, commit=c)` and `qso qec --window W --commit C` decode the same
shots in windows. On a distance-5 surface code over 40 rounds at
p = 5·10⁻³, 20 000 shots fail 7829 times in windows of one round (no
buffer), 2646 times in windows of four committing two, 2206 in windows
of ten committing five and 2197 in windows of twenty, as decoding the
whole experiment does. The time per round stays at 1.1 to 1.5 µs per
shot from 100 to 4000 rounds. On the [[72, 12, 6]] bicycle code over 48
rounds, BP+OSD in windows of six rounds committing three took 59 s for
2000 shots (165 failures) against 76 s on the whole model (196); its
time grows in proportion to the rounds. The error model is built for
the whole experiment, in one pass backwards through the circuit (0.26 s
for 1000 rounds of distance 5, 1.05 s for 4000). Where a window finds no
correction, because the only way out of a carried defect leads back
(as when the only boundary is an error at the start), a fallback with
the errors that reach before the window decodes it.

## Long experiments: repeated blocks and parallel windows

A flat `DetectorCircuit` holds 2²⁴ operations (about 97 000 rounds of a
distance-5 surface code), and the memory experiments build at most
10 000 rounds flat. A `RepeatedCircuit` keeps repeated blocks as
Stim's `REPEAT` does: `RepeatedCircuit.from_stim(text)` reads Stim's
format without expanding its blocks, and `surface_code_memory_repeated`,
`repetition_code_memory_repeated` and `css_code_memory_repeated` build
the memory experiments with rounds 2 to n as one block, for up to 10⁷
rounds; `flatten()` expands one into the flat circuit (where it fits),
`to_stim()` writes it with its blocks. Its records count back from the
operation that reads them (`rec[-k]`), so every iteration reads the
same relative records. `sample` runs the operations in order and keeps
only the records a detector can still reach, with the same shots as
the expanded circuit for the same seed as long as both take the same
batches of shots (1024, unless the records and detectors of a batch
pass 128 MiB).

The detector error model comes from one pass backwards through the
circuit, which carries for every qubit the detectors and observables an
X or a Z there would flip. Through a block, once the state at the start
of an iteration is its state at the end with every detector and record
shifted by one iteration's, each earlier iteration repeats the last
one's errors shifted, and the pass jumps to the start of the block (as
Stim folds loops). A memory experiment repeats after two or three
rounds. The model equals the expanded circuit's term by term, and
Stim's for its generated circuits of 1000 to 20 000 rounds (relative
differences 3·10⁻¹⁶). A block whose iterations have no detectors adds
its errors' j iterations at once; a block without operations is left
out.

`WindowDecoder.parallel(model, layers, commit, buffer)` decodes in two
passes, as Skoric et al. propose: windows of `buffer + commit + buffer`
layers, each decoded on the detection events as sampled with both ends
as boundaries and committing its middle `commit` layers, run in
parallel; then the seams between their commit regions, which take the
detection events the first pass left and commit everything, also in
parallel. Between two commit regions lie `buffer + span` layers (`span`
the most layers one error spans), so that no window of the first pass
changes what another commits. A stream (`push_layer`) decodes the same
windows in order with the same prediction. `logical_error_rate(...,
decoder="parallel_window_matching" | "parallel_window_bposd",
commit=c, buffer=b)` and `qso qec --parallel --commit C --buffer B`
decode in parallel windows; `logical_error_rate` takes a
`RepeatedCircuit` as well.

```python
import metatron_qso as q

# Ten thousand rounds of a distance-5 surface code: the round is stored
# once, and the model folds once the rounds repeat.
c = q.surface_code_memory_repeated(5, 10_000, 1e-3)
print(c.n_stored_operations, c.n_operations)   # 432 of 1 720 088
model = c.detector_error_model(decompose=True)
print(len(model.errors))                        # 560 021, in under a second

# Where the expanded circuit fits, the models agree term by term.
short = q.surface_code_memory_repeated(3, 200, 1e-3)
flat = q.surface_code_memory(3, 200, 1e-3)
pairs = zip(short.detector_error_model().errors, flat.detector_error_model().errors)
assert all(a[1:] == b[1:] and abs(a[0] - b[0]) <= 1e-13 * b[0] for a, b in pairs)

# Stim's REPEAT blocks are kept, and written back.
text = "R 0 1 2\nREPEAT 1000 {\n    X_ERROR(0.01) 0 1\n    CX 0 2 1 2\n    MR 2\n    DETECTOR rec[-1]\n}\n"
r = q.RepeatedCircuit.from_stim(text)
assert r.n_detectors == 1000 and "REPEAT 1000 {" in r.to_stim()

# Parallel windows: commit five rounds, five of buffer on each side.
rate = q.logical_error_rate(c, 256, seed=1, decoder="parallel_window_matching", commit=5)
per_round = (1 - (1 - 2 * rate.rate) ** (1 / 10_000)) / 2
print(rate.errors, f"{per_round:.1e}")           # 66 of 256: 3.6e-05 per round
```

At distance 5 and p = 10⁻³ (four cores, release build) the model of
10⁴ rounds takes 0.15 s and that of 10⁵ rounds 1.4 s (5.6 million
errors, about 350 MB); sampling costs 9 µs per round for a batch of up
to 64 shots. Sliding windows of ten rounds decode 0.18 to 0.20 µs per
round and shot at 10⁴ and 10⁵ rounds. Parallel windows committing five
with buffers of five take 0.19, 0.14 and 0.08 µs on one, two and four
threads at 10⁴ rounds (2.4 times faster on four) and 0.18, 0.11 and
0.09 µs at 10⁵ rounds (2.0 times), with the same failures; a shot's
syndrome and the commits are applied on one thread.

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

  With `symmetry=True` the search starts only from one error of each
  orbit of the model's symmetries, which divides its work by a factor that
  grows with their number (41 to 43 for the 72 translations of
  [[144, 12, 12]]; see below). Models with more than 4096 detectors are
  searched without them.
- **`error_model_symmetries(model)`** finds the symmetries of a model:
  permutations of its detectors that map the detectors of every error
  onto those of an error and logical errors onto logical errors, as
  lists `p` with detector `d` going to `p[d]`. QSO's automorphism search
  finds generators of the symmetries of the graph that joins every error
  to the detectors it flips; those that pass a check over GF(2) are kept.
  The check asks that every observable, read through the permutation, be
  a sum of detectors and observables. Exactly then does every set of
  errors that fires no detector and flips no observable map onto such a
  set. The generators kept generate a group of symmetries, though not
  always all of them: a symmetry that is a product of generators that
  fail is lost with them. `exact_circuit_distance(model,
  symmetries=[...])` takes declared permutations too (with or without
  `symmetry=True`), checks them the same way and refuses those that
  fail.
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
| [[144, 12, 12]] | depth-8 cycle | 10 (exact, on the orbits) | ≤ 10 |
| Colour code, d = 3, 5 | greedy, or any of the 720 orders of directions | (d + 1)/2 (exact) | |
| Colour code, d = 7 | greedy, or the order 0, 1, 2, 3, 4, 5 | 4 (exact) | |

- **Bicycle codes.** `exact_circuit_distance` gives [[90, 8, 10]] and
  [[108, 8, 10]] their circuit distances in 12 s and 97 s on four
  cores, and in 0.9 s and 2.5 s with `symmetry=True`. The depth-8 cycle keeps
  the code's translations of the torus (72 for [[144, 12, 12]]), so each
  orbit holds one error of each kind at each place. For [[144, 12, 12]]
  the search without them rules out logical errors of 8 faults in four
  minutes, but not of 9 in twenty (each weight costs about 17 times the
  last); on the orbits it rules out 9 in 93 s and finds a logical error
  of 10 faults, in 12 minutes in all. The paper's bounds hold under its
  own noise model, which also strikes idle qubits; QSO's has no idle
  errors, and still has a logical error of 10 faults.
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
- On the orbits of symmetries, `exact_circuit_distance` equals
  enumeration on 900 random models made of the translations of random
  errors (over 700 of them with symmetries; some observables break
  them) and the search without symmetries on 150 larger ones, and gives
  the same witness on one to four threads. The bicycle codes' circuits
  and code-capacity models, the repetition, surface and colour codes
  keep their distances. Declared permutations that are not symmetries
  are refused: a swap of two detectors, and the swap of two triangles of
  errors that maps a harmless cycle onto a logical one.
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
  with hyperedges and [[72, 12, 6]] included, with and without
  symmetries.

## Failure rates below sampling

Sampling needs about a hundred failures for a useful estimate, 10¹⁰
shots at a failure rate of 10⁻⁸. Resource estimates need rates of 10⁻⁸
to 10⁻¹², so QSO carries the rate down from where sampling reaches.

- **`splitting_failure_rate(circuit_at, start, target)`** follows Bravyi
  and Vargo (Phys. Rev. A 88, 062308 (2013)). `circuit_at(p)` builds the
  circuit at noise strength `p`; its detector error model (without
  decomposition) must have the same errors at every `p`, only their
  probabilities change. At `start` the failure rate is sampled directly
  (`start_shots`), and the failing configurations found start the chains.
  The rates then fall by a factor `e^{−1/σ}`, where `σ` is the spread of
  the number of errors the chains sampled (at most a factor of 4), down
  to `target`. At each rate, independent Metropolis chains (`chains`,
  run in parallel) sample the configurations the decoder of that rate
  fails on, each with its probability `Π p_i Π (1 − p_j)`. A move removes
  an error, adds one, or swaps one for a neighbour (an error sharing a
  detector), with the Hastings factors of these choices. The ratio of
  consecutive failure rates comes from the chains of both rates by
  Bennett's acceptance ratio. The chains of one rate enter two
  consecutive ratios, so the standard error comes from each chain's
  influence on the whole product (its deviations in both ratios added,
  then the spread over the chains). The result's error is relative
  (`log_standard_error`).
- **Burn-in matters.** Each chain starts from a configuration of the last
  rate, heavier than the new rate's typical ones; too short a burn-in
  biases the ratio downwards. With 5 000 steps after 1 000 of burn-in,
  eight seeds of the surface code of distance 5 at `p = 10⁻⁴` gave 1.17
  to 1.39·10⁻⁷, a spread of 5.7 % against 4.3 % stated. The defaults
  are 16 chains of 20 000 steps after 5 000 of burn-in; with them the
  eight seeds gave 1.34 to 1.41·10⁻⁷, a spread of 1.7 % against 2.3 %
  stated.
- **`leading_failures(circuit)`** gives the leading term: the fewest
  errors that fail, the failing configurations of that weight, their
  summed product of probabilities (`leading_term`, the `A·p^w` of the
  failure rate) and their exact share of it (`probability`, a lower
  bound). When the configurations of at most `⌈d/2⌉` errors number at
  most `max_configurations`, every one is decoded and the result is exact
  (`exhaustive`). Otherwise the subsets of `⌈d/2⌉` errors of every
  undetectable logical error of at most `2⌈d/2⌉` errors are decoded. That
  count is complete only when the decoder never corrects with more errors
  than the configuration has. BP+OSD, and matching with the unequal
  probabilities of circuit noise, can correct with more. Then the subsets
  give a lower bound: under BP+OSD they miss failing configurations of
  [[72, 12, 6]]. Fewer errors can then also fail: two, not three, for a
  repetition code of distance 5 whose measurements err 5 000 times more
  often than its gates.
- **`enumerated_failure_rate(circuit, max_weight)`** decodes every
  configuration of at most `max_weight` errors. It also bounds the
  probability of the heavier ones (`remainder`), so that the exact rate
  lies in `[rate, rate + remainder]`. This is the reference for small
  models.

```python
import metatron_qso as q

at = lambda p: q.surface_code_memory(5, 5, p)
r = q.splitting_failure_rate(at, 0.01, 1e-4)
print(r["rate"], r["log_standard_error"])   # 1.3e-7, 2 %
lead = q.leading_failures(at(1e-4))
print(lead["configurations"], lead["probability"])   # 26384 of 3 errors, 1.3e-7
```

Measured on four cores, with circuit noise (rounds = d) and matching,
from `start = 0.01`:

| Code | p | Splitting | Time | Check |
|---|---|---|---|---|
| surface d = 3 | 10⁻³ | 7.93·10⁻⁴ ± 1.4 % | 0.4 s | sampled: 8.13·10⁻⁴ ± 2.5 % |
| surface d = 3 | 10⁻⁴ | 8.12·10⁻⁶ ± 1.4 % | 0.6 s | exact: 8.28·10⁻⁶ (the test's enumeration) |
| surface d = 5 | 10⁻³ | 1.31·10⁻⁴ ± 2.2 % | 4 s | sampled: 1.26·10⁻⁴ ± 6.3 % |
| surface d = 5 | 10⁻⁴ | 1.34·10⁻⁷ ± 2.5 % | 4 s | weight-3 failures alone: at least 1.30·10⁻⁷ |
| surface d = 7 | 10⁻³ | 1.81·10⁻⁵ ± 3.5 % | 18 s | |
| surface d = 7 | 10⁻⁴ | 1.76·10⁻⁹ ± 4.0 % | 20 s | |
| [[72, 12, 6]], data only, BP+OSD | 10⁻² | 4.52·10⁻⁴ ± 3.0 % | 20 s | sampled: 5.2·10⁻⁴ ± 9.8 % |
| [[72, 12, 6]], data only, BP+OSD | 10⁻³ | 4.41·10⁻⁷ ± 4.2 % | 20 s | exact: 4.484 to 4.501·10⁻⁷ |
| [[72, 12, 6]], data only, BP+OSD | 10⁻⁵ | 4.69·10⁻¹³ ± 8.7 % | 26 s | exact: 4.498·10⁻¹³ |

The bicycle code starts from 0.06, with 20 000 start shots; its exact
rates are enumerations of all configurations of up to four errors (5 s
each).

## Stim circuits and the command line

`to_stim()` writes a circuit in Stim's text format, with relative
measurement records (`rec[-k]`) and exact probabilities.
`DetectorCircuit.from_stim(text)` reads one, expanding its REPEAT
blocks; `RepeatedCircuit.from_stim(text)` keeps them (see above). Both
understand the following instructions:

- the Clifford gates H, S, S_DAG, X, Y, Z, SQRT_X(_DAG), SQRT_Y(_DAG), CX,
  CY, CZ, XCZ, YCZ and SWAP;
- measurements and resets in the X, Y and Z bases (M, MX, MY, R, RX, RY,
  MR, MRX, MRY), with an optional flip probability;
- the noise channels X_ERROR, Y_ERROR, Z_ERROR, DEPOLARIZE1 and
  DEPOLARIZE2;
- the heralded channels HERALDED_ERASE and HERALDED_PAULI_CHANNEL_1
  (see [Erasures with heralds](#erasures-with-heralds));
- DETECTOR, OBSERVABLE_INCLUDE and REPEAT blocks.

TICK and coordinates are ignored. Inverted measurement targets (`!q`)
read the same as plain ones, since detection events are relative to the
noiseless circuit. As in Stim, noise needs its probability, gates take
no arguments, and a REPEAT block repeats at least once. Any other instruction is an error that names it and
its line: classical feedback, MPP, correlated noise and the Pauli
channels PAULI_CHANNEL_1 and PAULI_CHANNEL_2.

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
qso qec --code bb --distance 144 --rounds 3 --decoder bposd --schedule bicycle --shots 1000 \
    --circuit-distance --symmetry --max-seconds 3600
qso qec --circuit surface.stim --decoder correlated-matching
qso qec --distance 3,5,7 --p 0.02 --erasure-share 0.98 --decoder matching
qso qec --distance 5 --rounds 100000 --p 0.001 --shots 100 --decoder matching \
    --parallel --commit 5 --buffer 5
```

For `--code bb`, `--distance` names the code by its qubits.
`--schedule` takes `greedy`, `bicycle` or six directions (`0,1,2,3,4,5`,
colour codes only); it is refused with `--circuit`. `--circuit-distance` adds the bounds on the circuit
distance to each line and to the JSON output; `--symmetry` searches it
on the orbits of the circuit's symmetries. Beyond 10 000 rounds a
memory experiment is built with its rounds as one repeated block, and a
Stim file too long to expand is read with its REPEAT blocks kept.
`--parallel` decodes in parallel windows committing `--commit` rounds
with `--buffer` (default `--commit`) on each side. `--erasure-share e` makes the share e of
the noise heralded erasures (see [Erasures with
heralds](#erasures-with-heralds)); the decoders then condition on the
heralds, and each JSON row carries `erasure_share` and
`herald_detectors`.

## Erasures with heralds

Neutral atoms with erasure conversion and dual-rail superconducting
qubits turn most of their errors into *erasures* whose place is
heralded: a separate measurement tells which qubit was struck, and a
decoder told so tolerates several times more error. QSO takes Stim's
model of them, in which nothing leaves the qubit space (leakage is not
modelled):

- `heralded_erase(p, qubits)` is Stim's `HERALDED_ERASE(p)`: for each
  target in turn, with probability p the qubit is replaced by a maximally
  mixed one (I, X, Y or Z, each with probability p/4) and its *herald*, a
  new measurement record, reads 1; otherwise the herald reads 0;
- `heralded_pauli_channel_1(p_i, p_x, p_y, p_z, qubits)` is Stim's
  `HERALDED_PAULI_CHANNEL_1`: for each target, disjointly, the herald
  fires with I, X, Y or Z with these probabilities (Stim's argument
  order), and with 1 − Σ neither happens.

Both return the herald's record of each target, which detectors and
observables read like any other record; heralds are never flipped. They
count in `n_measurements` (as in Stim's `num_measurements`) and in
`len(c)`, `without_noise()` keeps them at probability 0, and both are
read and written in Stim's format (an inverted target `!q` reads as `q`;
the four probabilities may exceed 1 by at most 10⁻⁷, as in Stim). The
frame sampler draws the herald and the Pauli together, so sampling is
exact.

A detector whose only record is a herald, and which is the only reader
of that herald, is a *herald detector*: a decoder conditions on it.
Heralds that nothing reads, that an observable reads, or that a detector
reads together with other records are noise like any other.

**The error model.** The outcomes of a heralded channel are disjoint,
not independent. `detector_error_model` converts them by Stim's
approximate rule (`approximate_disjoint_errors=True`): outcomes with
equal effects are summed into one error, and the errors are then taken
as independent. `HERALDED_ERASE(0.1)` before a measurement gives two
errors of probability 0.05, one on the herald alone (I or Z) and one on
the herald and the measurement (X or Y):

```python
import metatron_qso as q

c = q.DetectorCircuit(1)
herald = c.heralded_erase(0.1, [0])          # its record, [0]
m = c.measure(0)
c.detector(herald)                            # D0 reads the herald alone
c.detector([m])                               # D1 the measurement
print(c.to_stim())        # HERALDED_ERASE(0.1) 0, M 0, two DETECTORs
print(c.detector_error_model(decompose=False).errors)
# [(0.05, [0], 0), (0.05, [0, 1], 0)]

model = c.heralded_error_model()
print(model.herald_detectors, model.herald_probability(0), model.outcomes(0))
# [0] 0.1 [(0.5, [1], 0)]: given its herald, D1 flips with probability ½
print(model.conditioned([0]).errors, model.conditioned([]).errors)
# [(0.5, [1], 0)] []
```

Taken as independent, those two errors fire the herald with probability
0.095, not 0.1. Sampling is exact, but anything that reads the model's
errors as independent configurations is not: `splitting_failure_rate`,
`enumerated_failure_rate`, `leading_failures` and `explain_errors`
refuse circuits with heralded operations ("does not take heralded
operations yet"), and so does `qso qec --circuit-distance`.
`circuit_distance` and `graphlike_distance` take a model, in which each
outcome of a heralded channel counts as one error, as in Stim; one whose
herald a detector reads is never undetectable.

**The heralded model.** `heralded_error_model(decompose=True)` splits
the model for decoding: its `base` holds the errors of every channel
whose herald no herald detector owns (decomposed as by
`detector_error_model`), and for each herald detector it holds the
probability that it fires and the parts of its channel given that it
fired: the X part with probability (p_x + p_y)/t and the Z part with
(p_z + p_y)/t, t the sum of the four, each exactly ½ for an erasure, the
herald left out. `conditioned(heralds)` gives the model of one shot: the
base with the parts of the fired heralds merged in as independent
errors (p ⊕ q = p(1 − q) + q(1 − p), and ½ ⊕ q = ½ exactly). For the
heralded Pauli channel, treating its X and Z parts as independent and
merging equal parts of different channels is an approximation; for an
erasure both are exact. Where a channel's X and Z have the same effect
(before a Y-basis readout), it has one part, with probability
(p_x + p_z)/t, exactly. The decomposed model, the default, is the one
union-find and matching need; BP+OSD decodes the model without
decomposition (`heralded_error_model(decompose=False)`, the only one a
colour code has) and refuses a decomposed one. `model.decomposed` says
which a model is, and `ConditionedDecoder.for_circuit(circuit, decoder)`
builds the right one for the decoder. Read from Stim's text:

```python
import metatron_qso as q

text = """H 0
HERALDED_PAULI_CHANNEL_1(0.01, 0.02, 0.03, 0.04) !0
H 0
M 0
DETECTOR rec[-2]
DETECTOR rec[-1]
"""
c = q.DetectorCircuit.from_stim(text)
print(c.detector_error_model(decompose=False).errors)
# [(0.03, [0], 0), (0.07, [0, 1], 0)], as Stim 1.16 gives it
m = c.heralded_error_model()
print(m.herald_probability(0), m.outcomes(0))   # 0.1 [(0.7…, [1], 0)]: (0.04 + 0.03)/0.1
```

**Decoding with heralds.** `ConditionedDecoder.for_circuit(circuit,
decoder=...)` builds the heralded model the decoder needs and the
decoder on it (`ConditionedDecoder(model, decoder=...)` takes a model
built before); it merges the parts of the fired heralds into each shot's model and then
decodes on the other detectors:

- `"union_find"` joins the erased edges (probability ½) into clusters
  before it grows them (Delfosse and Nickerson, Quantum 5, 595 (2021));
- `"matching"` sets the weights ln((1 − p)/p) of the shot's edges (0 for
  an erased one) in the sparse matching decoder and restores them after;
- `"bposd"` sets the priors of BP+OSD (with its options as keyword
  arguments).

Edges that only a herald turns on stay in the graph, never used until it
fires. A part can be more likely than not given its herald (the Z part
above, 0.7): the decoder applies it first, adding its detectors to the
shot's and its observables to the prediction, and decodes the rest with
1 − p, as PyMatching does with a negative weight ln((1 − p)/p). `decode(fired, heralds)` takes the fired ordinary and herald
detectors apart, `decode(fired)` and `decode_batch(detectors)` take them
together. `logical_error_rate` and `qso qec` route a circuit with herald
detectors to it for `union_find`, `matching` and `bposd`; correlated
matching and the window decoders refuse such circuits ("does not use
heralds yet"), and the exact `MatchingDecoder` is not conditioned. Union-find
and matching need every base error and every herald part to flip at
most two ordinary detectors: so it is in QSO's repetition and rotated
surface-code memories, while in colour codes a single X flips three, and
BP+OSD decodes them, on the model without decomposition, which
`for_circuit` builds for it (the decomposed model refuses such errors).

A repetition code whose d data qubits are each erased with probability
p and read out perfectly fails exactly when all are erased and the
flips leave the wrong majority, with probability p^d/2. Enumerating every
erased set and every pattern of flips on it:

```python
import itertools
import metatron_qso as q

d, p = 5, 0.1
c = q.DetectorCircuit(d)
heralds = c.heralded_erase(p, list(range(d)))
data = [c.measure(k) for k in range(d)]
for h in heralds:
    c.detector([h])                           # D0..D4: the heralds
for k in range(d - 1):
    c.detector([data[k], data[k + 1]])        # D5..D8: Z_k Z_k+1
c.observable(0, [data[0]])
for method in ["union_find", "matching", "bposd"]:
    dec = q.ConditionedDecoder.for_circuit(c, decoder=method)
    fail = 0.0
    for erased in itertools.product([0, 1], repeat=d):
        E = [k for k in range(d) if erased[k]]
        for flips in itertools.product([0, 1], repeat=len(E)):
            x = dict(zip(E, flips))
            fired = [d + k for k in range(d - 1) if x.get(k, 0) != x.get(k + 1, 0)]
            if dec.decode(fired, heralds=E) != x.get(0, 0):
                fail += p ** len(E) * (1 - p) ** (d - len(E)) / 2 ** len(E)
    print(method, fail, p**d / 2)             # 5e-06 for each, p^d/2
```

At code capacity, the rotated surface code whose data qubits are erased
with probability ½ fails with probability exactly ¼ at every odd
distance: a quarter turn maps it onto itself with X and Z exchanged, and
an erased set supports the X logical exactly when its complement does
not support the Z logical, so half of all sets support it, and on those
the decoder can only guess. The tests enumerate all 3⁹ cases of d = 3
(exactly ¼ for union-find, matching and BP+OSD with OSD-0 and OSD-CS)
and sample d = 5 and 7 with 20 000 shots each, the cross-validation
d = 3, 5, 7 and 9.

**Erasure-biased circuit noise.** `ErasureNoise(noise, erasure_share=e)`
(`noise` a `CircuitNoise` or a probability) makes the share e of every
noise probability p of a memory experiment a heralded erasure of each
qubit of the operation, `HERALDED_ERASE(e·p)`, after resets and gates,
before measurements and on the data before each round, and leaves
(1 − e)·p to the circuit noise as before. Each round declares the
heralds of its operations as herald detectors just before its own
detectors (the final block those of the data measurement). With e = 0
the circuit equals the plain one; with e = 1 no Pauli noise is left. A
two-qubit gate erases each of its qubits with e·p independently, so its
total error is about (1 + e)·p; the pair erasure of Wu et al. has no
Stim instruction. `repetition_code_memory`, `surface_code_memory` and
`css_code_memory` and their `_repeated` forms accept it, and so does
`qso qec --erasure-share e`:

```python
import metatron_qso as q

noise = q.ErasureNoise(0.01, erasure_share=0.9)
c = q.surface_code_memory(5, 5, noise)
model = c.heralded_error_model()
print(c.n_detectors, len(model.herald_detectors))   # 1431 1359
for decoder in ["union_find", "matching", "bposd"]:
    r = q.logical_error_rate(c, 10_000, seed=1, decoder=decoder)
    print(decoder, r.errors)                         # 27 each
plain = q.surface_code_memory(5, 5, 0.01)
print(q.logical_error_rate(plain, 10_000, seed=1, decoder="matching").errors)  # 879

dec = q.ConditionedDecoder(model, decoder="matching")
detectors, observables = c.sample(10_000, seed=2)
predicted = dec.decode_batch(detectors)
print((predicted != observables[:, 0]).mean())   # 0.0014
```

**Measured.** Rotated surface code in the Z basis, `d` rounds, uniform
circuit noise p of which the share e is heralded erasure, 10⁵ shots per
point at d = 3, 5 and 7 (`qso qec --code surface --distance 3,5,7 --p …
--erasure-share e --shots 100000`, p from 0.4 % to 5 %); the threshold
is read off where the rates of two distances cross (logarithmic
interpolation between grid points):

| Erasure share e | 0 | 0.5 | 0.9 | 0.98 | 1 |
|---|---|---|---|---|---|
| Matching: crossing d = 3/5, 5/7 | 0.65 %, 0.75 % | 1.21 %, 1.20 % | 2.72 %, 2.64 % | 3.51 %, 3.67 % | 3.84 %, 3.93 % |
| Union-find: crossing d = 3/5, 5/7 | 0.62 %, 0.71 % | 1.14 %, 1.15 % | 2.67 %, 2.48 % | 3.51 %, 3.63 % | 3.83 %, 3.92 % |
| BP+OSD: crossing d = 3/5 (10⁵ shots, 6 or 7 probabilities around each crossing; ±0.02–0.05 %) | 0.68 % | 1.21 % | 2.74 % | 3.52 % | 3.80 % |
| Matching at p = 1 %, d = 3 / 5 / 7 | 6.0 / 8.3 / 10.3 % | 2.8 / 2.4 / 2.0 % | 0.59 / 0.17 / 0.04 % | 0.19 / 0.02 / 0.002 % | 0.09 / 0.007 % / none in 10⁵ |

The threshold rises about five-fold from e = 0 to e = 0.98, as in Wu et
al.'s model (0.937 % to 4.15 %, Nat. Commun. 13, 4657 (2022)); the
numbers differ because the noise differs: here every qubit of a
two-qubit gate is erased on its own, and the rest of the noise is
QSO's circuit noise. With heralds the decoders get faster, as fewer
edges are uncertain: at p = 1 % and d = 7, 6.2 µs per shot for
union-find, 11.5 µs for matching and 230 µs for BP+OSD at e = 0.9,
against 23, 18 and 557 µs at e = 0 (d = 9: 14.5 and 28 µs against 75
and 48). The heralded model of d = 11 with 11 rounds (15 333 herald
detectors) takes 19 ms. Herald detectors outnumber the others 17 to 21
times (d = 3 to 11), so the sampler draws 1.4·10⁶ shots a second at d = 5 against
3.1·10⁶ without heralds, and a folded model of 5·10⁴ rounds of d = 5
holds 1.3·10⁷ herald detectors (6.1 s, 4 GB).

## Limits and costs

| Quantity | Limit |
|---|---|
| Qubits of a detector circuit | 65 536 |
| Operations | 2²⁴ |
| Observables | 64 |
| Results of one `sample` call | shots × (detectors + 64) ≤ 2³³ bits (1 GiB); in Python, shots × (detectors + observables) ≤ 2³⁰ |
| Error models | the distinct errors' detector lists 2²⁷ entries (1 GiB), merged as the pass goes; the pass backwards, 2³⁴ entries of effects touched (about fifteen seconds); `explain_errors` 2²² error components |
| Window decoder | layers non-decreasing in the detector index; one decoder per distinct window shape, and the window's errors indexed for every window |
| Code distance, rounds | 101, 10 000, and at most 2²⁴ operations (checked before building); repeated memory experiments 10⁷ rounds |
| Repeated circuits | 2²⁴ operations stored, 2³¹ expanded; models 2²⁴ detectors; sampling keeps as many records as a detector reaches back |
| Imported Stim circuits | 65 536 qubits; 2²⁴ operations and annotations after expanding REPEAT blocks (checked before expanding); blocks nested at most 64 deep |
| Matching decoder (`MatchingDecoder`) | cached distances 2²¹ entries (32 MiB), bounded searches beyond; at most 2²² pairs of events per shot |
| Sparse matching decoder | no limit on detection events; time grows faster than linearly where blossoms nest deeply (51 ms for 2001 events in one chain) |
| BP+OSD decoder | 16 384 detectors; OSD keeps a dense basis and its combinations, about 2 · (detectors)² bits (64 MiB at the limit), and takes time in proportion to errors × rank × detectors / 64 per shot that BP does not settle |
| CSS codes | 8192 qubits (the logical operators cost about n³/64 word operations); bivariate bicycle codes l · m ≤ 4096 |
| Detector error models built directly | 2²⁴ detectors, 64 observables |
| Heralded operations | counted among the 2²⁴ operations, a record per target; the four probabilities of `HERALDED_PAULI_CHANNEL_1` add up to at most 1 + 10⁻⁷ (Stim's tolerance) |
| Decoding with heralds | union-find and matching: every base error and herald part flips at most two ordinary detectors (colour codes need BP+OSD, on the model without decomposition); BP+OSD: 16 384 ordinary detectors, herald detectors not counted; correlated matching and windows refuse circuits with herald detectors; splitting, enumeration, explanations and `qso qec --circuit-distance` refuse any heralded operation |
| Circuit distance | exact and polynomial for graph-like models (a breadth-first search per start node, bounded by the best cycle found); otherwise BP+OSD searches (each like one decoding) and an exhaustive search of at most `max_work` sets (2²⁴ by default; some 10⁷ a second per thread). `exact_circuit_distance` runs until `max_seconds` (its polynomial preliminaries aside, the check of symmetries included); symmetries are checked for up to 4096 detectors, by an elimination of about n² · m / 64 word operations for n detectors and m errors, and the automorphism search behind `symmetry` stops at `max_seconds`; the generator matrices of the Brouwer–Zimmermann enumeration take at most 2³¹ bits (256 MiB) together, on information sets whose rank falls at most 64 short of the dimension, and while the first is built the detectors' matrix takes up to as much again |

- **Sampling.** Each operation costs one word operation per 64 shots.
- **Error model.** The model takes one pass backwards through the
  circuit, in time linear in it; a repeated block costs a few
  iterations, and the rest its errors' count.
- **Matching.** With `MatchingDecoder`, a shot with k detection events
  costs k distance lookups (or bounded searches) and O(k³) for the
  blossom algorithm. `SparseMatchingDecoder` costs time in proportion to
  the events and the area their regions cover (see the table above).
- **Example timing.** On one core, a distance-21 surface code with 21
  rounds (881 qubits, 4840 detectors) needs about 0.15 s for its error
  model and 15 s for 100 000 shots decoded by union-find at p = 0.001. The error model
  of d = 41 with 41 rounds (35 280 detectors) takes 1.0 s (30 s before
  the pass backwards).
