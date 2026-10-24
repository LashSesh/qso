# Boson sampling

## Interferometers and permanents

An m-mode linear interferometer is an m × m unitary U. When n photons enter
with occupation S and leave with occupation T, the output probability is a
matrix permanent (Aaronson & Arkhipov 2011):

- for indistinguishable photons, P(T) = |perm(U_{T,S})|² / (Π s! Π t!);
- for distinguishable photons, P(T) = perm(|U_{T,S}|²) / Π t!.

QSO computes permanents with Glynn's formula in Gray-code order, which costs
O(n·2ⁿ) operations. The result is the same for any number of threads. For
n = 24 it takes about 0.14 s on four cores, and it agrees with The Walrus
to 3·10⁻¹⁴.

```python
import numpy as np
import metatron_qso as q

assert q.permanent(np.ones((6, 6))) == 720           # perm(J) = n!
u = q.haar_unitary(8, seed=1)                        # Haar-random (Mezzadri)
assert np.allclose(u.conj().T @ u, np.eye(8), atol=1e-12)
```

## Exact probabilities and sampling

```python
import math
import numpy as np
import metatron_qso as q

# Hong–Ou–Mandel: two photons on a 50:50 beam splitter leave together.
hom = q.BosonSampler(q.beam_splitter(math.pi / 4), [1, 1])
assert hom.probability([1, 1]) < 1e-30
assert abs(hom.probability_distinguishable([1, 1]) - 0.5) < 1e-15

# Four photons in eight modes: full distribution and exact samples.
s = q.BosonSampler(q.haar_unitary(8, seed=1), [1, 1, 1, 1, 0, 0, 0, 0])
patterns, probs = s.distribution()
assert abs(probs.sum() - 1) < 1e-12
samples = s.sample(10_000, seed=2)           # Clifford & Clifford, exact

# Full bunching is enhanced by n! over distinguishable photons.
bunched = [0, 0, 4, 0, 0, 0, 0, 0]
ratio = s.probability(bunched) / s.probability_distinguishable(bunched)
assert abs(ratio - 24) < 1e-9
```

Exact sampling uses the algorithm of Clifford & Clifford (2018) and needs
at most one photon per input mode. The test suite checks the samples
against the exact distribution with a χ² test.

## Limits

| Quantity | Limit |
|---|---|
| Modes | 1024 |
| Photons | 30 |
| Permanent size | 40 × 40 |
| Patterns of a full distribution | 2·10⁶ patterns, 2²⁶ occupation numbers |
| Samples per call | 2²⁶ occupation numbers (shots × modes) |
