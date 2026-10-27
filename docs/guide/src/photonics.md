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

## Imperfect photons

Real photons are not perfectly identical, and some are lost.
`ImperfectBosonSampler` adds both to a `BosonSampler`. Each photon carries
an internal state (arrival time, frequency, polarisation) that the
interferometer does not touch. Only the Gram matrix S_jk = ⟨ψ_j|ψ_k⟩ of
these states matters. It can be given directly (`gram`), from state
vectors (`internal_states`), or as one overlap |⟨ψ_j|ψ_k⟩|² shared by all
pairs (`indistinguishability`). Photons are lost before the
interferometer with probability 1 − `transmission` of their input mode,
and detected with the `efficiency` of their output mode.

The probabilities are exact (Tichy 2015; Shchesnovich 2015). They cost n!
permanents, so partially distinguishable photons are limited to 8. Input
loss is a beam splitter from each input mode to an environment mode;
summing over the photons that end there keeps the internal state of a
photon whose partner in the same mode was lost exact. The tests compare
the probabilities with a simulation in which the internal states are
extra modes of an interferometer coupled to environment modes, with
partially distinguishable photons sharing lossy input modes. The Hong–Ou–Mandel dip follows the overlap:

```python
import math
import metatron_qso as q

hom = q.BosonSampler(q.beam_splitter(math.pi / 4), [1, 1])
for x in [0.0, 0.5, 1.0]:
    s = q.ImperfectBosonSampler(hom, indistinguishability=x)
    print(x, s.probability([1, 1]))               # (1 − x)/2

lossy = q.ImperfectBosonSampler(hom, indistinguishability=0.9,
                                transmission=[0.8, 0.8], efficiency=[0.9, 0.95])
patterns, probs = lossy.distribution()            # 0, 1 and 2 detected photons
print(abs(probs.sum() - 1) < 1e-12)
```

## Gaussian boson sampling

In Gaussian boson sampling, squeezed light enters the interferometer
instead of single photons (Hamilton et al. 2017). `GaussianState` holds
the covariance matrix and means of the quadratures (x₁…x_m, p₁…p_m), with
ħ = 2: the vacuum has covariance I. These are the conventions of The
Walrus and Strawberry Fields. Squeezing, displacement, phase rotation,
interferometers and loss (also into a thermal environment) act on it.

The probability of a photon-number pattern is a loop hafnian.
`probability` uses the formula of Björklund, Gupt and Quesada (2019). For a
pure state the 2N × 2N loop hafnian factorises into the squared modulus of
one over N rows, which is exponentially faster and more accurate. For
mixed states with many photons in few modes it uses Kan's formula over the
repeated rows when that is cheaper; Kan's alternating sum loses digits as
the photon number grows.
`click_probability` gives the probabilities of threshold detectors, which
only tell whether photons arrived; they are torontonians (Quesada et al.
2018). `sample` and `sample_clicks` draw exact samples mode by mode from
the marginal states:

```python
import numpy as np
import metatron_qso as q

state = q.GaussianState(4)
for mode in range(4):
    state.squeeze(mode, 0.5)
state.interferometer(q.haar_unitary(4, seed=3)).loss(2, 0.8)

print(state.mean_photon_numbers().sum())
print(state.probability([1, 1, 0, 0]), state.click_probability([True, True, False, False]))
print(state.sample(5, seed=1))
```

`hafnian`, `hafnian_repeated` and `torontonian` are also available on
their own. The hafnian of an adjacency matrix counts perfect matchings:
272 for the 4-cube and 589 185 for the 5-cube. The tests check closed
forms (squeezed, coherent, thermal and two-mode squeezed states, loss as
binomial thinning, (2n − 1)!! matchings of the complete graph) and brute
force. The weekly cross-validation compares hafnians, torontonians and
probabilities with The Walrus. On a 4-mode table with displacement and
loss, photon-number and click probabilities differ by about 10⁻¹⁵
(absolute). Relative differences grow with the photon number: 1.5·10⁻¹¹
for 20 photons from 10 squeezed modes after a random interferometer (pure),
3·10⁻¹³ for 12 photons from 6 lossy modes (mixed). A mixed state with 20
photons spread over 10 modes needs about 10¹² operations in either
algorithm, which is impractical. Hafnians
agree to 7·10⁻¹² relative up to 20 × 20: summing 2^{n/2} terms of
alternating sign costs digits, while The Walrus, whose algorithm differs,
reaches 10⁻¹⁵ against exact integer hafnians. Torontonians lose digits the
same way, in both codes: at 20 modes the two differ by 10⁻⁵ relative.

## Limits

| Quantity | Limit |
|---|---|
| Modes | 1024 |
| Photons | 30 |
| Permanent size | 40 × 40 |
| Patterns of a full distribution | 2·10⁶ patterns, 2²⁶ occupation numbers |
| Samples per call | 2²⁶ occupation numbers (shots × modes) |
| Partially distinguishable photons | 8 |
| Hafnian, loop hafnian, torontonian | 40 × 40 |
| Rows of a repeated hafnian | 40 |
| Photons in one Gaussian probability | 20 (clicks: 20 detectors) |
| Gaussian modes | 1024 |
