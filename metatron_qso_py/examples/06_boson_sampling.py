"""Boson sampling: interference of indistinguishable photons."""

import math

import numpy as np

import metatron_qso as q

# Hong–Ou–Mandel: two photons on a 50:50 beam splitter always leave together.
hom = q.BosonSampler(q.beam_splitter(math.pi / 4), [1, 1])
print("Hong-Ou-Mandel, P(one photon per port):")
print(f"  indistinguishable {hom.probability([1, 1]):.3e}")
print(f"  distinguishable   {hom.probability_distinguishable([1, 1]):.3f}")

# Four photons in a Haar-random 8-mode interferometer.
u = q.haar_unitary(8, seed=1)
sampler = q.BosonSampler(u, [1, 1, 1, 1, 0, 0, 0, 0])
patterns, p_boson = sampler.distribution()
_, p_dist = sampler.distribution(distinguishable=True)
print(f"\n4 photons in 8 modes: {len(patterns)} output patterns")
print(f"  total-variation distance boson vs distinguishable: {0.5 * np.abs(p_boson - p_dist).sum():.3f}")

# Exact samples follow the exact distribution.
shots = 50_000
samples = sampler.sample(shots, seed=2)
index = {tuple(p): k for k, p in enumerate(patterns)}
freq = np.bincount([index[tuple(s)] for s in samples], minlength=len(patterns)) / shots
print(f"  {shots} exact samples, distance to the exact distribution: {0.5 * np.abs(freq - p_boson).sum():.4f}")

# Full bunching is enhanced by n! relative to distinguishable photons.
bunched = [0, 0, 4, 0, 0, 0, 0, 0]
ratio = sampler.probability(bunched) / sampler.probability_distinguishable(bunched)
print(f"  P(all four in mode 2), boson / distinguishable = {ratio:.6f} (4! = 24)")
