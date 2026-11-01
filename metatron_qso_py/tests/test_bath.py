"""The nuclear-spin bath of an NV centre (cluster-correlation expansion)."""

import math

import numpy as np
import pytest

from metatron_qso import spin

MU0_4PI = 1.00000000055e-7
H = 6.62607015e-34
GAMMA_C = 6.728284e7 / (2 * math.pi)

SX = np.array([[0, 0.5], [0.5, 0]], complex)
SY = np.array([[0, -0.5j], [0.5j, 0]], complex)
SZ = np.array([[0.5, 0], [0, -0.5]], complex)


def embed(k: int, i: int, op: np.ndarray) -> np.ndarray:
    """`op` on spin `i` of `k` (spin 0 the least significant factor)."""
    out = np.eye(1, dtype=complex)
    for j in reversed(range(k)):
        out = np.kron(out, op if j == i else np.eye(2))
    return out


def expmh(h: np.ndarray, t: float) -> np.ndarray:
    w, v = np.linalg.eigh(h)
    return (v * np.exp(-1j * w * t)) @ v.conj().T


def dense_signal(field: float, transition: int, spins: list, events: list) -> float:
    """An independent NumPy propagation of the electron qubit and the bath."""
    k = len(spins)
    d = 2**k
    ops = [SX, SY, SZ]
    base = np.zeros((d, d), complex)
    for i in range(k):
        base += embed(k, i, SZ) * (-2 * math.pi * GAMMA_C * field)
        for j in range(i + 1, k):
            v = np.array(spins[i][0]) - np.array(spins[j][0])
            r = np.linalg.norm(v)
            n = v / r
            dd = MU0_4PI * GAMMA_C**2 * H / r**3
            for a in range(3):
                for b in range(3):
                    w = (a == b) - 3 * n[a] * n[b]
                    base += 2 * math.pi * dd * w * embed(k, i, ops[a]) @ embed(k, j, ops[b])
    hm = base.copy()
    for i, (_, hf) in enumerate(spins):
        for a in range(3):
            hm += 2 * math.pi * transition * hf[a] * embed(k, i, ops[a])
    pm, p0 = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])
    h = np.kron(pm, hm) + np.kron(p0, base)
    rho = np.kron(p0, np.eye(d) / d).astype(complex)
    for e in events:
        if e.kind == "free":
            u = expmh(h, e.duration)
        else:
            phi = math.radians(e.phase)
            g = np.array([[0, np.exp(-1j * phi)], [np.exp(1j * phi), 0]])
            u = np.kron(expmh(g, math.radians(e.angle) / 2), np.eye(d))
        rho = u @ rho @ u.conj().T
    return float(np.real(np.trace(rho[d:, d:])))


def compact(n: int) -> list:
    points = [
        [0.6, 0.2, 0.4],
        [0.9, 0.1, 0.1],
        [0.5, 0.6, 0.0],
        [0.8, 0.5, 0.5],
        [0.4, 0.3, 0.8],
    ]
    out = []
    for p in points[:n]:
        pos = [x * 1e-9 for x in p]
        out.append((pos, spin.point_dipole_hyperfine(pos)))
    return out


def odd_sequence(tau: float) -> list:
    return [
        spin.nv_pulse(70, 30),
        spin.nv_free(0.3 * tau),
        spin.nv_pulse(180, 45),
        spin.nv_free(0.7 * tau),
        spin.nv_pulse(110, -60),
    ]


def test_against_an_independent_dense_propagation():
    for transition in (-1, 1):
        spins = compact(4)
        bath = spin.SpinBath(0.02, spins, transition=transition)
        seqs = [spin.hahn(60e-6), spin.cpmg(3, 30e-6), spin.ramsey(15e-6), odd_sequence(80e-6)]
        exact = bath.signal(seqs, order=4, cutoff=math.inf)
        for s, x in zip(seqs, exact):
            assert abs(x - dense_signal(0.02, transition, spins, s)) < 1e-10
        # Lower orders are approximations of it.
        assert np.abs(bath.signal(seqs, order=1, cutoff=math.inf) - exact).max() > 1e-6


def test_without_couplings_it_is_the_nv_centre():
    spins = [([1e-9, 0, 0], [30e3, -20e3, 60e3]), ([0, 1e-9, 0], [-5e3, 12e3, -25e3])]
    bath = spin.SpinBath(0.03, spins)
    nv = spin.NvCenter(0.03, carbons=[(z, math.hypot(x, y)) for _, (x, y, z) in spins])
    seqs = [spin.hahn(20e-6), spin.xy8(1, 3e-6), odd_sequence(25e-6)]
    ours = bath.signal(seqs, order=1, cutoff=0.0)
    for s, x in zip(seqs, ours):
        assert abs(x - nv.signal(s)) < 1e-10


def test_lattice_and_point_dipole():
    a = spin.DIAMOND_LATTICE_CONSTANT
    sites = spin.diamond_sites(1.5e-9)
    expected = 8 * 4 / 3 * math.pi * (1.5e-9 / a) ** 3 - 2
    assert abs(len(sites) / expected - 1) < 0.05
    nearest = np.sort(np.linalg.norm(sites, axis=1))[:3]
    assert np.allclose(nearest, a * math.sqrt(3) / 4)
    a_zz = spin.point_dipole_hyperfine([0, 0, 1e-9])[2]
    assert abs(a_zz / (2 * MU0_4PI * spin.NV_GYROMAGNETIC_RATIO * GAMMA_C * H / 1e-27) - 1) < 1e-8
    bath = spin.SpinBath.diamond(0.05, radius=3e-9, seed=2)
    assert bath.num_spins > 100 and len(bath.spins) == bath.num_spins
    assert len(bath.clusters(1)) == bath.num_spins
    # Baths nest in the radius; the nearest nuclei can be left out.
    small = spin.SpinBath.diamond(0.05, radius=2e-9, seed=2)
    inner = [p for p, _ in bath.spins if np.linalg.norm(p) <= 2e-9]
    assert inner == [p for p, _ in small.spins]
    far = spin.SpinBath.diamond(0.05, radius=3e-9, seed=2, min_radius=1e-9)
    assert all(np.linalg.norm(p) >= 1e-9 for p, _ in far.spins)
    assert far.num_spins == bath.without_spins_within(1e-9).num_spins


def test_the_echo_revives_at_the_larmor_period():
    field = 0.005
    tl = 1 / (GAMMA_C * field)
    seqs = [spin.hahn(tl / 2), spin.hahn(tl)]
    mean, err = spin.bath_ensemble_signal(field, seqs, 6, radius=2.5e-9, seed=4)
    assert mean[0] < 0.52 and mean[1] > 0.98
    assert np.all(err >= 0)


def test_gaussian_noise_multiplies_the_coherence():
    bath = spin.SpinBath(0.05, compact(3))
    noise = spin.FieldNoise.ornstein_uhlenbeck(20e3, 30e-6)
    seqs = [spin.hahn(20e-6)]
    l = bath.coherence(seqs, order=3, cutoff=math.inf)
    mean, err = bath.signal_with_noise(seqs, noise, order=3, cutoff=math.inf)
    chi = spin.decay_exponent(seqs[0], noise)
    assert abs(mean[0] - (1 + (l[0] * math.exp(-chi)).real) / 2) < 1e-14
    assert err[0] == 0.0


def test_invalid_input():
    bath = spin.SpinBath(0.01, compact(2))
    with pytest.raises(ValueError):
        bath.signal([[spin.nv_pulse(90), spin.nv_pulse(90), spin.nv_pulse(90)]])
    with pytest.raises(ValueError):
        bath.signal([spin.hahn(1e-6)], order=0)
    with pytest.raises(ValueError):
        spin.SpinBath(0.01, compact(2), transition=0)
    with pytest.raises(ValueError):
        spin.SpinBath.diamond(0.01, abundance=0.0)
    with pytest.raises(ValueError):
        spin.SpinBath.diamond(0.01, radius=1.0)
    big = spin.SpinBath.diamond(0.01, abundance=1.0, radius=0.5e-9)
    with pytest.raises(ValueError):
        big.exact_coherence([spin.hahn(1e-6)])
