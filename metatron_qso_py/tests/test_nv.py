"""NV centres: an independent NumPy propagation, closed forms for noise."""

import math

import numpy as np
import pytest
from metatron_qso import spin

TWO_PI = 2 * math.pi
GAMMA_N14 = 3.0777e6
GAMMA_C13 = spin.gyromagnetic_ratio("13C") / TWO_PI


def ops(levels):
    """`(Sz, Sx, Sy)` of spin `(levels − 1)/2`, m descending."""
    s = (levels - 1) / 2
    m = s - np.arange(levels)
    plus = np.zeros((levels, levels))
    for k in range(1, levels):
        plus[k - 1, k] = math.sqrt(s * (s + 1) - m[k] * (m[k] + 1))
    return np.diag(m), (plus + plus.T) / 2, (plus - plus.T) / 2j


def kron(*mats):
    out = np.eye(1)
    for m in mats:
        out = np.kron(out, m)
    return out


def reference_hamiltonian(field, frequency, nitrogen, carbons):
    """`H` (rad/s) in the frame rotating at `frequency`, built from the
    secular NV Hamiltonian of Doherty et al. (2013) term by term."""
    d, gamma = 2.87e9, spin.NV_GYROMAGNETIC_RATIO
    sz = ops(3)[0]
    dims = [3] + ([3] if nitrogen else []) + [2] * len(carbons)

    def embed(k, op):
        return kron(*[op if j == k else np.eye(n) for j, n in enumerate(dims)])

    h = embed(0, (d - frequency) * sz @ sz + gamma * field * sz)
    k = 1
    if nitrogen:
        iz = ops(3)[0]
        h = h + -2.14e6 * embed(0, sz) @ embed(1, iz)
        h = h + -4.945e6 * embed(1, iz @ iz) - GAMMA_N14 * field * embed(1, iz)
        k = 2
    for a_par, a_perp in carbons:
        iz, ix, _ = ops(2)
        h = h + embed(0, sz) @ (a_par * embed(k, iz) + a_perp * embed(k, ix))
        h = h - GAMMA_C13 * field * embed(k, iz)
        k += 1
    return TWO_PI * h, embed


def reference_evolve(
    field, frequency, nitrogen, carbons, rho, events, transition, rabi
):
    """`U ρ U†` after `events` given as `("pulse", angle, phase)` or
    `("free", t)`. Finite pulses keep the co-rotating half of a carrier
    `√2·2πΩ cos(ωt + φ) S_x`: `2π(Ω/2) e^{−iφ}` from `|0⟩` to `|±1⟩`; ideal
    pulses rotate the two levels `{m, 0}` alone, in closed form."""
    scipy_linalg = pytest.importorskip("scipy.linalg")
    h0, embed = reference_hamiltonian(field, frequency, nitrogen, carbons)

    def duration(e):
        return abs(e[1]) / 360 / rabi if (rabi and e[0] == "pulse") else 0.0

    merged = []
    for e in events:
        if merged and e[0] == "free" and merged[-1][0] == "free":
            merged[-1] = ("free", merged[-1][1] + e[1])
        else:
            merged.append(e)
    for k, e in enumerate(merged):
        if e[0] == "free":
            t = e[1]
            if rabi:
                before = duration(merged[k - 1]) if k > 0 else 0.0
                after = duration(merged[k + 1]) if k + 1 < len(merged) else 0.0
                t -= (before + after) / 2
            u = scipy_linalg.expm(-1j * h0 * t)
        else:
            _, angle, phase = e
            if rabi:
                if angle < 0:
                    angle, phase = -angle, phase + 180
                up = np.zeros((3, 3), complex)  # |±1⟩⟨0|, levels m = +1, 0, −1
                up[0, 1] = up[2, 1] = 1
                half = TWO_PI * rabi / 2 * np.exp(-1j * math.radians(phase)) * up
                drive = half + half.conj().T
                u = scipy_linalg.expm(-1j * (h0 + embed(0, drive)) * angle / 360 / rabi)
            else:
                m = 2 if transition < 0 else 0
                g = np.zeros((3, 3), complex)
                g[m, 1] = np.exp(-1j * math.radians(phase))
                g[1, m] = np.exp(1j * math.radians(phase))
                p = np.zeros((3, 3))
                p[m, m] = p[1, 1] = 1
                c, s_ = (
                    math.cos(math.radians(angle) / 2),
                    math.sin(math.radians(angle) / 2),
                )
                u = embed(0, np.eye(3) - p + c * p - 1j * s_ * g)
        rho = u @ rho @ u.conj().T
    return rho


def as_tuples(events):
    return [
        ("free", e.duration) if e.kind == "free" else ("pulse", e.angle, e.phase)
        for e in events
    ]


@pytest.mark.parametrize("rabi", [None, 12e6])
@pytest.mark.parametrize("transition", [-1, 1])
def test_evolution_matches_an_independent_numpy_propagation(rabi, transition):
    field, nitrogen, carbons = 0.031, True, [(60e3, 30e3), (-20e3, 45e3)]
    nv = spin.NvCenter(field, nitrogen="14N", carbons=carbons)
    frequency = 2.87e9 + transition * spin.NV_GYROMAGNETIC_RATIO * field + 0.3e6
    h, _ = reference_hamiltonian(field, frequency, nitrogen, carbons)
    assert np.abs(nv.hamiltonian(frequency) - h).max() < 1e-12 * np.abs(h).max()
    # A random state: the phases of the pulses show in its coherences.
    rng = np.random.default_rng(4)
    a = rng.normal(size=(36, 36)) + 1j * rng.normal(size=(36, 36))
    rho = a @ a.conj().T
    rho /= np.trace(rho)
    sequences = [
        spin.ramsey(0.3e-6),
        spin.hahn(0.5e-6),
        spin.xy8(1, 0.2e-6),
        spin.cpmg(3, 0.25e-6),
        [spin.nv_pulse(-70, 20), spin.nv_free(0.1e-6), spin.nv_pulse(45, 100)],
        [
            spin.nv_pulse(90, 30),
            spin.nv_free(0.0),
            spin.nv_free(0.2e-6),
            spin.nv_pulse(135, 250),
        ],
    ]
    for events in sequences:
        kw = {"frequency": frequency, "transition": transition, "rabi": rabi}
        got = nv.evolve(rho, events, **kw)
        want = reference_evolve(
            field,
            frequency,
            nitrogen,
            carbons,
            rho,
            as_tuples(events),
            transition,
            rabi,
        )
        assert np.abs(got - want).max() < 1e-10, events
        start = nv.initial_state()
        p0 = reference_evolve(
            field,
            frequency,
            nitrogen,
            carbons,
            start,
            as_tuples(events),
            transition,
            rabi,
        )
        n = 12
        assert nv.signal(events, **kw) == pytest.approx(
            np.trace(p0[n : 2 * n, n : 2 * n]).real, abs=1e-10
        )


def ou_ramsey(sigma, tc, t):
    """Ramsey decay exponent of Ornstein–Uhlenbeck noise (Klauder and
    Anderson, Phys. Rev. 125, 912 (1962))."""
    x = t / tc
    return (TWO_PI * sigma * tc) ** 2 * (x - 1 + math.exp(-x))


def ou_hahn(sigma, tc, t):
    """Hahn-echo decay exponent of total time `t` (same source)."""
    x = t / tc
    return (TWO_PI * sigma * tc) ** 2 * (x - 3 + 4 * math.exp(-x / 2) - math.exp(-x))


def test_ornstein_uhlenbeck_decays_follow_their_closed_forms():
    for sigma, tc in [(1e5, 1e-6), (3e4, 5e-6), (2e5, 2e-7)]:
        noise = spin.FieldNoise.ornstein_uhlenbeck(sigma, tc)
        for tau in [1e-7, 1e-6, 4e-6]:
            chi = spin.decay_exponent(spin.ramsey(tau), noise)
            assert chi == pytest.approx(ou_ramsey(sigma, tc, tau), rel=1e-10)
            chi = spin.decay_exponent(spin.hahn(tau), noise)
            assert chi == pytest.approx(ou_hahn(sigma, tc, 2 * tau), rel=1e-9)


def segments(events):
    """Free periods with their toggling-frame signs (ideal π pulses)."""
    t, sign, out = 0.0, 1.0, []
    for kind, *values in as_tuples(events)[1:-1]:
        if kind == "free":
            out.append((t, t + values[0], sign))
            t += values[0]
        else:
            sign = -sign
    return out


def pink_spectrum(sigma, lowest, highest, n, w):
    """`S(ω)` of `n` Ornstein–Uhlenbeck fluctuators of variance `σ²/n` with
    corners `f_k` evenly spaced in `log f`."""
    f = lowest * (highest / lowest) ** (np.arange(n) / (n - 1))
    theta = TWO_PI * f
    w = np.asarray(w, float)[..., None]
    return (2 * sigma**2 / n * theta / (theta**2 + w**2)).sum(axis=-1)


def test_filter_functions_and_their_integral():
    sigma, lowest, highest, n = 1e5, 1e3, 3e5, 12
    noise = spin.FieldNoise.pink(sigma, lowest, highest, fluctuators=n)
    grid = np.array([0.0, 1e3, 1e6, 3e7])
    assert np.allclose(
        noise.spectrum(grid), pink_spectrum(sigma, lowest, highest, n, grid)
    )
    nodes, weights = np.polynomial.legendre.leggauss(24)
    for events in [spin.cpmg(4, 0.7e-6), spin.xy8(1, 0.4e-6), spin.ramsey(1e-6)]:
        segs = segments(events)

        def y2(w, segs=segs):
            y = np.zeros_like(w, dtype=complex)
            for a, b, s in segs:
                y += s * (np.exp(1j * w * b) - np.exp(1j * w * a)) / (1j * w)
            return np.abs(y) ** 2

        omegas = np.array([1e4, 3.3e6, 2.2e7, 1e8])
        got = spin.filter_function(events, omegas)
        assert np.allclose(got, y2(omegas), rtol=1e-10, atol=0)
        total = segs[-1][1]
        assert spin.filter_function(events, [0.0])[0] == pytest.approx(
            sum(s * (b - a) for a, b, s in segs) ** 2, abs=1e-30
        )
        # χ = ½⟨φ²⟩ = (2π)²/(2·2π) ∫ S|Y|² dω = 2π ∫₀^∞ S|Y|² dω, by
        # Gauss–Legendre on 4000 pieces up to 1000/T (the integrand falls
        # as ω⁻⁴ beyond the highest corner, so the rest is below 10⁻⁶).
        edges = np.linspace(0, 1000 / total, 4001)
        mid, half = (edges[1:] + edges[:-1]) / 2, (edges[1:] - edges[:-1]) / 2
        w = (mid[:, None] + half[:, None] * nodes[None, :]).ravel()
        integral = (
            np.repeat(half, nodes.size)
            * np.tile(weights, mid.size)
            * (pink_spectrum(sigma, lowest, highest, n, w) * y2(w))
        ).sum()
        chi = spin.decay_exponent(events, noise)
        assert TWO_PI * integral == pytest.approx(chi, rel=1e-5)


def test_trajectories_match_the_gaussian_decay_and_static_telegraph_noise():
    nv = spin.NvCenter(0.02)
    noise = spin.FieldNoise.ornstein_uhlenbeck(2e5, 1e-6)
    for events in [spin.hahn(1.5e-6), spin.cpmg(4, 0.8e-6), spin.xy8(1, 0.5e-6)]:
        mean, err = nv.signal_with_noise(events, noise, 4000, seed=11)
        want = (1 + math.exp(-spin.decay_exponent(events, noise))) / 2
        assert abs(mean - want) < 4 * err + 1e-12, (mean, err, want)
    # Static ±a (no flips): every Ramsey trajectory gives (1 + cos 2πaτ)/2.
    static = spin.FieldNoise.telegraph(3e5, 0.0)
    assert not static.is_gaussian
    mean, err = nv.signal_with_noise(spin.ramsey(0.7e-6), static, 64, seed=2)
    assert mean == pytest.approx((1 + math.cos(TWO_PI * 3e5 * 0.7e-6)) / 2, abs=1e-12)
    assert err < 1e-12
    # …and an echo refocuses it.
    mean, _ = nv.signal_with_noise(spin.hahn(0.7e-6), static, 16, seed=2)
    assert mean == pytest.approx(1.0, abs=1e-12)


def test_odmr_lines_and_dips():
    nv = spin.NvCenter(0.01, nitrogen="15N")
    lines = nv.lines()
    want = sorted(
        2.87e9 + ms * 28.025e9 * 0.01 + ms * 3.03e6 * mi
        for ms in (1, -1)
        for mi in (0.5, -0.5)
    )
    assert np.allclose([f for f, _ in lines], want, rtol=0, atol=1e-3)
    # A π pulse at each ¹⁵N line of 0 ↔ −1 flips half the nuclear
    # population (the other line, 3 MHz away, barely responds).
    centre = 2.87e9 - 28.025e9 * 0.01
    p0 = nv.odmr(centre - 3.03e6 * np.array([0.5, -0.5]), 0.25e6, 2e-6)
    assert np.allclose(p0, 0.5, atol=0.01), p0
    assert nv.odmr([centre], 0.25e6, 2e-6)[0] > 0.9


def test_nv_bindings():
    nv = spin.NvCenter(0.05, nitrogen="14N", carbons=[(1e4, 2e3)])
    assert nv.dimension == 18 and nv.field == 0.05
    assert repr(nv) == "NvCenter(0.05 T, 14N, 1 carbons)"
    again = spin.NvCenter.from_json(nv.to_json())
    assert again.to_json() == nv.to_json()
    rho = nv.initial_state()
    assert rho.shape == (18, 18) and np.trace(rho).real == pytest.approx(1.0)
    out = nv.evolve(rho, spin.hahn(1e-6), rabi=30e6)
    assert np.trace(out).real == pytest.approx(1.0)
    assert [e.kind for e in spin.ramsey(1e-6)] == ["pulse", "free", "pulse"]
    # π/2, then 16 × (free, π), free, π/2.
    assert len(spin.xy8(2, 1e-6)) == 35
    event = spin.nv_pulse(90, 30)
    assert (event.angle, event.phase, event.duration) == (90, 30, None)
    assert spin.nv_free(2e-6).duration == 2e-6
    assert spin.NV_ZERO_FIELD_SPLITTING == 2.87e9 and spin.MAX_NV_DIMENSION == 768
    noise = spin.FieldNoise.telegraph(1e5, 1e4)
    assert noise.correlation([0.0])[0] == pytest.approx(1e10)
    assert noise.spectrum([0.0])[0] == pytest.approx(2 * 1e10 * 2e4 / (2e4) ** 2)
    with pytest.raises(ValueError):
        spin.NvCenter(0.01, nitrogen="13C")
    with pytest.raises(ValueError):
        spin.NvCenter(0.01, nitrogen="14N", carbons=[(1e3, 0.0)] * 7)
    with pytest.raises(ValueError):
        spin.NvCenter(float("nan"))
    with pytest.raises(ValueError):
        nv.signal(spin.ramsey(1e-6), transition=0)
    with pytest.raises(ValueError):
        nv.signal(spin.ramsey(1e-6), rabi=-1.0)
    with pytest.raises(ValueError):
        nv.signal([spin.nv_free(-1.0)])
    with pytest.raises(ValueError):
        nv.signal(spin.cpmg(2, 1e-9), rabi=1e6)
    with pytest.raises(ValueError):
        nv.evolve(np.eye(3), spin.ramsey(1e-6))
    with pytest.raises(ValueError):
        spin.FieldNoise.ornstein_uhlenbeck(1.0, 0.0)
    with pytest.raises(ValueError):
        spin.FieldNoise.pink(1.0, 10.0, 1.0)
    with pytest.raises(ValueError):
        spin.filter_function([spin.nv_free(1e-6)], [1.0])
    with pytest.raises(ValueError):
        nv.signal_with_noise(spin.hahn(1e-6), noise, 0)
    with pytest.raises(ValueError):
        spin.NvCenter.from_json('{"field": 1}')
    with pytest.raises(ValueError):
        nv.signal(spin.ramsey(1e-6), transition=128)
    with pytest.raises(ValueError):
        spin.cpmg(0, 1e-6)
    with pytest.raises(ValueError):
        spin.xy8(0, 1e-6)
    with pytest.raises(ValueError):
        spin.NvCenter(0.01, nitrogen=("14N", 1.0))
    # Custom nitrogen couplings and the getters.
    custom = spin.NvCenter(0.02, nitrogen=("15N", 3.1e6, 0.0), carbons=[(5e4, 1e4)])
    assert custom.nitrogen == ("15N", 3.1e6, 0.0)
    assert custom.carbons == [(5e4, 1e4)]
    assert nv.nitrogen == ("14N", -2.14e6, -4.945e6)
    assert custom.zero_field_splitting == 2.87e9
    assert custom.gyromagnetic_ratio == spin.NV_GYROMAGNETIC_RATIO
    assert spin.NvCenter.from_json(custom.to_json()).nitrogen == custom.nitrogen
    # Lines carry half the strength per transition.
    assert sum(w for _, w in nv.lines()) == pytest.approx(1.0)
    # evolve takes any operator (here a coherence) linearly.
    x = np.zeros((18, 18), complex)
    x[6, 12] = 1.0
    y = nv.evolve(x, spin.hahn(1e-6))
    assert np.allclose(y + y.conj().T, nv.evolve(x + x.conj().T, spin.hahn(1e-6)))
    _, err = nv.signal_with_noise(spin.hahn(1e-6), noise, 1)
    assert math.isnan(err)
