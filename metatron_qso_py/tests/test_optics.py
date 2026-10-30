"""The optical cycle of an NV centre, readout and magnetometer sensitivity."""

import math

import numpy as np
import pytest

from metatron_qso import spin

KR, K47, K57, K71, K72 = 65.9e6, 7.9e6, 53.3e6, 1.0e6, 0.7e6


def test_rates_and_steady_state_against_numpy():
    # The seven rate equations along the axis, solved by NumPy.
    pump = 4e6
    nv = spin.OpticalNv((0, 0, 0.01), dephasing=0.0, spin_lattice=0.0)
    assert nv.rates == [KR, K47, K57, K71, K72]
    rho = nv.steady_state(pump)
    # Levels: ground 0, -1, +1 (ascending at 10 mT), excited likewise, singlet.
    rates = np.zeros((7, 7))
    for g, e in ((0, 3), (1, 4), (2, 5)):
        rates[g, e] = pump
        rates[e, g] = KR
    rates[3, 6], rates[4, 6], rates[5, 6] = K47, K57, K57
    rates[6, 0], rates[6, 1], rates[6, 2] = K71, K72, K72
    a = rates.T - np.diag(rates.sum(axis=1))
    a[0, :] = 1.0
    b = np.zeros(7)
    b[0] = 1.0
    want = np.linalg.solve(a, b)
    assert np.abs(np.diag(rho).real - want).max() < 1e-12
    photons = nv.fluorescence(pump)
    assert abs(photons - 0.01 * KR * want[3:6].sum()) < 1e-6


def dreau_population(gp, gc, g1, g2s, detuning, rabi):
    g2 = g2s + gc
    a = 2 * g1 + gp
    num = g1 / a * (detuning**2 + g2**2) + g2 * rabi**2 / (2 * a)
    return num / (detuning**2 + g2**2 + g2 * rabi**2 / a)


def test_two_level_model_follows_dreau():
    m = spin.TwoLevelOdmr(2.5e6, 4e7, 1e3, 2e5, 2.5e5, 1.5e5)
    for rabi in (1e6, 2e7):
        for det in (0.0, 5e6, -4e7):
            assert abs(m.population(det, rabi) - dreau_population(2.5e6, 4e7, 1e3, 2e5, det, rabi)) < 1e-12
    g2, a, rabi = 4.02e7, 2e3 + 2.5e6, 2e7
    width = math.sqrt(g2**2 + rabi**2 * g2 / a) / math.pi
    line = m.line(20 * 2 * math.pi * width, rabi, points=8001)
    assert abs(line["linewidth"] / width - 1) < 1e-5


def test_levels_off_the_axis_are_the_roots_of_the_cubic():
    rng = np.random.default_rng(3)
    d = spin.NV_ZERO_FIELD_SPLITTING
    g = spin.NV_GYROMAGNETIC_RATIO
    for _ in range(20):
        b = rng.normal(size=3) * 0.02
        nv = spin.OpticalNv(tuple(b))
        levels, _ = nv.ground_levels()
        bb, bp = g * np.linalg.norm(b), g * math.hypot(b[0], b[1])
        roots = np.sort(np.roots([1, -2 * d, d * d - bb * bb, d * bp * bp]).real)
        assert np.abs(levels - roots).max() < 1e-3


def test_field_vector_from_four_orientations():
    rng = np.random.default_rng(5)
    for _ in range(10):
        b = rng.normal(size=3) * 0.01
        lines = spin.ensemble_lines(tuple(b))
        back = np.array(spin.field_from_lines(lines))
        sign = 1 if np.dot(b, spin.nv_orientations()[0]) >= 0 else -1
        assert np.abs(back - sign * b).max() < 1e-11
    axis = spin.nv_orientations()[0]
    along, across = spin.axial_and_transverse(spin.ensemble_lines((0.003, 0.0, 0.0))[0])
    assert abs(along - 0.003 * axis[0]) < 1e-12


def test_cw_line_pulsed_readout_and_sensitivity():
    nv = spin.OpticalNv((0, 0, 0.005))
    f, strength = nv.lines()[0]
    assert abs(strength - 0.5) < 1e-12
    line = nv.cw_line(f, 20e6, 0.5e6, 5e6)
    assert 0.01 < line["contrast"] < 0.35
    lorentzian = 4 / (3 * math.sqrt(3)) * line["linewidth"] / (
        spin.NV_GYROMAGNETIC_RATIO * line["contrast"] * math.sqrt(line["off_resonance"])
    )
    assert abs(line["sensitivity"] / lorentzian - 1) < 0.02
    r = nv.pulsed_readout(KR, 2e-6, 2000)
    assert 0.01 < r["readout_contrast"] < 0.05
    assert r["photons_bright"] > r["photons_dark"]
    assert abs(r["readout_contrast"] - spin.readout_contrast(r["photons_bright"], r["photons_dark"])) < 1e-15
    # Ramsey with a Gaussian decay: the optimum at T2*/2.
    t2, c = 3e-6, 0.03
    times = np.arange(1, 20001) * 1e-10
    tau, eta = spin.best_sensitivity(list(times), list(np.exp(-((times / t2) ** 2))), c)
    assert abs(tau - t2 / 2) < 2e-10
    want = math.exp(0.25) * math.sqrt(2 / t2) / (2 * math.pi * spin.NV_GYROMAGNETIC_RATIO * c)
    assert abs(eta / want - 1) < 1e-6
    assert abs(spin.echo_sensitivity(1e-3, 0.5, c) / spin.ramsey_sensitivity(1e-3, 0.5, c) - math.pi / 2) < 1e-12


def test_invalid_input():
    with pytest.raises(ValueError):
        spin.OpticalNv((0, 0, 0.01), collection=0.0)
    nv = spin.OpticalNv((0, 0, 0.01))
    with pytest.raises(ValueError):
        nv.steady_state(-1.0)
    with pytest.raises(ValueError):
        nv.pulsed_readout(1e6, 1e-6, transition=0)
    with pytest.raises(ValueError):
        spin.ramsey_sensitivity(1e-6, 0.5, 0.0)
