"""The NV centre with its nuclei near the level anticrossings."""

import numpy as np
import pytest

from metatron_qso import spin

KR = 65.9e6


def test_decoupled_nucleus_equals_the_optics():
    zero = [[0.0] * 3 for _ in range(3)]
    nv = spin.NuclearOpticalNv(
        (0, 0, 0.02),
        nitrogen="15N",
        nitrogen_ground=zero,
        nitrogen_excited=zero,
        nuclear_relaxation=1e3,
    )
    assert nv.dimension == 14
    optics = spin.OpticalNv((0, 0, 0.02))
    want = optics.fluorescence(0.1 * KR)
    got = nv.steady_state(0.1 * KR)
    assert got["fluorescence"] == pytest.approx(want, rel=1e-10)
    assert abs(got["polarisation"][0]) < 1e-8


def test_nitrogen_15_polarises_at_the_excited_anticrossing():
    # Duarte et al.'s couplings with Gupta's rates: about 95 %.
    nv = spin.NuclearOpticalNv(
        (0, 0, 0.051),
        nitrogen="15N",
        rates="gupta",
        nuclear_relaxation=10.0,
        nitrogen_ground=[[3.9e6, 0, 0], [0, 3.9e6, 0], [0, 0, 3.4e6]],
        nitrogen_excited=[[-38.5e6, 0, 0], [0, -38.5e6, 0], [0, 0, -58.1e6]],
    )
    s = nv.steady_state(4e6)
    assert 0.94 < s["polarisation"][0] < 0.96
    assert s["polarisation_zero"][0] == pytest.approx(
        nv.nuclear_polarisation_in(s["rho"], 0, 1)
    )
    rates, pols = nv.field_scan([(0, 0, 0.02), (0, 0, 0.051)], 4e6)
    assert pols[0][0] < pols[1][0]
    back = spin.NuclearOpticalNv.from_json(nv.to_json())
    assert back.steady_state(4e6)["polarisation"][0] == pytest.approx(s["polarisation"][0])


def test_ground_anticrossing_and_microwave_in_the_laboratory():
    crossing = 2.87e9 / 28.025e9
    nv = spin.NuclearOpticalNv((1e-4, 0, crossing), nitrogen="14N")
    far = spin.NuclearOpticalNv((1e-4, 0, crossing - 2e-3), nitrogen="14N")
    assert nv.steady_state(0.1 * KR)["fluorescence"] < 0.8 * far.steady_state(0.1 * KR)[
        "fluorescence"
    ]
    eta, rate, slope = nv.field_sensitivity(0.1 * KR, step=1e-6)
    assert eta > 0 and rate > 0
    # A microwave on the line at 5 mT: a dip, as in the rotating-wave model.
    weak = spin.NuclearOpticalNv((0, 0, 0.005), nitrogen="15N", nuclear_relaxation=1e3)
    f0 = abs(spin.OpticalNv((0, 0, 0.005)).lines()[0][0])
    on, off = weak.cw_odmr(np.array([f0, f0 + 20e6]), 0.5e6, 0.05 * KR)
    assert on < 0.98 * off
    driven = weak.steady_state(0.05 * KR, frequency=f0, rabi=0.5e6)
    assert driven["harmonics"] >= 2
    with pytest.raises(ValueError):
        spin.NuclearOpticalNv(nitrogen="13C")
    with pytest.raises(ValueError):
        spin.NuclearOpticalNv(rates="unknown")
