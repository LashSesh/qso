"""Nuclear spin dynamics: closed forms, and QuTiP when it is installed."""

import math

import numpy as np
import pytest

import metatron_qso as q
from metatron_qso import spin


def two_protons(field=14.1):
    sys = q.SpinSystem(field)
    a = sys.add_spin("1H", 1.0)
    b = sys.add_spin("1H", 11.0)
    sys.couple(a, b, 7.0)
    return sys


def test_fid_of_one_spin():
    sys = q.SpinSystem(9.4)
    sys.add_spin("1H", 2.0, t1=0.2, t2=0.05)
    rho0 = sys.thermal_state()
    p = sys.polarization(rho0, 0)
    x = 1.054571817e-34 * q.spin.gyromagnetic_ratio("1H") * 9.4 * (1 + 2e-6)
    assert p == pytest.approx(math.tanh(x / (2 * 1.380649e-23 * 298.15)), rel=1e-12)
    rho = sys.evolve(rho0, [spin.pulse("1H", 90)])
    omega = -q.spin.gyromagnetic_ratio("1H") * 9.4 * 2e-6
    t = np.arange(64) * 1e-4
    want = -0.5j * p * np.exp(1j * omega * t - t / 0.05)
    assert np.abs(sys.fid(rho, "1H", 1e-4, 64) - want).max() < 1e-14
    freqs, values = spin.spectrum(sys.fid(rho, "1H", 1e-4, 1024), 1e-4)
    assert abs(freqs[np.argmax(np.abs(values))] - omega / (2 * np.pi)) < 10.0
    (line,) = sys.lines(rho, "1H")
    assert line["ppm"] == pytest.approx(2.0)


def test_inept_and_its_phase_cycle():
    sys = q.SpinSystem(9.4)
    h = sys.add_spin("1H", 4.0)
    c = sys.add_spin("13C", 60.0)
    sys.couple(h, c, 140.0)
    eq = sys.thermal_state()
    direct = sys.lines(sys.evolve(eq, [spin.pulse("13C", 90)]), "13C")
    cycle = spin.inept_cycle("1H", "13C", 140.0)
    fid = sys.phase_cycle(eq, cycle, "13C", 1e-4, 4)
    inept = [sys.lines(sys.evolve(eq, events), "13C") for events, _ in cycle]
    lines = [(a["amplitude"] - b["amplitude"]) / 2 for a, b in zip(*inept)]
    ratio = sys.polarization(eq, h) / sys.polarization(eq, c)
    assert ratio == pytest.approx(3.976, abs=1e-3)
    mean_direct = np.mean([abs(d["amplitude"]) for d in direct])
    assert abs(lines[0]) / mean_direct == pytest.approx(ratio, rel=1e-8)
    assert abs(lines[0] + lines[1]) < 1e-9 * abs(lines[0])
    assert fid[0] == pytest.approx(sum(lines), abs=1e-15)


def test_pasadena_lines():
    sys = two_protons()
    para = sys.time_average(sys.parahydrogen_state(0, 1))
    rho = sys.evolve(para, [spin.pulse("1H", 45)])
    lines = sys.lines(rho, "1H")
    assert len(lines) == 4
    for line in lines:
        assert abs(line["amplitude"]) == pytest.approx(1 / 8, rel=3e-3)
    assert abs(sum(line["amplitude"] for line in lines)) < 1e-10


def test_sabre_peaks_at_the_level_anti_crossing():
    sys = q.SpinSystem(0.0)
    for shift in (-22.0, -22.0, 8.0):
        sys.add_spin("1H", shift)
    sys.couple(0, 1, -7.0)
    sys.couple(0, 2, 1.2)
    nu = q.spin.gyromagnetic_ratio("1H") / (2 * np.pi) * 30e-6
    lac = (7.0 + 1.2 / 4) / nu
    fields = lac * np.linspace(0.5, 1.5, 201)
    transfer = []
    for b in fields:
        at = sys.with_field(b)
        transfer.append(
            abs(at.polarization(at.time_average(at.parahydrogen_state(0, 1)), 2))
        )
    assert fields[int(np.argmax(transfer))] / lac == pytest.approx(1.0, abs=0.02)


def test_sweeps_shaped_pulses_json_and_errors():
    sys = two_protons(0.0)
    carried = sys.evolve(
        sys.parahydrogen_state(0, 1),
        [spin.sweep(1.0, 20.0, 20000), spin.set_field(14.1)],
    )
    assert max(carried[1, 1].real, carried[2, 2].real) > 0.995
    eq = q.SpinSystem(9.4)
    eq.add_spin("1H")
    rho = eq.thermal_state()
    shaped = eq.evolve(rho, [spin.shaped("1H", [(1e-5, 25_000.0, 30.0)])])
    hard = eq.evolve(rho, [spin.pulse(0, 90, 30)])
    assert np.abs(shaped - hard).max() < 1e-14
    again = q.SpinSystem.from_json(sys.to_json())
    assert again.couplings == sys.couplings and again.dimension == 4
    assert sys.coupling_graph().num_edges == 1
    with pytest.raises(ValueError):
        q.SpinSystem(-1.0)
    with pytest.raises(ValueError, match="isotope"):
        sys.add_spin("17O")
    with pytest.raises(ValueError):
        sys.add_spin("1H", t1=1.0, t2=3.0)
    assert sys.dimension == 4  # unchanged by the failed call
    with pytest.raises(ValueError):
        sys.evolve(np.eye(3), [])
    with pytest.raises(ValueError):
        sys.evolve(sys.identity_state(), [spin.pulse("13C", 90)])
    assert "2 spins" in repr(sys)


def test_relaxation_matches_qutip():
    qt = pytest.importorskip("qutip")
    # Three coupled protons with T1 and T2, from thermal equilibrium after
    # a 70° pulse: ⟨F_x⟩, ⟨F_y⟩ and ⟨I_z⟩ of each against mesolve.
    sys = q.SpinSystem(1.0, temperature=0.05)
    for k, shift in enumerate((0.0, 2.0, 5.0)):
        sys.add_spin("1H", shift, t1=0.4 + 0.2 * k, t2=0.1 + 0.05 * k)
    sys.couple(0, 1, 6.0)
    sys.couple(1, 2, -3.0)
    rho0 = sys.evolve(sys.thermal_state(), [spin.pulse("1H", 70, 20)])
    h = sys.hamiltonian()
    gamma = q.spin.gyromagnetic_ratio("1H")
    c_ops = []
    for i, (_, shift, t1, t2) in enumerate(sys.spins):
        x = 1.054571817e-34 * gamma * 1.0 * (1 + shift * 1e-6) / (1.380649e-23 * 0.05)
        up, down = (
            1 / t1 / (1 + math.exp(-x)),
            1 / t1 * math.exp(-x) / (1 + math.exp(-x)),
        )
        iz = sys.spin_operator(i, "z")
        ip = sys.spin_operator(i, "+")
        c_ops += [
            math.sqrt(down) * ip.conj().T,
            math.sqrt(up) * ip,
            math.sqrt(2 * (1 / t2 - 1 / (2 * t1))) * iz,
        ]
    dims = [[2, 2, 2], [2, 2, 2]]
    times = np.linspace(0, 0.2, 21)
    ops = [sys.total_operator("1H", "x"), sys.total_operator("1H", "y")] + [
        sys.spin_operator(i, "z") for i in range(3)
    ]
    result = qt.mesolve(
        qt.Qobj(h, dims=dims),
        qt.Qobj(rho0, dims=dims),
        times,
        [qt.Qobj(c, dims=dims) for c in c_ops],
        e_ops=[qt.Qobj(o, dims=dims) for o in ops],
        options={"atol": 1e-12, "rtol": 1e-10},
    )
    for k, t in enumerate(times):
        rho = sys.evolve(rho0, [spin.delay(t)])
        for op, ref in zip(ops, result.expect):
            assert sys.expectation(rho, op).real == pytest.approx(ref[k], abs=1e-7)
