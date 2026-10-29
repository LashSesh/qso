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
    # 1H-1H-13C-15N at 9.4 T and room temperature (polarisations of 10⁻⁵),
    # built independently with QuTiP's spin operators: the Hamiltonian of
    # the rotating frame, the thermal state, the pulses and the Lindblad
    # operators; against SpinSystem.evolve, relative to the largest
    # expectation.
    spins = [("1H", 1.0, 0.5, 0.2), ("1H", 3.0, 0.7, 0.3), ("13C", 40.0, 1.5, 0.5), ("15N", 120.0, 2.0, 0.6)]
    couplings = [(0, 1, 7.0), (0, 2, 140.0), (1, 2, 5.0), (2, 3, -12.0)]
    field, temperature = 9.4, 298.15
    sys = q.SpinSystem(field, temperature)
    for iso, shift, t1, t2 in spins:
        sys.add_spin(iso, shift, t1=t1, t2=t2)
    for i, j, jh in couplings:
        sys.couple(i, j, jh)

    def op(i, which):
        return qt.tensor([qt.jmat(0.5, which) if k == i else qt.qeye(2) for k in range(4)])

    gamma = {iso: q.spin.gyromagnetic_ratio(iso) for iso, *_ in spins}
    h = sum(-gamma[iso] * field * shift * 1e-6 * op(i, "z") for i, (iso, shift, _, _) in enumerate(spins))
    for i, j, jh in couplings:
        if spins[i][0] == spins[j][0]:
            h += 2 * math.pi * jh * sum(op(i, w) * op(j, w) for w in "xyz")
        else:
            h += 2 * math.pi * jh * op(i, "z") * op(j, "z")
    assert np.abs(sys.hamiltonian() - h.full()).max() < 1e-13 * np.abs(h.full()).max()
    factors, c_ops = [], []
    for i, (iso, shift, t1, t2) in enumerate(spins):
        x = 1.054571817e-34 * gamma[iso] * field * (1 + shift * 1e-6) / (1.380649e-23 * temperature)
        w = np.array([math.exp(x / 2), math.exp(-x / 2)])
        factors.append(qt.Qobj(np.diag(w / w.sum())))
        up = 1 / t1 / (1 + math.exp(-x))
        c_ops += [
            math.sqrt(1 / t1 - up) * op(i, "-"),
            math.sqrt(up) * op(i, "+"),
            math.sqrt(2 * (1 / t2 - 1 / (2 * t1))) * op(i, "z"),
        ]
    rho = qt.tensor(factors)
    for iso, angle, phase in (("1H", 70.0, 20.0), ("13C", 90.0, 0.0)):
        a, p = math.radians(angle), math.radians(phase)
        g = sum(math.cos(p) * op(i, "x") + math.sin(p) * op(i, "y") for i, s in enumerate(spins) if s[0] == iso)
        u = (-1j * a * g).expm()
        rho = u * rho * u.dag()
    ops = [op(i, "z") for i in range(4)] + [sum(op(i, w) for i in (0, 1)) for w in "xy"]
    times = np.linspace(0, 0.3, 7)
    rho0 = sys.evolve(sys.thermal_state(), [spin.pulse("1H", 70, 20), spin.pulse("13C", 90)])
    assert np.abs(rho0 - rho.full()).max() < 1e-15
    # Exactly: the exponential of QuTiP's Liouvillian on the deviation from
    # 1/16, with L(1/16) as an augmented column.
    scipy_linalg = pytest.importorskip("scipy.linalg")
    lv = qt.liouvillian(h, c_ops).full()
    mixed = (np.eye(16) / 16).reshape(-1, order="F")
    dev = (rho.full() - np.eye(16) / 16).reshape(-1, order="F")
    exact = []
    for t in times:
        aug = np.zeros((257, 257), dtype=complex)
        aug[:256, :256] = lv * t
        aug[:256, 256] = lv @ mixed * t
        p = scipy_linalg.expm(aug)
        exact.append((p[:256, :256] @ dev + p[:256, 256] + mixed).reshape(16, 16, order="F"))
    # And by mesolve, whose adaptive steps lose about 10⁻⁶ of the deviation.
    options = {"atol": 1e-15, "rtol": 1e-12, "nsteps": 10**7}
    integrated = qt.mesolve(h, rho, times, c_ops, e_ops=ops, options=options).expect
    scale = max(np.abs(e).max() for e in integrated)
    assert 1e-6 < scale < 1e-4
    for k, t in enumerate(times):
        state = sys.evolve(rho0, [spin.delay(t)])
        for o, ref in zip(ops, integrated):
            value = sys.expectation(state, o.full()).real
            assert abs(value - np.trace(exact[k] @ o.full()).real) < 1e-9 * scale
            assert abs(value - ref[k]) < 1e-4 * scale


def test_bindings_reject_bools_and_name_operators():
    sys = two_protons()
    with pytest.raises(ValueError, match="bool"):
        spin.pulse(True, 90)
    with pytest.raises(ValueError, match="an operator"):
        sys.expectation(sys.identity_state(), np.eye(2))
    with pytest.raises(ValueError, match="inept"):
        spin.inept("1H", "1H", 140.0)
    with pytest.raises(ValueError):
        spin.inept_cycle("1H", "13C", 0.0)
    warm = sys.with_temperature(77.0)
    assert warm.temperature == 77.0 and sys.temperature == 298.15
    assert sys.with_carrier("1H", 4.7).carrier("1H") == 4.7 and sys.carrier("1H") == 0.0
    events = [spin.delay(1e-3), spin.set_field(2.0), spin.sweep(0.5, 1.0, 4)]
    assert sys.field_after(events) == 0.5
    sys.set_relaxation(0, 2.0, 1.0)
    with pytest.raises(ValueError):
        sys.set_relaxation(0, 1.0, 3.0)
    assert sys.spins[0][2:] == (2.0, 1.0)
