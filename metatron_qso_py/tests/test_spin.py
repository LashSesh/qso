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


def _rotations(n=8):
    """Rotations and weights that average polynomials of degree < n in the
    entries of a rotation matrix exactly: Euler angles with Gauss–Legendre
    nodes in cos β."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    out = []
    for a in np.arange(n) * 2 * np.pi / n:
        for cb, wb in zip(nodes, weights):
            for g in np.arange(n) * 2 * np.pi / n:
                b = math.acos(cb)
                rz = lambda x: np.array([[math.cos(x), -math.sin(x), 0], [math.sin(x), math.cos(x), 0], [0, 0, 1]])
                ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
                out.append((rz(a) @ ry @ rz(g), wb / (2 * n * n)))
    return out


def test_redfield_matches_qutip_bloch_redfield():
    qt = pytest.importorskip("qutip")
    scipy_linalg = pytest.importorskip("scipy.linalg")
    # ¹³C with a shielding tensor and two protons, laboratory frame at 1 T,
    # τ_c = 1 ns (ωτ_c ≈ 0.3 for ¹H). The bath operators come from an
    # exact numerical average over orientations (not from the closed form
    # QSO uses), the Redfield tensor from QuTiP's bloch_redfield_tensor.
    field, tau = 1.0, 1e-9
    spins = [("13C", 60.0, (0.0, 0.0, 0.0)), ("1H", 2.0, (1.09, 0.0, 0.0)), ("1H", 3.5, (-0.36, 1.03, 0.0))]
    csa = np.array([[-40.0, 12.0, 0.0], [12.0, 15.0, -8.0], [0.0, -8.0, 25.0]])
    couplings = [(0, 1, 140.0), (0, 2, 135.0), (1, 2, -12.0)]
    sys = q.SpinSystem(field, temperature=1e9, frame="laboratory")
    for k, (iso, shift, pos) in enumerate(spins):
        sys.add_spin(iso, shift)
        sys.set_position(k, pos)
    sys.set_csa(0, csa)
    for i, j, jh in couplings:
        sys.couple(i, j, jh)
    sys = sys.with_correlation_time(tau)

    def op(i, which):
        return qt.tensor([qt.jmat(0.5, which) if k == i else qt.qeye(2) for k in range(3)])

    gamma = {iso: q.spin.gyromagnetic_ratio(iso) for iso, *_ in spins}
    h = sum(-gamma[iso] * field * (1 + shift * 1e-6) * op(i, "z") for i, (iso, shift, _) in enumerate(spins))
    for i, j, jh in couplings:
        h += 2 * math.pi * jh * sum(op(i, w) * op(j, w) for w in "xyz")
    # Interactions Σ_ab A_ab Q_ab: dipolar pairs and the anisotropic shielding.
    interactions = []
    for i in range(3):
        for j in range(i + 1, 3):
            r = np.subtract(spins[j][2], spins[i][2]) * 1e-10
            dist = np.linalg.norm(r)
            b = 1.00000000055e-7 * gamma[spins[i][0]] * gamma[spins[j][0]] * 1.054571817e-34 / dist**3
            n = r / dist
            interactions.append((b * (np.eye(3) - 3 * np.outer(n, n)), lambda a, c, i=i, j=j: op(i, "xyz"[a]) * op(j, "xyz"[c])))
    aniso = (csa + csa.T) / 2 - np.trace(csa) / 3 * np.eye(3)
    interactions.append((gamma["13C"] * field * 1e-6 * aniso, lambda a, c: op(0, "xyz"[a]) if c == 2 else 0 * op(0, "z")))
    keys = [(k, a, c) for k in range(len(interactions)) for a in range(3) for c in range(3)]
    cov = np.zeros((len(keys), len(keys)))
    for rot, w in _rotations():
        v = np.array([(rot @ interactions[k][0] @ rot.T)[a, c] for k, a, c in keys])
        cov += w * np.outer(v, v)
    values, vectors = np.linalg.eigh(cov)
    spectrum = lambda w: 2 * tau / (1 + (w * tau) ** 2)
    a_ops = []
    for lam, vec in zip(values, vectors.T):
        if lam > 1e-12 * values.max():
            a = sum(x * interactions[k][1](a, c) for x, (k, a, c) in zip(vec, keys) if abs(x) > 1e-14)
            a_ops.append((math.sqrt(lam) * a, spectrum))
    tensor = qt.bloch_redfield_tensor(h, a_ops, sec_cutoff=-1, fock_basis=True)
    full = tensor.full()
    dissipation = full - qt.liouvillian(h).full()
    # Rates among observables.
    ops = [op(0, "z"), op(1, "z"), op(2, "z"), op(0, "+"), op(1, "+"), 2 * op(0, "z") * op(1, "z")]
    rates = sys.relaxation_rates([o.full() for o in ops])
    vec = lambda m: np.asarray(m).reshape(-1, order="F")
    want = np.array([[-(vec(oi.full()).conj() @ dissipation @ vec(oj.full())).real / (oi.dag() * oi).tr().real for oj in ops] for oi in ops])
    assert np.abs(rates - want).max() < 1e-9 * np.abs(want).max()
    assert want[0, 1] != 0 and want[0, 5] != 0  # cross-relaxation and cross-correlation
    # Dynamics: the exponential of QuTiP's tensor against evolve.
    rho0 = np.eye(8) / 8 + (0.1 * op(0, "z") + 0.05 * op(1, "x") + 0.03 * 2 * op(1, "z") * op(0, "z")).full()
    for t in (0.02, 0.2):
        ref = (scipy_linalg.expm(full * t) @ vec(rho0)).reshape(8, 8, order="F")
        got = sys.evolve(rho0, [spin.delay(t)])
        for o in ops[:3] + [op(1, "x")]:
            a, b = np.trace(got @ o.full()).real, np.trace(ref @ o.full()).real
            assert abs(a - b) < 1e-7 * 0.1, (t, a, b)


def test_redfield_bindings():
    sys = q.SpinSystem(9.4)
    c = sys.add_spin("13C", 170.0)
    h = sys.add_spin("1H", 8.0)
    sys.set_position(c, (0.0, 0.0, 0.0))
    sys.set_position(h, (1.1, 0.0, 0.0))
    sys.set_csa(c, spin.axial_csa(150.0, (0.0, 0.0, 1.0)))
    assert sys.position(h) == [1.1, 0.0, 0.0] and sys.position(c) == [0.0, 0.0, 0.0]
    assert sys.csa(c)[2, 2] == pytest.approx(100.0) and sys.csa(h) is None
    assert sys.correlation_time is None
    with pytest.raises(ValueError):
        sys.set_position(h, (0.0, 0.0, 0.0))
    assert sys.position(h) == [1.1, 0.0, 0.0]
    moving = sys.with_correlation_time(2e-11)
    assert moving.correlation_time == 2e-11
    assert moving.dipolar_coupling(0, 1) == pytest.approx(1.00000000055e-7 * 6.728284e7 * 26.7522128e7 * 1.054571817e-34 / 1.1e-10**3)
    g = moving.relaxation_rates([moving.spin_operator(0, "z"), moving.spin_operator(1, "z")])
    assert g.shape == (2, 2) and g[0, 0] > 0 and g[0, 1] > 0
    assert sys.relaxation_rates([sys.spin_operator(0, "z")])[0, 0] == 0.0
    back = q.SpinSystem.from_json(moving.to_json())
    assert back.correlation_time == 2e-11 and back.position(1) == [1.1, 0.0, 0.0]
    with pytest.raises(ValueError):
        sys.with_correlation_time(-1.0)


def test_exchange_matches_a_monte_carlo_of_exchange_events():
    # A SABRE complex (two hydrides and a bound 15N, laboratory frame near
    # the anti-crossing): hydrogen exchange renews the hydrides from
    # parahydrogen at kh, ligand exchange renews the 15N thermal at kd.
    # Each molecule evolves coherently between exponentially distributed
    # events (as in the simulations of Lindale et al. 2019), written here
    # with NumPy alone; the average over molecules is what Exchange
    # computes.
    rng = np.random.default_rng(7)
    field, kh, kd = 1.4e-7, 3.0, 2.0
    gamma = [q.spin.gyromagnetic_ratio(x) for x in ("1H", "1H", "15N")]
    shifts = (-22.0, -22.0, 250.0)
    sx = np.array([[0, 0.5], [0.5, 0]])
    sy = np.array([[0, -0.5j], [0.5j, 0]])
    sz = np.diag([0.5, -0.5])

    def op(i, s):
        mats = [np.eye(2)] * 3
        mats[i] = s
        return np.kron(np.kron(mats[0], mats[1]), mats[2])

    h = sum(-g * field * (1 + d * 1e-6) * op(i, sz) for i, (g, d) in enumerate(zip(gamma, shifts)))
    for i, j, jh in ((0, 1, -7.0), (0, 2, -2.0)):
        h = h + 2 * np.pi * jh * sum(op(i, s) @ op(j, s) for s in (sx, sy, sz))
    energies, vectors = np.linalg.eigh(h)

    def propagate(rho, t):
        u = vectors @ np.diag(np.exp(-1j * energies * t)) @ vectors.conj().T
        return u @ rho @ u.conj().T

    singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
    para = np.outer(singlet, singlet)
    n_thermal = np.eye(2) / 2  # 10⁻¹² polarisation at 0.14 μT

    def hydrides(rho):  # Tr_N
        return rho.reshape(4, 2, 4, 2).trace(axis1=1, axis2=3)

    def nitrogen(rho):  # Tr_H
        return rho.reshape(4, 2, 4, 2).trace(axis1=0, axis2=2)

    start = np.kron(para, n_thermal)
    times = (0.2, 0.6, 1.5)
    nz = op(2, sz)
    samples = np.zeros((4000, len(times)))
    for k in range(samples.shape[0]):
        rho, now = start, 0.0
        for m, t in enumerate(times):
            while True:
                wait = rng.exponential(1 / (kh + kd))
                if now + wait > t:
                    rho = propagate(rho, t - now)
                    now = t
                    break
                rho = propagate(rho, wait)
                now += wait
                if rng.random() < kh / (kh + kd):
                    rho = np.kron(para, nitrogen(rho))
                else:
                    rho = np.kron(hydrides(rho), n_thermal)
            samples[k, m] = np.trace(rho @ nz).real
    complex_ = q.SpinSystem(field, frame="laboratory")
    for d, x in zip(shifts, ("1H", "1H", "15N")):
        complex_.add_spin(x, d)
    complex_.couple(0, 1, -7.0)
    complex_.couple(0, 2, -2.0)
    x = spin.Exchange()
    x.add_site(complex_, 1.0)
    x.add_channel(0, 0, kh, carried=[(2, 2)], parahydrogen=(0, 1))
    x.add_channel(0, 0, kd, carried=[(0, 0), (1, 1)])
    states = [complex_.parahydrogen_state(0, 1)]
    mean, error = samples.mean(axis=0), samples.std(axis=0) / np.sqrt(samples.shape[0])
    for m, t in enumerate(times):
        got = x.evolve(states, [spin.delay(t)])[0]
        value = np.trace(got @ complex_.spin_operator(2, "z")).real
        assert abs(value - mean[m]) < 4 * error[m] + 1e-4, (t, value, mean[m], error[m])
    assert abs(mean[-1]) > 20 * error[-1]  # the polarisation is resolved


# An independent dense reference for Exchange: the homogeneous equations
# dσ_B/dt = L_B σ_B − Σ k σ_B + Σ k ρ_fresh ⊗ Tr_traced(σ_A) for σ_s = c_s ρ_s
# of all sites as one vector (row-major), with partial traces by einsum
# and spin orders by axis permutations. Only the Hamiltonian of each site
# comes from QSO.
_SPIN = {"1H": 0.5, "2H": 1.0, "13C": 0.5, "15N": 0.5}


def _ladder(i):
    m = np.arange(i, -i - 1, -1)
    return np.diag(m).astype(complex), np.diag(np.sqrt(i * (i + 1) - m[1:] * (m[1:] + 1)), 1).astype(complex)


class _Site:
    def __init__(self, spins, couplings, conc, field, frame="rotating"):
        self.spins, self.couplings, self.conc, self.field, self.frame = spins, couplings, conc, field, frame
        self.levels = [int(2 * _SPIN[s[0]] + 1) for s in spins]
        self.dim = int(np.prod(self.levels))

    def system(self, b=None):
        s = q.SpinSystem(self.field if b is None else b, frame=self.frame)
        for iso, shift, t1, t2 in self.spins:
            s.add_spin(iso, shift, t1=t1, t2=t2)
        for i, j, jh in self.couplings:
            s.couple(i, j, jh)
        return s

    def embed(self, i, m):
        out = np.eye(1)
        for k, n in enumerate(self.levels):
            out = np.kron(out, m if k == i else np.eye(n))
        return out

    def boltzmann(self, i, b):
        iso, shift = self.spins[i][:2]
        x = 1.054571817e-34 * q.spin.gyromagnetic_ratio(iso) * b * (1 + shift * 1e-6) / (1.380649e-23 * 298.15)
        m = np.arange(_SPIN[iso], -_SPIN[iso] - 1, -1)
        w = np.exp(x * m - np.max(x * m))
        return w / w.sum(), x

    def liouvillian(self, b):
        h = np.asarray(self.system(b).hamiltonian())
        e = np.eye(self.dim)
        out = -1j * (np.kron(h, e) - np.kron(e, h.T))
        for i, (iso, _, t1, t2) in enumerate(self.spins):
            iz, ip = _ladder(_SPIN[iso])
            jumps = []
            r1 = 0.0 if t1 is None else 1 / t1
            if r1 > 0:
                _, x = self.boltzmann(i, b)
                jumps += [math.sqrt(r1 / (1 + math.exp(-x))) * ip, math.sqrt(r1 / (1 + math.exp(x))) * ip.conj().T]
            if t2 is not None and 1 / t2 - r1 / 2 > 0:
                jumps.append(math.sqrt(2 * (1 / t2 - r1 / 2)) * iz)
            for a in (self.embed(i, j) for j in jumps):
                ada = a.conj().T @ a
                out += np.kron(a, a.conj()) - 0.5 * np.kron(ada, e) - 0.5 * np.kron(e, ada.T)
        return out


def _reorder(m, dims, order, target):
    """`m` over the spins `order` (with `dims`) as over the spins `target`."""
    n = len(order)
    perm = [order.index(s) for s in target]
    t = m.reshape(dims + dims).transpose(perm + [p + n for p in perm])
    return t.reshape(m.shape)


def _keep(m, dims, keep):
    n = len(dims)
    letters = "abcdefghijklmnopqrstuvwxyz"
    cols = [letters[i + n] if i in keep else letters[i] for i in range(n)]
    out = [letters[i] for i in keep] + [letters[i + n] for i in keep]
    d = int(np.prod([dims[i] for i in keep]))
    return np.einsum(letters[:n] + "".join(cols) + "->" + "".join(out), m.reshape(dims + dims)).reshape(d, d)


def _gains(a, b, carried, para, fraction, field):
    """Superoperator from σ_A to the gain of σ_B."""
    to = dict(carried)
    keep = sorted(to)
    fresh = [j for j in range(len(b.levels)) if j not in to.values()]
    factors, order = np.eye(1), []
    if para:
        singlet = np.array([0, 1, -1, 0]) / math.sqrt(2)
        s = np.outer(singlet, singlet)
        factors = fraction * s + (1 - fraction) / 3 * (np.eye(4) - s)
        order = list(para)
    for j in fresh:
        if j not in order:
            factors = np.kron(factors, np.diag(b.boltzmann(j, field)[0]))
            order.append(j)
    state = _reorder(factors, [b.levels[j] for j in order], order, fresh) if fresh else factors
    out = np.zeros((b.dim**2, a.dim**2), complex)
    for col in range(a.dim**2):
        e = np.zeros(a.dim**2, complex)
        e[col] = 1
        m = np.kron(_keep(e.reshape(a.dim, a.dim), a.levels, keep), state)
        spins = [to[i] for i in keep] + fresh
        out[:, col] = _reorder(m, [b.levels[j] for j in spins], spins, list(range(len(b.levels)))).reshape(-1)
    return out


def _generator(sites, channels, field):
    offsets = np.cumsum([0] + [s.dim**2 for s in sites])
    g = np.zeros((offsets[-1], offsets[-1]), complex)
    block = lambda s: slice(offsets[s], offsets[s + 1])
    for s, site in enumerate(sites):
        g[block(s), block(s)] += site.liouvillian(field)
    for f, t, k, carried, para, fraction in channels:
        g[block(f), block(f)] -= k * np.eye(sites[f].dim ** 2)
        g[block(t), block(f)] += k * _gains(sites[f], sites[t], carried, para, fraction, field)
    return g, offsets


def _expm(a):
    s = max(0, int(np.ceil(np.log2(max(np.abs(a).sum(axis=0).max(), 1e-300) / 0.25))))
    a = a / 2.0**s
    out = term = np.eye(len(a), dtype=complex)
    for k in range(1, 18):
        term = term @ a / k
        out = out + term
    for _ in range(s):
        out = out @ out
    return out


def _exchange_cases():
    def balanced(sites, a, b, k, there, back):
        return [(a, b, k, there, None, 1.0), (b, a, k * sites[a].conc / sites[b].conc, back, None, 1.0)]

    # Rotating frame, three sites with different isotopes; channels that
    # reorder spins, a channel within a site that swaps its protons, and
    # channels at rate 0.
    s = [
        _Site([("1H", 1.0, 2.0, 0.5), ("13C", 30.0, 5.0, None), ("1H", 3.0, None, None)], [(0, 1, 140.0), (0, 2, 7.0), (1, 2, -5.0)], 0.7, 2.0),
        _Site([("13C", 35.0, None, None), ("1H", 2.0, None, None), ("15N", 100.0, 3.0, None), ("1H", 4.5, None, 0.3)], [(0, 1, 150.0), (1, 3, 6.0), (2, 3, -90.0)], 1.3, 2.0),
        _Site([("15N", 120.0, None, None), ("1H", 8.0, 1.5, 1.0)], [(0, 1, -90.0)], 0.5, 2.0),
    ]
    ch = balanced(s, 0, 1, 7.0, [(2, 1), (1, 0)], [(0, 1), (3, 0)]) + balanced(s, 1, 2, 3.0, [(2, 0)], [(0, 2), (1, 1)])
    ch += [(0, 0, 1.5, [(1, 1), (2, 0), (0, 2)], None, 1.0), (0, 1, 0.0, [], None, 1.0), (1, 0, 0.0, [], None, 1.0)]
    yield "three sites", s, ch, "1H"
    # Parahydrogen on two spins that are not neighbours, in reverse order,
    # between carried spins, with fractions 0.8 and 1.
    s = [
        _Site([("15N", 200.0, 4.0, None), ("13C", 10.0, None, None)], [(0, 1, 12.0)], 2.0, 1.0),
        _Site([("1H", -20.0, None, 0.5), ("15N", 250.0, None, None), ("1H", -24.0, None, None), ("13C", 15.0, 10.0, None)], [(0, 2, -7.0), (0, 1, -20.0), (1, 2, 3.0), (2, 3, 4.0), (1, 3, 15.0)], 0.5, 1.0),
    ]
    ch = [(0, 1, 1.0, [(1, 3), (0, 1)], (2, 0), 0.8), (1, 0, 4.0, [(3, 1), (1, 0)], None, 1.0), (1, 1, 2.5, [(1, 1), (3, 3)], (0, 2), 1.0)]
    yield "parahydrogen", s, ch, "1H"
    # Laboratory frame at 0.3 mT, deuterium carried and fresh, a ¹³C only
    # in the second site.
    s = [
        _Site([("2H", 2.0, 1.0, None), ("1H", 1.0, None, None)], [(0, 1, 1.5)], 1.0, 3e-4, "laboratory"),
        _Site([("1H", 4.0, 3.0, None), ("2H", 2.5, None, 0.7), ("13C", 50.0, None, None)], [(0, 1, 2.0), (1, 2, 20.0), (0, 2, 140.0)], 3.0, 3e-4, "laboratory"),
    ]
    yield "laboratory", s, balanced(s, 0, 1, 5.0, [(0, 1), (1, 0)], [(1, 0), (0, 1)]), "13C"


def _build(sites, channels):
    x = spin.Exchange()
    for site in sites:
        x.add_site(site.system(), site.conc)
    for f, t, k, carried, para, fraction in channels:
        x.add_channel(f, t, k, carried=carried, parahydrogen=para, fraction=fraction if para else None)
    return x


@pytest.mark.parametrize("case", range(3))
def test_exchange_matches_a_dense_reference(case):
    name, sites, channels, isotope = list(_exchange_cases())[case]
    x = _build(sites, channels)
    field = sites[0].field
    g, offsets = _generator(sites, channels, field)
    pack = lambda states: np.concatenate([s.conc * np.asarray(r).reshape(-1) for s, r in zip(sites, states)])
    unpack = lambda v: [v[offsets[i] : offsets[i + 1]].reshape(s.dim, s.dim) / s.conc for i, s in enumerate(sites)]
    rng = np.random.default_rng(case)
    start = []
    for site in sites:
        m = rng.normal(size=(site.dim, site.dim)) + 1j * rng.normal(size=(site.dim, site.dim))
        m = m @ m.conj().T
        start.append(m / np.trace(m))

    def close(got, want, tol):
        scale = max(np.abs(w - np.trace(w) / len(w) * np.eye(len(w))).max() for w in want)
        diff = max(np.abs(np.asarray(a) - b).max() for a, b in zip(got, want))
        assert diff <= tol * scale + 1e-14, (name, diff, scale)

    want = unpack(_expm(g * 0.13) @ pack(start))
    close(x.evolve(start, [spin.delay(0.13)]), want, 1e-10)
    # A pulse on an isotope that not every site has, then a delay.
    pulsed = {"three sites": "15N", "parahydrogen": "1H", "laboratory": "13C"}[name]
    rotated = []
    for site, rho in zip(sites, start):
        f = sum((site.embed(i, _ladder(_SPIN[iso])[1]) for i, (iso, *_) in enumerate(site.spins) if iso == pulsed), np.zeros((site.dim, site.dim)))
        phase = math.radians(30)
        values, vectors = np.linalg.eigh((np.exp(-1j * phase) * f + np.exp(1j * phase) * f.conj().T) / 2)
        u = vectors @ np.diag(np.exp(-1j * math.radians(90) * values)) @ vectors.conj().T
        rotated.append(u @ rho @ u.conj().T)
    want = unpack(_expm(g * 0.01) @ pack(rotated))
    close(x.evolve(start, [spin.pulse(pulsed, 90, 30), spin.delay(0.01)]), want, 1e-10)
    # A sweep: steps at the field of their middle, then a delay.
    v = pack(start)
    for k in range(7):
        v = _expm(_generator(sites, channels, field * (1 - 0.7 * (k + 0.5) / 7))[0] * (0.2 / 7)) @ v
    v = _expm(_generator(sites, channels, 0.3 * field)[0] * 0.05) @ v
    close(x.evolve(start, [spin.sweep(0.3 * field, 0.2, 7), spin.delay(0.05)]), unpack(v), 1e-10)
    # The FID of a state with coherences of the isotope.
    fplus = [sum((site.embed(i, _ladder(_SPIN[iso])[1]) for i, (iso, *_) in enumerate(site.spins) if iso == isotope), np.zeros((site.dim, site.dim))) for site in sites]
    step, v, fid = _expm(g * 1e-3), pack(start), []
    for k in range(40):
        fid.append(sum(np.trace(r * s.conc @ f) for r, s, f in zip(unpack(v), sites, fplus)))
        v = step @ v
    got = np.asarray(x.fid(start, isotope, 1e-3, 40))
    assert np.abs(got - fid).max() < 1e-11 * np.abs(fid).max(), name
    if sites[0].frame == "rotating":
        # The steady state: the population equation of each site replaced
        # by its amount.
        a, rhs = g.copy(), np.zeros(len(g), complex)
        for s, site in enumerate(sites):
            first = offsets[s]
            a[first] = 0
            a[first, [offsets[s] + i * (site.dim + 1) for i in range(site.dim)]] = 1
            rhs[first] = site.conc
        close(x.steady_state(), unpack(np.linalg.solve(a, rhs)), 1e-9)


def test_exchange_steady_state_in_extended_precision():
    # SABRE in the laboratory frame from millitesla to 9.4 T, where
    # zero-quantum frequencies exceed the rates by up to 10⁹: the
    # homogeneous equations solved with 40 digits.
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40
    for field in (1e-3, 1.0, 9.4):
        sites = [
            _Site([("1H", -22.0, None, None), ("1H", -22.0, None, None), ("15N", 250.0, None, None)], [(0, 1, -7.0), (0, 2, -2.0)], 1.0, field, "laboratory"),
            _Site([("15N", 310.0, 30.0, None)], [], 10.0, field, "laboratory"),
        ]
        channels = [(0, 1, 2.0, [(2, 0)], None, 1.0), (1, 0, 0.2, [(0, 2)], (0, 1), 1.0), (0, 0, 0.5, [(2, 2)], (0, 1), 1.0)]
        g, offsets = _generator(sites, channels, field)
        a = mp.matrix([[mp.mpc(z.real, z.imag) for z in row] for row in g])
        rhs = mp.matrix(len(g), 1)
        for s, site in enumerate(sites):
            first = int(offsets[s])
            for j in range(len(g)):
                a[first, j] = 0
            for i in range(site.dim):
                a[first, first + i * (site.dim + 1)] = 1
            rhs[first, 0] = site.conc
        v = mp.lu_solve(a, rhs)
        free = int(offsets[1])
        want = float(mp.re(v[free, 0] - v[free + 3, 0])) / 10.0
        x = _build(sites, channels)
        got = x.polarization(x.steady_state(), 1, 0)
        # To 10⁻¹⁴ of the hydrides' parahydrogen order: at 1 mT that is
        # 10⁻⁵ of the nitrogen's polarisation (precision, not physics).
        assert abs(got - want) < 1e-14, (field, got, want)


def test_exchange_and_singlet_bindings():
    a = q.SpinSystem(9.4)
    a.add_spin("1H", 1.0, t1=2.0)
    b = q.SpinSystem(9.4)
    b.add_spin("1H", 1.0, t1=3.0)
    x = spin.Exchange()
    x.add_site(a, 0.5)
    x.add_site(b, 0.5)
    x.add_channel(0, 1, 40.0, carried=[(0, 0)])
    x.add_channel(1, 0, 40.0, carried=[(0, 0)])
    eq = x.thermal_state()
    start = x.evolve(eq, [spin.pulse("1H", 90)])
    fid = x.fid(start, "1H", 1e-4, 64)
    assert fid.shape == (64,) and abs(fid[0]) > 0
    steady = x.steady_state()
    assert np.abs(steady[0] - eq[0]).max() < 1e-12
    assert x.polarization(eq, 1, 0) == pytest.approx(a.polarization(a.thermal_state(), 0), rel=1e-12)
    # Without relaxation the populations of the two sites have no unique
    # steady state.
    still = spin.Exchange()
    for system in (q.SpinSystem(9.4), q.SpinSystem(9.4)):
        system.add_spin("1H", 1.0)
        still.add_site(system, 1.0)
    still.add_channel(0, 1, 5.0, carried=[(0, 0)])
    still.add_channel(1, 0, 5.0, carried=[(0, 0)])
    with pytest.raises(ValueError, match="unique"):
        still.steady_state()
    assert spin.Exchange.from_json(x.to_json()).sites[1][1] == 0.5
    with pytest.raises(ValueError, match="steady state"):
        y = spin.Exchange()
        y.add_site(a, 1.0)
        y.add_site(b, 1.0)
        y.add_channel(0, 1, 1.0, carried=[(0, 0)])
        y.validate()
    with pytest.raises(ValueError):
        x.evolve(eq[:1], [])
    with pytest.raises(ValueError, match="parahydrogen"):
        x.add_channel(0, 1, 1.0, carried=[(0, 0)], fraction=0.5)
    pair = q.SpinSystem(1.0)
    pair.add_spin("1H", 0.0)
    pair.add_spin("1H", 0.02)
    pair.couple(0, 1, 8.0)
    events = spin.m2s("1H", 8.0, 0.85)
    assert len(spin.s2m("1H", 8.0, 0.85)) == len(events)
    lock = spin.spin_lock("1H", 1000.0, 0.01)
    assert "Shaped" in repr(lock)


def _expm_hermitian(h, t):
    values, vectors = np.linalg.eigh(h)
    return vectors @ np.diag(np.exp(-1j * values * t)) @ vectors.conj().T


def test_control_matches_an_independent_propagation_and_its_gradient():
    # rf on ¹H and ¹³C with an offset member: the transfer from NumPy's own
    # propagation of H_k = H₀ + 2π Σ (x F_x + y F_y), and the gradient
    # against central differences.
    sys = q.SpinSystem(9.4)
    sys.add_spin("1H", 0.5)
    sys.add_spin("13C", 0.0)
    sys.couple(0, 1, 140.0)
    iz, sz = sys.spin_operator(0, "z"), sys.spin_operator(1, "z")
    c = spin.Control.rf(["1H", "13C"], 3e-3, 6, iz, sz, limit=2000.0)
    u = c.random_start(seed=4, scale=800.0)
    value, grad = sys.control_transfer(c, u)
    h0 = np.asarray(sys.hamiltonian())
    ops = [sys.spin_operator(0, "x"), sys.spin_operator(0, "y"), sys.spin_operator(1, "x"), sys.spin_operator(1, "y")]
    rho = iz.copy()
    for k in range(6):
        h = h0 + 2 * np.pi * sum(a * o for a, o in zip(u[4 * k : 4 * k + 4], ops))
        w = _expm_hermitian(h, 0.5e-3)
        rho = w @ rho @ w.conj().T
    assert value == pytest.approx(np.trace(sz.conj().T @ rho).real, abs=1e-12)
    for i in range(len(u)):
        up, down = u.copy(), u.copy()
        up[i] += 1e-3
        down[i] -= 1e-3
        fd = (sys.control_transfer(c, up)[0] - sys.control_transfer(c, down)[0]) / 2e-3
        assert abs(fd - grad[i]) < 1e-7 * np.abs(grad).max()
    events = sys.control_events(c, u)
    assert len(events) == 1 and "Control" in repr(events[0])
    after = sys.evolve(iz, events)
    assert np.abs(after - rho).max() < 1e-12
    same = sys.evolve(iz, [spin.control(["1H", "13C"], 0.5e-3, [[(u[4 * k], u[4 * k + 1]), (u[4 * k + 2], u[4 * k + 3])] for k in range(6)])])
    assert np.abs(same - rho).max() < 1e-12


def test_grape_reaches_the_unitary_bound_and_profiles_bindings():
    sys = q.SpinSystem(9.4)
    sys.add_spin("1H", 0.0)
    sys.add_spin("13C", 0.0)
    sys.couple(0, 1, 140.0)
    rho = sys.thermal_state()
    sz = sys.spin_operator(1, "z")
    bound = spin.unitary_bound(rho, sz)
    eig_rho = np.sort(np.linalg.eigvalsh(rho))[::-1]
    eig_sz = np.sort(np.linalg.eigvalsh(sz))[::-1]
    assert bound == pytest.approx(float(eig_rho @ eig_sz), rel=1e-12)
    c = spin.Control.rf(["1H", "13C"], 1.2 / 140, 40, rho, sz, limit=5000.0)
    best = max((sys.optimize_control(c, seed=s, scale=500.0, max_iterations=400) for s in range(3)), key=lambda r: r["transfer"])
    assert bound * (1 - 1e-6) <= best["transfer"] <= bound * (1 + 1e-9)
    assert best["controls"].shape == (160,) and isinstance(best["converged"], bool)
    c.ensemble = [(1.0, [0.9, 0.9], []), (1.0, [1.1, 1.1], [])]
    c.power, c.smoothness = 1e-12, 1e-12
    assert len(c.ensemble) == 2 and c.power == 1e-12 and len(c) == 160
    with pytest.raises(ValueError):
        sys.control_transfer(c, np.zeros(3))
    with pytest.raises(ValueError):
        spin.unitary_bound(np.eye(2), np.eye(3))
    # Field profiles of an exchange against evolve.
    a = q.SpinSystem(1e-6, frame="laboratory")
    a.add_spin("1H", 1.0, t1=2.0)
    a.add_spin("13C", 30.0, t1=5.0)
    a.couple(0, 1, 140.0)
    x = spin.Exchange()
    x.add_site(a, 1.0)
    start = [a.thermal_state()]
    target = [a.spin_operator(1, "z")]
    p = spin.Profile(0.0, 2e-6, 0.4, 4, start, target)
    fields = np.array([0.5e-6, 1e-6, 1.5e-6, 0.2e-6])
    value, grad = x.profile_transfer(p, fields)
    events = [e for b in fields for e in (spin.set_field(b), spin.delay(0.1))]
    want = np.trace(x.evolve(start, events)[0] @ target[0]).real
    assert value == pytest.approx(want, abs=1e-15)
    assert grad.shape == (4,)
    r = x.optimize_profile(p, fields, max_iterations=5)
    assert r["controls"].shape == (4,) and r["transfer"] >= value - 1e-18
