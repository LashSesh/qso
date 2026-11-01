"""The generalised cluster-correlation expansion of an NV centre's bath."""

import math

import numpy as np
import pytest

from metatron_qso import spin

MU0_4PI = 1.00000000055e-7
H = 6.62607015e-34
GAMMA_E = 28.025e9
GAMMA_C = 6.728284e7 / (2 * math.pi)
D = 2.87e9

S1 = 1 / math.sqrt(2)
SX1 = S1 * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], complex)
SY1 = S1 * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], complex)
SZ1 = np.diag([1.0, 0.0, -1.0]).astype(complex)
IX = np.array([[0, 0.5], [0.5, 0]], complex)
IY = np.array([[0, -0.5j], [0.5j, 0]], complex)
IZ = np.array([[0.5, 0], [0, -0.5]], complex)

POSITIONS = [
    [0.6e-9, 0.2e-9, 0.4e-9],
    [0.9e-9, 0.1e-9, 0.1e-9],
    [0.5e-9, 0.6e-9, 0.0],
]


def nuclear(k: int, i: int, op: np.ndarray) -> np.ndarray:
    out = np.eye(1, dtype=complex)
    for j in range(k):
        out = np.kron(out, op if j == i else np.eye(2))
    return out


def reference(field, spins, events, laboratory: bool):
    """An independent NumPy propagation of the electron (spin 1) and the
    nuclei in the laboratory frame: the signal after the ideal last pulse
    and the electron's density matrix before it (in its eigenbasis), with
    ideal pulses about axes fixed in the laboratory or in the frame rotating
    at the bare qubit's transition."""
    k = len(spins)
    n = 2**k
    ee = 2 * math.pi * (D * SZ1 @ SZ1 + GAMMA_E * (field[0] * SX1 + field[1] * SY1 + field[2] * SZ1))
    w, v = np.linalg.eigh(ee)
    alpha = int(np.argmax(abs(v[1, :])))
    beta = int(np.argmax([abs(v[2, j]) if j != alpha else -1 for j in range(3)]))
    h = np.kron(ee, np.eye(n))
    s_ops, i_ops = [SX1, SY1, SZ1], [IX, IY, IZ]
    for i, (_, a) in enumerate(spins):
        for c in range(3):
            h -= 2 * math.pi * GAMMA_C * field[c] * np.kron(np.eye(3), nuclear(k, i, i_ops[c]))
            for b in range(3):
                h += 2 * math.pi * a[c][b] * np.kron(s_ops[c], nuclear(k, i, i_ops[b]))
    for i in range(k):
        for j in range(i + 1, k):
            r = np.array(spins[i][0]) - np.array(spins[j][0])
            dist = np.linalg.norm(r)
            u = r / dist
            dd = MU0_4PI * GAMMA_C**2 * H / dist**3
            for a in range(3):
                for b in range(3):
                    coupling = 2 * math.pi * dd * ((a == b) - 3 * u[a] * u[b])
                    h += coupling * np.kron(np.eye(3), nuclear(k, i, i_ops[a]) @ nuclear(k, j, i_ops[b]))
    hw, hv = np.linalg.eigh(h)
    f = (w[beta] - w[alpha]) / (2 * math.pi)
    chi = np.ones(3)
    chi[alpha] = 0.0

    def frame(t):
        # The frame's rotation in the bare basis: V diag(e^{-2πi f t χ}) V†.
        return v @ np.diag(np.exp(-2j * math.pi * f * t * chi)) @ v.conj().T

    def ideal(angle, phase):
        c, s = math.cos(math.radians(angle) / 2), math.sin(math.radians(angle) / 2)
        p = np.exp(-1j * math.radians(phase))
        r = np.eye(3, dtype=complex)
        r[beta, beta] = r[alpha, alpha] = c
        r[beta, alpha] = -1j * s * p
        r[alpha, beta] = -1j * s * np.conj(p)
        return v @ r @ v.conj().T

    rho = np.kron(np.outer(v[:, alpha], v[:, alpha].conj()), np.eye(n) / n)
    t = 0.0
    for e in events[:-1]:
        if e.kind == "free":
            u = (hv * np.exp(-1j * hw * e.duration)) @ hv.conj().T
            t += e.duration
        else:
            r = ideal(e.angle, e.phase)
            if not laboratory:
                r = frame(t) @ r @ frame(t).conj().T
            u = np.kron(r, np.eye(n))
        rho = u @ rho @ u.conj().T
    rho_e = rho.reshape(3, n, 3, n).trace(axis1=1, axis2=3)
    if not laboratory:
        rho_e = frame(t).conj().T @ rho_e @ frame(t)
    rho_e = v.conj().T @ rho_e @ v
    last = v.conj().T @ ideal(events[-1].angle, events[-1].phase) @ v
    signal = (last @ rho_e @ last.conj().T)[alpha, alpha].real
    return signal, rho_e, alpha, beta


def bath(field, count=3, **kw):
    spins = [(p, spin.point_dipole_tensor(p)) for p in POSITIONS[:count]]
    return spin.GeneralisedBath(field, spins, **kw), spins


@pytest.mark.parametrize("pulses", ["ideal", "laboratory"])
def test_exact_and_expansion_against_numpy(pulses):
    field = (0.005 * math.sin(0.3), 0.0, 0.005 * math.cos(0.3))
    b, spins = bath(field)
    seqs = [spin.hahn(t) for t in (50e-6, 200e-6, 400e-6)] + [spin.cpmg(2, 0.1e-3)]
    coherence, signal = b.exact(seqs, pulses=pulses)
    full = b.signal(seqs, order=3, cutoff=math.inf, pulses=pulses)
    expanded = b.coherence(seqs, order=3, cutoff=math.inf, pulses=pulses)
    # The reference diagonalises the Hamiltonian in the laboratory frame
    # (D = 2.87 GHz): its phases after the 800 µs echo carry a rounding of
    # ε‖H‖t ≈ 3·10⁻⁹, and it differs between platforms (8·10⁻¹⁰ on x86-64,
    # 1.2·10⁻⁹ on aarch64).
    tol = 1e-8
    for k, events in enumerate(seqs):
        s, rho_e, alpha, beta = reference(field, spins, events, pulses == "laboratory")
        _, rho_0, _, _ = reference(field, [], events, pulses == "laboratory")
        assert abs(signal[k] - s) < tol
        assert abs(full[k] - s) < tol
        # The magnitude of the qubit's coherence relative to the electron
        # alone (|ρ_αβ| = |ρ_βα|: the branches' order does not matter).
        ratio = abs(rho_e[beta, alpha] / rho_0[beta, alpha])
        assert abs(abs(coherence[k]) - ratio) < tol
        assert abs(expanded[k] - coherence[k]) < 1e-10


def test_secular_tensors_reproduce_spin_bath():
    secular = spin.SpinBath(0.05, [(p, spin.point_dipole_hyperfine(p)) for p in POSITIONS])
    rows = [(p, [[0.0] * 3, [0.0] * 3, list(row)]) for p, row in secular.spins]
    general = spin.GeneralisedBath((0.0, 0.0, 0.05), rows)
    seqs = [spin.hahn(t) for t in (0.1e-3, 0.5e-3)]
    for order in (1, 2, 3):
        a = secular.coherence(seqs, order=order, cutoff=math.inf)
        b = general.coherence(seqs, order=order, cutoff=math.inf)
        assert np.abs(a - b).max() < 1e-8
    converted = spin.GeneralisedBath.from_bath(secular)
    assert converted.num_spins == 3 and not converted.dressed
    for (_, tensor), (_, row) in zip(converted.spins, secular.spins):
        assert np.allclose(tensor[2], row)


def test_pulses_frames_and_inputs():
    b, _ = bath((0.0, 0.0, 0.3))
    seqs = [spin.hahn(0.2e-3)]
    ideal = b.signal(seqs, cutoff=math.inf)
    finite = b.signal(seqs, cutoff=math.inf, rabi=16e6)
    gaussian = [(math.exp(-((x - 7.5) / 4) ** 2 / 2), 0.0) for x in range(16)]
    shaped = b.signal(seqs, cutoff=math.inf, rabi=16e6, envelope=gaussian)
    assert abs(finite[0] - ideal[0]) < 1e-3 and abs(shaped[0] - ideal[0]) < 1e-3
    assert finite[0] != shaped[0]
    rectangular = b.signal(seqs, cutoff=math.inf, pulses="finite", rabi=16e6, envelope=[(1.0, 0.0)] * 4)
    assert abs(rectangular[0] - finite[0]) < 1e-10
    dressed = b.with_dressed(True)
    assert dressed.dressed and "dressed" in repr(dressed)
    assert abs(dressed.signal(seqs, cutoff=math.inf)[0] - ideal[0]) < 1e-4
    assert b.frame_frequency() == pytest.approx(D - GAMMA_E * 0.3, rel=1e-6)
    for bad in (
        dict(pulses="finite"),
        dict(pulses="ideal", rabi=1e6),
        dict(pulses="laboratory", envelope=[(1.0, 0.0)]),
        dict(pulses="square", rabi=1e6),
        dict(rabi=-1.0),
        dict(rabi=1e6, envelope=[]),
        dict(rabi=1e6, envelope=[(-1.0, 0.0)]),
        dict(order=9),
    ):
        with pytest.raises(ValueError):
            b.signal(seqs, **bad)
    with pytest.raises(ValueError):
        spin.GeneralisedBath((math.nan, 0.0, 0.0), [])
    with pytest.raises(ValueError):
        spin.GeneralisedBath((0.0, 0.0, 0.05), [], transition=0)
    with pytest.raises(ValueError):
        b.signal([[spin.nv_free(1e-6)]])
    # At the crossing of m_s = 0 and -1 a nucleus 0.25 nm from the vacancy
    # mixes the levels: the dressed levels do not exist.
    near = [[0.2e-9, 0.1e-9, 0.12e-9]] + POSITIONS
    crossing = spin.GeneralisedBath(
        (0.0, 0.0, 0.1024), [(p, spin.point_dipole_tensor(p)) for p in near], dressed=True
    )
    with pytest.raises(ValueError):
        crossing.exact(seqs)
    assert crossing.with_dressed(False).exact(seqs)[1].shape == (1,)


def test_point_dipole_tensor_and_baths():
    p = [0.7e-9, -0.4e-9, 0.9e-9]
    a = np.array(spin.point_dipole_tensor(p))
    assert np.allclose(a, a.T) and abs(np.trace(a)) < 1e-6
    assert np.allclose(a[2], spin.point_dipole_hyperfine(p))
    with pytest.raises(ValueError):
        spin.point_dipole_tensor([0.0, 0.0, 0.0])
    tilted = spin.GeneralisedBath.diamond((0.01, 0.0, 0.03), radius=1.5e-9, seed=3)
    same = spin.SpinBath.diamond(0.03, radius=1.5e-9, seed=3)
    assert tilted.num_spins == same.num_spins
    assert tilted.without_spins_within(1e-9).num_spins < tilted.num_spins
    seqs = [spin.hahn(t) for t in (0.1e-3, 0.3e-3)]
    s = tilted.signal(seqs)
    assert np.all((s > 0) & (s <= 1))
    mean, err = spin.generalised_ensemble_signal((0.01, 0.0, 0.03), seqs, 3, radius=1.5e-9, seed=2)
    assert mean.shape == (2,) and np.all(err >= 0)
    with pytest.raises(ValueError):
        spin.generalised_ensemble_signal((0.0, 0.0, 0.05), seqs, 0)
    assert spin.MAX_GENERALISED_ORDER == 8 and spin.MAX_GENERALISED_EXACT == 8
    assert spin.MAX_ENVELOPE_SEGMENTS == 1024


def test_sampled_bath_states_with_finite_pulses():
    # Sampled product states put the detuning by all nuclei into every
    # cluster: with finite pulses their mean matches the exact propagation.
    b, _ = bath((0.0, 0.0, 0.05))
    seqs = [spin.cpmg(8, 10e-6), spin.cpmg(16, 10e-6)]
    _, exact = b.exact(seqs, rabi=50e6)
    est = b.estimate(seqs, order=2, cutoff=math.inf, rabi=50e6, bath_states=300, seed=1)
    assert set(est) == {"coherence", "coherence_error", "signal", "signal_error"}
    assert np.all(est["signal_error"] > 0)
    assert np.all(np.abs(est["signal"] - exact) < 5 * est["signal_error"] + 2e-4)
    assert np.allclose(b.signal(seqs, order=2, cutoff=math.inf, rabi=50e6, bath_states=300, seed=1), est["signal"])
    plain = b.estimate(seqs, cutoff=math.inf)
    assert np.all(plain["signal_error"] == 0.0)
    assert math.isnan(b.estimate(seqs, bath_states=1)["signal_error"][0])
    with pytest.raises(ValueError):
        b.signal(seqs, bath_states=spin.MAX_BATH_STATES + 1)


def test_review_fixes():
    b, spins = bath((0.0, 0.0, 0.05))
    # Inner pulses that leave the electron alone at a pole before the read
    # pulse: exact at full order (the expansion runs in the carried basis).
    pole = [[spin.nv_pulse(90), spin.nv_free(20e-6), spin.nv_pulse(90), spin.nv_pulse(90)]]
    _, exact = b.exact(pole)
    assert abs(b.signal(pole, order=3, cutoff=math.inf)[0] - exact[0]) < 1e-10
    # Ramsey read out by an identity pulse.
    ramsey_read = [[spin.nv_pulse(90), spin.nv_free(20e-6), spin.nv_pulse(90, 180), spin.nv_pulse(0)]]
    _, exact = b.exact(ramsey_read)
    assert abs(b.signal(ramsey_read, order=3, cutoff=math.inf)[0] - exact[0]) < 1e-10
    # No coherence between the branches after a first π pulse: NaN.
    flip = [[spin.nv_pulse(180), spin.nv_free(10e-6), spin.nv_pulse(90)]]
    assert math.isnan(b.coherence(flip, cutoff=math.inf)[0].real)
    # Fields across the axis mix m_s = ±1: refused.
    with pytest.raises(ValueError):
        spin.GeneralisedBath((0.01, 0.0, 0.0), spins)
    # The rotating-wave approximation: at most a tenth of the frame's frequency.
    near = spin.GeneralisedBath((0.0, 0.0, 0.1024), spins)
    with pytest.raises(ValueError):
        near.signal([spin.hahn(20e-6)], rabi=1e6)
    with pytest.raises(ValueError):
        b.signal([spin.hahn(20e-6)], rabi=1e9)
    # Radii that are not numbers, or negative, are refused.
    for radius in (math.nan, -1e-9):
        with pytest.raises(ValueError):
            b.without_spins_within(radius)
        with pytest.raises(ValueError):
            spin.GeneralisedBath.diamond((0.0, 0.0, 0.05), radius=1.5e-9, min_radius=radius)
        with pytest.raises(ValueError):
            spin.SpinBath.diamond(0.05, radius=1.5e-9, min_radius=radius)
