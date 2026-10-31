"""Channels on more qubits, crosstalk on spectators, and noise correlated in time."""

import math

import numpy as np
import pytest

import metatron_qso as q


def embed(kraus, qubits, n):
    # Full matrix of a local operator on `qubits` (first = most significant).
    dim = 1 << n
    k = len(qubits)
    out = np.zeros((dim, dim), complex)
    for r in range(dim):
        for c in range(dim):
            if all((r >> q) & 1 == (c >> q) & 1 for q in range(n) if q not in qubits):
                lr = sum(((r >> q) & 1) << (k - 1 - j) for j, q in enumerate(qubits))
                lc = sum(((c >> q) & 1) << (k - 1 - j) for j, q in enumerate(qubits))
                out[r, c] = kraus[lr, lc]
    return out


def test_three_and_four_qubit_channels():
    ad = q.Channel.amplitude_damping(0.3)
    ch = q.Channel.pauli({"ZZZ": 0.1, "XIX": 0.05}).compose(q.Channel.rotation("XYZ", 0.3))
    assert ch.n_qubits == 3
    circuit = q.Circuit(4)
    circuit.h(0).ry(1, 0.7).cx(0, 2).rx(3, 1.1)
    rho = q.DensityMatrix.simulate(circuit)
    rho.apply_channel(ad, [1])
    before = rho.to_matrix()
    rho.apply_channel(ch, [3, 0, 1])
    four = q.Channel.rotation("ZXZY", 0.4)
    rho.apply_channel(four, [2, 0, 3, 1])
    expected = sum(embed(k, [3, 0, 1], 4) @ before @ embed(k, [3, 0, 1], 4).conj().T
                   for k in ch.kraus_operators())
    u = embed(four.kraus_operators()[0], [2, 0, 3, 1], 4)
    expected = u @ expected @ u.conj().T
    assert np.abs(rho.to_matrix() - expected).max() < 1e-12
    with pytest.raises(ValueError):
        q.Channel.pauli({"ZZZZZ": 0.1})


def test_spectator_crosstalk_matches_the_density_matrix():
    zz = q.Channel.rotation("ZZ", 0.3)
    device = q.DeviceModel("spectator").with_rule(gate="cx", spectators=[([1, 2], zz)])
    c = q.Circuit(3)
    c.h(0).h(2).cx(0, 1).h(2)
    rho = q.DensityMatrix.simulate(c, device=device)
    first = q.Circuit(3)
    first.h(0).h(2).cx(0, 1)
    expected = q.DensityMatrix.simulate(first)
    expected.apply_channel(zz, [1, 2])
    last = q.Circuit(3)
    last.h(2)
    expected.apply(last)
    assert np.abs(rho.to_matrix() - expected.to_matrix()).max() < 1e-13


def ramsey(device, n=4000):
    c = q.Circuit(1)
    c.h(0).id(0).h(0)
    return q.TrajectorySimulator(device, n, 7).expectation(c, q.PauliSum("Z0"))


def test_quasi_static_detuning_and_telegraph_noise():
    sigma, t = 0.8, 1.5
    device = (
        q.DeviceModel("ramsey")
        .with_default_qubit(detuning=("gaussian", sigma))
        .with_rule(gate="id", duration=t)
    )
    assert not device.is_markovian
    e = ramsey(device)
    assert abs(e.value - math.exp(-((sigma * t) ** 2) / 2)) < 4 * e.std_error
    # Density matrices average the same noise exactly (1.44).
    c = q.Circuit(1)
    c.h(0).id(0).h(0)
    exact = q.DensityMatrix.simulate(c, device=device).expectation(q.PauliSum("Z0"))
    assert abs(exact - math.exp(-((sigma * t) ** 2) / 2)) < 1e-12
    nu, gamma = 1.0, 5.0
    telegraph = (
        q.DeviceModel("telegraph")
        .with_default_qubit(telegraph=(nu, gamma))
        .with_rule(gate="id", duration=t)
    )
    mu = math.sqrt(gamma**2 - nu**2)
    exact = math.exp(-gamma * t) * (math.cosh(mu * t) + gamma / mu * math.sinh(mu * t))
    e = ramsey(telegraph)
    assert abs(e.value - exact) < 4 * e.std_error
    with pytest.raises(ValueError, match="spread"):
        q.DeviceModel().with_default_qubit(detuning=("cauchy", 1.0))


def test_over_rotation_spread_and_json():
    device = q.DeviceModel("spread").with_rule(
        gate="rx", over_rotation_spread=("uniform", 0.2)
    )
    c = q.Circuit(1)
    c.rx(0, 1.0)
    e = q.TrajectorySimulator(device, 4000, 3).expectation(c, q.PauliSum("Z0"))
    # ⟨cos(1 + η)⟩ for η uniform on [−0.2, 0.2].
    exact = math.cos(1.0) * math.sin(0.2) / 0.2
    assert abs(e.value - exact) < 4 * e.std_error
    again = q.DeviceModel.from_json(device.to_json())
    assert again.to_json() == device.to_json()
    assert q.DeviceModel().is_markovian
