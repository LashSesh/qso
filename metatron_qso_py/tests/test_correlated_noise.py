"""Channels on more qubits, crosstalk on spectators, and noise correlated in time."""

import math

import numpy as np
import pytest

import metatron_qso as q


def test_three_and_four_qubit_channels():
    ch = q.Channel.pauli({"ZZZ": 0.1, "XIX": 0.05}).compose(q.Channel.rotation("XYZ", 0.3))
    assert ch.n_qubits == 3
    rho = q.DensityMatrix.zero(4)
    rho.apply_channel(q.Channel.rotation("ZZZZ", 0.2), [0, 1, 2, 3])
    rho.apply_channel(ch, [3, 0, 1])
    assert abs(rho.trace() - 1) < 1e-12
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
    with pytest.raises(ValueError, match="trajectories"):
        q.DensityMatrix.simulate(q.Circuit(1).h(0), device=device)
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
