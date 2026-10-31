"""Slow noise averaged exactly (milestone 1.44)."""

import cmath
import math

import numpy as np
import pytest

import metatron_qso as q


def telegraph_free(a, g, t):
    """Kubo–Anderson free induction under telegraph noise ±a, rate g."""
    mu = cmath.sqrt(g * g - a * a)
    slow, fast = cmath.exp(-a * a / (mu + g) * t), cmath.exp(-(mu + g) * t)
    return (0.5 * ((1 + g / mu) * slow + (1 - g / mu) * fast)).real


def test_telegraph_and_ornstein_uhlenbeck_processes():
    for a, g in [(1.0, 0.2), (1.0, 3.0)]:
        p = q.MarkovProcess.telegraph(a, g)
        for t in [0.5, 2.0]:
            assert abs(p.dephasing(t) - telegraph_free(a, g, t)) < 1e-12
    assert p.num_states == 2 and p.initial == [0.5, 0.5]
    assert p.rates == [(0, 1, 3.0), (1, 0, 3.0)]
    # The Ehrenfest chain of 64 fluctuators is their product exactly, and
    # its correlation is exponential.
    sigma, tc, t = 1.3, 0.7, 2.0
    chain = q.MarkovProcess.ornstein_uhlenbeck(sigma, tc, 65)
    one = q.MarkovProcess.telegraph(sigma / 8, 0.5 / tc)
    assert abs(chain.dephasing(t, [t / 2]) - one.dephasing(t, [t / 2]) ** 64) < 1e-12
    assert abs(chain.correlation(0.3) - sigma**2 * math.exp(-0.3 / tc)) < 1e-12
    assert abs(chain.mean(1.0)) < 1e-12


def test_fluctuators_and_quasi_static_noise():
    noise = q.Fluctuators.one_over_f(0.3, 0.01, 10.0, 5)
    assert len(noise) == 5 and len(noise.processes) == 5
    joint = noise.combined()
    assert joint.num_states == 32
    assert abs(noise.dephasing(4.0) - joint.dephasing(4.0)) < 1e-12
    assert abs(noise.correlation(1.0) - joint.correlation(1.0)) < 1e-12
    p = q.MarkovProcess.quasi_static("gaussian", 0.8, 40)
    assert abs(p.dephasing(2.0) - math.exp(-0.32 * 4)) < 1e-12
    assert abs(p.dephasing(2.0, [1.0]) - 1.0) < 1e-12
    u = q.MarkovProcess.quasi_static("uniform", 1.1, 20)
    assert abs(u.dephasing(3.0) - math.sin(3.3) / 3.3) < 1e-12
    with pytest.raises(ValueError):
        q.MarkovProcess.quasi_static("cauchy", 1.0, 4)
    with pytest.raises(ValueError):
        q.MarkovProcess([0.0, 1.0], [(0, 1, 1.0)], [0.5, 0.6])
    with pytest.raises(ValueError):
        p.dephasing(1.0, [2.0])


def test_stochastic_liouville_equation():
    z = np.diag([0.5, -0.5]).astype(complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    zero = np.zeros((2, 2))
    p = q.MarkovProcess.ornstein_uhlenbeck(0.8, 0.5, 8)
    sle = q.StochasticLiouville(2, [(p, z)])
    assert (sle.dimension, sle.num_joint_states) == (2, 8)
    s = sle.start(np.full((2, 2), 0.5))
    assert s.shape == (8, 2, 2)
    out = sle.evolve(s, zero, 1.7)
    assert abs(out.sum(axis=0)[0, 1] - 0.5 * p.dephasing(1.7)) < 1e-11
    half = sle.evolve(s, zero, 0.85)
    echo = sle.evolve(sle.pulse(half, x), zero, 0.85).sum(axis=0)
    assert abs(echo[0, 1] - 0.5 * np.conj(p.dephasing(1.7, [0.85]))) < 1e-11
    # Silent noise with amplitude damping: e^{−γt} for the excited state.
    lowering = np.array([[0, 1], [0, 0]], dtype=complex) * 0.3
    damp = q.StochasticLiouville(
        2, [(q.MarkovProcess.telegraph(0.0, 1.0), z)], dissipators=[lowering]
    )
    excited = np.diag([0.0, 1.0]).astype(complex)
    r = damp.evolve(damp.start(excited), zero, 2.0).sum(axis=0)
    assert abs(r[1, 1] - math.exp(-0.09 * 2.0)) < 1e-11
    with pytest.raises(ValueError):
        sle.evolve(np.zeros((3, 2, 2)), zero, 1.0)
    with pytest.raises(ValueError):
        sle.pulse(s, 2 * x)


def test_density_matrices_average_slow_noise():
    ramsey = q.Circuit(1).h(0).id(0).h(0)
    telegraph = (
        q.DeviceModel("telegraph")
        .with_default_qubit(telegraph=(1.0, 0.3))
        .with_rule(gate="id", duration=2.0)
    )
    rho = q.DensityMatrix.simulate(ramsey, device=telegraph)
    exact = telegraph_free(1.0, 0.3, 2.0)
    assert abs(rho.expectation(q.PauliSum("Z0")) - exact) < 1e-12
    e = q.TrajectorySimulator(telegraph, 4000, 3).expectation(ramsey, q.PauliSum("Z0"))
    assert abs(e.value - exact) < 5 * e.std_error
    gauss = (
        q.DeviceModel("gauss")
        .with_default_qubit(detuning=("gaussian", 0.8))
        .with_rule(gate="id", duration=2.0)
    )
    rho = q.DensityMatrix.simulate(ramsey, device=gauss)
    assert abs(rho.expectation(q.PauliSum("Z0")) - math.exp(-0.32 * 4)) < 1e-12
