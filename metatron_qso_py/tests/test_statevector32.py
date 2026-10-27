"""Single-precision state vectors."""

import numpy as np
import pytest

import metatron_qso as q


def layers(n: int, reps: int) -> q.Circuit:
    c = q.Circuit(n)
    for r in range(reps):
        for i in range(n):
            c.rx(i, 0.3 + 0.1 * i + r).rz(i, 0.2)
        for i in range(0, n - 1, 2):
            c.cx(i, i + 1)
        for i in range(1, n - 1, 2):
            c.cz(i, i + 1)
    return c


def test_single_precision_follows_double_precision() -> None:
    c = layers(12, 20)
    exact = q.simulate(c)
    single = q.simulate(c, precision="single")
    assert isinstance(single, q.StateVector32)
    assert single.amplitudes.dtype == np.complex64
    assert 1 - single.fidelity(exact) < 1e-11
    assert abs(single.norm() - 1) < 1e-5
    h = q.PauliSum("0.5 Z0 Z1 + 0.3 X0 Y2 - 0.7 X1")
    assert single.expectation(h) == pytest.approx(exact.expectation(h), abs=1e-6)
    assert single.probabilities().sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(single.probabilities(), exact.probabilities(), atol=1e-6)
    # Built gate by gate, rounded from a double state, and back.
    step = q.StateVector32.zero(12).apply(c)
    assert 1 - step.fidelity(exact) < 1e-11
    assert 1 - q.StateVector32.from_state(exact).fidelity(exact) < 1e-12
    assert 1 - exact.fidelity(single.to_state()) < 1e-10


def test_single_precision_samples_and_rejects_bad_input() -> None:
    c = q.Circuit(3)
    c.h(0).cx(0, 1).cx(1, 2)
    s = q.simulate(c, precision="single")
    counts = s.sample(2000, seed=3)
    assert set(counts) == {"000", "111"}
    assert abs(counts["000"] - 1000) < 150
    assert "StateVector32(n_qubits=3)" in repr(s)
    assert q.StateVector32.basis(2, 3).probabilities()[3] == 1.0
    with pytest.raises(ValueError):
        q.StateVector32.zero(0)
    with pytest.raises(ValueError):
        q.StateVector32.basis(2, 4)
    with pytest.raises(ValueError):
        q.simulate(c, precision="half")
    with pytest.raises(ValueError):
        q.StateVector32.zero(2).apply(c)
