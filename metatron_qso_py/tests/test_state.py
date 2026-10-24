import math

import numpy as np
import pytest
from conftest import I2, X, Y, Z, kron_all

import metatron_qso as q


def test_arrays_have_numpy_dtypes_and_shapes():
    s = q.simulate(q.Circuit(3).h(0).cx(0, 1))
    assert s.n_qubits == 3
    amps = s.amplitudes
    assert isinstance(amps, np.ndarray) and amps.dtype == np.complex128 and amps.shape == (8,)
    p = s.probabilities()
    assert p.dtype == np.float64 and p.shape == (8,)
    assert p.sum() == pytest.approx(1.0)
    rho = s.reduced_density_matrix([0])
    assert rho.dtype == np.complex128 and rho.shape == (2, 2)
    assert np.allclose(rho, np.eye(2) / 2)


def test_constructors():
    assert np.allclose(q.StateVector.zero(2).probabilities(), [1, 0, 0, 0])
    assert np.allclose(q.StateVector.plus(2).probabilities(), [0.25] * 4)
    assert np.allclose(q.StateVector.basis(2, 2).probabilities(), [0, 0, 1, 0])
    s = q.StateVector.from_amplitudes([1, 1j])
    assert np.allclose(s.amplitudes, np.array([1, 1j]) / math.sqrt(2))
    r1, r2 = q.StateVector.random(4, seed=7), q.StateVector.random(4, seed=7)
    assert np.array_equal(r1.amplitudes, r2.amplitudes)
    assert np.linalg.norm(r1.amplitudes) == pytest.approx(1.0)


def test_sampling_uses_qubit_n_minus_1_first():
    s = q.simulate(q.Circuit(3).x(0))
    assert s.sample(10) == {"001": 10}
    assert q.bitstring(1, 3) == "001"
    bell = q.simulate(q.Circuit(2).h(0).cx(0, 1))
    counts = bell.sample(4000, seed=11)
    assert set(counts) == {"00", "11"} and sum(counts.values()) == 4000
    assert abs(counts["00"] - 2000) < 200
    assert bell.sample(100, seed=3) == bell.sample(100, seed=3)


def test_expectation_fidelity_entropy():
    bell = q.simulate(q.Circuit(2).h(0).cx(0, 1))
    assert bell.expectation(q.PauliSum("Z0 Z1")) == pytest.approx(1.0)
    assert bell.expectation(q.PauliSum("X0 X1")) == pytest.approx(1.0)
    assert bell.expectation(q.PauliSum("Y0 Y1")) == pytest.approx(-1.0)
    assert bell.expectation(q.PauliSum("Z0")) == pytest.approx(0.0, abs=1e-15)
    assert bell.entanglement_entropy([0]) == pytest.approx(1.0)
    assert q.StateVector.zero(2).entanglement_entropy([1]) == pytest.approx(0.0, abs=1e-12)
    plus = q.StateVector.plus(1)
    assert plus.fidelity(q.StateVector.zero(1)) == pytest.approx(0.5)
    assert plus.inner(q.StateVector.zero(1)) == pytest.approx(1 / math.sqrt(2))


def test_apply_is_in_place_and_parametrised():
    c = q.Circuit(1)
    c.ry(0, c.new_parameter())
    s = q.StateVector.zero(1)
    assert s.apply(c, [math.pi]) is s
    assert np.allclose(s.probabilities(), [0, 1])


def test_pauli_sum_algebra_and_matrix():
    h = q.PauliSum("0.5*Z0 Z1 - 1.2 X0 + 0.3*Y1 - 1.05")
    assert h.num_terms == 4 and h.min_qubits == 2
    assert h.constant == pytest.approx(-1.05)
    assert sorted(s for _, s in h.terms()) == sorted(["I", "X0", "Y1", "Z0 Z1"])
    m = h.to_matrix()
    ref = (
        0.5 * kron_all(Z, Z) - 1.2 * kron_all(I2, X) + 0.3 * kron_all(Y, I2) - 1.05 * np.eye(4)
    )
    assert m.dtype == np.complex128 and np.allclose(m, ref)
    assert h == q.PauliSum(str(h))
    assert h + h == 2.0 * h == h * 2.0
    assert (h - h).num_terms == 0
    assert -h == -1.0 * h
    assert (h + 1.05).constant == pytest.approx(0.0)
    assert (1.0 - h) == -h + 1.0
    assert h / 2 == 0.5 * h
    assert sum([q.PauliSum("Z0"), q.PauliSum("Z1")]) == q.PauliSum("Z0 + Z1")
    assert q.PauliSum.from_terms([(0.5, "Z0 Z1"), (-1.0, "X2")]) == q.PauliSum("0.5 Z0 Z1 - X2")
    assert q.PauliSum().num_terms == 0
    psi = q.StateVector.random(2, seed=4)
    v = psi.amplitudes
    assert h.expectation(psi) == pytest.approx(np.vdot(v, ref @ v).real)


def test_ground_state_matches_dense_diagonalisation():
    tfim = q.transverse_field_ising(6, 1.0, 0.7, periodic=True)
    exact = np.linalg.eigvalsh(tfim.to_matrix(6))[0]
    assert tfim.ground_state_energy(6) == pytest.approx(exact, abs=1e-9)
    e, state = tfim.ground_state(6)
    assert e == pytest.approx(exact, abs=1e-9)
    assert state.expectation(tfim) == pytest.approx(exact, abs=1e-8)
    xxz = q.heisenberg_xxz(4, 1.0, 0.5)
    assert xxz.ground_state_energy() == pytest.approx(np.linalg.eigvalsh(xxz.to_matrix())[0])
    # Transverse-field Ising at h = 0 is diagonal: ground energy −J·(bonds).
    assert q.transverse_field_ising(5, 1.0, 0.0).ground_state_energy() == pytest.approx(-4.0)


@pytest.mark.parametrize(
    "build",
    [
        lambda: q.StateVector.zero(0),
        lambda: q.StateVector.zero(q.MAX_QUBITS + 1),
        lambda: q.StateVector.from_amplitudes([1, 0, 0]),
        lambda: q.StateVector.from_amplitudes([0, 0]),
        lambda: q.StateVector.basis(2, 4),
        lambda: q.StateVector.zero(2).fidelity(q.StateVector.zero(3)),
        lambda: q.StateVector.zero(2).expectation(q.PauliSum("Z2")),
        lambda: q.StateVector.zero(1).apply(q.Circuit(2)),
        lambda: q.simulate(q.real_amplitudes(2, 1), [0.1]),
        lambda: q.PauliSum("Z0 Q1"),
        lambda: q.PauliSum("Z0 X0"),
        lambda: q.PauliSum("Z0").to_matrix(15),
        lambda: q.PauliSum("Z0").ground_state_energy(60),
        lambda: q.transverse_field_ising(65, 1.0, 1.0),
        lambda: q.bitstring(8, 3),
    ],
)
def test_invalid_states_raise_value_error(build):
    with pytest.raises(ValueError):
        build()
