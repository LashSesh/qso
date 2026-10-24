import numpy as np
import pytest

import metatron_qso as q


def test_noiseless_density_matrix_matches_state_vector():
    c = q.efficient_su2(3, 1).bind(np.linspace(0.1, 1.2, 12))
    rho = q.DensityMatrix.simulate(c)
    psi = q.simulate(c)
    m = rho.to_matrix()
    assert m.dtype == np.complex128 and m.shape == (8, 8)
    assert np.allclose(m, np.outer(psi.amplitudes, psi.amplitudes.conj()), atol=1e-12)
    assert rho.purity() == pytest.approx(1.0) and rho.trace() == pytest.approx(1.0)
    assert rho.fidelity_with_pure(psi) == pytest.approx(1.0)
    h = q.PauliSum("Z0 Z1 + 0.3 X2 - 0.2 Y0")
    assert rho.expectation(h) == pytest.approx(psi.expectation(h), abs=1e-12)
    p = rho.probabilities()
    assert p.dtype == np.float64 and np.allclose(p, psi.probabilities())


@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.75, 1.0])
def test_depolarizing_purity(p):
    rho = q.DensityMatrix.zero(1).apply_channel(q.Channel.depolarizing(p), [0])
    r = 1 - 4 * p / 3  # Bloch-vector length
    assert rho.purity() == pytest.approx((1 + r**2) / 2, abs=1e-12)
    assert rho.trace() == pytest.approx(1.0, abs=1e-12)


def test_amplitude_and_phase_damping():
    rho = q.DensityMatrix.simulate(q.Circuit(1).x(0))
    rho.apply_channel(q.Channel.amplitude_damping(0.25), [0])
    assert np.allclose(rho.probabilities(), [0.25, 0.75])
    assert rho.purity() == pytest.approx(0.25**2 + 0.75**2)
    plus = q.DensityMatrix.from_pure(q.StateVector.plus(1))
    plus.apply_channel(q.Channel.phase_damping(0.36), [0])
    assert plus.to_matrix()[0, 1] == pytest.approx(0.5 * 0.8)
    t1, t2, t = 50.0, 30.0, 10.0
    relaxed = q.DensityMatrix.from_pure(q.StateVector.plus(1))
    relaxed.apply_channel(q.Channel.thermal_relaxation(t1, t2, t), [0])
    m = relaxed.to_matrix()
    assert abs(m[0, 1]) == pytest.approx(0.5 * np.exp(-t / t2))
    assert m[1, 1].real == pytest.approx(0.5 * np.exp(-t / t1))


def test_maximally_mixed_and_full_depolarisation():
    assert q.DensityMatrix.maximally_mixed(3).purity() == pytest.approx(1 / 8)
    bell = q.Circuit(2).h(0).cx(0, 1)
    rho = q.DensityMatrix.simulate(bell)
    rho.apply_channel(q.Channel.depolarizing_2q(15 / 16), [0, 1])
    assert rho.purity() == pytest.approx(1 / 4, abs=1e-12)
    assert np.allclose(rho.to_matrix(), np.eye(4) / 4, atol=1e-12)


def test_noise_model_reduces_purity_monotonically():
    c = q.Circuit(2).h(0).cx(0, 1).ry(1, 0.3).cz(0, 1)
    purities = [
        q.DensityMatrix.simulate(c, noise=q.NoiseModel.depolarizing(p, 2 * p)).purity()
        for p in [0.0, 0.01, 0.05, 0.2]
    ]
    assert purities[0] == pytest.approx(1.0)
    assert all(b < a for a, b in zip(purities, purities[1:]))
    model = q.NoiseModel(
        single_qubit=[q.Channel.amplitude_damping(0.02)],
        multi_qubit=[q.Channel.depolarizing_2q(0.03), q.Channel.phase_flip(0.01)],
        readout=(0.02, 0.05),
    )
    assert [ch.n_qubits for ch in model.multi_qubit] == [2, 1]
    assert model.readout == (0.02, 0.05)
    rho = q.DensityMatrix.zero(2).apply(c, noise=model)
    assert rho.trace() == pytest.approx(1.0, abs=1e-12) and rho.purity() < 1.0


def test_kraus_channels():
    ch = q.Channel.bit_flip(0.2)
    ops = ch.kraus_operators()
    assert all(k.dtype == np.complex128 and k.shape == (2, 2) for k in ops)
    assert np.allclose(sum(k.conj().T @ k for k in ops), np.eye(2))
    custom = q.Channel.from_kraus(ops, name="my_flip")
    assert custom.name == "my_flip" and custom.n_qubits == 1
    rho = q.DensityMatrix.zero(1).apply_channel(custom, [0])
    assert np.allclose(rho.probabilities(), [0.8, 0.2])


def test_shot_based_estimation():
    h2 = q.h2_sto3g()
    ansatz = q.real_amplitudes(2, 1)
    r = q.run_vqe(h2["hamiltonian"], ansatz, restarts=2)
    exact = r["energy"]
    est = q.estimate_expectation(ansatz, h2["hamiltonian"], r["parameters"], shots=200_000, seed=1)
    assert est == pytest.approx(exact, abs=5e-3)
    assert est == q.estimate_expectation(
        ansatz, h2["hamiltonian"], r["parameters"], shots=200_000, seed=1
    )
    noisy = q.estimate_expectation(
        ansatz,
        h2["hamiltonian"],
        r["parameters"],
        shots=200_000,
        seed=1,
        noise=q.NoiseModel.depolarizing(0.05, 0.1),
    )
    assert noisy > exact + 0.02  # noise pushes the energy up from the ground state


def test_sampling_with_readout_error():
    rho = q.DensityMatrix.zero(1)
    assert rho.sample(1000) == {"0": 1000}
    counts = rho.sample(20_000, seed=2, readout=(0.1, 0.0))
    assert abs(counts["1"] / 20_000 - 0.1) < 0.01


@pytest.mark.parametrize(
    "call",
    [
        lambda: q.Channel.depolarizing(1.5),
        lambda: q.Channel.amplitude_damping(-0.1),
        lambda: q.Channel.thermal_relaxation(1.0, 3.0, 0.1),
        lambda: q.Channel.from_kraus([np.eye(2), np.eye(2)]),
        lambda: q.NoiseModel(readout=(1.5, 0.0)),
        lambda: q.DensityMatrix.zero(q.MAX_DENSITY_QUBITS + 1),
        lambda: q.DensityMatrix.simulate(q.Circuit(15)),
        lambda: q.DensityMatrix.zero(2).apply_channel(q.Channel.depolarizing(0.1), [0, 1]),
        lambda: q.DensityMatrix.zero(2).apply_channel(q.Channel.depolarizing(0.1), [2]),
        lambda: q.estimate_expectation(q.Circuit(1), q.PauliSum("Z0"), shots=0),
        lambda: q.estimate_expectation(q.Circuit(1), q.PauliSum("Z1")),
    ],
)
def test_invalid_noise_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()
