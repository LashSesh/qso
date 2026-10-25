"""Device noise, trajectories, estimation and error mitigation."""

import json

import numpy as np
import pytest

import metatron_qso as q


def bell() -> q.Circuit:
    return q.Circuit(2).h(0).cx(0, 1)


def device() -> q.DeviceModel:
    return (
        q.DeviceModel("toy")
        .with_default_qubit(t1=50.0, t2=40.0, readout=(0.02, 0.05))
        .with_rule(gate="cx", duration=0.3, depolarizing=0.02,
                   channels=[q.Channel.rotation("ZZ", 0.05)])
        .with_rule(num_qubits=1, duration=0.03, depolarizing=0.001)
    )


def test_channels():
    zz = q.Channel.pauli({"ZZ": 0.1, "XI": 0.05})
    p = zz.pauli_probabilities()
    assert p is not None and p[15] == pytest.approx(0.1) and p[0] == pytest.approx(0.85)
    ad = q.Channel.amplitude_damping(0.3)
    assert ad.pauli_probabilities() is None
    tw = ad.twirled().pauli_probabilities()
    assert tw is not None and tw.sum() == pytest.approx(1.0)
    both = q.Channel.depolarizing(0.1).compose(q.Channel.phase_flip(0.2))
    assert both.n_qubits == 1 and both.pauli_probabilities() is not None
    rz = q.Channel.rotation("Z", 0.4)
    rho = q.DensityMatrix.simulate(q.Circuit(1).h(0))
    rho.apply_channel(rz, [0])
    psi = q.simulate(q.Circuit(1).h(0).rz(0, 0.4))
    assert rho.fidelity_with_pure(psi) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        q.Channel.unitary(np.eye(2) * 2)
    with pytest.raises(ValueError):
        q.Channel.pauli({"Q": 0.1})


def test_device_models():
    d = device()
    assert "DeviceModel" in repr(d) and d.idle_noise
    again = q.DeviceModel.from_json(d.to_json())
    assert json.loads(again.to_json()) == json.loads(d.to_json())
    assert d.readout_errors(2) == [(0.02, 0.05), (0.02, 0.05)]
    with pytest.raises(ValueError):
        d.with_qubit(0, t1=10.0, t2=30.0)
    with pytest.raises(ValueError):
        q.DeviceModel.from_json('{"colour": "blue"}')
    # A NoiseModel and its device form give the same density matrix.
    model = q.NoiseModel(single_qubit=[q.Channel.depolarizing(0.02)],
                         multi_qubit=[q.Channel.depolarizing_2q(0.05)])
    a = q.DensityMatrix.simulate(bell(), noise=model)
    b = q.DensityMatrix.simulate(bell(), device=q.DeviceModel.from_noise_model(model))
    assert np.allclose(a.to_matrix(), b.to_matrix(), atol=1e-13)
    with pytest.raises(ValueError):
        q.DensityMatrix.simulate(bell(), noise=model, device=d)
    rho = q.DensityMatrix.simulate(bell(), device=d)
    counts = rho.sample_with_readout(1000, d.readout_errors(2), seed=1)
    assert sum(counts.values()) == 1000


def test_trajectories_match_density_matrices():
    d = device()
    c = q.Circuit(3).h(0).cx(0, 1).ry(2, 0.7).cx(1, 2)
    h = q.PauliSum("Z0 Z1 + 0.5 X2")
    exact = q.DensityMatrix.simulate(c, device=d).expectation(h)
    sim = q.TrajectorySimulator(d, 3000, seed=4)
    e = sim.expectation(c, h)
    assert isinstance(e, q.Estimate) and e.samples == 3000
    assert abs(e.value - exact) < 4 * e.std_error
    lo, hi = e.confidence_interval(0.99)
    assert lo < e.value < hi
    assert float(e) == e.value
    es = sim.expectations(c, [h, q.PauliSum("Z0")])
    assert es[0].value == e.value
    counts = sim.sample(c, 500)
    assert sum(counts.values()) == 500 and all(len(k) == 3 for k in counts)
    avg = sim.density_matrix(c)
    assert avg.trace() == pytest.approx(1.0)
    assert sim.trajectory_state(c, 0).n_qubits == 3
    # Shot-based estimates on the device include its readout errors, the
    # same for density matrices and trajectories.
    via_dm = q.estimate_expectation(c, h, shots=50000, device=d, seed=2)
    via_tr = q.estimate_expectation(c, h, shots=50000, device=d, trajectories=5000, seed=3)
    assert abs(via_dm - via_tr) < 0.05
    assert via_dm < exact - 0.05
    with pytest.raises(ValueError):
        q.estimate_expectation(c, h, trajectories=10)


def test_estimates_and_allocation():
    c = q.Circuit(2).ry(0, 0.4).ry(1, -1.1).cx(0, 1)
    h = q.PauliSum("5 Z0 Z1 + 0.1 X0 X1 + 0.1 Y0 Y1")
    exact = h.expectation(q.simulate(c))
    est, groups = q.estimate(c, h, 6000, allocation="adaptive", pilot=0.2, seed=3)
    assert abs(est.value - exact) < 4 * est.std_error
    assert sum(g["shots"] for g in groups) == 6000 and len(groups) == 3
    assert groups[0]["basis"] == "Z0 Z1"
    with pytest.raises(ValueError):
        q.estimate(c, h, 100, allocation="random")
    # Readout mitigation removes the readout bias.
    noisy = q.DeviceModel("readout").with_default_qubit(readout=(0.05, 0.1))
    cal = q.ReadoutCalibration.tensored(2, 100000, seed=1, device=noisy)
    raw, _ = q.estimate(c, h, 200000, device=noisy, seed=5)
    mit, _ = q.estimate(c, h, 200000, device=noisy, seed=5, readout_calibration=cal)
    assert abs(raw.value - exact) > 10 * raw.std_error
    assert abs(mit.value - exact) < 4 * mit.std_error + 0.01


def test_readout_calibration():
    cal = q.ReadoutCalibration.from_errors([(0.1, 0.2)])
    quasi = cal.mitigate({"0": 200, "1": 800})
    assert quasi["1"] == pytest.approx(1.0)
    assert cal.qubit_errors() == [(0.1, 0.2)]
    full = q.ReadoutCalibration.from_matrix(1, [[0.9, 0.2], [0.1, 0.8]])
    assert full.mitigate({"0": 200, "1": 800})["1"] == pytest.approx(1.0)
    p = q.nearest_probability_distribution({"00": 0.6, "01": 0.5, "10": -0.1})
    assert p == pytest.approx({"00": 0.55, "01": 0.45})
    with pytest.raises(ValueError):
        cal.mitigate({"2": 5})


def test_shadows():
    psi = q.simulate(bell())
    shadow = q.ClassicalShadow.from_state(psi, 20000, seed=1)
    assert len(shadow) == 20000 and shadow.n_qubits == 2
    e = shadow.estimate(q.PauliSum("Z0 Z1"))
    assert abs(e.value - 1.0) < 4 * e.std_error
    assert abs(shadow.median_of_means(q.PauliSum("X0 X1"), 10) - 1.0) < 0.2
    noisy = q.ClassicalShadow.collect(bell(), 5000, seed=2, noise=q.NoiseModel.depolarizing(0.01, 0.02))
    assert len(noisy) == 5000


def test_zero_noise_extrapolation():
    d = q.DeviceModel("dep").with_rule(num_qubits=2, depolarizing=0.02).with_rule(depolarizing=0.002)
    c = q.Circuit(2).h(0).cx(0, 1).rz(1, 0.3).cx(0, 1).h(0)
    z = q.PauliSum("Z0")
    ideal = z.expectation(q.simulate(c))

    def run(folded: q.Circuit) -> float:
        return q.DensityMatrix.simulate(folded, device=d).expectation(z)

    folded, scale = q.fold(c, 3.0)
    assert scale == 3.0 and len(folded.instructions()) == 3 * len(c.instructions())
    r = q.zne(c, [1.0, 3.0, 5.0], run)
    assert abs(r["estimate"].value - ideal) < 0.01 * abs(run(c) - ideal)
    assert r["scales"] == [1.0, 3.0, 5.0] and len(r["values"]) == 3
    e = q.extrapolate([1.0, 3.0], [0.8, q.Estimate(0.6, 0.02)], method="linear")
    assert e.value == pytest.approx(0.9) and e.std_error == pytest.approx(0.01)
    with pytest.raises(ValueError):
        q.extrapolate([1.0, 2.0], [0.5, 0.4], method="polynomial")

    def broken(_: q.Circuit) -> float:
        raise RuntimeError("device offline")

    with pytest.raises(RuntimeError, match="offline"):
        q.zne(c, [1.0, 3.0], broken)


def test_pec_and_twirling():
    d = q.DeviceModel("pauli").with_rule(gate="cx", depolarizing=0.05)
    h = q.PauliSum("Z0 Z1")
    r = q.pec_expectation(bell(), h, d, 4000, seed=1)
    assert abs(r["estimate"].value - 1.0) < 4 * r["estimate"].std_error
    assert r["overhead"] > 1.0 and r["noisy_operations"] == 1
    with pytest.raises(ValueError):
        q.pec_expectation(bell(), h, device(), 10)
    t = q.twirl(bell().cz(0, 1), seed=3)
    assert q.simulate(t).fidelity(q.simulate(bell().cz(0, 1))) == pytest.approx(1.0)
    coherent = q.DeviceModel("zz").with_rule(gate="cx", channels=[q.Channel.rotation("ZZ", 0.5)])
    c = q.Circuit(2).h(0).cx(0, 1).rz(1, 0.3).cx(0, 1).h(0)
    z = q.PauliSum("Z0")
    avg = q.twirled_average(c, 500, lambda tc: q.DensityMatrix.simulate(tc, device=coherent).expectation(z), seed=2)
    target = q.DensityMatrix.simulate(
        c, device=q.DeviceModel("tw").with_rule(gate="cx", channels=[q.Channel.rotation("ZZ", 0.5).twirled()])
    ).expectation(z)
    assert abs(avg.value - target) < 4 * avg.std_error
