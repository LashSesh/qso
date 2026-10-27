"""Noise learning: sparse Pauli–Lindblad models, PEC and PEA with them."""

import pytest

import metatron_qso as q


def model() -> q.SparsePauliLindblad:
    return q.SparsePauliLindblad(
        2,
        ["X0", "Z0", "Z1", "Z0 Z1", "X0 X1", "Y0 Y1"],
        [0.004, 0.006, 0.005, 0.006, 0.003, 0.003],
    )


def device_for(m: q.SparsePauliLindblad) -> q.DeviceModel:
    def label(s: str) -> str:
        letters = {0: "I", 1: "I"}
        for factor in s.split():
            letters[int(factor[1:])] = factor[0]
        return letters[0] + letters[1]

    probs = {label(s): p for s, p in m.error_probabilities().items() if s != "I"}
    return q.DeviceModel("learned", idle_noise=False).with_rule(
        gate="cx", channels=[q.Channel.pauli(probs)]
    )


def layer() -> q.Circuit:
    return q.Circuit(2).cx(0, 1)


def blocks() -> list:
    return [
        q.Circuit(2).ry(0, 0.7).rz(1, 1.3).ry(1, -0.4),
        q.Circuit(2).ry(0, -1.1).rz(0, 0.5).ry(1, 2.0),
        q.Circuit(2).ry(0, 0.3).ry(1, -0.9),
    ]


def exact(obs: q.PauliSum) -> float:
    c = q.Circuit(2)
    for i, b in enumerate(blocks()):
        if i:
            c.append(layer())
        c.append(b)
    return q.DensityMatrix.simulate(c).expectation(obs)


def test_model():
    m = model()
    assert m.n_qubits == 2 and len(m.generators) == 6
    assert m.generators[3] == "Z0 Z1"
    probs = m.error_probabilities()
    assert sum(probs.values()) == pytest.approx(1.0)
    # X0 anticommutes with Z0 and Z0 Z1 (and Y0 Y1).
    assert m.fidelity("X0") == pytest.approx(2.718281828459045 ** (-2 * 0.015))
    assert m.overhead() == pytest.approx(2.718281828459045 ** (2 * sum(m.rates)))
    assert m.scaled(2.0).rates == pytest.approx([2 * r for r in m.rates])
    assert "generators=6" in repr(m)
    with pytest.raises(ValueError):
        q.SparsePauliLindblad(2, ["X0"], [-1.0])
    with pytest.raises(ValueError):
        q.SparsePauliLindblad(2, ["0.5*X0"], [0.1])


def test_learning_and_cancellation():
    truth = model()
    device = device_for(truth)
    learned = q.learn_layer(layer(), device, seed=3)
    m = learned["model"]
    assert len(m.generators) == 15 and len(learned["std_errors"]) == 15
    assert "X0" in learned["symmetrised"] and "Z0" not in learned["symmetrised"]
    assert len(learned["products"]) == 15
    obs = q.PauliSum("Z0 Z1 + 0.5*X0")
    want = exact(obs)
    r = q.pec_layered(blocks(), layer(), m, device, obs, 2000, seed=1)
    est = r["estimate"]
    assert r["noisy_operations"] == 2
    assert abs(est.value - want) < 4 * est.std_error
    z = q.pea_layered(blocks(), layer(), truth, device, obs, [1.0, 2.0, 3.0], 2000, seed=2)
    assert z["scales"] == [1.0, 2.0, 3.0] and len(z["values"]) == 3
    assert abs(z["estimate"].value - want) < abs(z["values"][0].value - want) + 4 * z["estimate"].std_error


def test_invalid_layers():
    device = q.DeviceModel("ideal")
    with pytest.raises(ValueError):
        q.learn_layer(q.Circuit(2).h(0), device)
    with pytest.raises(ValueError):
        q.learn_layer(layer(), device, depths=[2, 3])
    with pytest.raises(ValueError):
        q.pea_layered(blocks(), layer(), model(), device, q.PauliSum("Z0"), [1.0], 10)
