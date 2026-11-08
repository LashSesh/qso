"""Expectation values of large noisy circuits by Pauli propagation (1.64)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import metatron_qso as q

# The layers of the dense reference with a Pauli–Lindblad layer in the
# middle (design §6.1, row 7).
RATES_TEXT = "0.1*X0 + 0.05*Z0 Z1 + 0.02*Y1 + 0.03*X1 Z2"
RATES_LIST = [(0.1, "X0"), (0.05, "Z0 Z1"), (0.02, "Y1"), (0.03, "X1 Z2")]
OBSERVABLE = "Z0 Z1 + 0.5*X0 + 0.25*X1 X2 - 0.3*Z2"
MID_LAYER = {
    0.0: 0.8610609940028848,
    1.0: 0.6223387667165513,
    2.5: 0.37507840660135505,
    -1.0: 1.1833022950695018,
    -0.5: 1.0101202024739016,
}


def mid_layer_circuit(rates):
    before = q.Circuit(3).h(0).cx(0, 1).ry(2, 0.7)
    after = q.Circuit(3).rx(1, 0.4).cx(1, 2)
    return [before, *rates, after, (q.Channel.amplitude_damping(0.2), [2])]


def ghz_qasm(n):
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];", "h q[0];"]
    lines += [f"cx q[{k}],q[{k + 1}];" for k in range(n - 1)]
    return "\n".join(lines) + "\n"


def heavy_hex():
    path = Path(__file__).resolve().parents[2] / "metatron-qso-rs/tests/data/heavy_hex_7.json"
    data = json.loads(path.read_text())
    return data["n_qubits"], data["colours"], data["central_qubit"]


def kicked_ising_qasm(n, gates):
    head = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];"]
    return "\n".join(head + gates) + "\n"


def test_single_qubit_channels_in_closed_form():
    cases = [
        (
            [q.Circuit(1).ry(0, 0.7), (q.Channel.amplitude_damping(0.2), [0])],
            "Z0",
            0.8118737498275908,
        ),
        ([q.Circuit(1).x(0), (q.Channel.amplitude_damping(0.3), [0])], "Z0", -0.4),
        (
            [q.Circuit(1).h(0), (q.Channel.thermal_relaxation(50, 70, 10), [0])],
            "X0",
            0.8668778997501816,
        ),
        (
            [q.Circuit(1).x(0), (q.Channel.thermal_relaxation(50, 70, 10), [0])],
            "Z0",
            -0.6374615061559636,
        ),
        (
            [q.Circuit(2).h(0).cx(0, 1), (q.Channel.depolarizing_2q(0.06), [0, 1])],
            "X0 X1",
            0.936,
        ),
    ]
    for layers, observable, expected in cases:
        r = q.propagate(layers, observable)
        assert abs(r["value"] - expected) <= 1e-15
        assert r["truncation_bound"] <= 1e-15
        assert r["dropped_terms"] == 0
        assert r["operator"] is None


def test_mid_circuit_pauli_lindblad_layer_matches_density_matrix():
    model = q.SparsePauliLindblad(3, ["X0", "Z0 Z1", "Y1", "X1 Z2"], [0.1, 0.05, 0.02, 0.03])
    for alpha, expected in MID_LAYER.items():
        forms = [
            [(RATES_TEXT, alpha)],
            [(RATES_LIST, alpha)],
            [(model, [0, 1, 2], alpha)],
        ]
        for rates in forms:
            r = q.propagate(mid_layer_circuit(rates), OBSERVABLE)
            assert abs(r["value"] - expected) <= 1e-14, (alpha, rates)
            assert r["truncation_bound"] <= 1e-14
    # Amplifying and then inverting the noise restores the value at α = 0.
    r = q.propagate(mid_layer_circuit([(RATES_TEXT, 1.0), (RATES_TEXT, -1.0)]), OBSERVABLE)
    assert abs(r["value"] - MID_LAYER[0.0]) <= 1e-14
    # A model on other register qubits: its qubit i sits on qubits[i].
    moved = q.SparsePauliLindblad(3, ["X2", "Z2 Z0", "Y0", "X0 Z1"], [0.1, 0.05, 0.02, 0.03])
    r = q.propagate(mid_layer_circuit([(moved, [1, 2, 0], 1.0)]), OBSERVABLE)
    assert abs(r["value"] - MID_LAYER[1.0]) <= 1e-14


def test_truncation_bound_is_tight_and_restricted_to_the_light_cone():
    r = q.propagate([("0.1*X0 + 0.3*X1", -1.0)], "Z0", min_coefficient=2.0)
    assert r["value"] == 0.0
    assert r["terms"] == 0 and r["dropped_terms"] == 1
    # The true value is e^{0.2}; the generator on qubit 1 lies outside the
    # light cone and does not enter the bound (it would give e^{0.8}).
    assert abs(r["truncation_bound"] - 1.2214027581601699) <= 1e-15
    exact = q.propagate([("0.1*X0 + 0.3*X1", -1.0)], "Z0")
    assert abs(exact["value"] - math.exp(0.2)) <= 1e-15


def test_truncated_values_lie_within_the_bound():
    n = 8
    c = q.Circuit(n)
    for layer in range(4):
        for k in range(n):
            c.rx(k, 0.3 + 0.1 * k).rz(k, 0.2 * layer + 0.05)
        for k in range(layer % 2, n - 1, 2):
            c.cx(k, k + 1)
    layers = [c, (q.Channel.amplitude_damping(0.05), [3]), (q.Channel.depolarizing(0.02), [4])]
    observable = "Z3 Z4 + 0.5*X0"
    dm = q.DensityMatrix.simulate(c)
    dm.apply_channel(q.Channel.amplitude_damping(0.05), [3])
    dm.apply_channel(q.Channel.depolarizing(0.02), [4])
    exact = dm.expectation(q.PauliSum(observable))
    full = q.propagate(layers, observable)
    assert abs(full["value"] - exact) <= 1e-12
    for options in [{"min_coefficient": 1e-2}, {"max_weight": 2}, {"min_coefficient": 5e-2}]:
        r = q.propagate(layers, observable, **options)
        assert r["dropped_terms"] > 0
        assert abs(r["value"] - exact) <= r["truncation_bound"] + 1e-14
        assert r["truncation_bound"] > 0


def test_gate_set_matches_density_matrix():
    rng = np.random.default_rng(1640)
    for trial in range(6):
        n = 5
        c = q.Circuit(n)
        for _ in range(25):
            a, b, t = (int(x) for x in rng.choice(n, 3, replace=False))
            x = float(rng.uniform(-math.pi, math.pi))
            kind = int(rng.integers(12))
            if kind == 0:
                c.h(a).s(b).t(t)
            elif kind == 1:
                c.rx(a, x).ry(b, x / 2).rz(t, -x)
            elif kind == 2:
                c.rzz(a, b, x).rxx(b, t, x / 3)
            elif kind == 3:
                c.ryy(a, t, x)
            elif kind == 4:
                c.ch(a, b).cz(b, t)
            elif kind == 5:
                c.crx(a, b, x).cry(b, t, x / 2)
            elif kind == 6:
                c.crz(a, b, x).cp(b, t, -x)
            elif kind == 7:
                c.ccx(a, b, t)
            elif kind == 8:
                c.cswap(a, b, t)
            elif kind == 9:
                c.u(a, x, x / 2, -x / 3).sx(b).sxdg(t)
            elif kind == 10:
                c.p(a, x).sdg(b).tdg(t).y(a)
            else:
                c.cx(a, b).cy(b, t).swap(a, t)
        observable = "Z0 Z1 + 0.5*X2 - 0.25*Y3 X4 + 0.1"
        exact = q.DensityMatrix.simulate(c).expectation(q.PauliSum(observable))
        r = q.propagate(c, observable)
        assert abs(r["value"] - exact) <= 1e-12, trial
        assert r["truncation_bound"] <= 1e-12


def test_parameters_are_bound_with_theta():
    c = q.Circuit(2)
    a, b = c.new_parameters(2)
    c.ry(0, a).cx(0, 1).rz(1, b).rx(1, 2.0 * a)
    theta = [0.4, -1.1]
    exact = q.DensityMatrix.simulate(c, theta).expectation(q.PauliSum("Z0 Z1 + X1"))
    r = q.propagate([c, (q.Channel.depolarizing(0.0), [0])], "Z0 Z1 + X1", theta=theta)
    assert abs(r["value"] - exact) <= 1e-14


def test_device_noise_matches_density_matrix():
    device = (
        q.DeviceModel("dev")
        .with_default_qubit(t1=80.0, t2=60.0)
        .with_qubit(0, t1=50.0, t2=70.0)
        .with_rule(gate="cx", duration=0.3, depolarizing=0.01)
        .with_rule(num_qubits=1, duration=0.05, over_rotation=0.02)
    )
    c = q.Circuit(4).h(0).cx(0, 1).ry(2, 0.7).cx(1, 2).rx(3, 0.4).cx(2, 3).h(1)
    observable = "Z0 Z1 + 0.5*X1 X2 - 0.3*Z3 + 0.2*Y2"
    exact = q.DensityMatrix.simulate(c, device=device).expectation(q.PauliSum(observable))
    r = q.propagate(c, observable, device=device)
    assert abs(r["value"] - exact) <= 1e-12
    # The JSON path takes per-qubit entries beyond the simulators' 34.
    spec = json.loads(device.to_json())
    spec["qubits"] = spec["qubits"] + [spec["default_qubit"]] * 99
    text = json.dumps(spec)
    with pytest.raises(ValueError):
        q.DeviceModel.from_json(text)
    r = q.propagate(c, observable, device=text)
    assert abs(r["value"] - exact) <= 1e-12
    # Slow noise is no product of channels.
    slow = device.with_qubit(1, t1=80.0, t2=60.0, detuning=("gaussian", 0.1))
    with pytest.raises(ValueError, match="slow noise"):
        q.propagate(c, observable, device=slow)


def test_one_hundred_qubit_openqasm_input():
    text = ghz_qasm(100)
    every_x = " ".join(f"X{k}" for k in range(100))
    r = q.propagate(text, every_x)
    assert r["value"] == 1.0 and r["truncation_bound"] == 0.0
    r = q.propagate(q.Program.from_qasm(text), "Z0 Z99 + 0.5*Z42", return_operator=True)
    assert r["value"] == 1.0
    assert r["operator"] is not None
    # Z_k runs back through the ladder to Z_0…Z_k and H turns Z_0 into X_0;
    # Z_0 Z_99 becomes Z_1…Z_99.
    chain = " ".join(f"Z{k}" for k in range(1, 100))
    half = "X0 " + " ".join(f"Z{k}" for k in range(1, 43))
    assert sorted(r["operator"]) == [(0.5, half), (1.0, chain)]
    # Noise after the circuit on a qubit far outside 64.
    r = q.propagate([text, (q.Channel.amplitude_damping(0.25), [99])], "Z0 Z99")
    assert abs(r["value"] - 0.75) <= 1e-15
    r = q.propagate([text, ("0.05*X99", 1.0)], "Z0 Z99")
    assert abs(r["value"] - math.exp(-0.1)) <= 1e-15


def test_return_operator_is_the_heisenberg_observable():
    c = q.Circuit(2).h(0).cx(0, 1).ry(1, 0.3)
    r = q.propagate(c, [(1.0, "Z1"), (0.5, "X0 X1"), (-0.25, "I")], return_operator=True)
    assert r["operator"] is not None
    assert r["terms"] == len(r["operator"])
    total = q.PauliSum.from_terms(r["operator"])
    assert abs(total.expectation(q.StateVector.zero(2)) - r["value"]) <= 1e-15
    exact = q.DensityMatrix.simulate(c).expectation(q.PauliSum("Z1 + 0.5*X0 X1 - 0.25"))
    assert abs(r["value"] - exact) <= 1e-15
    # A PauliSum observable gives the same.
    s = q.propagate(c, q.PauliSum("Z1 + 0.5*X0 X1 - 0.25"), return_operator=True)
    assert s == r


def test_product_input_states():
    c = q.Circuit(3).cx(0, 1).ry(2, 0.6).cz(1, 2).h(0)
    observable = "Z0 Z1 + X2 + 0.5*Y0 Z2"
    flipped = q.Circuit(3).x(0).x(2).append(c)
    expected = q.propagate(flipped, observable)["value"]
    assert abs(q.propagate(c, observable, state="101")["value"] - expected) <= 1e-15
    bloch = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    assert abs(q.propagate(c, observable, state=bloch)["value"] - expected) <= 1e-15
    # A pure Bloch vector on qubit 1 equals an RY rotation of |0⟩; a mixed
    # one is the mixture of the two antipodal pure states.
    t = 0.9
    rotated = q.Circuit(3).ry(1, t).append(c)
    up = q.propagate(rotated, observable)["value"]
    down = q.propagate(q.Circuit(3).ry(1, t + math.pi).append(c), observable)["value"]
    r_hat = [math.sin(t), 0.0, math.cos(t)]
    pure = [[0.0, 0.0, 1.0], r_hat, [0.0, 0.0, 1.0]]
    assert abs(q.propagate(c, observable, state=pure)["value"] - up) <= 1e-14
    s = 0.6
    mixed = [[0.0, 0.0, 1.0], [s * x for x in r_hat], [0.0, 0.0, 1.0]]
    value = q.propagate(c, observable, state=mixed)["value"]
    assert abs(value - ((1 + s) / 2 * up + (1 - s) / 2 * down)) <= 1e-14


def test_errors():
    c = q.Circuit(2).h(0).cx(0, 1)
    with pytest.raises(ValueError, match="a layer must be"):
        q.propagate([c, 3], "Z0")
    with pytest.raises(ValueError, match="tuple of 4 items"):
        q.propagate([(1, 2, 3, 4)], "Z0")
    with pytest.raises(ValueError, match="SparsePauliLindblad"):
        q.propagate([(q.Channel.depolarizing(0.1), [0], 1.0)], "Z0")
    with pytest.raises(ValueError, match="no circuit has parameters"):
        q.propagate(c, "Z0", theta=[0.1])
    with pytest.raises(ValueError, match="must be finite"):
        q.propagate(q.Circuit(1).rx(0, q.Circuit(1).new_parameter()), "Z0", theta=[math.nan])
    with pytest.raises(ValueError, match="'0' and '1'"):
        q.propagate(c, "Z0", state="0+")
    with pytest.raises(ValueError, match="the register 2"):
        q.propagate(c, "Z0", state="010", n_qubits=2)
    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        q.propagate(c, "Z0", state=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="length at most 1"):
        q.propagate(c, "Z0", state=[[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="DeviceModel or its JSON"):
        q.propagate(c, "Z0", device=3)
    with pytest.raises(ValueError, match="device description"):
        q.propagate(c, "Z0", device="{")
    with pytest.raises(ValueError, match="observable must be"):
        q.propagate(c, 3)
    with pytest.raises(ValueError):
        q.propagate(c, "Z0 + Q1")
    with pytest.raises(ValueError):
        q.propagate(c, "Z5", n_qubits=2)
    with pytest.raises(ValueError):
        q.propagate(c, "Z0", n_qubits=16385)
    with pytest.raises(ValueError, match="at most 4"):
        q.propagate(q.Circuit(5).mcx([0, 1, 2, 3], 4), "Z4")
    with pytest.raises(ValueError, match="max_terms"):
        q.propagate(q.Circuit(2).rx(0, 0.3).rx(1, 0.3), "Z0 Z1", max_terms=2)
    with pytest.raises(ValueError, match="max_terms"):
        q.propagate(c, "Z0", max_terms=0)
    with pytest.raises(ValueError, match="min_coefficient"):
        q.propagate(c, "Z0", min_coefficient=-1.0)
    with pytest.raises(ValueError):
        q.propagate([("-0.1*X0", 1.0)], "Z0")
    with pytest.raises(ValueError):
        q.propagate([(q.Channel.depolarizing(0.1), [0, 1])], "Z0")
    with pytest.raises(ValueError):
        q.propagate(ghz_qasm(3).replace("h q[0];", "creg c[1];\nh q[0];\nmeasure q[0] -> c[0];"), "Z0")


def test_kicked_ising_on_heavy_hex_matches_aer():
    n, colours, centre = heavy_hex()
    assert n == 115 and centre == 57

    def kick(theta):
        return [f"rx({theta!r}) q[{k}];" for k in range(n)]

    def colour(c):
        return [f"rzz(-pi/2) q[{a}],q[{b}];" for a, b in colours[c]]

    one_step = [g for c in range(3) for g in colour(c)]
    # Light-cone values of Aer and the string counts, without truncation.
    for theta, steps, expected, terms in [
        (math.pi / 4, 4, 237 / 512, 1620),
        (1.0, 5, 0.1515659742586892, 134744),
    ]:
        gates = [g for _ in range(steps) for g in kick(theta) + one_step]
        r = q.propagate(kicked_ising_qasm(n, gates), f"Z{centre}")
        assert abs(r["value"] - expected) <= 1e-10
        assert r["terms"] == terms and r["dropped_terms"] == 0
        assert r["truncation_bound"] < 1e-13
    # The uniform Pauli–Lindblad model after every colour, scaled by α
    # (values of qiskit-addon-obp, design §6.2).
    single = [(1e-3, f"{p}{k}") for k in range(n) for p in "XYZ"]

    def noisy(alphas):
        layers = []
        for _ in range(3):
            layers.append(kicked_ising_qasm(n, kick(1.0)))
            for c in range(3):
                layers.append(kicked_ising_qasm(n, colour(c)))
                pairs = [(2e-4, f"X{a} X{b}") for a, b in colours[c]]
                pairs += [(5e-4, f"Z{a} Z{b}") for a, b in colours[c]]
                layers += [(single + pairs, alpha) for alpha in alphas]
        return q.propagate(layers, f"Z{centre}")["value"]

    for alpha, expected in [
        (0.0, 0.4044866033136744),
        (1.0, 0.3825995702888403),
        (2.0, 0.3620177587647080),
        (-1.0, 0.4277696033925958),
    ]:
        assert abs(noisy([alpha]) - expected) <= 1e-13
    assert abs(noisy([1.0, -1.0]) - 0.4044866033136744) <= 1e-13


def test_review_regressions():
    # Final measurements are left out, as on the command line; a
    # measurement followed by a gate on its qubit is refused.
    measured = ghz_qasm(2).replace("qreg q[2];", "qreg q[2];\ncreg c[2];") + "measure q -> c;\n"
    assert q.propagate(measured, "Z0 Z1")["value"] == 1.0
    assert q.propagate(q.Program.from_qasm(measured), "Z0 Z1")["value"] == 1.0
    mid = measured.replace("h q[0];", "h q[0];\nmeasure q[0] -> c[0];\nh q[0];")
    with pytest.raises(ValueError, match="measured"):
        q.propagate(mid, "Z0")
    # Several layers in a tuple: the message says to use a list.
    c = q.Circuit(2).h(0)
    with pytest.raises(ValueError, match="list"):
        q.propagate((c, c), "Z0 Z1")
    # An empty sum is +0.
    for r in [
        q.propagate(q.Circuit(1), ""),
        q.propagate([("0.1*X0 + 0.3*X1", -1.0)], "Z0", min_coefficient=2.0),
    ]:
        assert r["value"] == 0.0 and math.copysign(1.0, r["value"]) == 1.0
    # Overflow is an error, not inf or NaN.
    for gates in [q.Circuit(1).h(0).h(0), q.Circuit(1).ry(0, 0.3)]:
        for keep in [False, True]:
            with pytest.raises(ValueError, match="factor"):
                q.propagate([gates, ("300*X0", -1.0)], "1e100*Z0", return_operator=keep)
    # Large observables and rates in list form are merged at once.
    n = 4000
    obs = [(1.0, f"Z{k}") for k in range(n)]
    rates = [(1e-3, f"{p}{k}") for k in range(n) for p in "XY"]
    r = q.propagate([q.Circuit(1), (rates, 1.0)], obs, n_qubits=n)
    assert abs(r["value"] - n * math.exp(-4e-3)) <= 1e-9 * n
