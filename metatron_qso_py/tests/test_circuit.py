import math

import numpy as np
import pytest
from conftest import I2, X, Z, assert_same_up_to_phase, kron_all

import metatron_qso as q


def test_builder_chains_and_counts():
    c = q.Circuit(3)
    a, b = c.new_parameters(2)
    out = c.h(0).cx(0, 1).ry(2, a).rzz(1, 2, 2.0 * b + 0.5).ccx(0, 1, 2)
    assert out is c
    assert (len(c), c.depth, c.num_parameters, c.n_qubits) == (5, 4, 2, 3)
    assert c.gate_counts() == {"h": 1, "cx": 1, "ry": 1, "rzz": 1, "ccx": 1}
    name, targets, controls, params = c.instructions()[3]
    assert (name, targets, controls) == ("rzz", [1, 2], [])
    p = params[0]
    assert isinstance(p, q.Parameter)
    assert (p.index, p.scale, p.offset) == (1, 2.0, 0.5)
    bound = c.bind([0.1, 0.2])
    assert bound.num_parameters == 0
    assert bound.instructions()[3][3] == [pytest.approx(0.9)]


def test_parameter_arithmetic():
    c = q.Circuit(1)
    p = c.new_parameter()
    for expr, (scale, offset) in [
        (2.0 * p, (2.0, 0.0)),
        (p * 2.0, (2.0, 0.0)),
        (p + 0.5, (1.0, 0.5)),
        (0.5 + p, (1.0, 0.5)),
        (p - 0.5, (1.0, -0.5)),
        (0.5 - p, (-1.0, 0.5)),
        (-p, (-1.0, 0.0)),
        (p / 4, (0.25, 0.0)),
        (3 * (p + 1), (3.0, 3.0)),
    ]:
        assert (expr.scale, expr.offset) == (pytest.approx(scale), pytest.approx(offset))
        assert expr.value([0.7]) == pytest.approx(scale * 0.7 + offset)


def test_symbolic_and_bound_circuits_agree():
    c = q.Circuit(2)
    t = c.new_parameter()
    c.h(0).crx(0, 1, 2.0 * t - 0.3).u(1, t, 0.2, -t).cp(1, 0, t + 1.0)
    theta = [0.37]
    a = q.simulate(c, theta).amplitudes
    b = q.simulate(c.bind(theta)).amplitudes
    assert np.allclose(a, b, atol=1e-13)


def test_gate_matrices_follow_qiskit_conventions():
    # RX(θ) = exp(-iθX/2), little-endian qubit order.
    th = 0.81
    c = q.Circuit(2).rx(1, th)
    rx = math.cos(th / 2) * I2 - 1j * math.sin(th / 2) * X
    expected = kron_all(rx, I2)[:, 0]
    assert np.allclose(q.simulate(c).amplitudes, expected)
    # CX with control 0 maps |01> (index 1) to |11> (index 3).
    s = q.StateVector.basis(2, 1).apply(q.Circuit(2).cx(0, 1))
    assert np.allclose(s.probabilities(), [0, 0, 0, 1])


def test_custom_unitary_targets0_is_most_significant():
    # CNOT matrix with control = first target (most significant bit).
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    a = q.Circuit(2).x(1).unitary(cnot, [1, 0])  # control 1, target 0
    b = q.Circuit(2).x(1).cx(1, 0)
    assert np.allclose(q.simulate(a).amplitudes, q.simulate(b).amplitudes)
    # Controlled custom unitary.
    c = q.Circuit(2).x(0).unitary(X, [1], controls=[0])
    assert np.allclose(q.simulate(c).probabilities(), [0, 0, 0, 1])


def test_inverse_undoes_circuit():
    c = q.efficient_su2(3, 2, "full").bind(np.linspace(-1, 1, 18))
    c.ccx(0, 1, 2).cswap(2, 0, 1).sx(0).t(1).mcz([0, 1, 2])
    psi = q.StateVector.random(3, seed=1)
    out = psi.copy().apply(c).apply(c.inverse())
    assert out.fidelity(psi) == pytest.approx(1.0, abs=1e-12)


def test_append_and_append_mapped():
    bell = q.Circuit(2).h(0).cx(0, 1)
    c = q.Circuit(3).append_mapped(bell, [2, 0])
    p = q.simulate(c).probabilities()
    assert p[0] == pytest.approx(0.5) and p[0b101] == pytest.approx(0.5)
    d = q.Circuit(3).append(bell)
    assert len(d) == 2
    with pytest.raises(ValueError):
        q.Circuit(1).append(bell)


def full_gate_circuit():
    c = q.Circuit(4)
    c.h(0).x(1).y(2).z(3).s(0).sdg(1).t(2).tdg(3).sx(0).sxdg(1)
    c.rx(0, 0.3).ry(1, -1.2).rz(2, 2.2).p(3, 0.4).u(0, 0.5, 1.1, -0.7)
    c.rxx(0, 1, 0.6).ryy(1, 2, -0.8).rzz(2, 3, 1.3)
    c.cx(0, 1).cy(1, 2).cz(2, 3).ch(3, 0).swap(0, 2)
    c.ccx(0, 1, 2).cswap(3, 0, 1).mcx([0, 2], 3).mcz([0, 1, 3])
    c.crx(0, 3, 0.9).cry(1, 0, -0.4).crz(2, 1, 1.7).cp(3, 2, 0.25)
    c.unitary(np.array([[0, 1j], [1j, 0]]), [2])
    c.unitary(np.array([[1, 1], [1, -1]]) / math.sqrt(2), [1], controls=[0])
    return c


def test_qasm_round_trip_preserves_state_up_to_global_phase():
    c = full_gate_circuit()
    text = c.to_qasm()
    assert text.startswith("OPENQASM 2.0;")
    back = q.Circuit.from_qasm(text)
    assert back.n_qubits == 4
    psi0 = q.StateVector.random(4, seed=3)
    a = psi0.copy().apply(c).amplitudes
    b = psi0.copy().apply(back).amplitudes
    assert_same_up_to_phase(a, b)


def test_qasm_exports_multi_controlled_gates():
    c = q.Circuit(5)
    for k in range(5):
        c.ry(k, 0.3 + 0.1 * k)
    c.mcx([0, 1, 2, 3], 4).mcz([0, 1, 2, 3, 4]).mcx([1, 2, 3], 0)
    c.unitary(np.array([[0, 1j], [1j, 0]]), [2], controls=[0, 1, 4])
    again = q.Circuit.from_qasm(c.to_qasm())
    assert_same_up_to_phase(q.simulate(c).amplitudes, q.simulate(again).amplitudes)


def test_qasm_parameters_and_measurement():
    ansatz = q.real_amplitudes(2, 1)
    theta = [0.1, 0.2, 0.3, 0.4]
    text = ansatz.to_qasm(theta, measure=True)
    assert "creg c[2];" in text and "measure q -> c;" in text
    back = q.Circuit.from_qasm(text)
    assert_same_up_to_phase(q.simulate(ansatz, theta).amplitudes, q.simulate(back).amplitudes)
    parsed = q.Circuit.from_qasm(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\nrz(pi/4) q[1];\n'
    )
    assert parsed.gate_counts() == {"h": 1, "cx": 1, "rz": 1}


@pytest.mark.parametrize(
    "build",
    [
        lambda: q.Circuit(0),
        lambda: q.Circuit(q.MAX_QUBITS + 1),
        lambda: q.Circuit(2).h(2),
        lambda: q.Circuit(2).cx(1, 1),
        lambda: q.Circuit(1).rx(0, float("inf")),
        lambda: q.Circuit(2).mcz([]),
        lambda: q.Circuit(1).unitary([[1, 1], [1, 1]], [0]),
        lambda: q.Circuit(2).unitary(np.eye(4), [0]),
        lambda: q.Circuit(2).unitary(np.ones((2, 3)), [0]),
        lambda: q.real_amplitudes(2, 1).bind([0.1]),
        lambda: q.real_amplitudes(2, 1).to_qasm(),
        lambda: q.Circuit.from_qasm("OPENQASM 2.0;\nqreg q[2];\nfoo q[0];"),
        lambda: q.Circuit.from_qasm("OPENQASM 2.0;\nqreg q[64];\nh q[0];"),
        lambda: q.Circuit(1).new_parameter() / 0.0,
        # No OpenQASM 2 (qelib1) form for X with three controls.
    ],
)
def test_invalid_circuits_raise_value_error(build):
    with pytest.raises(ValueError):
        build()


def test_angle_type_error():
    with pytest.raises(TypeError):
        q.Circuit(1).rx(0, "pi")
