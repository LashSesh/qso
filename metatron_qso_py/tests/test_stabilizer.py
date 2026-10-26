"""Stabilizer simulation: tableaux against the state vector, graph states
and Clifford programs (the same checks as metatron-qso-rs/src/stabilizer)."""

import numpy as np
import pytest

import metatron_qso as q


def test_clifford_circuit_matches_the_state_vector() -> None:
    c = q.Circuit(4).h(0).cx(0, 1).s(1).sx(2).cz(2, 3).ry(3, np.pi / 2).swap(0, 3)
    assert q.is_clifford(c)
    s = q.StabilizerState.from_circuit(c)
    psi = q.simulate(c)
    assert s.to_statevector().fidelity(psi) == pytest.approx(1.0, abs=1e-12)
    for pauli in ["Z0 Z1", "X2", "Y1 X0", "Z3"]:
        assert s.expectation(pauli) == pytest.approx(q.PauliSum(pauli).expectation(psi), abs=1e-12)
    h = q.PauliSum("0.5*Z0 Z1 - X2")
    assert s.expectation(h) == pytest.approx(h.expectation(psi), abs=1e-12)
    assert not q.is_clifford(q.Circuit(1).t(0))
    with pytest.raises(ValueError, match="Clifford"):
        q.StabilizerState.from_circuit(q.Circuit(1).rz(0, 0.3))


def test_ghz_on_a_thousand_qubits() -> None:
    n = 1000
    s = q.StabilizerState(n, seed=3).h(0)
    for k in range(1, n):
        s.cx(k - 1, k)
    assert s.expectation("Z0 Z999") == 1.0
    assert s.expectation("Z500") == 0.0
    counts = s.sample(100, seed=1)
    assert set(counts) == {"0" * n, "1" * n}
    v = s.measure(10)
    assert all(s.measure(k) == v for k in (0, 500, 999))
    assert s.measure_pauli("-Z0") == 1 - v


def test_graph_states_and_local_complementation() -> None:
    g = q.Graph.from_spec("petersen")
    s = q.StabilizerState.graph_state(g)
    assert len(s.stabilizers()) == 10
    part = [0, 1, 2, 3, 4]
    assert s.entanglement_entropy(part) == pytest.approx(
        s.to_statevector().entanglement_entropy(part), abs=1e-8
    )
    h = q.local_complementation(g, 0)
    t = s.copy().local_complement(g, 0)
    assert t.same_state(q.StabilizerState.graph_state(h))
    star = q.local_complementation(q.Graph.from_spec("metatron"), 0)
    assert star.num_edges == 12


def test_clifford_programs_choose_the_stabilizer_simulator() -> None:
    n = q.MAX_QUBITS + 10
    src = f"qubit[{n}] q; bit[2] c; h q[0];" + "".join(
        f" cx q[{k - 1}], q[{k}];" for k in range(1, n)
    ) + f" c[0] = measure q[0]; c[1] = measure q[{n - 1}];"
    p = q.Program.from_qasm(src)
    assert p.is_clifford
    assert set(p.sample(200, seed=1)) == {"00", "11"}
    with pytest.raises(ValueError):
        p.sample(10, method="statevector")
    small = q.Program.from_qasm("qubit q; bit c; h q; c = measure q;")
    assert small.sample(500, seed=2, method="stabilizer").keys() == {"0", "1"}
    with pytest.raises(ValueError, match="method"):
        small.sample(10, method="tableau")


def test_invalid_input() -> None:
    with pytest.raises(ValueError):
        q.StabilizerState(0)
    s = q.StabilizerState(2)
    with pytest.raises(ValueError):
        s.h(2)
    with pytest.raises(ValueError):
        s.expectation("X0 X0")
    with pytest.raises(ValueError):
        s.sample(10**8)
    with pytest.raises(ValueError):
        q.StabilizerState(21).to_statevector()


def test_review_regressions() -> None:
    c = q.Circuit(1)
    p = c.new_parameter()
    c.rx(0, p)
    assert not q.is_clifford(c)
    with pytest.raises(ValueError):
        q.StabilizerState.from_circuit(c)
    assert q.is_clifford(c, [np.pi / 2])
    assert not q.is_clifford(q.Circuit(1).rx(0, 1e9))
    with pytest.raises(ValueError, match="shots"):
        q.StabilizerState(16384).sample(5000)
