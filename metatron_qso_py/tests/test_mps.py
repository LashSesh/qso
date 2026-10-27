"""Matrix-product states against the state vector and closed forms (the same
checks as metatron-qso-rs/src/mps and tests/validation.rs)."""

import numpy as np
import pytest

import metatron_qso as q


def test_a_circuit_matches_the_state_vector() -> None:
    c = q.Circuit(6).h(0).cx(0, 5).ry(2, 0.7).cz(1, 4).rzz(3, 0, 0.4).t(5).swap(1, 3).ccx(0, 2, 4)
    psi = q.simulate(c)
    m = q.MatrixProductState.from_circuit(c)
    assert m.fidelity_lower_bound == pytest.approx(1.0, abs=1e-12)
    assert m.to_statevector().fidelity(psi) == pytest.approx(1.0, abs=1e-12)
    amps = psi.amplitudes
    assert m.amplitude("000000") == pytest.approx(amps[0], abs=1e-12)
    assert m.amplitude("100001") == pytest.approx(amps[0b100001], abs=1e-12)
    for pauli in ["Z0 Z5", "X1", "Y2 X4", "Z3"]:
        assert m.expectation(pauli) == pytest.approx(q.PauliSum(pauli).expectation(psi), abs=1e-10)
    h = q.PauliSum("0.5*Z0 Z5 - X2")
    assert m.expectation(h) == pytest.approx(h.expectation(psi), abs=1e-10)
    assert m.expectation([(0.5, "Z0 Z5"), (-1.0, "X2")]) == pytest.approx(h.expectation(psi), abs=1e-10)
    assert len(m.bond_dimensions()) == 5


def test_ghz_on_a_thousand_qubits() -> None:
    n = 1000
    m = q.MatrixProductState(n, seed=1).h(0)
    for k in range(1, n):
        m.cx(k - 1, k)
    assert m.max_bond_dimension == 2
    assert m.expectation("Z0 Z999") == pytest.approx(1.0)
    assert m.expectation("Z500") == pytest.approx(0.0, abs=1e-12)
    assert all(s == pytest.approx(1.0) for s in m.bond_entropies())
    counts = m.sample(50, seed=2)
    assert set(counts) <= {"0" * n, "1" * n}
    first = m.copy().measure(0)
    assert first in (0, 1)


def test_truncation_is_bounded() -> None:
    rng = np.random.default_rng(3)
    c = q.Circuit(10)
    for _ in range(60):
        a, b = rng.choice(10, size=2, replace=False)
        c.ry(int(a), float(rng.uniform(-3, 3))).cx(int(a), int(b))
    psi = q.simulate(c)
    m = q.MatrixProductState.from_circuit(c, max_bond=4)
    assert m.max_bond_dimension <= 4
    assert m.truncations > 0 and m.discarded_weight > 0
    bound = m.fidelity_lower_bound
    assert bound is not None and m.to_statevector().fidelity(psi) >= bound - 1e-12


def test_programs_run_on_the_mps() -> None:
    p = q.Program.from_qasm(
        'OPENQASM 3.0; include "stdgates.inc"; qubit[60] q; bit[2] c;'
        "h q[0]; cx q[0], q[59]; t q[7]; c[0] = measure q[0]; c[1] = measure q[59];"
    )
    counts = p.sample(400, seed=1, method="mps")
    assert set(counts) == {"00", "11"}
    with pytest.raises(ValueError, match="unknown method"):
        p.sample(10, method="tensor")


def test_invalid_input_raises() -> None:
    with pytest.raises(ValueError):
        q.MatrixProductState(0)
    with pytest.raises(ValueError):
        q.MatrixProductState(3, max_bond=0)
    m = q.MatrixProductState(3)
    with pytest.raises(ValueError):
        m.cx(0, 3)
    with pytest.raises(ValueError):
        m.apply_matrix(np.ones((2, 2)), [0])
    with pytest.raises(ValueError):
        m.amplitude("01")
    with pytest.raises(ValueError):
        q.MatrixProductState(30).to_statevector()
