"""OpenQASM 2 and 3 programs: the OpenQASM project's examples, round trips
and validation (the same vectors as metatron-qso-rs/tests/openqasm.rs)."""

import math
from pathlib import Path

import numpy as np
import pytest

import metatron_qso as q

DATA = Path(__file__).resolve().parents[2] / "metatron-qso-rs" / "tests" / "data" / "openqasm"


def read(version: str, name: str) -> str:
    return (DATA / version / f"{name}.qasm").read_text()


def same(a: dict[str, float], b: dict[str, float]) -> bool:
    return a.keys() == b.keys() and all(abs(a[k] - b[k]) < 1e-10 for k in a)


def test_openqasm2_examples() -> None:
    adder = q.Program.from_qasm(read("v2", "adder"))
    assert adder.n_qubits == 10
    (record, p), = adder.probabilities().items()
    assert int(record, 2) == 16 and p == pytest.approx(1.0)
    # Teleportation: bit 2 reads 1 with probability sin²(0.15).
    tele = q.Program.from_qasm(read("v2", "teleport"))
    p1 = sum(p for k, p in tele.probabilities().items() if k[-3] == "1")
    assert p1 == pytest.approx(math.sin(0.15) ** 2, abs=1e-12)
    assert [r["name"] for r in tele.bit_registers] == ["c0", "c1", "c2"]
    for name in ["invalid_gate_no_found", "invalid_missing_semicolon"]:
        with pytest.raises(ValueError):
            q.Program.from_qasm(read("v2", name))


def test_openqasm3_examples_agree_with_openqasm2() -> None:
    for name in ["teleport", "inverseqft1", "inverseqft2", "qft", "rb", "qpt"]:
        v3 = q.Program.from_qasm(read("v3", name)).probabilities()
        v2 = q.Program.from_qasm(read("v2", name)).probabilities()
        assert same(v3, v2), name
    with pytest.raises(ValueError, match="not supported|unsupported"):
        q.Program.from_qasm(read("v3", "vqe"))


def test_round_trips_and_sampling() -> None:
    p = q.Program.from_qasm(read("v2", "inverseqft1"))
    assert not p.is_unitary
    d = p.probabilities()
    assert same(q.Program.from_qasm(p.to_qasm3()).probabilities(), d)
    assert same(q.Program.from_qasm(p.to_qasm2()).probabilities(), d)
    counts = p.sample(2000, seed=5)
    assert sum(counts.values()) == 2000 and counts == p.sample(2000, seed=5)
    assert set(counts) <= set(d)
    with pytest.raises(ValueError, match="measurements"):
        p.to_circuit()


def test_dynamic_circuit_feed_forward() -> None:
    # Measure-and-correct: q[1] ends in |1⟩ whatever q[0] read.
    src = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c[0] = measure q[0];
if (c[0] == 0) x q[1];
c[1] = measure q[1];
reset q[0];
"""
    p = q.Program.from_qasm(src)
    assert p.probabilities() == pytest.approx({"10": 0.5, "11": 0.5})
    assert len(p) == 7


def test_circuit_conversions() -> None:
    c = q.Circuit(3).h(0).cx(0, 1).ry(2, 0.4)
    p = q.Program.from_circuit(c, measure=True)
    assert (p.n_qubits, p.n_bits) == (3, 3)
    exact = q.simulate(c).probabilities()
    for k, v in p.probabilities().items():
        assert v == pytest.approx(exact[int(k, 2)], abs=1e-12)
    # A unitary program converts back; the OpenQASM 3 export of a circuit
    # reads back to the same state.
    u = q.Program.from_circuit(c)
    assert u.is_unitary and u.n_bits == 0
    back = q.Program.from_qasm(c.to_qasm3()).to_circuit()
    f = q.simulate(back).fidelity(q.simulate(c))
    assert f == pytest.approx(1.0, abs=1e-12)
    assert np.isclose(
        q.simulate(u.to_circuit()).probabilities(), exact
    ).all()
    assert repr(p) == "Program(n_qubits=3, n_bits=3, operations=6)"
