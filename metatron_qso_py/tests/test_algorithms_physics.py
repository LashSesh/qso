"""Hamiltonian simulation, chemistry, excited states, QUBO/QAOA and
amplitude estimation."""

import json
from pathlib import Path

import numpy as np
import pytest

import metatron_qso as q

MOLECULES = Path(__file__).resolve().parents[2] / "metatron-qso-rs" / "examples" / "molecules"


def expm(h: np.ndarray, t: float) -> np.ndarray:
    """exp(-iHt) of a Hermitian matrix."""
    w, v = np.linalg.eigh(h)
    return (v * np.exp(-1j * w * t)) @ v.conj().T


def unitary(c: q.Circuit) -> np.ndarray:
    n = c.n_qubits
    cols = [q.simulate(_basis(n, k).append(c)).amplitudes for k in range(2**n)]
    return np.array(cols).T


def _basis(n: int, k: int) -> q.Circuit:
    c = q.Circuit(n)
    for b in range(n):
        if (k >> b) & 1:
            c.x(b)
    return c


def test_pauli_rotation_and_trotter_bounds():
    c = q.Circuit(3).pauli_rotation("X0 Y1 Z2", 0.7)
    h = q.PauliSum("X0 Y1 Z2")
    assert np.allclose(unitary(c), expm(h.to_matrix(3), 0.35), atol=1e-12)
    with pytest.raises(ValueError):
        q.Circuit(2).pauli_rotation("2 X0", 0.1)
    with pytest.raises(ValueError):
        q.Circuit(2).pauli_rotation("X0 + Z1", 0.1)
    ham = q.heisenberg_xxz(3, 1.0, 0.6, False)
    exact = expm(ham.to_matrix(3), 1.0)
    errors = []
    for order in (1, 2, 4):
        u = unitary(q.trotter_circuit(ham, 3, 1.0, steps=8, order=order))
        err = np.linalg.norm(u - exact, 2)
        assert err <= q.trotter_error_bound(ham, 1.0, steps=8, order=order)
        errors.append(err)
    assert errors[0] > errors[1] > errors[2]
    assert q.qdrift_error_bound(ham, 1.0, 1000) < q.qdrift_error_bound(ham, 1.0, 100)
    assert len(q.qdrift_circuit(ham, 3, 1.0, 50, seed=1)) >= 50
    with pytest.raises(ValueError):
        q.trotter_circuit(ham, 3, 1.0, steps=1, order=3)


def test_fermions_and_molecules():
    a = q.FermionOperator("[0]")
    b = q.FermionOperator("[0^]")
    assert (a * b + b * a).normal_ordered() == q.FermionOperator("1 []")
    hop = q.FermionOperator("0.5 [1^ 0] + 0.5 [0^ 1]")
    assert hop.is_hermitian()
    for mapping in ("jordan_wigner", "parity", "bravyi_kitaev"):
        h = hop.to_qubit(mapping).to_pauli_sum()
        assert sorted(np.linalg.eigvalsh(h.to_matrix(2))) == pytest.approx([-0.5, 0, 0, 0.5])
    with pytest.raises(ValueError):
        q.FermionOperator("[0^").normal_ordered()
    with pytest.raises(ValueError):
        q.FermionOperator("[1^ 0]").to_qubit("jw").to_pauli_sum()

    ref = json.loads((MOLECULES / "reference.json").read_text())
    lih = q.MolecularIntegrals.from_file(MOLECULES / "lih_sto3g.fcidump")
    assert (lih.n_orbitals, lih.n_alpha, lih.n_beta) == (6, 2, 2)
    assert lih.hartree_fock_energy() == pytest.approx(ref["lih_sto3g"]["hartree_fock"], abs=1e-9)
    assert lih.fci_energies(2) == pytest.approx(ref["lih_sto3g"]["fci"][:2], abs=1e-8)
    h = lih.qubit_hamiltonian("parity", two_qubit_reduction=True)
    hf = lih.hartree_fock_circuit("parity", two_qubit_reduction=True)
    assert hf.n_qubits == 10
    assert h.expectation(q.simulate(hf)) == pytest.approx(lih.hartree_fock_energy(), abs=1e-9)
    again = q.MolecularIntegrals.from_fcidump(lih.to_fcidump())
    assert again.two_body(0, 1, 0, 1) == pytest.approx(lih.two_body(0, 1, 0, 1))
    with pytest.raises(ValueError):
        lih.qubit_hamiltonian("jordan_wigner", two_qubit_reduction=True)
    with pytest.raises(ValueError):
        q.MolecularIntegrals.from_fcidump("&FCI NORB=2 &END\n")

    h2 = q.MolecularIntegrals.from_file(MOLECULES / "h2_sto3g.fcidump")
    fci = h2.fci_energies()[0]
    ansatz = q.uccsd(h2, "bravyi_kitaev")
    res = q.run_vqe(h2.qubit_hamiltonian("bravyi_kitaev"), ansatz,
                    initial_parameters=np.zeros(ansatz.num_parameters))
    assert res["energy"] == pytest.approx(fci, abs=1e-8)
    assert q.k_upccgsd(h2, 2).num_parameters == 4
    adapt = q.adapt_vqe(h2, "parity", two_qubit_reduction=True, gradient_tolerance=1e-6)
    assert adapt["converged"] and adapt["energy"] == pytest.approx(fci, abs=1e-9)
    assert adapt["circuit"].n_qubits == 2


def test_geometry_and_excited_states():
    c = q.efficient_su2(2, 1)
    theta = np.linspace(-1.0, 1.0, c.num_parameters)
    f = q.quantum_fisher_information(c, theta)
    assert f.shape == (c.num_parameters, c.num_parameters)
    assert np.allclose(f, f.T) and np.linalg.eigvalsh(f).min() > -1e-10
    ham = q.h2_sto3g()["hamiltonian"]
    ng = q.natural_gradient(c, ham, theta)
    assert ng["direction"].shape == theta.shape
    ra = q.real_amplitudes(2, 2)
    qng = q.quantum_natural_gradient(ra, ham, np.full(ra.num_parameters, 0.1), learning_rate=0.2)
    exact = np.linalg.eigvalsh(ham.to_matrix(2))
    assert qng["converged"] and qng["energy"] == pytest.approx(exact[0], abs=1e-7)
    ite = q.imaginary_time_evolution(ra, ham, np.full(ra.num_parameters, 0.1), time=1.0, steps=20)
    assert ite["parameters"].shape == (21, ra.num_parameters)
    assert np.all(np.diff(ite["energies"]) <= 1e-12)
    ansatz = q.efficient_su2(2, 2)
    v = q.vqd(ham, ansatz, 4, restarts=6, seed=2)
    s = q.ssvqe(ham, ansatz, 4, restarts=6, seed=2)
    assert v["energies"] == pytest.approx(exact, abs=1e-6)
    assert s["energies"] == pytest.approx(exact, abs=1e-6)
    with pytest.raises(ValueError):
        q.vqd(ham, ansatz, 0)


def test_qubo_qaoa_and_amplitude_estimation():
    rng = np.random.default_rng(3)
    m = rng.uniform(-1, 1, (5, 5))
    qubo = q.Qubo(m, offset=0.5)
    best, arg = qubo.brute_force()
    assert qubo.value(arg[0]) == pytest.approx(best)
    assert qubo.value([1, 0, 1, 0, 0]) == pytest.approx(qubo.value(0b00101))
    ising = qubo.to_ising()
    assert len(ising["h"]) == 5
    cost = qubo.to_pauli_sum()
    diag = np.real(np.diag(cost.to_matrix(5)))
    assert diag.min() == pytest.approx(best)
    xy = q.Qaoa(cost, 5, 2, mixer="xy_ring", weight=2)
    r = xy.optimize(restarts=4, seed=1)
    assert r["feasible_probability"] == pytest.approx(1.0)
    warm = q.Qaoa(cost, 5, 1, mixer="warm_start", warm_start=[0.5] * 5)
    assert warm.analyze([0.0, 0.0])["expectation"] == pytest.approx(
        np.mean([qubo.value(x) for x in range(32)]))
    with pytest.raises(ValueError):
        q.Qaoa(cost, 5, 1, mixer="xy_ring")
    with pytest.raises(ValueError):
        q.Qaoa(q.PauliSum("X0"), 1, 1)

    problem = q.AmplitudeProblem.bernoulli(0.3)
    assert problem.exact() == pytest.approx(0.3)
    can = q.canonical_amplitude_estimation(problem, 6)
    assert abs(can["estimate"] - 0.3) < np.pi / 64
    it = q.iterative_amplitude_estimation(problem, epsilon=0.005, alpha=0.05, shots=100, seed=1)
    assert it["interval"][0] <= 0.3 <= it["interval"][1]
    ml = q.ml_amplitude_estimation(problem, levels=6, shots=100, seed=1)
    assert abs(ml["estimate"] - 0.3) < 0.01
    with pytest.raises(ValueError):
        q.AmplitudeProblem(q.Circuit(1), 2)
