import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import metatron_qso as q


def test_h2_vqe_reaches_full_ci_energy():
    h2 = q.h2_sto3g()
    assert h2["n_qubits"] == 2 and isinstance(h2["hamiltonian"], q.PauliSum)
    result = q.run_vqe(h2["hamiltonian"], q.real_amplitudes(2, 1), restarts=4, seed=0)
    total = result["energy"] + h2["nuclear_repulsion"]
    assert total == pytest.approx(-1.137306, abs=1e-5)
    assert h2["reference_total_energy"] == pytest.approx(-1.137306)
    assert result["converged"]
    params = result["parameters"]
    assert isinstance(params, np.ndarray) and params.dtype == np.float64 and params.shape == (4,)
    assert result["history"].dtype == np.float64
    assert result["restart_energies"].shape == (4,)
    # The returned parameters reproduce the energy.
    state = q.simulate(q.real_amplitudes(2, 1), params)
    assert state.expectation(h2["hamiltonian"]) == pytest.approx(result["energy"], abs=1e-12)
    exact = h2["hamiltonian"].ground_state_energy()
    assert result["energy"] == pytest.approx(exact, abs=1e-8)


@pytest.mark.parametrize("method", ["adam", "nelder_mead", "spsa"])
def test_vqe_other_optimisers(method):
    h = q.PauliSum("Z0 + 0.5 X0")
    r = q.run_vqe(h, q.real_amplitudes(1, 1), method=method, restarts=2, max_iterations=3000)
    assert r["energy"] == pytest.approx(-math.sqrt(1.25), abs=1e-3)


def test_vqe_initial_parameters_and_gradient_choice():
    h = q.transverse_field_ising(3, 1.0, 0.8)
    ansatz = q.efficient_su2(3, 2)
    exact = h.ground_state_energy()
    r = q.run_vqe(
        h,
        ansatz,
        restarts=2,
        gradient="parameter_shift",
        initial_parameters=np.zeros(ansatz.num_parameters),
    )
    assert r["energy"] >= exact - 1e-9
    assert r["energy"] == pytest.approx(exact, abs=1e-4)


def gradient_test_circuit():
    c = q.Circuit(3)
    t = c.new_parameters(6)
    c.h(0).ry(1, t[0]).rx(2, 2.0 * t[1] - 0.3).cx(0, 1)
    c.crx(1, 2, t[2]).cp(2, 0, -t[3]).rzz(0, 2, t[4] + 0.1)
    c.u(1, t[5], 0.3, t[0]).ryy(1, 2, t[2]).cry(0, 1, 0.5 * t[3])
    return c


@pytest.mark.parametrize("method", ["adjoint", "parameter_shift"])
def test_exact_gradients_match_finite_differences(method):
    c = gradient_test_circuit()
    h = q.PauliSum("0.7 Z0 Z1 - 0.4 X2 + 0.25 Y0 Y2 + 0.1 Z1")
    theta = np.array([0.3, -0.7, 1.1, 0.25, -1.4, 0.8])
    value, grad = q.expectation_and_gradient(c, h, theta, method=method)
    fd_value, fd = q.expectation_and_gradient(c, h, theta, method="finite_difference", step=1e-6)
    assert grad.dtype == np.float64 and grad.shape == (6,)
    assert value == pytest.approx(fd_value, abs=1e-12)
    assert value == pytest.approx(q.simulate(c, theta).expectation(h), abs=1e-12)
    assert np.allclose(grad, fd, atol=1e-7)


def test_gradient_with_initial_state():
    c = q.Circuit(1)
    c.ry(0, c.new_parameter())
    one = q.StateVector.basis(1, 1)
    value, grad = q.expectation_and_gradient(c, q.PauliSum("Z0"), [0.4], initial_state=one)
    assert value == pytest.approx(-math.cos(0.4))
    assert grad[0] == pytest.approx(math.sin(0.4))


def test_qaoa_ring_p1_ratio_is_three_quarters():
    qaoa = q.MaxCutQaoa(q.Graph.cycle(8), 1)
    assert (qaoa.max_cut, qaoa.num_parameters, qaoa.p, qaoa.n_qubits) == (8.0, 2, 1, 8)
    r = qaoa.optimize()
    assert r["approximation_ratio"] == pytest.approx(0.75, abs=1e-6)
    assert r["expectation"] == pytest.approx(6.0, abs=1e-5)
    assert r["gammas"].dtype == np.float64 and r["gammas"].shape == (1,)
    assert r["most_likely_bitstring"] in ("10101010", "01010101")
    assert r["most_likely_cut"] == 8.0
    assert sorted(qaoa.optimal_cuts()) == [0b01010101, 0b10101010]
    assert qaoa.cut_value(0b01010101) == 8.0


def test_qaoa_hypercube_q3_p1():
    r = q.MaxCutQaoa(q.Graph.hypercube(3), 1).optimize()
    assert r["approximation_ratio"] == pytest.approx(0.69245, abs=1e-5)


def test_qaoa_gradient_and_state_consistency():
    g = q.Graph.from_edges(5, [(0, 1, 1.0), (1, 2, 0.5), (2, 3, 2.0), (3, 4, 1.0), (4, 0, 1.5)])
    qaoa = q.MaxCutQaoa(g, 2)
    x = np.array([0.4, 0.9, 0.3, 0.2])
    value, grad = qaoa.expectation_and_gradient(x)
    assert value == pytest.approx(qaoa.expectation(x))
    h = 1e-6
    fd = [
        (qaoa.expectation(x + h * e) - qaoa.expectation(x - h * e)) / (2 * h)
        for e in np.eye(4)
    ]
    assert np.allclose(grad, fd, atol=1e-7)
    # Explicit gate circuit and diagonal cost operator agree with the fast simulator.
    circuit = q.qaoa_maxcut_circuit(g, 2)
    state = q.simulate(circuit, x)
    assert state.expectation(q.maxcut_cost(g)) == pytest.approx(value, abs=1e-10)
    assert state.fidelity(qaoa.state(x)) == pytest.approx(1.0, abs=1e-10)
    stats = qaoa.analyze(x)
    assert stats["expectation"] == pytest.approx(value)


def test_qaoa_layerwise_improves_with_depth():
    results = q.optimize_qaoa_layerwise(q.Graph.cycle(6), 3, restarts=2)
    ratios = [r["approximation_ratio"] for r in results]
    assert [r["p"] for r in results] == [1, 2, 3]
    assert ratios[0] == pytest.approx(0.75, abs=1e-6)
    assert all(b >= a - 1e-9 for a, b in zip(ratios, ratios[1:]))
    assert ratios[-1] == pytest.approx(1.0, abs=1e-6)  # p = n/2 solves the even ring


def test_threads_can_run_concurrently():
    qaoa = q.MaxCutQaoa(q.Graph.cycle(10), 2)
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda s: qaoa.optimize(restarts=2, seed=s), range(4)))
    assert all(r["approximation_ratio"] > 0.8 for r in results)


@pytest.mark.parametrize("n, marked", [(3, [5]), (5, [3, 17]), (6, [0, 9, 33, 62])])
def test_grover_follows_sine_law(n, marked):
    theta = math.asin(math.sqrt(len(marked) / 2**n))
    for k in range(6):
        r = q.grover_search(n, marked, iterations=k)
        expected = math.sin((2 * k + 1) * theta) ** 2
        assert r["success_probability"] == pytest.approx(expected, abs=1e-12)
        assert r["theoretical_success_probability"] == pytest.approx(expected, abs=1e-12)
    best = q.grover_search(n, marked)
    assert best["success_probability"] > 0.9
    circuit = q.grover_circuit(n, marked, best["iterations"])
    p = q.simulate(circuit).probabilities()
    assert sum(p[m] for m in marked) == pytest.approx(best["success_probability"])


def test_qft_is_the_dft_matrix():
    n = 4
    dim = 2**n
    c = q.qft(n)
    dft = np.exp(2j * np.pi * np.outer(np.arange(dim), np.arange(dim)) / dim) / math.sqrt(dim)
    for x in range(dim):
        out = q.StateVector.basis(n, x).apply(c).amplitudes
        assert np.allclose(out, dft[:, x], atol=1e-12)
    psi = q.StateVector.random(n, seed=9)
    back = psi.copy().apply(c).apply(q.inverse_qft(n))
    assert back.fidelity(psi) == pytest.approx(1.0, abs=1e-12)


def test_phase_estimation_is_exact_for_dyadic_phases():
    phi = 0.3125  # 5/16
    u = np.diag([1, np.exp(2j * np.pi * phi)])
    r = q.estimate_phase(4, u, [0, 1])
    assert r["phase"] == pytest.approx(phi)
    assert r["probability"] == pytest.approx(1.0, abs=1e-12)
    dist = r["distribution"]
    assert dist.dtype == np.float64 and dist.shape == (16,)
    # Non-dyadic phase: most likely outcome is the nearest 4-bit fraction.
    r = q.estimate_phase(4, np.diag([1, np.exp(2j * np.pi * 0.3)]), [0, 1])
    assert r["phase"] == pytest.approx(5 / 16)
    assert r["probability"] > 4 / math.pi**2


def test_variational_classifier_learns_blobs():
    rng = np.random.default_rng(1)
    ys = np.arange(40) % 2
    centres = np.where(ys[:, None] == 1, 0.8, -0.8)
    xs = centres + 0.3 * rng.standard_normal((40, 2))
    clf = q.VariationalClassifier(2, 2, 2)
    assert clf.num_parameters == 8
    report = clf.fit(xs, ys, max_iterations=150, seed=0)
    assert report["accuracy"] >= 0.9
    assert clf.accuracy(xs, ys) == pytest.approx(report["accuracy"])
    probs = clf.predict_proba(xs)
    assert probs.dtype == np.float64 and probs.shape == (40,)
    assert ((probs >= 0) & (probs <= 1)).all()
    single = clf.predict_proba(xs[0])
    assert isinstance(single, float) and single == pytest.approx(probs[0])
    labels = clf.predict(xs)
    assert labels.dtype == np.bool_ and np.mean(labels == ys) >= 0.9
    assert isinstance(clf.predict(xs[0]), bool)
    # Parameters round-trip.
    clone = q.VariationalClassifier(2, 2, 2)
    clone.parameters = clf.parameters
    assert np.allclose(clone.predict_proba(xs), probs)


@pytest.mark.parametrize(
    "call",
    [
        lambda: q.real_amplitudes(2, 1, entanglement="star"),
        lambda: q.hardware_efficient(2, 1, rotations=["ry", "rq"]),
        lambda: q.expectation_and_gradient(
            q.real_amplitudes(2, 1), q.PauliSum("Z0"), [0.0] * 3
        ),
        lambda: q.expectation_and_gradient(
            q.real_amplitudes(2, 1), q.PauliSum("Z0"), [0.0] * 4, method="magic"
        ),
        lambda: q.run_vqe(q.PauliSum("Z0"), q.Circuit(1).h(0)),
        lambda: q.run_vqe(q.PauliSum("Z0"), q.real_amplitudes(1, 1), method="magic"),
        lambda: q.run_vqe(q.PauliSum("Z0"), q.real_amplitudes(1, 1), initial_parameters=[1.0]),
        lambda: q.run_vqe(q.PauliSum("Z3"), q.real_amplitudes(1, 1)),
        lambda: q.MaxCutQaoa(q.Graph(3), 1),
        lambda: q.MaxCutQaoa(q.Graph.cycle(3), 0),
        lambda: q.MaxCutQaoa(q.Graph.cycle(3), 1).expectation([0.1]),
        lambda: q.MaxCutQaoa(q.Graph.cycle(3), 1).optimize(initial=[0.1]),
        lambda: q.grover_search(3, [8]),
        lambda: q.grover_search(3, []),
        lambda: q.qft(0),
        lambda: q.estimate_phase(3, np.eye(4), [1, 0]),
        lambda: q.estimate_phase(3, [[1, 1], [0, 1]], [1, 0]),
        lambda: q.VariationalClassifier(2, 2, 1).predict_proba([0.1, 0.2]),
        lambda: q.VariationalClassifier(2, 2, 1).fit([[0.0, 1.0]], [2]),
        lambda: q.VariationalClassifier(2, 2, 1).fit([[0.0, np.nan]], [1]),
        lambda: q.VariationalClassifier(2, 2, 1).fit([[0.0, 1.0, 2.0]], [1]),
    ],
)
def test_invalid_algorithm_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()
