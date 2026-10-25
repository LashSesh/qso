import math

import numpy as np
import pytest

import metatron_qso as q


def test_metatron_cube_limiting_distribution():
    walk = q.ContinuousWalk(q.Graph.metatron_cube())
    limit = walk.limiting_distribution(0)
    assert limit.dtype == np.float64 and limit.shape == (13,)
    assert limit[0] == pytest.approx(145 / 169, abs=1e-12)
    assert np.allclose(limit[1:], 2 / 169, atol=1e-12)
    assert limit.sum() == pytest.approx(1.0)
    # Laplacian spectrum of K13: 0 once, 13 twelve times.
    assert np.allclose(walk.spectrum, [0] + [13] * 12)
    assert (walk.num_nodes, walk.gamma, walk.generator) == (13, 1.0, "laplacian")


def test_complete_graph_return_probability_closed_form():
    # K_N, Laplacian: P(start, t) = |(1 + (N−1)e^{−iNt})/N|².
    n, t = 13, 0.37
    p = q.ContinuousWalk(q.Graph.metatron_cube()).probabilities(0, t)
    expected = abs((1 + (n - 1) * np.exp(-1j * n * t)) / n) ** 2
    assert p[0] == pytest.approx(expected, abs=1e-12)
    assert p.sum() == pytest.approx(1.0, abs=1e-12)


def test_propagator_and_evolve():
    g = q.Graph.fruit_of_life()
    walk = q.ContinuousWalk(g, "adjacency", 0.7)
    u = walk.propagator(1.3)
    assert u.dtype == np.complex128 and u.shape == (13, 13)
    assert np.allclose(u.conj().T @ u, np.eye(13), atol=1e-12)
    w, v = np.linalg.eigh(-0.7 * g.adjacency_matrix())
    ref = v @ np.diag(np.exp(-1.3j * w)) @ v.T
    assert np.allclose(u, ref, atol=1e-12)
    psi0 = np.zeros(13, dtype=complex)
    psi0[[0, 3]] = [0.6, 0.8j]
    out = walk.evolve(psi0, 1.3)
    assert out.dtype == np.complex128 and np.allclose(out, ref @ psi0, atol=1e-12)


def test_time_average_converges_to_limit():
    walk = q.ContinuousWalk(q.Graph.fruit_of_life())
    limit = walk.limiting_distribution(0)
    tv = lambda p: 0.5 * np.abs(p - limit).sum()  # noqa: E731
    assert tv(walk.average_distribution(0, 500.0)) < tv(walk.average_distribution(0, 5.0))
    assert tv(walk.average_distribution(0, 500.0)) < 0.01
    mix = walk.mixing_time(0, 0.05, 200.0, steps=400)
    assert mix is not None and 0 < mix <= 200.0


def test_spatial_search_on_metatron_cube_is_exact():
    g = q.Graph.metatron_cube()
    search = q.SpatialSearch(g, [4], gamma=1 / 13)
    t = math.pi * math.sqrt(13) / 2
    assert search.success_probability(t) == pytest.approx(1.0, abs=1e-9)
    assert search.success_probability(0.0) == pytest.approx(1 / 13)
    best = search.optimize_time(2 * t)
    assert best["success_probability"] == pytest.approx(1.0, abs=1e-9)
    assert best["time"] == pytest.approx(t, abs=1e-4)
    assert best["gamma"] == pytest.approx(1 / 13)
    # Default hopping rate is the Childs–Goldstone critical value (N−1)/N² for K_N.
    assert q.SpatialSearch.critical_gamma(g) == pytest.approx(12 / 169)
    default = q.SpatialSearch(g, 4)
    assert default.gamma == pytest.approx(12 / 169) and default.marked == [4]


def test_dephasing_walk_limits():
    g = q.Graph.cycle(6)
    coherent = q.ContinuousWalk(g).probabilities(0, 1.7)
    open_walk = q.DephasingWalk(g, "laplacian", 1.0, 0.0)
    assert np.allclose(open_walk.populations(0, 1.7), coherent, atol=1e-8)
    # Strong dephasing relaxes to the uniform distribution.
    p = q.DephasingWalk(g, "laplacian", 1.0, 0.5).populations(0, 300.0)
    assert np.allclose(p, 1 / 6, atol=1e-4)
    rho0 = np.zeros((6, 6), dtype=complex)
    rho0[0, 0] = 1
    rho = q.DephasingWalk(g, rate=0.3).evolve(rho0, 2.0)
    assert rho.shape == (6, 6) and np.trace(rho).real == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(rho, rho.conj().T, atol=1e-10)


def test_classical_baselines():
    g = q.Graph.metatron_cube()
    assert np.allclose(q.classical_hitting_times(g, 0)[1:], 12.0)
    c = q.Graph.cycle(10)
    h = q.classical_hitting_times(c, 0)
    assert np.allclose(h, [min(u, 10 - u) * (10 - min(u, 10 - u)) for u in range(10)])
    p = q.classical_walk_probabilities(q.Graph.petersen(), 0, 100.0)
    assert p.dtype == np.float64 and np.allclose(p, 0.1)
    disconnected = q.Graph.from_edges(3, [(0, 1)])
    assert math.isinf(q.classical_hitting_times(disconnected, 0)[2])


def test_large_graph_uses_krylov_propagation():
    g = q.Graph.cycle(2100)  # above the dense limit
    walk = q.ContinuousWalk(g)
    assert walk.spectrum is None
    p = walk.probabilities(0, 2.0)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    small = q.ContinuousWalk(q.Graph.cycle(200)).probabilities(0, 2.0)
    assert p[:20] == pytest.approx(small[:20], abs=1e-10)
    # Limits come from the start node's cyclic subspace: on C_n a walker
    # sees n/2 + 1 distinct eigenvalues, and the limit is mirror-symmetric.
    assert walk.cyclic_dimension(0) == 1051
    limit = walk.limiting_distribution(0)
    assert limit.sum() == pytest.approx(1.0, abs=1e-10)
    assert np.allclose(limit[1:], limit[1:][::-1], atol=1e-12)


@pytest.mark.parametrize(
    "call",
    [
        lambda: q.ContinuousWalk(q.Graph.cycle(4), gamma=0.0),
        lambda: q.ContinuousWalk(q.Graph.cycle(4), "hamiltonian"),
        lambda: q.ContinuousWalk(q.Graph.cycle(4)).probabilities(4, 1.0),
        lambda: q.ContinuousWalk(q.Graph.cycle(4)).probabilities(0, math.inf),
        lambda: q.ContinuousWalk(q.Graph.cycle(4)).evolve([1, 0], 1.0),
        lambda: q.SpatialSearch(q.Graph.cycle(4), [9]),
        lambda: q.SpatialSearch(q.Graph.cycle(4), []),
        lambda: q.SpatialSearch(q.Graph(3), 0),
        lambda: q.SpatialSearch(q.Graph.cycle(4), 0).optimize_time(-1.0),
        lambda: q.DephasingWalk(q.Graph.cycle(4), rate=-1.0),
        lambda: q.DephasingWalk(q.Graph.cycle(4)).populations(0, -1.0),
        lambda: q.DephasingWalk(q.Graph.cycle(4)).evolve(np.eye(3), 1.0),
        lambda: q.classical_walk_probabilities(q.Graph.cycle(4), 7, 1.0),
        lambda: q.classical_hitting_times(q.Graph.cycle(4), 7),
        lambda: q.ContinuousWalk(q.Graph.cycle(4)).mixing_time(0, math.nan, 1.0),
        lambda: q.ContinuousWalk(q.Graph.cycle(4)).mixing_time(0, -0.1, 1.0),
        lambda: q.ContinuousWalk(q.Graph.cycle(4)).mixing_time(0, 0.1, math.inf),
        lambda: q.ContinuousWalk(q.Graph.cycle(4)).mixing_time(0, 0.1, 0.0),
    ],
)
def test_invalid_walk_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()


def test_joint_search_optimisation_finds_exact_rate_on_metatron_cube():
    g = q.Graph.from_spec("metatron")
    r = q.SpatialSearch.optimize(g, [3])
    assert r["gamma"] == pytest.approx(1 / 13, abs=1e-4)
    assert r["success_probability"] == pytest.approx(1.0, abs=1e-7)
    assert r["time"] == pytest.approx(np.pi * np.sqrt(13) / 2, abs=1e-3)
    # The asymptotic critical rate alone falls short on K13.
    plain = q.SpatialSearch(g, [3]).optimize_time(np.pi * np.sqrt(13))
    assert plain["success_probability"] < 0.99


def test_graph_specifications():
    assert q.Graph.from_spec("fruit-of-life").num_edges == 18
    assert q.Graph.from_spec("platonic:dodecahedron").automorphism_group_order() == 120
    assert q.Graph.from_spec("torus:3x4").num_edges == 24
    for bad in ["cycle:2", "nonsense", "complete:0"]:
        with pytest.raises(ValueError):
            q.Graph.from_spec(bad)


def test_quotient_walk_matches_full_walk():
    g = q.Graph.petersen()
    quotient = q.Quotient.with_singletons(g, 0)
    assert list(quotient.sizes) == [1, 3, 6]
    cell = quotient.cell_of[0]
    reduced = q.QuotientWalk(quotient, "adjacency", 0.8)
    full = q.ContinuousWalk(g, "adjacency", 0.8)
    for t in (0.3, 2.0):
        p = quotient.node_probabilities(reduced.cell_probabilities(cell, t))
        assert np.allclose(p, full.probabilities(0, t), atol=1e-12)
    limit = quotient.node_probabilities(reduced.limiting_distribution(cell))
    assert np.allclose(limit, full.limiting_distribution(0), atol=1e-12)
    assert np.allclose(np.sort(np.linalg.eigvalsh(quotient.adjacency())), [-2, 1, 3])


def test_hypercube_of_dimension_thirty():
    quotient = q.Quotient.hypercube(30)
    assert quotient.num_cells == 31 and quotient.num_nodes == 2.0**30
    walk = q.QuotientWalk(quotient)
    t = 0.4
    p = walk.cell_probabilities(0, t)
    k = np.arange(31)
    binom = np.array([math.comb(30, int(i)) for i in k], dtype=float)
    expected = binom * np.sin(t) ** (2 * k) * np.cos(t) ** (2 * (30 - k))
    assert np.allclose(p, expected, atol=1e-12)
    assert walk.cell_probabilities(0, math.pi / 2)[30] == pytest.approx(1.0, abs=1e-10)
    st = q.StateTransfer.from_quotient(quotient, 0, 30)
    assert st.perfect_transfer_time == pytest.approx(math.pi / 2, abs=1e-9)
    assert abs(st.amplitude(t)) == pytest.approx(math.sin(t) ** 30, abs=1e-14)


def test_quotient_search_and_critical_gamma():
    g = q.Graph.hypercube(8)
    quotient = q.Quotient.hypercube(8)
    gc = q.SpatialSearch.critical_gamma(g)
    assert q.QuotientSearch.critical_gamma(quotient, 0) == pytest.approx(gc, abs=1e-12)
    full = q.SpatialSearch(g, 0, "adjacency", gc)
    reduced = q.QuotientSearch(quotient, 0, "adjacency")
    assert reduced.gamma == pytest.approx(gc)
    for t in (5.0, 27.5):
        assert reduced.success_probability(t) == pytest.approx(full.success_probability(t), abs=1e-12)
    big = q.QuotientSearch.optimize(q.Quotient.complete(10**6), 0)
    assert big["success_probability"] == pytest.approx(1.0, abs=1e-6)


def test_state_transfer_on_paths():
    p3 = q.StateTransfer(q.Graph.path(3), 0, 2, "adjacency")
    assert p3.strongly_cospectral and p3.signs == [1, -1, 1]
    assert p3.perfect_transfer_time == pytest.approx(math.pi / math.sqrt(2))
    p4 = q.StateTransfer(q.Graph.path(4), 0, 3, "adjacency")
    assert p4.perfect_transfer_time is None
    best = p4.best_transfer(50.0, 20000)
    assert best["fidelity"] == pytest.approx(0.992356742, abs=1e-6)
    with pytest.raises(ValueError):
        q.StateTransfer(q.Graph.path(3), 1, 1)


def test_coined_walks():
    walk = q.CoinedWalk(q.Graph.hypercube(10), marked=[0])
    result = walk.search(100)
    assert result["steps"] == 38
    assert result["success_probability"] == pytest.approx(0.4350064336, abs=1e-9)
    assert result["history"].shape == (101,)
    k = q.CoinedWalk(q.Graph.complete(7), "fourier")
    psi = k.evolve(k.localized_state(0), 5)
    assert np.linalg.norm(psi) == pytest.approx(1.0)
    assert np.allclose(k.node_probabilities(psi), k.probabilities(0, 5))
    x = np.array([[0, 1], [1, 0]])
    assert q.CoinedWalk(q.Graph.cycle(6), x).num_arcs == 12
    with pytest.raises(ValueError):
        q.CoinedWalk(q.Graph.petersen(), "hadamard")
    with pytest.raises(ValueError):
        q.CoinedWalk(q.Graph.petersen(), "nope")


def test_hadamard_walk_and_szegedy_hitting_time():
    steps = 400
    p = q.LineWalk().distribution(steps)
    x = np.arange(-steps, steps + 1)
    sigma = math.sqrt((p * x**2).sum() - (p * x).sum() ** 2)
    assert sigma / steps == pytest.approx(math.sqrt(1 - 1 / math.sqrt(2)), abs=1e-3)
    assert [q.SzegedyWalk(q.Graph.complete(n), 0).hitting_time(10 * n) for n in (16, 64)] == [3, 6]
    assert q.SzegedyWalk(q.Graph.cycle(64), 0).hitting_time(1000) == 16


def test_walk_circuits_reproduce_the_walk():
    g = q.Graph.cycle(8)
    exact = q.ContinuousWalk(g).probabilities(0, 0.9)
    for circuit in (
        q.circulant_walk_circuit(g, 0.9),
        q.trotter_walk_circuit(g, 0.9, 200, order=2),
    ):
        state = q.simulate(circuit)
        assert np.allclose(state.probabilities(), exact, atol=1e-5)
    cube = q.Graph.hypercube(4)
    state = q.simulate(q.cayley_walk_circuit(cube, 1.1, "adjacency"))
    assert np.allclose(state.probabilities(), q.ContinuousWalk(cube, "adjacency").probabilities(0, 1.1))
    d = q.diagonal_unitary_circuit([0.1, 0.2, -0.4, 1.0])
    assert q.Circuit.from_qasm(d.to_qasm()).n_qubits == 2
    with pytest.raises(ValueError):
        q.circulant_walk_circuit(q.Graph.path(8), 1.0)


def test_cyclic_subspace_beyond_dense_limit():
    walk = q.ContinuousWalk(q.Graph.hypercube(12))
    assert walk.spectrum is None
    assert walk.cyclic_dimension(0) == 13
    limit = walk.limiting_distribution(0)
    assert limit.sum() == pytest.approx(1.0, abs=1e-10)
