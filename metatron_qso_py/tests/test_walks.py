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
    with pytest.raises(ValueError):
        walk.limiting_distribution(0)


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
