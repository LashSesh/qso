"""Driven walks and decoherent coined walks."""

import math

import numpy as np
import pytest

import metatron_qso as q


def test_static_walk_matches_continuous_walk():
    g = q.Graph.hypercube(3)
    walk = q.DrivenWalk(8, [("laplacian", g)])
    psi0 = np.zeros(8, complex)
    psi0[0] = 1
    r = walk.evolve(psi0, 0.0, 1.5, lambda t: [0.5])
    exact = q.ContinuousWalk(g, gamma=0.5).evolve(psi0, 1.5)
    assert np.abs(r["state"] - exact).max() < 1e-8
    assert r["steps"] > 0 and r["rejected"] >= 0 and r["error_estimate"] < 1e-8


def test_local_adiabatic_search():
    n, marked, eps = 64, 5, 0.1
    walk = q.DrivenWalk(n, [("uniform", None), ("projector", [marked])])
    rn = math.sqrt(n - 1)
    total = n / (eps * rn) * math.atan(rn)

    def s(t: float) -> float:
        return 0.5 * (1 + math.tan(2 * eps * rn * t / n - math.atan(rn)) / rn)

    psi0 = np.full(n, 1 / math.sqrt(n), complex)
    r = walk.evolve(psi0, 0.0, total, lambda t: [-(1 - s(t)), -s(t)])
    assert abs(r["state"][marked]) ** 2 > 1 - eps**2
    single = q.DrivenWalk(n, [("uniform", None), ("projector", marked)])
    capped = single.evolve(psi0, 0.0, total, lambda t: [-(1 - s(t)), -s(t)], max_step=1.0)
    assert np.abs(capped["state"] - r["state"]).max() < 1e-6
    assert capped["forced"] == 0


def test_driven_walk_errors():
    g = q.Graph.cycle(4)
    with pytest.raises(ValueError):
        q.DrivenWalk(4, [("nonsense", None)])
    with pytest.raises(ValueError):
        q.DrivenWalk(4, [("laplacian", [1, 2])])
    walk = q.DrivenWalk(4, [("laplacian", g), ("potential", [1.0, 0.0, 0.0, 0.0])])
    psi0 = np.array([1, 0, 0, 0], complex)
    with pytest.raises(ValueError):
        walk.evolve(psi0, 0.0, 1.0, lambda t: [1.0])

    def boom(t: float) -> list:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        walk.evolve(psi0, 0.0, 1.0, boom)


def test_fully_coin_dephased_walk_is_classical():
    g = q.Graph.petersen()
    walk = q.CoinedWalk(g, "fourier")
    open_walk = q.DecoherentCoinedWalk(walk, coin_dephasing=1.0)
    assert open_walk.num_arcs == 30 and open_walk.coin_dephasing == 1.0
    rho = open_walk.evolve(open_walk.localized_mixed(0), 3)
    p = open_walk.node_probabilities(rho)
    a = np.zeros((10, 10))
    for u, v, _ in g.edges:
        a[u, v] = a[v, u] = 1
    p_classical = np.linalg.matrix_power(a / 3, 3)[0]
    assert np.abs(p - p_classical).max() < 1e-12
    coherent = q.DecoherentCoinedWalk(walk)
    assert np.allclose(coherent.probabilities(0, 4), walk.probabilities(0, 4))
    assert np.isclose(np.trace(open_walk.pure(walk.localized_state(1))), 1)
    with pytest.raises(ValueError):
        q.DecoherentCoinedWalk(walk, coin_dephasing=2.0)
