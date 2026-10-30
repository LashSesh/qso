"""Open and driven walks reduced by symmetry."""

import math

import numpy as np
import pytest

import metatron_qso as q


def bloch_z(gamma: float, kappa: float, t: float) -> float:
    w = math.sqrt(4 * gamma**2 - kappa**2 / 4)
    return math.exp(-kappa * t / 2) * (math.cos(w * t) + kappa / (2 * w) * math.sin(w * t))


def test_orbital_counts_and_stabilisers():
    for d in range(1, 6):
        g = q.Graph.hypercube(d)
        o = q.Orbitals(1 << d, g.stabilizer_generators([0]))
        assert len(o) == math.comb(d + 3, 3)
        assert len(q.Orbitals(1 << d, q.coordinate_permutations(d))) == len(o)
    assert q.Graph.petersen().stabilizer_order([0]) == 12
    assert q.Graph.hypercube(4).stabilizer_order([0]) == 24
    assert len(q.Orbitals(4)) == 16
    o = q.Orbitals(10, q.Graph.petersen().stabilizer_generators([0]))
    x = np.arange(len(o)) + 0.5j
    assert np.array_equal(o.coefficients(o.operator(x)), x)
    a, b = o.representative(3)
    assert o.of(a, b) == 3 and o.size(3) >= 1
    with pytest.raises(ValueError):
        o.of(0, 10)
    with pytest.raises(ValueError):
        o.size(len(o))


def test_site_dephasing_matches_the_density_matrix():
    g = q.Graph.petersen()
    full = q.DephasingWalk(g, gamma=0.8, rate=0.6)
    walk = q.SymmetricOpenWalk(
        g, gamma=0.8, dephasing=[(0.6, list(range(10)))],
        symmetries=g.stabilizer_generators([0]),
    )
    assert walk.num_orbitals < 25
    p = walk.populations(walk.evolve(walk.localized(0), 2.3))
    assert np.abs(p - full.populations(0, 2.3)).max() < 1e-9
    rho = walk.orbitals.operator(walk.evolve(walk.localized(0), 2.3))
    assert abs(np.trace(rho) - 1) < 1e-12


def test_coordinate_dephasing_is_binomial():
    d, gamma, kappa, t = 8, 0.7, 0.9, 1.3
    n = 1 << d
    walk = q.SymmetricOpenWalk(
        q.Graph.hypercube(d),
        gamma=gamma,
        dephasing=[(kappa, [(u >> k) & 1 for u in range(n)]) for k in range(d)],
        symmetries=q.coordinate_permutations(d),
    )
    assert walk.num_orbitals == math.comb(d + 3, 3)
    p = walk.populations(walk.evolve(walk.localized(0), t))
    flip = (1 - bloch_z(gamma, kappa, t)) / 2
    for u in range(n):
        k = bin(u).count("1")
        assert abs(p[u] - flip**k * (1 - flip) ** (d - k)) < 1e-12


def test_product_walk_of_a_thousand_edges():
    gamma, kappa, t = 0.7, 0.9, 1.3
    edge = q.DephasingWalk(q.Graph.complete(2), gamma=gamma, rate=kappa)
    product = q.ProductOpenWalk([(edge, 1000)])
    marginals = product.marginals([0], t)
    flip = (1 - bloch_z(gamma, kappa, t)) / 2
    assert abs(marginals[0][1] - flip) < 1e-10
    dist = product.count_distribution(marginals, [1])
    assert len(dist) == 1001
    assert abs(dist.sum() - 1) < 1e-12
    assert abs((np.arange(1001) * dist).sum() - 1000 * flip) < 1e-7
    with pytest.raises(ValueError):
        q.ProductOpenWalk([(edge, 0)])
    with pytest.raises(ValueError):
        product.count_distribution(marginals, [5])


def test_coined_walk_matches_the_density_matrix():
    g = q.Graph.hypercube(4)
    full = q.DecoherentCoinedWalk(q.CoinedWalk(g, "grover"), position_dephasing=0.2)
    walk = q.SymmetricCoinedWalk(full, g.stabilizer_generators([0]))
    assert walk.num_orbitals < 64 * 64 // 4
    p = walk.node_probabilities(walk.evolve(walk.localized(0), 9))
    assert np.abs(p - full.probabilities(0, 9)).max() < 1e-12
    coin = q.DecoherentCoinedWalk(q.CoinedWalk(g, "grover"), coin_dephasing=0.3)
    with pytest.raises(ValueError):
        q.SymmetricCoinedWalk(coin, g.stabilizer_generators([0]))


def test_driven_quotient_walk_matches_the_full_walk():
    g = q.Graph.hypercube(6)
    n = 64
    quotient = q.Quotient.with_singletons(g, [0])
    small = q.QuotientDrivenWalk(quotient, [("laplacian", None), ("cells", [0]), ("uniform", None)])
    large = q.DrivenWalk(n, [("laplacian", g), ("projector", [0]), ("uniform", None)])

    def c(t: float) -> list[float]:
        return [0.3 + 0.2 * t, -2 * math.sin(t), -1 + 0.5 * t]

    a = small.evolve(small.uniform_state(), 0.0, 2.0, c, fixed_step=0.02)
    b = large.evolve(np.full(n, 1 / math.sqrt(n), complex), 0.0, 2.0, c, fixed_step=0.02)
    assert np.abs(quotient.lift(a["state"]) - b["state"]).max() < 1e-10
    assert abs(small.cell_probabilities(a["state"]).sum() - 1) < 1e-10
    assert small.quotient.num_cells == quotient.num_cells
    with pytest.raises(ValueError):
        q.QuotientDrivenWalk(quotient, [("cells", [99])])
    with pytest.raises(ValueError):
        q.QuotientDrivenWalk(quotient, [("bogus", None)])
    with pytest.raises(ValueError):
        q.Quotient.hypercube(3).lift(np.ones(4))


def test_adiabatic_search_on_k_two_to_the_thirty():
    n = float(1 << 30)
    walk = q.QuotientDrivenWalk(q.Quotient.complete(1 << 30), [("uniform", None), ("cells", [0])])
    eps, rn = 0.1, math.sqrt(n - 1)
    total = n / (eps * rn) * math.atan(rn)

    def s(t: float) -> float:
        return 0.5 * (1 + math.tan(2 * eps * rn * t / n - math.atan(rn)) / rn)

    r = walk.evolve(walk.uniform_state(), 0.0, total, lambda t: [-(1 - s(t)), -s(t)])
    assert walk.cell_probabilities(r["state"])[0] > 0.98


def test_errors():
    g = q.Graph.petersen()
    sym = g.stabilizer_generators([0])
    walk = q.SymmetricOpenWalk(g, dephasing=[(0.1, list(range(10)))], symmetries=sym)
    with pytest.raises(ValueError):
        walk.localized(1)
    with pytest.raises(ValueError):
        walk.evolve(walk.uniform(), -1.0)
    with pytest.raises(ValueError):
        walk.populations(np.ones(2))
    with pytest.raises(ValueError):
        q.SymmetricOpenWalk(g, symmetries=[[1, 0] + list(range(2, 10))])
    with pytest.raises(ValueError):
        q.SymmetricOpenWalk(g, generator="bogus")
    with pytest.raises(ValueError):
        q.coordinate_permutations(14)
