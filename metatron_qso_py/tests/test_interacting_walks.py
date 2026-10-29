"""Walks of several interacting particles, and the Chang graphs."""

import itertools
import math

import numpy as np
import pytest

import metatron_qso as q


def test_one_particle_is_the_continuous_walk():
    g = q.Graph.petersen()
    for statistics in ["bosons", "fermions", "distinguishable"]:
        w = q.InteractingWalk(g, 1, statistics, "laplacian", gamma=0.7)
        exact = q.ContinuousWalk(g, "laplacian", 0.7).propagator(1.3)
        assert np.abs(w.propagator(1.3) - exact).max() < 1e-12


def test_free_bosons_follow_permanents():
    # Two bosons from (0, 1) to (a, b): U_a0 U_b1 + U_a1 U_b0, divided by
    # √2 when a = b.
    g = q.Graph.cycle(5)
    t = 0.8
    u1 = q.ContinuousWalk(g, "adjacency", 1.0).propagator(t)
    w = q.InteractingWalk(g, 2)
    assert (w.dimension, w.statistics, w.generator) == (15, "bosons", "adjacency")
    psi = w.evolve(w.basis_state([0, 1]), t)
    for a, b in itertools.combinations_with_replacement(range(5), 2):
        amp = u1[a, 0] * u1[b, 1] + u1[a, 1] * u1[b, 0]
        if a == b:
            amp /= math.sqrt(2)
        assert abs(psi[w.index_of([b, a])] - amp) < 1e-12


def test_hong_ou_mandel_and_correlations():
    g = q.Graph.path(2)
    t = math.pi / 4
    bosons = q.InteractingWalk(g, 2, "bosons")
    p = bosons.probabilities([0, 1], t)
    assert p[bosons.index_of([0, 1])] < 1e-15
    psi = bosons.evolve(bosons.basis_state([0, 1]), t)
    assert np.allclose(bosons.densities(psi), [1, 1])
    corr = bosons.pair_correlations(psi)
    assert abs(corr[0, 1]) < 1e-15 and abs(corr[0, 0] - 2) < 1e-12
    dist = q.InteractingWalk(g, 2, "distinguishable")
    p = dist.probabilities([0, 1], t)
    assert abs(p[dist.index_of([0, 1])] + p[dist.index_of([1, 0])] - 0.5) < 1e-12


def test_hubbard_dimer_spectrum():
    u = 3.0
    w = q.InteractingWalk(q.Graph.path(2), 2, on_site=u)
    r = math.sqrt(u * u + 16)
    assert np.allclose(w.spectrum, sorted([(u - r) / 2, u, (u + r) / 2]), atol=1e-12)
    assert w.on_site == u and w.neighbour == 0.0
    f = q.InteractingWalk(q.Graph.path(2), 2, "fermions", neighbour=0.7)
    assert np.allclose(f.spectrum, [0.7])


def test_certificates_tell_chang_graphs_from_t8():
    t8 = q.Graph.johnson(8, 2)
    chang = q.Graph.chang(1)
    assert chang.strongly_regular_parameters() == (28, 12, 6, 4)
    assert not chang.is_isomorphic(t8)

    def cert(g, u):
        return q.InteractingWalk(g, 2, on_site=u).certificate([1.0])

    assert q.interacting_certificate_distance(cert(t8, 0.0), cert(chang, 0.0)) < 1e-12
    assert q.interacting_certificate_distance(cert(t8, 1.0), cert(chang, 1.0)) > 1e-6
    assert q.interacting_certificate_distance([1.0], [1.0, 2.0]) == math.inf


def test_seidel_switching():
    g = q.Graph.johnson(8, 2)
    s = g.seidel_switching([0, 5, 9])
    rest = [u for u in range(28) if u not in (0, 5, 9)]
    assert s.edges == g.seidel_switching(rest).edges
    assert sorted(g.seidel_switching([]).edges) == sorted(g.edges)


def test_errors():
    g = q.Graph.cycle(4)
    with pytest.raises(ValueError, match="statistics"):
        q.InteractingWalk(g, 2, "anyons")
    with pytest.raises(ValueError):
        q.InteractingWalk(g, 5, "fermions")
    w = q.InteractingWalk(g, 2, "fermions")
    with pytest.raises(ValueError):
        w.index_of([1, 1])
    with pytest.raises(ValueError):
        w.evolve([1.0], 0.5)
    with pytest.raises(ValueError):
        q.Graph.chang(4)
    assert "2 fermions" in repr(w)
    assert w.configuration(0) == [0, 1]
