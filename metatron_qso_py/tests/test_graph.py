import math

import numpy as np
import pytest

import metatron_qso as q


@pytest.mark.parametrize(
    "graph, order",
    [
        (q.Graph.metatron_cube(), 6227020800),  # 13!
        (q.Graph.fruit_of_life(), 12),  # D6
        (q.Graph.petersen(), 120),  # S5
        (q.Graph.platonic("dodecahedron"), 120),  # A5 x Z2
        (q.Graph.platonic("cube"), 48),
        (q.Graph.hypercube(4), 384),  # 2^4 * 4!
        (q.Graph.cycle(9), 18),
    ],
)
def test_automorphism_orders(graph, order):
    assert graph.automorphism_group_order() == order
    assert isinstance(graph.automorphism_group_order(), int)
    assert graph.automorphism_log10_order() == pytest.approx(math.log10(order))


def test_huge_automorphism_group_order_overflows_to_none():
    k35 = q.Graph.complete(35)  # 35! > 2^128
    assert k35.automorphism_group_order() is None
    assert k35.automorphism_log10_order() == pytest.approx(math.log10(math.factorial(35)))


def test_orbits():
    assert q.Graph.metatron_cube().automorphism_orbits() == [list(range(13))]
    # Fruit of Life: centre, inner ring, outer ring.
    assert q.Graph.fruit_of_life().automorphism_orbits() == [
        [0],
        list(range(1, 7)),
        list(range(7, 13)),
    ]
    star = q.Graph.star(5)
    assert star.automorphism_orbits() == [[0], [1, 2, 3, 4]]
    for perm in star.automorphism_generators():
        assert sorted(perm) == list(range(5))


def test_metatron_cube_structure():
    g = q.Graph.metatron_cube()
    assert (g.num_nodes, g.num_edges) == (13, 78)
    assert g.is_connected() and g.diameter() == 1
    s = g.summary()
    assert s["regular"] and s["triangles"] == 286 and not s["bipartite"]
    assert s["algebraic_connectivity"] == pytest.approx(13.0)
    pos = g.positions
    assert pos.shape == (13, 3) and pos.dtype == np.float64
    assert np.allclose(np.linalg.norm(pos[1:7], axis=1), 1.0)
    assert np.allclose(np.linalg.norm(pos[7:], axis=1), 2.0)


def test_metatron_line_lengths():
    lengths = q.metatron_line_lengths()
    assert sum(c for _, c in lengths) == 78
    assert lengths[0] == (pytest.approx(1.0), 18)
    assert [l for l, _ in lengths] == sorted(l for l, _ in lengths)
    # Keeping only unit-length lines gives the Fruit of Life.
    assert q.Graph.metatron_distance_graph(1.0).num_edges == 18
    assert q.Graph.metatron_distance_graph(math.inf).num_edges == 78


def test_matrices_and_spectrum():
    g = q.Graph.cycle(6)
    a, lap = g.adjacency_matrix(), g.laplacian_matrix()
    assert a.shape == lap.shape == (6, 6)
    assert a.dtype == lap.dtype == np.float64
    assert np.allclose(lap, np.diag(a.sum(axis=1)) - a)
    expected = sorted(2 - 2 * math.cos(2 * math.pi * k / 6) for k in range(6))
    assert np.allclose(g.laplacian_spectrum(), expected)
    assert np.allclose(np.sort(g.adjacency_spectrum()), np.linalg.eigvalsh(a))


def test_construction_and_weights():
    g = q.Graph.from_edges(4, [(0, 1), (1, 2), (2, 3)], weights=[1.0, 2.0, 0.5])
    assert g.edges == [(0, 1, 1.0), (1, 2, 2.0), (2, 3, 0.5)]
    h = q.Graph.from_edges(4, g.edges)
    assert h.edges == g.edges
    k = q.Graph.from_edges(4, np.array([[0, 1], [1, 2], [2, 3]]), np.array([1.0, 2.0, 0.5]))
    assert k.edges == g.edges
    with pytest.raises(TypeError):
        q.Graph.from_edges(4, [(0, 1, 2, 3)])
    g.add_edge(3, 0, weight=-1.5)
    assert g.num_edges == 4 and g.neighbors(0) == [(1, 1.0), (3, -1.5)]
    assert g.degrees() == [2, 2, 2, 2]
    g.name = "square"
    assert g.name == "square" and "square" in repr(g)
    assert g.positions is None
    assert not q.Graph(3).is_connected() and q.Graph(3).diameter() is None


def test_generators():
    assert q.Graph.grid(3, 4).num_edges == 17
    assert q.Graph.grid(3, 4, periodic=True).num_edges == 24
    assert q.Graph.hypercube(3).summary()["regular"]
    for name, (v, e) in {
        "tetrahedron": (4, 6),
        "cube": (8, 12),
        "octahedron": (6, 12),
        "dodecahedron": (20, 30),
        "icosahedron": (12, 30),
    }.items():
        g = q.Graph.platonic(name)
        assert (g.num_nodes, g.num_edges) == (v, e)
    rr = q.Graph.random_regular(20, 3, seed=5)
    assert set(rr.degrees()) == {3}
    assert rr.edges == q.Graph.random_regular(20, 3, seed=5).edges
    assert q.Graph.erdos_renyi(30, 0.0).num_edges == 0
    assert q.Graph.erdos_renyi(10, 1.0).num_edges == 45
    assert q.Graph.complete_bipartite(2, 3).num_edges == 6


@pytest.mark.parametrize(
    "build",
    [
        lambda: q.Graph(0),
        lambda: q.Graph.cycle(2),
        lambda: q.Graph.hypercube(21),
        lambda: q.Graph.grid(2, 5, periodic=True),
        lambda: q.Graph.platonic("sphere"),
        lambda: q.Graph.random_regular(5, 3),
        lambda: q.Graph.erdos_renyi(5, 1.5),
        lambda: q.Graph(3).add_edge(1, 1),
        lambda: q.Graph(3).add_edge(0, 3),
        lambda: q.Graph(3).add_edge(0, 1, float("nan")),
        lambda: q.Graph.from_edges(3, [(0, 1), (0, 1)]),
        lambda: q.Graph.from_edges(3, [(0, 1)], weights=[1.0, 2.0]),
        lambda: q.Graph(3).neighbors(5),
        # Sizes beyond the generator limits are rejected before anything is
        # allocated (they used to abort the interpreter or hang).
        lambda: q.Graph(10**12),
        lambda: q.Graph.from_edges(10**12, []),
        lambda: q.Graph.cycle(10**12),
        lambda: q.Graph.path(2**21),
        lambda: q.Graph.complete(10**6),
        lambda: q.Graph.complete_bipartite(10**5, 10**5),
        lambda: q.Graph.grid(10**6, 10**6),
        lambda: q.Graph.random_regular(10**6, 40),
        lambda: q.Graph.erdos_renyi(10**5, 0.0),
        lambda: q.Graph.from_spec("K:204204"),
    ],
)
def test_invalid_graphs_raise_value_error(build):
    with pytest.raises(ValueError):
        build()
