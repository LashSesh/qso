import math

import numpy as np
import pytest

import metatron_qso as q


def wang_edge(du, dv, lam, gamma, beta):
    """One-layer QAOA expected cut of an edge (Wang et al. 2018, Thm. 1)."""
    c, c2 = math.cos(gamma), math.cos(2 * gamma)
    return (
        0.5
        + 0.25 * math.sin(4 * beta) * math.sin(gamma) * (c ** (du - 1) + c ** (dv - 1))
        - 0.25 * math.sin(2 * beta) ** 2 * c ** (du + dv - 2 - 2 * lam) * (1 - c2**lam)
    )


def test_feature_matches_closed_form():
    g = q.Graph.from_spec("random-regular:10,3,5")
    nbrs = {u: set() for u in range(g.num_nodes)}
    for u, v, _ in g.edges:
        nbrs[u].add(v)
        nbrs[v].add(u)
    gamma, beta = 0.7, -0.4
    expected = np.mean(
        [wang_edge(3, 3, len(nbrs[u] & nbrs[v]), gamma, beta) for u, v, _ in g.edges]
    )
    model = q.GraphClassifier(1)
    assert model.feature(g, angles=[gamma, beta]) == pytest.approx(expected, abs=1e-12)


def regular_graphs(n, count, seed, predicate):
    out, s = [], seed
    while len(out) < count:
        g = q.Graph.from_spec(f"random-regular:{n},3,{s}")
        s += 1
        if predicate(g):
            out.append(g)
    return out


def test_learns_triangle_detection():
    pos = regular_graphs(10, 8, 0, lambda g: g.triangle_count() > 0)
    neg = regular_graphs(10, 8, 1000, lambda g: g.triangle_count() == 0)
    graphs, labels = pos + neg, [1] * 8 + [0] * 8
    model = q.GraphClassifier(1)
    report = model.fit(graphs, labels, restarts=3, seed=1)
    assert report["accuracy"] == 1.0
    assert model.num_parameters == 4 and len(model.parameters) == 4
    assert model.predict(pos[0]) is True
    assert 0.0 <= model.predict_proba(neg[0]) < 0.5
    assert model.accuracy(graphs, labels) == 1.0


def test_girth_heawood_and_triangles():
    assert q.Graph.heawood().girth() == 6
    assert q.Graph.petersen().girth() == 5
    assert q.Graph.path(5).girth() is None
    assert q.Graph.complete(5).triangle_count() == 10


@pytest.mark.parametrize(
    "call",
    [
        lambda: q.GraphClassifier(0),
        lambda: q.GraphClassifier(1).predict(q.Graph.cycle(4)),
        lambda: q.GraphClassifier(1).feature(q.Graph.cycle(4), angles=[0.1]),
        lambda: q.GraphClassifier(1).fit([q.Graph.cycle(4)], [1, 0]),
        lambda: q.GraphClassifier(1).fit([q.Graph.cycle(4)], [0.5]),
    ],
)
def test_invalid_input(call):
    with pytest.raises(ValueError):
        call()
