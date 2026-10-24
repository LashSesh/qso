"""Classifying graphs with QAOA-type quantum layers.

One layer sees the neighbourhood of radius 1 of every edge, so it can
detect triangles; it cannot tell triangle-free cubic graphs with and
without 4-cycles apart, which two layers can.
"""

import metatron_qso as q


def cubic_graphs(n, count, seed, predicate):
    graphs, s = [], seed
    while len(graphs) < count:
        g = q.Graph.from_spec(f"random-regular:{n},3,{s}")
        s += 1
        if predicate(g):
            graphs.append(g)
    return graphs


def has_square(g):
    nbrs = {u: set() for u in range(g.num_nodes)}
    for u, v, _ in g.edges:
        nbrs[u].add(v)
        nbrs[v].add(u)
    return any(len(nbrs[u] & nbrs[v]) >= 2 for u in nbrs for v in nbrs if u < v)


def dataset(n, per_class, seed, label):
    pos = cubic_graphs(n, per_class, seed, lambda g: label(g) is True)
    neg = cubic_graphs(n, per_class, seed + 10_000, lambda g: label(g) is False)
    return pos + neg, [1] * per_class + [0] * per_class


tasks = {
    "triangle present (12 nodes)": (12, lambda g: g.triangle_count() > 0),
    "4-cycle present (14 nodes, triangle-free)": (
        14, lambda g: None if g.triangle_count() else has_square(g)),
}
for name, (n, label) in tasks.items():
    train, y_train = dataset(n, 12, 1, label)
    test, y_test = dataset(n, 12, 50_000, label)
    print(name)
    for layers in (1, 2):
        model = q.GraphClassifier(layers)
        report = model.fit(train, y_train, restarts=3, seed=0)
        print(f"  {layers} layer(s): train {report['accuracy']:.2f}, test {model.accuracy(test, y_test):.2f}")
