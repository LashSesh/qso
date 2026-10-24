# Graphs

## Building graphs

`Graph` is an undirected graph with real, non-zero edge weights, no loops and
no multi-edges. Graphs can be built edge by edge, from an edge list, by name
or from a short specification:

```python
import metatron_qso as q

g = q.Graph.from_edges(4, [(0, 1), (1, 2), (2, 3, 0.5)])
g.add_edge(3, 0)
print(g.summary())

for spec in ["metatron", "metatron:1", "fruit-of-life", "petersen", "heawood",
             "hypercube:4", "torus:4x5", "platonic:dodecahedron", "johnson:7,3",
             "kneser:7,2", "hamming:3,4", "paley:13", "rook:3x4", "shrikhande",
             "circulant:12,1,5", "random-regular:12,3,7", "erdos-renyi:20,0.2,1"]:
    h = q.Graph.from_spec(spec)
    print(f"{spec:24} {h.num_nodes:3} nodes {h.num_edges:3} edges")
```

Each family also has a constructor: `Graph.johnson(n, k)`, `kneser(n, k)`,
`hamming(d, q)`, `paley(q)`, `rook(m, n)`, `shrikhande()`,
`circulant(n, jumps)` and `cayley_z2(d, generators)` (node `x` joined to
`x XOR g`), besides the older `complete`, `cycle`, `path`, `star`, `grid`,
`hypercube`, `platonic`, `petersen`, `heawood` and the random graphs.
`g.cartesian_product(h)` builds products such as prisms and rook's graphs.

Specifications and the Python constructors reject graphs with more than
2²⁰ nodes or 2²² edges before building anything. Hypercubes up to dimension
20 are the exception.

## Structure and spectra

| Method | Result |
|---|---|
| `summary()` | degrees, connectivity, bipartiteness, diameter, triangles, algebraic connectivity |
| `girth()`, `triangle_count()`, `diameter()` | |
| `adjacency_matrix()`, `laplacian_matrix()` and their spectra | NumPy arrays |
| `automorphism_group_order()`, `automorphism_orbits()`, `automorphism_generators()` | exact symmetry group |

Automorphism groups are computed exactly, by colour refinement with
individualisation and orbit pruning. The order is an exact integer even for
13! = 6 227 020 800:

```python
import metatron_qso as q

assert q.Graph.metatron_cube().automorphism_group_order() == 6227020800
assert q.Graph.fruit_of_life().automorphism_group_order() == 12      # D6
assert q.Graph.platonic("dodecahedron").automorphism_group_order() == 120
assert q.Graph.heawood().girth() == 6
```

## Isomorphism and canonical forms

`canonical_labelling()` renumbers the nodes so that isomorphic graphs get
identical edge lists (`canonical_form()`); edge weights count.
`is_isomorphic` and `isomorphism` compare two graphs. The algorithm is
individualisation–refinement with automorphism pruning, as in nauty: fast
for the structured graphs of this library (K₆₀ in under 0.1 s, the
1024-node hypercube in 0.1 s), exponential only on specially constructed
hard instances.

```python
import metatron_qso as q

s, r = q.Graph.shrikhande(), q.Graph.rook(4, 4)
print(s.strongly_regular_parameters(), r.strongly_regular_parameters())  # both (16, 6, 2, 2)
assert not s.is_isomorphic(r)                     # same spectrum, different graphs
print(q.Graph.kneser(5, 2).isomorphism(q.Graph.petersen()))
g = q.Graph.platonic("icosahedron")
h = g.relabeled([11, 3, 7, 0, 9, 1, 4, 10, 2, 8, 6, 5])
assert g.canonical_form().edges == h.canonical_form().edges
```

## Reading and writing graphs

Graphs read and write **edge lists** (one `u v [weight]` per line, `#`
comments, `# nodes N` for isolated nodes) and **GraphML** (nodes, edges and
a `weight` key; node ids may be any strings). `Graph.from_networkx` takes a
NetworkX graph, or anything with `nodes()` and `edges(data=True)`, without
importing NetworkX.

```python
import metatron_qso as q

g = q.Graph.petersen()
text = g.to_edge_list()
assert q.Graph.from_edge_list(text).is_isomorphic(g)
assert q.Graph.from_graphml(g.to_graphml()).edges == g.edges

graphml = """<graphml><key id="w" for="edge" attr.name="weight"/>
<graph edgedefault="undirected"><node id="a"/><node id="b"/><node id="c"/>
<edge source="a" target="b"><data key="w">2.5</data></edge>
<edge source="b" target="c"/></graph></graphml>"""
print(q.Graph.from_graphml(graphml).edges)        # [(0, 1, 2.5), (1, 2, 1.0)]
```

Both readers reject malformed input with a `ValueError` and enforce the size
limits above.

## Metatron's Cube

The Fruit of Life consists of 13 equal circles: one in the centre, six
touching it, and six more along the same rays. Joining every pair of
centres gives the 78 lines of Metatron's Cube.

- **As a graph it is K₁₃.** The full symmetric group S₁₃ acts on it, and the
  Laplacian spectrum is {0, 13¹²}.
- **The geometry is in the embedding.** The lines have seven distinct
  lengths: 1, √3, 2, √7, 3, 2√3 and 4.
- **Short lines recover the structure.** `metatron:<length>` keeps only the
  lines up to a length. The contact graph (`metatron:1`, tangent circles)
  has exactly the hexagonal symmetry D₆.

```python
import metatron_qso as q

for length, count in q.metatron_line_lengths():
    print(f"{length:.4f}  {count}")
```

QSO makes no other claim about Metatron's Cube. What can be said is graph
theory or physics, and every such statement is tested.
