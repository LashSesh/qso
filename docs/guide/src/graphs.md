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
             "hypercube:4", "torus:4x5", "platonic:dodecahedron",
             "random-regular:12,3,7", "erdos-renyi:20,0.2,1"]:
    h = q.Graph.from_spec(spec)
    print(f"{spec:24} {h.num_nodes:3} nodes {h.num_edges:3} edges")
```

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
