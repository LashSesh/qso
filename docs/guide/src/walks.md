# Quantum walks

## Continuous-time walks

A continuous-time quantum walk evolves a state on the nodes of a graph with
`exp(−iHt)`, where `H = γL` (Laplacian, the default) or `H = −γA`
(adjacency). Up to 2048 nodes QSO diagonalises `H` exactly, which gives
probabilities at any time, limiting and time-averaged distributions, and
mixing times. Beyond that size, `evolve` and `probabilities` propagate with
a Krylov method, and the limiting and averaged distributions come from the
start node's *cyclic subspace* — the smallest invariant subspace containing
it, whose dimension is the number of distinct eigenvalues the walker sees
(13 on the 4096-node hypercube Q₁₂). For walks on far larger symmetric
graphs, see [Symmetry and state transfer](symmetry.md).

```python
import metatron_qso as q

walk = q.ContinuousWalk(q.Graph.metatron_cube())
print(walk.probabilities(0, 1.3))
limit = walk.limiting_distribution(0)
assert abs(limit[0] - 145 / 169) < 1e-12
```

The time-averaged distribution converges to the limiting distribution. The
**mixing time** M_ε (Aharonov et al. 2001) is the time from which on the
average stays within total-variation distance ε of the limit:

```python
import metatron_qso as q

walk = q.ContinuousWalk(q.Graph.metatron_cube())
print(walk.mixing_time(0, 0.02, 20.0, steps=2000))
```

```python
import metatron_qso as q

big = q.ContinuousWalk(q.Graph.hypercube(12))     # 4096 nodes
assert big.spectrum is None and big.cyclic_dimension(0) == 13
print(big.limiting_distribution(0)[:3])
```

## Spatial search

Childs–Goldstone search adds an oracle term −|w⟩⟨w| for the marked nodes w.
At the critical rate γ the walk moves amplitude to the marked node.
`SpatialSearch.optimize` tunes γ and the measurement time together:

```python
import math
import metatron_qso as q

cube = q.Graph.metatron_cube()
search = q.SpatialSearch(cube, marked=[4], gamma=1 / 13)
assert abs(search.success_probability(math.pi * math.sqrt(13) / 2) - 1) < 1e-9

best = q.SpatialSearch.optimize(q.Graph.fruit_of_life(), [0])
print(best)            # gamma, time, success probability, expected time
print(q.classical_hitting_times(cube, 4)[0])        # 12 steps for a random walk
```

On complete graphs the search time grows as √N, the quadratic speed-up over
classical search. The technical note checks this for N up to 1024.

## Decoherence

`DephasingWalk` adds Lindblad dephasing at a given rate κ. With κ = 0 it
reproduces the coherent walk. For strong dephasing the populations follow
the classical master equation dp/dt = −(2γ²/κ)·L·p. This is the quantum
Zeno regime: the walk becomes classical and slows down as κ grows.

```python
import numpy as np
import metatron_qso as q

g = q.Graph.fruit_of_life()
for rate in [0.0, 1.0, 10.0]:
    p = q.DephasingWalk(g, rate=rate).populations(0, 3.0)
    print(rate, np.round(p[:3], 4))
print(np.round(q.classical_walk_probabilities(g, 0, 3.0)[:3], 4))
```

## Time-dependent Hamiltonians

`DrivenWalk` evolves a walker under H(t) = Σ_j c_j(t)·H_j. A term is a
pair: `("laplacian", graph)`, `("adjacency", graph)` (minus the
adjacency matrix), `("projector", nodes)`, `("potential", values)` or
`("uniform", None)`, the projector onto the uniform superposition (dense,
but applied in O(N)). The coefficients come from any function of time.

`evolve` uses the fourth-order commutator-free Magnus integrator with two
exponentials per step (Blanes and Moan 2006). The exponentials use Krylov
propagation, so sparse walks on large graphs stay cheap. The step size
is controlled by step doubling: a step is accepted when one step of h
and two of h/2 agree to the tolerance, and the two half steps are kept.
The coefficients are sampled only inside each step: split the
evolution at jumps of the drive, and give short pulses a `max_step`.
The Python callable runs with the interpreter lock held.
`fixed_step` switches the control off. In the tests, halving a fixed
step divides the error by 16.1–18.5, the factor 2⁴ of a fourth-order
method.

The local adiabatic search of Roland and Cerf (2002) is an example. On
N nodes, H(s) = (1 − s)(I − |u⟩⟨u|) + s(I − |m⟩⟨m|) moves the
uniform state |u⟩ to the marked node m. The schedule ds/dt = ε·g(s)²
slows down where the gap g(s) is small and takes
T = N/(ε√(N − 1))·arctan√(N − 1) ≈ (π/2ε)·√N, the quadratic speed-up.
The identity only adds a global phase and is left out:

```python
import math
import numpy as np
import metatron_qso as q

n, marked, eps = 256, 3, 0.1
walk = q.DrivenWalk(n, [("uniform", None), ("projector", [marked])])
rn = math.sqrt(n - 1)
total = n / (eps * rn) * math.atan(rn)

def s(t: float) -> float:   # the inverse of t(s)
    return 0.5 * (1 + math.tan(2 * eps * rn * t / n - math.atan(rn)) / rn)

psi0 = np.full(n, 1 / math.sqrt(n), complex)
local = walk.evolve(psi0, 0.0, total, lambda t: [-(1 - s(t)), -s(t)])
linear = walk.evolve(psi0, 0.0, total, lambda t: [-(1 - t / total), -t / total])
print(abs(local["state"][marked]) ** 2, abs(linear["state"][marked]) ** 2)
# 0.999868 0.524902
```

The validation suite checks N = 64, 256 and 1024. The local schedule
reaches the marked node with probability 0.99942, 0.99987 and 0.99996,
and a linear schedule over the same time with 0.765, 0.525 and 0.315.
All six values agree to 10⁻⁶ with an independent integration of the
two-dimensional subspace (SciPy).

## Several particles

`InteractingWalk` puts `N` particles on a graph: bosons, fermions or
distinguishable particles. They hop with the walk Hamiltonian and
interact with each other:

```text
H = Σ_ij h_ij a†_i a_j + (U/2) Σ_i n_i (n_i − 1) + V Σ_{edges ij} n_i n_j
```

The default hopping is `"adjacency"` (`h = −γA`), so bosons give the
Bose–Hubbard model on the graph. States are vectors over the
configurations, which are multisets of sites for bosons, sets for
fermions, and the site of each particle for distinguishable particles.
Up to 2048 configurations the walk is diagonalised once; above that it
propagates by Krylov.

Two bosons that start on the two ends of an edge leave together. A
quarter period later they are never found apart, which is the
Hong–Ou–Mandel effect. Distinguishable particles are found apart half of
the time:

```python
import math
import metatron_qso as q

edge = q.Graph.path(2)
bosons = q.InteractingWalk(edge, 2)
p = bosons.probabilities([0, 1], math.pi / 4)
print(p[bosons.index_of([0, 1])] < 1e-15, p[bosons.index_of([0, 0])])
# True 0.4999999999999996
labelled = q.InteractingWalk(edge, 2, "distinguishable")
p = labelled.probabilities([0, 1], math.pi / 4)
print(round(p[labelled.index_of([0, 1])] + p[labelled.index_of([1, 0])], 12))
# 0.5
```

A strong on-site interaction binds two bosons on the same site into a
pair, even though the interaction is repulsive: the pair cannot shed the
energy `U` (Winkler et al., Nature 441, 853 (2006)). On a cycle of 12
sites the two bosons are still found together with probability 0.92 at
`t = 5` for `U = 10`, against 0.19 without interaction:

```python
import numpy as np
import metatron_qso as q

ring = q.Graph.cycle(12)
for u in [0.0, 10.0]:
    walk = q.InteractingWalk(ring, 2, on_site=u)
    probs = np.abs(walk.evolve(walk.basis_state([0, 0]), 5.0)) ** 2
    print(u, round(sum(probs[walk.index_of([i, i])] for i in range(12)), 4))
# 0.0 0.1892
# 10.0 0.9244
```

`densities` and `pair_correlations` give `⟨n_i⟩` and `⟨n_i n_j⟩` of any
state.

### Telling graphs apart

One walker sees only the spectrum and the eigenvectors of a graph, so
strongly regular graphs with the same parameters look alike to it. Two
free particles cannot tell them apart either; two interacting bosons can
(Gamble, Friesen, Zhou, Joynt and Coppersmith, Phys. Rev. A 81, 052313
(2010)). `certificate(times)` lists the transition probabilities between
all configurations, sorted, at each time. Isomorphic graphs give the
same list to rounding, so a difference above about 10⁻⁹ proves that two
graphs are not isomorphic:

```python
import metatron_qso as q

def certificate(graph, u):
    return q.InteractingWalk(graph, 2, on_site=u).certificate([1.0])

t8, chang = q.Graph.johnson(8, 2), q.Graph.chang(1)   # both (28, 12, 6, 4)
for u in [0.0, 1.0]:
    d = q.interacting_certificate_distance(certificate(t8, u), certificate(chang, u))
    print(u, d > 1e-9)
# 0.0 False
# 1.0 True
```

The validation suite checks this for all six pairs among `T(8)` and
the three Chang graphs (`Graph.chang(1)` to `Graph.chang(3)`, obtained
from `T(8)` by `seidel_switching`), and for the Shrikhande and 4 × 4
rook's graphs. Without interaction the certificates of every pair agree
to 10⁻¹². With `U = 1` they differ by 1.7·10⁻⁴ to 3.4·10⁻⁴. The same
comparison runs from the command line with
`qso particles --graph shrikhande --compare rook:4x4 --on-site 1`.

The amplitudes are checked in the tests against independent references:

- without interaction, against permanents (bosons) and determinants
  (fermions) of the single-walker propagator, also for 2925
  configurations propagated by Krylov;
- against the Hamiltonian built on the full tensor-product space and
  restricted to the symmetric or antisymmetric subspace;
- against the closed-form spectrum of the Hubbard dimer.

## In Rust

```rust
use metatron_qso::prelude::*;

fn main() -> Result<()> {
    let cube = metatron::metatron_cube();
    let walk = ContinuousWalk::new(&cube, Generator::Laplacian, 1.0)?;
    let limit = walk.limiting_distribution(0)?;
    assert!((limit[0] - 145.0 / 169.0).abs() < 1e-12);

    let search = SpatialSearch::new(&cube, &[3], Generator::Laplacian, 1.0 / 13.0)?;
    let t = std::f64::consts::PI * 13f64.sqrt() / 2.0;
    assert!((search.success_probability(t) - 1.0).abs() < 1e-9);
    Ok(())
}
```
