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
For N = 64 and 256 both values agree to 10⁻⁶ with an independent
integration of the two-dimensional subspace.

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
