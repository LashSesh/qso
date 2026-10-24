# Quantum walks

## Continuous-time walks

A continuous-time quantum walk evolves a state on the nodes of a graph with
`exp(−iHt)`, where `H = γL` (Laplacian, the default) or `H = −γA`
(adjacency). Up to 2048 nodes QSO diagonalises `H` exactly, which gives
probabilities at any time, limiting and time-averaged distributions, and
mixing times. Beyond that size, `evolve` and `probabilities` propagate with
a Krylov method.

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
