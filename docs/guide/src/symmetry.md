# Symmetry and state transfer

## Walks reduced by symmetry

A partition of the nodes into cells is *equitable* if every node of a cell
has the same total edge weight into each other cell. The normalised cell
states then span a subspace that `H` leaves invariant, and a walker that
starts in it — at a node that is a cell of its own, or in the uniform
superposition — never leaves it. Its dynamics are exact in as many
dimensions as there are cells, however large the graph:

| Family | Nodes | Cells from a node |
|---|---|---|
| hypercube Q_d | 2^d | d + 1 |
| Hamming H(d, q) | q^d | d + 1 |
| Johnson J(n, k) | C(n, k) | min(k, n − k) + 1 |
| complete K_n | n | 2 |
| complete multipartite | Σ parts | ≤ number of parts + 1 |

For these families `Quotient` writes the partition down without building the
graph, so `Quotient.hypercube(30)` is a 31-dimensional problem for 2³⁰ nodes.
For any other graph, `Quotient.with_singletons(graph, nodes)` finds the
coarsest equitable partition in which the given nodes are cells of their own
(by colour refinement), and `node_probabilities` turns cell probabilities
into node probabilities.

```python
import math
import numpy as np
import metatron_qso as q

walk = q.QuotientWalk(q.Quotient.hypercube(30))
p = walk.cell_probabilities(0, 0.5)                 # by Hamming weight
k = np.arange(31)
binom = np.array([math.comb(30, int(i)) for i in k], dtype=float)
assert np.allclose(p, binom * np.sin(0.5) ** (2 * k) * np.cos(0.5) ** (2 * (30 - k)))

g = q.Graph.petersen()
quotient = q.Quotient.with_singletons(g, 0)
print(quotient.sizes)                               # [1. 3. 6.]
reduced = q.QuotientWalk(quotient)
p = quotient.node_probabilities(reduced.cell_probabilities(quotient.cell_of[0], 1.2))
assert np.allclose(p, q.ContinuousWalk(g).probabilities(0, 1.2))
```

`QuotientSearch` runs the Childs–Goldstone search on a quotient; its
critical rate ⟨w|L⁺|w⟩ is computed from the reduced Laplacian and equals the
usual (1/N)·Σ 1/λ on vertex-transitive graphs. A search among a million
nodes of K_N or of the hypercube Q₂₀ takes milliseconds:

```python
import metatron_qso as q

best = q.QuotientSearch.optimize(q.Quotient.hypercube(20), 0, t_max=3000, resolution=3000)
print(best)       # success probability ≈ 0.93 at t ≈ 1670
```

## Perfect state transfer

A walk has *perfect state transfer* (PST) from u to v at time τ if
|⟨v|e^{−iHτ}|u⟩| = 1. `StateTransfer` decides it from the spectrum: u and v
must be strongly cospectral, and the eigenvalues in the support of u must be
spaced by integer multiples of one gap, with odd multiples exactly where the
eigenprojections of u and v differ in sign. The earliest time is π/gap; every
positive answer is confirmed by evaluating the amplitude. Without PST,
`best_transfer` finds the highest fidelity within a time window.

```python
import math
import metatron_qso as q

assert abs(q.StateTransfer(q.Graph.path(3), 0, 2, "adjacency").perfect_transfer_time
           - math.pi / math.sqrt(2)) < 1e-9
p4 = q.StateTransfer(q.Graph.path(4), 0, 3, "adjacency")
print(p4.strongly_cospectral, p4.perfect_transfer_time, p4.best_transfer(50.0))

q30 = q.StateTransfer.from_quotient(q.Quotient.hypercube(30), 0, 30)
assert abs(q30.perfect_transfer_time - math.pi / 2) < 1e-9   # antipodes of Q30
```

The analysis runs on the quotient in which u is a cell of its own: v must be
a cell of its own too, so symmetric graphs reduce to a few dimensions.

## In Rust

```rust
use metatron_qso::prelude::*;
use metatron_qso::walks::{Quotient, QuotientWalk, StateTransfer};

fn main() -> Result<()> {
    let q = Quotient::hypercube(30)?;
    let walk = QuotientWalk::new(&q, Generator::Laplacian, 1.0)?;
    let p = walk.cell_probabilities(0, std::f64::consts::FRAC_PI_2)?;
    assert!((p[30] - 1.0).abs() < 1e-10);
    let pst = StateTransfer::from_quotient(&q, Generator::Laplacian, 1.0, 0, 30)?;
    assert!(pst.perfect_transfer_time().is_some());
    Ok(())
}
```
