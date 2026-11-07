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

## Driven and decoherent walks reduced by symmetry

The same reduction carries over to time-dependent Hamiltonians and to
density matrices.

**Driven walks on quotients.** `QuotientDrivenWalk` takes a `Quotient` and
terms that respect its partition — `("laplacian", None)`,
`("adjacency", None)`, `("cells", cells)` (the projector onto those cells),
`("potential", values)` (one value per cell) and `("uniform", None)` — and
integrates `H(t) = Σ_j c_j(t)·H_j` on the cells with the Magnus integrator
of `DrivenWalk`. `Quotient.lift` turns cell amplitudes into node
amplitudes for a partition of an explicit graph. The local adiabatic
search of Roland and Cerf on K_N with N = 2³⁰ is a two-dimensional
problem; it runs in 0.02 s and ends at the marked node with probability
0.999999, after a time within 10⁻⁴ of the closed form `(π/2ε)√N`:

```python
import math
import metatron_qso as q

n = float(1 << 30)
walk = q.QuotientDrivenWalk(q.Quotient.complete(1 << 30), [("uniform", None), ("cells", [0])])
eps, rn = 0.1, math.sqrt(n - 1)
total = n / (eps * rn) * math.atan(rn)

def s(t: float) -> float:
    return 0.5 * (1 + math.tan(2 * eps * rn * t / n - math.atan(rn)) / rn)

r = walk.evolve(walk.uniform_state(), 0.0, total, lambda t: [-(1 - s(t)), -s(t)])
assert walk.cell_probabilities(r["state"])[0] > 0.999
```

**Open walks on orbitals.** If the Hamiltonian, the dephasing and the start
are invariant under a group of automorphisms, so is the density matrix at
every time, and it is constant on each *orbital*, an orbit of the group on
ordered pairs of nodes. `SymmetricOpenWalk` evolves one coefficient per
orbital. Dephasing is given as `(rate, block)` pairs:
`(κ, list(range(n)))` is the site dephasing of `DephasingWalk`, and
`(κ, [(u >> k) & 1 for u in range(n)])` dephases coordinate `k` of a
hypercube. The symmetries come from `graph.stabilizer_generators(nodes)`
or, for hypercubes, `coordinate_permutations(d)`: two permutations that
generate all permutations of the coordinates. Small generating sets keep
the union–find over the `n²` pairs fast. With site dephasing, Q₁₂ from a
node has 455 orbitals instead of 1.7·10⁷ entries: 0.26 s to set up and
16 ms to evolve.

```python
import numpy as np
import metatron_qso as q

g = q.Graph.petersen()
walk = q.SymmetricOpenWalk(g, gamma=0.8, dephasing=[(0.6, list(range(10)))],
                           symmetries=g.stabilizer_generators([0]))
x = walk.evolve(walk.localized(0), 2.3)
full = q.DephasingWalk(g, gamma=0.8, rate=0.6)
assert np.abs(walk.populations(x) - full.populations(0, 2.3)).max() < 1e-9
print(walk.num_orbitals)                            # 15 instead of 100
```

Beyond 8192 nodes the `n²` pairs no longer fit, and the orbitals come from
the orbits of the stabiliser of one node of each orbit on nodes, in
memory of the nodes times the orbits on nodes (one row of labels per
orbit on nodes, at most 2²⁷ labels). For small orbits every Schreier generator of the
stabiliser is used; otherwise random elements of the group (seeded, so
reproducible) until 64 in a row merge nothing. A missed merge would only
split an orbital into several classes, which keeps every invariant state
exact. The
dense `coefficients` and `operator` stop at 8192 nodes, and the dephasing
must be kept term by term: every symmetry maps each partition onto one
with the same rate (terms with the same partition count as one, with
their rates added). From a node with site and coordinate dephasing,
Q₁₈ (262 144 nodes, a density matrix of 6.9·10¹⁰ entries) has 1330
orbitals: 6.8 s to set up and 0.06 s to evolve on four cores; the Hamming
graph H(7, 5) (78 125 nodes) has 330 and takes 1.0 s.

```python
import numpy as np
import metatron_qso as q

d = 14                                              # 16 384 nodes
n = 1 << d
dephasing = [(0.3, list(range(n)))] + [(0.2, [(u >> k) & 1 for u in range(n)]) for k in range(d)]
walk = q.SymmetricOpenWalk(q.Graph.hypercube(d), dephasing=dephasing,
                           symmetries=q.coordinate_permutations(d))
print(walk.num_orbitals)                            # C(17, 3) = 680
p = walk.populations(walk.evolve(walk.localized(0), 0.9))
by_distance = np.bincount([bin(u).count("1") for u in range(n)], weights=p)
ref = q.HypercubeOpenWalk(d, site_rate=0.3, coordinate_rate=0.2)
assert np.abs(by_distance - ref.distance_distribution(ref.evolve(ref.localized(), 0.9))).max() < 1e-12
```

`HypercubeOpenWalk(d, gamma, site_rate, coordinate_rate)` does the same for
the hypercube without building it. A pair of nodes is characterised by how
many coordinates have bits 00, 01, 10 and 11, which gives `C(d + 3, 3)`
orbitals for 2^d nodes, up to d = 230. Its coefficients are those of ρ in
the orthonormal basis of the orbitals, where the Hamiltonian part is
unitary, so they are accurate to rounding in the Hilbert–Schmidt norm. The
probability of distance k from the start carries an error of about
`10⁻¹⁶·√C(d, k)`. On four cores, Q₆₀ with site dephasing (39 711
orbitals) evolves for γt = 1 in 3.3 s, and Q₁₀₀ (176 851 orbitals) in
15 s. With dephasing on every coordinate the walk is a product of
two-level systems, and the distance follows a binomial law. Without
dephasing it mixes uniformly at γt = π/4 (Moore and Russell):

```python
import math
import numpy as np
import metatron_qso as q

d, gamma, kappa, t = 40, 0.7, 0.9, 1.3
walk = q.HypercubeOpenWalk(d, gamma=gamma, coordinate_rate=kappa)
p = walk.distance_distribution(walk.evolve(walk.localized(), t))
w = math.sqrt(4 * gamma**2 - kappa**2 / 4)
z = math.exp(-kappa * t / 2) * (math.cos(w * t) + kappa / (2 * w) * math.sin(w * t))
flip = (1 - z) / 2
binom = np.array([math.comb(d, k) * flip**k * (1 - flip) ** (d - k) for k in range(d + 1)])
assert np.abs(p - binom).max() < 1e-9

coherent = q.HypercubeOpenWalk(d)
x = coherent.evolve(coherent.localized(), math.pi / 4)
assert abs(abs(coherent.entry(x, 0, 0, 7)) - 2.0**-d) < 1e-20   # every node 2^-d
```

`ProductOpenWalk` takes this to any number of factors. A walk on a
Cartesian product whose factors dephase independently stays a product
state, so each factor evolves on its own: `[(DephasingWalk(edge, …), 1000)]`
is Q₁₀₀₀ with coordinate dephasing, and `count_distribution` gives the
distribution of the distance travelled. Its factors may carry site
energies and dephasing per node, but not traps or losses: a factor that
loses population has marginals that are not distributions, so such
factors are refused (use the walk on the product graph).

**Traps on orbitals and hypercubes.** The [transport to a
trap](walks.md#transport-to-a-trap) reduces in the same way.
`SymmetricOpenWalk` takes `site_energies`, `node_dephasing` (added to
its `dephasing` terms), `traps` and `losses`, each constant on every
orbit of the symmetries on nodes; anything else is refused, naming two
nodes of one orbit with different values. A whole orbit of traps with
equal rates is allowed, and the start must be fixed by every symmetry.
`transport`, `dephasing_scan`, `optimal_dephasing` and `transport_trace`
then work on the orbitals. On the complete graph K_N, the symmetries
that fix the start and the trap leave ten orbitals at any N, and
seventeen when they also fix a dephased node:

```python
import metatron_qso as q

n = 1024
fix_01 = [[0, 1, 3, 2] + list(range(4, n)), [0, 1] + list(range(3, n)) + [2]]
walk = q.SymmetricOpenWalk(q.Graph.complete(n), "adjacency", symmetries=fix_01, traps={1: 1.0})
r = walk.transport(0)
print(walk.num_orbitals, round(r["efficiency"] * (n - 1), 9), round(r["arrival_time"], 4))  # 10 1.0 511.501
assert abs(r["arrival_time"] - (n * n - 2 * n + 3) / (2 * (n - 1))) < 1e-9

fix_012 = [[0, 1, 2, 4, 3] + list(range(5, n)), [0, 1, 2] + list(range(4, n)) + [3]]
one = q.SymmetricOpenWalk(q.Graph.complete(n), "adjacency", symmetries=fix_012,
                          traps={1: 1.0}, node_dephasing=[0, 0, 0.8] + [0] * (n - 3))
r = one.transport(0)
print(one.num_orbitals, round(r["efficiency"] * (n - 2), 9))            # 17 1.0
```

The solves are GMRES on the orbital generator in the orthonormal basis
of the orbitals, preconditioned by a banded LU of the generator shifted
by `max(10⁻²·hopping, 10⁻⁶·‖M‖∞)`, its orbitals ordered by reverse
Cuthill–McKee to keep the band narrow. Orbitals cannot be split by a
closure into bright and dark states as the dense path does, so the
first solve is the adjoint one: it gives, for every orbital, the
probability of being trapped or lost from there, which projects the
start onto the bright states. The forward solve starts from that
projection, so it stays consistent where the generator is singular, also
on graphs with signed couplings, and the efficiency is read from both
solves; their difference enters `error`. When every node loses
population and the smallest loss is below 10⁻⁶ of the generator's norm,
it is treated as a uniform shift, as on the dense path: the projection comes from the generator without it, and the
start's weight outside the bright states counts as lost. Without a
closure, "dark" on orbitals can only mean "not absorbed within about
1/(ε·‖M‖)": when `condition·ε` exceeds 10³·`dark_tolerance`, or the
dark weight is below 10³·`condition·ε`, a dark weight above
`dark_tolerance` is added to `error` and τ is withheld, since a bright state
decaying below rounding would look the same (raise `dark_tolerance` to
accept it as dark). The arrival time is checked by a second, goal-oriented adjoint
solve and reported only when both agree to 10⁻³.

`HypercubeOpenWalk` takes the same terms by Hamming weight (arrays of
`d + 1` values) and traps as `(weight, κ)`: every node of that weight is
a trap, and `(d, κ)` is the antipode of the start. Transport runs up to
d = 61, where the banded LU reaches its limit of 2²⁷ entries (2 GiB).
Without dephasing at the interior weights, the walk from the start
stays on the distance partition (one cell per weight, the column chain of
`Quotient.hypercube`) at any d, so a `DephasingWalk` on the weighted path
with couplings `√((k + 1)(d − k))` gives the same results beyond that:

```python
import math
import numpy as np
import metatron_qso as q

cube = q.HypercubeOpenWalk(10, losses=0.05, traps=[(10, 1.0)])
scan = cube.dephasing_scan([0, 0.1, 1, 10])          # dephasing on every node
print(np.round(scan["efficiency"], 6))               # [0.831038 0.690879 0.234277 0.011426]
print(np.round(scan["arrival_time"], 4))             # [ 3.1184  2.6003  2.9052 25.5786]

def chain(d: int) -> q.DephasingWalk:
    edges = [(k, k + 1) for k in range(d)]
    weights = [math.sqrt((k + 1) * (d - k)) for k in range(d)]
    return q.DephasingWalk(q.Graph.from_edges(d + 1, edges, weights=weights), "adjacency",
                           traps={d: 1.0}, losses=0.05)

q20 = q.HypercubeOpenWalk(20, losses=0.05, traps=[(20, 1.0)]).transport()
assert abs(q20["efficiency"] - chain(20).transport(0)["efficiency"]) < 1e-12
far = chain(200).transport(0)                        # Q₂₀₀
print(f"{far['efficiency']:.6f} {far['arrival_time']:.4f}")
```

The chain's own Laplacian is not the hypercube's, so the chain takes the
adjacency generator (a constant diagonal changes nothing). With
dephasing inside, the populations of one weight no longer move together
and the chain is not exact (Q₄ with dephasing 0.5 and loss 0.05: 0.7104
on the cube, 0.8497 on the chain).

**Accuracy on large hypercubes.** Two kinds of hypercube transport are
ill-conditioned, and only one shows it. Without dephasing and loss, some
states of the chain reach the trap extremely slowly; η = 1 stays exact,
but the arrival time from the orbitals is off by 2.3·10⁻¹⁰ (relative) at
Q₃₀ and 1.2·10⁻⁴ at Q₄₀ against an 80-digit solution of the chain, while
`error` (10⁻¹⁵) and `condition` (2.7·10²) look harmless. The dense solve of
the chain is off by 3.8·10⁻⁷ at Q₄₀ and does report the conditioning
(4.4·10¹³). With dephasing on every node and no loss the walk is nearly
classical and the arrival time grows to 6.6·10¹¹ at Q₄₀; there η is off
by 1.0·10⁻⁹ at Q₃₀, 7.1·10⁻⁷ at Q₄₀ and 4.1·10⁻³ at Q₅₀, which `error`
does show (3.1·10⁻⁹, 2.1·10⁻⁶, 1.2·10⁻²), and τ is not resolved at Q₅₀.
With a loss of 0.05 on every node, as in the examples above, the error
estimates stay below 3·10⁻¹⁵ up to Q₄₀ (condition below 2·10³). Slow
bright modes cost τ digits on small graphs as well, for instance with a
loss of 10⁻⁶ on most nodes of K₁₆; `condition` then is large.

**Coined walks.** `SymmetricCoinedWalk` reduces a `DecoherentCoinedWalk` to
the orbitals of pairs of arcs. The coin must be invariant under the
symmetries; the Grover coin always is. Coin dephasing measures each arc's
label, by default its index among the arcs leaving its node, and that
label is rarely invariant. On a hypercube, label every arc by the
coordinate it flips (`with_coin_labels`) and coin dephasing reduces under
the coordinate permutations. Q₈ with position dephasing (2048 arcs, the
limit of the full walk) has 1824 orbitals: twenty steps take 1 ms instead
of 4.9 s on the full density matrix.

```python
import numpy as np
import metatron_qso as q

g = q.Graph.hypercube(4)
walk = q.CoinedWalk(g, "grover")
labels = [(u ^ v).bit_length() - 1 for u, v in walk.arcs()]
full = q.DecoherentCoinedWalk(walk, 0.3, 0.2).with_coin_labels(labels)
reduced = q.SymmetricCoinedWalk(full, q.coordinate_permutations(4))
p = reduced.node_probabilities(reduced.evolve(reduced.localized(0), 7))
assert np.abs(p - full.probabilities(0, 7)).max() < 1e-12
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

## Pretty good state transfer

Perfect state transfer is rare. *Pretty good* state transfer (PGST) asks
less: fidelities arbitrarily close to 1 at suitable times. For strongly
cospectral u and v the transfer amplitude is `Σ_r σ_r ‖E_r e_u‖² e^{−iθ_r t}`,
and by Kronecker's theorem it comes arbitrarily close to modulus 1 exactly
when every integer relation `Σ ℓ_r (θ_r − θ_0) = 0` among the eigenvalues in
the support has an even `Σ ℓ_r δ_r`, where `δ_r` marks the eigenspaces in
which u and v have opposite signs (Banchi, Coutinho, Godsil and Severini
2017). The relations are a number-theoretic property of the eigenvalues
that floating point cannot see, so `PrettyGoodTransfer` computes them
exactly for graphs whose eigenvalues are known in a cyclotomic field
`Q(ζ_N)`:

- paths (adjacency and Laplacian),
- cycles,
- Cayley graphs of finite abelian groups `Z_{m_1} × … × Z_{m_k}`: circulants,
  hypercubes and their relatives.

Each eigenvalue becomes an integer vector in the power basis of `Q(ζ_N)`
(powers of ζ reduced modulo the cyclotomic polynomial); the relations are
the integer kernel of these vectors, found by integer column reduction,
and the parity condition is checked on a basis of it. When there is no
PGST, `obstruction` returns a relation with odd parity. The decision is
exact. When there is PGST and the differences `θ_r − θ_0` span a space of
rank 1, there is also perfect state transfer, at the time π/g read off the
integer data. `fidelity(t)` is the exact transfer probability for any pair,
from the closed-form eigenvectors. `best_transfer` searches a window for a
high fidelity, but PGST times can lie far beyond any window.

```python
import metatron_qso as q

for n in [4, 8, 9, 10]:
    t = q.PrettyGoodTransfer.path(n, 0, n - 1)
    print(n, t.pretty_good, t.perfect_time)        # True, False, True, True
p4 = q.PrettyGoodTransfer.path(4, 0, 3)
print(p4.best_transfer(2000.0, 200_000))           # fidelity > 0.999
print(q.PrettyGoodTransfer.path(4, 0, 3, "laplacian").pretty_good)   # True: 4 = 2²
print(q.PrettyGoodTransfer.cycle(12, 0, 6).pretty_good)              # False
```

The tests check three theorems for every size in a range: the ends of the
path `P_n` have PGST exactly when n + 1 is a prime, twice a prime or a power
of two (Godsil, Kirkland, Severini and Smith 2012; n ≤ 300), the ends of the
Laplacian path (the Heisenberg chain) exactly when n is a power of two
(Banchi et al. 2017; n ≤ 200), and antipodal vertices of the cycle `C_n`
exactly when n is a power of two (Pal 2017; n ≤ 512). They also check
strong cospectrality, PST times and fidelities (for every pair of nodes)
against `StateTransfer` on hypercubes, circulants, paths, tori, `Z_3 × Z_5`
and the Hamming graph H(2, 3), every obstruction, and a time with fidelity
above 0.999 on `P_4` against the propagated walk.

The integer arithmetic is exact or fails: on overflow of 128-bit integers
the answer is an error, never a guess. The cost grows with the number of
distinct eigenvalues m and the field degree d, roughly as m²(m + d), and
the search holds m(m + d) integers, at most 2²⁴: larger groups are refused
before the search starts. In an independent review on one core, cycles
with 2^k nodes were fast (C₄₀₉₆ in 0.6 s, Q₁₆ in 65 ms), but other cycle
sizes above about 2000 nodes took 10 to 60 s and some overflowed, the path
on 1000 nodes overflowed while those on 1200 and 2000 were decided, and
Laplacian paths of prime length near 800 took about 7 s. `qso transfer`
prints the exact decision for paths and cycles, and says why when it is
undecided.

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

The open walks of this chapter in Rust:

```rust
use metatron_qso::prelude::*;
use metatron_qso::walks::symmetric::{HypercubeOpenWalk, coordinate_permutations};
use metatron_qso::walks::{Dephasing, SymmetricOpenWalk};

fn main() -> Result<()> {
    let g = metatron_qso::graph::generators::hypercube(10);
    let walk = SymmetricOpenWalk::new(
        &g,
        Generator::Laplacian,
        1.0,
        &[Dephasing::sites(1024, 0.5)],
        &coordinate_permutations(10)?,
    )?;
    assert_eq!(walk.num_orbitals(), 286);
    let p = walk.populations(&walk.evolve(&walk.localized(0)?, 1.0)?)?;
    assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);

    let big = HypercubeOpenWalk::new(50, 1.0, 0.5, 0.0)?;
    let d = big.distance_distribution(&big.evolve(&big.localized(), 0.5)?)?;
    assert!((d.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    Ok(())
}
```
