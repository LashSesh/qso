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

**Site energies** `ε_j` add `Σ_j ε_j|j⟩⟨j|` to `H`; they are not scaled
by γ. A detuning holds the walker back: on a dimer with energies 3 and 0
at most 4/13 of it crosses over, and 2/13 on average. Beyond 2048 nodes,
energies that take one value on each cell of the start's equitable
partition (colour refinement now also separates the energies) keep the
walk on the cells, so the hypercube Q₁₂ with energies by distance from
the start still has 13 dimensions; otherwise the cyclic subspace or Krylov
propagation takes over, as without energies.

```python
import math
import metatron_qso as q

dimer = q.ContinuousWalk(q.Graph.path(2), "adjacency", site_energies=[3.0, 0.0])
print(dimer.probabilities(0, math.pi / math.sqrt(13)))       # 9/13 and 4/13
assert abs(dimer.limiting_distribution(0)[1] - 2 / 13) < 1e-12

d = 12
weights = [bin(u).count("1") for u in range(1 << d)]
cube = q.ContinuousWalk(q.Graph.hypercube(d), site_energies=[0.1 * w for w in weights])
assert cube.spectrum is None and cube.cyclic_dimension(0) == 13
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

`DephasingWalk` adds Lindblad dephasing at a given `rate`. With rate 0 it
reproduces the coherent walk. For strong dephasing the populations follow
the classical master equation dp/dt = −(2γ²/rate)·L·p. This is the quantum
Zeno regime: the walk becomes classical and slows down as the rate grows.
On symmetric graphs the density matrix reduces to the orbitals of the
symmetry group, which reaches hypercubes of any dimension up to 230 (see
[Symmetry and state transfer](symmetry.md)). The density matrix is
evolved for up to 512 nodes.

```python
import numpy as np
import metatron_qso as q

g = q.Graph.fruit_of_life()
for rate in [0.0, 1.0, 10.0]:
    p = q.DephasingWalk(g, rate=rate).populations(0, 3.0)
    print(rate, np.round(p[:3], 4))
print(np.round(q.classical_walk_probabilities(g, 0, 3.0)[:3], 4))
```

**Dephasing per node.** `node_dephasing` gives every node its own rate
`γ_j` instead of one `rate` (give one or the other), and `site_energies`
adds energies as in `ContinuousWalk`. The master equation is
`dρ/dt = −i[H, ρ] + Σ_j γ_j (P_j ρ P_j − ½{P_j, ρ})` with `P_j = |j⟩⟨j|`,
so the entries of ρ change as follows:

| Entry | Decays at | Rotates at |
|---|---|---|
| `ρ_jj` | — (with traps and losses: `2κ_j + Γ_j`) | — |
| `ρ_ab`, `a ≠ b` | `(γ_a + γ_b)/2` (with traps and losses: `+ k_a + k_b`, `k = κ + Γ/2`) | `−i(ε_a − ε_b)` |

Equal rates on every node are the uniform `rate`, bit for bit:

```python
import numpy as np
import metatron_qso as q

g = q.Graph.fruit_of_life()
uniform = q.DephasingWalk(g, rate=0.5).populations(0, 1.7)
per_node = q.DephasingWalk(g, node_dephasing=[0.5] * 13).populations(0, 1.7)
assert np.array_equal(uniform, per_node)
one_node = q.DephasingWalk(g, node_dephasing=[2.0] + [0.0] * 12, site_energies=[0.3] * 13)
print(np.round(one_node.populations(0, 1.7)[:3], 4))
```

## Transport to a trap

An excitation hops over a network until a trap absorbs it or it is lost
on the way: the chromophores of a light-harvesting complex such as FMO,
photonic waveguides, trapped ions and superconducting circuits in which
dephasing is added on purpose. The questions are how much of it reaches
the trap (the *efficiency*), how fast (the *mean arrival time*) and at
which dephasing rate transport works best. `DephasingWalk` answers them
with linear solves, not by propagating to a guessed time.

### The model

```text
dρ/dt = −i(H_eff ρ − ρ H_eff†) + Σ_j γ_j (P_j ρ P_j − ½{P_j, ρ})
H_eff = H − i Σ_t κ_t P_t − (i/2) Σ_j Γ_j P_j,     P_j = |j⟩⟨j|
```

| Symbol | Meaning | Argument |
|---|---|---|
| `H` | `γL` or `−γA` plus `diag(ε)` | `generator`, `gamma`, `site_energies` |
| `γ_j` | dephasing rate of node `j` | `rate` (every node) or `node_dephasing` |
| `κ_t` | trap rate: the trap's sink fills at `2κ_t ρ_tt` | `traps`, a dict `{node: κ}` or pairs |
| `Γ_j` | loss rate: node `j` loses population at `Γ_j ρ_jj` | `losses`, one number for every node or one per node |

The literature writes these rates in more than one way. Converted to QSO:

| Written as | In QSO |
|---|---|
| dephasing `2γ_C(PρP − ½{P, ρ})` (Caruso, Chin, Datta, Huelga and Plenio, J. Chem. Phys. 131, 105106 (2009); Plenio and Huelga 2008) | `γ = 2γ_C` |
| loss `−iΓ_R P` in the effective Hamiltonian (Rebentrost, Mohseni, Kassal, Lloyd and Aspuru-Guzik, New J. Phys. 11, 033003 (2009)) | `Γ = 2Γ_R` |
| a sink fed at `κ′ρ_tt` | `κ = κ′/2` |

The factor matters: with the dephasing written the other way, the dimer
below has the efficiency 0.830723 at rate 1, not 0.823024.

### Efficiency and arrival time

The time integrals `X = ∫₀^∞ ρ dt` and `Y = ∫₀^∞ tρ dt` solve the linear
equations `𝓛(X) = −ρ₀` and `𝓛(Y) = −X`. From them come

- the efficiency `η = 2Σ_t κ_t X_tt`, and per trap in `trapped`;
- the population `lost` on the way, `Σ_j Γ_j X_jj`;
- the mean arrival time of the trapped walker, `τ = 2Σ_t κ_t Y_tt/η`;
- the weight `dark` that is never absorbed, so that
  `η + lost + dark = 1`. When every node loses population nothing stays
  dark, and that weight counts as lost.

The detuned dimer is the textbook case: the start at energy 3, the trap
at energy 0 with κ = 1, loss 0.05 and dephasing 1 on both sites, all in
units of the hopping. Its efficiency is 32000/38881 and its arrival time
139260/38881 exactly:

```python
import metatron_qso as q

dimer = q.DephasingWalk(q.Graph.path(2), "adjacency", rate=1.0,
                        site_energies=[3.0, 0.0], traps={1: 1.0}, losses=0.05)
r = dimer.transport(0)
print(f"{r['efficiency']:.6f} {r['arrival_time']:.4f}")       # 0.823024 3.5817
assert abs(r["efficiency"] - 32000 / 38881) < 1e-12
assert abs(r["arrival_time"] / (139260 / 38881) - 1) < 1e-10
assert abs(r["efficiency"] + r["lost"] + r["dark"] - 1) < 1e-13
print([(s["name"], s["iterations"], f"{s['condition']:.0f}") for s in r["solves"]])
```

The result also reports how the solves went. `dimension` is the number
of bright states the solves ran on, `solves` lists every solve with its
GMRES iterations, backward error and a lower estimate of its condition
number (the largest is `condition`), and `error` estimates the absolute
error of η: how far `η + lost + dark` is from 1, plus the last
correction of the iterative refinement. `transport_from(rho0)` starts
from a density matrix instead of a node.

### Dark states

A walker can be trapped only from states that reach a trap. On the
complete graph K_N without dephasing, the start and the trap span a
two-dimensional bright space; the start's weight outside it,
`(N − 2)/(N − 1)`, never arrives, so `η = 1/(N − 1)` (Caruso et al.,
Eq. 5). Dephasing at one more node gives `1/(N − 2)`, and dephasing at
every node of a connected graph without loss gives `η = 1`.
`dark_states()` returns an orthonormal basis of the states that never
reach a trap or a node with more than the smallest loss. Interference can
also hide a start completely: on a cycle of four nodes with one negative
coupling, the paths from the far node to the trap cancel.

```python
import numpy as np
import metatron_qso as q

k5 = q.Graph.complete(5)
r = q.DephasingWalk(k5, "adjacency", traps={1: 1.0}).transport(0)
print(r["efficiency"], r["dark"], r["dimension"])               # 1/4, 3/4, 2
dark = q.DephasingWalk(k5, "adjacency", traps={1: 1.0}).dark_states()
assert dark.shape == (5, 3) and abs(dark[0] @ dark[0] - 0.75) < 1e-12
one = q.DephasingWalk(k5, "adjacency", traps={1: 1.0}, node_dephasing=[0, 0, 0.8, 0, 0])
assert abs(one.transport(0)["efficiency"] - 1 / 3) < 1e-12
every = q.DephasingWalk(k5, "adjacency", rate=1.0, traps={1: 1.0}).transport(0)
assert abs(every["efficiency"] - 1) < 1e-12 and abs(every["arrival_time"] - 35 / 4) < 1e-9

c4 = q.Graph.from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)], weights=[1, 1, 1, -1])
hidden = q.DephasingWalk(c4, "adjacency", traps={0: 1.0}).transport(2)
print(hidden["efficiency"], hidden["dark"], hidden["arrival_time"])   # 0.0 1.0 None
opened = q.DephasingWalk(c4, "adjacency", traps={0: 1.0}, node_dephasing=[0, 0.5, 0, 0])
assert abs(opened.transport(2)["arrival_time"] - 377 / 20) < 1e-9
```

The bright states are found by a closure: start from the traps and the
nodes with more than the smallest loss, apply `H` and, at dephased nodes,
the node projectors, and keep every new direction longer than
`dark_tolerance` (default 10⁻¹⁰, relative to the largest entry of the
centred `H`). A state coupled more weakly than that counts as dark. An
almost dark state shows as a small `slowest_decay`, the slowest decay
rate of the bright states without the refeeding by dephasing; a bright
state that decays below rounding is refused with the advice to raise
`dark_tolerance`.

### Scans and the best dephasing rate

Moderate dephasing helps transport: it moves the walker out of the
superpositions that interfere destructively at the trap or that sit off
resonance, until the quantum Zeno effect freezes it (Plenio and Huelga,
New J. Phys. 10, 113019 (2008); Rebentrost et al.). `dephasing_scan` adds
a uniform rate to every node and returns arrays of the results (an
arrival time that is not resolved is NaN). A uniform rate shifts `H_eff`
by a constant on the bright states, so at most two Schur forms serve the
whole scan: one for the positive rates and, when the scan includes rate 0
and the walk does not dephase every node already, one for rate 0.
`optimal_dephasing(start, low, high)` evaluates a logarithmic grid of at
least 13 rates over `[low, high]` and refines the best one by golden
section in the logarithm of the rate; `interior` is false if the best
rate is an end of the range or η is flat. Only positive rates are
searched, because η can jump at rate 0 when dephasing opens dark states
(K₅ above: ¼ at rate 0, 1 at any positive rate).

```python
import numpy as np
import metatron_qso as q

dimer = q.DephasingWalk(q.Graph.path(2), "adjacency",
                        site_energies=[3.0, 0.0], traps={1: 1.0}, losses=0.05)
scan = dimer.dephasing_scan(0, [0, 1, 3, 10, 100])
print(np.round(scan["efficiency"], 6))     # [0.771278 0.823024 0.825951 0.738538 0.274598]
print(np.round(scan["arrival_time"], 4))   # [ 4.5256  3.5817  3.5553  5.3563 14.8622]
best = dimer.optimal_dephasing(0, 0.01, 100)
print(f"{best['rate']:.2f} {best['efficiency']:.6f}")    # 1.95 0.830737
assert abs(best["rate"] - 39 / 20) < 1e-6 and abs(best["efficiency"] - 800 / 963) < 1e-12
```

The dimer's best rate is 39/20 with η = 800/963 exactly; the search finds
it to about 10⁻⁷, since η is flat at its maximum.

### Traces

`transport_trace(start, times)` propagates the density matrix (up to 512
nodes, by the Runge–Kutta integrator of `evolve`) together with the sink
of every trap and the population lost, and returns them at the given
times. Their sum with the populations left is 1, and the flux into the
sink gives the arrival time again:

```python
import numpy as np
import metatron_qso as q

dimer = q.DephasingWalk(q.Graph.path(2), "adjacency", rate=1.0,
                        site_energies=[3.0, 0.0], traps={1: 1.0}, losses=0.05)
times = np.linspace(0.0, 60.0, 601)
trace = dimer.transport_trace(0, times)
sink = trace["trapped"][:, 0]                       # one column per trap
assert np.abs(sink + trace["lost"] + trace["populations"].sum(axis=1) - 1).max() < 1e-9
mid = (times[1:] + times[:-1]) / 2
print(f"{sink[-1]:.6f} {np.sum(mid * np.diff(sink)) / sink[-1]:.3f}")   # 0.823024 3.582
```

### FMO, a tutorial input

The seven-site Hamiltonian of the Fenna–Matthews–Olson complex of
*C. tepidum* by Adolphs and Renger (Biophys. J. 91, 2778 (2006)), as
tabulated by Sarovar et al. (Nat. Phys. 6, 462 (2010), supplementary
information), is the usual test input; this is no claim about the
complex. In picoseconds, with the trap at BChl 3 (κ = 1/ps) and a loss
of 1/ns: a common energy offset changes nothing, so the energies are
taken relative to their mean.

```python
import math
import numpy as np
import metatron_qso as q

fmo = np.array([   # cm⁻¹
    [12410, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9],
    [-87.7, 12530, 30.8, 8.2, 0.7, 11.8, 4.3],
    [5.5, 30.8, 12210, -53.5, -2.2, -9.6, 6.0],
    [-5.9, 8.2, -53.5, 12320, -70.7, -17.0, -63.3],
    [6.7, 0.7, -2.2, -70.7, 12480, 81.1, -1.3],
    [-13.7, 11.8, -9.6, -17.0, 81.1, 12630, 39.7],
    [-9.9, 4.3, 6.0, -63.3, -1.3, 39.7, 12440]])
cm = 2 * math.pi * 2.99792458e-2                    # 1 cm⁻¹ in rad/ps
edges = [(i, j) for i in range(7) for j in range(i + 1, 7)]
g = q.Graph.from_edges(7, edges, weights=[-fmo[i, j] * cm for i, j in edges])   # −A carries V_ij
walk = q.DephasingWalk(g, "adjacency", site_energies=(np.diag(fmo) - np.diag(fmo).mean()) * cm,
                       traps={2: 1.0}, losses=1e-3)
scan = walk.dephasing_scan(0, [0, 1, 10, 100])
print(np.round(scan["efficiency"], 4), np.round(scan["arrival_time"], 2))
# [0.9468 0.9819 0.9937 0.9931] [47.09 18.15  6.29  6.91]
best = walk.optimal_dephasing(0, 1, 1000)
print(f"{best['rate']:.1f} /ps ({best['rate'] / cm:.0f} cm⁻¹): {best['efficiency']:.5f}")
# 26.0 /ps (138 cm⁻¹): 0.99439
```

Without dephasing 5.3 % of the excitation is lost; at the best rate,
0.56 %, and it arrives eight times sooner (5.6 ps instead of 47 ps). These values agree with
QuTiP's `mesolve` integrated until less than 10⁻¹² of the population is
left (see [validation](validation.md)).

### Accuracy and limits

- **Sizes.** The dense solves take up to 2048 nodes; evolution and
  traces up to 512. A transport call estimates its work and stops beyond
  2·10¹³ operations (`MAX_TRANSPORT_WORK`). On four cores a random graph
  of 1024 nodes with dephasing everywhere takes 26 s for the Schur form
  and 47 s for the solves; at 2048 nodes 5.5 and 6 minutes, and there
  the limit leaves room for only about a hundred iterations over the
  dephased populations. On symmetric graphs the
  solves run on orbitals, for graphs far larger (see
  [Symmetry and state transfer](symmetry.md)).
- **η** comes with `error`, an estimate of its absolute error.
- **τ** has no error estimate of its own. It is reported only when
  `η ≥ max(10⁻¹², 10³·error)` and is `None` otherwise: for a walker that
  hardly arrives, τ is not resolved. Its accuracy follows the
  conditioning of the problem, which `condition` estimates for each
  solve; on large coherent hypercubes the orbital solves lose digits
  that their `condition` does not show (see [accuracy on large
  hypercubes](symmetry.md#driven-and-decoherent-walks-reduced-by-symmetry)).
- **Optimum.** The best rate is found to about 10⁻⁷ relative where η is
  flat at its maximum (η itself to rounding).

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

On a strongly regular graph the walk propagator `exp(−iAt)` is a
combination of `I`, `A` and `J − I − A` whose coefficients depend only on
the graph's parameters. So one walker cannot tell apart strongly regular
graphs with the same parameters, and two free particles cannot either.
Two interacting bosons can (Gamble, Friesen, Zhou, Joynt and
Coppersmith, Phys. Rev. A 81, 052313 (2010)), though not on every pair
of non-isomorphic graphs (Smith, arXiv:1004.0206 (2010)).

`certificate(times)` lists the transition probabilities between all
configurations, sorted, at each time. Isomorphic graphs give the same
list up to rounding, which grows with the time.
`certificate_tolerance(times)` is `10⁻¹² · max(1, |t| · ‖H‖)`, with `‖H‖`
bounded by the largest absolute row sum: a thousand times the rounding
measured on isomorphic graphs up to `t = 10⁶`, or more. A
larger distance shows, up to floating-point error, that two graphs are
not isomorphic:

```python
import metatron_qso as q

def certificate(graph, u):
    return q.InteractingWalk(graph, 2, on_site=u).certificate([1.0])

t8, chang = q.Graph.johnson(8, 2), q.Graph.chang(1)   # both (28, 12, 6, 4)
for u in [0.0, 1.0]:
    d = q.interacting_certificate_distance(certificate(t8, u), certificate(chang, u))
    tolerance = q.InteractingWalk(t8, 2, on_site=u).certificate_tolerance([1.0])
    print(u, d > tolerance)
# 0.0 False
# 1.0 True
```

The validation suite checks this for all six pairs among `T(8)` and
the three Chang graphs (`Graph.chang(1)` to `Graph.chang(3)`, obtained
from `T(8)` by `seidel_switching`), and for the Shrikhande and 4 × 4
rook's graphs. Without interaction the certificates of every pair agree
to 10⁻¹². With `U = 1` they differ by 1.7·10⁻⁴ to 3.4·10⁻⁴ (`T(8)` and
the Chang graphs) and by 5.4·10⁻³ (Shrikhande and rook), at least a
million times the tolerance. The same comparison runs from the command
line with `qso particles --graph shrikhande --compare rook:4x4 --on-site 1`.

The amplitudes are checked in the tests against independent references:

- without interaction, against permanents (bosons) and determinants
  (fermions) of the single-walker propagator, also for 2925
  configurations propagated by Krylov; with interaction, the Krylov
  propagation of 3003 fermion and 2744 distinguishable configurations
  against an independent Taylor integration;
- against the Hamiltonian built on the full tensor-product space and
  restricted to the symmetric or antisymmetric subspace, up to four
  particles on a complete graph with weights of both signs;
- against the closed-form spectrum of the Hubbard dimer.

### Symmetric graphs

The configurations grow fast: three bosons on the hypercube `Q₈` have
2 829 056 of them, beyond the 2²² that `InteractingWalk` accepts. A walk
that starts in a state left alone by some automorphisms of the graph
stays in such states, and those need one amplitude per *orbit* of the
automorphisms on the configurations. `SymmetricInteractingWalk` works on
the orbits. `from_start(graph, start)` takes the automorphisms that fix
every occupied node (`graph.stabilizer_generators(start)`), and the
constructor takes any generators of automorphisms:

```python
import metatron_qso as q

g = q.Graph.hypercube(8)
walk = q.SymmetricInteractingWalk.from_start(g, [0, 0, 0], on_site=1.0)
print(walk.num_orbits, walk.num_configurations)
# 1324 2829056
x = walk.evolve(walk.state_of([0, 0, 0]), 1.0)
print(round(walk.densities(x).sum(), 12), walk.probability_of(x, [0, 0, 0]) < 1e-4)
# 3.0 True
```

States are vectors over the orbits, one unit vector per orbit:
`|O⟩ = |O|^{−1/2} Σ_{c∈O} ±|c⟩`. For fermions the sign is that of sorting
their sites after the permutation; a configuration that some
automorphism maps to minus itself has no symmetric state, and its orbit
drops out (`state_of` refuses it); when every orbit drops out, as for two
fermions under the whole group of a distance-transitive graph, the walk
is refused. `densities`, `pair_correlations`, `probability_of`,
`orbit_probabilities` and `lift` (the amplitudes on all configurations,
in `InteractingWalk`'s order) read the state. The orbits are found by
union–find over all configurations, so at most 2²⁷ of them; give a small
set of generators. The reduced Hamiltonian may have 2²⁶ entries.

On the hypercube the orbits of the coordinate permutations need not be
found: `HypercubeParticles(d, particles)` writes them down as the numbers
of coordinates in which the particles' bits form each pattern, for up to
four particles and without building the graph. Two bosons on `Q₄₀`
(2⁴⁰ nodes) have 6391 states. Free bosons both arrive at the antipode at
`γt = π/2`, as one walker does; an on-site interaction spoils the
transfer:

```python
import math
import metatron_qso as q

for u in [0.0, 1.0, 10.0]:
    cube = q.HypercubeParticles(40, 2, on_site=u)
    x = cube.evolve(cube.localized(), math.pi / 2)
    print(u, cube.num_states, round(cube.antipode_probability(x), 4))
# 0.0 6391 1.0
# 1.0 6391 0.9938
# 10.0 6391 0.5364
```

`weights` gives the expected number of particles at each Hamming
weight, `pair_distances` the distribution of the distance between two
particles. With the same 2²⁶ entries, two bosons reach `Q₄₄₄`, three
`Q₃₂` and four `Q₁₂`.

Beyond the hypercube from node 0, `CoordinateInteractingWalk` takes any
start on a graph whose symmetries act on the places of its nodes' words:
Hamming graphs `H(d, q)` (with `Q_d` and the complete graph `K_q`),
Johnson graphs `J(n, k)` and Kneser graphs `K(n, k)`, numbered as
`Graph.hamming`, `Graph.johnson` and `Graph.kneser` number them (`word`
and `node` convert). The group is the start's pointwise stabiliser in the
permutations of the places (with, for Hamming graphs, the symbols renamed
in each place). A configuration with the start's nodes above it is a
matrix of letters, and its canonical form, the columns renamed in order
of first appearance and sorted, minimised over the orders of the
particles, names its orbit. The same form counts its stabiliser, which
gives the orbit's size, and for fermions the parity of the order gives its
sign. The orbits are found by breadth-first search from the start, a
canonical form per hop, so neither the graph nor the configurations are
ever listed:

```python
import metatron_qso as q

walk = q.CoordinateInteractingWalk.johnson(40, 3, [0, 0, 1], on_site=1.0)
print(walk.num_nodes, walk.num_orbits)   # 9880 1674 (1.6·10¹¹ configurations)
x = walk.evolve(walk.state_of([0, 0, 1]), 1.0)
print(round(walk.probability_of(x, [0, 0, 1]), 4))   # 0.7273
```

`node_densities` gives the density per orbit of nodes. On four cores
(release build) the orbits took:

| Graph | Particles from | Configurations | Orbits | Search | Evolution to t = 1 |
|---|---|---|---|---|---|
| `Q₂₀` (10⁶ nodes) | 3 bosons, node 0 | 1.9·10¹⁷ | 156 240 | 9.0 s | 2.3 s |
| `Q₁₂` | 3 fermions, nodes 0, 1, 2 | 1.1·10¹⁰ | 199 000 | 6.7 s | 0.4 s |
| `H(8, 4)` (65 536 nodes) | 3 fermions, nodes 0, 1, 2 | 4.7·10¹³ | 1 204 136 | 104 s | 18 s |
| `Q₆` | 5 bosons, node 0 | 1.0·10⁷ | 28 576 | 4.7 s | 0.5 s |
| `J(40, 3)` | 3 bosons, nodes 0, 0, 1 | 1.6·10¹¹ | 1674 | 0.5 s | 4.4 s |
| `K(30, 3)` | 2 distinguishable, nodes 0, 1 | 1.6·10⁷ | 228 | 0.4 s | < 0.1 s |

At most six particles, 2²² orbits and 2²⁶ entries. Only a lower bound
on the orbits is checked before the search, so a start whose stabiliser
is small can search for minutes before it is refused (three bosons on
nodes 0, 1 and 2 of `Q₂₀` exceed the entries). The complement of
`J(2k, k)` and the pair swaps of `K(2k, k)` are not used; on `K(2k, k)`,
a perfect matching, the start reaches only the configurations on its own
pairs, `num_orbits` counts those, and other configurations have
probability zero (`state_of` refuses them).

On any other graph, `GroupInteractingWalk` takes a group of the graph's
automorphisms and finds the orbits from minimal images: the smallest
configuration in each orbit, found along a chain of point stabilisers
(Jefferson, Jonauskyte, Pfeiffer and Waldecker, J. Algebra 521, 481
(2019)). The smallest node any symmetry can bring a particle to is the
smallest orbit representative among the particles' nodes; every
particle that reaches it is a branch, and the search goes on in the
stabiliser of that node. Each stabiliser met is built once by a
randomised Schreier–Sims algorithm, stopped when its order reaches the
group's order divided by the orbit, which makes it exact. The surviving
branches count the stabiliser of the configuration, and for fermions
they must agree on the sign. With `whole_group=False` the group is the
stabiliser of the start's nodes and the start is the initial state; with
`whole_group=True` it is the whole automorphism group, and the states are
orbit states:

```python
import metatron_qso as q

g = q.Graph.paley(401)
walk = q.GroupInteractingWalk(g, [0, 1, 3], whole_group=True,
                              statistics="fermions", on_site=1.0)
print(walk.num_orbits)   # 132 orbits of 1.1·10⁷ configurations
x = walk.evolve(walk.state_of([0, 1, 3]), 1.0)
print(round(walk.probability_of(x, [0, 1, 3]), 8))   # 7.52e-06
```

On four cores (release build, interaction `U = 1`, `t = 1`):

| Graph | Particles from | Group | Configurations | Orbits | Group, search, evolution |
|---|---|---|---|---|---|
| Paley(401) | 3 bosons, node 0 | stabiliser of the node (200) | 1.1·10⁷ | 54 139 | 0.04 s, 29 s, 54 s |
| Paley(401) | 3 fermions, nodes 0, 1, 3 | whole (8.0·10⁴) | 1.1·10⁷ | 132 | 0.04 s, 0.1 s, < 0.01 s |
| Paley(1601) | 3 fermions, nodes 0, 1, 3 | whole (1.3·10⁶) | 6.8·10⁸ | 532 | 1.2 s, 1.1 s, 0.1 s |
| Paley(3209) | 3 fermions, nodes 0, 1, 3 | whole (5.2·10⁶) | 5.5·10⁹ | 1068 | 5.1 s, 5.3 s, 1.2 s |
| Paley(101) | 4 fermions, nodes 0, 1, 3, 7 | whole (5050) | 4.1·10⁶ | 833 | < 0.01 s, 0.1 s, 0.5 s |
| circulant(2000; 1, 7, 40) | 3 bosons, node 0 | whole (4000) | 1.3·10⁹ | 334 334 | 0.08 s, 7.9 s, 1.5 s |

A dense graph with a small group stays dense on its orbits (Paley(401)
from one node: 600 hops per orbit, 3·10⁷ entries), and Paley(809) from
one node exceeds the 2²⁶ entries. Where the configurations fit within
the 2²⁷ of `SymmetricInteractingWalk`, its union–find is faster (9.3 s
for Paley(401) from one node). A group with many small stabilisers, such
as a large abelian group, keeps many tables: three bosons on a comb of 100
spine nodes with two leaves each (group 2¹⁰⁰) take 1.5 GB and 32 s, and
the stabilisers' tables are bounded by 2²⁸ node labels (four per node and
group, and one per node for each generator). The group comes from the
automorphism search of `Graph`, which takes seconds for these graphs, and
its order is checked; the other limits are those of
`CoordinateInteractingWalk`.

Certificates need every configuration as a start, but the
automorphisms of the graph map starts onto starts with the same
transition probabilities. `certificate_by_orbits(times, generators)`
computes one row per orbit and repeats it, by Krylov steps unless the
walk was diagonalised already. It gives the same certificate: for two
bosons on the Paley graph of order 61 in 0.22 s instead of 8.1 s, and
beyond the 2048 configurations of `certificate`, up to 8192 (two bosons
on the Paley graph of order 101, 5151 configurations, in 1.3 s).

The checks:

- The reduced walks equal `InteractingWalk` on the Petersen graph, the
  4 × 4 rook's graph, `Q₄`, `C₈` and random weighted graphs closed under
  a permutation, for all three statistics, two and three particles, both
  generators and `U` and `V` both on; under the whole automorphism group
  too, orbit by orbit, with the fermionic signs.
- `CoordinateInteractingWalk` equals the full walk configuration by
  configuration on `Q₄`, `H(3, 3)`, `K₅`, `J(6, 2)`, `K(7, 2)`,
  `J(4, 2)`, `K(4, 2)` and `K₂` for all three statistics and both
  generators with `U` and `V` (starts with repeated and distinct nodes),
  with the orbit counts of `SymmetricInteractingWalk` (finer ones on
  `J(2k, k)`, whose complement 1.43 uses; on `K(2k, k)` only the
  reachable ones). Two bosons
  on `K(40, 20)` stay on their start's pair. Free
  particles follow the permanents and determinants of the walker's
  closed-form amplitudes on `Q₂₀` (three bosons from node 0, the 156 240
  states of `HypercubeParticles`), `H(8, 4)`, `Q₁₀` and `H(6, 4)`.
- `GroupInteractingWalk` equals the full walk configuration by
  configuration (Petersen, Paley(13), a circulant, the 3 × 4 rook's
  graph, Shrikhande, `Q₃`; three statistics, both generators, `U` and
  `V`), the orbits of `SymmetricInteractingWalk` from one configuration
  and under the whole group (orbit states), and the canonical forms of
  `CoordinateInteractingWalk` on `H(3, 3)`, `J(7, 2)` and `K(7, 2)`. Its
  orbit counts equal Burnside's lemma over every element of the group
  (bosons and distinguishable particles; for fermions the signed count,
  the trace of the group on the exterior power). Its minimal images are
  the smallest element of the orbit, and the stabilisers those of the
  group's elements (Paley(29)). Free bosons on Paley(101) follow the
  permanents of the closed-form amplitudes of a strongly regular graph,
  and free fermions on the orbit state of {0, 1, 3} under the whole group
  of Paley(401) the sum of determinants over the orbit with its signs.
- `HypercubeParticles` equals the reduction of the graph `Q₃` and `Q₄` by
  the stabiliser of node 0, for bosons and distinguishable particles (two
  and three) and fermions (two to four), and the full walk on `Q₈` for
  two particles of each kind.
- Free particles from node 0 of `Q_d` follow the single walker: the
  weights are `N·Bin(d, sin²γt)`, two particles differ in a coordinate
  with probability `2q(1 − q)`, and all are at the antipode with
  probability `q^{dN}`, up to perfect transfer (`Q₃₀` and `Q₄₀` with two
  particles, `Q₁₂` with three, `Q₆` with four). Two free fermions follow
  their Slater determinant on `Q₅` and `Q₃₀`.
- Two bosons on a ring of 101 sites with `U` have the bound pair of zero
  momentum at `E = ±√(U² + 16J²)` (Winkler et al. 2006) to 10⁻¹², and
  the rest of their spectrum in the band `[−4J, 4J]`.
- Certificates by orbits equal `certificate` for the Shrikhande, rook,
  Chang and Petersen graphs, with dense and with Krylov rows, and a
  relabelled `Q₆` gives the same certificate beyond the dense limit.

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

Transport to a trap, with the builders of `DephasingWalk`:

```rust
use metatron_qso::graph::generators;
use metatron_qso::prelude::*;
use metatron_qso::walks::TransportOptions;

fn main() -> Result<()> {
    let options = TransportOptions::default();
    let dimer = DephasingWalk::new(&generators::path(2), Generator::Adjacency, 1.0, 0.0)?
        .with_site_energies(&[3.0, 0.0])?
        .with_traps(&[(1, 1.0)])?
        .with_losses(&[0.05; 2])?;
    let scan = dimer.dephasing_scan(0, &[0.0, 1.0, 3.0, 10.0, 100.0], &options)?;
    assert!((scan[1].efficiency - 32000.0 / 38881.0).abs() < 1e-12);
    let tau = scan[1].arrival_time.expect("resolved");
    assert!((tau / (139260.0 / 38881.0) - 1.0).abs() < 1e-10);
    let best = dimer.optimal_dephasing(0, 0.01, 100.0, &options)?;
    assert!(best.interior && (best.rate / 1.95 - 1.0).abs() < 1e-6);
    assert!((best.result.efficiency - 800.0 / 963.0).abs() < 1e-12);
    Ok(())
}
```
