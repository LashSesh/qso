# Variational algorithms

## VQE

`run_vqe` minimises ⟨ψ(θ)|H|ψ(θ)⟩ over the parameters of an ansatz. Each run
starts from several seeded random points, which are evaluated in parallel,
and uses exact gradients:

```python
import metatron_qso as q

h2 = q.h2_sto3g()                                   # 2 qubits, parity mapping
result = q.run_vqe(h2["hamiltonian"], q.real_amplitudes(2, 1), restarts=4)
total = result["energy"] + h2["nuclear_repulsion"]
assert abs(total - (-1.137306)) < 1e-6              # full CI of the basis
```

Three ansatz families are available:

- `real_amplitudes`: RY layers with CX entanglers.
- `efficient_su2`: RY and RZ layers with CX entanglers.
- `hardware_efficient`: choose the rotations and the entanglement pattern
  (`linear`, `circular` or `full`).

Any parametrised `Circuit` works as an ansatz. For larger models, compare
with the exact ground energy from Lanczos:

```python
import metatron_qso as q

h = q.transverse_field_ising(6, 1.0, 0.8)
exact = h.ground_state_energy()
vqe = q.run_vqe(h, q.efficient_su2(6, 4), restarts=4, seed=1)
print(f"VQE {vqe['energy']:.6f}, exact {exact:.6f}")
```

## QAOA for MaxCut

`MaxCutQaoa` simulates QAOA for weighted MaxCut in a diagonal basis, which is
much faster than a generic circuit. It also finds the exact optimum by
enumeration, so every result comes with its approximation ratio:

```python
import metatron_qso as q

ring = q.MaxCutQaoa(q.Graph.cycle(10), p=1).optimize(restarts=8, seed=0)
assert abs(ring["approximation_ratio"] - 0.75) < 1e-9   # 3/4 on rings at p = 1
print(ring["most_likely_bitstring"])

for r in q.optimize_qaoa_layerwise(q.Graph.metatron_cube(), 3):
    print(r["p"], round(r["approximation_ratio"], 5))
```

`optimize_qaoa_layerwise` seeds each depth with the INTERP extrapolation of
the previous optimum (Zhou et al. 2020), so the ratio improves monotonically.
The validated reference values are listed in the technical note:

| Graph | Depth | QSO ratio | Reference |
|---|---|---|---|
| Ring | p = 1 … 4 | 0.75, 0.8333, 0.875, 0.9 | (2p+1)/(2p+2) |
| 3-regular, triangle-free | 1 | 0.692450 | 0.6925 |
| Heawood graph | 2 | 0.7559 | 0.7559 |

### Large instances

`MaxCutQaoa` stores the cut of every bit string and one state vector (two
for gradients), so the memory is the limit: 26 qubits take 2.2 GiB in
double precision and 1.2 GiB with `precision="single"`, 28 qubits 8.5 and
4.5 GiB. On a 4-core VM an energy and gradient at p = 10 takes 16 s at 26
qubits and a minute at 28 (single precision). Optimising from random
starts at that size takes hours; QAOA angles of random 3-regular graphs
concentrate, so angles optimised on a small graph are a good start on a
large one:

```python
import metatron_qso as q

small = q.Graph.random_regular(12, 3, seed=1)
best = q.optimize_qaoa_layerwise(small, 4)[-1]
angles = list(best["gammas"]) + list(best["betas"])

large = q.MaxCutQaoa(q.Graph.random_regular(18, 3, seed=1), 4, precision="single")
print(large.analyze(angles)["approximation_ratio"])             # carried over
print(large.optimize(restarts=1, initial=angles, max_iterations=20)["approximation_ratio"])
```

`examples/qaoa_large.rs` does this at depth 10 on 28 qubits: angles from 16
nodes give the ratio 0.962 at once and 0.966 after 30 L-BFGS iterations,
in 37 minutes and 4.5 GiB ([ROADMAP](https://github.com/LashSesh/qso/blob/main/docs/ROADMAP.md), 1.28).

## Classifiers

`VariationalClassifier` is a data re-uploading classifier for points in the
plane. `GraphClassifier` classifies graphs. Each graph defines its own cut
Hamiltonian, the model applies the same trainable QAOA layers to every
graph, and the normalised expected cut is the feature:

```python
import metatron_qso as q

def cubic(n, count, seed, want_triangle):
    graphs, s = [], seed
    while len(graphs) < count:
        g = q.Graph.random_regular(n, 3, seed=s)
        s += 1
        if (g.triangle_count() > 0) == want_triangle:
            graphs.append(g)
    return graphs

train = cubic(12, 10, 1, True) + cubic(12, 10, 500, False)
labels = [1] * 10 + [0] * 10
model = q.GraphClassifier(1)
report = model.fit(train, labels, restarts=3, seed=0)
print(report["accuracy"])          # one layer sees triangles: 1.0
```

After p layers, the model sees each edge's neighbourhood up to distance p,
which is the QAOA light cone. Depth therefore decides which structures the
model can detect: one layer sees triangles, two layers see 4- and 5-cycles.
Section 3.6 of the technical note quantifies this.

## In Rust

```rust
use metatron_qso::prelude::*;

fn main() -> Result<()> {
    let h2 = models::h2_sto3g();
    let ansatz = ansatz::real_amplitudes(2, 1, Entanglement::Linear);
    let vqe = Vqe::new(h2.hamiltonian.clone(), ansatz)?.with_restarts(4).run()?;
    assert!((vqe.energy + h2.nuclear_repulsion + 1.137306).abs() < 1e-6);

    let qaoa = MaxCutQaoa::new(&generators::petersen(), 2)?;
    let r = qaoa.optimize(&QaoaConfig::default())?;
    println!("Petersen, p = 2: ratio {:.4}", r.approximation_ratio.unwrap());
    Ok(())
}
```
