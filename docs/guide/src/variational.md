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
