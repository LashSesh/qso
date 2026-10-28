# Parameters and gradients

## Symbolic parameters

`new_parameter` returns a parameter `p[k]` of the circuit. A gate angle can
be any affine expression `a·p[k] + b`. The parameter values `theta` are
passed when the circuit is simulated, so one circuit serves every evaluation:

```python
import metatron_qso as q

c = q.Circuit(2)
a, b = c.new_parameters(2)
c.ry(0, a).cx(0, 1).rz(1, 2.0 * b + 0.5)
print(c.num_parameters)                       # 2
state = q.simulate(c, [0.3, 0.1])
fixed = c.bind([0.3, 0.1])                    # the same circuit with numbers
```

## Gradients

`expectation_and_gradient` returns ⟨ψ(θ)|H|ψ(θ)⟩ and its gradient with
respect to all parameters:

```python
import numpy as np
import metatron_qso as q

c = q.Circuit(2)
a, b = c.new_parameters(2)
c.ry(0, a).cx(0, 1).rz(1, 2.0 * b + 0.5).rx(0, b)
h = q.PauliSum("0.5*Z0 Z1 - X0 + 0.2 Y1")
theta = [0.3, 0.1]

value, adjoint = q.expectation_and_gradient(c, h, theta)                    # default
_, shift = q.expectation_and_gradient(c, h, theta, method="parameter_shift")
_, fd = q.expectation_and_gradient(c, h, theta, method="finite_difference")
assert np.allclose(adjoint, shift, atol=1e-12)
assert np.allclose(adjoint, fd, atol=1e-6)
```

| Method | Cost | Accuracy | Notes |
|---|---|---|---|
| `adjoint` (default) | about three circuit runs, independent of the number of parameters | exact | Simulation only |
| `parameter_shift` | two (four for controlled rotations) runs per parameter use | exact | What hardware can do |
| `finite_difference` | two runs per parameter | O(step²) | For checks |

With 100 parameters on 10 qubits, the adjoint method is about 50 times
faster than parameter shift ([BENCHMARKS.md](https://github.com/LashSesh/qso/blob/main/docs/BENCHMARKS.md)).
Shared parameters and affine angles are handled by the chain rule: in the
example, `b` enters two gates with factors 2 and 1.

### Single precision

With `precision="single"`, the state and the co-state of the adjoint pass
are single-precision vectors: half the memory (`run_vqe` keeps a given
`initial_state` in double precision as well). Inner products, matrix
elements and expectation values are summed in double precision, energy
and gradient are divided by the state's squared norm, which rounding
moves away from one, and the observable's constant term is added to the
energy instead of entering the co-state. `run_vqe`, `MaxCutQaoa` and
`optimize_qaoa_layerwise` take the same argument, and `qso vqe` and
`qso qaoa` take `--single`:

```python
import numpy as np
import metatron_qso as q

h = q.transverse_field_ising(12, 1.0, 0.7)
c = q.efficient_su2(12, 2)
theta = np.random.default_rng(5).uniform(-np.pi, np.pi, c.num_parameters)
e, g = q.expectation_and_gradient(c, h, theta)
e32, g32 = q.expectation_and_gradient(c, h, theta, precision="single")
print(abs(e32 - e) / abs(e), np.abs(g32 - g).max() / np.abs(g).max())

qaoa = q.MaxCutQaoa(q.Graph.cycle(10), 3, precision="single")
print(qaoa.optimize(restarts=2)["approximation_ratio"])
```

| Measured (one call, relative error) | 20 qubits | 22 qubits | 24 qubits |
|---|---:|---:|---:|
| VQE energy (EfficientSU2, 4 layers, Ising chain) | 3.1·10⁻⁸ | 2.7·10⁻⁸ | 8.0·10⁻⁹ |
| VQE gradient | 6.0·10⁻⁷ | 7.2·10⁻⁷ | 6.7·10⁻⁷ |
| QAOA energy (`MaxCutQaoa`, p = 10, 3-regular) | 3.2·10⁻⁹ | 1.9·10⁻⁹ | 3.0·10⁻⁹ |
| QAOA gradient | 9.4·10⁻⁷ | 1.0·10⁻⁶ | 1.1·10⁻⁶ |

At 28 qubits an energy and gradient takes 4.0 GiB instead of 8.0 GiB
(VQE) and 4.5 GiB instead of 8.5 GiB (`MaxCutQaoa`), and a fifth to a third less
time on a 4-core VM ([ROADMAP](https://github.com/LashSesh/qso/blob/main/docs/ROADMAP.md), 1.27). The
gradient error grows with the depth of the circuit: 1.7·10⁻⁴ for a UCCSD
ansatz of 16 794 gates (LiH, 10 qubits), whose VQE still ends 2·10⁻⁸
from double precision (with 69 evaluations instead of 27).

The optimisers know that single-precision energies are only good to about
10⁻⁷ of their value: L-BFGS treats energies within that noise as equal and
lets the slope decide (Hager and Zhang's approximate Wolfe condition),
and L-BFGS and Adam stop when the gradient reaches its own rounding. QAOA
then reaches the same approximation ratios as double precision to 10⁻⁸,
with up to 1.4 times the evaluations. VQE on rugged landscapes may end in
a different local minimum, because rounding perturbs every step: in 12
runs from the same starts, 6 ended where double precision did (to
3·10⁻⁸), 2 lower and 4 higher, by 6·10⁻⁵ to 4·10⁻² (double precision
started 10⁻⁷ away does so in 2 of the 12); restarts take the best. A
reported single-precision energy carries its rounding and can lie a few
10⁻⁸ below an exact bound. Only the adjoint method is available in single
precision: parameter shift and finite differences subtract nearby
energies and would lose the digits single precision has.

## Optimisers

`run_vqe` and `MaxCutQaoa.optimize` accept `method=` with one of:

- `lbfgs` (default; L-BFGS with a strong-Wolfe line search);
- `adam`;
- `gradient_descent`;
- `nelder_mead`;
- `spsa`.

In Rust, `optimize::minimize` works with any objective that implements the
`Objective` trait:

```rust
use metatron_qso::optimize::{Method, Objective, Options, minimize};
use metatron_qso::prelude::*;

/// Rosenbrock's function with its gradient.
struct Rosenbrock;

impl Objective for Rosenbrock {
    fn dim(&self) -> usize {
        2
    }
    fn value(&self, x: &[f64]) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
    }
    fn value_and_gradient(&self, x: &[f64]) -> Option<(f64, Vec<f64>)> {
        let g = vec![
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]),
            200.0 * (x[1] - x[0] * x[0]),
        ];
        Some((self.value(x), g))
    }
}

fn main() -> Result<()> {
    let r = minimize(&Rosenbrock, &[-1.2, 1.0], &Method::lbfgs(), &Options::default())?;
    assert!((r.x[0] - 1.0).abs() < 1e-6 && (r.x[1] - 1.0).abs() < 1e-6);
    println!("{} iterations", r.iterations);
    Ok(())
}
```
