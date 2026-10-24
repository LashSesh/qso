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
