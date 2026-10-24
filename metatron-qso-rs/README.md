# metatron-qso-rs

Validated quantum simulation in pure Rust — the core library of
[QSO 2.0](https://github.com/LashSesh/qso).

* **Statevector simulator** with multi-threaded kernels, arbitrary controls,
  custom unitaries, measurement and entanglement entropy (up to 34 qubits).
* **Circuits** with differentiable symbolic parameters and OpenQASM 2.0
  import/export.
* **Pauli-sum observables** with matrix-free expectation values and Lanczos
  ground states.
* **Density matrices and noise** (depolarising, damping, thermal relaxation,
  readout error).
* **Adjoint and parameter-shift gradients**; L-BFGS, Adam, Nelder–Mead, SPSA.
* **Algorithms**: VQE, QAOA-MaxCut, Grover, QFT, phase estimation,
  variational classifier.
* **Graphs and quantum walks**: Platonic solids, Metatron's Cube and the
  Fruit of Life, exact automorphism groups, CTQW (dense and Krylov), spatial
  search, dephasing.

Every algorithm is tested against closed-form or literature results (QAOA
ring ratios (2p+1)/(2p+2), H₂ full-CI energy, Grover's sine law, QFT = DFT,
exact phase estimation, automorphism group orders, …).

```rust
use metatron_qso::prelude::*;

fn main() -> Result<()> {
    let qaoa = MaxCutQaoa::new(&generators::cycle(10), 1)?;
    let result = qaoa.optimize(&QaoaConfig::default())?;
    assert!((result.approximation_ratio.unwrap() - 0.75).abs() < 1e-6);
    Ok(())
}
```

Features: `parallel` (default, rayon), `cli` (the `qso` binary).

Conventions: qubit 0 is the least significant bit; `exp(−iHt)`;
`R_P(θ) = exp(−iθP/2)`. All randomness is seeded and reproducible.

License: Apache-2.0.
