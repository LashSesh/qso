# Excited states and state geometry

## Excited states

Two methods find the lowest few eigenstates of `H`.

- **VQD** (variational quantum deflation; Higgott, Wang and Brierley,
  Quantum 3, 156 (2019)) finds the states one after another. State `k`
  minimises `⟨H⟩ + Σ_{j<k} β_j |⟨ψ_j|ψ⟩|²`. Each penalty `β_j` must exceed
  the gap `E_k − E_j`. Every gap is at most twice the coefficient
  one-norm of `H`; the default, three times that norm, always exceeds it.
- **SSVQE** (subspace-search VQE; Nakanishi, Mitarai and Fujii, Phys. Rev.
  Research 1, 033062 (2019)) applies one circuit to `|0⟩, |1⟩, …` and
  minimises a weighted sum of their energies. The weights must decrease
  strictly; with equal weights the result is only the subspace sum.
  It needs a more expressive circuit than VQD, because one unitary must map
  all inputs to eigenstates.

Both methods use exact adjoint gradients and seeded restarts:

```python
import numpy as np
import metatron_qso as q

h = q.h2_sto3g()["hamiltonian"]                       # two qubits
exact = np.linalg.eigvalsh(h.to_matrix(2))
v = q.vqd(h, q.efficient_su2(2, 2), 4, restarts=6, seed=2)
s = q.ssvqe(h, q.efficient_su2(2, 2), 4, restarts=6, seed=2)
assert np.allclose(v["energies"], exact, atol=1e-6)
assert np.allclose(s["energies"], exact, atol=1e-6)
```

The tests check both methods against exact diagonalisation: all four
levels of H₂ and the three lowest levels of a three-qubit Ising chain.

## Quantum Fisher information

For `|ψ(θ)⟩ = U(θ)|ψ₀⟩`, the quantum Fisher information is

```text
F_ij = 4 Re(⟨∂_iψ|∂_jψ⟩ − ⟨∂_iψ|ψ⟩⟨ψ|∂_jψ⟩).
```

It is four times the Fubini–Study metric, which measures how fast the
state moves when the parameters change. `quantum_fisher_information`
computes it exactly. One forward pass through the circuit carries one
derivative vector per parameter. The tests compare `F` with finite
differences of the fidelity `|⟨ψ(θ)|ψ(θ + δ)⟩|²`.

## Natural gradient and imaginary time

- `quantum_natural_gradient` (Stokes, Izaac, Killoran and Carleo, Quantum
  4, 269 (2020)) steps along `(F/4 + λI)⁻¹ ∇E`. This is the steepest
  descent in the geometry of states, not of parameters. On H₂ with
  `real_amplitudes(2, 2)` and step size 0.2, it converges in 64 steps,
  where plain gradient descent needs 151.
- `imaginary_time_evolution` (McArdle et al., npj Quantum Inf. 5, 75
  (2019)) integrates McLachlan's equation `θ̇ = −2F⁺∇E` with fourth-order
  Runge–Kutta. Redundant parameters make `F` singular, so `F⁺` is a
  pseudo-inverse. When the ansatz can represent the normalised state
  `e^{−Hτ}|ψ₀⟩`, the method follows it: the tests reproduce the exact
  energy curve to 10⁻⁶.

```python
import numpy as np
import metatron_qso as q

h = q.PauliSum("Z0 + 0.6 X0 X1 - 0.4 Z1 + 0.3 X1")
ansatz = q.real_amplitudes(2, 2)
theta0 = np.full(ansatz.num_parameters, 0.2)
ite = q.imaginary_time_evolution(ansatz, h, theta0, time=3.0, steps=100)
qng = q.quantum_natural_gradient(ansatz, h, theta0, learning_rate=0.2)
print(ite["energies"][-1], qng["energy"], h.ground_state_energy())
```
