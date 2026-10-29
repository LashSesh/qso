# Hamiltonian simulation

Hamiltonian simulation builds a circuit for `exp(−iHt)`, with
`H = Σ_γ h_γ P_γ` a sum of Pauli strings. QSO does this two ways:
product formulas, which come with rigorous error bounds, and qDRIFT, a
randomised method. Both use `Circuit.pauli_rotation`, which appends
`exp(−iθP/2)` for any Pauli string `P`. One-qubit strings and `XX`, `YY`,
`ZZ` become a single gate. Longer strings become a basis change, a CNOT
ladder and one `rz`.

## Product formulas

`trotter_circuit(h, n_qubits, t, steps, order)` splits the time into
`steps` steps of length `τ = t/steps`:

| order | one step |
|---|---|
| 1 (Lie–Trotter) | `exp(−iτ h_γ P_γ)` for γ = 1, …, Γ |
| 2 (Strang) | half steps for γ = 1, …, Γ and back again |
| 2k ≥ 4 (Suzuki) | `S₂ₖ(τ) = S₂ₖ₋₂(uτ)² S₂ₖ₋₂((1 − 4u)τ) S₂ₖ₋₂(uτ)²`, `u = 1/(4 − 4^{1/(2k−1)})` |

`trotter_error_bound` returns an upper bound on the spectral-norm distance
between the circuit and `exp(−iHt)`: the smallest of three bounds, which
`trotter_error_bounds` returns one by one. The commutators in them are
computed exactly in the Pauli algebra.

- **`nested`**, for every order p. One step `S(τ)` is a product of
  exponentials, and `S' = (−iH + T) S` defines its error generator `T`;
  then `‖S(τ) − exp(−iHτ)‖ ≤ ∫₀^τ ‖T‖` (the error representation of
  Childs, Su, Tran, Wiebe and Zhu, Phys. Rev. X 11, 011020 (2021)). The
  Taylor coefficients of `T` vanish below `τ^p`. From `τ^p` on they are
  nested commutators of order p + 1 and higher, computed as Pauli
  operators up to a degree that grows until the bound on the rest is
  below 1 % of the computed part. The rest is bounded by the multinomial
  theorem. Large Hamiltonians and few steps can put this bound out of
  reach (it is then infinite); a fixed budget keeps its cost below about
  a quarter of a second.
- **`stages`**: order 1, `τ²/2 · Σ_γ ‖[Σ_{γ'>γ} H_γ', H_γ]‖` (Proposition
  15 there); order 2, the nested-commutator bound of their Proposition 16;
  order 4 and higher, the order-2 bound summed over the second-order
  stages, of third order only.
- **`taylor`**: the Taylor remainder of a formula exact to order p.

Each bound is multiplied by the number of steps.

```python
import numpy as np
import metatron_qso as q

h = q.heisenberg_xxz(4, 1.0, 0.5)
w, v = np.linalg.eigh(h.to_matrix(4))
exact = (v * np.exp(-1j * w)) @ v.conj().T          # exp(-iH), t = 1

def unitary(c):
    cols = []
    for k in range(16):
        prep = q.Circuit(4)
        for b in range(4):
            if k >> b & 1:
                prep.x(b)
        cols.append(q.simulate(prep.append(c)).amplitudes)
    return np.array(cols).T

for order in (1, 2, 4):
    c = q.trotter_circuit(h, 4, 1.0, steps=10, order=order)
    error = np.linalg.norm(unitary(c) - exact, 2)
    bound = q.trotter_error_bound(h, 1.0, steps=10, order=order)
    assert error <= bound
    print(order, len(c), f"{error:.2e} <= {bound:.2e}")
```

The identity term of `H` only adds a global phase, so circuits leave it
out. In the tests (Heisenberg, Ising and random four-qubit Hamiltonians),
the first- and second-order bounds are 3 to 13 times the actual error.
From 8 to 32 steps the nested bound is 7 to 11 times the actual error at
order 4 and 8 to 24 times at order 6, and it falls as `r^{−p}` like the
error. The stage-by-stage bound of these orders falls only as `r⁻²` and
is 800 to 10⁹ times the error. Doubling the steps divides the actual
error by `2^order`, and the tests check this.

## qDRIFT

`qdrift_circuit(h, n_qubits, t, samples, seed)` draws `N = samples` terms
at random, with probability `|h_γ|/λ` (`λ = Σ|h_γ|`). Each drawn term is
applied for time `λt/N` (Campbell, Phys. Rev. Lett. 123, 070503 (2019)).
The average over seeds is a channel within diamond distance
`qdrift_error_bound = 2λ²t²/N · e^{2λt/N}` of the exact evolution. The tests
compute that channel exactly. Its error falls as `1/N` and stays below the
bound.

```python
import metatron_qso as q

h = q.PauliSum("0.8 X0 X1 + 0.5 Z1 + 0.3 Y1 Y2 - 0.6 Z0 Z2")
c = q.qdrift_circuit(h, 3, 1.0, samples=200, seed=7)
print(len(c), q.qdrift_error_bound(h, 1.0, 200))
```

qDRIFT pays off when `H` has many small terms, because its cost depends
on `λ` and not on the number of terms.

## On the command line

```text
$ qso trotter --hamiltonian "Z0 Z1 + 0.5 X0 + 0.5 X1" --time 1 --steps 10 --order 2
order 2, 10 steps, t = 1: 41 gates, depth 31, error bound 5.000e-3, largest column error 1.938e-3 (a lower bound on ‖U − exp(−iHt)‖)
```

`--output FILE` writes the circuit as OpenQASM 2.0.
