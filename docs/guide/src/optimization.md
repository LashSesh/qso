# Optimisation problems and amplitude estimation

## QUBO and Ising problems

A `Qubo` asks for the minimum of `f(x) = Σ Q_ij x_i x_j + c` over bit
strings `x`. The equivalent Ising problem uses spins `s_i = 1 − 2x_i`.
`to_pauli_sum()` turns either into the diagonal cost Hamiltonian
`Σ h_i Z_i + Σ J_ij Z_i Z_j + c`, in which bit `x_i = 1` is the qubit state
|1⟩. `brute_force()` returns the exact minimum for up to 30 variables. The
tests check that QUBO, Ising and Hamiltonian give the same value on every
bit string.

## QAOA with problem-specific mixers

`Qaoa(cost, n_qubits, p, mixer=…)` builds a QAOA circuit for any diagonal
cost and minimises `⟨C⟩` with exact gradients. The mixer decides where the
search starts and which states it can reach:

| `mixer` | initial state | mixing step |
|---|---|---|
| `"x"` | `|+⟩ⁿ` | `RX(2β)` on every qubit |
| `"xy_ring"`, `"xy_complete"` | Dicke state of Hamming weight `weight` | `exp(−iβ(XX + YY)/2)` on the pairs of a ring or of all qubits |
| `"warm_start"` | `⊗ RY(2 arcsin √c_i)|0⟩` | `RY(θ_i) RZ(−2β) RY(−θ_i)` |

Every XY factor conserves the Hamming weight. The state therefore never
leaves the strings with exactly `weight` ones, and constraints such as
"choose k of n" need no penalty terms (Hadfield et al., Algorithms 12, 34
(2019)). The warm start (Egger, Mareček and Woerner, Quantum 5, 479 (2021))
begins at a relaxed or heuristic solution `c ∈ [0, 1]ⁿ`. Values are
clipped to `[ε, 1 − ε]` so that the search can still move. The result
reports the exact optimum over the feasible strings, the probability of
measuring it, and the approximation ratio
`(worst − ⟨C⟩)/(worst − optimum)`.

```python
import numpy as np
import metatron_qso as q

rng = np.random.default_rng(1)
qubo = q.Qubo(rng.uniform(-1, 1, (6, 6)))
cost = qubo.to_pauli_sum()
choose3 = q.Qaoa(cost, 6, 2, mixer="xy_ring", weight=3).optimize(seed=1)
assert abs(choose3["feasible_probability"] - 1) < 1e-12
best, strings = qubo.brute_force()
guess = [(strings[0] >> i) & 1 for i in range(6)]
warm = q.Qaoa(cost, 6, 1, mixer="warm_start", warm_start=guess).optimize()
plain = q.Qaoa(cost, 6, 1).optimize()
print(choose3["approximation_ratio"], warm["optimal_probability"], plain["optimal_probability"])
```

## Amplitude estimation

A state-preparation circuit `A` and an objective qubit define the
amplitude `a = P(objective = 1)` of `A|0⟩`, with `a = sin²θ`. The Grover
operator `Q` rotates by `2θ`, so a measurement of `Q^k A|0⟩` gives 1 with
probability `sin²((2k + 1)θ)`. QSO counts oracle calls as applications of
`A` or `A†`. One shot of `Q^k A|0⟩` therefore costs `2k + 1` of them.

| function | method | output |
|---|---|---|
| `canonical_amplitude_estimation` | phase estimation of `Q` with `m` qubits | exact outcome distribution of `sin²(πy/2^m)` |
| `iterative_amplitude_estimation` | Grinko et al., npj Quantum Inf. 7, 52 (2021) | interval of half-width ε at confidence 1 − α |
| `ml_amplitude_estimation` | Suzuki et al., Quantum Inf. Process. 19, 75 (2020), powers 0, 1, 2, 4, … | maximum-likelihood estimate, Fisher interval |

The error of amplitude estimation falls as `1/N` in the number of oracle
calls `N`. Plain sampling only achieves `1/√N`. The tests fit
log(RMSE) against log(oracle calls) over 200 seeds: the slope is −0.99 for
maximum-likelihood estimation and −0.49 for sampling with `k = 0` only.
They also check the error bound of canonical estimation (met with
probability at least 8/π²) and the coverage of the iterative intervals
(1 miss in 200 runs at α = 0.05).

```python
import metatron_qso as q

problem = q.AmplitudeProblem.bernoulli(0.3)       # or AmplitudeProblem(circuit, objective)
print(q.canonical_amplitude_estimation(problem, 6)["estimate"])
iae = q.iterative_amplitude_estimation(problem, epsilon=0.001, alpha=0.05, seed=1)
mle = q.ml_amplitude_estimation(problem, levels=8, shots=100, seed=1)
print(iae["interval"], iae["oracle_calls"], mle["estimate"], mle["oracle_calls"])
```
