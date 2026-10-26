# Matrix-product states

A state vector of n qubits has 2ⁿ amplitudes and stops at 34 qubits. Many
interesting circuits never use most of that space. They create little
entanglement across any cut of a chain of qubits:

- time evolution of spin chains for moderate times;
- shallow QAOA on rings and other sparse graphs;
- many walk and Hamiltonian-simulation circuits.

A matrix-product state (MPS) stores such a state as n small tensors
`A[q]` of shape `χ_l × 2 × χ_r`. An amplitude is the product of one matrix
per qubit. The bond dimension χ at a cut grows with the entanglement
across it, not with n: a GHZ state of any size has χ = 2, and k ebits
across a cut need χ ≥ 2ᵏ there (Vidal, Phys. Rev. Lett. 91, 147902
(2003); Schollwöck, Ann. Phys. 326, 96 (2011)).

`MatrixProductState` simulates any circuit of the gate set on up to 65 536
qubits:

```python
import metatron_qso as q

n = 1000
m = q.MatrixProductState(n).h(0)
for k in range(1, n):
    m.cx(k - 1, k)                      # GHZ on 1000 qubits
assert m.max_bond_dimension == 2
assert abs(m.expectation("Z0 Z999") - 1.0) < 1e-12
assert all(abs(s - 1.0) < 1e-12 for s in m.bond_entropies())
counts = m.sample(100, seed=1)
assert set(counts) <= {"0" * n, "1" * n}
```

## Gates and truncation

QSO keeps qubit q at site q, and the state in mixed canonical form around
an orthogonality centre. A gate on several qubits goes through three steps:

1. **Route.** Swaps move the qubits to adjacent sites.
2. **Apply.** The gate is applied to those sites contracted with the
   centre.
3. **Split.** The result is split again into one tensor per site, and the
   qubits are swapped back.

Consecutive gates on the same one or two qubits are multiplied into one
matrix first. Controls count as qubits, and a gate may act on up to 8
qubits.

A split keeps the Schmidt decomposition across each new bond. Two options
decide how much of it survives:

| Option | Meaning | Default |
|---|---|---|
| `max_bond` | most Schmidt values kept at a bond | 256 |
| `cutoff` | largest total weight (fraction of the norm²) of the smallest Schmidt values a split may drop | 10⁻¹⁴ |

QSO computes the split from the Hermitian eigendecomposition of a Gram
matrix, followed by a Householder QR. It does not use nalgebra's complex
SVD, which reconstructs rank-deficient matrices with errors of several
percent; rank-deficient matrices are the normal case in an MPS.

## The error is bounded

Every truncation is measured exactly. If a split discards the fraction ε
of the norm², the normalised state moves by `√(2 − 2√(1 − ε))` in 2-norm.
Gates preserve distances, so the final state is within the sum δ of
these distances of the exact one, global phase included. Its fidelity
satisfies

  F = |⟨ψ_exact|ψ⟩|² ≥ (1 − δ²/2)²,

which `fidelity_lower_bound` reports. A measurement after a truncation
makes the "exact" state depend on the outcome, so the bound is then
`None`.

```python
import metatron_qso as q

n = 14                                   # an XX chain, small enough to check
c = q.Circuit(n)
for j in range(0, n, 2):
    c.x(j)                               # Néel state
for _ in range(5):                       # five Trotter steps
    for start in (0, 1):
        for j in range(start, n - 1, 2):
            c.rxx(j, j + 1, 0.25).ryy(j, j + 1, 0.25)
exact = q.simulate(c)
for max_bond in (6, 8, 12):
    m = q.MatrixProductState.from_circuit(c, max_bond=max_bond)
    f = m.to_statevector().fidelity(exact)
    print(max_bond, m.discarded_weight, m.fidelity_lower_bound, f)
    assert f >= m.fidelity_lower_bound - 1e-12
```

| `max_bond` | discarded weight | `fidelity_lower_bound` | true fidelity |
|---|---|---|---|
| 6 | 4.0·10⁻³ | 0.949 | 0.9941 |
| 8 | 7.5·10⁻⁵ | 0.99937 | 0.99988 |
| 12 | 3.9·10⁻⁷ | 0.9999975 | 0.9999995 |

The bound is conservative, since it adds the distances of all
truncations. It holds up to floating-point rounding: the weight a split
discards is computed from its residual, accurate to about 10⁻³⁰, and
splits that keep everything add nothing. Exact circuits therefore keep a
bound of 1 even after many routing swaps; discarded weights up to 10⁻²⁰
count as rounding, not as truncations. In a program with mid-circuit
measurements, the bound is `None` as soon as one shot measured after a
truncation.

## Readout

| Method | Result | Cost |
|---|---|---|
| `expectation("Z0 Z99")`, a `PauliSum` (qubits below 64), or a list of `(coefficient, "string")` terms | ⟨O⟩ | χ³ per site between the first and last factor (the centre moves there) |
| `sample(shots, seed)` | `{bitstring: count}`, exact sampling site by site | χ² per site and shot |
| `measure(q)`, `reset(q)` | collapse, with the Born probabilities | one move of the centre |
| `amplitude("0101…")` | ⟨b\|ψ⟩, global phase included | χ² per site |
| `bond_entropies()` | entanglement entropy in bits between qubits 0…b and the rest, for every b | χ³ per site |
| `to_statevector()` | the state vector, up to 24 qubits | 2ⁿ |

OpenQASM programs run on an MPS with `Program.sample(method="mps",
max_bond=…)` and `qso qasm FILE --mps --max-bond N`. Programs whose
measurements all come at the end are sampled once from the final state;
any other program is simulated shot by shot.

## Two exact checks far beyond 34 qubits

**Free fermions.** The gate `exp(−iθ(XX + YY)/2)` on neighbours j, j + 1
hops a Jordan–Wigner fermion between modes j and j + 1. A Trotterized XX
chain is therefore a free-fermion circuit: its single-particle unitary U
is the product of 2 × 2 rotations. From the Néel state,
⟨Z_j⟩ = 1 − 2 Σ_{k occupied} |U_jk|², and Wick's theorem gives ⟨Z_i Z_j⟩.
These values are exact for the circuit itself, Trotter error included.
On 100 qubits after 6 Trotter steps, the MPS agrees to 10⁻⁸ (test
`matrix_product_states_follow_free_fermions_and_qaoa_rings`).

**QAOA on a ring.** On a ring of n ≥ 2p + 2 nodes every edge sees the same
light cone. Angles optimised on a ring of 10 therefore give the proven
optimum (2p + 1)/(2p + 2) (Mbeng, Fazio and Santoro 2019) on any larger
ring. On 100 and on 200 nodes, the MPS reproduces 0.75, 0.8333 and 0.875
for p = 1, 2, 3 to 10⁻⁶.

```rust
use metatron_qso::algorithms::ansatz::qaoa_maxcut_circuit;
use metatron_qso::algorithms::qaoa::{QaoaConfig, optimize_layerwise};
use metatron_qso::graph::generators;
use metatron_qso::mps::{Mps, MpsOptions};

fn main() -> metatron_qso::Result<()> {
    let ring = generators::cycle(100);
    for r in optimize_layerwise(&generators::cycle(10), 2, &QaoaConfig::default())? {
        let theta: Vec<f64> = r.gammas.iter().chain(&r.betas).copied().collect();
        let circuit = qaoa_maxcut_circuit(&ring, r.p)?;
        let mut state = Mps::from_circuit(&circuit, &theta, MpsOptions::default())?;
        let mut cut = 0.0;
        for &(a, b, _) in ring.edges() {
            cut += (1.0 - state.expectation_str(&format!("Z{a} Z{b}"))?) / 2.0;
        }
        println!("p = {}: cut fraction {:.6}", r.p, cut / 100.0);
    }
    Ok(())
}
```

`cargo run --release --example mps` runs both checks on 200 qubits and
prints how the entanglement of the XX chain grows.

## Limits and costs

| Quantity | Limit |
|---|---|
| Qubits | 65 536 |
| `max_bond` | 4096 |
| Tensor entries in all | 2²⁸ (4 GiB) |
| Qubits per gate (controls included) | 8 |
| Sampling | shots × tensor entries ≤ 2³⁶; shots × qubits ≤ 2²⁸ returned bits |

- **Two-qubit gates.** A gate between neighbours costs O(χ³). On one core
  that is about 15 ms at χ = 64, 0.13 s at χ = 128 and 0.9 s at χ = 256.
- **Distant gates.** A gate between distant qubits costs one swap per site
  in between, twice. A ring edge from qubit n − 1 to qubit 0 crosses the
  whole chain.
- **Large gates.** A gate on k qubits contracts a tensor of
  χ² · 2ᵏ entries and splits it k − 1 times; at large χ and k one gate
  takes minutes. Nothing bounds this time beyond `max_bond` and the
  limit of 8 qubits per gate.
- **Errors.** A gate that would exceed the limits fails before it changes
  the state. In `apply_circuit`, every instruction is checked first, so
  invalid input changes nothing; a limit reached part of the way leaves
  the gates before it applied.
- **When not to use an MPS.** Circuits that entangle many qubits across a
  cut (deep random circuits, the QFT on a superposition) need χ = 2^(n/2)
  and belong on the state vector.
