# Architecture

This document explains how QSO 2.0 is built and why. API details live in
the rustdoc (`cargo doc -p metatron-qso-rs --open`).

## Layers

```
            ┌─────────────────────── qso CLI ─── Python (PyO3) ─┐
algorithms  │ VQE · QAOA · Grover · QFT/QPE · classifier · models│
            │ gradients (adjoint / parameter shift / FD)         │
optimize    │ L-BFGS · Adam · GD · Nelder–Mead · SPSA            │
walks       │ CTQW · spatial search · dephasing · classical      │
backend     │ statevector / density-matrix backends, shot estimates│
quantum     │ StateVector · Circuit · PauliSum · OpenQASM        │
noise       │ DensityMatrix · Kraus channels                      │
graph       │ Graph · generators · Metatron · automorphisms      │
linalg      │ dense eigh/expm · CSR · Lanczos · Krylov           │
infra       │ error · rng (xoshiro256++) · parallel (rayon)       │
            └────────────────────────────────────────────────────┘
```

Lower layers never depend on higher ones.

## Simulation kernels (`quantum/kernels.rs`)

Amplitude `k` stores basis state `|b_{n−1} … b_0⟩` with `k = Σ b_q 2^q`.
A gate on targets `T` with controls `C` touches groups of `2^|T|` amplitudes
whose indices differ only in the target bits and have all control bits set.
The kernels enumerate those groups by *inserting zero bits*: for
`k ∈ 0 … 2^{n−|T|−|C|}`, insert zeros at the sorted positions of `T ∪ C`,
then OR in the control mask. Distinct `k` give disjoint groups, so the loop
parallelises without locks (`SyncPtr` documents this safety contract).

Specialised paths:

* uncontrolled single-qubit gates iterate over contiguous half-blocks
  (`parallel::for_each_pair`), which the compiler vectorises;
* `X`, `SWAP` and their controlled forms are permutations (no arithmetic);
* `Z, S, T, P, RZ, RZZ` and the QAOA cost layer are diagonal phase
  multiplications.

The kernels never assume unitarity. The same code therefore applies

* derivative matrices `∂U/∂θ` in adjoint differentiation, and
* superoperators in the density-matrix simulator: with
  `vec(ρ)[r + (c ≪ n)] = ρ[r, c]`, `ρ ↦ UρU†` is `U ⊗ U*` acting on a
  2n-qubit vector, i.e. the gate applied to the row qubits plus its complex
  conjugate applied to the column qubits (offset `n`). A one-qubit channel
  `{K_i}` becomes the 4×4 matrix `Σ K_i ⊗ K_i*` on the (row, column) bit pair.

Parallelism starts at 2¹⁶ amplitudes; below that rayon's scheduling
overhead exceeds the work of a gate (measured, see
[BENCHMARKS.md](BENCHMARKS.md)).

## Determinism

* `rng::Rng` is xoshiro256++ seeded through SplitMix64, implemented in the
  crate so that a seed means the same stream on every platform and across
  dependency upgrades. A test pins the first outputs (cross-checked against
  an independent implementation).
* Floating-point reductions (`parallel::sum_f64`, `sum_c64`) sum fixed-size
  chunks and then combine the partial sums in order, so results do not
  depend on the number of threads.
* Multi-start optimisers derive one independent stream per restart with
  `Rng::fork`, so parallel restarts are reproducible.

## Gradients (`algorithms/gradients.rs`)

Adjoint differentiation (Jones & Gacon 2020) keeps three vectors: the state
`φ`, the co-state `λ = H|ψ⟩`, and a scratch copy. Walking the circuit
backwards, each gate is undone on `φ` and `λ`, and every parameter of the
gate contributes `2·Re⟨λ|∂U φ⟩·scale`. The cost is about three circuit
simulations regardless of the parameter count (1.6 ms vs 68 ms for
parameter shift with 100 parameters, see BENCHMARKS.md). Parameter-shift rules are provided for
hardware-faithful gradients and as an independent check: two terms for
`exp(−iθG/2)` with `G² = I` and phase gates, four terms for single-controlled
rotations.

QAOA has its own specialised adjoint pass: the cost layer generator is the
diagonal cut vector and the mixer generator is `Σ X_q`, both applied in
O(n·2ⁿ).

## Krylov methods (`linalg/krylov.rs`)

Operators are closures `y = H·x`, so the same code serves sparse graph
Hamiltonians and Pauli sums on 2ⁿ amplitudes.

* Ground states: explicitly restarted Lanczos with full
  re-orthogonalisation ("twice is enough"), convergence on the Ritz
  residual, verified by an explicit residual at the end.
* Propagation `exp(−iHt)v`: Lanczos basis per sub-step, exact exponential of
  the tridiagonal matrix, step-size control from the a-posteriori error
  estimate `β₀·β_m·|[e^{−iτT}e₁]_m|`.

## Automorphism groups (`graph/automorphism.rs`)

A stabiliser chain is built point by point. For base point `b` and candidate
image `w`, an automorphism fixing the earlier base points and mapping
`b ↦ w` is searched by individualisation–refinement: matched vertices of two
copies of the graph get equal colours, 1-WL colour refinement (respecting
edge weights) runs on both copies jointly so colours are comparable,
incompatible colour-class sizes prune the branch, and the search branches on
the smallest non-trivial class. Found automorphisms close the orbit, so
candidates already reached are skipped. The order is the product of the
basic orbit lengths (orbit–stabiliser theorem), which is exact. Test cases
cover Kₙ, Cₙ, Pₙ, stars, grids, hypercubes, the Petersen graph, all
Platonic solids and the Metatron family.

## Quantum walks (`walks/`)

* Dense path (n ≤ 2048): eigendecomposition of the real symmetric
  Hamiltonian once; evolution, propagators, limiting distributions and exact
  time averages are spectral formulas. Degenerate eigenvalues are grouped
  into eigenspaces before computing limiting distributions; v0.1 omitted
  this and produced symmetry-violating results.
* Sparse path (any n): CSR Hamiltonian and Krylov propagation.
* Spatial search optimises the hopping rate around the Childs–Goldstone
  critical value `γ_c = (1/N) Σ 1/λ_k` (exactly 1/N on K_N) together with
  the measurement time.
* Dephasing walks integrate the Lindblad equation with adaptive RK4 (step
  doubling, Richardson extrapolation).

## Error handling

All user-facing operations return `Result<_, QsoError>`. Fluent circuit
builder methods (`c.h(0).cx(0, 1)`) panic on invalid qubit indices, as
indexing would; use `Circuit::try_push` for untrusted input. The Python
bindings convert every `QsoError` into `ValueError`.
