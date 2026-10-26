# Conventions

## Qubit order

Qubit 0 is the **least significant** bit of a basis index, as in Qiskit. The
amplitude at index `k` belongs to the basis state with bit `q` of `k` on
qubit `q`. Bit strings are printed with qubit n−1 first, so `"001"` means
qubit 0 is 1.

```python
import metatron_qso as q

state = q.simulate(q.Circuit(3).x(0))
assert abs(state.amplitudes[1]) == 1.0      # index 1 = 0b001
assert q.bitstring(1, 3) == "001"
```

Cirq and PennyLane order qubits the other way. The cross-framework
comparison reverses the qubits when it compares their states with QSO's.

## Gates and time evolution

| Object | Convention |
|---|---|
| Rotations | `R_P(θ) = exp(−iθP/2)` for P ∈ {X, Y, Z} |
| Two-qubit rotations | `R_PP(θ) = exp(−iθ P⊗P/2)` |
| Phase gate | `P(λ) = diag(1, e^{iλ})` |
| `U(θ, φ, λ)` | the U3 matrix `[[cos θ/2, −e^{iλ} sin θ/2], [e^{iφ} sin θ/2, e^{i(φ+λ)} cos θ/2]]`; OpenQASM 2's `U` up to a global phase, OpenQASM 3's `U` divided by `e^{iθ/2}` |
| Time evolution | `exp(−iHt)` |
| Quantum walks | `H = γL` (Laplacian) or `H = −γA` (adjacency) |

## Randomness

All randomness goes through one portable, seeded generator (xoshiro256++).
Functions that sample or restart take a `seed`. The same seed gives the same
result on every platform and for every thread count.

## Errors

Invalid input raises `ValueError` in Python and returns `QsoError` in Rust,
never a panic. This covers, among others, out-of-range qubits, wrong
parameter counts, non-finite values, unsupported OpenQASM and oversized
requests. The fuzz tests check this for the parsers.

```python
import metatron_qso as q

try:
    q.Circuit(2).cx(0, 2)
except ValueError as e:
    print(e)
```

## Limits

| Limit | Value |
|---|---|
| Pure-state register | 34 qubits (16 bytes per amplitude); 26 in WebAssembly |
| Stabilizer register (Clifford circuits) | 16384 qubits |
| Density-matrix register | 14 qubits |
| Pauli strings | qubits 0–63 |
| Custom unitary gates | up to 6 target qubits |
| Exact walk analysis | up to 2048 nodes (Krylov propagation beyond) |
| Graphs from specifications and Python constructors | 2²⁰ nodes, 2²² edges (hypercubes up to dimension 20) |
| OpenQASM import | version 2.0 and an OpenQASM 3 subset, up to 2¹⁴ qubits, 64 bits and 2²² operations |
