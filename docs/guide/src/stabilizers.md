# Stabilizer simulation and graph states

Circuits made only of Clifford gates map Pauli operators to Pauli
operators. Their states are stabilizer states, which a tableau of 2n
Pauli strings describes exactly (Aaronson and Gottesman, Phys. Rev. A 70,
052328 (2004)). `StabilizerState` simulates such circuits on up to 16384
qubits, where a state vector stops at 34.

## Gates, measurements and expectation values

The Clifford gates are:

- H, S, S†, X, Y, Z, √X, √X† and SWAP;
- CX, CY and CZ;
- RX, RY, RZ, P, RXX, RYY, RZZ and U, when every angle is a multiple of
  π/2.

Any other gate is an error, and so is an angle that is not a multiple of
π/2 to within 10⁻⁹ (plus the rounding of large angles; beyond about 5·10⁸
rad, where that rounding exceeds 10⁻⁶, angles are refused). A stabilizer
state is defined only up to a global phase.

```python
import metatron_qso as q

n = 1000
ghz = q.StabilizerState(n, seed=1).h(0)
for k in range(1, n):
    ghz.cx(k - 1, k)
assert ghz.expectation("Z0 Z999") == 1.0     # perfectly correlated
assert ghz.expectation("Z500") == 0.0        # each qubit is random
counts = ghz.sample(100, seed=2)
assert set(counts) == {"0" * n, "1" * n}
first = ghz.measure(0)                       # collapses the state
assert ghz.measure(999) == first
```

Pauli operators are written sparsely, `"X0 Z1 Y1000"`, with an optional
sign in front. `expectation` returns +1, −1 or 0 (a Pauli operator either
stabilizes the state up to sign or is completely random). It also takes
a `PauliSum`.

- `measure` and `measure_pauli` collapse the state.
- `sample` measures every qubit on copies of the state: the outcomes are
  uniform over an affine subspace of {0,1}ⁿ, which QSO computes once by
  Gaussian elimination.
- `Circuit`s run with `StabilizerState.from_circuit` or `apply`;
  `is_clifford(circuit)` tells whether they can.

## Graph states

The graph state of a graph `G` is `|G⟩ = Π_{(u,v)∈E} CZ_uv |+⟩^⊗n`. Its
stabilizers are `K_v = X_v Π_{w∈N(v)} Z_w`. Graph states turn graph theory
into entanglement, and QSO builds them from any of its graphs:

- The entanglement entropy across a bipartition equals the rank over
  GF(2) of the adjacency matrix between the two sides, its cut-rank (Hein,
  Eisert and Briegel, Phys. Rev. A 69, 062311 (2004)).
  `entanglement_entropy` computes it for any stabilizer state.
- Local complementation at `v` toggles the edges between the neighbours
  of `v`. It is a local Clifford operation: `√(−iX_v) Π_{w∈N(v)} √(iZ_w)`
  maps `|G⟩` to `|G * v⟩`.

```python
import metatron_qso as q

cube = q.Graph.from_spec("metatron")          # K13
s = q.StabilizerState.graph_state(cube)
assert s.entanglement_entropy([0, 1, 2, 3, 4, 5]) == 1

star = q.local_complementation(cube, 0)       # K13 * 0 is the star
assert star.num_edges == 12
assert s.local_complement(cube, 0).same_state(q.StabilizerState.graph_state(star))

q10 = q.StabilizerState.graph_state(q.Graph.from_spec("hypercube:10"))
print(q10.entanglement_entropy(list(range(512))))   # 512: a perfect matching
```

Local complementation at any node of the complete graph K₁₃ removes every
other edge and leaves the star. The graph state of Metatron's Cube is
therefore a 13-qubit GHZ state up to local Cliffords: one ebit across
every bipartition.

## Clifford programs

OpenQASM programs made of Clifford gates run on the stabilizer simulator,
mid-circuit measurements, resets and conditions included. A program of
gates followed by measurements is simulated once and then sampled; any
other program is simulated shot by shot.

- **Python:** `Program.sample(shots, method="auto")` chooses the stabilizer
  simulator for Clifford programs with more qubits than a state vector
  holds; `method="stabilizer"` forces it.
- **Command line:** `qso qasm` chooses the same way; `--stabilizer`
  forces it.
- **WebAssembly:** the module and the playground choose the same way.

## In Rust

```rust
use metatron_qso::Result;
use metatron_qso::graph::generators::from_spec;
use metatron_qso::rng::Rng;
use metatron_qso::stabilizer::StabilizerState;

fn main() -> Result<()> {
    let mut ghz = StabilizerState::zero(2000)?;
    ghz.h(0)?;
    for q in 1..2000 {
        ghz.cx(q - 1, q)?;
    }
    assert_eq!(ghz.expectation_str("Z0 Z1999")?, 1.0);
    let mut rng = Rng::seed_from_u64(1);
    let v = ghz.measure(7, &mut rng)?;
    assert_eq!(ghz.measure(1999, &mut rng)?, v);

    let c1000 = StabilizerState::graph_state(&from_spec("cycle:1000")?)?;
    let arc: Vec<usize> = (0..500).collect();
    assert_eq!(c1000.entanglement_entropy(&arc)?, 2); // two edges cross
    Ok(())
}
```

## How it is checked

- 200 random Clifford circuits on 1–8 qubits, drawn from every supported
  gate form, give the same states as the state-vector simulator, and the
  same expectation values of random Pauli operators.
- After a random Pauli measurement, the state equals the projected state
  vector; repeating the measurement repeats the outcome.
- Sampled outcomes lie in the support of the state vector, which has
  `2^k` equally likely outcomes.
- The graph states of six graphs (Petersen, a cycle, the Fruit of Life,
  Metatron's Cube, a path, a star) equal H followed by CZ on every edge.
  Their entanglement equals both the state-vector entropy and the
  cut-rank, and local complementation equals the local Clifford of Hein
  et al.
- A GHZ state on 10 000 qubits has its known correlations. The cut-ranks
  of the hypercube Q₁₀ (512) and the cycle C₁₀₀₀ (2) are reproduced.
- Clifford programs agree with the exact outcome distributions in the
  tests (counts within a few standard deviations). A fuzz target runs
  random Clifford programs on the stabilizer simulator and checks that
  every record it produces has non-zero exact probability.

## Limits

| Limit | Value |
|---|---|
| Register | 16384 qubits |
| Memory | about n²/2 bytes (128 MiB at the limit), twice that while measuring |
| Gate | O(n/32) word operations |
| Measurement, expectation value, `same_state` | O(n²): the tableau is transposed to rows (about a second per measurement at the limit) |
| Sampling every qubit | O(n³/64) once (about two minutes at the limit), then O(n²/64) per shot; shots × qubits at most 2²⁶ |
| Programs of gates followed by measurements | simulated once, then O(1) per shot for their at most 64 bits |
| Other programs | simulated shot by shot; shots × operations × qubits² at most 2³⁶ |
| Conversion to a state vector | 20 qubits |
