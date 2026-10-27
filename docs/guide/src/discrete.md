# Discrete-time walks and circuits

## Coined walks

`CoinedWalk` is the coined walk on the arcs u → v of a graph: each step
applies a coin to the arcs leaving every node, then the flip-flop shift
|u → v⟩ ↦ |v → u⟩. The coin is `"grover"` (2/d·J − I, the default),
`"fourier"`, `"hadamard"` (degree 2) or any unitary matrix (regular graphs).
Marked nodes get the coin −I, which turns the walk into the search of
Shenvi, Kempe and Whaley: on the hypercube the success probability reaches
about 1/2 after about (π/2)·√(N/2) steps.

```python
import metatron_qso as q

walk = q.CoinedWalk(q.Graph.hypercube(10), "grover", marked=[0])
result = walk.search(60)
print(result["steps"], result["success_probability"])     # 38 0.435
psi = walk.evolve(walk.uniform_state(), 38)
print(walk.node_probabilities(psi)[0])
```

`LineWalk` is the walk on the integer line with the moving shift. The
Hadamard walk spreads ballistically, with standard deviation
√(1 − 1/√2)·t ≈ 0.541·t:

```python
import math
import numpy as np
import metatron_qso as q

steps = 500
p = q.LineWalk().distribution(steps)                # positions −500..500
x = np.arange(-steps, steps + 1)
print(math.sqrt((p * x**2).sum()) / steps)          # 0.5412
```

## Decoherent coined walks

`DecoherentCoinedWalk` runs a coined walk on density matrices, so it can
lose coherence. Every step applies the walk's unitary step and then
dephasing in the arc basis. Coin dephasing measures the coin register,
the index of the arc among those leaving its node, with probability p_c
per step: coherences between arcs with different indices shrink by
1 − p_c. Position dephasing measures the node with probability p_x.
Both are mixtures of the identity and a measurement, so states stay
valid density matrices. With p_c = p_x = 0 it is the coherent walk.

When the coin is measured after every step, the node probabilities
follow a classical Markov chain on the arcs. From an arc arriving at a
node, the walker leaves along arc j with probability |C_jk|². For the
Fourier coin all these are 1/d, so the walk becomes the simple random
walk. The tests check this exactly on the Petersen graph, the cube, a
cycle and Metatron's Cube (K₁₃), and check the arc chain of the Grover
coin, which steps back with probability (1 − 2/d)².

```python
import numpy as np
import metatron_qso as q

walk = q.CoinedWalk(q.Graph.metatron_cube(), "fourier")
for p in [0.0, 0.2, 1.0]:
    open_walk = q.DecoherentCoinedWalk(walk, coin_dephasing=p)
    rho = open_walk.evolve(open_walk.localized_mixed(0), 5)
    print(p, np.round(open_walk.node_probabilities(rho)[:3], 4))
```

Density matrices of arcs grow as (2|E|)²; at most 2048 arcs are allowed.

## Szegedy walks

`SzegedyWalk` quantises the random walk P_uv = w_uv/d_u with the marked
nodes absorbing. Its quantum hitting time (Szegedy 2004) grows as the square
root of the classical hitting time:

```python
import metatron_qso as q

for n in (16, 64, 256):
    g = q.Graph.cycle(n)
    print(n, q.SzegedyWalk(g, 0).hitting_time(10 * n), q.classical_hitting_times(g, 0).mean())
```

## Walks as circuits

Four builders turn exp(−iHt) into a gate circuit on ⌈log₂ N⌉ qubits:

| Function | Graphs | Accuracy |
|---|---|---|
| `circulant_walk_circuit` | circulant graphs on 2ⁿ nodes | exact: QFT, diagonal phases, inverse QFT |
| `cayley_walk_circuit` | Cayley graphs of Z₂ⁿ (hypercubes and relatives) | exact: commuting X-string rotations |
| `trotter_walk_circuit` | any graph | product formula of order 1 or 2; error ∝ steps^−order |
| `diagonal_unitary_circuit` | — | any diagonal unitary, exact with its global phase |

```python
import numpy as np
import metatron_qso as q

g = q.Graph.cycle(16)
exact = q.ContinuousWalk(g).probabilities(0, 2.0)
circuit = q.circulant_walk_circuit(g, 2.0)
assert np.allclose(q.simulate(circuit).probabilities(), exact)

approx = q.trotter_walk_circuit(q.Graph.petersen(), 1.0, steps=20, order=2)
print(approx.n_qubits, len(approx))
qasm = approx.to_qasm()                             # multi-controlled gates decomposed
```

Each edge of a product formula is a rotation between two basis states: a
gate on one qubit controlled by all others. OpenQASM export decomposes such
gates without ancillas into singly controlled gates and Toffolis (Barenco et
al. 1995), O(k²) gates for k controls.

## In Rust

```rust
use metatron_qso::prelude::*;
use metatron_qso::walks::{Coin, CoinedWalk, SzegedyWalk, circuits};

fn main() -> Result<()> {
    let cube = generators::hypercube(10);
    let r = CoinedWalk::new(&cube, Coin::Grover)?.with_marked(&[0])?.search(60)?;
    assert_eq!(r.steps, 38);
    assert_eq!(SzegedyWalk::new(&generators::cycle(64), &[0])?.hitting_time(1000), Some(16));
    let c = circuits::circulant_walk(&generators::cycle(16), Generator::Laplacian, 1.0, 2.0)?;
    println!("{}", to_qasm2(&c, &[], false)?.lines().count());
    Ok(())
}
```
