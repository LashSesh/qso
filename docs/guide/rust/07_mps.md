# Matrix-product states

A state vector of 200 qubits does not fit in any computer, but states
with little entanglement do as matrix-product states (MPS): a chain of
tensors whose bond dimension χ grows with the entanglement across each
cut. `metatron-qso-rs/examples/mps.rs` simulates a quench and QAOA on 200
qubits.

## A quench of the XX chain

Start from the Néel state |0101…⟩ and apply Trotter steps of the XX
Hamiltonian (RXX and RYY on every neighbouring pair, even bonds then odd
bonds). The excitations spread ballistically, so the entanglement entropy
across the middle grows linearly with time, and with it the bond
dimension. `MpsOptions::default()` keeps up to 256 singular values per
bond and drops only numerically negligible weight (10⁻¹⁴ per split);
`fidelity_lower_bound` turns the truncations into a lower bound on the
fidelity with the exact state, so the result comes with an error bar.

{{example mps.rs:quench}}

## QAOA on a ring of 200 nodes

For MaxCut on a ring, the term of an edge (a, b) after depth-p QAOA
depends only on the 2p + 2 nodes from a − p to b + p. On every ring
longer than that the expectation is the same, so angles optimised on a
ring of 10 (enough for p ≤ 3) work on a ring of 200, and the expected
cut fraction is the proven value (2p + 1)/(2p + 2). The program
optimises on the small ring with the state-vector simulator, then
evaluates every edge's ⟨Z_a Z_b⟩ on the large ring with an MPS.

{{example mps.rs:qaoa}}

## Output

{{output mps}}

[Matrix-product states](../mps.md) describes the options, the routing of
gates between distant qubits and the validation against state vectors.
