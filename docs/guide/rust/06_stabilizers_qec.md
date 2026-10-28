# Stabilizers and error correction

Clifford circuits (H, S, CX and measurements) can be simulated in
polynomial time with stabilizer tableaux, and that is what makes
error-correction experiments on hundreds of qubits possible. Two
examples: `stabilizers.rs` (large GHZ and graph states) and
`surface_code.rs` (memory experiments with decoders).

## A GHZ state on 10 000 qubits

`StabilizerState` stores a tableau of 2n Pauli strings, bit-packed. A GHZ
state on 10 000 qubits takes a Hadamard and 9 999 CNOTs; measuring the
first and the last qubit gives random but equal outcomes.

{{example stabilizers.rs:ghz}}

## Entanglement of graph states

A graph state has one qubit per node, prepared in |+⟩, and a CZ for every
edge. Its entanglement across a cut, in ebits, is the rank over GF(2) of
the adjacency matrix between the two sides; `entanglement_entropy`
computes it from the tableau. `generators::from_spec` builds graphs from
short names (see [Graphs](../graphs.md)).

{{example stabilizers.rs:graph_states}}

## Local complementation

Local complementation at a node complements the edges among its
neighbours; on graph states it is a local Clifford operation. Metatron's
Cube is the complete graph K₁₃, so local complementation at node 0
removes every edge among the other twelve and leaves a star, whose graph
state is a GHZ state up to Hadamards on the leaves. `same_state` compares
two stabilizer states exactly.

{{example stabilizers.rs:local_complement}}

Output of `cargo run --release --example stabilizers`:

{{output stabilizers}}

## Surface-code memory

`surface_code_memory(d, rounds, noise)` builds a rotated surface code of
distance `d` with `rounds` rounds of syndrome extraction under
circuit-level noise of strength `p` (`CircuitNoise::uniform`:
depolarising noise after every gate and on the data qubits before each
round, flipped measurement results and faulty resets, all with
probability `p`), with detectors and the logical observable annotated. `logical_error_rate` samples it with Pauli
frames and decodes with union-find; `logical_error_rate_with` takes the
decoder, here minimum-weight matching. Below the threshold, which for
this noise lies between 0.3 % and 1 %, the logical error rate falls with
the distance; above it, it grows.

{{example surface_code.rs:surface}}

The same in the X basis (a |+⟩ logical state):

{{example surface_code.rs:x_basis}}

## Repetition code and the error model

The repetition code protects against bit flips only and has a higher
threshold. `detector_error_model` lists every independent error
mechanism of a circuit with the detectors and observables it flips (the
input of the decoders); with `true` it splits errors that flip more than
two detectors into graph-like parts, as Stim does.

{{example surface_code.rs:repetition}}

{{example surface_code.rs:dem}}

Output of `cargo run --release --example surface_code` (the rates are
estimates from the shots; [Error-correction experiments](../qec.md)
compares them with Stim and PyMatching):

{{output surface_code}}
