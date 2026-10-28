# Circuits and algorithms

`metatron-qso-rs/examples/circuits.rs` builds and samples a GHZ state,
exports it to OpenQASM, and runs Grover search and phase estimation.

## A GHZ state

A `Circuit` is a list of gates on `n` qubits; the gate methods return the
circuit so they can be chained. `StateVector::from_circuit` simulates it
(the empty slice is the parameter vector: this circuit has none).
Sampling takes a seeded `Rng`, so the counts are the same on every run
and every machine. Qubit 0 is the least significant bit of a basis index
(`bitstring` prints it rightmost; see [Conventions](../conventions.md)).

The entanglement entropy of one qubit of a GHZ state is one bit, and
`to_qasm2` writes the circuit as an OpenQASM 2 program that other
toolkits can read.

{{example circuits.rs:ghz}}

## Grover search

`grover_search(6, &[42], Some(k))` searches 64 items for item 42 with `k`
Grover iterations. With one marked item of `N`, the success probability
after `k` iterations is sin²((2k + 1)θ) with sin θ = 1/√N; the program
prints the simulated probability next to that formula.

{{example circuits.rs:grover}}

## Phase estimation

Quantum phase estimation with 4 counting qubits on the phase gate
P(2π·0.3125): 0.3125 = 5/16 has an exact 4-bit binary expansion, so the
estimate is exact with probability one.

{{example circuits.rs:qpe}}

## Output

{{output circuits}}
