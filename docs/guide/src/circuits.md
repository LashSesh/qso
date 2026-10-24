# Circuits and states

## Building circuits

A `Circuit` holds a fixed number of qubits and a list of gates. Builder
methods return the circuit, so calls chain:

```python
import metatron_qso as q

ghz = q.Circuit(3).h(0).cx(0, 1).cx(1, 2)
print(len(ghz), ghz.depth, ghz.gate_counts())
```

The gate set:

- **Single-qubit gates:** H, X, Y, Z, S, S†, T, T†, √X, √X†.
- **Rotations:** RX, RY, RZ, P and U.
- **Two-qubit gates:** SWAP, RXX, RYY and RZZ.
- **Controlled gates:** CX, CY, CZ, CH, CRX, CRY, CRZ, CP, CCX, CSWAP, and
  multi-controlled X and Z.
- **Custom unitaries** on up to six qubits, with any number of controls.

```python
import numpy as np
import metatron_qso as q

c = q.Circuit(3)
c.ccx(0, 1, 2).mcz([0, 1, 2]).rzz(0, 2, 0.4)
iswap = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
c.unitary(iswap, targets=[1, 2], controls=[0])
```

## Simulating

`simulate` starts from |0…0⟩ and returns a `StateVector`:

```python
import numpy as np
import metatron_qso as q

state = q.simulate(q.Circuit(3).h(0).cx(0, 1).cx(1, 2))
assert np.allclose(state.probabilities(), [0.5, 0, 0, 0, 0, 0, 0, 0.5])
print(state.sample(1000, seed=1))              # {'000': ..., '111': ...}
print(state.entanglement_entropy([0]))         # 1.0 bit
```

Other starting states are available as `StateVector.zero`, `plus`, `basis`,
`random` and `from_amplitudes`. `state.apply(circuit)` continues from any
state. States support inner products, fidelities and reduced density
matrices.

## Observables

Observables are sums of Pauli strings with real coefficients. The parser
accepts expressions like the ones in papers:

```python
import metatron_qso as q

h = q.PauliSum("0.5*Z0 Z1 - 1.2 X0 + 0.3*Y2*Y3 - 1.05")
print(h.num_terms, h.constant)
state = q.simulate(q.Circuit(4).h(0))
print(h.expectation(state))
energy, ground = h.ground_state()            # Lanczos
print(energy)
```

Sums can be added, subtracted and scaled. The models `transverse_field_ising`,
`heisenberg_xxz`, `maxcut_cost` and `h2_sto3g` build common Hamiltonians.
Expectation values are computed without building a matrix.

## OpenQASM 2.0

Circuits export to OpenQASM 2.0, using only gates of the original
`qelib1.inc`. Gates outside that set are decomposed exactly.
Import reads the same gates plus Qiskit's extensions:

```python
import metatron_qso as q

c = q.Circuit(2).h(0).cx(0, 1).rz(1, 0.25)
text = c.to_qasm(measure=True)
again = q.Circuit.from_qasm(text)
assert abs(q.simulate(c).fidelity(q.simulate(again)) - 1) < 1e-12
```

Import rejects, with an error, the following:

- custom `gate` definitions, `reset` and classical conditions;
- parameters that are not finite;
- programs larger than 2¹⁴ qubits or 2²² gates.

## In Rust

The Rust interface mirrors the Python one:

```rust
use metatron_qso::prelude::*;

fn main() -> Result<()> {
    let mut ghz = Circuit::new(3);
    ghz.h(0).cx(0, 1).cx(1, 2);
    let state = StateVector::from_circuit(&ghz, &[])?;
    let counts = state.sample(1000, &mut Rng::seed_from_u64(1));
    println!("{counts:?}");
    let zz: PauliSum = "Z0 Z2".parse()?;
    assert!((zz.expectation(&state)? - 1.0).abs() < 1e-12);
    println!("{}", to_qasm2(&ghz, &[], true)?);
    Ok(())
}
```
