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

### Single precision

`simulate(circuit, precision="single")` returns a `StateVector32`. Its
amplitudes are `complex64`, half the memory of a `StateVector`, so a
register can be one qubit larger (35 qubits on 64-bit machines instead of
34). The gates run through the same fused and blocked kernels. On x86-64
with AVX, single precision handles four amplitudes per vector register
instead of two.

Gate matrices are formed in double precision and rounded once per gate.
Norms, probabilities, overlaps and expectation values are summed in
double precision, and probabilities and expectation values are divided by
the squared norm. Rounding moves mostly that norm, by up to 10⁻⁵ over a
few thousand gates. The direction of the state stays within
1 − F ≈ 10⁻¹² of the double-precision state, and expectation values agree
to 10⁻⁶ or better. The validation checks these bounds on 2140 gates with 18
qubits, and the cross-validation checks QSO's and Qiskit Aer's single
precision against double precision.

```python
import metatron_qso as q

c = q.Circuit(16)
for r in range(10):
    for i in range(16):
        c.rx(i, 0.3 + 0.1 * i).rz(i, 0.2)
    for i in range(15):
        c.cx(i, i + 1)
single = q.simulate(c, precision="single")
exact = q.simulate(c)
assert 1 - single.fidelity(exact) < 1e-10
print(single.sample(5, seed=1))
```

`StateVector32` offers `zero`, `basis`, `from_state` (rounding a double
state), `to_state`, `apply`, `probabilities`, `sample`, `expectation`,
`fidelity` against a double state and `norm`. `qso qasm --single` samples
a program without classical bits in single precision.

The cross-validation times layers of RX and RZ on every qubit with a brick
of CX and CZ gates. It runs on 4 cores of an Intel Xeon at 2.8 GHz, with
medians of 3 interleaved rounds:

| Qubits, layers | QSO double | QSO single | Qiskit Aer double | Qiskit Aer single |
|---|---:|---:|---:|---:|
| 16, 1 | 1.3 ms | 1.0 ms | 7.1 ms | 7.3 ms |
| 20, 1 | 10.8 ms | 6.5 ms | 57 ms | 55 ms |
| 22, 1 | 70 ms | 37 ms | 334 ms | 331 ms |
| 24, 1 | 277 ms | 150 ms | 1155 ms | 1935 ms |
| 20, 10 | 103 ms | 64 ms | 539 ms | 590 ms |
| 22, 10 | 537 ms | 252 ms | 2554 ms | 2752 ms |

On large registers, where memory bandwidth limits every gate, single
precision takes about half the time. Qiskit Aer's single precision was no
faster than its double precision on this machine. Qulacs offers double
precision only; Cirq and PennyLane Lightning have single precision too but
are timed here in double precision. On
aarch64 the single-precision primitives use NEON, as the double-precision
ones do.

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

## OpenQASM

Circuits export to OpenQASM 2.0 (`to_qasm`), using only gates of the
original `qelib1.inc`, and to OpenQASM 3 (`to_qasm3`), using the gates of
`stdgates.inc` and `ctrl @` modifiers. Gates outside those sets are
decomposed exactly. `Circuit.from_qasm` reads OpenQASM 2.0, including
`gate` definitions and Qiskit's extensions of `qelib1.inc`, and drops
measurements:

```python
import metatron_qso as q

c = q.Circuit(2).h(0).cx(0, 1).rz(1, 0.25)
text = c.to_qasm(measure=True)
again = q.Circuit.from_qasm(text)
assert abs(q.simulate(c).fidelity(q.simulate(again)) - 1) < 1e-12
```

`Circuit.from_qasm` rejects `reset` and classical conditions, parameters
that are not finite, and programs larger than 2¹⁴ qubits or 2²² gates.
Programs with mid-circuit measurements, resets, conditions or OpenQASM 3
syntax are read as a `Program` ([OpenQASM programs](programs.md)).

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
