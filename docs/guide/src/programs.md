# OpenQASM programs

A `Program` holds what a `Circuit` cannot: measurements into classical
bits, resets, and gates conditioned on measured bits. QSO reads programs
from OpenQASM 2 and from a subset of OpenQASM 3, and writes them back in
either version.

## Running a program

`probabilities()` returns the exact distribution of the classical bits.
At each mid-circuit measurement, QSO follows both outcomes, each with its
weight, so the result has no sampling error. `sample(shots, seed)` draws
seeded shots from that distribution. Bit strings put bit 0 on the right,
as for qubits.

```python
import math
import metatron_qso as q

teleport = q.Program.from_qasm("""
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[3] c;
U(0.3, 0.2, 0.1) q[0];      // the state to teleport
h q[1];
cx q[1], q[2];
cx q[0], q[1];
h q[0];
c[0] = measure q[0];
c[1] = measure q[1];
if (c[1] == 1) x q[2];      // corrections fed forward
if (c[0] == 1) z q[2];
c[2] = measure q[2];
""")
p = teleport.probabilities()
p1 = sum(v for k, v in p.items() if k[0] == "1")    # bit 2
assert abs(p1 - math.sin(0.15) ** 2) < 1e-12
print(teleport.sample(1000, seed=1))
print(teleport.to_qasm3())
```

Bit 2 reads 1 with probability sin²(0.15), whatever bits 0 and 1 read:
the corrections undo the random outcomes.

A program with only gates converts to a `Circuit` (`to_circuit()`), and
any circuit becomes a program with `Program.from_circuit(circuit,
measure=True)`. `Circuit.to_qasm3()` writes a circuit as OpenQASM 3.

## In Rust

```rust
use metatron_qso::quantum::Program;
use metatron_qso::Result;

fn main() -> Result<()> {
    let program = Program::from_qasm(
        "OPENQASM 3.0;
         include \"stdgates.inc\";
         qubit[2] q;
         bit[2] c;
         h q[0];
         c[0] = measure q[0];
         if (c[0] == 1) x q[1];
         c[1] = measure q[1];",
    )?;
    // Records 00 and 11 (bit 0 on the right), each with probability 1/2.
    let p = program.probabilities()?;
    assert!((p[&0b00] - 0.5).abs() < 1e-12 && (p[&0b11] - 0.5).abs() < 1e-12);
    println!("{}", program.to_qasm3()?);
    Ok(())
}
```

## The supported language

One parser reads both versions. The `OPENQASM` line selects the version;
a program without it is read as OpenQASM 3. Supported are:

- `qreg`/`creg` and `qubit`/`bit` declarations, several registers,
  indices and inclusive slices `r[a:b]`;
- `include "qelib1.inc"` and `include "stdgates.inc"` (the standard gates
  are built in);
- `gate` definitions with parameters; `U`, `CX` and `gphase`;
- the modifiers `ctrl @`, `ctrl(n) @`, `negctrl @`, `inv @` and
  `pow(k) @` for integer `k`;
- `measure`, `reset` and `barrier` (ignored);
- conditions on a register, a bit or `int[n](c)` with `==` or `!=`, with
  a statement or a block, and `else`;
- register broadcasting, and parameter expressions with the usual
  operators, functions and constants.

Anything else in OpenQASM 3 is an error that names the feature:
classical variables and arithmetic, loops, subroutines, `defcal`, timing,
`opaque`, hardware qubits and non-integer powers. Such a program is
rejected, never misread.

The semantics follow each version's specification, including global
phases where they matter. In OpenQASM 3, the built-in `U(θ, φ, λ)` carries
the phase `e^{iθ/2}` relative to OpenQASM 2's `U`. That phase and
`gphase` become relative phases under `ctrl @`.

Exports use only standard gates. OpenQASM 2 export fails with an error
when a program needs OpenQASM 3: OpenQASM 2 conditions compare a whole
register, so a condition on one bit of a larger register (as in the
example above) has no OpenQASM 2 form.

## How it is checked

The tests run the example programs of the OpenQASM project
(`metatron-qso-rs/tests/data/openqasm`) and check their known results:

- the ripple-carry adder computes 1 + 15 = 16;
- phase estimation of `e^{i·3π/8}` gives 3/16 exactly in four bits;
- teleportation transfers the state;
- the W state has three outcomes of probability 1/3;
- the repetition code locates and corrects its error.

The OpenQASM 3 versions of six examples give the same distributions as
their OpenQASM 2 versions; the other fifteen OpenQASM 3 examples use
features outside the subset and are rejected. Every example also
survives a round trip through both exports. A fuzz target checks the same
round trip on random programs.

## Limits

| Limit | Value |
|---|---|
| Declared qubits | 2¹⁴ (simulation: 34 qubits, 26 in WebAssembly) |
| Classical bits | 64 |
| Operations after expanding gate definitions | 2²² |
| Exact distribution | up to 4096 measurement branches and 2²⁶ amplitudes held at once |
| Shot-by-shot sampling (beyond that) | shots × operations up to 2³⁰ |
