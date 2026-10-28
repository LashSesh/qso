# WebAssembly and JavaScript

The npm package `metatron-qso` contains QSO compiled to WebAssembly. It
needs no other package, and the module needs no imports. It runs in
Node.js 18 or later, in browsers and in Web Workers.

```js
import { load } from "metatron-qso";

const qso = await load();
const r = qso.qasm(`OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;`, { shots: 1000, seed: 1 });
console.log(r.probabilities);                       // { '00': 0.5, '11': 0.5 }
console.log(qso.graph("metatron").automorphism_order);  // '6227020800' (13!)
console.log(qso.search("metatron", [0]).success_probability);
```

| Method | Result |
|---|---|
| `qasm(source, {shots, seed})` | exact outcome probabilities, seeded samples, the final state of a program without measurements, and exports to OpenQASM 3 and 2 |
| `mps(source, {shots, seed, maxBond, cutoff, observables})` | the program on a [matrix-product state](mps.md): Pauli expectation values, bond dimensions and entropies, the fidelity bound, and samples, for hundreds or thousands of qubits at low entanglement |
| `graph(graph)` | nodes, edges, positions, summary, automorphism group, spectra |
| `walk(graph, {start, times, gamma, generator})` | node probabilities of a continuous-time walk |
| `search(graph, marked, options)` | optimised spatial search and its success curve |
| `isomorphism(first, second)` | isomorphism test with a mapping |

A graph is a specification such as `metatron` or `johnson:6,3`, or
`{edgeList: "0 1\n1 2"}`. Invalid input throws a `QsoError` with the
message of the Rust error. The package README lists all parameters.

## Playground

The [playground](playground/) runs the package in the browser: it
simulates OpenQASM programs (on a matrix-product state too, for hundreds
of qubits), animates quantum walks on Metatron's Cube and other graphs,
and plots spatial search. The module runs in a Web Worker,
so long computations do not block the page; one that runs longer than
30 seconds is stopped. Nothing is sent to a server.

## How it is checked

CI builds the core library for `wasm32-wasip1` and runs its unit,
validation, OpenQASM and robustness tests under Node's WASI. These are the
same test vectors as the native build. Property tests and benchmarks need
native threads and run natively only. Tests that expect a panic are
skipped, because WebAssembly aborts on a panic.

The package has its own tests (`npm test`), which check the OpenQASM
examples and graph results against the native values. For matrix-product
states they replay requests whose native responses are recorded in
`js/test/mps_native.json` (a 40-qubit circuit with truncation, an
8-qubit one with amplitudes, a 300-qubit GHZ state with samples and a
program with mid-circuit measurements) and require the same results to
10⁻¹⁰; a test of the Rust crate checks that the recording still matches
the native build.

## Limits

WebAssembly has a 4 GiB address space, so a state vector holds at most
26 qubits (1 GiB). Clifford programs with more qubits run on the
[stabilizer simulator](stabilizers.md) (up to 16384 qubits); they are
sampled, so they need `shots`. Matrix-product states take up to 16384
qubits; their memory grows with the square of the bond dimension, so
strongly entangling programs on many qubits need a small `maxBond`.
The graph and walk methods accept up to 2048 nodes,
because they use dense eigendecompositions. One call samples at most
10⁶ shots and evaluates at most 10⁴ times.

## Building

```text
rustup target add wasm32-unknown-unknown
python scripts/build_wasm.py          # writes metatron_qso_wasm/js/qso.wasm
cd metatron_qso_wasm/js && npm test
```
