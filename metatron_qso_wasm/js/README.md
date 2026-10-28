# metatron-qso

QSO, a validated quantum simulator written in Rust, compiled to
WebAssembly. The package has no dependencies, and the module needs no
imports, so it runs in Node.js 18+, in browsers and in Web Workers.

It runs OpenQASM 2 and 3 programs, analyses graphs and their symmetry, and
simulates continuous-time quantum walks and spatial search, including on
Metatron's Cube. The results match the native library on the same test
vectors (`test/`). A [playground](https://lashsesh.github.io/qso/playground/)
uses this package.

```js
import { load } from "metatron-qso";

const qso = await load();
const bell = qso.qasm(`
  OPENQASM 3.0;
  include "stdgates.inc";
  qubit[2] q;
  bit[2] c;
  h q[0];
  cx q[0], q[1];
  c = measure q;
`, { shots: 1000, seed: 1 });
bell.probabilities;   // { "00": 0.5, "11": 0.5 }

qso.graph("metatron").automorphism_order;   // "6227020800" (13!)
qso.search("metatron", [0]).success_probability;   // 1.0 (to 1e-15)
```

## Methods

Every method runs synchronously. Invalid input throws a `QsoError` whose
message names the problem. For long computations in a browser, run the
module in a Web Worker.

| Method | Result |
| --- | --- |
| `qasm(source, {shots, seed})` | Exact outcome probabilities of the classical bits, seeded samples, the final state of a program without measurements, and the program as OpenQASM 3 and (when possible) OpenQASM 2. Beyond 4096 measurement branches the probabilities are `null` and the shots are simulated one by one. Clifford programs with more than 26 qubits run on the stabilizer simulator (thousands of qubits; `method: "stabilizer"`, shots only) |
| `mps(source, {shots, seed, maxBond, cutoff, observables})` | The program on a matrix-product state, for many qubits at low entanglement: expectation values of Pauli strings (`"Z0 Z1"`, `"-X3"`), bond dimensions, entanglement entropies, amplitudes (up to 10 qubits), the discarded weight and a lower bound on the fidelity with the exact state; seeded samples of the classical bits (as `qasm`), or of every qubit for a program without bits. Programs with measurements or resets are sampled only, through their classical bits. `maxBond` (default 256) and `cutoff` (default 10⁻¹⁴) set the truncation |
| `graph(graph)` | Nodes, edges, positions, summary, automorphism group order and orbits, adjacency and Laplacian spectra, strongly regular parameters |
| `walk(graph, {start, times, gamma, generator})` | Node probabilities of `exp(−iHt)` from `start` at each time, and the limiting distribution |
| `search(graph, marked, {gamma, generator, tMax, points})` | Best hopping rate (unless given) and time, success probability, and the success curve over `[0, tMax]` |
| `isomorphism(first, second)` | Whether two graphs are isomorphic, and a mapping |
| `version()` | Version of the QSO library inside the module |

A graph is a specification string, or `{edgeList: "0 1\n1 2\n…"}`.
Specifications include `metatron`, `fruit-of-life`, `complete:N`,
`cycle:N`, `path:N`, `hypercube:D`, `grid:RxC`, `petersen`,
`platonic:NAME`, `johnson:N,K`, `kneser:N,K`, `hamming:D,Q`, `paley:Q`
and `circulant:N,S1,S2,…`. The generator is `laplacian` (`H = γL`, the
default) or `adjacency` (`H = −γA`).

The bit strings put bit 0 (or qubit 0) on the right, as Qiskit does.
Results are plain objects; JavaScript lists keys that look like integers
(`"10"`) before others, so sort the keys when the order matters.

## Limits

- Programs: OpenQASM 2, and the OpenQASM 3 subset of QSO (gates and
  gate definitions, modifiers, measurement, reset and simple
  conditions); at most 26 qubits in WebAssembly's 4 GiB address space,
  and up to 16384 for Clifford programs (stabilizer simulator).
- Matrix-product states: up to 16384 qubits (the limit of programs) and
  bond dimension 4096; the memory and time grow with the number of qubits
  times the square (memory) or cube (time) of the bond dimension, so keep
  `maxBond` small for strongly entangling programs on many qubits. Samples
  list at most 2²⁶ bits per call (shots × qubits, or × bits).
- Graphs: at most 2048 nodes, because spectra and walks use dense
  eigendecompositions.
- At most 10⁶ shots and 10⁴ times or curve points per call.

## Lower-level interface

`qso.call(method, params)` sends `{"method", "params"}` to the module and
returns the result object. The module's exports are `qso_alloc`,
`qso_call`, `qso_response_len`, `qso_free` and `memory`; `index.js` shows
how to use them from any WebAssembly host.

## Licence

Apache-2.0. Source: <https://github.com/LashSesh/qso>.
