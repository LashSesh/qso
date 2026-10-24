# Benchmarks

All numbers on this page are produced by

```bash
cargo run --release --example benchmark_report
```

which prints the tables below (timings are medians of repeated runs after a
warm-up). Criterion micro-benchmarks for the same workloads are available
with `cargo bench -p metatron-qso-rs`. Numbers depend on the machine; run
the report on yours before comparing.

**Machine for the numbers below:** cloud VM, 4 vCPU Intel Xeon @ 2.10 GHz
(AVX-512), 15 GB RAM, Linux 6.18, rustc 1.97.0, QSO 1.3.0, release profile
(`opt-level = 3`, thin LTO, `codegen-units = 1`), default features (rayon
parallelism). A comparison with Qiskit Aer, Qulacs, Cirq and PennyLane on
the same machine is in the [technical note](technical-note/) (Section 3.8).

The machine is a shared virtual machine: repeated runs varied by up to 2×
for sub-millisecond workloads and by tens of percent for larger ones. The
tables show one complete run.

<!-- BEGIN REPORT -->
## Statevector throughput (layer = H + RZ on every qubit, CX chain)

| qubits | gates/layer | time/layer | ns per gate·amplitude |
|---:|---:|---:|---:|
| 10 | 29 | 14.4 µs | 0.485 |
| 14 | 41 | 319.8 µs | 0.476 |
| 16 | 47 | 546.4 µs | 0.177 |
| 18 | 53 | 2.20 ms | 0.159 |
| 20 | 59 | 8.72 ms | 0.141 |
| 22 | 65 | 34.36 ms | 0.126 |
| 24 | 71 | 210.71 ms | 0.177 |

## Cost per gate type (22 qubits, 5 passes over all qubits)

| gate | time per pass | ns per gate·amplitude |
|---|---:|---:|
| H | 8.17 ms | 0.089 |
| RY | 8.44 ms | 0.092 |
| RZ | 12.07 ms | 0.131 |
| CX | 7.83 ms | 0.085 |

## Gradients (TFIM, 10 qubits, EfficientSU2 with 4 layers = 100 parameters)

| method | time per full gradient |
|---|---:|
| adjoint | 465.9 µs |
| parameter shift | 24.53 ms |
| finite differences | 22.44 ms |

## Batched evaluation (same circuit and Hamiltonian, 256 parameter vectors)

| quantity | one call per vector | one batch call | speed-up |
|---|---:|---:|---:|
| expectation values | 26.76 ms | 8.24 ms | 3.2× |
| values + adjoint gradients | 128.26 ms | 34.27 ms | 3.7× |

## QAOA value + gradient (p = 3, random 3-regular graphs)

| nodes | time |
|---:|---:|
| 10 | 231.5 µs |
| 14 | 4.61 ms |
| 18 | 35.34 ms |
| 20 | 128.54 ms |

## Quantum walks

| case | time |
|---|---:|
| Metatron's Cube, P(t) (dense spectral) | 0.4 µs |
| 100×100 torus (10 000 nodes), P(t = 5) (Krylov) | 78.39 ms |
| automorphism group of Metatron's Cube (13!) | 2.73 ms |

## Validated results (reproduced by this run)

| quantity | QSO | reference |
|---|---:|---:|
| H₂/STO-3G total energy, VQE (Ha) | -1.137306 | −1.137306 (FCI) |
| QAOA p=1 ratio, ring C₁₀ | 0.750000 | 0.75 (Farhi et al.) |
| QAOA p=1 ratio, cube graph Q₃ | 0.692450 | 0.6925 (3-regular, triangle-free) |
| QAOA p=3 ratio, Metatron's Cube (K₁₃, C_max = 42) | 0.999971 | — |
| CTQW limiting P(centre), Metatron's Cube | 0.857988 | 145/169 = 0.857988 |
| spatial search on K₁₃, P at t = π√13/2 | 1.0000000000 | 1 |
| Grover, 10 qubits, 25 iterations | 0.999461 | 0.999461 = sin²((2k+1)θ) |
| \|Aut\| Metatron's Cube / Fruit of Life / dodecahedron | 6227020800 / 12 / 120 | 13! / 12 / 120 |
<!-- END REPORT -->

## 1.2.0 and 1.3.0 on the same machine

Medians of three runs of each version, alternating between the two builds
(the 1.2.0 report with a 16-qubit row added):

| workload | 1.2.0 | 1.3.0 | speed-up |
|---|---:|---:|---:|
| layer, 10 qubits | 14.9 µs | 14.4 µs | 1.0× |
| layer, 14 qubits | 317.4 µs | 312.5 µs | 1.0× |
| layer, 16 qubits | 2.56 ms | 546.4 µs | 4.7× |
| layer, 18 qubits | 5.56 ms | 2.16 ms | 2.6× |
| layer, 20 qubits | 14.74 ms | 9.37 ms | 1.6× |
| layer, 22 qubits | 65.69 ms | 38.08 ms | 1.7× |
| layer, 24 qubits | 476.39 ms | 210.71 ms | 2.3× |
| H pass, 22 qubits | 19.58 ms | 8.77 ms | 2.2× |
| RY pass, 22 qubits | 19.39 ms | 8.44 ms | 2.3× |
| RZ pass, 22 qubits | 23.93 ms | 12.08 ms | 2.0× |
| CX pass, 22 qubits | 13.70 ms | 7.91 ms | 1.7× |
| adjoint gradient, 10 qubits | 531.7 µs | 473.4 µs | 1.1× |
| parameter shift gradient, 10 qubits | 26.55 ms | 24.75 ms | 1.1× |
| QAOA value + gradient, 10 nodes | 355.6 µs | 244.4 µs | 1.4× |
| QAOA value + gradient, 14 nodes | 7.44 ms | 5.09 ms | 1.5× |
| QAOA value + gradient, 18 nodes | 41.89 ms | 36.03 ms | 1.2× |
| QAOA value + gradient, 20 nodes | 171.30 ms | 129.97 ms | 1.3× |
| Metatron's Cube, P(t) (dense spectral) | 0.4 µs | 0.4 µs | 1.0× |
| 100×100 torus (10 000 nodes), P(t = 5) (Krylov) | 82.81 ms | 85.12 ms | 1.0× |
| automorphism group of Metatron's Cube (13!) | 2.88 ms | 2.81 ms | 1.0× |

Release 1.3.0 applies gates on the lower 14 qubits, and diagonal gates on
any qubit, block by block to cache-sized blocks of 2¹⁴ amplitudes, the
blocks in parallel, and fuses runs of single-qubit gates on higher qubits
(see [ARCHITECTURE.md](ARCHITECTURE.md)). Registers of 14 qubits or fewer
fit into one block and run as before.

## Reading the numbers

* **ns per gate·amplitude** normalises a layer's time by gates × 2ⁿ.
  Registers of up to 14 qubits fit into one cache block and are simulated
  gate by gate on one thread. Larger registers are split into blocks that
  run in parallel, and the cost per amplitude drops to 0.14–0.19 ns from
  16 qubits on; at 24 qubits one state vector takes 256 MB.
* **Cost per gate type**: uncontrolled real matrices (H, RY) and the
  diagonal RZ have dedicated vectorisable paths; singly controlled X swaps
  contiguous runs of amplitudes (see [ARCHITECTURE.md](ARCHITECTURE.md)).
* **Adjoint differentiation** costs about three circuit evaluations for all
  100 parameters, while parameter shift and finite differences need two
  evaluations per parameter. The ratio grows linearly with the parameter
  count.
* **Batched evaluation** runs one parameter vector per thread for
  registers below 16 qubits, where a single evaluation cannot use all
  threads. Run-to-run variation of the speed-up was 2.7–4.4×.
* **QAOA** uses the fast diagonal path: the cut values of all 2ⁿ bit strings
  are precomputed once, a cost layer multiplies by phases from a table of
  the distinct cut values, and cost and mixer layers share one blocked pass
  over the state.
* **Quantum walks** on small graphs use one dense eigendecomposition, after
  which evaluating P(t) costs O(n²). Large graphs use sparse Krylov
  propagation, whose cost grows with ‖H‖·t.

## v0.1 comparison

v0.1 advertised "31 941 ops/sec" for quantum walks. That number appears in
none of its benchmark outputs (see [AUDIT_v0.1.md](AUDIT_v0.1.md)). Its
"VQE" and "QAOA" operated on 13×13 matrices, so their timings are not
comparable with simulations of n-qubit registers.
