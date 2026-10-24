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
(AVX-512), 15 GB RAM, Linux 6.18, rustc 1.97.0, QSO 1.1.0, release profile
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
| 10 | 29 | 14.5 µs | 0.489 |
| 14 | 41 | 307.7 µs | 0.458 |
| 18 | 53 | 5.16 ms | 0.371 |
| 20 | 59 | 15.25 ms | 0.246 |
| 22 | 65 | 56.48 ms | 0.207 |
| 24 | 71 | 459.19 ms | 0.385 |

## Cost per gate type (22 qubits, 5 passes over all qubits)

| gate | time per pass | ns per gate·amplitude |
|---|---:|---:|
| H | 18.52 ms | 0.201 |
| RY | 20.31 ms | 0.220 |
| RZ | 25.92 ms | 0.281 |
| CX | 13.78 ms | 0.149 |

## Gradients (TFIM, 10 qubits, EfficientSU2 with 4 layers = 100 parameters)

| method | time per full gradient |
|---|---:|
| adjoint | 529.1 µs |
| parameter shift | 27.68 ms |
| finite differences | 23.57 ms |

## QAOA value + gradient (p = 3, random 3-regular graphs)

| nodes | time |
|---:|---:|
| 10 | 343.3 µs |
| 14 | 7.59 ms |
| 18 | 45.26 ms |
| 20 | 173.48 ms |

## Quantum walks

| case | time |
|---|---:|
| Metatron's Cube, P(t) (dense spectral) | 0.4 µs |
| 100×100 torus (10 000 nodes), P(t = 5) (Krylov) | 83.77 ms |
| automorphism group of Metatron's Cube (13!) | 2.96 ms |

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

## Reading the numbers

* **ns per gate·amplitude** normalises a layer's time by gates × 2ⁿ.
  Registers below 2¹⁶ amplitudes run single-threaded. The cost per
  amplitude falls up to 22 qubits as all threads take part and rises again
  at 24 qubits, where one state vector takes 256 MB.
* **Cost per gate type**: uncontrolled real matrices (H, RY) and the
  diagonal RZ have dedicated vectorisable paths; singly controlled X swaps
  contiguous runs of amplitudes (see [ARCHITECTURE.md](ARCHITECTURE.md)).
* **Adjoint differentiation** costs about three circuit evaluations for all
  100 parameters, while parameter shift and finite differences need two
  evaluations per parameter. The ratio grows linearly with the parameter
  count.
* **QAOA** uses the fast diagonal path: the cut values of all 2ⁿ bit strings
  are precomputed once, a cost layer is one phase multiplication and a mixer
  layer is n single-qubit rotations.
* **Quantum walks** on small graphs use one dense eigendecomposition, after
  which evaluating P(t) costs O(n²). Large graphs use sparse Krylov
  propagation, whose cost grows with ‖H‖·t.

## v0.1 comparison

v0.1 advertised "31 941 ops/sec" for quantum walks. That number appears in
none of its benchmark outputs (see [AUDIT_v0.1.md](AUDIT_v0.1.md)). Its
"VQE" and "QAOA" operated on 13×13 matrices, so their timings are not
comparable with simulations of n-qubit registers.
