# Benchmarks

All numbers on this page are produced by

```bash
cargo run --release --example benchmark_report
```

which prints the tables below (timings are medians of repeated runs after a
warm-up). Criterion micro-benchmarks for the same workloads are available
with `cargo bench -p metatron-qso-rs`. Numbers depend on the machine; run
the report on yours before comparing.

**Machine for the numbers below:** cloud container, 4 hardware threads,
Linux 6.18, rustc 1.97.0, release profile (`opt-level = 3`, thin LTO,
`codegen-units = 1`), default features (rayon parallelism).

The container is a shared virtual machine: repeated runs of the report
varied by up to 2× for sub-millisecond workloads (e.g. a 10-qubit layer
took between 39 µs and 82 µs, a 100-parameter adjoint gradient between
1.0 ms and 2.3 ms). Larger workloads varied less but noticeably: an
18-qubit layer between 12 and 17 ms, the 10 000-node walk between 88 and
133 ms. The tables show one complete run.

<!-- BEGIN REPORT -->
## Statevector throughput (layer = H + RZ on every qubit, CX chain)

| qubits | gates/layer | time/layer | ns per gate·amplitude |
|---:|---:|---:|---:|
| 10 | 29 | 70.6 µs | 2.377 |
| 14 | 41 | 913.8 µs | 1.360 |
| 18 | 53 | 15.73 ms | 1.132 |
| 20 | 59 | 53.26 ms | 0.861 |
| 22 | 65 | 242.97 ms | 0.891 |
| 24 | 71 | 936.11 ms | 0.786 |

## Gradients (TFIM, 10 qubits, EfficientSU2 with 4 layers = 100 parameters)

| method | time per full gradient |
|---|---:|
| adjoint | 1.59 ms |
| parameter shift | 68.34 ms |
| finite differences | 87.82 ms |

## QAOA value + gradient (p = 3, random 3-regular graphs)

| nodes | time |
|---:|---:|
| 10 | 697.4 µs |
| 14 | 10.51 ms |
| 18 | 82.35 ms |
| 20 | 308.98 ms |

## Quantum walks

| case | time |
|---|---:|
| Metatron's Cube, P(t) (dense spectral) | 0.9 µs |
| 100×100 torus (10 000 nodes), P(t = 5) (Krylov) | 133.24 ms |
| automorphism group of Metatron's Cube (13!) | 3.23 ms |

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
  Small registers fit in cache and run single-threaded (parallelism starts
  at 2¹⁶ amplitudes). From about 20 qubits the simulator is limited by memory
  bandwidth: a gate reads and writes every amplitude (32 bytes), so
  0.79 ns per gate·amplitude at 24 qubits corresponds to ≈ 40 GB/s of
  memory traffic.
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
