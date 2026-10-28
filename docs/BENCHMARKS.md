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
(AVX-512), 15 GB RAM, Linux 6.18, rustc 1.97.0, QSO 1.6.0 as of
milestone 1.11 (AVX kernels; 1.12 and 1.13 add modules these benchmarks do
not use), release profile
(`opt-level = 3`, thin LTO, `codegen-units = 1`), default features (rayon
parallelism). A comparison with Qiskit Aer, Qulacs, Cirq, PennyLane
Lightning and The Walrus on the same machine is
[below](#comparison-with-other-simulators-130).

The machine is a shared virtual machine: repeated runs varied by up to 2×
for sub-millisecond workloads and by tens of percent for larger ones. The
tables show one complete run.

<!-- BEGIN REPORT -->
## Statevector throughput (layer = H + RZ on every qubit, CX chain)

| qubits | gates/layer | time/layer | ns per gate·amplitude |
|---:|---:|---:|---:|
| 10 | 29 | 9.8 µs | 0.331 |
| 14 | 41 | 227.9 µs | 0.339 |
| 16 | 47 | 1.07 ms | 0.349 |
| 18 | 53 | 2.59 ms | 0.186 |
| 20 | 59 | 6.97 ms | 0.113 |
| 22 | 65 | 45.45 ms | 0.167 |
| 24 | 71 | 205.84 ms | 0.173 |

## Cost per gate type (22 qubits, 5 passes over all qubits)

| gate | time per pass | ns per gate·amplitude |
|---|---:|---:|
| H | 10.07 ms | 0.109 |
| RY | 8.95 ms | 0.097 |
| RZ | 7.36 ms | 0.080 |
| CX | 14.69 ms | 0.159 |

## Kernels against memory bandwidth (24 qubits, all threads)

Reference: the fastest in-place pass over the 256 MiB state (one, two or four streams per thread) takes 29.37 ms on one thread (18.3 GB/s read + write) and 8.68 ms on all threads (61.8 GB/s).

| gate | qubits 0, 1 | × reference | qubits 21, 20 | × reference |
|---|---:|---:|---:|---:|
| H | 10.48 ms | 1.21 | 8.46 ms | 0.97 |
| RZ | 10.51 ms | 1.21 | 10.72 ms | 1.23 |
| U (complex 2×2) | 12.03 ms | 1.39 | 11.46 ms | 1.32 |
| CX | 14.01 ms | 1.61 | 4.18 ms | 0.48 |
| CZ | 12.38 ms | 1.43 | 2.41 ms | 0.28 |
| CRX | 14.49 ms | 1.67 | 4.51 ms | 0.52 |
| RZZ | 15.74 ms | 1.81 | 10.40 ms | 1.20 |
| SWAP | 11.88 ms | 1.37 | 5.24 ms | 0.60 |
| dense 4×4 | 25.01 ms | 2.88 | 8.53 ms | 0.98 |

## Gradients (TFIM, 10 qubits, EfficientSU2 with 4 layers = 100 parameters)

| method | time per full gradient |
|---|---:|
| adjoint | 525.1 µs |
| parameter shift | 24.68 ms |
| finite differences | 21.00 ms |
| adjoint, 20 qubits (200 parameters) | 696.53 ms |

## Batched evaluation (same circuit and Hamiltonian, 256 parameter vectors)

| quantity | one call per vector | one batch call | speed-up |
|---|---:|---:|---:|
| expectation values | 29.06 ms | 7.68 ms | 3.8× |
| values + adjoint gradients | 132.35 ms | 43.91 ms | 3.0× |

## QAOA value + gradient (p = 3, random 3-regular graphs)

| nodes | time |
|---:|---:|
| 10 | 120.8 µs |
| 14 | 2.62 ms |
| 18 | 27.66 ms |
| 20 | 106.45 ms |

## Quantum walks

| case | time |
|---|---:|
| Metatron's Cube, P(t) (dense spectral) | 0.5 µs |
| 100×100 torus (10 000 nodes), P(t = 5) (Krylov) | 86.65 ms |
| automorphism group of Metatron's Cube (13!) | 3.03 ms |

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

## Noisy circuits (1.4.0)

Release 1.4.0 adds this section to the report. It was measured in a later
session on the same machine type, which was slower at the time: in
alternating runs of the new build and the 1.3.0 build (three each,
medians), a statevector layer took 0.99 ms (new) and 1.15 ms (1.3.0) at
16 qubits, 10.1 ms and 10.6 ms at 20 qubits, and 265 ms and 289 ms at
24 qubits, against 0.55 ms, 8.7 ms and 211 ms in the tables above. The
statevector kernels did not change after 1.3.0.

Device: T₁ = 100, T₂ = 80, gate durations, depolarising errors. Circuit:
four layers of RY on every qubit and a brick of CX. "One trajectory"
compiles the device model for the circuit and runs one trajectory.

| qubits | gates | noiseless | one trajectory | ratio |
|---:|---:|---:|---:|---:|
| 12 | 70 | 101.5 µs | 661.6 µs | 6.5 |
| 16 | 94 | 2.35 ms | 7.11 ms | 3.0 |
| 20 | 118 | 22.62 ms | 72.03 ms | 3.2 |

| 200 runs, 16 qubits | time | per run |
|---|---:|---:|
| noiseless | 375.59 ms | 1.88 ms |
| trajectories (1.3× the noiseless time) | 481.88 ms | 2.41 ms |

| density matrix, same circuit | time |
|---|---:|
| 8 qubits | 46.72 ms |
| 10 qubits | 268.71 ms |

A single trajectory pays for compiling the device model (scheduling,
merging channels, Kraus decompositions) and runs stretches of gates
between jumps as separate fused circuits. Over many trajectories the
compilation is shared and the trajectories run in parallel, so 200
trajectories cost only 1.2–1.3 times as much as 200 noiseless runs. In two other runs
the ratios were 3.1–3.5 at 16–20 qubits and 6–7.5 at 12 qubits, and 200
trajectories took 1.2 times the noiseless time. The density matrix of
10 qubits holds as many amplitudes as a 20-qubit state vector and costs
about ten times as much as one 20-qubit trajectory.

## Comparison with other simulators (1.3.0)

Measured with [`benchmarks/cross_framework`](../benchmarks/cross_framework/)
on the same machine (five interleaved rounds, a fresh process per
measurement, every framework with its default threading and warmed up for
at least 0.25 s; medians, the other framework's time relative to QSO in
brackets). Raw data:
[`results-1.3.0.json`](../benchmarks/cross_framework/results-1.3.0.json).
Versions: Qulacs 0.6.14, Qiskit Aer 0.18.0, PennyLane Lightning 0.45.0,
Cirq 1.7.0 (built-in simulator), The Walrus 0.22.0. Final states agree to
≤ 2.5·10⁻¹⁵, gradients to ≤ 1.4·10⁻¹⁵, permanents to 7.3·10⁻¹⁴ (relative).

| workload | QSO | Qulacs | Qiskit Aer | PennyLane Lightning | Cirq |
|---|---:|---:|---:|---:|---:|
| one layer, 16 qubits | 0.92 ms | 0.71 ms (0.8×) | 4.50 ms (4.9×) | 3.71 ms (4.0×) | 5.63 ms (6.1×) |
| one layer, 20 qubits | 10.2 ms | 15.5 ms (1.5×) | 41.6 ms (4.1×) | 44.0 ms (4.3×) | 37.9 ms (3.7×) |
| one layer, 22 qubits | 49.9 ms | 81.7 ms (1.6×) | 199 ms (4.0×) | 184 ms (3.7×) | — |
| one layer, 24 qubits | 251 ms | 642 ms (2.6×) | 836 ms (3.3×) | 2.31 s (9.2×) | — |
| ten layers, 20 qubits | 89.6 ms | 147 ms (1.6×) | 385 ms (4.3×) | 403 ms (4.5×) | — |
| ten layers, 22 qubits | 381 ms | 697 ms (1.8×) | 1.79 s (4.7×) | 1.71 s (4.5×) | — |
| value + gradient, 10 qubits, 100 parameters | 0.52 ms | 1.99 ms (3.8×, backprop) | — | 19.2 ms (37×, adjoint) | — |

| permanent, n × n | QSO | The Walrus |
|---:|---:|---:|
| 10 | 13 µs | 25 µs (2.0×) |
| 14 | 0.26 ms | 0.48 ms (1.8×) |
| 18 | 1.9 ms | 11.4 ms (6.0×) |
| 20 | 8.6 ms | 47.7 ms (5.5×) |
| 22 | 30.3 ms | 264 ms (8.7×) |
| 24 | 178 ms | 853 ms (4.8×) |

In 1.3.0, from 20 qubits on QSO was the fastest of these simulators. At
16 qubits Qulacs was faster in the median; QSO's 16-qubit times varied
more (fastest run 0.44 ms against Qulacs' 0.60 ms). Since 1.24 QSO is
ahead at 16 qubits too (below).

**Warm-up mattered on virtual machines.** Up to 1.23 QSO's worker threads
(rayon) slept as soon as they were idle, and on this VM waking them is
slow, so the first tens of milliseconds of a burst of work ran at close to
single-thread speed: with three warm-up runs (5 ms) instead of 0.25 s, the
16-qubit layer took 1.57 ms against Qulacs' 0.75 ms. Qulacs' OpenMP
workers spin instead; with `OMP_WAIT_POLICY=passive` its 16-qubit layer
takes 3–4 ms on the same machine. The technical note (Section 3.8) has the
comparison for QSO 1.1.

Since 1.24 the state-vector kernels run on their own worker pool, whose
threads spin for 200 µs after each job before parking. One layer after a
pause (`benchmark_report`, 4 cores, median of 21):

| qubits | warm | after 0.5 ms | after 20 ms |
|---:|---:|---:|---:|
| 12 | 52.5 µs | 57.6 µs | 74.1 µs |
| 14 | 234 µs | 253 µs | 260 µs |
| 16 | 418 µs | 522 µs | 624 µs |
| 18 | 1.65 ms | 1.93 ms | 1.97 ms |

With rayon a 16-qubit layer after 20 ms took 1.16–1.22 ms, and in Python
QSO is now ahead of Qulacs cold and warm from 10 to 18 qubits (the
[roadmap](ROADMAP.md), 1.24, has both comparisons). The cross-validation
times these cold starts (`1@500`, `1@20000`) beside the warmed-up runs.

## Reading the numbers

* **ns per gate·amplitude** normalises a layer's time by gates × 2ⁿ.
  Registers of up to 14 qubits fit into one cache block and are simulated
  gate by gate on one thread. Larger registers are split into blocks that
  run in parallel, and the cost per amplitude drops to 0.14–0.19 ns from
  16 qubits on; at 24 qubits one state vector takes 256 MB.
* **Cost per gate type**: uncontrolled real matrices (H, RY) and the
  diagonal RZ have dedicated vectorisable paths; singly controlled X swaps
  contiguous runs of amplitudes (see [ARCHITECTURE.md](ARCHITECTURE.md)).
* **Kernels against memory bandwidth** (1.11): the reference is the
  fastest in-place pass over the 24-qubit state with one, two or four
  memory streams per thread. A gate that reads and writes every amplitude
  cannot beat 1.0. CX, CZ, CRX and SWAP touch only part of the state and
  can. On high qubits, H, the dense 4×4 matrix, RZ, RZZ and the complex
  U run at 0.97–1.32 of the reference. On qubits 0 and 1 the kernels
  take 1.2–1.8 times the reference. The dense 4×4 matrix on qubits 0 and
  1 takes 2.9 times, because the two lowest qubits leave no contiguous
  runs and its element loop is limited by arithmetic.
* **AVX against the portable kernels** (1.11, same machine, medians of
  repeated runs): the layer of H, RZ and CX takes 9.8 against 12.9 ms at
  20 qubits, and a layer of RY, RZ, CX and RZZ 15.7 against 20.4 ms. The
  1.10 code took 14.6 and 30.3 ms for the same layers. At 24 qubits the
  layers take 222 and 280 ms, against 264 and 414 ms with the 1.10 code.
* **Adjoint differentiation** costs about three circuit evaluations for all
  100 parameters, while parameter shift and finite differences need two
  evaluations per parameter. The ratio grows linearly with the parameter
  count. Since 1.11 the backward pass runs block by block on
  registers larger than a cache block. On 20 qubits with 200 parameters a
  gradient takes 0.7–0.8 s, against 1.76 s with the 1.10 code (2.9× at
  16 qubits).
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
