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

<!-- BEGIN REPORT -->
<!-- END REPORT -->

## Reading the numbers

* **ns per gate·amplitude** normalises a layer's time by gates × 2ⁿ.
  Small registers fit in cache and run single-threaded (parallelism starts
  at 2¹⁶ amplitudes). From about 20 qubits the simulator is limited by memory
  bandwidth: a gate reads and writes every amplitude (32 bytes), so
  0.7 ns per amplitude corresponds to ≈ 45 GB/s.
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
