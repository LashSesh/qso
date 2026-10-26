# Cross-framework comparison

Checks QSO against established simulators and measures their speed on the
same workloads. The results enter the technical note
([`docs/technical-note/`](../../docs/technical-note/)).

| Framework | Used for |
|---|---|
| Qiskit Aer 0.18 | state vectors, timing |
| Qulacs 0.6 | state vectors, gradients (`backprop`), timing |
| Cirq 1.7 (built-in simulator) | state vectors, timing up to 20 qubits |
| PennyLane Lightning 0.45 | state vectors, adjoint gradients, timing |
| The Walrus 0.22 | permanents |
| Stim 1.16 | detector error models, detection-event rates |
| PyMatching 2.4 | logical error rates (minimum-weight matching, for reference) |

ProjectQ, which the 2025 note listed, is not included: its last release
(0.8.0, 2022) does not build on current CPython.

## What is measured

**Correctness**
- final state vectors of 20 random 8-qubit circuits (60 gates from
  H, X, Y, Z, S, T, RX, RY, RZ, CX, CZ, SWAP);
- energy and gradient of a 10-qubit transverse-field Ising model with a
  100-parameter ansatz;
- permanents of random complex matrices up to 12 × 12;
- detector error models of repetition and surface-code memory experiments
  (exported with `DetectorCircuit.to_stim()`), term by term against Stim's,
  and their detection-event rates against Stim's sampler; QSO's union-find
  logical error rate is reported next to PyMatching's (minimum-weight
  matching).

**Speed** (interleaved rounds, fresh process per measurement)
- one layer (H and RZ on every qubit, CX chain) at 16–24 qubits;
- ten layers at 20 and 22 qubits;
- value and gradient of the Ising energy above;
- permanents up to 24 × 24.

Every framework runs with its default threading and is warmed up for at
least 0.25 s (and at least one to three runs) before timing: on virtual
machines, waking sleeping worker threads is slow, and a thread pool whose
workers sleep as soon as they are idle (rayon, used by QSO) reaches full
speed only after tens of milliseconds of sustained work, while OpenMP
workers (as in Qulacs) spin and are fast at once. Circuit construction
and Aer's transpilation (at `optimization_level=0`, which keeps the gate
list unchanged) are not timed. On shared machines timings fluctuate in bursts;
the interleaved rounds spread the load over all frameworks, and the minimum
over all runs is reported next to the median.

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin -r benchmarks/cross_framework/requirements.txt
maturin build --release -m metatron_qso_py/Cargo.toml -o dist && pip install dist/*.whl
python benchmarks/cross_framework/run.py --out cross_framework.json            # ~15 min
python benchmarks/cross_framework/run.py --out cross_framework.json --quick    # smoke test
```

The *Benchmarks* workflow (`.github/workflows/benchmarks.yml`, started
manually) runs this together with `examples/technical_note.rs` and uploads
both JSON files.
