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
| The Walrus 0.22 | permanents, hafnians, torontonians, Gaussian boson sampling |
| Stim 1.16 | detector error models, detection-event rates |
| PyMatching 2.4 | logical error rates (minimum-weight matching, for reference) |
| PySCF 2.14 | Gaussian integrals, Hartree–Fock and CASCI energies |
| QuTiP 5.3 | nuclear spin dynamics with relaxation (`liouvillian`, `mesolve`, `bloch_redfield_tensor`) |

ProjectQ, which the 2025 note listed, is not included: its last release
(0.8.0, 2022) does not build on current CPython.

## What is measured

**Correctness**
- final state vectors of 20 random 8-qubit circuits (60 gates from
  H, X, Y, Z, S, T, RX, RY, RZ, CX, CZ, SWAP);
- energy and gradient of a 10-qubit transverse-field Ising model with a
  100-parameter ansatz;
- permanents of random complex matrices up to 12 × 12;
- hafnians up to 20 × 20, torontonians, and photon-number and click
  probabilities of Gaussian states (also at 12 and 20 photons);
- overlap, kinetic, nuclear-attraction and electron-repulsion integrals
  of H₂O and NH₃ in STO-3G and 6-31G, and restricted, unrestricted and
  frozen-core CASCI energies of H₂O, BeH₂, OH and triplet CH₂;
- nuclear spin dynamics with T₁/T₂ relaxation of five spin systems (three
  protons at 0.05 K; ¹H–¹H–¹³C–¹⁵N after INEPT, five protons and two
  protons in the laboratory frame, all at 9.4 T and 298 K; SABRE-SHEATH in
  the laboratory frame), built independently with QuTiP's spin operators
  (Hamiltonian, thermal state, pulses, Lindblad operators): ⟨I_z⟩ of every
  spin and ⟨F_x⟩, ⟨F_y⟩ of every isotope, relative to the largest, against
  the exact exponential of QuTiP's `liouvillian` and against `mesolve`;
- Redfield relaxation of ¹³C–¹H₂, ¹³CH₃ and an amide ¹⁵N–¹H with
  dipolar–CSA cross-correlation in the laboratory frame, against QuTiP's
  non-secular `bloch_redfield_tensor` with bath operators from an exact
  numerical average over orientations: the rates among I_z, I_+ and
  two-spin orders, and expectations after exact propagation;
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
workers sleep as soon as they are idle (rayon, which QSO used for its
kernels up to 1.23) reaches full speed only after tens of milliseconds of
sustained work, while OpenMP workers (as in Qulacs) spin and are fast at
once. The configurations `1@500` and `1@20000` time that cost: QSO,
Qulacs and Aer at 10–18 qubits, every timed layer after a pause of 0.5 or
20 ms (the first simulation after an optimiser step or Python code). Circuit construction
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
