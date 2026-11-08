# Cross-framework comparison

Checks QSO against established simulators and measures their speed on the
same workloads. The results enter the technical note
([`docs/technical-note/`](../../docs/technical-note/)).

| Framework | Used for |
|---|---|
| Qiskit Aer 0.18 | state vectors, timing; light cones of the kicked Ising circuits (state vectors, and density matrices with Pauli–Lindblad noise) |
| Qulacs 0.6 | state vectors, gradients (`backprop`), timing |
| Cirq 1.7 (built-in simulator) | state vectors, timing up to 20 qubits |
| PennyLane Lightning 0.45 | state vectors, adjoint gradients, timing |
| The Walrus 0.22 | permanents, hafnians, torontonians, Gaussian boson sampling |
| Stim 1.16 | detector error models, detection-event rates; Heisenberg operators of Clifford circuits (`PauliString.before`) |
| qiskit-addon-obp 0.3 | Heisenberg operators of the kicked Ising circuits, string for string, with and without Pauli–Lindblad noise |
| PyMatching 2.4 | logical error rates (minimum-weight matching, for reference) |
| PySCF 2.14 | Gaussian integrals, Hartree–Fock and CASCI energies |
| QuTiP 5.3 | nuclear spin dynamics with relaxation (`liouvillian`, `mesolve`, `bloch_redfield_tensor`); transport to a trap (`liouvillian`, `mesolve`) |
| SymPy 1.14 | transport to a trap in exact rationals (the detuned dimer, K_N on ten orbitals) |
| PyCCE 1.1 | the ¹³C bath of an NV centre by the cluster-correlation expansion (reference only; its last release, imported with aliases for names NumPy, SciPy and pandas removed) |

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
- the Hahn-echo coherence of an NV centre's ¹³C bath (six close ¹³C
  with every pair, and a natural bath of 182 ¹³C within 3 nm with pairs
  within 0.8 nm, 50 mT) against PyCCE's conventional CCE to second order
  on the same positions and hyperfine couplings (3·10⁻⁶ and 7·10⁻⁶:
  PyCCE keeps the full spin-1 electron, QSO the secular Hamiltonian);
- detector error models of repetition and surface-code memory experiments
  (exported with `DetectorCircuit.to_stim()`), term by term against Stim's,
  and their detection-event rates against Stim's sampler; QSO's union-find
  logical error rate is reported next to PyMatching's (minimum-weight
  matching);
- transport to a trap (efficiency η and mean arrival time τ with traps,
  losses, site energies and dephasing per node) of QSO's dense,
  orbital and hypercube solves: against SymPy's exact rationals (the
  detuned dimer at dephasing 0–100, K_N to N = 9 dense and on the ten
  orbitals at N = 64 and 1024, the reduction checked against SymPy's full
  Liouvillian at N = 4–6) and the closed forms on K_N without loss
  (Caruso et al.), to 2·10⁻¹⁴ in η and 4·10⁻¹⁴ in τ (relative); against
  QuTiP's `liouvillian` solved by sparse LU (also random weighted graphs,
  the FMO complex and hypercubes Q₃–Q₅ on the full cube), to 1.7·10⁻¹³
  in η (FMO without dephasing, condition 9·10⁴) and 3·10⁻¹⁴ in τ; and
  against `mesolve` with an accumulator for τ, to 6.7·10⁻¹³ and
  6.8·10⁻¹¹. The dimer's best dephasing rate, 39/20 exactly, is found to
  1.3·10⁻⁷. This check runs in a child process pinned to one thread
  (QuTiP's OpenMP threads on tiny matrices) and takes 6 s (`--quick`)
  or 28 s.
- Pauli propagation (`propagate`) of ⟨Z₅₇⟩ on the 115-qubit kicked Ising
  circuits of Kim et al. on `CouplingMap.from_heavy_hex(7)` (RX(θ_h) on
  every qubit, then RZZ(−π/2) on three colour classes of edges), each
  layer passed as OpenQASM: Heisenberg operators of seeded Clifford
  circuits on 10³ and 10⁴ qubits (3·10⁴ and 1.5·10⁵ gates) and of the
  Clifford points θ_h = 0 and π/2 (1–20 steps) against Stim's
  `PauliString.before`, string and sign exactly; θ_h ∈ {0.3, π/4, 1} at
  1–5 steps against Aer's state vector of the light cone (7.7·10⁻¹⁵) and
  string for string against qiskit-addon-obp (134 744 strings at 5
  steps, 6.3·10⁻¹⁵; obp with its gate tolerance lowered to 10⁻¹⁶ and
  trimming at 10⁻¹⁵); truncated runs at 6 and 7 steps against Aer within
  the reported bound; and scaled sparse Pauli–Lindblad noise after every
  RZZ layer (factors 0, 1, 1.5, 2, −1 and 1 then −1, a uniform and a
  seeded model) against obp and Aer's density matrix (2.1·10⁻¹⁵). It
  takes 12 s (`--quick`) or 62 s, mostly in the references.

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
