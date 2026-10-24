# Audit of QSO v0.1 (claims vs. reality)

This document records why QSO 2.0 is a rewrite rather than an incremental
release. Every finding below was verified against the v0.1 source tree
(commit `b688a23`, still available in the git history); file references are
to that tree. The audit was carried out in October 2026.

**Summary.** v0.1 compiled and its 716 workspace tests passed, but the tests
mostly asserted `is_finite()`/`is_ok()` or echoed constructor arguments. The
central physics was either trivial or wrong, several subsystems returned
hard-coded or placeholder data, and the headline numbers in the README were
not reproducible.

## 1. Core library (`metatron-qso-rs`)

| Claim | Finding |
|---|---|
| "13-dimensional Metatron Cube with 78 edges, embedding all 5 Platonic solids, symmetry group G_M" | 13 nodes with 78 edges is C(13,2): the graph is the **complete graph K₁₃** (every degree is 12). Its automorphism group is the full symmetric group S₁₃ (order 13! = 6 227 020 800). The hand-written `enumerate_automorphisms` sampled a few permutations and reported a small "group". A 20-vertex dodecahedron cannot embed in a 13-vertex graph. |
| "Quantum computing framework", "13 physical qubits" | The Hilbert space was **13-dimensional** (one 13-level system, ≈ 3.7 qubits), not 13 qubits (8192 amplitudes). Ansatz "qubits" were basis states. |
| VQE "E₀ = −12.9997" | H = −L(K₁₃) has eigenvalues {0, −13 (12-fold)}. Any state orthogonal to the uniform superposition is a ground state, and exact diagonalisation of a 13×13 matrix is instantaneous. The optimiser did not even reach −13 exactly. |
| QAOA MaxCut, "ratio 0.9974" | The "MaxCut Hamiltonian" was −L/2 of the problem graph on the 13-dim space, not an Ising cost on 2ⁿ states. The mixer was an anti-diagonal 13×13 matrix, `exp(−iHt)` was approximated by the non-unitary `I − iHt`, and the "classical optimum" was `1.5 × ⟨uniform|H|uniform⟩`. 0.9974 was actually a VQE quality score. |
| Quantum-walk mixing baseline | The recorded time-averaged distribution on K₁₃ gave nodes 1–12 *different* probabilities, which symmetry forbids (correct: 145/169 at the start node, 2/169 elsewhere). Cause: degenerate eigenspaces were ignored. |
| Quantum "hitting time speedup" | Probabilities at successive times were treated as independent measurements without collapse and then compared with classical step counts. |
| Grover search | Success probability ≈ 0.075 ≈ 1/13 (no amplification). |
| Boson sampling | Single-particle walk sampling only; multi-photon path returned an error. Benchmark figures 0.75/0.65 were literals. |
| "Topological code, d ≥ 6" | No code construction with a verified distance exists. |
| DTL "58.5 % information advantage" | log₂3 / log₂2 − 1: a three-state symbol carries log₂3 ≈ 1.585 bits. This is a definition, not an advantage. |
| Cross-system comparison | Qiskit/Cirq/PennyLane/ProjectQ numbers were hard-coded constants. |
| "31 941 ops/sec" | Appears in no baseline or benchmark output. |

## 2. Peripheral subsystems

* **`metatron_backend`** — the "local simulator" ignored the circuit and
  always returned `|0⟩` of the 13-dim space. There was no qubit gate math in
  the repository. The IBM backend was a stub (`bail!("not yet implemented")`)
  with fabricated dry-run counts and an environment-variable mapping that
  silently dropped `IBM_BACKEND_NAME`/`IBM_MAX_SHOTS`.
* **Python SDK** — `pyproject.toml` lacked `module-name`, so `import
  metatron_qso` failed after `maturin develop`; every example and the
  notebook were therefore broken. `quantum_walk_centrality` returned all
  ones for every graph (column sums of a unitary), so anomaly scores were
  always 0. `from_adjacency` ignored its input.
* **SCS (Seraphic Calibration Shell)** — never executed a benchmark. Every
  step re-read static JSON files, scored candidates with fixed bonuses
  (+0.02 for a "Metatron" ansatz, +0.01 for Adam, …) and accepted them via
  `psi_new >= psi_old`, which holds by construction ("Proof-of-Resonance").
  The 16-dim "Mandorla field" had no influence on any decision. `step -n 11`
  crashed and truncated the state file. The CI workflow committed the
  fabricated state back to the repository.
* **TRITON** — golden-angle offsets span only a 2-D plane in any dimension.
  In a faithful port it performed no better than random search (median
  distance to optimum 0.326 vs 0.324 for d = 6; a (1+1)-ES reached 0.005).
  Its SCS adapter used a dummy evaluator returning (0.5, 0.5, 0.5).
* **DioniceOS (`external/dioniceos`, ≈ 58 k lines)** — no quantum content.
  The only reachable path (`trichter`) was effectively an identity map; "zk
  inference" returned `{"hash": sha256(input), "valid": true}`;
  "Proof-of-Resonance" was a distance plus cosine similarity; determinism
  claims were violated by `HashMap` iteration order and `thread_rng`. About
  22 k lines were orphan crates outside the workspace. A real SHA-256 hash
  chain (mef-ledger) was the only competent piece, and it is unrelated to
  quantum simulation. The ledger already names its own repository
  (`LashSesh/infinityledger`).
* **Telemetry dashboard** — all data was synthetic (50 demo points on linear
  ramps, "calibration" added constants after `sleep(3)`, energies were
  `−1.137 + rand·0.01`). CORS allowed any origin on unauthenticated POSTs.
  The TOML config was never actually applied.

## 3. Build, CI and documentation

* Release workflow failed (`cp -r target/doc`), no wheels were uploaded;
  the SCS workflow used the retired `upload-artifact@v3`.
* `benchmark_compare` passed on unknown benchmark types and counted missing
  metrics as "stable"; several baselines missed the project's own policy
  targets.
* README Rust examples did not compile (`QSO`, `VQE::builder`,
  `MaxCutProblem`, `ContinuousQuantumWalk` do not exist; wrong crate name).
  At least ten referenced documents did not exist; the roadmap marked
  non-existent deliverables as complete.
* Counts were inflated ("24 crates" vs 21; "17 200 lines of Python bindings"
  vs 510). nalgebra was compiled in two versions; editions/MSRV were
  inconsistent.

## 4. Consequences for 2.0

Nothing from v0.1 was kept verbatim. The 2.0 plan
([QSO_2.0_PLAN.md](QSO_2.0_PLAN.md)) replaces the 13-dim toy with a real
n-qubit simulator, keeps Metatron's Cube as an honestly analysed showcase
graph, and makes every claim in the documentation traceable to a test or a
reproducible benchmark.
