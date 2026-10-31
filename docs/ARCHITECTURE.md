# Architecture

This document explains how QSO is built and why. API details live in
the rustdoc (`cargo doc -p metatron-qso-rs --open`).

## Layers

```
            ┌─────── qso CLI ─── Python (PyO3) ─── WebAssembly ─┐
chemistry   │ fermion operators · mappings · FCIDUMP · full CI · │
            │ UCCSD / k-UpCCGSD · ADAPT-VQE                      │
algorithms  │ VQE · QAOA (+ mixers, QUBO) · Grover · QFT/QPE ·   │
            │ excited states · geometry · time evolution ·       │
            │ amplitude estimation · classifier · models         │
            │ gradients (adjoint / parameter shift / FD)         │
optimize    │ L-BFGS · Adam · GD · Nelder–Mead · SPSA            │
walks       │ CTQW · quotients · transfer · coined/Szegedy ·     │
            │ search · walk circuits · dephasing · classical ·   │
            │ interacting walkers · orbitals (open walks)        │
mitigation  │ readout calibration · ZNE · PEC · Pauli twirling    │
estimation  │ estimates with errors · shot allocation · shadows   │
backend     │ statevector / density-matrix / device backends      │
quantum     │ StateVector · Circuit · plans (fusion, blocking) · │
            │ Program · PauliSum · OpenQASM 2/3 · decompositions │
noise       │ DensityMatrix · channels · device models ·          │
            │ compiled noise programs · quantum trajectories      │
graph       │ Graph · generators · Metatron · automorphisms ·    │
            │ canonical forms · edge-list/GraphML I/O            │
linalg      │ dense eigh/expm · CSR · Lanczos · Krylov           │
infra       │ error · rng (xoshiro256++) · parallel (rayon)       │
            └────────────────────────────────────────────────────┘
```

Lower layers never depend on higher ones.

## Simulation kernels (`quantum/kernels.rs`)

Amplitude `k` stores basis state `|b_{n−1} … b_0⟩` with `k = Σ b_q 2^q`.
A gate on targets `T` with controls `C` touches groups of `2^|T|` amplitudes
whose indices differ only in the target bits and have all control bits set.
The kernels enumerate those groups by *inserting zero bits*: for
`k ∈ 0 … 2^{n−|T|−|C|}`, insert zeros at the sorted positions of `T ∪ C`,
then OR in the control mask. Distinct `k` give disjoint groups, so the loop
parallelises without locks (`SyncPtr` documents this safety contract).

Since 1.11 the arithmetic runs on *runs*: amplitudes whose bits at the
gate's positions are fixed form contiguous runs of `2^p` amplitudes, `p`
the lowest position. The kernels hand pairs of runs (a 2×2 matrix),
single runs (a phase) or four runs (a 4×4 matrix) to the primitives of
`quantum/simd.rs`. Those have a portable version and vector versions:
AVX on x86-64, detected once at run time, with two amplitudes per
register, and NEON on aarch64 with one. A complex product is
`(a·c − b·d, a·d + b·c)` with separate multiplications and additions (AVX
`mul` and `addsub`; NEON adds the negated term), in the same order as
the scalar code, so every version gives the same bits. `set_simd(false)`
forces the portable path, and a test compares the two on the one- and
two-qubit kernels, phases and RZZ, on small registers and on 17 qubits.
Uncontrolled one-qubit gates on low qubits are handled a chunk at a time
(adjacent pairs for qubit 0). Controlled gates, multi-qubit phases and
two-qubit gates whose lowest qubit (target or control) is 0 or 1 have runs
shorter than four amplitudes and keep the element loops, where a call per
run would cost more than it saves.

Specialised paths:

* uncontrolled single-qubit gates process a whole chunk per call when
  their stride is small and pairs of runs otherwise; diagonal matrices
  (`RZ`) scale each half without branching, and real matrices (`H`,
  `RY`) use real instead of complex products;
* `RZZ` reads the lower and upper halves of the register in step, and
  `RZ` on a high qubit the two halves of each block (pieces of 512
  amplitudes): the memory system serves two streams faster than one,
  about 64 against 44 GB/s on the test machine; `Z, S, T, P` scale only
  the half they act on;
* `X`, `SWAP` and their controlled forms are permutations (no arithmetic);
  a singly controlled `X` swaps contiguous runs of `2^min(c, t)`
  amplitudes, split into fixed pieces so that gates on high qubits
  parallelise too;
* `Z, S, T, P, RZZ` and the QAOA cost layer are diagonal phase
  multiplications.

A new register is written in parallel, so its memory pages are faulted in
by all threads rather than inside the first gate.

These paths came out of a comparison with Qulacs and Qiskit Aer (see
[BENCHMARKS.md](BENCHMARKS.md)), which traced the remaining gap to the
generic 2×2 kernel for `H` and the per-index loop for `CX`.

The kernels never assume unitarity. The same code therefore applies

* derivative matrices `∂U/∂θ` in adjoint differentiation, and
* superoperators in the density-matrix simulator: with
  `vec(ρ)[r + (c ≪ n)] = ρ[r, c]`, `ρ ↦ UρU†` is `U ⊗ U*` acting on a
  2n-qubit vector, i.e. the gate applied to the row qubits plus its complex
  conjugate applied to the column qubits (offset `n`). A one-qubit channel
  `{K_i}` becomes the 4×4 matrix `Σ K_i ⊗ K_i*` on the (row, column) bit pair.

Parallelism inside a gate starts at 2¹⁶ amplitudes; below that rayon's
scheduling overhead exceeds the work of a gate (measured, see
[BENCHMARKS.md](BENCHMARKS.md)).

## Execution plans (`quantum/plan.rs`)

`StateVector::apply_circuit` does not apply gates one by one once the
register exceeds 2¹⁴ amplitudes (256 KiB, a comfortable fit in a per-core L2
cache). It first fuses runs of uncontrolled single-qubit gates on qubits
≥ 14 into one matrix, then splits the circuit into runs of *block-local*
operations: gates whose targets all lie below qubit 14 (their controls may
lie anywhere — on a block, the high control bits are fixed by the block's
index), and diagonal gates anywhere (on a block, the high qubits contribute
a constant phase or select a branch). A run is applied block by block: each
block of 2¹⁴ amplitudes stays in cache for all gates of the run, and the
blocks are processed in parallel. Other gates are applied as before, with
parallelism inside the gate. Registers of at most 2¹⁴ amplitudes are
simulated gate by gate on one thread.

Two alternatives were measured and dropped: fusing gates *inside* a block
(the specialised real, diagonal and permutation kernels beat a fused
complex matrix once the data is in cache), and gathering a few high qubits
into each block (qHiPSTER-style blocking, with a greedy scheduler: not
faster on the 4-core test machine). A random-circuit test over the whole
gate set (every kind, unitaries on up to three targets, controls, symbolic
angles, fusible runs) compares plans with gate-by-gate simulation, with
the production block size on 15 and 16 qubits and with small blocks down
to 2 amplitudes, also under Miri with 8-amplitude blocks.

## Multi-controlled gates in OpenQASM (`quantum/decompose.rs`)

`qelib1.inc` has at most two controls (`ccx`). Export decomposes larger
controlled gates exactly and without ancillas (Barenco et al. 1995):
`C^k(X)` with `k − 2` idle qubits as dirty ancillas takes `4(k − 2)`
Toffolis (Lemma 7.2), with one idle qubit two halves (Lemma 7.3); `C^k(U)`
becomes `C(V)·C^{k−1}(X)·C(V†)·C^{k−1}(X)·C^{k−1}(V)` with `V² = U`
(Lemma 7.5), the target serving as idle qubit — `O(k²)` gates in all.

## Programs and OpenQASM (`quantum/program.rs`, `quantum/openqasm.rs`)

A `Program` is a list of operations: gates, measurements into classical
bits, resets, and blocks conditioned on bits. `probabilities` computes the
exact distribution of the classical record by carrying a list of branches,
each a weight, a normalised state and the bits written so far. A
measurement splits every branch into its two outcomes (dropping outcomes of
probability zero), a reset splits and flips, and a condition runs its block
on the branches whose record satisfies it. The number of branches and the
amplitudes they hold are bounded (4096 branches, 2²⁶ amplitudes). When a
program is gates followed by final measurements, one state vector suffices
and no branches are made. Beyond the bounds, `sample` simulates shot by
shot.

One recursive-descent parser reads both OpenQASM versions; the header
selects the version-specific rules (built-in gates, `U`'s phase,
statement forms). Gate definitions are expanded at the call site, with
modifiers applied while expanding: `inv` reverses and inverts the body,
`pow(k)` repeats it, and `ctrl`/`negctrl` add controls to every gate of
the body, turning the body's global phases (`gphase`, and the phases by
which OpenQASM 3's `U`, `u2` and `u3` differ from QSO's gates) into
controlled phases. The phase of the whole program is dropped. Expansion
counts operations and nesting depth, so hostile input fails with an error
instead of exhausting memory or the stack.

## Stabilizer tableaux (`stabilizer/`)

A stabilizer state of n qubits is stored as Aaronson and Gottesman's
tableau: n destabilizer and n stabilizer generators, each a signed Pauli
string. The bits are stored by qubit: column `q` of the X part (and of
the Z part) is a bit vector over the 2n rows, and the signs are one more
such vector. A gate on qubits `a` and `b` is then a handful of word
operations on two or four contiguous columns, O(n/32) words in all.

Measurements need whole rows. `measure_pauli` transposes the tableau to
rows in 64 × 64 bit blocks, runs the Aaronson–Gottesman update (row
products with their phases summed word by word), and transposes back.
The transpositions dominate: O(n²) per measurement, about a second at
16 384 qubits. Programs that end in measurements avoid them per shot: the
record is a linear function of the outcomes, so QSO samples the image of
the outcome space, at most 64 basis vectors.

Sampling every qubit uses the structure of stabilizer states. Gaussian
elimination of the stabilizers on their X parts leaves signed Z strings,
which are parity constraints on the outcome. The outcomes are uniform
over the solutions: a particular solution plus the span of a null-space
basis. This is computed once, O(n³/64), and each shot is then a random
combination of basis vectors.

Graph states are built directly from their generators
`X_v Π_{w∈N(v)} Z_w`, with destabilizers `Z_v`. Entanglement entropy
across a bipartition is the GF(2) rank of the generators restricted to
one side minus its size (Fattal et al. 2004); for graph states this is
the cut-rank of the adjacency matrix.

## Error-correction experiments (`qec/`)

A `DetectorCircuit` is a list of operations: Clifford gates, Pauli noise,
measurements (each appends one bit to the record), resets, detectors and
observables (parities of record bits). Three components share one
representation of Pauli frames, `Frames`: X and Z bits by qubit, `w`
words per qubit, bit s of word k belonging to frame 64k + s. `Frames`
implements the same `CliffordTarget` trait as the tableau, so both run
every Clifford gate through one function, `apply_clifford`, and the
frame update for each gate is derived once.

- **Sampling.** Each shot's frame is its error relative to a noiseless
  run. Detectors are deterministic, so detection events are parities of
  measurement flips, and the reference run is never needed. The Z part
  of every fresh or measured qubit is randomized, which turns the random
  outcomes of the reference into random flips. Noise is applied by
  geometric skipping, at a cost proportional to the number of errors.
- **Error models.** Every noise operation contributes an X and a Z
  component per qubit, and every noisy measurement a flip. The components
  are propagated, without randomization, as up to 4096 frames per pass;
  the detector and observable rows of the pass are read back as each
  component's signature. Independent error terms (the exact decomposition
  of depolarizing channels into Paulis) take the XOR of their components'
  signatures, and terms with equal signatures merge.
- **Decoding.** The matching graph has a node per detector plus one
  boundary node. Union-find growth is event-driven: each round finds the
  frontier edges of the growing clusters, advances all of them by the
  smallest remaining length (divided by the number of clusters growing
  into the edge) and merges along the edges that complete, so weights are
  exact. Nodes whose edges are all inside their cluster are skipped from
  then on. The decoder keeps all its
  arrays across shots and resets only what a shot touched, so a shot
  costs time in proportion to its clusters, not to the graph. Peeling
  roots a cluster at the boundary only when its parity needs it.
- **Matching.** The matching decoder pairs events by savings
  s_ij = b_i + b_j − d_ij against the boundary (b: distance to the
  boundary, precomputed by one Dijkstra from the boundary node). A
  maximum-weight matching on the positive savings is a minimum-weight
  correction, and pairs with d_ij ≥ b_i + b_j are never needed, which
  bounds each search. Events without a path to the boundary pay a
  stand-in cost larger than all their pairs together, so they are matched
  whenever possible. Distance rows are cached per detector up to 2²¹
  entries. The blossom algorithm is van Rantwijk's O(n³) primal-dual
  method on integer weights (largest edge 2²⁰), so its dual updates are
  exact.
- **Stim import.** The text is parsed into a tree of instructions and
  REPEAT blocks. The expanded operation count and the largest qubit are
  computed before anything is built, so a short text cannot expand
  beyond the limits. Other bases are rotated to Z around measurements
  and resets.
- **Logical error rates.** `logical_error_rate` samples in blocks of
  whole batches (at most 65 536 shots and 1 GiB of results) from one
  random stream, so it returns the same result as one call of
  `sample_detectors` with the same seed, in bounded memory. A batch is
  1024 shots, or fewer when its frames and records would exceed 128 MiB.
- **Exact circuit distance** (`qec/distance.rs`,
  `qec/distance/zimmermann.rs`). The lower bound rises in steps of two
  kinds, chosen by their estimated cost (`SEARCH_NANOS` per set visited,
  `ENUMERATION_NANOS` per word operation). The search's next step is
  extrapolated to the weight it will search, by the growth per weight of
  its last two steps; the enumeration may have raised the bound by
  several since.
  - The search of 1.22 proves that no logical error of the current
    bound's weight exists. Its first errors are shared out between
    threads in tasks of `n / (16 · threads)`, each of which records, per
    first error, the sets it visited and what it found. The answer is
    read off these records in index order, as the sequential search
    would have met them: the set found from the lowest first error, and
    the work counted up to it. Where the threads stopped short (at the
    work limit), the search goes on alone from the first unfinished first
    error. So the bounds and the witness do not depend on the threads,
    the work limit included. A step in which no branch was cut by the
    weight proves that no logical error exists. Each walker keeps one
    buffer of fired detectors per depth and allocates nothing while it
    walks.
  - A level of the Brouwer–Zimmermann enumeration. The generator
    matrices are built only when, at their best (disjoint information
    sets of full rank), the levels could cost less than the search. The
    kernel of `H` comes from its reduced row echelon form, systematic on
    the free columns. Further information sets come from eliminating on
    the columns not yet used, as long as their rank deficit is at most
    64. Each level enumerates the sums of `w` rows by depth-first
    search, one task per first row, with a running XOR per depth. A
    matrix of rank `r < k` joins at level `k − r` and then enumerates
    every lower level once; the bound needs them.
  - Both steps check the deadline: the search every 4096 sets, the
    enumeration every 2²⁰ word operations, in every task. A step it
    interrupts leaves the bounds where they were.
- **Circuit locations.** `explain_errors` replays each noise channel's
  components (`dem::each_channel`, also used to build the model). For
  every combination of a channel's Paulis it takes the XOR of their
  signatures and keeps those equal to the error's effect, as
  `CircuitFault`s with the operation index.

## Matrix-product states (`mps/`)

Site q holds qubit q as a tensor `l × 2 × r`, stored row-major with index
`(a·2 + s)·r + b`. The same buffer is then the `(2l) × r` matrix (for
moving the centre right and for contracting with the next site) and the
`l × (2r)` matrix (for moving it left), so no reshuffling is needed.

- **Canonical form.** Sites left of the orthogonality centre are
  left-canonical and sites right of it right-canonical, kept by
  Householder QR when the centre moves. One-qubit gates act on a site in
  place and keep the form.
- **Gates.** A gate on k qubits moves the centre into the block, routes
  the qubits next to each other by swaps, and contracts the k sites into
  one tensor `θ` of shape `(l·2ᵏ) × r`. It applies the gate (its matrix
  reordered from gate order to site order) and splits θ from the left;
  the swaps are then undone. Keeping qubit q at site q keeps
  nearest-neighbour circuits free of swaps.
- **Splits.** A split of an `m × n` matrix M takes the eigendecomposition
  of the smaller Gram matrix (MM† or M†M) and chooses the kept subspace
  by the options. It orthonormalises the subspace's basis by Householder
  QR into an isometry Q and keeps F = Q†M. The discarded weight is
  ‖M − QF‖²/‖M‖², measured rather than estimated, so the error accounting
  is exact even when the eigenvectors are not; computing it from the
  residual instead of as 1 − ‖F‖²/‖M‖² avoids a cancellation that would
  charge every exact split 10⁻¹⁶. A split computes all new tensors before
  it writes any, so a split beyond the memory limit leaves the state as
  it was. nalgebra's complex SVD was not used because it fails on
  rank-deficient matrices.
- **Fast products.** Complex matrix products run as three real products
  (Gauss's trick) on nalgebra's optimised real kernels; this is about
  five times faster than nalgebra's generic complex product.
- **Readout.** Expectation values contract from the first to the last
  factor of the string, with the centre moved into that range. Sampling
  moves the centre to site 0, so that every other site is right-canonical
  and a prefix's probability is the norm² of its left vector.

## WebAssembly (`metatron_qso_wasm`)

The core has no platform code beyond optional rayon, so it compiles for
`wasm32` without the `parallel` feature. `MAX_QUBITS` follows the pointer
width: 26 qubits (1 GiB of amplitudes) in a 32-bit address space.

The module exposes one function. A JavaScript caller allocates a buffer in
the module's memory (`qso_alloc`), writes a UTF-8 JSON request, calls
`qso_call`, reads the response (`qso_response_len`) and frees it
(`qso_free`). This needs no generated bindings and no imports, so the
module runs in any WebAssembly host, and the JavaScript wrapper is a page
of code. Every parameter is validated and unknown parameters are errors.
A panic aborts the module; the wrapper then replaces the instance.

## Device models and trajectories (`noise/`)

A `DeviceModel` is compiled per circuit into a program of ideal gate
segments and channels on one to four qubits (`noise/device.rs`). The
compiler does four things:

* it schedules gates as soon as their qubits (and the spectators of
  their rule) are free, inserting relaxation for waiting qubits and for
  each gate's duration, and, for qubits with slow noise, a phase step
  for the same time (consecutive ones of a qubit joined);
* it applies the first matching gate rule;
* it splits thermal relaxation into amplitude damping and a phase flip
  (the same channel);
* it merges consecutive channels of one kind on the same qubits. Pauli
  channels merge by convolving their probabilities, other channels get a
  minimal Kraus set from the Choi matrix.

Density matrices apply the program with the state-vector kernels on the
vectorised ρ and multiply consecutive superoperators first. A k-qubit
channel is a `4ᵏ × 4ᵏ` superoperator on the k row and k column bits; four
qubits need eight targets, which `apply_kq` handles with a larger buffer
than gates use. Programs with slow noise do not run on density matrices.

Slow noise is part of a trajectory's random choices, drawn before it
runs: the over-rotation spread of every rule, the detuning of every
qubit, and a telegraph process per qubit, integrated exactly (exponential
waiting times) over the phase steps in time order. A phase step becomes
an `RZ` of that angle and a spread gate its rotation with scaled angles,
inside the fused blocks. Markovian programs draw nothing extra, so their
trajectories are unchanged.

Trajectories (`noise/trajectories.rs`) use the waiting-time form of the
Monte-Carlo wave-function method.

* **Unitary mixtures** (Pauli, depolarising, dephasing, coherent errors)
  are sampled before the run, because their choice does not depend on
  the state.
* **Other channels** apply their dominant Kraus operator without
  renormalising. The squared norm of the state is then the probability
  that no jump has happened. A jump happens where it falls below a
  uniform random threshold.
* **Execution.** A stretch without jumps runs as one fused, cache-blocked
  circuit through the execution plans, with the non-unitary no-jump
  operators as ordinary local matrices. When the norm ends below the
  threshold, bisection over checkpoints finds the step of the jump. The
  last few steps are simulated one by one, with Born weights from the
  reduced density matrix of the channel's qubits.

Each trajectory owns a stream `Rng::fork(k)`, and results are collected in
trajectory order, so they do not depend on the number of threads.

## Estimation and mitigation (`estimation/`, `mitigation/`)

Shot-based estimates compute one value per shot for every measurement
group: the sum of the group's terms on the measured bits. The standard
error follows from the spread of these values. With a readout
calibration, every term gets exact per-outcome weights. A tensored
calibration factorises them over qubits, and a full one gets `A⁻ᵀ s`. The
mitigated estimate stays a mean over shots, so its standard error
includes the cost of mitigation.

Each mitigation method returns its estimate with a propagated error:

* zero-noise extrapolation is a linear combination of the folded values
  (linearised for the exponential fit);
* probabilistic error cancellation samples Pauli corrections from the
  quasi-probability inverse of every Pauli channel in the compiled
  program, and evaluates each sample with a density matrix or one
  trajectory.

Noise learning (`noise/learning.rs`) runs its circuits on density
matrices up to ten qubits and on Pauli frames above. A learning circuit
is Clifford, and without noise it returns a fixed outcome, because the
basis changes and the self-inverse layers cancel. So a shot only needs
its frame: Pauli gates and sampled errors flip X and Z bits, other
Clifford gates conjugate them. The frames run through the compiled
program, with every channel replaced by its Pauli twirl and errors
placed by geometric gaps over the shots. The rate fit is non-negative
least squares. Up to 64 generators it is Lawson–Hanson with an SVD per
subproblem. Above, it starts from coordinate descent on the normal
equations and finishes with the same active-set iteration on the
support found, with Cholesky factors of the Gram matrix. The samples of
layered PEC and PEA are density matrices, trajectories, or
matrix-product states on which every channel, a mixture of unitaries,
is one of its unitaries.

## Spin dynamics (`spin/`)

A `SpinSystem` works on dense density matrices over the product basis of
its spins (local dimension 2 or 3). Hamiltonians are built entry by entry
from the basis states (Zeeman and `I_z I_z` terms on the diagonal,
flip-flop terms off it), so building one costs `O(d · couplings)`. Pulses
act spin by spin: a local `k × k` rotation is applied to the rows and
columns of ρ without forming the `d × d` operator.

Everything else uses the block structure (`spin/blocks.rs`). Every basis
state carries a label, `2M` of each isotope in the rotating frame or the
total `2M` in the laboratory frame, which `H` conserves. Each block of `H`
is shifted by its mean diagonal and diagonalised on its own; spectra take
the pairs of blocks that `F₊` connects, time averages compare energies
with the carriers added back, and free evolution conjugates block by
block (complex products as three real GEMMs). With relaxation, every jump
acts on one spin and keeps the coherence order `label(a) − label(b)` of
`|a⟩⟨b|`, so the Lindblad superoperator splits into blocks by order. The
state is propagated as its deviation from `1/d`, with `L(1/d)` as an
augmented column of the population block; coherence blocks are shifted by
their mean diagonal first. Only the blocks the deviation occupies (and
the population block, if it drifts) move. Each block is planned before
anything is built: its entries are counted and its `‖A‖∞` bounded column
by column from the sparse `H` and the local jumps (and the Redfield
terms, below), which gives the work of a dense exponential (Taylor degree
16 by Paterson–Stockmeyer, scaling and squaring, in
`linalg::dense::expm`; up to 2048 elements) and of Krylov steps
(`spin/krylov.rs`), and the memory of either. The cheaper runs; all the
blocks together are compared with Dormand–Prince 5(4) steps on the whole
state (sparse `H`, jumps applied in place, error relative to the
deviation), and more than `MAX_RELAXATION_WORK` operations or
`MAX_RELAXATION_MEMORY` bytes is refused. A block is then built column by
column (in parallel, each column merged in a dense accumulator) into a
compressed sparse row matrix. Krylov steps are Arnoldi's process on
subspaces of dimension 30 with Expokit's error estimate and step control
(Sidje 1998), to 10⁻¹² of the propagated vector in the 2-norm; the
augmented entry is set to the size of the deviation (or of equilibrium's,
if larger), so that the tolerance stays relative to it. A Krylov space
that is invariant to the tolerance ends the propagation in one step. So
do populations that move no faster than the rounding of the state
explains (forming the deviation leaves `ε·Tr(ρ)/d` on each population) or
by less than 10⁻¹¹ of themselves over the interval: when the a priori
estimate is large, `‖A y‖` of their start vector is computed first, so
that equilibrium is not refused for the steps it will not take. Blocks
are built with one dense accumulator per task (about four per thread),
which the memory estimate counts. FIDs need only the block of order `−1`:
one exponential for the dwell time and a matrix-vector product per point,
or Krylov steps of one dwell time each. Shaped pulses mix the orders and
use one block. Sweeps are piecewise constant, with the field at the
middle of each piece.

Declared symmetries (`spin/symmetry.rs`) are permutations of the spins,
checked against isotopes, shifts, relaxation times, couplings and the
Gram matrix `A^k : A^l` of the relaxing interactions (on which isotropic
tumbling makes `R` depend). They induce permutations of the basis states
and of each block's elements. When `H` (to 10⁻¹² of its largest entry)
and the block's start vector (to 10⁻¹¹) are invariant, union–find gives
the orbits of the elements; after the built block is checked entry by
entry against its images, it is reduced to `M_OP = Σ_{c ∈ P} A_{o,c}` at
a representative `o` of each orbit `O`, and the result is lifted back.
Anything not invariant propagates in full.

Redfield relaxation (`spin/redfield.rs`) works on the same blocks. The
spatial tensors of the interactions (dipolar `b(1 − 3nnᵀ)`, anisotropic
shielding `γBσ`) are expanded in an orthonormal basis `E_μ` of traceless
symmetric tensors; the isotropic average of their correlations makes the
superoperator a sum of 30 independent terms `[𝒬, [Y(𝒬), ·]]`, with
`𝒬 = Σ_k (A^k : E_μ) Q^k,S_ab` and `Y` the multiplication by `J(ω_mn)` in
the eigenbasis of the laboratory-frame `H` (its blocks by total `M`).
The 30 terms are kept as sparse operators on the basis states ordered by
label (every label's states contiguous), `𝒬` from the local two-spin
tensors and `Y(𝒬)` from one pair of eigenblocks at a time (entries below
10⁻¹⁵ of their pair's largest dropped as rounding), together with the
part of `Σ w 𝒬Y` within each label. A column `R(|a⟩⟨b|)` of a
coherence-order block of the frame's labels (in the rotating frame the
secular projection) is the sum of that left product's columns `a` and `b`
(the right product is its adjoint) and of the products `𝒬_ca Y_be` and
`Y_ca 𝒬_be` over the pairs whose labels keep the order, found by binary
search per label; a block costs its entries, not their square. The
entries are bounded from the union of the terms' patterns before a block
is built, and the terms' own memory and work from the sizes of the
laboratory blocks before the terms are. Blocks are built into the
generator where they are used and kept only for Dormand–Prince steps,
which apply them as matrix-vector products. `R(ρ_eq − 1/d)` joins
`L(1/d)` in the drift of the deviation's equation (`R(1) = 0`, so the
identity's image would be rounding of `R`'s size).

Chemical exchange (`spin/exchange.rs`) joins the coherence-order blocks
of all sites: labels are written over the isotopes of all sites, each
site's Liouville block of order `q` is placed at an offset in the joint
block `q`, and `generator_entries` (shared with the single system) fills
each site's evolution and relaxation. A channel adds `−k` on the source's
diagonal and, for every element `(a, a')` of the source whose traced spins
agree, `k · ρ_fresh(f, f')` at the target element composed of the carried
levels and the fresh ones `(f, f')`; the fresh state has order 0, so this
stays within the block. The deviation from `c_s/d_s` is propagated, with
the relaxation drift and the exchange of `c_s/d_s` in the augmented
column. Steady states replace one population equation per site by
`Tr Δ_s = 0`, scale rows and columns, and solve by fully pivoted LU,
refusing pivots below 10⁻¹² of the largest.

Optimal control (`spin/control.rs`) evaluates the transfer `Re Tr(C†
ρ(T))` of piecewise-constant controls and its gradient in one forward and
one backward pass. Without relaxation each step's `H_k` is diagonalised:
`U_k` follows from the eigenvalues, and the derivative of `U_k` in a
direction `∂H` is `V(G ∘ V†∂HV)V†` with the divided differences
`G_ab = −iΔt e^{−i(E_a+E_b)Δt/2} sinc((E_a−E_b)Δt/2)`, so that one matrix
`X = V(Gᵀ ∘ V†(ρU†λ† + ρ†U†λ)V)V†` per step gives every control's
gradient as `Re Tr(X ∂H_c)`. With relaxation the state is the vector
`[vec(ρ − Tr ρ/d); Tr ρ]` of Liouville space, as `evolve` propagates it,
under the full generator (the system's dissipators and `−R`, the drift
towards equilibrium in the last column); the step propagator and its
derivative act on vectors only: `e^{LΔt}x` and the upper block of
`exp([[LΔt, EΔt], [0, LΔt]])[0; x]` (Van Loan) as Taylor series in
substeps of norm ½, summed until a term falls below rounding. Bounded
amplitudes and fields are mapped by a sine from free variables (Hz for
rf, a twentieth of the range for fields), and L-BFGS sees the objective
divided by its natural size. Field profiles of an exchange use the same
block exponentials per coherence-order block (a field keeps the orders),
with the field derivative of the whole generator by Richardson-
extrapolated central differences.

NV centres (`spin/nv.rs`) use the secular Hamiltonian, which is block
diagonal in the electron level `a` (`m = +1, 0, −1`) and, within a block,
a sum of one term per nucleus (`S_z` alone couples to the nuclei, and
they do not couple to each other): `e^{−iHt} = ⊕_a e^{−iE_a t} ⊗_k
e^{−i h_a^k t}`. The density matrix is kept as its 3 × 3 blocks; a free
period multiplies each block above the diagonal by the factors of its two
levels, one 2 × 2 or 3 × 3 matrix per nucleus applied along that
nucleus's index (`apply_factors`), and mirrors the rest; an ideal pulse
combines blocks with the electron's 3 × 3 unitary; a finite pulse, which
mixes the levels while the nuclei evolve, is the exponential of the full
Hamiltonian (`cmul` on the whole matrix). A `Plan` computes each distinct
period and pulse once. The drive is `e^{−iφS_z²} S_x e^{iφS_z²}`, the
co-rotating part of a linearly polarised carrier, so both transitions see
one phase. Noise trajectories add `exp(−2πi I S_z)` to each period, a
phase per block, with `I = ∫b dt` sampled exactly from coefficients
computed once per period length; during a finite pulse the noise is held
at its exact mean over pieces of at most `1/(16θ)`, up to 16. Resonance
lines use the product structure too: the levels of each block are
products of one state per nucleus, so a line's strength is a product of
one overlap per nucleus. The Gaussian decay exponent sums the exactly
integrated exponential correlations of every pair of free periods as a
running sum, in one pass.

The optical cycle (`spin/nv/optics.rs`) is a Lindblad equation on seven
levels: ground eigenstates, excited eigenstates, singlet. The jump
operators are the zero-field ones (`|e_p⟩⟨g_p|` and so on, one per spin
projection) written in the eigenbases, with both triplets' Hamiltonians
in the laboratory frame (from their means). Their generator is assembled
on `vec ρ` as `L̄ ⊗ L − ½(1 ⊗ K + Kᵀ ⊗ 1)`, `K = Σ L†L`. With a
microwave, the two ground levels other than the one closest to
`m_s = 0` rotate at the signed frequency, as the `S_z²` frame of 1.37
does along the axis. Every operator then keeps only its parts within the
blocks {that level} and {the other two}: the parts between blocks rotate
at the microwave frequency.

The steady state is an LU solve with one equation replaced by the trace,
checked to be a state. Readouts propagate with one `expm` per time step,
and steps or durations whose norm would need too many squarings are
refused. After the laser, the population that the excited levels and the
singlet hand to the ground state is one linear solve, `M_go (−M_oo)⁻¹
x_o`, the integral of the inflow, not a long exponential. The ensemble inversion uses the
symmetric functions of the three levels of each orientation, which give
`B²` and `B_⊥²` without solving the cubic. The signs of the projections
come from `Σ_k B·n_k = 0`, and Gauss–Newton on the eight lines refines
the field.

The ¹³C bath (`spin/nv/bath.rs`) works with ideal pulses only, so the
electron is in one of two levels at any moment. A sequence is parsed into
two branches with their levels per free period and two weights from the
first and last pulses (π pulses only swap the branches and add phases).
The bath enters only through `L = Tr(U_m U_0†)/2^N`. Clusters are the
connected sets of the graph of pairs within the cutoff, grown one spin at
a time and deduplicated. Each is propagated densely: Hamiltonians built
by bit operations on the basis states and diagonalised once per level,
period exponentials from the spectrum and cached by length. The expansion
divides each cluster's coherence by those of its connected subclusters,
found as the subsets present in the map of smaller clusters; disconnected
subsets contribute exactly 1 because their couplings are not kept.
Clusters of one size are evaluated in parallel.

## Hamiltonian simulation and chemistry (`algorithms/time_evolution.rs`, `chemistry/`)

Every exponential of a Pauli string becomes `Circuit::pauli_rotation`: a
basis change (H for X, H·S† for Y), a CNOT ladder onto the last qubit, one
`rz` and the reverse. The angle may be a symbolic parameter, so UCC
generators, QAOA cost layers and Trotter steps differentiate with the
adjoint method like any other gate. Product formulas list the rotations of
one step, merge adjacent equal strings (also across steps) and repeat.
The error bounds are computed in the Pauli algebra with `QubitOperator`
(a `BTreeMap` from string to complex coefficient, deterministic in order):
nested commutators of suffix sums, bounded by the coefficient one-norm.

The nested error bound of a product formula follows one step through
its exponentials by Horner's scheme, `Y_j = B_j + exp(τ ad_{B_j}) Y_{j−1}`,
on power series in τ whose coefficients are Pauli operators in hash maps
keyed by the two masks. One pass up to degree D costs about D²/2
commutator operations per exponential and string; the degree grows
until the multinomial tail is small, and a budget of operations ends the
attempt early (the bound is then infinite and the stage-by-stage one
applies).

The repulsion integrals are screened before they are computed: one pass
over the diagonal quartets `(ab|ab)` gives the Schwarz factor of every
pair of shell groups, and the list of quartets keeps only those whose
product of factors reaches the threshold. Skipped quartets stay zero in
the tensor.

The three fermion mappings are one algorithm. An encoding is a lower
unitriangular binary matrix `A` with `b = A n`; `a_j† = X_F Z_P (1 + Z_N)/2`
where `F` is column `j` of `A`, `P` encodes `n_0 ⊕ … ⊕ n_{j−1}` and `N`
encodes `n_j`, both read off rows of `A⁻¹` (forward substitution over
GF(2)). Jordan–Wigner, parity and Bravyi–Kitaev differ only in `A`. Since
`A` is a CNOT circuit, the mappings send Pauli strings to Pauli strings,
and all of them give the same number of terms.

Full CI does not use the full register: it builds the Hamiltonian's matrix
in the span of the encoded determinants by applying every Pauli term to
each basis state and summing the images (single terms leave the sector,
their sum does not). Up to 4096 determinants the matrix is diagonalised
densely, beyond that the ground state comes from Lanczos.

## State geometry (`algorithms/geometry.rs`)

`state_derivatives` runs the circuit once and carries one vector per
parameter: at each parametrised gate it adds `scale · (dU/dφ)|ψ⟩` to that
parameter's vector, and every later gate is applied to all vectors. This
costs parameters × gates gate applications and parameters × 2ⁿ memory
(capped at 2²⁷ amplitudes); from the vectors come the quantum Fisher
information, the natural gradient and McLachlan's imaginary-time flow.
VQD reuses the adjoint method with the observable `H + Σ β_j |ψ_j⟩⟨ψ_j|`.

## Determinism

* `rng::Rng` is xoshiro256++ seeded through SplitMix64, implemented in the
  crate so that a seed means the same stream on every platform and across
  dependency upgrades. A test pins the first outputs (cross-checked against
  an independent implementation).
* Floating-point reductions (`parallel::sum_f64`, `sum_c64`) sum fixed-size
  chunks and then combine the partial sums in order, so results do not
  depend on the number of threads.
* Multi-start optimisers derive one independent stream per restart with
  `Rng::fork`, so parallel restarts are reproducible.

## Gradients (`algorithms/gradients.rs`)

Adjoint differentiation (Jones & Gacon 2020) keeps two vectors: the state
`φ` and the co-state `λ = H|ψ⟩`. Walking the circuit backwards, each gate
is undone on `φ` and `λ`, and every parameter of the gate contributes
`2·Re⟨λ|∂U φ⟩·scale`; since 1.26 that matrix element is computed in one
reading pass over both vectors (`kernels::derivative_element`) instead of
applying the derivative to a copy of `φ`. On registers larger than a
cache block, runs of gates whose targets lie inside a block, and diagonal
gates on any qubits (per block a phase, or a diagonal gate on their one
low target), are undone and differentiated block by block, like the
forward plans: the block of `φ` and `λ` stays in cache through the whole
run, and the contributions of each gate are summed over the blocks in
block order, so the result does not depend on the number of threads. A
gradient costs about four to six energy evaluations regardless of the
parameter count (ROADMAP, 1.26; 0.53 ms vs 28 ms for parameter shift with
100 parameters at 10 qubits, see BENCHMARKS.md). Diagonal observables
are evaluated chunk by chunk: the probabilities of a chunk stay in cache
while every `Z` string passes over them. Parameter-shift rules are provided for
hardware-faithful gradients and as an independent check: two terms for
`exp(−iθG/2)` with `G² = I` and phase gates, four terms for single-controlled
rotations.

The adjoint pass, the kernels it calls and the QAOA pass are generic over
the amplitude type (`Amplitude`: `Complex64` or `Complex32`). In single
precision the states are `Complex32`, and every inner product, matrix
element and expectation value is summed in double precision; energy and
gradient are divided by the state's norm², which rounding moves away from
one, and the observable's constant term is added to the energy instead of
entering the co-state (ROADMAP, 1.27). The optimisers are told the
energies' rounding (`optimize::minimize_at`): the L-BFGS line search
treats values within it as equal and accepts Hager and Zhang's
approximate Wolfe condition there, and L-BFGS and Adam stop when the
gradient reaches its rounding instead of on a relative decrease.

QAOA has its own specialised adjoint pass: the cost layer generator is the
diagonal cut vector and the mixer generator is `Σ X_q`, both applied in
O(n·2ⁿ). Cost phases come from a table over the distinct cut values (at
most 2¹⁶, always for unweighted graphs), and the cost layer and the
rotations of the low qubits share one blocked pass.

`expectation_batch` and `expectation_and_gradient_batch` evaluate many
parameter vectors; below 16 qubits, where one simulation uses at most two
threads, they run in parallel across the vectors.

## Krylov methods (`linalg/krylov.rs`)

Operators are closures `y = H·x`, so the same code serves sparse graph
Hamiltonians and Pauli sums on 2ⁿ amplitudes.

* Ground states: explicitly restarted Lanczos with full
  re-orthogonalisation ("twice is enough"), convergence on the Ritz
  residual, verified by an explicit residual at the end.
* Propagation `exp(−iHt)v`: Lanczos basis per sub-step, exact exponential of
  the tridiagonal matrix, step-size control from the a-posteriori error
  estimate `β₀·β_m·|[e^{−iτT}e₁]_m|`. That estimate holds only where the
  Krylov approximation converges, so a step never exceeds `m` divided by
  the spread of the Ritz values (Hochbruck and Lubich); beyond, it can be
  small by accident, as on the equally spaced spectra of hypercubes.
  Estimates at rounding level (64 ε) count as converged.
* Cyclic subspaces (`cyclic_spectrum`): real Lanczos from a start vector
  until the next vector vanishes to rounding error, with full
  re-orthogonalisation (a second pass only when the first removed much of
  the vector). The subspace is the smallest invariant one containing the
  start vector; its dimension is the number of distinct eigenvalues the
  vector sees (21 for a node of the 2²⁰-node hypercube), and its spectral
  decomposition gives limiting and time-averaged distributions exactly.

## Automorphism groups (`graph/automorphism.rs`)

A stabiliser chain is built point by point. For base point `b` and candidate
image `w`, an automorphism fixing the earlier base points and mapping
`b ↦ w` is searched by individualisation–refinement: matched vertices of two
copies of the graph get equal colours, 1-WL colour refinement (respecting
edge weights) runs on both copies jointly so colours are comparable,
incompatible colour-class sizes prune the branch, and the search branches on
the smallest non-trivial class. Found automorphisms close the orbit, so
candidates already reached are skipped. The order is the product of the
basic orbit lengths (orbit–stabiliser theorem), which is exact. Test cases
cover Kₙ, Cₙ, Pₙ, stars, grids, hypercubes, the Petersen graph, all
Platonic solids and the Metatron family.

## Canonical labelling (`graph/canonical.rs`)

The search tree of individualisation–refinement (as in nauty): refine,
individualise each vertex of the smallest non-trivial colour class, refine
again, until the colouring is discrete. Each leaf numbers the vertices; the
canonical form is the leaf whose relabelled edge list is smallest.
Refinement keeps an ordered partition and splits it as in Hopcroft's
algorithm (`O(m log n)` splitting steps); colours are class positions, so
every step commutes with relabelling. Two leaves with equal edge lists give
an automorphism; it fixes the common prefix of their paths, so the search
jumps back there and prunes children in the same orbit of the automorphisms
found so far. Twins (vertices with equal neighbourhoods apart from each
other) are interchangeable, so only one per class is explored, and a colouring
whose classes are all twin classes is finished directly. K₄₀₀, stars,
complete bipartite and empty graphs need a single leaf (milliseconds); paths
and cycles with thousands of vertices take well under a second; random cubic
graphs with 2000 vertices take seconds. Cospectral strongly regular graphs
(Shrikhande vs 4×4 rook's graph) are separated, and isomorphism decisions
agree with brute force on random graphs.

## Quantum walks (`walks/`)

* Dense path (n ≤ 2048): eigendecomposition of the real symmetric
  Hamiltonian once; evolution, propagators, limiting distributions and exact
  time averages are spectral formulas. Degenerate eigenvalues are grouped
  into eigenspaces before computing limiting distributions; v0.1 omitted
  this and produced symmetry-violating results.
* Sparse path (any n): CSR Hamiltonian and Krylov propagation.
* Beyond 2048 nodes, limiting and averaged distributions use the start
  node's cyclic subspace (see Krylov methods).
* Quotients (`walks/quotient.rs`): for an equitable partition with
  quotient matrix `B` (weight from a node of cell `i` into cell `j`), the
  normalised cell states span an invariant subspace on which the adjacency
  matrix is `A_red[i][j] = √(B_ij·B_ji)` and the Laplacian
  `diag(d) − A_red`. The coarsest equitable partition with given singletons
  comes from the same colour refinement as the automorphism search;
  hypercubes, Hamming, Johnson, complete and complete multipartite graphs
  have closed-form distance partitions, so their walks never build the
  graph.
* Driven walks on quotients (`walks/driven.rs`, `QuotientDrivenWalk`)
  reduce every term to the cells (the uniform projector to `s sᵀ` with
  `s_i = √(|C_i|/N)`) and hand the `k × k` problem to `DrivenWalk`.
* Open walks on orbitals (`walks/symmetric.rs`): an operator invariant
  under a permutation group is constant on its orbits on ordered pairs.
  The orbits come from union–find over the `n²` pairs, with every root the
  smallest pair of its set, so one pass relabels them in order. The
  Lindblad generator maps invariant operators to invariant ones, so row
  `O` of the reduced map is read off at one representative pair of `O`;
  it is applied by a Taylor series in substeps of ∞-norm ½. The coined
  walk's step is the conjugation `Σ_cd U_ac conj(U_bd)` followed by the
  dephasing factor of each orbital. `HypercubeOpenWalk` indexes the
  orbitals of the coordinate permutations by the counts of bit pairs
  (ranked in closed form) and works in the orthonormal basis `B_O/√|O|`,
  where the moves of one coordinate between classes have entries
  `γ√(n_c (n_c' + 1))`, the same structure as boson hopping.
* State transfer (`walks/transfer.rs`) works on the quotient in which the
  source is a singleton: perfect transfer needs the target to be a
  singleton too, strong cospectrality (`E_θ e_v = ±E_θ e_u`) and eigenvalue
  differences `n_k·g` with `n_k` odd exactly where the signs differ; ratios
  are matched to fractions with denominators up to 10⁴, and a positive
  answer is confirmed by evaluating the amplitude.
* Discrete-time walks (`walks/discrete.rs`) act on arcs (coined walks:
  coin per node, flip-flop shift; Szegedy: two reflections, with marked
  nodes made absorbing by a self-arc).
* Walk circuits (`walks/circuits.rs`): circulant Hamiltonians are diagonal
  in the Fourier basis (QFT, then a diagonal from its Walsh decomposition —
  one `RZ` and a `CX` parity chain per coefficient), Cayley graphs of Z₂ⁿ
  are sums of commuting X-strings, and product formulas rotate each edge
  in the two-dimensional span of its end nodes (CX gates bring the two
  basis states to differ in one qubit, then a gate controlled by all other
  qubits rotates them).
* Spatial search optimises the hopping rate around the Childs–Goldstone
  critical value `γ_c = (1/N) Σ 1/λ_k` (exactly 1/N on K_N) together with
  the measurement time.
* Dephasing walks integrate the Lindblad equation with adaptive RK4 (step
  doubling, Richardson extrapolation).
* Interacting walkers (`walks/interacting.rs`) work in the occupation
  basis. Configurations of bosons and fermions are sorted site lists,
  ranked in colex order by the combinatorial number system (a boson
  multiset `s₁ ≤ … ≤ s_N` as the set `s_k + k`), so a configuration's
  index is computed, not looked up. Distinguishable particles are base-n
  tuples. The Hamiltonian is built once as a sparse matrix from the
  single-particle `h` (hopping amplitudes `√(n_j (n_i + 1))` for bosons,
  the Jordan–Wigner sign of the sites in between for fermions). Up to
  2048 configurations it is diagonalised; beyond that it propagates by
  Krylov. `Terms::apply` gives a configuration's diagonal and hops, and
  serves both the full Hamiltonian and the reduced ones.
* Interacting walkers on orbits (`walks/interacting/symmetric.rs`): a
  union–find over the ranks of all configurations links each to its
  images under the generators, with the relative sign `ψ(g·c) = s(g, c)
  ψ(c)` in the top bit of every 4-byte word. Roots are the smallest rank,
  so a scan in rank order meets every orbit's representative first and
  relabels the words in place as orbit numbers; a cycle of signs that
  multiplies to −1 marks the orbit as carrying no invariant state. The
  surviving roots are counted before their sites are stored (at most
  2²⁶ sites), and a walk without any is refused. The reduced Hamiltonian
  is read off at the representatives,
  `⟨O′|H|O⟩ = √(|O|/|O′|) Σ φ(c′)⟨c′|H|r⟩`.
* On the hypercube (`walks/interacting/hypercube.rs`) the orbits of the
  coordinate permutations are compositions of `d` into `2^N` bit
  patterns, ranked by their bars in colex order; the particle
  permutations act on the patterns, and each composition's word holds
  its class (the lowest-ranked image) and the sign of the permutation to
  it. Heap's algorithm lists the permutations with one parity flag shared
  by its whole recursion (every swap is a transposition). A hop changes
  one pattern `p` into `p ⊕ e_j`, with weight
  `−γ √(n_p (n_q + 1) k_n / k_m)` (`k`: compositions per class); the
  entries are bounded before the matrix is built, by compositions over
  permutations first and by the classes once they are known.
* Certificates by orbits (`InteractingWalk::certificate_by_orbits`): one
  row of transition probabilities per orbit of the automorphisms, by
  Krylov steps to 10⁻¹⁴ per unit time unless the walk was diagonalised
  already, repeated by the orbit's size before sorting.

## Error handling

All user-facing operations return `Result<_, QsoError>`. Fluent circuit
builder methods (`c.h(0).cx(0, 1)`) panic on invalid qubit indices, as
indexing would; use `Circuit::try_push` for untrusted input. The Python
bindings convert every `QsoError` into `ValueError`.
