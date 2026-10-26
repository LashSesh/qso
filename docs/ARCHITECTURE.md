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
            │ search · walk circuits · dephasing · classical     │
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

Specialised paths:

* uncontrolled single-qubit gates iterate over contiguous half-blocks
  (`parallel::for_each_pair`), which the compiler vectorises; diagonal
  matrices (`RZ`) scale each half without branching, and real matrices
  (`H`, `RY`) use real instead of complex products, about twice as fast;
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
- **Logical error rates.** `logical_error_rate` samples in blocks of
  whole batches (at most 65 536 shots and 1 GiB of results) from one
  random stream, so it returns the same result as one call of
  `sample_detectors` with the same seed, in bounded memory. A batch is
  1024 shots, or fewer when its frames and records would exceed 128 MiB.

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
  1 − ‖F‖²/‖M‖², measured rather than estimated, so the error accounting
  is exact even when the eigenvectors are not. nalgebra's complex SVD
  was not used because it fails on rank-deficient matrices.
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
segments and channels on one or two qubits (`noise/device.rs`). The
compiler does four things:

* it schedules gates as soon as their qubits are free, inserting
  relaxation for waiting qubits and for each gate's duration;
* it applies the first matching gate rule;
* it splits thermal relaxation into amplitude damping and a phase flip
  (the same channel);
* it merges consecutive channels of one kind on the same qubits. Pauli
  channels merge by convolving their probabilities, other channels get a
  minimal Kraus set from the Choi matrix.

Density matrices apply the program with the state-vector kernels on the
vectorised ρ and multiply consecutive superoperators first.

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

Adjoint differentiation (Jones & Gacon 2020) keeps three vectors: the state
`φ`, the co-state `λ = H|ψ⟩`, and a scratch copy. Walking the circuit
backwards, each gate is undone on `φ` and `λ`, and every parameter of the
gate contributes `2·Re⟨λ|∂U φ⟩·scale`. The cost is about three circuit
simulations regardless of the parameter count (0.53 ms vs 28 ms for
parameter shift with 100 parameters, see BENCHMARKS.md). Parameter-shift rules are provided for
hardware-faithful gradients and as an independent check: two terms for
`exp(−iθG/2)` with `G² = I` and phase gates, four terms for single-controlled
rotations.

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
  estimate `β₀·β_m·|[e^{−iτT}e₁]_m|`.
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

## Error handling

All user-facing operations return `Result<_, QsoError>`. Fluent circuit
builder methods (`c.h(0).cx(0, 1)`) panic on invalid qubit indices, as
indexing would; use `Circuit::try_push` for untrusted input. The Python
bindings convert every `QsoError` into `ValueError`.
