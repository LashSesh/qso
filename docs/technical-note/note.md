---
title: "Validated Quantum Algorithm Benchmarks of the Metatron QSO Simulation Framework"
subtitle: "Technical Note, version 2 — corrects and replaces version 1 of 26 November 2025"
author: "Sebastian Klemm"
date: "QSO {{tn.environment.qso_version}} · {{tn.environment.date}}{{doi_suffix}}"
---

<div class="abstract">

This note reports benchmarks of Metatron QSO {{tn.environment.qso_version}}, an
open-source quantum simulation framework written in Rust with Python
bindings. It covers the categories of version 1 of this note — Grover and
multi-target Grover search, continuous-time quantum walks on Metatron's
Cube, QAOA, VQE, variational classifiers, quantum machine learning on
graphs, boson sampling and a comparison with other frameworks — and
replaces all of its results, which are withdrawn (Section 1). Every number
here is computed by code in the public repository and compared with an
independent reference: a closed form, a published value, or a second
implementation. Simulated Grover success probabilities match
sin²((2k+1)θ) to {{tn.grover.max_abs_error:sci}}; the time-averaged walk
on Metatron's Cube converges to the exact value 145/169 at the start
vertex; QAOA reproduces the proven optima on rings and the published
values 0.6925 (p = 1) and 0.7559 (p = 2) on large-girth cubic graphs; VQE
reaches the full-CI energy of H₂ to {{tn.vqe.h2.abs_error:sci}} Ha; the
exact boson sampler passes a χ² test against exactly computed output
distributions. State vectors agree with Qiskit Aer, Qulacs, Cirq and
PennyLane Lightning to within {{derived.max_state_diff:sci}}.

</div>

# 1 Correction of version 1

<div class="notice">

Version 1 of this note (26 November 2025) reported benchmark results of
QSO 0.1. **All of its quantitative results are withdrawn.** An audit of
the QSO 0.1 code (published with the repository as `docs/AUDIT_v0.1.md`)
showed that the "quantum system" was a single 13-level system rather than
a register of qubits, that several algorithms were implemented
incorrectly, and that some reported numbers were constants written into
the code rather than measurements. Version 1 was described as
"CI-validated"; the continuous-integration checks only verified that
result files existed. The author regrets the error.

</div>

Table 1 lists every claim of version 1 and the reason it is withdrawn.
QSO was subsequently rewritten from scratch (version 1.0, October 2026);
the results below are produced by the rewritten code and supersede
version 1 completely.

**Table 1.** Results of version 1 and their status.

{{@v1_claims}}

# 2 Scope and methods

**Software.** QSO {{tn.environment.qso_version}} simulates n-qubit pure
states (up to 34 qubits) and density matrices (up to 14 qubits) with
multi-threaded kernels, computes exact gradients by adjoint
differentiation [17], and provides the algorithms benchmarked here.
Conventions: qubit 0 is the least significant bit of a basis index, time
evolution is $e^{-iHt}$ and rotations are $R_P(\theta) = e^{-i\theta P/2}$. All random
numbers come from one seeded generator (xoshiro256++), and parallel sums
are combined in a fixed order, so every result is bit-for-bit
reproducible on the same platform for any number of threads.

**Validation principle.** A benchmark value is reported only together
with an independent reference: a closed-form result, a value published in
the literature, an exact computation by a different method (for example
Lanczos diagonalisation for VQE), or a different software package. The
test-suite of the repository ({{tests.rust}} Rust tests and
{{tests.python}} Python tests, run on Linux, macOS and Windows for every
change) checks the same references; this note collects them with their
numerical agreement.

**Reproducibility.** All numbers are produced by two programs in the
repository and stored as JSON next to this note:

```
cargo run --release --example technical_note -- --out technical_note.json
python benchmarks/cross_framework/run.py --out cross_framework.json
```

The data in this version were computed from commit
`{{tn.environment.commit}}` on a {{tn.environment.machine}}
({{tn.environment.threads}} threads). The *Benchmarks* workflow of the
repository reruns both programs on GitHub-hosted machines.

**Timing.** Times are wall-clock medians of repeated runs after a warm-up.
The machine is a shared virtual machine whose speed varies in bursts, so
absolute times can differ by tens of percent between runs; the
cross-framework comparison therefore interleaves the frameworks (Section
3.8) and also reports the minimum over all runs.

# 3 Results

## 3.1 Grover search

Grover's algorithm [1] with M marked items among N = 2ⁿ reaches success
probability sin²((2k+1)θ) after k iterations, where sin θ = √(M/N); the
optimal k is ⌊π/(4θ)⌋ [2]. QSO builds the oracle and diffusion operator as
circuits of multi-controlled gates and simulates them on the full state
vector.

**Table 2.** One marked item: simulated and exact success probability at
the optimal iteration count, and the number of oracle queries of a
classical search ((N+1)/2 on average) divided by k.

{{@grover_single}}

**Table 3.** Multiple marked items (n = 10, N = 1024).

{{@grover_multi}}

The largest deviation from the exact formula over all single-target sizes
(n = 2 … 16) and the multi-target cases is
{{tn.grover.max_abs_error:sci}}, i.e. rounding error. The query ratio
grows as √N, the quadratic speed-up of unstructured search. (Version 1
reported a success probability of 0.0754 ≈ 1/13 — the probability of
guessing one of 13 states — and called the ratio of two unrelated
quantities a speed-up.)

## 3.2 Quantum walks on Metatron's Cube

Metatron's Cube joins all 13 centres of the Fruit of Life by straight
lines, so as a graph it is the complete graph K₁₃ (78 edges, all degrees
12). Its symmetry group is the full symmetric group S₁₃; the geometry
lives in the embedding (seven distinct line lengths). Keeping only the
lines between tangent circles gives the Fruit-of-Life contact graph with
18 edges and the hexagonal symmetry group D₆. QSO computes automorphism
groups exactly (individualisation–refinement with orbit–stabiliser
bookkeeping):

**Table 4.** Automorphism group orders.

{{@automorphisms}}

**Continuous-time walk.** For H = L (graph Laplacian) and a walker started
at vertex 0 of K₁₃, the Laplacian spectrum {0, 13¹²} gives
P₀(t) = (145 + 24 cos 13t)/169. The time-averaged distribution therefore
converges to 145/169 at the start vertex and 2/169 elsewhere. QSO computes
limiting distributions from the eigendecomposition with degenerate
eigenspaces grouped; it returns P₀ = {{tn.walks.metatron_limiting_centre:.15f}}
(exact: {{tn.walks.metatron_limiting_centre_reference:.15f}}).

The mixing time $M_\varepsilon$ of Aharonov et al. [3] is the smallest $T$
from which on the time-averaged distribution stays within total-variation
distance $\varepsilon$ of the limit. On K₁₃ that distance is
$\tfrac{24}{169}\,\lvert \sin(13T)/(13T) \rvert$, which gives
$M_\varepsilon$ in closed form (evaluated on the same time grid, spacing
0.001):

**Table 5.** Mixing time of the walk on Metatron's Cube from vertex 0.

{{@mixing}}

The Fruit-of-Life contact graph mixes more slowly ($M_{0.05}$ =
{{tn.walks.fruit_mixing_time_eps005:.2f}}) and keeps
{{tn.walks.fruit_limiting_centre:.4f}} of the probability at the centre in
the limit.

**Spatial search.** Childs and Goldstone [4] search for a marked vertex w
with $H = \gamma L - |w\rangle\langle w|$. On the complete graph $K_N$ with
$\gamma = 1/N$ the walk rotates the uniform superposition into
$|w\rangle$, with success probability 1 at $T = \pi\sqrt{N}/2$. On Metatron's Cube QSO obtains
P(T) = {{tn.walks.search_exact.success:.15f}} at T = {{tn.walks.search_exact.time:.6f}};
its rate optimiser, started from no prior knowledge, finds
γ = {{tn.walks.search_optimised.gamma:.9f}} (1/13 = 0.076923077) and
P = {{tn.walks.search_optimised.success:.12f}}. A classical random walk
needs {{tn.walks.classical_mean_hitting_time:.1f}} steps on average to hit
the marked vertex. Table 6 shows the √N scaling on larger complete graphs.

**Table 6.** Search on complete graphs: quantum evolution time against the
mean classical hitting time N − 1.

{{@search_scaling}}

On the Fruit of Life, which lacks the symmetry of K₁₃, optimal search
depends on the marked vertex and does not reach probability 1:

**Table 7.** Search on the Fruit-of-Life contact graph (rate and time
optimised, Laplacian generator).

{{@fruit_search}}

## 3.3 QAOA for MaxCut

QAOA [5] alternates $e^{-i\gamma C}$ for the cut operator
$C = \sum_{(u,v) \in E} (1 - Z_u Z_v)/2$ with the mixer
$e^{-i\beta \sum_q X_q}$ and maximises $\langle C \rangle$. QSO simulates it
on the full $2^n$-dimensional state, with exact gradients and layer-wise
(INTERP) initialisation, and reports the approximation ratio
$\langle C \rangle / C_{\max}$ with $C_{\max}$ found by enumeration.

**Table 8.** QAOA against analytic and published values.

{{@qaoa_reference}}

On the ring the optimum (2p+1)/(2p+2) is proven [6]. For cubic graphs in
which no edge lies on a cycle of length at most 2p+1, every edge sees the
same tree-like neighbourhood, so the optimal ratio is a universal constant:
0.6925 for p = 1 [5] and 0.7559 for p = 2 [8]. The 3-cube Q₃ (girth 4) and
the Heawood graph (girth 6; bipartite, so $C_{\max} = |E| = 21$) satisfy the
conditions. In addition, the one-layer expected cut, summed over edges,
matches the closed form of Wang et al. [7] to 10⁻¹² on graphs with
triangles, varying degrees and random structure (part of the
test-suite).

**Table 9.** QAOA on Metatron's Cube (maximum cut {{tn.qaoa.metatron_max_cut:.0f}}),
the Petersen graph and random cubic graphs (mean ± standard deviation over
{{tn.qaoa.random_regular.instances}} instances), with the probability of
measuring an optimal cut.

{{@qaoa_depth}}

## 3.4 Variational quantum eigensolver

For H₂ in the STO-3G basis at 0.735 Å (two-qubit parity-mapped
Hamiltonian), VQE [9] with a one-layer real-amplitude ansatz reaches
E = {{tn.vqe.h2.vqe:.10f}} Ha including nuclear repulsion; exact
diagonalisation of the same Hamiltonian gives {{tn.vqe.h2.exact:.10f}} Ha,
the full-CI energy of the basis (−1.137306 Ha [10]).

For {{tn.vqe.qubits}}-qubit spin chains the reference is the ground-state
energy from Lanczos diagonalisation. Table 10 shows how the error depends
on the ansatz and its depth (four seeded restarts, L-BFGS with adjoint
gradients). It is not monotone in the depth, because the optimiser can end
in a local minimum; these are the values the software produces, not
selected results.

**Table 10.** VQE on the open transverse-field Ising chain at the critical
point (TFIM, J = h = 1) and the XXZ chain (J = 1, Δ = 0.5). RealAmplitudes
uses RY rotations, EfficientSU2 RY and RZ rotations, both with a CX chain.

{{@vqe}}

## 3.5 Variational classifier

The classifier uses data re-uploading [11]: 2 qubits, 4 layers, 16
parameters, output p(y = 1|x) = (1 − ⟨Z₀⟩)/2, trained on the cross-entropy
with exact gradients. Each dataset has {{tn.vqc.datasets.0.samples_per_split}}
training and {{tn.vqc.datasets.0.samples_per_split}} test points; the table
reports mean ± standard deviation over {{tn.vqc.datasets.0.seeds}} seeds and
compares with logistic regression on the raw features and on quadratic
features.

**Table 11.** Accuracy of the variational classifier and classical
baselines.

{{@vqc}}

The variational model separates the non-linear data sets (circle, moons)
where linear logistic regression cannot, but a classical model with the
right features does as well or better. The benchmark therefore shows a
correct and reproducible training pipeline, not an advantage.

## 3.6 Graph classification with quantum layers

Version 1 reported a quantum machine-learning experiment on graphs. QSO
now provides a graph classifier in which every input graph $G$ defines its
own cut operator $C_G$: $p$ shared, trainable layers
$e^{-i\beta \sum_q X_q} e^{-i\gamma C_G}$ act on $|{+}\rangle^{\otimes n}$, and
$p(y = 1 \mid G) = \sigma\big(a\,(\langle C_G \rangle / |E| - \tfrac12) + b\big)$. Because the evolution is
local, after p layers each edge's contribution depends only on the
subgraph within distance p of the edge (the QAOA light cone). This gives a
falsifiable prediction: one layer can detect triangles, but it cannot
distinguish triangle-free cubic graphs with and without 4-cycles, because
their radius-1 neighbourhoods are identical; two layers can.

The data are random cubic graphs
({{tn.graph_classifier.triangles.graphs_per_class_and_split}} graphs per
class in each of the training and test sets,
{{tn.graph_classifier.triangles.seeds}} seeds). Simple degree statistics are
identical for all cubic graphs, so a classifier using them cannot exceed
50 %; counting triangles or 4-cycles classifies perfectly.

**Table 12.** Graph classification of random cubic graphs (mean ± standard
deviation over seeds; the 4-cycle task uses triangle-free graphs only).

{{@graph_classifier}}

The one-layer model is at exactly 50 % on the 4-cycle task, as predicted;
two layers reach high accuracy. Cubic graphs with girth ≥ 5 on 14
vertices are rare: each training set of
{{tn.graph_classifier.squares.graphs_per_class_and_split}} such graphs
contains only {{derived.girth5_distinct_min}}–{{derived.girth5_distinct_max}}
distinct graphs (distinct adjacency spectra), so training and test sets
share isomorphic graphs. The result demonstrates the light cone and the
training procedure, not generalisation to unseen structures.

## 3.7 Boson sampling

$n$ single photons entering an $m$-mode interferometer $U$ with input
occupation $S$ leave in occupation pattern $T$ with probability
$|\operatorname{perm}(U_{T,S})|^2 / \prod_i t_i!$ if they are
indistinguishable and $\operatorname{perm}(|U_{T,S}|^2) / \prod_i t_i!$ if
they are distinguishable, where $U_{T,S}$ repeats row $i$ of $U$ $t_i$
times and takes the columns of the occupied input modes [12]. QSO
computes permanents with Glynn's formula in Gray-code order
(O(n·2ⁿ)) [14], draws Haar-random interferometers [16] and samples exactly
with the algorithm of Clifford and Clifford [13].

*Permanents* agree with the defining sum over permutations to a relative
{{tn.boson_sampling.permanent_max_relative_error_n_le_8:sci}} (n ≤ 8) and
with The Walrus to {{xf.correctness.permanent_max_relative_difference_vs_thewalrus:sci}}
(n ≤ 12).

*Hong–Ou–Mandel effect* [15]. Two photons on a 50:50 beam splitter leave
through different ports with probability
{{tn.boson_sampling.hom.p_coincidence_boson:sci}} if indistinguishable
and {{tn.boson_sampling.hom.p_coincidence_distinguishable:.3f}} if
distinguishable: visibility {{tn.boson_sampling.hom.visibility:.6f}}.

*Bunching.* If all n photons leave through one mode, the submatrix has
rank one, so $P_\text{boson}/P_\text{dist} = n!$ for every interferometer:

**Table 13.** Full-bunching enhancement in Haar-random 8-mode
interferometers.

{{@bunching}}

*Exact sampling.* For {{tn.boson_sampling.sampler_validation.photons}}
photons in {{tn.boson_sampling.sampler_validation.modes}} modes
({{tn.boson_sampling.sampler_validation.patterns}} output patterns), the
exact distribution sums to 1 within
{{tn.boson_sampling.sampler_validation.normalisation_error:sci}}.
{{tn.boson_sampling.sampler_validation.shots}} samples have total-variation
distance {{tn.boson_sampling.sampler_validation.tvd_empirical_vs_exact:.4f}}
from it; the expected distance for a perfect sampler with this number of
samples is {{tn.boson_sampling.sampler_validation.tvd_expected_for_exact_sampler:.4f}}.
Pearson's χ² is {{tn.boson_sampling.sampler_validation.chi2:.1f}} over
{{tn.boson_sampling.sampler_validation.chi2_cells}} cells with expected
count ≥ 5, consistent with the χ² distribution of that many degrees of
freedom.

*Quantum versus classical statistics.* The distance between the bosonic and
the distinguishable-particle distributions grows with the photon number:

**Table 14.** Mean total-variation distance between indistinguishable and
distinguishable photons in m = n² modes (10 Haar-random interferometers).

{{@boson_distance}}

**Table 15.** Cost of one exact sample of n photons in m = n² modes
(permanent timings are compared with The Walrus in Table 20).

{{@sampling_timing}}

## 3.8 Comparison with other frameworks

Version 1 listed scores for Qiskit, Cirq, PennyLane and ProjectQ that were
constants in the code. The comparison below runs the actual packages
(Qiskit Aer {{xf.environment.packages.qiskit-aer}},
Qulacs {{xf.environment.packages.qulacs}} [18],
Cirq {{xf.environment.packages.cirq-core}},
PennyLane Lightning {{xf.environment.packages.pennylane-lightning}} [19],
The Walrus {{xf.environment.packages.thewalrus}} [20]). ProjectQ is not
included: its last release (0.8.0, 2022) does not build on current Python.

**Correctness.** {{xf.correctness.random_circuits}} random 8-qubit circuits
with 60 gates each (H, X, Y, Z, S, T, RX, RY, RZ, CX, CZ, SWAP) and a
two-layer 10-qubit circuit were simulated by every package; the final
states agree with QSO to rounding error:

**Table 16.** Largest amplitude difference to QSO.

{{@correctness}}

**Speed.** Every framework ran with its default multi-threading. Each
configuration ran in a fresh process; {{xf.rounds}} rounds were
interleaved with a rotating order. Entries are medians, with minima in
parentheses. QSO and Qulacs include the allocation of the state; Qiskit
Aer includes copying the state to Python; PennyLane includes building the
circuit on each call; Cirq uses its built-in Python simulator (qsim was not
tested).

**Table 17.** One layer (H and RZ on every qubit, CX chain).

{{@layers}}

**Table 18.** Ten layers.

{{@deep}}

**Table 19.** Energy and gradient of a 10-qubit transverse-field Ising
model with a 100-parameter ansatz.

{{@gradients}}

From 20 qubits on, QSO is as fast as Qulacs or faster and two to four
times faster than Qiskit Aer and PennyLane Lightning on this machine. On
16 qubits, where one gate takes only tens of microseconds, Qulacs is two
to three times faster; at that size the cost of distributing work over the
threads dominates, and QSO's handling of it is a topic for future work.
The adjoint gradient is the fastest of the three methods compared;
PennyLane's adjoint method is dominated by Python overhead at this size.
Two QSO kernels (H and CX) were optimised during this work after a first
comparison had shown a 1.2–1.7× gap to Qulacs on large registers; all
numbers above are for the optimised code.

**Table 20.** Permanents: QSO against The Walrus (median time per
permanent of a random complex n × n matrix).

{{@permanent_comparison}}

## 3.9 Test-suite

Version 1 reported "integration benchmarks" with a compatibility score of
1.0. Their role is taken by the test-suite: {{tests.rust}} Rust tests
(unit, validation, robustness, property and documentation tests) and
{{tests.python}} Python tests run on Linux, macOS and Windows, with the
minimum supported Rust version and on Python 3.9 and 3.13, for every
change. Releases are built and tested on all supported platforms before
they are published.

# 4 Simulator performance

**Table 21.** One layer of the benchmark circuit in QSO (Rust, no Python).

{{@performance}}

Registers below 2¹⁶ amplitudes run on one thread; the cost per gate and
amplitude falls up to 22 qubits as all threads take part and rises again at
24 qubits, where one state vector takes 256 MB. A full gradient of the
100-parameter ansatz of Table 19 takes
{{tn.performance.gradients_tfim10_100_parameters.0.seconds:ms}} by adjoint
differentiation, {{tn.performance.gradients_tfim10_100_parameters.1.seconds:ms}}
by parameter shift and {{tn.performance.gradients_tfim10_100_parameters.2.seconds:ms}}
by finite differences.

# 5 Limitations

* All results are classical simulations on one machine with four
  threads; nothing here is measured on quantum hardware, and timings on
  other machines will differ.
* Problem sizes are those a classical simulator handles quickly. The
  benchmarks validate correctness and characterise performance; they do
  not claim a quantum advantage.
* The graph-classification task is small and partly memorised (Section
  3.6); the variational classifier does not beat classical baselines with
  suitable features (Section 3.5).
* Other frameworks were used with default settings; tuned configurations
  (for example Qulacs with gate fusion, Cirq with qsim) may be faster.

# 6 Conclusion

QSO {{tn.environment.qso_version}} reproduces every analytic and published
reference used here to within rounding or statistical error, agrees with
four independent simulators and a permanent library to machine precision,
and is competitive in speed with established simulators. All code, data and
the scripts that produce this note are public, so every number can be
regenerated and checked.

# Data and code availability

Source code, data and the programs that produce this note:
<https://github.com/LashSesh/qso> (Apache-2.0). Packages:
`pip install metatron-qso`, `cargo add metatron-qso-rs`. The JSON data of
this version are in `docs/technical-note/data/`.

# Acknowledgements

The rewrite of QSO, the audit of version 0.1 and the benchmarks in this
note were developed with the assistance of an AI coding assistant (Claude,
Anthropic). The author reviewed the results and is responsible for the
content.

# References {#references}

<ol class="references">
<li>L. K. Grover, “A fast quantum mechanical algorithm for database search,” <i>Proc. 28th ACM STOC</i>, 212–219 (1996).</li>
<li>M. Boyer, G. Brassard, P. Høyer, A. Tapp, “Tight bounds on quantum searching,” <i>Fortschr. Phys.</i> 46, 493–505 (1998).</li>
<li>D. Aharonov, A. Ambainis, J. Kempe, U. Vazirani, “Quantum walks on graphs,” <i>Proc. 33rd ACM STOC</i>, 50–59 (2001).</li>
<li>A. M. Childs, J. Goldstone, “Spatial search by quantum walk,” <i>Phys. Rev. A</i> 70, 022314 (2004).</li>
<li>E. Farhi, J. Goldstone, S. Gutmann, “A quantum approximate optimization algorithm,” arXiv:1411.4028 (2014).</li>
<li>G. B. Mbeng, R. Fazio, G. E. Santoro, “Quantum annealing: a journey through digitalization, control, and hybrid quantum variational schemes,” arXiv:1906.08948 (2019).</li>
<li>Z. Wang, S. Hadfield, Z. Jiang, E. G. Rieffel, “Quantum approximate optimization algorithm for MaxCut: a fermionic view,” <i>Phys. Rev. A</i> 97, 022304 (2018).</li>
<li>J. Wurtz, P. Love, “MaxCut quantum approximate optimization algorithm performance guarantees for p > 1,” <i>Phys. Rev. A</i> 103, 042612 (2021).</li>
<li>A. Peruzzo et al., “A variational eigenvalue solver on a photonic quantum processor,” <i>Nat. Commun.</i> 5, 4213 (2014).</li>
<li>A. Szabo, N. S. Ostlund, <i>Modern Quantum Chemistry</i>, Dover (1996).</li>
<li>A. Pérez-Salinas, A. Cervera-Lierta, E. Gil-Fuster, J. I. Latorre, “Data re-uploading for a universal quantum classifier,” <i>Quantum</i> 4, 226 (2020).</li>
<li>S. Aaronson, A. Arkhipov, “The computational complexity of linear optics,” <i>Theory of Computing</i> 9, 143–252 (2013).</li>
<li>P. Clifford, R. Clifford, “The classical complexity of boson sampling,” <i>Proc. 29th ACM-SIAM SODA</i>, 146–155 (2018).</li>
<li>D. G. Glynn, “The permanent of a square matrix,” <i>Eur. J. Combin.</i> 31, 1887–1891 (2010).</li>
<li>C. K. Hong, Z. Y. Ou, L. Mandel, “Measurement of subpicosecond time intervals between two photons by interference,” <i>Phys. Rev. Lett.</i> 59, 2044–2046 (1987).</li>
<li>F. Mezzadri, “How to generate random matrices from the classical compact groups,” <i>Notices Amer. Math. Soc.</i> 54, 592–604 (2007).</li>
<li>T. Jones, J. Gacon, “Efficient calculation of gradients in classical simulations of variational quantum algorithms,” arXiv:2009.02823 (2020).</li>
<li>Y. Suzuki et al., “Qulacs: a fast and versatile quantum circuit simulator for research purpose,” <i>Quantum</i> 5, 559 (2021).</li>
<li>V. Bergholm et al., “PennyLane: automatic differentiation of hybrid quantum-classical computations,” arXiv:1811.04968 (2018).</li>
<li>B. Gupt, J. Izaac, N. Quesada, “The Walrus: a library for the calculation of hafnians, Hermite polynomials and Gaussian boson sampling,” <i>J. Open Source Softw.</i> 4, 1705 (2019).</li>
</ol>
