# Command line

`qso <command> [options]`. Every command accepts `--json` for
machine-readable output; `qso <command> --help` lists all options.

| Command | Purpose | Main options |
|---|---|---|
| `graph [SPEC]` | Structure, Laplacian spectrum, automorphism group and orbits, strongly regular parameters | `--save FILE` (edge list, or GraphML for `.graphml`/`.xml`) |
| `iso A B` | Isomorphism test with an explicit mapping | — |
| `walk` | Node probabilities at time t and limiting distribution | `--graph`, `--start`, `--time`, `--gamma`, `--generator laplacian\|adjacency`, `--energies 0.5,0,…` (site energies, one per node) |
| `transport` | Transport to a trap ([Quantum walks](walks.md#transport-to-a-trap)): efficiency, mean arrival time, loss and the weight never absorbed of an open walk with traps, losses, site energies and dephasing per node (dense, up to 2048 nodes); optionally a scan over a dephasing rate added to every node and the best such rate | `--graph` (default `path:2`), `--start`, `--trap 1,3`, `--kappa` (one for every trap or one per trap), `--loss`, `--dephasing` (one for every node or one per node), `--energies`, `--gamma`, `--generator` (default `adjacency`, like `particles`), `--scan 0,1,10`, `--best-rate LOW,HIGH`, `--dark-tolerance` |
| `spins` | Nuclear spin dynamics: a spin system (JSON), a pulse sequence (JSON events, or one pulse on the observed isotope), and the stick spectrum and polarisations after it | `SYSTEM.json`, `--observe 13C`, `--sequence EVENTS.json`, `--pulse 90`, `--initial thermal\|parahydrogen:0,1\|pasadena:0,1`, `--rates` (T₁ and T₂ of every spin) |
| `radical-pair` | Radical pairs ([Spin dynamics](spins.md#radical-pairs)): a spin system (JSON) with exactly two electrons and their nuclei; singlet, triplet and escape yields per pair born at each field, with the fields of the lowest and highest `Φ_S` and the half-effect field B½ of a sweep from 0 T (half the change of `Φ_S` from 0 T to the end of the sweep), and ⟨I_z⟩ of every nucleus in each product (CIDNP). JSON keys `electrons`, `nuclei`, `rates`, `recombination`, `birth`, `yields`, `sweep` (`singlet_lowest`, `singlet_highest`, `half_effect_field` or null, `half_effect_reference`; null for one field) | `SYSTEM.json` (e.g. `examples/spins/flavin_tryptophan.json`), `--rates KS,KT,KE` (s⁻¹, required), `--recombination haberkorn\|jones-maeda-hore`, `--birth singlet\|t0\|t+\|t-\|triplet\|uncorrelated` (also `triplet-zero`, `triplet-plus`, `triplet-minus`, as JSON writes them), `--fields B1,B2,…\|FROM:TO:POINTS` (T; default the system's field), `--cidnp` |
| `particles` | Walk of several bosons, fermions or distinguishable particles with interaction: site densities, and with `--compare GRAPH` the certificate distance against its tolerance | `--graph`, `--particles`, `--statistics bosons\|fermions\|distinguishable`, `--start 0,1`, `--time`, `--gamma`, `--generator` (default `adjacency`, unlike `walk` and `search`), `--on-site U`, `--neighbour V`, `--compare` |
| `search` | Spatial search; rate and time are optimised unless `--gamma` is given | `--graph`, `--marked 3,7`, `--gamma`, `--generator`, `--t-max` |
| `reduce` | Equitable partition from a start node and the walk on it | `--graph`, `--start`, `--time`, `--gamma`, `--generator` |
| `transfer` | Perfect state transfer, or the best transfer within a window | `--graph`, `--from`, `--to`, `--generator`, `--t-max` |
| `coined` | Coined (Shenvi–Kempe–Whaley) search | `--graph`, `--marked`, `--coin grover\|fourier\|hadamard`, `--steps` |
| `hitting` | Szegedy quantum hitting time and the classical hitting time | `--graph`, `--marked`, `--max-steps` |
| `walk-circuit` | A walk as an OpenQASM 2.0 circuit | `--graph`, `--time`, `--method exact\|trotter`, `--steps`, `--order` |
| `qaoa` | QAOA for MaxCut with exact optimum and approximation ratio | `--graph`, `--p`, `--restarts`, `--seed`, `--layerwise`, `--single` |
| `vqe` | VQE with an EfficientSU2 ansatz, compared with the exact ground energy | `--model h2\|tfim:N,J,H\|heisenberg:N,J,Δ`, `--hamiltonian "<Pauli sum>"`, `--layers`, `--method`, `--restarts`, `--single` |
| `grover` | Grover search, simulated against theoretical success | `--qubits`, `--marked`, `--iterations` |
| `qasm FILE` | Run an OpenQASM 2 or 3 program (mid-circuit measurement, reset, conditions) and sample its bits; or convert it | `--shots`, `--seed`, `--export qasm2\|qasm3`, `--stabilizer`, `--mps`, `--max-bond N`, `--noise p1,p2`, `--device FILE`, `--trajectories N` |
| `estimate FILE` | ⟨H⟩ of an OpenQASM circuit with standard error and 95 % interval, optionally mitigated | `--observable`, `--shots`, `--allocation`, `--device`, `--trajectories`, `--mitigate-readout`, `--zne 1,3,5`, `--extrapolation`, `--pec N` |
| `propagate FILE` | ⟨O⟩ of an OpenQASM circuit on up to 16384 qubits by Pauli propagation (gates on at most four qubits; final measurements are left out, others, resets and conditions refused): exact, or truncated with a rigorous bound on the error; with a device, its gate rules, idle relaxation and spectators as channels (readout errors are not applied). JSON keys `value`, `truncation_bound`, `terms`, `peak_terms`, `dropped_terms`, `operator` | `--observable`, `--device FILE` (up to 16384 per-qubit entries), `--state BITS` (qubit 0 on the right), `--min-coefficient`, `--max-weight`, `--max-terms`, `--operator` |
| `molecule FILE` | Qubit Hamiltonian, Hartree–Fock and full-CI energies of an FCIDUMP molecule, and a variational solution | `--mapping jordan-wigner\|parity\|bravyi-kitaev`, `--two-qubit-reduction`, `--method none\|uccsd\|upccgsd\|adapt`, `--k`, `--roots` |
| `trotter` | Trotter–Suzuki circuit for exp(−iHt) with its error bound | `--hamiltonian`, `--qubits`, `--time`, `--steps`, `--order`, `--output FILE` |
| `qec` | Repetition, surface, colour or bivariate bicycle code memory experiment under circuit-level noise, optionally with a share of heralded erasures, or a circuit in Stim's format: logical error rate with union-find, matching, correlated matching or BP+OSD decoding, conditioned on the heralds that fired; optionally the exact circuit distance, or bounds when `--max-seconds` runs out ([Error correction](qec.md)). JSON keys per row include `detectors`, `herald_detectors`, `erasure_share` (null with `--circuit`), `logical_error_rate` and `circuit_distance` | `--code surface\|repetition\|color\|bb`, `--basis z\|x`, `--decoder union-find\|matching\|correlated-matching\|bposd`, `--erasure-share`, `--window`, `--commit`, `--parallel`, `--buffer`, `--schedule greedy\|bicycle\|0,1,2,3,4,5`, `--circuit-distance`, `--max-seconds`, `--symmetry`, `--distance 3,5,7`, `--rounds`, `--p 0.001,0.01`, `--shots`, `--seed`, `--stim FILE`, `--circuit FILE` |

Graphs are given as specifications (see [Graphs](graphs.md)), for example
`metatron`, `fruit-of-life`, `petersen`, `hypercube:8`, `johnson:7,3` or
`random-regular:16,3,1`, or as files: an edge list, or GraphML for
`.graphml` and `.xml` (`file:PATH` forces a file).

```text
$ qso graph fruit-of-life
Fruit of Life (contact graph)
  nodes 13, edges 18, degree 1..6, connected true, bipartite false
  diameter Some(4), triangles 6, algebraic connectivity 0.585786
  |Aut| = 12 (log10 1.079), orbits 3

$ qso iso shrikhande rook:4x4
Shrikhande and rook(4x4) are not isomorphic

$ qso transfer --graph path:4 --t-max 30
P4: no perfect state transfer 0 → 3 (strongly cospectral: true); best fidelity 0.992357 at t = 28.099259 within [0, 30]

$ qso hitting --graph cycle:64
C64: Szegedy quantum hitting time 16, classical hitting time 682.5000

$ qso qaoa --graph petersen --p 4 --layerwise
p=1  <C>=10.386751  C_max=12  ratio=0.865563  P(optimal)=0.1682  most likely 1110001011 (cut 12)
p=2  <C>=11.105320  C_max=12  ratio=0.925443  P(optimal)=0.4490  most likely 1110001011 (cut 12)
p=3  <C>=11.648892  C_max=12  ratio=0.970741  P(optimal)=0.7626  most likely 1100110110 (cut 12)
p=4  <C>=11.936517  C_max=12  ratio=0.994710  P(optimal)=0.9767  most likely 1001101101 (cut 12)
```

`transport` solves the [transport to a trap](walks.md#transport-to-a-trap)
on the dense path. The detuned dimer, start at energy 3 and trap at 0,
with loss 0.05 on both sites, a scan over a dephasing rate added to every
node and the best such rate:

```text
$ qso transport --graph path:2 --trap 1 --loss 0.05 --energies 3,0 --scan 0,1,3,10,100 --best-rate 0.01,100
transport from node 0 to trap 1 on P2:
  efficiency 0.771278 (lost 0.228722, never absorbed 0)
  mean arrival time 4.525622
  [bright dimension 2, slowest decay 0.208451, condition 2.23e1, error 0.0e0]
dephasing rate added to every node:
  rate 0: efficiency 0.771278, arrival 4.525622, lost 0.228722, never absorbed 0
  rate 1: efficiency 0.823024, arrival 3.581698, lost 0.176976, never absorbed 0
  rate 3: efficiency 0.825951, arrival 3.555290, lost 0.174049, never absorbed 0
  rate 10: efficiency 0.738538, arrival 5.356309, lost 0.261462, never absorbed 0
  rate 100: efficiency 0.274598, arrival 14.862230, lost 0.725402, never absorbed 0
best rate 1.950000: efficiency 0.830737, arrival 3.447560
```

With `--dephasing` the walk dephases on its own, and `--best-rate`
reports the rate added on top of that.

`qasm` runs a program exactly, mid-circuit measurements and conditions
included, and samples its classical bits (every qubit if it has none).
The OpenQASM 3 teleportation example of the OpenQASM project sends
`U(0.3, 0.2, 0.1)|0⟩`, so bit 2 (the left digit) reads 1 with probability
sin²(0.15) = 0.022:

```text
$ qso qasm teleport.qasm --shots 1000 --seed 2
3 qubits, 3 bits, 15 operations, backend program
  000  258
  001  245
  011  245
  010  239
  111  5
  100  4
  101  2
  110  2
```

Clifford programs too large for a state vector run on the
[stabilizer simulator](stabilizers.md); `--stabilizer` chooses it for
any Clifford program. With `--noise` or `--device` the program must be a circuit; `--export
qasm2` or `--export qasm3` prints the program in that version instead of
running it.

Device files are JSON (see [Noise](noise.md)). With a device, `estimate`
prints the estimate, the noiseless value for comparison and the mitigated
estimates. For a three-qubit GHZ state on the example device
(`metatron-qso-rs/examples/devices/toy_device.json`):

```text
$ qso estimate ghz.qasm --observable "Z0 Z1 + Z1 Z2" --device toy_device.json --shots 50000
<H> = 1.738600 ± 0.003859 (95% CI [1.731036, 1.746164], 50000 samples)
noiseless <H> = 2.000000

$ qso estimate ghz.qasm --observable "Z0 Z1 + Z1 Z2" --device toy_device.json --shots 50000 \
      --mitigate-readout --zne 1,3,5
<H> (readout mitigated) = 1.963079 ± 0.004379 (95% CI [1.954496, 1.971662], 50000 samples)
noiseless <H> = 2.000000
ZNE ([1.0, 3.0, 5.0]) <H> = 1.998613 ± 0.010336 (95% CI [1.978355, 2.018871], 150000 samples)
```

`propagate` computes one expectation value without shots on circuits of
up to 16384 qubits by carrying the observable backwards as Pauli strings
(see [Pauli propagation](propagation.md)). A device file may list up to
16384 qubits there. For a 100-qubit GHZ circuit followed by RY(0.2) on
every qubit, exactly and on a device with T₁ = 100, T₂ = 80 and
depolarising noise on every `cx` (`chain.json`):

```text
$ qso propagate ghz100.qasm --observable "Z0 Z99 + 0.5*X50"
<O> = 0.960530497001 (truncation bound 0.000e0)
100 qubits, 1 layer; 6 strings at the start, at most 6 held, 0 dropped

$ qso propagate ghz100.qasm --observable "Z0 Z99" --device chain.json
<O> = 0.224540781668 (truncation bound 1.702e-15)
100 qubits, 791 layers; 303 strings at the start, at most 303 held, 0 dropped
```

`qec --erasure-share e` turns the share e of every noise probability of
a generated memory experiment into heralded erasures (Stim's
`HERALDED_ERASE`) of each qubit of the operation, each herald a detector
of its round, and leaves the rest, (1 − e)·p, to the usual
circuit-level noise (see [Error correction](qec.md)). Union-find,
matching and BP+OSD then condition on the heralds that fired, shot by
shot, and the decoder's name says "with heralds". Circuit files with
`HERALDED_ERASE` or `HERALDED_PAULI_CHANNEL_1` are decoded in the same
way without the option, which `--circuit` refuses. At p = 0.01 on the
rotated surface code, nine tenths of the noise heralded:

```text
$ qso qec --distance 3,5 --p 0.01 --shots 10000 --decoder matching
p = 0.01, d = 3, 3 rounds: 17 qubits, 16 detectors; logical error rate 6.030e-2 ± 2.4e-3 (603 of 10000 shots, matching)
p = 0.01, d = 5, 5 rounds: 49 qubits, 72 detectors; logical error rate 8.370e-2 ± 2.8e-3 (837 of 10000 shots, matching)

$ qso qec --distance 3,5 --p 0.01 --erasure-share 0.9 --shots 10000 --decoder matching
p = 0.01, d = 3, 3 rounds, erasure share 0.9: 17 qubits, 285 detectors; logical error rate 5.900e-3 ± 7.7e-4 (59 of 10000 shots, matching with heralds)
p = 0.01, d = 5, 5 rounds, erasure share 0.9: 49 qubits, 1431 detectors; logical error rate 2.100e-3 ± 4.6e-4 (21 of 10000 shots, matching with heralds)
```

The detector counts include the herald detectors. Each JSON row adds
`erasure_share` (null for `--circuit`) and `herald_detectors`, the
detectors the decoder conditions on. Circuits with heralded operations
are refused, with the library's error, by `--circuit-distance` ("the
circuit distance does not take heralded operations yet"), and circuits
with herald detectors by `--window`, `--parallel` and `--decoder
correlated-matching` ("does not use heralds yet"); heralds that no
detector reads alone enter the model by Stim's rule and decode with any
decoder.

`molecule` reads an FCIDUMP file (see [Electronic structure](chemistry.md))
and `trotter` builds a product-formula circuit (see
[Hamiltonian simulation](simulation.md)):

```text
$ qso molecule beh2_sto3g.fcidump --mapping parity --two-qubit-reduction --method adapt
7 orbitals, 3 + 3 electrons; 12 qubits (parity, two-qubit reduction), 666 Pauli terms
Hartree–Fock   -15.5603349360
full CI        -15.5951823567
ADAPT-VQE      -15.5948030942  (36 parameters, error vs full CI 3.79e-4)

$ qso trotter --hamiltonian "Z0 Z1 + Z1 Z2 + 0.7 X0 + 0.7 X1 + 0.7 X2" --time 2 --steps 20 --order 4
order 4, 20 steps, t = 2: 801 gates, depth 501, error bound 2.286e-2, largest column error 5.687e-6 (a lower bound on ‖U − exp(−iHt)‖)
```
