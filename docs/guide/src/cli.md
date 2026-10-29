# Command line

`qso <command> [options]`. Every command accepts `--json` for
machine-readable output; `qso <command> --help` lists all options.

| Command | Purpose | Main options |
|---|---|---|
| `graph [SPEC]` | Structure, Laplacian spectrum, automorphism group and orbits, strongly regular parameters | `--save FILE` (edge list, or GraphML for `.graphml`/`.xml`) |
| `iso A B` | Isomorphism test with an explicit mapping | — |
| `walk` | Node probabilities at time t and limiting distribution | `--graph`, `--start`, `--time`, `--gamma`, `--generator laplacian\|adjacency` |
| `spins` | Nuclear spin dynamics: a spin system (JSON), a pulse sequence (JSON events, or one pulse on the observed isotope), and the stick spectrum and polarisations after it | `SYSTEM.json`, `--observe 13C`, `--sequence EVENTS.json`, `--pulse 90`, `--initial thermal\|parahydrogen:0,1\|pasadena:0,1` |
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
| `molecule FILE` | Qubit Hamiltonian, Hartree–Fock and full-CI energies of an FCIDUMP molecule, and a variational solution | `--mapping jordan-wigner\|parity\|bravyi-kitaev`, `--two-qubit-reduction`, `--method none\|uccsd\|upccgsd\|adapt`, `--k`, `--roots` |
| `trotter` | Trotter–Suzuki circuit for exp(−iHt) with its error bound | `--hamiltonian`, `--qubits`, `--time`, `--steps`, `--order`, `--output FILE` |
| `qec` | Repetition or surface-code memory experiment under circuit-level noise, or a circuit in Stim's format: logical error rate with union-find or matching decoding ([Error correction](qec.md)) | `--code surface\|repetition`, `--basis z\|x`, `--decoder union-find\|matching`, `--distance 3,5,7`, `--rounds`, `--p 0.001,0.01`, `--shots`, `--seed`, `--stim FILE`, `--circuit FILE` |

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
