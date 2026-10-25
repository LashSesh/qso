# Command line

`qso <command> [options]`. Every command accepts `--json` for
machine-readable output; `qso <command> --help` lists all options.

| Command | Purpose | Main options |
|---|---|---|
| `graph [SPEC]` | Structure, Laplacian spectrum, automorphism group and orbits, strongly regular parameters | `--save FILE` (edge list, or GraphML for `.graphml`/`.xml`) |
| `iso A B` | Isomorphism test with an explicit mapping | — |
| `walk` | Node probabilities at time t and limiting distribution | `--graph`, `--start`, `--time`, `--gamma`, `--generator laplacian\|adjacency` |
| `search` | Spatial search; rate and time are optimised unless `--gamma` is given | `--graph`, `--marked 3,7`, `--gamma`, `--generator`, `--t-max` |
| `reduce` | Equitable partition from a start node and the walk on it | `--graph`, `--start`, `--time`, `--gamma`, `--generator` |
| `transfer` | Perfect state transfer, or the best transfer within a window | `--graph`, `--from`, `--to`, `--generator`, `--t-max` |
| `coined` | Coined (Shenvi–Kempe–Whaley) search | `--graph`, `--marked`, `--coin grover\|fourier\|hadamard`, `--steps` |
| `hitting` | Szegedy quantum hitting time and the classical hitting time | `--graph`, `--marked`, `--max-steps` |
| `walk-circuit` | A walk as an OpenQASM 2.0 circuit | `--graph`, `--time`, `--method exact\|trotter`, `--steps`, `--order` |
| `qaoa` | QAOA for MaxCut with exact optimum and approximation ratio | `--graph`, `--p`, `--restarts`, `--seed`, `--layerwise` |
| `vqe` | VQE with an EfficientSU2 ansatz, compared with the exact ground energy | `--model h2\|tfim:N,J,H\|heisenberg:N,J,Δ`, `--hamiltonian "<Pauli sum>"`, `--layers`, `--method`, `--restarts` |
| `grover` | Grover search, simulated against theoretical success | `--qubits`, `--marked`, `--iterations` |
| `qasm FILE` | Simulate an OpenQASM 2.0 file and sample counts | `--shots`, `--seed`, `--noise p1,p2`, `--device FILE`, `--trajectories N` |
| `estimate FILE` | ⟨H⟩ of an OpenQASM 2.0 circuit with standard error and 95 % interval, optionally mitigated | `--observable`, `--shots`, `--allocation`, `--device`, `--trajectories`, `--mitigate-readout`, `--zne 1,3,5`, `--extrapolation`, `--pec N` |
| `molecule FILE` | Qubit Hamiltonian, Hartree–Fock and full-CI energies of an FCIDUMP molecule, and a variational solution | `--mapping jordan-wigner\|parity\|bravyi-kitaev`, `--two-qubit-reduction`, `--method none\|uccsd\|upccgsd\|adapt`, `--k`, `--roots` |
| `trotter` | Trotter–Suzuki circuit for exp(−iHt) with its error bound | `--hamiltonian`, `--qubits`, `--time`, `--steps`, `--order`, `--output FILE` |

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
