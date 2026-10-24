# Command line

`qso <command> [options]`. Every command accepts `--json` for
machine-readable output; `qso <command> --help` lists all options.

| Command | Purpose | Main options |
|---|---|---|
| `graph [SPEC]` | Structure, Laplacian spectrum, automorphism group and orbits | — |
| `walk` | Node probabilities at time t and limiting distribution | `--graph`, `--start`, `--time`, `--gamma`, `--generator laplacian\|adjacency` |
| `search` | Spatial search; rate and time are optimised unless `--gamma` is given | `--graph`, `--marked 3,7`, `--gamma`, `--generator`, `--t-max` |
| `qaoa` | QAOA for MaxCut with exact optimum and approximation ratio | `--graph`, `--p`, `--restarts`, `--seed`, `--layerwise` |
| `vqe` | VQE with an EfficientSU2 ansatz, compared with the exact ground energy | `--model h2\|tfim:N,J,H\|heisenberg:N,J,Δ`, `--hamiltonian "<Pauli sum>"`, `--layers`, `--method`, `--restarts` |
| `grover` | Grover search, simulated against theoretical success | `--qubits`, `--marked`, `--iterations` |
| `qasm FILE` | Simulate an OpenQASM 2.0 file and sample counts | `--shots`, `--seed`, `--noise p1,p2` |

Graphs are given as specifications (see [Graphs](graphs.md)), for example
`metatron`, `fruit-of-life`, `petersen`, `hypercube:8` or
`random-regular:16,3,1`.

```text
$ qso graph fruit-of-life
Fruit of Life (contact graph)
  nodes 13, edges 18, degree 1..6, connected true, bipartite false
  diameter Some(4), triangles 6, algebraic connectivity 0.585786
  |Aut| = 12 (log10 1.079), orbits 3

$ qso qaoa --graph petersen --p 4 --layerwise
p=1  <C>=10.386751  C_max=12  ratio=0.865563  P(optimal)=0.1682  most likely 1110001011 (cut 12)
p=2  <C>=11.105320  C_max=12  ratio=0.925443  P(optimal)=0.4490  most likely 1110001011 (cut 12)
p=3  <C>=11.648892  C_max=12  ratio=0.970741  P(optimal)=0.7626  most likely 1100110110 (cut 12)
p=4  <C>=11.936517  C_max=12  ratio=0.994710  P(optimal)=0.9767  most likely 1001101101 (cut 12)
```
