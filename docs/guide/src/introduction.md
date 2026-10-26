# QSO User Guide

QSO is a quantum simulator written in pure Rust, with a Python package and a
command-line tool. It simulates quantum circuits exactly, as pure states or as
density matrices or quantum trajectories with device noise, mitigates errors
and estimates results with error bars, and runs variational and textbook
algorithms on top of them: VQE and QAOA, excited states, Hamiltonian
simulation, electronic structure and amplitude estimation. It also
simulates continuous- and discrete-time quantum walks on arbitrary graphs —
reduced by symmetry where possible, with state-transfer analysis and
compilation to circuits — and samples photons in linear interferometers.
Its signature geometry is **Metatron's Cube**, the complete graph K₁₃ on the
13 centres of the Fruit of Life.

The rule of the project is **no claim without a test**. Every algorithm is
compared with an independent reference: a closed form, a published value, a
dense-matrix computation, a second method or another simulator. The chapter
[How QSO is validated](validation.md) describes the checks, and the
[technical note](https://github.com/LashSesh/qso/tree/main/docs/technical-note)
reports them in full.

## What this guide covers

| Chapter | Topics |
|---|---|
| [Circuits and states](circuits.md) | Gates, simulation, sampling, observables, OpenQASM |
| [OpenQASM programs](programs.md) | OpenQASM 2 and 3, mid-circuit measurement, reset, feed-forward |
| [Stabilizer simulation and graph states](stabilizers.md) | Clifford circuits on thousands of qubits, graph states, local complementation, entanglement from cut-ranks |
| [Error-correction experiments](qec.md) | Noisy Clifford circuits with detectors, Pauli-frame sampling, detector error models, union-find decoding, repetition and surface-code memory experiments |
| [Parameters and gradients](gradients.md) | Symbolic parameters, adjoint and parameter-shift gradients |
| [Variational algorithms](variational.md) | VQE, QAOA, variational and graph classifiers |
| [Excited states and state geometry](excited.md) | VQD, SSVQE, quantum Fisher information, natural gradient, imaginary-time evolution |
| [Hamiltonian simulation](simulation.md) | Trotter–Suzuki circuits with error bounds, qDRIFT |
| [Electronic structure](chemistry.md) | Fermionic operators, qubit mappings, FCIDUMP molecules, full CI, UCCSD, k-UpCCGSD, ADAPT-VQE |
| [Optimisation problems and amplitude estimation](optimization.md) | QUBO/Ising, QAOA mixers, canonical, iterative and maximum-likelihood amplitude estimation |
| [Noise](noise.md) | Channels, noise models, device models, density matrices, quantum trajectories |
| [Estimation and error mitigation](mitigation.md) | Estimates with error bars, shot allocation, readout mitigation, ZNE, PEC, twirling, classical shadows |
| [Graphs](graphs.md) | Graph library, specifications, symmetry groups, isomorphism, files |
| [Quantum walks](walks.md) | Continuous-time walks, spatial search, dephasing |
| [Symmetry and state transfer](symmetry.md) | Quotients by equitable partitions, perfect state transfer |
| [Discrete-time walks and circuits](discrete.md) | Coined, line and Szegedy walks; walks as circuits |
| [Boson sampling](photonics.md) | Permanents, interferometers, exact sampling |
| [WebAssembly and JavaScript](webassembly.md) | The npm package and the browser playground |
| [Command line](cli.md) | The `qso` tool |

The **tutorials** are Jupyter notebooks from
[`docs/tutorials`](https://github.com/LashSesh/qso/tree/main/docs/tutorials).
They are executed every time this site is built, so their output always comes
from the current code.

Examples use the Python package, with Rust equivalents where the interfaces
differ. The two share one implementation. Every code block in this guide is
run when the site is built.

## Versions

This guide describes the code on the `main` branch. Released versions are on
[PyPI](https://pypi.org/project/metatron-qso/) and
[crates.io](https://crates.io/crates/metatron-qso-rs); the
[changelog](https://github.com/LashSesh/qso/blob/main/CHANGELOG.md) lists what
changed in each release.
