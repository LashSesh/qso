# Tutorials

Jupyter notebooks that walk through QSO with plots and comparisons against
exact results. They are stored without outputs and executed whenever the
[documentation site](https://lashsesh.github.io/qso/) is built, so the
published versions always show the output of the current code.

| Notebook | Content |
|---|---|
| [01_circuits_and_gradients](01_circuits_and_gradients.ipynb) | GHZ states and sampling, an energy landscape with exact derivatives, the cost of adjoint vs. parameter-shift gradients, OpenQASM |
| [02_vqe_and_qaoa](02_vqe_and_qaoa.ipynb) | VQE for H₂ and a transverse-field Ising chain against exact energies; QAOA depth series on rings, the Petersen graph and Metatron's Cube |
| [03_noise](03_noise.ipynb) | Depolarising noise on a Bell state, T₁/T₂ decay against closed forms, shot noise, readout errors |
| [04_metatron_walks](04_metatron_walks.ipynb) | Symmetry of Metatron's Cube, walks from the centre, spatial search, the transition to classical walks under dephasing |
| [05_symmetry_transfer_discrete](05_symmetry_transfer_discrete.ipynb) | Isomorphism of cospectral graphs, a walk on the 2³⁰-node hypercube in 31 dimensions, perfect state transfer on paths, the Hadamard walk, coined and Szegedy search, walk circuits and product-formula errors, the Metatron family |
| [06_noise_mitigation](06_noise_mitigation.ipynb) | A device model with T₁/T₂, gate and readout errors; trajectories converging to the density matrix; shot allocation; readout mitigation, zero-noise extrapolation, probabilistic error cancellation and Pauli twirling against noiseless values; classical shadows; trajectories on 16 qubits |
| [07_chemistry_dynamics_estimation](07_chemistry_dynamics_estimation.ipynb) | LiH under four qubit encodings against PySCF's full CI; UCCSD and ADAPT-VQE; VQD excited states of H₂; product-formula errors against their bounds; imaginary-time evolution; QAOA with a weight-conserving XY mixer; the 1/N error scaling of amplitude estimation |
| [09_hyperpolarisation](09_hyperpolarisation.ipynb) | Thermal polarisation; a free induction decay and its spectrum; INEPT; PASADENA and ALTADENA spectra; the field dependence of SABRE for ¹H and ¹⁵N; SABRE with chemical exchange; transfer to ¹³C by field cycling; the decay of hyperpolarisation by T₁; T₁ of ¹³C from Redfield theory, against field and deuteration |
| [10_optimal_control](10_optimal_control.ipynb) | GRAPE against the unitary bound of polarisation transfer and the time-optimal limit of two coupled spins; a pulse robust to rf errors against INEPT; a field profile that carries parahydrogen order to ¹³C in two seconds |
| [08_interacting_walks](08_interacting_walks.ipynb) | The Hong–Ou–Mandel dip of bosons, fermions and distinguishable particles on an edge; pairs bound by repulsion; light cones and correlations of two particles on a chain; T(8), the Chang graphs and the Shrikhande graph told apart by interacting bosons |

To run them locally:

```bash
pip install metatron-qso matplotlib jupyter
jupyter lab docs/tutorials
```

To build the whole site (guide, tutorials, Python reference):

```bash
pip install -r docs/requirements.txt maturin
maturin develop --release -m metatron_qso_py/Cargo.toml
cargo install mdbook
python docs/build_site.py --out site
```
