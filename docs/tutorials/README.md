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
