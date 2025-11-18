# Metatron QSO Python SDK

Python bindings for the Metatron Quantum State Operator - a high-performance quantum computing framework built on sacred geometry.

## Overview

The `metatron_qso` Python package provides an easy-to-use interface to the Rust-based quantum computing core, enabling:

- **Quantum Walks**: Continuous-time quantum walks on the Metatron Cube
- **QAOA**: Quantum Approximate Optimization Algorithm for combinatorial problems
- **VQE**: Variational Quantum Eigensolver for ground state energy
- **High Performance**: Leveraging Rust's speed with Python's ease of use

## Installation

### From Source (Development)

Requirements:
- Python 3.8+
- Rust toolchain (install from [rustup.rs](https://rustup.rs))
- maturin (`pip install maturin`)

```bash
# Clone the repository
cd metatron_qso_py

# Install in development mode
maturin develop --release

# Or build a wheel
maturin build --release
```

### Using pip (Future)

```bash
pip install metatron-qso
```

## Quick Start

```python
import metatron_qso

# Create the Metatron Cube graph
graph = metatron_qso.MetatronGraph()
print(graph)  # MetatronGraph(nodes=13, edges=78)

# Run a quantum walk
result = metatron_qso.run_quantum_walk(
    graph=graph,
    source_nodes=[0],  # Start at central node
    t_max=5.0,
    dt=0.1
)

# Access results
print(f"Final probabilities: {result['final_state']}")
```

## API Reference

### MetatronGraph

The core graph structure representing the Metatron Cube.

```python
# Create the default Metatron graph
graph = metatron_qso.MetatronGraph()

# Get graph properties
num_nodes = graph.num_nodes()      # Returns: 13
num_edges = graph.num_edges()      # Returns: 78
adj_list = graph.adjacency_list()  # Returns: list of lists
```

### run_quantum_walk

Execute a continuous-time quantum walk.

```python
result = metatron_qso.run_quantum_walk(
    graph,              # MetatronGraph instance
    source_nodes,       # List of initial nodes (e.g., [0] or [1, 2, 3])
    t_max=10.0,        # Maximum evolution time
    dt=0.1             # Time step
)

# Returns dictionary with:
# - 'times': List of time points
# - 'probabilities': List of probability distributions
# - 'final_state': Final probability distribution
```

### solve_maxcut_qaoa

Solve MaxCut optimization using QAOA.

```python
result = metatron_qso.solve_maxcut_qaoa(
    graph,              # MetatronGraph instance
    depth=3,           # QAOA circuit depth (p)
    max_iters=100      # Maximum optimization iterations
)

# Returns dictionary with:
# - 'cut_value': Best cut value found
# - 'assignment': Binary node assignment (list of 0s and 1s)
# - 'approximation_ratio': Solution quality (0 to 1)
# - 'meta': Additional optimization metadata
```

### run_vqe

Find ground state energy using VQE.

```python
result = metatron_qso.run_vqe(
    graph,                              # MetatronGraph instance
    depth=2,                           # Ansatz depth
    max_iters=100,                     # Maximum iterations
    ansatz_type="hardware_efficient"   # Ansatz: "hardware_efficient",
                                       #         "metatron", or "efficient_su2"
)

# Returns dictionary with:
# - 'ground_state_energy': Computed ground state energy
# - 'classical_ground_energy': Exact energy for comparison
# - 'error': Absolute error
# - 'iterations': Number of optimization steps
# - 'final_state': Final quantum state probabilities
```

## Examples

The `examples/` directory contains complete demonstrations:

- **01_quantum_walk_basic.py**: Basic quantum walk demo
- **02_qaoa_maxcut_basic.py**: MaxCut optimization with QAOA
- **03_vqe_ground_state.py**: Ground state energy computation

Run an example:

```bash
cd metatron_qso_py
python examples/01_quantum_walk_basic.py
```

## Jupyter Notebooks

Interactive tutorials are available in `notebooks/`:

- **QuantumWalk_Intro.ipynb**: Comprehensive quantum walk tutorial with visualizations

To use notebooks:

```bash
pip install jupyter matplotlib numpy
cd metatron_qso_py/notebooks
jupyter notebook QuantumWalk_Intro.ipynb
```

## Performance Notes

- **Rust Backend**: All computationally intensive operations run in optimized Rust code
- **Parallelization**: Many operations use rayon for multi-threaded execution
- **Memory Efficiency**: Minimal data copying between Python and Rust

Typical performance (Intel i7-12700K):
- Quantum Walk (single step): ~31 μs
- QAOA iteration (depth=3): ~2-5 ms
- VQE iteration (depth=2): ~3-8 ms

## Development

### Running Tests

```bash
# Rust tests
cd metatron_qso_py
cargo test

# Python tests (if implemented)
pytest tests/
```

### Building Documentation

```bash
# Rust documentation
cargo doc --open

# Python documentation
pdoc metatron_qso
```

## Architecture

The Python SDK architecture:

```
┌─────────────────────────────────────┐
│   Python API (metatron_qso)        │
│   - MetatronGraph class             │
│   - run_quantum_walk()              │
│   - solve_maxcut_qaoa()             │
│   - run_vqe()                       │
└─────────────────────────────────────┘
                ↓ PyO3
┌─────────────────────────────────────┐
│   Rust Core (metatron-qso-rs)      │
│   - Graph structures                │
│   - Quantum state evolution         │
│   - VQA algorithms                  │
│   - Linear algebra (nalgebra)       │
└─────────────────────────────────────┘
```

## Related Documentation

- [Rust Core Guide](../metatron-qso-rs/docs/RUST_CORE_GUIDE.md)
- [Python SDK Guide](../docs/PYTHON_SDK_GUIDE.md)
- [VQA Implementation](../VQA_IMPLEMENTATION_GUIDE.md)
- [Root README](../README.md)

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/LashSesh/qso/issues)
- **Documentation**: [Project Docs](https://github.com/LashSesh/qso/tree/main/docs)

## Citation

If you use Metatron QSO in your research, please cite:

```bibtex
@software{metatron_qso,
  title = {Metatron Quantum State Operator},
  author = {Q⊗DASH Team},
  year = {2024},
  url = {https://github.com/LashSesh/qso}
}
```
