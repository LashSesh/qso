# Python SDK Guide

Complete guide for using the Metatron QSO Python SDK.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [API Reference](#api-reference)
5. [Advanced Usage](#advanced-usage)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- **Python**: Version 3.8 or higher
- **Rust**: Install via [rustup.rs](https://rustup.rs)
- **maturin**: Python-Rust build tool

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin
```

### Development Installation

For development or local use:

```bash
# Navigate to Python SDK directory
cd metatron_qso_py

# Build and install in development mode (editable)
maturin develop --release

# Verify installation
python -c "import metatron_qso; print(metatron_qso.__version__)"
```

### Building Wheels

To create distributable wheels:

```bash
cd metatron_qso_py

# Build wheel for current platform
maturin build --release

# The wheel will be in target/wheels/
ls target/wheels/
```

### Installation via pip (Future)

Once published:

```bash
pip install metatron-qso
```

---

## Quick Start

### Minimal Example

```python
import metatron_qso

# Create graph
graph = metatron_qso.MetatronGraph()

# Run quantum walk
result = metatron_qso.run_quantum_walk(graph, [0], t_max=5.0, dt=0.1)
print(result['final_state'])
```

### Complete Workflow

```python
import metatron_qso

# 1. Create the Metatron Cube graph
graph = metatron_qso.MetatronGraph()
print(f"Graph: {graph.num_nodes()} nodes, {graph.num_edges()} edges")

# 2. Run a quantum walk
qw_result = metatron_qso.run_quantum_walk(
    graph=graph,
    source_nodes=[0],  # Start at central node
    t_max=10.0,
    dt=0.1
)

# 3. Solve MaxCut with QAOA
qaoa_result = metatron_qso.solve_maxcut_qaoa(
    graph=graph,
    depth=3,
    max_iters=100
)

# 4. Find ground state with VQE
vqe_result = metatron_qso.run_vqe(
    graph=graph,
    depth=2,
    max_iters=150,
    ansatz_type="hardware_efficient"
)

# Display results
print(f"QAOA cut value: {qaoa_result['cut_value']:.4f}")
print(f"VQE ground energy: {vqe_result['ground_state_energy']:.10f}")
```

---

## Core Concepts

### The Metatron Cube

The Metatron Cube is a 13-node graph representing sacred geometry:

```
Structure:
- 1 central node (0)
- 6 hexagon vertices (1-6)
- 6 cube vertices (7-12)
- 78 edges (highly connected)
```

Properties:
- **High symmetry**: Contains all Platonic solids
- **Strong connectivity**: Average degree ≈ 12
- **Quantum advantage**: Ideal for quantum walks and VQA

### Quantum Walks

Quantum walks are quantum analogs of random walks, using:
- **Superposition**: State exists across multiple nodes
- **Interference**: Constructive/destructive probability amplitudes
- **Unitary evolution**: Reversible, coherent dynamics

### Variational Quantum Algorithms (VQA)

VQA methods optimize parameterized quantum circuits:
- **VQE**: Finds ground state energy of Hamiltonians
- **QAOA**: Solves combinatorial optimization (MaxCut, etc.)
- **VQC**: Quantum machine learning classifier

---

## API Reference

### MetatronGraph Class

#### Constructor

```python
graph = metatron_qso.MetatronGraph()
```

Creates the default 13-node Metatron Cube graph.

#### Methods

**`num_nodes() -> int`**

Returns the number of nodes (always 13 for Metatron Cube).

```python
n = graph.num_nodes()  # 13
```

**`num_edges() -> int`**

Returns the number of edges (78 for Metatron Cube).

```python
e = graph.num_edges()  # 78
```

**`adjacency_list() -> list[list[int]]`**

Returns adjacency list representation.

```python
adj = graph.adjacency_list()
print(adj[0])  # Neighbors of node 0
```

---

### run_quantum_walk Function

Execute a continuous-time quantum walk.

#### Signature

```python
run_quantum_walk(
    graph: MetatronGraph,
    source_nodes: list[int],
    t_max: float = 10.0,
    dt: float = 0.1
) -> dict
```

#### Parameters

- **graph**: MetatronGraph instance
- **source_nodes**: List of initial node indices (uniform superposition)
- **t_max**: Maximum evolution time (default: 10.0)
- **dt**: Time step size (default: 0.1)

#### Returns

Dictionary with keys:
- **'times'**: `list[float]` - Time points
- **'probabilities'**: `list[list[float]]` - Probability distributions at each time
- **'final_state'**: `list[float]` - Final probability distribution

#### Example

```python
result = metatron_qso.run_quantum_walk(
    graph=graph,
    source_nodes=[0, 1, 2],  # Start from 3 nodes
    t_max=8.0,
    dt=0.05
)

# Analyze spreading
import numpy as np
entropy = -np.sum([p * np.log(p) for p in result['final_state'] if p > 0])
print(f"Entropy: {entropy:.4f}")
```

#### Error Handling

```python
try:
    result = metatron_qso.run_quantum_walk(graph, [0], t_max=-1.0)
except ValueError as e:
    print(f"Error: {e}")  # "t_max must be positive"
```

---

### solve_maxcut_qaoa Function

Solve the MaxCut problem using QAOA.

#### Signature

```python
solve_maxcut_qaoa(
    graph: MetatronGraph,
    depth: int = 3,
    max_iters: int = 100
) -> dict
```

#### Parameters

- **graph**: MetatronGraph instance
- **depth**: QAOA circuit depth p (default: 3)
- **max_iters**: Maximum optimization iterations (default: 100)

#### Returns

Dictionary with keys:
- **'cut_value'**: `float` - Best cut value found
- **'assignment'**: `list[int]` - Binary node assignment (0 or 1)
- **'approximation_ratio'**: `float` - Quality metric (0 to 1)
- **'meta'**: `dict` - Optimization metadata
  - `'iterations'`: Number of optimizer steps
  - `'mean_cost'`: Average cost from sampling
  - `'std_dev'`: Standard deviation
  - `'depth'`: Circuit depth used

#### Example

```python
result = metatron_qso.solve_maxcut_qaoa(
    graph=graph,
    depth=3,
    max_iters=200
)

print(f"Cut value: {result['cut_value']:.4f}")
print(f"Approximation ratio: {result['approximation_ratio']:.4f}")

# Visualize partition
assignment = result['assignment']
set_0 = [i for i, val in enumerate(assignment) if val == 0]
set_1 = [i for i, val in enumerate(assignment) if val == 1]
print(f"Partition: {set_0} | {set_1}")
```

---

### run_vqe Function

Find ground state energy using VQE.

#### Signature

```python
run_vqe(
    graph: MetatronGraph,
    depth: int = 2,
    max_iters: int = 100,
    ansatz_type: str = "hardware_efficient"
) -> dict
```

#### Parameters

- **graph**: MetatronGraph instance
- **depth**: Ansatz circuit depth (default: 2)
- **max_iters**: Maximum iterations (default: 100)
- **ansatz_type**: Type of variational ansatz
  - `"hardware_efficient"`: General-purpose ansatz
  - `"metatron"`: Optimized for Metatron symmetry
  - `"efficient_su2"`: Efficient SU(2) ansatz

#### Returns

Dictionary with keys:
- **'ground_state_energy'**: `float` - Computed energy
- **'classical_ground_energy'**: `float` - Exact ground energy
- **'error'**: `float` - Absolute error
- **'iterations'**: `int` - Optimization steps
- **'final_state'**: `list[float]` - Final state probabilities

#### Example

```python
# Compare different ansätze
ansatz_types = ["hardware_efficient", "metatron", "efficient_su2"]

for ansatz in ansatz_types:
    result = metatron_qso.run_vqe(
        graph=graph,
        depth=2,
        max_iters=150,
        ansatz_type=ansatz
    )
    print(f"{ansatz}: E = {result['ground_state_energy']:.10f}, "
          f"error = {result['error']:.2e}")
```

---

## Advanced Usage

### Custom Initial States

```python
# Multi-node superposition
result = metatron_qso.run_quantum_walk(
    graph=graph,
    source_nodes=[1, 2, 3, 4, 5, 6],  # Hexagon layer
    t_max=5.0,
    dt=0.1
)
```

### Parameter Sweeps

```python
# QAOA depth sweep
depths = [1, 2, 3, 4, 5]
results = []

for depth in depths:
    result = metatron_qso.solve_maxcut_qaoa(graph, depth=depth, max_iters=100)
    results.append(result['approximation_ratio'])

# Plot results
import matplotlib.pyplot as plt
plt.plot(depths, results, marker='o')
plt.xlabel('QAOA Depth')
plt.ylabel('Approximation Ratio')
plt.show()
```

### Integration with NumPy

```python
import numpy as np

result = metatron_qso.run_quantum_walk(graph, [0], t_max=10.0, dt=0.1)

# Convert to NumPy arrays
times = np.array(result['times'])
probs = np.array(result['probabilities'])

# Analyze dynamics
central_prob = probs[:, 0]
fft = np.fft.fft(central_prob)
print(f"Dominant frequency: {np.argmax(np.abs(fft))}")
```

---

## Performance Optimization

### Tips

1. **Use Release Mode**: Always build with `--release` for production

```bash
maturin develop --release
```

2. **Adjust Time Steps**: Larger `dt` = faster but less accurate

```python
# Fast (coarse)
result = metatron_qso.run_quantum_walk(graph, [0], t_max=10.0, dt=0.5)

# Accurate (fine)
result = metatron_qso.run_quantum_walk(graph, [0], t_max=10.0, dt=0.01)
```

3. **Limit Iterations**: Set realistic `max_iters` for VQA

```python
# Quick test
result = metatron_qso.run_vqe(graph, depth=1, max_iters=50)

# Production
result = metatron_qso.run_vqe(graph, depth=3, max_iters=500)
```

### Benchmarking

```python
import time

start = time.time()
result = metatron_qso.run_quantum_walk(graph, [0], t_max=5.0, dt=0.1)
elapsed = time.time() - start

print(f"Quantum walk completed in {elapsed:.4f} seconds")
print(f"Steps: {len(result['times'])}")
print(f"Time per step: {elapsed / len(result['times']) * 1000:.2f} ms")
```

---

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'metatron_qso'`

**Solution**: Rebuild with maturin

```bash
cd metatron_qso_py
maturin develop --release
```

---

**Issue**: `ValueError: t_max must be positive`

**Solution**: Check function arguments

```python
# Wrong
result = metatron_qso.run_quantum_walk(graph, [0], t_max=-1.0)

# Correct
result = metatron_qso.run_quantum_walk(graph, [0], t_max=5.0)
```

---

**Issue**: Slow performance

**Solution**: Ensure release build

```bash
# Debug (slow)
maturin develop

# Release (fast)
maturin develop --release
```

---

**Issue**: `Node index 15 out of bounds`

**Solution**: Metatron graph has 13 nodes (0-12)

```python
# Wrong
result = metatron_qso.run_quantum_walk(graph, [15], t_max=5.0)

# Correct
result = metatron_qso.run_quantum_walk(graph, [0], t_max=5.0)
```

---

## Examples and Notebooks

See:
- **Examples**: `metatron_qso_py/examples/`
- **Notebooks**: `metatron_qso_py/notebooks/`

Run examples:

```bash
python metatron_qso_py/examples/01_quantum_walk_basic.py
python metatron_qso_py/examples/02_qaoa_maxcut_basic.py
python metatron_qso_py/examples/03_vqe_ground_state.py
```

---

## Further Reading

- [Rust Core Documentation](../metatron-qso-rs/docs/RUST_CORE_GUIDE.md)
- [VQA Implementation Guide](../VQA_IMPLEMENTATION_GUIDE.md)
- [Architecture Overview](../metatron-qso-rs/docs/ARCHITECTURE.md)

---

## Support

For issues, questions, or contributions:
- **GitHub Issues**: [LashSesh/qdash/issues](https://github.com/LashSesh/qso/issues)
- **Documentation**: [Project Docs](https://github.com/LashSesh/qso/tree/main/docs)
