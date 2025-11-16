# Metatron QSO Core - Rust Developer Guide

Welcome to the Metatron Quantum State Operator (QSO) Core library! This guide will help you understand and use the Rust API effectively.

## Table of Contents

1. [What is Metatron QSO?](#what-is-metatron-qso)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Quick Start Examples](#quick-start-examples)
5. [API Reference](#api-reference)
6. [Features & Configuration](#features--configuration)
7. [Performance Considerations](#performance-considerations)
8. [Limitations & Future Work](#limitations--future-work)

---

## What is Metatron QSO?

The Metatron Quantum State Operator is a quantum computing framework built around **sacred geometry** — specifically, the **Metatron Cube**, a 13-dimensional graph structure that embeds all five Platonic solids.

### Why Metatron Geometry?

- **Rich Connectivity**: 13 nodes with 78 edges (highly connected graph)
- **Symmetry**: Contains all Platonic solids (tetrahedron, cube, octahedron, dodecahedron, icosahedron)
- **Topological Properties**: Code distance d ≥ 6 for error correction
- **Information Advantage**: Dynamic Tripolar Logic provides 58.5% more capacity than binary

### What Can You Do With It?

- **Quantum Walks**: Simulate continuous-time quantum walks on the Metatron graph
- **Variational Quantum Algorithms**: Run VQE (ground state finding), QAOA (combinatorial optimization), VQC (classification)
- **Quantum Machine Learning**: Graph kernels, quantum neural networks
- **Error Correction**: Symmetry-protected topological codes

---

## Installation

### As a Library Dependency

Add to your `Cargo.toml`:

```toml
[dependencies]
metatron-qso-rs = "0.1"
```

Or with specific features:

```toml
[dependencies]
metatron-qso-rs = { version = "0.1", features = ["walks", "vqa"] }
```

### Available Features

- **`walks`** (default) - Quantum walk algorithms
- **`vqa`** (default) - Variational Quantum Algorithms (VQE, QAOA, VQC)
- **`dtl`** (default) - Dynamic Tripolar Logic
- **`codes`** - Topological error correction codes
- **`advanced`** - Advanced algorithms (Grover search, Boson sampling)

### Minimal Installation

For a minimal build (core only):

```toml
[dependencies]
metatron-qso-rs = { version = "0.1", default-features = false }
```

---

## Core Concepts

### 1. Metatron Graph

The foundation of everything is the **Metatron Cube** graph:

```rust
use metatron_qso::MetatronGraph;

let graph = MetatronGraph::new();
println!("Nodes: {}", graph.nodes().len());  // 13
println!("Edges: {}", graph.edges().len());  // 78
```

**Node Structure:**
- **Node 0**: Central node
- **Nodes 1-6**: Hexagon layer (connected to center)
- **Nodes 7-12**: Cube layer (outer vertices)

### 2. Quantum States

Quantum states are represented as complex vectors in a 13-dimensional Hilbert space:

```rust
use metatron_qso::QuantumState;

// Basis state |0⟩ (all amplitude on node 0)
let state = QuantumState::basis_state(0);

// Uniform superposition |+⟩ = (|0⟩ + |1⟩ + ... + |12⟩) / √13
let uniform = QuantumState::uniform_superposition();

// Custom state from amplitudes
use num_complex::Complex64;
let amplitudes = vec![Complex64::new(1.0, 0.0); 13];
let custom = QuantumState::from_amplitudes(amplitudes)?;
```

### 3. Hamiltonians

The system's dynamics are governed by the **graph Hamiltonian** H = -L (negative Laplacian):

```rust
use metatron_qso::MetatronHamiltonian;

let graph = MetatronGraph::new();
let hamiltonian = graph.hamiltonian();

// Get eigenvalues and eigenvectors
let spectrum = hamiltonian.spectrum();
println!("Ground state energy: {:.6}", spectrum.eigenvalues[0]);

// Get ground state
let ground_state = hamiltonian.ground_state();
```

### 4. Quantum Walks

Continuous-time quantum walks evolve states via U(t) = exp(-iHt):

```rust
use metatron_qso::prelude::*;

let graph = MetatronGraph::new();
let qw = ContinuousTimeQuantumWalk::new(graph);

let initial = QuantumState::basis_state(0);
let evolved = qw.evolve(&initial, 1.0)?;  // Evolve for time t=1.0

let probs = evolved.probabilities();
println!("Probability at node 0: {:.4}", probs[0]);
```

---

## Quick Start Examples

### Example 1: Basic Quantum Walk

```rust
use metatron_qso::prelude::*;

fn main() -> Result<(), String> {
    // Create graph
    let graph = MetatronGraph::new();

    // Initialize state on central node
    let initial_state = QuantumState::basis_state(0);

    // Run quantum walk
    let qw = ContinuousTimeQuantumWalk::new(graph);
    let evolved = qw.evolve(&initial_state, 1.0)?;

    // Analyze probabilities
    let probs = evolved.probabilities();
    for (i, &p) in probs.iter().enumerate() {
        println!("Node {}: {:.6}", i, p);
    }

    Ok(())
}
```

**Run:**
```bash
cargo run --example quantum_walk_basic
```

### Example 2: VQE Ground State Calculation

```rust
use metatron_qso::prelude::*;

fn main() -> Result<(), String> {
    let graph = MetatronGraph::new();
    let hamiltonian = graph.hamiltonian();

    // Configure VQE
    let vqe = VQEBuilder::new()
        .hamiltonian(hamiltonian.matrix().clone())
        .ansatz_type(AnsatzType::HardwareEfficient)
        .depth(2)
        .optimizer(OptimizerType::ADAM)
        .max_iterations(500)
        .build()?;

    // Run optimization
    let result = vqe.run()?;

    println!("Ground state energy: {:.6}", result.ground_energy);
    println!("Converged in {} iterations", result.iterations);

    Ok(())
}
```

### Example 3: QAOA for MaxCut

```rust
use metatron_qso::prelude::*;

fn main() -> Result<(), String> {
    let graph = MetatronGraph::new();
    let hamiltonian = graph.hamiltonian();

    // Configure QAOA
    let qaoa = QAOABuilder::new()
        .hamiltonian(hamiltonian.matrix().clone())
        .depth(3)  // p=3 layers
        .max_iterations(200)
        .build()?;

    // Run optimization
    let result = qaoa.run()?;

    println!("Best energy: {:.6}", result.best_energy);
    println!("Parameters: {:?}", result.best_params);

    Ok(())
}
```

**Run:**
```bash
cargo run --example qaoa_maxcut_basic --release
```

### Example 4: Dynamic Tripolar Logic (DTL)

```rust
use metatron_qso::prelude::*;

fn main() -> Result<(), String> {
    // Create DTL resonator network
    let network = DTLResonatorNetwork::new(13);

    // Initialize with random phases
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let phases: Vec<f64> = (0..13).map(|_| rng.gen_range(0.0..2.0*std::f64::consts::PI)).collect();

    // Evolve network
    let final_phases = network.evolve(&phases, 10.0, 0.01)?;

    // Compute synchronization order parameter
    let order = network.order_parameter(&final_phases);
    println!("Synchronization: {:.4}", order);

    Ok(())
}
```

---

## API Reference

### Core Types

#### `MetatronGraph`

```rust
impl MetatronGraph {
    pub fn new() -> Self;
    pub fn nodes(&self) -> &[Node];
    pub fn edges(&self) -> &[Edge];
    pub fn adjacency_matrix(&self) -> DMatrix<f64>;
    pub fn laplacian_matrix(&self) -> DMatrix<f64>;
    pub fn hamiltonian(&self) -> MetatronHamiltonian;
}
```

#### `QuantumState`

```rust
impl QuantumState {
    pub fn basis_state(node: usize) -> Self;
    pub fn uniform_superposition() -> Self;
    pub fn from_amplitudes(amplitudes: Vec<Complex64>) -> Result<Self, String>;
    pub fn probabilities(&self) -> Vec<f64>;
    pub fn amplitudes(&self) -> &SVector<Complex64, METATRON_DIMENSION>;
    pub fn is_normalized(&self) -> bool;
}
```

#### `MetatronHamiltonian`

```rust
impl MetatronHamiltonian {
    pub fn new(graph: &MetatronGraph) -> Self;
    pub fn matrix(&self) -> &SMatrix<f64, METATRON_DIMENSION, METATRON_DIMENSION>;
    pub fn spectrum(&self) -> &SpectrumInfo;
    pub fn ground_state(&self) -> QuantumState;
    pub fn eigenstate(&self, index: usize) -> QuantumState;
}
```

### Quantum Walks (feature: `walks`)

#### `ContinuousTimeQuantumWalk`

```rust
impl ContinuousTimeQuantumWalk {
    pub fn new(graph: MetatronGraph) -> Self;
    pub fn evolve(&self, initial: &QuantumState, time: f64) -> Result<QuantumState, String>;
    pub fn propagator(&self, time: f64) -> DMatrix<Complex64>;
}
```

#### `KrylovEvolution`

```rust
impl KrylovEvolution {
    pub fn new(hamiltonian: MetatronHamiltonian, krylov_dim: usize) -> Self;
    pub fn evolve(&self, initial: &QuantumState, time: f64) -> Result<QuantumState, String>;
}
```

### VQA (feature: `vqa`)

#### `VQE` (Variational Quantum Eigensolver)

```rust
impl VQEBuilder {
    pub fn new() -> Self;
    pub fn hamiltonian(self, h: DMatrix<f64>) -> Self;
    pub fn ansatz_type(self, ansatz: AnsatzType) -> Self;
    pub fn depth(self, d: usize) -> Self;
    pub fn optimizer(self, opt: OptimizerType) -> Self;
    pub fn max_iterations(self, n: usize) -> Self;
    pub fn tolerance(self, tol: f64) -> Self;
    pub fn build(self) -> Result<VQE, String>;
}

impl VQE {
    pub fn run(&self) -> Result<VQEResult, String>;
}
```

#### `QAOA` (Quantum Approximate Optimization Algorithm)

```rust
impl QAOABuilder {
    pub fn new() -> Self;
    pub fn hamiltonian(self, h: DMatrix<f64>) -> Self;
    pub fn depth(self, p: usize) -> Self;
    pub fn max_iterations(self, n: usize) -> Self;
    pub fn tolerance(self, tol: f64) -> Self;
    pub fn build(self) -> Result<QAOA, String>;
}

impl QAOA {
    pub fn run(&self) -> Result<QAOAResult, String>;
}
```

#### `VQC` (Variational Quantum Classifier)

```rust
impl VQCBuilder {
    pub fn new() -> Self;
    pub fn num_classes(self, n: usize) -> Self;
    pub fn depth(self, d: usize) -> Self;
    pub fn max_epochs(self, n: usize) -> Self;
    pub fn learning_rate(self, lr: f64) -> Self;
    pub fn build(self) -> Result<VQC, String>;
}

impl VQC {
    pub fn train(&mut self, data: &[(Vec<f64>, usize)]) -> Result<(), String>;
    pub fn predict(&self, input: &[f64]) -> Result<usize, String>;
}
```

---

## Features & Configuration

### Feature Flags

Control which modules are compiled:

```toml
# Default features (walks, vqa, dtl)
metatron-qso-rs = "0.1"

# Minimal (core only)
metatron-qso-rs = { version = "0.1", default-features = false }

# Custom feature set
metatron-qso-rs = { version = "0.1", default-features = false, features = ["walks", "codes"] }

# Everything
metatron-qso-rs = { version = "0.1", features = ["walks", "vqa", "dtl", "codes", "advanced"] }
```

### Conditional Compilation

Your code can adapt to available features:

```rust
#[cfg(feature = "walks")]
use metatron_qso::quantum_walk::ContinuousTimeQuantumWalk;

#[cfg(feature = "vqa")]
use metatron_qso::vqa::VQE;

fn my_function() {
    #[cfg(feature = "walks")]
    {
        let qw = ContinuousTimeQuantumWalk::new(graph);
        // ... walk-specific code
    }

    #[cfg(not(feature = "walks"))]
    {
        println!("Quantum walks not available (enable 'walks' feature)");
    }
}
```

---

## Performance Considerations

### Compilation Flags

**Always use `--release` for performance-critical code:**

```bash
cargo build --release
cargo run --release --example qaoa_maxcut_basic
```

Release builds are **10-100× faster** than debug builds due to:
- LLVM optimizations
- Inlining
- SIMD vectorization
- Loop unrolling

### Parallelism

The library uses `rayon` for parallel operations where applicable:

```rust
use rayon::prelude::*;

// Parallel computation over multiple states
let results: Vec<_> = states.par_iter()
    .map(|state| qw.evolve(state, 1.0))
    .collect();
```

### Memory Usage

**13-dimensional complex vectors:**
- QuantumState: ~208 bytes (13 × 16 bytes)
- Hamiltonian matrix: ~2.7 KB (13² × 16 bytes)

**For large batches**, pre-allocate:

```rust
let mut states = Vec::with_capacity(1000);
for i in 0..1000 {
    states.push(QuantumState::basis_state(i % 13));
}
```

### Numerical Precision

The library uses `f64` (64-bit floating point) for all calculations:
- Typical error: ~10⁻¹⁰ to 10⁻¹⁴
- Use `approx` crate for fuzzy comparisons:

```rust
use approx::assert_relative_eq;

let prob_sum: f64 = state.probabilities().iter().sum();
assert_relative_eq!(prob_sum, 1.0, epsilon = 1e-10);
```

---

## Limitations & Future Work

### Current Limitations

1. **Fixed Dimension**: Currently hardcoded to 13 nodes (Metatron Cube)
   - Future: Generic graph support

2. **No GPU Acceleration**: All computations run on CPU
   - Future: CUDA/ROCm backends for large-scale simulations

3. **Classical Simulation**: Not connected to real quantum hardware
   - Future: Qiskit/Braket/Cirq backends

4. **Limited Ansätze**: Only 3 ansatz types for VQA
   - Future: Custom ansatz API

5. **Single-threaded VQA**: Optimizers don't use parallelism
   - Future: Parallel parameter search

### Upcoming Features (v0.2+)

- **Noise Models**: Decoherence, gate errors
- **Tensor Network Methods**: MPS, PEPS for larger systems
- **Quantum Error Correction**: Full stabilizer formalism
- **Hardware Backends**: IBM, AWS, Google integration
- **Visualization**: State tomography, Bloch spheres

### Contributing

We welcome contributions! Areas of interest:
- GPU kernels for matrix operations
- Additional VQA optimizers (L-BFGS, CMA-ES)
- Quantum circuit transpilation
- Documentation improvements

See: [CONTRIBUTING.md](../CONTRIBUTING.md) (if available)

---

## Getting Help

- **Documentation**: Run `cargo doc --open` for API docs
- **Examples**: See `examples/` directory
- **Issues**: [GitHub Issues](https://github.com/LashSesh/qdash/issues)
- **Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Example Workflows

### Workflow 1: Quantum Walk Study

```rust
use metatron_qso::prelude::*;

fn study_mixing_time() -> Result<(), String> {
    let graph = MetatronGraph::new();
    let qw = ContinuousTimeQuantumWalk::new(graph);
    let initial = QuantumState::basis_state(0);

    // Sample times
    for t in (0..100).map(|i| i as f64 * 0.1) {
        let state = qw.evolve(&initial, t)?;
        let probs = state.probabilities();

        // Compute entropy (measure of mixing)
        let entropy: f64 = probs.iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| -p * p.log2())
            .sum();

        println!("t={:.2}, H={:.4}", t, entropy);
    }

    Ok(())
}
```

### Workflow 2: VQE Parameter Scan

```rust
use metatron_qso::prelude::*;

fn scan_vqe_depths() -> Result<(), String> {
    let graph = MetatronGraph::new();
    let hamiltonian = graph.hamiltonian();

    for depth in 1..=5 {
        let vqe = VQEBuilder::new()
            .hamiltonian(hamiltonian.matrix().clone())
            .ansatz_type(AnsatzType::HardwareEfficient)
            .depth(depth)
            .max_iterations(1000)
            .build()?;

        let result = vqe.run()?;

        println!("Depth {}: E={:.6}, iters={}",
                 depth, result.ground_energy, result.iterations);
    }

    Ok(())
}
```

---

## Cheat Sheet

```rust
// Import prelude
use metatron_qso::prelude::*;

// Create graph
let graph = MetatronGraph::new();

// Quantum state
let state = QuantumState::basis_state(0);
let uniform = QuantumState::uniform_superposition();

// Hamiltonian
let H = graph.hamiltonian();
let E0 = H.ground_state();

// Quantum walk
let qw = ContinuousTimeQuantumWalk::new(graph);
let evolved = qw.evolve(&state, 1.0)?;

// VQE
let vqe = VQEBuilder::new()
    .hamiltonian(H.matrix().clone())
    .depth(2)
    .build()?;
let result = vqe.run()?;

// QAOA
let qaoa = QAOABuilder::new()
    .hamiltonian(H.matrix().clone())
    .depth(3)
    .build()?;
let result = qaoa.run()?;
```

---

**Happy Quantum Coding!** 🌌

*Last updated: 2025-11-16*
