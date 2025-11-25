# Q⊗DASH - Metatron Quantum State Operator Framework

[![Rust](https://img.shields.io/badge/rust-1.85.0-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Workspace](https://img.shields.io/badge/workspace-24_crates-blue.svg)]()
[![Quantum Benchmarks](https://github.com/LashSesh/qso/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/LashSesh/qso/actions/workflows/benchmarks.yml)

**Enterprise-grade Quantum Computing Framework with 13-dimensional Metatron Geometry**

Q⊗DASH (MetatronQSO - Quantum State Operator) is a state-of-the-art quantum computing framework in pure Rust, based on the Sacred Geometry of Metatron's Cube. The system implements a complete stack of quantum algorithms, variational methods, dynamic tripolar logic, and automatic calibration through the Seraphic Calibration Shell.

## 🌟 Highlights

- **13-dimensional Metatron Cube** - Complete geometric quantum structure with 78 edges
- **Variational Quantum Algorithms** - VQE, QAOA, VQC with 3 ansatz types
- **Dynamic Tripolar Logic (DTL)** - 58.5% information advantage over binary systems
- **Seraphic Calibration Shell (SCS)** - Automatic hyperparameter optimization
- **DioniceOS Integration** - 4D-funnel system for 4D-5D coupling
- **Python SDK** - High-performance bindings via PyO3
- **Backend Abstraction** - Unified interface for Local/IBM/Cloud backends
- **Telemetry & Dashboard** - Real-time monitoring with REST API
- **24 Rust Crates** - Modular workspace architecture

## 📦 Workspace Overview

The project is organized as a Cargo workspace with 24 crates:

### Main Components (6 Crates)

| Crate | Description | Type |
|-------|-------------|------|
| **metatron-qso-rs** | Core Quantum Computing Library | lib + 8 bins |
| **metatron_qso_py** | Python SDK (PyO3 Bindings) | cdylib |
| **metatron_backend** | Backend Abstraction (Local/IBM) | lib |
| **metatron_dionice_bridge** | DioniceOS 4D-5D Integration | lib |
| **metatron_triton** | TRITON Spiral Search Optimizer | lib |
| **metatron_telemetry** | HTTP Telemetry Server | bin |

### DioniceOS Integration (18 Crates)

- **apollyon_5d/** (3 Crates) - 5D dynamic system framework
  - `core` - Dynamics, coupling, ensemble, stability
  - `bridge` - Integration layer
  - `metatron` - Geometric cognition engine

- **infinity-ledger/** (13 Crates) - MEF Pipeline System
  - `mef-core` - Core MEF pipeline
  - `mef-ledger` - Hash-chained ledger
  - `mef-memory` - Vector memory with adaptive routing
  - `mef-router` - S7 routing system
  - `mef-spiral`, `mef-storage`, `mef-hdag`, `mef-topology`, `mef-coupling`, `mef-schemas`
  - `mef-solvecoagula` - Double-kick operators
  - Additional: acquisition, domains, knowledge, api, audit, cli, benchmarks, tic, vector-db

- **apollyon-mef-bridge/** - APOLLYON-5D ⟷ Infinity-Ledger Bridge
  - 4D-funnel system (8 modules)
  - 4 bidirectional adapters
  - Unified cognitive engine

- **overlay/** - Unified 5D Cube Overlay

## 🚀 Quick Start

### Prerequisites

**For Windows users**: Before building, you need to install additional tools:
- **CMake** (required for cryptographic dependencies)
- **NASM** (recommended for performance)
- **Visual Studio Build Tools 2022** with C++ workload

📖 **See [docs/WINDOWS_SETUP.md](docs/WINDOWS_SETUP.md) for detailed Windows installation instructions**

**For Linux/macOS users**:
- Rust 1.85.0+
- Standard build tools (gcc/clang)
- CMake (usually available via package manager: `apt install cmake` / `brew install cmake`)

### Installation

```bash
# Clone repository
git clone https://github.com/LashSesh/qso.git
cd qso

# Build core library
cargo build --release -p metatron-qso-rs

# Run all tests
cargo test --workspace

# Run benchmarks
cargo run --release --bin quantum_walk_bench
```

### First Quantum Program

```rust
use metatron_qso_rs::prelude::*;

fn main() -> Result<()> {
    // Initialize Metatron QSO
    let qso = QSO::new(QSOParameters::default())?;

    // Quantum walk from center node
    let initial = QuantumState::basis_state(0); // Node 0 = center
    let evolved = qso.evolve_state(&initial, 1.0)?;

    // Probability distribution
    for (node, prob) in evolved.probabilities().iter().enumerate() {
        println!("Node {}: {:.4}", node, prob);
    }

    Ok(())
}
```

### Python SDK Usage

```bash
# Install Python SDK
cd metatron_qso_py
pip install maturin
maturin develop --release
```

```python
import metatron_qso

# Create Metatron graph
graph = metatron_qso.MetatronGraph()

# Run quantum walk
result = metatron_qso.run_quantum_walk(
    graph=graph,
    source_nodes=[0],
    t_max=5.0,
    dt=0.1
)

# QAOA for MaxCut
qaoa_result = metatron_qso.solve_maxcut_qaoa(
    graph=graph,
    depth=3,
    max_iters=100
)

# VQE ground state
vqe_result = metatron_qso.run_vqe(
    graph=graph,
    depth=2,
    ansatz_type="hardware_efficient"
)
```

## 🧬 Core Features - metatron-qso-rs

### Metatron Geometry

**13-dimensional Cube** based on Sacred Geometry:
- **1 center node** (Node 0)
- **6 hexagon nodes** (Nodes 1-6)
- **6 cube nodes** (Nodes 7-12)
- **78 edges** with complete connectivity
- Embedding of all 5 Platonic solids
- Symmetry group G_M for error-resistant operations

**Graph Properties**:
- Average degree: 12
- Algebraic connectivity: λ₁ > 0 (high)
- Code distance: d ≥ 6 (topological error correction)

### Quantum Algorithms

#### Variational Quantum Algorithms (VQA)

**VQE (Variational Quantum Eigensolver)**:
```rust
use metatron_qso_rs::vqa::{VQE, AnsatzType};

let vqe = VQE::builder()
    .hamiltonian(qso.hamiltonian().clone())
    .ansatz_type(AnsatzType::HardwareEfficient)
    .depth(2)
    .optimizer_name("ADAM")
    .max_iterations(1000)
    .build()?;

let result = vqe.run()?;
println!("Ground Energy: {:.10}", result.ground_energy);
```

**3 Ansatz Types**:
- `HardwareEfficient` - Optimized for hardware implementation
- `EfficientSU2` - SU(2)-based
- `MetatronAnsatz` - Specialized for Metatron geometry

**QAOA (Quantum Approximate Optimization Algorithm)**:
```rust
use metatron_qso_rs::vqa::{QAOA, MaxCutProblem};

let graph = MetatronGraph::new();
let problem = MaxCutProblem::from_graph(&graph);
let qaoa = QAOA::new(problem.hamiltonian(), 3);
let result = qaoa.run()?;

println!("Approximation ratio: {:.4}", result.approximation_ratio);
```

**VQC (Variational Quantum Classifier)**:
- Binary and multi-class classification
- Parameter shift rule for gradients
- Training/test split support

**Optimizers**: COBYLA, ADAM, L-BFGS-B

#### Quantum Walks

**4 Implementations**:
- **CTQW (Continuous-Time Quantum Walk)** - Spectral propagator method
- **Krylov Methods** - Lanczos algorithm for large systems
- **Scattering Analysis** - Density of states, scattering channels
- **Benchmark Suite** - Hitting time, mixing time, fidelity

```rust
use metatron_qso_rs::quantum_walk::*;

let walk = ContinuousQuantumWalk::new(graph.adjacency_matrix());
let result = walk.evolve(initial_state, time)?;
```

#### Advanced Algorithms

- **Grover Search** - Metatron-specific variant
- **Boson Sampling** - Platonic-solid interference
- **Quantum Machine Learning** - Graph-structured ML

### Dynamic Tripolar Logic (DTL)

**58.5% information advantage** over binary systems:

**3 States**:
- **L+** (active) - High activation
- **L-** (inactive) - Low activation
- **Ld** (dynamic/undetermined) - Superposition

**Features**:
- Kuramoto synchronization networks
- Resonator dynamics
- Tripolar gate operations
- Network coupling

**Information Capacity** (13 nodes):
- Binary: 13.0 bit
- Tripolar: 20.6 bit (+58.5%)
- With phase: 46.6 bit (+258%)

### Symmetry & Error Correction

- **G_M Symmetry Group** - Metatron-specific symmetries
- **Topological Codes** - Code distance d ≥ 6
- **Error-resistant Operations** - Symmetry-protected gates

## 🐍 Python SDK - metatron_qso_py

**High-Performance Python Bindings via PyO3**

### Installation

```bash
cd metatron_qso_py
pip install maturin
maturin develop --release
```

### Features

- ✅ **Python-idiomatic API** - dict returns, list parameters
- ✅ **Rust performance** - Zero-cost bindings
- ✅ **Jupyter-ready** - Interactive notebooks
- ✅ **Type safety** - Clear error handling

### Examples

```bash
# Run examples
python metatron_qso_py/examples/01_quantum_walk_basic.py
python metatron_qso_py/examples/02_qaoa_maxcut_basic.py
python metatron_qso_py/examples/03_vqe_ground_state.py

# Jupyter notebook
jupyter notebook metatron_qso_py/notebooks/QuantumWalk_Intro.ipynb
```

### Auto-Tuning Integration

```python
import metatron_qso

# QAOA with automatic calibration
result, proposal = metatron_qso.solve_maxcut_qaoa_with_tuning(
    graph=graph,
    depth=3,
    max_iters=100,
    auto_calibrate=True
)

if proposal.por_accepted:
    print(f"SCS suggests: depth={proposal.config.ansatz_depth}")
```

## 🔧 Seraphic Calibration Shell (SCS)

**Automatic Hyperparameter Optimization for Quantum Algorithms**

The SCS is a meta-algorithm for automatic calibration of quantum algorithms. It uses field-theoretic feedback and fixpoint dynamics.

### Core Concepts

**Performance Triplet Φ(c) = (ψ, ρ, ω)**:
- **ψ (Quality)** - Algorithm-specific quality
- **ρ (Stability)** - Robustness across multiple runs
- **ω (Efficiency)** - Computational efficiency

**Mandorla Field M(t)**:
- 16-dimensional resonance field
- Historical performance patterns
- Guides configuration changes

**Double-Kick Operator T = Φ_V ∘ Φ_U**:
- Update-Kick Φ_U: Improves quality
- Stabilization-Kick Φ_V: Optimizes stability
- Converges to fixpoint attractors

**Proof-of-Resonance (PoR)**:
- Acceptance criterion for new configurations
- Guarantees monotonic quality improvement
- Validates field resonance

**CRI (Calibration Regime Initialization)**:
- Detects stagnation in local optimum
- Automatically switches to new regime
- Enables global exploration

### CLI Usage

```bash
# Initialize SCS
python -m scs.cli init

# Execute 5 calibration steps
python -m scs.cli step -n 5

# Show status
python -m scs.cli status

# Export best configuration
python -m scs.cli export -o best_config.json
```

### Python API

```python
from scs import AutoTuner

tuner = AutoTuner(benchmark_dir="benchmarks", enabled=True)
tuner.initialize()

for iteration in range(10):
    result = run_algorithm()
    metrics = {"psi": 0.85, "rho": 0.80, "omega": 0.72}

    tuner.ingest_benchmark("qaoa", config, metrics, result)
    proposal = tuner.propose_new_config()

    if proposal.por_accepted:
        config = proposal.config
```

## 🌐 DioniceOS Integration

**4D-5D Coupling System for Cognitive Quantum Processing**

### Architecture

```
4D-Funnel (Gabriel) ←→ APOLLYON-5D ←→ Infinity-Ledger (MEF)
                               ↓
                     Metatron QSO (via Bridge)
```

### 4D-Funnel System

**Components**:
- **Funnel Graph** - Directed graph with Hebbian learning
- **Hyperbion Layer** - Morphodynamic 4D-5D coupling
- **HDAG Field** - 5D resonance lattice (hyperdimensional acyclic)
- **Policies** - Explore, exploit, homeostasis

**Properties**:
- Deterministic: Same inputs → identical outputs
- Proof-carrying: Cryptographic verification
- Coordinate mapping: SCS metrics → 4D state space

### 5D Coordinate Space

Unified 5D Space **(x, y, z, ψ, ω)**:
- **x, y, z** - Spatial coordinates
- **ψ** (psi) - Semantic weight / resonance
- **ω** (omega) - Temporal phase / oscillation

### Bidirectional Adapters

**4 Adapters** for seamless integration:
- **State Adapter** - 5D ⟷ Spiral
- **Spectral Adapter** - Features ⟷ Signature
- **Metatron Adapter** - Cube-13 ⟷ S7
- **Resonance Adapter** - Field ⟷ PoR

### Integration Flow

1. **SCS State** (ψ, ρ, ω, algorithm) → QDashCalibrationState
2. **Bridge Mapping** → 4D state space
3. **4D-Funnel Coupling Tick**:
   - Lift 4D → 5D
   - Hyperbion absorption
   - HDAG relaxation & gradient
   - Project 5D → 4D
   - Funnel advection
4. **Calibration Suggestion** generation

### Test Coverage

- APOLLYON-5D: 109 tests
- Infinity-Ledger: Complete MEF tests
- Bridge: 84 tests (41 for 4D-funnel)

## 🔌 Backend Abstraction - metatron_backend

**Unified Interface for Multiple Quantum Backends**

### Supported Backends

- **Local Simulator** (default) - Pure Rust simulation
- **IBM Quantum** (feature-gated) - IBM Cloud integration
- Extensible for AWS Braket, IonQ, Rigetti

### Usage

```rust
use metatron_backend::*;

// Backend registry
let registry = BackendRegistry::new();

// Local backend
let local = registry.get_backend("local")?;

// Execute circuit
let circuit = Circuit::new(num_qubits);
circuit.add_gate(Gate::H(0));
circuit.add_gate(Gate::CNOT(0, 1));

let result = local.execute(&circuit)?;
println!("Measurements: {:?}", result.measurements);
```

### Features

- **Provider Abstraction** - Unified API for all backends
- **Circuit Representation** - Backend-agnostic format
- **Registry Pattern** - Factory for backend instances
- **Feature Gates** - Optional IBM/Cloud integration

## 📊 Telemetry & Dashboard - metatron_telemetry

**Real-time Monitoring with HTTP REST API**

### Features

- **REST API** - Complete HTTP endpoints
- **Real-time Metrics** - Live performance tracking
- **Historical Data** - Persistent storage
- **Web Dashboard** - Browser-based UI
- **Demo Mode** - Sample data for testing

### Starting the Server

```bash
cargo run --release --bin metatron_telemetry
```

```
🚀 Telemetry server running on http://localhost:3000

Endpoints:
  GET  /health              - Health check
  GET  /api/metrics         - Current metrics
  POST /api/metrics         - Add metric
  GET  /api/metrics/history - Historical data
  GET  /dashboard           - Web dashboard
```

### API Usage

```bash
# Health check
curl http://localhost:3000/health

# Get metrics
curl http://localhost:3000/api/metrics

# Send metric
curl -X POST http://localhost:3000/api/metrics \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "vqe", "energy": -12.9997, "iterations": 150}'
```

### Web Dashboard

```bash
# Open dashboard in browser
open http://localhost:3000/dashboard
```

Features:
- Live metric visualization
- Algorithm performance charts
- Historical trend analysis
- Export to JSON/CSV

## 🔍 TRITON - Spiral Search Optimizer

**Evolutionary Spiral-Search for SCS Calibration**

### Concept

TRITON uses golden-angle spirals for efficient hyperparameter exploration:

**SpectralSignature (ψ, ρ, ω)**:
- 3D quality metric
- Momentum-based search
- Adaptive step size

### Usage

```rust
use metatron_triton::*;

let search = TritonSearch::new(config_space);
let signature = SpectralSignature::new(0.85, 0.80, 0.72);

let proposal = search.evolve(current_config, signature)?;

if proposal.quality_improved() {
    apply_config(proposal.config);
}
```

### Features

- Golden-angle spirals (φ = 137.5°)
- Momentum-driven evolution
- Calibration proposals
- Integration with SCS

## 🧪 Testing & Benchmarking

### Running Tests

```bash
# All unit tests in workspace
cargo test --workspace

# Test specific crate
cargo test -p metatron-qso-rs
cargo test -p metatron_dionice_bridge

# DioniceOS tests
cargo test -p apollyon_5d            # 109 tests
cargo test -p apollyon-mef-bridge    # 84 tests (41 for 4D-funnel)
```

### Benchmark Suite

**8 Benchmark Binaries**:

```bash
# Core benchmarks
cargo run --release --bin quantum_walk_bench
cargo run --release --bin vqe_bench
cargo run --release --bin qaoa_bench
cargo run --release --bin vqc_bench

# Comparison benchmarks
cargo run --release --bin integration_bench
cargo run --release --bin cross_system_bench
cargo run --release --bin advanced_algorithms_bench
cargo run --release --bin benchmark_compare
```

### Performance Baselines

| Benchmark | Performance | Convergence |
|-----------|------------|-------------|
| Quantum Walk | 31,941 ops/sec | 100% |
| VQE (HardwareEfficient) | ~150 iters | E₀ = -12.9997 |
| QAOA (depth=3) | ~100 iters | ratio = 0.9974 |
| VQC (binary) | ~200 epochs | acc = 50-90% |

### CI/CD Integration

**GitHub Actions** with automatic baseline comparison:
- Parallel test execution
- Performance regression detection
- Baseline tracking in `metatron-qso-rs/ci/`
- Daily performance metrics

## 📖 Documentation

### Overview Documents (Root)

- **[PRODUCT_OVERVIEW.md](PRODUCT_OVERVIEW.md)** - Architecture overview
- **[DIONICEOS_INTEGRATION.md](DIONICEOS_INTEGRATION.md)** - DioniceOS integration guide
- **[VQA_IMPLEMENTATION_GUIDE.md](VQA_IMPLEMENTATION_GUIDE.md)** - VQA algorithms guide
- **[QUANTENINFORMATIONSVERARBEITUNG_DOKUMENTATION.md](QUANTENINFORMATIONSVERARBEITUNG_DOKUMENTATION.md)** - Quantum info (DE)
- **[BENCHMARK_SUITE_DOCUMENTATION.md](BENCHMARK_SUITE_DOCUMENTATION.md)** - Benchmark system
- **[CHANGELOG.md](CHANGELOG.md)** - Version history
- **[RELEASE_PLAN.md](RELEASE_PLAN.md)** - Packaging strategy
- **[DEV_SETUP.md](DEV_SETUP.md)** - Development setup

### SCS Documentation (docs/)

- **[docs/SCS_CORE_DESIGN.md](docs/SCS_CORE_DESIGN.md)** - Architecture & data flow
- **[docs/SCS_BENCHMARK_SCHEMA.md](docs/SCS_BENCHMARK_SCHEMA.md)** - JSON schema specification
- **[docs/SCS_USAGE_GUIDE.md](docs/SCS_USAGE_GUIDE.md)** - Workflows & best practices
- **[docs/seraphic_calibration_shell.md](docs/seraphic_calibration_shell.md)** - Overview

### System Documentation (docs/)

- **[docs/backend_system.md](docs/backend_system.md)** - Backend architecture
- **[docs/telemetry_and_dashboard.md](docs/telemetry_and_dashboard.md)** - Telemetry system
- **[docs/pyo3_integration.md](docs/pyo3_integration.md)** - Python bindings
- **[docs/PYTHON_SDK_GUIDE.md](docs/PYTHON_SDK_GUIDE.md)** - Python API reference
- **[docs/CI_PIPELINE_OVERVIEW.md](docs/CI_PIPELINE_OVERVIEW.md)** - CI/CD pipeline

### DioniceOS Documentation

- **[docs/dioniceos/README.md](docs/dioniceos/README.md)** - Complete DioniceOS guide
- **[docs/dioniceos/QUICK_START.md](docs/dioniceos/QUICK_START.md)** - Quick start

### Core Library Documentation (metatron-qso-rs/docs/)

- **[metatron-qso-rs/docs/ARCHITECTURE.md](metatron-qso-rs/docs/ARCHITECTURE.md)** - Core architecture
- **[metatron-qso-rs/docs/RUST_CORE_GUIDE.md](metatron-qso-rs/docs/RUST_CORE_GUIDE.md)** - Developer guide
- **quantum_walk_mixing.md**, **cross_system_vqe_scoring.md**, **vqe_tuning.md**, **vqc_overview.md**

### Setup Guides (docs/)

- **[docs/WINDOWS_SETUP.md](docs/WINDOWS_SETUP.md)** - **Windows 10/11 setup guide** (EN) 🔥
- **[docs/SCHNELLANLEITUNG.md](docs/SCHNELLANLEITUNG.md)** - Quick guide (DE)

### API Documentation

```bash
# Generate and open rustdoc
cargo doc --open --workspace
```

## 🛠️ Development

### Prerequisites

- **Rust** 1.85.0+ (Edition 2021)
- **Cargo** with workspace support
- **Python** 3.8+ (for Python SDK)
- **Maturin** (for Python bindings)

### Building the Project

```bash
# Build entire workspace
cargo build --release --workspace

# Build individual crate
cargo build --release -p metatron-qso-rs
cargo build --release -p metatron_backend
cargo build --release -p metatron_telemetry

# Build Python SDK
cd metatron_qso_py
maturin develop --release
```

### Code Quality

```bash
# Formatting
cargo fmt --all

# Linting
cargo clippy --workspace -- -D warnings

# Python linting
cd scs
ruff check .
ruff format .
```

### Features

**metatron-qso-rs Features**:
- `walks` - Quantum walk algorithms
- `vqa` - Variational quantum algorithms
- `dtl` - Dynamic tripolar logic
- `codes` - Symmetry codes
- `advanced` - Advanced algorithms

**metatron_backend Features**:
- `local` (default) - Local simulator
- `ibm` - IBM Quantum integration
- `all-backends` - All backends

```bash
# Build with specific features
cargo build --release -p metatron-qso-rs --features "walks,vqa,dtl"
cargo build --release -p metatron_backend --features "ibm"
```

## 🎯 Roadmap

### ✅ Phase 1: Core Implementation (Completed)
- [x] Metatron geometry (13 nodes, 78 edges)
- [x] Quantum state & operator primitives
- [x] DTL system (4 modules)
- [x] Quantum walks (CTQW, Krylov, scattering)
- [x] Hamiltonian & spectral analysis

### ✅ Phase 2: Variational Algorithms (Completed)
- [x] VQE with 3 ansatz types
- [x] QAOA for combinatorial optimization
- [x] VQC for classification
- [x] Parameter shift rule gradients
- [x] 3 optimizers (COBYLA, ADAM, L-BFGS-B)

### ✅ Phase 3: Benchmarking & CI/CD (Completed)
- [x] 8 comprehensive benchmark suites
- [x] Automatic baseline comparisons
- [x] GitHub Actions integration
- [x] Performance regression detection

### ✅ Phase 4: Advanced Features (Completed)
- [x] Seraphic Calibration Shell (SCS)
- [x] DioniceOS 4D-5D integration
- [x] Backend abstraction layer
- [x] Telemetry & dashboard
- [x] TRITON spiral search
- [x] Python SDK (PyO3)
- [x] Grover search & boson sampling

### 🚧 Phase 5: Production Ready (In Progress)
- [ ] GPU acceleration (CUDA/ROCm)
- [ ] Advanced visualization
- [ ] IBM Quantum backend (complete)
- [ ] AWS Braket integration
- [ ] Advanced error correction
- [ ] Performance optimizations

### 🔮 Phase 6: Hardware Integration (Planned)
- [ ] IonQ/Rigetti support
- [ ] Photonic chip design
- [ ] Quantum annealer integration
- [ ] NISQ-device deployment

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Q⊗DASH Workspace                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐      ┌──────────────────┐                   │
│  │ metatron-qso-rs  │◄────►│ metatron_backend │                   │
│  │  • Quantum Core  │      │  • Local/IBM     │                   │
│  │  • VQA/QAOA/VQC  │      │  • Circuit API   │                   │
│  │  • Quantum Walks │      └──────────────────┘                   │
│  │  • DTL System    │                                             │
│  └────────┬─────────┘                                             │
│           │                                                        │
│           ▼                                                        │
│  ┌──────────────────┐      ┌──────────────────────────┐          │
│  │ metatron_qso_py  │      │ metatron_dionice_bridge  │          │
│  │  • PyO3 Bindings │      │  • 4D-Trichter System    │          │
│  │  • Python API    │◄────►│  • 4D-5D Coupling        │          │
│  │  • Auto-Tuning   │      │  • 4 Adapters            │          │
│  └──────────────────┘      └───────────┬──────────────┘          │
│                                        │                          │
│  ┌──────────────────┐                 ▼                          │
│  │ metatron_triton  │      ┌──────────────────────────┐          │
│  │  • Spiral Search │      │   DioniceOS (18 Crates)  │          │
│  │  • TRITON        │      ├──────────────────────────┤          │
│  └──────────────────┘      │ • apollyon_5d (3)        │          │
│                            │ • infinity-ledger (13)   │          │
│  ┌──────────────────┐      │ • apollyon-mef-bridge    │          │
│  │ metatron_telemetry│      │ • unified_5d_cube        │          │
│  │  • HTTP Server   │      └──────────────────────────┘          │
│  │  • REST API      │                                             │
│  │  • Dashboard     │                                             │
│  └──────────────────┘                                             │
│                                                                     │
│  ┌──────────────────────────────────────────┐                     │
│  │         SCS (Python 12 Modules)          │                     │
│  │  • Calibrator • Field • Operators • PoR  │                     │
│  │  • CRI • CLI • AutoTuner • Benchmark     │                     │
│  └──────────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 🔬 Scientific Background

### Information-theoretic Advantage

**Metatron System (13 nodes)**:
```
Binary:             13.0 bit (classical)
Tripolar:           20.6 bit (+58.5%)
Tripolar with phase: 46.6 bit (+258%)
```

### Quantum Algorithm Complexity

| Algorithm | Complexity | Speedup vs. Classical |
|-----------|------------|----------------------|
| Quantum Walk Search | O(√N) | ~3.6× |
| VQE Ground State | O(poly(n)) | Exponential |
| QAOA MaxCut | O(p·M) | >0.75 approximation |
| Boson Sampling | #P-hard | Classically intractable |
| Grover Search | O(√N) | Quadratic |

### 4D-5D Coupling Theory

**5D State Space**: (x, y, z, ψ, ω)
- Semantic dimension: ψ (resonance/weight)
- Temporal dimension: ω (phase/oscillation)

**Coupling Operator**:
```
Lift:    4D → 5D  (Hyperbion absorption)
Relax:   5D → 5D  (HDAG gradient descent)
Project: 5D → 4D  (Funnel advection)
```

## 🤝 Contributing

Contributions are welcome! Please note:

1. **Fork** the repository
2. **Create feature branch** (`git checkout -b feature/amazing-feature`)
3. **Add tests** (`cargo test --workspace`)
4. **Check formatting** (`cargo fmt --all && cargo clippy --workspace`)
5. **Commit** (`git commit -m 'Add amazing feature'`)
6. **Push** to branch (`git push origin feature/amazing-feature`)
7. **Open pull request**

### Development Guidelines

- All new features need tests
- Documentation with rustdoc
- Update benchmark baselines when performance changes
- Add Python examples for new APIs

## 📝 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Sacred Geometry** - Metatron's Cube as fundamental structure
- **Quantum Computing** - VQE/QAOA/VQC research
- **Rust Community** - nalgebra, petgraph, rayon, pyo3
- **DioniceOS** - 4D-5D integration framework

## 📧 Contact & Support

- **GitHub Issues**: [https://github.com/LashSesh/qso/issues](https://github.com/LashSesh/qso/issues)
- **Documentation**: See [docs/](docs/) directory
- **Examples**: [metatron_qso_py/examples/](metatron_qso_py/examples/)

## 📈 Status & Metrics

- **Lines of Code**: ~8,222 Rust (Core) + ~17,200 Rust (Python bindings) + ~3,204 Python (SCS)
- **Test Coverage**: 109 tests (APOLLYON-5D) + 84 tests (Bridge) + inline tests
- **Workspace Crates**: 24 (6 main + 18 DioniceOS)
- **Benchmark Suites**: 8 executables with CI/CD integration
- **Documentation Files**: 30+ Markdown files

---

**Made with ❤️ in Rust** | **Powered by Quantum Geometry** | **© 2025 Sebastian Klemm (Aion-Chronos)**
