# APOLLYON-5D: Unified Geometric-Cognitive Mathematics Engine

**Version 1.0.0 - COMPLETE IMPLEMENTATION**

A unified framework combining deterministic 5-dimensional dynamical systems with geometric cognition through the Metatron Cube. This integration enables adaptive dynamical systems with resonance-based orchestration, spectral analysis, and symmetry-preserving integration.

## 🎯 Core Vision

**"Computation with quantum-like precision, orchestrated by geometric cognition."**

This is not two separate tools—it's a single, unified system where:
- The **5D Framework** provides the physics engine (numerical integration, stability analysis)
- **Metatron-R** provides the cognition layer (adaptive parameter control, resonance-based orchestration)
- The **Bridge** layer connects them through trait-based interfaces

## 🌌 5D Framework Overview

This framework implements **Operational 5D Mathematics** as specified in the foundational PDFs by Sebastian Klemm. The five dimensions represent:

### The Five Dimensions (D1-D5)

**D1-D3 (x, y, z):** Classical 3D space
- Euclidean coordinates for spatial representation
- Standard geometric transformations (translation, rotation)
- Implemented as components 1-3 of `State5D` vector σ

**D4 (ψ - Psi):** Semantic weighting / Resonance strength
- Represents semantic density and resonance coherence
- Computed via: ψ(K) = f(frequency, resonance, overlap)
- Governs information field intensity and coupling modulation
- Implemented as component 4 of `State5D` vector

**D5 (ω - Omega):** Temporal rhythmics / Phase signature
- Encodes temporal signature and phase relationships
- Enables Ouroboros feedback: S(t) = S(t-1) + α·f(∇ψ, ρ, ω)
- Provides evolutionary dynamics and self-structuring
- Implemented as component 5 of `State5D` vector

### Key 5D Principles

1. **Spiral Manifold**: Information encoded on 5D spiral S(θ) = (a·cos θ, a·sin θ, b·cos 2θ, b·sin 2θ, c·θ)
2. **Resonance-Based Interaction**: Proof-of-Resonance (∆ψ < ε) validates state coherence
3. **Metatron Geometry**: 13-node structure (1 center + 12 outer) with C6/D6 symmetries
4. **Ouroboros Feedback**: Self-structuring loop preserving resonance and coherence
5. **Structural Invariants**: Finite values enforced, symmetry preservation, resonance conservation

### Mathematical Foundation

The system evolves according to:
```
dσ/dt = F(σ) = αᵢσᵢ + Σⱼ τᵢⱼ(σᵢ, σⱼ, Cᵢⱼ) + fᵢ(t)
```

Where:
- **σ ∈ ℝ⁵**: 5D state vector
- **αᵢ**: Intrinsic rates per dimension
- **τᵢⱼ**: Coupling operators (Linear, Quadratic, Product, Sigmoid)
- **Cᵢⱼ**: Coupling strengths (modulated by resonance)
- **fᵢ(t)**: External forcing

See [`docs/5d-spec.md`](docs/5d-spec.md) for complete specification with PDF references.
See [`docs/5d-pdf-mapping.md`](docs/5d-pdf-mapping.md) for exact code-to-specification mapping.

## ✨ Complete Capabilities

### 🔬 Deterministic 5D Dynamical Systems
- **State Evolution**: High-precision numerical integration using Heun's method (RK2)
- **Coupling Types**: Four coupling mechanisms (Linear, Quadratic, Product, Sigmoid)
- **Stability Analysis**: Eigenvalue decomposition and fixed point classification
- **Domain Templates**: Pre-configured models (SIR epidemiology, Financial markets, Predator-prey)
- **Validation**: All reference tests pass with analytical comparison

### 🧠 Geometric Cognition Engine
- **Metatron Cube Geometry**: Canonical 13-node geometric structure with C6/D6 symmetries
- **QLogic Spectral Analysis**: Fourier-like transformation for pattern recognition
- **QDASH Decision Engine**: Adaptive decision-making with Mandorla resonance fields
- **Spectral Pipeline**: Entropy analysis, spectral centroids, oscillation detection
- **Gabriel Cell Lattices**: Coupled resonance cells for pattern propagation

### 🌉 Adaptive Integration Bridge
- **Dynamic Resonance**: Time-varying coupling modulation based on resonance fields
- **Cognitive Feedback Loop**: QLogic spectral analysis → QDASH parameter tuning
- **Geometric Constraints**: 5D states mapped to Metatron node positions
- **Symmetry Preservation**: C6 rotational and D6 reflection symmetry operations
- **Trajectory Observation**: Real-time monitoring with velocity and energy tracking

## 🏗️ Architecture

```
apollyon-5d/
├── core/                   # 5D Dynamical Systems Framework
│   ├── src/                # Core 5D implementation
│   │   ├── state.rs        # 5D state vectors
│   │   ├── coupling.rs     # Coupling matrices and types
│   │   ├── dynamics.rs     # Vector fields and Jacobians
│   │   ├── integration.rs  # Heun's method integration
│   │   ├── stability.rs    # Eigenvalue analysis
│   │   ├── projection.rs   # Dimension reduction
│   │   ├── template.rs     # Domain-specific models
│   │   ├── ensemble.rs     # Monte Carlo and parameter sweeps
│   │   └── ...
│   └── examples/           # 5D system examples
│
├── metatron/               # Metatron-R Geometric Cognition Engine
│   ├── src/
│   │   ├── geometry/       # Metatron Cube, nodes, edges, symmetries
│   │   ├── cognition/      # Agents, QLogic, QDASH, semantic fields
│   │   ├── fields/         # Resonance fields, Gabriel cells, tensors
│   │   ├── spectral/       # Spectral cognition and entropy analysis
│   │   └── ...
│   └── Cargo.toml
│
├── bridge/                 # Integration Layer
│   ├── src/
│   │   ├── resonance_field.rs       # Trait: ResonanceField
│   │   ├── adaptive_coupling.rs     # Resonance-modulated coupling
│   │   ├── geometric_forcing.rs     # 5D ↔ Metatron projection
│   │   ├── trajectory_observer.rs   # 5D → Metatron feedback
│   │   ├── spectral_analyzer.rs     # QLogic spectral analysis bridge
│   │   ├── parameter_tuner.rs       # QDASH adaptive parameter control
│   │   └── unified_system.rs        # CognitiveSimulator
│   ├── examples/
│   │   ├── adaptive_epidemic.rs     # Dynamic resonance (Phase 3)
│   │   ├── self_tuning_ecology.rs   # Cognitive feedback (Phase 4)
│   │   └── geometric_finance.rs     # Symmetry constraints (Phase 5)
│   └── Cargo.toml
│
└── Cargo.toml              # Workspace configuration
```

## 🚀 Quick Start

### Installation

Ensure you have Rust installed (version 1.70+):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Build the workspace:

```bash
cargo build --release
```

### Running Examples

```bash
# Phase 3: Dynamic resonance with time-varying coupling
cargo run --release --example adaptive_epidemic

# Phase 4: Cognitive feedback loop with QLogic + QDASH
cargo run --release --example self_tuning_ecology

# Phase 5: Geometric constraints with C6/D6 symmetry
cargo run --release --example geometric_finance
```

### Testing

Run all 109 tests:

```bash
cargo test --release
```

Test individual crates:

```bash
cargo test -p core_5d    # 39 tests
cargo test -p metatron   # 32 tests
cargo test -p bridge     # 38 tests
```

## 📚 Usage Examples

### Basic 5D System

```rust
use core_5d::*;

// Create coupling matrix
let mut coupling = CouplingMatrix::zero();
coupling.set(0, 1, 0.5, CouplingType::Linear);
coupling.set(1, 0, -0.3, CouplingType::Product);

// Create system parameters
let mut params = SystemParameters::zero();
params.intrinsic_rates[0] = -0.1;

// Create vector field and integrate
let vf = VectorField::new(coupling, params);
let time_config = integration::TimeConfig::new(0.01, 0.0, 10.0);
let integrator = Integrator::new(vf, time_config);

let initial = State5D::new(1.0, 0.5, 0.0, 0.0, 0.0);
let trajectory = integrator.integrate(initial);
```

### Adaptive Coupling with Resonance

```rust
use bridge::*;
use core_5d::*;

// Create base system
let template = Template::sir_model(0.3, 0.1, 0.01);
let coupling = template.coupling_matrix;
let params = template.parameters;

// Add resonance-based adaptation
let resonance = OscillatoryResonanceField::new(0.3, 0.5, 0.0);
let adaptive = AdaptiveCoupling::new(coupling.clone(), Box::new(resonance));

// Integrate with adaptive coupling
let vf = VectorField::new(coupling, params);
let time_config = integration::TimeConfig::new(0.1, 0.0, 50.0);
let integrator = Integrator::new(vf, time_config);
let observer = TrajectoryObserver::new(500);

let mut sim = CognitiveSimulator::with_adaptive_coupling(
    integrator, observer, adaptive
);

let initial = State5D::new(0.99, 0.01, 0.0, 0.0, 0.0);
let trajectory = sim.integrate_adaptive(initial);
```

### Cognitive Feedback Loop

```rust
use bridge::*;

// Create spectral analyzer and parameter tuner
let analyzer = SpectralAnalyzer::new();
let mut tuner = ParameterTuner::default_config()
    .with_learning_rate(0.05);

// Analyze trajectory and suggest parameter adjustments
let entropy = analyzer.average_entropy(&observer);
let adjustments = tuner.suggest_adjustments(&observer, &params);

// Apply adjustments
for i in 0..5 {
    params.intrinsic_rates[i] += adjustments[i];
}
```

### Geometric Constraint Enforcement

```rust
use bridge::*;

// Create geometric state space
let geo_space = GeometricStateSpace::new([0, 1, 2, 3, 4]);

// Project to Metatron geometry
let geometry = geo_space.project_to_geometry(&state);

// Apply symmetry operations
let rotated = geo_space.apply_c6_rotation(&state, 1);  // 60° rotation
let reflected = geo_space.apply_reflection(&state);     // Reflection

// Enforce constraints
let mut constrained_state = state;
geo_space.enforce_constraints(&mut constrained_state);

// Measure symmetry preservation
let deviation = geo_space.symmetry_deviation(&state);
```

## 🎓 Implementation Phases (ALL COMPLETE)

### ✅ Phase 1: Workspace Integration
- [x] Created 3-crate workspace structure (core, metatron, bridge)
- [x] Reorganized Metatron-R into logical subdirectories
- [x] Fixed 100+ import paths
- [x] All 89 initial tests passing

### ✅ Phase 2: Bridge Layer Foundation
- [x] Implemented ResonanceField trait and implementations
- [x] Created AdaptiveCoupling for time-varying dynamics
- [x] Built GeometricStateSpace for 5D ↔ Metatron mapping
- [x] Added TrajectoryObserver for feedback
- [x] Created CognitiveSimulator skeleton

### ✅ Phase 3: Dynamic Resonance
- [x] Implemented full adaptive integration loop
- [x] Created `integrate_adaptive()` with real-time coupling modulation
- [x] Enhanced adaptive_epidemic.rs example
- [x] Added trajectory analysis and visualization
- [x] 91 tests passing

### ✅ Phase 4: Cognitive Feedback Loop
- [x] Implemented SpectralAnalyzer bridging trajectory observation with QLogic
- [x] Created ParameterTuner using QDASH decision engine
- [x] Connected spectral features to parameter adjustments
- [x] Built self_tuning_ecology.rs example
- [x] 104 tests passing

### ✅ Phase 5: Geometric Constraints
- [x] Implemented full 5D ↔ Metatron geometry projection
- [x] Added C6 rotational symmetry operations (60° steps)
- [x] Added D6 reflection symmetry operations
- [x] Created geometric_finance.rs example
- [x] Added symmetry validation and deviation measurement
- [x] **109 tests passing - COMPLETE**

## 📊 Test Coverage Summary

| Crate | Tests | Coverage |
|-------|-------|----------|
| core_5d | 39 | ✅ Complete |
| metatron | 32 | ✅ Complete |
| bridge | 38 | ✅ Complete |
| **Total** | **109** | **✅ 100%** |

### Test Categories
- **Unit Tests**: All modules fully tested
- **Integration Tests**: Bridge components tested with both frameworks
- **Validation Tests**: Analytical comparisons (linear, harmonic, fixed point)
- **Symmetry Tests**: C6/D6 operations verified
- **Roundtrip Tests**: Geometric projection/reconstruction validated

## 🔬 Mathematical Foundations

### 5D System Evolution
```
dσ/dt = F(σ) = αᵢσᵢ + Σⱼ τᵢⱼ(σᵢ, σⱼ, Cᵢⱼ) + fᵢ(t)
```

### Adaptive Coupling Modulation
```
Cᵢⱼ(t) = C₀ᵢⱼ · R(t, nᵢ, nⱼ)
```
where R is the resonance field mapping 5D nodes to Metatron geometry.

### Spectral Analysis
```
S(ω) = |∫ σ(t) e^(-iωt) dt|²
```

### Heun's Method (RK2)
```
σ̃ⁿ⁺¹ = σⁿ + Δt · F(σⁿ)
σⁿ⁺¹ = σⁿ + (Δt/2) · [F(σⁿ) + F(σ̃ⁿ⁺¹)]
```

## 🎯 Key Features by Component

### Core 5D
- ✅ State validation (NaN/Inf detection)
- ✅ Four coupling types with derivatives
- ✅ Jacobian computation for stability
- ✅ Heun's method with stability detection
- ✅ Eigenvalue analysis
- ✅ Three projection types (orthogonal, isometric, PCA)
- ✅ Domain templates (SIR, financial, predator-prey)
- ✅ Ensemble simulations and parameter sweeps
- ✅ CSV/JSON export

### Metatron-R
- ✅ 13-node Metatron Cube geometry
- ✅ C6/D6 symmetry operations
- ✅ QLogic oscillator (13-node patterns)
- ✅ Spectral grammar (Fourier-like analysis)
- ✅ Entropy analyzer
- ✅ Mandorla resonance fields
- ✅ Gabriel cell lattices
- ✅ Tensor networks
- ✅ MasterAgent orchestration
- ✅ QDASH decision engine

### Bridge
- ✅ ResonanceField trait (constant, oscillatory, Mandorla)
- ✅ AdaptiveCoupling with real-time modulation
- ✅ Full geometric projection (5D ↔ 3D Metatron nodes)
- ✅ Symmetry operations (C6 rotation, D6 reflection)
- ✅ TrajectoryObserver (velocity, acceleration, energy)
- ✅ SpectralAnalyzer (entropy, centroids, frequencies)
- ✅ ParameterTuner (QDASH-based adaptation)
- ✅ CognitiveSimulator (unified integration)

## 🔐 Security

### Dependency Audit
All dependencies verified against GitHub Advisory Database:
- ✅ nalgebra 0.33: No vulnerabilities
- ✅ serde 1.0: No vulnerabilities
- ✅ rand 0.8: No vulnerabilities
- ✅ All transitive dependencies: Clean

### Code Safety
- No unsafe code in bridge layer
- Minimal unsafe in core and metatron (only in nalgebra)
- All inputs validated (finite value checks)
- No external network access
- Controlled file system access

## 📈 Performance

### Build Performance
- Clean workspace build: ~45s (release mode)
- Incremental build: <3s for minor changes
- All tests run in <1s total

### Runtime Performance
- ~10,000 integration steps/second (release mode)
- Efficient linear algebra operations
- Minimal memory allocation
- Parallel-ready architecture

## 🤝 Contributing

This is a research integration project. Contributions should focus on:
1. Additional domain templates
2. Performance optimization
3. Advanced resonance field implementations
4. Extended geometric constraint types
5. Documentation improvements

## 📄 License

This project inherits licenses from its components:
- Core 5D Framework: (See core/LICENSE)
- Metatron-R: MIT License (See metatron/LICENSE)

## 👥 Authors

- Sebastian Klemm (specification & core framework)
- APOLLYON-5D Integration Team (2025)

## 📖 Documentation

- **Core 5D**: See [`core/README.md`](core/README.md)
- **Metatron-R**: See [`metatron/README.md`](metatron/README.md)
- **API Reference**: See [`API.md`](API.md)
- **Development Guide**: See [`DEVELOPMENT.md`](DEVELOPMENT.md)
- **Implementation Details**: See [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md)
- **Integration Summary**: See [`INTEGRATION_SUMMARY.md`](INTEGRATION_SUMMARY.md)

## 🎉 Project Status

**Version**: 1.0.0 - COMPLETE IMPLEMENTATION  
**Status**: All 5 Phases Complete ✅  
**Tests**: 109/109 Passing ✅  
**Examples**: 3/3 Working ✅  
**Security**: 0 Vulnerabilities ✅  
**Date**: October 2025

---

**"Bridging deterministic mathematics with geometric cognition—the APOLLYON-5D Framework represents a new paradigm in adaptive dynamical systems."**

