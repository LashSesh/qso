# TRITON Search Core

**T**ransformative **R**esonance **I**teration with **T**opological **O**ptimization **N**avigator

A golden-angle spiral evolution strategy for quantum algorithm calibration in Q⊗DASH.

## Overview

TRITON implements a momentum-based evolutionary search using the golden angle (2π / φ²) to explore parameter spaces efficiently. It evaluates configurations using a three-dimensional spectral signature (ψ, ρ, ω) and maximizes their product (resonance).

### Core Concepts

- **Spectral Signature**: Three-dimensional quality metric
  - `ψ` (Psi): Quality or fidelity
  - `ρ` (Rho): Stability or consistency
  - `ω` (Omega): Efficiency or speed

- **Resonance**: Product of the three metrics (ψ × ρ × ω), the optimization target

- **Golden Spiral**: Exploration pattern using the golden angle (≈137.5°) for optimal space coverage

- **Momentum**: Gradient-driven adaptation that learns from successful directions

## Architecture

```
metatron_triton/
├── signature.rs    # SpectralSignature (ψ, ρ, ω, resonance)
├── spiral.rs       # TritonSpiral (golden-angle evolution)
├── search.rs       # TritonSearch (search engine)
└── strategy.rs     # CalibrationSearchStrategy (SCS integration)
```

## Quick Start

### Basic Search Example

```rust
use metatron_triton::{TritonSearch, SpectralSignature};

// Define an evaluation function
let evaluator = |params: &[f64]| {
    // Your calibration logic here
    // Returns ψ, ρ, ω metrics
    SpectralSignature::new(0.8, 0.9, 0.75)
};

// Create search with 5 parameters, max 100 steps
let mut search = TritonSearch::new(5, 42, evaluator, 100);

// Run search loop
while !search.is_converged() {
    let result = search.step();
    println!("Step {}: resonance = {:.4}",
             result.step, result.signature.resonance());
}

// Get best configuration found
if let Some(best) = search.best_point() {
    println!("Best parameters: {:?}", best);
    println!("Best resonance: {:.4}",
             search.best_signature().unwrap().resonance());
}
```

### Integration with Seraphic Calibration Shell

```rust
use metatron_triton::{TritonSearchStrategy, ParameterSpec, TransformType};

// Define parameter space
let params = vec![
    ParameterSpec {
        name: "learning_rate".to_string(),
        min: 0.001,
        max: 0.1,
        transform: TransformType::Logarithmic,
    },
    ParameterSpec {
        name: "batch_size".to_string(),
        min: 16.0,
        max: 256.0,
        transform: TransformType::Integer,
    },
];

// Create strategy
let mut strategy = TritonSearchStrategy::new(params, 42);

// Use in calibration loop
loop {
    let proposal = strategy.propose_next();

    // Run calibration with proposed parameters
    let (psi, rho, omega) = run_calibration(&proposal.parameters);

    // Register result
    strategy.register_result(&CalibrationResult {
        parameters: proposal.parameters.clone(),
        psi,
        rho,
        omega,
    });

    if strategy.is_converged() {
        break;
    }
}

// Get best configuration
let best = strategy.best_configuration().unwrap();
println!("Optimal parameters: {:?}", best.parameters);
```

## API Reference

### SpectralSignature

Three-dimensional quality metric with resonance calculation.

```rust
pub struct SpectralSignature {
    pub psi: f64,    // Quality/Fidelity (0.0 - 1.0)
    pub rho: f64,    // Stability/Consistency (0.0 - 1.0)
    pub omega: f64,  // Efficiency/Speed (0.0 - 1.0)
}

impl SpectralSignature {
    pub fn new(psi: f64, rho: f64, omega: f64) -> Self;
    pub fn resonance(&self) -> f64;  // ψ × ρ × ω
    pub fn harmonic_mean(&self) -> f64;  // 3 / (1/ψ + 1/ρ + 1/ω)
}
```

### TritonSpiral

Golden-angle spiral evolution with momentum.

```rust
pub struct TritonSpiral { /* ... */ }

impl TritonSpiral {
    // Create with default parameters
    pub fn new(dimension: usize, seed: u64) -> Self;

    // Create with custom parameters
    pub fn with_params(
        dimension: usize,
        seed: u64,
        radius_base: f64,
        learning_rate: f64,
        momentum_decay: f64,
        noise_level: f64,
    ) -> Self;

    // Generate next point in [0, 1]^n
    pub fn next_point(&mut self) -> Vec<f64>;

    // Update momentum based on gradient and reward
    pub fn update_momentum(&mut self, gradient: &[f64], reward: f64);

    // Update position to best point found
    pub fn update_position(&mut self, new_position: &[f64]);

    // Reset to initial state
    pub fn reset(&mut self);

    // Getters
    pub fn step(&self) -> usize;
    pub fn position(&self) -> &[f64];
    pub fn momentum(&self) -> &[f64];
    pub fn radius(&self) -> f64;
}
```

**Default Parameters:**
- `radius_base`: 0.3 (initial exploration radius)
- `learning_rate`: 0.1 (momentum update rate)
- `momentum_decay`: 0.9 (momentum persistence)
- `noise_level`: 0.05 (exploration noise)

### TritonSearch

Complete search engine combining spiral and evaluation.

```rust
pub struct TritonSearch<Eval> { /* ... */ }

impl<Eval> TritonSearch<Eval>
where
    Eval: Fn(&[f64]) -> SpectralSignature,
{
    pub fn new(
        dimension: usize,
        seed: u64,
        evaluator: Eval,
        max_steps: usize,
    ) -> Self;

    // Run one search step
    pub fn step(&mut self) -> TritonStepResult;

    // Check convergence
    pub fn is_converged(&self) -> bool;

    // Get results
    pub fn best_signature(&self) -> Option<SpectralSignature>;
    pub fn best_point(&self) -> Option<&[f64]>;
    pub fn current_step(&self) -> usize;
    pub fn improvement_rate(&self) -> f64;
}

pub struct TritonStepResult {
    pub step: usize,
    pub point: Vec<f64>,
    pub signature: SpectralSignature,
    pub is_best: bool,
}
```

**Convergence Criteria:**
- No improvement in best resonance for 20 consecutive steps
- Maximum step limit reached

### CalibrationSearchStrategy (Trait)

Interface for integration with Seraphic Calibration Shell.

```rust
pub trait CalibrationSearchStrategy: Send {
    fn propose_next(&mut self) -> CalibrationProposal;
    fn register_result(&mut self, result: &CalibrationResult);
    fn best_configuration(&self) -> Option<CalibrationProposal>;
    fn statistics(&self) -> SearchStatistics;
    fn reset(&mut self);
    fn is_converged(&self) -> bool;
}
```

### TritonSearchStrategy

TRITON implementation of the calibration strategy.

```rust
pub struct TritonSearchStrategy { /* ... */ }

impl TritonSearchStrategy {
    pub fn new(parameters: Vec<ParameterSpec>, seed: u64) -> Self;
}

pub struct ParameterSpec {
    pub name: String,
    pub min: f64,
    pub max: f64,
    pub transform: TransformType,
}

pub enum TransformType {
    Linear,       // min + (max - min) × x
    Logarithmic,  // exp(log(min) + (log(max) - log(min)) × x)
    Integer,      // round(Linear)
}
```

## Mathematical Background

### Golden Angle

The golden angle is derived from the golden ratio φ = (1 + √5) / 2:

```
θ = 2π / φ² ≈ 2.399963229728653 radians ≈ 137.5°
```

This angle provides optimal space-filling properties, used in nature (phyllotaxis) and here for efficient parameter space exploration.

### Spiral Evolution

Point generation at step `n`:

```
θ_n = n × golden_angle
r_n = radius × √(n / dimension)

x_i = position_i + r_n × cos(θ_n + 2π × i / dimension) + momentum_i + noise_i
```

### Momentum Update

After evaluating a point:

```
gradient_norm = ||∇resonance||
momentum_i = decay × momentum_i + learning_rate × reward × (∇_i / gradient_norm)
```

Where `reward` is the current resonance value normalized to [0, 1].

### Resonance Metric

The optimization target:

```
resonance = ψ × ρ × ω
```

This multiplicative formulation ensures balanced improvement across all three dimensions, as a low value in any metric severely impacts the overall score.

## Configuration

### Spiral Tuning

```rust
let spiral = TritonSpiral::with_params(
    dimension,
    seed,
    0.5,    // radius_base: larger = wider exploration
    0.15,   // learning_rate: larger = faster adaptation
    0.85,   // momentum_decay: larger = longer memory
    0.1,    // noise_level: larger = more stochastic
);
```

**Recommendations:**
- **Smooth landscapes**: Lower noise (0.01-0.05), higher momentum decay (0.9-0.95)
- **Rugged landscapes**: Higher noise (0.05-0.15), lower momentum decay (0.8-0.9)
- **Large search spaces**: Larger radius_base (0.4-0.6), moderate learning rate (0.1-0.2)
- **Small search spaces**: Smaller radius_base (0.2-0.4), higher learning rate (0.15-0.25)

### Convergence Tuning

```rust
// Modify patience in TritonSearch
const PATIENCE: usize = 20;  // Steps without improvement before converging
```

## Performance Characteristics

- **Time Complexity**: O(n) per step, where n is dimension
- **Space Complexity**: O(n + h), where h is history size (default 50)
- **Typical Convergence**: 50-200 steps for 5-10 parameters
- **Recommended Maximum**: 500-1000 steps

## Testing

Run the test suite:

```bash
cargo test -p metatron_triton
```

Current test coverage:
- Unit tests: 25 passing
- Doc tests: 5 passing
- Integration tests: Included in strategy tests

## Integration Status

- ✅ Core implementation complete
- ✅ Unit tests passing
- ✅ Telemetry API integration (`TritonStatus` in `metatron_telemetry`)
- ⏳ Dashboard UI (pending)
- ⏳ SCS wiring (optional, trait-based interface ready)

## Telemetry Integration

TRITON metrics are exposed via `TritonStatus` in the telemetry API:

```rust
pub struct TritonStatus {
    pub step: usize,
    pub best_resonance: f64,
    pub current_resonance: f64,
    pub improvement_rate: f64,
    pub converged: bool,
    pub num_parameters: usize,
}
```

Access via:
```rust
let status = app_state.get_status().await;
if let Some(triton) = status.triton_status {
    println!("TRITON step {}: resonance = {:.4}",
             triton.step, triton.best_resonance);
}
```

## Examples

See individual module documentation for more examples:
- `signature.rs`: Metric calculation examples
- `spiral.rs`: Spiral evolution examples
- `search.rs`: Complete search examples
- `strategy.rs`: SCS integration examples

## References

- **Golden Angle**: Vogel, H. (1979). "A better way to construct the sunflower head"
- **Momentum Optimization**: Sutton, R.S. (1986). "Two problems with backpropagation"
- **Multi-objective Optimization**: Deb, K. (2001). "Multi-Objective Optimization using Evolutionary Algorithms"

## License

Part of Q⊗DASH (Quantum-Hybrid Differential Algorithmic Substrate).

---

**Version**: 0.1.0
**Last Updated**: 2025-01-13
**Stability**: Experimental
