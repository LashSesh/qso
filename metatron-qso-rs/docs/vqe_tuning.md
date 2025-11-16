# VQE Optimization and Tuning Guide

## Overview

This document describes the optimizations made to the Variational Quantum Eigensolver (VQE) implementation in Metatron QSO, specifically focusing on improving the Metatron ansatz performance and implementing robust convergence criteria.

## Summary of Improvements

### Before Optimization
- Metatron ansatz (depth=1): ground energy = **-12.586**
- Approximation error: **0.414** (3.2% relative error)
- Convergence rate: **0%** (none of the runs converged)
- Only one optimizer tested (Adam)
- Fixed, simple entanglement pattern

### After Optimization
- Metatron ansatz (depth=1): ground energy = **-12.999**
- Approximation error: **0.0012** (0.0095% relative error)
- Metatron ansatz (depth=3): ground energy = **-12.999**
- Convergence rate: **100%** (all runs converge)
- Multiple optimizers supported (Adam, L-BFGS, NelderMead, GradientDescent)
- Configurable entanglement strategies (Ring, Full)
- Multi-start capability (3-5 random initializations)
- Quality score metric for comparing runs

**Energy improvement: ~336x reduction in error!**

## 1. Ansatz Architecture Improvements

### 1.1 Metatron Ansatz Entanglement Strategies

The Metatron ansatz now supports two entanglement strategies:

#### Ring Entanglement (Default)
```rust
EntanglementStrategy::Ring
```
- Circular nearest-neighbor connections
- Each qubit connected to next in ring pattern
- **Parameters per layer**: `num_qubits + num_qubits` = 26 (for 13-dimensional system)
- Efficient for NISQ devices
- Good balance between expressiveness and parameter count

#### Full Entanglement
```rust
EntanglementStrategy::Full
```
- All-to-all connections between qubits
- Maximum expressiveness
- **Parameters per layer**: `num_qubits + num_qubits*(num_qubits-1)/2` = 91 (for 13-dimensional system)
- Higher computational cost but better representation capacity

### 1.2 Configurable Depth

The ansatz depth can now be configured (recommended range: 1-3):

- **Depth 1**: 26 parameters, fast convergence, excellent for Metatron symmetry
- **Depth 2**: 52 parameters, risk of local minima with certain initializations
- **Depth 3**: 78 parameters, highest quality, slower but very accurate

**Recommendation**: Start with depth=1 for Metatron ansatz. The inherent symmetry of the Metatron Cube structure makes shallow circuits surprisingly effective.

### 1.3 Usage Example

```rust
use metatron_qso::vqa::{MetatronAnsatz, EntanglementStrategy};

// Ring entanglement (default)
let ansatz = MetatronAnsatz::new(depth);

// Full entanglement
let ansatz = MetatronAnsatz::new_with_entanglement(
    depth,
    EntanglementStrategy::Full
);
```

## 2. Convergence Criteria

### 2.1 Dual Convergence Check

The optimizers now implement two convergence criteria (either triggers convergence):

#### Gradient-Based Convergence
```
|∇E| < tolerance (default: 1e-6)
```
- Traditional gradient norm criterion
- Strict but can be too conservative for VQE
- Good for smooth optimization landscapes

#### Energy-Based Convergence
```
|E_k - E_{k-1}| < energy_tolerance (default: 1e-3)
```
- Checks if energy change between iterations is below threshold
- More practical for VQE applications
- Allows convergence even when gradients remain non-zero
- **Key improvement**: Enables successful convergence on plateaus

### 2.2 Configuration

```rust
let vqe = VQEBuilder::new()
    .hamiltonian(hamiltonian)
    .tolerance(1e-6)           // Gradient tolerance
    .energy_tolerance(1e-3)    // Energy change tolerance
    .build();
```

### 2.3 Convergence Rate Metric

The benchmark suite now calculates:
```
convergence_rate = (number of converged runs) / (total runs)
```

This provides visibility into optimizer stability across different configurations.

## 3. Multi-Start Strategy

### 3.1 Motivation

The VQE optimization landscape can have multiple local minima. Multi-start strategy runs optimization multiple times with different random initializations and keeps the best result.

### 3.2 Implementation

```rust
let vqe = VQEBuilder::new()
    .hamiltonian(hamiltonian)
    .num_random_starts(3)  // Run 3 times, keep best
    .build();
```

### 3.3 Recommendations

- **Single run** (num_random_starts=1): For well-behaved ansätze (depth=1, depth=3)
- **3 starts**: Good balance for exploring parameter space
- **5 starts**: When optimization budget allows, provides robust results
- **> 5 starts**: Diminishing returns unless landscape is very complex

### 3.4 Observed Results

- Metatron depth=1: Single start sufficient (landscape is smooth)
- Metatron depth=2: Multi-start doesn't help (deterministic local minimum)
- Metatron depth=3: Single start sufficient

## 4. Optimizer Configurations

### 4.1 Adam (Recommended)

```rust
.optimizer(OptimizerType::Adam)
.max_iterations(100)
.learning_rate(0.01)
```

**Strengths**:
- Adaptive learning rate
- Works well with Metatron ansatz
- Fast convergence
- Robust to hyperparameter choices

**Results**:
- Metatron depth=1: -12.999 (excellent)
- Metatron depth=3: -12.999 (excellent)

### 4.2 L-BFGS

```rust
.optimizer(OptimizerType::LBFGS)
.max_iterations(100)
.learning_rate(0.01)
```

**Strengths**:
- Quasi-Newton method
- Good for smooth landscapes
- Memory-efficient (limited-memory)

**Known Issues**:
- Requires careful learning rate tuning
- May fail with default settings on Metatron depth=2
- Consider lowering learning_rate to 0.001 for Metatron ansatz

### 4.3 NelderMead

```rust
.optimizer(OptimizerType::NelderMead)
.max_iterations(200)
```

**Strengths**:
- Gradient-free
- Robust to noisy gradients
- Good for exploratory optimization

**Weaknesses**:
- Slower convergence
- Requires more iterations

### 4.4 GradientDescent

```rust
.optimizer(OptimizerType::GradientDescent)
.max_iterations(100)
.learning_rate(0.01)
```

**Strengths**:
- Simple and predictable
- With momentum: smoother convergence

**Weaknesses**:
- May require more careful learning rate tuning

## 5. Quality Score

### 5.1 Definition

The quality score combines energy accuracy and convergence status:

```rust
quality_score = convergence_factor * (1 - normalized_error)

where:
  normalized_error = min(approximation_error / |classical_ground|, 1.0)
  convergence_factor = 1.0 if converged, 0.5 otherwise
```

### 5.2 Interpretation

- **1.0**: Perfect convergence with exact ground state energy
- **0.95-0.99**: Excellent (< 5% error, converged)
- **0.9-0.95**: Good (5-10% error, converged)
- **0.5-0.9**: Converged but significant energy error
- **< 0.5**: Not converged or large error

### 5.3 Benchmark Results

| Configuration | Energy | Quality Score |
|--------------|--------|---------------|
| Metatron depth=1 | -12.999 | **1.000** |
| Metatron depth=3 | -12.999 | **1.000** |
| EfficientSU2 depth=2 | -12.977 | **0.998** |
| HardwareEfficient depth=2 | -12.967 | **0.997** |
| Metatron depth=2 | -11.827 | **0.910** |

## 6. Benchmark Suite

### 6.1 Running Benchmarks

```bash
# Build the benchmark
cargo build --release --bin vqe_bench

# Run and save results
cargo run --release --bin vqe_bench -- ci/vqe_baseline.json

# View results
cat ci/vqe_baseline.json | jq
```

### 6.2 Benchmark Configurations

The benchmark suite tests:
1. **Baseline**: HardwareEfficient, EfficientSU2, Metatron (depth=1)
2. **Enhanced Metatron**: depth=2, depth=3
3. **Multi-start**: Metatron depth=2 with 3 random starts
4. **Alternative optimizer**: Metatron with L-BFGS

### 6.3 Key Metrics

- `ground_energy`: VQE-optimized ground state energy
- `classical_ground`: Exact ground state from diagonalization
- `approximation_error`: Absolute difference
- `converged`: Whether optimization converged
- `quality_score`: Combined quality metric (0-1)
- `quantum_evaluations`: Number of quantum circuit evaluations
- `execution_time_ms`: Wall-clock time

## 7. Best Practices

### 7.1 For Production Use

1. **Start simple**: Begin with Metatron depth=1
2. **Use energy convergence**: Set `energy_tolerance = 1e-3`
3. **Stick with Adam**: Unless you have specific reasons to change
4. **Monitor quality_score**: Aim for > 0.95
5. **Check convergence**: Ensure `converged = true`

### 7.2 For Research/Exploration

1. **Test multiple depths**: 1, 2, 3
2. **Compare optimizers**: Adam, L-BFGS, NelderMead
3. **Use multi-start**: 3-5 starts for robustness
4. **Analyze history**: Look at optimization trajectories
5. **Plot convergence**: Energy vs iteration

### 7.3 Troubleshooting

#### Low quality score (< 0.9)
- Increase ansatz depth
- Try multi-start strategy
- Lower learning rate
- Switch to NelderMead (gradient-free)

#### Not converging
- Relax energy_tolerance to 1e-2
- Increase max_iterations to 200
- Try different optimizer
- Check ansatz parameter count

#### Very slow
- Reduce ansatz depth
- Use Ring entanglement instead of Full
- Reduce num_random_starts
- Consider HardwareEfficient ansatz

## 8. Implementation Details

### 8.1 Key Code Locations

- **Ansatz**: `src/vqa/ansatz.rs`
  - MetatronAnsatz struct
  - EntanglementStrategy enum

- **Optimizers**: `src/vqa/optimizer.rs`
  - OptimizerConfig (tolerance, energy_tolerance)
  - Convergence checking logic

- **VQE**: `src/vqa/vqe.rs`
  - VQEConfig (num_random_starts)
  - Multi-start implementation

- **Benchmarks**: `src/bin/vqe_bench.rs`
  - Quality score calculation
  - Comprehensive test suite

### 8.2 Adding New Entanglement Strategies

```rust
// In src/vqa/ansatz.rs
pub enum EntanglementStrategy {
    Ring,
    Full,
    Linear,  // Add new strategy
}

impl MetatronAnsatz {
    fn num_entangling_gates(&self) -> usize {
        match self.entanglement_strategy {
            EntanglementStrategy::Ring => METATRON_DIMENSION,
            EntanglementStrategy::Full => METATRON_DIMENSION * (METATRON_DIMENSION - 1) / 2,
            EntanglementStrategy::Linear => METATRON_DIMENSION - 1,  // Implement
        }
    }

    // Update apply() method to handle new strategy
}
```

## 9. Future Improvements

### Potential Enhancements

1. **Adaptive depth**: Automatically select ansatz depth based on convergence
2. **Layer-wise training**: Train one layer at a time (analogous to greedy pre-training)
3. **Parameter initialization**: Smarter initialization based on problem structure
4. **Gradient clipping**: Prevent gradient explosions
5. **Learning rate scheduling**: Adaptive learning rate decay
6. **Natural gradient**: Use quantum natural gradient (QNG) for faster convergence
7. **COBYLA optimizer**: Add gradient-free COBYLA as alternative

## 10. References

- VQE original paper: Peruzzo et al., Nature Comm. 5, 4213 (2014)
- Hardware-efficient ansätze: Kandala et al., Nature 549, 242–246 (2017)
- Convergence criteria: McClean et al., Nat. Comm. 9, 4812 (2018)

## Appendix: Benchmark Results Summary

```
╔════════════════════════════════════════════════════════╗
║   BENCHMARK SUMMARY                                    ║
╚════════════════════════════════════════════════════════╝
Configurations Tested:  8
Best Ground Energy:     -12.9992084007
Convergence Rate:       100.0%
Total Time:             4374.70ms
Evaluations/sec:        9453.67
```

### Classical Ground State
- Energy: **-13.0000000000**

### Top Performers
1. Metatron depth=3: -12.999208 (0.0061% error, quality=1.000)
2. Metatron depth=1: -12.998769 (0.0095% error, quality=1.000)
3. EfficientSU2 depth=2: -12.976725 (0.179% error, quality=0.998)

---

*Last updated: 2025-11-13*
*Metatron QSO v0.1.0*
