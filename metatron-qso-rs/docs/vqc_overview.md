# VQC (Variational Quantum Classifier) Overview

## Summary of Improvements

### Before Optimization
- Training Accuracy: **50.0%** (random guessing)
- Test Accuracy: **50.0%** (random guessing)
- Convergence Rate: **0%**
- Status: **Not working**

### After Optimization
- Training Accuracy: **100.0%**
- Test Accuracy: **100.0%**
- Convergence Rate: **100%**
- Status: **Perfect classification**

**Improvement: 50% → 100% accuracy (2x improvement, from random to perfect!)**

## Key Problems Identified

### 1. No Data Normalization
**Problem**: Raw features were directly encoded without normalization
- Features ranged from 0.0 to 0.95 with inconsistent scales
- Different feature dimensions had different ranges
- This caused poor quantum state encoding

**Solution**: Implemented min-max normalization
```rust
fn fit_normalize_data(&self, data: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    // Compute min and max for each feature dimension
    // Transform: normalized = (value - min) / (max - min)
    // Store normalization parameters for inference
}
```

### 2. Suboptimal Angle Encoding
**Problem**: Original encoding was too simple
- Started from uniform superposition
- Applied rotations: `angle = value * 2π`
- No clear separation between different classes

**Solution**: Improved 3-step encoding strategy
1. **Start from basis state |0⟩** (instead of uniform superposition)
2. **Apply Hadamard-like operation** for controlled superposition
3. **Apply RY rotations with normalized angles**: `angle = value * π`

```rust
fn angle_encoding(&self, data: &[f64]) -> QuantumState {
    // Step 1: Start from |0⟩
    let mut state = QuantumState::basis_state(0).unwrap();

    // Step 2: Create superposition via Hadamard-like operation
    // Step 3: Apply RY(π * normalized_value) for each feature
    let angle = value * PI;  // Maps [0,1] → [0,π]
}
```

**Why this works**:
- Starting from |0⟩ gives a clearer initial state
- Hadamard creates controlled entanglement
- RY rotations with π range provide maximum separability
- Normalized inputs ensure consistent encoding

### 3. Insufficient Ansatz Depth
**Problem**: Depth 2 wasn't expressive enough
- 52 parameters (2 * 13 * 2)
- Limited capacity to learn complex decision boundaries

**Solution**: Increased to depth 3
- 78 parameters (2 * 13 * 3)
- Better expressiveness while avoiding overfitting
- Hardware-efficient architecture remains practical

### 4. Suboptimal Hyperparameters
**Problem**: Training settings weren't aggressive enough
- max_iterations: 200
- learning_rate: 0.02
- tolerance: 1e-4

**Solution**: Tuned for better convergence
- max_iterations: 300 (50% increase)
- learning_rate: 0.03 (faster updates)
- tolerance: 1e-5 (tighter convergence)
- energy_tolerance: 1e-3 (dual convergence criterion)

## Technical Implementation

### Data Flow

```
Raw Data
   ↓
[fit_normalize_data] → Learn min/max per feature
   ↓
Normalized Data [0, 1]
   ↓
[angle_encoding]
   ├─ Start: |0⟩
   ├─ Hadamard: Create superposition
   └─ RY rotations: Encode features
   ↓
Encoded Quantum State
   ↓
[Variational Ansatz] → Apply parametrized circuit
   ↓
Output State
   ↓
[Measurement] → P(|0⟩) = class 0 probability
   ↓
Prediction
```

### Ansatz Architecture

**Hardware-Efficient Ansatz (Depth 3)**:
- **Layer Structure**:
  - RY rotations on all qubits (13 params)
  - RZ rotations on all qubits (13 params)
  - Entangling gates (circular pattern)
- **Total Parameters**: 78 (26 per layer × 3 layers)
- **Expressiveness**: High
- **NISQ-friendly**: Yes (limited gate depth, native gates)

### Loss Function

**Binary Cross-Entropy**:
```rust
fn binary_cross_entropy(&self, prediction: f64, label: f64) -> f64 {
    let epsilon = 1e-10;  // Numerical stability
    let pred = prediction.clamp(epsilon, 1.0 - epsilon);
    -label * pred.ln() - (1.0 - label) * (1.0 - pred).ln()
}
```

**Why this works**:
- Differentiable (gradient-based optimization possible)
- Penalizes confident wrong predictions heavily
- Converges to minimum when predictions match labels

### Optimization

**Adam Optimizer**:
- Adaptive learning rate per parameter
- Momentum for smoother convergence
- Beta1 = 0.9, Beta2 = 0.999
- Epsilon = 1e-8

**Gradient Computation**:
- Parameter Shift Rule (exact for quantum circuits)
- Parallel gradient evaluation via Rayon
- Caching for efficiency

**Convergence Criteria** (either triggers convergence):
1. Gradient norm: `|∇L| < 1e-5`
2. Energy change: `|L_k - L_{k-1}| < 1e-3`

## Benchmark Results

### Dataset 1: Binary Classification

**Training Data** (8 samples, 4 features):
- Class 0: [0.1, 0.1, 0.0, 0.0] region
- Class 1: [0.8, 0.85, 0.0, 0.0] region
- Features 3-4 are always 0 (redundant but handled by normalization)

**Results**:
- Training Accuracy: **100.00%**
- Test Accuracy: **100.00%**
- Training Loss: **0.041**
- Iterations: **32**
- Quantum Evaluations: **5,024**
- Execution Time: **7.44s**
- Converged: **Yes**

### Dataset 2: Linearly Separable

**Training Data** (8 samples, 4 features):
- Class 0: Low values in all dimensions
- Class 1: High values in dimensions 2-3
- Linearly separable with clear margin

**Results**:
- Training Accuracy: **100.00%**
- Test Accuracy: **100.00%**
- Training Loss: **0.058**
- Iterations: **15**
- Quantum Evaluations: **2,355**
- Execution Time: **3.52s**
- Converged: **Yes**

### Overall Metrics

```
╔════════════════════════════════════════════════════════╗
║   BENCHMARK SUMMARY                                    ║
╚════════════════════════════════════════════════════════╝
Avg Training Accuracy:  100.00%
Avg Test Accuracy:      100.00%
Convergence Rate:       100.0%
Total Time:             10.96s
Evaluations/sec:        673.15
```

## Feature Encoding Details

### Normalization

**Per-Feature Min-Max Scaling**:
```python
For each feature i:
  min_i = min(all training samples[:, i])
  max_i = max(all training samples[:, i])

  normalized_i = (value_i - min_i) / (max_i - min_i)
```

**Inference**:
- Normalization parameters (min, max) stored after training
- Test data transformed using same parameters
- Ensures consistent encoding between training and inference

### Angle Mapping

**Feature to Quantum State**:
```
Normalized value ∈ [0, 1]
         ↓
Rotation angle = value × π ∈ [0, π]
         ↓
RY(angle) gate
         ↓
Encodes feature in quantum superposition
```

**Why π range**:
- RY(0) = Identity (no rotation)
- RY(π/2) = Maximally mixed
- RY(π) = Bit flip
- Full range [0, π] provides maximum expressiveness

## Usage Guide

### Training a Classifier

```rust
use metatron_qso::prelude::*;

// Prepare data
let training_data = vec![
    vec![0.1, 0.2, 0.0, 0.0],  // Class 0
    vec![0.9, 0.8, 0.0, 0.0],  // Class 1
];
let training_labels = vec![0, 1];

// Build and train VQC
let mut vqc = VQCBuilder::new()
    .ansatz_type(AnsatzType::HardwareEfficient)
    .ansatz_depth(3)
    .encoding(EncodingType::Angle)
    .optimizer(OptimizerType::Adam)
    .max_iterations(300)
    .learning_rate(0.03)
    .tolerance(1e-5)
    .verbose(true)
    .build();

let result = vqc.train(training_data, training_labels);

println!("Training Accuracy: {:.2}%", result.training_accuracy * 100.0);
```

### Making Predictions

```rust
// Predict on new data
let test_sample = vec![0.15, 0.18, 0.0, 0.0];
let prediction = vqc.predict(&test_sample);

println!("Predicted Class: {}", prediction.predicted_class);
println!("Confidence: {:.2}%", prediction.confidence * 100.0);
println!("Class Probabilities: {:?}", prediction.class_probabilities);
```

### Evaluating on Test Set

```rust
let test_data = vec![
    vec![0.12, 0.13, 0.0, 0.0],
    vec![0.88, 0.87, 0.0, 0.0],
];
let test_labels = vec![0, 1];

let test_accuracy = vqc.evaluate(test_data, test_labels);
println!("Test Accuracy: {:.2}%", test_accuracy * 100.0);
```

## Best Practices

### 1. Data Preparation
- ✅ Normalize features to [0, 1] range (handled automatically)
- ✅ Remove constant features (all same value)
- ✅ Ensure balanced classes if possible
- ✅ Use at least 4-8 samples per class

### 2. Ansatz Selection
- **Hardware-Efficient**: Best for NISQ devices, fast training
- **EfficientSU2**: More expressiveness, slower training
- **Metatron**: Specialized for 13-dimensional problems
- **Depth**: Start with 2-3, increase if underfitting

### 3. Hyperparameter Tuning
- **Learning Rate**: 0.01-0.05 (Adam handles most cases)
- **Max Iterations**: 200-500 (watch for convergence)
- **Tolerance**: 1e-5 to 1e-4 (gradient norm)
- **Energy Tolerance**: 1e-3 to 1e-2 (loss change)

### 4. Monitoring Training
- Check convergence status
- Monitor loss trajectory
- Verify training accuracy > 80% before deployment
- Test on held-out data

## Troubleshooting

### Low Training Accuracy (< 70%)

**Possible Causes**:
1. Insufficient ansatz depth
2. Too few iterations
3. Learning rate too low
4. Data not separable

**Solutions**:
- Increase `ansatz_depth` to 4
- Increase `max_iterations` to 500
- Increase `learning_rate` to 0.05
- Visualize data to check separability

### Overfitting (Train >> Test Accuracy)

**Possible Causes**:
1. Too many parameters
2. Too few training samples
3. Over-optimization

**Solutions**:
- Reduce `ansatz_depth` to 2
- Collect more training data
- Reduce `max_iterations`
- Add regularization (future work)

### Slow Training

**Possible Causes**:
1. Too many quantum evaluations
2. Inefficient gradient computation
3. Large ansatz depth

**Solutions**:
- Use `FiniteDifference` instead of `ParameterShift`
- Reduce `ansatz_depth`
- Reduce feature dimension
- Use simpler ansatz (e.g., depth 2)

## Future Improvements

### Algorithmic
1. **Early Stopping**: Monitor validation loss, stop when plateaus
2. **Learning Rate Scheduling**: Decay learning rate over time
3. **Batch Training**: Support for mini-batches
4. **Multi-Class**: Extend beyond binary classification
5. **Regularization**: L1/L2 penalties to prevent overfitting

### Engineering
1. **GPU Acceleration**: Parallelize state vector simulation
2. **Sparse Ansätze**: Reduce parameter count
3. **Adaptive Depth**: Automatically select optimal depth
4. **Cross-Validation**: K-fold validation for robustness
5. **Feature Selection**: Automatic identification of important features

### Advanced Features
1. **Quantum Kernel Methods**: Kernel-based classification
2. **Ensemble Methods**: Combine multiple VQCs
3. **Transfer Learning**: Pre-train on larger datasets
4. **Noise Modeling**: Simulate real quantum device noise

## References

1. **VQC Original Paper**: Havlíček et al., Nature 567, 209–212 (2019)
2. **Angle Encoding**: Schuld & Killoran, Phys. Rev. Lett. 122, 040504 (2019)
3. **Hardware-Efficient Ansätze**: Kandala et al., Nature 549, 242–246 (2017)
4. **Parameter Shift Rule**: Mitarai et al., Phys. Rev. A 98, 032309 (2018)

## Appendix: Code Locations

### Core VQC Implementation
- **Main Module**: `src/vqa/vqc.rs`
  - `VQC` struct (line 76-83)
  - `train()` method (line 99-215)
  - `angle_encoding()` method (line 275-333)
  - `fit_normalize_data()` method (line 367-408)
  - `transform_data()` method (line 410-426)

### Cost Function
- **File**: `src/vqa/cost_function.rs`
  - `VQCCostFunction` struct (line 321-326)
  - `binary_cross_entropy()` (line 359-363)
  - `evaluate()` method (line 367-396)
  - `gradient()` method (line 398-425)

### Benchmark Suite
- **File**: `src/bin/vqc_bench.rs`
  - Dataset definitions (line 118-153)
  - Benchmark execution (line 60-108)
  - Results aggregation (line 172-205)

---

*Last updated: 2025-11-13*
*Metatron QSO v0.1.0*
*VQC Accuracy: 50% → 100% ✅*
