# Seraphic Calibration Shell (SCS) for Q⊗DASH (Metatron VM)

## Overview

The **Seraphic Calibration Shell (SCS)** is a meta-layer around the Q⊗DASH (Metatron VM) quantum-hybrid system that enforces fixpoint-directed optimization dynamics. It automatically calibrates algorithmic configurations to move the system monotonically towards optimal fixpoint attractors.

## What is the SCS?

The SCS wraps the existing Q⊗DASH core (VQE, QAOA, quantum walks, Grover, Boson sampling, VQC/QML) with an intelligent feedback and contraction layer that:

1. **Monitors performance** through a triplet Φ(c) = (ψ, ρ, ω) measuring:
   - ψ: semantic quality (how well algorithms perform)
   - ρ: stability/path invariance (robustness under variations)
   - ω: phase readiness/efficiency (resource usage)

2. **Accumulates field intelligence** in a Mandorla-like calibration field M(t) that encodes resonance patterns from benchmark history

3. **Applies a double-kick operator** T = Φ_V ∘ Φ_U that:
   - First improves quality (update kick Φ_U)
   - Then stabilizes and optimizes efficiency (stabilization kick Φ_V)
   - Creates locally contractive dynamics towards fixpoint attractors

4. **Tests acceptance** via Proof-of-Resonance (PoR) criterion ensuring:
   - Quality never decreases
   - Stability remains consistent
   - Efficiency stays above threshold
   - Field resonance is positive

5. **Enables regime switching** through CRI-style resonance impulses when the system stagnates, allowing controlled transitions between different algorithm families

## Key Concepts

### Performance Triplet Φ(c) = (ψ, ρ, ω)

For any configuration c (set of hyperparameters, ansatz choices, optimizer settings):

- **ψ(c)** ∈ [0, 1]: Quality score aggregated from:
  - VQE ground-state energy accuracy
  - QAOA approximation ratios
  - Grover search success rates
  - Boson sampling visibility
  - QML classification accuracy

- **ρ(c)** ∈ [0, 1]: Stability score measuring:
  - Low variance across random seeds
  - Consistent performance across problem instances
  - Convergence reliability

- **ω(c)** ∈ [0, 1]: Efficiency score based on:
  - Quantum evaluations per second
  - Resource usage
  - Time to solution

### Mandorla Field M(t)

A dynamical field M(t) ∈ ℝ^16 that accumulates traces from benchmark results:

```
M(t+1) = Norm(α M(t) + Σᵢ βᵢ Gᵢ(t) + γ Iₜ)
```

where:
- α = 0.95: memory decay factor
- Gᵢ(t): resonant submodule contributions from different algorithm families
- γ = 0.5: injection weight
- Iₜ: seraphic feedback vector from current benchmarks

The field creates a resonance landscape that guides configuration updates.

### Double-Kick Operator T

The operator T = Φ_V ∘ Φ_U moves configurations towards fixpoints:

1. **Update kick Φ_U**: c → c + η_U ∇ψ(c)
   - Increases semantic quality
   - Explores configurations with better performance

2. **Stabilization kick Φ_V**: c' → c' + η_V R(c')
   - Improves stability ρ and efficiency ω
   - Regularizes configuration without degrading quality

By design, T is locally contractive in regions around good configurations, ensuring convergence to fixpoint attractors (Temporal Information Crystals).

### Proof-of-Resonance (PoR)

A candidate c' = T(c) is accepted only if:

1. ✓ **Quality non-decrease**: ψ(c') ≥ ψ(c)
2. ✓ **Stability consistency**: ρ(c') ≥ ρ(c) - tolerance
3. ✓ **Efficiency threshold**: ω(c') ≥ minimum
4. ✓ **Field resonance**: M(t) · I(c') > 0

This ensures every accepted move improves or maintains the system state.

### CRI Regime Switching

When the global functional J(t) = ψ_avg · ρ_avg · ω_avg stagnates, the SCS can trigger a controlled phase transition:

- Switch algorithm family (VQE ↔ QAOA)
- Switch ansatz type (Metatron → EfficientSU2 → HardwareEfficient)
- Switch optimizer (Adam → LBFGS → GradientDescent)

This prevents the system from getting trapped in locally optimal but globally suboptimal configurations.

## How to Use the SCS

### Installation

The SCS is a pure Python package with minimal dependencies:

```bash
pip install -r requirements-scs.txt
```

### Command-Line Interface

#### 1. Initialize SCS

```bash
python -m scs.cli init --benchmark-dir metatron-qso-rs/ci
```

This:
- Creates initial configuration (default or from file)
- Loads benchmark results
- Computes initial performance triplet
- Initializes Mandorla field
- Saves state to `scs_state.json`

#### 2. Run Calibration Steps

```bash
python -m scs.cli step -n 5
```

This runs 5 calibration steps, each executing:
1. Load benchmarks → 2. Update field → 3. Apply double-kick → 4. Check PoR → 5. Check CRI

Results are saved to:
- `scs_state.json`: Current state
- `scs_history.json`: Full calibration history
- `scs_best_config.json`: Current best configuration

#### 3. Check Status

```bash
python -m scs.cli status
```

Shows:
- Current configuration
- Performance triplet (ψ, ρ, ω)
- CRI diagnostics
- Calibration step count

#### 4. Export Configuration

```bash
python -m scs.cli export -o best_config.json
```

Exports the current best configuration for use in benchmarks.

### Integration with CI/Benchmarks

The SCS can be integrated into CI pipelines in two ways:

#### Option 1: GitHub Actions Workflow

The workflow `.github/workflows/scs_calibration.yml` runs automatically after benchmark workflows (if `SCS_ENABLED=true` is set as a repository variable).

To enable:
```bash
# In GitHub repository settings → Variables → Actions
# Add variable: SCS_ENABLED = true
```

#### Option 2: Manual Invocation

Add to your benchmark script:

```bash
# After running benchmarks
cargo run --release --bin quantum_walk_bench metatron-qso-rs/ci/quantum_walk_baseline.json

# Run SCS calibration
python -m scs.cli step -n 1

# Use best configuration for next benchmark
python -m scs.cli export -o current_config.json
```

### Programmatic API

```python
from scs import SeraphicCalibrator, CalibratorConfig, Configuration

# Initialize calibrator
config = CalibratorConfig(
    benchmark_dir="metatron-qso-rs/ci",
    enabled=True
)
calibrator = SeraphicCalibrator(config)
calibrator.initialize()

# Run calibration steps
for i in range(5):
    result = calibrator.calibration_step()
    print(f"Step {i+1}: J(t) = {result['j_t']:.4f}")

# Get best configuration
best = calibrator.get_best_configuration()
print(f"Best: {best.ansatz_type} depth {best.ansatz_depth}")

# Save state
calibrator.save_state()
```

## Configuration

### SCS State Files

- **scs_state.json**: Current calibrator state (config, performance, field, CRI)
- **scs_history.json**: Complete history of all calibration steps
- **scs_best_config.json**: Current best configuration

These files are preserved across calibration runs.

### Tunable Parameters

In `CalibratorConfig`:

```python
config = CalibratorConfig(
    # Paths
    benchmark_dir="metatron-qso-rs/ci",
    state_file="scs_state.json",
    history_file="scs_history.json",

    # Double-kick operator
    update_kick_step=0.3,        # η_U: quality improvement step
    stabilization_kick_step=0.2,  # η_V: stability improvement step

    # PoR criteria
    por_criteria=PoRCriteria(
        min_quality_delta=0.0,      # Minimum quality improvement
        stability_tolerance=0.1,     # Max stability degradation
        min_efficiency=0.3,          # Minimum efficiency
        min_field_resonance=0.0,     # Min field correlation
    ),

    # CRI configuration
    cri_config=ResonanceImpulseConfig(
        min_steps=10,                # Steps before CRI can trigger
        stagnation_threshold=0.01,   # J(t) variance threshold
        degradation_threshold=0.05,  # J(t) decline threshold
        min_field_resonance=0.3,     # Min field energy for CRI
    ),

    # Field dimension
    field_dimension=16,  # Dimension of M(t)

    # Enable/disable
    enabled=True,  # Master switch
)
```

### Opt-In Design

The SCS is **opt-in** and **non-intrusive**:

- Does not modify existing benchmark code
- Can be enabled/disabled via `enabled=True/False`
- Runs as a separate post-processing step
- Existing workflows continue to work without SCS

## How It Interacts with Benchmarks

### Data Flow

```
┌─────────────────────────────────────────────────┐
│  Q⊗DASH (Metatron VM) Core                      │
│  ├─ VQE, QAOA, Quantum Walks                    │
│  ├─ Grover, Boson Sampling, VQC/QML             │
│  └─ Benchmark suite (Rust)                      │
└────────────────┬────────────────────────────────┘
                 │
                 │ Produces: vqe_baseline.json
                 │           qaoa_baseline.json
                 │           quantum_walk_baseline.json
                 │           advanced_algorithms_baseline.json
                 │           vqc_baseline.json
                 │           cross_system_baseline.json
                 │           integration_baseline.json
                 ↓
┌─────────────────────────────────────────────────┐
│  Seraphic Calibration Shell (SCS)               │
│  ├─ Load benchmarks                             │
│  ├─ Compute Φ(c) = (ψ, ρ, ω)                   │
│  ├─ Update Mandorla field M(t)                  │
│  ├─ Apply T = Φ_V ∘ Φ_U                         │
│  ├─ Check PoR                                    │
│  └─ Check CRI                                    │
└────────────────┬────────────────────────────────┘
                 │
                 │ Outputs: scs_best_config.json
                 │          scs_state.json
                 │          scs_history.json
                 ↓
       ┌─────────────────────┐
       │ Next benchmark run  │
       │ (optional: use best │
       │  configuration)     │
       └─────────────────────┘
```

### Benchmark JSON Schema

The SCS reads standard benchmark JSON files. Example structure:

**VQE** (`vqe_baseline.json`):
```json
{
  "results": [
    {
      "ansatz_type": "Metatron",
      "ansatz_depth": 2,
      "optimizer": "Adam",
      "quality_score": 0.9999,
      "execution_time_ms": 382.7,
      "converged": true,
      ...
    }
  ],
  "quality_metrics": {
    "best_ground_energy": -12.999,
    "convergence_rate": 1.0
  },
  "performance_metrics": {
    "evaluations_per_second": 9453.67
  }
}
```

The SCS extracts:
- **ψ**: from `quality_score`, approximation ratios, success rates
- **ρ**: from variance of scores, convergence consistency
- **ω**: from `evaluations_per_second`, execution times

## Mathematical Foundation

The SCS implements a **fixpoint-directed dynamical system** in configuration space:

### Theorem (Fixpoint Convergence)

If D ⊂ C is a local contraction region of T with Lipschitz constant L ∈ (0, 1), then:

1. There exists unique c* ∈ D such that T(c*) = c*
2. For any c₀ ∈ D, the sequence cₜ₊₁ = T(cₜ) converges to c*
3. ||cₜ - c*|| ≤ Lᵗ ||c₀ - c*||

This guarantees that SCS calibration **always converges** to a fixpoint (Temporal Information Crystal) within each contraction region. The PoR criterion ensures we only accept moves that maintain contraction. The CRI mechanism allows jumping between regions when needed.

## Debugging and Monitoring

### View Calibration History

```python
import json

with open('scs_history.json') as f:
    history = json.load(f)

for step in history['history']:
    print(f"Step {step['step']}: J(t)={step['j_t']:.4f}, "
          f"ψ={step['performance']['psi']:.4f}")
```

### Visualize Performance Evolution

```python
import json
import matplotlib.pyplot as plt

with open('scs_history.json') as f:
    history = json.load(f)

steps = [s['step'] for s in history['history']]
psi = [s['performance']['psi'] for s in history['history']]
rho = [s['performance']['rho'] for s in history['history']]
omega = [s['performance']['omega'] for s in history['history']]

plt.plot(steps, psi, label='ψ (quality)')
plt.plot(steps, rho, label='ρ (stability)')
plt.plot(steps, omega, label='ω (efficiency)')
plt.xlabel('Calibration Step')
plt.ylabel('Performance')
plt.legend()
plt.show()
```

## FAQ

**Q: Do I need to modify existing benchmark code?**
A: No. The SCS reads standard benchmark JSON files and operates as a separate post-processing layer.

**Q: Will SCS slow down benchmarks?**
A: No. SCS runs after benchmarks complete. Calibration steps are lightweight (mainly configuration selection heuristics).

**Q: Can I disable SCS?**
A: Yes. Set `enabled=False` in `CalibratorConfig` or don't run the CLI commands. Everything works without SCS.

**Q: How do I use the best configuration?**
A: Export it with `python -m scs.cli export -o config.json` and load it in your benchmark setup.

**Q: What if SCS suggests a bad configuration?**
A: The PoR criterion ensures quality never decreases. If a candidate degrades performance, it's rejected. The CRI mechanism provides escape when trapped.

**Q: Can I customize the performance triplet computation?**
A: Yes. Edit `scs/performance.py` to adjust how ψ, ρ, ω are computed from benchmarks.

## References

- **PDF Specification**: `SeraphicCalibrationModule.pdf` (repository root)
- **SCS Package**: `scs/` directory
- **CLI**: `python -m scs.cli --help`
- **Benchmark Suite**: `BENCHMARK_SUITE_DOCUMENTATION.md`

## License

The SCS is part of the Q⊗DASH (Metatron VM) project and follows the same license.
