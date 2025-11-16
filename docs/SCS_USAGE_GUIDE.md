# SCS Usage Guide

**Seraphic Calibration Shell - Workflow & Best Practices**

## Overview

This guide walks you through practical workflows for using SCS (Seraphic Calibration Shell) as an auto-tuner for Q⊗DASH quantum algorithms.

## Installation & Setup

### Prerequisites

```bash
# Ensure Python 3.8+ is installed
python --version

# Navigate to project root
cd /path/to/qso
```

### Verify SCS Installation

```bash
# Test SCS import
python -c "import scs; print(scs.__version__)"

# Expected output: 0.1.0
```

### Create Benchmark Directory

```bash
mkdir -p benchmarks
```

## Quick Start: 5-Minute Auto-Tuning

### Step 1: Run a QAOA Algorithm

```python
import metatron_qso

# Create Metatron graph
graph = metatron_qso.MetatronGraph()

# Run QAOA with auto-calibration
result, proposal = metatron_qso.solve_maxcut_qaoa_with_tuning(
    graph=graph,
    depth=3,
    max_iters=100,
    auto_calibrate=True,
    benchmark_dir="benchmarks"
)

print(f"Cut value: {result['cut_value']:.2f}")
print(f"Approximation ratio: {result['approximation_ratio']:.3f}")

if proposal:
    print(f"\nSCS Proposal:")
    print(f"  PoR accepted: {proposal.por_accepted}")
    print(f"  CRI triggered: {proposal.cri_triggered}")
    print(f"  Suggested config: {proposal.config.to_dict()}")
```

### Step 2: Apply Suggested Configuration

```python
# If SCS accepted the new config
if proposal and proposal.por_accepted:
    new_depth = proposal.config.ansatz_depth
    new_optimizer = proposal.config.optimizer

    # Run again with new parameters
    result2, proposal2 = metatron_qso.solve_maxcut_qaoa_with_tuning(
        graph=graph,
        depth=new_depth,
        max_iters=100,
        auto_calibrate=True,
        benchmark_dir="benchmarks"
    )

    print(f"\nSecond run with SCS config:")
    print(f"  Approximation ratio: {result2['approximation_ratio']:.3f}")
```

## Workflow 1: Command-Line Auto-Tuning

### Initialize SCS

```bash
python -m scs.cli init \
    --benchmark-dir metatron-qso-rs/ci \
    --state-file scs_state.json \
    --output initial_config.json
```

**Output:**
```
Initializing Seraphic Calibration Shell...
Using default initial configuration

Initial Performance:
  ψ (quality):     0.8500
  ρ (stability):   0.8000
  ω (efficiency):  0.7200

Saved configuration to initial_config.json
```

### Run Calibration Steps

```bash
python -m scs.cli step -n 5 --output best_config.json
```

**Output:**
```
Running 5 calibration step(s)...

Step 1:
  Accepted: True
  J(t): 0.4896
  CRI triggered: False
  Performance: ψ=0.8700, ρ=0.8100, ω=0.7350

Step 2:
  Accepted: True
  J(t): 0.5324
  CRI triggered: False
  Performance: ψ=0.9000, ρ=0.8300, ω=0.7500

...

Final Performance:
  ψ (quality):     0.9200
  ρ (stability):   0.8500
  ω (efficiency):  0.7800
  Harmonic mean:   0.8445

Saved best configuration to best_config.json
```

### Check SCS Status

```bash
python -m scs.cli status
```

**Output:**
```
Seraphic Calibration Shell Status
==================================================
Step count: 5

Current configuration:
  Algorithm: VQE
  Ansatz: Metatron (depth 2)
  Optimizer: Adam (lr 0.01)

Current performance:
  ψ (quality):     0.9200
  ρ (stability):   0.8500
  ω (efficiency):  0.7800
  Harmonic mean:   0.8445

CRI diagnostics:
  Steps since impulse: 5
  Current J(t): 0.5532
  Stagnating: False
  Degrading: False
```

### Export Configuration

```bash
python -m scs.cli export -o production_config.json --stdout
```

## Workflow 2: Python API Auto-Tuning

### Using the AutoTuner Class

```python
from scs import AutoTuner, Configuration

# Create auto-tuner
tuner = AutoTuner(
    benchmark_dir="benchmarks",
    state_file="scs_state.json",
    enabled=True
)

# Initialize with default or custom config
initial_config = Configuration(
    algorithm="VQE",
    ansatz_type="Metatron",
    ansatz_depth=2,
    optimizer="Adam",
    learning_rate=0.01,
    max_iterations=100
)
tuner.initialize(initial_config)

# Run your algorithm and get metrics
# (This example assumes you have metrics from a benchmark)
metrics = {
    "psi": 0.85,
    "rho": 0.80,
    "omega": 0.72
}

# Ingest benchmark
tuner.ingest_benchmark(
    system="vqe",
    config=initial_config.to_dict(),
    metrics=metrics,
    raw_results={"ground_energy": -12.9997, "iterations": 87}
)

# Get new configuration proposal
proposal = tuner.propose_new_config()

print(f"Current Performance: ψ={proposal.current_performance.psi:.3f}")
print(f"Estimated Performance: ψ={proposal.estimated_performance.psi:.3f}")
print(f"Delta Φ: {proposal.delta_phi:.4f}")
print(f"PoR Accepted: {proposal.por_accepted}")
print(f"CRI Triggered: {proposal.cri_triggered}")

if proposal.por_accepted:
    print(f"\nNew Config:")
    print(f"  Ansatz depth: {proposal.config.ansatz_depth}")
    print(f"  Optimizer: {proposal.config.optimizer}")
    print(f"  Learning rate: {proposal.config.learning_rate}")
```

### Running Multiple Tuning Steps

```python
from scs import create_auto_tuner

# Create and initialize tuner
tuner = create_auto_tuner(benchmark_dir="benchmarks", enabled=True)

# Run 10 auto-tuning steps (or until quality > 0.9)
proposals = tuner.run_auto_tuning(num_steps=10, min_quality_threshold=0.9)

print(f"Completed {len(proposals)} tuning steps")

# Get final best configuration
best_config = tuner.get_current_config()
final_perf = tuner.get_current_performance()

print(f"\nFinal Configuration:")
print(f"  {best_config.to_dict()}")
print(f"\nFinal Performance:")
print(f"  ψ={final_perf.psi:.3f}, ρ={final_perf.rho:.3f}, ω={final_perf.omega:.3f}")
```

## Workflow 3: Integrated Quantum Algorithm Auto-Tuning

### Full QAOA Auto-Tuning Loop

```python
import metatron_qso
from scs import AutoTuner, Configuration

# Setup
graph = metatron_qso.MetatronGraph()
tuner = AutoTuner(benchmark_dir="benchmarks/qaoa", enabled=True)
tuner.initialize()

# Initial configuration
depth = 3
max_iters = 100

# Run loop: QAOA → Benchmark → SCS → New Config
for iteration in range(5):
    print(f"\n=== Iteration {iteration + 1} ===")

    # Run QAOA
    result = metatron_qso.solve_maxcut_qaoa(graph, depth, max_iters)

    # Compute metrics
    psi = result['approximation_ratio']
    rho = 0.85 if result['meta']['iterations'] < max_iters else 0.70
    omega = 1.0 - (result['meta']['iterations'] / max_iters)
    omega = max(0.3, omega)

    metrics = {"psi": psi, "rho": rho, "omega": omega}

    print(f"QAOA Result:")
    print(f"  Cut value: {result['cut_value']:.2f}")
    print(f"  Approximation ratio: {psi:.3f}")
    print(f"  Iterations: {result['meta']['iterations']}")

    # Ingest into SCS
    config = {
        "algorithm": "QAOA",
        "depth": depth,
        "optimizer": "COBYLA",
        "max_iterations": max_iters
    }

    tuner.ingest_benchmark(
        system="qaoa_maxcut",
        config=config,
        metrics=metrics,
        raw_results=result
    )

    # Get new config proposal
    proposal = tuner.propose_new_config()

    print(f"\nSCS Proposal:")
    print(f"  J(t): {proposal.j_t:.4f}")
    print(f"  PoR accepted: {proposal.por_accepted}")
    print(f"  CRI triggered: {proposal.cri_triggered}")

    # Apply changes if accepted
    if proposal.por_accepted:
        # Update depth and iterations based on SCS suggestion
        depth = proposal.config.ansatz_depth
        max_iters = proposal.config.max_iterations

        print(f"  → Applying new depth={depth}, max_iters={max_iters}")

    # Check convergence
    if proposal.current_performance.psi >= 0.95:
        print("\n✓ Converged to high quality!")
        break

print("\n=== Final Results ===")
final_config = tuner.get_current_config()
final_perf = tuner.get_current_performance()
print(f"Final config: {final_config.to_dict()}")
print(f"Final performance: ψ={final_perf.psi:.3f}, ρ={final_perf.rho:.3f}, ω={final_perf.omega:.3f}")
```

### Quantum Walk Auto-Tuning

```python
import metatron_qso

graph = metatron_qso.MetatronGraph()

# Initial parameters
t_max = 5.0
dt = 0.1

# Run with auto-tuning
for step in range(3):
    print(f"\n=== Step {step + 1} ===")

    result, proposal = metatron_qso.run_quantum_walk_with_tuning(
        graph=graph,
        source_nodes=[0],
        t_max=t_max,
        dt=dt,
        auto_tune=True,
        benchmark_dir="benchmarks/qw"
    )

    # Print results
    final_state = result['final_state']
    print(f"Quantum Walk completed:")
    print(f"  t_max: {t_max}")
    print(f"  dt: {dt}")
    print(f"  Max probability: {max(final_state):.4f}")

    if proposal and proposal.por_accepted:
        # Adjust parameters based on SCS
        suggested_config = proposal.config
        if 't_max' in suggested_config.params:
            t_max = suggested_config.params['t_max']
        if 'dt' in suggested_config.params:
            dt = suggested_config.params['dt']

        print(f"  → SCS suggests: t_max={t_max}, dt={dt}")
```

## Workflow 4: Benchmark-Driven Development

### Generate Benchmarks from Algorithm Runs

```python
from scs import write_benchmark
import metatron_qso

graph = metatron_qso.MetatronGraph()

# Run VQE
result = metatron_qso.run_vqe(graph, depth=2, max_iters=150, ansatz_type="Metatron")

# Create benchmark record
config = {
    "algorithm": "VQE",
    "ansatz_type": "Metatron",
    "ansatz_depth": 2,
    "optimizer": "Adam",
    "max_iterations": 150
}

metrics = {
    "psi": result['quality_score'],
    "rho": 0.85,
    "omega": 0.75
}

# Write to file
benchmark_path = write_benchmark(
    system="vqe",
    config=config,
    metrics=metrics,
    raw_results=result,
    aux={"hardware": "CPU", "threads": 8}
)

print(f"Benchmark saved to: {benchmark_path}")
```

### Load and Analyze Benchmarks

```python
from scs import load_benchmarks, filter_benchmarks, aggregate_benchmarks

# Load all benchmarks from directory
records = load_benchmarks("benchmarks/**/*.json")

print(f"Loaded {len(records)} benchmark records")

# Filter for high-quality VQE runs
high_quality_vqe = filter_benchmarks(
    records,
    system="vqe",
    min_psi=0.9
)

print(f"Found {len(high_quality_vqe)} high-quality VQE benchmarks")

# Aggregate statistics
stats = aggregate_benchmarks(high_quality_vqe)
print(f"\nAggregate statistics:")
print(f"  Mean ψ: {stats['metrics']['psi']['mean']:.3f}")
print(f"  Std ψ: {stats['metrics']['psi']['std']:.3f}")
```

## Workflow 5: CRI-Triggered Regime Switching

### Detecting and Handling Stagnation

```python
from scs import AutoTuner

tuner = AutoTuner(benchmark_dir="benchmarks", enabled=True)
tuner.initialize()

# Run calibration loop
for step in range(20):
    # ... run algorithm, ingest benchmark ...

    proposal = tuner.propose_new_config()

    if proposal.cri_triggered:
        print(f"\n!!! CRI TRIGGERED at step {step} !!!")
        print(f"  Switching regime:")
        print(f"    Old algorithm: {tuner.calibrator.config_space.history[-2].algorithm}")
        print(f"    New algorithm: {proposal.config.algorithm}")
        print(f"    Old ansatz: {tuner.calibrator.config_space.history[-2].ansatz_type}")
        print(f"    New ansatz: {proposal.config.ansatz_type}")

        # Regime switch has occurred - start fresh exploration
        print("  → Starting new exploration in different parameter regime")
```

## Best Practices

### 1. Benchmark Management

**DO:**
- Organize benchmarks by system type in subdirectories:
  ```
  benchmarks/
    vqe/
    qaoa/
    quantum_walk/
  ```
- Use descriptive config_ids
- Include `aux` metadata (hardware, git commit, etc.)
- Regularly clean old benchmarks

**DON'T:**
- Mix benchmarks from different problem instances in the same directory
- Manually edit benchmark JSON files
- Delete benchmarks during active auto-tuning

### 2. Performance Metrics

**Computing ψ (Quality):**
- VQE: Use approximation error or quality score
- QAOA: Use approximation ratio
- QW: Use spreading entropy or success probability

**Computing ρ (Stability):**
- Run multiple trials and use variance
- Compare results across different random seeds
- Measure robustness to parameter perturbations

**Computing ω (Efficiency):**
- Normalize by target performance (e.g., 10k evals/sec)
- Consider iteration count vs max iterations
- Factor in wallclock time

### 3. Tuning Strategy

**Start Conservative:**
```python
config = Configuration(
    algorithm="VQE",
    ansatz_type="Metatron",
    ansatz_depth=2,  # Start small
    optimizer="Adam",
    learning_rate=0.01,  # Moderate learning rate
    max_iterations=100,
    num_random_starts=1  # Single start
)
```

**Let SCS Explore:**
- Run at least 10-20 calibration steps before judging
- Allow CRI to trigger (don't stop too early)
- Track J(t) over time to monitor global progress

**Production Deployment:**
- Use the best configuration from extensive tuning
- Disable SCS in production (`enabled=False`)
- Monitor performance and re-tune periodically

### 4. Debugging

**Check SCS State:**
```python
# View field state
field = tuner.get_field_state()
print(f"Field norm: {np.linalg.norm(field.field_state)}")

# View history
history = tuner.get_calibration_history()
for step in history[-5:]:
    print(f"Step {step['step']}: J(t)={step['j_t']:.4f}, PoR={step['por_accepted']}")
```

**Validate Benchmarks:**
```python
from scs import validate_benchmark, load_benchmark

try:
    record = load_benchmark("benchmarks/vqe_run_42.json")
    print("✓ Benchmark valid")
except Exception as e:
    print(f"✗ Benchmark invalid: {e}")
```

## Troubleshooting

### Issue: PoR never accepts new configs

**Cause:** Criteria too strict or performance not improving.

**Solution:**
```python
from scs import PoRCriteria

# Relax criteria
relaxed_criteria = PoRCriteria(
    min_quality_delta=-0.05,  # Allow slight quality decrease
    stability_tolerance=0.2,   # More tolerance for stability drop
    min_efficiency=0.2         # Lower efficiency floor
)

# Apply to calibrator
tuner.calibrator.por.criteria = relaxed_criteria
```

### Issue: CRI triggers too frequently

**Cause:** Stagnation threshold too sensitive.

**Solution:**
```python
from scs import ResonanceImpulseConfig

# Less sensitive CRI
cri_config = ResonanceImpulseConfig(
    min_steps=20,  # Wait longer before allowing CRI
    stagnation_threshold=0.005,  # Stricter stagnation detection
    degradation_threshold=0.1    # Stricter degradation detection
)

tuner.calibrator.cri.config = cri_config
```

### Issue: SCS not available in Python SDK

**Cause:** Import path issue or SCS not in parent directory.

**Solution:**
```python
import sys
from pathlib import Path

# Add SCS to path
scs_path = Path("/path/to/qso")
sys.path.insert(0, str(scs_path))

# Now import
import metatron_qso
print(f"SCS Available: {metatron_qso.SCS_AVAILABLE}")
```

## Advanced Topics

### Custom Performance Triplet Computation

```python
from scs import PerformanceTriplet

def custom_compute_performance(result: dict) -> PerformanceTriplet:
    """Custom metric computation for specialized algorithm."""
    # Your custom logic here
    psi = compute_custom_quality(result)
    rho = compute_custom_stability(result)
    omega = compute_custom_efficiency(result)

    return PerformanceTriplet(psi=psi, rho=rho, omega=omega)
```

### Multi-System Aggregation

```python
from scs import compute_performance_triplet

benchmarks = {
    'vqe': load_vqe_benchmarks(),
    'qaoa': load_qaoa_benchmarks(),
    'quantum_walk': load_qw_benchmarks()
}

# Custom weights for different systems
weights = {
    'vqe': 1.5,  # Prioritize VQE
    'qaoa': 1.0,
    'quantum_walk': 0.5  # De-emphasize QW
}

aggregate_perf = compute_performance_triplet(benchmarks, algorithm_weights=weights)
```

## Summary

**Key Takeaways:**

1. **SCS is opt-in:** Enable with `auto_calibrate=True` or `auto_tune=True`
2. **Workflows:** CLI, Python API, or integrated with algorithms
3. **Benchmark → SCS → Config:** Core loop for auto-tuning
4. **PoR guards quality:** Ensures monotonic improvement
5. **CRI escapes local optima:** Switches regimes when stuck

**Next Steps:**

- Read [SCS_CORE_DESIGN.md](SCS_CORE_DESIGN.md) for architecture details
- Review [SCS_BENCHMARK_SCHEMA.md](SCS_BENCHMARK_SCHEMA.md) for schema spec
- Experiment with examples in `metatron_qso_py/examples/`
- Join discussions in GitHub Issues

---

**Version:** 0.1.0
**Last Updated:** 2025-11-16
**Author:** Q⊗DASH Project / Sebastian Klemm
