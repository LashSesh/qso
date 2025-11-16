# SCS Benchmark JSON Schema

**Generic Benchmark Format for Seraphic Calibration Shell**

## Overview

The SCS Benchmark JSON Schema provides a standardized format for recording algorithm performance data. This schema enables SCS to compute performance triplets Φ(c) = (ψ, ρ, ω) and make informed auto-tuning decisions across different quantum algorithms and problem instances.

## Core Schema Structure

### BenchmarkRecord

A single benchmark record represents one execution of an algorithm with a specific configuration.

```json
{
  "system": "string",
  "config_id": "string",
  "timestamp": "ISO8601 string or Unix epoch",
  "config": {
    "algorithm": "string",
    "...": "algorithm-specific parameters"
  },
  "metrics": {
    "psi": "float [0, 1]",
    "rho": "float [0, 1]",
    "omega": "float [0, 1]"
  },
  "raw_results": {
    "...": "algorithm-specific raw data"
  },
  "aux": {
    "...": "optional auxiliary information"
  }
}
```

## Field Descriptions

### Required Fields

#### `system` (string)

Identifies the quantum algorithm or problem class.

**Standard Values:**
- `"vqe"` - Variational Quantum Eigensolver
- `"qaoa_maxcut"` - QAOA for MaxCut problem
- `"qaoa_graph_coloring"` - QAOA for graph coloring
- `"quantum_walk"` - Continuous-time quantum walk
- `"vqc"` - Variational Quantum Classifier
- `"grover"` - Grover search
- `"boson_sampling"` - Boson sampling
- `"integration"` - Cross-algorithm integration benchmark

**Custom Systems:**
Users can define custom system identifiers for domain-specific algorithms.

#### `config_id` (string)

Unique identifier for this configuration.

**Format:**
```
{algorithm}_{hash_of_config_params}_{run_id}
```

**Example:**
```
"vqe_metatron_depth2_adam_lr001_run0042"
```

#### `timestamp` (string or number)

Time when the benchmark was executed.

**Formats:**
- ISO8601: `"2025-11-16T14:30:00Z"`
- Unix epoch: `1700145000.123`

#### `config` (object)

Configuration parameters used for this run.

**Common Fields:**
```json
{
  "algorithm": "VQE",
  "ansatz_type": "Metatron",
  "ansatz_depth": 2,
  "optimizer": "Adam",
  "learning_rate": 0.01,
  "max_iterations": 100,
  "num_random_starts": 1
}
```

**Algorithm-Specific Examples:**

**VQE:**
```json
{
  "algorithm": "VQE",
  "ansatz_type": "HardwareEfficient",
  "ansatz_depth": 3,
  "optimizer": "LBFGS",
  "max_iterations": 150,
  "convergence_threshold": 1e-6
}
```

**QAOA:**
```json
{
  "algorithm": "QAOA",
  "problem": "maxcut",
  "depth": 3,
  "optimizer": "COBYLA",
  "max_iterations": 200,
  "mixer_type": "X"
}
```

**Quantum Walk:**
```json
{
  "algorithm": "QuantumWalk",
  "walk_type": "CTQW",
  "t_max": 5.0,
  "dt": 0.1,
  "source_nodes": [0],
  "krylov_dimension": 10
}
```

#### `metrics` (object)

**The performance triplet Φ(c) = (ψ, ρ, ω).**

All values must be in range [0, 1].

```json
{
  "psi": 0.85,    // Semantic quality
  "rho": 0.78,    // Stability / path invariance
  "omega": 0.65   // Efficiency / phase readiness
}
```

**Computation Guidelines:**

**ψ (Quality):**
- VQE: `1 - min(1, |E_computed - E_exact| / |E_exact|)`
- QAOA: `cut_value / optimal_cut_value`
- Quantum Walk: success probability or mixing quality
- VQC: classification accuracy

**ρ (Stability):**
- `1 - min(1, 10 × Var(results_across_runs))`
- Consistency across different random seeds
- Robustness to small parameter perturbations

**ω (Efficiency):**
- `min(1, evaluations_per_second / 10000)`
- `1 - (iteration_count / max_iterations)` if converged early
- Inverse of wall-clock time (normalized)

### Optional Fields

#### `raw_results` (object)

Algorithm-specific raw output data.

**VQE Example:**
```json
{
  "ground_energy": -12.9997,
  "exact_ground_energy": -13.0,
  "approximation_error": 0.0003,
  "num_iterations": 87,
  "final_parameters": [0.1, 0.5, ...],
  "energy_history": [-10.5, -11.2, -12.5, ...],
  "convergence_achieved": true
}
```

**QAOA Example:**
```json
{
  "cut_value": 37.5,
  "optimal_cut_value": 39.0,
  "approximation_ratio": 0.962,
  "assignment": [0, 1, 0, 1, 1, 0, ...],
  "num_iterations": 142,
  "cost_history": [-30.0, -35.0, -37.5],
  "final_parameters": {"gamma": [0.5, 0.3], "beta": [0.2, 0.4]}
}
```

**Quantum Walk Example:**
```json
{
  "final_state": [0.02, 0.15, 0.08, ...],
  "entropy": 2.456,
  "max_entropy": 2.565,
  "spreading_percentage": 95.8,
  "evolution_time": 5.0,
  "time_steps": 50,
  "wallclock_time_ms": 123.45
}
```

#### `aux` (object)

Auxiliary metadata for debugging or analysis.

**Example:**
```json
{
  "hardware": "CPU",
  "num_threads": 8,
  "graph_properties": {
    "num_nodes": 13,
    "num_edges": 78,
    "algebraic_connectivity": 6.5
  },
  "runtime_ms": 1234.56,
  "memory_mb": 128.5,
  "git_commit": "a3f9d2c",
  "notes": "High-quality run with new optimizer settings"
}
```

## Batch Benchmark Format

For multiple benchmark records (e.g., from a batch run):

```json
{
  "batch_id": "vqe_sweep_2025_11_16",
  "batch_timestamp": "2025-11-16T14:00:00Z",
  "description": "VQE parameter sweep over ansatz depths",
  "benchmarks": [
    {
      "system": "vqe",
      "config_id": "vqe_depth1_run001",
      "...": "..."
    },
    {
      "system": "vqe",
      "config_id": "vqe_depth2_run001",
      "...": "..."
    },
    ...
  ]
}
```

## Complete Examples

### Example 1: VQE Benchmark

```json
{
  "system": "vqe",
  "config_id": "vqe_metatron_d2_adam_lr001_42",
  "timestamp": "2025-11-16T14:30:00Z",
  "config": {
    "algorithm": "VQE",
    "ansatz_type": "Metatron",
    "ansatz_depth": 2,
    "optimizer": "Adam",
    "learning_rate": 0.01,
    "max_iterations": 100,
    "num_random_starts": 1
  },
  "metrics": {
    "psi": 0.9997,
    "rho": 0.85,
    "omega": 0.72
  },
  "raw_results": {
    "ground_energy": -12.9997,
    "exact_ground_energy": -13.0,
    "approximation_error": 0.0003,
    "num_iterations": 87,
    "convergence_achieved": true,
    "final_parameters": [0.123, 0.456, 0.789, 0.234],
    "energy_history": [-10.5, -11.8, -12.5, -12.85, -12.9997]
  },
  "aux": {
    "runtime_ms": 2345.67,
    "evaluations_per_second": 8543,
    "hardware": "CPU",
    "num_threads": 8
  }
}
```

### Example 2: QAOA MaxCut Benchmark

```json
{
  "system": "qaoa_maxcut",
  "config_id": "qaoa_maxcut_d3_cobyla_200_17",
  "timestamp": 1700145000.123,
  "config": {
    "algorithm": "QAOA",
    "problem": "maxcut",
    "depth": 3,
    "optimizer": "COBYLA",
    "max_iterations": 200,
    "graph": "metatron_cube"
  },
  "metrics": {
    "psi": 0.962,
    "rho": 0.88,
    "omega": 0.65
  },
  "raw_results": {
    "cut_value": 37.5,
    "optimal_cut_value": 39.0,
    "approximation_ratio": 0.962,
    "assignment": [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
    "num_iterations": 142,
    "cost_history": [-30.0, -33.5, -35.8, -37.0, -37.5],
    "final_gamma": [0.52, 0.31, 0.18],
    "final_beta": [0.24, 0.38, 0.45]
  },
  "aux": {
    "runtime_ms": 5678.90,
    "evaluations_per_second": 4231,
    "graph_properties": {
      "num_nodes": 13,
      "num_edges": 78
    }
  }
}
```

### Example 3: Quantum Walk Benchmark

```json
{
  "system": "quantum_walk",
  "config_id": "qw_ctqw_tmax5_dt01_node0_8",
  "timestamp": "2025-11-16T15:00:00Z",
  "config": {
    "algorithm": "QuantumWalk",
    "walk_type": "CTQW",
    "t_max": 5.0,
    "dt": 0.1,
    "source_nodes": [0],
    "krylov_dimension": 10
  },
  "metrics": {
    "psi": 0.92,
    "rho": 0.90,
    "omega": 0.95
  },
  "raw_results": {
    "final_state": [
      0.023, 0.156, 0.089, 0.112, 0.078, 0.134, 0.092,
      0.101, 0.067, 0.088, 0.095, 0.082, 0.083
    ],
    "entropy": 2.456,
    "max_entropy": 2.565,
    "spreading_percentage": 95.8,
    "evolution_time": 5.0,
    "time_steps": 50
  },
  "aux": {
    "runtime_ms": 234.56,
    "wallclock_time_ms": 235.12,
    "evaluations_per_second": 31941,
    "method": "krylov_expm"
  }
}
```

### Example 4: VQC Classification Benchmark

```json
{
  "system": "vqc",
  "config_id": "vqc_iris_depth2_adam_lr005_3",
  "timestamp": "2025-11-16T15:30:00Z",
  "config": {
    "algorithm": "VQC",
    "dataset": "iris",
    "ansatz_type": "EfficientSU2",
    "ansatz_depth": 2,
    "optimizer": "Adam",
    "learning_rate": 0.005,
    "max_epochs": 100,
    "batch_size": 32
  },
  "metrics": {
    "psi": 0.85,
    "rho": 0.75,
    "omega": 0.68
  },
  "raw_results": {
    "train_accuracy": 0.92,
    "test_accuracy": 0.85,
    "train_loss": 0.23,
    "test_loss": 0.35,
    "num_epochs": 78,
    "confusion_matrix": [[28, 2, 0], [1, 27, 2], [0, 1, 29]]
  },
  "aux": {
    "dataset_size": 150,
    "train_test_split": "80/20",
    "runtime_ms": 12345.67
  }
}
```

## Validation Rules

### Required Field Validation

1. `system` must be non-empty string
2. `config_id` must be non-empty string
3. `timestamp` must be valid ISO8601 or Unix epoch
4. `config` must be object with at least `algorithm` field
5. `metrics` must contain `psi`, `rho`, `omega` all in [0, 1]

### Type Validation

```python
{
  "system": str,
  "config_id": str,
  "timestamp": Union[str, float],
  "config": dict,
  "metrics": {
    "psi": float,  # range [0, 1]
    "rho": float,  # range [0, 1]
    "omega": float  # range [0, 1]
  },
  "raw_results": Optional[dict],
  "aux": Optional[dict]
}
```

### Range Validation

- All `metrics` values: `0.0 ≤ value ≤ 1.0`
- `config.ansatz_depth`: `1 ≤ value ≤ 10` (if present)
- `config.learning_rate`: `0.0 < value ≤ 1.0` (if present)
- `config.max_iterations`: `value ≥ 1` (if present)

## Usage with SCS

### Loading Benchmarks

```python
from scs.performance import BenchmarkLoader

loader = BenchmarkLoader("benchmarks/")
benchmarks = loader.load_all_benchmarks()

# Or load specific benchmark
vqe_benchmark = loader.load_vqe_benchmark("benchmarks/vqe_run_42.json")
```

### Validating Benchmarks

```python
from scs.benchmark import validate_benchmark

# Validate single record
is_valid = validate_benchmark(benchmark_record)

# Validate batch
batch = load_benchmark_batch("benchmarks/batch.json")
all_valid = all(validate_benchmark(b) for b in batch["benchmarks"])
```

### Computing Performance Triplet

```python
from scs.performance import compute_performance_triplet

benchmarks = {
    "vqe": load_vqe_benchmark(),
    "qaoa": load_qaoa_benchmark(),
    "quantum_walk": load_qw_benchmark()
}

triplet = compute_performance_triplet(benchmarks)
print(f"ψ={triplet.psi:.3f}, ρ={triplet.rho:.3f}, ω={triplet.omega:.3f}")
```

### Writing Benchmarks from Algorithm Runs

```python
import json
from datetime import datetime

def write_benchmark(system, config, metrics, raw_results=None, aux=None):
    benchmark = {
        "system": system,
        "config_id": generate_config_id(config),
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": metrics,
        "raw_results": raw_results or {},
        "aux": aux or {}
    }

    output_path = f"benchmarks/{system}_{benchmark['config_id']}.json"
    with open(output_path, 'w') as f:
        json.dump(benchmark, f, indent=2)

    return output_path

# Example usage
metrics = {"psi": 0.92, "rho": 0.85, "omega": 0.78}
config = {"algorithm": "VQE", "ansatz_type": "Metatron", "ansatz_depth": 2}
raw = {"ground_energy": -12.9997, "iterations": 87}

write_benchmark("vqe", config, metrics, raw)
```

## Best Practices

### 1. Consistent Naming

Use consistent `system` identifiers across all benchmarks:
- `"vqe"` not `"VQE"` or `"variational_quantum_eigensolver"`
- `"qaoa_maxcut"` not `"QAOA-MaxCut"` or `"qaoa_max_cut"`

### 2. Meaningful config_id

Include key parameters in `config_id` for easy identification:
```
{algorithm}_{ansatz}_{depth}_{optimizer}_{lr}_{run_number}
```

### 3. Preserve Raw Results

Always include `raw_results` for debugging and detailed analysis.

### 4. Timestamp Precision

Use ISO8601 with timezone (UTC preferred) or high-precision Unix epoch.

### 5. Auxiliary Metadata

Include runtime environment info in `aux` for reproducibility:
- Hardware type (CPU/GPU)
- Number of threads
- Software versions
- Git commit hash

## Integration with CI/CD

Benchmarks can be automatically generated and validated in CI pipelines:

```yaml
# .github/workflows/benchmark.yml
- name: Run Benchmarks
  run: |
    python run_benchmarks.py --output benchmarks/
    python validate_benchmarks.py benchmarks/*.json
```

## Versioning

**Schema Version:** 1.0.0

Future versions may add:
- Multi-objective metrics beyond (ψ, ρ, ω)
- Uncertainty quantification fields
- Distributed/parallel execution metadata
- Hardware-specific performance counters

---

**Version:** 1.0.0
**Last Updated:** 2025-11-16
**Author:** Sebastian Klemm / Q⊗DASH Project
