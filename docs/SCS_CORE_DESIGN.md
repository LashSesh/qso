# SCS Core Design

**Seraphic Calibration Shell (SCS) for Q⊗DASH (Metatron VM)**

## Overview

The Seraphic Calibration Shell (SCS) is a meta-model for fixpoint-directed quantum-hybrid optimization that wraps the Q⊗DASH core with field-theoretic feedback and contraction dynamics. SCS acts as a **generic auto-tuner** for quantum algorithms, automatically optimizing hyperparameters and configuration choices based on benchmark performance.

## Architecture

### Core Components

```
┌──────────────────────────────────────────────────────────────┐
│                  Seraphic Calibrator                         │
│  (Main orchestrator: SeraphicCalibrator)                     │
└───────────────┬──────────────────────────────────────────────┘
                │
    ┌───────────┴───────────────────────────────────┐
    │                                               │
    ▼                                               ▼
┌─────────────────┐                        ┌──────────────────┐
│  Configuration  │                        │  Benchmark Data  │
│     Space       │                        │   (JSON files)   │
│   (config.py)   │                        │ (performance.py) │
└────────┬────────┘                        └─────────┬────────┘
         │                                           │
         │                                           │
         └────────────┬──────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │  Performance Triplet Φ(c)  │
         │    (ψ, ρ, ω)               │
         │  - ψ: semantic quality     │
         │  - ρ: stability            │
         │  - ω: efficiency           │
         └────────────┬───────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
         ▼                         ▼
┌──────────────────┐      ┌───────────────────┐
│  Mandorla Field  │      │ Seraphic Feedback │
│      M(t)        │◄─────│    Encoder g_SFM  │
│   (field.py)     │      │    (field.py)     │
└────────┬─────────┘      └───────────────────┘
         │
         │
         ▼
┌──────────────────────────────────────────────┐
│          Double-Kick Operator T               │
│          T = Φ_V ∘ Φ_U                        │
│  - Φ_U: Update kick (quality ↑)              │
│  - Φ_V: Stabilization kick (stability ↑)     │
│           (operators.py)                      │
└────────────────┬─────────────────────────────┘
                 │
                 ▼
         ┌──────────────────┐
         │ Candidate Config  │
         │       c'          │
         └───────┬───────────┘
                 │
                 ▼
┌────────────────────────────────────────────┐
│      Proof-of-Resonance (PoR)              │
│  - Quality non-decrease: ψ(c') ≥ ψ(c)     │
│  - Stability consistency: ρ(c') ≥ ρ(c)-ε  │
│  - Efficiency threshold: ω(c') ≥ ω_min    │
│  - Field resonance: M(t)·I(c') > 0        │
│           (por.py)                         │
└──────────────┬─────────────────────────────┘
               │
          Accept/Reject
               │
               ▼
    ┌──────────────────────┐
    │  CRI Check           │
    │  J(t) = ψ·ρ·ω       │
    │  - Stagnation?      │
    │  - Degradation?     │
    │  → Regime Switch    │
    │    (cri.py)         │
    └─────────────────────┘
```

## Data Flow

### 1. Benchmark → Performance Triplet

```python
Benchmarks (JSON) → compute_performance_triplet() → Φ(c) = (ψ, ρ, ω)
```

**Inputs:**
- Benchmark JSON files (e.g., `vqe_baseline.json`, `qaoa_baseline.json`)
- Each contains: quality metrics, performance metrics, algorithm results

**Processing:**
- **ψ (quality)**: Computed from approximation ratios, ground state errors, convergence rates
- **ρ (stability)**: Variance across runs, robustness to parameter changes
- **ω (efficiency)**: Evaluations per second, resource usage, iteration counts

**Output:**
- `PerformanceTriplet(psi, rho, omega)` where each component ∈ [0, 1]

### 2. Seraphic Feedback → Field Update

```python
Φ(c) → g_SFM (Seraphic Encoder) → I_t (injection) → M(t+1)
```

**Encoding (g_SFM):**
```python
I_t[0:3] = [ψ, ρ, ω]                    # Direct triplet
I_t[3:6] = [harmonic_mean, geometric_mean, norm]
I_t[6:9] = [ψ·ρ, ψ·ω, ψ·ρ·ω]           # Product features
I_t[9:16] = harmonic_structure(ψ, ρ, ω) # Resonance patterns
```

**Field Update (M(t)):**
```python
M(t+1) = Norm(α·M(t) + Σ_i β_i·G_i(t) + γ·I_t)
```

Where:
- `α = 0.95`: Memory decay factor
- `β_i`: Submodule weights for different algorithms (VQE, QAOA, QW)
- `γ = 0.5`: Injection weight
- `G_i(t)`: Resonant submodule states (algorithm-specific accumulators)

### 3. Double-Kick Operator

**Update Kick Φ_U (quality improvement):**
```python
c_intermediate = Φ_U(c) = c + η_U · ∇_c ψ(c)
```

- Generates 8 neighbor configurations
- Heuristically estimates quality improvement
- Selects best neighbor based on quality score

**Stabilization Kick Φ_V (stability & efficiency improvement):**
```python
c' = Φ_V(c_intermediate) = c_intermediate + η_V · R(c)
```

- Generates 8 more neighbors
- Optimizes for `0.6·ρ + 0.4·ω` (weighted stability/efficiency)
- Ensures quality is not degraded

**Result:**
```python
c' = T(c) = Φ_V(Φ_U(c))
```

### 4. Proof-of-Resonance (PoR) Decision

**Acceptance Criteria:**

1. **Quality non-decrease:** `ψ(c') ≥ ψ(c) + δ` (δ ≥ 0)
2. **Stability consistency:** `ρ(c') ≥ ρ(c) - ε` (tolerance ε ~ 0.1)
3. **Efficiency threshold:** `ω(c') ≥ ω_min` (ω_min ~ 0.3)
4. **Field resonance:** `⟨M(t), I(c')⟩ ≥ threshold` (threshold ≥ 0)

**Decision:**
```python
if PoR.check(c, Φ(c), c', Φ(c')):
    accept c' → new current configuration
else:
    reject c' → keep c
```

### 5. CRI (Calibration Regime Initialization) Check

**Global Calibration Functional:**
```python
J(t) = ψ(t) · ρ(t) · ω(t)
```

**Stagnation Detection:**
- Track `J(t)` over window (e.g., last 5-10 steps)
- If `Var(J[t-w:t]) < threshold` → stagnating
- If `mean(J[t-w:t]) < mean(J[t-2w:t-w])` → degrading

**Regime Switch (when stagnating/degrading):**
```python
if stagnating or degrading:
    c_new = switch_regime(c_current, field_state)
    # E.g., VQE → QAOA, Metatron → EfficientSU2, Adam → LBFGS
```

**Strategy:**
1. Switch algorithm family (VQE ↔ QAOA ↔ QuantumWalk)
2. Switch ansatz type (Metatron → EfficientSU2 → HardwareEfficient)
3. Switch optimizer (Adam → LBFGS → GradientDescent)

## Configuration Space

### Representation

A configuration `c ∈ C` is a point in hyperparameter space:

```python
c = Configuration(
    algorithm: str,          # "VQE" | "QAOA" | "QuantumWalk" | ...
    ansatz_type: str,        # "Metatron" | "EfficientSU2" | "HardwareEfficient"
    ansatz_depth: int,       # 1-10
    optimizer: str,          # "Adam" | "LBFGS" | "GradientDescent" | "COBYLA"
    learning_rate: float,    # 0.001 - 0.1
    max_iterations: int,     # 10 - 500
    num_random_starts: int,  # 1 - 5
    # ... additional parameters
)
```

### Neighbor Generation

For local exploration, neighbors are generated by:
- Perturbing one parameter at a time
- Discrete parameters: random choice from allowed values
- Continuous parameters: multiplicative perturbation (×0.8 to ×1.2) or additive (±1)

### Distance Metric

```python
d(c1, c2) = Σ discrete_diffs + Σ normalized_continuous_diffs
```

## Performance Triplet Components

### ψ (Semantic Quality)

**For VQE:**
```python
ψ = quality_score = 1 - |E_computed - E_exact| / |E_exact|
```

**For QAOA:**
```python
ψ = approximation_ratio = cut_value / optimal_cut_value
```

**For Quantum Walk:**
```python
ψ = success_probability or convergence_quality
```

### ρ (Stability / Path Invariance)

**Variance-based:**
```python
ρ = 1 - min(1, 10 × Var(quality_scores_across_runs))
```

**Cross-algorithm consistency:**
```python
ρ = 1 - |quality_VQE - quality_QAOA|
```

### ω (Phase Readiness / Efficiency)

**Computational efficiency:**
```python
ω = min(1, evaluations_per_second / target_eps)
```
where `target_eps = 10,000` for normalization.

**Resource usage:**
```python
ω = 1 - (iteration_count / max_iterations)
```

## MandorlaField Properties

### Field State M(t)

- **Dimension:** 16 (configurable)
- **Norm:** L2-normalized to 1 (lives on unit hypersphere)
- **Dynamics:** Leaky integrator with submodule resonance

### Resonance Computation

```python
resonance = ⟨M(t), I(c')⟩ = Σ_i M_i(t) · I_i(c') / (||M|| · ||I||)
```

- Resonance ∈ [-1, 1]
- Positive resonance indicates alignment with field patterns
- Negative resonance indicates anti-alignment

### Submodules G_i(t)

Three submodules for different algorithm families:
- `G_0`: VQE, Grover
- `G_1`: QAOA, Boson Sampling
- `G_2`: Quantum Walk, VQC

Each submodule accumulates algorithm-specific resonance patterns.

## Fixpoint Dynamics

### Local Contractivity

The double-kick operator `T = Φ_V ∘ Φ_U` is designed to be **locally contractive**:

```python
||T(c_i+1) - T(c_i)|| ≤ κ · ||c_i+1 - c_i||
```

where `κ < 1` is the Lipschitz constant.

**Convergence:**
- Iterating `c_{n+1} = T(c_n)` converges to fixpoint `c*` where `T(c*) = c*`
- Fixpoint represents optimal configuration in current calibration regime

### Global Escape via CRI

When stuck in local fixpoint (stagnation/degradation):
- CRI triggers regime switch
- Jump to new region of configuration space
- Begin new fixpoint search in different attractor basin

## Integration Points

### 1. Quantum Walk Toolkit

**Benchmark Generation:**
```python
result = run_quantum_walk(graph, source_nodes, t_max, dt)
benchmark_json = {
    "system": "quantum_walk",
    "config": {"t_max": t_max, "dt": dt, "sources": source_nodes},
    "metrics": {
        "psi": spreading_quality,
        "rho": consistency_score,
        "omega": evaluations_per_second / 10000.0
    },
    "timestamp": current_time
}
```

**Auto-Tuning Hook:**
```python
if auto_tune:
    write_benchmark(benchmark_json)
    scs.calibration_step()
    new_config = scs.get_best_configuration()
    # Apply new_config for next run
```

### 2. QAOA Optimizer

**Benchmark Generation:**
```python
result = solve_maxcut_qaoa(graph, depth, max_iters)
benchmark_json = {
    "system": "qaoa_maxcut",
    "config": {"depth": depth, "max_iters": max_iters, "optimizer": optimizer},
    "metrics": {
        "psi": approximation_ratio,
        "rho": stability_score,
        "omega": efficiency_score
    }
}
```

**Auto-Calibrate Parameter:**
```python
def solve_maxcut_qaoa(graph, depth, max_iters, auto_calibrate=False):
    result = run_qaoa_optimizer(...)

    if auto_calibrate:
        write_benchmark(result_to_benchmark(result))
        scs.calibration_step()
        new_config = scs.get_best_configuration()
        return result, new_config

    return result
```

## State Persistence

### State File Format

```json
{
  "step_count": 42,
  "current_config": { ... },
  "current_performance": {"psi": 0.85, "rho": 0.78, "omega": 0.65},
  "field": {
    "dimension": 16,
    "field_state": [0.12, 0.34, ...],
    "alpha": 0.95,
    "gamma": 0.5,
    "beta_weights": [0.1, 0.1, 0.1],
    "submodule_states": [...]
  },
  "cri_diagnostics": {
    "steps_since_impulse": 8,
    "current_j_t": 0.435,
    "j_t_history": [0.42, 0.43, 0.435, ...]
  }
}
```

### History File Format

```json
{
  "history": [
    {
      "step": 1,
      "config": { ... },
      "performance": {"psi": 0.75, "rho": 0.70, "omega": 0.60},
      "j_t": 0.315,
      "por_accepted": true,
      "cri_triggered": false,
      "timestamp": 1709123456.789
    },
    ...
  ]
}
```

## Algorithmic Guarantees

### Monotonic Quality Improvement (with PoR)

With PoR acceptance criterion `ψ(c') ≥ ψ(c)`:
```
ψ(c_0) ≤ ψ(c_1) ≤ ψ(c_2) ≤ ... → ψ*
```

### Bounded Exploration (without CRI)

Without regime switches, configurations remain in local neighborhood:
```
||c_n - c_0|| ≤ Σ_{i=0}^{n-1} ||c_{i+1} - c_i|| ≤ geometric_sum → finite
```

### Global Exploration (with CRI)

CRI enables escape from local optima:
```
At stagnation: jump to new regime → explore new attractor basin
```

## Performance Characteristics

### Computational Complexity

- **Per calibration step:** O(k) where k = number of neighbor configurations (typically 8-16)
- **Field update:** O(m) where m = field dimension (typically 16)
- **PoR check:** O(m) for field resonance computation
- **CRI check:** O(w) where w = history window size (typically 5-10)

**Total per step:** O(k + m + w) ~ O(10-50) lightweight operations

### Memory Usage

- **State file:** ~10-50 KB (field state + config + diagnostics)
- **History file:** ~1-10 MB for 1000 steps (depending on detail level)
- **In-memory:** ~1-5 MB (field state, history buffers, current config)

### Convergence Rate

Empirically:
- **Typical fixpoint convergence:** 5-20 steps
- **CRI trigger frequency:** Every 10-30 steps (when stagnating)
- **Global optimum approach:** 50-200 total steps across multiple regimes

## References

**Mathematical Foundations:**
- Fixpoint theory and Banach contraction mapping theorem
- Field-theoretic optimization (inspiration from quantum field theory)
- Resonance-based feedback (inspired by coupled oscillator systems)

**Quantum Algorithm Benchmarking:**
- VQE quality metrics (ground state approximation error)
- QAOA approximation ratios (Goemans-Williamson bound)
- Quantum walk mixing times and spreading

**Meta-Learning & Auto-ML:**
- Hyperparameter optimization (Bayesian optimization, genetic algorithms)
- Configuration space search (SMAC, Optuna)
- Performance modeling (surrogate models)

---

**Version:** 0.1.0
**Last Updated:** 2025-11-16
**Author:** Sebastian Klemm / Q⊗DASH Project
