# Benchmark Policy - Q⊗DASH Quantum Benchmarks

## Purpose

This document defines the benchmark policy for Q⊗DASH, including:
- Which benchmarks exist and what they measure
- When benchmarks are executed
- Acceptable performance thresholds and regression criteria
- How benchmark failures are handled

---

## Benchmark Inventory

### 1. Quantum Walk Benchmark

**Binary:** `quantum_walk_bench`
**Baseline:** `metatron-qso-rs/ci/quantum_walk_baseline.json`
**Execution Time:** ~5-10 minutes (comprehensive)

**Metrics Tracked:**

| Metric | Description | Baseline Target | Regression Threshold |
|--------|-------------|----------------|---------------------|
| `mixing_time.mixing_time` | Steps to convergence | < 100 steps | +20% |
| `hitting_time.quantum_average_steps` | Avg steps to target (quantum) | < 50 steps | +15% |
| `hitting_time.classical_average_steps` | Avg steps to target (classical) | > 100 steps | -10% (worse) |
| `hitting_time.speedup_factor` | Quantum speedup ratio | > 2.0x | -15% |
| `hitting_time.mean_success_probability` | Success rate | > 0.80 | -10% |

**Failure Criteria:**
- Speedup factor drops below 1.7x (< 15% regression)
- Success probability drops below 0.72 (< 10% regression)
- Quantum average steps increases by > 20%

---

### 2. VQE (Variational Quantum Eigensolver) Benchmark

**Binary:** `vqe_bench`
**Baseline:** `metatron-qso-rs/ci/vqe_baseline.json`
**Execution Time:** ~1-2 seconds

**Metrics Tracked:**

| Metric | Description | Baseline Target | Regression Threshold |
|--------|-------------|----------------|---------------------|
| `quality_metrics.best_ground_energy` | Lowest energy found | Close to -13.0 | +2% error |
| `quality_metrics.convergence_rate` | % of runs converged | 100% | < 90% |
| `performance_metrics.evaluations_per_second` | Throughput | > 9000 evals/sec | -15% |
| `quality_metrics.avg_ground_energy` | Average energy quality | < -9.0 | +10% |
| `performance_metrics.total_execution_time_ms` | Total runtime | < 5000 ms | +20% |

**Key Quality Indicators:**
- **Best Ground Energy:** Must be within 0.1% of classical ground state (-13.0)
- **Convergence Rate:** All ansatz types must converge (100%)
- **Metatron Ansatz Advantage:** Metatron ansatz should achieve quality_score > 0.999

**Failure Criteria:**
- Best ground energy regresses by > 2%
- Convergence rate drops below 90%
- Evaluations per second drops by > 15%
- Any ansatz fails to converge

---

### 3. QAOA (Quantum Approximate Optimization) Benchmark

**Binary:** `qaoa_bench`
**Baseline:** `metatron-qso-rs/ci/qaoa_baseline.json`
**Execution Time:** < 1 second

**Metrics Tracked:**

| Metric | Description | Baseline Target | Regression Threshold |
|--------|-------------|----------------|---------------------|
| `quality_metrics.best_approximation_ratio` | Best solution quality | > 0.95 | -5% |
| `quality_metrics.avg_approximation_ratio` | Average solution quality | > 0.85 | -10% |
| `quality_metrics.convergence_rate` | % of successful runs | 100% | < 90% |
| `performance_metrics.total_execution_time_ms` | Total runtime | < 1000 ms | +25% |
| `performance_metrics.evaluations_per_second` | Throughput | > 1000 evals/sec | -15% |

**Problem-Specific Targets:**
- **Triangle MaxCut:** Approximation ratio > 0.95
- **Square MaxCut:** Approximation ratio > 0.90
- **Pentagram MaxCut:** Approximation ratio > 0.85

**Failure Criteria:**
- Best approximation ratio drops below 0.90
- Any problem instance fails to find a valid solution
- Convergence rate drops below 90%

---

### 4. VQC (Variational Quantum Classifier) Benchmark

**Binary:** `vqc_bench`
**Baseline:** `metatron-qso-rs/ci/vqc_baseline.json`
**Execution Time:** ~10-15 seconds

**Metrics Tracked:**

| Metric | Description | Baseline Target | Regression Threshold |
|--------|-------------|----------------|---------------------|
| `quality_metrics.avg_training_accuracy` | Training accuracy | > 0.95 | -5% |
| `quality_metrics.avg_test_accuracy` | Test accuracy (generalization) | > 0.90 | -10% |
| `quality_metrics.convergence_rate` | % of trainings converged | 100% | < 90% |
| `performance_metrics.total_execution_time_ms` | Total runtime | < 20000 ms | +25% |
| `performance_metrics.evaluations_per_second` | Throughput | > 2000 evals/sec | -15% |

**Overfitting Check:**
- `avg_training_accuracy - avg_test_accuracy` should be < 0.10 (10% gap)

**Failure Criteria:**
- Test accuracy drops below 0.81 (> 10% regression)
- Training accuracy drops below 0.90
- Overfitting gap exceeds 0.15 (15%)
- Convergence rate drops below 90%

---

### 5. Integration Benchmark

**Binary:** `integration_bench`
**Baseline:** `metatron-qso-rs/ci/integration_baseline.json`
**Execution Time:** < 1 second

**Metrics Tracked:**

| Metric | Description | Baseline Target | Regression Threshold |
|--------|-------------|----------------|---------------------|
| `cross_module_metrics.overall_compatibility_score` | System compatibility | > 0.90 | -10% |
| `cross_module_metrics.integration_success_rate` | % successful integrations | 100% | < 100% |
| `cross_module_metrics.total_execution_time_ms` | Total runtime | < 1000 ms | +25% |
| `cross_module_metrics.avg_performance_overhead` | Integration overhead | < 0.15 | +20% |

**Integration Tests:**
1. **VQE + Metatron Hamiltonian:** Must execute without errors
2. **QAOA + Graph Systems:** Must successfully solve MaxCut
3. **Quantum Walk + Metatron Graph:** Must converge

**Failure Criteria:**
- Any integration test fails (success_rate < 100%)
- Overall compatibility score drops below 0.81
- Performance overhead exceeds 0.18 (18%)

---

### 6. Cross-System Comparison Benchmark

**Binary:** `cross_system_bench`
**Baseline:** `metatron-qso-rs/ci/cross_system_baseline.json`
**Execution Time:** < 1 second

**Metrics Tracked:**

| Metric | Description | Baseline Target | Regression Threshold |
|--------|-------------|----------------|---------------------|
| `comparison_metrics.metatron_rank` | Ranking (1-5, 1=best) | Rank 1-2 | Rank > 2 |
| `comparison_metrics.systems_outperformed` | # of systems beaten | ≥ 3 out of 4 | < 2 |
| `comparison_metrics.performance_advantage` | Relative performance gain | > 0% | < -10% |
| `comparison_metrics.quality_advantage` | Solution quality gain | > 0% | < -5% |
| `comparison_metrics.speed_advantage` | Speed gain | > 0% | < -10% |

**Competitive Targets:**
- **Metatron QSO** should rank in top 2 among all systems
- Should outperform at least 3 out of 4 competitors
- Positive advantage in performance, quality, OR speed

**Failure Criteria:**
- Metatron ranks worse than 2nd place
- Outperforms fewer than 2 competitors
- All advantages (performance, quality, speed) are negative

---

### 7. Advanced Algorithms Benchmark

**Binary:** `advanced_algorithms_bench`
**Baseline:** `metatron-qso-rs/ci/advanced_algorithms_baseline.json`
**Execution Time:** ~2-5 seconds

**Metrics Tracked:**

| Metric | Description | Baseline Target | Regression Threshold |
|--------|-------------|----------------|---------------------|
| `grover_search.success_probability` | Grover search success | > 0.95 | -10% |
| `grover_search.speedup` | Speedup over classical | > 1.5x | -15% |
| `multi_target_grover.success_probability` | Multi-target success | > 0.90 | -10% |
| `boson_sampling.interference_visibility` | Boson interference quality | > 0.80 | -15% |
| `quantum_ml.test_accuracy` | Quantum ML accuracy | > 0.85 | -10% |
| `performance_metrics.total_execution_time_ms` | Total runtime | < 10000 ms | +25% |

**Failure Criteria:**
- Grover success probability drops below 0.855 (> 10% regression)
- Any algorithm fails to execute successfully
- Total execution time exceeds 12500 ms

---

## Benchmark Execution Schedule

### Automatic Triggers

| When | Which Benchmarks | Purpose |
|------|------------------|---------|
| **Every PR/Push (Core CI)** | NONE | Core CI is lightweight; no heavy benchmarks |
| **Nightly (2 AM UTC)** | ALL quantum benchmarks | Track performance trends, detect regressions |
| **Weekly (Sunday 3 AM UTC)** | SCS Calibration | Auto-tune SCS parameters |
| **Release Tags (`v*.*.*`)** | ALL quantum benchmarks | Final validation before release |
| **Manual (`workflow_dispatch`)** | Selected benchmark(s) | Performance-critical PRs, debugging |

### Manual Execution

Maintainers can trigger benchmarks manually via GitHub Actions:

1. Navigate to **Actions** → **Quantum Benchmarks**
2. Click **Run workflow**
3. Select branch
4. (Optional) Check "Update baselines" if improvement is verified

---

## Regression Handling

### Detection

Regressions are detected by `benchmark_compare` binary, which compares current results against baselines using defined thresholds.

**Comparison Logic:**
```
For each metric:
  baseline_value = baseline[metric]
  current_value = current[metric]

  # For "higher is better" metrics (e.g., speedup, accuracy)
  regression = (current_value < baseline_value * (1 - threshold))

  # For "lower is better" metrics (e.g., execution time, error)
  regression = (current_value > baseline_value * (1 + threshold))
```

### Response Protocol

When a benchmark fails due to regression:

#### 1. **Investigate** (< 24 hours)
- Review PR changes that may have caused regression
- Check if regression is real or baseline outdated
- Run benchmark locally to reproduce

#### 2. **Categorize**

**A. False Positive (Baseline Outdated)**
- Previous improvement not captured in baseline
- **Action:** Update baseline with verified improvement
- **Command:**
  ```bash
  cp target/<benchmark>_bench.json ci/<benchmark>_baseline.json
  git add ci/<benchmark>_baseline.json
  git commit -m "Update <benchmark> baseline after verified improvement"
  ```

**B. Acceptable Trade-off**
- Minor regression in exchange for major improvement elsewhere
- **Action:** Document in PR, adjust threshold if needed, update baseline
- **Example:** "5% slower but 20% more accurate"

**C. Real Regression (Bug or Performance Issue)**
- **Action:**
  - Mark PR as "needs work"
  - Investigate root cause
  - Fix performance issue
  - Re-run benchmarks
  - Merge only after benchmarks pass

#### 3. **Document**
- Add entry to `CHANGELOG.md` if baseline updated
- Note performance changes in PR description
- Update this policy if thresholds need adjustment

---

## Threshold Adjustment Policy

Thresholds are **not arbitrary** - they reflect acceptable engineering trade-offs:

- **Quality metrics:** Strict thresholds (5-10%) - solution quality is critical
- **Performance metrics:** Moderate thresholds (15-20%) - some variance expected
- **Timing metrics:** Relaxed thresholds (20-25%) - hardware/CI variance

**When to Adjust Thresholds:**
- Persistent false positives across multiple PRs
- Algorithm change fundamentally alters performance profile
- Hardware/CI environment changes

**Process:**
1. Open issue documenting proposed threshold change
2. Provide rationale and supporting data
3. Get approval from 2+ maintainers
4. Update this document with new thresholds
5. Commit updated policy

---

## Baseline Update Policy

**When to Update Baselines:**

✅ **DO UPDATE** when:
- Verified algorithmic improvement (higher quality, faster execution)
- New feature intentionally changes metrics (e.g., added ansatz)
- Threshold adjustment approved

❌ **DO NOT UPDATE** when:
- "Fixing" regression without understanding root cause
- Making tests pass without investigation
- Hiding performance degradation

**Process:**
1. Run benchmark: `cargo run --release --bin <benchmark> ci/<benchmark>_baseline.json`
2. Verify improvement: `cargo run --release --bin benchmark_compare OLD_baseline.json ci/<benchmark>_baseline.json`
3. Document in commit message: What improved, why, by how much
4. Submit PR with baseline update
5. Get review from maintainer

---

## Continuous Monitoring

### Nightly Benchmark Runs

Benchmarks run nightly to:
- Track performance trends over time
- Detect gradual degradation
- Validate baseline accuracy

Results are uploaded as artifacts (30-day retention).

### Performance Trend Tracking (Future)

Planned enhancements:
- Historical performance database
- Trend visualization dashboard
- Automated alerts for gradual degradation
- Comparative analysis reports

---

## Benchmark Artifacts

All benchmark runs produce JSON artifacts:

**Retention:**
- **30 days:** Normal benchmark results
- **90 days:** Regression failure reports
- **Permanent:** Release tag benchmarks (via GitHub Releases)

**Accessing Artifacts:**
1. Navigate to **Actions** → Failed/successful workflow
2. Scroll to **Artifacts** section
3. Download `<benchmark-name>-benchmark.zip`
4. Extract and review JSON results

---

## SCS Calibration Policy

SCS (Seraphic Calibration Shell) calibration is a **special case** benchmark:

**Purpose:** Auto-tune SCS parameters (ψ, ρ, ω) for optimal performance

**Execution:**
- **Weekly:** Sunday 3 AM UTC (automated)
- **Manual:** On-demand via `workflow_dispatch`

**Metrics:**
- ψ (Psi): Quantum state coherence
- ρ (Rho): Density matrix fidelity
- ω (Omega): Oscillation frequency

**Artifacts:**
- `scs_state.json`: Current calibration state
- `scs_history.json`: Historical calibration log
- `scs_best_config.json`: Optimal configuration

**Failure Handling:**
- SCS calibration failures do NOT block PRs
- Failures indicate calibration drift, not code regression
- Manual intervention may be needed to reset state

**Optional State Commit:**
- If `vars.SCS_COMMIT_STATE == 'true'`, SCS state auto-commits to repo
- Use with caution - can create commit noise

---

## Developer Quick Reference

### Before Merging a PR

```bash
# 1. Run relevant benchmarks locally (if performance-critical)
cd metatron-qso-rs
cargo run --release --bin <benchmark> /tmp/<benchmark>_test.json

# 2. Compare against baseline
cargo run --release --bin benchmark_compare ci/<benchmark>_baseline.json /tmp/<benchmark>_test.json

# 3. If regression detected, investigate before merging
```

### Manually Triggering Benchmarks

1. **GitHub UI:** Actions → Quantum Benchmarks → Run workflow
2. **CLI (gh):** `gh workflow run benchmarks.yml`

### Updating a Baseline

```bash
# After verified improvement
cp target/<benchmark>_bench.json ci/<benchmark>_baseline.json
git add ci/<benchmark>_baseline.json
git commit -m "Update <benchmark> baseline: [reason]"
```

---

## Related Documentation

- **CI Pipeline Overview:** [`CI_PIPELINE_OVERVIEW.md`](./CI_PIPELINE_OVERVIEW.md)
- **Benchmark Suite Docs:** [`../BENCHMARK_SUITE_DOCUMENTATION.md`](../BENCHMARK_SUITE_DOCUMENTATION.md)
- **VQA Implementation:** [`../VQA_IMPLEMENTATION_GUIDE.md`](../VQA_IMPLEMENTATION_GUIDE.md)
- **Quantum Walk Benchmarks:** [`../BENCHMARK_QUANTUM_WALK.md`](../BENCHMARK_QUANTUM_WALK.md)

---

**Last Updated:** 2025-11-16
**Version:** 1.0.0
**Maintained By:** CI/CD Team
