# Comprehensive Benchmark Suite Documentation

This document describes the comprehensive benchmarking infrastructure for the Metatron QSO system, including performance, quality, volume/capacity, and cross-system benchmarks.

## Overview

The benchmark suite provides comprehensive testing and validation across all major modules:

1. **Quantum Walk Benchmarks** - Original quantum walk performance metrics
2. **VQE Benchmarks** - Variational Quantum Eigensolver performance
3. **QAOA Benchmarks** - Quantum Approximate Optimization Algorithm performance
4. **VQC Benchmarks** - Variational Quantum Classifier performance
5. **Integration Benchmarks** - Cross-module compatibility and integration
6. **Cross-System Benchmarks** - Comparison against competing quantum frameworks

## Benchmark Binaries

### 1. Quantum Walk Benchmark (`quantum_walk_bench`)

**Purpose:** Measure quantum walk performance on Metatron graph structure.

**Metrics:**
- Mixing time and convergence
- Hitting time statistics
- Quantum vs classical speedup
- Mean success probability

**Usage:**
```bash
cargo run --release --bin quantum_walk_bench [output.json]
```

**Output:** `QuantumWalkBenchmarkSuite` JSON structure

---

### 2. VQE Benchmark (`vqe_bench`)

**Purpose:** Benchmark Variational Quantum Eigensolver across different ansätze.

**Metrics:**
- Ground state energy approximation
- Convergence rate
- Execution time per ansatz type
- Quantum evaluations per second
- Gradient computation performance

**Ansätze Tested:**
- Hardware-Efficient (depth 2)
- EfficientSU2 (depth 2)
- Metatron-Optimized (depth 1)

**Usage:**
```bash
cargo run --release --bin vqe_bench [output.json]
```

**Output:** `VQEBenchmarkSuite` JSON structure

**Key Quality Metrics:**
- `best_ground_energy`: Lowest energy found across all ansätze
- `convergence_rate`: Percentage of runs that converged
- `energy_variance`: Variance in energy across ansätze

---

### 3. QAOA Benchmark (`qaoa_bench`)

**Purpose:** Benchmark Quantum Approximate Optimization Algorithm on combinatorial problems.

**Metrics:**
- Approximation ratio for MaxCut problems
- Optimization quality
- Convergence characteristics
- Solution sampling statistics

**Problems Tested:**
- Triangle MaxCut (3 nodes, 3 edges)
- Square MaxCut (4 nodes, 4 edges)
- Pentagram MaxCut (5 nodes, 5 edges)

**Usage:**
```bash
cargo run --release --bin qaoa_bench [output.json]
```

**Output:** `QAOABenchmarkSuite` JSON structure

**Key Quality Metrics:**
- `best_approximation_ratio`: Best approximation quality achieved
- `avg_approximation_ratio`: Average performance across problems
- `convergence_rate`: Successful optimization rate

---

### 4. VQC Benchmark (`vqc_bench`)

**Purpose:** Benchmark Variational Quantum Classifier on binary classification tasks.

**Metrics:**
- Training accuracy
- Test accuracy
- Training loss
- Convergence rate
- Quantum evaluations

**Problems Tested:**
- Binary Classification (well-separated classes)
- Linearly Separable (with margin)

**Usage:**
```bash
cargo run --release --bin vqc_bench [output.json]
```

**Output:** `VQCBenchmarkSuite` JSON structure

**Key Quality Metrics:**
- `avg_training_accuracy`: Average training accuracy
- `avg_test_accuracy`: Generalization performance
- `convergence_rate`: Training convergence rate

---

### 5. Integration Benchmark (`integration_bench`)

**Purpose:** Test cross-module integration and compatibility.

**Metrics:**
- VQE + Metatron Hamiltonian integration
- QAOA + Graph systems integration
- Quantum Walk + Metatron graph integration
- Overall compatibility score
- Integration success rate

**Usage:**
```bash
cargo run --release --bin integration_bench [output.json]
```

**Output:** `IntegrationBenchmarkSuite` JSON structure

**Key Metrics:**
- `overall_compatibility_score`: System-wide compatibility (0-1)
- `integration_success_rate`: Percentage of successful integrations
- `avg_performance_overhead`: Average overhead from integration

---

### 6. Cross-System Benchmark (`cross_system_bench`)

**Purpose:** Compare Metatron QSO against competing quantum frameworks.

**Metrics:**
- Overall performance score
- VQE and QAOA performance
- Relative ranking
- Performance advantage/disadvantage

**Systems Compared:**
- Metatron QSO (this system)
- Qiskit VQA (IBM)
- Google Cirq
- PennyLane (Xanadu)
- ProjectQ

**Usage:**
```bash
cargo run --release --bin cross_system_bench [output.json]
```

**Output:** `CrossSystemBenchmarkSuite` JSON structure

**Key Metrics:**
- `metatron_rank`: Ranking among all systems (1-5)
- `systems_outperformed`: Number of systems outperformed
- `performance_advantage`: Relative performance gain (%)
- `quality_advantage`: Solution quality gain (%)
- `speed_advantage`: Speed gain (%)

---

### 7. Benchmark Compare (`benchmark_compare`)

**Purpose:** Compare benchmark results against baseline and detect regressions.

**Features:**
- Automatic benchmark type detection
- Threshold-based regression detection
- Detailed comparison reporting
- Fails CI on regression

**Usage:**
```bash
cargo run --release --bin benchmark_compare <baseline.json> <current.json> [threshold_percent]
```

**Arguments:**
- `baseline.json`: Baseline benchmark results
- `current.json`: Current benchmark results
- `threshold_percent`: Acceptable deviation percentage (default: 10%)

**Exit Codes:**
- 0: No regressions detected
- 1: Regressions detected (CI failure)

---

## CI/CD Integration

### Main CI Workflow (`ci.yml`)

**Trigger:** Push to main/develop, PRs
**Purpose:** Quick build validation and smoke tests

**Steps:**
1. Build all benchmark binaries
2. Run quick smoke tests
3. Upload artifacts

### Comprehensive Benchmark Workflow (`comprehensive_benchmarks.yml`)

**Trigger:** Push to main/develop/claude/**, PRs, Daily at 2 AM UTC
**Purpose:** Full benchmark suite execution

**Jobs:**
1. `quantum-walk-benchmark` - Quantum walk benchmarks
2. `vqe-benchmark` - VQE benchmarks
3. `qaoa-benchmark` - QAOA benchmarks
4. `vqc-benchmark` - VQC benchmarks
5. `integration-benchmark` - Integration tests
6. `cross-system-benchmark` - Cross-system comparison
7. `comment-pr-results` - Post results to PR

Each job:
- Runs specific benchmark
- Compares against baseline
- Uploads artifacts
- Fails on regression

**PR Comments:**
Automated PR comments include:
- All benchmark results
- Performance metrics
- Quality metrics
- Cross-system comparison
- Integration status

---

## Baseline Files

Baseline files are stored in `metatron-qso-rs/ci/`:

- `quantum_walk_baseline.json` - Quantum walk baseline
- `vqe_baseline.json` - VQE baseline
- `qaoa_baseline.json` - QAOA baseline
- `vqc_baseline.json` - VQC baseline
- `integration_baseline.json` - Integration baseline
- `cross_system_baseline.json` - Cross-system baseline

### Updating Baselines

To update a baseline after verified improvement:

```bash
# Run benchmark
cargo run --release --bin <benchmark> ci/<benchmark>_baseline.json

# Verify results
cat ci/<benchmark>_baseline.json | jq .

# Commit updated baseline
git add ci/<benchmark>_baseline.json
git commit -m "Update <benchmark> baseline"
```

---

## Performance Characteristics

### VQE Benchmarks
- **Duration:** ~1-2 seconds
- **Quantum Evaluations:** 20,000-30,000
- **Memory:** < 50MB

### QAOA Benchmarks
- **Duration:** < 1 second
- **Quantum Evaluations:** ~100-1000
- **Memory:** < 30MB

### VQC Benchmarks
- **Duration:** ~10-15 seconds
- **Quantum Evaluations:** 40,000-45,000
- **Memory:** < 50MB

### Integration Benchmarks
- **Duration:** < 1 second
- **Quantum Evaluations:** ~2,500
- **Memory:** < 30MB

### Cross-System Benchmarks
- **Duration:** < 1 second
- **Quantum Evaluations:** ~2,500
- **Memory:** < 30MB

---

## Troubleshooting

### Build Issues

```bash
# Clean build
cargo clean

# Rebuild benchmarks
cargo build --release --bins
```

### JSON Parse Errors

```bash
# Validate JSON
jq . < ci/baseline.json

# Check schema
jq 'keys' < ci/baseline.json
```

### Regression False Positives

If you get false positive regressions:
1. Check if baseline needs updating
2. Adjust threshold: `benchmark_compare baseline.json current.json 15`
3. Investigate numerical stability

### Timeout Issues

VQC benchmarks may timeout in CI:
- Default timeout: 180 seconds
- Adjust in workflow if needed
- Reduce iterations in benchmark code

---

## Future Enhancements

Planned improvements:
- [ ] Criterion.rs integration for micro-benchmarks
- [ ] Historical trend tracking
- [ ] Performance regression visualization
- [ ] Automated baseline updates on main branch
- [ ] Benchmark result database
- [ ] Comparative analysis reports
- [ ] Memory profiling integration
- [ ] GPU/hardware acceleration benchmarks

---

## Contributing

When adding new benchmarks:

1. Create benchmark binary in `src/bin/<name>_bench.rs`
2. Add to `Cargo.toml` `[[bin]]` section
3. Generate baseline: `cargo run --release --bin <name>_bench ci/<name>_baseline.json`
4. Add comparison support to `benchmark_compare.rs`
5. Add CI job to `comprehensive_benchmarks.yml`
6. Update this README
7. Submit PR with all changes

---

## References

- [VQA Implementation Guide](../VQA_IMPLEMENTATION_GUIDE.md)
- [Quantum Walk Benchmark Documentation](../BENCHMARK_QUANTUM_WALK.md)
- [Metatron QSO Documentation](../QUANTENINFORMATIONSVERARBEITUNG_DOKUMENTATION.md)

---

**Last Updated:** 2025-11-12
**Version:** 1.0.0
