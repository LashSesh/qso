# How to Run the Q⊗DASH Core — Step-by-Step Guide

This guide provides **step-by-step instructions** for running the Metatron Quantum State Operator (QSO) Core and the Seraphic Calibration Shell (SCS).

## Prerequisites

Before proceeding, ensure you have completed the setup in [`DEV_SETUP.md`](DEV_SETUP.md):

- ✅ Rust 1.85.0+ installed
- ✅ Python 3.8+ installed
- ✅ Repository cloned and dependencies installed

---

## Part 1: Building and Running the QSO Core

### Step 1: Navigate to the QSO Core Directory

```bash
cd metatron-qso-rs
```

### Step 2: Build the QSO Core

**Option A: Development Build (faster compilation, slower execution)**

```bash
cargo build
```

**Option B: Release Build (optimized for performance, recommended)**

```bash
cargo build --release
```

**Expected output:**
```
   Compiling metatron-qso-rs v0.1.0 (/path/to/qdash/metatron-qso-rs)
    Finished `release` profile [optimized] target(s) in 18.43s
```

### Step 3: Run Your First Quantum Walk Demo

The quantum walk benchmark demonstrates quantum probability distribution evolution on the 13-node Metatron graph.

```bash
cargo run --release --bin quantum_walk_bench
```

**What this does:**
- Initializes a quantum state on the central node (node 0)
- Evolves the state through continuous-time quantum walk
- Measures probability distribution across all 13 nodes
- Compares performance metrics against baselines

**Expected output (sample):**
```
=== Metatron Quantum Walk Benchmark ===

Initial state: |0⟩ (center node)
Evolution time: t = 1.0

Probability distribution:
  Node  0 (Center):    0.2456
  Node  1 (Hexagon):   0.0812
  Node  2 (Hexagon):   0.0812
  ...
  Node 12 (Cube):      0.0234

Benchmark Results:
  Operations/sec: 31,941
  Mixing time:    τ_mix = 2.34
  Status:         ✓ PASS (within 5% of baseline)
```

### Step 4: Run VQE Ground State Calculation

The Variational Quantum Eigensolver (VQE) finds the ground state energy of the Metatron Hamiltonian.

```bash
cargo run --release --bin vqe_bench
```

**What this does:**
- Constructs the Metatron graph Hamiltonian (-Laplacian)
- Uses variational ansatz (Hardware Efficient or Efficient SU(2))
- Optimizes parameters via ADAM optimizer
- Finds ground state energy and convergence metrics

**Expected output (sample):**
```
=== VQE Benchmark (Hardware Efficient Ansatz) ===

Hamiltonian: -Laplacian (13×13)
Ansatz:      Hardware Efficient, depth=2
Optimizer:   ADAM (lr=0.01)
Max iters:   1000

Iteration   0: Energy = -8.2345
Iteration  50: Energy = -12.8934
Iteration 100: Energy = -12.9987
...
Converged in 147 iterations

Ground State Energy: -12.999734
Approximation error: 0.000266 (0.002%)

✓ PASS (E₀ within tolerance)
```

### Step 5: Run QAOA for MaxCut Problem

Quantum Approximate Optimization Algorithm (QAOA) solves combinatorial optimization problems.

```bash
cargo run --release --bin qaoa_bench
```

**What this does:**
- Defines MaxCut problem on Metatron graph
- Constructs QAOA mixer and problem Hamiltonians
- Optimizes (γ, β) parameters for depth p=3
- Finds approximate solution and approximation ratio

**Expected output (sample):**
```
=== QAOA MaxCut Benchmark (depth=3) ===

Graph:       Metatron (13 nodes, 78 edges)
Classical best cut: 39 edges
QAOA depth:  p = 3

Optimization progress:
  Iteration   0: Cut value = 28.4
  Iteration  50: Cut value = 37.2
  Iteration 100: Cut value = 38.9
...
Converged in 112 iterations

Best cut value:      38.9
Classical optimum:   39.0
Approximation ratio: 0.9974 (99.74%)

✓ PASS (ratio > 0.75)
```

### Step 6: Run All Benchmarks

To run the comprehensive benchmark suite:

```bash
# Quantum Walk benchmarks
cargo run --release --bin quantum_walk_bench

# VQE benchmarks
cargo run --release --bin vqe_bench

# QAOA benchmarks
cargo run --release --bin qaoa_bench

# VQC (Quantum Classifier) benchmarks
cargo run --release --bin vqc_bench

# Integration benchmarks (all algorithms)
cargo run --release --bin integration_bench

# Cross-system comparison
cargo run --release --bin cross_system_bench
```

**Note:** The full benchmark suite takes 10-15 minutes to complete.

---

## Part 2: Running the Seraphic Calibration Shell (SCS)

The SCS provides intelligent configuration optimization through fixpoint-directed calibration.

### Step 1: Navigate to Repository Root

```bash
cd /path/to/qdash  # Repository root, not metatron-qso-rs/
```

### Step 2: Initialize SCS

Initialize SCS with the benchmark baseline directory:

```bash
python -m scs.cli init --benchmark-dir metatron-qso-rs/ci
```

**What this does:**
- Scans benchmark JSON files in `metatron-qso-rs/ci/`
- Initializes Mandorla Field M(t) = 0
- Creates initial configuration state
- Sets up Proof-of-Resonance (PoR) tracking

**Expected output:**
```
[SCS] Initializing Seraphic Calibration Shell...
[SCS] Benchmark directory: metatron-qso-rs/ci
[SCS] Found 6 baseline files:
  - quantum_walk_baseline.json
  - vqe_baseline.json
  - qaoa_baseline.json
  - vqc_baseline.json
  - integration_baseline.json
  - cross_system_baseline.json
[SCS] Mandorla Field initialized: M(t=0) = 0.0
[SCS] Configuration initialized
✓ SCS ready
```

### Step 3: Check SCS Status

```bash
python -m scs.cli status
```

**Expected output:**
```
=== Seraphic Calibration Shell Status ===

Mandorla Field:       M(t) = 0.0000
Performance Triplet:  Φ = (ψ: 0.00, ρ: 1.00, ω: 1.00)
Configuration:        Default (not calibrated)
CRI Regime:           STANDARD
PoR Violations:       0
Calibration Steps:    0

Status: ⚙ Initialized, ready for calibration
```

### Step 4: Run Calibration Steps

Execute 5 calibration steps to optimize configuration:

```bash
python -m scs.cli step -n 5
```

**What this does:**
- **Step 1:** Analyzes benchmark performance
- **Step 2-5:** Applies Double-Kick operator T = Φ_V ∘ Φ_U
- Updates Mandorla Field with resonance patterns
- Validates each step with Proof-of-Resonance (PoR)
- Adjusts configuration parameters monotonically

**Expected output (sample):**
```
=== Calibration Step 1/5 ===
[SCS] Computing Performance Triplet Φ(c)...
  ψ (quality):    0.8234
  ρ (stability):  0.9456
  ω (efficiency): 0.8901
[SCS] Applying Double-Kick Operator...
[SCS] Mandorla Field: M(t) → 0.0823
[PoR] ✓ Quality non-degradation verified
[SCS] Configuration updated

=== Calibration Step 2/5 ===
...

=== Calibration Complete ===
Total steps:     5
Final M(t):      0.4521
Best Φ:          (ψ: 0.9123, ρ: 0.9678, ω: 0.9234)
PoR violations:  0
Status:          ✓ Converging towards fixpoint
```

### Step 5: Check Updated Status

```bash
python -m scs.cli status
```

**Expected output:**
```
=== Seraphic Calibration Shell Status ===

Mandorla Field:       M(t) = 0.4521
Performance Triplet:  Φ = (ψ: 0.91, ρ: 0.97, ω: 0.92)
Configuration:        Calibrated (5 steps)
CRI Regime:           STANDARD → AGGRESSIVE (pending)
PoR Violations:       0
Calibration Steps:    5

Status: ✓ Calibrated, quality improving
```

### Step 6: Export Best Configuration

Export the optimized configuration to a JSON file:

```bash
python -m scs.cli export -o scs_best_config.json
```

**What this does:**
- Exports the current best configuration
- Includes all calibration metadata
- Can be loaded into production deployment

**Expected output:**
```
[SCS] Exporting configuration to: scs_best_config.json
[SCS] Configuration includes:
  - Performance Triplet Φ
  - Mandorla Field state M(t)
  - CRI regime settings
  - PoR validation history
✓ Export complete
```

**Exported JSON structure (sample):**
```json
{
  "version": "1.0.0",
  "timestamp": "2025-11-16T12:34:56Z",
  "mandorla_field": 0.4521,
  "performance_triplet": {
    "quality": 0.9123,
    "stability": 0.9678,
    "efficiency": 0.9234
  },
  "cri_regime": "AGGRESSIVE",
  "por_violations": 0,
  "calibration_steps": 5,
  "configuration": {
    "vqe_depth": 3,
    "qaoa_depth": 4,
    "optimizer_lr": 0.015
  }
}
```

---

## Part 3: Combining QSO Core + SCS (Full Workflow)

### Complete Calibration Workflow

1. **Run baseline benchmarks** to generate performance data:
   ```bash
   cd metatron-qso-rs
   cargo run --release --bin quantum_walk_bench
   cargo run --release --bin vqe_bench
   cargo run --release --bin qaoa_bench
   ```

2. **Initialize SCS** with benchmark results:
   ```bash
   cd ..  # Back to repository root
   python -m scs.cli init --benchmark-dir metatron-qso-rs/ci
   ```

3. **Calibrate configuration** over multiple steps:
   ```bash
   python -m scs.cli step -n 10
   ```

4. **Export optimized config**:
   ```bash
   python -m scs.cli export -o production_config.json
   ```

5. **Re-run benchmarks** with new configuration (manual integration):
   ```bash
   # Apply configuration to QSO Core (requires manual code integration)
   # Then re-run benchmarks to validate improvement
   cd metatron-qso-rs
   cargo run --release --bin vqe_bench
   ```

6. **Iterate** until convergence (PoR violations = 0, Φ stabilizes)

---

## Part 4: Example Outputs

### Example: Quantum Walk Probability Distribution

```
Time evolution from central node |0⟩:

t = 0.0:   P(0) = 1.0000  (all probability at center)
t = 0.5:   P(0) = 0.6234, P(1-6) ≈ 0.0627 each (spreading to hexagon)
t = 1.0:   P(0) = 0.2456, P(1-6) ≈ 0.0812, P(7-12) ≈ 0.0234 (full spread)
t = 2.0:   Nearly uniform: P(i) ≈ 0.0769 for all i (mixed state)

Mixing time: τ_mix ≈ 2.34
```

### Example: VQE Convergence Plot (text representation)

```
Energy
  -13.0 |                            ******
  -12.5 |                   *********
  -12.0 |             ******
  -11.5 |        *****
  -11.0 |    ****
  -10.5 |  **
   -9.0 | *
        +----------------------------------
           0    50   100  150  200  Iteration

Ground state: E₀ = -12.999734 Ha
Convergence:  147 iterations
Error:        0.0003% from exact
```

### Example: SCS Mandorla Field Evolution

```
M(t)
 0.5 |                                  ****
 0.4 |                            ******
 0.3 |                      ******
 0.2 |                ******
 0.1 |          ******
 0.0 | ********
     +----------------------------------
        0     2     4     6     8    Step

Fixpoint trajectory: M(t) → M* ≈ 0.52 (approaching attractor)
```

---

## Part 5: Troubleshooting

### Issue: Benchmark Fails with "Baseline not found"

**Solution:**
```bash
# Ensure CI baselines exist
ls metatron-qso-rs/ci/*.json

# If missing, run benchmarks first to generate baselines
cd metatron-qso-rs
cargo run --release --bin quantum_walk_bench
```

### Issue: SCS Cannot Find Benchmarks

**Solution:**
```bash
# Ensure you're in repository ROOT, not metatron-qso-rs/
pwd  # Should show /path/to/qdash

# Use absolute path
python -m scs.cli init --benchmark-dir $(pwd)/metatron-qso-rs/ci
```

### Issue: VQE Not Converging

**Possible causes:**
- Ansatz depth too shallow → Increase depth to 3-4
- Learning rate too high → Reduce to 0.005-0.01
- Max iterations too low → Increase to 2000+

**Solution:**
```bash
# Edit VQE parameters in src/vqa/vqe.rs or via config file
# Then rebuild and re-run
cargo build --release
cargo run --release --bin vqe_bench
```

---

## Part 6: Next Steps

After successfully running the core:

1. **Explore the API** — Generate and browse Rust documentation:
   ```bash
   cd metatron-qso-rs
   cargo doc --open
   ```

2. **Write your own quantum program** — See examples in:
   - `metatron-qso-rs/examples/vqa_demo.rs`
   - README.md Quick Start section

3. **Integrate with backend services** — See:
   - `metatron_backend/` for REST API
   - `metatron_telemetry/` for monitoring

4. **Read the scientific papers**:
   - `SeraphicCalibrationModule.pdf` — SCS mathematical foundation
   - `QUANTENINFORMATIONSVERARBEITUNG_DOKUMENTATION.md` — Quantum theory

5. **Contribute** — See Contributing section in README.md

---

## Summary of Commands

### QSO Core Quick Reference

```bash
cd metatron-qso-rs
cargo build --release               # Build optimized
cargo test --lib                    # Run tests
cargo run --release --bin quantum_walk_bench  # Quantum walk
cargo run --release --bin vqe_bench           # VQE
cargo run --release --bin qaoa_bench          # QAOA
cargo doc --open                    # Documentation
```

### SCS Quick Reference

```bash
cd /path/to/qdash  # Repository root
python -m scs.cli init --benchmark-dir metatron-qso-rs/ci  # Initialize
python -m scs.cli step -n 5                                # Calibrate
python -m scs.cli status                                   # Check
python -m scs.cli export -o config.json                    # Export
```

---

**You're now ready to run quantum algorithms on the Metatron geometry! 🚀**

*For support: [GitHub Issues](https://github.com/LashSesh/qso/issues)*

*Last updated: 2025-11-16*
