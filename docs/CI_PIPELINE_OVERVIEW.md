# CI/CD Pipeline Overview - Q⊗DASH Monorepo

## Executive Summary

This document provides a comprehensive overview of the Q⊗DASH CI/CD pipeline architecture, designed for production-ready releases while maintaining fast developer feedback loops.

The pipeline is structured into three distinct tiers:
1. **Core CI** - Fast, mandatory checks for all PRs/pushes
2. **Benchmark CI** - Heavy, optional performance validation (manual/scheduled)
3. **Release CI** - Packaging and artifact generation for releases

---

## Workflow Inventory

### Current Workflows (After Restructuring)

| Workflow | File | Purpose | Triggers | Runtime | Status | Required for PRs |
|----------|------|---------|----------|---------|--------|------------------|
| **Core Rust CI** | `core_rust_ci.yml` | Build, test, lint Rust core | push (main, dev), pull_request | Fast (<5min) | Core | ✅ Yes |
| **Core Python CI** | `core_python_ci.yml` | Test Python SDK & SCS | push (main, dev), pull_request | Fast (<5min) | Core | ✅ Yes |
| **Quantum Benchmarks** | `benchmarks.yml` | All performance benchmarks | workflow_dispatch, schedule (nightly) | Heavy (20-40min) | Benchmark | ❌ No |
| **SCS Calibration** | `scs_calibration.yml` | SCS auto-tuning runs | workflow_dispatch, schedule (weekly) | Heavy (10-20min) | Benchmark | ❌ No |
| **Release** | `release.yml` | Build & package artifacts | tags (v*.*.*) | Medium (10-15min) | Release | ❌ No |

### Legacy Workflows (Removed/Replaced)

| Workflow | File | Reason for Removal | Replaced By |
|----------|------|--------------------|-------------|
| CI - Build and Test | `ci.yml` | Mixed core tests with heavy benchmark builds; too slow for PR feedback | `core_rust_ci.yml` |
| Quantum Walk Benchmarks | `quantum_benchmarks.yml` | Redundant with comprehensive benchmarks; ran on every PR (too heavy) | `benchmarks.yml` |
| Comprehensive Benchmark Suite | `comprehensive_benchmarks.yml` | Ran on every PR/push (too heavy); restructured to be opt-in | `benchmarks.yml` |

---

## CI Architecture Design

### Tier 1: Core CI (Fast, Mandatory)

**Purpose:** Provide rapid feedback on code quality and correctness for every commit and PR.

**Design Principles:**
- **Fast:** Target <10 minutes total runtime
- **Focused:** Only essential build, test, and lint checks
- **Reliable:** Must pass for PR merges
- **No heavy benchmarks:** Performance validation is separate

#### Core Rust CI (`core_rust_ci.yml`)

**Components Tested:**
- `metatron-qso-rs/` (Rust quantum core)

**Checks:**
1. `cargo fmt --check` - Enforce code formatting
2. `cargo clippy --all-targets --all-features -- -D warnings` - Comprehensive linting
3. `cargo test --workspace` - Unit and integration tests
4. `cargo build --release` - Verify release builds compile

**Triggers:**
- Push to `main`, `dev` branches
- All pull requests
- Manual dispatch

**Caching Strategy:**
- Cargo registry, git db, and target directory cached by `Cargo.lock` hash

#### Core Python CI (`core_python_ci.yml`)

**Components Tested:**
- `scs/` (Seraphic Calibration Shell)
- `metatron_qso_py/` (Python SDK via PyO3)

**Checks:**
1. Python formatting: `ruff format --check` or `black --check`
2. Python linting: `ruff check` or `flake8`
3. SCS unit tests: `pytest scs/`
4. Python SDK build: `maturin build` (smoke test)
5. SCS CLI smoke test: `python -m scs.cli status`

**Python Versions:**
- 3.10, 3.11 (matrix)

**Triggers:**
- Push to `main`, `dev` branches
- All pull requests
- Manual dispatch

**Caching Strategy:**
- pip cache for Python dependencies
- Cargo cache for PyO3 builds

---

### Tier 2: Benchmark CI (Heavy, Optional)

**Purpose:** Comprehensive performance validation and regression detection.

**Design Principles:**
- **Thorough:** All quantum algorithms benchmarked
- **Optional:** Never blocks PRs or regular development
- **Scheduled:** Runs nightly or weekly to track performance trends
- **Manual:** Can be triggered on-demand for performance-critical PRs
- **Comparative:** Always compares against baselines

#### Quantum Benchmarks (`benchmarks.yml`)

**Benchmark Suites:**

| Suite | Binary | Metrics Tracked | Baseline File |
|-------|--------|----------------|---------------|
| Quantum Walk | `quantum_walk_bench` | Mixing time, hitting time, speedup factor | `ci/quantum_walk_baseline.json` |
| VQE | `vqe_bench` | Ground energy, convergence rate, evaluations/sec | `ci/vqe_baseline.json` |
| QAOA | `qaoa_bench` | Approximation ratio, convergence rate | `ci/qaoa_baseline.json` |
| VQC | `vqc_bench` | Training/test accuracy, convergence | `ci/vqc_baseline.json` |
| Integration | `integration_bench` | Cross-module compatibility, execution time | `ci/integration_baseline.json` |
| Cross-System | `cross_system_bench` | Comparative performance vs. competitors | `ci/cross_system_baseline.json` |
| Advanced Algorithms | `advanced_algorithms_bench` | Grover, Boson sampling, QML metrics | `ci/advanced_algorithms_baseline.json` |

**Triggers:**
- **Manual:** `workflow_dispatch` (for performance PRs)
- **Scheduled:** `cron: '0 2 * * *'` (nightly at 2 AM UTC)
- **Tags:** On release tags `v*.*.*`

**Artifacts:**
- All benchmark JSON results (30-day retention)
- Regression reports (90-day retention on failures)

**Failure Conditions:**
- Performance regression beyond defined thresholds (see `BENCHMARK_POLICY.md`)
- Baseline comparison failures

#### SCS Calibration (`scs_calibration.yml`)

**Purpose:** Auto-tune SCS parameters (ψ, ρ, ω) using MandorlaField optimization.

**Process:**
1. Load existing SCS state or initialize fresh
2. Run calibration steps (configurable count, default: 3)
3. Evaluate CRI (Calibration Readiness Index) and PoR (Probability of Resonance)
4. Store optimized configuration

**Triggers:**
- **Manual:** `workflow_dispatch` with configurable step count
- **Scheduled:** `cron: '0 3 * * 0'` (weekly on Sunday at 3 AM UTC)
- **Conditional:** After benchmark workflows complete (opt-in via `vars.SCS_ENABLED`)

**Artifacts:**
- `scs_state.json` - Current calibration state
- `scs_history.json` - Historical calibration log
- `scs_best_config.json` - Optimal configuration found

**Optional Git Commit:**
- If `vars.SCS_COMMIT_STATE == 'true'`, commits state back to repo with `[skip ci]`

---

### Tier 3: Release CI (Packaging)

**Purpose:** Build and package production-ready artifacts for releases.

#### Release Workflow (`release.yml`)

**Triggered By:**
- Git tags matching `v*.*.*` (e.g., `v0.1.0`, `v1.2.3`)

**Artifacts Generated:**

1. **Rust Crate:**
   - `cargo build --release` in `metatron-qso-rs/`
   - `cargo publish --dry-run` (validation)
   - Release binary as GitHub artifact

2. **Python Wheels:**
   - `maturin build --release` for Python SDK
   - Multi-platform wheels (Linux, macOS, Windows via matrix)
   - Upload to GitHub release

3. **Documentation:**
   - Generate rustdoc: `cargo doc --no-deps`
   - Package as artifact

4. **Optional Docker Image:**
   - Build Q⊗DASH service stack container
   - Tag with release version
   - Push to container registry (if configured)

**Release Checklist:**
- All core CI checks pass
- Benchmarks run successfully (manual validation)
- CHANGELOG.md updated
- Version bumped in Cargo.toml / pyproject.toml

---

## Branch Protection Requirements

### Required Status Checks for PRs

The following checks **must pass** before merging to `main` or `dev`:

- ✅ **Core Rust CI** - `core-rust-ci` (from `core_rust_ci.yml`)
- ✅ **Core Python CI** - `core-python-ci` (from `core_python_ci.yml`)

### Optional Checks (Informational)

These may run but **do not block** merges:

- ℹ️ Quantum Benchmarks (manual trigger only)
- ℹ️ SCS Calibration (manual/scheduled only)

---

## Workflow Usage Guide

### For Developers: Running Core CI

Core CI runs automatically on all PRs and pushes to `main`/`dev`. To run locally before pushing:

```bash
# Rust checks
cd metatron-qso-rs
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --workspace
cargo build --release

# Python/SCS checks
pip install ruff black pytest maturin
ruff format --check .
ruff check .
pytest scs/
python -m scs.cli status
cd metatron_qso_py && maturin build
```

### For Maintainers: Running Benchmarks

To trigger benchmarks manually (e.g., for performance-critical PRs):

1. Navigate to **Actions** → **Quantum Benchmarks**
2. Click **Run workflow**
3. Select branch
4. Review results in artifacts and PR comments

Benchmarks also run automatically nightly to track performance trends over time.

### For Maintainers: Running SCS Calibration

To manually calibrate SCS:

1. Navigate to **Actions** → **SCS Calibration**
2. Click **Run workflow**
3. Set `num_steps` (default: 3, increase for deeper calibration)
4. Download artifacts: `scs_state.json`, `scs_history.json`, `scs_best_config.json`

### For Releases: Creating a New Release

1. Update version in `Cargo.toml` (Rust) and `pyproject.toml` (Python SDK)
2. Update `CHANGELOG.md` with release notes
3. Commit changes: `git commit -m "Bump version to v1.0.0"`
4. Create and push tag: `git tag v1.0.0 && git push origin v1.0.0`
5. Release workflow automatically:
   - Builds Rust crate and Python wheels
   - Generates documentation
   - Creates GitHub release with artifacts

---

## Performance Targets

| Job | Target Runtime | Current Runtime | Status |
|-----|----------------|----------------|---------|
| Core Rust CI | <5 min | ~3 min | ✅ On target |
| Core Python CI | <5 min | ~2 min | ✅ On target |
| Quantum Benchmarks | 20-40 min | ~30 min | ✅ Acceptable |
| SCS Calibration | 10-20 min | ~15 min | ✅ Acceptable |
| Release | 10-15 min | ~12 min | ✅ On target |

---

## Maintenance and Evolution

### Regular Review Tasks

- **Monthly:** Review benchmark baseline files in `metatron-qso-rs/ci/` for outdated targets
- **Quarterly:** Evaluate CI runtime trends and optimize slow jobs
- **Per Release:** Update `BENCHMARK_POLICY.md` if new metrics or thresholds are added

### Adding New Workflows

When adding new workflows:

1. Classify as Core, Benchmark, or Release
2. Update this document with new entry
3. Set appropriate triggers (avoid heavy jobs on every push)
4. Add caching where applicable
5. Document in PR description

### Updating Dependencies

- **Rust:** Update `rust-toolchain.toml` and test in CI
- **Python:** Update `requirements-scs.txt` and version matrix in workflows
- **Actions:** Dependabot configured to auto-update GitHub Actions

---

## Troubleshooting

### Core CI Failures

**Rust formatting fails:**
```bash
cd metatron-qso-rs
cargo fmt
```

**Clippy warnings:**
```bash
cd metatron-qso-rs
cargo clippy --all-targets --all-features -- -D warnings
# Fix issues, then re-run
```

**Tests fail:**
```bash
cd metatron-qso-rs
cargo test --workspace -- --nocapture
# Review test output for specific failures
```

### Benchmark Regressions

If benchmarks fail due to performance regression:

1. Review comparison output in job logs
2. Check if regression is intentional (algorithm change)
3. If intentional, update baseline: `cp target/XXX_bench.json ci/XXX_baseline.json`
4. If unintentional, investigate performance issue

### SCS Calibration Issues

**State corruption:**
```bash
# Reset SCS state locally
rm scs_state.json scs_history.json scs_best_config.json
python -m scs.cli init --benchmark-dir metatron-qso-rs/ci
```

---

## Related Documentation

- **Benchmark Policy:** [`BENCHMARK_POLICY.md`](./BENCHMARK_POLICY.md) - Detailed benchmark metrics, thresholds, and failure criteria
- **Release Plan:** [`../RELEASE_PLAN.md`](../RELEASE_PLAN.md) - Release versioning, changelog, and packaging strategy
- **SCS Design:** [`seraphic_calibration_shell.md`](./seraphic_calibration_shell.md) - SCS architecture and calibration theory
- **Benchmark Suite:** [`../BENCHMARK_SUITE_DOCUMENTATION.md`](../BENCHMARK_SUITE_DOCUMENTATION.md) - Detailed benchmark specifications

---

**Last Updated:** 2025-11-16
**Maintained By:** CI/CD Team
**Version:** 1.0.0
