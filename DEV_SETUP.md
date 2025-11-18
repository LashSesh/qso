# Q⊗DASH Developer Setup Guide

Welcome to the Q⊗DASH (Metatron VM) development environment! This guide will help you set up your local development environment and get you productive quickly.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Repository Setup](#repository-setup)
3. [Building the Quantum Core](#building-the-quantum-core)
4. [Setting up the Calibration Shell](#setting-up-the-calibration-shell)
5. [Running Tests](#running-tests)
6. [Code Quality Tools](#code-quality-tools)
7. [CI/CD Local Checks](#cicd-local-checks)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Toolchain Versions

**Rust:**
- **Version:** 1.85.0 or later
- **Edition:** 2024
- **Installation:**
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustup update
  rustc --version  # Should be 1.85.0+
  ```

**Python:**
- **Version:** 3.8 or later (3.11+ recommended)
- **Installation:**
  ```bash
  python3 --version  # Should be 3.8+
  pip3 --version
  ```

### Optional but Recommended

- **Git:** 2.30+
- **Just:** Task runner (optional, for automation)
  ```bash
  cargo install just
  ```
- **Rust Analyzer:** LSP for IDE integration
- **Python Virtual Environment:** For isolated SCS development

---

## Repository Setup

### 1. Clone the Repository

```bash
git clone https://github.com/LashSesh/qso.git
cd qdash
```

### 2. Repository Structure Overview

```
qdash/
├── metatron-qso-rs/          # Rust quantum core (Product Slice 1)
├── scs/                       # Seraphic Calibration Shell (Product Slice 1)
├── metatron_backend/          # Backend services (Product Slice 2)
├── metatron_telemetry/        # Observability (Product Slice 2)
├── metatron_triton/           # Triton integration (Product Slice 2)
├── metatron_dionice_bridge/   # DioniceOS bridge (Product Slice 2)
├── external/                  # External integrations (Product Slice 3)
├── Cargo.toml                 # Workspace configuration
└── requirements-scs.txt       # Python dependencies
```

For detailed architecture, see: [`PRODUCT_OVERVIEW.md`](PRODUCT_OVERVIEW.md)

---

## Building the Quantum Core

### Step 1: Build the Metatron QSO Core

The quantum core is implemented in Rust and located in `metatron-qso-rs/`.

```bash
cd metatron-qso-rs

# Development build (faster compilation, slower runtime)
cargo build

# Release build (optimized for performance)
cargo build --release

# Check for compilation errors without building
cargo check
```

**Expected output:**
```
   Compiling metatron-qso-rs v0.1.0 (/path/to/qdash/metatron-qso-rs)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 16.76s
```

### Step 2: Verify Build Success

```bash
# Run all unit tests
cargo test --lib

# Run specific test module
cargo test --lib quantum::state

# Show test output (including println! statements)
cargo test --lib -- --nocapture
```

**Expected result:** All tests should pass (34 tests as of latest version)

---

## Setting up the Calibration Shell

### Step 1: Install Python Dependencies

From the repository root:

```bash
# Option 1: System-wide installation
pip3 install -r requirements-scs.txt

# Option 2: Virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements-scs.txt
```

### Step 2: Verify SCS Installation

```bash
# Test CLI help
python -m scs.cli --help

# Test module imports
python -c "import scs; from scs import calibrator, cli; print('✓ SCS modules loaded successfully')"
```

**Expected output:**
```
usage: cli.py [-h] [--benchmark-dir BENCHMARK_DIR] ...
✓ SCS modules loaded successfully
```

### Step 3: Initialize SCS (Optional)

```bash
# Initialize SCS with benchmark directory
python -m scs.cli init --benchmark-dir metatron-qso-rs/ci

# Check status
python -m scs.cli status
```

---

## Running Tests

### Rust Core Tests

```bash
cd metatron-qso-rs

# All unit tests
cargo test --lib

# Integration tests (benchmark binaries)
cargo test --bins

# Run with full backtrace on failure
RUST_BACKTRACE=full cargo test --lib

# Run specific test
cargo test --lib test_vqe_basic
```

### Python SCS Tests

```bash
# Test module imports
python -m pytest scs/ -v  # If pytest is available

# Manual test
python -c "
from scs.calibrator import Calibrator
from scs.config import SCSConfig
print('✓ SCS calibrator initialized successfully')
"
```

---

## Code Quality Tools

### Rust

#### Formatting

```bash
cd metatron-qso-rs

# Check formatting (no changes)
cargo fmt --check

# Auto-format all code
cargo fmt
```

#### Linting

```bash
# Run Clippy (Rust linter)
cargo clippy

# Fail on warnings (strict mode)
cargo clippy -- -D warnings

# Auto-fix simple issues
cargo clippy --fix
```

#### Documentation

```bash
# Generate and open documentation
cargo doc --open

# Check for broken doc links
cargo doc --no-deps
```

### Python

#### Formatting (if tools available)

```bash
# Using Black (install: pip install black)
black scs/

# Using autopep8
autopep8 --in-place --recursive scs/

# Check with flake8
flake8 scs/ --max-line-length=100
```

---

## CI/CD Local Checks

Before pushing code, run these checks to match CI/CD pipeline:

### Rust Pipeline Simulation

```bash
cd metatron-qso-rs

# 1. Format check
cargo fmt --check

# 2. Clippy (linting)
cargo clippy -- -D warnings

# 3. Build
cargo build --release

# 4. Tests
cargo test --lib

# 5. Benchmarks (optional, takes longer)
cargo run --release --bin quantum_walk_bench
```

### SCS Pipeline Simulation

```bash
# 1. Module import test
python -c "import scs; print('✓ OK')"

# 2. CLI test
python -m scs.cli --help > /dev/null && echo "✓ CLI OK"

# 3. Basic functionality
python -m scs.cli init --benchmark-dir metatron-qso-rs/ci
python -m scs.cli status
```

---

## Troubleshooting

### Common Issues

#### 1. Rust Compilation Errors

**Problem:** `error: edition 2024 is unstable`

**Solution:**
```bash
rustup update
rustc --version  # Ensure 1.85.0+
```

**Problem:** `nalgebra` version mismatch

**Solution:**
```bash
cargo clean
cargo update
cargo build
```

#### 2. Python Import Errors

**Problem:** `ModuleNotFoundError: No module named 'scs'`

**Solution:**
```bash
# Ensure you're in the repository root
cd /path/to/qdash

# Re-install dependencies
pip install -r requirements-scs.txt

# Verify PYTHONPATH
python -c "import sys; print('\\n'.join(sys.path))"
```

#### 3. Test Failures

**Problem:** Quantum walk tests fail with numerical precision errors

**Solution:** This is expected for some advanced algorithms. Check:
```bash
# Run with verbose output
cargo test --lib test_quantum_walk -- --nocapture

# Check if only specific tests fail
cargo test --lib -- --test-threads=1
```

#### 4. Clippy Warnings

**Problem:** Too many clippy warnings

**Solution:**
```bash
# Focus on critical warnings only
cargo clippy -- -W clippy::correctness

# Allow specific lints temporarily
cargo clippy -- -A clippy::needless_range_loop
```

---

## Development Workflow

### Typical Development Cycle

1. **Make changes** to Rust or Python code
2. **Format** with `cargo fmt` / `black`
3. **Lint** with `cargo clippy` / `flake8`
4. **Test** with `cargo test --lib`
5. **Commit** with descriptive message
6. **Push** and create pull request

### Recommended VS Code Extensions

- **rust-analyzer** — LSP for Rust
- **Python** — Microsoft Python extension
- **Even Better TOML** — TOML syntax support
- **GitLens** — Git integration

---

## Next Steps

Once your development environment is set up:

1. **Run a quantum walk demo:**
   ```bash
   cd metatron-qso-rs
   cargo run --release --bin quantum_walk_bench
   ```

2. **Try SCS calibration:**
   ```bash
   python -m scs.cli step -n 5
   ```

3. **Read the architecture guide:**
   - [`PRODUCT_OVERVIEW.md`](PRODUCT_OVERVIEW.md)
   - [`HOWTO_RUN_CORE.md`](HOWTO_RUN_CORE.md)

4. **Explore the codebase:**
   ```bash
   cargo doc --open  # Browse Rust API docs
   ```

---

## Getting Help

- **Documentation:** See [`docs/`](docs/) directory
- **Issues:** [GitHub Issues](https://github.com/LashSesh/qso/issues)
- **Architecture:** [`PRODUCT_OVERVIEW.md`](PRODUCT_OVERVIEW.md)
- **API Docs:** `cargo doc --open` in `metatron-qso-rs/`

---

**Happy Quantum Coding! 🌌**

*Last updated: 2025-11-16*
