# Release Plan - Q⊗DASH v0.1.0

**Metatron VM / Quantum State Operator Framework**

## Overview

This document outlines the packaging, distribution, and deployment strategy for Q⊗DASH version 0.1.0, the first production-ready release of our quantum computing framework with integrated SCS auto-tuning.

## Release Components

### 1. Rust Crate - `metatron-qso`

**Target:** Rust developers and systems programmers

#### Package Structure
```toml
[package]
name = "metatron-qso"
version = "0.1.0"
edition = "2024"
rust-version = "1.85.0"
authors = ["Sebastian Klemm", "Q⊗DASH Contributors"]
license = "MIT"
description = "Quantum State Operator framework based on Metatron Cube geometry with DTL"
repository = "https://github.com/LashSesh/qso"
documentation = "https://docs.rs/metatron-qso"
keywords = ["quantum", "metatron", "vqa", "qaoa", "quantum-walk"]
categories = ["science", "simulation", "algorithms"]
```

#### Publishing to crates.io

**Pre-Release Checklist:**
- [ ] All tests passing (`cargo test --all`)
- [ ] Documentation complete (`cargo doc --no-deps`)
- [ ] Examples compile and run
- [ ] Benchmarks execute successfully
- [ ] README.md up to date
- [ ] CHANGELOG.md updated
- [ ] Version bumped in all Cargo.toml files
- [ ] Git tag created: `v0.1.0`

**Publishing Steps:**
```bash
# Verify package
cd metatron-qso-rs
cargo package --allow-dirty
cargo package --list

# Publish to crates.io
cargo publish

# Verify on crates.io
open https://crates.io/crates/metatron-qso
```

**Post-Release:**
- Monitor download stats
- Respond to issues/questions
- Update documentation based on feedback

---

### 2. Python Package - `metatron-qso`

**Target:** Data scientists, ML researchers, Python developers

#### Package Structure

**pyproject.toml:**
```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "metatron-qso"
version = "0.1.0"
description = "High-performance quantum computing framework powered by Rust"
authors = [
    {name = "Sebastian Klemm", email = "sebastian@qdash.dev"}
]
license = {text = "MIT"}
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Rust",
    "Topic :: Scientific/Engineering :: Physics",
]
keywords = ["quantum", "quantum-computing", "metatron", "vqa", "qaoa"]
dependencies = [
    "numpy>=1.20.0",
]

[project.optional-dependencies]
scs = [
    # SCS is included but optional (graceful degradation)
]
dev = [
    "pytest>=7.0.0",
    "jupyter>=1.0.0",
    "matplotlib>=3.5.0",
]

[project.urls]
Homepage = "https://github.com/LashSesh/qso"
Documentation = "https://qdash.readthedocs.io"
Repository = "https://github.com/LashSesh/qso"
Changelog = "https://github.com/LashSesh/qso/blob/main/CHANGELOG.md"

[tool.maturin]
python-source = "python"
module-name = "metatron_qso._metatron_qso_internal"
```

#### Building Wheels

**Prerequisites:**
```bash
pip install maturin twine
rustup target add x86_64-unknown-linux-gnu
rustup target add x86_64-apple-darwin
rustup target add aarch64-apple-darwin
```

**Build for Multiple Platforms:**

**Linux (x86_64):**
```bash
cd metatron_qso_py
maturin build --release --target x86_64-unknown-linux-gnu
```

**macOS (Intel):**
```bash
maturin build --release --target x86_64-apple-darwin
```

**macOS (Apple Silicon):**
```bash
maturin build --release --target aarch64-apple-darwin
```

**Windows (x86_64):**
```bash
maturin build --release --target x86_64-pc-windows-msvc
```

#### Publishing to PyPI

**Test PyPI First:**
```bash
cd metatron_qso_py

# Build
maturin build --release

# Upload to Test PyPI
maturin publish --repository testpypi

# Test installation
pip install --index-url https://test.pypi.org/simple/ metatron-qso

# Run tests
python -c "import metatron_qso; print(metatron_qso.__version__)"
```

**Production PyPI:**
```bash
# Publish to PyPI
maturin publish

# Verify
pip install metatron-qso
python -c "import metatron_qso; graph = metatron_qso.MetatronGraph(); print(graph)"
```

**Post-Release:**
- Monitor PyPI stats
- Update conda-forge recipe (future)
- Announce on Python communities

---

### 3. SCS Auto-Tuner - Standalone Python Package

**Target:** Researchers, quantum algorithm developers

#### Option A: Bundled with Python SDK (Current)

SCS is included in `metatron-qso` Python package:
- Located in top-level `scs/` directory
- Automatically available when `metatron-qso` is installed
- Optional dependency (graceful fallback if missing)

**Advantages:**
- Single installation for users
- Seamless integration
- No dependency management issues

**Disadvantages:**
- Increases package size
- Couples SCS to Metatron QSO

#### Option B: Separate Package (Future)

Create standalone `scs-autotuner` package:

**pyproject.toml:**
```toml
[project]
name = "scs-autotuner"
version = "0.1.0"
description = "Seraphic Calibration Shell - Generic auto-tuner for quantum algorithms"
dependencies = [
    "numpy>=1.20.0",
]
```

**Benefits:**
- Independent versioning
- Usable with other quantum frameworks
- Cleaner separation of concerns

**Migration Path:**
1. v0.1.0: Bundled (current)
2. v0.2.0: Standalone package, backward-compatible
3. v0.3.0: Deprecate bundled version

---

### 4. Docker Container

**Target:** DevOps, cloud deployment, reproducible environments

#### Dockerfile

```dockerfile
FROM rust:1.85-bookworm as builder

# Install Python and maturin
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install maturin

# Build workspace
WORKDIR /build
COPY . .

# Build Rust core
WORKDIR /build/metatron-qso-rs
RUN cargo build --release

# Build Python SDK
WORKDIR /build/metatron_qso_py
RUN maturin build --release

# Runtime image
FROM python:3.11-slim-bookworm

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy built artifacts
COPY --from=builder /build/metatron_qso_py/target/wheels/*.whl /tmp/
COPY --from=builder /build/scs /app/scs

# Install Python wheel
RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Install SCS
WORKDIR /app
ENV PYTHONPATH=/app:$PYTHONPATH

# Create benchmark directory
RUN mkdir -p /app/benchmarks

# Default command
CMD ["python3", "-m", "scs.cli", "--help"]
```

#### Docker Compose

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  qdash:
    build: .
    container_name: qdash_v0.1.0
    volumes:
      - ./benchmarks:/app/benchmarks
      - ./configs:/app/configs
    environment:
      - PYTHONUNBUFFERED=1
    command: python3 -m scs.cli status

  qdash-notebook:
    build: .
    container_name: qdash_notebook
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/app/notebooks
      - ./benchmarks:/app/benchmarks
    command: jupyter notebook --ip=0.0.0.0 --allow-root --no-browser
```

#### Building and Publishing

```bash
# Build image
docker build -t qdash/metatron-qso:0.1.0 .
docker build -t qdash/metatron-qso:latest .

# Test locally
docker run -it qdash/metatron-qso:0.1.0 python3 -c "import metatron_qso; print(metatron_qso.__version__)"

# Push to Docker Hub
docker login
docker push qdash/metatron-qso:0.1.0
docker push qdash/metatron-qso:latest

# Or push to GitHub Container Registry
docker tag qdash/metatron-qso:0.1.0 ghcr.io/lashsesh/qdash:0.1.0
docker push ghcr.io/lashsesh/qdash:0.1.0
```

---

## Distribution Channels

### 1. GitHub Release

**Steps:**
1. Create git tag: `git tag -a v0.1.0 -m "Release v0.1.0"`
2. Push tag: `git push origin v0.1.0`
3. Create GitHub Release at https://github.com/LashSesh/qso/releases/new
4. Attach artifacts:
   - Source tarball (auto-generated by GitHub)
   - Pre-built wheels for major platforms
   - Docker image reference
   - CHANGELOG.md excerpt
5. Write release notes (use CHANGELOG.md as template)

**Release Assets:**
- `metatron-qso-0.1.0.tar.gz` (source)
- `metatron_qso-0.1.0-cp38-abi3-linux_x86_64.whl`
- `metatron_qso-0.1.0-cp38-abi3-macosx_x86_64.whl`
- `metatron_qso-0.1.0-cp38-abi3-macosx_aarch64.whl`
- `metatron_qso-0.1.0-cp38-abi3-win_amd64.whl`
- `docker-compose.yml`
- `RELEASE_NOTES.md`

### 2. crates.io

**URL:** https://crates.io/crates/metatron-qso

**Installation:**
```bash
cargo add metatron-qso
```

**Documentation:** Automatically published to https://docs.rs/metatron-qso

### 3. PyPI

**URL:** https://pypi.org/project/metatron-qso/

**Installation:**
```bash
pip install metatron-qso
```

**Documentation:** Link to https://github.com/LashSesh/qso

### 4. Docker Hub / GHCR

**Docker Hub:** https://hub.docker.com/r/qdash/metatron-qso
**GHCR:** https://github.com/LashSesh/qso/pkgs/container/qdash

**Usage:**
```bash
docker pull qdash/metatron-qso:0.1.0
docker run -it qdash/metatron-qso:0.1.0
```

---

## Documentation Deployment

### 1. GitHub Pages

**Host:** https://lashsesh.github.io/qso/

**Structure:**
```
docs/
├── index.html (generated from README)
├── api/
│   ├── rust/ (rustdoc)
│   └── python/ (pdoc or sphinx)
├── guides/
│   ├── scs-core-design.html
│   ├── scs-benchmark-schema.html
│   └── scs-usage-guide.html
└── examples/
    └── notebooks/ (rendered Jupyter notebooks)
```

**Deployment:**
```bash
# Generate Rust docs
cd metatron-qso-rs
cargo doc --no-deps
cp -r target/doc ../docs/api/rust

# Generate Python docs
cd ../metatron_qso_py
pdoc --html --output-dir ../docs/api/python metatron_qso

# Deploy to GitHub Pages
cd ..
git checkout gh-pages
git add docs/
git commit -m "Update docs for v0.1.0"
git push origin gh-pages
```

### 2. ReadTheDocs (Future)

**URL:** https://qdash.readthedocs.io/

**Setup:**
- Link GitHub repository
- Configure `.readthedocs.yml`
- Use Sphinx for documentation generation
- Auto-build on commits

---

## Version Management

### Semantic Versioning

Follow [SemVer 2.0.0](https://semver.org/):
- **Major (X.0.0):** Breaking changes
- **Minor (0.X.0):** New features, backward-compatible
- **Patch (0.0.X):** Bug fixes, backward-compatible

### Version Locations

Update version in all locations:
1. `metatron-qso-rs/Cargo.toml` → `version = "0.1.0"`
2. `metatron_qso_py/Cargo.toml` → `version = "0.1.0"`
3. `metatron_qso_py/pyproject.toml` → `version = "0.1.0"`
4. `scs/__init__.py` → `__version__ = "0.1.0"`
5. `CHANGELOG.md` → New version entry
6. Git tag → `v0.1.0`

### Release Cadence

- **v0.1.x:** Monthly patch releases (bug fixes)
- **v0.x.0:** Quarterly minor releases (new features)
- **v1.0.0:** When API is stable and production-tested

---

## Testing & Validation

### Pre-Release Testing

**Rust:**
```bash
cd metatron-qso-rs
cargo test --all
cargo bench
cargo clippy -- -D warnings
cargo fmt --check
```

**Python:**
```bash
cd metatron_qso_py
maturin develop
pytest tests/
python examples/01_quantum_walk_basic.py
python examples/02_qaoa_maxcut_basic.py
```

**SCS:**
```bash
python -m scs.cli init
python -m scs.cli step -n 3
python -m scs.cli status
```

**Integration:**
```python
import metatron_qso

# Test auto-tuning
graph = metatron_qso.MetatronGraph()
result, proposal = metatron_qso.solve_maxcut_qaoa_with_tuning(
    graph, depth=3, max_iters=100, auto_calibrate=True
)
assert result['approximation_ratio'] > 0.7
assert proposal is not None
```

### Platform Testing Matrix

| Platform | Rust | Python | Docker |
|----------|------|--------|--------|
| Linux x86_64 | ✓ | ✓ | ✓ |
| macOS Intel | ✓ | ✓ | ✓ |
| macOS ARM64 | ✓ | ✓ | ✓ |
| Windows x86_64 | ✓ | ✓ | ✓ |

---

## Release Checklist

### Pre-Release (1-2 weeks before)

- [ ] All features complete and merged
- [ ] All tests passing on CI
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in all locations
- [ ] Examples and notebooks verified
- [ ] Benchmark baselines updated
- [ ] Security audit (dependencies, code)

### Release Day

- [ ] Create git tag `v0.1.0`
- [ ] Push to GitHub
- [ ] Create GitHub Release with notes
- [ ] Publish Rust crate to crates.io
- [ ] Publish Python wheel to PyPI
- [ ] Build and push Docker image
- [ ] Deploy documentation to GitHub Pages
- [ ] Announce on social media / forums
- [ ] Update project website

### Post-Release (1 week after)

- [ ] Monitor issue tracker
- [ ] Respond to user questions
- [ ] Collect feedback
- [ ] Plan patch releases if needed
- [ ] Update roadmap based on feedback
- [ ] Write blog post / technical article

---

## Communication Plan

### Announcement Channels

**Internal:**
- Team Slack/Discord
- Project mailing list

**External:**
- GitHub Discussions
- Reddit: r/QuantumComputing, r/rust, r/Python
- Twitter/X: @QDashProject (create account)
- Hacker News (Show HN)
- Quantum Computing Stack Exchange
- Research Gate / arXiv (if applicable)

### Release Notes Template

```markdown
# Q⊗DASH v0.1.0 Released! 🎉

We're excited to announce the first production release of Q⊗DASH (Metatron VM), a comprehensive quantum computing framework built on Rust with Python bindings and advanced auto-tuning.

## 🌟 Highlights

- **Metatron QSO Core:** 13-dimensional quantum state operator with 58.5% information advantage
- **Variational Algorithms:** VQE, QAOA, VQC with multiple ansatz types
- **Quantum Walks:** CTQW with Krylov methods and toolkit functions
- **SCS Auto-Tuner:** Automatic hyperparameter optimization with fixpoint dynamics
- **Python SDK:** Zero-cost bindings for easy integration
- **Production-Ready:** Comprehensive tests, benchmarks, and documentation

## 📦 Installation

**Rust:**
```bash
cargo add metatron-qso
```

**Python:**
```bash
pip install metatron-qso
```

**Docker:**
```bash
docker pull qdash/metatron-qso:0.1.0
```

## 📚 Resources

- [Documentation](https://github.com/LashSesh/qso)
- [Examples](https://github.com/LashSesh/qso/tree/main/metatron_qso_py/examples)
- [Changelog](https://github.com/LashSesh/qso/blob/main/CHANGELOG.md)
- [Release Plan](https://github.com/LashSesh/qso/blob/main/RELEASE_PLAN.md)

## 🤝 Contributing

We welcome contributions! See our GitHub repository for details.

**Made with ❤️ in Rust** | **Powered by Quantum Geometry**
```

---

## Support & Maintenance

### Issue Triage

**Priority Levels:**
1. **Critical:** Crashes, data loss, security vulnerabilities
2. **High:** Major bugs, missing features, performance issues
3. **Medium:** Minor bugs, UX improvements
4. **Low:** Documentation, cosmetic issues

**Response SLA:**
- Critical: 24 hours
- High: 72 hours
- Medium: 1 week
- Low: Best effort

### Patch Release Policy

Release patches (v0.1.1, v0.1.2, etc.) for:
- Security vulnerabilities
- Critical bugs
- Regression fixes
- Documentation corrections

**Frequency:** As needed, but at least monthly review

---

## Future Packaging Plans

### v0.2.0 (Planned)

- [ ] Separate `scs-autotuner` PyPI package
- [ ] Conda-forge packages for both Rust and Python
- [ ] ARM Linux support
- [ ] Enhanced Docker images with GPU support

### v1.0.0 (Future)

- [ ] Debian/Ubuntu .deb packages
- [ ] RPM packages for Fedora/RHEL
- [ ] Homebrew formula for macOS
- [ ] Windows installer (.msi)
- [ ] Snap package (Ubuntu Software)

---

## Appendix

### Useful Commands

**Check versions:**
```bash
cargo tree | grep metatron-qso
pip show metatron-qso
docker images | grep qdash
```

**Clean build:**
```bash
cargo clean
rm -rf target/
find . -type d -name '__pycache__' -exec rm -rf {} +
```

**Verify signatures (future):**
```bash
gpg --verify metatron-qso-0.1.0.tar.gz.asc
```

---

**Document Version:** 1.0
**Last Updated:** 2025-11-16
**Authors:** Q⊗DASH Release Team
