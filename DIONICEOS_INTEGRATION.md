# dioniceOS Backend Integration for Q⊗DASH

## Overview

This document describes the integration of **dioniceOS** (4D-5D geometric-cognitive computing platform) as a backend for Q⊗DASH's Seraphic Calibration Shell (SCS).

## What Was Done

### 1. Clean Extraction of dioniceOS Crates

From `dioniceOS-main.zip`, we extracted only the essential Rust crates:

```
external/dioniceos/
├── apollyon_5d/          # 5D dynamical systems framework
│   ├── core/             # Core 5D mathematics
│   ├── bridge/           # Integration layer
│   └── metatron/         # Geometric cognition
├── infinity-ledger/      # Proof-carrying vector ledger
│   ├── mef-core/         # Core MEF pipeline
│   ├── mef-ledger/       # Hash-chained ledger
│   ├── mef-memory/       # Vector memory
│   ├── mef-router/       # S7 routing
│   ├── mef-spiral/       # Snapshot system
│   ├── mef-schemas/      # Data schemas
│   ├── mef-storage/      # Storage backend
│   ├── mef-hdag/         # HDAG structures
│   ├── mef-topology/     # Topology management
│   └── mef-coupling/     # Coupling logic
├── apollyon-mef-bridge/  # Integration bridge
└── overlay/              # Unified 5D cube
    └── unified_5d_cube/
```

**Documentation**: Only 2 high-level docs in `docs/dioniceos/`:
- `README.md` - Comprehensive overview
- `QUICK_START.md` - Getting started guide

All development notes and legacy markdown files were excluded.

### 2. Workspace Configuration

Created a root `Cargo.toml` workspace that includes:
- Q⊗DASH core (`metatron-qso-rs`)
- dioniceOS crates (all extracted crates)
- Bridge crate (`metatron_dionice_bridge`)

The workspace builds successfully with all dependencies resolved.

### 3. Bridge Crate: `metatron_dionice_bridge`

Created a clean encapsulation layer between SCS and dioniceOS:

#### API

```rust
pub struct DioniceKernel {
    // Encapsulates: Funnel, HDAG, Hyperbion, Policy
}

pub struct QDashCalibrationState {
    pub psi: f64,      // Quality (0-1)
    pub rho: f64,      // Stability (0-1)
    pub omega: f64,    // Efficiency (0-1)
    pub algorithm: String,  // "VQE", "QAOA", "VQC"
}

pub struct QDashCalibrationSuggestion {
    pub new_config: serde_json::Value,
    pub notes: String,
    pub resonance_score: f64,
    pub regime_change_suggested: bool,
}

impl DioniceKernel {
    pub fn new() -> Self;
    pub fn ingest_state(&mut self, state: QDashCalibrationState) -> Result<()>;
    pub fn step(&mut self) -> Result<QDashCalibrationSuggestion>;
    pub fn switch_to_explore/exploit/homeostasis(&mut self);
}
```

#### Coordinate Mapping

The bridge maps SCS metrics to dioniceOS 4D state space:

| SCS Metric | 4D Coord | Mapping |
|------------|----------|---------|
| ρ (stability) | x | (rho - 0.5) × 2 → [-1, 1] |
| algorithm | y | VQE=1, QAOA=2, VQC=3 |
| ω (efficiency) | z | (omega - 0.5) × 2 → [-1, 1] |
| ψ (quality) | ψ | Direct (semantic weight) |

The 5th dimension (ω temporal phase) is set to the time step.

#### How It Works

1. **Ingest**: SCS state (ψ, ρ, ω, algorithm) → 4D state space
2. **Evolve**: 4D-Trichter coupling tick:
   - Lift 4D → 5D
   - Hyperbion absorption (morphodynamic coupling)
   - HDAG relaxation (resonance field)
   - Funnel advection (pattern evolution)
   - Project 5D → 4D
3. **Analyze**: Extract suggestions from state evolution:
   - Stability changes → ensemble size recommendations
   - Quality changes → algorithm tuning
   - Efficiency changes → computational budget
   - Large regime shifts → algorithm family switch

### 4. Testing

All tests pass (7/7):
- ✅ Kernel initialization
- ✅ State ingestion
- ✅ Full calibration cycle
- ✅ Algorithm encoding/decoding
- ✅ Policy switching
- ✅ Multiple calibration steps

### 5. Documentation

Created comprehensive documentation:
- `metatron_dionice_bridge/README.md` - API reference and integration guide
- `docs/dioniceos/README.md` - dioniceOS overview (from upstream)
- `docs/dioniceos/QUICK_START.md` - Quick start guide (from upstream)
- This file (`DIONICEOS_INTEGRATION.md`) - Integration summary

## Integration with SCS

### Current State

The bridge is ready for integration. SCS currently operates independently in Python.

### Integration Approach

Two options:

#### Option 1: CLI Wrapper (Recommended for MVP)

Create a simple CLI tool:

```rust
// metatron_dionice_bridge/src/bin/dionice_cli.rs
use metatron_dionice_bridge::*;
use std::io::{self, Read};

fn main() -> anyhow::Result<()> {
    let mut kernel = DioniceKernel::new();
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;

    let state: QDashCalibrationState = serde_json::from_str(&buffer)?;
    kernel.ingest_state(state)?;
    let suggestion = kernel.step()?;

    println!("{}", serde_json::to_string(&suggestion)?);
    Ok(())
}
```

Then from SCS Python:

```python
def get_dionice_suggestion(psi, rho, omega, algorithm):
    state = {"psi": psi, "rho": rho, "omega": omega, "algorithm": algorithm, "extra_params": {}}
    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "dionice_cli"],
        input=json.dumps(state), capture_output=True, text=True
    )
    return json.loads(result.stdout)

# In SCS calibrator.py after step 3 (Double-Kick):
dionice_suggestion = get_dionice_suggestion(
    self.current_performance.psi,
    self.current_performance.rho,
    self.current_performance.omega,
    self.current_config.algorithm
)

# Combine with SCS field gradients
notes.append(f"dioniceOS: {dionice_suggestion['notes']}")
```

#### Option 2: PyO3 Bindings (For Production)

Add PyO3 support for direct Python calls:

```toml
[dependencies]
pyo3 = { version = "0.21", features = ["extension-module"] }
```

```rust
use pyo3::prelude::*;

#[pyclass]
struct PyDioniceKernel {
    kernel: DioniceKernel,
}

#[pymethods]
impl PyDioniceKernel {
    #[new]
    fn new() -> Self { Self { kernel: DioniceKernel::new() } }

    fn step(&mut self, psi: f64, rho: f64, omega: f64, algorithm: String) -> PyResult<String> {
        // ...
    }
}

#[pymodule]
fn metatron_dionice_bridge(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDioniceKernel>()?;
    Ok(())
}
```

Then from Python:
```python
from metatron_dionice_bridge import PyDioniceKernel

kernel = PyDioniceKernel()
suggestion = kernel.step(psi, rho, omega, algorithm)
```

## Dual-Layer Calibration

With this integration, Q⊗DASH now has **two complementary calibration layers**:

```
┌─────────────────────────────────────────────────────┐
│  SCS (Seraphic Calibration Shell)                   │
│  - Fixpoint-directed configuration evolution        │
│  - Mandorla field accumulation                      │
│  - Double-Kick operator (contractive updates)       │
│  - Proof-of-Resonance validation                    │
│  - CRI regime switching                             │
└────────────────┬────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────┐
│  dioniceOS Backend (via metatron_dionice_bridge)    │
│  - 4D-5D geometric-cognitive dynamics               │
│  - Funnel graph (Hebbian pattern evolution)         │
│  - Hyperbion (morphodynamic coupling)               │
│  - HDAG (5D resonance field)                        │
│  - Explore/Exploit/Homeostasis policies             │
└─────────────────────────────────────────────────────┘
```

**Synergy**:
- SCS provides high-level guidance toward fixpoints
- dioniceOS provides low-level geometric pattern dynamics
- Suggestions from both layers can be combined to inform configuration updates

## Verification

```bash
# Build entire workspace
cargo build --workspace

# Test bridge
cargo test -p metatron_dionice_bridge

# Run Q⊗DASH benchmarks (existing)
cd metatron-qso-rs
cargo test --release

# Run SCS (existing)
python -m scs.cli status
```

All builds and tests pass ✅

## Repository Structure (After Integration)

```
qdash/
├── Cargo.toml                      # Root workspace
├── metatron-qso-rs/                # Q⊗DASH core
├── metatron_dionice_bridge/        # NEW: Bridge crate
│   ├── src/lib.rs
│   └── README.md
├── external/dioniceos/             # NEW: dioniceOS crates
│   ├── apollyon_5d/
│   ├── infinity-ledger/
│   ├── apollyon-mef-bridge/
│   └── overlay/
├── scs/                            # Seraphic Calibration Shell (Python)
├── docs/dioniceos/                 # NEW: Curated docs (2 files only)
│   ├── README.md
│   └── QUICK_START.md
├── DIONICEOS_INTEGRATION.md        # NEW: This file
└── dioniceOS-main.zip              # Original archive (can be removed after verification)
```

## Next Steps

### Immediate (To Complete Integration)

1. **Create CLI wrapper** (Option 1 above) for SCS to call
2. **Modify SCS** `calibrator.py` to invoke dioniceOS after Double-Kick step
3. **Test end-to-end** calibration with both layers active
4. **Document integration** in SCS README

### Future Enhancements

1. **PyO3 bindings** for tighter Python integration
2. **Async support** for non-blocking calibration
3. **Persistent state** save/load between sessions
4. **Visualization** of 4D-5D state evolution
5. **Hyperparameter tuning** for Funnel/HDAG/Hyperbion
6. **Benchmarking** calibration quality improvements

## Key Achievements

✅ Clean extraction of dioniceOS (code only, no markdown noise)
✅ Workspace builds successfully with all crates
✅ Bridge crate with clean, documented API
✅ Comprehensive tests (7/7 passing)
✅ Minimal documentation surface (2 curated docs)
✅ Zero modifications to existing Q⊗DASH functionality
✅ Encapsulated integration (additive only)
✅ Ready for SCS integration

## License

All dioniceOS crates: MIT
Bridge crate: MIT
Integration with Q⊗DASH: MIT

---

**Integration completed**: 2025-11-13
**Status**: Ready for SCS integration ✨
