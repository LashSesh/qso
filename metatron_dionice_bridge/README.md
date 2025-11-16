# Metatron-Dionice Bridge

This crate bridges Q⊗DASH's Seraphic Calibration Shell (SCS) with the dioniceOS backend, enabling geometric-cognitive calibration through 4D-5D morphodynamics.

## Architecture

```
SCS Calibration State (ψ, ρ, ω)
         ↓
  DioniceKernel::ingest_state()
         ↓
4D-Trichter Coupling Tick
  (Funnel + Hyperbion + HDAG)
         ↓
  DioniceKernel::step()
         ↓
Calibration Suggestion
```

## Coordinate Mapping

The bridge maps Q⊗DASH's Performance Triplet to dioniceOS's 4D state space:

| SCS Metric | 4D Coordinate | Meaning |
|------------|---------------|---------|
| ρ (rho) | x | Stability mapped to [-1, 1] |
| algorithm | y | Algorithm family encoding (VQE=1, QAOA=2, VQC=3) |
| ω (omega) | z | Efficiency mapped to [-1, 1] |
| ψ (psi) | ψ | Quality (semantic weight) |

The 5th dimension (ω temporal phase) is set to the time step.

## Usage

### Rust

```rust
use metatron_dionice_bridge::{DioniceKernel, QDashCalibrationState};
use std::collections::HashMap;

let mut kernel = DioniceKernel::new();

let state = QDashCalibrationState {
    psi: 0.85,      // Quality
    rho: 0.90,      // Stability
    omega: 0.75,    // Efficiency
    algorithm: "VQE".to_string(),
    extra_params: HashMap::new(),
};

kernel.ingest_state(state)?;
let suggestion = kernel.step()?;

println!("Resonance score: {}", suggestion.resonance_score);
println!("Suggestions: {}", suggestion.notes);
println!("Config updates: {}", suggestion.new_config);
```

### Python Integration (via JSON-RPC or subprocess)

For Python/SCS integration, you can:

1. **Build a CLI wrapper** (recommended):
```rust
// In metatron_dionice_bridge/src/bin/dionice_cli.rs
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

Then from Python:
```python
import subprocess
import json

def get_dionice_suggestion(psi, rho, omega, algorithm):
    state = {
        "psi": psi,
        "rho": rho,
        "omega": omega,
        "algorithm": algorithm,
        "extra_params": {}
    }

    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "dionice_cli"],
        input=json.dumps(state),
        capture_output=True,
        text=True,
        cwd="/path/to/qdash"
    )

    return json.loads(result.stdout)

# In your SCS calibrator:
suggestion = get_dionice_suggestion(
    psi=current_performance.psi,
    rho=current_performance.rho,
    omega=current_performance.omega,
    algorithm=current_config.algorithm
)

print(f"dioniceOS suggests: {suggestion['notes']}")
```

2. **Use PyO3** (for tighter integration):
   - Add `pyo3` feature to `Cargo.toml`
   - Create Python bindings
   - Call directly from Python

## API Reference

### `DioniceKernel`

Main kernel interfacing Q⊗DASH with dioniceOS.

- `new()` - Create with Explore policy
- `with_policy(policy)` - Create with specific policy
- `ingest_state(state)` - Ingest calibration state
- `step()` - Execute coupling tick and get suggestion
- `switch_to_explore/exploit/homeostasis()` - Change policy

### `QDashCalibrationState`

Input state for calibration.

- `psi: f64` - Quality metric (0.0 - 1.0)
- `rho: f64` - Stability metric (0.0 - 1.0)
- `omega: f64` - Efficiency metric (0.0 - 1.0)
- `algorithm: String` - Algorithm family ("VQE", "QAOA", "VQC")
- `extra_params: HashMap<String, f64>` - Additional parameters

### `QDashCalibrationSuggestion`

Output from calibration step.

- `new_config: serde_json::Value` - Suggested configuration updates
- `notes: String` - Human-readable suggestions
- `resonance_score: f64` - Quality score (0.0 - 1.0)
- `regime_change_suggested: bool` - Whether to switch algorithms

## dioniceOS Policies

The kernel supports three operation modes:

- **Explore**: High discovery, low consolidation (for new search spaces)
- **Exploit**: Low discovery, high consolidation (for optimization)
- **Homeostasis**: Adaptive balance (for stable operation)

Switch policies based on SCS phase:
```rust
// Early calibration: explore
kernel.switch_to_explore();

// Mid calibration: exploit
kernel.switch_to_exploit();

// Near fixpoint: homeostasis
kernel.switch_to_homeostasis();
```

## Testing

```bash
cargo test -p metatron_dionice_bridge
```

All tests should pass:
- ✅ Kernel initialization
- ✅ State ingestion
- ✅ Calibration cycle
- ✅ Algorithm encoding/decoding
- ✅ Policy switching
- ✅ Multiple calibration steps

## Integration with SCS

To integrate with the Seraphic Calibration Shell:

1. After each SCS calibration step, extract (ψ, ρ, ω, algorithm)
2. Pass to `DioniceKernel`
3. Get suggestions from dioniceOS 4D-Trichter evolution
4. Use suggestions to inform SCS Double-Kick operator
5. Combine SCS field gradients with dioniceOS funnel dynamics

This provides a **dual-layer calibration**:
- **SCS**: High-level fixpoint-directed configuration evolution
- **dioniceOS**: Low-level geometric-cognitive pattern dynamics

## License

MIT
