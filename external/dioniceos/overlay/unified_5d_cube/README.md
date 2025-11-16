# Unified 5D Cube Overlay

Non-invasive overlay integrating APOLLYON-5D, 4D-Trichter, and MEF-Core into a unified execution pipeline.

## Quick Start

```rust
use unified_5d_cube::{InterlockConfig, InterlockAdapter, tick_5d_cube, ShadowController, ActivationCriteria};
use core_5d::State5D;

// 1. Create configuration
let config = InterlockConfig::default();

// 2. Create interlock adapter
let mut adapter = InterlockAdapter::new(config);

// 3. Create shadow controller
let criteria = ActivationCriteria::default();
let mut shadow = ShadowController::new(criteria, 10);

// 4. Execute ticks
let state = State5D::from_array([1.0, 0.0, 0.0, 0.5, 0.3]);
let result = tick_5d_cube(&mut adapter, &state, None, 0.0, 0);

// 5. Update shadow controller
shadow.update(result.metrics, result.gate_decision);

// 6. Check if active
if shadow.is_active() {
    println!("System activated!");
}
```

## Features

- **Non-invasive**: Uses only public APIs from existing components
- **Shadow Mode**: Default mode with no side effects
- **Activation Criteria**: Automatic activation when stability criteria met
- **Metrics Collection**: CSV/JSON export of all metrics
- **Deterministic**: Reproducible with fixed seeds
- **✨ Future Extensions**: All extensions now implemented and integrated!
  - ✅ Full HDAG relaxation with Hyperbion fields
  - ✅ Complete Funnel operations (split/merge/prune)
  - ✅ Actual MEF Ledger writes
  - ✅ Full 8D vector pipeline
  - ✅ Metatron S7 router integration

**See [FUTURE_EXTENSIONS.md](FUTURE_EXTENSIONS.md) for detailed documentation of all extensions.**

## Architecture

See [interlock_map.md](interlock_map.md) for detailed architecture documentation.

## Components

### InterlockAdapter

Connects APOLLYON, Trichter, and MEF through public APIs:

- `apollyon_to_trichter()` - STATE_IN conversion
- `compute_guidance()` - FIELD_IO computation
- `evaluate_gate()` - GATE evaluation
- `condense()` - CONDENSE operation
- `prepare_commit()` - EVENT_OUT preparation

### tick_5d_cube()

Main pipeline executing 6 phases:

1. STATE_IN: Convert APOLLYON → Trichter
2. Solve/Relax: (conceptual, state already solved)
3. FIELD_IO: Compute guidance field ∇Φ
4. GATE: Evaluate Proof-of-Resonance
5. CONDENSE: Apply coagulation
6. Commit: Prepare commit data (if FIRE)

### MetricsCollector

Collects and exports metrics:

- BI (Betti number)
- ΔF (energy delta)
- W2_step (Wasserstein distance)
- λ_gap (spectral gap)
- S_mand (Mandorla score)
- Duty/PoR (PoR validity)

Export formats: CSV, JSON

### ShadowController

Manages shadow mode and activation:

- Starts in shadow mode (no side effects)
- Tracks activation criteria over windows
- Auto-activates when criteria met
- Auto-rollback on instability

## Configuration

```rust
let config = InterlockConfig {
    seed: 42,                      // Deterministic seed
    gate_phi_threshold: 0.5,       // Minimum alignment
    gate_delta_pi_max: 0.1,        // Maximum path invariance
    enable_logging: true,          // Enable logging
    shadow_mode: true,             // Start in shadow
};
```

## Activation Criteria

```rust
let criteria = ActivationCriteria {
    window_count: 3,               // Consecutive windows required
    max_delta_f: 0.0,              // ΔF threshold
    min_gate_stability: 0.8,       // Minimum gate stability
    min_coherence: 0.7,            // Minimum coherence
};
```

## Examples

See `examples/` directory for complete examples:

- `simple.rs` - Basic usage
- `metrics.rs` - Metrics collection
- `shadow.rs` - Shadow mode activation

## Testing

```bash
cargo test --package unified-5d-cube
```

## Feature Flags

- `activate` (default: OFF) - Enable shadow → active transition

```toml
[dependencies]
unified-5d-cube = { path = "overlay/unified_5d_cube", features = ["activate"] }
```

## Documentation

- [interlock_map.md](interlock_map.md) - Architecture and interlock points
- API docs: `cargo doc --open`

## License

MIT
