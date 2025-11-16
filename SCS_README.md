# Seraphic Calibration Shell (SCS)

The **Seraphic Calibration Shell** is a meta-layer that wraps Q⊗DASH (Metatron VM) with intelligent, fixpoint-directed configuration optimization.

## Quick Start

```bash
# Install dependencies
pip install -r requirements-scs.txt

# Initialize SCS
python -m scs.cli init --benchmark-dir metatron-qso-rs/ci

# Run 5 calibration steps
python -m scs.cli step -n 5

# Check status
python -m scs.cli status

# Export best configuration
python -m scs.cli export -o scs_best_config.json
```

## What is SCS?

SCS enforces a dynamics where every configuration update moves the system monotonically towards fixpoint attractors through:

- **Performance Triplet Φ(c) = (ψ, ρ, ω)**: Quality, stability, efficiency
- **Mandorla Field M(t)**: Accumulates resonance patterns from benchmarks
- **Double-Kick Operator T = Φ_V ∘ Φ_U**: Locally contractive configuration updates
- **Proof-of-Resonance (PoR)**: Ensures quality never degrades
- **CRI Regime Switching**: Controlled transitions between algorithm families

## Documentation

See `docs/seraphic_calibration_shell.md` for complete documentation including:
- Mathematical foundation
- Configuration options
- CI/CD integration
- Debugging and monitoring

## Architecture

```
Benchmarks (JSON) → Performance Triplet → Field Update → Double-Kick
                                                            ↓
                                        Configuration ← PoR Check
                                                            ↓
                                                        CRI Check
```

## Features

- ✅ Non-intrusive opt-in design
- ✅ No modifications to existing code required
- ✅ Can be enabled/disabled via config
- ✅ Preserves all existing functionality
- ✅ GitHub Actions integration included
- ✅ Comprehensive test coverage

Based on: `SeraphicCalibrationModule.pdf`
