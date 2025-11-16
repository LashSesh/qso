# PyO3 Integration Guide

This guide explains how to use the PyO3 bindings for `metatron_dionice_bridge` to integrate dioniceOS with the Python-based Seraphic Calibration Shell (SCS).

## Overview

The `metatron_dionice_bridge` crate now includes optional PyO3 bindings that allow direct Python access to the dioniceOS backend without subprocess overhead.

## Building the Python Module

### Prerequisites

```bash
pip install maturin
```

### Build for Development

```bash
cd metatron_dionice_bridge
maturin develop --features python
```

This compiles and installs the module in your current Python environment.

### Build for Production

```bash
maturin build --release --features python
```

This creates a wheel file in `target/wheels/`.

## Python API

### Basic Usage

```python
from metatron_dionice_bridge import DioniceKernel

# Create kernel
kernel = DioniceKernel()

# Run calibration step
result = kernel.step(
    psi=0.85,      # Quality metric
    rho=0.90,      # Stability metric
    omega=0.75,    # Efficiency metric
    algorithm="VQE"
)

# Access results
print(f"Notes: {result.notes}")
print(f"Resonance score: {result.resonance_score}")
print(f"Config updates: {result.get_config()}")

if result.regime_change_suggested:
    print("dioniceOS suggests changing algorithm family!")
```

### Integration with SCS

#### Option 1: Direct Integration

Modify `scs/calibrator.py`:

```python
from metatron_dionice_bridge import DioniceKernel

class SeraphicCalibrator:
    def __init__(self, config: Optional[CalibratorConfig] = None):
        # ... existing init code ...

        # Add dioniceOS kernel
        self.dionice_kernel = DioniceKernel()

    def calibration_step(self) -> Dict[str, Any]:
        # ... existing calibration code ...

        # Step 3.5: Get dioniceOS feedback
        dionice_result = self.dionice_kernel.step(
            psi=self.current_performance.psi,
            rho=self.current_performance.rho,
            omega=self.current_performance.omega,
            algorithm=self.current_config.algorithm
        )

        # Incorporate dioniceOS suggestions into SCS decision
        step_result['dionice_feedback'] = {
            'notes': dionice_result.notes,
            'resonance_score': dionice_result.resonance_score,
            'config': dionice_result.get_config()
        }

        # Optionally: use dionice suggestions to adjust candidate_config
        if dionice_result.regime_change_suggested:
            dionice_config = dionice_result.get_config()
            if 'suggested_algorithm' in dionice_config:
                # Consider dioniceOS recommendation
                candidate_config.algorithm = dionice_config['suggested_algorithm']

        # ... continue with PoR check, etc. ...

        return step_result
```

#### Option 2: Parallel Analysis

Use dioniceOS as a secondary analysis tool:

```python
# After SCS makes its decision
scs_candidate = self.double_kick.apply(...)

# Get dioniceOS opinion
dionice_result = self.dionice_kernel.step(
    psi=candidate_performance.psi,
    rho=candidate_performance.rho,
    omega=candidate_performance.omega,
    algorithm=candidate_config.algorithm
)

# Compare SCS and dioniceOS recommendations
print(f"SCS suggests: {scs_candidate.to_dict()}")
print(f"dioniceOS suggests: {dionice_result.notes}")

# Make final decision based on both
```

### Policy Management

The dioniceOS kernel supports three operation modes:

```python
kernel = DioniceKernel()

# Explore mode: high discovery, low consolidation
kernel.switch_to_explore()

# Exploit mode: low discovery, high consolidation
kernel.switch_to_exploit()

# Homeostasis mode: adaptive balance
kernel.switch_to_homeostasis()

# Check funnel metrics
print(f"Funnel density: {kernel.funnel_density()}")
print(f"Funnel nodes: {kernel.funnel_node_count()}")
```

### Complete Example

```python
#!/usr/bin/env python3
"""
Example: Dual-layer calibration with SCS + dioniceOS
"""

from metatron_dionice_bridge import DioniceKernel
from scs.calibrator import SeraphicCalibrator, CalibratorConfig
from scs.config import Configuration

def main():
    # Initialize both systems
    scs = SeraphicCalibrator(CalibratorConfig())
    dionice = DioniceKernel()

    # Start with explore mode
    dionice.switch_to_explore()

    # Initialize calibration
    initial_config = Configuration.default()
    scs.initialize(initial_config)

    # Run dual-layer calibration
    for step in range(10):
        print(f"\n=== Calibration Step {step + 1} ===")

        # SCS calibration step
        scs_result = scs.calibration_step()

        # Get dioniceOS feedback
        dionice_result = dionice.step(
            psi=scs.current_performance.psi,
            rho=scs.current_performance.rho,
            omega=scs.current_performance.omega,
            algorithm=scs.current_config.algorithm
        )

        # Print results
        print(f"SCS: {scs_result['por_result']}")
        print(f"dioniceOS: {dionice_result.notes}")
        print(f"Resonance: {dionice_result.resonance_score:.4f}")

        # Switch to exploit mode after initial exploration
        if step == 5:
            print("Switching to exploit mode...")
            dionice.switch_to_exploit()

    # Get final configuration
    final_config = scs.get_best_configuration()
    print(f"\nFinal config: {final_config.to_dict()}")

    # Save results
    scs.save_state()
    scs.save_history()

if __name__ == "__main__":
    main()
```

## Performance Considerations

### Overhead

PyO3 bindings add minimal overhead:
- Function call: ~50-100 ns
- Data marshalling: ~1-5 μs per call
- Total per-step overhead: < 10 μs

This is negligible compared to calibration runtime (seconds to minutes).

### Memory

Each `DioniceKernel` instance maintains:
- Funnel graph (grows with exploration)
- HDAG field (bounded)
- State history (bounded to 1000 points)

Typical memory usage: 1-10 MB per kernel.

### Concurrency

PyO3 releases the GIL for expensive operations, allowing:
- Multiple kernels in parallel
- Non-blocking calibration
- Concurrent SCS + dioniceOS execution

```python
from concurrent.futures import ThreadPoolExecutor

kernels = [DioniceKernel() for _ in range(4)]

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(k.step, psi=0.85, rho=0.90, omega=0.75, algorithm="VQE")
        for k in kernels
    ]
    results = [f.result() for f in futures]
```

## Troubleshooting

### Import Error

```python
ImportError: No module named 'metatron_dionice_bridge'
```

**Solution**: Build and install the module:
```bash
cd metatron_dionice_bridge
maturin develop --features python
```

### Version Mismatch

```python
ImportError: ... version mismatch ...
```

**Solution**: Rebuild after updating Rust code:
```bash
maturin develop --features python --force
```

### Performance Issues

If dioniceOS calls are slow:
1. Use `--release` build: `maturin develop --release --features python`
2. Reduce funnel history: modify kernel parameters
3. Profile with `py-spy` to identify bottlenecks

## Advanced: Custom Policies

You can extend dioniceOS policies from Python (future feature):

```python
# Future API (not yet implemented)
from metatron_dionice_bridge import DioniceKernel, CustomPolicy

class MyPolicy(CustomPolicy):
    def get_params(self):
        return {
            'alpha_hebb': 0.3,
            'decay': 0.02,
            'merge_threshold': 0.85
        }

kernel = DioniceKernel()
kernel.set_custom_policy(MyPolicy())
```

## Testing

Test the Python bindings:

```python
import unittest
from metatron_dionice_bridge import DioniceKernel

class TestDioniceKernel(unittest.TestCase):
    def test_basic_step(self):
        kernel = DioniceKernel()
        result = kernel.step(0.85, 0.90, 0.75, "VQE")

        self.assertIsInstance(result.notes, str)
        self.assertGreaterEqual(result.resonance_score, 0.0)
        self.assertLessEqual(result.resonance_score, 1.0)

    def test_policy_switching(self):
        kernel = DioniceKernel()
        kernel.switch_to_explore()
        kernel.switch_to_exploit()
        kernel.switch_to_homeostasis()
        # Should not raise

    def test_multiple_steps(self):
        kernel = DioniceKernel()
        for i in range(10):
            result = kernel.step(
                0.80 + i * 0.01,
                0.85 + i * 0.005,
                0.75 + i * 0.01,
                "VQE"
            )
            self.assertGreater(len(result.notes), 0)

if __name__ == '__main__':
    unittest.main()
```

## See Also

- [Bridge README](../metatron_dionice_bridge/README.md) - Rust API documentation
- [dioniceOS Integration](../DIONICEOS_INTEGRATION.md) - Overall integration architecture
- [SCS README](../SCS_README.md) - Seraphic Calibration Shell
- [PyO3 Documentation](https://pyo3.rs/) - Official PyO3 guide
