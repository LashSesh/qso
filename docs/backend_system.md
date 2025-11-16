# Q⊗DASH Quantum Backend System

This document describes the quantum backend abstraction system in Q⊗DASH, which provides a unified interface for executing quantum circuits across multiple providers.

## Overview

The backend system provides a provider-agnostic way to execute quantum circuits. You can write your algorithm once and run it on different backends (local simulator, IBM Quantum, Azure Quantum, etc.) without changing your code.

## Architecture

### Core Components

1. **`QuantumBackend` Trait** (`metatron_backend/src/backends/mod.rs`)
   - Unified interface for all quantum backends
   - Methods: `info()`, `run_circuit()`, `can_run()`

2. **`MetatronCircuit`** (`metatron_backend/src/circuit.rs`)
   - Backend-agnostic circuit representation
   - Supports common gates: H, X, Y, Z, S, T, RX, RY, RZ, CNOT, CZ, SWAP, Toffoli
   - Builder pattern for easy construction

3. **`BackendRegistry`** (`metatron_backend/src/registry.rs`)
   - Manages available backends
   - Selects appropriate backend based on mode and requirements
   - Three modes: SimulationOnly, QpuEnabledAuto, ForceProvider

4. **Backend Implementations**
   - **LocalSimulatorBackend**: Wraps existing Q⊗DASH 13-dimensional simulator
   - **IbmQuantumBackend**: Integration with IBM Quantum Platform (with safety features)

## Safety-First Design

### Default Behavior

**The system defaults to simulation-only mode with NO QPU ACCESS.**

This ensures:
- No accidental consumption of QPU time
- No unexpected costs
- Safe for development and testing

### IBM Backend Safety

The IBM backend has three modes:

1. **Disabled** (DEFAULT)
   - Backend cannot execute circuits
   - Returns error if circuit execution is attempted
   - Zero risk of QPU usage

2. **Dry-Run**
   - Validates and logs circuits
   - Returns stubbed results for testing
   - NO QPU time consumed
   - Perfect for development

3. **Enabled**
   - Executes on real IBM hardware
   - **CONSUMES QPU TIME**
   - Only use when explicitly authorized

## Quick Start

### 1. Using the Local Simulator

```rust
use metatron_backend::prelude::*;

// Create local simulator backend
let backend = LocalSimulatorBackend::new();

// Build a circuit
let circuit = MetatronCircuit::new(2)
    .h(0)
    .cnot(0, 1)
    .measure_all();

// Execute
let result = backend.run_circuit(&circuit, 1000)?;
println!("Counts: {:?}", result.counts);
```

### 2. Using the Backend Registry

```rust
use metatron_backend::prelude::*;

// Create registry (defaults to simulation-only)
let mut registry = BackendRegistry::new();

// Register backends
registry.register(
    "local_sim".to_string(),
    Box::new(LocalSimulatorBackend::new())
)?;

// Select backend for circuit
let circuit = MetatronCircuit::new(2).h(0).cnot(0, 1);
let backend = registry.select_backend_for(&circuit)?;

// Execute
let result = backend.run_circuit(&circuit, 1000)?;
```

### 3. Using IBM Backend (Dry-Run)

```rust
use metatron_backend::{IbmQuantumBackend, IbmConfig, IbmMode};

// Create IBM backend in dry-run mode
let config = IbmConfig {
    token: None,  // No token needed for dry-run
    backend_name: "ibm_kyoto".to_string(),
    mode: IbmMode::DryRun,
    max_shots: 8192,
};

let backend = IbmQuantumBackend::new(config)?;

// Execute (logs circuit, returns stubbed result)
let circuit = MetatronCircuit::new(2).h(0).measure_all();
let result = backend.run_circuit(&circuit, 100)?;
```

## Configuration

### Environment Variables

Configuration is done via environment variables. See `.env.example` for all options.

#### Essential IBM Configuration

```bash
# REQUIRED for enabled mode
IBM_QUANTUM_TOKEN=your_api_token_here

# Backend to use
IBM_BACKEND_NAME=ibm_kyoto

# Mode (IMPORTANT!)
IBM_BACKEND_MODE=disabled  # or dry-run, or enabled

# Max shots per job
IBM_MAX_SHOTS=8192
```

#### Backend Registry Mode

```bash
# simulation_only (DEFAULT - SAFE)
BACKEND_REGISTRY_MODE=simulation_only

# qpu_enabled_auto (allows QPUs if available)
BACKEND_REGISTRY_MODE=qpu_enabled_auto

# force_provider (only use specific provider)
BACKEND_REGISTRY_MODE=force_provider:ibm
```

### Loading Configuration

```rust
// From environment variables
let backend = IbmQuantumBackend::from_env()?;

// Or manually
let config = IbmConfig::from_env()?;
let backend = IbmQuantumBackend::new(config)?;
```

## Backend Modes

### BackendMode (Registry)

Controls which backends are eligible for selection:

```rust
pub enum BackendMode {
    SimulationOnly,              // Only simulators (DEFAULT)
    QpuEnabledAuto,              // Prefer simulators, allow QPUs
    ForceProvider(String),       // Only specific provider
}
```

### IbmMode (IBM Backend)

Controls IBM backend execution:

```rust
pub enum IbmMode {
    Disabled,  // No execution (DEFAULT)
    DryRun,    // Validate only, no QPU time
    Enabled,   // Real hardware (CONSUMES QPU TIME)
}
```

## Telemetry Integration

The backend system is integrated with the telemetry API:

### API Response

```json
GET /api/status

{
  "algorithm": "VQE",
  "mode": "Explore",
  "psi": 0.85,
  "rho": 0.90,
  "omega": 0.75,
  "backend_info": {
    "provider": "local",
    "name": "local_sim",
    "num_qubits": 13,
    "is_simulator": true,
    "mode": null
  },
  "available_backends": ["local_sim"],
  ...
}
```

### Dashboard UI

The web dashboard displays:
- Current backend name and type (SIMULATOR/QPU)
- Number of qubits
- Backend mode (for QPUs)

## Migration Guide

### For Existing Q⊗DASH Code

To migrate existing code to use the backend abstraction:

**Before:**
```rust
// Direct simulator usage
let state = QuantumState::basis_state(0)?;
let result = simulate_circuit(circuit);
```

**After:**
```rust
// Use backend abstraction
let backend = LocalSimulatorBackend::new();
let result = backend.run_circuit(&circuit, shots)?;
```

Benefits:
- Same code works with any backend
- Easy to switch between simulator and QPU
- Built-in safety checks

## Future Backends

The system is designed to easily add new backends:

- **Azure Quantum**: Microsoft's quantum cloud service
- **IonQ**: Trapped-ion quantum computers
- **Rigetti**: Superconducting quantum processors
- **AWS Braket**: Amazon's quantum service

To add a new backend, implement the `QuantumBackend` trait and handle circuit translation.

## Best Practices

### Development

1. **Always start with SimulationOnly mode**
   ```bash
   BACKEND_REGISTRY_MODE=simulation_only
   IBM_BACKEND_MODE=disabled
   ```

2. **Use dry-run for testing IBM integration**
   ```bash
   IBM_BACKEND_MODE=dry-run
   ```

3. **Validate circuits before QPU execution**
   - Run on simulator first
   - Check circuit depth and gate count
   - Verify measurement results make sense

### Production

1. **Explicit QPU Authorization**
   - Only enable QPU access when authorized
   - Document who approved QPU usage
   - Track QPU time consumption

2. **Cost Management**
   - Set reasonable `max_shots` limits
   - Monitor job execution
   - Use dry-run mode for pre-flight checks

3. **Error Handling**
   - Handle backend unavailability gracefully
   - Implement retry logic for transient failures
   - Log all QPU executions

## Security

### API Token Handling

**NEVER commit API tokens to git!**

```bash
# .env is in .gitignore
echo "IBM_QUANTUM_TOKEN=your_token" >> .env

# Use .env.example for documentation
cp .env.example .env
```

### Token Storage

- Store tokens in environment variables or secure vaults
- Use separate tokens for dev/staging/production
- Rotate tokens regularly
- Revoke tokens when no longer needed

## Troubleshooting

### "Backend is disabled"

**Problem**: `IBM backend 'ibm_kyoto' is disabled`

**Solution**: Set `IBM_BACKEND_MODE`:
```bash
IBM_BACKEND_MODE=dry-run  # For testing
IBM_BACKEND_MODE=enabled  # For real hardware (requires token)
```

### "Token is required"

**Problem**: `token is required for enabled mode`

**Solution**: Set your IBM Quantum token:
```bash
IBM_QUANTUM_TOKEN=your_actual_token_here
IBM_BACKEND_MODE=enabled
```

### "No available backend found"

**Problem**: Registry cannot find suitable backend

**Solution**: Check that:
1. Backend is registered in the registry
2. Backend has enough qubits for your circuit
3. Backend mode allows the backend type (simulator/QPU)

### Build Errors

**Problem**: `cannot find crate metatron_backend`

**Solution**: Ensure workspace includes the crate:
```toml
# Cargo.toml
[workspace]
members = [
    "metatron_backend",
    ...
]
```

## API Reference

### MetatronCircuit Builder

```rust
let circuit = MetatronCircuit::new(num_qubits)
    .h(qubit)                          // Hadamard
    .x(qubit)                          // Pauli X
    .y(qubit)                          // Pauli Y
    .z(qubit)                          // Pauli Z
    .s(qubit)                          // S gate
    .t(qubit)                          // T gate
    .rx(qubit, theta)                  // Rotation X
    .ry(qubit, theta)                  // Rotation Y
    .rz(qubit, theta)                  // Rotation Z
    .u(qubit, theta, phi, lambda)      // Arbitrary unitary
    .cnot(control, target)             // CNOT
    .cz(control, target)               // Controlled-Z
    .swap(qubit1, qubit2)              // SWAP
    .measure(qubit)                    // Measure single qubit
    .measure_all();                    // Measure all qubits
```

### BackendCapabilities

```rust
pub struct BackendCapabilities {
    pub provider: String,       // "local", "ibm", "azure", etc.
    pub name: String,           // "local_sim", "ibm_kyoto", etc.
    pub num_qubits: u32,        // Number of qubits
    pub is_simulator: bool,     // true for simulators
    pub max_shots: Option<u32>, // Max shots per job
    pub available: bool,        // Currently available?
    pub metadata: Value,        // Additional metadata
}
```

### MeasurementResult

```rust
pub struct MeasurementResult {
    pub counts: HashMap<String, u64>,  // Bitstring -> count
    pub shots: u32,                     // Total shots
    pub execution_time_ms: Option<f64>, // Execution time
    pub backend_name: String,           // Backend used
}

// Helper methods
result.probability("00");           // Get probability of outcome
result.most_likely_outcome();       // Get most frequent result
result.expectation_z(qubit);        // Compute <Z> on qubit
```

## Additional Resources

- [IBM Quantum Documentation](https://docs.quantum.ibm.com/)
- [Qiskit Runtime API](https://docs.quantum.ibm.com/api/qiskit-ibm-runtime)
- [Q⊗DASH Architecture](./architecture.md)
- [Seraphic Calibration Shell Guide](./scs_guide.md)

## Support

For questions or issues:
1. Check this documentation
2. Review `.env.example` for configuration
3. Check logs (set `RUST_LOG=debug` for verbose output)
4. Open an issue on GitHub
