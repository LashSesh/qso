# Q⊗DASH Telemetry & Dashboard

Complete guide to the web dashboard and PyO3 integration for Q⊗DASH.

## Quick Links

- [Web Dashboard README](../metatron_telemetry/README.md) - Dashboard usage and API reference
- [PyO3 Integration Guide](./pyo3_integration.md) - Python bindings for dioniceOS
- [dioniceOS Integration](../DIONICEOS_INTEGRATION.md) - Overall backend integration

## Overview

This document provides a high-level overview of the telemetry and control infrastructure for Q⊗DASH, including:

1. **Web Dashboard**: Real-time monitoring and control interface
2. **PyO3 Bindings**: Direct Python access to dioniceOS backend
3. **Integration Strategies**: How to connect all components

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  User Interface Layer                                    │
│  ┌──────────────┐              ┌──────────────────────┐ │
│  │ Web Dashboard│              │ Python CLI/Scripts   │ │
│  │ (Browser)    │              │ (PyO3 bindings)      │ │
│  └──────┬───────┘              └──────────┬───────────┘ │
└─────────┼──────────────────────────────────┼────────────┘
          │ HTTP/JSON                        │ Direct FFI
┌─────────▼──────────────────────────────────▼────────────┐
│  Telemetry & Control Layer                              │
│  ┌────────────────────┐    ┌───────────────────────┐   │
│  │ metatron_telemetry │    │metatron_dionice_bridge│   │
│  │ • HTTP API         │    │ • PyO3 bindings       │   │
│  │ • State management │    │ • Rust ↔ Python       │   │
│  │ • Job tracking     │    │                       │   │
│  └────────┬───────────┘    └───────────┬───────────┘   │
└───────────┼─────────────────────────────┼───────────────┘
            │                             │
┌───────────▼─────────────────────────────▼───────────────┐
│  Core Q⊗DASH Layer                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  SCS (Python)│  │  dioniceOS   │  │ metatron-qso │  │
│  │  Calibration │  │  4D-Trichter │  │ Quantum Core │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────┘
```

## Components

### 1. Web Dashboard (`metatron_telemetry`)

**Purpose**: Real-time monitoring and control via web browser

**Features**:
- Live metrics display (ψ, ρ, ω)
- Job status tracking
- Historical charts
- One-click calibration start
- Backend health indicators

**Tech Stack**:
- **Backend**: Axum (Rust web framework)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Charts**: Chart.js
- **Updates**: Automatic polling (5s interval)

**Access**: http://localhost:8080 (default)

**See**: [Dashboard README](../metatron_telemetry/README.md)

### 2. PyO3 Bindings (`metatron_dionice_bridge`)

**Purpose**: Direct Python access to dioniceOS backend

**Features**:
- Zero-copy Python ↔ Rust calls
- Minimal overhead (<10 μs per call)
- GIL management for concurrency
- Pythonic API

**Usage**:
```python
from metatron_dionice_bridge import DioniceKernel

kernel = DioniceKernel()
result = kernel.step(psi=0.85, rho=0.90, omega=0.75, algorithm="VQE")
print(result.notes)
```

**See**: [PyO3 Integration Guide](./pyo3_integration.md)

### 3. HTTP API (`/api/*`)

**Purpose**: Programmatic access for automation and monitoring

**Endpoints**:
- `GET /api/status` - Current system state
- `GET /api/jobs` - Recent jobs
- `GET /api/history` - Metrics time series
- `POST /api/control/start_calibration` - Trigger run

**Example**:
```bash
curl http://localhost:8080/api/status | jq
```

**See**: [API Reference](../metatron_telemetry/README.md#api-endpoints)

## Getting Started

### Option A: Web Dashboard Only

For visual monitoring and manual control:

```bash
# 1. Start the telemetry server
cargo run --release --bin metatron_telemetry

# 2. Open dashboard in browser
open http://localhost:8080
```

### Option B: PyO3 Integration

For programmatic control and SCS integration:

```bash
# 1. Build Python module
cd metatron_dionice_bridge
pip install maturin
maturin develop --features python --release

# 2. Use from Python
python3 -c "from metatron_dionice_bridge import DioniceKernel; print(DioniceKernel())"
```

### Option C: Full Stack

Both web dashboard and Python integration:

```bash
# Terminal 1: Start telemetry server
cargo run --release --bin metatron_telemetry

# Terminal 2: Build Python bindings
cd metatron_dionice_bridge
maturin develop --features python --release

# Terminal 3: Run SCS with dioniceOS
cd scs
python -m scs.cli step -n 5
```

## Integration Patterns

### Pattern 1: SCS with dioniceOS via PyO3

**Use case**: Tight integration, best performance

```python
# scs/calibrator.py
from metatron_dionice_bridge import DioniceKernel

class SeraphicCalibrator:
    def __init__(self, config=None):
        # ... existing init ...
        self.dionice = DioniceKernel()

    def calibration_step(self):
        # ... SCS logic ...

        # Get dioniceOS recommendation
        dionice_result = self.dionice.step(
            psi=self.current_performance.psi,
            rho=self.current_performance.rho,
            omega=self.current_performance.omega,
            algorithm=self.current_config.algorithm
        )

        # Incorporate into decision
        step_result['dionice'] = {
            'notes': dionice_result.notes,
            'score': dionice_result.resonance_score
        }

        return step_result
```

**Pros**:
- Minimal overhead
- Direct access to all dioniceOS features
- Type safety

**Cons**:
- Requires building Python module
- Platform-specific binaries

### Pattern 2: Telemetry as Orchestrator

**Use case**: Loose coupling, multiple services

```python
# External script or SCS integration
import requests

# Update telemetry after calibration
requests.post('http://localhost:8080/api/update_metrics', json={
    'psi': current_psi,
    'rho': current_rho,
    'omega': current_omega
})

# Trigger calibration via telemetry
response = requests.post(
    'http://localhost:8080/api/control/start_calibration'
)
job_id = response.json()['job_id']

# Poll for completion
while True:
    job = requests.get(f'http://localhost:8080/api/jobs/{job_id}').json()
    if job['status'] in ['completed', 'failed']:
        break
    time.sleep(1)
```

**Pros**:
- Language-agnostic
- Easy remote access
- Simple debugging

**Cons**:
- Network overhead
- More moving parts

### Pattern 3: Hybrid

**Use case**: Best of both worlds

```python
# Use PyO3 for performance-critical path
from metatron_dionice_bridge import DioniceKernel
dionice = DioniceKernel()

# Use HTTP API for monitoring/debugging
import requests
telemetry_url = "http://localhost:8080/api"

# Main loop
for step in range(100):
    # Fast: Direct dioniceOS call
    result = dionice.step(psi, rho, omega, algorithm)

    # Async: Update telemetry (non-blocking)
    requests.post(f'{telemetry_url}/update', json={
        'step': step,
        'metrics': {'psi': psi, 'rho': rho, 'omega': omega}
    }, timeout=0.1)
```

**Pros**:
- Performance where needed
- Visibility for debugging
- Flexible deployment

**Cons**:
- More complex setup

## Configuration

### Telemetry Server

`metatron_telemetry.toml`:
```toml
[server]
host = "127.0.0.1"  # Use "0.0.0.0" for remote access
port = 8080

static_dir = "metatron_telemetry/static"
```

Or via environment:
```bash
export METATRON__SERVER__HOST="0.0.0.0"
export METATRON__SERVER__PORT="3000"
```

### PyO3 Module

No runtime configuration needed. Compile-time features:

```bash
# With Python support
maturin develop --features python

# Without Python support (default)
cargo build -p metatron_dionice_bridge
```

## Monitoring & Debugging

### Check Health

```bash
# Telemetry server
curl http://localhost:8080/api/health

# PyO3 module
python3 -c "from metatron_dionice_bridge import DioniceKernel; k = DioniceKernel(); print('OK')"
```

### View Logs

```bash
# Telemetry server (verbose)
RUST_LOG=debug cargo run -p metatron_telemetry

# Watch logs
RUST_LOG=metatron_telemetry=info cargo run -p metatron_telemetry 2>&1 | grep -E "(INFO|WARN|ERROR)"
```

### Metrics

Dashboard shows:
- Current ψ, ρ, ω values
- Backend health (SCS, dioniceOS, Q⊗DASH)
- Recent jobs with status
- 50-point historical chart

Programmatic access:
```python
import requests
status = requests.get('http://localhost:8080/api/status').json()
print(f"Quality: {status['psi']:.4f}")
```

## Performance

### Telemetry Overhead

- HTTP polling: ~5ms per request (local)
- State updates: ~100μs
- Job tracking: ~50μs per job

**Impact**: Negligible for calibration runs (seconds to minutes)

### PyO3 Overhead

- Function call: ~50ns
- Data marshalling: ~1-5μs
- Total per-step: <10μs

**Impact**: ~0.001% of typical calibration step

### Scalability

- Telemetry: Handles 1000s requests/sec
- PyO3: No limit (pure in-process)
- Memory: 1-10MB per kernel instance

## Security

### Current State

⚠️ **Designed for local/trusted network use**

No authentication, authorization, or encryption by default.

### Production Checklist

For production deployment:

- [ ] Add authentication (JWT, API keys)
- [ ] Enable HTTPS (TLS certificates)
- [ ] Restrict CORS origins
- [ ] Rate limit API endpoints
- [ ] Add input validation
- [ ] Enable audit logging
- [ ] Use firewall rules
- [ ] Regular security updates

### Quick Security

```bash
# Bind to localhost only (default)
METATRON__SERVER__HOST="127.0.0.1" cargo run -p metatron_telemetry

# Use SSH tunnel for remote access
ssh -L 8080:localhost:8080 user@remote-server
# Then access http://localhost:8080 locally
```

## Troubleshooting

### Dashboard Not Loading

1. Check server is running: `curl http://localhost:8080/api/health`
2. Check static files exist: `ls metatron_telemetry/static/`
3. Check browser console (F12) for errors
4. Verify CORS settings in `metatron_telemetry/src/api/routes.rs`

### PyO3 Import Error

```python
ImportError: No module named 'metatron_dionice_bridge'
```

**Solution**:
```bash
cd metatron_dionice_bridge
maturin develop --features python --force
```

### Port Already in Use

```
Error: Address already in use (os error 98)
```

**Solution**:
```bash
# Find process using port
lsof -i :8080
kill <PID>

# Or use different port
METATRON__SERVER__PORT=9000 cargo run -p metatron_telemetry
```

### Metrics Not Updating

1. Check backend health on dashboard
2. Verify SCS is running and calling telemetry
3. Check server logs: `RUST_LOG=debug cargo run -p metatron_telemetry`
4. Ensure auto-refresh is enabled (dashboard refreshes every 5s)

## Future Enhancements

Potential improvements:

- [ ] WebSocket support for real-time updates
- [ ] Authentication and authorization
- [ ] Multi-user support
- [ ] Persistent storage (database)
- [ ] Advanced analytics (trend analysis, anomaly detection)
- [ ] Configuration management UI
- [ ] Job scheduling and queueing
- [ ] Export metrics (Prometheus, InfluxDB)
- [ ] Alert system (email, Slack)
- [ ] Mobile-responsive dashboard

## Contributing

To extend the telemetry system:

1. **Add API endpoint**: Edit `metatron_telemetry/src/api/handlers.rs`
2. **Add dashboard panel**: Edit `metatron_telemetry/static/index.html`
3. **Add PyO3 method**: Edit `metatron_dionice_bridge/src/python.rs`
4. **Add state field**: Edit `metatron_telemetry/src/state.rs`

## See Also

- [Dashboard README](../metatron_telemetry/README.md)
- [PyO3 Guide](./pyo3_integration.md)
- [dioniceOS Integration](../DIONICEOS_INTEGRATION.md)
- [SCS Documentation](../SCS_README.md)
- [Q⊗DASH Core](../README.md)
