# Metatron Telemetry - Q⊗DASH Dashboard

Web dashboard and HTTP API for monitoring and controlling the Q⊗DASH quantum-hybrid calibration system.

## Features

- **Real-time System Monitoring**: Track calibration metrics (ψ, ρ, ω) in real-time
- **Job Management**: View and manage calibration/benchmark runs
- **Historical Analytics**: Time-series visualization of system performance
- **Remote Control**: Trigger calibration runs via web interface or API
- **Backend Health**: Monitor SCS, dioniceOS, and Q⊗DASH status

## Quick Start

### 1. Start the Telemetry Server

From the repository root:

```bash
cargo run --release --bin metatron_telemetry
```

The server will start on `http://127.0.0.1:8080` by default.

### 2. Open the Dashboard

Navigate to http://127.0.0.1:8080 in your web browser.

You'll see:
- **Top Left**: Current system status (algorithm, mode, metrics, backend health)
- **Top Right**: Recent calibration/benchmark jobs with status
- **Bottom Left**: Line chart showing historical ψ, ρ, ω values
- **Bottom Right**: Control panel to start new calibration runs

### 3. Use the HTTP API

```bash
# Get system status
curl http://localhost:8080/api/status

# Get recent jobs
curl http://localhost:8080/api/jobs?limit=10

# Get historical metrics
curl http://localhost:8080/api/history?limit=50

# Start new calibration run
curl -X POST http://localhost:8080/api/control/start_calibration \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Configuration

Create a `metatron_telemetry.toml` file in the repository root:

```toml
[server]
host = "127.0.0.1"
port = 8080

static_dir = "metatron_telemetry/static"
```

Or use environment variables:

```bash
export METATRON__SERVER__HOST="0.0.0.0"
export METATRON__SERVER__PORT="3000"
export METATRON__STATIC_DIR="metatron_telemetry/static"
```

## API Endpoints

### GET `/api/status`

Returns current system status.

**Response:**
```json
{
  "algorithm": "VQE",
  "mode": "Explore",
  "psi": 0.8542,
  "rho": 0.9012,
  "omega": 0.7834,
  "backend_health": {
    "scs_ready": true,
    "dionice_ready": true,
    "qdash_ready": true
  },
  "last_update": "2025-11-13T10:30:45Z"
}
```

### GET `/api/jobs?limit=10`

Returns recent calibration/benchmark jobs.

**Query Parameters:**
- `limit` (optional): Maximum number of jobs to return

**Response:**
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "job_type": "calibration",
    "status": "completed",
    "started_at": "2025-11-13T10:25:00Z",
    "completed_at": "2025-11-13T10:25:03Z",
    "metrics": {
      "energy": -1.1372,
      "accuracy": 0.8567,
      "duration_secs": 3.0,
      "iterations": 100
    }
  }
]
```

### GET `/api/jobs/:id`

Returns details for a specific job.

**Response:** Same as single job object above.

### GET `/api/history?limit=50`

Returns historical metrics time series.

**Query Parameters:**
- `limit` (optional): Number of historical points (default: 1000)

**Response:**
```json
[
  {
    "timestamp": "2025-11-13T10:20:00Z",
    "psi": 0.8234,
    "rho": 0.8912,
    "omega": 0.7645,
    "algorithm": "VQE"
  },
  ...
]
```

### POST `/api/control/start_calibration`

Starts a new calibration run.

**Request Body (optional):**
```json
{
  "algorithm": "VQE",
  "mode": "Explore"
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Calibration job started"
}
```

### GET `/api/health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "service": "metatron_telemetry"
}
```

## Integration with SCS

The telemetry server is designed to integrate with the Seraphic Calibration Shell (SCS):

1. **State Updates**: After each SCS calibration step, send metrics to telemetry:
   ```python
   import requests

   # In your SCS calibrator.py
   def update_telemetry(psi, rho, omega, algorithm):
       # This is a placeholder - actual integration would use
       # the AppState directly or via a shared channel
       pass
   ```

2. **Job Tracking**: When SCS starts a calibration run, create a job entry:
   ```python
   job_id = uuid.uuid4()
   # Track job in telemetry state
   ```

3. **Backend Integration**: The telemetry server can call into:
   - SCS Python code (via subprocess or PyO3 bindings)
   - dioniceOS kernel (via `metatron_dionice_bridge`)
   - Q⊗DASH core (via `metatron-qso-rs`)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Web Dashboard (HTML/JS/CSS)                        │
│  - Auto-refreshes every 5 seconds                   │
│  - Chart.js for metrics visualization               │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP/JSON
┌──────────────────▼──────────────────────────────────┐
│  Axum HTTP Server                                    │
│  - REST API endpoints                                │
│  - Static file serving                               │
│  - CORS enabled                                      │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│  AppState (Arc<RwLock>)                              │
│  - Current system status                             │
│  - Job queue (last 100)                              │
│  - Metrics history (last 1000 points)                │
└──────────────────┬──────────────────────────────────┘
                   │
       ┌───────────┴───────────┐
       │                       │
┌──────▼──────┐        ┌───────▼────────┐
│  SCS Python │        │  dioniceOS     │
│  (via PyO3) │        │  (via bridge)  │
└─────────────┘        └────────────────┘
```

## Development

### Build

```bash
cargo build --release -p metatron_telemetry
```

### Run in Development Mode

```bash
RUST_LOG=debug cargo run -p metatron_telemetry
```

### Test

```bash
cargo test -p metatron_telemetry
```

### Customize Dashboard

The dashboard is in `metatron_telemetry/static/`:
- `index.html` - Main page structure
- `css/style.css` - Styling
- `js/app.js` - Application logic

Edit these files to customize the dashboard appearance and behavior.

## Production Deployment

### 1. Build Release Binary

```bash
cargo build --release --bin metatron_telemetry
```

Binary location: `target/release/metatron_telemetry`

### 2. Run as Service

```bash
./target/release/metatron_telemetry
```

### 3. Reverse Proxy (Optional)

Use nginx or Apache to proxy requests:

```nginx
server {
    listen 80;
    server_name qdash.example.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## Security Considerations

⚠️ **This telemetry server is designed for local/trusted network use.**

For production:
1. Add authentication (JWT tokens, API keys)
2. Enable HTTPS (TLS certificates)
3. Restrict CORS origins
4. Rate limit API endpoints
5. Add input validation
6. Log all control actions

## Troubleshooting

### Server won't start

Check if port is already in use:
```bash
lsof -i :8080
```

Change port in config or use:
```bash
METATRON__SERVER__PORT=9000 cargo run -p metatron_telemetry
```

### Dashboard shows "Loading..."

1. Check API endpoints are accessible:
   ```bash
   curl http://localhost:8080/api/status
   ```

2. Check browser console for errors (F12)

3. Ensure static files are in correct location:
   ```bash
   ls metatron_telemetry/static/
   ```

### Metrics not updating

The dashboard auto-refreshes every 5 seconds. If updates stop:
1. Check backend health indicators on dashboard
2. Verify SCS is running and updating metrics
3. Check server logs: `RUST_LOG=debug cargo run -p metatron_telemetry`

## License

MIT
