// Q⊗DASH Dashboard Application

const API_BASE = '/api';
const REFRESH_INTERVAL = 5000; // 5 seconds

// Global state
let metricsChart = null;
let autoRefresh = true;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeChart();
    setupEventListeners();
    refreshAll();

    // Start auto-refresh
    setInterval(() => {
        if (autoRefresh) {
            refreshAll();
        }
    }, REFRESH_INTERVAL);
});

// Setup event listeners
function setupEventListeners() {
    document.getElementById('start-calibration').addEventListener('click', startCalibration);
    document.getElementById('refresh-all').addEventListener('click', refreshAll);
}

// Refresh all data
async function refreshAll() {
    await Promise.all([
        fetchStatus(),
        fetchJobs(),
        fetchHistory()
    ]);
}

// Fetch system status
async function fetchStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const status = await response.json();
        updateStatusPanel(status);
    } catch (error) {
        console.error('Error fetching status:', error);
    }
}

// Fetch jobs list
async function fetchJobs() {
    try {
        const response = await fetch(`${API_BASE}/jobs?limit=10`);
        const jobs = await response.json();
        updateJobsPanel(jobs);
    } catch (error) {
        console.error('Error fetching jobs:', error);
    }
}

// Fetch historical data
async function fetchHistory() {
    try {
        const response = await fetch(`${API_BASE}/history?limit=50`);
        const history = await response.json();
        updateChart(history);
    } catch (error) {
        console.error('Error fetching history:', error);
    }
}

// Start calibration
async function startCalibration() {
    const button = document.getElementById('start-calibration');
    const messageDiv = document.getElementById('control-message');

    button.disabled = true;
    button.innerHTML = '<span class="btn-icon">⏳</span> Starting...';

    try {
        const response = await fetch(`${API_BASE}/control/start_calibration`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({})
        });

        const result = await response.json();

        messageDiv.textContent = `✓ ${result.message} (Job ID: ${result.job_id.substring(0, 8)}...)`;
        messageDiv.className = 'control-message success';

        // Refresh jobs immediately
        setTimeout(fetchJobs, 500);

        // Clear message after 5 seconds
        setTimeout(() => {
            messageDiv.className = 'control-message';
        }, 5000);

    } catch (error) {
        messageDiv.textContent = `✗ Error: ${error.message}`;
        messageDiv.className = 'control-message error';
    } finally {
        button.disabled = false;
        button.innerHTML = '<span class="btn-icon">▶</span> Start Calibration';
    }
}

// Update status panel
function updateStatusPanel(status) {
    document.getElementById('algorithm').textContent = status.algorithm;
    document.getElementById('mode').textContent = status.mode;
    document.getElementById('psi').textContent = status.psi.toFixed(4);
    document.getElementById('rho').textContent = status.rho.toFixed(4);
    document.getElementById('omega').textContent = status.omega.toFixed(4);

    // Update health indicators
    updateHealthDot('scs-health', status.backend_health.scs_ready);
    updateHealthDot('dionice-health', status.backend_health.dionice_ready);
    updateHealthDot('qdash-health', status.backend_health.qdash_ready);

    // Update backend info
    if (status.backend_info) {
        const backendType = document.getElementById('backend-type');
        const backendName = document.getElementById('backend-name');
        const backendQubits = document.getElementById('backend-qubits');

        const typeLabel = status.backend_info.is_simulator ? 'SIMULATOR' : 'QPU';
        backendType.textContent = typeLabel;
        backendType.className = status.backend_info.is_simulator ? 'backend-type' : 'backend-type qpu';

        backendName.textContent = status.backend_info.name;
        backendQubits.textContent = `${status.backend_info.num_qubits} qubits`;

        // If there's a mode (for QPUs), show it
        if (status.backend_info.mode) {
            backendQubits.textContent += ` (${status.backend_info.mode})`;
        }
    }

    // Update timestamp
    const timestamp = new Date(status.last_update);
    document.getElementById('last-update').textContent = timestamp.toLocaleTimeString();
}

// Update health dot
function updateHealthDot(elementId, isHealthy) {
    const dot = document.getElementById(elementId);
    dot.className = `health-dot ${isHealthy ? 'healthy' : 'unhealthy'}`;
}

// Update jobs panel
function updateJobsPanel(jobs) {
    const jobsList = document.getElementById('jobs-list');

    if (jobs.length === 0) {
        jobsList.innerHTML = '<p class="loading">No jobs yet</p>';
        return;
    }

    // Sort by start time, newest first
    jobs.sort((a, b) => new Date(b.started_at) - new Date(a.started_at));

    jobsList.innerHTML = jobs.map(job => `
        <div class="job-item">
            <div class="job-header">
                <span class="job-id">${job.id.substring(0, 8)}... (${job.job_type})</span>
                <span class="job-status ${job.status}">${job.status.toUpperCase()}</span>
            </div>
            <div class="job-metrics">
                ${job.metrics.energy ? `<span>Energy: ${job.metrics.energy.toFixed(4)}</span>` : ''}
                ${job.metrics.accuracy ? `<span>Accuracy: ${job.metrics.accuracy.toFixed(4)}</span>` : ''}
                ${job.metrics.duration_secs ? `<span>Duration: ${job.metrics.duration_secs.toFixed(2)}s</span>` : ''}
                ${job.metrics.iterations ? `<span>Iterations: ${job.metrics.iterations}</span>` : ''}
            </div>
            <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 5px;">
                Started: ${new Date(job.started_at).toLocaleTimeString()}
            </div>
        </div>
    `).join('');
}

// Initialize Chart.js chart
function initializeChart() {
    const ctx = document.getElementById('metricsChart').getContext('2d');

    metricsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'ψ (Quality)',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'ρ (Stability)',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'ω (Efficiency)',
                    data: [],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#f1f5f9'
                    }
                },
                title: {
                    display: false
                }
            },
            scales: {
                x: {
                    grid: {
                        color: '#334155'
                    },
                    ticks: {
                        color: '#94a3b8',
                        maxTicksLimit: 10
                    }
                },
                y: {
                    min: 0,
                    max: 1,
                    grid: {
                        color: '#334155'
                    },
                    ticks: {
                        color: '#94a3b8'
                    }
                }
            }
        }
    });
}

// Update chart with new data
function updateChart(history) {
    if (!metricsChart || history.length === 0) return;

    // Extract data
    const labels = history.map(h => {
        const date = new Date(h.timestamp);
        return date.toLocaleTimeString();
    });

    const psiData = history.map(h => h.psi);
    const rhoData = history.map(h => h.rho);
    const omegaData = history.map(h => h.omega);

    // Update chart
    metricsChart.data.labels = labels;
    metricsChart.data.datasets[0].data = psiData;
    metricsChart.data.datasets[1].data = rhoData;
    metricsChart.data.datasets[2].data = omegaData;
    metricsChart.update('none'); // Update without animation for smoother updates
}
