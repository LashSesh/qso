//! System state management
//!
//! Maintains the current calibration state, metrics, and job history.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Current system status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    /// Current algorithm (VQE, QAOA, VQC)
    pub algorithm: String,
    /// Current mode (Explore, Exploit, Homeostasis)
    pub mode: String,
    /// Quality metric (0.0 - 1.0)
    pub psi: f64,
    /// Stability metric (0.0 - 1.0)
    pub rho: f64,
    /// Efficiency metric (0.0 - 1.0)
    pub omega: f64,
    /// Backend health flags
    pub backend_health: BackendHealth,
    /// Current quantum backend info
    pub backend_info: BackendInfo,
    /// Available quantum backends
    pub available_backends: Vec<String>,
    /// TRITON search status (if active)
    pub triton_status: Option<TritonStatus>,
    /// Last update timestamp
    pub last_update: DateTime<Utc>,
}

/// Backend health indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendHealth {
    /// SCS calibrator status
    pub scs_ready: bool,
    /// dioniceOS kernel status
    pub dionice_ready: bool,
    /// Q⊗DASH core status
    pub qdash_ready: bool,
}

/// Quantum backend information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendInfo {
    /// Backend provider (local, ibm, azure, etc.)
    pub provider: String,
    /// Backend name (local_sim, ibm_kyoto, etc.)
    pub name: String,
    /// Number of qubits
    pub num_qubits: u32,
    /// Is simulator (true) or QPU (false)
    pub is_simulator: bool,
    /// Backend mode (for QPUs: disabled, dry-run, enabled)
    pub mode: Option<String>,
}

/// TRITON search status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TritonStatus {
    /// Current search step
    pub step: usize,
    /// Best resonance found (ψ × ρ × ω)
    pub best_resonance: f64,
    /// Current resonance
    pub current_resonance: f64,
    /// Recent improvement rate
    pub improvement_rate: f64,
    /// Is the search converged?
    pub converged: bool,
    /// Number of parameters being optimized
    pub num_parameters: usize,
}

/// Job status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

/// Calibration or benchmark job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    /// Unique job ID
    pub id: Uuid,
    /// Job type (calibration, benchmark, etc.)
    pub job_type: String,
    /// Current status
    pub status: JobStatus,
    /// Start timestamp
    pub started_at: DateTime<Utc>,
    /// Completion timestamp (if done)
    pub completed_at: Option<DateTime<Utc>>,
    /// Short metrics summary
    pub metrics: JobMetrics,
}

/// Job metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobMetrics {
    /// Final energy (for quantum algorithms)
    pub energy: Option<f64>,
    /// Accuracy score
    pub accuracy: Option<f64>,
    /// Execution time (seconds)
    pub duration_secs: Option<f64>,
    /// Number of iterations
    pub iterations: Option<u32>,
    /// Additional metadata
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

/// Historical data point for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryPoint {
    pub timestamp: DateTime<Utc>,
    pub psi: f64,
    pub rho: f64,
    pub omega: f64,
    pub algorithm: String,
}

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    inner: Arc<RwLock<AppStateInner>>,
}

struct AppStateInner {
    /// Current system status
    status: SystemStatus,
    /// Recent jobs (last 100)
    jobs: Vec<Job>,
    /// Historical metrics (last 1000 points)
    history: Vec<HistoryPoint>,
}

impl AppState {
    /// Create new application state
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(AppStateInner {
                status: SystemStatus {
                    algorithm: "VQE".to_string(),
                    mode: "Explore".to_string(),
                    psi: 0.85,
                    rho: 0.90,
                    omega: 0.75,
                    backend_health: BackendHealth {
                        scs_ready: true,
                        dionice_ready: true,
                        qdash_ready: true,
                    },
                    backend_info: BackendInfo {
                        provider: "local".to_string(),
                        name: "local_sim".to_string(),
                        num_qubits: 13,
                        is_simulator: true,
                        mode: None,
                    },
                    available_backends: vec!["local_sim".to_string()],
                    triton_status: None,
                    last_update: Utc::now(),
                },
                jobs: Vec::new(),
                history: Vec::new(),
            })),
        }
    }

    /// Get current system status
    pub async fn get_status(&self) -> SystemStatus {
        self.inner.read().await.status.clone()
    }

    /// Update system status
    pub async fn update_status(&self, psi: f64, rho: f64, omega: f64, algorithm: String) {
        let mut state = self.inner.write().await;
        state.status.psi = psi;
        state.status.rho = rho;
        state.status.omega = omega;
        state.status.algorithm = algorithm.clone();
        state.status.last_update = Utc::now();

        // Add to history
        state.history.push(HistoryPoint {
            timestamp: Utc::now(),
            psi,
            rho,
            omega,
            algorithm,
        });

        // Keep last 1000 points
        let history_len = state.history.len();
        if history_len > 1000 {
            state.history.drain(0..history_len - 1000);
        }
    }

    /// Set system mode
    pub async fn set_mode(&self, mode: String) {
        self.inner.write().await.status.mode = mode;
    }

    /// Update TRITON search status
    pub async fn update_triton_status(&self, status: Option<TritonStatus>) {
        self.inner.write().await.status.triton_status = status;
    }

    /// Get all jobs
    pub async fn get_jobs(&self) -> Vec<Job> {
        self.inner.read().await.jobs.clone()
    }

    /// Get specific job
    pub async fn get_job(&self, id: Uuid) -> Option<Job> {
        self.inner
            .read()
            .await
            .jobs
            .iter()
            .find(|j| j.id == id)
            .cloned()
    }

    /// Add new job
    pub async fn add_job(&self, job: Job) {
        let mut state = self.inner.write().await;
        state.jobs.push(job);

        // Keep last 100 jobs
        let jobs_len = state.jobs.len();
        if jobs_len > 100 {
            state.jobs.drain(0..jobs_len - 100);
        }
    }

    /// Update job status
    pub async fn update_job(&self, id: Uuid, status: JobStatus, metrics: Option<JobMetrics>) {
        let mut state = self.inner.write().await;
        if let Some(job) = state.jobs.iter_mut().find(|j| j.id == id) {
            job.status = status;
            if let Some(m) = metrics {
                job.metrics = m;
            }
            if job.status == JobStatus::Completed || job.status == JobStatus::Failed {
                job.completed_at = Some(Utc::now());
            }
        }
    }

    /// Get history (last N points)
    pub async fn get_history(&self, limit: Option<usize>) -> Vec<HistoryPoint> {
        let state = self.inner.read().await;
        let history = &state.history;
        let limit = limit.unwrap_or(1000).min(history.len());
        history[history.len().saturating_sub(limit)..].to_vec()
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
