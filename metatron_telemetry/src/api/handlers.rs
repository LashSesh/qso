//! HTTP request handlers

use crate::state::{AppState, Job, JobMetrics, JobStatus};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// GET /status - Get current system status
pub async fn get_status(State(state): State<AppState>) -> impl IntoResponse {
    let status = state.get_status().await;
    Json(status)
}

/// GET /jobs - Get all jobs
#[derive(Debug, Deserialize)]
pub struct JobsQuery {
    #[serde(default)]
    limit: Option<usize>,
}

pub async fn get_jobs(
    State(state): State<AppState>,
    Query(query): Query<JobsQuery>,
) -> impl IntoResponse {
    let mut jobs = state.get_jobs().await;

    // Apply limit
    if let Some(limit) = query.limit {
        let start = jobs.len().saturating_sub(limit);
        jobs = jobs[start..].to_vec();
    }

    Json(jobs)
}

/// GET /jobs/:id - Get specific job
pub async fn get_job(State(state): State<AppState>, Path(id): Path<Uuid>) -> impl IntoResponse {
    match state.get_job(id).await {
        Some(job) => (StatusCode::OK, Json(job)).into_response(),
        None => (StatusCode::NOT_FOUND, "Job not found").into_response(),
    }
}

/// GET /history - Get historical metrics
#[derive(Debug, Deserialize)]
pub struct HistoryQuery {
    #[serde(default)]
    limit: Option<usize>,
}

pub async fn get_history(
    State(state): State<AppState>,
    Query(query): Query<HistoryQuery>,
) -> impl IntoResponse {
    let history = state.get_history(query.limit).await;
    Json(history)
}

/// POST /control/start_calibration - Start new calibration run
#[derive(Debug, Deserialize)]
pub struct StartCalibrationRequest {
    #[serde(default)]
    algorithm: Option<String>,
    #[serde(default)]
    mode: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct StartCalibrationResponse {
    job_id: Uuid,
    message: String,
}

pub async fn start_calibration(
    State(state): State<AppState>,
    Json(req): Json<StartCalibrationRequest>,
) -> impl IntoResponse {
    // Create new job
    let job_id = Uuid::new_v4();
    let job = Job {
        id: job_id,
        job_type: "calibration".to_string(),
        status: JobStatus::Pending,
        started_at: Utc::now(),
        completed_at: None,
        metrics: JobMetrics {
            energy: None,
            accuracy: None,
            duration_secs: None,
            iterations: None,
            extra: serde_json::json!({}),
        },
    };

    state.add_job(job).await;

    // If mode specified, update it
    if let Some(mode) = req.mode {
        state.set_mode(mode).await;
    }

    // Spawn background task to simulate calibration
    let state_clone = state.clone();
    tokio::spawn(async move {
        simulate_calibration_run(state_clone, job_id).await;
    });

    let response = StartCalibrationResponse {
        job_id,
        message: "Calibration job started".to_string(),
    };

    (StatusCode::ACCEPTED, Json(response))
}

/// Simulate a calibration run (placeholder for actual integration)
async fn simulate_calibration_run(state: AppState, job_id: Uuid) {
    // Update to running
    state.update_job(job_id, JobStatus::Running, None).await;

    // Simulate work
    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

    // Get current status
    let current_status = state.get_status().await;

    // Simulate improvement
    let new_psi = (current_status.psi + 0.01).min(1.0);
    let new_rho = (current_status.rho + 0.005).min(1.0);
    let new_omega = (current_status.omega + 0.008).min(1.0);

    // Update status
    state
        .update_status(
            new_psi,
            new_rho,
            new_omega,
            current_status.algorithm.clone(),
        )
        .await;

    // Complete job
    let metrics = JobMetrics {
        energy: Some(-1.137 + rand::random::<f64>() * 0.01),
        accuracy: Some(new_psi),
        duration_secs: Some(3.0),
        iterations: Some(100),
        extra: serde_json::json!({
            "psi": new_psi,
            "rho": new_rho,
            "omega": new_omega,
        }),
    };

    state
        .update_job(job_id, JobStatus::Completed, Some(metrics))
        .await;

    tracing::info!("Calibration job {} completed", job_id);
}

/// Health check endpoint
pub async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "service": "metatron_telemetry"
    }))
}
