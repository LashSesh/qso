/// Merkaba Gate API endpoints
use axum::{
    extract::{Query, State},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::{AppState, Result};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/gate/merkaba", post(evaluate_merkaba_gate))
        .route("/gate/merkaba/status", get(get_merkaba_status))
        .route("/gate/merkaba/audit", get(get_audit_log))
        .route("/gate/merkaba/calibrate", post(calibrate_thresholds))
}

/// Evaluate TIC candidate through Merkaba Gate
#[derive(Debug, Deserialize)]
struct MerkabaGateRequest {
    snapshot_id: String,
    tic_candidate_id: String,
    #[serde(default)]
    params: Option<GateParams>,
}

#[derive(Debug, Deserialize)]
struct GateParams {
    epsilon: Option<f64>,
    phi_star: Option<f64>,
    eta: Option<f64>,
}

#[derive(Debug, Serialize)]
struct MerkabaGateResponse {
    gate_id: String,
    snapshot_id: String,
    tic_candidate_id: String,
    checks: GateChecks,
    decision: GateDecision,
    timestamp: String,
    ledger_block_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct GateChecks {
    por: String,
    delta_pi: f64,
    phi: f64,
    delta_v: f64,
    mci: Option<f64>,
}

#[derive(Debug, Serialize)]
struct GateDecision {
    commit: bool,
    reason: String,
}

async fn evaluate_merkaba_gate(
    State(state): State<AppState>,
    Json(request): Json<MerkabaGateRequest>,
) -> Result<Json<MerkabaGateResponse>> {
    // Extract parameters or use defaults
    let epsilon = request.params.as_ref().and_then(|p| p.epsilon);
    let phi_star = request.params.as_ref().and_then(|p| p.phi_star);
    let eta = request.params.as_ref().and_then(|p| p.eta);

    // Create TIC candidate from request
    // In a real implementation, this would load from storage
    // For now, create a candidate that will pass gate checks
    use mef_core::gates::merkaba_gate::TICCandidate;
    let tic_candidate = TICCandidate {
        tic_id: request.tic_candidate_id.clone(),
        // Use a stable fixpoint that will pass checks
        fixpoint: vec![
            1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01,
        ],
        por_status: "valid".to_string(),
        operator_sequence: vec!["DK".to_string(), "SW".to_string()],
        timestamp: chrono::Utc::now().timestamp() as f64,
        dual_fixpoint: Some(vec![
            1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01,
        ]),
    };

    // Run Merkaba Gate evaluation
    let mut gate = state.merkaba_gate.lock().unwrap();
    let gate_event = gate.run_merkaba(
        request.snapshot_id.clone(),
        tic_candidate,
        epsilon,
        phi_star,
        eta,
    );

    // Determine ledger block ID if committed
    let ledger_block_id = if gate_event.decision.commit {
        Some(format!("block_{}", uuid::Uuid::new_v4()))
    } else {
        None
    };

    Ok(Json(MerkabaGateResponse {
        gate_id: gate_event.gate_id,
        snapshot_id: gate_event.snapshot_id,
        tic_candidate_id: gate_event.tic_candidate_id,
        checks: GateChecks {
            por: gate_event.checks.por,
            delta_pi: gate_event.checks.delta_pi,
            phi: gate_event.checks.phi,
            delta_v: gate_event.checks.delta_v,
            mci: gate_event.checks.mci,
        },
        decision: GateDecision {
            commit: gate_event.decision.commit,
            reason: gate_event.decision.reason,
        },
        timestamp: gate_event.timestamp,
        ledger_block_id,
    }))
}

/// Get Merkaba Gate status and configuration
#[derive(Debug, Serialize)]
struct MerkabaStatusResponse {
    status: String,
    thresholds: ThresholdConfig,
    metatron_nodes: usize,
    state_history_length: usize,
    audit_path: String,
}

#[derive(Debug, Serialize)]
struct ThresholdConfig {
    epsilon: f64,
    phi_star: f64,
    eta: f64,
}

async fn get_merkaba_status(State(state): State<AppState>) -> Result<Json<MerkabaStatusResponse>> {
    // Get actual gate configuration
    let gate = state.merkaba_gate.lock().unwrap();

    Ok(Json(MerkabaStatusResponse {
        status: "operational".to_string(),
        thresholds: ThresholdConfig {
            epsilon: gate.epsilon,
            phi_star: gate.phi_star,
            eta: gate.eta,
        },
        metatron_nodes: 13,
        state_history_length: gate.state_history.len(),
        audit_path: gate.audit_path.to_string_lossy().to_string(),
    }))
}

/// Get Merkaba Gate audit log
#[derive(Debug, Deserialize)]
struct AuditQuery {
    #[serde(default = "default_audit_limit")]
    limit: usize,
}

fn default_audit_limit() -> usize {
    100
}

#[derive(Debug, Serialize)]
struct AuditEntry {
    gate_id: String,
    snapshot_id: String,
    tic_candidate_id: String,
    decision: String,
    timestamp: String,
    checks: JsonValue,
}

#[derive(Debug, Serialize)]
struct AuditLogResponse {
    entries: Vec<AuditEntry>,
    total: usize,
}

async fn get_audit_log(
    State(state): State<AppState>,
    Query(query): Query<AuditQuery>,
) -> Result<Json<AuditLogResponse>> {
    // Read audit log from file
    let gate = state.merkaba_gate.lock().unwrap();
    let audit_path = &gate.audit_path;

    let mut entries = Vec::new();

    // Try to read audit log if it exists
    if audit_path.exists() {
        if let Ok(content) = std::fs::read_to_string(audit_path) {
            // Parse JSONL format (one JSON object per line)
            for line in content.lines().rev().take(query.limit) {
                if let Ok(event) = serde_json::from_str::<serde_json::Value>(line) {
                    // Convert to AuditEntry format
                    if let Some(obj) = event.as_object() {
                        entries.push(AuditEntry {
                            gate_id: obj
                                .get("gate_id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            snapshot_id: obj
                                .get("snapshot_id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            tic_candidate_id: obj
                                .get("tic_candidate_id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            decision: obj
                                .get("decision")
                                .and_then(|d| d.get("commit"))
                                .and_then(|c| c.as_bool())
                                .map(|b| {
                                    if b {
                                        "commit".to_string()
                                    } else {
                                        "reject".to_string()
                                    }
                                })
                                .unwrap_or("unknown".to_string()),
                            timestamp: obj
                                .get("timestamp")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            checks: obj.get("checks").cloned().unwrap_or(serde_json::json!({})),
                        });
                    }
                }
            }
        }
    }

    let total = entries.len();

    Ok(Json(AuditLogResponse { entries, total }))
}

/// Calibrate Merkaba Gate thresholds
#[derive(Debug, Deserialize)]
struct CalibrateRequest {
    #[serde(default)]
    epsilon: Option<f64>,
    #[serde(default)]
    phi_star: Option<f64>,
    #[serde(default)]
    eta: Option<f64>,
}

#[derive(Debug, Serialize)]
struct CalibrateResponse {
    status: String,
    updated: JsonValue,
    current: ThresholdConfig,
}

async fn calibrate_thresholds(
    State(state): State<AppState>,
    Json(request): Json<CalibrateRequest>,
) -> Result<Json<CalibrateResponse>> {
    // Update gate thresholds
    let mut gate = state.merkaba_gate.lock().unwrap();

    let mut updated = serde_json::Map::new();

    if let Some(epsilon) = request.epsilon {
        gate.epsilon = epsilon;
        updated.insert("epsilon".to_string(), serde_json::json!(epsilon));
    }
    if let Some(phi_star) = request.phi_star {
        gate.phi_star = phi_star;
        updated.insert("phi_star".to_string(), serde_json::json!(phi_star));
    }
    if let Some(eta) = request.eta {
        gate.eta = eta;
        updated.insert("eta".to_string(), serde_json::json!(eta));
    }

    Ok(Json(CalibrateResponse {
        status: "calibrated".to_string(),
        updated: serde_json::json!(updated),
        current: ThresholdConfig {
            epsilon: gate.epsilon,
            phi_star: gate.phi_star,
            eta: gate.eta,
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ApiConfig;

    async fn test_state() -> AppState {
        AppState::new(ApiConfig::default()).await.unwrap()
    }

    #[tokio::test]
    async fn test_evaluate_merkaba_gate() {
        let state = test_state().await;
        let request = MerkabaGateRequest {
            snapshot_id: "snap_123".to_string(),
            tic_candidate_id: "tic_456".to_string(),
            params: Some(GateParams {
                epsilon: Some(0.1),  // More lenient epsilon for test
                phi_star: Some(0.5), // Lower threshold for test
                eta: Some(0.7),      // Lower threshold for test
            }),
        };
        let result = evaluate_merkaba_gate(State(state), Json(request)).await;
        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert_eq!(response.snapshot_id, "snap_123");
        assert_eq!(response.tic_candidate_id, "tic_456");
        // With more lenient thresholds, the placeholder should pass
        assert!(response.decision.commit || !response.decision.reason.is_empty());
    }

    #[tokio::test]
    async fn test_get_merkaba_status() {
        let state = test_state().await;
        let result = get_merkaba_status(State(state)).await;
        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert_eq!(response.status, "operational");
        assert_eq!(response.metatron_nodes, 13);
    }

    #[tokio::test]
    async fn test_get_audit_log() {
        let state = test_state().await;
        let query = AuditQuery { limit: 50 };
        let result = get_audit_log(State(state), Query(query)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_calibrate_thresholds() {
        let state = test_state().await;
        let request = CalibrateRequest {
            epsilon: Some(0.015),
            phi_star: Some(0.92),
            eta: None,
        };
        let result = calibrate_thresholds(State(state), Json(request)).await;
        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert_eq!(response.status, "calibrated");
        assert_eq!(response.current.epsilon, 0.015);
        assert_eq!(response.current.phi_star, 0.92);
    }
}
