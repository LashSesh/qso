/// System metrics, gate FSM, and mode endpoints
use axum::{extract::State, routing::get, Json, Router};
use serde::Serialize;
use serde_json::Value as JsonValue;
use std::collections::HashMap;

use crate::{error::ApiError, AppState, Result};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/gate/fsm", get(get_gate_fsm))
        .route("/mode", get(get_mode))
        .route("/metrics", get(get_metrics))
        .route("/stats", get(get_stats))
}

/// Gate FSM state
#[derive(Debug, Serialize)]
struct GateFsmResponse {
    state: String,
    reasons: HashMap<String, JsonValue>,
}

async fn get_gate_fsm() -> Result<Json<GateFsmResponse>> {
    // For now, return a placeholder FSM state
    // In a real implementation, this would track actual gate decisions
    let mut reasons = HashMap::new();
    reasons.insert("phi".to_string(), JsonValue::Null);
    reasons.insert("mci".to_string(), JsonValue::Null);
    reasons.insert("por".to_string(), JsonValue::Null);
    reasons.insert("deltaV".to_string(), JsonValue::Null);
    reasons.insert("t".to_string(), JsonValue::Null);
    reasons.insert("t'".to_string(), JsonValue::Null);

    Ok(Json(GateFsmResponse {
        state: "idle".to_string(),
        reasons,
    }))
}

/// System mode
#[derive(Debug, Serialize)]
struct ModeResponse {
    mode: String,
    timestamp: String,
}

async fn get_mode() -> Result<Json<ModeResponse>> {
    Ok(Json(ModeResponse {
        mode: "production".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    }))
}

/// Prometheus metrics
async fn get_metrics(State(_state): State<AppState>) -> Result<String> {
    // Return Prometheus-formatted metrics
    // In a real implementation, this would collect actual metrics
    let metrics = r#"# HELP mef_api_requests_total Total number of API requests
# TYPE mef_api_requests_total counter
mef_api_requests_total{endpoint="/ping"} 0
mef_api_requests_total{endpoint="/search"} 0
mef_api_requests_total{endpoint="/ingest"} 0

# HELP mef_api_request_duration_seconds Request duration in seconds
# TYPE mef_api_request_duration_seconds histogram
mef_api_request_duration_seconds_bucket{endpoint="/search",le="0.005"} 0
mef_api_request_duration_seconds_bucket{endpoint="/search",le="0.01"} 0
mef_api_request_duration_seconds_bucket{endpoint="/search",le="0.025"} 0
mef_api_request_duration_seconds_bucket{endpoint="/search",le="0.05"} 0
mef_api_request_duration_seconds_bucket{endpoint="/search",le="0.1"} 0
mef_api_request_duration_seconds_bucket{endpoint="/search",le="0.25"} 0
mef_api_request_duration_seconds_bucket{endpoint="/search",le="0.5"} 0
mef_api_request_duration_seconds_bucket{endpoint="/search",le="1"} 0
mef_api_request_duration_seconds_bucket{endpoint="/search",le="+Inf"} 0
mef_api_request_duration_seconds_sum{endpoint="/search"} 0
mef_api_request_duration_seconds_count{endpoint="/search"} 0
"#;

    Ok(metrics.to_string())
}

/// System statistics
#[derive(Debug, Serialize)]
struct StatsResponse {
    total_snapshots: usize,
    total_tics: usize,
    total_blocks: usize,
    total_vectors: usize,
    uptime_seconds: f64,
}

async fn get_stats(State(state): State<AppState>) -> Result<Json<StatsResponse>> {
    // Get ledger stats
    let ledger = state
        .ledger
        .lock()
        .map_err(|e| ApiError::Internal(format!("Failed to lock ledger: {}", e)))?;

    let chain_stats = ledger
        .get_chain_statistics()
        .map_err(|e| ApiError::Ledger(format!("Failed to get chain statistics: {}", e)))?;

    // Get vector DB stats
    let index_manager = state
        .index_manager
        .lock()
        .map_err(|e| ApiError::Internal(format!("Failed to lock index manager: {}", e)))?;

    let total_vectors: usize = index_manager
        .collections
        .values()
        .map(|c| c.vectors.len())
        .sum();

    Ok(Json(StatsResponse {
        total_snapshots: 0, // Would need to track this
        total_tics: 0,      // Would need to track this
        total_blocks: chain_stats.total_blocks as usize,
        total_vectors,
        uptime_seconds: 0.0, // Would need to track this
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ApiConfig;

    #[tokio::test]
    async fn test_get_gate_fsm() {
        let result = get_gate_fsm().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_get_mode() {
        let result = get_mode().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_get_stats() {
        let config = ApiConfig::default();
        let state = AppState::new(config).await.unwrap();

        let result = get_stats(State(state)).await;
        assert!(result.is_ok());
    }
}
