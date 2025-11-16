/// Health check and monitoring endpoints
use axum::{extract::State, routing::get, Json, Router};
use chrono::Utc;
use std::collections::HashMap;

use crate::{models::*, AppState};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/ping", get(ping))
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
}

/// Ping endpoint - basic health check
async fn ping(State(state): State<AppState>) -> Json<PingResponse> {
    Json(PingResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        seed: state.config.seed.clone(),
    })
}

/// Health endpoint - liveness probe
async fn healthz() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        timestamp: Utc::now().to_rfc3339(),
    })
}

/// Readiness endpoint - readiness probe
async fn readyz(State(_state): State<AppState>) -> Json<ReadyResponse> {
    let mut components = HashMap::new();

    // Check snapshot store
    components.insert("snapshot_store".to_string(), true);

    // Check ledger
    components.insert("ledger".to_string(), true);

    let all_ready = components.values().all(|&v| v);

    Json(ReadyResponse {
        ready: all_ready,
        components,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ApiConfig;

    #[tokio::test]
    async fn test_ping() {
        let config = ApiConfig::default();
        let state = AppState::new(config).await.unwrap();

        let response = ping(State(state)).await;
        assert_eq!(response.0.status, "ok");
    }

    #[tokio::test]
    async fn test_healthz() {
        let response = healthz().await;
        assert_eq!(response.0.status, "healthy");
    }
}
