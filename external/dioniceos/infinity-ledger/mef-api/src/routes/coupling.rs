/// Coupling and spiral navigation endpoints
use axum::{
    extract::{Path, State},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::{error::ApiError, AppState, Result};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/coupling/seed", post(coupling_seed))
        .route("/coupling/sync", post(coupling_sync))
        .route("/spiral/nav", post(spiral_nav))
        .route("/spiral/condense", post(spiral_condense))
        .route("/spiral/:id", get(get_spiral))
}

/// Seed coupling with event
#[derive(Debug, Deserialize)]
struct CouplingSeedRequest {
    event: JsonValue,
}

#[derive(Debug, Serialize)]
struct CouplingSeedResponse {
    status: String,
    result: JsonValue,
}

async fn coupling_seed(
    State(state): State<AppState>,
    Json(request): Json<CouplingSeedRequest>,
) -> Result<Json<CouplingSeedResponse>> {
    let mut coupling_engine = state
        .coupling_engine
        .lock()
        .map_err(|e| ApiError::Internal(format!("Failed to lock coupling engine: {}", e)))?;

    let result = coupling_engine
        .inject_seed(&request.event)
        .map_err(|e| ApiError::Processing(format!("Failed to inject seed: {}", e)))?;

    Ok(Json(CouplingSeedResponse {
        status: "ok".to_string(),
        result,
    }))
}

/// Sync coupling with HDAG
#[derive(Debug, Deserialize)]
struct CouplingSyncRequest {
    threshold: f64,
}

#[derive(Debug, Serialize)]
struct CouplingSyncResponse {
    status: String,
    result: JsonValue,
}

async fn coupling_sync(
    State(state): State<AppState>,
    Json(request): Json<CouplingSyncRequest>,
) -> Result<Json<CouplingSyncResponse>> {
    let mut coupling_engine = state
        .coupling_engine
        .lock()
        .map_err(|e| ApiError::Internal(format!("Failed to lock coupling engine: {}", e)))?;

    let result = coupling_engine
        .sync_hdag(request.threshold)
        .map_err(|e| ApiError::Processing(format!("Failed to sync HDAG: {}", e)))?;

    Ok(Json(CouplingSyncResponse {
        status: "ok".to_string(),
        result,
    }))
}

/// Navigate spiral
#[derive(Debug, Deserialize)]
struct SpiralNavRequest {
    theta_current: f64,
    candidates: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct SpiralNavResponse {
    best_theta: f64,
    best_score: f64,
}

async fn spiral_nav(
    State(state): State<AppState>,
    Json(request): Json<SpiralNavRequest>,
) -> Result<Json<SpiralNavResponse>> {
    let mut coupling_engine = state
        .coupling_engine
        .lock()
        .map_err(|e| ApiError::Internal(format!("Failed to lock coupling engine: {}", e)))?;

    let result = coupling_engine
        .navigate_spiral(
            request.theta_current,
            &request.candidates,
            None, // params
        )
        .map_err(|e| ApiError::Processing(format!("Failed to navigate spiral: {}", e)))?;

    // Extract best_theta and best_score from result
    let best_theta = result
        .get("best_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    let best_score = result
        .get("best_score")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    Ok(Json(SpiralNavResponse {
        best_theta,
        best_score,
    }))
}

/// Condense spiral histories
#[derive(Debug, Deserialize)]
struct SpiralCondenseRequest {
    histories: Vec<Vec<f64>>,
    mode: String,
}

#[derive(Debug, Serialize)]
struct SpiralCondenseResponse {
    condensed: Vec<f64>,
    variance: f64,
}

async fn spiral_condense(
    State(state): State<AppState>,
    Json(request): Json<SpiralCondenseRequest>,
) -> Result<Json<SpiralCondenseResponse>> {
    let mut coupling_engine = state
        .coupling_engine
        .lock()
        .map_err(|e| ApiError::Internal(format!("Failed to lock coupling engine: {}", e)))?;

    let result = coupling_engine
        .condense_histories(&request.histories, &request.mode)
        .map_err(|e| ApiError::Processing(format!("Failed to condense histories: {}", e)))?;

    let condensed = result
        .get("condensed")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|x| x.as_f64()).collect())
        .unwrap_or_else(Vec::new);

    let variance = result
        .get("variance")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    Ok(Json(SpiralCondenseResponse {
        condensed,
        variance,
    }))
}

/// Get spiral by ID
#[derive(Debug, Serialize)]
struct SpiralResponse {
    id: String,
    coordinates: Vec<f64>,
    phase: f64,
    timestamp: String,
}

async fn get_spiral(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<SpiralResponse>> {
    use mef_spiral::SpiralSnapshot;

    // Create spiral snapshot handler
    let spiral = SpiralSnapshot::new(
        state.spiral_config.as_ref().clone(),
        state.store_path.as_ref(),
    )
    .map_err(|e| ApiError::Internal(format!("Failed to create spiral snapshot: {}", e)))?;

    // Load snapshot
    let snapshot_opt = spiral
        .load_snapshot(&id)
        .map_err(|e| ApiError::NotFound(format!("Failed to load snapshot: {}", e)))?;

    let snapshot =
        snapshot_opt.ok_or_else(|| ApiError::NotFound(format!("Snapshot {} not found", id)))?;

    Ok(Json(SpiralResponse {
        id: snapshot.id,
        coordinates: snapshot.coordinates,
        phase: snapshot.phase,
        timestamp: snapshot.timestamp,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ApiConfig;

    #[tokio::test]
    async fn test_coupling_seed() {
        let config = ApiConfig::default();
        let state = AppState::new(config).await.unwrap();

        let request = CouplingSeedRequest {
            event: serde_json::json!({"type": "test"}),
        };

        let result = coupling_seed(State(state), Json(request)).await;
        assert!(result.is_ok());
    }
}
