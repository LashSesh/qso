/// Ingestion endpoints
use axum::{extract::State, routing::post, Json, Router};
use chrono::Utc;
use serde_json::Value as JsonValue;

use crate::{error::ApiError, models::*, AppState, Result};
use mef_ingestion::normalize_payload;
use mef_spiral::SpiralSnapshot;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/ingest", post(ingest))
        .route("/acquisition", post(acquisition))
}

/// Ingest data into MEF-Core system
async fn ingest(
    State(state): State<AppState>,
    Json(request): Json<IngestRequest>,
) -> Result<Json<IngestResponse>> {
    // Parse data as JSON
    let data_json: JsonValue = serde_json::from_str(&request.data)
        .unwrap_or_else(|_| JsonValue::String(request.data.clone()));

    // Normalize the payload
    let normalized = normalize_payload(&data_json);

    // Create spiral snapshot handler
    let spiral = SpiralSnapshot::new(
        state.spiral_config.as_ref().clone(),
        state.store_path.as_ref(),
    )
    .map_err(|e| ApiError::Internal(format!("Failed to create spiral snapshot: {}", e)))?;

    // Create snapshot
    let snapshot = spiral
        .create_snapshot(&normalized, &request.seed, None)
        .map_err(|e| ApiError::Processing(format!("Failed to create snapshot: {}", e)))?;

    // Save the snapshot
    let _snapshot_path = spiral
        .save_snapshot(&snapshot)
        .map_err(|e| ApiError::Storage(format!("Failed to save snapshot: {}", e)))?;

    // Get snapshot ID
    let snapshot_id = snapshot.id.clone();

    // Get metrics
    let phase = snapshot.phase;
    let por_result = snapshot.metrics.por.clone();

    // Compute hash from the snapshot ID
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(snapshot_id.as_bytes());
    let hash = format!("{:x}", hasher.finalize());

    Ok(Json(IngestResponse {
        snapshot_id: snapshot_id.clone(),
        phase,
        por: por_result,
        hash,
        timestamp: Utc::now().to_rfc3339(),
    }))
}

/// Acquisition endpoint - alternative ingestion method
async fn acquisition(
    State(state): State<AppState>,
    Json(request): Json<AcquisitionRequest>,
) -> Result<Json<AcquisitionResponse>> {
    // Use default seed from config
    let seed = &state.config.seed;

    // Normalize the payload
    let normalized = normalize_payload(&request.data);

    // Create spiral snapshot handler
    let spiral = SpiralSnapshot::new(
        state.spiral_config.as_ref().clone(),
        state.store_path.as_ref(),
    )
    .map_err(|e| ApiError::Internal(format!("Failed to create spiral snapshot: {}", e)))?;

    // Create snapshot
    let snapshot = spiral
        .create_snapshot(&normalized, seed, None)
        .map_err(|e| ApiError::Processing(format!("Failed to create snapshot: {}", e)))?;

    // Save snapshot
    let _snapshot_path = spiral
        .save_snapshot(&snapshot)
        .map_err(|e| ApiError::Storage(format!("Failed to save snapshot: {}", e)))?;

    let snapshot_id = snapshot.id.clone();

    Ok(Json(AcquisitionResponse {
        success: true,
        vector_id: snapshot_id,
        collection: state.config.quality_collection.clone(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ApiConfig;

    #[tokio::test]
    async fn test_ingest_basic() {
        let config = ApiConfig::default();
        let state = AppState::new(config).await.unwrap();

        let request = IngestRequest {
            data: r#"{"test": "data"}"#.to_string(),
            data_type: "json".to_string(),
            seed: "test_seed".to_string(),
        };

        let result = ingest(State(state), Json(request)).await;
        assert!(result.is_ok());
    }
}
