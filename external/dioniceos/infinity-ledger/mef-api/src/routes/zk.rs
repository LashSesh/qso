/// Zero-knowledge inference endpoints
use axum::{extract::State, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::{error::ApiError, AppState, Result};

pub fn router() -> Router<AppState> {
    Router::new().route("/zk/infer", post(zk_infer))
}

/// Zero-knowledge inference request
#[derive(Debug, Deserialize)]
struct ZkInferRequest {
    input: JsonValue,
}

/// Zero-knowledge inference response
#[derive(Debug, Serialize)]
struct ZkInferResponse {
    result: JsonValue,
    proof: String,
    timestamp: String,
}

async fn zk_infer(
    State(state): State<AppState>,
    Json(request): Json<ZkInferRequest>,
) -> Result<Json<ZkInferResponse>> {
    let mut coupling_engine = state
        .coupling_engine
        .lock()
        .map_err(|e| ApiError::Internal(format!("Failed to lock coupling engine: {}", e)))?;

    let result = coupling_engine
        .zk_infer(&request.input)
        .map_err(|e| ApiError::Processing(format!("ZK inference failed: {}", e)))?;

    // Extract proof from result
    let proof = result
        .get("proof")
        .and_then(|v| v.as_str())
        .unwrap_or("0x0000...")
        .to_string();

    Ok(Json(ZkInferResponse {
        result,
        proof,
        timestamp: chrono::Utc::now().to_rfc3339(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ApiConfig;

    #[tokio::test]
    async fn test_zk_infer() {
        let config = ApiConfig::default();
        let state = AppState::new(config).await.unwrap();

        let request = ZkInferRequest {
            input: serde_json::json!({"data": [1.0, 2.0, 3.0]}),
        };

        let result = zk_infer(State(state), Json(request)).await;
        assert!(result.is_ok());
    }
}
