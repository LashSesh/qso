/// TIC and proof endpoints
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
        .route("/tic/:id", get(get_tic))
        .route("/tic/query", post(tic_query))
        .route("/proof/:id", get(get_proof))
        .route("/proof/batch", post(batch_proofs))
}

/// Get TIC by ID
#[derive(Debug, Serialize)]
struct TicResponse {
    id: String,
    snapshot_id: String,
    eigenvalues: Vec<f64>,
    timestamp: String,
}

async fn get_tic(
    State(_state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<TicResponse>> {
    // For now, return a placeholder
    // In a real implementation, we would load the TIC from storage
    Ok(Json(TicResponse {
        id: id.clone(),
        snapshot_id: format!("snapshot_{}", id),
        eigenvalues: vec![1.0, 0.5, 0.25],
        timestamp: chrono::Utc::now().to_rfc3339(),
    }))
}

/// Query TICs
#[derive(Debug, Deserialize)]
struct TicQueryRequest {
    vector: Vec<f64>,
    #[serde(default = "default_k")]
    k: usize,
}

fn default_k() -> usize {
    5
}

#[derive(Debug, Serialize)]
struct TicQueryResponse {
    results: Vec<TicQueryResult>,
}

#[derive(Debug, Serialize)]
struct TicQueryResult {
    tic_id: String,
    score: f64,
    metadata: JsonValue,
}

async fn tic_query(
    State(state): State<AppState>,
    Json(request): Json<TicQueryRequest>,
) -> Result<Json<TicQueryResponse>> {
    let coupling_engine = state
        .coupling_engine
        .lock()
        .map_err(|e| ApiError::Internal(format!("Failed to lock coupling engine: {}", e)))?;

    let results = coupling_engine
        .query_tics(&request.vector, request.k)
        .map_err(|e| ApiError::Processing(format!("Failed to query TICs: {}", e)))?;

    let query_results: Vec<TicQueryResult> = results
        .into_iter()
        .map(|r| TicQueryResult {
            tic_id: r
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            score: r.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0),
            metadata: r.clone(),
        })
        .collect();

    Ok(Json(TicQueryResponse {
        results: query_results,
    }))
}

/// Get membership proof by ID
#[derive(Debug, Serialize)]
struct ProofResponse {
    id: String,
    proof_type: String,
    merkle_root: String,
    path: Vec<String>,
    valid: bool,
}

async fn get_proof(Path(id): Path<String>) -> Result<Json<ProofResponse>> {
    // For now, return a placeholder
    // In a real implementation, we would load the proof from storage
    Ok(Json(ProofResponse {
        id: id.clone(),
        proof_type: "membership".to_string(),
        merkle_root: "0x1234...".to_string(),
        path: vec!["0xabcd...".to_string(), "0xef01...".to_string()],
        valid: true,
    }))
}

/// Batch proof requests
#[derive(Debug, Deserialize)]
struct BatchProofRequest {
    ids: Vec<String>,
}

#[derive(Debug, Serialize)]
struct BatchProofResponse {
    proofs: Vec<ProofResponse>,
}

async fn batch_proofs(Json(request): Json<BatchProofRequest>) -> Result<Json<BatchProofResponse>> {
    // For now, return placeholders for each ID
    let proofs = request
        .ids
        .into_iter()
        .map(|id| ProofResponse {
            id: id.clone(),
            proof_type: "membership".to_string(),
            merkle_root: "0x1234...".to_string(),
            path: vec!["0xabcd...".to_string(), "0xef01...".to_string()],
            valid: true,
        })
        .collect();

    Ok(Json(BatchProofResponse { proofs }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ApiConfig;

    #[tokio::test]
    async fn test_get_tic() {
        let config = ApiConfig::default();
        let state = AppState::new(config).await.unwrap();

        let result = get_tic(State(state), Path("test_tic".to_string())).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_get_proof() {
        let result = get_proof(Path("test_proof".to_string())).await;
        assert!(result.is_ok());
    }
}
