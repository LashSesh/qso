/// Ledger endpoints - blockchain operations and auditing
use axum::{
    extract::{Path, State},
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use serde_json::json;
use std::collections::HashMap;

use crate::{error::ApiError, models::*, AppState, Result};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/ledger", post(append_ledger))
        .route("/ledger/:index", get(get_block))
        .route("/audit", get(audit))
}

/// Append a new block to the ledger
async fn append_ledger(
    State(state): State<AppState>,
    Json(request): Json<LedgerAppendRequest>,
) -> Result<Json<LedgerAppendResponse>> {
    // Create JSON values for TIC and snapshot with proper structure
    // In a real implementation, these would be loaded from storage
    let tic_json = json!({
        "tic_id": request.tic_id,
        "seed": "default_seed",
        "fixpoint": [1.0, 0.0, 0.0],
        "window": ["w1", "w2"],
        "invariants": {},
        "sigma_bar": {},
        "proof": null,
    });
    let snapshot_json = json!({ "id": request.snapshot_id });

    // Append to ledger - need to lock the mutex
    let mut ledger = state
        .ledger
        .lock()
        .map_err(|e| ApiError::Ledger(format!("Failed to lock ledger: {}", e)))?;

    let block = ledger
        .append_block(&tic_json, &snapshot_json)
        .map_err(|e| ApiError::Ledger(format!("Failed to append block: {}", e)))?;

    Ok(Json(LedgerAppendResponse {
        block_index: block.index as usize,
        block_hash: block.hash.clone(),
        timestamp: Utc::now().to_rfc3339(),
    }))
}

/// Get a specific block by index
async fn get_block(
    State(state): State<AppState>,
    Path(index): Path<usize>,
) -> Result<Json<serde_json::Value>> {
    let ledger = state
        .ledger
        .lock()
        .map_err(|e| ApiError::Ledger(format!("Failed to lock ledger: {}", e)))?;

    let block = ledger
        .get_block(index as i32)
        .map_err(|e| ApiError::NotFound(format!("Failed to get block: {}", e)))?
        .ok_or_else(|| ApiError::NotFound(format!("Block {} not found", index)))?;

    Ok(Json(json!({
        "index": block.index,
        "hash": block.hash,
        "previous_hash": block.previous_hash,
        "timestamp": block.timestamp,
        "tic_id": block.tic_id,
        "snapshot_hash": block.snapshot_hash,
    })))
}

/// Audit the entire ledger
async fn audit(State(state): State<AppState>) -> Result<Json<AuditResponse>> {
    let ledger = state
        .ledger
        .lock()
        .map_err(|e| ApiError::Ledger(format!("Failed to lock ledger: {}", e)))?;

    // Validate the chain from start
    let valid = ledger
        .verify_chain_integrity(0)
        .map_err(|e| ApiError::Ledger(format!("Chain validation failed: {}", e)))?;

    // Get chain statistics
    let stats = ledger
        .get_chain_statistics()
        .map_err(|e| ApiError::Ledger(format!("Failed to get statistics: {}", e)))?;

    let blocks = stats.total_blocks as usize;
    let chain_hash = ledger
        .get_last_hash()
        .unwrap_or_else(|_| "empty".to_string());

    let mut statistics = HashMap::new();
    statistics.insert("blocks".to_string(), json!(blocks));
    statistics.insert("valid".to_string(), json!(valid));
    statistics.insert("total_size_mb".to_string(), json!(stats.total_size_mb));

    Ok(Json(AuditResponse {
        valid,
        blocks,
        chain_hash,
        statistics,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ApiConfig;

    #[tokio::test]
    async fn test_append_ledger() {
        let config = ApiConfig::default();
        let state = AppState::new(config).await.unwrap();

        let request = LedgerAppendRequest {
            tic_id: "tic_123".to_string(),
            snapshot_id: "snapshot_456".to_string(),
        };

        let result = append_ledger(State(state), Json(request)).await;
        if let Err(e) = &result {
            eprintln!("Error: {:?}", e);
        }
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_audit() {
        let config = ApiConfig::default();
        let state = AppState::new(config).await.unwrap();

        let result = audit(State(state)).await;
        assert!(result.is_ok());
    }
}
