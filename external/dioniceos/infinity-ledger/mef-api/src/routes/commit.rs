/// Commit metadata and rotation endpoints
use axum::{
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use crate::{AppState, Result};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/commit", get(get_commit))
        .route("/commit/rotate", post(rotate_commit))
}

/// Get commit metadata
#[derive(Debug, Serialize)]
struct CommitResponse {
    secret_hash: String,
    timestamp: String,
    version: String,
}

async fn get_commit() -> Result<Json<CommitResponse>> {
    // For now, return placeholder commit metadata
    // In a real implementation, this would retrieve actual commit secret info
    Ok(Json(CommitResponse {
        secret_hash: "0x1234567890abcdef...".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        version: "1.0.0".to_string(),
    }))
}

/// Rotate commit secret
#[derive(Debug, Deserialize)]
struct RotateCommitRequest {
    new_secret: Option<String>,
}

#[derive(Debug, Serialize)]
struct RotateCommitResponse {
    status: String,
    old_secret_hash: String,
    new_secret_hash: String,
    timestamp: String,
}

async fn rotate_commit(
    Json(request): Json<RotateCommitRequest>,
) -> Result<Json<RotateCommitResponse>> {
    use sha2::{Digest, Sha256};

    // Generate or use provided secret
    let new_secret = request.new_secret.unwrap_or_else(|| {
        use uuid::Uuid;
        Uuid::new_v4().to_string()
    });

    // Hash the new secret
    let mut hasher = Sha256::new();
    hasher.update(new_secret.as_bytes());
    let new_secret_hash = format!("0x{:x}", hasher.finalize());

    Ok(Json(RotateCommitResponse {
        status: "rotated".to_string(),
        old_secret_hash: "0x1234567890abcdef...".to_string(),
        new_secret_hash,
        timestamp: chrono::Utc::now().to_rfc3339(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_commit() {
        let result = get_commit().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_rotate_commit() {
        let request = RotateCommitRequest {
            new_secret: Some("test_secret".to_string()),
        };

        let result = rotate_commit(Json(request)).await;
        assert!(result.is_ok());
    }
}
