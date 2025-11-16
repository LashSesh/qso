/// Index management endpoints
use axum::{
    extract::{Query, State},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;

use crate::{error::ApiError, AppState, Result};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/index/providers", get(list_providers))
        .route("/index/build", post(build_index))
        .route("/index/status", get(index_status))
        .route("/debug/search-plan", get(debug_search_plan))
}

/// List index providers
#[derive(Debug, Serialize)]
struct ProvidersResponse {
    providers: HashMap<String, ProviderInfo>,
}

#[derive(Debug, Serialize)]
struct ProviderInfo {
    name: String,
    version: String,
    description: String,
}

async fn list_providers(State(state): State<AppState>) -> Result<Json<ProvidersResponse>> {
    let index_manager = state
        .index_manager
        .lock()
        .map_err(|e| ApiError::Internal(format!("Failed to lock index manager: {}", e)))?;

    let providers_raw = index_manager.list_providers();

    let mut providers = HashMap::new();
    for (name, info) in providers_raw {
        providers.insert(
            name.clone(),
            ProviderInfo {
                name: name.clone(),
                version: info
                    .get("version")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                description: info
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
            },
        );
    }

    Ok(Json(ProvidersResponse { providers }))
}

/// Build index for a collection
#[derive(Debug, Deserialize)]
struct BuildIndexRequest {
    collection: String,
}

#[derive(Debug, Serialize)]
struct BuildIndexResponse {
    collection: String,
    status: String,
    result: JsonValue,
}

async fn build_index(
    State(state): State<AppState>,
    Json(request): Json<BuildIndexRequest>,
) -> Result<Json<BuildIndexResponse>> {
    let mut index_manager = state
        .index_manager
        .lock()
        .map_err(|e| ApiError::Internal(format!("Failed to lock index manager: {}", e)))?;

    let result = index_manager
        .build_index(&request.collection)
        .map_err(|e| ApiError::VectorDB(format!("Failed to build index: {}", e)))?;

    Ok(Json(BuildIndexResponse {
        collection: request.collection,
        status: "built".to_string(),
        result: serde_json::to_value(result).unwrap_or(JsonValue::Null),
    }))
}

/// Get index status
#[derive(Debug, Deserialize)]
struct IndexStatusQuery {
    collection: String,
}

#[derive(Debug, Serialize)]
struct IndexStatusResponse {
    collection: String,
    status: JsonValue,
}

async fn index_status(
    State(state): State<AppState>,
    Query(query): Query<IndexStatusQuery>,
) -> Result<Json<IndexStatusResponse>> {
    let index_manager = state
        .index_manager
        .lock()
        .map_err(|e| ApiError::Internal(format!("Failed to lock index manager: {}", e)))?;

    let status = index_manager.get_index_status(&query.collection);

    Ok(Json(IndexStatusResponse {
        collection: query.collection,
        status: serde_json::to_value(status).unwrap_or(JsonValue::Null),
    }))
}

/// Debug search plan
#[derive(Debug, Serialize)]
struct SearchPlanResponse {
    plan: JsonValue,
}

async fn debug_search_plan(State(state): State<AppState>) -> Result<Json<SearchPlanResponse>> {
    let index_manager = state
        .index_manager
        .lock()
        .map_err(|e| ApiError::Internal(format!("Failed to lock index manager: {}", e)))?;

    let plan = index_manager.last_search_plan();

    Ok(Json(SearchPlanResponse {
        plan: serde_json::to_value(plan).unwrap_or(JsonValue::Null),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ApiConfig;

    #[tokio::test]
    async fn test_list_providers() {
        let config = ApiConfig::default();
        let state = AppState::new(config).await.unwrap();

        let result = list_providers(State(state)).await;
        assert!(result.is_ok());
    }
}
