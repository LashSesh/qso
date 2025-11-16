use axum::{
    extract::{Path, State},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::ApiError;
use mef_knowledge::ExtensionPipeline;
use mef_schemas::{KnowledgeObject, MemoryItem, RouteSpec};

#[derive(Clone)]
pub struct ExtensionState {
    pub pipeline: Arc<tokio::sync::Mutex<ExtensionPipeline>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeriveKnowledgeRequest {
    pub tic_id: String,
    pub route_id: String,
    pub seed_path: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeriveKnowledgeResponse {
    pub mef_id: String,
    pub knowledge: KnowledgeObject,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StoreResponse {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query_vector: Vec<f64>,
    pub k: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub item: MemoryItem,
    pub distance: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SelectRouteRequest {
    pub seed: String,
    pub metrics: HashMap<String, f64>,
}

pub fn router(state: ExtensionState) -> Router {
    Router::new()
        .route("/knowledge/derive", post(derive_knowledge))
        .route("/knowledge/:mef_id", get(get_knowledge))
        .route("/memory/store", post(store_memory))
        .route("/memory/search", post(search_memory))
        .route("/router/select", post(select_route))
        .with_state(state)
}

async fn derive_knowledge(
    State(_state): State<ExtensionState>,
    Json(req): Json<DeriveKnowledgeRequest>,
) -> Result<Json<DeriveKnowledgeResponse>, ApiError> {
    // Placeholder implementation
    // In a full implementation, this would use mef_knowledge::derive_seed and compute_mef_id
    let mef_id = format!(
        "mef_{}",
        uuid::Uuid::new_v4().to_string().replace("-", "")[..16].to_string()
    );

    let knowledge = KnowledgeObject::new(
        mef_id.clone(),
        req.tic_id,
        req.route_id,
        req.seed_path,
        vec![], // Empty derived seed for placeholder
        None,
    );

    Ok(Json(DeriveKnowledgeResponse { mef_id, knowledge }))
}

async fn get_knowledge(
    State(_state): State<ExtensionState>,
    Path(mef_id): Path<String>,
) -> Result<Json<KnowledgeObject>, ApiError> {
    // Placeholder implementation
    // In a full implementation, this would retrieve from storage
    Err(ApiError::NotFound(format!(
        "Knowledge object {} not found",
        mef_id
    )))
}

async fn store_memory(
    State(state): State<ExtensionState>,
    Json(item): Json<MemoryItem>,
) -> Result<Json<StoreResponse>, ApiError> {
    let mut pipeline = state.pipeline.lock().await;

    pipeline
        .store_memory(item)
        .map_err(|e| ApiError::Internal(format!("Failed to store memory: {}", e)))?;

    Ok(Json(StoreResponse {
        success: true,
        message: "Memory item stored successfully".to_string(),
    }))
}

async fn search_memory(
    State(_state): State<ExtensionState>,
    Json(_req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    // Placeholder implementation
    // In a full implementation, this would search the memory store
    Ok(Json(SearchResponse { results: vec![] }))
}

async fn select_route(
    State(state): State<ExtensionState>,
    Json(req): Json<SelectRouteRequest>,
) -> Result<Json<RouteSpec>, ApiError> {
    let pipeline = state.pipeline.lock().await;

    let route = pipeline
        .select_route(&req.seed, &req.metrics)
        .map_err(|e| ApiError::Internal(format!("Failed to select route: {}", e)))?;

    match route {
        Some(route_spec) => Ok(Json(route_spec)),
        None => Err(ApiError::Internal("Router not enabled".to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mef_knowledge::config::{
        BackendConfigs, CacheConfig, DerivationSettings, ExtensionSettings, InMemoryConfig,
        InferenceSettings, KnowledgeConfig, MemoryConfig, RouterConfig, ServiceConfig,
    };

    fn test_config() -> ExtensionSettings {
        ExtensionSettings {
            knowledge: KnowledgeConfig {
                enabled: false,
                inference: InferenceSettings {
                    threshold: 0.5,
                    max_iterations: 100,
                },
                derivation: DerivationSettings {
                    root_seed_env: "MEF_ROOT_SEED".to_string(),
                    default_path_prefix: "MEF".to_string(),
                },
            },
            memory: MemoryConfig {
                enabled: true,
                backend: "inmemory".to_string(),
                backends: BackendConfigs {
                    inmemory: InMemoryConfig { max_items: 10000 },
                    faiss: None,
                    hnsw: None,
                },
            },
            router: RouterConfig {
                enabled: true,
                mode: "inproc".to_string(),
                service: ServiceConfig {
                    url: "http://router-service:8080".to_string(),
                    timeout_ms: 5000,
                },
                cache: CacheConfig {
                    enabled: true,
                    s7_permutations: true,
                },
            },
        }
    }

    #[test]
    fn test_extension_state_creation() {
        let config = test_config();
        let pipeline = ExtensionPipeline::new(config);
        let _state = ExtensionState {
            pipeline: Arc::new(tokio::sync::Mutex::new(pipeline)),
        };
    }
}
