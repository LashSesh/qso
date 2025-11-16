/// Vector database endpoints - search, collections, upsert
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    routing::{get, patch, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use crate::{error::ApiError, models::*, AppState, Result};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/search", post(search))
        .route("/collections", get(list_collections))
        .route("/collections/:name/upsert", post(upsert_collection_vectors))
        .route("/collections/:name/vectors", get(list_collection_vectors))
        .route(
            "/collections/:name/provider",
            patch(update_collection_provider),
        )
        .route("/points/bulk", post(bulk_upsert_points))
        .route("/points/bulk/:job_id", get(bulk_job_status))
}

/// Search for vectors in a collection
async fn search(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>> {
    let start = std::time::Instant::now();

    // Get the index manager (thread-safe access)
    let mut index_manager = state
        .index_manager
        .lock()
        .map_err(|e| ApiError::Internal(format!("Failed to lock index manager: {}", e)))?;

    // Perform search with all required parameters
    let results = index_manager
        .search_vectors(
            &request.collection,
            &request.query_vector,
            request.top_k,
            None, // provider
            None, // mode
            None, // ef_search
        )
        .map_err(|e| ApiError::VectorDB(format!("Search failed: {}", e)))?;

    // Convert results to SearchResult format
    let search_results: Vec<SearchResult> = results
        .into_iter()
        .map(|r| SearchResult {
            id: r
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            score: r.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0),
            metadata: r
                .get("metadata")
                .and_then(|v| v.as_object())
                .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
        })
        .collect();

    let elapsed = start.elapsed().as_secs_f64() * 1000.0; // Convert to ms

    Ok(Json(SearchResponse {
        results: search_results,
        collection: request.collection,
        query_time_ms: elapsed,
    }))
}

/// List all collections
async fn list_collections(State(state): State<AppState>) -> Result<Json<CollectionsResponse>> {
    let index_manager = state
        .index_manager
        .lock()
        .map_err(|e| ApiError::Internal(format!("Failed to lock index manager: {}", e)))?;

    let mut collections = Vec::new();

    for (name, coll_state) in index_manager.collections.iter() {
        // Get dimensions from first vector if available
        let dimensions = coll_state
            .vectors
            .values()
            .next()
            .and_then(|v| v.get("vector"))
            .and_then(|v| v.as_array())
            .map(|arr| arr.len());

        collections.push(CollectionInfo {
            name: name.clone(),
            vectors: coll_state.vectors.len(),
            dimensions,
            provider: coll_state
                .indexes
                .get("provider")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        });
    }

    // Sort by name for deterministic output
    collections.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(Json(CollectionsResponse { collections }))
}

/// Upsert vectors into a collection
#[derive(Debug, Deserialize)]
struct UpsertRequest {
    vectors: Vec<VectorPayload>,
}

#[derive(Debug, Serialize)]
struct UpsertResponse {
    count: usize,
    collection: String,
}

async fn upsert_collection_vectors(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    Json(request): Json<UpsertRequest>,
) -> Result<Json<UpsertResponse>> {
    use mef_vector_db::VectorRecord;

    let mut index_manager = state
        .index_manager
        .lock()
        .map_err(|e| ApiError::Internal(format!("Failed to lock index manager: {}", e)))?;

    // Convert VectorPayload to VectorRecord, using epoch from payload or default to 1 for benchmarking
    let records: Vec<VectorRecord> = request
        .vectors
        .into_iter()
        .map(|v| {
            VectorRecord::new(
                v.id,
                v.vector,
                v.metadata
                    .map(|m| m.into_iter().collect())
                    .unwrap_or_default(),
                v.epoch, // Use epoch from payload
            )
        })
        .collect();

    let count = records.len();

    // Provide default epoch of 1 for records that don't have one
    index_manager
        .upsert_vectors(&collection, records, Some(1), None)
        .map_err(|e| ApiError::VectorDB(format!("Failed to upsert vectors: {}", e)))?;

    Ok(Json(UpsertResponse { count, collection }))
}

/// List vectors in a collection
#[derive(Debug, Deserialize)]
struct ListVectorsQuery {
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    offset: Option<usize>,
}

#[derive(Debug, Serialize)]
struct ListVectorsResponse {
    vectors: Vec<VectorPayload>,
    total: usize,
    collection: String,
}

async fn list_collection_vectors(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    Query(params): Query<ListVectorsQuery>,
) -> Result<Json<ListVectorsResponse>> {
    let index_manager = state
        .index_manager
        .lock()
        .map_err(|e| ApiError::Internal(format!("Failed to lock index manager: {}", e)))?;

    let coll_state = index_manager
        .collections
        .get(&collection)
        .ok_or_else(|| ApiError::NotFound(format!("Collection {} not found", collection)))?;

    let total = coll_state.vectors.len();
    let offset = params.offset.unwrap_or(0);
    let limit = params.limit.unwrap_or(100).min(1000); // Cap at 1000

    // Convert HashMap entries to VectorPayload
    let vectors: Vec<VectorPayload> = coll_state
        .vectors
        .iter()
        .skip(offset)
        .take(limit)
        .map(|(id, vec_data)| {
            let vector = vec_data
                .get("vector")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|x| x.as_f64()).collect())
                .unwrap_or_else(Vec::new);

            let metadata = vec_data
                .get("metadata")
                .and_then(|v| v.as_object())
                .cloned();

            let epoch = vec_data.get("epoch").and_then(|v| v.as_i64());

            VectorPayload {
                id: id.clone(),
                vector,
                metadata,
                epoch,
            }
        })
        .collect();

    Ok(Json(ListVectorsResponse {
        vectors,
        total,
        collection,
    }))
}

/// Update collection provider
#[derive(Debug, Deserialize)]
struct UpdateProviderRequest {
    provider: String,
}

#[derive(Debug, Serialize)]
struct UpdateProviderResponse {
    collection: String,
    provider: String,
    status: String,
}

async fn update_collection_provider(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    Json(request): Json<UpdateProviderRequest>,
) -> Result<Json<UpdateProviderResponse>> {
    let mut index_manager = state
        .index_manager
        .lock()
        .map_err(|e| ApiError::Internal(format!("Failed to lock index manager: {}", e)))?;

    index_manager
        .set_collection_provider(&collection, &request.provider)
        .map_err(|e| ApiError::VectorDB(format!("Failed to set provider: {}", e)))?;

    Ok(Json(UpdateProviderResponse {
        collection,
        provider: request.provider,
        status: "updated".to_string(),
    }))
}

/// Bulk upsert points (async operation)
#[derive(Debug, Deserialize)]
struct BulkPointsRequest {
    collection: String,
    points: Vec<VectorPayload>,
}

#[derive(Debug, Serialize)]
struct BulkPointsResponse {
    job_id: String,
    status: String,
    total_points: usize,
}

async fn bulk_upsert_points(
    State(state): State<AppState>,
    Json(request): Json<BulkPointsRequest>,
) -> Result<(StatusCode, Json<BulkPointsResponse>)> {
    use mef_vector_db::VectorRecord;
    use uuid::Uuid;

    let job_id = Uuid::new_v4().to_string();
    let total_points = request.points.len();

    // For now, we'll process synchronously but could make async in future
    let mut index_manager = state
        .index_manager
        .lock()
        .map_err(|e| ApiError::Internal(format!("Failed to lock index manager: {}", e)))?;

    // Convert points to VectorRecord
    let records: Vec<VectorRecord> = request
        .points
        .into_iter()
        .map(|v| {
            VectorRecord::new(
                v.id,
                v.vector,
                v.metadata
                    .map(|m| m.into_iter().collect())
                    .unwrap_or_default(),
                None, // epoch
            )
        })
        .collect();

    index_manager
        .upsert_vectors(&request.collection, records, None, None)
        .map_err(|e| ApiError::VectorDB(format!("Failed to upsert vectors: {}", e)))?;

    Ok((
        StatusCode::ACCEPTED,
        Json(BulkPointsResponse {
            job_id,
            status: "completed".to_string(),
            total_points,
        }),
    ))
}

/// Get bulk job status
#[derive(Debug, Serialize)]
struct BulkJobStatusResponse {
    job_id: String,
    status: String,
    progress: f64,
    total: usize,
    processed: usize,
}

async fn bulk_job_status(Path(job_id): Path<String>) -> Result<Json<BulkJobStatusResponse>> {
    // For now, all jobs complete immediately, so always return completed
    Ok(Json(BulkJobStatusResponse {
        job_id,
        status: "completed".to_string(),
        progress: 1.0,
        total: 0,
        processed: 0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ApiConfig;

    #[tokio::test]
    async fn test_list_collections() {
        let config = ApiConfig::default();
        let state = AppState::new(config).await.unwrap();

        let result = list_collections(State(state)).await;
        assert!(result.is_ok());
    }
}
