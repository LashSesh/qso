/// Request and response models for API
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Health & Status Models
// ============================================================================

#[derive(Debug, Serialize)]
pub struct PingResponse {
    pub status: String,
    pub version: String,
    pub seed: String,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub timestamp: String,
}

#[derive(Debug, Serialize)]
pub struct ReadyResponse {
    pub ready: bool,
    pub components: HashMap<String, bool>,
}

// ============================================================================
// Ingestion Models
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct IngestRequest {
    pub data: String,
    pub data_type: String,
    pub seed: String,
}

#[derive(Debug, Serialize)]
pub struct IngestResponse {
    pub snapshot_id: String,
    pub phase: f64,
    pub por: String,
    pub hash: String,
    pub timestamp: String,
}

// ============================================================================
// Processing Models
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct ProcessRequest {
    pub snapshot_id: String,
    pub commit: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct ProcessResponse {
    pub tic_id: String,
    pub converged: bool,
    pub iterations: usize,
    pub final_eigenvalue: f64,
    pub timestamp: String,
}

#[derive(Debug, Deserialize)]
pub struct SolveRequest {
    pub snapshot_id: String,
}

#[derive(Debug, Serialize)]
pub struct SolveResponse {
    pub tic_id: String,
    pub status: String,
    pub steps: usize,
}

// ============================================================================
// Validation Models
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct ValidateRequest {
    pub snapshot_id: String,
}

#[derive(Debug, Serialize)]
pub struct ValidateResponse {
    pub snapshot_id: String,
    pub overall_valid: bool,
    pub resonance: ResonanceMetrics,
    pub stability: StabilityMetrics,
}

#[derive(Debug, Serialize)]
pub struct ResonanceMetrics {
    pub fft_resonance: f64,
    pub spectral_gap: f64,
    pub phase_coherence: f64,
}

#[derive(Debug, Serialize)]
pub struct StabilityMetrics {
    pub convergence_rate: f64,
    pub entropy: f64,
}

// ============================================================================
// Ledger Models
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct LedgerAppendRequest {
    pub tic_id: String,
    pub snapshot_id: String,
}

#[derive(Debug, Serialize)]
pub struct LedgerAppendResponse {
    pub block_index: usize,
    pub block_hash: String,
    pub timestamp: String,
}

#[derive(Debug, Serialize)]
pub struct AuditResponse {
    pub valid: bool,
    pub blocks: usize,
    pub chain_hash: String,
    pub statistics: HashMap<String, serde_json::Value>,
}

// ============================================================================
// Vector/Search Models
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub collection: String,
    pub query_vector: Vec<f64>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub membership_proof: bool,
    #[serde(default)]
    pub pipeline_proof: bool,
}

fn default_top_k() -> usize {
    5
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub collection: String,
    pub query_time_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f64,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

// ============================================================================
// Vector Payload Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorPayload {
    pub id: String,
    pub vector: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epoch: Option<i64>,
}

// ============================================================================
// Acquisition Models
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct AcquisitionRequest {
    pub data: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct AcquisitionResponse {
    pub success: bool,
    pub vector_id: String,
    pub collection: String,
}

// ============================================================================
// Collection Models
// ============================================================================

#[derive(Debug, Serialize)]
pub struct CollectionInfo {
    pub name: String,
    pub vectors: usize,
    pub dimensions: Option<usize>,
    pub provider: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CollectionsResponse {
    pub collections: Vec<CollectionInfo>,
}

// ============================================================================
// Metrics Models
// ============================================================================

#[derive(Debug, Serialize)]
pub struct MetricsSnapshot {
    pub operations: HashMap<String, OperationMetrics>,
}

#[derive(Debug, Serialize)]
pub struct OperationMetrics {
    pub count: f64,
    pub window_count: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub avg: f64,
}

// ============================================================================
// Spiral/Embedding Models
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct EmbedRequest {
    pub data: serde_json::Value,
    pub seed: String,
}

#[derive(Debug, Serialize)]
pub struct EmbedResponse {
    pub embedding: Vec<f64>,
    pub phase: f64,
    pub snapshot_id: String,
}

// ============================================================================
// Export Models
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct ExportRequest {
    pub format: String,
    pub start_index: Option<usize>,
    pub end_index: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ExportResponse {
    pub data: serde_json::Value,
    pub format: String,
    pub record_count: usize,
}
