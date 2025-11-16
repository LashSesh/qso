//! # Memory Operations
//!
//! Request/response types for memory API operations.

use mef_schemas::MemoryItem;
use serde::{Deserialize, Serialize};

/// Request to upsert memory items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertRequest {
    /// Memory items to upsert
    pub items: Vec<MemoryItem>,
}

/// Request to search memory index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    /// Query vector (8D)
    pub query_vector8: Vec<f64>,

    /// Number of results to return
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Optional metadata filters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<serde_json::Value>,
}

fn default_top_k() -> usize {
    10
}

/// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Matching memory item
    pub item: MemoryItem,

    /// Similarity score (cosine or distance)
    pub score: f64,
}

/// Response from search operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    /// List of results, sorted by relevance
    pub results: Vec<SearchResult>,

    /// Query time in milliseconds
    pub query_time_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use mef_schemas::{PorStatus, SpectralSignature};

    #[test]
    fn test_upsert_request_serialization() {
        let item = MemoryItem::new_extended(
            "test".to_string(),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            SpectralSignature {
                psi: 0.3,
                rho: 0.3,
                omega: 0.4,
            },
            PorStatus::Valid,
            "TIC-123".to_string(),
        );

        let request = UpsertRequest { items: vec![item] };
        let json = serde_json::to_string(&request).unwrap();
        let deserialized: UpsertRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.items.len(), 1);
    }

    #[test]
    fn test_search_request_default() {
        let request = SearchRequest {
            query_vector8: vec![1.0; 8],
            top_k: default_top_k(),
            filters: None,
        };

        assert_eq!(request.top_k, 10);
    }
}
