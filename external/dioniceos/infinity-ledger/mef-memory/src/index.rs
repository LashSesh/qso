//! # Memory Index
//!
//! Provides abstraction for vector database backends with feature flag control.
//!
//! ## SPEC-006 Reference
//!
//! From Part 1, Section 3:
//! - memory.enabled = false (default)
//! - When disabled, all operations are no-ops
//! - paths.memory must be set to enable persistence

use mef_schemas::MemoryItem;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IndexError {
    #[error("Memory index is disabled (memory.enabled=false)")]
    Disabled,

    #[error("Index path not configured")]
    PathNotConfigured,

    #[error("Backend error: {0}")]
    BackendError(String),

    #[error("Invalid vector dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },

    #[error("Invalid spectral signature: {0}")]
    InvalidSignature(String),
}

/// Memory index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Feature flag: enable memory indexing
    pub enabled: bool,

    /// Path to index storage (empty = disabled)
    pub path: Option<String>,

    /// Vector dimension (must be 8 for MEF)
    pub dimension: usize,

    /// Distance metric (cosine, l2, or inner_product)
    pub metric: String,

    /// Backend type (in-memory, faiss, hnswlib, etc.)
    pub backend: String,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            path: None,
            dimension: 8,
            metric: "cosine".to_string(),
            backend: "in-memory".to_string(),
        }
    }
}

/// Memory index abstraction
///
/// Provides a unified interface for vector storage and similarity search
/// across different backend implementations.
///
/// ## Backends
///
/// - `in-memory`: Simple in-memory storage (testing/development)
/// - `faiss`: Facebook AI Similarity Search (high-performance)
/// - `hnswlib`: Hierarchical Navigable Small World graphs
///
/// ## Feature Flags
///
/// When `config.enabled = false`, all operations return immediately without error.
/// This ensures zero overhead when the feature is disabled.
pub struct MemoryIndex {
    config: MemoryConfig,
    // TODO: Add backend-specific state here
    // backend: Box<dyn VectorBackend>,
}

impl MemoryIndex {
    /// Create a new memory index with given configuration
    pub fn new(config: MemoryConfig) -> Result<Self, IndexError> {
        if config.enabled && config.path.is_none() {
            return Err(IndexError::PathNotConfigured);
        }

        // TODO: Initialize backend based on config.backend
        // For now, just validate configuration

        Ok(Self { config })
    }

    /// Upsert a memory item into the index
    ///
    /// ## No-op when disabled
    ///
    /// If `config.enabled = false`, this returns Ok(()) immediately.
    ///
    /// ## Arguments
    ///
    /// * `item` - Memory item with 8D vector and metadata
    ///
    /// ## TODO
    ///
    /// - Implement backend-specific upsert
    /// - Add batch upsert for efficiency
    /// - Add deduplication logic
    pub async fn upsert(&mut self, item: MemoryItem) -> Result<(), IndexError> {
        if !self.config.enabled {
            return Ok(()); // No-op when disabled
        }

        // Validate dimension
        if item.get_vector().len() != self.config.dimension {
            return Err(IndexError::InvalidDimension {
                expected: self.config.dimension,
                actual: item.get_vector().len(),
            });
        }

        // TODO: Call backend.upsert(item)
        tracing::debug!("Memory upsert: {} (TODO: implement backend)", item.id);

        Ok(())
    }

    /// Search for similar vectors
    ///
    /// ## No-op when disabled
    ///
    /// If `config.enabled = false`, returns empty results.
    ///
    /// ## Arguments
    ///
    /// * `query_vector` - 8D query vector
    /// * `top_k` - Number of results to return
    /// * `filters` - Optional metadata filters
    ///
    /// ## Returns
    ///
    /// List of (item_id, distance) tuples, sorted by similarity
    ///
    /// ## TODO
    ///
    /// - Implement backend-specific search
    /// - Add filter support
    /// - Add approximate vs exact search modes
    pub async fn search(
        &self,
        query_vector: &[f64],
        top_k: usize,
        _filters: Option<serde_json::Value>,
    ) -> Result<Vec<(String, f64)>, IndexError> {
        if !self.config.enabled {
            return Ok(Vec::new()); // Empty results when disabled
        }

        // Validate dimension
        if query_vector.len() != self.config.dimension {
            return Err(IndexError::InvalidDimension {
                expected: self.config.dimension,
                actual: query_vector.len(),
            });
        }

        // TODO: Call backend.search(query_vector, top_k, filters)
        tracing::debug!("Memory search: top_k={} (TODO: implement backend)", top_k);

        Ok(Vec::new())
    }

    /// Get a memory item by ID
    ///
    /// ## No-op when disabled
    ///
    /// Returns None if disabled or item not found.
    ///
    /// ## TODO
    ///
    /// - Implement backend-specific retrieval
    pub async fn get(&self, id: &str) -> Result<Option<MemoryItem>, IndexError> {
        if !self.config.enabled {
            return Ok(None);
        }

        // TODO: Call backend.get(id)
        tracing::debug!("Memory get: {} (TODO: implement backend)", id);

        Ok(None)
    }

    /// Delete a memory item by ID
    ///
    /// ## No-op when disabled
    ///
    /// ## TODO
    ///
    /// - Implement backend-specific deletion
    pub async fn delete(&mut self, id: &str) -> Result<(), IndexError> {
        if !self.config.enabled {
            return Ok(());
        }

        // TODO: Call backend.delete(id)
        tracing::debug!("Memory delete: {} (TODO: implement backend)", id);

        Ok(())
    }

    /// Get index statistics
    ///
    /// Returns number of vectors, memory usage, etc.
    ///
    /// ## TODO
    ///
    /// - Implement backend-specific stats
    pub async fn stats(&self) -> Result<serde_json::Value, IndexError> {
        if !self.config.enabled {
            return Ok(serde_json::json!({
                "enabled": false,
                "count": 0,
            }));
        }

        // TODO: Call backend.stats()
        Ok(serde_json::json!({
            "enabled": true,
            "count": 0,
            "note": "TODO: implement backend stats"
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mef_schemas::{PorStatus, SpectralSignature};

    #[test]
    fn test_index_disabled() {
        let config = MemoryConfig::default(); // enabled = false
        let index = MemoryIndex::new(config).unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();

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

        let mut index_mut = index;
        let result = runtime.block_on(index_mut.upsert(item));

        // Should succeed as no-op
        assert!(result.is_ok());
    }

    #[test]
    fn test_index_enabled_no_path() {
        let config = MemoryConfig {
            enabled: true,
            path: None,
            ..Default::default()
        };

        let result = MemoryIndex::new(config);

        // Should fail without path
        assert!(matches!(result, Err(IndexError::PathNotConfigured)));
    }

    #[test]
    fn test_index_search_disabled() {
        let config = MemoryConfig::default();
        let index = MemoryIndex::new(config).unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results = runtime.block_on(index.search(&query, 10, None)).unwrap();

        // Should return empty results
        assert!(results.is_empty());
    }
}
