//! MEF Memory - Vector database abstraction
//!
//! This module provides:
//! - Pluggable backend system with trait-based interface
//! - Complete in-memory backend implementation
//! - Feature-gated for zero overhead when disabled
//! - Support for FAISS/HNSW backends (scaffolded for future)

pub mod backend;
pub mod backends;
pub mod index;
pub mod inmemory;
pub mod operations;

// Performance optimization backends
#[cfg(any(
    feature = "stability-filter",
    feature = "ophan-sharding",
    feature = "adaptive-routing",
    feature = "mandorla"
))]
pub mod backends_opt;

pub use backend::{MemoryBackend, SearchResult};
pub use backends::{InMemoryBackend as InMemoryBackendV2, VectorBackend};
pub use index::{MemoryConfig, MemoryIndex};
pub use inmemory::InMemoryBackend;
pub use operations::{
    SearchRequest, SearchResponse, SearchResult as SearchResultV2, UpsertRequest,
};

// Re-export optimization components when features are enabled
#[cfg(feature = "stability-filter")]
pub use backends_opt::{FilteredBackend, FilterStats, StabilityFilter, StabilityFilterConfig};

#[cfg(feature = "ophan-sharding")]
pub use backends_opt::OphanBackend;

#[cfg(feature = "adaptive-routing")]
pub use backends_opt::{AdaptiveRouter, RouterConfig, SearchStrategy};

#[cfg(feature = "mandorla")]
pub use backends_opt::{IndexCoverageStats, MandorlaBackend, MandorlaConfig, MandorlaRefiner};

use mef_schemas::MemoryItem;

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("Backend error: {0}")]
    Backend(String),

    #[error("Item not found: {0}")]
    NotFound(String),

    #[error("Invalid query: {0}")]
    InvalidQuery(String),
}

pub type Result<T> = std::result::Result<T, MemoryError>;

/// Memory store with pluggable backend
pub struct MemoryStore {
    backend: Box<dyn MemoryBackend>,
}

impl MemoryStore {
    /// Create a new memory store with the given backend
    pub fn new(backend: Box<dyn MemoryBackend>) -> Self {
        Self { backend }
    }

    /// Create an in-memory backend store
    #[cfg(feature = "inmemory")]
    pub fn in_memory() -> Self {
        Self::new(Box::new(InMemoryBackend::new()))
    }

    /// Store a memory item
    pub fn store(&mut self, item: MemoryItem) -> Result<()> {
        self.backend.store(item)
    }

    /// Retrieve a memory item by ID
    pub fn get(&self, id: &str) -> Result<Option<MemoryItem>> {
        self.backend.get(id)
    }

    /// Search for similar vectors
    pub fn search(&self, query: &[f64], k: usize) -> Result<Vec<SearchResult>> {
        self.backend.search(query, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mef_schemas::SpectralSignature;

    #[test]
    #[cfg(feature = "inmemory")]
    fn test_memory_store_in_memory() {
        let mut store = MemoryStore::in_memory();

        let val = 1.0 / (8.0_f64).sqrt();
        let vector = vec![val; 8];
        let spectral = SpectralSignature {
            psi: 0.3,
            rho: 0.3,
            omega: 0.4,
        };

        let item = MemoryItem::new("mem_001".to_string(), vector, spectral, None).unwrap();

        let result = store.store(item);
        assert!(result.is_ok());

        let retrieved = store.get("mem_001");
        assert!(retrieved.is_ok());
        assert!(retrieved.unwrap().is_some());
    }
}
