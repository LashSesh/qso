//! Trait-based backend abstraction for memory storage

use mef_schemas::MemoryItem;

/// Search result with distance metric
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Memory item
    pub item: MemoryItem,

    /// Distance metric (lower is more similar)
    pub distance: f64,
}

/// Memory backend trait for pluggable implementations
pub trait MemoryBackend: Send + Sync {
    /// Store a memory item
    fn store(&mut self, item: MemoryItem) -> crate::Result<()>;

    /// Retrieve a memory item by ID
    fn get(&self, id: &str) -> crate::Result<Option<MemoryItem>>;

    /// Search for k nearest neighbors
    fn search(&self, query: &[f64], k: usize) -> crate::Result<Vec<SearchResult>>;

    /// Remove a memory item
    fn remove(&mut self, id: &str) -> crate::Result<()>;

    /// Clear all stored items
    fn clear(&mut self) -> crate::Result<()>;

    /// Get count of stored items
    fn count(&self) -> usize;
}
