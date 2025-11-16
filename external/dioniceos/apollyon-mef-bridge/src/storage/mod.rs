//! Storage integration for cognitive processing results
//!
//! Provides connection between UnifiedCognitiveEngine and MEF Ledger,
//! enabling persistent storage of knowledge objects and processing results.

pub mod ledger_storage;
pub mod memory_storage;

pub use ledger_storage::LedgerStorage;
pub use memory_storage::MemoryStorage;

use crate::unified::CognitiveOutput;
use async_trait::async_trait;
use thiserror::Error;

/// Errors that can occur during storage operations
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Storage initialization failed: {0}")]
    InitializationError(String),

    #[error("Write operation failed: {0}")]
    WriteError(String),

    #[error("Read operation failed: {0}")]
    ReadError(String),

    #[error("Storage not available: {0}")]
    Unavailable(String),

    #[error("Invalid data: {0}")]
    InvalidData(String),
}

/// Result type for storage operations
pub type StorageResult<T> = Result<T, StorageError>;

/// Trait for storage backends
///
/// Implementors provide different storage mechanisms for cognitive processing results.
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Store a cognitive output
    ///
    /// # Arguments
    /// * `output` - The cognitive output to store
    ///
    /// # Returns
    /// Storage ID or error
    async fn store(&mut self, output: &CognitiveOutput) -> StorageResult<String>;

    /// Retrieve a cognitive output by ID
    ///
    /// # Arguments
    /// * `id` - The storage ID
    ///
    /// # Returns
    /// Cognitive output or error
    async fn retrieve(&self, id: &str) -> StorageResult<CognitiveOutput>;

    /// Check if storage is available and healthy
    async fn health_check(&self) -> StorageResult<bool>;

    /// Get storage statistics
    async fn stats(&self) -> StorageResult<StorageStats>;
}

/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStats {
    /// Total number of stored items
    pub total_items: usize,

    /// Total storage size in bytes
    pub total_size_bytes: u64,

    /// Number of successful writes
    pub successful_writes: u64,

    /// Number of failed writes
    pub failed_writes: u64,

    /// Storage backend type
    pub backend_type: String,
}

impl Default for StorageStats {
    fn default() -> Self {
        Self {
            total_items: 0,
            total_size_bytes: 0,
            successful_writes: 0,
            failed_writes: 0,
            backend_type: "unknown".to_string(),
        }
    }
}
