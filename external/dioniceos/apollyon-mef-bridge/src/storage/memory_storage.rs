//! In-memory storage backend for development and testing

use super::{StorageBackend, StorageError, StorageResult, StorageStats};
use crate::unified::CognitiveOutput;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// In-memory storage backend
///
/// Stores cognitive outputs in memory using a HashMap.
/// Useful for testing and development, not suitable for production.
#[derive(Clone)]
pub struct MemoryStorage {
    store: Arc<RwLock<HashMap<String, CognitiveOutput>>>,
    stats: Arc<RwLock<StorageStats>>,
}

impl MemoryStorage {
    /// Create a new memory storage backend
    pub fn new() -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(StorageStats {
                backend_type: "memory".to_string(),
                ..Default::default()
            })),
        }
    }

    /// Clear all stored data
    pub fn clear(&mut self) {
        if let Ok(mut store) = self.store.write() {
            store.clear();
        }
        if let Ok(mut stats) = self.stats.write() {
            *stats = StorageStats {
                backend_type: "memory".to_string(),
                ..Default::default()
            };
        }
    }

    /// Get the number of items in storage
    pub fn len(&self) -> usize {
        self.store.read().map(|s| s.len()).unwrap_or(0)
    }

    /// Check if storage is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl StorageBackend for MemoryStorage {
    async fn store(&mut self, output: &CognitiveOutput) -> StorageResult<String> {
        // Generate storage ID from knowledge object
        let storage_id = if let Some(ref knowledge) = output.knowledge {
            knowledge.mef_id.clone()
        } else {
            // Fallback to TIC ID if no knowledge object
            format!("MEM-{}", chrono::Utc::now().timestamp_millis())
        };

        // Estimate size (rough approximation)
        let size = std::mem::size_of_val(output) as u64;

        // Store the output
        match self.store.write() {
            Ok(mut store) => {
                store.insert(storage_id.clone(), output.clone());

                // Update stats
                if let Ok(mut stats) = self.stats.write() {
                    stats.total_items = store.len();
                    stats.total_size_bytes += size;
                    stats.successful_writes += 1;
                }

                Ok(storage_id)
            }
            Err(e) => {
                // Update failure stats
                if let Ok(mut stats) = self.stats.write() {
                    stats.failed_writes += 1;
                }
                Err(StorageError::WriteError(format!(
                    "Failed to acquire write lock: {}",
                    e
                )))
            }
        }
    }

    async fn retrieve(&self, id: &str) -> StorageResult<CognitiveOutput> {
        match self.store.read() {
            Ok(store) => store
                .get(id)
                .cloned()
                .ok_or_else(|| StorageError::ReadError(format!("Item not found: {}", id))),
            Err(e) => Err(StorageError::ReadError(format!(
                "Failed to acquire read lock: {}",
                e
            ))),
        }
    }

    async fn health_check(&self) -> StorageResult<bool> {
        // Memory storage is always healthy if we can access it
        match self.store.read() {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    async fn stats(&self) -> StorageResult<StorageStats> {
        match self.stats.read() {
            Ok(stats) => Ok(stats.clone()),
            Err(e) => Err(StorageError::ReadError(format!(
                "Failed to read stats: {}",
                e
            ))),
        }
    }
}

// CognitiveOutput needs to be Clone for memory storage
impl Clone for CognitiveOutput {
    fn clone(&self) -> Self {
        Self {
            trajectory: self.trajectory.clone(),
            spectral_signature: self.spectral_signature.clone(),
            route: self.route.clone(),
            proof: self.proof.clone(),
            gate_decision: self.gate_decision,
            knowledge: self.knowledge.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_5d::{State5D, SystemParameters};
    use mef_schemas::{GateDecision, KnowledgeObject, RouteSpec, SpectralSignature};

    fn create_test_output() -> CognitiveOutput {
        let knowledge = KnowledgeObject::new(
            "TEST-MEF-001".to_string(),
            "TIC-001".to_string(),
            "ROUTE-001".to_string(),
            "MEF/test/0001".to_string(),
            vec![1, 2, 3],
            None,
        );

        CognitiveOutput {
            trajectory: vec![State5D::new(1.0, 0.0, 0.0, 0.0, 0.0)],
            spectral_signature: SpectralSignature {
                psi: 0.5,
                rho: 0.7,
                omega: 2.1,
            },
            route: RouteSpec::new("ROUTE-001".to_string(), vec![0, 1, 2, 3, 4, 5, 6], 0.8)
                .unwrap(),
            proof: Default::default(),
            gate_decision: GateDecision::FIRE,
            knowledge: Some(knowledge),
        }
    }

    #[tokio::test]
    async fn test_memory_storage_store_retrieve() {
        let mut storage = MemoryStorage::new();
        let output = create_test_output();

        let id = storage.store(&output).await.unwrap();
        let retrieved = storage.retrieve(&id).await.unwrap();

        assert_eq!(retrieved.knowledge, output.knowledge);
    }

    #[tokio::test]
    async fn test_memory_storage_stats() {
        let mut storage = MemoryStorage::new();
        let output = create_test_output();

        storage.store(&output).await.unwrap();

        let stats = storage.stats().await.unwrap();
        assert_eq!(stats.total_items, 1);
        assert_eq!(stats.successful_writes, 1);
        assert_eq!(stats.failed_writes, 0);
    }

    #[tokio::test]
    async fn test_memory_storage_clear() {
        let mut storage = MemoryStorage::new();
        let output = create_test_output();

        storage.store(&output).await.unwrap();
        assert_eq!(storage.len(), 1);

        storage.clear();
        assert_eq!(storage.len(), 0);
    }

    #[tokio::test]
    async fn test_memory_storage_health_check() {
        let storage = MemoryStorage::new();
        let healthy = storage.health_check().await.unwrap();
        assert!(healthy);
    }
}
