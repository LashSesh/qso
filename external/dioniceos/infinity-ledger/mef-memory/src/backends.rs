//! # Vector Database Backends
//!
//! Pluggable backends for different vector database implementations.
//!
//! ## Supported Backends
//!
//! - `InMemory`: Simple HashMap-based storage (testing/development)
//! - `Faiss`: Facebook AI Similarity Search (TODO: implement when feature enabled)
//! - `Hnswlib`: Hierarchical Navigable Small World (TODO: implement when feature enabled)

use async_trait::async_trait;
use mef_schemas::MemoryItem;
use std::collections::HashMap;

/// Trait for vector database backends
///
/// Defines the interface that all backends must implement.
#[async_trait]
pub trait VectorBackend: Send + Sync {
    /// Initialize the backend
    async fn init(&mut self) -> Result<(), String>;

    /// Insert or update a memory item
    async fn upsert(&mut self, item: MemoryItem) -> Result<(), String>;

    /// Search for similar vectors
    async fn search(
        &self,
        query: &[f64],
        top_k: usize,
        filters: Option<serde_json::Value>,
    ) -> Result<Vec<(String, f64)>, String>;

    /// Get item by ID
    async fn get(&self, id: &str) -> Result<Option<MemoryItem>, String>;

    /// Delete item by ID
    async fn delete(&mut self, id: &str) -> Result<(), String>;

    /// Get statistics
    async fn stats(&self) -> Result<serde_json::Value, String>;
}

/// In-memory backend for testing and development
///
/// This is a simple HashMap-based implementation that stores all vectors in memory.
/// Not suitable for production use with large datasets.
pub struct InMemoryBackend {
    items: HashMap<String, MemoryItem>,
}

impl Default for InMemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryBackend {
    /// Create a new in-memory backend
    pub fn new() -> Self {
        Self {
            items: HashMap::new(),
        }
    }

    /// Compute cosine similarity between two vectors
    fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

#[async_trait]
impl VectorBackend for InMemoryBackend {
    async fn init(&mut self) -> Result<(), String> {
        // No initialization needed for in-memory
        Ok(())
    }

    async fn upsert(&mut self, item: MemoryItem) -> Result<(), String> {
        self.items.insert(item.id.clone(), item);
        Ok(())
    }

    async fn search(
        &self,
        query: &[f64],
        top_k: usize,
        _filters: Option<serde_json::Value>,
    ) -> Result<Vec<(String, f64)>, String> {
        // Brute-force search (O(n) - not efficient for large datasets)
        let mut results: Vec<(String, f64)> = self
            .items
            .iter()
            .map(|(id, item)| {
                let similarity = Self::cosine_similarity(query, item.get_vector());
                (id.clone(), similarity)
            })
            .collect();

        // Sort by similarity (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        results.truncate(top_k);

        Ok(results)
    }

    async fn get(&self, id: &str) -> Result<Option<MemoryItem>, String> {
        Ok(self.items.get(id).cloned())
    }

    async fn delete(&mut self, id: &str) -> Result<(), String> {
        self.items.remove(id);
        Ok(())
    }

    async fn stats(&self) -> Result<serde_json::Value, String> {
        Ok(serde_json::json!({
            "backend": "in-memory",
            "count": self.items.len(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mef_schemas::{PorStatus, SpectralSignature};

    #[tokio::test]
    async fn test_in_memory_backend() {
        let mut backend = InMemoryBackend::new();
        backend.init().await.unwrap();

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

        backend.upsert(item.clone()).await.unwrap();

        let retrieved = backend.get("test").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "test");
    }

    #[tokio::test]
    async fn test_in_memory_search() {
        let mut backend = InMemoryBackend::new();

        let item1 = MemoryItem::new_extended(
            "item1".to_string(),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            SpectralSignature {
                psi: 0.3,
                rho: 0.3,
                omega: 0.4,
            },
            PorStatus::Valid,
            "TIC-1".to_string(),
        );

        let item2 = MemoryItem::new_extended(
            "item2".to_string(),
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            SpectralSignature {
                psi: 0.3,
                rho: 0.3,
                omega: 0.4,
            },
            PorStatus::Valid,
            "TIC-2".to_string(),
        );

        backend.upsert(item1).await.unwrap();
        backend.upsert(item2).await.unwrap();

        let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results = backend.search(&query, 1, None).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "item1");
    }
}
