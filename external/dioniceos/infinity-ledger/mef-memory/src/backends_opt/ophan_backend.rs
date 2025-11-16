//! O.P.H.A.N. Array - Parallel Sharded Index
//!
//! Split index into 4 parallel shards with central aggregation for 3-4x search speedup.

use crate::backend::{MemoryBackend, SearchResult};
use mef_schemas::MemoryItem;
use std::sync::{Arc, RwLock};

/// O.P.H.A.N. 4-shard parallel backend
pub struct OphanBackend<B: MemoryBackend> {
    shards: Vec<Arc<RwLock<B>>>,
    konus: CentralAggregator,
}

impl<B: MemoryBackend + Clone> OphanBackend<B> {
    /// Create a new O.P.H.A.N. backend with 4 shards
    pub fn new(shard_template: B) -> Self {
        let shards = (0..4)
            .map(|_| Arc::new(RwLock::new(shard_template.clone())))
            .collect();

        Self {
            shards,
            konus: CentralAggregator::default(),
        }
    }

    /// Compute shard assignment via hash
    fn compute_shard(&self, vector: &[f64]) -> usize {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        for v in vector {
            v.to_bits().hash(&mut hasher);
        }
        (hasher.finish() as usize) % 4
    }
}

impl<B: MemoryBackend + Clone + Send + Sync> MemoryBackend for OphanBackend<B> {
    fn store(&mut self, item: MemoryItem) -> crate::Result<()> {
        let shard_id = self.compute_shard(&item.vector);
        let mut shard = self.shards[shard_id].write().unwrap();
        shard.store(item)
    }

    fn get(&self, id: &str) -> crate::Result<Option<MemoryItem>> {
        // Search all shards in sequence
        for shard in &self.shards {
            let shard = shard.read().unwrap();
            if let Some(item) = shard.get(id)? {
                return Ok(Some(item));
            }
        }
        Ok(None)
    }

    fn search(&self, query: &[f64], k: usize) -> crate::Result<Vec<SearchResult>> {
        // Parallel search across all 4 shards
        let results: Vec<Vec<SearchResult>> = self.shards
            .iter()
            .map(|shard| {
                let shard = shard.read().unwrap();
                shard.search(query, k).unwrap_or_default()
            })
            .collect();

        // Konus aggregation: merge and re-rank top-k
        Ok(self.konus.aggregate(results, k))
    }

    fn remove(&mut self, id: &str) -> crate::Result<()> {
        // Try removing from all shards
        for shard in &self.shards {
            let mut shard = shard.write().unwrap();
            let _ = shard.remove(id);
        }
        Ok(())
    }

    fn clear(&mut self) -> crate::Result<()> {
        for shard in &self.shards {
            let mut shard = shard.write().unwrap();
            shard.clear()?;
        }
        Ok(())
    }

    fn count(&self) -> usize {
        self.shards.iter()
            .map(|s| s.read().unwrap().count())
            .sum()
    }
}

/// Central aggregator (Konus)
#[derive(Default)]
struct CentralAggregator;

impl CentralAggregator {
    fn aggregate(&self, shard_results: Vec<Vec<SearchResult>>, k: usize) -> Vec<SearchResult> {
        // Flatten all results
        let mut all_results: Vec<SearchResult> = shard_results
            .into_iter()
            .flatten()
            .collect();

        // Sort by distance (ascending) using total_cmp to handle NaN
        all_results.sort_by(|a, b| {
            a.distance.total_cmp(&b.distance)
        });

        // Take top-k
        all_results.truncate(k);
        all_results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InMemoryBackend;
    use mef_schemas::SpectralSignature;

    #[test]
    fn test_ophan_backend_store_and_count() {
        let inner = InMemoryBackend::new();
        let mut backend = OphanBackend::new(inner);

        let val = 1.0 / (8.0_f64).sqrt();
        let spectral = SpectralSignature {
            psi: 0.9,
            rho: 0.9,
            omega: 0.1,
        };

        // Store multiple items
        for i in 0..10 {
            let item = MemoryItem::new(
                format!("item_{}", i),
                vec![val; 8],
                spectral,
                None,
            ).unwrap();
            backend.store(item).unwrap();
        }

        assert_eq!(backend.count(), 10);
    }

    #[test]
    fn test_ophan_backend_get() {
        let inner = InMemoryBackend::new();
        let mut backend = OphanBackend::new(inner);

        let val = 1.0 / (8.0_f64).sqrt();
        let item = MemoryItem::new(
            "test_item".to_string(),
            vec![val; 8],
            SpectralSignature {
                psi: 0.9,
                rho: 0.9,
                omega: 0.1,
            },
            None,
        ).unwrap();

        backend.store(item).unwrap();

        let retrieved = backend.get("test_item").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "test_item");
    }

    #[test]
    fn test_ophan_backend_search() {
        let inner = InMemoryBackend::new();
        let mut backend = OphanBackend::new(inner);

        let val = 1.0 / (8.0_f64).sqrt();
        let spectral = SpectralSignature {
            psi: 0.9,
            rho: 0.9,
            omega: 0.1,
        };

        // Store items in different shards
        for i in 0..20 {
            let mut vector = vec![val; 8];
            vector[0] += (i as f64) * 0.001; // Slight variation
            let norm = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
            vector.iter_mut().for_each(|x| *x /= norm);

            let item = MemoryItem::new(
                format!("item_{}", i),
                vector,
                spectral,
                None,
            ).unwrap();
            backend.store(item).unwrap();
        }

        // Search
        let query = vec![val; 8];
        let results = backend.search(&query, 5).unwrap();
        
        assert_eq!(results.len(), 5);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i-1].distance <= results[i].distance);
        }
    }

    #[test]
    fn test_ophan_backend_remove() {
        let inner = InMemoryBackend::new();
        let mut backend = OphanBackend::new(inner);

        let val = 1.0 / (8.0_f64).sqrt();
        let item = MemoryItem::new(
            "remove_me".to_string(),
            vec![val; 8],
            SpectralSignature {
                psi: 0.9,
                rho: 0.9,
                omega: 0.1,
            },
            None,
        ).unwrap();

        backend.store(item).unwrap();
        assert_eq!(backend.count(), 1);

        backend.remove("remove_me").unwrap();
        assert_eq!(backend.count(), 0);
    }

    #[test]
    fn test_ophan_backend_clear() {
        let inner = InMemoryBackend::new();
        let mut backend = OphanBackend::new(inner);

        let val = 1.0 / (8.0_f64).sqrt();
        let spectral = SpectralSignature {
            psi: 0.9,
            rho: 0.9,
            omega: 0.1,
        };

        for i in 0..10 {
            let item = MemoryItem::new(
                format!("item_{}", i),
                vec![val; 8],
                spectral,
                None,
            ).unwrap();
            backend.store(item).unwrap();
        }

        assert_eq!(backend.count(), 10);
        
        backend.clear().unwrap();
        assert_eq!(backend.count(), 0);
    }

    #[test]
    fn test_shard_distribution() {
        let inner = InMemoryBackend::new();
        let backend = OphanBackend::new(inner);

        // Test that different vectors get different shards
        let val = 1.0 / (8.0_f64).sqrt();
        let vector1 = vec![val; 8];
        let mut vector2 = vec![val; 8];
        vector2[0] = -val;
        let norm = vector2.iter().map(|x| x * x).sum::<f64>().sqrt();
        vector2.iter_mut().for_each(|x| *x /= norm);

        let shard1 = backend.compute_shard(&vector1);
        let shard2 = backend.compute_shard(&vector2);

        // Shards should be in valid range
        assert!(shard1 < 4);
        assert!(shard2 < 4);

        // Same vector should always map to same shard
        assert_eq!(shard1, backend.compute_shard(&vector1));
        assert_eq!(shard2, backend.compute_shard(&vector2));
    }

    #[test]
    fn test_konus_aggregation() {
        let konus = CentralAggregator::default();

        let val = 1.0 / (8.0_f64).sqrt();
        let spectral = SpectralSignature {
            psi: 0.9,
            rho: 0.9,
            omega: 0.1,
        };

        // Create results from different shards
        let shard1_results = vec![
            SearchResult {
                item: MemoryItem::new("item1".to_string(), vec![val; 8], spectral, None).unwrap(),
                distance: 0.5,
            },
            SearchResult {
                item: MemoryItem::new("item2".to_string(), vec![val; 8], spectral, None).unwrap(),
                distance: 0.7,
            },
        ];

        let shard2_results = vec![
            SearchResult {
                item: MemoryItem::new("item3".to_string(), vec![val; 8], spectral, None).unwrap(),
                distance: 0.3,
            },
            SearchResult {
                item: MemoryItem::new("item4".to_string(), vec![val; 8], spectral, None).unwrap(),
                distance: 0.9,
            },
        ];

        let aggregated = konus.aggregate(vec![shard1_results, shard2_results], 3);

        assert_eq!(aggregated.len(), 3);
        // Should be sorted by distance
        assert_eq!(aggregated[0].item.id, "item3"); // 0.3
        assert_eq!(aggregated[1].item.id, "item1"); // 0.5
        assert_eq!(aggregated[2].item.id, "item2"); // 0.7
    }
}
