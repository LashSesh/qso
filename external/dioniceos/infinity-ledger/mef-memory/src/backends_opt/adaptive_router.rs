//! Chronokrator - Adaptive Query Router
//!
//! Dynamically select search strategy based on query characteristics (k, dimension, time budget).

use crate::backend::{MemoryBackend, SearchResult};
use mef_schemas::MemoryItem;

/// Search strategy enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchStrategy {
    /// Brute force O(n)
    Exact,
    /// FAISS/HNSW O(log n)
    Approximate,
    /// Adaptive blend
    Hybrid,
}

/// Chronokrator adaptive router configuration
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// k < threshold → Exact
    pub small_k_threshold: usize,
    /// k > threshold → Approximate
    pub large_k_threshold: usize,
    /// dim > threshold → Approximate
    pub high_dim_threshold: usize,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            small_k_threshold: 10,
            large_k_threshold: 100,
            high_dim_threshold: 128,
        }
    }
}

/// Adaptive query router
pub struct AdaptiveRouter<B: MemoryBackend> {
    backend: B,
    config: RouterConfig,
}

impl<B: MemoryBackend> AdaptiveRouter<B> {
    /// Create a new adaptive router
    pub fn new(backend: B, config: RouterConfig) -> Self {
        Self { backend, config }
    }

    /// Get a reference to the underlying backend
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Get a mutable reference to the underlying backend
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    /// Route query to optimal strategy
    fn route_search(&self, query: &[f64], k: usize) -> crate::Result<Vec<SearchResult>> {
        let strategy = self.select_strategy(query, k);
        
        match strategy {
            SearchStrategy::Exact => {
                // Use backend's native search (already brute-force in InMemory)
                self.backend.search(query, k)
            },
            SearchStrategy::Approximate => {
                // For future FAISS/HNSW backends
                self.backend.search(query, k)
            },
            SearchStrategy::Hybrid => {
                // Blend: exact for small k, approx for remainder
                let exact_k = self.config.small_k_threshold;
                let mut results = self.backend.search(query, exact_k)?;
                
                if k > exact_k {
                    let approx_results = self.backend.search(query, k)?;
                    // Merge results, avoiding duplicates
                    let existing_ids: std::collections::HashSet<_> = 
                        results.iter().map(|r| r.item.id.clone()).collect();
                    
                    for result in approx_results {
                        if !existing_ids.contains(&result.item.id) && results.len() < k {
                            results.push(result);
                        }
                    }
                    
                    results.sort_by(|a, b| a.distance.total_cmp(&b.distance));
                    results.truncate(k);
                }
                
                Ok(results)
            }
        }
    }

    /// Chronokrator decision logic
    fn select_strategy(&self, query: &[f64], k: usize) -> SearchStrategy {
        let dim = query.len();

        // Decision tree based on query profile
        if k < self.config.small_k_threshold {
            SearchStrategy::Exact  // Small k: brute force is faster
        } else if k > self.config.large_k_threshold || dim > self.config.high_dim_threshold {
            SearchStrategy::Approximate  // Large k or high dim: use approximation
        } else {
            SearchStrategy::Hybrid  // Middle ground: blend strategies
        }
    }
}

impl<B: MemoryBackend> MemoryBackend for AdaptiveRouter<B> {
    fn store(&mut self, item: MemoryItem) -> crate::Result<()> {
        self.backend.store(item)
    }

    fn get(&self, id: &str) -> crate::Result<Option<MemoryItem>> {
        self.backend.get(id)
    }

    fn search(&self, query: &[f64], k: usize) -> crate::Result<Vec<SearchResult>> {
        self.route_search(query, k)
    }

    fn remove(&mut self, id: &str) -> crate::Result<()> {
        self.backend.remove(id)
    }

    fn clear(&mut self) -> crate::Result<()> {
        self.backend.clear()
    }

    fn count(&self) -> usize {
        self.backend.count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InMemoryBackend;
    use mef_schemas::SpectralSignature;

    #[test]
    fn test_strategy_selection() {
        let backend = InMemoryBackend::new();
        let router = AdaptiveRouter::new(backend, RouterConfig::default());

        let query = vec![0.0; 8];

        // Small k -> Exact
        assert_eq!(router.select_strategy(&query, 5), SearchStrategy::Exact);

        // Medium k -> Hybrid
        assert_eq!(router.select_strategy(&query, 50), SearchStrategy::Hybrid);

        // Large k -> Approximate
        assert_eq!(router.select_strategy(&query, 150), SearchStrategy::Approximate);

        // High dimension -> Approximate
        let high_dim_query = vec![0.0; 200];
        assert_eq!(router.select_strategy(&high_dim_query, 20), SearchStrategy::Approximate);
    }

    #[test]
    fn test_adaptive_router_basic() {
        let backend = InMemoryBackend::new();
        let mut router = AdaptiveRouter::new(backend, RouterConfig::default());

        let val = 1.0 / (8.0_f64).sqrt();
        let spectral = SpectralSignature {
            psi: 0.9,
            rho: 0.9,
            omega: 0.1,
        };

        // Store items
        for i in 0..20 {
            let item = MemoryItem::new(
                format!("item_{}", i),
                vec![val; 8],
                spectral,
                None,
            ).unwrap();
            router.store(item).unwrap();
        }

        assert_eq!(router.count(), 20);

        // Search with small k (should use Exact strategy)
        let query = vec![val; 8];
        let results = router.search(&query, 5).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_adaptive_router_get() {
        let backend = InMemoryBackend::new();
        let mut router = AdaptiveRouter::new(backend, RouterConfig::default());

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

        router.store(item).unwrap();

        let retrieved = router.get("test_item").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "test_item");
    }

    #[test]
    fn test_adaptive_router_remove() {
        let backend = InMemoryBackend::new();
        let mut router = AdaptiveRouter::new(backend, RouterConfig::default());

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

        router.store(item).unwrap();
        assert_eq!(router.count(), 1);

        router.remove("remove_me").unwrap();
        assert_eq!(router.count(), 0);
    }

    #[test]
    fn test_adaptive_router_clear() {
        let backend = InMemoryBackend::new();
        let mut router = AdaptiveRouter::new(backend, RouterConfig::default());

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
            router.store(item).unwrap();
        }

        assert_eq!(router.count(), 10);
        
        router.clear().unwrap();
        assert_eq!(router.count(), 0);
    }

    #[test]
    fn test_custom_router_config() {
        let backend = InMemoryBackend::new();
        let config = RouterConfig {
            small_k_threshold: 5,
            large_k_threshold: 50,
            high_dim_threshold: 64,
        };
        let router = AdaptiveRouter::new(backend, config);

        let query = vec![0.0; 8];

        // k < 5 -> Exact
        assert_eq!(router.select_strategy(&query, 3), SearchStrategy::Exact);

        // 5 <= k <= 50 -> Hybrid
        assert_eq!(router.select_strategy(&query, 25), SearchStrategy::Hybrid);

        // k > 50 -> Approximate
        assert_eq!(router.select_strategy(&query, 100), SearchStrategy::Approximate);
    }
}
