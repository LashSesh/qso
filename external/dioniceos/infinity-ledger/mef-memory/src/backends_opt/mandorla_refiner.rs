//! Mandorla Logic - Query Refinement
//!
//! Refine search space by intersecting query manifold with index coverage (precision boost).

use crate::backend::{MemoryBackend, SearchResult};
use mef_schemas::MemoryItem;

/// Mandorla query refiner configuration
#[derive(Debug, Clone)]
pub struct MandorlaConfig {
    /// Overlap threshold for query-index intersection
    pub overlap_threshold: f64,
}

impl Default for MandorlaConfig {
    fn default() -> Self {
        Self {
            overlap_threshold: 0.7,
        }
    }
}

/// Index coverage statistics
#[derive(Debug, Clone, Default)]
pub struct IndexCoverageStats {
    pub min_vector: Vec<f64>,
    pub max_vector: Vec<f64>,
    pub mean_vector: Vec<f64>,
}

/// Mandorla refiner
pub struct MandorlaRefiner {
    config: MandorlaConfig,
    index_stats: IndexCoverageStats,
}

impl MandorlaRefiner {
    /// Create a new Mandorla refiner
    pub fn new(config: MandorlaConfig) -> Self {
        Self {
            config,
            index_stats: IndexCoverageStats::default(),
        }
    }

    /// Update index statistics
    pub fn update_stats(&mut self, items: &[MemoryItem]) {
        if items.is_empty() {
            return;
        }

        let dim = items[0].vector.len();
        
        let mut min_vec = vec![f64::INFINITY; dim];
        let mut max_vec = vec![f64::NEG_INFINITY; dim];
        let mut mean_vec = vec![0.0; dim];

        for item in items {
            for (i, &v) in item.vector.iter().enumerate() {
                min_vec[i] = min_vec[i].min(v);
                max_vec[i] = max_vec[i].max(v);
                mean_vec[i] += v;
            }
        }

        for v in &mut mean_vec {
            *v /= items.len() as f64;
        }

        self.index_stats = IndexCoverageStats {
            min_vector: min_vec,
            max_vector: max_vec,
            mean_vector: mean_vec,
        };
    }

    /// Refine query to intersection with index space
    pub fn refine_query(&self, query: &[f64]) -> Option<Vec<f64>> {
        if self.index_stats.mean_vector.is_empty() {
            return Some(query.to_vec());  // No stats yet
        }

        // Compute overlap score between query and index coverage
        let overlap = self.compute_overlap(query);

        if overlap >= self.config.overlap_threshold {
            // Query is within index coverage - use as-is
            Some(query.to_vec())
        } else {
            // Query is outside index coverage - project into covered space
            Some(self.project_into_coverage(query))
        }
    }

    fn compute_overlap(&self, query: &[f64]) -> f64 {
        // Compute cosine similarity with index mean
        let dot: f64 = query.iter()
            .zip(&self.index_stats.mean_vector)
            .map(|(q, m)| q * m)
            .sum();

        let query_norm: f64 = query.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mean_norm: f64 = self.index_stats.mean_vector.iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();

        if query_norm * mean_norm > 1e-10 {
            (dot / (query_norm * mean_norm)).abs()
        } else {
            0.0
        }
    }

    fn project_into_coverage(&self, query: &[f64]) -> Vec<f64> {
        // Project query into index bounding box
        query.iter()
            .zip(&self.index_stats.min_vector)
            .zip(&self.index_stats.max_vector)
            .map(|((q, min), max)| q.clamp(*min, *max))
            .collect()
    }
}

/// Backend wrapper with Mandorla query refinement
pub struct MandorlaBackend<B: MemoryBackend> {
    inner: B,
    refiner: MandorlaRefiner,
}

impl<B: MemoryBackend> MandorlaBackend<B> {
    /// Create a new Mandorla backend
    pub fn new(inner: B, refiner: MandorlaRefiner) -> Self {
        Self { inner, refiner }
    }

    /// Get a reference to the refiner
    pub fn refiner(&self) -> &MandorlaRefiner {
        &self.refiner
    }

    /// Get a mutable reference to the refiner
    pub fn refiner_mut(&mut self) -> &mut MandorlaRefiner {
        &mut self.refiner
    }
}

impl<B: MemoryBackend> MemoryBackend for MandorlaBackend<B> {
    fn store(&mut self, item: MemoryItem) -> crate::Result<()> {
        self.inner.store(item)
    }

    fn get(&self, id: &str) -> crate::Result<Option<MemoryItem>> {
        self.inner.get(id)
    }

    fn search(&self, query: &[f64], k: usize) -> crate::Result<Vec<SearchResult>> {
        // Refine query before search
        let refined_query = self.refiner.refine_query(query)
            .unwrap_or_else(|| query.to_vec());
        
        self.inner.search(&refined_query, k)
    }

    fn remove(&mut self, id: &str) -> crate::Result<()> {
        self.inner.remove(id)
    }

    fn clear(&mut self) -> crate::Result<()> {
        self.inner.clear()
    }

    fn count(&self) -> usize {
        self.inner.count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InMemoryBackend;
    use mef_schemas::SpectralSignature;

    #[test]
    fn test_mandorla_refiner_no_stats() {
        let refiner = MandorlaRefiner::new(MandorlaConfig::default());
        
        let query = vec![0.5; 8];
        let refined = refiner.refine_query(&query);
        
        assert!(refined.is_some());
        assert_eq!(refined.unwrap(), query);
    }

    #[test]
    fn test_update_stats() {
        let mut refiner = MandorlaRefiner::new(MandorlaConfig::default());
        
        let val = 1.0 / (8.0_f64).sqrt();
        let spectral = SpectralSignature {
            psi: 0.9,
            rho: 0.9,
            omega: 0.1,
        };

        let items: Vec<_> = (0..10).map(|i| {
            MemoryItem::new(
                format!("item_{}", i),
                vec![val; 8],
                spectral,
                None,
            ).unwrap()
        }).collect();

        refiner.update_stats(&items);

        assert_eq!(refiner.index_stats.mean_vector.len(), 8);
        assert!(!refiner.index_stats.min_vector.is_empty());
        assert!(!refiner.index_stats.max_vector.is_empty());
    }

    #[test]
    fn test_compute_overlap() {
        let mut refiner = MandorlaRefiner::new(MandorlaConfig::default());
        
        let val = 1.0 / (8.0_f64).sqrt();
        let spectral = SpectralSignature {
            psi: 0.9,
            rho: 0.9,
            omega: 0.1,
        };

        let items: Vec<_> = (0..10).map(|i| {
            MemoryItem::new(
                format!("item_{}", i),
                vec![val; 8],
                spectral,
                None,
            ).unwrap()
        }).collect();

        refiner.update_stats(&items);

        // Query similar to mean should have high overlap
        let query = vec![val; 8];
        let overlap = refiner.compute_overlap(&query);
        assert!(overlap > 0.9);
    }

    #[test]
    fn test_project_into_coverage() {
        let mut refiner = MandorlaRefiner::new(MandorlaConfig::default());
        
        let val = 1.0 / (8.0_f64).sqrt();
        let spectral = SpectralSignature {
            psi: 0.9,
            rho: 0.9,
            omega: 0.1,
        };

        let items: Vec<_> = (0..10).map(|i| {
            MemoryItem::new(
                format!("item_{}", i),
                vec![val; 8],
                spectral,
                None,
            ).unwrap()
        }).collect();

        refiner.update_stats(&items);

        // Query outside coverage
        let query = vec![10.0; 8];
        let projected = refiner.project_into_coverage(&query);

        // Projected values should be clamped to index bounds
        for i in 0..8 {
            assert!(projected[i] >= refiner.index_stats.min_vector[i]);
            assert!(projected[i] <= refiner.index_stats.max_vector[i]);
        }
    }

    #[test]
    fn test_mandorla_backend() {
        let backend = InMemoryBackend::new();
        let refiner = MandorlaRefiner::new(MandorlaConfig::default());
        let mut mandorla = MandorlaBackend::new(backend, refiner);

        let val = 1.0 / (8.0_f64).sqrt();
        let spectral = SpectralSignature {
            psi: 0.9,
            rho: 0.9,
            omega: 0.1,
        };

        // Store items
        for i in 0..10 {
            let item = MemoryItem::new(
                format!("item_{}", i),
                vec![val; 8],
                spectral,
                None,
            ).unwrap();
            mandorla.store(item).unwrap();
        }

        assert_eq!(mandorla.count(), 10);

        // Search
        let query = vec![val; 8];
        let results = mandorla.search(&query, 5).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_mandorla_backend_get() {
        let backend = InMemoryBackend::new();
        let refiner = MandorlaRefiner::new(MandorlaConfig::default());
        let mut mandorla = MandorlaBackend::new(backend, refiner);

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

        mandorla.store(item).unwrap();

        let retrieved = mandorla.get("test_item").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "test_item");
    }

    #[test]
    fn test_mandorla_backend_remove() {
        let backend = InMemoryBackend::new();
        let refiner = MandorlaRefiner::new(MandorlaConfig::default());
        let mut mandorla = MandorlaBackend::new(backend, refiner);

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

        mandorla.store(item).unwrap();
        assert_eq!(mandorla.count(), 1);

        mandorla.remove("remove_me").unwrap();
        assert_eq!(mandorla.count(), 0);
    }

    #[test]
    fn test_mandorla_backend_clear() {
        let backend = InMemoryBackend::new();
        let refiner = MandorlaRefiner::new(MandorlaConfig::default());
        let mut mandorla = MandorlaBackend::new(backend, refiner);

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
            mandorla.store(item).unwrap();
        }

        assert_eq!(mandorla.count(), 10);
        
        mandorla.clear().unwrap();
        assert_eq!(mandorla.count(), 0);
    }
}
