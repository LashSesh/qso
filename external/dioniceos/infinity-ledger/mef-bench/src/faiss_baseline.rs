/*!
 * FAISS-backed brute force baseline driver for recall ground truth.
 *
 * Migrated from MEF-Core_v1.0/src/bench/drivers/faiss_baseline.py
 *
 * This implementation uses ndarray for brute-force exact nearest-neighbor search.
 */

use crate::base::{UpsertItem, Vector, VectorStoreDriver};
use anyhow::{Context, Result};
use ndarray::{s, Array1, Array2, Axis};

/// Brute-force exact nearest-neighbor search using ndarray
pub struct FaissBaselineDriver {
    metric: String,
    ids: Vec<String>,
    vectors: Vec<Array1<f32>>,
    dimension: Option<usize>,
}

impl FaissBaselineDriver {
    /// Create a new FAISS baseline driver
    pub fn new(metric: Option<&str>) -> Self {
        let metric = metric.unwrap_or("cosine").to_lowercase();
        Self {
            metric,
            ids: Vec::new(),
            vectors: Vec::new(),
            dimension: None,
        }
    }

    /// Prepare a vector for storage (convert and normalize if needed)
    fn prepare_vector(&mut self, vector: &[f64]) -> Result<Array1<f32>> {
        let array = Array1::from_vec(vector.iter().map(|&v| v as f32).collect());

        if array.ndim() != 1 {
            return Err(anyhow::anyhow!("Vector must be 1-dimensional"));
        }

        if let Some(dim) = self.dimension {
            if array.len() != dim {
                return Err(anyhow::anyhow!(
                    "dimension mismatch: expected {}, received {}",
                    dim,
                    array.len()
                ));
            }
        } else {
            self.dimension = Some(array.len());
        }

        Ok(self.normalize(array))
    }

    /// Prepare a query vector
    fn prepare_query(&mut self, vector: &[f64]) -> Result<Array1<f32>> {
        let array = Array1::from_vec(vector.iter().map(|&v| v as f32).collect());

        if let Some(dim) = self.dimension {
            if array.len() != dim {
                return Err(anyhow::anyhow!(
                    "query dimension mismatch: expected {}, received {}",
                    dim,
                    array.len()
                ));
            }
        } else {
            self.dimension = Some(array.len());
        }

        Ok(self.normalize(array))
    }

    /// Normalize a vector if using cosine/ip metric
    fn normalize(&self, mut array: Array1<f32>) -> Array1<f32> {
        if self.metric == "cosine" || self.metric == "ip" {
            let norm = array.iter().map(|&v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                array.mapv_inplace(|v| v / norm);
            }
        }
        array
    }
}

impl VectorStoreDriver for FaissBaselineDriver {
    fn name(&self) -> &str {
        "faiss-baseline"
    }

    fn metric(&self) -> &str {
        &self.metric
    }

    fn connect(&mut self) -> Result<(), anyhow::Error> {
        // No connection needed for in-memory driver
        Ok(())
    }

    fn clear(&mut self, _namespace: &str) -> Result<(), anyhow::Error> {
        self.ids.clear();
        self.vectors.clear();
        self.dimension = None;
        Ok(())
    }

    fn upsert(
        &mut self,
        items: Vec<UpsertItem>,
        _namespace: &str,
        _batch_size: usize,
    ) -> Result<(), anyhow::Error> {
        for (identifier, vector, _metadata) in items {
            let prepared = self
                .prepare_vector(&vector)
                .context("Failed to prepare vector")?;
            self.ids.push(identifier);
            self.vectors.push(prepared);
        }
        Ok(())
    }

    fn search(
        &self,
        query: &Vector,
        k: usize,
        _namespace: &str,
    ) -> Result<Vec<(String, f64)>, anyhow::Error> {
        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        // Clone self to get a mutable reference for prepare_query
        let mut driver_copy = Self {
            metric: self.metric.clone(),
            ids: self.ids.clone(),
            vectors: Vec::new(), // Don't need to copy vectors
            dimension: self.dimension,
        };

        let vector = driver_copy
            .prepare_query(query)
            .context("Failed to prepare query vector")?;

        // Build matrix from all vectors
        let n_vectors = self.vectors.len();
        let dim = self.dimension.unwrap();
        let mut matrix = Array2::<f32>::zeros((n_vectors, dim));
        for (i, vec) in self.vectors.iter().enumerate() {
            matrix.slice_mut(s![i, ..]).assign(vec);
        }

        // Compute scores based on metric
        let scores = if self.metric == "cosine" || self.metric == "ip" {
            // Inner product (cosine for normalized vectors)
            matrix.dot(&vector)
        } else {
            // L2 distance (negative for ranking)
            let diff = &matrix - &vector;
            -(&diff * &diff).sum_axis(Axis(1))
        };

        // Get top-k indices
        let mut indexed_scores: Vec<(usize, f32)> =
            scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();

        // Sort by score descending
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k and convert to result format
        let top_k = k.min(self.ids.len());
        let results: Vec<(String, f64)> = indexed_scores
            .iter()
            .take(top_k)
            .map(|(idx, score)| (self.ids[*idx].clone(), *score as f64))
            .collect();

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_faiss_driver_creation() {
        let driver = FaissBaselineDriver::new(None);
        assert_eq!(driver.name(), "faiss-baseline");
        assert_eq!(driver.metric(), "cosine");
    }

    #[test]
    fn test_faiss_driver_custom_metric() {
        let driver = FaissBaselineDriver::new(Some("l2"));
        assert_eq!(driver.metric(), "l2");
    }

    #[test]
    fn test_faiss_driver_connect() {
        let mut driver = FaissBaselineDriver::new(None);
        let result = driver.connect();
        assert!(result.is_ok());
    }

    #[test]
    fn test_faiss_driver_clear() {
        let mut driver = FaissBaselineDriver::new(None);

        // Add some data
        let items = vec![
            ("id1".to_string(), vec![1.0, 2.0, 3.0], None),
            ("id2".to_string(), vec![4.0, 5.0, 6.0], None),
        ];
        driver.upsert(items, "test", 1000).unwrap();
        assert_eq!(driver.ids.len(), 2);

        // Clear
        driver.clear("test").unwrap();
        assert_eq!(driver.ids.len(), 0);
        assert_eq!(driver.vectors.len(), 0);
        assert!(driver.dimension.is_none());
    }

    #[test]
    fn test_faiss_driver_upsert() {
        let mut driver = FaissBaselineDriver::new(None);

        let items = vec![
            ("id1".to_string(), vec![1.0, 2.0, 3.0], None),
            ("id2".to_string(), vec![4.0, 5.0, 6.0], None),
        ];

        let result = driver.upsert(items, "test", 1000);
        assert!(result.is_ok());
        assert_eq!(driver.ids.len(), 2);
        assert_eq!(driver.vectors.len(), 2);
        assert_eq!(driver.dimension, Some(3));
    }

    #[test]
    fn test_faiss_driver_dimension_mismatch() {
        let mut driver = FaissBaselineDriver::new(None);

        let items = vec![("id1".to_string(), vec![1.0, 2.0, 3.0], None)];
        driver.upsert(items, "test", 1000).unwrap();

        // Try to insert vector with different dimension
        let items2 = vec![("id2".to_string(), vec![1.0, 2.0], None)];
        let result = driver.upsert(items2, "test", 1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_faiss_driver_search_cosine() {
        let mut driver = FaissBaselineDriver::new(Some("cosine"));

        let items = vec![
            ("id1".to_string(), vec![1.0, 0.0, 0.0], None),
            ("id2".to_string(), vec![0.0, 1.0, 0.0], None),
            ("id3".to_string(), vec![1.0, 1.0, 0.0], None),
        ];
        driver.upsert(items, "test", 1000).unwrap();

        let query = vec![1.0, 0.5, 0.0];
        let results = driver.search(&query, 2, "test").unwrap();

        assert_eq!(results.len(), 2);
        // First result should be id3 (closest to query)
        assert_eq!(results[0].0, "id3");
    }

    #[test]
    fn test_faiss_driver_search_empty() {
        let driver = FaissBaselineDriver::new(None);
        let query = vec![1.0, 2.0, 3.0];
        let results = driver.search(&query, 10, "test").unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_faiss_driver_search_l2() {
        let mut driver = FaissBaselineDriver::new(Some("l2"));

        let items = vec![
            ("id1".to_string(), vec![1.0, 0.0, 0.0], None),
            ("id2".to_string(), vec![0.0, 1.0, 0.0], None),
            ("id3".to_string(), vec![0.0, 0.0, 1.0], None),
        ];
        driver.upsert(items, "test", 1000).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = driver.search(&query, 3, "test").unwrap();

        assert_eq!(results.len(), 3);
        // First result should be id1 (exact match)
        assert_eq!(results[0].0, "id1");
    }
}
