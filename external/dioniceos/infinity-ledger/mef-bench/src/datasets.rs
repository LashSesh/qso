/*!
 * Synthetic datasets and helpers for quality and benchmark runs.
 *
 * Migrated from MEF-Core_v1.0/tests/bench/datasets.py
 */

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

/// Generate `count` deterministic spiral vectors in `dim` dimensions
pub fn generate_spiral_points(count: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut points = Vec::with_capacity(count);

    for index in 0..count {
        let theta = (index as f64) * 0.017 + rng.gen::<f64>() * 1e-4;
        let mut vector = Vec::with_capacity(dim);

        for j in 0..(dim - 1) {
            let value = (theta + (j as f64) * 0.1).cos() * (1.0 + (j as f64) * 0.05);
            vector.push(value);
        }
        vector.push(theta);

        points.push(vector);
    }

    points
}

/// Return deterministic IDs and vectors for the spiral benchmark corpus
pub fn build_spiral_corpus(count: usize, seed: u64) -> (Vec<String>, Vec<Vec<f64>>) {
    let vectors = generate_spiral_points(count, 5, seed);
    let ids: Vec<String> = (0..count)
        .map(|index| format!("spiral-{:06}", index))
        .collect();
    (ids, vectors)
}

/// Record for bulk ingestion APIs
#[derive(Debug, Clone)]
pub struct Record {
    pub id: String,
    pub vector: Vec<f64>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Yield payload dictionaries suitable for bulk ingestion APIs
pub fn iter_records<'a>(
    ids: &'a [String],
    vectors: &'a [Vec<f64>],
) -> impl Iterator<Item = Record> + 'a {
    ids.iter()
        .zip(vectors.iter())
        .enumerate()
        .map(|(index, (id, vector))| {
            let mut metadata = HashMap::new();
            metadata.insert("source".to_string(), serde_json::json!("bench"));
            metadata.insert("index".to_string(), serde_json::json!(index));

            Record {
                id: id.clone(),
                vector: vector.clone(),
                metadata,
            }
        })
}

/// Yield records in batches of `size` (last chunk may be smaller)
pub fn chunked(records: Vec<Record>, size: usize) -> impl Iterator<Item = Vec<Record>> {
    let mut batches = Vec::new();
    let mut batch = Vec::with_capacity(size);

    for record in records {
        batch.push(record);
        if batch.len() >= size {
            batches.push(batch);
            batch = Vec::with_capacity(size);
        }
    }

    if !batch.is_empty() {
        batches.push(batch);
    }

    batches.into_iter()
}

/// Create jittered query vectors derived from the provided corpus
pub fn generate_query_vectors(points: &[Vec<f64>], count: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);

    if points.is_empty() {
        return Vec::new();
    }

    let mut queries = Vec::with_capacity(count);

    for _ in 0..count {
        // Choose a random base point
        let base_idx = rng.gen_range(0..points.len());
        let base = &points[base_idx];

        // Add Gaussian jitter
        let jittered: Vec<f64> = base
            .iter()
            .map(|&value| {
                let jitter: f64 = rng.sample(rand_distr::StandardNormal);
                value + jitter * 0.01
            })
            .collect();

        queries.push(jittered);
    }

    queries
}

/// Return the indices of the top-k corpus entries for query
pub fn brute_force_top_k(
    query: &[f64],
    corpus: &[Vec<f64>],
    k: usize,
    metric: &str,
) -> Vec<(usize, f64)> {
    let scorer: fn(&[f64], &[f64]) -> f64 =
        if metric.to_lowercase() == "l2" || metric.to_lowercase() == "l2sq" {
            negative_l2_squared
        } else {
            cosine_similarity
        };

    let mut scored: Vec<(usize, f64)> = corpus
        .iter()
        .enumerate()
        .map(|(idx, vector)| (idx, scorer(query, vector)))
        .collect();

    // Sort by score descending, then by index ascending for ties
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    scored.into_iter().take(k).collect()
}

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|y| y * y).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Compute negative L2 squared distance (higher is better)
pub fn negative_l2_squared(a: &[f64], b: &[f64]) -> f64 {
    -a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_spiral_points() {
        let points = generate_spiral_points(10, 5, 123);
        assert_eq!(points.len(), 10);
        assert_eq!(points[0].len(), 5);

        // Check determinism
        let points2 = generate_spiral_points(10, 5, 123);
        assert_eq!(points, points2);

        // Different seed should give different results
        let points3 = generate_spiral_points(10, 5, 456);
        assert_ne!(points, points3);
    }

    #[test]
    fn test_build_spiral_corpus() {
        let (ids, vectors) = build_spiral_corpus(5, 123);
        assert_eq!(ids.len(), 5);
        assert_eq!(vectors.len(), 5);
        assert_eq!(ids[0], "spiral-000000");
        assert_eq!(ids[4], "spiral-000004");

        // Check determinism
        let (ids2, vectors2) = build_spiral_corpus(5, 123);
        assert_eq!(ids, ids2);
        assert_eq!(vectors, vectors2);
    }

    #[test]
    fn test_iter_records() {
        let ids = vec!["id1".to_string(), "id2".to_string()];
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let records: Vec<Record> = iter_records(&ids, &vectors).collect();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].id, "id1");
        assert_eq!(records[0].vector, vec![1.0, 2.0]);
        assert_eq!(
            records[0].metadata.get("source").unwrap(),
            &serde_json::json!("bench")
        );
        assert_eq!(
            records[0].metadata.get("index").unwrap(),
            &serde_json::json!(0)
        );
    }

    #[test]
    fn test_chunked() {
        let ids = vec![
            "id1".to_string(),
            "id2".to_string(),
            "id3".to_string(),
            "id4".to_string(),
            "id5".to_string(),
        ];
        let vectors = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];

        let records: Vec<Record> = iter_records(&ids, &vectors).collect();
        let batches: Vec<Vec<Record>> = chunked(records, 2).collect();

        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 2);
        assert_eq!(batches[2].len(), 1);
    }

    #[test]
    fn test_chunked_empty() {
        let records = Vec::new();
        let batches: Vec<Vec<Record>> = chunked(records, 2).collect();
        assert_eq!(batches.len(), 0);
    }

    #[test]
    fn test_chunked_exact_fit() {
        let ids = vec![
            "id1".to_string(),
            "id2".to_string(),
            "id3".to_string(),
            "id4".to_string(),
        ];
        let vectors = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];

        let records: Vec<Record> = iter_records(&ids, &vectors).collect();
        let batches: Vec<Vec<Record>> = chunked(records, 2).collect();

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 2);
    }

    #[test]
    fn test_generate_query_vectors() {
        let points = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let queries = generate_query_vectors(&points, 5, 321);
        assert_eq!(queries.len(), 5);
        assert_eq!(queries[0].len(), 3);

        // Check determinism
        let queries2 = generate_query_vectors(&points, 5, 321);
        assert_eq!(queries, queries2);
    }

    #[test]
    fn test_generate_query_vectors_empty() {
        let points: Vec<Vec<f64>> = Vec::new();
        let queries = generate_query_vectors(&points, 5, 321);
        assert_eq!(queries.len(), 0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-10);

        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-10);

        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_negative_l2_squared() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(negative_l2_squared(&a, &b), 0.0);

        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        assert_eq!(negative_l2_squared(&a, &b), -2.0);
    }

    #[test]
    fn test_brute_force_top_k_cosine() {
        let corpus = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];
        let query = vec![1.0, 0.0, 0.0];

        let results = brute_force_top_k(&query, &corpus, 2, "cosine");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // First result should be exact match
        assert!((results[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_brute_force_top_k_l2() {
        let corpus = vec![vec![1.0, 0.0], vec![2.0, 0.0], vec![0.0, 1.0]];
        let query = vec![0.0, 0.0];

        let results = brute_force_top_k(&query, &corpus, 2, "l2");
        assert_eq!(results.len(), 2);
        // Closest points to origin should be [1.0, 0.0] and [0.0, 1.0] (both distance 1)
        // With tie-breaking by index, should get indices 0 and 2
        assert!(results[0].0 == 0 || results[0].0 == 2);
    }

    #[test]
    fn test_brute_force_top_k_more_than_corpus() {
        let corpus = vec![vec![1.0, 0.0], vec![2.0, 0.0]];
        let query = vec![0.0, 0.0];

        let results = brute_force_top_k(&query, &corpus, 10, "cosine");
        assert_eq!(results.len(), 2); // Should only return corpus size
    }
}
