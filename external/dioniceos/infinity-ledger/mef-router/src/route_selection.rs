//! Route selection logic

use crate::mesh_metrics::compute_mesh_score;
use crate::s7_space::generate_s7_permutations;
use mef_schemas::RouteSpec;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Select a route from S7 space deterministically
/// route = S₇[(SHA256(seed||metrics) + k) mod 5040]
pub fn select_route(seed: &str, metrics: &HashMap<String, f64>) -> crate::Result<RouteSpec> {
    // Compute mesh score
    let mesh_score = compute_mesh_score(metrics)?;

    // Generate S7 permutations (cached in real implementation)
    let s7 = generate_s7_permutations();

    // Compute deterministic index
    let mut hasher = Sha256::new();
    hasher.update(seed.as_bytes());
    hasher.update(format!("{:.6}", mesh_score).as_bytes()); // Stable precision
    let hash = hasher.finalize();

    // Convert first 8 bytes to u64
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&hash[..8]);
    let hash_value = u64::from_le_bytes(bytes);

    // Select permutation
    let index = (hash_value as usize) % s7.len();
    let permutation = s7[index].clone();

    // Create route ID
    let route_id = format!("route_{:04}", index);

    RouteSpec::new(route_id, permutation, mesh_score)
        .map_err(|e| crate::RouterError::Selection(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_route() {
        let mut metrics = HashMap::new();
        metrics.insert("betti".to_string(), 2.0);
        metrics.insert("lambda_gap".to_string(), 0.5);
        metrics.insert("persistence".to_string(), 0.3);

        let route = select_route("seed123", &metrics);
        assert!(route.is_ok());

        let r = route.unwrap();
        assert_eq!(r.permutation.len(), 7);
        assert!(r.route_id.starts_with("route_"));
    }

    #[test]
    fn test_determinism() {
        let mut metrics = HashMap::new();
        metrics.insert("betti".to_string(), 2.0);
        metrics.insert("lambda_gap".to_string(), 0.5);
        metrics.insert("persistence".to_string(), 0.3);

        let route1 = select_route("seed123", &metrics).unwrap();
        let route2 = select_route("seed123", &metrics).unwrap();

        assert_eq!(route1.route_id, route2.route_id);
        assert_eq!(route1.permutation, route2.permutation);
    }

    #[test]
    fn test_different_seeds() {
        let mut metrics = HashMap::new();
        metrics.insert("betti".to_string(), 2.0);
        metrics.insert("lambda_gap".to_string(), 0.5);
        metrics.insert("persistence".to_string(), 0.3);

        let route1 = select_route("seed123", &metrics).unwrap();
        let route2 = select_route("seed456", &metrics).unwrap();

        // Different seeds should likely produce different routes
        // (not guaranteed but highly probable)
        assert!(route1.route_id != route2.route_id || route1.permutation != route2.permutation);
    }

    #[test]
    fn test_uniform_distribution() {
        let mut metrics = HashMap::new();
        metrics.insert("betti".to_string(), 2.0);
        metrics.insert("lambda_gap".to_string(), 0.5);
        metrics.insert("persistence".to_string(), 0.3);

        // Test multiple seeds produce different routes
        let mut route_ids = std::collections::HashSet::new();
        for i in 0..100 {
            let seed = format!("seed{}", i);
            let route = select_route(&seed, &metrics).unwrap();
            route_ids.insert(route.route_id);
        }

        // Should have at least 80 unique routes out of 100 (reasonable diversity)
        assert!(route_ids.len() >= 80);
    }
}
