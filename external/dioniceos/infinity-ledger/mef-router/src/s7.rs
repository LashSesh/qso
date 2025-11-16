//! # S7 Permutation Space
//!
//! Generates and selects routes from the S7 permutation space (7! = 5040 routes).
//!
//! ## SPEC-006 Reference
//!
//! From Part 4, Section 1.4:
//! - SLOTS = ["DK", "SW", "PI", "WT", "RES1", "ADAPTER", "RES2"]
//! - Generate all permutations of (1, 2, 3, 4, 5, 6, 7)
//! - Deterministic selection via hash(seed + mesh_metrics)

use mef_schemas::{OperatorSlot, RouteSpec};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Generate all 7! = 5040 permutations of [1, 2, 3, 4, 5, 6, 7]
///
/// ## Performance Note
///
/// This generates all 5040 permutations on each call. For production use,
/// consider caching or lazy generation.
///
/// TODO: Consider implementing as a static or lazy_static for efficiency
pub fn generate_permutations() -> Vec<Vec<u8>> {
    let mut perms = Vec::with_capacity(5040); // 7! = 5040
    let elements = vec![1u8, 2, 3, 4, 5, 6, 7];

    // Heap's algorithm for permutation generation
    fn heaps_algorithm(k: usize, arr: &mut Vec<u8>, output: &mut Vec<Vec<u8>>) {
        if k == 1 {
            output.push(arr.clone());
        } else {
            for i in 0..k {
                heaps_algorithm(k - 1, arr, output);
                if k % 2 == 0 {
                    arr.swap(i, k - 1);
                } else {
                    arr.swap(0, k - 1);
                }
            }
        }
    }

    let mut working = elements;
    heaps_algorithm(7, &mut working, &mut perms);

    perms
}

/// Select a route deterministically based on seed and mesh metrics
///
/// ## Algorithm
///
/// 1. Generate all permutations S7
/// 2. Compute mesh score J(m) from metrics
/// 3. Hash seed concatenated with metrics
/// 4. Compute index: (hash + k) mod |S7|, where k = |J(m)| * 1000 mod |S7|
/// 5. Select permutation at index
///
/// ## SPEC-006 Reference
///
/// From Part 4, Section 1.4
///
/// ## Arguments
///
/// * `seed` - Deterministic seed string
/// * `mesh_metrics` - Mesh topology metrics (betti, lambda_gap, persistence)
///
/// ## Returns
///
/// Selected RouteSpec with deterministic route_id
pub fn select_route(seed: &str, mesh_metrics: &HashMap<String, f64>) -> RouteSpec {
    let perms = generate_permutations();

    // Compute mesh score
    let score = crate::scoring::mesh_score(mesh_metrics);

    // Serialize metrics deterministically (sorted keys)
    let mut keys: Vec<_> = mesh_metrics.keys().cloned().collect();
    keys.sort();
    let metrics_str: String = keys
        .iter()
        .map(|k| format!("{}:{:.6}", k, mesh_metrics.get(k).unwrap_or(&0.0)))
        .collect::<Vec<_>>()
        .join(",");

    // Compute hash
    let base = format!("{}:{}", seed, metrics_str);
    let mut hasher = Sha256::new();
    hasher.update(base.as_bytes());
    let hash_bytes = hasher.finalize();
    let hash_int = u128::from_be_bytes([
        hash_bytes[0],
        hash_bytes[1],
        hash_bytes[2],
        hash_bytes[3],
        hash_bytes[4],
        hash_bytes[5],
        hash_bytes[6],
        hash_bytes[7],
        hash_bytes[8],
        hash_bytes[9],
        hash_bytes[10],
        hash_bytes[11],
        hash_bytes[12],
        hash_bytes[13],
        hash_bytes[14],
        hash_bytes[15],
    ]);

    // Compute index
    let k = ((score.abs() * 1000.0) as usize) % perms.len();
    let idx = ((hash_int as usize) + k) % perms.len();

    // Select permutation
    let sigma = perms[idx].clone();

    // Map to operator slots
    let slots = [
        OperatorSlot::DK,
        OperatorSlot::SW,
        OperatorSlot::PI,
        OperatorSlot::WT,
        OperatorSlot::RES1,
        OperatorSlot::ADAPTER,
        OperatorSlot::RES2,
    ];

    let permutation: Vec<OperatorSlot> = sigma.iter().map(|&i| slots[(i - 1) as usize]).collect();

    // Compute route_id
    let sigma_str = sigma
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let route_str = format!("{}:{}", seed, sigma_str);
    let mut hasher = Sha256::new();
    hasher.update(route_str.as_bytes());
    let route_hash = format!("{:x}", hasher.finalize());
    let route_id = route_hash.chars().take(16).collect();

    RouteSpec::new_extended(route_id, sigma, permutation, score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_permutations_count() {
        let perms = generate_permutations();
        assert_eq!(perms.len(), 5040); // 7!
    }

    #[test]
    fn test_generate_permutations_unique() {
        let perms = generate_permutations();
        let mut seen = std::collections::HashSet::new();

        for perm in &perms {
            let key = format!("{:?}", perm);
            assert!(!seen.contains(&key), "Duplicate permutation found");
            seen.insert(key);
        }
    }

    #[test]
    fn test_generate_permutations_valid() {
        let perms = generate_permutations();

        for perm in &perms {
            assert_eq!(perm.len(), 7);

            // Check all elements are present
            let mut sorted = perm.clone();
            sorted.sort();
            assert_eq!(sorted, vec![1, 2, 3, 4, 5, 6, 7]);
        }
    }

    #[test]
    fn test_select_route_deterministic() {
        let mut metrics = HashMap::new();
        metrics.insert("lambda_gap".to_string(), 0.5);
        metrics.insert("betti".to_string(), 2.0);
        metrics.insert("persistence".to_string(), 0.3);

        let route1 = select_route("test_seed", &metrics);
        let route2 = select_route("test_seed", &metrics);

        assert_eq!(route1.route_id, route2.route_id);
        assert_eq!(route1.sigma, route2.sigma);
        assert_eq!(route1.permutation, route2.permutation);
    }

    #[test]
    fn test_select_route_different_seeds() {
        let mut metrics = HashMap::new();
        metrics.insert("lambda_gap".to_string(), 0.5);

        let route1 = select_route("seed1", &metrics);
        let route2 = select_route("seed2", &metrics);

        // Different seeds should (likely) produce different routes
        // This is probabilistic but with 5040 options, collision is unlikely
        assert_ne!(route1.route_id, route2.route_id);
    }

    #[test]
    fn test_select_route_valid() {
        let metrics = HashMap::new();
        let route = select_route("test", &metrics);

        assert_eq!(route.sigma.as_ref().unwrap().len(), 7);
        assert_eq!(route.permutation.len(), 7);
        assert!(route.validate().is_ok());
    }
}
