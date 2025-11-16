//! # Mesh Scoring
//!
//! Computes the mesh metric J(m) for route selection.
//!
//! ## SPEC-006 Reference
//!
//! From Part 4, Section 1.4:
//! - J(m) = 0.10*b + 0.70*λ + 0.20*p
//! - b: sum of Betti numbers
//! - λ: spectral gap (lambda_gap)
//! - p: sum of persistence intervals

use std::collections::HashMap;

/// Compute mesh score J(m) from topology metrics
///
/// ## Formula
///
/// ```text
/// J(m) = 0.10 * betti + 0.70 * lambda_gap + 0.20 * persistence
/// ```
///
/// ## Arguments
///
/// * `metrics` - Map of metric names to values
///   - "betti": Sum of Betti numbers (default 0.0)
///   - "lambda_gap": Spectral gap of normalized Laplacian (default 0.0)
///   - "persistence": Sum of persistence intervals (default 0.0)
///
/// ## Returns
///
/// Mesh score as f64
pub fn mesh_score(metrics: &HashMap<String, f64>) -> f64 {
    let betti = metrics.get("betti").copied().unwrap_or(0.0);
    let lambda_gap = metrics.get("lambda_gap").copied().unwrap_or(0.0);
    let persistence = metrics.get("persistence").copied().unwrap_or(0.0);

    // Weights from SPEC-006
    const W_BETTI: f64 = 0.10;
    const W_LAMBDA: f64 = 0.70;
    const W_PERSISTENCE: f64 = 0.20;

    W_BETTI * betti + W_LAMBDA * lambda_gap + W_PERSISTENCE * persistence
}

/// Extract mesh metrics from a generic metrics structure
///
/// Handles different possible representations of the same metrics.
///
/// TODO: Define standardized metric structure with mef-topology
pub fn extract_mesh_metrics(raw: &serde_json::Value) -> HashMap<String, f64> {
    let mut metrics = HashMap::new();

    // Try to extract from various possible structures
    if let Some(obj) = raw.as_object() {
        // Direct fields
        if let Some(val) = obj.get("betti").and_then(|v| v.as_f64()) {
            metrics.insert("betti".to_string(), val);
        }
        if let Some(val) = obj.get("lambda_gap").and_then(|v| v.as_f64()) {
            metrics.insert("lambda_gap".to_string(), val);
        }
        if let Some(val) = obj.get("persistence").and_then(|v| v.as_f64()) {
            metrics.insert("persistence".to_string(), val);
        }

        // Nested under "invariants"
        if let Some(inv) = obj.get("invariants").and_then(|v| v.as_object()) {
            if let Some(val) = inv.get("betti").and_then(|v| v.as_f64()) {
                metrics.insert("betti".to_string(), val);
            }
            if let Some(val) = inv.get("lambda_gap").and_then(|v| v.as_f64()) {
                metrics.insert("lambda_gap".to_string(), val);
            }
            if let Some(val) = inv.get("persistence").and_then(|v| v.as_f64()) {
                metrics.insert("persistence".to_string(), val);
            }
        }
    }

    metrics
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_mesh_score_balanced() {
        let mut metrics = HashMap::new();
        metrics.insert("betti".to_string(), 2.0);
        metrics.insert("lambda_gap".to_string(), 0.5);
        metrics.insert("persistence".to_string(), 0.3);

        let score = mesh_score(&metrics);

        // 0.10 * 2.0 + 0.70 * 0.5 + 0.20 * 0.3 = 0.2 + 0.35 + 0.06 = 0.61
        assert!((score - 0.61).abs() < 1e-9);
    }

    #[test]
    fn test_mesh_score_empty() {
        let metrics = HashMap::new();
        let score = mesh_score(&metrics);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_mesh_score_lambda_dominated() {
        let mut metrics = HashMap::new();
        metrics.insert("lambda_gap".to_string(), 1.0);

        let score = mesh_score(&metrics);

        // 0.70 * 1.0 = 0.70
        assert_eq!(score, 0.70);
    }

    #[test]
    fn test_extract_mesh_metrics_direct() {
        let raw = json!({
            "betti": 2.0,
            "lambda_gap": 0.5,
            "persistence": 0.3
        });

        let metrics = extract_mesh_metrics(&raw);

        assert_eq!(metrics.get("betti"), Some(&2.0));
        assert_eq!(metrics.get("lambda_gap"), Some(&0.5));
        assert_eq!(metrics.get("persistence"), Some(&0.3));
    }

    #[test]
    fn test_extract_mesh_metrics_nested() {
        let raw = json!({
            "invariants": {
                "betti": 3.0,
                "lambda_gap": 0.8
            }
        });

        let metrics = extract_mesh_metrics(&raw);

        assert_eq!(metrics.get("betti"), Some(&3.0));
        assert_eq!(metrics.get("lambda_gap"), Some(&0.8));
    }

    #[test]
    fn test_extract_mesh_metrics_empty() {
        let raw = json!({});
        let metrics = extract_mesh_metrics(&raw);
        assert!(metrics.is_empty());
    }
}
