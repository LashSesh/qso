//! Mesh metric computation
//! J(m) = 0.10·betti + 0.70·λ_gap + 0.20·persistence

use std::collections::HashMap;

/// Mesh metric weights
const WEIGHT_BETTI: f64 = 0.10;
const WEIGHT_LAMBDA_GAP: f64 = 0.70;
const WEIGHT_PERSISTENCE: f64 = 0.20;

/// Compute mesh score from metrics
/// J(m) = 0.10·betti + 0.70·λ_gap + 0.20·persistence
pub fn compute_mesh_score(metrics: &HashMap<String, f64>) -> crate::Result<f64> {
    let betti = metrics
        .get("betti")
        .ok_or_else(|| crate::RouterError::InvalidMetrics("Missing 'betti' metric".to_string()))?;

    let lambda_gap = metrics.get("lambda_gap").ok_or_else(|| {
        crate::RouterError::InvalidMetrics("Missing 'lambda_gap' metric".to_string())
    })?;

    let persistence = metrics.get("persistence").ok_or_else(|| {
        crate::RouterError::InvalidMetrics("Missing 'persistence' metric".to_string())
    })?;

    let score =
        WEIGHT_BETTI * betti + WEIGHT_LAMBDA_GAP * lambda_gap + WEIGHT_PERSISTENCE * persistence;

    Ok(score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_mesh_score() {
        let mut metrics = HashMap::new();
        metrics.insert("betti".to_string(), 2.0);
        metrics.insert("lambda_gap".to_string(), 0.5);
        metrics.insert("persistence".to_string(), 0.3);

        let score = compute_mesh_score(&metrics);
        assert!(score.is_ok());

        // Expected: 0.10*2.0 + 0.70*0.5 + 0.20*0.3 = 0.2 + 0.35 + 0.06 = 0.61
        let expected = 0.61;
        let actual = score.unwrap();
        assert!((actual - expected).abs() < 1e-6);
    }

    #[test]
    fn test_missing_metric() {
        let mut metrics = HashMap::new();
        metrics.insert("betti".to_string(), 2.0);

        let score = compute_mesh_score(&metrics);
        assert!(score.is_err());
    }

    #[test]
    fn test_weights_sum_to_one() {
        let sum = WEIGHT_BETTI + WEIGHT_LAMBDA_GAP + WEIGHT_PERSISTENCE;
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
