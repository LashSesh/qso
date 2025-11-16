/// Weight-Transfer (WT) operator implementation.
/// Scale-based weight redistribution across micro, meso, and macro levels.
///
/// WT(v) = Σ_{ℓ∈L} w'_ℓ · P_ℓ(v)
/// where w'_ℓ = (1-γ)w_ℓ + γw̃_ℓ, 0 < γ ≤ 0.5
/// Maintains contractivity through convex combination.
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Scale level for weight redistribution
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScaleLevel {
    Micro,
    Meso,
    Macro,
}

impl ScaleLevel {
    fn as_str(&self) -> &str {
        match self {
            ScaleLevel::Micro => "micro",
            ScaleLevel::Meso => "meso",
            ScaleLevel::Macro => "macro",
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s {
            "micro" => Some(ScaleLevel::Micro),
            "meso" => Some(ScaleLevel::Meso),
            "macro" => Some(ScaleLevel::Macro),
            _ => None,
        }
    }
}

/// Non-expansiveness verification results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonExpansiveResults {
    pub projection_norms: HashMap<String, f64>,
    pub weights_sum_to_one: bool,
    pub empirical_ratios: EmpiricalRatios,
    pub is_non_expansive: bool,
}

/// Empirical ratio statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpiricalRatios {
    pub mean: f64,
    pub max: f64,
}

/// Weight-Transfer operator information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightTransferInfo {
    pub gamma: f64,
    pub levels: Vec<String>,
    pub current_weights: HashMap<String, f64>,
    pub target_weights: HashMap<String, f64>,
    pub is_convex: bool,
    pub non_expansive: NonExpansiveResults,
}

/// Weight-Transfer operator redistributes weights across scales
#[derive(Debug, Clone)]
pub struct WeightTransfer {
    gamma: f64,
    levels: Vec<ScaleLevel>,
    weights: HashMap<ScaleLevel, f64>,
    target_weights: HashMap<ScaleLevel, f64>,
    projections: HashMap<ScaleLevel, Array2<f64>>,
}

impl WeightTransfer {
    /// Create new Weight-Transfer operator
    ///
    /// # Arguments
    /// * `gamma` - Transfer rate (0 < γ ≤ 0.5)
    /// * `levels` - Scale levels to use
    pub fn new(gamma: f64, levels: Vec<ScaleLevel>) -> Self {
        assert!(
            gamma > 0.0 && gamma <= 0.5,
            "Gamma must be in (0, 0.5], got {}",
            gamma
        );

        let mut wt = Self {
            gamma,
            levels: levels.clone(),
            weights: HashMap::new(),
            target_weights: HashMap::new(),
            projections: HashMap::new(),
        };

        wt.initialize_weights();
        wt.initialize_projections();

        wt
    }

    /// Create from config
    pub fn from_config(config: &HashMap<String, String>) -> Self {
        let gamma = config
            .get("gamma")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.1);

        let levels = config
            .get("levels")
            .map(|s| {
                s.split(',')
                    .filter_map(|level| ScaleLevel::from_str(level.trim()))
                    .collect()
            })
            .unwrap_or_else(|| vec![ScaleLevel::Micro, ScaleLevel::Meso, ScaleLevel::Macro]);

        Self::new(gamma, levels)
    }

    /// Initialize scale weights ensuring sum to 1
    fn initialize_weights(&mut self) {
        // Initial uniform weights
        let n_levels = self.levels.len() as f64;
        for level in &self.levels {
            self.weights.insert(level.clone(), 1.0 / n_levels);
        }

        // Target weights for transfer (slight bias towards meso scale)
        self.target_weights.insert(ScaleLevel::Micro, 0.25);
        self.target_weights.insert(ScaleLevel::Meso, 0.45);
        self.target_weights.insert(ScaleLevel::Macro, 0.30);

        // Normalize to ensure sum = 1
        let total: f64 = self.target_weights.values().sum();
        for weight in self.target_weights.values_mut() {
            *weight /= total;
        }
    }

    /// Initialize scale projection matrices (deterministic)
    fn initialize_projections(&mut self) {
        for level in &self.levels {
            let projection = match level {
                ScaleLevel::Micro => {
                    // Micro: focus on individual components
                    Array2::from_diag(&Array1::from(vec![1.2, 0.8, 1.0, 0.9, 1.1]))
                }
                ScaleLevel::Meso => {
                    // Meso: focus on component pairs
                    #[rustfmt::skip]
                    let data = vec![
                        0.7, 0.3, 0.0, 0.0, 0.0,
                        0.3, 0.7, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.6, 0.4, 0.0,
                        0.0, 0.0, 0.4, 0.6, 0.0,
                        0.0, 0.0, 0.0, 0.0, 1.0,
                    ];
                    Array2::from_shape_vec((5, 5), data).unwrap()
                }
                ScaleLevel::Macro => {
                    // Macro: global averaging with diagonal emphasis
                    let mut p = Array2::from_elem((5, 5), 0.2); // 1/5 for off-diagonal
                    for i in 0..5 {
                        p[[i, i]] = 0.4; // Diagonal emphasis
                    }
                    p
                }
            };

            // Normalize to ensure ||P||_2 ≤ 1 (spectral norm)
            let spectral_norm = self.compute_spectral_norm(&projection);
            let normalized = if spectral_norm > 1.0 {
                &projection / spectral_norm
            } else {
                projection
            };

            self.projections.insert(level.clone(), normalized);
        }
    }

    /// Compute spectral norm (largest singular value) approximation
    fn compute_spectral_norm(&self, matrix: &Array2<f64>) -> f64 {
        // For a symmetric matrix, spectral norm = max eigenvalue
        // Use power iteration for approximation
        let n = matrix.nrows();
        let mut v = Array1::from(vec![1.0 / (n as f64).sqrt(); n]);

        for _ in 0..20 {
            // Power iteration
            let mv = matrix.dot(&v);
            let norm = mv.dot(&mv).sqrt();
            if norm > 0.0 {
                v = &mv / norm;
            }
        }

        let mv = matrix.dot(&v);
        mv.dot(&mv).sqrt()
    }

    /// Update weights using transfer rule w'_ℓ = (1-γ)w_ℓ + γw̃_ℓ
    fn update_weights(&mut self) {
        let mut new_weights = HashMap::new();

        for level in &self.levels {
            let old_w = self.weights.get(level).copied().unwrap_or(0.0);
            let target_w = self.target_weights.get(level).copied().unwrap_or(old_w);
            let new_w = (1.0 - self.gamma) * old_w + self.gamma * target_w;
            new_weights.insert(level.clone(), new_w);
        }

        // Normalize to ensure sum = 1
        let total: f64 = new_weights.values().sum();
        self.weights = new_weights
            .into_iter()
            .map(|(k, v)| (k, v / total))
            .collect();
    }

    /// Apply Weight-Transfer operator
    ///
    /// # Arguments
    /// * `v` - Input vector (5D)
    ///
    /// # Returns
    /// Transformed vector with redistributed scale weights
    pub fn apply(&mut self, v: &Array1<f64>) -> Array1<f64> {
        // Update weights
        self.update_weights();

        // Apply weighted sum of scale projections
        let mut result = Array1::zeros(v.len());

        for level in &self.levels {
            if let (Some(projection), Some(&weight)) =
                (self.projections.get(level), self.weights.get(level))
            {
                result += &(projection.dot(v) * weight);
            }
        }

        result
    }

    /// Verify that the operator maintains convexity
    pub fn verify_convexity(&self) -> bool {
        let total_weight: f64 = self.weights.values().sum();
        let all_positive = self.weights.values().all(|&w| w > 0.0);
        (total_weight - 1.0).abs() < 1e-10 && all_positive
    }

    /// Verify non-expansiveness of the operator
    pub fn verify_non_expansive(&mut self) -> NonExpansiveResults {
        use rand::SeedableRng;
        use rand_distr::{Distribution, StandardNormal};

        // Compute projection norms
        let mut projection_norms = HashMap::new();
        for level in &self.levels {
            if let Some(projection) = self.projections.get(level) {
                let norm = self.compute_spectral_norm(projection);
                projection_norms.insert(level.as_str().to_string(), norm);
            }
        }

        // Test empirically
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let normal = StandardNormal;
        let mut test_results = Vec::new();

        for _ in 0..10 {
            let v1 = Array1::from((0..5).map(|_| normal.sample(&mut rng)).collect::<Vec<_>>());
            let v2 = Array1::from((0..5).map(|_| normal.sample(&mut rng)).collect::<Vec<_>>());

            let wt_v1 = self.apply(&v1);
            let wt_v2 = self.apply(&v2);

            let dist_before = (&v2 - &v1).dot(&(&v2 - &v1)).sqrt();
            let dist_after = (&wt_v2 - &wt_v1).dot(&(&wt_v2 - &wt_v1)).sqrt();

            if dist_before > 0.0 {
                let ratio = dist_after / dist_before;
                test_results.push(ratio);
            }
        }

        let mean = test_results.iter().sum::<f64>() / test_results.len() as f64;
        let max = test_results
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        NonExpansiveResults {
            projection_norms,
            weights_sum_to_one: self.verify_convexity(),
            empirical_ratios: EmpiricalRatios { mean, max },
            is_non_expansive: max <= 1.0,
        }
    }

    /// Get individual scale contributions to output
    ///
    /// # Arguments
    /// * `v` - Input vector
    ///
    /// # Returns
    /// Dictionary of scale contributions
    pub fn get_scale_contributions(&self, v: &Array1<f64>) -> HashMap<String, Array1<f64>> {
        let mut contributions = HashMap::new();

        for level in &self.levels {
            if let (Some(projection), Some(&weight)) =
                (self.projections.get(level), self.weights.get(level))
            {
                contributions.insert(level.as_str().to_string(), projection.dot(v) * weight);
            }
        }

        contributions
    }

    /// Get operator information
    pub fn get_info(&mut self) -> WeightTransferInfo {
        let current_weights: HashMap<String, f64> = self
            .weights
            .iter()
            .map(|(k, &v)| (k.as_str().to_string(), v))
            .collect();

        let target_weights: HashMap<String, f64> = self
            .target_weights
            .iter()
            .map(|(k, &v)| (k.as_str().to_string(), v))
            .collect();

        let levels: Vec<String> = self.levels.iter().map(|l| l.as_str().to_string()).collect();

        WeightTransferInfo {
            gamma: self.gamma,
            levels,
            current_weights,
            target_weights,
            is_convex: self.verify_convexity(),
            non_expansive: self.verify_non_expansive(),
        }
    }
}

impl Default for WeightTransfer {
    fn default() -> Self {
        Self::new(
            0.1,
            vec![ScaleLevel::Micro, ScaleLevel::Meso, ScaleLevel::Macro],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_weight_transfer() {
        let wt = WeightTransfer::default();
        assert_eq!(wt.gamma, 0.1);
        assert_eq!(wt.levels.len(), 3);
    }

    #[test]
    fn test_custom_gamma() {
        let wt = WeightTransfer::new(0.3, vec![ScaleLevel::Micro, ScaleLevel::Meso]);
        assert_eq!(wt.gamma, 0.3);
        assert_eq!(wt.levels.len(), 2);
    }

    #[test]
    #[should_panic(expected = "Gamma must be in (0, 0.5]")]
    fn test_invalid_gamma_too_large() {
        WeightTransfer::new(0.6, vec![ScaleLevel::Micro]);
    }

    #[test]
    #[should_panic(expected = "Gamma must be in (0, 0.5]")]
    fn test_invalid_gamma_zero() {
        WeightTransfer::new(0.0, vec![ScaleLevel::Micro]);
    }

    #[test]
    fn test_apply_weight_transfer() {
        let mut wt = WeightTransfer::default();
        let v = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let result = wt.apply(&v);

        assert_eq!(result.len(), 5);
        // Result should be a weighted combination of projections
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_verify_convexity() {
        let wt = WeightTransfer::default();
        assert!(wt.verify_convexity());
    }

    #[test]
    fn test_update_weights() {
        let mut wt = WeightTransfer::default();
        let initial_weights = wt.weights.clone();

        wt.update_weights();

        // Weights should have changed
        assert!(wt.weights != initial_weights);
        // But still sum to 1
        assert!(wt.verify_convexity());
    }

    #[test]
    fn test_verify_non_expansive() {
        let mut wt = WeightTransfer::default();
        let results = wt.verify_non_expansive();

        assert!(results.weights_sum_to_one);
        assert!(results.empirical_ratios.mean >= 0.0);
        assert!(results.empirical_ratios.max >= 0.0);
    }

    #[test]
    fn test_get_scale_contributions() {
        let wt = WeightTransfer::default();
        let v = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let contributions = wt.get_scale_contributions(&v);

        assert_eq!(contributions.len(), 3);
        assert!(contributions.contains_key("micro"));
        assert!(contributions.contains_key("meso"));
        assert!(contributions.contains_key("macro"));
    }

    #[test]
    fn test_get_info() {
        let mut wt = WeightTransfer::default();
        let info = wt.get_info();

        assert_eq!(info.gamma, 0.1);
        assert_eq!(info.levels.len(), 3);
        assert!(info.is_convex);
        assert!(info.current_weights.contains_key("micro"));
    }

    #[test]
    fn test_from_config() {
        let mut config = HashMap::new();
        config.insert("gamma".to_string(), "0.2".to_string());
        config.insert("levels".to_string(), "micro,meso".to_string());

        let wt = WeightTransfer::from_config(&config);

        assert_eq!(wt.gamma, 0.2);
        assert_eq!(wt.levels.len(), 2);
    }
}
