/*!
 * Mandorla Module - Global Decision and Resonance Field
 *
 * This module provides the MandorlaField class, which represents a global
 * decision and resonance field (Vesica Piscis / Mandorla). It couples inputs
 * from various sources (seeds, spiral memory, Gabriel cells, external sensors),
 * aggregates field resonance, and triggers excalibration events when maximum
 * convergence is reached.
 *
 * The field supports both static and dynamic thresholding. With dynamic
 * thresholding, the threshold θ(t) is computed as α·Entropy + β·Variance,
 * allowing adaptive decision triggers based on field coherence.
 */

use ndarray::Array1;

/// Global decision and resonance field (Vesica Piscis / Mandorla)
///
/// Couples inputs from multiple sources, computes field resonance through
/// pairwise cosine similarity, and triggers decisions when resonance exceeds
/// a threshold. Supports both static and dynamic (entropy+variance based)
/// thresholding.
#[derive(Debug, Clone)]
pub struct MandorlaField {
    /// Input vectors stored in the field
    pub inputs: Vec<Array1<f64>>,
    /// Static threshold for backwards compatibility
    pub threshold: f64,
    /// Weight for entropy in dynamic threshold calculation
    pub alpha: f64,
    /// Weight for variance in dynamic threshold calculation
    pub beta: f64,
    /// Current resonance value
    pub resonance: f64,
    /// History of resonance values
    pub history: Vec<f64>,
    /// Current computed threshold (updated by decision_trigger)
    pub current_theta: f64,
}

impl MandorlaField {
    /// Create a new MandorlaField with the given parameters
    ///
    /// # Arguments
    ///
    /// * `threshold` - Static threshold for backwards compatibility (default: 0.985)
    /// * `alpha` - Weight for entropy in dynamic threshold (default: 0.5)
    /// * `beta` - Weight for variance in dynamic threshold (default: 0.5)
    ///
    /// # Returns
    ///
    /// A new MandorlaField instance
    pub fn new(threshold: f64, alpha: f64, beta: f64) -> Self {
        Self {
            inputs: Vec::new(),
            threshold,
            alpha,
            beta,
            resonance: 0.0,
            history: Vec::new(),
            current_theta: threshold,
        }
    }

    /// Add an input vector to the field
    ///
    /// # Arguments
    ///
    /// * `vec` - The input vector to add
    pub fn add_input(&mut self, vec: Array1<f64>) {
        self.inputs.push(vec);
    }

    /// Clear all inputs from the field
    pub fn clear_inputs(&mut self) {
        self.inputs.clear();
    }

    /// Calculate the resonance of the current input field
    ///
    /// Computes the mean pairwise cosine similarity between all input vectors.
    /// Requires at least 2 inputs, returns 0.0 otherwise.
    ///
    /// # Returns
    ///
    /// The mean resonance value (cosine similarity)
    pub fn calc_resonance(&mut self) -> f64 {
        if self.inputs.len() < 2 {
            self.resonance = 0.0;
            return 0.0;
        }

        let mut similarities = Vec::new();
        for i in 0..self.inputs.len() {
            for j in (i + 1)..self.inputs.len() {
                let a = &self.inputs[i];
                let b = &self.inputs[j];

                // Compute cosine similarity: dot(a, b) / (||a|| * ||b||)
                let dot_product = a.dot(b);
                let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
                let norm_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();

                let similarity = dot_product / (norm_a * norm_b + 1e-12);
                similarities.push(similarity);
            }
        }

        self.resonance = similarities.iter().sum::<f64>() / similarities.len() as f64;
        self.history.push(self.resonance);
        self.resonance
    }

    /// Compute the entropy of the current input field
    ///
    /// Each input vector is normalized to yield a probability distribution.
    /// The Shannon entropy is computed across concatenated inputs. If no
    /// inputs are present, zero entropy is returned.
    ///
    /// # Returns
    ///
    /// The Shannon entropy of the field
    pub fn calc_entropy(&self) -> f64 {
        if self.inputs.is_empty() {
            return 0.0;
        }

        // Concatenate all input vectors
        let mut data = Vec::new();
        for input in &self.inputs {
            for &val in input.iter() {
                data.push(val.abs());
            }
        }

        // Normalize to probability distribution
        let sum: f64 = data.iter().sum::<f64>() + 1e-12;
        let probs: Vec<f64> = data.iter().map(|x| x / sum).collect();

        // Compute Shannon entropy: -Σ(p * log2(p))
        let entropy: f64 = probs
            .iter()
            .map(|&p| {
                if p > 1e-12 {
                    -p * (p + 1e-12).log2()
                } else {
                    0.0
                }
            })
            .sum();

        entropy
    }

    /// Compute the variance of the input amplitudes across all inputs
    ///
    /// The variance captures how spread out the current resonance field is.
    ///
    /// # Returns
    ///
    /// The variance of all input values
    pub fn calc_variance(&self) -> f64 {
        if self.inputs.is_empty() {
            return 0.0;
        }

        // Collect all values from all inputs
        let mut all_values = Vec::new();
        for input in &self.inputs {
            for &val in input.iter() {
                all_values.push(val);
            }
        }

        if all_values.is_empty() {
            return 0.0;
        }

        // Calculate mean
        let mean = all_values.iter().sum::<f64>() / all_values.len() as f64;

        // Calculate variance: E[(X - μ)²]
        let variance =
            all_values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / all_values.len() as f64;

        variance
    }

    /// Determine whether a decision should be triggered
    ///
    /// If dynamic threshold parameters α or β are non-zero, the threshold is
    /// computed at each call as θ(t) = α·Entropy + β·Variance. Otherwise the
    /// static threshold value is used. Returns true if the current resonance
    /// exceeds θ(t). The computed threshold is stored in current_theta.
    ///
    /// # Returns
    ///
    /// True if resonance exceeds the threshold, false otherwise
    pub fn decision_trigger(&mut self) -> bool {
        let res = self.calc_resonance();

        // Dynamic threshold if alpha or beta is non-zero
        if self.alpha != 0.0 || self.beta != 0.0 {
            let entropy = self.calc_entropy();
            let variance = self.calc_variance();
            self.current_theta = self.alpha * entropy + self.beta * variance;
        } else {
            self.current_theta = self.threshold;
        }

        res > self.current_theta
    }
}

impl Default for MandorlaField {
    /// Create a default MandorlaField with standard parameters
    fn default() -> Self {
        Self::new(0.985, 0.5, 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_create_default_mandorla() {
        let field = MandorlaField::default();
        assert_eq!(field.threshold, 0.985);
        assert_eq!(field.alpha, 0.5);
        assert_eq!(field.beta, 0.5);
        assert_eq!(field.resonance, 0.0);
        assert!(field.inputs.is_empty());
        assert!(field.history.is_empty());
    }

    #[test]
    fn test_create_custom_mandorla() {
        let field = MandorlaField::new(0.9, 0.3, 0.7);
        assert_eq!(field.threshold, 0.9);
        assert_eq!(field.alpha, 0.3);
        assert_eq!(field.beta, 0.7);
    }

    #[test]
    fn test_add_and_clear_inputs() {
        let mut field = MandorlaField::default();

        field.add_input(array![1.0, 2.0, 3.0]);
        field.add_input(array![4.0, 5.0, 6.0]);
        assert_eq!(field.inputs.len(), 2);

        field.clear_inputs();
        assert!(field.inputs.is_empty());
    }

    #[test]
    fn test_calc_resonance_insufficient_inputs() {
        let mut field = MandorlaField::default();

        // No inputs
        let res = field.calc_resonance();
        assert_eq!(res, 0.0);

        // One input
        field.add_input(array![1.0, 2.0, 3.0]);
        let res = field.calc_resonance();
        assert_eq!(res, 0.0);
    }

    #[test]
    fn test_calc_resonance_identical_vectors() {
        let mut field = MandorlaField::default();

        let v = array![1.0, 2.0, 3.0];
        field.add_input(v.clone());
        field.add_input(v.clone());

        let res = field.calc_resonance();
        // Identical vectors have cosine similarity of 1.0
        assert!((res - 1.0).abs() < 1e-10);
        assert_eq!(field.history.len(), 1);
    }

    #[test]
    fn test_calc_resonance_orthogonal_vectors() {
        let mut field = MandorlaField::default();

        field.add_input(array![1.0, 0.0]);
        field.add_input(array![0.0, 1.0]);

        let res = field.calc_resonance();
        // Orthogonal vectors have cosine similarity of 0.0
        assert!(res.abs() < 1e-10);
    }

    #[test]
    fn test_calc_resonance_opposite_vectors() {
        let mut field = MandorlaField::default();

        field.add_input(array![1.0, 2.0, 3.0]);
        field.add_input(array![-1.0, -2.0, -3.0]);

        let res = field.calc_resonance();
        // Opposite vectors have cosine similarity of -1.0
        assert!((res + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_calc_entropy_empty() {
        let field = MandorlaField::default();
        let entropy = field.calc_entropy();
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_calc_entropy_single_value() {
        let mut field = MandorlaField::default();
        field.add_input(array![1.0]);

        let entropy = field.calc_entropy();
        // Single value has zero entropy
        assert!(entropy.abs() < 1e-10);
    }

    #[test]
    fn test_calc_entropy_uniform() {
        let mut field = MandorlaField::default();
        // Uniform distribution has maximum entropy
        field.add_input(array![1.0, 1.0, 1.0, 1.0]);

        let entropy = field.calc_entropy();
        // For 4 uniform values, entropy should be log2(4) = 2.0
        assert!((entropy - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_calc_variance_empty() {
        let field = MandorlaField::default();
        let variance = field.calc_variance();
        assert_eq!(variance, 0.0);
    }

    #[test]
    fn test_calc_variance_constant() {
        let mut field = MandorlaField::default();
        field.add_input(array![5.0, 5.0, 5.0]);

        let variance = field.calc_variance();
        // Constant values have zero variance
        assert!(variance.abs() < 1e-10);
    }

    #[test]
    fn test_calc_variance_varied() {
        let mut field = MandorlaField::default();
        field.add_input(array![1.0, 2.0, 3.0]);

        let variance = field.calc_variance();
        // Variance of [1, 2, 3] is 2/3 ≈ 0.667
        assert!((variance - 0.6666666).abs() < 0.001);
    }

    #[test]
    fn test_decision_trigger_static_threshold() {
        let mut field = MandorlaField::new(0.5, 0.0, 0.0);

        // Low resonance - should not trigger
        field.add_input(array![1.0, 0.0]);
        field.add_input(array![0.0, 1.0]);
        assert!(!field.decision_trigger());
        assert_eq!(field.current_theta, 0.5);

        // High resonance - should trigger
        field.clear_inputs();
        field.add_input(array![1.0, 2.0, 3.0]);
        field.add_input(array![1.0, 2.0, 3.0]);
        assert!(field.decision_trigger());
    }

    #[test]
    fn test_decision_trigger_dynamic_threshold() {
        let mut field = MandorlaField::new(0.985, 0.5, 0.5);

        field.add_input(array![1.0, 2.0, 3.0, 4.0]);
        field.add_input(array![1.1, 2.1, 3.1, 4.1]);

        let triggered = field.decision_trigger();
        // current_theta should be α·Entropy + β·Variance
        let expected_theta = 0.5 * field.calc_entropy() + 0.5 * field.calc_variance();
        assert!((field.current_theta - expected_theta).abs() < 1e-10);

        // With high similarity, resonance should be high
        assert!(field.resonance > 0.9);
    }

    #[test]
    fn test_resonance_history() {
        let mut field = MandorlaField::default();

        field.add_input(array![1.0, 0.0]);
        field.add_input(array![0.0, 1.0]);
        field.calc_resonance();

        field.add_input(array![1.0, 1.0]);
        field.calc_resonance();

        assert_eq!(field.history.len(), 2);
        assert!(field.history[0].abs() < 1e-10); // Orthogonal vectors
        assert!(field.history[1] > 0.0); // Some similarity
    }
}
