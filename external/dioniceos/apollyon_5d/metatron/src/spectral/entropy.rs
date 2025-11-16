// Entropy estimation for spectral analysis

/// Analyzer for computing entropy of field patterns
#[derive(Debug, Default, Clone)]
pub struct EntropyAnalyzer;

impl EntropyAnalyzer {
    /// Compute the Shannon entropy of a field pattern
    ///
    /// The field values are treated as a probability distribution
    /// after normalization by their sum.
    pub fn entropy(&self, field: &[f64]) -> f64 {
        let mut probs: Vec<f64> = field.iter().map(|v| v.abs()).collect();
        let sum: f64 = probs.iter().sum();
        if sum == 0.0 {
            return 0.0;
        }
        for p in &mut probs {
            *p /= sum;
        }
        probs.iter().map(|p| -p * (p + 1e-12).log2()).sum::<f64>()
    }

    /// Compute normalized entropy (scaled to [0, 1])
    pub fn normalized_entropy(&self, field: &[f64]) -> f64 {
        let n = field.len();
        if n <= 1 {
            return 0.0;
        }
        let max_entropy = (n as f64).log2();
        let entropy = self.entropy(field);
        entropy / max_entropy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_calculation() {
        let analyzer = EntropyAnalyzer::default();
        let uniform = vec![1.0; 4];
        let entropy = analyzer.entropy(&uniform);
        // Uniform distribution should have maximum entropy
        assert!((entropy - 2.0).abs() < 0.01);
    }

    #[test]
    fn normalized_entropy_bounds() {
        let analyzer = EntropyAnalyzer::default();
        let field = vec![1.0, 2.0, 3.0, 4.0];
        let norm_entropy = analyzer.normalized_entropy(&field);
        assert!(norm_entropy >= 0.0 && norm_entropy <= 1.0);
    }

    #[test]
    fn zero_field_entropy() {
        let analyzer = EntropyAnalyzer::default();
        let zero_field = vec![0.0; 5];
        let entropy = analyzer.entropy(&zero_field);
        assert_eq!(entropy, 0.0);
    }
}
