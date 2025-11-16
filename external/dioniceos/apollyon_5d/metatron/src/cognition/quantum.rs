use num_complex::Complex64;
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;

use crate::geometry::operators::permutation_matrix;

#[derive(Debug, Clone)]
pub struct QuantumState {
    pub amplitudes: Vec<Complex64>,
}

impl QuantumState {
    pub fn new(mut amplitudes: Vec<Complex64>, normalize: bool) -> Self {
        if amplitudes.len() > 13 {
            amplitudes.truncate(13);
        }
        while amplitudes.len() < 13 {
            amplitudes.push(Complex64::new(0.0, 0.0));
        }
        let mut state = Self { amplitudes };
        if normalize {
            state.normalise();
        }
        state
    }

    pub fn normalise(&mut self) {
        let norm = self
            .amplitudes
            .iter()
            .map(|c| c.norm_sqr())
            .sum::<f64>()
            .sqrt();
        if norm > 0.0 {
            for amp in &mut self.amplitudes {
                *amp /= norm;
            }
        }
    }

    pub fn inner_product(&self, other: &QuantumState) -> Complex64 {
        self.amplitudes
            .iter()
            .zip(other.amplitudes.iter())
            .map(|(a, b)| a.conj() * b)
            .sum()
    }

    pub fn apply(&self, operator: &QuantumOperator) -> QuantumState {
        let mut result = vec![Complex64::new(0.0, 0.0); 13];
        for i in 0..13 {
            for j in 0..13 {
                result[i] += operator.matrix[i][j] * self.amplitudes[j];
            }
        }
        QuantumState::new(result, true)
    }

    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|c| c.norm_sqr()).collect()
    }

    pub fn measure(&mut self) -> usize {
        let probs = self.probabilities();
        let dist = WeightedIndex::new(&probs).unwrap();
        let mut rng = thread_rng();
        let idx = dist.sample(&mut rng);
        self.amplitudes = vec![Complex64::new(0.0, 0.0); 13];
        self.amplitudes[idx] = Complex64::new(1.0, 0.0);
        idx + 1
    }
}

#[derive(Debug, Clone)]
pub struct QuantumOperator {
    pub matrix: Vec<Vec<Complex64>>,
}

impl QuantumOperator {
    pub fn new(matrix: Vec<Vec<Complex64>>) -> Self {
        assert!(matrix.len() == 13 && matrix.iter().all(|row| row.len() == 13));
        Self { matrix }
    }

    pub fn from_permutation(sigma: &[usize]) -> Self {
        let mat = permutation_matrix(sigma, 13);
        let matrix = mat
            .into_iter()
            .map(|row| row.into_iter().map(|v| Complex64::new(v, 0.0)).collect())
            .collect();
        Self { matrix }
    }

    pub fn compose(&self, other: &QuantumOperator) -> QuantumOperator {
        let mut result = vec![vec![Complex64::new(0.0, 0.0); 13]; 13];
        for i in 0..13 {
            for j in 0..13 {
                for k in 0..13 {
                    result[i][j] += self.matrix[i][k] * other.matrix[k][j];
                }
            }
        }
        QuantumOperator { matrix: result }
    }

    pub fn is_unitary(&self, atol: f64) -> bool {
        for i in 0..13 {
            for j in 0..13 {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..13 {
                    sum += self.matrix[i][k] * self.matrix[j][k].conj();
                }
                if i == j {
                    if (sum - Complex64::new(1.0, 0.0)).norm() > atol {
                        return false;
                    }
                } else if sum.norm() > atol {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_normalisation() {
        let mut state = QuantumState::new(vec![Complex64::new(1.0, 0.0); 13], true);
        let probs = state.probabilities();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-8);
        state.measure();
    }
}
