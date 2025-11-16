/*!
 * Spiral Memory Module
 *
 * Encodes information (strings, seeds) as 5D point clouds in spiral space,
 * performs semantic evaluation, mutation, and feedback (Ouroboros) processing,
 * and emits excalibration events upon convergence.
 */

use ndarray::Array1;

/// Spiral Memory system for encoding and evolving information in 5D spiral space
///
/// The SpiralMemory encodes strings as 5D vectors using Fourier-like embeddings,
/// then iteratively optimizes point configurations to maximize a resonance metric
/// (psi). When convergence is detected (proof of resonance), the system stores
/// the configuration in memory.
#[derive(Debug, Clone)]
pub struct SpiralMemory {
    /// Memory storage: list of (point_cloud, psi_value) tuples
    pub memory: Vec<(Vec<Array1<f64>>, f64)>,
    /// Adaptation rate for gradient-based mutation
    pub alpha: f64,
    /// History of psi values across iterations
    pub history: Vec<f64>,
}

impl SpiralMemory {
    /// Create a new SpiralMemory with specified adaptation rate
    ///
    /// # Arguments
    ///
    /// * `alpha` - Adaptation rate for gradient-based updates (default: 0.1)
    pub fn new(alpha: f64) -> Self {
        Self {
            memory: Vec::new(),
            alpha,
            history: Vec::new(),
        }
    }

    /// Embed a string sequence into a 5D vector
    ///
    /// Uses Fourier-like encoding: each dimension is a weighted sum of
    /// character codes multiplied by cosine functions with different frequencies.
    /// The result is normalized to unit length.
    ///
    /// # Arguments
    ///
    /// * `sequence` - Input string to embed
    ///
    /// # Returns
    ///
    /// A normalized 5D vector representation
    pub fn embed(&self, sequence: &str) -> Array1<f64> {
        let base: Vec<f64> = sequence.chars().map(|c| c as u32 as f64).collect();
        let mut v = Array1::zeros(5);
        let n = base.len();

        for i in 0..5 {
            let mut sum = 0.0;
            for (j, &base_val) in base.iter().enumerate().take(n) {
                let freq = 2.0 * std::f64::consts::PI * (i + 1) as f64 * j as f64 / (n + 5) as f64;
                sum += base_val * freq.cos();
            }
            v[i] = sum;
        }

        // Normalize
        let norm = v.dot(&v).sqrt();
        if norm > 1e-12 {
            v / norm
        } else {
            v
        }
    }

    /// Spiralize a list of strings into 5D point cloud
    ///
    /// # Arguments
    ///
    /// * `elements` - List of strings to embed
    ///
    /// # Returns
    ///
    /// Vector of 5D embedded points
    pub fn spiralize(&self, elements: &[String]) -> Vec<Array1<f64>> {
        elements.iter().map(|e| self.embed(e)).collect()
    }

    /// Compute psi resonance metric between two 5D points
    ///
    /// Combines three components:
    /// - Stability (stab): cosine similarity between vectors (50% weight)
    /// - Convergence (conv): inverse distance (30% weight)
    /// - Reactivity (react): absolute sine of sum of differences (20% weight)
    ///
    /// # Arguments
    ///
    /// * `vi` - First 5D point
    /// * `vj` - Second 5D point
    ///
    /// # Returns
    ///
    /// Psi resonance metric value
    pub fn psi(&self, vi: &Array1<f64>, vj: &Array1<f64>) -> f64 {
        let eps = 1e-12;

        // Stability: cosine similarity
        let dot = vi.dot(vj);
        let norm_i = vi.dot(vi).sqrt();
        let norm_j = vj.dot(vj).sqrt();
        let stab = dot / (norm_i * norm_j + eps);

        // Convergence: inverse distance
        let diff = vi - vj;
        let dist = diff.dot(&diff).sqrt();
        let conv = 1.0 / (1.0 + dist);

        // Reactivity: absolute sine of sum of differences
        let sum_diff: f64 = diff.iter().sum();
        let react = sum_diff.sin().abs();

        0.5 * stab + 0.3 * conv + 0.2 * react
    }

    /// Compute total psi resonance for a sequence of points
    ///
    /// Sums pairwise psi values between consecutive points.
    ///
    /// # Arguments
    ///
    /// * `points` - List of 5D points
    ///
    /// # Returns
    ///
    /// Total psi resonance value
    pub fn psi_total(&self, points: &[Array1<f64>]) -> f64 {
        if points.len() < 2 {
            return 0.0;
        }

        let mut total = 0.0;
        for i in 0..points.len() - 1 {
            total += self.psi(&points[i], &points[i + 1]);
        }
        total
    }

    /// Compute gradient field for point cloud optimization
    ///
    /// For each pair of consecutive points, computes normalized direction vector.
    /// Last point gets negative of previous gradient (boundary condition).
    ///
    /// # Arguments
    ///
    /// * `points` - List of 5D points
    ///
    /// # Returns
    ///
    /// Vector of gradient vectors
    pub fn gradient(&self, points: &[Array1<f64>]) -> Vec<Array1<f64>> {
        let eps = 1e-12;
        let mut grads = Vec::new();

        if points.len() < 2 {
            return grads;
        }

        for i in 0..points.len() - 1 {
            let diff = &points[i + 1] - &points[i];
            let norm = diff.dot(&diff).sqrt();
            let grad = if norm > eps {
                diff / norm
            } else {
                diff.clone()
            };
            grads.push(grad);
        }

        // Last gradient is negative of previous
        if let Some(last_grad) = grads.last() {
            grads.push(-last_grad);
        }

        grads
    }

    /// Mutate point cloud along gradient direction
    ///
    /// # Arguments
    ///
    /// * `points` - Current point cloud
    /// * `grads` - Gradient vectors
    ///
    /// # Returns
    ///
    /// Updated point cloud
    pub fn mutate(&self, points: &[Array1<f64>], grads: &[Array1<f64>]) -> Vec<Array1<f64>> {
        points
            .iter()
            .zip(grads.iter())
            .map(|(p, g)| p + self.alpha * g)
            .collect()
    }

    /// Check if proof of resonance (convergence) is achieved
    ///
    /// # Arguments
    ///
    /// * `psi_old` - Previous psi value
    /// * `psi_new` - New psi value
    /// * `epsilon` - Convergence threshold (default: 1e-4)
    ///
    /// # Returns
    ///
    /// True if converged, false otherwise
    pub fn proof_of_resonance(&self, psi_old: f64, psi_new: f64, epsilon: f64) -> bool {
        (psi_new - psi_old).abs() < epsilon
    }

    /// Execute one optimization step with convergence detection
    ///
    /// Embeds input strings, then iteratively mutates point cloud to maximize
    /// psi resonance. Stops when convergence is detected or max_iter reached.
    ///
    /// # Arguments
    ///
    /// * `elements` - List of strings to process
    /// * `max_iter` - Maximum number of iterations (default: 30)
    ///
    /// # Returns
    ///
    /// Tuple of (optimized_points, final_psi_value)
    pub fn step(&mut self, elements: &[String], max_iter: usize) -> (Vec<Array1<f64>>, f64) {
        let mut points = self.spiralize(elements);
        let mut psi_val = self.psi_total(&points);

        for _ in 0..max_iter {
            let grads = self.gradient(&points);
            let new_points = self.mutate(&points, &grads);
            let new_psi = self.psi_total(&new_points);

            self.history.push(new_psi);

            if self.proof_of_resonance(psi_val, new_psi, 1e-4) {
                self.memory.push((new_points.clone(), new_psi));
                return (new_points, new_psi);
            }

            points = new_points;
            psi_val = new_psi;
        }

        self.memory.push((points.clone(), psi_val));
        (points, psi_val)
    }

    /// Clear all memory and history
    pub fn clear(&mut self) {
        self.memory.clear();
        self.history.clear();
    }

    /// Get the number of stored memories
    pub fn memory_size(&self) -> usize {
        self.memory.len()
    }

    /// Get the history length
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}

impl Default for SpiralMemory {
    /// Create a default SpiralMemory with alpha = 0.1
    fn default() -> Self {
        Self::new(0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_default() {
        let sm = SpiralMemory::default();
        assert_eq!(sm.alpha, 0.1);
        assert_eq!(sm.memory.len(), 0);
        assert_eq!(sm.history.len(), 0);
    }

    #[test]
    fn test_create_custom_alpha() {
        let sm = SpiralMemory::new(0.08);
        assert_eq!(sm.alpha, 0.08);
    }

    #[test]
    fn test_embed_produces_5d_vector() {
        let sm = SpiralMemory::default();
        let vec = sm.embed("TEST");
        assert_eq!(vec.len(), 5);
    }

    #[test]
    fn test_embed_normalized() {
        let sm = SpiralMemory::default();
        let vec = sm.embed("HELLO");
        let norm = vec.dot(&vec).sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_embed_deterministic() {
        let sm = SpiralMemory::default();
        let vec1 = sm.embed("AETHER");
        let vec2 = sm.embed("AETHER");

        for i in 0..5 {
            assert!((vec1[i] - vec2[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_spiralize() {
        let sm = SpiralMemory::default();
        let elements = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let points = sm.spiralize(&elements);
        assert_eq!(points.len(), 3);
        assert_eq!(points[0].len(), 5);
    }

    #[test]
    fn test_psi_identical_vectors() {
        let sm = SpiralMemory::default();
        let v = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        let psi_val = sm.psi(&v, &v);

        // stab = 1.0, conv = 1.0, react = 0.0
        // psi = 0.5 * 1.0 + 0.3 * 1.0 + 0.2 * 0.0 = 0.8
        assert!((psi_val - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_psi_total_empty() {
        let sm = SpiralMemory::default();
        let points = vec![];
        assert_eq!(sm.psi_total(&points), 0.0);
    }

    #[test]
    fn test_psi_total_single() {
        let sm = SpiralMemory::default();
        let points = vec![Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0])];
        assert_eq!(sm.psi_total(&points), 0.0);
    }

    #[test]
    fn test_gradient_computation() {
        let sm = SpiralMemory::default();
        let p1 = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        let p2 = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        let points = vec![p1, p2];

        let grads = sm.gradient(&points);
        assert_eq!(grads.len(), 2);

        // First gradient should point from p1 to p2
        assert!((grads[0][0] - 1.0).abs() < 1e-10);
        assert!((grads[0][1]).abs() < 1e-10);

        // Second gradient should be negative of first
        assert!((grads[1][0] + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mutate() {
        let sm = SpiralMemory::new(0.1);
        let p = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        let g = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        let points = vec![p];
        let grads = vec![g];

        let new_points = sm.mutate(&points, &grads);
        assert_eq!(new_points.len(), 1);
        assert!((new_points[0][0] - 1.1).abs() < 1e-10);
    }

    #[test]
    fn test_proof_of_resonance_converged() {
        let sm = SpiralMemory::default();
        assert!(sm.proof_of_resonance(1.0, 1.00005, 1e-4));
    }

    #[test]
    fn test_proof_of_resonance_not_converged() {
        let sm = SpiralMemory::default();
        assert!(!sm.proof_of_resonance(1.0, 1.001, 1e-4));
    }

    #[test]
    fn test_step_basic() {
        let mut sm = SpiralMemory::new(0.08);
        let elements = vec![
            "AETHER".to_string(),
            "SILICIUM".to_string(),
            "CYBER".to_string(),
        ];

        let (points, psi_val) = sm.step(&elements, 30);

        assert_eq!(points.len(), 3);
        assert!(psi_val > 0.0);
        assert!(sm.history.len() > 0);
        assert_eq!(sm.memory.len(), 1);
    }

    #[test]
    fn test_step_stores_in_memory() {
        let mut sm = SpiralMemory::new(0.05);
        let elements = vec!["A".to_string(), "B".to_string()];

        sm.step(&elements, 10);
        assert_eq!(sm.memory.len(), 1);

        sm.step(&elements, 10);
        assert_eq!(sm.memory.len(), 2);
    }

    #[test]
    fn test_clear() {
        let mut sm = SpiralMemory::default();
        let elements = vec!["TEST".to_string(), "DATA".to_string()];
        sm.step(&elements, 5);

        assert!(sm.memory.len() > 0);
        assert!(sm.history.len() > 0);

        sm.clear();
        assert_eq!(sm.memory.len(), 0);
        assert_eq!(sm.history.len(), 0);
    }

    #[test]
    fn test_memory_and_history_tracking() {
        let mut sm = SpiralMemory::new(0.1);
        let elements = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];

        sm.step(&elements, 20);

        assert_eq!(sm.memory_size(), 1);
        assert!(sm.history_len() > 0);
        assert!(sm.history_len() <= 20);
    }

    #[test]
    fn test_step_different_max_iter() {
        let mut sm1 = SpiralMemory::new(0.1);
        let mut sm2 = SpiralMemory::new(0.1);
        let elements = vec!["TEST".to_string()];

        sm1.step(&elements, 5);
        sm2.step(&elements, 10);

        // More iterations might lead to different history length
        assert!(sm1.history_len() <= 5);
        assert!(sm2.history_len() <= 10);
    }
}
