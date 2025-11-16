//! TRITON Evolutionary Spiral
//!
//! Implements a golden-angle spiral evolution strategy with momentum-based exploration.
//! The spiral expands and contracts based on the quality of recent explorations,
//! creating an adaptive search pattern that balances exploration and exploitation.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Golden ratio φ = (1 + √5) / 2
const GOLDEN_RATIO: f64 = 1.618033988749895;

/// Golden angle in radians: 2π / φ²
const GOLDEN_ANGLE: f64 = 2.399963229728653;

/// TRITON evolutionary spiral
///
/// Generates points on a golden-angle spiral in n-dimensional space,
/// with momentum-based adaptation to guide exploration.
#[derive(Debug, Clone)]
pub struct TritonSpiral {
    /// Problem dimension
    dimension: usize,

    /// Current step counter
    step: usize,

    /// Random number generator
    rng: StdRng,

    /// Current position in parameter space
    position: Vec<f64>,

    /// Momentum vector (direction and magnitude of recent progress)
    momentum: Vec<f64>,

    /// Base radius for spiral expansion
    radius_base: f64,

    /// Current spiral radius (adaptive)
    radius: f64,

    /// Learning rate for momentum updates
    learning_rate: f64,

    /// Momentum decay factor (0.0 = no memory, 1.0 = full memory)
    momentum_decay: f64,

    /// Exploration noise level
    noise_level: f64,
}

impl TritonSpiral {
    /// Create a new TRITON spiral
    ///
    /// # Arguments
    /// * `dimension` - Dimensionality of the search space
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Example
    /// ```
    /// use metatron_triton::TritonSpiral;
    ///
    /// let mut spiral = TritonSpiral::new(3, 42);
    /// let point = spiral.next_point();
    /// assert_eq!(point.len(), 3);
    /// ```
    pub fn new(dimension: usize, seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);

        Self {
            dimension,
            step: 0,
            rng,
            position: vec![0.5; dimension],
            momentum: vec![0.0; dimension],
            radius_base: 0.3,
            radius: 0.3,
            learning_rate: 0.1,
            momentum_decay: 0.9,
            noise_level: 0.05,
        }
    }

    /// Create a spiral with custom parameters
    pub fn with_params(
        dimension: usize,
        seed: u64,
        radius_base: f64,
        learning_rate: f64,
        momentum_decay: f64,
        noise_level: f64,
    ) -> Self {
        let mut spiral = Self::new(dimension, seed);
        spiral.radius_base = radius_base;
        spiral.radius = radius_base;
        spiral.learning_rate = learning_rate;
        spiral.momentum_decay = momentum_decay;
        spiral.noise_level = noise_level;
        spiral
    }

    /// Generate the next point on the evolutionary spiral
    ///
    /// Uses golden-angle spiral coordinates combined with momentum and exploration noise.
    ///
    /// # Returns
    /// A point in [0, 1]^n parameter space
    pub fn next_point(&mut self) -> Vec<f64> {
        self.step += 1;

        // Compute golden spiral coordinates
        let theta = self.step as f64 * GOLDEN_ANGLE;
        let r = self.radius * (self.step as f64).sqrt() / (self.dimension as f64).sqrt();

        // Generate spiral offset in each dimension
        let mut point = Vec::with_capacity(self.dimension);

        for i in 0..self.dimension {
            // Golden spiral component
            let angle = theta + (i as f64 * std::f64::consts::PI * 2.0 / self.dimension as f64);
            let spiral_component = r * angle.cos();

            // Momentum component
            let momentum_component = self.momentum[i];

            // Exploration noise
            let noise = self.rng.gen_range(-self.noise_level..self.noise_level);

            // Combine components
            let value = self.position[i] + spiral_component + momentum_component + noise;

            // Clamp to [0, 1] and apply soft boundaries
            let clamped = value.clamp(0.0, 1.0);
            point.push(clamped);
        }

        point
    }

    /// Update momentum based on a gradient signal and reward
    ///
    /// # Arguments
    /// * `gradient` - Direction of improvement (positive = increase parameter)
    /// * `reward` - Magnitude of improvement (0.0 to 1.0)
    ///
    /// The gradient is typically computed as (new_signature - old_signature) / param_delta
    pub fn update_momentum(&mut self, gradient: &[f64], reward: f64) {
        assert_eq!(gradient.len(), self.dimension, "Gradient dimension mismatch");

        // Normalize gradient
        let grad_norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        let normalized_gradient: Vec<f64> = if grad_norm > 1e-10 {
            gradient.iter().map(|g| g / grad_norm).collect()
        } else {
            vec![0.0; self.dimension]
        };

        // Update momentum with decay and new gradient
        for i in 0..self.dimension {
            self.momentum[i] = self.momentum_decay * self.momentum[i]
                + self.learning_rate * reward * normalized_gradient[i];

            // Clamp momentum to prevent divergence
            self.momentum[i] = self.momentum[i].clamp(-0.5, 0.5);
        }

        // Adapt radius based on reward
        let radius_adjustment = reward * 0.1 - 0.05; // Expand on success, contract on failure
        self.radius = (self.radius + radius_adjustment).clamp(0.05, 1.0);
    }

    /// Update position (typically to the best point found so far)
    pub fn update_position(&mut self, new_position: &[f64]) {
        assert_eq!(new_position.len(), self.dimension, "Position dimension mismatch");
        self.position.copy_from_slice(new_position);
    }

    /// Reset the spiral to initial state (useful for restart strategies)
    pub fn reset(&mut self) {
        self.step = 0;
        self.position.fill(0.5);
        self.momentum.fill(0.0);
        self.radius = self.radius_base;
    }

    /// Get current step counter
    pub fn step(&self) -> usize {
        self.step
    }

    /// Get current position
    pub fn position(&self) -> &[f64] {
        &self.position
    }

    /// Get current momentum
    pub fn momentum(&self) -> &[f64] {
        &self.momentum
    }

    /// Get current radius
    pub fn radius(&self) -> f64 {
        self.radius
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spiral_creation() {
        let spiral = TritonSpiral::new(3, 42);
        assert_eq!(spiral.dimension, 3);
        assert_eq!(spiral.step, 0);
    }

    #[test]
    fn test_next_point_dimension() {
        let mut spiral = TritonSpiral::new(5, 42);
        let point = spiral.next_point();
        assert_eq!(point.len(), 5);
        assert_eq!(spiral.step(), 1);
    }

    #[test]
    fn test_next_point_bounds() {
        let mut spiral = TritonSpiral::new(3, 42);
        for _ in 0..100 {
            let point = spiral.next_point();
            for &val in &point {
                assert!(val >= 0.0 && val <= 1.0, "Point value {} out of bounds", val);
            }
        }
    }

    #[test]
    fn test_momentum_update() {
        let mut spiral = TritonSpiral::new(3, 42);

        // Simulate improvement in direction [1, 0, -1]
        let gradient = vec![1.0, 0.0, -1.0];
        spiral.update_momentum(&gradient, 0.8);

        // Momentum should be non-zero after update
        assert!(spiral.momentum().iter().any(|&m| m != 0.0));
    }

    #[test]
    fn test_position_update() {
        let mut spiral = TritonSpiral::new(3, 42);
        let new_pos = vec![0.3, 0.7, 0.5];
        spiral.update_position(&new_pos);
        assert_eq!(spiral.position(), &new_pos);
    }

    #[test]
    fn test_reset() {
        let mut spiral = TritonSpiral::new(3, 42);

        // Take some steps
        for _ in 0..10 {
            spiral.next_point();
        }
        spiral.update_momentum(&vec![1.0, 1.0, 1.0], 0.9);

        // Reset
        spiral.reset();

        assert_eq!(spiral.step(), 0);
        assert!(spiral.momentum().iter().all(|&m| m == 0.0));
    }

    #[test]
    fn test_golden_spiral_exploration() {
        let mut spiral = TritonSpiral::new(2, 42);
        let mut points = Vec::new();

        // Generate several points
        for _ in 0..50 {
            points.push(spiral.next_point());
        }

        // Check that we generated the correct number of points
        assert_eq!(points.len(), 50);

        // Check that all points are within bounds
        for point in &points {
            for &val in point {
                assert!(val >= 0.0 && val <= 1.0, "Point value {} out of bounds", val);
            }
        }

        // Check that at least some points are different (not all the same)
        let first = &points[0];
        let has_variation = points.iter().any(|p| {
            p.iter().zip(first.iter()).any(|(a, b)| (a - b).abs() > 1e-6)
        });
        assert!(has_variation, "All points are identical - spiral not working");
    }
}
