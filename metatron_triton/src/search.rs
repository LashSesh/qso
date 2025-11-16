//! TRITON Search Engine
//!
//! Integrates the evolutionary spiral with spectral signature evaluation
//! to perform adaptive parameter space exploration.

use crate::{SpectralSignature, TritonSpiral};
use serde::{Deserialize, Serialize};

/// Result of a single TRITON search step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TritonStepResult {
    /// Parameter point evaluated
    pub point: Vec<f64>,

    /// Spectral signature of this point
    pub signature: SpectralSignature,

    /// Best resonance found so far
    pub best_resonance: f64,

    /// Current step index
    pub step_index: usize,

    /// Current spiral radius
    pub radius: f64,

    /// Improvement over previous step (resonance delta)
    pub improvement: f64,
}

/// TRITON search engine
///
/// Combines the evolutionary spiral with a user-provided evaluation function
/// to perform adaptive optimization in the SOLVE phase.
///
/// # Type Parameters
/// * `Eval` - Evaluation function: `fn(&[f64]) -> SpectralSignature`
///
/// # Example
/// ```
/// use metatron_triton::{SpectralSignature, TritonSearch};
///
/// let evaluator = |params: &[f64]| {
///     // Simple quadratic function with optimum at [0.5, 0.5, 0.5]
///     let psi = 1.0 - 4.0 * (params[0] - 0.5).powi(2);
///     let rho = 1.0 - 4.0 * (params[1] - 0.5).powi(2);
///     let omega = 1.0 - 4.0 * (params[2] - 0.5).powi(2);
///     SpectralSignature::new(psi.max(0.0), rho.max(0.0), omega.max(0.0))
/// };
///
/// let mut search = TritonSearch::new(3, 42, 100, evaluator);
///
/// for _ in 0..50 {
///     let result = search.step();
///     if result.step_index % 10 == 0 {
///         println!("Step {}: resonance = {:.6}", result.step_index, result.signature.resonance());
///     }
/// }
///
/// let (best_point, best_sig) = search.best().unwrap();
/// println!("Best resonance: {:.6}", best_sig.resonance());
/// ```
pub struct TritonSearch<Eval>
where
    Eval: Fn(&[f64]) -> SpectralSignature,
{
    /// Evolutionary spiral generator
    spiral: TritonSpiral,

    /// Evaluation function
    evaluator: Eval,

    /// Best signature found so far
    best_signature: Option<SpectralSignature>,

    /// Best point found so far
    best_point: Option<Vec<f64>>,

    /// Previous step's resonance (for tracking improvement)
    prev_resonance: f64,

    /// Step counter
    step: usize,

    /// Maximum steps (for auto-termination)
    max_steps: usize,

    /// History of resonance values (last N steps)
    resonance_history: Vec<f64>,

    /// Maximum history size
    history_size: usize,
}

impl<Eval> TritonSearch<Eval>
where
    Eval: Fn(&[f64]) -> SpectralSignature,
{
    /// Create a new TRITON search
    ///
    /// # Arguments
    /// * `dimension` - Dimensionality of the search space
    /// * `seed` - Random seed for reproducibility
    /// * `max_steps` - Maximum number of search steps
    /// * `evaluator` - Function mapping parameters to spectral signature
    pub fn new(dimension: usize, seed: u64, max_steps: usize, evaluator: Eval) -> Self {
        Self {
            spiral: TritonSpiral::new(dimension, seed),
            evaluator,
            best_signature: None,
            best_point: None,
            prev_resonance: 0.0,
            step: 0,
            max_steps,
            resonance_history: Vec::new(),
            history_size: 100,
        }
    }

    /// Create a search with custom spiral parameters
    pub fn with_spiral_params(
        dimension: usize,
        seed: u64,
        max_steps: usize,
        evaluator: Eval,
        radius_base: f64,
        learning_rate: f64,
        momentum_decay: f64,
        noise_level: f64,
    ) -> Self {
        let spiral = TritonSpiral::with_params(
            dimension,
            seed,
            radius_base,
            learning_rate,
            momentum_decay,
            noise_level,
        );

        Self {
            spiral,
            evaluator,
            best_signature: None,
            best_point: None,
            prev_resonance: 0.0,
            step: 0,
            max_steps,
            resonance_history: Vec::new(),
            history_size: 100,
        }
    }

    /// Perform one SOLVE-phase step
    ///
    /// 1. Generate next point from spiral
    /// 2. Evaluate spectral signature
    /// 3. Update best if improved
    /// 4. Compute gradient and update spiral momentum
    /// 5. Return step result
    pub fn step(&mut self) -> TritonStepResult {
        // Generate next candidate point
        let point = self.spiral.next_point();

        // Evaluate spectral signature
        let signature = (self.evaluator)(&point);
        let resonance = signature.resonance();

        // Track improvement
        let improvement = resonance - self.prev_resonance;
        self.prev_resonance = resonance;

        // Update best if this is better
        let mut best_resonance = resonance;
        if let Some(best_sig) = &self.best_signature {
            best_resonance = best_sig.resonance();

            if resonance > best_resonance {
                self.best_signature = Some(signature);
                self.best_point = Some(point.clone());
                best_resonance = resonance;

                // Move spiral center to new best point
                self.spiral.update_position(&point);

                tracing::debug!(
                    "TRITON: New best at step {} - resonance: {:.6}",
                    self.step + 1,
                    resonance
                );
            }
        } else {
            // First evaluation
            self.best_signature = Some(signature);
            self.best_point = Some(point.clone());
            self.spiral.update_position(&point);
        }

        // Compute gradient (simplified: direction to best point)
        let gradient = if let Some(best_point) = &self.best_point {
            point
                .iter()
                .zip(best_point.iter())
                .map(|(current, best)| best - current)
                .collect()
        } else {
            vec![0.0; point.len()]
        };

        // Compute reward: how much better is this than the previous best?
        let reward = if best_resonance > 1e-10 {
            (resonance / best_resonance).clamp(0.0, 1.0)
        } else {
            0.5
        };

        // Update spiral momentum
        self.spiral.update_momentum(&gradient, reward);

        // Update history
        self.resonance_history.push(resonance);
        if self.resonance_history.len() > self.history_size {
            self.resonance_history.remove(0);
        }

        self.step += 1;

        TritonStepResult {
            point,
            signature,
            best_resonance,
            step_index: self.step,
            radius: self.spiral.radius(),
            improvement,
        }
    }

    /// Run the search until max_steps or convergence
    ///
    /// # Arguments
    /// * `convergence_threshold` - Stop if improvement < threshold for N consecutive steps
    /// * `patience` - Number of non-improving steps before stopping
    ///
    /// # Returns
    /// Final best point and signature
    pub fn run(&mut self, convergence_threshold: f64, patience: usize) -> (Vec<f64>, SpectralSignature) {
        let mut no_improvement_count = 0;
        let mut prev_best_resonance = 0.0;

        while self.step < self.max_steps {
            let result = self.step();

            // Check convergence
            let resonance_improvement = result.best_resonance - prev_best_resonance;
            if resonance_improvement < convergence_threshold {
                no_improvement_count += 1;
            } else {
                no_improvement_count = 0;
            }

            prev_best_resonance = result.best_resonance;

            // Early stopping
            if no_improvement_count >= patience {
                tracing::info!(
                    "TRITON: Converged at step {} (no improvement for {} steps)",
                    self.step,
                    patience
                );
                break;
            }

            // Periodic logging
            if self.step % 10 == 0 {
                tracing::debug!(
                    "TRITON step {}/{}: resonance = {:.6}, radius = {:.4}",
                    self.step,
                    self.max_steps,
                    result.best_resonance,
                    result.radius
                );
            }
        }

        self.best()
            .expect("Search should have at least one evaluation")
    }

    /// Get the best point and signature found so far
    pub fn best(&self) -> Option<(Vec<f64>, SpectralSignature)> {
        if let (Some(point), Some(sig)) = (&self.best_point, &self.best_signature) {
            Some((point.clone(), *sig))
        } else {
            None
        }
    }

    /// Get the best point found so far
    pub fn best_point(&self) -> Option<&[f64]> {
        self.best_point.as_deref()
    }

    /// Get the best signature found so far
    pub fn best_signature(&self) -> Option<&SpectralSignature> {
        self.best_signature.as_ref()
    }

    /// Get current step count
    pub fn current_step(&self) -> usize {
        self.step
    }

    /// Check if search has reached max steps
    pub fn is_finished(&self) -> bool {
        self.step >= self.max_steps
    }

    /// Get resonance history
    pub fn resonance_history(&self) -> &[f64] {
        &self.resonance_history
    }

    /// Get average improvement rate over last N steps
    pub fn average_improvement_rate(&self, n: usize) -> f64 {
        let history = &self.resonance_history;
        if history.len() < n + 1 {
            return 0.0;
        }

        let start_idx = history.len() - n - 1;
        let start_val = history[start_idx];
        let end_val = history[history.len() - 1];

        (end_val - start_val) / n as f64
    }

    /// Reset the search to initial state
    pub fn reset(&mut self) {
        self.spiral.reset();
        self.best_signature = None;
        self.best_point = None;
        self.prev_resonance = 0.0;
        self.step = 0;
        self.resonance_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_creation() {
        let evaluator = |_params: &[f64]| SpectralSignature::new(0.8, 0.9, 0.7);
        let search = TritonSearch::new(3, 42, 100, evaluator);
        assert_eq!(search.current_step(), 0);
        assert!(!search.is_finished());
    }

    #[test]
    fn test_search_step() {
        let evaluator = |_params: &[f64]| SpectralSignature::new(0.8, 0.9, 0.7);
        let mut search = TritonSearch::new(3, 42, 100, evaluator);

        let result = search.step();
        assert_eq!(result.step_index, 1);
        assert_eq!(result.point.len(), 3);
        assert_eq!(search.current_step(), 1);
    }

    #[test]
    fn test_search_finds_optimum() {
        // Quadratic function with optimum at [0.5, 0.5, 0.5]
        let evaluator = |params: &[f64]| {
            let psi = 1.0 - 4.0 * (params[0] - 0.5).powi(2);
            let rho = 1.0 - 4.0 * (params[1] - 0.5).powi(2);
            let omega = 1.0 - 4.0 * (params[2] - 0.5).powi(2);
            SpectralSignature::new(psi.max(0.0), rho.max(0.0), omega.max(0.0))
        };

        let mut search = TritonSearch::new(3, 42, 200, evaluator);

        // Run search
        for _ in 0..200 {
            search.step();
        }

        let (best_point, best_sig) = search.best().unwrap();

        // Should find a point close to [0.5, 0.5, 0.5]
        for &val in &best_point {
            assert!((val - 0.5).abs() < 0.2, "Point {} far from optimum 0.5", val);
        }

        // Resonance should be high (close to 1.0)
        assert!(best_sig.resonance() > 0.5, "Resonance too low: {}", best_sig.resonance());
    }

    #[test]
    fn test_search_run_with_convergence() {
        let evaluator = |params: &[f64]| {
            let psi = 1.0 - (params[0] - 0.5).powi(2);
            let rho = 1.0 - (params[1] - 0.5).powi(2);
            let omega = 1.0 - (params[2] - 0.5).powi(2);
            SpectralSignature::new(psi.max(0.0), rho.max(0.0), omega.max(0.0))
        };

        let mut search = TritonSearch::new(3, 42, 1000, evaluator);

        let (_, final_sig) = search.run(1e-6, 50);

        // Should converge before max steps
        assert!(search.current_step() < 1000);

        // Should find a reasonable solution
        assert!(final_sig.resonance() > 0.3);
    }

    #[test]
    fn test_resonance_history() {
        let evaluator = |_params: &[f64]| SpectralSignature::new(0.8, 0.9, 0.7);
        let mut search = TritonSearch::new(3, 42, 100, evaluator);

        for _ in 0..10 {
            search.step();
        }

        assert_eq!(search.resonance_history().len(), 10);
    }

    #[test]
    fn test_reset() {
        let evaluator = |_params: &[f64]| SpectralSignature::new(0.8, 0.9, 0.7);
        let mut search = TritonSearch::new(3, 42, 100, evaluator);

        for _ in 0..10 {
            search.step();
        }

        search.reset();

        assert_eq!(search.current_step(), 0);
        assert!(search.best().is_none());
        assert_eq!(search.resonance_history().len(), 0);
    }
}
