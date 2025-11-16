// Parameter tuner: Adaptive parameter control using QDASH

use crate::spectral_analyzer::SpectralAnalyzer;
use crate::trajectory_observer::TrajectoryObserver;
use core_5d::SystemParameters;
use metatron::cognition::qdash::QDASHAgent;

/// Adaptive parameter tuner using QDASH decision-making
///
/// This component uses QDASH's cognitive decision engine to adaptively
/// tune system parameters based on trajectory analysis and spectral features.
pub struct ParameterTuner {
    qdash: QDASHAgent,
    analyzer: SpectralAnalyzer,
    learning_rate: f64,
}

impl ParameterTuner {
    /// Create a new parameter tuner
    ///
    /// # Arguments
    /// * `n_cells` - Number of Gabriel cells in QDASH
    /// * `alpha` - Mandorla field alpha parameter
    /// * `beta` - Mandorla field beta parameter
    pub fn new(n_cells: usize, alpha: f64, beta: f64) -> Self {
        Self {
            qdash: QDASHAgent::new(n_cells, alpha, beta),
            analyzer: SpectralAnalyzer::new(),
            learning_rate: 0.1,
        }
    }

    /// Create with default QDASH parameters
    pub fn default_config() -> Self {
        Self::new(5, 0.5, 0.5)
    }

    /// Set learning rate for parameter updates
    pub fn with_learning_rate(mut self, rate: f64) -> Self {
        self.learning_rate = rate;
        self
    }

    /// Analyze trajectory and suggest parameter adjustments
    ///
    /// Returns suggested modifications to intrinsic rates based on
    /// spectral analysis and QDASH decision making.
    pub fn suggest_adjustments(
        &mut self,
        observer: &TrajectoryObserver,
        _current_params: &SystemParameters,
    ) -> [f64; 5] {
        // Extract spectral features from trajectory
        let entropy = self.analyzer.average_entropy(observer);
        let centroids = self.analyzer.spectral_centroids(observer);

        // Normalize features for QDASH input
        let mut input_vector = vec![entropy / 10.0]; // Normalize entropy
        for (_, centroid) in centroids.iter().take(4) {
            input_vector.push(centroid / 10.0);
        }

        // Run QDASH decision cycle
        let outcome = self.qdash.decision_cycle(&input_vector, 10, 0.01);

        // Convert QDASH output to parameter adjustments
        let mut adjustments = [0.0; 5];

        // Use oscillator signal to modulate adjustments
        for i in 0..5.min(outcome.oscillator_signal.len()) {
            let signal = outcome.oscillator_signal[i];

            // Adjust intrinsic rates based on signal and resonance
            let adjustment = self.learning_rate * signal * outcome.resonance;
            adjustments[i] = adjustment;
        }

        adjustments
    }

    /// Apply suggested adjustments to parameters
    pub fn apply_adjustments(
        &mut self,
        observer: &TrajectoryObserver,
        params: &mut SystemParameters,
    ) -> bool {
        let adjustments = self.suggest_adjustments(observer, params);

        // Check if any significant adjustment is suggested
        let has_adjustment = adjustments.iter().any(|&a| a.abs() > 0.001);

        if has_adjustment {
            for i in 0..5 {
                params.intrinsic_rates[i] += adjustments[i];
            }
        }

        has_adjustment
    }

    /// Get QDASH resonance value
    pub fn resonance(&self) -> f64 {
        self.qdash.mandorla.resonance
    }

    /// Get QDASH decision threshold
    pub fn threshold(&self) -> f64 {
        self.qdash.mandorla.current_theta
    }

    /// Reset internal state
    pub fn reset(&mut self) {
        self.qdash.time = 0.0;
        self.qdash.last_decision = None;
        self.qdash.mandorla.clear_inputs();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_5d::State5D;

    #[test]
    fn parameter_tuner_creation() {
        let tuner = ParameterTuner::new(5, 0.5, 0.5);
        assert_eq!(tuner.qdash.cells.len(), 5);
        assert_eq!(tuner.learning_rate, 0.1);
    }

    #[test]
    fn parameter_tuner_default_config() {
        let tuner = ParameterTuner::default_config();
        assert_eq!(tuner.learning_rate, 0.1);
    }

    #[test]
    fn parameter_tuner_with_learning_rate() {
        let tuner = ParameterTuner::default_config().with_learning_rate(0.05);
        assert_eq!(tuner.learning_rate, 0.05);
    }

    #[test]
    fn suggest_adjustments() {
        let mut tuner = ParameterTuner::default_config();
        let mut observer = TrajectoryObserver::new(100);

        // Add some trajectory data
        for i in 0..30 {
            let t = i as f64 * 0.1;
            observer.observe(State5D::new(t.sin(), t.cos(), (t * 0.5).sin(), 0.5, 0.3));
        }

        let params = SystemParameters::zero();
        let adjustments = tuner.suggest_adjustments(&observer, &params);

        // Should return 5 adjustment values
        assert_eq!(adjustments.len(), 5);

        // Adjustments should be finite
        for &adj in &adjustments {
            assert!(adj.is_finite());
        }
    }

    #[test]
    fn apply_adjustments() {
        let mut tuner = ParameterTuner::default_config().with_learning_rate(0.01);
        let mut observer = TrajectoryObserver::new(100);

        // Add trajectory data
        for i in 0..30 {
            let t = i as f64 * 0.1;
            observer.observe(State5D::new(t, t * 2.0, t * 3.0, 0.0, 0.0));
        }

        let mut params = SystemParameters::zero();
        let _original_rates = params.intrinsic_rates;

        tuner.apply_adjustments(&observer, &mut params);

        // Parameters should be modified (or at least attempted)
        // Note: might not change if adjustments are too small
        assert!(params.intrinsic_rates.iter().all(|&r| r.is_finite()));
    }

    #[test]
    fn resonance_and_threshold_access() {
        let tuner = ParameterTuner::default_config();

        let resonance = tuner.resonance();
        let threshold = tuner.threshold();

        assert!(resonance.is_finite());
        assert!(threshold.is_finite());
    }

    #[test]
    fn parameter_tuner_reset() {
        let mut tuner = ParameterTuner::default_config();
        tuner.qdash.time = 10.0;

        tuner.reset();
        assert_eq!(tuner.qdash.time, 0.0);
    }
}
