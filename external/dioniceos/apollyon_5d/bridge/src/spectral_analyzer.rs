// Spectral analyzer: Connect trajectory observation with QLogic spectral analysis

use crate::trajectory_observer::TrajectoryObserver;
use metatron::spectral::{SpectralOutput, SpectralPipeline};

/// Bridge component that analyzes 5D trajectories using Metatron's spectral cognition
///
/// This component extracts time-series data from trajectory observations and
/// performs spectral analysis using QLogic's Fourier-like transformation.
pub struct SpectralAnalyzer {
    pipeline: SpectralPipeline,
    /// Which component(s) to analyze (0-4 for 5D state)
    component_indices: Vec<usize>,
}

impl SpectralAnalyzer {
    /// Create a new spectral analyzer
    pub fn new() -> Self {
        Self {
            pipeline: SpectralPipeline::new(),
            component_indices: vec![0, 1, 2, 3, 4], // All components by default
        }
    }

    /// Create analyzer for specific components only
    pub fn for_components(indices: Vec<usize>) -> Self {
        Self {
            pipeline: SpectralPipeline::new(),
            component_indices: indices,
        }
    }

    /// Analyze trajectory history for a specific component
    pub fn analyze_component(
        &self,
        observer: &TrajectoryObserver,
        component: usize,
    ) -> Option<SpectralOutput> {
        if component >= 5 {
            return None;
        }

        let history = observer.history();
        if history.is_empty() {
            return None;
        }

        // Extract time series for this component
        let field: Vec<f64> = history.iter().map(|state| state.get(component)).collect();

        // Perform spectral analysis
        Some(self.pipeline.analyze(&field, None))
    }

    /// Analyze all configured components and return their spectral outputs
    pub fn analyze_all(&self, observer: &TrajectoryObserver) -> Vec<(usize, SpectralOutput)> {
        let mut results = Vec::new();

        for &component in &self.component_indices {
            if let Some(output) = self.analyze_component(observer, component) {
                results.push((component, output));
            }
        }

        results
    }

    /// Get dominant frequency for a component (peak in spectrum)
    pub fn dominant_frequency(
        &self,
        observer: &TrajectoryObserver,
        component: usize,
    ) -> Option<f64> {
        let output = self.analyze_component(observer, component)?;

        // Find peak in spectrum
        let (max_idx, _) = output
            .spectrum
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;

        Some(max_idx as f64 / output.spectrum.len() as f64)
    }

    /// Compute average entropy across all components
    pub fn average_entropy(&self, observer: &TrajectoryObserver) -> f64 {
        let results = self.analyze_all(observer);
        if results.is_empty() {
            return 0.0;
        }

        let sum: f64 = results.iter().map(|(_, output)| output.entropy).sum();
        sum / results.len() as f64
    }

    /// Detect if system exhibits oscillatory behavior
    pub fn is_oscillatory(&self, observer: &TrajectoryObserver, threshold: f64) -> bool {
        let results = self.analyze_all(observer);

        // Check if any component has high spectral concentration (low sparsity)
        results
            .iter()
            .any(|(_, output)| output.sparsity < threshold)
    }

    /// Get spectral centroids for all components
    pub fn spectral_centroids(&self, observer: &TrajectoryObserver) -> Vec<(usize, f64)> {
        self.analyze_all(observer)
            .into_iter()
            .map(|(idx, output)| (idx, output.spectral_centroid))
            .collect()
    }
}

impl Default for SpectralAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_5d::State5D;

    #[test]
    fn spectral_analyzer_creation() {
        let analyzer = SpectralAnalyzer::new();
        assert_eq!(analyzer.component_indices.len(), 5);
    }

    #[test]
    fn spectral_analyzer_for_specific_components() {
        let analyzer = SpectralAnalyzer::for_components(vec![0, 1]);
        assert_eq!(analyzer.component_indices.len(), 2);
    }

    #[test]
    fn analyze_component_with_data() {
        let analyzer = SpectralAnalyzer::new();
        let mut observer = TrajectoryObserver::new(100);

        // Add some oscillatory data
        for i in 0..50 {
            let t = i as f64 * 0.1;
            let state = State5D::new(t.sin(), (2.0 * t).sin(), (3.0 * t).sin(), 0.0, 0.0);
            observer.observe(state);
        }

        let output = analyzer.analyze_component(&observer, 0);
        assert!(output.is_some());

        let output = output.unwrap();
        assert_eq!(output.field.len(), 50);
        assert_eq!(output.spectrum.len(), 50);
        assert!(output.entropy > 0.0);
    }

    #[test]
    fn analyze_all_components() {
        let analyzer = SpectralAnalyzer::for_components(vec![0, 1, 2]);
        let mut observer = TrajectoryObserver::new(100);

        for i in 0..20 {
            let t = i as f64 * 0.1;
            observer.observe(State5D::new(t, t * 2.0, t * 3.0, 0.0, 0.0));
        }

        let results = analyzer.analyze_all(&observer);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn average_entropy_calculation() {
        let analyzer = SpectralAnalyzer::new();
        let mut observer = TrajectoryObserver::new(100);

        // Add some varying data
        for i in 0..30 {
            let t = i as f64 * 0.1;
            observer.observe(State5D::new(t.sin(), t.cos(), (t * 0.5).sin(), 0.5, 0.3));
        }

        let entropy = analyzer.average_entropy(&observer);
        assert!(entropy > 0.0);
        assert!(entropy < 10.0); // Reasonable entropy range
    }

    #[test]
    fn spectral_centroids_computation() {
        let analyzer = SpectralAnalyzer::for_components(vec![0, 1]);
        let mut observer = TrajectoryObserver::new(100);

        for i in 0..25 {
            let t = i as f64 * 0.1;
            observer.observe(State5D::new(t.sin(), t.cos(), 0.0, 0.0, 0.0));
        }

        let centroids = analyzer.spectral_centroids(&observer);
        assert_eq!(centroids.len(), 2);
        assert!(centroids[0].1 >= 0.0);
        assert!(centroids[1].1 >= 0.0);
    }
}
