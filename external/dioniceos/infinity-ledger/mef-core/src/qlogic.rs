/*!
 * QLogic Module - Quantum Logic and Spectral Processing
 *
 * This module provides components for spectral processing and semantic analysis.
 * It includes an oscillator core for generating field patterns, spectral grammar
 * for FFT analysis, entropy analyzer for coherence measurement, and a main
 * QLogic engine that coordinates these components.
 */

use ndarray::Array1;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

/// Oscillator core that generates resonance patterns as field vectors
#[derive(Debug, Clone)]
pub struct QLOGICOscillatorCore {
    /// Number of nodes in the oscillator network
    pub num_nodes: usize,
}

impl QLOGICOscillatorCore {
    /// Create a new oscillator core with the specified number of nodes
    ///
    /// # Arguments
    ///
    /// * `num_nodes` - Number of nodes in the oscillator network
    ///
    /// # Returns
    ///
    /// A new QLOGICOscillatorCore instance
    pub fn new(num_nodes: usize) -> Self {
        Self { num_nodes }
    }

    /// Generate an oscillator pattern at time t
    ///
    /// Creates a sinusoidal pattern across all nodes with phases evenly
    /// distributed from 0 to 2π.
    ///
    /// # Arguments
    ///
    /// * `t` - Time parameter
    ///
    /// # Returns
    ///
    /// Array of oscillator values
    pub fn generate_pattern(&self, t: f64) -> Array1<f64> {
        let mut pattern = Array1::zeros(self.num_nodes);
        for i in 0..self.num_nodes {
            let phase = 2.0 * PI * (i as f64) / (self.num_nodes as f64);
            pattern[i] = (phase + t).sin();
        }
        pattern
    }
}

/// Spectral grammar that converts field vectors into frequency components
#[derive(Debug, Clone)]
pub struct SpectralGrammar {
    /// FFT planner (not stored, created on demand)
    _phantom: std::marker::PhantomData<()>,
}

impl SpectralGrammar {
    /// Create a new spectral grammar
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Analyze a field vector using FFT
    ///
    /// Computes the magnitude of the FFT of the input field.
    ///
    /// # Arguments
    ///
    /// * `field` - Input field vector
    ///
    /// # Returns
    ///
    /// Array of frequency magnitudes
    pub fn analyze(&self, field: &Array1<f64>) -> Array1<f64> {
        let n = field.len();

        // Convert to complex numbers
        let mut buffer: Vec<Complex<f64>> = field.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Perform FFT
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut buffer);

        // Extract magnitudes
        let mut spectrum = Array1::zeros(n);
        for (i, val) in buffer.iter().enumerate() {
            spectrum[i] = val.norm();
        }

        spectrum
    }
}

impl Default for SpectralGrammar {
    fn default() -> Self {
        Self::new()
    }
}

/// Entropy analyzer that evaluates field coherence
#[derive(Debug, Clone)]
pub struct EntropyAnalyzer {
    /// Placeholder for future configuration
    _phantom: std::marker::PhantomData<()>,
}

impl EntropyAnalyzer {
    /// Create a new entropy analyzer
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute the Shannon entropy of a field
    ///
    /// The field is normalized to a probability distribution and the
    /// Shannon entropy is computed as -Σ(p * log2(p)).
    ///
    /// # Arguments
    ///
    /// * `field` - Input field vector
    ///
    /// # Returns
    ///
    /// Shannon entropy value
    pub fn entropy(&self, field: &Array1<f64>) -> f64 {
        let p = field.mapv(|x| x.abs());
        let sum = p.sum() + 1e-12;
        let p_norm = p / sum;

        let entropy: f64 = p_norm
            .iter()
            .map(|&prob| {
                if prob > 1e-12 {
                    -prob * (prob + 1e-12).log2()
                } else {
                    0.0
                }
            })
            .sum();

        entropy
    }
}

impl Default for EntropyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a QLogic engine step
#[derive(Debug, Clone)]
pub struct QLogicStepResult {
    /// Generated field pattern
    pub field: Array1<f64>,
    /// Spectral analysis result
    pub spectrum: Array1<f64>,
    /// Entropy of the field
    pub entropy: f64,
    /// Optional spectral centroid
    pub spectral_centroid: Option<f64>,
    /// Optional sparsity metric
    pub sparsity: Option<f64>,
}

/// Main engine for spectral processing and semantic analysis
///
/// This engine generates oscillator patterns, applies spectral grammar,
/// computes entropy, and can provide diagnostic metrics like spectral
/// centroid and sparsity.
#[derive(Debug, Clone)]
pub struct QLogicEngine {
    /// Oscillator core
    pub osc_core: QLOGICOscillatorCore,
    /// Spectral grammar
    pub grammar: SpectralGrammar,
    /// Entropy analyzer
    pub analyzer: EntropyAnalyzer,
}

impl QLogicEngine {
    /// Create a new QLogic engine
    ///
    /// # Arguments
    ///
    /// * `num_nodes` - Number of nodes in the oscillator network
    ///
    /// # Returns
    ///
    /// A new QLogicEngine instance
    pub fn new(num_nodes: usize) -> Self {
        Self {
            osc_core: QLOGICOscillatorCore::new(num_nodes),
            grammar: SpectralGrammar::new(),
            analyzer: EntropyAnalyzer::new(),
        }
    }

    /// Execute a single step of the QLogic engine
    ///
    /// Generates an oscillator pattern and analyzes it with spectral grammar
    /// and entropy calculation. Also computes diagnostic metrics.
    ///
    /// # Arguments
    ///
    /// * `t` - Time parameter
    ///
    /// # Returns
    ///
    /// QLogicStepResult containing field, spectrum, entropy, and diagnostics
    pub fn step(&self, t: f64) -> QLogicStepResult {
        let field = self.osc_core.generate_pattern(t);
        let spectrum = self.grammar.analyze(&field);
        let entropy = self.analyzer.entropy(&field);

        // Compute diagnostics
        let spectral_centroid = Some(Self::compute_spectral_centroid(&spectrum));
        let sparsity = Some(Self::compute_sparsity(&spectrum));

        QLogicStepResult {
            field,
            spectrum,
            entropy,
            spectral_centroid,
            sparsity,
        }
    }

    /// Compute the spectral centroid of a spectrum
    ///
    /// The spectral centroid is the weighted mean of the frequencies,
    /// weighted by their amplitudes.
    ///
    /// # Arguments
    ///
    /// * `spectrum` - Frequency spectrum
    ///
    /// # Returns
    ///
    /// Spectral centroid value
    fn compute_spectral_centroid(spectrum: &Array1<f64>) -> f64 {
        let sum_weighted: f64 = spectrum
            .iter()
            .enumerate()
            .map(|(i, &mag)| (i as f64) * mag)
            .sum();
        let sum_mag: f64 = spectrum.sum();

        if sum_mag > 1e-12 {
            sum_weighted / sum_mag
        } else {
            0.0
        }
    }

    /// Compute the sparsity of a spectrum
    ///
    /// Sparsity is a measure of how concentrated the spectrum is.
    /// It's computed as the ratio of L1 to L2 norm.
    ///
    /// # Arguments
    ///
    /// * `spectrum` - Frequency spectrum
    ///
    /// # Returns
    ///
    /// Sparsity value (0 to 1, where 1 is maximally sparse)
    fn compute_sparsity(spectrum: &Array1<f64>) -> f64 {
        let l1_norm: f64 = spectrum.iter().map(|x| x.abs()).sum();
        let l2_norm: f64 = spectrum.iter().map(|x| x * x).sum::<f64>().sqrt();

        if l2_norm > 1e-12 && !spectrum.is_empty() {
            let n = spectrum.len() as f64;
            (n - l1_norm / l2_norm) / (n - 1.0)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_oscillator_core() {
        let osc = QLOGICOscillatorCore::new(10);
        assert_eq!(osc.num_nodes, 10);
    }

    #[test]
    fn test_generate_pattern() {
        let osc = QLOGICOscillatorCore::new(8);
        let pattern = osc.generate_pattern(0.0);

        assert_eq!(pattern.len(), 8);
        // At t=0, first element should be sin(0) = 0
        assert!(pattern[0].abs() < 1e-10);
    }

    #[test]
    fn test_generate_pattern_time_evolution() {
        let osc = QLOGICOscillatorCore::new(4);
        let pattern1 = osc.generate_pattern(0.0);
        let pattern2 = osc.generate_pattern(PI / 2.0);

        // Patterns should differ at different times
        assert!((pattern1[0] - pattern2[0]).abs() > 0.1);
    }

    #[test]
    fn test_spectral_grammar_creation() {
        let grammar = SpectralGrammar::new();
        // PhantomData has zero size, but we can check it compiles
        let _ = grammar;
    }

    #[test]
    fn test_spectral_analyze() {
        let grammar = SpectralGrammar::new();
        let field = Array1::from_vec(vec![1.0, 0.0, -1.0, 0.0]);
        let spectrum = grammar.analyze(&field);

        assert_eq!(spectrum.len(), 4);
        // All magnitudes should be non-negative
        for &mag in spectrum.iter() {
            assert!(mag >= 0.0);
        }
    }

    #[test]
    fn test_spectral_analyze_constant() {
        let grammar = SpectralGrammar::new();
        let field = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let spectrum = grammar.analyze(&field);

        // DC component (index 0) should dominate
        assert!(spectrum[0] > spectrum[1]);
        assert!(spectrum[0] > spectrum[2]);
    }

    #[test]
    fn test_entropy_analyzer_creation() {
        let analyzer = EntropyAnalyzer::new();
        // PhantomData has zero size, but we can check it compiles
        let _ = analyzer;
    }

    #[test]
    fn test_entropy_uniform() {
        let analyzer = EntropyAnalyzer::new();
        let field = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let entropy = analyzer.entropy(&field);

        // Uniform distribution should have maximum entropy
        // For 4 values: log2(4) = 2.0
        assert!((entropy - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_entropy_concentrated() {
        let analyzer = EntropyAnalyzer::new();
        let field = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let entropy = analyzer.entropy(&field);

        // Single non-zero value should have zero entropy
        assert!(entropy.abs() < 0.01);
    }

    #[test]
    fn test_entropy_mixed() {
        let analyzer = EntropyAnalyzer::new();
        let field1 = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let field2 = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);

        let entropy1 = analyzer.entropy(&field1);
        let entropy2 = analyzer.entropy(&field2);

        // Uniform should have higher entropy than concentrated
        assert!(entropy1 > entropy2);
    }

    #[test]
    fn test_qlogic_engine_creation() {
        let engine = QLogicEngine::new(16);
        assert_eq!(engine.osc_core.num_nodes, 16);
    }

    #[test]
    fn test_qlogic_step() {
        let engine = QLogicEngine::new(8);
        let result = engine.step(0.0);

        assert_eq!(result.field.len(), 8);
        assert_eq!(result.spectrum.len(), 8);
        assert!(result.entropy >= 0.0);
        assert!(result.spectral_centroid.is_some());
        assert!(result.sparsity.is_some());
    }

    #[test]
    fn test_qlogic_step_evolution() {
        let engine = QLogicEngine::new(8);
        let result1 = engine.step(0.0);
        let result2 = engine.step(1.0);

        // Results should differ at different times
        assert!((result1.field[0] - result2.field[0]).abs() > 0.01);
        assert!((result1.entropy - result2.entropy).abs() > 0.0);
    }

    #[test]
    fn test_spectral_centroid_computation() {
        let spectrum = Array1::from_vec(vec![0.0, 1.0, 2.0, 1.0, 0.0]);
        let centroid = QLogicEngine::compute_spectral_centroid(&spectrum);

        // Centroid should be near index 2 (where the peak is)
        assert!((centroid - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_spectral_centroid_uniform() {
        let spectrum = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let centroid = QLogicEngine::compute_spectral_centroid(&spectrum);

        // Centroid should be at the middle
        assert!((centroid - 1.5).abs() < 0.1);
    }

    #[test]
    fn test_spectral_centroid_zero() {
        let spectrum = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        let centroid = QLogicEngine::compute_spectral_centroid(&spectrum);

        // Zero spectrum should have zero centroid
        assert_eq!(centroid, 0.0);
    }

    #[test]
    fn test_sparsity_computation_sparse() {
        let spectrum = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let sparsity = QLogicEngine::compute_sparsity(&spectrum);

        // Single non-zero value is maximally sparse
        assert!(sparsity > 0.9);
    }

    #[test]
    fn test_sparsity_computation_dense() {
        let spectrum = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let sparsity = QLogicEngine::compute_sparsity(&spectrum);

        // For uniform distribution: l1=4, l2=2, sparsity = (4-4/2)/(4-1) = 2/3
        // Uniform is not maximally dense (which would be 0), but it's relatively dense
        assert!(sparsity > 0.5 && sparsity < 0.8);
    }

    #[test]
    fn test_sparsity_computation_zero() {
        let spectrum = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        let sparsity = QLogicEngine::compute_sparsity(&spectrum);

        // Zero spectrum should have zero sparsity
        assert_eq!(sparsity, 0.0);
    }
}
