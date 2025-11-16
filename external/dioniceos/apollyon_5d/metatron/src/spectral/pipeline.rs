// Spectral cognition pipeline for analyzing patterns and frequencies
// This combines oscillator pattern generation with spectral analysis

use crate::spectral::entropy::EntropyAnalyzer;

/// Core spectral grammar analyzer that performs Fourier-like spectral analysis
#[derive(Debug, Default, Clone)]
pub struct SpectralGrammar;

impl SpectralGrammar {
    /// Analyze the field using a discrete Fourier-like transformation
    pub fn analyze(&self, field: &[f64]) -> Vec<f64> {
        let n = field.len();
        let mut spectrum = Vec::with_capacity(n);
        for k in 0..n {
            let mut real = 0.0;
            let mut imag = 0.0;
            for (t, &value) in field.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * k as f64 * t as f64 / n as f64;
                real += value * angle.cos();
                imag += value * angle.sin();
            }
            spectrum.push((real.powi(2) + imag.powi(2)).sqrt());
        }
        spectrum
    }
}

/// Diagnostics for resonance patterns
#[derive(Debug, Clone)]
pub struct ResonanceDiagnostics;

impl ResonanceDiagnostics {
    /// Compute the spectral centroid (center of mass of the spectrum)
    pub fn spectral_centroid(spectrum: &[f64]) -> f64 {
        let total: f64 = spectrum.iter().sum();
        if total == 0.0 {
            return 0.0;
        }
        spectrum
            .iter()
            .enumerate()
            .map(|(i, &val)| i as f64 * val)
            .sum::<f64>()
            / total
    }

    /// Compute sparsity (how concentrated the spectrum is)
    pub fn sparsity(spectrum: &[f64]) -> f64 {
        let n = spectrum.len() as f64;
        let l1_norm: f64 = spectrum.iter().map(|v| v.abs()).sum();
        let l2_norm: f64 = spectrum.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
        if l2_norm == 0.0 {
            return 0.0;
        }
        (n.sqrt() - l1_norm / l2_norm) / (n.sqrt() - 1.0)
    }
}

/// Complete spectral cognition pipeline output
#[derive(Debug, Clone)]
pub struct SpectralOutput {
    pub field: Vec<f64>,
    pub spectrum: Vec<f64>,
    pub entropy: f64,
    pub classification: Option<(String, f64)>,
    pub spectral_centroid: f64,
    pub sparsity: f64,
}

/// Spectral cognition pipeline combining pattern generation and analysis
#[derive(Debug, Clone)]
pub struct SpectralPipeline {
    grammar: SpectralGrammar,
    analyzer: EntropyAnalyzer,
}

impl SpectralPipeline {
    pub fn new() -> Self {
        Self {
            grammar: SpectralGrammar::default(),
            analyzer: EntropyAnalyzer::default(),
        }
    }

    /// Analyze a field pattern to produce spectral output
    pub fn analyze(&self, field: &[f64], classification: Option<(String, f64)>) -> SpectralOutput {
        let spectrum = self.grammar.analyze(field);
        let entropy = self.analyzer.entropy(field);
        let spectral_centroid = ResonanceDiagnostics::spectral_centroid(&spectrum);
        let sparsity = ResonanceDiagnostics::sparsity(&spectrum);

        SpectralOutput {
            field: field.to_vec(),
            spectrum,
            entropy,
            classification,
            spectral_centroid,
            sparsity,
        }
    }
}

impl Default for SpectralPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_grammar_analysis() {
        let grammar = SpectralGrammar::default();
        let field = vec![1.0, 0.0, -1.0, 0.0];
        let spectrum = grammar.analyze(&field);
        assert_eq!(spectrum.len(), 4);
        assert!(spectrum.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn spectral_centroid_calculation() {
        let spectrum = vec![0.0, 1.0, 2.0, 1.0, 0.0];
        let centroid = ResonanceDiagnostics::spectral_centroid(&spectrum);
        assert!((centroid - 2.0).abs() < 0.01);
    }

    #[test]
    fn pipeline_analysis() {
        let pipeline = SpectralPipeline::new();
        let field = vec![1.0; 13];
        let output = pipeline.analyze(&field, None);
        assert_eq!(output.field.len(), 13);
        assert_eq!(output.spectrum.len(), 13);
        assert!(output.entropy.is_finite());
    }
}
