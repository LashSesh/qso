use crate::cognition::semantic_field::{ResonanceDiagnostics, SemanticField};

#[derive(Debug, Clone)]
pub struct QLOGICOscillatorCore {
    pub num_nodes: usize,
}

impl QLOGICOscillatorCore {
    pub fn new(num_nodes: usize) -> Self {
        Self { num_nodes }
    }

    pub fn generate_pattern(&self, t: f64) -> Vec<f64> {
        (0..self.num_nodes)
            .map(|i| {
                let phase = 2.0 * std::f64::consts::PI * i as f64 / self.num_nodes as f64;
                (phase + t).sin()
            })
            .collect()
    }
}

#[derive(Debug, Default, Clone)]
pub struct SpectralGrammar;

impl SpectralGrammar {
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

#[derive(Debug, Default, Clone)]
pub struct EntropyAnalyzer;

impl EntropyAnalyzer {
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
}

#[derive(Debug, Clone)]
pub struct QLogicOutput {
    pub field: Vec<f64>,
    pub spectrum: Vec<f64>,
    pub entropy: f64,
    pub classification: Option<(String, f64)>,
    pub diagnostics: Option<Diagnostics>,
}

#[derive(Debug, Clone)]
pub struct Diagnostics {
    pub spectral_centroid: f64,
    pub sparsity: f64,
}

#[derive(Debug, Clone)]
pub struct QLogicEngine {
    pub osc_core: QLOGICOscillatorCore,
    grammar: SpectralGrammar,
    analyzer: EntropyAnalyzer,
    pub semantic_field: Option<SemanticField>,
}

impl QLogicEngine {
    pub fn new(num_nodes: usize, semantic_field: Option<SemanticField>) -> Self {
        Self {
            osc_core: QLOGICOscillatorCore::new(num_nodes),
            grammar: SpectralGrammar::default(),
            analyzer: EntropyAnalyzer::default(),
            semantic_field,
        }
    }

    pub fn step(&mut self, t: f64) -> QLogicOutput {
        let field = self.osc_core.generate_pattern(t);
        let spectrum = self.grammar.analyze(&field);
        let entropy = self.analyzer.entropy(&field);
        let classification = self
            .semantic_field
            .as_mut()
            .and_then(|sf| sf.classify(&spectrum, 1).into_iter().next());
        let diagnostics = Some(Diagnostics {
            spectral_centroid: ResonanceDiagnostics::spectral_centroid(&spectrum),
            sparsity: ResonanceDiagnostics::sparsity(&spectrum),
        });
        QLogicOutput {
            field,
            spectrum,
            entropy,
            classification,
            diagnostics,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qlogic_step() {
        let mut engine = QLogicEngine::new(13, None);
        let res = engine.step(0.0);
        assert_eq!(res.field.len(), 13);
        assert!(res.entropy.is_finite());
    }
}
