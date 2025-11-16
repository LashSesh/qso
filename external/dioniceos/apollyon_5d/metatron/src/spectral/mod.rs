// Spectral module - Spectral cognition and entropy analysis

pub mod entropy;
pub mod pipeline;

pub use entropy::EntropyAnalyzer;
pub use pipeline::{ResonanceDiagnostics, SpectralGrammar, SpectralOutput, SpectralPipeline};
