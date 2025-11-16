// Metatron-R: Post-symbolic cognition engine with geometric reasoning
//
// This crate provides a unified framework for geometric-cognitive mathematics,
// combining sacred geometry (Metatron Cube) with adaptive cognition and
// resonance-based orchestration.

pub mod api;
pub mod cognition;
pub mod config;
pub mod engine;
pub mod error;
pub mod fields;
pub mod geometry;
pub mod history;
pub mod spectral;
pub mod visualization;

// Re-export main types
pub use cognition::{MasterAgent, MetatronAgent, QDASHAgent};
pub use config::{AgentConfig, EngineConfig, MasterAgentConfig};
pub use engine::{EngineSnapshot, MetatronEngine};
pub use error::{EngineError, EngineResult};
pub use fields::{MandorlaField, ResonanceTensorField, TensorNetwork};
pub use geometry::{MetatronCube, MetatronCubeGraph};
pub use spectral::{EntropyAnalyzer, SpectralPipeline};
