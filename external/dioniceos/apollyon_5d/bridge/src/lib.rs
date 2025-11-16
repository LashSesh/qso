// Bridge module: Integration layer between 5D framework and Metatron-R
//
// This module provides trait-based interfaces for connecting the numerical
// 5D dynamical systems framework with the geometric-cognitive Metatron-R engine.

pub mod adaptive_coupling;
pub mod geometric_forcing;
pub mod mandorla_field;
pub mod parameter_tuner;
pub mod resonance_field;
pub mod spectral_analyzer;
pub mod trajectory_observer;
pub mod unified_system;

pub use adaptive_coupling::AdaptiveCoupling;
pub use geometric_forcing::GeometricStateSpace;
pub use mandorla_field::MandorlaResonanceField;
pub use parameter_tuner::ParameterTuner;
pub use resonance_field::{ConstantResonanceField, OscillatoryResonanceField, ResonanceField};
pub use spectral_analyzer::SpectralAnalyzer;
pub use trajectory_observer::TrajectoryObserver;
pub use unified_system::CognitiveSimulator;
