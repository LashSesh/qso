//! Unified cognitive engine combining APOLLYON and MEF

pub mod async_engine;
pub mod cognitive_engine;
pub mod types;

pub use async_engine::AsyncUnifiedCognitiveEngine;
pub use cognitive_engine::UnifiedCognitiveEngine;
pub use types::{BatchResult, CognitiveInput, CognitiveOutput, GateConfig};
