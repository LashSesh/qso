//! MEF Schemas - Type system and JSON schema definitions
//!
//! This module provides the core type definitions for the MEF Knowledge Engine extension:
//! - RouteSpec: S7 route specification with 7-slot permutation
//! - MemoryItem: 8D normalized vector with spectral signature
//! - KnowledgeObject: TIC binding with route and seed path
//! - MerkabaGateEvent: Gate decision events (FIRE/HOLD)

pub mod gate;
pub mod gate_event;
pub mod knowledge;
pub mod knowledge_object;
pub mod memory_item;
pub mod route_spec;

pub use gate::{
    GateChecks, GateDecision as GateDecisionV2, MerkabaGateEvent as MerkabaGateEventV2,
};
pub use gate_event::{GateDecision, MerkabaGateEvent};
pub use knowledge::{
    KnowledgeContext, KnowledgeObject as KnowledgeObjectV2, RouteReference, TicReference,
};
pub use knowledge_object::KnowledgeObject;
pub use memory_item::{MemoryItem, PorStatus, SpectralSignature};
pub use route_spec::{OperatorSlot, RouteSpec};

#[derive(Debug, thiserror::Error)]
pub enum SchemaError {
    #[error("Invalid route specification: {0}")]
    InvalidRoute(String),

    #[error("Invalid vector dimension: expected {expected}, got {got}")]
    InvalidDimension { expected: usize, got: usize },

    #[error("Invalid spectral signature: {0}")]
    InvalidSpectral(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, SchemaError>;
