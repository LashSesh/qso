//! MEF Knowledge - Knowledge processing and derivation
//!
//! This module provides:
//! - Canonical JSON serialization (deterministic, stable key order)
//! - Content-addressed knowledge IDs via SHA256 hashing
//! - HD-style seed derivation using HMAC-SHA256
//! - 8D vector construction from 5D spiral + 3D spectral features
//! - Knowledge inference and projection engine (scaffold)

pub mod canonical;
pub mod config;
pub mod content_address;
pub mod derivation;
pub mod inference;
pub mod metric;
pub mod pipeline;
pub mod primitives;
pub mod seed_derivation;
pub mod vector8;

pub use canonical::canonical_json;
pub use config::{ExtensionConfig, ExtensionSettings, KnowledgeConfig, MemoryConfig, RouterConfig};
pub use content_address::compute_mef_id;
pub use derivation::{DeriveRequest, DeriveResponse, KnowledgeDerivation};
pub use metric::{Vector8Builder as Vector8BuilderV2, Vector8Weights};
pub use pipeline::ExtensionPipeline;
pub use primitives::{
    canonical_json as canonical_json_v2, compute_content_hash, compute_mef_id as compute_mef_id_v2,
    derive_seed as derive_seed_v2,
};
pub use seed_derivation::derive_seed;
pub use vector8::{Vector8Builder, Vector8Config};

#[derive(Debug, thiserror::Error)]
pub enum KnowledgeError {
    #[error("Canonical JSON error: {0}")]
    Canonical(String),

    #[error("Content addressing error: {0}")]
    ContentAddress(String),

    #[error("Seed derivation error: {0}")]
    SeedDerivation(String),

    #[error("Vector construction error: {0}")]
    VectorConstruction(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, KnowledgeError>;
