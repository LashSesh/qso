//! MEF-Core Spiral Module
//!
//! 5D spiral snapshot and storage implementation with deterministic addressing.
//!
//! Migrated from: MEF-Core_v1.0/src/spiral/

pub mod proof_of_resonance;
pub mod snapshot;
pub mod storage;

pub use proof_of_resonance::{
    BandEnergy, BatchValidationResults, ProofOfResonance, ResonanceData, StabilityData,
    ValidationReport,
};
pub use snapshot::{Metrics, Payload, Sigma, Snapshot, SpiralConfig, SpiralSnapshot};
pub use storage::SpiralStorage;
