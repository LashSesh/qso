//! Adapters for converting between APOLLYON-5D and MEF-Core types
//!
//! This module provides bidirectional conversion between:
//! - APOLLYON State5D ⟷ MEF SpiralCoordinates
//! - APOLLYON SpectralFeatures ⟷ MEF SpectralSignature
//! - APOLLYON Metatron-R ⟷ MEF S7 Router
//! - APOLLYON ResonanceField ⟷ MEF Proof-of-Resonance

pub mod metatron_adapter;
pub mod resonance_adapter;
pub mod spectral_adapter;
pub mod state_adapter;

pub use metatron_adapter::MetatronBridge;
pub use resonance_adapter::{ProofOfResonanceData, ResonanceBridge};
pub use spectral_adapter::SpectralAdapter;
pub use state_adapter::StateAdapter;
