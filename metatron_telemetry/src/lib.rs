//! Metatron Telemetry - HTTP API and dashboard for Q⊗DASH
//!
//! This crate provides a lightweight telemetry and control interface for the
//! Q⊗DASH quantum-hybrid calibration system.

pub mod api;
pub mod config;
pub mod state;

pub use config::Config;
pub use state::AppState;
