//! # Unified 5D Cube Overlay
//!
//! Non-invasive overlay integrating APOLLYON-5D, 4D-Trichter, and MEF-Core
//! into a cohesive 5D Cube execution pipeline.
//!
//! This overlay uses ONLY public APIs from existing components and introduces
//! no changes to the base systems.

pub mod interlock;
pub mod tick;
pub mod metrics;
pub mod shadow;

pub use interlock::{InterlockAdapter, InterlockConfig, ExtendedCommitData, CommitData, SimpleProofOfResonance};
pub use tick::{tick_5d_cube, TickResult};
pub use metrics::{MetricsCollector, MetricsFormat, TickMetrics};
pub use shadow::{ShadowMode, ActivationCriteria, ShadowController};

/// Version of the 5D Cube overlay
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration for deterministic operation
pub const DEFAULT_SEED: u64 = 42;

/// Activation feature flag check
pub fn is_activated() -> bool {
    cfg!(feature = "activate")
}
