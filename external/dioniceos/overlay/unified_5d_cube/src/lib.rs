//! # Unified 5D Cube Overlay
//!
//! Non-invasive overlay integrating APOLLYON-5D, 4D-Trichter, and MEF-Core
//! into a cohesive 5D Cube execution pipeline.
//!
//! This overlay uses ONLY public APIs from existing components and introduces
//! no changes to the base systems.

pub mod interlock;
pub mod metrics;
pub mod shadow;
pub mod tick;

pub use interlock::{
    CommitData, ExtendedCommitData, InterlockAdapter, InterlockConfig, SimpleProofOfResonance,
};
pub use metrics::{MetricsCollector, MetricsFormat, TickMetrics};
pub use shadow::{ActivationCriteria, ShadowController, ShadowMode};
pub use tick::{tick_5d_cube, TickResult};

/// Version of the 5D Cube overlay
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration for deterministic operation
pub const DEFAULT_SEED: u64 = 42;

/// Activation feature flag check
pub fn is_activated() -> bool {
    cfg!(feature = "activate")
}
