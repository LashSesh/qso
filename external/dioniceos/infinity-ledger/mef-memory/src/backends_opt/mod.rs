//! Performance optimization backends
//!
//! This module contains wrapper backends that implement the MEF optimization
//! components described in mef_integration_spec.md

#[cfg(feature = "stability-filter")]
pub mod stability_filter;

#[cfg(feature = "ophan-sharding")]
pub mod ophan_backend;

#[cfg(feature = "adaptive-routing")]
pub mod adaptive_router;

#[cfg(feature = "mandorla")]
pub mod mandorla_refiner;

// Re-exports
#[cfg(feature = "stability-filter")]
pub use stability_filter::{FilteredBackend, FilterStats, StabilityFilter, StabilityFilterConfig};

#[cfg(feature = "ophan-sharding")]
pub use ophan_backend::OphanBackend;

#[cfg(feature = "adaptive-routing")]
pub use adaptive_router::{AdaptiveRouter, RouterConfig, SearchStrategy};

#[cfg(feature = "mandorla")]
pub use mandorla_refiner::{IndexCoverageStats, MandorlaBackend, MandorlaConfig, MandorlaRefiner};
