//! Performance optimization components
//!
//! Re-exports optimization backends from the backends_opt module

#[cfg(feature = "stability-filter")]
pub use crate::backends_opt::stability_filter::{FilteredBackend, FilterStats, StabilityFilter, StabilityFilterConfig};

#[cfg(feature = "ophan-sharding")]
pub use crate::backends_opt::ophan_backend::OphanBackend;

#[cfg(feature = "adaptive-routing")]
pub use crate::backends_opt::adaptive_router::{AdaptiveRouter, RouterConfig, SearchStrategy};

#[cfg(feature = "mandorla")]
pub use crate::backends_opt::mandorla_refiner::{IndexCoverageStats, MandorlaBackend, MandorlaConfig, MandorlaRefiner};
