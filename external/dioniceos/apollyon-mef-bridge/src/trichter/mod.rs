//! # 4D-Trichter (Gabriel) Implementation
//!
//! This module implements the 4D-Trichter (Funnel) system as specified in the
//! 4D_Trichter.pdf Delta-Blueprint. It provides deterministic, offline coupling
//! between 4D process space and 5D resonance fields through a Hyperbion layer.
//!
//! ## Key Components
//!
//! - **Funnel**: 4D kinetic compressor for directed pattern condensation
//! - **Hyperbion**: Viscoelastic morphodynamic coupling layer
//! - **HDAG**: Hyperdimensional acyclic resonance grid (5D tensors)
//! - **Policies**: Explore, Exploit, Homeostasis modes

pub mod funnel;
pub mod hdag;
pub mod hyperbion;
pub mod lift;
pub mod policies;
pub mod tick;
pub mod types;

pub use funnel::FunnelGraph;
pub use hdag::HDAGField;
pub use hyperbion::Hyperbion;
pub use lift::{lift, proj_4d};
pub use policies::{Policy, PolicyParams};
pub use tick::{coupling_tick, TickResult};
pub use types::*;
