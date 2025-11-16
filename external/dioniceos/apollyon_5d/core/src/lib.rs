//! # 5D System Framework
//!
//! A deterministic framework for simulating coupled dynamical systems in five dimensions.
//! Based on the mathematical specification in 5D_System_Framework.pdf
//!
//! ## Modules
//! - `state`: 5D state vectors and operations
//! - `coupling`: Coupling matrix and interaction types
//! - `dynamics`: Vector field and evolution operators
//! - `integration`: Numerical integration schemes
//! - `stability`: Stability analysis and Lyapunov exponents
//! - `projection`: Dimension reduction and visualization
//! - `template`: Domain-specific instantiation templates
//! - `export`: Data export in CSV and JSON formats
//! - `validation`: Reference solutions for testing

pub mod coupling;
pub mod dynamics;
pub mod ensemble;
pub mod export;
pub mod integration;
pub mod projection;
pub mod stability;
pub mod state;
pub mod template;
pub mod validation;

pub use coupling::{CouplingMatrix, CouplingType};
pub use dynamics::{SystemParameters, VectorField};
pub use integration::Integrator;
pub use state::State5D;
pub use template::Template;
