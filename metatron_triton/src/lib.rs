//! TRITON Spiral Search Core for Q⊗DASH
//!
//! This module implements the TRITON evolutionary spiral search algorithm,
//! inspired by Informationsalchemie, for use as an optimization strategy
//! within the Seraphic Calibration Shell.
//!
//! ## Core Concepts
//!
//! - **SpectralSignature**: A three-dimensional quality metric (ψ, ρ, ω)
//! - **TritonSpiral**: Golden-angle spiral evolution with momentum
//! - **TritonSearch**: Complete search engine integrating spiral + evaluation
//!
//! ## Example
//!
//! ```rust
//! use metatron_triton::{SpectralSignature, TritonSearch};
//!
//! // Define an evaluation function
//! let evaluator = |params: &[f64]| {
//!     let psi = 1.0 - (params[0] - 0.5).powi(2);
//!     let rho = 1.0 - (params[1] - 0.3).powi(2);
//!     let omega = 1.0 - (params[2] - 0.7).powi(2);
//!     SpectralSignature::new(psi, rho, omega)
//! };
//!
//! // Create TRITON search
//! let mut search = TritonSearch::new(3, 42, 100, evaluator);
//!
//! // Run optimization
//! for _ in 0..100 {
//!     let result = search.step();
//!     println!("Step {}: resonance = {:.6}", result.step_index, result.signature.resonance());
//! }
//!
//! let best = search.best_point().unwrap();
//! println!("Best point: {:?}", best);
//! ```

pub mod signature;
pub mod spiral;
pub mod search;
pub mod strategy;

pub use signature::SpectralSignature;
pub use spiral::TritonSpiral;
pub use search::{TritonSearch, TritonStepResult};
pub use strategy::{CalibrationSearchStrategy, TritonSearchStrategy, CalibrationProposal, CalibrationResult};
