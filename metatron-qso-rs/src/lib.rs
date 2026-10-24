//! # Metatron QSO 2.0
//!
//! Validated quantum simulation in pure Rust.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations, rust_2018_idioms)]

pub mod error;
pub mod graph;
pub mod linalg;
pub mod noise;
pub(crate) mod parallel;
pub mod quantum;
pub mod rng;
pub mod walks;

pub use error::{QsoError, Result};
pub use num_complex::Complex64 as C64;
pub use rng::Rng;
