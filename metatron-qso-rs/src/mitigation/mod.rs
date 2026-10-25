//! Error mitigation: estimates of noiseless expectation values from noisy
//! executions.
//!
//! * [`readout`] — readout calibration (tensored or full) and mitigation
//!   of measured counts.
//!
//! Every mitigated value is an [`Estimate`](crate::estimation::Estimate)
//! whose standard error includes the statistical cost of mitigation.

pub mod readout;

pub use readout::{ReadoutCalibration, nearest_probability_distribution};
