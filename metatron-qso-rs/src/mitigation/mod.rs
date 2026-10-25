//! Error mitigation: estimates of noiseless expectation values from noisy
//! executions.
//!
//! * [`readout`] — readout calibration (tensored or full) and mitigation
//!   of measured counts.
//! * [`zne`] — zero-noise extrapolation by unitary folding.
//! * [`pec`] — probabilistic error cancellation of Pauli noise.
//! * [`twirl`] — Pauli twirling of two-qubit Clifford gates.
//!
//! Every mitigated value is an [`Estimate`](crate::estimation::Estimate)
//! whose standard error includes the statistical cost of mitigation.

pub mod pec;
pub mod readout;
pub mod twirl;
pub mod zne;

pub use readout::{ReadoutCalibration, nearest_probability_distribution};
