//! Error mitigation: estimates of noiseless expectation values from noisy
//! executions.
//!
//! * [`readout`] — readout calibration (tensored or full) and mitigation
//!   of measured counts.
//! * [`zne`] — zero-noise extrapolation by unitary folding.
//! * [`pec`] — probabilistic error cancellation of Pauli noise.
//! * [`twirl`] — Pauli twirling of two-qubit Clifford gates.
//! * [`distillation`] — virtual distillation: expectation values of the
//!   purified state `ρᵏ/Tr(ρᵏ)`, exactly or with circuits on `k` copies.
//!
//! Every mitigated value is an [`Estimate`](crate::estimation::Estimate)
//! whose standard error includes the statistical cost of mitigation.

pub mod distillation;
pub mod pec;
pub mod readout;
pub mod twirl;
pub mod zne;

pub use distillation::{
    DistillationCircuits, DistillationSetting, MAX_COPIES, distilled_expectation,
    dominant_expectation,
};
pub use readout::{ReadoutCalibration, nearest_probability_distribution};
