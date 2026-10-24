//! Quantum algorithms built on the simulators.
//!
//! * [`gradients`] — adjoint, parameter-shift and finite-difference gradients
//! * [`ansatz`] — hardware-efficient templates and the explicit QAOA circuit
//! * [`models`] — Ising, Heisenberg, MaxCut and H₂ Hamiltonians
//! * [`vqe`] — variational eigensolver
//! * [`qaoa`] — QAOA for weighted MaxCut with exact optimum and INTERP
//! * [`grover`] — Grover search
//! * [`qft`] — quantum Fourier transform and phase estimation
//! * [`classifier`] — variational classifier with data re-uploading

pub mod ansatz;
pub mod classifier;
pub mod gradients;
pub mod grover;
pub mod models;
pub mod qaoa;
pub mod qft;
pub mod vqe;

pub use gradients::GradientMethod;
pub use qaoa::{MaxCutQaoa, QaoaConfig, QaoaResult};
pub use vqe::{Vqe, VqeResult};
