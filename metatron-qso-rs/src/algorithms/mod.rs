//! Quantum algorithms built on the simulators.
//!
//! * [`gradients`] — adjoint, parameter-shift and finite-difference gradients
//! * [`geometry`] — quantum Fisher information, natural gradient,
//!   variational imaginary-time evolution
//! * [`ansatz`] — hardware-efficient templates and the explicit QAOA circuit
//! * [`models`] — Ising, Heisenberg, MaxCut and H₂ Hamiltonians
//! * [`vqe`] — variational eigensolver
//! * [`excited`] — excited states by VQD and SSVQE
//! * [`qaoa`] — QAOA for weighted MaxCut with exact optimum and INTERP
//! * [`combinatorial`] — QUBO and Ising problems; QAOA with X, XY and
//!   warm-start mixers
//! * [`grover`] — Grover search
//! * [`qft`] — quantum Fourier transform and phase estimation
//! * [`time_evolution`] — Trotter–Suzuki and qDRIFT circuits with error bounds
//! * [`classifier`] — variational classifier with data re-uploading
//! * [`graph_classifier`] — graph classifier with QAOA-type quantum layers

pub mod ansatz;
pub mod classifier;
pub mod combinatorial;
pub mod excited;
pub mod geometry;
pub mod gradients;
pub mod graph_classifier;
pub mod grover;
pub mod models;
pub mod qaoa;
pub mod qft;
pub mod time_evolution;
pub mod vqe;

pub use gradients::GradientMethod;
pub use graph_classifier::GraphClassifier;
pub use qaoa::{MaxCutQaoa, QaoaConfig, QaoaResult};
pub use vqe::{Vqe, VqeResult};
