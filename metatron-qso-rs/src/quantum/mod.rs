//! Gate-based quantum simulation: states, gates, circuits and observables.
//!
//! * [`StateVector`] — n-qubit pure states with multi-threaded gate kernels.
//! * [`Circuit`] — gate lists with symbolic, differentiable parameters.
//! * [`PauliSum`] — Hermitian observables, matrix-free expectation values and
//!   exact diagonalisation via Lanczos.
//! * [`qasm`] — OpenQASM 2.0 import/export.
//!
//! Qubit order is little-endian throughout (qubit 0 is the least
//! significant bit of a basis index), matching Qiskit.

pub mod circuit;
pub(crate) mod decompose;
pub mod gates;
pub(crate) mod kernels;
pub mod pauli;
pub mod qasm;
pub mod state;

pub use circuit::{Circuit, Instruction, Param, Parameter};
pub use gates::GateKind;
pub use pauli::{Pauli, PauliString, PauliSum};
pub use qasm::{from_qasm2, to_qasm2};
pub use state::{Counts, MAX_QUBITS, StateVector, bitstring};
