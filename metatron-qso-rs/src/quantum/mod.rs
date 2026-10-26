//! Gate-based quantum simulation: states, gates, circuits and observables.
//!
//! * [`StateVector`] — n-qubit pure states with multi-threaded gate kernels.
//! * [`Circuit`] — gate lists with symbolic, differentiable parameters.
//! * [`PauliSum`] — Hermitian observables, matrix-free expectation values and
//!   exact diagonalisation via Lanczos; [`QubitOperator`] for complex
//!   combinations of Pauli strings.
//! * [`qasm`] — OpenQASM 2.0 import/export; [`openqasm`] — OpenQASM 2 and
//!   3 into [`Program`]s with measurements, resets and conditions.
//!
//! Qubit order is little-endian throughout (qubit 0 is the least
//! significant bit of a basis index), matching Qiskit.

pub mod circuit;
pub(crate) mod decompose;
pub mod gates;
pub(crate) mod kernels;
pub mod openqasm;
pub mod pauli;
pub(crate) mod plan;
pub mod program;
pub mod qasm;
pub mod qubit_operator;
pub(crate) mod simd;
pub mod state;

pub use circuit::{Circuit, Instruction, Param, Parameter};
pub use gates::GateKind;
pub use openqasm::to_qasm3;
pub use pauli::{Pauli, PauliString, PauliSum};
pub use program::{Condition, Operation, Program, Register, record_bitstring};
pub use qasm::{from_qasm2, to_qasm2};
pub use qubit_operator::QubitOperator;
pub use simd::{set_simd, simd_level};
pub use state::{Counts, MAX_QUBITS, StateVector, bitstring};
