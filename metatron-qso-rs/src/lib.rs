//! # Metatron QSO — validated quantum simulation in pure Rust
//!
//! QSO simulates quantum circuits and quantum walks exactly and runs the
//! standard variational and textbook algorithms on top. Every algorithm is
//! tested against closed-form or literature results; the test-suite is the
//! specification.
//!
//! | Module | Contents |
//! |---|---|
//! | [`quantum`] | n-qubit [`StateVector`](quantum::StateVector), gates with arbitrary controls, parametrised [`Circuit`](quantum::Circuit)s, [`PauliSum`](quantum::PauliSum) observables, OpenQASM 2.0 |
//! | [`noise`] | [`DensityMatrix`](noise::DensityMatrix) simulator with Kraus channels and noise models |
//! | [`algorithms`] | adjoint gradients, VQE, QAOA-MaxCut, Grover, QFT/phase estimation, variational classifier, model Hamiltonians |
//! | [`optimize`] | L-BFGS, Adam, Nelder–Mead, SPSA, gradient descent |
//! | [`graph`] | graphs, Platonic solids, Metatron's Cube / Fruit of Life, exact automorphism groups |
//! | [`walks`] | continuous-time quantum walks, spatial search, dephasing, classical baselines |
//! | [`photonics`] | linear interferometers, permanents, exact boson sampling |
//! | [`backend`] | execution backends and shot-based expectation values |
//! | [`linalg`] | Hermitian eigensolvers, Lanczos, Krylov propagation, sparse matrices |
//!
//! ## Conventions
//!
//! * Qubit 0 is the least significant bit of a basis index (as in Qiskit);
//!   bit strings print qubit `n−1` first ([`quantum::bitstring`]).
//! * Time evolution is `exp(−iHt)`; rotations are `R_P(θ) = exp(−iθP/2)`.
//! * All randomness is explicit and seeded ([`Rng`]); parallel reductions
//!   are bit-for-bit reproducible for any number of threads.
//!
//! ## Example
//!
//! ```
//! use metatron_qso::prelude::*;
//!
//! // Ground state of H₂ (STO-3G, 0.735 Å) with VQE.
//! let h2 = models::h2_sto3g();
//! let ansatz = ansatz::real_amplitudes(2, 1, Entanglement::Linear);
//! let vqe = Vqe::new(h2.hamiltonian.clone(), ansatz)?.with_restarts(4);
//! let energy = vqe.run()?.energy + h2.nuclear_repulsion;
//! assert!((energy - (-1.137306)).abs() < 1e-5);
//!
//! // QAOA on Metatron's Cube (= K13, maximum cut 42).
//! let qaoa = MaxCutQaoa::new(&metatron::metatron_cube(), 2)?;
//! let result = qaoa.optimize(&QaoaConfig::default())?;
//! assert!(result.approximation_ratio.unwrap() > 0.9);
//! # Ok::<(), metatron_qso::QsoError>(())
//! ```

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations, rust_2018_idioms)]

pub mod algorithms;
pub mod backend;
pub mod error;
pub mod estimation;
pub mod graph;
pub mod linalg;
pub mod mitigation;
pub mod noise;
pub mod optimize;
pub(crate) mod parallel;
pub mod photonics;
pub mod quantum;
pub mod rng;
pub mod walks;

pub use error::{QsoError, Result};
pub use num_complex::Complex64 as C64;
pub use rng::Rng;

/// Convenient glob import of the most used items.
pub mod prelude {
    pub use crate::C64;
    pub use crate::algorithms::ansatz::{self, Entanglement, Rotation};
    pub use crate::algorithms::{
        GradientMethod, MaxCutQaoa, QaoaConfig, QaoaResult, Vqe, VqeResult, grover, models, qft,
    };
    pub use crate::backend::{
        Backend, DensityMatrixBackend, StatevectorBackend, estimate_expectation,
    };
    pub use crate::error::{QsoError, Result};
    pub use crate::graph::{Graph, generators, metatron};
    pub use crate::noise::{Channel, DensityMatrix, NoiseModel};
    pub use crate::optimize::{Method, Options, minimize};
    pub use crate::photonics::{self, BosonSampler};
    pub use crate::quantum::{
        Circuit, GateKind, Param, Parameter, Pauli, PauliString, PauliSum, StateVector, bitstring,
        from_qasm2, to_qasm2,
    };
    pub use crate::rng::Rng;
    pub use crate::walks::{ContinuousWalk, DephasingWalk, Generator, SpatialSearch};
}
