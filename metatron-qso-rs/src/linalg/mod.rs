//! Linear-algebra building blocks.
//!
//! * [`dense`] — Hermitian eigendecomposition and `exp(−iHt)` for dense
//!   matrices (nalgebra).
//! * [`sparse`] — compressed sparse row matrices for graph Hamiltonians.
//! * [`krylov`] — matrix-free Lanczos ground states, Krylov propagation
//!   `exp(−iHt)|v⟩` and cyclic-subspace spectra for operators far too large
//!   to store densely.
//!
//! Crate-internal solvers (used by the steady states of exchange, by the
//! transport to a trap and by radical pairs):
//!
//! * `gmres` — restarted GMRES with right preconditioning, the iteration of
//!   the exchange steady states (1.54) with a backward-error stopping rule,
//!   a relative breakdown test and a condition estimate.
//! * `sylvester` — the complex Schur form `A = QTQ†` with a bounded
//!   iteration and one retry, and Bartels–Stewart solves of
//!   `AX − XA† − σX = C` (recursive blocked, products in fixed panels, so
//!   bit-identical on any number of threads).
//! * `banded` — reverse Cuthill–McKee ordering and LU with partial pivoting
//!   in band storage, with solves for `A` and `A†`.

pub mod dense;
pub mod krylov;
pub mod sparse;

// Used from 1.63's transport code.
#[allow(dead_code)]
pub(crate) mod banded;
// Used from 1.63's transport code and 1.65's radical pairs (and by the
// exchange steady states).
#[allow(dead_code)]
pub(crate) mod gmres;
// Used from 1.63's transport code and 1.65's radical pairs.
#[allow(dead_code)]
pub(crate) mod sylvester;

pub use dense::{CMatrix, RMatrix, eigh, eigh_real, expm_hermitian};
pub use krylov::{
    CyclicSpectrum, KrylovOptions, LanczosResult, cyclic_spectrum, expm_multiply,
    lanczos_ground_state,
};
pub use sparse::CsrMatrix;

use num_complex::Complex64 as C64;

/// Euclidean norm of a complex vector.
pub fn norm(v: &[C64]) -> f64 {
    crate::parallel::sum_f64(v.len(), |i| v[i].norm_sqr()).sqrt()
}

/// Inner product `⟨a|b⟩ = Σ conj(aᵢ)·bᵢ`.
pub fn inner(a: &[C64], b: &[C64]) -> C64 {
    debug_assert_eq!(a.len(), b.len());
    crate::parallel::sum_c64(a.len(), |i| a[i].conj() * b[i])
}
