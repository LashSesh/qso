//! Linear-algebra building blocks.
//!
//! * [`dense`] — Hermitian eigendecomposition and `exp(−iHt)` for dense
//!   matrices (nalgebra).
//! * [`sparse`] — compressed sparse row matrices for graph Hamiltonians.
//! * [`krylov`] — matrix-free Lanczos ground states, Krylov propagation
//!   `exp(−iHt)|v⟩` and cyclic-subspace spectra for operators far too large
//!   to store densely.

pub mod dense;
pub mod krylov;
pub mod sparse;

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
