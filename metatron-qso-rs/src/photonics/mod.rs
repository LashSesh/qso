//! Linear optics: interferometers, permanents and boson sampling.
//!
//! An `m`-mode linear interferometer is an `m × m` unitary `U` that maps the
//! creation operator of input mode `j` to `Σ_i U_ij a_i†`. For `n` photons
//! entering with occupation `S` and leaving with occupation `T`, the output
//! probability is
//!
//! ```text
//! P(T) = |perm(U_{T,S})|² / (Π_j s_j! · Π_i t_i!)      (indistinguishable)
//! P(T) = perm(|U_{T,S}|²) / Π_i t_i!                   (distinguishable)
//! ```
//!
//! where `U_{T,S}` repeats row `i` `t_i` times and column `j` `s_j` times
//! (Aaronson & Arkhipov 2011). See [`BosonSampler`].

mod boson;
pub mod permanent;

pub use boson::{
    BosonSampler, Distribution, MAX_OUTPUT_ENTRIES, MAX_PATTERNS, count_patterns,
    occupation_patterns,
};
pub use permanent::{MAX_PERMANENT_DIM, permanent, permanent_real};

use crate::error::{QsoError, Result};
use crate::linalg::CMatrix;
use crate::{C64, Rng};

/// Largest number of modes of an interferometer (the unitary is dense).
pub const MAX_MODES: usize = 1024;

/// Largest number of photons. Every probability and every sample costs at
/// least one `n × n` permanent, i.e. `O(n·2ⁿ)` operations.
pub const MAX_PHOTONS: usize = 30;

/// Haar-random `m × m` unitary (QR decomposition of a complex Ginibre matrix
/// with the phases of `R`'s diagonal divided out; Mezzadri 2007).
pub fn haar_unitary(m: usize, rng: &mut Rng) -> Result<CMatrix> {
    check_modes(m)?;
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let z = CMatrix::from_fn(m, m, |_, _| C64::new(rng.normal() * s, rng.normal() * s));
    let qr = z.qr();
    let (q, r) = (qr.q(), qr.r());
    Ok(CMatrix::from_fn(m, m, |i, j| {
        let d = r[(j, j)];
        let phase = if d.norm() > 0.0 {
            d / d.norm()
        } else {
            C64::new(1.0, 0.0)
        };
        q[(i, j)] * phase
    }))
}

/// Two-mode beam splitter
/// `[[cos θ, −e^{−iφ} sin θ], [e^{iφ} sin θ, cos θ]]`; `θ = π/4` is 50:50.
pub fn beam_splitter(theta: f64, phi: f64) -> CMatrix {
    let (s, c) = theta.sin_cos();
    CMatrix::from_row_slice(
        2,
        2,
        &[
            C64::new(c, 0.0),
            -C64::from_polar(s, -phi),
            C64::from_polar(s, phi),
            C64::new(c, 0.0),
        ],
    )
}

/// Largest entry of `|U†U − I|`; zero for an exactly unitary matrix and
/// infinite for a matrix that is not square or whose deviation is not
/// finite, so that every tolerance check rejects it.
pub fn unitarity_error(u: &CMatrix) -> f64 {
    if !u.is_square() {
        return f64::INFINITY;
    }
    let m = u.nrows();
    let g = u.adjoint() * u;
    (0..m)
        .flat_map(|i| (0..m).map(move |j| (i, j)))
        .map(|(i, j)| {
            let target = if i == j { 1.0 } else { 0.0 };
            (g[(i, j)] - target).norm()
        })
        .fold(0.0, |worst, e| {
            if e.is_nan() {
                f64::INFINITY
            } else {
                worst.max(e)
            }
        })
}

fn check_modes(m: usize) -> Result<()> {
    if m == 0 || m > MAX_MODES {
        return Err(QsoError::invalid(
            "modes",
            format!("number of modes must lie in 1..={MAX_MODES}, got {m}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haar_unitaries_are_unitary_and_reproducible() {
        let mut rng = Rng::seed_from_u64(3);
        let u = haar_unitary(12, &mut rng).unwrap();
        assert!(unitarity_error(&u) < 1e-12);
        let again = haar_unitary(12, &mut Rng::seed_from_u64(3)).unwrap();
        assert_eq!(u, again);
    }

    #[test]
    fn haar_measure_has_the_right_second_moment() {
        // For Haar-random U(m), E|U_11|² = 1/m and E|U_11|⁴ = 2/(m(m+1)).
        let (m, samples) = (4, 20_000);
        let mut rng = Rng::seed_from_u64(9);
        let (mut s2, mut s4) = (0.0, 0.0);
        for _ in 0..samples {
            let p = haar_unitary(m, &mut rng).unwrap()[(0, 0)].norm_sqr();
            s2 += p;
            s4 += p * p;
        }
        let (s2, s4) = (s2 / samples as f64, s4 / samples as f64);
        assert!((s2 - 0.25).abs() < 0.01, "{s2}");
        assert!((s4 - 0.1).abs() < 0.01, "{s4}");
    }

    #[test]
    fn beam_splitter_is_unitary() {
        assert!(unitarity_error(&beam_splitter(0.3, 1.1)) < 1e-15);
        assert!(haar_unitary(0, &mut Rng::seed_from_u64(1)).is_err());
    }

    #[test]
    fn unitarity_error_rejects_malformed_matrices() {
        let c = |re: f64| C64::new(re, 0.0);
        assert_eq!(
            unitarity_error(&CMatrix::from_element(3, 2, c(0.5))),
            f64::INFINITY
        );
        assert_eq!(
            unitarity_error(&CMatrix::from_element(2, 3, c(0.5))),
            f64::INFINITY
        );
        let mut u = CMatrix::identity(2, 2);
        u[(0, 1)] = c(f64::NAN);
        assert_eq!(unitarity_error(&u), f64::INFINITY);
        u[(0, 1)] = c(f64::INFINITY);
        assert_eq!(unitarity_error(&u), f64::INFINITY);
        assert_eq!(unitarity_error(&CMatrix::zeros(0, 0)), 0.0);
    }
}
