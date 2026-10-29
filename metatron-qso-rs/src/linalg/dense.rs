//! Dense Hermitian linear algebra.

use nalgebra::DMatrix;
use num_complex::Complex64 as C64;

/// Dense complex matrix.
pub type CMatrix = DMatrix<C64>;
/// Dense real matrix.
pub type RMatrix = DMatrix<f64>;

/// Eigendecomposition of a Hermitian matrix.
///
/// Returns eigenvalues in ascending order and the matching orthonormal
/// eigenvectors as the columns of the second component. Only the lower
/// triangle of `m` is read.
pub fn eigh(m: &CMatrix) -> (Vec<f64>, CMatrix) {
    assert!(m.is_square(), "eigh requires a square matrix");
    let eig = m.clone().symmetric_eigen();
    let mut order: Vec<usize> = (0..eig.eigenvalues.len()).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));
    let values = order.iter().map(|&i| eig.eigenvalues[i]).collect();
    let vectors = CMatrix::from_fn(m.nrows(), m.ncols(), |r, c| eig.eigenvectors[(r, order[c])]);
    (values, vectors)
}

/// Eigendecomposition of a real symmetric matrix (ascending eigenvalues).
pub fn eigh_real(m: &RMatrix) -> (Vec<f64>, RMatrix) {
    assert!(m.is_square(), "eigh_real requires a square matrix");
    let eig = m.clone().symmetric_eigen();
    let mut order: Vec<usize> = (0..eig.eigenvalues.len()).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));
    let values = order.iter().map(|&i| eig.eigenvalues[i]).collect();
    let vectors = RMatrix::from_fn(m.nrows(), m.ncols(), |r, c| eig.eigenvectors[(r, order[c])]);
    (values, vectors)
}

/// `exp(−iHt)` of a Hermitian matrix via its eigendecomposition.
pub fn expm_hermitian(h: &CMatrix, t: f64) -> CMatrix {
    let (values, vectors) = eigh(h);
    from_spectrum(&values, &vectors, |e| C64::from_polar(1.0, -e * t))
}

/// Complex matrix product through three real products (Gauss), which
/// run on nalgebra's optimised real kernels: several times faster than
/// its generic complex product.
pub(crate) fn cmul(a: &CMatrix, b: &CMatrix) -> CMatrix {
    let (ar, ai) = (a.map(|z| z.re), a.map(|z| z.im));
    let (br, bi) = (b.map(|z| z.re), b.map(|z| z.im));
    let p1 = &ar * &br;
    let p2 = &ai * &bi;
    let p3 = (ar + ai) * (br + bi);
    CMatrix::from_fn(p1.nrows(), p1.ncols(), |i, j| {
        C64::new(
            p1[(i, j)] - p2[(i, j)],
            p3[(i, j)] - p1[(i, j)] - p2[(i, j)],
        )
    })
}

/// Largest norm `‖A/2^s‖₁` at which [`expm`] evaluates its polynomial.
const EXPM_THETA: f64 = 0.8;

/// Squarings [`expm`] takes for a matrix of 1-norm `norm`.
pub(crate) fn expm_squarings(norm: f64) -> u32 {
    if norm > EXPM_THETA {
        (norm / EXPM_THETA).log2().ceil() as u32
    } else {
        0
    }
}

/// Largest column sum of absolute values.
pub(crate) fn one_norm(a: &CMatrix) -> f64 {
    a.column_iter()
        .map(|c| c.iter().map(|z| z.norm()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// `exp(A)` of a general complex matrix: the Taylor polynomial of degree
/// 16 of `B = A/2^s`, `‖B‖₁ ≤ 0.8`, in six products (Paterson–Stockmeyer),
/// squared `s` times. The polynomial's truncation error is at most
/// `Σ_{k>16} 0.8^k/k! < 7·10⁻¹⁷`, against `‖exp(B)‖ ≥ e^{−0.8}`; the
/// squarings multiply rounding errors by about `2^s ≈ ‖A‖₁`. Products run
/// on [`cmul`].
pub(crate) fn expm(a: &CMatrix) -> CMatrix {
    let n = a.nrows();
    let s = expm_squarings(one_norm(a));
    let b = a * C64::new(0.5f64.powi(s as i32), 0.0);
    let b2 = cmul(&b, &b);
    let b3 = cmul(&b2, &b);
    let b4 = cmul(&b2, &b2);
    let mut inv = [1.0f64; 17];
    for k in 1..17 {
        inv[k] = inv[k - 1] / k as f64;
    }
    // Σ_{i<4} B^i/(4j + i)!.
    let chunk = |j: usize| -> CMatrix {
        let mut c = &b * C64::new(inv[4 * j + 1], 0.0)
            + &b2 * C64::new(inv[4 * j + 2], 0.0)
            + &b3 * C64::new(inv[4 * j + 3], 0.0);
        for i in 0..n {
            c[(i, i)] += inv[4 * j];
        }
        c
    };
    let mut t = chunk(3) + &b4 * C64::new(inv[16], 0.0);
    for j in (0..3).rev() {
        t = chunk(j) + cmul(&b4, &t);
    }
    for _ in 0..s {
        t = cmul(&t, &t);
    }
    t
}

/// `V · diag(f(λ)) · V†`.
pub(crate) fn from_spectrum(values: &[f64], vectors: &CMatrix, f: impl Fn(f64) -> C64) -> CMatrix {
    let n = vectors.nrows();
    let mut scaled = vectors.clone();
    for (c, &e) in values.iter().enumerate() {
        let w = f(e);
        for r in 0..n {
            scaled[(r, c)] *= w;
        }
    }
    scaled * vectors.adjoint()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_hermitian(n: usize, seed: u64) -> CMatrix {
        let mut rng = crate::rng::Rng::seed_from_u64(seed);
        let a = CMatrix::from_fn(n, n, |_, _| C64::new(rng.normal(), rng.normal()));
        (&a + a.adjoint()) * C64::new(0.5, 0.0)
    }

    #[test]
    fn eigh_reconstructs_matrix() {
        let h = random_hermitian(9, 1);
        let (values, vectors) = eigh(&h);
        assert!(values.windows(2).all(|w| w[0] <= w[1]));
        let rebuilt = from_spectrum(&values, &vectors, |e| C64::new(e, 0.0));
        assert!((rebuilt - &h).norm() < 1e-10);
        let gram = vectors.adjoint() * &vectors;
        assert!((gram - CMatrix::identity(9, 9)).norm() < 1e-10);
    }

    #[test]
    fn general_exponential_matches_the_spectral_one() {
        // A Hermitian generator −iHt, at norms that need no squaring and
        // many, against the eigendecomposition.
        let h = random_hermitian(12, 3);
        for t in [0.05, 1.0, 40.0] {
            let a = &h * C64::new(0.0, -t);
            let want = expm_hermitian(&h, t);
            assert!((expm(&a) - &want).norm() < 1e-12 * (1.0 + t), "{t}");
        }
        // A non-normal one: exp of a nilpotent Jordan block is exact.
        let mut j = CMatrix::zeros(4, 4);
        for i in 0..3 {
            j[(i, i + 1)] = C64::new(2.0, 0.0);
        }
        let e = expm(&j);
        // (2^k / k!) on the k-th superdiagonal.
        for (k, want) in [1.0, 2.0, 2.0, 4.0 / 3.0].into_iter().enumerate() {
            assert!((e[(0, k)].re - want).abs() < 1e-15, "{k}");
        }
        let a = random_hermitian(7, 4);
        let b = random_hermitian(7, 5);
        assert!((cmul(&a, &b) - &a * &b).norm() < 1e-12);
    }

    #[test]
    fn expm_is_unitary_and_composes() {
        let h = random_hermitian(6, 2);
        let u1 = expm_hermitian(&h, 0.3);
        let u2 = expm_hermitian(&h, 0.5);
        let u12 = expm_hermitian(&h, 0.8);
        assert!((u1.adjoint() * &u1 - CMatrix::identity(6, 6)).norm() < 1e-10);
        assert!((&u2 * &u1 - u12).norm() < 1e-10);
    }
}
