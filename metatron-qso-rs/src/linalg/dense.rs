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
    fn expm_is_unitary_and_composes() {
        let h = random_hermitian(6, 2);
        let u1 = expm_hermitian(&h, 0.3);
        let u2 = expm_hermitian(&h, 0.5);
        let u12 = expm_hermitian(&h, 0.8);
        assert!((u1.adjoint() * &u1 - CMatrix::identity(6, 6)).norm() < 1e-10);
        assert!((&u2 * &u1 - u12).norm() < 1e-10);
    }
}
