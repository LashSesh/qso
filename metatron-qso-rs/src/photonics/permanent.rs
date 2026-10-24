//! Matrix permanents.
//!
//! `perm(A) = Σ_σ Π_i a_{i,σ(i)}` is the bosonic counterpart of the
//! determinant: output amplitudes of indistinguishable photons are
//! permanents of submatrices of the interferometer unitary. Computing it is
//! #P-hard; the best known exact algorithms take `O(n·2ⁿ)` operations.
//!
//! This module uses Glynn's formula (Glynn 2010) with the sign vectors
//! enumerated in Gray-code order, so each of the `2ⁿ⁻¹` terms costs `O(n)`:
//!
//! ```text
//! perm(A) = 2^{1−n} Σ_{δ ∈ {±1}ⁿ, δ₁ = 1} (Π_k δ_k) Π_j Σ_i δ_i a_{ij}
//! ```
//!
//! The enumeration is split into fixed chunks that are summed in order, so
//! the result is bit-for-bit identical for any number of threads.

use nalgebra::ComplexField;

use crate::C64;
use crate::error::{QsoError, Result};
use crate::linalg::{CMatrix, RMatrix};
use crate::parallel;

/// Largest dimension accepted by [`permanent`] and [`permanent_real`].
///
/// A 40 × 40 permanent already needs about 2·10¹³ operations; larger
/// matrices are rejected rather than left running for days.
pub const MAX_PERMANENT_DIM: usize = 40;

/// Permanent of a square complex matrix (Glynn's formula, `O(n·2ⁿ)`).
///
/// The permanent of the 0 × 0 matrix is 1.
pub fn permanent(a: &CMatrix) -> Result<C64> {
    let (n, data) = row_major("matrix", a.nrows(), a.ncols(), |r, c| a[(r, c)])?;
    Ok(glynn(&data, n))
}

/// Permanent of a square real matrix (Glynn's formula, `O(n·2ⁿ)`).
///
/// Used for distinguishable particles, whose output probabilities are
/// permanents of the entry-wise squared moduli `|U_{ij}|²`.
pub fn permanent_real(a: &RMatrix) -> Result<f64> {
    let (n, data) = row_major("matrix", a.nrows(), a.ncols(), |r, c| a[(r, c)])?;
    Ok(glynn(&data, n))
}

fn row_major<T>(
    name: &'static str,
    rows: usize,
    cols: usize,
    get: impl Fn(usize, usize) -> T,
) -> Result<(usize, Vec<T>)> {
    if rows != cols {
        return Err(QsoError::invalid(
            name,
            format!("permanent needs a square matrix, got {rows} × {cols}"),
        ));
    }
    if rows > MAX_PERMANENT_DIM {
        return Err(QsoError::invalid(
            name,
            format!("dimension {rows} exceeds the supported maximum {MAX_PERMANENT_DIM}"),
        ));
    }
    let data = (0..rows)
        .flat_map(|r| (0..cols).map(move |c| (r, c)))
        .map(|(r, c)| get(r, c))
        .collect();
    Ok((rows, data))
}

/// Glynn's formula on a row-major `n × n` matrix.
pub(crate) fn glynn<T>(a: &[T], n: usize) -> T
where
    T: ComplexField<RealField = f64> + Copy + Send + Sync + std::iter::Sum<T>,
{
    match n {
        0 => return T::one(),
        1 => return a[0],
        _ => {}
    }
    let terms = 1u64 << (n - 1);
    let total: T = parallel::sum_ranges(terms, |start, end| glynn_range(a, n, start, end));
    total * T::from_real(0.5f64.powi((n - 1) as i32))
}

/// Terms `start..end` of Glynn's sum. Term `k` uses the sign vector given by
/// the Gray code `k ^ (k >> 1)`: bit `b` set means `δ_{b+1} = −1`.
fn glynn_range<T>(a: &[T], n: usize, start: u64, end: u64) -> T
where
    T: ComplexField + Copy,
{
    let gray = start ^ (start >> 1);
    let mut negative: Vec<bool> = (0..n)
        .map(|i| i > 0 && (gray >> (i - 1)) & 1 == 1)
        .collect();
    let mut odd = gray.count_ones() % 2 == 1;
    // Column sums s_j = Σ_i δ_i a_ij.
    let mut s = vec![T::zero(); n];
    for (i, row) in a.chunks_exact(n).enumerate() {
        for (sj, &aij) in s.iter_mut().zip(row) {
            if negative[i] {
                *sj -= aij;
            } else {
                *sj += aij;
            }
        }
    }
    let mut acc = T::zero();
    let mut k = start;
    loop {
        let product = s.iter().fold(T::one(), |p, &x| p * x);
        if odd {
            acc -= product;
        } else {
            acc += product;
        }
        k += 1;
        if k == end {
            break;
        }
        // Gray codes of k − 1 and k differ in bit trailing_zeros(k).
        let i = k.trailing_zeros() as usize + 1;
        let row = &a[i * n..(i + 1) * n];
        let now_negative = !negative[i];
        negative[i] = now_negative;
        for (sj, &aij) in s.iter_mut().zip(row) {
            if now_negative {
                *sj -= aij + aij;
            } else {
                *sj += aij + aij;
            }
        }
        odd = !odd;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;

    /// Definition: sum over all permutations (Heap's algorithm).
    fn naive(a: &CMatrix) -> C64 {
        let n = a.nrows();
        let mut perm: Vec<usize> = (0..n).collect();
        let mut c = vec![0usize; n];
        let term = |p: &[usize]| (0..n).fold(C64::new(1.0, 0.0), |acc, i| acc * a[(i, p[i])]);
        let mut total = term(&perm);
        let mut i = 0;
        while i < n {
            if c[i] < i {
                if i % 2 == 0 {
                    perm.swap(0, i);
                } else {
                    perm.swap(c[i], i);
                }
                total += term(&perm);
                c[i] += 1;
                i = 0;
            } else {
                c[i] = 0;
                i += 1;
            }
        }
        total
    }

    fn random_matrix(n: usize, rng: &mut Rng) -> CMatrix {
        CMatrix::from_fn(n, n, |_, _| C64::new(rng.normal(), rng.normal()))
    }

    #[test]
    fn matches_the_definition() {
        let mut rng = Rng::seed_from_u64(11);
        for n in 1..=7 {
            let a = random_matrix(n, &mut rng);
            let p = permanent(&a).unwrap();
            let q = naive(&a);
            assert!((p - q).norm() < 1e-10 * q.norm().max(1.0), "n = {n}");
        }
    }

    #[test]
    fn closed_forms() {
        assert_eq!(
            permanent(&CMatrix::zeros(0, 0)).unwrap(),
            C64::new(1.0, 0.0)
        );
        // perm of the all-ones n × n matrix is n!.
        let mut factorial = 1.0;
        for n in 1..=12usize {
            factorial *= n as f64;
            let p = permanent_real(&RMatrix::from_element(n, n, 1.0)).unwrap();
            assert!((p - factorial).abs() < 1e-9 * factorial, "n = {n}");
        }
        // perm(I) = 1 and perm of a 2 × 2 matrix is ad + bc.
        assert!((permanent_real(&RMatrix::identity(9, 9)).unwrap() - 1.0).abs() < 1e-12);
        let m = RMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert!((permanent_real(&m).unwrap() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn parallel_chunks_agree_with_one_serial_pass() {
        // n = 18 has 2¹⁷ terms, enough to cross the parallel threshold.
        let mut rng = Rng::seed_from_u64(5);
        let a = random_matrix(18, &mut rng);
        let data: Vec<C64> = (0..18)
            .flat_map(|r| (0..18).map(move |c| (r, c)))
            .map(|(r, c)| a[(r, c)])
            .collect();
        let serial = glynn_range(&data, 18, 0, 1 << 17) * 0.5f64.powi(17);
        let chunked = permanent(&a).unwrap();
        assert!((serial - chunked).norm() < 1e-9 * serial.norm());
        // Deterministic for any thread count: two evaluations are bit-identical.
        let again = permanent(&a).unwrap();
        assert_eq!(chunked.re.to_bits(), again.re.to_bits());
        assert_eq!(chunked.im.to_bits(), again.im.to_bits());
    }

    #[test]
    fn rejects_bad_shapes() {
        assert!(permanent(&CMatrix::zeros(2, 3)).is_err());
        assert!(permanent_real(&RMatrix::zeros(41, 41)).is_err());
    }
}
