//! Hafnians, loop hafnians and torontonians.
//!
//! The hafnian of a symmetric `2n × 2n` matrix sums the products of its
//! entries over the perfect matchings of `{1, …, 2n}`; the loop hafnian
//! also allows loops, weighted by the diagonal. Photon-number
//! probabilities of Gaussian states are (loop) hafnians, click
//! probabilities of threshold detectors are torontonians.
//!
//! Hafnians use the formula of Björklund, Gupt and Quesada (2019): with
//! `X` the pairwise swap and `B_Z` the rows and columns of the pairs `Z`,
//!
//! ```text
//! haf(A) = Σ_{Z ⊆ [n]} (−1)^{n−|Z|} [λⁿ] exp(Σ_k (Tr(C^k)/(2k) + ½ yᵀC^{k−1}x) λ^k)
//! ```
//!
//! with `C = X·B_Z` (and, for the loop hafnian, `C = X·(B_Z − diag B_Z)`,
//! `y = diag B_Z`, `x = X·y`). The power traces come from powers of `C`
//! by multiplication, so each of the `2ⁿ` terms costs `O(n⁴)`; traces from
//! the characteristic polynomial would cost `O(n³)` but lose two orders of
//! magnitude in accuracy. As for permanents, the terms are summed in fixed
//! chunks, in order, so results do not depend on the number of threads.

use crate::C64;
use crate::error::{QsoError, Result};
use crate::linalg::CMatrix;
use crate::parallel;

/// Largest dimension accepted by [`hafnian`], [`loop_hafnian`] and
/// [`torontonian`] (2²⁰ terms of `O(40³)` operations).
pub const MAX_HAFNIAN_DIM: usize = 40;

const ZERO: C64 = C64::new(0.0, 0.0);
const ONE: C64 = C64::new(1.0, 0.0);

fn check(name: &'static str, a: &CMatrix, symmetric: bool) -> Result<usize> {
    let (r, c) = a.shape();
    if r != c {
        return Err(QsoError::invalid(
            name,
            format!("a square matrix is needed, got {r} × {c}"),
        ));
    }
    if r > MAX_HAFNIAN_DIM {
        return Err(QsoError::invalid(
            name,
            format!("dimension {r} exceeds the supported maximum {MAX_HAFNIAN_DIM}"),
        ));
    }
    if a.iter().any(|z| !z.is_finite()) {
        return Err(QsoError::invalid(name, "entries must be finite"));
    }
    if symmetric {
        let scale = a.iter().map(|z| z.norm()).fold(0.0, f64::max);
        for i in 0..r {
            for j in 0..i {
                if (a[(i, j)] - a[(j, i)]).norm() > 1e-12 * scale.max(1.0) {
                    return Err(QsoError::invalid(
                        name,
                        format!("the matrix is not symmetric (entries ({i}, {j}) and ({j}, {i}))"),
                    ));
                }
            }
        }
    }
    Ok(r)
}

/// Hafnian of a symmetric complex matrix: `Σ` over perfect matchings of
/// the products of the matched entries (the diagonal is ignored). Odd
/// dimensions give 0, the 0 × 0 matrix 1.
pub fn hafnian(a: &CMatrix) -> Result<C64> {
    let n = check("matrix", a, true)?;
    if n % 2 == 1 {
        return Ok(ZERO);
    }
    Ok(subset_sum(a, false))
}

/// Loop hafnian of a symmetric complex matrix: `Σ` over matchings with
/// loops, a loop at `i` weighted by `a_ii`. The 0 × 0 matrix gives 1.
pub fn loop_hafnian(a: &CMatrix) -> Result<C64> {
    let n = check("matrix", a, true)?;
    if n % 2 == 1 {
        // A further vertex with a loop of weight 1 and no edges leaves the
        // loop hafnian unchanged and makes the dimension even.
        let mut padded = CMatrix::zeros(n + 1, n + 1);
        padded.view_mut((0, 0), (n, n)).copy_from(a);
        padded[(n, n)] = ONE;
        return Ok(subset_sum(&padded, true));
    }
    Ok(subset_sum(a, true))
}

/// `Σ_{Z ⊆ [n]} (−1)^{n−|Z|} f(Z)` over the pairs `{2i, 2i+1}`.
fn subset_sum(a: &CMatrix, loops: bool) -> C64 {
    let n = a.nrows() / 2;
    if n == 0 {
        return ONE;
    }
    let terms = 1u64 << n;
    parallel::sum_ranges(terms, |start, end| {
        let mut acc = ZERO;
        for z in start..end {
            if z == 0 {
                continue;
            }
            let f = term(a, z, n, loops);
            if (n - z.count_ones() as usize) % 2 == 1 {
                acc -= f;
            } else {
                acc += f;
            }
        }
        acc
    })
}

/// `[λⁿ] exp(Σ_k (Tr(C^k)/(2k) + ½ yᵀC^{k−1}x) λ^k)` for the pairs in `z`.
fn term(a: &CMatrix, z: u64, n: usize, loops: bool) -> C64 {
    let idx: Vec<usize> = (0..n)
        .filter(|&i| (z >> i) & 1 == 1)
        .flat_map(|i| [2 * i, 2 * i + 1])
        .collect();
    let m = idx.len();
    // C = X·B: row 2i of C is row 2i+1 of B and vice versa.
    let c = CMatrix::from_fn(m, m, |r, col| {
        if loops && idx[r ^ 1] == idx[col] {
            ZERO
        } else {
            a[(idx[r ^ 1], idx[col])]
        }
    });
    let traces = power_traces(&c, n);
    let mut e = vec![ZERO; n + 1];
    for k in 1..=n {
        e[k] = traces[k] / (2.0 * k as f64);
    }
    if loops {
        let y: Vec<C64> = idx.iter().map(|&i| a[(i, i)]).collect();
        let mut v: Vec<C64> = (0..m).map(|r| y[r ^ 1]).collect();
        for ek in e.iter_mut().skip(1) {
            let dot: C64 = y.iter().zip(&v).map(|(p, q)| p * q).sum();
            *ek += dot * 0.5;
            v = (0..m)
                .map(|r| (0..m).map(|s| c[(r, s)] * v[s]).sum())
                .collect();
        }
    }
    exp_coefficient(&e, n)
}

/// `[λⁿ] exp(Σ_{k≥1} e_k λ^k)`.
fn exp_coefficient(e: &[C64], n: usize) -> C64 {
    let mut g = vec![ZERO; n + 1];
    g[0] = ONE;
    for m in 1..=n {
        let s: C64 = (1..=m).map(|k| e[k] * (k as f64) * g[m - k]).sum();
        g[m] = s / m as f64;
    }
    g[n]
}

/// `Tr(C^k)` for `k = 0..=kmax`: the powers `C^j` up to `j = ⌈kmax/2⌉` by
/// multiplication, and `Tr(C^k) = Σ_{ab} (C^i)_{ab} (C^{k−i})_{ba}`. Traces
/// from the characteristic polynomial (Newton's identities) or from
/// eigenvalues cost a factor `kmax/2` less but lose two orders of
/// magnitude in accuracy on random 18 × 18 matrices.
fn power_traces(c: &CMatrix, kmax: usize) -> Vec<C64> {
    let m = c.nrows();
    let half = kmax.div_ceil(2);
    let mut powers: Vec<CMatrix> = Vec::with_capacity(half + 1);
    powers.push(CMatrix::identity(m, m));
    for j in 1..=half {
        let next = &powers[j - 1] * c;
        powers.push(next);
    }
    (0..=kmax)
        .map(|k| {
            let i = k / 2;
            let (p, q) = (&powers[i], &powers[k - i]);
            let mut t = ZERO;
            for a in 0..m {
                for b in 0..m {
                    t += p[(a, b)] * q[(b, a)];
                }
            }
            t
        })
        .collect()
}

/// Most terms `Π (r_i + 1)` of [`hafnian_repeated`] and
/// [`loop_hafnian_repeated`].
pub const MAX_REPEATED_TERMS: u64 = 1 << 32;

/// Hafnian of the matrix that repeats row and column `i` of the symmetric
/// `a` `rpt[i]` times (Kan 2008):
///
/// ```text
/// haf = Σ_{0 ≤ ν ≤ rpt} (−1)^{Σν} Π C(rpt_i, ν_i) (½ hᵀAh)^{N/2} / (N/2)!,   h = rpt/2 − ν
/// ```
///
/// `Π (rpt_i + 1)` terms instead of `2^{N/2}`, and stable where many rows
/// repeat (many photons in few modes).
pub fn hafnian_repeated(a: &CMatrix, rpt: &[usize]) -> Result<C64> {
    repeated(a, None, rpt)
}

/// Loop hafnian of the repeated matrix of [`hafnian_repeated`] whose
/// diagonal is replaced by `diag` (repeated likewise).
pub fn loop_hafnian_repeated(a: &CMatrix, diag: &[C64], rpt: &[usize]) -> Result<C64> {
    repeated(a, Some(diag), rpt)
}

fn repeated(a: &CMatrix, diag: Option<&[C64]>, rpt: &[usize]) -> Result<C64> {
    let m = a.nrows();
    if a.ncols() != m || rpt.len() != m || diag.is_some_and(|d| d.len() != m) {
        return Err(QsoError::DimensionMismatch {
            expected: m,
            actual: rpt.len(),
        });
    }
    if a.iter().chain(diag.unwrap_or(&[])).any(|z| !z.is_finite()) {
        return Err(QsoError::invalid("matrix", "entries must be finite"));
    }
    let scale = a.iter().map(|z| z.norm()).fold(0.0, f64::max).max(1.0);
    for i in 0..m {
        for j in 0..i {
            if (a[(i, j)] - a[(j, i)]).norm() > 1e-12 * scale {
                return Err(QsoError::invalid("matrix", "the matrix is not symmetric"));
            }
        }
    }
    // Only the repeated indices matter.
    let keep: Vec<usize> = (0..m).filter(|&i| rpt[i] > 0).collect();
    let r: Vec<usize> = keep.iter().map(|&i| rpt[i]).collect();
    let total: usize = r.iter().sum();
    if total > 2 * MAX_HAFNIAN_DIM {
        return Err(QsoError::invalid(
            "rpt",
            format!("at most {} rows in total", 2 * MAX_HAFNIAN_DIM),
        ));
    }
    let terms = r
        .iter()
        .try_fold(1u64, |t, &x| t.checked_mul(x as u64 + 1))
        .filter(|&t| t <= MAX_REPEATED_TERMS)
        .ok_or_else(|| {
            QsoError::invalid(
                "rpt",
                format!("more than {MAX_REPEATED_TERMS} terms Π (rpt_i + 1)"),
            )
        })?;
    if diag.is_none() && total % 2 == 1 {
        return Ok(ZERO);
    }
    let k = keep.len();
    let sub = CMatrix::from_fn(k, k, |i, j| a[(keep[i], keep[j])]);
    let mu: Option<Vec<C64>> = diag.map(|d| keep.iter().map(|&i| d[i]).collect());
    let half = total / 2;
    let fact = |n: usize| (1..=n).map(|i| i as f64).product::<f64>();
    let binom = |n: usize, j: usize| fact(n) / (fact(j) * fact(n - j));
    Ok(parallel::sum_ranges(terms, |start, end| {
        let mut acc = ZERO;
        let mut h = vec![0.0; k];
        for t in start..end {
            // Mixed-radix digits of t are ν.
            let mut rest = t;
            let mut coef = 1.0;
            let mut odd = false;
            for (i, &ri) in r.iter().enumerate() {
                let nu = (rest % (ri as u64 + 1)) as usize;
                rest /= ri as u64 + 1;
                h[i] = ri as f64 / 2.0 - nu as f64;
                coef *= binom(ri, nu);
                odd ^= nu % 2 == 1;
            }
            let q: C64 = (0..k)
                .map(|i| {
                    let row: C64 = (0..k).map(|j| sub[(i, j)] * h[j]).sum();
                    row * h[i]
                })
                .sum::<C64>()
                * 0.5;
            let value = match &mu {
                None => q.powu(half as u32) / fact(half),
                Some(mu) => {
                    let l: C64 = mu.iter().zip(&h).map(|(m, hi)| m * hi).sum();
                    (0..=half)
                        .map(|j| {
                            q.powu(j as u32) * l.powu((total - 2 * j) as u32)
                                / (fact(j) * fact(total - 2 * j))
                        })
                        .sum()
                }
            };
            if odd {
                acc -= value * coef;
            } else {
                acc += value * coef;
            }
        }
        acc
    }))
}

/// Torontonian of a `2N × 2N` matrix `O` in the ordering `(a₁…a_N,
/// a₁†…a_N†)`: `Σ_{Z ⊆ [N]} (−1)^{N−|Z|} / √det(I − O_Z)`, `O_Z` the rows
/// and columns `{i, i + N : i ∈ Z}`. Threshold-detector click
/// probabilities of Gaussian states are torontonians (Quesada, Arrazola
/// and Killoran 2018). The 0 × 0 matrix gives 1.
pub fn torontonian(o: &CMatrix) -> Result<C64> {
    let dim = check("matrix", o, false)?;
    if dim % 2 == 1 {
        return Err(QsoError::invalid(
            "matrix",
            "a torontonian needs an even dimension 2N",
        ));
    }
    let n = dim / 2;
    if n == 0 {
        return Ok(ONE);
    }
    Ok(parallel::sum_ranges(1u64 << n, |start, end| {
        let mut acc = ZERO;
        for z in start..end {
            let modes: Vec<usize> = (0..n).filter(|&i| (z >> i) & 1 == 1).collect();
            let k = modes.len();
            let idx: Vec<usize> = modes
                .iter()
                .copied()
                .chain(modes.iter().map(|&i| i + n))
                .collect();
            let m = CMatrix::from_fn(2 * k, 2 * k, |r, c| {
                let v = -o[(idx[r], idx[c])];
                if r == c { v + ONE } else { v }
            });
            let f = if k == 0 {
                ONE
            } else {
                m.determinant().sqrt().inv()
            };
            if (n - k) % 2 == 1 {
                acc -= f;
            } else {
                acc += f;
            }
        }
        acc
    }))
}

#[cfg(test)]
mod tests;
