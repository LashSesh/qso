//! Matrix-free Krylov-subspace methods for Hermitian operators.
//!
//! The operator is supplied as a closure `apply(x, y)` computing `y = H·x`,
//! so the same code serves sparse graph Hamiltonians, Pauli sums on 2ⁿ
//! amplitudes and anything else that can be applied to a vector.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::{inner, norm};
use crate::error::{QsoError, Result};
use crate::rng::Rng;

/// Tuning knobs for the Krylov methods.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KrylovOptions {
    /// Maximum Krylov-subspace dimension per cycle.
    pub krylov_dim: usize,
    /// Convergence tolerance (relative residual for Lanczos, absolute error
    /// per unit time for propagation).
    pub tol: f64,
    /// Maximum number of restarts (Lanczos) or sub-steps (propagation).
    pub max_restarts: usize,
    /// Seed for the random start vector of Lanczos.
    pub seed: u64,
}

impl Default for KrylovOptions {
    fn default() -> Self {
        Self {
            krylov_dim: 40,
            tol: 1e-10,
            max_restarts: 500,
            seed: 0x5150,
        }
    }
}

/// Lowest eigenpair found by [`lanczos_ground_state`].
#[derive(Clone, Debug)]
pub struct LanczosResult {
    pub eigenvalue: f64,
    pub eigenvector: Vec<C64>,
    /// Final residual `‖Hx − λx‖`.
    pub residual: f64,
    /// Number of operator applications.
    pub matvecs: usize,
}

fn axpy(y: &mut [C64], a: C64, x: &[C64]) {
    crate::parallel::for_each_mut(y, |i, yi| *yi += a * x[i]);
}

fn scale(y: &mut [C64], a: f64) {
    crate::parallel::for_each_mut(y, |_, yi| *yi *= a);
}

/// Lanczos basis built from a normalised start vector.
struct LanczosBasis {
    vectors: Vec<Vec<C64>>,
    alphas: Vec<f64>,
    betas: Vec<f64>,
    /// `true` if an invariant subspace was found (basis is exact).
    breakdown: bool,
}

/// Build at most `m` Lanczos vectors with full re-orthogonalisation.
///
/// `stop(alphas, betas)` is consulted after each step and may end the
/// iteration early (used for on-the-fly convergence checks).
fn build_basis<F>(
    apply: &F,
    start: Vec<C64>,
    m: usize,
    matvecs: &mut usize,
    mut stop: impl FnMut(&[f64], &[f64]) -> bool,
) -> LanczosBasis
where
    F: Fn(&[C64], &mut [C64]),
{
    let dim = start.len();
    let mut vectors: Vec<Vec<C64>> = vec![start];
    let mut alphas = Vec::with_capacity(m);
    let mut betas = Vec::with_capacity(m);
    let mut w = vec![C64::new(0.0, 0.0); dim];
    for j in 0..m {
        apply(&vectors[j], &mut w);
        *matvecs += 1;
        let alpha = inner(&vectors[j], &w).re;
        alphas.push(alpha);
        // Full re-orthogonalisation, applied twice ("twice is enough").
        for _ in 0..2 {
            for v in &vectors {
                let c = inner(v, &w);
                axpy(&mut w, -c, v);
            }
        }
        let beta = norm(&w);
        // Scale of H seen so far: the tridiagonal entries bound ‖H‖ from below.
        let scale_ref = alphas
            .iter()
            .map(|a| a.abs())
            .chain(betas.iter().copied())
            .fold(1.0, f64::max);
        betas.push(beta);
        if beta <= 1e-13 * scale_ref || vectors.len() == dim {
            return LanczosBasis {
                vectors,
                alphas,
                betas,
                breakdown: true,
            };
        }
        if stop(&alphas, &betas) || j + 1 == m {
            break;
        }
        let mut next = w.clone();
        scale(&mut next, 1.0 / beta);
        vectors.push(next);
    }
    LanczosBasis {
        vectors,
        alphas,
        betas,
        breakdown: false,
    }
}

/// Eigendecomposition of the symmetric tridiagonal matrix (α on the
/// diagonal, β off-diagonal), ascending.
fn tridiagonal_eigh(alphas: &[f64], betas: &[f64]) -> (Vec<f64>, DMatrix<f64>) {
    let k = alphas.len();
    let mut t = DMatrix::<f64>::zeros(k, k);
    for i in 0..k {
        t[(i, i)] = alphas[i];
        if i + 1 < k {
            t[(i, i + 1)] = betas[i];
            t[(i + 1, i)] = betas[i];
        }
    }
    super::dense::eigh_real(&t)
}

fn combine(vectors: &[Vec<C64>], coeffs: &[C64]) -> Vec<C64> {
    let dim = vectors[0].len();
    crate::parallel::map_collect(dim, |i| {
        let mut acc = C64::new(0.0, 0.0);
        for (v, c) in vectors.iter().zip(coeffs) {
            acc += v[i] * c;
        }
        acc
    })
}

/// Lowest eigenvalue and eigenvector of a Hermitian operator of dimension
/// `dim`, by explicitly restarted Lanczos with full re-orthogonalisation.
///
/// If `start` is given it is used as the initial vector; otherwise a
/// seeded random vector is drawn (a random start has overlap with every
/// eigenspace, a structured one may not — e.g. a symmetric start vector only
/// explores its own symmetry sector).
pub fn lanczos_ground_state<F>(
    dim: usize,
    apply: F,
    start: Option<&[C64]>,
    opts: &KrylovOptions,
) -> Result<LanczosResult>
where
    F: Fn(&[C64], &mut [C64]),
{
    if dim == 0 {
        return Err(QsoError::invalid(
            "dim",
            "operator dimension must be positive",
        ));
    }
    let mut x: Vec<C64> = match start {
        Some(s) => {
            if s.len() != dim {
                return Err(QsoError::DimensionMismatch {
                    expected: dim,
                    actual: s.len(),
                });
            }
            s.to_vec()
        }
        None => {
            let mut rng = Rng::seed_from_u64(opts.seed);
            (0..dim)
                .map(|_| C64::new(rng.normal(), rng.normal()))
                .collect()
        }
    };
    let nx = norm(&x);
    if nx == 0.0 {
        return Err(QsoError::ZeroNorm);
    }
    scale(&mut x, 1.0 / nx);

    let m = opts.krylov_dim.max(2).min(dim);
    let mut matvecs = 0usize;
    let mut last_estimate = f64::INFINITY;
    for _cycle in 0..=opts.max_restarts {
        let tol = opts.tol;
        let basis = build_basis(&apply, x.clone(), m, &mut matvecs, |a, b| {
            let (vals, vecs) = tridiagonal_eigh(a, b);
            let k = a.len();
            let resid = b[k - 1] * vecs[(k - 1, 0)].abs();
            resid <= tol * vals[0].abs().max(1.0)
        });
        let k = basis.alphas.len();
        let (vals, vecs) = tridiagonal_eigh(&basis.alphas, &basis.betas[..k]);
        let y: Vec<C64> = (0..basis.vectors.len().min(k))
            .map(|i| C64::new(vecs[(i, 0)], 0.0))
            .collect();
        x = combine(&basis.vectors[..y.len()], &y);
        let nx = norm(&x);
        scale(&mut x, 1.0 / nx);
        let estimate = if basis.breakdown {
            0.0
        } else {
            basis.betas[k - 1] * vecs[(k - 1, 0)].abs()
        };
        last_estimate = estimate;
        if estimate <= opts.tol * vals[0].abs().max(1.0) {
            // Verify with an explicit residual.
            let mut hx = vec![C64::new(0.0, 0.0); dim];
            apply(&x, &mut hx);
            matvecs += 1;
            let lambda = inner(&x, &hx).re;
            axpy(&mut hx, C64::new(-lambda, 0.0), &x);
            let residual = norm(&hx);
            if residual <= 10.0 * opts.tol * lambda.abs().max(1.0) || basis.breakdown {
                return Ok(LanczosResult {
                    eigenvalue: lambda,
                    eigenvector: x,
                    residual,
                    matvecs,
                });
            }
        }
    }
    Err(QsoError::NoConvergence {
        method: "Lanczos",
        iterations: matvecs,
        residual: last_estimate,
    })
}

/// Spectral decomposition of a vector within its cyclic subspace; see
/// [`cyclic_spectrum`].
#[derive(Clone, Debug)]
pub struct CyclicSpectrum {
    /// The distinct eigenvalues `λ_k` of `H` with `Π_λ v ≠ 0`, ascending.
    pub values: Vec<f64>,
    /// Components `w_k = ⟨x_k|v⟩`, so that `v = Σ_k w_k x_k`.
    pub weights: Vec<f64>,
    /// Normalised eigenvectors `x_k = Π_{λ_k} v / ‖Π_{λ_k} v‖` as columns.
    pub vectors: DMatrix<f64>,
}

/// Spectral decomposition of a real vector `v` under a real symmetric
/// operator, restricted to the cyclic (Krylov) subspace spanned by
/// `v, Hv, H²v, …`.
///
/// That subspace is the smallest invariant subspace of `H` containing `v`;
/// its dimension is the number of distinct eigenvalues `λ` with
/// `Π_λ v ≠ 0`. For start vectors with symmetry it is far smaller than the
/// dimension of `H`: 21 for a node of the 2²⁰-node hypercube. Everything a
/// walk started at `v` can do (its limiting and time-averaged distributions
/// included) is determined by this decomposition.
///
/// Lanczos with full re-orthogonalisation builds the subspace until the
/// next Lanczos vector vanishes to rounding error; if that does not happen
/// within `max_dim` vectors the call fails with
/// [`QsoError::NoConvergence`].
pub fn cyclic_spectrum<F>(apply: F, v: &[f64], max_dim: usize) -> Result<CyclicSpectrum>
where
    F: Fn(&[C64], &mut [C64]),
{
    let dim = v.len();
    let nv = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if dim == 0 || nv == 0.0 {
        return Err(QsoError::ZeroNorm);
    }
    let start: Vec<C64> = v.iter().map(|&x| C64::new(x / nv, 0.0)).collect();
    let mut matvecs = 0;
    let basis = build_basis(&apply, start, max_dim.max(1), &mut matvecs, |_, _| false);
    let k = basis.alphas.len();
    if !basis.breakdown {
        return Err(QsoError::NoConvergence {
            method: "cyclic subspace (Lanczos)",
            iterations: k,
            residual: basis.betas[k - 1],
        });
    }
    let (values, y) = tridiagonal_eigh(&basis.alphas, &basis.betas[..k - 1]);
    let m = basis.vectors.len().min(k);
    let mut vectors = DMatrix::<f64>::zeros(dim, k);
    for c in 0..k {
        let coeffs: Vec<C64> = (0..m).map(|i| C64::new(y[(i, c)], 0.0)).collect();
        let x = combine(&basis.vectors[..m], &coeffs);
        for (r, xr) in x.iter().enumerate() {
            vectors[(r, c)] = xr.re;
        }
    }
    let weights = (0..k).map(|c| nv * y[(0, c)]).collect();
    Ok(CyclicSpectrum {
        values,
        weights,
        vectors,
    })
}

/// `exp(−iHt)·v` for a Hermitian operator, by adaptive Krylov time stepping.
///
/// The error estimate per sub-step is the standard a-posteriori bound
/// `β₀·β_m·|[exp(−iτT)e₁]_m|`; steps are halved until it falls below
/// `tol·τ/|t|·‖v‖`, so the accumulated error is bounded by ≈ `tol·‖v‖`.
pub fn expm_multiply<F>(apply: F, v: &[C64], t: f64, opts: &KrylovOptions) -> Result<Vec<C64>>
where
    F: Fn(&[C64], &mut [C64]),
{
    let dim = v.len();
    if !t.is_finite() {
        return Err(QsoError::invalid("t", "evolution time must be finite"));
    }
    let mut w = v.to_vec();
    if t == 0.0 || dim == 0 {
        return Ok(w);
    }
    let total_norm = norm(v);
    if total_norm == 0.0 {
        return Ok(w);
    }
    let m = opts.krylov_dim.max(2).min(dim);
    let sign = t.signum();
    let mut remaining = t.abs();
    let mut tau = remaining;
    let mut matvecs = 0usize;
    let mut steps = 0usize;
    while remaining > 0.0 {
        steps += 1;
        if steps > opts.max_restarts.max(1) * 50 {
            return Err(QsoError::NoConvergence {
                method: "Krylov expm",
                iterations: steps,
                residual: remaining,
            });
        }
        let beta0 = norm(&w);
        let mut start = w.clone();
        scale(&mut start, 1.0 / beta0);
        let basis = build_basis(&apply, start, m, &mut matvecs, |_, _| false);
        let k = basis.alphas.len();
        let (vals, vecs) = tridiagonal_eigh(&basis.alphas, &basis.betas[..k]);
        let n_vec = basis.vectors.len().min(k);
        tau = tau.min(remaining);
        loop {
            // c = S · diag(e^{−iθτ}) · Sᵀ e₁
            let phases: Vec<C64> = vals
                .iter()
                .map(|&th| C64::from_polar(1.0, -sign * th * tau))
                .collect();
            let s_e1 = DVector::from_fn(k, |j, _| vecs[(0, j)]);
            let c: Vec<C64> = (0..k)
                .map(|i| (0..k).map(|j| vecs[(i, j)] * phases[j] * s_e1[j]).sum())
                .collect();
            let err = if basis.breakdown {
                0.0
            } else {
                beta0 * basis.betas[k - 1] * c[k - 1].norm()
            };
            if err <= opts.tol * total_norm * tau / t.abs() || tau <= t.abs() * 1e-12 {
                let coeffs: Vec<C64> = c[..n_vec].iter().map(|z| z * beta0).collect();
                w = combine(&basis.vectors[..n_vec], &coeffs);
                remaining -= tau;
                if remaining < t.abs() * 1e-15 {
                    remaining = 0.0;
                }
                if basis.breakdown {
                    tau = remaining;
                } else {
                    tau *= 1.5;
                }
                break;
            }
            tau /= 2.0;
        }
    }
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::dense::{CMatrix, eigh, expm_hermitian};

    fn random_hermitian(n: usize, seed: u64) -> CMatrix {
        let mut rng = Rng::seed_from_u64(seed);
        let a = CMatrix::from_fn(n, n, |_, _| C64::new(rng.normal(), rng.normal()));
        (&a + a.adjoint()) * C64::new(0.5, 0.0)
    }

    fn matvec(h: &CMatrix) -> impl Fn(&[C64], &mut [C64]) + '_ {
        move |x: &[C64], y: &mut [C64]| {
            let n = h.nrows();
            for r in 0..n {
                y[r] = (0..n).map(|c| h[(r, c)] * x[c]).sum();
            }
        }
    }

    #[test]
    fn lanczos_matches_dense_ground_state() {
        for (n, seed) in [(5usize, 1u64), (60, 2), (150, 3)] {
            let h = random_hermitian(n, seed);
            let (values, _) = eigh(&h);
            let opts = KrylovOptions {
                krylov_dim: 30,
                ..Default::default()
            };
            let res = lanczos_ground_state(n, matvec(&h), None, &opts).unwrap();
            assert!((res.eigenvalue - values[0]).abs() < 1e-8, "n={n}");
            assert!(res.residual < 1e-7);
        }
    }

    #[test]
    fn krylov_propagation_matches_dense_expm() {
        let n = 80;
        let h = random_hermitian(n, 4);
        let mut rng = Rng::seed_from_u64(9);
        let v: Vec<C64> = (0..n)
            .map(|_| C64::new(rng.normal(), rng.normal()))
            .collect();
        for t in [0.1, 1.7, -2.3, 12.0] {
            let exact = expm_hermitian(&h, t) * nalgebra::DVector::from_column_slice(&v);
            let approx = expm_multiply(matvec(&h), &v, t, &KrylovOptions::default()).unwrap();
            let err: f64 = exact
                .iter()
                .zip(&approx)
                .map(|(a, b)| (a - b).norm_sqr())
                .sum::<f64>()
                .sqrt();
            assert!(err < 1e-8 * norm(&v), "t={t} err={err}");
        }
    }
}
