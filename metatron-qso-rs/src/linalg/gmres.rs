//! Restarted GMRES with right preconditioning (crate-internal).
//!
//! This is the iteration of 1.54's steady states (`spin/exchange/sparse.rs`),
//! moved here step for step with its constants and its error made
//! parameters, so that the exchange steady states keep their results bit for
//! bit while other solvers use the same code:
//!
//! * the iteration starts from `x = 0` and preconditions on the right
//!   (`z = M⁻¹v_k`, `w = A z`, and at the end of a cycle `x += M⁻¹V y`);
//! * Arnoldi orthogonalises by one pass of modified Gram–Schmidt and
//!   triangularises the Hessenberg matrix by complex Givens rotations;
//! * sums are sequential (no chunked reductions, whose rounding would change
//!   the last bits beyond 4096 elements), so results do not depend on the
//!   thread count.
//!
//! Two rules are new and off by default (`scale = 0`, `breakdown = 0` give
//! 1.54's iteration exactly). The backward-error rule
//! `‖b − Ax‖ ≤ tol·(‖b‖ + scale·‖x‖)` stops where an operator applied with
//! rounding errors far above `tol·‖b‖` (relative to `‖A‖‖x‖`) would keep a
//! relative-residual rule iterating forever. The relative breakdown test
//! ends a cycle once the new Arnoldi vector is rounding noise, instead of
//! normalising that noise into a basis vector; the true residual then
//! decides between convergence (a singular but consistent system) and
//! [`GmresEnd::Breakdown`].

use nalgebra::linalg::SVD;
use num_complex::Complex64 as C64;

use super::CMatrix;

const ZERO: C64 = C64::new(0.0, 0.0);

/// Largest last-cycle triangle whose condition number [`gmres`] computes
/// from its singular values; beyond it the diagonal gives a lower bound (an
/// SVD of `k × k` costs more than the cycle itself once `k` approaches `n`).
const CONDITION_BY_SVD: usize = 256;

/// Settings of [`gmres`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct GmresOptions {
    /// Relative tolerance of the stopping rule (see `scale`).
    pub tol: f64,
    /// Basis vectors before a restart (at least 1; 0 counts as 1).
    pub restart: usize,
    /// Most iterations (applications of the preconditioned operator).
    pub max_iterations: usize,
    /// Estimated operations of one iteration besides the
    /// orthogonalisation, and the most operations in all: before iteration
    /// `k` of a cycle, `per + 16(k + 1)n` are added to the operations spent,
    /// and past the limit the iteration ends with [`GmresEnd::Work`]
    /// (`f64::INFINITY`: no limit).
    pub work: (f64, f64),
    /// 0: stop once `‖b − Ax‖ ≤ tol·‖b‖` (1.54). Positive: an estimate of
    /// `‖A‖` for the backward-error rule `‖b − Ax‖ ≤ tol·(‖b‖ + scale·‖x‖)`,
    /// with `‖x‖` renewed at every restart.
    pub scale: f64,
    /// 0: a cycle ends early only when the new Arnoldi vector is exactly
    /// zero, and the iteration restarts (1.54). Positive: a cycle also ends
    /// once the new vector's norm after orthogonalisation is at most
    /// `breakdown` times its norm before; after the update the iteration
    /// ends, [`GmresEnd::Converged`] if the true residual meets the target,
    /// [`GmresEnd::Breakdown`] otherwise.
    pub breakdown: f64,
}

/// How [`gmres`] ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GmresEnd {
    /// The true residual met the target.
    Converged,
    /// `max_iterations` ran out.
    Iterations,
    /// The (preconditioned) operator is singular on the Krylov space: an
    /// exact zero on the diagonal of the triangular factor, or (with
    /// `breakdown > 0`) an invariant Krylov space without convergence.
    Breakdown,
    /// The next iteration would pass the work limit.
    Work,
}

/// What [`gmres`] returns.
#[derive(Clone, Debug)]
pub(crate) struct GmresSolution {
    /// The last iterate (`x = 0` before the first update).
    pub x: Vec<C64>,
    /// How the iteration ended.
    pub end: GmresEnd,
    /// Iterations, including one refused by the work limit.
    pub iterations: usize,
    /// Restart cycles begun.
    pub cycles: usize,
    /// `‖b − Ax‖/(‖b‖ + scale·‖x‖)` at the last true-residual check (0 for
    /// `b = 0`); its target is `tol`. After an end inside a cycle
    /// ([`GmresEnd::Work`], or an exact zero pivot) it belongs to the
    /// iterate before that cycle's update.
    pub residual: f64,
    /// Condition estimate of the preconditioned operator: `σ_max/σ_min` of
    /// the last cycle's triangular factor (1 before the first cycle). Up to
    /// 256 columns it comes from the singular values, beyond from the lower
    /// bound `max|R_ii|/min|R_ii|`; infinite for a singular factor, NaN for
    /// non-finite entries.
    pub condition: f64,
    /// Estimated operations of the iterations (by `work`), including one
    /// refused by the work limit.
    pub spent: f64,
}

/// `‖x‖₂`, summed sequentially (as in 1.54).
fn norm2(x: &[C64]) -> f64 {
    x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

/// `A x = b` by GMRES (`apply`: `y = A x`), preconditioned on the right by
/// `precondition` (`M⁻¹r`), restarted after `opts.restart` basis vectors,
/// from `x = 0`; see [`GmresOptions`] for the stopping rules. With
/// `scale = 0` and `breakdown = 0` this is the arithmetic of 1.54's steady
/// states, step for step.
pub(crate) fn gmres(
    n: usize,
    apply: impl Fn(&[C64], &mut [C64]),
    precondition: impl Fn(&[C64]) -> Vec<C64>,
    b: &[C64],
    opts: &GmresOptions,
) -> GmresSolution {
    debug_assert_eq!(b.len(), n);
    let bnorm = norm2(b);
    let mut target = opts.tol * bnorm;
    let mut x = vec![ZERO; n];
    if target == 0.0 {
        return GmresSolution {
            x,
            end: GmresEnd::Converged,
            iterations: 0,
            cycles: 0,
            residual: 0.0,
            condition: 1.0,
            spent: 0.0,
        };
    }
    let restart = opts.restart.max(1);
    let (per, limit) = opts.work;
    let mut spent = 0.0;
    let mut iterations = 0;
    let mut cycles = 0;
    let mut residual: f64;
    let mut w = vec![ZERO; n];
    // The rotated Hessenberg columns of the current (finally the last)
    // cycle: column j holds R[0..=j, j] and a zeroed subdiagonal entry.
    let mut h: Vec<Vec<C64>> = Vec::new();
    // A relative breakdown ended the last cycle.
    let mut broke = false;
    let end = 'outer: loop {
        apply(&x, &mut w);
        let r: Vec<C64> = b.iter().zip(&w).map(|(b, ax)| b - ax).collect();
        let beta = norm2(&r);
        let mut size = bnorm;
        if opts.scale > 0.0 {
            size = bnorm + opts.scale * norm2(&x);
            target = opts.tol * size;
        }
        residual = beta / size;
        if beta <= target {
            break GmresEnd::Converged;
        }
        if broke {
            break GmresEnd::Breakdown;
        }
        if iterations >= opts.max_iterations {
            break GmresEnd::Iterations;
        }
        cycles += 1;
        let mut basis: Vec<Vec<C64>> = vec![r.iter().map(|v| v / beta).collect()];
        h.clear();
        let mut rotations: Vec<(f64, C64)> = Vec::new();
        let mut g = vec![ZERO; restart + 1];
        g[0] = C64::new(beta, 0.0);
        let mut k = 0;
        while k < restart && iterations < opts.max_iterations {
            iterations += 1;
            spent += per + 16.0 * ((k + 1) * n) as f64;
            if spent > limit {
                break 'outer GmresEnd::Work;
            }
            let z = precondition(&basis[k]);
            apply(&z, &mut w);
            let before = if opts.breakdown > 0.0 { norm2(&w) } else { 0.0 };
            let mut column = vec![ZERO; k + 2];
            for (i, v) in basis.iter().enumerate() {
                let hij: C64 = v.iter().zip(&w).map(|(a, b)| a.conj() * b).sum();
                column[i] = hij;
                for (x, b) in w.iter_mut().zip(v) {
                    *x -= hij * b;
                }
            }
            let s = norm2(&w);
            column[k + 1] = C64::new(s, 0.0);
            for (i, &(c, sn)) in rotations.iter().enumerate() {
                let (p, q) = (column[i], column[i + 1]);
                column[i] = p * c + sn * q;
                column[i + 1] = -sn.conj() * p + q * c;
            }
            let (p, q) = (column[k], column[k + 1]);
            // A relative breakdown: the new vector lies in the Krylov space
            // to rounding. If its diagonal entry vanishes too, the operator
            // is singular on that space (an inconsistent system, say): the
            // direction cannot lower the residual and dividing by that
            // entry would only blow up rounding, so the update leaves it
            // out (its column still enters the condition estimate).
            let broken = opts.breakdown > 0.0 && s <= opts.breakdown * before;
            let singular = broken && p.norm() <= opts.breakdown * before;
            let r = (p.norm_sqr() + q.norm_sqr()).sqrt();
            let (c, sn) = if r == 0.0 {
                (1.0, ZERO)
            } else if p.norm() == 0.0 {
                (0.0, C64::new(1.0, 0.0))
            } else {
                (p.norm() / r, (p / p.norm()) * q.conj() / r)
            };
            column[k] = p * c + sn * q;
            column[k + 1] = ZERO;
            if singular {
                h.push(column);
                broke = true;
                break;
            }
            g[k + 1] = -sn.conj() * g[k];
            g[k] *= c;
            rotations.push((c, sn));
            h.push(column);
            k += 1;
            if g[k].norm() <= target {
                break;
            }
            if opts.breakdown > 0.0 {
                if broken {
                    broke = true;
                    break;
                }
            } else if s == 0.0 {
                break;
            }
            basis.push(w.iter().map(|v| v / s).collect());
        }
        // y from the triangle (its first k columns), x += M⁻¹ V y.
        let mut y = vec![ZERO; k];
        for i in (0..k).rev() {
            let mut acc = g[i];
            for j in i + 1..k {
                acc -= h[j][i] * y[j];
            }
            if h[i][i].norm() == 0.0 {
                break 'outer GmresEnd::Breakdown;
            }
            y[i] = acc / h[i][i];
        }
        let mut update = vec![ZERO; n];
        for (v, yi) in basis.iter().zip(&y) {
            for (u, b) in update.iter_mut().zip(v) {
                *u += yi * b;
            }
        }
        for (xi, d) in x.iter_mut().zip(precondition(&update)) {
            *xi += d;
        }
    };
    GmresSolution {
        x,
        end,
        iterations,
        cycles,
        residual,
        condition: triangle_condition(&h),
        spent,
    }
}

/// `σ_max/σ_min` of the `k × k` upper triangle whose column `j` is
/// `h[j][..=j]` (see [`GmresSolution::condition`]).
fn triangle_condition(h: &[Vec<C64>]) -> f64 {
    let k = h.len();
    if k == 0 {
        return 1.0;
    }
    let entry = |i: usize, j: usize| if i <= j { h[j][i] } else { ZERO };
    if (0..k).any(|j| (0..=j).any(|i| !(entry(i, j).re.is_finite() && entry(i, j).im.is_finite())))
    {
        return f64::NAN;
    }
    let ratio = |max: f64, min: f64| if min > 0.0 { max / min } else { f64::INFINITY };
    if k <= CONDITION_BY_SVD {
        let r = CMatrix::from_fn(k, k, entry);
        // nalgebra's SVD without a budget never returns on some inputs.
        if let Some(svd) = SVD::try_new_unordered(r, false, false, f64::EPSILON, 30 * k.max(10)) {
            let s = &svd.singular_values;
            return ratio(
                s.iter().copied().fold(0.0, f64::max),
                s.iter().copied().fold(f64::INFINITY, f64::min),
            );
        }
    }
    let d = (0..k).map(|i| h[i][i].norm());
    ratio(
        d.clone().fold(0.0, f64::max),
        d.fold(f64::INFINITY, f64::min),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use std::cell::Cell;

    fn options(tol: f64, restart: usize, max_iterations: usize) -> GmresOptions {
        GmresOptions {
            tol,
            restart,
            max_iterations,
            work: (1.0, f64::INFINITY),
            scale: 0.0,
            breakdown: 0.0,
        }
    }

    fn residual_norm(apply: impl Fn(&[C64], &mut [C64]), x: &[C64], b: &[C64]) -> f64 {
        let mut ax = vec![ZERO; b.len()];
        apply(x, &mut ax);
        norm2(&b.iter().zip(&ax).map(|(b, a)| b - a).collect::<Vec<_>>())
    }

    #[test]
    fn solves_a_preconditioned_nonnormal_system_and_reports() {
        // A convection-diffusion ring with a complex shift: restarts at 10
        // basis vectors, Jacobi on the right.
        let n = 300;
        let diag = |i: usize| C64::new(2.5 + 0.01 * i as f64, 0.3);
        let apply = |x: &[C64], y: &mut [C64]| {
            for i in 0..n {
                y[i] = diag(i) * x[i] - 0.7 * x[(i + 1) % n] - 1.3 * x[(i + n - 1) % n];
            }
        };
        let jacobi = |r: &[C64]| -> Vec<C64> { (0..n).map(|i| r[i] / diag(i)).collect() };
        let mut rng = Rng::seed_from_u64(3);
        let b: Vec<C64> = (0..n)
            .map(|_| C64::new(rng.normal(), rng.normal()))
            .collect();
        let out = gmres(n, apply, jacobi, &b, &options(1e-12, 10, 1000));
        assert_eq!(out.end, GmresEnd::Converged);
        assert!(out.cycles > 1, "{out:?}");
        let r = residual_norm(apply, &out.x, &b) / norm2(&b);
        assert!(
            r <= 1e-12 && (r - out.residual).abs() <= 1e-15,
            "{r:e} {:e}",
            out.residual
        );
        assert!(
            out.condition >= 1.0 && out.condition < 100.0,
            "{}",
            out.condition
        );
        // Work: per + 16(k + 1)n before iteration k of a cycle (one cycle).
        let per = 5.0;
        let mut opts = options(1e-12, n, 1000);
        opts.work = (per, f64::INFINITY);
        let single = gmres(n, apply, jacobi, &b, &opts);
        assert_eq!((single.end, single.cycles), (GmresEnd::Converged, 1));
        let k = single.iterations;
        let want = per * k as f64 + 16.0 * n as f64 * (k * (k + 1) / 2) as f64;
        assert_eq!(single.spent, want);
        // A limit that bites ends with Work before the next iteration.
        opts.work = (per, 0.5 * want);
        let cut = gmres(n, apply, jacobi, &b, &opts);
        assert_eq!(cut.end, GmresEnd::Work);
        assert!(cut.spent > 0.5 * want && cut.iterations < k);
        // b = 0 needs no operator application.
        let zero = gmres(n, |_, _| unreachable!(), jacobi, &vec![ZERO; n], &opts);
        assert_eq!(zero.end, GmresEnd::Converged);
        assert!(zero.x.iter().all(|z| *z == ZERO) && zero.iterations == 0);
        // No iterations allowed.
        let none = gmres(n, apply, jacobi, &b, &options(1e-12, 10, 0));
        assert_eq!(
            (none.end, none.iterations, none.condition),
            (GmresEnd::Iterations, 0, 1.0)
        );
    }

    #[test]
    fn the_backward_error_rule_stops_above_the_rounding_of_the_operator() {
        // A = diag(1e-3 … 1) applied with errors of 1e-14‖A‖‖x‖, as rounding
        // in an operator with cancellations would make them: ‖x‖ ≈ 1e3‖b‖,
        // so the floor 1e-11‖b‖ lies above tol·‖b‖ = 1e-13‖b‖ but far below
        // tol·(‖b‖ + ‖A‖‖x‖) ≈ 1e-10‖b‖.
        let n = 80;
        let d: Vec<f64> = (0..n)
            .map(|i| 10f64.powf(-3.0 * (i as f64) / (n - 1) as f64))
            .collect();
        let calls = Cell::new(0u64);
        let apply = |x: &[C64], y: &mut [C64]| {
            let mut rng = Rng::seed_from_u64(1000 + calls.get());
            calls.set(calls.get() + 1);
            let noise = 1e-14 * norm2(x) / (n as f64).sqrt();
            for i in 0..n {
                y[i] = d[i] * x[i] + C64::new(rng.normal(), rng.normal()) * (noise / 2f64.sqrt());
            }
        };
        let identity = |r: &[C64]| r.to_vec();
        let b: Vec<C64> = (0..n).map(|i| C64::new(1.0, (i as f64).sin())).collect();
        let mut opts = options(1e-13, n, 200);
        let plain = gmres(n, apply, identity, &b, &opts);
        assert_eq!(plain.end, GmresEnd::Iterations, "{:e}", plain.residual);
        opts.scale = 1.0;
        let scaled = gmres(n, apply, identity, &b, &opts);
        assert_eq!(scaled.end, GmresEnd::Converged);
        assert!(scaled.iterations < 200 && scaled.residual <= 1e-13);
        let exact: Vec<C64> = b.iter().zip(&d).map(|(b, d)| b / d).collect();
        let error = norm2(
            &scaled
                .x
                .iter()
                .zip(&exact)
                .map(|(x, e)| x - e)
                .collect::<Vec<_>>(),
        );
        assert!(error <= 1e-9 * norm2(&exact), "{:e}", error / norm2(&exact));
    }

    #[test]
    fn singular_systems_converge_if_consistent_and_break_down_otherwise() {
        // A = diag over eigenvalues {0, 1, 2, 3}: every Krylov space is at
        // most four-dimensional and A is singular on it when b reaches the
        // kernel.
        let n = 40;
        let d: Vec<f64> = (0..n).map(|i| (i % 4) as f64).collect();
        let apply = |x: &[C64], y: &mut [C64]| {
            for i in 0..n {
                y[i] = d[i] * x[i];
            }
        };
        let identity = |r: &[C64]| r.to_vec();
        let consistent: Vec<C64> = (0..n)
            .map(|i| {
                if d[i] == 0.0 {
                    ZERO
                } else {
                    C64::new(1.0 + 0.1 * i as f64, -0.5)
                }
            })
            .collect();
        let inconsistent: Vec<C64> = (0..n).map(|i| C64::new(1.0, 0.1 * i as f64)).collect();
        let mut opts = options(1e-13, 40, 400);
        opts.breakdown = 1e-14;
        let out = gmres(n, apply, identity, &consistent, &opts);
        assert_eq!(out.end, GmresEnd::Converged);
        assert!(residual_norm(apply, &out.x, &consistent) <= 1e-13 * norm2(&consistent));
        let out = gmres(n, apply, identity, &inconsistent, &opts);
        assert_eq!(out.end, GmresEnd::Breakdown, "{out:?}");
        assert!(
            out.iterations <= 5 && out.cycles == 1,
            "{} {}",
            out.iterations,
            out.cycles
        );
        // 1.54's rule turns the rounding noise into new basis vectors and
        // keeps iterating (or stops on an exact zero): never Converged.
        opts.breakdown = 0.0;
        let out = gmres(n, apply, identity, &inconsistent, &opts);
        assert_ne!(out.end, GmresEnd::Converged);
        // The zero operator: an exact breakdown either way.
        for breakdown in [0.0, 1e-14] {
            opts.breakdown = breakdown;
            let out = gmres(n, |_, y| y.fill(ZERO), identity, &inconsistent, &opts);
            assert_eq!(out.end, GmresEnd::Breakdown);
            assert_eq!(out.condition, f64::INFINITY);
        }
    }

    #[test]
    fn the_condition_estimate_falls_back_to_the_diagonal() {
        // R = [[1, 10], [0, 1e-3]]: σ_max/σ_min = σ_max²/|det R| ≈ 1.01e5,
        // above max|R_ii|/min|R_ii| = 1e3.
        let h = vec![
            vec![C64::new(1.0, 0.0), ZERO],
            vec![C64::new(10.0, 0.0), C64::new(1e-3, 0.0), ZERO],
        ];
        let exact = triangle_condition(&h);
        assert!((exact / 1.01e5 - 1.0).abs() < 1e-6, "{exact}");
        // A diagonal triangle of 300 columns: the bound is exact there.
        let h: Vec<Vec<C64>> = (0..300)
            .map(|j| {
                let mut c = vec![ZERO; j + 2];
                c[j] = C64::new(1.0 + j as f64, 0.0);
                c
            })
            .collect();
        assert_eq!(triangle_condition(&h), 300.0);
        let mut h = h;
        h[7][3] = C64::new(f64::NAN, 0.0);
        assert!(triangle_condition(&h).is_nan());
    }
}
