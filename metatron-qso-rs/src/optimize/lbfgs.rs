//! L-BFGS with a strong-Wolfe line search (Nocedal & Wright, Numerical
//! Optimization, 2nd ed., Algorithms 7.4, 3.5 and 3.6).

use std::collections::VecDeque;

use super::{Counted, OptimizeResult, Options, inf_norm};

const C1: f64 = 1e-4;
const C2: f64 = 0.9;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

struct Point {
    alpha: f64,
    f: f64,
    g: Vec<f64>,
    d: f64,
}

/// Minimiser of the cubic interpolating `(a, fa, da)` and `(b, fb, db)`,
/// safeguarded to stay well inside the interval.
fn cubic_min(a: f64, fa: f64, da: f64, b: f64, fb: f64, db: f64) -> f64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    let d1 = da + db - 3.0 * (fa - fb) / (a - b);
    let disc = d1 * d1 - da * db;
    let mid = 0.5 * (a + b);
    if disc < 0.0 || !disc.is_finite() {
        return mid;
    }
    let d2 = (b - a).signum() * disc.sqrt();
    let t = b - (b - a) * (db + d2 - d1) / (db - da + 2.0 * d2);
    let margin = 0.1 * (hi - lo);
    if !t.is_finite() || t < lo + margin || t > hi - margin {
        mid
    } else {
        t
    }
}

/// With noise, gradients below this many noise units of the run's first
/// gradient are rounding too (the gradient's rounding grows with its
/// scale: QAOA at depth 5–6 otherwise wandered at its noise for four times
/// the evaluations; 10 was not enough).
const START_GRADIENT_NOISE: f64 = 30.0;

/// Hager and Zhang's δ for the approximate Wolfe conditions.
const APPROX_DELTA: f64 = 0.1;

/// Returns a point satisfying the strong Wolfe conditions, or the best
/// sufficient-decrease point found, or `None`.
///
/// For an objective with rounding noise (`obj.noise > 0`), values within
/// the noise of each other are treated as equal, and a point within the
/// noise of `f0` is accepted if its slope satisfies Hager and Zhang's
/// approximate Wolfe condition `d(α) ≤ (2δ − 1)·d0` (SIAM J. Optim. 16,
/// 2005): near a minimum, or on a plateau, decreases fall below the noise
/// of the values while the gradient still shows the way.
fn line_search(
    obj: &Counted<'_>,
    x: &[f64],
    p: &[f64],
    f0: f64,
    d0: f64,
    alpha0: f64,
) -> Option<Point> {
    let resolution = obj.noise * f0.abs().max(1.0);
    let noisy = obj.noise > 0.0;
    let eval = |alpha: f64| {
        let xa: Vec<f64> = x.iter().zip(p).map(|(xi, pi)| xi + alpha * pi).collect();
        let (f, g) = obj.value_and_gradient(&xa);
        let d = dot(&g, p);
        Point { alpha, f, g, d }
    };
    // Sufficient decrease, exactly or (with noise) approximately.
    let decreases = |pt: &Point| {
        pt.f.is_finite()
            && (pt.f <= f0 + C1 * pt.alpha * d0
                || (noisy && pt.f <= f0 + resolution && pt.d <= (2.0 * APPROX_DELTA - 1.0) * d0))
    };
    // `pt` is no better than `other`: higher beyond the noise.
    let higher = |pt: &Point, other: &Point| {
        if noisy {
            pt.f > other.f + resolution
        } else {
            pt.f >= other.f
        }
    };
    let zoom = |mut lo: Point, mut hi: Point| -> Option<Point> {
        for _ in 0..30 {
            let alpha = cubic_min(lo.alpha, lo.f, lo.d, hi.alpha, hi.f, hi.d);
            let pt = eval(alpha);
            if !decreases(&pt) || higher(&pt, &lo) {
                hi = pt;
            } else {
                if pt.d.abs() <= -C2 * d0 {
                    return Some(pt);
                }
                if pt.d * (hi.alpha - lo.alpha) >= 0.0 {
                    hi = lo;
                }
                lo = pt;
            }
            if (hi.alpha - lo.alpha).abs() < 1e-16 * lo.alpha.abs().max(1.0) {
                break;
            }
        }
        (lo.alpha > 0.0 && decreases(&lo)).then_some(lo)
    };

    let mut prev = Point {
        alpha: 0.0,
        f: f0,
        g: Vec::new(),
        d: d0,
    };
    let mut alpha = alpha0;
    for i in 0..30 {
        let pt = eval(alpha);
        if !decreases(&pt) || (i > 0 && higher(&pt, &prev)) {
            return zoom(prev, pt);
        }
        if pt.d.abs() <= -C2 * d0 {
            return Some(pt);
        }
        if pt.d >= 0.0 {
            return zoom(pt, prev);
        }
        prev = pt;
        alpha *= 2.0;
    }
    None
}

pub(super) fn run(
    obj: &Counted<'_>,
    x0: &[f64],
    memory: usize,
    options: &Options,
) -> OptimizeResult {
    let mut x = x0.to_vec();
    let (mut f, mut g) = obj.value_and_gradient(&x);
    let mut history = Vec::new();
    if options.record_history {
        history.push(f);
    }
    let mut pairs: VecDeque<(Vec<f64>, Vec<f64>, f64)> = VecDeque::with_capacity(memory);
    let mut message = String::from("maximum number of iterations reached");
    let mut converged = false;
    let mut iterations = 0;

    let noisy = obj.noise > 0.0;
    let start_floor = START_GRADIENT_NOISE * obj.noise * inf_norm(&g);
    for it in 0..options.max_iterations {
        if inf_norm(&g) <= options.g_tol.max(obj.gradient_floor(f)).max(start_floor) {
            message = "gradient norm below tolerance".into();
            converged = true;
            break;
        }
        // Two-loop recursion: p = −H·g.
        let mut q = g.clone();
        let mut alphas = Vec::with_capacity(pairs.len());
        for (s, y, rho) in pairs.iter().rev() {
            let a = rho * dot(s, &q);
            q.iter_mut().zip(y).for_each(|(qi, yi)| *qi -= a * yi);
            alphas.push(a);
        }
        if let Some((s, y, _)) = pairs.back() {
            let gamma = dot(s, y) / dot(y, y);
            q.iter_mut().for_each(|qi| *qi *= gamma);
        }
        for ((s, y, rho), a) in pairs.iter().zip(alphas.iter().rev()) {
            let b = rho * dot(y, &q);
            q.iter_mut().zip(s).for_each(|(qi, si)| *qi += (a - b) * si);
        }
        let mut p: Vec<f64> = q.iter().map(|v| -v).collect();
        let mut d0 = dot(&g, &p);
        if d0 >= 0.0 {
            // Not a descent direction (should not happen); restart with −g.
            pairs.clear();
            p = g.iter().map(|v| -v).collect();
            d0 = dot(&g, &p);
        }
        let alpha0 = if pairs.is_empty() {
            (1.0 / inf_norm(&g)).min(1.0)
        } else {
            1.0
        };
        let Some(pt) = line_search(obj, &x, &p, f, d0, alpha0) else {
            if obj.noise > 0.0 {
                // No decrease the objective resolves along this direction.
                message = "no decrease beyond the objective's precision".into();
                converged = true;
            } else {
                message = "line search failed to make progress".into();
                converged = inf_norm(&g) <= options.g_tol.sqrt();
            }
            iterations = it;
            break;
        };
        let s: Vec<f64> = p.iter().map(|pi| pt.alpha * pi).collect();
        let y: Vec<f64> = pt.g.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        x.iter_mut().zip(&s).for_each(|(xi, si)| *xi += si);
        let decrease = f - pt.f;
        f = pt.f;
        g = pt.g;
        if options.record_history {
            history.push(f);
        }
        iterations = it + 1;
        if sy > 1e-12 * dot(&y, &y).sqrt() * dot(&s, &s).sqrt() {
            if pairs.len() == memory {
                pairs.pop_front();
            }
            pairs.push_back((s, y, 1.0 / sy));
        }
        // With noise, decreases below it say nothing; the gradient decides.
        if !noisy && decrease <= options.f_tol * f.abs().max(1.0) {
            message = "relative decrease below tolerance".into();
            converged = true;
            break;
        }
    }
    if inf_norm(&g) <= options.g_tol && !converged {
        converged = true;
        message = "gradient norm below tolerance".into();
    }
    OptimizeResult {
        x,
        value: f,
        iterations,
        evaluations: 0,
        converged,
        message,
        history,
    }
}
