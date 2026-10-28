//! Adam and gradient descent with momentum.

use super::{Counted, OptimizeResult, Options, inf_norm};

/// Adam, or (with `plain_momentum`) heavy-ball gradient descent where
/// `beta1` is the momentum coefficient. Returns the best point visited.
#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    obj: &Counted<'_>,
    x0: &[f64],
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    options: &Options,
    plain_momentum: bool,
) -> OptimizeResult {
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut m = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut best = (f64::INFINITY, x.clone());
    let mut history = Vec::new();
    let mut converged = false;
    let mut message = String::from("maximum number of iterations reached");
    let mut iterations = 0;
    let mut previous = f64::INFINITY;
    let mut stalled = 0usize;

    for t in 1..=options.max_iterations {
        let (f, g) = obj.value_and_gradient(&x);
        if f < best.0 {
            best = (f, x.clone());
        }
        if options.record_history {
            history.push(f);
        }
        iterations = t;
        if inf_norm(&g) <= options.g_tol.max(obj.gradient_floor(f)) {
            converged = true;
            message = "gradient norm below tolerance".into();
            break;
        }
        // Stop after 20 consecutive iterations without meaningful progress.
        if (previous - f).abs() <= options.f_tol * f.abs().max(1.0) {
            stalled += 1;
            if stalled >= 20 {
                converged = true;
                message = "objective stalled".into();
                break;
            }
        } else {
            stalled = 0;
        }
        previous = f;
        if plain_momentum {
            for i in 0..n {
                m[i] = beta1 * m[i] + g[i];
                x[i] -= learning_rate * m[i];
            }
        } else {
            let b1t = 1.0 - beta1.powi(t as i32);
            let b2t = 1.0 - beta2.powi(t as i32);
            for i in 0..n {
                m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
                v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
                x[i] -= learning_rate * (m[i] / b1t) / ((v[i] / b2t).sqrt() + epsilon);
            }
        }
    }
    let f_final = obj.value(&x);
    if f_final < best.0 {
        best = (f_final, x);
    }
    OptimizeResult {
        x: best.1,
        value: best.0,
        iterations,
        evaluations: 0,
        converged,
        message,
        history,
    }
}
