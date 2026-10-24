//! Simultaneous-perturbation stochastic approximation (Spall, IEEE Trans.
//! Aerosp. Electron. Syst. 34, 817–823, 1998).
//!
//! Each iteration estimates the gradient from two evaluations at
//! `x ± c_k·Δ` with a random ±1 vector `Δ`, which makes SPSA robust to
//! shot noise. Gains follow `a_k = a/(k+1+A)^α`, `c_k = c/(k+1)^γ`.

use super::{Counted, OptimizeResult, Options};
use crate::rng::Rng;

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    obj: &Counted<'_>,
    x0: &[f64],
    a: f64,
    c: f64,
    alpha: f64,
    gamma: f64,
    stability: f64,
    options: &Options,
) -> OptimizeResult {
    let n = x0.len();
    let mut rng = Rng::seed_from_u64(options.seed);
    let mut x = x0.to_vec();
    let mut history = Vec::new();
    let mut delta = vec![0.0; n];
    for k in 0..options.max_iterations {
        let ak = a / (k as f64 + 1.0 + stability).powf(alpha);
        let ck = c / (k as f64 + 1.0).powf(gamma);
        delta.iter_mut().for_each(|d| *d = rng.sign());
        let xp: Vec<f64> = x.iter().zip(&delta).map(|(xi, d)| xi + ck * d).collect();
        let xm: Vec<f64> = x.iter().zip(&delta).map(|(xi, d)| xi - ck * d).collect();
        let (fp, fm) = (obj.value(&xp), obj.value(&xm));
        let scale = (fp - fm) / (2.0 * ck);
        x.iter_mut()
            .zip(&delta)
            .for_each(|(xi, d)| *xi -= ak * scale * d);
        if options.record_history {
            history.push(0.5 * (fp + fm));
        }
    }
    let value = obj.value(&x);
    OptimizeResult {
        x,
        value,
        iterations: options.max_iterations,
        evaluations: 0,
        converged: true,
        message: "completed the configured number of SPSA iterations".into(),
        history,
    }
}
