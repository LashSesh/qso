//! Classical optimisers for variational algorithms.
//!
//! All methods share one entry point, [`minimize`], and one result type.
//! Gradient-based methods use the objective's analytic gradient when it
//! provides one (VQE and QAOA do, via adjoint differentiation) and fall back
//! to central finite differences otherwise.
//!
//! | Method | Uses gradient | Typical use |
//! |---|---|---|
//! | [`Method::Lbfgs`] | yes | smooth noiseless objectives (default) |
//! | [`Method::Adam`] | yes | many parameters, rough landscapes |
//! | [`Method::GradientDescent`] | yes | baselines, teaching |
//! | [`Method::NelderMead`] | no | few parameters, no gradient |
//! | [`Method::Spsa`] | no | noisy (shot-based) objectives |

mod adam;
mod lbfgs;
mod nelder_mead;
mod spsa;

use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};

/// A function to minimise.
pub trait Objective: Sync {
    /// Number of variables.
    fn dim(&self) -> usize;

    /// Objective value.
    fn value(&self, x: &[f64]) -> f64;

    /// Value and gradient, if an analytic gradient is available.
    fn value_and_gradient(&self, _x: &[f64]) -> Option<(f64, Vec<f64>)> {
        None
    }
}

/// [`Objective`] from a closure (no analytic gradient).
pub struct FnObjective<F> {
    dim: usize,
    f: F,
}

impl<F: Fn(&[f64]) -> f64 + Sync> FnObjective<F> {
    pub fn new(dim: usize, f: F) -> Self {
        Self { dim, f }
    }
}

impl<F> std::fmt::Debug for FnObjective<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnObjective")
            .field("dim", &self.dim)
            .finish()
    }
}

impl<F: Fn(&[f64]) -> f64 + Sync> Objective for FnObjective<F> {
    fn dim(&self) -> usize {
        self.dim
    }
    fn value(&self, x: &[f64]) -> f64 {
        (self.f)(x)
    }
}

/// [`Objective`] from a closure returning value and gradient.
pub struct GradObjective<F> {
    dim: usize,
    f: F,
}

impl<F: Fn(&[f64]) -> (f64, Vec<f64>) + Sync> GradObjective<F> {
    pub fn new(dim: usize, f: F) -> Self {
        Self { dim, f }
    }
}

impl<F> std::fmt::Debug for GradObjective<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradObjective")
            .field("dim", &self.dim)
            .finish()
    }
}

impl<F: Fn(&[f64]) -> (f64, Vec<f64>) + Sync> Objective for GradObjective<F> {
    fn dim(&self) -> usize {
        self.dim
    }
    fn value(&self, x: &[f64]) -> f64 {
        (self.f)(x).0
    }
    fn value_and_gradient(&self, x: &[f64]) -> Option<(f64, Vec<f64>)> {
        Some((self.f)(x))
    }
}

/// Optimisation algorithm and its hyper-parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Method {
    /// Limited-memory BFGS with a strong-Wolfe line search.
    Lbfgs { memory: usize },
    /// Adam (Kingma & Ba, 2015).
    Adam {
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
    /// Gradient descent with heavy-ball momentum.
    GradientDescent { learning_rate: f64, momentum: f64 },
    /// Nelder–Mead simplex with dimension-adaptive coefficients.
    NelderMead { initial_step: f64 },
    /// Simultaneous-perturbation stochastic approximation (Spall, 1998).
    Spsa {
        a: f64,
        c: f64,
        alpha: f64,
        gamma: f64,
        stability: f64,
    },
}

impl Method {
    pub fn lbfgs() -> Self {
        Method::Lbfgs { memory: 10 }
    }

    pub fn adam(learning_rate: f64) -> Self {
        Method::Adam {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

    pub fn gradient_descent(learning_rate: f64) -> Self {
        Method::GradientDescent {
            learning_rate,
            momentum: 0.9,
        }
    }

    pub fn nelder_mead() -> Self {
        Method::NelderMead { initial_step: 0.25 }
    }

    pub fn spsa() -> Self {
        Method::Spsa {
            a: 0.2,
            c: 0.1,
            alpha: 0.602,
            gamma: 0.101,
            stability: 10.0,
        }
    }

    /// Short identifier.
    pub fn name(&self) -> &'static str {
        match self {
            Method::Lbfgs { .. } => "lbfgs",
            Method::Adam { .. } => "adam",
            Method::GradientDescent { .. } => "gradient-descent",
            Method::NelderMead { .. } => "nelder-mead",
            Method::Spsa { .. } => "spsa",
        }
    }

    /// Parse a method name with default hyper-parameters.
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().replace('_', "-").as_str() {
            "lbfgs" | "l-bfgs" => Ok(Method::lbfgs()),
            "adam" => Ok(Method::adam(0.05)),
            "gd" | "gradient-descent" => Ok(Method::gradient_descent(0.05)),
            "nelder-mead" | "neldermead" | "nm" => Ok(Method::nelder_mead()),
            "spsa" => Ok(Method::spsa()),
            other => Err(QsoError::invalid(
                "method",
                format!("unknown optimiser '{other}'"),
            )),
        }
    }
}

impl Default for Method {
    fn default() -> Self {
        Method::lbfgs()
    }
}

/// Largest number of random restarts any optimiser in QSO accepts.
pub const MAX_RESTARTS: usize = 10_000;

/// Reject a restart count above [`MAX_RESTARTS`].
pub(crate) fn check_restarts(restarts: usize) -> Result<()> {
    if restarts > MAX_RESTARTS {
        return Err(QsoError::invalid(
            "restarts",
            format!("at most {MAX_RESTARTS} restarts, got {restarts}"),
        ));
    }
    Ok(())
}

/// Stopping criteria and bookkeeping.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Options {
    pub max_iterations: usize,
    /// Stop when the (relative) decrease of the objective falls below this.
    pub f_tol: f64,
    /// Stop when the gradient's ∞-norm falls below this.
    pub g_tol: f64,
    /// Seed for stochastic methods (SPSA).
    pub seed: u64,
    /// Record the objective value after every iteration.
    pub record_history: bool,
}

/// The smallest relative decrease that single-precision energies resolve
/// (their rounding error is a few times 10⁻⁸ of the value).
pub(crate) const SINGLE_F_TOL: f64 = 1e-7;

impl Options {
    /// These options with a decrease tolerance no finer than the
    /// objective's precision resolves: below it, steps only follow
    /// rounding and the line search spends its evaluations on noise.
    pub(crate) fn at_precision(&self, precision: crate::quantum::Precision) -> Options {
        let mut o = self.clone();
        if precision != crate::quantum::Precision::Double {
            o.f_tol = o.f_tol.max(SINGLE_F_TOL);
        }
        o
    }
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            f_tol: 1e-12,
            g_tol: 1e-8,
            seed: 0,
            record_history: true,
        }
    }
}

/// Outcome of [`minimize`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimizeResult {
    pub x: Vec<f64>,
    pub value: f64,
    pub iterations: usize,
    /// Objective evaluations (each value-and-gradient call counts once).
    pub evaluations: usize,
    pub converged: bool,
    pub message: String,
    pub history: Vec<f64>,
}

/// Wraps an objective, counts evaluations and supplies gradients.
pub(crate) struct Counted<'a> {
    obj: &'a dyn Objective,
    pub(crate) evaluations: std::cell::Cell<usize>,
}

impl<'a> Counted<'a> {
    fn new(obj: &'a dyn Objective) -> Self {
        Self {
            obj,
            evaluations: std::cell::Cell::new(0),
        }
    }

    pub(crate) fn value(&self, x: &[f64]) -> f64 {
        self.evaluations.set(self.evaluations.get() + 1);
        self.obj.value(x)
    }

    /// Analytic gradient, or central differences with step `1e-6·max(1,|x|)`.
    pub(crate) fn value_and_gradient(&self, x: &[f64]) -> (f64, Vec<f64>) {
        if let Some(vg) = self.obj.value_and_gradient(x) {
            self.evaluations.set(self.evaluations.get() + 1);
            return vg;
        }
        let f = self.value(x);
        let mut g = vec![0.0; x.len()];
        let mut xp = x.to_vec();
        for i in 0..x.len() {
            let h = 1e-6 * x[i].abs().max(1.0);
            xp[i] = x[i] + h;
            let fp = self.value(&xp);
            xp[i] = x[i] - h;
            let fm = self.value(&xp);
            xp[i] = x[i];
            g[i] = (fp - fm) / (2.0 * h);
        }
        (f, g)
    }
}

pub(crate) fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

/// Minimise `objective` starting from `x0`.
///
/// ```
/// use metatron_qso::optimize::{minimize, GradObjective, Method, Options};
///
/// // Rosenbrock function with analytic gradient.
/// let rosen = GradObjective::new(2, |x: &[f64]| {
///     let (a, b) = (1.0 - x[0], x[1] - x[0] * x[0]);
///     (a * a + 100.0 * b * b, vec![-2.0 * a - 400.0 * x[0] * b, 200.0 * b])
/// });
/// let r = minimize(&rosen, &[-1.2, 1.0], &Method::lbfgs(), &Options::default()).unwrap();
/// assert!((r.x[0] - 1.0).abs() < 1e-6 && (r.x[1] - 1.0).abs() < 1e-6);
/// ```
pub fn minimize(
    objective: &dyn Objective,
    x0: &[f64],
    method: &Method,
    options: &Options,
) -> Result<OptimizeResult> {
    if x0.len() != objective.dim() {
        return Err(QsoError::DimensionMismatch {
            expected: objective.dim(),
            actual: x0.len(),
        });
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(QsoError::invalid("x0", "initial point must be finite"));
    }
    let counted = Counted::new(objective);
    let mut result = match *method {
        Method::Lbfgs { memory } => lbfgs::run(&counted, x0, memory.max(1), options),
        Method::Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
        } => adam::run(
            &counted,
            x0,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            options,
            false,
        ),
        Method::GradientDescent {
            learning_rate,
            momentum,
        } => adam::run(
            &counted,
            x0,
            learning_rate,
            momentum,
            0.0,
            0.0,
            options,
            true,
        ),
        Method::NelderMead { initial_step } => {
            nelder_mead::run(&counted, x0, initial_step, options)
        }
        Method::Spsa {
            a,
            c,
            alpha,
            gamma,
            stability,
        } => spsa::run(&counted, x0, a, c, alpha, gamma, stability, options),
    };
    result.evaluations = counted.evaluations.get();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rosenbrock() -> GradObjective<impl Fn(&[f64]) -> (f64, Vec<f64>) + Sync> {
        GradObjective::new(2, |x: &[f64]| {
            let (a, b) = (1.0 - x[0], x[1] - x[0] * x[0]);
            (
                a * a + 100.0 * b * b,
                vec![-2.0 * a - 400.0 * x[0] * b, 200.0 * b],
            )
        })
    }

    fn quadratic(dim: usize) -> GradObjective<impl Fn(&[f64]) -> (f64, Vec<f64>) + Sync> {
        // f = Σ (i+1)(x_i − 1)²
        GradObjective::new(dim, |x: &[f64]| {
            let f = x
                .iter()
                .enumerate()
                .map(|(i, v)| (i + 1) as f64 * (v - 1.0).powi(2))
                .sum();
            let g = x
                .iter()
                .enumerate()
                .map(|(i, v)| 2.0 * (i + 1) as f64 * (v - 1.0))
                .collect();
            (f, g)
        })
    }

    #[test]
    fn lbfgs_solves_rosenbrock() {
        let r = minimize(
            &rosenbrock(),
            &[-1.2, 1.0],
            &Method::lbfgs(),
            &Options::default(),
        )
        .unwrap();
        assert!(r.converged, "{}", r.message);
        assert!(
            (r.x[0] - 1.0).abs() < 1e-7 && (r.x[1] - 1.0).abs() < 1e-7,
            "{:?}",
            r.x
        );
        assert!(r.iterations < 100);
        // Monotone decrease is guaranteed by the line search.
        assert!(r.history.windows(2).all(|w| w[1] <= w[0] + 1e-15));
    }

    #[test]
    fn lbfgs_without_analytic_gradient_uses_finite_differences() {
        let f = FnObjective::new(3, |x: &[f64]| x.iter().map(|v| (v - 2.0).powi(2)).sum());
        let r = minimize(&f, &[0.0, 0.0, 0.0], &Method::lbfgs(), &Options::default()).unwrap();
        assert!(r.x.iter().all(|v| (v - 2.0).abs() < 1e-6));
    }

    #[test]
    fn nelder_mead_solves_rosenbrock() {
        let opts = Options {
            max_iterations: 5000,
            ..Default::default()
        };
        let r = minimize(&rosenbrock(), &[-1.2, 1.0], &Method::nelder_mead(), &opts).unwrap();
        assert!(
            (r.x[0] - 1.0).abs() < 1e-4 && (r.x[1] - 1.0).abs() < 1e-4,
            "{:?}",
            r.x
        );
        assert!(r.converged);
    }

    #[test]
    fn adam_and_momentum_minimise_quadratic() {
        let opts = Options {
            max_iterations: 3000,
            ..Default::default()
        };
        for method in [Method::adam(0.05), Method::gradient_descent(0.02)] {
            let r = minimize(&quadratic(5), &[0.0; 5], &method, &opts).unwrap();
            assert!(
                r.x.iter().all(|v| (v - 1.0).abs() < 1e-4),
                "{} {:?}",
                method.name(),
                r.x
            );
        }
    }

    #[test]
    fn spsa_handles_noisy_objective() {
        // Deterministic pseudo-noise keyed on the input keeps the test stable.
        let noisy = FnObjective::new(4, |x: &[f64]| {
            let base: f64 = x.iter().map(|v| (v - 0.5).powi(2)).sum();
            let h = x
                .iter()
                .fold(0u64, |acc, v| acc.rotate_left(7) ^ v.to_bits());
            base + 1e-3 * ((h % 1000) as f64 / 1000.0 - 0.5)
        });
        let opts = Options {
            max_iterations: 2000,
            seed: 7,
            ..Default::default()
        };
        let r = minimize(&noisy, &[2.0, -1.0, 0.0, 1.5], &Method::spsa(), &opts).unwrap();
        assert!(r.x.iter().all(|v| (v - 0.5).abs() < 0.05), "{:?}", r.x);
        // Same seed, same trajectory.
        let r2 = minimize(&noisy, &[2.0, -1.0, 0.0, 1.5], &Method::spsa(), &opts).unwrap();
        assert_eq!(r.x, r2.x);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(minimize(&quadratic(2), &[0.0], &Method::lbfgs(), &Options::default()).is_err());
        assert!(
            minimize(
                &quadratic(1),
                &[f64::NAN],
                &Method::lbfgs(),
                &Options::default()
            )
            .is_err()
        );
        assert!(Method::from_name("cobyla").is_err());
        assert_eq!(Method::from_name("L_BFGS").unwrap(), Method::lbfgs());
    }
}
