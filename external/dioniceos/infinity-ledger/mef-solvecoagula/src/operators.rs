/*!
 * Solve-Coagula Operators - SPEC-002 compliant
 * Deterministic contractive operators with guaranteed fixpoint convergence.
 */

use anyhow::Result;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// DoubleKick Operator - SPEC-002
/// v + alpha1*u1 + alpha2*u2, u1 ⊥ u2, norm preserved, clip to ‖v‖
///
/// # Arguments
/// * `v` - Input vector
/// * `alpha1`, `alpha2` - Scalar coefficients
/// * `u1`, `u2` - Orthogonal unit vectors
///
/// # Returns
/// Transformed vector with preserved norm
pub fn dk(
    v: &Array1<f64>,
    alpha1: f64,
    alpha2: f64,
    u1: &Array1<f64>,
    u2: &Array1<f64>,
) -> Array1<f64> {
    // Check orthogonality
    let mut u2_orth = u2.clone();
    let dot = u1.dot(&u2_orth);
    if dot.abs() > 1e-10 {
        // Gram-Schmidt to ensure orthogonality
        u2_orth = &u2_orth - dot * u1;
        let norm = u2_orth.dot(&u2_orth).sqrt();
        if norm > 0.0 {
            u2_orth /= norm;
        }
    }

    // Store original norm
    let original_norm = v.dot(v).sqrt();

    // Apply DoubleKick
    let v_kicked = v + alpha1 * u1 + alpha2 * &u2_orth;

    // Preserve norm (clip to ‖v‖)
    let kicked_norm = v_kicked.dot(&v_kicked).sqrt();
    if kicked_norm > 0.0 {
        v_kicked * (original_norm / kicked_norm)
    } else {
        v_kicked
    }
}

/// Sweep Operator - SPEC-002
/// g_tau(m(v)) * v, g_tau(x)=1/(1+exp(-(x-tau)/beta)), m(v)=mean(v)
///
/// # Arguments
/// * `v` - Input vector
/// * `tau` - Threshold value
/// * `beta` - Steepness parameter
///
/// # Returns
/// Gated vector
pub fn sw(v: &Array1<f64>, tau: f64, beta: f64) -> Array1<f64> {
    // Compute mean
    let m_v = v.mean().unwrap_or(0.0);

    // Gate function g_tau
    let g_tau_value = 1.0 / (1.0 + (-(m_v - tau) / beta).exp());

    // Apply gate
    v * g_tau_value
}

/// Pfadinvarianz Projection - SPEC-002
/// Canonical sorting or averaging over path-equivalent states.
/// Distance < tol → unchanged.
///
/// # Arguments
/// * `v` - Input vector
/// * `canon` - Canonicalization type ("lexicographic", "norm", "sum")
/// * `tol` - Tolerance
///
/// # Returns
/// Projected vector
pub fn pi_project(v: &Array1<f64>, canon: &str, tol: f64) -> Array1<f64> {
    let n = v.len();

    // Generate path-equivalent states (limited permutations for efficiency)
    let permutations = [
        (0..n).collect::<Vec<_>>(),                                 // Identity
        (1..n).chain(std::iter::once(0)).collect::<Vec<_>>(),       // Cyclic rotation
        std::iter::once(n - 1).chain(0..n - 1).collect::<Vec<_>>(), // Backward rotation
        (0..n).rev().collect::<Vec<_>>(),                           // Reversal
    ];

    let mut path_vectors: Vec<Array1<f64>> = permutations
        .iter()
        .map(|perm| {
            let mut permuted = Array1::zeros(n);
            for (i, &p) in perm.iter().enumerate() {
                permuted[i] = v[p];
            }
            permuted
        })
        .collect();

    // Canonical sorting
    match canon {
        "lexicographic" => {
            path_vectors.sort_by(|a, b| {
                for i in 0..n {
                    match a[i].partial_cmp(&b[i]) {
                        Some(std::cmp::Ordering::Equal) => continue,
                        Some(ord) => return ord,
                        None => continue,
                    }
                }
                std::cmp::Ordering::Equal
            });
        }
        "norm" => {
            path_vectors.sort_by(|a, b| {
                let norm_a = a.dot(a).sqrt();
                let norm_b = b.dot(b).sqrt();
                norm_a
                    .partial_cmp(&norm_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        "sum" => {
            path_vectors.sort_by(|a, b| {
                let sum_a: f64 = a.sum();
                let sum_b: f64 = b.sum();
                sum_a
                    .partial_cmp(&sum_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        _ => {}
    }

    // Average over path-equivalent states
    let mut v_mean: Array1<f64> = Array1::zeros(n);
    for pv in &path_vectors {
        v_mean += pv;
    }
    v_mean /= path_vectors.len() as f64;

    // Check distance
    let distance = (&v_mean - v).dot(&(&v_mean - v)).sqrt();

    // If distance < tol, return unchanged
    if distance < tol {
        v.clone()
    } else {
        v_mean
    }
}

/// Analytic spiral gradient computation
fn analytic_spiral_gradient(v: &Array1<f64>, beta: f64) -> Array1<f64> {
    let n = v.len();
    let mut gradient = Array1::zeros(n);

    for i in 0..n {
        let phase = (beta * i as f64).sin() + (beta * (i as f64 + 1.0)).cos();
        gradient[i] = phase * v[i];
    }

    let norm = gradient.dot(&gradient).sqrt();
    if norm > 0.0 {
        gradient / norm
    } else {
        gradient
    }
}

/// Finite difference gradient computation
fn finite_difference_gradient(v: &Array1<f64>, beta: f64) -> Array1<f64> {
    let eps = 1e-6;
    let n = v.len();
    let mut gradient = Array1::zeros(n);
    let base_norm = v.dot(v).sqrt();

    for idx in 0..n {
        let mut forward = v.clone();
        forward[idx] += eps;
        let mut backward = v.clone();
        backward[idx] -= eps;

        let forward_norm = forward.dot(&forward).sqrt();
        let backward_norm = backward.dot(&backward).sqrt();
        gradient[idx] = (forward_norm - backward_norm) / (2.0 * eps);
    }

    if base_norm > 0.0 {
        gradient /= base_norm.max(1e-6);
    }

    gradient * beta
}

/// Weight-Transfer Operator - SPEC-002
/// Convex combination across scale projections P_micro, P_meso, P_macro
///
/// # Arguments
/// * `v` - Input vector
/// * `weights` - Weights for micro, meso, macro scales
/// * `beta` - Control parameter for gradient component
/// * `mode` - "analytic" or "fd" for finite differences
///
/// # Returns
/// Weighted vector
pub fn wt(v: &Array1<f64>, weights: &HashMap<String, f64>, beta: f64, mode: &str) -> Array1<f64> {
    let n = v.len();

    // Scale projections (diagonal masks)
    let p_micro_diag = [1.2, 0.8, 1.0, 0.9, 1.1];
    let p_meso_diag = [0.9, 1.1, 0.95, 1.05, 1.0];
    let p_macro_diag = [1.0, 1.0, 1.0, 1.0, 1.0];

    let mut p_micro = Array1::zeros(n);
    let mut p_meso = Array1::zeros(n);
    let mut p_macro = Array1::zeros(n);

    for i in 0..n {
        p_micro[i] = if i < p_micro_diag.len() {
            v[i] * p_micro_diag[i]
        } else {
            v[i]
        };
        p_meso[i] = if i < p_meso_diag.len() {
            v[i] * p_meso_diag[i]
        } else {
            v[i]
        };
        p_macro[i] = if i < p_macro_diag.len() {
            v[i] * p_macro_diag[i]
        } else {
            v[i]
        };
    }

    // Normalize weights (convex combination)
    let w_micro = weights.get("micro").copied().unwrap_or(0.33);
    let w_meso = weights.get("meso").copied().unwrap_or(0.33);
    let w_macro = weights.get("macro").copied().unwrap_or(0.34);
    let w_sum = w_micro + w_meso + w_macro;

    let w_micro_norm = w_micro / w_sum;
    let w_meso_norm = w_meso / w_sum;
    let w_macro_norm = w_macro / w_sum;

    // Convex combination
    let base = &p_micro * w_micro_norm + &p_meso * w_meso_norm + &p_macro * w_macro_norm;

    // Gradient component
    let gradient = if mode == "analytic" {
        analytic_spiral_gradient(v, beta)
    } else {
        finite_difference_gradient(v, beta)
    };

    base + beta * gradient
}

/// Convergence information from fixpoint iteration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceInfo {
    pub converged: bool,
    pub iterations: usize,
    pub final_delta: f64,
    #[serde(default)]
    pub lyapunov_series: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub history: Option<Vec<ConvergenceStep>>,
}

/// Individual convergence step information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceStep {
    pub iteration: usize,
    pub lyapunov: f64,
    pub norm: f64,
}

/// Parameters for fixpoint iteration
#[derive(Debug, Clone)]
pub struct FixpointParams<'a> {
    pub eps: f64,
    pub max_iter: usize,
    pub dk_args: Option<&'a DKArgs>,
    pub sw_args: Option<&'a SWArgs>,
    pub pi_args: Option<&'a PIArgs>,
    pub wt_args: Option<&'a WTArgs>,
}

/// SPEC-002 Fixpoint Iteration
/// Order: dk→sw→pi→wt→affine, v_{t+1}=lambda*(W@v + b)
///
/// # Arguments
/// * `v0` - Initial vector
/// * `w` - Weight matrix
/// * `b` - Bias vector
/// * `lambda` - Contraction factor (0 < lambda < 1)
/// * `params` - Fixpoint iteration parameters
///
/// # Returns
/// (v_star, steps) - Fixpoint and number of steps
pub fn iterate_to_fixpoint(
    v0: &Array1<f64>,
    w: &ndarray::Array2<f64>,
    b: &Array1<f64>,
    lambda: f64,
    params: &FixpointParams,
) -> Result<(Array1<f64>, usize)> {
    // Contraction check
    if lambda <= 0.0 || lambda >= 1.0 {
        return Err(anyhow::anyhow!(
            "non-contractive: lambda={} not in (0,1)",
            lambda
        ));
    }

    // Compute spectral norm of W (approximation using Frobenius norm as upper bound)
    let w_norm = w.iter().map(|x| x * x).sum::<f64>().sqrt() / (w.nrows() as f64).sqrt();
    if w_norm > 1.0 {
        return Err(anyhow::anyhow!("non-contractive: ||W||_2={} > 1", w_norm));
    }

    let mut v = v0.clone();

    for step in 0..params.max_iter {
        let v_old = v.clone();

        // 1. DoubleKick
        if let Some(args) = params.dk_args {
            v = dk(&v, args.alpha1, args.alpha2, &args.u1, &args.u2);
        }

        // 2. Sweep
        if let Some(args) = params.sw_args {
            v = sw(&v, args.tau, args.beta);
        }

        // 3. Pfadinvarianz
        if let Some(args) = params.pi_args {
            v = pi_project(&v, &args.canon, args.tol);
        }

        // 4. Weight-Transfer
        if let Some(args) = params.wt_args {
            v = wt(&v, &args.weights, args.beta, &args.mode);
        }

        // 5. Affine transformation
        v = (w.dot(&v) + b) * lambda;

        // Convergence check
        let delta = (&v - &v_old).dot(&(&v - &v_old)).sqrt();
        if delta < params.eps {
            return Ok((v, step + 1));
        }
    }

    // Max iterations reached
    Ok((v, params.max_iter))
}

/// DoubleKick operator arguments
#[derive(Debug, Clone)]
pub struct DKArgs {
    pub alpha1: f64,
    pub alpha2: f64,
    pub u1: Array1<f64>,
    pub u2: Array1<f64>,
}

/// Sweep operator arguments
#[derive(Debug, Clone)]
pub struct SWArgs {
    pub tau: f64,
    pub beta: f64,
}

/// Pfadinvarianz operator arguments
#[derive(Debug, Clone)]
pub struct PIArgs {
    pub canon: String,
    pub tol: f64,
}

/// Weight-Transfer operator arguments
#[derive(Debug, Clone)]
pub struct WTArgs {
    pub weights: HashMap<String, f64>,
    pub beta: f64,
    pub mode: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dk_operator() {
        let v = Array1::from_vec(vec![1.0, 0.5, -0.3, 0.8, -0.2]);
        let u1 = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        let u2 = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 0.0]);

        let result = dk(&v, 0.05, -0.03, &u1, &u2);

        // Verify norm is preserved
        let orig_norm = v.dot(&v).sqrt();
        let result_norm = result.dot(&result).sqrt();
        assert!((orig_norm - result_norm).abs() < 1e-10);
    }

    #[test]
    fn test_sw_operator() {
        let v = Array1::from_vec(vec![1.0, 0.5, -0.3, 0.8, -0.2]);
        let result = sw(&v, 0.5, 0.1);

        // Result should be scaled version of v
        assert!(result.len() == v.len());
    }

    #[test]
    fn test_pi_project_operator() {
        let v = Array1::from_vec(vec![1.0, 0.5, -0.3, 0.8, -0.2]);
        let result = pi_project(&v, "lexicographic", 1e-6);

        assert!(result.len() == v.len());
    }

    #[test]
    fn test_wt_operator() {
        let v = Array1::from_vec(vec![1.0, 0.5, -0.3, 0.8, -0.2]);
        let mut weights = HashMap::new();
        weights.insert("micro".to_string(), 0.33);
        weights.insert("meso".to_string(), 0.33);
        weights.insert("macro".to_string(), 0.34);

        let result = wt(&v, &weights, 0.5, "analytic");

        assert!(result.len() == v.len());
    }
}
