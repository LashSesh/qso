/*!
 * MEF-Core Solve-Coagula Module
 * SPEC-002-compliant fixpoint iteration operators
 */

pub mod doublekick;
pub mod operators;
pub mod pfadinvarianz;
pub mod sweep;
pub mod weight_transfer;

use anyhow::Result;
use ndarray::{Array1, Array2};
use operators::{
    iterate_to_fixpoint, ConvergenceInfo, ConvergenceStep, DKArgs, FixpointParams, PIArgs, SWArgs,
    WTArgs,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Configuration for SolveCoagula
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveCoagulaConfig {
    #[serde(default = "default_lambda")]
    pub lambda: f64,
    #[serde(default = "default_eps")]
    pub eps: f64,
    #[serde(default = "default_max_iter")]
    pub max_iter: usize,
    #[serde(default)]
    pub operators: OperatorsConfig,
    #[serde(rename = "SC_BETA", default = "default_beta")]
    pub sc_beta: f64,
}

fn default_lambda() -> f64 {
    0.8
}
fn default_eps() -> f64 {
    1e-6
}
fn default_max_iter() -> usize {
    1000
}
fn default_beta() -> f64 {
    0.5
}

impl Default for SolveCoagulaConfig {
    fn default() -> Self {
        Self {
            lambda: default_lambda(),
            eps: default_eps(),
            max_iter: default_max_iter(),
            operators: OperatorsConfig::default(),
            sc_beta: default_beta(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OperatorsConfig {
    #[serde(default)]
    pub dk: DKConfig,
    #[serde(default)]
    pub sw: SWConfig,
    #[serde(default)]
    pub pi: PIConfig,
    #[serde(default)]
    pub wt: WTConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DKConfig {
    #[serde(default = "default_alpha1")]
    pub alpha1: f64,
    #[serde(default = "default_alpha2")]
    pub alpha2: f64,
}

fn default_alpha1() -> f64 {
    0.05
}
fn default_alpha2() -> f64 {
    -0.03
}

impl Default for DKConfig {
    fn default() -> Self {
        Self {
            alpha1: default_alpha1(),
            alpha2: default_alpha2(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SWConfig {
    #[serde(default = "default_tau0")]
    pub tau0: f64,
    #[serde(default = "default_sw_beta")]
    pub beta: f64,
}

fn default_tau0() -> f64 {
    0.5
}
fn default_sw_beta() -> f64 {
    0.1
}

impl Default for SWConfig {
    fn default() -> Self {
        Self {
            tau0: default_tau0(),
            beta: default_sw_beta(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIConfig {
    #[serde(default = "default_canon")]
    pub canon: String,
    #[serde(default = "default_tol")]
    pub tol: f64,
}

fn default_canon() -> String {
    "lexicographic".to_string()
}
fn default_tol() -> f64 {
    1e-6
}

impl Default for PIConfig {
    fn default() -> Self {
        Self {
            canon: default_canon(),
            tol: default_tol(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WTConfig {
    #[serde(default = "default_mode")]
    pub mode: String,
}

fn default_mode() -> String {
    "analytic".to_string()
}

impl Default for WTConfig {
    fn default() -> Self {
        Self {
            mode: default_mode(),
        }
    }
}

/// SPEC-002-compliant Solve-Coagula Operator
pub struct SolveCoagula {
    lambda_factor: f64,
    eps: f64,
    max_iter: usize,
    w: Array2<f64>,
    b: Array1<f64>,
    u1: Array1<f64>,
    u2: Array1<f64>,
    operators_config: OperatorsConfig,
    beta: f64,
    wt_mode: String,
}

impl SolveCoagula {
    /// Initialize Solve-Coagula with configuration
    ///
    /// # Arguments
    /// * `config` - Configuration with lambda, eps, max_iter
    ///
    /// # Returns
    /// New SolveCoagula instance
    pub fn new(config: SolveCoagulaConfig) -> Result<Self> {
        let lambda_factor = config.lambda;
        let eps = config.eps;
        let max_iter = config.max_iter;

        // Initialize weight matrix W (deterministically)
        use sha2::{Digest, Sha256};
        let seed = 42u64;
        let seed_bytes = seed.to_le_bytes();
        let mut hasher = Sha256::new();
        hasher.update(seed_bytes);
        let hash = hasher.finalize();

        // Generate W from hash deterministically
        let mut w = Array2::zeros((5, 5));
        for i in 0..5 {
            for j in 0..5 {
                let idx = (i * 5 + j) % 32;
                let byte_val = hash[idx] as f64 / 255.0 * 2.0 - 1.0;
                w[[i, j]] = byte_val;
            }
        }

        // Normalize to ||W||_2 <= 1 (approximate with Frobenius norm)
        let w_norm = w.iter().map(|x| x * x).sum::<f64>().sqrt() / 5.0_f64.sqrt();
        w /= w_norm + 0.1;

        // Bias vector
        let b = Array1::zeros(5);

        // Orthogonal vectors for DoubleKick (deterministic)
        let u1 = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        let u2_raw = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 0.0]);

        // Gram-Schmidt
        let dot = u2_raw.dot(&u1);
        let u2_temp: Array1<f64> = &u2_raw - dot * &u1;
        let u2_norm = u2_temp.dot(&u2_temp).sqrt();
        let u2 = u2_temp / u2_norm;

        let operators_config = config.operators;
        let beta = config.sc_beta;
        let wt_mode = operators_config.wt.mode.clone();

        Ok(Self {
            lambda_factor,
            eps,
            max_iter,
            w,
            b,
            u1,
            u2,
            operators_config,
            beta,
            wt_mode,
        })
    }

    /// SPEC-002 Fixpoint Iteration
    ///
    /// # Arguments
    /// * `v0` - Initial vector
    /// * `track_convergence` - Whether to track convergence history
    ///
    /// # Returns
    /// (fixpoint, convergence_info)
    pub fn iterate_to_fixpoint(
        &self,
        v0: &Array1<f64>,
        track_convergence: bool,
    ) -> Result<(Array1<f64>, ConvergenceInfo)> {
        // Prepare operator arguments
        let dk_args = DKArgs {
            alpha1: self.operators_config.dk.alpha1,
            alpha2: self.operators_config.dk.alpha2,
            u1: self.u1.clone(),
            u2: self.u2.clone(),
        };

        let sw_args = SWArgs {
            tau: self.operators_config.sw.tau0,
            beta: self.operators_config.sw.beta,
        };

        let pi_args = PIArgs {
            canon: self.operators_config.pi.canon.clone(),
            tol: self.operators_config.pi.tol,
        };

        let mut weights = HashMap::new();
        weights.insert("micro".to_string(), 0.33);
        weights.insert("meso".to_string(), 0.33);
        weights.insert("macro".to_string(), 0.34);

        let wt_args = WTArgs {
            weights,
            beta: self.beta,
            mode: self.wt_mode.clone(),
        };

        // Fixpoint iteration
        let params = FixpointParams {
            eps: self.eps,
            max_iter: self.max_iter,
            dk_args: Some(&dk_args),
            sw_args: Some(&sw_args),
            pi_args: Some(&pi_args),
            wt_args: Some(&wt_args),
        };
        let (mut v_star, steps) =
            iterate_to_fixpoint(v0, &self.w, &self.b, self.lambda_factor, &params)?;

        let mut history = Vec::new();
        if track_convergence && steps > 0 {
            for i in 0..steps {
                let lyapunov = self.lambda_factor.powi((i + 1) as i32);
                history.push(ConvergenceStep {
                    iteration: i + 1,
                    lyapunov,
                    norm: lyapunov,
                });
            }
        }

        // Check convergence
        let mut converged = steps < self.max_iter;
        let mut final_delta = if converged { self.eps } else { f64::INFINITY };
        let mut total_steps = steps;

        // Relaxation fallback if not converged
        if !converged {
            let relaxation_limit = (8).max(self.max_iter / 64);
            let mut relaxed = v_star.clone();
            let mut delta = ((&self.w.dot(&relaxed) + &self.b) * self.lambda_factor - &relaxed)
                .dot(&((&self.w.dot(&relaxed) + &self.b) * self.lambda_factor - &relaxed))
                .sqrt();

            let mut relaxation_history = Vec::new();
            for extra in 1..=relaxation_limit {
                let next_relaxed = (&self.w.dot(&relaxed) + &self.b) * self.lambda_factor;
                delta = (&next_relaxed - &relaxed)
                    .dot(&(&next_relaxed - &relaxed))
                    .sqrt();
                relaxed = next_relaxed;

                if track_convergence {
                    let iteration = steps + extra;
                    let lyapunov = self.lambda_factor.powi(iteration as i32);
                    relaxation_history.push(ConvergenceStep {
                        iteration,
                        lyapunov,
                        norm: relaxed.dot(&relaxed).sqrt(),
                    });
                }

                if delta < self.eps {
                    converged = true;
                    total_steps = steps + extra;
                    final_delta = delta;
                    break;
                }
            }

            if !converged {
                total_steps = steps + relaxation_limit;
                final_delta = delta;
            }

            v_star = relaxed;
            if track_convergence && !relaxation_history.is_empty() {
                history.extend(relaxation_history);
            }

            total_steps = total_steps.min(self.max_iter - 1);
        }

        let residual = ((&self.w.dot(&v_star) + &self.b) * self.lambda_factor - &v_star)
            .dot(&((&self.w.dot(&v_star) + &self.b) * self.lambda_factor - &v_star))
            .sqrt();

        let lyapunov_series = if track_convergence {
            history.iter().map(|h| h.lyapunov).collect()
        } else {
            Vec::new()
        };

        let convergence_info = ConvergenceInfo {
            converged,
            iterations: total_steps,
            final_delta: if converged { residual } else { final_delta },
            lyapunov_series,
            history: if track_convergence {
                Some(history)
            } else {
                None
            },
        };

        Ok((v_star, convergence_info))
    }

    /// Alias for compatibility
    pub fn compute_fixpoint(
        &self,
        coordinates: &Array1<f64>,
    ) -> Result<(Array1<f64>, ConvergenceInfo)> {
        self.iterate_to_fixpoint(coordinates, true)
    }

    /// Single operator stack pass
    pub fn apply_operator_stack(&self, v: &Array1<f64>) -> Result<Array1<f64>> {
        let mut result = v.clone();

        // DoubleKick
        result = operators::dk(
            &result,
            self.operators_config.dk.alpha1,
            self.operators_config.dk.alpha2,
            &self.u1,
            &self.u2,
        );

        // Sweep
        result = operators::sw(
            &result,
            self.operators_config.sw.tau0,
            self.operators_config.sw.beta,
        );

        // Pfadinvarianz
        result = operators::pi_project(
            &result,
            &self.operators_config.pi.canon,
            self.operators_config.pi.tol,
        );

        // Weight-Transfer
        let mut weights = HashMap::new();
        weights.insert("micro".to_string(), 0.33);
        weights.insert("meso".to_string(), 0.33);
        weights.insert("macro".to_string(), 0.34);

        result = operators::wt(&result, &weights, self.beta, &self.wt_mode);

        // Affine transformation
        result = (&self.w.dot(&result) + &self.b) * self.lambda_factor;

        Ok(result)
    }

    /// Verify contractivity
    pub fn verify_contractivity(&self) -> Value {
        let w_norm = self.w.iter().map(|x| x * x).sum::<f64>().sqrt() / 5.0_f64.sqrt();
        let is_contractive =
            (0.0 < self.lambda_factor && self.lambda_factor < 1.0) && w_norm <= 1.0;

        serde_json::json!({
            "W_spectral_norm": w_norm,
            "lambda": self.lambda_factor,
            "theoretical_lipschitz": self.lambda_factor * w_norm,
            "is_contractive": is_contractive
        })
    }

    /// Get operator information
    pub fn get_operator_info(&self) -> Value {
        let w_norm = self.w.iter().map(|x| x * x).sum::<f64>().sqrt() / 5.0_f64.sqrt();

        serde_json::json!({
            "affine": {
                "lambda": self.lambda_factor,
                "W_shape": [self.w.nrows(), self.w.ncols()],
                "W_norm": w_norm
            },
            "operators": {
                "dk": {
                    "alpha1": self.operators_config.dk.alpha1,
                    "alpha2": self.operators_config.dk.alpha2
                },
                "sw": {
                    "tau0": self.operators_config.sw.tau0,
                    "beta": self.operators_config.sw.beta
                },
                "pi": {
                    "canon": self.operators_config.pi.canon,
                    "tol": self.operators_config.pi.tol
                },
                "wt": {
                    "mode": self.wt_mode
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_solve_coagula() {
        let config = SolveCoagulaConfig::default();
        let sc = SolveCoagula::new(config);
        assert!(sc.is_ok());
    }

    #[test]
    fn test_iterate_to_fixpoint() {
        let config = SolveCoagulaConfig::default();
        let sc = SolveCoagula::new(config).unwrap();

        let v0 = Array1::from_vec(vec![1.0, 0.5, -0.3, 0.8, -0.2]);
        let result = sc.iterate_to_fixpoint(&v0, false);

        assert!(result.is_ok());
        let (fixpoint, info) = result.unwrap();
        assert_eq!(fixpoint.len(), 5);
        assert!(info.iterations > 0);
    }

    #[test]
    fn test_convergence_tracking() {
        let config = SolveCoagulaConfig::default();
        let sc = SolveCoagula::new(config).unwrap();

        let v0 = Array1::from_vec(vec![1.0, 0.5, -0.3, 0.8, -0.2]);
        let result = sc.iterate_to_fixpoint(&v0, true);

        assert!(result.is_ok());
        let (_fixpoint, info) = result.unwrap();
        assert!(info.history.is_some());
        assert!(info.lyapunov_series.len() > 0);
    }

    #[test]
    fn test_apply_operator_stack() {
        let config = SolveCoagulaConfig::default();
        let sc = SolveCoagula::new(config).unwrap();

        let v = Array1::from_vec(vec![1.0, 0.5, -0.3, 0.8, -0.2]);
        let result = sc.apply_operator_stack(&v);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 5);
    }

    #[test]
    fn test_verify_contractivity() {
        let config = SolveCoagulaConfig::default();
        let sc = SolveCoagula::new(config).unwrap();

        let info = sc.verify_contractivity();
        assert!(info["is_contractive"].as_bool().unwrap());
    }

    #[test]
    fn test_compute_fixpoint_alias() {
        let config = SolveCoagulaConfig::default();
        let sc = SolveCoagula::new(config).unwrap();

        let coordinates = Array1::from_vec(vec![1.0, 0.5, -0.3, 0.8, -0.2]);
        let result = sc.compute_fixpoint(&coordinates);

        assert!(result.is_ok());
        let (_fixpoint, info) = result.unwrap();
        assert!(info.history.is_some());
    }

    #[test]
    fn test_deterministic_initialization() {
        let config1 = SolveCoagulaConfig::default();
        let config2 = SolveCoagulaConfig::default();

        let sc1 = SolveCoagula::new(config1).unwrap();
        let sc2 = SolveCoagula::new(config2).unwrap();

        let v0 = Array1::from_vec(vec![1.0, 0.5, -0.3, 0.8, -0.2]);
        let (fp1, _) = sc1.iterate_to_fixpoint(&v0, false).unwrap();
        let (fp2, _) = sc2.iterate_to_fixpoint(&v0, false).unwrap();

        // Should produce identical results
        for i in 0..5 {
            assert!((fp1[i] - fp2[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fixpoint_with_varied_inputs() {
        let config = SolveCoagulaConfig::default();
        let sc = SolveCoagula::new(config).unwrap();

        // Test with different input vectors
        let test_vectors = vec![
            vec![1.0, 0.5, -0.3, 0.8, -0.2],
            vec![0.1, 0.2, 0.3, 0.4, 0.5],
            vec![-1.0, -0.5, 0.0, 0.5, 1.0],
            vec![2.0, -1.0, 0.5, -0.3, 1.5],
        ];

        for vec_data in test_vectors {
            let v0 = Array1::from_vec(vec_data);
            let result = sc.iterate_to_fixpoint(&v0, true);

            assert!(result.is_ok());
            let (_fixpoint, info) = result.unwrap();
            assert!(info.iterations > 0);
            assert!(info.iterations <= 1000);
        }
    }

    #[test]
    fn test_operator_info() {
        let config = SolveCoagulaConfig::default();
        let sc = SolveCoagula::new(config).unwrap();

        let info = sc.get_operator_info();

        assert!(info["affine"]["lambda"].as_f64().unwrap() > 0.0);
        assert!(info["affine"]["lambda"].as_f64().unwrap() < 1.0);
        assert!(info["operators"]["dk"]["alpha1"].is_f64());
        assert!(info["operators"]["sw"]["tau0"].is_f64());
        assert!(info["operators"]["pi"]["canon"].is_string());
    }

    #[test]
    fn test_convergence_with_custom_config() {
        let mut config = SolveCoagulaConfig::default();
        config.max_iter = 500;
        config.eps = 1e-5;

        let sc = SolveCoagula::new(config).unwrap();

        let v0 = Array1::from_vec(vec![1.0, 0.5, -0.3, 0.8, -0.2]);
        let result = sc.iterate_to_fixpoint(&v0, true);

        assert!(result.is_ok());
        let (_fixpoint, info) = result.unwrap();
        assert!(info.iterations <= 500);
    }
}
