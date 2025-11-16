//! Classical Optimizers for Hybrid Quantum-Classical Algorithms
//!
//! Provides various optimization methods:
//! - ADAM: Gradient-based adaptive learning rate
//! - NelderMead: Gradient-free simplex method
//! - LBFGS: Limited-memory quasi-Newton method
//! - GradientDescent: Simple gradient descent with momentum

use crate::vqa::cost_function::{CostFunction, GradientMethod};
use crate::vqa::{HistoryEntry, OptimizationHistory, ParameterVector};
use std::sync::Arc;
use std::time::Instant;

/// Optimizer type selection
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OptimizerType {
    Adam,
    NelderMead,
    LBFGS,
    GradientDescent,
}

/// Configuration for optimizers
#[derive(Clone, Debug)]
pub struct OptimizerConfig {
    pub max_iterations: usize,
    /// Gradient norm tolerance for convergence
    pub tolerance: f64,
    /// Energy change tolerance for convergence: |E_k - E_{k-1}| < energy_tolerance
    pub energy_tolerance: f64,
    pub learning_rate: f64,
    pub gradient_method: GradientMethod,
    pub verbose: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            // More relaxed energy convergence (1e-3 is reasonable for VQE)
            energy_tolerance: 1e-3,
            learning_rate: 0.01,
            gradient_method: GradientMethod::ParameterShift,
            verbose: true,
        }
    }
}

/// Optimization result
#[derive(Clone, Debug)]
pub struct OptimizationResult {
    pub optimal_parameters: ParameterVector,
    pub optimal_cost: f64,
    pub iterations: usize,
    pub converged: bool,
    pub history: OptimizationHistory,
}

/// Main optimizer orchestrator
pub struct Optimizer {
    optimizer_type: OptimizerType,
    config: OptimizerConfig,
}

impl Optimizer {
    pub fn new(optimizer_type: OptimizerType, config: OptimizerConfig) -> Self {
        Self {
            optimizer_type,
            config,
        }
    }

    /// Run optimization with given cost function
    pub fn optimize(
        &self,
        cost_function: Arc<dyn CostFunction>,
        initial_parameters: ParameterVector,
    ) -> OptimizationResult {
        match self.optimizer_type {
            OptimizerType::Adam => self.optimize_adam(cost_function, initial_parameters),
            OptimizerType::NelderMead => {
                self.optimize_nelder_mead(cost_function, initial_parameters)
            }
            OptimizerType::LBFGS => self.optimize_lbfgs(cost_function, initial_parameters),
            OptimizerType::GradientDescent => {
                self.optimize_gradient_descent(cost_function, initial_parameters)
            }
        }
    }

    /// ADAM Optimizer (Adaptive Moment Estimation)
    fn optimize_adam(
        &self,
        cost_function: Arc<dyn CostFunction>,
        initial_parameters: ParameterVector,
    ) -> OptimizationResult {
        let start_time = Instant::now();
        let mut params = initial_parameters.clone();
        let mut history = OptimizationHistory::new();

        // ADAM hyperparameters
        let alpha = self.config.learning_rate;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let epsilon = 1e-8;

        let mut m = vec![0.0; params.len()]; // First moment
        let mut v = vec![0.0; params.len()]; // Second moment

        let mut best_cost = f64::INFINITY;
        let mut best_params = params.clone();
        let mut prev_cost = f64::INFINITY;

        for iter in 0..self.config.max_iterations {
            // Evaluate cost and gradient
            let cost = cost_function.evaluate(&params);
            let gradient = cost_function.gradient(&params, self.config.gradient_method.clone());

            // Update biased moment estimates
            for i in 0..params.len() {
                m[i] = beta1 * m[i] + (1.0 - beta1) * gradient[i];
                v[i] = beta2 * v[i] + (1.0 - beta2) * gradient[i] * gradient[i];
            }

            // Bias correction
            let m_hat: Vec<f64> = m
                .iter()
                .map(|mi| mi / (1.0 - beta1.powi((iter + 1) as i32)))
                .collect();
            let v_hat: Vec<f64> = v
                .iter()
                .map(|vi| vi / (1.0 - beta2.powi((iter + 1) as i32)))
                .collect();

            // Update parameters
            for i in 0..params.len() {
                params[i] -= alpha * m_hat[i] / (v_hat[i].sqrt() + epsilon);
            }

            // Track best solution
            if cost < best_cost {
                best_cost = cost;
                best_params = params.clone();
            }

            // Record history
            let gradient_norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            history.add_entry(HistoryEntry {
                iteration: iter,
                parameters: params.clone(),
                cost,
                gradient_norm: Some(gradient_norm),
                elapsed_time: start_time.elapsed().as_secs_f64(),
            });
            history.total_quantum_evaluations += 1 + params.len() * 2; // Cost + gradient evals

            // Verbose output
            if self.config.verbose && iter % 10 == 0 {
                println!(
                    "ADAM Iter {}: cost = {:.8}, |∇| = {:.6e}, ΔE = {:.6e}",
                    iter,
                    cost,
                    gradient_norm,
                    (cost - prev_cost).abs()
                );
            }

            // Convergence check: gradient norm OR energy change
            let energy_change = (cost - prev_cost).abs();
            let gradient_converged = gradient_norm < self.config.tolerance;
            let energy_converged = iter > 0 && energy_change < self.config.energy_tolerance;

            if gradient_converged || energy_converged {
                if self.config.verbose {
                    let reason = if gradient_converged {
                        "gradient"
                    } else {
                        "energy"
                    };
                    println!("Converged ({}) after {} iterations", reason, iter + 1);
                }
                return OptimizationResult {
                    optimal_parameters: best_params,
                    optimal_cost: best_cost,
                    iterations: iter + 1,
                    converged: true,
                    history,
                };
            }

            prev_cost = cost;
        }

        OptimizationResult {
            optimal_parameters: best_params,
            optimal_cost: best_cost,
            iterations: self.config.max_iterations,
            converged: false,
            history,
        }
    }

    /// Nelder-Mead Simplex Optimizer (gradient-free)
    fn optimize_nelder_mead(
        &self,
        cost_function: Arc<dyn CostFunction>,
        initial_parameters: ParameterVector,
    ) -> OptimizationResult {
        let start_time = Instant::now();
        let n = initial_parameters.len();
        let mut history = OptimizationHistory::new();

        // Reflection, expansion, contraction, shrinkage coefficients
        let alpha = 1.0; // reflection
        let gamma = 2.0; // expansion
        let rho = 0.5; // contraction
        let sigma = 0.5; // shrinkage

        // Initialize simplex
        let mut simplex: Vec<(ParameterVector, f64)> = Vec::with_capacity(n + 1);
        simplex.push((
            initial_parameters.clone(),
            cost_function.evaluate(&initial_parameters),
        ));

        // Create initial simplex with perturbations
        for i in 0..n {
            let mut vertex = initial_parameters.clone();
            vertex[i] += 0.1;
            let cost = cost_function.evaluate(&vertex);
            simplex.push((vertex, cost));
        }

        for iter in 0..self.config.max_iterations {
            // Sort simplex by cost
            simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let best_cost = simplex[0].1;
            let worst_cost = simplex[n].1;

            // Record history
            history.add_entry(HistoryEntry {
                iteration: iter,
                parameters: simplex[0].0.clone(),
                cost: best_cost,
                gradient_norm: None,
                elapsed_time: start_time.elapsed().as_secs_f64(),
            });
            history.total_quantum_evaluations += 1;

            if self.config.verbose && iter % 10 == 0 {
                println!("Nelder-Mead Iter {}: best_cost = {:.8}", iter, best_cost);
            }

            // Convergence check
            if (worst_cost - best_cost).abs() < self.config.tolerance {
                if self.config.verbose {
                    println!("Converged after {} iterations", iter + 1);
                }
                return OptimizationResult {
                    optimal_parameters: simplex[0].0.clone(),
                    optimal_cost: best_cost,
                    iterations: iter + 1,
                    converged: true,
                    history,
                };
            }

            // Compute centroid (excluding worst point)
            let mut centroid = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    centroid[j] += simplex[i].0[j];
                }
            }
            for j in 0..n {
                centroid[j] /= n as f64;
            }

            // Reflection
            let mut reflected = vec![0.0; n];
            for j in 0..n {
                reflected[j] = centroid[j] + alpha * (centroid[j] - simplex[n].0[j]);
            }
            let reflected_cost = cost_function.evaluate(&reflected);

            if reflected_cost < simplex[n - 1].1 && reflected_cost >= simplex[0].1 {
                simplex[n] = (reflected, reflected_cost);
                continue;
            }

            // Expansion
            if reflected_cost < simplex[0].1 {
                let mut expanded = vec![0.0; n];
                for j in 0..n {
                    expanded[j] = centroid[j] + gamma * (reflected[j] - centroid[j]);
                }
                let expanded_cost = cost_function.evaluate(&expanded);

                if expanded_cost < reflected_cost {
                    simplex[n] = (expanded, expanded_cost);
                } else {
                    simplex[n] = (reflected, reflected_cost);
                }
                continue;
            }

            // Contraction
            let mut contracted = vec![0.0; n];
            for j in 0..n {
                contracted[j] = centroid[j] + rho * (simplex[n].0[j] - centroid[j]);
            }
            let contracted_cost = cost_function.evaluate(&contracted);

            if contracted_cost < simplex[n].1 {
                simplex[n] = (contracted, contracted_cost);
                continue;
            }

            // Shrinkage
            for i in 1..=n {
                for j in 0..n {
                    simplex[i].0[j] = simplex[0].0[j] + sigma * (simplex[i].0[j] - simplex[0].0[j]);
                }
                simplex[i].1 = cost_function.evaluate(&simplex[i].0);
            }
        }

        simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        OptimizationResult {
            optimal_parameters: simplex[0].0.clone(),
            optimal_cost: simplex[0].1,
            iterations: self.config.max_iterations,
            converged: false,
            history,
        }
    }

    /// L-BFGS Optimizer (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)
    fn optimize_lbfgs(
        &self,
        cost_function: Arc<dyn CostFunction>,
        initial_parameters: ParameterVector,
    ) -> OptimizationResult {
        let start_time = Instant::now();
        let mut params = initial_parameters.clone();
        let mut history = OptimizationHistory::new();

        let memory_size = 10;
        let mut s_list: Vec<ParameterVector> = Vec::new();
        let mut y_list: Vec<ParameterVector> = Vec::new();

        let mut prev_gradient =
            cost_function.gradient(&params, self.config.gradient_method.clone());
        let mut best_cost = cost_function.evaluate(&params);
        let mut best_params = params.clone();
        let mut prev_cost = best_cost;

        for iter in 0..self.config.max_iterations {
            let cost = cost_function.evaluate(&params);
            let gradient = cost_function.gradient(&params, self.config.gradient_method.clone());

            if cost < best_cost {
                best_cost = cost;
                best_params = params.clone();
            }

            let gradient_norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();

            history.add_entry(HistoryEntry {
                iteration: iter,
                parameters: params.clone(),
                cost,
                gradient_norm: Some(gradient_norm),
                elapsed_time: start_time.elapsed().as_secs_f64(),
            });
            history.total_quantum_evaluations += 1 + params.len() * 2;

            if self.config.verbose && iter % 10 == 0 {
                println!(
                    "L-BFGS Iter {}: cost = {:.8}, |∇| = {:.6e}, ΔE = {:.6e}",
                    iter,
                    cost,
                    gradient_norm,
                    (cost - prev_cost).abs()
                );
            }

            // Convergence check: gradient norm OR energy change
            let energy_change = (cost - prev_cost).abs();
            let gradient_converged = gradient_norm < self.config.tolerance;
            let energy_converged = iter > 0 && energy_change < self.config.energy_tolerance;

            if gradient_converged || energy_converged {
                if self.config.verbose {
                    let reason = if gradient_converged {
                        "gradient"
                    } else {
                        "energy"
                    };
                    println!("Converged ({}) after {} iterations", reason, iter + 1);
                }
                return OptimizationResult {
                    optimal_parameters: best_params,
                    optimal_cost: best_cost,
                    iterations: iter + 1,
                    converged: true,
                    history,
                };
            }

            prev_cost = cost;

            // Compute search direction using L-BFGS two-loop recursion
            let mut q = gradient.clone();
            let mut alpha_list = Vec::new();

            for i in (0..s_list.len()).rev() {
                let rho = 1.0
                    / s_list[i]
                        .iter()
                        .zip(y_list[i].iter())
                        .map(|(s, y)| s * y)
                        .sum::<f64>();
                let alpha = rho
                    * s_list[i]
                        .iter()
                        .zip(q.iter())
                        .map(|(s, qi)| s * qi)
                        .sum::<f64>();
                alpha_list.push(alpha);
                for j in 0..q.len() {
                    q[j] -= alpha * y_list[i][j];
                }
            }

            // Scale
            let mut r = q.clone();
            if !y_list.is_empty() {
                let gamma = s_list[s_list.len() - 1]
                    .iter()
                    .zip(y_list[y_list.len() - 1].iter())
                    .map(|(s, y)| s * y)
                    .sum::<f64>()
                    / y_list[y_list.len() - 1].iter().map(|y| y * y).sum::<f64>();
                for j in 0..r.len() {
                    r[j] *= gamma;
                }
            }

            alpha_list.reverse();
            for i in 0..s_list.len() {
                let rho = 1.0
                    / s_list[i]
                        .iter()
                        .zip(y_list[i].iter())
                        .map(|(s, y)| s * y)
                        .sum::<f64>();
                let beta = rho
                    * y_list[i]
                        .iter()
                        .zip(r.iter())
                        .map(|(y, ri)| y * ri)
                        .sum::<f64>();
                for j in 0..r.len() {
                    r[j] += s_list[i][j] * (alpha_list[i] - beta);
                }
            }

            // Line search (simple backtracking)
            let step_size = self.config.learning_rate;
            let mut new_params = params.clone();
            for j in 0..new_params.len() {
                new_params[j] -= step_size * r[j];
            }

            // Update history
            let s: ParameterVector = params
                .iter()
                .zip(new_params.iter())
                .map(|(old, new)| new - old)
                .collect();
            let y: ParameterVector = gradient
                .iter()
                .zip(prev_gradient.iter())
                .map(|(g_new, g_old)| g_new - g_old)
                .collect();

            s_list.push(s);
            y_list.push(y);

            if s_list.len() > memory_size {
                s_list.remove(0);
                y_list.remove(0);
            }

            params = new_params;
            prev_gradient = gradient;
        }

        OptimizationResult {
            optimal_parameters: best_params,
            optimal_cost: best_cost,
            iterations: self.config.max_iterations,
            converged: false,
            history,
        }
    }

    /// Simple Gradient Descent with Momentum
    fn optimize_gradient_descent(
        &self,
        cost_function: Arc<dyn CostFunction>,
        initial_parameters: ParameterVector,
    ) -> OptimizationResult {
        let start_time = Instant::now();
        let mut params = initial_parameters.clone();
        let mut history = OptimizationHistory::new();

        let momentum = 0.9;
        let mut velocity = vec![0.0; params.len()];

        let mut best_cost = f64::INFINITY;
        let mut best_params = params.clone();

        for iter in 0..self.config.max_iterations {
            let cost = cost_function.evaluate(&params);
            let gradient = cost_function.gradient(&params, self.config.gradient_method.clone());

            if cost < best_cost {
                best_cost = cost;
                best_params = params.clone();
            }

            // Update velocity with momentum
            for i in 0..params.len() {
                velocity[i] = momentum * velocity[i] + self.config.learning_rate * gradient[i];
                params[i] -= velocity[i];
            }

            let gradient_norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();

            history.add_entry(HistoryEntry {
                iteration: iter,
                parameters: params.clone(),
                cost,
                gradient_norm: Some(gradient_norm),
                elapsed_time: start_time.elapsed().as_secs_f64(),
            });
            history.total_quantum_evaluations += 1 + params.len() * 2;

            if self.config.verbose && iter % 10 == 0 {
                println!(
                    "GradientDescent Iter {}: cost = {:.8}, |∇| = {:.6e}",
                    iter, cost, gradient_norm
                );
            }

            if gradient_norm < self.config.tolerance {
                if self.config.verbose {
                    println!("Converged after {} iterations", iter + 1);
                }
                return OptimizationResult {
                    optimal_parameters: best_params,
                    optimal_cost: best_cost,
                    iterations: iter + 1,
                    converged: true,
                    history,
                };
            }
        }

        OptimizationResult {
            optimal_parameters: best_params,
            optimal_cost: best_cost,
            iterations: self.config.max_iterations,
            converged: false,
            history,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestCostFunction;

    impl CostFunction for TestCostFunction {
        fn evaluate(&self, parameters: &[f64]) -> f64 {
            // Rosenbrock function
            parameters
                .windows(2)
                .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
                .sum()
        }

        fn gradient(&self, parameters: &[f64], _method: GradientMethod) -> ParameterVector {
            let mut grad = vec![0.0; parameters.len()];
            for i in 0..parameters.len() - 1 {
                grad[i] +=
                    -400.0 * parameters[i] * (parameters[i + 1] - parameters[i] * parameters[i])
                        - 2.0 * (1.0 - parameters[i]);
                grad[i + 1] += 200.0 * (parameters[i + 1] - parameters[i] * parameters[i]);
            }
            grad
        }

        fn dimension(&self) -> usize {
            2
        }
    }

    #[test]
    fn test_adam_optimizer() {
        let cost_fn = Arc::new(TestCostFunction);
        let initial = vec![-1.0, -1.0];

        let config = OptimizerConfig {
            max_iterations: 100,
            learning_rate: 0.1,
            verbose: false,
            ..Default::default()
        };

        let optimizer = Optimizer::new(OptimizerType::Adam, config);
        let result = optimizer.optimize(cost_fn, initial);

        assert!(result.optimal_cost < 1.0);
    }
}
