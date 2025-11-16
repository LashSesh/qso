//! Ensemble and Parameter Exploration
//!
//! Section 9 - Monte Carlo ensembles and parameter sweeps.

use crate::dynamics::VectorField;
use crate::integration::{Integrator, TimeConfig};
use crate::state::State5D;
use crate::template::Template;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for Monte Carlo ensemble simulation
#[derive(Debug, Clone)]
pub struct EnsembleConfig {
    pub num_runs: usize,
    pub initial_state_mean: State5D,
    pub initial_state_std: f64,
}

impl EnsembleConfig {
    pub fn new(num_runs: usize, initial_state_mean: State5D, initial_state_std: f64) -> Self {
        EnsembleConfig {
            num_runs,
            initial_state_mean,
            initial_state_std,
        }
    }
}

/// Result of an ensemble simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleResult {
    pub trajectories: Vec<Vec<State5D>>,
    pub mean_trajectory: Vec<State5D>,
    pub std_trajectory: Vec<State5D>,
}

impl EnsembleResult {
    /// Compute statistics from multiple trajectories
    pub fn from_trajectories(trajectories: Vec<Vec<State5D>>) -> Self {
        if trajectories.is_empty() {
            return EnsembleResult {
                trajectories,
                mean_trajectory: vec![],
                std_trajectory: vec![],
            };
        }

        let num_steps = trajectories[0].len();
        let num_runs = trajectories.len();

        let mut mean_trajectory = Vec::with_capacity(num_steps);
        let mut std_trajectory = Vec::with_capacity(num_steps);

        for step in 0..num_steps {
            // Compute mean at this time step
            let mut mean = State5D::zero();
            for traj in &trajectories {
                if step < traj.len() {
                    mean = mean + traj[step];
                }
            }
            mean = mean.scale(1.0 / num_runs as f64);

            // Compute standard deviation at this time step
            let mut variance = State5D::zero();
            for traj in &trajectories {
                if step < traj.len() {
                    let diff = traj[step].sub(&mean);
                    for i in 0..5 {
                        let val = variance.get(i) + diff.get(i) * diff.get(i);
                        variance.set(i, val);
                    }
                }
            }

            let mut std = State5D::zero();
            for i in 0..5 {
                let val = (variance.get(i) / num_runs as f64).sqrt();
                std.set(i, val);
            }

            mean_trajectory.push(mean);
            std_trajectory.push(std);
        }

        EnsembleResult {
            trajectories,
            mean_trajectory,
            std_trajectory,
        }
    }
}

/// Parameter sweep configuration
#[derive(Debug, Clone)]
pub struct ParameterSweep {
    pub parameter_name: String,
    pub values: Vec<f64>,
}

impl ParameterSweep {
    pub fn new(parameter_name: String, start: f64, end: f64, num_points: usize) -> Self {
        let step = (end - start) / (num_points - 1) as f64;
        let values: Vec<f64> = (0..num_points).map(|i| start + i as f64 * step).collect();

        ParameterSweep {
            parameter_name,
            values,
        }
    }
}

/// Run Monte Carlo ensemble simulation
///
/// Generates multiple trajectories with random initial conditions sampled
/// from a Gaussian distribution around the mean state.
///
/// # Arguments
/// * `config` - Ensemble configuration (number of runs, mean, std)
/// * `vf` - Vector field defining the dynamics
/// * `tc` - Time configuration for integration
///
/// # Returns
/// EnsembleResult containing all trajectories and statistics
pub fn run_ensemble(config: &EnsembleConfig, vf: &VectorField, tc: &TimeConfig) -> EnsembleResult {
    let mut rng = rand::thread_rng();
    let mut trajectories = Vec::with_capacity(config.num_runs);

    for _ in 0..config.num_runs {
        // Generate random initial condition
        let mut initial = State5D::zero();
        for i in 0..5 {
            let mean = config.initial_state_mean.get(i);
            let std = config.initial_state_std;
            let value = rng.gen_range((mean - 3.0 * std)..(mean + 3.0 * std));
            initial.set(i, value);
        }

        // Run integration
        let integrator = Integrator::new(vf.clone(), *tc);
        let trajectory = integrator.integrate_states(initial);
        trajectories.push(trajectory);
    }

    EnsembleResult::from_trajectories(trajectories)
}

/// Run parameter sweep
///
/// Varies a single parameter across a range of values and runs
/// integration for each value.
///
/// # Arguments
/// * `sweep` - Parameter sweep configuration
/// * `base_template` - Base template to modify
/// * `initial` - Initial state for all runs
/// * `tc` - Time configuration for integration
///
/// # Returns
/// Vector of trajectories, one for each parameter value
pub fn run_parameter_sweep(
    sweep: &ParameterSweep,
    base_template: &Template,
    initial: State5D,
    tc: &TimeConfig,
) -> Vec<Vec<State5D>> {
    let mut results = Vec::with_capacity(sweep.values.len());

    for &param_value in &sweep.values {
        // Create modified template with new parameter value
        let mut modified_template = base_template.clone();

        // Modify the template based on parameter name
        // This is a simplified version - in practice, you'd need more sophisticated
        // parameter mapping based on the template structure
        if sweep.parameter_name.contains("rate") {
            // Modify intrinsic rates proportionally
            for i in 0..5 {
                modified_template.parameters.intrinsic_rates[i] *= param_value;
            }
        } else if sweep.parameter_name.contains("coupling") {
            // Scale all coupling strengths
            for i in 0..5 {
                for j in 0..5 {
                    let strength = modified_template.coupling_matrix.get_strength(i, j);
                    modified_template.coupling_matrix.set(
                        i,
                        j,
                        strength * param_value,
                        modified_template.coupling_matrix.get_type(i, j),
                    );
                }
            }
        }

        // Run integration with modified template
        let vf = modified_template.to_vector_field();
        let integrator = Integrator::new(vf, *tc);
        let trajectory = integrator.integrate_states(initial);
        results.push(trajectory);
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_config() {
        let mean = State5D::new(1.0, 2.0, 3.0, 4.0, 5.0);
        let config = EnsembleConfig::new(10, mean, 0.1);

        assert_eq!(config.num_runs, 10);
        assert_eq!(config.initial_state_std, 0.1);
    }

    #[test]
    fn test_parameter_sweep() {
        let sweep = ParameterSweep::new("beta".to_string(), 0.0, 1.0, 11);

        assert_eq!(sweep.values.len(), 11);
        assert_eq!(sweep.values[0], 0.0);
        assert_eq!(sweep.values[10], 1.0);
    }

    #[test]
    fn test_ensemble_result() {
        let traj1 = vec![
            State5D::new(1.0, 0.0, 0.0, 0.0, 0.0),
            State5D::new(2.0, 0.0, 0.0, 0.0, 0.0),
        ];
        let traj2 = vec![
            State5D::new(3.0, 0.0, 0.0, 0.0, 0.0),
            State5D::new(4.0, 0.0, 0.0, 0.0, 0.0),
        ];

        let result = EnsembleResult::from_trajectories(vec![traj1, traj2]);

        assert_eq!(result.mean_trajectory.len(), 2);
        assert_eq!(result.mean_trajectory[0].get(0), 2.0); // (1+3)/2
        assert_eq!(result.mean_trajectory[1].get(0), 3.0); // (2+4)/2
    }

    #[test]
    fn test_run_ensemble() {
        use crate::coupling::CouplingMatrix;
        use crate::dynamics::{SystemParameters, VectorField};
        use crate::integration::TimeConfig;

        // Create simple system with linear decay
        let coupling = CouplingMatrix::zero();
        let mut params = SystemParameters::zero();
        params.intrinsic_rates = [-0.1, -0.1, -0.1, -0.1, -0.1];
        let vf = VectorField::new(coupling, params);

        // Configure ensemble
        let mean_state = State5D::new(1.0, 1.0, 1.0, 1.0, 1.0);
        let config = EnsembleConfig::new(10, mean_state, 0.1);

        // Time configuration
        let tc = TimeConfig::new(0.1, 0.0, 1.0);

        // Run ensemble
        let result = run_ensemble(&config, &vf, &tc);

        assert_eq!(result.trajectories.len(), 10);
        assert!(result.mean_trajectory.len() > 0);
        assert!(result.std_trajectory.len() > 0);
    }

    #[test]
    fn test_run_parameter_sweep() {
        use crate::integration::TimeConfig;
        use crate::template::Template;

        // Create base template
        let template = Template::sir_model(0.3, 0.1, 0.01);

        // Configure parameter sweep
        let sweep = ParameterSweep::new("coupling".to_string(), 0.5, 1.5, 5);

        // Initial state
        let initial = State5D::new(0.99, 0.01, 0.0, 0.0, 0.0);

        // Time configuration
        let tc = TimeConfig::new(0.1, 0.0, 10.0);

        // Run sweep
        let results = run_parameter_sweep(&sweep, &template, initial, &tc);

        assert_eq!(results.len(), 5);
        for traj in results {
            assert!(traj.len() > 0);
        }
    }
}
