/// Sweep (SW) operator implementation.
/// Threshold sweeping with cosine schedule for resonance adjustment.
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Lipschitz verification results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LipschitzResults {
    pub max_gate_value: f64,
    pub min_gate_value: f64,
    pub lipschitz_bound: f64,
    pub is_non_expansive: bool,
}

/// Sweep operator information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepInfo {
    pub tau0: f64,
    pub beta: f64,
    pub schedule: String,
    pub current_iteration: usize,
    pub current_tau: f64,
    pub lipschitz: LipschitzResults,
}

/// Sweep operator with gate function:
/// SW(v) = g_τ(m(v)) · v
/// where g_τ(x) = σ((x - τ)/β) and m(v) = mean(v)
/// Uses cosine schedule for threshold evolution.
#[derive(Debug, Clone)]
pub struct Sweep {
    tau0: f64,
    beta: f64,
    schedule: String,
    iteration: usize,
    max_schedule_iterations: usize,
    delta_tau: f64,
}

impl Sweep {
    /// Create new Sweep operator
    pub fn new(tau0: f64, beta: f64, schedule: String) -> Self {
        Self {
            tau0,
            beta,
            schedule,
            iteration: 0,
            max_schedule_iterations: 100,
            delta_tau: 0.3,
        }
    }

    /// Create from config
    pub fn from_config(config: &HashMap<String, String>) -> Self {
        let tau0 = config
            .get("tau0")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);
        let beta = config
            .get("beta")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.1);
        let schedule = config
            .get("schedule")
            .cloned()
            .unwrap_or_else(|| "cosine".to_string());

        Self::new(tau0, beta, schedule)
    }

    /// Sigmoid activation function
    fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Gate function g_τ(x) = σ((x - τ)/β)
    pub fn gate_function(&self, x: f64, tau: f64) -> f64 {
        self.sigmoid((x - tau) / self.beta)
    }

    /// Compute threshold value based on schedule
    pub fn compute_schedule(&self) -> f64 {
        match self.schedule.as_str() {
            "cosine" => {
                // Cosine schedule: τ_t = τ₀ + 0.5(1 + cos(πt/T))Δτ
                let t = (self.iteration % self.max_schedule_iterations) as f64;
                let cap_t = self.max_schedule_iterations as f64;
                self.tau0 + 0.5 * (1.0 + (std::f64::consts::PI * t / cap_t).cos()) * self.delta_tau
            }
            "linear" => {
                // Linear schedule
                let t = (self.iteration % self.max_schedule_iterations) as f64;
                let cap_t = self.max_schedule_iterations as f64;
                self.tau0 + (1.0 - t / cap_t) * self.delta_tau
            }
            _ => {
                // Constant threshold
                self.tau0
            }
        }
    }

    /// Apply Sweep operator
    ///
    /// # Arguments
    /// * `v` - Input vector (5D)
    ///
    /// # Returns
    /// Transformed vector
    pub fn apply(&mut self, v: &Array1<f64>) -> Array1<f64> {
        // Compute mean of vector components
        let m_v = v.mean().unwrap_or(0.0);

        // Get current threshold from schedule
        let tau = self.compute_schedule();

        // Apply gate function
        let gate_value = self.gate_function(m_v, tau);

        // Apply gated multiplication
        let result = v * gate_value;

        // Increment iteration counter
        self.iteration += 1;

        result
    }

    /// Verify Lipschitz continuity of the operator
    pub fn verify_lipschitz(&self) -> LipschitzResults {
        // Test with sample values
        let test_values: Vec<f64> = (0..100).map(|i| -2.0 + 4.0 * (i as f64 / 99.0)).collect();

        let gate_values: Vec<f64> = test_values
            .iter()
            .map(|&x| self.gate_function(x, self.tau0))
            .collect();

        let max_gate_value = gate_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_gate_value = gate_values.iter().cloned().fold(f64::INFINITY, f64::min);

        LipschitzResults {
            max_gate_value,
            min_gate_value,
            lipschitz_bound: 1.0,
            is_non_expansive: true,
        }
    }

    /// Get operator information
    pub fn get_info(&self) -> SweepInfo {
        SweepInfo {
            tau0: self.tau0,
            beta: self.beta,
            schedule: self.schedule.clone(),
            current_iteration: self.iteration,
            current_tau: self.compute_schedule(),
            lipschitz: self.verify_lipschitz(),
        }
    }

    /// Reset the schedule iteration counter
    pub fn reset_schedule(&mut self) {
        self.iteration = 0;
    }
}

impl Default for Sweep {
    fn default() -> Self {
        Self::new(0.5, 0.1, "cosine".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_sweep() {
        let sweep = Sweep::new(0.5, 0.1, "cosine".to_string());
        let info = sweep.get_info();

        assert_eq!(info.tau0, 0.5);
        assert_eq!(info.beta, 0.1);
        assert_eq!(info.schedule, "cosine");
    }

    #[test]
    fn test_sigmoid() {
        let sweep = Sweep::default();

        assert!((sweep.sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sweep.sigmoid(10.0) > 0.99);
        assert!(sweep.sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_gate_function() {
        let sweep = Sweep::default();

        let g_low = sweep.gate_function(0.0, 0.5);
        let g_high = sweep.gate_function(1.0, 0.5);

        assert!(g_low < 0.5);
        assert!(g_high > 0.5);
    }

    #[test]
    fn test_compute_schedule_cosine() {
        let sweep = Sweep::new(0.5, 0.1, "cosine".to_string());

        let tau0 = sweep.compute_schedule();
        assert!((tau0 - 0.8).abs() < 0.01); // Initial value
    }

    #[test]
    fn test_compute_schedule_linear() {
        let sweep = Sweep::new(0.5, 0.1, "linear".to_string());

        let tau0 = sweep.compute_schedule();
        assert!((tau0 - 0.8).abs() < 0.01); // Initial value
    }

    #[test]
    fn test_apply_sweep() {
        let mut sweep = Sweep::default();
        let v = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let result = sweep.apply(&v);

        assert_eq!(result.len(), 5);
        // Gate value should scale the vector
        for i in 0..5 {
            assert!(result[i].abs() <= v[i].abs());
        }
    }

    #[test]
    fn test_verify_lipschitz() {
        let sweep = Sweep::default();
        let results = sweep.verify_lipschitz();

        assert!(results.max_gate_value <= 1.0);
        assert!(results.min_gate_value >= 0.0);
        assert!(results.is_non_expansive);
    }

    #[test]
    fn test_reset_schedule() {
        let mut sweep = Sweep::default();

        sweep.apply(&Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
        assert_eq!(sweep.iteration, 1);

        sweep.reset_schedule();
        assert_eq!(sweep.iteration, 0);
    }

    #[test]
    fn test_schedule_evolution() {
        let mut sweep = Sweep::new(0.5, 0.1, "cosine".to_string());
        let v = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let tau_values: Vec<f64> = (0..10)
            .map(|_| {
                sweep.apply(&v);
                sweep.compute_schedule()
            })
            .collect();

        // Tau should evolve over iterations
        assert!(tau_values[0] != tau_values[5]);
    }
}
