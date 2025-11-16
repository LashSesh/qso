//! Numerical Integration Schemes
//!
//! Implements Heun's method (RK2) for time-stepping the dynamical system.
//! Section 4 of the specification.

use crate::dynamics::VectorField;
use crate::state::State5D;

/// Time discretization parameters
#[derive(Debug, Clone, Copy)]
pub struct TimeConfig {
    /// Time step Δt
    pub dt: f64,
    /// Initial time t₀
    pub t0: f64,
    /// Final time T
    pub t_final: f64,
}

impl TimeConfig {
    /// Create a new time configuration
    pub fn new(dt: f64, t0: f64, t_final: f64) -> Self {
        assert!(dt > 0.0, "Time step must be positive");
        assert!(t_final >= t0, "Final time must be >= initial time");

        TimeConfig { dt, t0, t_final }
    }

    /// Number of time steps
    pub fn num_steps(&self) -> usize {
        ((self.t_final - self.t0) / self.dt).ceil() as usize
    }

    /// Get time at step n
    pub fn time_at_step(&self, n: usize) -> f64 {
        self.t0 + n as f64 * self.dt
    }
}

/// Numerical integrator for the dynamical system
///
/// Implements Heun's method (RK2) as specified in Equations (12)-(13):
/// σ̃ⁿ⁺¹ = σⁿ + Δt · F(σⁿ)
/// σⁿ⁺¹ = σⁿ + (Δt/2) · [F(σⁿ) + F(σ̃ⁿ⁺¹)]
#[derive(Debug)]
pub struct Integrator {
    pub vector_field: VectorField,
    pub time_config: TimeConfig,
}

impl Integrator {
    /// Create a new integrator
    pub fn new(vector_field: VectorField, time_config: TimeConfig) -> Self {
        Integrator {
            vector_field,
            time_config,
        }
    }

    /// Perform a single Heun step: σⁿ → σⁿ⁺¹
    ///
    /// # Arguments
    /// * `state` - Current state σⁿ
    ///
    /// # Returns
    /// Next state σⁿ⁺¹
    pub fn step(&self, state: &State5D) -> State5D {
        let dt = self.time_config.dt;

        // Predictor step: σ̃ⁿ⁺¹ = σⁿ + Δt · F(σⁿ)
        let k1 = self.vector_field.evaluate(state);
        let state_predictor = state.add(&k1.scale(dt));

        // Check if predictor is valid
        if !state_predictor.is_valid() {
            // Return state unchanged if predictor failed
            eprintln!("Warning: Heun predictor produced invalid state, keeping current state");
            return *state;
        }

        // Corrector step: F(σ̃ⁿ⁺¹)
        let k2 = self.vector_field.evaluate(&state_predictor);

        // Final update: σⁿ⁺¹ = σⁿ + (Δt/2) · [F(σⁿ) + F(σ̃ⁿ⁺¹)]
        let state_new = state.add(&(k1.add(&k2)).scale(dt / 2.0));

        // Validate result
        if !state_new.is_valid() {
            eprintln!("Warning: Heun corrector produced invalid state, keeping current state");
            return *state;
        }

        state_new
    }

    /// Integrate from initial state to final time, storing all states
    ///
    /// # Arguments
    /// * `initial_state` - Initial condition σ⁰
    ///
    /// # Returns
    /// Vector of (time, state) pairs for the trajectory
    pub fn integrate(&self, initial_state: State5D) -> Vec<(f64, State5D)> {
        let num_steps = self.time_config.num_steps();
        let mut trajectory = Vec::with_capacity(num_steps + 1);

        let mut state = initial_state;
        trajectory.push((self.time_config.t0, state));

        for n in 0..num_steps {
            state = self.step(&state);
            let t = self.time_config.time_at_step(n + 1);
            trajectory.push((t, state));

            // Stop if state becomes invalid
            if !state.is_valid() {
                eprintln!("Integration stopped at step {} due to invalid state", n + 1);
                break;
            }
        }

        trajectory
    }

    /// Integrate and return only state values (without times)
    pub fn integrate_states(&self, initial_state: State5D) -> Vec<State5D> {
        self.integrate(initial_state)
            .into_iter()
            .map(|(_, state)| state)
            .collect()
    }

    /// Integrate and return only final state
    pub fn integrate_final(&self, initial_state: State5D) -> State5D {
        let num_steps = self.time_config.num_steps();
        let mut state = initial_state;

        for _ in 0..num_steps {
            state = self.step(&state);
            if !state.is_valid() {
                break;
            }
        }

        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coupling::CouplingMatrix;
    use crate::dynamics::{SystemParameters, VectorField};

    #[test]
    fn test_time_config() {
        let tc = TimeConfig::new(0.1, 0.0, 1.0);
        assert_eq!(tc.num_steps(), 10);
        assert_eq!(tc.time_at_step(0), 0.0);
        assert_eq!(tc.time_at_step(5), 0.5);
    }

    #[test]
    fn test_integrator_constant() {
        // F(σ) = 0 everywhere, so state should remain unchanged
        let coupling = CouplingMatrix::zero();
        let params = SystemParameters::zero();
        let vf = VectorField::new(coupling, params);

        let tc = TimeConfig::new(0.1, 0.0, 1.0);
        let integrator = Integrator::new(vf, tc);

        let initial = State5D::new(1.0, 2.0, 3.0, 4.0, 5.0);
        let trajectory = integrator.integrate(initial);

        // All states should be the same as initial
        for (_, state) in trajectory.iter() {
            for i in 0..5 {
                assert!((state.get(i) - initial.get(i)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_integrator_linear_decay() {
        // F(σ) = -σ, so dσ/dt = -σ, solution: σ(t) = σ₀ · exp(-t)
        let mut coupling = CouplingMatrix::zero();
        // Set diagonal to -1
        for i in 0..5 {
            coupling.strengths[i][i] = -1.0;
        }

        let vf = VectorField::from_coupling(coupling);
        let tc = TimeConfig::new(0.01, 0.0, 1.0);
        let integrator = Integrator::new(vf, tc);

        let initial = State5D::new(1.0, 1.0, 1.0, 1.0, 1.0);
        let final_state = integrator.integrate_final(initial);

        // After t=1, exp(-1) ≈ 0.3679
        let expected = (-1.0_f64).exp();
        for i in 0..5 {
            assert!((final_state.get(i) - expected).abs() < 0.01);
        }
    }

    #[test]
    fn test_heun_single_step() {
        let coupling = CouplingMatrix::identity();
        let vf = VectorField::from_coupling(coupling);
        let tc = TimeConfig::new(0.1, 0.0, 1.0);
        let integrator = Integrator::new(vf, tc);

        let state = State5D::new(1.0, 0.0, 0.0, 0.0, 0.0);
        let next = integrator.step(&state);

        // With F(σ) = σ and Heun's method:
        // Predictor: σ̃ = σ + dt*σ = σ(1+dt)
        // Corrector: σ_new = σ + dt/2 * (σ + σ̃) = σ(1 + dt + dt²/2)
        let dt = 0.1;
        let expected = 1.0 * (1.0 + dt + dt * dt / 2.0);
        assert!((next.get(0) - expected).abs() < 1e-10);
    }
}
