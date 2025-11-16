// Unified system: Cognitive simulator combining 5D dynamics with Metatron cognition

use crate::adaptive_coupling::AdaptiveCoupling;
use crate::trajectory_observer::TrajectoryObserver;
use core_5d::{integration::Integrator, State5D, VectorField};

/// Cognitive simulator that integrates 5D dynamics with adaptive feedback
///
/// This is the high-level unified system that combines:
/// - 5D numerical integration (physics engine)
/// - Trajectory observation (feature extraction)
/// - Cognitive feedback (adaptive parameter control)
#[derive(Debug)]
pub struct CognitiveSimulator {
    integrator: Integrator,
    observer: TrajectoryObserver,
    adaptive_coupling: Option<AdaptiveCoupling>,
}

impl CognitiveSimulator {
    /// Create a new cognitive simulator
    pub fn new(integrator: Integrator, observer: TrajectoryObserver) -> Self {
        Self {
            integrator,
            observer,
            adaptive_coupling: None,
        }
    }

    /// Create a cognitive simulator with adaptive coupling
    pub fn with_adaptive_coupling(
        integrator: Integrator,
        observer: TrajectoryObserver,
        adaptive_coupling: AdaptiveCoupling,
    ) -> Self {
        Self {
            integrator,
            observer,
            adaptive_coupling: Some(adaptive_coupling),
        }
    }

    /// Perform one integration step with cognitive observation
    pub fn step(&mut self, sigma: State5D, _dt: f64) -> State5D {
        // 1. Integrate one step using 5D framework
        let sigma_next = self.integrator.step(&sigma);

        // 2. Observe the new state
        self.observer.observe(sigma_next);

        // 3. In the future, this is where cognitive feedback would occur
        // (e.g., adjust coupling based on trajectory analysis)

        sigma_next
    }

    /// Integrate a full trajectory with adaptive coupling modulation
    pub fn integrate_adaptive(&mut self, initial: State5D) -> Vec<(f64, State5D)> {
        if self.adaptive_coupling.is_none() {
            // Fall back to standard integration if no adaptive coupling
            return self.integrator.integrate(initial);
        }

        let adaptive = self.adaptive_coupling.as_ref().unwrap();
        let time_config = &self.integrator.time_config;
        let params = self.integrator.vector_field.parameters.clone();

        let mut trajectory = Vec::new();
        let mut current_state = initial;
        let mut current_time = time_config.t0;

        trajectory.push((current_time, current_state));
        self.observer.observe(current_state);

        // Integrate with time-varying coupling
        while current_time < time_config.t_final {
            // Compute adaptive coupling at current time
            let coupling_t = adaptive.compute_coupling(current_time);

            // Create temporary vector field with updated coupling
            let vf_t = VectorField::new(coupling_t, params.clone());

            // Perform Heun step manually
            let k1 = vf_t.evaluate(&current_state);
            let predictor = current_state + k1 * time_config.dt;
            let k2 = vf_t.evaluate(&predictor);
            let next_state = current_state + (k1 + k2) * (time_config.dt * 0.5);

            // Check for non-finite values
            if !next_state.is_valid() {
                break;
            }

            current_state = next_state;
            current_time += time_config.dt;

            trajectory.push((current_time, current_state));
            self.observer.observe(current_state);
        }

        trajectory
    }

    /// Integrate a full trajectory (uses adaptive if available)
    pub fn integrate(&mut self, initial: State5D) -> Vec<(f64, State5D)> {
        if self.adaptive_coupling.is_some() {
            self.integrate_adaptive(initial)
        } else {
            let traj = self.integrator.integrate(initial);
            // Observe all states in trajectory
            for (_, state) in &traj {
                self.observer.observe(*state);
            }
            traj
        }
    }

    /// Get reference to the trajectory observer
    pub fn observer(&self) -> &TrajectoryObserver {
        &self.observer
    }

    /// Get mutable reference to the trajectory observer
    pub fn observer_mut(&mut self) -> &mut TrajectoryObserver {
        &mut self.observer
    }

    /// Get reference to the integrator
    pub fn integrator(&self) -> &Integrator {
        &self.integrator
    }

    /// Get reference to adaptive coupling if present
    pub fn adaptive_coupling(&self) -> Option<&AdaptiveCoupling> {
        self.adaptive_coupling.as_ref()
    }

    /// Reset the cognitive state
    pub fn reset(&mut self) {
        self.observer.clear();
        if let Some(adaptive) = &mut self.adaptive_coupling {
            adaptive.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resonance_field::OscillatoryResonanceField;
    use core_5d::{
        integration::TimeConfig, CouplingMatrix, CouplingType, SystemParameters, VectorField,
    };

    #[test]
    fn cognitive_simulator_step() {
        let coupling = CouplingMatrix::identity();
        let params = SystemParameters::zero();
        let vf = VectorField::new(coupling, params);
        let time_config = TimeConfig::new(0.01, 0.0, 1.0);
        let integrator = Integrator::new(vf, time_config);
        let observer = TrajectoryObserver::new(100);

        let mut sim = CognitiveSimulator::new(integrator, observer);

        let initial = State5D::new(1.0, 0.0, 0.0, 0.0, 0.0);
        let next = sim.step(initial, 0.01);

        assert!(next.get(0).is_finite());
        assert_eq!(sim.observer().history().len(), 1);
    }

    #[test]
    fn cognitive_simulator_integrate() {
        let coupling = CouplingMatrix::identity();
        let mut params = SystemParameters::zero();
        params.intrinsic_rates = [-0.1, 0.0, 0.0, 0.0, 0.0];
        let vf = VectorField::new(coupling, params);
        let time_config = TimeConfig::new(0.1, 0.0, 2.0);
        let integrator = Integrator::new(vf, time_config);
        let observer = TrajectoryObserver::new(100);

        let mut sim = CognitiveSimulator::new(integrator, observer);

        let initial = State5D::new(1.0, 0.0, 0.0, 0.0, 0.0);
        let trajectory = sim.integrate(initial);

        assert!(trajectory.len() > 1);
    }

    #[test]
    fn cognitive_simulator_adaptive_integration() {
        let mut base_coupling = CouplingMatrix::zero();
        base_coupling.set(0, 1, 0.5, CouplingType::Linear);
        let mut params = SystemParameters::zero();
        params.intrinsic_rates = [-0.1, 0.0, 0.0, 0.0, 0.0];

        let vf = VectorField::new(base_coupling.clone(), params);
        let time_config = TimeConfig::new(0.1, 0.0, 1.0);
        let integrator = Integrator::new(vf, time_config);
        let observer = TrajectoryObserver::new(100);

        let resonance = Box::new(OscillatoryResonanceField::new(0.2, 1.0, 0.0));
        let adaptive = AdaptiveCoupling::new(base_coupling, resonance);

        let mut sim = CognitiveSimulator::with_adaptive_coupling(integrator, observer, adaptive);

        let initial = State5D::new(1.0, 0.0, 0.0, 0.0, 0.0);
        let trajectory = sim.integrate_adaptive(initial);

        assert!(trajectory.len() > 1);
        assert!(sim.observer().history().len() > 0);
    }

    #[test]
    fn cognitive_simulator_reset() {
        let coupling = CouplingMatrix::identity();
        let params = SystemParameters::zero();
        let vf = VectorField::new(coupling, params);
        let time_config = TimeConfig::new(0.1, 0.0, 1.0);
        let integrator = Integrator::new(vf, time_config);
        let observer = TrajectoryObserver::new(100);

        let mut sim = CognitiveSimulator::new(integrator, observer);

        let initial = State5D::new(1.0, 0.0, 0.0, 0.0, 0.0);
        let _ = sim.integrate(initial);
        assert!(sim.observer().history().len() > 0);

        sim.reset();
        assert_eq!(sim.observer().history().len(), 0);
    }
}
