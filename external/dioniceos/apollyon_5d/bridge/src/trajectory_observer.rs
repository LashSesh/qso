// Trajectory observer: Monitor 5D trajectories and provide feedback

use core_5d::State5D;

/// Observes 5D trajectories and extracts features for cognitive feedback
#[derive(Debug, Clone)]
pub struct TrajectoryObserver {
    history: Vec<State5D>,
    max_history: usize,
}

impl TrajectoryObserver {
    /// Create a new trajectory observer with specified history length
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Vec::new(),
            max_history,
        }
    }

    /// Record a new state observation
    pub fn observe(&mut self, state: State5D) {
        self.history.push(state);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Get the current trajectory history
    pub fn history(&self) -> &[State5D] {
        &self.history
    }

    /// Compute velocity (finite difference)
    pub fn velocity(&self) -> Option<State5D> {
        if self.history.len() < 2 {
            return None;
        }
        let n = self.history.len();
        let current = self.history[n - 1];
        let previous = self.history[n - 2];

        Some(State5D::new(
            current.get(0) - previous.get(0),
            current.get(1) - previous.get(1),
            current.get(2) - previous.get(2),
            current.get(3) - previous.get(3),
            current.get(4) - previous.get(4),
        ))
    }

    /// Compute acceleration (second finite difference)
    pub fn acceleration(&self) -> Option<State5D> {
        if self.history.len() < 3 {
            return None;
        }
        let n = self.history.len();
        let current = self.history[n - 1];
        let prev1 = self.history[n - 2];
        let prev2 = self.history[n - 3];

        Some(State5D::new(
            current.get(0) - 2.0 * prev1.get(0) + prev2.get(0),
            current.get(1) - 2.0 * prev1.get(1) + prev2.get(1),
            current.get(2) - 2.0 * prev1.get(2) + prev2.get(2),
            current.get(3) - 2.0 * prev1.get(3) + prev2.get(3),
            current.get(4) - 2.0 * prev1.get(4) + prev2.get(4),
        ))
    }

    /// Estimate trajectory energy (sum of squared components)
    pub fn energy(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let state = self.history.last().unwrap();
        state.norm().powi(2)
    }

    /// Clear observation history
    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// Check if trajectory appears to be approaching a fixed point
    pub fn is_converging(&self, threshold: f64) -> bool {
        self.velocity()
            .map(|v| v.norm() < threshold)
            .unwrap_or(false)
    }
}

impl Default for TrajectoryObserver {
    fn default() -> Self {
        Self::new(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observer_records_history() {
        let mut obs = TrajectoryObserver::new(3);
        obs.observe(State5D::new(1.0, 0.0, 0.0, 0.0, 0.0));
        obs.observe(State5D::new(2.0, 0.0, 0.0, 0.0, 0.0));
        assert_eq!(obs.history().len(), 2);
    }

    #[test]
    fn observer_limits_history() {
        let mut obs = TrajectoryObserver::new(2);
        obs.observe(State5D::new(1.0, 0.0, 0.0, 0.0, 0.0));
        obs.observe(State5D::new(2.0, 0.0, 0.0, 0.0, 0.0));
        obs.observe(State5D::new(3.0, 0.0, 0.0, 0.0, 0.0));
        assert_eq!(obs.history().len(), 2);
        assert_eq!(obs.history()[0].get(0), 2.0);
    }

    #[test]
    fn velocity_calculation() {
        let mut obs = TrajectoryObserver::new(10);
        obs.observe(State5D::new(0.0, 0.0, 0.0, 0.0, 0.0));
        obs.observe(State5D::new(1.0, 2.0, 3.0, 4.0, 5.0));

        let vel = obs.velocity().unwrap();
        assert_eq!(vel.get(0), 1.0);
        assert_eq!(vel.get(1), 2.0);
    }

    #[test]
    fn convergence_detection() {
        let mut obs = TrajectoryObserver::new(10);
        obs.observe(State5D::new(1.0, 0.0, 0.0, 0.0, 0.0));
        obs.observe(State5D::new(1.001, 0.0, 0.0, 0.0, 0.0));

        assert!(obs.is_converging(0.01));
        assert!(!obs.is_converging(0.0001));
    }
}
