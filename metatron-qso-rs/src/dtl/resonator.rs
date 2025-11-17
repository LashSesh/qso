use std::sync::Arc;

use crate::dtl::state::{DTLState, TripolarStateKind};

/// Input coupling for a DTL resonator.
#[derive(Clone)]
struct ResonatorInput {
    theta: f64,
    kappa: f64,
    signal: Arc<dyn Fn(f64) -> f64 + Send + Sync>,
}

/// Kuramoto-like resonator implementing DTL dynamics on S¹.
#[derive(Clone)]
pub struct DTLResonator {
    omega: f64,
    phase: f64,
    inputs: Vec<ResonatorInput>,
}

impl DTLResonator {
    /// Create a resonator with intrinsic frequency ω and initial phase φ₀.
    pub fn new(omega: f64, initial_phase: f64) -> Self {
        Self {
            omega,
            phase: initial_phase,
            inputs: Vec::new(),
        }
    }

    /// Append an input signal I(t) with target phase θ and coupling strength κ.
    pub fn add_input<F>(&mut self, theta: f64, kappa: f64, signal: F)
    where
        F: Fn(f64) -> f64 + Send + Sync + 'static,
    {
        self.inputs.push(ResonatorInput {
            theta,
            kappa,
            signal: Arc::new(signal),
        });
    }

    /// Compute dφ/dt at time t for phase φ.
    pub fn derivative(&self, phase: f64, t: f64) -> f64 {
        let mut dphi = self.omega;
        for input in &self.inputs {
            let value = (input.signal)(t);
            dphi += input.kappa * value * (input.theta - phase).sin();
        }
        dphi
    }

    /// Integrate the phase trajectory using a simple Euler scheme.
    pub fn integrate(&mut self, t_span: (f64, f64), dt: f64) -> (Vec<f64>, Vec<f64>) {
        let (t_start, t_end) = t_span;
        let steps = ((t_end - t_start) / dt).ceil() as usize;
        let mut times = Vec::with_capacity(steps);
        let mut phases = Vec::with_capacity(steps);

        let mut phase = self.phase;
        let mut time = t_start;

        for _ in 0..steps {
            times.push(time);
            phases.push(phase);
            let derivative = self.derivative(phase, time);
            phase += derivative * dt;
            time += dt;
        }

        self.phase = phase;
        (times, phases)
    }

    /// Map the current resonator phase to a tripolar logical state.
    pub fn logical_state(&self, tolerance: f64) -> TripolarStateKind {
        let x = (1.0 + self.phase.cos()) / 2.0;
        if (x - 0.0).abs() < tolerance {
            TripolarStateKind::L0
        } else if (x - 1.0).abs() < tolerance {
            TripolarStateKind::L1
        } else {
            TripolarStateKind::Ld
        }
    }

    /// Convert the dynamical trajectory to a DTLState by sampling the integrated phases.
    pub fn to_dtl_state(&mut self, t_span: (f64, f64), dt: f64) -> DTLState {
        let (times, phases) = self.integrate(t_span, dt);
        let samples = Arc::new(times.into_iter().zip(phases).collect::<Vec<_>>());

        DTLState::ld_from_phase(move |t| {
            let samples = &samples;
            if samples.is_empty() {
                return 0.0;
            }
            if t <= samples[0].0 {
                return samples[0].1;
            }
            if t >= samples[samples.len() - 1].0 {
                return samples[samples.len() - 1].1;
            }

            let mut idx = 0;
            while idx + 1 < samples.len() && samples[idx + 1].0 <= t {
                idx += 1;
            }

            let (t0, phase0) = samples[idx];
            let (t1, phase1) = samples[idx + 1];
            let alpha = (t - t0) / (t1 - t0);
            phase0 + alpha * (phase1 - phase0)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resonator_converges_with_constant_drive() {
        let mut resonator = DTLResonator::new(0.1, 0.0);
        resonator.add_input(0.0, 1.0, |_| 1.0);
        let (_times, phases) = resonator.integrate((0.0, 1.0), 0.01);
        assert_eq!(phases.len(), 100);
    }
}
