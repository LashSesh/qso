use nalgebra::SymmetricEigen;
use num_complex::Complex64;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::graph::metatron::MetatronGraph;
use crate::params::QSOParameters;

/// Network of 13 coupled DTL resonators placed on the Metatron Cube graph.
pub struct DTLResonatorNetwork {
    graph: MetatronGraph,
    params: QSOParameters,
    phases: [f64; crate::quantum::METATRON_DIMENSION],
}

impl DTLResonatorNetwork {
    /// Instantiate the network with random initial phases.
    pub fn new(graph: MetatronGraph, params: QSOParameters) -> Self {
        let mut rng = SmallRng::from_entropy();
        let mut phases = [0.0; crate::quantum::METATRON_DIMENSION];
        for phase in phases.iter_mut() {
            *phase = rng.gen_range(0.0..std::f64::consts::TAU);
        }
        Self {
            graph,
            params,
            phases,
        }
    }

    /// Replace the internal phase vector (useful for deterministic tests).
    pub fn with_phases(mut self, phases: [f64; crate::quantum::METATRON_DIMENSION]) -> Self {
        self.phases = phases;
        self
    }

    /// Compute time-derivatives dφ/dt for the current phase configuration.
    pub fn derivative(
        &self,
        phases: &[f64; crate::quantum::METATRON_DIMENSION],
        _t: f64,
    ) -> [f64; crate::quantum::METATRON_DIMENSION] {
        let adjacency = self.graph.adjacency_matrix();
        let mut derivatives = self.params.omega;
        for i in 0..crate::quantum::METATRON_DIMENSION {
            let mut coupling = 0.0;
            for j in 0..crate::quantum::METATRON_DIMENSION {
                if adjacency[(i, j)] != 0.0 {
                    coupling += self.params.kappa * (phases[j] - phases[i]).sin();
                }
            }
            derivatives[i] += coupling;
        }
        derivatives
    }

    /// Integrate the Kuramoto system with explicit Euler integration.
    pub fn integrate(
        &mut self,
        t_span: (f64, f64),
        dt: f64,
    ) -> (Vec<f64>, Vec<[f64; crate::quantum::METATRON_DIMENSION]>) {
        let (t_start, t_end) = t_span;
        let steps = ((t_end - t_start) / dt).ceil() as usize;
        let mut times = Vec::with_capacity(steps);
        let mut history = Vec::with_capacity(steps);
        let mut phases = self.phases;
        let mut time = t_start;

        for _ in 0..steps {
            times.push(time);
            history.push(phases);
            let derivatives = self.derivative(&phases, time);
            for i in 0..crate::quantum::METATRON_DIMENSION {
                phases[i] += derivatives[i] * dt;
            }
            time += dt;
        }

        self.phases = phases;
        (times, history)
    }

    /// Kuramoto order parameter r ∈ \[0,1\].
    pub fn order_parameter(&self, phases: &[f64; crate::quantum::METATRON_DIMENSION]) -> f64 {
        let sum: Complex64 = phases
            .iter()
            .map(|&phi| Complex64::from_polar(1.0, phi))
            .sum();
        (sum / (crate::quantum::METATRON_DIMENSION as f64)).norm()
    }

    /// Convert instantaneous phases to DTL amplitude values in \[0,1\].
    pub fn phases_to_dtl(
        &self,
        phases: &[f64; crate::quantum::METATRON_DIMENSION],
    ) -> [f64; crate::quantum::METATRON_DIMENSION] {
        let mut values = [0.0; crate::quantum::METATRON_DIMENSION];
        for (i, &phi) in phases.iter().enumerate() {
            values[i] = (1.0 + phi.cos()) / 2.0;
        }
        values
    }

    /// Approximate synchronization threshold κ_c ≈ 2|ω_max| / λ₂.
    pub fn synchronization_threshold(&self) -> f64 {
        let laplacian = self.graph.laplacian_matrix();
        let eigen = SymmetricEigen::new(laplacian);
        let mut eigenvalues = eigen.eigenvalues.data.as_slice().to_vec();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lambda_2 = eigenvalues.get(1).copied().unwrap_or(0.0);
        let omega_max = self
            .params
            .omega
            .iter()
            .map(|w| w.abs())
            .fold(0.0, f64::max);
        if lambda_2 > 0.0 {
            2.0 * omega_max / lambda_2
        } else {
            f64::INFINITY
        }
    }

    /// Access internal phases (useful for analysis).
    pub fn phases(&self) -> &[f64; crate::quantum::METATRON_DIMENSION] {
        &self.phases
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::QSOParameters;

    #[test]
    fn order_parameter_with_identical_phases_is_one() {
        let graph = MetatronGraph::new();
        let params = QSOParameters::default();
        let network = DTLResonatorNetwork::new(graph, params);
        let phases = [0.0; crate::quantum::METATRON_DIMENSION];
        assert!((network.order_parameter(&phases) - 1.0).abs() < 1e-12);
    }
}
