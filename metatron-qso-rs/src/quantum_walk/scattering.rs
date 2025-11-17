use std::f64::consts::PI;

use num_complex::Complex64;
use serde::Serialize;

use crate::hamiltonian::MetatronHamiltonian;
use crate::quantum::operator::OperatorMatrix;
use crate::quantum::state::{METATRON_DIMENSION, StateVector};

/// Gaussian-smoothed density of states around an energy level.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct DensityOfStates {
    pub energy: f64,
    pub sigma: f64,
    pub value: f64,
}

/// Coupling channel utilised in the scattering-matrix construction.
#[derive(Clone, Debug)]
pub struct ScatteringChannel {
    pub vector: StateVector,
    pub coupling: f64,
}

impl ScatteringChannel {
    pub fn new(vector: StateVector, coupling: f64) -> Self {
        Self { vector, coupling }
    }
}

/// Result of a scattering analysis at a given probe energy.
#[derive(Clone, Debug)]
pub struct ScatteringAnalysis {
    pub energy: f64,
    pub eta: f64,
    pub density: DensityOfStates,
    pub matrix: OperatorMatrix,
}

impl ScatteringAnalysis {
    pub fn new(energy: f64, eta: f64, density: DensityOfStates, matrix: OperatorMatrix) -> Self {
        Self {
            energy,
            eta,
            density,
            matrix,
        }
    }
}

fn density_of_states(
    eigenvalues: &[f64; METATRON_DIMENSION],
    energy: f64,
    sigma: f64,
) -> DensityOfStates {
    let norm = 1.0 / (sigma * (2.0 * PI).sqrt());
    let mut value = 0.0;
    for &lambda in eigenvalues.iter() {
        let diff = energy - lambda;
        value += norm * (-0.5 * (diff / sigma).powi(2)).exp();
    }
    DensityOfStates {
        energy,
        sigma,
        value,
    }
}

fn build_coupling_matrix(channels: &[ScatteringChannel]) -> OperatorMatrix {
    let mut matrix = OperatorMatrix::zeros();
    for channel in channels {
        let projector = channel.vector * channel.vector.adjoint();
        matrix += projector * Complex64::new(channel.coupling, 0.0);
    }
    matrix
}

fn resolve_resolvent(hamiltonian: &MetatronHamiltonian, energy: f64, eta: f64) -> OperatorMatrix {
    let mut resolvent = OperatorMatrix::identity() * Complex64::new(energy, eta)
        - hamiltonian.as_complex_operator();
    if let Some(inverse) = resolvent.try_inverse() {
        return inverse;
    }
    for idx in 0..METATRON_DIMENSION {
        resolvent[(idx, idx)] += Complex64::new(0.0, 1e-9);
    }
    resolvent
        .try_inverse()
        .unwrap_or_else(OperatorMatrix::identity)
}

/// Compute the scattering matrix `S(E)` for a set of coupling channels.
pub fn scattering_matrix(
    hamiltonian: &MetatronHamiltonian,
    channels: &[ScatteringChannel],
    energy: f64,
    eta: f64,
    sigma: f64,
) -> ScatteringAnalysis {
    let density = density_of_states(hamiltonian.eigenvalues(), energy, sigma);
    let coupling = build_coupling_matrix(channels);
    let resolvent = resolve_resolvent(hamiltonian, energy, eta);

    let prefactor = Complex64::new(0.0, 2.0 * PI * density.value);
    let kernel = coupling * resolvent * coupling;
    let scattering = OperatorMatrix::identity() - kernel * prefactor;

    ScatteringAnalysis::new(energy, eta, density, scattering)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::metatron::MetatronGraph;
    use crate::params::QSOParameters;

    #[test]
    fn scattering_matrix_behaves() {
        let params = QSOParameters::default();
        let graph = MetatronGraph::new();
        let hamiltonian = MetatronHamiltonian::new(&graph, &params);
        let channel =
            ScatteringChannel::new(StateVector::from_element(Complex64::new(1.0, 0.0)), 0.2);
        let analysis = scattering_matrix(&hamiltonian, &[channel], 0.5, 0.05, 0.1);
        let trace = analysis.matrix.trace();
        assert!(trace.im.abs() <= 13.0);
    }
}
