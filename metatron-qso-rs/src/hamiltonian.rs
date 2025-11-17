use nalgebra::{SMatrix, SymmetricEigen};
use num_complex::Complex64;
use serde::Serialize;

use crate::graph::metatron::MetatronGraph;
use crate::params::QSOParameters;
use crate::quantum::operator::{OperatorMatrix, QuantumOperator};
use crate::quantum::state::{METATRON_DIMENSION, QuantumState, StateVector};

/// Real-valued Hamiltonian matrix type alias.
pub type HamiltonianMatrix = SMatrix<f64, 13, 13>;

/// Spectral summary of the Metatron Hamiltonian.
#[derive(Clone, Debug, Serialize)]
pub struct SpectrumInfo {
    pub eigenvalues: Vec<f64>,
    pub ground_state_energy: f64,
    pub energy_gap: f64,
    pub max_energy: f64,
    pub energy_spread: f64,
}

/// Tight-binding Hamiltonian on the Metatron Cube graph.
pub struct MetatronHamiltonian {
    matrix: HamiltonianMatrix,
    eigenvalues: [f64; METATRON_DIMENSION],
    eigenvectors: Vec<StateVector>,
}

impl MetatronHamiltonian {
    /// Construct the Hamiltonian H = -J·L + diag(ε).
    pub fn new(graph: &MetatronGraph, params: &QSOParameters) -> Self {
        let laplacian = graph.laplacian_matrix();
        let mut matrix = HamiltonianMatrix::zeros();
        for i in 0..METATRON_DIMENSION {
            for j in 0..METATRON_DIMENSION {
                matrix[(i, j)] = -params.j * laplacian[(i, j)];
            }
            matrix[(i, i)] += params.epsilon[i];
        }

        let eigen = SymmetricEigen::new(matrix);
        let eigenvalues_vec = eigen.eigenvalues.data.as_slice().to_vec();
        let eigenvectors_matrix = eigen.eigenvectors;

        // Create indices sorted by eigenvalue (ascending order, so [0] is minimum/ground state)
        let mut indices: Vec<usize> = (0..METATRON_DIMENSION).collect();
        indices.sort_by(|&a, &b| eigenvalues_vec[a].partial_cmp(&eigenvalues_vec[b]).unwrap());

        // Reorder eigenvalues and eigenvectors so that index 0 is the ground state
        let eigenvalues: [f64; METATRON_DIMENSION] = indices
            .iter()
            .map(|&i| eigenvalues_vec[i])
            .collect::<Vec<_>>()
            .try_into()
            .expect("expected 13 eigenvalues");

        let eigenvectors: Vec<StateVector> = indices
            .iter()
            .map(|&col| {
                let mut vector = StateVector::zeros();
                for row in 0..METATRON_DIMENSION {
                    vector[row] = Complex64::new(eigenvectors_matrix[(row, col)], 0.0);
                }
                vector
            })
            .collect();

        Self {
            matrix,
            eigenvalues,
            eigenvectors,
        }
    }

    /// Construct Hamiltonian directly from a matrix
    pub fn from_matrix(matrix: HamiltonianMatrix) -> Self {
        let eigen = SymmetricEigen::new(matrix);
        let eigenvalues_vec = eigen.eigenvalues.data.as_slice().to_vec();
        let eigenvectors_matrix = eigen.eigenvectors;

        // Create indices sorted by eigenvalue (ascending order, so [0] is minimum/ground state)
        let mut indices: Vec<usize> = (0..METATRON_DIMENSION).collect();
        indices.sort_by(|&a, &b| eigenvalues_vec[a].partial_cmp(&eigenvalues_vec[b]).unwrap());

        // Reorder eigenvalues and eigenvectors so that index 0 is the ground state
        let eigenvalues: [f64; METATRON_DIMENSION] = indices
            .iter()
            .map(|&i| eigenvalues_vec[i])
            .collect::<Vec<_>>()
            .try_into()
            .expect("expected 13 eigenvalues");

        let eigenvectors: Vec<StateVector> = indices
            .iter()
            .map(|&col| {
                let mut vector = StateVector::zeros();
                for row in 0..METATRON_DIMENSION {
                    vector[row] = Complex64::new(eigenvectors_matrix[(row, col)], 0.0);
                }
                vector
            })
            .collect();

        Self {
            matrix,
            eigenvalues,
            eigenvectors,
        }
    }

    /// Get ground state energy
    pub fn ground_state_energy(&self) -> f64 {
        self.eigenvalues[0]
    }

    /// Access raw Hamiltonian matrix.
    pub fn matrix(&self) -> &HamiltonianMatrix {
        &self.matrix
    }

    /// Access eigenvalues.
    pub fn eigenvalues(&self) -> &[f64; METATRON_DIMENSION] {
        &self.eigenvalues
    }

    /// Access eigenvectors as column-major state vectors.
    pub fn eigenvectors(&self) -> &[StateVector] {
        &self.eigenvectors
    }

    /// Project an arbitrary state onto the Hamiltonian eigenbasis.
    pub fn project_onto_eigenbasis(&self, state: &QuantumState) -> Vec<Complex64> {
        let amplitudes = state.amplitudes();
        self.eigenvectors
            .iter()
            .map(|vec| vec.dotc(amplitudes))
            .collect()
    }

    /// Return the Hamiltonian as a complex operator matrix.
    pub fn as_complex_operator(&self) -> OperatorMatrix {
        OperatorMatrix::from_fn(|i, j| Complex64::new(self.matrix[(i, j)], 0.0))
    }

    /// Retrieve nth eigenstate as (energy, QuantumState).
    pub fn eigenstate(&self, index: usize) -> Option<(f64, QuantumState)> {
        self.eigenvalues.get(index).map(|&energy| {
            (
                energy,
                QuantumState::from_vector(self.eigenvectors[index], false),
            )
        })
    }

    /// Ground-state wavefunction.
    pub fn ground_state(&self) -> QuantumState {
        QuantumState::from_vector(self.eigenvectors[0], false)
    }

    /// Time-evolution operator U(t) = exp(-iHt).
    pub fn time_evolution_operator(&self, time: f64) -> QuantumOperator {
        let mut matrix = OperatorMatrix::zeros();
        for (energy, eigenvector) in self.eigenvalues.iter().zip(self.eigenvectors.iter()) {
            let phase = Complex64::from_polar(1.0, -energy * time);
            let projector = *eigenvector * eigenvector.adjoint();
            matrix += projector * phase;
        }
        QuantumOperator::from_matrix(matrix)
    }

    /// Evolve a quantum state |ψ(0)⟩ to time t.
    pub fn evolve_state(&self, state: &QuantumState, time: f64) -> QuantumState {
        let operator = self.time_evolution_operator(time);
        state.apply(&operator)
    }

    /// Derive spectral diagnostics for reporting.
    pub fn spectrum_info(&self) -> SpectrumInfo {
        let eigenvalues = self.eigenvalues.to_vec();
        let ground_state_energy = eigenvalues[0];
        let max_energy = *eigenvalues.last().unwrap();
        let energy_gap = if eigenvalues.len() > 1 {
            eigenvalues[1] - eigenvalues[0]
        } else {
            0.0
        };
        let energy_spread = max_energy - ground_state_energy;

        SpectrumInfo {
            eigenvalues,
            ground_state_energy,
            energy_gap,
            max_energy,
            energy_spread,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::QSOParameters;

    #[test]
    fn ground_state_is_normalized() {
        let graph = MetatronGraph::new();
        let params = QSOParameters::default();
        let hamiltonian = MetatronHamiltonian::new(&graph, &params);
        let ground = hamiltonian.ground_state();
        assert!(ground.is_normalized(1e-10));
    }
}
