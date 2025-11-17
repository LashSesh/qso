use nalgebra::{DMatrix, DVector, SymmetricEigen};
use num_complex::Complex64;

use crate::hamiltonian::MetatronHamiltonian;
use crate::quantum::state::{QuantumState, StateVector};

/// Output of the Lanczos tridiagonalisation process.
#[derive(Clone, Debug)]
pub struct LanczosResult {
    pub basis: Vec<StateVector>,
    pub alpha: Vec<f64>,
    pub beta: Vec<f64>,
}

impl LanczosResult {
    pub fn dimension(&self) -> usize {
        self.alpha.len()
    }

    pub fn tridiagonal(&self) -> DMatrix<f64> {
        let dim = self.dimension();
        let mut matrix = DMatrix::<f64>::zeros(dim, dim);
        for i in 0..dim {
            matrix[(i, i)] = self.alpha[i];
        }
        for i in 0..(dim.saturating_sub(1)) {
            let value = self.beta[i];
            matrix[(i, i + 1)] = value;
            matrix[(i + 1, i)] = value;
        }
        matrix
    }
}

/// Krylov projection container storing Lanczos data and initial coefficients.
#[derive(Clone, Debug)]
pub struct KrylovProjection {
    pub lanczos: LanczosResult,
}

/// Result of evolving a state via the Krylov approximation.
#[derive(Clone, Debug)]
pub struct KrylovEvolution {
    pub state: QuantumState,
    pub residual_norm: f64,
}

pub fn lanczos_tridiagonalisation(
    hamiltonian: &MetatronHamiltonian,
    initial: &QuantumState,
    dimension: usize,
    tolerance: f64,
) -> LanczosResult {
    let mut basis = Vec::new();
    let mut alpha = Vec::new();
    let mut beta = Vec::new();

    let mut current = *initial.amplitudes();
    let norm = current.norm();
    if norm > 0.0 {
        current *= Complex64::new(1.0 / norm, 0.0);
    } else {
        current[0] = Complex64::new(1.0, 0.0);
    }
    basis.push(current);

    let mut previous = StateVector::zeros();
    let mut previous_beta = 0.0;
    let h = hamiltonian.as_complex_operator();

    for iteration in 0..dimension {
        let mut w = h * current;
        if iteration > 0 {
            w -= previous * Complex64::new(previous_beta, 0.0);
        }

        let alpha_value = current.dotc(&w).re;
        w -= current * Complex64::new(alpha_value, 0.0);
        alpha.push(alpha_value);

        if iteration + 1 >= dimension {
            break;
        }

        let beta_value = w.norm();
        if beta_value < tolerance {
            break;
        }

        beta.push(beta_value);
        previous = current;
        current = w * Complex64::new(1.0 / beta_value, 0.0);
        basis.push(current);
        previous_beta = beta_value;
    }

    LanczosResult { basis, alpha, beta }
}

pub fn krylov_projection(
    hamiltonian: &MetatronHamiltonian,
    initial: &QuantumState,
    dimension: usize,
    tolerance: f64,
) -> KrylovProjection {
    let lanczos = lanczos_tridiagonalisation(hamiltonian, initial, dimension, tolerance);
    KrylovProjection { lanczos }
}

impl KrylovProjection {
    pub fn evolve(&self, time: f64) -> KrylovEvolution {
        let dim = self.lanczos.dimension();
        let tridiagonal = self.lanczos.tridiagonal();
        let eigen = SymmetricEigen::new(tridiagonal);
        let eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;

        let mut e1 = DVector::<f64>::zeros(dim);
        e1[0] = 1.0;
        let coefficients = eigenvectors.transpose() * e1;

        let mut rotated = vec![Complex64::new(0.0, 0.0); dim];
        for k in 0..dim {
            let phase = Complex64::from_polar(1.0, -eigenvalues[k] * time);
            let weight = Complex64::new(coefficients[k], 0.0) * phase;
            for i in 0..dim {
                rotated[i] += weight * eigenvectors[(i, k)];
            }
        }

        let mut vector = StateVector::zeros();
        for (coeff, basis_vector) in rotated.iter().zip(self.lanczos.basis.iter()) {
            vector += *basis_vector * *coeff;
        }

        let state = QuantumState::from_vector(vector, true);
        let residual_norm = self
            .lanczos
            .beta
            .last()
            .map(|beta| beta * rotated.last().unwrap().norm())
            .unwrap_or(0.0);

        KrylovEvolution {
            state,
            residual_norm,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::metatron::MetatronGraph;
    use crate::params::QSOParameters;

    #[test]
    fn krylov_matches_exact_evolution() {
        let params = QSOParameters::default();
        let graph = MetatronGraph::new();
        let hamiltonian = MetatronHamiltonian::new(&graph, &params);
        let initial = QuantumState::basis_state(0).unwrap();
        let projection = krylov_projection(&hamiltonian, &initial, 6, 1e-10);
        let approx = projection.evolve(0.25);
        let exact = hamiltonian.evolve_state(&initial, 0.25);
        let diff = approx.state.amplitudes() - exact.amplitudes();
        let error = diff.norm();
        assert!(error < 1e-6);
    }
}
