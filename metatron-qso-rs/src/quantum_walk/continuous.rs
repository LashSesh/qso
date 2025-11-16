use num_complex::Complex64;

use crate::hamiltonian::MetatronHamiltonian;
use crate::quantum::state::{METATRON_DIMENSION, QuantumState, StateVector};

/// Continuous-time quantum walk engine backed by the Metatron Hamiltonian.
pub struct ContinuousTimeQuantumWalk<'a> {
    hamiltonian: &'a MetatronHamiltonian,
    dephasing_rate: f64,
}

impl<'a> ContinuousTimeQuantumWalk<'a> {
    /// Create a new continuous-time quantum walk wrapper.
    pub fn new(hamiltonian: &'a MetatronHamiltonian) -> Self {
        Self {
            hamiltonian,
            dephasing_rate: 0.0,
        }
    }

    /// Create a quantum walk with dephasing.
    pub fn with_dephasing(hamiltonian: &'a MetatronHamiltonian, dephasing_rate: f64) -> Self {
        Self {
            hamiltonian,
            dephasing_rate,
        }
    }

    /// Construct a spectral propagator for a specific initial state.
    pub fn propagator(&self, initial: &QuantumState) -> SpectralPropagator<'a> {
        let overlaps = self.hamiltonian.project_onto_eigenbasis(initial);
        SpectralPropagator {
            hamiltonian: self.hamiltonian,
            overlaps,
            dephasing_rate: self.dephasing_rate,
        }
    }

    /// Exact evolution using the stored Hamiltonian exponentiation.
    pub fn evolve(&self, initial: &QuantumState, time: f64) -> QuantumState {
        self.hamiltonian.evolve_state(initial, time)
    }
}

/// Spectral propagator caching the eigenbasis overlap for repeated evaluations.
pub struct SpectralPropagator<'a> {
    hamiltonian: &'a MetatronHamiltonian,
    overlaps: Vec<Complex64>,
    dephasing_rate: f64,
}

impl<'a> SpectralPropagator<'a> {
    /// Evaluate the full quantum state at time `t`.
    pub fn state_at(&self, time: f64) -> QuantumState {
        let mut vector = StateVector::zeros();
        for ((&energy, eigenvector), overlap) in self
            .hamiltonian
            .eigenvalues()
            .iter()
            .zip(self.hamiltonian.eigenvectors().iter())
            .zip(self.overlaps.iter())
        {
            let phase = Complex64::from_polar(1.0, -energy * time);
            let contribution = eigenvector.clone() * (*overlap * phase);
            vector += contribution;
        }
        QuantumState::from_vector(vector, false)
    }

    /// Probability distribution across nodes at time `t`.
    pub fn probabilities_at(&self, time: f64) -> [f64; METATRON_DIMENSION] {
        if self.dephasing_rate == 0.0 {
            // Pure unitary evolution
            self.state_at(time).probabilities()
        } else {
            // Dephased evolution: mix unitary and stationary distributions
            let unitary_probs = self.state_at(time).probabilities();
            let stationary = self.time_average_distribution();
            let dephasing_factor = (-self.dephasing_rate * time).exp();

            let mut dephased = [0.0; METATRON_DIMENSION];
            for i in 0..METATRON_DIMENSION {
                dephased[i] =
                    dephasing_factor * unitary_probs[i] + (1.0 - dephasing_factor) * stationary[i];
            }
            dephased
        }
    }

    /// Long-time average (Cesàro mean) probability distribution.
    pub fn time_average_distribution(&self) -> [f64; METATRON_DIMENSION] {
        let mut distribution = [0.0; METATRON_DIMENSION];
        for (overlap, eigenvector) in self
            .overlaps
            .iter()
            .zip(self.hamiltonian.eigenvectors().iter())
        {
            let weight = overlap.norm_sqr();
            for idx in 0..METATRON_DIMENSION {
                distribution[idx] += weight * eigenvector[idx].norm_sqr();
            }
        }

        let norm: f64 = distribution.iter().sum();
        if norm > 0.0 {
            for value in &mut distribution {
                *value /= norm;
            }
        }
        distribution
    }

    /// Access cached overlaps (useful for diagnostics).
    pub fn overlaps(&self) -> &[Complex64] {
        &self.overlaps
    }
}
