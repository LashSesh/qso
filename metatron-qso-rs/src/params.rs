use serde::{Deserialize, Serialize};

use crate::quantum::METATRON_DIMENSION;

/// Global configuration for the Metatron QSO components.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QSOParameters {
    /// Coupling constant for the Laplacian contribution to the Hamiltonian.
    pub j: f64,
    /// On-site potentials εᵢ.
    pub epsilon: [f64; METATRON_DIMENSION],
    /// Intrinsic resonator frequencies ωᵢ.
    pub omega: [f64; METATRON_DIMENSION],
    /// Scalar Kuramoto coupling strength.
    pub kappa: f64,
    /// Dephasing rate γ for quantum walk mixing (0.0 = pure unitary).
    pub dephasing_rate: f64,
}

impl Default for QSOParameters {
    fn default() -> Self {
        Self {
            j: 1.0,
            epsilon: [0.0; METATRON_DIMENSION],
            omega: [0.0; METATRON_DIMENSION],
            kappa: 1.0,
            dephasing_rate: 0.0,
        }
    }
}

impl QSOParameters {
    /// Create parameters with optional overrides.
    pub fn new(
        j: f64,
        epsilon: [f64; METATRON_DIMENSION],
        omega: [f64; METATRON_DIMENSION],
        kappa: f64,
    ) -> Self {
        Self {
            j,
            epsilon,
            omega,
            kappa,
            dephasing_rate: 0.0,
        }
    }

    /// Create parameters with dephasing.
    pub fn with_dephasing(mut self, dephasing_rate: f64) -> Self {
        self.dephasing_rate = dephasing_rate;
        self
    }
}
