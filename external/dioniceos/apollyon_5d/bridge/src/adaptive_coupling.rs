// Adaptive coupling: Resonance-modulated coupling matrices

use crate::resonance_field::ResonanceField;
use core_5d::CouplingMatrix;

/// Adaptive coupling that modulates base coupling strengths using resonance fields
pub struct AdaptiveCoupling {
    base_coupling: CouplingMatrix,
    resonance: Box<dyn ResonanceField>,
}

impl std::fmt::Debug for AdaptiveCoupling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveCoupling")
            .field("base_coupling", &self.base_coupling)
            .field("resonance", &"<ResonanceField>")
            .finish()
    }
}

impl AdaptiveCoupling {
    /// Create a new adaptive coupling system
    pub fn new(base_coupling: CouplingMatrix, resonance: Box<dyn ResonanceField>) -> Self {
        Self {
            base_coupling,
            resonance,
        }
    }

    /// Compute the time-varying coupling matrix at time t
    pub fn compute_coupling(&self, t: f64) -> CouplingMatrix {
        let mut modulated = self.base_coupling.clone();

        // Apply resonance modulation to each coupling strength
        for i in 0..5 {
            for j in 0..5 {
                let base_value = self.base_coupling.get_strength(i, j);
                let modulation = self.resonance.modulation(t, i, j);
                modulated.set(
                    i,
                    j,
                    base_value * modulation,
                    self.base_coupling.get_type(i, j),
                );
            }
        }

        modulated
    }

    /// Get reference to base coupling
    pub fn base_coupling(&self) -> &CouplingMatrix {
        &self.base_coupling
    }

    /// Reset resonance field state
    pub fn reset(&mut self) {
        self.resonance.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resonance_field::ConstantResonanceField;
    use core_5d::CouplingType;

    #[test]
    fn adaptive_coupling_constant_modulation() {
        let mut base = CouplingMatrix::zero();
        base.set(0, 1, 0.5, CouplingType::Linear);
        base.set(1, 0, -0.3, CouplingType::Linear);

        let resonance = Box::new(ConstantResonanceField::new(2.0));
        let adaptive = AdaptiveCoupling::new(base, resonance);

        let modulated = adaptive.compute_coupling(0.0);
        assert_eq!(modulated.get_strength(0, 1), 1.0); // 0.5 * 2.0
        assert_eq!(modulated.get_strength(1, 0), -0.6); // -0.3 * 2.0
    }

    #[test]
    fn adaptive_coupling_preserves_types() {
        let mut base = CouplingMatrix::zero();
        base.set(0, 1, 0.5, CouplingType::Linear);
        base.set(1, 2, 0.3, CouplingType::Quadratic);

        let resonance = Box::new(ConstantResonanceField::new(1.5));
        let adaptive = AdaptiveCoupling::new(base, resonance);

        let modulated = adaptive.compute_coupling(0.0);
        assert_eq!(modulated.get_type(0, 1), CouplingType::Linear);
        assert_eq!(modulated.get_type(1, 2), CouplingType::Quadratic);
    }
}
