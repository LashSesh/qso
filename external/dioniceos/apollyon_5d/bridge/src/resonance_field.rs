// Resonance field trait for modulating coupling dynamics

/// Trait for resonance fields that can modulate coupling strengths
/// between components in the 5D dynamical system.
///
/// This provides the interface between Metatron's geometric resonance
/// fields and the 5D framework's coupling matrices.
pub trait ResonanceField: Send + Sync {
    /// Compute the modulation factor for coupling between nodes i and j at time t
    ///
    /// # Arguments
    /// * `t` - Current time
    /// * `node_i` - Index of first node (0-4 for 5D system, can map to Metatron nodes)
    /// * `node_j` - Index of second node
    ///
    /// # Returns
    /// Multiplicative modulation factor (typically in range [0.5, 2.0])
    fn modulation(&self, t: f64, node_i: usize, node_j: usize) -> f64;

    /// Optional: Compute gradient of modulation for adaptive control
    fn modulation_gradient(&self, _t: f64, _node_i: usize, _node_j: usize) -> f64 {
        0.0
    }

    /// Optional: Reset internal state (for adaptive fields)
    fn reset(&mut self) {}
}

/// Simple constant resonance field (no modulation)
#[derive(Debug, Clone)]
pub struct ConstantResonanceField {
    pub factor: f64,
}

impl ConstantResonanceField {
    pub fn new(factor: f64) -> Self {
        Self { factor }
    }
}

impl Default for ConstantResonanceField {
    fn default() -> Self {
        Self { factor: 1.0 }
    }
}

impl ResonanceField for ConstantResonanceField {
    fn modulation(&self, _t: f64, _node_i: usize, _node_j: usize) -> f64 {
        self.factor
    }
}

/// Time-varying sinusoidal resonance field
#[derive(Debug, Clone)]
pub struct OscillatoryResonanceField {
    pub amplitude: f64,
    pub frequency: f64,
    pub phase: f64,
}

impl OscillatoryResonanceField {
    pub fn new(amplitude: f64, frequency: f64, phase: f64) -> Self {
        Self {
            amplitude,
            frequency,
            phase,
        }
    }
}

impl ResonanceField for OscillatoryResonanceField {
    fn modulation(&self, t: f64, _node_i: usize, _node_j: usize) -> f64 {
        1.0 + self.amplitude * (2.0 * std::f64::consts::PI * self.frequency * t + self.phase).sin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_field_modulation() {
        let field = ConstantResonanceField::new(1.5);
        assert_eq!(field.modulation(0.0, 0, 1), 1.5);
        assert_eq!(field.modulation(10.0, 2, 3), 1.5);
    }

    #[test]
    fn oscillatory_field_modulation() {
        let field = OscillatoryResonanceField::new(0.2, 1.0, 0.0);
        let m0 = field.modulation(0.0, 0, 1);
        assert!((m0 - 1.0).abs() < 0.01);

        let m_quarter = field.modulation(0.25, 0, 1);
        assert!(m_quarter > 1.0);
    }
}
