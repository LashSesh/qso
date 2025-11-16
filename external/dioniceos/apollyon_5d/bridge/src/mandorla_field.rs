// Mandorla-based resonance field implementation for bridge integration

use crate::resonance_field::ResonanceField;
use metatron::fields::MandorlaField;

/// Resonance field based on Metatron's Mandorla geometry
///
/// This wraps the MandorlaField from metatron to provide resonance-based
/// modulation for 5D coupling matrices. The resonance is computed based on
/// the similarity between node states mapped through Metatron geometry.
pub struct MandorlaResonanceField {
    field: MandorlaField,
    base_frequency: f64,
    amplitude: f64,
    /// Mapping from 5D indices to Metatron node indices
    node_mapping: [usize; 5],
}

impl MandorlaResonanceField {
    /// Create a new Mandorla resonance field
    pub fn new(threshold: f64, alpha: f64, beta: f64) -> Self {
        Self {
            field: MandorlaField::new(threshold, alpha, beta),
            base_frequency: 1.0,
            amplitude: 0.2,
            node_mapping: [0, 1, 2, 3, 4], // Default: first 5 Metatron nodes
        }
    }

    /// Create with default parameters
    pub fn default_params() -> Self {
        Self::new(0.985, 0.5, 0.5)
    }

    /// Set the oscillation parameters
    pub fn with_oscillation(mut self, frequency: f64, amplitude: f64) -> Self {
        self.base_frequency = frequency;
        self.amplitude = amplitude;
        self
    }

    /// Set custom node mapping from 5D to Metatron nodes
    pub fn with_node_mapping(mut self, mapping: [usize; 5]) -> Self {
        self.node_mapping = mapping;
        self
    }

    /// Get the current resonance value
    pub fn resonance(&self) -> f64 {
        self.field.resonance
    }

    /// Compute time-dependent resonance modulation
    fn time_modulation(&self, t: f64) -> f64 {
        1.0 + self.amplitude * (2.0 * std::f64::consts::PI * self.base_frequency * t).sin()
    }

    /// Compute spatial modulation based on node geometry
    fn spatial_modulation(&self, node_i: usize, node_j: usize) -> f64 {
        // Map 5D indices to Metatron nodes
        let metatron_i = self.node_mapping[node_i];
        let metatron_j = self.node_mapping[node_j];

        // Compute geometric distance in Metatron space
        // For now, use a simple distance-based modulation
        let dist = ((metatron_i as f64 - metatron_j as f64).abs() / 13.0).min(1.0);

        // Stronger coupling for nearby nodes
        1.0 + 0.5 * (1.0 - dist) * self.field.resonance
    }
}

impl ResonanceField for MandorlaResonanceField {
    fn modulation(&self, t: f64, node_i: usize, node_j: usize) -> f64 {
        let time_mod = self.time_modulation(t);
        let spatial_mod = self.spatial_modulation(node_i, node_j);

        // Combine temporal and spatial modulation
        time_mod * spatial_mod
    }

    fn modulation_gradient(&self, t: f64, _node_i: usize, _node_j: usize) -> f64 {
        // Gradient of time modulation
        let freq = self.base_frequency;
        2.0 * std::f64::consts::PI
            * freq
            * self.amplitude
            * (2.0 * std::f64::consts::PI * freq * t).cos()
    }

    fn reset(&mut self) {
        self.field.clear_inputs();
        self.field.history.clear();
        self.field.resonance = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mandorla_field_creation() {
        let field = MandorlaResonanceField::default_params();
        assert_eq!(field.base_frequency, 1.0);
        assert_eq!(field.amplitude, 0.2);
    }

    #[test]
    fn mandorla_modulation() {
        let mut field = MandorlaResonanceField::default_params();
        // Add some inputs to generate non-zero resonance
        field.field.add_input(vec![1.0, 0.5, 0.3]);
        field.field.add_input(vec![0.9, 0.6, 0.4]);
        field.field.calc_resonance();

        let mod_0 = field.modulation(0.0, 0, 1);
        assert!(mod_0 > 0.5 && mod_0 < 3.0);

        // Different nodes should have different modulation
        let mod_1 = field.modulation(0.0, 0, 4);
        assert!((mod_0 - mod_1).abs() > 0.01);
    }

    #[test]
    fn mandorla_time_variation() {
        let mut field = MandorlaResonanceField::default_params().with_oscillation(1.0, 0.5); // Higher amplitude, frequency 1 Hz

        // Add inputs to generate resonance
        field.field.add_input(vec![1.0, 0.5]);
        field.field.add_input(vec![0.9, 0.6]);
        field.field.calc_resonance();

        let mod_t0 = field.modulation(0.0, 0, 1); // At t=0, sin(0)=0
        let mod_t1 = field.modulation(0.25, 0, 1); // At t=0.25, sin(π/2)=1

        // Modulation should vary with time due to oscillation
        // At t=0, time_mod = 1.0 + 0.5*0 = 1.0
        // At t=0.25, time_mod = 1.0 + 0.5*1 = 1.5
        // So difference should be substantial
        assert!((mod_t0 - mod_t1).abs() > 0.2);
    }

    #[test]
    fn mandorla_reset() {
        let mut field = MandorlaResonanceField::default_params();
        field.reset();
        assert!(field.field.history.is_empty());
        assert_eq!(field.field.resonance, 0.0);
    }
}
