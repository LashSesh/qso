/*!
 * Resonance Tensor Module - Multidimensional Resonance Dynamics
 *
 * This module defines a simple resonant tensor field used for simulating
 * the multidimensional resonance dynamics. It serves as a foundation for
 * modeling tripolar oscillatory fields and tensor dynamics in a post-symbolic
 * cognition engine.
 *
 * The core ResonanceTensorField manages a 3-dimensional grid of oscillators.
 * Each cell in the grid has amplitude A, frequency ω, and phase φ parameters.
 * At each time step, the resonance value is updated according to:
 *
 * ```text
 * R(t)[i,j,k] = A[i,j,k] * sin(ω[i,j,k] * t + φ[i,j,k])
 * ```
 *
 * External input can modulate either the amplitudes or the phase offsets.
 */

use ndarray::Array3;

/// A simple 3D resonance tensor field
///
/// Manages a 3D grid of oscillators with amplitude, frequency, and phase
/// parameters. Provides time evolution, coherence metrics, and singularity
/// detection based on gradient stabilization.
#[derive(Debug, Clone)]
pub struct ResonanceTensorField {
    /// Dimensions of the resonance tensor (Nx, Ny, Nz)
    pub shape: (usize, usize, usize),
    /// Initial amplitude for all cells
    pub initial_amplitude: f64,
    /// Initial frequency for all cells
    pub initial_frequency: f64,
    /// Initial phase offset (in radians) for all cells
    pub initial_phase: f64,
    /// Threshold on gradient norm below which a singularity is detected
    pub gradient_threshold: f64,
    /// Current time
    pub time: f64,
    /// Amplitude array
    amplitude: Array3<f64>,
    /// Frequency array
    frequency: Array3<f64>,
    /// Phase array
    phase: Array3<f64>,
    /// Previous state for gradient computation
    prev_state: Option<Array3<f64>>,
}

impl ResonanceTensorField {
    /// Create a new ResonanceTensorField
    ///
    /// # Arguments
    ///
    /// * `shape` - Dimensions of the resonance tensor (Nx, Ny, Nz)
    /// * `initial_amplitude` - Initial amplitude for all cells (default: 1.0)
    /// * `initial_frequency` - Initial frequency for all cells (default: 1.0)
    /// * `initial_phase` - Initial phase offset in radians (default: 0.0)
    /// * `gradient_threshold` - Threshold for singularity detection (default: 1e-3)
    ///
    /// # Returns
    ///
    /// A new ResonanceTensorField instance
    pub fn new(
        shape: (usize, usize, usize),
        initial_amplitude: f64,
        initial_frequency: f64,
        initial_phase: f64,
        gradient_threshold: f64,
    ) -> Self {
        let amplitude = Array3::from_elem(shape, initial_amplitude);
        let frequency = Array3::from_elem(shape, initial_frequency);
        let phase = Array3::from_elem(shape, initial_phase);

        Self {
            shape,
            initial_amplitude,
            initial_frequency,
            initial_phase,
            gradient_threshold,
            time: 0.0,
            amplitude,
            frequency,
            phase,
            prev_state: None,
        }
    }

    /// Get the current resonance values R(t) as a 3D array
    ///
    /// # Returns
    ///
    /// Array3 containing the resonance values at the current time
    pub fn get_state(&self) -> Array3<f64> {
        let t = self.time;
        let mut state = Array3::zeros(self.shape);

        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                for k in 0..self.shape.2 {
                    let a = self.amplitude[[i, j, k]];
                    let omega = self.frequency[[i, j, k]];
                    let phi = self.phase[[i, j, k]];
                    state[[i, j, k]] = a * (omega * t + phi).sin();
                }
            }
        }

        state
    }

    /// Advance the resonance field by dt in time
    ///
    /// Optionally modulate the phase using an input tensor of the same shape.
    /// If provided, the modulation is added to the phase parameter, allowing
    /// external signals to perturb the resonance pattern.
    ///
    /// # Arguments
    ///
    /// * `dt` - Time increment
    /// * `input_modulation` - Optional 3D array for phase modulation
    ///
    /// # Returns
    ///
    /// The new resonance state R(t + dt)
    pub fn step(&mut self, dt: f64, input_modulation: Option<&Array3<f64>>) -> Array3<f64> {
        // Update phase with input modulation
        if let Some(modulation) = input_modulation {
            if modulation.shape() != [self.shape.0, self.shape.1, self.shape.2] {
                panic!("input_modulation must have the same shape as the field");
            }
            self.phase = &self.phase + modulation;
        }

        // Save previous state for gradient computation
        let prev = self.get_state();

        // Update time
        self.time += dt;

        // Compute new state
        let new_state = self.get_state();
        self.prev_state = Some(prev);

        new_state
    }

    /// Compute a global coherence metric from the current state
    ///
    /// The coherence is defined as the mean pairwise cosine similarity
    /// between all cells in the resonance tensor. It generalizes the
    /// Mandorla resonance calculation to three dimensions.
    ///
    /// # Returns
    ///
    /// Coherence value (between -1 and 1)
    pub fn coherence(&self) -> f64 {
        let state = self.get_state();
        let r: Vec<f64> = state.iter().copied().collect();

        if r.iter().all(|&x| x.abs() < 1e-12) {
            return 0.0;
        }

        // Compute pairwise similarities
        let mut similarities = Vec::new();
        for i in 0..r.len() {
            for j in (i + 1)..r.len() {
                let a = r[i];
                let b = r[j];
                // Treat scalars as 1-D vectors
                let sim = (a * b) / ((a.abs() + 1e-12) * (b.abs() + 1e-12));
                similarities.push(sim);
            }
        }

        if similarities.is_empty() {
            0.0
        } else {
            similarities.iter().sum::<f64>() / similarities.len() as f64
        }
    }

    /// Compute the L2 norm of the difference between current and previous state
    ///
    /// If no previous state is stored, zero is returned.
    ///
    /// # Returns
    ///
    /// The gradient norm (L2 norm of state difference)
    pub fn gradient_norm(&self) -> f64 {
        if let Some(prev_state) = &self.prev_state {
            let current = self.get_state();
            let diff = &current - prev_state;

            // Compute L2 norm
            diff.iter().map(|x| x * x).sum::<f64>().sqrt()
        } else {
            0.0
        }
    }

    /// Determine whether the field has reached a singularity (stabilized)
    ///
    /// A singularity event is considered to occur when the gradient norm
    /// between successive states falls below gradient_threshold.
    ///
    /// # Returns
    ///
    /// True if singularity detected, false otherwise
    pub fn detect_singularity(&self) -> bool {
        self.gradient_norm() < self.gradient_threshold
    }

    /// Reset the field to initial conditions
    pub fn reset(&mut self) {
        self.time = 0.0;
        self.amplitude = Array3::from_elem(self.shape, self.initial_amplitude);
        self.frequency = Array3::from_elem(self.shape, self.initial_frequency);
        self.phase = Array3::from_elem(self.shape, self.initial_phase);
        self.prev_state = None;
    }

    /// Get the amplitude array (read-only)
    pub fn get_amplitude(&self) -> &Array3<f64> {
        &self.amplitude
    }

    /// Get the frequency array (read-only)
    pub fn get_frequency(&self) -> &Array3<f64> {
        &self.frequency
    }

    /// Get the phase array (read-only)
    pub fn get_phase(&self) -> &Array3<f64> {
        &self.phase
    }

    /// Set amplitude at a specific cell
    pub fn set_amplitude(&mut self, i: usize, j: usize, k: usize, value: f64) {
        if i < self.shape.0 && j < self.shape.1 && k < self.shape.2 {
            self.amplitude[[i, j, k]] = value;
        }
    }

    /// Set frequency at a specific cell
    pub fn set_frequency(&mut self, i: usize, j: usize, k: usize, value: f64) {
        if i < self.shape.0 && j < self.shape.1 && k < self.shape.2 {
            self.frequency[[i, j, k]] = value;
        }
    }

    /// Set phase at a specific cell
    pub fn set_phase(&mut self, i: usize, j: usize, k: usize, value: f64) {
        if i < self.shape.0 && j < self.shape.1 && k < self.shape.2 {
            self.phase[[i, j, k]] = value;
        }
    }
}

impl Default for ResonanceTensorField {
    /// Create a default ResonanceTensorField with standard parameters
    fn default() -> Self {
        Self::new((4, 4, 4), 1.0, 1.0, 0.0, 1e-3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_create_default_field() {
        let field = ResonanceTensorField::default();
        assert_eq!(field.shape, (4, 4, 4));
        assert_eq!(field.initial_amplitude, 1.0);
        assert_eq!(field.initial_frequency, 1.0);
        assert_eq!(field.initial_phase, 0.0);
        assert_eq!(field.gradient_threshold, 1e-3);
        assert_eq!(field.time, 0.0);
    }

    #[test]
    fn test_create_custom_field() {
        let field = ResonanceTensorField::new((2, 3, 4), 2.0, 3.0, 0.5, 1e-4);
        assert_eq!(field.shape, (2, 3, 4));
        assert_eq!(field.initial_amplitude, 2.0);
        assert_eq!(field.initial_frequency, 3.0);
        assert_eq!(field.initial_phase, 0.5);
        assert_eq!(field.gradient_threshold, 1e-4);
    }

    #[test]
    fn test_get_state_initial() {
        let field = ResonanceTensorField::new((2, 2, 2), 1.0, 1.0, 0.0, 1e-3);
        let state = field.get_state();

        assert_eq!(state.shape(), &[2, 2, 2]);
        // At t=0 and phase=0, sin(0) = 0
        for &val in state.iter() {
            assert!(val.abs() < 1e-10);
        }
    }

    #[test]
    fn test_get_state_nonzero_phase() {
        let field = ResonanceTensorField::new((2, 2, 2), 1.0, 1.0, PI / 2.0, 1e-3);
        let state = field.get_state();

        // At t=0 and phase=π/2, sin(π/2) = 1
        for &val in state.iter() {
            assert!((val - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_step_without_modulation() {
        let mut field = ResonanceTensorField::new((2, 2, 2), 1.0, 2.0, 0.0, 1e-3);

        let state1 = field.step(0.1, None);
        assert_eq!(field.time, 0.1);
        assert_eq!(state1.shape(), &[2, 2, 2]);

        let state2 = field.step(0.1, None);
        assert_eq!(field.time, 0.2);

        // States should differ
        assert!((state1[[0, 0, 0]] - state2[[0, 0, 0]]).abs() > 0.01);
    }

    #[test]
    fn test_step_with_modulation() {
        let mut field = ResonanceTensorField::new((2, 2, 2), 1.0, 1.0, 0.0, 1e-3);

        let modulation = Array3::from_elem((2, 2, 2), 0.5);
        field.step(0.1, Some(&modulation));

        // Phase should have been updated
        assert!((field.get_phase()[[0, 0, 0]] - 0.5).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "input_modulation must have the same shape")]
    fn test_step_wrong_modulation_shape() {
        let mut field = ResonanceTensorField::new((2, 2, 2), 1.0, 1.0, 0.0, 1e-3);
        let modulation = Array3::from_elem((3, 3, 3), 0.5);
        field.step(0.1, Some(&modulation));
    }

    #[test]
    fn test_coherence_uniform() {
        let mut field = ResonanceTensorField::new((2, 2, 2), 1.0, 1.0, PI / 2.0, 1e-3);
        field.step(0.0, None);

        let coherence = field.coherence();
        // All cells have same value, so coherence should be 1.0
        assert!((coherence - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_coherence_zero_state() {
        let field = ResonanceTensorField::new((2, 2, 2), 0.0, 1.0, 0.0, 1e-3);
        let coherence = field.coherence();

        // Zero state should have zero coherence
        assert_eq!(coherence, 0.0);
    }

    #[test]
    fn test_gradient_norm_no_previous() {
        let field = ResonanceTensorField::new((2, 2, 2), 1.0, 1.0, 0.0, 1e-3);
        let norm = field.gradient_norm();

        // No previous state, should be 0
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_gradient_norm_with_previous() {
        let mut field = ResonanceTensorField::new((2, 2, 2), 1.0, 2.0, 0.0, 1e-3);

        field.step(0.1, None);
        field.step(0.1, None);

        let norm = field.gradient_norm();
        assert!(norm > 0.0);
    }

    #[test]
    fn test_gradient_norm_stable() {
        let mut field = ResonanceTensorField::new((2, 2, 2), 1.0, 0.0, 0.0, 1e-3);

        field.step(0.1, None);
        field.step(0.1, None);

        // With zero frequency, state doesn't change
        let norm = field.gradient_norm();
        assert!(norm < 1e-10);
    }

    #[test]
    fn test_detect_singularity_stable() {
        let mut field = ResonanceTensorField::new((2, 2, 2), 1.0, 0.0, 0.0, 1e-3);

        field.step(0.1, None);
        field.step(0.1, None);

        // Stable field should detect singularity
        assert!(field.detect_singularity());
    }

    #[test]
    fn test_detect_singularity_oscillating() {
        let mut field = ResonanceTensorField::new((2, 2, 2), 1.0, 10.0, 0.0, 1e-3);

        field.step(0.01, None);
        field.step(0.01, None);

        // Oscillating field should not detect singularity
        assert!(!field.detect_singularity());
    }

    #[test]
    fn test_reset() {
        let mut field = ResonanceTensorField::new((2, 2, 2), 1.0, 2.0, 0.5, 1e-3);

        field.step(0.5, None);
        field.set_amplitude(0, 0, 0, 5.0);

        assert_eq!(field.time, 0.5);
        assert_eq!(field.get_amplitude()[[0, 0, 0]], 5.0);

        field.reset();

        assert_eq!(field.time, 0.0);
        assert_eq!(field.get_amplitude()[[0, 0, 0]], 1.0);
        assert!(field.prev_state.is_none());
    }

    #[test]
    fn test_set_amplitude() {
        let mut field = ResonanceTensorField::default();
        field.set_amplitude(0, 0, 0, 5.0);
        assert_eq!(field.get_amplitude()[[0, 0, 0]], 5.0);
    }

    #[test]
    fn test_set_frequency() {
        let mut field = ResonanceTensorField::default();
        field.set_frequency(1, 1, 1, 3.0);
        assert_eq!(field.get_frequency()[[1, 1, 1]], 3.0);
    }

    #[test]
    fn test_set_phase() {
        let mut field = ResonanceTensorField::default();
        field.set_phase(2, 2, 2, 1.5);
        assert_eq!(field.get_phase()[[2, 2, 2]], 1.5);
    }

    #[test]
    fn test_set_out_of_bounds() {
        let mut field = ResonanceTensorField::new((2, 2, 2), 1.0, 1.0, 0.0, 1e-3);

        // Should not panic with out-of-bounds indices
        field.set_amplitude(10, 10, 10, 5.0);
        field.set_frequency(10, 10, 10, 5.0);
        field.set_phase(10, 10, 10, 5.0);
    }

    #[test]
    fn test_time_evolution() {
        let mut field = ResonanceTensorField::new((2, 2, 2), 1.0, 1.0, 0.0, 1e-3);

        let state1 = field.get_state();
        field.step(PI / 2.0, None);
        let state2 = field.get_state();
        field.step(PI / 2.0, None);
        let state3 = field.get_state();

        // At t=0: sin(0) = 0
        assert!(state1[[0, 0, 0]].abs() < 1e-10);
        // At t=π/2: sin(π/2) = 1
        assert!((state2[[0, 0, 0]] - 1.0).abs() < 1e-10);
        // At t=π: sin(π) ≈ 0
        assert!(state3[[0, 0, 0]].abs() < 1e-10);
    }
}
