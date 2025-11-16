/*!
 * QDASH Agent Module
 *
 * This module implements a simplified QDASH (Quantum-Dash) agent by
 * combining the oscillator logic from QLOGIC with the Mandorla resonance
 * field and SpiralMemory. The agent follows the decision cycle:
 *
 * 1. Input transduction - External input transformed into oscillator signal
 * 2. Resonance coupling - Oscillator signal added to Mandorla field
 * 3. Coherence measurement - Global resonance as coherence metric
 * 4. Singularity trigger - Decision event when coherence exceeds threshold
 * 5. Feedback encoding - Internal state updated with new input
 */

use ndarray::Array1;
use serde::{Deserialize, Serialize};

use crate::gabriel_cell::GabrielCell;
use crate::mandorla::MandorlaField;
use crate::qlogic::QLogicEngine;
use crate::spiral_memory::SpiralMemory;

/// Result of a QDASH decision cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QDASHResult {
    /// Oscillator signal generated from input
    pub oscillator_signal: Vec<f64>,
    /// Current resonance value
    pub resonance: f64,
    /// Adaptive threshold used for decision
    pub threshold: f64,
    /// Decision flag (true if threshold exceeded)
    pub decision: bool,
    /// Spiral memory points
    pub spiral_points: Vec<Vec<f64>>,
    /// Gabriel cell outputs
    pub gabriel_outputs: Vec<f64>,
}

/// A resonant agent implementing a simplified QDASH decision cycle
///
/// Combines QLogic oscillator patterns, Mandorla resonance fields,
/// SpiralMemory encoding, and Gabriel cell feedback to implement
/// a decision-making system that triggers on coherence thresholds.
#[derive(Debug, Clone)]
pub struct QDASHAgent {
    /// QLogic engine for oscillator patterns
    pub qlogic: QLogicEngine,
    /// Mandorla field for resonance computation
    pub mandorla: MandorlaField,
    /// Spiral memory for semantic encoding
    pub spiral: SpiralMemory,
    /// Gabriel cells for feedback coupling
    pub cells: Vec<GabrielCell>,
    /// Current time parameter for oscillator phase
    pub time: f64,
    /// Last decision output
    pub last_decision: Option<bool>,
}

impl QDASHAgent {
    /// Create a new QDASH agent
    ///
    /// # Arguments
    ///
    /// * `n_cells` - Number of Gabriel cells for feedback coupling (default: 4)
    /// * `alpha` - Weight of entropy in adaptive threshold (default: 0.5)
    /// * `beta` - Weight of variance in adaptive threshold (default: 0.5)
    pub fn new(n_cells: usize, alpha: f64, beta: f64) -> Self {
        let mut cells = Vec::new();
        for _ in 0..n_cells {
            cells.push(GabrielCell::default());
        }

        // Couple consecutive cells
        for i in 0..n_cells.saturating_sub(1) {
            if i + 1 < cells.len() {
                crate::gabriel_cell::couple_cells(&mut cells, i, i + 1);
            }
        }

        Self {
            qlogic: QLogicEngine::new(13), // 13 nodes for Metatron pattern
            mandorla: MandorlaField::new(0.985, alpha, beta),
            spiral: SpiralMemory::new(0.07),
            cells,
            time: 0.0,
            last_decision: None,
        }
    }

    /// Transform external input into oscillator signal
    ///
    /// Maps sensory input to oscillator signatures using amplitude modulation.
    /// In this minimal implementation, we sum the input and scale the QLOGIC
    /// oscillator pattern accordingly.
    ///
    /// # Arguments
    ///
    /// * `input_vector` - External input vector
    ///
    /// # Returns
    ///
    /// Oscillator signal as array
    pub fn trm_transform(&self, input_vector: &[f64]) -> Array1<f64> {
        // Compute amplitude as sum of input
        let amplitude: f64 = input_vector.iter().sum();

        // Generate oscillator pattern at current time
        let result = self.qlogic.step(self.time);
        let pattern = result.field;

        // Scale pattern by amplitude
        pattern * amplitude
    }

    /// Update Gabriel cells from new SpiralMemory points
    ///
    /// Activates each Gabriel cell with the sum of corresponding spiral point,
    /// then adds Gabriel outputs to Mandorla field as resonance contributions.
    ///
    /// # Arguments
    ///
    /// * `spiral_points` - Points from spiral memory embedding
    pub fn update_internal_state(&mut self, spiral_points: &[Array1<f64>]) {
        // Update Gabriel cells with spiral points
        for (i, point) in spiral_points.iter().enumerate() {
            if i >= self.cells.len() {
                break;
            }
            let sum: f64 = point.iter().sum();
            self.cells[i].activate(Some(sum));
        }

        // Add Gabriel outputs to Mandorla (using 5D vectors)
        for cell in &self.cells {
            let output_vec = Array1::from_elem(5, cell.output);
            self.mandorla.add_input(output_vec);
        }
    }

    /// Convert oscillator signal to 5D vector for Mandorla
    ///
    /// Takes the first 5 elements of the oscillator signal
    fn osc_to_5d(&self, osc_signal: &Array1<f64>) -> Array1<f64> {
        if osc_signal.len() >= 5 {
            osc_signal.slice(ndarray::s![0..5]).to_owned()
        } else {
            // Pad with zeros if signal is too short
            let mut result = Array1::zeros(5);
            for (i, &val) in osc_signal.iter().enumerate() {
                result[i] = val;
            }
            result
        }
    }

    /// Run the QDASH decision cycle on a single input vector
    ///
    /// # Arguments
    ///
    /// * `input_vector` - External input to be processed
    /// * `max_iter` - Maximum resonance iterations (default: 3)
    /// * `dt` - Time increment between iterations (default: 1.0)
    ///
    /// # Returns
    ///
    /// QDASHResult containing signals, resonance, and decision
    pub fn decision_cycle(
        &mut self,
        input_vector: &[f64],
        max_iter: usize,
        dt: f64,
    ) -> QDASHResult {
        // Reset Mandorla inputs
        self.mandorla.clear_inputs();

        // Step the SpiralMemory to embed the input
        let string_inputs: Vec<String> = input_vector.iter().map(|x| x.to_string()).collect();
        let (points, _psi) = self.spiral.step(&string_inputs, 18);

        // Update Gabriel cells and add outputs to Mandorla
        let cell_count = self.cells.len();
        let points_to_use = if points.len() < cell_count {
            &points[..]
        } else {
            &points[..cell_count]
        };
        self.update_internal_state(points_to_use);

        // Generate oscillator signal from input
        let mut osc_signal = self.trm_transform(input_vector);

        // Add oscillator pattern to Mandorla inputs (convert to 5D)
        let osc_5d = self.osc_to_5d(&osc_signal);
        self.mandorla.add_input(osc_5d);

        // Iterate resonance until decision or max_iter
        let mut decision = false;
        for _ in 0..max_iter {
            let resonance = self.mandorla.calc_resonance();
            let threshold = self.mandorla.current_theta;

            if resonance > threshold {
                decision = true;
                break;
            }

            // Update time for oscillator phase
            self.time += dt;

            // Update oscillator signal for next iteration
            osc_signal = self.trm_transform(input_vector);
            let osc_5d = self.osc_to_5d(&osc_signal);
            self.mandorla.add_input(osc_5d);
        }

        self.last_decision = Some(decision);

        QDASHResult {
            oscillator_signal: osc_signal.to_vec(),
            resonance: self.mandorla.resonance,
            threshold: self.mandorla.current_theta,
            decision,
            spiral_points: points.iter().map(|p| p.to_vec()).collect(),
            gabriel_outputs: self.cells.iter().map(|c| c.output).collect(),
        }
    }

    /// Get the number of Gabriel cells
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Reset the agent state
    pub fn reset(&mut self) {
        self.time = 0.0;
        self.last_decision = None;
        self.mandorla.clear_inputs();
        self.spiral.clear();

        // Reset Gabriel cells
        for cell in &mut self.cells {
            *cell = GabrielCell::default();
        }
    }
}

impl Default for QDASHAgent {
    /// Create a default QDASH agent with 4 cells and balanced α/β = 0.5
    fn default() -> Self {
        Self::new(4, 0.5, 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_default() {
        let agent = QDASHAgent::default();
        assert_eq!(agent.cell_count(), 4);
        assert_eq!(agent.time, 0.0);
        assert_eq!(agent.last_decision, None);
    }

    #[test]
    fn test_create_custom() {
        let agent = QDASHAgent::new(8, 0.3, 0.7);
        assert_eq!(agent.cell_count(), 8);
        assert_eq!(agent.mandorla.alpha, 0.3);
        assert_eq!(agent.mandorla.beta, 0.7);
    }

    #[test]
    fn test_trm_transform() {
        let agent = QDASHAgent::default();
        let input = vec![1.0, 2.0, 3.0];
        let signal = agent.trm_transform(&input);

        assert_eq!(signal.len(), 13); // 13 nodes in QLogic
    }

    #[test]
    fn test_decision_cycle_basic() {
        let mut agent = QDASHAgent::default();
        let input = vec![1.0, 2.0, 3.0, 4.0];

        let result = agent.decision_cycle(&input, 3, 1.0);

        assert_eq!(result.oscillator_signal.len(), 13);
        assert!(result.resonance >= 0.0);
        assert!(result.threshold >= 0.0);
        assert_eq!(result.gabriel_outputs.len(), 4);
    }

    #[test]
    fn test_decision_cycle_updates_time() {
        let mut agent = QDASHAgent::default();
        let input = vec![1.0, 2.0];

        let initial_time = agent.time;
        agent.decision_cycle(&input, 3, 0.5);

        // Time should increase (unless decision was immediate)
        assert!(agent.time >= initial_time);
    }

    #[test]
    fn test_decision_cycle_stores_decision() {
        let mut agent = QDASHAgent::default();
        let input = vec![1.0, 2.0, 3.0];

        agent.decision_cycle(&input, 3, 1.0);

        assert!(agent.last_decision.is_some());
    }

    #[test]
    fn test_update_internal_state() {
        let mut agent = QDASHAgent::default();
        let points = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0]),
        ];

        agent.update_internal_state(&points);

        // Should have added inputs to mandorla
        assert!(agent.mandorla.inputs.len() > 0);
    }

    #[test]
    fn test_reset() {
        let mut agent = QDASHAgent::default();
        let input = vec![1.0, 2.0, 3.0];

        agent.decision_cycle(&input, 3, 1.0);
        agent.time = 5.0;

        agent.reset();

        assert_eq!(agent.time, 0.0);
        assert_eq!(agent.last_decision, None);
        assert_eq!(agent.mandorla.inputs.len(), 0);
    }

    #[test]
    fn test_cells_are_coupled() {
        let agent = QDASHAgent::new(3, 0.5, 0.5);

        // Check that cells have neighbors (coupling)
        // First cell should have second as neighbor
        assert!(agent.cells[0].neighbors.len() > 0);
    }

    #[test]
    fn test_multiple_cycles() {
        let mut agent = QDASHAgent::default();
        let input1 = vec![1.0, 2.0];
        let input2 = vec![3.0, 4.0];

        let result1 = agent.decision_cycle(&input1, 2, 0.5);
        let result2 = agent.decision_cycle(&input2, 2, 0.5);

        // Both should produce valid results
        assert!(result1.resonance >= 0.0);
        assert!(result2.resonance >= 0.0);
    }

    #[test]
    fn test_spiral_points_in_result() {
        let mut agent = QDASHAgent::default();
        let input = vec![1.0, 2.0, 3.0];

        let result = agent.decision_cycle(&input, 3, 1.0);

        assert!(result.spiral_points.len() > 0);
        assert_eq!(result.spiral_points[0].len(), 5); // 5D points
    }

    #[test]
    fn test_different_max_iter() {
        let mut agent = QDASHAgent::default();
        let input = vec![0.5, 1.5];

        let result1 = agent.decision_cycle(&input, 1, 1.0);
        agent.reset();
        let result2 = agent.decision_cycle(&input, 5, 1.0);

        // More iterations might lead to different outcomes
        assert!(result1.resonance >= 0.0);
        assert!(result2.resonance >= 0.0);
    }

    #[test]
    fn test_zero_cells() {
        let agent = QDASHAgent::new(0, 0.5, 0.5);
        assert_eq!(agent.cell_count(), 0);
    }

    #[test]
    fn test_large_input() {
        let mut agent = QDASHAgent::default();
        let input = vec![1.0; 100]; // Large input vector

        let result = agent.decision_cycle(&input, 2, 1.0);

        assert!(result.oscillator_signal.len() == 13);
        assert!(result.resonance >= 0.0);
    }
}
