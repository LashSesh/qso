use crate::cognition::qlogic::{QLogicEngine, QLogicOutput};
use crate::cognition::semantic_field::SemanticField;
use crate::cognition::spiralmemory::SpiralMemory;
use crate::fields::gabriel::GabrielCell;
use crate::fields::resonance::MandorlaField;

#[derive(Debug, Clone)]
pub struct DecisionOutcome {
    pub oscillator_signal: Vec<f64>,
    pub resonance: f64,
    pub threshold: f64,
    pub decision: bool,
    pub spiral_points: Vec<Vec<f64>>,
    pub gabriel_outputs: Vec<f64>,
    pub qlogic: QLogicOutput,
}

#[derive(Debug)]
pub struct QDASHAgent {
    pub qlogic: QLogicEngine,
    pub mandorla: MandorlaField,
    pub spiral: SpiralMemory,
    pub cells: Vec<GabrielCell>,
    pub time: f64,
    pub last_decision: Option<bool>,
}

impl QDASHAgent {
    pub fn new(n_cells: usize, alpha: f64, beta: f64) -> Self {
        let semantic_field = {
            let mut sf = SemanticField::new();
            sf.add_prototype("baseline", &vec![1.0; 13]);
            Some(sf)
        };
        let mut cells = Vec::with_capacity(n_cells);
        for _ in 0..n_cells {
            cells.push(GabrielCell::simple());
        }
        for i in 0..(n_cells.saturating_sub(1)) {
            let (left, right) = cells.split_at_mut(i + 1);
            let a = &mut left[i];
            let b = &mut right[0];
            GabrielCell::couple_pair(a, b);
        }
        Self {
            qlogic: QLogicEngine::new(13, semantic_field),
            mandorla: MandorlaField::new(0.985, alpha, beta),
            spiral: SpiralMemory::default(),
            cells,
            time: 0.0,
            last_decision: None,
        }
    }

    pub fn trm_transform(&mut self, input_vector: &[f64]) -> Vec<f64> {
        let amplitude: f64 = input_vector.iter().sum();
        self.qlogic
            .osc_core
            .generate_pattern(self.time)
            .into_iter()
            .map(|v| v * amplitude)
            .collect()
    }

    fn update_internal_state(&mut self, spiral_points: &[Vec<f64>]) {
        for (cell, vec) in self.cells.iter_mut().zip(spiral_points.iter()) {
            let sum: f64 = vec.iter().sum();
            cell.activate(Some(sum));
        }
        self.mandorla.clear_inputs();
        for cell in &self.cells {
            self.mandorla.add_input(vec![cell.output; 5]);
        }
    }

    pub fn decision_cycle(
        &mut self,
        input_vector: &[f64],
        max_iter: usize,
        dt: f64,
    ) -> DecisionOutcome {
        let elements: Vec<String> = input_vector.iter().map(|v| format!("{:.3}", v)).collect();
        let (points, _) = self.spiral.step(&elements, 18);
        self.update_internal_state(&points);
        let mut osc_signal = self.trm_transform(input_vector);
        self.mandorla.add_input(osc_signal.clone());
        let mut decision = false;
        for _ in 0..max_iter {
            if self.mandorla.decision_trigger() {
                decision = true;
                break;
            }
            self.time += dt;
            osc_signal = self.trm_transform(input_vector);
            self.mandorla.add_input(osc_signal.clone());
        }
        let qlogic_output = self.qlogic.step(self.time);
        let resonance = self.mandorla.resonance;
        let threshold = self.mandorla.current_theta;
        let outputs: Vec<f64> = self.cells.iter().map(|c| c.output).collect();
        self.last_decision = Some(decision);
        DecisionOutcome {
            oscillator_signal: osc_signal,
            resonance,
            threshold,
            decision,
            spiral_points: points,
            gabriel_outputs: outputs,
            qlogic: qlogic_output,
        }
    }
}

impl Default for QDASHAgent {
    fn default() -> Self {
        Self::new(4, 0.5, 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_cycle() {
        let mut agent = QDASHAgent::default();
        let result = agent.decision_cycle(&[1.0, 0.5, 0.2], 3, 1.0);
        assert_eq!(result.oscillator_signal.len(), 13);
    }
}
