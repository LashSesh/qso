use crate::cognition::seraphic_feedback::{SeraphicFeedbackModule, SeraphicValue};
use crate::cognition::spiralmemory::SpiralMemory;
use crate::config::MasterAgentConfig;
use crate::error::{EngineError, EngineResult};
use crate::fields::gabriel::GabrielCell;
use crate::fields::resonance::MandorlaField;

#[derive(Debug, Clone)]
pub struct MasterAgentOutput {
    pub inputs: Vec<SeraphicValue>,
    pub spiral_points: Vec<Vec<f64>>,
    pub gabriel_outputs: Vec<f64>,
    pub mandorla_resonance: f64,
    pub decision: bool,
}

#[derive(Debug)]
pub struct MasterAgent {
    pub spiral_memory: SpiralMemory,
    pub gabriel_cells: Vec<GabrielCell>,
    pub mandorla: MandorlaField,
    pub seraphic: SeraphicFeedbackModule,
    pub last_decision: Option<bool>,
}

impl MasterAgent {
    pub fn from_config(config: MasterAgentConfig) -> EngineResult<Self> {
        if config.gabriel_cells == 0 {
            return Err(EngineError::Misconfigured(
                "master agent requires at least one Gabriel cell".into(),
            ));
        }
        if !(0.0..=1.0).contains(&config.mandorla_alpha) {
            return Err(EngineError::Misconfigured(
                "mandorla_alpha must be between 0 and 1".into(),
            ));
        }
        if !(0.0..=1.0).contains(&config.mandorla_beta) {
            return Err(EngineError::Misconfigured(
                "mandorla_beta must be between 0 and 1".into(),
            ));
        }
        if config.spiral_alpha <= 0.0 {
            return Err(EngineError::Misconfigured(
                "spiral_alpha must be positive".into(),
            ));
        }

        let mut cells = Vec::with_capacity(config.gabriel_cells);
        for _ in 0..config.gabriel_cells {
            cells.push(GabrielCell::simple());
        }
        for i in 0..(config.gabriel_cells.saturating_sub(1)) {
            let (left, right) = cells.split_at_mut(i + 1);
            let a = &mut left[i];
            let b = &mut right[0];
            GabrielCell::couple_pair(a, b);
        }
        Ok(Self {
            spiral_memory: SpiralMemory::new(config.spiral_alpha),
            gabriel_cells: cells,
            mandorla: MandorlaField::new(0.985, config.mandorla_alpha, config.mandorla_beta),
            seraphic: SeraphicFeedbackModule::new(),
            last_decision: None,
        })
    }

    pub fn process<V>(
        &mut self,
        raw_inputs: impl IntoIterator<Item = V>,
    ) -> EngineResult<MasterAgentOutput>
    where
        V: Into<SeraphicValue>,
    {
        let inputs: Vec<SeraphicValue> = raw_inputs.into_iter().map(Into::into).collect();
        if inputs.is_empty() {
            return Err(EngineError::Misconfigured(
                "master agent requires at least one input value".into(),
            ));
        }

        let feedback_vectors = self.seraphic.map_inputs(inputs.clone());
        let string_inputs: Vec<String> = inputs.iter().map(describe_value).collect();
        let (points, _) = self.spiral_memory.step(&string_inputs, 18);
        for (cell, vec) in self.gabriel_cells.iter_mut().zip(points.iter()) {
            let sum: f64 = vec.iter().sum();
            cell.activate(Some(sum));
        }
        self.mandorla.clear_inputs();
        for cell in &self.gabriel_cells {
            self.mandorla.add_input(vec![cell.output; 5]);
        }
        for vec in feedback_vectors {
            self.mandorla.add_input(vec);
        }
        let decision = self.mandorla.decision_trigger();
        self.last_decision = Some(decision);
        Ok(MasterAgentOutput {
            inputs,
            spiral_points: points,
            gabriel_outputs: self.gabriel_cells.iter().map(|c| c.output).collect(),
            mandorla_resonance: self.mandorla.resonance,
            decision,
        })
    }
}

impl Default for MasterAgent {
    fn default() -> Self {
        Self::from_config(MasterAgentConfig::default())
            .expect("default master agent config is valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_inputs() {
        let mut agent = MasterAgent::from_config(MasterAgentConfig {
            gabriel_cells: 3,
            mandorla_alpha: 0.5,
            mandorla_beta: 0.5,
            spiral_alpha: 0.1,
        })
        .unwrap();
        let result = agent
            .process([
                SeraphicValue::from("CYBER"),
                SeraphicValue::from("7.5"),
                SeraphicValue::from("[1,2,3]"),
            ])
            .unwrap();
        assert_eq!(result.spiral_points.len(), 3);
    }
}

fn describe_value(value: &SeraphicValue) -> String {
    match value {
        SeraphicValue::Text(text) => text.clone(),
        SeraphicValue::Number(num) => format!("{num:.3}"),
        SeraphicValue::Sequence(seq) => {
            if seq.is_empty() {
                "[]".to_string()
            } else {
                let preview: Vec<String> = seq.iter().take(6).map(|v| format!("{v:.3}")).collect();
                format!(
                    "[{}{}]",
                    preview.join(","),
                    if seq.len() > preview.len() {
                        ", ..."
                    } else {
                        ""
                    }
                )
            }
        }
    }
}
