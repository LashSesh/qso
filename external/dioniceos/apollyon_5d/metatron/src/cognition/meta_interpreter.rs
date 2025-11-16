use std::collections::VecDeque;

use crate::cognition::qdash::QDASHAgent;

#[derive(Debug)]
pub struct MetaInterpreter {
    pub agent: QDASHAgent,
    window: VecDeque<bool>,
    window_size: usize,
    target_rate: f64,
}

impl MetaInterpreter {
    pub fn new(agent: QDASHAgent, window_size: usize, target_rate: f64) -> Self {
        Self {
            agent,
            window: VecDeque::new(),
            window_size,
            target_rate,
        }
    }

    pub fn record_decision(&mut self, decision: bool) {
        if self.window.len() == self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(decision);
    }

    pub fn adjust_parameters(&mut self) {
        if self.window.is_empty() {
            return;
        }
        let rate = self.window.iter().filter(|&&d| d).count() as f64 / self.window.len() as f64;
        if rate > self.target_rate + 0.1 {
            self.agent.mandorla.alpha *= 1.05;
            self.agent.mandorla.beta *= 1.05;
        } else if rate < self.target_rate - 0.1 {
            self.agent.mandorla.alpha *= 0.95;
            self.agent.mandorla.beta *= 0.95;
        }
    }

    pub fn modulate_oscillator_frequency(&mut self, factor: f64) {
        let nodes = (self.agent.qlogic.osc_core.num_nodes as f64 * factor).round() as usize;
        self.agent.qlogic.osc_core.num_nodes = nodes.max(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptation() {
        let agent = QDASHAgent::default();
        let mut meta = MetaInterpreter::new(agent, 5, 0.5);
        for _ in 0..5 {
            meta.record_decision(true);
        }
        meta.adjust_parameters();
        assert!(meta.agent.mandorla.alpha > 0.5);
    }
}
