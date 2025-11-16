use crate::cognition::agent::{AgentStep, MetatronAgent};
use crate::cognition::master::{MasterAgent, MasterAgentOutput};
use crate::cognition::seraphic_feedback::SeraphicValue;
use crate::config::EngineConfig;
use crate::error::EngineResult;

/// High-level orchestrator that wires the Metatron agent and master agent
/// together into a cohesive post-symbolic cognition engine.
#[derive(Debug)]
pub struct MetatronEngine {
    pub agent: MetatronAgent,
    pub master: MasterAgent,
    time: f64,
}

/// Snapshot of a full processing cycle combining the low-level agent and the
/// high-level master agent outputs.
#[derive(Debug)]
pub struct EngineSnapshot {
    pub time: f64,
    pub agent_step: AgentStep,
    pub master_output: MasterAgentOutput,
}

impl MetatronEngine {
    /// Build an engine from the provided configuration structure.
    pub fn from_config(config: EngineConfig) -> EngineResult<Self> {
        let agent = MetatronAgent::from_config(config.agent)?;
        let master = MasterAgent::from_config(config.master)?;
        Ok(Self {
            agent,
            master,
            time: 0.0,
        })
    }

    /// Create an engine using the default configuration.
    pub fn new() -> EngineResult<Self> {
        Self::from_config(EngineConfig::default())
    }

    /// Execute one cognition cycle consuming the provided inputs.
    pub fn cycle<I>(&mut self, inputs: I) -> EngineResult<EngineSnapshot>
    where
        I: IntoIterator,
        I::Item: Into<SeraphicValue>,
    {
        let master_output = self.master.process(inputs)?;
        let agent_step = self.agent.step(self.time)?;
        let snapshot = EngineSnapshot {
            time: self.time,
            agent_step,
            master_output,
        };
        self.time += 1.0;
        Ok(snapshot)
    }

    /// Reset the internal time tracking back to the origin.
    pub fn reset(&mut self) {
        self.time = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_cycle() {
        let mut engine = MetatronEngine::new().unwrap();
        let snapshot = engine.cycle(["alpha", "beta"]).unwrap();
        assert!(!snapshot.master_output.gabriel_outputs.is_empty());
    }
}
