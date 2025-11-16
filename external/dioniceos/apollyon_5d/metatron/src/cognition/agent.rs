use crate::cognition::monolith::{DecisionState, MonolithDecision, OphanKernel, TripolarOperator};
use crate::cognition::qlogic::{QLogicEngine, QLogicOutput};
use crate::cognition::semantic_field::SemanticField;
use crate::config::{AgentConfig, PrototypeConfig};
use crate::error::{EngineError, EngineResult};
use crate::geometry::graph::MetatronCubeGraph;

#[derive(Debug, Clone)]
pub struct AgentStep {
    pub qlogic: QLogicOutput,
    pub monolith: DecisionState,
}

#[derive(Debug)]
pub struct MetatronAgent {
    pub graph: MetatronCubeGraph,
    pub qlogic: QLogicEngine,
    pub monolith: MonolithDecision,
    pub memory: Vec<AgentStep>,
}

impl MetatronAgent {
    pub fn from_config(config: AgentConfig) -> EngineResult<Self> {
        if config.qlogic_nodes == 0 {
            return Err(EngineError::Misconfigured(
                "qlogic_nodes must be greater than zero".into(),
            ));
        }

        if config.operators.is_empty() {
            return Err(EngineError::Misconfigured(
                "at least one tripolar operator must be configured".into(),
            ));
        }

        let graph = MetatronCubeGraph::new();
        let semantic_field =
            build_semantic_field(&config.semantic_prototypes, config.qlogic_nodes)?;

        let qlogic = QLogicEngine::new(config.qlogic_nodes, semantic_field);
        let operators = config
            .operators
            .iter()
            .map(|op| TripolarOperator::new(op.psi, op.rho, op.omega))
            .collect();
        let kernel = OphanKernel::new(operators);

        Ok(Self {
            graph,
            qlogic,
            monolith: MonolithDecision::new(kernel, config.monolith_threshold),
            memory: Vec::new(),
        })
    }

    pub fn new() -> EngineResult<Self> {
        Self::from_config(AgentConfig::default())
    }

    pub fn step(&mut self, t: f64) -> EngineResult<AgentStep> {
        let qlogic = self.qlogic.step(t);
        let decision = self.monolith.evaluate();
        let record = AgentStep {
            qlogic: qlogic.clone(),
            monolith: decision,
        };
        self.memory.push(record.clone());
        Ok(record)
    }
}

impl Default for MetatronAgent {
    fn default() -> Self {
        Self::new().expect("default configuration must be valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_step() {
        let mut agent = MetatronAgent::new().unwrap();
        let res = agent.step(0.0).unwrap();
        assert_eq!(res.qlogic.field.len(), 13);
        assert!(matches!(
            res.monolith,
            DecisionState::Pending | DecisionState::Excalibration
        ));
    }
}

fn build_semantic_field(
    prototypes: &[PrototypeConfig],
    qlogic_nodes: usize,
) -> EngineResult<Option<SemanticField>> {
    if prototypes.is_empty() {
        return Ok(None);
    }

    let mut field = SemanticField::new();
    for proto in prototypes {
        if proto.spectrum.len() != qlogic_nodes {
            return Err(EngineError::DimensionMismatch {
                expected: qlogic_nodes,
                actual: proto.spectrum.len(),
            });
        }
        field.add_prototype(&proto.name, &proto.spectrum);
    }
    Ok(Some(field))
}
