// Cognition module - Agents and cognitive processing

pub mod agent;
pub mod master;
pub mod meta_interpreter;
pub mod monolith;
pub mod qdash;
pub mod qlogic;
pub mod quantum;
pub mod semantic_field;
pub mod seraphic_feedback;
pub mod spiralmemory;

pub use agent::{AgentStep, MetatronAgent};
pub use master::{MasterAgent, MasterAgentOutput};
pub use meta_interpreter::MetaInterpreter;
pub use monolith::{DecisionState, MonolithDecision, OphanKernel, TripolarOperator};
pub use qdash::QDASHAgent;
pub use qlogic::{Diagnostics, QLOGICOscillatorCore, QLogicEngine, QLogicOutput};
pub use semantic_field::{ResonanceDiagnostics, SemanticField};
pub use seraphic_feedback::SeraphicValue;
pub use spiralmemory::SpiralMemory;
