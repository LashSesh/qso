use serde::{Deserialize, Serialize};

/// Configuration for the tripolar operators inside the monolith kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripolarConfig {
    pub psi: f64,
    pub rho: f64,
    pub omega: f64,
}

impl TripolarConfig {
    pub fn value(&self) -> f64 {
        self.psi * self.rho * self.omega
    }
}

impl Default for TripolarConfig {
    fn default() -> Self {
        Self {
            psi: 0.6,
            rho: 0.6,
            omega: 0.6,
        }
    }
}

/// Prototype spectrum used to seed the semantic field classifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrototypeConfig {
    pub name: String,
    pub spectrum: Vec<f64>,
}

impl PrototypeConfig {
    pub fn new(name: impl Into<String>, spectrum: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            spectrum,
        }
    }
}

/// Configuration for the low-level Metatron agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub qlogic_nodes: usize,
    pub operators: Vec<TripolarConfig>,
    #[serde(default)]
    pub semantic_prototypes: Vec<PrototypeConfig>,
    #[serde(default = "AgentConfig::default_threshold")]
    pub monolith_threshold: f64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            qlogic_nodes: 13,
            operators: vec![TripolarConfig::default(); 4],
            semantic_prototypes: vec![PrototypeConfig::new("baseline", vec![1.0; 13])],
            monolith_threshold: Self::default_threshold(),
        }
    }
}

impl AgentConfig {
    const fn default_threshold() -> f64 {
        1.0
    }
}

/// Configuration for the higher-level master agent orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasterAgentConfig {
    pub gabriel_cells: usize,
    pub mandorla_alpha: f64,
    pub mandorla_beta: f64,
    pub spiral_alpha: f64,
}

impl Default for MasterAgentConfig {
    fn default() -> Self {
        Self {
            gabriel_cells: 4,
            mandorla_alpha: 0.5,
            mandorla_beta: 0.5,
            spiral_alpha: 0.1,
        }
    }
}

/// Configuration for the full post-symbolic engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub agent: AgentConfig,
    pub master: MasterAgentConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            agent: AgentConfig::default(),
            master: MasterAgentConfig::default(),
        }
    }
}
