//! Variational Quantum Algorithms (VQA) Suite
//!
//! This module provides a comprehensive suite of hybrid quantum-classical algorithms
//! for optimization, eigenvalue problems, and quantum machine learning. All algorithms
//! are implemented in pure Rust with high-performance numerics and idiomatic design.
//!
//! ## Algorithms
//!
//! - **VQE (Variational Quantum Eigensolver)**: Find ground state energies
//! - **QAOA (Quantum Approximate Optimization Algorithm)**: Solve combinatorial problems
//! - **VQC (Variational Quantum Classifier)**: Quantum machine learning
//!
//! ## Core Components
//!
//! - **Ansatz**: Parametrized quantum circuits
//! - **Cost Functions**: Problem-specific objectives with gradient computation
//! - **Optimizers**: Classical optimization algorithms (COBYLA, ADAM, L-BFGS-B)
//! - **Hybrid Loop**: Orchestration of quantum-classical iterations

pub mod ansatz;
pub mod cost_function;
pub mod optimizer;
pub mod qaoa;
pub mod vqc;
pub mod vqe;

pub use ansatz::{
    Ansatz, AnsatzType, EfficientSU2Ansatz, EntanglementStrategy, HardwareEfficientAnsatz,
    MetatronAnsatz,
};
pub use cost_function::{
    CostFunction, GradientMethod, QAOACostFunction, VQCCostFunction, VQECostFunction,
};
pub use optimizer::{OptimizationResult, Optimizer, OptimizerConfig, OptimizerType};
pub use qaoa::QAOA;
pub use vqc::VQC;
pub use vqe::VQE;

/// Parameter vector type for variational algorithms
pub type ParameterVector = Vec<f64>;

/// History entry for optimization tracking
#[derive(Clone, Debug)]
pub struct HistoryEntry {
    pub iteration: usize,
    pub parameters: ParameterVector,
    pub cost: f64,
    pub gradient_norm: Option<f64>,
    pub elapsed_time: f64,
}

/// Complete optimization history
#[derive(Clone, Debug, Default)]
pub struct OptimizationHistory {
    pub entries: Vec<HistoryEntry>,
    pub total_quantum_evaluations: usize,
}

impl OptimizationHistory {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            total_quantum_evaluations: 0,
        }
    }

    pub fn add_entry(&mut self, entry: HistoryEntry) {
        self.entries.push(entry);
    }

    pub fn best_cost(&self) -> Option<f64> {
        self.entries
            .iter()
            .map(|e| e.cost)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
    }

    pub fn final_cost(&self) -> Option<f64> {
        self.entries.last().map(|e| e.cost)
    }
}
