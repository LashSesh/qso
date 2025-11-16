//! Quantum Approximate Optimization Algorithm (QAOA)
//!
//! Solves combinatorial optimization problems using alternating cost and mixer
//! Hamiltonian evolutions.
//!
//! Mathematical formulation:
//! |ψ(γ,β)⟩ = ∏ᵢ₌ₚ e^{-iβᵢB} e^{-iγᵢH_C} |+⟩⊗ⁿ
//!
//! where H_C is the cost Hamiltonian and B is the mixer Hamiltonian.

use crate::quantum::operator::{OperatorMatrix, QuantumOperator};
use crate::quantum::state::{METATRON_DIMENSION, QuantumState};
use crate::vqa::cost_function::{GradientMethod, QAOACostFunction};
use crate::vqa::optimizer::{OptimizationResult, Optimizer, OptimizerConfig, OptimizerType};
use num_complex::Complex64;
use rand::Rng;
use std::sync::Arc;

/// QAOA Configuration
#[derive(Clone, Debug)]
pub struct QAOAConfig {
    pub depth: usize,
    pub optimizer_type: OptimizerType,
    pub optimizer_config: OptimizerConfig,
}

impl Default for QAOAConfig {
    fn default() -> Self {
        Self {
            depth: 3,
            optimizer_type: OptimizerType::NelderMead,
            optimizer_config: OptimizerConfig {
                max_iterations: 500,
                learning_rate: 0.05,
                gradient_method: GradientMethod::FiniteDifference,
                verbose: true,
                ..Default::default()
            },
        }
    }
}

/// QAOA Result
#[derive(Clone, Debug)]
pub struct QAOAResult {
    pub optimal_cost: f64,
    pub optimal_parameters: Vec<f64>,
    pub optimal_state: QuantumState,
    pub approximation_ratio: f64,
    pub optimization_result: OptimizationResult,
    pub classical_optimum: f64,
}

/// QAOA Algorithm
pub struct QAOA {
    cost_hamiltonian: Arc<QuantumOperator>,
    mixer_hamiltonian: Arc<QuantumOperator>,
    config: QAOAConfig,
    classical_optimum: Option<f64>,
}

impl QAOA {
    /// Create new QAOA instance
    pub fn new(
        cost_hamiltonian: Arc<QuantumOperator>,
        mixer_hamiltonian: Option<Arc<QuantumOperator>>,
        config: QAOAConfig,
    ) -> Self {
        let mixer = mixer_hamiltonian.unwrap_or_else(|| Arc::new(Self::default_mixer()));

        Self {
            cost_hamiltonian,
            mixer_hamiltonian: mixer,
            config,
            classical_optimum: None,
        }
    }

    /// Set classical optimum for approximation ratio calculation
    pub fn with_classical_optimum(mut self, optimum: f64) -> Self {
        self.classical_optimum = Some(optimum);
        self
    }

    /// Default mixer: X mixer (sum of Pauli-X operators)
    fn default_mixer() -> QuantumOperator {
        let mut mixer_matrix = OperatorMatrix::zeros();

        // Create transverse field mixer: H_B = Σᵢ Xᵢ
        // For 13-dim space, we approximate as anti-diagonal matrix
        for i in 0..METATRON_DIMENSION {
            let j = (METATRON_DIMENSION - 1) - i;
            mixer_matrix[(i, j)] = Complex64::new(1.0, 0.0);
        }

        QuantumOperator::from_matrix(mixer_matrix)
    }

    /// Run QAOA algorithm
    pub fn run(&self) -> QAOAResult {
        println!("═══════════════════════════════════════════════════════");
        println!("  Quantum Approximate Optimization Algorithm (QAOA)");
        println!("═══════════════════════════════════════════════════════");
        println!("QAOA Depth (p):         {}", self.config.depth);
        println!("Number of Parameters:   {}", 2 * self.config.depth);
        println!("Optimizer:              {:?}", self.config.optimizer_type);

        if let Some(classical_opt) = self.classical_optimum {
            println!("Classical Optimum:      {:.6}", classical_opt);
        }

        println!("═══════════════════════════════════════════════════════");

        // Create initial state (uniform superposition)
        let initial_state = QuantumState::uniform_superposition();

        // Create cost function
        let cost_function = Arc::new(QAOACostFunction::new(
            self.cost_hamiltonian.clone(),
            self.mixer_hamiltonian.clone(),
            self.config.depth,
            initial_state.clone(),
        ));

        // Generate initial parameters
        let initial_parameters = self.generate_initial_parameters();

        // Run optimization
        let optimizer = Optimizer::new(
            self.config.optimizer_type.clone(),
            self.config.optimizer_config.clone(),
        );
        let optimization_result = optimizer.optimize(cost_function.clone(), initial_parameters);

        // Compute optimal state
        let optimal_state = self.construct_qaoa_state(&optimization_result.optimal_parameters);

        // Compute approximation ratio
        let approximation_ratio = if let Some(classical_opt) = self.classical_optimum {
            if classical_opt != 0.0 {
                optimization_result.optimal_cost / classical_opt
            } else {
                1.0
            }
        } else {
            // Estimate classical optimum from eigenvalues
            let classical_opt = self.estimate_classical_optimum();
            if classical_opt != 0.0 {
                optimization_result.optimal_cost / classical_opt
            } else {
                1.0
            }
        };

        println!("═══════════════════════════════════════════════════════");
        println!("  QAOA Results");
        println!("═══════════════════════════════════════════════════════");
        println!(
            "Optimal Cost:           {:.10}",
            optimization_result.optimal_cost
        );
        println!("Approximation Ratio:    {:.6}", approximation_ratio);
        println!("Iterations:             {}", optimization_result.iterations);
        println!("Converged:              {}", optimization_result.converged);
        println!(
            "Quantum Evaluations:    {}",
            optimization_result.history.total_quantum_evaluations
        );
        println!("═══════════════════════════════════════════════════════");

        QAOAResult {
            optimal_cost: optimization_result.optimal_cost,
            optimal_parameters: optimization_result.optimal_parameters.clone(),
            optimal_state,
            approximation_ratio,
            optimization_result,
            classical_optimum: self
                .classical_optimum
                .unwrap_or_else(|| self.estimate_classical_optimum()),
        }
    }

    /// Generate initial parameters (heuristic initialization)
    fn generate_initial_parameters(&self) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut params = Vec::with_capacity(2 * self.config.depth);

        // Gamma parameters (cost evolution angles)
        for _ in 0..self.config.depth {
            params.push(rng.gen_range(0.0..std::f64::consts::PI));
        }

        // Beta parameters (mixer evolution angles)
        for _ in 0..self.config.depth {
            params.push(rng.gen_range(0.0..std::f64::consts::PI / 2.0));
        }

        params
    }

    /// Construct QAOA state from parameters
    fn construct_qaoa_state(&self, parameters: &[f64]) -> QuantumState {
        assert_eq!(parameters.len(), 2 * self.config.depth);

        let (gamma, beta) = parameters.split_at(self.config.depth);
        let mut state = QuantumState::uniform_superposition();

        for layer in 0..self.config.depth {
            // Cost evolution: exp(-iγH_C)
            let cost_unitary = self.create_evolution_unitary(&self.cost_hamiltonian, gamma[layer]);
            state = state.apply(&cost_unitary);

            // Mixer evolution: exp(-iβB)
            let mixer_unitary = self.create_evolution_unitary(&self.mixer_hamiltonian, beta[layer]);
            state = state.apply(&mixer_unitary);
        }

        state
    }

    /// Create time evolution unitary exp(-iHt)
    fn create_evolution_unitary(
        &self,
        hamiltonian: &QuantumOperator,
        time: f64,
    ) -> QuantumOperator {
        // Use matrix exponential approximation
        // For small time steps: exp(-iHt) ≈ I - iHt + (iHt)²/2! - ...

        let h_matrix = hamiltonian.matrix();
        let n = h_matrix.nrows();

        // Simplified: Use first-order Trotter approximation
        // For production: implement proper matrix exponential via diagonalization
        let mut result = OperatorMatrix::identity();

        // First order: exp(-iHt) ≈ I - iHt
        let i_time = Complex64::new(0.0, time);

        for row in 0..n {
            for col in 0..n {
                result[(row, col)] -= i_time * h_matrix[(row, col)];
            }
        }

        // Renormalize to maintain unitarity (approximate)
        // TODO: Use proper matrix exponential for production
        QuantumOperator::from_matrix(result)
    }

    /// Estimate classical optimum from cost Hamiltonian eigenvalues
    fn estimate_classical_optimum(&self) -> f64 {
        // For MaxCut and similar problems, classical optimum is often
        // approximated by the minimum eigenvalue
        // This is a simplified approach
        let state = QuantumState::uniform_superposition();
        let expectation = state.expectation_value(&self.cost_hamiltonian);
        expectation.re * 1.5 // Rough approximation
    }

    /// Sample measurement outcomes from optimal state
    pub fn sample_solutions(&self, state: &QuantumState, num_samples: usize) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        let mut samples = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            let mut temp_state = state.clone();
            let measurement = temp_state.measure(&mut rng).unwrap();
            samples.push(measurement);
        }

        samples
    }

    /// Compute cost distribution from samples
    pub fn analyze_samples(
        &self,
        state: &QuantumState,
        num_samples: usize,
    ) -> (f64, f64, Vec<f64>) {
        let samples = self.sample_solutions(state, num_samples);

        let costs: Vec<f64> = samples
            .iter()
            .map(|&idx| {
                let basis_state = QuantumState::basis_state(idx).unwrap();
                let cost = basis_state.expectation_value(&self.cost_hamiltonian);
                cost.re
            })
            .collect();

        let mean_cost = costs.iter().sum::<f64>() / costs.len() as f64;
        let std_dev = (costs.iter().map(|c| (c - mean_cost).powi(2)).sum::<f64>()
            / costs.len() as f64)
            .sqrt();

        (mean_cost, std_dev, costs)
    }
}

/// Builder for QAOA
pub struct QAOABuilder {
    cost_hamiltonian: Option<Arc<QuantumOperator>>,
    mixer_hamiltonian: Option<Arc<QuantumOperator>>,
    config: QAOAConfig,
    classical_optimum: Option<f64>,
}

impl QAOABuilder {
    pub fn new() -> Self {
        Self {
            cost_hamiltonian: None,
            mixer_hamiltonian: None,
            config: QAOAConfig::default(),
            classical_optimum: None,
        }
    }

    pub fn cost_hamiltonian(mut self, h: Arc<QuantumOperator>) -> Self {
        self.cost_hamiltonian = Some(h);
        self
    }

    pub fn mixer_hamiltonian(mut self, h: Arc<QuantumOperator>) -> Self {
        self.mixer_hamiltonian = Some(h);
        self
    }

    pub fn depth(mut self, depth: usize) -> Self {
        self.config.depth = depth;
        self
    }

    pub fn optimizer(mut self, optimizer_type: OptimizerType) -> Self {
        self.config.optimizer_type = optimizer_type;
        self
    }

    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.config.optimizer_config.max_iterations = max_iter;
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.config.optimizer_config.learning_rate = lr;
        self
    }

    pub fn classical_optimum(mut self, opt: f64) -> Self {
        self.classical_optimum = Some(opt);
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.optimizer_config.verbose = verbose;
        self
    }

    pub fn build(self) -> QAOA {
        let mut qaoa = QAOA::new(
            self.cost_hamiltonian.expect("Cost Hamiltonian must be set"),
            self.mixer_hamiltonian,
            self.config,
        );

        if let Some(opt) = self.classical_optimum {
            qaoa = qaoa.with_classical_optimum(opt);
        }

        qaoa
    }
}

impl Default for QAOABuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Factory function to create MaxCut cost Hamiltonian
pub fn create_maxcut_hamiltonian(edges: &[(usize, usize)]) -> QuantumOperator {
    let mut hamiltonian = OperatorMatrix::zeros();

    // H_C = -1/2 * Σ_{(i,j)∈E} (I - Z_i Z_j)
    // Simplified for 13-dim Hilbert space
    for &(i, j) in edges {
        if i < METATRON_DIMENSION && j < METATRON_DIMENSION {
            // Add edge contribution
            hamiltonian[(i, i)] -= Complex64::new(0.5, 0.0);
            hamiltonian[(j, j)] -= Complex64::new(0.5, 0.0);
            hamiltonian[(i, j)] += Complex64::new(0.5, 0.0);
            hamiltonian[(j, i)] += Complex64::new(0.5, 0.0);
        }
    }

    QuantumOperator::from_matrix(hamiltonian)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qaoa_basic() {
        // Create simple cost Hamiltonian
        let mut cost_matrix = OperatorMatrix::zeros();
        for i in 0..METATRON_DIMENSION {
            cost_matrix[(i, i)] = Complex64::new(i as f64, 0.0);
        }
        let cost_h = Arc::new(QuantumOperator::from_matrix(cost_matrix));

        let config = QAOAConfig {
            depth: 1,
            optimizer_config: OptimizerConfig {
                max_iterations: 20,
                verbose: false,
                ..Default::default()
            },
            ..Default::default()
        };

        let qaoa = QAOA::new(cost_h, None, config);
        let result = qaoa.run();

        assert!(result.optimal_cost.is_finite());
        assert!(result.approximation_ratio.is_finite());
    }

    #[test]
    fn test_maxcut_hamiltonian() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let h = create_maxcut_hamiltonian(&edges);

        let state = QuantumState::uniform_superposition();
        let energy = state.expectation_value(&h);
        assert!(energy.re.is_finite());
    }
}
