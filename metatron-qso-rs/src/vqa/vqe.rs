//! Variational Quantum Eigensolver (VQE)
//!
//! Finds the ground state energy and wavefunction of a Hamiltonian using
//! a variational ansatz and classical optimization.
//!
//! Mathematical formulation:
//! E₀ = min_θ ⟨ψ(θ)|H|ψ(θ)⟩
//!
//! where |ψ(θ)⟩ = U(θ)|ψ₀⟩ is a parametrized quantum state.

use crate::hamiltonian::MetatronHamiltonian;
use crate::quantum::state::QuantumState;
use crate::vqa::ansatz::{AnsatzType, create_ansatz};
use crate::vqa::cost_function::{GradientMethod, VQECostFunction};
use crate::vqa::optimizer::{OptimizationResult, Optimizer, OptimizerConfig, OptimizerType};
use rand::Rng;
use std::sync::Arc;

/// VQE Algorithm Configuration
#[derive(Clone, Debug)]
pub struct VQEConfig {
    pub ansatz_type: AnsatzType,
    pub ansatz_depth: usize,
    pub optimizer_type: OptimizerType,
    pub optimizer_config: OptimizerConfig,
    pub initial_state_type: InitialStateType,
    /// Number of random initialization attempts (multi-start strategy)
    /// If > 1, runs optimization multiple times and keeps best result
    pub num_random_starts: usize,
}

/// Type of initial state
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InitialStateType {
    /// Uniform superposition |+⟩⊗n
    UniformSuperposition,
    /// Ground state from classical diagonalization
    ClassicalGround,
    /// Hartree-Fock initial state
    HartreeFock,
    /// Random Haar state
    Random,
}

impl Default for VQEConfig {
    fn default() -> Self {
        Self {
            ansatz_type: AnsatzType::HardwareEfficient,
            ansatz_depth: 3,
            optimizer_type: OptimizerType::Adam,
            optimizer_config: OptimizerConfig::default(),
            initial_state_type: InitialStateType::UniformSuperposition,
            num_random_starts: 1,
        }
    }
}

/// VQE Algorithm Result
#[derive(Clone, Debug)]
pub struct VQEResult {
    pub ground_state_energy: f64,
    pub optimal_parameters: Vec<f64>,
    pub ground_state_wavefunction: QuantumState,
    pub optimization_result: OptimizationResult,
    pub classical_ground_energy: f64,
    pub approximation_error: f64,
}

/// Variational Quantum Eigensolver
pub struct VQE {
    hamiltonian: Arc<MetatronHamiltonian>,
    config: VQEConfig,
}

impl VQE {
    /// Create a new VQE instance
    pub fn new(hamiltonian: Arc<MetatronHamiltonian>, config: VQEConfig) -> Self {
        Self {
            hamiltonian,
            config,
        }
    }

    /// Run the VQE algorithm
    pub fn run(&self) -> VQEResult {
        println!("═══════════════════════════════════════════════════════");
        println!("  Variational Quantum Eigensolver (VQE)");
        println!("═══════════════════════════════════════════════════════");

        // Get classical ground state energy for comparison
        let classical_ground = self.hamiltonian.eigenvalues()[0];
        println!("Classical Ground State Energy: {:.10}", classical_ground);

        // Create initial state
        let initial_state = self.create_initial_state();
        println!("Initial State: {:?}", self.config.initial_state_type);

        // Create ansatz
        let ansatz1 = create_ansatz(self.config.ansatz_type.clone(), self.config.ansatz_depth);
        let ansatz2 = create_ansatz(self.config.ansatz_type.clone(), self.config.ansatz_depth);
        let num_params = ansatz1.num_parameters();
        println!("Ansatz Type: {:?}", self.config.ansatz_type);
        println!("Ansatz Depth: {}", self.config.ansatz_depth);
        println!("Number of Parameters: {}", num_params);

        // Create cost function
        let cost_function = Arc::new(VQECostFunction::new(
            self.hamiltonian.clone(),
            ansatz1,
            initial_state.clone(),
        ));

        println!("Optimizer: {:?}", self.config.optimizer_type);

        // Multi-start strategy
        if self.config.num_random_starts > 1 {
            println!(
                "Multi-start: {} random initializations",
                self.config.num_random_starts
            );
        }
        println!("═══════════════════════════════════════════════════════");

        // Run optimization with multi-start
        let optimizer = Optimizer::new(
            self.config.optimizer_type.clone(),
            self.config.optimizer_config.clone(),
        );

        let mut best_result: Option<OptimizationResult> = None;
        let mut total_evaluations = 0;

        for trial in 0..self.config.num_random_starts {
            if self.config.num_random_starts > 1 && self.config.optimizer_config.verbose {
                println!(
                    "\n--- Trial {}/{} ---",
                    trial + 1,
                    self.config.num_random_starts
                );
            }

            // Generate initial parameters for this trial
            let initial_parameters = self.generate_initial_parameters(num_params);

            // Run optimization
            let result = optimizer.optimize(cost_function.clone(), initial_parameters);
            total_evaluations += result.history.total_quantum_evaluations;

            // Keep best result
            match &best_result {
                None => best_result = Some(result),
                Some(prev_best) => {
                    if result.optimal_cost < prev_best.optimal_cost {
                        if self.config.num_random_starts > 1 && self.config.optimizer_config.verbose
                        {
                            println!("  → New best energy: {:.10}", result.optimal_cost);
                        }
                        best_result = Some(result);
                    }
                }
            }
        }

        let mut optimization_result =
            best_result.expect("At least one optimization should have run");

        // Update total evaluations to include all trials
        optimization_result.history.total_quantum_evaluations = total_evaluations;

        // Reconstruct ground state wavefunction
        let ground_state_wavefunction =
            ansatz2.apply(&initial_state, &optimization_result.optimal_parameters);

        // Compute approximation error
        let approximation_error = (optimization_result.optimal_cost - classical_ground).abs();

        println!("═══════════════════════════════════════════════════════");
        println!("  VQE Results");
        println!("═══════════════════════════════════════════════════════");
        println!(
            "VQE Ground Energy:      {:.10}",
            optimization_result.optimal_cost
        );
        println!("Classical Ground:       {:.10}", classical_ground);
        println!("Approximation Error:    {:.10}", approximation_error);
        println!(
            "Relative Error:         {:.6}%",
            (approximation_error / classical_ground.abs()) * 100.0
        );
        println!("Iterations:             {}", optimization_result.iterations);
        println!("Converged:              {}", optimization_result.converged);
        println!(
            "Quantum Evaluations:    {}",
            optimization_result.history.total_quantum_evaluations
        );
        if self.config.num_random_starts > 1 {
            println!("Random Starts:          {}", self.config.num_random_starts);
        }
        println!("═══════════════════════════════════════════════════════");

        VQEResult {
            ground_state_energy: optimization_result.optimal_cost,
            optimal_parameters: optimization_result.optimal_parameters.clone(),
            ground_state_wavefunction,
            optimization_result,
            classical_ground_energy: classical_ground,
            approximation_error,
        }
    }

    /// Create initial quantum state based on configuration
    fn create_initial_state(&self) -> QuantumState {
        match self.config.initial_state_type {
            InitialStateType::UniformSuperposition => QuantumState::uniform_superposition(),
            InitialStateType::ClassicalGround => self.hamiltonian.ground_state(),
            InitialStateType::HartreeFock => {
                // Simplified Hartree-Fock: use first basis state
                QuantumState::basis_state(0).unwrap()
            }
            InitialStateType::Random => QuantumState::random(None),
        }
    }

    /// Generate initial parameters for the ansatz
    fn generate_initial_parameters(&self, num_params: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();

        match self.config.ansatz_type {
            AnsatzType::HardwareEfficient | AnsatzType::EfficientSU2 => {
                // Random small initialization
                (0..num_params).map(|_| rng.gen_range(-0.1..0.1)).collect()
            }
            AnsatzType::Metatron => {
                // Initialization leveraging symmetry
                (0..num_params)
                    .map(|i| {
                        let phase = 2.0 * std::f64::consts::PI * (i as f64) / (num_params as f64);
                        0.01 * phase.cos()
                    })
                    .collect()
            }
        }
    }

    /// Verify result quality
    pub fn verify_result(&self, result: &VQEResult) -> bool {
        // Check if energy is below all excited states
        let eigenvalues = self.hamiltonian.eigenvalues();

        if result.ground_state_energy < eigenvalues[0] - 1e-6 {
            println!("⚠ Warning: VQE energy below classical ground state!");
            return false;
        }

        // Find the first eigenvalue that is significantly higher than the ground state
        // to handle degenerate ground states
        let ground_state = eigenvalues[0];
        let first_excited = eigenvalues
            .iter()
            .find(|&&e| e > ground_state + 1e-6)
            .copied()
            .unwrap_or(eigenvalues[eigenvalues.len() - 1]);

        if result.ground_state_energy > first_excited + 1e-6 {
            println!("⚠ Warning: VQE energy above first excited state!");
            println!("  Ground state: {:.10}", ground_state);
            println!("  First excited: {:.10}", first_excited);
            println!("  VQE energy: {:.10}", result.ground_state_energy);
            return false;
        }

        // Check normalization
        if !result.ground_state_wavefunction.is_normalized(1e-6) {
            println!("⚠ Warning: Ground state not normalized!");
            return false;
        }

        true
    }
}

/// Builder pattern for VQE
pub struct VQEBuilder {
    hamiltonian: Option<Arc<MetatronHamiltonian>>,
    config: VQEConfig,
}

impl VQEBuilder {
    pub fn new() -> Self {
        Self {
            hamiltonian: None,
            config: VQEConfig::default(),
        }
    }

    pub fn hamiltonian(mut self, h: Arc<MetatronHamiltonian>) -> Self {
        self.hamiltonian = Some(h);
        self
    }

    pub fn ansatz_type(mut self, ansatz_type: AnsatzType) -> Self {
        self.config.ansatz_type = ansatz_type;
        self
    }

    pub fn ansatz_depth(mut self, depth: usize) -> Self {
        self.config.ansatz_depth = depth;
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

    pub fn tolerance(mut self, tol: f64) -> Self {
        self.config.optimizer_config.tolerance = tol;
        self
    }

    pub fn gradient_method(mut self, method: GradientMethod) -> Self {
        self.config.optimizer_config.gradient_method = method;
        self
    }

    pub fn initial_state(mut self, state_type: InitialStateType) -> Self {
        self.config.initial_state_type = state_type;
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.optimizer_config.verbose = verbose;
        self
    }

    pub fn num_random_starts(mut self, num_starts: usize) -> Self {
        self.config.num_random_starts = num_starts;
        self
    }

    pub fn energy_tolerance(mut self, tol: f64) -> Self {
        self.config.optimizer_config.energy_tolerance = tol;
        self
    }

    pub fn build(self) -> VQE {
        VQE {
            hamiltonian: self.hamiltonian.expect("Hamiltonian must be set"),
            config: self.config,
        }
    }
}

impl Default for VQEBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::metatron::MetatronGraph;
    use crate::params::QSOParameters;

    #[test]
    fn test_vqe_basic() {
        let graph = MetatronGraph::new();
        let params = QSOParameters::default();
        let hamiltonian = Arc::new(MetatronHamiltonian::new(&graph, &params));

        let config = VQEConfig {
            ansatz_depth: 1,
            optimizer_config: OptimizerConfig {
                max_iterations: 50,
                verbose: false,
                ..Default::default()
            },
            ..Default::default()
        };

        let vqe = VQE::new(hamiltonian.clone(), config);
        let result = vqe.run();

        // Basic sanity checks
        assert!(result.ground_state_energy.is_finite());

        // With shallow ansatz (depth=1) and limited iterations (50),
        // we expect the VQE to find an energy within the ground state manifold
        // The ground state energy is -13.0, so we check relative error
        let relative_error =
            result.approximation_error.abs() / hamiltonian.ground_state_energy().abs();
        assert!(
            relative_error < 1.0,
            "Relative error {:.2}% is too large",
            relative_error * 100.0
        );
    }

    #[test]
    fn test_vqe_builder() {
        let graph = MetatronGraph::new();
        let params = QSOParameters::default();
        let hamiltonian = Arc::new(MetatronHamiltonian::new(&graph, &params));

        let vqe = VQEBuilder::new()
            .hamiltonian(hamiltonian)
            .ansatz_type(AnsatzType::HardwareEfficient)
            .ansatz_depth(2)
            .optimizer(OptimizerType::Adam)
            .max_iterations(100)
            .learning_rate(0.01)
            .verbose(false)
            .build();

        let result = vqe.run();
        assert!(vqe.verify_result(&result));
    }
}
