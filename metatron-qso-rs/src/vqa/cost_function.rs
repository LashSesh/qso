//! Cost Functions for Variational Quantum Algorithms
//!
//! Provides objective functions with gradient computation via Parameter Shift Rule.
//! Supports VQE (eigenvalue), QAOA (optimization), and VQC (classification) cost functions.

use crate::hamiltonian::MetatronHamiltonian;
use crate::quantum::operator::QuantumOperator;
use crate::quantum::state::QuantumState;
use crate::vqa::ParameterVector;
use crate::vqa::ansatz::Ansatz;
use num_complex::Complex64;
use rayon::prelude::*;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

/// Gradient computation methods
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GradientMethod {
    /// Parameter Shift Rule (exact, hardware-friendly)
    ParameterShift,
    /// Finite differences (numerical approximation)
    FiniteDifference,
    /// No gradient computation
    None,
}

/// Base trait for all cost functions
pub trait CostFunction: Send + Sync {
    /// Evaluate the cost function at given parameters
    fn evaluate(&self, parameters: &[f64]) -> f64;

    /// Compute gradient using specified method
    fn gradient(&self, parameters: &[f64], method: GradientMethod) -> ParameterVector;

    /// Compute Hessian (second derivatives) for advanced optimizers
    fn hessian(&self, parameters: &[f64]) -> Vec<Vec<f64>> {
        // Default: finite difference approximation
        let n = parameters.len();
        let mut hessian = vec![vec![0.0; n]; n];
        let h = 1e-5;

        for i in 0..n {
            for j in i..n {
                let mut params_pp = parameters.to_vec();
                let mut params_pm = parameters.to_vec();
                let mut params_mp = parameters.to_vec();
                let mut params_mm = parameters.to_vec();

                params_pp[i] += h;
                params_pp[j] += h;
                params_pm[i] += h;
                params_pm[j] -= h;
                params_mp[i] -= h;
                params_mp[j] += h;
                params_mm[i] -= h;
                params_mm[j] -= h;

                let f_pp = self.evaluate(&params_pp);
                let f_pm = self.evaluate(&params_pm);
                let f_mp = self.evaluate(&params_mp);
                let f_mm = self.evaluate(&params_mm);

                hessian[i][j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h);
                hessian[j][i] = hessian[i][j];
            }
        }

        hessian
    }

    /// Get problem dimension
    fn dimension(&self) -> usize;
}

/// VQE Cost Function: ⟨ψ(θ)|H|ψ(θ)⟩
///
/// Computes expectation value of Hamiltonian for finding ground state energy
pub struct VQECostFunction<A: Ansatz> {
    hamiltonian: Arc<MetatronHamiltonian>,
    ansatz: A,
    initial_state: QuantumState,
    cache: Arc<Mutex<HashMap<String, f64>>>,
}

impl<A: Ansatz> VQECostFunction<A> {
    pub fn new(
        hamiltonian: Arc<MetatronHamiltonian>,
        ansatz: A,
        initial_state: QuantumState,
    ) -> Self {
        Self {
            hamiltonian,
            ansatz,
            initial_state,
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn params_to_key(&self, parameters: &[f64]) -> String {
        parameters
            .iter()
            .map(|p| format!("{:.10}", p))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Parameter Shift Rule for single parameter
    fn parameter_shift_single(&self, parameters: &[f64], param_idx: usize) -> f64 {
        let mut params_plus = parameters.to_vec();
        let mut params_minus = parameters.to_vec();

        params_plus[param_idx] += PI / 2.0;
        params_minus[param_idx] -= PI / 2.0;

        let f_plus = self.evaluate(&params_plus);
        let f_minus = self.evaluate(&params_minus);

        (f_plus - f_minus) / 2.0
    }
}

impl<A: Ansatz> CostFunction for VQECostFunction<A> {
    fn evaluate(&self, parameters: &[f64]) -> f64 {
        // Check cache first
        let key = self.params_to_key(parameters);
        {
            let cache = self.cache.lock().unwrap();
            if let Some(&value) = cache.get(&key) {
                return value;
            }
        }

        // Apply ansatz to initial state
        let psi = self.ansatz.apply(&self.initial_state, parameters);

        // Compute ⟨ψ|H|ψ⟩
        let h_operator = QuantumOperator::from_matrix(self.hamiltonian.as_complex_operator());
        let expectation = psi.expectation_value(&h_operator);
        let energy = expectation.re;

        // Cache result
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(key, energy);
        }

        energy
    }

    fn gradient(&self, parameters: &[f64], method: GradientMethod) -> ParameterVector {
        match method {
            GradientMethod::ParameterShift => {
                // Parallel computation of gradient components
                (0..parameters.len())
                    .into_par_iter()
                    .map(|i| self.parameter_shift_single(parameters, i))
                    .collect()
            }
            GradientMethod::FiniteDifference => {
                let h = 1e-7;
                (0..parameters.len())
                    .into_par_iter()
                    .map(|i| {
                        let mut params_plus = parameters.to_vec();
                        params_plus[i] += h;
                        let f_plus = self.evaluate(&params_plus);
                        let f_0 = self.evaluate(parameters);
                        (f_plus - f_0) / h
                    })
                    .collect()
            }
            GradientMethod::None => vec![0.0; parameters.len()],
        }
    }

    fn dimension(&self) -> usize {
        self.ansatz.num_parameters()
    }
}

/// QAOA Cost Function: ⟨ψ(γ,β)|H_C|ψ(γ,β)⟩
///
/// For combinatorial optimization problems
pub struct QAOACostFunction {
    cost_hamiltonian: Arc<QuantumOperator>,
    mixer_hamiltonian: Arc<QuantumOperator>,
    depth: usize,
    initial_state: QuantumState,
    cache: Arc<Mutex<HashMap<String, f64>>>,
}

impl QAOACostFunction {
    pub fn new(
        cost_hamiltonian: Arc<QuantumOperator>,
        mixer_hamiltonian: Arc<QuantumOperator>,
        depth: usize,
        initial_state: QuantumState,
    ) -> Self {
        Self {
            cost_hamiltonian,
            mixer_hamiltonian,
            depth,
            initial_state,
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn params_to_key(&self, parameters: &[f64]) -> String {
        parameters
            .iter()
            .map(|p| format!("{:.10}", p))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Apply QAOA circuit: alternating cost and mixer evolutions
    fn apply_qaoa_circuit(&self, gamma: &[f64], beta: &[f64]) -> QuantumState {
        let mut state = self.initial_state.clone();

        for layer in 0..self.depth {
            // Cost Hamiltonian evolution: exp(-iγH_C)
            let cost_evolution =
                self.create_evolution_operator(&self.cost_hamiltonian, gamma[layer]);
            state = state.apply(&cost_evolution);

            // Mixer Hamiltonian evolution: exp(-iβB)
            let mixer_evolution =
                self.create_evolution_operator(&self.mixer_hamiltonian, beta[layer]);
            state = state.apply(&mixer_evolution);
        }

        state
    }

    /// Create time evolution operator exp(-iHt) using first-order approximation
    fn create_evolution_operator(
        &self,
        hamiltonian: &QuantumOperator,
        time: f64,
    ) -> QuantumOperator {
        // For simplicity: exp(-iHt) ≈ I - iHt (first order)
        // For production: use proper matrix exponential
        let matrix = hamiltonian.matrix();
        let mut evolved = matrix.clone();

        for i in 0..evolved.nrows() {
            for j in 0..evolved.ncols() {
                if i == j {
                    evolved[(i, j)] =
                        Complex64::new(1.0, 0.0) - Complex64::new(0.0, 1.0) * matrix[(i, j)] * time;
                } else {
                    evolved[(i, j)] = -Complex64::new(0.0, 1.0) * matrix[(i, j)] * time;
                }
            }
        }

        QuantumOperator::from_matrix(evolved)
    }
}

impl CostFunction for QAOACostFunction {
    fn evaluate(&self, parameters: &[f64]) -> f64 {
        assert_eq!(
            parameters.len(),
            2 * self.depth,
            "QAOA requires 2*depth parameters"
        );

        // Check cache
        let key = self.params_to_key(parameters);
        {
            let cache = self.cache.lock().unwrap();
            if let Some(&value) = cache.get(&key) {
                return value;
            }
        }

        let (gamma, beta) = parameters.split_at(self.depth);
        let state = self.apply_qaoa_circuit(gamma, beta);

        // Compute ⟨ψ|H_C|ψ⟩
        let expectation = state.expectation_value(&self.cost_hamiltonian);
        let cost = expectation.re;

        // Cache result
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(key, cost);
        }

        cost
    }

    fn gradient(&self, parameters: &[f64], method: GradientMethod) -> ParameterVector {
        match method {
            GradientMethod::ParameterShift => (0..parameters.len())
                .into_par_iter()
                .map(|i| {
                    let mut params_plus = parameters.to_vec();
                    let mut params_minus = parameters.to_vec();
                    params_plus[i] += PI / 2.0;
                    params_minus[i] -= PI / 2.0;
                    (self.evaluate(&params_plus) - self.evaluate(&params_minus)) / 2.0
                })
                .collect(),
            GradientMethod::FiniteDifference => {
                let h = 1e-7;
                (0..parameters.len())
                    .into_par_iter()
                    .map(|i| {
                        let mut params_plus = parameters.to_vec();
                        params_plus[i] += h;
                        (self.evaluate(&params_plus) - self.evaluate(parameters)) / h
                    })
                    .collect()
            }
            GradientMethod::None => vec![0.0; parameters.len()],
        }
    }

    fn dimension(&self) -> usize {
        2 * self.depth
    }
}

/// VQC Cost Function: Classification loss
///
/// Binary cross-entropy for quantum classification
pub struct VQCCostFunction<A: Ansatz> {
    ansatz: A,
    training_data: Vec<QuantumState>,
    training_labels: Vec<f64>,
    cache: Arc<Mutex<HashMap<String, f64>>>,
}

impl<A: Ansatz> VQCCostFunction<A> {
    pub fn new(ansatz: A, training_data: Vec<QuantumState>, training_labels: Vec<f64>) -> Self {
        assert_eq!(training_data.len(), training_labels.len());
        Self {
            ansatz,
            training_data,
            training_labels,
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn params_to_key(&self, parameters: &[f64]) -> String {
        parameters
            .iter()
            .map(|p| format!("{:.10}", p))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Predict probability for class 0
    fn predict(&self, state: &QuantumState, parameters: &[f64]) -> f64 {
        let output_state = self.ansatz.apply(state, parameters);
        let probs = output_state.probabilities();
        probs[0] // Probability of measuring |0⟩
    }

    /// Binary cross-entropy loss
    fn binary_cross_entropy(&self, prediction: f64, label: f64) -> f64 {
        let epsilon = 1e-10;
        let pred = prediction.clamp(epsilon, 1.0 - epsilon);
        -label * pred.ln() - (1.0 - label) * (1.0 - pred).ln()
    }
}

impl<A: Ansatz> CostFunction for VQCCostFunction<A> {
    fn evaluate(&self, parameters: &[f64]) -> f64 {
        // Check cache
        let key = self.params_to_key(parameters);
        {
            let cache = self.cache.lock().unwrap();
            if let Some(&value) = cache.get(&key) {
                return value;
            }
        }

        let total_loss: f64 = self
            .training_data
            .iter()
            .zip(self.training_labels.iter())
            .map(|(state, &label)| {
                let prediction = self.predict(state, parameters);
                self.binary_cross_entropy(prediction, label)
            })
            .sum();

        let avg_loss = total_loss / self.training_data.len() as f64;

        // Cache result
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(key, avg_loss);
        }

        avg_loss
    }

    fn gradient(&self, parameters: &[f64], method: GradientMethod) -> ParameterVector {
        match method {
            GradientMethod::ParameterShift => (0..parameters.len())
                .into_par_iter()
                .map(|i| {
                    let mut params_plus = parameters.to_vec();
                    let mut params_minus = parameters.to_vec();
                    params_plus[i] += PI / 2.0;
                    params_minus[i] -= PI / 2.0;
                    (self.evaluate(&params_plus) - self.evaluate(&params_minus)) / 2.0
                })
                .collect(),
            GradientMethod::FiniteDifference => {
                let h = 1e-7;
                (0..parameters.len())
                    .into_par_iter()
                    .map(|i| {
                        let mut params_plus = parameters.to_vec();
                        params_plus[i] += h;
                        (self.evaluate(&params_plus) - self.evaluate(parameters)) / h
                    })
                    .collect()
            }
            GradientMethod::None => vec![0.0; parameters.len()],
        }
    }

    fn dimension(&self) -> usize {
        self.ansatz.num_parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::metatron::MetatronGraph;
    use crate::params::QSOParameters;
    use crate::vqa::ansatz::{AnsatzType, create_ansatz};

    #[test]
    fn test_vqe_cost_evaluation() {
        let graph = MetatronGraph::new();
        let params = QSOParameters::default();
        let hamiltonian = Arc::new(MetatronHamiltonian::new(&graph, &params));
        let ansatz = create_ansatz(AnsatzType::HardwareEfficient, 1);
        let initial_state = QuantumState::uniform_superposition();

        let cost_fn = VQECostFunction::new(hamiltonian.clone(), ansatz, initial_state);

        let parameters = vec![0.1; cost_fn.dimension()];
        let energy = cost_fn.evaluate(&parameters);

        assert!(energy.is_finite());
    }

    #[test]
    fn test_parameter_shift_rule() {
        let graph = MetatronGraph::new();
        let params = QSOParameters::default();
        let hamiltonian = Arc::new(MetatronHamiltonian::new(&graph, &params));
        let ansatz = create_ansatz(AnsatzType::HardwareEfficient, 1);
        let initial_state = QuantumState::uniform_superposition();

        let cost_fn = VQECostFunction::new(hamiltonian, ansatz, initial_state);

        let parameters = vec![0.5; cost_fn.dimension()];
        let gradient = cost_fn.gradient(&parameters, GradientMethod::ParameterShift);

        assert_eq!(gradient.len(), parameters.len());
        assert!(gradient.iter().all(|g| g.is_finite()));
    }
}
