//! Variational Quantum Classifier (VQC)
//!
//! Quantum machine learning for binary and multi-class classification.
//! Uses data encoding in quantum states and parametrized quantum circuits
//! for prediction.
//!
//! Mathematical formulation:
//! L(θ) = Σᵢ loss(y_i, P₀(x_i, θ))
//! where P₀(x, θ) = |⟨0|U(x,θ)|ψ⟩|²

use crate::quantum::state::QuantumState;
use crate::vqa::ansatz::{Ansatz, AnsatzType, create_ansatz};
use crate::vqa::cost_function::{GradientMethod, VQCCostFunction};
use crate::vqa::optimizer::{OptimizationResult, Optimizer, OptimizerConfig, OptimizerType};
use rand::Rng;
use std::sync::Arc;

/// VQC Configuration
#[derive(Clone, Debug)]
pub struct VQCConfig {
    pub ansatz_type: AnsatzType,
    pub ansatz_depth: usize,
    pub optimizer_type: OptimizerType,
    pub optimizer_config: OptimizerConfig,
    pub encoding_type: EncodingType,
}

/// Data encoding type
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EncodingType {
    /// Amplitude encoding: data directly as amplitudes
    Amplitude,
    /// Angle encoding: data as rotation angles
    Angle,
    /// Basis encoding: data as basis state superposition
    Basis,
}

impl Default for VQCConfig {
    fn default() -> Self {
        Self {
            ansatz_type: AnsatzType::HardwareEfficient,
            ansatz_depth: 2,
            optimizer_type: OptimizerType::Adam,
            optimizer_config: OptimizerConfig {
                max_iterations: 500,
                learning_rate: 0.01,
                gradient_method: GradientMethod::ParameterShift,
                verbose: true,
                tolerance: 1e-4,
                energy_tolerance: 1e-3,
            },
            encoding_type: EncodingType::Angle,
        }
    }
}

/// VQC Training Result
#[derive(Clone, Debug)]
pub struct VQCResult {
    pub optimal_parameters: Vec<f64>,
    pub training_accuracy: f64,
    pub training_loss: f64,
    pub optimization_result: OptimizationResult,
}

/// VQC Prediction Result
#[derive(Clone, Debug)]
pub struct VQCPrediction {
    pub class_probabilities: Vec<f64>,
    pub predicted_class: usize,
    pub confidence: f64,
}

/// Variational Quantum Classifier
pub struct VQC {
    config: VQCConfig,
    ansatz: Box<dyn Ansatz>,
    optimal_parameters: Option<Vec<f64>>,
    // Normalization parameters learned from training data
    feature_min: Option<Vec<f64>>,
    feature_max: Option<Vec<f64>>,
}

impl VQC {
    /// Create new VQC instance
    pub fn new(config: VQCConfig) -> Self {
        let ansatz = create_ansatz(config.ansatz_type.clone(), config.ansatz_depth);

        Self {
            config,
            ansatz,
            optimal_parameters: None,
            feature_min: None,
            feature_max: None,
        }
    }

    /// Train the classifier on training data
    pub fn train(
        &mut self,
        training_data: Vec<Vec<f64>>,
        training_labels: Vec<usize>,
    ) -> VQCResult {
        println!("═══════════════════════════════════════════════════════");
        println!("  Variational Quantum Classifier (VQC)");
        println!("═══════════════════════════════════════════════════════");
        println!("Training Samples:       {}", training_data.len());
        println!("Feature Dimension:      {}", training_data[0].len());
        println!("Ansatz Type:            {:?}", self.config.ansatz_type);
        println!("Ansatz Depth:           {}", self.config.ansatz_depth);
        println!("Encoding Type:          {:?}", self.config.encoding_type);
        println!("Number of Parameters:   {}", self.ansatz.num_parameters());
        println!("Optimizer:              {:?}", self.config.optimizer_type);
        println!("═══════════════════════════════════════════════════════");

        // Compute and store normalization parameters
        let (normalized_data, min_vals, max_vals) = self.fit_normalize_data(&training_data);
        self.feature_min = Some(min_vals);
        self.feature_max = Some(max_vals);

        // Encode training data as quantum states
        let encoded_states: Vec<QuantumState> = normalized_data
            .iter()
            .map(|data| self.encode_data(data))
            .collect();

        // Convert labels to probabilities (binary classification)
        let label_probs: Vec<f64> = training_labels
            .iter()
            .map(|&label| if label == 0 { 1.0 } else { 0.0 })
            .collect();

        // Create cost function - we need to box the ansatz for polymorphism
        // For now, we'll use a different approach with trait objects
        struct AnsatzWrapper {
            inner: Box<dyn Ansatz>,
        }

        impl Ansatz for AnsatzWrapper {
            fn apply(&self, state: &QuantumState, parameters: &[f64]) -> QuantumState {
                self.inner.apply(state, parameters)
            }

            fn num_parameters(&self) -> usize {
                self.inner.num_parameters()
            }

            fn ansatz_type(&self) -> AnsatzType {
                self.inner.ansatz_type()
            }

            fn depth(&self) -> usize {
                self.inner.depth()
            }
        }

        let wrapped_ansatz = AnsatzWrapper {
            inner: create_ansatz(self.config.ansatz_type.clone(), self.config.ansatz_depth),
        };

        let cost_function = Arc::new(VQCCostFunction::new(
            wrapped_ansatz,
            encoded_states.clone(),
            label_probs.clone(),
        ));

        // Generate initial parameters
        let initial_parameters = self.generate_initial_parameters();

        // Run optimization
        let optimizer = Optimizer::new(
            self.config.optimizer_type.clone(),
            self.config.optimizer_config.clone(),
        );
        let optimization_result = optimizer.optimize(cost_function.clone(), initial_parameters);

        // Store optimal parameters
        self.optimal_parameters = Some(optimization_result.optimal_parameters.clone());

        // Compute training accuracy
        let predictions: Vec<usize> = encoded_states
            .iter()
            .map(|state| {
                let pred = self.predict_with_params(state, &optimization_result.optimal_parameters);
                pred.predicted_class
            })
            .collect();

        let correct = predictions
            .iter()
            .zip(training_labels.iter())
            .filter(|(pred, label)| **pred == **label)
            .count();

        let training_accuracy = correct as f64 / training_labels.len() as f64;
        let training_loss = optimization_result.optimal_cost;

        println!("═══════════════════════════════════════════════════════");
        println!("  VQC Training Results");
        println!("═══════════════════════════════════════════════════════");
        println!("Training Accuracy:      {:.2}%", training_accuracy * 100.0);
        println!("Training Loss:          {:.6}", training_loss);
        println!("Iterations:             {}", optimization_result.iterations);
        println!("Converged:              {}", optimization_result.converged);
        println!(
            "Quantum Evaluations:    {}",
            optimization_result.history.total_quantum_evaluations
        );
        println!("═══════════════════════════════════════════════════════");

        VQCResult {
            optimal_parameters: optimization_result.optimal_parameters.clone(),
            training_accuracy,
            training_loss,
            optimization_result,
        }
    }

    /// Predict class for new data
    pub fn predict(&self, data: &[f64]) -> VQCPrediction {
        let params = self
            .optimal_parameters
            .as_ref()
            .expect("Model not trained. Call train() first.");

        // Normalize data using learned parameters
        let normalized = self.transform_data(data);

        let state = self.encode_data(&normalized);
        self.predict_with_params(&state, params)
    }

    /// Predict using specific parameters (for training)
    fn predict_with_params(&self, state: &QuantumState, parameters: &[f64]) -> VQCPrediction {
        // Apply ansatz to encoded state
        let output_state = self.ansatz.apply(state, parameters);

        // Get probabilities
        let probs = output_state.probabilities();

        // For binary classification: P(class 0) vs P(class 1)
        let prob_class_0 = probs[0];
        let prob_class_1 = 1.0 - prob_class_0;

        let class_probabilities = vec![prob_class_0, prob_class_1];
        let predicted_class = if prob_class_0 > prob_class_1 { 0 } else { 1 };
        let confidence = class_probabilities[predicted_class];

        VQCPrediction {
            class_probabilities,
            predicted_class,
            confidence,
        }
    }

    /// Encode classical data as quantum state
    fn encode_data(&self, data: &[f64]) -> QuantumState {
        match self.config.encoding_type {
            EncodingType::Amplitude => self.amplitude_encoding(data),
            EncodingType::Angle => self.angle_encoding(data),
            EncodingType::Basis => self.basis_encoding(data),
        }
    }

    /// Amplitude encoding: data directly as state amplitudes
    fn amplitude_encoding(&self, data: &[f64]) -> QuantumState {
        use num_complex::Complex64;

        // Pad or truncate data to match Hilbert space dimension
        let mut amplitudes =
            vec![Complex64::new(0.0, 0.0); crate::quantum::state::METATRON_DIMENSION];

        for (i, &value) in data.iter().take(amplitudes.len()).enumerate() {
            amplitudes[i] = Complex64::new(value, 0.0);
        }

        // Normalize
        let norm: f64 = amplitudes.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        if norm > 0.0 {
            for amp in amplitudes.iter_mut() {
                *amp /= norm;
            }
        }

        QuantumState::try_new(&amplitudes, false).unwrap()
    }

    /// Angle encoding: data as rotation angles
    ///
    /// Encodes classical data into quantum state via parameterized rotations.
    /// Uses RY rotations for feature encoding with proper normalization.
    ///
    /// Strategy:
    /// 1. Start from |0⟩ state
    /// 2. Apply Hadamard-like operation for superposition
    /// 3. Apply feature-dependent RY rotations
    ///
    /// Feature i is encoded as: RY(π * value_i) on qubit i
    fn angle_encoding(&self, data: &[f64]) -> QuantumState {
        use crate::quantum::operator::{OperatorMatrix, QuantumOperator};
        use num_complex::Complex64;
        use std::f64::consts::PI;

        // Start from basis state |0⟩ for better classification
        let mut state = QuantumState::basis_state(0).unwrap();

        // Apply Hadamard-like operation for initial superposition
        let mut hadamard = OperatorMatrix::identity();
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        for i in 0..crate::quantum::state::METATRON_DIMENSION.min(2) {
            hadamard[(i, i)] = Complex64::new(sqrt2_inv, 0.0);
            if i + 1 < crate::quantum::state::METATRON_DIMENSION {
                hadamard[(i, i + 1)] = Complex64::new(sqrt2_inv, 0.0);
                hadamard[(i + 1, i)] = Complex64::new(sqrt2_inv, 0.0);
                hadamard[(i + 1, i + 1)] = Complex64::new(-sqrt2_inv, 0.0);
            }
        }
        let h_op = QuantumOperator::from_matrix(hadamard);
        state = state.apply(&h_op);

        // Apply feature-dependent rotations (RY gates)
        for (i, &value) in data.iter().enumerate() {
            if i >= crate::quantum::state::METATRON_DIMENSION - 1 {
                break;
            }

            // Map normalized data [0,1] to rotation angle [0, π]
            // This creates better separation for classification
            let angle = value * PI;

            // Create RY rotation matrix
            let mut rotation = OperatorMatrix::identity();
            let cos_half = (angle / 2.0).cos();
            let sin_half = (angle / 2.0).sin();

            rotation[(i, i)] = Complex64::new(cos_half, 0.0);
            rotation[(i, i + 1)] = Complex64::new(-sin_half, 0.0);
            rotation[(i + 1, i)] = Complex64::new(sin_half, 0.0);
            rotation[(i + 1, i + 1)] = Complex64::new(cos_half, 0.0);

            let operator = QuantumOperator::from_matrix(rotation);
            state = state.apply(&operator);
        }

        state
    }

    /// Basis encoding: data as superposition of basis states
    fn basis_encoding(&self, data: &[f64]) -> QuantumState {
        use num_complex::Complex64;

        // Interpret data as coefficients for basis state superposition
        let mut amplitudes =
            vec![Complex64::new(0.0, 0.0); crate::quantum::state::METATRON_DIMENSION];

        // Create superposition based on data values
        for (i, &value) in data.iter().take(amplitudes.len()).enumerate() {
            if value > 0.5 {
                amplitudes[i] = Complex64::new(1.0, 0.0);
            }
        }

        // If all zeros, use uniform superposition
        if amplitudes.iter().all(|a| a.norm_sqr() == 0.0) {
            return QuantumState::uniform_superposition();
        }

        QuantumState::try_new(&amplitudes, true).unwrap()
    }

    /// Fit normalization parameters and transform training data
    ///
    /// Computes min and max for each feature dimension and returns:
    /// (normalized_data, min_values, max_values)
    fn fit_normalize_data(&self, data: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        if data.is_empty() {
            return (vec![], vec![], vec![]);
        }

        let num_features = data[0].len();
        let mut min_vals = vec![f64::INFINITY; num_features];
        let mut max_vals = vec![f64::NEG_INFINITY; num_features];

        // Find min and max for each feature
        for sample in data {
            for (i, &value) in sample.iter().enumerate() {
                min_vals[i] = min_vals[i].min(value);
                max_vals[i] = max_vals[i].max(value);
            }
        }

        // Normalize each sample
        let normalized: Vec<Vec<f64>> = data
            .iter()
            .map(|sample| {
                sample
                    .iter()
                    .enumerate()
                    .map(|(i, &value)| {
                        let range = max_vals[i] - min_vals[i];
                        if range < 1e-10 {
                            0.5
                        } else {
                            (value - min_vals[i]) / range
                        }
                    })
                    .collect()
            })
            .collect();

        (normalized, min_vals, max_vals)
    }

    /// Transform new data using fitted normalization parameters
    fn transform_data(&self, data: &[f64]) -> Vec<f64> {
        let min_vals = self.feature_min.as_ref().expect("Model not fitted");
        let max_vals = self.feature_max.as_ref().expect("Model not fitted");

        data.iter()
            .enumerate()
            .map(|(i, &value)| {
                let range = max_vals[i] - min_vals[i];
                if range < 1e-10 {
                    0.5
                } else {
                    (value - min_vals[i]) / range
                }
            })
            .collect()
    }

    /// Generate initial parameters
    fn generate_initial_parameters(&self) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        (0..self.ansatz.num_parameters())
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect()
    }

    /// Evaluate model on test data
    pub fn evaluate(&self, test_data: Vec<Vec<f64>>, test_labels: Vec<usize>) -> f64 {
        let predictions: Vec<usize> = test_data
            .iter()
            .map(|data| self.predict(data).predicted_class)
            .collect();

        let correct = predictions
            .iter()
            .zip(test_labels.iter())
            .filter(|(pred, label)| **pred == **label)
            .count();

        correct as f64 / test_labels.len() as f64
    }
}

/// Builder for VQC
pub struct VQCBuilder {
    config: VQCConfig,
}

impl VQCBuilder {
    pub fn new() -> Self {
        Self {
            config: VQCConfig::default(),
        }
    }

    pub fn ansatz_type(mut self, ansatz_type: AnsatzType) -> Self {
        self.config.ansatz_type = ansatz_type;
        self
    }

    pub fn ansatz_depth(mut self, depth: usize) -> Self {
        self.config.ansatz_depth = depth;
        self
    }

    pub fn encoding(mut self, encoding_type: EncodingType) -> Self {
        self.config.encoding_type = encoding_type;
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

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.optimizer_config.verbose = verbose;
        self
    }

    pub fn build(self) -> VQC {
        VQC::new(self.config)
    }
}

impl Default for VQCBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vqc_basic() {
        let config = VQCConfig {
            ansatz_depth: 1,
            optimizer_config: OptimizerConfig {
                max_iterations: 50,
                verbose: false,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut vqc = VQC::new(config);

        // Simple linearly separable data
        let training_data = vec![
            vec![0.1, 0.1, 0.0, 0.0],
            vec![0.9, 0.9, 0.0, 0.0],
            vec![0.1, 0.2, 0.0, 0.0],
            vec![0.8, 0.9, 0.0, 0.0],
        ];
        let training_labels = vec![0, 1, 0, 1];

        let result = vqc.train(training_data, training_labels);

        assert!(result.training_accuracy >= 0.5);
        assert!(result.training_loss.is_finite());
    }

    #[test]
    fn test_vqc_prediction() {
        let mut vqc = VQCBuilder::new()
            .ansatz_depth(1)
            .max_iterations(30)
            .verbose(false)
            .build();

        let training_data = vec![vec![0.0, 0.0, 0.0, 0.0], vec![1.0, 1.0, 0.0, 0.0]];
        let training_labels = vec![0, 1];

        vqc.train(training_data, training_labels);

        let prediction = vqc.predict(&[0.1, 0.1, 0.0, 0.0]);
        assert!(prediction.predicted_class <= 1);
        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
    }
}
