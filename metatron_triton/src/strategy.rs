//! Calibration Search Strategy Integration
//!
//! Connects the TRITON search core to the Seraphic Calibration Shell,
//! enabling TRITON to be used as an adaptive parameter optimization strategy.

use crate::{SpectralSignature, TritonSearch};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type alias for the TRITON search evaluator function
type TritonEvaluator = Box<dyn Fn(&[f64]) -> SpectralSignature + Send>;

/// Calibration parameter proposal
///
/// Represents a suggested configuration for the next calibration run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationProposal {
    /// Parameter values (mapped from spiral point)
    pub parameters: HashMap<String, f64>,

    /// Raw point from search space [0, 1]^n
    pub raw_point: Vec<f64>,

    /// Search step index
    pub step: usize,

    /// Estimated quality (if available from search)
    pub estimated_resonance: Option<f64>,
}

/// Calibration result feedback
///
/// Contains the measured metrics from a calibration run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    /// Parameter values that were tested
    pub parameters: HashMap<String, f64>,

    /// Quality metric (typically accuracy or fidelity)
    pub psi: f64,

    /// Stability metric (typically variance or consistency)
    pub rho: f64,

    /// Efficiency metric (typically runtime or resource usage)
    pub omega: f64,

    /// Additional metrics (optional)
    pub extra_metrics: HashMap<String, f64>,
}

impl CalibrationResult {
    /// Create a result from spectral signature
    pub fn from_signature(params: HashMap<String, f64>, sig: SpectralSignature) -> Self {
        Self {
            parameters: params,
            psi: sig.psi,
            rho: sig.rho,
            omega: sig.omega,
            extra_metrics: HashMap::new(),
        }
    }

    /// Convert to spectral signature
    pub fn to_signature(&self) -> SpectralSignature {
        SpectralSignature::new(self.psi, self.rho, self.omega)
    }
}

/// Trait for calibration search strategies
///
/// Implementations of this trait can be used by the Seraphic Calibration Shell
/// to guide the exploration of the parameter space.
pub trait CalibrationSearchStrategy: Send {
    /// Propose the next calibration configuration
    fn propose_next(&mut self) -> CalibrationProposal;

    /// Register the result of a calibration run
    ///
    /// This allows the strategy to learn from feedback and adapt its search.
    fn register_result(&mut self, result: &CalibrationResult);

    /// Get the best configuration found so far
    fn best_configuration(&self) -> Option<CalibrationProposal>;

    /// Get current search statistics
    fn statistics(&self) -> SearchStatistics;

    /// Reset the strategy to initial state
    fn reset(&mut self);

    /// Check if the search has converged or reached a termination criterion
    fn is_converged(&self) -> bool;
}

/// Statistics about the search progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchStatistics {
    /// Current step number
    pub step: usize,

    /// Best resonance found
    pub best_resonance: f64,

    /// Current resonance
    pub current_resonance: f64,

    /// Recent improvement rate
    pub improvement_rate: f64,

    /// Additional strategy-specific metrics
    pub extra: HashMap<String, f64>,
}

/// Parameter mapping configuration
///
/// Defines how to map the normalized search space [0, 1]^n to actual parameter ranges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterMapping {
    /// Parameter name
    pub name: String,

    /// Minimum value
    pub min: f64,

    /// Maximum value
    pub max: f64,

    /// Transform type (linear, logarithmic, etc.)
    pub transform: TransformType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TransformType {
    /// Linear mapping: min + (max - min) * x
    Linear,

    /// Logarithmic mapping: exp(log(min) + (log(max) - log(min)) * x)
    Logarithmic,

    /// Integer mapping: round(Linear)
    Integer,
}

impl ParameterMapping {
    /// Create a linear parameter mapping
    pub fn linear(name: impl Into<String>, min: f64, max: f64) -> Self {
        Self {
            name: name.into(),
            min,
            max,
            transform: TransformType::Linear,
        }
    }

    /// Create a logarithmic parameter mapping
    pub fn logarithmic(name: impl Into<String>, min: f64, max: f64) -> Self {
        assert!(
            min > 0.0 && max > 0.0,
            "Logarithmic mapping requires positive bounds"
        );
        Self {
            name: name.into(),
            min,
            max,
            transform: TransformType::Logarithmic,
        }
    }

    /// Create an integer parameter mapping
    pub fn integer(name: impl Into<String>, min: i32, max: i32) -> Self {
        Self {
            name: name.into(),
            min: min as f64,
            max: max as f64,
            transform: TransformType::Integer,
        }
    }

    /// Map a normalized value [0, 1] to the parameter range
    pub fn map(&self, normalized: f64) -> f64 {
        let clamped = normalized.clamp(0.0, 1.0);

        match self.transform {
            TransformType::Linear => self.min + (self.max - self.min) * clamped,
            TransformType::Logarithmic => {
                let log_min = self.min.ln();
                let log_max = self.max.ln();
                (log_min + (log_max - log_min) * clamped).exp()
            }
            TransformType::Integer => (self.min + (self.max - self.min) * clamped).round(),
        }
    }

    /// Inverse map: parameter value to normalized [0, 1]
    pub fn inverse_map(&self, value: f64) -> f64 {
        match self.transform {
            TransformType::Linear => ((value - self.min) / (self.max - self.min)).clamp(0.0, 1.0),
            TransformType::Logarithmic => {
                let log_min = self.min.ln();
                let log_max = self.max.ln();
                ((value.ln() - log_min) / (log_max - log_min)).clamp(0.0, 1.0)
            }
            TransformType::Integer => {
                let rounded = value.round();
                ((rounded - self.min) / (self.max - self.min)).clamp(0.0, 1.0)
            }
        }
    }
}

/// TRITON search strategy for calibration
///
/// Wraps the TRITON search engine and provides parameter mapping for Q⊗DASH calibration.
pub struct TritonSearchStrategy {
    /// Parameter mappings
    mappings: Vec<ParameterMapping>,

    /// TRITON search engine
    search: TritonSearch<TritonEvaluator>,

    /// Last proposed configuration
    last_proposal: Option<CalibrationProposal>,

    /// Convergence tracking
    convergence_patience: usize,
    convergence_threshold: f64,
    no_improvement_count: usize,
    prev_best_resonance: f64,
}

impl TritonSearchStrategy {
    /// Create a new TRITON calibration strategy
    ///
    /// # Arguments
    /// * `mappings` - Parameter mappings from [0, 1] to actual ranges
    /// * `seed` - Random seed
    /// * `max_steps` - Maximum optimization steps
    /// * `convergence_threshold` - Minimum improvement to reset patience counter
    /// * `convergence_patience` - Steps without improvement before declaring convergence
    pub fn new(
        mappings: Vec<ParameterMapping>,
        seed: u64,
        max_steps: usize,
        convergence_threshold: f64,
        convergence_patience: usize,
    ) -> Self {
        let dimension = mappings.len();

        // Create a dummy evaluator (will be updated via register_result)
        let evaluator: TritonEvaluator =
            Box::new(|_params: &[f64]| SpectralSignature::new(0.5, 0.5, 0.5));

        let search = TritonSearch::new(dimension, seed, max_steps, evaluator);

        Self {
            mappings,
            search,
            last_proposal: None,
            convergence_patience,
            convergence_threshold,
            no_improvement_count: 0,
            prev_best_resonance: 0.0,
        }
    }

    /// Create with custom spiral parameters
    #[allow(clippy::too_many_arguments)]
    pub fn with_spiral_params(
        mappings: Vec<ParameterMapping>,
        seed: u64,
        max_steps: usize,
        convergence_threshold: f64,
        convergence_patience: usize,
        radius_base: f64,
        learning_rate: f64,
        momentum_decay: f64,
        noise_level: f64,
    ) -> Self {
        let dimension = mappings.len();

        let evaluator: TritonEvaluator =
            Box::new(|_params: &[f64]| SpectralSignature::new(0.5, 0.5, 0.5));

        let search = TritonSearch::with_spiral_params(
            dimension,
            seed,
            max_steps,
            evaluator,
            radius_base,
            learning_rate,
            momentum_decay,
            noise_level,
        );

        Self {
            mappings,
            search,
            last_proposal: None,
            convergence_patience,
            convergence_threshold,
            no_improvement_count: 0,
            prev_best_resonance: 0.0,
        }
    }

    /// Map a raw point [0, 1]^n to actual parameters
    fn map_point(&self, point: &[f64]) -> HashMap<String, f64> {
        assert_eq!(point.len(), self.mappings.len());

        self.mappings
            .iter()
            .zip(point.iter())
            .map(|(mapping, &val)| (mapping.name.clone(), mapping.map(val)))
            .collect()
    }
}

impl CalibrationSearchStrategy for TritonSearchStrategy {
    fn propose_next(&mut self) -> CalibrationProposal {
        // Run one TRITON step (evaluation will happen via register_result)
        let result = self.search.step();

        // Map raw point to parameters
        let parameters = self.map_point(&result.point);

        let proposal = CalibrationProposal {
            parameters,
            raw_point: result.point,
            step: result.step_index,
            estimated_resonance: Some(result.best_resonance),
        };

        self.last_proposal = Some(proposal.clone());
        proposal
    }

    fn register_result(&mut self, result: &CalibrationResult) {
        let sig = result.to_signature();
        let resonance = sig.resonance();

        // Update convergence tracking
        let improvement = resonance - self.prev_best_resonance;
        if improvement < self.convergence_threshold {
            self.no_improvement_count += 1;
        } else {
            self.no_improvement_count = 0;
            self.prev_best_resonance = resonance;
        }

        tracing::debug!(
            "TRITON strategy: Step {}, resonance = {:.6}, improvement = {:.6}",
            self.search.current_step(),
            resonance,
            improvement
        );
    }

    fn best_configuration(&self) -> Option<CalibrationProposal> {
        self.search.best_point().map(|point| {
            let parameters = self.map_point(point);
            let best_resonance = self.search.best_signature().map(|s| s.resonance());

            CalibrationProposal {
                parameters,
                raw_point: point.to_vec(),
                step: self.search.current_step(),
                estimated_resonance: best_resonance,
            }
        })
    }

    fn statistics(&self) -> SearchStatistics {
        let best_resonance = self
            .search
            .best_signature()
            .map(|s| s.resonance())
            .unwrap_or(0.0);

        let history = self.search.resonance_history();
        let current_resonance = history.last().copied().unwrap_or(0.0);

        let improvement_rate = self.search.average_improvement_rate(10);

        let mut extra = HashMap::new();
        extra.insert(
            "no_improvement_count".to_string(),
            self.no_improvement_count as f64,
        );
        extra.insert(
            "convergence_patience".to_string(),
            self.convergence_patience as f64,
        );

        SearchStatistics {
            step: self.search.current_step(),
            best_resonance,
            current_resonance,
            improvement_rate,
            extra,
        }
    }

    fn reset(&mut self) {
        self.search.reset();
        self.last_proposal = None;
        self.no_improvement_count = 0;
        self.prev_best_resonance = 0.0;
    }

    fn is_converged(&self) -> bool {
        self.no_improvement_count >= self.convergence_patience || self.search.is_finished()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_mapping_linear() {
        let mapping = ParameterMapping::linear("learning_rate", 0.001, 0.1);
        assert_eq!(mapping.map(0.0), 0.001);
        assert_eq!(mapping.map(1.0), 0.1);
        assert!((mapping.map(0.5) - 0.0505).abs() < 1e-10);
    }

    #[test]
    fn test_parameter_mapping_logarithmic() {
        let mapping = ParameterMapping::logarithmic("learning_rate", 1e-4, 1e-1);
        let val = mapping.map(0.5);
        // Should be geometric mean: sqrt(1e-4 * 1e-1) = sqrt(1e-5)
        let expected = (1e-4_f64 * 1e-1_f64).sqrt();
        assert!((val - expected).abs() < 1e-10);
    }

    #[test]
    fn test_parameter_mapping_integer() {
        let mapping = ParameterMapping::integer("depth", 1, 10);
        assert_eq!(mapping.map(0.0), 1.0);
        assert_eq!(mapping.map(1.0), 10.0);
        assert_eq!(mapping.map(0.5), 6.0); // Rounded
    }

    #[test]
    fn test_triton_strategy_creation() {
        let mappings = vec![
            ParameterMapping::linear("lr", 0.001, 0.1),
            ParameterMapping::integer("depth", 1, 10),
            ParameterMapping::logarithmic("noise", 1e-4, 1e-1),
        ];

        let strategy = TritonSearchStrategy::new(mappings, 42, 100, 1e-6, 50);
        assert!(!strategy.is_converged());
    }

    #[test]
    fn test_triton_strategy_propose() {
        let mappings = vec![
            ParameterMapping::linear("param1", 0.0, 1.0),
            ParameterMapping::linear("param2", 0.0, 1.0),
        ];

        let mut strategy = TritonSearchStrategy::new(mappings, 42, 100, 1e-6, 50);
        let proposal = strategy.propose_next();

        assert_eq!(proposal.parameters.len(), 2);
        assert!(proposal.parameters.contains_key("param1"));
        assert!(proposal.parameters.contains_key("param2"));
    }

    #[test]
    fn test_triton_strategy_convergence() {
        let mappings = vec![ParameterMapping::linear("param", 0.0, 1.0)];

        let mut strategy = TritonSearchStrategy::new(mappings, 42, 100, 1e-6, 10);

        // Simulate calibration runs with no improvement
        for _ in 0..15 {
            strategy.propose_next();

            // Always return same mediocre result
            let result = CalibrationResult {
                parameters: HashMap::new(),
                psi: 0.5,
                rho: 0.5,
                omega: 0.5,
                extra_metrics: HashMap::new(),
            };
            strategy.register_result(&result);
        }

        // Should converge after patience runs out
        assert!(strategy.is_converged());
    }
}
