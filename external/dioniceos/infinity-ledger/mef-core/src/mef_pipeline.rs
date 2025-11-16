/*!
 * MEF-Core Pipeline - Main Interface
 *
 * Provides a simplified high-level interface for the complete MEF-Core pipeline,
 * integrating all components: ingestion, spiral, solve-coagula, TIC, ledger, HDAG, and audit.
 */

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Main MEF-Core interface for simplified usage
#[derive(Debug)]
pub struct MEFCore {
    /// Deterministic seed
    pub seed: String,
    /// Configuration
    pub config: MEFCoreConfig,
}

/// Configuration for MEF-Core pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MEFCoreConfig {
    pub seed: String,
    #[serde(default)]
    pub spiral: SpiralConfig,
    #[serde(default)]
    pub solvecoagula: SolveCoagulaConfig,
    #[serde(default)]
    pub gate: GateConfig,
}

/// Spiral configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpiralConfig {
    #[serde(default = "default_r")]
    pub r: f64,
    #[serde(default = "default_a")]
    pub a: f64,
    #[serde(default = "default_b")]
    pub b: f64,
    #[serde(default = "default_c")]
    pub c: f64,
    #[serde(default = "default_k")]
    pub k: i32,
    #[serde(default = "default_step")]
    pub step: f64,
}

fn default_r() -> f64 {
    1.0
}
fn default_a() -> f64 {
    0.05
}
fn default_b() -> f64 {
    0.2
}
fn default_c() -> f64 {
    0.2
}
fn default_k() -> i32 {
    2
}
fn default_step() -> f64 {
    0.01
}

impl Default for SpiralConfig {
    fn default() -> Self {
        Self {
            r: default_r(),
            a: default_a(),
            b: default_b(),
            c: default_c(),
            k: default_k(),
            step: default_step(),
        }
    }
}

/// Solve-Coagula configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveCoagulaConfig {
    #[serde(default = "default_lambda")]
    pub lambda: f64,
    #[serde(default = "default_eps")]
    pub eps: f64,
    #[serde(default = "default_max_iter")]
    pub max_iter: usize,
    #[serde(default)]
    pub operators: OperatorsConfig,
}

fn default_lambda() -> f64 {
    0.8
}
fn default_eps() -> f64 {
    1e-6
}
fn default_max_iter() -> usize {
    1000
}

impl Default for SolveCoagulaConfig {
    fn default() -> Self {
        Self {
            lambda: default_lambda(),
            eps: default_eps(),
            max_iter: default_max_iter(),
            operators: OperatorsConfig::default(),
        }
    }
}

/// Operators configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OperatorsConfig {
    #[serde(default)]
    pub dk: DKConfig,
    #[serde(default)]
    pub sw: SWConfig,
    #[serde(default)]
    pub pi: PIConfig,
    #[serde(default)]
    pub wt: WTConfig,
}

/// DoubleKick configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DKConfig {
    #[serde(default = "default_alpha1")]
    pub alpha1: f64,
    #[serde(default = "default_alpha2")]
    pub alpha2: f64,
}

fn default_alpha1() -> f64 {
    0.05
}
fn default_alpha2() -> f64 {
    -0.03
}

impl Default for DKConfig {
    fn default() -> Self {
        Self {
            alpha1: default_alpha1(),
            alpha2: default_alpha2(),
        }
    }
}

/// Sweep configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SWConfig {
    #[serde(default = "default_tau0")]
    pub tau0: f64,
    #[serde(default = "default_beta")]
    pub beta: f64,
    #[serde(default = "default_schedule")]
    pub schedule: String,
}

fn default_tau0() -> f64 {
    0.5
}
fn default_beta() -> f64 {
    0.1
}
fn default_schedule() -> String {
    "cosine".to_string()
}

impl Default for SWConfig {
    fn default() -> Self {
        Self {
            tau0: default_tau0(),
            beta: default_beta(),
            schedule: default_schedule(),
        }
    }
}

/// Pfadinvarianz configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIConfig {
    #[serde(default = "default_canon")]
    pub canon: String,
    #[serde(default = "default_tol")]
    pub tol: f64,
}

fn default_canon() -> String {
    "lexicographic".to_string()
}
fn default_tol() -> f64 {
    1e-6
}

impl Default for PIConfig {
    fn default() -> Self {
        Self {
            canon: default_canon(),
            tol: default_tol(),
        }
    }
}

/// Weight-Transfer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WTConfig {
    #[serde(default = "default_gamma")]
    pub gamma: f64,
    #[serde(default = "default_levels")]
    pub levels: Vec<String>,
}

fn default_gamma() -> f64 {
    0.1
}
fn default_levels() -> Vec<String> {
    vec!["micro".to_string(), "meso".to_string(), "macro".to_string()]
}

impl Default for WTConfig {
    fn default() -> Self {
        Self {
            gamma: default_gamma(),
            levels: default_levels(),
        }
    }
}

/// Gate configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateConfig {
    #[serde(default = "default_por_delta")]
    pub por_delta: f64,
    #[serde(default = "default_phi_star")]
    pub phi_star: f64,
    #[serde(default = "default_mci_min")]
    pub mci_min: f64,
}

fn default_por_delta() -> f64 {
    0.02
}
fn default_phi_star() -> f64 {
    0.6
}
fn default_mci_min() -> f64 {
    0.9
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            por_delta: default_por_delta(),
            phi_star: default_phi_star(),
            mci_min: default_mci_min(),
        }
    }
}

impl Default for MEFCoreConfig {
    fn default() -> Self {
        Self::with_seed("MEF_SEED_42")
    }
}

impl MEFCoreConfig {
    /// Create a new configuration with a specific seed
    pub fn with_seed(seed: &str) -> Self {
        Self {
            seed: seed.to_string(),
            spiral: SpiralConfig::default(),
            solvecoagula: SolveCoagulaConfig::default(),
            gate: GateConfig::default(),
        }
    }
}

/// Processing result from the MEF-Core pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    pub snapshot_id: String,
    pub snapshot_phase: f64,
    pub por: f64,
    pub tic_id: String,
    pub converged: bool,
    pub iterations: usize,
    pub committed: bool,
    pub block_index: Option<usize>,
    pub block_hash: Option<String>,
}

impl MEFCore {
    /// Initialize MEF-Core with seed and optional configuration
    pub fn new(seed: &str, config: Option<MEFCoreConfig>) -> Result<Self> {
        let config = config.unwrap_or_else(|| MEFCoreConfig::with_seed(seed));

        Ok(Self {
            seed: seed.to_string(),
            config,
        })
    }

    /// Process data through complete MEF-Core pipeline
    ///
    /// Note: This is a simplified interface. The actual implementation would
    /// require initializing and coordinating all the component modules
    /// (triton, spiral, solve-coagula, TIC, ledger, HDAG, audit).
    ///
    /// # Arguments
    /// * `data` - Input data as JSON Value
    /// * `data_type` - Type of data (text, json, numeric, binary, raw)
    /// * `auto_commit` - Whether to auto-commit to ledger
    ///
    /// # Returns
    /// Processing results
    pub fn process(
        &self,
        _data: Value,
        _data_type: &str,
        _auto_commit: bool,
    ) -> Result<ProcessingResult> {
        // This would involve:
        // 1. Normalize through Triton
        // 2. Create Spiral snapshot
        // 3. Apply Solve-Coagula
        // 4. Create TIC
        // 5. Commit to ledger if valid

        // For now, return a placeholder result
        // Full implementation would require all modules to be instantiated

        Ok(ProcessingResult {
            snapshot_id: "placeholder".to_string(),
            snapshot_phase: 0.0,
            por: 0.0,
            tic_id: "placeholder".to_string(),
            converged: false,
            iterations: 0,
            committed: false,
            block_index: None,
            block_hash: None,
        })
    }

    /// Get the current configuration
    pub fn get_config(&self) -> &MEFCoreConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_mef_core() {
        let mef = MEFCore::new("MEF_SEED_42", None).unwrap();
        assert_eq!(mef.seed, "MEF_SEED_42");
        assert_eq!(mef.config.seed, "MEF_SEED_42");
    }

    #[test]
    fn test_default_config() {
        let config = MEFCoreConfig::default();
        assert_eq!(config.seed, "MEF_SEED_42");
        assert_eq!(config.spiral.r, 1.0);
        assert_eq!(config.spiral.a, 0.05);
        assert_eq!(config.solvecoagula.lambda, 0.8);
        assert_eq!(config.gate.por_delta, 0.02);
    }

    #[test]
    fn test_custom_config() {
        let mut config = MEFCoreConfig::with_seed("CUSTOM_SEED");
        config.spiral.r = 2.0;

        let mef = MEFCore::new("CUSTOM_SEED", Some(config.clone())).unwrap();
        assert_eq!(mef.config.spiral.r, 2.0);
    }

    #[test]
    fn test_serialize_config() {
        let config = MEFCoreConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("MEF_SEED_42"));
    }

    #[test]
    fn test_deserialize_config() {
        let json = r#"{
            "seed": "TEST_SEED",
            "spiral": {"r": 1.5, "a": 0.1, "b": 0.3, "c": 0.3, "k": 3, "step": 0.02},
            "solvecoagula": {"lambda": 0.9, "eps": 1e-7, "max_iter": 500, "operators": {}},
            "gate": {"por_delta": 0.03, "phi_star": 0.7, "mci_min": 0.95}
        }"#;

        let config: MEFCoreConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.seed, "TEST_SEED");
        assert_eq!(config.spiral.r, 1.5);
        assert_eq!(config.solvecoagula.lambda, 0.9);
    }

    #[test]
    fn test_process_placeholder() {
        let mef = MEFCore::new("MEF_SEED_42", None).unwrap();
        let data = serde_json::json!({"test": "data"});
        let result = mef.process(data, "json", true).unwrap();

        // Placeholder result
        assert_eq!(result.snapshot_id, "placeholder");
        assert!(!result.converged);
    }
}
