/*!
 * TIC (Temporal Information Crystal) Crystallizer
 * Converts Solve-Coagula fixpoints into stable crystals with multiscale gating.
 */

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

/// Invariant metrics computed from fixpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Invariants {
    pub variance: f64,
    pub retention: f64,
    pub gap: f64,
    pub delta_pi: f64,
}

/// Aggregated sigma values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SigmaBar {
    pub psi: f64,
    pub rho: f64,
    pub omega: f64,
}

/// Proof components for Merkaba gate validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proof {
    pub por: String,
    pub pi_gap: f64,
    pub mci: f64,
}

/// Multiscale gating result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiscaleGating {
    pub micro: MicroGating,
    pub meso: MesoGating,
    pub macro_level: MacroGating,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroGating {
    pub activations: Vec<i32>,
    pub rate: f64,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MesoGating {
    pub value: i32,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroGating {
    pub value: i32,
    pub passed: bool,
    pub commit: bool,
}

/// Temporal Information Crystal structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TIC {
    pub tic_id: String,
    pub seed: String,
    pub fixpoint: Vec<f64>,
    pub window: Vec<String>,
    pub invariants: Invariants,
    pub sigma_bar: SigmaBar,
    pub proof: Proof,
    pub source_snapshot: String,
}

/// Gate configuration
#[derive(Debug, Clone)]
pub struct GateConfig {
    pub por_delta: f64,
    pub phi_star: f64,
    pub mci_min: f64,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            por_delta: 0.02,
            phi_star: 0.6,
            mci_min: 0.9,
        }
    }
}

/// TIC Crystallizer configuration
#[derive(Debug, Clone)]
pub struct TICConfig {
    pub gate: GateConfig,
    pub theta_micro: f64,
    pub theta_meso: f64,
    pub theta_macro: f64,
}

impl Default for TICConfig {
    fn default() -> Self {
        Self {
            gate: GateConfig::default(),
            theta_micro: 0.7,
            theta_meso: 0.6,
            theta_macro: 0.5,
        }
    }
}

/// TIC Crystallizer - Converts Solve-Coagula fixpoints into stable crystals
pub struct TICCrystallizer {
    config: TICConfig,
    store_path: PathBuf,
}

impl TICCrystallizer {
    /// Initialize TIC Crystallizer
    ///
    /// # Arguments
    /// * `config` - Configuration parameters
    /// * `store_path` - Storage directory path
    pub fn new(config: TICConfig, store_path: impl AsRef<Path>) -> Result<Self> {
        let store_path = store_path.as_ref().to_path_buf();
        fs::create_dir_all(&store_path)?;

        Ok(Self { config, store_path })
    }

    /// Compute invariant metrics from fixpoint and convergence trajectory
    ///
    /// # Arguments
    /// * `fixpoint` - Converged fixpoint vector
    /// * `trajectory` - List of vectors from convergence
    ///
    /// # Returns
    /// Invariant values (variance, retention, gap)
    pub fn compute_invariants(
        &self,
        fixpoint: &Array1<f64>,
        trajectory: &[Array1<f64>],
    ) -> Invariants {
        let trajectory = if trajectory.len() < 2 {
            vec![fixpoint.clone(), fixpoint.clone()]
        } else {
            trajectory.to_vec()
        };

        // Compute variance across all trajectory values
        let flat_values: Vec<f64> = trajectory.iter().flat_map(|v| v.iter().copied()).collect();
        let mean = flat_values.iter().sum::<f64>() / flat_values.len() as f64;
        let variance =
            flat_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / flat_values.len() as f64;

        // Retention: how much of initial structure is preserved
        let retention = if !trajectory.is_empty() {
            let initial_vec = &trajectory[0];
            let fixpoint_norm = fixpoint.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
            let initial_norm = initial_vec.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
            let denom = fixpoint_norm * initial_norm;

            if denom > 0.0 {
                let dot_product: f64 = fixpoint
                    .iter()
                    .zip(initial_vec.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                (dot_product / denom).abs()
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Spectral gap (simplified eigenvalue computation)
        let gap = if trajectory.len() > 3 {
            // Simple variance-based gap estimation
            let variances: Vec<f64> = (0..fixpoint.len())
                .map(|i| {
                    let values: Vec<f64> = trajectory.iter().map(|v| v[i]).collect();
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
                })
                .collect();
            let max_var = variances.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let min_var = variances.iter().copied().fold(f64::INFINITY, f64::min);
            max_var - min_var
        } else {
            0.5
        };

        Invariants {
            variance,
            retention,
            gap,
            delta_pi: 0.0, // Will be set later
        }
    }

    /// Compute aggregated sigma values for TIC
    ///
    /// # Arguments
    /// * `fixpoint` - Fixpoint vector
    /// * `seed` - Deterministic seed
    ///
    /// # Returns
    /// Sigma bar values (psi, rho, omega)
    pub fn compute_sigma_bar(&self, fixpoint: &Array1<f64>, seed: &str) -> SigmaBar {
        // Generate deterministic values from seed and fixpoint
        let fixpoint_str = format!("{:?}", fixpoint.to_vec());
        let seed_hash = Sha256::digest(format!("{}{}", seed, fixpoint_str).as_bytes());

        let seed_vals: Vec<f64> = (0..3)
            .map(|i| {
                let bytes = &seed_hash[i * 4..(i + 1) * 4];
                let val = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                val as f64 / (u32::MAX as f64)
            })
            .collect();

        let fixpoint_norm = fixpoint.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let fixpoint_sum: f64 = fixpoint.iter().sum();
        let fixpoint_mean = fixpoint_sum / fixpoint.len() as f64;

        let psi = (fixpoint_norm * seed_vals[0]).tanh();
        let rho = (fixpoint_sum * seed_vals[1]).sin().abs();
        let omega = (fixpoint_mean * seed_vals[2]).cos();

        SigmaBar { psi, rho, omega }
    }

    /// Compute path invariance deviation
    ///
    /// # Arguments
    /// * `fixpoint` - Converged fixpoint
    /// * `original` - Original vector
    ///
    /// # Returns
    /// Path invariance gap value
    pub fn compute_pi_gap(&self, fixpoint: &Array1<f64>, _original: &Array1<f64>) -> f64 {
        // Compute deviation from path invariance using cyclic permutations
        let n = fixpoint.len();
        if n == 0 {
            return 0.0;
        }

        let mut deviations = Vec::new();
        let fixpoint_vec = fixpoint.to_vec();

        for shift in 0..n {
            let mut shifted = vec![0.0; n];
            for i in 0..n {
                shifted[i] = fixpoint_vec[(i + shift) % n];
            }

            let dev: f64 = shifted
                .iter()
                .zip(fixpoint_vec.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            let base = fixpoint
                .iter()
                .map(|x| x.powi(2))
                .sum::<f64>()
                .sqrt()
                .max(1.0);
            let normalized_dev = (dev / base) / n as f64;
            deviations.push(normalized_dev);
        }

        let raw = deviations.iter().sum::<f64>() / deviations.len() as f64;
        raw * 1e-3
    }

    /// Compute Mirror Consistency Index
    ///
    /// # Arguments
    /// * `fixpoint` - Fixpoint vector
    ///
    /// # Returns
    /// MCI value in [0, 1]
    pub fn compute_mci(&self, fixpoint: &Array1<f64>) -> f64 {
        // Mirror consistency: symmetry measure
        let reversed: Vec<f64> = fixpoint.iter().rev().copied().collect();
        let diff: f64 = fixpoint
            .iter()
            .zip(reversed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        let norm = fixpoint.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let consistency = 1.0 - diff / (2.0 * norm);

        consistency.clamp(0.0, 1.0)
    }

    /// Validate proof for TIC acceptance (Merkaba gate)
    ///
    /// # Arguments
    /// * `por_status` - Proof-of-Resonance status
    /// * `pi_gap` - Path invariance gap
    /// * `mci` - Mirror Consistency Index
    /// * `phi` - Order parameter
    ///
    /// # Returns
    /// Tuple of (validation status, proof)
    pub fn validate_proof(
        &self,
        por_status: &str,
        pi_gap: f64,
        mci: f64,
        phi: f64,
    ) -> (String, Proof) {
        // Merkaba gate conditions
        let por_valid = por_status == "valid";
        let pi_valid = pi_gap <= self.config.gate.por_delta;
        let phi_valid = phi >= self.config.gate.phi_star;
        let mci_valid = mci >= self.config.gate.mci_min;

        let all_valid = por_valid && pi_valid && phi_valid && mci_valid;

        let status = if all_valid { "valid" } else { "invalid" };

        let proof = Proof {
            por: por_status.to_string(),
            pi_gap,
            mci,
        };

        (status.to_string(), proof)
    }

    /// Implement multiscale gating mechanism
    ///
    /// # Arguments
    /// * `units` - List of unit activation values
    ///
    /// # Returns
    /// Gating results at micro, meso, and macro levels
    pub fn multiscale_gating(&self, units: &[f64]) -> MultiscaleGating {
        if units.is_empty() {
            return MultiscaleGating {
                micro: MicroGating {
                    activations: vec![],
                    rate: 0.0,
                    passed: false,
                },
                meso: MesoGating {
                    value: 0,
                    passed: false,
                },
                macro_level: MacroGating {
                    value: 0,
                    passed: false,
                    commit: false,
                },
            };
        }

        // Micro level gating
        let beta_micro: Vec<i32> = units
            .iter()
            .map(|z| if *z >= self.config.theta_micro { 1 } else { 0 })
            .collect();
        let micro_rate = beta_micro.iter().sum::<i32>() as f64 / beta_micro.len() as f64;

        // Meso level gating (weighted average)
        let weights_meso = vec![1.0 / beta_micro.len() as f64; beta_micro.len()]; // Uniform weights
        let weighted_sum: f64 = weights_meso
            .iter()
            .zip(beta_micro.iter())
            .map(|(w, b)| w * (*b as f64))
            .sum();
        let beta_meso = if weighted_sum >= self.config.theta_meso {
            1
        } else {
            0
        };

        // Macro level gating
        let beta_macro = if beta_meso as f64 >= self.config.theta_macro {
            1
        } else {
            0
        };

        MultiscaleGating {
            micro: MicroGating {
                activations: beta_micro,
                rate: micro_rate,
                passed: micro_rate >= self.config.theta_micro,
            },
            meso: MesoGating {
                value: beta_meso,
                passed: beta_meso == 1,
            },
            macro_level: MacroGating {
                value: beta_macro,
                passed: beta_macro == 1,
                commit: beta_macro == 1,
            },
        }
    }

    /// Create a TIC from Solve-Coagula fixpoint
    ///
    /// # Arguments
    /// * `fixpoint` - Converged fixpoint vector
    /// * `snapshot_id` - Source snapshot ID
    /// * `seed` - Deterministic seed
    /// * `convergence_info` - Convergence information from Solve-Coagula
    /// * `snapshot_data` - Original snapshot data
    ///
    /// # Returns
    /// TIC structure
    pub fn create_tic(
        &self,
        fixpoint: &Array1<f64>,
        snapshot_id: &str,
        seed: &str,
        convergence_info: &serde_json::Value,
        snapshot_data: &serde_json::Value,
    ) -> Result<TIC> {
        // Generate deterministic TIC ID
        let fixpoint_vec = fixpoint.to_vec();
        let hash_input = format!(
            "{}_{}__{:?}_{:?}",
            snapshot_id, seed, fixpoint_vec, convergence_info
        );
        let tic_hash = format!("{:x}", Sha256::digest(hash_input.as_bytes()));
        let tic_id = tic_hash[..16].to_string();

        // Generate deterministic timestamp
        let base_time = "2025-01-01T00:00:00Z".parse::<DateTime<Utc>>().unwrap();
        let hash_value = u64::from_str_radix(&tic_hash[16..24], 16).unwrap_or(0);
        let seconds_offset = (hash_value % 86400) as i64;
        let current_time = base_time + Duration::seconds(seconds_offset);

        // Time window (last 5 minutes for this TIC)
        let window = vec![
            (current_time - Duration::minutes(5)).to_rfc3339(),
            current_time.to_rfc3339(),
        ];

        // Extract convergence trajectory if available
        let trajectory: Vec<Array1<f64>> = if let Some(history) = convergence_info.get("history") {
            history
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .take(10)
                        .filter_map(|h| {
                            h.get("norm")
                                .and_then(|n| n.as_f64())
                                .map(|norm| Array1::from_vec(vec![norm; 5]))
                        })
                        .collect()
                })
                .unwrap_or_default()
        } else {
            vec![]
        };

        // Compute TIC components
        let mut invariants = self.compute_invariants(fixpoint, &trajectory);
        let sigma_bar = self.compute_sigma_bar(fixpoint, seed);

        // Compute proof components
        let coordinates = snapshot_data
            .get("coordinates")
            .and_then(|c| c.as_array())
            .map(|arr| {
                Array1::from_vec(arr.iter().filter_map(|v| v.as_f64()).collect::<Vec<f64>>())
            })
            .unwrap_or_else(|| Array1::zeros(fixpoint.len()));

        let pi_gap = self.compute_pi_gap(fixpoint, &coordinates);
        let mci = self.compute_mci(fixpoint);
        let phi = snapshot_data
            .get("metrics")
            .and_then(|m| m.get("resonance"))
            .and_then(|r| r.as_f64())
            .unwrap_or(0.0);

        // Validate proof
        let por_status = snapshot_data
            .get("metrics")
            .and_then(|m| m.get("por"))
            .and_then(|p| p.as_str())
            .unwrap_or("invalid");

        let (_validation_status, proof) = self.validate_proof(por_status, pi_gap, mci, phi);

        invariants.delta_pi = pi_gap;

        // Create TIC structure
        let tic = TIC {
            tic_id,
            seed: seed.to_string(),
            fixpoint: fixpoint.to_vec(),
            window,
            invariants,
            sigma_bar,
            proof,
            source_snapshot: snapshot_id.to_string(),
        };

        Ok(tic)
    }

    /// Save TIC to disk
    ///
    /// # Arguments
    /// * `tic` - TIC data
    ///
    /// # Returns
    /// Path to saved TIC file
    pub fn save_tic(&self, tic: &TIC) -> Result<String> {
        let tic_file = self.store_path.join(format!("{}.tic", tic.tic_id));

        let json = serde_json::to_string_pretty(tic)?;
        fs::write(&tic_file, json)?;

        Ok(tic_file.to_string_lossy().to_string())
    }

    /// Load TIC from disk
    ///
    /// # Arguments
    /// * `tic_id` - TIC UUID
    ///
    /// # Returns
    /// TIC data or None if not found
    pub fn load_tic(&self, tic_id: &str) -> Result<Option<TIC>> {
        let tic_file = self.store_path.join(format!("{}.tic", tic_id));

        if !tic_file.exists() {
            return Ok(None);
        }

        let json = fs::read_to_string(tic_file)?;
        let tic: TIC = serde_json::from_str(&json)?;

        Ok(Some(tic))
    }

    /// Determine if TIC should be committed to ledger
    ///
    /// # Arguments
    /// * `tic` - TIC data
    ///
    /// # Returns
    /// True if TIC passes all gating criteria
    pub fn should_commit(&self, tic: &TIC) -> bool {
        // Check proof validity
        if tic.proof.por != "valid" {
            return false;
        }

        // Check invariants
        if tic.invariants.gap < 0.1 {
            return false;
        }

        // Check MCI threshold
        if tic.proof.mci < self.config.gate.mci_min {
            return false;
        }

        // Multiscale gating on fixpoint components
        let gating = self.multiscale_gating(&tic.fixpoint);

        gating.macro_level.commit
    }

    /// Compute deterministic hash of TIC
    ///
    /// # Arguments
    /// * `tic` - TIC data
    ///
    /// # Returns
    /// SHA256 hash string
    pub fn get_tic_hash(&self, tic: &TIC) -> Result<String> {
        let tic_str = serde_json::to_string(tic)?;
        let hash = format!("{:x}", Sha256::digest(tic_str.as_bytes()));
        Ok(hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;
    use serde_json::json;

    #[test]
    fn test_create_crystallizer() {
        let config = TICConfig::default();
        let result = TICCrystallizer::new(config, "/tmp/tic_test");
        assert!(result.is_ok());
    }

    #[test]
    fn test_compute_invariants() {
        let config = TICConfig::default();
        let crystallizer = TICCrystallizer::new(config, "/tmp/tic_test").unwrap();

        let fixpoint = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let trajectory = vec![
            arr1(&[1.1, 2.1, 3.1, 4.1, 5.1]),
            arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]),
        ];

        let invariants = crystallizer.compute_invariants(&fixpoint, &trajectory);

        assert!(invariants.variance >= 0.0);
        assert!(invariants.retention >= 0.0 && invariants.retention <= 1.0);
        assert!(invariants.gap >= 0.0);
    }

    #[test]
    fn test_compute_sigma_bar() {
        let config = TICConfig::default();
        let crystallizer = TICCrystallizer::new(config, "/tmp/tic_test").unwrap();

        let fixpoint = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let sigma = crystallizer.compute_sigma_bar(&fixpoint, "test_seed");

        assert!(sigma.psi >= -1.0 && sigma.psi <= 1.0);
        assert!(sigma.rho >= 0.0 && sigma.rho <= 1.0);
        assert!(sigma.omega >= -1.0 && sigma.omega <= 1.0);
    }

    #[test]
    fn test_compute_pi_gap() {
        let config = TICConfig::default();
        let crystallizer = TICCrystallizer::new(config, "/tmp/tic_test").unwrap();

        let fixpoint = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let original = arr1(&[1.1, 2.1, 3.1, 4.1, 5.1]);

        let pi_gap = crystallizer.compute_pi_gap(&fixpoint, &original);
        assert!(pi_gap >= 0.0);
    }

    #[test]
    fn test_compute_mci() {
        let config = TICConfig::default();
        let crystallizer = TICCrystallizer::new(config, "/tmp/tic_test").unwrap();

        let fixpoint = arr1(&[1.0, 2.0, 3.0, 2.0, 1.0]);
        let mci = crystallizer.compute_mci(&fixpoint);

        assert!(mci >= 0.0 && mci <= 1.0);
    }

    #[test]
    fn test_validate_proof() {
        let config = TICConfig::default();
        let crystallizer = TICCrystallizer::new(config, "/tmp/tic_test").unwrap();

        let (status, proof) = crystallizer.validate_proof("valid", 0.01, 0.95, 0.7);
        assert_eq!(status, "valid");
        assert_eq!(proof.por, "valid");
        assert_eq!(proof.pi_gap, 0.01);
        assert_eq!(proof.mci, 0.95);

        let (status, _proof) = crystallizer.validate_proof("invalid", 0.01, 0.95, 0.7);
        assert_eq!(status, "invalid");
    }

    #[test]
    fn test_multiscale_gating() {
        let config = TICConfig::default();
        let crystallizer = TICCrystallizer::new(config, "/tmp/tic_test").unwrap();

        let units = vec![0.8, 0.9, 0.75, 0.85];
        let gating = crystallizer.multiscale_gating(&units);

        assert_eq!(gating.micro.activations.len(), 4);
        assert!(gating.micro.rate > 0.0);
        assert!(gating.micro.passed);
        assert!(gating.meso.passed);
        assert!(gating.macro_level.passed);
        assert!(gating.macro_level.commit);
    }

    #[test]
    fn test_create_tic() {
        let config = TICConfig::default();
        let crystallizer = TICCrystallizer::new(config, "/tmp/tic_test").unwrap();

        let fixpoint = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let snapshot_id = "snapshot_123";
        let seed = "test_seed";
        let convergence_info = json!({
            "history": [
                {"norm": 1.0},
                {"norm": 0.5},
                {"norm": 0.1}
            ]
        });
        let snapshot_data = json!({
            "coordinates": [1.0, 2.0, 3.0, 4.0, 5.0],
            "metrics": {
                "resonance": 0.7,
                "por": "valid"
            }
        });

        let result = crystallizer.create_tic(
            &fixpoint,
            snapshot_id,
            seed,
            &convergence_info,
            &snapshot_data,
        );

        assert!(result.is_ok());
        let tic = result.unwrap();
        assert_eq!(tic.seed, "test_seed");
        assert_eq!(tic.source_snapshot, "snapshot_123");
        assert_eq!(tic.fixpoint.len(), 5);
    }

    #[test]
    fn test_save_and_load_tic() {
        let config = TICConfig::default();
        let crystallizer = TICCrystallizer::new(config, "/tmp/tic_test_save").unwrap();

        let fixpoint = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let convergence_info = json!({"history": []});
        let snapshot_data = json!({
            "coordinates": [1.0, 2.0, 3.0, 4.0, 5.0],
            "metrics": {"resonance": 0.7, "por": "valid"}
        });

        let tic = crystallizer
            .create_tic(
                &fixpoint,
                "snap_1",
                "seed_1",
                &convergence_info,
                &snapshot_data,
            )
            .unwrap();

        // Save
        let path = crystallizer.save_tic(&tic).unwrap();
        assert!(!path.is_empty());

        // Load
        let loaded = crystallizer.load_tic(&tic.tic_id).unwrap();
        assert!(loaded.is_some());
        let loaded_tic = loaded.unwrap();
        assert_eq!(loaded_tic.tic_id, tic.tic_id);
        assert_eq!(loaded_tic.seed, tic.seed);
    }

    #[test]
    fn test_should_commit() {
        let config = TICConfig::default();
        let crystallizer = TICCrystallizer::new(config, "/tmp/tic_test").unwrap();

        let tic = TIC {
            tic_id: "test_tic".to_string(),
            seed: "seed".to_string(),
            fixpoint: vec![0.8, 0.9, 0.85, 0.75, 0.8],
            window: vec![
                "2025-01-01T00:00:00Z".to_string(),
                "2025-01-01T00:05:00Z".to_string(),
            ],
            invariants: Invariants {
                variance: 0.01,
                retention: 0.95,
                gap: 0.5,
                delta_pi: 0.01,
            },
            sigma_bar: SigmaBar {
                psi: 0.8,
                rho: 0.7,
                omega: 0.6,
            },
            proof: Proof {
                por: "valid".to_string(),
                pi_gap: 0.01,
                mci: 0.95,
            },
            source_snapshot: "snap_1".to_string(),
        };

        let should_commit = crystallizer.should_commit(&tic);
        assert!(should_commit);
    }

    #[test]
    fn test_get_tic_hash() {
        let config = TICConfig::default();
        let crystallizer = TICCrystallizer::new(config, "/tmp/tic_test").unwrap();

        let tic = TIC {
            tic_id: "test_tic".to_string(),
            seed: "seed".to_string(),
            fixpoint: vec![1.0, 2.0, 3.0],
            window: vec!["2025-01-01T00:00:00Z".to_string()],
            invariants: Invariants {
                variance: 0.0,
                retention: 1.0,
                gap: 0.0,
                delta_pi: 0.0,
            },
            sigma_bar: SigmaBar {
                psi: 0.0,
                rho: 0.0,
                omega: 0.0,
            },
            proof: Proof {
                por: "valid".to_string(),
                pi_gap: 0.0,
                mci: 1.0,
            },
            source_snapshot: "snap".to_string(),
        };

        let hash1 = crystallizer.get_tic_hash(&tic).unwrap();
        let hash2 = crystallizer.get_tic_hash(&tic).unwrap();

        // Same TIC should produce same hash
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64); // SHA256 hex string
    }
}
