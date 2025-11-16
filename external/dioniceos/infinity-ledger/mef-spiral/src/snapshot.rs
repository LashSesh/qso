//! Spiral Snapshot implementation for 5D storage.
//! Deterministic transformation and addressing system.
//!
//! Migrated from: MEF-Core_v1.0/src/spiral/snapshot.py

use anyhow::{Context, Result};
use chrono::Duration;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};
use std::f64::consts::PI;
use std::path::{Path, PathBuf};

/// Configuration for spiral snapshot system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpiralConfig {
    /// Radius parameter
    pub r: f64,
    /// Linear component parameter
    pub a: f64,
    /// First oscillation parameter
    pub b: f64,
    /// Second oscillation parameter
    pub c: f64,
    /// Frequency multiplier
    pub k: i32,
    /// Step size for phase search
    pub step: f64,
    /// PoR delta threshold
    #[serde(default = "default_por_delta")]
    pub por_delta: f64,
}

fn default_por_delta() -> f64 {
    0.02
}

impl Default for SpiralConfig {
    fn default() -> Self {
        Self {
            r: 1.0,
            a: 0.05,
            b: 0.2,
            c: 0.2,
            k: 2,
            step: 0.01,
            por_delta: 0.02,
        }
    }
}

/// Sigma values for resonance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sigma {
    pub psi: f64,
    pub rho: f64,
    pub omega: f64,
}

/// Snapshot metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    pub resonance: f64,
    pub stability: f64,
    pub por: String,
}

/// Snapshot payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Payload {
    pub data: JsonValue,
    #[serde(rename = "type")]
    pub data_type: String,
}

/// Spiral snapshot structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub id: String,
    pub timestamp: String,
    pub seed: String,
    pub phase: f64,
    pub coordinates: Vec<f64>,
    pub sigma: Sigma,
    pub metrics: Metrics,
    pub payload: Payload,
    pub hdag_node: String,
}

/// 5D Spiral storage implementation with deterministic addressing
pub struct SpiralSnapshot {
    config: SpiralConfig,
    store_path: PathBuf,
    sequence: std::cell::Cell<u64>,
}

impl SpiralSnapshot {
    /// Create a new SpiralSnapshot instance
    ///
    /// # Arguments
    /// * `config` - Spiral configuration parameters
    /// * `store_path` - Path to storage directory
    pub fn new(config: SpiralConfig, store_path: impl AsRef<Path>) -> Result<Self> {
        let store_path = store_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&store_path).context("Failed to create store directory")?;

        Ok(Self {
            config,
            store_path,
            sequence: std::cell::Cell::new(0),
        })
    }

    /// Compute 5D spiral coordinates for given phase
    /// s(θ) = (r cos θ, r sin θ, aθ, b sin(kθ), c cos(kθ))
    ///
    /// # Arguments
    /// * `theta` - Phase parameter
    /// * `seed` - Deterministic seed for modifications
    pub fn compute_coordinates(&self, theta: f64, seed: &str) -> Vec<f64> {
        // Generate deterministic modification from seed
        let mut hasher = Sha256::new();
        hasher.update(seed.as_bytes());
        let seed_hash = hasher.finalize();

        let seed_mod = u32::from_be_bytes([seed_hash[0], seed_hash[1], seed_hash[2], seed_hash[3]])
            as f64
            / (2_u64.pow(32) as f64);

        // Apply seed-based deterministic modification
        let r_mod = self.config.r * (1.0 + 0.1 * seed_mod);

        vec![
            r_mod * theta.cos(),                                  // x1
            r_mod * theta.sin(),                                  // x2
            self.config.a * theta,                                // x3
            self.config.b * (self.config.k as f64 * theta).sin(), // x4
            self.config.c * (self.config.k as f64 * theta).cos(), // x5
        ]
    }

    /// Compute sigma values for resonance metrics
    ///
    /// # Arguments
    /// * `coords` - 5D coordinates
    /// * `seed` - Deterministic seed
    pub fn compute_sigma(&self, coords: &[f64], seed: &str) -> Sigma {
        // Deterministic sigma computation
        let mut hasher = Sha256::new();
        hasher.update(seed.as_bytes());
        let seed_hash = hasher.finalize();

        let seed_vals: Vec<f64> = (0..3)
            .map(|i| {
                let offset = i * 4;
                u32::from_be_bytes([
                    seed_hash[offset],
                    seed_hash[offset + 1],
                    seed_hash[offset + 2],
                    seed_hash[offset + 3],
                ]) as f64
                    / (2_u64.pow(32) as f64)
            })
            .collect();

        let coords_array = Array1::from_vec(coords.to_vec());

        // Compute deterministic sigma values
        let dot_product = coords_array.dot(&coords_array);
        let psi = (dot_product * seed_vals[0]).tanh();

        let sum: f64 = coords_array.sum();
        let rho = (sum * seed_vals[1]).sin().abs();

        let prod: f64 = coords_array.iter().take(3).map(|x| x.abs()).product();
        let omega = (prod * seed_vals[2]).cos();

        Sigma { psi, rho, omega }
    }

    /// Compute resonance metric for stability validation
    ///
    /// # Arguments
    /// * `coords` - 5D coordinates
    /// * `sigma` - Sigma values
    pub fn compute_resonance(&self, coords: &[f64], sigma: &Sigma) -> f64 {
        let coords_array = Array1::from_vec(coords.to_vec());

        // Resonance function: F(q, θ) = σ(⟨u(q), u(θ)⟩ + α·κ(q,θ) - β·d_T(q,θ))
        let inner_product = coords_array.dot(&coords_array);
        let norm_squared: f64 = coords_array.iter().map(|x| x * x).sum();
        let kernel_value = (-norm_squared / 2.0).exp();
        let topology_distance = norm_squared.sqrt();

        let resonance = (inner_product * sigma.psi + 0.5 * kernel_value
            - 0.3 * topology_distance / (1.0 + topology_distance))
            .tanh();

        resonance.abs()
    }

    /// Compute stability metric for snapshot
    ///
    /// # Arguments
    /// * `coords` - 5D coordinates
    /// * `resonance` - Resonance value
    pub fn compute_stability(&self, coords: &[f64], resonance: f64) -> f64 {
        if coords.len() >= 3 {
            // Create a small Laplacian for local neighborhood
            let local_laplacian = Array2::from_shape_vec(
                (3, 3),
                vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0],
            )
            .unwrap();

            // Compute eigenvalues (simplified - using trace-based approximation)
            // For a proper implementation, we'd use ndarray-linalg
            // For determinism, we use a fixed calculation
            let _trace = local_laplacian.diag().sum();
            let spectral_gap = 1.0; // Simplified for determinism

            (spectral_gap * resonance).tanh().abs()
        } else {
            (resonance * 0.8).abs()
        }
    }

    /// Proof-of-Resonance validation
    ///
    /// # Arguments
    /// * `coords` - 5D coordinates
    /// * `resonance` - Computed resonance
    /// * `delta` - Acceptance threshold
    pub fn validate_por(&self, coords: &[f64], resonance: f64, delta: f64) -> String {
        // Simplified FFT validation (deterministic approximation)
        let coords_array = Array1::from_vec(coords.to_vec());

        // Approximate FFT magnitude using basic statistics
        let mean = coords_array.mean().unwrap_or(0.0);
        let variance: f64 =
            coords_array.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / coords.len() as f64;
        let std_dev = variance.sqrt();

        let fft_resonance = mean.abs() / (1.0 + std_dev);

        // Simplified spectral gap (deterministic)
        let spectral_gap = 1.0;

        // Validation criteria
        let resonance_check = (fft_resonance - resonance).abs() <= delta;
        let gap_check = spectral_gap >= 0.1;

        if resonance_check && gap_check {
            "valid".to_string()
        } else {
            "invalid".to_string()
        }
    }

    /// Create a new Spiral Snapshot from raw data
    ///
    /// # Arguments
    /// * `data` - Raw input data as JSON
    /// * `seed` - Deterministic seed
    /// * `phase` - Optional phase parameter (auto-computed if None)
    pub fn create_snapshot(
        &self,
        data: &JsonValue,
        seed: &str,
        phase: Option<f64>,
    ) -> Result<Snapshot> {
        // Generate snapshot ID
        let base_payload = serde_json::to_string(&data)?;
        let id_source = format!("{}_{}", base_payload, seed);
        let mut hasher = Sha256::new();
        hasher.update(id_source.as_bytes());
        let snapshot_id = format!("{:x}", hasher.finalize())[..16].to_string();

        // Generate deterministic timestamp
        let base_time = chrono::NaiveDate::from_ymd_opt(2025, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();

        let sequence = self.sequence.get();
        let sequence_offset = Duration::minutes((5 * sequence) as i64);

        let mut time_hasher = Sha256::new();
        time_hasher.update(format!("{}{}", snapshot_id, seed).as_bytes());
        let time_hash = time_hasher.finalize();
        let hash_offset_secs =
            u32::from_be_bytes([time_hash[0], time_hash[1], time_hash[2], time_hash[3]]) % 300;
        let hash_offset = Duration::seconds(hash_offset_secs as i64);

        let timestamp = (base_time + sequence_offset + hash_offset)
            .format("%Y-%m-%dT%H:%M:%S")
            .to_string();

        self.sequence.set(sequence + 1);

        // Compute phase if not provided
        let phase = if let Some(p) = phase {
            p
        } else {
            let data_str = serde_json::to_string(&data)?;
            let mut phase_hasher = Sha256::new();
            phase_hasher.update(format!("{}{}", data_str, seed).as_bytes());
            let phase_hash = phase_hasher.finalize();
            let phase_int =
                u32::from_be_bytes([phase_hash[0], phase_hash[1], phase_hash[2], phase_hash[3]]);
            (phase_int as f64 / (2_u64.pow(32) as f64)) * 2.0 * PI
        };

        // Compute spiral coordinates
        let coordinates = self.compute_coordinates(phase, seed);

        // Compute sigma values
        let sigma = self.compute_sigma(&coordinates, seed);

        // Compute metrics
        let resonance = self.compute_resonance(&coordinates, &sigma);
        let stability = self.compute_stability(&coordinates, resonance);
        let por_status = self.validate_por(&coordinates, resonance, self.config.por_delta);

        // Create snapshot structure
        Ok(Snapshot {
            id: snapshot_id.clone(),
            timestamp,
            seed: seed.to_string(),
            phase,
            coordinates,
            sigma,
            metrics: Metrics {
                resonance,
                stability,
                por: por_status,
            },
            payload: Payload {
                data: data.clone(),
                data_type: "JsonValue".to_string(),
            },
            hdag_node: format!("N-{}", snapshot_id),
        })
    }

    /// Save snapshot to disk
    ///
    /// # Arguments
    /// * `snapshot` - Snapshot data
    pub fn save_snapshot(&self, snapshot: &Snapshot) -> Result<PathBuf> {
        let snapshot_file = self.store_path.join(format!("{}.spiral", snapshot.id));

        let json =
            serde_json::to_string_pretty(snapshot).context("Failed to serialize snapshot")?;
        std::fs::write(&snapshot_file, json).context("Failed to write snapshot file")?;

        Ok(snapshot_file)
    }

    /// Load snapshot from disk
    ///
    /// # Arguments
    /// * `snapshot_id` - UUID of snapshot
    pub fn load_snapshot(&self, snapshot_id: &str) -> Result<Option<Snapshot>> {
        let snapshot_file = self.store_path.join(format!("{}.spiral", snapshot_id));

        if !snapshot_file.exists() {
            return Ok(None);
        }

        let contents =
            std::fs::read_to_string(&snapshot_file).context("Failed to read snapshot file")?;
        let snapshot: Snapshot =
            serde_json::from_str(&contents).context("Failed to deserialize snapshot")?;

        Ok(Some(snapshot))
    }

    /// Find optimal phase for query using resonance maximization
    /// θ* = arg max F(q, θ)
    ///
    /// # Arguments
    /// * `seed` - Deterministic seed
    pub fn find_optimal_phase(&self, seed: &str) -> f64 {
        let mut best_phase = 0.0;
        let mut best_resonance = f64::NEG_INFINITY;

        // Search over phase space
        for i in 0..100 {
            let test_phase = (i as f64) * 2.0 * PI / 100.0;
            let coords = self.compute_coordinates(test_phase, seed);
            let sigma = self.compute_sigma(&coords, seed);
            let resonance = self.compute_resonance(&coords, &sigma);

            if resonance > best_resonance {
                best_resonance = resonance;
                best_phase = test_phase;
            }
        }

        best_phase
    }

    /// Compute deterministic hash of snapshot
    ///
    /// # Arguments
    /// * `snapshot` - Snapshot data
    pub fn get_snapshot_hash(&self, snapshot: &Snapshot) -> Result<String> {
        let snapshot_str =
            serde_json::to_string(snapshot).context("Failed to serialize snapshot for hashing")?;

        let mut hasher = Sha256::new();
        hasher.update(snapshot_str.as_bytes());
        Ok(format!("{:x}", hasher.finalize()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_compute_coordinates() {
        let config = SpiralConfig::default();
        let temp_dir = std::env::temp_dir().join("test_spiral");
        let spiral = SpiralSnapshot::new(config, &temp_dir).unwrap();

        let coords = spiral.compute_coordinates(0.0, "MEF_SEED_42");
        assert_eq!(coords.len(), 5);

        // Determinism test
        let coords2 = spiral.compute_coordinates(0.0, "MEF_SEED_42");
        assert_eq!(coords, coords2);
    }

    #[test]
    fn test_create_snapshot() {
        let config = SpiralConfig::default();
        let temp_dir = std::env::temp_dir().join("test_spiral_create");
        let spiral = SpiralSnapshot::new(config, &temp_dir).unwrap();

        let data = json!({"test": "data"});
        let snapshot = spiral.create_snapshot(&data, "MEF_SEED_42", None).unwrap();

        assert!(!snapshot.id.is_empty());
        assert_eq!(snapshot.seed, "MEF_SEED_42");
        assert_eq!(snapshot.coordinates.len(), 5);
    }

    #[test]
    fn test_determinism() {
        let config = SpiralConfig::default();
        let temp_dir = std::env::temp_dir().join("test_spiral_determinism");
        let spiral = SpiralSnapshot::new(config, &temp_dir).unwrap();

        let data = json!({"test": "data"});
        let snapshot1 = spiral
            .create_snapshot(&data, "MEF_SEED_42", Some(1.0))
            .unwrap();

        // Reset sequence for second snapshot
        spiral.sequence.set(0);
        let snapshot2 = spiral
            .create_snapshot(&data, "MEF_SEED_42", Some(1.0))
            .unwrap();

        assert_eq!(snapshot1.id, snapshot2.id);
        assert_eq!(snapshot1.coordinates, snapshot2.coordinates);
    }
}
