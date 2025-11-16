//! Proof-of-Resonance (PoR) validation system.
//! Mathematical validation of Spiral snapshot stability.
//!
//! FFT(s) → ŝ; r' = g(ŝ) ∈ [0,1]
//! Acceptance: |r' - r_snapshot| ≤ δ ∧ λ_gap ≥ λ_min

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Band energy distribution across frequency bands
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandEnergy {
    pub low: f64,
    pub mid: f64,
    pub high: f64,
}

/// Resonance validation data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceData {
    pub fft_resonance: f64,
    pub claimed_resonance: f64,
    pub deviation: f64,
    pub spectral_gap: f64,
    pub band_energy: BandEnergy,
    pub resonance_valid: bool,
    pub gap_valid: bool,
    pub status: String,
}

/// Stability validation data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityData {
    pub computed: f64,
    pub claimed: f64,
    pub valid: bool,
}

/// Validation report for a snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub snapshot_id: String,
    pub resonance: ResonanceData,
    pub stability: StabilityData,
    pub por_status: String,
    pub overall_valid: bool,
    pub timestamp: String,
}

/// Batch validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchValidationResults {
    pub total: usize,
    pub valid: usize,
    pub invalid: usize,
    pub details: Vec<ValidationReport>,
    pub validation_rate: f64,
}

/// Proof-of-Resonance validator for Spiral snapshots
#[derive(Debug, Clone)]
pub struct ProofOfResonance {
    por_delta: f64,
    lambda_min: f64,
    #[allow(dead_code)]
    stability_threshold: f64,
}

impl ProofOfResonance {
    /// Create new PoR validator with default parameters
    pub fn new() -> Self {
        Self {
            por_delta: 0.02,
            lambda_min: 0.1,
            stability_threshold: 0.9,
        }
    }

    /// Create PoR validator with custom parameters
    pub fn with_config(por_delta: f64, lambda_min: f64, stability_threshold: f64) -> Self {
        Self {
            por_delta,
            lambda_min,
            stability_threshold,
        }
    }

    /// Compute resonance using FFT analysis
    ///
    /// # Arguments
    /// * `coordinates` - 5D spiral coordinates
    ///
    /// # Returns
    /// Tuple of (resonance value, FFT magnitude spectrum)
    pub fn compute_fft_resonance(&self, coordinates: &[f64]) -> (f64, Vec<f64>) {
        // Convert to complex numbers for FFT
        let mut buffer: Vec<num_complex::Complex<f64>> = coordinates
            .iter()
            .map(|&x| num_complex::Complex::new(x, 0.0))
            .collect();

        // Perform FFT using rustfft
        let mut planner = rustfft::FftPlanner::new();
        let fft = planner.plan_fft_forward(buffer.len());
        fft.process(&mut buffer);

        // Compute magnitudes
        let fft_magnitude: Vec<f64> = buffer.iter().map(|c| c.norm()).collect();

        // Normalize spectrum
        let max_mag = fft_magnitude
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let fft_normalized: Vec<f64> = if max_mag > 0.0 {
            fft_magnitude.iter().map(|&m| m / max_mag).collect()
        } else {
            fft_magnitude.clone()
        };

        // Compute resonance metric: r' = mean(magnitude) / (1 + std(magnitude))
        let mean_mag = fft_normalized.iter().sum::<f64>() / fft_normalized.len() as f64;
        let variance = fft_normalized
            .iter()
            .map(|&x| (x - mean_mag).powi(2))
            .sum::<f64>()
            / fft_normalized.len() as f64;
        let std_mag = variance.sqrt();

        let resonance = mean_mag / (1.0 + std_mag);

        (resonance, fft_normalized)
    }

    /// Compute spectral gap using local Laplacian
    ///
    /// # Arguments
    /// * `coordinates` - 5D coordinates
    ///
    /// # Returns
    /// Spectral gap value
    pub fn compute_spectral_gap(&self, coordinates: &[f64]) -> f64 {
        let n = coordinates.len();

        if n < 3 {
            return 0.5; // Too small for meaningful Laplacian
        }

        // Construct local Laplacian matrix (circulant structure for 5D topology)
        let mut laplacian = vec![vec![0.0; n]; n];

        for (i, row) in laplacian.iter_mut().enumerate() {
            row[i] = 2.0; // Diagonal
            let next = (i + 1) % n;
            let prev = (i + n - 1) % n;
            row[next] = -1.0; // Upper neighbor
            row[prev] = -1.0; // Lower neighbor
        }

        // Compute eigenvalues using nalgebra
        let matrix = nalgebra::DMatrix::from_fn(n, n, |i, j| laplacian[i][j]);
        let eigenvalues = matrix.symmetric_eigenvalues();

        // Convert to Vec and sort
        let mut eigs: Vec<f64> = eigenvalues.iter().cloned().collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Spectral gap is difference between first two eigenvalues
        if eigs.len() > 1 {
            (eigs[1] - eigs[0]).abs()
        } else {
            0.5
        }
    }

    /// Compute energy distribution across frequency bands
    ///
    /// # Arguments
    /// * `spectrum` - FFT magnitude spectrum
    ///
    /// # Returns
    /// Band energy distribution
    pub fn compute_band_energy(&self, spectrum: &[f64]) -> BandEnergy {
        let n = spectrum.len();

        // Define bands
        let bands = [("low", 0.0, 0.3), ("mid", 0.3, 0.7), ("high", 0.7, 1.0)];

        let mut energies = HashMap::new();

        for (name, low, high) in &bands {
            let low_idx = (low * n as f64) as usize;
            let high_idx = (high * n as f64) as usize;

            let band_spectrum = &spectrum[low_idx..high_idx.min(n)];
            let energy: f64 = band_spectrum.iter().map(|&v| v * v).sum();

            energies.insert(name.to_string(), energy);
        }

        // Normalize
        let total_energy: f64 = energies.values().sum();
        if total_energy > 0.0 {
            for v in energies.values_mut() {
                *v /= total_energy;
            }
        }

        BandEnergy {
            low: *energies.get("low").unwrap_or(&0.0),
            mid: *energies.get("mid").unwrap_or(&0.0),
            high: *energies.get("high").unwrap_or(&0.0),
        }
    }

    /// Validate resonance claim for coordinates
    ///
    /// # Arguments
    /// * `coordinates` - 5D coordinates
    /// * `claimed_resonance` - Claimed resonance value
    ///
    /// # Returns
    /// Tuple of (is_valid, validation_data)
    pub fn validate_resonance(
        &self,
        coordinates: &[f64],
        claimed_resonance: f64,
    ) -> (bool, ResonanceData) {
        // Compute FFT resonance
        let (fft_resonance, spectrum) = self.compute_fft_resonance(coordinates);

        // Compute spectral gap
        let spectral_gap = self.compute_spectral_gap(coordinates);

        // Compute band energies
        let band_energy = self.compute_band_energy(&spectrum);

        // Check resonance deviation
        let resonance_deviation = (fft_resonance - claimed_resonance).abs();
        let resonance_valid = resonance_deviation <= self.por_delta;

        // Check spectral gap
        let gap_valid = spectral_gap >= self.lambda_min;

        // Overall validation
        let is_valid = resonance_valid && gap_valid;

        let validation_data = ResonanceData {
            fft_resonance,
            claimed_resonance,
            deviation: resonance_deviation,
            spectral_gap,
            band_energy,
            resonance_valid,
            gap_valid,
            status: if is_valid {
                "valid".to_string()
            } else {
                "invalid".to_string()
            },
        };

        (is_valid, validation_data)
    }

    /// Compute stability metric for coordinates
    ///
    /// # Arguments
    /// * `coordinates` - Current coordinates
    /// * `history` - Optional history of previous coordinates
    ///
    /// # Returns
    /// Stability value in [0, 1]
    pub fn compute_stability_metric(
        &self,
        coordinates: &[f64],
        history: Option<&[Vec<f64>]>,
    ) -> f64 {
        // Base stability from norm
        let mean = coordinates.iter().sum::<f64>() / coordinates.len() as f64;
        let diff_norm: f64 = coordinates
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            .sqrt();
        let base_stability = 1.0 / (1.0 + diff_norm);

        let stability = if let Some(hist) = history {
            if hist.len() > 1 {
                // Compute variance across history for each dimension
                let dim = coordinates.len();
                let mut variances = vec![0.0; dim];

                for d in 0..dim {
                    let values: Vec<f64> = hist.iter().map(|h| h[d]).collect();
                    let mean_d = values.iter().sum::<f64>() / values.len() as f64;
                    let var = values.iter().map(|&v| (v - mean_d).powi(2)).sum::<f64>()
                        / values.len() as f64;
                    variances[d] = var;
                }

                let mean_variance = variances.iter().sum::<f64>() / variances.len() as f64;
                let history_stability = 1.0 / (1.0 + mean_variance);

                0.7 * base_stability + 0.3 * history_stability
            } else {
                base_stability
            }
        } else {
            base_stability
        };

        stability.clamp(0.0, 1.0)
    }

    /// Validate a Spiral snapshot
    ///
    /// # Arguments
    /// * `snapshot` - Snapshot data as JSON value
    ///
    /// # Returns
    /// Tuple of (is_valid, validation_report)
    pub fn validate_snapshot(
        &self,
        snapshot: &serde_json::Value,
    ) -> Result<(bool, ValidationReport), String> {
        let coordinates: Vec<f64> = snapshot["coordinates"]
            .as_array()
            .ok_or("Missing coordinates")?
            .iter()
            .map(|v| v.as_f64().ok_or("Invalid coordinate"))
            .collect::<Result<Vec<_>, _>>()?;

        let metrics = &snapshot["metrics"];
        let resonance = metrics["resonance"].as_f64().ok_or("Missing resonance")?;

        // Validate resonance
        let (resonance_valid, resonance_data) = self.validate_resonance(&coordinates, resonance);

        // Compute stability
        let stability = self.compute_stability_metric(&coordinates, None);
        let claimed_stability = metrics
            .get("stability")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let stability_valid = stability >= claimed_stability * 0.9;

        // Check PoR status
        let por_status = metrics
            .get("por")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let por_valid = por_status == "valid";

        // Overall validation
        let is_valid = resonance_valid && stability_valid && por_valid;

        let report = ValidationReport {
            snapshot_id: snapshot["id"].as_str().unwrap_or("unknown").to_string(),
            resonance: resonance_data,
            stability: StabilityData {
                computed: stability,
                claimed: claimed_stability,
                valid: stability_valid,
            },
            por_status,
            overall_valid: is_valid,
            timestamp: snapshot["timestamp"].as_str().unwrap_or("").to_string(),
        };

        Ok((is_valid, report))
    }

    /// Validate multiple snapshots in batch
    ///
    /// # Arguments
    /// * `snapshots` - List of snapshot data
    ///
    /// # Returns
    /// Batch validation results
    pub fn batch_validate(
        &self,
        snapshots: &[serde_json::Value],
    ) -> Result<BatchValidationResults, String> {
        let mut valid = 0;
        let mut invalid = 0;
        let mut details = Vec::new();

        for snapshot in snapshots {
            let (is_valid, report) = self.validate_snapshot(snapshot)?;

            if is_valid {
                valid += 1;
            } else {
                invalid += 1;
            }

            details.push(report);
        }

        let total = snapshots.len();
        let validation_rate = if total > 0 {
            valid as f64 / total as f64
        } else {
            0.0
        };

        Ok(BatchValidationResults {
            total,
            valid,
            invalid,
            details,
            validation_rate,
        })
    }

    /// Compute aggregate resonance across multiple snapshots
    ///
    /// # Arguments
    /// * `snapshots` - List of snapshots
    ///
    /// # Returns
    /// Network-wide resonance metric
    pub fn compute_network_resonance(&self, snapshots: &[serde_json::Value]) -> f64 {
        if snapshots.is_empty() {
            return 0.0;
        }

        let mut resonances = Vec::new();
        let mut weights = Vec::new();

        for snapshot in snapshots {
            if let Some(coords_arr) = snapshot["coordinates"].as_array() {
                let coords: Vec<f64> = coords_arr.iter().filter_map(|v| v.as_f64()).collect();

                if !coords.is_empty() {
                    let (resonance, _) = self.compute_fft_resonance(&coords);
                    resonances.push(resonance);

                    let stability = snapshot["metrics"]["stability"].as_f64().unwrap_or(0.5);
                    weights.push(stability);
                }
            }
        }

        if resonances.is_empty() {
            return 0.0;
        }

        // Weighted average
        let weight_sum: f64 = weights.iter().sum();
        if weight_sum > 0.0 {
            resonances
                .iter()
                .zip(weights.iter())
                .map(|(&r, &w)| r * w)
                .sum::<f64>()
                / weight_sum
        } else {
            resonances.iter().sum::<f64>() / resonances.len() as f64
        }
    }

    /// Generate cryptographic proof string for PoR
    ///
    /// # Arguments
    /// * `snapshot` - Snapshot data
    ///
    /// # Returns
    /// Proof string (hash)
    pub fn generate_por_proof(&self, snapshot: &serde_json::Value) -> Result<String, String> {
        let (is_valid, report) = self.validate_snapshot(snapshot)?;

        // Create proof data
        let proof_data = serde_json::json!({
            "snapshot_id": snapshot["id"],
            "coordinates": snapshot["coordinates"],
            "resonance": report.resonance.fft_resonance,
            "spectral_gap": report.resonance.spectral_gap,
            "stability": report.stability.computed,
            "valid": is_valid
        });

        // Generate deterministic proof string
        let proof_str = serde_json::to_string(&proof_data).map_err(|e| e.to_string())?;
        let mut hasher = Sha256::new();
        hasher.update(proof_str.as_bytes());
        let proof_hash = format!("{:x}", hasher.finalize());

        Ok(proof_hash)
    }
}

impl Default for ProofOfResonance {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_por_validator() {
        let por = ProofOfResonance::new();
        assert_eq!(por.por_delta, 0.02);
        assert_eq!(por.lambda_min, 0.1);
    }

    #[test]
    fn test_compute_fft_resonance() {
        let por = ProofOfResonance::new();
        let coords = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (resonance, spectrum) = por.compute_fft_resonance(&coords);

        assert!((0.0..=1.0).contains(&resonance));
        assert_eq!(spectrum.len(), coords.len());
    }

    #[test]
    fn test_compute_spectral_gap() {
        let por = ProofOfResonance::new();
        let coords = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gap = por.compute_spectral_gap(&coords);

        assert!(gap >= 0.0);
    }

    #[test]
    fn test_compute_band_energy() {
        let por = ProofOfResonance::new();
        let spectrum = vec![0.5, 0.3, 0.1, 0.2, 0.4];
        let band_energy = por.compute_band_energy(&spectrum);

        // Band energies should sum to approximately 1.0
        let total = band_energy.low + band_energy.mid + band_energy.high;
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_validate_resonance() {
        let por = ProofOfResonance::new();
        let coords = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let claimed = 0.5;

        let (_is_valid, data) = por.validate_resonance(&coords, claimed);

        assert_eq!(data.claimed_resonance, claimed);
        assert!(data.deviation >= 0.0);
        assert!(data.spectral_gap >= 0.0);
    }

    #[test]
    fn test_compute_stability_metric() {
        let por = ProofOfResonance::new();
        let coords = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stability = por.compute_stability_metric(&coords, None);

        assert!((0.0..=1.0).contains(&stability));
    }

    #[test]
    fn test_compute_stability_with_history() {
        let por = ProofOfResonance::new();
        let coords = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let history = vec![vec![0.9, 1.9, 2.9, 3.9, 4.9], vec![1.1, 2.1, 3.1, 4.1, 5.1]];
        let stability = por.compute_stability_metric(&coords, Some(&history));

        assert!((0.0..=1.0).contains(&stability));
    }

    #[test]
    fn test_validate_snapshot() {
        let por = ProofOfResonance::new();
        let snapshot = serde_json::json!({
            "id": "test-snapshot-123",
            "coordinates": [1.0, 2.0, 3.0, 4.0, 5.0],
            "metrics": {
                "resonance": 0.5,
                "stability": 0.8,
                "por": "valid"
            },
            "timestamp": "2025-01-01T00:00:00Z"
        });

        let result = por.validate_snapshot(&snapshot);
        assert!(result.is_ok());

        let (_, report) = result.unwrap();
        assert_eq!(report.snapshot_id, "test-snapshot-123");
    }

    #[test]
    fn test_batch_validate() {
        let por = ProofOfResonance::new();
        let snapshots = vec![
            serde_json::json!({
                "id": "snap-1",
                "coordinates": [1.0, 2.0, 3.0, 4.0, 5.0],
                "metrics": {"resonance": 0.5, "stability": 0.8, "por": "valid"},
                "timestamp": "2025-01-01T00:00:00Z"
            }),
            serde_json::json!({
                "id": "snap-2",
                "coordinates": [2.0, 3.0, 4.0, 5.0, 6.0],
                "metrics": {"resonance": 0.6, "stability": 0.7, "por": "valid"},
                "timestamp": "2025-01-01T00:01:00Z"
            }),
        ];

        let result = por.batch_validate(&snapshots);
        assert!(result.is_ok());

        let batch_results = result.unwrap();
        assert_eq!(batch_results.total, 2);
        assert_eq!(batch_results.details.len(), 2);
    }

    #[test]
    fn test_compute_network_resonance() {
        let por = ProofOfResonance::new();
        let snapshots = vec![
            serde_json::json!({
                "coordinates": [1.0, 2.0, 3.0, 4.0, 5.0],
                "metrics": {"stability": 0.8}
            }),
            serde_json::json!({
                "coordinates": [2.0, 3.0, 4.0, 5.0, 6.0],
                "metrics": {"stability": 0.7}
            }),
        ];

        let network_resonance = por.compute_network_resonance(&snapshots);
        assert!((0.0..=1.0).contains(&network_resonance));
    }

    #[test]
    fn test_generate_por_proof() {
        let por = ProofOfResonance::new();
        let snapshot = serde_json::json!({
            "id": "test-snapshot-123",
            "coordinates": [1.0, 2.0, 3.0, 4.0, 5.0],
            "metrics": {"resonance": 0.5, "stability": 0.8, "por": "valid"},
            "timestamp": "2025-01-01T00:00:00Z"
        });

        let result = por.generate_por_proof(&snapshot);
        assert!(result.is_ok());

        let proof = result.unwrap();
        assert_eq!(proof.len(), 64); // SHA256 hex string length
    }

    #[test]
    fn test_deterministic_proof() {
        let por = ProofOfResonance::new();
        let snapshot = serde_json::json!({
            "id": "test-snapshot-123",
            "coordinates": [1.0, 2.0, 3.0, 4.0, 5.0],
            "metrics": {"resonance": 0.5, "stability": 0.8, "por": "valid"},
            "timestamp": "2025-01-01T00:00:00Z"
        });

        let proof1 = por.generate_por_proof(&snapshot).unwrap();
        let proof2 = por.generate_por_proof(&snapshot).unwrap();

        assert_eq!(proof1, proof2); // Deterministic
    }
}
