/*!
 * Merkaba Gate Module
 *
 * Merkaba-Gate implementation as Mandorla layer between Solve-Coagula and TIC.
 * Integrates with Metatron Cube for topological routing and resonance calculations.
 *
 * This gate serves as a deterministic filter ensuring only stable, coherent states
 * proceed to TIC crystallization and eventual ledger commitment.
 */

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

use crate::cube::MetatronCube;
use crate::mandorla::MandorlaField;
use crate::qdash_agent::QDASHAgent;
use crate::qlogic::QLogicEngine;
use crate::resonance_tensor::ResonanceTensorField;
use crate::spiral_memory::SpiralMemory;

/// TIC candidate structure for gate evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TICCandidate {
    /// TIC identifier
    pub tic_id: String,
    /// Fixpoint coordinates
    pub fixpoint: Vec<f64>,
    /// Proof of resonance status
    pub por_status: String,
    /// Operator sequence applied
    pub operator_sequence: Vec<String>,
    /// Timestamp
    pub timestamp: f64,
    /// Optional dual fixpoint for MCI calculation
    pub dual_fixpoint: Option<Vec<f64>>,
}

/// Gate checks structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateChecks {
    /// Proof of Resonance status
    pub por: String,
    /// Path invariance deviation
    pub delta_pi: f64,
    /// Coherence measure
    pub phi: f64,
    /// Lyapunov stability change
    pub delta_v: f64,
    /// Optional Mirror Consistency Index
    pub mci: Option<f64>,
}

/// Gate decision structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateDecision {
    /// Commit decision
    pub commit: bool,
    /// Reason for decision
    pub reason: String,
}

/// Merkaba decision parameters for thresholds
#[derive(Debug, Clone)]
pub struct MerkabaDeCisionParams {
    /// Path invariance threshold
    pub eps: f64,
    /// Coherence threshold
    pub phi_star: f64,
    /// MCI threshold (if MCI provided)
    pub eta: Option<f64>,
}

/// Gate event artifact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateEvent {
    /// Gate event identifier
    pub gate_id: String,
    /// Snapshot identifier
    pub snapshot_id: String,
    /// TIC candidate identifier
    pub tic_candidate_id: String,
    /// Gate checks
    pub checks: GateChecks,
    /// Gate decision
    pub decision: GateDecision,
    /// Timestamp
    pub timestamp: String,
}

/// Merkaba Gate implementation utilizing Metatron Cube's topological routing
///
/// The gate validates states through multiple checks:
/// - Proof of Resonance (PoR)
/// - Path Invariance (ΔPI)
/// - Coherence measure (Φ)
/// - Lyapunov stability (ΔV)
/// - Optional Mirror Consistency Index (MCI) for dual-consensus
#[derive(Debug, Clone)]
pub struct MerkabaGate {
    /// Path invariance tolerance
    pub epsilon: f64,
    /// Minimum coherence threshold
    pub phi_star: f64,
    /// MCI threshold for dual-consensus
    pub eta: f64,
    /// Audit log path
    pub audit_path: PathBuf,

    // Components
    /// Metatron Cube
    pub metatron_cube: MetatronCube,
    /// Mandorla resonance field
    pub mandorla: MandorlaField,
    /// QLogic engine
    pub qlogic: QLogicEngine,
    /// Resonance tensor field
    pub resonance_field: ResonanceTensorField,
    /// Spiral memory
    pub spiral_memory: SpiralMemory,
    /// QDASH agent
    pub qdash: QDASHAgent,

    // State
    /// Historical states for Lyapunov calculation
    pub state_history: Vec<Array1<f64>>,
    /// Lyapunov window size
    pub lyapunov_window: usize,
}

impl MerkabaGate {
    // Default threshold constants
    pub const DEFAULT_EPSILON: f64 = 1e-6;
    pub const DEFAULT_PHI_STAR: f64 = 0.6;
    pub const DEFAULT_ETA: f64 = 0.85;
    pub const DEFAULT_LYAPUNOV_WINDOW: usize = 10;

    /// Create a new Merkaba Gate with default parameters
    ///
    /// # Arguments
    ///
    /// * `audit_path` - Path for audit logging
    pub fn new(audit_path: PathBuf) -> Self {
        Self::with_params(
            Self::DEFAULT_EPSILON,
            Self::DEFAULT_PHI_STAR,
            Self::DEFAULT_ETA,
            13,
            audit_path,
        )
    }

    /// Create a new Merkaba Gate with custom parameters
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Path invariance tolerance
    /// * `phi_star` - Minimum coherence threshold
    /// * `eta` - MCI threshold for dual-consensus
    /// * `metatron_nodes` - Number of nodes in Metatron Cube (default 13)
    /// * `audit_path` - Path for audit logging
    pub fn with_params(
        epsilon: f64,
        phi_star: f64,
        eta: f64,
        metatron_nodes: usize,
        audit_path: PathBuf,
    ) -> Self {
        // Create audit directory if needed
        if let Some(parent) = audit_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        Self {
            epsilon,
            phi_star,
            eta,
            audit_path,
            metatron_cube: MetatronCube::default(),
            mandorla: MandorlaField::new(0.985, 0.5, 0.5),
            qlogic: QLogicEngine::new(metatron_nodes),
            resonance_field: ResonanceTensorField::new((3, 3, 3), 1.0, 2.0, 0.0, 1e-3),
            spiral_memory: SpiralMemory::new(0.07),
            qdash: QDASHAgent::new(4, 0.5, 0.5),
            state_history: Vec::new(),
            lyapunov_window: Self::DEFAULT_LYAPUNOV_WINDOW,
        }
    }

    /// Compute coherence measure Φ using Metatron Cube's resonance calculations
    ///
    /// # Arguments
    ///
    /// * `tic_candidate` - TIC candidate with fixpoint and invariants
    ///
    /// # Returns
    ///
    /// Coherence measure Φ ∈ [0, 1]
    pub fn compute_phi(&mut self, tic_candidate: &TICCandidate) -> f64 {
        // Extract and pad fixpoint to 13 dimensions
        let mut fixpoint = Array1::zeros(13);
        for (i, &val) in tic_candidate.fixpoint.iter().enumerate() {
            if i >= 13 {
                break;
            }
            fixpoint[i] = val;
        }

        // Clear and populate Mandorla field
        self.mandorla.clear_inputs();

        // Add fixpoint vectors at different phases
        let phases = [0.0, std::f64::consts::PI / 4.0, std::f64::consts::PI / 2.0];
        for phase in phases.iter() {
            let phase_shifted: Vec<f64> =
                fixpoint.iter().take(5).map(|&x| x * phase.cos()).collect();
            self.mandorla.add_input(Array1::from_vec(phase_shifted));
        }

        // Compute global resonance
        let coherence = self.mandorla.calc_resonance();

        // Apply spectral analysis via QLOGIC
        let qlogic_result = self.qlogic.step(tic_candidate.timestamp);
        let entropy = qlogic_result.entropy;

        // Combine coherence with spectral entropy
        let max_entropy = 13.0_f64.log2();
        let phi = coherence * (1.0 - entropy / max_entropy);

        phi.clamp(0.0, 1.0)
    }

    /// Compute path invariance deviation ΔPI using symmetry operators
    ///
    /// # Arguments
    ///
    /// * `tic_candidate` - TIC candidate with operator history
    ///
    /// # Returns
    ///
    /// Path invariance deviation ΔPI
    pub fn compute_delta_pi(&self, tic_candidate: &TICCandidate) -> f64 {
        if tic_candidate.operator_sequence.is_empty() {
            return 0.0;
        }

        // Extract and pad fixpoint
        let mut fixpoint = Array1::zeros(13);
        for (i, &val) in tic_candidate.fixpoint.iter().enumerate() {
            if i >= 13 {
                break;
            }
            fixpoint[i] = val;
        }

        // Apply operator sequence to original path
        let mut original_result = fixpoint.clone();
        for op in &tic_candidate.operator_sequence {
            original_result = Self::apply_operator(original_result, op);
        }

        // For simplified implementation, return small deviation
        // In full implementation, would test multiple permutation paths
        let deviation = original_result
            .iter()
            .zip(fixpoint.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        deviation * 0.1 // Scale factor for typical deviations
    }

    /// Compute Lyapunov exponent change ΔV to verify stability
    ///
    /// # Arguments
    ///
    /// * `tic_candidate` - TIC candidate
    ///
    /// # Returns
    ///
    /// Lyapunov change ΔV (negative = stable)
    pub fn compute_delta_v(&mut self, tic_candidate: &TICCandidate) -> f64 {
        let fixpoint = Array1::from_vec(tic_candidate.fixpoint.clone());

        // Add to history
        self.state_history.push(fixpoint.clone());
        if self.state_history.len() > self.lyapunov_window {
            self.state_history.remove(0);
        }

        if self.state_history.len() < 2 {
            return 0.0;
        }

        // Calculate Lyapunov exponent
        let mut lyapunov_sum = 0.0;
        for i in 1..self.state_history.len() {
            let prev_state = &self.state_history[i - 1];
            let curr_state = &self.state_history[i];

            let delta = curr_state - prev_state;
            let prev_norm = prev_state.dot(prev_state).sqrt();

            if prev_norm > 0.0 {
                let delta_norm = delta.dot(&delta).sqrt();
                let divergence = (delta_norm / prev_norm + 1e-10).ln();
                lyapunov_sum += divergence;
            }
        }

        let lyapunov_avg = lyapunov_sum / (self.state_history.len() - 1) as f64;

        // Calculate change if we have enough history
        if self.state_history.len() >= 3 {
            let mut prev_sum = 0.0;
            for i in 1..(self.state_history.len() - 1) {
                let prev_state = &self.state_history[i - 1];
                let curr_state = &self.state_history[i];
                let delta = curr_state - prev_state;
                let prev_norm = prev_state.dot(prev_state).sqrt();

                if prev_norm > 0.0 {
                    let delta_norm = delta.dot(&delta).sqrt();
                    let divergence = (delta_norm / prev_norm + 1e-10).ln();
                    prev_sum += divergence;
                }
            }
            let prev_avg = prev_sum / (self.state_history.len() - 2) as f64;
            lyapunov_avg - prev_avg
        } else {
            lyapunov_avg
        }
    }

    /// Compute Mirror Consistency Index using dual paths
    ///
    /// # Arguments
    ///
    /// * `tic_candidate` - TIC candidate
    ///
    /// # Returns
    ///
    /// MCI value if dual consensus available, None otherwise
    pub fn compute_mci(&self, tic_candidate: &TICCandidate) -> Option<f64> {
        tic_candidate.dual_fixpoint.as_ref()?;

        let primal_fixpoint = Array1::from_vec(tic_candidate.fixpoint.clone());
        let dual_fixpoint = Array1::from_vec(tic_candidate.dual_fixpoint.as_ref().unwrap().clone());

        // Normalize for comparison
        let primal_norm = primal_fixpoint.dot(&primal_fixpoint).sqrt();
        let dual_norm = dual_fixpoint.dot(&dual_fixpoint).sqrt();

        if primal_norm == 0.0 || dual_norm == 0.0 {
            return Some(0.0);
        }

        let primal_normalized = &primal_fixpoint / primal_norm;
        let dual_normalized = &dual_fixpoint / dual_norm;

        // Calculate consistency as cosine similarity
        let dot_product = primal_normalized.dot(&dual_normalized);

        // Map to [0, 1] range
        let mci = (dot_product + 1.0) / 2.0;

        Some(mci)
    }

    /// Make gate decision based on all criteria
    ///
    /// # Arguments
    ///
    /// * `por` - Proof of Resonance status
    /// * `delta_pi` - Path invariance deviation
    /// * `phi` - Coherence measure
    /// * `delta_v` - Lyapunov change
    /// * `mci` - Mirror Consistency Index (optional)
    /// * `params` - Decision parameters (thresholds)
    ///
    /// # Returns
    ///
    /// Tuple of (commit decision, reason string)
    pub fn merkaba_decide(
        &self,
        por: &str,
        delta_pi: f64,
        phi: f64,
        delta_v: f64,
        mci: Option<f64>,
        params: &MerkabaDeCisionParams,
    ) -> (bool, String) {
        let mut reasons = Vec::new();

        // Check PoR
        if por != "valid" {
            reasons.push(format!("por={}", por));
            return (false, format!("rejected: {}", reasons.join(", ")));
        }

        // Check path invariance
        if delta_pi > params.eps {
            reasons.push(format!(
                "path_invariance_exceeded: {:.6} > {}",
                delta_pi, params.eps
            ));
            return (false, format!("rejected: {}", reasons.join(", ")));
        }

        // Check coherence
        if phi < params.phi_star {
            reasons.push(format!(
                "coherence_insufficient: {:.3} < {}",
                phi, params.phi_star
            ));
            return (false, format!("rejected: {}", reasons.join(", ")));
        }

        // Check Lyapunov stability
        if delta_v >= 0.0 {
            reasons.push(format!("lyapunov_unstable: {:.6} >= 0", delta_v));
            return (false, format!("rejected: {}", reasons.join(", ")));
        }

        // Check MCI if available
        if let (Some(mci_val), Some(eta_val)) = (mci, params.eta) {
            if mci_val < eta_val {
                reasons.push(format!("mci_insufficient: {:.3} < {}", mci_val, eta_val));
                return (false, format!("rejected: {}", reasons.join(", ")));
            }
        }

        (true, "all_thresholds_passed".to_string())
    }

    /// Execute complete Merkaba-Gate evaluation
    ///
    /// # Arguments
    ///
    /// * `snapshot_id` - Snapshot identifier
    /// * `tic_candidate` - TIC candidate to evaluate
    /// * `epsilon_override` - Optional epsilon override
    /// * `phi_star_override` - Optional phi_star override
    /// * `eta_override` - Optional eta override
    ///
    /// # Returns
    ///
    /// Gate event artifact
    pub fn run_merkaba(
        &mut self,
        snapshot_id: String,
        tic_candidate: TICCandidate,
        epsilon_override: Option<f64>,
        phi_star_override: Option<f64>,
        eta_override: Option<f64>,
    ) -> GateEvent {
        // Use provided overrides or defaults
        let eps = epsilon_override.unwrap_or(self.epsilon);
        let phi_star = phi_star_override.unwrap_or(self.phi_star);
        let eta = eta_override.unwrap_or(self.eta);

        // Compute all gate checks
        let por = tic_candidate.por_status.clone();
        let delta_pi = self.compute_delta_pi(&tic_candidate);
        let phi = self.compute_phi(&tic_candidate);
        let delta_v = self.compute_delta_v(&tic_candidate);
        let mci = self.compute_mci(&tic_candidate);

        // Make decision
        let params = MerkabaDeCisionParams {
            eps,
            phi_star,
            eta: if mci.is_some() { Some(eta) } else { None },
        };
        let (commit, reason) = self.merkaba_decide(&por, delta_pi, phi, delta_v, mci, &params);

        // Create gate event
        let gate_event = GateEvent {
            gate_id: uuid::Uuid::new_v4().to_string(),
            snapshot_id,
            tic_candidate_id: tic_candidate.tic_id,
            checks: GateChecks {
                por,
                delta_pi,
                phi,
                delta_v,
                mci,
            },
            decision: GateDecision { commit, reason },
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        // Audit log
        let _ = self.audit_event(&gate_event);

        gate_event
    }

    /// Apply MEF operator to state vector
    fn apply_operator(state: Array1<f64>, operator: &str) -> Array1<f64> {
        match operator {
            "DK" => {
                // DoubleKick: apply orthogonal impulses
                // Simplified - just add small perturbations
                state.mapv(|x| x + 0.05 * x.sin() - 0.03 * x.cos())
            }
            "SW" => {
                // Sweep: apply threshold gate
                let mean_val = state.mean().unwrap_or(0.0);
                let tau = 0.5;
                let beta = 0.1;
                let gate = 1.0 / (1.0 + (-(mean_val - tau) / beta).exp());
                state * gate
            }
            "PI" => {
                // Path Invariance: sort to canonical form
                let mut sorted = state.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                Array1::from_vec(sorted)
            }
            "WT" => {
                // Weight Transfer: redistribute between scales
                let gamma = 0.1;
                let len = state.len();
                let mut new_state = state.clone();
                let third = len / 3;

                if third > 0 {
                    for i in 0..third {
                        new_state[i] *= 1.0 - gamma;
                    }
                    for i in third..len {
                        new_state[i] *= 1.0 + gamma * 0.5;
                    }
                }
                new_state
            }
            _ => state, // Unknown operator, return unchanged
        }
    }

    /// Write gate event to audit log
    fn audit_event(&self, event: &GateEvent) -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.audit_path)?;

        let json = serde_json::to_string(event)?;
        writeln!(file, "{}", json)?;
        Ok(())
    }

    /// Clear state history
    pub fn clear_history(&mut self) {
        self.state_history.clear();
    }
}

impl Default for MerkabaGate {
    fn default() -> Self {
        Self::new(PathBuf::from("logs/gate_merkaba.jsonl"))
    }
}

/// Validate gate event against schema
pub fn validate_gate_event(event: &GateEvent) -> bool {
    // Check POR status
    if event.checks.por != "valid" && event.checks.por != "invalid" {
        return false;
    }

    // Check commit is boolean (implicitly true in Rust)
    // Check numeric fields are finite
    if !event.checks.delta_pi.is_finite()
        || !event.checks.phi.is_finite()
        || !event.checks.delta_v.is_finite()
    {
        return false;
    }

    // Check MCI if present
    if let Some(mci) = event.checks.mci {
        if !mci.is_finite() {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_create_default() {
        let gate = MerkabaGate::default();
        assert_eq!(gate.epsilon, MerkabaGate::DEFAULT_EPSILON);
        assert_eq!(gate.phi_star, MerkabaGate::DEFAULT_PHI_STAR);
        assert_eq!(gate.eta, MerkabaGate::DEFAULT_ETA);
    }

    #[test]
    fn test_create_with_params() {
        let temp_dir = env::temp_dir();
        let audit_path = temp_dir.join("test_gate.jsonl");
        let gate = MerkabaGate::with_params(1e-5, 0.7, 0.9, 13, audit_path);

        assert_eq!(gate.epsilon, 1e-5);
        assert_eq!(gate.phi_star, 0.7);
        assert_eq!(gate.eta, 0.9);
    }

    #[test]
    fn test_compute_phi() {
        let temp_dir = env::temp_dir();
        let audit_path = temp_dir.join("test_phi.jsonl");
        let mut gate = MerkabaGate::new(audit_path);

        let candidate = TICCandidate {
            tic_id: "test_tic".to_string(),
            fixpoint: vec![0.5, 0.3, 0.7, 0.2, 0.9],
            por_status: "valid".to_string(),
            operator_sequence: vec![],
            timestamp: 0.0,
            dual_fixpoint: None,
        };

        let phi = gate.compute_phi(&candidate);
        assert!(phi >= 0.0 && phi <= 1.0);
    }

    #[test]
    fn test_compute_delta_pi_empty_sequence() {
        let temp_dir = env::temp_dir();
        let audit_path = temp_dir.join("test_delta_pi.jsonl");
        let gate = MerkabaGate::new(audit_path);

        let candidate = TICCandidate {
            tic_id: "test_tic".to_string(),
            fixpoint: vec![0.5, 0.3, 0.7],
            por_status: "valid".to_string(),
            operator_sequence: vec![],
            timestamp: 0.0,
            dual_fixpoint: None,
        };

        let delta_pi = gate.compute_delta_pi(&candidate);
        assert_eq!(delta_pi, 0.0);
    }

    #[test]
    fn test_compute_delta_v() {
        let temp_dir = env::temp_dir();
        let audit_path = temp_dir.join("test_delta_v.jsonl");
        let mut gate = MerkabaGate::new(audit_path);

        let candidate = TICCandidate {
            tic_id: "test_tic".to_string(),
            fixpoint: vec![0.5, 0.3, 0.7],
            por_status: "valid".to_string(),
            operator_sequence: vec![],
            timestamp: 0.0,
            dual_fixpoint: None,
        };

        // First call should return 0.0 (not enough history)
        let delta_v = gate.compute_delta_v(&candidate);
        assert_eq!(delta_v, 0.0);

        // Add more states
        gate.compute_delta_v(&candidate);
        let delta_v = gate.compute_delta_v(&candidate);
        assert!(delta_v.is_finite());
    }

    #[test]
    fn test_compute_mci_none() {
        let temp_dir = env::temp_dir();
        let audit_path = temp_dir.join("test_mci.jsonl");
        let gate = MerkabaGate::new(audit_path);

        let candidate = TICCandidate {
            tic_id: "test_tic".to_string(),
            fixpoint: vec![0.5, 0.3, 0.7],
            por_status: "valid".to_string(),
            operator_sequence: vec![],
            timestamp: 0.0,
            dual_fixpoint: None,
        };

        let mci = gate.compute_mci(&candidate);
        assert!(mci.is_none());
    }

    #[test]
    fn test_compute_mci_with_dual() {
        let temp_dir = env::temp_dir();
        let audit_path = temp_dir.join("test_mci_dual.jsonl");
        let gate = MerkabaGate::new(audit_path);

        let candidate = TICCandidate {
            tic_id: "test_tic".to_string(),
            fixpoint: vec![0.5, 0.3, 0.7],
            por_status: "valid".to_string(),
            operator_sequence: vec![],
            timestamp: 0.0,
            dual_fixpoint: Some(vec![0.5, 0.3, 0.7]),
        };

        let mci = gate.compute_mci(&candidate);
        assert!(mci.is_some());
        assert!(mci.unwrap() >= 0.0 && mci.unwrap() <= 1.0);
    }

    #[test]
    fn test_merkaba_decide_valid() {
        let temp_dir = env::temp_dir();
        let audit_path = temp_dir.join("test_decide.jsonl");
        let gate = MerkabaGate::new(audit_path);

        let params = MerkabaDeCisionParams {
            eps: 1e-6,
            phi_star: 0.6,
            eta: None,
        };
        let (commit, reason) = gate.merkaba_decide("valid", 1e-7, 0.8, -0.1, None, &params);

        assert!(commit);
        assert_eq!(reason, "all_thresholds_passed");
    }

    #[test]
    fn test_merkaba_decide_invalid_por() {
        let temp_dir = env::temp_dir();
        let audit_path = temp_dir.join("test_decide_por.jsonl");
        let gate = MerkabaGate::new(audit_path);

        let params = MerkabaDeCisionParams {
            eps: 1e-6,
            phi_star: 0.6,
            eta: None,
        };
        let (commit, _reason) = gate.merkaba_decide("invalid", 1e-7, 0.8, -0.1, None, &params);

        assert!(!commit);
    }

    #[test]
    fn test_merkaba_decide_path_invariance_exceeded() {
        let temp_dir = env::temp_dir();
        let audit_path = temp_dir.join("test_decide_pi.jsonl");
        let gate = MerkabaGate::new(audit_path);

        let params = MerkabaDeCisionParams {
            eps: 1e-6,
            phi_star: 0.6,
            eta: None,
        };
        let (commit, _reason) = gate.merkaba_decide("valid", 1e-5, 0.8, -0.1, None, &params);

        assert!(!commit);
    }

    #[test]
    fn test_run_merkaba() {
        let temp_dir = env::temp_dir();
        let audit_path = temp_dir.join("test_run_merkaba.jsonl");
        let mut gate = MerkabaGate::new(audit_path);

        let candidate = TICCandidate {
            tic_id: "test_tic".to_string(),
            fixpoint: vec![0.5, 0.3, 0.7, 0.2, 0.9],
            por_status: "valid".to_string(),
            operator_sequence: vec!["DK".to_string(), "SW".to_string()],
            timestamp: 0.0,
            dual_fixpoint: None,
        };

        let event = gate.run_merkaba("snapshot_123".to_string(), candidate, None, None, None);

        assert!(!event.gate_id.is_empty());
        assert_eq!(event.snapshot_id, "snapshot_123");
        assert_eq!(event.tic_candidate_id, "test_tic");
    }

    #[test]
    fn test_apply_operator_dk() {
        let state = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = MerkabaGate::apply_operator(state.clone(), "DK");
        assert_eq!(result.len(), 3);
        assert_ne!(result, state); // Should be modified
    }

    #[test]
    fn test_apply_operator_unknown() {
        let state = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = MerkabaGate::apply_operator(state.clone(), "UNKNOWN");
        assert_eq!(result, state); // Should be unchanged
    }

    #[test]
    fn test_validate_gate_event_valid() {
        let event = GateEvent {
            gate_id: "gate_123".to_string(),
            snapshot_id: "snap_123".to_string(),
            tic_candidate_id: "tic_123".to_string(),
            checks: GateChecks {
                por: "valid".to_string(),
                delta_pi: 1e-7,
                phi: 0.8,
                delta_v: -0.1,
                mci: None,
            },
            decision: GateDecision {
                commit: true,
                reason: "all_thresholds_passed".to_string(),
            },
            timestamp: "2025-01-01T00:00:00Z".to_string(),
        };

        assert!(validate_gate_event(&event));
    }

    #[test]
    fn test_validate_gate_event_invalid_por() {
        let event = GateEvent {
            gate_id: "gate_123".to_string(),
            snapshot_id: "snap_123".to_string(),
            tic_candidate_id: "tic_123".to_string(),
            checks: GateChecks {
                por: "unknown".to_string(),
                delta_pi: 1e-7,
                phi: 0.8,
                delta_v: -0.1,
                mci: None,
            },
            decision: GateDecision {
                commit: true,
                reason: "all_thresholds_passed".to_string(),
            },
            timestamp: "2025-01-01T00:00:00Z".to_string(),
        };

        assert!(!validate_gate_event(&event));
    }

    #[test]
    fn test_clear_history() {
        let temp_dir = env::temp_dir();
        let audit_path = temp_dir.join("test_clear.jsonl");
        let mut gate = MerkabaGate::new(audit_path);

        let candidate = TICCandidate {
            tic_id: "test_tic".to_string(),
            fixpoint: vec![0.5, 0.3, 0.7],
            por_status: "valid".to_string(),
            operator_sequence: vec![],
            timestamp: 0.0,
            dual_fixpoint: None,
        };

        gate.compute_delta_v(&candidate);
        gate.compute_delta_v(&candidate);

        assert!(gate.state_history.len() > 0);

        gate.clear_history();
        assert_eq!(gate.state_history.len(), 0);
    }
}
