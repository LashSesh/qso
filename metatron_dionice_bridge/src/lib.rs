//! # Metatron-Dionice Bridge
//!
//! This crate bridges Q⊗DASH's Seraphic Calibration Shell (SCS) with the dioniceOS
//! backend, enabling geometric-cognitive calibration through 4D-5D morphodynamics.
//!
//! ## Architecture
//!
//! ```text
//! SCS Calibration State (ψ, ρ, ω)
//!          ↓
//!   DioniceKernel::ingest_state()
//!          ↓
//! 4D-Trichter Coupling Tick
//!   (Funnel + Hyperbion + HDAG)
//!          ↓
//!   DioniceKernel::step()
//!          ↓
//! Calibration Suggestion
//! ```
//!
//! ## Example
//!
//! ```rust
//! use metatron_dionice_bridge::{DioniceKernel, QDashCalibrationState};
//! use std::collections::HashMap;
//!
//! let mut kernel = DioniceKernel::new();
//! let state = QDashCalibrationState {
//!     psi: 0.85,
//!     rho: 0.90,
//!     omega: 0.75,
//!     algorithm: "VQE".to_string(),
//!     extra_params: HashMap::new(),
//! };
//!
//! kernel.ingest_state(state).unwrap();
//! let suggestion = kernel.step().unwrap();
//!
//! println!("Suggested updates: {}", suggestion.notes);
//! ```

#[cfg(feature = "python")]
pub mod python;

use anyhow::{Context, Result};
use apollyon_mef_bridge::trichter::{
    coupling_tick, FunnelGraph, HDAGField, Hyperbion, Policy, State4D, TickResult,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

/// Q⊗DASH calibration state mapped to dioniceOS coordinate space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QDashCalibrationState {
    /// Quality metric (ψ) - maps to 4D semantic weight
    pub psi: f64,
    /// Stability metric (ρ) - influences spatial x-coordinate
    pub rho: f64,
    /// Efficiency metric (ω) - maps to 5D temporal phase
    pub omega: f64,
    /// Current algorithm family (e.g., "VQE", "QAOA", "VQC")
    pub algorithm: String,
    /// Additional configuration parameters
    #[serde(default)]
    pub extra_params: HashMap<String, f64>,
}

/// Calibration suggestion produced by dioniceOS backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QDashCalibrationSuggestion {
    /// Suggested configuration updates as JSON
    pub new_config: serde_json::Value,
    /// Human-readable notes about the suggestion
    pub notes: String,
    /// Resonance quality score (0.0 - 1.0)
    pub resonance_score: f64,
    /// Whether a regime change is recommended
    pub regime_change_suggested: bool,
}

/// Main kernel interfacing Q⊗DASH with dioniceOS
pub struct DioniceKernel {
    /// 4D-Trichter funnel graph
    funnel: FunnelGraph,
    /// HDAG resonance field
    hdag: HDAGField,
    /// Hyperbion morphodynamic coupling layer
    hyperbion: Hyperbion,
    /// Current policy (Explore/Exploit/Homeostasis)
    policy: Policy,
    /// Time step counter
    time_step: f64,
    /// History of states for analysis
    state_history: Vec<State4D>,
}

impl DioniceKernel {
    /// Create a new DioniceKernel with default Explore policy
    pub fn new() -> Self {
        Self::with_policy(Policy::Explore)
    }

    /// Create a new DioniceKernel with specified policy
    pub fn with_policy(policy: Policy) -> Self {
        Self {
            funnel: FunnelGraph::new(),
            hdag: HDAGField::new(),
            hyperbion: Hyperbion::new(),
            policy,
            time_step: 0.0,
            state_history: Vec::new(),
        }
    }

    /// Ingest Q⊗DASH calibration state into the dioniceOS backend
    pub fn ingest_state(&mut self, state: QDashCalibrationState) -> Result<()> {
        // Map Q⊗DASH state to 4D state space
        // x: derived from rho (stability)
        // y: derived from algorithm family
        // z: combined metric
        // ψ: quality metric (psi)
        let state_4d = Self::map_to_4d(&state);

        self.state_history.push(state_4d);

        Ok(())
    }

    /// Execute one coupling tick and produce calibration suggestion
    pub fn step(&mut self) -> Result<QDashCalibrationSuggestion> {
        if self.state_history.is_empty() {
            anyhow::bail!("No state ingested. Call ingest_state() first.");
        }

        // Get current states (use last N states or just last one)
        let current_states = vec![*self.state_history.last().unwrap()];

        // Execute coupling tick
        let policy_params = self.policy.params();
        let tick_result = coupling_tick(
            &current_states,
            self.time_step,
            &policy_params,
            &self.hyperbion,
            &mut self.hdag,
            &mut self.funnel,
            false, // Don't compute cryptographic proofs for performance
        );

        self.time_step += 1.0;

        // Analyze the evolution to generate suggestions (borrow before extracting next_states)
        let suggestion =
            self.analyze_evolution(&current_states, &tick_result.states_4d_next, &tick_result)?;

        // Extract next states after analysis
        let next_states = tick_result.states_4d_next;

        // Update history
        if let Some(next_state) = next_states.first() {
            self.state_history.push(*next_state);
        }

        Ok(suggestion)
    }

    /// Map Q⊗DASH calibration state to 4D coordinate space
    fn map_to_4d(state: &QDashCalibrationState) -> State4D {
        // Coordinate mapping strategy:
        // x ← rho (stability) - centered around 0
        // y ← algorithm family encoding (e.g., VQE=1.0, QAOA=2.0, VQC=3.0)
        // z ← omega (efficiency)
        // ψ ← psi (quality)

        let x = (state.rho - 0.5) * 2.0; // Map [0,1] to [-1,1]
        let y = Self::encode_algorithm(&state.algorithm);
        let z = (state.omega - 0.5) * 2.0;
        let psi = state.psi;

        State4D::new(x, y, z, psi)
    }

    /// Encode algorithm family as numeric coordinate
    fn encode_algorithm(algo: &str) -> f64 {
        match algo.to_uppercase().as_str() {
            "VQE" => 1.0,
            "QAOA" => 2.0,
            "VQC" => 3.0,
            _ => 0.0,
        }
    }

    /// Decode algorithm family from numeric coordinate
    fn decode_algorithm(y: f64) -> String {
        let rounded = y.round();
        match rounded as i32 {
            1 => "VQE".to_string(),
            2 => "QAOA".to_string(),
            3 => "VQC".to_string(),
            _ => "VQE".to_string(), // Default fallback
        }
    }

    /// Analyze state evolution and generate calibration suggestion
    fn analyze_evolution(
        &self,
        current: &[State4D],
        next: &[State4D],
        tick_result: &TickResult,
    ) -> Result<QDashCalibrationSuggestion> {
        let curr = current.first().context("No current state")?;
        let nxt = next.first().context("No next state")?;

        // Compute deltas
        let delta_x = nxt.x - curr.x;
        let delta_y = nxt.y - curr.y;
        let delta_z = nxt.z - curr.z;
        let delta_psi = nxt.psi - curr.psi;

        // Determine if regime change is suggested (significant y-coordinate change)
        let regime_change = delta_y.abs() > 0.5;

        // Build configuration suggestion
        let mut config_updates = serde_json::Map::new();
        let mut notes = Vec::new();

        // Translate 4D changes to configuration hints

        // Stability (x-coordinate / rho)
        let new_rho = (nxt.x / 2.0) + 0.5; // Map [-1,1] back to [0,1]
        if delta_x.abs() > 0.1 {
            config_updates.insert("suggested_stability_target".to_string(), json!(new_rho));
            if delta_x > 0.0 {
                notes.push(
                    "Increase stability: Consider more random starts or larger ensemble"
                        .to_string(),
                );
            } else {
                notes.push(
                    "Decrease emphasis on stability: Optimize for quality/efficiency".to_string(),
                );
            }
        }

        // Quality (ψ-coordinate / psi)
        if delta_psi > 0.0 {
            notes.push(format!(
                "Quality improvement detected: Δψ = {:.4}",
                delta_psi
            ));
            config_updates.insert("quality_direction".to_string(), json!("improving"));
        } else if delta_psi < -0.05 {
            notes.push("Quality degradation: Consider reverting recent changes".to_string());
            config_updates.insert("quality_direction".to_string(), json!("degrading"));
        }

        // Efficiency (z-coordinate / omega)
        let new_omega = (nxt.z / 2.0) + 0.5;
        if delta_z > 0.1 {
            notes.push("Efficiency gain: Consider reducing ansatz depth or iterations".to_string());
            config_updates.insert("suggested_efficiency_target".to_string(), json!(new_omega));
        } else if delta_z < -0.1 {
            notes.push("Efficiency loss: May need to increase computational budget".to_string());
        }

        // Algorithm family (y-coordinate)
        if regime_change {
            let new_algo = Self::decode_algorithm(nxt.y);
            notes.push(format!(
                "Regime change suggested: Consider switching to {}",
                new_algo
            ));
            config_updates.insert("suggested_algorithm".to_string(), json!(new_algo));
        }

        // Funnel dynamics insights
        if tick_result.nodes_created > 0 {
            notes.push(format!(
                "Exploration: {} new patterns discovered",
                tick_result.nodes_created
            ));
        }
        if tick_result.nodes_merged > 0 {
            notes.push(format!(
                "Consolidation: {} patterns merged",
                tick_result.nodes_merged
            ));
        }

        // Compute resonance score from HDAG field
        let resonance_score = self.compute_resonance_score(nxt);

        let suggestion = QDashCalibrationSuggestion {
            new_config: serde_json::Value::Object(config_updates),
            notes: notes.join("; "),
            resonance_score,
            regime_change_suggested: regime_change,
        };

        Ok(suggestion)
    }

    /// Compute resonance quality score from HDAG field
    fn compute_resonance_score(&self, state: &State4D) -> f64 {
        // Use ψ (quality) as primary component, bounded to [0, 1]
        state.psi.clamp(0.0, 1.0)
    }

    /// Get current funnel density
    pub fn funnel_density(&self) -> f64 {
        self.funnel.density()
    }

    /// Get number of nodes in funnel
    pub fn funnel_node_count(&self) -> usize {
        self.funnel.node_count()
    }

    /// Switch to Exploit policy for consolidation
    pub fn switch_to_exploit(&mut self) {
        self.policy = Policy::Exploit;
    }

    /// Switch to Explore policy for discovery
    pub fn switch_to_explore(&mut self) {
        self.policy = Policy::Explore;
    }

    /// Switch to Homeostasis policy for stable operation
    pub fn switch_to_homeostasis(&mut self) {
        self.policy = Policy::Homeostasis;
    }
}

impl Default for DioniceKernel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_initialization() {
        let kernel = DioniceKernel::new();
        assert_eq!(kernel.time_step, 0.0);
        assert_eq!(kernel.state_history.len(), 0);
    }

    #[test]
    fn test_state_ingestion() {
        let mut kernel = DioniceKernel::new();
        let state = QDashCalibrationState {
            psi: 0.85,
            rho: 0.90,
            omega: 0.75,
            algorithm: "VQE".to_string(),
            extra_params: HashMap::new(),
        };

        kernel.ingest_state(state).unwrap();
        assert_eq!(kernel.state_history.len(), 1);
    }

    #[test]
    fn test_step_without_state_fails() {
        let mut kernel = DioniceKernel::new();
        let result = kernel.step();
        assert!(result.is_err());
    }

    #[test]
    fn test_full_calibration_cycle() {
        let mut kernel = DioniceKernel::new();

        // Ingest initial state
        let state = QDashCalibrationState {
            psi: 0.85,
            rho: 0.90,
            omega: 0.75,
            algorithm: "VQE".to_string(),
            extra_params: HashMap::new(),
        };
        kernel.ingest_state(state).unwrap();

        // Execute calibration step
        let suggestion = kernel.step().unwrap();

        // Verify suggestion structure
        assert!(suggestion.resonance_score >= 0.0 && suggestion.resonance_score <= 1.0);
        assert!(!suggestion.notes.is_empty());
    }

    #[test]
    fn test_algorithm_encoding_decoding() {
        assert_eq!(DioniceKernel::encode_algorithm("VQE"), 1.0);
        assert_eq!(DioniceKernel::encode_algorithm("QAOA"), 2.0);
        assert_eq!(DioniceKernel::encode_algorithm("VQC"), 3.0);

        assert_eq!(DioniceKernel::decode_algorithm(1.0), "VQE");
        assert_eq!(DioniceKernel::decode_algorithm(2.0), "QAOA");
        assert_eq!(DioniceKernel::decode_algorithm(3.0), "VQC");
    }

    #[test]
    fn test_policy_switching() {
        let mut kernel = DioniceKernel::new();

        kernel.switch_to_exploit();
        kernel.switch_to_explore();
        kernel.switch_to_homeostasis();

        // Just verify no panics
    }

    #[test]
    fn test_multiple_calibration_steps() {
        let mut kernel = DioniceKernel::new();

        let state = QDashCalibrationState {
            psi: 0.85,
            rho: 0.90,
            omega: 0.75,
            algorithm: "VQE".to_string(),
            extra_params: HashMap::new(),
        };
        kernel.ingest_state(state).unwrap();

        // Run multiple steps
        for i in 0..5 {
            let suggestion = kernel.step().unwrap();
            println!("Step {}: {}", i, suggestion.notes);

            // Ingest evolved state for next iteration
            let new_state = QDashCalibrationState {
                psi: (0.85 + i as f64 * 0.01).min(1.0),
                rho: 0.90,
                omega: 0.75,
                algorithm: "VQE".to_string(),
                extra_params: HashMap::new(),
            };
            kernel.ingest_state(new_state).unwrap();
        }

        assert!(kernel.state_history.len() > 5);
    }
}
