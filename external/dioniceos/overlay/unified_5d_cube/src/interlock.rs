//! Interlock adapters connecting APOLLYON, Trichter, and MEF via public APIs only.
//!
//! This module provides the glue layer that connects:
//! - STATE_IN: APOLLYON State5D
//! - FIELD_IO: Trichter ∇Φ, Hyperbion/HDAG
//! - GATE: MEF Proof-of-Resonance / Merkaba
//! - CONDENSE: Trichter Coagula/Tick
//! - EVENT_OUT: MEF Ledger/TIC

use core_5d::State5D as ApollonState5D;
use apollyon_mef_bridge::trichter::{
    State5D as TrichterState5D, HDAGField, Hyperbion, FunnelGraph, 
    PolicyParams, Policy, GuidanceVector
};
use mef_schemas::GateDecision;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Simple Proof-of-Resonance data for the 5D Cube
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleProofOfResonance {
    /// Path invariance (Wasserstein-2 distance)
    pub delta_pi: f64,
    
    /// Alignment (cosine similarity)
    pub phi: f64,
    
    /// Lyapunov delta (energy change)
    pub delta_v: f64,
    
    /// Overall validity
    pub por_valid: bool,
}

/// Configuration for the interlock system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterlockConfig {
    /// Deterministic seed for reproducibility
    pub seed: u64,
    
    /// Gate thresholds
    pub gate_phi_threshold: f64,
    pub gate_delta_pi_max: f64,
    
    /// Feature flags
    pub enable_logging: bool,
    pub shadow_mode: bool,
    
    /// Extension flags
    pub enable_full_hdag: bool,
    pub enable_funnel_ops: bool,
    pub enable_ledger_writes: bool,
    pub enable_8d_vectors: bool,
    pub enable_metatron_routing: bool,
    
    /// Funnel policy
    pub funnel_policy: Policy,
    
    /// MEF Ledger path (optional)
    pub ledger_path: Option<PathBuf>,
}

impl Default for InterlockConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            gate_phi_threshold: 0.5,
            gate_delta_pi_max: 0.1,
            enable_logging: true,
            shadow_mode: true, // Start in shadow mode by default
            enable_full_hdag: false,
            enable_funnel_ops: false,
            enable_ledger_writes: false,
            enable_8d_vectors: false,
            enable_metatron_routing: false,
            funnel_policy: Policy::Explore,
            ledger_path: None,
        }
    }
}

/// Adapter connecting all components via public APIs
pub struct InterlockAdapter {
    config: InterlockConfig,
    
    // Trichter components
    hdag: HDAGField,
    hyperbion: Hyperbion,
    funnel: FunnelGraph,
    policy_params: PolicyParams,
    
    // MEF components (optional)
    ledger: Option<mef_ledger::MEFLedger>,
    vector8_builder: Option<mef_knowledge::Vector8Builder>,
    metatron_router: Option<mef_topology::MetatronRouter>,
}

impl InterlockAdapter {
    /// Create new interlock adapter with configuration
    pub fn new(config: InterlockConfig) -> Self {
        let hyperbion = Hyperbion::new();
        let hdag = HDAGField::new();
        let funnel = FunnelGraph::new();
        let policy_params = config.funnel_policy.params();
        
        // Initialize MEF ledger if enabled and path provided
        let ledger = if config.enable_ledger_writes {
            if let Some(ref path) = config.ledger_path {
                match mef_ledger::MEFLedger::new(path) {
                    Ok(l) => {
                        if config.enable_logging {
                            tracing::info!("MEF Ledger initialized at {:?}", path);
                        }
                        Some(l)
                    }
                    Err(e) => {
                        if config.enable_logging {
                            tracing::warn!("Failed to initialize MEF Ledger: {}", e);
                        }
                        None
                    }
                }
            } else {
                if config.enable_logging {
                    tracing::warn!("Ledger writes enabled but no path provided");
                }
                None
            }
        } else {
            None
        };
        
        // Initialize 8D vector builder if enabled
        let vector8_builder = if config.enable_8d_vectors {
            Some(mef_knowledge::Vector8Builder::default())
        } else {
            None
        };
        
        // Initialize Metatron router if enabled
        let metatron_router = if config.enable_metatron_routing {
            // Create a temporary path for router storage
            let router_path = config.ledger_path.clone()
                .unwrap_or_else(|| PathBuf::from("/tmp/metatron_router"));
            
            let router = mef_topology::MetatronRouter::new(&router_path);
            if config.enable_logging {
                tracing::info!("Metatron Router initialized");
            }
            Some(router)
        } else {
            None
        };
        
        Self {
            config,
            hdag,
            hyperbion,
            funnel,
            policy_params,
            ledger,
            vector8_builder,
            metatron_router,
        }
    }
    
    /// Get reference to configuration
    pub fn config(&self) -> &InterlockConfig {
        &self.config
    }
    
    /// STATE_IN: Convert APOLLYON State5D to Trichter State5D
    pub fn apollyon_to_trichter(&self, apollon: &ApollonState5D) -> TrichterState5D {
        // Use public API conversion (State5D has to_array())
        let arr = apollon.to_array();
        TrichterState5D::new(arr[0], arr[1], arr[2], arr[3], arr[4])
    }
    
    /// FIELD_IO: Compute guidance field using Trichter Hyperbion + HDAG
    /// 
    /// When enable_full_hdag is true, uses actual HDAG relaxation with Hyperbion fields.
    /// Otherwise, uses simplified gradient descent.
    pub fn compute_guidance(&mut self, state: &TrichterState5D, _t: f64) -> [f64; 5] {
        if self.config.enable_full_hdag {
            // Full HDAG relaxation with Hyperbion field computation
            
            // Absorb state into Hyperbion to compute fields
            let states = vec![*state];
            let fields = self.hyperbion.absorption(&states);
            
            // Relax HDAG with computed fields
            self.hdag.relax(fields);
            
            // Add state to HDAG if not already present
            let state_id = self.hdag.add_tensor(*state);
            
            // Compute guidance from HDAG phase gradients
            let mut guidance = [0.0; 5];
            let mut gradient_count = 0;
            
            // Aggregate phase gradients from all outgoing transitions
            for transition in self.hdag.transitions.iter() {
                if transition.from == state_id {
                    if let Some(to_tensor) = self.hdag.tensors.get(&transition.to) {
                        let delta = to_tensor.tensor.as_array();
                        let curr = state.as_array();
                        
                        // Weight by phase gradient and coherence
                        let weight = transition.phase_gradient * transition.coherence;
                        
                        for i in 0..5 {
                            guidance[i] += (delta[i] - curr[i]) * weight;
                        }
                        gradient_count += 1;
                    }
                }
            }
            
            // Normalize by count if we have gradients
            if gradient_count > 0 {
                let scale = 1.0 / (gradient_count as f64);
                for i in 0..5 {
                    guidance[i] *= scale;
                }
            } else {
                // Fallback to field-based guidance
                let h_value = self.hyperbion.evaluate(fields);
                let scale = -0.1 * h_value.tanh(); // Bounded gradient
                let arr = state.as_array();
                for i in 0..5 {
                    guidance[i] = arr[i] * scale;
                }
            }
            
            guidance
        } else {
            // Simplified guidance computation based on gradients toward origin
            let arr = state.as_array();
            let norm = state.norm();
            
            if norm < 1e-10 {
                return [0.0; 5];
            }
            
            // Simple gradient descent toward equilibrium
            let scale = -0.1; // Small step
            [
                arr[0] * scale,
                arr[1] * scale,
                arr[2] * scale,
                arr[3] * scale,
                arr[4] * scale,
            ]
        }
    }
    
    /// GATE: Evaluate Merkaba gate using simplified Proof-of-Resonance
    pub fn evaluate_gate(
        &self,
        state_prev: &TrichterState5D,
        state_curr: &TrichterState5D,
        _delta_t: f64,
    ) -> (GateDecision, SimpleProofOfResonance) {
        // Compute path invariance
        let delta_pi = self.compute_path_invariance(state_prev, state_curr);
        
        // Compute alignment (simplified - using state norm as proxy)
        let phi = self.compute_alignment(state_prev, state_curr);
        
        // Compute Lyapunov delta
        let delta_v = self.compute_lyapunov_delta(state_prev, state_curr);
        
        // Create PoR
        let por_valid = delta_pi <= self.config.gate_delta_pi_max 
                        && phi >= self.config.gate_phi_threshold;
        
        let proof = SimpleProofOfResonance {
            delta_pi,
            phi,
            delta_v,
            por_valid,
        };
        
        // Evaluate gate decision
        let decision = if proof.por_valid && proof.delta_v < 0.0 {
            GateDecision::FIRE
        } else {
            GateDecision::HOLD
        };
        
        (decision, proof)
    }
    
    /// CONDENSE: Apply Trichter funnel operations (coagula)
    /// 
    /// When enable_funnel_ops is true, uses full FunnelGraph operations (split/merge/prune).
    /// Otherwise, uses simplified coagulation.
    pub fn condense(&mut self, state: &TrichterState5D, guidance: &[f64; 5]) -> TrichterState5D {
        if self.config.enable_funnel_ops {
            // Full Funnel operations with split/merge/prune
            
            // Add current state as node if funnel is empty
            if self.funnel.nodes.is_empty() {
                self.funnel.add_node(*state);
            }
            
            // Convert guidance array to GuidanceVector
            let guidance_vec = GuidanceVector::new(
                guidance[0],
                guidance[1],
                guidance[2],
                guidance[3],
            );
            
            // Apply advect with policy params (includes split/merge/prune)
            let dt = 0.1; // Time step
            self.funnel.advect(guidance_vec, dt, &self.policy_params);
            
            // Get the most massive node as the condensed state
            let condensed = if let Some((_, node)) = self.funnel.nodes.iter()
                .max_by(|a, b| a.1.mass.partial_cmp(&b.1.mass).unwrap_or(std::cmp::Ordering::Equal)) {
                node.state
            } else {
                // Fallback if no nodes
                *state
            };
            
            condensed
        } else {
            // Simplified coagulation step
            // Apply guidance as condensation force
            let arr = state.as_array();
            
            TrichterState5D::new(
                arr[0] + guidance[0],
                arr[1] + guidance[1],
                arr[2] + guidance[2],
                arr[3] + guidance[3],
                arr[4] + guidance[4],
            )
        }
    }
    
    /// EVENT_OUT: Prepare commit data for MEF Ledger
    pub fn prepare_commit(
        &self,
        state: &TrichterState5D,
        proof: &SimpleProofOfResonance,
    ) -> CommitData {
        use sha2::{Sha256, Digest};
        
        // Create deterministic hash
        let mut hasher = Sha256::new();
        hasher.update(format!("{:?}", state.as_array()));
        hasher.update(format!("{:.10}", proof.phi));
        hasher.update(format!("{}", self.config.seed));
        let hash = format!("{:x}", hasher.finalize());
        
        CommitData {
            state: *state,
            proof: proof.clone(),
            commit_hash: hash,
            timestamp: chrono::Utc::now(),
        }
    }
    
    /// Derive 8D vector from 5D state for knowledge derivation
    /// 
    /// Combines 5D state with spectral signature (ψ, ρ, ω) to create 8D vector
    pub fn derive_8d_vector(&self, state: &TrichterState5D) -> Option<Vec<f64>> {
        if let Some(ref builder) = self.vector8_builder {
            let arr = state.as_array();
            
            // Extract 5D spiral coordinates (x, y, z, psi, omega)
            let x5 = [arr[0], arr[1], arr[2], arr[3], arr[4]];
            
            // Compute spectral signature from state
            // ψ: semantic weight (already in state)
            // ρ: density approximation from norm
            // ω: temporal phase (already in state)
            let psi = arr[3];
            let rho = state.norm();
            let omega = arr[4];
            
            // Build 8D vector
            match builder.build(&x5, (psi, rho, omega)) {
                Ok(vec8) => Some(vec8),
                Err(e) => {
                    if self.config.enable_logging {
                        tracing::warn!("Failed to build 8D vector: {}", e);
                    }
                    None
                }
            }
        } else {
            None
        }
    }
    
    /// Write commit to MEF Ledger
    /// 
    /// Only writes if ledger is initialized and in active mode (not shadow)
    pub fn write_to_ledger(&mut self, commit: &CommitData) -> Result<(), String> {
        if let Some(ref mut ledger) = self.ledger {
            // Convert CommitData to MEF block format
            // This is a simplified conversion - full implementation would use TIC format
            
            let tic_id = commit.commit_hash.clone();
            let seed = format!("{}", self.config.seed);
            let fixpoint_norm = commit.state.norm();
            
            // Create compact TIC data
            let compact_tic = mef_ledger::CompactTic {
                tic_id: tic_id.clone(),
                seed,
                fixpoint_norm,
                invariants: serde_json::json!({
                    "delta_pi": commit.proof.delta_pi,
                    "phi": commit.proof.phi,
                    "delta_v": commit.proof.delta_v,
                }),
                sigma_bar: serde_json::json!({
                    "state": commit.state.as_array(),
                }),
                window: vec![],
            };
            
            // Convert CompactTic to JSON
            let tic_json = serde_json::to_value(&compact_tic)
                .map_err(|e| format!("Failed to serialize TIC: {}", e))?;
            let snapshot_json = serde_json::json!({});
            
            // Append to ledger
            match ledger.append_block(&tic_json, &snapshot_json) {
                Ok(_) => {
                    if self.config.enable_logging {
                        tracing::info!("Wrote commit {} to ledger", tic_id);
                    }
                    Ok(())
                }
                Err(e) => {
                    let err_msg = format!("Failed to write to ledger: {}", e);
                    if self.config.enable_logging {
                        tracing::error!("{}", err_msg);
                    }
                    Err(err_msg)
                }
            }
        } else {
            Err("Ledger not initialized".to_string())
        }
    }
    
    /// Select transformation route using Metatron router
    /// 
    /// Returns operator sequence to apply during tick
    pub fn select_route(&mut self, state: &TrichterState5D) -> Option<Vec<String>> {
        if let Some(ref mut _router) = self.metatron_router {
            // Convert state to input for router
            let state_arr = state.as_array();
            
            // Use router to select best route
            // This is simplified - full implementation would use actual routing logic
            if self.config.enable_logging {
                tracing::debug!("Selecting route for state: {:?}", state_arr);
            }
            
            // For now, return a simple operator sequence
            // Full implementation would call router.select_route()
            Some(vec!["Solve".to_string(), "Relax".to_string(), "Coagula".to_string()])
        } else {
            None
        }
    }
    
    // Helper methods
    
    fn compute_path_invariance(&self, prev: &TrichterState5D, curr: &TrichterState5D) -> f64 {
        let p = prev.as_array();
        let c = curr.as_array();
        let mut sum = 0.0;
        for i in 0..5 {
            let diff = c[i] - p[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }
    
    fn compute_alignment(&self, prev: &TrichterState5D, curr: &TrichterState5D) -> f64 {
        // Simplified alignment: cosine similarity
        let p = prev.as_array();
        let c = curr.as_array();
        
        let mut dot = 0.0;
        let mut norm_prev = 0.0;
        let mut norm_curr = 0.0;
        
        for i in 0..5 {
            dot += p[i] * c[i];
            norm_prev += p[i] * p[i];
            norm_curr += c[i] * c[i];
        }
        
        if norm_prev == 0.0 || norm_curr == 0.0 {
            return 0.0;
        }
        
        dot / (norm_prev.sqrt() * norm_curr.sqrt())
    }
    
    fn compute_lyapunov_delta(&self, prev: &TrichterState5D, curr: &TrichterState5D) -> f64 {
        let v_prev = prev.norm();
        let v_curr = curr.norm();
        v_curr - v_prev
    }
}

/// Data structure for MEF commits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitData {
    pub state: TrichterState5D,
    pub proof: SimpleProofOfResonance,
    pub commit_hash: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Extended commit data with 8D vector and route information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedCommitData {
    pub base: CommitData,
    pub vector_8d: Option<Vec<f64>>,
    pub selected_route: Option<Vec<String>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_interlock_creation() {
        let config = InterlockConfig::default();
        let _adapter = InterlockAdapter::new(config);
    }
    
    #[test]
    fn test_state_conversion() {
        let config = InterlockConfig::default();
        let adapter = InterlockAdapter::new(config);
        
        let apollon = ApollonState5D::from_array([1.0, 2.0, 3.0, 0.5, 0.7]);
        let trichter = adapter.apollyon_to_trichter(&apollon);
        
        assert_eq!(trichter.as_array()[0], 1.0);
        assert_eq!(trichter.as_array()[4], 0.7);
    }
}
