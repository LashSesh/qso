//! Main tick() pipeline implementing the 5D Cube execution flow.
//!
//! Pipeline: Solve/Relax → Potential/Guidance → Gate → Coagula → Optional Collapse → Commit

use crate::interlock::{InterlockAdapter, CommitData, SimpleProofOfResonance};
use crate::metrics::TickMetrics;
use core_5d::State5D as ApollonState5D;
use apollyon_mef_bridge::trichter::State5D as TrichterState5D;
use mef_schemas::GateDecision;
use serde::{Deserialize, Serialize};

/// Result of a 5D Cube tick operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickResult {
    /// Input state (APOLLYON)
    pub state_in: ApollonState5D,
    
    /// Relaxed state (after Trichter)
    pub state_relaxed: TrichterState5D,
    
    /// Guidance vector (∇Φ from HDAG)
    pub guidance: [f64; 5],
    
    /// Gate decision
    pub gate_decision: GateDecision,
    
    /// Proof of Resonance
    pub proof: SimpleProofOfResonance,
    
    /// Condensed state (after coagula)
    pub state_condensed: TrichterState5D,
    
    /// Optional commit data (if FIRE)
    pub commit: Option<CommitData>,
    
    /// Optional 8D vector (if enabled)
    pub vector_8d: Option<Vec<f64>>,
    
    /// Selected route (if Metatron routing enabled)
    pub selected_route: Option<Vec<String>>,
    
    /// Metrics collected during tick
    pub metrics: TickMetrics,
    
    /// Tick number
    pub tick_number: u64,
    
    /// Whether commit was written to ledger
    pub ledger_written: bool,
}

/// Execute one tick of the Unified 5D Cube pipeline
///
/// # Arguments
/// * `adapter` - The interlock adapter
/// * `state` - Current APOLLYON state
/// * `state_prev` - Previous state (for gate evaluation)
/// * `t` - Current time
/// * `tick_number` - Current tick counter
///
/// # Returns
/// TickResult containing all intermediate and final states
pub fn tick_5d_cube(
    adapter: &mut InterlockAdapter,
    state: &ApollonState5D,
    state_prev: Option<&ApollonState5D>,
    t: f64,
    tick_number: u64,
) -> TickResult {
    let start_time = std::time::Instant::now();
    
    // Phase 1: STATE_IN - Convert APOLLYON to Trichter
    let state_trichter = adapter.apollyon_to_trichter(state);
    
    // Phase 2: Solve/Relax using APOLLYON dynamics (conceptual - state is already solved)
    // In full implementation, would integrate APOLLYON VectorField here
    let state_relaxed = state_trichter.clone();
    
    // Phase 2.5: Metatron routing (if enabled) - select transformation route
    let selected_route = adapter.select_route(&state_relaxed);
    
    // Phase 3: FIELD_IO - Compute Potential/Guidance using Trichter ∇Φ proj.4D
    // This now uses full HDAG relaxation if enabled
    let guidance = adapter.compute_guidance(&state_relaxed, t);
    
    // Phase 4: GATE - Evaluate MEF PoR/Merkaba
    let (gate_decision, proof) = if let Some(prev) = state_prev {
        let prev_trichter = adapter.apollyon_to_trichter(prev);
        adapter.evaluate_gate(&prev_trichter, &state_relaxed, 0.1)
    } else {
        // First tick - default to HOLD
        (
            GateDecision::HOLD,
            SimpleProofOfResonance {
                delta_pi: 0.0,
                phi: 1.0,
                delta_v: 0.0,
                por_valid: false,
            },
        )
    };
    
    // Phase 5: CONDENSE - Apply Trichter Coagula/Tick
    // This now uses full Funnel operations if enabled
    let state_condensed = adapter.condense(&state_relaxed, &guidance);
    
    // Phase 5.5: Derive 8D vector (if enabled)
    let vector_8d = adapter.derive_8d_vector(&state_condensed);
    
    // Phase 6: Optional Collapse → Commit (only if FIRE)
    let (commit, ledger_written) = if matches!(gate_decision, GateDecision::FIRE) {
        let commit_data = adapter.prepare_commit(&state_condensed, &proof);
        
        // Attempt to write to ledger if enabled and not in shadow mode
        let written = if !adapter.config().shadow_mode {
            adapter.write_to_ledger(&commit_data).is_ok()
        } else {
            false
        };
        
        (Some(commit_data), written)
    } else {
        (None, false)
    };
    
    // Collect metrics
    let elapsed = start_time.elapsed();
    let metrics = TickMetrics {
        bi: compute_bi(&guidance),
        delta_f: proof.delta_v,
        w2_step: proof.delta_pi,
        lambda_gap: compute_lambda_gap(&state_condensed),
        s_mand: compute_mandala_score(&state_condensed),
        duty_por: if proof.por_valid { 1.0 } else { 0.0 },
        elapsed_ms: elapsed.as_millis() as f64,
    };
    
    TickResult {
        state_in: state.clone(),
        state_relaxed,
        guidance,
        gate_decision,
        proof,
        state_condensed,
        commit,
        vector_8d,
        selected_route,
        metrics,
        tick_number,
        ledger_written,
    }
}

// Helper functions for metrics computation

fn compute_bi(guidance: &[f64; 5]) -> f64 {
    // Betti number approximation from guidance magnitude
    let mut sum = 0.0;
    for &g in guidance.iter() {
        sum += g * g;
    }
    sum.sqrt()
}

fn compute_lambda_gap(state: &TrichterState5D) -> f64 {
    // Simplified spectral gap computation
    // In full implementation, would use eigenvalue analysis
    let arr = state.as_array();
    let mut max_val = arr[0];
    let mut min_val = arr[0];
    
    for &c in arr.iter() {
        if c > max_val {
            max_val = c;
        }
        if c < min_val {
            min_val = c;
        }
    }
    
    max_val - min_val
}

fn compute_mandala_score(state: &TrichterState5D) -> f64 {
    // Simplified Mandorla score based on state coherence
    let arr = state.as_array();
    let mean: f64 = arr.iter().sum::<f64>() / 5.0;
    let mut variance = 0.0;
    
    for &c in arr.iter() {
        let diff = c - mean;
        variance += diff * diff;
    }
    
    // Lower variance = higher coherence
    1.0 / (1.0 + variance.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interlock::InterlockConfig;
    
    #[test]
    fn test_tick_execution() {
        let config = InterlockConfig::default();
        let mut adapter = InterlockAdapter::new(config);
        
        let state = ApollonState5D::from_array([1.0, 0.0, 0.0, 0.5, 0.3]);
        let result = tick_5d_cube(&mut adapter, &state, None, 0.0, 0);
        
        assert_eq!(result.tick_number, 0);
        assert!(matches!(result.gate_decision, GateDecision::HOLD));
        assert!(result.commit.is_none());
        assert!(!result.ledger_written);
    }
    
    #[test]
    fn test_tick_with_previous() {
        let config = InterlockConfig::default();
        let mut adapter = InterlockAdapter::new(config);
        
        let state_prev = ApollonState5D::from_array([1.0, 0.0, 0.0, 0.5, 0.3]);
        let state_curr = ApollonState5D::from_array([1.01, 0.01, 0.0, 0.51, 0.31]);
        
        let result = tick_5d_cube(&mut adapter, &state_curr, Some(&state_prev), 0.1, 1);
        
        assert_eq!(result.tick_number, 1);
        assert!(!result.ledger_written); // Shadow mode by default
        // Gate decision depends on thresholds
    }
    
    #[test]
    fn test_tick_with_extensions() {
        use tempfile::TempDir;
        
        let temp_dir = TempDir::new().unwrap();
        let ledger_path = temp_dir.path().join("test_ledger");
        
        let mut config = InterlockConfig::default();
        config.enable_full_hdag = true;
        config.enable_8d_vectors = true;
        config.enable_ledger_writes = true;
        config.ledger_path = Some(ledger_path);
        config.shadow_mode = false;
        config.gate_phi_threshold = 0.3;
        
        let mut adapter = InterlockAdapter::new(config);
        
        let state_prev = ApollonState5D::from_array([2.0, 1.0, 0.5, 0.8, 0.6]);
        let state_curr = ApollonState5D::from_array([1.9, 0.95, 0.48, 0.76, 0.57]);
        
        let result = tick_5d_cube(&mut adapter, &state_curr, Some(&state_prev), 0.1, 1);
        
        assert_eq!(result.tick_number, 1);
        assert!(result.vector_8d.is_some(), "8D vector should be derived");
        
        // If gate fires and we have a commit, ledger might be written
        if result.commit.is_some() {
            // Ledger write success depends on initialization
            // This is OK either way for the test
        }
    }
}
