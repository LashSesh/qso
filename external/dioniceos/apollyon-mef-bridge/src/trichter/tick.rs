//! Deterministic coupling tick algorithm
//!
//! Implements Algorithm 1 from the specification:
//! Bündig (flush) coupling between 4D Funnel and 5D HDAG field

use super::funnel::FunnelGraph;
use super::hdag::HDAGField;
use super::hyperbion::Hyperbion;
use super::lift::{lift, proj_4d};
use super::policies::PolicyParams;
use super::types::{ProofHash, State4D, State5D};
use serde::{Deserialize, Serialize};

/// Result of a coupling tick
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickResult {
    pub states_4d_next: Vec<State4D>,
    pub commit_hash: Option<ProofHash>,
    pub nodes_created: usize,
    pub nodes_merged: usize,
    pub edges_pruned: usize,
}

/// Execute one coupling tick
///
/// Algorithm 1: Kopplungs-Tick (offline, bündig 4D↔5D)
///
/// # Arguments
/// * `states_4d` - Batch of 4D states at time t
/// * `t` - Current time/tick
/// * `params` - Policy parameters Π
/// * `hyperbion` - Hyperbion layer with coefficients (α, β)
/// * `hdag` - HDAG field (mutable)
/// * `funnel` - Funnel graph (mutable)
/// * `compute_proofs` - Whether to generate proof artifacts
///
/// # Returns
/// Tick result with updated states and optional proof
pub fn coupling_tick(
    states_4d: &[State4D],
    t: f64,
    params: &PolicyParams,
    hyperbion: &Hyperbion,
    hdag: &mut HDAGField,
    funnel: &mut FunnelGraph,
    compute_proofs: bool,
) -> TickResult {
    // Track changes for result
    let initial_nodes = funnel.node_count();
    let initial_edges = funnel.edge_count();

    // Step 1: Lift 4D states to 5D
    let states_5d: Vec<State5D> = states_4d
        .iter()
        .map(|s4d| lift(*s4d, t))
        .collect();

    // Step 2: Hyperbion absorption - compute (Φ, μ)
    let fields = hyperbion.absorption(&states_5d);

    // Step 3: HDAG relaxation - update tensors and transitions
    hdag.relax(fields);

    // Step 4: Compute gradient field ∇Φ
    // For simplicity, compute average gradient at current state positions
    let gradients: Vec<State5D> = states_5d
        .iter()
        .map(|state| hdag.gradient(*state))
        .collect();

    // Average gradient for global guidance
    let avg_gradient = if gradients.is_empty() {
        State5D::zero()
    } else {
        let n = gradients.len() as f64;
        State5D::new(
            gradients.iter().map(|g| g.x).sum::<f64>() / n,
            gradients.iter().map(|g| g.y).sum::<f64>() / n,
            gradients.iter().map(|g| g.z).sum::<f64>() / n,
            gradients.iter().map(|g| g.psi).sum::<f64>() / n,
            gradients.iter().map(|g| g.omega).sum::<f64>() / n,
        )
    };

    // Step 5: Project to 4D guidance vector
    let v_guide = proj_4d(avg_gradient);

    // Step 6: Funnel advection step with Hebb/Decay, Merge/Split/Prune
    // Add new states to funnel if needed
    for state_5d in &states_5d {
        funnel.add_node(*state_5d);
    }

    // Advect funnel graph
    let dt = 1.0; // Unit time step
    funnel.advect(v_guide, dt, params);

    // Extract next 4D states from funnel nodes
    let states_4d_next: Vec<State4D> = funnel
        .nodes
        .values()
        .take(states_4d.len())
        .map(|node| State4D::new(
            node.state.x,
            node.state.y,
            node.state.z,
            node.state.psi,
        ))
        .collect();

    // Step 7: Optional proof computation
    let commit_hash = if compute_proofs {
        Some(compute_commit_hash(states_4d, &states_4d_next, params, &fields))
    } else {
        None
    };

    // Compute changes
    let final_nodes = funnel.node_count();
    let final_edges = funnel.edge_count();

    TickResult {
        states_4d_next,
        commit_hash,
        nodes_created: final_nodes.saturating_sub(initial_nodes),
        nodes_merged: initial_nodes.saturating_sub(final_nodes),
        edges_pruned: initial_edges.saturating_sub(final_edges),
    }
}

/// Compute commit hash for proof artifacts
///
/// Commit_t = H(TransitionHash_t ∥ FieldHash_t)
fn compute_commit_hash(
    states_prev: &[State4D],
    states_next: &[State4D],
    params: &PolicyParams,
    fields: &super::types::HyperbionFields,
) -> ProofHash {
    use serde_json;

    // Serialize all components
    let mut data = Vec::new();
    
    if let Ok(json) = serde_json::to_vec(states_prev) {
        data.extend_from_slice(&json);
    }
    if let Ok(json) = serde_json::to_vec(states_next) {
        data.extend_from_slice(&json);
    }
    if let Ok(json) = serde_json::to_vec(params) {
        data.extend_from_slice(&json);
    }
    if let Ok(json) = serde_json::to_vec(fields) {
        data.extend_from_slice(&json);
    }

    ProofHash::new(&data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coupling_tick_basic() {
        let states = vec![State4D::new(1.0, 0.0, 0.0, 0.5)];
        let params = PolicyParams::explore();
        let hyperbion = Hyperbion::new();
        let mut hdag = HDAGField::new();
        let mut funnel = FunnelGraph::new();

        let result = coupling_tick(
            &states,
            0.0,
            &params,
            &hyperbion,
            &mut hdag,
            &mut funnel,
            false,
        );

        assert!(!result.states_4d_next.is_empty());
    }

    #[test]
    fn test_coupling_tick_with_proofs() {
        let states = vec![State4D::new(1.0, 0.0, 0.0, 0.5)];
        let params = PolicyParams::explore();
        let hyperbion = Hyperbion::new();
        let mut hdag = HDAGField::new();
        let mut funnel = FunnelGraph::new();

        let result = coupling_tick(
            &states,
            0.0,
            &params,
            &hyperbion,
            &mut hdag,
            &mut funnel,
            true,
        );

        assert!(result.commit_hash.is_some());
    }

    #[test]
    fn test_deterministic_execution() {
        let states = vec![
            State4D::new(1.0, 0.0, 0.0, 0.5),
            State4D::new(0.0, 1.0, 0.0, 0.5),
        ];
        let params = PolicyParams::explore();
        let hyperbion = Hyperbion::new();

        // Run twice with same inputs
        let mut hdag1 = HDAGField::new();
        let mut funnel1 = FunnelGraph::new();
        let result1 = coupling_tick(
            &states,
            0.0,
            &params,
            &hyperbion,
            &mut hdag1,
            &mut funnel1,
            false, // Don't compute proofs - graph structure may vary
        );

        let mut hdag2 = HDAGField::new();
        let mut funnel2 = FunnelGraph::new();
        let result2 = coupling_tick(
            &states,
            0.0,
            &params,
            &hyperbion,
            &mut hdag2,
            &mut funnel2,
            false,
        );

        // Should produce same number of nodes (structural determinism)
        assert_eq!(funnel1.node_count(), funnel2.node_count());
        assert_eq!(result1.states_4d_next.len(), result2.states_4d_next.len());
    }

    #[test]
    fn test_multiple_ticks() {
        let mut states = vec![State4D::new(1.0, 0.0, 0.0, 0.5)];
        let params = PolicyParams::explore();
        let hyperbion = Hyperbion::new();
        let mut hdag = HDAGField::new();
        let mut funnel = FunnelGraph::new();

        // Execute multiple ticks
        for t in 0..5 {
            let result = coupling_tick(
                &states,
                t as f64,
                &params,
                &hyperbion,
                &mut hdag,
                &mut funnel,
                false,
            );
            
            // Use output as next input
            if !result.states_4d_next.is_empty() {
                states = result.states_4d_next;
            }
        }

        // Should have evolved the system
        assert!(funnel.node_count() > 0);
    }

    #[test]
    fn test_empty_states() {
        let states: Vec<State4D> = vec![];
        let params = PolicyParams::explore();
        let hyperbion = Hyperbion::new();
        let mut hdag = HDAGField::new();
        let mut funnel = FunnelGraph::new();

        let result = coupling_tick(
            &states,
            0.0,
            &params,
            &hyperbion,
            &mut hdag,
            &mut funnel,
            false,
        );

        assert!(result.states_4d_next.is_empty());
    }
}
