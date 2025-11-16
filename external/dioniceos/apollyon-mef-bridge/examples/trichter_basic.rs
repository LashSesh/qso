//! Example: Basic 4D-Trichter Usage
//!
//! Demonstrates the complete 4D-Trichter workflow with morphodynamic
//! pattern compression.

use apollyon_mef_bridge::{
    coupling_tick, FunnelGraph, HDAGField, Hyperbion, Policy, State4D,
};

fn main() {
    println!("=== 4D-Trichter (Gabriel) Example ===\n");

    // Initialize the system components
    println!("Initializing system...");
    let policy = Policy::Explore.params();
    let hyperbion = Hyperbion::with_params(1.0, 0.5);
    let mut hdag = HDAGField::new();
    let mut funnel = FunnelGraph::new();

    // Create initial 4D states
    let initial_states = vec![
        State4D::new(1.0, 0.0, 0.0, 0.5),
        State4D::new(0.0, 1.0, 0.0, 0.5),
        State4D::new(-1.0, 0.0, 0.0, 0.3),
        State4D::new(0.0, -1.0, 0.0, 0.3),
    ];

    println!("Initial states: {} nodes", initial_states.len());
    for (i, state) in initial_states.iter().enumerate() {
        println!(
            "  State {}: ({:.2}, {:.2}, {:.2}, ψ={:.2})",
            i, state.x, state.y, state.z, state.psi
        );
    }

    // Run the 4D-Trichter for multiple ticks
    let mut states = initial_states.clone();
    let num_ticks = 10;

    println!("\nRunning {} ticks...\n", num_ticks);

    for t in 0..num_ticks {
        let result = coupling_tick(
            &states,
            t as f64,
            &policy,
            &hyperbion,
            &mut hdag,
            &mut funnel,
            t == 0 || t == num_ticks - 1, // Compute proofs at start and end
        );

        println!("Tick {}:", t);
        println!("  Funnel: {} nodes, {} edges", funnel.node_count(), funnel.edge_count());
        println!("  HDAG: {} tensors, {} transitions", hdag.tensor_count(), hdag.transition_count());
        println!("  Density: {:.2}", funnel.density());
        println!("  Changes: +{} nodes, -{} merged, -{} edges pruned",
                 result.nodes_created, result.nodes_merged, result.edges_pruned);

        if let Some(hash) = result.commit_hash {
            println!("  Proof hash: {:02x}{:02x}...{:02x}{:02x}",
                     hash.0[0], hash.0[1], hash.0[30], hash.0[31]);
        }

        // Use output as next input
        if !result.states_4d_next.is_empty() {
            states = result.states_4d_next;
        }

        println!();
    }

    // Final state
    println!("=== Final State ===");
    println!("Total nodes in funnel: {}", funnel.node_count());
    println!("Total edges in funnel: {}", funnel.edge_count());
    println!("Total tensors in HDAG: {}", hdag.tensor_count());
    println!("Final density: {:.2}", funnel.density());
    println!("\nFinal states ({} total):", states.len());
    for (i, state) in states.iter().take(5).enumerate() {
        println!(
            "  State {}: ({:.3}, {:.3}, {:.3}, ψ={:.3})",
            i, state.x, state.y, state.z, state.psi
        );
    }
    if states.len() > 5 {
        println!("  ... and {} more", states.len() - 5);
    }

    // Demonstrate different policies
    println!("\n=== Policy Comparison ===");
    demonstrate_policy_differences();
}

fn demonstrate_policy_differences() {
    let initial = vec![State4D::new(1.0, 1.0, 0.0, 0.5)];

    // Explore policy
    let mut funnel_explore = FunnelGraph::new();
    let mut hdag_explore = HDAGField::new();
    let policy_explore = Policy::Explore.params();
    let hyperbion = Hyperbion::new();

    println!("\nExplore Policy:");
    println!("  alpha_hebb={:.2}, decay={:.3}, theta_prune={:.3}",
             policy_explore.alpha_hebb, policy_explore.decay, policy_explore.theta_prune);

    let mut states = initial.clone();
    for t in 0..5 {
        let result = coupling_tick(
            &states,
            t as f64,
            &policy_explore,
            &hyperbion,
            &mut hdag_explore,
            &mut funnel_explore,
            false,
        );
        if !result.states_4d_next.is_empty() {
            states = result.states_4d_next;
        }
    }
    println!("  After 5 ticks: {} nodes, {} edges",
             funnel_explore.node_count(), funnel_explore.edge_count());

    // Exploit policy
    let mut funnel_exploit = FunnelGraph::new();
    let mut hdag_exploit = HDAGField::new();
    let policy_exploit = Policy::Exploit.params();

    println!("\nExploit Policy:");
    println!("  alpha_hebb={:.2}, decay={:.3}, theta_prune={:.3}",
             policy_exploit.alpha_hebb, policy_exploit.decay, policy_exploit.theta_prune);

    let mut states = initial.clone();
    for t in 0..5 {
        let result = coupling_tick(
            &states,
            t as f64,
            &policy_exploit,
            &hyperbion,
            &mut hdag_exploit,
            &mut funnel_exploit,
            false,
        );
        if !result.states_4d_next.is_empty() {
            states = result.states_4d_next;
        }
    }
    println!("  After 5 ticks: {} nodes, {} edges",
             funnel_exploit.node_count(), funnel_exploit.edge_count());

    // Homeostasis policy
    let mut funnel_homeo = FunnelGraph::new();
    let mut hdag_homeo = HDAGField::new();
    let policy_homeo = Policy::Homeostasis.params_with_density(10.0);

    println!("\nHomeostasis Policy (target density=10.0):");
    println!("  alpha_hebb={:.2}, decay={:.3}, theta_prune={:.3}",
             policy_homeo.alpha_hebb, policy_homeo.decay, policy_homeo.theta_prune);

    let mut states = initial.clone();
    for t in 0..5 {
        let result = coupling_tick(
            &states,
            t as f64,
            &policy_homeo,
            &hyperbion,
            &mut hdag_homeo,
            &mut funnel_homeo,
            false,
        );
        if !result.states_4d_next.is_empty() {
            states = result.states_4d_next;
        }
    }
    println!("  After 5 ticks: {} nodes, {} edges",
             funnel_homeo.node_count(), funnel_homeo.edge_count());
}
