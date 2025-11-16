//! # Quantum Walk Toolkit
//!
//! High-level toolkit for graph analysis using quantum walks.
//!
//! This module provides production-ready functions for:
//! - Node centrality/ranking via quantum walk statistics
//! - Anomaly detection in graph structure
//! - Connectivity analysis and resilience metrics
//!
//! ## Use Cases
//! - Social network analysis (influence ranking)
//! - Network monitoring (anomaly detection)
//! - Infrastructure resilience (connectivity metrics)

use crate::graph::metatron::MetatronGraph;
use crate::hamiltonian::MetatronHamiltonian;
use crate::params::QSOParameters;
use crate::quantum::state::QuantumState;
use crate::quantum_walk::continuous::ContinuousTimeQuantumWalk;
use serde::{Deserialize, Serialize};

/// Parameters for quantum walk toolkit operations
#[derive(Debug, Clone)]
pub struct QuantumWalkParams {
    /// Maximum evolution time
    pub t_max: f64,
    /// Time step for evolution
    pub dt: f64,
    /// Number of samples for statistical averaging
    pub samples: usize,
}

impl Default for QuantumWalkParams {
    fn default() -> Self {
        Self {
            t_max: 10.0,
            dt: 0.1,
            samples: 128,
        }
    }
}

/// Connectivity metrics from quantum walk analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityMetrics {
    /// Mixing time (time to reach near-uniform distribution)
    pub mixing_time: f64,
    /// Average hitting probabilities from source nodes
    pub hitting_probabilities: Vec<f64>,
    /// Variance in probability distribution (lower = better mixed)
    pub distribution_variance: f64,
    /// Effective graph diameter (quantum walk perspective)
    pub effective_diameter: f64,
}

/// Compute quantum walk centrality for each node
///
/// Returns a centrality score for each node based on quantum walk dynamics.
/// Higher scores indicate more "central" or influential nodes.
///
/// # Algorithm
/// - Runs quantum walks from each node
/// - Computes visitation probabilities over time
/// - Aggregates into a centrality score
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `params` - Quantum walk parameters
///
/// # Returns
/// Vector of centrality scores (one per node, normalized to [0, 1])
pub fn quantum_walk_centrality(graph: &MetatronGraph, params: &QuantumWalkParams) -> Vec<f64> {
    let n = graph.nodes().len();
    let qso_params = QSOParameters::default();
    let hamiltonian = MetatronHamiltonian::new(graph, &qso_params);
    let qw = ContinuousTimeQuantumWalk::new(&hamiltonian);

    let mut centrality = vec![0.0; n];

    // For each node, measure how "accessible" it is from all other nodes
    for start_node in 0..n {
        let initial_state = QuantumState::basis_state(start_node).unwrap();

        // Sample at different times and accumulate probabilities
        let num_steps = (params.t_max / params.dt).ceil() as usize;
        for step in 1..=num_steps {
            let t = (step as f64) * params.dt;
            let evolved = qw.evolve(&initial_state, t);
            let probs = evolved.probabilities();

            // Accumulate probability of being at each node
            for (i, &prob) in probs.iter().enumerate() {
                centrality[i] += prob;
            }
        }
    }

    // Normalize by number of steps and nodes
    let norm_factor = (n * ((params.t_max / params.dt).ceil() as usize)) as f64;
    for score in &mut centrality {
        *score /= norm_factor;
    }

    // Re-normalize to [0, 1]
    let max_score = centrality.iter().cloned().fold(0.0, f64::max);
    if max_score > 0.0 {
        for score in &mut centrality {
            *score /= max_score;
        }
    }

    centrality
}

/// Compute anomaly scores by comparing base graph to current graph
///
/// Detects structural changes between a baseline graph and current graph
/// using quantum walk dynamics.
///
/// # Arguments
/// * `base_graph` - Baseline/reference graph
/// * `current_graph` - Current graph to analyze
/// * `params` - Quantum walk parameters
///
/// # Returns
/// Vector of anomaly scores per node (higher = more anomalous)
///
/// # Note
/// Currently both graphs must have the same structure (same nodes).
/// For Metatron graph, this compares different edge configurations.
pub fn quantum_walk_anomaly_score(
    base_graph: &MetatronGraph,
    current_graph: &MetatronGraph,
    params: &QuantumWalkParams,
) -> Vec<f64> {
    // Compute centrality for both graphs
    let base_centrality = quantum_walk_centrality(base_graph, params);
    let current_centrality = quantum_walk_centrality(current_graph, params);

    // Anomaly = absolute difference in centrality
    base_centrality
        .iter()
        .zip(current_centrality.iter())
        .map(|(base, curr)| (base - curr).abs())
        .collect()
}

/// Analyze connectivity using quantum walks from source nodes
///
/// Computes various connectivity metrics based on quantum walk dynamics
/// starting from specified source nodes.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `source_nodes` - Starting nodes for quantum walk
/// * `params` - Quantum walk parameters
///
/// # Returns
/// Connectivity metrics structure
pub fn quantum_walk_connectivity(
    graph: &MetatronGraph,
    source_nodes: &[usize],
    params: &QuantumWalkParams,
) -> ConnectivityMetrics {
    let n = graph.nodes().len();
    let qso_params = QSOParameters::default();
    let hamiltonian = MetatronHamiltonian::new(graph, &qso_params);
    let qw = ContinuousTimeQuantumWalk::new(&hamiltonian);

    // Create initial state (uniform over source nodes)
    let mut amplitudes = vec![num_complex::Complex64::new(0.0, 0.0); n];
    let amplitude = num_complex::Complex64::new(
        1.0 / (source_nodes.len() as f64).sqrt(),
        0.0,
    );
    for &node in source_nodes {
        amplitudes[node] = amplitude;
    }
    let initial_state = QuantumState::from_amplitudes(amplitudes).unwrap();

    // Evolve and track metrics
    let num_steps = (params.t_max / params.dt).ceil() as usize;
    let mut mixing_time = params.t_max;
    let mut final_probs = vec![0.0; n];
    let uniform_prob = 1.0 / n as f64;
    let mixing_threshold = 0.1; // 10% deviation from uniform

    for step in 1..=num_steps {
        let t = (step as f64) * params.dt;
        let evolved = qw.evolve(&initial_state, t);
        let probs = evolved.probabilities();

        // Check if mixed (close to uniform distribution)
        let max_deviation = probs
            .iter()
            .map(|&p| (p - uniform_prob).abs())
            .fold(0.0, f64::max);

        if max_deviation < mixing_threshold && mixing_time == params.t_max {
            mixing_time = t;
        }

        if step == num_steps {
            final_probs = probs.to_vec();
        }
    }

    // Compute variance
    let mean_prob = final_probs.iter().sum::<f64>() / n as f64;
    let variance = final_probs
        .iter()
        .map(|&p| (p - mean_prob).powi(2))
        .sum::<f64>()
        / n as f64;

    // Effective diameter (heuristic based on mixing time)
    let effective_diameter = mixing_time / params.dt;

    ConnectivityMetrics {
        mixing_time,
        hitting_probabilities: final_probs,
        distribution_variance: variance,
        effective_diameter,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_walk_centrality() {
        let graph = MetatronGraph::new();
        let params = QuantumWalkParams {
            t_max: 5.0,
            dt: 0.5,
            samples: 16,
        };

        let centrality = quantum_walk_centrality(&graph, &params);

        // Should have one score per node
        assert_eq!(centrality.len(), 13);

        // All scores should be in [0, 1]
        for &score in &centrality {
            assert!(score >= 0.0 && score <= 1.0);
        }

        // Central node (0) should have high centrality
        assert!(centrality[0] > 0.5);
    }

    #[test]
    fn test_quantum_walk_connectivity() {
        let graph = MetatronGraph::new();
        let params = QuantumWalkParams {
            t_max: 10.0,
            dt: 0.1,
            samples: 64,
        };

        let metrics = quantum_walk_connectivity(&graph, &[0], &params);

        // Mixing time should be reasonable
        assert!(metrics.mixing_time > 0.0 && metrics.mixing_time <= params.t_max);

        // Should have hitting probs for all nodes
        assert_eq!(metrics.hitting_probabilities.len(), 13);

        // Variance should be small (high connectivity)
        assert!(metrics.distribution_variance < 0.01);
    }
}
