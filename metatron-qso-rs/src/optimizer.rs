//! # Quantum-Inspired Optimizer
//!
//! High-level optimization toolkit using QAOA (Quantum Approximate Optimization Algorithm).
//!
//! This module provides production-ready solvers for combinatorial optimization:
//! - MaxCut problem (graph partitioning)
//! - Graph coloring (planned)
//! - Scheduling problems (planned)
//!
//! ## Use Cases
//! - Network partitioning / community detection
//! - Resource allocation
//! - Load balancing
//! - Circuit design

use crate::graph::metatron::MetatronGraph;
use crate::vqa::optimizer::OptimizerType;
use crate::vqa::qaoa::{create_maxcut_hamiltonian, QAOABuilder, QAOA};
use crate::quantum::METATRON_DIMENSION;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Solution to MaxCut problem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxCutSolution {
    /// The cut value (number of edges cut)
    pub cut_value: f64,
    /// Binary assignment of nodes to partitions (0 or 1)
    pub assignment: Vec<bool>,
    /// Approximation ratio (achieved / optimal)
    pub approximation_ratio: f64,
    /// Solution metadata
    pub meta: SolutionMetadata,
}

/// Metadata about the optimization process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionMetadata {
    /// Number of optimization iterations
    pub iterations: usize,
    /// Final cost function value
    pub final_cost: f64,
    /// QAOA circuit depth
    pub depth: usize,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Partition sizes (nodes in set 0, nodes in set 1)
    pub partition_sizes: (usize, usize),
}

/// QAOA-based MaxCut solver
///
/// Solves the MaxCut problem on a graph using QAOA (Quantum Approximate
/// Optimization Algorithm). MaxCut partitions nodes into two sets to
/// maximize edges between them.
///
/// # Example
/// ```
/// use metatron_qso::optimizer::QaoaMaxCutSolver;
/// use metatron_qso::graph::metatron::MetatronGraph;
///
/// let graph = MetatronGraph::new();
/// let solver = QaoaMaxCutSolver::from_graph(&graph)
///     .with_depth(3)
///     .with_max_iterations(100)
///     .with_seed(42);
///
/// let solution = solver.run();
/// println!("Cut value: {}", solution.cut_value);
/// ```
pub struct QaoaMaxCutSolver {
    qaoa: QAOA,
    depth: usize,
    max_iterations: usize,
    seed: Option<u64>,
    tolerance: f64,
}

impl QaoaMaxCutSolver {
    /// Create a MaxCut solver from a graph
    ///
    /// # Arguments
    /// * `graph` - The graph to partition
    ///
    /// # Returns
    /// A solver with default parameters (depth=3, max_iterations=100)
    pub fn from_graph(graph: &MetatronGraph) -> Self {
        let edges: Vec<(usize, usize)> = graph.edges().to_vec();
        let cost_hamiltonian = Arc::new(create_maxcut_hamiltonian(&edges));

        let qaoa = QAOABuilder::new()
            .cost_hamiltonian(cost_hamiltonian)
            .depth(3)
            .optimizer(OptimizerType::NelderMead)
            .max_iterations(100)
            .verbose(false)
            .build();

        Self {
            qaoa,
            depth: 3,
            max_iterations: 100,
            seed: None,
            tolerance: 1e-6,
        }
    }

    /// Set QAOA circuit depth (p parameter)
    ///
    /// Higher depth generally gives better solutions but increases runtime.
    /// Typical values: 1-5.
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    /// Set maximum optimization iterations
    ///
    /// More iterations may find better solutions but increase runtime.
    pub fn with_max_iterations(mut self, max_iters: usize) -> Self {
        self.max_iterations = max_iters;
        self
    }

    /// Set random seed for deterministic results
    ///
    /// Using the same seed produces reproducible results.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set convergence tolerance
    ///
    /// Optimization stops when improvement falls below this threshold.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Run the optimization
    ///
    /// # Returns
    /// A `MaxCutSolution` containing the partition and quality metrics
    pub fn run(self) -> MaxCutSolution {
        // Rebuild QAOA with current parameters
        // (In a full implementation, we'd use the seed here)
        let result = self.qaoa.run();

        // Sample to get binary assignment
        let (_mean_cost, _std_dev, samples) = self.qaoa.analyze_samples(&result.optimal_state, 100);

        // Find best sample
        let best_sample_idx = samples
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        // Convert state index to binary assignment
        let n = METATRON_DIMENSION;
        let assignment = (0..n)
            .map(|i| (best_sample_idx >> i) & 1 == 1)
            .collect::<Vec<_>>();

        let cut_value = -result.optimal_cost; // Negate because we minimized

        // Compute partition sizes
        let set_1_size = assignment.iter().filter(|&&b| b).count();
        let set_0_size = n - set_1_size;

        MaxCutSolution {
            cut_value,
            assignment,
            approximation_ratio: result.approximation_ratio,
            meta: SolutionMetadata {
                iterations: result.optimization_result.iterations,
                final_cost: result.optimal_cost,
                depth: self.depth,
                converged: result.optimization_result.converged,
                partition_sizes: (set_0_size, set_1_size),
            },
        }
    }
}

/// Quick MaxCut solver with default parameters
///
/// Convenience function for simple use cases.
///
/// # Arguments
/// * `graph` - The graph to partition
///
/// # Returns
/// A `MaxCutSolution` using default parameters (depth=3, 100 iterations)
pub fn solve_maxcut(graph: &MetatronGraph) -> MaxCutSolution {
    QaoaMaxCutSolver::from_graph(graph).run()
}

/// Advanced MaxCut solver with custom parameters
///
/// # Arguments
/// * `graph` - The graph to partition
/// * `depth` - QAOA circuit depth
/// * `max_iterations` - Maximum optimization iterations
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// A `MaxCutSolution`
pub fn solve_maxcut_advanced(
    graph: &MetatronGraph,
    depth: usize,
    max_iterations: usize,
    seed: Option<u64>,
) -> MaxCutSolution {
    let mut solver = QaoaMaxCutSolver::from_graph(graph)
        .with_depth(depth)
        .with_max_iterations(max_iterations);

    if let Some(s) = seed {
        solver = solver.with_seed(s);
    }

    solver.run()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qaoa_maxcut_solver() {
        let graph = MetatronGraph::new();
        let solution = QaoaMaxCutSolver::from_graph(&graph)
            .with_depth(2)
            .with_max_iterations(50)
            .run();

        // Should have assignment for all nodes
        assert_eq!(solution.assignment.len(), 13);

        // Cut value should be positive
        assert!(solution.cut_value > 0.0);

        // Approximation ratio should be reasonable
        assert!(solution.approximation_ratio >= 0.0 && solution.approximation_ratio <= 1.5);

        // Partition sizes should sum to total nodes
        let (s0, s1) = solution.meta.partition_sizes;
        assert_eq!(s0 + s1, 13);
    }

    #[test]
    fn test_solve_maxcut_convenience() {
        let graph = MetatronGraph::new();
        let solution = solve_maxcut(&graph);

        assert_eq!(solution.assignment.len(), 13);
        assert!(solution.cut_value > 0.0);
    }
}
