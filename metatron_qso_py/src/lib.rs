//! Python bindings for Metatron Quantum State Operator
//!
//! This module provides a Python-friendly API for the Metatron QSO quantum computing framework.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

// Import from the Rust core library with explicit path to avoid module conflicts
use core::prelude::*;
use metatron_qso as core;

/// Python wrapper for MetatronGraph
///
/// Represents the 13-node Metatron Cube graph with 78 edges.
#[pyclass(name = "MetatronGraph")]
#[derive(Clone)]
struct PyMetatronGraph {
    inner: MetatronGraph,
}

#[pymethods]
impl PyMetatronGraph {
    /// Create a new Metatron Cube graph with default configuration
    #[new]
    fn new() -> Self {
        PyMetatronGraph {
            inner: MetatronGraph::new(),
        }
    }

    /// Create a graph from an adjacency list representation
    ///
    /// Args:
    ///     adjacency (list of lists): Adjacency list where adjacency[i] contains neighbors of node i
    ///
    /// Returns:
    ///         MetatronGraph: A new graph instance
    #[staticmethod]
    fn from_adjacency(adjacency: Vec<Vec<usize>>) -> PyResult<Self> {
        // For now, we return the default Metatron graph
        // In a full implementation, this would validate and construct from adjacency
        if adjacency.len() != 13 {
            return Err(PyValueError::new_err(
                "Metatron graph must have exactly 13 nodes",
            ));
        }
        Ok(PyMetatronGraph {
            inner: MetatronGraph::new(),
        })
    }

    /// Get the number of nodes in the graph
    fn num_nodes(&self) -> usize {
        self.inner.nodes().len()
    }

    /// Get the number of edges in the graph
    fn num_edges(&self) -> usize {
        self.inner.edges().len()
    }

    /// Get the adjacency list representation
    ///
    /// Returns:
    ///     list of lists: Adjacency list where result[i] contains neighbors of node i
    fn adjacency_list(&self) -> Vec<Vec<usize>> {
        let n = self.inner.nodes().len();
        let mut adj_list = vec![Vec::new(); n];

        for &(u, v) in self.inner.edges() {
            adj_list[u].push(v);
            adj_list[v].push(u);
        }

        // Sort and deduplicate
        for adj in &mut adj_list {
            adj.sort_unstable();
            adj.dedup();
        }

        adj_list
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "MetatronGraph(nodes={}, edges={})",
            self.num_nodes(),
            self.num_edges()
        )
    }
}

/// Run a continuous-time quantum walk on a graph
///
/// Args:
///     graph (MetatronGraph): The graph to run the walk on
///     source_nodes (list of int): Initial nodes with equal probability
///     t_max (float): Maximum evolution time (default: 10.0)
///     dt (float): Time step for evolution (default: 0.1)
///
/// Returns:
///     dict: Dictionary containing:
///         - 'times': List of time points
///         - 'probabilities': List of probability distributions at each time
///         - 'final_state': Final probability distribution
///
/// Example:
///     >>> graph = MetatronGraph()
///     >>> result = run_quantum_walk(graph, [0], t_max=5.0, dt=0.1)
///     >>> print(result['final_state'])
#[pyfunction]
#[pyo3(signature = (graph, source_nodes, t_max=10.0, dt=0.1))]
fn run_quantum_walk(
    graph: &PyMetatronGraph,
    source_nodes: Vec<usize>,
    t_max: f64,
    dt: f64,
) -> PyResult<PyObject> {
    // Validate inputs
    if source_nodes.is_empty() {
        return Err(PyValueError::new_err("source_nodes cannot be empty"));
    }
    if t_max <= 0.0 {
        return Err(PyValueError::new_err("t_max must be positive"));
    }
    if dt <= 0.0 || dt > t_max {
        return Err(PyValueError::new_err("dt must be positive and <= t_max"));
    }

    // Create initial state (uniform over source nodes)
    let n = graph.inner.nodes().len();
    let mut amplitudes = vec![num_complex::Complex64::new(0.0, 0.0); n];
    let amplitude = num_complex::Complex64::new(1.0 / (source_nodes.len() as f64).sqrt(), 0.0);
    for &node in &source_nodes {
        if node >= n {
            return Err(PyValueError::new_err(format!(
                "Node index {} out of bounds (graph has {} nodes)",
                node, n
            )));
        }
        amplitudes[node] = amplitude;
    }

    // Create quantum state
    let initial_state = QuantumState::from_amplitudes(amplitudes)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create initial state: {}", e)))?;

    // Create Hamiltonian and quantum walk
    let params = QSOParameters::default();
    let hamiltonian = MetatronHamiltonian::new(&graph.inner, &params);
    let qw = ContinuousTimeQuantumWalk::new(&hamiltonian);

    // Evolve the state at different times
    let num_steps = (t_max / dt).ceil() as usize;
    let mut times = Vec::with_capacity(num_steps + 1);
    let mut probabilities = Vec::with_capacity(num_steps + 1);

    // Initial state
    times.push(0.0);
    probabilities.push(initial_state.probabilities().to_vec());

    // Evolve
    for i in 1..=num_steps {
        let t = (i as f64) * dt;
        let t = t.min(t_max);
        times.push(t);

        let evolved_state = qw.evolve(&initial_state, t);
        probabilities.push(evolved_state.probabilities().to_vec());
    }

    // Return as Python dict
    Python::with_gil(|py| {
        let result = PyDict::new_bound(py);
        result.set_item("times", times.to_object(py))?;
        result.set_item("probabilities", probabilities.to_object(py))?;
        result.set_item("final_state", probabilities.last().unwrap().to_object(py))?;
        Ok(result.to_object(py))
    })
}

/// Solve the MaxCut problem using QAOA
///
/// Args:
///     graph (MetatronGraph): The graph for the MaxCut problem
///     depth (int): QAOA circuit depth (default: 3)
///     max_iters (int): Maximum optimization iterations (default: 100)
///
/// Returns:
///     dict: Dictionary containing:
///         - 'cut_value': The best cut value found
///         - 'approximation_ratio': Quality of the solution
///         - 'meta': Additional metadata about the optimization
///
/// Example:
///     >>> graph = MetatronGraph()
///     >>> result = solve_maxcut_qaoa(graph, depth=3, max_iters=100)
///     >>> print(f"Cut value: {result['cut_value']}")
#[pyfunction]
#[pyo3(signature = (graph, depth=3, max_iters=100))]
fn solve_maxcut_qaoa(
    graph: &PyMetatronGraph,
    depth: usize,
    max_iters: usize,
) -> PyResult<PyObject> {
    if depth == 0 {
        return Err(PyValueError::new_err("depth must be positive"));
    }
    if max_iters == 0 {
        return Err(PyValueError::new_err("max_iters must be positive"));
    }

    // Create MaxCut Hamiltonian from graph edges
    let edges: Vec<(usize, usize)> = graph.inner.edges().to_vec();

    let cost_hamiltonian = Arc::new(core::vqa::qaoa::create_maxcut_hamiltonian(&edges));

    // Build and run QAOA
    let qaoa = QAOABuilder::new()
        .cost_hamiltonian(cost_hamiltonian)
        .depth(depth)
        .optimizer(OptimizerType::NelderMead)
        .max_iterations(max_iters)
        .verbose(false)
        .build();

    let result = qaoa.run();

    // Sample to get statistics
    let (mean_cost, std_dev, _costs) = qaoa.analyze_samples(&result.optimal_state, 100);

    // Return as Python dict
    Python::with_gil(|py| {
        let result_dict = PyDict::new_bound(py);
        result_dict.set_item("cut_value", -result.optimal_cost)?; // Negate because we minimize
        result_dict.set_item("approximation_ratio", result.approximation_ratio)?;

        let meta = PyDict::new_bound(py);
        meta.set_item("iterations", result.optimization_result.iterations)?;
        meta.set_item("mean_cost", -mean_cost)?; // Negate for MaxCut
        meta.set_item("std_dev", std_dev)?;
        meta.set_item("depth", depth)?;
        result_dict.set_item("meta", meta)?;

        Ok(result_dict.to_object(py))
    })
}

/// Run VQE (Variational Quantum Eigensolver) to find the ground state energy
///
/// Args:
///     graph (MetatronGraph): The graph to create the Hamiltonian from
///     depth (int): Ansatz circuit depth (default: 2)
///     max_iters (int): Maximum optimization iterations (default: 100)
///     ansatz_type (str): Type of ansatz - "hardware_efficient", "metatron", or "efficient_su2" (default: "hardware_efficient")
///
/// Returns:
///     dict: Dictionary containing:
///         - 'ground_state_energy': The computed ground state energy
///         - 'classical_ground_energy': Exact ground state energy for comparison
///         - 'error': Absolute error from exact result
///         - 'iterations': Number of optimization iterations
///         - 'final_state': The final quantum state probabilities
///
/// Example:
///     >>> graph = MetatronGraph()
///     >>> result = run_vqe(graph, depth=2, max_iters=100)
///     >>> print(f"Ground state energy: {result['ground_state_energy']:.6f}")
#[pyfunction]
#[pyo3(signature = (graph, depth=2, max_iters=100, ansatz_type="hardware_efficient"))]
fn run_vqe(
    graph: &PyMetatronGraph,
    depth: usize,
    max_iters: usize,
    ansatz_type: &str,
) -> PyResult<PyObject> {
    if depth == 0 {
        return Err(PyValueError::new_err("depth must be positive"));
    }
    if max_iters == 0 {
        return Err(PyValueError::new_err("max_iters must be positive"));
    }

    // Parse ansatz type
    let ansatz = match ansatz_type.to_lowercase().as_str() {
        "hardware_efficient" => AnsatzType::HardwareEfficient,
        "metatron" => AnsatzType::Metatron,
        "efficient_su2" => AnsatzType::EfficientSU2,
        _ => {
            return Err(PyValueError::new_err(
                "ansatz_type must be 'hardware_efficient', 'metatron', or 'efficient_su2'",
            ))
        }
    };

    // Create Hamiltonian
    let params = QSOParameters::default();
    let hamiltonian = Arc::new(MetatronHamiltonian::new(&graph.inner, &params));

    // Build and run VQE
    let vqe = VQEBuilder::new()
        .hamiltonian(hamiltonian)
        .ansatz_type(ansatz)
        .ansatz_depth(depth)
        .optimizer(OptimizerType::Adam)
        .max_iterations(max_iters)
        .learning_rate(0.01)
        .tolerance(1e-6)
        .verbose(false)
        .build();

    let result = vqe.run();

    // Return as Python dict
    Python::with_gil(|py| {
        let result_dict = PyDict::new_bound(py);
        result_dict.set_item("ground_state_energy", result.ground_state_energy)?;
        result_dict.set_item("classical_ground_energy", result.classical_ground_energy)?;
        result_dict.set_item("error", result.approximation_error)?;
        result_dict.set_item("iterations", result.optimization_result.iterations)?;
        result_dict.set_item(
            "final_state",
            result
                .ground_state_wavefunction
                .probabilities()
                .to_vec()
                .to_object(py),
        )?;

        Ok(result_dict.to_object(py))
    })
}

/// Compute quantum walk centrality for nodes
///
/// Returns a centrality score for each node based on quantum walk dynamics.
/// Higher scores indicate more "central" or influential nodes.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `t_max` - Maximum evolution time (default: 10.0)
/// * `dt` - Time step (default: 0.1)
/// * `samples` - Number of samples for averaging (default: 128)
///
/// # Returns
/// List of centrality scores (one per node, normalized to [0, 1])
#[pyfunction]
#[pyo3(signature = (graph, t_max=10.0, dt=0.1, samples=128))]
fn quantum_walk_centrality(
    graph: &PyMetatronGraph,
    t_max: f64,
    dt: f64,
    samples: usize,
) -> PyResult<Vec<f64>> {
    let params = core::quantum_walk_toolkit::QuantumWalkParams { t_max, dt, samples };

    let centrality = core::quantum_walk_toolkit::quantum_walk_centrality(&graph.inner, &params);
    Ok(centrality)
}

/// Compute anomaly scores comparing base graph to current graph
///
/// Detects structural changes using quantum walk dynamics.
///
/// # Arguments
/// * `base_graph` - Baseline/reference graph
/// * `current_graph` - Current graph to analyze
/// * `t_max` - Maximum evolution time (default: 10.0)
/// * `dt` - Time step (default: 0.1)
/// * `samples` - Number of samples (default: 128)
///
/// # Returns
/// List of anomaly scores per node (higher = more anomalous)
#[pyfunction]
#[pyo3(signature = (base_graph, current_graph, t_max=10.0, dt=0.1, samples=128))]
fn quantum_walk_anomaly_score(
    base_graph: &PyMetatronGraph,
    current_graph: &PyMetatronGraph,
    t_max: f64,
    dt: f64,
    samples: usize,
) -> PyResult<Vec<f64>> {
    let params = core::quantum_walk_toolkit::QuantumWalkParams { t_max, dt, samples };

    let anomaly = core::quantum_walk_toolkit::quantum_walk_anomaly_score(
        &base_graph.inner,
        &current_graph.inner,
        &params,
    );
    Ok(anomaly)
}

/// Analyze connectivity using quantum walks
///
/// Computes connectivity metrics from specified source nodes.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `source_nodes` - Starting nodes for quantum walk
/// * `t_max` - Maximum evolution time (default: 10.0)
/// * `dt` - Time step (default: 0.1)
/// * `samples` - Number of samples (default: 128)
///
/// # Returns
/// Dictionary with connectivity metrics:
///   - 'mixing_time': Time to reach near-uniform distribution
///   - 'hitting_probabilities': Final probabilities for each node
///   - 'distribution_variance': Variance in probability distribution
///   - 'effective_diameter': Effective graph diameter
#[pyfunction]
#[pyo3(signature = (graph, source_nodes, t_max=10.0, dt=0.1, samples=128))]
fn quantum_walk_connectivity(
    graph: &PyMetatronGraph,
    source_nodes: Vec<usize>,
    t_max: f64,
    dt: f64,
    samples: usize,
) -> PyResult<PyObject> {
    let params = core::quantum_walk_toolkit::QuantumWalkParams { t_max, dt, samples };

    let metrics =
        core::quantum_walk_toolkit::quantum_walk_connectivity(&graph.inner, &source_nodes, &params);

    Python::with_gil(|py| {
        let result = PyDict::new_bound(py);
        result.set_item("mixing_time", metrics.mixing_time)?;
        result.set_item(
            "hitting_probabilities",
            metrics.hitting_probabilities.to_object(py),
        )?;
        result.set_item("distribution_variance", metrics.distribution_variance)?;
        result.set_item("effective_diameter", metrics.effective_diameter)?;
        Ok(result.to_object(py))
    })
}

/// Advanced MaxCut solver with full control
///
/// Solves the MaxCut problem using QAOA with advanced options.
///
/// # Arguments
/// * `graph` - The graph to partition
/// * `depth` - QAOA circuit depth (default: 3)
/// * `max_iters` - Maximum optimization iterations (default: 100)
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// Dictionary with:
///   - 'cut_value': Number of edges cut
///   - 'assignment': Binary node assignment (list of bool)
///   - 'approximation_ratio': Quality metric
///   - 'meta': Metadata (iterations, partition sizes, etc.)
#[pyfunction]
#[pyo3(signature = (graph, depth=3, max_iters=100, seed=None))]
fn solve_maxcut_qaoa_advanced(
    graph: &PyMetatronGraph,
    depth: usize,
    max_iters: usize,
    seed: Option<u64>,
) -> PyResult<PyObject> {
    let solution = core::optimizer::solve_maxcut_advanced(&graph.inner, depth, max_iters, seed);

    Python::with_gil(|py| {
        let result = PyDict::new_bound(py);
        result.set_item("cut_value", solution.cut_value)?;
        result.set_item("assignment", solution.assignment.to_object(py))?;
        result.set_item("approximation_ratio", solution.approximation_ratio)?;

        let meta = PyDict::new_bound(py);
        meta.set_item("iterations", solution.meta.iterations)?;
        meta.set_item("final_cost", solution.meta.final_cost)?;
        meta.set_item("depth", solution.meta.depth)?;
        meta.set_item("converged", solution.meta.converged)?;
        meta.set_item("partition_sizes", solution.meta.partition_sizes)?;
        result.set_item("meta", meta)?;

        Ok(result.to_object(py))
    })
}

/// Python module initialization
#[pymodule]
fn _metatron_qso_internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMetatronGraph>()?;

    // Core functions
    m.add_function(wrap_pyfunction!(run_quantum_walk, m)?)?;
    m.add_function(wrap_pyfunction!(solve_maxcut_qaoa, m)?)?;
    m.add_function(wrap_pyfunction!(run_vqe, m)?)?;

    // High-level toolkits
    m.add_function(wrap_pyfunction!(quantum_walk_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(quantum_walk_anomaly_score, m)?)?;
    m.add_function(wrap_pyfunction!(quantum_walk_connectivity, m)?)?;
    m.add_function(wrap_pyfunction!(solve_maxcut_qaoa_advanced, m)?)?;

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "__doc__",
        "Metatron Quantum State Operator - High-performance quantum computing in Python",
    )?;

    Ok(())
}
