/*!
 * Metatron Router Module
 *
 * Central topological routing system for MEF-Core transformations through
 * the Metatron Cube's 13-node topology.
 */

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use uuid::Uuid;

use mef_core::{
    canonical_nodes, couple_cells, generate_c6_subgroup, generate_d6_subgroup,
    generate_s7_permutations, permutation_matrix, GabrielCell, MandorlaField, MetatronCube,
    MetatronCubeGraph, QLogicEngine, ResonanceTensorField, SpiralMemory,
};

/// MEF-Core operator types that can be routed through Metatron topology
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperatorType {
    /// Double impulse operator
    DK,
    /// Threshold sweep operator
    SW,
    /// Path invariance projection
    PI,
    /// Scale weight transfer
    WT,
}

impl OperatorType {
    /// Get the display name for this operator
    pub fn value(&self) -> &'static str {
        match self {
            OperatorType::DK => "DoubleKick",
            OperatorType::SW => "Sweep",
            OperatorType::PI => "PathInvariance",
            OperatorType::WT => "WeightTransfer",
        }
    }
}

impl std::fmt::Display for OperatorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value())
    }
}

/// Specification for a transformation route through Metatron topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteSpec {
    /// Unique identifier for this route
    pub route_id: String,
    /// S7 permutation defining the route (13 elements)
    pub permutation: Vec<usize>,
    /// Ordered list of operators to apply
    pub operator_sequence: Vec<OperatorType>,
    /// Symmetry group used (C6, D6, S7, Identity)
    pub symmetry_group: String,
    /// Route quality score based on resonance metrics
    pub score: f64,
    /// Additional routing metadata
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Convergence metrics for a single transformation step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceStep {
    /// Operator applied in this step
    pub operator: String,
    /// Norm of state change
    pub delta_norm: f64,
    /// Resonance value after this step
    pub resonance: f64,
    /// Entropy after this step
    pub entropy: f64,
}

/// Resonance metrics for a complete transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceMetrics {
    /// Input resonance before transformation
    pub input_resonance: f64,
    /// Output resonance after transformation
    pub output_resonance: f64,
    /// Coherence between input and output
    pub coherence: f64,
    /// Stability (inverse of change magnitude)
    pub stability: f64,
    /// Convergence (reduction in entropy)
    pub convergence: f64,
}

/// Result of applying a transformation through Metatron routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationResult {
    /// Original input state
    pub input_vector: Vec<f64>,
    /// Transformed output state
    pub output_vector: Vec<f64>,
    /// Route specification used
    pub route_spec: RouteSpec,
    /// Resonance calculations along the path
    pub resonance_metrics: ResonanceMetrics,
    /// Convergence metrics for each step
    pub convergence_data: Vec<ConvergenceStep>,
    /// Transformation timestamp
    pub timestamp: String,
}

/// Central topological routing system for MEF-Core transformations
///
/// Manages the complete 5040-path operator space through the Metatron Cube's
/// 13-node topology, providing deterministic routing for all transformations.
#[derive(Debug, Clone)]
pub struct MetatronRouter {
    /// Deterministic seed for reproducibility
    pub seed: String,
    /// Storage path for route cache
    pub storage_path: PathBuf,
    /// Core Metatron Cube structure
    pub metatron: MetatronCube,
    /// Metatron Cube graph
    pub graph: MetatronCubeGraph,
    /// QLOGIC engine for spectral analysis
    pub qlogic: QLogicEngine,
    /// Mandorla field for resonance
    pub mandorla: MandorlaField,
    /// Resonance tensor field
    pub resonance_field: ResonanceTensorField,
    /// Spiral memory
    pub spiral: SpiralMemory,
    /// Gabriel cells for feedback coupling
    pub gabriel_cells: Vec<GabrielCell>,
    /// All S7 permutations (5040 elements)
    pub s7_perms: Vec<Vec<usize>>,
    /// C6 subgroup permutations (6 elements)
    pub c6_perms: Vec<Vec<usize>>,
    /// D6 subgroup permutations (12 elements)
    pub d6_perms: Vec<Vec<usize>>,
    /// Route cache for performance
    pub route_cache: HashMap<String, RouteSpec>,
    /// Whether caching is enabled
    pub cache_enabled: bool,
}

impl MetatronRouter {
    /// Create a new Metatron Router with default parameters
    ///
    /// # Arguments
    ///
    /// * `storage_path` - Path for storing route cache and metadata
    pub fn new<P: AsRef<Path>>(storage_path: P) -> Self {
        Self::with_params("MEF_METATRON_42", true, true, storage_path)
    }

    /// Create a new Metatron Router with custom parameters
    ///
    /// # Arguments
    ///
    /// * `seed` - Deterministic seed for reproducibility
    /// * `full_edges` - Use full 78-edge connectivity if true
    /// * `cache_routes` - Cache computed routes for performance
    /// * `storage_path` - Path for storing route cache and metadata
    pub fn with_params<P: AsRef<Path>>(
        seed: &str,
        full_edges: bool,
        cache_routes: bool,
        storage_path: P,
    ) -> Self {
        let storage_path = storage_path.as_ref().to_path_buf();

        // Create storage directory if it doesn't exist
        if let Err(e) = std::fs::create_dir_all(&storage_path) {
            eprintln!("Warning: Failed to create storage directory: {}", e);
        }

        // Initialize core Metatron Cube
        let metatron = MetatronCube::new(None, None, None, full_edges);
        let graph = MetatronCubeGraph::new();

        // Initialize resonance and quantum components
        let qlogic = QLogicEngine::new(13);
        let mandorla = MandorlaField::new(0.985, 0.5, 0.5);
        let resonance_field = ResonanceTensorField::new((3, 3, 3), 1.0, 1.5, 0.0, 1e-3);
        let spiral = SpiralMemory::new(0.07);

        // Gabriel cells for feedback coupling
        let mut gabriel_cells = vec![
            GabrielCell::default(),
            GabrielCell::default(),
            GabrielCell::default(),
            GabrielCell::default(),
        ];
        couple_cells(&mut gabriel_cells, 0, 1);
        couple_cells(&mut gabriel_cells, 1, 2);
        couple_cells(&mut gabriel_cells, 2, 3);

        // Generate and cache all S7, C6, D6 permutations
        let s7_perms = generate_s7_permutations();
        let c6_perms = generate_c6_subgroup();
        let d6_perms = generate_d6_subgroup();

        let mut router = Self {
            seed: seed.to_string(),
            storage_path,
            metatron,
            graph,
            qlogic,
            mandorla,
            resonance_field,
            spiral,
            gabriel_cells,
            s7_perms,
            c6_perms,
            d6_perms,
            route_cache: HashMap::new(),
            cache_enabled: cache_routes,
        };

        // Load existing cache if available
        if cache_routes {
            router.load_route_cache();
        }

        router
    }

    /// Select the optimal transformation route for given input state
    ///
    /// This method evaluates multiple routes through the S7 permutation space
    /// and selects the one with highest resonance score.
    ///
    /// # Arguments
    ///
    /// * `input_state` - Input vector to transform
    /// * `target_properties` - Optional target properties to optimize for
    ///
    /// # Returns
    ///
    /// Optimal route specification
    pub fn select_optimal_route(
        &mut self,
        input_state: &[f64],
        target_properties: Option<&HashMap<String, String>>,
    ) -> RouteSpec {
        // Generate cache key
        let cache_key = self.generate_cache_key(input_state, target_properties);

        // Check cache first
        if self.cache_enabled {
            if let Some(route) = self.route_cache.get(&cache_key) {
                return route.clone();
            }
        }

        // Pad input to 13 dimensions for Metatron operations
        let padded_input = self.pad_to_metatron_dims(input_state);

        // Evaluate subset of S7 permutations (full 5040 is expensive)
        let candidate_perms = self.select_candidate_permutations(&padded_input, 10);

        let mut best_route: Option<RouteSpec> = None;
        let mut best_score = f64::NEG_INFINITY;

        for perm in candidate_perms {
            // Generate operator sequence for this permutation
            let op_sequence = self.generate_operator_sequence(&perm);

            // Evaluate route quality
            let score = self.evaluate_route(&padded_input, &perm, &op_sequence);

            if score > best_score {
                best_score = score;

                // Hash input state for metadata
                let mut hasher = Sha256::new();
                for &val in input_state {
                    hasher.update(val.to_le_bytes());
                }
                let input_hash = format!("{:x}", hasher.finalize())[..16].to_string();

                let mut metadata = HashMap::new();
                metadata.insert("input_hash".to_string(), input_hash);
                metadata.insert("timestamp".to_string(), chrono::Utc::now().to_rfc3339());

                best_route = Some(RouteSpec {
                    route_id: Uuid::new_v4().to_string(),
                    permutation: perm.clone(),
                    operator_sequence: op_sequence,
                    symmetry_group: self.identify_symmetry_group(&perm),
                    score,
                    metadata,
                });
            }
        }

        let route = best_route.unwrap_or_else(|| {
            // Fallback: identity permutation with default sequence (1-indexed)
            RouteSpec {
                route_id: Uuid::new_v4().to_string(),
                permutation: (1..=13).collect(),
                operator_sequence: vec![
                    OperatorType::DK,
                    OperatorType::SW,
                    OperatorType::PI,
                    OperatorType::WT,
                ],
                symmetry_group: "Identity".to_string(),
                score: 0.0,
                metadata: HashMap::new(),
            }
        });

        // Cache the result
        if self.cache_enabled {
            self.route_cache.insert(cache_key, route.clone());
            self.save_route_cache();
        }

        route
    }

    /// Apply transformation through Metatron topology using specified or optimal route
    ///
    /// # Arguments
    ///
    /// * `input_vector` - Input state vector
    /// * `route_spec` - Specific route to use (if None, selects optimal)
    ///
    /// # Returns
    ///
    /// Complete transformation result with metrics
    pub fn transform(
        &mut self,
        input_vector: &[f64],
        route_spec: Option<&RouteSpec>,
    ) -> TransformationResult {
        // Select route if not provided
        let route = if let Some(spec) = route_spec {
            spec.clone()
        } else {
            self.select_optimal_route(input_vector, None)
        };

        // Pad input for Metatron operations
        let mut current_state = self.pad_to_metatron_dims(input_vector);

        // Apply permutation to initial state
        let perm_matrix = permutation_matrix(&route.permutation, 13);
        current_state = self.matrix_vector_product(&perm_matrix, &current_state);

        // Track convergence at each step
        let mut convergence_data = Vec::new();

        // Apply operator sequence
        for operator in &route.operator_sequence {
            let prev_state = current_state.clone();

            // Apply operator through Metatron topology
            current_state = match operator {
                OperatorType::DK => self.apply_double_kick(&current_state),
                OperatorType::SW => self.apply_sweep(&current_state),
                OperatorType::PI => self.apply_path_invariance(&current_state),
                OperatorType::WT => self.apply_weight_transfer(&current_state),
            };

            // Calculate convergence metrics
            let delta_norm = Self::vector_distance(&current_state, &prev_state);
            let resonance = self.calculate_resonance(&current_state);
            let entropy = Self::calculate_entropy(&current_state);

            convergence_data.push(ConvergenceStep {
                operator: operator.value().to_string(),
                delta_norm,
                resonance,
                entropy,
            });
        }

        // Apply inverse permutation to return to original basis
        let perm_matrix_t = Self::transpose_matrix(&perm_matrix);
        current_state = self.matrix_vector_product(&perm_matrix_t, &current_state);

        // Truncate to original dimensions
        let output_vector: Vec<f64> = current_state
            .iter()
            .take(input_vector.len())
            .copied()
            .collect();

        // Calculate final resonance metrics
        let resonance_metrics = self.calculate_resonance_metrics(input_vector, &output_vector);

        TransformationResult {
            input_vector: input_vector.to_vec(),
            output_vector,
            route_spec: route,
            resonance_metrics,
            convergence_data,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Get current topology metrics and router status
    ///
    /// # Returns
    ///
    /// Dictionary of topology metrics
    pub fn get_topology_metrics(&self) -> HashMap<String, serde_json::Value> {
        let adjacency = self.graph.get_adjacency_matrix();
        let edge_count = adjacency.iter().filter(|&&x| x > 0.0).count() / 2;

        let mut metrics = HashMap::new();
        metrics.insert("nodes".to_string(), serde_json::Value::from(13));
        metrics.insert("edges".to_string(), serde_json::Value::from(edge_count));
        metrics.insert(
            "s7_permutations".to_string(),
            serde_json::Value::from(self.s7_perms.len()),
        );
        metrics.insert(
            "c6_subgroup".to_string(),
            serde_json::Value::from(self.c6_perms.len()),
        );
        metrics.insert(
            "d6_subgroup".to_string(),
            serde_json::Value::from(self.d6_perms.len()),
        );
        metrics.insert(
            "cached_routes".to_string(),
            serde_json::Value::from(self.route_cache.len()),
        );
        metrics.insert(
            "cache_enabled".to_string(),
            serde_json::Value::from(self.cache_enabled),
        );

        metrics
    }

    /// Export route visualization in JSON format
    ///
    /// # Arguments
    ///
    /// * `route_spec` - Route to export
    ///
    /// # Returns
    ///
    /// JSON string representation of the route
    pub fn export_route_json(&self, route_spec: &RouteSpec) -> String {
        serde_json::to_string_pretty(route_spec).unwrap_or_else(|_| "{}".to_string())
    }

    // ===================================================================
    // Private helper methods
    // ===================================================================

    /// Apply DoubleKick operator through Metatron topology
    ///
    /// Applies two orthogonal impulses to destabilize local minima
    /// while maintaining contractivity.
    fn apply_double_kick(&self, state: &Array1<f64>) -> Array1<f64> {
        let nodes = canonical_nodes();

        // Generate orthogonal impulses using Metatron geometry
        let mut hexagon_direction = Array1::<f64>::zeros(13);
        let mut cube_direction = Array1::<f64>::zeros(13);

        for (i, node) in nodes.iter().enumerate() {
            if node.node_type == "hexagon" {
                hexagon_direction[i] = node.coords.0; // x-component
            } else if node.node_type == "cube" {
                cube_direction[i] = node.coords.1; // y-component
            }
        }

        // Normalize directions
        let hex_norm = hexagon_direction.mapv(|x| x * x).sum().sqrt();
        if hex_norm > 0.0 {
            hexagon_direction /= hex_norm;
        }

        let cube_norm = cube_direction.mapv(|x| x * x).sum().sqrt();
        if cube_norm > 0.0 {
            cube_direction /= cube_norm;
        }

        // Apply double kick with small amplitudes
        let alpha1 = 0.05;
        let alpha2 = -0.03;

        state + alpha1 * &hexagon_direction + alpha2 * &cube_direction
    }

    /// Apply Sweep operator using resonance field dynamics
    ///
    /// Modulates state through adaptive thresholding based on local resonance patterns.
    fn apply_sweep(&mut self, state: &Array1<f64>) -> Array1<f64> {
        // Calculate mean resonance
        self.mandorla.clear_inputs();

        // Add state projections at different scales
        for scale in &[1.0, 0.5, 2.0] {
            let scaled: Vec<f64> = state.iter().take(5).map(|&x| x * scale).collect();
            self.mandorla.add_input(Array1::from(scaled));
        }

        let resonance = self.mandorla.calc_resonance();

        // Apply threshold gate with cosine schedule
        let tau = 0.5 + 0.3 * (resonance * std::f64::consts::PI).cos();
        let beta = 0.1;

        // Sigmoid gate function
        let mean_state = state.mean().unwrap_or(0.0);
        let gate_value = 1.0 / (1.0 + (-(mean_state - tau) / beta).exp());

        state * gate_value
    }

    /// Apply Path Invariance projection through canonical ordering
    ///
    /// Ensures invariance under path reordering by projecting to canonical representation.
    fn apply_path_invariance(&self, state: &Array1<f64>) -> Array1<f64> {
        // Apply multiple symmetry operations and average
        let mut projections = Vec::new();

        // Identity
        projections.push(state.clone());

        // Apply C6 rotations (first 3)
        for perm in self.c6_perms.iter().take(3) {
            let perm_matrix = permutation_matrix(perm, 13);
            let projected = self.matrix_vector_product(&perm_matrix, state);
            projections.push(projected);
        }

        // Average all projections for invariance
        let n = projections.len() as f64;
        let averaged = projections
            .iter()
            .fold(Array1::<f64>::zeros(13), |acc, p| acc + p)
            / n;

        // Sort to canonical form
        let mut indexed: Vec<(usize, f64)> = averaged
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val.abs()))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut canonical = Array1::<f64>::zeros(13);
        let mut sorted_vals: Vec<f64> = averaged.iter().map(|&x| x.abs()).collect();
        sorted_vals.sort_by(|a, b| b.partial_cmp(a).unwrap());

        for (i, (orig_idx, _)) in indexed.iter().enumerate() {
            canonical[*orig_idx] = sorted_vals[i] * averaged[*orig_idx].signum();
        }

        canonical
    }

    /// Apply Weight Transfer between topological scales
    ///
    /// Redistributes weights across micro/meso/macro scales defined by Metatron topology.
    fn apply_weight_transfer(&self, state: &Array1<f64>) -> Array1<f64> {
        // Define scale regions based on node types
        let center_idx = vec![0]; // Center node (macro)
        let hexagon_idx: Vec<usize> = (1..7).collect(); // Hexagon nodes (meso)
        let cube_idx: Vec<usize> = (7..13).collect(); // Cube nodes (micro)

        // Calculate current scale weights
        let macro_weight: f64 = center_idx.iter().map(|&i| state[i].abs()).sum();
        let meso_weight: f64 = hexagon_idx.iter().map(|&i| state[i].abs()).sum();
        let micro_weight: f64 = cube_idx.iter().map(|&i| state[i].abs()).sum();

        let total_weight = macro_weight + meso_weight + micro_weight;

        if total_weight > 1e-10 {
            let gamma = 0.1;

            // Transfer from micro to meso
            let transfer_micro_meso = gamma * micro_weight;
            // Transfer from meso to macro
            let transfer_meso_macro = gamma * meso_weight * 0.5;

            // Apply transfers
            let mut new_state = state.clone();

            // Adjust micro scale
            if micro_weight > 1e-10 {
                for &idx in &cube_idx {
                    new_state[idx] *= 1.0 - gamma;
                }
            }

            // Adjust meso scale
            if meso_weight > 1e-10 {
                let adjustment =
                    (1.0 - gamma * 0.5) + (transfer_micro_meso / (meso_weight + 1e-10));
                for &idx in &hexagon_idx {
                    new_state[idx] *= adjustment;
                }
            }

            // Adjust macro scale
            if macro_weight > 1e-10 {
                let adjustment = 1.0 + (transfer_meso_macro / (macro_weight + 1e-10));
                for &idx in &center_idx {
                    new_state[idx] *= adjustment;
                }
            } else {
                for &idx in &center_idx {
                    new_state[idx] += transfer_meso_macro;
                }
            }

            new_state
        } else {
            state.clone()
        }
    }

    /// Pad vector to 13 dimensions for Metatron operations
    fn pad_to_metatron_dims(&self, vector: &[f64]) -> Array1<f64> {
        if vector.len() >= 13 {
            Array1::from(vector[..13].to_vec())
        } else {
            let mut padded = vec![0.0; 13];
            padded[..vector.len()].copy_from_slice(vector);
            Array1::from(padded)
        }
    }

    /// Select promising permutation candidates using heuristics
    fn select_candidate_permutations(
        &self,
        input_state: &Array1<f64>,
        n_candidates: usize,
    ) -> Vec<Vec<usize>> {
        let mut candidates = Vec::new();

        // Always include identity (1-indexed: 1, 2, 3, ..., 13)
        candidates.push((1..=13).collect());

        // Include C6 subgroup (more structured)
        for perm in self.c6_perms.iter().take(3.min(n_candidates - 1)) {
            // Extend to full 13-element permutation
            let mut extended = perm.clone();
            extended.extend(8..=13);
            candidates.push(extended);
        }

        // Include D6 subgroup
        let remaining = n_candidates.saturating_sub(candidates.len());
        for perm in self.d6_perms.iter().take(3.min(remaining)) {
            let mut extended = perm.clone();
            extended.extend(8..=13);
            candidates.push(extended);
        }

        // Add deterministic samples from S7 if needed
        if candidates.len() < n_candidates {
            // Use deterministic sampling based on input hash
            let mut hasher = Sha256::new();
            for &val in input_state.iter() {
                hasher.update(val.to_le_bytes());
            }
            let hash = hasher.finalize();
            let seed = u32::from_be_bytes([hash[0], hash[1], hash[2], hash[3]]);

            use std::collections::HashSet;
            let mut selected = HashSet::new();
            let mut rng_state = seed as usize;

            let remaining = n_candidates - candidates.len();
            while selected.len() < remaining && selected.len() < self.s7_perms.len() {
                rng_state = (rng_state.wrapping_mul(1103515245).wrapping_add(12345)) % (1 << 31);
                let idx = rng_state % self.s7_perms.len();

                if selected.insert(idx) {
                    let perm = &self.s7_perms[idx];
                    let mut extended = perm.clone();
                    extended.extend(8..=13);
                    candidates.push(extended);
                }
            }
        }

        candidates.into_iter().take(n_candidates).collect()
    }

    /// Generate operator sequence based on permutation properties
    fn generate_operator_sequence(&self, permutation: &[usize]) -> Vec<OperatorType> {
        // Calculate permutation signature to determine sequence
        let signature = permutation.iter().sum::<usize>() % 4;

        let sequences = [
            vec![
                OperatorType::DK,
                OperatorType::SW,
                OperatorType::PI,
                OperatorType::WT,
            ],
            vec![
                OperatorType::SW,
                OperatorType::DK,
                OperatorType::WT,
                OperatorType::PI,
            ],
            vec![
                OperatorType::PI,
                OperatorType::WT,
                OperatorType::DK,
                OperatorType::SW,
            ],
            vec![
                OperatorType::WT,
                OperatorType::PI,
                OperatorType::SW,
                OperatorType::DK,
            ],
        ];

        sequences[signature].clone()
    }

    /// Evaluate quality score for a specific route
    fn evaluate_route(
        &mut self,
        input_state: &Array1<f64>,
        permutation: &[usize],
        operator_sequence: &[OperatorType],
    ) -> f64 {
        // Apply permutation
        let perm_matrix = permutation_matrix(permutation, 13);
        let mut current = self.matrix_vector_product(&perm_matrix, input_state);

        // Track metrics through transformation
        let mut total_convergence = 0.0;
        let mut total_resonance = 0.0;

        for operator in operator_sequence {
            let prev = current.clone();

            current = match operator {
                OperatorType::DK => self.apply_double_kick(&current),
                OperatorType::SW => self.apply_sweep(&current),
                OperatorType::PI => self.apply_path_invariance(&current),
                OperatorType::WT => self.apply_weight_transfer(&current),
            };

            // Measure convergence
            let delta = Self::vector_distance(&current, &prev);
            let convergence = 1.0 / (1.0 + delta);
            total_convergence += convergence;

            // Measure resonance
            let resonance = self.calculate_resonance(&current);
            total_resonance += resonance;
        }

        // Combine metrics for overall score
        let n = operator_sequence.len() as f64;
        (total_convergence / n) * (total_resonance / n)
    }

    /// Calculate resonance metric for a state vector
    fn calculate_resonance(&mut self, _state: &Array1<f64>) -> f64 {
        // Use QLOGIC spectral analysis
        let qlogic_result = self.qlogic.step(0.0);
        let spectrum = &qlogic_result.spectrum;

        if !spectrum.is_empty() {
            // Normalize spectrum
            let sum: f64 = spectrum.iter().sum();
            if sum > 1e-10 {
                let spectrum_norm: Vec<f64> = spectrum.iter().map(|&x| x / sum).collect();

                // Calculate entropy (lower = more coherent)
                let entropy: f64 = spectrum_norm
                    .iter()
                    .filter(|&&x| x > 1e-10)
                    .map(|&x| -x * x.ln())
                    .sum();

                // Map to resonance (high coherence = high resonance)
                let max_entropy = (spectrum.len() as f64).ln();
                if max_entropy > 0.0 {
                    let resonance = 1.0 - (entropy / max_entropy);
                    return resonance.clamp(0.0, 1.0);
                }
            }
        }

        0.0
    }

    /// Calculate entropy of state vector
    fn calculate_entropy(state: &Array1<f64>) -> f64 {
        // Normalize to probability distribution
        let sum: f64 = state.iter().map(|x| x.abs()).sum();

        if sum < 1e-10 {
            return 0.0;
        }

        let probs: Vec<f64> = state.iter().map(|x| x.abs() / sum).collect();

        // Calculate Shannon entropy
        let entropy: f64 = probs
            .iter()
            .filter(|&&x| x > 1e-10)
            .map(|&x| -x * x.ln())
            .sum();

        entropy
    }

    /// Calculate comprehensive resonance metrics for transformation
    fn calculate_resonance_metrics(
        &mut self,
        input_vector: &[f64],
        output_vector: &[f64],
    ) -> ResonanceMetrics {
        let padded_input = self.pad_to_metatron_dims(input_vector);
        let padded_output = self.pad_to_metatron_dims(output_vector);

        // Input/output resonance
        let input_resonance = self.calculate_resonance(&padded_input);
        let output_resonance = self.calculate_resonance(&padded_output);

        // Coherence using Mandorla field
        self.mandorla.clear_inputs();

        let input_5d: Vec<f64> = if input_vector.len() >= 5 {
            input_vector[..5].to_vec()
        } else {
            let mut padded = vec![0.0; 5];
            padded[..input_vector.len()].copy_from_slice(input_vector);
            padded
        };

        let output_5d: Vec<f64> = if output_vector.len() >= 5 {
            output_vector[..5].to_vec()
        } else {
            let mut padded = vec![0.0; 5];
            padded[..output_vector.len()].copy_from_slice(output_vector);
            padded
        };

        self.mandorla.add_input(Array1::from(input_5d));
        self.mandorla.add_input(Array1::from(output_5d));
        let coherence = self.mandorla.calc_resonance();

        // Stability (inverse of change magnitude)
        let delta = Self::vector_distance_slice(output_vector, input_vector);
        let stability = 1.0 / (1.0 + delta);

        // Convergence (reduction in entropy)
        let input_entropy = Self::calculate_entropy(&padded_input);
        let output_entropy = Self::calculate_entropy(&padded_output);
        let convergence = if input_entropy > 1e-10 {
            ((input_entropy - output_entropy) / input_entropy).max(0.0)
        } else {
            0.0
        };

        ResonanceMetrics {
            input_resonance,
            output_resonance,
            coherence,
            stability,
            convergence,
        }
    }

    /// Identify which symmetry group a permutation belongs to
    fn identify_symmetry_group(&self, permutation: &[usize]) -> String {
        // Check if it's identity (1-indexed: 1, 2, 3, ..., 13)
        let identity: Vec<usize> = (1..=13).collect();
        if permutation == identity.as_slice() {
            return "Identity".to_string();
        }

        // Check C6 membership
        for c6_perm in &self.c6_perms {
            let mut extended = c6_perm.clone();
            extended.extend(8..=13);
            if extended == *permutation {
                return "C6".to_string();
            }
        }

        // Check D6 membership
        for d6_perm in &self.d6_perms {
            let mut extended = d6_perm.clone();
            extended.extend(8..=13);
            if extended == *permutation {
                return "D6".to_string();
            }
        }

        // Otherwise it's from general S7
        "S7".to_string()
    }

    /// Generate cache key for route lookup
    fn generate_cache_key(
        &self,
        input_state: &[f64],
        target_properties: Option<&HashMap<String, String>>,
    ) -> String {
        // Hash input state
        let mut hasher = Sha256::new();
        for &val in input_state {
            hasher.update(val.to_le_bytes());
        }
        let state_hash = format!("{:x}", hasher.finalize())[..16].to_string();

        // Hash target properties if provided
        let prop_hash = if let Some(props) = target_properties {
            let prop_json = serde_json::to_string(props).unwrap_or_default();
            let mut hasher = Sha256::new();
            hasher.update(prop_json.as_bytes());
            format!("{:x}", hasher.finalize())[..16].to_string()
        } else {
            "none".to_string()
        };

        format!("{}_{}", state_hash, prop_hash)
    }

    /// Load route cache from disk if available
    fn load_route_cache(&mut self) {
        let cache_file = self.storage_path.join("route_cache.json");

        if cache_file.exists() {
            if let Ok(contents) = std::fs::read_to_string(&cache_file) {
                if let Ok(cache_data) =
                    serde_json::from_str::<HashMap<String, RouteSpec>>(&contents)
                {
                    self.route_cache = cache_data;
                }
            }
        }
    }

    /// Save route cache to disk
    fn save_route_cache(&self) {
        let cache_file = self.storage_path.join("route_cache.json");

        if let Ok(cache_json) = serde_json::to_string_pretty(&self.route_cache) {
            let _ = std::fs::write(&cache_file, cache_json);
        }
    }

    // Helper functions for vector/matrix operations

    fn matrix_vector_product(&self, matrix: &Array2<f64>, vector: &Array1<f64>) -> Array1<f64> {
        matrix.dot(vector)
    }

    fn transpose_matrix(matrix: &Array2<f64>) -> Array2<f64> {
        matrix.t().to_owned()
    }

    fn vector_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        (a - b).mapv(|x| x * x).sum().sqrt()
    }

    fn vector_distance_slice(a: &[f64], b: &[f64]) -> f64 {
        let min_len = a.len().min(b.len());
        let sum: f64 = (0..min_len)
            .map(|i| {
                let diff = a[i] - b[i];
                diff * diff
            })
            .sum();
        sum.sqrt()
    }
}

impl Default for MetatronRouter {
    fn default() -> Self {
        Self::new("/tmp/mef/metatron")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_default() {
        let router = MetatronRouter::default();
        assert_eq!(router.seed, "MEF_METATRON_42");
        assert!(router.cache_enabled);
        assert_eq!(router.gabriel_cells.len(), 4);
    }

    #[test]
    fn test_create_custom() {
        let router = MetatronRouter::with_params("custom_seed", false, false, "/tmp/test_router");
        assert_eq!(router.seed, "custom_seed");
        assert!(!router.cache_enabled);
        assert_eq!(router.s7_perms.len(), 5040);
        assert_eq!(router.c6_perms.len(), 6);
        assert_eq!(router.d6_perms.len(), 12);
    }

    #[test]
    fn test_pad_to_metatron_dims() {
        let router = MetatronRouter::default();

        let input_3d = vec![1.0, 2.0, 3.0];
        let padded = router.pad_to_metatron_dims(&input_3d);
        assert_eq!(padded.len(), 13);
        assert_eq!(padded[0], 1.0);
        assert_eq!(padded[1], 2.0);
        assert_eq!(padded[2], 3.0);
        assert_eq!(padded[12], 0.0);

        let input_20d = vec![1.0; 20];
        let padded = router.pad_to_metatron_dims(&input_20d);
        assert_eq!(padded.len(), 13);
        assert!(padded.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_operator_type_display() {
        assert_eq!(OperatorType::DK.value(), "DoubleKick");
        assert_eq!(OperatorType::SW.value(), "Sweep");
        assert_eq!(OperatorType::PI.value(), "PathInvariance");
        assert_eq!(OperatorType::WT.value(), "WeightTransfer");
    }

    #[test]
    fn test_select_optimal_route() {
        let mut router = MetatronRouter::default();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let route = router.select_optimal_route(&input, None);

        assert!(!route.route_id.is_empty());
        assert_eq!(route.permutation.len(), 13);
        assert!(!route.operator_sequence.is_empty());
        assert!(!route.symmetry_group.is_empty());
        assert!(route.score >= 0.0);
    }

    #[test]
    fn test_transform_basic() {
        let mut router = MetatronRouter::default();
        let input = vec![1.0, 0.5, -0.5];

        let result = router.transform(&input, None);

        assert_eq!(result.input_vector.len(), 3);
        assert_eq!(result.output_vector.len(), 3);
        assert!(!result.convergence_data.is_empty());
        assert!(!result.timestamp.is_empty());
    }

    #[test]
    fn test_apply_double_kick() {
        let router = MetatronRouter::default();
        let state = Array1::from(vec![1.0; 13]);

        let kicked = router.apply_double_kick(&state);

        assert_eq!(kicked.len(), 13);
        // Should have applied some perturbation
        assert_ne!(kicked.sum(), state.sum());
    }

    #[test]
    fn test_apply_sweep() {
        let mut router = MetatronRouter::default();
        let state = Array1::from(vec![0.5; 13]);

        let swept = router.apply_sweep(&state);

        assert_eq!(swept.len(), 13);
        // Sweep applies scaling
        assert!(swept.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_apply_path_invariance() {
        let router = MetatronRouter::default();
        let state = Array1::from(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
        ]);

        let invariant = router.apply_path_invariance(&state);

        assert_eq!(invariant.len(), 13);
        // Should produce a canonical ordering
        assert!(invariant.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_apply_weight_transfer() {
        let router = MetatronRouter::default();
        let state = Array1::from(vec![1.0; 13]);

        let transferred = router.apply_weight_transfer(&state);

        assert_eq!(transferred.len(), 13);
        // Weight should be conserved (approximately)
        let original_sum: f64 = state.iter().sum();
        let new_sum: f64 = transferred.iter().sum();
        assert!((new_sum - original_sum).abs() < 0.1);
    }

    #[test]
    fn test_calculate_entropy() {
        let state = Array1::from(vec![1.0; 13]);
        let entropy = MetatronRouter::calculate_entropy(&state);

        // Uniform distribution has maximum entropy
        assert!(entropy > 2.0);

        let concentrated = Array1::from(vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        let low_entropy = MetatronRouter::calculate_entropy(&concentrated);

        // Concentrated distribution has lower entropy
        assert!(low_entropy < entropy);
    }

    #[test]
    fn test_identify_symmetry_group() {
        let router = MetatronRouter::default();

        let identity: Vec<usize> = (1..=13).collect();
        assert_eq!(router.identify_symmetry_group(&identity), "Identity");

        // Test with an extended C6 permutation (use index 1, not 0 which is identity)
        if router.c6_perms.len() > 1 {
            // The c6_perms are 7 elements, not 13 yet
            assert_eq!(router.c6_perms[1].len(), 7);

            // When extended to 13, they should match
            let mut c6_extended = router.c6_perms[1].clone();
            c6_extended.extend(8..=13);
            assert_eq!(c6_extended.len(), 13);
            // This extended permutation should be recognized as C6
            assert_eq!(router.identify_symmetry_group(&c6_extended), "C6");
        }
    }

    #[test]
    fn test_generate_operator_sequence() {
        let router = MetatronRouter::default();

        let identity: Vec<usize> = (0..13).collect();
        let sequence = router.generate_operator_sequence(&identity);

        assert_eq!(sequence.len(), 4);
        assert!(sequence.contains(&OperatorType::DK));
        assert!(sequence.contains(&OperatorType::SW));
        assert!(sequence.contains(&OperatorType::PI));
        assert!(sequence.contains(&OperatorType::WT));
    }

    #[test]
    fn test_get_topology_metrics() {
        let router = MetatronRouter::default();
        let metrics = router.get_topology_metrics();

        assert_eq!(metrics.get("nodes").unwrap().as_i64().unwrap(), 13);
        assert_eq!(
            metrics.get("s7_permutations").unwrap().as_i64().unwrap(),
            5040
        );
        assert_eq!(metrics.get("c6_subgroup").unwrap().as_i64().unwrap(), 6);
        assert_eq!(metrics.get("d6_subgroup").unwrap().as_i64().unwrap(), 12);
        assert_eq!(
            metrics.get("cache_enabled").unwrap().as_bool().unwrap(),
            true
        );
    }

    #[test]
    fn test_export_route_json() {
        let router = MetatronRouter::default();
        let route = RouteSpec {
            route_id: "test-route".to_string(),
            permutation: (1..=13).collect(),
            operator_sequence: vec![OperatorType::DK, OperatorType::SW],
            symmetry_group: "Identity".to_string(),
            score: 0.85,
            metadata: HashMap::new(),
        };

        let json = router.export_route_json(&route);
        assert!(json.contains("test-route"));
        assert!(json.contains("Identity"));
    }

    #[test]
    fn test_transform_with_specific_route() {
        let mut router = MetatronRouter::default();
        let input = vec![1.0, 2.0, 3.0];

        let route = RouteSpec {
            route_id: "specific-route".to_string(),
            permutation: (1..=13).collect(),
            operator_sequence: vec![OperatorType::DK, OperatorType::PI],
            symmetry_group: "Identity".to_string(),
            score: 1.0,
            metadata: HashMap::new(),
        };

        let result = router.transform(&input, Some(&route));

        assert_eq!(result.route_spec.route_id, "specific-route");
        assert_eq!(result.convergence_data.len(), 2); // Two operators
        assert_eq!(result.convergence_data[0].operator, "DoubleKick");
        assert_eq!(result.convergence_data[1].operator, "PathInvariance");
    }

    #[test]
    fn test_resonance_metrics() {
        let mut router = MetatronRouter::default();
        let input = vec![1.0, 0.5, 0.2];
        let output = vec![0.9, 0.6, 0.3];

        let metrics = router.calculate_resonance_metrics(&input, &output);

        assert!(metrics.input_resonance >= 0.0 && metrics.input_resonance <= 1.0);
        assert!(metrics.output_resonance >= 0.0 && metrics.output_resonance <= 1.0);
        assert!(metrics.coherence >= 0.0);
        assert!(metrics.stability >= 0.0 && metrics.stability <= 1.0);
        assert!(metrics.convergence >= 0.0 && metrics.convergence <= 1.0);
    }

    #[test]
    fn test_cache_functionality() {
        let temp_dir = std::env::temp_dir().join("mef_router_test");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut router = MetatronRouter::with_params("test_seed", true, true, &temp_dir);

        let input = vec![1.0, 2.0, 3.0];
        let route1 = router.select_optimal_route(&input, None);

        // Second call should return cached route
        let route2 = router.select_optimal_route(&input, None);

        assert_eq!(route1.route_id, route2.route_id);
        assert_eq!(route1.score, route2.score);

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_select_candidate_permutations() {
        let router = MetatronRouter::default();
        let input = Array1::from(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
        ]);

        let candidates = router.select_candidate_permutations(&input, 10);

        assert!(candidates.len() <= 10);
        assert!(candidates.len() > 0);

        // All candidates should be 13 elements
        for perm in &candidates {
            assert_eq!(perm.len(), 13);
        }

        // First candidate should be identity (1-indexed)
        assert_eq!(candidates[0], (1..=13).collect::<Vec<usize>>());
    }
}
