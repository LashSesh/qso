/*!
 * Cube Module - High-level Metatron Cube API
 *
 * This module provides a high-level API wrapper around the Metatron Cube graph,
 * symmetry operators, and serialization. The `MetatronCube` class exposes methods
 * to query nodes and edges, list elements by type, apply permutations, enumerate
 * symmetry groups, and export/validate configurations.
 *
 * This class is designed as a convenience layer over the underlying data structures
 * (`MetatronCubeGraph`, `Node`, etc.) and can be used directly in applications or
 * as a basis for a REST API.
 */

use anyhow::{anyhow, Result};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::geometry::{
    canonical_edges, canonical_nodes, complete_canonical_edges, find_node, Node,
};
use crate::graph::MetatronCubeGraph;
use crate::quantum::{QuantumOperator, QuantumState};
use crate::symmetries::{
    generate_alternating_group, generate_c6_subgroup, generate_d6_subgroup,
    generate_s7_permutations, generate_symmetric_group, permutation_matrix,
};

/// High-level API class representing a Metatron Cube instance
#[derive(Debug, Clone)]
pub struct MetatronCube {
    /// The nodes in the cube
    pub nodes: Vec<Node>,
    /// The edges in the cube (as pairs of 1-based indices)
    pub edges: Vec<(usize, usize)>,
    /// The underlying graph structure
    pub graph: MetatronCubeGraph,
    /// Operator registry: maps string IDs to permutation tuples (length 13)
    pub operators: HashMap<String, Vec<usize>>,
    /// Node membership mapping (node index -> list of solid names)
    pub node_membership: HashMap<usize, Vec<String>>,
}

/// Serializable node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: usize,
    pub label: String,
    #[serde(rename = "type")]
    pub node_type: String,
    pub coordinates: Vec<f64>,
    pub membership: Vec<String>,
}

/// Serializable edge information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeInfo {
    pub id: usize,
    pub from: usize,
    pub to: usize,
    pub label: String,
    #[serde(rename = "type")]
    pub edge_type: String,
    pub solids: Vec<String>,
}

/// Serializable operator information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorInfo {
    pub id: String,
    pub group: Option<String>,
    pub permutation: Vec<usize>,
    pub matrix: Vec<Vec<f64>>,
}

/// Heuristic node subsets for embedded Platonic solids
fn solid_sets() -> HashMap<String, Vec<Vec<usize>>> {
    let mut sets = HashMap::new();

    // Tetrahedra: two sample sets of four nodes
    sets.insert(
        "tetrahedron".to_string(),
        vec![vec![2, 4, 6, 8], vec![3, 5, 7, 9]],
    );

    // Cube: the six cube corner nodes
    sets.insert("cube".to_string(), vec![vec![8, 9, 10, 11, 12, 13]]);

    // Octahedron: the six hexagon nodes
    sets.insert("octahedron".to_string(), vec![vec![2, 3, 4, 5, 6, 7]]);

    // Icosahedron: the 12 non-centre nodes
    sets.insert("icosahedron".to_string(), vec![(2..=13).collect()]);

    // Dodecahedron: centre and 11 outer nodes (approximation)
    sets.insert(
        "dodecahedron".to_string(),
        vec![vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]],
    );

    sets
}

impl MetatronCube {
    /// Create a new Metatron Cube instance
    ///
    /// # Arguments
    ///
    /// * `nodes` - Optional custom nodes. If None, uses canonical nodes.
    /// * `edges` - Optional edges. If None and full_edges is false, uses canonical edges.
    /// * `operators` - Optional predefined operators mapping IDs to 13-length permutations.
    /// * `full_edges` - If true, use complete edge list (78 edges).
    ///
    /// # Returns
    ///
    /// A new MetatronCube instance
    pub fn new(
        nodes: Option<Vec<Node>>,
        edges: Option<Vec<(usize, usize)>>,
        operators: Option<HashMap<String, Vec<usize>>>,
        full_edges: bool,
    ) -> Self {
        let nodes = nodes.unwrap_or_else(canonical_nodes);

        let edges = if let Some(e) = edges {
            e
        } else if full_edges {
            complete_canonical_edges()
        } else {
            canonical_edges()
        };

        let graph =
            MetatronCubeGraph::with_nodes_and_edges(Some(nodes.clone()), Some(edges.clone()), None);

        let mut operators = operators.unwrap_or_default();

        // Prepopulate with basic groups if empty
        if operators.is_empty() {
            Self::register_basic_groups(&mut operators);
        }

        let node_membership = Self::init_solid_membership(&nodes);

        Self {
            nodes,
            edges,
            graph,
            operators,
            node_membership,
        }
    }
}

impl Default for MetatronCube {
    fn default() -> Self {
        Self::new(None, None, None, false)
    }
}

impl MetatronCube {
    /// Register a set of basic symmetry operators into the registry
    fn register_basic_groups(operators: &mut HashMap<String, Vec<usize>>) {
        // C6 rotations: six elements
        for (k, perm7) in generate_c6_subgroup().iter().enumerate() {
            let mut sigma = perm7.clone();
            sigma.extend(8..=13);
            operators.insert(format!("C6_rot_{}", k * 60), sigma);
        }

        // D6 reflections: indices 6..12 correspond to the 6 reflections
        let d6 = generate_d6_subgroup();
        for (idx, perm7) in d6[6..12].iter().enumerate() {
            let mut sigma = perm7.clone();
            sigma.extend(8..=13);
            operators.insert(format!("D6_ref_H{}", idx + 2), sigma);
        }
    }

    /// Initialize node-to-solid membership mapping
    fn init_solid_membership(nodes: &[Node]) -> HashMap<usize, Vec<String>> {
        let mut membership: HashMap<usize, Vec<String>> = HashMap::new();

        // Basic memberships by node type
        for node in nodes {
            let mut groups = Vec::new();
            match node.node_type.as_str() {
                "center" => groups.push("center".to_string()),
                "hexagon" => groups.push("hexagon".to_string()),
                "cube" => groups.push("cube".to_string()),
                _ => {}
            }
            membership.insert(node.index, groups);
        }

        // Assign platonic solids
        for (solid_name, subsets) in solid_sets() {
            for subset in subsets {
                for idx in subset {
                    membership.entry(idx).or_default().push(solid_name.clone());
                }
            }
        }

        // Deduplicate and sort memberships
        for groups in membership.values_mut() {
            groups.sort();
            groups.dedup();
        }

        membership
    }

    // ------------------------------------------------------------------
    // Solid API methods
    // ------------------------------------------------------------------

    /// Return the names of all predefined platonic solids
    pub fn list_solids(&self) -> Vec<String> {
        solid_sets().keys().cloned().collect()
    }

    /// Get the node index sets defining a solid
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the solid (e.g. "tetrahedron", "cube"). Case insensitive.
    ///
    /// # Returns
    ///
    /// A list where each entry is a list of node indices forming one instance of the solid.
    /// Returns None if the name is not recognized.
    pub fn get_solid_nodes(&self, name: &str) -> Option<Vec<Vec<usize>>> {
        solid_sets().get(&name.to_lowercase()).cloned()
    }

    /// Get the edge lists for each instance of a solid
    ///
    /// For each node subset defined in the solid, all possible edges among those
    /// nodes are returned (complete connectivity within the subset).
    pub fn get_solid_edges(&self, name: &str) -> Option<Vec<Vec<(usize, usize)>>> {
        let subsets = self.get_solid_nodes(name)?;

        let mut edge_sets = Vec::new();
        for subset in subsets {
            let mut edges = Vec::new();
            let mut sorted_subset = subset.clone();
            sorted_subset.sort();

            for i_idx in 0..sorted_subset.len() {
                for j_idx in (i_idx + 1)..sorted_subset.len() {
                    edges.push((sorted_subset[i_idx], sorted_subset[j_idx]));
                }
            }
            edge_sets.push(edges);
        }

        Some(edge_sets)
    }

    /// Enumerate the symmetry group of a given solid
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the solid (e.g. "tetrahedron", "cube", "octahedron")
    /// * `even_only` - If true, generate only even permutations (alternating group)
    ///
    /// # Returns
    ///
    /// A list of operator objects, or None if the solid is not known
    pub fn enumerate_solid_group(
        &mut self,
        name: &str,
        even_only: bool,
    ) -> Option<Vec<OperatorInfo>> {
        let subsets = self.get_solid_nodes(name)?;

        // For simplicity, take the first subset to define the symmetry group
        let subset = &subsets[0];

        let perms = if even_only {
            generate_alternating_group(subset, 13)
        } else {
            generate_symmetric_group(subset, 13)
        };

        let group_name = if even_only {
            format!("A{}", subset.len())
        } else {
            format!("S{}", subset.len())
        };

        let mut result = Vec::new();
        for (idx, perm) in perms.iter().enumerate() {
            let op_id = format!("{}_{}_elem_{}", name.to_lowercase(), group_name, idx);
            self.operators.insert(op_id.clone(), perm.clone());

            if let Some(mut op_info) = self.get_operator(&op_id) {
                op_info.group = Some(group_name.clone());
                result.push(op_info);
            }
        }

        Some(result)
    }

    // ------------------------------------------------------------------
    // Node and edge accessors
    // ------------------------------------------------------------------

    /// Return a node object by index or label
    ///
    /// # Arguments
    ///
    /// * `id_or_label` - Either a 1-based node index or a node label string
    ///
    /// # Returns
    ///
    /// NodeInfo if found, None otherwise
    pub fn get_node(&self, id_or_label: &str) -> Option<NodeInfo> {
        // Try to parse as index first
        let node = if let Ok(index) = id_or_label.parse::<usize>() {
            find_node(&self.nodes, None, Some(index))
        } else {
            find_node(&self.nodes, Some(id_or_label), None)
        };

        node.map(|n| NodeInfo {
            id: n.index,
            label: n.label.clone(),
            node_type: n.node_type.clone(),
            coordinates: vec![n.coords.0, n.coords.1, n.coords.2],
            membership: self
                .node_membership
                .get(&n.index)
                .cloned()
                .unwrap_or_default(),
        })
    }

    /// List all nodes or only those of a given type
    ///
    /// # Arguments
    ///
    /// * `node_type` - Optional filter by node type ("center", "hexagon", "cube")
    pub fn list_nodes(&self, node_type: Option<&str>) -> Vec<NodeInfo> {
        self.nodes
            .iter()
            .filter(|node| node_type.is_none() || node_type == Some(node.node_type.as_str()))
            .map(|node| NodeInfo {
                id: node.index,
                label: node.label.clone(),
                node_type: node.node_type.clone(),
                coordinates: vec![node.coords.0, node.coords.1, node.coords.2],
                membership: self
                    .node_membership
                    .get(&node.index)
                    .cloned()
                    .unwrap_or_default(),
            })
            .collect()
    }

    /// Return an edge object by index or by node pair
    ///
    /// # Arguments
    ///
    /// * `index` - 1-based edge index in the edge list
    ///
    /// # Returns
    ///
    /// EdgeInfo if found, None otherwise
    pub fn get_edge_by_index(&self, index: usize) -> Option<EdgeInfo> {
        if index < 1 || index > self.edges.len() {
            return None;
        }

        let (i, j) = self.edges[index - 1];
        self.build_edge_info(index, i, j)
    }

    /// Get edge by node pair
    pub fn get_edge_by_pair(&self, i: usize, j: usize) -> Option<EdgeInfo> {
        let pair = (i.min(j), i.max(j));

        for (idx, &(ei, ej)) in self.edges.iter().enumerate() {
            let edge_pair = (ei.min(ej), ei.max(ej));
            if edge_pair == pair {
                return self.build_edge_info(idx + 1, ei, ej);
            }
        }

        None
    }

    fn build_edge_info(&self, edge_id: usize, i: usize, j: usize) -> Option<EdgeInfo> {
        let n1 = find_node(&self.nodes, None, Some(i))?;
        let n2 = find_node(&self.nodes, None, Some(j))?;

        let label = format!("{}--{}", n1.label, n2.label);

        // Determine edge type heuristically
        let edge_type = match (n1.node_type.as_str(), n2.node_type.as_str()) {
            ("hexagon", "hexagon") => "hex",
            ("cube", "cube") => "cube",
            ("hexagon", "cube") | ("cube", "hexagon") => "cross",
            ("center", "hexagon") | ("hexagon", "center") => "center-hex",
            ("center", "cube") | ("cube", "center") => "center-cube",
            _ => "other",
        }
        .to_string();

        let n1_membership = self.node_membership.get(&i).cloned().unwrap_or_default();
        let n2_membership = self.node_membership.get(&j).cloned().unwrap_or_default();
        let mut solids: Vec<String> = n1_membership.into_iter().chain(n2_membership).collect();
        solids.sort();
        solids.dedup();

        Some(EdgeInfo {
            id: edge_id,
            from: i,
            to: j,
            label,
            edge_type,
            solids,
        })
    }

    /// List all edges or filter by edge type
    ///
    /// # Arguments
    ///
    /// * `edge_type` - Optional filter by edge type ("hex", "cube", "cross", "center-hex", etc.)
    pub fn list_edges(&self, edge_type: Option<&str>) -> Vec<EdgeInfo> {
        self.edges
            .iter()
            .enumerate()
            .filter_map(|(idx, &(i, j))| {
                let edge_info = self.build_edge_info(idx + 1, i, j)?;
                if edge_type.is_none() || edge_type == Some(edge_info.edge_type.as_str()) {
                    Some(edge_info)
                } else {
                    None
                }
            })
            .collect()
    }

    // ------------------------------------------------------------------
    // Operator management and application
    // ------------------------------------------------------------------

    /// Add a custom operator to the registry
    ///
    /// # Arguments
    ///
    /// * `operator_id` - String identifier for the operator
    /// * `permutation` - Permutation of 1..13
    pub fn add_operator(&mut self, operator_id: String, permutation: Vec<usize>) -> Result<()> {
        if permutation.len() != 13 {
            return Err(anyhow!("Operator permutation must have length 13"));
        }

        let mut sorted = permutation.clone();
        sorted.sort();
        if sorted != (1..=13).collect::<Vec<_>>() {
            return Err(anyhow!(
                "Operator permutation must be a permutation of 1..13"
            ));
        }

        self.operators.insert(operator_id, permutation);
        Ok(())
    }

    /// Return an operator object by ID
    pub fn get_operator(&self, operator_id: &str) -> Option<OperatorInfo> {
        let perm = self.operators.get(operator_id)?;
        let mat = permutation_matrix(perm, 13);

        // Determine group name heuristically
        let group = if operator_id.starts_with("C6") {
            Some("C6".to_string())
        } else if operator_id.starts_with("D6") {
            Some("D6".to_string())
        } else if operator_id.starts_with("S7") {
            Some("S7".to_string())
        } else {
            None
        };

        Some(OperatorInfo {
            id: operator_id.to_string(),
            group,
            permutation: perm.clone(),
            matrix: mat.outer_iter().map(|row| row.to_vec()).collect(),
        })
    }

    /// Apply an operator to the adjacency matrix
    ///
    /// # Arguments
    ///
    /// * `operator_id` - The ID of the registered operator
    ///
    /// # Returns
    ///
    /// The permuted adjacency matrix
    pub fn apply_operator_to_adjacency(&self, operator_id: &str) -> Result<Array2<f64>> {
        let perm = self
            .operators
            .get(operator_id)
            .ok_or_else(|| anyhow!("Operator '{}' not found", operator_id))?;

        let p = permutation_matrix(perm, 13);
        let a = self.graph.get_adjacency_matrix();

        Ok(p.dot(&a).dot(&p.t()))
    }

    /// Apply an operator to a vector
    ///
    /// # Arguments
    ///
    /// * `operator_id` - The ID of the registered operator
    /// * `vector` - A 13-element vector
    ///
    /// # Returns
    ///
    /// The permuted vector
    pub fn apply_operator_to_vector(&self, operator_id: &str, vector: &[f64]) -> Result<Vec<f64>> {
        if vector.len() != 13 {
            return Err(anyhow!(
                "Vector length must be 13 to match operator dimension"
            ));
        }

        let perm = self
            .operators
            .get(operator_id)
            .ok_or_else(|| anyhow!("Operator '{}' not found", operator_id))?;

        let p = permutation_matrix(perm, 13);
        let v = Array2::from_shape_vec((13, 1), vector.to_vec()).unwrap();
        let result = p.dot(&v);

        Ok(result.iter().copied().collect())
    }

    /// Apply an operator to return new node ordering
    ///
    /// # Arguments
    ///
    /// * `operator_id` - The ID of the registered operator
    ///
    /// # Returns
    ///
    /// A list of permuted node info objects
    pub fn apply_operator_to_nodes(&self, operator_id: &str) -> Result<Vec<NodeInfo>> {
        let perm = self
            .operators
            .get(operator_id)
            .ok_or_else(|| anyhow!("Operator '{}' not found", operator_id))?;

        let mut result = Vec::new();
        for &i in perm {
            if let Some(node_info) = self.get_node(&i.to_string()) {
                result.push(node_info);
            }
        }

        Ok(result)
    }

    // ------------------------------------------------------------------
    // Quantum state operations
    // ------------------------------------------------------------------

    /// Return a QuantumOperator corresponding to a registered operator
    ///
    /// # Arguments
    ///
    /// * `operator_id` - The key of the operator in the internal registry
    ///
    /// # Returns
    ///
    /// A quantum operator constructed from the stored permutation
    pub fn get_quantum_operator(&self, operator_id: &str) -> Result<QuantumOperator> {
        let perm = self
            .operators
            .get(operator_id)
            .ok_or_else(|| anyhow!("Operator '{}' not found", operator_id))?;

        Ok(QuantumOperator::from_permutation(perm))
    }

    /// Apply a registered permutation operator to a quantum state
    ///
    /// # Arguments
    ///
    /// * `operator_id` - The ID of the registered operator
    /// * `state` - The quantum state to transform
    ///
    /// # Returns
    ///
    /// The transformed quantum state
    pub fn apply_operator_to_state(
        &self,
        operator_id: &str,
        state: &QuantumState,
    ) -> Result<QuantumState> {
        let qop = self.get_quantum_operator(operator_id)?;
        state.apply(&qop)
    }

    /// Enumerate all operators in a given group
    ///
    /// # Arguments
    ///
    /// * `group_name` - Group name: "C6", "D6", "S7", "S4", "A4", "A5"
    /// * `subset` - For S4/A4/A5, a subset of node indices must be provided
    ///
    /// # Returns
    ///
    /// A list of operator objects
    pub fn enumerate_group(
        &mut self,
        group_name: &str,
        subset: Option<&[usize]>,
    ) -> Result<Vec<OperatorInfo>> {
        let group_name_upper = group_name.to_uppercase();

        let perms = match group_name_upper.as_str() {
            "C6" => {
                let perms7 = generate_c6_subgroup();
                perms7
                    .into_iter()
                    .map(|mut p| {
                        p.extend(8..=13);
                        p
                    })
                    .collect()
            }
            "D6" => {
                let perms7 = generate_d6_subgroup();
                perms7
                    .into_iter()
                    .map(|mut p| {
                        p.extend(8..=13);
                        p
                    })
                    .collect()
            }
            "S7" => {
                let perms7 = generate_s7_permutations();
                perms7
                    .into_iter()
                    .map(|mut p| {
                        p.extend(8..=13);
                        p
                    })
                    .collect()
            }
            "S4" | "A4" | "A5" => {
                let subset = subset.ok_or_else(|| {
                    anyhow!("subset must be provided for group {}", group_name_upper)
                })?;

                let expected_len = match group_name_upper.as_str() {
                    "S4" | "A4" => 4,
                    "A5" => 5,
                    _ => unreachable!(),
                };

                if subset.len() != expected_len {
                    return Err(anyhow!(
                        "Group {} requires a subset of length {}, got {}",
                        group_name_upper,
                        expected_len,
                        subset.len()
                    ));
                }

                if group_name_upper == "S4" {
                    generate_symmetric_group(subset, 13)
                } else {
                    generate_alternating_group(subset, 13)
                }
            }
            _ => return Err(anyhow!("Unsupported group name {}", group_name_upper)),
        };

        let mut result = Vec::new();
        for (idx, perm) in perms.iter().enumerate() {
            let op_id = format!("{}_elem_{}", group_name_upper, idx);
            self.operators.insert(op_id.clone(), perm.clone());

            if let Some(op_info) = self.get_operator(&op_id) {
                result.push(op_info);
            }
        }

        Ok(result)
    }

    // ------------------------------------------------------------------
    // Serialization and validation
    // ------------------------------------------------------------------

    /// Serialize the full cube (nodes, edges, adjacency, operators) to JSON
    ///
    /// # Returns
    ///
    /// A JSON string representation of the cube
    pub fn serialize(&self) -> String {
        use serde_json::json;

        let nodes: Vec<_> = self
            .nodes
            .iter()
            .map(|node| {
                json!({
                    "index": node.index,
                    "label": node.label,
                    "type": node.node_type,
                    "coords": node.coords,
                })
            })
            .collect();

        let edges: Vec<_> = self.edges.iter().map(|(i, j)| vec![i, j]).collect();

        let adjacency = self.graph.get_adjacency_matrix();
        let adjacency_list: Vec<Vec<f64>> =
            adjacency.outer_iter().map(|row| row.to_vec()).collect();

        let operators: HashMap<String, Vec<usize>> = self.operators.clone();

        let data = json!({
            "nodes": nodes,
            "edges": edges,
            "adjacency": adjacency_list,
            "operators": operators,
        });

        serde_json::to_string_pretty(&data).unwrap()
    }

    /// Validate a permutation or adjacency matrix
    ///
    /// # Arguments
    ///
    /// * `permutation` - Optional permutation to validate (must be of length 13 and contain 1..13)
    /// * `matrix` - Optional adjacency matrix to validate (must be 13x13, symmetric, binary, zero diagonal)
    ///
    /// # Returns
    ///
    /// true if the configuration is valid, false otherwise
    pub fn validate_permutation(&self, permutation: &[usize]) -> bool {
        if permutation.len() != 13 {
            return false;
        }

        let mut sorted = permutation.to_vec();
        sorted.sort();
        sorted == (1..=13).collect::<Vec<_>>()
    }

    /// Validate an adjacency matrix
    pub fn validate_adjacency(&self, matrix: &Array2<f64>) -> bool {
        if matrix.shape() != [13, 13] {
            return false;
        }

        // Check symmetry
        for i in 0..13 {
            for j in 0..13 {
                if (matrix[[i, j]] - matrix[[j, i]]).abs() > 1e-10 {
                    return false;
                }
            }
        }

        // Check binary values
        for &val in matrix.iter() {
            if (val - 0.0).abs() > 1e-10 && (val - 1.0).abs() > 1e-10 {
                return false;
            }
        }

        // Check zero diagonal
        for i in 0..13 {
            if matrix[[i, i]].abs() > 1e-10 {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn test_create_default_cube() {
        let cube = MetatronCube::default();
        assert_eq!(cube.nodes.len(), 13);
        // canonical_edges() has 23 entries (raw input)
        assert_eq!(cube.edges.len(), 23);
        // But the graph deduplicates, so only 21 unique edges
        assert_eq!(cube.graph.num_edges(), 21);
        assert!(!cube.operators.is_empty());
    }

    #[test]
    fn test_list_solids() {
        let cube = MetatronCube::default();
        let solids = cube.list_solids();
        assert!(solids.contains(&"tetrahedron".to_string()));
        assert!(solids.contains(&"cube".to_string()));
        assert!(solids.contains(&"octahedron".to_string()));
    }

    #[test]
    fn test_get_solid_nodes() {
        let cube = MetatronCube::default();
        let cube_nodes = cube.get_solid_nodes("cube").unwrap();
        assert_eq!(cube_nodes.len(), 1);
        assert_eq!(cube_nodes[0], vec![8, 9, 10, 11, 12, 13]);
    }

    #[test]
    fn test_get_node() {
        let cube = MetatronCube::default();
        let node = cube.get_node("1").unwrap();
        assert_eq!(node.id, 1);
        assert_eq!(node.label, "C");
        assert_eq!(node.node_type, "center");
    }

    #[test]
    fn test_list_nodes_by_type() {
        let cube = MetatronCube::default();
        let hexagon_nodes = cube.list_nodes(Some("hexagon"));
        assert_eq!(hexagon_nodes.len(), 6);

        let center_nodes = cube.list_nodes(Some("center"));
        assert_eq!(center_nodes.len(), 1);
    }

    #[test]
    fn test_get_edge() {
        let cube = MetatronCube::default();
        let edge = cube.get_edge_by_index(1).unwrap();
        assert_eq!(edge.from, 1);
        assert_eq!(edge.to, 2);
    }

    #[test]
    fn test_add_operator() {
        let mut cube = MetatronCube::default();
        let perm = vec![2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
        cube.add_operator("test_op".to_string(), perm.clone())
            .unwrap();

        let op = cube.get_operator("test_op").unwrap();
        assert_eq!(op.permutation, perm);
    }

    #[test]
    fn test_apply_operator_to_adjacency() {
        let cube = MetatronCube::default();
        // Use a registered C6 rotation
        let adj = cube.apply_operator_to_adjacency("C6_rot_0").unwrap();
        assert_eq!(adj.shape(), &[13, 13]);
    }

    #[test]
    fn test_validate_permutation() {
        let cube = MetatronCube::default();
        let valid = vec![2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
        assert!(cube.validate_permutation(&valid));

        let invalid = vec![1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
        assert!(!cube.validate_permutation(&invalid));
    }

    #[test]
    fn test_serialize() {
        let cube = MetatronCube::default();
        let json = cube.serialize();
        assert!(json.contains("nodes"));
        assert!(json.contains("edges"));
        assert!(json.contains("adjacency"));
        assert!(json.contains("operators"));
    }

    #[test]
    fn test_enumerate_group_c6() {
        let mut cube = MetatronCube::default();
        let ops = cube.enumerate_group("C6", None).unwrap();
        assert_eq!(ops.len(), 6);
    }

    #[test]
    fn test_get_quantum_operator() {
        let cube = MetatronCube::default();
        let qop = cube.get_quantum_operator("C6_rot_0").unwrap();
        assert!(qop.is_unitary(1e-10));
    }

    #[test]
    fn test_apply_operator_to_state() {
        let cube = MetatronCube::default();
        let amplitudes = vec![Complex64::new(1.0, 0.0)];
        let state = QuantumState::new(amplitudes, true).unwrap();

        let transformed = cube.apply_operator_to_state("C6_rot_0", &state).unwrap();
        assert_eq!(transformed.amplitudes.len(), 13);
    }
}
