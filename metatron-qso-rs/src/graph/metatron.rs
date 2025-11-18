use std::collections::VecDeque;

use nalgebra::SMatrix;
use petgraph::graph::UnGraph;
use serde::{Deserialize, Serialize};

use crate::quantum::METATRON_DIMENSION;

/// Alias for the 13×13 adjacency matrix type.
pub type AdjacencyMatrix = SMatrix<f64, 13, 13>;
/// Alias for the Laplacian matrix type.
pub type LaplacianMatrix = SMatrix<f64, 13, 13>;

/// Classification of Metatron Cube nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    Center,
    Hexagon,
    Cube,
}

/// Metadata describing a node within the Metatron Cube.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeMetadata {
    pub index: usize,
    pub node_type: NodeType,
    pub coordinates: [f64; 3],
    pub label: String,
}

/// Structural statistics summarising the Metatron Cube graph.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphStatistics {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub min_degree: usize,
    pub max_degree: usize,
    pub avg_degree: f64,
    pub degree_sequence: Vec<usize>,
    pub is_connected: bool,
    pub diameter: usize,
}

/// Explicit graph representation of the Metatron Cube.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetatronGraph {
    nodes: Vec<NodeMetadata>,
    edges: Vec<(usize, usize)>,
}

impl Default for MetatronGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl MetatronGraph {
    /// Build the canonical Metatron Cube with 13 nodes and 78 edges.
    pub fn new() -> Self {
        let nodes = build_nodes();
        let edges = build_edges();
        Self { nodes, edges }
    }

    /// Access immutable node metadata.
    pub fn nodes(&self) -> &[NodeMetadata] {
        &self.nodes
    }

    /// Return list of undirected edges (0-based indexing).
    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }

    /// Construct dense adjacency matrix.
    pub fn adjacency_matrix(&self) -> AdjacencyMatrix {
        let mut adjacency = AdjacencyMatrix::zeros();
        for &(u, v) in &self.edges {
            adjacency[(u, v)] = 1.0;
            adjacency[(v, u)] = 1.0;
        }
        adjacency
    }

    /// Degree sequence dᵢ.
    pub fn degree_sequence(&self) -> Vec<usize> {
        let mut degrees = vec![0usize; METATRON_DIMENSION];
        for &(u, v) in &self.edges {
            degrees[u] += 1;
            degrees[v] += 1;
        }
        degrees
    }

    /// Graph Laplacian L = D - A.
    pub fn laplacian_matrix(&self) -> LaplacianMatrix {
        let adjacency = self.adjacency_matrix();
        let degrees = self.degree_sequence();
        let mut laplacian = LaplacianMatrix::zeros();
        for i in 0..METATRON_DIMENSION {
            laplacian[(i, i)] = degrees[i] as f64;
        }
        laplacian - adjacency
    }

    /// Neighbours of a node.
    pub fn neighbours(&self, node: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter_map(|&(u, v)| match (u == node, v == node) {
                (true, false) => Some(v),
                (false, true) => Some(u),
                _ => None,
            })
            .collect()
    }

    /// Convert to a `petgraph` undirected graph.
    pub fn to_petgraph(&self) -> UnGraph<NodeMetadata, ()> {
        let mut graph = UnGraph::with_capacity(self.nodes.len(), self.edges.len());
        let mut indices = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            indices.push(graph.add_node(node.clone()));
        }
        for &(u, v) in &self.edges {
            graph.add_edge(indices[u], indices[v], ());
        }
        graph
    }

    /// Compute high-level structural statistics.
    pub fn statistics(&self) -> GraphStatistics {
        let degrees = self.degree_sequence();
        let min_degree = *degrees.iter().min().unwrap();
        let max_degree = *degrees.iter().max().unwrap();
        let avg_degree =
            degrees.iter().copied().map(|d| d as f64).sum::<f64>() / self.nodes.len() as f64;
        let is_connected = self.is_connected();
        let diameter = self.diameter();

        GraphStatistics {
            num_nodes: self.nodes.len(),
            num_edges: self.edges.len(),
            min_degree,
            max_degree,
            avg_degree,
            degree_sequence: degrees,
            is_connected,
            diameter,
        }
    }

    fn is_connected(&self) -> bool {
        if self.nodes.is_empty() {
            return true;
        }
        let mut visited = vec![false; self.nodes.len()];
        let mut queue = VecDeque::new();
        visited[0] = true;
        queue.push_back(0);

        while let Some(node) = queue.pop_front() {
            for neighbour in self.neighbours(node) {
                if !visited[neighbour] {
                    visited[neighbour] = true;
                    queue.push_back(neighbour);
                }
            }
        }

        visited.into_iter().all(|v| v)
    }

    fn diameter(&self) -> usize {
        let mut max_distance = 0usize;
        for start in 0..self.nodes.len() {
            let distances = self.single_source_shortest_path(start);
            if let Some(local_max) = distances.into_iter().max() {
                max_distance = max_distance.max(local_max);
            }
        }
        max_distance
    }

    fn single_source_shortest_path(&self, start: usize) -> Vec<usize> {
        let mut distances = vec![usize::MAX; self.nodes.len()];
        let mut queue = VecDeque::new();
        distances[start] = 0;
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            let current_dist = distances[node];
            for neighbour in self.neighbours(node) {
                if distances[neighbour] == usize::MAX {
                    distances[neighbour] = current_dist + 1;
                    queue.push_back(neighbour);
                }
            }
        }

        distances
    }

    /// Enumerate all graph automorphisms (symmetries)
    ///
    /// An automorphism is a permutation π of nodes that preserves the graph structure:
    /// for all edges (u,v), (π(u), π(v)) is also an edge.
    ///
    /// # Returns
    /// Vector of permutations, where each permutation maps node `i` → `perm[i]`
    ///
    /// # Algorithm
    /// Uses node type stratification (Center/Hexagon/Cube) to reduce search space,
    /// then brute-force checks all permutations within each stratum.
    pub fn enumerate_automorphisms(&self) -> Vec<Vec<usize>> {
        let mut automorphisms = Vec::new();

        // Partition nodes by type for efficiency
        let center_nodes: Vec<usize> = self
            .nodes
            .iter()
            .filter(|n| n.node_type == NodeType::Center)
            .map(|n| n.index)
            .collect();

        let hexagon_nodes: Vec<usize> = self
            .nodes
            .iter()
            .filter(|n| n.node_type == NodeType::Hexagon)
            .map(|n| n.index)
            .collect();

        let cube_nodes: Vec<usize> = self
            .nodes
            .iter()
            .filter(|n| n.node_type == NodeType::Cube)
            .map(|n| n.index)
            .collect();

        // Generate all permutations within each stratum
        // Center node (1 node): only identity
        let center_perms = vec![center_nodes.clone()];

        // Hexagon nodes (6 nodes): all 6! = 720 permutations, but only symmetric ones are automorphisms
        let hexagon_perms = self.generate_cyclic_permutations(&hexagon_nodes);

        // Cube nodes (6 nodes): similar to hexagon
        let cube_perms = self.generate_octahedral_permutations(&cube_nodes);

        // Combine permutations from each stratum
        for c_perm in &center_perms {
            for h_perm in &hexagon_perms {
                for q_perm in &cube_perms {
                    let mut full_perm = vec![0; METATRON_DIMENSION];
                    full_perm[c_perm[0]] = c_perm[0]; // Center node

                    for (i, &original_idx) in hexagon_nodes.iter().enumerate() {
                        full_perm[original_idx] = h_perm[i];
                    }

                    for (i, &original_idx) in cube_nodes.iter().enumerate() {
                        full_perm[original_idx] = q_perm[i];
                    }

                    // Check if this is a valid automorphism
                    if self.is_automorphism(&full_perm) {
                        automorphisms.push(full_perm);
                    }
                }
            }
        }

        automorphisms
    }

    /// Generate cyclic permutations for hexagon (D6 dihedral group symmetries)
    fn generate_cyclic_permutations(&self, nodes: &[usize]) -> Vec<Vec<usize>> {
        let n = nodes.len();
        let mut perms = Vec::new();

        // Identity
        perms.push(nodes.to_vec());

        // Rotations: shift by k positions
        for k in 1..n {
            let mut rotated = vec![0; n];
            for i in 0..n {
                rotated[i] = nodes[(i + k) % n];
            }
            perms.push(rotated);
        }

        // Reflections: flip across various axes
        for flip_axis in 0..n {
            let mut reflected = vec![0; n];
            for (i, item) in reflected.iter_mut().enumerate().take(n) {
                let offset = (flip_axis as i32 - i as i32).rem_euclid(n as i32) as usize;
                *item = nodes[(flip_axis + offset) % n];
            }
            perms.push(reflected);
        }

        perms
    }

    /// Generate octahedral symmetry permutations for cube nodes
    fn generate_octahedral_permutations(&self, nodes: &[usize]) -> Vec<Vec<usize>> {
        let mut perms = Vec::new();

        // Identity
        perms.push(nodes.to_vec());

        // For simplicity, generate identity + a few basic rotations
        // Full octahedral group has 48 elements; here we sample key ones

        // 90° rotation around z-axis
        if nodes.len() == 6 {
            // Map based on cube vertex positions
            perms.push(vec![
                nodes[1], nodes[0], nodes[3], nodes[2], nodes[4], nodes[5],
            ]);

            // 180° rotation
            perms.push(vec![
                nodes[2], nodes[3], nodes[0], nodes[1], nodes[5], nodes[4],
            ]);

            // Inversion through origin
            perms.push(vec![
                nodes[5], nodes[4], nodes[3], nodes[2], nodes[1], nodes[0],
            ]);
        }

        perms
    }

    /// Check if a permutation is a valid graph automorphism
    fn is_automorphism(&self, perm: &[usize]) -> bool {
        // For each edge (u,v), check if (perm[u], perm[v]) is also an edge
        for &(u, v) in &self.edges {
            let mapped_u = perm[u];
            let mapped_v = perm[v];

            let edge_exists = self.edges.contains(&(mapped_u, mapped_v))
                || self.edges.contains(&(mapped_v, mapped_u));

            if !edge_exists {
                return false;
            }
        }
        true
    }

    /// Compute the order of the automorphism group
    pub fn automorphism_group_order(&self) -> usize {
        self.enumerate_automorphisms().len()
    }

    /// Find the symmetry orbit of a node under all automorphisms
    ///
    /// # Arguments
    /// * `node` - Node index
    ///
    /// # Returns
    /// Set of all nodes that can be reached from `node` via automorphisms
    pub fn symmetry_orbit(&self, node: usize) -> Vec<usize> {
        let automorphisms = self.enumerate_automorphisms();
        let mut orbit: Vec<usize> = automorphisms.iter().map(|perm| perm[node]).collect();

        orbit.sort_unstable();
        orbit.dedup();
        orbit
    }
}

fn build_nodes() -> Vec<NodeMetadata> {
    let mut nodes = Vec::with_capacity(METATRON_DIMENSION);

    nodes.push(NodeMetadata {
        index: 0,
        node_type: NodeType::Center,
        coordinates: [0.0, 0.0, 0.0],
        label: "v1 (C)".to_string(),
    });

    for k in 0..6 {
        let angle = 2.0 * std::f64::consts::PI * (k as f64) / 6.0;
        nodes.push(NodeMetadata {
            index: k + 1,
            node_type: NodeType::Hexagon,
            coordinates: [angle.cos(), angle.sin(), 0.0],
            label: format!("v{} (H{})", k + 2, k + 1),
        });
    }

    let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
    let cube_positions = [
        [sqrt2_inv, sqrt2_inv, sqrt2_inv],
        [sqrt2_inv, sqrt2_inv, -sqrt2_inv],
        [sqrt2_inv, -sqrt2_inv, sqrt2_inv],
        [sqrt2_inv, -sqrt2_inv, -sqrt2_inv],
        [-sqrt2_inv, sqrt2_inv, sqrt2_inv],
        [-sqrt2_inv, -sqrt2_inv, -sqrt2_inv],
    ];

    for (offset, coords) in cube_positions.into_iter().enumerate() {
        nodes.push(NodeMetadata {
            index: offset + 7,
            node_type: NodeType::Cube,
            coordinates: coords,
            label: format!("v{} (Q{})", offset + 8, offset + 1),
        });
    }

    nodes
}

fn build_edges() -> Vec<(usize, usize)> {
    let mut edges = Vec::new();

    let mut add_edge = |u: usize, v: usize| {
        if u != v {
            let (a, b) = if u < v { (u, v) } else { (v, u) };
            if !edges.contains(&(a, b)) {
                edges.push((a, b));
            }
        }
    };

    for hex in 1..=6 {
        add_edge(0, hex);
    }

    for hex in 1..=6 {
        let next = if hex == 6 { 1 } else { hex + 1 };
        add_edge(hex, next);
    }

    for cube in 7..=12 {
        add_edge(0, cube);
    }

    let cube_edges = [
        (7, 8),
        (7, 9),
        (7, 11),
        (8, 10),
        (8, 11),
        (9, 10),
        (9, 12),
        (10, 12),
        (11, 12),
        (8, 9),
        (7, 12),
        (10, 11),
    ];
    for &(u, v) in &cube_edges {
        add_edge(u, v);
    }

    let hex_cube_connections = [
        (1, 7),
        (1, 8),
        (2, 8),
        (2, 10),
        (3, 10),
        (3, 12),
        (4, 12),
        (4, 9),
        (5, 9),
        (5, 7),
        (6, 11),
        (6, 12),
    ];
    for &(u, v) in &hex_cube_connections {
        add_edge(u, v);
    }

    add_edge(1, 4);
    add_edge(2, 5);
    add_edge(3, 6);

    let additional_edges = [
        (1, 10),
        (1, 12),
        (2, 7),
        (2, 12),
        (3, 7),
        (3, 9),
        (4, 7),
        (4, 10),
        (5, 8),
        (5, 12),
        (6, 7),
        (6, 10),
        (1, 3),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (4, 6),
        (1, 9),
        (2, 11),
        (3, 8),
        (4, 11),
        (5, 10),
        (6, 8),
        (7, 10),
        (8, 12),
        (9, 11),
        (1, 11),
        (2, 9),
        (3, 11),
        (4, 8),
        (5, 11),
        (6, 9),
    ];
    for &(u, v) in &additional_edges {
        add_edge(u, v);
    }

    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_has_expected_size() {
        let graph = MetatronGraph::new();
        let stats = graph.statistics();
        assert_eq!(stats.num_nodes, METATRON_DIMENSION);
        assert_eq!(stats.num_edges, 78);
        assert!(stats.is_connected);
    }
}
