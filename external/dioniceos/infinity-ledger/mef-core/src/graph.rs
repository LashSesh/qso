/*!
 * Graph Module - Metatron Cube Graph Representation
 *
 * This module implements the core data structure representing the Metatron Cube
 * as a graph. Each graph consists of a set of canonical nodes (from the geometry
 * module) and an undirected edge list. The class `MetatronCubeGraph` provides
 * methods to obtain the adjacency matrix, add or remove edges, and apply
 * permutations on the node order.
 *
 * While the default instance uses the canonical nodes and partial edge list from
 * the blueprint, the class accepts custom node/edge inputs for experimentation.
 * This allows integration with additional research modules without changing the
 * core definitions.
 *
 * The adjacency matrix is always symmetric, reflecting the undirected nature of
 * the Metatron Cube. Self-loops are not used and are explicitly prohibited.
 */

use anyhow::{anyhow, Result};
use ndarray::Array2;
use std::collections::HashMap;

use crate::geometry::{canonical_edges, canonical_nodes, Node};

/// A graph representation of the Metatron Cube
#[derive(Debug, Clone)]
pub struct MetatronCubeGraph {
    /// The nodes in the graph
    pub nodes: Vec<Node>,
    /// Edge weights as a map from (min_idx, max_idx) to weight
    pub edge_weights: HashMap<(usize, usize), f64>,
    /// Cached adjacency matrix
    adjacency: Array2<f64>,
}

impl MetatronCubeGraph {
    /// Create a new Metatron Cube graph with default nodes and edges
    pub fn new() -> Self {
        Self::with_nodes_and_edges(None, None, None)
    }

    /// Create a new Metatron Cube graph with custom nodes and/or edges
    ///
    /// # Arguments
    ///
    /// * `nodes` - Optional custom nodes. If None, uses canonical nodes.
    /// * `edges` - Optional unweighted edges. If None and weighted_edges is None, uses canonical edges.
    /// * `weighted_edges` - Optional weighted edges. Takes precedence over edges parameter.
    ///
    /// # Returns
    ///
    /// A new MetatronCubeGraph instance
    pub fn with_nodes_and_edges(
        nodes: Option<Vec<Node>>,
        edges: Option<Vec<(usize, usize)>>,
        weighted_edges: Option<Vec<(usize, usize, f64)>>,
    ) -> Self {
        // Use canonical definitions if not provided
        let nodes = nodes.unwrap_or_else(canonical_nodes);

        // Validate node indices are unique and consecutive
        let mut indices: Vec<usize> = nodes.iter().map(|n| n.index).collect();
        indices.sort();
        let expected: Vec<usize> = (1..=nodes.len()).collect();
        if indices != expected {
            panic!("Node indices must be unique, 1-based and consecutive");
        }

        // Build the internal edge list with weights
        let mut edge_weights = HashMap::new();

        if let Some(weighted) = weighted_edges {
            for (i, j, w) in weighted {
                if i == j {
                    panic!("Self-loops are not allowed: edge ({}, {})", i, j);
                }
                let key = (i.min(j), i.max(j));
                edge_weights.insert(key, w);
            }
        } else {
            // Fallback to unweighted edges (canonical if None)
            let edge_list = edges.unwrap_or_else(canonical_edges);
            for (i, j) in edge_list {
                if i == j {
                    panic!("Self-loops are not allowed: edge ({}, {})", i, j);
                }
                let key = (i.min(j), i.max(j));
                edge_weights.insert(key, 1.0);
            }
        }

        let adjacency = Self::compute_adjacency_matrix(&nodes, &edge_weights);

        MetatronCubeGraph {
            nodes,
            edge_weights,
            adjacency,
        }
    }

    /// Compute the symmetric adjacency matrix from the current edge list
    fn compute_adjacency_matrix(
        nodes: &[Node],
        edge_weights: &HashMap<(usize, usize), f64>,
    ) -> Array2<f64> {
        let n = nodes.len();
        let mut adjacency = Array2::zeros((n, n));

        for ((i, j), w) in edge_weights {
            let u = i - 1;
            let v = j - 1;
            adjacency[[u, v]] = *w;
            adjacency[[v, u]] = *w;
        }

        adjacency
    }

    /// Return a copy of the adjacency matrix
    pub fn get_adjacency_matrix(&self) -> Array2<f64> {
        self.adjacency.clone()
    }

    /// Return the neighboring node indices of a given node
    ///
    /// # Arguments
    ///
    /// * `index` - 1-based node index
    ///
    /// # Returns
    ///
    /// Vector of 1-based node indices adjacent to the given node
    pub fn neighbors(&self, index: usize) -> Result<Vec<usize>> {
        self.validate_node_index(index)?;
        let idx0 = index - 1;
        let row = self.adjacency.row(idx0);

        Ok(row
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0.0)
            .map(|(i, _)| i + 1)
            .collect())
    }

    /// Return the degree (number of incident edges) of a node
    pub fn degree(&self, index: usize) -> Result<usize> {
        Ok(self.neighbors(index)?.len())
    }

    /// Add an undirected edge between nodes i and j with weight 1.0
    pub fn add_edge(&mut self, i: usize, j: usize) -> Result<()> {
        self.add_weighted_edge(i, j, 1.0)
    }

    /// Add or update an undirected edge with the given weight
    ///
    /// # Arguments
    ///
    /// * `i` - 1-based node index
    /// * `j` - 1-based node index  
    /// * `weight` - Weight for the edge. If zero, the edge is removed.
    pub fn add_weighted_edge(&mut self, i: usize, j: usize, weight: f64) -> Result<()> {
        self.validate_node_index(i)?;
        self.validate_node_index(j)?;

        if i == j {
            return Err(anyhow!("Self-loops are not allowed"));
        }

        let key = (i.min(j), i.max(j));

        if weight == 0.0 {
            self.edge_weights.remove(&key);
        } else {
            self.edge_weights.insert(key, weight);
        }

        self.adjacency = Self::compute_adjacency_matrix(&self.nodes, &self.edge_weights);
        Ok(())
    }

    /// Remove the undirected edge between i and j if present
    pub fn remove_edge(&mut self, i: usize, j: usize) {
        let key = (i.min(j), i.max(j));
        if self.edge_weights.remove(&key).is_some() {
            self.adjacency = Self::compute_adjacency_matrix(&self.nodes, &self.edge_weights);
        }
    }

    /// Validate that a node index is within bounds
    fn validate_node_index(&self, index: usize) -> Result<()> {
        if index < 1 || index > self.nodes.len() {
            return Err(anyhow!(
                "Node index {} out of bounds (1..{})",
                index,
                self.nodes.len()
            ));
        }
        Ok(())
    }

    /// Return a new graph with node order permuted by sigma
    ///
    /// The permutation sigma must be a vector of length equal to the number of
    /// nodes, containing each integer from 1..n exactly once. It describes the
    /// new order of the nodes (1-based). For example, sigma = [1, 3, 2, 4, ...]
    /// swaps nodes 2 and 3.
    ///
    /// # Arguments
    ///
    /// * `sigma` - A permutation of 1..n
    ///
    /// # Returns
    ///
    /// A new graph with permuted nodes and edges
    pub fn permute(&self, sigma: &[usize]) -> Result<Self> {
        let n = self.nodes.len();

        if sigma.len() != n {
            return Err(anyhow!(
                "Permutation must have length {}, got {}",
                n,
                sigma.len()
            ));
        }

        // Validate sigma is a valid permutation
        let mut sigma_set: Vec<usize> = sigma.to_vec();
        sigma_set.sort();
        let expected: Vec<usize> = (1..=n).collect();
        if sigma_set != expected {
            return Err(anyhow!("sigma must be a permutation of 1..n"));
        }

        // Permute the nodes
        let new_nodes: Vec<Node> = sigma.iter().map(|&i| self.nodes[i - 1].clone()).collect();

        // Remap edges by applying sigma to each endpoint
        let idx_map: HashMap<usize, usize> = sigma
            .iter()
            .enumerate()
            .map(|(new_idx, &old)| (old, new_idx + 1))
            .collect();

        let new_weighted_edges: Vec<(usize, usize, f64)> = self
            .edge_weights
            .iter()
            .map(|((i, j), w)| {
                let new_i = idx_map[i];
                let new_j = idx_map[j];
                (new_i, new_j, *w)
            })
            .collect();

        Ok(Self::with_nodes_and_edges(
            Some(new_nodes),
            None,
            Some(new_weighted_edges),
        ))
    }

    /// Apply a permutation matrix to the adjacency matrix
    ///
    /// This method produces a new graph with the same node order but with
    /// adjacency corresponding to the permuted indices. The provided matrix P
    /// must be orthogonal and binary (a valid permutation matrix).
    ///
    /// # Arguments
    ///
    /// * `p` - Permutation matrix of shape (n, n)
    ///
    /// # Returns
    ///
    /// A new graph whose adjacency matrix equals P @ A @ P^T
    pub fn apply_permutation_matrix(&self, p: &Array2<f64>) -> Result<Self> {
        let a = self.get_adjacency_matrix();

        if p.shape() != a.shape() {
            return Err(anyhow!(
                "Permutation matrix must be same shape as adjacency matrix"
            ));
        }

        // Compute the new adjacency matrix: A' = P @ A @ P^T
        let a_prime = p.dot(&a).dot(&p.t());

        // Derive new edge list from A_prime
        let n = self.nodes.len();
        let mut new_edges = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                if a_prime[[i, j]] != 0.0 {
                    new_edges.push((i + 1, j + 1, a_prime[[i, j]]));
                }
            }
        }

        Ok(Self::with_nodes_and_edges(
            Some(self.nodes.clone()),
            None,
            Some(new_edges),
        ))
    }

    /// Get the number of nodes in the graph
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges in the graph
    pub fn num_edges(&self) -> usize {
        self.edge_weights.len()
    }
}

impl Default for MetatronCubeGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_default_graph() {
        let g = MetatronCubeGraph::new();
        assert_eq!(g.num_nodes(), 13);
        // canonical_edges() has 23 entries but 2 duplicates, so 21 unique edges
        assert_eq!(g.num_edges(), 21);
    }

    #[test]
    fn test_adjacency_matrix_shape() {
        let g = MetatronCubeGraph::new();
        let adj = g.get_adjacency_matrix();
        assert_eq!(adj.shape(), &[13, 13]);
    }

    #[test]
    fn test_adjacency_matrix_symmetric() {
        let g = MetatronCubeGraph::new();
        let adj = g.get_adjacency_matrix();

        // Check symmetry
        for i in 0..13 {
            for j in 0..13 {
                assert_eq!(adj[[i, j]], adj[[j, i]]);
            }
        }
    }

    #[test]
    fn test_degree_center_node() {
        let g = MetatronCubeGraph::new();
        let deg = g.degree(1).unwrap();
        assert_eq!(deg, 6); // center connected to 6 hexagon nodes
    }

    #[test]
    fn test_neighbors() {
        let g = MetatronCubeGraph::new();
        let neighbors = g.neighbors(1).unwrap();
        assert_eq!(neighbors.len(), 6);

        // Center should be connected to hexagon nodes 2-7
        for i in 2..=7 {
            assert!(neighbors.contains(&i));
        }
    }

    #[test]
    fn test_add_edge() {
        let mut g = MetatronCubeGraph::new();
        let initial_edges = g.num_edges();

        // Add an edge that doesn't exist yet (e.g., between two cube nodes not already connected)
        g.add_edge(8, 13).unwrap();
        assert_eq!(g.num_edges(), initial_edges + 1);

        // Check adjacency matrix updated
        let adj = g.get_adjacency_matrix();
        assert_eq!(adj[[7, 12]], 1.0);
        assert_eq!(adj[[12, 7]], 1.0);
    }

    #[test]
    fn test_remove_edge() {
        let mut g = MetatronCubeGraph::new();
        let initial_edges = g.num_edges();

        // Remove an existing edge (e.g., center to H1)
        g.remove_edge(1, 2);
        assert_eq!(g.num_edges(), initial_edges - 1);

        // Check adjacency matrix updated
        let adj = g.get_adjacency_matrix();
        assert_eq!(adj[[0, 1]], 0.0);
        assert_eq!(adj[[1, 0]], 0.0);
    }

    #[test]
    fn test_self_loop_rejected() {
        let mut g = MetatronCubeGraph::new();
        let result = g.add_edge(1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_edges() {
        let weighted = vec![(1, 2, 2.5), (2, 3, 1.5)];
        let g = MetatronCubeGraph::with_nodes_and_edges(None, None, Some(weighted));

        let adj = g.get_adjacency_matrix();
        assert_eq!(adj[[0, 1]], 2.5);
        assert_eq!(adj[[1, 2]], 1.5);
    }

    #[test]
    fn test_permute_identity() {
        let g = MetatronCubeGraph::new();
        let sigma: Vec<usize> = (1..=13).collect();
        let g2 = g.permute(&sigma).unwrap();

        let adj1 = g.get_adjacency_matrix();
        let adj2 = g2.get_adjacency_matrix();

        // Identity permutation should not change adjacency
        for i in 0..13 {
            for j in 0..13 {
                assert_eq!(adj1[[i, j]], adj2[[i, j]]);
            }
        }
    }

    #[test]
    fn test_permute_swap() {
        let g = MetatronCubeGraph::new();
        // Swap nodes 2 and 3 (H1 and H2)
        let sigma = vec![1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
        let g2 = g.permute(&sigma).unwrap();

        let adj1 = g.get_adjacency_matrix();
        let adj2 = g2.get_adjacency_matrix();

        // After swapping nodes 2 and 3, check that edges are remapped
        assert_eq!(adj2[[0, 1]], adj1[[0, 2]]); // new position of node 3
        assert_eq!(adj2[[0, 2]], adj1[[0, 1]]); // new position of node 2
    }

    #[test]
    fn test_invalid_node_index() {
        let g = MetatronCubeGraph::new();
        assert!(g.neighbors(0).is_err());
        assert!(g.neighbors(14).is_err());
    }

    #[test]
    fn test_invalid_permutation() {
        let g = MetatronCubeGraph::new();
        let bad_sigma = vec![1, 2, 3]; // too short
        assert!(g.permute(&bad_sigma).is_err());

        let bad_sigma2 = vec![1, 2, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]; // duplicate
        assert!(g.permute(&bad_sigma2).is_err());
    }
}
