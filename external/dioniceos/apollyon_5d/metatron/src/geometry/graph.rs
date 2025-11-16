use std::collections::HashMap;

use serde::Serialize;

use crate::error::{EngineError, EngineResult};
use crate::geometry::{canonical_edges, canonical_nodes, Node};

#[derive(Debug, Clone, Serialize)]
pub struct MetatronCubeGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<(usize, usize)>,
    edge_weights: HashMap<(usize, usize), f64>,
    adjacency: Vec<Vec<f64>>,
}

impl MetatronCubeGraph {
    pub fn new() -> Self {
        Self::with_edges(None, None)
    }

    pub fn with_edges(nodes: Option<Vec<Node>>, edges: Option<Vec<(usize, usize)>>) -> Self {
        let nodes = nodes.unwrap_or_else(canonical_nodes);
        let mut edge_weights = HashMap::new();
        let edges = edges.unwrap_or_else(canonical_edges);
        for (i, j) in edges.iter().copied() {
            if i == j {
                continue;
            }
            let key = if i < j { (i, j) } else { (j, i) };
            edge_weights.entry(key).or_insert(1.0);
        }
        let edges: Vec<(usize, usize)> = edge_weights.keys().copied().collect();
        let adjacency = Self::build_adjacency(nodes.len(), &edge_weights);
        Self {
            nodes,
            edges,
            edge_weights,
            adjacency,
        }
    }

    fn build_adjacency(n: usize, weights: &HashMap<(usize, usize), f64>) -> Vec<Vec<f64>> {
        let mut matrix = vec![vec![0.0; n]; n];
        for (&(i, j), &w) in weights {
            let ui = i - 1;
            let uj = j - 1;
            matrix[ui][uj] = w;
            matrix[uj][ui] = w;
        }
        matrix
    }

    pub fn get_adjacency_matrix(&self) -> Vec<Vec<f64>> {
        self.adjacency.clone()
    }

    fn validate_index(&self, index: usize) -> EngineResult<()> {
        if index == 0 || index > self.nodes.len() {
            Err(EngineError::InvalidNodeIndex {
                index,
                len: self.nodes.len(),
            })
        } else {
            Ok(())
        }
    }

    pub fn get_neighbors(&self, index: usize) -> EngineResult<Vec<&Node>> {
        self.validate_index(index)?;
        let row = &self.adjacency[index - 1];
        Ok(row
            .iter()
            .enumerate()
            .filter_map(|(idx, &val)| {
                if val != 0.0 {
                    Some(&self.nodes[idx])
                } else {
                    None
                }
            })
            .collect())
    }

    pub fn get_edge_list(&self) -> Vec<(usize, usize)> {
        let mut edges = self.edges.clone();
        edges.sort();
        edges
    }

    pub fn degree(&self, index: usize) -> EngineResult<usize> {
        Ok(self.get_neighbors(index)?.len())
    }

    pub fn add_edge(&mut self, i: usize, j: usize) -> EngineResult<()> {
        self.add_weighted_edge(i, j, 1.0)
    }

    pub fn add_weighted_edge(&mut self, i: usize, j: usize, weight: f64) -> EngineResult<()> {
        self.validate_index(i)?;
        self.validate_index(j)?;
        if i == j {
            return Ok(());
        }
        let key = if i < j { (i, j) } else { (j, i) };
        if weight == 0.0 {
            self.edge_weights.remove(&key);
        } else {
            self.edge_weights.insert(key, weight);
        }
        self.edges = self.edge_weights.keys().copied().collect();
        self.adjacency = Self::build_adjacency(self.nodes.len(), &self.edge_weights);
        Ok(())
    }

    pub fn remove_edge(&mut self, i: usize, j: usize) -> EngineResult<()> {
        self.validate_index(i)?;
        self.validate_index(j)?;
        let key = if i < j { (i, j) } else { (j, i) };
        self.edge_weights.remove(&key);
        self.edges = self.edge_weights.keys().copied().collect();
        self.adjacency = Self::build_adjacency(self.nodes.len(), &self.edge_weights);
        Ok(())
    }

    pub fn permute(&self, sigma: &[usize]) -> EngineResult<Self> {
        let n = self.nodes.len();
        if sigma.len() != n {
            return Err(EngineError::InvalidPermutation);
        }
        let mut sorted = sigma.to_vec();
        sorted.sort();
        if sorted != (1..=n).collect::<Vec<_>>() {
            return Err(EngineError::InvalidPermutation);
        }
        let new_nodes: Vec<Node> = sigma
            .iter()
            .map(|&idx| self.nodes[idx - 1].clone())
            .collect();
        let idx_map: HashMap<usize, usize> = sigma
            .iter()
            .enumerate()
            .map(|(new_pos, &old)| (old, new_pos + 1))
            .collect();
        let mut new_weights = HashMap::new();
        for (&(i, j), &w) in &self.edge_weights {
            let ni = *idx_map.get(&i).unwrap();
            let nj = *idx_map.get(&j).unwrap();
            let key = if ni < nj { (ni, nj) } else { (nj, ni) };
            new_weights.insert(key, w);
        }
        let edges = new_weights.keys().copied().collect();
        let adjacency = Self::build_adjacency(n, &new_weights);
        Ok(Self {
            nodes: new_nodes,
            edges,
            edge_weights: new_weights,
            adjacency,
        })
    }

    pub fn apply_permutation_matrix(&self, matrix: &[Vec<f64>]) -> EngineResult<Self> {
        let n = self.nodes.len();
        if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
            return Err(EngineError::DimensionMismatch {
                expected: n,
                actual: matrix.len(),
            });
        }
        let adjacency = self.get_adjacency_matrix();
        let mut result = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    for l in 0..n {
                        sum += matrix[i][k] * adjacency[k][l] * matrix[j][l];
                    }
                }
                result[i][j] = sum;
            }
        }
        let mut weights = HashMap::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if result[i][j].abs() > f64::EPSILON {
                    weights.insert((i + 1, j + 1), result[i][j]);
                }
            }
        }
        let edges = weights.keys().copied().collect();
        Ok(Self {
            nodes: self.nodes.clone(),
            edges,
            edge_weights: weights,
            adjacency: result,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacency_shape() {
        let g = MetatronCubeGraph::new();
        let m = g.get_adjacency_matrix();
        assert_eq!(m.len(), 13);
        assert!(m.iter().all(|row| row.len() == 13));
    }

    #[test]
    fn neighbors_of_center() {
        let g = MetatronCubeGraph::new();
        let neighbors = g.get_neighbors(1).unwrap();
        assert_eq!(neighbors.len(), 6);
        let mut labels: Vec<&str> = neighbors.iter().map(|n| n.label).collect();
        labels.sort();
        assert_eq!(labels, vec!["H1", "H2", "H3", "H4", "H5", "H6"]);
    }
}
