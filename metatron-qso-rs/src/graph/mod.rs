//! Undirected weighted graphs.
//!
//! [`Graph`] is the substrate for quantum walks, spatial search and QAOA.
//! [`generators`] provides standard families (including the five Platonic
//! solids), [`metatron`] the Fruit-of-Life / Metatron's-Cube family, and
//! [`automorphism`] computes symmetry groups exactly.

pub mod automorphism;
pub mod generators;
pub mod metatron;

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::linalg::{CsrMatrix, RMatrix, eigh_real};

pub use automorphism::{AutomorphismGroup, automorphism_group};

/// Simple undirected graph with real edge weights (no self loops, no
/// multi-edges) and optional 3-D node positions for plotting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    name: String,
    n: usize,
    /// Sorted adjacency lists `(neighbour, weight)`.
    adj: Vec<Vec<(usize, f64)>>,
    /// Edges `(u, v, w)` with `u < v`, in insertion order.
    edges: Vec<(usize, usize, f64)>,
    positions: Option<Vec<[f64; 3]>>,
}

impl Graph {
    /// Empty graph on `n` nodes.
    pub fn new(n: usize) -> Self {
        Self {
            name: format!("graph({n})"),
            n,
            adj: vec![Vec::new(); n],
            edges: Vec::new(),
            positions: None,
        }
    }

    /// Graph with unit-weight `edges`.
    pub fn from_edges(n: usize, edges: &[(usize, usize)]) -> Result<Self> {
        let mut g = Self::new(n);
        for &(u, v) in edges {
            g.add_edge(u, v)?;
        }
        Ok(g)
    }

    /// Graph with weighted `edges`.
    pub fn from_weighted_edges(n: usize, edges: &[(usize, usize, f64)]) -> Result<Self> {
        let mut g = Self::new(n);
        for &(u, v, w) in edges {
            g.add_weighted_edge(u, v, w)?;
        }
        Ok(g)
    }

    /// Add an edge of weight 1.
    pub fn add_edge(&mut self, u: usize, v: usize) -> Result<&mut Self> {
        self.add_weighted_edge(u, v, 1.0)
    }

    /// Add an edge of weight `w` (finite and non-zero).
    pub fn add_weighted_edge(&mut self, u: usize, v: usize, w: f64) -> Result<&mut Self> {
        for x in [u, v] {
            if x >= self.n {
                return Err(QsoError::NodeOutOfRange {
                    node: x,
                    n_nodes: self.n,
                });
            }
        }
        let invalid = |reason: &str| QsoError::InvalidEdge {
            u,
            v,
            reason: reason.to_string(),
        };
        if u == v {
            return Err(invalid("self loops are not allowed"));
        }
        if !w.is_finite() || w == 0.0 {
            return Err(invalid("weight must be finite and non-zero"));
        }
        if self.has_edge(u, v) {
            return Err(invalid("edge already present"));
        }
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        self.edges.push((a, b, w));
        for (x, y) in [(u, v), (v, u)] {
            let list = &mut self.adj[x];
            let pos = list.partition_point(|&(z, _)| z < y);
            list.insert(pos, (y, w));
        }
        Ok(self)
    }

    /// Set a descriptive name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Attach node coordinates (one per node).
    pub fn with_positions(mut self, positions: Vec<[f64; 3]>) -> Result<Self> {
        if positions.len() != self.n {
            return Err(QsoError::DimensionMismatch {
                expected: self.n,
                actual: positions.len(),
            });
        }
        self.positions = Some(positions);
        Ok(self)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn num_nodes(&self) -> usize {
        self.n
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Edges `(u, v, weight)` with `u < v`.
    pub fn edges(&self) -> &[(usize, usize, f64)] {
        &self.edges
    }

    pub fn positions(&self) -> Option<&[[f64; 3]]> {
        self.positions.as_deref()
    }

    /// Sorted neighbours of `u` with edge weights.
    pub fn neighbors(&self, u: usize) -> &[(usize, f64)] {
        &self.adj[u]
    }

    pub fn degree(&self, u: usize) -> usize {
        self.adj[u].len()
    }

    /// Sum of incident edge weights.
    pub fn weighted_degree(&self, u: usize) -> f64 {
        self.adj[u].iter().map(|&(_, w)| w).sum()
    }

    pub fn degrees(&self) -> Vec<usize> {
        (0..self.n).map(|u| self.degree(u)).collect()
    }

    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        self.edge_weight(u, v).is_some()
    }

    pub fn edge_weight(&self, u: usize, v: usize) -> Option<f64> {
        if u >= self.n || v >= self.n {
            return None;
        }
        let list = &self.adj[u];
        list.binary_search_by(|&(z, _)| z.cmp(&v))
            .ok()
            .map(|k| list[k].1)
    }

    /// Whether every edge has weight exactly 1.
    pub fn is_unweighted(&self) -> bool {
        self.edges.iter().all(|&(_, _, w)| w == 1.0)
    }

    /// Common degree if the graph is regular.
    pub fn regular_degree(&self) -> Option<usize> {
        if self.n == 0 {
            return None;
        }
        let d = self.degree(0);
        (0..self.n).all(|u| self.degree(u) == d).then_some(d)
    }

    /// Hop distances from `source` (`None` for unreachable nodes).
    pub fn distances_from(&self, source: usize) -> Vec<Option<usize>> {
        let mut dist = vec![None; self.n];
        if source >= self.n {
            return dist;
        }
        dist[source] = Some(0);
        let mut queue = VecDeque::from([source]);
        while let Some(u) = queue.pop_front() {
            let d = dist[u].expect("visited");
            for &(v, _) in &self.adj[u] {
                if dist[v].is_none() {
                    dist[v] = Some(d + 1);
                    queue.push_back(v);
                }
            }
        }
        dist
    }

    /// Connected components (each sorted), ordered by smallest node.
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let mut seen = vec![false; self.n];
        let mut comps = Vec::new();
        for s in 0..self.n {
            if seen[s] {
                continue;
            }
            let mut comp = Vec::new();
            let mut queue = VecDeque::from([s]);
            seen[s] = true;
            while let Some(u) = queue.pop_front() {
                comp.push(u);
                for &(v, _) in &self.adj[u] {
                    if !seen[v] {
                        seen[v] = true;
                        queue.push_back(v);
                    }
                }
            }
            comp.sort_unstable();
            comps.push(comp);
        }
        comps
    }

    pub fn is_connected(&self) -> bool {
        self.n <= 1 || self.connected_components().len() == 1
    }

    /// Largest hop distance (`None` if disconnected or empty).
    pub fn diameter(&self) -> Option<usize> {
        if self.n == 0 || !self.is_connected() {
            return None;
        }
        (0..self.n)
            .map(|s| {
                self.distances_from(s)
                    .into_iter()
                    .map(|d| d.expect("connected"))
                    .max()
                    .unwrap_or(0)
            })
            .max()
    }

    /// Two-colourability.
    pub fn is_bipartite(&self) -> bool {
        let mut side = vec![None; self.n];
        for s in 0..self.n {
            if side[s].is_some() {
                continue;
            }
            side[s] = Some(false);
            let mut queue = VecDeque::from([s]);
            while let Some(u) = queue.pop_front() {
                let su = side[u].expect("coloured");
                for &(v, _) in &self.adj[u] {
                    match side[v] {
                        None => {
                            side[v] = Some(!su);
                            queue.push_back(v);
                        }
                        Some(sv) if sv == su => return false,
                        _ => {}
                    }
                }
            }
        }
        true
    }

    /// Number of triangles.
    pub fn triangle_count(&self) -> usize {
        let mut count = 0;
        for &(u, v, _) in &self.edges {
            // Common neighbours w > v (each triangle counted once).
            let (a, b) = (&self.adj[u], &self.adj[v]);
            let (mut i, mut j) = (0, 0);
            while i < a.len() && j < b.len() {
                match a[i].0.cmp(&b[j].0) {
                    std::cmp::Ordering::Less => i += 1,
                    std::cmp::Ordering::Greater => j += 1,
                    std::cmp::Ordering::Equal => {
                        if a[i].0 > v {
                            count += 1;
                        }
                        i += 1;
                        j += 1;
                    }
                }
            }
        }
        count
    }

    /// Dense weighted adjacency matrix `A`.
    pub fn adjacency_matrix(&self) -> RMatrix {
        let mut a = RMatrix::zeros(self.n, self.n);
        for &(u, v, w) in &self.edges {
            a[(u, v)] = w;
            a[(v, u)] = w;
        }
        a
    }

    /// Dense combinatorial Laplacian `L = D − A`.
    pub fn laplacian_matrix(&self) -> RMatrix {
        let mut l = -self.adjacency_matrix();
        for u in 0..self.n {
            l[(u, u)] = self.weighted_degree(u);
        }
        l
    }

    /// Sparse adjacency matrix.
    pub fn adjacency_csr(&self) -> CsrMatrix {
        let triplets: Vec<(usize, usize, f64)> = self
            .edges
            .iter()
            .flat_map(|&(u, v, w)| [(u, v, w), (v, u, w)])
            .collect();
        CsrMatrix::from_triplets(self.n, &triplets).expect("valid graph indices")
    }

    /// Sparse Laplacian.
    pub fn laplacian_csr(&self) -> CsrMatrix {
        let mut triplets: Vec<(usize, usize, f64)> = self
            .edges
            .iter()
            .flat_map(|&(u, v, w)| [(u, v, -w), (v, u, -w)])
            .collect();
        triplets.extend((0..self.n).map(|u| (u, u, self.weighted_degree(u))));
        CsrMatrix::from_triplets(self.n, &triplets).expect("valid graph indices")
    }

    /// Eigenvalues of `A` (ascending).
    pub fn adjacency_spectrum(&self) -> Vec<f64> {
        eigh_real(&self.adjacency_matrix()).0
    }

    /// Eigenvalues of `L` (ascending).
    pub fn laplacian_spectrum(&self) -> Vec<f64> {
        eigh_real(&self.laplacian_matrix()).0
    }

    /// Second-smallest Laplacian eigenvalue (Fiedler value); positive iff
    /// the graph is connected.
    pub fn algebraic_connectivity(&self) -> f64 {
        self.laplacian_spectrum().get(1).copied().unwrap_or(0.0)
    }

    /// Copy with nodes renamed by `perm` (node `u` becomes `perm[u]`).
    pub fn relabeled(&self, perm: &[usize]) -> Result<Graph> {
        if perm.len() != self.n {
            return Err(QsoError::DimensionMismatch {
                expected: self.n,
                actual: perm.len(),
            });
        }
        let mut seen = vec![false; self.n];
        for &p in perm {
            if p >= self.n || seen[p] {
                return Err(QsoError::invalid("perm", "not a permutation"));
            }
            seen[p] = true;
        }
        let mut g = Graph::new(self.n).with_name(self.name.clone());
        for &(u, v, w) in &self.edges {
            g.add_weighted_edge(perm[u], perm[v], w)?;
        }
        if let Some(pos) = &self.positions {
            let mut new_pos = vec![[0.0; 3]; self.n];
            for (u, p) in pos.iter().enumerate() {
                new_pos[perm[u]] = *p;
            }
            g.positions = Some(new_pos);
        }
        Ok(g)
    }

    /// Symmetry group (see [`automorphism_group`]).
    pub fn automorphism_group(&self) -> AutomorphismGroup {
        automorphism_group(self)
    }

    /// Summary statistics.
    pub fn summary(&self) -> GraphSummary {
        let degrees = self.degrees();
        GraphSummary {
            name: self.name.clone(),
            nodes: self.n,
            edges: self.num_edges(),
            min_degree: degrees.iter().copied().min().unwrap_or(0),
            max_degree: degrees.iter().copied().max().unwrap_or(0),
            regular: self.regular_degree().is_some(),
            connected: self.is_connected(),
            bipartite: self.is_bipartite(),
            diameter: self.diameter(),
            triangles: self.triangle_count(),
            algebraic_connectivity: if self.n > 1 {
                self.algebraic_connectivity()
            } else {
                0.0
            },
        }
    }
}

/// Structural summary of a [`Graph`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphSummary {
    pub name: String,
    pub nodes: usize,
    pub edges: usize,
    pub min_degree: usize,
    pub max_degree: usize,
    pub regular: bool,
    pub connected: bool,
    pub bipartite: bool,
    pub diameter: Option<usize>,
    pub triangles: usize,
    pub algebraic_connectivity: f64,
}

#[cfg(test)]
mod tests {
    use super::generators::*;
    use super::*;

    #[test]
    fn construction_and_validation() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1)
            .unwrap()
            .add_weighted_edge(2, 1, 2.5)
            .unwrap();
        assert!(g.add_edge(1, 0).is_err());
        assert!(g.add_edge(3, 3).is_err());
        assert!(g.add_edge(0, 4).is_err());
        assert!(g.add_weighted_edge(0, 3, f64::NAN).is_err());
        assert_eq!(g.edge_weight(1, 2), Some(2.5));
        assert_eq!(g.neighbors(1), &[(0, 1.0), (2, 2.5)]);
        assert!(!g.is_connected());
        assert_eq!(g.connected_components(), vec![vec![0, 1, 2], vec![3]]);
        assert_eq!(g.diameter(), None);
    }

    #[test]
    fn structural_invariants_of_standard_graphs() {
        let k5 = complete(5);
        assert_eq!(k5.num_edges(), 10);
        assert_eq!(k5.triangle_count(), 10);
        assert_eq!(k5.diameter(), Some(1));
        assert!(!k5.is_bipartite());
        let c6 = cycle(6);
        assert!(c6.is_bipartite());
        assert_eq!(c6.diameter(), Some(3));
        assert_eq!(c6.regular_degree(), Some(2));
        let p = petersen();
        assert_eq!(
            (p.num_nodes(), p.num_edges(), p.triangle_count()),
            (10, 15, 0)
        );
        assert_eq!(p.diameter(), Some(2));
    }

    #[test]
    fn laplacian_spectra_match_closed_forms() {
        // L(K_n): 0 once, n with multiplicity n−1.
        let s = complete(7).laplacian_spectrum();
        assert!(s[0].abs() < 1e-12 && s[1..].iter().all(|&x| (x - 7.0).abs() < 1e-10));
        // L(C_n): 2 − 2cos(2πk/n).
        let n = 9;
        let mut expected: Vec<f64> = (0..n)
            .map(|k| 2.0 - 2.0 * (2.0 * std::f64::consts::PI * k as f64 / n as f64).cos())
            .collect();
        expected.sort_by(f64::total_cmp);
        for (a, b) in cycle(n).laplacian_spectrum().iter().zip(&expected) {
            assert!((a - b).abs() < 1e-10);
        }
        // Sparse and dense agree.
        let g = petersen();
        assert_eq!(g.laplacian_csr().to_dense(), g.laplacian_matrix());
    }
}
