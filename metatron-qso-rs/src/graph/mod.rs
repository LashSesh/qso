//! Undirected weighted graphs.
//!
//! [`Graph`] is the substrate for quantum walks, spatial search and QAOA.
//! [`generators`] provides standard families (the five Platonic solids,
//! distance-regular families such as hypercubes, Hamming, Johnson and
//! Kneser graphs, Paley graphs, circulants and Cayley graphs of `Z₂ᵈ`),
//! [`metatron`] the Fruit-of-Life / Metatron's-Cube family,
//! [`automorphism`] computes symmetry groups exactly, [`canonical`] gives
//! canonical forms and isomorphisms, and [`io`] reads and writes edge lists
//! and GraphML.

pub mod automorphism;
pub mod canonical;
pub mod generators;
pub mod io;
pub mod metatron;

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::linalg::{CsrMatrix, RMatrix, eigh_real};

pub use automorphism::{AutomorphismGroup, automorphism_group};

/// Simple undirected graph with real edge weights (no self loops, no
/// multi-edges) and optional 3-D node positions for plotting.
///
/// Deserialisation rebuilds the graph from its edge list with the same
/// checks as [`Graph::add_weighted_edge`] and verifies the stored adjacency
/// lists against it, so a graph read from untrusted JSON is well-formed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "GraphData")]
pub struct Graph {
    name: String,
    n: usize,
    /// Sorted adjacency lists `(neighbour, weight)`.
    adj: Vec<Vec<(usize, f64)>>,
    /// Edges `(u, v, w)` with `u < v`, in insertion order.
    edges: Vec<(usize, usize, f64)>,
    positions: Option<Vec<[f64; 3]>>,
}

/// Unvalidated serialised form of a [`Graph`].
#[derive(Deserialize)]
struct GraphData {
    name: String,
    n: usize,
    adj: Vec<Vec<(usize, f64)>>,
    edges: Vec<(usize, usize, f64)>,
    positions: Option<Vec<[f64; 3]>>,
}

impl TryFrom<GraphData> for Graph {
    type Error = QsoError;

    fn try_from(data: GraphData) -> Result<Self> {
        // The adjacency lists were read from the input, so this bounds the
        // allocation below by the input size.
        if data.adj.len() != data.n {
            return Err(QsoError::DimensionMismatch {
                expected: data.n,
                actual: data.adj.len(),
            });
        }
        let g = Graph::from_weighted_edges(data.n, &data.edges)?.with_name(data.name);
        if g.adj != data.adj {
            return Err(QsoError::invalid(
                "adj",
                "adjacency lists do not match the edge list",
            ));
        }
        match data.positions {
            Some(p) => g.with_positions(p),
            None => Ok(g),
        }
    }
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
        let weighted: Vec<_> = edges.iter().map(|&(u, v)| (u, v, 1.0)).collect();
        Self::from_weighted_edges(n, &weighted)
    }

    /// Graph with weighted `edges`, in `O(m log m)` time whatever their order.
    pub fn from_weighted_edges(n: usize, edges: &[(usize, usize, f64)]) -> Result<Self> {
        let mut g = Self::new(n);
        for &(u, v, w) in edges {
            g.check_edge(u, v, w)?;
            g.adj[u].push((v, w));
            g.adj[v].push((u, w));
        }
        for (u, list) in g.adj.iter_mut().enumerate() {
            list.sort_unstable_by_key(|&(z, _)| z);
            if let Some(pair) = list.windows(2).find(|p| p[0].0 == p[1].0) {
                return Err(QsoError::InvalidEdge {
                    u,
                    v: pair[0].0,
                    reason: "edge already present".to_string(),
                });
            }
        }
        g.edges = edges
            .iter()
            .map(|&(u, v, w)| (u.min(v), u.max(v), w))
            .collect();
        Ok(g)
    }

    /// Add an edge of weight 1.
    pub fn add_edge(&mut self, u: usize, v: usize) -> Result<&mut Self> {
        self.add_weighted_edge(u, v, 1.0)
    }

    /// Add an edge of weight `w` (finite and non-zero).
    pub fn add_weighted_edge(&mut self, u: usize, v: usize, w: f64) -> Result<&mut Self> {
        self.check_edge(u, v, w)?;
        if self.has_edge(u, v) {
            return Err(QsoError::InvalidEdge {
                u,
                v,
                reason: "edge already present".to_string(),
            });
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

    /// Range, loop and weight checks shared by the edge constructors.
    fn check_edge(&self, u: usize, v: usize, w: f64) -> Result<()> {
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
        Ok(())
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

    /// Length of the shortest cycle, or `None` for a forest.
    ///
    /// Breadth-first search from every vertex; `O(n·m)`.
    pub fn girth(&self) -> Option<usize> {
        let n = self.num_nodes();
        let mut best: Option<usize> = None;
        let mut dist = vec![usize::MAX; n];
        let mut parent = vec![usize::MAX; n];
        let mut queue = std::collections::VecDeque::new();
        for s in 0..n {
            dist.iter_mut().for_each(|d| *d = usize::MAX);
            dist[s] = 0;
            parent[s] = usize::MAX;
            queue.clear();
            queue.push_back(s);
            while let Some(u) = queue.pop_front() {
                // Every cycle closed from here on has length ≥ 2·dist[u].
                if best.is_some_and(|b| 2 * dist[u] >= b) {
                    break;
                }
                for &(v, _) in &self.adj[u] {
                    if dist[v] == usize::MAX {
                        dist[v] = dist[u] + 1;
                        parent[v] = u;
                        queue.push_back(v);
                    } else if parent[u] != v {
                        let len = dist[u] + dist[v] + 1;
                        best = Some(best.map_or(len, |b| b.min(len)));
                    }
                }
            }
        }
        best
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

    /// Cartesian product `G □ H`: node `(g, h)` has index `g·|H| + h`; it is
    /// joined to `(g', h)` for every edge `g g'` and to `(g, h')` for every
    /// edge `h h'`, with the same weights.
    pub fn cartesian_product(&self, other: &Graph) -> Result<Graph> {
        let (n, m) = (self.n, other.n);
        let nodes = n
            .checked_mul(m)
            .ok_or_else(|| QsoError::invalid("graph", "product too large"))?;
        generators::check_generated_size(
            nodes as u128,
            self.edges.len() as u128 * m as u128 + other.edges.len() as u128 * n as u128,
        )?;
        let mut edges = Vec::with_capacity(self.edges.len() * m + other.edges.len() * n);
        for &(u, v, w) in &self.edges {
            edges.extend((0..m).map(|h| (u * m + h, v * m + h, w)));
        }
        for &(u, v, w) in &other.edges {
            edges.extend((0..n).map(|x| (x * m + u, x * m + v, w)));
        }
        Ok(Graph::from_weighted_edges(nodes, &edges)?
            .with_name(format!("{} □ {}", self.name, other.name)))
    }

    /// Parameters `(n, k, λ, μ)` if the graph is strongly regular: unweighted,
    /// `k`-regular, every two adjacent nodes have `λ` common neighbours and
    /// every two distinct non-adjacent nodes `μ`. Complete and empty graphs
    /// are excluded, as usual. Takes `O(n² k)` time and `O(n)` memory.
    pub fn strongly_regular_parameters(&self) -> Option<(usize, usize, usize, usize)> {
        let k = self.regular_degree()?;
        if !self.is_unweighted() || k == 0 || k + 1 == self.n {
            return None;
        }
        // Count 2-paths from each node: O(n·k²) instead of O(n²·k).
        let (mut lambda, mut mu) = (None, None);
        let check = |slot: &mut Option<usize>, common: usize| match *slot {
            None => {
                *slot = Some(common);
                true
            }
            Some(c) => c == common,
        };
        let mut paths = vec![0usize; self.n];
        let mut adjacent = vec![false; self.n];
        let mut touched = Vec::new();
        for u in 0..self.n {
            for &(v, _) in self.neighbors(u) {
                adjacent[v] = true;
                for &(w, _) in self.neighbors(v) {
                    if w != u {
                        if paths[w] == 0 {
                            touched.push(w);
                        }
                        paths[w] += 1;
                    }
                }
            }
            let mut ok = self
                .neighbors(u)
                .iter()
                .all(|&(v, _)| check(&mut lambda, paths[v]));
            let mut distant = 0;
            for &w in &touched {
                if !adjacent[w] {
                    distant += 1;
                    ok &= check(&mut mu, paths[w]);
                }
            }
            if distant < self.n - 1 - k {
                ok &= check(&mut mu, 0);
            }
            if !ok {
                return None;
            }
            for &w in &touched {
                paths[w] = 0;
            }
            touched.clear();
            for &(v, _) in self.neighbors(u) {
                adjacent[v] = false;
            }
        }
        Some((self.n, k, lambda?, mu?))
    }

    /// Canonical labelling (see [`canonical`]): `labels[u]` is the canonical
    /// index of node `u`.
    pub fn canonical_labelling(&self) -> Vec<usize> {
        canonical::canonical_labelling(self)
    }

    /// The graph relabelled canonically; isomorphic graphs give equal
    /// edge lists.
    pub fn canonical_form(&self) -> Graph {
        let mut g = self
            .relabeled(&self.canonical_labelling())
            .expect("a labelling is a permutation");
        g.edges.sort_by_key(|&(u, v, _)| (u, v));
        g
    }

    /// Whether `other` is isomorphic to this graph (weights included).
    pub fn is_isomorphic(&self, other: &Graph) -> bool {
        canonical::isomorphism(self, other).is_some()
    }

    /// An isomorphism to `other` (node `u` ↦ `iso[u]`), if one exists.
    pub fn isomorphism(&self, other: &Graph) -> Option<Vec<usize>> {
        canonical::isomorphism(self, other)
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
    fn json_round_trip_and_validation() {
        let g = super::metatron::metatron_cube();
        let text = serde_json::to_string(&g).unwrap();
        assert_eq!(serde_json::from_str::<Graph>(&text).unwrap(), g);
        let mut w = Graph::new(3);
        w.add_weighted_edge(2, 0, -1.5).unwrap();
        let text = serde_json::to_string(&w).unwrap();
        assert_eq!(serde_json::from_str::<Graph>(&text).unwrap(), w);
        // Inconsistent or invalid data is rejected, not stored.
        let base = |n: &str, adj: &str, edges: &str, pos: &str| {
            format!(r#"{{"name":"g","n":{n},"adj":{adj},"edges":{edges},"positions":{pos}}}"#)
        };
        for bad in [
            base("1000000000000", "[]", "[]", "null"),
            base("2", "[[],[]]", "[[0,5,1.0]]", "null"),
            base("2", "[[],[]]", "[[0,0,1.0]]", "null"),
            base("2", "[[],[]]", "[[0,1,0.0]]", "null"),
            base(
                "2",
                "[[[1,1.0]],[[0,1.0]]]",
                "[[0,1,1.0],[1,0,1.0]]",
                "null",
            ),
            base("2", "[[],[]]", "[[0,1,1.0]]", "null"),
            base("2", "[[[1,1.0]],[[0,2.0]]]", "[[0,1,1.0]]", "null"),
            base("2", "[[[1,1.0]],[[0,1.0]]]", "[[0,1,1.0]]", "[[0,0,0]]"),
        ] {
            assert!(serde_json::from_str::<Graph>(&bad).is_err(), "{bad}");
        }
        let ok = base(
            "2",
            "[[[1,1.0]],[[0,1.0]]]",
            "[[1,0,1.0]]",
            "[[0,0,0],[1,0,0]]",
        );
        let g: Graph = serde_json::from_str(&ok).unwrap();
        assert_eq!(g.edges(), &[(0, 1, 1.0)]);
    }

    #[test]
    fn girth_of_standard_graphs() {
        assert_eq!(complete(4).girth(), Some(3));
        assert_eq!(hypercube(3).girth(), Some(4));
        assert_eq!(petersen().girth(), Some(5));
        assert_eq!(heawood().girth(), Some(6));
        assert!(heawood().is_bipartite() && heawood().regular_degree() == Some(3));
        assert_eq!(cycle(7).girth(), Some(7));
        assert_eq!(path(6).girth(), None);
        assert_eq!(star(5).girth(), None);
        assert_eq!(grid_2d(3, 4, false).girth(), Some(4));
        assert_eq!(platonic(PlatonicSolid::Dodecahedron).girth(), Some(5));
        // Brute force on random graphs: girth = shortest cycle through any
        // edge = 1 + shortest u–v path avoiding the edge itself.
        let mut rng = crate::Rng::seed_from_u64(31);
        for _ in 0..30 {
            let g = erdos_renyi(9, 0.3, &mut rng).unwrap();
            let expected = g
                .edges()
                .iter()
                .filter_map(|&(u, v, _)| {
                    let mut h = Graph::new(g.num_nodes());
                    for &(a, b, _) in g.edges() {
                        if (a, b) != (u, v) {
                            h.add_edge(a, b).unwrap();
                        }
                    }
                    h.distances_from(u)[v].map(|d| d + 1)
                })
                .min();
            assert_eq!(g.girth(), expected);
        }
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
