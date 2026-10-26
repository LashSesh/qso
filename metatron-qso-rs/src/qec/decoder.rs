//! Matching graphs and a union-find decoder.
//!
//! A detector error model whose errors flip at most two detectors is a
//! graph: detectors are nodes, an error flipping one detector connects it
//! to a boundary node. Decoding finds errors that explain the fired
//! detectors; the decoder predicts which observables they flipped.
//!
//! The union-find decoder of Delfosse and Nickerson (Quantum 5, 595
//! (2021)), with weighted growth (Huang, Newman and Brown, Phys. Rev. A
//! 102, 012419 (2020)): clusters around fired detectors grow along edges
//! in steps until each holds an even number of fired detectors or touches
//! the boundary; a spanning forest of each cluster is then peeled from the
//! leaves to choose the correction. It corrects every error of weight at
//! most (d − 1)/2 on a code of distance d, in almost linear time.

use super::dem::DetectorErrorModel;
use crate::error::{QsoError, Result};

/// An edge of a matching graph.
#[derive(Clone, Debug, PartialEq)]
pub struct MatchingEdge {
    pub a: usize,
    /// The other detector, or the boundary node (index `n_detectors`).
    pub b: usize,
    pub probability: f64,
    /// Observables the error flips.
    pub observables: u64,
    /// Growth steps to cover the edge (from its weight ln((1 − p)/p)).
    steps: u32,
}

/// Detectors as nodes (plus a boundary node), errors as edges.
#[derive(Clone, Debug, PartialEq)]
pub struct MatchingGraph {
    n_detectors: usize,
    edges: Vec<MatchingEdge>,
    adjacency: Vec<Vec<usize>>,
    /// Errors that flip observables but no detector: undetectable.
    undetectable: Vec<(f64, u64)>,
}

/// Growth resolution: the lightest edge takes this many steps.
const STEPS_PER_LIGHTEST: f64 = 2.0;
/// Heaviest edge in steps (caps the growth rounds).
const MAX_STEPS: f64 = 64.0;

impl MatchingGraph {
    /// The matching graph of a detector error model; every error must
    /// flip at most two detectors (see the `decompose` option of
    /// [`detector_error_model`](super::dem::detector_error_model)).
    pub fn from_dem(dem: &DetectorErrorModel) -> Result<Self> {
        let n = dem.n_detectors;
        let weight = |p: f64| ((1.0 - p) / p).ln().max(1e-6);
        let lightest = dem
            .errors
            .iter()
            .filter(|e| !e.detectors.is_empty())
            .map(|e| weight(e.probability.min(0.5)))
            .fold(f64::INFINITY, f64::min);
        let mut edges = Vec::new();
        let mut undetectable = Vec::new();
        for e in &dem.errors {
            let (a, b) = match e.detectors.as_slice() {
                [] => {
                    undetectable.push((e.probability, e.observables));
                    continue;
                }
                [a] => (*a, n),
                [a, b] => (*a, *b),
                more => {
                    return Err(QsoError::invalid(
                        "dem",
                        format!(
                            "an error flips {} detectors; a matching graph needs at most two",
                            more.len()
                        ),
                    ));
                }
            };
            let w = weight(e.probability.min(0.5));
            let steps = (STEPS_PER_LIGHTEST * w / lightest)
                .round()
                .clamp(1.0, MAX_STEPS) as u32;
            edges.push(MatchingEdge {
                a,
                b,
                probability: e.probability,
                observables: e.observables,
                steps,
            });
        }
        let mut adjacency = vec![Vec::new(); n + 1];
        for (k, e) in edges.iter().enumerate() {
            adjacency[e.a].push(k);
            adjacency[e.b].push(k);
        }
        Ok(MatchingGraph {
            n_detectors: n,
            edges,
            adjacency,
            undetectable,
        })
    }

    pub fn n_detectors(&self) -> usize {
        self.n_detectors
    }

    pub fn edges(&self) -> &[MatchingEdge] {
        &self.edges
    }

    /// Errors that flip observables without firing a detector, with their
    /// probabilities: no decoder can correct them.
    pub fn undetectable(&self) -> &[(f64, u64)] {
        &self.undetectable
    }
}

/// Union-find decoder over a [`MatchingGraph`]; reusable across shots.
#[derive(Clone, Debug)]
pub struct UnionFindDecoder {
    graph: MatchingGraph,
    parent: Vec<usize>,
    /// Per cluster root: number of fired detectors mod 2, boundary, nodes.
    odd: Vec<bool>,
    boundary: Vec<bool>,
    members: Vec<Vec<usize>>,
    support: Vec<u32>,
    grown: Vec<bool>,
    defect: Vec<bool>,
    touched_nodes: Vec<usize>,
    touched_edges: Vec<usize>,
}

impl UnionFindDecoder {
    pub fn new(graph: MatchingGraph) -> Self {
        let nodes = graph.n_detectors + 1;
        let e = graph.edges.len();
        UnionFindDecoder {
            parent: (0..nodes).collect(),
            odd: vec![false; nodes],
            boundary: (0..nodes).map(|v| v == graph.n_detectors).collect(),
            members: (0..nodes).map(|v| vec![v]).collect(),
            support: vec![0; e],
            grown: vec![false; e],
            defect: vec![false; nodes],
            touched_nodes: Vec::new(),
            touched_edges: Vec::new(),
            graph,
        }
    }

    pub fn graph(&self) -> &MatchingGraph {
        &self.graph
    }

    fn find(&mut self, mut v: usize) -> usize {
        while self.parent[v] != v {
            self.parent[v] = self.parent[self.parent[v]];
            v = self.parent[v];
        }
        v
    }

    fn touch(&mut self, v: usize) {
        self.touched_nodes.push(v);
    }

    fn union(&mut self, a: usize, b: usize) -> usize {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return ra;
        }
        let (big, small) = if self.members[ra].len() >= self.members[rb].len() {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big;
        let moved = std::mem::take(&mut self.members[small]);
        self.members[big].extend(moved);
        self.odd[big] ^= self.odd[small];
        self.boundary[big] |= self.boundary[small];
        big
    }

    /// Predicted observable flips for the fired detectors.
    pub fn decode(&mut self, fired: &[usize]) -> Result<u64> {
        let n = self.graph.n_detectors;
        for &d in fired {
            if d >= n {
                return Err(QsoError::invalid(
                    "fired",
                    format!("detector {d} does not exist ({n} detectors)"),
                ));
            }
            // A detector listed twice cancels.
            self.defect[d] ^= true;
            self.odd[d] ^= true;
            self.touch(d);
        }
        let mut active: Vec<usize> = fired.iter().copied().filter(|&d| self.odd[d]).collect();
        active.sort_unstable();
        active.dedup();
        // Growth.
        while !active.is_empty() {
            let mut newly = Vec::new();
            let mut progress = false;
            for &root in &active {
                let nodes = self.members[root].clone();
                for v in nodes {
                    for k in 0..self.graph.adjacency[v].len() {
                        let e = self.graph.adjacency[v][k];
                        if self.grown[e] {
                            continue;
                        }
                        let (ea, eb, steps) = {
                            let edge = &self.graph.edges[e];
                            (edge.a, edge.b, edge.steps)
                        };
                        let other = if ea == v { eb } else { ea };
                        if self.find(other) == root {
                            continue;
                        }
                        if self.support[e] == 0 {
                            self.touched_edges.push(e);
                        }
                        self.support[e] += 1;
                        progress = true;
                        if self.support[e] >= steps {
                            self.grown[e] = true;
                            newly.push(e);
                        }
                    }
                }
            }
            if !progress {
                // A cluster without edges to grow: no error explains it.
                self.reset();
                return Err(QsoError::invalid(
                    "fired",
                    "no combination of errors in the graph explains these detectors",
                ));
            }
            for e in newly {
                let (a, b) = (self.graph.edges[e].a, self.graph.edges[e].b);
                self.touch(a);
                self.touch(b);
                self.union(a, b);
            }
            let mut next: Vec<usize> = active.iter().map(|&r| self.find(r)).collect();
            next.sort_unstable();
            next.dedup();
            next.retain(|&r| self.odd[r] && !self.boundary[r]);
            active = next;
        }
        // Peeling: a spanning forest of the grown edges of each cluster,
        // rooted at the boundary if the cluster touches it; leaves first.
        let correction = self.peel();
        self.reset();
        Ok(correction)
    }

    fn peel(&mut self) -> u64 {
        let boundary_node = self.graph.n_detectors;
        let mut grown_adj: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for &e in &self.touched_edges {
            if self.grown[e] {
                let edge = &self.graph.edges[e];
                grown_adj.entry(edge.a).or_default().push(e);
                grown_adj.entry(edge.b).or_default().push(e);
            }
        }
        let mut syndrome: std::collections::HashMap<usize, bool> = self
            .touched_nodes
            .iter()
            .map(|&v| (v, self.defect[v]))
            .collect();
        let mut prediction = 0u64;
        let mut done = std::collections::HashSet::new();
        let mut nodes: Vec<usize> = self.touched_nodes.clone();
        nodes.sort_unstable();
        nodes.dedup();
        for v in nodes {
            let root = self.find(v);
            if !done.insert(root) {
                continue;
            }
            let members = self.members[root].clone();
            let with_boundary = members.contains(&boundary_node);
            // A cluster that holds an even number of fired detectors does
            // not need the boundary: peel it without, if every part of it
            // still holds an even number.
            if with_boundary {
                let defects = members.iter().filter(|&&u| self.defect[u]).count();
                if defects % 2 == 0 {
                    let parts = components(&members, &grown_adj, &self.graph, Some(boundary_node));
                    if parts
                        .iter()
                        .all(|p| p.iter().filter(|&&u| self.defect[u]).count() % 2 == 0)
                    {
                        for part in parts {
                            prediction ^= peel_tree(
                                part[0],
                                &grown_adj,
                                &self.graph,
                                Some(boundary_node),
                                &mut syndrome,
                            );
                        }
                        continue;
                    }
                }
                prediction ^=
                    peel_tree(boundary_node, &grown_adj, &self.graph, None, &mut syndrome);
            } else {
                prediction ^= peel_tree(members[0], &grown_adj, &self.graph, None, &mut syndrome);
            }
        }
        prediction
    }

    fn reset(&mut self) {
        for &v in &self.touched_nodes {
            self.parent[v] = v;
            self.odd[v] = false;
            self.boundary[v] = v == self.graph.n_detectors;
            self.members[v].clear();
            self.members[v].push(v);
            self.defect[v] = false;
        }
        for &e in &self.touched_edges {
            self.support[e] = 0;
            self.grown[e] = false;
        }
        self.touched_nodes.clear();
        self.touched_edges.clear();
    }
}

/// Connected parts of `members` over grown edges, avoiding node `skip`.
fn components(
    members: &[usize],
    adj: &std::collections::HashMap<usize, Vec<usize>>,
    graph: &MatchingGraph,
    skip: Option<usize>,
) -> Vec<Vec<usize>> {
    let mut seen = std::collections::HashSet::new();
    let mut parts = Vec::new();
    for &start in members {
        if Some(start) == skip || !seen.insert(start) {
            continue;
        }
        let mut part = vec![start];
        let mut i = 0;
        while i < part.len() {
            let v = part[i];
            i += 1;
            for &e in adj.get(&v).into_iter().flatten() {
                let edge = &graph.edges[e];
                let w = if edge.a == v { edge.b } else { edge.a };
                if Some(w) != skip && seen.insert(w) {
                    part.push(w);
                }
            }
        }
        parts.push(part);
    }
    parts
}

/// Peel a breadth-first spanning tree of grown edges from `root`
/// (avoiding node `skip`), leaves first: a fired leaf takes the edge to
/// its parent into the correction and passes its parity up. Returns the
/// observables the correction flips.
fn peel_tree(
    root: usize,
    adj: &std::collections::HashMap<usize, Vec<usize>>,
    graph: &MatchingGraph,
    skip: Option<usize>,
    syndrome: &mut std::collections::HashMap<usize, bool>,
) -> u64 {
    let mut order = vec![(root, usize::MAX)];
    let mut seen = std::collections::HashSet::from([root]);
    let mut i = 0;
    while i < order.len() {
        let v = order[i].0;
        i += 1;
        for &e in adj.get(&v).into_iter().flatten() {
            let edge = &graph.edges[e];
            let w = if edge.a == v { edge.b } else { edge.a };
            if Some(w) != skip && seen.insert(w) {
                order.push((w, e));
            }
        }
    }
    let mut prediction = 0;
    for &(v, e) in order.iter().skip(1).rev() {
        if syndrome.get(&v).copied().unwrap_or(false) {
            let edge = &graph.edges[e];
            let parent = if edge.a == v { edge.b } else { edge.a };
            syndrome.insert(v, false);
            let p = syndrome.entry(parent).or_insert(false);
            *p = !*p;
            prediction ^= edge.observables;
        }
    }
    prediction
}
