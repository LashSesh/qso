//! Matching graphs and a union-find decoder.
//!
//! A detector error model whose errors flip at most two detectors is a
//! graph: detectors are nodes, an error flipping one detector connects it
//! to a boundary node. Decoding finds errors that explain the fired
//! detectors; the decoder predicts which observables they flipped.
//!
//! The union-find decoder of Delfosse and Nickerson (Quantum 5, 595
//! (2021)), with weighted growth (Huang, Newman and Brown, Phys. Rev. A
//! 102, 012419 (2020)): clusters around fired detectors grow along edges,
//! whose lengths are the weights ln((1 − p)/p), until each holds an even
//! number of fired detectors or touches the boundary; a spanning forest of
//! each cluster is then peeled from the leaves to choose the correction.
//! Growth is event-driven: every round advances all growing clusters to
//! the next edge they complete, so weights are used exactly, not rounded
//! to steps. Union-find runs in almost linear time but is not a
//! minimum-weight decoder, and with unequal weights it can fail on some
//! errors of weight below d/2.

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
    /// Length of the edge in growth: its weight ln((1 − p)/p), or 0 for
    /// p ≥ 1/2.
    length: f64,
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

impl MatchingGraph {
    /// The matching graph of a detector error model; every error must
    /// flip at most two detectors (see the `decompose` option of
    /// [`detector_error_model`](super::dem::detector_error_model)).
    pub fn from_dem(dem: &DetectorErrorModel) -> Result<Self> {
        let n = dem.n_detectors;
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
            if a >= n || b > n || a == b {
                return Err(QsoError::invalid(
                    "dem",
                    format!("an error flips detectors {a} and {b} of {n}"),
                ));
            }
            let p = e.probability;
            if !(0.0..=1.0).contains(&p) {
                return Err(QsoError::invalid(
                    "dem",
                    format!("{p} is not a probability"),
                ));
            }
            let length = if p >= 0.5 { 0.0 } else { ((1.0 - p) / p).ln() };
            edges.push(MatchingEdge {
                a,
                b,
                probability: p,
                observables: e.observables,
                length,
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
    /// Growth covered so far, and growing clusters at each end this round.
    support: Vec<f64>,
    rate: Vec<u32>,
    frontier: Vec<usize>,
    /// Nodes whose edges are all grown or inside their cluster; they stay
    /// so, and growth skips them.
    interior: Vec<bool>,
    grown: Vec<bool>,
    /// Grown edges at each node.
    grown_at: Vec<Vec<usize>>,
    /// Fired detectors; peeling consumes them.
    defect: Vec<bool>,
    /// Traversal marks: node `v` is seen when `seen[v] == generation`.
    seen: Vec<u32>,
    generation: u32,
    touched_nodes: Vec<usize>,
    touched_edges: Vec<usize>,
    /// Scratch space kept across shots.
    active: Vec<usize>,
    newly: Vec<usize>,
    order: Vec<(usize, usize)>,
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
            support: vec![0.0; e],
            rate: vec![0; e],
            frontier: Vec::new(),
            interior: vec![false; nodes],
            grown: vec![false; e],
            grown_at: vec![Vec::new(); nodes],
            defect: vec![false; nodes],
            seen: vec![0; nodes],
            generation: 0,
            touched_nodes: Vec::new(),
            touched_edges: Vec::new(),
            active: Vec::new(),
            newly: Vec::new(),
            order: Vec::new(),
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
        // Move the members but keep both allocations for later shots.
        let mut moved = std::mem::take(&mut self.members[small]);
        self.members[big].extend_from_slice(&moved);
        moved.clear();
        self.members[small] = moved;
        self.odd[big] ^= self.odd[small];
        self.boundary[big] |= self.boundary[small];
        big
    }

    /// Predicted observable flips for the fired detectors.
    pub fn decode(&mut self, fired: &[usize]) -> Result<u64> {
        let n = self.graph.n_detectors;
        if let Some(&d) = fired.iter().find(|&&d| d >= n) {
            return Err(QsoError::invalid(
                "fired",
                format!("detector {d} does not exist ({n} detectors)"),
            ));
        }
        for &d in fired {
            // A detector listed twice cancels.
            self.defect[d] ^= true;
            self.odd[d] ^= true;
            self.touched_nodes.push(d);
        }
        let mut active = std::mem::take(&mut self.active);
        let mut newly = std::mem::take(&mut self.newly);
        active.clear();
        active.extend(fired.iter().copied().filter(|&d| self.odd[d]));
        active.sort_unstable();
        active.dedup();
        let mut frontier = std::mem::take(&mut self.frontier);
        // Growth: every odd cluster that does not touch the boundary grows
        // along each edge leaving it at unit rate (an edge between two
        // growing clusters at rate 2). Each round advances to the next
        // completed edge; its ends merge.
        while !active.is_empty() {
            frontier.clear();
            for &root in &active {
                for i in 0..self.members[root].len() {
                    let v = self.members[root][i];
                    if self.interior[v] {
                        continue;
                    }
                    let mut outward = false;
                    for k in 0..self.graph.adjacency[v].len() {
                        let e = self.graph.adjacency[v][k];
                        if self.grown[e] {
                            continue;
                        }
                        let edge = &self.graph.edges[e];
                        let other = if edge.a == v { edge.b } else { edge.a };
                        if self.find(other) == root {
                            continue;
                        }
                        outward = true;
                        if self.rate[e] == 0 {
                            frontier.push(e);
                        }
                        self.rate[e] += 1;
                    }
                    self.interior[v] = !outward;
                }
            }
            let progress = !frontier.is_empty();
            let left = |d: &Self, e: usize| {
                (d.graph.edges[e].length - d.support[e]).max(0.0) / f64::from(d.rate[e])
            };
            let delta = frontier
                .iter()
                .map(|&e| left(self, e))
                .fold(f64::INFINITY, f64::min);
            for &e in &frontier {
                if self.support[e] == 0.0 {
                    self.touched_edges.push(e);
                }
                if left(self, e) <= delta {
                    self.grown[e] = true;
                    self.support[e] = self.graph.edges[e].length;
                    newly.push(e);
                } else {
                    self.support[e] += delta * f64::from(self.rate[e]);
                }
                self.rate[e] = 0;
            }
            if !progress {
                // A cluster without edges to grow: no error explains it.
                (self.active, self.newly, self.frontier) = (active, newly, frontier);
                self.reset();
                return Err(QsoError::invalid(
                    "fired",
                    "no combination of errors in the graph explains these detectors",
                ));
            }
            for e in newly.drain(..) {
                let (a, b) = (self.graph.edges[e].a, self.graph.edges[e].b);
                self.touched_nodes.extend([a, b]);
                self.grown_at[a].push(e);
                self.grown_at[b].push(e);
                self.union(a, b);
            }
            for r in &mut active {
                *r = self.find(*r);
            }
            active.sort_unstable();
            active.dedup();
            active.retain(|&r| self.odd[r] && !self.boundary[r]);
        }
        (self.active, self.newly, self.frontier) = (active, newly, frontier);
        let correction = self.peel();
        self.reset();
        Ok(correction)
    }

    /// Peeling: a spanning forest of the grown edges of each cluster,
    /// rooted at the boundary if the cluster needs it; leaves first.
    fn peel(&mut self) -> u64 {
        let boundary_node = self.graph.n_detectors;
        let mut prediction = 0u64;
        let mut roots = std::mem::take(&mut self.active);
        roots.clear();
        for i in 0..self.touched_nodes.len() {
            let v = self.touched_nodes[i];
            roots.push(self.find(v));
        }
        roots.sort_unstable();
        roots.dedup();
        for &root in &roots {
            let members = std::mem::take(&mut self.members[root]);
            let mut done = false;
            // A cluster that holds an even number of fired detectors does
            // not need the boundary: peel it without, if every part of it
            // still holds an even number.
            if members.contains(&boundary_node)
                && members.iter().filter(|&&u| self.defect[u]).count() % 2 == 0
            {
                let parts = self.components(&members, boundary_node);
                if parts
                    .iter()
                    .all(|p| p.iter().filter(|&&u| self.defect[u]).count() % 2 == 0)
                {
                    for part in parts {
                        prediction ^= self.peel_tree(part[0], Some(boundary_node));
                    }
                    done = true;
                }
            }
            if !done {
                let start = if members.contains(&boundary_node) {
                    boundary_node
                } else {
                    members[0]
                };
                prediction ^= self.peel_tree(start, None);
            }
            self.members[root] = members;
        }
        self.active = roots;
        prediction
    }

    fn next_generation(&mut self) -> u32 {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.seen.iter_mut().for_each(|s| *s = 0);
            self.generation = 1;
        }
        self.generation
    }

    /// Connected parts of `members` over grown edges, avoiding `skip`.
    fn components(&mut self, members: &[usize], skip: usize) -> Vec<Vec<usize>> {
        let g = self.next_generation();
        self.seen[skip] = g;
        let mut parts = Vec::new();
        for &start in members {
            if self.seen[start] == g {
                continue;
            }
            self.seen[start] = g;
            let mut part = vec![start];
            let mut i = 0;
            while i < part.len() {
                let v = part[i];
                i += 1;
                for &e in &self.grown_at[v] {
                    let edge = &self.graph.edges[e];
                    let w = if edge.a == v { edge.b } else { edge.a };
                    if self.seen[w] != g {
                        self.seen[w] = g;
                        part.push(w);
                    }
                }
            }
            parts.push(part);
        }
        parts
    }

    /// Peel a breadth-first spanning tree of grown edges from `root`
    /// (avoiding `skip`), leaves first: a fired leaf takes the edge to its
    /// parent into the correction and passes its parity up. Returns the
    /// observables the correction flips.
    fn peel_tree(&mut self, root: usize, skip: Option<usize>) -> u64 {
        let g = self.next_generation();
        if let Some(s) = skip {
            self.seen[s] = g;
        }
        self.seen[root] = g;
        let mut order = std::mem::take(&mut self.order);
        order.clear();
        order.push((root, usize::MAX));
        let mut i = 0;
        while i < order.len() {
            let v = order[i].0;
            i += 1;
            for &e in &self.grown_at[v] {
                let edge = &self.graph.edges[e];
                let w = if edge.a == v { edge.b } else { edge.a };
                if self.seen[w] != g {
                    self.seen[w] = g;
                    order.push((w, e));
                }
            }
        }
        let mut prediction = 0;
        for &(v, e) in order.iter().skip(1).rev() {
            if self.defect[v] {
                let edge = &self.graph.edges[e];
                let parent = if edge.a == v { edge.b } else { edge.a };
                self.defect[v] = false;
                self.defect[parent] ^= true;
                prediction ^= edge.observables;
            }
        }
        self.order = order;
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
            self.interior[v] = false;
            self.grown_at[v].clear();
        }
        for &e in &self.touched_edges {
            self.support[e] = 0.0;
            self.grown[e] = false;
        }
        self.touched_nodes.clear();
        self.touched_edges.clear();
    }
}
