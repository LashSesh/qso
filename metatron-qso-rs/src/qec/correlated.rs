//! Correlated (two-pass) matching.
//!
//! Decomposing an error that flips more than two detectors (a Y error,
//! say, flips the detectors of its X part and of its Z part) makes the
//! model a graph but forgets that the parts happen together. Correlated
//! matching keeps that knowledge as reweighting rules and decodes twice
//! (the rule of PyMatching 2.4 with `enable_correlations=True`):
//!
//! 1. For every pair of graph edges `e ≠ f` that are parts of the same
//!    errors, the probability of `f` given `e`,
//!    `q = min(½, P(e and f) / P(e))`, where `P(e)` combines (as
//!    independent errors) every error with part `e` and `P(e and f)` every
//!    error with both, gives the rule "if `e` happened, `f` weighs
//!    `ln((1 − q)/q)`".
//! 2. The first pass matches as usual; the correction is the set of edges
//!    on shortest paths between the matched detection events (edges used
//!    an even number of times cancel).
//! 3. Every rule of a correction edge lowers the weight of its edge `f`
//!    (never raises it), and the second pass matches again on the
//!    reweighted graph; its prediction is the result.
//!
//! The weights are the integers of [`SparseMatchingDecoder`]. Shortest
//! paths with equal weight are chosen by the order of a Dijkstra search,
//! which can differ from PyMatching's choice, so a few shots may reweight
//! different edges.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap};

use super::decoder::MatchingGraph;
use super::dem::CorrelatedErrorModel;
use super::sparse::SparseMatchingDecoder;
use crate::error::{QsoError, Result};

const NONE: usize = usize::MAX;

/// Two-pass matching with the correlations of a decomposed model.
#[derive(Clone, Debug)]
pub struct CorrelatedMatchingDecoder {
    decoder: SparseMatchingDecoder,
    /// The node pair (detector, detector or boundary) of every edge.
    pair_of_edge: Vec<usize>,
    edges_of_pair: Vec<Vec<usize>>,
    /// Per node pair: the doubled integer weights its error implies for
    /// other pairs.
    rules: Vec<Vec<(usize, i64)>>,
    // Scratch space kept across shots.
    dist: Vec<i64>,
    back: Vec<usize>,
    touched: Vec<usize>,
    heap: BinaryHeap<Reverse<(i64, usize)>>,
    flipped: Vec<bool>,
    flipped_list: Vec<usize>,
    saved: Vec<(usize, i64)>,
}

impl CorrelatedMatchingDecoder {
    /// The decoder of a decomposed model. Every part must flip one or two
    /// detectors (parts that flip none are undetectable and left out of
    /// the rules), and no error may have probability above ½.
    pub fn new(model: &CorrelatedErrorModel) -> Result<Self> {
        let n = model.n_detectors;
        for (k, e) in model.errors.iter().enumerate() {
            let p = e.probability;
            if !(0.0..=0.5).contains(&p) {
                return Err(QsoError::invalid(
                    "model",
                    format!(
                        "error {k} has probability {p}; correlated matching takes errors of probability at most 1/2"
                    ),
                ));
            }
            for (dets, _) in &e.parts {
                let valid = match dets.as_slice() {
                    [] => true,
                    [a] => *a < n,
                    [a, b] => a < b && *b < n,
                    _ => false,
                };
                if !valid {
                    return Err(QsoError::invalid(
                        "model",
                        format!(
                            "a part of error {k} flips detectors {dets:?}; parts flip one or two ascending detectors of {n}"
                        ),
                    ));
                }
            }
        }
        let graph = MatchingGraph::from_dem(&model.to_dem())?;
        let key = |a: usize, b: usize| (a.min(b), a.max(b));
        let mut pair_index: HashMap<(usize, usize), usize> = HashMap::new();
        let mut pair_of_edge = Vec::with_capacity(graph.edges().len());
        let mut edges_of_pair: Vec<Vec<usize>> = Vec::new();
        for (k, e) in graph.edges().iter().enumerate() {
            let next = pair_index.len();
            let pid = *pair_index.entry(key(e.a, e.b)).or_insert(next);
            if pid == edges_of_pair.len() {
                edges_of_pair.push(Vec::new());
            }
            edges_of_pair[pid].push(k);
            pair_of_edge.push(pid);
        }
        let pairs = edges_of_pair.len();
        // Probabilities of every pair, and of every two pairs together.
        let xor = |a: f64, b: f64| a * (1.0 - b) + b * (1.0 - a);
        let mut marginal = vec![0.0; pairs];
        let mut joint: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        for e in &model.errors {
            if e.probability == 0.0 {
                continue;
            }
            let ids: Vec<usize> = e
                .parts
                .iter()
                .filter_map(|(dets, _)| match dets.as_slice() {
                    [a] => pair_index.get(&key(*a, n)).copied(),
                    [a, b] => pair_index.get(&key(*a, *b)).copied(),
                    _ => None,
                })
                .collect();
            for (i, &x) in ids.iter().enumerate() {
                marginal[x] = xor(marginal[x], e.probability);
                for &y in &ids[i + 1..] {
                    if x != y {
                        for k in [(x, y), (y, x)] {
                            let j = joint.entry(k).or_insert(0.0);
                            *j = xor(*j, e.probability);
                        }
                    }
                }
            }
        }
        let decoder = SparseMatchingDecoder::new(graph);
        let scale = decoder.scale;
        let mut rules = vec![Vec::new(); pairs];
        for ((x, y), pj) in joint {
            if marginal[x] == 0.0 {
                continue;
            }
            let q = (pj / marginal[x]).min(0.5);
            let w = (-q).ln_1p() - q.ln();
            rules[x].push((y, 2 * (w * scale).round() as i64));
        }
        let nodes = decoder.graph().n_detectors() + 1;
        Ok(CorrelatedMatchingDecoder {
            decoder,
            pair_of_edge,
            edges_of_pair,
            rules,
            dist: vec![i64::MAX; nodes],
            back: vec![NONE; nodes],
            touched: Vec::new(),
            heap: BinaryHeap::new(),
            flipped: vec![false; pairs],
            flipped_list: Vec::new(),
            saved: Vec::new(),
        })
    }

    /// The matching graph (every part an edge).
    pub fn graph(&self) -> &MatchingGraph {
        self.decoder.graph()
    }

    /// Predicted observable flips for the fired detectors (a detector
    /// listed twice cancels).
    pub fn decode(&mut self, fired: &[usize]) -> Result<u64> {
        let pairs = self.decoder.decode_pairs(fired)?;
        let n = self.decoder.graph().n_detectors();
        for (u, v) in pairs {
            let target = if v == u32::MAX { n } else { v as usize };
            self.path(u as usize, target);
        }
        for i in 0..self.flipped_list.len() {
            let pid = self.flipped_list[i];
            if !self.flipped[pid] {
                continue;
            }
            for &(other, w) in &self.rules[pid] {
                for &e in &self.edges_of_pair[other] {
                    if w < self.decoder.weight[e] {
                        self.saved.push((e, self.decoder.weight[e]));
                        self.decoder.weight[e] = w;
                    }
                }
            }
        }
        let result = self.decoder.decode(fired);
        // Restore the weights (latest first: an edge may have been lowered
        // twice) and the scratch space.
        while let Some((e, w)) = self.saved.pop() {
            self.decoder.weight[e] = w;
        }
        for &pid in &self.flipped_list {
            self.flipped[pid] = false;
        }
        self.flipped_list.clear();
        result
    }

    /// Flip the pairs on a shortest path from detector `source` to
    /// `target` (a detector, or the boundary node), not through the
    /// boundary.
    fn path(&mut self, source: usize, target: usize) {
        let graph = self.decoder.graph();
        let boundary = graph.n_detectors();
        self.dist[source] = 0;
        self.touched.push(source);
        self.heap.push(Reverse((0, source)));
        while let Some(Reverse((d, v))) = self.heap.pop() {
            if v == target {
                break;
            }
            if d > self.dist[v] || v == boundary {
                continue;
            }
            for &e in &graph.adjacency[v] {
                let edge = &graph.edges()[e];
                let w = if edge.a == v { edge.b } else { edge.a };
                let nd = d.saturating_add(self.decoder.weight[e]);
                if nd < self.dist[w] {
                    if self.dist[w] == i64::MAX {
                        self.touched.push(w);
                    }
                    self.dist[w] = nd;
                    self.back[w] = e;
                    self.heap.push(Reverse((nd, w)));
                }
            }
        }
        // The matching found the pair, so a path exists.
        let mut v = target;
        while v != source && self.back[v] != NONE {
            let e = self.back[v];
            let pid = self.pair_of_edge[e];
            self.flipped[pid] ^= true;
            self.flipped_list.push(pid);
            let edge = &graph.edges()[e];
            v = if edge.a == v { edge.b } else { edge.a };
        }
        for &x in &self.touched {
            self.dist[x] = i64::MAX;
            self.back[x] = NONE;
        }
        self.touched.clear();
        self.heap.clear();
    }
}

#[cfg(test)]
mod tests;
