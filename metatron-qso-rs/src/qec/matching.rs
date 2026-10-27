//! Minimum-weight perfect matching decoding.
//!
//! Each shot's detection events are matched in pairs or to the boundary
//! so that the total weight of the corrections is minimal: shortest paths
//! in the matching graph (Dijkstra) give the cost of every pair, and
//! Edmonds' blossom algorithm finds the best matching (Edmonds, Can. J.
//! Math. 17, 449 (1965); Dennis, Kitaev, Landahl and Preskill, J. Math.
//! Phys. 43, 4452 (2002)).
//!
//! The boundary is handled without copies of the events: matching i and j
//! saves `b_i + b_j − d_ij` against sending both to the boundary, where
//! `b` is the distance to the boundary, so a maximum-weight matching on
//! the positive savings is a minimum-weight correction. Pairs farther
//! apart than their boundary distances never save anything, so each
//! Dijkstra stops at radius `b_i + max_j b_j`. Weights are ln((1 − p)/p),
//! 0 for p ≥ 1/2, rounded to integers with the largest edge at 2²⁰, so
//! the blossom algorithm computes exactly; "minimum" refers to these
//! integer weights. A shot may weigh at most [`MAX_MATCHING_PAIRS`] pairs
//! of events.
//!
//! Where several corrections weigh the same, which one is returned can
//! depend on the order in which pairs were found, and so on which
//! distance rows were cached by earlier shots; every one is a
//! minimum-weight correction, and a seeded run is reproducible.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::decoder::MatchingGraph;
use crate::error::{QsoError, Result};

const NONE: usize = usize::MAX;
/// Most pairs of detection events one shot may weigh (about 200 MiB with
/// the blossom algorithm's arrays).
pub const MAX_MATCHING_PAIRS: usize = 1 << 22;
/// Largest stand-in cost of events without a boundary path; beyond it the
/// integer arithmetic of the blossom algorithm could overflow.
const MAX_STRANDED_COST: i64 = i64::MAX / 16;

/// Distance and path observables from one detector to every detector.
type Row = Box<[(i64, u64)]>;
/// Most cached distances (entries of rows of the distance table; 32 MiB).
const MAX_CACHED_DISTANCES: usize = 1 << 21;
/// The largest edge weight after rounding.
const WEIGHT_SCALE: f64 = (1u64 << 20) as f64;

/// Minimum-weight perfect matching decoder over a [`MatchingGraph`];
/// reusable across shots.
#[derive(Clone, Debug)]
pub struct MatchingDecoder {
    graph: MatchingGraph,
    /// Integer weight of every edge.
    pub(crate) weight: Vec<i64>,
    /// Integer weight per unit of length (ln((1 − p)/p)).
    scale: f64,
    /// Distance of every detector to the boundary (`i64::MAX` if it has
    /// no path there), and the observables along that path.
    to_boundary: Vec<i64>,
    boundary_flips: Vec<u64>,
    /// Scratch space kept across shots.
    dist: Vec<i64>,
    flips: Vec<u64>,
    index: Vec<usize>,
    touched: Vec<usize>,
    heap: BinaryHeap<Reverse<(i64, usize)>>,
    /// Distances and path observables from a detector to every detector,
    /// computed when it first fires, as long as the cache has room.
    rows: Vec<Option<Row>>,
    cached: usize,
    pub(crate) cache_capacity: usize,
}

impl MatchingDecoder {
    pub fn new(graph: MatchingGraph) -> Self {
        let longest = graph.edges().iter().map(|e| e.length).fold(0.0, f64::max);
        let scale = if longest > 0.0 {
            WEIGHT_SCALE / longest
        } else {
            1.0
        };
        let weight = graph
            .edges()
            .iter()
            .map(|e| (e.length * scale).round() as i64)
            .collect();
        let n = graph.n_detectors();
        let mut d = MatchingDecoder {
            weight,
            scale,
            to_boundary: Vec::new(),
            boundary_flips: Vec::new(),
            dist: vec![i64::MAX; n + 1],
            flips: vec![0; n + 1],
            index: vec![NONE; n + 1],
            touched: Vec::new(),
            heap: BinaryHeap::new(),
            rows: vec![None; n],
            cached: 0,
            cache_capacity: MAX_CACHED_DISTANCES,
            graph,
        };
        // One search from the boundary gives every detector's distance.
        d.search(n, i64::MAX, true, |_, _, _| i64::MAX);
        d.to_boundary = d.dist[..n].to_vec();
        d.boundary_flips = d.flips[..n].to_vec();
        d.clear();
        d
    }

    pub fn graph(&self) -> &MatchingGraph {
        &self.graph
    }

    /// Dijkstra from `source` up to distance `radius`, through the
    /// boundary node only if `through_boundary`; calls `found(node,
    /// distance, flips)` for every node settled, which returns the radius
    /// from then on (negative: stop). Leaves `dist`/`flips` set on the
    /// nodes in `touched`.
    fn search(
        &mut self,
        source: usize,
        mut radius: i64,
        through_boundary: bool,
        mut found: impl FnMut(usize, i64, u64) -> i64,
    ) {
        let boundary = self.graph.n_detectors();
        let mut heap = std::mem::take(&mut self.heap);
        heap.clear();
        self.dist[source] = 0;
        self.flips[source] = 0;
        self.touched.push(source);
        heap.push(Reverse((0i64, source)));
        while let Some(Reverse((d, v))) = heap.pop() {
            if d > radius {
                break;
            }
            if d > self.dist[v] {
                continue;
            }
            radius = found(v, d, self.flips[v]);
            if v == boundary && !through_boundary {
                continue;
            }
            for &e in &self.graph.adjacency[v] {
                let edge = &self.graph.edges()[e];
                let w = if edge.a == v { edge.b } else { edge.a };
                let nd = d.saturating_add(self.weight[e]);
                if nd > radius || nd >= self.dist[w] {
                    continue;
                }
                if self.dist[w] == i64::MAX {
                    self.touched.push(w);
                }
                self.dist[w] = nd;
                self.flips[w] = self.flips[v] ^ edge.observables;
                heap.push(Reverse((nd, w)));
            }
        }
        self.heap = heap;
    }

    /// Make sure the distance row of detector `v` is cached, if there is
    /// room; returns whether it is.
    fn cache_row(&mut self, v: usize) -> bool {
        if self.rows[v].is_some() {
            return true;
        }
        let n = self.graph.n_detectors();
        if self.cached + n > self.cache_capacity {
            return false;
        }
        self.search(v, i64::MAX, false, |_, _, _| i64::MAX);
        let row = (0..n).map(|w| (self.dist[w], self.flips[w])).collect();
        self.clear();
        self.rows[v] = Some(row);
        self.cached += n;
        true
    }

    fn clear(&mut self) {
        for &v in &self.touched {
            self.dist[v] = i64::MAX;
            self.flips[v] = 0;
        }
        self.touched.clear();
    }

    /// Predicted observable flips for the fired detectors (a detector
    /// listed twice cancels).
    pub fn decode(&mut self, fired: &[usize]) -> Result<u64> {
        Ok(self.decode_weighted(fired)?.0)
    }

    /// The prediction and the weight of the correction, `Σ ln((1 − p)/p)`
    /// over its errors (errors with `p ≥ ½` count 0; up to the rounding of
    /// the weights to integers).
    pub fn decode_with_weight(&mut self, fired: &[usize]) -> Result<(u64, f64)> {
        let (prediction, weight) = self.decode_weighted(fired)?;
        Ok((prediction, weight as f64 / self.scale))
    }

    /// The prediction and the integer weight of the correction.
    pub(crate) fn decode_weighted(&mut self, fired: &[usize]) -> Result<(u64, i64)> {
        let n = self.graph.n_detectors();
        if let Some(&d) = fired.iter().find(|&&d| d >= n) {
            return Err(QsoError::invalid(
                "fired",
                format!("detector {d} does not exist ({n} detectors)"),
            ));
        }
        let mut events: Vec<usize> = Vec::with_capacity(fired.len());
        for &d in fired {
            if self.index[d] == NONE {
                self.index[d] = events.len();
                events.push(d);
            } else {
                // Listed twice: remove it again.
                let k = self.index[d];
                events.swap_remove(k);
                self.index[d] = NONE;
                if k < events.len() {
                    self.index[events[k]] = k;
                }
            }
        }
        let result = self.match_events(&events);
        for &d in &events {
            self.index[d] = NONE;
        }
        result
    }

    fn match_events(&mut self, events: &[usize]) -> Result<(u64, i64)> {
        let k = events.len();
        if k == 0 {
            return Ok((0, 0));
        }
        // Pairs (i < j) whose path is shorter than both boundary paths.
        let mut pairs: Vec<(usize, usize, i64, u64)> = Vec::new();
        // The searches read these while they borrow `self` mutably.
        let index = std::mem::take(&mut self.index);
        let to_boundary = std::mem::take(&mut self.to_boundary);
        let bound: Vec<i64> = events.iter().map(|&d| to_boundary[d]).collect();
        let mut open = vec![false; k];
        for i in 0..k {
            let bi = bound[i];
            let stranded = bi == i64::MAX;
            // The later events that can still save something: those on
            // the same side of having a boundary path. The search stops
            // when none is left, or beyond b_i + max b_j over them.
            let mut left = 0;
            for j in i + 1..k {
                open[j] = (bound[j] == i64::MAX) == stranded;
                left += usize::from(open[j]);
            }
            if left == 0 {
                continue;
            }
            if self.cache_row(events[i]) {
                let row = self.rows[events[i]].as_deref().expect("cached");
                for j in (i + 1..k).filter(|&j| open[j]) {
                    let (d, f) = row[events[j]];
                    if d < i64::MAX && (stranded || d < bi.saturating_add(bound[j])) {
                        pairs.push((i, j, d, f));
                    }
                }
                continue;
            }
            let reach = |open: &[bool]| {
                if stranded {
                    return i64::MAX;
                }
                let far = (i + 1..k).filter(|&j| open[j]).map(|j| bound[j]).max();
                far.map_or(-1, |b| bi.saturating_add(b))
            };
            let mut radius = reach(&open);
            self.search(events[i], radius, false, |v, d, f| {
                let j = if v < index.len() { index[v] } else { NONE };
                if j != NONE && j > i && open[j] {
                    if stranded || d < bi.saturating_add(bound[j]) {
                        pairs.push((i, j, d, f));
                    }
                    open[j] = false;
                    left -= 1;
                    radius = if left == 0 { -1 } else { reach(&open) };
                }
                radius
            });
            self.clear();
            if pairs.len() > MAX_MATCHING_PAIRS {
                break;
            }
        }
        self.index = index;
        self.to_boundary = to_boundary;
        if pairs.len() > MAX_MATCHING_PAIRS {
            return Err(QsoError::invalid(
                "fired",
                format!(
                    "{k} detection events need more than {MAX_MATCHING_PAIRS} pairs for exact matching"
                ),
            ));
        }
        // Events without a boundary path pay `stranded_cost` there: more
        // than all their pairs together (each pair counts at most the
        // longest path from its first event), so they are matched
        // whenever possible.
        let mut longest = vec![0i64; k];
        for &(i, _, d, _) in &pairs {
            longest[i] = longest[i].max(d);
        }
        let stranded_cost = (0..k)
            .filter(|&i| bound[i] == i64::MAX)
            .fold(1i64, |acc, i| acc.saturating_add(longest[i]));
        if stranded_cost > MAX_STRANDED_COST {
            return Err(QsoError::invalid(
                "fired",
                "the detection events are too far apart for exact integer matching",
            ));
        }
        let cost = |d: usize| {
            let b = self.to_boundary[d];
            if b == i64::MAX { stranded_cost } else { b }
        };
        let edges: Vec<(usize, usize, i64)> = pairs
            .iter()
            .map(|&(i, j, d, _)| (i, j, cost(events[i]) + cost(events[j]) - d))
            .filter(|&(_, _, s)| s > 0)
            .collect();
        let mate = max_weight_matching(k, &edges);
        let mut prediction = 0;
        let mut total = 0i64;
        for &(i, j, d, f) in &pairs {
            if mate[i] == j {
                prediction ^= f;
                total += d;
            }
        }
        for i in 0..k {
            if mate[i] == NONE {
                let b = self.to_boundary[events[i]];
                if b == i64::MAX {
                    return Err(QsoError::invalid(
                        "fired",
                        "no set of errors explains the detection events",
                    ));
                }
                prediction ^= self.boundary_flips[events[i]];
                total += b;
            }
        }
        Ok((prediction, total))
    }
}

/// A maximum-weight matching of the graph on vertices `0..n` with edges
/// `(i, j, weight)` (integer weights; edges of weight ≤ 0 are never
/// needed): `mate[v]` is v's partner or `usize::MAX`.
///
/// Edmonds' blossom algorithm with dual variables in O(n³), after Joris
/// van Rantwijk's implementation (the one NetworkX uses), with integer
/// arithmetic throughout.
pub(crate) fn max_weight_matching(n: usize, edges: &[(usize, usize, i64)]) -> Vec<usize> {
    if edges.is_empty() {
        return vec![NONE; n];
    }
    let mut m = Blossoms::new(n, edges);
    m.solve();
    m.mate
        .iter()
        .map(|&p| if p == NONE { NONE } else { m.endpoint[p] })
        .collect()
}

/// State of the blossom algorithm. Vertices are `0..n`, blossoms
/// `n..2n`; edge `k` has endpoints `2k` and `2k + 1`. Vertex duals and
/// slacks are doubled, so all arithmetic stays in integers.
struct Blossoms<'a> {
    n: usize,
    edges: &'a [(usize, usize, i64)],
    endpoint: Vec<usize>,
    neighbend: Vec<Vec<usize>>,
    /// The remote endpoint of each vertex's matched edge.
    mate: Vec<usize>,
    /// 0 free, 1 S, 2 T (5 marks a breadcrumb in `scan_blossom`).
    label: Vec<i8>,
    labelend: Vec<usize>,
    inblossom: Vec<usize>,
    blossomparent: Vec<usize>,
    blossomchilds: Vec<Vec<usize>>,
    blossombase: Vec<usize>,
    blossomendps: Vec<Vec<usize>>,
    bestedge: Vec<usize>,
    blossombestedges: Vec<Option<Vec<usize>>>,
    unusedblossoms: Vec<usize>,
    dualvar: Vec<i64>,
    allowedge: Vec<bool>,
    queue: Vec<usize>,
}

impl<'a> Blossoms<'a> {
    fn new(n: usize, edges: &'a [(usize, usize, i64)]) -> Self {
        let maxweight = edges.iter().map(|e| e.2).max().unwrap_or(0).max(0);
        let mut neighbend = vec![Vec::new(); n];
        let mut endpoint = Vec::with_capacity(2 * edges.len());
        for (k, &(i, j, _)) in edges.iter().enumerate() {
            debug_assert!(i != j && i < n && j < n);
            endpoint.push(i);
            endpoint.push(j);
            neighbend[i].push(2 * k + 1);
            neighbend[j].push(2 * k);
        }
        Blossoms {
            n,
            edges,
            endpoint,
            neighbend,
            mate: vec![NONE; n],
            label: vec![0; 2 * n],
            labelend: vec![NONE; 2 * n],
            inblossom: (0..n).collect(),
            blossomparent: vec![NONE; 2 * n],
            blossomchilds: vec![Vec::new(); 2 * n],
            blossombase: (0..n).chain(std::iter::repeat_n(NONE, n)).collect(),
            blossomendps: vec![Vec::new(); 2 * n],
            bestedge: vec![NONE; 2 * n],
            blossombestedges: vec![None; 2 * n],
            unusedblossoms: (n..2 * n).collect(),
            dualvar: std::iter::repeat_n(maxweight, n)
                .chain(std::iter::repeat_n(0, n))
                .collect(),
            allowedge: vec![false; edges.len()],
            queue: Vec::new(),
        }
    }

    fn slack(&self, k: usize) -> i64 {
        let (i, j, w) = self.edges[k];
        self.dualvar[i] + self.dualvar[j] - 2 * w
    }

    /// The vertices inside blossom (or vertex) `b`.
    fn leaves(&self, b: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut stack = vec![b];
        while let Some(t) = stack.pop() {
            if t < self.n {
                out.push(t);
            } else {
                stack.extend(self.blossomchilds[t].iter().rev());
            }
        }
        out
    }

    fn assign_label(&mut self, w: usize, t: i8, p: usize) {
        let b = self.inblossom[w];
        debug_assert!(self.label[w] == 0 && self.label[b] == 0);
        self.label[w] = t;
        self.label[b] = t;
        self.labelend[w] = p;
        self.labelend[b] = p;
        self.bestedge[w] = NONE;
        self.bestedge[b] = NONE;
        if t == 1 {
            let leaves = self.leaves(b);
            self.queue.extend(leaves);
        } else {
            // A T-blossom: its mate becomes S.
            let base = self.blossombase[b];
            let m = self.mate[base];
            self.assign_label(self.endpoint[m], 1, m ^ 1);
        }
    }

    /// Trace back from S-vertices `v` and `w`: the base of a new blossom,
    /// or `NONE` for an augmenting path.
    fn scan_blossom(&mut self, mut v: usize, mut w: usize) -> usize {
        let mut path = Vec::new();
        let mut base = NONE;
        while v != NONE {
            let mut b = self.inblossom[v];
            if self.label[b] & 4 != 0 {
                base = self.blossombase[b];
                break;
            }
            path.push(b);
            self.label[b] = 5;
            if self.labelend[b] == NONE {
                v = NONE;
            } else {
                v = self.endpoint[self.labelend[b]];
                b = self.inblossom[v];
                v = self.endpoint[self.labelend[b]];
            }
            if w != NONE {
                std::mem::swap(&mut v, &mut w);
            }
        }
        for b in path {
            self.label[b] = 1;
        }
        base
    }

    fn add_blossom(&mut self, base: usize, k: usize) {
        let (mut v, mut w, _) = self.edges[k];
        let bb = self.inblossom[base];
        let mut bv = self.inblossom[v];
        let mut bw = self.inblossom[w];
        let b = self.unusedblossoms.pop().expect("at most n blossoms");
        self.blossombase[b] = base;
        self.blossomparent[b] = NONE;
        self.blossomparent[bb] = b;
        let mut path = Vec::new();
        let mut endps = Vec::new();
        while bv != bb {
            self.blossomparent[bv] = b;
            path.push(bv);
            endps.push(self.labelend[bv]);
            v = self.endpoint[self.labelend[bv]];
            bv = self.inblossom[v];
        }
        path.push(bb);
        path.reverse();
        endps.reverse();
        endps.push(2 * k);
        while bw != bb {
            self.blossomparent[bw] = b;
            path.push(bw);
            endps.push(self.labelend[bw] ^ 1);
            w = self.endpoint[self.labelend[bw]];
            bw = self.inblossom[w];
        }
        self.label[b] = 1;
        self.labelend[b] = self.labelend[bb];
        self.dualvar[b] = 0;
        self.blossomchilds[b] = path.clone();
        self.blossomendps[b] = endps;
        for v in self.leaves(b) {
            if self.label[self.inblossom[v]] == 2 {
                // A T-vertex inside a new S-blossom becomes S.
                self.queue.push(v);
            }
            self.inblossom[v] = b;
        }
        // The least-slack edges from the new blossom to each S-blossom.
        let mut bestedgeto = vec![NONE; 2 * self.n];
        for &bv in &path {
            let lists: Vec<Vec<usize>> = match self.blossombestedges[bv].take() {
                Some(list) => vec![list],
                None => self
                    .leaves(bv)
                    .into_iter()
                    .map(|v| self.neighbend[v].iter().map(|p| p / 2).collect())
                    .collect(),
            };
            for list in lists {
                for k in list {
                    let (i, j, _) = self.edges[k];
                    let j = if self.inblossom[j] == b { i } else { j };
                    let bj = self.inblossom[j];
                    if bj != b
                        && self.label[bj] == 1
                        && (bestedgeto[bj] == NONE || self.slack(k) < self.slack(bestedgeto[bj]))
                    {
                        bestedgeto[bj] = k;
                    }
                }
            }
            self.bestedge[bv] = NONE;
        }
        let list: Vec<usize> = bestedgeto.into_iter().filter(|&k| k != NONE).collect();
        let mut best = NONE;
        for &k in &list {
            if best == NONE || self.slack(k) < self.slack(best) {
                best = k;
            }
        }
        self.bestedge[b] = best;
        self.blossombestedges[b] = Some(list);
    }

    fn expand_blossom(&mut self, b: usize, endstage: bool) {
        let childs = self.blossomchilds[b].clone();
        for &s in &childs {
            self.blossomparent[s] = NONE;
            if s < self.n {
                self.inblossom[s] = s;
            } else if endstage && self.dualvar[s] == 0 {
                self.expand_blossom(s, endstage);
            } else {
                for v in self.leaves(s) {
                    self.inblossom[v] = s;
                }
            }
        }
        if !endstage && self.label[b] == 2 {
            // Relabel the sub-blossoms from the one through which b got
            // its label to the base.
            let endps = self.blossomendps[b].clone();
            let len = childs.len() as isize;
            let at = |j: isize| j.rem_euclid(len) as usize;
            let entrychild = self.inblossom[self.endpoint[self.labelend[b] ^ 1]];
            let mut j = childs
                .iter()
                .position(|&c| c == entrychild)
                .expect("entry child") as isize;
            let (jstep, trick) = if j & 1 == 1 {
                j -= len;
                (1, 0)
            } else {
                (-1, 1)
            };
            let mut p = self.labelend[b];
            while j != 0 {
                self.label[self.endpoint[p ^ 1]] = 0;
                let q = endps[at(j - trick as isize)];
                self.label[self.endpoint[q ^ trick ^ 1]] = 0;
                self.assign_label(self.endpoint[p ^ 1], 2, p);
                self.allowedge[q / 2] = true;
                j += jstep;
                p = endps[at(j - trick as isize)] ^ trick;
                self.allowedge[p / 2] = true;
                j += jstep;
            }
            // The base T-sub-blossom, without stepping to its mate.
            let bv = childs[at(j)];
            let ep = self.endpoint[p ^ 1];
            self.label[ep] = 2;
            self.label[bv] = 2;
            self.labelend[ep] = p;
            self.labelend[bv] = p;
            self.bestedge[bv] = NONE;
            j += jstep;
            while childs[at(j)] != entrychild {
                let bv = childs[at(j)];
                if self.label[bv] == 1 {
                    j += jstep;
                    continue;
                }
                if let Some(v) = self.leaves(bv).into_iter().find(|&v| self.label[v] != 0) {
                    debug_assert!(self.label[v] == 2 && self.inblossom[v] == bv);
                    self.label[v] = 0;
                    let m = self.mate[self.blossombase[bv]];
                    self.label[self.endpoint[m]] = 0;
                    self.assign_label(v, 2, self.labelend[v]);
                }
                j += jstep;
            }
        }
        self.label[b] = -1;
        self.labelend[b] = NONE;
        self.blossomchilds[b].clear();
        self.blossomendps[b].clear();
        self.blossombase[b] = NONE;
        self.blossombestedges[b] = None;
        self.bestedge[b] = NONE;
        self.unusedblossoms.push(b);
    }

    /// Swap matched and unmatched edges along the alternating path
    /// through blossom `b` from vertex `v` to its base.
    fn augment_blossom(&mut self, b: usize, v: usize) {
        let mut t = v;
        while self.blossomparent[t] != b {
            t = self.blossomparent[t];
        }
        if t >= self.n {
            self.augment_blossom(t, v);
        }
        let childs = self.blossomchilds[b].clone();
        let endps = self.blossomendps[b].clone();
        let len = childs.len() as isize;
        let at = |j: isize| j.rem_euclid(len) as usize;
        let i = childs.iter().position(|&c| c == t).expect("child");
        let mut j = i as isize;
        let (jstep, trick) = if i & 1 == 1 {
            j -= len;
            (1, 0)
        } else {
            (-1, 1)
        };
        while j != 0 {
            j += jstep;
            let t = childs[at(j)];
            let p = endps[at(j - trick as isize)] ^ trick;
            if t >= self.n {
                self.augment_blossom(t, self.endpoint[p]);
            }
            j += jstep;
            let t = childs[at(j)];
            if t >= self.n {
                self.augment_blossom(t, self.endpoint[p ^ 1]);
            }
            self.mate[self.endpoint[p]] = p ^ 1;
            self.mate[self.endpoint[p ^ 1]] = p;
        }
        self.blossomchilds[b].rotate_left(i);
        self.blossomendps[b].rotate_left(i);
        self.blossombase[b] = self.blossombase[self.blossomchilds[b][0]];
        debug_assert_eq!(self.blossombase[b], v);
    }

    fn augment_matching(&mut self, k: usize) {
        let (v, w, _) = self.edges[k];
        for (mut s, mut p) in [(v, 2 * k + 1), (w, 2 * k)] {
            loop {
                let bs = self.inblossom[s];
                if bs >= self.n {
                    self.augment_blossom(bs, s);
                }
                self.mate[s] = p;
                if self.labelend[bs] == NONE {
                    break;
                }
                let t = self.endpoint[self.labelend[bs]];
                let bt = self.inblossom[t];
                s = self.endpoint[self.labelend[bt]];
                let j = self.endpoint[self.labelend[bt] ^ 1];
                if bt >= self.n {
                    self.augment_blossom(bt, j);
                }
                self.mate[j] = self.labelend[bt];
                p = self.labelend[bt] ^ 1;
            }
        }
    }

    fn solve(&mut self) {
        let n = self.n;
        for _ in 0..n {
            // A stage: find one augmenting path.
            self.label.fill(0);
            self.bestedge.fill(NONE);
            for b in n..2 * n {
                self.blossombestedges[b] = None;
            }
            self.allowedge.fill(false);
            self.queue.clear();
            for v in 0..n {
                if self.mate[v] == NONE && self.label[self.inblossom[v]] == 0 {
                    self.assign_label(v, 1, NONE);
                }
            }
            let mut augmented = false;
            loop {
                while !augmented {
                    let Some(v) = self.queue.pop() else { break };
                    for idx in 0..self.neighbend[v].len() {
                        let p = self.neighbend[v][idx];
                        let k = p / 2;
                        let w = self.endpoint[p];
                        if self.inblossom[v] == self.inblossom[w] {
                            continue;
                        }
                        let mut kslack = 0;
                        if !self.allowedge[k] {
                            kslack = self.slack(k);
                            if kslack <= 0 {
                                self.allowedge[k] = true;
                            }
                        }
                        let bw = self.inblossom[w];
                        if self.allowedge[k] {
                            if self.label[bw] == 0 {
                                self.assign_label(w, 2, p ^ 1);
                            } else if self.label[bw] == 1 {
                                let base = self.scan_blossom(v, w);
                                if base != NONE {
                                    self.add_blossom(base, k);
                                } else {
                                    self.augment_matching(k);
                                    augmented = true;
                                    break;
                                }
                            } else if self.label[w] == 0 {
                                // w inside a T-blossom, reached for the
                                // first time.
                                self.label[w] = 2;
                                self.labelend[w] = p ^ 1;
                            }
                        } else if self.label[bw] == 1 {
                            let b = self.inblossom[v];
                            if self.bestedge[b] == NONE || kslack < self.slack(self.bestedge[b]) {
                                self.bestedge[b] = k;
                            }
                        } else if self.label[w] == 0
                            && (self.bestedge[w] == NONE || kslack < self.slack(self.bestedge[w]))
                        {
                            self.bestedge[w] = k;
                        }
                    }
                }
                if augmented {
                    break;
                }
                // No augmenting path yet: change the duals by delta.
                let mut deltatype = 1;
                let mut delta = *self.dualvar[..n].iter().min().expect("vertices");
                let mut deltaedge = NONE;
                let mut deltablossom = NONE;
                for v in 0..n {
                    if self.label[self.inblossom[v]] == 0 && self.bestedge[v] != NONE {
                        let d = self.slack(self.bestedge[v]);
                        if d < delta {
                            delta = d;
                            deltatype = 2;
                            deltaedge = self.bestedge[v];
                        }
                    }
                }
                for b in 0..2 * n {
                    if self.blossomparent[b] == NONE
                        && self.label[b] == 1
                        && self.bestedge[b] != NONE
                    {
                        let s = self.slack(self.bestedge[b]);
                        debug_assert!(s % 2 == 0);
                        if s / 2 < delta {
                            delta = s / 2;
                            deltatype = 3;
                            deltaedge = self.bestedge[b];
                        }
                    }
                }
                for b in n..2 * n {
                    if self.blossombase[b] != NONE
                        && self.blossomparent[b] == NONE
                        && self.label[b] == 2
                        && self.dualvar[b] < delta
                    {
                        delta = self.dualvar[b];
                        deltatype = 4;
                        deltablossom = b;
                    }
                }
                for v in 0..n {
                    match self.label[self.inblossom[v]] {
                        1 => self.dualvar[v] -= delta,
                        2 => self.dualvar[v] += delta,
                        _ => {}
                    }
                }
                for b in n..2 * n {
                    if self.blossombase[b] != NONE && self.blossomparent[b] == NONE {
                        match self.label[b] {
                            1 => self.dualvar[b] += delta,
                            2 => self.dualvar[b] -= delta,
                            _ => {}
                        }
                    }
                }
                match deltatype {
                    1 => break,
                    2 => {
                        self.allowedge[deltaedge] = true;
                        let (i, j, _) = self.edges[deltaedge];
                        let i = if self.label[self.inblossom[i]] == 0 {
                            j
                        } else {
                            i
                        };
                        self.queue.push(i);
                    }
                    3 => {
                        self.allowedge[deltaedge] = true;
                        self.queue.push(self.edges[deltaedge].0);
                    }
                    _ => self.expand_blossom(deltablossom, false),
                }
            }
            if !augmented {
                break;
            }
            // End of a stage: expand S-blossoms whose dual is zero.
            for b in n..2 * n {
                if self.blossomparent[b] == NONE
                    && self.blossombase[b] != NONE
                    && self.label[b] == 1
                    && self.dualvar[b] == 0
                {
                    self.expand_blossom(b, true);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// The largest matching weight on the vertices in `mask`, by dynamic
    /// programming over subsets.
    fn brute_force(n: usize, w: &[Vec<Option<i64>>]) -> i64 {
        let mut best = vec![0i64; 1 << n];
        for mask in 1usize..1 << n {
            let v = mask.trailing_zeros() as usize;
            let rest = mask & !(1 << v);
            let mut b = best[rest];
            for u in 0..n {
                if rest & (1 << u) != 0
                    && let Some(x) = w[v][u]
                {
                    b = b.max(x + best[rest & !(1 << u)]);
                }
            }
            best[mask] = b;
        }
        best[(1 << n) - 1]
    }

    #[test]
    fn random_graphs_match_the_brute_force_optimum() {
        let mut rng = Rng::seed_from_u64(1);
        for trial in 0..4000 {
            let n = 1 + trial % 14;
            let density = [0.2, 0.5, 0.9][trial % 3];
            // Few distinct weights make ties, and nested blossoms, common.
            let range = [3u64, 10, 1000][(trial / 3) % 3];
            let mut edges = Vec::new();
            for i in 0..n {
                for j in i + 1..n {
                    if rng.uniform() < density {
                        edges.push((i, j, rng.below(range) as i64 - 1));
                    }
                }
            }
            let mut w = vec![vec![None; n]; n];
            for &(i, j, x) in &edges {
                w[i][j] = Some(x);
                w[j][i] = Some(x);
            }
            let mate = max_weight_matching(n, &edges);
            let mut total = 0;
            for v in 0..n {
                if mate[v] != NONE {
                    assert_eq!(mate[mate[v]], v, "trial {trial}");
                    let x = w[v][mate[v]].expect("matched along an edge");
                    if v < mate[v] {
                        total += x;
                    }
                }
            }
            assert_eq!(total, brute_force(n, &w), "trial {trial}: {edges:?}");
        }
    }
}
