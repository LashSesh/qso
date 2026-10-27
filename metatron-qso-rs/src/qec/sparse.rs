//! Minimum-weight perfect matching by growing regions on the detector
//! graph (the idea of sparse blossom: Higgott and Gidney, Quantum 9, 1600
//! (2025), arXiv:2303.15933).
//!
//! [`MatchingDecoder`](super::MatchingDecoder) computes shortest paths
//! between detection events and runs Edmonds' blossom algorithm on the
//! complete graph of their pairs. This decoder runs the same primal-dual
//! algorithm without that graph: every detection event owns a region of
//! the detector graph whose radius is its dual variable. Regions grow,
//! freeze or shrink at unit speed; when two regions touch, the edge
//! between their detection events is tight and the alternating trees of
//! Edmonds' algorithm change (a tree grows, two trees augment, an odd
//! cycle becomes a blossom region, a shrinking blossom shatters). When a
//! growing region reaches the boundary its tree augments to it. Only the
//! nodes near detection events are ever touched, so a shot costs time in
//! proportion to its detection events and the area their regions cover,
//! not to the square of their number.
//!
//! The dual variables are the radii; they stay feasible for the linear
//! program of minimum-weight T-joins, whose optimum is the minimum-weight
//! correction, and the final matching is tight against them, so the
//! result is optimal. Edge weights are those of
//! [`MatchingDecoder`](super::MatchingDecoder) (integers, the largest
//! 2²⁰), doubled so that two regions growing towards each other meet at
//! an integer time; the correction has the same weight as the exact
//! decoder's on every shot (the tests check this shot by shot).
//!
//! A shot's events are processed in order of time and, at equal times, in
//! the order they were scheduled, so results are reproducible.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::decoder::MatchingGraph;
use crate::error::{QsoError, Result};

const NONE: u32 = u32::MAX;
/// Largest edge weight after rounding (as in `MatchingDecoder`), doubled.
const WEIGHT_SCALE: f64 = (1u64 << 20) as f64;

/// A tight path between two detection events (or one and the boundary):
/// `u` lies in one region, `v` in the other (`v == NONE` for the
/// boundary).
#[derive(Clone, Copy, Debug)]
struct Edge {
    u: u32,
    v: u32,
    obs: u64,
    len: i64,
}

impl Edge {
    fn reversed(self) -> Self {
        Edge {
            u: self.v,
            v: self.u,
            ..self
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mate {
    None,
    Region(u32),
    Boundary,
}

#[derive(Clone, Debug)]
struct Region {
    /// Enclosing blossom, or `NONE` for a top-level region.
    parent: u32,
    /// The detection event of a trivial region (`NONE` for a blossom).
    source: u32,
    /// Children of a blossom around its odd cycle; `cycle[i]` joins child
    /// `i` to child `i + 1 (mod len)`.
    children: Vec<u32>,
    cycle: Vec<Edge>,
    /// Radius `radius0 + rate·(t − t0)`; inner regions have rate 0.
    radius0: i64,
    t0: i64,
    rate: i64,
    /// Nodes reached while this region was top-level, in order of arrival
    /// (and of their arrival radius).
    shell: Vec<u32>,
    /// Matched partner and the tight edge to it (`u` in this region).
    mate: Mate,
    mate_edge: Edge,
    /// Alternating tree: `plus` (growing) or minus (shrinking) node, the
    /// tree parent and the edge to it (`u` in this region), children.
    in_tree: bool,
    plus: bool,
    tree_parent: u32,
    parent_edge: Edge,
    tree_children: Vec<u32>,
    /// Invalidates scheduled shrink events.
    version: u32,
}

#[derive(Clone, Copy, Debug)]
struct Node {
    /// The region that reached this node (`NONE`: empty).
    region: u32,
    /// The detection event whose region reached it, the distance from it
    /// and the observables along the way.
    source: u32,
    dist: i64,
    obs: u64,
    /// Radii of the regions between the source and the top-level region
    /// (frozen while the structure stays).
    wrapped: i64,
    /// Invalidates scheduled node events.
    version: u32,
}

const EMPTY: Node = Node {
    region: NONE,
    source: NONE,
    dist: 0,
    obs: 0,
    wrapped: 0,
    version: 0,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Event {
    /// Look at a node's edges.
    Node(u32, u32),
    /// A shrinking region releases a node or reaches radius zero.
    Region(u32, u32),
}

/// Minimum-weight perfect matching by region growth on the detector
/// graph; reusable across shots.
#[derive(Clone, Debug)]
pub struct SparseMatchingDecoder {
    graph: MatchingGraph,
    /// Doubled integer weight of every edge.
    weight: Vec<i64>,
    /// Integer weight (undoubled) per unit of length.
    scale: f64,
    nodes: Vec<Node>,
    touched: Vec<u32>,
    regions: Vec<Region>,
    queue: BinaryHeap<Reverse<(i64, u64, Event)>>,
    seq: u64,
    now: i64,
    /// Trees still growing.
    trees: usize,
    /// Blossoms formed, blossoms shattered, trivial regions shrunk to
    /// zero, trees grown (for the tests).
    #[cfg(test)]
    counts: [usize; 4],
}

impl SparseMatchingDecoder {
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
            .map(|e| 2 * (e.length * scale).round() as i64)
            .collect();
        let n = graph.n_detectors();
        Self {
            weight,
            scale,
            nodes: vec![EMPTY; n],
            touched: Vec::new(),
            regions: Vec::new(),
            queue: BinaryHeap::new(),
            seq: 0,
            now: 0,
            trees: 0,
            #[cfg(test)]
            counts: [0; 4],
            graph,
        }
    }

    pub fn graph(&self) -> &MatchingGraph {
        &self.graph
    }

    /// Predicted observable flips for the fired detectors (a detector
    /// listed twice cancels).
    pub fn decode(&mut self, fired: &[usize]) -> Result<u64> {
        Ok(self.decode_weighted(fired)?.0)
    }

    /// The prediction and the weight of the correction, `Σ ln((1 − p)/p)`
    /// over its errors (up to the rounding of the weights to integers).
    pub fn decode_with_weight(&mut self, fired: &[usize]) -> Result<(u64, f64)> {
        let (prediction, weight) = self.decode_weighted(fired)?;
        Ok((prediction, weight as f64 / self.scale))
    }

    /// The prediction and the integer weight of the correction (in the
    /// units of `MatchingDecoder`).
    pub(crate) fn decode_weighted(&mut self, fired: &[usize]) -> Result<(u64, i64)> {
        let n = self.graph.n_detectors();
        if let Some(&d) = fired.iter().find(|&&d| d >= n) {
            return Err(QsoError::invalid(
                "fired",
                format!("detector {d} does not exist ({n} detectors)"),
            ));
        }
        // Remove detectors listed an even number of times.
        let mut sorted: Vec<u32> = fired.iter().map(|&d| d as u32).collect();
        sorted.sort_unstable();
        let mut events: Vec<u32> = Vec::with_capacity(sorted.len());
        let mut k = 0;
        while k < sorted.len() {
            let mut end = k;
            while end < sorted.len() && sorted[end] == sorted[k] {
                end += 1;
            }
            if (end - k) % 2 == 1 {
                events.push(sorted[k]);
            }
            k = end;
        }
        let result = self.solve(&events);
        self.reset();
        result
    }

    fn reset(&mut self) {
        for &v in &self.touched {
            self.nodes[v as usize] = EMPTY;
        }
        self.touched.clear();
        self.regions.clear();
        self.queue.clear();
        self.seq = 0;
        self.now = 0;
        self.trees = 0;
    }

    fn solve(&mut self, events: &[u32]) -> Result<(u64, i64)> {
        for &d in events {
            let r = self.regions.len() as u32;
            self.regions.push(Region {
                parent: NONE,
                source: d,
                children: Vec::new(),
                cycle: Vec::new(),
                radius0: 0,
                t0: 0,
                rate: 1,
                shell: Vec::new(),
                mate: Mate::None,
                mate_edge: NO_EDGE,
                in_tree: true,
                plus: true,
                tree_parent: NONE,
                parent_edge: NO_EDGE,
                tree_children: Vec::new(),
                version: 0,
            });
            self.nodes[d as usize] = Node {
                region: r,
                source: d,
                dist: 0,
                obs: 0,
                wrapped: 0,
                version: 0,
            };
            self.touched.push(d);
            self.trees += 1;
        }
        for &d in events {
            self.schedule_node(d);
        }
        while self.trees > 0 {
            let Some(Reverse((t, _, event))) = self.queue.pop() else {
                return Err(QsoError::invalid(
                    "fired",
                    "no set of errors explains the detection events",
                ));
            };
            debug_assert!(t >= self.now);
            self.now = t;
            match event {
                Event::Node(v, version) => {
                    if self.nodes[v as usize].version == version {
                        self.process_node(v);
                    }
                }
                Event::Region(r, version) => {
                    if self.regions[r as usize].version == version {
                        self.process_shrink(r);
                    }
                }
            }
        }
        Ok(self.extract())
    }

    fn push(&mut self, t: i64, event: Event) {
        self.seq += 1;
        self.queue.push(Reverse((t, self.seq, event)));
    }

    fn top(&self, mut r: u32) -> u32 {
        while self.regions[r as usize].parent != NONE {
            r = self.regions[r as usize].parent;
        }
        r
    }

    fn radius(&self, r: u32) -> i64 {
        let g = &self.regions[r as usize];
        g.radius0 + g.rate * (self.now - g.t0)
    }

    /// Change the growth rate of a top-level region from now on.
    fn set_rate(&mut self, r: u32, rate: i64) {
        let radius = self.radius(r);
        let g = &mut self.regions[r as usize];
        g.radius0 = radius;
        g.t0 = self.now;
        g.rate = rate;
        g.version += 1;
    }

    /// Local radius of an owned node: how far its region reaches past it.
    fn local_radius(&self, v: u32, top: u32) -> i64 {
        let node = &self.nodes[v as usize];
        self.radius(top) + node.wrapped - node.dist
    }

    /// Every node a region owns (its sources and the shells of it and its
    /// descendants).
    fn coverage(&self, r: u32, out: &mut Vec<u32>) {
        let g = &self.regions[r as usize];
        if g.source != NONE {
            out.push(g.source);
        }
        out.extend_from_slice(&g.shell);
        for &c in &g.children {
            self.coverage(c, out);
        }
    }

    /// Reschedule every node of a region, and its shrink event.
    fn reschedule_region(&mut self, r: u32) {
        let mut nodes = Vec::new();
        self.coverage(r, &mut nodes);
        for v in nodes {
            self.schedule_node(v);
        }
        self.schedule_shrink(r);
    }

    fn schedule_shrink(&mut self, r: u32) {
        let g = &self.regions[r as usize];
        if g.rate >= 0 {
            return;
        }
        let version = g.version;
        let t = match g.shell.last() {
            Some(&v) => self.now + self.local_radius(v, r),
            None => self.now + self.radius(r),
        };
        self.push(t, Event::Region(r, version));
    }

    /// The earliest time an edge of `v` becomes tight (growth into an
    /// empty node, the boundary, or another region).
    fn schedule_node(&mut self, v: u32) {
        let node = self.nodes[v as usize];
        if node.region == NONE {
            return;
        }
        let top = self.top(node.region);
        let rate = self.regions[top as usize].rate;
        let lr = self.local_radius(v, top);
        let boundary = self.graph.n_detectors();
        let mut best = i64::MAX;
        for &e in &self.graph.adjacency[v as usize] {
            let edge = &self.graph.edges()[e];
            let w = if edge.a == v as usize { edge.b } else { edge.a };
            let weight = self.weight[e];
            if w == boundary || self.nodes[w].region == NONE {
                if rate > 0 {
                    debug_assert!(weight >= lr, "node {v} overshot an edge");
                    best = best.min(self.now + weight - lr);
                }
                continue;
            }
            let other = self.top(self.nodes[w].region);
            if other == top {
                continue;
            }
            let both = rate + self.regions[other as usize].rate;
            if both > 0 {
                let gap = weight - lr - self.local_radius(w as u32, other);
                debug_assert!(gap >= 0 && gap % both == 0, "gap {gap} rate {both}");
                best = best.min(self.now + gap / both);
            }
        }
        let node = &mut self.nodes[v as usize];
        node.version = node.version.wrapping_add(1);
        let version = node.version;
        if best < i64::MAX {
            self.push(best, Event::Node(v, version));
        }
    }

    fn process_node(&mut self, v: u32) {
        let node = self.nodes[v as usize];
        if node.region == NONE {
            return;
        }
        let top = self.top(node.region);
        let rate = self.regions[top as usize].rate;
        let lr = self.local_radius(v, top);
        let boundary = self.graph.n_detectors();
        let adjacency = std::mem::take(&mut self.graph.adjacency[v as usize]);
        let mut action = None;
        for &e in &adjacency {
            let edge = &self.graph.edges()[e];
            let w = if edge.a == v as usize { edge.b } else { edge.a };
            let weight = self.weight[e];
            if w == boundary {
                if rate > 0 && lr == weight {
                    action = Some((
                        top,
                        NONE,
                        Edge {
                            u: node.source,
                            v: NONE,
                            obs: node.obs ^ edge.observables,
                            len: node.dist + weight,
                        },
                    ));
                    break;
                }
                continue;
            }
            let target = self.nodes[w];
            if target.region == NONE {
                if rate > 0 && lr == weight {
                    self.nodes[w] = Node {
                        region: top,
                        source: node.source,
                        dist: node.dist + weight,
                        obs: node.obs ^ edge.observables,
                        wrapped: node.wrapped,
                        version: target.version,
                    };
                    self.touched.push(w as u32);
                    self.regions[top as usize].shell.push(w as u32);
                    self.schedule_node(w as u32);
                }
                continue;
            }
            let other = self.top(target.region);
            if other == top {
                continue;
            }
            let both = rate + self.regions[other as usize].rate;
            if both > 0 && lr + self.local_radius(w as u32, other) == weight {
                action = Some((
                    top,
                    other,
                    Edge {
                        u: node.source,
                        v: target.source,
                        obs: node.obs ^ edge.observables ^ target.obs,
                        len: node.dist + weight + target.dist,
                    },
                ));
                break;
            }
        }
        self.graph.adjacency[v as usize] = adjacency;
        match action {
            Some((r, NONE, edge)) => self.hit_boundary(r, edge),
            Some((r, s, edge)) => self.collide(r, s, edge),
            None => {}
        }
        self.schedule_node(v);
    }

    /// A shrinking region releases its outermost node, or reaches radius
    /// zero.
    fn process_shrink(&mut self, r: u32) {
        if let Some(v) = self.regions[r as usize].shell.pop() {
            debug_assert_eq!(self.local_radius(v, r), 0);
            let version = self.nodes[v as usize].version.wrapping_add(1);
            self.nodes[v as usize] = Node { version, ..EMPTY };
            // Neighbours may grow into the released node.
            let boundary = self.graph.n_detectors();
            let neighbours: Vec<u32> = self.graph.adjacency[v as usize]
                .iter()
                .map(|&e| {
                    let edge = &self.graph.edges()[e];
                    if edge.a == v as usize { edge.b } else { edge.a }
                })
                .filter(|&w| w != boundary)
                .map(|w| w as u32)
                .collect();
            for w in neighbours {
                self.schedule_node(w);
            }
            self.schedule_shrink(r);
            return;
        }
        debug_assert_eq!(self.radius(r), 0);
        if self.regions[r as usize].source != NONE {
            // A trivial minus region at radius zero: its parent and child
            // touch through its detection event and form a blossom.
            #[cfg(test)]
            {
                self.counts[2] += 1;
            }
            let g = &self.regions[r as usize];
            let parent = g.tree_parent;
            let child = g.tree_children[0];
            let up = g.parent_edge;
            let down = self.regions[child as usize].parent_edge;
            let closing = Edge {
                u: down.u,
                v: up.v,
                obs: down.obs ^ up.obs,
                len: down.len + up.len,
            };
            self.form_blossom(
                vec![parent, r, child],
                vec![up.reversed(), down.reversed(), closing],
            );
        } else {
            self.shatter(r);
        }
    }

    /// The root of the tree of a top-level region.
    fn root(&self, mut r: u32) -> u32 {
        while self.regions[r as usize].tree_parent != NONE {
            r = self.regions[r as usize].tree_parent;
        }
        r
    }

    fn hit_boundary(&mut self, r: u32, edge: Edge) {
        debug_assert!(self.regions[r as usize].plus);
        self.augment(r, Mate::Boundary, edge);
    }

    /// Two top-level regions touch (`r` or `s` growing) through `edge`
    /// (`u` in `r`).
    fn collide(&mut self, mut r: u32, mut s: u32, mut edge: Edge) {
        if !(self.regions[r as usize].in_tree && self.regions[r as usize].plus) {
            std::mem::swap(&mut r, &mut s);
            edge = edge.reversed();
        }
        let sg = &self.regions[s as usize];
        if !sg.in_tree {
            match sg.mate {
                Mate::Region(partner) => self.grow_tree(r, s, partner, edge),
                Mate::Boundary => {
                    // The boundary takes any number of regions: r's tree
                    // augments through s, whose boundary match ends.
                    self.regions[s as usize].mate = Mate::Region(r);
                    self.regions[s as usize].mate_edge = edge.reversed();
                    self.augment(r, Mate::Region(s), edge);
                }
                Mate::None => unreachable!("a region outside every tree is matched"),
            }
            return;
        }
        debug_assert!(sg.plus);
        let (root_r, root_s) = (self.root(r), self.root(s));
        if root_r != root_s {
            self.augment(r, Mate::Region(s), edge);
            self.augment(s, Mate::Region(r), edge.reversed());
        } else {
            self.blossom_from_collision(r, s, edge);
        }
    }

    /// `s` (matched to `partner`) joins the tree of `r` as a minus node,
    /// `partner` as a plus node below it.
    fn grow_tree(&mut self, r: u32, s: u32, partner: u32, edge: Edge) {
        #[cfg(test)]
        {
            self.counts[3] += 1;
        }
        let matched = self.regions[s as usize].mate_edge;
        {
            let g = &mut self.regions[s as usize];
            g.in_tree = true;
            g.plus = false;
            g.tree_parent = r;
            g.parent_edge = edge.reversed();
            g.tree_children = vec![partner];
            g.mate = Mate::None;
        }
        {
            let g = &mut self.regions[partner as usize];
            g.in_tree = true;
            g.plus = true;
            g.tree_parent = s;
            g.parent_edge = matched.reversed();
            g.tree_children.clear();
            g.mate = Mate::None;
        }
        self.regions[r as usize].tree_children.push(s);
        self.set_rate(s, -1);
        self.set_rate(partner, 1);
        self.reschedule_region(s);
        self.reschedule_region(partner);
    }

    /// Match `r` to `mate` through `edge`, flip the matching along the path
    /// to the root of its tree and dissolve the tree into matched pairs.
    fn augment(&mut self, r: u32, mate: Mate, edge: Edge) {
        let root = self.root(r);
        // Collect the tree before changing it.
        let mut members = Vec::new();
        let mut stack = vec![root];
        while let Some(x) = stack.pop() {
            members.push(x);
            stack.extend_from_slice(&self.regions[x as usize].tree_children);
        }
        // Minus nodes are matched to their (single) child.
        for &x in &members {
            let g = &self.regions[x as usize];
            if !g.plus {
                let child = g.tree_children[0];
                let e = self.regions[child as usize].parent_edge;
                self.regions[x as usize].mate = Mate::Region(child);
                self.regions[x as usize].mate_edge = e.reversed();
                self.regions[child as usize].mate = Mate::Region(x);
                self.regions[child as usize].mate_edge = e;
            }
        }
        // Along the path from r to the root: r takes the new mate, every
        // minus node its parent.
        self.regions[r as usize].mate = mate;
        self.regions[r as usize].mate_edge = edge;
        let mut x = r;
        while self.regions[x as usize].tree_parent != NONE {
            let minus = self.regions[x as usize].tree_parent;
            let plus = self.regions[minus as usize].tree_parent;
            let e = self.regions[minus as usize].parent_edge;
            self.regions[minus as usize].mate = Mate::Region(plus);
            self.regions[minus as usize].mate_edge = e;
            self.regions[plus as usize].mate = Mate::Region(minus);
            self.regions[plus as usize].mate_edge = e.reversed();
            x = plus;
        }
        for &x in &members {
            let g = &mut self.regions[x as usize];
            g.in_tree = false;
            g.tree_parent = NONE;
            g.tree_children.clear();
            self.set_rate(x, 0);
        }
        self.trees -= 1;
        for &x in &members {
            self.reschedule_region(x);
        }
    }

    /// Two plus regions of one tree touch: the cycle through their lowest
    /// common ancestor becomes a blossom.
    fn blossom_from_collision(&mut self, r: u32, s: u32, edge: Edge) {
        let path = |mut x: u32| {
            let mut p = vec![x];
            while self.regions[x as usize].tree_parent != NONE {
                x = self.regions[x as usize].tree_parent;
                p.push(x);
            }
            p
        };
        let (pr, ps) = (path(r), path(s));
        // Strip the common part above the lowest common ancestor.
        let (mut i, mut j) = (pr.len(), ps.len());
        while i > 0 && j > 0 && pr[i - 1] == ps[j - 1] {
            i -= 1;
            j -= 1;
        }
        // Cycle: lca → … → r, r → s, s → … → (child of lca).
        let mut regions: Vec<u32> = pr[..=i].iter().rev().copied().collect();
        let mut edges: Vec<Edge> = (1..regions.len())
            .map(|k| self.regions[regions[k] as usize].parent_edge.reversed())
            .collect();
        edges.push(edge);
        for &x in &ps[..j] {
            regions.push(x);
            edges.push(self.regions[x as usize].parent_edge);
        }
        self.form_blossom(regions, edges);
    }

    /// Make the odd cycle `regions[0] (closest to the root) → regions[1]
    /// → … → regions[0]` of one tree a plus blossom.
    fn form_blossom(&mut self, regions: Vec<u32>, edges: Vec<Edge>) {
        #[cfg(test)]
        {
            self.counts[0] += 1;
        }
        let b = self.regions.len() as u32;
        let base = regions[0];
        let base_region = &self.regions[base as usize];
        let (tree_parent, parent_edge) = (base_region.tree_parent, base_region.parent_edge);
        let mut tree_children = Vec::new();
        for &x in &regions {
            for &c in &self.regions[x as usize].tree_children {
                if !regions.contains(&c) {
                    tree_children.push(c);
                }
            }
        }
        if tree_parent != NONE {
            let siblings = &mut self.regions[tree_parent as usize].tree_children;
            if let Some(k) = siblings.iter().position(|&c| c == base) {
                siblings[k] = b;
            }
        }
        for &c in &tree_children {
            self.regions[c as usize].tree_parent = b;
        }
        // The members freeze; their radii move into `wrapped`.
        let mut nodes = Vec::new();
        for &x in &regions {
            let radius = self.radius(x);
            nodes.clear();
            self.coverage(x, &mut nodes);
            for &v in &nodes {
                self.nodes[v as usize].wrapped += radius;
            }
            let g = &mut self.regions[x as usize];
            g.radius0 = radius;
            g.t0 = self.now;
            g.rate = 0;
            g.version += 1;
            g.parent = b;
            g.in_tree = false;
            g.tree_parent = NONE;
            g.tree_children.clear();
            g.mate = Mate::None;
        }
        self.regions.push(Region {
            parent: NONE,
            source: NONE,
            children: regions,
            cycle: edges,
            radius0: 0,
            t0: self.now,
            rate: 1,
            shell: Vec::new(),
            mate: Mate::None,
            mate_edge: NO_EDGE,
            in_tree: true,
            plus: true,
            tree_parent,
            parent_edge,
            tree_children,
            version: 0,
        });
        self.reschedule_region(b);
    }

    /// The child of blossom `b` that contains detection event `d`.
    fn child_containing(&self, b: u32, d: u32) -> usize {
        // A source node keeps its trivial region for the whole shot.
        let mut x = self.nodes[d as usize].region;
        while self.regions[x as usize].parent != b {
            x = self.regions[x as usize].parent;
        }
        self.regions[b as usize]
            .children
            .iter()
            .position(|&c| c == x)
            .expect("child of the blossom")
    }

    /// A shrinking blossom at radius zero splits into its children: the
    /// even path between its tree edges stays in the tree, the rest pairs
    /// up along the cycle.
    fn shatter(&mut self, b: u32) {
        #[cfg(test)]
        {
            self.counts[1] += 1;
        }
        let g = self.regions[b as usize].clone();
        let child = g.tree_children[0];
        let down = self.regions[child as usize].parent_edge; // u in child, v in b
        let up = g.parent_edge; // u in b
        let len = g.children.len();
        let ia = self.child_containing(b, up.u);
        let ib = self.child_containing(b, down.v);
        let forward = (ib + len - ia) % len;
        // Path from ia to ib with an even number of steps.
        let mut path = vec![ia];
        let mut path_edges = Vec::new();
        if forward.is_multiple_of(2) {
            let mut k = ia;
            while k != ib {
                path_edges.push(g.cycle[k]);
                k = (k + 1) % len;
                path.push(k);
            }
        } else {
            let mut k = ia;
            while k != ib {
                let prev = (k + len - 1) % len;
                path_edges.push(g.cycle[prev].reversed());
                k = prev;
                path.push(k);
            }
        }
        // Children become top-level; their radii leave `wrapped`.
        let mut nodes = Vec::new();
        for &c in &g.children {
            let radius = self.regions[c as usize].radius0;
            nodes.clear();
            self.coverage(c, &mut nodes);
            for &v in &nodes {
                self.nodes[v as usize].wrapped -= radius;
            }
            let x = &mut self.regions[c as usize];
            x.parent = NONE;
            x.t0 = self.now;
        }
        // The tree path: minus, plus, …, minus.
        let parent = g.tree_parent;
        let siblings = &mut self.regions[parent as usize].tree_children;
        if let Some(k) = siblings.iter().position(|&c| c == b) {
            siblings[k] = g.children[ia];
        }
        for (k, &pos) in path.iter().enumerate() {
            let c = g.children[pos];
            let (tp, pe) = if k == 0 {
                (parent, up)
            } else {
                (g.children[path[k - 1]], path_edges[k - 1].reversed())
            };
            let x = &mut self.regions[c as usize];
            x.in_tree = true;
            x.plus = k % 2 == 1;
            x.tree_parent = tp;
            x.parent_edge = pe;
            x.mate = Mate::None;
            x.tree_children = if k + 1 < path.len() {
                vec![g.children[path[k + 1]]]
            } else {
                vec![child]
            };
            let rate = if k % 2 == 1 { 1 } else { -1 };
            self.set_rate(c, rate);
        }
        self.regions[child as usize].tree_parent = g.children[ib];
        // The other children pair up along the cycle.
        let on_path = |pos: usize| path.contains(&pos);
        let mut k = (ib + 1) % len;
        if forward % 2 == 1 {
            // The path ran backwards from ia; the rest runs forwards from ia + 1.
            k = (ia + 1) % len;
        }
        while !on_path(k) {
            let next = (k + 1) % len;
            let (x, y) = (g.children[k], g.children[next]);
            let e = g.cycle[k];
            self.regions[x as usize].mate = Mate::Region(y);
            self.regions[x as usize].mate_edge = e;
            self.regions[y as usize].mate = Mate::Region(x);
            self.regions[y as usize].mate_edge = e.reversed();
            for z in [x, y] {
                let r = &mut self.regions[z as usize];
                r.in_tree = false;
                r.tree_parent = NONE;
                r.tree_children.clear();
                self.set_rate(z, 0);
            }
            k = (next + 1) % len;
        }
        // The blossom is gone.
        let dead = &mut self.regions[b as usize];
        dead.children.clear();
        dead.in_tree = false;
        dead.version += 1;
        for &c in &g.children {
            self.reschedule_region(c);
        }
    }

    /// The matched pairs below the top-level matching: total observables
    /// and weight (in `MatchingDecoder` units).
    fn extract(&self) -> (u64, i64) {
        let mut obs = 0u64;
        let mut len = 0i64;
        let mut stack: Vec<(u32, u32)> = Vec::new();
        for (r, g) in self.regions.iter().enumerate() {
            let r = r as u32;
            if g.parent != NONE || (g.source == NONE && g.children.is_empty()) {
                continue;
            }
            match g.mate {
                Mate::Boundary => {
                    obs ^= g.mate_edge.obs;
                    len += g.mate_edge.len;
                    stack.push((r, g.mate_edge.u));
                }
                Mate::Region(s) if r < s => {
                    obs ^= g.mate_edge.obs;
                    len += g.mate_edge.len;
                    stack.push((r, g.mate_edge.u));
                    stack.push((s, g.mate_edge.v));
                }
                _ => {}
            }
        }
        // Inside a blossom matched through event d, the child holding d
        // is matched outside; the others pair up along the cycle.
        while let Some((b, d)) = stack.pop() {
            let g = &self.regions[b as usize];
            if g.children.is_empty() {
                continue;
            }
            let len_c = g.children.len();
            let j = self.child_containing(b, d);
            stack.push((g.children[j], d));
            let mut k = (j + 1) % len_c;
            while k != j {
                let e = g.cycle[k];
                obs ^= e.obs;
                len += e.len;
                stack.push((g.children[k], e.u));
                stack.push((g.children[(k + 1) % len_c], e.v));
                k = (k + 2) % len_c;
            }
        }
        (obs, len / 2)
    }
}

const NO_EDGE: Edge = Edge {
    u: NONE,
    v: NONE,
    obs: 0,
    len: 0,
};

#[cfg(test)]
mod tests;
