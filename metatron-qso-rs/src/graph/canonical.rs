//! Canonical labelling and isomorphism of weighted graphs.
//!
//! The canonical form is found by individualisation–refinement (McKay
//! 1981): colour refinement (1-dimensional Weisfeiler–Leman, respecting
//! edge weights) splits the nodes into cells; the search individualises,
//! one at a time, every node of the first smallest non-singleton cell and
//! refines again, until every cell is a single node. Each such *leaf*
//! labels the nodes `0..n` by their colours, and the canonical form is the
//! leaf whose relabelled edge list is lexicographically smallest. Colours
//! are positions in an ordered partition, split by sorted signatures, so
//! every step is invariant under relabelling and isomorphic graphs get
//! identical canonical forms. Refinement follows Hopcroft's scheme
//! (`O(m log n)` splitting steps).
//!
//! Two leaves with the same edge list differ by an automorphism. The search
//! records such automorphisms, jumps back to where the two paths part, and
//! skips children that an automorphism fixing the current path maps onto an
//! already explored child. Twins (nodes with the same neighbours apart from
//! each other) can be swapped by an automorphism, so only one twin of a
//! class is explored, and a node whose classes are all sets of twins is
//! finished directly: complete, empty, complete bipartite graphs and stars
//! need one leaf. The worst
//! case is still exponential, as for every known algorithm on hard
//! instances, and graphs without symmetry cost one refinement per node of
//! the first cell (2000-node random cubic graphs take seconds).

use super::Graph;

/// Edge list of a leaf: `(min label, max label, weight bits)`, sorted.
type Certificate = Vec<(u32, u32, u64)>;

/// Colour refinement of a single graph: the coarsest refinement of
/// `colours` in which nodes of one colour have the same multiset of edge
/// weights into every colour class. On return each node's colour is the
/// position of its class in an ordered partition (classes ordered by their
/// initial colour, then by the splitting steps), so the result is invariant
/// under relabelling, and a discrete colouring is a permutation of `0..n`.
pub(crate) fn refine(g: &Graph, colours: &mut [u32]) {
    refine_bounded(g, colours, usize::MAX);
}

/// [`refine`], stopping early (and returning `false`) as soon as there are
/// more than `max_colours` colours.
///
/// Partition refinement with a queue of splitter classes (Hopcroft): when a
/// class splits, its parts are queued, except the largest one if the class
/// was not queued itself, so every edge is examined `O(log n)` times.
pub(crate) fn refine_bounded(g: &Graph, colours: &mut [u32], max_colours: usize) -> bool {
    use std::collections::VecDeque;
    let n = g.num_nodes();
    if n == 0 {
        return true;
    }
    // Ordered partition: `order[start..end[start]]` is the class starting
    // at position `start`; `cell[v]` is the start of v's class.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&v| colours[v]);
    let mut pos = vec![0usize; n];
    for (i, &v) in order.iter().enumerate() {
        pos[v] = i;
    }
    let mut cell = vec![0usize; n];
    let mut end = vec![0usize; n];
    let mut queue = VecDeque::new();
    let mut queued = vec![false; n];
    let mut start = 0;
    let mut count = 0;
    for i in 1..=n {
        if i == n || colours[order[i]] != colours[order[i - 1]] {
            for &v in &order[start..i] {
                cell[v] = start;
            }
            end[start] = i;
            queue.push_back(start);
            queued[start] = true;
            count += 1;
            start = i;
        }
    }
    let finish = |cell: &[usize], colours: &mut [u32]| {
        for (c, &x) in colours.iter_mut().zip(cell) {
            *c = x as u32;
        }
    };
    if count > max_colours {
        finish(&cell, colours);
        return false;
    }
    let mut hits: Vec<(usize, usize, u64)> = Vec::new();
    let mut keys: Vec<(Vec<u64>, usize)> = Vec::new();
    let mut blocks: Vec<(usize, usize)> = Vec::new();
    while let Some(splitter) = queue.pop_front() {
        queued[splitter] = false;
        // (class of w, w, weight) for every edge from the splitter to w.
        hits.clear();
        for &v in &order[splitter..end[splitter]] {
            for &(w, wt) in g.neighbors(v) {
                hits.push((cell[w], w, wt.to_bits()));
            }
        }
        hits.sort_unstable();
        let mut i = 0;
        while i < hits.len() {
            let c = hits[i].0;
            let mut j = i;
            while j < hits.len() && hits[j].0 == c {
                j += 1;
            }
            let (cs, ce) = (c, end[c]);
            // Key of each touched member: its sorted weights into the
            // splitter. Untouched members have the empty key.
            keys.clear();
            let mut k = i;
            while k < j {
                let w = hits[k].1;
                let mut weights = Vec::new();
                while k < j && hits[k].1 == w {
                    weights.push(hits[k].2);
                    k += 1;
                }
                keys.push((weights, w));
            }
            i = j;
            let untouched = (ce - cs) - keys.len();
            keys.sort_by(|a, b| a.0.cmp(&b.0));
            if ce - cs < 2 || (untouched == 0 && keys[0].0 == keys[keys.len() - 1].0) {
                continue;
            }
            // Move the touched members to the end of the class, in key
            // order (O(touched) work, as Hopcroft's bound needs); the
            // untouched ones stay in front as the first block.
            let tail = ce - keys.len();
            for (idx, (_, v)) in keys.iter().enumerate() {
                let (from, to) = (pos[*v], tail + idx);
                let other = order[to];
                order.swap(from, to);
                pos[other] = from;
                pos[*v] = to;
            }
            // The swaps above may have moved touched members out of their
            // target slots again; place them once more in key order.
            for (idx, (_, v)) in keys.iter().enumerate() {
                let (from, to) = (pos[*v], tail + idx);
                if from != to {
                    let other = order[to];
                    order.swap(from, to);
                    pos[other] = from;
                    pos[*v] = to;
                }
            }
            blocks.clear();
            if untouched > 0 {
                blocks.push((cs, tail));
            }
            let mut b = tail;
            for idx in 1..=keys.len() {
                if idx == keys.len() || keys[idx].0 != keys[idx - 1].0 {
                    blocks.push((b, tail + idx));
                    b = tail + idx;
                }
            }
            for &(bs, be) in &blocks {
                end[bs] = be;
                for &v in &order[bs..be] {
                    cell[v] = bs;
                }
            }
            count += blocks.len() - 1;
            if count > max_colours {
                finish(&cell, colours);
                return false;
            }
            let largest = blocks
                .iter()
                .enumerate()
                .max_by(|a, b| (a.1.1 - a.1.0).cmp(&(b.1.1 - b.1.0)).then(b.0.cmp(&a.0)))
                .map(|(idx, _)| idx)
                .expect("at least two blocks");
            let was_queued = queued[cs];
            for (idx, &(bs, _)) in blocks.iter().enumerate() {
                if !queued[bs] && (was_queued || idx != largest) {
                    queued[bs] = true;
                    queue.push_back(bs);
                }
            }
        }
    }
    finish(&cell, colours);
    true
}

/// Twin classes of `g` within the colour classes: nodes `a`, `b` of one
/// colour are twins if `N(a) \ {b} = N(b) \ {a}`, weights included, so
/// that swapping them is an automorphism fixing every other node. Returns
/// a union–find forest whose roots are the twin classes.
fn twin_classes(g: &Graph, colours: &[u32]) -> Vec<usize> {
    use std::collections::HashMap;
    let n = g.num_nodes();
    let mut size = HashMap::new();
    for &c in colours {
        *size.entry(c).or_insert(0usize) += 1;
    }
    let mut parent: Vec<usize> = (0..n).collect();
    // False twins (not adjacent) have equal neighbourhoods; true twins
    // (adjacent with weight w) have equal "closed" neighbourhoods
    // N(a) ∪ {(a, w)}.
    type Key = (u32, bool, Vec<(usize, u64)>);
    let mut seen: HashMap<Key, usize> = HashMap::new();
    for a in 0..n {
        if size[&colours[a]] < 2 {
            continue;
        }
        let open: Vec<(usize, u64)> = g
            .neighbors(a)
            .iter()
            .map(|&(v, w)| (v, w.to_bits()))
            .collect();
        let mut inside: Vec<u64> = g
            .neighbors(a)
            .iter()
            .filter(|&&(v, _)| colours[v] == colours[a])
            .map(|&(_, w)| w.to_bits())
            .collect();
        inside.sort_unstable();
        inside.dedup();
        let mut keys = vec![(colours[a], false, open.clone())];
        for w in inside {
            let mut closed = open.clone();
            let pos = closed.partition_point(|&(v, _)| v < a);
            closed.insert(pos, (a, w));
            keys.push((colours[a], true, closed));
        }
        for key in keys {
            match seen.get(&key) {
                Some(&rep) => {
                    let (x, y) = (find(&mut parent, rep), find(&mut parent, a));
                    if x != y {
                        parent[x.max(y)] = x.min(y);
                    }
                }
                None => {
                    seen.insert(key, a);
                }
            }
        }
    }
    parent
}

/// Union–find root with path halving.
fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

struct Search<'a> {
    g: &'a Graph,
    /// Best leaf so far: path, labelling (node → label) and certificate.
    best: Option<Leaf>,
    /// First leaf (classic second reference for finding automorphisms).
    first: Option<Leaf>,
    /// Automorphisms found so far (node ↦ image).
    automorphisms: Vec<Vec<usize>>,
}

#[derive(Clone)]
struct Leaf {
    path: Vec<usize>,
    labels: Vec<usize>,
    certificate: Certificate,
}

fn common_prefix(a: &[usize], b: &[usize]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

impl Search<'_> {
    fn certificate(&self, labels: &[usize]) -> Certificate {
        let mut cert: Certificate = self
            .g
            .edges()
            .iter()
            .map(|&(u, v, w)| {
                let (a, b) = (labels[u] as u32, labels[v] as u32);
                (a.min(b), a.max(b), w.to_bits())
            })
            .collect();
        cert.sort_unstable();
        cert
    }

    /// The automorphism `x ↦ reference⁻¹(leaf(x))` between two leaves with
    /// equal certificates.
    fn automorphism(leaf: &[usize], reference: &[usize]) -> Vec<usize> {
        let mut node_of_label = vec![0; reference.len()];
        for (x, &l) in reference.iter().enumerate() {
            node_of_label[l] = x;
        }
        leaf.iter().map(|&l| node_of_label[l]).collect()
    }

    /// Record a leaf. If it is equivalent to the first or the best leaf,
    /// the automorphism between them maps that leaf's path onto this one
    /// and fixes their common prefix, so the search can jump back to where
    /// the two paths part: returns that depth.
    fn leaf(&mut self, colours: &[u32], path: &[usize]) -> Option<usize> {
        let labels: Vec<usize> = colours.iter().map(|&c| c as usize).collect();
        let certificate = self.certificate(&labels);
        for reference in [&self.first, &self.best].into_iter().flatten() {
            if reference.certificate == certificate {
                let gamma = Self::automorphism(&labels, &reference.labels);
                let depth = common_prefix(path, &reference.path);
                if gamma.iter().enumerate().any(|(x, &y)| x != y) {
                    self.automorphisms.push(gamma);
                }
                return Some(depth);
            }
        }
        let leaf = Leaf {
            path: path.to_vec(),
            labels,
            certificate,
        };
        if self.first.is_none() {
            self.first = Some(leaf.clone());
        }
        if self
            .best
            .as_ref()
            .is_none_or(|b| leaf.certificate < b.certificate)
        {
            self.best = Some(leaf);
        }
        None
    }

    /// Explore the subtree below `path`. Returns the depth to jump back to
    /// after an automorphism made the rest of this subtree redundant.
    fn explore(&mut self, mut colours: Vec<u32>, path: &mut Vec<usize>) -> Option<usize> {
        refine(self.g, &mut colours);
        let n = self.g.num_nodes();
        let max = colours.iter().copied().max().map_or(0, |m| m as usize + 1);
        let mut size = vec![0usize; max];
        for &c in &colours {
            size[c as usize] += 1;
        }
        let Some(target) = (0..max)
            .filter(|&c| size[c] > 1)
            .min_by_key(|&c| (size[c], c))
        else {
            return self.leaf(&colours, path);
        };
        let cell: Vec<usize> = (0..n).filter(|&v| colours[v] as usize == target).collect();
        let mut twins = twin_classes(self.g, &colours);
        // If every class is a set of twins, all leaves below are equivalent
        // (they differ by swaps of twins, which are automorphisms fixing the
        // path): finish one of them directly. Individualising a twin splits
        // only its own class, so the leaf numbers each class's members
        // consecutively from the class position.
        let mut root_of_class: Vec<Option<usize>> = vec![None; max];
        let all_twins = (0..n).all(|v| {
            let r = find(&mut twins, v);
            *root_of_class[colours[v] as usize].get_or_insert(r) == r
        });
        if all_twins {
            let mut next: Vec<u32> = (0..max as u32).collect();
            let labels: Vec<u32> = colours
                .iter()
                .map(|&c| {
                    let l = next[c as usize];
                    next[c as usize] += 1;
                    l
                })
                .collect();
            return self.leaf(&labels, path);
        }
        // Orbits of the automorphisms found so far that fix the path, and of
        // the swaps of twins, updated as the children below find new ones.
        let mut parent = twins;
        let mut merged = 0;
        let mut explored: Vec<usize> = Vec::new();
        for &v in &cell {
            for gamma in &self.automorphisms[merged..] {
                if path.iter().all(|&p| gamma[p] == p) {
                    for (x, &y) in gamma.iter().enumerate() {
                        let (a, b) = (find(&mut parent, x), find(&mut parent, y));
                        if a != b {
                            parent[a.max(b)] = a.min(b);
                        }
                    }
                }
            }
            merged = self.automorphisms.len();
            let root = find(&mut parent, v);
            if explored.iter().any(|&u| find(&mut parent, u) == root) {
                continue;
            }
            explored.push(v);
            let child: Vec<u32> = (0..n).map(|x| 2 * colours[x] + u32::from(x != v)).collect();
            path.push(v);
            let jump = self.explore(child, path);
            path.pop();
            if let Some(depth) = jump
                && depth < path.len()
            {
                return Some(depth);
            }
        }
        None
    }
}

/// Canonical labelling: `labels[u]` is the canonical index of node `u`.
/// Isomorphic graphs (with equal weights) have equal relabelled edge lists.
///
/// ```
/// use metatron_qso::graph::{canonical, generators};
///
/// let g = generators::petersen();
/// let shuffled = g.relabeled(&[3, 1, 4, 0, 5, 9, 2, 6, 8, 7]).unwrap();
/// assert!(g.is_isomorphic(&shuffled));
/// let labels = canonical::canonical_labelling(&g);
/// assert_eq!(g.relabeled(&labels).unwrap().edges().len(), 15);
/// ```
pub fn canonical_labelling(g: &Graph) -> Vec<usize> {
    let n = g.num_nodes();
    if n == 0 {
        return Vec::new();
    }
    let run = || {
        let mut search = Search {
            g,
            best: None,
            first: None,
            automorphisms: Vec::new(),
        };
        search.explore(vec![0; n], &mut Vec::new());
        search.best.expect("every search reaches a leaf").labels
    };
    // The search recurses once per individualised node. Large graphs get a
    // thread with a large stack, so that deep searches cannot overflow the
    // caller's (for example a Python thread's 512 KiB).
    if n <= 1024 {
        return run();
    }
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .stack_size(256 << 20)
            .spawn_scoped(scope, run)
            .map(|handle| handle.join().expect("canonical search panicked"))
            .unwrap_or_else(|_| run())
    })
}

/// Canonical certificate: the sorted edge list `(label u, label v, weight)`
/// under the canonical labelling, together with the number of nodes. Two
/// graphs are isomorphic exactly when their certificates are equal.
pub fn certificate(g: &Graph) -> (usize, Vec<(usize, usize, f64)>) {
    let labels = canonical_labelling(g);
    let mut edges: Vec<(usize, usize, f64)> = g
        .edges()
        .iter()
        .map(|&(u, v, w)| (labels[u].min(labels[v]), labels[u].max(labels[v]), w))
        .collect();
    edges.sort_by_key(|&(u, v, w)| (u, v, w.to_bits()));
    (g.num_nodes(), edges)
}

/// An isomorphism `g → h` (node `u` of `g` maps to `iso[u]` of `h`), if
/// the graphs are isomorphic.
pub fn isomorphism(g: &Graph, h: &Graph) -> Option<Vec<usize>> {
    if g.num_nodes() != h.num_nodes() || g.num_edges() != h.num_edges() {
        return None;
    }
    let (lg, lh) = (canonical_labelling(g), canonical_labelling(h));
    let cert = |graph: &Graph, labels: &[usize]| {
        let mut c: Certificate = graph
            .edges()
            .iter()
            .map(|&(u, v, w)| {
                let (a, b) = (labels[u] as u32, labels[v] as u32);
                (a.min(b), a.max(b), w.to_bits())
            })
            .collect();
        c.sort_unstable();
        c
    };
    if cert(g, &lg) != cert(h, &lh) {
        return None;
    }
    let mut node_of_label = vec![0; lh.len()];
    for (x, &l) in lh.iter().enumerate() {
        node_of_label[l] = x;
    }
    Some(lg.iter().map(|&l| node_of_label[l]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::automorphism::is_automorphism;
    use crate::graph::generators::*;
    use crate::graph::metatron;
    use crate::rng::Rng;

    fn shuffled(g: &Graph, seed: u64) -> (Graph, Vec<usize>) {
        let mut perm: Vec<usize> = (0..g.num_nodes()).collect();
        Rng::seed_from_u64(seed).shuffle(&mut perm);
        (g.relabeled(&perm).unwrap(), perm)
    }

    fn is_isomorphism(g: &Graph, h: &Graph, iso: &[usize]) -> bool {
        g.edges()
            .iter()
            .all(|&(u, v, w)| h.edge_weight(iso[u], iso[v]) == Some(w))
            && g.num_edges() == h.num_edges()
    }

    #[test]
    fn canonical_forms_are_invariant_under_relabelling() {
        let mut rng = Rng::seed_from_u64(5);
        let graphs = vec![
            petersen(),
            heawood(),
            complete(9),
            cycle(11),
            path(7),
            hypercube(4),
            grid_2d(4, 5, true),
            metatron::metatron_cube(),
            metatron::fruit_of_life(),
            platonic(PlatonicSolid::Dodecahedron),
            random_regular(20, 3, &mut rng).unwrap(),
            erdos_renyi(25, 0.3, &mut rng).unwrap(),
            Graph::from_weighted_edges(4, &[(0, 1, 2.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)])
                .unwrap(),
            Graph::new(5),
        ];
        for g in &graphs {
            for seed in 0..3 {
                let (h, _) = shuffled(g, seed);
                assert_eq!(certificate(g), certificate(&h), "{}", g.name());
                assert_eq!(g.canonical_form().edges(), h.canonical_form().edges());
                let iso = isomorphism(g, &h).expect("isomorphic");
                assert!(is_isomorphism(g, &h, &iso), "{}", g.name());
            }
        }
    }

    #[test]
    fn separates_non_isomorphic_graphs() {
        // Cospectral strongly regular graphs with parameters (16, 6, 2, 2).
        let (s, r) = (shrikhande(), rook(4, 4).unwrap());
        assert_eq!(s.adjacency_spectrum().len(), r.adjacency_spectrum().len());
        for (a, b) in s.adjacency_spectrum().iter().zip(r.adjacency_spectrum()) {
            assert!((a - b).abs() < 1e-9);
        }
        assert!(isomorphism(&s, &r).is_none());
        assert_ne!(certificate(&s), certificate(&r));
        // Same degree sequence, different structure.
        assert!(isomorphism(&cycle(6), &disjoint_triangles()).is_none());
        // Weights matter.
        let a = Graph::from_weighted_edges(3, &[(0, 1, 1.0), (1, 2, 2.0)]).unwrap();
        let b = Graph::from_weighted_edges(3, &[(0, 1, 1.0), (1, 2, 3.0)]).unwrap();
        assert!(isomorphism(&a, &b).is_none());
    }

    /// Isomorphism by trying all permutations (small graphs only).
    fn brute_force_isomorphic(g: &Graph, h: &Graph) -> bool {
        fn next_permutation(p: &mut [usize]) -> bool {
            let Some(i) = (1..p.len()).rev().find(|&i| p[i - 1] < p[i]) else {
                return false;
            };
            let j = (i..p.len()).rev().find(|&j| p[j] > p[i - 1]).unwrap();
            p.swap(i - 1, j);
            p[i..].reverse();
            true
        }
        if g.num_nodes() != h.num_nodes() || g.num_edges() != h.num_edges() {
            return false;
        }
        let mut p: Vec<usize> = (0..g.num_nodes()).collect();
        loop {
            if is_isomorphism(g, h, &p) {
                return true;
            }
            if !next_permutation(&mut p) {
                return false;
            }
        }
    }

    #[test]
    fn agrees_with_brute_force_on_small_random_graphs() {
        let mut rng = Rng::seed_from_u64(11);
        let mut random_graph = |n: usize| {
            let mut edges = Vec::new();
            for u in 0..n {
                for v in u + 1..n {
                    if rng.uniform() < 0.45 {
                        edges.push((u, v, if rng.uniform() < 0.2 { 2.0 } else { 1.0 }));
                    }
                }
            }
            Graph::from_weighted_edges(n, &edges).unwrap()
        };
        for k in 0..300 {
            let n = 1 + k % 7;
            let (g, h) = (random_graph(n), random_graph(n));
            let expected = brute_force_isomorphic(&g, &h);
            assert_eq!(
                isomorphism(&g, &h).is_some(),
                expected,
                "{:?} {:?}",
                g.edges(),
                h.edges()
            );
            assert_eq!(certificate(&g) == certificate(&h), expected);
            let (s, _) = shuffled(&g, k as u64);
            assert_eq!(certificate(&g), certificate(&s));
        }
    }

    #[test]
    fn twins_and_long_paths_are_fast() {
        // Classes of twins finish at once; long paths refine in O(m log n).
        let start = std::time::Instant::now();
        let graphs = [
            complete(400),
            Graph::new(20_000),
            star(2000),
            complete_bipartite(200, 200),
            path(20_000),
            cycle(8000),
        ];
        for (k, g) in graphs.iter().enumerate() {
            let (h, _) = shuffled(g, k as u64);
            assert_eq!(g.canonical_form().edges(), h.canonical_form().edges());
        }
        assert!(start.elapsed().as_secs() < 20, "{:?}", start.elapsed());
        // Weighted 4-cycles: twins only if their edge weights agree. In
        // `w`, 1 and 2 are twins (weights 2 to 0 and 1 to 3); `u` is `w`
        // relabelled; in `v` the weights alternate and nothing is a twin.
        let w =
            Graph::from_weighted_edges(4, &[(0, 1, 2.0), (0, 2, 2.0), (1, 3, 1.0), (2, 3, 1.0)])
                .unwrap();
        let u =
            Graph::from_weighted_edges(4, &[(0, 1, 2.0), (0, 2, 1.0), (1, 3, 2.0), (2, 3, 1.0)])
                .unwrap();
        let v =
            Graph::from_weighted_edges(4, &[(0, 1, 2.0), (0, 2, 1.0), (1, 3, 1.0), (2, 3, 2.0)])
                .unwrap();
        assert!(w.is_isomorphic(&u));
        assert!(!w.is_isomorphic(&v));
        assert_eq!(w.automorphism_group().order(), Some(2));
    }

    fn disjoint_triangles() -> Graph {
        Graph::from_edges(6, &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]).unwrap()
    }

    #[test]
    fn pruning_keeps_large_symmetric_graphs_fast() {
        // K60 has 60! automorphisms; twin pruning finishes it in one leaf.
        let start = std::time::Instant::now();
        let k = complete(60);
        let (h, _) = shuffled(&k, 1);
        assert!(isomorphism(&k, &h).is_some());
        assert!(start.elapsed().as_secs() < 20);
        // Leaf automorphisms are genuine.
        let g = hypercube(5);
        let mut search = Search {
            g: &g,
            best: None,
            first: None,
            automorphisms: Vec::new(),
        };
        search.explore(vec![0; g.num_nodes()], &mut Vec::new());
        assert!(!search.automorphisms.is_empty());
        assert!(search.automorphisms.iter().all(|p| is_automorphism(&g, p)));
    }
}
