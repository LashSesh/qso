//! Canonical labelling and isomorphism of weighted graphs.
//!
//! The canonical form is found by individualisation–refinement (McKay
//! 1981): colour refinement (1-dimensional Weisfeiler–Leman, respecting
//! edge weights) splits the nodes into cells; the search individualises,
//! one at a time, every node of the first smallest non-singleton cell and
//! refines again, until every cell is a single node. Each such *leaf*
//! labels the nodes `0..n` by their colours, and the canonical form is the
//! leaf whose relabelled edge list is lexicographically smallest. Colours
//! are assigned from sorted signatures, so every step is invariant under
//! relabelling and isomorphic graphs get identical canonical forms.
//!
//! Two leaves with the same edge list differ by an automorphism. The search
//! records such automorphisms, jumps back to where the two paths part, and
//! skips children that an automorphism fixing the current path maps onto an
//! already explored child; with this pruning complete graphs need `O(n)`
//! leaves instead of `n!`. The worst
//! case is still exponential, as for every known algorithm on hard
//! instances; the method is intended for graphs with up to a few hundred
//! nodes.

use super::Graph;

/// Edge list of a leaf: `(min label, max label, weight bits)`, sorted.
type Certificate = Vec<(u32, u32, u64)>;

/// Colour refinement of a single graph. Colours only ever split, and new
/// colours are numbered in sorted order of (old colour, multiset of
/// neighbour colours and weights), so the result is invariant under
/// relabelling.
pub(crate) fn refine(g: &Graph, colours: &mut [u32]) {
    refine_bounded(g, colours, usize::MAX);
}

/// [`refine`], stopping early (and returning `false`) as soon as there are
/// more than `max_colours` colours.
pub(crate) fn refine_bounded(g: &Graph, colours: &mut [u32], max_colours: usize) -> bool {
    let n = g.num_nodes();
    let mut count = distinct(colours);
    if count > max_colours {
        return false;
    }
    loop {
        type Signature = (u32, Vec<(u32, u64)>, usize);
        let mut keyed: Vec<Signature> = (0..n)
            .map(|v| {
                let mut sig: Vec<(u32, u64)> = g
                    .neighbors(v)
                    .iter()
                    .map(|&(w, wt)| (colours[w], wt.to_bits()))
                    .collect();
                sig.sort_unstable();
                (colours[v], sig, v)
            })
            .collect();
        keyed.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
        let mut id = 0u32;
        for k in 0..keyed.len() {
            if k > 0 && (keyed[k].0, &keyed[k].1) != (keyed[k - 1].0, &keyed[k - 1].1) {
                id += 1;
            }
            colours[keyed[k].2] = id;
        }
        let new_count = if n == 0 { 0 } else { id as usize + 1 };
        if new_count == count {
            return true;
        }
        if new_count > max_colours {
            return false;
        }
        count = new_count;
    }
}

fn distinct(colours: &[u32]) -> usize {
    let mut c = colours.to_vec();
    c.sort_unstable();
    c.dedup();
    c.len()
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
        // Orbits of the automorphisms found so far that fix the path,
        // updated as the children below find new ones.
        let mut parent: Vec<usize> = (0..n).collect();
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
    let mut search = Search {
        g,
        best: None,
        first: None,
        automorphisms: Vec::new(),
    };
    search.explore(vec![0; n], &mut Vec::new());
    search.best.expect("every search reaches a leaf").labels
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

    fn disjoint_triangles() -> Graph {
        Graph::from_edges(6, &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]).unwrap()
    }

    #[test]
    fn pruning_keeps_large_symmetric_graphs_fast() {
        // K60 has 60! automorphisms; the pruned search visits O(n) leaves.
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
