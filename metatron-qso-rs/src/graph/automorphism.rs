//! Exact automorphism groups of weighted graphs.
//!
//! The group is computed along a stabiliser chain
//! `G = G₀ ≥ G₁ ≥ … ≥ G_k = {id}`, where `G_i` fixes the base points
//! `b₁ … b_i`. For each level the orbit of `b_{i+1}` under `G_i` is found by
//! searching, for every candidate image `w`, for an automorphism that fixes
//! the base so far and maps `b_{i+1} ↦ w`. The search is
//! individualisation–refinement: matching vertices of two copies of the graph
//! are given equal colours, both copies are refined jointly with
//! 1-dimensional Weisfeiler–Leman colour refinement (respecting edge
//! weights), and the search branches on the smallest non-trivial colour
//! class. By the orbit–stabiliser theorem `|G| = Π |b_{i+1}^{G_i}|`, so the
//! order is exact; the automorphisms found form a strong generating set.
//!
//! The method is exhaustive and therefore exact, but its worst case is
//! exponential (as for every known algorithm on hard instances such as some
//! strongly regular graphs). It is intended for graphs with up to a few
//! hundred nodes.

use serde::{Deserialize, Serialize};

use super::Graph;

/// Automorphism group of a graph, described by a base and strong
/// generating set.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutomorphismGroup {
    n: usize,
    base: Vec<usize>,
    orbit_sizes: Vec<usize>,
    generators: Vec<Vec<usize>>,
}

impl AutomorphismGroup {
    /// Group order, or `None` if it overflows `u128` (e.g. `K_n`, n > 34).
    pub fn order(&self) -> Option<u128> {
        self.orbit_sizes
            .iter()
            .try_fold(1u128, |acc, &s| acc.checked_mul(s as u128))
    }

    /// `log₁₀ |G|` (always finite).
    pub fn log10_order(&self) -> f64 {
        self.orbit_sizes.iter().map(|&s| (s as f64).log10()).sum()
    }

    /// Strong generating set; permutation `p` maps node `u` to `p[u]`.
    pub fn generators(&self) -> &[Vec<usize>] {
        &self.generators
    }

    /// Base points of the stabiliser chain.
    pub fn base(&self) -> &[usize] {
        &self.base
    }

    /// Sizes of the basic orbits along the base.
    pub fn orbit_sizes(&self) -> &[usize] {
        &self.orbit_sizes
    }

    /// Partition of the nodes into orbits (each sorted, ordered by smallest
    /// element).
    pub fn orbits(&self) -> Vec<Vec<usize>> {
        let mut parent: Vec<usize> = (0..self.n).collect();
        fn find(p: &mut [usize], x: usize) -> usize {
            let mut r = x;
            while p[r] != r {
                r = p[r];
            }
            let mut y = x;
            while p[y] != r {
                let next = p[y];
                p[y] = r;
                y = next;
            }
            r
        }
        for g in &self.generators {
            for (u, &v) in g.iter().enumerate() {
                let (a, b) = (find(&mut parent, u), find(&mut parent, v));
                if a != b {
                    parent[a.max(b)] = a.min(b);
                }
            }
        }
        let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        for u in 0..self.n {
            let r = find(&mut parent, u);
            groups.entry(r).or_default().push(u);
        }
        groups.into_values().collect()
    }

    /// Every node can be mapped to every other node.
    pub fn is_vertex_transitive(&self) -> bool {
        self.orbits().len() <= 1
    }
}

/// Whether `perm` (node `u ↦ perm[u]`) preserves all edges and weights.
pub fn is_automorphism(g: &Graph, perm: &[usize]) -> bool {
    perm.len() == g.num_nodes()
        && g.edges()
            .iter()
            .all(|&(u, v, w)| g.edge_weight(perm[u], perm[v]) == Some(w))
}

/// Joint colour refinement of two copies of `g` (left: `0..n`, right:
/// `n..2n`). Colours only ever split, so refinement stops as soon as the
/// number of colours stays constant.
fn refine(g: &Graph, colours: &mut [u32]) {
    let n = g.num_nodes();
    let mut count = {
        let mut c = colours.to_vec();
        c.sort_unstable();
        c.dedup();
        c.len()
    };
    loop {
        // (current colour, sorted neighbour colours with weights, vertex)
        type Signature = (u32, Vec<(u32, u64)>, usize);
        let mut keyed: Vec<Signature> = (0..2 * n)
            .map(|x| {
                let (v, off) = if x < n { (x, 0) } else { (x - n, n) };
                let mut sig: Vec<(u32, u64)> = g
                    .neighbors(v)
                    .iter()
                    .map(|&(w, wt)| (colours[w + off], wt.to_bits()))
                    .collect();
                sig.sort_unstable();
                (colours[x], sig, x)
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
        let new_count = id as usize + 1;
        if new_count == count {
            return;
        }
        count = new_count;
    }
}

/// Refined colouring of the paired copies with `left[i] ↔ right[i]`
/// individualised. Returns `None` if the colour classes are incompatible.
fn paired_colouring(g: &Graph, left: &[usize], right: &[usize]) -> Option<Vec<u32>> {
    let n = g.num_nodes();
    let mut colours = vec![0u32; 2 * n];
    for (i, (&l, &r)) in left.iter().zip(right).enumerate() {
        if colours[l] != 0 || colours[n + r] != 0 {
            return None;
        }
        colours[l] = i as u32 + 1;
        colours[n + r] = i as u32 + 1;
    }
    refine(g, &mut colours);
    let max = *colours.iter().max().unwrap_or(&0) as usize;
    let mut cl = vec![0usize; max + 1];
    let mut cr = vec![0usize; max + 1];
    for v in 0..n {
        cl[colours[v] as usize] += 1;
        cr[colours[n + v] as usize] += 1;
    }
    (cl == cr).then_some(colours)
}

/// Search for an automorphism mapping `left[i] ↦ right[i]` for all `i`.
fn find_mapping(g: &Graph, left: &mut Vec<usize>, right: &mut Vec<usize>) -> Option<Vec<usize>> {
    let n = g.num_nodes();
    let colours = paired_colouring(g, left, right)?;
    let max = *colours.iter().max().unwrap_or(&0) as usize;
    let mut size = vec![0usize; max + 1];
    for &c in &colours[..n] {
        size[c as usize] += 1;
    }
    let branch = (0..=max)
        .filter(|&c| size[c] > 1)
        .min_by_key(|&c| (size[c], c));
    match branch {
        None => {
            let mut right_of = vec![0usize; max + 1];
            for r in 0..n {
                right_of[colours[n + r] as usize] = r;
            }
            let perm: Vec<usize> = (0..n).map(|l| right_of[colours[l] as usize]).collect();
            is_automorphism(g, &perm).then_some(perm)
        }
        Some(c) => {
            let u = (0..n)
                .find(|&v| colours[v] as usize == c)
                .expect("non-empty class");
            let candidates: Vec<usize> = (0..n).filter(|&r| colours[n + r] as usize == c).collect();
            for w in candidates {
                left.push(u);
                right.push(w);
                let found = find_mapping(g, left, right);
                left.pop();
                right.pop();
                if found.is_some() {
                    return found;
                }
            }
            None
        }
    }
}

/// Exact automorphism group of `g` (edge weights must be preserved).
///
/// ```
/// use metatron_qso::graph::{generators, metatron};
///
/// assert_eq!(generators::petersen().automorphism_group().order(), Some(120));
/// // Metatron's Cube is K13: every permutation is a symmetry.
/// assert_eq!(metatron::metatron_cube().automorphism_group().order(), Some(6_227_020_800));
/// // The Fruit-of-Life contact graph has the hexagonal symmetry D6.
/// assert_eq!(metatron::fruit_of_life().automorphism_group().order(), Some(12));
/// ```
pub fn automorphism_group(g: &Graph) -> AutomorphismGroup {
    let n = g.num_nodes();
    let mut base: Vec<usize> = Vec::new();
    let mut orbit_sizes = Vec::new();
    let mut generators = Vec::new();
    loop {
        let colours = paired_colouring(g, &base, &base).expect("identity pairing is compatible");
        let max = *colours.iter().max().unwrap_or(&0) as usize;
        let mut size = vec![0usize; max + 1];
        for &c in &colours[..n] {
            size[c as usize] += 1;
        }
        let Some(cell) = (0..=max)
            .filter(|&c| size[c] > 1)
            .min_by_key(|&c| (size[c], c))
        else {
            break;
        };
        let cell_nodes: Vec<usize> = (0..n).filter(|&v| colours[v] as usize == cell).collect();
        let b = cell_nodes[0];
        let mut level_generators: Vec<Vec<usize>> = Vec::new();
        let mut in_orbit = vec![false; n];
        in_orbit[b] = true;
        for &w in &cell_nodes[1..] {
            if in_orbit[w] {
                continue;
            }
            let mut left = base.clone();
            let mut right = base.clone();
            left.push(b);
            right.push(w);
            if let Some(sigma) = find_mapping(g, &mut left, &mut right) {
                level_generators.push(sigma);
                // Close the orbit of b under the generators of this level
                // (all of them fix the current base, so they lie in G_i).
                let mut stack = vec![b];
                in_orbit.iter_mut().for_each(|x| *x = false);
                in_orbit[b] = true;
                while let Some(x) = stack.pop() {
                    for s in &level_generators {
                        let y = s[x];
                        if !in_orbit[y] {
                            in_orbit[y] = true;
                            stack.push(y);
                        }
                    }
                }
            }
        }
        orbit_sizes.push(in_orbit.iter().filter(|&&x| x).count());
        base.push(b);
        generators.extend(level_generators);
    }
    AutomorphismGroup {
        n,
        base,
        orbit_sizes,
        generators,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::generators::*;
    use crate::graph::metatron;
    use crate::rng::Rng;

    fn factorial(n: u128) -> u128 {
        (1..=n).product()
    }

    #[test]
    fn known_group_orders() {
        for n in 1..=8u128 {
            assert_eq!(
                complete(n as usize).automorphism_group().order(),
                Some(factorial(n))
            );
            assert_eq!(
                Graph::new(n as usize).automorphism_group().order(),
                Some(factorial(n))
            );
        }
        for n in 3..=12 {
            assert_eq!(
                cycle(n).automorphism_group().order(),
                Some(2 * n as u128),
                "C{n}"
            );
        }
        for n in 2..=9 {
            assert_eq!(path(n).automorphism_group().order(), Some(2), "P{n}");
        }
        assert_eq!(star(7).automorphism_group().order(), Some(720));
        assert_eq!(petersen().automorphism_group().order(), Some(120));
        assert_eq!(hypercube(3).automorphism_group().order(), Some(48));
        assert_eq!(hypercube(4).automorphism_group().order(), Some(384));
        assert_eq!(
            complete_bipartite(3, 4).automorphism_group().order(),
            Some(144)
        );
        assert_eq!(grid_2d(3, 3, false).automorphism_group().order(), Some(8));
        assert_eq!(grid_2d(3, 4, false).automorphism_group().order(), Some(4));
    }

    #[test]
    fn platonic_symmetry_groups() {
        let expected = [24u128, 48, 48, 120, 120];
        for (solid, order) in PlatonicSolid::ALL.into_iter().zip(expected) {
            let group = platonic(solid).automorphism_group();
            assert_eq!(group.order(), Some(order), "{solid:?}");
            assert!(group.is_vertex_transitive());
        }
    }

    #[test]
    fn metatron_family() {
        let k13 = metatron::metatron_cube().automorphism_group();
        assert_eq!(k13.order(), Some(6_227_020_800));
        assert!(k13.is_vertex_transitive());
        let fol = metatron::fruit_of_life().automorphism_group();
        assert_eq!(fol.order(), Some(12));
        assert_eq!(
            fol.orbits(),
            vec![vec![0], (1..=6).collect(), (7..=12).collect()]
        );
        // Lines up to √3 keep the D6 symmetry as well.
        assert_eq!(
            metatron::distance_graph(3f64.sqrt())
                .automorphism_group()
                .order(),
            Some(12)
        );
    }

    #[test]
    fn generators_are_automorphisms_and_order_is_label_invariant() {
        let g = platonic(PlatonicSolid::Dodecahedron);
        let group = g.automorphism_group();
        assert!(group.generators().iter().all(|p| is_automorphism(&g, p)));
        let mut rng = Rng::seed_from_u64(3);
        let mut perm: Vec<usize> = (0..g.num_nodes()).collect();
        rng.shuffle(&mut perm);
        assert_eq!(
            g.relabeled(&perm).unwrap().automorphism_group().order(),
            Some(120)
        );
    }

    #[test]
    fn weights_break_symmetry() {
        assert_eq!(cycle(4).automorphism_group().order(), Some(8));
        let g =
            Graph::from_weighted_edges(4, &[(0, 1, 2.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)])
                .unwrap();
        assert_eq!(g.automorphism_group().order(), Some(2));
    }

    #[test]
    fn large_complete_graph_order_overflows_gracefully() {
        let group = complete(40).automorphism_group();
        assert_eq!(group.order(), None);
        let expected: f64 = (1..=40).map(|k| (k as f64).log10()).sum();
        assert!((group.log10_order() - expected).abs() < 1e-9);
    }
}
