//! Orbitals of permutation groups on many points, from the orbits of point
//! stabilisers.
//!
//! Let `A` be an orbit of the group `G` on points and `a` its smallest point.
//! For `u ∈ A` take any `t_u ∈ G` with `t_u(a) = u`. Then `(u, v)` and
//! `(a, t_u⁻¹(v))` lie in the same orbital, and `(a, w)`, `(a, w')` do
//! exactly when an element of the stabiliser `G_a` maps `w` to `w'`. So the
//! orbitals with first point in `A` are the orbits of `G_a` on the points:
//! `|A|` times fewer labels than the `n·|A|` pairs, one row of `n` labels
//! per orbit on points.
//!
//! * `t_u⁻¹` is a path in a Schreier tree of `A`, built by breadth-first
//!   search over the generators, ten random elements of `G` and their
//!   inverses. The random elements make the tree shallower than the
//!   generators alone would (a few levels for hypercubes, Hamming and
//!   Johnson graphs); for groups whose random elements are still short words, such
//!   as a cycle and a transposition on many points, it can stay deep, and
//!   `of` costs a step per level.
//! * The orbits of `G_a` come from a union–find on the points. For small
//!   orbits it merges along every Schreier generator `t_{s(x)}⁻¹ s t_x`
//!   (`x ∈ A`, `s` a generator), which generate `G_a` (Schreier's lemma):
//!   exact. Otherwise it merges along elements `t_{g(a)}⁻¹ g` of `G_a` with
//!   `g` from the product replacement algorithm (Celler et al., Comm.
//!   Algebra 23, 4931 (1995), with Leedham-Green's accumulator; every
//!   generator other than the identity starts in a slot of its own), until
//!   [`QUIET`] of them in a row merge nothing. For a uniform `g` such an
//!   element is uniform in `G_a`, and one that leaves every current class in
//!   place lies in a proper subgroup whenever the classes are still finer
//!   than the orbits, so each draw then merges with probability at least ½;
//!   product replacement only approximates uniform elements. Merges that
//!   are missed split an orbital into several classes: the states that are
//!   invariant under `G` stay exact on the finer classes, which only cost
//!   coefficients.
//!
//! The orbitals are numbered by their smallest pair `(a, min class)`, as
//! the union–find on all pairs numbers them (when no merge is missed).

use crate::error::{QsoError, Result};
use crate::rng::Rng;

/// Product replacement slots (at least; one per generator), and random
/// elements added to the Schreier trees.
const SLOTS: usize = 10;
/// Product replacement steps before the first random element is used.
const WARM_UP: usize = 64;
/// Consecutive random elements of a stabiliser that merge nothing before
/// its orbits are taken as found.
pub(super) const QUIET: usize = 64;
/// Schreier generators are all used when the orbit, times the generators,
/// times the points, is at most this many point images.
const EXHAUSTIVE_WORK: usize = 1 << 26;

/// Most labels (orbits on points times points) kept: 2²⁷, 512 MiB, and
/// 16 bytes more for each orbital (its smallest pair and size).
pub const MAX_SUBORBIT_LABELS: usize = 1 << 27;

/// Most point images that the search for the orbitals is estimated to
/// compute (the random draws are counted at twice 64 per orbit on
/// points; draws that merge restart that count, so the estimate is a lower
/// one).
pub const MAX_SUBORBIT_WORK: f64 = 1e11;

/// The orbitals found, with their smallest pairs and sizes, ascending.
pub(super) type Found = (Suborbits, Vec<(u32, u32)>, Vec<u64>);

/// The orbitals of a group, one row of labels per orbit on points.
#[derive(Clone, Debug)]
pub(super) struct Suborbits {
    /// Permutations along the tree edges: generators, random elements and
    /// their inverses.
    perms: Vec<Vec<u32>>,
    /// The orbit of each point.
    orbit: Vec<u32>,
    /// The smallest point of each orbit.
    reps: Vec<u32>,
    /// For each point, the permutation (index into `perms`) that maps it
    /// one edge closer to its orbit's smallest point (unused there).
    step: Vec<u32>,
    /// For each orbit `A`, the orbital of `(min A, w)` for every point `w`.
    labels: Vec<Vec<u32>>,
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let up = parent[parent[x as usize] as usize];
        parent[x as usize] = up;
        x = up;
    }
    x
}

/// Join the classes of `a` and `b`, keeping the smaller root; whether they
/// were apart.
fn union(parent: &mut [u32], a: u32, b: u32) -> bool {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra == rb {
        return false;
    }
    parent[ra.max(rb) as usize] = ra.min(rb);
    true
}

fn inverse(p: &[u32]) -> Vec<u32> {
    let mut inv = vec![0u32; p.len()];
    for (u, &v) in p.iter().enumerate() {
        inv[v as usize] = u as u32;
    }
    inv
}

/// Random elements of a group from its generators: product replacement with
/// an accumulator (Leedham-Green's "rattle"). Each step replaces one slot by
/// its product with another and multiplies the accumulator by it.
struct ProductReplacement {
    slots: Vec<Vec<u32>>,
    accumulator: Vec<u32>,
    rng: Rng,
    scratch: Vec<u32>,
}

impl ProductReplacement {
    fn new(generators: &[Vec<u32>], n: usize, seed: u64) -> Self {
        let identity: Vec<u32> = (0..n as u32).collect();
        // Every generator in a slot of its own, at least SLOTS slots.
        let count = SLOTS.max(generators.len());
        let slots = (0..count)
            .map(|k| {
                if generators.is_empty() {
                    identity.clone()
                } else {
                    generators[k % generators.len()].clone()
                }
            })
            .collect();
        let mut pr = ProductReplacement {
            slots,
            accumulator: identity,
            rng: Rng::seed_from_u64(seed),
            scratch: vec![0; n],
        };
        for _ in 0..WARM_UP.max(4 * count) {
            pr.next();
        }
        pr
    }

    /// The next random element (the accumulator).
    fn next(&mut self) -> &[u32] {
        let m = self.slots.len() as u64;
        let i = self.rng.below(m) as usize;
        let j = (i + 1 + self.rng.below(m - 1) as usize) % m as usize;
        let (left, right) = (self.rng.below(2) == 0, self.rng.below(2) == 0);
        let (a, b) = (&self.slots[i], &self.slots[j]);
        // slot_i ← slot_i·slot_j or slot_j·slot_i, slot_j inverted or not.
        let out = &mut self.scratch;
        if right {
            let inv = inverse(b);
            for u in 0..a.len() {
                out[u] = if left {
                    a[inv[u] as usize]
                } else {
                    inv[a[u] as usize]
                };
            }
        } else {
            for u in 0..a.len() {
                out[u] = if left {
                    a[b[u] as usize]
                } else {
                    b[a[u] as usize]
                };
            }
        }
        std::mem::swap(&mut self.slots[i], &mut self.scratch);
        // accumulator ← accumulator · slot_i
        let (acc, slot) = (&self.accumulator, &self.slots[i]);
        for (out, &x) in self.scratch.iter_mut().zip(slot) {
            *out = acc[x as usize];
        }
        std::mem::swap(&mut self.accumulator, &mut self.scratch);
        &self.accumulator
    }
}

impl Suborbits {
    /// The orbitals of the group generated by `generators` (checked
    /// permutations of `0..n`), their smallest pairs and sizes, ascending.
    pub(super) fn new(n: usize, generators: &[Vec<usize>], seed: u64) -> Result<Found> {
        Self::with_exhaustive_work(n, generators, seed, EXHAUSTIVE_WORK)
    }

    /// As [`Suborbits::new`], with all Schreier generators used up to
    /// `exhaustive_work` point images per orbit (0: random elements only).
    pub(super) fn with_exhaustive_work(
        n: usize,
        generators: &[Vec<usize>],
        seed: u64,
        exhaustive_work: usize,
    ) -> Result<Found> {
        let too_many = |k: usize| {
            QsoError::invalid(
                "symmetries",
                format!(
                    "{k} orbits on {n} points would take {} labels (limit {MAX_SUBORBIT_LABELS}): \
                     use more symmetries",
                    k.saturating_mul(n)
                ),
            )
        };
        // At least one orbit: refused before anything of size n is built.
        if n > MAX_SUBORBIT_LABELS {
            return Err(too_many(1));
        }
        // The generators without the identity and repetitions.
        let mut gens: Vec<Vec<u32>> = Vec::new();
        for g in generators {
            let g: Vec<u32> = g.iter().map(|&x| x as u32).collect();
            let moves = g.iter().enumerate().any(|(u, &x)| x as usize != u);
            if moves && !gens.contains(&g) {
                gens.push(g);
            }
        }
        if gens.is_empty() && n.saturating_mul(n) > MAX_SUBORBIT_LABELS {
            return Err(too_many(n));
        }
        // Orbits on points, numbered by their smallest point.
        let mut parent: Vec<u32> = (0..n as u32).collect();
        for g in &gens {
            for (u, &gu) in g.iter().enumerate() {
                union(&mut parent, u as u32, gu);
            }
        }
        let mut orbit = vec![u32::MAX; n];
        let mut reps = Vec::new();
        for u in 0..n {
            let r = find(&mut parent, u as u32) as usize;
            if orbit[r] == u32::MAX {
                orbit[r] = reps.len() as u32;
                reps.push(r as u32);
            }
            orbit[u] = orbit[r];
        }
        let k = reps.len();
        if k.saturating_mul(n) > MAX_SUBORBIT_LABELS {
            return Err(too_many(k));
        }
        let mut sizes_of_orbit = vec![0usize; k];
        for &o in &orbit {
            sizes_of_orbit[o as usize] += 1;
        }
        // Tree edges: generators, random elements and all their inverses.
        let mut random = ProductReplacement::new(&gens, n, seed);
        let mut perms: Vec<Vec<u32>> = gens.clone();
        if !gens.is_empty() {
            for _ in 0..SLOTS {
                perms.push(random.next().to_vec());
            }
        }
        let forward = perms.len();
        for p in 0..forward {
            let inv = inverse(&perms[p]);
            perms.push(inv);
        }
        let inverse_of = |p: usize| {
            if p < forward {
                p + forward
            } else {
                p - forward
            }
        };
        // Breadth-first trees from each orbit's smallest point.
        let mut step = vec![u32::MAX; n];
        let mut depth = vec![0u32; n];
        let mut queue = std::collections::VecDeque::new();
        for &r in &reps {
            step[r as usize] = u32::MAX - 1;
            queue.push_back(r);
            while let Some(x) = queue.pop_front() {
                for (p, perm) in perms.iter().enumerate() {
                    let y = perm[x as usize];
                    if step[y as usize] == u32::MAX {
                        // y = p(x): p⁻¹ takes y back to x.
                        step[y as usize] = inverse_of(p) as u32;
                        depth[y as usize] = depth[x as usize] + 1;
                        queue.push_back(y);
                    }
                }
            }
        }
        let suborbits = Suborbits {
            perms,
            orbit,
            reps,
            step,
            labels: Vec::new(),
        };
        // The work: each stabiliser takes at least QUIET draws (or all its
        // Schreier generators), each of n images through a path. Schreier
        // generators are used only while the whole estimate stays within
        // the limit.
        let deepest = depth.iter().copied().max().unwrap_or(0) as f64;
        let estimate = |exhaustive_work: usize| {
            sizes_of_orbit
                .iter()
                .map(|&size| {
                    let schreier = size * gens.len().max(1);
                    let draws = if schreier.saturating_mul(n) <= exhaustive_work {
                        schreier as f64
                    } else {
                        2.0 * QUIET as f64
                    };
                    draws * n as f64 * (2.0 * deepest + 2.0)
                })
                .sum::<f64>()
        };
        let exhaustive_work = if estimate(exhaustive_work) <= MAX_SUBORBIT_WORK {
            exhaustive_work
        } else {
            0
        };
        let estimate = estimate(exhaustive_work);
        if estimate > MAX_SUBORBIT_WORK {
            return Err(QsoError::invalid(
                "symmetries",
                format!(
                    "the orbitals of {k} orbits on {n} points need about {estimate:.1e} point \
                     images (limit {MAX_SUBORBIT_WORK:.1e})"
                ),
            ));
        }
        let mut suborbits = suborbits;
        let mut labels = Vec::with_capacity(k);
        let mut representatives = Vec::new();
        let mut sizes = Vec::new();
        let mut h = vec![0u32; n];
        let mut path = Vec::new();
        for (o, &orbit_size) in sizes_of_orbit.iter().enumerate() {
            let a = suborbits.reps[o];
            let mut classes: Vec<u32> = (0..n as u32).collect();
            let schreier = orbit_size * gens.len();
            if gens.is_empty() {
                // The trivial group: every point its own class.
            } else if schreier.saturating_mul(n) <= exhaustive_work {
                for x in (0..n as u32).filter(|&x| suborbits.orbit[x as usize] == o as u32) {
                    for s in &gens {
                        // h = t_{s(x)}⁻¹ · s · t_x, which fixes a.
                        suborbits.path_to_rep(s[x as usize], &mut path);
                        let back = path.clone();
                        suborbits.path_to_rep(x, &mut path);
                        for w in 0..n as u32 {
                            let mut y = suborbits.lift(&path, w);
                            y = s[y as usize];
                            for &p in &back {
                                y = suborbits.perms[p as usize][y as usize];
                            }
                            h[w as usize] = y;
                        }
                        for w in 0..n as u32 {
                            union(&mut classes, w, h[w as usize]);
                        }
                    }
                }
            } else {
                let mut quiet = 0;
                while quiet < QUIET {
                    let g = random.next();
                    // h = t_{g(a)}⁻¹ · g, which fixes a.
                    suborbits.path_to_rep(g[a as usize], &mut path);
                    for w in 0..n {
                        let mut y = g[w];
                        for &p in &path {
                            y = suborbits.perms[p as usize][y as usize];
                        }
                        h[w] = y;
                    }
                    let mut merged = false;
                    for w in 0..n as u32 {
                        merged |= union(&mut classes, w, h[w as usize]);
                    }
                    quiet = if merged { 0 } else { quiet + 1 };
                }
            }
            // Label the classes in order of their smallest points, which are
            // their roots.
            let mut label = vec![u32::MAX; n];
            let first = representatives.len() as u32;
            let mut count = vec![0u64; 0];
            let mut index_of_root = vec![u32::MAX; n];
            for (w, slot) in label.iter_mut().enumerate() {
                let r = find(&mut classes, w as u32) as usize;
                if index_of_root[r] == u32::MAX {
                    index_of_root[r] = count.len() as u32;
                    count.push(0);
                    representatives.push((a, r as u32));
                }
                let c = index_of_root[r];
                count[c as usize] += 1;
                *slot = first + c;
            }
            sizes.extend(count.iter().map(|&c| c * orbit_size as u64));
            labels.push(label);
        }
        suborbits.labels = labels;
        Ok((suborbits, representatives, sizes))
    }

    /// The permutations (indices into `perms`) that take `x` to its orbit's
    /// smallest point, in the order they apply.
    fn path_to_rep(&self, mut x: u32, path: &mut Vec<u32>) {
        path.clear();
        let rep = self.reps[self.orbit[x as usize] as usize];
        while x != rep {
            let p = self.step[x as usize];
            path.push(p);
            x = self.perms[p as usize][x as usize];
        }
    }

    /// `t_x(w)` for the path of `x`: the inverse of the path, which takes
    /// the smallest point to `x`.
    fn lift(&self, path: &[u32], mut w: u32) -> u32 {
        let half = self.perms.len() / 2;
        for &p in path.iter().rev() {
            let inv = if (p as usize) < half {
                p as usize + half
            } else {
                p as usize - half
            };
            w = self.perms[inv][w as usize];
        }
        w
    }

    /// The orbital of the pair `(u, v)`.
    pub(super) fn of(&self, u: usize, v: usize) -> usize {
        let o = self.orbit[u] as usize;
        let rep = self.reps[o];
        let (mut x, mut y) = (u as u32, v as u32);
        while x != rep {
            let p = &self.perms[self.step[x as usize] as usize];
            x = p[x as usize];
            y = p[y as usize];
        }
        self.labels[o][y as usize] as usize
    }
}
