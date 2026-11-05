//! Interacting walkers on any graph, reduced by a group of its
//! automorphisms, on the orbits of configurations found from minimal
//! images, without visiting every configuration.
//!
//! [`SymmetricInteractingWalk`](super::SymmetricInteractingWalk) (1.43)
//! finds the orbits of a group on the configurations by a union–find over
//! every configuration; [`CoordinateInteractingWalk`](super::CoordinateInteractingWalk)
//! (1.53) generates them from the start with canonical forms that only
//! Hamming, Johnson and Kneser graphs have. Here the canonical form of a
//! configuration is its *minimal image* under the group (Jefferson,
//! Jonauskyte, Pfeiffer and Waldecker, J. Algebra 521, 481 (2019)): the
//! lexicographically smallest configuration in its orbit, found along a
//! chain of point stabilisers.
//!
//! * **Groups.** A group is held by generators and its exact order (from
//!   the base and strong generating set of
//!   [`AutomorphismGroup`]).
//!   The stabiliser of a point `m` in a group `K` of known order has order
//!   `|K|/|m^K|`, so a randomised Schreier–Sims algorithm with base `[m, …]`
//!   is stopped exactly when its basic orbits multiply to the order (a
//!   proper subgroup is at least twice smaller): the stabiliser's strong
//!   generators are then complete, whatever the random elements.
//! * **Orbit forests.** For every group met, the orbits on the nodes are
//!   found by breadth-first search from the smallest node of each orbit
//!   (its root) over the generators and a few random elements, which keeps
//!   the trees shallow. Walking a node up its tree to the root with the
//!   inverse generators maps the whole configuration by an element of the
//!   group.
//! * **Minimal images.** For bosons and fermions (multisets of nodes) the
//!   smallest node any element can bring a particle to is the smallest
//!   root among the particles' nodes (preferring nodes holding more
//!   particles, whose image then repeats). Every node that reaches it is a
//!   branch: the configuration is mapped so that node sits at the root,
//!   and the search goes on in the stabiliser of the root. Branches that
//!   fall behind are dropped; those that survive to the end reach the
//!   minimal image, one per coset of the final stabiliser, so
//!   `|Stab(c)| = (branches) · |stabiliser of the image's nodes|`. For
//!   fermions every surviving branch must order the particles with the same
//!   parity; otherwise a symmetry maps the configuration to minus itself
//!   and the orbit carries no invariant state. For distinguishable
//!   particles (sequences) each particle in turn goes to its root, without
//!   branching.
//! * **Stabilisers met** are kept by the group they stabilise (keyed by the
//!   root), and found again by their fixed nodes when another order of the
//!   same roots leads to them; the stabiliser of a fixed node is the group
//!   itself, and one of order one ends the search (the remaining particles
//!   stay where they are). Their tables count against
//!   [`MAX_STABILISER_LABELS`]: a group with many small stabilisers (a large
//!   abelian group, say) fills them, and is better reduced by
//!   [`SymmetricInteractingWalk`](super::SymmetricInteractingWalk).
//! * **Orders are checked**: a run that cannot reach the order it was told
//!   fails, and two dozen random elements sifted after it would show a
//!   larger group, so a wrong order is an error (the whole group's when its
//!   first stabiliser is built, before the search).
//!
//! The orbits reachable from the start are then found by breadth-first
//! search, as in 1.53: the Hamiltonian of
//! [`InteractingWalk`](super::InteractingWalk) applied to each orbit's
//! representative gives configurations whose minimal images name orbits,
//! old or new. A graph whose stabiliser of the start leaves the walk on few
//! orbits, such as Paley graphs, circulants and strongly regular graphs,
//! is reduced as far as its symmetry allows, beyond the configurations a
//! union–find could visit.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use num_complex::Complex64 as C64;

use super::symmetric::MAX_ORBIT_ENTRIES;
use super::{Statistics, Terms, propagate};
use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::graph::automorphism::AutomorphismGroup;
use crate::linalg::{CsrMatrix, RMatrix, eigh_real};
use crate::rng::Rng;
use crate::walks::symmetric::check_permutation;
use crate::walks::{DENSE_LIMIT, Generator};

/// Most orbits found (each keeps its representative and its key).
pub const MAX_IMAGE_ORBITS: usize = 1 << 22;

/// Most particles (`6! = 720` branches at most per level).
pub const MAX_IMAGE_PARTICLES: usize = 6;

/// Most node labels the groups met may hold together, 2²⁸ (1 GiB): each
/// holds four per node for its orbit forest and one per node for each of
/// its generators and random elements (kept as inverse permutations).
pub const MAX_STABILISER_LABELS: usize = 1 << 28;

/// Random elements added to each orbit forest's generators.
const FOREST_RANDOM: usize = 10;

/// Random elements a Schreier–Sims run may draw without growing its chain
/// before it gives up (the order it was told is then too large).
const MAX_IDLE_DRAWS: usize = 4096;

/// Random elements sifted after a Schreier–Sims run has reached the order:
/// one that sifts to a non-identity residue shows the order too small
/// (each misses with probability at most ½).
const VERIFY_DRAWS: usize = 24;

const NONE: u32 = u32::MAX;

type Perm = Vec<u32>;

/// `a ∘ b`: `x ↦ a(b(x))`.
fn compose(a: &[u32], b: &[u32]) -> Perm {
    b.iter().map(|&x| a[x as usize]).collect()
}

fn invert(a: &[u32]) -> Perm {
    let mut inv = vec![0u32; a.len()];
    for (x, &y) in a.iter().enumerate() {
        inv[y as usize] = x as u32;
    }
    inv
}

fn is_identity(a: &[u32]) -> bool {
    a.iter().enumerate().all(|(x, &y)| x as u32 == y)
}

/// Random elements of the group generated by `gens` by product
/// replacement (Celler et al., Comm. Algebra 23, 4931 (1995), with
/// Leedham-Green's accumulator).
struct Random {
    slots: Vec<Perm>,
    acc: Perm,
    rng: Rng,
}

impl Random {
    fn new(n: usize, gens: &[Perm], seed: u64) -> Self {
        let identity: Perm = (0..n as u32).collect();
        let mut slots: Vec<Perm> = gens.iter().filter(|g| !is_identity(g)).cloned().collect();
        if slots.is_empty() {
            slots.push(identity.clone());
        }
        let base = slots.len();
        while slots.len() < 10 {
            let k = slots.len();
            slots.push(slots[k % base].clone());
        }
        let mut r = Random {
            slots,
            acc: identity,
            rng: Rng::seed_from_u64(seed),
        };
        for _ in 0..64 {
            r.next();
        }
        r
    }

    fn next(&mut self) -> Perm {
        let m = self.slots.len();
        let i = self.rng.below(m as u64) as usize;
        let mut j = self.rng.below(m as u64 - 1) as usize;
        if j >= i {
            j += 1;
        }
        let other = if self.rng.below(2) == 0 {
            self.slots[j].clone()
        } else {
            invert(&self.slots[j])
        };
        self.slots[i] = if self.rng.below(2) == 0 {
            compose(&self.slots[i], &other)
        } else {
            compose(&other, &self.slots[i])
        };
        self.acc = compose(&self.acc, &self.slots[i]);
        self.acc.clone()
    }
}

/// A Schreier tree: the generator (index) that reaches each node of the
/// orbit from its parent, `NONE` outside the orbit and at the root.
#[derive(Clone)]
struct Tree {
    root: u32,
    via: Vec<u32>,
    parent: Vec<u32>,
    len: usize,
}

impl Tree {
    fn new(n: usize, root: u32, gens: &[usize], strong: &[(Perm, Perm)]) -> Self {
        let mut via = vec![NONE; n];
        let mut parent = vec![NONE; n];
        let mut seen = vec![false; n];
        seen[root as usize] = true;
        let mut queue = vec![root];
        let mut head = 0;
        while head < queue.len() {
            let x = queue[head];
            head += 1;
            for &g in gens {
                let y = strong[g].0[x as usize];
                if !seen[y as usize] {
                    seen[y as usize] = true;
                    via[y as usize] = g as u32;
                    parent[y as usize] = x;
                    queue.push(y);
                }
            }
        }
        Tree {
            root,
            via,
            parent,
            len: queue.len(),
        }
    }

    fn contains(&self, x: u32) -> bool {
        x == self.root || self.parent[x as usize] != NONE
    }
}

/// The strong generators of the stabiliser of `point` in the group
/// generated by `gens`, of order `e^{log_order}`, by the randomised
/// Schreier–Sims algorithm with base `[point, …]`, stopped when the
/// basic orbits reach the order.
fn stabiliser(n: usize, gens: &[Perm], log_order: f64, point: u32, seed: u64) -> Result<Vec<Perm>> {
    let mut strong: Vec<(Perm, Perm)> = Vec::new();
    // Levels: base point, generators (indices into `strong`), tree.
    let mut levels: Vec<(u32, Vec<usize>, Tree)> =
        vec![(point, Vec::new(), Tree::new(n, point, &[], &strong))];
    let mut random = Random::new(n, gens, seed);
    let reached = |levels: &[(u32, Vec<usize>, Tree)]| -> f64 {
        levels.iter().map(|l| (l.2.len as f64).ln()).sum()
    };
    let mut idle = 0;
    while reached(&levels) < log_order - 0.3 {
        let g = random.next();
        // Sift.
        let mut h = g;
        let mut fail = levels.len();
        for (j, (b, _, tree)) in levels.iter().enumerate() {
            let mut beta = h[*b as usize];
            if !tree.contains(beta) {
                fail = j;
                break;
            }
            while beta != *b {
                let inv = &strong[tree.via[beta as usize] as usize].1;
                h = compose(inv, &h);
                beta = tree.parent[beta as usize];
            }
        }
        if fail == levels.len() && is_identity(&h) {
            idle += 1;
            if idle > MAX_IDLE_DRAWS {
                return Err(QsoError::invalid(
                    "group",
                    "the stabiliser chain does not reach the group's order (is the order right?)",
                ));
            }
            continue;
        }
        idle = 0;
        if fail == levels.len() {
            let moved = h
                .iter()
                .enumerate()
                .find(|&(x, &y)| x as u32 != y)
                .map(|(x, _)| x as u32)
                .expect("not the identity");
            levels.push((moved, Vec::new(), Tree::new(n, moved, &[], &strong)));
        }
        let inv = invert(&h);
        strong.push((h, inv));
        let idx = strong.len() - 1;
        for level in levels.iter_mut().take(fail + 1) {
            level.1.push(idx);
            level.2 = Tree::new(n, level.0, &level.1, &strong);
        }
    }
    // The order told is reached; a larger group would leave residues.
    for _ in 0..VERIFY_DRAWS {
        let mut h = random.next();
        for (b, _, tree) in &levels {
            let mut beta = h[*b as usize];
            if !tree.contains(beta) {
                return Err(QsoError::invalid(
                    "group",
                    "the group is larger than the order it was given",
                ));
            }
            while beta != *b {
                let inv = &strong[tree.via[beta as usize] as usize].1;
                h = compose(inv, &h);
                beta = tree.parent[beta as usize];
            }
        }
        if !is_identity(&h) {
            return Err(QsoError::invalid(
                "group",
                "the group is larger than the order it was given",
            ));
        }
    }
    // The strong generators that fix the point: those of the levels below.
    let mut out: Vec<usize> = levels[1..]
        .iter()
        .flat_map(|l| l.1.iter().copied())
        .collect();
    out.sort_unstable();
    out.dedup();
    Ok(out.into_iter().map(|i| strong[i].0.clone()).collect())
}

/// A configuration's minimal image, sorted for bosons and fermions, padded
/// with `NONE`.
type Key = [u32; MAX_IMAGE_PARTICLES];

/// Hasher for keys (multiply and rotate; the default SipHash costs more
/// than the rest of a hop).
#[derive(Clone, Copy, Default)]
struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(4) {
            let mut w = [0u8; 4];
            w[..chunk.len()].copy_from_slice(chunk);
            self.write_u32(u32::from_le_bytes(w));
        }
    }

    fn write_u32(&mut self, v: u32) {
        self.0 = (self.0.rotate_left(29) ^ u64::from(v)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}

type KeyMap<V> = HashMap<Key, V, BuildHasherDefault<KeyHasher>>;

/// The orbits of a group on the nodes: per node the root of its orbit
/// (its smallest node), the generator that reaches it from its parent and
/// the parent; per root the orbit's size.
struct Forest {
    root: Vec<u32>,
    via: Vec<u32>,
    parent: Vec<u32>,
    size: Vec<u32>,
}

/// A group met in the search: its order, the nodes it fixes beyond the
/// whole group's (sorted; its key in the cache), and, unless it is
/// trivial, the inverses of its generators and of a few random elements
/// (how many are generators) with its orbit forest; its stabilisers of
/// roots, once met.
struct Stage {
    log_order: f64,
    fixed: Vec<u32>,
    inner: Option<(Vec<Perm>, usize, Forest)>,
    children: RwLock<HashMap<u32, Arc<Stage>>>,
}

impl Stage {
    fn trivial(fixed: Vec<u32>) -> Self {
        Stage {
            log_order: 0.0,
            fixed,
            inner: None,
            children: RwLock::new(HashMap::new()),
        }
    }

    fn new(n: usize, gens: Vec<Perm>, log_order: f64, fixed: Vec<u32>, seed: u64) -> Result<Self> {
        if log_order < 0.3 {
            return Ok(Stage::trivial(fixed));
        }
        if gens.iter().all(|g| is_identity(g)) {
            return Err(QsoError::invalid(
                "group",
                "the generators are trivial but the order is not",
            ));
        }
        let count = gens.len();
        let mut all = gens;
        // Random elements keep the trees shallow where generators are few.
        if count < FOREST_RANDOM {
            let mut random = Random::new(n, &all, seed);
            for _ in 0..FOREST_RANDOM {
                all.push(random.next());
            }
        }
        let mut root = vec![NONE; n];
        let mut via = vec![NONE; n];
        let mut parent = vec![NONE; n];
        let mut size = vec![0u32; n];
        let mut queue = Vec::new();
        for start in 0..n as u32 {
            if root[start as usize] != NONE {
                continue;
            }
            root[start as usize] = start;
            queue.clear();
            queue.push(start);
            let mut head = 0;
            while head < queue.len() {
                let x = queue[head];
                head += 1;
                for (g, p) in all.iter().enumerate() {
                    let y = p[x as usize];
                    if root[y as usize] == NONE {
                        root[y as usize] = start;
                        via[y as usize] = g as u32;
                        parent[y as usize] = x;
                        queue.push(y);
                    }
                }
            }
            size[start as usize] = queue.len() as u32;
        }
        let inverses = all.iter().map(|g| invert(g)).collect();
        Ok(Stage {
            log_order,
            fixed,
            inner: Some((
                inverses,
                count,
                Forest {
                    root,
                    via,
                    parent,
                    size,
                },
            )),
            children: RwLock::new(HashMap::new()),
        })
    }

    /// Node labels held (see [`MAX_STABILISER_LABELS`]).
    fn labels(&self, n: usize) -> usize {
        match &self.inner {
            Some((inverses, _, _)) => (4 + inverses.len()) * n,
            None => 0,
        }
    }

    /// Map `x` to its root, and every point of `pts` by the same element.
    fn lift(&self, mut x: u32, pts: &mut [u32]) {
        let Some((inverses, _, forest)) = &self.inner else {
            return;
        };
        while forest.parent[x as usize] != NONE {
            let inv = &inverses[forest.via[x as usize] as usize];
            for p in pts.iter_mut() {
                *p = inv[*p as usize];
            }
            x = forest.parent[x as usize];
        }
    }

    fn root(&self, x: u32) -> u32 {
        match &self.inner {
            Some((_, _, f)) => f.root[x as usize],
            None => x,
        }
    }

    fn orbit_size(&self, root: u32) -> u32 {
        match &self.inner {
            Some((_, _, f)) => f.size[root as usize],
            None => 1,
        }
    }

    /// The generators (not the random elements).
    fn generators(&self) -> Vec<Perm> {
        match &self.inner {
            Some((inverses, count, _)) => inverses[..*count].iter().map(|g| invert(g)).collect(),
            None => Vec::new(),
        }
    }
}

/// A configuration's minimal image.
struct Form {
    key: Key,
    /// Parity of the particles' order in the image (fermions).
    odd: bool,
    /// `ln |Stab(c)|`.
    log_stabiliser: f64,
}

/// Minimal images under a group, with the stabilisers met cached: each
/// group keeps its stabilisers of roots, and a map from the fixed nodes
/// finds the same group reached in another order.
struct Images {
    n: usize,
    group: Arc<Stage>,
    trivial: Arc<Stage>,
    by_set: Mutex<HashMap<Vec<u32>, Arc<Stage>>>,
    labels: AtomicUsize,
}

impl std::fmt::Debug for Images {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Images {{ nodes: {}, labels: {} }}",
            self.n,
            self.labels.load(Ordering::Relaxed)
        )
    }
}

impl Images {
    /// The images under the group of these generators and order; its order
    /// is checked by building the stabiliser of its first moved root.
    fn new(n: usize, generators: &[Vec<usize>], log_order: f64) -> Result<Self> {
        let gens: Vec<Perm> = generators
            .iter()
            .map(|g| g.iter().map(|&x| x as u32).collect())
            .collect();
        let group = Arc::new(Stage::new(n, gens, log_order, Vec::new(), 1)?);
        let images = Images {
            n,
            labels: AtomicUsize::new(group.labels(n)),
            group: group.clone(),
            trivial: Arc::new(Stage::trivial(Vec::new())),
            by_set: Mutex::new(HashMap::new()),
        };
        if images.labels.load(Ordering::Relaxed) > MAX_STABILISER_LABELS {
            return Err(QsoError::invalid(
                "group",
                format!("the group's tables exceed {MAX_STABILISER_LABELS} node labels"),
            ));
        }
        // A Schreier–Sims run fails on an order too large, and its sifts
        // after the order is reached show one too small.
        if let Some((_, _, forest)) = &group.inner
            && let Some(m) =
                (0..n as u32).find(|&x| forest.root[x as usize] == x && forest.size[x as usize] > 1)
        {
            stabiliser(n, &group.generators(), log_order, m, 2)?;
        }
        Ok(images)
    }

    /// The stabiliser of the root `m` in `parent`.
    fn child(&self, parent: &Arc<Stage>, m: u32) -> Result<Arc<Stage>> {
        if parent.inner.is_none() || parent.orbit_size(m) == 1 {
            // A fixed node: the same group.
            return Ok(parent.clone());
        }
        if let Some(c) = parent
            .children
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .get(&m)
        {
            return Ok(c.clone());
        }
        let log = parent.log_order - (parent.orbit_size(m) as f64).ln();
        let child = if log < 0.3 {
            self.trivial.clone()
        } else {
            let mut fixed = parent.fixed.clone();
            let at = fixed.partition_point(|&x| x < m);
            fixed.insert(at, m);
            // Built under the lock: groups are built once, and rarely.
            let mut by_set = self.by_set.lock().unwrap_or_else(|p| p.into_inner());
            match by_set.get(&fixed) {
                Some(c) => c.clone(),
                None => {
                    let seed = fixed
                        .iter()
                        .fold(0x9e37_79b9u64, |h, &x| h.rotate_left(17) ^ u64::from(x));
                    let gens = stabiliser(self.n, &parent.generators(), parent.log_order, m, seed)?;
                    let stage = Stage::new(self.n, gens, log, fixed.clone(), seed ^ 1)?;
                    let added = stage.labels(self.n);
                    let total = self.labels.fetch_add(added, Ordering::Relaxed) + added;
                    if total > MAX_STABILISER_LABELS {
                        return Err(QsoError::invalid(
                            "group",
                            format!(
                                "the stabilisers met exceed {MAX_STABILISER_LABELS} node labels \
                                 (the group has many small stabilisers: its orbits on \
                                 configurations are better found by SymmetricInteractingWalk)"
                            ),
                        ));
                    }
                    let c = Arc::new(stage);
                    by_set.insert(fixed, c.clone());
                    c
                }
            }
        };
        parent
            .children
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .insert(m, child.clone());
        Ok(child)
    }

    /// The minimal image of `sites` (sorted for bosons and fermions);
    /// `None` when a symmetry maps the fermionic configuration to minus
    /// itself.
    fn form(&self, sites: &[usize], statistics: Statistics) -> Result<Option<Form>> {
        let k = sites.len();
        let mut start: Key = [NONE; MAX_IMAGE_PARTICLES];
        for (s, &x) in start.iter_mut().zip(sites) {
            *s = x as u32;
        }
        if statistics == Statistics::Distinguishable {
            let mut pts = start;
            let mut placed: Key = [NONE; MAX_IMAGE_PARTICLES];
            let mut np = 0;
            let mut stage = self.group.clone();
            let mut log = stage.log_order;
            for i in 0..k {
                let x = pts[i];
                if placed[..np].contains(&x) {
                    continue;
                }
                if stage.inner.is_none() {
                    break;
                }
                let root = stage.root(x);
                stage.lift(x, &mut pts[..k]);
                log = stage.log_order - (stage.orbit_size(root) as f64).ln();
                placed[np] = root;
                np += 1;
                if i + 1 < k {
                    stage = self.child(&stage, root)?;
                }
            }
            return Ok(Some(Form {
                key: pts,
                odd: false,
                log_stabiliser: if log < 0.3 { 0.0 } else { log },
            }));
        }
        // Multisets: branches of particle images (and the particles placed,
        // as a mask), all sharing the roots met.
        let all: u8 = ((1u16 << k) - 1) as u8;
        let mut branches: Vec<(Key, u8)> = vec![(start, 0)];
        let mut next: Vec<(Key, u8)> = Vec::new();
        let mut stage = self.group.clone();
        let mut log = stage.log_order;
        while branches[0].1 != all {
            if stage.inner.is_none() {
                // Everything left stays: the smallest sorted image wins.
                let sorted = |p: &Key| {
                    let mut s = *p;
                    s[..k].sort_unstable();
                    s
                };
                let best = branches
                    .iter()
                    .map(|(p, _)| sorted(p))
                    .min()
                    .expect("a branch");
                branches.retain(|(p, _)| sorted(p) == best);
                log = 0.0;
                break;
            }
            let open = |done: u8, i: usize| done >> i & 1 == 0;
            // The best (root, multiplicity) over the branches' open nodes.
            let mut best: Option<(u32, usize)> = None;
            for (pts, done) in &branches {
                for i in (0..k).filter(|&i| open(*done, i)) {
                    let mult = (0..k)
                        .filter(|&j| open(*done, j) && pts[j] == pts[i])
                        .count();
                    let cand = (stage.root(pts[i]), mult);
                    if best.is_none_or(|b| cand.0 < b.0 || (cand.0 == b.0 && cand.1 > b.1)) {
                        best = Some(cand);
                    }
                }
            }
            let (root, mult) = best.expect("an open particle");
            next.clear();
            for &(pts, done) in &branches {
                let mut tried: Key = [NONE; MAX_IMAGE_PARTICLES];
                let mut nt = 0;
                for i in (0..k).filter(|&i| open(done, i)) {
                    let x = pts[i];
                    if tried[..nt].contains(&x) {
                        continue;
                    }
                    tried[nt] = x;
                    nt += 1;
                    let m = (0..k).filter(|&j| open(done, j) && pts[j] == x).count();
                    if m != mult || stage.root(x) != root {
                        continue;
                    }
                    let mut p = pts;
                    stage.lift(x, &mut p[..k]);
                    let mut d = done;
                    for j in (0..k).filter(|&j| open(done, j) && pts[j] == x) {
                        d |= 1 << j;
                    }
                    next.push((p, d));
                }
            }
            log = stage.log_order - (stage.orbit_size(root) as f64).ln();
            std::mem::swap(&mut branches, &mut next);
            if branches[0].1 != all {
                stage = self.child(&stage, root)?;
            }
        }
        let parity = |p: &Key| {
            let mut odd = false;
            for a in 0..k {
                for b in a + 1..k {
                    if p[a] > p[b] {
                        odd = !odd;
                    }
                }
            }
            odd
        };
        let odd = parity(&branches[0].0);
        if statistics == Statistics::Fermions && branches.iter().any(|(p, _)| parity(p) != odd) {
            return Ok(None);
        }
        let mut key = branches[0].0;
        key[..k].sort_unstable();
        let log = if log < 0.3 { 0.0 } else { log };
        Ok(Some(Form {
            key,
            odd,
            log_stabiliser: (branches.len() as f64).ln() + log,
        }))
    }
}

/// An orbit: its representative (the first configuration met in it), the
/// parity of the representative's form and `ln |Stab|`.
#[derive(Clone, Debug)]
struct Orbit {
    sites: Vec<usize>,
    odd: bool,
    log_stabiliser: f64,
}

/// Interacting walkers on any graph, reduced by a group of its
/// automorphisms, on the orbits reachable from the start found by minimal
/// images (see the module documentation): the same Hamiltonian as
/// [`InteractingWalk`](super::InteractingWalk), reduced exactly as
/// [`SymmetricInteractingWalk`](super::SymmetricInteractingWalk) reduces
/// it.
#[derive(Debug)]
pub struct GroupInteractingWalk {
    terms: Terms,
    statistics: Statistics,
    generator: Generator,
    gamma: f64,
    start: Vec<usize>,
    log_group: f64,
    images: Arc<Images>,
    orbits: Vec<Orbit>,
    index: KeyMap<u32>,
    hopping: Vec<(usize, usize, f64)>,
    same: Vec<f64>,
    adjacent: Vec<f64>,
    potential: Vec<f64>,
    hamiltonian: CsrMatrix,
    spectral: OnceLock<Option<(Vec<f64>, RMatrix)>>,
}

/// Where a configuration lies.
enum Lookup {
    Found(usize, f64),
    NoInvariant,
    Unreachable,
}

/// The hops from one orbit into an orbit not yet found, merged.
struct Hop {
    key: Key,
    sites: Vec<usize>,
    odd: bool,
    log_stabiliser: f64,
    amplitude: f64,
}

/// `ln C(n, k)` for real `n ≥ k`.
fn log_binomial(n: f64, k: usize) -> f64 {
    (0..k).map(|i| ((n - i as f64) / (i + 1) as f64).ln()).sum()
}

impl GroupInteractingWalk {
    /// `start.len()` particles of the given statistics on `graph`, hopping
    /// at rate `gamma` (`h = −γA` or `γL`), without interaction, reduced by
    /// `group`, a group of automorphisms of the graph (weights included)
    /// with its exact order, on the orbits reachable from the orbit of
    /// `start`. The walk's states are invariant under the group: the start
    /// is the normalised orbit state of the start configuration (the
    /// configuration itself when the group fixes it).
    pub fn new(
        graph: &Graph,
        generator: Generator,
        gamma: f64,
        statistics: Statistics,
        start: &[usize],
        group: &AutomorphismGroup,
    ) -> Result<Self> {
        if !(gamma.is_finite() && gamma > 0.0) {
            return Err(QsoError::invalid(
                "gamma",
                "hopping rate must be positive and finite",
            ));
        }
        let k = start.len();
        if k == 0 || k > MAX_IMAGE_PARTICLES {
            return Err(QsoError::invalid(
                "start",
                format!("need 1 to {MAX_IMAGE_PARTICLES} particles"),
            ));
        }
        let n = graph.num_nodes();
        if n >= NONE as usize {
            return Err(QsoError::invalid("graph", "too many nodes"));
        }
        if let Some(&u) = start.iter().find(|&&u| u >= n) {
            return Err(QsoError::invalid("start", format!("node {u} of {n} nodes")));
        }
        let mut sites = start.to_vec();
        if statistics != Statistics::Distinguishable {
            sites.sort_unstable();
        }
        if statistics == Statistics::Fermions && sites.windows(2).any(|w| w[0] == w[1]) {
            return Err(QsoError::invalid("start", "fermions cannot share a site"));
        }
        let terms = Terms::new(graph, generator, gamma, statistics);
        for (j, g) in group.generators().iter().enumerate() {
            check_permutation(n, g, j)?;
            if !terms.is_symmetry(g) {
                return Err(QsoError::invalid(
                    "group",
                    format!("generator {j} is not an automorphism of the graph"),
                ));
            }
        }
        let log_group: f64 = group.orbit_sizes().iter().map(|&s| (s as f64).ln()).sum();
        let nodes = n as f64;
        let log_configurations = match statistics {
            Statistics::Distinguishable => k as f64 * nodes.ln(),
            Statistics::Bosons => log_binomial(nodes + k as f64 - 1.0, k),
            Statistics::Fermions => log_binomial(nodes, k),
        };
        if log_configurations - log_group > (MAX_IMAGE_ORBITS as f64).ln() + 1.0 {
            return Err(QsoError::invalid(
                "start",
                format!(
                    "at least {:.1e} orbits (limit {MAX_IMAGE_ORBITS})",
                    (log_configurations - log_group).exp()
                ),
            ));
        }
        let images = Arc::new(Images::new(n, group.generators(), log_group)?);
        let mut walk = GroupInteractingWalk {
            terms,
            statistics,
            generator,
            gamma,
            start: sites,
            log_group,
            images,
            orbits: Vec::new(),
            index: KeyMap::default(),
            hopping: Vec::new(),
            same: Vec::new(),
            adjacent: Vec::new(),
            potential: Vec::new(),
            hamiltonian: CsrMatrix::from_triplets(0, &[])?,
            spectral: OnceLock::new(),
        };
        walk.search(graph)?;
        walk.assemble()?;
        Ok(walk)
    }

    /// The walk started from the configuration `start`, reduced by the
    /// automorphisms that fix each of its occupied nodes
    /// ([`Graph::stabilizer`]); the start's orbit is the configuration
    /// alone.
    pub fn from_start(
        graph: &Graph,
        generator: Generator,
        gamma: f64,
        statistics: Statistics,
        start: &[usize],
    ) -> Result<Self> {
        let n = graph.num_nodes();
        if let Some(&u) = start.iter().find(|&&u| u >= n) {
            return Err(QsoError::invalid("start", format!("node {u} of {n} nodes")));
        }
        let mut fixed = start.to_vec();
        fixed.sort_unstable();
        fixed.dedup();
        let group = graph.stabilizer(&fixed)?;
        Self::new(graph, generator, gamma, statistics, start, &group)
    }

    /// The breadth-first search for the orbits and the hopping between
    /// them, a batch of orbits at a time: the hops of a batch's orbits in
    /// parallel, those into orbits found before the batch resolved there,
    /// then the new orbits numbered in order.
    fn search(&mut self, graph: &Graph) -> Result<()> {
        let statistics = self.statistics;
        let images = self.images.clone();
        let Some(form) = images.form(&self.start, statistics)? else {
            return Err(QsoError::invalid(
                "start",
                "no invariant state contains the start (a symmetry maps it to minus itself)",
            ));
        };
        self.index.insert(form.key, 0);
        self.orbits.push(Orbit {
            sites: self.start.clone(),
            odd: form.odd,
            log_stabiliser: form.log_stabiliser,
        });
        let fermions = statistics == Statistics::Fermions;
        let k = self.start.len();
        let max_degree = (0..graph.num_nodes())
            .map(|u| graph.neighbors(u).len())
            .max()
            .unwrap_or(0);
        let per_orbit = (k as u64).saturating_mul(max_degree.max(1) as u64);
        let batch = ((1u64 << 22) / per_orbit).clamp(1, 1 << 16) as usize;
        let neighbours: Vec<Vec<usize>> = (0..graph.num_nodes())
            .map(|u| {
                let mut v: Vec<usize> = graph.neighbors(u).iter().map(|&(v, _)| v).collect();
                v.sort_unstable();
                v
            })
            .collect();
        // The entry of a hop into orbit `rep` from an orbit of `ln |Stab|`
        // `source`: the amplitudes were signed by their forms' parity, so
        // relative to the target's representative; the source's
        // representative is the source.
        let entry = |rep_odd: bool, rep_log: f64, source: f64, signed: f64| {
            let sign = if fermions && rep_odd { -1.0 } else { 1.0 };
            sign * (0.5 * (rep_log - source)).exp() * signed
        };
        let mut lo = 0;
        while lo < self.orbits.len() {
            let hi = self.orbits.len().min(lo + batch);
            let tasks = (crate::parallel::threads() * 8).max(1);
            let per = (hi - lo).div_ceil(tasks).max(1);
            let chunks = (hi - lo).div_ceil(per);
            let orbits = &self.orbits;
            let index = &self.index;
            let terms = &self.terms;
            let neighbours = &neighbours;
            let found = crate::parallel::map_tasks(chunks, |c| -> Result<Vec<_>> {
                let mut out = Vec::new();
                let mut scratch = terms.scratch(k);
                for (offset, orbit) in orbits[lo + c * per..(lo + (c + 1) * per).min(hi)]
                    .iter()
                    .enumerate()
                {
                    let o = lo + c * per + offset;
                    let sites = &orbit.sites;
                    let mut distinct: Vec<(usize, usize)> = Vec::new();
                    for &s in sites {
                        match distinct.iter_mut().find(|e| e.0 == s) {
                            Some(e) => e.1 += 1,
                            None => distinct.push((s, 1)),
                        }
                    }
                    let same: usize = distinct.iter().map(|&(_, m)| m * (m - 1) / 2).sum();
                    let mut adjacent = 0usize;
                    for (x, &(u, mu)) in distinct.iter().enumerate() {
                        for &(v, mv) in &distinct[x + 1..] {
                            if neighbours[u].binary_search(&v).is_ok() {
                                adjacent += mu * mv;
                            }
                        }
                    }
                    let mut potential = 0.0;
                    // Hops into known orbits by target, and into new ones
                    // by key, in order of first appearance.
                    let mut known: Vec<(u32, f64)> = Vec::new();
                    let mut fresh: KeyMap<usize> = KeyMap::default();
                    let mut hops: Vec<Hop> = Vec::new();
                    let mut error = None;
                    terms.apply(sites, &mut scratch, |target, amplitude| {
                        if error.is_some() {
                            return;
                        }
                        if target == sites.as_slice() {
                            potential += amplitude;
                            return;
                        }
                        match images.form(target, statistics) {
                            Ok(Some(form)) => {
                                let signed = if fermions && form.odd {
                                    -amplitude
                                } else {
                                    amplitude
                                };
                                if let Some(&t) = index.get(&form.key) {
                                    match known.iter_mut().find(|e| e.0 == t) {
                                        Some(e) => e.1 += signed,
                                        None => known.push((t, signed)),
                                    }
                                    return;
                                }
                                let h = *fresh.entry(form.key).or_insert_with(|| {
                                    hops.push(Hop {
                                        key: form.key,
                                        sites: target.to_vec(),
                                        odd: form.odd,
                                        log_stabiliser: form.log_stabiliser,
                                        amplitude: 0.0,
                                    });
                                    hops.len() - 1
                                });
                                hops[h].amplitude += signed;
                            }
                            Ok(None) => {}
                            Err(e) => error = Some(e),
                        }
                    });
                    if let Some(e) = error {
                        return Err(e);
                    }
                    let source = orbit.log_stabiliser;
                    let entries: Vec<(usize, usize, f64)> = known
                        .into_iter()
                        .filter_map(|(t, signed)| {
                            let rep = &orbits[t as usize];
                            let v = entry(rep.odd, rep.log_stabiliser, source, signed);
                            (v != 0.0).then_some((t as usize, o, v))
                        })
                        .collect();
                    out.push((same, adjacent, potential, entries, hops));
                }
                Ok(out)
            });
            let mut o = lo;
            for chunk in found {
                for (same, adjacent, potential, entries, hops) in chunk? {
                    self.same.push(same as f64);
                    self.adjacent.push(adjacent as f64);
                    self.potential.push(potential);
                    self.hopping.extend(entries);
                    let source = self.orbits[o].log_stabiliser;
                    for Hop {
                        key,
                        sites: moved,
                        odd,
                        log_stabiliser,
                        amplitude: signed,
                    } in hops
                    {
                        let (target, v) = match self.index.get(&key) {
                            Some(&t) => {
                                let rep = &self.orbits[t as usize];
                                (t, entry(rep.odd, rep.log_stabiliser, source, signed))
                            }
                            None => {
                                let v = entry(odd, log_stabiliser, source, signed);
                                if v == 0.0 {
                                    // Cancelled: not reached through here.
                                    continue;
                                }
                                if self.orbits.len() >= MAX_IMAGE_ORBITS {
                                    return Err(QsoError::invalid(
                                        "start",
                                        format!("more than {MAX_IMAGE_ORBITS} orbits"),
                                    ));
                                }
                                let t = self.orbits.len() as u32;
                                self.orbits.push(Orbit {
                                    sites: moved,
                                    odd,
                                    log_stabiliser,
                                });
                                self.index.insert(key, t);
                                (t, v)
                            }
                        };
                        if v != 0.0 {
                            self.hopping.push((target as usize, o, v));
                        }
                    }
                    if self.hopping.len() + self.orbits.len() > MAX_ORBIT_ENTRIES {
                        return Err(QsoError::invalid(
                            "start",
                            format!("the reduced Hamiltonian exceeds {MAX_ORBIT_ENTRIES} entries"),
                        ));
                    }
                    o += 1;
                }
            }
            lo = hi;
        }
        Ok(())
    }

    /// The reduced Hamiltonian from the hopping and the interaction.
    fn assemble(&mut self) -> Result<()> {
        let d = self.orbits.len();
        let hops = self.hopping.len();
        let (on_site, neighbour) = (self.terms.on_site, self.terms.neighbour);
        for o in 0..d {
            let diag = self.potential[o] + on_site * self.same[o] + neighbour * self.adjacent[o];
            if diag != 0.0 {
                self.hopping.push((o, o, diag));
            }
        }
        // The diagonal rides on the hopping while the matrix is built.
        let built = CsrMatrix::from_triplets(d, &self.hopping);
        self.hopping.truncate(hops);
        self.hamiltonian = built?;
        self.spectral = OnceLock::new();
        Ok(())
    }

    /// The same walk with on-site interaction `on_site` (U) and interaction
    /// `neighbour` (V) between particles on adjacent nodes.
    pub fn with_interaction(mut self, on_site: f64, neighbour: f64) -> Result<Self> {
        if !(on_site.is_finite() && neighbour.is_finite()) {
            return Err(QsoError::invalid(
                "interaction",
                "interaction strengths must be finite",
            ));
        }
        self.terms.on_site = on_site;
        self.terms.neighbour = neighbour;
        self.assemble()?;
        Ok(self)
    }

    pub fn generator(&self) -> Generator {
        self.generator
    }

    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// The on-site interaction `U`.
    pub fn on_site(&self) -> f64 {
        self.terms.on_site
    }

    /// The interaction `V` between particles on adjacent nodes.
    pub fn neighbour(&self) -> f64 {
        self.terms.neighbour
    }

    pub fn statistics(&self) -> Statistics {
        self.statistics
    }

    pub fn particles(&self) -> usize {
        self.start.len()
    }

    pub fn num_nodes(&self) -> usize {
        self.images.n
    }

    /// The start (sorted for bosons and fermions).
    pub fn start(&self) -> &[usize] {
        &self.start
    }

    /// Number of orbits reachable from the start that carry an invariant
    /// state.
    pub fn num_orbits(&self) -> usize {
        self.orbits.len()
    }

    /// `ln` of the number of configurations of the particles.
    pub fn log_num_configurations(&self) -> f64 {
        let nodes = self.num_nodes() as f64;
        let k = self.start.len();
        match self.statistics {
            Statistics::Distinguishable => k as f64 * nodes.ln(),
            Statistics::Bosons => log_binomial(nodes + k as f64 - 1.0, k),
            Statistics::Fermions => log_binomial(nodes, k),
        }
    }

    /// `ln` of the group's order.
    pub fn log_group_order(&self) -> f64 {
        self.log_group
    }

    /// The representative of orbit `o`.
    pub fn representative(&self, o: usize) -> &[usize] {
        &self.orbits[o].sites
    }

    /// The configurations in orbit `o` (from `|G|/|Stab|` in logarithms,
    /// rounded; approximate beyond 2⁵³).
    pub fn orbit_size(&self, o: usize) -> f64 {
        (self.log_group - self.orbits[o].log_stabiliser)
            .exp()
            .round()
    }

    pub fn hamiltonian(&self) -> &CsrMatrix {
        &self.hamiltonian
    }

    fn spectral(&self) -> Option<&(Vec<f64>, RMatrix)> {
        self.spectral
            .get_or_init(|| {
                let d = self.num_orbits();
                (d <= DENSE_LIMIT).then(|| {
                    let mut h = RMatrix::zeros(d, d);
                    for o in 0..d {
                        for (o2, v) in self.hamiltonian.row(o) {
                            h[(o, o2)] = v;
                        }
                    }
                    eigh_real(&h)
                })
            })
            .as_ref()
    }

    /// Eigenvalues of the reduced Hamiltonian (up to [`DENSE_LIMIT`]
    /// orbits).
    pub fn spectrum(&self) -> Option<&[f64]> {
        self.spectral().map(|(v, _)| v.as_slice())
    }

    fn lookup(&self, sites: &[usize]) -> Result<Lookup> {
        if sites.len() != self.start.len() {
            return Err(QsoError::DimensionMismatch {
                expected: self.start.len(),
                actual: sites.len(),
            });
        }
        let n = self.num_nodes();
        if let Some(&u) = sites.iter().find(|&&u| u >= n) {
            return Err(QsoError::invalid("sites", format!("node {u} of {n} nodes")));
        }
        let mut sorted = sites.to_vec();
        if self.statistics != Statistics::Distinguishable {
            sorted.sort_unstable();
        }
        if self.statistics == Statistics::Fermions && sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(QsoError::invalid(
                "sites",
                "two fermions cannot occupy the same site",
            ));
        }
        let Some(form) = self.images.form(&sorted, self.statistics)? else {
            return Ok(Lookup::NoInvariant);
        };
        let Some(&o) = self.index.get(&form.key) else {
            return Ok(Lookup::Unreachable);
        };
        let sign =
            if self.statistics == Statistics::Fermions && form.odd != self.orbits[o as usize].odd {
                -1.0
            } else {
                1.0
            };
        Ok(Lookup::Found(o as usize, sign))
    }

    /// The orbit and the sign of the configuration `sites` relative to
    /// its representative; `None` if no invariant state holds it, an error
    /// if the start cannot reach it.
    pub fn locate(&self, sites: &[usize]) -> Result<Option<(usize, f64)>> {
        match self.lookup(sites)? {
            Lookup::Found(o, sign) => Ok(Some((o, sign))),
            Lookup::NoInvariant => Ok(None),
            Lookup::Unreachable => Err(QsoError::invalid(
                "sites",
                "the configuration is not reachable from the start",
            )),
        }
    }

    /// The orbit state of the configuration `sites` (the start's is the
    /// start).
    pub fn state_of(&self, sites: &[usize]) -> Result<Vec<C64>> {
        let (o, sign) = self.locate(sites)?.ok_or_else(|| {
            QsoError::invalid(
                "sites",
                "no invariant state contains this configuration (a symmetry maps it to minus \
                 itself)",
            )
        })?;
        let mut x = vec![C64::new(0.0, 0.0); self.num_orbits()];
        x[o] = C64::new(sign, 0.0);
        Ok(x)
    }

    fn check_state(&self, x: &[C64]) -> Result<()> {
        if x.len() != self.num_orbits() {
            return Err(QsoError::DimensionMismatch {
                expected: self.num_orbits(),
                actual: x.len(),
            });
        }
        Ok(())
    }

    /// `e^{−iHt} x` on the orbits.
    pub fn evolve(&self, x0: &[C64], t: f64) -> Result<Vec<C64>> {
        self.check_state(x0)?;
        propagate(&self.hamiltonian, self.spectral(), x0, t)
    }

    /// The probability of each orbit.
    pub fn orbit_probabilities(&self, x: &[C64]) -> Result<Vec<f64>> {
        self.check_state(x)?;
        Ok(x.iter().map(|a| a.norm_sqr()).collect())
    }

    /// The probability of the configuration `sites` (its orbit's over its
    /// size; zero for a configuration no invariant state holds or the
    /// start cannot reach).
    pub fn probability_of(&self, x: &[C64], sites: &[usize]) -> Result<f64> {
        self.check_state(x)?;
        Ok(match self.lookup(sites)? {
            Lookup::Found(o, _) => x[o].norm_sqr() / self.orbit_size(o),
            Lookup::NoInvariant | Lookup::Unreachable => 0.0,
        })
    }

    /// The density of particles on every node: each orbit's probability
    /// spread evenly over the orbits of the nodes its representative
    /// occupies.
    pub fn densities(&self, x: &[C64]) -> Result<Vec<f64>> {
        self.check_state(x)?;
        let n = self.num_nodes();
        let group = &self.images.group;
        // Per orbit of nodes (by its root), the density on each of its
        // nodes; then spread over the nodes.
        let mut per_class = vec![0.0; n];
        for (o, orbit) in self.orbits.iter().enumerate() {
            let p = x[o].norm_sqr();
            for &u in &orbit.sites {
                let r = group.root(u as u32);
                per_class[r as usize] += p / group.orbit_size(r) as f64;
            }
        }
        Ok((0..n)
            .map(|v| per_class[group.root(v as u32) as usize])
            .collect())
    }
}

#[cfg(test)]
mod tests;
