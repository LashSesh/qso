//! Interacting walkers on graphs whose symmetries act on coordinates, on
//! the orbits of configurations found from canonical forms, without
//! visiting every configuration or building the graph.
//!
//! The nodes of a Hamming graph `H(d, q)` (the hypercube `Q_d = H(d, 2)`,
//! the complete graph `K_q = H(1, q)`) are words of `d` letters from `q`
//! symbols; its automorphisms permute the places and, in each place, the
//! symbols (`S_q ≀ S_d`). The nodes of the Johnson graph `J(n, k)` and the
//! Kneser graph `K(n, k)` are the `k`-subsets of `n` points, words of `n`
//! bits with `k` ones; the permutations of the points act on them by
//! permuting places alone (`S_n`, all their automorphisms except the
//! complement of `J(2k, k)`).
//!
//! A configuration of particles on the nodes, with the start's distinct
//! nodes above it in a fixed order, is a matrix of letters with a column
//! per place. An element of the group that fixes every start node maps it
//! to the matrix with permuted columns (and, for Hamming graphs, symbols
//! renamed within each column) that keeps the start's rows. So two
//! configurations lie in one orbit of the start's pointwise stabiliser
//! exactly when, for some order of their particles, their matrices have
//! the same canonical form: each column's symbols renamed in order of
//! first appearance from the top (Hamming graphs), the columns sorted. The
//! smallest form over the orders of the particles is the configuration's
//! key. The orders that reach it count the stabiliser
//! `|Stab(c)| = #orders · Π_{equal columns} m! · Π_columns (q − u_j)!`
//! (`u_j` the symbols column `j` uses), so the orbit sizes come with it,
//! and for fermions the parity of the order is the sign of the
//! configuration relative to its orbit (an orbit reached with both
//! parities carries no invariant state).
//!
//! The orbits are found by breadth-first search from the start: the
//! Hamiltonian applied to each orbit's representative gives configurations
//! whose keys either name an orbit already found or a new one. Their number
//! and the entries of the reduced Hamiltonian are bounded; the work is a
//! canonical form per hop, `k!` orders of `k` particles times the places.

use std::collections::HashMap;
use std::sync::OnceLock;

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::symmetric::MAX_ORBIT_ENTRIES;
use super::{Statistics, propagate};
use crate::error::{QsoError, Result};
use crate::graph::{Graph, generators};
use crate::linalg::{CsrMatrix, RMatrix, eigh_real};
use crate::walks::{DENSE_LIMIT, Generator};

/// Most orbits found (each keeps its representative and its key).
pub const MAX_COORDINATE_ORBITS: usize = 1 << 22;

/// Most particles (`6! = 720` orders per canonical form).
pub const MAX_COORDINATE_PARTICLES: usize = 6;

/// Most nodes of a graph (`2⁶²`, node indices in 64 bits).
const MAX_NODES: u128 = 1 << 62;

/// A graph whose automorphisms act on the places of its nodes' words.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CoordinateGraph {
    /// `H(d, q)`: words of `d` letters from `q ≤ 255` symbols, adjacent
    /// when they differ in one place; node `Σ x_j q^j` as
    /// [`generators::hamming`] numbers them (`Q_d` is `H(d, 2)`, `K_q` is
    /// `H(1, q)`).
    Hamming { d: usize, q: usize },
    /// `J(n, k)`: the `k`-subsets of `n ≤ 62` points, adjacent when they
    /// share `k − 1`; nodes in colex order, as [`generators::johnson`].
    Johnson { n: usize, k: usize },
    /// `K(n, k)`: the `k`-subsets of `n ≤ 62` points, adjacent when
    /// disjoint; nodes in colex order, as [`generators::kneser`].
    Kneser { n: usize, k: usize },
}

/// `C(n, k)` for `n ≤ 62`: exact in 64 bits.
fn binomial(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut c: u128 = 1;
    for i in 0..k {
        c = c * (n - i) as u128 / (i + 1) as u128;
    }
    c as u64
}

/// `ln n!`.
fn log_factorial(n: usize) -> f64 {
    (2..=n).map(|i| (i as f64).ln()).sum()
}

impl CoordinateGraph {
    fn check(&self) -> Result<()> {
        match *self {
            CoordinateGraph::Hamming { d, q } => {
                if d == 0 || !(2..=255).contains(&q) {
                    return Err(QsoError::invalid("d/q", "need d ≥ 1 and 2 ≤ q ≤ 255"));
                }
                let nodes = u32::try_from(d)
                    .ok()
                    .and_then(|e| (q as u128).checked_pow(e));
                if nodes.is_none_or(|n| n > MAX_NODES) {
                    return Err(QsoError::invalid(
                        "d/q",
                        "q^d nodes must be at most 2^62".to_string(),
                    ));
                }
            }
            CoordinateGraph::Johnson { n, k } => {
                if !(1..=62).contains(&n) || k == 0 || k >= n {
                    return Err(QsoError::invalid("n/k", "need 0 < k < n ≤ 62"));
                }
            }
            CoordinateGraph::Kneser { n, k } => {
                if !(1..=62).contains(&n) || k == 0 || 2 * k > n {
                    return Err(QsoError::invalid("n/k", "need 0 < 2k ≤ n ≤ 62"));
                }
            }
        }
        Ok(())
    }

    /// Places of a word.
    fn places(&self) -> usize {
        match *self {
            CoordinateGraph::Hamming { d, .. } => d,
            CoordinateGraph::Johnson { n, .. } | CoordinateGraph::Kneser { n, .. } => n,
        }
    }

    /// Symbols, and whether the group renames them within a place.
    fn symbols(&self) -> (usize, bool) {
        match *self {
            CoordinateGraph::Hamming { q, .. } => (q, true),
            _ => (2, false),
        }
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> u64 {
        match *self {
            CoordinateGraph::Hamming { d, q } => (q as u64).pow(d as u32),
            CoordinateGraph::Johnson { n, k } | CoordinateGraph::Kneser { n, k } => {
                binomial(n as u64, k as u64)
            }
        }
    }

    /// Neighbours of every node.
    pub fn degree(&self) -> u64 {
        match *self {
            CoordinateGraph::Hamming { d, q } => (d * (q - 1)) as u64,
            CoordinateGraph::Johnson { n, k } => (k * (n - k)) as u64,
            CoordinateGraph::Kneser { n, k } => binomial((n - k) as u64, k as u64),
        }
    }

    /// The graph itself (for comparisons; it may be far too large).
    pub fn graph(&self) -> Result<Graph> {
        self.check()?;
        match *self {
            CoordinateGraph::Hamming { d, q } => generators::hamming(d, q),
            CoordinateGraph::Johnson { n, k } => generators::johnson(n, k),
            CoordinateGraph::Kneser { n, k } => generators::kneser(n, k),
        }
    }

    /// The word of `node`: its letters (Hamming) or the bits of its subset.
    pub fn word(&self, node: u64) -> Result<Vec<usize>> {
        self.check()?;
        if node >= self.num_nodes() {
            return Err(QsoError::invalid(
                "node",
                format!("node {node} of {} nodes", self.num_nodes()),
            ));
        }
        let mut out = vec![0u8; self.places()];
        self.letters(node, &mut out);
        Ok(out.into_iter().map(usize::from).collect())
    }

    /// The node of a word (letters, or the bits of a subset).
    pub fn node(&self, word: &[usize]) -> Result<u64> {
        self.check()?;
        if word.len() != self.places() {
            return Err(QsoError::DimensionMismatch {
                expected: self.places(),
                actual: word.len(),
            });
        }
        let (q, _) = self.symbols();
        if word.iter().any(|&x| x >= q) {
            return Err(QsoError::invalid("word", "a letter is out of range"));
        }
        match *self {
            CoordinateGraph::Hamming { q, .. } => Ok(word
                .iter()
                .rev()
                .fold(0u64, |acc, &x| acc * q as u64 + x as u64)),
            CoordinateGraph::Johnson { k, .. } | CoordinateGraph::Kneser { k, .. } => {
                let mask = word
                    .iter()
                    .enumerate()
                    .fold(0u64, |m, (i, &x)| m | (x as u64) << i);
                if mask.count_ones() as usize != k {
                    return Err(QsoError::invalid("word", format!("a subset of {k} points")));
                }
                Ok(colex_rank(mask))
            }
        }
    }

    /// The letters of `node` into `out`.
    fn letters(&self, node: u64, out: &mut [u8]) {
        match *self {
            CoordinateGraph::Hamming { q, .. } => {
                let mut x = node;
                for o in out.iter_mut() {
                    *o = (x % q as u64) as u8;
                    x /= q as u64;
                }
            }
            CoordinateGraph::Johnson { k, .. } | CoordinateGraph::Kneser { k, .. } => {
                let mask = colex_unrank(node, k);
                for (i, o) in out.iter_mut().enumerate() {
                    *o = (mask >> i & 1) as u8;
                }
            }
        }
    }

    /// `f(v)` for every neighbour `v` of `node`.
    fn neighbours(&self, node: u64, mut f: impl FnMut(u64)) {
        match *self {
            CoordinateGraph::Hamming { d, q } => {
                let mut place = 1u64;
                let mut x = node;
                for _ in 0..d {
                    let digit = x % q as u64;
                    for s in 0..q as u64 {
                        if s != digit {
                            f(node - digit * place + s * place);
                        }
                    }
                    x /= q as u64;
                    place *= q as u64;
                }
            }
            CoordinateGraph::Johnson { n, k } => {
                let mask = colex_unrank(node, k);
                for out in (0..n).filter(|&i| mask >> i & 1 == 1) {
                    for inn in (0..n).filter(|&i| mask >> i & 1 == 0) {
                        f(colex_rank(mask & !(1 << out) | 1 << inn));
                    }
                }
            }
            CoordinateGraph::Kneser { n, k } => {
                let mask = colex_unrank(node, k);
                let free: Vec<usize> = (0..n).filter(|&i| mask >> i & 1 == 0).collect();
                // The k-subsets of the complement, by Gosper's hack.
                let m = free.len();
                let mut x: u64 = (1u64 << k) - 1;
                while x < 1u64 << m {
                    let sub = (0..m)
                        .filter(|&j| x >> j & 1 == 1)
                        .fold(0u64, |acc, j| acc | 1 << free[j]);
                    f(colex_rank(sub));
                    let c = x & x.wrapping_neg();
                    let r = x + c;
                    x = (((r ^ x) >> 2) / c) | r;
                }
            }
        }
    }

    /// Whether nodes `a ≠ b` are adjacent.
    fn adjacent(&self, a: u64, b: u64) -> bool {
        match *self {
            CoordinateGraph::Hamming { d, q } => {
                let (mut x, mut y, mut differ) = (a, b, 0);
                for _ in 0..d {
                    if x % q as u64 != y % q as u64 {
                        differ += 1;
                    }
                    x /= q as u64;
                    y /= q as u64;
                }
                differ == 1
            }
            CoordinateGraph::Johnson { k, .. } => {
                (colex_unrank(a, k) & colex_unrank(b, k)).count_ones() as usize == k - 1
            }
            CoordinateGraph::Kneser { k, .. } => colex_unrank(a, k) & colex_unrank(b, k) == 0,
        }
    }
}

/// The colex rank of a subset mask: `Σ_i C(c_i, i + 1)`.
fn colex_rank(mut mask: u64) -> u64 {
    let (mut rank, mut i) = (0u64, 0u64);
    while mask != 0 {
        let c = mask.trailing_zeros() as u64;
        rank += binomial(c, i + 1);
        i += 1;
        mask &= mask - 1;
    }
    rank
}

/// The `k`-subset of colex rank `rank`.
fn colex_unrank(mut rank: u64, k: usize) -> u64 {
    let mut mask = 0u64;
    for i in (1..=k as u64).rev() {
        // The largest c with C(c, i) ≤ rank.
        let mut c = i - 1;
        while binomial(c + 1, i) <= rank {
            c += 1;
        }
        rank -= binomial(c, i);
        mask |= 1 << c;
    }
    mask
}

/// Canonical forms of configurations under the pointwise stabiliser of
/// the start's nodes.
struct Canon {
    graph: CoordinateGraph,
    places: usize,
    q: usize,
    relabel: bool,
    /// The letters of the start's distinct nodes, row by row.
    fixed: Vec<u8>,
    rows_fixed: usize,
}

/// A configuration's canonical form.
struct Form {
    /// The sorted column codes, run-length encoded (`code << 8 | run`).
    key: Vec<u64>,
    /// Parity of the order of the particles that reaches it.
    odd: bool,
    /// `ln |Stab(c)|`.
    log_stabiliser: f64,
}

impl Canon {
    fn new(graph: CoordinateGraph, fixed_nodes: &[u64]) -> Self {
        let places = graph.places();
        let (q, relabel) = graph.symbols();
        let mut fixed = vec![0u8; places * fixed_nodes.len()];
        for (r, &u) in fixed_nodes.iter().enumerate() {
            graph.letters(u, &mut fixed[r * places..(r + 1) * places]);
        }
        Canon {
            graph,
            places,
            q,
            relabel,
            fixed,
            rows_fixed: fixed_nodes.len(),
        }
    }

    /// The key of the rows in the order `order` (indices into `rows`,
    /// each `places` letters): every column as a code of four bits per
    /// row, the top row highest, its symbols renamed in order of first
    /// appearance (Hamming graphs); the codes sorted and run-length
    /// encoded. At most twelve rows (six fixed, six moving), so a code
    /// takes 48 bits and a run 8.
    fn key(&self, rows: &[u8], order: &[usize], codes: &mut Vec<u64>, key: &mut Vec<u64>) {
        codes.clear();
        let mut label = [u8::MAX; 256];
        let mut used = [0u8; 12];
        for j in 0..self.places {
            let mut code = 0u64;
            let mut next = 0u8;
            let fixed = (0..self.rows_fixed).map(|i| self.fixed[i * self.places + j]);
            let moving = order.iter().map(|&i| rows[i * self.places + j]);
            for x in fixed.chain(moving) {
                let y = if self.relabel {
                    if label[x as usize] == u8::MAX {
                        label[x as usize] = next;
                        used[next as usize] = x;
                        next += 1;
                    }
                    label[x as usize]
                } else {
                    x
                };
                code = code << 4 | y as u64;
            }
            for &x in &used[..next as usize] {
                label[x as usize] = u8::MAX;
            }
            codes.push(code);
        }
        codes.sort_unstable();
        key.clear();
        let mut k = 0;
        while k < codes.len() {
            let mut run = 1;
            while k + run < codes.len() && codes[k + run] == codes[k] {
                run += 1;
            }
            key.push(codes[k] << 8 | run as u64);
            k += run;
        }
    }

    /// The canonical form of the configuration `sites` (sorted for
    /// bosons and fermions), or `None` when its orbit carries no invariant
    /// state (fermions reached with both parities).
    fn form(&self, sites: &[u64], statistics: Statistics) -> Option<Form> {
        let k = sites.len();
        let mut rows = vec![0u8; k * self.places];
        for (i, &s) in sites.iter().enumerate() {
            self.graph
                .letters(s, &mut rows[i * self.places..(i + 1) * self.places]);
        }
        let mut order: Vec<usize> = (0..k).collect();
        let (mut codes, mut key) = (Vec::with_capacity(self.places), Vec::new());
        let mut best: Vec<u64> = Vec::new();
        let (mut odd, mut count, mut conflict) = (false, 0usize, false);
        let mut consider = |order: &[usize], parity: bool| {
            // Orders that give the same sequence of sites are one order:
            // equal sites (adjacent, as `sites` is sorted) only in their
            // own order.
            for a in 0..k {
                for b in a + 1..k {
                    if order[a] > order[b] && sites[order[a]] == sites[order[b]] {
                        return;
                    }
                }
            }
            self.key(&rows, order, &mut codes, &mut key);
            if count == 0 || key < best {
                std::mem::swap(&mut best, &mut key);
                odd = parity;
                count = 1;
                conflict = false;
            } else if key == best {
                count += 1;
                if parity != odd {
                    conflict = true;
                }
            }
        };
        if statistics == Statistics::Distinguishable {
            consider(&order, false);
        } else {
            // Heap's algorithm: every order, one swap apart.
            let mut c = vec![0usize; k];
            let mut parity = false;
            consider(&order, parity);
            let mut i = 1;
            while i < k {
                if c[i] < i {
                    if i % 2 == 0 {
                        order.swap(0, i);
                    } else {
                        order.swap(c[i], i);
                    }
                    parity = !parity;
                    consider(&order, parity);
                    c[i] += 1;
                    i = 1;
                } else {
                    c[i] = 0;
                    i += 1;
                }
            }
        }
        if statistics == Statistics::Fermions && conflict {
            return None;
        }
        let log_stabiliser = (count as f64).ln() + self.automorphisms(&best);
        Some(Form {
            key: best,
            odd,
            log_stabiliser,
        })
    }

    /// `ln` of the group elements that keep every row of the matrix of
    /// `key`: `m!` for each run of `m` equal columns, and `(q − u)!` for
    /// each column using `u` symbols (Hamming graphs).
    fn automorphisms(&self, key: &[u64]) -> f64 {
        let mut total = 0.0;
        for &entry in key {
            let run = (entry & 0xff) as usize;
            total += log_factorial(run);
            if self.relabel {
                // Labels are given in order of first appearance: the
                // largest nibble plus one is the number of symbols used.
                let mut code = entry >> 8;
                let mut largest = 0u64;
                while code != 0 {
                    largest = largest.max(code & 0xf);
                    code >>= 4;
                }
                total += run as f64 * log_factorial(self.q - (largest as usize + 1));
            }
        }
        total
    }
}

/// An orbit found: its representative (sorted for bosons and fermions),
/// the parity of its canonical order and `ln |Stab|`.
#[derive(Clone, Debug)]
struct Orbit {
    sites: Vec<u64>,
    odd: bool,
    log_stabiliser: f64,
}

/// Interacting walkers on a [`CoordinateGraph`], on the orbits of the
/// pointwise stabiliser of the start's nodes (see the module
/// documentation): the same Hamiltonian as
/// [`InteractingWalk`](super::InteractingWalk), reduced exactly as
/// [`SymmetricInteractingWalk`](super::SymmetricInteractingWalk) reduces
/// it, with orbits found from canonical forms.
#[derive(Clone, Debug)]
pub struct CoordinateInteractingWalk {
    graph: CoordinateGraph,
    generator: Generator,
    gamma: f64,
    statistics: Statistics,
    on_site: f64,
    neighbour: f64,
    start: Vec<u64>,
    /// The start's distinct nodes, ascending (the rows kept fixed).
    fixed: Vec<u64>,
    orbits: Vec<Orbit>,
    index: HashMap<Vec<u64>, u32>,
    /// `ln` of the order of the stabiliser of the start's nodes.
    log_group: f64,
    /// Hopping between orbits, and per orbit the pairs on one site and on
    /// adjacent sites and the potential, from the representative.
    hopping: Vec<(usize, usize, f64)>,
    same: Vec<f64>,
    adjacent: Vec<f64>,
    potential: Vec<f64>,
    hamiltonian: CsrMatrix,
    spectral: OnceLock<Option<(Vec<f64>, RMatrix)>>,
}

impl CoordinateInteractingWalk {
    /// `start.len()` particles of the given statistics started on the
    /// nodes `start` of `graph`, hopping at rate `gamma` (`h = −γA` or
    /// `γL`), without interaction, reduced by the automorphisms of the
    /// column group that fix every start node.
    pub fn from_start(
        graph: CoordinateGraph,
        generator: Generator,
        gamma: f64,
        statistics: Statistics,
        start: &[u64],
    ) -> Result<Self> {
        graph.check()?;
        if !(gamma.is_finite() && gamma > 0.0) {
            return Err(QsoError::invalid(
                "gamma",
                "hopping rate must be positive and finite",
            ));
        }
        let k = start.len();
        if k == 0 || k > MAX_COORDINATE_PARTICLES {
            return Err(QsoError::invalid(
                "start",
                format!("need 1 to {MAX_COORDINATE_PARTICLES} particles"),
            ));
        }
        let n = graph.num_nodes();
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
        let mut fixed = sites.clone();
        fixed.sort_unstable();
        fixed.dedup();
        let canon = Canon::new(graph, &fixed);
        // The stabiliser's order: the automorphisms of the start's rows.
        let start_key = {
            let none = Canon {
                rows_fixed: 0,
                fixed: Vec::new(),
                ..Canon::new(graph, &[])
            };
            let mut rows = vec![0u8; fixed.len() * canon.places];
            for (i, &u) in fixed.iter().enumerate() {
                graph.letters(u, &mut rows[i * canon.places..(i + 1) * canon.places]);
            }
            let order: Vec<usize> = (0..fixed.len()).collect();
            let (mut codes, mut key) = (Vec::new(), Vec::new());
            none.key(&rows, &order, &mut codes, &mut key);
            none.automorphisms(&key)
        };
        // The configurations over the group's order bound the orbits from
        // below.
        let nodes = n as f64;
        let log_configurations = match statistics {
            Statistics::Distinguishable => k as f64 * nodes.ln(),
            Statistics::Bosons => log_binomial(nodes + k as f64 - 1.0, k),
            Statistics::Fermions => log_binomial(nodes, k),
        };
        if log_configurations - start_key > (MAX_COORDINATE_ORBITS as f64).ln() {
            return Err(QsoError::invalid(
                "start",
                format!(
                    "at least {:.1e} orbits (limit {MAX_COORDINATE_ORBITS})",
                    (log_configurations - start_key).exp()
                ),
            ));
        }
        let mut walk = CoordinateInteractingWalk {
            graph,
            generator,
            gamma,
            statistics,
            on_site: 0.0,
            neighbour: 0.0,
            start: sites,
            fixed,
            orbits: Vec::new(),
            index: HashMap::new(),
            log_group: start_key,
            hopping: Vec::new(),
            same: Vec::new(),
            adjacent: Vec::new(),
            potential: Vec::new(),
            hamiltonian: CsrMatrix::from_triplets(0, &[])?,
            spectral: OnceLock::new(),
        };
        walk.search(&canon)?;
        walk.assemble()?;
        Ok(walk)
    }

    /// The breadth-first search for the orbits and the hopping between
    /// them, a level at a time: the canonical forms of every hop from the
    /// level's orbits in parallel, then the new orbits numbered in order.
    fn search(&mut self, canon: &Canon) -> Result<()> {
        let statistics = self.statistics;
        let Some(form) = canon.form(&self.start, statistics) else {
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
        let hop = -self.gamma;
        let diagonal = match self.generator {
            Generator::Adjacency => 0.0,
            Generator::Laplacian => self.gamma * self.graph.degree() as f64,
        };
        let graph = self.graph;
        let mut lo = 0;
        while lo < self.orbits.len() {
            let hi = self.orbits.len();
            let tasks = (crate::parallel::threads() * 8).max(1);
            let per = (hi - lo).div_ceil(tasks).max(1);
            let chunks = (hi - lo).div_ceil(per);
            let orbits = &self.orbits;
            // Per orbit: the pairs on one site and on adjacent sites, and
            // every hop's target with its form and amplitude.
            let found = crate::parallel::map_tasks(chunks, |c| {
                let mut out = Vec::new();
                for orbit in &orbits[lo + c * per..(lo + (c + 1) * per).min(hi)] {
                    let sites = &orbit.sites;
                    let mut distinct: Vec<(u64, usize)> = Vec::new();
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
                            if graph.adjacent(u, v) {
                                adjacent += mu * mv;
                            }
                        }
                    }
                    let occupation = |u: u64| sites.iter().filter(|&&s| s == u).count();
                    let mut hops: Vec<(Vec<u64>, Form, f64)> = Vec::new();
                    for (k, &j) in sites.iter().enumerate() {
                        if statistics == Statistics::Bosons && k > 0 && sites[k - 1] == j {
                            continue;
                        }
                        let oj = occupation(j);
                        graph.neighbours(j, |i| {
                            let amplitude = match statistics {
                                Statistics::Distinguishable => hop,
                                Statistics::Bosons => {
                                    hop * ((oj * (occupation(i) + 1)) as f64).sqrt()
                                }
                                Statistics::Fermions => {
                                    if occupation(i) > 0 {
                                        return;
                                    }
                                    let (a, b) = (i.min(j), i.max(j));
                                    let between = sites.iter().filter(|&&s| a < s && s < b).count();
                                    if between % 2 == 0 { hop } else { -hop }
                                }
                            };
                            let mut moved = sites.clone();
                            moved[k] = i;
                            if statistics != Statistics::Distinguishable {
                                moved.sort_unstable();
                            }
                            if let Some(form) = canon.form(&moved, statistics) {
                                hops.push((moved, form, amplitude));
                            }
                        });
                    }
                    out.push((same, adjacent, hops));
                }
                out
            });
            let mut o = lo;
            for chunk in found {
                for (same, adjacent, hops) in chunk {
                    self.same.push(same as f64);
                    self.adjacent.push(adjacent as f64);
                    self.potential.push(diagonal * self.start.len() as f64);
                    let size_log = self.orbits[o].log_stabiliser;
                    let mut row: HashMap<u32, f64> = HashMap::new();
                    for (moved, form, amplitude) in hops {
                        let target = match self.index.get(&form.key) {
                            Some(&t) => t,
                            None => {
                                if self.orbits.len() >= MAX_COORDINATE_ORBITS {
                                    return Err(QsoError::invalid(
                                        "start",
                                        format!("more than {MAX_COORDINATE_ORBITS} orbits"),
                                    ));
                                }
                                let t = self.orbits.len() as u32;
                                self.orbits.push(Orbit {
                                    sites: moved,
                                    odd: form.odd,
                                    log_stabiliser: form.log_stabiliser,
                                });
                                self.index.insert(form.key, t);
                                t
                            }
                        };
                        let rep = &self.orbits[target as usize];
                        let sign = if statistics == Statistics::Fermions && form.odd != rep.odd {
                            -1.0
                        } else {
                            1.0
                        };
                        // √(|O|/|O′|) = √(|Stab(c′)|/|Stab(c)|).
                        let ratio = (0.5 * (rep.log_stabiliser - size_log)).exp();
                        *row.entry(target).or_default() += sign * ratio * amplitude;
                    }
                    for (t, v) in row {
                        if v != 0.0 {
                            self.hopping.push((t as usize, o, v));
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
        let mut triplets = self.hopping.clone();
        for o in 0..d {
            let diag =
                self.potential[o] + self.on_site * self.same[o] + self.neighbour * self.adjacent[o];
            if diag != 0.0 {
                triplets.push((o, o, diag));
            }
        }
        self.hamiltonian = CsrMatrix::from_triplets(d, &triplets)?;
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
        self.on_site = on_site;
        self.neighbour = neighbour;
        self.assemble()?;
        Ok(self)
    }

    pub fn graph(&self) -> CoordinateGraph {
        self.graph
    }

    pub fn generator(&self) -> Generator {
        self.generator
    }

    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// The on-site interaction `U`.
    pub fn on_site(&self) -> f64 {
        self.on_site
    }

    /// The interaction `V` between particles on adjacent nodes.
    pub fn neighbour(&self) -> f64 {
        self.neighbour
    }

    pub fn statistics(&self) -> Statistics {
        self.statistics
    }

    pub fn particles(&self) -> usize {
        self.start.len()
    }

    /// The start (sorted for bosons and fermions).
    pub fn start(&self) -> &[u64] {
        &self.start
    }

    /// Number of orbits that carry an invariant state.
    pub fn num_orbits(&self) -> usize {
        self.orbits.len()
    }

    /// `ln` of the number of configurations of the particles.
    pub fn log_num_configurations(&self) -> f64 {
        let nodes = self.graph.num_nodes() as f64;
        let k = self.start.len();
        match self.statistics {
            Statistics::Distinguishable => k as f64 * nodes.ln(),
            Statistics::Bosons => log_binomial(nodes + k as f64 - 1.0, k),
            Statistics::Fermions => log_binomial(nodes, k),
        }
    }

    /// `ln` of the order of the group: the stabiliser of the start's nodes
    /// in the column group.
    pub fn log_group_order(&self) -> f64 {
        self.log_group
    }

    /// The representative of orbit `o`.
    pub fn representative(&self, o: usize) -> &[u64] {
        &self.orbits[o].sites
    }

    /// The configurations in orbit `o`.
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

    /// The orbit and the sign of the configuration `sites` relative to
    /// its representative; `None` if no invariant state holds it, an error
    /// if it lies in no orbit found (not reachable from the start).
    pub fn locate(&self, sites: &[u64]) -> Result<Option<(usize, f64)>> {
        if sites.len() != self.start.len() {
            return Err(QsoError::DimensionMismatch {
                expected: self.start.len(),
                actual: sites.len(),
            });
        }
        let n = self.graph.num_nodes();
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
        let canon = Canon::new(self.graph, &self.fixed);
        let Some(form) = canon.form(&sorted, self.statistics) else {
            return Ok(None);
        };
        let Some(&o) = self.index.get(&form.key) else {
            return Err(QsoError::invalid(
                "sites",
                "the configuration is not reachable from the start",
            ));
        };
        let sign =
            if self.statistics == Statistics::Fermions && form.odd != self.orbits[o as usize].odd {
                -1.0
            } else {
                1.0
            };
        Ok(Some((o as usize, sign)))
    }

    /// The orbit state of the configuration `sites` (the start's alone is
    /// the start).
    pub fn state_of(&self, sites: &[u64]) -> Result<Vec<C64>> {
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
    /// size).
    pub fn probability_of(&self, x: &[C64], sites: &[u64]) -> Result<f64> {
        self.check_state(x)?;
        Ok(match self.locate(sites)? {
            Some((o, _)) => x[o].norm_sqr() / self.orbit_size(o),
            None => 0.0,
        })
    }

    /// The density of particles on the nodes, per orbit of nodes: its
    /// smallest node found among the representatives, its size and the
    /// density on each of its nodes (equal by symmetry).
    pub fn node_densities(&self, x: &[C64]) -> Result<Vec<(u64, f64, f64)>> {
        self.check_state(x)?;
        let canon = Canon::new(self.graph, &self.fixed);
        let mut of_node: HashMap<u64, usize> = HashMap::new();
        let mut keys: HashMap<Vec<u64>, usize> = HashMap::new();
        let mut out: Vec<(u64, f64, f64)> = Vec::new();
        for (o, orbit) in self.orbits.iter().enumerate() {
            let p = x[o].norm_sqr();
            for &u in &orbit.sites {
                let id = match of_node.get(&u) {
                    Some(&id) => id,
                    None => {
                        let form = canon
                            .form(&[u], Statistics::Distinguishable)
                            .expect("a single node has a form");
                        let id = match keys.get(&form.key) {
                            Some(&id) => {
                                out[id].0 = out[id].0.min(u);
                                id
                            }
                            None => {
                                let size = (self.log_group - form.log_stabiliser).exp().round();
                                out.push((u, size, 0.0));
                                keys.insert(form.key, out.len() - 1);
                                out.len() - 1
                            }
                        };
                        of_node.insert(u, id);
                        id
                    }
                };
                out[id].2 += p;
            }
        }
        for e in &mut out {
            e.2 /= e.1;
        }
        out.sort_by_key(|e| e.0);
        Ok(out)
    }
}

/// `ln C(n, k)` for real `n ≥ k`.
fn log_binomial(n: f64, k: usize) -> f64 {
    (0..k).map(|i| ((n - i as f64) / (i + 1) as f64).ln()).sum()
}

#[cfg(test)]
mod tests;
