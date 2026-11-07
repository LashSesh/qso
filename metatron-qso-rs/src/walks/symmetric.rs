//! Open walks reduced by symmetry: density matrices on orbitals.
//!
//! A permutation `g` of the nodes acts on operators as `ρ ↦ P_g ρ P_g†`.
//! If the Hamiltonian, the dissipation and the initial state are invariant
//! under a group `G` of such permutations, so is the state at every time,
//! and an invariant operator is constant on each orbit of `G` on ordered
//! pairs of nodes (an *orbital*): `ρ = Σ_O x_O B_O`, with `B_O` the 0/1
//! matrix of orbital `O`. The evolution is a linear map on the
//! coefficients `x`, as many as there are orbitals: for the stabiliser of a
//! node of the hypercube `Q_d`, `C(d + 3, 3)` (455 for `Q₁₂`) instead of
//! `4^d` entries. Row `P` of the map is read off at one representative
//! pair `(a, b)` of `P`, because the image of an invariant operator is
//! invariant again.
//!
//! * [`Orbitals`] — the orbits of a group, given by generators, on pairs.
//! * [`SymmetricOpenWalk`] — continuous-time walks with dephasing: `dρ/dt =
//!   −i[H, ρ] + Σ_m κ_m (Σ_j P_{m,j} ρ P_{m,j} − ρ)`, the projectors
//!   `P_{m,j}` onto the blocks of partitions of the nodes ([`Dephasing`]):
//!   singletons give the site dephasing of [`super::DephasingWalk`], the
//!   coordinates of a hypercube dephasing per coordinate. Entry `ρ_uv`
//!   decays at `Σ_m κ_m [u, v in different blocks of m]`. Site energies,
//!   traps, losses and per-node dephasing that are constant on every orbit
//!   of points add to the diagonal of the reduced generator, and
//!   [`SymmetricOpenWalk::transport`] gives the efficiency and mean
//!   arrival time at the traps by linear solves on the orbitals (see
//!   [`super::transport`]).
//! * [`SymmetricCoinedWalk`] — the coined walks with decoherence of
//!   [`super::DecoherentCoinedWalk`], on orbitals of pairs of arcs.
//! * [`HypercubeOpenWalk`] — the hypercube with site and coordinate
//!   dephasing on the `C(d + 3, 3)` orbitals of the coordinate
//!   permutations, written down without building the graph (any `d` up to
//!   230), with site terms by Hamming weight and transport up to `d = 61`.
//! * [`ProductOpenWalk`] — walks on Cartesian products whose factors
//!   dephase independently: from a product state the state stays a
//!   product, so every factor evolves on its own, at any number of factors.
//!
//! Up to `√`[`MAX_ORBITAL_PAIRS`] (8192) points, orbitals are found by
//! union–find on all `n²` pairs. Beyond, they come from the orbits of the
//! stabilisers of one point of each orbit on points (`suborbits.rs`, 1.48),
//! in memory of the points times the orbits on points (at most
//! [`MAX_SUBORBIT_LABELS`]) and 16 bytes per orbital. For large orbits the
//! stabilisers' orbits are found from random elements of the group (seeded,
//! so reproducible); merges that are missed split an orbital into several
//! classes, which keeps invariant states exact and only costs
//! coefficients. The work grows with the number of
//! generators: give a small generating set (two permutations generate the
//! coordinate permutations of a hypercube, [`coordinate_permutations`])
//! rather than every automorphism. Coined walks have at most
//! [`super::decoherent::MAX_DECOHERENT_ARCS`] (2048) arcs, as their full
//! counterpart. The reduced generator is limited to [`MAX_ORBITAL_ENTRIES`]
//! entries; the reduced state is what stays small.
//!
//! ```
//! use metatron_qso::graph::generators;
//! use metatron_qso::walks::Generator;
//! use metatron_qso::walks::symmetric::{Dephasing, SymmetricOpenWalk};
//!
//! // Q₈ with site dephasing, from node 0: its stabiliser permutes the
//! // coordinates, 165 orbitals instead of 65 536 entries.
//! let g = generators::hypercube(8);
//! let symmetries = g.stabilizer(&[0]).unwrap().generators().to_vec();
//! let walk = SymmetricOpenWalk::new(
//!     &g, Generator::Laplacian, 1.0, &[Dephasing::sites(256, 0.5)], &symmetries,
//! ).unwrap();
//! assert_eq!(walk.num_orbitals(), 165);
//! let p = walk.populations(&walk.evolve(&walk.localized(0).unwrap(), 2.0).unwrap()).unwrap();
//! assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
//! ```

use std::sync::Arc;

use num_complex::Complex64 as C64;

use super::transport::{
    DephasingOptimum, TransportOptions, TransportResult, TransportState, check_energies,
    check_rates, check_traps,
};
use super::{DecoherentCoinedWalk, DephasingWalk, Generator};
use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::graph::automorphism::is_automorphism;
use crate::linalg::CMatrix;
use crate::parallel::for_each_mut;

/// Most ordered pairs (`n²`) whose orbitals are computed: 2²⁶, a 4-byte
/// word each, and 16 bytes per orbital (256 MiB and at most 1 GiB more).
pub const MAX_ORBITAL_PAIRS: usize = 1 << 26;

/// Most union–find steps (generators times pairs) while orbitals are found.
pub const MAX_ORBITAL_STEPS: u64 = 1 << 34;

/// Most entries of a reduced generator (20 bytes each, 1.25 GiB).
pub const MAX_ORBITAL_ENTRIES: usize = 1 << 26;

/// Most operations (estimated) of one evolution on orbitals.
pub const MAX_ORBITAL_WORK: f64 = 2e11;

/// Marks an orbital index while pair labels are rewritten in place.
const LABEL: u32 = 1 << 31;

/// The orbits of a permutation group on ordered pairs of points.
#[derive(Clone, Debug)]
pub struct Orbitals {
    n: usize,
    labels: Labels,
    /// The first (smallest) pair of every orbital.
    representatives: Vec<(u32, u32)>,
    sizes: Vec<u64>,
}

/// Where the orbital of a pair is read.
#[derive(Clone, Debug)]
enum Labels {
    /// Orbital of pair `(u, v)` at `u·n + v`.
    Pairs(Vec<u32>),
    /// From the orbits of point stabilisers.
    Suborbits(Suborbits),
}

pub(crate) fn check_permutation(n: usize, p: &[usize], k: usize) -> Result<()> {
    if p.len() != n {
        return Err(QsoError::invalid(
            "symmetries",
            format!("permutation {k} has {} entries for {n} points", p.len()),
        ));
    }
    let mut seen = vec![false; n];
    for &x in p {
        if x >= n || std::mem::replace(&mut seen[x], true) {
            return Err(QsoError::invalid(
                "symmetries",
                format!("entry {k} is not a permutation of 0..{n}"),
            ));
        }
    }
    Ok(())
}

impl Orbitals {
    /// The orbitals of the group generated by `generators` (permutations
    /// `u ↦ g[u]` of `0..n`; none for the trivial group, whose orbitals are
    /// the single pairs): by union–find on all pairs up to 8192 points, from
    /// the orbits of point stabilisers beyond (see the module
    /// documentation).
    pub fn new(n: usize, generators: &[Vec<usize>]) -> Result<Self> {
        if n == 0 {
            return Err(QsoError::invalid("n", "need at least one point"));
        }
        for (k, g) in generators.iter().enumerate() {
            check_permutation(n, g, k)?;
        }
        match n.checked_mul(n).filter(|&p| p <= MAX_ORBITAL_PAIRS) {
            Some(pairs) => Self::by_pairs(n, generators, pairs),
            None => Self::by_stabilisers(n, generators),
        }
    }

    /// From the orbits of point stabilisers, at any size within the limits.
    fn by_stabilisers(n: usize, generators: &[Vec<usize>]) -> Result<Self> {
        let (suborbits, representatives, sizes) = Suborbits::new(n, generators, 0)?;
        Ok(Orbitals {
            n,
            labels: Labels::Suborbits(suborbits),
            representatives,
            sizes,
        })
    }

    /// By union–find on all `pairs = n²` pairs.
    fn by_pairs(n: usize, generators: &[Vec<usize>], pairs: usize) -> Result<Self> {
        if (generators.len() as u64).saturating_mul(pairs as u64) > MAX_ORBITAL_STEPS {
            return Err(QsoError::invalid(
                "generators",
                format!(
                    "{} generators on {pairs} pairs exceed {MAX_ORBITAL_STEPS} steps: give a \
                     smaller generating set",
                    generators.len()
                ),
            ));
        }
        // Union–find in which every root is the smallest pair of its set, so
        // that every pointer goes down: parent[p] ≤ p.
        let mut parent: Vec<u32> = (0..pairs as u32).collect();
        fn find(parent: &mut [u32], mut x: u32) -> u32 {
            while parent[x as usize] != x {
                let up = parent[parent[x as usize] as usize];
                parent[x as usize] = up;
                x = up;
            }
            x
        }
        for g in generators {
            for u in 0..n {
                for v in 0..n {
                    let (a, b) = ((u * n + v) as u32, (g[u] * n + g[v]) as u32);
                    let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                    if ra != rb {
                        parent[ra.max(rb) as usize] = ra.min(rb);
                    }
                }
            }
        }
        // Rewrite pointers to orbital indices in increasing order: a root
        // takes the next index, every other pair that of its parent, which
        // is smaller and so already rewritten.
        let mut representatives = Vec::new();
        for p in 0..pairs {
            let up = parent[p];
            parent[p] = if up as usize == p {
                representatives.push(((p / n) as u32, (p % n) as u32));
                (representatives.len() as u32 - 1) | LABEL
            } else {
                parent[up as usize]
            };
        }
        let mut sizes = vec![0u64; representatives.len()];
        for x in parent.iter_mut() {
            *x &= !LABEL;
            sizes[*x as usize] += 1;
        }
        Ok(Orbitals {
            n,
            labels: Labels::Pairs(parent),
            representatives,
            sizes,
        })
    }

    /// Number of points.
    pub fn num_points(&self) -> usize {
        self.n
    }

    /// Number of orbitals.
    pub fn len(&self) -> usize {
        self.sizes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sizes.is_empty()
    }

    /// The orbital of the pair `(u, v)` (panics if `u` or `v` is not a
    /// point).
    pub fn of(&self, u: usize, v: usize) -> usize {
        assert!(
            u < self.n && v < self.n,
            "pair ({u}, {v}) of {} points",
            self.n
        );
        match &self.labels {
            Labels::Pairs(of) => of[u * self.n + v] as usize,
            Labels::Suborbits(s) => s.of(u, v),
        }
    }

    /// The first pair of orbital `o`.
    pub fn representative(&self, o: usize) -> (usize, usize) {
        let (a, b) = self.representatives[o];
        (a as usize, b as usize)
    }

    /// Number of pairs in orbital `o`.
    pub fn size(&self, o: usize) -> usize {
        self.sizes[o] as usize
    }

    /// Whether every pair has its own label (up to 8192 points), so that
    /// dense operators can be read and written.
    fn check_dense(&self) -> Result<()> {
        match self.labels {
            Labels::Pairs(_) => Ok(()),
            Labels::Suborbits(_) => Err(QsoError::invalid(
                "n",
                format!("dense operators on at most 8192 points, not {}", self.n),
            )),
        }
    }

    /// The coefficients of an operator on the orbitals, if it is constant
    /// on each of them (to `tol` of its largest entry; up to 8192 points).
    pub fn coefficients(&self, m: &CMatrix, tol: f64) -> Result<Vec<C64>> {
        self.check_dense()?;
        let n = self.n;
        if m.nrows() != n || m.ncols() != n {
            return Err(QsoError::DimensionMismatch {
                expected: n,
                actual: m.nrows(),
            });
        }
        if !(tol.is_finite() && tol >= 0.0) {
            return Err(QsoError::invalid(
                "tol",
                "the tolerance must be finite and ≥ 0",
            ));
        }
        let x: Vec<C64> = (0..self.len())
            .map(|o| {
                let (a, b) = self.representative(o);
                m[(a, b)]
            })
            .collect();
        let scale = m
            .iter()
            .fold(0.0f64, |s, z| s.max(z.norm()))
            .max(f64::MIN_POSITIVE);
        for u in 0..n {
            for v in 0..n {
                if (m[(u, v)] - x[self.of(u, v)]).norm() > tol * scale {
                    return Err(QsoError::invalid(
                        "state",
                        "the operator is not invariant under the symmetries",
                    ));
                }
            }
        }
        Ok(x)
    }

    /// The operator with coefficients `x` (dense, `n × n`; up to 8192
    /// points).
    pub fn operator(&self, x: &[C64]) -> Result<CMatrix> {
        self.check_dense()?;
        self.check_len(x.len())?;
        Ok(CMatrix::from_fn(self.n, self.n, |u, v| x[self.of(u, v)]))
    }

    fn check_len(&self, len: usize) -> Result<()> {
        if len != self.len() {
            return Err(QsoError::DimensionMismatch {
                expected: self.len(),
                actual: len,
            });
        }
        Ok(())
    }
}

/// A linear map on orbital coefficients, row by row (compressed).
#[derive(Clone, Debug)]
struct OrbitalMap {
    indptr: Vec<usize>,
    cols: Vec<u32>,
    vals: Vec<C64>,
}

impl OrbitalMap {
    /// From the entries of each row, merging repeated columns.
    fn from_rows(rows: Vec<Vec<(u32, C64)>>) -> Self {
        let mut rows = rows.into_iter();
        Self::build(rows.len(), 0, |_, row| row.extend(rows.next().unwrap()))
    }

    /// `num_rows` rows, `fill(r, row)` pushing the entries of row `r`
    /// (columns may repeat); `capacity` entries are reserved.
    fn build<F>(num_rows: usize, capacity: usize, mut fill: F) -> Self
    where
        F: FnMut(usize, &mut Vec<(u32, C64)>),
    {
        let mut indptr = Vec::with_capacity(num_rows + 1);
        indptr.push(0);
        let (mut cols, mut vals) = (Vec::with_capacity(capacity), Vec::with_capacity(capacity));
        let mut row = Vec::new();
        for r in 0..num_rows {
            row.clear();
            fill(r, &mut row);
            row.sort_by_key(|e| e.0);
            for &(c, v) in &row {
                if cols.len() > *indptr.last().unwrap() && *cols.last().unwrap() == c {
                    *vals.last_mut().unwrap() += v;
                } else {
                    cols.push(c);
                    vals.push(v);
                }
            }
            indptr.push(cols.len());
        }
        OrbitalMap { indptr, cols, vals }
    }

    fn nnz(&self) -> usize {
        self.cols.len()
    }

    /// `D^{½}MD^{−½}` with `D = diag(sizes)`: entry `(O, O′)` times
    /// `√(|O|/|O′|)`, the map in orthonormal coordinates `√|O|·x_O`.
    fn scaled(&self, sizes: &[f64]) -> OrbitalMap {
        let mut vals = self.vals.clone();
        for r in 0..self.indptr.len() - 1 {
            for k in self.indptr[r]..self.indptr[r + 1] {
                vals[k] *= (sizes[r] / sizes[self.cols[k] as usize]).sqrt();
            }
        }
        OrbitalMap {
            indptr: self.indptr.clone(),
            cols: self.cols.clone(),
            vals,
        }
    }

    /// The conjugate transpose (columns of every row ascending).
    fn adjoint(&self) -> OrbitalMap {
        let n = self.indptr.len() - 1;
        let mut indptr = vec![0usize; n + 1];
        for &c in &self.cols {
            indptr[c as usize + 1] += 1;
        }
        for r in 0..n {
            indptr[r + 1] += indptr[r];
        }
        let mut next = indptr.clone();
        let mut cols = vec![0u32; self.nnz()];
        let mut vals = vec![C64::new(0.0, 0.0); self.nnz()];
        for r in 0..n {
            for k in self.indptr[r]..self.indptr[r + 1] {
                let c = self.cols[k] as usize;
                cols[next[c]] = r as u32;
                vals[next[c]] = self.vals[k].conj();
                next[c] += 1;
            }
        }
        OrbitalMap { indptr, cols, vals }
    }

    /// The map of `[x; y]` with `x′ = Mx` and `y′ = x`, so that `y` is the
    /// time integral of `x` (rows `N..2N` hold the single entry 1 at
    /// column `r − N`).
    fn doubled(&self) -> OrbitalMap {
        let n = self.indptr.len() - 1;
        let mut indptr = self.indptr.clone();
        let mut cols = self.cols.clone();
        let mut vals = self.vals.clone();
        indptr.reserve(n);
        cols.reserve(n);
        vals.reserve(n);
        for r in 0..n {
            cols.push(r as u32);
            vals.push(C64::new(1.0, 0.0));
            indptr.push(cols.len());
        }
        OrbitalMap { indptr, cols, vals }
    }

    /// Position of row `r`'s diagonal entry in `vals` (every generator of
    /// an open walk has one).
    fn diagonal(&self, r: usize) -> usize {
        let (from, to) = (self.indptr[r], self.indptr[r + 1]);
        from + self.cols[from..to]
            .binary_search(&(r as u32))
            .expect("every row of the generator holds its diagonal")
    }

    /// Largest absolute row sum (the ∞-norm).
    fn norm(&self) -> f64 {
        (0..self.indptr.len() - 1)
            .map(|r| {
                self.vals[self.indptr[r]..self.indptr[r + 1]]
                    .iter()
                    .map(|z| z.norm())
                    .sum::<f64>()
            })
            .fold(0.0, f64::max)
    }

    fn apply(&self, x: &[C64], y: &mut [C64]) {
        for_each_mut(y, |r, out| {
            let mut acc = C64::new(0.0, 0.0);
            for k in self.indptr[r]..self.indptr[r + 1] {
                acc += self.vals[k] * x[self.cols[k] as usize];
            }
            *out = acc;
        });
    }

    /// `exp(t·M)·x` by its Taylor series, in substeps of norm ½, each
    /// summed until a term falls below rounding.
    fn exp_apply(&self, x: &[C64], t: f64) -> Result<Vec<C64>> {
        let norm = self.norm();
        let substeps = (2.0 * norm * t).ceil().max(1.0);
        let work = substeps * 30.0 * 8.0 * self.nnz() as f64;
        if work > MAX_ORBITAL_WORK {
            return Err(QsoError::invalid(
                "t",
                format!(
                    "this evolution needs about {work:.1e} operations (limit {MAX_ORBITAL_WORK:.1e})"
                ),
            ));
        }
        let h = t / substeps;
        let mut out = x.to_vec();
        let mut term = vec![C64::new(0.0, 0.0); x.len()];
        let mut next = term.clone();
        for _ in 0..substeps as usize {
            term.copy_from_slice(&out);
            for k in 1..=60 {
                self.apply(&term, &mut next);
                let scale = C64::new(h / k as f64, 0.0);
                let mut size = 0.0f64;
                for (t, n) in term.iter_mut().zip(&next) {
                    *t = n * scale;
                    size = size.max(t.norm());
                }
                let mut total = 0.0f64;
                for (o, t) in out.iter_mut().zip(&term) {
                    *o += t;
                    total = total.max(o.norm());
                }
                if size <= 1e-17 * total.max(f64::MIN_POSITIVE) {
                    break;
                }
            }
        }
        Ok(out)
    }
}

/// Generators of the permutations of the coordinates of the hypercube
/// `Q_d`, as permutations of its `2^d` nodes: the swap of the first two
/// coordinates and the cycle of all of them (none for `d = 1`). They
/// generate the stabiliser of node `0`, without the search of
/// [`Graph::stabilizer`], which takes seconds from `Q₁₂` on. At most
/// `d = 20` (2²⁰ nodes, the largest generated graphs).
pub fn coordinate_permutations(d: usize) -> Result<Vec<Vec<usize>>> {
    if d > 20 {
        return Err(QsoError::invalid("d", "at most 20 coordinates (2²⁰ nodes)"));
    }
    let n = 1usize << d;
    let permute = |image: &dyn Fn(usize) -> usize| -> Vec<usize> {
        (0..n)
            .map(|u| {
                (0..d)
                    .filter(|&k| (u >> k) & 1 == 1)
                    .map(|k| 1 << image(k))
                    .sum()
            })
            .collect()
    };
    Ok(match d {
        0 | 1 => Vec::new(),
        2 => vec![permute(&|k| 1 - k)],
        _ => vec![
            permute(&|k| if k < 2 { 1 - k } else { k }),
            permute(&|k| (k + 1) % d),
        ],
    })
}

/// Dephasing of a continuous-time walk at `rate` between the blocks of a
/// partition of the nodes: `rate·(Σ_j P_j ρ P_j − ρ)`, `P_j` the projector
/// onto block `j` (`block[u]` is the block of node `u`).
#[derive(Clone, Debug, PartialEq)]
pub struct Dephasing {
    pub rate: f64,
    pub block: Vec<usize>,
}

impl Dephasing {
    /// Every node its own block: site dephasing.
    pub fn sites(n: usize, rate: f64) -> Self {
        Dephasing {
            rate,
            block: (0..n).collect(),
        }
    }

    /// Bit `k` of the node index: dephasing of coordinate `k` of a
    /// hypercube.
    pub fn coordinate(n: usize, k: usize, rate: f64) -> Self {
        Dephasing {
            rate,
            block: (0..n)
                .map(|u| u.checked_shr(k as u32).unwrap_or(0) & 1)
                .collect(),
        }
    }
}

/// A partition given by block labels, relabelled by first appearance, so
/// that equal partitions get equal labels.
fn canonical_blocks<T: Copy + Eq + std::hash::Hash>(block: &[T]) -> Vec<u32> {
    let mut first: std::collections::HashMap<T, u32> = std::collections::HashMap::new();
    block
        .iter()
        .map(|&b| {
            let next = first.len() as u32;
            *first.entry(b).or_insert(next)
        })
        .collect()
}

/// The number of entries of a reduced generator, from the entries of each
/// row, if it is within [`MAX_ORBITAL_ENTRIES`].
fn check_entries(rows: impl Iterator<Item = usize>) -> Result<usize> {
    let mut total = 0usize;
    for r in rows {
        total = total.saturating_add(r);
        if total > MAX_ORBITAL_ENTRIES {
            return Err(QsoError::invalid(
                "symmetries",
                format!(
                    "the reduced generator would have more than {MAX_ORBITAL_ENTRIES} entries: \
                     use more symmetries or a smaller graph"
                ),
            ));
        }
    }
    Ok(total)
}

/// A continuous-time walk with dephasing, on the orbitals of a group of
/// symmetries (see the module documentation).
#[derive(Clone, Debug)]
pub struct SymmetricOpenWalk {
    orbitals: Arc<Orbitals>,
    generator: OrbitalMap,
    /// `γ·max|w_e|`, the hopping scale of the transport preconditioner.
    hopping: f64,
    /// Traps, losses, site energies and per-node dephasing (none until a
    /// builder sets them).
    sites: Option<Box<OrbitalSites>>,
}

/// The site terms of a [`SymmetricOpenWalk`], constant on every orbit of
/// points.
#[derive(Clone, Debug)]
struct OrbitalSites {
    /// Traps `(node, κ)` in the order given.
    traps: Vec<(usize, f64)>,
    /// Every diagonal orbital (an orbit of points), ascending, with its
    /// site energy `ε`, trap rate `κ`, loss `Γ` and dephasing `γ`.
    values: Vec<(u32, [f64; 4])>,
    /// The diagonal of the generator as [`SymmetricOpenWalk::new`] built
    /// it.
    base: Vec<C64>,
    /// Position of every row's diagonal entry in the generator's values.
    diagonal: Vec<usize>,
}

/// Columns of [`OrbitalSites::values`].
const ENERGY: usize = 0;
const TRAP: usize = 1;
const LOSS: usize = 2;
const DEPHASING: usize = 3;

impl OrbitalSites {
    /// `[ε, κ, Γ, γ]` of the diagonal orbital `o`.
    fn of(&self, o: usize) -> &[f64; 4] {
        let k = self
            .values
            .binary_search_by_key(&(o as u32), |v| v.0)
            .expect("a diagonal orbital");
        &self.values[k].1
    }
}

/// Refuse `f` (one value per point) unless it is constant on every orbit
/// of points: `O(n)` lookups, exact because the diagonal orbitals are the
/// orbits on points. Energies agree to `1e-12·(1 + |f(a)|)`; rates (any
/// `name` but `"energies"`) must vanish on the same points, since the
/// representative's value decides the whole orbit, and agree to
/// `1e-12·max(|f(u)|, |f(a)|)`.
fn check_invariant(orbitals: &Orbitals, name: &'static str, f: &[f64]) -> Result<()> {
    let rate = name != "energies";
    for (u, &x) in f.iter().enumerate() {
        let (a, _) = orbitals.representative(orbitals.of(u, u));
        let differ = if rate {
            (x == 0.0) != (f[a] == 0.0) || (x - f[a]).abs() > 1e-12 * x.abs().max(f[a].abs())
        } else {
            (x - f[a]).abs() > 1e-12 * (1.0 + f[a].abs())
        };
        if differ {
            return Err(QsoError::invalid(
                name,
                format!(
                    "not invariant under the symmetries: nodes {u} and {a} lie in one orbit with \
                     {x} and {}",
                    f[a]
                ),
            ));
        }
    }
    Ok(())
}

impl SymmetricOpenWalk {
    /// The walk `H = γ·L` or `−γ·A` of `graph` with the given dephasing,
    /// reduced by the group generated by `symmetries` (automorphisms of the
    /// graph that also preserve the dephasing; none for no reduction).
    pub fn new(
        graph: &Graph,
        generator: Generator,
        gamma: f64,
        dephasing: &[Dephasing],
        symmetries: &[Vec<usize>],
    ) -> Result<Self> {
        let n = graph.num_nodes();
        if !(gamma.is_finite() && gamma > 0.0) {
            return Err(QsoError::invalid(
                "gamma",
                "the hopping rate must be positive",
            ));
        }
        for (k, d) in dephasing.iter().enumerate() {
            if !(d.rate.is_finite() && d.rate >= 0.0) || d.block.len() != n {
                return Err(QsoError::invalid(
                    "dephasing",
                    format!("term {k} needs a finite rate ≥ 0 and a block for each of {n} nodes"),
                ));
            }
        }
        let orbitals = Orbitals::new(n, symmetries)?;
        for (k, g) in symmetries.iter().enumerate() {
            if !is_automorphism(graph, g) {
                return Err(QsoError::invalid(
                    "symmetries",
                    format!("permutation {k} is not an automorphism of the graph"),
                ));
            }
        }
        let decay = |u: usize, v: usize| -> f64 {
            dephasing
                .iter()
                .filter(|d| d.block[u] != d.block[v])
                .map(|d| d.rate)
                .sum()
        };
        // The dephasing must be constant on every orbital: checked on every
        // pair up to 8192 nodes; beyond, terms with the same partition are
        // merged (their rates added), and every symmetry must map each
        // partition onto one of the same rate. A symmetry permutes the
        // partitions, so this keeps the decay of every pair.
        let refused = || {
            QsoError::invalid(
                "dephasing",
                "the dephasing is not invariant under the symmetries",
            )
        };
        if matches!(orbitals.labels, Labels::Pairs(_)) {
            for u in 0..n {
                for v in 0..n {
                    let (a, b) = orbitals.representative(orbitals.of(u, v));
                    if (decay(u, v) - decay(a, b)).abs() > 1e-12 * (1.0 + decay(a, b)) {
                        return Err(refused());
                    }
                }
            }
        } else {
            let mut merged: std::collections::HashMap<Vec<u32>, f64> =
                std::collections::HashMap::new();
            for d in dephasing.iter().filter(|d| d.rate > 0.0) {
                *merged.entry(canonical_blocks(&d.block)).or_insert(0.0) += d.rate;
            }
            let mut moved = vec![0u32; n];
            for g in symmetries {
                for (p, &rate) in &merged {
                    // The partition g(P): the block of g(u) is that of u.
                    for u in 0..n {
                        moved[g[u]] = p[u];
                    }
                    let kept = merged
                        .get(&canonical_blocks(&moved))
                        .is_some_and(|&r| (r - rate).abs() <= 1e-12 * (1.0 + r.max(rate)));
                    if !kept {
                        return Err(refused());
                    }
                }
            }
        }
        let h = match generator {
            Generator::Laplacian => graph.laplacian_csr().scaled(gamma),
            Generator::Adjacency => graph.adjacency_csr().scaled(-gamma),
        };
        let i = C64::new(0.0, 1.0);
        let entries = check_entries((0..orbitals.len()).map(|o| {
            let (a, b) = orbitals.representative(o);
            1 + h.row(a).count() + h.row(b).count()
        }))?;
        let generator = OrbitalMap::build(orbitals.len(), entries, |o, row| {
            let (a, b) = orbitals.representative(o);
            row.push((o as u32, C64::new(-decay(a, b), 0.0)));
            // (Hρ)_ab = Σ_w H_aw ρ_wb and (ρH)_ab = Σ_w ρ_aw H_wb.
            for (w, x) in h.row(a) {
                row.push((orbitals.of(w, b) as u32, -i * x));
            }
            for (w, x) in h.row(b) {
                row.push((orbitals.of(a, w) as u32, i * x));
            }
        });
        let hopping = (0..n)
            .flat_map(|a| h.row(a).filter(move |&(w, _)| w != a))
            .fold(0.0f64, |m, (_, x)| m.max(x.abs()));
        Ok(SymmetricOpenWalk {
            orbitals: Arc::new(orbitals),
            generator,
            hopping,
            sites: None,
        })
    }

    /// The orbitals the state lives on (shared, so that it can be kept
    /// without a copy).
    pub fn orbitals(&self) -> &Arc<Orbitals> {
        &self.orbitals
    }

    pub fn num_orbitals(&self) -> usize {
        self.orbitals.len()
    }

    /// The walker at `node` (which every symmetry must fix).
    pub fn localized(&self, node: usize) -> Result<Vec<C64>> {
        localized(&self.orbitals, &[node])
    }

    /// The uniform superposition of all nodes.
    pub fn uniform(&self) -> Vec<C64> {
        vec![C64::new(1.0 / self.orbitals.n as f64, 0.0); self.orbitals.len()]
    }

    /// The coefficients after time `t ≥ 0` from `x0`.
    pub fn evolve(&self, x0: &[C64], t: f64) -> Result<Vec<C64>> {
        self.orbitals.check_len(x0.len())?;
        if !(t.is_finite() && t >= 0.0) {
            return Err(QsoError::invalid("t", "time must be finite and ≥ 0"));
        }
        if x0.iter().any(|z| !z.is_finite()) {
            return Err(QsoError::invalid("x0", "coefficients must be finite"));
        }
        self.generator.exp_apply(x0, t)
    }

    /// The population of every node.
    pub fn populations(&self, x: &[C64]) -> Result<Vec<f64>> {
        let o = &self.orbitals;
        o.check_len(x.len())?;
        Ok((0..o.n).map(|u| x[o.of(u, u)].re).collect())
    }

    /// Site energies `ε_j` added to the diagonal of `H` (not scaled by the
    /// hopping rate): one per node, equal on every orbit of points of the
    /// symmetries; replaces earlier energies.
    pub fn with_site_energies(self, energies: &[f64]) -> Result<Self> {
        check_energies(energies, self.orbitals.n)?;
        self.with_site_term(ENERGY, "energies", energies)
    }

    /// Traps `(node, κ)`, the sink of node `t` filling at `2κ ρ_tt` (nodes
    /// distinct, `κ > 0`); `κ` must be equal on every orbit of points, so
    /// a whole orbit of traps of equal rate is allowed. Replaces earlier
    /// traps (`&[]` removes them).
    pub fn with_traps(mut self, traps: &[(usize, f64)]) -> Result<Self> {
        let n = self.orbitals.n;
        check_traps(traps, n)?;
        let mut kappa = vec![0.0; n];
        for &(t, k) in traps {
            kappa[t] = k;
        }
        self = self.with_site_term(TRAP, "traps", &kappa)?;
        self.sites.as_mut().expect("set by the term").traps = traps.to_vec();
        Ok(self)
    }

    /// Loss rates `Γ_j ≥ 0`, one per node and equal on every orbit of
    /// points: population leaves node `j` at `Γ_j ρ_jj`. Replaces earlier
    /// losses.
    pub fn with_losses(self, rates: &[f64]) -> Result<Self> {
        check_rates("losses", rates, self.orbitals.n)?;
        self.with_site_term(LOSS, "losses", rates)
    }

    /// Dephasing rates `γ_j ≥ 0`, one per node and equal on every orbit of
    /// points, added to the partition [`Dephasing`] terms of [`Self::new`]
    /// (entry `ρ_ab` decays at `(γ_a + γ_b)/2` more). Replaces earlier
    /// per-node rates.
    pub fn with_node_dephasing(self, rates: &[f64]) -> Result<Self> {
        check_rates("dephasing", rates, self.orbitals.n)?;
        self.with_site_term(DEPHASING, "dephasing", rates)
    }

    /// The traps `(node, κ)` in the order given.
    pub fn traps(&self) -> &[(usize, f64)] {
        self.sites.as_ref().map_or(&[], |s| &s.traps)
    }

    /// Set column `column` of the site terms to `f` (checked for
    /// invariance) and update the diagonal of the generator.
    fn with_site_term(mut self, column: usize, name: &'static str, f: &[f64]) -> Result<Self> {
        let orbitals = &self.orbitals;
        check_invariant(orbitals, name, f)?;
        let generator = &self.generator;
        let sites = self.sites.get_or_insert_with(|| {
            let mut diagonal: Vec<u32> =
                (0..orbitals.n).map(|u| orbitals.of(u, u) as u32).collect();
            diagonal.sort_unstable();
            diagonal.dedup();
            let positions: Vec<usize> =
                (0..orbitals.len()).map(|o| generator.diagonal(o)).collect();
            Box::new(OrbitalSites {
                traps: Vec::new(),
                values: diagonal.into_iter().map(|o| (o, [0.0; 4])).collect(),
                base: positions.iter().map(|&k| generator.vals[k]).collect(),
                diagonal: positions,
            })
        });
        for (o, v) in &mut sites.values {
            let (a, _) = orbitals.representative(*o as usize);
            v[column] = f[a];
        }
        // Row O at its representative (a, b) gains
        // −(k_a + k_b) − [a ≠ b](γ_a + γ_b)/2 − i(ε_a − ε_b), k = κ + Γ/2,
        // always from the diagonal of `new` and in this order, so that the
        // order of the builders changes no bit and zero terms none at all.
        for o in 0..orbitals.len() {
            let (a, b) = orbitals.representative(o);
            let (va, vb) = (sites.of(orbitals.of(a, a)), sites.of(orbitals.of(b, b)));
            let (ka, kb) = (va[TRAP] + 0.5 * va[LOSS], vb[TRAP] + 0.5 * vb[LOSS]);
            let mut decay = -(ka + kb);
            if a != b {
                decay -= 0.5 * (va[DEPHASING] + vb[DEPHASING]);
            }
            let term = C64::new(decay, -(va[ENERGY] - vb[ENERGY]));
            self.generator.vals[sites.diagonal[o]] = if term == C64::new(0.0, 0.0) {
                sites.base[o]
            } else {
                sites.base[o] + term
            };
        }
        Ok(self)
    }

    /// The transport problem from `start` (which every symmetry must fix).
    fn problem(&self, start: usize, options: &TransportOptions) -> Result<OrbitalProblem<'_>> {
        options.validate()?;
        let sites = match &self.sites {
            Some(s) if !s.traps.is_empty() => s,
            _ => {
                return Err(QsoError::invalid(
                    "traps",
                    "transport needs at least one trap (with_traps)",
                ));
            }
        };
        let orbitals = &*self.orbitals;
        localized(orbitals, &[start])?;
        let start_orbital = orbitals.of(start, start);
        let size: Vec<f64> = orbitals.sizes.iter().map(|&s| s as f64).collect();
        let sqrt_size: Vec<f64> = size.iter().map(|s| s.sqrt()).collect();
        let n = orbitals.len();
        let mut traps = vec![C64::new(0.0, 0.0); n];
        let mut losses = vec![C64::new(0.0, 0.0); n];
        let mut lost = Vec::new();
        for &(o, v) in &sites.values {
            let o = o as usize;
            traps[o] = C64::new(sqrt_size[o] * 2.0 * v[TRAP], 0.0);
            if v[LOSS] > 0.0 {
                losses[o] = C64::new(sqrt_size[o] * v[LOSS], 0.0);
                lost.push((o, v[LOSS] * sqrt_size[o]));
            }
        }
        let all_lose = sites.values.iter().all(|v| v.1[LOSS] > 0.0);
        let gamma0 = sites
            .values
            .iter()
            .map(|v| v.1[LOSS])
            .fold(f64::INFINITY, f64::min);
        let mut excess = vec![C64::new(0.0, 0.0); n];
        let mut any_excess = false;
        for &(o, v) in &sites.values {
            if v[LOSS] > gamma0 {
                excess[o as usize] = C64::new(sqrt_size[o as usize] * (v[LOSS] - gamma0), 0.0);
                any_excess = true;
            }
        }
        let trapped = sites
            .traps
            .iter()
            .map(|&(t, kappa)| {
                let o = orbitals.of(t, t);
                (o, 2.0 * kappa / sqrt_size[o])
            })
            .collect();
        let coherence = (0..n)
            .map(|o| {
                let (a, b) = orbitals.representative(o);
                a != b
            })
            .collect();
        let sqrt = sqrt_size.clone();
        let bright = Box::new(move |p: &[C64]| -> (Vec<C64>, f64) {
            let q: Vec<C64> = (0..orbitals.n).map(|u| p[orbitals.of(u, start)]).collect();
            let qn = q.iter().map(|z| z.norm_sqr()).sum();
            let x0 = (0..orbitals.len())
                .map(|o| {
                    let (a, b) = orbitals.representative(o);
                    q[a] * q[b].conj() * sqrt[o]
                })
                .collect();
            (x0, qn)
        });
        Ok(OrbitalProblem {
            m: self.generator.scaled(&size),
            coherence,
            hopping: self.hopping,
            traps,
            losses: if lost.is_empty() { None } else { Some(losses) },
            start: start_orbital,
            all_lose,
            gamma0: if all_lose { gamma0 } else { 0.0 },
            excess: (all_lose && any_excess).then_some(excess),
            trapped,
            lost,
            sqrt_size,
            bright,
        })
    }

    /// Efficiency, mean arrival time, losses and dark weight of a walker
    /// started at `start` (fixed by every symmetry), by linear solves on
    /// the orbitals (see [`super::transport`]); needs a trap.
    pub fn transport(&self, start: usize, options: &TransportOptions) -> Result<TransportResult> {
        self.problem(start, options)?.transport(options)
    }

    /// One result per rate `e ≥ 0` (in input order), with `e` added to
    /// every node's dephasing; one factorisation per rate.
    pub fn dephasing_scan(
        &self,
        start: usize,
        rates: &[f64],
        options: &TransportOptions,
    ) -> Result<Vec<TransportResult>> {
        self.problem(start, options)?.scan(rates, options)
    }

    /// The uniform rate in `[low, high]` (`0 < low < high`, finite) which,
    /// added to every node's dephasing, gives the highest efficiency (see
    /// [`super::DephasingWalk::optimal_dephasing`]).
    pub fn optimal_dephasing(
        &self,
        start: usize,
        low: f64,
        high: f64,
        options: &TransportOptions,
    ) -> Result<DephasingOptimum> {
        self.problem(start, options)?.optimum(low, high, options)
    }

    /// Propagated populations (per node), sink populations (per trap, in
    /// the order of [`Self::traps`]) and the population lost at each of
    /// `times` (finite, `≥ 0`, non-decreasing), from `start` (fixed by
    /// every symmetry): the time integral of the populations is propagated
    /// with them.
    pub fn transport_trace(&self, start: usize, times: &[f64]) -> Result<Vec<TransportState>> {
        let x0 = self.localized(start)?;
        let o = &*self.orbitals;
        let mut reads = TraceReads {
            populations: (0..o.n).map(|u| (o.of(u, u), 1.0)).collect(),
            trapped: Vec::new(),
            lost: Vec::new(),
        };
        if let Some(sites) = &self.sites {
            reads.trapped = sites
                .traps
                .iter()
                .map(|&(t, kappa)| (o.of(t, t), 2.0 * kappa))
                .collect();
            reads.lost = sites
                .values
                .iter()
                .filter(|v| v.1[LOSS] > 0.0)
                .map(|&(d, v)| (d as usize, v[LOSS] * o.size(d as usize) as f64))
                .collect();
        }
        trace(&self.generator, &x0, times, &reads)
    }
}

/// `|ψ⟩⟨ψ|` for the uniform superposition of `points`, which must be a
/// union of orbits on points (every symmetry maps it to itself).
fn localized(orbitals: &Orbitals, points: &[usize]) -> Result<Vec<C64>> {
    let n = orbitals.n;
    if let Some(&u) = points.iter().find(|&&u| u >= n) {
        return Err(QsoError::NodeOutOfRange {
            node: u,
            n_nodes: n,
        });
    }
    let w = C64::new(1.0 / points.len() as f64, 0.0);
    let mut x = vec![C64::new(0.0, 0.0); orbitals.len()];
    let mut count = vec![0usize; orbitals.len()];
    for &a in points {
        for &b in points {
            let o = orbitals.of(a, b);
            x[o] = w;
            count[o] += 1;
        }
    }
    if count
        .iter()
        .zip(&orbitals.sizes)
        .any(|(&c, &s)| c != 0 && c as u64 != s)
    {
        return Err(QsoError::invalid(
            "node",
            "the start is not invariant under the symmetries",
        ));
    }
    Ok(x)
}

/// A coined walk with decoherence ([`DecoherentCoinedWalk`]) on the
/// orbitals of a group of symmetries of its graph, acting on the arcs.
#[derive(Clone, Debug)]
pub struct SymmetricCoinedWalk {
    orbitals: Arc<Orbitals>,
    /// `x ↦` the coefficients of `UρU†`.
    conjugation: OrbitalMap,
    /// The dephasing factor of every orbital.
    factor: Vec<f64>,
    tails: Vec<usize>,
    num_nodes: usize,
    walk: DecoherentCoinedWalk,
}

impl SymmetricCoinedWalk {
    /// `walk` reduced by the group generated by `symmetries` (automorphisms
    /// of its graph, as node permutations). The coin and the dephasing must
    /// be invariant: the Grover coin always is, the dephasing of the coin
    /// register only under symmetries that permute its labels
    /// ([`DecoherentCoinedWalk::with_coin_labels`]; on a hypercube, label
    /// every arc by the coordinate it flips).
    pub fn new(walk: &DecoherentCoinedWalk, symmetries: &[Vec<usize>]) -> Result<Self> {
        let coined = walk.walk();
        let n = coined.num_nodes();
        let arcs = coined.arcs();
        let m = arcs.len();
        // Arcs are grouped by tail, heads increasing.
        let mut first = vec![0usize; n + 1];
        for &(u, _) in &arcs {
            first[u + 1] += 1;
        }
        for u in 0..n {
            first[u + 1] += first[u];
        }
        let index = |u: usize, v: usize| -> Option<usize> {
            arcs[first[u]..first[u + 1]]
                .binary_search_by_key(&v, |a| a.1)
                .ok()
                .map(|k| first[u] + k)
        };
        let mut arc_symmetries = Vec::with_capacity(symmetries.len());
        for (k, g) in symmetries.iter().enumerate() {
            check_permutation(n, g, k)?;
            let p: Option<Vec<usize>> = arcs.iter().map(|&(u, v)| index(g[u], g[v])).collect();
            arc_symmetries.push(p.ok_or_else(|| {
                QsoError::invalid(
                    "symmetries",
                    format!("permutation {k} is not an automorphism of the graph"),
                )
            })?);
        }
        // The step U column by column, then by rows.
        let mut rows: Vec<Vec<(usize, C64)>> = vec![Vec::new(); m];
        let mut e = vec![C64::new(0.0, 0.0); m];
        for c in 0..m {
            e[c] = C64::new(1.0, 0.0);
            for (a, z) in coined.step(&e)?.into_iter().enumerate() {
                if z != C64::new(0.0, 0.0) {
                    rows[a].push((c, z));
                }
            }
            e[c] = C64::new(0.0, 0.0);
        }
        let entry = |a: usize, c: usize| -> C64 {
            rows[a]
                .binary_search_by_key(&c, |x| x.0)
                .map(|k| rows[a][k].1)
                .unwrap_or(C64::new(0.0, 0.0))
        };
        for (k, p) in arc_symmetries.iter().enumerate() {
            for (a, row) in rows.iter().enumerate() {
                for &(c, z) in row {
                    if (entry(p[a], p[c]) - z).norm() > 1e-12 {
                        return Err(QsoError::invalid(
                            "symmetries",
                            format!("the coin is not invariant under permutation {k}"),
                        ));
                    }
                }
            }
        }
        let orbitals = Orbitals::new(m, &arc_symmetries)?;
        let tails: Vec<usize> = arcs.iter().map(|a| a.0).collect();
        let labels = walk.coin_labels();
        let (keep_x, keep_c) = (1.0 - walk.position_dephasing(), 1.0 - walk.coin_dephasing());
        let f = |a: usize, b: usize| -> f64 {
            let mut f = 1.0;
            if tails[a] != tails[b] {
                f *= keep_x;
            }
            if labels[a] != labels[b] {
                f *= keep_c;
            }
            f
        };
        for a in 0..m {
            for b in 0..m {
                let (r, s) = orbitals.representative(orbitals.of(a, b));
                if f(a, b) != f(r, s) {
                    return Err(QsoError::invalid(
                        "symmetries",
                        "the dephasing is not invariant under the symmetries (the coin \
                         register's labels change)",
                    ));
                }
            }
        }
        let factor = (0..orbitals.len())
            .map(|o| {
                let (a, b) = orbitals.representative(o);
                f(a, b)
            })
            .collect();
        // (UρU†)_ab = Σ_cd U_ac ρ_cd conj(U_bd).
        let entries = check_entries((0..orbitals.len()).map(|o| {
            let (a, b) = orbitals.representative(o);
            rows[a].len() * rows[b].len()
        }))?;
        let conjugation = OrbitalMap::build(orbitals.len(), entries, |o, row| {
            let (a, b) = orbitals.representative(o);
            for &(c, x) in &rows[a] {
                for &(d, y) in &rows[b] {
                    row.push((orbitals.of(c, d) as u32, x * y.conj()));
                }
            }
        });
        Ok(SymmetricCoinedWalk {
            orbitals: Arc::new(orbitals),
            conjugation,
            factor,
            tails,
            num_nodes: n,
            walk: walk.clone(),
        })
    }

    /// The orbitals the state lives on (shared, so that it can be kept
    /// without a copy).
    pub fn orbitals(&self) -> &Arc<Orbitals> {
        &self.orbitals
    }

    pub fn num_orbitals(&self) -> usize {
        self.orbitals.len()
    }

    /// The walker at `node` with the uniform coin state (`node` must be
    /// fixed by every symmetry).
    pub fn localized(&self, node: usize) -> Result<Vec<C64>> {
        if node >= self.num_nodes {
            return Err(QsoError::NodeOutOfRange {
                node,
                n_nodes: self.num_nodes,
            });
        }
        let arcs: Vec<usize> = (0..self.tails.len())
            .filter(|&a| self.tails[a] == node)
            .collect();
        if arcs.is_empty() {
            return Err(QsoError::invalid(
                "node",
                format!("node {node} has no edges"),
            ));
        }
        localized(&self.orbitals, &arcs)
    }

    /// The coefficients after `steps` steps from `x0`.
    pub fn evolve(&self, x0: &[C64], steps: usize) -> Result<Vec<C64>> {
        self.orbitals.check_len(x0.len())?;
        if x0.iter().any(|z| !z.is_finite()) {
            return Err(QsoError::invalid("x0", "coefficients must be finite"));
        }
        let mut x = x0.to_vec();
        let mut next = x.clone();
        for _ in 0..steps {
            self.conjugation.apply(&x, &mut next);
            for ((y, z), f) in x.iter_mut().zip(&next).zip(&self.factor) {
                *y = z * f;
            }
        }
        Ok(x)
    }

    /// Probability of each node: the weight of the arcs leaving it.
    pub fn node_probabilities(&self, x: &[C64]) -> Result<Vec<f64>> {
        self.orbitals.check_len(x.len())?;
        let mut p = vec![0.0; self.num_nodes];
        for (a, &u) in self.tails.iter().enumerate() {
            p[u] += x[self.orbitals.of(a, a)].re;
        }
        Ok(p)
    }

    /// The underlying walk.
    pub fn walk(&self) -> &DecoherentCoinedWalk {
        &self.walk
    }
}

/// A walk on the Cartesian product of graphs whose factors dephase
/// independently: `H = Σ_k H_k` and the site dephasing of each factor.
/// From a product state the state stays a product, so each factor evolves
/// on its own. Factors are given with multiplicities: the hypercube `Q_d`
/// is `d` copies of an edge.
#[derive(Clone, Debug)]
pub struct ProductOpenWalk {
    factors: Vec<(DephasingWalk, usize)>,
}

impl ProductOpenWalk {
    /// Factors with their multiplicities (at least one copy in all). A
    /// factor with traps or a positive loss is refused: it loses
    /// population, so its marginals are not distributions (site energies
    /// and per-node dephasing factorise and are allowed).
    pub fn new(factors: Vec<(DephasingWalk, usize)>) -> Result<Self> {
        if factors.iter().map(|f| f.1).sum::<usize>() == 0 {
            return Err(QsoError::invalid("factors", "at least one factor"));
        }
        if factors
            .iter()
            .any(|(w, _)| !w.traps().is_empty() || w.losses().iter().any(|&g| g > 0.0))
        {
            return Err(QsoError::invalid(
                "factors",
                "a factor with traps or losses loses population, so its marginals are not \
                 distributions; use the walk on the product graph",
            ));
        }
        Ok(ProductOpenWalk { factors })
    }

    /// The node populations of each distinct factor at time `t`, every copy
    /// started at `starts[k]`.
    pub fn marginals(&self, starts: &[usize], t: f64) -> Result<Vec<Vec<f64>>> {
        if starts.len() != self.factors.len() {
            return Err(QsoError::DimensionMismatch {
                expected: self.factors.len(),
                actual: starts.len(),
            });
        }
        self.factors
            .iter()
            .zip(starts)
            .map(|((walk, _), &s)| walk.populations(s, t))
            .collect()
    }

    /// The distribution of the number of copies at their target node
    /// (`targets[k]` for every copy of factor `k`), from the marginals: for
    /// a hypercube from `0…0` and targets `1`, the distribution of the
    /// distance travelled.
    pub fn count_distribution(
        &self,
        marginals: &[Vec<f64>],
        targets: &[usize],
    ) -> Result<Vec<f64>> {
        for len in [marginals.len(), targets.len()] {
            if len != self.factors.len() {
                return Err(QsoError::DimensionMismatch {
                    expected: self.factors.len(),
                    actual: len,
                });
            }
        }
        for ((walk, _), p) in self.factors.iter().zip(marginals) {
            if p.len() != walk.num_nodes() {
                return Err(QsoError::DimensionMismatch {
                    expected: walk.num_nodes(),
                    actual: p.len(),
                });
            }
            if p.iter().any(|&q| !(-1e-12..=1.0 + 1e-12).contains(&q)) {
                return Err(QsoError::invalid(
                    "marginals",
                    "populations must lie in [0, 1]",
                ));
            }
        }
        let total: usize = self.factors.iter().map(|f| f.1).sum();
        let mut dist = vec![0.0; total + 1];
        dist[0] = 1.0;
        let mut reached = 0;
        for ((_, copies), (p, &target)) in self.factors.iter().zip(marginals.iter().zip(targets)) {
            let q = p
                .get(target)
                .copied()
                .ok_or(QsoError::NodeOutOfRange {
                    node: target,
                    n_nodes: p.len(),
                })?
                .clamp(0.0, 1.0);
            for _ in 0..*copies {
                reached += 1;
                for k in (0..=reached).rev() {
                    let stay = dist[k] * (1.0 - q);
                    dist[k] = if k > 0 { stay + dist[k - 1] * q } else { stay };
                }
            }
        }
        Ok(dist)
    }
}

mod hypercube;
mod suborbits;
mod transport;
pub use hypercube::{HypercubeOpenWalk, MAX_HYPERCUBE_ORBITALS};
use suborbits::Suborbits;
pub use suborbits::{MAX_SUBORBIT_LABELS, MAX_SUBORBIT_WORK};
use transport::{OrbitalProblem, TraceReads, trace};

#[cfg(test)]
mod tests;
