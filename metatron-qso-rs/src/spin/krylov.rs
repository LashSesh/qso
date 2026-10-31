//! Sparse blocks of the Liouville superoperator, propagated by Krylov
//! subspaces, and their reduction by spin permutations.
//!
//! A block of the master equation's generator is kept as a compressed
//! sparse matrix and only applied to vectors: `exp(tL) v` comes from
//! Arnoldi's process with the step-size control of Expokit (Sidje, ACM
//! Trans. Math. Softw. 24, 130 (1998)), so blocks far beyond what a dense
//! exponential can hold propagate in memory linear in their entries.
//!
//! Spin permutations that leave the Hamiltonian and the relaxation
//! invariant (the protons of a methyl group under its threefold rotation,
//! for example) map the elements `|a⟩⟨b|` of a block onto each other. A
//! state invariant under them stays invariant and is constant on each
//! orbit of elements: the block reduces to one coefficient per orbit, its
//! generator to `M_OP = Σ_{c ∈ P} L_{o,c}` read at a representative `o` of
//! each orbit `O`.

use num_complex::Complex64 as C64;

use crate::error::{QsoError, Result};
use crate::linalg::CMatrix;
use crate::linalg::dense::expm;
use crate::parallel::{CHUNK, for_each_block};

/// Dimension of the Krylov subspaces.
const KRYLOV_DIM: usize = 30;

/// Most Krylov steps of one propagation.
const MAX_KRYLOV_STEPS: usize = 1 << 22;

/// Error of a Krylov propagation in the 2-norm, relative to the size of
/// the vector propagated (the deviation from the maximally mixed state,
/// or its equilibrium value when larger).
pub(super) const KRYLOV_TOLERANCE: f64 = 1e-12;

/// A vector that moves by less than this fraction of itself over the
/// rest of the interval is taken as stationary (one step). The deviation
/// of a thermal state from `1/d`, parts in 10⁵ of it, is itself only known
/// to about 10⁻¹¹, so equilibrium seems to move at that level.
pub(super) const KRYLOV_STATIONARY: f64 = 1e-11;

/// A square sparse matrix by rows.
#[derive(Clone, Debug)]
pub(super) struct SparseBlock {
    pub n: usize,
    indptr: Vec<usize>,
    cols: Vec<u32>,
    vals: Vec<C64>,
}

impl SparseBlock {
    /// From `(row, column, value)` triplets; repeated entries add up and
    /// zeros are dropped.
    pub fn from_triplets(n: usize, mut triplets: Vec<(u32, u32, C64)>) -> Self {
        triplets.sort_unstable_by_key(|&(r, c, _)| (r, c));
        let mut indptr = vec![0usize; n + 1];
        let mut cols: Vec<u32> = Vec::with_capacity(triplets.len());
        let mut vals: Vec<C64> = Vec::with_capacity(triplets.len());
        let mut last: Option<(u32, u32)> = None;
        for (r, c, v) in triplets {
            if last == Some((r, c)) {
                *vals.last_mut().expect("an entry") += v;
            } else {
                cols.push(c);
                vals.push(v);
                indptr[r as usize + 1] += 1;
                last = Some((r, c));
            }
        }
        for r in 0..n {
            indptr[r + 1] += indptr[r];
        }
        let mut block = SparseBlock {
            n,
            indptr,
            cols,
            vals,
        };
        block.drop_zeros();
        block
    }

    /// From its columns, each a list of `(row, value)` without repeated
    /// rows or zeros.
    pub fn from_columns(n: usize, columns: &[Vec<(u32, C64)>]) -> Self {
        let mut indptr = vec![0usize; n + 1];
        for column in columns {
            for &(r, _) in column {
                indptr[r as usize + 1] += 1;
            }
        }
        for r in 0..n {
            indptr[r + 1] += indptr[r];
        }
        let nnz = indptr[n];
        let mut next = indptr.clone();
        let mut cols = vec![0u32; nnz];
        let mut vals = vec![C64::new(0.0, 0.0); nnz];
        // Columns in order: every row's entries come out sorted.
        for (c, column) in columns.iter().enumerate() {
            for &(r, v) in column {
                let k = next[r as usize];
                cols[k] = c as u32;
                vals[k] = v;
                next[r as usize] += 1;
            }
        }
        SparseBlock {
            n,
            indptr,
            cols,
            vals,
        }
    }

    /// From its rows, each a list of `(column, value)` by ascending
    /// column without zeros.
    pub fn from_rows(n: usize, rows: Vec<Vec<(u32, C64)>>) -> Self {
        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0);
        let nnz = rows.iter().map(Vec::len).sum();
        let mut cols = Vec::with_capacity(nnz);
        let mut vals = Vec::with_capacity(nnz);
        for row in rows {
            for (c, v) in row {
                cols.push(c);
                vals.push(v);
            }
            indptr.push(cols.len());
        }
        SparseBlock {
            n,
            indptr,
            cols,
            vals,
        }
    }

    fn drop_zeros(&mut self) {
        let zero = C64::new(0.0, 0.0);
        let mut indptr = vec![0usize; self.n + 1];
        let mut cols = Vec::with_capacity(self.cols.len());
        let mut vals = Vec::with_capacity(self.vals.len());
        for r in 0..self.n {
            for k in self.indptr[r]..self.indptr[r + 1] {
                if self.vals[k] != zero {
                    cols.push(self.cols[k]);
                    vals.push(self.vals[k]);
                }
            }
            indptr[r + 1] = cols.len();
        }
        self.indptr = indptr;
        self.cols = cols;
        self.vals = vals;
    }

    pub fn nnz(&self) -> usize {
        self.vals.len()
    }

    /// Bytes held.
    pub fn bytes(&self) -> usize {
        self.vals.len() * (16 + 4) + self.indptr.len() * 8
    }

    /// The entries of row `r`.
    pub fn row(&self, r: usize) -> impl Iterator<Item = (usize, C64)> + '_ {
        (self.indptr[r]..self.indptr[r + 1]).map(|k| (self.cols[k] as usize, self.vals[k]))
    }

    /// The columns and values of row `r`, by ascending column.
    pub fn row_parts(&self, r: usize) -> (&[u32], &[C64]) {
        let range = self.indptr[r]..self.indptr[r + 1];
        (&self.cols[range.clone()], &self.vals[range])
    }

    /// The entry at `(r, c)`.
    pub fn get(&self, r: usize, c: usize) -> C64 {
        let row = &self.cols[self.indptr[r]..self.indptr[r + 1]];
        match row.binary_search(&(c as u32)) {
            Ok(k) => self.vals[self.indptr[r] + k],
            Err(_) => C64::new(0.0, 0.0),
        }
    }

    /// `y = A x`, rows in parallel once the entries are many.
    pub fn apply(&self, x: &[C64], y: &mut [C64]) {
        let row = |r: usize| {
            let mut acc = C64::new(0.0, 0.0);
            for k in self.indptr[r]..self.indptr[r + 1] {
                acc += self.vals[k] * x[self.cols[k] as usize];
            }
            acc
        };
        // About 2¹⁶ entries per task.
        let per_row = self.nnz().div_ceil(self.n.max(1)).max(1);
        let rows = (16 * CHUNK / per_row).max(64);
        for_each_block(y, rows, |k, chunk| {
            for (i, out) in chunk.iter_mut().enumerate() {
                *out = row(k * rows + i);
            }
        });
    }

    /// The largest absolute entry.
    pub fn largest(&self) -> f64 {
        self.vals.iter().fold(0.0, |m, z| m.max(z.norm()))
    }

    /// The largest absolute row sum.
    pub fn norm_inf(&self) -> f64 {
        (0..self.n)
            .map(|r| {
                self.vals[self.indptr[r]..self.indptr[r + 1]]
                    .iter()
                    .map(|z| z.norm())
                    .sum::<f64>()
            })
            .fold(0.0, f64::max)
    }

    pub fn to_dense(&self) -> CMatrix {
        let mut m = CMatrix::zeros(self.n, self.n);
        for r in 0..self.n {
            for (c, v) in self.row(r) {
                m[(r, c)] += v;
            }
        }
        m
    }
}

/// Sums over a block's elements that remember which they touched, for
/// building a sparse column.
pub(super) struct Accumulator {
    values: Vec<C64>,
    seen: Vec<bool>,
    touched: Vec<u32>,
}

impl Accumulator {
    pub fn new(n: usize) -> Self {
        Accumulator {
            values: vec![C64::new(0.0, 0.0); n],
            seen: vec![false; n],
            touched: Vec::new(),
        }
    }

    #[inline]
    pub fn add(&mut self, i: usize, v: C64) {
        if !self.seen[i] {
            self.seen[i] = true;
            self.touched.push(i as u32);
        }
        self.values[i] += v;
    }

    /// The nonzero sums by ascending index, leaving the accumulator empty.
    pub fn take(&mut self) -> Vec<(u32, C64)> {
        self.touched.sort_unstable();
        let mut out = Vec::with_capacity(self.touched.len());
        for &i in &self.touched {
            let v = std::mem::replace(&mut self.values[i as usize], C64::new(0.0, 0.0));
            self.seen[i as usize] = false;
            if v != C64::new(0.0, 0.0) {
                out.push((i, v));
            }
        }
        self.touched.clear();
        out
    }
}

fn norm2(x: &[C64]) -> f64 {
    x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

/// `t` rounded up to two significant digits (as Expokit does).
fn round2(t: f64) -> f64 {
    if !(t.is_finite() && t > 0.0) {
        return t;
    }
    let s = 10f64.powf(t.log10().floor() - 1.0);
    (t / s).ceil() * s
}

/// Estimated operations of [`expv`] for a block of `n` elements with
/// `nnz` entries and `‖A‖∞ = norm` over `t`: a Krylov step covers about
/// `m/4` of `‖A‖ t`, and costs `m + 1` products, the orthogonalisation and
/// a small exponential.
pub(super) fn krylov_work(n: usize, nnz: usize, norm: f64, t: f64) -> f64 {
    let m = KRYLOV_DIM.min(n.max(1)) as f64;
    let steps = (norm * t / (0.25 * m)).ceil() + 1.0;
    let n = n as f64;
    steps * (8.0 * (m + 1.0) * nnz as f64 + 8.0 * m * m * n + 120.0 * (m + 2.0).powi(3))
}

/// Bytes [`expv`] holds for a vector of `n` entries.
pub(super) fn krylov_bytes(n: usize) -> f64 {
    16.0 * (KRYLOV_DIM + 4) as f64 * n as f64
}

/// `exp(t A) v` by Arnoldi's process on Krylov subspaces of dimension 30
/// with the step-size control of Expokit, to an absolute error of about
/// `tol` in the 2-norm.
pub(super) fn expv(a: &SparseBlock, v: &[C64], t: f64, tol: f64) -> Result<Vec<C64>> {
    let n = v.len();
    let mut w = v.to_vec();
    let mut beta = norm2(&w);
    if beta == 0.0 || t == 0.0 || n == 0 {
        return Ok(w);
    }
    let m = KRYLOV_DIM.min(n);
    let anorm = a.norm_inf().max(f64::MIN_POSITIVE);
    // Expokit bounds the error per unit time.
    let tol = (tol / t).max(f64::MIN_POSITIVE);
    let (delta, gamma) = (1.2, 0.9);
    let mut xm = 1.0 / m as f64;
    let mf = m as f64;
    let fact = ((mf + 1.0) / std::f64::consts::E).powf(mf + 1.0)
        * (2.0 * std::f64::consts::PI * (mf + 1.0)).sqrt();
    let mut t_new = round2((1.0 / anorm) * ((fact * tol) / (4.0 * beta * anorm)).powf(xm));
    let mut t_now = 0.0;
    let mut basis: Vec<Vec<C64>> = vec![vec![C64::new(0.0, 0.0); n]; m + 1];
    let mut p = vec![C64::new(0.0, 0.0); n];
    let mut steps = 0usize;
    while t_now < t {
        steps += 1;
        if steps > MAX_KRYLOV_STEPS {
            return Err(QsoError::NoConvergence {
                method: "Krylov propagation",
                iterations: steps,
                residual: t - t_now,
            });
        }
        let mut t_step = (t - t_now).min(t_new);
        let mut h = CMatrix::zeros(m + 2, m + 2);
        for (b, x) in basis[0].iter_mut().zip(&w) {
            *b = x / beta;
        }
        // k1 = 0 after a breakdown: the subspace of dimension mb is
        // invariant to the tolerance.
        let (mut k1, mut mb) = (2usize, m);
        for j in 0..m {
            a.apply(&basis[j], &mut p);
            for i in 0..=j {
                let hij: C64 = basis[i].iter().zip(&p).map(|(b, x)| b.conj() * x).sum();
                h[(i, j)] = hij;
                for (x, b) in p.iter_mut().zip(&basis[i]) {
                    *x -= hij * b;
                }
            }
            let s = norm2(&p);
            // Dropping the rest errs by at most β·s per unit time.
            if s <= tol / beta || s <= 1e-15 * anorm || s * (t - t_now) <= KRYLOV_STATIONARY {
                k1 = 0;
                mb = j + 1;
                t_step = t - t_now;
                break;
            }
            h[(j + 1, j)] = C64::new(s, 0.0);
            for (b, x) in basis[j + 1].iter_mut().zip(&p) {
                *b = x / s;
            }
        }
        let mut avnorm = 0.0;
        if k1 != 0 {
            h[(m + 1, m)] = C64::new(1.0, 0.0);
            a.apply(&basis[m], &mut p);
            avnorm = norm2(&p);
        }
        let mut rejected = 0;
        let (f, err) = loop {
            let mx = mb + k1;
            let small = CMatrix::from_fn(mx, mx, |r, c| h[(r, c)] * t_step);
            let f = expm(&small);
            if k1 == 0 {
                break (f, 0.0);
            }
            let phi1 = (f[(m, 0)] * beta).norm();
            let phi2 = (f[(m + 1, 0)] * beta * avnorm).norm();
            let err = if phi1 > 10.0 * phi2 {
                xm = 1.0 / mf;
                phi2
            } else if phi1 > phi2 {
                xm = 1.0 / mf;
                phi1 * phi2 / (phi1 - phi2)
            } else {
                xm = 1.0 / (mf - 1.0).max(1.0);
                phi1
            };
            if err <= delta * t_step * tol {
                break (f, err);
            }
            rejected += 1;
            if rejected > 10 {
                return Err(QsoError::NoConvergence {
                    method: "Krylov propagation (step rejected)",
                    iterations: steps,
                    residual: err,
                });
            }
            t_step = round2(gamma * t_step * (t_step * tol / err).powf(xm));
        };
        let mx = mb + k1.saturating_sub(1);
        for x in w.iter_mut() {
            *x = C64::new(0.0, 0.0);
        }
        for i in 0..mx.min(m + 1) {
            let c = f[(i, 0)] * beta;
            for (x, b) in w.iter_mut().zip(&basis[i]) {
                *x += c * b;
            }
        }
        beta = norm2(&w);
        // The last step lands on t exactly.
        t_now = if t_step >= t - t_now {
            t
        } else {
            t_now + t_step
        };
        if beta == 0.0 {
            break;
        }
        // Expokit's next step, growing at most tenfold (an error far below
        // the tolerance, or zero, would let it jump arbitrarily).
        let grow = if err > 0.0 {
            (t_step * tol / err).powf(xm)
        } else {
            f64::INFINITY
        };
        t_new = round2((gamma * t_step * grow).min(10.0 * t_step));
    }
    Ok(w)
}

/// The orbits of a block's elements under permutations of them.
#[derive(Clone)]
pub(super) struct Orbits {
    /// Orbit of every element.
    pub of: Vec<u32>,
    /// One element of every orbit.
    pub representatives: Vec<u32>,
    /// Elements of every orbit.
    pub sizes: Vec<u32>,
}

impl Orbits {
    /// Orbits of `n` elements under the maps `images[g][x]`.
    pub fn new(n: usize, images: &[Vec<u32>]) -> Self {
        let mut parent: Vec<u32> = (0..n as u32).collect();
        fn find(parent: &mut [u32], mut x: u32) -> u32 {
            while parent[x as usize] != x {
                let up = parent[parent[x as usize] as usize];
                parent[x as usize] = up;
                x = up;
            }
            x
        }
        for map in images {
            for x in 0..n as u32 {
                let (a, b) = (find(&mut parent, x), find(&mut parent, map[x as usize]));
                if a != b {
                    parent[a.max(b) as usize] = a.min(b);
                }
            }
        }
        let mut of = vec![0u32; n];
        let mut representatives = Vec::new();
        let mut sizes = Vec::new();
        let mut index = vec![u32::MAX; n];
        for (x, orbit) in of.iter_mut().enumerate() {
            let root = find(&mut parent, x as u32) as usize;
            if index[root] == u32::MAX {
                index[root] = representatives.len() as u32;
                representatives.push(root as u32);
                sizes.push(0);
            }
            *orbit = index[root];
            sizes[index[root] as usize] += 1;
        }
        Orbits {
            of,
            representatives,
            sizes,
        }
    }

    pub fn len(&self) -> usize {
        self.representatives.len()
    }

    /// The generator on orbit coefficients: row `O` is row `o` of `a`
    /// summed over the orbits of its columns.
    pub fn reduce(&self, a: &SparseBlock) -> SparseBlock {
        let mut triplets = Vec::new();
        for (o, &rep) in self.representatives.iter().enumerate() {
            for (c, v) in a.row(rep as usize) {
                triplets.push((o as u32, self.of[c], v));
            }
        }
        SparseBlock::from_triplets(self.len(), triplets)
    }

    /// The mean of `x` over every orbit.
    pub fn average(&self, x: &[C64]) -> Vec<C64> {
        let mut out = vec![C64::new(0.0, 0.0); self.len()];
        for (v, &o) in x.iter().zip(&self.of) {
            out[o as usize] += v;
        }
        for (v, &s) in out.iter_mut().zip(&self.sizes) {
            *v /= f64::from(s);
        }
        out
    }

    /// The largest difference between `x` and its orbit means.
    pub fn asymmetry(&self, x: &[C64]) -> f64 {
        let mean = self.average(x);
        x.iter()
            .zip(&self.of)
            .map(|(v, &o)| (v - mean[o as usize]).norm())
            .fold(0.0, f64::max)
    }

    /// Whether `a` commutes with the permutations `images`: every entry
    /// equals its image within `tol`.
    pub fn invariant(a: &SparseBlock, images: &[Vec<u32>], tol: f64) -> bool {
        images.iter().all(|map| {
            (0..a.n).all(|r| {
                a.row(r)
                    .all(|(c, v)| (a.get(map[r] as usize, map[c] as usize) - v).norm() <= tol)
            })
        })
    }
}

#[cfg(test)]
mod tests;
