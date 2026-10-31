//! Block structure of spin dynamics. The free Hamiltonian conserves `2M`
//! of every isotope in the rotating frame and of all spins in the
//! laboratory frame (the *labels* of the basis states), and the Lindblad
//! equation conserves the coherence order `label(a) − label(b)` of every
//! element `|a⟩⟨b|`, as each jump operator acts on one spin. Spectra, time
//! averages and propagation therefore work block by block: the
//! eigendecomposition of each block of `H`, shifted by the block's mean
//! energy so that a large Zeeman energy costs no precision, and the
//! exponential of each block of the Lindblad superoperator.
//!
//! Relaxation is propagated for the deviation `Δ = ρ − Tr(ρ)/d` from the
//! maximally mixed state, which obeys `dΔ/dt = L(Δ) + Tr(ρ) L(1/d)`:
//! thermal polarisations are parts in 10⁵ of `1/d`, and errors relative to
//! `Δ` rather than to `ρ` keep them accurate.

use std::collections::BTreeMap;

use num_complex::Complex64 as C64;

use super::krylov::{
    Accumulator, KRYLOV_STATIONARY, KRYLOV_TOLERANCE, Orbits, SparseBlock, accumulator_bytes,
    column_ranges, expv, krylov_bytes, krylov_work,
};
use super::redfield::RedfieldBlocks;
use super::{Frame, Isotope, ONE, SpinSystem, ZERO};
use crate::error::{QsoError, Result};
use crate::linalg::dense::{cmul, expm, expm_squarings, one_norm};
use crate::linalg::{CMatrix, eigh};
use crate::parallel::map_tasks;

/// Energies of one block closer than this fraction of its spread are
/// degenerate (the rounding of its eigendecomposition).
const DEGENERATE: f64 = 1e3 * f64::EPSILON;
/// Lines weaker than this fraction of the largest entry of the state are
/// rounding (the identity part of ρ is about `1/d`).
const LINE_FLOOR: f64 = 1e-13;
/// Local error of a Dormand–Prince step, relative to the largest entry of
/// the deviation `Δ`.
const DP_TOLERANCE: f64 = 1e-11;
/// Rotation angle (rad) per Dormand–Prince step at [`DP_TOLERANCE`], for
/// estimating its work.
const DP_RADIANS_PER_STEP: f64 = 0.06;
/// Relative cost of the Dormand–Prince kernels (scalar loops) against the
/// real matrix products of the exponentials.
const DP_PENALTY: f64 = 4.0;
/// A block and a state are invariant under a symmetry when every entry
/// equals its image within this fraction of their largest entry (of the
/// whole deviation, for a state).
const SYMMETRY_TOLERANCE: f64 = 1e-11;
/// Blocks of the deviation below this fraction of its largest entry are
/// rounding, and left empty.
const NEGLIGIBLE: f64 = 1e-15;

/// A block of basis states with the same labels, and its part of the
/// eigendecomposition of `H`.
pub(super) struct Eigenblock {
    pub states: Vec<usize>,
    pub label: Vec<i32>,
    /// Mean of the block's diagonal of `H`, subtracted before the
    /// decomposition.
    pub shift: f64,
    /// Eigenvalues minus `shift`, ascending.
    pub values: Vec<f64>,
    pub vectors: CMatrix,
}

impl Eigenblock {
    fn spread(&self) -> f64 {
        self.values.iter().fold(0.0, |m, v| m.max(v.abs()))
    }
}

/// `M[A, B]` for blocks of states `A` and `B`.
fn submatrix(m: &CMatrix, a: &[usize], b: &[usize]) -> CMatrix {
    CMatrix::from_fn(a.len(), b.len(), |r, c| m[(a[r], b[c])])
}

/// `V_A† M[A, B] V_B`.
fn to_eigenbasis(sub: &CMatrix, a: &Eigenblock, b: &Eigenblock) -> CMatrix {
    cmul(&cmul(&a.vectors.adjoint(), sub), &b.vectors)
}

/// Write `sub` into `out[A, B]`, times `w`.
fn scatter(out: &mut CMatrix, sub: &CMatrix, a: &[usize], b: &[usize], w: C64) {
    for (r, &i) in a.iter().enumerate() {
        for (c, &j) in b.iter().enumerate() {
            out[(i, j)] = sub[(r, c)] * w;
        }
    }
}

fn max_entry(m: &CMatrix) -> f64 {
    m.iter().fold(0.0, |x, z| x.max(z.norm()))
}

/// Nonzero entries of a matrix by row and by column, and its diagonal.
pub(super) struct Sparse {
    rows: Vec<Vec<(usize, C64)>>,
    cols: Vec<Vec<(usize, C64)>>,
    diagonal: Vec<C64>,
}

impl Sparse {
    pub(super) fn new(h: &CMatrix) -> Self {
        let d = h.nrows();
        let mut rows = vec![Vec::new(); d];
        let mut cols = vec![Vec::new(); d];
        for c in 0..d {
            for r in 0..d {
                let v = h[(r, c)];
                if v != ZERO {
                    rows[r].push((c, v));
                    cols[c].push((r, v));
                }
            }
        }
        let diagonal = (0..d).map(|i| h[(i, i)]).collect();
        Sparse {
            rows,
            cols,
            diagonal,
        }
    }

    fn entries(&self) -> usize {
        self.rows.iter().map(Vec::len).sum()
    }

    /// The largest absolute row sum off the diagonal.
    fn off_diagonal(&self) -> f64 {
        self.rows
            .iter()
            .enumerate()
            .map(|(r, row)| {
                row.iter()
                    .filter(|&&(c, _)| c != r)
                    .map(|(_, v)| v.norm())
                    .sum::<f64>()
            })
            .fold(0.0, f64::max)
    }

    /// Width of the spectrum of a Hermitian matrix, bounded by
    /// Gershgorin's discs: the largest angular frequency of `[H, ·]`.
    fn spread(&self) -> f64 {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for (r, row) in self.rows.iter().enumerate() {
            let mut centre = 0.0;
            let mut radius = 0.0;
            for &(c, v) in row {
                if c == r {
                    centre = v.re;
                } else {
                    radius += v.norm();
                }
            }
            lo = lo.min(centre - radius);
            hi = hi.max(centre + radius);
        }
        if hi >= lo { hi - lo } else { 0.0 }
    }
}

/// A jump operator on one spin, for in-place dissipators: the states by
/// their level of the spin, and the nonzero entries of `A` and `A†A`.
struct Jump {
    stride: usize,
    members: Vec<Vec<usize>>,
    op: Vec<(usize, usize, C64)>,
    ada: Vec<(usize, usize, C64)>,
}

fn nonzeros(m: &CMatrix) -> Vec<(usize, usize, C64)> {
    let mut out = Vec::new();
    for c in 0..m.ncols() {
        for r in 0..m.nrows() {
            if m[(r, c)] != ZERO {
                out.push((r, c, m[(r, c)]));
            }
        }
    }
    out
}

/// `index + (to − from)·stride`.
fn moved(index: usize, from: usize, to: usize, stride: usize) -> usize {
    index + to * stride - from * stride
}

impl Jump {
    /// `out += A x A† − ½(A†A x + x A†A)`, `A` on one spin.
    fn dissipate(&self, out: &mut CMatrix, x: &CMatrix) {
        let d = x.nrows();
        let s = self.stride;
        for &(a, a2, v) in &self.op {
            for &(b, b2, w) in &self.op {
                let f = v * w.conj();
                for &t in &self.members[b2] {
                    let c = moved(t, b2, b, s);
                    for &r in &self.members[a2] {
                        out[(moved(r, a2, a, s), c)] += f * x[(r, t)];
                    }
                }
            }
        }
        for &(a, a2, v) in &self.ada {
            let f = v * -0.5;
            for t in 0..d {
                for &r in &self.members[a2] {
                    out[(moved(r, a2, a, s), t)] += f * x[(r, t)];
                }
            }
        }
        for &(b2, b, w) in &self.ada {
            let f = w * -0.5;
            for &t in &self.members[b2] {
                let c = moved(t, b2, b, s);
                for r in 0..d {
                    out[(r, c)] += x[(r, t)] * f;
                }
            }
        }
    }
}

/// A coherence order and its elements `(a, b)`.
type OrderBlock = (Vec<i32>, Vec<(usize, usize)>);

/// What relaxes the state at one field: the Lindblad operators of `T₁`
/// and `T₂`, the Redfield superoperator, and the drift `L(1/d) + R(ρ_eq)`
/// of the deviation's equation (for unit trace).
pub(super) struct Relaxation {
    jumps: Vec<(usize, CMatrix)>,
    prepared: Vec<Jump>,
    redfield: Option<RedfieldBlocks>,
    pub(super) drift: CMatrix,
    /// A bound on the relaxation rates.
    pub(super) rate: f64,
    /// `‖ρ_eq − 1/d‖` (Frobenius) of thermal equilibrium at the field.
    equilibrium: f64,
}

impl Relaxation {
    /// `out += D(x)`: the dissipators and `−R(x)`, without the drift; the
    /// Redfield blocks it uses are kept for the next call.
    fn dissipate_into(&self, out: &mut CMatrix, x: &CMatrix) {
        for jump in &self.prepared {
            jump.dissipate(out, x);
        }
        if let Some(r) = &self.redfield {
            r.apply(x, out, -1.0);
        }
    }

    /// `D(x)`, keeping nothing.
    pub(super) fn dissipate(&self, x: &CMatrix) -> CMatrix {
        let mut out = CMatrix::zeros(x.nrows(), x.ncols());
        for jump in &self.prepared {
            jump.dissipate(&mut out, x);
        }
        if let Some(r) = &self.redfield {
            r.image(x, &mut out, -1.0);
        }
        out
    }

    /// Entries (at most) of the Redfield blocks of `liouville`'s block
    /// `k` (all its blocks for `None`), and the work of building them.
    fn redfield_entries(&self, liouville: &Liouville, k: Option<usize>) -> (f64, f64) {
        let Some(r) = &self.redfield else {
            return (0.0, 0.0);
        };
        let mut entries = 0.0;
        for (q, (_, elements)) in r.liouville.blocks.iter().enumerate() {
            let (a, b) = elements[0];
            if k.is_none_or(|k| liouville.block_of(a, b) == Some(k)) {
                entries += r.count(q) as f64;
            }
        }
        (entries, 16.0 * r.terms() as f64 * entries)
    }
}

/// A jump operator with `A†A`, on its spin, for building generators.
struct LocalJump<'a> {
    stride: usize,
    n: usize,
    op: &'a CMatrix,
    ada: CMatrix,
}

/// The entries of `−i[H, |a⟩⟨b|]` and of the Lindblad dissipators at
/// `|a⟩⟨b|`: `add(c, e, value)` for `|c⟩⟨e|` (repeated `(c, e)` add up).
fn coherent_column(
    a: usize,
    b: usize,
    h: &Sparse,
    locals: &[LocalJump<'_>],
    mut add: impl FnMut(usize, usize, C64),
) {
    let mi = C64::new(0.0, -1.0);
    // −i(H|a⟩⟨b| − |a⟩⟨b|H), the diagonal as one difference: the Zeeman
    // energies cancel there before anything else is added.
    let own = h.diagonal[a] - h.diagonal[b];
    if own != ZERO {
        add(a, b, mi * own);
    }
    for &(c, v) in &h.cols[a] {
        if c != a {
            add(c, b, mi * v);
        }
    }
    for &(c, v) in &h.rows[b] {
        if c != b {
            add(a, c, -mi * v);
        }
    }
    // A|a⟩⟨b|A† − ½A†A|a⟩⟨b| − ½|a⟩⟨b|A†A on one spin.
    for jump in locals {
        let (stride, n, op, ada) = (jump.stride, jump.n, jump.op, &jump.ada);
        let (ka, kb) = ((a / stride) % n, (b / stride) % n);
        for r in 0..n {
            let x = op[(r, ka)];
            if x == ZERO {
                continue;
            }
            for s in 0..n {
                let y = op[(s, kb)].conj();
                if y != ZERO {
                    add(
                        a + r * stride - ka * stride,
                        b + s * stride - kb * stride,
                        x * y,
                    );
                }
            }
        }
        for r in 0..n {
            let x = ada[(r, ka)];
            if x != ZERO {
                add(a + r * stride - ka * stride, b, -x * 0.5);
            }
            let y = ada[(kb, r)];
            if y != ZERO {
                add(a, b + r * stride - kb * stride, -y * 0.5);
            }
        }
    }
}

/// The elements `|a⟩⟨b|` of a `d × d` operator grouped by coherence
/// order, each with its block and its index in it.
pub(super) struct Liouville {
    pub blocks: Vec<OrderBlock>,
    /// Index of `(a, b)` in its block, at `a·d + b` (`u32::MAX`: not
    /// kept).
    position: Vec<u32>,
    /// The block of `(a, b)`, at `a·d + b`.
    block: Vec<u32>,
    d: usize,
}

fn difference(x: &[i32], y: &[i32]) -> Vec<i32> {
    x.iter().zip(y).map(|(a, b)| a - b).collect()
}

/// Distinct labels, and the index of each state's label among them.
pub(super) fn distinct(labels: &[Vec<i32>]) -> (Vec<Vec<i32>>, Vec<usize>) {
    let mut ids: BTreeMap<&Vec<i32>, usize> = BTreeMap::new();
    for l in labels {
        let next = ids.len();
        ids.entry(l).or_insert(next);
    }
    let mut list = vec![Vec::new(); ids.len()];
    for (l, &k) in &ids {
        list[k] = (*l).clone();
    }
    let id = labels.iter().map(|l| ids[l]).collect();
    (list, id)
}

impl Liouville {
    /// The block of `(a, b)`, if kept.
    pub fn block_of(&self, a: usize, b: usize) -> Option<usize> {
        let k = self.block[a * self.d + b];
        (k != u32::MAX).then_some(k as usize)
    }

    /// The index of `(a, b)` in its block.
    pub fn position_of(&self, a: usize, b: usize) -> usize {
        self.position[a * self.d + b] as usize
    }

    /// Every block, or only the one of order `only`.
    pub fn new(labels: &[Vec<i32>], only: Option<&[i32]>) -> Self {
        let d = labels.len();
        let (list, id) = distinct(labels);
        let m = list.len();
        let mut order: BTreeMap<Vec<i32>, usize> = BTreeMap::new();
        let mut blocks: Vec<OrderBlock> = Vec::new();
        let mut pair = vec![usize::MAX; m * m];
        for x in 0..m {
            for y in 0..m {
                let q = difference(&list[x], &list[y]);
                if only.is_some_and(|o| o != q.as_slice()) {
                    continue;
                }
                let k = *order.entry(q.clone()).or_insert_with(|| {
                    blocks.push((q, Vec::new()));
                    blocks.len() - 1
                });
                pair[x * m + y] = k;
            }
        }
        let mut position = vec![u32::MAX; d * d];
        let mut block = vec![u32::MAX; d * d];
        for a in 0..d {
            for b in 0..d {
                let k = pair[id[a] * m + id[b]];
                if k != usize::MAX {
                    let list = &mut blocks[k].1;
                    position[a * d + b] = list.len() as u32;
                    block[a * d + b] = k as u32;
                    list.push((a, b));
                }
            }
        }
        Liouville {
            blocks,
            position,
            block,
            d,
        }
    }
}

/// A block of the Lindblad superoperator, ready to exponentiate.
pub(super) struct LiouvilleBlock {
    /// Elements of the block (without the drift column).
    size: usize,
    /// The generator times `t`, shifted by its mean diagonal `shift` for
    /// coherences; for populations augmented by the column of the
    /// inhomogeneous term.
    pub generator: CMatrix,
    pub shift: C64,
    augmented: bool,
}

impl LiouvilleBlock {
    /// A block of `n` elements without their labels.
    pub(super) fn sized(n: usize, mut m: CMatrix, augmented: bool, t: f64) -> Self {
        let shift = if augmented {
            ZERO
        } else {
            let mean = (0..n).map(|i| m[(i, i)]).sum::<C64>() / n as f64;
            C64::new(0.0, mean.im)
        };
        for i in 0..n {
            m[(i, i)] -= shift;
        }
        LiouvilleBlock {
            size: n,
            generator: m * C64::new(t, 0.0),
            shift: shift * t,
            augmented,
        }
    }

    /// `x(t)` from `x(0)` given by `value(element)`: `P x` (plus the drift
    /// column for populations), times the shift's phase.
    pub(super) fn apply(&self, value: impl Fn(usize) -> C64) -> Vec<C64> {
        let p = expm(&self.generator);
        let n = self.size;
        let w = self.shift.exp();
        let x: Vec<C64> = (0..n).map(&value).collect();
        (0..n)
            .map(|i| {
                let mut v = if self.augmented { p[(i, n)] } else { ZERO };
                for (j, xj) in x.iter().enumerate() {
                    v += p[(i, j)] * xj;
                }
                v * w
            })
            .collect()
    }

    /// Floating-point operations of its exponential.
    pub(super) fn work(&self) -> f64 {
        let n = self.generator.nrows() as f64;
        (6.0 + f64::from(expm_squarings(one_norm(&self.generator)))) * 6.0 * n * n * n
    }
}

/// A method of propagation, forced in tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) enum Method {
    /// Dense exponentials of the blocks.
    Dense,
    /// Krylov steps on the blocks.
    Krylov,
    /// Dormand–Prince steps on the whole state.
    Steps,
    /// The blocks as planned, without the symmetry.
    Full,
}

/// How one block of the deviation propagates, with its estimated cost.
struct BlockPlan {
    k: usize,
    /// The block's deviation at the start.
    x: Vec<C64>,
    /// Subtracted from the diagonal and restored as `e^{shift·t}`.
    shift: C64,
    /// For the populations with a drift: the trace and the value of the
    /// augmented entry (the drift column is divided by it).
    augmented: Option<(C64, f64)>,
    /// The orbits of the elements and the permutations, when the block
    /// reduces.
    orbits: Option<(Orbits, Vec<Vec<u32>>)>,
    dense: bool,
    /// Taken as stationary below this fraction (the populations of one
    /// propagation; 0 otherwise).
    stationary: f64,
    work: f64,
    /// Peak bytes.
    memory: f64,
}

enum Propagator {
    /// `exp(tA)`.
    Dense(CMatrix),
    /// `A`, applied by Krylov steps over `t` to `tol` (and taken as
    /// stationary below `stationary`, see [`expv`]).
    Krylov {
        a: SparseBlock,
        t: f64,
        tol: f64,
        stationary: f64,
    },
}

/// A block's propagator over one interval, on the orbits when reduced.
struct Stepper {
    kind: Propagator,
    orbits: Option<Orbits>,
    /// The start vector (reduced and augmented).
    start: Vec<C64>,
    /// Elements of the block.
    n: usize,
    phase: C64,
}

impl Stepper {
    fn step(&self, y: &[C64]) -> Result<Vec<C64>> {
        let mut out = match &self.kind {
            Propagator::Dense(p) => (0..y.len())
                .map(|r| y.iter().enumerate().map(|(c, v)| p[(r, c)] * v).sum())
                .collect(),
            Propagator::Krylov {
                a,
                t,
                tol,
                stationary,
            } => expv(a, y, *t, *tol, *stationary)?,
        };
        if self.phase != ONE {
            for v in out.iter_mut() {
                *v *= self.phase;
            }
        }
        Ok(out)
    }

    /// The block's elements from a propagated vector.
    fn finish(&self, y: &[C64]) -> Vec<C64> {
        match &self.orbits {
            Some(o) => o.of[..self.n].iter().map(|&k| y[k as usize]).collect(),
            None => y[..self.n].to_vec(),
        }
    }

    /// `f` on the propagated vector: summed over each orbit when reduced.
    fn observable(&self, f: &[C64]) -> Vec<C64> {
        match &self.orbits {
            Some(o) => {
                let mut out = vec![ZERO; o.len()];
                for (v, &k) in f.iter().zip(&o.of) {
                    out[k as usize] += v;
                }
                out
            }
            None => f.to_vec(),
        }
    }
}

fn norm2(x: &[C64]) -> f64 {
    x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

impl SpinSystem {
    /// The distinct isotopes, in the order of [`Isotope::ALL`].
    pub(super) fn isotope_list(&self) -> Vec<Isotope> {
        Isotope::ALL
            .into_iter()
            .filter(|x| self.spins.iter().any(|s| s.isotope == *x))
            .collect()
    }

    /// The conserved `2M` of every basis state: per isotope (rotating
    /// frame) or of all spins (laboratory frame).
    pub(super) fn labels(&self) -> Vec<Vec<i32>> {
        let n = self.spins.len();
        let twice = |i: usize, s: usize| (2.0 * self.m(i, s)).round() as i32;
        let d = self.dimension();
        match self.frame {
            Frame::Laboratory => (0..d)
                .map(|s| vec![(0..n).map(|i| twice(i, s)).sum()])
                .collect(),
            Frame::Rotating => {
                let isotopes = self.isotope_list();
                (0..d)
                    .map(|s| {
                        isotopes
                            .iter()
                            .map(|x| {
                                (0..n)
                                    .filter(|&i| self.spins[i].isotope == *x)
                                    .map(|i| twice(i, s))
                                    .sum()
                            })
                            .collect()
                    })
                    .collect()
            }
        }
    }

    /// The change of the labels under `F₊` of an isotope.
    fn raise(&self, isotope: Isotope) -> Vec<i32> {
        match self.frame {
            Frame::Laboratory => vec![2],
            Frame::Rotating => self
                .isotope_list()
                .into_iter()
                .map(|x| if x == isotope { 2 } else { 0 })
                .collect(),
        }
    }

    /// The energy (rad/s) the rotating frame removes from the states of a
    /// label at field `b`, `Σ_X −γ_X b (1 + c_X·10⁻⁶) M_X` (0 in the
    /// laboratory frame).
    fn carrier_energy(&self, label: &[i32], b: f64) -> f64 {
        match self.frame {
            Frame::Laboratory => 0.0,
            Frame::Rotating => self
                .isotope_list()
                .into_iter()
                .zip(label)
                .map(|(x, &l)| -x.gamma() * b * (1.0 + self.carrier(x) * 1e-6) * f64::from(l) / 2.0)
                .sum(),
        }
    }

    /// The eigendecomposition of `h`, block by block when it conserves
    /// the labels (else in one block).
    pub(super) fn eigenblocks(&self, h: &CMatrix) -> Vec<Eigenblock> {
        let mut labels = self.labels();
        let d = h.nrows();
        let conserves = (0..d).all(|c| (0..d).all(|r| h[(r, c)] == ZERO || labels[r] == labels[c]));
        if !conserves {
            labels = vec![Vec::new(); d];
        }
        let mut groups: BTreeMap<Vec<i32>, Vec<usize>> = BTreeMap::new();
        for (s, l) in labels.into_iter().enumerate() {
            groups.entry(l).or_default().push(s);
        }
        groups
            .into_iter()
            .map(|(label, states)| {
                let n = states.len();
                let shift = states.iter().map(|&s| h[(s, s)].re).sum::<f64>() / n as f64;
                let mut sub = submatrix(h, &states, &states);
                for i in 0..n {
                    sub[(i, i)] -= shift;
                }
                let (values, vectors) = eigh(&sub);
                Eigenblock {
                    states,
                    label,
                    shift,
                    values,
                    vectors,
                }
            })
            .collect()
    }

    /// The state with the coherences between eigenstates of different
    /// energies removed. Energies are compared in the laboratory frame
    /// (the rotating frame adds the carriers back), equal within
    /// [`DEGENERATE`] of the blocks' spreads.
    pub(super) fn average(&self, rho: &CMatrix) -> CMatrix {
        let blocks = self.eigenblocks(&self.hamiltonian_at(self.field));
        let lab: Vec<f64> = blocks
            .iter()
            .map(|k| k.shift + self.carrier_energy(&k.label, self.field))
            .collect();
        let spread: Vec<f64> = blocks.iter().map(Eigenblock::spread).collect();
        let d = rho.nrows();
        let mut out = CMatrix::zeros(d, d);
        for (x, a) in blocks.iter().enumerate() {
            for (y, b) in blocks.iter().enumerate() {
                let (gap, tol) = if x == y {
                    (0.0, DEGENERATE * 2.0 * spread[x])
                } else {
                    let scale = spread[x] + spread[y] + lab[x].abs() + lab[y].abs();
                    (lab[x] - lab[y], DEGENERATE * scale)
                };
                // Every pair of the two blocks differs by at least this.
                if gap.abs() - spread[x] - spread[y] > tol {
                    continue;
                }
                let sub = submatrix(rho, &a.states, &b.states);
                if sub.iter().all(|z| *z == ZERO) {
                    continue;
                }
                let mut t = to_eigenbasis(&sub, a, b);
                for (r, &ea) in a.values.iter().enumerate() {
                    for (c, &eb) in b.values.iter().enumerate() {
                        if (gap + ea - eb).abs() > tol {
                            t[(r, c)] = ZERO;
                        }
                    }
                }
                let back = cmul(&cmul(&a.vectors, &t), &b.vectors.adjoint());
                scatter(&mut out, &back, &a.states, &b.states, C64::new(1.0, 0.0));
            }
        }
        out
    }

    /// The transitions of `F₊` of an isotope in `rho` under the free
    /// Hamiltonian: `(angular frequency, amplitude)` of
    /// `ρ_ab (F₊)_ba e^{i(E_b − E_a)t}`, ascending. Transitions at equal
    /// frequencies (within the rounding of the energies, at least 10⁻⁶ Hz)
    /// are merged, and those below [`LINE_FLOOR`] of the largest entry of
    /// `rho` or 10⁻¹⁴ of the strongest dropped.
    pub(super) fn transitions(&self, rho: &CMatrix, isotope: Isotope) -> Result<Vec<(f64, C64)>> {
        let fp = self.fplus(isotope)?;
        let blocks = self.eigenblocks(&self.hamiltonian_at(self.field));
        let raise = self.raise(isotope);
        let index: BTreeMap<&Vec<i32>, usize> = blocks
            .iter()
            .enumerate()
            .map(|(k, b)| (&b.label, k))
            .collect();
        let mut raw: Vec<(f64, C64)> = Vec::new();
        for a in &blocks {
            let up: Vec<i32> = a.label.iter().zip(&raise).map(|(x, y)| x + y).collect();
            let Some(&k) = index.get(&up) else { continue };
            let b = &blocks[k];
            let sub = submatrix(rho, &a.states, &b.states);
            if sub.iter().all(|z| *z == ZERO) {
                continue;
            }
            let r = to_eigenbasis(&sub, a, b);
            let f = to_eigenbasis(&submatrix(&fp, &b.states, &a.states), b, a);
            for (i, &ea) in a.values.iter().enumerate() {
                for (j, &eb) in b.values.iter().enumerate() {
                    let amp = r[(i, j)] * f[(j, i)];
                    if amp != ZERO {
                        raw.push((b.shift - a.shift + eb - ea, amp));
                    }
                }
            }
        }
        let energy = blocks
            .iter()
            .map(|k| k.shift.abs() + k.spread())
            .fold(0.0, f64::max);
        let tol = (2.0 * std::f64::consts::PI * 1e-6).max(DEGENERATE * energy);
        let merged = cluster(raw, tol);
        let strongest = merged.iter().map(|(_, a)| a.norm()).fold(0.0, f64::max);
        let floor = (LINE_FLOOR * max_entry(rho) * max_entry(&fp)).max(1e-14 * strongest);
        Ok(merged
            .into_iter()
            .filter(|(_, a)| a.norm() > floor)
            .collect())
    }

    /// `U ρ U†` with `U = exp(−iHt)` from the blocks of `H`.
    pub(super) fn unitary(&self, rho: &CMatrix, blocks: &[Eigenblock], t: f64) -> CMatrix {
        let us: Vec<CMatrix> = blocks
            .iter()
            .map(|k| {
                let mut scaled = k.vectors.clone();
                for (c, &e) in k.values.iter().enumerate() {
                    let w = C64::from_polar(1.0, -e * t);
                    for r in 0..scaled.nrows() {
                        scaled[(r, c)] *= w;
                    }
                }
                cmul(&scaled, &k.vectors.adjoint())
            })
            .collect();
        let d = rho.nrows();
        let mut out = CMatrix::zeros(d, d);
        for (x, a) in blocks.iter().enumerate() {
            for (y, b) in blocks.iter().enumerate() {
                let sub = submatrix(rho, &a.states, &b.states);
                if sub.iter().all(|z| *z == ZERO) {
                    continue;
                }
                let phase = C64::from_polar(1.0, -(a.shift - b.shift) * t);
                let moved = cmul(&cmul(&us[x], &sub), &us[y].adjoint());
                scatter(&mut out, &moved, &a.states, &b.states, phase);
            }
        }
        out
    }

    /// `L(1/d)` of the `T₁` jumps at field `b`, diagonal.
    fn jump_drift(&self, b: f64) -> Vec<f64> {
        // Σ_i (k↑ − k↓)[I₊, I₋]/d with [I₊, I₋] = 2I_z and k↑ − k↓ =
        // tanh(x/2)/T₁, written without the difference of the two rates,
        // which would lose the thermal polarisation's digits at low field.
        let d = self.dimension();
        let mut out = vec![0.0; d];
        for (i, spin) in self.spins().iter().enumerate() {
            let Some(t1) = spin.t1 else { continue };
            let excess = (self.boltzmann(i, b) / 2.0).tanh() / t1;
            for (s, o) in out.iter_mut().enumerate() {
                *o += 2.0 * excess * self.m(i, s) / d as f64;
            }
        }
        out
    }

    /// What relaxes the state at field `b`.
    pub(super) fn relaxation(&self, b: f64) -> Result<Relaxation> {
        let d = self.dimension();
        let jumps = self.jumps(b);
        let prepared = self.prepare(&jumps);
        let redfield = self.redfield(b)?;
        let mut drift = CMatrix::zeros(d, d);
        for (i, x) in self.jump_drift(b).into_iter().enumerate() {
            drift[(i, i)] = C64::new(x, 0.0);
        }
        let thermal = self.thermal_at(b);
        let mut equilibrium = thermal.clone();
        for i in 0..d {
            equilibrium[(i, i)] -= 1.0 / d as f64;
        }
        // −R(ρ − ρ_eq) = −R(Δ) + R(ρ_eq − 1/d), as R(1) = 0: without the
        // identity, whose image would be rounding of R's size.
        if let Some(r) = &redfield {
            r.image(&equilibrium, &mut drift, 1.0);
        }
        let rate = jumps.iter().map(|(_, a)| a.norm_squared()).sum::<f64>()
            + redfield.as_ref().map_or(0.0, RedfieldBlocks::rate);
        let equilibrium = equilibrium.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        Ok(Relaxation {
            jumps,
            prepared,
            redfield,
            drift,
            rate,
            equilibrium,
        })
    }

    fn local_jumps<'a>(&self, relaxation: &'a Relaxation) -> Vec<LocalJump<'a>> {
        relaxation
            .jumps
            .iter()
            .map(|(i, a)| LocalJump {
                stride: self.stride(*i),
                n: a.nrows(),
                op: a,
                ada: a.adjoint() * a,
            })
            .collect()
    }

    /// The entries of the generator of the master equation, `−i[H, ·]`,
    /// the Lindblad dissipators and `−R`, on the elements of `liouville`:
    /// `add(block, row, column, value)` with rows and columns indexed
    /// within the block.
    pub(super) fn generator_entries(
        &self,
        liouville: &Liouville,
        h: &Sparse,
        relaxation: &Relaxation,
        mut add: impl FnMut(usize, usize, usize, C64),
    ) {
        let d = self.dimension();
        let locals = self.local_jumps(relaxation);
        let at = |a: usize, b: usize| liouville.position[a * d + b] as usize;
        for (k, (_, elements)) in liouville.blocks.iter().enumerate() {
            for (j, &(a, b)) in elements.iter().enumerate() {
                coherent_column(a, b, h, &locals, |c, e, v| add(k, at(c, e), j, v));
            }
        }
        // −R, whose blocks lie inside these.
        if let Some(redfield) = &relaxation.redfield {
            for (q, (_, elements)) in redfield.liouville.blocks.iter().enumerate() {
                let (a, b) = elements[0];
                let k = liouville.block[a * d + b];
                if k == u32::MAX {
                    continue;
                }
                let targets = redfield.targets(q);
                let mut acc = Accumulator::new(elements.len());
                for (j, &(a, b)) in elements.iter().enumerate() {
                    redfield.column(q, j, &targets, -ONE, |c, e, v| {
                        acc.add(redfield.liouville.position_of(c, e), v)
                    });
                    let col = at(a, b);
                    for (i, v) in acc.take() {
                        let (c, e) = elements[i as usize];
                        add(k as usize, at(c, e), col, v);
                    }
                }
            }
        }
    }

    /// Entries (at most, repeated ones counted) of `−i[H, ·]` and the
    /// dissipators on block `k` of `liouville`, the mean of its diagonal,
    /// and the largest absolute off-diagonal row sum with the diagonal's
    /// largest distance from that mean: without building the block.
    fn block_estimate(
        &self,
        liouville: &Liouville,
        k: usize,
        h: &Sparse,
        relaxation: &Relaxation,
    ) -> (usize, C64, f64) {
        let elements = &liouville.blocks[k].1;
        let n = elements.len();
        let locals = self.local_jumps(relaxation);
        let mut diagonal = vec![ZERO; n];
        let mut rows = vec![0.0f64; n];
        let mut count = 0usize;
        for (j, &(a, b)) in elements.iter().enumerate() {
            coherent_column(a, b, h, &locals, |c, e, v| {
                count += 1;
                if (c, e) == (a, b) {
                    diagonal[j] += v;
                } else {
                    rows[liouville.position_of(c, e)] += v.norm();
                }
            });
        }
        let mean = diagonal.iter().sum::<C64>() / n.max(1) as f64;
        let norm = diagonal
            .iter()
            .zip(&rows)
            .map(|(z, r)| (z - mean).norm() + r)
            .fold(0.0, f64::max);
        (count, mean, norm)
    }

    /// Block `k` of the generator on `liouville` (as
    /// [`SpinSystem::generator_entries`]) as a sparse matrix, its diagonal
    /// less `shift`; with `drift`, augmented by a last column of the drift
    /// times `drift` (and an empty last row).
    fn sparse_block(
        &self,
        liouville: &Liouville,
        k: usize,
        h: &Sparse,
        relaxation: &Relaxation,
        shift: C64,
        drift: Option<C64>,
    ) -> SparseBlock {
        let elements = &liouville.blocks[k].1;
        let n = elements.len();
        let size = n + usize::from(drift.is_some());
        let locals = self.local_jumps(relaxation);
        let redfield = relaxation.redfield.as_ref();
        let targets: Vec<Vec<Option<u32>>> = redfield.map_or(Vec::new(), |r| {
            (0..r.liouville.blocks.len())
                .map(|q| r.targets(q))
                .collect()
        });
        let ranges = column_ranges(n);
        let parts: Vec<Vec<Vec<(u32, C64)>>> = map_tasks(ranges.len(), |c| {
            let mut acc = Accumulator::new(size);
            ranges[c]
                .clone()
                .map(|j| {
                    let (a, b) = elements[j];
                    coherent_column(a, b, h, &locals, |r, e, v| {
                        acc.add(liouville.position_of(r, e), v)
                    });
                    if let Some(rf) = redfield {
                        let q = rf.liouville.block_of(a, b).expect("every element");
                        let jq = rf.liouville.position_of(a, b);
                        rf.column(q, jq, &targets[q], -ONE, |r, e, v| {
                            acc.add(liouville.position_of(r, e), v)
                        });
                    }
                    if shift != ZERO {
                        acc.add(j, -shift);
                    }
                    acc.take()
                })
                .collect()
        });
        let mut columns: Vec<Vec<(u32, C64)>> = parts.into_iter().flatten().collect();
        if let Some(factor) = drift {
            columns.push(
                elements
                    .iter()
                    .enumerate()
                    .filter_map(|(j, &(a, b))| {
                        let v = relaxation.drift[(a, b)] * factor;
                        (v != ZERO).then_some((j as u32, v))
                    })
                    .collect(),
            );
        }
        SparseBlock::from_columns(size, &columns)
    }

    /// `‖A y‖` for the block [`SpinSystem::sparse_block`] would build,
    /// column by column without building it.
    #[allow(clippy::too_many_arguments)]
    fn probe(
        &self,
        liouville: &Liouville,
        k: usize,
        h: &Sparse,
        relaxation: &Relaxation,
        shift: C64,
        drift: Option<C64>,
        y: &[C64],
    ) -> f64 {
        let elements = &liouville.blocks[k].1;
        let n = elements.len();
        let locals = self.local_jumps(relaxation);
        let redfield = relaxation.redfield.as_ref();
        let targets: Vec<Vec<Option<u32>>> = redfield.map_or(Vec::new(), |r| {
            (0..r.liouville.blocks.len())
                .map(|q| r.targets(q))
                .collect()
        });
        let ranges = column_ranges(n);
        let parts: Vec<Vec<(u32, C64)>> = map_tasks(ranges.len(), |c| {
            let mut acc = Accumulator::new(n);
            for j in ranges[c].clone() {
                let v = y[j];
                if v == ZERO {
                    continue;
                }
                let (a, b) = elements[j];
                coherent_column(a, b, h, &locals, |r, e, w| {
                    acc.add(liouville.position_of(r, e), w * v)
                });
                if let Some(rf) = redfield {
                    let q = rf.liouville.block_of(a, b).expect("every element");
                    let jq = rf.liouville.position_of(a, b);
                    rf.column(q, jq, &targets[q], -v, |r, e, w| {
                        acc.add(liouville.position_of(r, e), w)
                    });
                }
                acc.add(j, -shift * v);
            }
            acc.take()
        });
        let mut out = vec![ZERO; n];
        for (i, v) in parts.into_iter().flatten() {
            out[i as usize] += v;
        }
        if let Some(factor) = drift {
            for (o, &(a, b)) in out.iter_mut().zip(elements) {
                *o += relaxation.drift[(a, b)] * factor * y[n];
            }
        }
        norm2(&out)
    }

    /// How block `k` of `liouville` with the deviation `x` propagates over
    /// `t`, `repeats` times: its work and memory estimated without building
    /// it, densely or by Krylov steps (whichever costs less), on the orbits
    /// of the symmetry `maps` when the block and `x` are invariant. `trace`
    /// marks the populations' block with the drift (times the trace).
    #[allow(clippy::too_many_arguments)]
    fn plan_block(
        &self,
        liouville: &Liouville,
        k: usize,
        h: &Sparse,
        relaxation: &Relaxation,
        t: f64,
        repeats: usize,
        (x, scale, level): (Vec<C64>, f64, f64),
        trace: Option<C64>,
        maps: Option<&[Vec<usize>]>,
    ) -> BlockPlan {
        let elements = &liouville.blocks[k].1;
        let n = elements.len();
        let (count, mean, spread) = self.block_estimate(liouville, k, h, relaxation);
        let (redfield, redfield_build) = relaxation.redfield_entries(liouville, Some(k));
        // Coherences shift by their mean precession; the populations' block
        // keeps its diagonal, as shifting the decay would let e^{−rt}
        // underflow.
        let shift = if trace.is_some() {
            ZERO
        } else {
            C64::new(0.0, mean.im)
        };
        let norm = spread
            + (mean - shift).norm()
            + relaxation
                .redfield
                .as_ref()
                .map_or(0.0, RedfieldBlocks::rate);
        // The drift enters as a last column against an entry of the size of
        // the deviation or of equilibrium's, whichever is larger, so the
        // tolerance stays relative to it (the drift's integral is bounded
        // by equilibrium's deviation).
        let augmented = trace.and_then(|trace| {
            let mut s = norm2(&x).max(relaxation.equilibrium * trace.norm());
            if s == 0.0 {
                let drift = relaxation.drift.iter().map(|z| z.norm_sqr()).sum::<f64>();
                s = drift.sqrt() * trace.norm() * t;
            }
            (s > 0.0).then_some((trace, s))
        });
        let size = n + usize::from(augmented.is_some());
        let orbits = maps.and_then(|maps| {
            let images: Vec<Vec<u32>> = maps
                .iter()
                .map(|m| {
                    (0..size)
                        .map(|j| match elements.get(j) {
                            Some(&(a, b)) => liouville.position_of(m[a], m[b]) as u32,
                            None => j as u32,
                        })
                        .collect()
                })
                .collect();
            let orbits = Orbits::new(size, &images);
            let mut start = x.clone();
            start.extend(augmented.map(|(_, s)| C64::new(s, 0.0)));
            // Invariant to rounding of the whole deviation: asymmetric
            // rounding in a nearly empty block does not count.
            (orbits.len() < size && orbits.asymmetry(&start) <= SYMMETRY_TOLERANCE * scale)
                .then_some((orbits, images))
        });
        let nnz = count + redfield as usize + n;
        let (m, entries) = match &orbits {
            Some((o, _)) => (o.len(), nnz * o.len() / size + o.len()),
            None => (size, nnz),
        };
        let mf = m as f64;
        let dense_work = if m <= super::MAX_LIOUVILLE_BLOCK {
            (6.0 + f64::from(expm_squarings(norm * t))) * 6.0 * mf * mf * mf
                + repeats as f64 * 8.0 * mf * mf
        } else {
            f64::INFINITY
        };
        let mut krylov = repeats as f64 * krylov_work(m, entries, norm, t);
        let build_memory = 44.0 * nnz as f64
            + accumulator_bytes(size)
            + if orbits.is_some() {
                44.0 * entries as f64
            } else {
                0.0
            };
        // The populations (equilibrium) may not move at all: then the
        // Krylov process ends at once, in one step, not the a priori ones.
        // They count as stationary when they move by less than
        // KRYLOV_STATIONARY over the interval, or no faster than the
        // rounding of the state explains: forming the deviation from ρ
        // leaves `ε·Tr(ρ)/d` on each population, which relaxation and the
        // couplings move at most at `rate + 2‖H_off‖` (with a factor 2).
        // Checked by one pass over the columns, and only where a block can
        // be built at all. On orbits the vector's norm differs by up to the
        // root of the ratio of their sizes.
        let populations = repeats <= 1 && liouville.blocks[k].0.iter().all(|&q| q == 0);
        let mut stationary = 0.0;
        if populations {
            let mut y = x.clone();
            y.extend(augmented.map(|(_, s)| C64::new(s, 0.0)));
            let size_y = norm2(&y);
            let diagonal = elements.iter().filter(|&&(a, b)| a == b).count() as f64;
            let rounding = f64::EPSILON * level * diagonal.sqrt();
            let speed = relaxation.rate + 2.0 * h.off_diagonal();
            if size_y > 0.0 && t > 0.0 {
                stationary = KRYLOV_STATIONARY.max(2.0 * speed * rounding * t / size_y);
            }
            if krylov > 1e9
                && stationary > 0.0
                && build_memory + krylov_bytes(m) <= super::MAX_RELAXATION_MEMORY
            {
                let factor = augmented.map(|(trace, s)| trace / s);
                let moving = self.probe(liouville, k, h, relaxation, shift, factor, &y);
                let spread = orbits.as_ref().map_or(1.0, |(o, _)| {
                    let (lo, hi) = o
                        .sizes
                        .iter()
                        .fold((u32::MAX, 0), |(lo, hi), &s| (lo.min(s), hi.max(s)));
                    (f64::from(hi) / f64::from(lo)).sqrt()
                });
                if moving * t * spread <= 0.5 * stationary * size_y {
                    krylov = krylov_work(m, entries, 0.0, t);
                }
            }
        }
        let dense = dense_work <= krylov;
        let build = 50.0 * count as f64
            + redfield_build
            + if orbits.is_some() {
                30.0 * nnz as f64
            } else {
                0.0
            };
        let memory = build_memory
            + if dense {
                160.0 * mf * mf
            } else {
                krylov_bytes(m)
            };
        BlockPlan {
            k,
            x,
            shift,
            augmented,
            orbits,
            dense,
            stationary,
            work: build + dense_work.min(krylov),
            memory,
        }
    }

    /// The propagator over `t` of a planned block, to the Krylov tolerance
    /// spread over `repeats` steps.
    fn stepper(
        &self,
        plan: &BlockPlan,
        liouville: &Liouville,
        h: &Sparse,
        relaxation: &Relaxation,
        t: f64,
        repeats: usize,
    ) -> Result<Stepper> {
        let drift = plan.augmented.map(|(trace, s)| trace / s);
        let mut a = self.sparse_block(liouville, plan.k, h, relaxation, plan.shift, drift);
        let mut orbits = None;
        if let Some((o, images)) = &plan.orbits {
            // Every entry equal to its image, as the declared symmetry
            // promises; else the full block.
            if Orbits::invariant(&a, images, SYMMETRY_TOLERANCE * a.largest()) {
                a = o.reduce(&a);
                orbits = Some(o.clone());
            }
        }
        let mut start = plan.x.clone();
        start.extend(plan.augmented.map(|(_, s)| C64::new(s, 0.0)));
        let start = match &orbits {
            Some(o) => o.average(&start),
            None => start,
        };
        let kind = if plan.dense && a.n <= super::MAX_LIOUVILLE_BLOCK {
            Propagator::Dense(expm(&(a.to_dense() * C64::new(t, 0.0))))
        } else {
            let tol = KRYLOV_TOLERANCE * norm2(&start) / repeats.max(1) as f64;
            Propagator::Krylov {
                a,
                t,
                tol,
                stationary: plan.stationary,
            }
        };
        Ok(Stepper {
            kind,
            orbits,
            start,
            n: plan.x.len(),
            phase: (plan.shift * t).exp(),
        })
    }

    /// Floating-point operations Dormand–Prince steps over `t` would take,
    /// weighted by [`DP_PENALTY`], with `redfield` entries of the Redfield
    /// blocks applied at each evaluation.
    fn dp_work(&self, h: &Sparse, relaxation: &Relaxation, t: f64, redfield: f64) -> f64 {
        let d = self.dimension() as f64;
        let steps = ((h.spread() + relaxation.rate) * t / DP_RADIANS_PER_STEP).ceil() + 1.0;
        let jump_work: f64 = relaxation
            .jumps
            .iter()
            .map(|(_, a)| 32.0 * a.nrows() as f64 * d * d)
            .sum();
        let evaluation = 16.0 * h.entries() as f64 * d + jump_work + 8.0 * redfield + 60.0 * d * d;
        DP_PENALTY * 6.0 * steps * evaluation
    }

    /// Bytes Dormand–Prince steps hold: the stages and, kept in the
    /// relaxation, the Redfield blocks they apply.
    fn dp_memory(&self, relaxation: &Relaxation, redfield: f64) -> f64 {
        let d = self.dimension() as f64;
        let kept = relaxation
            .redfield
            .as_ref()
            .map_or(0, RedfieldBlocks::bytes) as f64;
        16.0 * 16.0 * d * d + 44.0 * redfield + kept
    }

    fn too_much(&self, what: &str, t: f64, work: f64) -> QsoError {
        QsoError::invalid(
            "relaxation",
            format!(
                "{what} over {t} s with relaxation at dimension {} needs about {work:.1e} operations \
                 (limit {:.1e}); use the rotating frame, fewer spins, shorter times or no relaxation",
                self.dimension(),
                super::MAX_RELAXATION_WORK
            ),
        )
    }

    fn too_large(&self, what: &str, memory: f64) -> QsoError {
        QsoError::invalid(
            "relaxation",
            format!(
                "{what} with relaxation at dimension {} needs about {:.1} GB (limit {:.1} GB); \
                 use the rotating frame, a symmetry, fewer spins or no relaxation",
                self.dimension(),
                memory / 1e9,
                super::MAX_RELAXATION_MEMORY / 1e9
            ),
        )
    }

    /// `ρ(t)` under the master equation with Hamiltonian `h` and the
    /// relaxation at field `b` (tests): block exponentials when `h` conserves the
    /// labels (`conserving`) and they are cheaper, else Dormand–Prince
    /// steps.
    #[cfg(test)]
    pub(super) fn relax(
        &self,
        rho: &CMatrix,
        h: &CMatrix,
        t: f64,
        b: f64,
        conserving: bool,
    ) -> Result<CMatrix> {
        self.relax_with(rho, h, t, conserving, &self.relaxation(b)?)
    }

    /// [`SpinSystem::relax`] with the relaxation at the field given: each
    /// block of the deviation that moves (the coherence orders of the labels
    /// when `h` conserves them, else one block) propagates densely or by
    /// Krylov steps, reduced by the symmetry when it applies, unless
    /// Dormand–Prince steps on the whole state cost less.
    pub(super) fn relax_with(
        &self,
        rho: &CMatrix,
        h: &CMatrix,
        t: f64,
        conserving: bool,
        relaxation: &Relaxation,
    ) -> Result<CMatrix> {
        self.relax_planned(rho, h, t, conserving, relaxation, None)
    }

    /// [`SpinSystem::relax_with`], with the method forced in tests.
    pub(super) fn relax_planned(
        &self,
        rho: &CMatrix,
        h: &CMatrix,
        t: f64,
        conserving: bool,
        relaxation: &Relaxation,
        force: Option<Method>,
    ) -> Result<CMatrix> {
        let d = self.dimension();
        let trace = rho.trace();
        let level = trace / d as f64;
        let mut delta = rho.clone();
        for i in 0..d {
            delta[(i, i)] -= level;
        }
        let sparse = Sparse::new(h);
        let labels = if conserving {
            self.labels()
        } else {
            vec![Vec::new(); d]
        };
        let liouville = Liouville::new(&labels, None);
        let maps = match force {
            Some(Method::Full) => None,
            _ => self.symmetry_maps(h),
        };
        let mut plans = self.plans(&delta, trace, &liouville, &sparse, relaxation, t, maps);
        for plan in &mut plans {
            match force {
                Some(Method::Dense) => plan.dense = true,
                Some(Method::Krylov) => plan.dense = false,
                _ => {}
            }
        }
        let block_work: f64 = plans.iter().map(|p| p.work).sum();
        // The largest block at a time, next to what stays held: the index
        // of the elements and the Redfield terms.
        let held = self.held_bytes(relaxation);
        let block_memory = plans.iter().map(|p| p.memory).fold(0.0, f64::max) + held;
        // Dormand–Prince steps apply (and keep) every Redfield block the
        // state reaches: those it occupies and, with a drift, the
        // populations', or all of them when the orders mix.
        let drifting = trace != ZERO && relaxation.drift.iter().any(|z| *z != ZERO);
        let (redfield, redfield_build) =
            self.redfield_touched(relaxation, &delta, !conserving, drifting);
        let dp_work = self.dp_work(&sparse, relaxation, t, redfield) + redfield_build;
        let dp_memory = self.dp_memory(relaxation, redfield) + held;
        let limit = super::MAX_RELAXATION_MEMORY;
        let use_blocks = match force {
            Some(Method::Steps) => false,
            Some(_) => true,
            None => {
                if block_memory > limit && dp_memory > limit {
                    return Err(self.too_large("a delay", block_memory.min(dp_memory)));
                }
                block_memory <= limit && (dp_memory > limit || block_work <= dp_work)
            }
        };
        let work = if use_blocks { block_work } else { dp_work };
        if force.is_none() && work > super::MAX_RELAXATION_WORK {
            return Err(self.too_much("a delay", t, work));
        }
        let mut next = if use_blocks {
            let mut next = CMatrix::zeros(d, d);
            for plan in &plans {
                let stepper = self.stepper(plan, &liouville, &sparse, relaxation, t, 1)?;
                let moved = stepper.finish(&stepper.step(&stepper.start)?);
                for (&(a, b), v) in liouville.blocks[plan.k].1.iter().zip(moved) {
                    next[(a, b)] = v;
                }
            }
            next
        } else {
            let c = &relaxation.drift * trace;
            self.dormand_prince(&delta, &sparse, relaxation, &c, t)?
        };
        for i in 0..d {
            next[(i, i)] += level;
        }
        Ok(next)
    }

    /// The plans of the blocks of `liouville` that move: a nonzero
    /// deviation, or the populations with a drift.
    #[allow(clippy::too_many_arguments)]
    fn plans(
        &self,
        delta: &CMatrix,
        trace: C64,
        liouville: &Liouville,
        h: &Sparse,
        relaxation: &Relaxation,
        t: f64,
        maps: Option<Vec<Vec<usize>>>,
    ) -> Vec<BlockPlan> {
        let drifting = trace != ZERO && relaxation.drift.iter().any(|z| *z != ZERO);
        let scale = max_entry(delta);
        let mut plans = Vec::new();
        for (k, (order, elements)) in liouville.blocks.iter().enumerate() {
            let x: Vec<C64> = elements.iter().map(|&(a, b)| delta[(a, b)]).collect();
            let drift = (drifting && order.iter().all(|&q| q == 0)).then_some(trace);
            // A block holding nothing but rounding (the multiple-quantum
            // orders of a pulsed thermal state, 10⁻²⁵ of it) stays empty.
            let largest = x.iter().fold(0.0f64, |m, z| m.max(z.norm()));
            if drift.is_none() && largest <= NEGLIGIBLE * scale {
                continue;
            }
            plans.push(self.plan_block(
                liouville,
                k,
                h,
                relaxation,
                t,
                1,
                (x, scale, trace.norm() / self.dimension() as f64),
                drift,
                maps.as_deref(),
            ));
        }
        plans
    }

    /// For the blocks that move under the free Hamiltonian from `rho`:
    /// their elements, and how many coefficients propagate (fewer on the
    /// orbits of a symmetry).
    #[cfg(test)]
    pub(super) fn planned_sizes(&self, rho: &CMatrix) -> Result<Vec<(usize, usize)>> {
        let d = self.dimension();
        let h = self.hamiltonian_at(self.field);
        let relaxation = self.relaxation(self.field)?;
        let trace = rho.trace();
        let mut delta = rho.clone();
        for i in 0..d {
            delta[(i, i)] -= trace / d as f64;
        }
        let liouville = Liouville::new(&self.labels(), None);
        let maps = self.symmetry_maps(&h);
        let plans = self.plans(
            &delta,
            trace,
            &liouville,
            &Sparse::new(&h),
            &relaxation,
            1e-3,
            maps,
        );
        Ok(plans
            .iter()
            .map(|p| {
                let size = p.x.len() + usize::from(p.augmented.is_some());
                (size, p.orbits.as_ref().map_or(size, |(o, _)| o.len()))
            })
            .collect())
    }

    /// Entries (at most) and build work of the Redfield blocks on which `x`
    /// is nonzero.
    fn redfield_touched(
        &self,
        relaxation: &Relaxation,
        x: &CMatrix,
        all: bool,
        drifting: bool,
    ) -> (f64, f64) {
        let Some(r) = &relaxation.redfield else {
            return (0.0, 0.0);
        };
        let (mut entries, mut build) = (0.0, 0.0);
        for (q, (order, elements)) in r.liouville.blocks.iter().enumerate() {
            let reached = all
                || (drifting && order.iter().all(|&o| o == 0))
                || elements.iter().any(|&(a, b)| x[(a, b)] != ZERO);
            if reached {
                entries += r.count(q) as f64;
                if !r.is_built(q) {
                    build += 16.0 * r.terms() as f64 * r.count(q) as f64;
                }
            }
        }
        (entries, build)
    }

    /// Bytes held through a propagation: the index of the elements and the
    /// Redfield terms.
    fn held_bytes(&self, relaxation: &Relaxation) -> f64 {
        let d = self.dimension() as f64;
        24.0 * d * d
            + relaxation
                .redfield
                .as_ref()
                .map_or(0, RedfieldBlocks::terms_bytes) as f64
    }

    /// The free induction decay with relaxation: `Tr(ρ(k·dwell) F₊)`. Only
    /// the coherences of order `−1` of the isotope contribute, and they
    /// evolve on their own: the block's propagator over the dwell time
    /// (dense or by Krylov steps, reduced by the symmetry when it applies),
    /// else Dormand–Prince steps between the points.
    pub(super) fn relaxing_fid(
        &self,
        rho: &CMatrix,
        isotope: Isotope,
        dwell: f64,
        points: usize,
    ) -> Result<Vec<C64>> {
        self.relaxing_fid_planned(rho, isotope, dwell, points, None)
    }

    /// [`SpinSystem::relaxing_fid`], with the method forced in tests.
    pub(super) fn relaxing_fid_planned(
        &self,
        rho: &CMatrix,
        isotope: Isotope,
        dwell: f64,
        points: usize,
        force: Option<Method>,
    ) -> Result<Vec<C64>> {
        let d = self.dimension();
        let fp = self.fplus(isotope)?;
        if points == 1 {
            return Ok(vec![self.expectation(rho, &fp)?]);
        }
        let h = self.hamiltonian_at(self.field);
        let relaxation = self.relaxation(self.field)?;
        let sparse = Sparse::new(&h);
        let labels = self.labels();
        let order: Vec<i32> = self.raise(isotope).iter().map(|x| -x).collect();
        let liouville = Liouville::new(&labels, Some(&order));
        let maps = match force {
            Some(Method::Full) => None,
            _ => self.symmetry_maps(&h),
        };
        let repeats = points - 1;
        // F₊ is traceless: the deviation carries the signal.
        let trace = rho.trace();
        let mut delta = rho.clone();
        for i in 0..d {
            delta[(i, i)] -= trace / d as f64;
        }
        let plan = (!liouville.blocks.is_empty()).then(|| {
            let x = liouville.blocks[0]
                .1
                .iter()
                .map(|&(a, b)| rho[(a, b)])
                .collect();
            let mut plan = self.plan_block(
                &liouville,
                0,
                &sparse,
                &relaxation,
                dwell,
                repeats,
                (x, max_entry(&delta), trace.norm() / d as f64),
                None,
                maps.as_deref(),
            );
            match force {
                Some(Method::Dense) => plan.dense = true,
                Some(Method::Krylov) => plan.dense = false,
                Some(Method::Steps) => plan.memory = f64::INFINITY,
                _ => {}
            }
            plan
        });
        let drifting = trace != ZERO && relaxation.drift.iter().any(|z| *z != ZERO);
        let (redfield, redfield_build) =
            self.redfield_touched(&relaxation, &delta, false, drifting);
        let dp_work =
            repeats as f64 * self.dp_work(&sparse, &relaxation, dwell, redfield) + redfield_build;
        let held = self.held_bytes(&relaxation);
        let dp_memory = self.dp_memory(&relaxation, redfield) + held;
        let limit = super::MAX_RELAXATION_MEMORY;
        let fid_time = dwell * points as f64;
        if let Some(plan) = plan.filter(|p| {
            p.memory + held <= limit && (dp_memory > limit || p.work <= dp_work || force.is_some())
        }) {
            if force.is_none() && plan.work > super::MAX_RELAXATION_WORK {
                return Err(self.too_much("a free induction decay", fid_time, plan.work));
            }
            let stepper = self.stepper(&plan, &liouville, &sparse, &relaxation, dwell, repeats)?;
            let f: Vec<C64> = liouville.blocks[0]
                .1
                .iter()
                .map(|&(a, b)| fp[(b, a)])
                .collect();
            let f = stepper.observable(&f);
            let mut y = stepper.start.clone();
            let mut out = Vec::with_capacity(points);
            for k in 0..points {
                if k > 0 {
                    y = stepper.step(&y)?;
                }
                out.push(y.iter().zip(&f).map(|(a, b)| a * b).sum());
            }
            return Ok(out);
        }
        if dp_memory > limit {
            return Err(self.too_large("a free induction decay", dp_memory));
        }
        if force.is_none() && dp_work > super::MAX_RELAXATION_WORK {
            return Err(self.too_much("a free induction decay", fid_time, dp_work));
        }
        let c = &relaxation.drift * trace;
        let mut out = Vec::with_capacity(points);
        for k in 0..points {
            if k > 0 {
                delta = self.dormand_prince(&delta, &sparse, &relaxation, &c, dwell)?;
            }
            out.push(self.expectation(&delta, &fp)?);
        }
        Ok(out)
    }

    fn prepare(&self, jumps: &[(usize, CMatrix)]) -> Vec<Jump> {
        let d = self.dimension();
        jumps
            .iter()
            .map(|(i, a)| {
                let (stride, k) = (self.stride(*i), a.nrows());
                let mut members = vec![Vec::new(); k];
                for s in 0..d {
                    members[(s / stride) % k].push(s);
                }
                Jump {
                    stride,
                    members,
                    op: nonzeros(a),
                    ada: nonzeros(&(a.adjoint() * a)),
                }
            })
            .collect()
    }

    /// `L(x) + c` with `L` the generator of the master equation, `H`
    /// sparse.
    fn lindblad(&self, x: &CMatrix, h: &Sparse, relaxation: &Relaxation, c: &CMatrix) -> CMatrix {
        let d = x.nrows();
        let mut out = c.clone();
        let mi = C64::new(0.0, -1.0);
        // −i H x, column by column.
        for k in 0..d {
            let col = x.column(k);
            for (r, row) in h.rows.iter().enumerate() {
                let mut s = ZERO;
                for &(c, v) in row {
                    s += v * col[c];
                }
                out[(r, k)] += mi * s;
            }
        }
        // +i x H: column c of the product is Σ_r H_rc x[:, r].
        for (c, entries) in h.cols.iter().enumerate() {
            for &(r, v) in entries {
                let w = -mi * v;
                for k in 0..d {
                    let add = x[(k, r)] * w;
                    out[(k, c)] += add;
                }
            }
        }
        relaxation.dissipate_into(&mut out, x);
        out
    }

    /// Adaptive Dormand–Prince 5(4) steps of `dx/dt = L(x) + c` over `t`,
    /// the local error of each step below [`DP_TOLERANCE`] of the largest
    /// entry of `x`.
    pub(super) fn dormand_prince(
        &self,
        x: &CMatrix,
        h: &Sparse,
        relaxation: &Relaxation,
        c: &CMatrix,
        t: f64,
    ) -> Result<CMatrix> {
        const A: [[f64; 6]; 6] = [
            [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
            [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
            [
                19372.0 / 6561.0,
                -25360.0 / 2187.0,
                64448.0 / 6561.0,
                -212.0 / 729.0,
                0.0,
                0.0,
            ],
            [
                9017.0 / 3168.0,
                -355.0 / 33.0,
                46732.0 / 5247.0,
                49.0 / 176.0,
                -5103.0 / 18656.0,
                0.0,
            ],
            [
                35.0 / 384.0,
                0.0,
                500.0 / 1113.0,
                125.0 / 192.0,
                -2187.0 / 6784.0,
                11.0 / 84.0,
            ],
        ];
        // Fifth-order weights are the last row of A; the embedded fourth
        // order differs by E.
        const E: [f64; 7] = [
            71.0 / 57600.0,
            0.0,
            -71.0 / 16695.0,
            71.0 / 1920.0,
            -17253.0 / 339200.0,
            22.0 / 525.0,
            -1.0 / 40.0,
        ];
        let omega = h.spread() + relaxation.rate;
        let mut dt = (DP_RADIANS_PER_STEP / omega.max(1e-300)).min(t);
        let mut now = 0.0;
        let mut x = x.clone();
        let mut k1 = self.lindblad(&x, h, relaxation, c);
        let mut steps = 0usize;
        while now < t {
            steps += 1;
            if steps > 100_000_000 {
                return Err(QsoError::NoConvergence {
                    method: "Dormand–Prince",
                    iterations: steps,
                    residual: t - now,
                });
            }
            let step = dt.min(t - now);
            // y += f·k without temporaries.
            let axpy = |y: &mut CMatrix, f: f64, k: &CMatrix| {
                for (a, b) in y.iter_mut().zip(k.iter()) {
                    *a += b * f;
                }
            };
            let mut k = vec![k1.clone()];
            for row in A.iter() {
                let mut y = x.clone();
                for (j, kj) in k.iter().enumerate() {
                    if row[j] != 0.0 {
                        axpy(&mut y, step * row[j], kj);
                    }
                }
                k.push(self.lindblad(&y, h, relaxation, c));
            }
            // k[6] is the derivative at the fifth-order solution (FSAL).
            let mut next = x.clone();
            for (j, kj) in k.iter().take(6).enumerate() {
                if A[5][j] != 0.0 {
                    axpy(&mut next, step * A[5][j], kj);
                }
            }
            let mut err = CMatrix::zeros(x.nrows(), x.ncols());
            for (j, kj) in k.iter().enumerate() {
                if E[j] != 0.0 {
                    axpy(&mut err, step * E[j], kj);
                }
            }
            let scale = max_entry(&x).max(max_entry(&next));
            let e = max_entry(&err) / (DP_TOLERANCE * scale).max(f64::MIN_POSITIVE);
            if e <= 1.0 || step <= 1e-14 * t.max(1.0) {
                x = next;
                now += step;
                k1 = k.pop().expect("seven stages");
                let grow = if e > 0.0 { 0.9 * e.powf(-0.2) } else { 5.0 };
                dt = step * grow.clamp(0.2, 5.0);
            } else {
                dt = step * (0.9 * e.powf(-0.2)).clamp(0.1, 0.9);
            }
        }
        Ok(x)
    }
}

/// Merge transitions whose frequencies follow each other within `tol`
/// (single linkage, so the result does not depend on the order), at the
/// amplitude-weighted mean frequency.
fn cluster(mut raw: Vec<(f64, C64)>, tol: f64) -> Vec<(f64, C64)> {
    raw.sort_by(|x, y| x.0.total_cmp(&y.0));
    // (amplitude, Σ|a|·f, Σ|a|, last frequency).
    let mut groups: Vec<(C64, f64, f64, f64)> = Vec::new();
    for (f, a) in raw {
        match groups.last_mut() {
            Some(g) if f - g.3 <= tol => {
                g.0 += a;
                g.1 += a.norm() * f;
                g.2 += a.norm();
                g.3 = f;
            }
            _ => groups.push((a, a.norm() * f, a.norm(), f)),
        }
    }
    groups
        .into_iter()
        .map(|(a, wf, w, last)| (if w > 0.0 { wf / w } else { last }, a))
        .collect()
}
