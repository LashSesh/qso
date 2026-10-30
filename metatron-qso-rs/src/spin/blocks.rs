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

use super::redfield::RedfieldBlocks;
use super::{Frame, Isotope, ONE, SpinSystem, ZERO};
use crate::error::{QsoError, Result};
use crate::linalg::dense::{cmul, expm, expm_squarings, one_norm};
use crate::linalg::{CMatrix, eigh};

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

/// Nonzero entries of a matrix by row and by column.
pub(super) struct Sparse {
    rows: Vec<Vec<(usize, C64)>>,
    cols: Vec<Vec<(usize, C64)>>,
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
        Sparse { rows, cols }
    }

    fn entries(&self) -> usize {
        self.rows.iter().map(Vec::len).sum()
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
    rate: f64,
}

impl Relaxation {
    /// `out += D(x)`: the dissipators and `−R(x)`, without the drift.
    fn dissipate_into(&self, out: &mut CMatrix, x: &CMatrix) {
        for jump in &self.prepared {
            jump.dissipate(out, x);
        }
        if let Some(r) = &self.redfield {
            r.apply(x, out, -1.0);
        }
    }

    /// `D(x)`.
    pub(super) fn dissipate(&self, x: &CMatrix) -> CMatrix {
        let mut out = CMatrix::zeros(x.nrows(), x.ncols());
        self.dissipate_into(&mut out, x);
        out
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

/// `Σ n²` over the coherence-order blocks.
pub(super) fn coherence_entries(labels: &[Vec<i32>]) -> usize {
    coherence_sizes(labels).values().map(|&n| n * n).sum()
}

/// Elements of the largest coherence-order block.
pub(super) fn largest_coherence_block(labels: &[Vec<i32>]) -> usize {
    coherence_sizes(labels).values().copied().max().unwrap_or(0)
}

fn difference(x: &[i32], y: &[i32]) -> Vec<i32> {
    x.iter().zip(y).map(|(a, b)| a - b).collect()
}

/// Distinct labels, and the index of each state's label among them.
fn distinct(labels: &[Vec<i32>]) -> (Vec<Vec<i32>>, Vec<usize>) {
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

/// Size of every coherence-order block, without forming them.
fn coherence_sizes(labels: &[Vec<i32>]) -> BTreeMap<Vec<i32>, usize> {
    let (list, id) = distinct(labels);
    let mut count = vec![0usize; list.len()];
    for &k in &id {
        count[k] += 1;
    }
    let mut sizes: BTreeMap<Vec<i32>, usize> = BTreeMap::new();
    for (x, lx) in list.iter().enumerate() {
        for (y, ly) in list.iter().enumerate() {
            *sizes.entry(difference(lx, ly)).or_default() += count[x] * count[y];
        }
    }
    sizes
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
    pub elements: Vec<(usize, usize)>,
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
    /// A block from its generator `m` (augmented by a drift column for
    /// populations), scaled by `t` and, for coherences, shifted by the mean
    /// precession: a large one in the laboratory frame would cost
    /// precision, while shifting the mean decay would let `e^{−rt}`
    /// underflow as the exponential of the rest overflows.
    pub(super) fn new(elements: Vec<(usize, usize)>, m: CMatrix, augmented: bool, t: f64) -> Self {
        let n = elements.len();
        let mut block = Self::sized(n, m, augmented, t);
        block.elements = elements;
        block
    }

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
            elements: Vec::new(),
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

    /// `(A A† − A†A)/d` of the jumps: `L(1/d)`, diagonal.
    fn jump_drift(&self, jumps: &[(usize, CMatrix)]) -> Vec<f64> {
        let d = self.dimension();
        let mut out = vec![0.0; d];
        for (i, a) in jumps {
            let k = a.nrows();
            let stride = self.stride(*i);
            let comm = a * a.adjoint() - a.adjoint() * a;
            for (s, o) in out.iter_mut().enumerate() {
                let ks = (s / stride) % k;
                *o += comm[(ks, ks)].re / d as f64;
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
        for (i, x) in self.jump_drift(&jumps).into_iter().enumerate() {
            drift[(i, i)] = C64::new(x, 0.0);
        }
        // −R(ρ − ρ_eq) = −R(Δ) + R(ρ_eq), as R(1) = 0.
        if let Some(r) = &redfield {
            r.apply(&self.thermal_at(b), &mut drift, 1.0);
        }
        let rate = jumps.iter().map(|(_, a)| a.norm_squared()).sum::<f64>()
            + redfield.as_ref().map_or(0.0, RedfieldBlocks::rate);
        Ok(Relaxation {
            jumps,
            prepared,
            redfield,
            drift,
            rate,
        })
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
        let local: Vec<(usize, usize, &CMatrix, CMatrix)> = relaxation
            .jumps
            .iter()
            .map(|(i, a)| (self.stride(*i), a.nrows(), a, a.adjoint() * a))
            .collect();
        let mi = C64::new(0.0, -1.0);
        let at = |a: usize, b: usize| liouville.position[a * d + b] as usize;
        for (k, (_, elements)) in liouville.blocks.iter().enumerate() {
            for (j, &(a, b)) in elements.iter().enumerate() {
                // −i(H|a⟩⟨b| − |a⟩⟨b|H).
                for &(c, v) in &h.cols[a] {
                    add(k, at(c, b), j, mi * v);
                }
                for &(c, v) in &h.rows[b] {
                    add(k, at(a, c), j, -mi * v);
                }
                // A|a⟩⟨b|A† − ½A†A|a⟩⟨b| − ½|a⟩⟨b|A†A on one spin.
                for &(stride, n, op, ref ada) in &local {
                    let (ka, kb) = ((a / stride) % n, (b / stride) % n);
                    for r in 0..n {
                        let x = op[(r, ka)];
                        if x == ZERO {
                            continue;
                        }
                        for s in 0..n {
                            let y = op[(s, kb)].conj();
                            if y != ZERO {
                                let (c, e) =
                                    (a + r * stride - ka * stride, b + s * stride - kb * stride);
                                add(k, at(c, e), j, x * y);
                            }
                        }
                    }
                    for r in 0..n {
                        let x = ada[(r, ka)];
                        if x != ZERO {
                            add(k, at(a + r * stride - ka * stride, b), j, -x * 0.5);
                        }
                        let y = ada[(kb, r)];
                        if y != ZERO {
                            add(k, at(a, b + r * stride - kb * stride), j, -y * 0.5);
                        }
                    }
                }
            }
        }
        // −R, whose blocks lie inside these.
        if let Some(redfield) = &relaxation.redfield {
            for (elements, r) in &redfield.blocks {
                let (a, b) = elements[0];
                let k = liouville.block[a * d + b];
                if k == u32::MAX {
                    continue;
                }
                let index: Vec<usize> = elements.iter().map(|&(c, e)| at(c, e)).collect();
                for (j, &cj) in index.iter().enumerate() {
                    for (i, &ri) in index.iter().enumerate() {
                        add(k as usize, ri, cj, -r[(i, j)]);
                    }
                }
            }
        }
    }

    /// The blocks of the Lindblad superoperator (times `t`) on the elements
    /// of `liouville`, for `H` and the relaxation; the population block is
    /// augmented by the drift times `trace`.
    fn liouville_blocks(
        &self,
        liouville: &Liouville,
        h: &Sparse,
        relaxation: &Relaxation,
        trace: C64,
        t: f64,
    ) -> Vec<LiouvilleBlock> {
        let mut matrices: Vec<(CMatrix, bool)> = liouville
            .blocks
            .iter()
            .map(|(order, elements)| {
                let augmented = order.iter().all(|&q| q == 0);
                let size = elements.len() + usize::from(augmented);
                (CMatrix::zeros(size, size), augmented)
            })
            .collect();
        self.generator_entries(liouville, h, relaxation, |k, r, c, v| {
            matrices[k].0[(r, c)] += v;
        });
        for ((_, elements), (m, augmented)) in liouville.blocks.iter().zip(matrices.iter_mut()) {
            if *augmented {
                let n = elements.len();
                for (j, &(a, b)) in elements.iter().enumerate() {
                    m[(j, n)] = relaxation.drift[(a, b)] * trace;
                }
            }
        }
        liouville
            .blocks
            .iter()
            .zip(matrices)
            .map(|((_, elements), (m, augmented))| {
                LiouvilleBlock::new(elements.clone(), m, augmented, t)
            })
            .collect()
    }

    /// Floating-point operations Dormand–Prince steps over `t` would take,
    /// weighted by [`DP_PENALTY`].
    fn dp_work(&self, h: &Sparse, relaxation: &Relaxation, t: f64) -> f64 {
        let d = self.dimension() as f64;
        let steps = ((h.spread() + relaxation.rate) * t / DP_RADIANS_PER_STEP).ceil() + 1.0;
        let jump_work: f64 = relaxation
            .jumps
            .iter()
            .map(|(_, a)| 32.0 * a.nrows() as f64 * d * d)
            .sum();
        let redfield_work = relaxation
            .redfield
            .as_ref()
            .map_or(0.0, |r| 8.0 * r.entries() as f64);
        let evaluation = 16.0 * h.entries() as f64 * d + jump_work + redfield_work + 60.0 * d * d;
        DP_PENALTY * 6.0 * steps * evaluation
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

    /// [`SpinSystem::relax`] with the relaxation at the field given.
    pub(super) fn relax_with(
        &self,
        rho: &CMatrix,
        h: &CMatrix,
        t: f64,
        conserving: bool,
        relaxation: &Relaxation,
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
        let sizes = coherence_sizes(&labels);
        let dp_work = self.dp_work(&sparse, relaxation, t);
        // Six products without squarings: no exponential costs less.
        let least: f64 = sizes.values().map(|&n| 36.0 * (n as f64).powi(3)).sum();
        let largest = sizes.values().copied().max().unwrap_or(0);
        let mut blocks = None;
        let mut exp_work = f64::INFINITY;
        if largest <= super::MAX_LIOUVILLE_BLOCK && least < dp_work {
            let liouville = Liouville::new(&labels, None);
            let built = self.liouville_blocks(&liouville, &sparse, relaxation, trace, t);
            exp_work = built.iter().map(LiouvilleBlock::work).sum();
            blocks = Some(built);
        }
        let work = exp_work.min(dp_work);
        if work > super::MAX_RELAXATION_WORK {
            return Err(self.too_much("a delay", t, work));
        }
        let mut next = match blocks {
            Some(blocks) if exp_work <= dp_work => {
                let mut next = CMatrix::zeros(d, d);
                for block in &blocks {
                    let moved = block.apply(|j| {
                        let (c, e) = block.elements[j];
                        delta[(c, e)]
                    });
                    for (&(a, b), v) in block.elements.iter().zip(moved) {
                        next[(a, b)] = v;
                    }
                }
                next
            }
            _ => {
                let c = &relaxation.drift * trace;
                self.dormand_prince(&delta, &sparse, relaxation, &c, t)?
            }
        };
        for i in 0..d {
            next[(i, i)] += level;
        }
        Ok(next)
    }

    /// The free induction decay with relaxation: `Tr(ρ(k·dwell) F₊)`. Only
    /// the coherences of order `−1` of the isotope contribute, and they
    /// evolve on their own: one block exponential for the dwell time when
    /// it fits, else Dormand–Prince steps between the points.
    pub(super) fn relaxing_fid(
        &self,
        rho: &CMatrix,
        isotope: Isotope,
        dwell: f64,
        points: usize,
    ) -> Result<Vec<C64>> {
        let d = self.dimension();
        let fp = self.fplus(isotope)?;
        let h = self.hamiltonian_at(self.field);
        let relaxation = self.relaxation(self.field)?;
        let sparse = Sparse::new(&h);
        let labels = self.labels();
        let order: Vec<i32> = self.raise(isotope).iter().map(|x| -x).collect();
        let size = coherence_sizes(&labels).get(&order).copied().unwrap_or(0);
        let dp_work = points as f64 * self.dp_work(&sparse, &relaxation, dwell);
        if size <= super::MAX_LIOUVILLE_BLOCK {
            let liouville = Liouville::new(&labels, Some(&order));
            let block = self
                .liouville_blocks(&liouville, &sparse, &relaxation, ONE, dwell)
                .pop()
                .expect("one block");
            let n = block.elements.len() as f64;
            let work = block.work() + 8.0 * points as f64 * n * n;
            if work <= dp_work || work <= super::MAX_RELAXATION_WORK {
                if work > super::MAX_RELAXATION_WORK {
                    return Err(self.too_much(
                        "a free induction decay",
                        dwell * points as f64,
                        work,
                    ));
                }
                let p = expm(&block.generator) * block.shift.exp();
                let mut x = nalgebra::DVector::from_iterator(
                    block.elements.len(),
                    block.elements.iter().map(|&(a, b)| rho[(a, b)]),
                );
                let f: Vec<C64> = block.elements.iter().map(|&(a, b)| fp[(b, a)]).collect();
                let mut out = Vec::with_capacity(points);
                for k in 0..points {
                    if k > 0 {
                        x = &p * x;
                    }
                    out.push(x.iter().zip(&f).map(|(a, b)| a * b).sum());
                }
                return Ok(out);
            }
        }
        if dp_work > super::MAX_RELAXATION_WORK {
            return Err(self.too_much("a free induction decay", dwell * points as f64, dp_work));
        }
        // F₊ is traceless: the deviation carries the signal.
        let trace = rho.trace();
        let mut delta = rho.clone();
        for i in 0..d {
            delta[(i, i)] -= trace / d as f64;
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
