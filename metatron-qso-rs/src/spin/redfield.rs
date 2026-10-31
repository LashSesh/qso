//! Relaxation from first principles: Bloch–Wangsness–Redfield theory for a
//! rigid molecule tumbling isotropically with correlation time `τ_c`.
//!
//! Each interaction `k` is `Σ_ab A^k_ab(t) Q^k_ab` with a traceless
//! spatial tensor `A^k` that rotates with the molecule: the dipolar
//! coupling `b_ij (1 − 3nnᵀ)` between `I_i` and `I_j`
//! (`b_ij = (μ₀/4π) γ_i γ_j ħ / r³`, `Q_ab = I_ia I_jb`), and the
//! anisotropic shielding `γ_i B σ` of spin `i` (`Q_ab = I_ia δ_bz`). Under
//! isotropic rotational diffusion,
//! `⟨A^k_ab(0) A^l_cd(τ)⟩ = (A^k : A^l)/10 · (δ_ac δ_bd + δ_ad δ_bc −
//! ⅔ δ_ab δ_cd) e^{−τ/τ_c}`, so cross-correlations between interactions
//! follow from the contractions `A^k : A^l`. Written in an orthonormal
//! basis `E_μ` of traceless symmetric tensors, the second-order master
//! equation becomes a sum of independent terms,
//!
//! ```text
//! dρ/dt = −i[H₀, ρ] − R(ρ − ρ_eq),
//! R(ρ) = (1/5) Σ_μ Σ_ab [𝒬_μab, [Y(𝒬_μab), ρ]],   𝒬_μab = Σ_k (A^k : E_μ) Q^k,S_ab,
//! ```
//!
//! with `Q^S` the traceless symmetric part of `Q` and `Y(X)_mn = X_mn
//! J(ω_mn)` in the eigenbasis of the laboratory-frame `H₀`,
//! `J(ω) = τ_c/(1 + ω²τ_c²)`: the full (non-secular) Redfield tensor
//! with real spectral densities (dynamic frequency shifts are left out).
//! `ρ_eq` is the thermal state, a fixed point at any temperature; away
//! from it this semiclassical correction is first order in `ħω/k_BT`
//! (no detailed balance), so at polarisations near 1 (millikelvin) states
//! far from equilibrium can leave the physical ones. In the laboratory
//! frame the flip-flop terms between isotopes shift the steady state by
//! about `J/Δν`. Redfield theory needs fast motion, `|H₁|·τ_c ≪ 1`;
//! interactions with `|A|·τ_c > 1` are refused. In the rotating frame
//! only the part of `R` that keeps the coherence order of every isotope
//! is kept (the secular approximation with respect to the carriers); in
//! the laboratory frame `R` keeps the total coherence order exactly.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use num_complex::Complex64 as C64;

use super::blocks::{Eigenblock, Liouville, distinct};
use super::krylov::{Accumulator, SparseBlock};
use super::{Frame, HBAR, ONE, SpinSystem, ZERO, spin_ops};
use crate::error::{QsoError, Result};
use crate::linalg::dense::cmul;
use crate::linalg::{CMatrix, RMatrix};
use crate::parallel::map_tasks;

/// `μ₀/4π` (T² m³ J⁻¹, CODATA 2018).
pub const MU0_OVER_4PI: f64 = 1.000_000_000_55e-7;
/// Largest `|H₁|·τ_c` (Frobenius norm of an interaction's spatial tensor
/// in rad/s times the correlation time) accepted; Redfield theory needs it
/// far below 1.
const MAX_NARROWING: f64 = 1.0;

/// The shielding tensor (ppm) of an axially symmetric anisotropy
/// `Δσ = σ_∥ − σ_⊥` along `axis`: `(Δσ/3)(3nnᵀ − 1)`, traceless.
pub fn axial_csa(anisotropy: f64, axis: [f64; 3]) -> Result<[[f64; 3]; 3]> {
    let norm = axis.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !(anisotropy.is_finite() && norm.is_finite() && norm > 0.0) {
        return Err(QsoError::invalid(
            "csa",
            "a finite anisotropy along a finite, non-zero axis",
        ));
    }
    let n = axis.map(|x| x / norm);
    let mut t = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let delta = if a == b { 1.0 } else { 0.0 };
            t[a][b] = anisotropy / 3.0 * (3.0 * n[a] * n[b] - delta);
        }
    }
    Ok(t)
}

/// The traceless symmetric part of a 3 × 3 tensor.
fn anisotropic(t: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let trace = (t[0][0] + t[1][1] + t[2][2]) / 3.0;
    let mut out = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            out[a][b] = (t[a][b] + t[b][a]) / 2.0 - if a == b { trace } else { 0.0 };
        }
    }
    out
}

/// An orthonormal basis of traceless symmetric 3 × 3 tensors.
fn basis() -> [[[f64; 3]; 3]; 5] {
    let (s2, s6) = (2f64.sqrt(), 6f64.sqrt());
    let mut e = [[[0.0; 3]; 3]; 5];
    e[0][0][0] = 1.0 / s2;
    e[0][1][1] = -1.0 / s2;
    e[1][0][0] = -1.0 / s6;
    e[1][1][1] = -1.0 / s6;
    e[1][2][2] = 2.0 / s6;
    for (k, (a, b)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
        e[2 + k][a][b] = 1.0 / s2;
        e[2 + k][b][a] = 1.0 / s2;
    }
    e
}

/// The spin part of an interaction.
#[derive(Clone, Copy)]
enum Coupling {
    Dipolar(usize, usize),
    Shielding(usize),
}

/// The basis states ordered by their labels, each label's states
/// contiguous.
struct Grouping {
    /// The state at each position.
    order: Vec<usize>,
    /// The position of each state.
    rank: Vec<u32>,
    /// The label (an index into `labels`) at each position.
    label: Vec<u32>,
    /// `starts[l]..starts[l + 1]`: the positions of label `l`.
    starts: Vec<usize>,
    index: BTreeMap<Vec<i32>, u32>,
}

impl Grouping {
    fn new(labels: &[Vec<i32>]) -> Self {
        let (list, id) = distinct(labels);
        let mut order: Vec<usize> = (0..labels.len()).collect();
        order.sort_by_key(|&s| (id[s], s));
        let mut rank = vec![0u32; labels.len()];
        for (p, &s) in order.iter().enumerate() {
            rank[s] = p as u32;
        }
        let label: Vec<u32> = order.iter().map(|&s| id[s] as u32).collect();
        let mut starts = vec![0usize; list.len() + 1];
        for &l in &label {
            starts[l as usize + 1] += 1;
        }
        for l in 0..list.len() {
            starts[l + 1] += starts[l];
        }
        let index = list
            .into_iter()
            .enumerate()
            .map(|(k, l)| (l, k as u32))
            .collect();
        Grouping {
            order,
            rank,
            label,
            starts,
            index,
        }
    }

    /// For every label `l`, the label `l − q`, if a state has it.
    fn shifted(&self, q: &[i32]) -> Vec<Option<u32>> {
        let mut out = vec![None; self.index.len()];
        for (l, &k) in &self.index {
            let target: Vec<i32> = l.iter().zip(q).map(|(x, y)| x - y).collect();
            out[k as usize] = self.index.get(&target).copied();
        }
        out
    }
}

/// One term `w (𝒬 ⊗ Y(𝒬))` of the superoperator: both operators
/// Hermitian, by rows in the positions of the grouping.
struct Term {
    q: SparseBlock,
    y: SparseBlock,
    w: f64,
}

/// The Redfield superoperator, block by block of coherence order. Its
/// columns come from the terms on demand, so a block takes memory only
/// while it is used; [`RedfieldBlocks::apply`] keeps the blocks it built.
pub(super) struct RedfieldBlocks {
    grouping: Grouping,
    terms: Vec<Term>,
    /// Column `a` of `Σ w 𝒬 Y(𝒬)` within the label of `a`, as row `a`.
    left: SparseBlock,
    /// The union of the terms' patterns, for estimates.
    pattern: SparseBlock,
    /// The blocks of coherence order.
    pub liouville: Liouville,
    built: Vec<OnceLock<SparseBlock>>,
    counts: Vec<OnceLock<usize>>,
    rate: f64,
}

impl RedfieldBlocks {
    /// For the labels of `c`, the labels `e` must have for `|c⟩⟨e|` to lie
    /// in block `k`.
    pub fn targets(&self, k: usize) -> Vec<Option<u32>> {
        self.grouping.shifted(&self.liouville.blocks[k].0)
    }

    /// `scale · R(|a⟩⟨b|)` for the element `j = (a, b)` of block `k`:
    /// `add(c, e, v)` for its entries at `|c⟩⟨e|` (repeated `(c, e)` add
    /// up). `targets` are the block's.
    pub fn column(
        &self,
        k: usize,
        j: usize,
        targets: &[Option<u32>],
        scale: C64,
        mut add: impl FnMut(usize, usize, C64),
    ) {
        let g = &self.grouping;
        let (a, b) = self.liouville.blocks[k].1[j];
        let (pa, pb) = (g.rank[a] as usize, g.rank[b] as usize);
        // Σ w 𝒬Y|a⟩⟨b| from column a of the left product, Σ w |a⟩⟨b|Y𝒬
        // from its column b conjugated (Y𝒬 = (𝒬Y)†).
        for (pc, v) in self.left.row(pa) {
            add(g.order[pc], b, v * scale);
        }
        for (pe, v) in self.left.row(pb) {
            add(a, g.order[pe], v.conj() * scale);
        }
        for term in &self.terms {
            let f = scale * -term.w;
            self.cross(&term.q, &term.y, pa, pb, targets, f, &mut add);
            self.cross(&term.y, &term.q, pa, pb, targets, f, &mut add);
        }
    }

    /// `f · X|a⟩⟨b|Z` on the block: `X_ca = conj(X_ac)` from row `a` of
    /// `X` (Hermitian) and `Z_be` from row `b` of `Z`, each label of `c`
    /// with the states `e` of its target label.
    #[allow(clippy::too_many_arguments)]
    fn cross(
        &self,
        x: &SparseBlock,
        z: &SparseBlock,
        pa: usize,
        pb: usize,
        targets: &[Option<u32>],
        f: C64,
        add: &mut impl FnMut(usize, usize, C64),
    ) {
        let g = &self.grouping;
        let (xc, xv) = x.row_parts(pa);
        let (zc, zv) = z.row_parts(pb);
        if zc.is_empty() {
            return;
        }
        let mut i = 0;
        while i < xc.len() {
            let l = g.label[xc[i] as usize];
            let mut end = i + 1;
            while end < xc.len() && g.label[xc[end] as usize] == l {
                end += 1;
            }
            if let Some(t) = targets[l as usize] {
                let (lo, hi) = (g.starts[t as usize] as u32, g.starts[t as usize + 1] as u32);
                let s0 = zc.partition_point(|&c| c < lo);
                let s1 = s0 + zc[s0..].partition_point(|&c| c < hi);
                for r in i..end {
                    let c = g.order[xc[r] as usize];
                    let w = xv[r].conj() * f;
                    for s in s0..s1 {
                        add(c, g.order[zc[s] as usize], w * zv[s]);
                    }
                }
            }
            i = end;
        }
    }

    /// Products `X_ca Z_be` the union pattern gives for the element at
    /// positions `(pa, pb)`: a bound on the entries of its column.
    fn products(&self, pa: usize, pb: usize, targets: &[Option<u32>]) -> usize {
        let g = &self.grouping;
        let (xc, _) = self.pattern.row_parts(pa);
        let (zc, _) = self.pattern.row_parts(pb);
        let mut count = 0;
        let mut i = 0;
        while i < xc.len() {
            let l = g.label[xc[i] as usize];
            let mut end = i + 1;
            while end < xc.len() && g.label[xc[end] as usize] == l {
                end += 1;
            }
            if let Some(t) = targets[l as usize] {
                let (lo, hi) = (g.starts[t as usize] as u32, g.starts[t as usize + 1] as u32);
                let s0 = zc.partition_point(|&c| c < lo);
                let s1 = s0 + zc[s0..].partition_point(|&c| c < hi);
                count += (end - i) * (s1 - s0);
            }
            i = end;
        }
        count
    }

    /// At most how many entries block `k` has (without building it).
    pub fn count(&self, k: usize) -> usize {
        *self.counts[k].get_or_init(|| self.bound(k))
    }

    fn bound(&self, k: usize) -> usize {
        let g = &self.grouping;
        let elements = &self.liouville.blocks[k].1;
        let n = elements.len();
        let targets = self.targets(k);
        let chunk = 1024;
        map_tasks(n.div_ceil(chunk), |c| {
            (c * chunk..((c + 1) * chunk).min(n))
                .map(|j| {
                    let (a, b) = elements[j];
                    let (pa, pb) = (g.rank[a] as usize, g.rank[b] as usize);
                    let total = self.left.row_parts(pa).0.len()
                        + self.left.row_parts(pb).0.len()
                        + self.products(pa, pb, &targets);
                    total.min(n)
                })
                .sum::<usize>()
        })
        .into_iter()
        .sum()
    }

    /// Block `k` as a sparse matrix (columns in parallel).
    fn build(&self, k: usize) -> SparseBlock {
        let n = self.liouville.blocks[k].1.len();
        let targets = self.targets(k);
        let chunk = 256;
        let parts: Vec<Vec<Vec<(u32, C64)>>> = map_tasks(n.div_ceil(chunk), |c| {
            let mut acc = Accumulator::new(n);
            (c * chunk..((c + 1) * chunk).min(n))
                .map(|j| {
                    self.column(k, j, &targets, C64::new(1.0, 0.0), |r, e, v| {
                        acc.add(self.liouville.position_of(r, e), v)
                    });
                    acc.take()
                })
                .collect()
        });
        let columns: Vec<Vec<(u32, C64)>> = parts.into_iter().flatten().collect();
        SparseBlock::from_columns(n, &columns)
    }

    /// Block `k`, built once.
    pub fn block(&self, k: usize) -> &SparseBlock {
        self.built[k].get_or_init(|| self.build(k))
    }

    /// Bytes of the blocks built so far.
    pub fn bytes(&self) -> usize {
        self.built
            .iter()
            .filter_map(OnceLock::get)
            .map(SparseBlock::bytes)
            .sum()
    }

    /// `out += scale · R(x)` from the blocks `x` touches, built once (for
    /// repeated use).
    pub fn apply(&self, x: &CMatrix, out: &mut CMatrix, scale: f64) {
        for (k, (_, elements)) in self.liouville.blocks.iter().enumerate() {
            let v: Vec<C64> = elements.iter().map(|&(a, b)| x[(a, b)]).collect();
            if v.iter().all(|z| *z == ZERO) {
                continue;
            }
            let mut y = vec![ZERO; v.len()];
            self.block(k).apply(&v, &mut y);
            for (&(a, b), yi) in elements.iter().zip(y) {
                out[(a, b)] += yi * scale;
            }
        }
    }

    /// `out += scale · R(x)`, column by column without keeping anything.
    pub fn image(&self, x: &CMatrix, out: &mut CMatrix, scale: f64) {
        for (k, (_, elements)) in self.liouville.blocks.iter().enumerate() {
            if elements.iter().all(|&(a, b)| x[(a, b)] == ZERO) {
                continue;
            }
            let targets = self.targets(k);
            let mut acc = Accumulator::new(elements.len());
            for (j, &(a, b)) in elements.iter().enumerate() {
                let v = x[(a, b)];
                if v != ZERO {
                    self.column(k, j, &targets, v * scale, |r, e, w| {
                        acc.add(self.liouville.position_of(r, e), w)
                    });
                }
            }
            for (i, v) in acc.take() {
                let (a, b) = elements[i as usize];
                out[(a, b)] += v;
            }
        }
    }

    /// A bound on the largest column sum of `|R|`, and so on its rates.
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Number of terms `𝒬 ⊗ Y(𝒬)`.
    pub fn terms(&self) -> usize {
        self.terms.len()
    }
}

/// `Y(X)_mn = X_mn J(ω_mn)` of a Hermitian `X` (by rows in the positions of
/// `grouping`) in the eigenbasis of the laboratory `H₀`, one pair of its
/// blocks at a time. Entries below 10⁻¹⁵ of the largest of their pair of
/// blocks are the rounding of the transformation and dropped.
fn spectral(x: &SparseBlock, eigen: &[Eigenblock], grouping: &Grouping, tau: f64) -> SparseBlock {
    let d = x.n;
    // The block of every state, and its index there.
    let mut home = vec![(0u32, 0u32); d];
    for (k, block) in eigen.iter().enumerate() {
        for (i, &s) in block.states.iter().enumerate() {
            home[s] = (k as u32, i as u32);
        }
    }
    let parts: Vec<Vec<(u32, u32, C64)>> = map_tasks(eigen.len(), |ka| {
        let a = &eigen[ka];
        let mut subs: BTreeMap<u32, CMatrix> = BTreeMap::new();
        for (i, &s) in a.states.iter().enumerate() {
            for (pc, v) in x.row(grouping.rank[s] as usize) {
                let (kb, j) = home[grouping.order[pc]];
                subs.entry(kb).or_insert_with(|| {
                    CMatrix::zeros(a.states.len(), eigen[kb as usize].states.len())
                })[(i, j as usize)] = v;
            }
        }
        let mut out = Vec::new();
        for (kb, sub) in subs {
            let b = &eigen[kb as usize];
            let mut t = cmul(&cmul(&a.vectors.adjoint(), &sub), &b.vectors);
            for (r, &ea) in a.values.iter().enumerate() {
                for (c, &eb) in b.values.iter().enumerate() {
                    let w = (a.shift - b.shift) + ea - eb;
                    t[(r, c)] *= tau / (1.0 + w * w * tau * tau);
                }
            }
            let back = cmul(&cmul(&a.vectors, &t), &b.vectors.adjoint());
            let floor = 1e-15 * back.iter().fold(0.0f64, |m, z| m.max(z.norm()));
            for (r, &s) in a.states.iter().enumerate() {
                for (c, &e) in b.states.iter().enumerate() {
                    let v = back[(r, c)];
                    if v.norm() > floor {
                        out.push((grouping.rank[s], grouping.rank[e], v));
                    }
                }
            }
        }
        out
    });
    SparseBlock::from_triplets(d, parts.into_iter().flatten().collect())
}

/// Column `a` of `Σ w 𝒬 Y(𝒬)` within the label of `a` (the part a
/// column of a block can reach), as row `a`: `(𝒬Y)_ca = Σ_m conj(𝒬_mc
/// Y_am)`.
fn left_products(terms: &[Term], g: &Grouping) -> SparseBlock {
    let d = g.order.len();
    let chunk = 64;
    let rows: Vec<Vec<(u32, C64)>> = map_tasks(d.div_ceil(chunk), |c| {
        (c * chunk..((c + 1) * chunk).min(d))
            .map(|pa| {
                let l = g.label[pa] as usize;
                let (lo, hi) = (g.starts[l], g.starts[l + 1]);
                let mut acc = vec![ZERO; hi - lo];
                for term in terms {
                    let (yc, yv) = term.y.row_parts(pa);
                    for (&pm, &yam) in yc.iter().zip(yv) {
                        let (qc, qv) = term.q.row_parts(pm as usize);
                        let s0 = qc.partition_point(|&c| (c as usize) < lo);
                        let s1 = s0 + qc[s0..].partition_point(|&c| (c as usize) < hi);
                        let f = yam.conj() * term.w;
                        for s in s0..s1 {
                            acc[qc[s] as usize - lo] += qv[s].conj() * f;
                        }
                    }
                }
                acc.into_iter()
                    .enumerate()
                    .filter(|(_, v)| *v != ZERO)
                    .map(|(i, v)| ((lo + i) as u32, v))
                    .collect()
            })
            .collect::<Vec<_>>()
    })
    .into_iter()
    .flatten()
    .collect();
    SparseBlock::from_rows(d, rows)
}

impl SpinSystem {
    /// Set the position (Å) of spin `i`, for dipolar relaxation.
    pub fn set_position(&mut self, i: usize, position: [f64; 3]) -> Result<&mut Self> {
        self.check_spin(i)?;
        if position.iter().any(|x| !x.is_finite()) {
            return Err(QsoError::invalid("position", "positions must be finite"));
        }
        let old = self.spins[i].position.replace(position);
        if let Err(e) = self.validate_motion() {
            self.spins[i].position = old;
            return Err(e);
        }
        Ok(self)
    }

    /// Set the shielding tensor (ppm, molecular frame) of spin `i`, or
    /// remove it; only its traceless symmetric part relaxes.
    pub fn set_csa(&mut self, i: usize, tensor: Option<[[f64; 3]; 3]>) -> Result<&mut Self> {
        self.check_spin(i)?;
        if tensor.is_some_and(|t| t.iter().flatten().any(|x| !x.is_finite())) {
            return Err(QsoError::invalid("csa", "shielding tensors must be finite"));
        }
        self.spins[i].csa = tensor;
        Ok(self)
    }

    /// The same system tumbling isotropically with correlation time
    /// `τ_c` (s), or without Redfield relaxation (`None`).
    pub fn with_correlation_time(mut self, tau_c: Option<f64>) -> Result<Self> {
        if let Some(t) = tau_c
            && !(t.is_finite() && t > 0.0)
        {
            return Err(QsoError::invalid(
                "correlation_time",
                "the correlation time must be finite and positive",
            ));
        }
        self.correlation_time = tau_c;
        Ok(self)
    }

    /// The rotational correlation time (s), if set.
    pub fn correlation_time(&self) -> Option<f64> {
        self.correlation_time
    }

    /// The dipolar coupling constant `b_ij = (μ₀/4π) γ_i γ_j ħ / r³`
    /// (rad/s) of two spins with positions.
    pub fn dipolar_coupling(&self, i: usize, j: usize) -> Result<f64> {
        self.check_spin(i)?;
        self.check_spin(j)?;
        match (self.spins[i].position, self.spins[j].position) {
            (Some(p), Some(q)) if i != j => {
                let r = distance(p, q) * 1e-10;
                Ok(MU0_OVER_4PI
                    * self.spins[i].isotope.gamma()
                    * self.spins[j].isotope.gamma()
                    * HBAR
                    / (r * r * r))
            }
            _ => Err(QsoError::invalid(
                "dipolar",
                "a dipolar coupling joins two different spins with positions",
            )),
        }
    }

    pub(super) fn validate_motion(&self) -> Result<()> {
        if let Some(t) = self.correlation_time
            && !(t.is_finite() && t > 0.0)
        {
            return Err(QsoError::invalid(
                "correlation_time",
                "the correlation time must be finite and positive",
            ));
        }
        for (i, s) in self.spins.iter().enumerate() {
            if s.position.is_some_and(|p| p.iter().any(|x| !x.is_finite()))
                || s.csa
                    .is_some_and(|t| t.iter().flatten().any(|x| !x.is_finite()))
            {
                return Err(QsoError::invalid(
                    "spins",
                    "positions and shielding tensors must be finite",
                ));
            }
            for j in i + 1..self.spins.len() {
                if let (Some(p), Some(q)) = (s.position, self.spins[j].position)
                    && distance(p, q) == 0.0
                {
                    return Err(QsoError::invalid(
                        "position",
                        format!("spins {i} and {j} share a position"),
                    ));
                }
            }
        }
        Ok(())
    }

    /// The interactions that relax, with their spatial tensors (rad/s).
    fn interactions(&self, b: f64) -> Vec<([[f64; 3]; 3], Coupling)> {
        let mut out = Vec::new();
        let n = self.spins.len();
        for i in 0..n {
            for j in i + 1..n {
                if let (Some(p), Some(q)) = (self.spins[i].position, self.spins[j].position) {
                    let coupling = self.dipolar_coupling(i, j).expect("positions");
                    let r = distance(p, q);
                    let u = [(q[0] - p[0]) / r, (q[1] - p[1]) / r, (q[2] - p[2]) / r];
                    let mut a = [[0.0; 3]; 3];
                    for (x, row) in a.iter_mut().enumerate() {
                        for (y, v) in row.iter_mut().enumerate() {
                            let delta = if x == y { 1.0 } else { 0.0 };
                            *v = coupling * (delta - 3.0 * u[x] * u[y]);
                        }
                    }
                    out.push((a, Coupling::Dipolar(i, j)));
                }
            }
            if let Some(t) = self.spins[i].csa {
                let w = self.spins[i].isotope.gamma() * b * 1e-6;
                let a = anisotropic(&t).map(|row| row.map(|x| x * w));
                if a.iter().flatten().any(|x| *x != 0.0) {
                    out.push((a, Coupling::Shielding(i)));
                }
            }
        }
        out
    }

    /// Whether Redfield relaxation is on and has an interaction.
    pub(super) fn redfield_active(&self) -> bool {
        self.correlation_time.is_some() && !self.interactions(1.0).is_empty()
    }

    /// The six components `𝒬^S_xy` (`xx, yy, zz, xy, xz, yz`) of an
    /// interaction's spin tensor, on the spins it acts on (the first the
    /// most significant): `(I_x I_y + I_y I_x)/2 − δ_xy I·I/3` for a dipolar
    /// pair, `(I_x δ_yz + I_y δ_xz)/2 − δ_xy I_z/3` for a shielding.
    fn local_tensor(&self, c: Coupling) -> (Vec<usize>, Vec<CMatrix>) {
        let pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
        let ops = |i: usize| {
            let [iz, _, _, ix, iy] = spin_ops(self.spins[i].isotope.levels());
            [ix, iy, iz]
        };
        let half = C64::new(0.5, 0.0);
        let third = C64::new(1.0 / 3.0, 0.0);
        match c {
            Coupling::Dipolar(i, j) => {
                let (a, b) = (ops(i), ops(j));
                let product = |x: usize, y: usize| a[x].kronecker(&b[y]);
                let dot = product(0, 0) + product(1, 1) + product(2, 2);
                let tensor = pairs
                    .iter()
                    .map(|&(x, y)| {
                        let mut q = (product(x, y) + product(y, x)) * half;
                        if x == y {
                            q -= &dot * third;
                        }
                        q
                    })
                    .collect();
                (vec![i, j], tensor)
            }
            Coupling::Shielding(i) => {
                let a = ops(i);
                let n = a[0].nrows();
                let tensor = pairs
                    .iter()
                    .map(|&(x, y)| {
                        let mut q = CMatrix::zeros(n, n);
                        if y == 2 {
                            q += &a[x] * half;
                        }
                        if x == 2 {
                            q += &a[y] * half;
                        }
                        if x == y {
                            q -= &a[2] * third;
                        }
                        q
                    })
                    .collect();
                (vec![i], tensor)
            }
        }
    }

    /// The entries `(row, column, value)` of an operator `local` on
    /// `spins` (the first the most significant) in the full space.
    fn embed_local(&self, spins: &[usize], local: &CMatrix) -> Vec<(usize, usize, C64)> {
        let strides: Vec<usize> = spins.iter().map(|&i| self.stride(i)).collect();
        let levels: Vec<usize> = spins
            .iter()
            .map(|&i| self.spins[i].isotope.levels())
            .collect();
        let n = local.nrows();
        // The part of a state's index that each local index stands for.
        let offset: Vec<usize> = (0..n)
            .map(|l| {
                let mut rest = l;
                let mut out = 0;
                for k in (0..spins.len()).rev() {
                    out += (rest % levels[k]) * strides[k];
                    rest /= levels[k];
                }
                out
            })
            .collect();
        let mut out = Vec::new();
        for c in 0..self.dimension() {
            let lc = (0..spins.len()).fold(0, |l, k| l * levels[k] + (c / strides[k]) % levels[k]);
            let base = c - offset[lc];
            for (lr, &o) in offset.iter().enumerate() {
                let v = local[(lr, lc)];
                if v != ZERO {
                    out.push((base + o, c, v));
                }
            }
        }
        out
    }

    /// The Redfield superoperator at field `b`, block by block of the
    /// coherence orders of the labels (`None` without motion or
    /// interactions). Its memory and the work of building its terms are
    /// estimated first and refused above [`super::MAX_RELAXATION_MEMORY`]
    /// and [`super::MAX_RELAXATION_WORK`]; the blocks themselves are built
    /// where they are used.
    pub(super) fn redfield(&self, b: f64) -> Result<Option<RedfieldBlocks>> {
        let Some(tau) = self.correlation_time else {
            return Ok(None);
        };
        let interactions = self.interactions(b);
        if interactions.is_empty() {
            return Ok(None);
        }
        // Redfield theory needs fast motion: |H₁| τ_c ≪ 1.
        let strength = interactions
            .iter()
            .map(|(a, _)| a.iter().flatten().map(|x| x * x).sum::<f64>().sqrt())
            .fold(0.0, f64::max);
        let narrowing = strength * tau;
        if narrowing.is_nan() || narrowing > MAX_NARROWING {
            return Err(QsoError::invalid(
                "redfield",
                format!(
                    "an interaction of {strength:.3e} rad/s with τ_c = {tau:e} s is outside \
                     Redfield theory, which needs |H₁|·τ_c ≪ 1 (here {narrowing:.3e})"
                ),
            ));
        }
        let labels = self.labels();
        let mut lab = self.clone();
        lab.frame = Frame::Laboratory;
        let (work, memory) = redfield_cost(&lab.labels(), &labels, interactions.len());
        if work > super::MAX_RELAXATION_WORK || memory > super::MAX_RELAXATION_MEMORY {
            return Err(QsoError::invalid(
                "redfield",
                format!(
                    "the Redfield superoperator at dimension {} needs about {work:.1e} \
                     operations and {:.1} GB to build (limits {:.1e} and {:.1} GB)",
                    self.dimension(),
                    memory / 1e9,
                    super::MAX_RELAXATION_WORK,
                    super::MAX_RELAXATION_MEMORY / 1e9
                ),
            ));
        }
        let d = self.dimension();
        let eigen = lab.eigenblocks(&lab.hamiltonian_at(b));
        let grouping = Grouping::new(&labels);
        let locals: Vec<(Vec<usize>, Vec<CMatrix>)> = interactions
            .iter()
            .map(|&(_, c)| self.local_tensor(c))
            .collect();
        let pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
        // Terms (𝒬, Y(𝒬), weight).
        let mut terms = Vec::new();
        for basis_tensor in &basis() {
            let alpha: Vec<f64> = interactions
                .iter()
                .map(|(a, _)| {
                    (0..3)
                        .map(|x| (0..3).map(|y| a[x][y] * basis_tensor[x][y]).sum::<f64>())
                        .sum()
                })
                .collect();
            if alpha.iter().all(|x| *x == 0.0) {
                continue;
            }
            for (p, &(x, y)) in pairs.iter().enumerate() {
                let mut triplets = Vec::new();
                for ((spins, local), &weight) in locals.iter().zip(&alpha) {
                    if weight != 0.0 {
                        for (r, c, v) in self.embed_local(spins, &local[p]) {
                            triplets.push((grouping.rank[r], grouping.rank[c], v * weight));
                        }
                    }
                }
                let q = SparseBlock::from_triplets(d, triplets);
                if q.nnz() == 0 {
                    continue;
                }
                let yq = spectral(&q, &eigen, &grouping, tau);
                let weight = if x == y { 1.0 } else { 2.0 } / 5.0;
                terms.push(Term {
                    q,
                    y: yq,
                    w: weight,
                });
            }
        }
        let left = left_products(&terms, &grouping);
        let finite = |m: &SparseBlock| {
            (0..m.n).all(|r| m.row(r).all(|(_, v)| v.re.is_finite() && v.im.is_finite()))
        };
        if !(finite(&left) && terms.iter().all(|t| finite(&t.q) && finite(&t.y))) {
            return Err(QsoError::invalid(
                "redfield",
                "the Redfield superoperator is not finite",
            ));
        }
        let mut triplets = Vec::new();
        for t in &terms {
            for m in [&t.q, &t.y] {
                for r in 0..d {
                    triplets.extend(m.row(r).map(|(c, _)| (r as u32, c as u32, ONE)));
                }
            }
        }
        let pattern = SparseBlock::from_triplets(d, triplets);
        // Σ_ce |R_ce,ab| ≤ |𝒬Y col a| + |𝒬Y col b| + Σ w (‖𝒬‖‖Y‖ + ‖Y‖‖𝒬‖).
        let rate = 2.0 * left.norm_inf()
            + terms
                .iter()
                .map(|t| 2.0 * t.w * t.q.norm_inf() * t.y.norm_inf())
                .sum::<f64>();
        let liouville = Liouville::new(&labels, None);
        let built = (0..liouville.blocks.len())
            .map(|_| OnceLock::new())
            .collect();
        let counts = (0..liouville.blocks.len())
            .map(|_| OnceLock::new())
            .collect();
        Ok(Some(RedfieldBlocks {
            grouping,
            terms,
            left,
            pattern,
            liouville,
            built,
            counts,
            rate,
        }))
    }

    /// Whether the Redfield relaxation is unchanged by the spin
    /// permutation `p`: it maps the relaxing interactions onto each other
    /// and keeps the contractions `A^k : A^l` of their tensors, on which
    /// alone isotropic tumbling makes the relaxation depend (to 10⁻⁹ of the
    /// largest).
    pub(super) fn redfield_invariant(&self, p: &[usize]) -> bool {
        if self.correlation_time.is_none() {
            return true;
        }
        let interactions = self.interactions(1.0);
        let key = |c: Coupling| match c {
            Coupling::Dipolar(i, j) => (i.min(j), i.max(j), false),
            Coupling::Shielding(i) => (i, i, true),
        };
        let index: BTreeMap<(usize, usize, bool), usize> = interactions
            .iter()
            .enumerate()
            .map(|(k, &(_, c))| (key(c), k))
            .collect();
        let image: Option<Vec<usize>> = interactions
            .iter()
            .map(|&(_, c)| {
                let moved = match c {
                    Coupling::Dipolar(i, j) => Coupling::Dipolar(p[i], p[j]),
                    Coupling::Shielding(i) => Coupling::Shielding(p[i]),
                };
                index.get(&key(moved)).copied()
            })
            .collect();
        let Some(image) = image else {
            return false;
        };
        let gram = |k: usize, l: usize| -> f64 {
            let (a, b) = (&interactions[k].0, &interactions[l].0);
            (0..3)
                .map(|x| (0..3).map(|y| a[x][y] * b[x][y]).sum::<f64>())
                .sum()
        };
        let n = interactions.len();
        let largest = (0..n).map(|k| gram(k, k)).fold(0.0, f64::max);
        (0..n).all(|k| {
            (0..n).all(|l| (gram(image[k], image[l]) - gram(k, l)).abs() <= 1e-9 * largest)
        })
    }

    /// Relaxation rates among observables: `Γ_ij = −Re Tr(O_i† D(O_j)) /
    /// Tr(O_i† O_i)`, with `D` the relaxation of the master equation (the
    /// Lindblad operators of `T₁` and `T₂` and the Redfield superoperator,
    /// without `H`) at the system's field. For orthogonal traceless `O_j`
    /// and `ρ = 1/d + Σ x_j O_j` near equilibrium, `dx_i/dt = −Σ Γ_ij (x_j
    /// − x_j^eq)`: `Γ_ii` is the auto-relaxation rate of `O_i` (`1/T₁` for
    /// `I_z`, `1/T₂` for `I₊` of an isolated spin), `Γ_ij` the
    /// cross-relaxation.
    pub fn relaxation_rates(&self, ops: &[CMatrix]) -> Result<RMatrix> {
        self.validate()?;
        let d = self.dimension();
        if let Some(o) = ops.iter().find(|o| o.shape() != (d, d)) {
            return Err(QsoError::DimensionMismatch {
                expected: d,
                actual: o.nrows(),
            });
        }
        let relaxation = self.relaxation(self.field)?;
        let images: Vec<CMatrix> = ops.iter().map(|o| relaxation.dissipate(o)).collect();
        let mut out = RMatrix::zeros(ops.len(), ops.len());
        for (i, oi) in ops.iter().enumerate() {
            let norm: f64 = oi.iter().map(|z| z.norm_sqr()).sum();
            if norm == 0.0 {
                return Err(QsoError::invalid("operators", "an observable is zero"));
            }
            for (j, image) in images.iter().enumerate() {
                let t: C64 = oi.iter().zip(image.iter()).map(|(a, b)| a.conj() * b).sum();
                // + 0.0: no rate is −0.
                out[(i, j)] = -t.re / norm + 0.0;
            }
        }
        Ok(out)
    }
}

/// Estimated operations and bytes of building the Redfield terms: 30
/// operators, their spectral images (at most the pairs of laboratory
/// blocks whose `2M` differ by up to 4), their union, the left products,
/// the laboratory Hamiltonian's eigenvectors and the index of the blocks.
fn redfield_cost(lab: &[Vec<i32>], labels: &[Vec<i32>], interactions: usize) -> (f64, f64) {
    let d = labels.len() as f64;
    let mut sizes: BTreeMap<i32, f64> = BTreeMap::new();
    for l in lab {
        *sizes.entry(l[0]).or_default() += 1.0;
    }
    let (mut y, mut spectral, mut eigen) = (0.0, 0.0, 0.0);
    for (&ma, &na) in &sizes {
        eigen += 40.0 * na * na * na;
        for (&mb, &nb) in &sizes {
            if (ma - mb).abs() <= 4 {
                y += na * nb;
                spectral += 16.0 * na * nb * (na + nb);
            }
        }
    }
    let q = (interactions as f64 * 9.0 * d).min(d * d);
    let terms = 30.0;
    let mut own: BTreeMap<&Vec<i32>, f64> = BTreeMap::new();
    for l in labels {
        *own.entry(l).or_default() += 1.0;
    }
    let left: f64 = own.values().map(|n| n * n).sum();
    let work = eigen + terms * (spectral + 64.0 * (q + y)) + 64.0 * left;
    let memory = terms * (q + y) * (20.0 + 24.0) + left * 20.0 + d * d * (3.0 * 16.0 + 24.0);
    (work, memory)
}

fn distance(p: [f64; 3], q: [f64; 3]) -> f64 {
    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
}

#[cfg(test)]
mod tests;
