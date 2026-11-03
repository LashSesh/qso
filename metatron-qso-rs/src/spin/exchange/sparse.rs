//! The joint generator of exchange on sparse blocks.
//!
//! The blocks of coherence order over all sites are kept as sparse
//! matrices, their entries those of the dense blocks, and propagated by
//! the Krylov steps of 1.41 when that costs less than a dense exponential
//! (always beyond [`MAX_LIOUVILLE_BLOCK`](super::super::MAX_LIOUVILLE_BLOCK)
//! elements). The drift of a zero-order block (its inhomogeneous term) is
//! an augmented coordinate scaled to the size of the deviation, so the
//! Krylov tolerance is relative to the deviation, not to `c_s/d_s`.
//!
//! Steady states solve the zero-order blocks by GMRES, preconditioned by
//! the generator in the eigenbasis of every site's Hamiltonian (see
//! [`secular`]).

use num_complex::Complex64 as C64;

use super::super::blocks::LiouvilleBlock;
use super::super::krylov::{
    Augmented, KRYLOV_STATIONARY, KRYLOV_TOLERANCE, SparseBlock, expv, krylov_bytes, krylov_work,
};
use super::super::{MAX_LIOUVILLE_BLOCK, MAX_RELAXATION_MEMORY, ZERO};
use super::{Exchange, Joint};
use crate::error::{QsoError, Result};
use crate::linalg::CMatrix;
use crate::linalg::dense::{expm, expm_squarings};

/// How the blocks of an exchange propagate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::spin) enum Propagation {
    /// Each block by whichever of a dense exponential and Krylov steps
    /// costs less (dense only up to `MAX_LIOUVILLE_BLOCK` elements).
    Auto,
    /// Dense exponentials (tests).
    Dense,
    /// Krylov steps (tests).
    Krylov,
}

/// Relative residual of GMRES on the scaled steady-state equations.
const GMRES_TOLERANCE: f64 = 1e-13;

/// Dimension of the Krylov subspace of GMRES before it restarts.
const GMRES_RESTART: usize = 60;

/// Most GMRES iterations.
const GMRES_ITERATIONS: usize = 20_000;

/// A joint block of the generator as a sparse matrix.
pub(super) struct SparseJoint {
    /// Elements.
    pub n: usize,
    /// The block, its diagonal less `shift`.
    pub a: SparseBlock,
    /// The mean precession subtracted from the diagonal (zero for
    /// zero-order blocks).
    pub shift: C64,
    /// The drift column of a zero-order block.
    pub drift: Option<Vec<C64>>,
}

/// `‖x‖₂`.
fn norm2(x: &[C64]) -> f64 {
    x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

impl SparseJoint {
    /// Coordinates of a dense exponential: the elements and the drift.
    fn dense_size(&self) -> usize {
        self.n + usize::from(self.drift.is_some())
    }

    /// `‖[A, drift]‖∞`.
    fn norm(&self) -> f64 {
        let mut rows: Vec<f64> = (0..self.n)
            .map(|r| self.a.row(r).map(|(_, v)| v.norm()).sum())
            .collect();
        if let Some(d) = &self.drift {
            for (r, v) in rows.iter_mut().zip(d) {
                *r += v.norm();
            }
        }
        rows.into_iter().fold(0.0, f64::max)
    }

    /// Operations of a dense exponential over `t`, if the block may be
    /// dense.
    fn dense_work(&self, t: f64) -> Option<f64> {
        let m = self.dense_size();
        (m <= MAX_LIOUVILLE_BLOCK).then(|| {
            // ‖·‖₁ ≤ m‖·‖∞ bounds the squarings from above.
            let norm = self.a.norm_one().max(self.norm()) * t;
            let m = m as f64;
            (6.0 + f64::from(expm_squarings(norm))) * 6.0 * m * m * m
        })
    }

    /// Operations of Krylov steps over `t`.
    fn krylov_work(&self, t: f64) -> f64 {
        let drift = self.drift.as_ref().map_or(0, |d| d.len());
        krylov_work(self.dense_size(), self.a.nnz() + drift, self.norm(), t)
    }

    /// The cheaper method over `t` and its operations.
    pub fn plan(&self, t: f64, method: Propagation) -> (bool, f64) {
        let krylov = self.krylov_work(t);
        match (method, self.dense_work(t)) {
            (Propagation::Krylov, _) | (_, None) => (false, krylov),
            (Propagation::Dense, Some(d)) => (true, d),
            (Propagation::Auto, Some(d)) => {
                if d <= krylov {
                    (true, d)
                } else {
                    (false, krylov)
                }
            }
        }
    }

    /// Bytes held while it propagates (densely: the matrix, its exponential
    /// and the exponential's workspace).
    pub fn bytes(&self, dense: bool) -> f64 {
        let m = self.dense_size() as f64;
        let own = self.a.bytes() as f64 + 16.0 * m;
        if dense {
            own + 6.0 * 16.0 * m * m
        } else {
            own + krylov_bytes(self.dense_size())
        }
    }

    /// The dense generator `[A, drift; 0, 0]` (shifted) times `t`.
    fn dense(&self, t: f64) -> CMatrix {
        let m = self.dense_size();
        let mut g = CMatrix::zeros(m, m);
        for r in 0..self.n {
            for (c, v) in self.a.row(r) {
                g[(r, c)] += v * t;
            }
        }
        if let Some(d) = &self.drift {
            for (r, v) in d.iter().enumerate() {
                g[(r, self.n)] = v * t;
            }
        }
        g
    }

    /// The dense block as [`LiouvilleBlock`] (for the dense code paths).
    pub fn liouville_block(&self, t: f64) -> LiouvilleBlock {
        LiouvilleBlock::shifted(self.n, self.dense(t), self.shift * t, self.drift.is_some())
    }

    /// The block's elements after `t` from `x`, densely or by Krylov steps.
    pub fn propagate(&self, x: &[C64], t: f64, dense: bool) -> Result<Vec<C64>> {
        debug_assert_eq!(x.len(), self.n);
        let phase = (self.shift * t).exp();
        if dense {
            let p = expm(&self.dense(t));
            let n = self.n;
            return Ok((0..n)
                .map(|i| {
                    let mut v = if self.drift.is_some() {
                        p[(i, n)]
                    } else {
                        ZERO
                    };
                    for (j, xj) in x.iter().enumerate() {
                        v += p[(i, j)] * xj;
                    }
                    v * phase
                })
                .collect());
        }
        match &self.drift {
            None => {
                let beta = norm2(x);
                if beta == 0.0 {
                    return Ok(x.to_vec());
                }
                let mut y = expv(&self.a, x, t, KRYLOV_TOLERANCE * beta, 0.0)?;
                if phase != C64::new(1.0, 0.0) {
                    for v in &mut y {
                        *v *= phase;
                    }
                }
                Ok(y)
            }
            Some(drift) => {
                // The augmented coordinate at the deviation's size (or its
                // equilibrium value, the drift over the rates).
                let rates = self.a.norm_inf().max(f64::MIN_POSITIVE);
                let s = norm2(x).max(norm2(drift) / rates);
                if s == 0.0 {
                    return Ok(x.to_vec());
                }
                let column: Vec<C64> = drift.iter().map(|v| v / s).collect();
                let op = Augmented::new(&self.a, ZERO, vec![column], vec![vec![ZERO]]);
                let mut v = x.to_vec();
                v.push(C64::new(s, 0.0));
                let tol = KRYLOV_TOLERANCE * norm2(&v);
                let mut y = expv(&op, &v, t, tol, KRYLOV_STATIONARY)?;
                y.truncate(self.n);
                Ok(y)
            }
        }
    }
}

impl Exchange {
    /// The joint blocks of the generator at field `b` as sparse matrices
    /// (`only`: a single coherence order), refused beyond
    /// [`MAX_RELAXATION_MEMORY`] before they are built.
    pub(super) fn sparse_generators(
        &self,
        b: f64,
        only: Option<&[i32]>,
    ) -> Result<(Joint, Vec<SparseJoint>)> {
        let joint = self.joint(only);
        let parts = self.site_parts(b)?;
        // Count the entries first: triplets and the matrix they become.
        let mut counts = vec![0usize; joint.blocks.len()];
        self.terms(&joint, b, &parts, |j, _, _, _| counts[j] += 1);
        let bytes: f64 = counts
            .iter()
            .zip(&joint.blocks)
            .map(|(&c, (_, _, n))| 44.0 * c as f64 + krylov_bytes(n + 1))
            .sum();
        if bytes > MAX_RELAXATION_MEMORY {
            return Err(QsoError::invalid(
                "exchange",
                format!(
                    "the joint blocks need about {:.1} GB (limit {:.1} GB); use fewer spins \
                     or the rotating frame",
                    bytes / 1e9,
                    MAX_RELAXATION_MEMORY / 1e9
                ),
            ));
        }
        let mut triplets: Vec<Vec<(u32, u32, C64)>> =
            counts.iter().map(|&c| Vec::with_capacity(c)).collect();
        let mut drifts: Vec<Option<Vec<C64>>> = joint
            .blocks
            .iter()
            .map(|(order, _, n)| order.iter().all(|&q| q == 0).then(|| vec![ZERO; *n]))
            .collect();
        self.terms(&joint, b, &parts, |j, r, c, v| {
            let n = joint.blocks[j].2;
            if c == n {
                if let Some(d) = drifts[j].as_mut() {
                    d[r] += v;
                }
            } else {
                triplets[j].push((r as u32, c as u32, v));
            }
        });
        let blocks = joint
            .blocks
            .iter()
            .zip(triplets)
            .zip(drifts)
            .map(|(((_, _, n), mut t), drift)| {
                let n = *n;
                let shift = if drift.is_some() || n == 0 {
                    ZERO
                } else {
                    let mean = t
                        .iter()
                        .filter(|&&(r, c, _)| r == c)
                        .map(|&(_, _, v)| v)
                        .sum::<C64>()
                        / n as f64;
                    C64::new(0.0, mean.im)
                };
                if shift != ZERO {
                    t.extend((0..n as u32).map(|i| (i, i, -shift)));
                }
                SparseJoint {
                    n,
                    a: SparseBlock::from_triplets(n, t),
                    shift,
                    drift,
                }
            })
            .collect();
        Ok((joint, blocks))
    }

    /// The steady state of the zero-order blocks on sparse matrices at
    /// field `b`: `G x = −drift` with one equation per site replaced by
    /// `Tr Δ_s = 0` (see [`secular`]), by GMRES preconditioned with the
    /// secular approximation.
    pub(super) fn sparse_steady(
        &self,
        joint: &Joint,
        blocks: &[SparseJoint],
        b: f64,
    ) -> Result<Vec<CMatrix>> {
        let mut out: Vec<CMatrix> = self
            .sites
            .iter()
            .map(|s| CMatrix::zeros(s.system.dimension(), s.system.dimension()))
            .collect();
        let refuse = || {
            QsoError::invalid(
                "steady state",
                "no unique steady state: some populations neither exchange nor relax",
            )
        };
        for (j, ((_, parts, n), block)) in joint.blocks.iter().zip(blocks).enumerate() {
            let n = *n;
            let Some(drift) = &block.drift else { continue };
            let want: Vec<C64> = drift.iter().map(|v| -v).collect();
            // Each site's populations: Tr Δ_s = 0 completes the equations.
            let mut diagonal: Vec<Vec<usize>> = vec![Vec::new(); self.sites.len()];
            let mut i = 0;
            for &(s, k, _) in parts {
                for &(r, c) in &joint.liouvilles[s].blocks[k].1 {
                    if r == c {
                        diagonal[s].push(i);
                    }
                    i += 1;
                }
            }
            let pre = Secular::new(self, joint, j, block, b)?;
            let apply = |x: &[C64], y: &mut [C64]| {
                block.a.apply(x, y);
                for (d, &rho) in diagonal.iter().zip(&pre.rho) {
                    let trace: C64 = d.iter().map(|&k| x[k]).sum::<C64>() * rho;
                    for &k in d {
                        y[k] += trace;
                    }
                }
            };
            let x = gmres(n, apply, |r| pre.apply(r), &want).ok_or_else(refuse)?;
            // The equations hold, relative to the generator applied to the
            // whole state (as the dense solve checks).
            let max_norm = |v: &[C64]| v.iter().fold(0.0f64, |m, z| m.max(z.norm()));
            let mut gx = vec![ZERO; n];
            block.a.apply(&x, &mut gx);
            let residual = gx
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).norm())
                .fold(0.0, f64::max);
            let level = self
                .sites
                .iter()
                .map(|s| s.concentration / s.system.dimension() as f64)
                .fold(0.0, f64::max);
            let scale = block.a.norm_inf() * (max_norm(&x) + level) + max_norm(&want);
            if residual.is_nan() || residual > 1e-8 * scale {
                return Err(refuse());
            }
            let mut i = 0;
            for &(s, k, _) in parts {
                for &(a2, b2) in &joint.liouvilles[s].blocks[k].1 {
                    out[s][(a2, b2)] = x[i];
                    i += 1;
                }
            }
        }
        Ok(out)
    }
}

/// `A x = b` by GMRES (`apply`: `y = A x`), preconditioned on the right by
/// `precondition`, restarted after as many basis vectors as a quarter of
/// the memory limit holds (at least [`GMRES_RESTART`]), to
/// [`GMRES_TOLERANCE`] of `‖b‖`; `None` if it does not converge within
/// [`GMRES_ITERATIONS`].
fn gmres(
    n: usize,
    apply: impl Fn(&[C64], &mut [C64]),
    precondition: impl Fn(&[C64]) -> Vec<C64>,
    b: &[C64],
) -> Option<Vec<C64>> {
    let target = GMRES_TOLERANCE * norm2(b);
    let mut x = vec![ZERO; n];
    if target == 0.0 {
        return Some(x);
    }
    let room = (MAX_RELAXATION_MEMORY / 4.0 / (16.0 * n.max(1) as f64)) as usize;
    let restart = room.max(GMRES_RESTART).min(n.max(1));
    let mut iterations = 0;
    let mut w = vec![ZERO; n];
    loop {
        apply(&x, &mut w);
        let r: Vec<C64> = b.iter().zip(&w).map(|(b, ax)| b - ax).collect();
        let beta = norm2(&r);
        if beta <= target {
            return Some(x);
        }
        if iterations >= GMRES_ITERATIONS {
            return None;
        }
        let mut basis: Vec<Vec<C64>> = vec![r.iter().map(|v| v / beta).collect()];
        let mut h: Vec<Vec<C64>> = Vec::new();
        let mut rotations: Vec<(f64, C64)> = Vec::new();
        let mut g = vec![ZERO; restart + 1];
        g[0] = C64::new(beta, 0.0);
        let mut k = 0;
        while k < restart && iterations < GMRES_ITERATIONS {
            iterations += 1;
            let z = precondition(&basis[k]);
            apply(&z, &mut w);
            let mut column = vec![ZERO; k + 2];
            for (i, v) in basis.iter().enumerate() {
                let hij: C64 = v.iter().zip(&w).map(|(a, b)| a.conj() * b).sum();
                column[i] = hij;
                for (x, b) in w.iter_mut().zip(v) {
                    *x -= hij * b;
                }
            }
            let s = norm2(&w);
            column[k + 1] = C64::new(s, 0.0);
            for (i, &(c, sn)) in rotations.iter().enumerate() {
                let (p, q) = (column[i], column[i + 1]);
                column[i] = p * c + sn * q;
                column[i + 1] = -sn.conj() * p + q * c;
            }
            let (p, q) = (column[k], column[k + 1]);
            let r = (p.norm_sqr() + q.norm_sqr()).sqrt();
            let (c, sn) = if r == 0.0 {
                (1.0, ZERO)
            } else if p.norm() == 0.0 {
                (0.0, C64::new(1.0, 0.0))
            } else {
                (p.norm() / r, (p / p.norm()) * q.conj() / r)
            };
            column[k] = p * c + sn * q;
            column[k + 1] = ZERO;
            g[k + 1] = -sn.conj() * g[k];
            g[k] *= c;
            rotations.push((c, sn));
            h.push(column);
            k += 1;
            if g[k].norm() <= target || s == 0.0 {
                break;
            }
            basis.push(w.iter().map(|v| v / s).collect());
        }
        // y from the triangle, x += M⁻¹ V y.
        let mut y = vec![ZERO; k];
        for i in (0..k).rev() {
            let mut acc = g[i];
            for j in i + 1..k {
                acc -= h[j][i] * y[j];
            }
            if h[i][i].norm() == 0.0 {
                return None;
            }
            y[i] = acc / h[i][i];
        }
        let mut update = vec![ZERO; n];
        for (v, yi) in basis.iter().zip(&y) {
            for (u, b) in update.iter_mut().zip(v) {
                *u += yi * b;
            }
        }
        for (xi, d) in x.iter_mut().zip(precondition(&update)) {
            *xi += d;
        }
    }
}

mod secular;

use secular::Secular;

#[cfg(test)]
pub(crate) mod tests;
