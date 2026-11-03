//! A preconditioner for the steady state of exchange: the zero-order
//! block in the eigenbasis of every site's Hamiltonian.
//!
//! The steady-state equations `G x = −f` are singular (the chemistry
//! conserves the amounts) and complete only with `Tr Δ_s = 0` for every
//! site; the solution is that of `(G + Σ_s ρ_s t_s t_sᵀ) x = −f`, `t_s`
//! the trace of site `s` and `ρ_s` its fastest rate over its dimension,
//! which is not singular. In the eigenbasis of every site's Hamiltonian
//! `−i[H_s, ·]` is diagonal, `−iω_αβ` on `|α⟩⟨β|`, and `t_s` lies on the
//! populations. The slow elements (the populations of the eigenstates and
//! the coherences with `|ω_αβ|` within ten times the site's fastest rate)
//! of all sites are solved together with the exact generator restricted to
//! them and the trace terms (dense LU); every other coherence is divided by
//! `−iω_αβ − γ̄_s`, `γ̄_s` the site's mean decay. This is exact where the
//! frequencies dominate the rates (the secular regime) and where they
//! vanish; GMRES corrects the rest.

use nalgebra::DVector;
use num_complex::Complex64 as C64;

use super::super::super::ZERO;
use super::super::super::blocks::Eigenblock;
use super::super::{Exchange, Joint};
use super::SparseJoint;
use crate::error::{QsoError, Result};
use crate::linalg::CMatrix;
use crate::parallel::map_tasks;

/// Coherences whose frequency is within this many times the site's
/// fastest rate are slow.
const SLOW: f64 = 10.0;

/// Most slow elements solved densely.
const MAX_SLOW: usize = 4096;

/// One eigenblock of a site: its states' positions in the joint block
/// (`at[i·m + j]` for `|s_i⟩⟨s_j|`), eigenvectors and energies.
struct Part {
    site: usize,
    m: usize,
    at: Vec<usize>,
    vectors: CMatrix,
    values: Vec<f64>,
}

impl Part {
    /// `V† X V` of the block's elements of `x`.
    fn to_eigen(&self, x: &[C64]) -> CMatrix {
        let m = self.m;
        let block = CMatrix::from_fn(m, m, |i, j| x[self.at[i * m + j]]);
        self.vectors.adjoint() * block * &self.vectors
    }

    /// `V Y V†` into the block's elements of `out`.
    fn back_from_eigen(&self, y: &CMatrix, out: &mut [C64]) {
        let m = self.m;
        let back = &self.vectors * y * self.vectors.adjoint();
        for i in 0..m {
            for j in 0..m {
                out[self.at[i * m + j]] = back[(i, j)];
            }
        }
    }
}

/// The preconditioner of one zero-order joint block.
pub(super) struct Secular {
    n: usize,
    parts: Vec<Part>,
    /// Mean decay of each site's coherences.
    decay: Vec<f64>,
    /// The slow elements `(part, α, β)` and the factorised generator on
    /// them, trace rows included.
    slow: Vec<(usize, usize, usize)>,
    lu: nalgebra::LU<C64, nalgebra::Dyn, nalgebra::Dyn>,
    /// `ρ_s` of each site.
    pub rho: Vec<f64>,
}

impl Secular {
    /// From the joint block `j` of `joint` at field `b`.
    pub fn new(x: &Exchange, joint: &Joint, j: usize, block: &SparseJoint, b: f64) -> Result<Self> {
        let n = block.n;
        let (_, parts_of, _) = &joint.blocks[j];
        let mut parts = Vec::new();
        let mut decay = vec![0.0f64; x.sites.len()];
        let mut fastest = vec![0.0f64; x.sites.len()];
        for &(s, k, offset) in parts_of {
            let sys = &x.sites[s].system;
            let l = &joint.liouvilles[s];
            let elements = &l.blocks[k].1;
            let (mut sum, mut count) = (0.0, 0usize);
            for &(a, c) in elements {
                let i = offset + l.position_of(a, c);
                let rate = -block.a.get(i, i).re;
                fastest[s] = fastest[s].max(rate.abs());
                if a != c {
                    sum += rate;
                    count += 1;
                }
            }
            decay[s] = if count > 0 {
                sum / count as f64
            } else {
                fastest[s]
            };
            let blocks: Vec<Eigenblock> = sys.eigenblocks(&sys.hamiltonian_at(b));
            for e in blocks {
                let m = e.states.len();
                let mut at = Vec::with_capacity(m * m);
                for &p in &e.states {
                    for &q in &e.states {
                        if l.block_of(p, q) != Some(k) {
                            return Err(QsoError::invalid(
                                "steady state",
                                "the Hamiltonian mixes coherence orders",
                            ));
                        }
                        at.push(offset + l.position_of(p, q));
                    }
                }
                parts.push(Part {
                    site: s,
                    m,
                    at,
                    vectors: e.vectors,
                    values: e.values,
                });
            }
        }
        // The slow elements: populations first, then the slowest
        // coherences up to the limit.
        let mut slow: Vec<(usize, usize, usize)> = Vec::new();
        let mut coherences: Vec<(f64, (usize, usize, usize))> = Vec::new();
        for (p, part) in parts.iter().enumerate() {
            let cut = SLOW * fastest[part.site];
            for a in 0..part.m {
                for c in 0..part.m {
                    let w = (part.values[a] - part.values[c]).abs();
                    if a == c {
                        slow.push((p, a, c));
                    } else if w <= cut {
                        coherences.push((w, (p, a, c)));
                    }
                }
            }
        }
        if slow.len() > MAX_SLOW {
            return Err(QsoError::invalid(
                "steady state",
                format!(
                    "{} populations of eigenstates (at most {MAX_SLOW})",
                    slow.len()
                ),
            ));
        }
        coherences.sort_by(|a, b| a.0.total_cmp(&b.0));
        slow.extend(
            coherences
                .into_iter()
                .take(MAX_SLOW - slow.len())
                .map(|(_, e)| e),
        );
        let rho: Vec<f64> = x
            .sites
            .iter()
            .zip(&fastest)
            .map(|(site, &f)| f.max(f64::MIN_POSITIVE) / site.system.dimension() as f64)
            .collect();
        // The generator on the slow elements: G applied to each, read back
        // in the eigenbasis.
        let m = slow.len();
        let mut index = std::collections::HashMap::new();
        for (q, &e) in slow.iter().enumerate() {
            index.insert(e, q);
        }
        let columns: Vec<Vec<(usize, C64)>> = map_tasks(m, |q| {
            let (p, a, c) = slow[q];
            let part = &parts[p];
            let mut e = vec![ZERO; n];
            for i in 0..part.m {
                for k in 0..part.m {
                    e[part.at[i * part.m + k]] = part.vectors[(i, a)] * part.vectors[(k, c)].conj();
                }
            }
            let mut y = vec![ZERO; n];
            block.a.apply(&e, &mut y);
            let mut out = Vec::new();
            for (r, other) in parts.iter().enumerate() {
                let t = other.to_eigen(&y);
                for a2 in 0..other.m {
                    for c2 in 0..other.m {
                        if let Some(&row) = index.get(&(r, a2, c2)) {
                            let v = t[(a2, c2)];
                            if v != ZERO {
                                out.push((row, v));
                            }
                        }
                    }
                }
            }
            out
        });
        let mut k = CMatrix::zeros(m, m);
        for (q, column) in columns.into_iter().enumerate() {
            for (row, v) in column {
                k[(row, q)] = v;
            }
        }
        // ρ_s t_s t_sᵀ on the populations of each site (the trace is the
        // same in every basis).
        for (q, &(p, a, c)) in slow.iter().enumerate() {
            if a != c {
                continue;
            }
            let site = parts[p].site;
            for (q2, &(p2, a2, c2)) in slow.iter().enumerate() {
                if a2 == c2 && parts[p2].site == site {
                    k[(q, q2)] += C64::new(rho[site], 0.0);
                }
            }
        }
        Ok(Secular {
            n,
            parts,
            decay,
            slow,
            lu: k.lu(),
            rho,
        })
    }

    /// The approximate solution `z` of `(G + Σ_s ρ_s t_s t_sᵀ) z = r`.
    pub fn apply(&self, r: &[C64]) -> Vec<C64> {
        let mut eigen: Vec<CMatrix> = self.parts.iter().map(|p| p.to_eigen(r)).collect();
        // Fast coherences.
        for (part, y) in self.parts.iter().zip(eigen.iter_mut()) {
            let gamma = self.decay[part.site];
            for a in 0..part.m {
                for c in 0..part.m {
                    if a != c {
                        let w = part.values[a] - part.values[c];
                        y[(a, c)] /= C64::new(-gamma, -w);
                    }
                }
            }
        }
        // Slow elements, exactly.
        let rhs = DVector::from_iterator(
            self.slow.len(),
            self.slow.iter().map(|&(p, a, c)| {
                let part = &self.parts[p];
                let gamma = self.decay[part.site];
                let w = part.values[a] - part.values[c];
                // Undo the division above for slow coherences.
                if a == c {
                    eigen[p][(a, c)]
                } else {
                    eigen[p][(a, c)] * C64::new(-gamma, -w)
                }
            }),
        );
        if let Some(w) = self.lu.solve(&rhs) {
            for (&(p, a, c), v) in self.slow.iter().zip(w.iter()) {
                eigen[p][(a, c)] = *v;
            }
        }
        let mut out = vec![ZERO; self.n];
        for (part, y) in self.parts.iter().zip(&eigen) {
            part.back_from_eigen(y, &mut out);
        }
        out
    }
}
