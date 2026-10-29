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
//! `ρ_eq` is the thermal state, so relaxation ends there at any
//! temperature. In the rotating frame only the part of `R` that keeps the
//! coherence order of every isotope is kept (the secular approximation
//! with respect to the carriers); in the laboratory frame `R` keeps the
//! total coherence order exactly.

use num_complex::Complex64 as C64;

use super::blocks::{Eigenblock, Liouville};
use super::{Frame, HBAR, SpinSystem, ZERO};
use crate::error::{QsoError, Result};
use crate::linalg::dense::cmul;
use crate::linalg::{CMatrix, RMatrix};

/// `μ₀/4π` (T² m³ J⁻¹, CODATA 2018).
pub const MU0_OVER_4PI: f64 = 1.000_000_000_55e-7;

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

/// The Redfield superoperator block by block of coherence order.
pub(super) struct RedfieldBlocks {
    /// `(elements, R restricted to them)`.
    pub blocks: Vec<(Vec<(usize, usize)>, CMatrix)>,
}

impl RedfieldBlocks {
    /// `out += scale · R(x)`.
    pub fn apply(&self, x: &CMatrix, out: &mut CMatrix, scale: f64) {
        for (elements, r) in &self.blocks {
            let v: Vec<C64> = elements.iter().map(|&(a, b)| x[(a, b)]).collect();
            if v.iter().all(|z| *z == ZERO) {
                continue;
            }
            for (i, &(a, b)) in elements.iter().enumerate() {
                let mut s = ZERO;
                for (j, vj) in v.iter().enumerate() {
                    s += r[(i, j)] * vj;
                }
                out[(a, b)] += s * scale;
            }
        }
    }

    /// The largest column sum of `|R|`: a bound on its rates.
    pub fn rate(&self) -> f64 {
        self.blocks
            .iter()
            .map(|(_, r)| crate::linalg::dense::one_norm(r))
            .fold(0.0, f64::max)
    }

    /// Entries of all blocks (the work of one application).
    pub fn entries(&self) -> usize {
        self.blocks.iter().map(|(e, _)| e.len() * e.len()).sum()
    }
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

    /// `Y(X)_mn = X_mn J(ω_mn)` in the eigenbasis of the laboratory `H₀`.
    fn spectral(x: &CMatrix, blocks: &[Eigenblock], tau: f64) -> CMatrix {
        let d = x.nrows();
        let mut out = CMatrix::zeros(d, d);
        for a in blocks {
            for b in blocks {
                let sub = CMatrix::from_fn(a.states.len(), b.states.len(), |r, c| {
                    x[(a.states[r], b.states[c])]
                });
                if sub.iter().all(|z| *z == ZERO) {
                    continue;
                }
                let mut t = cmul(&cmul(&a.vectors.adjoint(), &sub), &b.vectors);
                for (r, &ea) in a.values.iter().enumerate() {
                    for (c, &eb) in b.values.iter().enumerate() {
                        let w = (a.shift - b.shift) + ea - eb;
                        t[(r, c)] *= tau / (1.0 + w * w * tau * tau);
                    }
                }
                let back = cmul(&cmul(&a.vectors, &t), &b.vectors.adjoint());
                for (r, &i) in a.states.iter().enumerate() {
                    for (c, &j) in b.states.iter().enumerate() {
                        out[(i, j)] += back[(r, c)];
                    }
                }
            }
        }
        out
    }

    /// The Redfield superoperator at field `b`, block by block of the
    /// coherence orders of the labels (`None` without motion or
    /// interactions). Blocks above [`super::MAX_LIOUVILLE_BLOCK`] are an
    /// error.
    pub(super) fn redfield(&self, b: f64) -> Result<Option<RedfieldBlocks>> {
        let Some(tau) = self.correlation_time else {
            return Ok(None);
        };
        let interactions = self.interactions(b);
        if interactions.is_empty() {
            return Ok(None);
        }
        let labels = self.labels();
        let largest = super::blocks::largest_coherence_block(&labels);
        if largest > super::MAX_LIOUVILLE_BLOCK {
            return Err(QsoError::invalid(
                "redfield",
                format!(
                    "Redfield relaxation at dimension {} needs blocks of {largest} elements \
                     (at most {})",
                    self.dimension(),
                    super::MAX_LIOUVILLE_BLOCK
                ),
            ));
        }
        let d = self.dimension();
        let ops: Vec<[CMatrix; 3]> = (0..self.spins.len())
            .map(|i| {
                [
                    self.ix(i).expect("spin"),
                    self.iy(i).expect("spin"),
                    self.iz(i).expect("spin"),
                ]
            })
            .collect();
        // Q^S_ab of every interaction, for a ≤ b.
        let pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
        let spin_tensor = |c: Coupling| -> Vec<CMatrix> {
            match c {
                Coupling::Dipolar(i, j) => {
                    let dot =
                        &ops[i][0] * &ops[j][0] + &ops[i][1] * &ops[j][1] + &ops[i][2] * &ops[j][2];
                    pairs
                        .iter()
                        .map(|&(x, y)| {
                            let mut q = (&ops[i][x] * &ops[j][y] + &ops[i][y] * &ops[j][x])
                                * C64::new(0.5, 0.0);
                            if x == y {
                                q -= &dot * C64::new(1.0 / 3.0, 0.0);
                            }
                            q
                        })
                        .collect()
                }
                Coupling::Shielding(i) => pairs
                    .iter()
                    .map(|&(x, y)| {
                        let mut q = CMatrix::zeros(d, d);
                        if y == 2 {
                            q += &ops[i][x] * C64::new(0.5, 0.0);
                        }
                        if x == 2 {
                            q += &ops[i][y] * C64::new(0.5, 0.0);
                        }
                        if x == y {
                            q -= &ops[i][2] * C64::new(1.0 / 3.0, 0.0);
                        }
                        q
                    })
                    .collect(),
            }
        };
        let tensors: Vec<Vec<CMatrix>> =
            interactions.iter().map(|&(_, c)| spin_tensor(c)).collect();
        let e = basis();
        let mut lab = self.clone();
        lab.frame = Frame::Laboratory;
        let eigen = lab.eigenblocks(&lab.hamiltonian_at(b));
        // Terms (𝒬, Y(𝒬), weight).
        let mut terms: Vec<(CMatrix, CMatrix, f64)> = Vec::new();
        for basis_tensor in &e {
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
                let mut q = CMatrix::zeros(d, d);
                for (k, t) in tensors.iter().enumerate() {
                    if alpha[k] != 0.0 {
                        q += &t[p] * C64::new(alpha[k], 0.0);
                    }
                }
                let weight = if x == y { 1.0 } else { 2.0 } / 5.0;
                let yq = Self::spectral(&q, &eigen, tau);
                terms.push((q, yq, weight));
            }
        }
        // R(|a⟩⟨b|) = Σ w (QY|a⟩⟨b| − Q|a⟩⟨b|Y − Y|a⟩⟨b|Q + |a⟩⟨b|YQ).
        let mut left = CMatrix::zeros(d, d);
        let mut right = CMatrix::zeros(d, d);
        for (q, y, w) in &terms {
            left += cmul(q, y) * C64::new(*w, 0.0);
            right += cmul(y, q) * C64::new(*w, 0.0);
        }
        let liouville = Liouville::new(&labels, None);
        let mut blocks = Vec::with_capacity(liouville.blocks.len());
        for (_, elements) in &liouville.blocks {
            let n = elements.len();
            let mut r = CMatrix::zeros(n, n);
            for (j, &(a, b)) in elements.iter().enumerate() {
                for (i, &(c, e)) in elements.iter().enumerate() {
                    let mut v = ZERO;
                    if e == b {
                        v += left[(c, a)];
                    }
                    if c == a {
                        v += right[(b, e)];
                    }
                    for (q, y, w) in &terms {
                        v -= (q[(c, a)] * y[(b, e)] + y[(c, a)] * q[(b, e)]) * *w;
                    }
                    r[(i, j)] = v;
                }
            }
            blocks.push((elements.clone(), r));
        }
        Ok(Some(RedfieldBlocks { blocks }))
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
                out[(i, j)] = -t.re / norm;
            }
        }
        Ok(out)
    }
}

fn distance(p: [f64; 3], q: [f64; 3]) -> f64 {
    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
}

#[cfg(test)]
mod tests;
