//! Optimal control of spin experiments: gradient ascent pulse engineering
//! (GRAPE; Khaneja, Reiss, Kehlet, Schulte-Herbrüggen and Glaser,
//! J. Magn. Reson. 172, 296 (2005)).
//!
//! The controls are piecewise constant over `steps` equal steps of a fixed
//! duration: the `x` and `y` amplitudes (Hz, as nutation frequencies) of
//! radiofrequency fields on channels in the rotating frame, or the field
//! itself (T). GRAPE maximises the transfer `Re Tr(C† ρ(T))` from an initial
//! operator `ρ(0)` to a target `C`, averaged over an ensemble of rf
//! scalings and offsets, less penalties on power and roughness. Its
//! gradients are exact up to rounding:
//!
//! - without relaxation `ρ` evolves under `U_k = e^{−iH_kΔt}`, and the
//!   derivative of `U_k` follows from the eigenvectors of `H_k` (divided
//!   differences of `e^{−iEΔt}`);
//! - with relaxation, in Liouville space, the derivative of `e^{L_kΔt}`
//!   applied to the state is the corner of the exponential of
//!   `[[L_kΔt, ∂L_kΔt], [0, L_kΔt]]` (Van Loan, IEEE Trans. Autom.
//!   Control 23, 395 (1978)), summed as a Taylor series of matrix-vector
//!   products to rounding.
//!
//! Field profiles with relaxation take the derivative of the relaxation
//! with respect to the field (thermal populations, spectral densities) by
//! central differences, to about 10⁻⁹ of it; the rest is exact.

use nalgebra::DVector;
use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::blocks::{Liouville, Sparse};
use super::{Event, Frame, SpinSystem, TWO_PI, Target, ZERO, check_time};
use crate::error::{QsoError, Result};
use crate::linalg::CMatrix;
use crate::optimize::{GradObjective, Method, Options, minimize};

/// What the controls drive.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
pub enum Drive {
    /// Radiofrequency fields on channels (rotating frame): an `x` and a `y`
    /// amplitude (Hz) per channel and step, each pair at most `limit` Hz
    /// (`√(x² + y²)`) if given.
    Rf {
        targets: Vec<Target>,
        limit: Option<f64>,
    },
    /// The field (T) of every step, between `min` and `max`.
    Field { min: f64, max: f64 },
}

/// A member of the ensemble the controls must work for: rf amplitudes
/// scaled per channel (empty: all 1) and the spins of each channel shifted
/// by an offset (Hz, `H += 2πΔF_z`; empty: none), with a weight.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Member {
    pub weight: f64,
    #[serde(default)]
    pub scales: Vec<f64>,
    #[serde(default)]
    pub offsets: Vec<f64>,
}

impl Member {
    /// The ideal member: weight 1, no scaling, no offsets.
    pub fn ideal() -> Self {
        Member {
            weight: 1.0,
            scales: Vec::new(),
            offsets: Vec::new(),
        }
    }
}

/// An optimal-control problem: drive `initial` towards `target` in
/// `duration` s of `steps` constant pieces.
#[derive(Clone, Debug, PartialEq)]
pub struct Control {
    pub drive: Drive,
    pub duration: f64,
    pub steps: usize,
    pub initial: CMatrix,
    pub target: CMatrix,
    /// Empty: the ideal member alone.
    pub ensemble: Vec<Member>,
    /// Weight of `Σ u² Δt` (Hz² s or T² s) subtracted from the transfer.
    pub power: f64,
    /// Weight of `Σ (u_{k+1} − u_k)²` (Hz² or T²) subtracted from the
    /// transfer.
    pub smoothness: f64,
}

/// The outcome of [`SpinSystem::optimize_control`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Optimized {
    /// Physical controls (Hz or T), as [`SpinSystem::control_transfer`]
    /// takes them.
    pub controls: Vec<f64>,
    /// The transfer averaged over the ensemble.
    pub transfer: f64,
    /// The transfer of each member.
    pub members: Vec<f64>,
    /// Transfer less penalties.
    pub objective: f64,
    pub iterations: usize,
    pub evaluations: usize,
    pub converged: bool,
    pub message: String,
}

impl Control {
    /// Radiofrequency controls on `targets`.
    pub fn rf(
        targets: Vec<Target>,
        limit: Option<f64>,
        duration: f64,
        steps: usize,
        initial: CMatrix,
        target: CMatrix,
    ) -> Self {
        Self::new(
            Drive::Rf { targets, limit },
            duration,
            steps,
            initial,
            target,
        )
    }

    /// A field profile between `min` and `max` (T).
    pub fn field(
        min: f64,
        max: f64,
        duration: f64,
        steps: usize,
        initial: CMatrix,
        target: CMatrix,
    ) -> Self {
        Self::new(Drive::Field { min, max }, duration, steps, initial, target)
    }

    fn new(drive: Drive, duration: f64, steps: usize, initial: CMatrix, target: CMatrix) -> Self {
        Control {
            drive,
            duration,
            steps,
            initial,
            target,
            ensemble: Vec::new(),
            power: 0.0,
            smoothness: 0.0,
        }
    }

    /// Controls per step: `2·channels` (x and y) or 1 (the field).
    pub fn per_step(&self) -> usize {
        match &self.drive {
            Drive::Rf { targets, .. } => 2 * targets.len(),
            Drive::Field { .. } => 1,
        }
    }

    /// Number of controls.
    pub fn len(&self) -> usize {
        self.steps * self.per_step()
    }

    /// Whether there are no controls.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Length of one step (s).
    pub fn step(&self) -> f64 {
        self.duration / self.steps as f64
    }

    fn members(&self) -> Vec<Member> {
        if self.ensemble.is_empty() {
            vec![Member::ideal()]
        } else {
            self.ensemble.clone()
        }
    }

    /// Seeded random controls: amplitudes uniform in a disc of radius
    /// `scale` Hz (rf), or fields uniform in `[min, max]`.
    pub fn random_start(&self, seed: u64, scale: f64) -> Vec<f64> {
        let mut rng = crate::rng::Rng::seed_from_u64(seed);
        match &self.drive {
            Drive::Rf { limit, .. } => {
                let r = limit.map_or(scale, |l| scale.min(0.9 * l));
                let mut out = Vec::with_capacity(self.len());
                while out.len() < self.len() {
                    let (x, y) = (2.0 * rng.uniform() - 1.0, 2.0 * rng.uniform() - 1.0);
                    if x * x + y * y <= 1.0 {
                        out.extend([r * x, r * y]);
                    }
                }
                out
            }
            Drive::Field { min, max } => (0..self.len())
                .map(|_| min + (max - min) * rng.uniform())
                .collect(),
        }
    }

    /// The penalties and their gradient.
    pub fn penalty(&self, u: &[f64]) -> (f64, Vec<f64>) {
        let (n, dt) = (self.per_step(), self.step());
        let mut value = 0.0;
        let mut grad = vec![0.0; u.len()];
        for (i, &x) in u.iter().enumerate() {
            value += self.power * x * x * dt;
            grad[i] += 2.0 * self.power * x * dt;
            if i + n < u.len() {
                let d = u[i + n] - x;
                value += self.smoothness * d * d;
                grad[i] -= 2.0 * self.smoothness * d;
                grad[i + n] += 2.0 * self.smoothness * d;
            }
        }
        (value, grad)
    }

    /// Physical controls from the optimiser's free variables, and the
    /// derivatives `∂u/∂v` (per pair for rf: `[[∂x/∂v₁, ∂x/∂v₂], [∂y/∂v₁,
    /// ∂y/∂v₂]]`, flattened; per field: one). The free variables have the
    /// units of the controls, and the map is the identity near zero
    /// amplitude (or the middle of the field range), so that the
    /// optimiser's steps keep their scale.
    fn physical(&self, v: &[f64]) -> (Vec<f64>, Vec<f64>) {
        match &self.drive {
            Drive::Rf { limit: None, .. } => (v.to_vec(), Vec::new()),
            Drive::Rf { limit: Some(l), .. } => {
                let mut u = vec![0.0; v.len()];
                let mut jac = vec![0.0; 2 * v.len()];
                for (p, pair) in v.chunks(2).enumerate() {
                    let (a, b) = (pair[0], pair[1]);
                    // u = v·F(s), F(s) = tanh(s)/s, s = |v|/l; G = F'(s)/s.
                    let s = a.hypot(b) / l;
                    let (f, g) = if s < 1e-4 {
                        (1.0 - s * s / 3.0, -2.0 / 3.0 + 8.0 * s * s / 15.0)
                    } else {
                        let t = s.tanh();
                        (t / s, ((1.0 - t * t) * s - t) / (s * s * s))
                    };
                    let g = g / (l * l);
                    u[2 * p] = f * a;
                    u[2 * p + 1] = f * b;
                    jac[4 * p..4 * p + 4].copy_from_slice(&[
                        f + g * a * a,
                        g * a * b,
                        g * a * b,
                        f + g * b * b,
                    ]);
                }
                (u, jac)
            }
            Drive::Field { min, max } => {
                let (mid, half) = ((max + min) / 2.0, (max - min) / 2.0);
                let u = v
                    .iter()
                    .map(|&x| mid + half * ((x - mid) / half).tanh())
                    .collect();
                let jac = v
                    .iter()
                    .map(|&x| 1.0 - ((x - mid) / half).tanh().powi(2))
                    .collect();
                (u, jac)
            }
        }
    }

    /// Free variables of physical controls (inside the limits).
    fn free(&self, u: &[f64]) -> Vec<f64> {
        match &self.drive {
            Drive::Rf { limit: None, .. } => u.to_vec(),
            Drive::Rf { limit: Some(l), .. } => {
                let mut v = vec![0.0; u.len()];
                for (p, pair) in u.chunks(2).enumerate() {
                    // |u| = l·tanh(s), v = u·s/tanh(s).
                    let t = (pair[0].hypot(pair[1]) / l).min(1.0 - 1e-9);
                    let k = if t < 1e-12 { 1.0 } else { t.atanh() / t };
                    v[2 * p] = k * pair[0];
                    v[2 * p + 1] = k * pair[1];
                }
                v
            }
            Drive::Field { min, max } => {
                let (mid, half) = ((max + min) / 2.0, (max - min) / 2.0);
                u.iter()
                    .map(|&b| {
                        mid + half * ((b - mid) / half).clamp(-1.0 + 1e-9, 1.0 - 1e-9).atanh()
                    })
                    .collect()
            }
        }
    }

    /// Gradient with respect to the free variables from that with respect
    /// to the physical controls.
    fn chain(&self, jac: &[f64], g: &[f64]) -> Vec<f64> {
        match &self.drive {
            Drive::Rf { limit: None, .. } => g.to_vec(),
            Drive::Rf { limit: Some(_), .. } => {
                let mut out = vec![0.0; g.len()];
                for p in 0..g.len() / 2 {
                    let j = &jac[4 * p..4 * p + 4];
                    out[2 * p] = g[2 * p] * j[0] + g[2 * p + 1] * j[2];
                    out[2 * p + 1] = g[2 * p] * j[1] + g[2 * p + 1] * j[3];
                }
                out
            }
            Drive::Field { .. } => g.iter().zip(jac).map(|(a, b)| a * b).collect(),
        }
    }
}

/// A sparse matrix as `(row, column, value)` entries.
#[derive(Clone, Debug, Default)]
struct Entries(Vec<(usize, usize, C64)>);

impl Entries {
    fn from_dense(m: &CMatrix) -> Self {
        let mut out = Vec::new();
        for c in 0..m.ncols() {
            for r in 0..m.nrows() {
                if m[(r, c)] != ZERO {
                    out.push((r, c, m[(r, c)]));
                }
            }
        }
        Entries(out)
    }

    /// `y += s·M x`.
    fn apply_add(&self, x: &DVector<C64>, s: C64, y: &mut DVector<C64>) {
        for &(r, c, v) in &self.0 {
            y[r] += s * v * x[c];
        }
    }

    fn one_norm(&self, n: usize) -> f64 {
        let mut col = vec![0.0; n];
        for &(_, c, v) in &self.0 {
            col[c] += v.norm();
        }
        col.into_iter().fold(0.0, f64::max)
    }
}

/// `−i[A, ·]` on row-major vectors of `d × d` matrices.
fn commutator(a: &CMatrix) -> Entries {
    let d = a.nrows();
    let mi = C64::new(0.0, -1.0);
    let mut out = Vec::new();
    for r in 0..d {
        for c in 0..d {
            let v = a[(r, c)];
            if v == ZERO {
                continue;
            }
            for k in 0..d {
                // (Aρ)_{rk} = Σ_c A_rc ρ_ck; (ρA)_{kc} = Σ_r ρ_kr A_rc.
                out.push((r * d + k, c * d + k, mi * v));
                out.push((k * d + c, k * d + r, -mi * v));
            }
        }
    }
    Entries(out)
}

/// A linear generator `L = dense + Σ_j u_j ops_j`, applied to vectors.
struct Generator<'a> {
    dense: &'a CMatrix,
    ops: &'a [Entries],
    u: &'a [f64],
    /// `L†` instead of `L`.
    adjoint: bool,
}

impl Generator<'_> {
    /// `y = t·L x`.
    fn apply(&self, x: &DVector<C64>, t: f64) -> DVector<C64> {
        let mut y = if self.adjoint {
            self.dense.ad_mul(x)
        } else {
            self.dense * x
        };
        y *= C64::new(t, 0.0);
        for (op, &u) in self.ops.iter().zip(self.u) {
            if u == 0.0 {
                continue;
            }
            if self.adjoint {
                for &(r, c, v) in &op.0 {
                    y[c] += v.conj() * x[r] * (u * t);
                }
            } else {
                op.apply_add(x, C64::new(u * t, 0.0), &mut y);
            }
        }
        y
    }

    fn one_norm(&self, norms: &[f64], dense_norm: f64) -> f64 {
        dense_norm
            + norms
                .iter()
                .zip(self.u)
                .map(|(n, u)| n * u.abs())
                .sum::<f64>()
    }
}

fn max_abs(v: &DVector<C64>) -> f64 {
    v.iter().fold(0.0f64, |m, z| m.max(z.norm()))
}

/// Substeps for a Taylor series of an exponential of 1-norm `norm`.
fn substeps(norm: f64) -> usize {
    (norm / 0.5).ceil().max(1.0) as usize
}

/// `e^{tL} x` by Taylor series in `substeps(t‖L‖)` pieces, each summed to
/// rounding.
fn expmv(g: &Generator<'_>, x: &DVector<C64>, t: f64, norm: f64) -> DVector<C64> {
    let s = substeps(t * norm);
    let h = t / s as f64;
    let mut v = x.clone();
    for _ in 0..s {
        let mut term = v.clone();
        let mut acc = v.clone();
        for k in 1..100 {
            term = g.apply(&term, h / k as f64);
            acc += &term;
            if max_abs(&term) <= f64::EPSILON * 1e-2 * max_abs(&acc) {
                break;
            }
        }
        v = acc;
    }
    v
}

/// `Re λ† ∂(e^{tL} x)/∂u_j` for every direction `E_j = ∂L/∂u_j`: the
/// upper block of `exp(t[[L, E_j], [0, L]])` applied to `[0; x]`. That
/// block is linear in `E_j`, so the substeps follow `t‖L‖` alone.
fn frechet(
    g: &Generator<'_>,
    directions: &[&Entries],
    x: &DVector<C64>,
    lambda: &DVector<C64>,
    t: f64,
    norm: f64,
) -> Vec<f64> {
    let s = substeps(t * norm);
    let h = t / s as f64;
    let mut y = x.clone();
    let mut parts: Vec<DVector<C64>> = vec![DVector::zeros(x.len()); directions.len()];
    for _ in 0..s {
        let mut ty = y.clone();
        let mut tx = parts.clone();
        let mut acc_y = y.clone();
        let mut acc_x = parts.clone();
        for k in 1..100 {
            let f = h / k as f64;
            let mut small = true;
            for (j, e) in directions.iter().enumerate() {
                let mut next = g.apply(&tx[j], f);
                e.apply_add(&ty, C64::new(f, 0.0), &mut next);
                acc_x[j] += &next;
                small &= max_abs(&next) <= f64::EPSILON * 1e-2 * max_abs(&acc_x[j]);
                tx[j] = next;
            }
            ty = g.apply(&ty, f);
            acc_y += &ty;
            small &= max_abs(&ty) <= f64::EPSILON * 1e-2 * max_abs(&acc_y);
            if small {
                break;
            }
        }
        y = acc_y;
        parts = acc_x;
    }
    parts.iter().map(|p| lambda.dotc(p).re).collect()
}

fn vectorise(m: &CMatrix) -> DVector<C64> {
    let d = m.nrows();
    DVector::from_iterator(d * d, (0..d * d).map(|i| m[(i / d, i % d)]))
}

impl SpinSystem {
    /// Check a control problem against the system.
    fn check_control(&self, c: &Control) -> Result<()> {
        self.validate()?;
        let d = self.dimension();
        for m in [&c.initial, &c.target] {
            if m.nrows() != d || m.ncols() != d {
                return Err(QsoError::DimensionMismatch {
                    expected: d,
                    actual: m.nrows(),
                });
            }
        }
        check_time("duration", c.duration)?;
        if c.duration == 0.0 || c.steps == 0 || c.steps > super::MAX_SEGMENTS {
            return Err(QsoError::invalid(
                "control",
                format!("a positive duration in 1..={} steps", super::MAX_SEGMENTS),
            ));
        }
        if !(c.power >= 0.0
            && c.power.is_finite()
            && c.smoothness >= 0.0
            && c.smoothness.is_finite())
        {
            return Err(QsoError::invalid(
                "control",
                "penalties must be finite and ≥ 0",
            ));
        }
        match &c.drive {
            Drive::Rf { targets, limit } => {
                if self.frame() != Frame::Rotating {
                    return Err(QsoError::invalid(
                        "frame",
                        "rf controls need the rotating frame",
                    ));
                }
                if targets.is_empty() {
                    return Err(QsoError::invalid("control", "no rf channel"));
                }
                for &t in targets {
                    self.targets(t)?;
                }
                if limit.is_some_and(|l| !(l.is_finite() && l > 0.0)) {
                    return Err(QsoError::invalid("limit", "a positive amplitude limit"));
                }
                for m in &c.ensemble {
                    for (what, list) in [("scales", &m.scales), ("offsets", &m.offsets)] {
                        if !list.is_empty() && list.len() != targets.len() {
                            return Err(QsoError::invalid(what, "one per channel, or none"));
                        }
                        if list.iter().any(|x| !x.is_finite()) {
                            return Err(QsoError::invalid(what, "must be finite"));
                        }
                    }
                }
            }
            Drive::Field { min, max } => {
                if !(min.is_finite() && max.is_finite() && min < max && *min >= 0.0) {
                    return Err(QsoError::invalid("field", "0 ≤ min < max"));
                }
                if c.ensemble
                    .iter()
                    .any(|m| !m.scales.is_empty() || !m.offsets.is_empty())
                {
                    return Err(QsoError::invalid(
                        "ensemble",
                        "field profiles take members with weights only",
                    ));
                }
            }
        }
        if c.ensemble
            .iter()
            .any(|m| !(m.weight.is_finite() && m.weight >= 0.0))
            || (!c.ensemble.is_empty() && c.ensemble.iter().all(|m| m.weight == 0.0))
        {
            return Err(QsoError::invalid("ensemble", "weights ≥ 0, not all zero"));
        }
        Ok(())
    }

    /// The x, y and z operators of an rf channel.
    pub(super) fn channel(&self, target: Target) -> Result<[CMatrix; 3]> {
        let d = self.dimension();
        let mut out = [
            CMatrix::zeros(d, d),
            CMatrix::zeros(d, d),
            CMatrix::zeros(d, d),
        ];
        for i in self.targets(target)? {
            for (o, which) in out.iter_mut().zip([3, 4, 0]) {
                *o += self.embed(i, &self.local(i, which))?;
            }
        }
        Ok(out)
    }

    /// `(H(0), ∂H/∂u_j)` of a member: the free Hamiltonian with the
    /// member's offsets and the derivatives with respect to the controls
    /// of one step.
    fn control_hamiltonians(&self, c: &Control, m: &Member) -> Result<(CMatrix, Vec<CMatrix>)> {
        match &c.drive {
            Drive::Rf { targets, .. } => {
                let mut h0 = self.hamiltonian_at(self.field());
                let mut ops = Vec::with_capacity(2 * targets.len());
                for (k, &t) in targets.iter().enumerate() {
                    let [fx, fy, fz] = self.channel(t)?;
                    let scale = m.scales.get(k).copied().unwrap_or(1.0);
                    if let Some(&offset) = m.offsets.get(k) {
                        h0 += fz * C64::new(TWO_PI * offset, 0.0);
                    }
                    ops.push(fx * C64::new(TWO_PI * scale, 0.0));
                    ops.push(fy * C64::new(TWO_PI * scale, 0.0));
                }
                Ok((h0, ops))
            }
            Drive::Field { .. } => {
                // H(B) is affine in B.
                let h0 = self.hamiltonian_at(0.0);
                let d = self.dimension();
                let mut dh = CMatrix::zeros(d, d);
                for i in 0..self.spins().len() {
                    let w = self.zeeman(i, 1.0);
                    for s in 0..d {
                        dh[(s, s)].re += w * self.m(i, s);
                    }
                }
                Ok((h0, vec![dh]))
            }
        }
    }

    /// The whole generator (row-major Liouville space) of the master
    /// equation for `h` with the relaxation at field `b`: the drift of
    /// relaxation towards equilibrium enters through `Tr ρ`, as in
    /// [`SpinSystem::evolve`].
    fn full_generator(&self, h: &CMatrix, b: f64) -> Result<CMatrix> {
        let d = self.dimension();
        let n = d * d;
        let liouville = Liouville::new(&vec![vec![0]; d], None);
        let index: Vec<usize> = liouville.blocks[0]
            .1
            .iter()
            .map(|&(a, b)| a * d + b)
            .collect();
        let relaxation = self.relaxation(b)?;
        let mut out = CMatrix::zeros(n, n);
        self.generator_entries(&liouville, &Sparse::new(h), &relaxation, |_, r, c, v| {
            out[(index[r], index[c])] += v;
        });
        // dρ/dt = G ρ + Tr(ρ)(drift − G(1)/d).
        let mut r = vectorise(&relaxation.drift);
        for a in 0..d {
            let col = out.column(a * d + a).into_owned();
            r -= col * C64::new(1.0 / d as f64, 0.0);
        }
        for a in 0..d {
            let mut col = out.column_mut(a * d + a);
            col += &r;
        }
        Ok(out)
    }

    /// The transfer `Re Tr(C† ρ(T))` averaged over the ensemble, and its
    /// exact gradient with respect to the physical controls `u` (Hz or T;
    /// per step the x and y amplitude of each channel, or the field).
    pub fn control_transfer(&self, c: &Control, u: &[f64]) -> Result<(f64, Vec<f64>)> {
        let (value, grad, _) = self.control_members(c, u)?;
        Ok((value, grad))
    }

    /// Transfer, gradient and the transfer of each member.
    fn control_members(&self, c: &Control, u: &[f64]) -> Result<(f64, Vec<f64>, Vec<f64>)> {
        self.check_control(c)?;
        if u.len() != c.len() {
            return Err(QsoError::DimensionMismatch {
                expected: c.len(),
                actual: u.len(),
            });
        }
        if u.iter().any(|x| !x.is_finite()) {
            return Err(QsoError::invalid("controls", "must be finite"));
        }
        if let Drive::Field { min, max } = c.drive
            && u.iter().any(|&b| b < min || b > max)
        {
            return Err(QsoError::invalid("controls", "fields outside [min, max]"));
        }
        let members = c.members();
        let total: f64 = members.iter().map(|m| m.weight).sum();
        let results = crate::parallel::map_tasks(members.len(), |k| {
            if self.relaxes() {
                self.member_liouville(c, &members[k], u)
            } else {
                self.member_hilbert(c, &members[k], u)
            }
        });
        let mut value = 0.0;
        let mut grad = vec![0.0; u.len()];
        let mut each = Vec::with_capacity(members.len());
        for (m, r) in members.iter().zip(results) {
            let (v, g) = r?;
            value += m.weight * v / total;
            for (a, b) in grad.iter_mut().zip(g) {
                *a += m.weight * b / total;
            }
            each.push(v);
        }
        Ok((value, grad, each))
    }

    /// One member without relaxation.
    fn member_hilbert(&self, c: &Control, m: &Member, u: &[f64]) -> Result<(f64, Vec<f64>)> {
        let (h0, ops) = self.control_hamiltonians(c, m)?;
        let (n, dt) = (c.per_step(), c.step());
        let d = self.dimension();
        let work = 40.0 * (d * d * d) as f64 * c.steps as f64;
        if work > super::MAX_RELAXATION_WORK {
            return Err(QsoError::invalid(
                "control",
                format!(
                    "one evaluation needs about {work:.1e} operations (limit {:.1e})",
                    super::MAX_RELAXATION_WORK
                ),
            ));
        }
        let mut rhos = Vec::with_capacity(c.steps + 1);
        let mut eigen = Vec::with_capacity(c.steps);
        rhos.push(c.initial.clone());
        for k in 0..c.steps {
            let mut h = h0.clone();
            for (op, &x) in ops.iter().zip(&u[k * n..(k + 1) * n]) {
                if x != 0.0 {
                    h += op * C64::new(x, 0.0);
                }
            }
            let (e, v) = crate::linalg::eigh(&h);
            let mut vp = v.clone();
            for (j, &x) in e.iter().enumerate() {
                let phase = C64::new(0.0, -x * dt).exp();
                for r in 0..d {
                    vp[(r, j)] *= phase;
                }
            }
            let unitary = &vp * v.adjoint();
            let next = &unitary * &rhos[k] * unitary.adjoint();
            rhos.push(next);
            eigen.push((e, v, unitary));
        }
        let value = c
            .target
            .iter()
            .zip(rhos[c.steps].iter())
            .map(|(a, b)| a.conj() * b)
            .sum::<C64>()
            .re;
        let mut grad = vec![0.0; u.len()];
        let mut lambda = c.target.clone();
        for k in (0..c.steps).rev() {
            let (e, v, unitary) = &eigen[k];
            let rho = &rhos[k];
            let a = rho * unitary.adjoint() * lambda.adjoint()
                + rho.adjoint() * unitary.adjoint() * &lambda;
            let mut mm = v.adjoint() * a * v;
            for p in 0..d {
                for q in 0..d {
                    // (Gᵀ ∘ M)_pq = G_qp M_pq.
                    let x = (e[q] - e[p]) * dt / 2.0;
                    let sinc = if x.abs() < 1e-8 {
                        1.0 - x * x / 6.0
                    } else {
                        x.sin() / x
                    };
                    let g =
                        C64::new(0.0, -dt) * C64::new(0.0, -(e[p] + e[q]) * dt / 2.0).exp() * sinc;
                    mm[(p, q)] *= g;
                }
            }
            let x = v * mm * v.adjoint();
            for (j, op) in ops.iter().enumerate() {
                let mut s = 0.0;
                for r in 0..d {
                    for q in 0..d {
                        let o = op[(q, r)];
                        if o != ZERO {
                            s += (x[(r, q)] * o).re;
                        }
                    }
                }
                grad[k * n + j] = s;
            }
            lambda = unitary.adjoint() * &lambda * unitary;
        }
        Ok((value, grad))
    }

    /// One member with relaxation, in Liouville space.
    fn member_liouville(&self, c: &Control, m: &Member, u: &[f64]) -> Result<(f64, Vec<f64>)> {
        let (n, dt) = (c.per_step(), c.step());
        let d = self.dimension();
        let big = d * d;
        let (h0, ops) = self.control_hamiltonians(c, m)?;
        // Per step: the dense part of L and the directions ∂L/∂u.
        enum Parts {
            Rf { dense: CMatrix, ops: Vec<Entries> },
            Field { h0: CMatrix, dh: CMatrix },
        }
        let parts = match &c.drive {
            Drive::Rf { .. } => Parts::Rf {
                dense: self.full_generator(&h0, self.field())?,
                ops: ops.iter().map(commutator).collect(),
            },
            Drive::Field { .. } => Parts::Field {
                h0,
                dh: ops.into_iter().next().expect("one field derivative"),
            },
        };
        let field_step = |b: f64| -> Result<(CMatrix, Entries)> {
            let Parts::Field { h0, dh } = &parts else {
                unreachable!("field parts")
            };
            let h = h0 + dh * C64::new(b, 0.0);
            let dense = self.full_generator(&h, b)?;
            // ∂L/∂B: the Hamiltonian's part exactly, the relaxation's by
            // central differences.
            let span = match c.drive {
                Drive::Field { max, .. } => max,
                Drive::Rf { .. } => 1.0,
            };
            let step = 1e-5 * b.abs().max(span);
            let plus = self.full_generator(&h, b + step)?;
            let minus = self.full_generator(&h, b - step)?;
            let mut deriv = (plus - minus) * C64::new(0.5 / step, 0.0);
            for (r, col, v) in commutator(dh).0 {
                deriv[(r, col)] += v;
            }
            Ok((dense, Entries::from_dense(&deriv)))
        };
        let dense_norm = |m: &CMatrix| crate::linalg::dense::one_norm(m);
        // Cost: per step about substeps × terms × (2 + directions) products.
        let probe_norm = match &parts {
            Parts::Rf { dense, ops } => {
                let peak = u.iter().fold(0.0f64, |a, x| a.max(x.abs()));
                dense_norm(dense) + ops.iter().map(|o| o.one_norm(big)).sum::<f64>() * peak
            }
            Parts::Field { .. } => {
                let (dense, _) = field_step(u.iter().fold(0.0f64, |a, &x| a.max(x)))?;
                dense_norm(&dense)
            }
        };
        let products = (substeps(probe_norm * dt) * 20 * (4 + n)) as f64 * c.steps as f64;
        let work = products * (big * big) as f64 * 8.0;
        if work > super::MAX_RELAXATION_WORK {
            return Err(QsoError::invalid(
                "control",
                format!(
                    "one evaluation with relaxation needs about {work:.1e} operations (limit {:.1e}); \
                     use fewer spins or steps",
                    super::MAX_RELAXATION_WORK
                ),
            ));
        }
        let zero_u = vec![0.0; 0];
        let mut states = Vec::with_capacity(c.steps + 1);
        states.push(vectorise(&c.initial));
        let mut field_cache: Vec<Option<(CMatrix, Entries)>> = Vec::new();
        for k in 0..c.steps {
            let step_u = &u[k * n..(k + 1) * n];
            let next = match &parts {
                Parts::Rf { dense, ops } => {
                    let g = Generator {
                        dense,
                        ops,
                        u: step_u,
                        adjoint: false,
                    };
                    let norms: Vec<f64> = ops.iter().map(|o| o.one_norm(big)).collect();
                    expmv(&g, &states[k], dt, g.one_norm(&norms, dense_norm(dense)))
                }
                Parts::Field { .. } => {
                    let (dense, deriv) = field_step(step_u[0])?;
                    let g = Generator {
                        dense: &dense,
                        ops: &[],
                        u: &zero_u,
                        adjoint: false,
                    };
                    let v = expmv(&g, &states[k], dt, dense_norm(&dense));
                    field_cache.push(Some((dense, deriv)));
                    v
                }
            };
            states.push(next);
        }
        let target = vectorise(&c.target);
        let value = target.dotc(&states[c.steps]).re;
        let mut grad = vec![0.0; u.len()];
        let mut lambda = target;
        for k in (0..c.steps).rev() {
            let step_u = &u[k * n..(k + 1) * n];
            match &parts {
                Parts::Rf { dense, ops } => {
                    let norms: Vec<f64> = ops.iter().map(|o| o.one_norm(big)).collect();
                    let g = Generator {
                        dense,
                        ops,
                        u: step_u,
                        adjoint: false,
                    };
                    let norm = g.one_norm(&norms, dense_norm(dense));
                    let directions: Vec<&Entries> = ops.iter().collect();
                    let g_k = frechet(&g, &directions, &states[k], &lambda, dt, norm);
                    grad[k * n..(k + 1) * n].copy_from_slice(&g_k);
                    let back = Generator {
                        dense,
                        ops,
                        u: step_u,
                        adjoint: true,
                    };
                    lambda = expmv(&back, &lambda, dt, norm);
                }
                Parts::Field { .. } => {
                    let (dense, deriv) = field_cache[k].take().expect("cached step");
                    let g = Generator {
                        dense: &dense,
                        ops: &[],
                        u: &zero_u,
                        adjoint: false,
                    };
                    let norm = dense_norm(&dense);
                    let g_k = frechet(&g, &[&deriv], &states[k], &lambda, dt, norm);
                    grad[k] = g_k[0];
                    let back = Generator {
                        dense: &dense,
                        ops: &[],
                        u: &zero_u,
                        adjoint: true,
                    };
                    lambda = expmv(&back, &lambda, dt, norm);
                }
            }
        }
        Ok((value, grad))
    }

    /// Optimise the controls from `start` (physical, inside the limits)
    /// with L-BFGS, maximising the transfer less the penalties.
    pub fn optimize_control(
        &self,
        c: &Control,
        start: &[f64],
        options: &Options,
    ) -> Result<Optimized> {
        self.check_control(c)?;
        if start.len() != c.len() {
            return Err(QsoError::DimensionMismatch {
                expected: c.len(),
                actual: start.len(),
            });
        }
        // Check once, so that the objective's evaluations cannot fail.
        self.control_transfer(c, &c.physical(&c.free(start)).0)?;
        // The optimiser sees the objective in units of its natural size,
        // `‖C‖‖ρ(0) − Tr ρ(0)/d‖` (thermal transfers are of order 10⁻⁵).
        let frobenius = |m: &CMatrix| m.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        let d = self.dimension();
        let mut deviation = c.initial.clone();
        let trace = c.initial.trace() / d as f64;
        for i in 0..d {
            deviation[(i, i)] -= trace;
        }
        let size = frobenius(&c.target)
            * [frobenius(&deviation), frobenius(&c.initial)]
                .into_iter()
                .find(|&x| x > 0.0)
                .unwrap_or(1.0);
        let size = if size > 0.0 { size } else { 1.0 };
        let failure = std::sync::Mutex::new(None);
        let objective = GradObjective::new(c.len(), |v: &[f64]| {
            let (u, jac) = c.physical(v);
            match self.control_transfer(c, &u) {
                Ok((value, grad)) => {
                    let (p, pg) = c.penalty(&u);
                    let g: Vec<f64> = grad.iter().zip(&pg).map(|(a, b)| (b - a) / size).collect();
                    ((p - value) / size, c.chain(&jac, &g))
                }
                Err(e) => {
                    *failure.lock().expect("unpoisoned") = Some(e);
                    (f64::INFINITY, vec![0.0; v.len()])
                }
            }
        });
        let result = minimize(&objective, &c.free(start), &Method::lbfgs(), options)?;
        if let Some(e) = failure.into_inner().expect("unpoisoned") {
            return Err(e);
        }
        let (controls, _) = c.physical(&result.x);
        let (transfer, _, members) = self.control_members(c, &controls)?;
        let objective = transfer - c.penalty(&controls).0;
        Ok(Optimized {
            controls,
            transfer,
            members,
            objective,
            iterations: result.iterations,
            evaluations: result.evaluations,
            converged: result.converged,
            message: result.message,
        })
    }

    /// The events that apply controls `u` in [`SpinSystem::evolve`]: one
    /// [`Event::Control`] (rf), or a field and a delay per step.
    pub fn control_events(&self, c: &Control, u: &[f64]) -> Result<Vec<Event>> {
        self.check_control(c)?;
        if u.len() != c.len() {
            return Err(QsoError::DimensionMismatch {
                expected: c.len(),
                actual: u.len(),
            });
        }
        let dt = c.step();
        Ok(match &c.drive {
            Drive::Rf { targets, .. } => vec![Event::Control {
                targets: targets.clone(),
                step: dt,
                amplitudes: u
                    .chunks(2 * targets.len())
                    .map(|step| step.chunks(2).map(|p| [p[0], p[1]]).collect())
                    .collect(),
            }],
            Drive::Field { .. } => u
                .iter()
                .flat_map(|&b| [Event::Field(b), Event::Delay(dt)])
                .collect(),
        })
    }
}

/// The largest `Re Tr(C ρ')` over all `ρ' = UρU†` for Hermitian `ρ` and `C`:
/// the sum of the products of their eigenvalues, both in descending order
/// (the unitary bound of Glaser et al., Science 280, 421 (1998)).
pub fn unitary_bound(rho: &CMatrix, target: &CMatrix) -> Result<f64> {
    if rho.shape() != target.shape() || rho.nrows() != rho.ncols() {
        return Err(QsoError::DimensionMismatch {
            expected: rho.nrows(),
            actual: target.nrows(),
        });
    }
    let hermitian = |m: &CMatrix| {
        (m - m.adjoint())
            .iter()
            .all(|z| z.norm() <= 1e-12 * (1.0 + m.iter().fold(0.0f64, |a, z| a.max(z.norm()))))
    };
    if !hermitian(rho) || !hermitian(target) {
        return Err(QsoError::invalid("bound", "needs Hermitian operators"));
    }
    let (mut a, _) = crate::linalg::eigh(rho);
    let (mut b, _) = crate::linalg::eigh(target);
    a.sort_by(|x, y| y.total_cmp(x));
    b.sort_by(|x, y| y.total_cmp(x));
    Ok(a.iter().zip(&b).map(|(x, y)| x * y).sum())
}

#[cfg(test)]
mod tests;
