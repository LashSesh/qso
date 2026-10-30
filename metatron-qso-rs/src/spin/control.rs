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
//! The state is propagated, as in [`SpinSystem::evolve`], as its
//! deviation from the identity with the drift towards equilibrium in an
//! augmented column, so thermal polarisations keep their digits. Field
//! profiles with relaxation take the derivative of the relaxation with
//! respect to the field (thermal populations, spectral densities) by
//! Richardson-extrapolated central differences; the rest is exact.

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
    /// amplitude (Hz) per channel and step. [`SpinSystem::optimize_control`]
    /// keeps each pair within `limit` Hz (`√(x² + y²)`) if given;
    /// [`SpinSystem::control_transfer`] evaluates any amplitudes.
    Rf {
        targets: Vec<Target>,
        limit: Option<f64>,
    },
    /// The field (T) of every step, between `min` and `max`.
    Field { min: f64, max: f64 },
}

/// A member of the ensemble the controls must work for: rf amplitudes
/// scaled per channel (empty: all 1) and the spins of each channel shifted
/// by an offset (Hz, `H += 2πΔF_z`: a line at `+Δ` Hz in
/// [`SpinSystem::lines`], `−Δ/ν₀` ppm; empty: none), with a weight.
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
    /// `scale` Hz (at most 0.9 of the limit), or fields uniform in the
    /// inner 96 % of `[min, max]` (`scale` unused).
    pub fn random_start(&self, seed: u64, scale: f64) -> Vec<f64> {
        let mut rng = crate::rng::Rng::seed_from_u64(seed);
        match &self.drive {
            Drive::Rf { limit, .. } => {
                let scale = if scale.is_finite() { scale.abs() } else { 0.0 };
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
                .map(|_| min + (max - min) * (0.02 + 0.96 * rng.uniform()))
                .collect(),
        }
    }

    /// `start` checked against the limits, with values in the outer 2 %
    /// moved inside (the bounded map cannot leave its edge).
    pub(super) fn inside(&self, start: &[f64]) -> Result<Vec<f64>> {
        if start.iter().any(|x| !x.is_finite()) {
            return Err(QsoError::invalid("start", "must be finite"));
        }
        match &self.drive {
            Drive::Rf { limit: None, .. } => Ok(start.to_vec()),
            Drive::Rf { limit: Some(l), .. } => {
                let mut out = start.to_vec();
                for pair in out.chunks_mut(2) {
                    let r = pair[0].hypot(pair[1]);
                    if r > l * (1.0 + 1e-12) {
                        return Err(QsoError::invalid("start", "rf amplitudes beyond the limit"));
                    }
                    if r > 0.98 * l {
                        pair[0] *= 0.98 * l / r;
                        pair[1] *= 0.98 * l / r;
                    }
                }
                Ok(out)
            }
            Drive::Field { min, max } => {
                let margin = 0.02 * (max - min);
                start
                    .iter()
                    .map(|&b| {
                        if b < *min || b > *max {
                            Err(QsoError::invalid("start", "fields outside [min, max]"))
                        } else {
                            Ok(b.clamp(min + margin, max - margin))
                        }
                    })
                    .collect()
            }
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
    /// ∂y/∂v₂]]`, flattened; per field: one). Bounds are kept by a sine:
    /// an amplitude `|u| = l·sin(|v|/l)` (the identity in Hz near zero),
    /// a field `B = mid + half·sin(v/10)` (one unit of `v` a twentieth of
    /// the range). A bound is reached at a finite `v`, with zero slope but
    /// no plateau beyond it, so optima on a bound stay reachable.
    pub(super) fn physical(&self, v: &[f64]) -> (Vec<f64>, Vec<f64>) {
        match &self.drive {
            Drive::Rf { limit: None, .. } => (v.to_vec(), Vec::new()),
            Drive::Rf { limit: Some(l), .. } => {
                let mut u = vec![0.0; v.len()];
                let mut jac = vec![0.0; 2 * v.len()];
                for (p, pair) in v.chunks(2).enumerate() {
                    let (a, b) = (pair[0], pair[1]);
                    // u = v·F(r), F(r) = l·sin(r/l)/r; G = F'(r)/r.
                    let r = a.hypot(b);
                    let z = r / l;
                    let (f, g) = if z < 1e-4 {
                        (1.0 - z * z / 6.0, (-1.0 / 3.0 + z * z / 30.0) / (l * l))
                    } else {
                        (z.sin() / z, (z.cos() * z - z.sin()) / (z * z * z) / (l * l))
                    };
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
                    .map(|&x| (mid + half * (x / 10.0).sin()).clamp(*min, *max))
                    .collect();
                let jac = v.iter().map(|&x| half / 10.0 * (x / 10.0).cos()).collect();
                (u, jac)
            }
        }
    }

    /// Free variables of physical controls (inside the limits).
    pub(super) fn free(&self, u: &[f64]) -> Vec<f64> {
        match &self.drive {
            Drive::Rf { limit: None, .. } => u.to_vec(),
            Drive::Rf { limit: Some(l), .. } => {
                let mut v = vec![0.0; u.len()];
                for (p, pair) in u.chunks(2).enumerate() {
                    // |u| = l·sin(r/l), v = u·r/|u|.
                    let t = (pair[0].hypot(pair[1]) / l).min(1.0);
                    let k = if t < 1e-12 { 1.0 } else { t.asin() / t };
                    v[2 * p] = k * pair[0];
                    v[2 * p + 1] = k * pair[1];
                }
                v
            }
            Drive::Field { min, max } => {
                let (mid, half) = ((max + min) / 2.0, (max - min) / 2.0);
                u.iter()
                    .map(|&b| 10.0 * ((b - mid) / half).clamp(-1.0, 1.0).asin())
                    .collect()
            }
        }
    }

    /// Gradient with respect to the free variables from that with respect
    /// to the physical controls.
    pub(super) fn chain(&self, jac: &[f64], g: &[f64]) -> Vec<f64> {
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
pub(super) struct Entries(Vec<(usize, usize, C64)>);

impl Entries {
    pub(super) fn from_dense(m: &CMatrix) -> Self {
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

    pub(super) fn one_norm(&self, n: usize) -> f64 {
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
pub(super) struct Generator<'a> {
    pub(super) dense: &'a CMatrix,
    pub(super) ops: &'a [Entries],
    pub(super) u: &'a [f64],
    /// `L†` instead of `L`.
    pub(super) adjoint: bool,
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
pub(super) fn substeps(norm: f64) -> usize {
    (norm / 0.5).ceil().max(1.0) as usize
}

/// `e^{tL} x` by Taylor series in `substeps(t‖L‖)` pieces, each summed to
/// rounding.
pub(super) fn expmv(g: &Generator<'_>, x: &DVector<C64>, t: f64, norm: f64) -> DVector<C64> {
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
pub(super) fn frechet(
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

/// `[vec(m − Tr m/d); Tr m]`: the deviation from the identity and the
/// trace, as [`SpinSystem::evolve`] propagates them with relaxation.
fn augment(m: &CMatrix) -> DVector<C64> {
    let d = m.nrows();
    let trace = m.trace();
    let mut v = DVector::zeros(d * d + 1);
    for a in 0..d {
        for b in 0..d {
            v[a * d + b] = m[(a, b)];
        }
        v[a * d + a] -= trace / d as f64;
    }
    v[d * d] = trace;
    v
}

/// Largest Liouville space (`d²` elements) for control with relaxation:
/// six spins ½. Its dense generator takes 270 MB.
pub const MAX_CONTROL_LIOUVILLE: usize = 4096;

/// Most memory (bytes) one evaluation of a control problem may keep:
/// states and eigenvectors of every step, of every member at once.
pub const MAX_CONTROL_MEMORY: f64 = 2.0 * (1u64 << 30) as f64;

/// Refuse `what` above the work or memory limit.
fn check_cost(what: &str, work: f64, memory: f64) -> Result<()> {
    if work > super::MAX_RELAXATION_WORK {
        return Err(QsoError::invalid(
            "control",
            format!(
                "{what} needs about {work:.1e} operations per evaluation (limit {:.1e}); \
                 use fewer spins, steps or members, or weaker rf",
                super::MAX_RELAXATION_WORK
            ),
        ));
    }
    if memory > MAX_CONTROL_MEMORY {
        return Err(QsoError::invalid(
            "control",
            format!(
                "{what} needs about {:.1} GB per evaluation (limit {:.0} GB); use fewer \
                 spins, steps or members",
                memory / 1e9,
                MAX_CONTROL_MEMORY / 1e9
            ),
        ));
    }
    Ok(())
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
            if m.iter().any(|z| !(z.re.is_finite() && z.im.is_finite())) {
                return Err(QsoError::invalid(
                    "control",
                    "initial and target operators must be finite",
                ));
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
        let total: f64 = c.ensemble.iter().map(|m| m.weight).sum();
        if c.ensemble
            .iter()
            .any(|m| !(m.weight.is_finite() && m.weight >= 0.0))
            || (!c.ensemble.is_empty() && !(total > 0.0 && total.is_finite()))
        {
            return Err(QsoError::invalid(
                "ensemble",
                "weights ≥ 0 with a finite, positive sum",
            ));
        }
        if self.relaxes() && d * d > MAX_CONTROL_LIOUVILLE {
            return Err(QsoError::invalid(
                "control",
                format!(
                    "control with relaxation works in Liouville space of at most {MAX_CONTROL_LIOUVILLE} \
                     elements (six spins ½); this system has {}",
                    d * d
                ),
            ));
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

    /// The generator of `[vec Δ; Tr ρ]` (row-major Liouville space, `Δ =
    /// ρ − Tr ρ/d`) for `h` with the relaxation at field `b`, as
    /// [`SpinSystem::evolve`] propagates it: the master equation on `Δ`,
    /// and the drift towards equilibrium times the trace in the last
    /// column.
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
        let mut out = CMatrix::zeros(n + 1, n + 1);
        self.generator_entries(&liouville, &Sparse::new(h), &relaxation, |_, r, c, v| {
            out[(index[r], index[c])] += v;
        });
        for (i, v) in vectorise(&relaxation.drift).iter().enumerate() {
            out[(i, n)] = *v;
        }
        Ok(out)
    }

    /// `∂/∂B` of the relaxation part of [`SpinSystem::full_generator`]
    /// (the Hamiltonian `h` held fixed): central differences of steps
    /// `δ` and `δ/2`, `δ = 10⁻²·max(|b|, span)`, Richardson-extrapolated,
    /// so that both rounding and truncation stay near 10⁻⁸ of it.
    fn relaxation_derivative(&self, h: &CMatrix, b: f64, span: f64) -> Result<CMatrix> {
        let step = 1e-2 * b.abs().max(span);
        let central = |s: f64| -> Result<CMatrix> {
            Ok(
                (self.full_generator(h, b + s)? - self.full_generator(h, b - s)?)
                    * C64::new(0.5 / s, 0.0),
            )
        };
        let (wide, narrow) = (central(step)?, central(step / 2.0)?);
        Ok(narrow * C64::new(4.0 / 3.0, 0.0) - wide * C64::new(1.0 / 3.0, 0.0))
    }

    /// Estimated operations and memory of one evaluation, with rf of at
    /// most `peak` Hz or fields of at most `peak` T.
    fn control_cost(&self, c: &Control, peak: f64) -> Result<(f64, f64)> {
        let d = self.dimension() as f64;
        let members = match c.drive {
            Drive::Rf { .. } => c.members().len() as f64,
            Drive::Field { .. } => 1.0,
        };
        let steps = c.steps as f64;
        if !self.relaxes() {
            // Measured: about 600 d³ operations per step (eigenvectors and
            // products); ρ and the eigenvectors of every step are kept.
            let work = 600.0 * d * d * d * steps * members;
            let memory = (32.0 * d * d + 8.0 * d) * (steps + 1.0) * members;
            return Ok((work, memory));
        }
        let n = d * d + 1.0;
        let per_step = c.per_step() as f64;
        // ‖L‖₁ ≤ 2‖H‖₁ + 2·(the relaxation's rate bound), without building
        // L (up to 270 MB).
        let (h, b) = match &c.drive {
            Drive::Rf { targets, .. } => {
                let (mut h, _) = self.control_hamiltonians(c, &Member::ideal())?;
                let scale = c
                    .members()
                    .iter()
                    .flat_map(|m| m.scales.iter().map(|s| s.abs()))
                    .fold(1.0f64, f64::max);
                for &t in targets {
                    let [fx, fy, _] = self.channel(t)?;
                    h += (fx + fy) * C64::new(TWO_PI * peak * scale, 0.0);
                }
                (h, self.field())
            }
            Drive::Field { .. } => {
                let (h0, dh) = self.control_hamiltonians(c, &Member::ideal())?;
                (h0 + &dh[0] * C64::new(peak, 0.0), peak)
            }
        };
        let norm = 2.0 * crate::linalg::dense::one_norm(&h) + 2.0 * self.relaxation(b)?.rate;
        // Measured: about 5 × (substeps × 20 terms × (4 + directions))
        // products of n² per step.
        let products = (substeps(norm * c.step()) * 20) as f64 * (4.0 + per_step);
        let mut work = 5.0 * products * n * n * 8.0 * steps * members;
        if let Drive::Field { .. } = c.drive {
            // Five generators per step (one forward, four for the
            // derivative), built entry by entry.
            work += 5.0 * 2.0 * steps * n * n * 40.0;
        }
        let memory = (16.0 * n * n * 6.0 + 16.0 * n * (steps + 1.0)) * members;
        Ok((work, memory))
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
        let peak = u.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let (work, memory) = self.control_cost(c, peak)?;
        check_cost("this control problem", work, memory)?;
        let members = c.members();
        let total: f64 = members.iter().map(|m| m.weight).sum();
        // Field profiles of one system: the members differ by weight only.
        let distinct = match c.drive {
            Drive::Rf { .. } => members.len(),
            Drive::Field { .. } => 1,
        };
        let results = crate::parallel::map_tasks(distinct, |k| {
            if self.relaxes() {
                self.member_liouville(c, &members[k], u)
            } else {
                self.member_hilbert(c, &members[k], u)
            }
        });
        let mut results = results.into_iter().collect::<Result<Vec<_>>>()?;
        if distinct == 1 && members.len() > 1 {
            let only = results[0].clone();
            results = vec![only; members.len()];
        }
        let mut value = 0.0;
        let mut grad = vec![0.0; u.len()];
        let mut each = Vec::with_capacity(members.len());
        for (m, (v, g)) in members.iter().zip(results) {
            let w = m.weight / total;
            value += w * v;
            for (a, b) in grad.iter_mut().zip(g) {
                *a += w * b;
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
        let unitary = |e: &[f64], v: &CMatrix| {
            let mut vp = v.clone();
            for (j, &x) in e.iter().enumerate() {
                let phase = C64::new(0.0, -x * dt).exp();
                for r in 0..d {
                    vp[(r, j)] *= phase;
                }
            }
            &vp * v.adjoint()
        };
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
            let w = unitary(&e, &v);
            let next = &w * &rhos[k] * w.adjoint();
            rhos.push(next);
            eigen.push((e, v));
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
            let (e, v) = &eigen[k];
            let w = unitary(e, v);
            let rho = &rhos[k];
            let a = rho * w.adjoint() * lambda.adjoint() + rho.adjoint() * w.adjoint() * &lambda;
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
            lambda = w.adjoint() * &lambda * w;
        }
        Ok((value, grad))
    }

    /// One member with relaxation, in Liouville space: `[vec Δ; Tr ρ]`
    /// under the generator of [`SpinSystem::full_generator`].
    fn member_liouville(&self, c: &Control, m: &Member, u: &[f64]) -> Result<(f64, Vec<f64>)> {
        let (n, dt) = (c.per_step(), c.step());
        let big = self.dimension() * self.dimension() + 1;
        let (h0, ops) = self.control_hamiltonians(c, m)?;
        let dense_norm = |m: &CMatrix| crate::linalg::dense::one_norm(m);
        let no_ops: [Entries; 0] = [];
        let mut states = Vec::with_capacity(c.steps + 1);
        states.push(augment(&c.initial));
        // λ pairs with [vec Δ; Tr ρ]: Tr(C†ρ) = ⟨vec C, vec Δ⟩ + conj(Tr C) Tr ρ/d.
        let d = self.dimension();
        let target = DVector::from_iterator(
            big,
            vectorise(&c.target)
                .iter()
                .copied()
                .chain(std::iter::once(c.target.trace() / d as f64)),
        );
        let mut grad = vec![0.0; u.len()];
        match &c.drive {
            Drive::Rf { .. } => {
                let dense = self.full_generator(&h0, self.field())?;
                let ops: Vec<Entries> = ops.iter().map(commutator).collect();
                let norms: Vec<f64> = ops.iter().map(|o| o.one_norm(big)).collect();
                let base = dense_norm(&dense);
                for k in 0..c.steps {
                    let g = Generator {
                        dense: &dense,
                        ops: &ops,
                        u: &u[k * n..(k + 1) * n],
                        adjoint: false,
                    };
                    let next = expmv(&g, &states[k], dt, g.one_norm(&norms, base));
                    states.push(next);
                }
                let value = target.dotc(&states[c.steps]).re;
                let mut lambda = target;
                let directions: Vec<&Entries> = ops.iter().collect();
                for k in (0..c.steps).rev() {
                    let step_u = &u[k * n..(k + 1) * n];
                    let g = Generator {
                        dense: &dense,
                        ops: &ops,
                        u: step_u,
                        adjoint: false,
                    };
                    let norm = g.one_norm(&norms, base);
                    let g_k = frechet(&g, &directions, &states[k], &lambda, dt, norm);
                    grad[k * n..(k + 1) * n].copy_from_slice(&g_k);
                    let back = Generator { adjoint: true, ..g };
                    lambda = expmv(&back, &lambda, dt, norm);
                }
                Ok((value, grad))
            }
            Drive::Field { max, .. } => {
                let dh = &ops[0];
                let hamiltonian = |b: f64| &h0 + dh * C64::new(b, 0.0);
                for (k, &b) in u.iter().enumerate() {
                    let dense = self.full_generator(&hamiltonian(b), b)?;
                    let g = Generator {
                        dense: &dense,
                        ops: &no_ops,
                        u: &[],
                        adjoint: false,
                    };
                    let next = expmv(&g, &states[k], dt, dense_norm(&dense));
                    states.push(next);
                }
                let value = target.dotc(&states[c.steps]).re;
                let mut lambda = target;
                let exact = commutator(dh);
                for k in (0..c.steps).rev() {
                    let b = u[k];
                    let h = hamiltonian(b);
                    let dense = self.full_generator(&h, b)?;
                    // ∂L/∂B: the Hamiltonian's part exactly, the
                    // relaxation's by extrapolated differences.
                    let mut deriv = self.relaxation_derivative(&h, b, *max)?;
                    for &(r, col, v) in &exact.0 {
                        deriv[(r, col)] += v;
                    }
                    let deriv = Entries::from_dense(&deriv);
                    let g = Generator {
                        dense: &dense,
                        ops: &no_ops,
                        u: &[],
                        adjoint: false,
                    };
                    let norm = dense_norm(&dense);
                    grad[k] = frechet(&g, &[&deriv], &states[k], &lambda, dt, norm)[0];
                    let back = Generator { adjoint: true, ..g };
                    lambda = expmv(&back, &lambda, dt, norm);
                }
                Ok((value, grad))
            }
        }
    }

    /// Optimise the controls with L-BFGS from `start` (physical, within
    /// the limits; values in the outer 2 % of a range, which the bounded
    /// map cannot leave, are moved inside), maximising the transfer less
    /// the penalties. With relaxation and rf, the work of an evaluation
    /// grows with the amplitude: give a `limit`, or an evaluation the
    /// optimiser tries may exceed the work limit and end the run with an
    /// error.
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
        let start = c.inside(start)?;
        // The most expensive evaluation the optimiser can try.
        let worst = match &c.drive {
            Drive::Rf { limit: Some(l), .. } => *l,
            Drive::Field { max, .. } => *max,
            Drive::Rf { limit: None, .. } => start.iter().fold(0.0f64, |m, x| m.max(x.abs())),
        };
        let (work, memory) = self.control_cost(c, worst)?;
        check_cost("this control problem", work, memory)?;
        let (first, _) = self.control_transfer(c, &start)?;
        // The optimiser sees the objective in units of its natural size:
        // the transfer at the start, or `‖C‖‖ρ(0) − Tr ρ(0)/d‖` if larger.
        let frobenius = |m: &CMatrix| m.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        let d = self.dimension();
        let mut deviation = c.initial.clone();
        let trace = c.initial.trace() / d as f64;
        for i in 0..d {
            deviation[(i, i)] -= trace;
        }
        let size = [
            first
                .abs()
                .max(frobenius(&c.target) * frobenius(&deviation)),
            frobenius(&c.target) * frobenius(&c.initial),
            1.0,
        ]
        .into_iter()
        .find(|&x| x > 0.0 && x.is_finite())
        .unwrap_or(1.0);
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
        let result = minimize(&objective, &c.free(&start), &Method::lbfgs(), options)?;
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
