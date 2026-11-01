//! Motion beyond isotropic tumbling: rotational diffusion with any
//! diffusion tensor (Woessner, J. Chem. Phys. 37, 647 (1962)) and internal
//! motion in the model-free form of Lipari and Szabo (J. Am. Chem. Soc.
//! 104, 4546 (1982)) and Clore et al. (J. Am. Chem. Soc. 112, 4989
//! (1990)).
//!
//! The spatial tensor `A^k` of an interaction is fixed in the molecular
//! frame. Rotational diffusion with tensor `D` (molecular frame) acts on
//! the traceless symmetric tensors as the operator `M = −Σ_ab D_ab G_a G_b`,
//! `G_a(T) = [Ω_a, T]` the rotation generators: in the orthonormal basis
//! `E_μ` a symmetric 5 × 5 matrix with eigenvalues `λ_j` and eigentensors
//! `V_j`. Averaged over the initial orientation, the laboratory-frame
//! correlation of two interactions is that of isotropic tumbling with
//! `(A^k : A^l) e^{−τ/τ_c}` replaced by
//! `Σ_j (A^k : V_j)(A^l : V_j) e^{−λ_j τ}`; for `D = d·1`, `M = 6d` and
//! `τ_c = 1/(6d)`. Internal motion that is independent of the overall
//! motion multiplies it by
//! `C_I(τ) = S² + (1 − S_f²) e^{−τ/τ_f} + (S_f² − S²) e^{−τ/τ_s}`,
//! `S² = S_f² S_s²` (Lipari–Szabo: `S_s² = 1`). Interactions of one group
//! share their internal motion (as the dipolar coupling and shielding of
//! a rigid peptide plane do); between groups, and with interactions that
//! move only with the molecule, only the order parameters correlate:
//! `S_k S_l`. Each exponential `e^{−(λ_j + 1/τ)τ}` is a *mode* with a
//! weight per interaction, and the Redfield sum runs over the modes, each
//! with its own spectral density `λ/(λ² + ω²)`.

use serde::{Deserialize, Serialize};

use super::{SpinSystem, basis};
use crate::error::{QsoError, Result};
use crate::linalg::RMatrix;

/// An interaction that relaxes: the dipolar coupling of two spins (the
/// lower index first) or the shielding anisotropy of one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Interaction {
    Dipolar(usize, usize),
    Shielding(usize),
}

impl Interaction {
    /// The same interaction with a dipolar pair in ascending order.
    pub fn normalized(self) -> Self {
        match self {
            Interaction::Dipolar(i, j) => Interaction::Dipolar(i.min(j), i.max(j)),
            s => s,
        }
    }
}

/// Model-free internal motion: the correlation function
/// `C_I(τ) = S² + (1 − S_f²) e^{−τ/τ_f} + (S_f² − S²) e^{−τ/τ_s}` with
/// `S² = S_f² S_s²` (Clore et al.); Lipari and Szabo's form has `S_s² = 1`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelFree {
    s2_fast: f64,
    tau_fast: f64,
    s2_slow: f64,
    tau_slow: f64,
}

impl ModelFree {
    /// Lipari and Szabo: order parameter `S²` and internal correlation time
    /// `τ_e` (s).
    pub fn lipari_szabo(s2: f64, tau_e: f64) -> Result<Self> {
        Self::extended(s2, tau_e, 1.0, tau_e)
    }

    /// Clore et al.: a fast motion (`S_f²`, `τ_f`) and a slow one (`S_s²`,
    /// `τ_s`), times in s.
    pub fn extended(s2_fast: f64, tau_fast: f64, s2_slow: f64, tau_slow: f64) -> Result<Self> {
        let m = ModelFree {
            s2_fast,
            tau_fast,
            s2_slow,
            tau_slow,
        };
        m.validate()?;
        Ok(m)
    }

    pub(super) fn validate(&self) -> Result<()> {
        let unit = |s: f64| (0.0..=1.0).contains(&s);
        // Positive, finite, and with a finite rate 1/t (no subnormals).
        let time = |t: f64| t.is_finite() && t > 0.0 && (1.0 / t).is_finite();
        if !(unit(self.s2_fast) && unit(self.s2_slow) && time(self.tau_fast) && time(self.tau_slow))
        {
            return Err(QsoError::invalid(
                "internal_motion",
                "order parameters in [0, 1] and finite positive correlation times",
            ));
        }
        Ok(())
    }

    /// The order parameter `S² = S_f² S_s²`.
    pub fn order_parameter(&self) -> f64 {
        self.s2_fast * self.s2_slow
    }

    /// `(S_f², τ_f, S_s², τ_s)`.
    pub fn parameters(&self) -> (f64, f64, f64, f64) {
        (self.s2_fast, self.tau_fast, self.s2_slow, self.tau_slow)
    }

    /// The internal correlation function `C_I(τ)`.
    pub fn correlation(&self, tau: f64) -> f64 {
        self.terms()
            .iter()
            .map(|&(w, rate)| w * (-rate * tau).exp())
            .sum()
    }

    /// `(weight, rate)` of the exponentials of `C_I` (rates in s⁻¹; the
    /// constant `S²` has rate 0).
    fn terms(&self) -> [(f64, f64); 3] {
        let s2 = self.order_parameter();
        [
            (s2, 0.0),
            (1.0 - self.s2_fast, 1.0 / self.tau_fast),
            (self.s2_fast - s2, 1.0 / self.tau_slow),
        ]
    }
}

/// Interactions that share one internal motion.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::spin) struct InternalMotion {
    interactions: Vec<Interaction>,
    motion: ModelFree,
}

/// The diffusion tensor (s⁻¹) of axially symmetric rotational diffusion
/// with `D_∥` about `axis` and `D_⊥` perpendicular to it.
pub fn axial_diffusion(parallel: f64, perpendicular: f64, axis: [f64; 3]) -> Result<[[f64; 3]; 3]> {
    let norm = axis.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !(parallel.is_finite()
        && perpendicular.is_finite()
        && parallel > 0.0
        && perpendicular > 0.0
        && norm.is_finite()
        && norm > 0.0)
    {
        return Err(QsoError::invalid(
            "diffusion",
            "positive finite diffusion constants about a finite, non-zero axis",
        ));
    }
    let n = axis.map(|x| x / norm);
    let mut d = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let delta = if a == b { 1.0 } else { 0.0 };
            d[a][b] = perpendicular * delta + (parallel - perpendicular) * n[a] * n[b];
        }
    }
    Ok(d)
}

/// Checks a diffusion tensor: finite, symmetric (to 10⁻¹² of its largest
/// entry) and positive definite.
pub(in crate::spin) fn check_diffusion(d: &[[f64; 3]; 3]) -> Result<()> {
    let bad = || {
        QsoError::invalid(
            "diffusion",
            "the diffusion tensor must be finite, symmetric and positive definite",
        )
    };
    if d.iter().flatten().any(|x| !x.is_finite()) {
        return Err(bad());
    }
    let largest = d.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
    let asymmetric = (0..3).any(|a| (0..3).any(|b| (d[a][b] - d[b][a]).abs() > 1e-12 * largest));
    if asymmetric {
        return Err(bad());
    }
    let m = RMatrix::from_fn(3, 3, |a, b| (d[a][b] + d[b][a]) / 2.0);
    let eigen = m.symmetric_eigen();
    if eigen.eigenvalues.iter().any(|&x| x.is_nan() || x <= 0.0) {
        return Err(bad());
    }
    Ok(())
}

/// The rotational diffusion operator on the traceless symmetric tensors,
/// in the basis `E_μ`: `M_μν = −Σ_ab D_ab ⟨E_μ, G_a G_b E_ν⟩`.
pub(in crate::spin) fn diffusion_operator(d: &[[f64; 3]; 3]) -> RMatrix {
    let e = basis();
    // Ω_a: (Ω_a)_bc = −ε_abc.
    let omega = |a: usize| {
        let mut w = [[0.0; 3]; 3];
        let (b, c) = ((a + 1) % 3, (a + 2) % 3);
        w[b][c] = -1.0;
        w[c][b] = 1.0;
        w
    };
    let mul = |x: &[[f64; 3]; 3], y: &[[f64; 3]; 3]| {
        let mut z = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                z[i][j] = (0..3).map(|k| x[i][k] * y[k][j]).sum();
            }
        }
        z
    };
    let generator = |a: usize, t: &[[f64; 3]; 3]| {
        let w = omega(a);
        let (l, r) = (mul(&w, t), mul(t, &w));
        let mut z = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                z[i][j] = l[i][j] - r[i][j];
            }
        }
        z
    };
    let dot = |x: &[[f64; 3]; 3], y: &[[f64; 3]; 3]| -> f64 {
        (0..3)
            .map(|i| (0..3).map(|j| x[i][j] * y[i][j]).sum::<f64>())
            .sum()
    };
    RMatrix::from_fn(5, 5, |mu, nu| {
        let mut m = 0.0;
        for (a, row) in d.iter().enumerate() {
            for (b, &dab) in row.iter().enumerate() {
                if dab != 0.0 {
                    let ab = generator(a, &generator(b, &e[nu]));
                    let ba = generator(b, &generator(a, &e[nu]));
                    m -= dab * 0.5 * (dot(&e[mu], &ab) + dot(&e[mu], &ba));
                }
            }
        }
        m
    })
}

/// An interaction's spatial tensor (molecular frame) and the interaction.
pub(in crate::spin) type Tensor = ([[f64; 3]; 3], Interaction);

/// One exponential of the correlation functions: `C_kl(τ) ∝ Σ w_k w_l
/// e^{−rate·τ}` over the modes.
#[derive(Clone, Debug)]
pub(in crate::spin) struct Mode {
    pub rate: f64,
    pub weights: Vec<f64>,
}

impl SpinSystem {
    /// The same system with rotational diffusion tensor `D` (s⁻¹, molecular
    /// frame; finite, symmetric, positive definite) in place of isotropic
    /// tumbling, or without it (`None`, which leaves a correlation time
    /// alone). It replaces a correlation time; `D = d·1` is isotropic
    /// tumbling with `τ_c = 1/(6d)`. Declared symmetries must keep it.
    pub fn with_diffusion(mut self, tensor: Option<[[f64; 3]; 3]>) -> Result<Self> {
        if let Some(d) = &tensor {
            check_diffusion(d)?;
            self.correlation_time = None;
        }
        self.diffusion = tensor;
        self.validate_symmetries()?;
        Ok(self)
    }

    /// The rotational diffusion tensor (s⁻¹), if set.
    pub fn diffusion(&self) -> Option<[[f64; 3]; 3]> {
        self.diffusion
    }

    /// Give `interactions` one shared internal motion (they leave any group
    /// they were in), or move only with the molecule (`None`). Interactions
    /// that do not relax (spins without positions or shielding tensor) are
    /// kept and take effect once they do. Declared symmetries must keep the
    /// groups; on an error nothing changes.
    pub fn set_internal_motion(
        &mut self,
        interactions: &[Interaction],
        motion: Option<ModelFree>,
    ) -> Result<&mut Self> {
        let list: Vec<Interaction> = interactions.iter().map(|k| k.normalized()).collect();
        for k in &list {
            match *k {
                Interaction::Dipolar(i, j) => {
                    self.check_spin(i)?;
                    self.check_spin(j)?;
                    if i == j {
                        return Err(QsoError::invalid(
                            "interaction",
                            "a dipolar coupling joins two different spins",
                        ));
                    }
                }
                Interaction::Shielding(i) => {
                    self.check_spin(i)?;
                }
            }
        }
        if let Some(m) = &motion {
            m.validate()?;
        }
        let old = self.internal_motions.clone();
        for group in &mut self.internal_motions {
            group.interactions.retain(|k| !list.contains(k));
        }
        self.internal_motions.retain(|g| !g.interactions.is_empty());
        if let Some(motion) = motion {
            let mut interactions = list;
            interactions.sort();
            interactions.dedup();
            if !interactions.is_empty() {
                self.internal_motions.push(InternalMotion {
                    interactions,
                    motion,
                });
            }
        }
        if let Err(e) = self.validate_symmetries() {
            self.internal_motions = old;
            return Err(e);
        }
        Ok(self)
    }

    /// The internal motion of an interaction, if it has one.
    pub fn internal_motion(&self, interaction: Interaction) -> Option<ModelFree> {
        let k = interaction.normalized();
        self.internal_motions
            .iter()
            .find(|g| g.interactions.contains(&k))
            .map(|g| g.motion)
    }

    pub(in crate::spin) fn validate_internal_motions(&self) -> Result<()> {
        if let Some(d) = &self.diffusion {
            check_diffusion(d)?;
            if self.correlation_time.is_some() {
                return Err(QsoError::invalid(
                    "motion",
                    "a correlation time and a diffusion tensor exclude each other",
                ));
            }
        }
        let n = self.spins.len();
        let mut seen = std::collections::BTreeSet::new();
        for g in &self.internal_motions {
            g.motion.validate()?;
            if g.interactions.is_empty() {
                return Err(QsoError::invalid(
                    "internal_motion",
                    "a group of internal motion needs an interaction",
                ));
            }
            for k in &g.interactions {
                let ok = match *k {
                    Interaction::Dipolar(i, j) => i < j && j < n,
                    Interaction::Shielding(i) => i < n,
                };
                if !ok || !seen.insert(*k) {
                    return Err(QsoError::invalid(
                        "internal_motion",
                        "each interaction of existing spins in at most one group",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Whether the overall motion is set (a correlation time or a diffusion
    /// tensor).
    pub(in crate::spin) fn has_motion(&self) -> bool {
        self.correlation_time.is_some() || self.diffusion.is_some()
    }

    /// The modes of the correlation functions of `interactions` (spatial
    /// tensors in the molecular frame), modes whose weights all vanish left
    /// out; none without motion.
    pub(in crate::spin) fn modes(
        &self,
        interactions: &[([[f64; 3]; 3], Interaction)],
    ) -> Vec<Mode> {
        let e = basis();
        let overall: Vec<(f64, [[f64; 3]; 3])> = if let Some(tau) = self.correlation_time {
            e.iter().map(|t| (1.0 / tau, *t)).collect()
        } else if let Some(d) = &self.diffusion {
            let eigen = diffusion_operator(d).symmetric_eigen();
            (0..5)
                .map(|j| {
                    let mut t = [[0.0; 3]; 3];
                    for (mu, basis_tensor) in e.iter().enumerate() {
                        let c = eigen.eigenvectors[(mu, j)];
                        for x in 0..3 {
                            for y in 0..3 {
                                t[x][y] += c * basis_tensor[x][y];
                            }
                        }
                    }
                    (eigen.eigenvalues[j], t)
                })
                .collect()
        } else {
            return Vec::new();
        };
        let contract = |a: &[[f64; 3]; 3], t: &[[f64; 3]; 3]| -> f64 {
            (0..3)
                .map(|x| (0..3).map(|y| a[x][y] * t[x][y]).sum::<f64>())
                .sum()
        };
        // The group of each interaction, if it moves internally.
        let group: Vec<Option<usize>> = interactions
            .iter()
            .map(|(_, k)| {
                self.internal_motions
                    .iter()
                    .position(|g| g.interactions.contains(k))
            })
            .collect();
        let order: Vec<f64> = group
            .iter()
            .map(|g| {
                g.map_or(1.0, |g| {
                    self.internal_motions[g].motion.order_parameter().sqrt()
                })
            })
            .collect();
        let mut modes = Vec::new();
        for (rate, t) in &overall {
            let projection: Vec<f64> = interactions.iter().map(|(a, _)| contract(a, t)).collect();
            modes.push(Mode {
                rate: *rate,
                weights: projection.iter().zip(&order).map(|(p, s)| p * s).collect(),
            });
            for (g, motion) in self.internal_motions.iter().enumerate() {
                for &(w, internal) in &motion.motion.terms()[1..] {
                    if w <= 0.0 {
                        continue;
                    }
                    let root = w.sqrt();
                    modes.push(Mode {
                        rate: rate + internal,
                        weights: projection
                            .iter()
                            .zip(&group)
                            .map(|(p, h)| if *h == Some(g) { root * p } else { 0.0 })
                            .collect(),
                    });
                }
            }
        }
        modes.retain(|m| m.weights.iter().any(|w| *w != 0.0));
        modes
    }

    /// The tensor of an active interaction at unit field (for the shielding,
    /// per tesla), and the interactions with it.
    fn tensor_of(&self, k: Interaction) -> Result<(Vec<Tensor>, usize)> {
        let interactions = self.interactions(1.0);
        let k = k.normalized();
        let index = interactions
            .iter()
            .position(|(_, c)| *c == k)
            .ok_or_else(|| {
                QsoError::invalid(
                    "interaction",
                    format!("{k:?} does not relax (no positions or shielding tensor)"),
                )
            })?;
        Ok((interactions, index))
    }

    /// The normalised weights `w_k/|A^k|` and `w_l/|A^l|` of every mode.
    fn normalised_modes(&self, k: Interaction, l: Interaction) -> Result<Vec<(f64, f64, f64)>> {
        if !self.has_motion() {
            return Err(QsoError::invalid(
                "motion",
                "set a correlation time or a diffusion tensor first",
            ));
        }
        let (interactions, a) = self.tensor_of(k)?;
        let (_, b) = self.tensor_of(l)?;
        let norm = |i: usize| {
            interactions[i]
                .0
                .iter()
                .flatten()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt()
        };
        let (na, nb) = (norm(a), norm(b));
        Ok(self
            .modes(&interactions)
            .into_iter()
            .map(|m| (m.rate, m.weights[a] / na, m.weights[b] / nb))
            .collect())
    }

    /// The normalised correlation function of two interactions,
    /// `C_kl(τ) = Σ_modes ŵ_k ŵ_l e^{−λτ}` with `ŵ = w/|A|`: `C_kk(0) = 1`,
    /// and for a rigid molecule tumbling isotropically `(Â^k : Â^l)
    /// e^{−τ/τ_c}` (`e^{−τ/τ_c}` for `k = l`, `±P₂(cos β)` times it for two
    /// axial tensors at an angle `β`). The
    /// relaxation by `k` and `l` is that of 1.34 with `(A^k : A^l)
    /// e^{−τ/τ_c}` replaced by `|A^k||A^l| C_kl(τ)`.
    pub fn correlation_function(&self, k: Interaction, l: Interaction, tau: f64) -> Result<f64> {
        if !(tau.is_finite() && tau >= 0.0) {
            return Err(QsoError::invalid("tau", "the delay must be finite and ≥ 0"));
        }
        Ok(self
            .normalised_modes(k, l)?
            .iter()
            .map(|&(rate, a, b)| a * b * (-rate * tau).exp())
            .sum())
    }

    /// The normalised spectral density `J_kl(ω) = ∫₀^∞ C_kl(τ) cos(ωτ) dτ =
    /// Σ_modes ŵ_k ŵ_l λ/(λ² + ω²)` (s; `ω` in rad/s): `J_kk =
    /// τ_c/(1 + ω²τ_c²)` for a rigid molecule tumbling isotropically (`J_kl`
    /// carries `Â^k : Â^l`), 5/2 times the
    /// `J(ω) = (2/5) Σ A_i τ_i/(1 + ω²τ_i²)` of the NMR literature.
    pub fn spectral_density(&self, k: Interaction, l: Interaction, omega: f64) -> Result<f64> {
        if !omega.is_finite() {
            return Err(QsoError::invalid("omega", "the frequency must be finite"));
        }
        Ok(self
            .normalised_modes(k, l)?
            .iter()
            .map(|&(rate, a, b)| a * b * rate / (rate * rate + omega * omega))
            .sum())
    }
}
