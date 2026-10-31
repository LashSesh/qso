//! The generalised cluster-correlation expansion of an NV centre's bath:
//! the electron and each cluster of nuclei propagated together.
//!
//! The expansion of [`bath`](super::bath) holds the electron in one level
//! along each branch of the coherence, which needs ideal pulses and the
//! secular hyperfine coupling. Here every cluster carries the electron as
//! well (Onizhuk, Miao, Chou, Galli, Bhatt and Awschalom, PRX Quantum 2,
//! 010311 (2021)), with the spin Hamiltonian in the laboratory frame,
//!
//! ```text
//! H/2π = D S_z² + γ_e B·S + Σ_i [S·A_i·I_i − γ_C B·I_i]
//!      + Σ_{i<j} d_ij [I_i·I_j − 3 (I_i·n_ij)(I_j·n_ij)],
//! ```
//!
//! the field `B` in any direction, the full hyperfine tensors `A_i`, and
//! the dipolar couplings of the pairs closer than the cutoff. The qubit
//! `{α, β}` is formed by the eigenstates of the electron's own Hamiltonian
//! `D S_z² + γ_e B·S` closest to `m_s = 0` and `m_s = m`; the electron
//! starts in `α`, the nuclei unpolarised (or in sampled product states,
//! below).
//!
//! Pulses act in the frame rotating at the microwave frequency `f`, which
//! turns the electron's levels other than `α` (the frame of
//! [`NvCenter`](super::NvCenter), `exp(−2πi f t S_z²)` for a field along
//! the axis); `f` is resonant with the bare qubit unless set. Ideal pulses
//! rotate the qubit in that frame; finite pulses of Rabi frequency `Ω`,
//! rectangular or shaped, drive both transitions from `α` in the
//! rotating-wave approximation, and leave out, while they last, the terms
//! that the frame makes oscillate at `f` (a transverse field, the
//! non-secular hyperfine coupling). Free periods are exact.
//! [`Pulses::Laboratory`] applies ideal pulses about axes fixed in the
//! laboratory frame instead, as PyCCE does.
//!
//! The non-secular coupling mixes the electron's levels with the nuclei to
//! first order in `A/Δ` (`Δ` the electron's transitions, GHz): the
//! populations and coherences of the bare levels oscillate at the
//! electron's frequencies with amplitudes of order `A/Δ`, which the other
//! nuclei dephase. A product over clusters cannot follow that dephasing,
//! and for nuclei coupled by a MHz or more (closer than about 0.3 nm) the
//! expansion stalls or diverges. With [`GeneralisedBath::dressed`] each
//! cluster's Hamiltonian is taken block-diagonal between `α` and the other
//! levels, by the direct rotation closest to the identity (computed
//! exactly, not by perturbation theory): the electron is prepared, pulsed
//! and read out in its levels dressed by the cluster's nuclei, the fast
//! oscillations are gone, and the mixing acts on the nuclei to all orders
//! (shifts of order `A²/Δ`, couplings between nuclei through the
//! electron). Near a crossing of `α` with another level (0.1024 T for
//! `m_s = −1`) the dressed levels do not exist and are refused.
//!
//! The electron's density matrix just before the last pulse, `ρ_C` for
//! each cluster `C` (and `ρ_∅` for the electron alone), expands element by
//! element on the qubit `{α, β}`, whose elements the coherence and the
//! readout take (the third level's elements, and elements of the electron
//! alone below 10⁻⁹, keep the electron's values):
//!
//! ```text
//! ρ = ρ_∅ ⊙ Π_C r̃_C,   r_C = ρ_C ⊘ ρ_∅,   r̃_C = r_C ⊘ Π_{∅≠C′⊊C} r̃_{C′},
//! ```
//!
//! truncated at clusters of `order` nuclei and exact once `order` reaches
//! the largest connected cluster. The last pulse, ideal, reads out the
//! population of `α`.
//!
//! With unpolarised nuclei a product over clusters represents only what
//! factorises over them: phases from independent nuclei do, but a finite
//! pulse depends on the electron's detuning by all nuclei together, and
//! the expansion then moves away from the exact result as the order grows.
//! [`GeneralisedCce::bath_states`] samples product states of the nuclei
//! (up or down along the field) and expands in each, every cluster in the
//! static field of the nuclei outside it, on the electron and, within the
//! cutoff, on its own nuclei (PyCCE's bath sampling with its mean field);
//! the results are averaged, with their standard errors.

use std::collections::{HashMap, HashSet};

use nalgebra::Matrix3;
use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::bath::{BathSpin, SpinBath};
use super::{Microwave, NV_GYROMAGNETIC_RATIO, NV_ZERO_FIELD_SPLITTING, NvCenter, NvEvent};
use crate::error::{QsoError, Result};
use crate::linalg::dense::{cmul, from_spectrum};
use crate::linalg::{CMatrix, eigh};
use crate::spin::{HBAR, Isotope, MU0_OVER_4PI, TWO_PI, ZERO, spin_ops};

/// Largest cluster of the generalised expansion (dimension `3·2⁸ = 768`).
pub const MAX_GENERALISED_ORDER: usize = 8;

/// Most nuclei of an exact propagation of the electron with its whole
/// bath.
pub const MAX_GENERALISED_EXACT: usize = 8;

/// Most operations (estimated) of one expansion: a few minutes on one
/// core.
pub const MAX_GENERALISED_WORK: f64 = 5e11;

/// Most 3 × 3 matrices an expansion holds at once (144 bytes each): the
/// contributions of the clusters below the order (for each bath state in
/// flight when sampled) and a result per bath state, each for every
/// sequence.
pub const MAX_GENERALISED_MATRICES: usize = 1 << 22;

/// Most sampled bath states ([`GeneralisedCce::bath_states`]).
pub const MAX_BATH_STATES: usize = 1 << 16;

/// The electron's density matrix (eigenbasis of its own Hamiltonian).
type Rho = Matrix3<C64>;

/// A nucleus of the bath with its full hyperfine tensor `A` (Hz), the
/// coupling `S·A·I` with the electron (row: electron component, column:
/// nuclear component).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FullSpin {
    pub position: [f64; 3],
    pub hyperfine: [[f64; 3]; 3],
}

/// `(μ₀/4π) γ_e γ_C h` (Hz m³).
fn hyperfine_scale() -> f64 {
    MU0_OVER_4PI * NV_GYROMAGNETIC_RATIO * gamma_c() * TWO_PI * HBAR
}

/// `γ_C/2π` (Hz/T).
fn gamma_c() -> f64 {
    Isotope::C13.gamma() / TWO_PI
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

impl FullSpin {
    /// A ¹³C at `position` (m, NV frame) with the hyperfine tensor of a
    /// point dipole at the vacancy, `A = (μ₀/4π) γ_e γ_C h (3 n nᵀ − 1)/r³`
    /// (the electron's moment is opposite to its spin; the row `A_z·` is
    /// [`BathSpin::point_dipole`]'s coupling).
    pub fn point_dipole(position: [f64; 3]) -> Result<Self> {
        let r = norm(position);
        if !(r.is_finite() && r > 0.0) {
            return Err(QsoError::invalid(
                "position",
                "a finite position away from the vacancy",
            ));
        }
        let n = position.map(|x| x / r);
        let k = hyperfine_scale() / (r * r * r);
        let mut hyperfine = [[0.0; 3]; 3];
        for (a, row) in hyperfine.iter_mut().enumerate() {
            for (b, x) in row.iter_mut().enumerate() {
                *x = k * (3.0 * n[a] * n[b] - if a == b { 1.0 } else { 0.0 });
            }
        }
        Ok(FullSpin {
            position,
            hyperfine,
        })
    }
}

/// Most segments of the envelope of a shaped pulse.
pub const MAX_ENVELOPE_SEGMENTS: usize = 1024;

/// How the pulses of a sequence act.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Pulses {
    /// Instantaneous rotations of the qubit in the rotating frame (the limit
    /// of finite pulses).
    #[default]
    Ideal,
    /// Rectangular pulses of Rabi frequency `rabi` (Hz) on the qubit, in the
    /// rotating-wave approximation: a microwave field along the x axis of
    /// the centre, scaled so that it turns the qubit at `rabi` about the
    /// axes of the ideal pulses (and the other transition from `α` at its
    /// own strength). A pulse of angle `θ` lasts `θ/(360° Ω)`, and free
    /// periods count between pulse centres. The last pulse reads out
    /// ideally.
    Finite { rabi: f64 },
    /// Shaped pulses: the envelope's segments, of equal length, each a
    /// relative amplitude (≥ 0) and a phase (degrees) added to the pulse's,
    /// at the Rabi frequency `rabi` (Hz) for amplitude 1. A pulse of angle
    /// `θ` lasts `θ/(360° Ω ā)`, `ā` the mean amplitude, so that the area
    /// of its envelope is `θ`; otherwise as [`Pulses::Finite`].
    Shaped {
        rabi: f64,
        envelope: Vec<(f64, f64)>,
    },
    /// Instantaneous rotations about axes fixed in the laboratory frame
    /// (PyCCE's convention; the coherence then carries the electron's
    /// precession, which the normalisation by the electron alone removes).
    Laboratory,
}

/// The envelope of a rectangular pulse.
const RECTANGULAR: &[(f64, f64)] = &[(1.0, 0.0)];

impl Pulses {
    /// The Rabi frequency for amplitude 1 and the envelope of finite and
    /// shaped pulses; `None` for instantaneous ones.
    fn drive(&self) -> Option<(f64, &[(f64, f64)])> {
        match self {
            Pulses::Finite { rabi } => Some((*rabi, RECTANGULAR)),
            Pulses::Shaped { rabi, envelope } => Some((*rabi, envelope)),
            Pulses::Ideal | Pulses::Laboratory => None,
        }
    }

    /// The Rabi frequency of the rectangular pulse of the same area (which
    /// sets the durations); `None` for instantaneous pulses.
    fn area_rabi(&self) -> Option<f64> {
        self.drive().map(|(rabi, envelope)| {
            rabi * envelope.iter().map(|s| s.0).sum::<f64>() / envelope.len() as f64
        })
    }

    /// Check the Rabi frequency and the envelope.
    pub fn validate(&self) -> Result<()> {
        let Some((rabi, envelope)) = self.drive() else {
            return Ok(());
        };
        if !(rabi.is_finite() && rabi > 0.0) {
            return Err(QsoError::invalid(
                "rabi",
                "a positive, finite Rabi frequency",
            ));
        }
        if envelope.is_empty() || envelope.len() > MAX_ENVELOPE_SEGMENTS {
            return Err(QsoError::invalid(
                "envelope",
                format!("1 to {MAX_ENVELOPE_SEGMENTS} segments"),
            ));
        }
        if envelope
            .iter()
            .any(|&(a, phase)| !(a.is_finite() && a >= 0.0 && phase.is_finite()))
        {
            return Err(QsoError::invalid(
                "envelope",
                "finite amplitudes ≥ 0 and finite phases",
            ));
        }
        if !self.area_rabi().is_some_and(|r| r > 0.0 && r.is_finite()) {
            return Err(QsoError::invalid(
                "envelope",
                "an envelope with a positive, finite area",
            ));
        }
        Ok(())
    }
}

/// The generalised expansion: clusters of up to `order` nuclei connected by
/// pairs closer than `cutoff` (m, whose couplings are kept), how the
/// pulses act, and the state of the nuclei.
///
/// With `bath_states = 0` every cluster starts unpolarised. Otherwise the
/// expansion runs in each of `bath_states` product states of the nuclei,
/// each nucleus up or down along the field (seeded by `seed`), and the
/// results are averaged: every cluster then feels the static field of the
/// nuclei outside it (on the electron, and on its own nuclei from those
/// within the cutoff), as in PyCCE's bath sampling. A finite pulse depends
/// on the electron's detuning by all nuclei together, which a product over
/// clusters of unpolarised nuclei cannot represent; in a sampled state that
/// detuning is in every cluster.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralisedCce {
    pub order: usize,
    pub cutoff: f64,
    #[serde(default)]
    pub pulses: Pulses,
    #[serde(default)]
    pub bath_states: usize,
    #[serde(default)]
    pub seed: u64,
}

impl Default for GeneralisedCce {
    /// Second order, pairs within 0.8 nm, ideal pulses, unpolarised nuclei.
    fn default() -> Self {
        GeneralisedCce {
            order: 2,
            cutoff: 8e-10,
            pulses: Pulses::Ideal,
            bath_states: 0,
            seed: 0,
        }
    }
}

/// The coherence and the signal after a sequence with their standard errors
/// over the sampled bath states (0 for unpolarised nuclei, NaN for one
/// state).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BathEstimate {
    pub coherence: C64,
    pub coherence_error: f64,
    pub signal: f64,
    pub signal_error: f64,
}

/// What a sampled bath state adds to a cluster: the static field (Hz, by
/// the electron's components) of the nuclei outside it on the electron,
/// the static field (Hz) of those within the cutoff on each member, and the
/// members' states (spinors, basis `m = +½, −½`).
struct MeanField {
    electron: [f64; 3],
    nuclei: Vec<[f64; 3]>,
    spinors: Vec<[C64; 2]>,
}

/// The couplings through which sampled bath states act on clusters: the
/// axis of the states, the hyperfine tensors and, for each nucleus, its
/// neighbours within the cutoff with their dipolar tensors (Hz).
struct BathFields {
    axis: [f64; 3],
    hyperfine: Vec<[[f64; 3]; 3]>,
    neighbours: Vec<Vec<(usize, [[f64; 3]; 3])>>,
}

impl BathFields {
    /// The mean field on the cluster `members` in the bath state with the
    /// projections `m` (±½ along the axis).
    fn mean_field(&self, m: &[f64], members: &[usize]) -> MeanField {
        let spin = |j: usize| self.axis.map(|x| x * m[j]);
        let inside = |j: usize| members.contains(&j);
        let mut electron = [0.0; 3];
        for (j, a) in self.hyperfine.iter().enumerate() {
            if inside(j) {
                continue;
            }
            let v = spin(j);
            for (c, row) in a.iter().enumerate() {
                electron[c] += row[0] * v[0] + row[1] * v[1] + row[2] * v[2];
            }
        }
        let nuclei = members
            .iter()
            .map(|&i| {
                let mut g = [0.0; 3];
                for &(j, ref d) in &self.neighbours[i] {
                    if inside(j) {
                        continue;
                    }
                    let v = spin(j);
                    for (a, row) in d.iter().enumerate() {
                        g[a] += row[0] * v[0] + row[1] * v[1] + row[2] * v[2];
                    }
                }
                g
            })
            .collect();
        // |±⟩ along the axis (polar angles θ, φ): (cos θ/2, e^{iφ} sin θ/2)
        // and (−e^{−iφ} sin θ/2, cos θ/2).
        let [x, y, z] = self.axis;
        let (half, phi) = (z.clamp(-1.0, 1.0).acos() / 2.0, y.atan2(x));
        let up = [C64::new(half.cos(), 0.0), C64::from_polar(half.sin(), phi)];
        let down = [
            C64::from_polar(-half.sin(), -phi),
            C64::new(half.cos(), 0.0),
        ];
        let spinors = members
            .iter()
            .map(|&i| if m[i] > 0.0 { up } else { down })
            .collect();
        MeanField {
            electron,
            nuclei,
            spinors,
        }
    }
}

/// An NV centre's electron (spin 1) and its nuclear bath in a field of any
/// direction (see the module documentation).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralisedBath {
    /// Field (T) in the frame of the centre (z along the NV axis).
    pub field: [f64; 3],
    /// The qubit `{0, transition}`: `−1` or `+1`.
    pub transition: i8,
    /// Zero-field splitting `D` (Hz).
    pub zero_field_splitting: f64,
    /// The electron's gyromagnetic ratio (Hz/T).
    pub gyromagnetic_ratio: f64,
    /// Microwave frequency (Hz) of the rotating frame: `None` for the bare
    /// qubit's transition.
    pub frequency: Option<f64>,
    pub spins: Vec<FullSpin>,
    /// Whether the electron's levels are those dressed by the nuclei of
    /// each cluster: its Hamiltonian block-diagonal between `α` and the
    /// other levels (see the module documentation).
    #[serde(default)]
    pub dressed: bool,
}

/// The electron's own Hamiltonian diagonalised: its eigenvectors (columns,
/// in the basis `m = +1, 0, −1`), the eigenvalues (rad/s), and which of
/// them are `α`, `β` and the third level `γ`.
struct Electron {
    vectors: CMatrix,
    values: [f64; 3],
    alpha: usize,
    beta: usize,
    /// `⟨β|S_x|α⟩`: how the drive couples the qubit (its magnitude and the
    /// phase of the eigenvectors).
    coupling: C64,
}

/// One sequence ready to run: its steps before the last pulse, the pulses
/// between the first and the last (which carry the branches of the
/// coherence), and the last pulse (ideal, on the electron).
struct Sequence {
    steps: Vec<NvEvent>,
    inner: Vec<(f64, f64)>,
    last: (f64, f64),
}

impl GeneralisedBath {
    /// The centre with `D` and `γ_e` of [`NvCenter`] and the frame resonant
    /// with the bare qubit.
    pub fn new(field: [f64; 3], transition: i8, spins: Vec<FullSpin>) -> Self {
        GeneralisedBath {
            field,
            transition,
            zero_field_splitting: NV_ZERO_FIELD_SPLITTING,
            gyromagnetic_ratio: NV_GYROMAGNETIC_RATIO,
            frequency: None,
            spins,
            dressed: false,
        }
    }

    /// The bath of 1.39 (field along the axis): each nucleus with the
    /// tensor of a point dipole at its position whose row `A_z·` is the
    /// bath's coupling. For the baths of [`diamond_bath`] the two agree;
    /// couplings given for nuclei next to the vacancy keep their row, the
    /// others those of the point dipole (give the whole tensor as a
    /// [`FullSpin`] where the non-secular terms of such a nucleus matter).
    ///
    /// [`diamond_bath`]: super::bath::diamond_bath
    pub fn from_bath(bath: &SpinBath) -> Result<Self> {
        bath.validate()?;
        let spins = bath
            .spins
            .iter()
            .map(|s| {
                let mut full = FullSpin::point_dipole(s.position)?;
                full.hyperfine[2] = s.hyperfine;
                Ok(full)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::new([0.0, 0.0, bath.field], bath.transition, spins))
    }

    /// A random bath of [`diamond_bath`] (the same nuclei for the same
    /// seed) with the point-dipole tensors, in the field `field` (T, any
    /// direction).
    ///
    /// [`diamond_bath`]: super::bath::diamond_bath
    pub fn diamond(
        field: [f64; 3],
        transition: i8,
        abundance: f64,
        radius: f64,
        seed: u64,
    ) -> Result<Self> {
        let spins = super::bath::diamond_bath(abundance, radius, seed)?
            .iter()
            .map(|s| FullSpin::point_dipole(s.position))
            .collect::<Result<Vec<_>>>()?;
        let bath = Self::new(field, transition, spins);
        bath.validate()?;
        Ok(bath)
    }

    /// The bath without the nuclei closer than `radius` (m) to the vacancy.
    pub fn without_spins_within(mut self, radius: f64) -> Self {
        self.spins.retain(|s| norm(s.position) >= radius);
        self
    }

    /// The same nuclei as a bath of 1.39 (for its clusters).
    fn as_bath(&self) -> SpinBath {
        SpinBath::new(
            self.field[2],
            self.transition,
            self.spins
                .iter()
                .map(|s| BathSpin {
                    position: s.position,
                    hyperfine: s.hyperfine[2],
                })
                .collect(),
        )
    }

    /// Check the field, the electron's parameters and the nuclei.
    pub fn validate(&self) -> Result<()> {
        if self.field.iter().any(|x| !x.is_finite())
            || !self.zero_field_splitting.is_finite()
            || !self.gyromagnetic_ratio.is_finite()
            || self.frequency.is_some_and(|f| !f.is_finite())
        {
            return Err(QsoError::invalid(
                "bath",
                "the field, D, γ_e and the frequency must be finite",
            ));
        }
        if self
            .spins
            .iter()
            .any(|s| s.hyperfine.iter().flatten().any(|x| !x.is_finite()))
        {
            return Err(QsoError::invalid(
                "spins",
                "hyperfine tensors must be finite",
            ));
        }
        self.as_bath().validate()
    }

    /// The connected clusters of up to `order` nuclei (pairs closer than
    /// `cutoff`), as for the expansion of 1.39.
    pub fn clusters(&self, order: usize, cutoff: f64) -> Result<Vec<Vec<usize>>> {
        self.validate()?;
        if order > MAX_GENERALISED_ORDER {
            return Err(QsoError::invalid(
                "order",
                format!("clusters of 1 to {MAX_GENERALISED_ORDER} nuclei"),
            ));
        }
        self.as_bath().clusters(order, cutoff)
    }

    /// The electron's Hamiltonian `D S_z² + γ_e B·S` diagonalised.
    fn electron(&self) -> Electron {
        let [sz, _, _, sx, sy] = spin_ops(3);
        let c = |x: f64| C64::new(TWO_PI * x, 0.0);
        let g = self.gyromagnetic_ratio;
        let h = &sz * &sz * c(self.zero_field_splitting)
            + &sx * c(g * self.field[0])
            + &sy * c(g * self.field[1])
            + &sz * c(g * self.field[2]);
        let (values, vectors) = eigh(&h);
        // The eigenstate closest to |m_s⟩ (basis +1, 0, −1), each taken once.
        let closest = |level: usize, taken: &[usize]| -> usize {
            (0..3)
                .filter(|k| !taken.contains(k))
                .max_by(|&a, &b| {
                    vectors[(level, a)]
                        .norm_sqr()
                        .total_cmp(&vectors[(level, b)].norm_sqr())
                })
                .expect("three eigenvectors")
        };
        let alpha = closest(1, &[]);
        let beta = closest(if self.transition < 0 { 2 } else { 0 }, &[alpha]);
        let coupling = (vectors.adjoint() * sx * &vectors)[(beta, alpha)];
        Electron {
            vectors,
            values: [values[0], values[1], values[2]],
            alpha,
            beta,
            coupling,
        }
    }

    /// The frame's frequency (Hz): set, or the bare qubit's transition
    /// `(E_β − E_α)/h` (negative where `β` lies below `α`).
    pub fn frame_frequency(&self) -> f64 {
        self.frequency.unwrap_or_else(|| {
            let e = self.electron();
            (e.values[e.beta] - e.values[e.alpha]) / TWO_PI
        })
    }

    /// The Hamiltonian (rad/s) of the electron and the nuclei `members` in
    /// the electron's eigenbasis (index `e·2^k + s`, nucleus `a` of the
    /// cluster at bit `k − 1 − a`), with the dipolar couplings of the pairs
    /// closer than `cutoff`.
    fn hamiltonian(
        &self,
        members: &[usize],
        cutoff: f64,
        electron: &Electron,
        mean: Option<&MeanField>,
    ) -> CMatrix {
        let k = members.len();
        let n = 1usize << k;
        let d = 3 * n;
        let mut h = CMatrix::zeros(d, d);
        // The electron: diagonal in its eigenbasis.
        for e in 0..3 {
            for s in 0..n {
                h[(e * n + s, e * n + s)] += C64::new(electron.values[e], 0.0);
            }
        }
        let [sz, _, _, sx, sy] = spin_ops(3);
        let rotate = |m: &CMatrix| electron.vectors.adjoint() * m * &electron.vectors;
        let s_eig = [rotate(&sx), rotate(&sy), rotate(&sz)];
        // The static field of the nuclei outside the cluster on the electron.
        if let Some(mean) = mean {
            let mut part = CMatrix::zeros(3, 3);
            for (c, sc) in s_eig.iter().enumerate() {
                part += sc * C64::new(TWO_PI * mean.electron[c], 0.0);
            }
            for e in 0..3 {
                for f in 0..3 {
                    for s in 0..n {
                        h[(e * n + s, f * n + s)] += part[(e, f)];
                    }
                }
            }
        }
        let [iz, _, _, ix, iy] = spin_ops(2);
        let i_ops = [ix, iy, iz];
        let bit = |a: usize| k - 1 - a;
        // Nuclear Zeeman and hyperfine terms.
        let gc = gamma_c();
        for (a, &i) in members.iter().enumerate() {
            let b_bit = bit(a);
            let tensor = self.spins[i].hyperfine;
            // Σ_c A[c][b] S_c (electron, eigenbasis) for each nuclear axis b.
            for (nb, ib) in i_ops.iter().enumerate() {
                let mut electron_part = CMatrix::zeros(3, 3);
                for (c, sc) in s_eig.iter().enumerate() {
                    if tensor[c][nb] != 0.0 {
                        electron_part += sc * C64::new(TWO_PI * tensor[c][nb], 0.0);
                    }
                }
                let zeeman =
                    -TWO_PI * gc * self.field[nb] + mean.map_or(0.0, |m| TWO_PI * m.nuclei[a][nb]);
                for e in 0..3 {
                    for f in 0..3 {
                        let mut coefficient = electron_part[(e, f)];
                        if e == f {
                            coefficient += C64::new(zeeman, 0.0);
                        }
                        if coefficient == ZERO {
                            continue;
                        }
                        for s in 0..n {
                            let x = (s >> b_bit) & 1;
                            for y in 0..2 {
                                let z = ib[(y, x)];
                                if z != ZERO {
                                    let t = (s & !(1 << b_bit)) | (y << b_bit);
                                    h[(e * n + t, f * n + s)] += coefficient * z;
                                }
                            }
                        }
                    }
                }
            }
        }
        // Dipolar couplings between the nuclei of the cluster.
        let dipolar = MU0_OVER_4PI * gc * gc * TWO_PI * HBAR;
        for (a, &i) in members.iter().enumerate() {
            for (b, &j) in members.iter().enumerate().skip(a + 1) {
                let (p, q) = (self.spins[i].position, self.spins[j].position);
                let v = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
                let r = norm(v);
                if r >= cutoff {
                    continue;
                }
                let nv = v.map(|x| x / r);
                let scale = TWO_PI * dipolar / (r * r * r);
                let (ba, bb) = (bit(a), bit(b));
                for (u, ou) in i_ops.iter().enumerate() {
                    for (w, ow) in i_ops.iter().enumerate() {
                        let c = scale * (if u == w { 1.0 } else { 0.0 } - 3.0 * nv[u] * nv[w]);
                        if c == 0.0 {
                            continue;
                        }
                        for s in 0..n {
                            let (xa, xb) = ((s >> ba) & 1, (s >> bb) & 1);
                            for ya in 0..2 {
                                let za = ou[(ya, xa)];
                                if za == ZERO {
                                    continue;
                                }
                                for yb in 0..2 {
                                    let zb = ow[(yb, xb)];
                                    if zb == ZERO {
                                        continue;
                                    }
                                    let t = (s & !(1 << ba) & !(1 << bb)) | (ya << ba) | (yb << bb);
                                    for e in 0..3 {
                                        h[(e * n + t, e * n + s)] += za * zb * c;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        h
    }

    /// Whether each electron level turns with the frame (all but `α`).
    fn turning(electron: &Electron) -> [f64; 3] {
        let mut chi = [1.0; 3];
        chi[electron.alpha] = 0.0;
        chi
    }

    /// An ideal pulse on the qubit (`|β⟩` first, as [`NvCenter`]'s pulses
    /// with `|m⟩` first): `exp(−iθ/2 (cos φ σ_x + sin φ σ_y))` on `{β, α}`,
    /// the identity on the third level (3 × 3, eigenbasis).
    fn ideal(electron: &Electron, angle: f64, phase: f64) -> CMatrix {
        let (c, s) = (
            (angle.to_radians() / 2.0).cos(),
            (angle.to_radians() / 2.0).sin(),
        );
        let p = C64::from_polar(1.0, -phase.to_radians());
        let mut r = CMatrix::identity(3, 3);
        let (b, a) = (electron.beta, electron.alpha);
        r[(b, b)] = C64::new(c, 0.0);
        r[(a, a)] = C64::new(c, 0.0);
        r[(b, a)] = C64::new(0.0, -s) * p;
        r[(a, b)] = C64::new(0.0, -s) * p.conj();
        r
    }

    /// The sequences: checked, adjacent free periods merged (with finite
    /// pulses, measured between pulse centres), split before the last
    /// pulse.
    fn parse(&self, sequences: &[Vec<NvEvent>], pulses: &Pulses) -> Result<Vec<Sequence>> {
        pulses.validate()?;
        if pulses.drive().is_some() && self.electron().coupling.norm() < 0.05 {
            return Err(QsoError::invalid(
                "pulses",
                "a drive along x hardly couples the qubit's levels in this field",
            ));
        }
        // Durations follow from the Rabi frequency of equal area.
        let drive = Microwave {
            frequency: 0.0,
            transition: self.transition,
            rabi: pulses.area_rabi(),
        };
        sequences
            .iter()
            .map(|events| {
                NvCenter::check_events(events)?;
                let Some(&NvEvent::Pulse { angle, phase }) = events.last() else {
                    return Err(QsoError::invalid(
                        "sequence",
                        "a sequence ends with the pulse that reads it out",
                    ));
                };
                // The last pulse reads out ideally and takes no time: the
                // periods before it count to the end.
                let steps = NvCenter::centred(&events[..events.len() - 1], &drive)?;
                let pulses: Vec<(f64, f64)> = events[..events.len() - 1]
                    .iter()
                    .filter_map(|e| match *e {
                        NvEvent::Pulse { angle, phase } => Some((angle, phase)),
                        NvEvent::Free(_) => None,
                    })
                    .collect();
                Ok(Sequence {
                    steps,
                    inner: pulses.iter().skip(1).copied().collect(),
                    last: (angle, phase),
                })
            })
            .collect()
    }

    /// The electron's density matrix (3 × 3, eigenbasis, rotating frame)
    /// just before the last pulse of each sequence, with the nuclei
    /// `members`: unpolarised, or in a sampled bath state (`mean`).
    #[allow(clippy::too_many_arguments)]
    fn cluster_states(
        &self,
        members: &[usize],
        sequences: &[Sequence],
        cutoff: f64,
        pulses: &Pulses,
        electron: &Electron,
        frequency: f64,
        mean: Option<&MeanField>,
    ) -> Result<Vec<Rho>> {
        let k = members.len();
        let n = 1usize << k;
        let d = 3 * n;
        let h = self.hamiltonian(members, cutoff, electron, mean);
        let (values, bare) = eigh(&h);
        let chi = Self::turning(electron);
        // The eigenvectors in the frame of the propagation: block-diagonal
        // in the dressed frame.
        let vectors = if self.dressed {
            Self::dress(&bare, &chi, n).ok_or_else(|| {
                QsoError::invalid(
                    "dressed",
                    format!(
                        "the electron's levels mix with the nuclei {members:?} (an avoided \
                         crossing): use the electron's bare levels (dressed = false)"
                    ),
                )
            })?
        } else {
            bare.clone()
        };
        let lab = matches!(pulses, Pulses::Laboratory);
        // The frame's phase on level e at time t: e^{−2πi f t χ_e}.
        let frame = |t: f64| -> [C64; 3] {
            let cycles = (frequency * t).fract();
            chi.map(|x| C64::from_polar(1.0, -TWO_PI * cycles * x))
        };
        // The Hamiltonian of finite pulses in the frame: the blocks of
        // levels that turn alike, less 2πf on the turning levels.
        let drive = pulses.drive();
        let pulse_base = drive.map(|_| {
            let mut hr = if self.dressed {
                from_spectrum(&values, &vectors, |e| C64::new(e, 0.0))
            } else {
                h.clone()
            };
            for r in 0..d {
                for c in 0..d {
                    if chi[r / n] != chi[c / n] {
                        hr[(r, c)] = ZERO;
                    }
                }
                hr[(r, r)] -= C64::new(TWO_PI * frequency * chi[r / n], 0.0);
            }
            hr
        });
        // The electron's S_x in the frame of the propagation (T† S_x T in
        // the dressed frame, T = V V′†).
        let sx_full = drive.map(|_| {
            let [_, _, _, sx, _] = spin_ops(3);
            let sx_eig = electron.vectors.adjoint() * sx * &electron.vectors;
            let mut x = CMatrix::zeros(d, d);
            for e in 0..3 {
                for f in 0..3 {
                    for s in 0..n {
                        x[(e * n + s, f * n + s)] = sx_eig[(e, f)];
                    }
                }
            }
            if self.dressed {
                let t = &bare * vectors.adjoint();
                t.adjoint() * x * t
            } else {
                x
            }
        });
        // The propagator of one segment of a pulse: its length, Rabi
        // frequency and phase (degrees).
        let segment = |base: &CMatrix, length: f64, rabi: f64, phase: f64| -> CMatrix {
            let mut hp = base.clone();
            let x = sx_full.as_ref().expect("a finite drive");
            // The co-rotating part of a drive along S_x, scaled so that it
            // turns the qubit at Ω about the axis at φ of the ideal pulses:
            // (2πΩ/2) e^{−iφ} |β⟩⟨α| (and the other transition from α at its
            // own strength).
            let up = C64::from_polar(TWO_PI * rabi / 2.0, -phase.to_radians()) / electron.coupling;
            for r in 0..d {
                for c in 0..d {
                    let z = x[(r, c)];
                    if z == ZERO {
                        continue;
                    }
                    hp[(r, c)] += match chi[r / n] - chi[c / n] {
                        y if y > 0.5 => up * z,
                        y if y < -0.5 => up.conj() * z,
                        _ => continue,
                    };
                }
            }
            let (pv, pw) = eigh(&hp);
            from_spectrum(&pv, &pw, |e| C64::from_polar(1.0, -e * length))
        };
        let mut free_cache: HashMap<u64, CMatrix> = HashMap::new();
        let mut pulse_cache: HashMap<(u64, u64), CMatrix> = HashMap::new();
        let mut segment_cache: HashMap<(u64, u64, u64), CMatrix> = HashMap::new();
        Ok(sequences
            .iter()
            .map(|seq| {
                // Columns |α⟩ ⊗ |s⟩: the unpolarised nuclei, one at a time;
                // or the one product state of a sampled bath.
                let mut psi = match mean {
                    None => {
                        let mut psi = CMatrix::zeros(d, n);
                        for s in 0..n {
                            psi[(electron.alpha * n + s, s)] = C64::new(1.0, 0.0);
                        }
                        psi
                    }
                    Some(mean) => CMatrix::from_fn(d, 1, |r, _| {
                        if r / n != electron.alpha {
                            return ZERO;
                        }
                        let s = r % n;
                        (0..k).fold(C64::new(1.0, 0.0), |z, a| {
                            z * mean.spinors[a][(s >> (k - 1 - a)) & 1]
                        })
                    }),
                };
                let cols = psi.ncols();
                let mut t = 0.0f64;
                for step in &seq.steps {
                    match *step {
                        NvEvent::Free(tau) => {
                            if tau == 0.0 {
                                continue;
                            }
                            let x = free_cache.entry(tau.to_bits()).or_insert_with(|| {
                                from_spectrum(&values, &vectors, |e| C64::from_polar(1.0, -e * tau))
                            });
                            if lab {
                                psi = cmul(x, &psi);
                            } else {
                                // U_f(t + τ)† e^{−iHτ} U_f(t), on the columns.
                                let (before, after) = (frame(t), frame(t + tau));
                                let mut moved = psi.clone();
                                for r in 0..d {
                                    let ph = before[r / n];
                                    for c in 0..cols {
                                        moved[(r, c)] *= ph;
                                    }
                                }
                                psi = cmul(x, &moved);
                                for r in 0..d {
                                    let ph = after[r / n].conj();
                                    for c in 0..cols {
                                        psi[(r, c)] *= ph;
                                    }
                                }
                            }
                            t += tau;
                        }
                        NvEvent::Pulse { angle, phase } => match (drive, &pulse_base) {
                            (Some((rabi, envelope)), Some(base)) => {
                                let phase = if angle < 0.0 { phase + 180.0 } else { phase };
                                let area = pulses.area_rabi().expect("a finite drive");
                                let duration = angle.abs() / 360.0 / area;
                                let key = (duration.to_bits(), phase.to_bits());
                                let u = pulse_cache.entry(key).or_insert_with(|| {
                                    // The segments in turn (equal ones once).
                                    let length = duration / envelope.len() as f64;
                                    let mut u: Option<CMatrix> = None;
                                    for &(amplitude, offset) in envelope {
                                        let key = (
                                            length.to_bits(),
                                            amplitude.to_bits(),
                                            (phase + offset).to_bits(),
                                        );
                                        let x = segment_cache.entry(key).or_insert_with(|| {
                                            segment(base, length, rabi * amplitude, phase + offset)
                                        });
                                        u = Some(match u {
                                            None => x.clone(),
                                            Some(before) => cmul(x, &before),
                                        });
                                    }
                                    u.expect("at least one segment")
                                });
                                psi = cmul(u, &psi);
                                t += duration;
                            }
                            _ => {
                                let r = Self::ideal(electron, angle, phase);
                                let mut out = CMatrix::zeros(d, cols);
                                for e in 0..3 {
                                    for f in 0..3 {
                                        let z = r[(e, f)];
                                        if z == ZERO {
                                            continue;
                                        }
                                        for s in 0..n {
                                            for c in 0..cols {
                                                out[(e * n + s, c)] += z * psi[(f * n + s, c)];
                                            }
                                        }
                                    }
                                }
                                psi = out;
                            }
                        },
                    }
                }
                // ρ_e = Tr_nuclei |ψ⟩⟨ψ| averaged over the columns.
                Rho::from_fn(|e, f| {
                    let mut acc = ZERO;
                    for s in 0..n {
                        for c in 0..cols {
                            acc += psi[(e * n + s, c)] * psi[(f * n + s, c)].conj();
                        }
                    }
                    acc / cols as f64
                })
            })
            .collect())
    }

    /// The eigenvectors `V′` of a cluster's Hamiltonian in the frame where
    /// it is block-diagonal between the levels of `α` and the others: the
    /// direct rotation `T = V V′†` closest to the identity, each block of
    /// `V′` the unitary factor `Z W†` of the polar decomposition of the
    /// block `Z Σ W†` of `V` (eigenvectors grouped by the levels that hold
    /// most of their weight). `None` where the levels mix (an avoided
    /// crossing: a block without as many eigenvectors as levels, or with a
    /// singular value below ½).
    fn dress(vectors: &CMatrix, chi: &[f64; 3], n: usize) -> Option<CMatrix> {
        let d = 3 * n;
        let turning = |r: usize| chi[r / n] > 0.5;
        let rows: [Vec<usize>; 2] = [
            (0..d).filter(|&r| !turning(r)).collect(),
            (0..d).filter(|&r| turning(r)).collect(),
        ];
        let mut columns: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
        for j in 0..d {
            let still: f64 = rows[0].iter().map(|&r| vectors[(r, j)].norm_sqr()).sum();
            columns[usize::from(still <= 0.5)].push(j);
        }
        if columns[0].len() != rows[0].len() {
            return None;
        }
        let mut out = CMatrix::zeros(d, d);
        for (rows, columns) in rows.iter().zip(&columns) {
            let block = CMatrix::from_fn(rows.len(), columns.len(), |a, b| {
                vectors[(rows[a], columns[b])]
            });
            let svd = block.svd(true, true);
            if svd.singular_values.iter().any(|&x| x < 0.5) {
                return None;
            }
            let polar = svd.u? * svd.v_t?;
            for (a, &r) in rows.iter().enumerate() {
                for (b, &c) in columns.iter().enumerate() {
                    out[(r, c)] = polar[(a, b)];
                }
            }
        }
        Some(out)
    }

    /// Operations (estimated): per cluster a diagonalisation, per distinct
    /// free period an exponential, per distinct segment of a finite pulse a
    /// diagonalisation, per step a product with the columns (`2^k`, or one
    /// in a sampled state), all `repeats` times.
    fn check_work(
        sizes: &[usize],
        sequences: &[Sequence],
        pulses: &Pulses,
        repeats: usize,
    ) -> Result<()> {
        let mut free = HashSet::new();
        let mut driven = HashSet::new();
        let mut steps = 0usize;
        for step in sequences.iter().flat_map(|s| &s.steps) {
            steps += 1;
            match *step {
                NvEvent::Free(tau) => {
                    free.insert(tau.to_bits());
                }
                NvEvent::Pulse { angle, phase } => {
                    driven.insert((angle.to_bits(), phase.to_bits()));
                }
            }
        }
        let segments = pulses.drive().map_or(0, |(_, envelope)| envelope.len());
        let (free, driven) = (free.len() as f64, (driven.len() * segments) as f64);
        let columns = |n: f64| if repeats > 1 { 1.0 } else { n };
        let work: f64 = repeats.max(1) as f64
            * sizes
                .iter()
                .map(|&k| {
                    let n = (1u64 << k) as f64;
                    let d = 3.0 * n;
                    d * d * d * (60.0 + 10.0 * free + 80.0 * driven)
                        + steps as f64 * 8.0 * d * d * columns(n)
                })
                .sum::<f64>();
        if work > MAX_GENERALISED_WORK {
            return Err(QsoError::invalid(
                "cce",
                format!(
                    "about {work:.1e} operations (limit {MAX_GENERALISED_WORK:.1e}): lower the \
                     order, the cutoff, the bath states or the number of points"
                ),
            ));
        }
        Ok(())
    }

    /// The electron's density matrix before the last pulse of each
    /// sequence by the expansion `cce`, for each sampled bath state (one
    /// for unpolarised nuclei), and that of the electron alone (eigenbasis,
    /// rotating frame).
    fn expanded(
        &self,
        sequences: &[Sequence],
        cce: &GeneralisedCce,
    ) -> Result<(Vec<Vec<Rho>>, Vec<Rho>)> {
        let clusters = self.clusters(cce.order, cce.cutoff)?;
        if cce.bath_states > MAX_BATH_STATES {
            return Err(QsoError::invalid(
                "bath_states",
                format!("at most {MAX_BATH_STATES} sampled bath states"),
            ));
        }
        let sizes: Vec<usize> = std::iter::once(0)
            .chain(clusters.iter().map(Vec::len))
            .collect();
        Self::check_work(&sizes, sequences, &cce.pulses, cce.bath_states)?;
        // The contributions of the clusters below the order are held for
        // their supersets (in every bath state in flight), and a result per
        // bath state.
        let below = clusters.iter().filter(|c| c.len() < cce.order).count() + 1;
        let in_flight = if cce.bath_states > 0 {
            crate::parallel::threads().min(cce.bath_states)
        } else {
            1
        };
        let held = below
            .saturating_mul(in_flight)
            .saturating_add(cce.bath_states)
            .saturating_mul(sequences.len());
        if held > MAX_GENERALISED_MATRICES {
            return Err(QsoError::invalid(
                "cce",
                format!(
                    "the expansion would hold {held} matrices (limit \
                     {MAX_GENERALISED_MATRICES}): lower the order, the cutoff, the bath states \
                     or the number of sequences"
                ),
            ));
        }
        let electron = self.electron();
        let frequency = self.frame_frequency();
        let run = |members: &[usize], mean: Option<&MeanField>| {
            self.cluster_states(
                members,
                sequences,
                cce.cutoff,
                &cce.pulses,
                &electron,
                frequency,
                mean,
            )
        };
        let bare = run(&[], None)?;
        // r = ρ_C ⊘ ρ_∅ on the qubit's elements, where the coherence and
        // the readout take theirs; those of the third level, and those of
        // the electron alone below 10⁻⁹, keep the electron's values (their
        // ratios are unstable and unused).
        let qubit = |e: usize| e == electron.alpha || e == electron.beta;
        let ratio = |rho: &Rho, base: &Rho| -> Rho {
            Rho::from_fn(|e, f| {
                if qubit(e) && qubit(f) && base[(e, f)].norm() > 1e-9 {
                    rho[(e, f)] / base[(e, f)]
                } else {
                    C64::new(1.0, 0.0)
                }
            })
        };
        // Divide each cluster's ratios by the contributions of its
        // subclusters, multiply them into `total`, and keep them below the
        // order.
        let absorb = |members: &[usize],
                      mut r: Vec<Rho>,
                      total: &mut [Rho],
                      reduced: &mut HashMap<Vec<usize>, Vec<Rho>>| {
            let size = members.len();
            for mask in 1..(1u32 << size) - 1 {
                let sub: Vec<usize> = (0..size)
                    .filter(|&b| mask >> b & 1 == 1)
                    .map(|b| members[b])
                    .collect();
                if let Some(below) = reduced.get(&sub) {
                    for (x, y) in r.iter_mut().zip(below) {
                        x.zip_apply(y, |a, b| *a /= b);
                    }
                }
            }
            for (t, x) in total.iter_mut().zip(&r) {
                t.zip_apply(x, |a, b| *a *= b);
            }
            if size < cce.order {
                reduced.insert(members.to_vec(), r);
            }
        };
        let diverges = |total: &[Rho], state: Option<usize>| -> Result<()> {
            if let Some(k) = total.iter().position(|rho| {
                rho.iter()
                    .any(|z| !(z.norm().is_finite() && z.norm() <= 1.0 + 1e-6))
            }) {
                let state = state.map_or(String::new(), |m| format!(" (bath state {m})"));
                return Err(QsoError::invalid(
                    "cce",
                    format!(
                        "the expansion diverges at sequence {k}{state}: lower the order or the \
                         cutoff{}",
                        if self.dressed {
                            ""
                        } else {
                            ", or take the dressed levels (nuclei coupled strongly to the \
                             electron)"
                        }
                    ),
                ));
            }
            Ok(())
        };
        if cce.bath_states == 0 {
            let mut total = bare.clone();
            let mut reduced: HashMap<Vec<usize>, Vec<Rho>> = HashMap::new();
            for size in 1..=cce.order {
                let list: Vec<&Vec<usize>> = clusters.iter().filter(|c| c.len() == size).collect();
                // Clusters in chunks of about 2²⁰ matrices.
                let chunk = ((1usize << 20) / sequences.len().max(1)).max(1);
                for part in list.chunks(chunk) {
                    let raw = crate::parallel::map_tasks(part.len(), |j| run(part[j], None))
                        .into_iter()
                        .collect::<Result<Vec<_>>>()?;
                    for (&members, states) in part.iter().zip(raw) {
                        let r = states
                            .iter()
                            .zip(&bare)
                            .map(|(rho, base)| ratio(rho, base))
                            .collect();
                        absorb(members, r, &mut total, &mut reduced);
                    }
                }
            }
            diverges(&total, None)?;
            return Ok((vec![total], bare));
        }
        // Sampled bath states, each expanded on its own (in parallel), every
        // cluster in the static field of the nuclei outside it.
        let fields = self.bath_fields(cce.cutoff)?;
        let states = self.bath_states(cce.bath_states, cce.seed);
        let totals = crate::parallel::map_tasks(states.len(), |m| -> Result<Vec<Rho>> {
            let mean = |members: &[usize]| fields.mean_field(&states[m], members);
            let zero = run(&[], Some(&mean(&[])))?;
            let mut total = zero.clone();
            let mut reduced: HashMap<Vec<usize>, Vec<Rho>> = HashMap::new();
            for size in 1..=cce.order {
                for members in clusters.iter().filter(|c| c.len() == size) {
                    let r = run(members, Some(&mean(members)))?
                        .iter()
                        .zip(&zero)
                        .map(|(rho, base)| ratio(rho, base))
                        .collect();
                    absorb(members, r, &mut total, &mut reduced);
                }
            }
            diverges(&total, Some(m))?;
            Ok(total)
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
        Ok((totals, bare))
    }

    /// The bath's projections (±½ along the field, along z without one) in
    /// `count` product states drawn with `seed`.
    fn bath_states(&self, count: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = crate::rng::Rng::seed_from_u64(seed);
        (0..count)
            .map(|_| {
                let mut word = 0u64;
                (0..self.spins.len())
                    .map(|i| {
                        if i % 64 == 0 {
                            word = rng.next_u64();
                        }
                        if word >> (i % 64) & 1 == 0 { 0.5 } else { -0.5 }
                    })
                    .collect()
            })
            .collect()
    }

    /// The axis of the sampled states and the couplings their static fields
    /// act through.
    fn bath_fields(&self, cutoff: f64) -> Result<BathFields> {
        let b = norm(self.field);
        let axis = if b > 0.0 {
            self.field.map(|x| x / b)
        } else {
            [0.0, 0.0, 1.0]
        };
        // The dipolar couplings (Hz) between nuclei closer than the cutoff.
        let dipolar = MU0_OVER_4PI * gamma_c() * gamma_c() * TWO_PI * HBAR;
        let pairs = self.as_bath().clusters(2, cutoff)?;
        let mut neighbours: Vec<Vec<(usize, [[f64; 3]; 3])>> = vec![Vec::new(); self.spins.len()];
        for pair in pairs.iter().filter(|c| c.len() == 2) {
            let (i, j) = (pair[0], pair[1]);
            let (p, q) = (self.spins[i].position, self.spins[j].position);
            let v = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
            let r = norm(v);
            let u = v.map(|x| x / r);
            let scale = dipolar / (r * r * r);
            let tensor: [[f64; 3]; 3] = std::array::from_fn(|a| {
                std::array::from_fn(|c| {
                    scale * (if a == c { 1.0 } else { 0.0 } - 3.0 * u[a] * u[c])
                })
            });
            neighbours[i].push((j, tensor));
            neighbours[j].push((i, tensor));
        }
        Ok(BathFields {
            axis,
            hyperfine: self.spins.iter().map(|s| s.hyperfine).collect(),
            neighbours,
        })
    }

    /// The coherence and the signal after each sequence by the expansion
    /// `cce`, with their standard errors over the sampled bath states (see
    /// [`coherence`](Self::coherence) and [`signal`](Self::signal)).
    pub fn estimate(
        &self,
        sequences: &[Vec<NvEvent>],
        cce: &GeneralisedCce,
    ) -> Result<Vec<BathEstimate>> {
        let parsed = self.parse(sequences, &cce.pulses)?;
        let (totals, bare) = self.expanded(&parsed, cce)?;
        let electron = self.electron();
        let m = totals.len() as f64;
        let error = |sum_of_squares: f64| match cce.bath_states {
            0 => 0.0,
            1 => f64::NAN,
            _ => (sum_of_squares / (m - 1.0) / m).sqrt(),
        };
        Ok(parsed
            .iter()
            .enumerate()
            .map(|(k, seq)| {
                let l: Vec<C64> = totals
                    .iter()
                    .map(|t| Self::branch_coherence(&electron, seq, &t[k], &bare[k]))
                    .collect();
                let s: Vec<f64> = totals
                    .iter()
                    .map(|t| Self::readout(&electron, &t[k], seq.last))
                    .collect();
                let coherence = l.iter().sum::<C64>() / m;
                let signal = s.iter().sum::<f64>() / m;
                BathEstimate {
                    coherence,
                    coherence_error: error(l.iter().map(|x| (x - coherence).norm_sqr()).sum()),
                    signal,
                    signal_error: error(s.iter().map(|x| (x - signal).powi(2)).sum()),
                }
            })
            .collect())
    }

    /// The coherence of the electron before the last pulse of each
    /// sequence, `⟨β′|ρ|α′⟩` between where the branches that started in
    /// `β` and `α` have been carried by the pulses between the first and
    /// the last (the `L` of [`SpinBath::coherence`]), by the expansion
    /// `cce`, relative to the electron alone: the bath's factor (1 without
    /// nuclei). With sampled bath states, their mean.
    pub fn coherence(&self, sequences: &[Vec<NvEvent>], cce: &GeneralisedCce) -> Result<Vec<C64>> {
        Ok(self
            .estimate(sequences, cce)?
            .iter()
            .map(|e| e.coherence)
            .collect())
    }

    /// `⟨β′|ρ|α′⟩/⟨β′|ρ_∅|α′⟩` with `α′`, `β′` the qubit's states carried
    /// by the inner pulses (ideally): where the branches that started in
    /// `α` and `β` are, as [`SpinBath::coherence`] follows them.
    fn branch_coherence(electron: &Electron, seq: &Sequence, rho: &Rho, base: &Rho) -> C64 {
        let mut carry = CMatrix::identity(3, 3);
        for &(angle, phase) in &seq.inner {
            carry = Self::ideal(electron, angle, phase) * carry;
        }
        let alpha = carry.column(electron.alpha).clone_owned();
        let beta = carry.column(electron.beta).clone_owned();
        let element = |m: &Rho| -> C64 {
            let mut z = ZERO;
            for e in 0..3 {
                for f in 0..3 {
                    z += beta[e].conj() * m[(e, f)] * alpha[f];
                }
            }
            z
        };
        element(rho) / element(base)
    }

    /// The population of `α` after each sequence, its last pulse ideal,
    /// by the expansion `cce`. With sampled bath states, their mean.
    pub fn signal(&self, sequences: &[Vec<NvEvent>], cce: &GeneralisedCce) -> Result<Vec<f64>> {
        Ok(self
            .estimate(sequences, cce)?
            .iter()
            .map(|e| e.signal)
            .collect())
    }

    /// The population of `α` after the ideal pulse `last`.
    fn readout(electron: &Electron, rho: &Rho, last: (f64, f64)) -> f64 {
        let r = Self::ideal(electron, last.0, last.1);
        let mut population = 0.0;
        let a = electron.alpha;
        for e in 0..3 {
            for f in 0..3 {
                population += (r[(a, e)] * rho[(e, f)] * r[(a, f)].conj()).re;
            }
        }
        population
    }

    /// The electron and the whole bath (at most
    /// [`MAX_GENERALISED_EXACT`] nuclei) propagated together, with the
    /// couplings of the pairs closer than `cutoff`: the coherence and the
    /// signal of each sequence.
    pub fn exact(
        &self,
        sequences: &[Vec<NvEvent>],
        cutoff: f64,
        pulses: &Pulses,
    ) -> Result<Vec<(C64, f64)>> {
        self.validate()?;
        if cutoff.is_nan() || cutoff < 0.0 {
            return Err(QsoError::invalid(
                "cutoff",
                "a distance ≥ 0 (∞ for every pair)",
            ));
        }
        if self.spins.len() > MAX_GENERALISED_EXACT {
            return Err(QsoError::invalid(
                "spins",
                format!("an exact propagation holds at most {MAX_GENERALISED_EXACT} nuclei"),
            ));
        }
        let parsed = self.parse(sequences, pulses)?;
        Self::check_work(&[0, self.spins.len()], &parsed, pulses, 0)?;
        let electron = self.electron();
        let frequency = self.frame_frequency();
        let members: Vec<usize> = (0..self.spins.len()).collect();
        let zero = self.cluster_states(&[], &parsed, cutoff, pulses, &electron, frequency, None)?;
        let full = self.cluster_states(
            &members, &parsed, cutoff, pulses, &electron, frequency, None,
        )?;
        Ok(full
            .iter()
            .zip(&zero)
            .zip(&parsed)
            .map(|((rho, base), seq)| {
                (
                    Self::branch_coherence(&electron, seq, rho, base),
                    Self::readout(&electron, rho, seq.last),
                )
            })
            .collect())
    }
}

/// The population after each sequence averaged over `configurations`
/// random baths of [`diamond_bath`](super::bath::diamond_bath) (seeded as
/// [`ensemble_signal`](super::bath::ensemble_signal)) in the field `field`
/// (T, any direction), by the generalised expansion `cce`, with its
/// standard error over the configurations (NaN for one).
#[allow(clippy::too_many_arguments)]
pub fn generalised_ensemble_signal(
    field: [f64; 3],
    transition: i8,
    abundance: f64,
    radius: f64,
    configurations: usize,
    seed: u64,
    sequences: &[Vec<NvEvent>],
    cce: &GeneralisedCce,
) -> Result<Vec<(f64, f64)>> {
    if configurations == 0 {
        return Err(QsoError::invalid("configurations", "at least one"));
    }
    let root = crate::rng::Rng::seed_from_u64(seed);
    // Every bath first, so that one out of range fails before any work.
    let baths = (0..configurations)
        .map(|k| {
            let bath_seed = root.fork(k as u64).next_u64();
            GeneralisedBath::diamond(field, transition, abundance, radius, bath_seed)
        })
        .collect::<Result<Vec<_>>>()?;
    let runs = baths
        .iter()
        .map(|bath| bath.signal(sequences, cce))
        .collect::<Result<Vec<_>>>()?;
    let n = configurations as f64;
    Ok((0..sequences.len())
        .map(|j| {
            let mean = runs.iter().map(|r| r[j]).sum::<f64>() / n;
            let error = if configurations < 2 {
                f64::NAN
            } else {
                (runs.iter().map(|r| (r[j] - mean).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt()
            };
            (mean, error)
        })
        .collect())
}

#[cfg(test)]
mod tests;
