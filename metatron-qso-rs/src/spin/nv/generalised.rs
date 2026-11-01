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
//! `D S_z² + γ_e B·S` closest to `m_s = 0` and `m_s = m`; fields that leave
//! `β` less than [`MIN_QUBIT_WEIGHT`] of `|m⟩` (across the axis, where they
//! mix `m_s = ±1`) are refused. Their phases follow a convention, not the
//! eigensolver: `α` real and positive on `|0⟩`, `⟨β|S_x|α⟩` real and
//! positive. The electron starts in `α`, the nuclei unpolarised (or in
//! sampled product states, below).
//!
//! Pulses act in the frame rotating at the microwave frequency `f`, which
//! turns the electron's levels other than `α` (the frame of
//! [`NvCenter`](super::NvCenter), `exp(−2πi f t S_z²)` for a field along
//! the axis); `f` is resonant with the bare qubit unless set. Ideal pulses
//! rotate the qubit in that frame; finite pulses of Rabi frequency `Ω`,
//! rectangular or shaped, drive both transitions from `α` in the
//! rotating-wave approximation, and leave out, while they last, the terms
//! that the frame makes oscillate at `f` (a transverse field, the
//! non-secular hyperfine coupling); their peak Rabi frequency must stay
//! below [`RWA_LIMIT`] of `|f|`. A pulse of phase `φ` is a carrier
//! `cos(2πft + φ)` along x at the frame's `t = 0`, scaled to turn the qubit
//! at its Rabi frequency. Free periods are exact. [`Pulses::Laboratory`]
//! applies ideal pulses about axes fixed in the laboratory frame instead,
//! as PyCCE does.
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
//! electron). Where the nuclei mix `α` with another level strongly (near
//! the crossing at 0.1024 T for `m_s = −1`, with a strongly coupled
//! nucleus) the dressed levels do not exist and are refused; with weak
//! couplings near the crossing they exist but differ from the bare ones by
//! the mixing, a few per cent.
//!
//! The electron's density matrix just before the last pulse, `ρ_C` for
//! each cluster `C` (and `ρ_∅` for the electron alone), expands element by
//! element in the basis the inner pulses carry the qubit to, `ρ̃ = C† ρ C`
//! (`C` their ideal product), where the electron alone keeps the state its
//! first pulse made wherever the inner pulses leave it:
//!
//! ```text
//! ρ̃ = ρ̃_∅ ⊙ Π_C r̃_C,   r_C = ρ̃_C ⊘ ρ̃_∅,   r̃_C = r_C ⊘ Π_{∅≠C′⊊C} r̃_{C′},
//! ```
//!
//! for the elements of the qubit that are not small in the electron alone,
//! and by sums of differences for the others (the third level's, and the
//! qubit's after a first pulse of 0° or 180°). It is truncated at clusters
//! of `order` nuclei and exact when `order` reaches the number of nuclei
//! and the cutoff joins them all: unlike the expansion of
//! [`bath`](super::bath), the electron couples separate groups of nuclei,
//! so a cluster's subsets that are not connected do not factorise. With
//! inner π pulses the expansion converges quickly; other inner pulses mix
//! the branches' populations and coherences, and it converges more slowly.
//! The last pulse, ideal, reads out the population of `α`; the coherence
//! is undefined (NaN) where the electron alone holds none between the
//! branches.
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

/// Least weight of the qubit's level `β` on `|m_s = m⟩`: below it the field
/// mixes `m_s = ±1` and the levels are not labelled by `m`.
pub const MIN_QUBIT_WEIGHT: f64 = 0.75;

/// Largest peak Rabi frequency of finite pulses, as a fraction of the
/// frame's frequency: the rotating-wave approximation fails beyond.
pub const RWA_LIMIT: f64 = 0.1;

/// Elements of the electron's density matrix (in the carried basis) smaller
/// than this in the electron alone are expanded by sums, not products.
const SMALL_ELEMENT: f64 = 1e-6;

/// Most bytes the propagators of finite pulses, kept for reuse, may take in
/// one expansion (one matrix of the cluster's dimension per distinct pulse,
/// for every cluster in flight).
pub const MAX_GENERALISED_CACHE: f64 = (1u64 << 30) as f64;

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

/// The static fields of all nuclei in one bath state: on the electron, and
/// on each nucleus from its neighbours within the cutoff (Hz).
struct FieldSums {
    electron: [f64; 3],
    nuclei: Vec<[f64; 3]>,
}

impl BathFields {
    /// `A·v` for a 3 × 3 tensor and a vector.
    fn apply(a: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
        a.map(|row| row[0] * v[0] + row[1] * v[1] + row[2] * v[2])
    }

    /// The fields of all nuclei in the bath state with the projections `m`
    /// (±½ along the axis), once per state.
    fn totals(&self, m: &[f64]) -> FieldSums {
        let spin = |j: usize| self.axis.map(|x| x * m[j]);
        let mut electron = [0.0; 3];
        for (j, a) in self.hyperfine.iter().enumerate() {
            let f = Self::apply(a, spin(j));
            for c in 0..3 {
                electron[c] += f[c];
            }
        }
        let nuclei = self
            .neighbours
            .iter()
            .map(|list| {
                let mut g = [0.0; 3];
                for &(j, ref d) in list {
                    let f = Self::apply(d, spin(j));
                    for c in 0..3 {
                        g[c] += f[c];
                    }
                }
                g
            })
            .collect();
        FieldSums { electron, nuclei }
    }

    /// The mean field on the cluster `members` in the bath state with the
    /// projections `m`: the fields of all nuclei less those of the members.
    fn mean_field(&self, m: &[f64], sums: &FieldSums, members: &[usize]) -> MeanField {
        let spin = |j: usize| self.axis.map(|x| x * m[j]);
        let mut electron = sums.electron;
        for &j in members {
            let f = Self::apply(&self.hyperfine[j], spin(j));
            for c in 0..3 {
                electron[c] -= f[c];
            }
        }
        let nuclei = members
            .iter()
            .map(|&i| {
                let mut g = sums.nuclei[i];
                for &(j, ref d) in &self.neighbours[i] {
                    if members.contains(&j) {
                        let f = Self::apply(d, spin(j));
                        for c in 0..3 {
                            g[c] -= f[c];
                        }
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
    /// `⟨β|S_x|α⟩`: how the drive couples the qubit (real and positive by
    /// the phase convention, where it does not vanish).
    coupling: C64,
    /// The weights of `α` on `|0⟩` and of `β` on `|m⟩`.
    weights: (f64, f64),
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

    /// The bath without the nuclei closer than `radius` (m) to the vacancy
    /// (a radius that is not a number removes none).
    pub fn without_spins_within(mut self, radius: f64) -> Self {
        self.spins
            .retain(|s| radius.is_nan() || norm(s.position) >= radius);
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
        self.as_bath().validate()?;
        // The qubit {0, m}: its levels must be recognisable.
        let electron = self.electron();
        if electron.values.iter().any(|x| !x.is_finite())
            || electron.vectors.iter().any(|z| !z.norm().is_finite())
        {
            return Err(QsoError::invalid(
                "field",
                "the electron's levels are not finite: the field is too large",
            ));
        }
        let (alpha, beta) = electron.weights;
        if alpha < 0.5 {
            return Err(QsoError::invalid(
                "field",
                format!(
                    "the field mixes m_s = 0 with m_s = ±1 (weight {alpha:.2} on |0⟩): the \
                     qubit is not defined"
                ),
            ));
        }
        if beta < MIN_QUBIT_WEIGHT {
            return Err(QsoError::invalid(
                "field",
                format!(
                    "the field mixes m_s = +1 and −1 (weight {beta:.2} on |m⟩, below \
                     {MIN_QUBIT_WEIGHT}): no level is that of m_s = m; tilt it towards the axis"
                ),
            ));
        }
        Ok(())
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
        let (values, mut vectors) = eigh(&h);
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
        let level = if self.transition < 0 { 2 } else { 0 };
        let alpha = closest(1, &[]);
        let beta = closest(level, &[alpha]);
        let weights = (
            vectors[(1, alpha)].norm_sqr(),
            vectors[(level, beta)].norm_sqr(),
        );
        // The eigenvectors' phases, which the solver leaves arbitrary: α real
        // and positive on |0⟩, ⟨β|S_x|α⟩ real and positive (or β real and
        // positive on |m⟩ where the drive does not couple them), the third
        // level real and positive on its largest component.
        let gauge = |vectors: &mut CMatrix, j: usize, z: C64| {
            if z.norm() > 0.0 {
                let unit = z.conj() / z.norm();
                for r in 0..3 {
                    vectors[(r, j)] *= unit;
                }
            }
        };
        let z = vectors[(1, alpha)];
        gauge(&mut vectors, alpha, z);
        let x = (vectors.column(beta).adjoint() * &sx * vectors.column(alpha))[(0, 0)];
        let z = if x.norm() > 1e-6 {
            x.conj()
        } else {
            vectors[(level, beta)]
        };
        gauge(&mut vectors, beta, z);
        let third = 3 - alpha - beta;
        let largest = (0..3)
            .max_by(|&a, &b| {
                vectors[(a, third)]
                    .norm_sqr()
                    .total_cmp(&vectors[(b, third)].norm_sqr())
            })
            .expect("three components");
        let z = vectors[(largest, third)];
        gauge(&mut vectors, third, z);
        let coupling = (vectors.adjoint() * sx * &vectors)[(beta, alpha)];
        Electron {
            vectors,
            values: [values[0], values[1], values[2]],
            alpha,
            beta,
            coupling,
            weights,
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
        if let Some((rabi, envelope)) = pulses.drive() {
            if self.electron().coupling.norm() < 0.05 {
                return Err(QsoError::invalid(
                    "pulses",
                    "a drive along x hardly couples the qubit's levels in this field",
                ));
            }
            // The rotating-wave approximation needs the drive far slower than
            // the frame.
            let peak = rabi * envelope.iter().map(|s| s.0).fold(0.0, f64::max);
            let frame = self.frame_frequency().abs();
            if peak > RWA_LIMIT * frame {
                return Err(QsoError::invalid(
                    "rabi",
                    format!(
                        "a peak Rabi frequency of {:.3e} Hz exceeds {RWA_LIMIT} of the frame's \
                         frequency ({:.3e} Hz): the rotating-wave approximation fails",
                        peak, frame
                    ),
                ));
            }
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
        // Free periods apply V e^{−iEτ} V† to the columns; the propagators of
        // finite pulses are kept per duration and phase (bounded by
        // MAX_GENERALISED_CACHE before the expansion starts).
        let inverse = vectors.adjoint();
        let mut pulse_cache: HashMap<(u64, u64), CMatrix> = HashMap::new();
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
                            let free = |psi: &CMatrix| -> CMatrix {
                                let mut y = cmul(&inverse, psi);
                                for (r, e) in values.iter().enumerate() {
                                    let phase = C64::from_polar(1.0, -e * tau);
                                    for c in 0..cols {
                                        y[(r, c)] *= phase;
                                    }
                                }
                                cmul(&vectors, &y)
                            };
                            if lab {
                                psi = free(&psi);
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
                                psi = free(&moved);
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
                                    // The segments in turn.
                                    let length = duration / envelope.len() as f64;
                                    let mut u: Option<CMatrix> = None;
                                    for &(amplitude, offset) in envelope {
                                        let x =
                                            segment(base, length, rabi * amplitude, phase + offset);
                                        u = Some(match u {
                                            None => x,
                                            Some(before) => cmul(&x, &before),
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
    /// segment of a finite pulse a diagonalisation and an exponential, per
    /// free period two products with the columns (`2^k`, or one in a
    /// sampled state) and per pulse one, all `repeats` times.
    fn check_work(
        sizes: &[usize],
        sequences: &[Sequence],
        pulses: &Pulses,
        repeats: usize,
    ) -> Result<()> {
        let mut driven = HashSet::new();
        let (mut free, mut pulsed) = (0usize, 0usize);
        for step in sequences.iter().flat_map(|s| &s.steps) {
            match *step {
                NvEvent::Free(_) => free += 1,
                NvEvent::Pulse { angle, phase } => {
                    pulsed += 1;
                    driven.insert((angle.to_bits(), phase.to_bits()));
                }
            }
        }
        let segments = pulses.drive().map_or(0, |(_, envelope)| envelope.len());
        let driven = (driven.len() * segments) as f64;
        let columns = |n: f64| if repeats > 0 { 1.0 } else { n };
        let work: f64 = repeats.max(1) as f64
            * sizes
                .iter()
                .map(|&k| {
                    let n = (1u64 << k) as f64;
                    let d = 3.0 * n;
                    d * d * d * (60.0 + 80.0 * driven)
                        + (16.0 * free as f64 + 8.0 * pulsed as f64) * d * d * columns(n)
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
    ///
    /// The expansion runs in the basis the inner pulses carry the qubit to,
    /// `ρ̃ = C† ρ C` (`C` their ideal product): there the electron alone keeps
    /// the state its first pulse made, wherever the inner pulses leave it.
    /// Elements of `ρ̃` that are not small in the electron alone expand by
    /// products (`r_C = ρ̃_C ⊘ ρ̃_∅`), the others (the third level's, and the
    /// qubit's after a first pulse of 0° or 180°) by sums of differences:
    /// both are exact at full order.
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
            crate::parallel::threads()
        };
        let held = below
            .saturating_mul(if cce.bath_states > 0 { in_flight } else { 1 })
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
        Self::check_cache(
            sizes.iter().copied().max().unwrap_or(0),
            sequences,
            &cce.pulses,
            in_flight,
        )?;
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
        Self::check_finite(&bare)?;
        // The carried basis of each sequence.
        let carries: Vec<Rho> = sequences
            .iter()
            .map(|seq| Self::carry(&electron, seq))
            .collect();
        let tilde = |states: Vec<Rho>| -> Vec<Rho> {
            states
                .iter()
                .zip(&carries)
                .map(|(rho, c)| c.adjoint() * rho * c)
                .collect()
        };
        // Which elements expand by products: the qubit's that are not small
        // in the electron alone.
        let qubit = |e: usize| e == electron.alpha || e == electron.beta;
        let products = |zero: &[Rho]| -> Vec<Matrix3<bool>> {
            zero.iter()
                .map(|z| {
                    Matrix3::from_fn(|e, f| {
                        qubit(e) && qubit(f) && z[(e, f)].norm() > SMALL_ELEMENT
                    })
                })
                .collect()
        };
        // The cluster's contributions before its subclusters are taken out.
        let contributions = |rho: &[Rho], zero: &[Rho], modes: &[Matrix3<bool>]| -> Vec<Rho> {
            rho.iter()
                .zip(zero)
                .zip(modes)
                .map(|((r, z), m)| {
                    Rho::from_fn(|e, f| {
                        if m[(e, f)] {
                            r[(e, f)] / z[(e, f)]
                        } else {
                            r[(e, f)] - z[(e, f)]
                        }
                    })
                })
                .collect()
        };
        // Take the subclusters' irreducible contributions out of a cluster's,
        // add it to `total`, and keep it below the order.
        let absorb = |members: &[usize],
                      mut r: Vec<Rho>,
                      modes: &[Matrix3<bool>],
                      total: &mut [Rho],
                      reduced: &mut HashMap<Vec<usize>, Vec<Rho>>| {
            let size = members.len();
            for mask in 1..(1u32 << size) - 1 {
                let sub: Vec<usize> = (0..size)
                    .filter(|&b| mask >> b & 1 == 1)
                    .map(|b| members[b])
                    .collect();
                if let Some(below) = reduced.get(&sub) {
                    for ((x, y), m) in r.iter_mut().zip(below).zip(modes) {
                        for e in 0..3 {
                            for f in 0..3 {
                                if m[(e, f)] {
                                    x[(e, f)] /= y[(e, f)];
                                } else {
                                    x[(e, f)] -= y[(e, f)];
                                }
                            }
                        }
                    }
                }
            }
            for ((t, x), m) in total.iter_mut().zip(&r).zip(modes) {
                for e in 0..3 {
                    for f in 0..3 {
                        if m[(e, f)] {
                            t[(e, f)] *= x[(e, f)];
                        } else {
                            t[(e, f)] += x[(e, f)];
                        }
                    }
                }
            }
            if size < cce.order {
                reduced.insert(members.to_vec(), r);
            }
        };
        // Back from the carried basis, checked.
        let finish = |total: Vec<Rho>, state: Option<usize>| -> Result<Vec<Rho>> {
            let total: Vec<Rho> = total
                .iter()
                .zip(&carries)
                .map(|(t, c)| c * t * c.adjoint())
                .collect();
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
            Ok(total)
        };
        if cce.bath_states == 0 {
            let zero = tilde(bare.clone());
            let modes = products(&zero);
            let mut total = zero.clone();
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
                        let r = contributions(&tilde(states), &zero, &modes);
                        absorb(members, r, &modes, &mut total, &mut reduced);
                    }
                }
            }
            return Ok((vec![finish(total, None)?], bare));
        }
        // Sampled bath states, each expanded on its own (in parallel), every
        // cluster in the static field of the nuclei outside it.
        let fields = self.bath_fields(cce.cutoff)?;
        let states = self.bath_states(cce.bath_states, cce.seed);
        let totals = crate::parallel::map_tasks(states.len(), |m| -> Result<Vec<Rho>> {
            let sums = fields.totals(&states[m]);
            let mean = |members: &[usize]| fields.mean_field(&states[m], &sums, members);
            let zero = tilde(run(&[], Some(&mean(&[])))?);
            let modes = products(&zero);
            let mut total = zero.clone();
            let mut reduced: HashMap<Vec<usize>, Vec<Rho>> = HashMap::new();
            for size in 1..=cce.order {
                for members in clusters.iter().filter(|c| c.len() == size) {
                    let rho = tilde(run(members, Some(&mean(members)))?);
                    let r = contributions(&rho, &zero, &modes);
                    absorb(members, r, &modes, &mut total, &mut reduced);
                }
            }
            finish(total, Some(m))
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
        Ok((totals, bare))
    }

    /// Refuse an electron alone whose propagation is not finite (fields,
    /// times or Rabi frequencies beyond double precision).
    fn check_finite(states: &[Rho]) -> Result<()> {
        if states
            .iter()
            .any(|rho| rho.iter().any(|z| !(z.re.is_finite() && z.im.is_finite())))
        {
            return Err(QsoError::invalid(
                "sequence",
                "the propagation is not finite: the field, the times or the Rabi frequency are \
                 beyond double precision",
            ));
        }
        Ok(())
    }

    /// Refuse caches of finite pulses beyond [`MAX_GENERALISED_CACHE`]: a
    /// matrix of the largest cluster's dimension per distinct pulse, for
    /// every cluster in flight.
    fn check_cache(
        largest: usize,
        sequences: &[Sequence],
        pulses: &Pulses,
        in_flight: usize,
    ) -> Result<()> {
        if pulses.drive().is_none() {
            return Ok(());
        }
        let distinct: HashSet<(u64, u64)> = sequences
            .iter()
            .flat_map(|s| &s.steps)
            .filter_map(|step| match *step {
                NvEvent::Pulse { angle, phase } => Some((angle.to_bits(), phase.to_bits())),
                NvEvent::Free(_) => None,
            })
            .collect();
        let d = (3u64 << largest) as f64;
        let bytes = distinct.len() as f64 * 16.0 * d * d * in_flight as f64;
        if bytes > MAX_GENERALISED_CACHE {
            return Err(QsoError::invalid(
                "pulses",
                format!(
                    "{} distinct finite pulses would keep {:.1} GB of propagators (limit {:.1} \
                     GB): use fewer pulse angles and phases, or a lower order",
                    distinct.len(),
                    bytes / 1e9,
                    MAX_GENERALISED_CACHE / 1e9
                ),
            ));
        }
        Ok(())
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
        let carry = Self::carry(electron, seq);
        let element =
            |m: &Rho| -> C64 { (carry.adjoint() * m * carry)[(electron.beta, electron.alpha)] };
        let reference = element(base);
        // Undefined where the electron alone holds no coherence between the
        // branches (a first pulse of 0° or 180°).
        if reference.norm() < 1e-9 {
            return C64::new(f64::NAN, f64::NAN);
        }
        element(rho) / reference
    }

    /// The inner pulses of a sequence, ideal: they carry `α` and `β` to the
    /// branches `α′`, `β′` (its columns).
    fn carry(electron: &Electron, seq: &Sequence) -> Rho {
        let mut carry = CMatrix::identity(3, 3);
        for &(angle, phase) in &seq.inner {
            carry = Self::ideal(electron, angle, phase) * carry;
        }
        Rho::from_fn(|r, c| carry[(r, c)])
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
        Self::check_cache(self.spins.len(), &parsed, pulses, 1)?;
        let electron = self.electron();
        let frequency = self.frame_frequency();
        let members: Vec<usize> = (0..self.spins.len()).collect();
        let zero = self.cluster_states(&[], &parsed, cutoff, pulses, &electron, frequency, None)?;
        let full = self.cluster_states(
            &members, &parsed, cutoff, pulses, &electron, frequency, None,
        )?;
        Self::check_finite(&zero)?;
        Self::check_finite(&full)?;
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
    let bath = |k: usize| {
        GeneralisedBath::diamond(
            field,
            transition,
            abundance,
            radius,
            root.fork(k as u64).next_u64(),
        )
    };
    // The settings and the sequences are checked on the first bath before any
    // work; the baths are built one at a time.
    let first = bath(0)?;
    first.parse(sequences, &cce.pulses)?;
    first.clusters(cce.order, cce.cutoff)?;
    // Each configuration samples its own bath states.
    let states = crate::rng::Rng::seed_from_u64(cce.seed);
    let runs = (0..configurations)
        .map(|k| {
            let own = GeneralisedCce {
                seed: states.fork(u64::MAX - k as u64).next_u64(),
                ..cce.clone()
            };
            if k == 0 {
                first.signal(sequences, &own)
            } else {
                bath(k)?.signal(sequences, &own)
            }
        })
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
