//! Nuclear spin dynamics and magnetic resonance: spin systems of a
//! molecule, pulse sequences, relaxation, spectra and polarisation
//! transfer, for designing experiments such as hyperpolarisation by
//! parahydrogen.
//!
//! A [`SpinSystem`] lists nuclei by [`Isotope`] (spin ½, or spin 1 for
//! ²H), their chemical shifts in ppm, scalar couplings `J` in Hz, the
//! field in tesla and the temperature. Its Hamiltonian (angular
//! frequency, rad/s) is, in the laboratory frame,
//!
//! ```text
//! H = Σ_i −γ_i B (1 + δ_i·10⁻⁶) I_zi + Σ_{i<j} 2π J_ij I_i·I_j
//! ```
//!
//! and in the rotating frame of every isotope (at its carrier, in ppm)
//! the offsets `Ω_i = −γ_i B (δ_i − c_X)·10⁻⁶`, full couplings between
//! spins of one isotope and secular ones, `2π J I_zi I_zj`, between
//! isotopes. The rotating frame is exact for one isotope at any field;
//! between isotopes it needs `|γ_i − γ_j| B ≫ 2π|J|`, so low-field
//! experiments with several isotopes (SABRE-SHEATH) use the laboratory
//! frame.
//!
//! The state is a density matrix over the product basis, spin 0 the most
//! significant factor and, within a spin, `m = I, I − 1, …, −I` (for spin
//! ½: `|α⟩ = |0⟩`, `|β⟩ = |1⟩`). Sequences of [`Event`]s act on it:
//! ideal pulses `exp(−iθ(cos φ F_x + sin φ F_y))` on an isotope or a spin,
//! shaped pulses of piecewise-constant amplitude and phase (rotating
//! frame, rotating-wave approximation), delays, and changes and sweeps of
//! the field.
//!
//! `H` conserves `2M` of every isotope in the rotating frame and of all
//! spins in the laboratory frame, so free evolution, spectra and time
//! averages use the eigendecomposition of each block of `H` (exact). With
//! relaxation the Lindblad equation conserves the matching coherence
//! orders; each block of its superoperator is propagated for the deviation
//! `ρ − 1/d`, so that thermal polarisations of parts in 10⁵ keep their
//! relative precision: exponentiated exactly (up to
//! [`MAX_LIOUVILLE_BLOCK`] elements) or, as a sparse matrix, by Krylov
//! steps to 10⁻¹² of the deviation, whichever costs less. Short intervals
//! and shaped pulses use adaptive Dormand–Prince steps instead, with a
//! local error of 10⁻¹¹ of that deviation, when they are cheaper.
//! Permutations of equivalent spins ([`SpinSystem::set_symmetries`])
//! reduce the blocks of states they leave unchanged, exactly. A
//! propagation that would take more than [`MAX_RELAXATION_WORK`]
//! operations or [`MAX_RELAXATION_MEMORY`] bytes is refused before
//! anything is built.
//!
//! Relaxation is phenomenological, per spin: `T₁` towards thermal
//! equilibrium at the current field and temperature (jump operators
//! `√k↓ I₋` and `√k↑ I₊` with `k↑ + k↓ = 1/T₁` and
//! `k↑/k↓ = exp(ħγB(1 + δ·10⁻⁶)/k_BT)`), and pure dephasing `√(2γ_φ) I_z`
//! with `γ_φ = 1/T₂ − 1/(2T₁)`, so `T₂ ≤ 2T₁`. For spin ½ this gives
//! exactly the decay times `T₁` and `T₂` of `⟨I_z⟩` and `⟨I₊⟩`; for spin 1
//! the same operators give them in the high-temperature limit, and the
//! equilibrium is thermal for both.
//!
//! Relaxation from first principles: with positions (Å) and shielding
//! tensors (ppm) of the spins and a rotational correlation time
//! ([`SpinSystem::with_correlation_time`]), the dipolar couplings and
//! shielding anisotropies relax the system by Bloch–Wangsness–Redfield
//! theory for isotropic tumbling, cross-correlations included, towards
//! thermal equilibrium (see the `redfield` source for the formulas).
//! [`SpinSystem::relaxation_rates`] gives auto- and cross-relaxation rates
//! of observables. Quadrupolar relaxation and internal motion are not
//! modelled.
//!
//! The signal is `s(t) = Tr(ρ(t) F₊)`, `F₊ = Σ_{i ∈ X} I₊ᵢ`: a spin with
//! offset Ω gives `e^{iΩt}`. [`SpinSystem::lines`] gives the exact stick
//! spectrum of a state, [`SpinSystem::fid`] the free induction decay and
//! [`Spectrum::from_fid`] its Fourier transform.
//!
//! Chemical exchange between spin systems is modelled by [`Exchange`];
//! optimal control (GRAPE) of rf pulses and field profiles, with exact
//! gradients, by [`Control`] and [`SpinSystem::optimize_control`], and
//! [`unitary_bound`] gives the largest transfer any pulse sequence can
//! reach. [`NvCenter`] is the electron spin of an NV centre in diamond
//! with its nitrogen and ¹³C nuclei, for sensing sequences (Ramsey,
//! echoes, CPMG, XY-8) under classical [`FieldNoise`].
//!
//! ```
//! use metatron_qso::spin::{Event, Isotope, SpinSystem, Target};
//!
//! // ¹³C–¹H at 9.4 T: INEPT moves the proton polarisation onto carbon.
//! let mut sys = SpinSystem::new(9.4);
//! let h = sys.add_spin(Isotope::H1, 4.0).unwrap();
//! let c = sys.add_spin(Isotope::C13, 60.0).unwrap();
//! sys.couple(h, c, 140.0).unwrap();
//! let rho = sys.thermal_state().unwrap();
//! let direct = sys.evolve(&rho, &[Event::pulse(Target::Isotope(Isotope::C13), 90.0, 0.0)]).unwrap();
//! let inept = metatron_qso::spin::inept(Isotope::H1, Isotope::C13, 140.0).unwrap();
//! let moved = sys.evolve(&rho, &inept).unwrap();
//! let peak = |r| {
//!     sys.lines(r, Isotope::C13).unwrap().iter().map(|l| l.amplitude.norm()).fold(0.0, f64::max)
//! };
//! assert!(peak(&moved) / peak(&direct) > 4.9); // γ_H/γ_C + 1 on the stronger line
//! ```

use std::collections::BTreeMap;

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::linalg::{CMatrix, expm_hermitian};

mod blocks;
mod control;
mod exchange;
pub(crate) mod krylov;
#[cfg(test)]
mod large_tests;
mod nv;
mod redfield;
mod symmetry;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod transfer_tests;

pub use control::{Control, Drive, Member, Optimized, unitary_bound};
pub use exchange::{Channel, Exchange, Fresh, Profile, Site};
pub use nv::{
    Carbon, FieldNoise, MAX_NV_DIMENSION, Microwave, NV_GYROMAGNETIC_RATIO,
    NV_ZERO_FIELD_SPLITTING, Nitrogen, NitrogenIsotope, NvCenter, NvEvent, cpmg, decay_exponent,
    filter_function, hahn, ramsey, xy8,
};
pub use redfield::{MU0_OVER_4PI, axial_csa};

/// The nuclear-spin bath of an NV centre (cluster-correlation expansion).
pub use nv::bath;
pub use nv::bath::{BathSpin, Cce, SpinBath, diamond_bath};
/// The optical cycle of an NV centre and magnetometer sensitivity.
pub use nv::optics;
pub use nv::optics::{OpticalNv, Photophysics, TwoLevelOdmr};

/// Reduced Planck constant (J s, CODATA 2018, exact).
pub const HBAR: f64 = 1.054_571_817e-34;
/// Boltzmann constant (J/K, exact).
pub const BOLTZMANN: f64 = 1.380_649e-23;
/// Largest dimension of a spin system's Hilbert space (11 spins ½).
pub const MAX_SPIN_DIMENSION: usize = 2048;
/// Largest block of the Lindblad superoperator (elements of one coherence
/// order) that is exponentiated densely; larger ones take Krylov steps.
/// Six spins ½ of one isotope have blocks of up to 924 elements, seven
/// of 3432.
pub const MAX_LIOUVILLE_BLOCK: usize = 2048;
/// Most floating-point operations (estimated) of one propagation with
/// relaxation, about a minute on one core; more is refused. (One second
/// of six relaxing protons at 9.4 T takes about 3·10¹¹, ten seconds.)
pub const MAX_RELAXATION_WORK: f64 = 1.5e12;
/// Most memory (bytes, estimated) one propagation with relaxation may
/// hold: its largest block, or the Redfield superoperator's terms.
pub const MAX_RELAXATION_MEMORY: f64 = 2.0 * (1u64 << 30) as f64;
/// Most points of a free induction decay.
pub const MAX_FID_POINTS: usize = 1 << 20;
/// Most segments of a shaped pulse or a field sweep.
pub const MAX_SEGMENTS: usize = 1 << 20;

const ZERO: C64 = C64::new(0.0, 0.0);
const ONE: C64 = C64::new(1.0, 0.0);
const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// A magnetic nucleus.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Isotope {
    #[serde(rename = "1H")]
    H1,
    #[serde(rename = "2H")]
    H2,
    #[serde(rename = "13C")]
    C13,
    #[serde(rename = "15N")]
    N15,
    #[serde(rename = "19F")]
    F19,
    #[serde(rename = "31P")]
    P31,
}

impl Isotope {
    /// Every isotope.
    pub const ALL: [Isotope; 6] = [
        Isotope::H1,
        Isotope::H2,
        Isotope::C13,
        Isotope::N15,
        Isotope::F19,
        Isotope::P31,
    ];

    /// Gyromagnetic ratio γ in rad s⁻¹ T⁻¹ (IUPAC, Harris et al., Pure
    /// Appl. Chem. 73, 1795 (2001)).
    pub fn gamma(self) -> f64 {
        1e7 * match self {
            Isotope::H1 => 26.752_212_8,
            Isotope::H2 => 4.106_627_91,
            Isotope::C13 => 6.728_284,
            Isotope::N15 => -2.712_618_04,
            Isotope::F19 => 25.181_48,
            Isotope::P31 => 10.839_4,
        }
    }

    /// Spin quantum number: ½, or 1 for ²H.
    pub fn spin(self) -> f64 {
        if self == Isotope::H2 { 1.0 } else { 0.5 }
    }

    /// Number of levels, `2I + 1`.
    pub fn levels(self) -> usize {
        if self == Isotope::H2 { 3 } else { 2 }
    }

    /// `"1H"`, `"13C"`, …
    pub fn name(self) -> &'static str {
        match self {
            Isotope::H1 => "1H",
            Isotope::H2 => "2H",
            Isotope::C13 => "13C",
            Isotope::N15 => "15N",
            Isotope::F19 => "19F",
            Isotope::P31 => "31P",
        }
    }

    /// From `"1H"`, `"H1"`, `"H"`, `"13C"`, `"C13"`, `"D"`, … (case
    /// ignored).
    pub fn from_name(name: &str) -> Result<Self> {
        let n = name.trim().to_ascii_uppercase();
        Ok(match n.as_str() {
            "1H" | "H1" | "H" => Isotope::H1,
            "2H" | "H2" | "D" => Isotope::H2,
            "13C" | "C13" | "C" => Isotope::C13,
            "15N" | "N15" | "N" => Isotope::N15,
            "19F" | "F19" | "F" => Isotope::F19,
            "31P" | "P31" | "P" => Isotope::P31,
            _ => {
                return Err(QsoError::invalid(
                    "isotope",
                    format!("unknown isotope '{name}' (1H, 2H, 13C, 15N, 19F or 31P)"),
                ));
            }
        })
    }

    /// Larmor frequency `γB/2π` at field `b` (Hz; negative for ¹⁵N).
    pub fn larmor(self, b: f64) -> f64 {
        self.gamma() * b / (2.0 * std::f64::consts::PI)
    }
}

/// One nucleus of a spin system.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Spin {
    pub isotope: Isotope,
    /// Chemical shift δ in ppm.
    pub shift: f64,
    /// Longitudinal relaxation time `T₁` in s (`None`: none).
    #[serde(default)]
    pub t1: Option<f64>,
    /// Transverse relaxation time `T₂` in s (`None`: `2T₁`, no pure
    /// dephasing).
    #[serde(default)]
    pub t2: Option<f64>,
    /// Position in the molecule (Å), for dipolar relaxation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<[f64; 3]>,
    /// Chemical-shielding tensor σ (ppm) in the molecule's frame, for
    /// relaxation by its anisotropy: only the traceless symmetric part
    /// counts, the isotropic part is ignored (the shift sets the
    /// frequency). A chemical-shift tensor is `−σ`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub csa: Option<[[f64; 3]; 3]>,
}

/// Laboratory or rotating frame (see the module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Frame {
    #[default]
    Rotating,
    Laboratory,
}

/// What a pulse acts on.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Target {
    /// Every spin of an isotope (a non-selective pulse on its channel).
    Isotope(Isotope),
    /// One spin (an ideal selective pulse).
    Spin(usize),
}

/// A segment of a shaped pulse: duration (s), amplitude as nutation
/// frequency `ν₁` (Hz) and phase (degrees).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Segment {
    pub duration: f64,
    pub amplitude: f64,
    pub phase: f64,
}

/// A step of a pulse sequence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
#[non_exhaustive]
pub enum Event {
    /// An ideal (instantaneous) pulse of `angle` degrees about the axis
    /// at `phase` degrees from x in the xy plane.
    Pulse {
        target: Target,
        angle: f64,
        phase: f64,
    },
    /// A shaped pulse: free evolution plus `2πν₁(cos φ F_x + sin φ F_y)`
    /// on the target during each segment (rotating frame only, in the
    /// rotating-wave approximation: the field acts on its isotope alone).
    Shaped {
        target: Target,
        segments: Vec<Segment>,
    },
    /// Radiofrequency fields on several channels at once, constant over
    /// steps of `step` s: during step `k` the free evolution plus
    /// `Σ_c 2π(x F_x + y F_y)` on target `c` with `[x, y] =
    /// amplitudes[k][c]` (Hz; rotating frame only, as [`Event::Shaped`]).
    /// The pulses of [`SpinSystem::optimize_control`].
    Control {
        targets: Vec<Target>,
        step: f64,
        amplitudes: Vec<Vec<[f64; 2]>>,
    },
    /// Free evolution (with relaxation) for a time in s.
    Delay(f64),
    /// Set the field (T) at once.
    Field(f64),
    /// Change the field linearly from its current value to `to` (T) over
    /// `duration` s, in `steps` constant pieces.
    Sweep {
        to: f64,
        duration: f64,
        steps: usize,
    },
}

impl Event {
    /// [`Event::Pulse`] with angle and phase in degrees.
    pub fn pulse(target: Target, angle: f64, phase: f64) -> Self {
        Event::Pulse {
            target,
            angle,
            phase,
        }
    }
}

/// A line of a stick spectrum: frequency in the frame (Hz), chemical
/// shift (ppm) and complex amplitude of `e^{2πi f t}` in `Tr(ρ(t) F₊)`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Line {
    pub frequency: f64,
    pub ppm: f64,
    pub amplitude: C64,
}

/// A spectrum: frequencies (Hz, ascending) and complex values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Spectrum {
    pub frequencies: Vec<f64>,
    pub values: Vec<C64>,
}

/// A molecule's nuclear spins, their couplings, the field, the
/// temperature and the frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpinSystem {
    spins: Vec<Spin>,
    /// `(i, j, J)` with `i < j`, J in Hz.
    couplings: Vec<(usize, usize, f64)>,
    /// Field in T.
    field: f64,
    /// Temperature in K.
    #[serde(default = "room_temperature")]
    temperature: f64,
    #[serde(default)]
    frame: Frame,
    /// Carrier of each isotope in ppm (rotating frame).
    #[serde(default)]
    carriers: BTreeMap<Isotope, f64>,
    /// Rotational correlation time τ_c (s) of isotropic tumbling, which
    /// turns on Redfield relaxation by the dipolar couplings and shielding
    /// anisotropies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    correlation_time: Option<f64>,
    /// Permutations of the spins that leave the system unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    symmetries: Vec<Vec<usize>>,
}

fn room_temperature() -> f64 {
    298.15
}

fn check_time(name: &'static str, t: f64) -> Result<()> {
    if !(t.is_finite() && t >= 0.0) {
        return Err(QsoError::invalid(
            name,
            format!("{t} is not a finite, non-negative time"),
        ));
    }
    Ok(())
}

fn check_field(b: f64) -> Result<()> {
    if !(b.is_finite() && b >= 0.0) {
        return Err(QsoError::invalid(
            "field",
            format!("the field must be finite and non-negative, not {b}"),
        ));
    }
    Ok(())
}

/// Relaxation times are finite and positive, and `T₂ ≤ 2T₁`.
fn check_relaxation(t1: Option<f64>, t2: Option<f64>) -> Result<()> {
    for t in [t1, t2].into_iter().flatten() {
        if !(t.is_finite() && t > 0.0) {
            return Err(QsoError::invalid(
                "relaxation",
                format!("relaxation times must be finite and positive, not {t}"),
            ));
        }
    }
    if let (Some(t1), Some(t2)) = (t1, t2)
        && t2 > 2.0 * t1 * (1.0 + 1e-12)
    {
        return Err(QsoError::invalid(
            "relaxation",
            format!("T2 = {t2} exceeds 2·T1 = {}", 2.0 * t1),
        ));
    }
    Ok(())
}

/// Single-spin operators of spin `s` (`levels` = 2s + 1), basis
/// `m = s, …, −s`: `(I_z, I₊, I₋, I_x, I_y)`.
fn spin_ops(levels: usize) -> [CMatrix; 5] {
    let s = (levels as f64 - 1.0) / 2.0;
    let m = |k: usize| s - k as f64;
    let iz = CMatrix::from_fn(levels, levels, |r, c| {
        if r == c { C64::new(m(r), 0.0) } else { ZERO }
    });
    // ⟨m + 1| I₊ |m⟩ = √(s(s+1) − m(m+1)): row k − 1, column k.
    let ip = CMatrix::from_fn(levels, levels, |r, c| {
        if c == r + 1 {
            let mm = m(c);
            C64::new((s * (s + 1.0) - mm * (mm + 1.0)).sqrt(), 0.0)
        } else {
            ZERO
        }
    });
    let im = ip.adjoint();
    let ix = (&ip + &im) * C64::new(0.5, 0.0);
    let iy = (&ip - &im) * C64::new(0.0, -0.5);
    [iz, ip, im, ix, iy]
}

impl SpinSystem {
    /// An empty system at field `b` (T), 298.15 K, in the rotating frame.
    pub fn new(b: f64) -> Self {
        SpinSystem {
            spins: Vec::new(),
            couplings: Vec::new(),
            field: b,
            temperature: room_temperature(),
            frame: Frame::Rotating,
            carriers: BTreeMap::new(),
            correlation_time: None,
            symmetries: Vec::new(),
        }
    }

    /// Add a nucleus with chemical shift `shift` (ppm); returns its index.
    pub fn add_spin(&mut self, isotope: Isotope, shift: f64) -> Result<usize> {
        if !shift.is_finite() {
            return Err(QsoError::invalid("shift", "the shift must be finite"));
        }
        let d = self.dimension() * isotope.levels();
        if d > MAX_SPIN_DIMENSION {
            return Err(QsoError::TooManyQubits {
                what: "spin system (Hilbert-space dimension)",
                max: MAX_SPIN_DIMENSION,
                requested: d,
            });
        }
        self.spins.push(Spin {
            isotope,
            shift,
            t1: None,
            t2: None,
            position: None,
            csa: None,
        });
        // The new spin stays where it is under the symmetries.
        let i = self.spins.len() - 1;
        for p in &mut self.symmetries {
            p.push(i);
        }
        Ok(i)
    }

    /// Set the scalar coupling `J` (Hz) between spins `i` and `j`
    /// (replacing an earlier one; 0 removes it).
    pub fn couple(&mut self, i: usize, j: usize, j_hz: f64) -> Result<&mut Self> {
        self.check_spin(i)?;
        self.check_spin(j)?;
        if i == j || !j_hz.is_finite() {
            return Err(QsoError::invalid(
                "coupling",
                "a coupling joins two different spins and is finite",
            ));
        }
        let (a, b) = (i.min(j), i.max(j));
        self.couplings.retain(|&(x, y, _)| (x, y) != (a, b));
        if j_hz != 0.0 {
            self.couplings.push((a, b, j_hz));
            self.couplings.sort_by_key(|&(x, y, _)| (x, y));
        }
        Ok(self)
    }

    /// Set `T₁` and `T₂` (s) of spin `i`; `None` for none (`T₂` then
    /// `2T₁`). `T₂ ≤ 2T₁`, as the dephasing rate cannot be negative. On an
    /// error the spin keeps its times.
    pub fn set_relaxation(
        &mut self,
        i: usize,
        t1: Option<f64>,
        t2: Option<f64>,
    ) -> Result<&mut Self> {
        self.check_spin(i)?;
        check_relaxation(t1, t2)?;
        self.spins[i].t1 = t1;
        self.spins[i].t2 = t2;
        Ok(self)
    }

    /// The same system at another field (T).
    pub fn with_field(mut self, b: f64) -> Result<Self> {
        check_field(b)?;
        self.field = b;
        Ok(self)
    }

    /// The same system at another temperature (K).
    pub fn with_temperature(mut self, kelvin: f64) -> Result<Self> {
        if !(kelvin.is_finite() && kelvin > 0.0) {
            return Err(QsoError::invalid(
                "temperature",
                "the temperature must be finite and positive",
            ));
        }
        self.temperature = kelvin;
        Ok(self)
    }

    /// The same system in another frame.
    pub fn with_frame(mut self, frame: Frame) -> Self {
        self.frame = frame;
        self
    }

    /// Set the carrier of an isotope (ppm, rotating frame).
    pub fn with_carrier(mut self, isotope: Isotope, ppm: f64) -> Result<Self> {
        if !ppm.is_finite() {
            return Err(QsoError::invalid("carrier", "the carrier must be finite"));
        }
        self.carriers.insert(isotope, ppm);
        Ok(self)
    }

    /// A system from JSON (the fields of this struct), checked.
    pub fn from_json(text: &str) -> Result<Self> {
        let sys: SpinSystem = serde_json::from_str(text)
            .map_err(|e| QsoError::invalid("json", format!("not a spin system: {e}")))?;
        sys.validate()?;
        Ok(sys)
    }

    /// The system as JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("spin systems serialise")
    }

    /// Check a deserialised system.
    pub fn validate(&self) -> Result<()> {
        check_field(self.field)?;
        if !(self.temperature.is_finite() && self.temperature > 0.0) {
            return Err(QsoError::invalid(
                "temperature",
                "the temperature must be finite and positive",
            ));
        }
        if self.spins.is_empty() {
            return Err(QsoError::invalid("spins", "a spin system needs a spin"));
        }
        let d: usize = self
            .spins
            .iter()
            .try_fold(1usize, |d, s| d.checked_mul(s.isotope.levels()))
            .unwrap_or(usize::MAX);
        if d > MAX_SPIN_DIMENSION {
            return Err(QsoError::TooManyQubits {
                what: "spin system (Hilbert-space dimension)",
                max: MAX_SPIN_DIMENSION,
                requested: d,
            });
        }
        for s in &self.spins {
            if !s.shift.is_finite() {
                return Err(QsoError::invalid("shift", "shifts must be finite"));
            }
            check_relaxation(s.t1, s.t2)?;
        }
        let mut seen = std::collections::BTreeSet::new();
        for &(i, j, jh) in &self.couplings {
            if i >= j || j >= self.spins.len() || !jh.is_finite() || !seen.insert((i, j)) {
                return Err(QsoError::invalid(
                    "couplings",
                    "couplings are (i, j, J) with i < j below the number of spins, once each, J finite",
                ));
            }
        }
        if self.carriers.values().any(|c| !c.is_finite()) {
            return Err(QsoError::invalid("carriers", "carriers must be finite"));
        }
        self.validate_motion()?;
        self.validate_symmetries()
    }

    fn check_spin(&self, i: usize) -> Result<()> {
        if i >= self.spins.len() {
            return Err(QsoError::invalid(
                "spin",
                format!("spin {i} out of range ({} spins)", self.spins.len()),
            ));
        }
        Ok(())
    }

    pub fn spins(&self) -> &[Spin] {
        &self.spins
    }

    pub fn couplings(&self) -> &[(usize, usize, f64)] {
        &self.couplings
    }

    pub fn field(&self) -> f64 {
        self.field
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    pub fn frame(&self) -> Frame {
        self.frame
    }

    /// Carrier of an isotope (ppm, 0 by default).
    pub fn carrier(&self, isotope: Isotope) -> f64 {
        self.carriers.get(&isotope).copied().unwrap_or(0.0)
    }

    /// Hilbert-space dimension, the product of `2I + 1`.
    pub fn dimension(&self) -> usize {
        self.spins.iter().map(|s| s.isotope.levels()).product()
    }

    /// The coupling network as a graph, edge weights J in Hz.
    pub fn coupling_graph(&self) -> Result<Graph> {
        Graph::from_weighted_edges(self.spins.len(), &self.couplings)
    }

    /// Stride of spin `i` in the product basis.
    fn stride(&self, i: usize) -> usize {
        self.spins[i + 1..]
            .iter()
            .map(|s| s.isotope.levels())
            .product()
    }

    /// `m` of spin `i` in basis state `index`.
    fn m(&self, i: usize, index: usize) -> f64 {
        let levels = self.spins[i].isotope.levels();
        let k = (index / self.stride(i)) % levels;
        self.spins[i].isotope.spin() - k as f64
    }

    /// A single-spin operator (`levels × levels`) on spin `i`, embedded.
    pub fn embed(&self, i: usize, op: &CMatrix) -> Result<CMatrix> {
        self.check_spin(i)?;
        let k = self.spins[i].isotope.levels();
        if op.nrows() != k || op.ncols() != k {
            return Err(QsoError::DimensionMismatch {
                expected: k,
                actual: op.nrows(),
            });
        }
        let d = self.dimension();
        let lo = self.stride(i);
        let hi = d / (lo * k);
        let mut out = CMatrix::zeros(d, d);
        for h in 0..hi {
            for r in 0..k {
                for c in 0..k {
                    let v = op[(r, c)];
                    if v == ZERO {
                        continue;
                    }
                    for l in 0..lo {
                        out[((h * k + r) * lo + l, (h * k + c) * lo + l)] = v;
                    }
                }
            }
        }
        Ok(out)
    }

    fn local(&self, i: usize, which: usize) -> CMatrix {
        spin_ops(self.spins[i].isotope.levels())[which].clone()
    }

    /// `I_z` of spin `i`.
    pub fn iz(&self, i: usize) -> Result<CMatrix> {
        self.check_spin(i)?;
        self.embed(i, &self.local(i, 0))
    }

    /// `I_x` of spin `i`.
    pub fn ix(&self, i: usize) -> Result<CMatrix> {
        self.check_spin(i)?;
        self.embed(i, &self.local(i, 3))
    }

    /// `I_y` of spin `i`.
    pub fn iy(&self, i: usize) -> Result<CMatrix> {
        self.check_spin(i)?;
        self.embed(i, &self.local(i, 4))
    }

    /// `I₊` of spin `i`.
    pub fn iplus(&self, i: usize) -> Result<CMatrix> {
        self.check_spin(i)?;
        self.embed(i, &self.local(i, 1))
    }

    /// Spins of an isotope (an error if there are none).
    fn of_isotope(&self, isotope: Isotope) -> Result<Vec<usize>> {
        let v: Vec<usize> = (0..self.spins.len())
            .filter(|&i| self.spins[i].isotope == isotope)
            .collect();
        if v.is_empty() {
            return Err(QsoError::invalid(
                "isotope",
                format!("the system has no {} spin", isotope.name()),
            ));
        }
        Ok(v)
    }

    fn targets(&self, target: Target) -> Result<Vec<usize>> {
        match target {
            Target::Isotope(x) => self.of_isotope(x),
            Target::Spin(i) => {
                self.check_spin(i)?;
                Ok(vec![i])
            }
        }
    }

    /// `Σ` of a single-spin operator over the spins of an isotope
    /// (`which`: 0 = z, 1 = +, 3 = x, 4 = y).
    fn total(&self, isotope: Isotope, which: usize) -> Result<CMatrix> {
        let d = self.dimension();
        let mut out = CMatrix::zeros(d, d);
        for i in self.of_isotope(isotope)? {
            out += self.embed(i, &self.local(i, which))?;
        }
        Ok(out)
    }

    /// `F_z = Σ I_z` over the spins of an isotope.
    pub fn fz(&self, isotope: Isotope) -> Result<CMatrix> {
        self.total(isotope, 0)
    }

    /// `F_x`.
    pub fn fx(&self, isotope: Isotope) -> Result<CMatrix> {
        self.total(isotope, 3)
    }

    /// `F_y`.
    pub fn fy(&self, isotope: Isotope) -> Result<CMatrix> {
        self.total(isotope, 4)
    }

    /// `F₊`, the operator of the detected signal.
    pub fn fplus(&self, isotope: Isotope) -> Result<CMatrix> {
        self.total(isotope, 1)
    }

    /// Angular frequency of spin `i`'s Zeeman term at field `b`.
    fn zeeman(&self, i: usize, b: f64) -> f64 {
        let s = &self.spins[i];
        match self.frame {
            Frame::Laboratory => -s.isotope.gamma() * b * (1.0 + s.shift * 1e-6),
            Frame::Rotating => -s.isotope.gamma() * b * (s.shift - self.carrier(s.isotope)) * 1e-6,
        }
    }

    /// The free Hamiltonian (rad/s) at the system's field.
    pub fn hamiltonian(&self) -> Result<CMatrix> {
        self.validate()?;
        Ok(self.hamiltonian_at(self.field))
    }

    fn hamiltonian_at(&self, b: f64) -> CMatrix {
        let d = self.dimension();
        let mut h = CMatrix::zeros(d, d);
        for i in 0..self.spins.len() {
            let w = self.zeeman(i, b);
            for s in 0..d {
                h[(s, s)].re += w * self.m(i, s);
            }
        }
        let two_pi = 2.0 * std::f64::consts::PI;
        for &(i, j, jh) in &self.couplings {
            let full =
                self.frame == Frame::Laboratory || self.spins[i].isotope == self.spins[j].isotope;
            self.add_dot(&mut h, i, j, two_pi * jh, full);
        }
        h
    }

    /// Add `c·I_i·I_j` to `h`, or only `c·I_zi I_zj` (`full` false), built
    /// entry by entry.
    fn add_dot(&self, h: &mut CMatrix, i: usize, j: usize, c: f64, full: bool) {
        let (si, sj) = (self.spins[i].isotope.spin(), self.spins[j].isotope.spin());
        let (li, lj) = (
            self.spins[i].isotope.levels(),
            self.spins[j].isotope.levels(),
        );
        let (stride_i, stride_j) = (self.stride(i), self.stride(j));
        for s in 0..h.nrows() {
            let (mi, mj) = (self.m(i, s), self.m(j, s));
            h[(s, s)].re += c * mi * mj;
            if !full {
                continue;
            }
            // ½(I₊ᵢI₋ⱼ + I₋ᵢI₊ⱼ): |…m_i…m_j…⟩ → |…m_i+1…m_j−1…⟩ and back.
            let ki = (s / stride_i) % li;
            let kj = (s / stride_j) % lj;
            if ki > 0 && kj + 1 < lj {
                // m_i + 1 (index ki − 1), m_j − 1 (index kj + 1).
                let a = (si * (si + 1.0) - mi * (mi + 1.0)).sqrt();
                let b = (sj * (sj + 1.0) - mj * (mj - 1.0)).sqrt();
                let t = s - stride_i + stride_j;
                let v = C64::new(c * 0.5 * a * b, 0.0);
                h[(t, s)] += v;
                h[(s, t)] += v;
            }
        }
    }

    /// Thermal equilibrium at the system's field and temperature: every
    /// spin with Boltzmann populations of its Zeeman levels (couplings
    /// neglected, as they shift the levels by parts in 10⁶).
    pub fn thermal_state(&self) -> Result<CMatrix> {
        self.validate()?;
        Ok(self.thermal_at(self.field))
    }

    fn boltzmann(&self, i: usize, b: f64) -> f64 {
        let s = &self.spins[i];
        HBAR * s.isotope.gamma() * b * (1.0 + s.shift * 1e-6) / (BOLTZMANN * self.temperature)
    }

    /// Boltzmann populations of spin `i`'s levels at field `b`,
    /// `e^{x m}/Σ e^{x m}` with the largest exponent `x·I` subtracted so
    /// that no field or temperature overflows.
    fn populations(&self, i: usize, b: f64) -> Vec<f64> {
        let x = self.boltzmann(i, b);
        let s = self.spins[i].isotope.spin();
        let top = x.abs() * s;
        let w: Vec<f64> = (0..self.spins[i].isotope.levels())
            .map(|k| (x * (s - k as f64) - top).exp())
            .collect();
        let z: f64 = w.iter().sum();
        w.into_iter().map(|v| v / z).collect()
    }

    fn thermal_at(&self, b: f64) -> CMatrix {
        let d = self.dimension();
        let n = self.spins.len();
        let populations: Vec<Vec<f64>> = (0..n).map(|i| self.populations(i, b)).collect();
        CMatrix::from_fn(d, d, |r, c| {
            if r != c {
                return ZERO;
            }
            let p: f64 = (0..n)
                .map(|i| {
                    let k = (r / self.stride(i)) % self.spins[i].isotope.levels();
                    populations[i][k]
                })
                .product();
            C64::new(p, 0.0)
        })
    }

    /// The maximally mixed state `1/d`.
    pub fn identity_state(&self) -> CMatrix {
        let d = self.dimension();
        CMatrix::identity(d, d) * C64::new(1.0 / d as f64, 0.0)
    }

    /// Hydrogen from parahydrogen on spins `i` and `j` (both spin ½):
    /// the pair in the singlet with probability `fraction` and in the
    /// three triplet states equally otherwise (0.25 is normal hydrogen),
    /// every other spin maximally mixed. `1/d + 4(1 − 4f)/(3d) I_i·I_j`.
    pub fn parahydrogen_state(&self, i: usize, j: usize, fraction: f64) -> Result<CMatrix> {
        self.check_spin(i)?;
        self.check_spin(j)?;
        if i == j || self.spins[i].isotope.levels() != 2 || self.spins[j].isotope.levels() != 2 {
            return Err(QsoError::invalid(
                "spins",
                "parahydrogen occupies two different spins ½",
            ));
        }
        if !(0.0..=1.0).contains(&fraction) {
            return Err(QsoError::invalid(
                "fraction",
                "the parahydrogen fraction lies in [0, 1]",
            ));
        }
        let d = self.dimension() as f64;
        let ii = self.dot(i, j)?;
        Ok(self.identity_state() + ii * C64::new(4.0 * (1.0 - 4.0 * fraction) / (3.0 * d), 0.0))
    }

    /// `I_i·I_j`.
    fn dot(&self, i: usize, j: usize) -> Result<CMatrix> {
        let d = self.dimension();
        let mut out = CMatrix::zeros(d, d);
        self.add_dot(&mut out, i, j, 1.0, true);
        Ok(out)
    }

    /// `Tr(ρ O)`.
    pub fn expectation(&self, rho: &CMatrix, op: &CMatrix) -> Result<C64> {
        self.check_state(rho)?;
        if op.shape() != rho.shape() {
            return Err(QsoError::DimensionMismatch {
                expected: rho.nrows(),
                actual: op.nrows(),
            });
        }
        // Tr(ρO) = Σ ρ_ab O_ba.
        let mut t = ZERO;
        for a in 0..rho.nrows() {
            for b in 0..rho.ncols() {
                t += rho[(a, b)] * op[(b, a)];
            }
        }
        Ok(t)
    }

    /// Polarisation `⟨I_z⟩/I` of spin `i` (1 when fully along the field).
    pub fn polarization(&self, rho: &CMatrix, i: usize) -> Result<f64> {
        let v = self.expectation(rho, &self.iz(i)?)?;
        Ok(v.re / self.spins[i].isotope.spin())
    }

    fn check_state(&self, rho: &CMatrix) -> Result<()> {
        let d = self.dimension();
        if rho.nrows() != d || rho.ncols() != d {
            return Err(QsoError::DimensionMismatch {
                expected: d,
                actual: rho.nrows(),
            });
        }
        Ok(())
    }

    /// The state averaged over a long, random time of free evolution:
    /// coherences between eigenstates of `H` with different energies
    /// removed. Energies are compared in the laboratory frame, so in the
    /// rotating frame coherences between different `M` of an isotope go
    /// too (they oscillate at its Larmor frequency); equal means equal to
    /// the rounding of the eigendecomposition (about 2·10⁻¹³ of the
    /// energies' spread). Models a state formed incoherently over a long
    /// time, such as parahydrogen added at high field (PASADENA).
    pub fn time_average(&self, rho: &CMatrix) -> Result<CMatrix> {
        self.validate()?;
        self.check_state(rho)?;
        Ok(self.average(rho))
    }

    /// Whether any spin relaxes.
    fn relaxes(&self) -> bool {
        self.spins.iter().any(|s| s.t1.is_some() || s.t2.is_some()) || self.redfield_active()
    }

    /// Jump operators of spin `i` at field `b`: local `(spin, matrix)`.
    fn jumps(&self, b: f64) -> Vec<(usize, CMatrix)> {
        let mut out = Vec::new();
        for (i, s) in self.spins.iter().enumerate() {
            let [iz, ip, im, _, _] = spin_ops(s.isotope.levels());
            let rate1 = s.t1.map_or(0.0, |t| 1.0 / t);
            if rate1 > 0.0 {
                let x = self.boltzmann(i, b);
                // k↑/k↓ = e^x, k↑ + k↓ = 1/T₁ (written stably).
                let (up, down) = if x >= 0.0 {
                    let e = (-x).exp();
                    (rate1 / (1.0 + e), rate1 * e / (1.0 + e))
                } else {
                    let e = x.exp();
                    (rate1 * e / (1.0 + e), rate1 / (1.0 + e))
                };
                out.push((i, im * C64::new(down.sqrt(), 0.0)));
                out.push((i, ip * C64::new(up.sqrt(), 0.0)));
            }
            let rate2 = match (s.t1, s.t2) {
                (_, Some(t2)) => 1.0 / t2 - rate1 / 2.0,
                _ => 0.0,
            };
            if rate2 > 0.0 {
                out.push((i, iz * C64::new((2.0 * rate2).sqrt(), 0.0)));
            }
        }
        out
    }

    /// Apply a sequence of events to `rho`. The field starts at the
    /// system's and may change; the system keeps its own, so a state left
    /// at another field is observed with
    /// `with_field(sys.field_after(events))`.
    pub fn evolve(&self, rho: &CMatrix, events: &[Event]) -> Result<CMatrix> {
        self.validate()?;
        self.check_state(rho)?;
        let mut b = self.field;
        let mut rho = rho.clone();
        // The blocks of H at the last field of a delay, and the relaxation
        // at the last field.
        let mut cache: Option<(f64, Vec<blocks::Eigenblock>)> = None;
        let mut relaxation: Option<(f64, blocks::Relaxation)> = None;
        for e in events {
            match e {
                Event::Pulse {
                    target,
                    angle,
                    phase,
                } => {
                    if !(angle.is_finite() && phase.is_finite()) {
                        return Err(QsoError::invalid("pulse", "angles must be finite"));
                    }
                    for i in self.targets(*target)? {
                        let r = self.rotation(i, angle.to_radians(), phase.to_radians());
                        rho = self.conjugate_local(&rho, i, &r);
                    }
                }
                Event::Shaped { target, segments } => {
                    if self.frame != Frame::Rotating {
                        return Err(QsoError::invalid(
                            "frame",
                            "shaped pulses need the rotating frame",
                        ));
                    }
                    if segments.len() > MAX_SEGMENTS {
                        return Err(QsoError::invalid("segments", "too many segments"));
                    }
                    let targets = self.targets(*target)?;
                    let h0 = self.hamiltonian_at(b);
                    for seg in segments {
                        check_time("duration", seg.duration)?;
                        if !(seg.amplitude.is_finite() && seg.phase.is_finite()) {
                            return Err(QsoError::invalid(
                                "segment",
                                "amplitudes and phases must be finite",
                            ));
                        }
                        let w = 2.0 * std::f64::consts::PI * seg.amplitude;
                        let (c, s) = (seg.phase.to_radians().cos(), seg.phase.to_radians().sin());
                        let mut h = h0.clone();
                        for &i in &targets {
                            let op = self.local(i, 3) * C64::new(w * c, 0.0)
                                + self.local(i, 4) * C64::new(w * s, 0.0);
                            h += self.embed(i, &op)?;
                        }
                        rho = self.propagate(&rho, &h, seg.duration, b, None, &mut relaxation)?;
                    }
                }
                Event::Control {
                    targets,
                    step,
                    amplitudes,
                } => {
                    if self.frame != Frame::Rotating {
                        return Err(QsoError::invalid(
                            "frame",
                            "rf controls need the rotating frame",
                        ));
                    }
                    check_time("step", *step)?;
                    if amplitudes.len() > MAX_SEGMENTS {
                        return Err(QsoError::invalid("control", "too many steps"));
                    }
                    let channels = targets
                        .iter()
                        .map(|&t| self.channel(t))
                        .collect::<Result<Vec<_>>>()?;
                    let h0 = self.hamiltonian_at(b);
                    for row in amplitudes {
                        if row.len() != targets.len() {
                            return Err(QsoError::invalid(
                                "control",
                                "one [x, y] per channel and step",
                            ));
                        }
                        let mut h = h0.clone();
                        for ([fx, fy, _], &[x, y]) in channels.iter().zip(row) {
                            if !(x.is_finite() && y.is_finite()) {
                                return Err(QsoError::invalid(
                                    "control",
                                    "amplitudes must be finite",
                                ));
                            }
                            h += fx * C64::new(TWO_PI * x, 0.0) + fy * C64::new(TWO_PI * y, 0.0);
                        }
                        rho = self.propagate(&rho, &h, *step, b, None, &mut relaxation)?;
                    }
                }
                Event::Delay(t) => {
                    check_time("delay", *t)?;
                    rho = self.propagate(
                        &rho,
                        &self.hamiltonian_at(b),
                        *t,
                        b,
                        Some(&mut cache),
                        &mut relaxation,
                    )?;
                }
                Event::Field(to) => {
                    check_field(*to)?;
                    b = *to;
                }
                Event::Sweep {
                    to,
                    duration,
                    steps,
                } => {
                    check_field(*to)?;
                    check_time("duration", *duration)?;
                    if *steps == 0 || *steps > MAX_SEGMENTS {
                        return Err(QsoError::invalid(
                            "steps",
                            format!("a sweep takes 1..={MAX_SEGMENTS} steps"),
                        ));
                    }
                    let from = b;
                    let dt = duration / *steps as f64;
                    for k in 0..*steps {
                        // The field at the middle of each piece.
                        let bk = from + (to - from) * (k as f64 + 0.5) / *steps as f64;
                        rho = self.propagate(
                            &rho,
                            &self.hamiltonian_at(bk),
                            dt,
                            bk,
                            Some(&mut None),
                            &mut relaxation,
                        )?;
                    }
                    b = *to;
                }
            }
        }
        Ok(rho)
    }

    /// The field at the end of a sequence of events (T).
    pub fn field_after(&self, events: &[Event]) -> f64 {
        events.iter().fold(self.field, |b, e| match e {
            Event::Field(to) | Event::Sweep { to, .. } => *to,
            _ => b,
        })
    }

    /// `exp(−iθ(cos φ I_x + sin φ I_y))` of spin `i` (local matrix).
    fn rotation(&self, i: usize, theta: f64, phi: f64) -> CMatrix {
        let g = self.local(i, 3) * C64::new(phi.cos(), 0.0)
            + self.local(i, 4) * C64::new(phi.sin(), 0.0);
        expm_hermitian(&g, theta)
    }

    /// `R ρ R†` for a local `R` on spin `i`.
    fn conjugate_local(&self, rho: &CMatrix, i: usize, r: &CMatrix) -> CMatrix {
        let left = self.left_local(rho, i, r);
        self.right_local(&left, i, &r.adjoint())
    }

    /// `(R ⊗ 1) M`: `R` on spin `i` acting on the rows.
    fn left_local(&self, m: &CMatrix, i: usize, r: &CMatrix) -> CMatrix {
        let k = r.nrows();
        let lo = self.stride(i);
        let d = m.nrows();
        let hi = d / (lo * k);
        let mut out = CMatrix::zeros(d, m.ncols());
        for col in 0..m.ncols() {
            for h in 0..hi {
                for l in 0..lo {
                    for a in 0..k {
                        let mut v = ZERO;
                        for b in 0..k {
                            v += r[(a, b)] * m[((h * k + b) * lo + l, col)];
                        }
                        out[((h * k + a) * lo + l, col)] = v;
                    }
                }
            }
        }
        out
    }

    /// `M (R ⊗ 1)`: `R` on spin `i` acting on the columns.
    fn right_local(&self, m: &CMatrix, i: usize, r: &CMatrix) -> CMatrix {
        let k = r.nrows();
        let lo = self.stride(i);
        let d = m.ncols();
        let hi = d / (lo * k);
        let mut out = CMatrix::zeros(m.nrows(), d);
        for h in 0..hi {
            for l in 0..lo {
                for a in 0..k {
                    let ca = (h * k + a) * lo + l;
                    for b in 0..k {
                        let w = r[(b, a)];
                        if w == ZERO {
                            continue;
                        }
                        let cb = (h * k + b) * lo + l;
                        for row in 0..m.nrows() {
                            out[(row, ca)] += m[(row, cb)] * w;
                        }
                    }
                }
            }
        }
        out
    }

    /// Free evolution under `h` for `t` at field `b` (for the thermal
    /// target of relaxation). `blocks` is a cache of the blocks of `h` when
    /// it is the free Hamiltonian, which conserves the labels; `None` for
    /// a shaped pulse. `relaxation` caches the relaxation at a field.
    fn propagate(
        &self,
        rho: &CMatrix,
        h: &CMatrix,
        t: f64,
        b: f64,
        blocks: Option<&mut Option<(f64, Vec<blocks::Eigenblock>)>>,
        relaxation: &mut Option<(f64, blocks::Relaxation)>,
    ) -> Result<CMatrix> {
        if t == 0.0 {
            return Ok(rho.clone());
        }
        if self.relaxes() {
            if !matches!(relaxation, Some((field, _)) if *field == b) {
                *relaxation = Some((b, self.relaxation(b)?));
            }
            let (_, r) = relaxation.as_ref().expect("filled");
            return self.relax_with(rho, h, t, blocks.is_some(), r);
        }
        match blocks {
            Some(cache) => {
                if !matches!(cache, Some((field, _)) if *field == b) {
                    *cache = Some((b, self.eigenblocks(h)));
                }
                let (_, eigen) = cache.as_ref().expect("filled");
                Ok(self.unitary(rho, eigen, t))
            }
            None => {
                let u = expm_hermitian(h, t);
                Ok(&u * rho * u.adjoint())
            }
        }
    }

    /// Exact stick spectrum of `rho` on an isotope under free evolution
    /// (no relaxation): in the eigenbasis of `H`, every pair of levels
    /// gives `ρ_ab (F₊)_ba e^{i(E_b − E_a)t}`. Lines at one frequency (to
    /// the rounding of the energies, at least 10⁻⁶ Hz) are merged; lines
    /// below 10⁻¹³ of the largest entry of `rho` (rounding) or 10⁻¹⁴ of
    /// the strongest are dropped; sorted by frequency.
    pub fn lines(&self, rho: &CMatrix, isotope: Isotope) -> Result<Vec<Line>> {
        self.validate()?;
        self.check_state(rho)?;
        let two_pi = 2.0 * std::f64::consts::PI;
        Ok(self
            .transitions(rho, isotope)?
            .into_iter()
            .map(|(w, a)| Line {
                frequency: w / two_pi,
                ppm: self.ppm(isotope, w / two_pi),
                amplitude: a,
            })
            .collect())
    }

    /// Chemical shift (ppm) of a frequency (Hz) in the system's frame (NaN
    /// at zero field, where shifts have no frequency).
    pub fn ppm(&self, isotope: Isotope, frequency: f64) -> f64 {
        let nu = isotope.larmor(self.field);
        if nu == 0.0 {
            return f64::NAN;
        }
        match self.frame {
            // f = −ν (δ − c)·10⁻⁶.
            Frame::Rotating => self.carrier(isotope) - frequency / nu * 1e6,
            // f = −ν (1 + δ·10⁻⁶).
            Frame::Laboratory => (-frequency / nu - 1.0) * 1e6,
        }
    }

    /// Free induction decay on an isotope: `Tr(ρ(k·dwell) F₊)` for
    /// `points` points, with relaxation if any spin relaxes (else from the
    /// transitions of [`SpinSystem::lines`]).
    pub fn fid(
        &self,
        rho: &CMatrix,
        isotope: Isotope,
        dwell: f64,
        points: usize,
    ) -> Result<Vec<C64>> {
        self.validate()?;
        self.check_state(rho)?;
        if !(dwell.is_finite() && dwell > 0.0) || points == 0 || points > MAX_FID_POINTS {
            return Err(QsoError::invalid(
                "fid",
                format!("a positive dwell time and 1..={MAX_FID_POINTS} points"),
            ));
        }
        if self.relaxes() {
            return self.relaxing_fid(rho, isotope, dwell, points);
        }
        let terms = self.transitions(rho, isotope)?;
        let mut out = vec![ZERO; points];
        for (w, a) in terms {
            // e^{iωk·dwell} by multiplication, set exactly every 64 points.
            let step = C64::from_polar(1.0, w * dwell);
            let mut z = ONE;
            for (k, o) in out.iter_mut().enumerate() {
                if k % 64 == 0 {
                    z = C64::from_polar(1.0, w * dwell * k as f64);
                }
                *o += a * z;
                z *= step;
            }
        }
        Ok(out)
    }

    /// The free induction decay summed over a phase cycle: every entry
    /// `(events, receiver phase in degrees)` is run from `rho`, its FID
    /// multiplied by `e^{−iφ}`, and the sum divided by the number of
    /// entries.
    pub fn phase_cycle(
        &self,
        rho: &CMatrix,
        cycle: &[(Vec<Event>, f64)],
        isotope: Isotope,
        dwell: f64,
        points: usize,
    ) -> Result<Vec<C64>> {
        if cycle.is_empty() {
            return Err(QsoError::invalid("cycle", "a phase cycle needs a step"));
        }
        let mut sum = vec![ZERO; points];
        for (events, receiver) in cycle {
            if !receiver.is_finite() {
                return Err(QsoError::invalid("receiver", "phases must be finite"));
            }
            let w = C64::from_polar(1.0, -receiver.to_radians());
            let after = self.evolve(rho, events)?;
            for (s, v) in sum
                .iter_mut()
                .zip(self.fid(&after, isotope, dwell, points)?)
            {
                *s += v * w;
            }
        }
        let n = C64::new(cycle.len() as f64, 0.0);
        Ok(sum.into_iter().map(|s| s / n).collect())
    }
}

impl Spectrum {
    /// The Fourier transform `S(f) = Σ_k s_k e^{−2πi f t_k}` of a free
    /// induction decay with dwell time `dwell` (s), after exponential line
    /// broadening `exp(−π·lb·t)` (`lb` in Hz) and halving the first point,
    /// zero-filled to twice the next power of two; frequencies from
    /// `−1/(2·dwell)` up.
    pub fn from_fid(fid: &[C64], dwell: f64, broadening: f64) -> Result<Self> {
        let broadening_ok = broadening.is_finite() && broadening >= 0.0;
        if fid.is_empty() || !(dwell.is_finite() && dwell > 0.0) || !broadening_ok {
            return Err(QsoError::invalid(
                "fid",
                "a non-empty FID, a positive dwell time and a non-negative broadening",
            ));
        }
        let n = 2 * fid.len().next_power_of_two();
        let mut data = vec![ZERO; n];
        for (k, &s) in fid.iter().enumerate() {
            let w = (-std::f64::consts::PI * broadening * k as f64 * dwell).exp();
            data[k] = s * w * if k == 0 { 0.5 } else { 1.0 };
        }
        fft(&mut data);
        // Shift so that frequencies ascend from −N/2.
        let half = n / 2;
        let values: Vec<C64> = (0..n).map(|j| data[(j + half) % n]).collect();
        let frequencies = (0..n)
            .map(|j| (j as f64 - half as f64) / (n as f64 * dwell))
            .collect();
        Ok(Spectrum {
            frequencies,
            values,
        })
    }
}

/// In-place radix-2 FFT, `X_j = Σ_k x_k e^{−2πi jk/N}` (`N` a power of
/// two).
fn fft(data: &mut [C64]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            data.swap(i, j);
        }
    }
    // e^{−2πik/N}, each computed directly.
    let twiddles: Vec<C64> = (0..n / 2)
        .map(|k| C64::from_polar(1.0, -2.0 * std::f64::consts::PI * k as f64 / n as f64))
        .collect();
    let mut len = 2;
    while len <= n {
        let stride = n / len;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let a = data[start + k];
                let b = data[start + k + len / 2] * twiddles[k * stride];
                data[start + k] = a + b;
                data[start + k + len / 2] = a - b;
            }
        }
        len <<= 1;
    }
}

/// INEPT from `source` to `target` for coupling `j_hz`:
/// `90°x(S) – τ – 180°x(S, T) – τ – 90°y(S) 90°x(T)`, `τ = 1/(4|J|)`. From
/// thermal equilibrium the target's lines become antiphase with
/// `γ_S/γ_T` times the amplitude of direct excitation, plus the target's
/// own (direct) contribution. The isotopes differ and `J` is finite and
/// not zero.
pub fn inept(source: Isotope, target: Isotope, j_hz: f64) -> Result<Vec<Event>> {
    inept_with(source, target, j_hz, 90.0)
}

/// INEPT with the last source pulse at `phase`.
fn inept_with(source: Isotope, target: Isotope, j_hz: f64, phase: f64) -> Result<Vec<Event>> {
    if source == target {
        return Err(QsoError::invalid(
            "inept",
            "INEPT moves polarisation between two isotopes",
        ));
    }
    if !(j_hz.is_finite() && j_hz != 0.0) {
        return Err(QsoError::invalid(
            "inept",
            format!("the coupling must be finite and not zero, not {j_hz}"),
        ));
    }
    let tau = 1.0 / (4.0 * j_hz.abs());
    Ok(vec![
        Event::pulse(Target::Isotope(source), 90.0, 0.0),
        Event::Delay(tau),
        Event::pulse(Target::Isotope(source), 180.0, 0.0),
        Event::pulse(Target::Isotope(target), 180.0, 0.0),
        Event::Delay(tau),
        Event::pulse(Target::Isotope(source), 90.0, phase),
        Event::pulse(Target::Isotope(target), 90.0, 0.0),
    ])
}

/// INEPT with the two-step phase cycle that removes the target's own
/// magnetisation: the last source pulse at ±y, the receiver at 0° and
/// 180°.
pub fn inept_cycle(source: Isotope, target: Isotope, j_hz: f64) -> Result<Vec<(Vec<Event>, f64)>> {
    Ok(vec![
        (inept_with(source, target, j_hz, 90.0)?, 0.0),
        (inept_with(source, target, j_hz, 270.0)?, 180.0),
    ])
}

/// A spin echo on an isotope: `90°x – τ – 180°y – τ`.
pub fn spin_echo(isotope: Isotope, tau: f64) -> Vec<Event> {
    vec![
        Event::pulse(Target::Isotope(isotope), 90.0, 0.0),
        Event::Delay(tau),
        Event::pulse(Target::Isotope(isotope), 180.0, 90.0),
        Event::Delay(tau),
    ]
}

/// Inversion recovery on an isotope: `180°x – t – 90°x`.
pub fn inversion_recovery(isotope: Isotope, t: f64) -> Vec<Event> {
    vec![
        Event::pulse(Target::Isotope(isotope), 180.0, 0.0),
        Event::Delay(t),
        Event::pulse(Target::Isotope(isotope), 90.0, 0.0),
    ]
}

/// Magnetisation-to-singlet conversion (M2S) of a near-equivalent pair of
/// an isotope with coupling `j_hz` and shift difference `delta_nu` (Hz)
/// (Pileio, Carravetta and Levitt, PNAS 107, 17135 (2010)):
/// `90°(−y) − [τ − 180°x − τ]^{n₁} − 90°x − τ − [τ − 180°x − τ]^{n₂}` with
/// `τ = 1/(4J)`, `n₁ = round(π/(2 arctan(Δν/J)))` and `n₂ = n₁/2` rounded
/// down. It turns `I_z` into singlet order; [`s2m`] turns it back. Needs
/// `0 < Δν < J`.
pub fn m2s(isotope: Isotope, j_hz: f64, delta_nu: f64) -> Result<Vec<Event>> {
    let (j, dnu) = (j_hz.abs(), delta_nu.abs());
    if !(j.is_finite() && dnu.is_finite() && dnu > 0.0 && dnu < j) {
        return Err(QsoError::invalid(
            "m2s",
            "M2S needs a coupling larger than a non-zero shift difference",
        ));
    }
    let tau = 1.0 / (4.0 * j);
    let n1 = (std::f64::consts::PI / (2.0 * (dnu / j).atan())).round() as usize;
    if n1 > MAX_SEGMENTS {
        return Err(QsoError::invalid(
            "m2s",
            "the shift difference is too small",
        ));
    }
    let n2 = n1 / 2;
    let target = Target::Isotope(isotope);
    let echo = |out: &mut Vec<Event>, n: usize| {
        for _ in 0..n {
            out.push(Event::Delay(tau));
            out.push(Event::pulse(target, 180.0, 0.0));
            out.push(Event::Delay(tau));
        }
    };
    let mut out = vec![Event::pulse(target, 90.0, 270.0)];
    echo(&mut out, n1);
    out.push(Event::pulse(target, 90.0, 0.0));
    out.push(Event::Delay(tau));
    echo(&mut out, n2);
    Ok(out)
}

/// Singlet-to-magnetisation conversion (S2M): [`m2s`] in reverse
/// chronological order.
pub fn s2m(isotope: Isotope, j_hz: f64, delta_nu: f64) -> Result<Vec<Event>> {
    let mut out = m2s(isotope, j_hz, delta_nu)?;
    out.reverse();
    Ok(out)
}

/// Continuous irradiation of an isotope at nutation frequency `nu1` (Hz)
/// and `phase` (degrees) for `duration` s: a spin lock, which suppresses
/// shift differences much smaller than `nu1` and so keeps a
/// near-equivalent pair's singlet an eigenstate (rotating frame).
pub fn spin_lock(isotope: Isotope, nu1: f64, duration: f64, phase: f64) -> Event {
    Event::Shaped {
        target: Target::Isotope(isotope),
        segments: vec![Segment {
            duration,
            amplitude: nu1,
            phase,
        }],
    }
}
