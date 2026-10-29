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
//! shaped pulses of piecewise-constant amplitude and phase, delays, and
//! changes and sweeps of the field. Free evolution is exact (from the
//! eigendecomposition of `H`); with relaxation it is the Lindblad
//! equation, exact through the exponential of its superoperator up to
//! [`MAX_SUPEROPERATOR_DIMENSION`] and by adaptive Dormand–Prince steps
//! (local error 10⁻¹²) above.
//!
//! Relaxation is phenomenological, per spin: `T₁` towards thermal
//! equilibrium at the current field and temperature (jump operators
//! `√k↓ I₋` and `√k↑ I₊` with `k↑ + k↓ = 1/T₁` and
//! `k↑/k↓ = exp(ħγB/k_BT)`), and pure dephasing `√(2γ_φ) I_z` with
//! `γ_φ = 1/T₂ − 1/(2T₁)`. For spin ½ this gives exactly the decay
//! times `T₁` and `T₂`; for spin 1 the same operators, so `T₂` is only
//! approximate. Cross-relaxation and Redfield theory are not modelled.
//!
//! The signal is `s(t) = Tr(ρ(t) F₊)`, `F₊ = Σ_{i ∈ X} I₊ᵢ`: a spin with
//! offset Ω gives `e^{iΩt}`. [`SpinSystem::lines`] gives the exact stick
//! spectrum of a state, [`SpinSystem::fid`] the free induction decay and
//! [`Spectrum::from_fid`] its Fourier transform.
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
//! let inept = metatron_qso::spin::inept(Isotope::H1, Isotope::C13, 140.0);
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
use crate::linalg::{CMatrix, eigh, expm_hermitian};

#[cfg(test)]
mod tests;

/// Reduced Planck constant (J s, CODATA 2018, exact).
pub const HBAR: f64 = 1.054_571_817e-34;
/// Boltzmann constant (J/K, exact).
pub const BOLTZMANN: f64 = 1.380_649e-23;
/// Largest dimension of a spin system's Hilbert space (11 spins ½).
pub const MAX_SPIN_DIMENSION: usize = 2048;
/// Largest dimension whose relaxation during a delay is propagated by the
/// exact superoperator exponential (`d² × d²`, three spins ½); above,
/// adaptive Dormand–Prince steps are faster.
pub const MAX_SUPEROPERATOR_DIMENSION: usize = 8;
/// Largest dimension whose free induction decay with relaxation steps by
/// one superoperator exponential for all its points (four spins ½).
pub const MAX_FID_SUPEROPERATOR_DIMENSION: usize = 16;
/// Most points of a free induction decay.
pub const MAX_FID_POINTS: usize = 1 << 20;
/// Most segments of a shaped pulse or a field sweep.
pub const MAX_SEGMENTS: usize = 1 << 20;

const ZERO: C64 = C64::new(0.0, 0.0);
const ONE: C64 = C64::new(1.0, 0.0);

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
pub struct Segment {
    pub duration: f64,
    pub amplitude: f64,
    pub phase: f64,
}

/// A step of a pulse sequence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
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
    /// on the target during each segment (rotating frame only).
    Shaped {
        target: Target,
        segments: Vec<Segment>,
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
        });
        Ok(self.spins.len() - 1)
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
    /// `2T₁`). For spin ½, `T₂ ≤ 2T₁`.
    pub fn set_relaxation(
        &mut self,
        i: usize,
        t1: Option<f64>,
        t2: Option<f64>,
    ) -> Result<&mut Self> {
        self.check_spin(i)?;
        self.spins[i].t1 = t1;
        self.spins[i].t2 = t2;
        self.check_relaxation(i)?;
        Ok(self)
    }

    fn check_relaxation(&self, i: usize) -> Result<()> {
        let s = &self.spins[i];
        for t in [s.t1, s.t2].into_iter().flatten() {
            if !(t.is_finite() && t > 0.0) {
                return Err(QsoError::invalid(
                    "relaxation",
                    format!("relaxation times must be finite and positive, not {t}"),
                ));
            }
        }
        if let (Some(t1), Some(t2)) = (s.t1, s.t2)
            && t2 > 2.0 * t1 * (1.0 + 1e-12)
        {
            return Err(QsoError::invalid(
                "relaxation",
                format!("T2 = {t2} exceeds 2·T1 = {}", 2.0 * t1),
            ));
        }
        Ok(())
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
        for (i, s) in self.spins.iter().enumerate() {
            if !s.shift.is_finite() {
                return Err(QsoError::invalid("shift", "shifts must be finite"));
            }
            self.check_relaxation(i)?;
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
        Ok(())
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
        let n = self.spins.len();
        let mut h = CMatrix::zeros(d, d);
        let omegas: Vec<f64> = (0..n).map(|i| self.zeeman(i, b)).collect();
        let strides: Vec<usize> = (0..n).map(|i| self.stride(i)).collect();
        let spin = |i: usize| self.spins[i].isotope.spin();
        let levels = |i: usize| self.spins[i].isotope.levels();
        let two_pi = 2.0 * std::f64::consts::PI;
        for s in 0..d {
            let mut diag = 0.0;
            for (i, &w) in omegas.iter().enumerate() {
                diag += w * self.m(i, s);
            }
            for &(i, j, jh) in &self.couplings {
                diag += two_pi * jh * self.m(i, s) * self.m(j, s);
            }
            h[(s, s)] = C64::new(diag, 0.0);
            for &(i, j, jh) in &self.couplings {
                let full = self.frame == Frame::Laboratory
                    || self.spins[i].isotope == self.spins[j].isotope;
                if !full {
                    continue;
                }
                // ½(I₊ᵢI₋ⱼ + I₋ᵢI₊ⱼ): |…m_i…m_j…⟩ → |…m_i+1…m_j−1…⟩ and back.
                let (mi, mj) = (self.m(i, s), self.m(j, s));
                let (si, sj) = (spin(i), spin(j));
                let ki = (s / strides[i]) % levels(i);
                let kj = (s / strides[j]) % levels(j);
                if ki > 0 && kj + 1 < levels(j) {
                    // m_i + 1 (index ki − 1), m_j − 1 (index kj + 1).
                    let a = (si * (si + 1.0) - mi * (mi + 1.0)).sqrt();
                    let c = (sj * (sj + 1.0) - mj * (mj - 1.0)).sqrt();
                    let t = s - strides[i] + strides[j];
                    let v = C64::new(two_pi * jh * 0.5 * a * c, 0.0);
                    h[(t, s)] += v;
                    h[(s, t)] += v;
                }
            }
        }
        h
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

    fn thermal_at(&self, b: f64) -> CMatrix {
        let d = self.dimension();
        let n = self.spins.len();
        let x: Vec<f64> = (0..n).map(|i| self.boltzmann(i, b)).collect();
        // Normalise per spin: Σ_m e^{x m}.
        let z: Vec<f64> = (0..n)
            .map(|i| {
                let s = self.spins[i].isotope.spin();
                (0..self.spins[i].isotope.levels())
                    .map(|k| (x[i] * (s - k as f64)).exp())
                    .sum()
            })
            .collect();
        CMatrix::from_fn(d, d, |r, c| {
            if r != c {
                return ZERO;
            }
            let p: f64 = (0..n).map(|i| (x[i] * self.m(i, r)).exp() / z[i]).product();
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
        let mut out = self.iz(i)? * self.iz(j)?;
        out += self.ix(i)? * self.ix(j)?;
        out += self.iy(i)? * self.iy(j)?;
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
    /// removed (|ΔE| above 10⁻⁹ of the largest |E|, at least 10⁻⁹ rad/s).
    /// Models a state formed incoherently over a long time, such as
    /// parahydrogen added at high field (PASADENA).
    pub fn time_average(&self, rho: &CMatrix) -> Result<CMatrix> {
        self.validate()?;
        self.check_state(rho)?;
        let (e, v) = eigh(&self.hamiltonian_at(self.field));
        let scale = e.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let tol = (1e-9 * scale).max(1e-9);
        let mut t = v.adjoint() * rho * &v;
        let d = e.len();
        for a in 0..d {
            for b in 0..d {
                if (e[a] - e[b]).abs() > tol {
                    t[(a, b)] = ZERO;
                }
            }
        }
        Ok(&v * t * v.adjoint())
    }

    /// Whether any spin relaxes.
    fn relaxes(&self) -> bool {
        self.spins.iter().any(|s| s.t1.is_some() || s.t2.is_some())
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

    /// Apply a sequence of events to `rho` (the field starts at the
    /// system's and may change).
    pub fn evolve(&self, rho: &CMatrix, events: &[Event]) -> Result<CMatrix> {
        self.validate()?;
        self.check_state(rho)?;
        let mut b = self.field;
        let mut rho = rho.clone();
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
                        rho = self.propagate(&rho, &h, seg.duration, b)?;
                    }
                }
                Event::Delay(t) => {
                    check_time("delay", *t)?;
                    rho = self.propagate(&rho, &self.hamiltonian_at(b), *t, b)?;
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
                        rho = self.propagate(&rho, &self.hamiltonian_at(bk), dt, bk)?;
                    }
                    b = *to;
                }
            }
        }
        Ok(rho)
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
    /// target of relaxation).
    fn propagate(&self, rho: &CMatrix, h: &CMatrix, t: f64, b: f64) -> Result<CMatrix> {
        if t == 0.0 {
            return Ok(rho.clone());
        }
        if !self.relaxes() {
            let u = expm_hermitian(h, t);
            return Ok(&u * rho * u.adjoint());
        }
        let jumps = self.jumps(b);
        if self.dimension() <= MAX_SUPEROPERATOR_DIMENSION {
            let p = self.superoperator(h, &jumps)? * C64::new(t, 0.0);
            let p = p.exp();
            let v = &p * CMatrix::from_column_slice(rho.len(), 1, rho.as_slice());
            return Ok(CMatrix::from_column_slice(
                rho.nrows(),
                rho.ncols(),
                v.as_slice(),
            ));
        }
        self.dormand_prince(rho, h, &jumps, t)
    }

    /// The Lindblad superoperator on column-stacked `vec(ρ)`:
    /// `vec(AρB) = (Bᵀ ⊗ A) vec(ρ)`.
    fn superoperator(&self, h: &CMatrix, jumps: &[(usize, CMatrix)]) -> Result<CMatrix> {
        let d = self.dimension();
        let id = CMatrix::identity(d, d);
        let mut l = (id.kronecker(h) - h.transpose().kronecker(&id)) * C64::new(0.0, -1.0);
        for (i, op) in jumps {
            let a = self.embed(*i, op)?;
            let ada = a.adjoint() * &a;
            l += a.conjugate().kronecker(&a);
            l -= id.kronecker(&ada) * C64::new(0.5, 0.0);
            l -= ada.transpose().kronecker(&id) * C64::new(0.5, 0.0);
        }
        Ok(l)
    }

    /// `dρ/dt = −i[H, ρ] + Σ (LρL† − ½{L†L, ρ})` with local jumps.
    fn lindblad(&self, rho: &CMatrix, h: &CMatrix, jumps: &[(usize, CMatrix)]) -> CMatrix {
        let mut out = (h * rho - rho * h) * C64::new(0.0, -1.0);
        for (i, op) in jumps {
            let ada = op.adjoint() * op;
            let lr = self.left_local(rho, *i, op);
            out += self.right_local(&lr, *i, &op.adjoint());
            let a = self.left_local(rho, *i, &ada);
            let c = self.right_local(rho, *i, &ada);
            out -= (a + c) * C64::new(0.5, 0.0);
        }
        out
    }

    /// Adaptive Dormand–Prince 5(4) steps of the Lindblad equation, local
    /// error per step below 10⁻¹² (largest entry).
    fn dormand_prince(
        &self,
        rho: &CMatrix,
        h: &CMatrix,
        jumps: &[(usize, CMatrix)],
        t: f64,
    ) -> Result<CMatrix> {
        const TOL: f64 = 1e-12;
        const A: [[f64; 6]; 6] = [
            [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
            [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
            [
                19372.0 / 6561.0,
                -25360.0 / 2187.0,
                64448.0 / 6561.0,
                -212.0 / 729.0,
                0.0,
                0.0,
            ],
            [
                9017.0 / 3168.0,
                -355.0 / 33.0,
                46732.0 / 5247.0,
                49.0 / 176.0,
                -5103.0 / 18656.0,
                0.0,
            ],
            [
                35.0 / 384.0,
                0.0,
                500.0 / 1113.0,
                125.0 / 192.0,
                -2187.0 / 6784.0,
                11.0 / 84.0,
            ],
        ];
        // Fifth-order weights are the last row of A; the embedded fourth
        // order differs by E.
        const E: [f64; 7] = [
            71.0 / 57600.0,
            0.0,
            -71.0 / 16695.0,
            71.0 / 1920.0,
            -17253.0 / 339200.0,
            22.0 / 525.0,
            -1.0 / 40.0,
        ];
        let norm = h.iter().map(|z| z.norm()).fold(0.0, f64::max) * h.nrows() as f64
            + jumps
                .iter()
                .map(|(_, op)| op.iter().map(|z| z.norm_sqr()).sum::<f64>())
                .sum::<f64>();
        let mut dt = (0.5 / norm.max(1e-300)).min(t);
        let mut now = 0.0;
        let mut rho = rho.clone();
        let mut k1 = self.lindblad(&rho, h, jumps);
        let mut steps = 0usize;
        while now < t {
            steps += 1;
            if steps > 100_000_000 {
                return Err(QsoError::NoConvergence {
                    method: "Dormand–Prince",
                    iterations: steps,
                    residual: t - now,
                });
            }
            let step = dt.min(t - now);
            let mut k = vec![k1.clone()];
            for row in A.iter().take(6) {
                let mut y = rho.clone();
                for (j, kj) in k.iter().enumerate() {
                    if row[j] != 0.0 {
                        y += kj * C64::new(step * row[j], 0.0);
                    }
                }
                k.push(self.lindblad(&y, h, jumps));
            }
            // k[6] is the derivative at the fifth-order solution (FSAL).
            let mut next = rho.clone();
            for (j, kj) in k.iter().take(6).enumerate() {
                if A[5][j] != 0.0 {
                    next += kj * C64::new(step * A[5][j], 0.0);
                }
            }
            let mut err = CMatrix::zeros(rho.nrows(), rho.ncols());
            for (j, kj) in k.iter().enumerate() {
                if E[j] != 0.0 {
                    err += kj * C64::new(step * E[j], 0.0);
                }
            }
            let e = err.iter().map(|z| z.norm()).fold(0.0, f64::max);
            if e <= TOL || step <= 1e-14 * t.max(1.0) {
                rho = next;
                now += step;
                k1 = k.pop().expect("seven stages");
                let grow = if e > 0.0 {
                    0.9 * (TOL / e).powf(0.2)
                } else {
                    5.0
                };
                dt = step * grow.clamp(0.2, 5.0);
            } else {
                dt = step * (0.9 * (TOL / e).powf(0.2)).clamp(0.1, 0.9);
            }
        }
        Ok(rho)
    }

    /// Exact stick spectrum of `rho` on an isotope under free evolution
    /// (no relaxation): in the eigenbasis of `H`, every pair of levels
    /// gives `ρ_ab (F₊)_ba e^{i(E_b − E_a)t}`. Lines at one frequency (to
    /// 10⁻⁶ Hz) are merged and those below 10⁻¹⁴ of the largest dropped;
    /// sorted by frequency.
    pub fn lines(&self, rho: &CMatrix, isotope: Isotope) -> Result<Vec<Line>> {
        self.validate()?;
        self.check_state(rho)?;
        let fp = self.fplus(isotope)?;
        let (e, v) = eigh(&self.hamiltonian_at(self.field));
        let rt = v.adjoint() * rho * &v;
        let ft = v.adjoint() * fp * &v;
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut raw: Vec<(f64, C64)> = Vec::new();
        let d = e.len();
        for a in 0..d {
            for b in 0..d {
                let amp = rt[(a, b)] * ft[(b, a)];
                if amp != ZERO {
                    raw.push(((e[b] - e[a]) / two_pi, amp));
                }
            }
        }
        raw.sort_by(|x, y| x.0.total_cmp(&y.0));
        let mut merged: Vec<(f64, C64)> = Vec::new();
        for (f, a) in raw {
            match merged.last_mut() {
                Some((g, b)) if (f - *g).abs() <= 1e-6 => *b += a,
                _ => merged.push((f, a)),
            }
        }
        let largest = merged.iter().map(|(_, a)| a.norm()).fold(0.0, f64::max);
        Ok(merged
            .into_iter()
            .filter(|(_, a)| a.norm() > 1e-14 * largest)
            .map(|(f, a)| Line {
                frequency: f,
                ppm: self.ppm(isotope, f),
                amplitude: a,
            })
            .collect())
    }

    /// Chemical shift (ppm) of a frequency (Hz) in the system's frame.
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
    /// `points` points, with relaxation if any spin relaxes.
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
        let fp = self.fplus(isotope)?;
        let h = self.hamiltonian_at(self.field);
        if !self.relaxes() {
            let (e, v) = eigh(&h);
            let rt = v.adjoint() * rho * &v;
            let ft = v.adjoint() * fp * &v;
            let d = e.len();
            let mut terms: Vec<(f64, C64)> = Vec::new();
            for a in 0..d {
                for b in 0..d {
                    let amp = rt[(a, b)] * ft[(b, a)];
                    if amp.norm() > 0.0 {
                        terms.push((e[b] - e[a], amp));
                    }
                }
            }
            let largest = terms.iter().map(|t| t.1.norm()).fold(0.0, f64::max);
            terms.retain(|t| t.1.norm() > 1e-15 * largest);
            return Ok((0..points)
                .map(|k| {
                    let t = k as f64 * dwell;
                    terms
                        .iter()
                        .map(|&(w, a)| a * C64::from_polar(1.0, w * t))
                        .sum()
                })
                .collect());
        }
        let mut out = Vec::with_capacity(points);
        let mut r = rho.clone();
        let jumps = self.jumps(self.field);
        let step = if self.dimension() <= MAX_FID_SUPEROPERATOR_DIMENSION {
            Some((self.superoperator(&h, &jumps)? * C64::new(dwell, 0.0)).exp())
        } else {
            None
        };
        for k in 0..points {
            if k > 0 {
                r = match &step {
                    Some(p) => {
                        let v = p * CMatrix::from_column_slice(r.len(), 1, r.as_slice());
                        CMatrix::from_column_slice(r.nrows(), r.ncols(), v.as_slice())
                    }
                    None => self.dormand_prince(&r, &h, &jumps, dwell)?,
                };
            }
            out.push(self.expectation(&r, &fp)?);
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
    let mut len = 2;
    while len <= n {
        let w = C64::from_polar(1.0, -2.0 * std::f64::consts::PI / len as f64);
        for start in (0..n).step_by(len) {
            let mut wk = ONE;
            for k in 0..len / 2 {
                let a = data[start + k];
                let b = data[start + k + len / 2] * wk;
                data[start + k] = a + b;
                data[start + k + len / 2] = a - b;
                wk *= w;
            }
        }
        len <<= 1;
    }
}

/// INEPT from `source` to `target` for coupling `j_hz`:
/// `90°x(S) – τ – 180°x(S, T) – τ – 90°y(S) 90°x(T)`, `τ = 1/(4J)`. From
/// thermal equilibrium the target's lines become antiphase with
/// `γ_S/γ_T` times the amplitude of direct excitation, plus the target's
/// own (direct) contribution.
pub fn inept(source: Isotope, target: Isotope, j_hz: f64) -> Vec<Event> {
    let tau = 1.0 / (4.0 * j_hz.abs());
    vec![
        Event::pulse(Target::Isotope(source), 90.0, 0.0),
        Event::Delay(tau),
        Event::pulse(Target::Isotope(source), 180.0, 0.0),
        Event::pulse(Target::Isotope(target), 180.0, 0.0),
        Event::Delay(tau),
        Event::pulse(Target::Isotope(source), 90.0, 90.0),
        Event::pulse(Target::Isotope(target), 90.0, 0.0),
    ]
}

/// INEPT with the two-step phase cycle that removes the target's own
/// magnetisation: the last source pulse at ±y, the receiver at 0° and
/// 180°.
pub fn inept_cycle(source: Isotope, target: Isotope, j_hz: f64) -> Vec<(Vec<Event>, f64)> {
    let plus = inept(source, target, j_hz);
    let mut minus = plus.clone();
    minus[5] = Event::pulse(Target::Isotope(source), 90.0, 270.0);
    vec![(plus, 0.0), (minus, 180.0)]
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
