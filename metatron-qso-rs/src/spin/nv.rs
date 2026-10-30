//! Quantum sensing with nitrogen-vacancy (NV) centres in diamond.
//!
//! An [`NvCenter`] is the electron spin 1 of the centre's ground state with
//! its zero-field splitting `D`, the Zeeman term of a field along the NV
//! axis, the nitrogen nucleus (¹⁴N with its quadrupole splitting, or ¹⁵N)
//! and nearby ¹³C nuclei. In the secular approximation (`D` far above the
//! hyperfine couplings) only `S_z` couples to the nuclei:
//!
//! ```text
//! H/2π = (D − f) S_z² + γ_e B S_z + A_N S_z I_z^N + P (I_z^N)² − γ_N B I_z^N
//!      + Σ_k [S_z (A_∥ I_z + A_⊥ I_x)_k − γ_C B I_z^k]
//! ```
//!
//! in the frame rotating at the microwave frequency `f` (Hz), where a
//! microwave field of Rabi frequency `Ω` drives both electron transitions
//! `0 ↔ ±1` as `2π(Ω/√2)(cos φ S_x + sin φ S_y)` (rotating-wave
//! approximation). Pulses act on the qubit `{0, m}` of one transition
//! (`m = −1` by default), ideally (instantaneous and selective) or with
//! finite duration. The electron starts in `m_s = 0` (optical pumping), the
//! nuclei unpolarised, and the readout is the population of `m_s = 0`.
//!
//! Classical field noise `b(t)` along the axis adds `2π b(t) S_z` (Hz):
//! [`FieldNoise`] (telegraph, Ornstein–Uhlenbeck, 1/f) gives its spectrum,
//! the Gaussian decay `e^{−χ}` of a sequence from its filter function
//! (Cywiński et al., Phys. Rev. B 77, 174509 (2008)), and exact
//! trajectories: between pulses `S_z` commutes with the Hamiltonian, so a
//! trajectory needs only the integral of the noise over each interval,
//! which is sampled exactly (jointly Gaussian for Ornstein–Uhlenbeck,
//! from the jump times for telegraph noise).

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::{Isotope, TWO_PI, ZERO, spin_ops};
use crate::error::{QsoError, Result};
use crate::linalg::{CMatrix, expm_hermitian};
use crate::rng::Rng;

/// Zero-field splitting of the NV ground state (Hz; Doherty et al.,
/// Phys. Rep. 528, 1 (2013)).
pub const NV_ZERO_FIELD_SPLITTING: f64 = 2.87e9;

/// Gyromagnetic ratio of the NV electron spin, `g = 2.0028` (Hz/T).
pub const NV_GYROMAGNETIC_RATIO: f64 = 28.025e9;

/// Gyromagnetic ratio of ¹⁴N (Hz/T).
const N14_GAMMA: f64 = 3.077_7e6;

/// Largest dimension of an NV centre with its nuclei.
pub const MAX_NV_DIMENSION: usize = 768;

/// The nitrogen of the centre.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NitrogenIsotope {
    #[serde(rename = "14N")]
    N14,
    #[serde(rename = "15N")]
    N15,
}

/// The nitrogen nucleus: its hyperfine coupling along the axis `A_N` and,
/// for ¹⁴N, its quadrupole splitting `P` (Hz).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Nitrogen {
    pub isotope: NitrogenIsotope,
    pub parallel: f64,
    pub quadrupole: f64,
}

impl Nitrogen {
    /// ¹⁴N with `A_N = −2.14 MHz` and `P = −4.945 MHz` (Doherty et al.
    /// 2013).
    pub fn n14() -> Self {
        Nitrogen {
            isotope: NitrogenIsotope::N14,
            parallel: -2.14e6,
            quadrupole: -4.945e6,
        }
    }

    /// ¹⁵N with `A_N = 3.03 MHz`.
    pub fn n15() -> Self {
        Nitrogen {
            isotope: NitrogenIsotope::N15,
            parallel: 3.03e6,
            quadrupole: 0.0,
        }
    }

    fn levels(&self) -> usize {
        match self.isotope {
            NitrogenIsotope::N14 => 3,
            NitrogenIsotope::N15 => 2,
        }
    }

    /// Gyromagnetic ratio (Hz/T).
    fn gamma(&self) -> f64 {
        match self.isotope {
            NitrogenIsotope::N14 => N14_GAMMA,
            NitrogenIsotope::N15 => Isotope::N15.gamma() / TWO_PI,
        }
    }
}

/// A ¹³C nucleus: the hyperfine couplings `A_∥` (along the axis) and `A_⊥`
/// (perpendicular, along x) with the electron's `S_z` (Hz).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Carbon {
    pub parallel: f64,
    pub perpendicular: f64,
}

/// An NV centre with its nuclei in a field `field` (T) along its axis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NvCenter {
    pub field: f64,
    pub zero_field_splitting: f64,
    pub gyromagnetic_ratio: f64,
    pub nitrogen: Option<Nitrogen>,
    pub carbons: Vec<Carbon>,
}

/// A step of a sensing sequence on the qubit `{0, m}`: a pulse (angle and
/// phase in degrees) or free evolution (s).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
pub enum NvEvent {
    Pulse { angle: f64, phase: f64 },
    Free(f64),
}

/// How pulses are applied: ideal (instantaneous, on the chosen transition
/// alone) or with a Rabi frequency (Hz) that drives both transitions; then
/// a pulse of angle `θ` lasts `θ/(2πΩ)` and free periods count between
/// pulse centres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Microwave {
    /// Microwave frequency (Hz), the frame of the Hamiltonian.
    pub frequency: f64,
    /// The qubit `{0, m}`: `m = −1` or `+1`.
    pub transition: i8,
    /// `None`: ideal pulses.
    pub rabi: Option<f64>,
}

fn pulse(angle: f64, phase: f64) -> NvEvent {
    NvEvent::Pulse { angle, phase }
}

/// Ramsey: `π/2 − τ − π/2` (the last about `−x`, so that the signal is
/// `(1 + coherence)/2`).
pub fn ramsey(tau: f64) -> Vec<NvEvent> {
    vec![pulse(90.0, 0.0), NvEvent::Free(tau), pulse(90.0, 180.0)]
}

/// Hahn echo: `π/2_x − τ − π_y − τ − π/2_{−x}`.
pub fn hahn(tau: f64) -> Vec<NvEvent> {
    cpmg(1, tau)
}

/// CPMG with `n` π pulses about y spaced `2τ`: `π/2_x − (τ − π_y − τ)^n −
/// π/2_{−x}`.
pub fn cpmg(n: usize, tau: f64) -> Vec<NvEvent> {
    spaced(&vec![90.0; n], tau)
}

/// XY-8: `n` blocks of π pulses with phases `x y x y y x y x`, spaced `2τ`.
pub fn xy8(n: usize, tau: f64) -> Vec<NvEvent> {
    let block = [0.0, 90.0, 0.0, 90.0, 90.0, 0.0, 90.0, 0.0];
    let phases: Vec<f64> = (0..n).flat_map(|_| block).collect();
    spaced(&phases, tau)
}

/// `π/2_x − (τ − π_φ − τ)… − π/2_{−x}`, adjacent free periods merged.
fn spaced(phases: &[f64], tau: f64) -> Vec<NvEvent> {
    let mut out = vec![pulse(90.0, 0.0)];
    for (k, &phase) in phases.iter().enumerate() {
        out.push(NvEvent::Free(if k == 0 { tau } else { 2.0 * tau }));
        out.push(pulse(180.0, phase));
    }
    out.push(NvEvent::Free(tau));
    out.push(pulse(90.0, 180.0));
    out
}

/// Kronecker product of a list of matrices.
fn kron(mats: &[CMatrix]) -> CMatrix {
    mats.iter()
        .skip(1)
        .fold(mats[0].clone(), |acc, m| acc.kronecker(m))
}

impl NvCenter {
    /// A bare centre (no nuclei) in a field `b` (T) along its axis.
    pub fn new(field: f64) -> Self {
        NvCenter {
            field,
            zero_field_splitting: NV_ZERO_FIELD_SPLITTING,
            gyromagnetic_ratio: NV_GYROMAGNETIC_RATIO,
            nitrogen: None,
            carbons: Vec::new(),
        }
    }

    /// With a nitrogen nucleus.
    pub fn with_nitrogen(mut self, nitrogen: Nitrogen) -> Self {
        self.nitrogen = Some(nitrogen);
        self
    }

    /// With a ¹³C nucleus of hyperfine couplings `A_∥`, `A_⊥` (Hz).
    pub fn with_carbon(mut self, parallel: f64, perpendicular: f64) -> Self {
        self.carbons.push(Carbon {
            parallel,
            perpendicular,
        });
        self
    }

    /// Levels of each factor: electron, nitrogen, carbons.
    fn levels(&self) -> Vec<usize> {
        let mut out = vec![3];
        if let Some(n) = &self.nitrogen {
            out.push(n.levels());
        }
        out.extend(self.carbons.iter().map(|_| 2));
        out
    }

    /// Hilbert-space dimension.
    pub fn dimension(&self) -> usize {
        self.levels().iter().product()
    }

    /// Check the parameters.
    pub fn validate(&self) -> Result<()> {
        let mut values = vec![
            self.field,
            self.zero_field_splitting,
            self.gyromagnetic_ratio,
        ];
        if let Some(n) = &self.nitrogen {
            values.extend([n.parallel, n.quadrupole]);
        }
        for c in &self.carbons {
            values.extend([c.parallel, c.perpendicular]);
        }
        if values.iter().any(|x| !x.is_finite()) {
            return Err(QsoError::invalid("nv", "parameters must be finite"));
        }
        if self.dimension() > MAX_NV_DIMENSION {
            return Err(QsoError::invalid(
                "nv",
                format!(
                    "dimension {} exceeds {MAX_NV_DIMENSION} (at most six ¹³C with ¹⁴N)",
                    self.dimension()
                ),
            ));
        }
        Ok(())
    }

    /// `op` on factor `k` (0: electron), identity elsewhere.
    fn embed(&self, k: usize, op: &CMatrix) -> CMatrix {
        let levels = self.levels();
        let mats: Vec<CMatrix> = levels
            .iter()
            .enumerate()
            .map(|(j, &n)| {
                if j == k {
                    op.clone()
                } else {
                    CMatrix::identity(n, n)
                }
            })
            .collect();
        kron(&mats)
    }

    /// Electron operators `[S_z, S_x, S_y]` (spin 1, `m = +1, 0, −1`).
    fn electron() -> [CMatrix; 3] {
        let [z, _, _, x, y] = spin_ops(3);
        [z, x, y]
    }

    /// The Hamiltonian (rad/s) in the frame rotating at `frequency` (Hz).
    pub fn hamiltonian(&self, frequency: f64) -> Result<CMatrix> {
        self.validate()?;
        let [sz, _, _] = Self::electron();
        let b = self.field;
        let c = |x: f64| C64::new(TWO_PI * x, 0.0);
        let electron = &sz * &sz * c(self.zero_field_splitting - frequency)
            + &sz * c(self.gyromagnetic_ratio * b);
        let mut h = self.embed(0, &electron);
        let sz_full = self.embed(0, &sz);
        let mut k = 1;
        if let Some(n) = &self.nitrogen {
            let [iz, _, _, _, _] = spin_ops(n.levels());
            let iz_full = self.embed(k, &iz);
            h += &sz_full * &iz_full * c(n.parallel);
            h += &iz_full * &iz_full * c(n.quadrupole);
            h -= &iz_full * c(n.gamma() * b);
            k += 1;
        }
        let gamma_c = Isotope::C13.gamma() / TWO_PI;
        for carbon in &self.carbons {
            let [iz, _, _, ix, _] = spin_ops(2);
            let (iz_full, ix_full) = (self.embed(k, &iz), self.embed(k, &ix));
            h += &sz_full * (&iz_full * c(carbon.parallel) + &ix_full * c(carbon.perpendicular));
            h -= &iz_full * c(gamma_c * b);
            k += 1;
        }
        Ok(h)
    }

    /// The optically pumped state: `m_s = 0`, nuclei unpolarised.
    pub fn initial_state(&self) -> CMatrix {
        let d = self.dimension();
        let nuclear = d / 3;
        CMatrix::from_fn(d, d, |r, c| {
            if r == c && r / nuclear == 1 {
                C64::new(1.0 / nuclear as f64, 0.0)
            } else {
                ZERO
            }
        })
    }

    /// The population of `m_s = 0` in `rho`.
    pub fn bright(&self, rho: &CMatrix) -> f64 {
        let nuclear = self.dimension() / 3;
        (nuclear..2 * nuclear).map(|i| rho[(i, i)].re).sum()
    }

    /// Microwave drive `2π(Ω/√2)(cos φ S_x + sin φ S_y)` (rad/s).
    fn drive(&self, rabi: f64, phase: f64) -> CMatrix {
        let [_, sx, sy] = Self::electron();
        let (c, s) = (phase.to_radians().cos(), phase.to_radians().sin());
        self.embed(
            0,
            &((sx * C64::new(c, 0.0) + sy * C64::new(s, 0.0))
                * C64::new(TWO_PI * rabi / std::f64::consts::SQRT_2, 0.0)),
        )
    }

    /// An ideal pulse on the qubit `{0, m}`: `exp(−iθ/2 (cos φ X + sin φ Y))`
    /// with `X`, `Y` the drive's operators restricted to that transition.
    fn ideal_pulse(&self, transition: i8, angle: f64, phase: f64) -> CMatrix {
        let [_, sx, sy] = Self::electron();
        let (c, s) = (phase.to_radians().cos(), phase.to_radians().sin());
        let g = (sx * C64::new(c, 0.0) + sy * C64::new(s, 0.0))
            * C64::new(std::f64::consts::SQRT_2, 0.0);
        let keep = if transition < 0 { [1, 2] } else { [0, 1] };
        let block = CMatrix::from_fn(3, 3, |r, col| {
            if keep.contains(&r) && keep.contains(&col) {
                g[(r, col)]
            } else {
                ZERO
            }
        });
        self.embed(0, &expm_hermitian(&block, angle.to_radians() / 2.0))
    }

    fn check_drive(&self, drive: &Microwave) -> Result<()> {
        self.validate()?;
        if drive.transition != 1 && drive.transition != -1 {
            return Err(QsoError::invalid("transition", "−1 or +1"));
        }
        if !drive.frequency.is_finite() || drive.rabi.is_some_and(|r| !(r.is_finite() && r > 0.0)) {
            return Err(QsoError::invalid(
                "drive",
                "a finite frequency and a positive Rabi frequency",
            ));
        }
        Ok(())
    }

    fn check_events(events: &[NvEvent]) -> Result<()> {
        for e in events {
            match *e {
                NvEvent::Pulse { angle, phase } if !(angle.is_finite() && phase.is_finite()) => {
                    return Err(QsoError::invalid("pulse", "angles must be finite"));
                }
                NvEvent::Free(t) if !(t.is_finite() && t >= 0.0) => {
                    return Err(QsoError::invalid("free", "times must be finite and ≥ 0"));
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// With finite pulses, free periods are measured between pulse
    /// centres: each is shortened by half the duration of each pulse next
    /// to it (an error if that leaves less than nothing).
    fn centred(events: &[NvEvent], drive: &Microwave) -> Result<Vec<NvEvent>> {
        let Some(rabi) = drive.rabi else {
            return Ok(events.to_vec());
        };
        let half = |e: Option<&NvEvent>| match e {
            Some(NvEvent::Pulse { angle, .. }) => angle.abs() / 360.0 / rabi / 2.0,
            _ => 0.0,
        };
        events
            .iter()
            .enumerate()
            .map(|(k, e)| match *e {
                NvEvent::Free(t) => {
                    let left = t
                        - half(k.checked_sub(1).and_then(|j| events.get(j)))
                        - half(events.get(k + 1));
                    if left < -1e-15 * t.max(1e-300) {
                        Err(QsoError::invalid(
                            "free",
                            "free periods (between pulse centres) shorter than the pulses",
                        ))
                    } else {
                        Ok(NvEvent::Free(left.max(0.0)))
                    }
                }
                pulse => Ok(pulse),
            })
            .collect()
    }

    /// The `m_s = 0` population after `events` from the pumped state.
    pub fn signal(&self, events: &[NvEvent], drive: &Microwave) -> Result<f64> {
        let rho = self.evolve(&self.initial_state(), events, drive)?;
        Ok(self.bright(&rho))
    }

    /// Apply `events` to `rho`.
    pub fn evolve(&self, rho: &CMatrix, events: &[NvEvent], drive: &Microwave) -> Result<CMatrix> {
        self.check_drive(drive)?;
        Self::check_events(events)?;
        let events = &Self::centred(events, drive)?;
        let d = self.dimension();
        if rho.nrows() != d || rho.ncols() != d {
            return Err(QsoError::DimensionMismatch {
                expected: d,
                actual: rho.nrows(),
            });
        }
        let h0 = self.hamiltonian(drive.frequency)?;
        let (e, v) = crate::linalg::eigh(&h0);
        let free = |t: f64| {
            let mut vp = v.clone();
            for (j, &x) in e.iter().enumerate() {
                let p = C64::from_polar(1.0, -x * t);
                for r in 0..d {
                    vp[(r, j)] *= p;
                }
            }
            &vp * v.adjoint()
        };
        let mut rho = rho.clone();
        for event in events {
            let u = match *event {
                NvEvent::Free(t) => free(t),
                NvEvent::Pulse { angle, phase } => match drive.rabi {
                    None => self.ideal_pulse(drive.transition, angle, phase),
                    Some(rabi) => {
                        expm_hermitian(&(&h0 + self.drive(rabi, phase)), angle / 360.0 / rabi)
                    }
                },
            };
            rho = &u * rho * u.adjoint();
        }
        Ok(rho)
    }

    /// The electron-spin resonance lines (Hz, from `m_s = 0`-like to `±1`-like
    /// eigenstates of the Hamiltonian at `f = 0`) with their strengths
    /// `|⟨b|S_x|a⟩|²` averaged over the unpolarised nuclei; lines below
    /// 10⁻⁶ of the strongest are dropped; sorted by frequency.
    pub fn lines(&self) -> Result<Vec<(f64, f64)>> {
        let h = self.hamiltonian(0.0)?;
        let (e, v) = crate::linalg::eigh(&h);
        let [sz, sx, _] = Self::electron();
        let sz2 = self.embed(0, &(&sz * &sz));
        let sx = self.embed(0, &sx);
        let d = self.dimension();
        let weight_zero: Vec<bool> = (0..d)
            .map(|j| {
                let col = v.column(j);
                (col.adjoint() * &sz2 * col)[(0, 0)].re < 0.5
            })
            .collect();
        let m = v.adjoint() * sx * &v;
        let nuclear = (d / 3) as f64;
        let mut out = Vec::new();
        for a in (0..d).filter(|&a| weight_zero[a]) {
            for b in (0..d).filter(|&b| !weight_zero[b]) {
                let strength = m[(b, a)].norm_sqr() / nuclear;
                out.push(((e[b] - e[a]) / TWO_PI, strength));
            }
        }
        let top = out.iter().fold(0.0f64, |t, l| t.max(l.1));
        out.retain(|l| l.1 > 1e-6 * top);
        out.sort_by(|a, b| a.0.total_cmp(&b.0));
        Ok(out)
    }

    /// Pulsed ODMR: the `m_s = 0` population after a microwave pulse of
    /// Rabi frequency `rabi` (Hz) and `duration` (s) at each frequency.
    pub fn odmr(&self, frequencies: &[f64], rabi: f64, duration: f64) -> Result<Vec<f64>> {
        self.validate()?;
        if !(rabi.is_finite() && rabi > 0.0 && duration.is_finite() && duration >= 0.0) {
            return Err(QsoError::invalid(
                "odmr",
                "a positive Rabi frequency and a duration ≥ 0",
            ));
        }
        if frequencies.iter().any(|f| !f.is_finite()) {
            return Err(QsoError::invalid("odmr", "frequencies must be finite"));
        }
        let rho = self.initial_state();
        let drive = self.drive(rabi, 0.0);
        let results = crate::parallel::map_tasks(frequencies.len(), |k| {
            let h = self.hamiltonian(frequencies[k])? + &drive;
            let u = expm_hermitian(&h, duration);
            Ok(self.bright(&(&u * &rho * u.adjoint())))
        });
        results.into_iter().collect()
    }
}

/// Classical noise `b(t)` (Hz) of the field along the NV axis, coupling as
/// `2π b(t) S_z`: a sum of components with exponential correlations
/// `c² e^{−θ|τ|}`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum FieldNoise {
    /// `±amplitude` (Hz) switching at the random times of a Poisson
    /// process of `rate` flips per second (non-Gaussian).
    Telegraph { amplitude: f64, rate: f64 },
    /// A Gaussian Ornstein–Uhlenbeck process of standard deviation `sigma`
    /// (Hz) and correlation time `correlation_time` (s).
    OrnsteinUhlenbeck { sigma: f64, correlation_time: f64 },
    /// 1/f noise between `lowest` and `highest` (Hz): `fluctuators`
    /// Ornstein–Uhlenbeck processes of equal variance (total `sigma²`) with
    /// corner frequencies spaced evenly in their logarithm (Gaussian).
    Pink {
        sigma: f64,
        lowest: f64,
        highest: f64,
        fluctuators: usize,
    },
}

impl FieldNoise {
    /// Check the parameters.
    pub fn validate(&self) -> Result<()> {
        let ok = match *self {
            FieldNoise::Telegraph { amplitude, rate } => {
                amplitude.is_finite() && rate.is_finite() && rate >= 0.0
            }
            FieldNoise::OrnsteinUhlenbeck {
                sigma,
                correlation_time,
            } => {
                sigma.is_finite()
                    && sigma >= 0.0
                    && correlation_time.is_finite()
                    && correlation_time > 0.0
            }
            FieldNoise::Pink {
                sigma,
                lowest,
                highest,
                fluctuators,
            } => {
                sigma.is_finite()
                    && sigma >= 0.0
                    && lowest.is_finite()
                    && highest.is_finite()
                    && lowest > 0.0
                    && highest >= lowest
                    && (1..=1000).contains(&fluctuators)
            }
        };
        if ok {
            Ok(())
        } else {
            Err(QsoError::invalid(
                "noise",
                "finite amplitudes, positive rates and 1..=1000 fluctuators",
            ))
        }
    }

    /// `(c², θ)` of each exponential component (Hz², 1/s).
    fn components(&self) -> Vec<(f64, f64)> {
        match *self {
            FieldNoise::Telegraph { amplitude, rate } => vec![(amplitude * amplitude, 2.0 * rate)],
            FieldNoise::OrnsteinUhlenbeck {
                sigma,
                correlation_time,
            } => vec![(sigma * sigma, 1.0 / correlation_time)],
            FieldNoise::Pink {
                sigma,
                lowest,
                highest,
                fluctuators,
            } => (0..fluctuators)
                .map(|k| {
                    let x = if fluctuators == 1 {
                        0.5
                    } else {
                        k as f64 / (fluctuators - 1) as f64
                    };
                    let f = lowest * (highest / lowest).powf(x);
                    (sigma * sigma / fluctuators as f64, TWO_PI * f)
                })
                .collect(),
        }
    }

    /// Whether the noise is Gaussian (the decay `e^{−χ}` is then exact).
    pub fn is_gaussian(&self) -> bool {
        !matches!(self, FieldNoise::Telegraph { .. })
    }

    /// `⟨b(t) b(t + τ)⟩` (Hz²).
    pub fn correlation(&self, tau: f64) -> f64 {
        self.components()
            .iter()
            .map(|(c2, theta)| c2 * (-theta * tau.abs()).exp())
            .sum()
    }

    /// The two-sided spectrum `S(ω) = ∫ ⟨b(0)b(τ)⟩ e^{iωτ} dτ` (Hz²·s) at
    /// angular frequency `ω` (rad/s).
    pub fn spectrum(&self, omega: f64) -> f64 {
        self.components()
            .iter()
            .map(|(c2, theta)| 2.0 * c2 * theta / (theta * theta + omega * omega))
            .sum()
    }
}

/// The free periods of a sequence of ideal pulses with their signs in the
/// toggling frame: `π/2`, free periods and `π` pulses, `π/2`.
fn segments(events: &[NvEvent]) -> Result<Vec<(f64, f64, f64)>> {
    let bad = || {
        QsoError::invalid(
            "sequence",
            "filter functions need π/2, free periods and π pulses, π/2",
        )
    };
    let n = events.len();
    let is_angle = |e: &NvEvent, a: f64| matches!(*e, NvEvent::Pulse { angle, .. } if (angle - a).abs() < 1e-9);
    if n < 2 || !is_angle(&events[0], 90.0) || !is_angle(&events[n - 1], 90.0) {
        return Err(bad());
    }
    let (mut t, mut sign) = (0.0, 1.0);
    let mut out = Vec::new();
    for e in &events[1..n - 1] {
        match *e {
            NvEvent::Free(dt) => {
                out.push((t, t + dt, sign));
                t += dt;
            }
            _ if is_angle(e, 180.0) => sign = -sign,
            _ => return Err(bad()),
        }
    }
    Ok(out)
}

/// The filter function `|Y(ω)|²` (s²) of a sequence of ideal pulses at
/// angular frequency `ω`: `Y(ω) = ∫ y(t) e^{iωt} dt` with `y = ±1` the sign
/// of the coherence in the toggling frame.
pub fn filter_function(events: &[NvEvent], omega: f64) -> Result<f64> {
    let mut y = C64::new(0.0, 0.0);
    for (a, b, s) in segments(events)? {
        y += if omega == 0.0 {
            C64::new(s * (b - a), 0.0)
        } else {
            (C64::from_polar(1.0, omega * b) - C64::from_polar(1.0, omega * a))
                / C64::new(0.0, omega)
                * s
        };
    }
    Ok(y.norm_sqr())
}

/// The Gaussian decay exponent `χ` of the coherence of the qubit `{0, ±1}`
/// under `noise` for a sequence of ideal pulses: `χ = ½⟨φ²⟩`, `φ = 2π∫b y
/// dt`, computed exactly in the time domain from the exponential
/// correlations (equal to `(2π)²/(2·2π) ∫ S(ω)|Y(ω)|² dω`). For Gaussian
/// noise the coherence decays as `e^{−χ}`.
pub fn decay_exponent(events: &[NvEvent], noise: &FieldNoise) -> Result<f64> {
    noise.validate()?;
    let segs = segments(events)?;
    let mut phase2 = 0.0;
    for (c2, theta) in noise.components() {
        let g = |x: f64| -(-x).exp_m1();
        for (i, &(a1, b1, s1)) in segs.iter().enumerate() {
            let l1 = b1 - a1;
            // ∫∫ over the same segment: (2/θ²)(θL − 1 + e^{−θL}).
            let x = theta * l1;
            let same = if x < 1e-4 {
                l1 * l1 * (1.0 - x / 3.0 + x * x / 12.0)
            } else {
                2.0 * (x - g(x)) / (theta * theta)
            };
            phase2 += c2 * same;
            for &(a2, b2, s2) in &segs[i + 1..] {
                let l2 = b2 - a2;
                let pair = if theta == 0.0 {
                    l1 * l2
                } else {
                    g(theta * l1) * g(theta * l2) * (-theta * (a2 - b1)).exp() / (theta * theta)
                };
                phase2 += 2.0 * c2 * s1 * s2 * pair;
            }
        }
    }
    Ok(0.5 * TWO_PI * TWO_PI * phase2)
}

/// One sampled path of the noise, advanced interval by interval.
struct NoisePath {
    /// Current value of each component (Hz).
    values: Vec<f64>,
    components: Vec<(f64, f64)>,
    telegraph: bool,
}

impl NoisePath {
    fn new(noise: &FieldNoise, rng: &mut Rng) -> Self {
        let components = noise.components();
        let telegraph = !noise.is_gaussian();
        let values = components
            .iter()
            .map(|&(c2, _)| {
                if telegraph {
                    if rng.uniform() < 0.5 {
                        c2.sqrt()
                    } else {
                        -c2.sqrt()
                    }
                } else {
                    c2.sqrt() * rng.normal()
                }
            })
            .collect();
        NoisePath {
            values,
            components,
            telegraph,
        }
    }

    /// `∫ b dt` over the next `dt` (exactly), advancing the path.
    fn integral(&mut self, dt: f64, rng: &mut Rng) -> f64 {
        let mut total = 0.0;
        for (x, &(c2, theta)) in self.values.iter_mut().zip(&self.components) {
            if self.telegraph {
                // Flips at rate θ/2 (θ = 2·rate); waiting times are
                // memoryless, so each interval draws its own.
                let rate = theta / 2.0;
                let mut left = dt;
                loop {
                    let wait = if rate > 0.0 {
                        -(1.0 - rng.uniform()).ln() / rate
                    } else {
                        f64::INFINITY
                    };
                    if wait >= left {
                        total += *x * left;
                        break;
                    }
                    total += *x * wait;
                    left -= wait;
                    *x = -*x;
                }
            } else {
                // (x', I) jointly Gaussian given x (Ornstein–Uhlenbeck).
                let y = theta * dt;
                let e = (-y).exp();
                let g = -(-y).exp_m1();
                let mean_x = *x * e;
                let mean_i = *x * g / theta;
                let var_x = c2 * -(-2.0 * y).exp_m1();
                let var_i = if y < 1e-3 {
                    c2 * dt * dt * y * (2.0 / 3.0 - y / 2.0 + 7.0 * y * y / 30.0)
                } else {
                    c2 / (theta * theta) * (2.0 * y - 3.0 + 4.0 * e - e * e)
                };
                let cov = c2 / theta * g * g;
                let (z1, z2) = (rng.normal(), rng.normal());
                let sx = var_x.max(0.0).sqrt();
                let (i_from_x, i_rest) = if sx > 0.0 {
                    let k = cov / sx;
                    (k * z1, (var_i - k * k).max(0.0).sqrt() * z2)
                } else {
                    (0.0, var_i.max(0.0).sqrt() * z2)
                };
                total += mean_i + i_from_x + i_rest;
                *x = mean_x + sx * z1;
            }
        }
        total
    }
}

impl NvCenter {
    /// The `m_s = 0` population after `events` averaged over
    /// `trajectories` exact paths of `noise` (seeded), with its standard
    /// error. Between pulses the noise enters as a phase `exp(−2πi I S_z)`
    /// with `I = ∫b dt` sampled exactly; during finite pulses it is
    /// constant over each of 16 pieces (its exact mean there).
    pub fn signal_with_noise(
        &self,
        events: &[NvEvent],
        drive: &Microwave,
        noise: &FieldNoise,
        trajectories: usize,
        seed: u64,
    ) -> Result<(f64, f64)> {
        self.check_drive(drive)?;
        Self::check_events(events)?;
        let events = &Self::centred(events, drive)?;
        noise.validate()?;
        if trajectories == 0 {
            return Err(QsoError::invalid("trajectories", "at least one"));
        }
        let d = self.dimension();
        let pulses = events
            .iter()
            .filter(|e| matches!(e, NvEvent::Pulse { .. }))
            .count();
        let per = (2 * events.len()
            + if drive.rabi.is_some() {
                16 * 12 * pulses
            } else {
                0
            }) as f64
            * (d * d * d) as f64;
        let work = per * trajectories as f64;
        if work > super::MAX_RELAXATION_WORK {
            return Err(QsoError::invalid(
                "nv",
                format!(
                    "{trajectories} trajectories need about {work:.1e} operations (limit {:.1e})",
                    super::MAX_RELAXATION_WORK
                ),
            ));
        }
        let h0 = self.hamiltonian(drive.frequency)?;
        let (e, v) = crate::linalg::eigh(&h0);
        let [sz, _, _] = Self::electron();
        let sz_diag: Vec<f64> = (0..d).map(|i| sz[(i / (d / 3), i / (d / 3))].re).collect();
        let start = self.initial_state();
        let root = Rng::seed_from_u64(seed);
        let values: Vec<f64> = crate::parallel::map_tasks(trajectories, |k| {
            let mut rng = root.fork(k as u64);
            let mut path = NoisePath::new(noise, &mut rng);
            let mut rho = start.clone();
            for event in events {
                let u = match *event {
                    NvEvent::Free(t) => {
                        let phase = TWO_PI * path.integral(t, &mut rng);
                        let mut vp = v.clone();
                        for (j, &x) in e.iter().enumerate() {
                            let p = C64::from_polar(1.0, -x * t);
                            for r in 0..d {
                                vp[(r, j)] *= p;
                            }
                        }
                        let mut u = &vp * v.adjoint();
                        // exp(−iφ S_z) commutes with the free evolution.
                        for (r, &m) in sz_diag.iter().enumerate() {
                            let p = C64::from_polar(1.0, -phase * m);
                            for c in 0..d {
                                u[(r, c)] *= p;
                            }
                        }
                        u
                    }
                    NvEvent::Pulse { angle, phase } => match drive.rabi {
                        None => self.ideal_pulse(drive.transition, angle, phase),
                        Some(rabi) => {
                            let duration = angle / 360.0 / rabi;
                            let piece = duration / 16.0;
                            let h = &h0 + self.drive(rabi, phase);
                            let mut u = CMatrix::identity(d, d);
                            for _ in 0..16 {
                                let mean = path.integral(piece, &mut rng) / piece;
                                let mut hk = h.clone();
                                for (r, &m) in sz_diag.iter().enumerate() {
                                    hk[(r, r)] += C64::new(TWO_PI * mean * m, 0.0);
                                }
                                u = expm_hermitian(&hk, piece) * u;
                            }
                            u
                        }
                    },
                };
                rho = &u * rho * u.adjoint();
            }
            self.bright(&rho)
        });
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        Ok((mean, (var / n).sqrt()))
    }
}

#[cfg(test)]
mod tests;
