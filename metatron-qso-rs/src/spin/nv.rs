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

use std::collections::HashMap;

use super::{Isotope, TWO_PI, ZERO, spin_ops};
use crate::error::{QsoError, Result};
use crate::linalg::dense::{cmul, from_spectrum};
use crate::linalg::{CMatrix, eigh, expm_hermitian};
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

impl Microwave {
    /// On resonance with the bare transition `0 ↔ m` (`D + m γ_e B`).
    pub fn resonant(nv: &NvCenter, transition: i8, rabi: Option<f64>) -> Self {
        Microwave {
            frequency: nv.zero_field_splitting
                + f64::from(transition) * nv.gyromagnetic_ratio * nv.field,
            transition,
            rabi,
        }
    }
}

fn pulse(angle: f64, phase: f64) -> NvEvent {
    NvEvent::Pulse { angle, phase }
}

/// Duration (s) and phase (degrees) of a finite pulse: a negative angle is
/// the positive one about the opposite axis.
fn finite(angle: f64, phase: f64, rabi: f64) -> (f64, f64) {
    let phase = if angle < 0.0 { phase + 180.0 } else { phase };
    (angle.abs() / 360.0 / rabi, phase)
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

/// The electron levels `m = +1, 0, −1`, in the order of its basis.
const LEVELS: [f64; 3] = [1.0, 0.0, -1.0];

/// The free evolution of the secular Hamiltonian. Only `S_z` couples to
/// the nuclei and the nuclei do not couple to each other, so the
/// Hamiltonian is block diagonal in the electron level `a`, and within a
/// block a sum of one term per nucleus: `exp(−iHt) = ⊕_a e^{−iE_a t}
/// ⊗_k exp(−i h_a^k t)`.
struct Secular {
    /// Levels of each nucleus, the first the most significant factor.
    dims: Vec<usize>,
    /// `E_a` (rad/s).
    energy: [f64; 3],
    /// Eigenvalues and eigenvectors of each `h_a^k` (rad/s).
    nuclei: [Vec<(Vec<f64>, CMatrix)>; 3],
}

/// One free period: its length, the phases `E_a t` and the factors
/// `exp(−i h_a^k t)`.
struct FreeStep {
    t: f64,
    phase: [f64; 3],
    factors: [Vec<CMatrix>; 3],
}

impl Secular {
    fn step(&self, t: f64) -> FreeStep {
        FreeStep {
            t,
            phase: self.energy.map(|e| e * t),
            factors: std::array::from_fn(|a| {
                self.nuclei[a]
                    .iter()
                    .map(|(e, v)| from_spectrum(e, v, |x| C64::from_polar(1.0, -x * t)))
                    .collect()
            }),
        }
    }
}

/// `m ← (⊗_k u_k) m` (`left`) or `m ← m (⊗_k u_k)†` for a square block
/// whose index runs over the nuclei of levels `dims`: one small product
/// per nucleus instead of one product with the full Kronecker product.
fn apply_factors(m: &mut CMatrix, factors: &[CMatrix], dims: &[usize], left: bool) {
    let n = m.nrows();
    let data = m.as_mut_slice();
    let mut stride = n;
    let mut buf = [ZERO; 3];
    for (u, &dk) in factors.iter().zip(dims) {
        stride /= dk;
        let outer = n / (stride * dk);
        for other in 0..n {
            for hi in 0..outer {
                for lo in 0..stride {
                    // Column-major: (row, column) at column · n + row.
                    let at = |i: usize| {
                        let k = (hi * dk + i) * stride + lo;
                        if left { other * n + k } else { k * n + other }
                    };
                    for (i, x) in buf.iter_mut().enumerate().take(dk) {
                        *x = data[at(i)];
                    }
                    for i in 0..dk {
                        let mut sum = ZERO;
                        for (j, x) in buf.iter().enumerate().take(dk) {
                            sum += if left {
                                u[(i, j)] * x
                            } else {
                                x * u[(i, j)].conj()
                            };
                        }
                        data[at(i)] = sum;
                    }
                }
            }
        }
    }
}

/// A density matrix as the `3 × 3` blocks of the electron levels, block
/// `(a, b)` at `3a + b`.
#[derive(Clone)]
struct Blocks {
    b: Vec<CMatrix>,
}

impl Blocks {
    fn from_full(rho: &CMatrix) -> Self {
        let n = rho.nrows() / 3;
        Blocks {
            b: (0..9)
                .map(|k| rho.view((k / 3 * n, k % 3 * n), (n, n)).into_owned())
                .collect(),
        }
    }

    fn to_full(&self) -> CMatrix {
        let n = self.b[0].nrows();
        let mut out = CMatrix::zeros(3 * n, 3 * n);
        for (k, block) in self.b.iter().enumerate() {
            out.view_mut((k / 3 * n, k % 3 * n), (n, n))
                .copy_from(block);
        }
        out
    }

    /// The population of `m_s = 0`.
    fn bright(&self) -> f64 {
        self.b[4].trace().re
    }

    /// The blocks below the diagonal from those above.
    fn mirror(&mut self) {
        for a in 0..3 {
            for b in a + 1..3 {
                self.b[3 * b + a] = self.b[3 * a + b].adjoint();
            }
        }
    }

    /// A free period followed by `exp(−iφ S_z)` (the noise; it commutes).
    #[allow(clippy::needless_range_loop)] // a and b index four arrays
    fn free(&mut self, step: &FreeStep, dims: &[usize], noise: f64) {
        for a in 0..3 {
            for b in a..3 {
                let m = &mut self.b[3 * a + b];
                apply_factors(m, &step.factors[a], dims, true);
                apply_factors(m, &step.factors[b], dims, false);
                if a != b {
                    let phase = step.phase[b] - step.phase[a] + noise * (LEVELS[b] - LEVELS[a]);
                    *m *= C64::from_polar(1.0, phase);
                }
            }
        }
        self.mirror();
    }

    /// A unitary `p` on the electron alone: `ρ_ab ← Σ p_ac ρ_cd p_bd*`.
    fn electron(&mut self, p: &CMatrix) {
        let n = self.b[0].nrows();
        let mut out = vec![CMatrix::zeros(n, n); 9];
        for a in 0..3 {
            for b in a..3 {
                let block = &mut out[3 * a + b];
                for c in 0..3 {
                    for d in 0..3 {
                        let w = p[(a, c)] * p[(b, d)].conj();
                        if w != ZERO {
                            block.zip_apply(&self.b[3 * c + d], |x, y| *x += w * y);
                        }
                    }
                }
            }
        }
        self.b = out;
        self.mirror();
    }

    /// A unitary `u` on the whole centre.
    fn full(&mut self, u: &CMatrix) {
        *self = Blocks::from_full(&cmul(&cmul(u, &self.to_full()), &u.adjoint()));
    }
}

/// A step of a planned sequence.
enum Step {
    /// A free period (index into `Plan::free`).
    Free(usize),
    /// An ideal pulse (index into `Plan::pulses`, 3 × 3).
    Ideal(usize),
    /// A finite pulse: its full unitary (index into `Plan::pulses`) if
    /// computed, its duration (s) and phase (degrees).
    Finite {
        unitary: Option<usize>,
        duration: f64,
        phase: f64,
    },
}

/// A sequence ready to run, each distinct period and pulse computed once.
struct Plan {
    secular: Secular,
    free: Vec<FreeStep>,
    pulses: Vec<CMatrix>,
    steps: Vec<Step>,
    /// The full Hamiltonian (rad/s), for finite pulses.
    h0: Option<CMatrix>,
}

impl Plan {
    /// Apply a step without noise.
    fn apply(&self, state: &mut Blocks, step: &Step) {
        match *step {
            Step::Free(k) => state.free(&self.free[k], &self.secular.dims, 0.0),
            Step::Ideal(k) => state.electron(&self.pulses[k]),
            Step::Finite { unitary, .. } => {
                state.full(&self.pulses[unitary.expect("noiseless plans hold their unitaries")])
            }
        }
    }
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

    /// A centre from JSON (checked).
    pub fn from_json(text: &str) -> Result<Self> {
        let nv: NvCenter = serde_json::from_str(text)
            .map_err(|e| QsoError::invalid("json", format!("not an NV centre: {e}")))?;
        nv.validate()?;
        Ok(nv)
    }

    /// The centre as JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("NV centres serialise")
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
    /// with `X`, `Y` the drive's operators restricted to that transition
    /// (on the electron, 3 × 3).
    fn ideal_pulse(transition: i8, angle: f64, phase: f64) -> CMatrix {
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
        expm_hermitian(&block, angle.to_radians() / 2.0)
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
        let plan = self.plan(events, drive, true)?;
        let mut state = self.initial_blocks();
        for step in &plan.steps {
            plan.apply(&mut state, step);
        }
        Ok(state.bright())
    }

    /// Apply `events` to `rho`.
    pub fn evolve(&self, rho: &CMatrix, events: &[NvEvent], drive: &Microwave) -> Result<CMatrix> {
        let d = self.dimension();
        let plan = self.plan(events, drive, true)?;
        if rho.nrows() != d || rho.ncols() != d {
            return Err(QsoError::DimensionMismatch {
                expected: d,
                actual: rho.nrows(),
            });
        }
        let mut state = Blocks::from_full(rho);
        for step in &plan.steps {
            plan.apply(&mut state, step);
        }
        Ok(state.to_full())
    }

    /// The pumped state in blocks.
    fn initial_blocks(&self) -> Blocks {
        let n = self.dimension() / 3;
        let mut b = vec![CMatrix::zeros(n, n); 9];
        b[4] = CMatrix::identity(n, n) * C64::new(1.0 / n as f64, 0.0);
        Blocks { b }
    }

    /// The free evolution of the secular Hamiltonian in the frame rotating
    /// at `frequency` (Hz), level by level.
    fn secular(&self, frequency: f64) -> Secular {
        let b = self.field;
        let c = |x: f64| C64::new(TWO_PI * x, 0.0);
        let gamma_c = Isotope::C13.gamma() / TWO_PI;
        let mut dims = Vec::new();
        if let Some(n) = &self.nitrogen {
            dims.push(n.levels());
        }
        dims.extend(self.carbons.iter().map(|_| 2));
        let nuclei = LEVELS.map(|m| {
            let mut out = Vec::new();
            if let Some(n) = &self.nitrogen {
                let [iz, _, _, _, _] = spin_ops(n.levels());
                let h = &iz * &iz * c(n.quadrupole) + &iz * c(m * n.parallel - n.gamma() * b);
                out.push(eigh(&h));
            }
            for carbon in &self.carbons {
                let [iz, _, _, ix, _] = spin_ops(2);
                let h =
                    &iz * c(m * carbon.parallel - gamma_c * b) + &ix * c(m * carbon.perpendicular);
                out.push(eigh(&h));
            }
            out
        });
        let energy = LEVELS.map(|m| {
            TWO_PI
                * ((self.zero_field_splitting - frequency) * m * m
                    + self.gyromagnetic_ratio * b * m)
        });
        Secular {
            dims,
            energy,
            nuclei,
        }
    }

    /// The steps of `events` (checked and centred) with each distinct free
    /// period and pulse computed once; noiseless finite pulses as full
    /// unitaries only if `unitaries`.
    fn plan(&self, events: &[NvEvent], drive: &Microwave, unitaries: bool) -> Result<Plan> {
        self.check_drive(drive)?;
        Self::check_events(events)?;
        let events = Self::centred(events, drive)?;
        let secular = self.secular(drive.frequency);
        let h0 = match drive.rabi {
            Some(_) => Some(self.hamiltonian(drive.frequency)?),
            None => None,
        };
        let mut free: Vec<FreeStep> = Vec::new();
        let mut pulses: Vec<CMatrix> = Vec::new();
        let mut seen_free: HashMap<u64, usize> = HashMap::new();
        let mut seen_pulse: HashMap<(u64, u64), usize> = HashMap::new();
        let mut steps = Vec::with_capacity(events.len());
        for event in &events {
            steps.push(match *event {
                NvEvent::Free(t) => {
                    Step::Free(*seen_free.entry(t.to_bits()).or_insert_with(|| {
                        free.push(secular.step(t));
                        free.len() - 1
                    }))
                }
                NvEvent::Pulse { angle, phase } => match (drive.rabi, &h0) {
                    (Some(rabi), Some(h0)) => {
                        let (duration, phase) = finite(angle, phase, rabi);
                        let unitary = if unitaries {
                            let key = (duration.to_bits(), phase.to_bits());
                            Some(*seen_pulse.entry(key).or_insert_with(|| {
                                pulses.push(expm_hermitian(
                                    &(h0 + self.drive(rabi, phase)),
                                    duration,
                                ));
                                pulses.len() - 1
                            }))
                        } else {
                            None
                        };
                        Step::Finite {
                            unitary,
                            duration,
                            phase,
                        }
                    }
                    _ => {
                        let key = (angle.to_bits(), phase.to_bits());
                        Step::Ideal(*seen_pulse.entry(key).or_insert_with(|| {
                            pulses.push(Self::ideal_pulse(drive.transition, angle, phase));
                            pulses.len() - 1
                        }))
                    }
                },
            });
        }
        Ok(Plan {
            secular,
            free,
            pulses,
            steps,
            h0,
        })
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
        noise.validate()?;
        if trajectories == 0 {
            return Err(QsoError::invalid("trajectories", "at least one"));
        }
        let plan = self.plan(events, drive, false)?;
        let d = self.dimension() as f64;
        let n = d / 3.0;
        let nuclear: f64 = plan.secular.dims.iter().map(|&k| (k * k) as f64).sum();
        // Operations (estimated): a free period applies one small factor
        // per nucleus on each side of six blocks; an ideal pulse combines
        // blocks; each of the 16 pieces of a finite pulse is a full
        // exponential (about 12 d³) and two products.
        let per: f64 = plan
            .steps
            .iter()
            .map(|step| match step {
                Step::Free(_) => 8.0 * 12.0 * n * n * (nuclear + 1.0),
                Step::Ideal(_) => 8.0 * 54.0 * n * n,
                Step::Finite { .. } => 16.0 * 18.0 * d * d * d,
            })
            .sum();
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
        let sz = self.embed(0, &Self::electron()[0]);
        let start = self.initial_blocks();
        let root = Rng::seed_from_u64(seed);
        let values: Vec<f64> = crate::parallel::map_tasks(trajectories, |k| {
            let mut rng = root.fork(k as u64);
            let mut path = NoisePath::new(noise, &mut rng);
            let mut state = start.clone();
            for step in &plan.steps {
                match *step {
                    Step::Free(j) => {
                        let free = &plan.free[j];
                        let phase = TWO_PI * path.integral(free.t, &mut rng);
                        state.free(free, &plan.secular.dims, phase);
                    }
                    Step::Ideal(j) => state.electron(&plan.pulses[j]),
                    Step::Finite {
                        duration, phase, ..
                    } => {
                        let (Some(h0), Some(rabi)) = (&plan.h0, drive.rabi) else {
                            unreachable!("finite pulses have a Hamiltonian and a Rabi frequency")
                        };
                        let piece = duration / 16.0;
                        let h = h0 + self.drive(rabi, phase);
                        let mut u = CMatrix::identity(sz.nrows(), sz.nrows());
                        for _ in 0..if piece > 0.0 { 16 } else { 0 } {
                            let mean = path.integral(piece, &mut rng) / piece;
                            let hk = &h + &sz * C64::new(TWO_PI * mean, 0.0);
                            u = cmul(&expm_hermitian(&hk, piece), &u);
                        }
                        state.full(&u);
                    }
                }
            }
            state.bright()
        });
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        Ok((mean, (var / n).sqrt()))
    }
}

#[cfg(test)]
mod tests;
