//! The optical cycle of an NV centre and what a magnetometer measures.
//!
//! [`OpticalNv`] is the seven-level model of Tetienne et al. (New J. Phys.
//! 14, 103033 (2012)): the ground triplet, the excited triplet and one
//! metastable singlet, in a field of any direction. Optical excitation and
//! radiative decay conserve the spin projection `p` along the NV axis; the
//! excited `m_s = ±1` cross to the singlet faster than `m_s = 0`, and the
//! singlet decays preferentially to the ground `m_s = 0`, which polarises
//! the spin and makes the fluorescence spin dependent. Both triplets
//! evolve under
//!
//! ```text
//! H/h = D S_z² + γ_e B·S      (ground: D = 2.87 GHz; excited: 1.42 GHz).
//! ```
//!
//! The model is a Lindblad equation on the seven levels, its jumps in the
//! zero-field basis (`√k |e_p⟩⟨g_p|`, `√k_r |g_p⟩⟨e_p|`, `√k |s⟩⟨e_p|`,
//! `√k |g_p⟩⟨s|`, one per `p`), with pure dephasing `√(2Γ₂*) S_z` and
//! spin-lattice relaxation `Γ₁` in the ground state. Where the levels are
//! split by more than the rates, this is the rate model of Tetienne et al.
//! in the eigenbases, `k_ij = Σ_pq |⟨i|p⟩|² |⟨j|q⟩|² k⁰_pq`; unlike that
//! model it stays continuous at degeneracies and level anticrossings
//! (`B → 0`, the excited-state anticrossing near 51 mT).
//!
//! A microwave of frequency `f` drives the ground state in the
//! rotating-wave approximation: in the ground eigenbasis the two levels
//! other than the one closest to `m_s = 0` rotate at `f`, as the frame of
//! `S_z²` does along the axis in [`super::NvCenter`]; every operator then
//! keeps only its parts within the blocks {level closest to 0} and {the
//! other two}, whose couplings rotate at `f`. This fails within the Rabi
//! frequency and the rates of the ground-state anticrossing at 102.4 mT.
//! The detected signal is the fluorescence, `η k_r` times the excited
//! population.
//!
//! [`TwoLevelOdmr`] is the reduced model of Dréau et al. (Phys. Rev. B 84,
//! 195204 (2011)), with closed forms for the CW-ODMR line. The functions
//! [`ramsey_sensitivity`], [`echo_sensitivity`] and [`best_sensitivity`]
//! turn a readout and a coherence (of the bath of [`super::bath`], for
//! example) into a shot-noise-limited sensitivity, and
//! [`ensemble_lines`] and [`field_from_lines`] connect the four
//! orientations of an ensemble with the field vector.

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::{NV_GYROMAGNETIC_RATIO, NV_ZERO_FIELD_SPLITTING};
use crate::error::{QsoError, Result};
use crate::linalg::dense::{expm, one_norm};
use crate::linalg::{CMatrix, eigh};
use crate::spin::{TWO_PI, spin_ops};

/// Zero-field splitting of the excited triplet (Hz; Tetienne et al. 2012).
pub const NV_EXCITED_SPLITTING: f64 = 1.42e9;

/// Most frequencies of one CW-ODMR spectrum.
pub const MAX_ODMR_POINTS: usize = 1 << 16;

/// Most time steps of one pulsed readout.
pub const MAX_READOUT_STEPS: usize = 1 << 16;

/// Largest `‖L‖₁ t` of one time evolution: beyond it the squarings of the
/// exponential lose accuracy (the steady state is then the answer).
const MAX_EVOLUTION_NORM: f64 = 1e8;

/// Largest `‖L‖₁ Δt` of one step of a pulsed readout.
const MAX_STEP_NORM: f64 = 1e3;

/// The zero-field rates of the seven-level model (1/s).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Photophysics {
    /// `k_r`: spin-conserving radiative decay of the excited triplet.
    pub radiative: f64,
    /// `k₄₇`: from the excited `m_s = 0` to the singlet.
    pub isc_zero: f64,
    /// `k₅₇ = k₆₇`: from each excited `m_s = ±1` to the singlet.
    pub isc_plus_minus: f64,
    /// `k₇₁`: from the singlet to the ground `m_s = 0`.
    pub singlet_zero: f64,
    /// `k₇₂ = k₇₃`: from the singlet to each ground `m_s = ±1`.
    pub singlet_plus_minus: f64,
}

impl Photophysics {
    /// The means of the four centres of Tetienne et al. (2012): `k_r` =
    /// 65.9, `k₄₇` = 7.9, `k₅₇` = 53.3, `k₇₁` = 1.0, `k₇₂` = 0.7 μs⁻¹.
    pub fn tetienne() -> Self {
        Photophysics {
            radiative: 65.9e6,
            isc_zero: 7.9e6,
            isc_plus_minus: 53.3e6,
            singlet_zero: 1.0e6,
            singlet_plus_minus: 0.7e6,
        }
    }

    fn validate(&self) -> Result<()> {
        let all = [
            self.radiative,
            self.isc_zero,
            self.isc_plus_minus,
            self.singlet_zero,
            self.singlet_plus_minus,
        ];
        if all.iter().any(|r| !(r.is_finite() && *r >= 0.0)) || self.radiative == 0.0 {
            return Err(QsoError::invalid(
                "rates",
                "finite rates ≥ 0 and a positive radiative rate",
            ));
        }
        if self.singlet_zero + self.singlet_plus_minus == 0.0 {
            return Err(QsoError::invalid("rates", "the singlet must decay"));
        }
        Ok(())
    }
}

impl Default for Photophysics {
    fn default() -> Self {
        Self::tetienne()
    }
}

/// A Lindblad generator on `n` levels: a Hamiltonian (rad/s) and jump
/// operators (with the square root of their rates).
struct Lindblad {
    n: usize,
    h: CMatrix,
    ops: Vec<CMatrix>,
}

impl Lindblad {
    /// The generator on `vec ρ` (column-major: `ρ_rc` at `r + n c`).
    fn superoperator(&self) -> CMatrix {
        let n = self.n;
        let zero = C64::new(0.0, 0.0);
        let i = C64::new(0.0, 1.0);
        let mut m = CMatrix::zeros(n * n, n * n);
        let at = |r: usize, c: usize| r + n * c;
        // −i(Hρ − ρH), then −½(Kρ + ρK) with K = Σ L†L.
        let mut k = CMatrix::zeros(n, n);
        for l in &self.ops {
            k += l.adjoint() * l;
        }
        let g = self.h.map(|z| -i * z) - k.map(|z| z * 0.5);
        for r in 0..n {
            for c in 0..n {
                for q in 0..n {
                    if g[(r, q)] != zero {
                        m[(at(r, c), at(q, c))] += g[(r, q)];
                    }
                    // ρ(iH − ½K).
                    let right = i * self.h[(q, c)] - k[(q, c)] * 0.5;
                    if right != zero {
                        m[(at(r, c), at(r, q))] += right;
                    }
                }
            }
        }
        // L ρ L†: ρ_ab → L_ra conj(L_cb).
        for l in &self.ops {
            let nz: Vec<(usize, usize, C64)> = (0..n)
                .flat_map(|r| (0..n).map(move |a| (r, a, l[(r, a)])))
                .filter(|&(_, _, x)| x != zero)
                .collect();
            for &(r, a, x) in &nz {
                for &(c, b, y) in &nz {
                    m[(at(r, c), at(a, b))] += x * y.conj();
                }
            }
        }
        m
    }

    /// The state with `L ρ = 0` and unit trace.
    fn steady_state(&self) -> Result<CMatrix> {
        let n = self.n;
        let mut m = self.superoperator();
        let mut rhs = nalgebra::DVector::from_element(n * n, C64::new(0.0, 0.0));
        for c in 0..n * n {
            m[(0, c)] = C64::new(0.0, 0.0);
        }
        for k in 0..n {
            m[(0, k + n * k)] = C64::new(1.0, 0.0);
        }
        rhs[0] = C64::new(1.0, 0.0);
        let refuse = || QsoError::invalid("model", "the optical cycle has no unique steady state");
        let x = m.full_piv_lu().solve(&rhs).ok_or_else(refuse)?;
        let rho = CMatrix::from_fn(n, n, |r, c| x[r + n * c]);
        // A singular system can slip through the pivoting: the result must
        // be a state.
        if rho.iter().any(|z| !z.is_finite()) || (0..n).any(|k| rho[(k, k)].re < -1e-9) {
            return Err(refuse());
        }
        Ok(rho)
    }
}

/// `|vector component|²` of each eigenvector (column) of `v` on each basis
/// state: `w[i][p] = |⟨p|i⟩|²`.
fn weights(v: &CMatrix) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|p| v[(p, i)].norm_sqr()))
}

/// The NV electron spin in its optical cycle (see the module docs).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpticalNv {
    /// The field (T) in the frame of the centre: z along the NV axis.
    pub field: [f64; 3],
    pub zero_field_splitting: f64,
    pub excited_splitting: f64,
    pub gyromagnetic_ratio: f64,
    pub rates: Photophysics,
    /// `Γ₂*`: decay rate of the ground coherences `0 ↔ ±1` (1/s).
    pub dephasing: f64,
    /// `Γ₁`: spin-lattice rate between the ground `m_s = 0` and each
    /// `m_s = ±1`, both ways (1/s).
    pub spin_lattice: f64,
    /// Fraction of the emitted photons that is detected.
    pub collection: f64,
}

/// A CW-ODMR line and the sensitivity it gives.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CwOdmr {
    /// The frequencies (Hz) and detected photon rates (1/s).
    pub frequencies: Vec<f64>,
    pub rates: Vec<f64>,
    /// The photon rate without the microwave.
    pub off_resonance: f64,
    /// Depth of the dip relative to the off-resonance rate.
    pub contrast: f64,
    /// Full width at half depth (Hz); NaN when the span does not hold
    /// both half-depth points.
    pub linewidth: f64,
    /// Steepest slope of the rate (1/s per Hz).
    pub slope: f64,
    /// How fast the line moves with a field along the axis (Hz/T): `γ_e`
    /// along the axis, `γ_e (⟨j|S_z|j⟩ − ⟨0|S_z|0⟩)` in general.
    pub shift: f64,
    /// `√R/(|shift| max|∂I/∂ν|)` (T/√Hz), the shot-noise limit for a field
    /// along the axis with the off-resonance rate `R`, as in Dréau et al.
    pub sensitivity: f64,
}

/// The fluorescence of a pulsed readout from the two spin states.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PulsedReadout {
    /// Times since the laser was switched on (s).
    pub times: Vec<f64>,
    /// Detected photon rate (1/s) from the ground state the laser leaves
    /// behind.
    pub bright: Vec<f64>,
    /// From the same state after an ideal π pulse on the transition.
    pub dark: Vec<f64>,
    /// The counting window `[0, window]` that distinguishes them best:
    /// largest `|N₀ − N₁|/√(N₀ + N₁)`.
    pub window: f64,
    /// Photons detected in the window from each state.
    pub photons_bright: f64,
    pub photons_dark: f64,
}

impl PulsedReadout {
    /// The readout parameter `C = [1 + 2(N₀ + N₁)/(N₀ − N₁)²]^{−1/2}`
    /// (Taylor et al., Nature Phys. 4, 810 (2008)): 1 for a perfect
    /// single-shot readout.
    pub fn readout_contrast(&self) -> f64 {
        readout_contrast(self.photons_bright, self.photons_dark).unwrap_or(0.0)
    }
}

/// `C = [1 + 2(a + b)/(a − b)²]^{−1/2}` for mean photon numbers `a`, `b`
/// per readout (0 when they are equal).
pub fn readout_contrast(a: f64, b: f64) -> Result<f64> {
    if !(a.is_finite() && b.is_finite() && a >= 0.0 && b >= 0.0) {
        return Err(QsoError::invalid("photons", "finite photon numbers ≥ 0"));
    }
    let d = a - b;
    if d == 0.0 {
        return Ok(0.0);
    }
    Ok((1.0 + 2.0 * (a + b) / (d * d)).powf(-0.5))
}

/// The eigenbases of both triplets and the ground levels closest to
/// `m_s = 0`, `+1` and `−1`.
struct Levels {
    eg: Vec<f64>,
    vg: CMatrix,
    ee: Vec<f64>,
    ve: CMatrix,
    zero: usize,
    plus: usize,
    minus: usize,
}

/// `|a⟩⟨b|`.
fn outer(a: &[C64], b: &[C64]) -> CMatrix {
    CMatrix::from_fn(a.len(), b.len(), |r, c| a[r] * b[c].conj())
}

/// `⟨i|op|j⟩` for eigenvectors (columns) of `v`.
fn element(v: &CMatrix, i: usize, op: &CMatrix, j: usize) -> C64 {
    let mut z = C64::new(0.0, 0.0);
    for p in 0..3 {
        for q in 0..3 {
            z += v[(p, i)].conj() * op[(p, q)] * v[(q, j)];
        }
    }
    z
}

impl OpticalNv {
    /// A centre in a field `field` (T, frame of the centre) with the rates
    /// of Tetienne et al., `Γ₂* = 2·10⁵ s⁻¹` and `Γ₁ = 10³ s⁻¹` (Dréau et
    /// al.) and 1 % of the photons detected.
    pub fn new(field: [f64; 3]) -> Self {
        OpticalNv {
            field,
            zero_field_splitting: NV_ZERO_FIELD_SPLITTING,
            excited_splitting: NV_EXCITED_SPLITTING,
            gyromagnetic_ratio: NV_GYROMAGNETIC_RATIO,
            rates: Photophysics::tetienne(),
            dephasing: 2e5,
            spin_lattice: 1e3,
            collection: 0.01,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.field.iter().any(|b| !b.is_finite()) {
            return Err(QsoError::invalid("field", "the field must be finite"));
        }
        for (name, x) in [
            ("zero_field_splitting", self.zero_field_splitting),
            ("excited_splitting", self.excited_splitting),
            ("gyromagnetic_ratio", self.gyromagnetic_ratio),
        ] {
            if !(x.is_finite() && x > 0.0) {
                return Err(QsoError::invalid(name, "a positive finite value"));
            }
        }
        for (name, x) in [
            ("dephasing", self.dephasing),
            ("spin_lattice", self.spin_lattice),
        ] {
            if !(x.is_finite() && x >= 0.0) {
                return Err(QsoError::invalid(name, "a finite rate ≥ 0"));
            }
        }
        if !(self.collection > 0.0 && self.collection <= 1.0) {
            return Err(QsoError::invalid("collection", "a fraction in (0, 1]"));
        }
        self.rates.validate()
    }

    fn triplet(&self, d: f64) -> CMatrix {
        let [sz, _, _, sx, sy] = spin_ops(3);
        let g = self.gyromagnetic_ratio;
        let [bx, by, bz] = self.field;
        let c = |x: f64| C64::new(x, 0.0);
        &sz * &sz * c(d) + sx * c(g * bx) + sy * c(g * by) + sz * c(g * bz)
    }

    /// `H/h` of the ground triplet (Hz), basis `m_s = +1, 0, −1` along the
    /// axis.
    pub fn ground_hamiltonian(&self) -> CMatrix {
        self.triplet(self.zero_field_splitting)
    }

    /// `H/h` of the excited triplet (Hz).
    pub fn excited_hamiltonian(&self) -> CMatrix {
        self.triplet(self.excited_splitting)
    }

    /// Energies (Hz, ascending) and eigenvectors of the ground triplet.
    pub fn ground_levels(&self) -> Result<(Vec<f64>, CMatrix)> {
        self.validate()?;
        Ok(eigh(&self.ground_hamiltonian()))
    }

    /// Energies and eigenvectors of the excited triplet.
    pub fn excited_levels(&self) -> Result<(Vec<f64>, CMatrix)> {
        self.validate()?;
        Ok(eigh(&self.excited_hamiltonian()))
    }

    fn levels(&self) -> Result<Levels> {
        self.validate()?;
        let (eg, vg) = eigh(&self.ground_hamiltonian());
        let (ee, ve) = eigh(&self.excited_hamiltonian());
        let w = weights(&vg);
        // The level with the most m_s = 0; of the others, the one with more
        // +1 than −1 (the higher one if they tie, as across the axis).
        let zero = (0..3)
            .max_by(|&a, &b| w[a][1].total_cmp(&w[b][1]))
            .expect("three levels");
        let mut rest: Vec<usize> = (0..3).filter(|&i| i != zero).collect();
        let lean = |i: usize| w[i][0] - w[i][2];
        if (lean(rest[0]) - lean(rest[1])).abs() > 1e-12 {
            rest.sort_by(|&a, &b| lean(a).total_cmp(&lean(b)));
        } else {
            rest.sort_by(|&a, &b| eg[a].total_cmp(&eg[b]));
        }
        Ok(Levels {
            eg,
            vg,
            ee,
            ve,
            zero,
            plus: rest[1],
            minus: rest[0],
        })
    }

    /// The two ESR lines: frequencies from the level closest to `m_s = 0`
    /// to those closest to `m_s = −1` and `+1` (Hz, signed: negative when
    /// the level lies below), with their strengths `|⟨j|S_x|0⟩|²` for a
    /// microwave along x (½ each along the axis).
    pub fn lines(&self) -> Result<[(f64, f64); 2]> {
        let l = self.levels()?;
        let [_, _, _, sx, _] = spin_ops(3);
        Ok([l.minus, l.plus].map(|j| {
            (
                l.eg[j] - l.eg[l.zero],
                element(&l.vg, j, &sx, l.zero).norm_sqr(),
            )
        }))
    }

    /// The Lindblad model (see the module docs) with optical excitation at
    /// `pump` (1/s per spin-conserving transition) and, if given, a
    /// microwave of frequency `drive.0` (Hz, signed as
    /// [`OpticalNv::lines`]) and Rabi frequency `drive.1` (Hz, as in
    /// [`super::NvCenter`]: `2π(Ω/2)` couples `0` and `±1` along the axis).
    fn model(&self, pump: f64, drive: Option<(f64, f64)>) -> Result<(Lindblad, Levels)> {
        if !(pump.is_finite() && pump >= 0.0) {
            return Err(QsoError::invalid("pump", "a finite rate ≥ 0"));
        }
        if let Some((f, rabi)) = drive
            && !(f.is_finite() && rabi.is_finite() && rabi >= 0.0)
        {
            return Err(QsoError::invalid(
                "drive",
                "a finite frequency and a Rabi frequency ≥ 0",
            ));
        }
        let l = self.levels()?;
        let r = &self.rates;
        let zero7 = C64::new(0.0, 0.0);
        // Zero-field states p = +1, 0, −1 in the eigenbases (7 levels:
        // ground 0..3, excited 3..6, singlet 6).
        let ket = |v: &CMatrix, offset: usize, p: usize| -> Vec<C64> {
            let mut k = vec![zero7; 7];
            for i in 0..3 {
                k[offset + i] = v[(p, i)].conj();
            }
            k
        };
        let g: [Vec<C64>; 3] = std::array::from_fn(|p| ket(&l.vg, 0, p));
        let e: [Vec<C64>; 3] = std::array::from_fn(|p| ket(&l.ve, 3, p));
        let mut singlet = vec![zero7; 7];
        singlet[6] = C64::new(1.0, 0.0);
        // With a microwave the ground state splits into the blocks {zero}
        // and {the others}, whose couplings rotate at f and are dropped.
        let blocks: Vec<Vec<bool>> = match drive {
            Some(_) => vec![
                (0..7).map(|k| k == l.zero).collect(),
                (0..7).map(|k| k < 3 && k != l.zero).collect(),
            ],
            None => vec![(0..7).map(|k| k < 3).collect()],
        };
        let split = |k: &[C64]| -> Vec<Vec<C64>> {
            blocks
                .iter()
                .map(|mask| {
                    k.iter()
                        .zip(mask)
                        .map(|(&z, &m)| if m { z } else { zero7 })
                        .collect::<Vec<C64>>()
                })
                .filter(|part| part.iter().any(|z| z.norm_sqr() > 0.0))
                .collect()
        };
        let mut ops: Vec<CMatrix> = Vec::new();
        let mut push = |op: CMatrix, rate: f64| {
            if rate > 0.0 {
                ops.push(op * C64::new(rate.sqrt(), 0.0));
            }
        };
        // Per spin projection p = +1, 0, −1.
        let isc = [r.isc_plus_minus, r.isc_zero, r.isc_plus_minus];
        let back = [r.singlet_plus_minus, r.singlet_zero, r.singlet_plus_minus];
        for p in 0..3 {
            for gp in split(&g[p]) {
                push(outer(&e[p], &gp), pump);
                push(outer(&gp, &e[p]), r.radiative);
                push(outer(&gp, &singlet), back[p]);
            }
            push(outer(&singlet, &e[p]), isc[p]);
        }
        for (p, q) in [(1, 0), (0, 1), (1, 2), (2, 1)] {
            for gq in split(&g[q]) {
                for gp in split(&g[p]) {
                    push(outer(&gq, &gp), self.spin_lattice);
                }
            }
        }
        // Pure dephasing by S_z (its parts within the blocks).
        let [sz, _, _, sx, _] = spin_ops(3);
        let mut z = CMatrix::zeros(7, 7);
        for i in 0..3 {
            for j in 0..3 {
                let same = blocks.iter().any(|m| m[i] && m[j]);
                if same {
                    z[(i, j)] = element(&l.vg, i, &sz, j);
                }
            }
        }
        push(z, 2.0 * self.dephasing);
        // Hamiltonians (rad/s): the excited triplet in the laboratory frame
        // (from its mean), the ground one too without a microwave, else in
        // the rotating frame.
        let mut h = CMatrix::zeros(7, 7);
        let mean = |e: &[f64]| e.iter().sum::<f64>() / e.len() as f64;
        for j in 0..3 {
            h[(3 + j, 3 + j)] = C64::new(TWO_PI * (l.ee[j] - mean(&l.ee)), 0.0);
        }
        match drive {
            None => {
                for i in 0..3 {
                    h[(i, i)] = C64::new(TWO_PI * (l.eg[i] - mean(&l.eg)), 0.0);
                }
            }
            Some((frequency, rabi)) => {
                let root2 = std::f64::consts::SQRT_2;
                for j in (0..3).filter(|&j| j != l.zero) {
                    // Both other levels rotate at the (signed) frequency,
                    // as the frame of S_z² in 1.37 does along the axis.
                    let gap = l.eg[j] - l.eg[l.zero];
                    h[(j, j)] = C64::new(TWO_PI * (gap - frequency), 0.0);
                    let c = element(&l.vg, j, &sx, l.zero) * (TWO_PI * rabi / 2.0 * root2);
                    h[(j, l.zero)] = c;
                    h[(l.zero, j)] = c.conj();
                }
            }
        }
        Ok((Lindblad { n: 7, h, ops }, l))
    }

    /// The steady state (7 × 7: ground levels ascending, excited levels
    /// ascending, singlet; ground coherences in the rotating frame when
    /// driven) under `pump` and an optional microwave. It needs the laser
    /// or spin-lattice relaxation to be unique.
    pub fn steady_state(&self, pump: f64, drive: Option<(f64, f64)>) -> Result<CMatrix> {
        if pump == 0.0 && self.spin_lattice == 0.0 {
            return Err(QsoError::invalid(
                "pump",
                "without the laser and spin-lattice relaxation the steady state is not unique",
            ));
        }
        self.model(pump, drive)?.0.steady_state()
    }

    /// The state after `duration` (s) from `rho` (7 × 7, as
    /// [`OpticalNv::steady_state`]) under `pump` and an optional microwave.
    /// Durations beyond about 10⁸ of the fastest rates (10 ms) are refused:
    /// use the steady state.
    pub fn evolve(
        &self,
        rho: &CMatrix,
        duration: f64,
        pump: f64,
        drive: Option<(f64, f64)>,
    ) -> Result<CMatrix> {
        if rho.nrows() != 7 || rho.ncols() != 7 {
            return Err(QsoError::DimensionMismatch {
                expected: 7,
                actual: rho.nrows(),
            });
        }
        if !(duration.is_finite() && duration >= 0.0) {
            return Err(QsoError::invalid("duration", "a finite duration ≥ 0"));
        }
        let m = self.model(pump, drive)?.0.superoperator();
        if one_norm(&m) * duration > MAX_EVOLUTION_NORM {
            return Err(QsoError::invalid(
                "duration",
                "too long for a time evolution: use the steady state",
            ));
        }
        let x = nalgebra::DVector::from_iterator(49, (0..49).map(|k| rho[(k % 7, k / 7)]));
        let x = expm(&(m * C64::new(duration, 0.0))) * x;
        Ok(CMatrix::from_fn(7, 7, |r, c| x[r + 7 * c]))
    }

    /// Detected photons per second in a state of the seven levels.
    pub fn fluorescence(&self, rho: &CMatrix) -> f64 {
        let excited: f64 = (3..6).map(|j| rho[(j, j)].re).sum();
        self.collection * self.rates.radiative * excited
    }

    /// The photon rate under `pump` at each microwave frequency (Hz).
    pub fn cw_odmr(&self, frequencies: &[f64], rabi: f64, pump: f64) -> Result<Vec<f64>> {
        if frequencies.len() > MAX_ODMR_POINTS {
            return Err(QsoError::invalid(
                "frequencies",
                format!("at most {MAX_ODMR_POINTS}"),
            ));
        }
        crate::parallel::map_tasks(frequencies.len(), |k| {
            Ok(self.fluorescence(&self.steady_state(pump, Some((frequencies[k], rabi)))?))
        })
        .into_iter()
        .collect()
    }

    /// The CW-ODMR line around `centre` (Hz) over `span` at `points`
    /// frequencies, its contrast and width, and the sensitivity it gives
    /// to a field along the axis (through the line nearest `centre`).
    pub fn cw_line(
        &self,
        centre: f64,
        span: f64,
        points: usize,
        rabi: f64,
        pump: f64,
    ) -> Result<CwOdmr> {
        if !(span.is_finite() && span > 0.0 && centre.is_finite())
            || !(5..=MAX_ODMR_POINTS).contains(&points)
        {
            return Err(QsoError::invalid(
                "span",
                format!("a finite centre, a positive span and 5 to {MAX_ODMR_POINTS} points"),
            ));
        }
        if rabi.is_nan() || rabi <= 0.0 {
            return Err(QsoError::invalid("rabi", "a positive Rabi frequency"));
        }
        let l = self.levels()?;
        let [sz, ..] = spin_ops(3);
        let j = [l.minus, l.plus]
            .into_iter()
            .min_by(|&a, &b| {
                let d = |j: usize| (l.eg[j] - l.eg[l.zero] - centre).abs();
                d(a).total_cmp(&d(b))
            })
            .expect("two lines");
        let shift = self.gyromagnetic_ratio
            * (element(&l.vg, j, &sz, j).re - element(&l.vg, l.zero, &sz, l.zero).re);
        let frequencies: Vec<f64> = (0..points)
            .map(|k| centre - span / 2.0 + span * k as f64 / (points - 1) as f64)
            .collect();
        let rates = self.cw_odmr(&frequencies, rabi, pump)?;
        let off = self.fluorescence(&self.steady_state(pump, None)?);
        Ok(line_shape(frequencies, rates, off, shift))
    }

    /// The ground state (3 × 3, eigenbasis) after the laser has been on for
    /// `duration` (from the unpolarised ground state; ∞ for the steady
    /// state) and the excited levels and the singlet have decayed into it
    /// in the dark.
    fn polarised(&self, pump: f64, duration: f64) -> Result<CMatrix> {
        if duration.is_nan() || duration < 0.0 {
            return Err(QsoError::invalid("duration", "a duration ≥ 0"));
        }
        let x = if duration.is_infinite() {
            self.steady_state(pump, None)?
        } else {
            let mut start = CMatrix::zeros(7, 7);
            for k in 0..3 {
                start[(k, k)] = C64::new(1.0 / 3.0, 0.0);
            }
            self.evolve(&start, duration, pump, None)?
        };
        // In the dark everything outside the ground block decays into it:
        // the ground block gains M_go (−M_oo)⁻¹ x_o, the integral of the
        // inflow (the ground's own slow dynamics afterwards left out).
        let mut dark = self.clone();
        dark.spin_lattice = 0.0;
        let m = dark.model(0.0, None)?.0.superoperator();
        let ground: Vec<usize> = (0..49).filter(|&k| k % 7 < 3 && k / 7 < 3).collect();
        let other: Vec<usize> = (0..49).filter(|k| !ground.contains(k)).collect();
        let moo = CMatrix::from_fn(other.len(), other.len(), |a, b| -m[(other[a], other[b])]);
        let xo =
            nalgebra::DVector::from_iterator(other.len(), other.iter().map(|&k| x[(k % 7, k / 7)]));
        let integral = moo.full_piv_lu().solve(&xo).ok_or_else(|| {
            QsoError::invalid("rates", "the excited levels and the singlet must decay")
        })?;
        let mut out = CMatrix::zeros(3, 3);
        for &kg in &ground {
            let mut z = x[(kg % 7, kg / 7)];
            for (b, &ko) in other.iter().enumerate() {
                z += m[(kg, ko)] * integral[b];
            }
            out[(kg % 7, kg / 7)] = z;
        }
        Ok(out)
    }

    /// The population of the level closest to `m_s = 0` after the laser
    /// has been on for `duration` (from the unpolarised ground state; ∞ for
    /// the steady state) and the excited levels and the singlet have
    /// decayed.
    pub fn polarisation(&self, pump: f64, duration: f64) -> Result<f64> {
        let ground = self.polarised(pump, duration)?;
        let zero = self.levels()?.zero;
        Ok(ground[(zero, zero)].re)
    }

    /// The fluorescence after the laser is switched on at `pump`, at
    /// `steps` times up to `duration`, with the best counting window. Both
    /// start from the ground state the same laser leaves behind (its steady
    /// state after the excited levels and the singlet have decayed in the
    /// dark): `bright` as it is, `dark` after an ideal π pulse that swaps
    /// the levels closest to `m_s = 0` and to `m_s = transition`.
    pub fn pulsed_readout(
        &self,
        pump: f64,
        duration: f64,
        steps: usize,
        transition: i8,
    ) -> Result<PulsedReadout> {
        if !(duration.is_finite() && duration > 0.0) || !(1..=MAX_READOUT_STEPS).contains(&steps) {
            return Err(QsoError::invalid(
                "duration",
                format!("a positive duration and 1 to {MAX_READOUT_STEPS} steps"),
            ));
        }
        if transition != 1 && transition != -1 {
            return Err(QsoError::invalid("transition", "−1 or +1"));
        }
        let (model, l) = self.model(pump, None)?;
        let other = if transition < 0 { l.minus } else { l.plus };
        let ground = self.polarised(pump, f64::INFINITY)?;
        // The ideal π pulse swaps the two levels.
        let mut swap = CMatrix::identity(3, 3);
        for (a, b) in [
            (l.zero, l.zero),
            (other, other),
            (l.zero, other),
            (other, l.zero),
        ] {
            swap[(a, b)] = if a == b {
                C64::new(0.0, 0.0)
            } else {
                C64::new(1.0, 0.0)
            };
        }
        let swapped = &swap * &ground * swap.adjoint();
        let dt = duration / steps as f64;
        let m = model.superoperator();
        if one_norm(&m) * dt > MAX_STEP_NORM {
            return Err(QsoError::invalid(
                "steps",
                "steps too long for the fastest rates and splittings: use more",
            ));
        }
        let step = expm(&(m * C64::new(dt, 0.0)));
        let trace = |start: &CMatrix| -> Vec<f64> {
            let mut x = nalgebra::DVector::from_element(49, C64::new(0.0, 0.0));
            for r in 0..3 {
                for c in 0..3 {
                    x[r + 7 * c] = start[(r, c)];
                }
            }
            let mut out = Vec::with_capacity(steps + 1);
            for k in 0..=steps {
                if k > 0 {
                    x = &step * &x;
                }
                let excited: f64 = (3..6).map(|j| x[j + 7 * j].re).sum();
                out.push(self.collection * self.rates.radiative * excited);
            }
            out
        };
        let (bright, dark) = (trace(&ground), trace(&swapped));
        let times: Vec<f64> = (0..=steps).map(|k| k as f64 * dt).collect();
        // Photons in [0, t] by the trapezoidal rule; the best window.
        let (mut n0, mut n1) = (0.0, 0.0);
        let mut best = (0.0, 0.0, 0.0, f64::NEG_INFINITY);
        for k in 1..=steps {
            n0 += 0.5 * dt * (bright[k - 1] + bright[k]);
            n1 += 0.5 * dt * (dark[k - 1] + dark[k]);
            let score = (n0 - n1).abs() / (n0 + n1).sqrt();
            if score > best.3 {
                best = (times[k], n0, n1, score);
            }
        }
        Ok(PulsedReadout {
            times,
            bright,
            dark,
            window: best.0,
            photons_bright: best.1,
            photons_dark: best.2,
        })
    }
}

/// Contrast, width, slope and sensitivity of a computed line.
fn line_shape(frequencies: Vec<f64>, rates: Vec<f64>, off: f64, shift: f64) -> CwOdmr {
    let (k, &low) = rates
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .expect("at least 5 points");
    let half = off - (off - low) / 2.0;
    let crossing = |range: Box<dyn Iterator<Item = usize>>| -> f64 {
        let mut prev: Option<usize> = None;
        for j in range {
            if let Some(i) = prev
                && (rates[i] - half) * (rates[j] - half) <= 0.0
                && rates[i] != rates[j]
            {
                let f = (half - rates[i]) / (rates[j] - rates[i]);
                return frequencies[i] + f * (frequencies[j] - frequencies[i]);
            }
            prev = Some(j);
        }
        f64::NAN
    };
    let left = crossing(Box::new((0..=k).rev()));
    let right = crossing(Box::new(k..rates.len()));
    let slope = frequencies
        .windows(2)
        .zip(rates.windows(2))
        .map(|(f, r)| ((r[1] - r[0]) / (f[1] - f[0])).abs())
        .fold(0.0, f64::max);
    CwOdmr {
        contrast: (off - low) / off,
        linewidth: right - left,
        slope,
        shift,
        sensitivity: off.sqrt() / (shift.abs() * slope),
        off_resonance: off,
        frequencies,
        rates,
    }
}

/// The reduced model of Dréau et al. (Phys. Rev. B 84, 195204 (2011)): the
/// ground qubit `{0, 1}` with optical polarisation `Γ_p` (1 → 0),
/// spin-lattice relaxation `Γ₁`, coherence decay `Γ₂ = Γ₂* + Γ_c` and a
/// photon rate `α σ₀₀ + β σ₁₁` (rates in 1/s, the Rabi frequency `Ω_R` in
/// rad/s as in the paper).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TwoLevelOdmr {
    pub polarisation: f64,
    pub coherence: f64,
    pub spin_lattice: f64,
    pub dephasing: f64,
    pub bright: f64,
    pub dark: f64,
}

impl TwoLevelOdmr {
    /// Check the rates: finite, ≥ 0, `Γ_p + Γ₁ > 0` and `Γ₂ ≥ Γ_p/2 + Γ₁`
    /// (the jumps alone decay the coherence that fast).
    pub fn validate(&self) -> Result<()> {
        let all = [
            self.polarisation,
            self.coherence,
            self.spin_lattice,
            self.dephasing,
            self.bright,
            self.dark,
        ];
        if all.iter().any(|x| !(x.is_finite() && *x >= 0.0)) {
            return Err(QsoError::invalid("model", "finite rates ≥ 0"));
        }
        if self.polarisation + self.spin_lattice == 0.0 {
            return Err(QsoError::invalid("model", "Γ_p + Γ₁ must be positive"));
        }
        if self.dephasing + self.coherence < self.polarisation / 2.0 + self.spin_lattice {
            return Err(QsoError::invalid(
                "model",
                "Γ₂* + Γ_c must be at least Γ_p/2 + Γ₁",
            ));
        }
        Ok(())
    }

    /// `σ₁₁` in the steady state at detuning `Δ = ω₀ − ω_m` (rad/s), solved
    /// as a Lindblad equation (not from the closed form).
    pub fn population(&self, detuning: f64, rabi: f64) -> Result<f64> {
        self.validate()?;
        if !(detuning.is_finite() && rabi.is_finite()) {
            return Err(QsoError::invalid(
                "detuning",
                "a finite detuning and Rabi frequency",
            ));
        }
        let mut h = CMatrix::zeros(2, 2);
        h[(1, 1)] = C64::new(detuning, 0.0);
        h[(0, 1)] = C64::new(rabi / 2.0, 0.0);
        h[(1, 0)] = C64::new(rabi / 2.0, 0.0);
        // Jumps decay the coherence by Γ_p/2 + Γ₁; pure dephasing makes up
        // the rest of Γ₂.
        let extra = self.dephasing + self.coherence - self.polarisation / 2.0 - self.spin_lattice;
        let op = |r: usize, c: usize, rate: f64| {
            let mut m = CMatrix::zeros(2, 2);
            m[(r, c)] = C64::new(rate.sqrt(), 0.0);
            m
        };
        let model = Lindblad {
            n: 2,
            h,
            ops: vec![
                op(0, 1, self.polarisation + self.spin_lattice),
                op(1, 0, self.spin_lattice),
                op(1, 1, 2.0 * extra),
            ],
        };
        Ok(model.steady_state()?[(1, 1)].re)
    }

    /// The photon rate at detuning `Δ` (rad/s).
    pub fn rate(&self, detuning: f64, rabi: f64) -> Result<f64> {
        let p = self.population(detuning, rabi)?;
        Ok(self.bright * (1.0 - p) + self.dark * p)
    }

    /// The line at `points` detunings over `span` (rad/s), with its
    /// contrast, width (Hz) and sensitivity (T/√Hz for the electron's
    /// `γ_e`).
    pub fn line(&self, span: f64, points: usize, rabi: f64) -> Result<CwOdmr> {
        if !(span.is_finite() && span > 0.0) || !(5..=MAX_ODMR_POINTS).contains(&points) {
            return Err(QsoError::invalid(
                "span",
                format!("a positive span and 5 to {MAX_ODMR_POINTS} points"),
            ));
        }
        let detunings: Vec<f64> = (0..points)
            .map(|k| -span / 2.0 + span * k as f64 / (points - 1) as f64)
            .collect();
        let rates = detunings
            .iter()
            .map(|&d| self.rate(d, rabi))
            .collect::<Result<Vec<_>>>()?;
        // Off resonance: no microwave at all.
        let off = self.rate(0.0, 0.0)?;
        let frequencies = detunings.iter().map(|d| d / TWO_PI).collect();
        Ok(line_shape(frequencies, rates, off, NV_GYROMAGNETIC_RATIO))
    }
}

fn check_readout(c: f64) -> Result<()> {
    if !(c > 0.0 && c <= 1.0) {
        return Err(QsoError::invalid("readout", "a readout contrast in (0, 1]"));
    }
    Ok(())
}

/// The shot-noise-limited sensitivity (T/√Hz) of Ramsey magnetometry of a
/// static field: free precession `tau`, coherence `|L(τ)| ≤ 1`, readout
/// contrast `C` ([`PulsedReadout::readout_contrast`]) and `overhead` per
/// repetition (initialisation and readout, s): `√(τ + t_o)/(2π γ_e τ C |L|)`
/// (∞ for `L = 0`).
pub fn ramsey_sensitivity(tau: f64, coherence: f64, readout: f64, overhead: f64) -> Result<f64> {
    check_readout(readout)?;
    if !(tau > 0.0 && tau.is_finite() && overhead >= 0.0 && overhead.is_finite()) {
        return Err(QsoError::invalid("tau", "τ > 0 and an overhead ≥ 0"));
    }
    let l = coherence.abs();
    if !(l.is_finite() && l <= 1.0 + 1e-9) {
        return Err(QsoError::invalid(
            "coherence",
            "a coherence of modulus at most 1",
        ));
    }
    if l == 0.0 {
        return Ok(f64::INFINITY);
    }
    Ok((tau + overhead).sqrt() / (TWO_PI * NV_GYROMAGNETIC_RATIO * tau * readout * l))
}

/// The same for a Hahn echo of total duration `total` and an AC field of
/// period `total` in its best phase: the phase is `2/π` of a static
/// field's, so the sensitivity is `π/2` times the Ramsey expression.
pub fn echo_sensitivity(total: f64, coherence: f64, readout: f64, overhead: f64) -> Result<f64> {
    Ok(std::f64::consts::FRAC_PI_2 * ramsey_sensitivity(total, coherence, readout, overhead)?)
}

/// The best of the sensitivities at the given times and coherences
/// (`echo`: Hahn echo, else Ramsey): `(time, sensitivity)`.
pub fn best_sensitivity(
    times: &[f64],
    coherences: &[f64],
    readout: f64,
    overhead: f64,
    echo: bool,
) -> Result<(f64, f64)> {
    if times.len() != coherences.len() || times.is_empty() {
        return Err(QsoError::invalid(
            "times",
            "as many times as coherences, at least one",
        ));
    }
    let mut best = (f64::NAN, f64::INFINITY);
    for (&t, &l) in times.iter().zip(coherences) {
        let eta = if echo {
            echo_sensitivity(t, l, readout, overhead)?
        } else {
            ramsey_sensitivity(t, l, readout, overhead)?
        };
        if eta < best.1 {
            best = (t, eta);
        }
    }
    if best.1.is_infinite() {
        return Err(QsoError::invalid("coherences", "no time with a coherence"));
    }
    Ok(best)
}

/// The NV axes of the four orientations in the cubic frame of the crystal.
pub fn nv_orientations() -> [[f64; 3]; 4] {
    let s = 1.0 / 3f64.sqrt();
    [[s, s, s], [s, -s, -s], [-s, s, -s], [-s, -s, s]]
}

/// The field's components along and across the axis `n` (T).
fn split(field: [f64; 3], n: [f64; 3]) -> (f64, f64) {
    let along = field[0] * n[0] + field[1] * n[1] + field[2] * n[2];
    let total: f64 = field.iter().map(|b| b * b).sum();
    (along, (total - along * along).max(0.0).sqrt())
}

fn check_constants(d: f64, gamma: f64) -> Result<()> {
    if !(d.is_finite() && d > 0.0 && gamma.is_finite() && gamma > 0.0) {
        return Err(QsoError::invalid("constants", "finite D > 0 and γ > 0"));
    }
    Ok(())
}

/// The two ESR lines of each orientation (Hz, ascending, signed from the
/// level closest to `m_s = 0`) for a field `field` (T) in the cubic frame.
pub fn ensemble_lines(
    field: [f64; 3],
    zero_field_splitting: f64,
    gamma: f64,
) -> Result<[[f64; 2]; 4]> {
    check_constants(zero_field_splitting, gamma)?;
    let mut out = [[0.0; 2]; 4];
    for (k, n) in nv_orientations().into_iter().enumerate() {
        let (along, across) = split(field, n);
        let mut nv = OpticalNv::new([across, 0.0, along]);
        nv.zero_field_splitting = zero_field_splitting;
        nv.gyromagnetic_ratio = gamma;
        let lines = nv.lines()?;
        let (a, b) = (lines[0].0, lines[1].0);
        out[k] = [a.min(b), a.max(b)];
    }
    Ok(out)
}

/// The field along and across the axis (`|B_∥|`, `B_⊥`, T) from the two
/// lines of one orientation (signed from the level closest to `m_s = 0`,
/// in either order), exactly: with `λ₀ = (2D − f₁ − f₂)/3` the levels are
/// `λ₀`, `λ₀ + f₁`, `λ₀ + f₂`, and the invariants of `D S_z² + γB·S` give
/// `γ²B² = D² − e₂` and `γ²B_⊥² = −e₃/D`.
pub fn axial_and_transverse(
    lines: [f64; 2],
    zero_field_splitting: f64,
    gamma: f64,
) -> Result<(f64, f64)> {
    check_constants(zero_field_splitting, gamma)?;
    if lines.iter().any(|f| !f.is_finite()) {
        return Err(QsoError::invalid("lines", "finite lines"));
    }
    let d = zero_field_splitting;
    let l0 = (2.0 * d - lines[0] - lines[1]) / 3.0;
    let (l1, l2) = (l0 + lines[0], l0 + lines[1]);
    let e2 = l0 * l1 + l0 * l2 + l1 * l2;
    let e3 = l0 * l1 * l2;
    let b2 = (d * d - e2).max(0.0);
    let perp2 = (-e3 / d).clamp(0.0, b2);
    Ok(((b2 - perp2).sqrt() / gamma, perp2.sqrt() / gamma))
}

/// The field vector (T, cubic frame) from the lines of the four
/// orientations, given in the order of [`nv_orientations`] (each pair in
/// either order, signed as [`ensemble_lines`]), with the root-mean-square
/// misfit of the eight lines (Hz). Every orientation gives `|B·n_k|`; of
/// the sign patterns, the one whose field reproduces all eight lines best
/// is refined by Gauss–Newton. The field is fixed up to its sign (returned
/// with `B·n₀ ≥ 0`). A field along a cube axis ⟨100⟩ gives the same lines
/// for all four orientations and cannot be told from the other cube axes.
pub fn field_from_lines(
    lines: [[f64; 2]; 4],
    zero_field_splitting: f64,
    gamma: f64,
) -> Result<([f64; 3], f64)> {
    check_constants(zero_field_splitting, gamma)?;
    let sorted = lines.map(|[a, b]| [a.min(b), a.max(b)]);
    let axes = nv_orientations();
    let mut along = [0.0; 4];
    for k in 0..4 {
        along[k] = axial_and_transverse(sorted[k], zero_field_splitting, gamma)?.0;
    }
    let mut best = ([0.0; 3], f64::INFINITY);
    for signs in 0..8u32 {
        let s: [f64; 4] = std::array::from_fn(|k| {
            if k > 0 && signs >> (k - 1) & 1 == 1 {
                -1.0
            } else {
                1.0
            }
        });
        let mut b = [0.0; 3];
        for k in 0..4 {
            for c in 0..3 {
                b[c] += 0.75 * s[k] * along[k] * axes[k][c];
            }
        }
        let misfit: f64 = residuals(b, zero_field_splitting, gamma, &sorted)?
            .iter()
            .map(|r| r * r)
            .sum();
        if misfit < best.1 {
            best = (b, misfit);
        }
    }
    // Gauss–Newton on the eight lines, numerical Jacobian.
    let mut b = best.0;
    let scale = b.iter().map(|x| x.abs()).fold(0.0, f64::max).max(1e-9);
    for _ in 0..20 {
        let r0 = residuals(b, zero_field_splitting, gamma, &sorted)?;
        let mut jac = nalgebra::DMatrix::<f64>::zeros(8, 3);
        for c in 0..3 {
            let mut bp = b;
            let h = 1e-6 * scale;
            bp[c] += h;
            let rp = residuals(bp, zero_field_splitting, gamma, &sorted)?;
            for i in 0..8 {
                jac[(i, c)] = (rp[i] - r0[i]) / h;
            }
        }
        let r = nalgebra::DVector::from_column_slice(&r0);
        let jt = jac.transpose();
        let Some(step) = (&jt * &jac).lu().solve(&(&jt * &r)) else {
            break;
        };
        let trial: [f64; 3] = std::array::from_fn(|c| b[c] - step[c]);
        let norm2 = |x: &[f64; 8]| x.iter().map(|v| v * v).sum::<f64>();
        if norm2(&residuals(trial, zero_field_splitting, gamma, &sorted)?) > norm2(&r0) {
            break;
        }
        b = trial;
        if step.norm() < 1e-15 * scale {
            break;
        }
    }
    let first = b[0] * axes[0][0] + b[1] * axes[0][1] + b[2] * axes[0][2];
    if first < 0.0 {
        b = b.map(|x| -x);
    }
    let rms = (residuals(b, zero_field_splitting, gamma, &sorted)?
        .iter()
        .map(|r| r * r)
        .sum::<f64>()
        / 8.0)
        .sqrt();
    Ok((b, rms))
}

/// Model minus given lines, each pair ascending.
fn residuals(b: [f64; 3], d: f64, gamma: f64, lines: &[[f64; 2]; 4]) -> Result<[f64; 8]> {
    let model = ensemble_lines(b, d, gamma)?;
    Ok(std::array::from_fn(|i| {
        model[i / 2][i % 2] - lines[i / 2][i % 2]
    }))
}

#[cfg(test)]
mod tests;
