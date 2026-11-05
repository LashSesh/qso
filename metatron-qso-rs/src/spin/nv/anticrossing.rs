//! The optical cycle of an NV centre with its nuclei, near the level
//! anticrossings, without the rotating-wave approximation.
//!
//! [`NuclearOpticalNv`] is the seven-level model of [`super::optics`] (the
//! ground and excited triplets and the metastable singlet, the jumps of
//! Tetienne et al. acting on the electron in the zero-field basis) with
//! the nitrogen nucleus and optionally one ¹³C, in the product basis of
//! electron levels and nuclear states:
//!
//! ```text
//! H_g/h = D S_z² + γ_e B·S + Σ_k [S·A_g^k·I^k − γ_k B·I^k + P_k (I_z^k)²]
//! H_e/h = D_e S_z² + γ_e B·S + Σ_k [S·A_e^k·I^k − γ_k B·I^k + P_k (I_z^k)²]
//! H_s/h = Σ_k [−γ_k B·I^k + P_k (I_z^k)²]
//! ```
//!
//! The full hyperfine tensors mix `|0, m⟩` with `|∓1, m ± 1⟩`: near the
//! excited-state anticrossing (`γ_e B ≈ D_e`, 51 mT) every optical cycle
//! moves nuclear polarisation into the state `|0⟩` keeps (Jacques et al.,
//! Phys. Rev. Lett. 102, 057403 (2009)), and near the ground-state
//! anticrossing (102.4 mT) the mixing of `|0⟩` and `|−1⟩` dims the
//! fluorescence without any microwave. The nuclei may relax towards the
//! unpolarised state at `Γ_n` in every electronic level. The electron
//! dephases (`√(2Γ₂*) S_z`) in the ground triplet only, as in
//! [`super::optics`]: the excited triplet has no dephasing of its own
//! beyond its 12 ns lifetime.
//!
//! A microwave `√2·2πΩ cos(2πft) S_x`, acting on the ground triplet only
//! (not on the excited one), is kept in the laboratory frame. The periodic steady state `ρ(t) = Σ_n ρ_n e^{i n ω t}`
//! solves `(L₀ − inω) ρ_n + V (ρ_{n−1} + ρ_{n+1}) = 0` (`V` half the
//! drive's superoperator), truncated at `|n| ≤ K` and solved by a matrix
//! continued fraction, `K` doubled until the period average `ρ₀` settles
//! (Floquet); the time-averaged fluorescence is that of `ρ₀`.

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::optics::{Lindblad, OpticalNv};
use crate::error::{QsoError, Result};
use crate::linalg::{CMatrix, eigh};
use crate::spin::{Isotope, MAX_RELAXATION_WORK, TWO_PI, spin_ops};

/// Gyromagnetic ratio of ¹⁴N (Hz/T).
const N14_GAMMA: f64 = 3.077_7e6;

/// Most nuclei with the electron (the nitrogen and one ¹³C).
pub const MAX_ANTICROSSING_NUCLEI: usize = 2;

/// Most harmonics `K` of a periodic steady state.
pub const MAX_FLOQUET_HARMONICS: usize = 64;

/// The change of the period average, relative to itself, below which the
/// harmonics are taken as enough.
const FLOQUET_TOLERANCE: f64 = 1e-10;

/// Most work (operations, estimated) of a [`NuclearOpticalNv::field_scan`]
/// or [`NuclearOpticalNv::cw_odmr`] over all its points: eight times the
/// limit of one steady state.
pub const MAX_ANTICROSSING_SCAN_WORK: f64 = 8.0 * MAX_RELAXATION_WORK;

/// Work (real operations, estimated) of the steady state without microwave
/// on a superoperator of size `m`: its full-pivot LU in complex arithmetic.
fn still_work(m: usize) -> f64 {
    4.0 * (m as f64).powi(3)
}

/// Work of one round of [`floquet`] with `harmonics` harmonics: per
/// harmonic an LU, an `m`-column solve and the product `V R` in complex
/// arithmetic (about `24 m³`), and the central full-pivot LU. About 30 s
/// per harmonic for 42 levels (`m` = 1764) on the test machine.
fn floquet_work(m: usize, harmonics: usize) -> f64 {
    24.0 * harmonics as f64 * (m as f64).powi(3) + still_work(m)
}

/// The least work of a steady state with microwave: the rounds of 2 and 4
/// harmonics that the first comparison needs.
fn least_floquet_work(m: usize) -> f64 {
    floquet_work(m, 2) + floquet_work(m, 4)
}

/// A nucleus coupled to the electron in both triplets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NuclearSpin {
    /// `2I + 1` (2 or 3).
    pub levels: usize,
    /// `γ` (Hz/T) of the Zeeman term `−γ B·I`.
    pub gyromagnetic_ratio: f64,
    /// The hyperfine tensors `S·A·I` (Hz, frame of the centre) of the
    /// ground and the excited triplet.
    pub ground: [[f64; 3]; 3],
    pub excited: [[f64; 3]; 3],
    /// `P I_z²` (Hz), the same in every level.
    pub quadrupole: f64,
}

fn axial(perpendicular: f64, parallel: f64) -> [[f64; 3]; 3] {
    [
        [perpendicular, 0.0, 0.0],
        [0.0, perpendicular, 0.0],
        [0.0, 0.0, parallel],
    ]
}

impl NuclearSpin {
    /// ¹⁵N: in the ground state `A_∥ = 3.03` and `A_⊥ = 3.65 MHz` (Felton
    /// et al., Phys. Rev. B 79, 075203 (2009), the sign taken from ¹⁴N),
    /// in the excited state an isotropic 60 MHz (Jacques et al. 2009). Only
    /// the size of the excited coupling is measured (61 ± 6 MHz, Fuchs et
    /// al., Phys. Rev. Lett. 101, 117601 (2008)); its sign here is the
    /// ground state's, as ¹⁴N's −40 MHz scaled by the gyromagnetic ratios
    /// (+56 MHz) also gives. Duarte et al.'s ab initio −58.1 and −38.5 MHz
    /// (Phys. Rev. B 103, 195443 (2021)) polarise ¹⁵N at 51 mT to 0.951
    /// instead of 0.948, at 100 mT to 0.871 instead of 0.833.
    pub fn nitrogen_15() -> Self {
        NuclearSpin {
            levels: 2,
            gyromagnetic_ratio: Isotope::N15.gamma() / TWO_PI,
            ground: axial(3.65e6, 3.03e6),
            excited: axial(60e6, 60e6),
            quadrupole: 0.0,
        }
    }

    /// ¹⁴N: in the ground state `A_∥ = −2.14` and `A_⊥ = −2.70 MHz`
    /// (Felton et al. 2009) and `P = −4.945 MHz` (Doherty et al. 2013, as
    /// [`super::Nitrogen::n14`]; Felton et al. give −5.01), in the excited
    /// state `A_∥ = −40` and `A_⊥ = −23 MHz` (Poggiali et al., Phys. Rev.
    /// B 95, 195308 (2017), who measure `A_⊥ = −23 ± 3 MHz` and find that
    /// an isotropic −40 MHz does not fit).
    pub fn nitrogen_14() -> Self {
        NuclearSpin {
            levels: 3,
            gyromagnetic_ratio: N14_GAMMA,
            ground: axial(-2.70e6, -2.14e6),
            excited: axial(-23e6, -40e6),
            quadrupole: -4.945e6,
        }
    }

    /// A ¹³C with the given hyperfine tensors (Hz).
    pub fn carbon_13(ground: [[f64; 3]; 3], excited: [[f64; 3]; 3]) -> Self {
        NuclearSpin {
            levels: 2,
            gyromagnetic_ratio: Isotope::C13.gamma() / TWO_PI,
            ground,
            excited,
            quadrupole: 0.0,
        }
    }

    fn validate(&self) -> Result<()> {
        if !(self.levels == 2 || self.levels == 3) {
            return Err(QsoError::invalid(
                "nucleus",
                "a spin ½ or 1 (2 or 3 levels)",
            ));
        }
        let finite = self
            .ground
            .iter()
            .chain(&self.excited)
            .flatten()
            .chain([&self.gyromagnetic_ratio, &self.quadrupole])
            .all(|x| x.is_finite());
        if !finite {
            return Err(QsoError::invalid("nucleus", "finite couplings"));
        }
        Ok(())
    }
}

/// The steady state and what it shows.
#[derive(Clone, Debug, PartialEq)]
pub struct AnticrossingState {
    /// The density matrix (product basis: electron level major — ground
    /// `+1, 0, −1`, excited `+1, 0, −1`, singlet — then the nuclei, the
    /// first most significant, `m` descending); with a microwave its
    /// period average.
    pub rho: CMatrix,
    /// Detected photons per second.
    pub fluorescence: f64,
    /// `⟨I_z⟩/I` of each nucleus, over all levels.
    pub polarisation: Vec<f64>,
    /// `⟨I_z⟩/I` of each nucleus within the ground `m_s = 0` (what the
    /// nuclear-spin-resolved ODMR lines show).
    pub polarisation_zero: Vec<f64>,
    /// The harmonics a microwave needed (0 without).
    pub harmonics: usize,
}

/// The NV electron in its optical cycle with its nuclei (see the module
/// docs).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NuclearOpticalNv {
    /// The electron: field, splittings, rates, dephasing, spin-lattice
    /// relaxation and collection.
    pub electron: OpticalNv,
    /// The nitrogen first, then at most one more nucleus.
    pub nuclei: Vec<NuclearSpin>,
    /// `Γ_n`: each nucleus's `⟨I_z⟩` relaxes towards zero at this rate in
    /// every level (1/s).
    #[serde(default)]
    pub nuclear_relaxation: f64,
}

impl NuclearOpticalNv {
    /// A centre with `nitrogen` in `field` (T, frame of the centre) and the
    /// electron of [`OpticalNv::new`].
    pub fn new(field: [f64; 3], nitrogen: NuclearSpin) -> Self {
        NuclearOpticalNv {
            electron: OpticalNv::new(field),
            nuclei: vec![nitrogen],
            nuclear_relaxation: 0.0,
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.electron.validate()?;
        if self.nuclei.is_empty() || self.nuclei.len() > MAX_ANTICROSSING_NUCLEI {
            return Err(QsoError::invalid(
                "nuclei",
                format!("1 to {MAX_ANTICROSSING_NUCLEI} nuclei"),
            ));
        }
        for n in &self.nuclei {
            n.validate()?;
        }
        if !(self.nuclear_relaxation.is_finite() && self.nuclear_relaxation >= 0.0) {
            return Err(QsoError::invalid("nuclear_relaxation", "a finite rate ≥ 0"));
        }
        Ok(())
    }

    /// From JSON (the fields of this struct), checked.
    pub fn from_json(text: &str) -> Result<Self> {
        let nv: NuclearOpticalNv = serde_json::from_str(text)
            .map_err(|e| QsoError::invalid("json", format!("not a centre with nuclei: {e}")))?;
        nv.validate()?;
        Ok(nv)
    }

    /// As JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("models serialise")
    }

    /// The dimension of the nuclei.
    fn nuclear_dimension(&self) -> usize {
        self.nuclei.iter().map(|n| n.levels).product()
    }

    /// The dimension of the model: 7 electron levels times the nuclei.
    pub fn dimension(&self) -> usize {
        7 * self.nuclear_dimension()
    }

    /// `op` on nucleus `k` within the nuclei.
    fn on_nucleus(&self, k: usize, op: &CMatrix) -> CMatrix {
        let mut out = CMatrix::identity(1, 1);
        for (j, n) in self.nuclei.iter().enumerate() {
            let factor = if j == k {
                op.clone()
            } else {
                CMatrix::identity(n.levels, n.levels)
            };
            out = out.kronecker(&factor);
        }
        out
    }

    /// The nuclei's own terms `Σ_k −γ_k B·I^k + P_k (I_z^k)²` (Hz).
    fn nuclear_terms(&self) -> CMatrix {
        let nn = self.nuclear_dimension();
        let [bx, by, bz] = self.electron.field;
        let c = |x: f64| C64::new(x, 0.0);
        let mut h = CMatrix::zeros(nn, nn);
        for (k, n) in self.nuclei.iter().enumerate() {
            let [iz, _, _, ix, iy] = spin_ops(n.levels);
            let g = n.gyromagnetic_ratio;
            let zeeman = &ix * c(-g * bx) + &iy * c(-g * by) + &iz * c(-g * bz);
            h += self.on_nucleus(k, &(zeeman + &iz * &iz * c(n.quadrupole)));
        }
        h
    }

    /// A triplet with zero-field splitting `d` and the hyperfine tensors
    /// `tensor(nucleus)` (Hz).
    fn triplet(&self, d: f64, excited: bool) -> CMatrix {
        let nn = self.nuclear_dimension();
        let [sz, _, _, sx, sy] = spin_ops(3);
        let s = [&sx, &sy, &sz];
        let [bx, by, bz] = self.electron.field;
        let g = self.electron.gyromagnetic_ratio;
        let c = |x: f64| C64::new(x, 0.0);
        let electron = &sz * &sz * c(d) + &sx * c(g * bx) + &sy * c(g * by) + &sz * c(g * bz);
        let mut h = electron.kronecker(&CMatrix::identity(nn, nn));
        for (k, n) in self.nuclei.iter().enumerate() {
            let [iz, _, _, ix, iy] = spin_ops(n.levels);
            let i = [&ix, &iy, &iz];
            let a = if excited { &n.excited } else { &n.ground };
            for (p, row) in a.iter().enumerate() {
                for (q, &v) in row.iter().enumerate() {
                    if v != 0.0 {
                        h += s[p].kronecker(&self.on_nucleus(k, i[q])) * c(v);
                    }
                }
            }
        }
        h + CMatrix::identity(3, 3).kronecker(&self.nuclear_terms())
    }

    /// `H/h` of the ground triplet with the nuclei (Hz; electron `+1, 0,
    /// −1` major).
    pub fn ground_hamiltonian(&self) -> CMatrix {
        self.triplet(self.electron.zero_field_splitting, false)
    }

    /// `H/h` of the excited triplet with the nuclei (Hz).
    pub fn excited_hamiltonian(&self) -> CMatrix {
        self.triplet(self.electron.excited_splitting, true)
    }

    /// Energies (Hz, ascending) and eigenvectors of the ground triplet with
    /// the nuclei.
    pub fn ground_levels(&self) -> Result<(Vec<f64>, CMatrix)> {
        self.validate()?;
        Ok(eigh(&self.ground_hamiltonian()))
    }

    /// Energies and eigenvectors of the excited triplet with the nuclei.
    pub fn excited_levels(&self) -> Result<(Vec<f64>, CMatrix)> {
        self.validate()?;
        Ok(eigh(&self.excited_hamiltonian()))
    }

    /// The Lindblad model without a microwave under `pump` (rad/s).
    fn model(&self, pump: f64) -> Result<Lindblad> {
        self.validate()?;
        if !(pump.is_finite() && pump >= 0.0) {
            return Err(QsoError::invalid("pump", "a finite rate ≥ 0"));
        }
        let nn = self.nuclear_dimension();
        let n = 7 * nn;
        let id = CMatrix::identity(nn, nn);
        let mut h = CMatrix::zeros(n, n);
        let mut place = |block: &CMatrix, offset: usize| {
            for r in 0..block.nrows() {
                for c in 0..block.ncols() {
                    h[(offset + r, offset + c)] = block[(r, c)] * TWO_PI;
                }
            }
        };
        place(&self.ground_hamiltonian(), 0);
        place(&self.excited_hamiltonian(), 3 * nn);
        place(&self.nuclear_terms(), 6 * nn);
        // Electron jumps (zero-field basis: ground 0..3, excited 3..6,
        // singlet 6, each m_s = +1, 0, −1) on the identity of the nuclei.
        let unit = |to: usize, from: usize| {
            let mut m = CMatrix::zeros(7, 7);
            m[(to, from)] = C64::new(1.0, 0.0);
            m
        };
        let mut ops = Vec::new();
        let mut push = |op: CMatrix, rate: f64| {
            if rate > 0.0 {
                ops.push(op * C64::new(rate.sqrt(), 0.0));
            }
        };
        let r = &self.electron.rates;
        let isc = [r.isc_plus_minus, r.isc_zero, r.isc_plus_minus];
        let back = [r.singlet_plus_minus, r.singlet_zero, r.singlet_plus_minus];
        for p in 0..3 {
            push(unit(3 + p, p).kronecker(&id), pump);
            push(unit(p, 3 + p).kronecker(&id), r.radiative);
            push(unit(6, 3 + p).kronecker(&id), isc[p]);
            push(unit(p, 6).kronecker(&id), back[p]);
        }
        for (p, q) in [(1, 0), (0, 1), (1, 2), (2, 1)] {
            push(unit(q, p).kronecker(&id), self.electron.spin_lattice);
        }
        let mut z = CMatrix::zeros(7, 7);
        z[(0, 0)] = C64::new(1.0, 0.0);
        z[(2, 2)] = C64::new(-1.0, 0.0);
        push(z.kronecker(&id), 2.0 * self.electron.dephasing);
        // Each nucleus relaxes by I₊ and I₋ at Γ_n/2 each: ⟨I_z⟩ decays at
        // Γ_n.
        for (k, nucleus) in self.nuclei.iter().enumerate() {
            let [_, ip, im, _, _] = spin_ops(nucleus.levels);
            for op in [ip, im] {
                push(
                    CMatrix::identity(7, 7).kronecker(&self.on_nucleus(k, &op)),
                    self.nuclear_relaxation / 2.0,
                );
            }
        }
        Ok(Lindblad { n, h, ops })
    }

    /// The superoperator of `√2·2πΩ cos(2πft) S_x` on the ground triplet,
    /// without its time dependence (`L₁`, rad/s).
    fn drive(&self, rabi: f64) -> CMatrix {
        let nn = self.nuclear_dimension();
        let n = 7 * nn;
        let [_, _, _, sx, _] = spin_ops(3);
        let block = sx.kronecker(&CMatrix::identity(nn, nn));
        let scale = TWO_PI * rabi * std::f64::consts::SQRT_2;
        let mut h = CMatrix::zeros(n, n);
        for r in 0..3 * nn {
            for c in 0..3 * nn {
                h[(r, c)] = block[(r, c)] * scale;
            }
        }
        Lindblad {
            n,
            h,
            ops: Vec::new(),
        }
        .superoperator()
    }

    /// A state of the model must be `dimension × dimension`.
    fn check_state(&self, rho: &CMatrix) -> Result<()> {
        let n = self.dimension();
        if rho.nrows() != n || rho.ncols() != n {
            return Err(QsoError::DimensionMismatch {
                expected: n,
                actual: if rho.nrows() != n {
                    rho.nrows()
                } else {
                    rho.ncols()
                },
            });
        }
        Ok(())
    }

    /// Detected photons per second in a state of the model.
    pub fn fluorescence(&self, rho: &CMatrix) -> Result<f64> {
        self.check_state(rho)?;
        let nn = self.nuclear_dimension();
        let excited: f64 = (3 * nn..6 * nn).map(|j| rho[(j, j)].re).sum();
        Ok(self.electron.collection * self.electron.rates.radiative * excited)
    }

    /// `⟨I_z⟩/I` of nucleus `k` in a state of the model, over all levels.
    pub fn nuclear_polarisation(&self, rho: &CMatrix, k: usize) -> Result<f64> {
        self.check_state(rho)?;
        let nucleus = self.nuclei.get(k).ok_or_else(|| {
            QsoError::invalid("nucleus", format!("nucleus {k} of {}", self.nuclei.len()))
        })?;
        let [iz, ..] = spin_ops(nucleus.levels);
        let op = CMatrix::identity(7, 7).kronecker(&self.on_nucleus(k, &iz));
        let spin = (nucleus.levels as f64 - 1.0) / 2.0;
        Ok((rho * op).trace().re / spin)
    }

    /// `⟨I_z⟩/I` of nucleus `k` within electron level `level` of the
    /// zero-field basis (ground `+1, 0, −1` = 0, 1, 2; excited 3, 4, 5;
    /// singlet 6), normalised by that level's population.
    pub fn nuclear_polarisation_in(&self, rho: &CMatrix, k: usize, level: usize) -> Result<f64> {
        self.check_state(rho)?;
        let nucleus = self.nuclei.get(k).ok_or_else(|| {
            QsoError::invalid("nucleus", format!("nucleus {k} of {}", self.nuclei.len()))
        })?;
        if level >= 7 {
            return Err(QsoError::invalid("level", "an electron level 0 to 6"));
        }
        let nn = self.nuclear_dimension();
        let [iz, ..] = spin_ops(nucleus.levels);
        let op = self.on_nucleus(k, &iz);
        let block = rho.view((level * nn, level * nn), (nn, nn)).clone_owned();
        let population = block.trace().re;
        if population <= 0.0 {
            return Ok(0.0);
        }
        let spin = (nucleus.levels as f64 - 1.0) / 2.0;
        Ok((block * op).trace().re / (population * spin))
    }

    fn summary(&self, rho: CMatrix, harmonics: usize) -> Result<AnticrossingState> {
        let polarisation = (0..self.nuclei.len())
            .map(|k| self.nuclear_polarisation(&rho, k))
            .collect::<Result<Vec<_>>>()?;
        let polarisation_zero = (0..self.nuclei.len())
            .map(|k| self.nuclear_polarisation_in(&rho, k, 1))
            .collect::<Result<Vec<_>>>()?;
        Ok(AnticrossingState {
            fluorescence: self.fluorescence(&rho)?,
            polarisation,
            polarisation_zero,
            rho,
            harmonics,
        })
    }

    /// The steady state under `pump` (1/s per spin-conserving transition)
    /// and, if given, a microwave of frequency `drive.0` (Hz, positive) and
    /// Rabi frequency `drive.1` (Hz, as in [`OpticalNv`]) in the laboratory
    /// frame (its period average). It needs the laser or spin-lattice
    /// relaxation to be unique, and the nuclei need `Γ_n` or hyperfine
    /// couplings that flip them (`A_⊥`, or `A_∥` with a field across the
    /// axis: not none at all, nor `A_∥` alone along the axis); a steady
    /// state that is not unique (a pivot of its system below `1e-14` of
    /// the largest) is refused.
    pub fn steady_state(&self, pump: f64, drive: Option<(f64, f64)>) -> Result<AnticrossingState> {
        if pump == 0.0 && self.electron.spin_lattice == 0.0 {
            return Err(QsoError::invalid(
                "pump",
                "without the laser and spin-lattice relaxation the steady state is not unique",
            ));
        }
        let model = self.model(pump)?;
        let Some((frequency, rabi)) = drive else {
            return self.summary(model.steady_state()?, 0);
        };
        if !(frequency.is_finite() && frequency > 0.0 && rabi.is_finite() && rabi >= 0.0) {
            return Err(QsoError::invalid(
                "drive",
                "a positive finite frequency and a Rabi frequency ≥ 0",
            ));
        }
        if rabi == 0.0 {
            return self.summary(model.steady_state()?, 0);
        }
        let l0 = model.superoperator();
        let v = self.drive(rabi) * C64::new(0.5, 0.0);
        let omega = TWO_PI * frequency;
        let m = l0.nrows();
        let mut harmonics = 2;
        let mut last: Option<CMatrix> = None;
        let mut spent = 0.0;
        loop {
            // The rounds so far and the next, at the start with the one
            // after it: one round alone shows no convergence.
            let mut work = spent + floquet_work(m, harmonics);
            let mut upto = harmonics;
            if last.is_none() {
                upto *= 2;
                work += floquet_work(m, upto);
            }
            if work > MAX_RELAXATION_WORK {
                return Err(QsoError::invalid(
                    "drive",
                    format!(
                        "{upto} harmonics of a model of {} levels need about {work:.1e} \
                         operations with the rounds before (limit {:.1e}); use fewer nuclei, a \
                         weaker drive or a higher frequency",
                        self.dimension(),
                        MAX_RELAXATION_WORK
                    ),
                ));
            }
            spent += floquet_work(m, harmonics);
            let rho = floquet(&l0, &v, omega, harmonics, self.dimension())?;
            if let Some(prev) = &last {
                let change = (&rho - prev).norm() / rho.norm();
                if change <= FLOQUET_TOLERANCE {
                    return self.summary(rho, harmonics);
                }
            }
            if harmonics >= MAX_FLOQUET_HARMONICS {
                return Err(QsoError::invalid(
                    "drive",
                    format!(
                        "the period average had not settled at {MAX_FLOQUET_HARMONICS} harmonics \
                         (a drive far stronger than its frequency)"
                    ),
                ));
            }
            last = Some(rho);
            harmonics *= 2;
        }
    }

    /// A scan estimated above [`MAX_ANTICROSSING_SCAN_WORK`] is refused.
    fn check_scan(&self, work: f64) -> Result<()> {
        if work > MAX_ANTICROSSING_SCAN_WORK {
            return Err(QsoError::invalid(
                "scan",
                format!(
                    "the points of a model of {} levels need at least about {work:.1e} \
                     operations (limit {MAX_ANTICROSSING_SCAN_WORK:.1e}); use fewer points or \
                     fewer nuclei",
                    self.dimension()
                ),
            ));
        }
        Ok(())
    }

    /// The fluorescence and nuclear polarisations without a microwave at
    /// each field (T, frame of the centre), in parallel. At most
    /// [`MAX_ANTICROSSING_SCAN_WORK`] in total (`4 m³` per point, `m` the
    /// square of [`NuclearOpticalNv::dimension`]: about 550 points of 42
    /// levels).
    pub fn field_scan(&self, fields: &[[f64; 3]], pump: f64) -> Result<Vec<(f64, Vec<f64>)>> {
        if fields.len() > super::optics::MAX_ODMR_POINTS {
            return Err(QsoError::invalid(
                "fields",
                format!("at most {}", super::optics::MAX_ODMR_POINTS),
            ));
        }
        let m = self.dimension().pow(2);
        self.check_scan(fields.len() as f64 * still_work(m))?;
        crate::parallel::map_tasks(fields.len(), |k| {
            let mut nv = self.clone();
            nv.electron.field = fields[k];
            let s = nv.steady_state(pump, None)?;
            Ok((s.fluorescence, s.polarisation))
        })
        .into_iter()
        .collect()
    }

    /// The time-averaged photon rate under `pump` at each microwave
    /// frequency (Hz), the microwave in the laboratory frame. At most
    /// [`MAX_ANTICROSSING_SCAN_WORK`] in total, each point counted with the
    /// 2 and 4 harmonics it needs at least (about 14 points of 42 levels,
    /// 900 of 21, 10⁴ of 14).
    pub fn cw_odmr(&self, frequencies: &[f64], rabi: f64, pump: f64) -> Result<Vec<f64>> {
        if frequencies.len() > super::optics::MAX_ODMR_POINTS {
            return Err(QsoError::invalid(
                "frequencies",
                format!("at most {}", super::optics::MAX_ODMR_POINTS),
            ));
        }
        if rabi > 0.0 {
            let m = self.dimension().pow(2);
            self.check_scan(frequencies.len() as f64 * least_floquet_work(m))?;
        }
        crate::parallel::map_tasks(frequencies.len(), |k| {
            Ok(self
                .steady_state(pump, Some((frequencies[k], rabi)))?
                .fluorescence)
        })
        .into_iter()
        .collect()
    }

    /// The shot-noise-limited sensitivity (T/√Hz) of the fluorescence to a
    /// field along `direction` (a unit vector is taken of it) without a
    /// microwave: `√R / |∂R/∂B|`, the derivative by central differences
    /// of `step` (T), and the rate `R` and the derivative.
    pub fn field_sensitivity(
        &self,
        pump: f64,
        direction: [f64; 3],
        step: f64,
    ) -> Result<(f64, f64, f64)> {
        let size = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !(size.is_finite() && size > 0.0) {
            return Err(QsoError::invalid("direction", "a finite nonzero direction"));
        }
        if !(step.is_finite() && step > 0.0) {
            return Err(QsoError::invalid("step", "a positive finite step (T)"));
        }
        let at = |sign: f64| -> Result<f64> {
            let mut nv = self.clone();
            for (b, d) in nv.electron.field.iter_mut().zip(direction) {
                *b += sign * step * d / size;
            }
            Ok(nv.steady_state(pump, None)?.fluorescence)
        };
        let rate = self.steady_state(pump, None)?.fluorescence;
        let slope = (at(1.0)? - at(-1.0)?) / (2.0 * step);
        Ok((rate.sqrt() / slope.abs(), rate, slope))
    }
}

/// The period average `ρ₀` of the periodic steady state of `L₀ + 2V cos ωt`
/// on `n` levels, the harmonics truncated at `|k| ≤ harmonics`, by a matrix
/// continued fraction: `ρ_k = R_k ρ_{k−1}` with `R_k = −(L₀ − ikω +
/// V R_{k+1})⁻¹ V` from `R_{K+1} = 0`, the negative side by conjugation
/// (`ρ_{−k} = ρ_k†`), and `(L₀ + V (R₁ + R₋₁)) ρ₀ = 0` with unit trace.
fn floquet(l0: &CMatrix, v: &CMatrix, omega: f64, harmonics: usize, n: usize) -> Result<CMatrix> {
    let m = l0.nrows();
    let refuse = || QsoError::invalid("drive", "the periodic steady state is not unique");
    let mut r = CMatrix::zeros(m, m);
    for k in (1..=harmonics).rev() {
        let mut a = l0 + v * &r;
        for i in 0..m {
            a[(i, i)] -= C64::new(0.0, k as f64 * omega);
        }
        let lu = a.lu();
        r = -lu.solve(v).ok_or_else(refuse)?;
    }
    // R₋₁ = P conj(R₁) P with P the transposition of vec ρ (r + n c ↔ c +
    // n r): the map ρ ↦ ρ† on vectors.
    let swap = |i: usize| (i % n) * n + i / n;
    let r_minus = CMatrix::from_fn(m, m, |a, b| r[(swap(a), swap(b))].conj());
    let mut a = l0 + v * (&r + &r_minus);
    let mut rhs = nalgebra::DVector::from_element(m, C64::new(0.0, 0.0));
    for c in 0..m {
        a[(0, c)] = C64::new(0.0, 0.0);
    }
    for k in 0..n {
        a[(0, k + n * k)] = C64::new(1.0, 0.0);
    }
    rhs[0] = C64::new(1.0, 0.0);
    let x = super::optics::solve_unique(a, &rhs).ok_or_else(refuse)?;
    let rho = CMatrix::from_fn(n, n, |r, c| x[r + n * c]);
    if rho.iter().any(|z| !z.is_finite()) {
        return Err(refuse());
    }
    // Hermitian up to rounding.
    Ok((&rho + rho.adjoint()) * C64::new(0.5, 0.0))
}

#[cfg(test)]
mod tests;
