//! Radical pairs: spin-selective recombination, magnetic-field effects on
//! reaction yields and chemically induced dynamic nuclear polarisation
//! (CIDNP).
//!
//! A radical pair is two electrons, one on each radical, with the nuclei of
//! both in one [`SpinSystem`] (two electrons, one ¹⁴N and seven protons take
//! 1536 of its 2048 dimensions). It is born correlated in its cage, in the
//! singlet or a triplet state ([`Birth`]; the nuclei maximally mixed), and
//! recombines from the singlet at `k_S` (the cage product), from the triplet
//! at `k_T`, or escapes at `k_E`, so that its products carry the nuclear
//! polarisation the hyperfine couplings sorted into them (Steiner and
//! Ulrich, Chem. Rev. 89, 51 (1989); Timmel, Till, Brocklehurst, McLauchlan
//! and Hore, Mol. Phys. 95, 71 (1998)).
//!
//! # Units
//!
//! Angular frequencies are in rad/s and rates in s⁻¹. The Hamiltonian is
//! the system's ([`SpinSystem::hamiltonian`]); in the laboratory frame
//!
//! ```text
//! H = Σ_e |γ_e|B(1 + δ_e·10⁻⁶) S_ez − Σ_n γ_n B(1 + δ_n·10⁻⁶) I_nz
//!     + Σ_(e,n) |γ_e| a_en·10⁻³ S_e·I_n − 2|γ_e| J·10⁻³ S₁·S₂        (a, J in mT)
//! ```
//!
//! with the g-values as shifts `δ = (g/g_e − 1)·10⁶` ppm
//! ([`SpinSystem::add_electron`]), the hyperfine couplings
//! ([`SpinSystem::hyperfine`]) and the exchange
//! ([`SpinSystem::exchange_coupling`]) as couplings in Hz. In the rotating
//! frame the Zeeman terms are offsets (`|γ_e|B(δ − c_e)·10⁻⁶ S_z` for an
//! electron), the hyperfine couplings are secular (`a S_z I_z`) and the
//! exchange stays full, so `Δω = |γ_e|B(δ₁ − δ₂)·10⁻⁶ = (g₁ − g₂)μ_B B/ħ` in
//! both frames. The rotating frame needs `|γ_e|B` far above the hyperfine
//! couplings; below about ten times the largest coupling (in mT) use the
//! laboratory frame. `T₁` and `T₂` of electrons and nuclei relax as in
//! every spin system, towards thermal equilibrium at the full Larmor
//! frequency.
//!
//! # The equation of motion
//!
//! With `P_S` and `P_T = 1 − P_S` the projectors on the electrons' singlet
//! and triplet,
//!
//! ```text
//! dρ/dt = −i[H, ρ] − (k_S/2){P_S, ρ} − (k_T/2){P_T, ρ} − k_E ρ
//!         − c (P_S ρ P_T + P_T ρ P_S)
//!         + Σ_j (L_j ρ L_j† − ½{L_j†L_j, ρ})
//! ```
//!
//! where `c = 0` for Haberkorn's term (Mol. Phys. 32, 1491 (1976)) and
//! `c = (k_S + k_T)/2` for that of Jones, Maeda and Hore (Chem. Phys. Lett.
//! 507, 269 (2011)), and `L_j` are the system's `T₁` and `T₂` jumps. The
//! trace falls as the pair reacts. With `W = P_T − P_S` (the swap of the two
//! electrons), `K = k_S P_S + k_T P_T` and `Γ = Σ_j L_j†L_j`,
//!
//! ```text
//! A    = H − (i/2)(K + Γ),       σ = i(k_E + c/2),
//! 𝓕(X) = Σ_j L_j X L_j† + (c/2) W X W,
//! 𝓛(X) = −i(A X − X A† − σX) + 𝓕(X),
//! ```
//!
//! and `X = ∫₀^∞ ρ dt` solves `𝓛(X) = −ρ₀`. The yields are
//! `Φ_S = k_S Tr(P_S X)`, `Φ_T = k_T Tr(P_T X)` and `Φ_E = k_E Tr X`, and
//! they sum to one. The polarisation of nucleus `j` in product `P` is
//! `k_P Tr(P_P I_jz X)` (`P_E = 1`), the product's nuclear state
//! `k_P Tr_e(P_P X)`.
//!
//! # Solves
//!
//! `H`, `K`, `W` and the jumps keep the system's labels (`2M` per isotope
//! in the rotating frame, the total in the laboratory frame), so a start
//! without coherences between labels (every [`Birth`]) gives a
//! block-diagonal `X`, solved block by block (other starts take one block
//! of all states):
//!
//! * with Haberkorn's term and no jumps, `𝓕 = 0` and each block is one
//!   Sylvester equation `A X − X A† − i k_E X = −iρ₀`, the roadmap's
//!   `H_eff X − X H_eff† = −iρ₀` with `H_eff = H − (i/2)(K + k_E)`, solved on
//!   the complex Schur form of `A` (Bartels–Stewart, the solver of 1.63) and
//!   refined twice at most; blocks the start does not occupy are zero;
//! * with the term of Jones, Maeda and Hore, or with jumps, by restarted
//!   GMRES (tolerance 10⁻¹³ of a backward error), preconditioned on the
//!   right by the same Sylvester solves, over the blocks the start occupies
//!   and those the jumps reach from them; the others are zero.
//!
//! [`PairYields::backward_error`] reports `‖𝓛X + ρ₀‖/(‖𝓛‖‖X‖ + ‖ρ₀‖)`.
//! The blocks are solved in parallel and every result is bit-identical on
//! any number of threads.
//!
//! # Refusals and limits
//!
//! States that never decay (`k_E = 0` and `k_S = 0` or `k_T = 0`, with an
//! eigenstate of `H` wholly in the other manifold, such as the `F = 3/2`
//! states of a pair with one proton at zero field) leave no finite `X` in
//! their block of the labels: a start that occupies such a block is
//! refused, even where it does not occupy those states themselves (a
//! singlet-born pair at zero field with `k_S` alone), and an escape rate is
//! the remedy. The same refusal covers decays below 10⁻¹⁴ of the block's
//! largest frequency, which double precision cannot resolve. With
//! relaxation such states (those that the jumps leave alone, such as
//! `|ααα⟩` under electron dephasing alone) are found after the solve, as a
//! backward error alone would accept an `X` grown without bound: a result
//! is accepted with a residual `‖𝓛X + ρ₀‖/‖ρ₀‖` of at most 10⁻⁶, or with a
//! backward error of at most 10⁻¹² and `‖X‖` within 10³ times `‖ρ₀‖/g`, `g`
//! the slowest decay of `A`. The second covers slow rates in the
//! laboratory frame at high field, where `‖𝓛‖ ≈ 2|γ_e|B` is 10⁹ to 10¹⁴
//! times the slowest decay and an accurate `X` leaves a residual near
//! 10⁻¹⁶ times that ratio; the yields are then accurate to about 10⁻¹⁰ to
//! 10⁻⁷ (2·10⁻⁸ for one ¹H and one ¹⁴N at 3 T with `k = 0.1 s⁻¹`, 6·10⁻⁸
//! with `k = 0.01 s⁻¹`, against a 50-digit solution). A solve is refused
//! before anything is factorised when it would take more than
//! [`super::MAX_RELAXATION_WORK`] operations or
//! [`super::MAX_RELAXATION_MEMORY`] bytes; [`RadicalPair::evolve`] with
//! the term of Jones, Maeda and Hore or jumps needs a Liouville space of at
//! most [`super::MAX_LIOUVILLE_BLOCK`] elements. Redfield relaxation is
//! not part of the model.
//!
//! # Conventions of RadicalPy
//!
//! | RadicalPy 1.0.9 | QSO |
//! |---|---|
//! | `Exponential(k)` (`HilbertSimulation`) | `with_rates(k, k, 0)`, Haberkorn |
//! | `Haberkorn(k, S)`, `Haberkorn(k, T)`, `HaberkornFree(k)` | `k_S`, `k_T`, `k_E` |
//! | `JonesHore(kS, kT)` | [`Recombination::JonesMaedaHore`] (the same operator) |
//! | `experiments.cidnp(model "a", ks)` | `2·cidnp[j][1]` of the exponential model at `k = ks/2`, singlet birth, rotating frame, radical 1 the donor |
//! | `exchange_hamiltonian(J)` | [`SpinSystem::exchange_coupling`]`(J)` |
//! | hyperfine of a `Nucleus` (mT) | [`SpinSystem::hyperfine`]`(a)` |
//!
//! RadicalPy's Liouvillian runs time backwards (its `ρ(t)` is the complex
//! conjugate, so yields agree for real `H`); its `mary` multiplies every
//! yield by the first kinetic rate; its trapezoid quadrature errs by
//! `(k·dt)²/12`; its ¹H γ is CODATA's, 2.2·10⁻⁷ above QSO's IUPAC value.
//!
//! # CIDNP
//!
//! Kaptein's rule (J. Chem. Soc. D, 732 (1971)): the polarisation of a
//! nucleus in a product has the sign `μ·ε·sgn(Δg)·sgn(a)`, with `μ = −1`
//! for a singlet precursor and `+1` for a triplet one, `ε = +1` for the
//! singlet (cage) product and `−1` for the triplet and escape products, `Δg`
//! the g-value of the nucleus's radical less the other's; positive is
//! enhanced absorption for `γ > 0`. The rule is that of S–T₀ mixing at
//! high field (the rotating frame), whose Hamiltonian keeps each nucleus's
//! `I_z`: there, without nuclear relaxation, the products' polarisations
//! cancel nucleus by nucleus, and with `k_S = k_T` the escape product
//! carries none. In the laboratory frame S–T± mixing passes polarisation
//! between the electrons and the nuclei, so the products' nuclear
//! polarisations need not cancel, not even in total over the nuclei
//! (except at zero field, nucleus by nucleus by isotropy), and with
//! exchange or a T± birth the escape product is polarised with `k_S = k_T`
//! too. Without relaxation, both terms keep the total: `⟨S_z + Σ I_z⟩`
//! summed over the products, electrons included, equals its value at
//! birth.
//!
//! ```
//! use metatron_qso::spin::{Birth, Frame, Isotope, RadicalPair, SpinSystem, FREE_ELECTRON_G};
//!
//! // One proton at zero field, k = a/2: Φ_S = 5/8 + (3/8)k²/(k² + a²) = 0.7.
//! let mut sys = SpinSystem::new(0.0).with_frame(Frame::Laboratory);
//! let e1 = sys.add_electron(FREE_ELECTRON_G).unwrap();
//! let e2 = sys.add_electron(FREE_ELECTRON_G).unwrap();
//! let h = sys.add_spin(Isotope::H1, 0.0).unwrap();
//! sys.hyperfine(e1, h, 1.0).unwrap();
//! let k = 0.5 * 1.760_859_630_23e11 * 1e-3;
//! let pair = RadicalPair::new(sys).unwrap().with_rates(k, k, 0.0).unwrap();
//! let yields = pair.solve(Birth::Singlet).unwrap().yields().clone();
//! assert!((yields.singlet - 0.7).abs() < 1e-13);
//! assert!((yields.singlet + yields.triplet + yields.escape - 1.0).abs() < 1e-13);
//! # let _ = e2;
//! ```

use serde::{Deserialize, Serialize};

use super::exchange::{Fresh, carry_state, check_no_electrons};
use super::{Isotope, SpinSystem, ZERO, check_field, check_time};
use crate::error::{QsoError, Result};
use crate::linalg::CMatrix;
use num_complex::Complex64 as C64;

mod solve;

#[cfg(test)]
mod reference_tests;
#[cfg(test)]
mod tests;

use solve::Options;

/// The electron state a pair is born in; the nuclei are maximally mixed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum Birth {
    /// From a singlet precursor.
    #[default]
    Singlet,
    /// `|T₀⟩ = (|αβ⟩ + |βα⟩)/√2`.
    TripletZero,
    /// `|T₊⟩ = |αα⟩` (`M_S = +1`).
    TripletPlus,
    /// `|T₋⟩ = |ββ⟩` (`M_S = −1`).
    TripletMinus,
    /// The three triplet states equally (a triplet precursor).
    Triplet,
    /// No spin correlation: radicals that met at random (F-pairs), `1/d`.
    Uncorrelated,
}

/// The spin-selective recombination term.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum Recombination {
    /// `−(k_S/2){P_S, ρ} − (k_T/2){P_T, ρ}` (Haberkorn, Mol. Phys. 32, 1491
    /// (1976)).
    #[default]
    Haberkorn,
    /// Haberkorn's term and `−((k_S + k_T)/2)(P_S ρ P_T + P_T ρ P_S)`, an
    /// extra decay of singlet–triplet coherences (Jones, Maeda and Hore,
    /// Chem. Phys. Lett. 507, 269 (2011); RadicalPy's `JonesHore`).
    JonesMaedaHore,
}

/// A product of the pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum Product {
    /// Recombination from the singlet (the cage product), rate `k_S`.
    Singlet,
    /// Recombination from the triplet, rate `k_T`.
    Triplet,
    /// Escape from the cage, rate `k_E`.
    Escape,
}

/// Two radicals, their nuclei and their reactions (see the module docs).
/// Electron 1 (radical 1) is the electron of lower index.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct RadicalPair {
    system: SpinSystem,
    electrons: [usize; 2],
    /// `k_S`, `k_T`, `k_E` (s⁻¹).
    rates: [f64; 3],
    recombination: Recombination,
}

/// Yields and nuclear polarisations per pair born (a start of trace 1).
#[derive(Clone, Debug, PartialEq, Serialize)]
#[non_exhaustive]
pub struct PairYields {
    /// Field (T).
    pub field: f64,
    /// `Φ_S = k_S Tr(P_S X)`.
    pub singlet: f64,
    /// `Φ_T = k_T Tr(P_T X)`.
    pub triplet: f64,
    /// `Φ_E = k_E Tr X`.
    pub escape: f64,
    /// For each nucleus (in the order of [`RadicalPair::nuclei`]), `⟨I_z⟩`
    /// in the singlet, triplet and escape products per pair born,
    /// `k_P Tr(P_P I_z X)` (`P_E = 1`). CIDNP: positive is enhanced
    /// absorption for `γ > 0`. For spin ½, RadicalPy's `P_α − P_β` is
    /// `2⟨I_z⟩`.
    pub cidnp: Vec<[f64; 3]>,
    /// `‖𝓛X + ρ₀‖_F / (‖𝓛‖_est‖X‖_F + ‖ρ₀‖_F)`, with `‖𝓛‖_est` a bound on
    /// the operator's norm.
    pub backward_error: f64,
    /// GMRES iterations (0: direct Sylvester solves).
    pub iterations: usize,
}

/// A solved pair: its yields and `X = ∫₀^∞ ρ dt`, for the products.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PairSolution {
    /// The pair at the solved field.
    pair: RadicalPair,
    yields: PairYields,
    /// `X`, `d × d` (zero between blocks).
    x: CMatrix,
}

impl RadicalPair {
    /// The pair of the system's two electrons, with all rates 0 and
    /// Haberkorn's term.
    ///
    /// Errors: an invalid system; not exactly two electrons; Redfield
    /// relaxation (a correlation time or a diffusion tensor) in the system.
    pub fn new(system: SpinSystem) -> Result<Self> {
        if system.has_motion() {
            return Err(QsoError::invalid(
                "system",
                "Redfield relaxation is not part of the radical-pair model; give T₁ and T₂",
            ));
        }
        system.validate()?;
        let electrons: Vec<usize> = (0..system.spins().len())
            .filter(|&i| system.spins()[i].isotope == Isotope::Electron)
            .collect();
        if electrons.len() != 2 {
            return Err(QsoError::invalid(
                "system",
                format!(
                    "a radical pair has exactly two electrons, not {}",
                    electrons.len()
                ),
            ));
        }
        Ok(RadicalPair {
            system,
            electrons: [electrons[0], electrons[1]],
            rates: [0.0; 3],
            recombination: Recombination::Haberkorn,
        })
    }

    /// The same pair with recombination from the singlet at `singlet`, from
    /// the triplet at `triplet` and escape at `escape` (s⁻¹; finite and
    /// non-negative).
    pub fn with_rates(mut self, singlet: f64, triplet: f64, escape: f64) -> Result<Self> {
        for k in [singlet, triplet, escape] {
            if !(k.is_finite() && k >= 0.0) {
                return Err(QsoError::invalid(
                    "rates",
                    format!("rates must be finite and non-negative, not {k}"),
                ));
            }
        }
        self.rates = [singlet, triplet, escape];
        Ok(self)
    }

    /// The same pair with another recombination term.
    pub fn with_recombination(mut self, model: Recombination) -> Self {
        self.recombination = model;
        self
    }

    /// The same pair at field `b` (T), as [`SpinSystem::with_field`].
    pub fn with_field(mut self, b: f64) -> Result<Self> {
        self.system = self.system.with_field(b)?;
        Ok(self)
    }

    /// The spin system of the pair.
    pub fn system(&self) -> &SpinSystem {
        &self.system
    }

    /// The indices of the two electrons, the lower first.
    pub fn electrons(&self) -> [usize; 2] {
        self.electrons
    }

    /// The nuclei in index order: the order of [`PairYields::cidnp`] and of
    /// [`PairSolution::product`].
    pub fn nuclei(&self) -> Vec<usize> {
        (0..self.system.spins().len())
            .filter(|i| !self.electrons.contains(i))
            .collect()
    }

    /// `[k_S, k_T, k_E]` (s⁻¹).
    pub fn rates(&self) -> [f64; 3] {
        self.rates
    }

    /// The recombination term.
    pub fn recombination(&self) -> Recombination {
        self.recombination
    }

    /// For every basis state, the state with the two electrons' levels
    /// exchanged: `W|s⟩ = |π(s)⟩`.
    fn swap(&self) -> Vec<usize> {
        let [e1, e2] = self.electrons;
        let (s1, s2) = (self.system.stride(e1), self.system.stride(e2));
        (0..self.system.dimension())
            .map(|s| {
                let (d1, d2) = ((s / s1) % 2, (s / s2) % 2);
                s - d1 * s1 - d2 * s2 + d2 * s1 + d1 * s2
            })
            .collect()
    }

    /// The electron projector of a birth (`P_S`, `P_T₀`, `P_T₊`, `P_T₋`,
    /// `P_T`, or the identity for [`Birth::Uncorrelated`]) over the whole
    /// space.
    pub fn projector(&self, which: Birth) -> CMatrix {
        let d = self.system.dimension();
        let pi = self.swap();
        let [e1, e2] = self.electrons;
        let (s1, s2) = (self.system.stride(e1), self.system.stride(e2));
        // Levels 0 (α) and 1 (β) of the electrons.
        let both = |s: usize, level: usize| (s / s1) % 2 == level && (s / s2) % 2 == level;
        let half = C64::new(0.5, 0.0);
        let mut p = CMatrix::zeros(d, d);
        for s in 0..d {
            match which {
                Birth::Singlet => {
                    p[(s, s)] += half;
                    p[(pi[s], s)] -= half;
                }
                Birth::Triplet => {
                    p[(s, s)] += half;
                    p[(pi[s], s)] += half;
                }
                Birth::TripletZero => {
                    if !(both(s, 0) || both(s, 1)) {
                        p[(s, s)] += half;
                        p[(pi[s], s)] += half;
                    }
                }
                Birth::TripletPlus => {
                    if both(s, 0) {
                        p[(s, s)] = C64::new(1.0, 0.0);
                    }
                }
                Birth::TripletMinus => {
                    if both(s, 1) {
                        p[(s, s)] = C64::new(1.0, 0.0);
                    }
                }
                Birth::Uncorrelated => p[(s, s)] = C64::new(1.0, 0.0),
            }
        }
        p
    }

    /// `projector(birth)/Tr`: the electrons in that state, the nuclei
    /// maximally mixed.
    pub fn birth_state(&self, birth: Birth) -> CMatrix {
        let p = self.projector(birth);
        let trace: f64 = (0..p.nrows()).map(|s| p[(s, s)].re).sum();
        p.unscale(trace)
    }

    /// Yields, CIDNP and `X = ∫₀^∞ ρ dt` at the system's field, from
    /// `birth`.
    ///
    /// Errors: all rates zero; a start that never decays (see the module
    /// docs); the limits of work and memory; GMRES without convergence.
    pub fn solve(&self, birth: Birth) -> Result<PairSolution> {
        self.solve_with(
            &self.birth_state(birth),
            self.system.field(),
            Options::default(),
        )
    }

    /// The same from any start: Hermitian (to 10⁻¹² of its largest entry),
    /// finite, with a positive trace; it is normalised to trace 1.
    pub fn solve_from(&self, rho0: &CMatrix) -> Result<PairSolution> {
        let rho0 = self.normalised_start(rho0)?;
        self.solve_with(&rho0, self.system.field(), Options::default())
    }

    /// Yields and CIDNP at each field (T) from `birth`: the MARY curve. Each
    /// field is one solve (`X` is not kept); the fields run in order and the
    /// blocks of each in parallel.
    pub fn field_effect(&self, birth: Birth, fields: &[f64]) -> Result<Vec<PairYields>> {
        for &b in fields {
            check_field(b)?;
        }
        let rho0 = self.birth_state(birth);
        fields
            .iter()
            .map(|&b| {
                let (yields, _) = self.solve_at(&rho0, b, Options::default())?;
                Ok(yields)
            })
            .collect()
    }

    /// `ρ(t)` under the pair's equation of motion (see the module docs) at
    /// the system's field, from `rho` (any `d × d` matrix); the trace falls
    /// as the pair reacts. With all rates zero this is
    /// `system().evolve(rho, &[Event::Delay(t)])`. With Haberkorn's term and
    /// no jumps the blocks of `H` are exponentiated in Hilbert space;
    /// otherwise `rho` must keep the labels (no coherences between them),
    /// and the generator is exponentiated in a Liouville space of at most
    /// [`super::MAX_LIOUVILLE_BLOCK`] elements.
    pub fn evolve(&self, rho: &CMatrix, t: f64) -> Result<CMatrix> {
        let d = self.system.dimension();
        if rho.shape() != (d, d) {
            return Err(QsoError::DimensionMismatch {
                expected: d,
                actual: rho.nrows(),
            });
        }
        check_time("t", t)?;
        if t == 0.0 {
            return Ok(rho.clone());
        }
        self.evolve_at(rho, t, self.system.field())
    }

    /// `ρ₀` checked and normalised: `(ρ₀ + ρ₀†)/(2 Tr ρ₀)`.
    fn normalised_start(&self, rho0: &CMatrix) -> Result<CMatrix> {
        let d = self.system.dimension();
        if rho0.shape() != (d, d) {
            return Err(QsoError::DimensionMismatch {
                expected: d,
                actual: rho0.nrows(),
            });
        }
        if rho0.iter().any(|z| !(z.re.is_finite() && z.im.is_finite())) {
            return Err(QsoError::invalid("rho0", "the start must be finite"));
        }
        let largest = rho0.iter().map(|z| z.norm()).fold(0.0, f64::max);
        let skew = (0..d)
            .flat_map(|r| (0..d).map(move |c| (r, c)))
            .map(|(r, c)| (rho0[(r, c)] - rho0[(c, r)].conj()).norm())
            .fold(0.0, f64::max);
        if skew > 1e-12 * largest {
            return Err(QsoError::invalid("rho0", "the start must be Hermitian"));
        }
        let trace: f64 = (0..d).map(|s| rho0[(s, s)].re).sum();
        if trace <= 0.0 {
            return Err(QsoError::invalid(
                "rho0",
                format!("the start must have a positive trace, not {trace}"),
            ));
        }
        Ok((rho0 + rho0.adjoint()).unscale(2.0 * trace))
    }

    /// The solution from the normalised start `rho0` at field `b`.
    fn solve_with(&self, rho0: &CMatrix, b: f64, opts: Options) -> Result<PairSolution> {
        check_field(b)?;
        let (yields, x) = self.solve_at(rho0, b, opts)?;
        Ok(PairSolution {
            pair: self.clone().with_field(b)?,
            yields,
            x,
        })
    }

    /// The yields and observables of `X`.
    fn observe(&self, x: &CMatrix, b: f64, backward_error: f64, iterations: usize) -> PairYields {
        let pi = self.swap();
        let [ks, kt, ke] = self.rates;
        let d = x.nrows();
        let trace: f64 = (0..d).map(|s| x[(s, s)].re).sum();
        let swapped: f64 = (0..d).map(|s| x[(pi[s], s)].re).sum();
        let cidnp = self
            .nuclei()
            .into_iter()
            .map(|j| {
                let (mut diag, mut cross) = (0.0, 0.0);
                for s in 0..d {
                    let m = self.system.m(j, s);
                    diag += m * x[(s, s)].re;
                    cross += m * x[(pi[s], s)].re;
                }
                [
                    0.5 * ks * (diag - cross),
                    0.5 * kt * (diag + cross),
                    ke * diag,
                ]
            })
            .collect();
        PairYields {
            field: b,
            singlet: 0.5 * ks * (trace - swapped),
            triplet: 0.5 * kt * (trace + swapped),
            escape: ke * trace,
            cidnp,
            backward_error,
            iterations,
        }
    }
}

impl PairSolution {
    /// The yields and CIDNP.
    pub fn yields(&self) -> &PairYields {
        &self.yields
    }

    /// `X = ∫₀^∞ ρ dt` (s).
    pub fn integral(&self) -> &CMatrix {
        &self.x
    }

    /// `k_P P_P X` over the whole space (`(WX)[r, c] = X[π(r), c]`).
    fn weighted(&self, which: Product) -> CMatrix {
        let [ks, kt, ke] = self.pair.rates;
        let x = &self.x;
        match which {
            Product::Escape => x * C64::new(ke, 0.0),
            Product::Singlet | Product::Triplet => {
                let pi = self.pair.swap();
                let (k, sign) = if which == Product::Singlet {
                    (ks, -1.0)
                } else {
                    (kt, 1.0)
                };
                CMatrix::from_fn(x.nrows(), x.ncols(), |r, c| {
                    (x[(r, c)] + x[(pi[r], c)] * sign) * (0.5 * k)
                })
            }
        }
    }

    /// The yield of a product.
    fn yield_of(&self, which: Product) -> f64 {
        match which {
            Product::Singlet => self.yields.singlet,
            Product::Triplet => self.yields.triplet,
            Product::Escape => self.yields.escape,
        }
    }

    /// The product's nuclear state over [`RadicalPair::nuclei`] in their
    /// order (the basis of a [`SpinSystem`] of those nuclei), unnormalised:
    /// `k_P Tr_e(P_P X)`, Hermitian and positive semidefinite. Its trace is
    /// the product's yield; it is `1 × 1` for a pair without nuclei.
    pub fn product(&self, which: Product) -> CMatrix {
        let m = self.weighted(which);
        let sys = &self.pair.system;
        let [e1, e2] = self.pair.electrons;
        let (s1, s2) = (sys.stride(e1), sys.stride(e2));
        let d = sys.dimension();
        let n = d / 4;
        // Each state's nuclear index (the electrons' digits removed) and
        // the four states of each nuclear index by the electrons' levels.
        let mut states = vec![[0usize; 4]; n];
        for s in 0..d {
            let (l1, l2) = ((s / s1) % 2, (s / s2) % 2);
            let u = nuclear_index(s, s1, s2);
            states[u][2 * l1 + l2] = s;
        }
        let mut out = CMatrix::from_fn(n, n, |u, v| {
            (0..4)
                .map(|e| m[(states[u][e], states[v][e])])
                .fold(ZERO, |a, b| a + b)
        });
        hermitise(&mut out);
        out
    }

    /// The product's state on `target`, normalised: pair spin `i` becomes
    /// target spin `j` for each `(i, j)` of `spins` (as the routes of
    /// [`super::Route`]: the same isotope, each spin once); the pair's other
    /// spins are traced out, and the target's other spins are `fresh`
    /// (thermal at the target's field and temperature, or parahydrogen).
    /// Such a state goes on to [`SpinSystem::evolve`], or to
    /// [`super::Reactions::add_source`] as a product formed at a constant
    /// rate.
    ///
    /// Errors: those of a route (`"channel"`); an electron among `spins` or
    /// in `target`; a product of yield at most 10⁻¹⁴ (`"product"`).
    pub fn product_on(
        &self,
        which: Product,
        target: &SpinSystem,
        spins: &[(usize, usize)],
        fresh: Fresh,
    ) -> Result<CMatrix> {
        target.validate()?;
        check_no_electrons(target)?;
        if spins.iter().any(|&(i, _)| self.pair.electrons.contains(&i)) {
            return Err(QsoError::invalid(
                "channel",
                "the products are nuclear: electron spins are not carried",
            ));
        }
        let phi = self.yield_of(which);
        if phi.is_nan() || phi <= 1e-14 {
            return Err(QsoError::invalid(
                "product",
                "the pair forms no such product",
            ));
        }
        let m = self.weighted(which);
        let mut out = carry_state(&self.pair.system, target, spins, fresh, &m)?.unscale(phi);
        hermitise(&mut out);
        Ok(out)
    }
}

/// The index of state `s` with the two electrons' digits (strides `s1`,
/// `s2`, two levels each) removed: the nuclear basis index.
fn nuclear_index(s: usize, s1: usize, s2: usize) -> usize {
    let (hi, lo) = if s1 > s2 { (s1, s2) } else { (s2, s1) };
    // Digits below `lo`, between `lo` and `hi`, and above `hi`.
    let low = s % lo;
    let mid = (s / (2 * lo)) % (hi / (2 * lo));
    let top = s / (2 * hi);
    (top * (hi / (2 * lo)) + mid) * lo + low
}

/// `(M + M†)/2` in place.
fn hermitise(m: &mut CMatrix) {
    let n = m.nrows();
    for r in 0..n {
        for c in r..n {
            let v = (m[(r, c)] + m[(c, r)].conj()) * 0.5;
            m[(r, c)] = v;
            m[(c, r)] = v.conj();
        }
    }
}
