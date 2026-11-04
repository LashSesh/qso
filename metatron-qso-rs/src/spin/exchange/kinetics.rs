//! Spin dynamics with reaction kinetics: species whose amounts change.
//!
//! [`Reactions`] holds spin sites (a [`SpinSystem`] and its initial amount,
//! which may be zero), species without spins (such as parahydrogen or a
//! catalyst) and reactions of first and second order among them. The
//! amounts follow the rate equations `dc_i/dt = Σ_r ν_ir k_r Π c_reactants`,
//! independent of the spins, integrated by Dormand–Prince steps to 10⁻¹²
//! of each amount. Each site's density matrix `σ_s` is unnormalised, its
//! trace the site's amount, and
//!
//! ```text
//! dσ_s/dt = L_s σ_s − κ_s(t) σ_s + Σ_{r → s} g_r(t) ρ_fresh ⊗ Tr_traced(σ_source)
//! ```
//!
//! with `κ_s` the pseudo-first-order rate at which `s` reacts (`k_r` times
//! the amounts of the other reactants) and `g_r` the same for the source
//! of a reaction that forms `s` (or the whole rate `k_r Π c` times `ρ_fresh`
//! of every spin when no reactant carries spins). As in [`Exchange`], the
//! carried spins keep their state, the source's others are traced out and
//! the product's others arrive fresh (thermal, or two in the singlet of
//! parahydrogen).
//!
//! With [`Reactions::add_routed_reaction`], [`Route`]s carry spins from
//! any reactant site into any product site. A product site that takes
//! spins from both reactants of `A + B → C` forms in the product of their
//! reduced states (the reactants uncorrelated, as in a well-mixed
//! solution): `dσ_C/dt ∋ k Emb(σ_A' ⊗ σ_B' ⊗ ρ_fresh)`, quadratic in the
//! states. It is written exactly as transfers from each reactant with the
//! other's reference state `ρ̂ = 1/d + ŷ` in its spins (order zero, so the
//! blocks of coherence order stay apart), the drift `v(1/(d_A d_B) − ŷ_A ⊗
//! ŷ_B) ⊗ ρ_fresh` and the remainder `k Emb((x_A' − c_A ŷ_A) ⊗ (x_B' − c_B
//! ŷ_B) ⊗ ρ_fresh)`, which vanishes when either reactant is in its
//! reference state: with a partner held in its state the step is that of
//! a linear reaction. The references are each step's predicted midpoint,
//! the remainder enters the inhomogeneous quadratic, and the step is
//! corrected until it settles; steps whose remainder matters are checked
//! against their halves to [`BILINEAR_TOLERANCE`].
//!
//! The spin equations are linear with coefficients that follow the
//! amounts. They are propagated as the deviation `σ_s − c_s(t)/d_s` (its
//! trace stays zero, so `Tr σ_s = c_s` exactly) by exponentials of the
//! joint blocks with the coefficients at each step's midpoint and the
//! inhomogeneous terms as quadratics, extrapolated from steps `h` and `h/2`
//! (Richardson; the midpoint rule is symmetric). The fast spin evolution is
//! in the exponentials, and the steps follow the kinetics: the amount of
//! every reactant changes by at most [`STEP_FRACTION`] of itself in a step,
//! so steady chemistry, however fast, takes one step. A reaction that
//! carries spins from a source at a pseudo-rate that changes (the source's
//! partner is consumed) is the exception: the rate frozen at the midpoint
//! would miss the phase of the coherences it carries, so its steps also
//! resolve the frequencies of the blocks it feeds (`‖h M‖ ≤ 1/4`).

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::super::blocks::LiouvilleBlock;
use super::super::krylov::{
    Augmented, KRYLOV_TOLERANCE, SparseBlock, SparseFamily, expv, krylov_bytes, krylov_work,
};
use super::super::{Event, Isotope, SpinSystem, Target, ZERO, check_field, check_time};
use super::Propagation;
use super::{Channel, Exchange, Fresh, Site};
use crate::error::{QsoError, Result};
use crate::linalg::CMatrix;
use crate::linalg::dense::{expm, expm_squarings};
use num_complex::Complex64 as C64;

/// The largest relative change of a reactant's amount in one step:
/// `h ≤ STEP_FRACTION · c_i / |ċ_i|`.
pub const STEP_FRACTION: f64 = 0.02;

/// Most steps of a delay, and most Dormand–Prince steps of the amounts
/// over one interval (stiff kinetics beyond are refused).
pub const MAX_KINETIC_STEPS: usize = 1_000_000;

/// The largest `h‖M‖` of a step for the blocks fed by a reaction that
/// carries spins at a pseudo-rate that changes: the frozen rate then errs
/// as `h⁴` after extrapolation (a quarter gives 10⁻¹² of the coherence
/// carried).
const COHERENT_PHASE: f64 = 0.25;

/// Relative tolerance of the Dormand–Prince steps of the amounts.
const KINETIC_TOLERANCE: f64 = 1e-12;

/// The largest difference, relative to the deviation from `c_s/d_s` of all
/// sites (plus 10⁻⁶ of their amounts, below which rounding of the terms of
/// order `c/d` dominates), between a step of `h` and two of `h/2` (before
/// their extrapolation) when a bilinear source's remainder matters in it;
/// longer steps are repeated shorter.
pub const BILINEAR_TOLERANCE: f64 = 1e-9;

/// Most corrections of a step with bilinear sources; a step whose
/// corrections have not settled by then is repeated shorter.
const MAX_CORRECTIONS: usize = 8;

/// A species without spins and its initial amount.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Species {
    pub name: String,
    pub amount: f64,
}

/// A participant of a reaction: a spin site or a species without spins
/// (indices into the sites and the species).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Participant {
    Site(usize),
    Species(usize),
}

/// A reaction of first or second order: `reactants → products` at rate
/// `k Π c_reactants` (a reactant may appear twice). The first product that
/// is a spin site receives the `carried` spins of the first site among the
/// reactants (its source) and `fresh` spins for the rest; other product
/// sites receive thermal spins. [`Reactions::add_routed_reaction`] carries
/// spins from any reactant site into any product site instead.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reaction {
    pub reactants: Vec<Participant>,
    pub products: Vec<Participant>,
    /// Rate constant (s⁻¹ for the first order, per amount and second for
    /// the second).
    pub rate: f64,
    /// `(spin of the source, spin of the product site)` pairs carried over.
    #[serde(default)]
    pub carried: Vec<(usize, usize)>,
    #[serde(default)]
    pub fresh: Fresh,
}

/// Spins a reaction carries from one of its reactant sites into one of its
/// product sites, by their positions among the reaction's reactants and
/// products (see [`Reactions::add_routed_reaction`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Route {
    /// Position among the reactants (a spin site).
    pub reactant: usize,
    /// Position among the products (a spin site).
    pub product: usize,
    /// `(spin of the reactant, spin of the product)` pairs.
    pub spins: Vec<(usize, usize)>,
}

impl Reaction {
    /// The first spin site among the reactants and its position there.
    fn source(&self) -> Option<(usize, usize)> {
        self.reactants
            .iter()
            .enumerate()
            .find_map(|(i, p)| match p {
                Participant::Site(s) => Some((i, *s)),
                Participant::Species(_) => None,
            })
    }
}

/// Spin sites, species without spins and the reactions among them.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reactions {
    /// The spin sites; `concentration` is the initial amount (≥ 0).
    sites: Vec<Site>,
    #[serde(default)]
    species: Vec<Species>,
    #[serde(default)]
    reactions: Vec<Reaction>,
    /// The routes of every reaction (none: its `carried` spins), or empty
    /// when no reaction has routes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    routes: Vec<Vec<Route>>,
}

/// What a reaction brings into one of its product sites.
struct Feed {
    site: usize,
    /// `(position among the reactants, its site, spin pairs)`, by position:
    /// none (fresh spins only), one (a transfer, linear in that reactant's
    /// state) or two (the product of both reactants' states).
    routes: Vec<(usize, usize, Vec<(usize, usize)>)>,
    fresh: Fresh,
}

/// A reactant's state reduced to the spins it carries into a product.
struct Reduced {
    dim: usize,
    /// The reactant's basis states grouped by the levels of its other
    /// (traced) spins: `(state, index here)`.
    groups: Vec<Vec<(u32, u32)>>,
    /// Per index here, its part of the product's basis index.
    place: Vec<usize>,
    /// The elements `(u, u')` of coherence order zero, as `u·dim + u'`.
    zero: Vec<u32>,
}

impl Reduced {
    /// `Tr_traced(x)` of the reactant's matrix `x`.
    fn of(&self, x: &CMatrix) -> CMatrix {
        let mut out = CMatrix::zeros(self.dim, self.dim);
        for group in &self.groups {
            for &(a, u) in group {
                for &(a2, u2) in group {
                    out[(u as usize, u2 as usize)] += x[(a as usize, a2 as usize)];
                }
            }
        }
        out
    }

    /// The elements of order zero of `m`, the others zero.
    fn zero_part(&self, m: &CMatrix) -> CMatrix {
        let mut out = CMatrix::zeros(self.dim, self.dim);
        for &e in &self.zero {
            let (u, u2) = (e as usize / self.dim, e as usize % self.dim);
            out[(u, u2)] = m[(u, u2)];
        }
        out
    }
}

/// A product site that receives spins from both reactants: the bilinear
/// source `k Emb(σ_0' ⊗ σ_1' ⊗ ρ_fresh)` (`σ_q'` reactant `q`'s state
/// reduced to its carried spins), handled as transfers from each reactant
/// with the other's reference state in place of its spins, a drift and a
/// remainder (see [`Reactions::remainder`]).
struct Bilinear {
    r: usize,
    /// The reactants' sites (positions 0 and 1 of the reaction).
    sources: [usize; 2],
    reduced: [Reduced; 2],
    /// The fresh spins' state.
    fresh: CMatrix,
    /// Per element of the product: `(block, position, p, p')`.
    elements: Vec<(u32, u32, u32, u32)>,
    /// Per basis state of the product: its index in each reactant's reduced
    /// space and among the fresh spins.
    index: Vec<[u32; 3]>,
    /// The transfers from reactant `q` with the other's reference in place
    /// of its spins: `(block, row, column, partner element or [`TRACE`],
    /// weight)`, and
    /// once the family exists, the entries' places in its blocks
    /// `(block, index among the values, partner element, weight)`.
    skeleton: [Vec<(usize, usize, usize, u32, C64)>; 2],
    places: [Vec<(usize, usize, u32, C64)>; 2],
}

/// The partner element of a bilinear source's transfer entry that takes
/// the source's trace off the product (weight one, no partner).
///
/// Each transfer moves only the traceless part of its source's deviation
/// `x` (its trace is zero but for rounding): the product loses `Tr x` at
/// its first population. Without, the traces of the deviations would
/// follow `C → A + B` into both products while `C` loses them once, and
/// rounding errors would grow at the rates of the chemistry (as `e^{0.85
/// t}` in a test, `e^{850}` over one step of 1000 s of steady chemistry).
const TRACE: u32 = u32::MAX;

/// The reference states of the bilinear sources: per [`Bilinear`], each
/// reactant's normalised reduced state less `1/d`, order zero only.
type References = Vec<[CMatrix; 2]>;

/// Per joint block, the remainder of the bilinear sources (if any).
type Remainder = Vec<Option<Vec<C64>>>;

/// The base of the generator at one field: each site's evolution and
/// relaxation, and the transfers of the reactions.
struct Base {
    joint: super::Joint,
    /// Per joint block, the generator's parts on their shared pattern:
    /// each site's evolution and relaxation, then the members of
    /// [`Base::members`].
    family: Vec<SparseFamily>,
    /// What each member after the first is.
    members: Vec<Member>,
    /// Per joint block, the one-norm of the first member less its mean
    /// precession: a bound on the block's frequencies.
    norms: Vec<f64>,
    augmented: Vec<bool>,
    /// The relaxation's drift of each site per unit amount: `(block,
    /// position, value)`.
    drift: Vec<Vec<(usize, usize, C64)>>,
    /// Every element of each site: `(block, position, is a population)`.
    elements: Vec<Vec<(usize, usize, bool)>>,
    /// The transfers into product sites from one reactant each.
    linear: Vec<Linear>,
    /// Per reaction and product site without carried spins: the fresh
    /// state's entries per unit rate `(reaction, [(block, row, w)])`.
    fresh: Vec<(usize, Vec<(usize, usize, C64)>)>,
    /// The product sites with spins from both reactants.
    bilinear: Vec<Bilinear>,
}

/// A transfer of reaction `r` from its reactant at position `from` (site
/// `source`) into a product site: the entries `(block, row, column, w)`
/// and the drift per unit pseudo-rate and unit amount of the source.
struct Linear {
    r: usize,
    from: usize,
    source: usize,
    entries: Vec<(usize, usize, usize, C64)>,
    drifts: Vec<(usize, usize, C64)>,
}

/// A part of the generator that scales with the amounts.
#[derive(Clone, Copy)]
enum Member {
    /// `−1` on the elements of the site reacting as reactant `pos` of
    /// reaction `r` (times its pseudo-rate).
    Loss { r: usize, pos: usize },
    /// The transfer [`Base::linear`]`[k]` (times its pseudo-rate).
    Transfer { k: usize },
    /// The pattern of a bilinear source's transfers, whose values each
    /// step sets (weight zero).
    Pattern,
}

/// Per reaction, the largest frequency of the blocks its transfers feed
/// and the positions of the reactants they come from.
type Coherence = Vec<(f64, Vec<usize>)>;

/// One step of [`Reactions::midpoint`]: the states, the size of the
/// bilinear sources' remainder over the step (`h` times its largest norm at
/// the nodes), the operations, and whether the corrections converged.
struct Step {
    states: Vec<CMatrix>,
    remainder: f64,
    work: f64,
    converged: bool,
}

/// The columns of augmented coordinates and their own derivatives.
type Coordinates = (Vec<Vec<C64>>, Vec<Vec<C64>>);

/// One step's block: the generator with the amounts at the midpoint and,
/// for zero-order blocks, the inhomogeneous term's quadratic.
struct StepBlock {
    n: usize,
    a: Arc<SparseBlock>,
    /// `(f₀, f_m, f₁)` at the step's start, midpoint and end.
    drift: Option<[Vec<C64>; 3]>,
    h: f64,
}

impl StepBlock {
    /// The same block over the first half of the step: the generator and
    /// the inhomogeneous term's quadratic restricted to `[0, h/2]` (its
    /// values at `0`, `h/4` and `h/2`).
    fn first_half(&self) -> StepBlock {
        StepBlock {
            n: self.n,
            a: Arc::clone(&self.a),
            drift: self.drift.as_ref().map(|[f0, fm, f1]| {
                let quarter = (0..self.n)
                    .map(|i| f0[i] * 0.375 + fm[i] * 0.75 - f1[i] * 0.125)
                    .collect();
                [f0.clone(), quarter, fm.clone()]
            }),
            h: self.h / 2.0,
        }
    }

    /// Mean precession of the diagonal (zero for populations).
    fn shift(&self) -> C64 {
        if self.drift.is_some() || self.n == 0 {
            return ZERO;
        }
        let mean = (0..self.n).map(|i| self.a.get(i, i)).sum::<C64>() / self.n as f64;
        C64::new(0.0, mean.im)
    }

    /// The columns of the augmented coordinates `[u₁, u₂, 1]` divided by
    /// `scale`, and their own derivatives.
    fn augmented(&self, scale: f64) -> Option<Coordinates> {
        let [f0, fm, f1] = self.drift.as_ref()?;
        let w = 1.0 / scale;
        let u1 = (0..self.n)
            .map(|i| (fm[i] * 4.0 - f0[i] * 3.0 - f1[i]) * w)
            .collect();
        let u2 = (0..self.n)
            .map(|i| (f0[i] * 2.0 - fm[i] * 4.0 + f1[i] * 2.0) * 2.0 * w)
            .collect();
        let one = f0.iter().map(|v| v * w).collect();
        let r = C64::new(1.0 / self.h, 0.0);
        let coordinates = vec![vec![ZERO, ZERO, r], vec![r, ZERO, ZERO], vec![ZERO; 3]];
        Some((vec![u1, u2, one], coordinates))
    }

    /// Whether to exponentiate densely (by `method`, or whichever is
    /// cheaper), and the operations.
    fn plan(&self, method: Propagation) -> (bool, f64) {
        let extra = if self.drift.is_some() { 3 } else { 0 };
        let m = self.n + extra;
        // The inhomogeneous columns as `propagate` scales them (by at least
        // `h` times their size): at most 25/h a row, whatever the amounts'
        // unit.
        let drift_norm = self.drift.as_ref().map_or(0.0, |[f0, fm, f1]| {
            let unscaled = (0..self.n)
                .map(|i| 8.0 * (f0[i].norm() + fm[i].norm() + f1[i].norm()))
                .fold(0.0, f64::max);
            unscaled.min(25.0 / self.h)
        });
        let norm = self.a.norm_inf().max(self.a.norm_one()) + drift_norm + 1.0 / self.h;
        let krylov = krylov_work(m, self.a.nnz() + extra * self.n, norm, self.h);
        let dense = (self.n <= super::super::MAX_LIOUVILLE_BLOCK).then(|| {
            let mf = m as f64;
            (6.0 + f64::from(expm_squarings(norm * self.h))) * 6.0 * mf * mf * mf
        });
        match (method, dense) {
            (Propagation::Krylov, _) | (_, None) => (false, krylov),
            (Propagation::Dense, Some(d)) => (true, d),
            (Propagation::Auto, Some(d)) => {
                if d <= krylov {
                    (true, d)
                } else {
                    (false, krylov)
                }
            }
        }
    }

    /// The block's elements after the step from `x`.
    fn propagate(&self, x: &[C64], dense: bool) -> Result<Vec<C64>> {
        let norm2 = |v: &[C64]| v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        // The constant coordinate at the size of the deviation (or of what
        // the inhomogeneous term adds over the step).
        let scale = match &self.drift {
            None => norm2(x),
            Some([f0, fm, f1]) => norm2(x).max(self.h * norm2(f0).max(norm2(fm)).max(norm2(f1))),
        };
        if scale == 0.0 {
            return Ok(x.to_vec());
        }
        let n = self.n;
        if dense {
            let mat = self.a.to_dense();
            let Some((columns, coordinates)) = self.augmented(scale) else {
                let block = LiouvilleBlock::sized(n, mat, false, self.h);
                return Ok(block.apply(|i| x[i]));
            };
            // f(s) = a + b s + e s² through the three values, with the
            // dimensionless u₁ = s/h and u₂ = (s/h)²/2 (u₁' = 1/h,
            // u₂' = u₁/h): f = a + (b h) u₁ + (2 e h²) u₂, all entries of
            // the order of the deviation however fast f changes.
            let mut m = CMatrix::zeros(n + 3, n + 3);
            m.view_mut((0, 0), (n, n)).copy_from(&mat);
            for (c, column) in columns.iter().enumerate() {
                for (i, v) in column.iter().enumerate() {
                    m[(i, n + c)] = *v;
                }
            }
            for (r, row) in coordinates.iter().enumerate() {
                for (c, v) in row.iter().enumerate() {
                    m[(n + r, n + c)] = *v;
                }
            }
            let p = expm(&(m * C64::new(self.h, 0.0)));
            return Ok((0..n)
                .map(|i| {
                    let mut v = p[(i, n + 2)] * scale;
                    for (j, xj) in x.iter().enumerate() {
                        v += p[(i, j)] * xj;
                    }
                    v
                })
                .collect());
        }
        let shift = self.shift();
        let (columns, coordinates) = self.augmented(scale).unwrap_or_default();
        let op = Augmented::new(&self.a, shift, columns, coordinates);
        let mut v = x.to_vec();
        if self.drift.is_some() {
            v.extend([ZERO, ZERO, C64::new(scale, 0.0)]);
        }
        let tol = KRYLOV_TOLERANCE * norm2(&v);
        let mut y = expv(&op, &v, self.h, tol, 0.0)?;
        y.truncate(self.n);
        let phase = (shift * self.h).exp();
        if phase != C64::new(1.0, 0.0) {
            for z in &mut y {
                *z *= phase;
            }
        }
        Ok(y)
    }
}

impl Reactions {
    /// No sites.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a spin site with its initial amount (≥ 0); returns its index.
    pub fn add_site(&mut self, system: SpinSystem, amount: f64) -> Result<usize> {
        system.validate()?;
        check_amount(amount)?;
        self.sites.push(Site {
            system,
            concentration: amount,
        });
        Ok(self.sites.len() - 1)
    }

    /// Add a species without spins with its initial amount (≥ 0) and a
    /// name of its own; returns its index.
    pub fn add_species(&mut self, name: &str, amount: f64) -> Result<usize> {
        check_amount(amount)?;
        if self.species.iter().any(|s| s.name == name) {
            return Err(QsoError::invalid(
                "species",
                format!("a species is already named {name:?}"),
            ));
        }
        self.species.push(Species {
            name: name.to_string(),
            amount,
        });
        Ok(self.species.len() - 1)
    }

    /// Add a reaction (checked).
    pub fn add_reaction(&mut self, reaction: Reaction) -> Result<&mut Self> {
        self.check_reaction(&reaction, &[])?;
        self.reactions.push(reaction);
        if !self.routes.is_empty() {
            self.routes.push(Vec::new());
        }
        Ok(self)
    }

    /// Add a reaction whose `routes` carry spins from its reactant sites
    /// into its product sites (checked; its `carried` must be empty). Each
    /// product site receives the reduced states of the reactants whose
    /// spins it takes, as a product when it takes spins from both (the
    /// reactants uncorrelated, as in a well-mixed solution): `A + B → C`
    /// forms `C` in `ρ_A ⊗ ρ_B` at the rate `k[A][B]`, and `C → A + B`
    /// gives each product its partial trace of `C`. The spins of a product
    /// site that no route reaches arrive fresh: `fresh` on the first
    /// product site, thermal on another.
    pub fn add_routed_reaction(
        &mut self,
        reaction: Reaction,
        routes: Vec<Route>,
    ) -> Result<&mut Self> {
        self.check_reaction(&reaction, &routes)?;
        if !routes.is_empty() || !self.routes.is_empty() {
            self.routes.resize(self.reactions.len(), Vec::new());
            self.routes.push(routes);
        }
        self.reactions.push(reaction);
        Ok(self)
    }

    /// The routes of reaction `reaction` (empty for one added without).
    pub fn routes(&self, reaction: usize) -> &[Route] {
        self.routes.get(reaction).map_or(&[], Vec::as_slice)
    }

    pub fn sites(&self) -> &[Site] {
        &self.sites
    }

    pub fn species(&self) -> &[Species] {
        &self.species
    }

    pub fn reactions(&self) -> &[Reaction] {
        &self.reactions
    }

    /// From JSON (the fields of this struct), checked.
    pub fn from_json(text: &str) -> Result<Self> {
        let x: Reactions = serde_json::from_str(text)
            .map_err(|e| QsoError::invalid("json", format!("not a set of reactions: {e}")))?;
        x.validate()?;
        Ok(x)
    }

    /// As JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("reactions serialise")
    }

    /// The sites as an exchange without channels (unit concentrations), for
    /// its blocks and transfers.
    fn shadow(&self) -> Exchange {
        Exchange {
            sites: self
                .sites
                .iter()
                .map(|s| Site {
                    system: s.system.clone(),
                    concentration: 1.0,
                })
                .collect(),
            channels: Vec::new(),
        }
    }

    /// What reaction `r` brings into each of its product sites.
    fn feeds(&self, r: usize) -> Vec<Feed> {
        let reaction = &self.reactions[r];
        let routes = self.routes(r);
        let mut out: Vec<Feed> = Vec::new();
        for (p, part) in reaction.products.iter().enumerate() {
            let Participant::Site(site) = *part else {
                continue;
            };
            let first = out.is_empty();
            let mut into = Vec::new();
            if routes.is_empty() {
                if first
                    && !reaction.carried.is_empty()
                    && let Some((pos, s)) = reaction.source()
                {
                    into.push((pos, s, reaction.carried.clone()));
                }
            } else {
                for route in routes.iter().filter(|x| x.product == p) {
                    if let Participant::Site(s) = reaction.reactants[route.reactant] {
                        into.push((route.reactant, s, route.spins.clone()));
                    }
                }
                into.sort_by_key(|x| x.0);
            }
            out.push(Feed {
                site,
                routes: into,
                fresh: if first {
                    reaction.fresh
                } else {
                    Fresh::Thermal
                },
            });
        }
        out
    }

    /// Whether a product site takes spins from both reactants.
    fn has_bilinear(&self) -> bool {
        (0..self.reactions.len()).any(|r| self.feeds(r).iter().any(|f| f.routes.len() == 2))
    }

    fn check_reaction(&self, r: &Reaction, routes: &[Route]) -> Result<()> {
        let bad = |why: &str| Err(QsoError::invalid("reaction", why.to_string()));
        let exists = |p: &Participant| match *p {
            Participant::Site(s) => s < self.sites.len(),
            Participant::Species(s) => s < self.species.len(),
        };
        if r.reactants.is_empty() || r.reactants.len() > 2 {
            return bad("a reaction has one or two reactants (first or second order)");
        }
        if r.products.len() > 2 {
            return bad("a reaction has at most two products");
        }
        if !r.reactants.iter().chain(&r.products).all(exists) {
            return bad("a participant is out of range");
        }
        if !(r.rate.is_finite() && r.rate >= 0.0) {
            return bad("the rate must be finite and non-negative");
        }
        let site = |p: &Participant| match *p {
            Participant::Site(s) => Some(s),
            Participant::Species(_) => None,
        };
        let Some(first) = r.products.iter().position(|p| site(p).is_some()) else {
            if !r.carried.is_empty() || r.fresh != Fresh::Thermal || !routes.is_empty() {
                return bad("carried and fresh spins need a product with spins");
            }
            return Ok(());
        };
        if !routes.is_empty() && !r.carried.is_empty() {
            return bad("a reaction carries spins by its routes or by `carried`, not both");
        }
        // Every carried pair as (reactant position, product position,
        // spin of the reactant, spin of the product).
        let mut pairs: Vec<(usize, usize, usize, usize)> = Vec::new();
        if !r.carried.is_empty() {
            let Some((pos, _)) = r.source() else {
                return bad("carried spins need a reactant with spins");
            };
            pairs.extend(r.carried.iter().map(|&(i, j)| (pos, first, i, j)));
        }
        for (k, route) in routes.iter().enumerate() {
            let from = r.reactants.get(route.reactant).and_then(site);
            let to = r.products.get(route.product).and_then(site);
            if from.is_none() || to.is_none() {
                return bad("a route joins a reactant and a product that are spin sites");
            }
            if route.spins.is_empty() {
                return bad("a route carries at least one spin");
            }
            if routes[..k]
                .iter()
                .any(|o| o.reactant == route.reactant && o.product == route.product)
            {
                return bad("two routes join the same reactant and product (merge their spins)");
            }
            pairs.extend(
                route
                    .spins
                    .iter()
                    .map(|&(i, j)| (route.reactant, route.product, i, j)),
            );
        }
        let spins_of = |p: &Participant| site(p).map_or(0, |s| self.sites[s].system.spins().len());
        let mut from_seen: Vec<Vec<bool>> = r
            .reactants
            .iter()
            .map(|p| vec![false; spins_of(p)])
            .collect();
        let mut to_seen: Vec<Vec<bool>> = r
            .products
            .iter()
            .map(|p| vec![false; spins_of(p)])
            .collect();
        for &(rp, pp, i, j) in &pairs {
            let ok = i < from_seen[rp].len()
                && j < to_seen[pp].len()
                && !from_seen[rp][i]
                && !to_seen[pp][j];
            if !ok {
                return bad(&format!(
                    "the carried pair ({i}, {j}) is out of range or repeated"
                ));
            }
            from_seen[rp][i] = true;
            to_seen[pp][j] = true;
            let (a, b) = (
                &self.sites[site(&r.reactants[rp]).expect("a site")].system,
                &self.sites[site(&r.products[pp]).expect("a site")].system,
            );
            if a.spins()[i].isotope != b.spins()[j].isotope {
                return bad(&format!("carried spins {i} and {j} are different isotopes"));
            }
        }
        if let Fresh::Parahydrogen {
            spins: [i, j],
            fraction,
        } = r.fresh
        {
            let b = &self.sites[site(&r.products[first]).expect("a site")].system;
            let free = &to_seen[first];
            let ok = i != j
                && i < free.len()
                && j < free.len()
                && !free[i]
                && !free[j]
                && b.spins()[i].isotope == b.spins()[j].isotope
                && b.spins()[i].isotope.levels() == 2
                && (0.0..=1.0).contains(&fraction);
            if !ok {
                return bad(
                    "parahydrogen occupies two fresh spins ½ of one isotope of the first product \
                     site, fraction in [0, 1]",
                );
            }
        }
        Ok(())
    }

    /// Check the sites (one field, temperature, frame and carriers), the
    /// amounts and the reactions.
    pub fn validate(&self) -> Result<()> {
        if self.sites.is_empty() {
            return Err(QsoError::invalid("sites", "reactions need a spin site"));
        }
        self.shadow().validate()?;
        for s in &self.sites {
            check_amount(s.concentration)?;
        }
        for (k, s) in self.species.iter().enumerate() {
            check_amount(s.amount)?;
            if self.species[..k].iter().any(|t| t.name == s.name) {
                return Err(QsoError::invalid(
                    "species",
                    format!("two species are named {:?}", s.name),
                ));
            }
        }
        if !self.routes.is_empty() && self.routes.len() != self.reactions.len() {
            return Err(QsoError::invalid(
                "routes",
                format!(
                    "{} lists of routes for {} reactions",
                    self.routes.len(),
                    self.reactions.len()
                ),
            ));
        }
        for (k, r) in self.reactions.iter().enumerate() {
            self.check_reaction(r, self.routes(k))?;
        }
        Ok(())
    }

    /// The field (T) of the sites.
    pub fn field(&self) -> f64 {
        self.sites.first().map_or(0.0, |s| s.system.field())
    }

    /// The initial amounts: the sites', then the species'.
    fn initial(&self) -> Vec<f64> {
        self.sites
            .iter()
            .map(|s| s.concentration)
            .chain(self.species.iter().map(|s| s.amount))
            .collect()
    }

    fn index(&self, p: &Participant) -> usize {
        match *p {
            Participant::Site(s) => s,
            Participant::Species(s) => self.sites.len() + s,
        }
    }

    /// The rate `k Π c` of every reaction.
    fn velocities(&self, c: &[f64]) -> Vec<f64> {
        self.reactions
            .iter()
            .map(|r| {
                r.reactants
                    .iter()
                    .fold(r.rate, |v, p| v * c[self.index(p)].max(0.0))
            })
            .collect()
    }

    /// `dc/dt` of all amounts.
    fn derivative(&self, c: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; c.len()];
        for (r, v) in self.reactions.iter().zip(self.velocities(c)) {
            for p in &r.reactants {
                out[self.index(p)] -= v;
            }
            for p in &r.products {
                out[self.index(p)] += v;
            }
        }
        out
    }

    /// Advance the amounts from `t0` to `t1` by Dormand–Prince steps
    /// (tolerance 10⁻¹² of each amount, and of 10⁻⁶ of the largest for
    /// amounts near zero); refused beyond [`MAX_KINETIC_STEPS`] steps
    /// (stiff kinetics) and when the amounts overflow.
    fn advance(&self, c: &mut [f64], t0: f64, t1: f64) -> Result<()> {
        // Dormand–Prince 5(4) tableau.
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
        const E: [f64; 7] = [
            71.0 / 57600.0,
            0.0,
            -71.0 / 16695.0,
            71.0 / 1920.0,
            -17253.0 / 339200.0,
            22.0 / 525.0,
            -1.0 / 40.0,
        ];
        let scale = c
            .iter()
            .fold(0.0f64, |m, x| m.max(x.abs()))
            .max(f64::MIN_POSITIVE);
        let mut t = t0;
        let mut h = (t1 - t0) / 16.0;
        let n = c.len();
        let mut steps = 0usize;
        while t < t1 {
            steps += 1;
            if steps > MAX_KINETIC_STEPS {
                return Err(QsoError::invalid(
                    "reactions",
                    format!(
                        "the amounts need more than {MAX_KINETIC_STEPS} steps over {:.3e} s: the \
                         kinetics are stiff (rates far faster than the time); use slower rates \
                         or shorter times",
                        t1 - t0
                    ),
                ));
            }
            if t + h > t1 {
                h = t1 - t;
            }
            let mut k = vec![self.derivative(c)];
            for row in &A {
                let y: Vec<f64> = (0..n)
                    .map(|i| c[i] + h * k.iter().zip(row).map(|(ki, a)| a * ki[i]).sum::<f64>())
                    .collect();
                k.push(self.derivative(&y));
            }
            // The fifth-order solution is the last stage's argument.
            let next: Vec<f64> = (0..n)
                .map(|i| c[i] + h * k.iter().zip(&A[5]).map(|(ki, a)| a * ki[i]).sum::<f64>())
                .collect();
            if !next.iter().all(|x| x.is_finite()) {
                return Err(QsoError::invalid(
                    "reactions",
                    "the amounts overflow (rate constants times amounts beyond the range of \
                     floating point)",
                ));
            }
            let error = (0..n)
                .map(|i| {
                    let e = (h * k.iter().zip(&E).map(|(ki, e)| e * ki[i]).sum::<f64>()).abs();
                    e / (KINETIC_TOLERANCE * (c[i].abs().max(next[i].abs()) + 1e-6 * scale))
                })
                .fold(0.0, f64::max);
            if error <= 1.0 || h <= 1e-14 * (t1 - t0).abs().max(f64::MIN_POSITIVE) {
                t += h;
                c.copy_from_slice(&next);
            }
            let factor = if error == 0.0 {
                5.0
            } else {
                (0.9 * error.powf(-0.2)).clamp(0.2, 5.0)
            };
            h *= factor;
        }
        Ok(())
    }

    /// The amounts after `t` (s): the sites', then the species' (rounding
    /// below zero is cut off).
    pub fn amounts(&self, t: f64) -> Result<Vec<f64>> {
        self.validate()?;
        check_time("t", t)?;
        let mut c = self.initial();
        if t > 0.0 {
            self.advance(&mut c, 0.0, t)?;
        }
        Ok(c.into_iter().map(|x| x.max(0.0)).collect())
    }

    /// The same reactions starting with the amounts after `t` (s), to go on
    /// from states returned by [`Reactions::evolve`] after delays of `t`.
    /// The sites keep their own field: after `Field` or `Sweep` events, set
    /// the field again with an event at the start of the next `evolve`.
    pub fn advanced(&self, t: f64) -> Result<Self> {
        let c = self.amounts(t)?;
        let mut out = self.clone();
        let n = out.sites.len();
        for (site, &a) in out.sites.iter_mut().zip(&c) {
            site.concentration = a;
        }
        for (species, &a) in out.species.iter_mut().zip(&c[n..]) {
            species.amount = a;
        }
        Ok(out)
    }

    /// Thermal equilibrium of every site times its initial amount.
    pub fn thermal_state(&self) -> Result<Vec<CMatrix>> {
        self.validate()?;
        self.sites
            .iter()
            .map(|s| Ok(s.system.thermal_state()? * C64::new(s.concentration, 0.0)))
            .collect()
    }

    /// The generator's base at field `b`.
    fn base(&self, b: f64) -> Result<Base> {
        let shadow = self.shadow();
        let joint = shadow.joint(None);
        let parts = shadow.site_parts(b)?;
        // Count the entries first: the triplets, the family on their
        // pattern (one value array per member) and the Krylov workspaces.
        let mut entries = 0usize;
        for (s, (site, (sparse, relaxation))) in self.sites.iter().zip(&parts).enumerate() {
            site.system.generator_entries(
                &joint.liouvilles[s],
                sparse,
                relaxation,
                |_, _, _, _| entries += 1,
            );
        }
        let feeds: Vec<Vec<Feed>> = (0..self.reactions.len()).map(|r| self.feeds(r)).collect();
        let member_count = self
            .reactions
            .iter()
            .map(|r| {
                r.reactants
                    .iter()
                    .filter(|p| matches!(p, Participant::Site(_)))
                    .count()
            })
            .sum::<usize>()
            + feeds
                .iter()
                .flatten()
                .map(|f| f.routes.len().min(2))
                .sum::<usize>();
        let member_count = member_count as f64;
        let elements_total: usize = joint.blocks.iter().map(|(_, _, n)| n + 3).sum();
        let bytes = entries as f64 * (12.0 + 20.0 + 16.0 * (1.0 + member_count))
            + krylov_bytes(elements_total)
            + 16.0 * 6.0 * elements_total as f64;
        if bytes > super::super::MAX_RELAXATION_MEMORY {
            return Err(QsoError::invalid(
                "reactions",
                format!(
                    "the joint blocks need about {:.1} GB (limit {:.1} GB); use fewer spins or \
                     the rotating frame",
                    bytes / 1e9,
                    super::super::MAX_RELAXATION_MEMORY / 1e9
                ),
            ));
        }
        let augmented: Vec<bool> = joint
            .blocks
            .iter()
            .map(|(order, _, _)| order.iter().all(|&q| q == 0))
            .collect();
        let mut triplets: Vec<Vec<(u32, u32, C64)>> = vec![Vec::new(); joint.blocks.len()];
        let at = |s: usize, a: usize, a2: usize| -> Option<(usize, usize)> {
            let l = &joint.liouvilles[s];
            let k = l.block_of(a, a2)?;
            let (j, offset) = joint.place[s][k];
            Some((j, offset + l.position_of(a, a2)))
        };
        let mut drift = Vec::with_capacity(self.sites.len());
        let mut elements = Vec::with_capacity(self.sites.len());
        for (s, (site, (sparse, relaxation))) in self.sites.iter().zip(&parts).enumerate() {
            let sys = &site.system;
            sys.generator_entries(&joint.liouvilles[s], sparse, relaxation, |k, r, c, v| {
                let (j, offset) = joint.place[s][k];
                triplets[j].push(((offset + r) as u32, (offset + c) as u32, v));
            });
            let d = sys.dimension();
            let mut here = Vec::new();
            let mut all = Vec::new();
            for a in 0..d {
                for a2 in 0..d {
                    if let Some((j, i)) = at(s, a, a2) {
                        all.push((j, i, a == a2));
                        let v = relaxation.drift[(a, a2)];
                        if v != ZERO {
                            here.push((j, i, v));
                        }
                    }
                }
            }
            drift.push(here);
            elements.push(all);
        }
        let mut linear = Vec::new();
        let mut fresh = Vec::new();
        let mut bilinear = Vec::new();
        for (r, list) in feeds.iter().enumerate() {
            for feed in list {
                match feed.routes.as_slice() {
                    [] => {
                        // The fresh state of every spin times the rate.
                        let sys = &self.sites[feed.site].system;
                        let channel = Channel {
                            from: feed.site,
                            to: feed.site,
                            rate: 0.0,
                            carried: Vec::new(),
                            fresh: feed.fresh,
                        };
                        let (spins, rho) = shadow.fresh_state(&channel, b);
                        let levels: Vec<usize> = spins
                            .iter()
                            .map(|&j| sys.spins()[j].isotope.levels())
                            .collect();
                        let index = |f: usize| -> usize {
                            let mut rest = f;
                            let mut out = 0;
                            for k in (0..spins.len()).rev() {
                                out += (rest % levels[k]) * sys.stride(spins[k]);
                                rest /= levels[k];
                            }
                            out
                        };
                        let mut drifts = Vec::new();
                        for f in 0..rho.nrows() {
                            for f2 in 0..rho.ncols() {
                                let v = rho[(f, f2)];
                                if v != ZERO
                                    && let Some((j, row)) = at(feed.site, index(f), index(f2))
                                {
                                    drifts.push((j, row, v));
                                }
                            }
                        }
                        fresh.push((r, drifts));
                    }
                    [(from, source, spins)] => {
                        let channel = Channel {
                            from: *source,
                            to: feed.site,
                            rate: 0.0,
                            carried: spins.clone(),
                            fresh: feed.fresh,
                        };
                        let da = self.sites[*source].system.dimension() as f64;
                        let mut entries = Vec::new();
                        let mut drifts = Vec::new();
                        shadow.transfer(&channel, b, |a, a2, t1, t2, w| {
                            let (Some((j, col)), Some((j2, row))) =
                                (at(*source, a, a2), at(feed.site, t1, t2))
                            else {
                                return;
                            };
                            debug_assert_eq!(j, j2, "reactions keep the coherence order");
                            entries.push((j, row, col, w));
                            if a == a2 {
                                drifts.push((j, row, w / da));
                            }
                        });
                        // Only the traceless part of the source moves (see
                        // `traceless`).
                        let (j0, row0) = at(feed.site, 0, 0).expect("a population");
                        for &(j, col, population) in &elements[*source] {
                            if population {
                                debug_assert_eq!(j, j0);
                                entries.push((j, row0, col, C64::new(-1.0, 0.0)));
                            }
                        }
                        linear.push(Linear {
                            r,
                            from: *from,
                            source: *source,
                            entries,
                            drifts,
                        });
                    }
                    _ => bilinear.push(self.bilinear(&shadow, &joint, r, feed, b)),
                }
            }
        }
        // The members: losses of every site reactant, the transfers, and
        // the patterns of the bilinear sources' transfers; each block on
        // the union of the patterns.
        let mut members = Vec::new();
        let mut parts: Vec<Vec<Vec<(u32, u32, C64)>>> = Vec::new();
        for (r, reaction) in self.reactions.iter().enumerate() {
            for (pos, p) in reaction.reactants.iter().enumerate() {
                let Participant::Site(s) = *p else { continue };
                let mut per = vec![Vec::new(); joint.blocks.len()];
                for &(j, i, _) in &elements[s] {
                    per[j].push((i as u32, i as u32, C64::new(-1.0, 0.0)));
                }
                members.push(Member::Loss { r, pos });
                parts.push(per);
            }
        }
        for (k, l) in linear.iter().enumerate() {
            let mut per = vec![Vec::new(); joint.blocks.len()];
            for &(j, row, col, w) in &l.entries {
                per[j].push((row as u32, col as u32, w));
            }
            members.push(Member::Transfer { k });
            parts.push(per);
        }
        for feed in &bilinear {
            for skeleton in &feed.skeleton {
                let mut per = vec![Vec::new(); joint.blocks.len()];
                for &(j, row, col, _, _) in skeleton {
                    per[j].push((row as u32, col as u32, C64::new(1.0, 0.0)));
                }
                members.push(Member::Pattern);
                parts.push(per);
            }
        }
        // Again with the members' entries, which the shared pattern holds
        // as well.
        let extra: usize = parts
            .iter()
            .map(|per| per.iter().map(Vec::len).sum::<usize>())
            .sum();
        let full = (entries + extra) as f64 * (12.0 + 20.0 + 16.0 * (1.0 + member_count))
            + krylov_bytes(elements_total)
            + 16.0 * 6.0 * elements_total as f64;
        if full > super::super::MAX_RELAXATION_MEMORY {
            return Err(QsoError::invalid(
                "reactions",
                format!(
                    "the joint blocks need about {:.1} GB (limit {:.1} GB); use fewer spins or \
                     the rotating frame",
                    full / 1e9,
                    super::super::MAX_RELAXATION_MEMORY / 1e9
                ),
            ));
        }
        let mut norms = Vec::with_capacity(joint.blocks.len());
        let family: Vec<SparseFamily> = triplets
            .into_iter()
            .enumerate()
            .map(|(j, t)| {
                let n = joint.blocks[j].2;
                let first = SparseBlock::from_triplets(n, t);
                norms.push(precession_norm(&first));
                let mut blocks = vec![first];
                for per in &mut parts {
                    blocks.push(SparseBlock::from_triplets(n, std::mem::take(&mut per[j])));
                }
                SparseFamily::new(&blocks)
            })
            .collect();
        // Where the bilinear sources' transfers sit among the values of
        // each combined block.
        let mut probes: Vec<Option<SparseBlock>> = vec![None; family.len()];
        let zeros = vec![0.0; members.len() + 1];
        for feed in &mut bilinear {
            for q in 0..2 {
                feed.places[q] = feed.skeleton[q]
                    .iter()
                    .map(|&(j, row, col, e, w)| {
                        let probe = probes[j].get_or_insert_with(|| family[j].combine(&zeros));
                        let at = probe.position(row, col).expect("the pattern holds it");
                        (j, at, e, w)
                    })
                    .collect();
            }
        }
        Ok(Base {
            joint,
            family,
            members,
            norms,
            augmented,
            drift,
            elements,
            linear,
            fresh,
            bilinear,
        })
    }

    /// The bilinear source of reaction `r` into the product site of `feed`
    /// (routes from both reactants) on `joint` at field `b`.
    fn bilinear(
        &self,
        shadow: &Exchange,
        joint: &super::Joint,
        r: usize,
        feed: &Feed,
        b: f64,
    ) -> Bilinear {
        let at = |s: usize, a: usize, a2: usize| -> Option<(usize, usize)> {
            let l = &joint.liouvilles[s];
            let k = l.block_of(a, a2)?;
            let (j, offset) = joint.place[s][k];
            Some((j, offset + l.position_of(a, a2)))
        };
        let product = &self.sites[feed.site].system;
        let dp = product.dimension();
        let level = |sys: &SpinSystem, i: usize, state: usize| {
            (state / sys.stride(i)) % sys.spins()[i].isotope.levels()
        };
        // The levels of `spins` in `state` as one index, the first spin
        // most significant.
        let digits = |sys: &SpinSystem, spins: &[usize], state: usize| -> usize {
            spins.iter().fold(0, |acc, &i| {
                acc * sys.spins()[i].isotope.levels() + level(sys, i, state)
            })
        };
        let lab = product.frame() == super::super::Frame::Laboratory;
        let targets: [Vec<usize>; 2] =
            [0, 1].map(|q| feed.routes[q].2.iter().map(|&(_, j)| j).collect());
        let reduced: [Reduced; 2] = [0, 1].map(|q| {
            let (_, source, spins) = &feed.routes[q];
            let src = &self.sites[*source].system;
            let from: Vec<usize> = spins.iter().map(|&(i, _)| i).collect();
            let traced: Vec<usize> = (0..src.spins().len())
                .filter(|i| !from.contains(i))
                .collect();
            let levels: Vec<usize> = from
                .iter()
                .map(|&i| src.spins()[i].isotope.levels())
                .collect();
            let dim: usize = levels.iter().product();
            let traced_dim: usize = traced
                .iter()
                .map(|&i| src.spins()[i].isotope.levels())
                .product();
            let mut groups = vec![Vec::new(); traced_dim];
            for a in 0..src.dimension() {
                groups[digits(src, &traced, a)].push((a as u32, digits(src, &from, a) as u32));
            }
            let split = |u: usize| -> Vec<usize> {
                let mut out = vec![0; levels.len()];
                let mut rest = u;
                for k in (0..levels.len()).rev() {
                    out[k] = rest % levels[k];
                    rest /= levels[k];
                }
                out
            };
            let place = (0..dim)
                .map(|u| {
                    split(u)
                        .iter()
                        .zip(&targets[q])
                        .map(|(&l, &j)| l * product.stride(j))
                        .sum()
                })
                .collect();
            // Order zero: the levels sum the same per isotope (rotating
            // frame) or over all spins (laboratory frame).
            let isotopes: Vec<_> = from.iter().map(|&i| src.spins()[i].isotope).collect();
            let mut zero = Vec::new();
            for u in 0..dim {
                let du = split(u);
                for u2 in 0..dim {
                    let du2 = split(u2);
                    let sum = |d: &[usize], x: Option<Isotope>| -> usize {
                        d.iter()
                            .zip(&isotopes)
                            .filter(|&(_, y)| x.is_none_or(|x| x == *y))
                            .map(|(l, _)| l)
                            .sum()
                    };
                    let same = if lab {
                        sum(&du, None) == sum(&du2, None)
                    } else {
                        isotopes
                            .iter()
                            .all(|&x| sum(&du, Some(x)) == sum(&du2, Some(x)))
                    };
                    if same {
                        zero.push((u * dim + u2) as u32);
                    }
                }
            }
            Reduced {
                dim,
                groups,
                place,
                zero,
            }
        });
        // The fresh spins: those no route reaches.
        let channel = Channel {
            from: feed.site,
            to: feed.site,
            rate: 0.0,
            carried: targets.iter().flatten().map(|&j| (j, j)).collect(),
            fresh: feed.fresh,
        };
        let (fresh_spins, fresh) = shadow.fresh_state(&channel, b);
        let fresh_place: Vec<usize> = (0..fresh.nrows())
            .map(|f| {
                let mut rest = f;
                let mut out = 0;
                for k in (0..fresh_spins.len()).rev() {
                    let levels = product.spins()[fresh_spins[k]].isotope.levels();
                    out += (rest % levels) * product.stride(fresh_spins[k]);
                    rest /= levels;
                }
                out
            })
            .collect();
        let index: Vec<[u32; 3]> = (0..dp)
            .map(|p| {
                [
                    digits(product, &targets[0], p) as u32,
                    digits(product, &targets[1], p) as u32,
                    digits(product, &fresh_spins, p) as u32,
                ]
            })
            .collect();
        let mut elements = Vec::new();
        for p in 0..dp {
            for p2 in 0..dp {
                if let Some((j, i)) = at(feed.site, p, p2) {
                    elements.push((j as u32, i as u32, p as u32, p2 as u32));
                }
            }
        }
        let skeleton = [0, 1].map(|q| {
            let source = feed.routes[q].1;
            let (own, other) = (&reduced[q], &reduced[1 - q]);
            let mut out = Vec::new();
            for group in &own.groups {
                for &(a, u) in group {
                    for &(a2, u2) in group {
                        let Some((j, col)) = at(source, a as usize, a2 as usize) else {
                            continue;
                        };
                        for &e in &other.zero {
                            let (w, w2) = (e as usize / other.dim, e as usize % other.dim);
                            for f in 0..fresh.nrows() {
                                for f2 in 0..fresh.ncols() {
                                    let v = fresh[(f, f2)];
                                    if v == ZERO {
                                        continue;
                                    }
                                    let p = own.place[u as usize] + other.place[w] + fresh_place[f];
                                    let p2 =
                                        own.place[u2 as usize] + other.place[w2] + fresh_place[f2];
                                    if let Some((j2, row)) = at(feed.site, p, p2) {
                                        debug_assert_eq!(j, j2, "reactions keep the order");
                                        out.push((j, row, col, e, v));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Only the traceless part of the source moves.
            let (j0, row0) = at(feed.site, 0, 0).expect("a population");
            let src = &self.sites[source].system;
            for a in 0..src.dimension() {
                if let Some((j, col)) = at(source, a, a) {
                    debug_assert_eq!(j, j0);
                    out.push((j, row0, col, TRACE, C64::new(-1.0, 0.0)));
                }
            }
            out
        });
        Bilinear {
            r,
            sources: [feed.routes[0].1, feed.routes[1].1],
            reduced,
            fresh,
            elements,
            index,
            skeleton,
            places: [Vec::new(), Vec::new()],
        }
    }

    /// The part of the generator that acts on the deviation, with the
    /// amounts `c` and the bilinear sources' references `refs` (each block
    /// `n × n`).
    fn matrix(&self, base: &Base, c: &[f64], refs: &References) -> Vec<SparseBlock> {
        let mut w = vec![1.0];
        for m in &base.members {
            w.push(match *m {
                Member::Loss { r, pos } => self.pseudo_rate(&self.reactions[r], pos, c),
                Member::Transfer { k } => {
                    let l = &base.linear[k];
                    self.pseudo_rate(&self.reactions[l.r], l.from, c)
                }
                Member::Pattern => 0.0,
            });
        }
        let mut blocks: Vec<SparseBlock> = base.family.iter().map(|f| f.combine(&w)).collect();
        // Each bilinear source's transfer from reactant q at the pseudo-rate
        // k·c_other, with the other's reference `1/d + ŷ` in its spins.
        for (feed, refs) in base.bilinear.iter().zip(refs) {
            let reaction = &self.reactions[feed.r];
            for q in 0..2 {
                let kappa = self.pseudo_rate(reaction, q, c);
                if kappa == 0.0 {
                    continue;
                }
                let (other, y) = (&feed.reduced[1 - q], &refs[1 - q]);
                let flat = 1.0 / other.dim as f64;
                for &(j, at, e, weight) in &feed.places[q] {
                    let rho = if e == TRACE {
                        C64::new(1.0, 0.0)
                    } else {
                        let (u, u2) = (e as usize / other.dim, e as usize % other.dim);
                        let mut rho = y[(u, u2)];
                        if u == u2 {
                            rho += flat;
                        }
                        rho
                    };
                    blocks[j].values_mut()[at] += rho * weight * kappa;
                }
            }
        }
        blocks
    }

    /// `k_r` times the amounts of the reactants other than the one at
    /// `pos`.
    fn pseudo_rate(&self, reaction: &Reaction, pos: usize, c: &[f64]) -> f64 {
        reaction
            .reactants
            .iter()
            .enumerate()
            .filter(|&(q, _)| q != pos)
            .fold(reaction.rate, |v, (_, x)| v * c[self.index(x)].max(0.0))
    }

    /// The inhomogeneous term of the deviation's equation with the amounts
    /// `c`, per block (populations and zero-quantum coherences only):
    /// the relaxation's drift, the losses and gains of `c_s/d_s`, the fresh
    /// states of reactions without carried spins, the bilinear sources'
    /// `v(1/(d₀d₁) − ŷ₀ ⊗ ŷ₁) ⊗ ρ_fresh`, and `−ċ_s/d_s`.
    fn drift(&self, base: &Base, c: &[f64], refs: &References) -> Vec<Vec<C64>> {
        let mut f: Vec<Vec<C64>> = base
            .joint
            .blocks
            .iter()
            .zip(&base.augmented)
            .map(|((_, _, n), &a)| vec![ZERO; if a { *n } else { 0 }])
            .collect();
        let dims: Vec<f64> = self
            .sites
            .iter()
            .map(|s| s.system.dimension() as f64)
            .collect();
        let rate = self.derivative(c);
        let mut push = |j: usize, i: usize, v: C64| {
            if base.augmented[j] {
                f[j][i] += v;
            }
        };
        for (s, drift) in base.drift.iter().enumerate() {
            for &(j, i, v) in drift {
                push(j, i, v * c[s]);
            }
            let fall = -rate[s] / dims[s];
            if fall != 0.0 {
                for &(j, i, population) in &base.elements[s] {
                    if population {
                        push(j, i, C64::new(fall, 0.0));
                    }
                }
            }
        }
        let velocities = self.velocities(c);
        for reaction in &self.reactions {
            for (pos, p) in reaction.reactants.iter().enumerate() {
                let Participant::Site(s) = *p else { continue };
                let kappa = self.pseudo_rate(reaction, pos, c);
                if kappa == 0.0 {
                    continue;
                }
                for &(j, i, population) in &base.elements[s] {
                    if population {
                        push(j, i, C64::new(-kappa * c[s] / dims[s], 0.0));
                    }
                }
            }
        }
        for l in &base.linear {
            let g = self.pseudo_rate(&self.reactions[l.r], l.from, c);
            for &(j, row, w) in &l.drifts {
                push(j, row, w * g * c[l.source]);
            }
        }
        for (r, drifts) in &base.fresh {
            for &(j, row, w) in drifts {
                push(j, row, w * velocities[*r]);
            }
        }
        for (feed, refs) in base.bilinear.iter().zip(refs) {
            let v = velocities[feed.r];
            if v == 0.0 {
                continue;
            }
            let flat = 1.0 / (feed.reduced[0].dim * feed.reduced[1].dim) as f64;
            for &(j, i, p, p2) in &feed.elements {
                let j = j as usize;
                if !base.augmented[j] {
                    continue;
                }
                let ([u0, u1, x], [w0, w1, x2]) = (feed.index[p as usize], feed.index[p2 as usize]);
                let fv = feed.fresh[(x as usize, x2 as usize)];
                if fv == ZERO {
                    continue;
                }
                let mut val =
                    -refs[0][(u0 as usize, w0 as usize)] * refs[1][(u1 as usize, w1 as usize)];
                if u0 == w0 && u1 == w1 {
                    val += flat;
                }
                push(j, i as usize, val * fv * v);
            }
        }
        f
    }

    /// The bilinear sources' references at the deviations `x` with the
    /// amounts `c`: each reactant's reduced state normalised, less `1/d`,
    /// order zero only (zero for a reactant without amount).
    fn references(&self, base: &Base, x: &[CMatrix], c: &[f64]) -> References {
        let total: f64 = c[..self.sites.len()].iter().sum();
        base.bilinear
            .iter()
            .map(|feed| {
                [0, 1].map(|q| {
                    let (s, red) = (feed.sources[q], &feed.reduced[q]);
                    if c[s] <= 1e-12 * total {
                        return CMatrix::zeros(red.dim, red.dim);
                    }
                    red.zero_part(&red.of(&x[s])) * C64::new(1.0 / c[s], 0.0)
                })
            })
            .collect()
    }

    /// The remainder of the bilinear sources at the deviations `x` with the
    /// amounts `c` and references `refs`: `k Emb((x₀' − c₀ŷ₀) ⊗ (x₁' −
    /// c₁ŷ₁) ⊗ ρ_fresh)`, all orders; with the transfers and drift it gives
    /// the source exactly, and it vanishes when either reactant is in its
    /// reference state. Per block (blocks it misses `None`), and its 2-norm.
    fn remainder(
        &self,
        base: &Base,
        x: &[CMatrix],
        c: &[f64],
        refs: &References,
    ) -> (Remainder, f64) {
        let mut out: Remainder = vec![None; base.joint.blocks.len()];
        for (feed, refs) in base.bilinear.iter().zip(refs) {
            let k = self.reactions[feed.r].rate;
            let z = [0, 1].map(|q| {
                let s = feed.sources[q];
                feed.reduced[q].of(&x[s]) - &refs[q] * C64::new(c[s], 0.0)
            });
            if k == 0.0 || z.iter().any(|m| m.iter().all(|v| *v == ZERO)) {
                continue;
            }
            for &(j, i, p, p2) in &feed.elements {
                let ([u0, u1, f], [w0, w1, f2]) = (feed.index[p as usize], feed.index[p2 as usize]);
                let fv = feed.fresh[(f as usize, f2 as usize)];
                if fv == ZERO {
                    continue;
                }
                let v =
                    z[0][(u0 as usize, w0 as usize)] * z[1][(u1 as usize, w1 as usize)] * fv * k;
                if v != ZERO {
                    let n = base.joint.blocks[j as usize].2;
                    out[j as usize].get_or_insert_with(|| vec![ZERO; n])[i as usize] += v;
                }
            }
        }
        let norm = out
            .iter()
            .flatten()
            .flatten()
            .map(|v| v.norm_sqr())
            .sum::<f64>()
            .sqrt();
        (out, norm)
    }

    /// The blocks of one step of `h`: the generator with the amounts at the
    /// midpoint `nodes[1]`, and the inhomogeneous term (with the bilinear
    /// sources' remainders `rems` at the nodes) interpolated by the
    /// quadratic through its values at the start, midpoint and end,
    /// integrated exactly through two more coordinates `u₁ = s/h`,
    /// `u₂ = (s/h)²/2` (the φ-functions of the exponential).
    fn step_blocks(
        &self,
        base: &Base,
        nodes: [&[f64]; 3],
        h: f64,
        refs: &References,
        rems: Option<&[Remainder; 3]>,
    ) -> Vec<StepBlock> {
        let matrices = self.matrix(base, nodes[1], refs);
        let mut f = nodes.map(|c| self.drift(base, c, refs).into_iter());
        base.joint
            .blocks
            .iter()
            .zip(matrices)
            .enumerate()
            .map(|(j, ((_, _, n), a))| {
                let mut values = [0, 1, 2].map(|k| f[k].next().expect("a block"));
                let fed = rems.is_some_and(|r| r.iter().any(|x| x[j].is_some()));
                if fed {
                    for (k, v) in values.iter_mut().enumerate() {
                        v.resize(*n, ZERO);
                        if let Some(add) = rems.and_then(|r| r[k][j].as_ref()) {
                            for (o, a) in v.iter_mut().zip(add) {
                                *o += a;
                            }
                        }
                    }
                }
                StepBlock {
                    n: *n,
                    a: Arc::new(a),
                    drift: (base.augmented[j] || fed).then_some(values),
                    h,
                }
            })
            .collect()
    }

    /// The states after propagating each block of `blocks` from `x`, and
    /// the operations it took.
    fn apply(
        &self,
        base: &Base,
        blocks: &[StepBlock],
        x: &[CMatrix],
        method: Propagation,
    ) -> Result<(Vec<CMatrix>, f64)> {
        let mut out: Vec<CMatrix> = x
            .iter()
            .map(|m| CMatrix::zeros(m.nrows(), m.ncols()))
            .collect();
        let mut work = 0.0;
        for ((_, parts, n), block) in base.joint.blocks.iter().zip(blocks) {
            let mut elements: Vec<(usize, usize, usize)> = Vec::with_capacity(*n);
            for &(s, k, _) in parts {
                for &(a, b) in &base.joint.liouvilles[s].blocks[k].1 {
                    elements.push((s, a, b));
                }
            }
            let v: Vec<C64> = elements.iter().map(|&(s, a, b)| x[s][(a, b)]).collect();
            let (dense, cost) = block.plan(method);
            work += cost;
            let moved = block.propagate(&v, dense)?;
            for (&(s, a, b), v) in elements.iter().zip(moved) {
                out[s][(a, b)] = v;
            }
        }
        Ok((out, work))
    }

    /// One step of `h` from `t` (amounts `c` at `t`). With bilinear sources,
    /// a predictor (the references of the step's start, the remainder held
    /// at its start's value) gives the states at the midpoint and the end;
    /// the step then takes the references of the predicted midpoint and the
    /// remainder at the three nodes, and is repeated with the corrected
    /// states until the end moves by less than a tenth of `tolerance`
    /// (each pass squares the error of the last: the remainder is the
    /// product of two deviations from the references), at most
    /// [`MAX_CORRECTIONS`] times.
    #[allow(clippy::too_many_arguments)]
    fn midpoint(
        &self,
        base: &Base,
        x: &[CMatrix],
        c: &[f64],
        t: f64,
        h: f64,
        method: Propagation,
        tolerance: f64,
    ) -> Result<Step> {
        let mut mid = c.to_vec();
        self.advance(&mut mid, t, t + h / 2.0)?;
        let mut end = mid.clone();
        self.advance(&mut end, t + h / 2.0, t + h)?;
        let nodes = [c, mid.as_slice(), end.as_slice()];
        if base.bilinear.is_empty() {
            let blocks = self.step_blocks(base, nodes, h, &Vec::new(), None);
            let (states, work) = self.apply(base, &blocks, x, method)?;
            return Ok(Step {
                states,
                remainder: 0.0,
                work,
                converged: true,
            });
        }
        let norm = |m: &[CMatrix]| m.iter().map(|x| x.norm_squared()).sum::<f64>().sqrt();
        let start = self.references(base, x, c);
        let (held, _) = self.remainder(base, x, c, &start);
        let held = [held.clone(), held.clone(), held];
        let blocks = self.step_blocks(base, nodes, h, &start, Some(&held));
        let (mut end_guess, mut work) = self.apply(base, &blocks, x, method)?;
        let halves: Vec<StepBlock> = blocks.iter().map(StepBlock::first_half).collect();
        let (mut mid_guess, w) = self.apply(base, &halves, x, method)?;
        work += w;
        for pass in 1..=MAX_CORRECTIONS {
            let refs = self.references(base, &mid_guess, &mid);
            let (r0, n0) = self.remainder(base, x, c, &refs);
            let (rm, nm) = self.remainder(base, &mid_guess, &mid, &refs);
            let (r1, n1) = self.remainder(base, &end_guess, &end, &refs);
            let blocks = self.step_blocks(base, nodes, h, &refs, Some(&[r0, rm, r1]));
            let (states, w) = self.apply(base, &blocks, x, method)?;
            work += w;
            let moved: Vec<CMatrix> = states.iter().zip(&end_guess).map(|(a, b)| a - b).collect();
            let converged = norm(&moved) <= 0.1 * tolerance;
            if converged || pass == MAX_CORRECTIONS {
                return Ok(Step {
                    states,
                    remainder: h * n0.max(nm).max(n1),
                    work,
                    converged,
                });
            }
            let halves: Vec<StepBlock> = blocks.iter().map(StepBlock::first_half).collect();
            let (m, w) = self.apply(base, &halves, x, method)?;
            work += w;
            mid_guess = m;
            end_guess = states;
        }
        unreachable!("the last pass returns")
    }

    /// Per reaction: a bound on the frequencies of the blocks its
    /// transfers feed (the one-norm of the block without its mean phase,
    /// zero without transfers), and the positions of the reactants the
    /// transfers come from.
    fn coherence(&self, base: &Base) -> Coherence {
        let mut out: Coherence = vec![(0.0, Vec::new()); self.reactions.len()];
        let mut note = |r: usize, bound: f64, from: &[usize]| {
            if bound > 0.0 {
                out[r].0 = out[r].0.max(bound);
                for &q in from {
                    if !out[r].1.contains(&q) {
                        out[r].1.push(q);
                    }
                }
            }
        };
        for l in &base.linear {
            let bound = l
                .entries
                .iter()
                .map(|&(j, _, _, _)| base.norms[j])
                .fold(0.0, f64::max);
            note(l.r, bound, &[l.from]);
        }
        for feed in &base.bilinear {
            let bound = feed
                .skeleton
                .iter()
                .flatten()
                .map(|&(j, _, _, _, _)| base.norms[j])
                .fold(0.0, f64::max);
            note(feed.r, bound, &[0, 1]);
        }
        out
    }

    /// The step from `t` with amounts `c`, at most `left`: for every
    /// reaction that still matters (more than 10⁻¹⁴ of the amounts over the
    /// rest of the delay) the amounts of its reactants change by at most
    /// [`STEP_FRACTION`] of themselves, so its rate changes by about that
    /// fraction; a reaction that carries spins at a pseudo-rate that
    /// changes also keeps `h` within `COHERENT_PHASE / coherence` (see
    /// [`Self::coherence`]).
    fn step_size(&self, c: &[f64], left: f64, coherence: &Coherence) -> f64 {
        let total: f64 = c.iter().map(|x| x.abs()).sum();
        let velocities = self.velocities(c);
        if total <= 0.0 {
            return left;
        }
        let mut h = left;
        let rate = self.derivative(c);
        let relative = |p: &Participant| {
            let i = self.index(p);
            if c[i] > 0.0 {
                (rate[i] / c[i]).abs()
            } else {
                0.0
            }
        };
        for (r, (reaction, &v)) in self.reactions.iter().zip(&velocities).enumerate() {
            if v * left <= 1e-14 * total {
                continue;
            }
            let change: f64 = reaction.reactants.iter().map(relative).sum();
            if change > 0.0 {
                h = h.min(STEP_FRACTION / change);
            }
            let (bound, sources) = &coherence[r];
            if *bound > 0.0 {
                let changes = sources.iter().any(|&pos| {
                    let partners: f64 = reaction
                        .reactants
                        .iter()
                        .enumerate()
                        .filter(|&(q, _)| q != pos)
                        .map(|(_, p)| relative(p))
                        .sum();
                    partners * left > 1e-12
                });
                if changes {
                    h = h.min(COHERENT_PHASE / bound);
                }
            }
        }
        h
    }

    /// Estimated operations of the delays and sweeps of `events` and the
    /// steps of the longest delay, from a run of the step rule over the
    /// amounts alone; refused beyond [`super::super::MAX_RELAXATION_WORK`]
    /// or [`MAX_KINETIC_STEPS`] steps in a delay.
    fn check_events(&self, events: &[Event], method: Propagation) -> Result<()> {
        let mut b = self.field();
        let mut c = self.initial();
        let mut now = 0.0;
        let mut work = 0.0;
        // The steps of a delay of `t` from `now` (amounts advanced) and the
        // first.
        let run =
            |c: &mut Vec<f64>, now: f64, t: f64, coherence: &Coherence| -> Result<(f64, f64)> {
                let (mut s, end) = (now, now + t);
                let mut steps = 0usize;
                let mut first = None;
                while s < end {
                    let h = self.step_size(c, end - s, coherence);
                    first.get_or_insert(h);
                    steps += 1;
                    if steps > MAX_KINETIC_STEPS {
                        return Err(QsoError::invalid(
                            "reactions",
                            format!("more than {MAX_KINETIC_STEPS} steps in a delay of {t} s"),
                        ));
                    }
                    self.advance(c, s, s + h)?;
                    s += h;
                }
                Ok((steps as f64, first.unwrap_or(t)))
            };
        // Three exponentials per step (nine with bilinear sources, whose
        // steps predict), the most of the fields' blocks.
        let exponentials = if self.has_bilinear() { 9.0 } else { 3.0 };
        let per_step = |bases: &[Base], c: &[f64], h: f64| {
            bases
                .iter()
                .map(|base| {
                    let refs = vec![[CMatrix::zeros(0, 0), CMatrix::zeros(0, 0)]; 0];
                    let refs = if base.bilinear.is_empty() {
                        refs
                    } else {
                        self.flat_references(base)
                    };
                    let probe = self.step_blocks(base, [c, c, c], h, &refs, None);
                    exponentials * probe.iter().map(|b| b.plan(method).1).sum::<f64>()
                })
                .fold(0.0, f64::max)
        };
        let coherence = |bases: &[Base]| -> Coherence {
            bases.iter().map(|b| self.coherence(b)).fold(
                vec![(0.0, Vec::new()); self.reactions.len()],
                merge_coherence,
            )
        };
        for e in events {
            match e {
                Event::Delay(t) => {
                    check_time("delay", *t)?;
                    if *t > 0.0 {
                        let bases = [self.base(b)?];
                        let start = c.clone();
                        let (steps, h) = run(&mut c, now, *t, &coherence(&bases))?;
                        work += steps * per_step(&bases, &start, h);
                    }
                    now += t;
                }
                Event::Field(to) => b = *to,
                Event::Sweep {
                    to,
                    duration,
                    steps,
                } => {
                    check_field(*to)?;
                    check_time("duration", *duration)?;
                    if *steps == 0 || *steps > super::super::MAX_SEGMENTS {
                        return Err(QsoError::invalid(
                            "steps",
                            format!("a sweep takes 1..={} steps", super::super::MAX_SEGMENTS),
                        ));
                    }
                    // Each sweep step holds a field; the work and the
                    // frequencies are taken as the most of the ends', one
                    // base held at a time.
                    let bound =
                        merge_coherence(coherence(&[self.base(b)?]), coherence(&[self.base(*to)?]));
                    let dt = duration / *steps as f64;
                    let start = c.clone();
                    let (mut total, mut first) = (0.0, f64::INFINITY);
                    for _ in 0..*steps {
                        if dt > 0.0 {
                            let (n, h) = run(&mut c, now, dt, &bound)?;
                            total += n;
                            first = first.min(h);
                        }
                        now += dt;
                    }
                    if total > 0.0 {
                        let most = per_step(&[self.base(b)?], &start, first).max(per_step(
                            &[self.base(*to)?],
                            &start,
                            first,
                        ));
                        work += total * most;
                    }
                    b = *to;
                }
                _ => {}
            }
            if work > super::super::MAX_RELAXATION_WORK {
                break;
            }
        }
        if work > super::super::MAX_RELAXATION_WORK {
            let lab = self
                .sites
                .iter()
                .any(|s| s.system.frame() == super::super::Frame::Laboratory);
            return Err(QsoError::invalid(
                "reactions",
                format!(
                    "these events need about {work:.1e} operations (limit {:.1e}); use fewer \
                     spins, shorter delays or fewer sweep steps{}",
                    super::super::MAX_RELAXATION_WORK,
                    if lab { ", or the rotating frame" } else { "" }
                ),
            ));
        }
        Ok(())
    }

    /// Reactions, evolution and relaxation for `t` at field `b` from time
    /// `t0`, amounts `c` (advanced); `work` counts the operations of the
    /// exponentials. Steps with a bilinear source that matters are checked
    /// against their halves (to [`BILINEAR_TOLERANCE`] of the deviation)
    /// and repeated shorter when they miss.
    #[allow(clippy::too_many_arguments)]
    fn delay(
        &self,
        x: &[CMatrix],
        c: &mut Vec<f64>,
        t0: f64,
        t: f64,
        b: f64,
        method: Propagation,
        work: &mut f64,
    ) -> Result<Vec<CMatrix>> {
        if t == 0.0 {
            return Ok(x.to_vec());
        }
        let base = self.base(b)?;
        let coherence = self.coherence(&base);
        let mut now = t0;
        let end = t0 + t;
        let mut x = x.to_vec();
        let mut steps = 0usize;
        let mut limit = f64::INFINITY;
        let norm = |m: &[CMatrix]| m.iter().map(|x| x.norm_squared()).sum::<f64>().sqrt();
        while now < end {
            let h = self.step_size(c, end - now, &coherence).min(limit);
            steps += 1;
            if steps > MAX_KINETIC_STEPS {
                return Err(QsoError::invalid(
                    "reactions",
                    format!("more than {MAX_KINETIC_STEPS} steps in a delay of {t} s"),
                ));
            }
            let sites: f64 = c[..self.sites.len()].iter().sum();
            // Relative to the deviation, but not below rounding of the
            // terms of order c/d that cancel in the drift.
            let tol = BILINEAR_TOLERANCE * (norm(&x) + 1e-6 * sites);
            let full = self.midpoint(&base, &x, c, now, h, method, tol)?;
            let first = self.midpoint(&base, &x, c, now, h / 2.0, method, tol)?;
            let mut c_half = c.clone();
            self.advance(&mut c_half, now, now + h / 2.0)?;
            let second = self.midpoint(
                &base,
                &first.states,
                &c_half,
                now + h / 2.0,
                h / 2.0,
                method,
                tol,
            )?;
            *work += full.work + first.work + second.work;
            if *work > super::super::MAX_RELAXATION_WORK {
                return Err(QsoError::invalid(
                    "reactions",
                    format!(
                        "the steps took more than {:.1e} operations: products of two reactants \
                         whose states change within a step (coherences in both) need short \
                         steps; use fewer spins or shorter delays",
                        super::super::MAX_RELAXATION_WORK
                    ),
                ));
            }
            let (half, remainder) = (
                second.states,
                full.remainder.max(first.remainder).max(second.remainder),
            );
            let settled = full.converged && first.converged && second.converged;
            let full = full.states;
            if !settled {
                limit = h / 4.0;
                continue;
            }
            if !base.bilinear.is_empty() {
                if remainder > 1e-3 * tol {
                    let diff: Vec<CMatrix> = half.iter().zip(&full).map(|(a, f)| a - f).collect();
                    let err = norm(&diff);
                    let factor = if err > 0.0 {
                        0.9 * (tol / err).cbrt()
                    } else {
                        4.0
                    };
                    if err > tol {
                        limit = h * factor.clamp(0.1, 0.5);
                        continue;
                    }
                    limit = h * factor.clamp(1.0, 4.0);
                } else {
                    limit *= 4.0;
                }
            }
            x = half
                .iter()
                .zip(&full)
                .map(|(a, f)| (a * C64::new(4.0, 0.0) - f) * C64::new(1.0 / 3.0, 0.0))
                .collect();
            *c = c_half;
            self.advance(c, now + h / 2.0, now + h)?;
            now += h;
        }
        Ok(x)
    }

    /// References `1/d` (zero deviations) for every bilinear source.
    fn flat_references(&self, base: &Base) -> References {
        base.bilinear
            .iter()
            .map(|f| [0, 1].map(|q| CMatrix::zeros(f.reduced[q].dim, f.reduced[q].dim)))
            .collect()
    }

    fn check_states(&self, states: &[CMatrix]) -> Result<()> {
        if states.len() != self.sites.len() {
            return Err(QsoError::invalid(
                "states",
                format!("{} states for {} sites", states.len(), self.sites.len()),
            ));
        }
        // Each site's trace to 10⁻⁹ of its own amount (of 10⁻¹² of all
        // sites' for a site without).
        let sites: f64 = self.sites.iter().map(|s| s.concentration).sum();
        for (rho, site) in states.iter().zip(&self.sites) {
            let total = site.concentration.max(1e-3 * sites).max(f64::MIN_POSITIVE);
            let d = site.system.dimension();
            if rho.shape() != (d, d) {
                return Err(QsoError::DimensionMismatch {
                    expected: d,
                    actual: rho.nrows(),
                });
            }
            let trace = rho.trace();
            if (trace.re - site.concentration).abs() > 1e-9 * total || trace.im.abs() > 1e-9 * total
            {
                return Err(QsoError::invalid(
                    "states",
                    format!(
                        "a site's state has trace {trace}, not its initial amount {}",
                        site.concentration
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Apply events from the initial amounts: ideal pulses on an isotope
    /// (on every site that has it), delays with reactions, evolution and
    /// relaxation, field jumps and sweeps. `states` are unnormalised, each
    /// with the trace of its site's initial amount, and so are the states
    /// returned (traces the amounts at the end).
    pub fn evolve(&self, states: &[CMatrix], events: &[Event]) -> Result<Vec<CMatrix>> {
        self.evolve_with(states, events, Propagation::Auto)
    }

    /// [`Self::evolve`] with the blocks propagated by `method`.
    pub(in crate::spin) fn evolve_with(
        &self,
        states: &[CMatrix],
        events: &[Event],
        method: Propagation,
    ) -> Result<Vec<CMatrix>> {
        self.validate()?;
        self.check_states(states)?;
        self.check_events(events, method)?;
        let mut b = self.field();
        let mut c = self.initial();
        let mut now = 0.0;
        let mut x = deviations(states, &c[..self.sites.len()]);
        let mut work = 0.0;
        for e in events {
            match e {
                Event::Pulse { target, .. } => {
                    let Target::Isotope(isotope) = target else {
                        return Err(QsoError::invalid(
                            "pulse",
                            "a pulse acts on an isotope (its spins in every site), not on \
                             single spins",
                        ));
                    };
                    let amounts = &c[..self.sites.len()];
                    let mut sigma = sigmas(x, amounts);
                    let mut any = false;
                    for (s, site) in self.sites.iter().enumerate() {
                        if !site.system.isotope_list().contains(isotope) {
                            continue;
                        }
                        any = true;
                        // A pulse is linear: it acts on σ = c·ρ as on ρ, and
                        // on a site without amount (σ traceless) through
                        // 1/d + ασ.
                        if amounts[s] > 0.0 {
                            let rho = &sigma[s] * C64::new(1.0 / amounts[s], 0.0);
                            sigma[s] = site.system.evolve(&rho, std::slice::from_ref(e))?
                                * C64::new(amounts[s], 0.0);
                        } else {
                            let largest = sigma[s].iter().fold(0.0f64, |m, v| m.max(v.norm()));
                            if largest > 0.0 {
                                let d = sigma[s].nrows();
                                let alpha = 1.0 / (2.0 * d as f64 * d as f64 * largest);
                                let identity =
                                    CMatrix::identity(d, d) * C64::new(1.0 / d as f64, 0.0);
                                let rho = &identity + &sigma[s] * C64::new(alpha, 0.0);
                                let moved = site.system.evolve(&rho, std::slice::from_ref(e))?;
                                sigma[s] = (moved - identity) * C64::new(1.0 / alpha, 0.0);
                            }
                        }
                    }
                    if !any {
                        return Err(QsoError::invalid(
                            "pulse",
                            format!("no site has a {} spin", isotope.name()),
                        ));
                    }
                    x = deviations(&sigma, amounts);
                }
                Event::Delay(t) => {
                    check_time("delay", *t)?;
                    x = self.delay(&x, &mut c, now, *t, b, method, &mut work)?;
                    now += t;
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
                    if *steps == 0 || *steps > super::super::MAX_SEGMENTS {
                        return Err(QsoError::invalid(
                            "steps",
                            format!("a sweep takes 1..={} steps", super::super::MAX_SEGMENTS),
                        ));
                    }
                    let from = b;
                    let dt = duration / *steps as f64;
                    for k in 0..*steps {
                        let field = from + (to - from) * (k as f64 + 0.5) / *steps as f64;
                        x = self.delay(&x, &mut c, now, dt, field, method, &mut work)?;
                        now += dt;
                    }
                    b = *to;
                }
                _ => {
                    return Err(QsoError::invalid(
                        "event",
                        "reactions take pulses on isotopes, delays, fields and sweeps",
                    ));
                }
            }
        }
        Ok(sigmas(x, &c[..self.sites.len()]))
    }
}

/// The most of two bounds per reaction, and the union of their sources.
fn merge_coherence(mut a: Coherence, b: Coherence) -> Coherence {
    for ((bound, sources), (other, more)) in a.iter_mut().zip(b) {
        *bound = bound.max(other);
        for q in more {
            if !sources.contains(&q) {
                sources.push(q);
            }
        }
    }
    a
}

/// The one-norm of `a` less its mean precession `i·mean(Im a_ii)`.
fn precession_norm(a: &SparseBlock) -> f64 {
    let n = a.n;
    if n == 0 {
        return 0.0;
    }
    let mean = (0..n).map(|i| a.get(i, i).im).sum::<f64>() / n as f64;
    let mut sums = vec![0.0f64; n];
    let mut diagonal = vec![false; n];
    for (r, has) in diagonal.iter_mut().enumerate() {
        for (c, v) in a.row(r) {
            if c == r {
                sums[c] += (v - C64::new(0.0, mean)).norm();
                *has = true;
            } else {
                sums[c] += v.norm();
            }
        }
    }
    for (s, d) in sums.iter_mut().zip(&diagonal) {
        if !d {
            *s += mean.abs();
        }
    }
    sums.into_iter().fold(0.0, f64::max)
}

fn check_amount(amount: f64) -> Result<()> {
    if !(amount.is_finite() && amount >= 0.0) {
        return Err(QsoError::invalid(
            "amount",
            "amounts must be finite and non-negative",
        ));
    }
    Ok(())
}

/// `σ_s − c_s/d_s`.
fn deviations(states: &[CMatrix], c: &[f64]) -> Vec<CMatrix> {
    states
        .iter()
        .zip(c)
        .map(|(sigma, &c)| {
            let d = sigma.nrows();
            let mut x = sigma.clone();
            for i in 0..d {
                x[(i, i)] -= c / d as f64;
            }
            x
        })
        .collect()
}

/// `x_s + c_s/d_s`.
fn sigmas(x: Vec<CMatrix>, c: &[f64]) -> Vec<CMatrix> {
    x.into_iter()
        .zip(c)
        .map(|(mut x, &c)| {
            let d = x.nrows();
            for i in 0..d {
                x[(i, i)] += c / d as f64;
            }
            x
        })
        .collect()
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod sparse_tests;

#[cfg(test)]
mod bilinear_tests;
