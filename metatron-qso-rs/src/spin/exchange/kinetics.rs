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

use serde::{Deserialize, Serialize};

use super::super::blocks::{LiouvilleBlock, Sparse};
use super::super::{Event, SpinSystem, Target, ZERO, check_field, check_time};
use super::{Channel, Exchange, Fresh, Site};
use crate::error::{QsoError, Result};
use crate::linalg::CMatrix;
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
/// `k Π c_reactants` (a reactant may appear twice). At most one product is
/// a spin site; it receives the carried spins of the first site among the
/// reactants (its source) and fresh spins for the rest.
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

    /// The spin site among the products.
    fn target(&self) -> Option<usize> {
        self.products.iter().find_map(|p| match p {
            Participant::Site(s) => Some(*s),
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
}

/// The base of the generator at one field: each site's evolution and
/// relaxation, and the transfers of the reactions.
struct Base {
    joint: super::Joint,
    matrices: Vec<CMatrix>,
    augmented: Vec<bool>,
    /// The relaxation's drift of each site per unit amount: `(block,
    /// position, value)`.
    drift: Vec<Vec<(usize, usize, C64)>>,
    /// Every element of each site: `(block, position, is a population)`.
    elements: Vec<Vec<(usize, usize, bool)>>,
    /// Per reaction with a product site: the transfer entries from its
    /// source `(block, row, column, w)` and the drift per unit pseudo rate
    /// and unit amount of the source, or (no source) the fresh state's
    /// entries per unit rate.
    transfers: Vec<Vec<(usize, usize, usize, C64)>>,
    drift_transfers: Vec<Vec<(usize, usize, C64)>>,
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
        self.check_reaction(&reaction)?;
        self.reactions.push(reaction);
        Ok(self)
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

    /// The channel through which a reaction carries spins into its product
    /// site (from the product itself when no reactant has spins).
    fn channel(r: &Reaction) -> Option<Channel> {
        let to = r.target()?;
        Some(Channel {
            from: r.source().map_or(to, |(_, s)| s),
            to,
            rate: r.rate,
            carried: r.carried.clone(),
            fresh: r.fresh,
        })
    }

    fn check_reaction(&self, r: &Reaction) -> Result<()> {
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
        let product_sites = r
            .products
            .iter()
            .filter(|p| matches!(p, Participant::Site(_)))
            .count();
        if product_sites > 1 {
            return bad("at most one product is a spin site");
        }
        if !(r.rate.is_finite() && r.rate >= 0.0) {
            return bad("the rate must be finite and non-negative");
        }
        match Self::channel(r) {
            Some(c) => {
                if r.source().is_none() && !r.carried.is_empty() {
                    return bad("carried spins need a reactant with spins");
                }
                self.shadow().check_channel(&c)?;
            }
            None => {
                if !r.carried.is_empty() || r.fresh != Fresh::Thermal {
                    return bad("carried and fresh spins need a product with spins");
                }
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
        for r in &self.reactions {
            self.check_reaction(r)?;
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
        let largest = joint.blocks.iter().map(|(_, _, n)| *n).max().unwrap_or(0);
        if largest > super::super::MAX_LIOUVILLE_BLOCK {
            return Err(QsoError::invalid(
                "reactions",
                format!(
                    "a block of coherence order over all sites has {largest} elements (at most {})",
                    super::super::MAX_LIOUVILLE_BLOCK
                ),
            ));
        }
        let augmented: Vec<bool> = joint
            .blocks
            .iter()
            .map(|(order, _, _)| order.iter().all(|&q| q == 0))
            .collect();
        let mut matrices: Vec<CMatrix> = joint
            .blocks
            .iter()
            .zip(&augmented)
            .map(|((_, _, n), &a)| CMatrix::zeros(n + usize::from(a), n + usize::from(a)))
            .collect();
        let at = |s: usize, a: usize, a2: usize| -> Option<(usize, usize)> {
            let l = &joint.liouvilles[s];
            let k = l.block_of(a, a2)?;
            let (j, offset) = joint.place[s][k];
            Some((j, offset + l.position_of(a, a2)))
        };
        let mut drift = Vec::with_capacity(self.sites.len());
        let mut elements = Vec::with_capacity(self.sites.len());
        for (s, site) in self.sites.iter().enumerate() {
            let sys = &site.system;
            let h = sys.hamiltonian_at(b);
            let relaxation = sys.relaxation(b)?;
            let sparse = Sparse::new(&h);
            sys.generator_entries(&joint.liouvilles[s], &sparse, &relaxation, |k, r, c, v| {
                let (j, offset) = joint.place[s][k];
                matrices[j][(offset + r, offset + c)] += v;
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
        let mut transfers = Vec::with_capacity(self.reactions.len());
        let mut drift_transfers = Vec::with_capacity(self.reactions.len());
        for r in &self.reactions {
            let mut entries = Vec::new();
            let mut drifts = Vec::new();
            if let Some(c) = Self::channel(r) {
                if r.source().is_some() {
                    let da = self.sites[c.from].system.dimension() as f64;
                    shadow.transfer(&c, b, |a, a2, t1, t2, w| {
                        let (Some((j, col)), Some((j2, row))) =
                            (at(c.from, a, a2), at(c.to, t1, t2))
                        else {
                            return;
                        };
                        debug_assert_eq!(j, j2, "reactions keep the coherence order");
                        entries.push((j, row, col, w));
                        if a == a2 {
                            drifts.push((j, row, w / da));
                        }
                    });
                } else {
                    let sys = &self.sites[c.to].system;
                    let (fresh, rho) = shadow.fresh_state(&c, b);
                    let levels: Vec<usize> = fresh
                        .iter()
                        .map(|&j| sys.spins()[j].isotope.levels())
                        .collect();
                    let index = |f: usize| -> usize {
                        let mut rest = f;
                        let mut out = 0;
                        for k in (0..fresh.len()).rev() {
                            out += (rest % levels[k]) * sys.stride(fresh[k]);
                            rest /= levels[k];
                        }
                        out
                    };
                    for f in 0..rho.nrows() {
                        for f2 in 0..rho.ncols() {
                            let v = rho[(f, f2)];
                            if v != ZERO
                                && let Some((j, row)) = at(c.to, index(f), index(f2))
                            {
                                drifts.push((j, row, v));
                            }
                        }
                    }
                }
            }
            transfers.push(entries);
            drift_transfers.push(drifts);
        }
        Ok(Base {
            joint,
            matrices,
            augmented,
            drift,
            elements,
            transfers,
            drift_transfers,
        })
    }

    /// The part of the generator that acts on the deviation, with the
    /// amounts `c` (each block `n × n`).
    fn matrix(&self, base: &Base, c: &[f64]) -> Vec<CMatrix> {
        let mut m: Vec<CMatrix> = base
            .matrices
            .iter()
            .zip(&base.joint.blocks)
            .map(|(full, (_, _, n))| full.view((0, 0), (*n, *n)).into_owned())
            .collect();
        for (r, reaction) in self.reactions.iter().enumerate() {
            // Each reactant site reacts at k times the other reactants.
            for (pos, p) in reaction.reactants.iter().enumerate() {
                let Participant::Site(s) = *p else { continue };
                let kappa = self.pseudo_rate(reaction, pos, c);
                if kappa == 0.0 {
                    continue;
                }
                for &(j, i, _) in &base.elements[s] {
                    m[j][(i, i)] -= kappa;
                }
            }
            if reaction.target().is_none() {
                continue;
            }
            if let Some((pos, _)) = reaction.source() {
                let g = self.pseudo_rate(reaction, pos, c);
                for &(j, row, col, w) in &base.transfers[r] {
                    m[j][(row, col)] += w * g;
                }
            }
        }
        m
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
    /// states of reactions without a source, and `−ċ_s/d_s`.
    fn drift(&self, base: &Base, c: &[f64]) -> Vec<Vec<C64>> {
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
        for (r, reaction) in self.reactions.iter().enumerate() {
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
            if reaction.target().is_none() {
                continue;
            }
            match reaction.source() {
                Some((pos, s)) => {
                    let g = self.pseudo_rate(reaction, pos, c);
                    for &(j, row, w) in &base.drift_transfers[r] {
                        push(j, row, w * g * c[s]);
                    }
                }
                None => {
                    for &(j, row, w) in &base.drift_transfers[r] {
                        push(j, row, w * velocities[r]);
                    }
                }
            }
        }
        f
    }

    /// The blocks of one step of `h`: the generator with the amounts at the
    /// midpoint `cm`, and the inhomogeneous term interpolated by the
    /// quadratic through its values at the start, midpoint and end
    /// (`c0`, `cm`, `c1`), integrated exactly through two more coordinates
    /// `u₁ = s/h`, `u₂ = (s/h)²/2` (the φ-functions of the exponential).
    fn step_blocks(
        &self,
        base: &Base,
        c0: &[f64],
        cm: &[f64],
        c1: &[f64],
        h: f64,
    ) -> Vec<LiouvilleBlock> {
        let matrices = self.matrix(base, cm);
        let (f0, fm, f1) = (
            self.drift(base, c0),
            self.drift(base, cm),
            self.drift(base, c1),
        );
        base.joint
            .blocks
            .iter()
            .zip(matrices)
            .enumerate()
            .map(|(j, ((_, _, n), mat))| {
                let n = *n;
                if !base.augmented[j] {
                    return LiouvilleBlock::sized(n, mat, false, h);
                }
                let mut m = CMatrix::zeros(n + 3, n + 3);
                m.view_mut((0, 0), (n, n)).copy_from(&mat);
                // f(s) = a + b s + e s² through the three values, with the
                // dimensionless u₁ = s/h and u₂ = (s/h)²/2 (u₁' = 1/h,
                // u₂' = u₁/h): f = a + (b h) u₁ + (2 e h²) u₂, all entries of
                // the order of f, however fast f changes.
                for i in 0..n {
                    let (f_0, f_m, f_1) = (f0[j][i], fm[j][i], f1[j][i]);
                    m[(i, n + 2)] = f_0;
                    m[(i, n)] = f_m * 4.0 - f_0 * 3.0 - f_1;
                    m[(i, n + 1)] = (f_0 * 2.0 - f_m * 4.0 + f_1 * 2.0) * 2.0;
                }
                m[(n, n + 2)] = C64::new(1.0 / h, 0.0);
                m[(n + 1, n)] = C64::new(1.0 / h, 0.0);
                LiouvilleBlock::sized(n + 2, m, true, h)
            })
            .collect()
    }

    /// One step of `h` from `t` (amounts `c` at `t`).
    fn midpoint(
        &self,
        base: &Base,
        x: &[CMatrix],
        c: &[f64],
        t: f64,
        h: f64,
    ) -> Result<Vec<CMatrix>> {
        let mut mid = c.to_vec();
        self.advance(&mut mid, t, t + h / 2.0)?;
        let mut end = mid.clone();
        self.advance(&mut end, t + h / 2.0, t + h)?;
        let blocks = self.step_blocks(base, c, &mid, &end, h);
        let mut out: Vec<CMatrix> = x
            .iter()
            .map(|m| CMatrix::zeros(m.nrows(), m.ncols()))
            .collect();
        for ((_, parts, n), block) in base.joint.blocks.iter().zip(&blocks) {
            let mut elements: Vec<(usize, usize, usize)> = Vec::with_capacity(*n);
            for &(s, k, _) in parts {
                for &(a, b) in &base.joint.liouvilles[s].blocks[k].1 {
                    elements.push((s, a, b));
                }
            }
            let moved = block.apply(|i| match elements.get(i) {
                Some(&(s, a, b)) => x[s][(a, b)],
                None => ZERO,
            });
            for (&(s, a, b), v) in elements.iter().zip(moved) {
                out[s][(a, b)] = v;
            }
        }
        Ok(out)
    }

    /// Per reaction that carries spins from a source into a product site:
    /// a bound on the frequencies of the blocks its transfers feed (the
    /// one-norm of the block without its mean phase), zero otherwise.
    fn coherence(&self, base: &Base) -> Vec<f64> {
        let norms: Vec<f64> = base
            .matrices
            .iter()
            .zip(&base.joint.blocks)
            .map(|(full, (_, _, n))| {
                let n = *n;
                let mut m = full.view((0, 0), (n, n)).into_owned();
                let mean = (0..n).map(|i| m[(i, i)].im).sum::<f64>() / n.max(1) as f64;
                for i in 0..n {
                    m[(i, i)] -= C64::new(0.0, mean);
                }
                crate::linalg::dense::one_norm(&m)
            })
            .collect();
        base.transfers
            .iter()
            .map(|entries| {
                entries
                    .iter()
                    .map(|&(j, _, _, _)| norms[j])
                    .fold(0.0, f64::max)
            })
            .collect()
    }

    /// The step from `t` with amounts `c`, at most `left`: for every
    /// reaction that still matters (more than 10⁻¹⁴ of the amounts over the
    /// rest of the delay) the amounts of its reactants change by at most
    /// [`STEP_FRACTION`] of themselves, so its rate changes by about that
    /// fraction; a reaction that carries spins at a pseudo-rate that
    /// changes also keeps `h` within `COHERENT_PHASE / coherence` (see
    /// [`Self::coherence`]).
    fn step_size(&self, c: &[f64], left: f64, coherence: &[f64]) -> f64 {
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
            if let Some((pos, _)) = reaction.source()
                && coherence[r] > 0.0
            {
                let partners: f64 = reaction
                    .reactants
                    .iter()
                    .enumerate()
                    .filter(|&(q, _)| q != pos)
                    .map(|(_, p)| relative(p))
                    .sum();
                if partners * left > 1e-12 {
                    h = h.min(COHERENT_PHASE / coherence[r]);
                }
            }
        }
        h
    }

    /// Estimated operations of the delays and sweeps of `events` and the
    /// steps of the longest delay, from a run of the step rule over the
    /// amounts alone; refused beyond [`super::super::MAX_RELAXATION_WORK`]
    /// or [`MAX_KINETIC_STEPS`] steps in a delay.
    fn check_events(&self, events: &[Event]) -> Result<()> {
        let mut b = self.field();
        let mut c = self.initial();
        let mut now = 0.0;
        let mut work = 0.0;
        // The steps of a delay of `t` from `now` (amounts advanced) and the
        // first.
        let run = |c: &mut Vec<f64>, now: f64, t: f64, coherence: &[f64]| -> Result<(f64, f64)> {
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
        // Three exponentials per step, the most of the fields' blocks.
        let per_step = |bases: &[Base], c: &[f64], h: f64| {
            bases
                .iter()
                .map(|base| {
                    let probe = self.step_blocks(base, c, c, c, h);
                    3.0 * probe.iter().map(LiouvilleBlock::work).sum::<f64>()
                })
                .fold(0.0, f64::max)
        };
        let coherence = |bases: &[Base]| -> Vec<f64> {
            bases
                .iter()
                .map(|b| self.coherence(b))
                .fold(vec![0.0; self.reactions.len()], |m, x| {
                    m.iter().zip(x).map(|(a, b)| a.max(b)).collect()
                })
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
                    // frequencies are taken as the most of the ends'.
                    let bases = [self.base(b)?, self.base(*to)?];
                    let bound = coherence(&bases);
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
                        work += total * per_step(&bases, &start, first);
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
    /// `t0`, amounts `c` (advanced).
    fn delay(
        &self,
        x: &[CMatrix],
        c: &mut Vec<f64>,
        t0: f64,
        t: f64,
        b: f64,
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
        while now < end {
            let h = self.step_size(c, end - now, &coherence);
            steps += 1;
            if steps > MAX_KINETIC_STEPS {
                return Err(QsoError::invalid(
                    "reactions",
                    format!("more than {MAX_KINETIC_STEPS} steps in a delay of {t} s"),
                ));
            }
            let full = self.midpoint(&base, &x, c, now, h)?;
            let half = self.midpoint(&base, &x, c, now, h / 2.0)?;
            let mut c_half = c.clone();
            self.advance(&mut c_half, now, now + h / 2.0)?;
            let half = self.midpoint(&base, &half, &c_half, now + h / 2.0, h / 2.0)?;
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
        self.validate()?;
        self.check_states(states)?;
        self.check_events(events)?;
        let mut b = self.field();
        let mut c = self.initial();
        let mut now = 0.0;
        let mut x = deviations(states, &c[..self.sites.len()]);
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
                    x = self.delay(&x, &mut c, now, *t, b)?;
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
                        x = self.delay(&x, &mut c, now, dt, field)?;
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
