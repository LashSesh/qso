//! Chemical exchange between spin systems: the Bloch–McConnell equations
//! (McConnell, J. Chem. Phys. 28, 430 (1958)) in Liouville space.
//!
//! An [`Exchange`] holds chemical sites, each a [`SpinSystem`] with a
//! concentration, and first-order channels between them. A channel moves
//! molecules from one site to another at a rate constant `k` (s⁻¹, the
//! fraction of the source converted per second). The spins it carries keep
//! their state; the source's other spins are traced out, and the target's
//! other spins arrive fresh: thermal, or two of them in the singlet of
//! parahydrogen. With `σ_s = c_s ρ_s` the amount of site `s`,
//!
//! ```text
//! dσ_B/dt = L_B σ_B − Σ_{B→·} k σ_B + Σ_{A→B} k · ρ_fresh ⊗ Tr_traced(σ_A),
//! ```
//!
//! with `L_B` the free evolution and relaxation of site `B`. This is the
//! ensemble average of molecules that change site at random, exponentially
//! distributed times. Concentrations must be a chemical steady state (every
//! site gains as many molecules per second as it loses), so bimolecular
//! steps enter with pseudo-first-order rates, and only the source of a
//! channel keeps its spins' state: a partner's arrive fresh.
//!
//! The equations are linear and keep every isotope's coherence order (or
//! the total one in the laboratory frame), as carried spins keep theirs,
//! traced spins contribute none and fresh states are populations and
//! zero-quantum coherences. They are propagated exactly, block by block of
//! that order over all sites, as the deviation from `c_s/d_s` with the rest
//! in an augmented column.

use std::collections::BTreeMap;

use nalgebra::DVector;
use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::blocks::{Liouville, LiouvilleBlock, Sparse};
use super::control::{Control, Entries, Generator, Optimized, expmv, frechet, substeps};
use super::{Event, Frame, Isotope, ONE, SpinSystem, Target, ZERO, check_field, check_time};
use crate::error::{QsoError, Result};
use crate::linalg::CMatrix;
use crate::optimize::{GradObjective, Method, Options, minimize};

/// Smallest pivot, relative to the largest, of the scaled steady-state
/// equations that counts as nonzero.
const SINGULAR: f64 = 1e-12;

/// A chemical site: a spin system and its concentration (any unit, the
/// same for all sites).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Site {
    pub system: SpinSystem,
    pub concentration: f64,
}

/// The state of the spins that arrive at a site without coming from the
/// source of a channel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
pub enum Fresh {
    /// Thermal equilibrium at the current field.
    #[default]
    Thermal,
    /// Two spins ½ of one isotope (spins of the target site) in the
    /// singlet with probability `fraction` and in the triplet otherwise,
    /// the rest thermal: hydrogen from parahydrogen.
    Parahydrogen { spins: [usize; 2], fraction: f64 },
}

/// A first-order channel from site `from` to site `to`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Channel {
    pub from: usize,
    pub to: usize,
    /// Rate constant (s⁻¹): the fraction of `from` converted per second.
    pub rate: f64,
    /// `(spin of from, spin of to)` pairs whose state is carried over.
    #[serde(default)]
    pub carried: Vec<(usize, usize)>,
    #[serde(default)]
    pub fresh: Fresh,
}

/// Sites and channels of chemical exchange.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Exchange {
    sites: Vec<Site>,
    channels: Vec<Channel>,
}

/// One block of coherence order over all sites: `(site, block of the
/// site's Liouville, offset in the joint block)`.
type JointPart = (usize, usize, usize);

/// The joint blocks: each order's parts and total size.
struct Joint {
    liouvilles: Vec<Liouville>,
    blocks: Vec<(Vec<i32>, Vec<JointPart>, usize)>,
    /// Joint block and offset of each site's block.
    place: Vec<Vec<(usize, usize)>>,
}

impl Exchange {
    /// No sites.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a site; returns its index.
    pub fn add_site(&mut self, system: SpinSystem, concentration: f64) -> Result<usize> {
        system.validate()?;
        if !(concentration.is_finite() && concentration > 0.0) {
            return Err(QsoError::invalid(
                "concentration",
                "concentrations must be finite and positive",
            ));
        }
        self.sites.push(Site {
            system,
            concentration,
        });
        Ok(self.sites.len() - 1)
    }

    /// Add a channel (checked; the steady state is checked by
    /// [`Exchange::validate`] once all channels are in).
    pub fn add_channel(&mut self, channel: Channel) -> Result<&mut Self> {
        self.check_channel(&channel)?;
        self.channels.push(channel);
        Ok(self)
    }

    pub fn sites(&self) -> &[Site] {
        &self.sites
    }

    pub fn channels(&self) -> &[Channel] {
        &self.channels
    }

    /// The field (T) of the sites.
    pub fn field(&self) -> f64 {
        self.sites.first().map_or(0.0, |s| s.system.field())
    }

    /// The same exchange at another field (T).
    pub fn with_field(mut self, b: f64) -> Result<Self> {
        for site in &mut self.sites {
            site.system = site.system.clone().with_field(b)?;
        }
        Ok(self)
    }

    /// From JSON (the fields of this struct), checked.
    pub fn from_json(text: &str) -> Result<Self> {
        let x: Exchange = serde_json::from_str(text)
            .map_err(|e| QsoError::invalid("json", format!("not an exchange: {e}")))?;
        x.validate()?;
        Ok(x)
    }

    /// As JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("exchanges serialise")
    }

    fn check_channel(&self, c: &Channel) -> Result<()> {
        let bad = |why: String| Err(QsoError::invalid("channel", why));
        let n = self.sites.len();
        if c.from >= n || c.to >= n {
            return bad(format!("sites {} and {} of {n}", c.from, c.to));
        }
        if !(c.rate.is_finite() && c.rate >= 0.0) {
            return bad(format!(
                "the rate must be finite and non-negative, not {}",
                c.rate
            ));
        }
        let (a, b) = (&self.sites[c.from].system, &self.sites[c.to].system);
        let mut from_seen = vec![false; a.spins().len()];
        let mut to_seen = vec![false; b.spins().len()];
        for &(i, j) in &c.carried {
            if i >= from_seen.len() || j >= to_seen.len() || from_seen[i] || to_seen[j] {
                return bad(format!(
                    "the carried pair ({i}, {j}) is out of range or repeated"
                ));
            }
            from_seen[i] = true;
            to_seen[j] = true;
            if a.spins()[i].isotope != b.spins()[j].isotope {
                return bad(format!("carried spins {i} and {j} are different isotopes"));
            }
        }
        if let Fresh::Parahydrogen {
            spins: [i, j],
            fraction,
        } = c.fresh
        {
            let ok = i != j
                && i < to_seen.len()
                && j < to_seen.len()
                && !to_seen[i]
                && !to_seen[j]
                && b.spins()[i].isotope == b.spins()[j].isotope
                && b.spins()[i].isotope.levels() == 2
                && (0.0..=1.0).contains(&fraction);
            if !ok {
                return bad(
                    "parahydrogen occupies two fresh spins ½ of one isotope, fraction in [0, 1]"
                        .to_string(),
                );
            }
        }
        Ok(())
    }

    /// Check the sites (one field, temperature, frame and carriers), the
    /// channels and the chemical steady state.
    pub fn validate(&self) -> Result<()> {
        let Some(first) = self.sites.first() else {
            return Err(QsoError::invalid("sites", "an exchange needs a site"));
        };
        for site in &self.sites {
            site.system.validate()?;
            let s = &site.system;
            let same = s.field() == first.system.field()
                && s.temperature() == first.system.temperature()
                && s.frame() == first.system.frame()
                && Isotope::ALL
                    .iter()
                    .all(|&x| s.carrier(x) == first.system.carrier(x));
            if !same {
                return Err(QsoError::invalid(
                    "sites",
                    "all sites share the field, temperature, frame and carriers",
                ));
            }
            if !(site.concentration.is_finite() && site.concentration > 0.0) {
                return Err(QsoError::invalid(
                    "concentration",
                    "concentrations must be finite and positive",
                ));
            }
        }
        for c in &self.channels {
            self.check_channel(c)?;
        }
        // Every site gains as many molecules per second as it loses.
        for (s, site) in self.sites.iter().enumerate() {
            let lost: f64 = self
                .channels
                .iter()
                .filter(|c| c.from == s)
                .map(|c| c.rate * site.concentration)
                .sum();
            let gained: f64 = self
                .channels
                .iter()
                .filter(|c| c.to == s)
                .map(|c| c.rate * self.sites[c.from].concentration)
                .sum();
            if (lost - gained).abs() > 1e-9 * lost.max(gained) {
                return Err(QsoError::invalid(
                    "channels",
                    format!(
                        "not a chemical steady state: site {s} loses {lost} and gains {gained} \
                         per second"
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Thermal equilibrium of every site (trace 1 each).
    pub fn thermal_state(&self) -> Result<Vec<CMatrix>> {
        self.validate()?;
        self.sites
            .iter()
            .map(|s| s.system.thermal_state())
            .collect()
    }

    fn check_states(&self, states: &[CMatrix]) -> Result<()> {
        if states.len() != self.sites.len() {
            return Err(QsoError::invalid(
                "states",
                format!("{} states for {} sites", states.len(), self.sites.len()),
            ));
        }
        for (rho, site) in states.iter().zip(&self.sites) {
            let d = site.system.dimension();
            if rho.shape() != (d, d) {
                return Err(QsoError::DimensionMismatch {
                    expected: d,
                    actual: rho.nrows(),
                });
            }
        }
        Ok(())
    }

    /// Polarisation `⟨I_z⟩/I` of a spin of a site.
    pub fn polarization(&self, states: &[CMatrix], site: usize, spin: usize) -> Result<f64> {
        self.check_states(states)?;
        let s = self
            .sites
            .get(site)
            .ok_or_else(|| QsoError::invalid("site", format!("site {site} out of range")))?;
        s.system.polarization(&states[site], spin)
    }

    /// The isotopes of all sites, in the order of [`Isotope::ALL`].
    fn isotopes(&self) -> Vec<Isotope> {
        Isotope::ALL
            .into_iter()
            .filter(|x| {
                self.sites
                    .iter()
                    .any(|s| s.system.isotope_list().contains(x))
            })
            .collect()
    }

    /// The labels of site `s` over all isotopes (rotating frame) or the
    /// total `2M` (laboratory frame).
    fn labels(&self, s: usize) -> Vec<Vec<i32>> {
        let sys = &self.sites[s].system;
        let local = sys.labels();
        if sys.frame() == Frame::Laboratory {
            return local;
        }
        let own = sys.isotope_list();
        let all = self.isotopes();
        local
            .into_iter()
            .map(|l| {
                all.iter()
                    .map(|x| own.iter().position(|y| y == x).map_or(0, |k| l[k]))
                    .collect()
            })
            .collect()
    }

    /// The coherence-order blocks of every site, joined by order.
    fn joint(&self, only: Option<&[i32]>) -> Joint {
        let liouvilles: Vec<Liouville> = (0..self.sites.len())
            .map(|s| Liouville::new(&self.labels(s), only))
            .collect();
        let mut index: BTreeMap<Vec<i32>, usize> = BTreeMap::new();
        let mut blocks: Vec<(Vec<i32>, Vec<JointPart>, usize)> = Vec::new();
        let mut place = Vec::with_capacity(liouvilles.len());
        for (s, l) in liouvilles.iter().enumerate() {
            let mut here = Vec::with_capacity(l.blocks.len());
            for (k, (order, elements)) in l.blocks.iter().enumerate() {
                let j = *index.entry(order.clone()).or_insert_with(|| {
                    blocks.push((order.clone(), Vec::new(), 0));
                    blocks.len() - 1
                });
                let offset = blocks[j].2;
                blocks[j].1.push((s, k, offset));
                blocks[j].2 += elements.len();
                here.push((j, offset));
            }
            place.push(here);
        }
        Joint {
            liouvilles,
            blocks,
            place,
        }
    }

    /// The fresh state of a channel at field `b`, over the fresh spins of
    /// its target (in their order there), and those spins.
    fn fresh_state(&self, c: &Channel, b: f64) -> (Vec<usize>, CMatrix) {
        let sys = &self.sites[c.to].system;
        let fresh: Vec<usize> = (0..sys.spins().len())
            .filter(|j| !c.carried.iter().any(|&(_, t)| t == *j))
            .collect();
        let levels: Vec<usize> = fresh
            .iter()
            .map(|&j| sys.spins()[j].isotope.levels())
            .collect();
        let size: usize = levels.iter().product();
        let digits = |mut f: usize| -> Vec<usize> {
            let mut out = vec![0; levels.len()];
            for k in (0..levels.len()).rev() {
                out[k] = f % levels[k];
                f /= levels[k];
            }
            out
        };
        let populations: Vec<Vec<f64>> = fresh.iter().map(|&j| sys.populations(j, b)).collect();
        // The parahydrogen pair's places among the fresh spins and its
        // state 1/4 + (1 − 4f)/3 I_i·I_j in the basis (m_i, m_j).
        let (pair, singlet) = match c.fresh {
            Fresh::Thermal => (None, CMatrix::zeros(4, 4)),
            Fresh::Parahydrogen {
                spins: [i, j],
                fraction,
            } => {
                let place = |s: usize| fresh.iter().position(|&x| x == s).expect("fresh");
                let [iz, ip, im, _, _] = super::spin_ops(2);
                let dot = iz.kronecker(&iz)
                    + (ip.kronecker(&im) + im.kronecker(&ip)) * C64::new(0.5, 0.0);
                let m = CMatrix::identity(4, 4) * C64::new(0.25, 0.0)
                    + dot * C64::new((1.0 - 4.0 * fraction) / 3.0, 0.0);
                (Some((place(i), place(j))), m)
            }
        };
        let rho = CMatrix::from_fn(size, size, |r, col| {
            let (x, y) = (digits(r), digits(col));
            let mut v = C64::new(1.0, 0.0);
            for k in 0..levels.len() {
                if pair.is_some_and(|(pi, pj)| k == pi || k == pj) {
                    continue;
                }
                if x[k] != y[k] {
                    return ZERO;
                }
                v *= populations[k][x[k]];
            }
            if let Some((pi, pj)) = pair {
                v *= singlet[(x[pi] * 2 + x[pj], y[pi] * 2 + y[pj])];
            }
            v
        });
        (fresh, rho)
    }

    /// For every element `(a, a')` of `from` whose traced spins agree, the
    /// elements `(b, b')` of `to` it feeds with their weights:
    /// `feed(a, a', b, b', weight)`.
    fn transfer(
        &self,
        c: &Channel,
        b_field: f64,
        mut feed: impl FnMut(usize, usize, usize, usize, C64),
    ) {
        let (src, dst) = (&self.sites[c.from].system, &self.sites[c.to].system);
        let (fresh, rho) = self.fresh_state(c, b_field);
        let level = |sys: &SpinSystem, i: usize, state: usize| {
            (state / sys.stride(i)) % sys.spins()[i].isotope.levels()
        };
        let traced: Vec<usize> = (0..src.spins().len())
            .filter(|i| !c.carried.iter().any(|&(f, _)| f == *i))
            .collect();
        let fresh_levels: Vec<usize> = fresh
            .iter()
            .map(|&j| dst.spins()[j].isotope.levels())
            .collect();
        let fresh_index = |f: usize| -> usize {
            // The fresh spins' part of a target index.
            let mut rest = f;
            let mut out = 0;
            for k in (0..fresh.len()).rev() {
                out += (rest % fresh_levels[k]) * dst.stride(fresh[k]);
                rest /= fresh_levels[k];
            }
            out
        };
        let fresh_part: Vec<usize> = (0..rho.nrows()).map(fresh_index).collect();
        let (da, _) = (src.dimension(), dst.dimension());
        for a in 0..da {
            let carried_a: usize = c
                .carried
                .iter()
                .map(|&(i, j)| level(src, i, a) * dst.stride(j))
                .sum();
            for a2 in 0..da {
                if traced
                    .iter()
                    .any(|&i| level(src, i, a) != level(src, i, a2))
                {
                    continue;
                }
                let carried_a2: usize = c
                    .carried
                    .iter()
                    .map(|&(i, j)| level(src, i, a2) * dst.stride(j))
                    .sum();
                for (f, &fa) in fresh_part.iter().enumerate() {
                    for (f2, &fb) in fresh_part.iter().enumerate() {
                        let v = rho[(f, f2)];
                        if v != ZERO {
                            feed(a, a2, carried_a + fa, carried_a2 + fb, v);
                        }
                    }
                }
            }
        }
    }

    /// The joint blocks of the generator at field `b`, scaled by `t`
    /// (`only`: a single coherence order).
    fn generators(
        &self,
        b: f64,
        t: f64,
        only: Option<&[i32]>,
    ) -> Result<(Joint, Vec<LiouvilleBlock>)> {
        let joint = self.joint(only);
        let largest = joint.blocks.iter().map(|(_, _, n)| *n).max().unwrap_or(0);
        if largest > super::MAX_LIOUVILLE_BLOCK {
            return Err(QsoError::invalid(
                "exchange",
                format!(
                    "a block of coherence order over all sites has {largest} elements (at most {})",
                    super::MAX_LIOUVILLE_BLOCK
                ),
            ));
        }
        let mut matrices: Vec<(CMatrix, bool)> = joint
            .blocks
            .iter()
            .map(|(order, _, n)| {
                let augmented = order.iter().all(|&q| q == 0);
                (
                    CMatrix::zeros(n + usize::from(augmented), n + usize::from(augmented)),
                    augmented,
                )
            })
            .collect();
        let at = |s: usize, a: usize, b: usize| -> Option<(usize, usize)> {
            let l = &joint.liouvilles[s];
            let k = l.block_of(a, b)?;
            let (j, offset) = joint.place[s][k];
            Some((j, offset + l.position_of(a, b)))
        };
        // Each site's own evolution and relaxation, and the drift of its
        // relaxation for c_s/d_s.
        let mut drift: Vec<BTreeMap<usize, C64>> = vec![BTreeMap::new(); joint.blocks.len()];
        for (s, site) in self.sites.iter().enumerate() {
            let sys = &site.system;
            let h = sys.hamiltonian_at(b);
            let relaxation = sys.relaxation(b)?;
            let sparse = Sparse::new(&h);
            sys.generator_entries(&joint.liouvilles[s], &sparse, &relaxation, |k, r, c, v| {
                let (j, offset) = joint.place[s][k];
                matrices[j].0[(offset + r, offset + c)] += v;
            });
            let d = sys.dimension();
            for a in 0..d {
                for a2 in 0..d {
                    let v = relaxation.drift[(a, a2)];
                    if v != ZERO
                        && let Some((j, i)) = at(s, a, a2)
                    {
                        *drift[j].entry(i).or_insert(ZERO) += v * site.concentration;
                    }
                }
            }
        }
        // Exchange: losses, gains, and the gains from c_A/d_A.
        for c in &self.channels {
            if c.rate == 0.0 {
                continue;
            }
            let from = &self.sites[c.from];
            let da = from.system.dimension();
            let level = from.concentration / da as f64;
            for a in 0..da {
                for a2 in 0..da {
                    if let Some((j, i)) = at(c.from, a, a2) {
                        matrices[j].0[(i, i)] -= c.rate;
                        // The loss of c_A/d_A.
                        if a == a2 {
                            *drift[j].entry(i).or_insert(ZERO) -= C64::new(c.rate * level, 0.0);
                        }
                    }
                }
            }
            self.transfer(c, b, |a, a2, t1, t2, w| {
                let (Some((j, col)), Some((j2, row))) = (at(c.from, a, a2), at(c.to, t1, t2))
                else {
                    return;
                };
                debug_assert_eq!(j, j2, "exchange keeps the coherence order");
                matrices[j].0[(row, col)] += w * c.rate;
                if a == a2 {
                    *drift[j].entry(row).or_insert(ZERO) += w * c.rate * level;
                }
            });
        }
        for (j, (order, _, n)) in joint.blocks.iter().enumerate() {
            if order.iter().all(|&q| q == 0) {
                for (&i, &v) in &drift[j] {
                    matrices[j].0[(i, *n)] += v;
                }
            }
        }
        let blocks = joint
            .blocks
            .iter()
            .zip(matrices)
            .map(|((_, _, n), (m, augmented))| LiouvilleBlock::sized(*n, m, augmented, t))
            .collect();
        Ok((joint, blocks))
    }

    /// `σ_s − c_s/d_s` of normalised states.
    fn deviations(&self, states: &[CMatrix]) -> Vec<CMatrix> {
        states
            .iter()
            .zip(&self.sites)
            .map(|(rho, site)| {
                let d = rho.nrows();
                let mut x = rho * C64::new(site.concentration, 0.0);
                for i in 0..d {
                    x[(i, i)] -= site.concentration / d as f64;
                }
                x
            })
            .collect()
    }

    /// Normalised states from deviations.
    fn states(&self, deviations: Vec<CMatrix>) -> Vec<CMatrix> {
        deviations
            .into_iter()
            .zip(&self.sites)
            .map(|(mut x, site)| {
                let d = x.nrows();
                for i in 0..d {
                    x[(i, i)] += site.concentration / d as f64;
                }
                x * C64::new(1.0 / site.concentration, 0.0)
            })
            .collect()
    }

    /// Free evolution with exchange and relaxation for `t` at field `b`.
    fn delay(&self, deviations: &[CMatrix], t: f64, b: f64) -> Result<Vec<CMatrix>> {
        if t == 0.0 {
            return Ok(deviations.to_vec());
        }
        let (joint, blocks) = self.generators(b, t, None)?;
        let work: f64 = blocks.iter().map(LiouvilleBlock::work).sum();
        check_work(&format!("exchange over {t} s"), work)?;
        let mut out: Vec<CMatrix> = deviations
            .iter()
            .map(|x| CMatrix::zeros(x.nrows(), x.ncols()))
            .collect();
        for ((_, parts, n), block) in joint.blocks.iter().zip(&blocks) {
            let mut elements: Vec<(usize, usize, usize)> = Vec::with_capacity(*n);
            for &(s, k, _) in parts {
                for &(a, b) in &joint.liouvilles[s].blocks[k].1 {
                    elements.push((s, a, b));
                }
            }
            let moved = block.apply(|i| {
                let (s, a, b) = elements[i];
                deviations[s][(a, b)]
            });
            for (&(s, a, b), v) in elements.iter().zip(moved) {
                out[s][(a, b)] = v;
            }
        }
        Ok(out)
    }

    /// Apply events to the states of all sites (normalised, trace 1 each):
    /// ideal pulses on an isotope (on every site that has it), delays with
    /// exchange and relaxation, field jumps and sweeps. The field starts at
    /// the sites' and may change; the sites keep their own.
    pub fn evolve(&self, states: &[CMatrix], events: &[Event]) -> Result<Vec<CMatrix>> {
        self.validate()?;
        self.check_states(states)?;
        let mut b = self.field();
        let mut x = self.deviations(states);
        for e in events {
            match e {
                Event::Pulse { target, .. } => {
                    let Target::Isotope(isotope) = target else {
                        return Err(QsoError::invalid(
                            "pulse",
                            "a pulse in an exchange acts on an isotope (its spins in every \
                             site), not on single spins",
                        ));
                    };
                    let mut any = false;
                    let mut rhos = self.states(x);
                    for (rho, site) in rhos.iter_mut().zip(&self.sites) {
                        if site.system.isotope_list().contains(isotope) {
                            *rho = site.system.evolve(rho, std::slice::from_ref(e))?;
                            any = true;
                        }
                    }
                    if !any {
                        return Err(QsoError::invalid(
                            "pulse",
                            format!("no site has a {} spin", isotope.name()),
                        ));
                    }
                    x = self.deviations(&rhos);
                }
                Event::Delay(t) => {
                    check_time("delay", *t)?;
                    x = self.delay(&x, *t, b)?;
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
                    if *steps == 0 || *steps > super::MAX_SEGMENTS {
                        return Err(QsoError::invalid(
                            "steps",
                            format!("a sweep takes 1..={} steps", super::MAX_SEGMENTS),
                        ));
                    }
                    let from = b;
                    let dt = duration / *steps as f64;
                    let at = |k: usize| from + (to - from) * (k as f64 + 0.5) / *steps as f64;
                    // Every step is as costly as the one at the strongest
                    // field, or less.
                    let strongest = if to.abs() > from.abs() { *steps - 1 } else { 0 };
                    let (_, blocks) = self.generators(at(strongest), dt, None)?;
                    let work: f64 = blocks.iter().map(LiouvilleBlock::work).sum();
                    check_work(&format!("a sweep of {steps} steps"), work * *steps as f64)?;
                    for k in 0..*steps {
                        x = self.delay(&x, dt, at(k))?;
                    }
                    b = *to;
                }
                _ => {
                    return Err(QsoError::invalid(
                        "event",
                        "an exchange takes pulses on isotopes, delays, fields and sweeps",
                    ));
                }
            }
        }
        Ok(self.states(x))
    }

    /// The steady state at the sites' field: populations and zero-quantum
    /// coherences where exchange and relaxation balance, other coherences
    /// zero. An error if it is not unique (nothing exchanges or relaxes).
    pub fn steady_state(&self) -> Result<Vec<CMatrix>> {
        self.validate()?;
        let b = self.field();
        let zero: Vec<i32> = match self.sites[0].system.frame() {
            Frame::Laboratory => vec![0],
            Frame::Rotating => vec![0; self.isotopes().len()],
        };
        let (joint, blocks) = self.generators(b, 1.0, Some(&zero))?;
        let mut out: Vec<CMatrix> = self
            .sites
            .iter()
            .map(|s| CMatrix::zeros(s.system.dimension(), s.system.dimension()))
            .collect();
        let refuse = || {
            QsoError::invalid(
                "steady state",
                "no unique steady state: some populations neither exchange nor relax",
            )
        };
        for ((_, parts, n), block) in joint.blocks.iter().zip(&blocks) {
            let n = *n;
            let g = block.generator.view((0, 0), (n, n));
            let want = DVector::from_iterator(n, (0..n).map(|i| -block.generator[(i, n)]));
            // G x = −drift is singular, as the chemistry conserves the
            // amounts of the sites. The populations of a site sum to a
            // combination of the sites' traces, so one of its equations
            // is replaced by Tr Δ_s = 0 without losing any.
            let mut diagonal: Vec<Vec<usize>> = vec![Vec::new(); self.sites.len()];
            let mut i = 0;
            for &(s, k, _) in parts {
                for &(r, c) in &joint.liouvilles[s].blocks[k].1 {
                    if r == c {
                        diagonal[s].push(i);
                    }
                    i += 1;
                }
            }
            let mut a = g.into_owned();
            let mut rhs = want.clone();
            for rows in &diagonal {
                if let Some(&first) = rows.first() {
                    a.row_mut(first).fill(ZERO);
                    for &j in rows {
                        a[(first, j)] = ONE;
                    }
                    rhs[first] = ZERO;
                }
            }
            // Rows and then columns scaled to 1: laboratory-frame
            // frequencies exceed the rates by ten orders of magnitude.
            for i in 0..n {
                let m = a.row(i).iter().fold(0.0f64, |m, z| m.max(z.norm()));
                if m == 0.0 {
                    return Err(refuse());
                }
                a.row_mut(i).scale_mut(1.0 / m);
                rhs[i] /= m;
            }
            let columns: Vec<f64> = (0..n)
                .map(|j| a.column(j).iter().fold(0.0f64, |m, z| m.max(z.norm())))
                .collect();
            for (j, &m) in columns.iter().enumerate() {
                if m == 0.0 {
                    return Err(refuse());
                }
                a.column_mut(j).scale_mut(1.0 / m);
            }
            let lu = a.full_piv_lu();
            let pivots: Vec<f64> = lu.u().diagonal().iter().map(|z| z.norm()).collect();
            let largest = pivots.iter().fold(0.0f64, |m, &p| m.max(p));
            let smallest = pivots.iter().fold(f64::INFINITY, |m, &p| m.min(p));
            if smallest.is_nan() || smallest <= SINGULAR * largest {
                return Err(refuse());
            }
            let mut x = lu.solve(&rhs).ok_or_else(refuse)?;
            for (v, &m) in x.iter_mut().zip(&columns) {
                *v /= m;
            }
            // The equations replaced hold too (the chemistry balances to
            // 10⁻⁹ of its rates), relative to the generator applied to
            // the whole state, deviation and `c_s/d_s`: the drift is the
            // small difference of such terms.
            let max_norm = |v: &DVector<C64>| v.iter().fold(0.0f64, |m, z| m.max(z.norm()));
            let residual = max_norm(&(g * &x - &want));
            let norm = g
                .row_iter()
                .map(|r| r.iter().map(|z| z.norm()).sum::<f64>())
                .fold(0.0, f64::max);
            let level = self
                .sites
                .iter()
                .map(|s| s.concentration / s.system.dimension() as f64)
                .fold(0.0, f64::max);
            let scale = norm * (max_norm(&x) + level) + max_norm(&want);
            if residual.is_nan() || residual > 1e-8 * scale {
                return Err(refuse());
            }
            let mut i = 0;
            for &(s, k, _) in parts {
                for &(a2, b2) in &joint.liouvilles[s].blocks[k].1 {
                    out[s][(a2, b2)] = x[i];
                    i += 1;
                }
            }
        }
        // Hermitian up to rounding.
        for x in &mut out {
            *x = (&*x + x.adjoint()) * C64::new(0.5, 0.0);
        }
        Ok(self.states(out))
    }

    /// The free induction decay of an isotope summed over the sites
    /// weighted by concentration, `Σ_s c_s Tr(ρ_s(k·dwell) F₊)`, with
    /// exchange and relaxation during acquisition.
    pub fn fid(
        &self,
        states: &[CMatrix],
        isotope: Isotope,
        dwell: f64,
        points: usize,
    ) -> Result<Vec<C64>> {
        self.validate()?;
        self.check_states(states)?;
        if !(dwell.is_finite() && dwell > 0.0) || points == 0 || points > super::MAX_FID_POINTS {
            return Err(QsoError::invalid(
                "fid",
                format!(
                    "a positive dwell time and 1..={} points",
                    super::MAX_FID_POINTS
                ),
            ));
        }
        let frame = self.sites[0].system.frame();
        let order: Vec<i32> = match frame {
            Frame::Laboratory => vec![-2],
            Frame::Rotating => self
                .isotopes()
                .into_iter()
                .map(|x| if x == isotope { -2 } else { 0 })
                .collect(),
        };
        if !self.isotopes().contains(&isotope) {
            return Err(QsoError::invalid(
                "isotope",
                format!("no site has a {} spin", isotope.name()),
            ));
        }
        let (joint, blocks) = self.generators(self.field(), dwell, Some(&order))?;
        let work: f64 = blocks
            .iter()
            .map(|block| block.work() + 8.0 * (points * block.generator.len()) as f64)
            .sum();
        check_work(&format!("an exchange FID of {points} points"), work)?;
        let x = self.deviations(states);
        let mut out = vec![ZERO; points];
        for ((_, parts, n), block) in joint.blocks.iter().zip(&blocks) {
            let mut elements: Vec<(usize, usize, usize)> = Vec::with_capacity(*n);
            for &(s, k, _) in parts {
                for &(a, b) in &joint.liouvilles[s].blocks[k].1 {
                    elements.push((s, a, b));
                }
            }
            let fps: Vec<Option<CMatrix>> = self
                .sites
                .iter()
                .map(|s| s.system.fplus(isotope).ok())
                .collect();
            let weights: Vec<C64> = elements
                .iter()
                .map(|&(s, a, b)| fps[s].as_ref().map_or(ZERO, |f| f[(b, a)]))
                .collect();
            let p = crate::linalg::dense::expm(&block.generator) * block.shift.exp();
            let mut v = nalgebra::DVector::from_iterator(
                *n,
                elements.iter().map(|&(s, a, b)| x[s][(a, b)]),
            );
            for (k, o) in out.iter_mut().enumerate() {
                if k > 0 {
                    v = &p * v;
                }
                *o += v.iter().zip(&weights).map(|(a, w)| a * w).sum::<C64>();
            }
        }
        Ok(out)
    }
}

/// A generator block at a field and, if asked for, its field derivative.
type BlockAt = (CMatrix, Option<Entries>);

/// A field profile for an [`Exchange`]: the field (T) constant over
/// `steps` equal steps of `duration` s and between `min` and `max`, taking
/// the states `initial` (normalised, one per site) towards the largest
/// `Σ_s Re Tr(C_s† ρ_s(T))` with `C_s = target[s]`, less `smoothness ·
/// Σ (B_{k+1} − B_k)²`. See [`Exchange::optimize_profile`].
#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    pub min: f64,
    pub max: f64,
    pub duration: f64,
    pub steps: usize,
    pub initial: Vec<CMatrix>,
    pub target: Vec<CMatrix>,
    pub smoothness: f64,
}

impl Profile {
    /// The equivalent control problem of one system, for its mapping of
    /// bounded fields and its penalty.
    fn shape(&self) -> Control {
        let mut c = Control::field(
            self.min,
            self.max,
            self.duration,
            self.steps,
            CMatrix::zeros(1, 1),
            CMatrix::zeros(1, 1),
        );
        c.smoothness = self.smoothness;
        c
    }
}

impl Exchange {
    fn check_profile(&self, p: &Profile, fields: &[f64]) -> Result<()> {
        self.validate()?;
        self.check_states(&p.initial)?;
        if p.target.len() != self.sites.len()
            || p.target.iter().zip(&self.sites).any(|(c, s)| {
                c.nrows() != s.system.dimension() || c.ncols() != s.system.dimension()
            })
        {
            return Err(QsoError::invalid(
                "target",
                "one operator per site, of its dimension",
            ));
        }
        check_time("duration", p.duration)?;
        if p.duration == 0.0 || p.steps == 0 || p.steps > super::MAX_SEGMENTS {
            return Err(QsoError::invalid(
                "profile",
                format!("a positive duration in 1..={} steps", super::MAX_SEGMENTS),
            ));
        }
        if !(p.min.is_finite() && p.max.is_finite() && 0.0 <= p.min && p.min < p.max) {
            return Err(QsoError::invalid("field", "0 ≤ min < max"));
        }
        if !(p.smoothness.is_finite() && p.smoothness >= 0.0) {
            return Err(QsoError::invalid("smoothness", "finite and ≥ 0"));
        }
        if p.initial
            .iter()
            .chain(&p.target)
            .any(|m| m.iter().any(|z| !(z.re.is_finite() && z.im.is_finite())))
        {
            return Err(QsoError::invalid(
                "profile",
                "initial states and targets must be finite",
            ));
        }
        if fields.len() != p.steps {
            return Err(QsoError::DimensionMismatch {
                expected: p.steps,
                actual: fields.len(),
            });
        }
        if fields
            .iter()
            .any(|b| !(b.is_finite() && p.min <= *b && *b <= p.max))
        {
            return Err(QsoError::invalid("fields", "finite and within [min, max]"));
        }
        Ok(())
    }

    /// The generator blocks at field `b` (unscaled, unshifted), and with
    /// `derivative` their derivatives with respect to the field: central
    /// differences of steps `δ` and `δ/2`, `δ = 10⁻²·max(|b|, span)`,
    /// Richardson-extrapolated (exact up to rounding for the Hamiltonian,
    /// which is linear in the field; near 10⁻⁸ for relaxation and thermal
    /// populations).
    fn blocks_at(
        &self,
        b: f64,
        span: f64,
        keep: &[bool],
        derivative: bool,
    ) -> Result<Vec<Option<BlockAt>>> {
        let full = |b: f64| -> Result<Vec<CMatrix>> {
            let (_, blocks) = self.generators(b, 1.0, None)?;
            Ok(blocks
                .into_iter()
                .map(|block| {
                    // Undo the shift by the mean precession (zero for the
                    // augmented block of populations).
                    let mut g = block.generator;
                    for i in 0..g.nrows() {
                        g[(i, i)] += block.shift;
                    }
                    g
                })
                .collect())
        };
        let centre = full(b)?;
        let derivatives: Vec<Option<CMatrix>> = if derivative {
            let h = 1e-2 * b.abs().max(span);
            let central = |s: f64| -> Result<Vec<CMatrix>> {
                Ok(full(b + s)?
                    .into_iter()
                    .zip(full(b - s)?)
                    .map(|(p, m)| (p - m) * C64::new(0.5 / s, 0.0))
                    .collect())
            };
            central(h)?
                .into_iter()
                .zip(central(h / 2.0)?)
                .map(|(wide, narrow)| {
                    Some(narrow * C64::new(4.0 / 3.0, 0.0) - wide * C64::new(1.0 / 3.0, 0.0))
                })
                .collect()
        } else {
            vec![None; centre.len()]
        };
        Ok(centre
            .into_iter()
            .zip(derivatives)
            .zip(keep)
            .map(|((g, d), &k)| k.then(|| (g, d.map(|d| Entries::from_dense(&d)))))
            .collect())
    }

    /// `Σ_s Re Tr(C_s† ρ_s(T))` under the field profile `fields` and its
    /// gradient with respect to the fields: exact up to rounding for the
    /// coherent part, the field dependence of relaxation and fresh thermal
    /// states by extrapolated central differences (near 10⁻⁸ of it).
    pub fn profile_transfer(&self, p: &Profile, fields: &[f64]) -> Result<(f64, Vec<f64>)> {
        self.check_profile(p, fields)?;
        let dt = p.duration / p.steps as f64;
        let joint = self.joint(None);
        let x0 = self.deviations(&p.initial);
        let mut constant = 0.0;
        for (c, site) in p.target.iter().zip(&self.sites) {
            constant += c.trace().re / site.system.dimension() as f64;
        }
        // Initial vectors and targets per block; blocks the target does
        // not see are skipped.
        let mut starts = Vec::new();
        let mut targets = Vec::new();
        let mut keep = Vec::new();
        for (order, parts, n) in &joint.blocks {
            let augmented = order.iter().all(|&q| q == 0);
            let size = n + usize::from(augmented);
            let mut x = DVector::zeros(size);
            let mut l = DVector::zeros(size);
            let mut i = 0;
            for &(s, k, _) in parts {
                let c = self.sites[s].concentration;
                for &(a, b) in &joint.liouvilles[s].blocks[k].1 {
                    x[i] = x0[s][(a, b)];
                    l[i] = p.target[s][(a, b)] / c;
                    i += 1;
                }
            }
            if augmented {
                x[*n] = C64::new(1.0, 0.0);
            }
            keep.push(l.iter().any(|z| *z != ZERO));
            starts.push(x);
            targets.push(l);
        }
        let span = p.max;
        // Cost of one evaluation, from the blocks at the strongest field
        // (the norms grow with it).
        let strongest = fields.iter().fold(0.0f64, |m, &b| m.max(b));
        let probe = self.blocks_at(strongest, span, &keep, false)?;
        let (mut work, mut memory) = (0.0, 0.0);
        for (block, _) in probe.iter().flatten() {
            let n = block.nrows() as f64;
            let norm = crate::linalg::dense::one_norm(block);
            work += 5.0 * (substeps(norm * dt) * 20 * 6) as f64 * 8.0 * n * n;
            memory += 16.0 * n * (p.steps as f64 + 1.0) + 16.0 * n * n * 6.0;
        }
        // Five generators per step (one forward, four for the derivative).
        let total: usize = joint.blocks.iter().map(|(_, _, n)| n + 1).sum();
        work += 5.0 * 40.0 * (total * total) as f64;
        check_work(
            &format!("a field profile of {} steps", p.steps),
            work * p.steps as f64,
        )?;
        if memory > super::control::MAX_CONTROL_MEMORY {
            return Err(QsoError::invalid(
                "profile",
                format!(
                    "needs about {:.1} GB; use fewer steps or spins",
                    memory / 1e9
                ),
            ));
        }
        let no_ops: [Entries; 0] = [];
        let mut states: Vec<Vec<DVector<C64>>> = vec![starts];
        for &b in fields {
            let blocks = self.blocks_at(b, span, &keep, false)?;
            let last = states.last().expect("start");
            let next = blocks
                .iter()
                .zip(last)
                .map(|(block, x)| match block {
                    Some((g, _)) => {
                        let forward = Generator {
                            dense: g,
                            ops: &no_ops,
                            u: &[],
                            adjoint: false,
                        };
                        expmv(&forward, x, dt, crate::linalg::dense::one_norm(g))
                    }
                    None => x.clone(),
                })
                .collect();
            states.push(next);
        }
        let end = states.last().expect("end");
        let value = constant
            + end
                .iter()
                .zip(&targets)
                .map(|(x, l)| l.dotc(x).re)
                .sum::<f64>();
        let mut grad = vec![0.0; p.steps];
        let mut lambdas = targets;
        for k in (0..p.steps).rev() {
            let blocks = self.blocks_at(fields[k], span, &keep, true)?;
            for ((block, x), lambda) in blocks.iter().zip(&states[k]).zip(lambdas.iter_mut()) {
                let Some((g, Some(d))) = block else { continue };
                let norm = crate::linalg::dense::one_norm(g);
                let forward = Generator {
                    dense: g,
                    ops: &no_ops,
                    u: &[],
                    adjoint: false,
                };
                grad[k] += frechet(&forward, &[d], x, lambda, dt, norm)[0];
                let back = Generator {
                    dense: g,
                    ops: &no_ops,
                    u: &[],
                    adjoint: true,
                };
                *lambda = expmv(&back, lambda, dt, norm);
            }
        }
        Ok((value, grad))
    }

    /// Optimise a field profile from `start` (fields within the limits)
    /// with L-BFGS.
    pub fn optimize_profile(
        &self,
        p: &Profile,
        start: &[f64],
        options: &Options,
    ) -> Result<Optimized> {
        self.check_profile(p, start)?;
        let shape = p.shape();
        let start = shape.inside(start)?;
        let (first, _) = self.profile_transfer(p, &start)?;
        // The optimiser sees the transfer in units of its starting size.
        let norm: f64 = p
            .target
            .iter()
            .map(|c| c.iter().map(|z| z.norm_sqr()).sum::<f64>())
            .sum::<f64>()
            .sqrt();
        let size = if first.abs() > 1e-12 * norm {
            first.abs()
        } else {
            norm.max(1e-300)
        };
        let failure = std::sync::Mutex::new(None);
        let objective = GradObjective::new(p.steps, |v: &[f64]| {
            let (u, jac) = shape.physical(v);
            match self.profile_transfer(p, &u) {
                Ok((value, grad)) => {
                    let (pen, pg) = shape.penalty(&u);
                    let g: Vec<f64> = grad.iter().zip(&pg).map(|(a, b)| (b - a) / size).collect();
                    ((pen - value) / size, shape.chain(&jac, &g))
                }
                Err(e) => {
                    *failure.lock().expect("unpoisoned") = Some(e);
                    (f64::INFINITY, vec![0.0; v.len()])
                }
            }
        });
        let result = minimize(&objective, &shape.free(&start), &Method::lbfgs(), options)?;
        if let Some(e) = failure.into_inner().expect("unpoisoned") {
            return Err(e);
        }
        let (controls, _) = shape.physical(&result.x);
        let (transfer, _) = self.profile_transfer(p, &controls)?;
        Ok(Optimized {
            objective: transfer - shape.penalty(&controls).0,
            members: vec![transfer],
            controls,
            transfer,
            iterations: result.iterations,
            evaluations: result.evaluations,
            converged: result.converged,
            message: result.message,
        })
    }
}

/// Refuse `what` above the work limit.
fn check_work(what: &str, work: f64) -> Result<()> {
    if work > super::MAX_RELAXATION_WORK {
        return Err(QsoError::invalid(
            "exchange",
            format!(
                "{what} needs about {work:.1e} operations (limit {:.1e}); use fewer spins, \
                 fewer steps or the rotating frame",
                super::MAX_RELAXATION_WORK
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
