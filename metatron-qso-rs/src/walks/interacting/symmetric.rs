//! Interacting walks reduced by symmetry: amplitudes on orbits of
//! configurations.
//!
//! An automorphism `g` of the graph permutes the nodes, and with them the
//! configurations of the particles: `U_g|c⟩ = s(g, c)|g·c⟩`, where `s` is
//! the sign of sorting the fermions' new sites (1 for bosons and
//! distinguishable particles). `U_g` commutes with the Hamiltonian of
//! [`InteractingWalk`](super::InteractingWalk), so a state invariant under
//! a group `G` of automorphisms stays invariant. Such a state is fixed by
//! one amplitude per orbit of `G` on the configurations, since
//! `ψ(g·c) = s(g, c) ψ(c)`. When these relations contradict each other on
//! an orbit (a fermionic configuration that some `g` maps to itself with
//! sign −1), every invariant state vanishes there and the orbit drops out.
//!
//! On the orbits that remain, the states
//! `|O⟩ = |O|^{−1/2} Σ_{c∈O} φ(c)|c⟩` (`φ = ±1` relative to the orbit's
//! first configuration) are orthonormal, and the Hamiltonian is
//! `⟨O′|H|O⟩ = √(|O|/|O′|) Σ_{c′∈O′} φ(c′) ⟨c′|H|r⟩`, read off at the
//! representative `r` of `O`. The walk then has as many amplitudes as there
//! are orbits: three bosons on the hypercube `Q₈` started at one node have
//! 2 829 056 configurations but, under the stabiliser of that node (the
//! coordinate permutations), 1324 orbits.
//!
//! Orbits are found by union–find over all configurations, with the signs
//! carried along, so at most [`MAX_ORBIT_CONFIGURATIONS`] configurations
//! (`2²⁷`, a 4-byte word each) are allowed, and the work grows with the
//! number of generators: give a small generating set, such as
//! [`Graph::stabilizer`]'s. Every configuration is mapped once by every
//! generator, about a microsecond each.
//!
//! ```
//! use metatron_qso::graph::generators;
//! use metatron_qso::walks::{Generator, Statistics, SymmetricInteractingWalk};
//!
//! // Two bosons on the 4 × 4 rook's graph, both at node 0, with an
//! // on-site interaction: the stabiliser of node 0 leaves 11 orbits of
//! // the 136 configurations.
//! let g = generators::rook(4, 4).unwrap();
//! let walk = SymmetricInteractingWalk::from_start(
//!     &g, Generator::Adjacency, 1.0, Statistics::Bosons, &[0, 0],
//! )
//! .unwrap()
//! .with_interaction(1.0, 0.0)
//! .unwrap();
//! assert_eq!((walk.num_orbits(), walk.num_configurations()), (11, 136));
//! let psi = walk.evolve(&walk.state_of(&[0, 0]).unwrap(), 1.0).unwrap();
//! let densities = walk.densities(&psi).unwrap();
//! assert!((densities.iter().sum::<f64>() - 2.0).abs() < 1e-10);
//! ```

use std::sync::{Arc, OnceLock};

use num_complex::Complex64 as C64;

use super::{Ranking, Statistics, Terms, count, propagate};
use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::graph::automorphism::is_automorphism;
use crate::linalg::{CsrMatrix, RMatrix, eigh_real};
use crate::walks::symmetric::check_permutation;
use crate::walks::{DENSE_LIMIT, Generator};

/// Most configurations whose orbits are found (`2²⁷`; a 4-byte word each,
/// 512 MiB, and a bit more for the orbits without an invariant state).
pub const MAX_ORBIT_CONFIGURATIONS: usize = 1 << 27;

/// Most steps of the union–find (generators times configurations).
pub const MAX_ORBIT_STEPS: u64 = 1 << 34;

/// Most entries of the reduced Hamiltonian (an upper bound is checked
/// before it is built).
pub const MAX_ORBIT_ENTRIES: usize = 1 << 26;

/// Largest `n²` of [`SymmetricInteractingWalk::pair_correlations`].
const MAX_PAIR_ENTRIES: usize = 1 << 26;

/// Sign bit of an orbit word: `φ(c) = −1`.
const SIGN: u32 = 1 << 31;

/// Orbit word of a configuration whose orbit carries no invariant state.
const DEAD: u32 = SIGN - 1;

/// The orbits of a group of node permutations on the configurations of
/// `particles` particles, with the sign of every configuration relative to
/// its orbit's first configuration in rank order (its representative).
/// Orbits without an invariant state (see the module documentation) are
/// left out of the numbering.
#[derive(Clone, Debug)]
pub struct ConfigurationOrbits {
    n: usize,
    particles: usize,
    statistics: Statistics,
    ranking: Ranking,
    count: usize,
    /// Per configuration (by rank): its orbit, or [`DEAD`], and its sign in
    /// bit 31.
    of: Vec<u32>,
    /// Sites of the representative of every orbit.
    representatives: Vec<usize>,
    sizes: Vec<u32>,
}

/// Root and sign of `x` relative to it, with the path compressed.
fn find(parent: &mut [u32], x: usize) -> (usize, u32) {
    let (mut root, mut sign) = (x, 0);
    loop {
        let word = parent[root];
        let up = (word & !SIGN) as usize;
        if up == root {
            break;
        }
        sign ^= word & SIGN;
        root = up;
    }
    // Point every node on the path at the root, with its sign to it.
    let (mut y, mut to_root) = (x, sign);
    while y != root {
        let word = parent[y];
        parent[y] = root as u32 | to_root;
        to_root ^= word & SIGN;
        y = (word & !SIGN) as usize;
    }
    (root, sign)
}

/// Sort `sites` and return the sign of the sorting permutation (in bit 31).
fn sort_with_sign(sites: &mut [usize]) -> u32 {
    let mut swaps = 0;
    // Insertion sort: few particles, and the number of swaps is the
    // number of inversions.
    for i in 1..sites.len() {
        let mut j = i;
        while j > 0 && sites[j - 1] > sites[j] {
            sites.swap(j - 1, j);
            swaps += 1;
            j -= 1;
        }
    }
    if swaps % 2 == 1 { SIGN } else { 0 }
}

impl ConfigurationOrbits {
    /// The orbits of the group generated by `generators` (permutations
    /// `u ↦ g[u]` of the `n` nodes; none for the trivial group) on the
    /// configurations of `particles` particles of the given statistics.
    pub fn new(
        n: usize,
        statistics: Statistics,
        particles: usize,
        generators: &[Vec<usize>],
    ) -> Result<Self> {
        if n == 0 {
            return Err(QsoError::invalid("n", "need at least one node"));
        }
        if particles == 0 {
            return Err(QsoError::invalid("particles", "need at least one particle"));
        }
        if statistics == Statistics::Fermions && particles > n {
            return Err(QsoError::invalid(
                "particles",
                format!("{particles} fermions do not fit on {n} sites"),
            ));
        }
        for (k, g) in generators.iter().enumerate() {
            check_permutation(n, g, k)?;
        }
        let too_many = |what: &str| {
            QsoError::invalid(
                "particles",
                format!(
                    "{particles} {} on {n} sites: too many {what}",
                    statistics.name()
                ),
            )
        };
        let total = count(statistics, n, particles, MAX_ORBIT_CONFIGURATIONS).ok_or_else(|| {
            too_many(&format!(
                "configurations for their orbits (more than {MAX_ORBIT_CONFIGURATIONS})"
            ))
        })?;
        if (generators.len() as u64).saturating_mul(total as u64) > MAX_ORBIT_STEPS {
            return Err(QsoError::invalid(
                "generators",
                format!(
                    "{} generators on {total} configurations exceed {MAX_ORBIT_STEPS} steps: \
                     give a smaller generating set",
                    generators.len()
                ),
            ));
        }
        let ranking =
            Ranking::new(n, particles, statistics, MAX_ORBIT_CONFIGURATIONS).map_err(too_many)?;
        // Union–find with signs: `ψ(b) = ±ψ(a)` along every link; a
        // contradiction marks the root.
        let mut parent: Vec<u32> = (0..total as u32).collect();
        let mut dead = vec![0u64; total.div_ceil(64)];
        let mut image = vec![0usize; particles];
        ranking.for_each(total, |a, sites| {
            for g in generators {
                for (x, &s) in image.iter_mut().zip(sites) {
                    *x = g[s];
                }
                let sign = match statistics {
                    Statistics::Fermions => sort_with_sign(&mut image),
                    Statistics::Bosons => {
                        image.sort_unstable();
                        0
                    }
                    Statistics::Distinguishable => 0,
                };
                let b = ranking.rank(&image);
                let (ra, sa) = find(&mut parent, a);
                let (rb, sb) = find(&mut parent, b);
                let relation = sa ^ sb ^ sign;
                if ra == rb {
                    if relation != 0 {
                        dead[ra / 64] |= 1 << (ra % 64);
                    }
                } else {
                    let (lo, hi) = (ra.min(rb), ra.max(rb));
                    parent[hi] = lo as u32 | relation;
                    if dead[hi / 64] >> (hi % 64) & 1 == 1 {
                        dead[lo / 64] |= 1 << (lo % 64);
                    }
                }
            }
        });
        for a in 0..total {
            find(&mut parent, a);
        }
        // Number the orbits in the order of their first configuration.
        // Every configuration now points at its root, which comes first and
        // is relabelled before it.
        let mut representatives = Vec::new();
        let mut sizes: Vec<u32> = Vec::new();
        ranking.for_each(total, |a, sites| {
            let word = parent[a];
            let root = (word & !SIGN) as usize;
            if root == a {
                let label = if dead[a / 64] >> (a % 64) & 1 == 1 {
                    DEAD
                } else {
                    representatives.extend_from_slice(sites);
                    sizes.push(1);
                    (sizes.len() - 1) as u32
                };
                parent[a] = label;
            } else {
                let label = parent[root];
                parent[a] = label | (word & SIGN);
                if label != DEAD {
                    sizes[label as usize] += 1;
                }
            }
        });
        Ok(ConfigurationOrbits {
            n,
            particles,
            statistics,
            ranking,
            count: total,
            of: parent,
            representatives,
            sizes,
        })
    }

    pub fn num_nodes(&self) -> usize {
        self.n
    }

    pub fn particles(&self) -> usize {
        self.particles
    }

    pub fn statistics(&self) -> Statistics {
        self.statistics
    }

    /// Number of configurations.
    pub fn num_configurations(&self) -> usize {
        self.count
    }

    /// Number of orbits that carry an invariant state.
    pub fn len(&self) -> usize {
        self.sizes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sizes.is_empty()
    }

    /// Sites of the representative of orbit `o` (its first configuration).
    pub fn representative(&self, o: usize) -> &[usize] {
        &self.representatives[o * self.particles..(o + 1) * self.particles]
    }

    /// Number of configurations in orbit `o`.
    pub fn size(&self, o: usize) -> usize {
        self.sizes[o] as usize
    }

    /// Orbit and sign `φ = ±1` of the configuration with particles on
    /// `sites`; `None` if its orbit carries no invariant state.
    pub fn locate(&self, sites: &[usize]) -> Result<Option<(usize, f64)>> {
        let sorted = self.check_sites(sites)?;
        Ok(self.locate_sorted(&sorted))
    }

    fn check_sites(&self, sites: &[usize]) -> Result<Vec<usize>> {
        if sites.len() != self.particles {
            return Err(QsoError::DimensionMismatch {
                expected: self.particles,
                actual: sites.len(),
            });
        }
        if let Some(&node) = sites.iter().find(|&&s| s >= self.n) {
            return Err(QsoError::NodeOutOfRange {
                node,
                n_nodes: self.n,
            });
        }
        let mut sorted = sites.to_vec();
        if self.statistics != Statistics::Distinguishable {
            sorted.sort_unstable();
        }
        if self.statistics == Statistics::Fermions && sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(QsoError::invalid(
                "sites",
                "two fermions cannot occupy the same site",
            ));
        }
        Ok(sorted)
    }

    /// [`locate`](Self::locate) for a configuration in rank order.
    fn locate_sorted(&self, sorted: &[usize]) -> Option<(usize, f64)> {
        let word = self.of[self.ranking.rank(sorted)];
        let label = word & !SIGN;
        (label != DEAD).then(|| (label as usize, if word & SIGN == 0 { 1.0 } else { -1.0 }))
    }

    /// `f(sites, orbit, φ)` for every configuration whose orbit carries an
    /// invariant state, in rank order.
    fn for_each_live(&self, mut f: impl FnMut(&[usize], usize, f64)) {
        self.ranking.for_each(self.count, |a, sites| {
            let word = self.of[a];
            let label = word & !SIGN;
            if label != DEAD {
                f(
                    sites,
                    label as usize,
                    if word & SIGN == 0 { 1.0 } else { -1.0 },
                );
            }
        });
    }
}

/// An interacting walk on the states invariant under a group of the
/// graph's automorphisms (see the module documentation).
#[derive(Clone, Debug)]
pub struct SymmetricInteractingWalk {
    terms: Terms,
    orbits: Arc<ConfigurationOrbits>,
    hamiltonian: CsrMatrix,
    /// Eigenvalues and eigenvectors, computed on first use (`None` above
    /// [`DENSE_LIMIT`] orbits).
    spectral: OnceLock<Option<(Vec<f64>, RMatrix)>>,
}

impl SymmetricInteractingWalk {
    /// `particles` particles of the given statistics on `graph` with
    /// hopping rate `gamma`, no interaction, reduced to the states
    /// invariant under the group generated by `symmetries` (each an
    /// automorphism of the graph, weights included).
    pub fn new(
        graph: &Graph,
        generator: Generator,
        gamma: f64,
        statistics: Statistics,
        particles: usize,
        symmetries: &[Vec<usize>],
    ) -> Result<Self> {
        if !(gamma.is_finite() && gamma > 0.0) {
            return Err(QsoError::invalid(
                "gamma",
                "hopping rate must be positive and finite",
            ));
        }
        let n = graph.num_nodes();
        for (k, g) in symmetries.iter().enumerate() {
            check_permutation(n, g, k)?;
            if !is_automorphism(graph, g) {
                return Err(QsoError::invalid(
                    "symmetries",
                    format!("permutation {k} is not an automorphism of the graph"),
                ));
            }
        }
        let orbits = ConfigurationOrbits::new(n, statistics, particles, symmetries)?;
        let max_degree = (0..n).map(|u| graph.neighbors(u).len()).max().unwrap_or(0);
        let entries = particles
            .checked_mul(max_degree)
            .and_then(|x| x.checked_add(1))
            .and_then(|x| x.checked_mul(orbits.len()));
        if entries.is_none_or(|x| x > MAX_ORBIT_ENTRIES) {
            return Err(QsoError::invalid(
                "particles",
                format!(
                    "the reduced Hamiltonian may have more than {MAX_ORBIT_ENTRIES} entries \
                     ({} orbits)",
                    orbits.len()
                ),
            ));
        }
        let mut walk = SymmetricInteractingWalk {
            terms: Terms::new(graph, generator, gamma, statistics),
            orbits: Arc::new(orbits),
            hamiltonian: CsrMatrix::from_triplets(0, &[])?,
            spectral: OnceLock::new(),
        };
        walk.build()?;
        Ok(walk)
    }

    /// The walk started from the configuration with particles on `start`,
    /// reduced by the automorphisms that fix each of its occupied nodes
    /// ([`Graph::stabilizer`]); its orbit is the configuration alone.
    pub fn from_start(
        graph: &Graph,
        generator: Generator,
        gamma: f64,
        statistics: Statistics,
        start: &[usize],
    ) -> Result<Self> {
        let group = graph.stabilizer(start)?;
        Self::new(
            graph,
            generator,
            gamma,
            statistics,
            start.len(),
            group.generators(),
        )
    }

    /// The same walk with on-site interaction `on_site` (U) and interaction
    /// `neighbour` (V) between particles on adjacent sites.
    pub fn with_interaction(mut self, on_site: f64, neighbour: f64) -> Result<Self> {
        if !(on_site.is_finite() && neighbour.is_finite()) {
            return Err(QsoError::invalid(
                "interaction",
                "interaction strengths must be finite",
            ));
        }
        if (on_site, neighbour) != (self.terms.on_site, self.terms.neighbour) {
            self.terms.on_site = on_site;
            self.terms.neighbour = neighbour;
            self.build()?;
        }
        Ok(self)
    }

    pub fn orbits(&self) -> &Arc<ConfigurationOrbits> {
        &self.orbits
    }

    /// Number of orbits: the dimension of the reduced walk.
    pub fn num_orbits(&self) -> usize {
        self.orbits.len()
    }

    /// Number of configurations of the unreduced walk.
    pub fn num_configurations(&self) -> usize {
        self.orbits.num_configurations()
    }

    pub fn num_nodes(&self) -> usize {
        self.orbits.n
    }

    pub fn particles(&self) -> usize {
        self.orbits.particles
    }

    pub fn statistics(&self) -> Statistics {
        self.orbits.statistics
    }

    pub fn generator(&self) -> Generator {
        self.terms.generator
    }

    pub fn gamma(&self) -> f64 {
        self.terms.gamma
    }

    /// On-site interaction U.
    pub fn on_site(&self) -> f64 {
        self.terms.on_site
    }

    /// Interaction V between neighbours.
    pub fn neighbour(&self) -> f64 {
        self.terms.neighbour
    }

    /// The Hamiltonian on the orbits, `⟨O′|H|O⟩`.
    pub fn hamiltonian(&self) -> &CsrMatrix {
        &self.hamiltonian
    }

    /// Eigenvalues of the reduced Hamiltonian (ascending), up to
    /// [`DENSE_LIMIT`] orbits.
    pub fn spectrum(&self) -> Option<&[f64]> {
        self.spectral().map(|(v, _)| v.as_slice())
    }

    /// The invariant state through the configuration with particles on
    /// `sites`, `|O|^{−1/2} Σ_{c∈O} φ(c)|c⟩` signed so that `|sites⟩` has a
    /// positive amplitude: the configuration itself when the group fixes
    /// it. An error if its orbit carries no invariant state.
    pub fn state_of(&self, sites: &[usize]) -> Result<Vec<C64>> {
        let (o, sign) = self.orbits.locate(sites)?.ok_or_else(|| {
            QsoError::invalid(
                "sites",
                "no invariant state contains this configuration (a symmetry maps it to minus \
                 itself)",
            )
        })?;
        let mut x = vec![C64::new(0.0, 0.0); self.num_orbits()];
        x[o] = C64::new(sign, 0.0);
        Ok(x)
    }

    /// `exp(−iHt)` applied to the orbit amplitudes `x0`.
    pub fn evolve(&self, x0: &[C64], t: f64) -> Result<Vec<C64>> {
        self.check_state(x0)?;
        propagate(&self.hamiltonian, self.spectral(), x0, t)
    }

    /// Probabilities of the orbits, `|x_O|²` (each shared equally by the
    /// configurations of the orbit).
    pub fn orbit_probabilities(&self, x: &[C64]) -> Result<Vec<f64>> {
        self.check_state(x)?;
        Ok(x.iter().map(|a| a.norm_sqr()).collect())
    }

    /// Probability of the configuration with particles on `sites`:
    /// `|x_O|²/|O|`, zero on orbits without an invariant state.
    pub fn probability_of(&self, x: &[C64], sites: &[usize]) -> Result<f64> {
        self.check_state(x)?;
        Ok(match self.orbits.locate(sites)? {
            Some((o, _)) => x[o].norm_sqr() / self.orbits.size(o) as f64,
            None => 0.0,
        })
    }

    /// Mean occupations `⟨n_i⟩` of the sites (one pass over all
    /// configurations).
    pub fn densities(&self, x: &[C64]) -> Result<Vec<f64>> {
        self.check_state(x)?;
        let per = self.per_configuration(x);
        let mut out = vec![0.0; self.num_nodes()];
        self.orbits.for_each_live(|sites, o, _| {
            for &s in sites {
                out[s] += per[o];
            }
        });
        Ok(out)
    }

    /// Density correlations `⟨n_i n_j⟩` (one pass over all
    /// configurations; at most 2²⁶ pairs of nodes).
    pub fn pair_correlations(&self, x: &[C64]) -> Result<RMatrix> {
        self.check_state(x)?;
        let n = self.num_nodes();
        if n.checked_mul(n).is_none_or(|p| p > MAX_PAIR_ENTRIES) {
            return Err(QsoError::invalid(
                "x",
                format!("{n}² pairs of nodes exceed {MAX_PAIR_ENTRIES}"),
            ));
        }
        let per = self.per_configuration(x);
        let mut out = RMatrix::zeros(n, n);
        self.orbits.for_each_live(|sites, o, _| {
            for &i in sites {
                for &j in sites {
                    out[(i, j)] += per[o];
                }
            }
        });
        Ok(out)
    }

    /// The amplitude on every configuration, orbit by orbit, `φ(c) x_O /
    /// √|O|`, in the order of [`InteractingWalk`](super::InteractingWalk)
    /// (at most [`super::MAX_CONFIGURATIONS`] configurations).
    pub fn lift(&self, x: &[C64]) -> Result<Vec<C64>> {
        self.check_state(x)?;
        let total = self.num_configurations();
        if total > super::MAX_CONFIGURATIONS {
            return Err(QsoError::invalid(
                "x",
                format!(
                    "{total} configurations exceed the {} of an unreduced walk",
                    super::MAX_CONFIGURATIONS
                ),
            ));
        }
        let mut out = vec![C64::new(0.0, 0.0); total];
        let norms: Vec<f64> = (0..self.num_orbits())
            .map(|o| (self.orbits.size(o) as f64).sqrt().recip())
            .collect();
        for (a, &word) in self.orbits.of.iter().enumerate() {
            let label = word & !SIGN;
            if label != DEAD {
                let sign = if word & SIGN == 0 { 1.0 } else { -1.0 };
                out[a] = x[label as usize] * (sign * norms[label as usize]);
            }
        }
        Ok(out)
    }

    /// Probability per configuration of every orbit, `|x_O|²/|O|`.
    fn per_configuration(&self, x: &[C64]) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(o, a)| a.norm_sqr() / self.orbits.size(o) as f64)
            .collect()
    }

    fn check_state(&self, x: &[C64]) -> Result<()> {
        if x.len() != self.num_orbits() {
            return Err(QsoError::DimensionMismatch {
                expected: self.num_orbits(),
                actual: x.len(),
            });
        }
        Ok(())
    }

    fn spectral(&self) -> Option<&(Vec<f64>, RMatrix)> {
        self.spectral
            .get_or_init(|| {
                (self.num_orbits() <= DENSE_LIMIT).then(|| eigh_real(&self.hamiltonian.to_dense()))
            })
            .as_ref()
    }

    /// `⟨O′|H|O⟩ = √(|O|/|O′|) Σ_{c′∈O′} φ(c′)⟨c′|H|r⟩` from the terms at
    /// each representative `r`.
    fn build(&mut self) -> Result<()> {
        let orbits = &self.orbits;
        let d = orbits.len();
        let mut triplets = Vec::new();
        let mut scratch = self.terms.scratch(orbits.particles);
        for o in 0..d {
            let size = orbits.size(o) as f64;
            self.terms.apply(
                orbits.representative(o),
                &mut scratch,
                |target, amplitude| {
                    if let Some((o2, sign)) = orbits.locate_sorted(target) {
                        let ratio = (size / orbits.size(o2) as f64).sqrt();
                        triplets.push((o2, o, sign * ratio * amplitude));
                    }
                },
            );
        }
        self.hamiltonian = CsrMatrix::from_triplets(d, &triplets)?;
        self.spectral = OnceLock::new();
        Ok(())
    }
}

#[cfg(test)]
mod tests;
