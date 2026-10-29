//! Walks of several particles on a graph: bosons, fermions or
//! distinguishable particles, with or without interaction.
//!
//! `N` particles hop along the edges of a graph with the single-particle
//! Hamiltonian `h` of a [`ContinuousWalk`](super::ContinuousWalk)
//! (`h = −γA` or `h = γL`), and interact on the same site and between
//! neighbouring sites:
//!
//! ```text
//! H = Σ_ij h_ij a†_i a_j + (U/2) Σ_i n_i (n_i − 1) + V Σ_{edges ij} n_i n_j
//! ```
//!
//! For bosons this is the Bose–Hubbard model on the graph, for spinless
//! fermions (which never share a site, so `U` has no effect) the model of
//! fermions with nearest-neighbour interaction. Distinguishable particles
//! hop independently and interact in the same way.
//!
//! The walk lives in the occupation basis: for bosons the multisets of `N`
//! sites (`C(n+N−1, N)` configurations), for fermions the sets of `N`
//! sites (`C(n, N)`), for distinguishable particles the tuples (`nᴺ`).
//! Configurations are listed as sites in ascending order (bosons and
//! fermions) or as the site of each particle (distinguishable), and the
//! fermionic sign follows that order. Up to [`DENSE_LIMIT`] configurations
//! the walk is solved by a dense eigendecomposition, above by Krylov
//! propagation.
//!
//! Without interaction the amplitudes are permanents (bosons) and
//! determinants (fermions) of submatrices of the single-particle
//! propagator. One walker's dynamics depend only on the spectrum and
//! eigenvectors of the graph, and two non-interacting particles still see
//! no difference between strongly regular graphs with the same parameters;
//! interacting bosons do (Gamble, Friesen, Zhou, Joynt and Coppersmith,
//! Phys. Rev. A 81, 052313 (2010)). [`InteractingWalk::certificate`] turns
//! this into a graph invariant: the sorted transition probabilities
//! between all configurations at given times.
//!
//! ```
//! use metatron_qso::graph::generators;
//! use metatron_qso::walks::{Generator, InteractingWalk, Statistics};
//!
//! // Two bosons on the Shrikhande and on the 4 × 4 rook's graph, both
//! // strongly regular (16, 6, 2, 2): without interaction the certificates
//! // agree, with an on-site interaction they do not.
//! let walk = |g, u| {
//!     InteractingWalk::new(g, Generator::Adjacency, 1.0, Statistics::Bosons, 2)
//!         .unwrap()
//!         .with_interaction(u, 0.0)
//!         .unwrap()
//! };
//! let (a, b) = (generators::shrikhande(), generators::rook(4, 4).unwrap());
//! let times = [1.0];
//! let free = walk(&a, 0.0).certificate(&times).unwrap();
//! let free_b = walk(&b, 0.0).certificate(&times).unwrap();
//! assert!(metatron_qso::walks::interacting::certificate_distance(&free, &free_b) < 1e-9);
//! let hub = walk(&a, 1.0).certificate(&times).unwrap();
//! let hub_b = walk(&b, 1.0).certificate(&times).unwrap();
//! assert!(metatron_qso::walks::interacting::certificate_distance(&hub, &hub_b) > 1e-6);
//! ```

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::{DENSE_LIMIT, Generator};
use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::linalg::{CMatrix, CsrMatrix, KrylovOptions, RMatrix, eigh_real, expm_multiply};

/// Largest number of configurations accepted (`2²²`, 4 194 304).
pub const MAX_CONFIGURATIONS: usize = 1 << 22;

/// Exchange statistics of the particles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Statistics {
    /// Any number of particles per site, symmetric states.
    Bosons,
    /// At most one particle per site, antisymmetric states.
    Fermions,
    /// Labelled particles, each with a site of its own.
    Distinguishable,
}

impl Statistics {
    pub fn name(self) -> &'static str {
        match self {
            Statistics::Bosons => "bosons",
            Statistics::Fermions => "fermions",
            Statistics::Distinguishable => "distinguishable",
        }
    }
}

/// Binomial coefficient, `None` if it exceeds [`MAX_CONFIGURATIONS`].
fn binomial(n: usize, k: usize) -> Option<usize> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut b: u128 = 1;
    for i in 0..k {
        b = b * (n - i) as u128 / (i + 1) as u128;
        if b > MAX_CONFIGURATIONS as u128 {
            return None;
        }
    }
    Some(b as usize)
}

/// Number of configurations, `None` beyond [`MAX_CONFIGURATIONS`].
fn count(statistics: Statistics, n: usize, particles: usize) -> Option<usize> {
    match statistics {
        Statistics::Bosons => binomial(n + particles - 1, particles),
        Statistics::Fermions => binomial(n, particles),
        Statistics::Distinguishable => {
            let mut d: usize = 1;
            for _ in 0..particles {
                d = d.checked_mul(n).filter(|&d| d <= MAX_CONFIGURATIONS)?;
            }
            Some(d)
        }
    }
}

/// Multi-particle walk on a graph (see the module documentation).
#[derive(Clone, Debug)]
pub struct InteractingWalk {
    n: usize,
    particles: usize,
    statistics: Statistics,
    generator: Generator,
    gamma: f64,
    on_site: f64,
    neighbour: f64,
    /// Single-particle Hamiltonian `h`: diagonal and off-diagonal entries.
    diagonal: Vec<f64>,
    hopping: Vec<Vec<(usize, f64)>>,
    /// The graph's edges, for the neighbour interaction.
    edges: Vec<(usize, usize)>,
    /// Binomial table `C(m, k)` for ranking: `binom[m][k]`.
    binom: Vec<Vec<usize>>,
    /// Configurations, `particles` sites each, in the order of their rank.
    configurations: Vec<usize>,
    hamiltonian: CsrMatrix,
    spectral: Option<(Vec<f64>, RMatrix)>,
}

impl InteractingWalk {
    /// `particles ≥ 1` particles of the given statistics on `graph`, with
    /// hopping rate `gamma > 0` and no interaction.
    pub fn new(
        graph: &Graph,
        generator: Generator,
        gamma: f64,
        statistics: Statistics,
        particles: usize,
    ) -> Result<Self> {
        if !(gamma.is_finite() && gamma > 0.0) {
            return Err(QsoError::invalid(
                "gamma",
                "hopping rate must be positive and finite",
            ));
        }
        let n = graph.num_nodes();
        if n == 0 {
            return Err(QsoError::invalid("graph", "graph has no nodes"));
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
        let dimension = count(statistics, n, particles).ok_or_else(|| {
            QsoError::invalid(
                "particles",
                format!(
                    "{particles} {} on {n} sites have more than \
                     {MAX_CONFIGURATIONS} configurations",
                    statistics.name()
                ),
            )
        })?;
        let h = match generator {
            Generator::Laplacian => graph.laplacian_csr().scaled(gamma),
            Generator::Adjacency => graph.adjacency_csr().scaled(-gamma),
        };
        let mut diagonal = vec![0.0; n];
        let mut hopping = vec![Vec::new(); n];
        for (i, row) in hopping.iter_mut().enumerate() {
            for (j, v) in h.row(i) {
                if i == j {
                    diagonal[i] = v;
                } else {
                    row.push((j, v));
                }
            }
        }
        // Ranks of sets of `particles` elements of a universe of
        // n + particles − 1 (bosons as sets after shifting by position).
        let universe = n + particles;
        let binom = (0..=universe)
            .map(|m| {
                (0..=particles)
                    .map(|k| binomial(m, k).unwrap_or(usize::MAX))
                    .collect()
            })
            .collect();
        let mut walk = Self {
            n,
            particles,
            statistics,
            generator,
            gamma,
            on_site: 0.0,
            neighbour: 0.0,
            diagonal,
            hopping,
            edges: graph.edges().iter().map(|&(u, v, _)| (u, v)).collect(),
            binom,
            configurations: Vec::new(),
            hamiltonian: CsrMatrix::from_triplets(0, &[])?,
            spectral: None,
        };
        let configurations = (0..dimension).flat_map(|r| walk.unrank(r)).collect();
        walk.configurations = configurations;
        walk.build()?;
        Ok(walk)
    }

    /// The same walk with on-site interaction `on_site` (U) and interaction
    /// `neighbour` (V) between particles on adjacent sites, each edge
    /// counted once whatever its weight.
    pub fn with_interaction(mut self, on_site: f64, neighbour: f64) -> Result<Self> {
        if !(on_site.is_finite() && neighbour.is_finite()) {
            return Err(QsoError::invalid(
                "interaction",
                "interaction strengths must be finite",
            ));
        }
        self.on_site = on_site;
        self.neighbour = neighbour;
        self.build()?;
        Ok(self)
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

    pub fn generator(&self) -> Generator {
        self.generator
    }

    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// On-site interaction U.
    pub fn on_site(&self) -> f64 {
        self.on_site
    }

    /// Interaction V between neighbours.
    pub fn neighbour(&self) -> f64 {
        self.neighbour
    }

    /// Number of configurations, the dimension of the walk.
    pub fn dimension(&self) -> usize {
        self.configurations.len() / self.particles
    }

    /// The sites of configuration `index` (ascending for bosons and
    /// fermions, particle by particle for distinguishable particles).
    pub fn configuration(&self, index: usize) -> Result<&[usize]> {
        if index >= self.dimension() {
            return Err(QsoError::invalid(
                "index",
                format!("configuration {index} of {}", self.dimension()),
            ));
        }
        let p = self.particles;
        Ok(&self.configurations[index * p..(index + 1) * p])
    }

    /// The many-particle Hamiltonian in the occupation basis.
    pub fn hamiltonian(&self) -> &CsrMatrix {
        &self.hamiltonian
    }

    /// Eigenvalues of the Hamiltonian (ascending), up to [`DENSE_LIMIT`]
    /// configurations.
    pub fn spectrum(&self) -> Option<&[f64]> {
        self.spectral.as_ref().map(|(v, _)| v.as_slice())
    }

    /// Index of the configuration with particles on `sites` (in any order
    /// for bosons and fermions; for distinguishable particles the site of
    /// each particle in turn).
    pub fn index_of(&self, sites: &[usize]) -> Result<usize> {
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
        Ok(self.rank(&sorted))
    }

    /// The basis state with particles on `sites`.
    pub fn basis_state(&self, sites: &[usize]) -> Result<Vec<C64>> {
        let mut psi = vec![C64::new(0.0, 0.0); self.dimension()];
        psi[self.index_of(sites)?] = C64::new(1.0, 0.0);
        Ok(psi)
    }

    /// `exp(−iHt)|ψ₀⟩`.
    pub fn evolve(&self, psi0: &[C64], t: f64) -> Result<Vec<C64>> {
        let d = self.dimension();
        if psi0.len() != d {
            return Err(QsoError::DimensionMismatch {
                expected: d,
                actual: psi0.len(),
            });
        }
        if !t.is_finite() {
            return Err(QsoError::invalid("t", "time must be finite"));
        }
        match &self.spectral {
            Some((values, vectors)) => {
                let mut c = vec![C64::new(0.0, 0.0); d];
                for (k, ck) in c.iter_mut().enumerate() {
                    let acc: C64 = (0..d).map(|j| psi0[j] * vectors[(j, k)]).sum();
                    *ck = acc * C64::from_polar(1.0, -values[k] * t);
                }
                Ok((0..d)
                    .map(|j| (0..d).map(|k| c[k] * vectors[(j, k)]).sum())
                    .collect())
            }
            None => {
                let h = &self.hamiltonian;
                expm_multiply(|x, y| h.matvec(x, y), psi0, t, &KrylovOptions::default())
            }
        }
    }

    /// Probabilities of all configurations at time `t`, started from the
    /// configuration with particles on `sites`.
    pub fn probabilities(&self, sites: &[usize], t: f64) -> Result<Vec<f64>> {
        let psi = self.evolve(&self.basis_state(sites)?, t)?;
        Ok(psi.iter().map(|a| a.norm_sqr()).collect())
    }

    /// Mean occupations `⟨n_i⟩` of the sites in the state `psi`; they sum
    /// to the number of particles times the norm.
    pub fn densities(&self, psi: &[C64]) -> Result<Vec<f64>> {
        self.check_state(psi)?;
        let mut out = vec![0.0; self.n];
        for (a, amp) in psi.iter().enumerate() {
            let p = amp.norm_sqr();
            for &s in self.config(a) {
                out[s] += p;
            }
        }
        Ok(out)
    }

    /// Density correlations `⟨n_i n_j⟩` in the state `psi` (the diagonal
    /// holds `⟨n_i²⟩`).
    pub fn pair_correlations(&self, psi: &[C64]) -> Result<RMatrix> {
        self.check_state(psi)?;
        let mut out = RMatrix::zeros(self.n, self.n);
        for (a, amp) in psi.iter().enumerate() {
            let p = amp.norm_sqr();
            let c = self.config(a);
            for &i in c {
                for &j in c {
                    out[(i, j)] += p;
                }
            }
        }
        Ok(out)
    }

    /// Dense propagator `exp(−iHt)` over the configurations, up to
    /// [`DENSE_LIMIT`] configurations.
    pub fn propagator(&self, t: f64) -> Result<CMatrix> {
        let (values, vectors) = self.require_dense()?;
        if !t.is_finite() {
            return Err(QsoError::invalid("t", "time must be finite"));
        }
        let v = vectors.map(|x| C64::new(x, 0.0));
        let mut left = v.clone();
        for (k, mut col) in left.column_iter_mut().enumerate() {
            col *= C64::from_polar(1.0, -values[k] * t);
        }
        Ok(left * v.transpose())
    }

    /// Transition probabilities `|⟨a|exp(−iHt)|b⟩|²` between all pairs of
    /// configurations.
    pub fn transition_probabilities(&self, t: f64) -> Result<RMatrix> {
        Ok(self.propagator(t)?.map(|z| z.norm_sqr()))
    }

    /// Graph certificate: for each time, the transition probabilities
    /// between all pairs of configurations, sorted; the lists of all times
    /// are concatenated. Isomorphic graphs give the same certificate (to
    /// rounding), so graphs whose certificates differ
    /// ([`certificate_distance`]) are not isomorphic. The converse does not
    /// hold: without interaction, strongly regular graphs with the same
    /// parameters have the same certificate.
    pub fn certificate(&self, times: &[f64]) -> Result<Vec<f64>> {
        if times.is_empty() {
            return Err(QsoError::invalid("times", "need at least one time"));
        }
        let mut out = Vec::with_capacity(times.len() * self.dimension().pow(2));
        for &t in times {
            let mut p: Vec<f64> = self.transition_probabilities(t)?.iter().copied().collect();
            p.sort_by(f64::total_cmp);
            out.extend(p);
        }
        Ok(out)
    }

    fn require_dense(&self) -> Result<&(Vec<f64>, RMatrix)> {
        self.spectral.as_ref().ok_or(QsoError::TooManyQubits {
            what: "dense interacting walk (configurations)",
            max: DENSE_LIMIT,
            requested: self.dimension(),
        })
    }

    fn check_state(&self, psi: &[C64]) -> Result<()> {
        if psi.len() != self.dimension() {
            return Err(QsoError::DimensionMismatch {
                expected: self.dimension(),
                actual: psi.len(),
            });
        }
        Ok(())
    }

    fn config(&self, index: usize) -> &[usize] {
        let p = self.particles;
        &self.configurations[index * p..(index + 1) * p]
    }

    /// Rank of a configuration (sorted for bosons and fermions).
    fn rank(&self, sites: &[usize]) -> usize {
        match self.statistics {
            Statistics::Distinguishable => sites.iter().fold(0, |r, &s| r * self.n + s),
            // Colex rank of the set {c_k}, c_k = s_k (+ k for bosons).
            Statistics::Fermions | Statistics::Bosons => {
                let shift = usize::from(self.statistics == Statistics::Bosons);
                sites
                    .iter()
                    .enumerate()
                    .map(|(k, &s)| self.binom[s + shift * k][k + 1])
                    .sum()
            }
        }
    }

    fn unrank(&self, mut r: usize) -> Vec<usize> {
        let p = self.particles;
        match self.statistics {
            Statistics::Distinguishable => {
                let mut sites = vec![0; p];
                for s in sites.iter_mut().rev() {
                    *s = r % self.n;
                    r /= self.n;
                }
                sites
            }
            Statistics::Fermions | Statistics::Bosons => {
                let shift = usize::from(self.statistics == Statistics::Bosons);
                let mut sites = vec![0; p];
                // Largest c with C(c, k + 1) ≤ r, from the last element down.
                let mut c = self.n + shift * (p - 1);
                for k in (0..p).rev() {
                    while self.binom[c][k + 1] > r {
                        c -= 1;
                    }
                    r -= self.binom[c][k + 1];
                    sites[k] = c - shift * k;
                }
                sites
            }
        }
    }

    /// Build the Hamiltonian (and its eigendecomposition if dense).
    fn build(&mut self) -> Result<()> {
        let d = self.dimension();
        let p = self.particles;
        let mut triplets = Vec::new();
        let mut occupation = vec![0usize; self.n];
        let mut moved = vec![0usize; p];
        for a in 0..d {
            let sites = self.config(a);
            for &s in sites {
                occupation[s] += 1;
            }
            // Diagonal: potential, on-site and neighbour interaction.
            let mut diag: f64 = sites.iter().map(|&s| self.diagonal[s]).sum();
            for (k, &s) in sites.iter().enumerate() {
                // Pairs on the same site and on adjacent sites, each once.
                for &q in &sites[k + 1..] {
                    if q == s {
                        diag += self.on_site;
                    }
                }
            }
            if self.neighbour != 0.0 {
                let pairs: usize = self
                    .edges
                    .iter()
                    .map(|&(u, v)| occupation[u] * occupation[v])
                    .sum();
                diag += self.neighbour * pairs as f64;
            }
            if diag != 0.0 {
                triplets.push((a, a, diag));
            }
            // Hopping of one particle from site j to a neighbour i.
            for (k, &j) in sites.iter().enumerate() {
                // Bosons: move each occupied site once (its occupation
                // enters the amplitude).
                if self.statistics == Statistics::Bosons && k > 0 && sites[k - 1] == j {
                    continue;
                }
                for &(i, h) in &self.hopping[j] {
                    let amplitude = match self.statistics {
                        Statistics::Distinguishable => h,
                        Statistics::Bosons => {
                            h * ((occupation[j] * (occupation[i] + 1)) as f64).sqrt()
                        }
                        Statistics::Fermions => {
                            if occupation[i] > 0 {
                                continue;
                            }
                            let (lo, hi) = (i.min(j), i.max(j));
                            let between = sites.iter().filter(|&&s| lo < s && s < hi).count();
                            if between % 2 == 0 { h } else { -h }
                        }
                    };
                    moved.copy_from_slice(sites);
                    moved[k] = i;
                    if self.statistics != Statistics::Distinguishable {
                        moved.sort_unstable();
                    }
                    triplets.push((self.rank(&moved), a, amplitude));
                }
            }
            for &s in self.config(a) {
                occupation[s] = 0;
            }
        }
        self.hamiltonian = CsrMatrix::from_triplets(d, &triplets)?;
        self.spectral = (d <= DENSE_LIMIT).then(|| eigh_real(&self.hamiltonian.to_dense()));
        Ok(())
    }
}

/// Largest difference between two certificates of
/// [`InteractingWalk::certificate`], infinite if their lengths differ.
/// Isomorphic graphs give distances at the level of rounding (below
/// 10⁻¹²); a distance above about 10⁻⁹ proves that two graphs are not
/// isomorphic.
pub fn certificate_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests;
