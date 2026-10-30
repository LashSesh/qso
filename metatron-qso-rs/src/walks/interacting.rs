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
//! the walk is solved by a dense eigendecomposition (computed on first
//! use), above by Krylov propagation.
//!
//! Without interaction the amplitudes are permanents (bosons) and
//! determinants (fermions) of submatrices of the single-particle
//! propagator. On a strongly regular graph that propagator `exp(−iAt)` is
//! a combination of `I`, `A` and `J − I − A` whose coefficients depend
//! only on the graph's parameters, so one walker, and two non-interacting
//! particles too, see no difference between strongly regular graphs with
//! the same parameters. Interacting bosons do (Gamble, Friesen, Zhou,
//! Joynt and Coppersmith, Phys. Rev. A 81, 052313 (2010)), though not on
//! every pair of graphs (Smith, arXiv:1004.0206 (2010)).
//! [`InteractingWalk::certificate`] turns this into a graph invariant: the
//! sorted transition probabilities between all configurations at given
//! times.
//!
//! ```
//! use metatron_qso::graph::generators;
//! use metatron_qso::walks::interacting::certificate_distance;
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
//! let tolerance = walk(&a, 0.0).certificate_tolerance(&times).unwrap();
//! assert!(certificate_distance(&free, &free_b) < tolerance);
//! let (ha, hb) = (walk(&a, 1.0), walk(&b, 1.0));
//! let distance = certificate_distance(&ha.certificate(&times).unwrap(), &hb.certificate(&times).unwrap());
//! assert!(distance > ha.certificate_tolerance(&times).unwrap());
//! ```

use std::sync::OnceLock;

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::{DENSE_LIMIT, Generator};
use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::linalg::{CMatrix, CsrMatrix, KrylovOptions, RMatrix, eigh_real, expm_multiply};

/// Largest number of configurations accepted (`2²²`, 4 194 304).
pub const MAX_CONFIGURATIONS: usize = 1 << 22;

/// Largest number of stored sites: configurations times particles, and
/// the entries of the ranking table (`2²⁵`).
const MAX_STORED_SITES: usize = 1 << 25;

/// Largest number of nonzero entries of the Hamiltonian (`2²⁵`, about
/// 1.3 GB while it is built).
const MAX_NONZEROS: usize = 1 << 25;

/// Largest certificate (`2²⁶` values, 512 MB).
const MAX_CERTIFICATE: usize = 1 << 26;

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
        Statistics::Bosons => binomial((n - 1).checked_add(particles)?, particles),
        Statistics::Fermions => binomial(n, particles),
        Statistics::Distinguishable => {
            let mut d: usize = 1;
            for _ in 0..particles {
                d = d.checked_mul(n).filter(|&d| d <= MAX_CONFIGURATIONS)?;
                if n == 1 {
                    break;
                }
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
    /// Sorted neighbours of every node, for the neighbour interaction.
    neighbours: Vec<Vec<usize>>,
    /// Binomial table `C(m, k)` for ranking (bosons and fermions):
    /// `binom[m][k]`.
    binom: Vec<Vec<usize>>,
    /// Configurations, `particles` sites each, in the order of their rank.
    configurations: Vec<usize>,
    hamiltonian: CsrMatrix,
    /// Eigenvalues and eigenvectors, computed on first use (`None` above
    /// [`DENSE_LIMIT`] configurations).
    spectral: OnceLock<Option<(Vec<f64>, RMatrix)>>,
}

impl InteractingWalk {
    /// `particles ≥ 1` particles of the given statistics on `graph`, with
    /// hopping rate `gamma > 0` and no interaction.
    ///
    /// Walks with more than [`MAX_CONFIGURATIONS`] configurations, more
    /// than 2²⁵ stored sites (configurations times particles) or a
    /// Hamiltonian with more than 2²⁵ nonzero entries are rejected.
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
        let too_many = |what: &str| {
            QsoError::invalid(
                "particles",
                format!(
                    "{particles} {} on {n} sites: too many {what}",
                    statistics.name()
                ),
            )
        };
        let dimension = count(statistics, n, particles)
            .ok_or_else(|| too_many(&format!("configurations (more than {MAX_CONFIGURATIONS})")))?;
        if dimension
            .checked_mul(particles)
            .is_none_or(|s| s > MAX_STORED_SITES)
        {
            return Err(too_many("stored sites (more than 2²⁵)"));
        }
        // Ranks of sets of `particles` elements of {0, …, n + particles − 2}
        // (bosons as sets after shifting by position).
        let binom = if statistics == Statistics::Distinguishable {
            Vec::new()
        } else {
            let universe = n
                .checked_add(particles)
                .ok_or_else(|| too_many("particles"))?;
            if (universe + 1)
                .checked_mul(particles + 1)
                .is_none_or(|s| s > MAX_STORED_SITES)
            {
                return Err(too_many("particles for the ranking table"));
            }
            (0..=universe)
                .map(|m| {
                    (0..=particles)
                        .map(|k| binomial(m, k).unwrap_or(usize::MAX))
                        .collect()
                })
                .collect()
        };
        // At most one diagonal entry and one entry per particle and
        // neighbour of its site in every column.
        let max_degree = (0..n).map(|u| graph.neighbors(u).len()).max().unwrap_or(0);
        let nonzeros = particles
            .checked_mul(max_degree)
            .and_then(|x| x.checked_add(1))
            .and_then(|x| x.checked_mul(dimension));
        if nonzeros.is_none_or(|x| x > MAX_NONZEROS) {
            return Err(too_many("nonzero Hamiltonian entries (more than 2²⁵)"));
        }
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
        let neighbours = (0..n)
            .map(|u| {
                let mut v: Vec<usize> = graph.neighbors(u).iter().map(|&(v, _)| v).collect();
                v.sort_unstable();
                v
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
            neighbours,
            binom,
            configurations: Vec::new(),
            hamiltonian: CsrMatrix::from_triplets(0, &[])?,
            spectral: OnceLock::new(),
        };
        walk.configurations = walk.enumerate(dimension);
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
        if (on_site, neighbour) != (self.on_site, self.neighbour) {
            self.on_site = on_site;
            self.neighbour = neighbour;
            self.build()?;
        }
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
        Ok(self.config(index))
    }

    /// The many-particle Hamiltonian in the occupation basis.
    pub fn hamiltonian(&self) -> &CsrMatrix {
        &self.hamiltonian
    }

    /// Eigenvalues of the Hamiltonian (ascending), up to [`DENSE_LIMIT`]
    /// configurations; the first call diagonalises the Hamiltonian.
    pub fn spectrum(&self) -> Option<&[f64]> {
        self.spectral().map(|(v, _)| v.as_slice())
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
        match self.spectral() {
            Some((values, vectors)) => {
                let mut c = vec![C64::new(0.0, 0.0); d];
                for (k, ck) in c.iter_mut().enumerate() {
                    let column = vectors.column(k);
                    let acc: C64 = psi0.iter().zip(column.iter()).map(|(p, &v)| p * v).sum();
                    *ck = acc * C64::from_polar(1.0, -values[k] * t);
                }
                let mut out = vec![C64::new(0.0, 0.0); d];
                for (k, ck) in c.iter().enumerate() {
                    for (o, &v) in out.iter_mut().zip(vectors.column(k).iter()) {
                        *o += ck * v;
                    }
                }
                Ok(out)
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
    /// to the number of particles times the squared norm of `psi`.
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
        // V·diag(e^{−iλt})·Vᵀ as two real products.
        let (mut re, mut im) = (vectors.clone(), vectors.clone());
        for (k, &lambda) in values.iter().enumerate() {
            let (sin, cos) = (lambda * t).sin_cos();
            re.column_mut(k).scale_mut(cos);
            im.column_mut(k).scale_mut(-sin);
        }
        let (re, im) = (re * vectors.transpose(), im * vectors.transpose());
        Ok(CMatrix::from_fn(re.nrows(), re.ncols(), |r, c| {
            C64::new(re[(r, c)], im[(r, c)])
        }))
    }

    /// Transition probabilities `|⟨a|exp(−iHt)|b⟩|²` between all pairs of
    /// configurations.
    pub fn transition_probabilities(&self, t: f64) -> Result<RMatrix> {
        Ok(self.propagator(t)?.map(|z| z.norm_sqr()))
    }

    /// Graph certificate: for each time, the transition probabilities
    /// between all pairs of configurations, sorted; the lists of all times
    /// are concatenated (at most 2²⁶ values). Isomorphic graphs give the
    /// same certificate up to rounding, which grows with the time; graphs
    /// whose certificates differ ([`certificate_distance`]) by more than
    /// [`certificate_tolerance`](Self::certificate_tolerance) are not
    /// isomorphic, up to floating-point error. The converse does not hold:
    /// without interaction, strongly regular graphs with the same
    /// parameters have the same certificate, and with interaction some
    /// non-isomorphic graphs do too.
    pub fn certificate(&self, times: &[f64]) -> Result<Vec<f64>> {
        self.check_times(times)?;
        let d = self.dimension();
        if times
            .len()
            .checked_mul(d * d)
            .is_none_or(|x| x > MAX_CERTIFICATE)
        {
            return Err(QsoError::invalid(
                "times",
                format!(
                    "{} times × {d}² configurations: more than 2²⁶ values",
                    times.len()
                ),
            ));
        }
        let mut out = Vec::with_capacity(times.len() * d * d);
        for &t in times {
            let mut p: Vec<f64> = self.transition_probabilities(t)?.iter().copied().collect();
            p.sort_by(f64::total_cmp);
            out.extend(p);
        }
        Ok(out)
    }

    /// Tolerance for comparing certificates at `times`:
    /// `10⁻¹² · max(1, |t|max · ‖H‖)`. The rounding of the eigenvalues
    /// turns into phase errors that grow linearly with the time; measured
    /// distances between certificates of isomorphic graphs stay below
    /// 10⁻¹⁵ · |t| · ‖H‖, a thousandth of this tolerance.
    pub fn certificate_tolerance(&self, times: &[f64]) -> Result<f64> {
        self.check_times(times)?;
        let (values, _) = self.require_dense()?;
        let norm = values.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let t = times.iter().fold(0.0f64, |m, t| m.max(t.abs()));
        Ok(1e-12 * (t * norm).max(1.0))
    }

    fn check_times(&self, times: &[f64]) -> Result<()> {
        if times.is_empty() {
            return Err(QsoError::invalid("times", "need at least one time"));
        }
        if times.iter().any(|t| !t.is_finite()) {
            return Err(QsoError::invalid("times", "times must be finite"));
        }
        Ok(())
    }

    fn spectral(&self) -> Option<&(Vec<f64>, RMatrix)> {
        self.spectral
            .get_or_init(|| {
                (self.dimension() <= DENSE_LIMIT).then(|| eigh_real(&self.hamiltonian.to_dense()))
            })
            .as_ref()
    }

    fn require_dense(&self) -> Result<&(Vec<f64>, RMatrix)> {
        self.spectral().ok_or_else(|| {
            QsoError::invalid(
                "dimension",
                format!(
                    "{} configurations; the dense propagator needs at most {DENSE_LIMIT}",
                    self.dimension()
                ),
            )
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

    /// All `dimension` configurations in the order of their rank, one
    /// successor at a time.
    fn enumerate(&self, dimension: usize) -> Vec<usize> {
        let p = self.particles;
        let mut out = Vec::with_capacity(dimension * p);
        match self.statistics {
            Statistics::Distinguishable => {
                // Base-n digits, the last particle the least significant.
                let mut sites = vec![0; p];
                for _ in 0..dimension {
                    out.extend_from_slice(&sites);
                    for s in sites.iter_mut().rev() {
                        *s += 1;
                        if *s < self.n {
                            break;
                        }
                        *s = 0;
                    }
                }
            }
            Statistics::Fermions | Statistics::Bosons => {
                // Colex successor of the set c₀ < c₁ < …: raise the lowest
                // element that has room below its upper neighbour and reset
                // the elements under it.
                let shift = usize::from(self.statistics == Statistics::Bosons);
                let top = self.n + shift * (p - 1);
                let mut c: Vec<usize> = (0..p).collect();
                for _ in 0..dimension {
                    out.extend(c.iter().enumerate().map(|(k, &x)| x - shift * k));
                    for k in 0..p {
                        let limit = if k + 1 < p { c[k + 1] } else { top };
                        if c[k] + 1 < limit {
                            c[k] += 1;
                            for (j, cj) in c[..k].iter_mut().enumerate() {
                                *cj = j;
                            }
                            break;
                        }
                    }
                }
            }
        }
        out
    }

    /// Build the Hamiltonian; the eigendecomposition follows on first use.
    fn build(&mut self) -> Result<()> {
        let d = self.dimension();
        let p = self.particles;
        let mut triplets = Vec::new();
        let mut occupation = vec![0usize; self.n];
        let mut distinct = Vec::with_capacity(p);
        let mut moved = vec![0usize; p];
        for a in 0..d {
            let sites = self.config(a);
            distinct.clear();
            for &s in sites {
                if occupation[s] == 0 {
                    distinct.push(s);
                }
                occupation[s] += 1;
            }
            // Diagonal: potential, pairs on one site, pairs on adjacent
            // sites (each edge once).
            let mut diag: f64 = sites.iter().map(|&s| self.diagonal[s]).sum();
            let same: usize = distinct
                .iter()
                .map(|&s| occupation[s] * (occupation[s] - 1) / 2)
                .sum();
            diag += self.on_site * same as f64;
            if self.neighbour != 0.0 {
                for (x, &u) in distinct.iter().enumerate() {
                    for &v in &distinct[x + 1..] {
                        if self.neighbours[u].binary_search(&v).is_ok() {
                            diag += self.neighbour * (occupation[u] * occupation[v]) as f64;
                        }
                    }
                }
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
            for &s in &distinct {
                occupation[s] = 0;
            }
        }
        self.hamiltonian = CsrMatrix::from_triplets(d, &triplets)?;
        self.spectral = OnceLock::new();
        Ok(())
    }
}

/// Largest difference between two certificates of
/// [`InteractingWalk::certificate`]: infinite if their lengths differ,
/// NaN if either holds NaN. Compare it with
/// [`InteractingWalk::certificate_tolerance`].
pub fn certificate_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let mut max = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        if d.is_nan() {
            return f64::NAN;
        }
        max = max.max(d);
    }
    max
}

#[cfg(test)]
mod tests;
