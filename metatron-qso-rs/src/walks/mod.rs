//! Continuous-time quantum walks (CTQW) on graphs.
//!
//! A walker on graph `G` lives in the n-dimensional space spanned by the
//! node states `|j⟩` and evolves as `|ψ(t)⟩ = exp(−iHt)|ψ(0)⟩` with
//! `H = γ·L` (Laplacian generator) or `H = −γ·A` (adjacency generator).
//! For real symmetric `H` node probabilities do not depend on the sign of
//! `H`, so both conventions in the literature agree.
//!
//! * [`ContinuousWalk`] — exact propagation through the eigendecomposition
//!   (n ≤ [`DENSE_LIMIT`]) or Krylov propagation on sparse matrices (any n);
//!   exact limiting and time-averaged distributions with correct handling of
//!   degenerate eigenvalues, beyond `DENSE_LIMIT` through symmetry
//!   reduction or the start node's cyclic subspace; optional site energies.
//! * [`search`] — Childs–Goldstone spatial search.
//! * [`quotient`] — exact walks and searches reduced by symmetry to a few
//!   dimensions (equitable partitions), for graphs of any size.
//! * [`transfer`] — perfect state transfer (decision and time) and the best
//!   transfer fidelity between two nodes.
//! * [`pretty_good`] — pretty good state transfer, decided exactly with
//!   cyclotomic arithmetic for paths, cycles and abelian Cayley graphs.
//! * [`circuits`] — walks as gate circuits: exact for circulant graphs and
//!   Cayley graphs of `Z₂ⁿ`, product formulas for any graph.
//! * [`dephasing`] — open-system walks with site dephasing (Lindblad),
//!   traps, losses and site energies.
//! * [`transport`] — transport to a trap: efficiency, mean arrival time,
//!   dark states, dephasing scans and the best dephasing rate, by linear
//!   solves instead of time steps.
//! * [`discrete`] — discrete-time walks: coined walks and their search,
//!   the walk on the line, and Szegedy walks with quantum hitting times.
//! * [`driven`] — walks with a time-dependent Hamiltonian (adiabatic and
//!   driven walks), by a fourth-order commutator-free Magnus integrator.
//! * [`decoherent`] — coined walks with coin and position dephasing, on
//!   density matrices.
//! * [`interacting`] — walks of several bosons, fermions or
//!   distinguishable particles with on-site and neighbour interaction, and
//!   graph certificates from them.
//! * [`classical`] — classical random-walk baselines.
//!
//! ```
//! use metatron_qso::graph::metatron;
//! use metatron_qso::walks::{ContinuousWalk, Generator};
//!
//! // On Metatron's Cube (K13) a walker started at the centre stays there
//! // with time-averaged probability exactly 145/169.
//! let walk = ContinuousWalk::new(&metatron::metatron_cube(), Generator::Laplacian, 1.0).unwrap();
//! let limit = walk.limiting_distribution(0).unwrap();
//! assert!((limit[0] - 145.0 / 169.0).abs() < 1e-12);
//! assert!((limit[5] - 2.0 / 169.0).abs() < 1e-12);
//! ```

pub mod circuits;
pub mod classical;
pub mod decoherent;
pub mod dephasing;
pub mod discrete;
pub mod driven;
pub mod interacting;
pub mod pretty_good;
pub mod quotient;
pub mod search;
pub mod symmetric;
pub mod transfer;
pub mod transport;

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::linalg::{
    CMatrix, CsrMatrix, KrylovOptions, RMatrix, cyclic_spectrum, eigh_real, expm_multiply,
};

pub use classical::{classical_hitting_times, classical_walk_probabilities};
pub use decoherent::DecoherentCoinedWalk;
pub use dephasing::DephasingWalk;
pub use discrete::{Coin, CoinedWalk, DiscreteSearchResult, LineWalk, SzegedyWalk};
pub use driven::{DrivenEvolution, DrivenWalk, MagnusOptions, QuotientDrivenWalk, QuotientTerm};
pub use interacting::{
    CoordinateGraph, CoordinateInteractingWalk, GroupInteractingWalk, HypercubeParticles,
    InteractingWalk, Statistics, SymmetricInteractingWalk,
};
pub use pretty_good::{PrettyGoodTransfer, TransferDecision};
pub use quotient::{Quotient, QuotientSearch, QuotientWalk};
pub use search::{MAX_RESOLUTION, SearchResult, SpatialSearch};
pub use symmetric::{
    Dephasing, HypercubeOpenWalk, Orbitals, ProductOpenWalk, SymmetricCoinedWalk,
    SymmetricOpenWalk, coordinate_permutations,
};
pub use transfer::{StateTransfer, TransferResult};
pub use transport::{
    DephasingOptimum, MAX_TRANSPORT_FACTOR_ENTRIES, MAX_TRANSPORT_WORK, SolveReport,
    TransportOptions, TransportResult, TransportState,
};

/// Largest graph for which dense eigendecompositions are used.
pub const DENSE_LIMIT: usize = 2048;

/// Which matrix generates the walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Generator {
    /// `H = γ·L` with `L = D − A`.
    Laplacian,
    /// `H = −γ·A`.
    Adjacency,
}

/// Relative tolerance below which eigenvalues count as equal: 10⁻¹¹ of the
/// largest eigenvalue magnitude. Degenerate eigenvalues of dense
/// eigendecompositions of up to 2048 nodes come out within 2·10⁻¹⁴ of
/// each other (complete graphs, hypercubes, tori, Johnson, Hamming and
/// Paley graphs measured), so the margin is a factor of about 500.
pub(crate) const EIGEN_TOL: f64 = 1e-11;

/// Group sorted eigenvalues into degenerate clusters (tolerance
/// [`EIGEN_TOL`] relative to the largest magnitude, so the grouping does
/// not depend on the overall scale of `H`).
pub(crate) fn eigenspaces(values: &[f64]) -> Vec<std::ops::Range<usize>> {
    let scale = values.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let tol = EIGEN_TOL * scale;
    let mut spaces = Vec::new();
    let mut start = 0;
    for i in 1..=values.len() {
        if i == values.len() || values[i] - values[i - 1] > tol {
            spaces.push(start..i);
            start = i;
        }
    }
    spaces
}

/// Spectral data of a start vector: eigenvalues `λ_k` (ascending), its
/// components `c_k = ⟨x_k|start⟩` and the eigenvectors `x_k` (columns), on
/// the full space or on the start vector's cyclic subspace.
pub(crate) struct StartSpectrum<'a> {
    pub values: std::borrow::Cow<'a, [f64]>,
    pub coeffs: Vec<f64>,
    pub vectors: std::borrow::Cow<'a, RMatrix>,
}

impl StartSpectrum<'_> {
    /// `P̄(j) = Σ_λ |⟨j|Π_λ|start⟩|²`, with degenerate eigenvalues grouped.
    pub fn limiting_distribution(&self) -> Vec<f64> {
        let n = self.vectors.nrows();
        let mut dist = vec![0.0; n];
        for space in eigenspaces(&self.values) {
            for (j, d) in dist.iter_mut().enumerate() {
                let amp: f64 = space
                    .clone()
                    .map(|k| self.vectors[(j, k)] * self.coeffs[k])
                    .sum();
                *d += amp * amp;
            }
        }
        dist
    }

    /// Exact time average `(1/T)∫₀ᵀ P(j, t) dt` (the limit for `T = ∞`).
    pub fn average_distribution(&self, t_max: f64) -> Vec<f64> {
        if t_max.is_infinite() {
            return self.limiting_distribution();
        }
        let m = self.vectors.ncols();
        // P(j,t) = Σ_ab c_a c_b x_a(j) x_b(j) e^{−i(λa−λb)t}; the average
        // replaces the phase by (1/T)∫ e^{−idt} dt, whose real part is
        // sin(dT)/(dT). With M_ab = c_a c_b sinc((λa−λb)T), the average is
        // the diagonal of X·M·Xᵀ.
        let sinc = |d: f64| {
            let x = d * t_max;
            if x.abs() < 1e-12 { 1.0 } else { x.sin() / x }
        };
        let kernel = RMatrix::from_fn(m, m, |a, b| {
            self.coeffs[a] * self.coeffs[b] * sinc(self.values[a] - self.values[b])
        });
        let xm = &*self.vectors * kernel;
        xm.row_iter()
            .zip(self.vectors.row_iter())
            .map(|(a, b)| a.dot(&b))
            .collect()
    }
}

/// Largest cyclic subspace used for walks beyond [`DENSE_LIMIT`] nodes: at
/// most `DENSE_LIMIT` vectors, `2²⁵` stored amplitudes and `n·m² ≤ 2³²`
/// operations (a few seconds).
fn cyclic_limit(n: usize) -> usize {
    let n = n.max(1);
    let by_cost = ((1u64 << 32) as f64 / n as f64).sqrt() as usize;
    DENSE_LIMIT.min((1usize << 25) / n).min(by_cost).max(1)
}

/// Total-variation distance `½ Σ |p − q|`.
pub fn total_variation(p: &[f64], q: &[f64]) -> f64 {
    0.5 * p.iter().zip(q).map(|(a, b)| (a - b).abs()).sum::<f64>()
}

/// Continuous-time quantum walk on a graph, optionally with site energies
/// ([`with_site_energies`](Self::with_site_energies)).
#[derive(Clone, Debug)]
pub struct ContinuousWalk {
    n: usize,
    gamma: f64,
    generator: Generator,
    hamiltonian: CsrMatrix,
    /// Eigenvalues (ascending) and eigenvectors (columns), computed on first
    /// use (`None` beyond DENSE_LIMIT).
    spectral: std::sync::OnceLock<Option<(Vec<f64>, RMatrix)>>,
    /// The graph, kept beyond DENSE_LIMIT for symmetry reduction.
    graph: Option<Graph>,
    /// Site energies `ε_j`, if set.
    energies: Option<Vec<f64>>,
    /// The graph part of `H` (`γL` or `−γA`), kept by the first
    /// `with_site_energies` so that later energies replace earlier ones.
    base: Option<CsrMatrix>,
}

impl ContinuousWalk {
    /// Walk on `graph` with hopping rate `gamma > 0`.
    pub fn new(graph: &Graph, generator: Generator, gamma: f64) -> Result<Self> {
        if !(gamma.is_finite() && gamma > 0.0) {
            return Err(QsoError::invalid(
                "gamma",
                "hopping rate must be positive and finite",
            ));
        }
        if graph.num_nodes() == 0 {
            return Err(QsoError::invalid("graph", "graph has no nodes"));
        }
        let hamiltonian = match generator {
            Generator::Laplacian => graph.laplacian_csr().scaled(gamma),
            Generator::Adjacency => graph.adjacency_csr().scaled(-gamma),
        };
        let n = graph.num_nodes();
        Ok(Self {
            n,
            gamma,
            generator,
            hamiltonian,
            spectral: std::sync::OnceLock::new(),
            graph: (n > DENSE_LIMIT).then(|| graph.clone()),
            energies: None,
            base: None,
        })
    }

    /// The same walk with site energies `ε_j`:
    /// `H = γL + Σ_j ε_j|j⟩⟨j|` (or `−γA + Σ_j ε_j|j⟩⟨j|`). The energies
    /// are not scaled by `γ`; one finite value per node. They replace
    /// earlier energies, and exact zeros leave `H` bit-identical to the walk
    /// without energies.
    ///
    /// Eigenvalues closer than 10⁻¹¹ of the largest eigenvalue magnitude
    /// count as degenerate, so give energies relative to a reference energy
    /// rather than with a large common offset. Beyond [`DENSE_LIMIT`] nodes
    /// the limiting and time-averaged distributions use the quotient in
    /// which nodes of different energies lie in different cells (see
    /// [`limiting_distribution`](Self::limiting_distribution)); with many
    /// distinct energies that has too many cells, and Lanczos takes over.
    ///
    /// ```
    /// use metatron_qso::graph::generators;
    /// use metatron_qso::walks::{ContinuousWalk, Generator};
    ///
    /// // A detuned dimer: H = [[3, −1], [−1, 0]]. From node 0 the walker
    /// // reaches node 1 with probability 4/13 at most (Rabi's formula
    /// // 4J²/(4J² + Δ²) with J = 1, Δ = 3).
    /// let walk = ContinuousWalk::new(&generators::path(2), Generator::Adjacency, 1.0)
    ///     .unwrap()
    ///     .with_site_energies(&[3.0, 0.0])
    ///     .unwrap();
    /// let t = std::f64::consts::PI / 13f64.sqrt();
    /// assert!((walk.probabilities(0, t).unwrap()[1] - 4.0 / 13.0).abs() < 1e-12);
    /// ```
    pub fn with_site_energies(mut self, energies: &[f64]) -> Result<Self> {
        if energies.len() != self.n {
            return Err(QsoError::DimensionMismatch {
                expected: self.n,
                actual: energies.len(),
            });
        }
        if energies.iter().any(|e| !e.is_finite()) {
            return Err(QsoError::invalid(
                "energies",
                "site energies must be finite",
            ));
        }
        let base = match self.base.take() {
            Some(base) => base,
            None => std::mem::replace(&mut self.hamiltonian, CsrMatrix::from_triplets(0, &[])?),
        };
        let mut triplets = Vec::with_capacity(base.nnz() + self.n);
        for r in 0..self.n {
            triplets.extend(base.row(r).map(|(c, v)| (r, c, v)));
        }
        // After the graph's entries: duplicates are summed in input order,
        // so a diagonal entry becomes H_jj + ε_j.
        triplets.extend(
            energies
                .iter()
                .enumerate()
                .filter(|&(_, &e)| e != 0.0)
                .map(|(j, &e)| (j, j, e)),
        );
        self.hamiltonian = CsrMatrix::from_triplets(self.n, &triplets)?;
        self.base = Some(base);
        self.energies = Some(energies.to_vec());
        self.spectral = std::sync::OnceLock::new();
        Ok(self)
    }

    /// The site energies `ε_j`, if set by
    /// [`with_site_energies`](Self::with_site_energies).
    pub fn site_energies(&self) -> Option<&[f64]> {
        self.energies.as_deref()
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.n
    }

    /// Hopping rate `γ`.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Which matrix generates the walk.
    pub fn generator(&self) -> Generator {
        self.generator
    }

    /// The (sparse) Hamiltonian, site energies included.
    pub fn hamiltonian(&self) -> &CsrMatrix {
        &self.hamiltonian
    }

    /// Eigenvalues of `H` (ascending) up to [`DENSE_LIMIT`] nodes; the
    /// eigendecomposition is computed on first use.
    pub fn spectrum(&self) -> Option<&[f64]> {
        self.spectral().map(|(v, _)| v.as_slice())
    }

    /// Eigenvalues and eigenvectors of `H`, computed on first use (`None`
    /// beyond [`DENSE_LIMIT`] nodes).
    fn spectral(&self) -> Option<&(Vec<f64>, RMatrix)> {
        self.spectral
            .get_or_init(|| {
                (self.n <= DENSE_LIMIT).then(|| eigh_real(&self.hamiltonian.to_dense()))
            })
            .as_ref()
    }

    fn require_dense(&self) -> Result<&(Vec<f64>, RMatrix)> {
        self.spectral().ok_or_else(|| {
            QsoError::invalid(
                "graph",
                format!(
                    "{} nodes; dense walk analysis needs at most {DENSE_LIMIT}",
                    self.n
                ),
            )
        })
    }

    fn check_node(&self, node: usize) -> Result<()> {
        if node >= self.n {
            return Err(QsoError::NodeOutOfRange {
                node,
                n_nodes: self.n,
            });
        }
        Ok(())
    }

    /// `exp(−iHt)|ψ₀⟩` for an arbitrary initial vector.
    pub fn evolve(&self, psi0: &[C64], t: f64) -> Result<Vec<C64>> {
        if psi0.len() != self.n {
            return Err(QsoError::DimensionMismatch {
                expected: self.n,
                actual: psi0.len(),
            });
        }
        match self.spectral() {
            Some((values, vectors)) => {
                // c = Vᵀψ₀, ψ(t) = V·diag(e^{−iλt})·c
                let n = self.n;
                let mut c = vec![C64::new(0.0, 0.0); n];
                for (k, ck) in c.iter_mut().enumerate() {
                    let mut acc = C64::new(0.0, 0.0);
                    for j in 0..n {
                        acc += psi0[j] * vectors[(j, k)];
                    }
                    *ck = acc * C64::from_polar(1.0, -values[k] * t);
                }
                Ok((0..n)
                    .map(|j| (0..n).map(|k| c[k] * vectors[(j, k)]).sum())
                    .collect())
            }
            None => {
                let h = &self.hamiltonian;
                expm_multiply(|x, y| h.matvec(x, y), psi0, t, &KrylovOptions::default())
            }
        }
    }

    /// Node probabilities at time `t` for a walker started at `start`.
    pub fn probabilities(&self, start: usize, t: f64) -> Result<Vec<f64>> {
        self.check_node(start)?;
        let mut psi0 = vec![C64::new(0.0, 0.0); self.n];
        psi0[start] = C64::new(1.0, 0.0);
        Ok(self
            .evolve(&psi0, t)?
            .iter()
            .map(|a| a.norm_sqr())
            .collect())
    }

    /// Dense propagator `U(t) = exp(−iHt)`.
    pub fn propagator(&self, t: f64) -> Result<CMatrix> {
        let (values, vectors) = self.require_dense()?;
        let n = self.n;
        let mut u = CMatrix::zeros(n, n);
        for k in 0..n {
            let phase = C64::from_polar(1.0, -values[k] * t);
            for r in 0..n {
                let vr = vectors[(r, k)] * phase;
                for c in 0..n {
                    u[(r, c)] += vr * vectors[(c, k)];
                }
            }
        }
        Ok(u)
    }

    /// Spectral data of the walker started at `start`: the full
    /// eigendecomposition up to [`DENSE_LIMIT`] nodes, beyond that the
    /// quotient of [`quotient_spectrum`](Self::quotient_spectrum) or the
    /// start node's cyclic subspace (see [`cyclic_spectrum`]).
    pub(crate) fn start_spectrum(&self, start: usize) -> Result<StartSpectrum<'_>> {
        self.check_node(start)?;
        match self.spectral() {
            Some((values, vectors)) => Ok(StartSpectrum {
                values: std::borrow::Cow::Borrowed(values),
                coeffs: (0..self.n).map(|k| vectors[(start, k)]).collect(),
                vectors: std::borrow::Cow::Borrowed(vectors),
            }),
            None => {
                let limit = cyclic_limit(self.n);
                // First an exact invariant subspace from an equitable
                // partition, solved densely.
                if let Some(graph) = &self.graph
                    && let Some(spectrum) = self.quotient_spectrum(graph, start, limit)
                {
                    return Ok(spectrum);
                }
                // Otherwise the start vector's cyclic subspace by Lanczos.
                let mut v = vec![0.0; self.n];
                v[start] = 1.0;
                let h = &self.hamiltonian;
                let c = cyclic_spectrum(|x, y| h.matvec_real(x, y), &v, limit)?;
                Ok(StartSpectrum {
                    values: std::borrow::Cow::Owned(c.values),
                    coeffs: c.weights,
                    vectors: std::borrow::Cow::Owned(c.vectors),
                })
            }
        }
    }

    /// Spectral data of the walker started at `start` on the quotient by
    /// the coarsest equitable partition of `graph` in which the start node
    /// is a cell of its own and, with site energies, nodes of different
    /// energies lie in different cells: every cell then has one energy, so
    /// the cell states span an exact invariant subspace of `H`. Solved
    /// densely and lifted to the nodes; `None` if the partition has more
    /// than `limit` cells.
    fn quotient_spectrum(
        &self,
        graph: &Graph,
        start: usize,
        limit: usize,
    ) -> Option<StartSpectrum<'static>> {
        let q = match &self.energies {
            None => Quotient::with_singletons(graph, &[start]),
            Some(energies) => Quotient::equitable(graph, &energy_colours(energies, start)),
        }
        .ok()
        .filter(|q| q.num_cells() <= limit)?;
        let cells = q.cell_of().expect("explicit graph");
        let mut h = q.hamiltonian(self.generator, self.gamma);
        if let Some(energies) = &self.energies {
            // The energy of each cell is that of its smallest node.
            let mut first = vec![0; q.num_cells()];
            for (v, &c) in cells.iter().enumerate().rev() {
                first[c] = v;
            }
            for (c, &v) in first.iter().enumerate() {
                if energies[v] != 0.0 {
                    h[(c, c)] += energies[v];
                }
            }
        }
        let (values, y) = eigh_real(&h);
        let sizes = q.sizes();
        let k = values.len();
        let vectors = RMatrix::from_fn(self.n, k, |j, a| y[(cells[j], a)] / sizes[cells[j]].sqrt());
        Some(StartSpectrum {
            values: std::borrow::Cow::Owned(values),
            coeffs: (0..k).map(|a| y[(cells[start], a)]).collect(),
            vectors: std::borrow::Cow::Owned(vectors),
        })
    }

    /// Dimension of the smallest invariant subspace containing `|start⟩`:
    /// the number of distinct eigenvalues the walker ever sees (those with
    /// a projection of the start node above 10⁻¹²).
    pub fn cyclic_dimension(&self, start: usize) -> Result<usize> {
        let spectrum = self.start_spectrum(start)?;
        Ok(eigenspaces(&spectrum.values)
            .into_iter()
            .filter(|space| space.clone().any(|k| spectrum.coeffs[k].abs() > 1e-12))
            .count())
    }

    /// Limiting (infinite-time-averaged) distribution
    /// `P̄(j) = Σ_λ |⟨j|Π_λ|start⟩|²`, where `Π_λ` projects onto the full
    /// eigenspace of `λ` (degeneracies are grouped, which v0.1 omitted).
    ///
    /// Beyond [`DENSE_LIMIT`] nodes the walk is reduced to the quotient by
    /// the coarsest equitable partition in which the start node is a cell
    /// of its own (exact; 21 cells on the 2²⁰-node hypercube, 561 on the
    /// 64×64 torus) and, with site energies, nodes of different energies
    /// lie in different cells, solved densely. If that has more than
    /// `min(DENSE_LIMIT, 2²⁵/n, √(2³²/n))` cells, the start node's cyclic
    /// subspace is built by Lanczos instead, which can fail when the walker
    /// sees many eigenvalues.
    pub fn limiting_distribution(&self, start: usize) -> Result<Vec<f64>> {
        Ok(self.start_spectrum(start)?.limiting_distribution())
    }

    /// Exact time average `(1/T)∫₀ᵀ P(j, t) dt` of the node probabilities
    /// (beyond [`DENSE_LIMIT`] nodes as for
    /// [`limiting_distribution`](Self::limiting_distribution)).
    pub fn average_distribution(&self, start: usize, t_max: f64) -> Result<Vec<f64>> {
        self.check_node(start)?;
        if t_max.is_nan() || t_max <= 0.0 {
            return self.probabilities(start, 0.0);
        }
        Ok(self.start_spectrum(start)?.average_distribution(t_max))
    }

    /// Mixing time of Aharonov, Ambainis, Kempe & Vazirani (2001): the
    /// smallest `T` on the grid `t_max·k/steps` such that the time-averaged
    /// distribution stays within total-variation distance `epsilon` of the
    /// limiting distribution at `T` and at every later grid point up to
    /// `t_max`. `None` if the bound fails at `t_max`.
    ///
    /// The time average oscillates while it converges, so a single grid
    /// point below `epsilon` is not enough; choose `t_max` well beyond the
    /// expected mixing time.
    pub fn mixing_time(
        &self,
        start: usize,
        epsilon: f64,
        t_max: f64,
        steps: usize,
    ) -> Result<Option<f64>> {
        if !(epsilon.is_finite() && epsilon >= 0.0) {
            return Err(QsoError::invalid(
                "epsilon",
                format!("must be finite and non-negative, got {epsilon}"),
            ));
        }
        if !(t_max.is_finite() && t_max > 0.0) {
            return Err(QsoError::invalid(
                "t_max",
                format!("must be finite and positive, got {t_max}"),
            ));
        }
        let spectrum = self.start_spectrum(start)?;
        let limit = spectrum.limiting_distribution();
        let steps = steps.max(1);
        let mut mixed = None;
        for k in (1..=steps).rev() {
            let t = t_max * k as f64 / steps as f64;
            if total_variation(&spectrum.average_distribution(t), &limit) > epsilon {
                break;
            }
            mixed = Some(t);
        }
        Ok(mixed)
    }
}

/// Colours that keep the start node and nodes of different energies apart:
/// 0 for the start, `1 + rank` of the energy among the distinct values for
/// every other node (exact equality, −0.0 equal to 0.0).
fn energy_colours(energies: &[f64], start: usize) -> Vec<usize> {
    // Adding 0.0 turns −0.0 into 0.0 and leaves every other value alone.
    let mut distinct: Vec<f64> = energies.iter().map(|&e| e + 0.0).collect();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    energies
        .iter()
        .enumerate()
        .map(|(u, &e)| {
            if u == start {
                0
            } else {
                1 + distinct
                    .binary_search_by(|x| x.total_cmp(&(e + 0.0)))
                    .expect("present")
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{generators, metatron};

    #[test]
    fn evolution_is_unitary_and_matches_krylov() {
        let g = generators::petersen();
        let walk = ContinuousWalk::new(&g, Generator::Adjacency, 0.7).unwrap();
        let p = walk.probabilities(3, 2.3).unwrap();
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let mut psi0 = vec![C64::new(0.0, 0.0); 10];
        psi0[3] = C64::new(1.0, 0.0);
        let h = walk.hamiltonian();
        let krylov =
            expm_multiply(|x, y| h.matvec(x, y), &psi0, 2.3, &KrylovOptions::default()).unwrap();
        let dense = walk.evolve(&psi0, 2.3).unwrap();
        for (a, b) in krylov.iter().zip(&dense) {
            assert!((a - b).norm() < 1e-9);
        }
        let u = walk.propagator(1.1).unwrap();
        assert!((u.adjoint() * &u - CMatrix::identity(10, 10)).norm() < 1e-10);
    }

    #[test]
    fn metatron_cube_dynamics_match_closed_form() {
        // On K_N with H = L: amplitude at the start node is
        // (1 + (N−1)e^{−iNt})/N, so P_start(t) = 1 − 2(N−1)(1 − cos Nt)/N².
        let walk =
            ContinuousWalk::new(&metatron::metatron_cube(), Generator::Laplacian, 1.0).unwrap();
        let n = 13.0;
        for t in [0.0, 0.1, 0.37, 1.0, 2.5] {
            let p = walk.probabilities(0, t).unwrap();
            let expected = 1.0 - 2.0 * (n - 1.0) * (1.0 - (n * t).cos()) / (n * n);
            assert!((p[0] - expected).abs() < 1e-12, "t={t}");
            // All other nodes are equivalent by symmetry.
            assert!(p[1..].iter().all(|&x| (x - p[1]).abs() < 1e-12));
        }
        let limit = walk.limiting_distribution(0).unwrap();
        assert!((limit[0] - 145.0 / 169.0).abs() < 1e-12);
        assert!(limit[1..].iter().all(|&x| (x - 2.0 / 169.0).abs() < 1e-12));
    }

    #[test]
    fn average_distribution_converges_to_limit() {
        let walk =
            ContinuousWalk::new(&metatron::fruit_of_life(), Generator::Laplacian, 1.0).unwrap();
        let limit = walk.limiting_distribution(0).unwrap();
        assert!((limit.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let short = total_variation(&walk.average_distribution(0, 5.0).unwrap(), &limit);
        let long = total_variation(&walk.average_distribution(0, 500.0).unwrap(), &limit);
        assert!(long < short && long < 0.01);
        // Numerical average agrees with the closed form.
        let t_max = 3.0;
        let steps = 3000;
        let mut num = vec![0.0; 13];
        for k in 0..steps {
            let t = (k as f64 + 0.5) * t_max / steps as f64;
            for (a, b) in num.iter_mut().zip(walk.probabilities(0, t).unwrap()) {
                *a += b / steps as f64;
            }
        }
        let exact = walk.average_distribution(0, t_max).unwrap();
        assert!(total_variation(&num, &exact) < 1e-5);
        let mix = walk.mixing_time(0, 0.05, 200.0, 400).unwrap();
        assert!(mix.is_some());
    }

    #[test]
    fn mixing_time_requires_the_bound_to_hold_from_then_on() {
        // On K13, started at node 0, the time-averaged distance to the limit
        // is (24/169)·|sin(13T)/(13T)|: for ε = 0.02 it first dips below ε
        // at 13T ≈ 2.7, rises above it again in the next lobe (0.217·24/169
        // ≈ 0.031 at 13T ≈ 4.49) and only then stays below.
        let walk =
            ContinuousWalk::new(&metatron::metatron_cube(), Generator::Laplacian, 1.0).unwrap();
        let (eps, t_max, steps) = (0.02, 20.0, 2000);
        let limit = walk.limiting_distribution(0).unwrap();
        let tv = |t: f64| total_variation(&walk.average_distribution(0, t).unwrap(), &limit);
        let grid = |k: usize| t_max * k as f64 / steps as f64;
        let first_dip = (1..=steps).map(grid).find(|&t| tv(t) <= eps).unwrap();
        let mix = walk.mixing_time(0, eps, t_max, steps).unwrap().unwrap();
        assert!(mix > first_dip);
        let k_mix = (mix / t_max * steps as f64).round() as usize;
        assert!((k_mix..=steps).all(|k| tv(grid(k)) <= eps));
        assert!(tv(grid(k_mix - 1)) > eps);
        assert_eq!(walk.mixing_time(0, 1e-9, 1.0, 10).unwrap(), None);
        // Comparisons with NaN are false, so a NaN bound would always "hold".
        for (eps, t) in [(f64::NAN, 1.0), (-0.1, 1.0), (f64::INFINITY, 1.0)] {
            assert!(walk.mixing_time(0, eps, t, 10).is_err());
        }
        for t in [f64::NAN, 0.0, -1.0, f64::INFINITY] {
            assert!(walk.mixing_time(0, 0.1, t, 10).is_err());
        }
    }

    #[test]
    fn large_walks_use_the_cyclic_subspace() {
        // Q12 has 4096 > DENSE_LIMIT nodes; a walker at a node only ever
        // sees the 13 distinct Laplacian eigenvalues 0, 2, …, 24.
        let d = 12;
        let g = generators::hypercube(d);
        let walk = ContinuousWalk::new(&g, Generator::Laplacian, 1.0).unwrap();
        assert!(walk.spectrum().is_none());
        assert_eq!(walk.cyclic_dimension(5).unwrap(), d + 1);
        let q = Quotient::hypercube(d).unwrap();
        let reduced = QuotientWalk::new(&q, Generator::Laplacian, 1.0).unwrap();
        let cell_limit = reduced.limiting_distribution(0).unwrap();
        let cell_avg = reduced.average_distribution(0, 3.0).unwrap();
        let limit = walk.limiting_distribution(0).unwrap();
        let avg = walk.average_distribution(0, 3.0).unwrap();
        for v in [0usize, 1, 6, 0b1010_1010_1010, 4095] {
            let k = v.count_ones() as usize;
            let size = q.sizes()[k];
            assert!((limit[v] - cell_limit[k] / size).abs() < 1e-12, "v={v}");
            assert!((avg[v] - cell_avg[k] / size).abs() < 1e-12, "v={v}");
        }
        assert!((limit.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert!(walk.mixing_time(0, 0.05, 50.0, 50).unwrap().is_some());
        // The 64×64 torus: 561 cells from a node, many distinct eigenvalues.
        let torus = generators::grid_2d(64, 64, true);
        let walk = ContinuousWalk::new(&torus, Generator::Laplacian, 1.0).unwrap();
        let limit = walk.limiting_distribution(0).unwrap();
        assert!((limit.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        // Symmetric under the dihedral group of the square fixing node 0.
        assert!((limit[1] - limit[64]).abs() < 1e-14 && (limit[1] - limit[63]).abs() < 1e-14);
        assert!((walk.average_distribution(0, f64::INFINITY).unwrap()[0] - limit[0]).abs() < 1e-15);
        // Cyclic and dense decompositions agree where both exist.
        let small =
            ContinuousWalk::new(&generators::petersen(), Generator::Adjacency, 1.0).unwrap();
        let h = small.hamiltonian();
        let mut v = vec![0.0; 10];
        v[2] = 1.0;
        let c = crate::linalg::cyclic_spectrum(|x, y| h.matvec_real(x, y), &v, 10).unwrap();
        assert_eq!(c.values.len(), 3);
        let cyclic = StartSpectrum {
            values: std::borrow::Cow::Owned(c.values),
            coeffs: c.weights,
            vectors: std::borrow::Cow::Owned(c.vectors),
        };
        let dense = small.limiting_distribution(2).unwrap();
        for (a, b) in cyclic.limiting_distribution().iter().zip(&dense) {
            assert!((a - b).abs() < 1e-12);
        }
        // A start vector without symmetry exceeds a small subspace limit.
        let r = generators::erdos_renyi(40, 0.3, &mut crate::rng::Rng::seed_from_u64(2)).unwrap();
        let h = r.laplacian_csr();
        let mut v = vec![0.0; 40];
        v[0] = 1.0;
        assert!(crate::linalg::cyclic_spectrum(|x, y| h.matvec_real(x, y), &v, 10).is_err());
    }

    #[test]
    fn bipartite_spectral_symmetry_and_sign_convention() {
        // Probabilities from the adjacency generator are invariant under
        // H → −H; check against Laplacian on a regular graph (L = dI − A,
        // so they differ by a global phase only).
        let g = generators::cycle(8);
        let a = ContinuousWalk::new(&g, Generator::Adjacency, 1.0).unwrap();
        let l = ContinuousWalk::new(&g, Generator::Laplacian, 1.0).unwrap();
        let pa = a.probabilities(0, 1.7).unwrap();
        let pl = l.probabilities(0, 1.7).unwrap();
        for (x, y) in pa.iter().zip(&pl) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    fn node_state(n: usize, j: usize) -> Vec<C64> {
        let mut psi = vec![C64::new(0.0, 0.0); n];
        psi[j] = C64::new(1.0, 0.0);
        psi
    }

    /// Site energies without any symmetry.
    fn rough_energies(n: usize) -> Vec<f64> {
        (0..n)
            .map(|j| 0.6 * (0.37 * j as f64).sin() + 0.25 * (1.9 * j as f64).cos() - 0.1)
            .collect()
    }

    #[test]
    fn site_energies_match_the_driven_walk() {
        // γL + E (or −γA + E) is the driven walk with the terms H_graph
        // and E and the constant coefficients γ and 1: by the dense
        // eigendecomposition on Q₄, by Krylov steps on C₃₀₀₀.
        for (g, start) in [(generators::hypercube(4), 0), (generators::cycle(3000), 7)] {
            let n = g.num_nodes();
            let energies = rough_energies(n);
            let psi0 = node_state(n, start);
            for generator in [Generator::Laplacian, Generator::Adjacency] {
                let walk = ContinuousWalk::new(&g, generator, 0.7)
                    .unwrap()
                    .with_site_energies(&energies)
                    .unwrap();
                assert_eq!(walk.spectrum().is_some(), n <= DENSE_LIMIT);
                let terms = vec![
                    driven::Term::graph(&g, generator),
                    driven::Term::potential(&energies).unwrap(),
                ];
                let driven = DrivenWalk::new(n, terms)
                    .unwrap()
                    .evolve(
                        &psi0,
                        0.0,
                        2.5,
                        |_| Ok(vec![0.7, 1.0]),
                        &MagnusOptions::default(),
                    )
                    .unwrap();
                let exact = walk.evolve(&psi0, 2.5).unwrap();
                let distance = driven
                    .state
                    .iter()
                    .zip(&exact)
                    .map(|(a, b)| (a - b).norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                assert!(distance < 1e-9, "{n} nodes, {generator:?}: {distance:e}");
            }
        }
    }

    #[test]
    fn site_energies_on_the_coloured_quotient() {
        // Q₁₂ (4096 > DENSE_LIMIT nodes) with energies by Hamming weight:
        // −1 for the weights 0, 3, 6, …, 0 for 1, 4, 7, … and 1 for the
        // rest. Colour refinement splits the three classes into the 13
        // weights again, and the walker from 0…0 lives on the 13-site
        // chain with the energies on its diagonal.
        let d = 12;
        let g = generators::hypercube(d);
        let by_weight: Vec<f64> = (0..=d).map(|k| (k % 3) as f64 - 1.0).collect();
        let energies: Vec<f64> = (0..g.num_nodes())
            .map(|v| by_weight[v.count_ones() as usize])
            .collect();
        let walk = ContinuousWalk::new(&g, Generator::Adjacency, 1.0)
            .unwrap()
            .with_site_energies(&energies)
            .unwrap();
        assert!(walk.spectrum().is_none());
        let q = Quotient::hypercube(d).unwrap();
        let mut chain = q.hamiltonian(Generator::Adjacency, 1.0);
        for (k, e) in by_weight.iter().enumerate() {
            chain[(k, k)] += e;
        }
        // An irreducible tridiagonal matrix has simple eigenvalues, so
        // P̄(k) = Σ_a (y_ka·y_0a)² over its eigenvectors y_a.
        let (_, y) = eigh_real(&chain);
        let cells: Vec<f64> = (0..=d)
            .map(|k| (0..=d).map(|a| (y[(k, a)] * y[(0, a)]).powi(2)).sum())
            .collect();
        let limit = walk.limiting_distribution(0).unwrap();
        for (v, p) in limit.iter().enumerate() {
            let k = v.count_ones() as usize;
            assert!((p - cells[k] / q.sizes()[k]).abs() < 1e-12, "v = {v}");
        }
        assert_eq!(walk.cyclic_dimension(0).unwrap(), d + 1);
        let plain = ContinuousWalk::new(&g, Generator::Adjacency, 1.0)
            .unwrap()
            .limiting_distribution(0)
            .unwrap();
        assert!(total_variation(&limit, &plain) > 0.01);
    }

    #[test]
    fn site_energies_reach_lanczos_when_the_quotient_is_too_large() {
        // Q₁₂ with energies by Hamming weight next to a path of 1000 nodes
        // with distinct energies: the coloured quotient has more than the
        // 918 cells allowed at 5096 nodes, so the walker from 0…0 is
        // followed by Lanczos, which must find the 13-site chain again.
        let d = 12;
        let cube = generators::hypercube(d);
        let m = cube.num_nodes();
        let mut edges: Vec<(usize, usize)> = cube.edges().iter().map(|&(u, v, _)| (u, v)).collect();
        edges.extend((1..1000).map(|j| (m + j - 1, m + j)));
        let g = Graph::from_edges(m + 1000, &edges).unwrap();
        let n = g.num_nodes();
        let by_weight: Vec<f64> = (0..=d).map(|k| 0.4 * (k % 4) as f64 - 0.5).collect();
        let mut energies: Vec<f64> = (0..m).map(|v| by_weight[v.count_ones() as usize]).collect();
        energies.extend(rough_energies(1000).iter().map(|e| e + 3.0));
        let walk = ContinuousWalk::new(&g, Generator::Laplacian, 1.0)
            .unwrap()
            .with_site_energies(&energies)
            .unwrap();
        assert!(walk.quotient_spectrum(&g, 0, cyclic_limit(n)).is_none());
        let q = Quotient::hypercube(d).unwrap();
        let mut chain = q.hamiltonian(Generator::Laplacian, 1.0);
        for (k, e) in by_weight.iter().enumerate() {
            chain[(k, k)] += e;
        }
        let (_, y) = eigh_real(&chain);
        let limit = walk.limiting_distribution(0).unwrap();
        let mut worst = 0.0f64;
        for (v, p) in limit.iter().enumerate() {
            let want = if v < m {
                let k = v.count_ones() as usize;
                (0..=d)
                    .map(|a| (y[(k, a)] * y[(0, a)]).powi(2))
                    .sum::<f64>()
                    / q.sizes()[k]
            } else {
                0.0
            };
            worst = worst.max((p - want).abs());
        }
        assert!(worst < 1e-12, "{worst:e}");
        assert_eq!(walk.cyclic_dimension(0).unwrap(), d + 1);
    }

    #[test]
    fn the_coloured_quotient_matches_the_dense_walk() {
        // Q₆ with energies by Hamming weight and a bump on node 5 (15 cells
        // from node 0): the coloured quotient, called directly because a
        // walk this small is dense, gives the dense walk's distributions.
        let g = generators::hypercube(6);
        let n = g.num_nodes();
        let mut energies: Vec<f64> = (0..n)
            .map(|v| {
                let w = v.count_ones() as f64;
                0.3 * w * w - 0.7 * w
            })
            .collect();
        energies[5] += 1.1;
        for generator in [Generator::Laplacian, Generator::Adjacency] {
            let walk = ContinuousWalk::new(&g, generator, 0.8)
                .unwrap()
                .with_site_energies(&energies)
                .unwrap();
            for start in [0, 5, 6, 63] {
                let reduced = walk.quotient_spectrum(&g, start, DENSE_LIMIT).unwrap();
                let dense = walk.start_spectrum(start).unwrap();
                if start == 0 {
                    assert_eq!(reduced.values.len(), 15);
                }
                assert!(reduced.values.len() < n);
                let pairs = [
                    (
                        reduced.limiting_distribution(),
                        dense.limiting_distribution(),
                    ),
                    (
                        reduced.average_distribution(2.3),
                        dense.average_distribution(2.3),
                    ),
                ];
                for (a, b) in pairs {
                    for (x, y) in a.iter().zip(&b) {
                        assert!((x - y).abs() < 1e-12, "start {start}: {x} against {y}");
                    }
                }
            }
            assert!(walk.quotient_spectrum(&g, 0, 14).is_none());
        }
        // Every energy distinct: no reduction.
        let walk = ContinuousWalk::new(&g, Generator::Laplacian, 1.0)
            .unwrap()
            .with_site_energies(&rough_energies(n))
            .unwrap();
        assert_eq!(walk.quotient_spectrum(&g, 0, n).unwrap().values.len(), n);
        assert!(walk.quotient_spectrum(&g, 0, n - 1).is_none());
    }

    #[test]
    fn site_energies_replace_earlier_ones_and_zeros_change_nothing() {
        let g = generators::petersen();
        let first = rough_energies(10);
        let second: Vec<f64> = (0..10).map(|j| 0.1 * j as f64 - 0.4).collect();
        let mut zeros = vec![0.0; 10];
        zeros[4] = -0.0;
        for generator in [Generator::Laplacian, Generator::Adjacency] {
            let plain = ContinuousWalk::new(&g, generator, 0.9).unwrap();
            assert_eq!(plain.site_energies(), None);
            let fresh = plain.clone().with_site_energies(&second).unwrap();
            let replaced = plain
                .clone()
                .with_site_energies(&first)
                .unwrap()
                .with_site_energies(&second)
                .unwrap();
            assert_eq!(replaced.site_energies(), Some(second.as_slice()));
            assert_eq!(replaced.hamiltonian(), fresh.hamiltonian());
            assert_eq!(replaced.spectrum(), fresh.spectrum());
            assert_eq!(
                replaced.probabilities(3, 1.3).unwrap(),
                fresh.probabilities(3, 1.3).unwrap()
            );
            assert_eq!(
                replaced.limiting_distribution(3).unwrap(),
                fresh.limiting_distribution(3).unwrap()
            );
            assert_ne!(fresh.hamiltonian(), plain.hamiltonian());
            // Exact zeros of either sign: H and everything computed from it
            // as without energies.
            let zero = plain.clone().with_site_energies(&zeros).unwrap();
            assert_eq!(zero.site_energies(), Some(zeros.as_slice()));
            assert_eq!(zero.hamiltonian(), plain.hamiltonian());
            assert_eq!(zero.spectrum(), plain.spectrum());
            assert_eq!(
                zero.probabilities(3, 1.3).unwrap(),
                plain.probabilities(3, 1.3).unwrap()
            );
            let cleared = fresh.with_site_energies(&zeros).unwrap();
            assert_eq!(cleared.hamiltonian(), plain.hamiltonian());
            assert_eq!(
                cleared.propagator(0.4).unwrap(),
                plain.propagator(0.4).unwrap()
            );
        }
        // A spectrum computed before the energies is not reused.
        let walk = ContinuousWalk::new(&g, Generator::Adjacency, 1.0).unwrap();
        let before = walk.spectrum().unwrap().to_vec();
        let walk = walk.with_site_energies(&first).unwrap();
        let fresh = ContinuousWalk::new(&g, Generator::Adjacency, 1.0)
            .unwrap()
            .with_site_energies(&first)
            .unwrap();
        assert_eq!(walk.spectrum(), fresh.spectrum());
        assert_ne!(walk.spectrum().unwrap(), before.as_slice());
        // Beyond the dense limit zero energies leave the quotient alone.
        let g = generators::hypercube(12);
        let mut zeros = vec![0.0; g.num_nodes()];
        zeros[77] = -0.0;
        let plain = ContinuousWalk::new(&g, Generator::Laplacian, 1.0).unwrap();
        let zero = plain.clone().with_site_energies(&zeros).unwrap();
        assert_eq!(zero.hamiltonian(), plain.hamiltonian());
        assert_eq!(
            zero.limiting_distribution(5).unwrap(),
            plain.limiting_distribution(5).unwrap()
        );
        assert_eq!(
            zero.average_distribution(5, 3.0).unwrap(),
            plain.average_distribution(5, 3.0).unwrap()
        );
    }

    #[test]
    fn site_energies_are_checked() {
        let walk = ContinuousWalk::new(&generators::cycle(5), Generator::Laplacian, 1.0).unwrap();
        for wrong in [&[0.0; 4][..], &[0.0; 6], &[]] {
            assert_eq!(
                walk.clone().with_site_energies(wrong).unwrap_err(),
                QsoError::DimensionMismatch {
                    expected: 5,
                    actual: wrong.len()
                }
            );
        }
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut energies = [0.5; 5];
            energies[2] = bad;
            assert_eq!(
                walk.clone().with_site_energies(&energies).unwrap_err(),
                QsoError::invalid("energies", "site energies must be finite")
            );
        }
    }

    #[test]
    fn site_energies_match_an_independent_eigendecomposition() {
        // The star S₆ (centre 0) with ε = (−0.3, 0.5, 0.5, 0.5, −1.1, −0.0):
        // the equal leaves 1, 2, 3 give a doubly degenerate eigenvalue
        // (γ + 0.5 with the Laplacian, 0.5 with the adjacency matrix).
        // References from LAPACK (numpy.linalg.eigh, degenerate eigenspaces
        // grouped) and scipy.linalg.expm, the time average by 400-point
        // Gauss–Legendre quadrature; the Laplacian limit agrees with a
        // 40-digit mpmath eigendecomposition to 4·10⁻¹⁷.
        let g = generators::star(6);
        let energies = [-0.3, 0.5, 0.5, 0.5, -1.1, -0.0];
        #[rustfmt::skip]
        let cases = [
            (Generator::Laplacian, 1, [-0.5893509203579675, 0.39378421972303546, 0.9076027081868844, 1.2, 1.2, 3.987963992448048],
             [0.05557310280981699, 0.4905330922347091, 0.1571997589013758, 0.1571997589013758, 0.01847948399985487, 0.12101480315286742],
             [0.0052835922954572, 0.7733631152578394, 0.07138184327263099, 0.07138184327263099, 0.02326892047898068, 0.05532068542246057],
             [0.06208929743721511, 0.8099790383208939, 0.03865634103854884, 0.03865634103854885, 0.01808002848414637, 0.03253895368064648]),
            (Generator::Adjacency, 0, [-1.8472550927577038, -0.7469532767120163, 0.12442222664427385, 0.5, 0.5, 1.5697861428254456],
             [0.36702244670486783, 0.09147933836189454, 0.09147933836189459, 0.09147933836189459, 0.2645252363346362, 0.09401430187481202],
             [0.568537862773277, 0.03886698154837016, 0.03886698154837021, 0.03886698154837018, 0.28890346716275145, 0.02595772541886146],
             [0.41989802217235506, 0.09641865988904619, 0.09641865988904619, 0.09641865988904619, 0.19579826780292744, 0.09504773035757916]),
        ];
        for (generator, start, spectrum, limit, p, average) in cases {
            let walk = ContinuousWalk::new(&g, generator, 0.7)
                .unwrap()
                .with_site_energies(&energies)
                .unwrap();
            let pairs = [
                (walk.spectrum().unwrap().to_vec(), spectrum),
                (walk.limiting_distribution(start).unwrap(), limit),
                (walk.probabilities(start, 1.9).unwrap(), p),
                (walk.average_distribution(start, 2.3).unwrap(), average),
            ];
            for (got, want) in pairs {
                for (x, y) in got.iter().zip(&want) {
                    assert!((x - y).abs() < 1e-12, "{generator:?}: {x} against {y}");
                }
            }
        }
        // The detuned dimer H = [[3, −1], [−1, 0]]: tan 2θ = 2/3, so the
        // walker from node 0 spends ½·sin²2θ = 2/13 of the time on node 1.
        let dimer = ContinuousWalk::new(&generators::path(2), Generator::Adjacency, 1.0)
            .unwrap()
            .with_site_energies(&[3.0, 0.0])
            .unwrap();
        let limit = dimer.limiting_distribution(0).unwrap();
        assert!((limit[1] - 2.0 / 13.0).abs() < 1e-15);
        assert!((limit[0] - 11.0 / 13.0).abs() < 1e-15);
    }

    #[test]
    fn uniform_and_negative_energies_only_shift_the_spectrum() {
        // ε_j = c for every node adds c·I: the spectrum moves by c and the
        // probabilities stay. Petersen (eigenvalues of multiplicity 4 and 5)
        // is dense; Q₁₂ goes through the coloured quotient, whose cells are
        // then those of the walk without energies.
        let c = -0.75;
        let g = generators::petersen();
        for generator in [Generator::Laplacian, Generator::Adjacency] {
            let plain = ContinuousWalk::new(&g, generator, 1.3).unwrap();
            let shifted = plain.clone().with_site_energies(&[c; 10]).unwrap();
            for (x, y) in shifted
                .spectrum()
                .unwrap()
                .iter()
                .zip(plain.spectrum().unwrap())
            {
                assert!((x - (y + c)).abs() < 1e-12);
            }
            for start in [0, 7] {
                let pairs = [
                    (
                        shifted.limiting_distribution(start).unwrap(),
                        plain.limiting_distribution(start).unwrap(),
                    ),
                    (
                        shifted.probabilities(start, 2.9).unwrap(),
                        plain.probabilities(start, 2.9).unwrap(),
                    ),
                    (
                        shifted.average_distribution(start, 1.7).unwrap(),
                        plain.average_distribution(start, 1.7).unwrap(),
                    ),
                ];
                for (a, b) in pairs {
                    for (x, y) in a.iter().zip(&b) {
                        assert!((x - y).abs() < 1e-12, "{generator:?}: {x} against {y}");
                    }
                }
                assert_eq!(
                    shifted.cyclic_dimension(start).unwrap(),
                    plain.cyclic_dimension(start).unwrap()
                );
            }
        }
        let g = generators::hypercube(12);
        let n = g.num_nodes();
        let plain = ContinuousWalk::new(&g, Generator::Laplacian, 1.0).unwrap();
        let shifted = plain.clone().with_site_energies(&vec![c; n]).unwrap();
        for start in [0, 1234] {
            let a = shifted.limiting_distribution(start).unwrap();
            let b = plain.limiting_distribution(start).unwrap();
            assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-12));
            assert_eq!(shifted.cyclic_dimension(start).unwrap(), 13);
        }
    }

    #[test]
    fn negative_zero_energies_are_zeros() {
        // −0.0 and 0.0 are the same energy: same Hamiltonian, same cells of
        // the coloured quotient, bit-identical results, dense and reduced.
        let by_weight = |d: usize, signed: bool| -> Vec<f64> {
            (0..1usize << d)
                .map(|v| match v.count_ones() % 3 {
                    0 if signed && v % 2 == 1 => -0.0,
                    0 => 0.0,
                    1 => 0.5,
                    _ => -1.25,
                })
                .collect()
        };
        for d in [6, 12] {
            let g = generators::hypercube(d);
            for generator in [Generator::Laplacian, Generator::Adjacency] {
                let walk = ContinuousWalk::new(&g, generator, 0.9).unwrap();
                let plus = walk
                    .clone()
                    .with_site_energies(&by_weight(d, false))
                    .unwrap();
                let minus = walk.with_site_energies(&by_weight(d, true)).unwrap();
                assert!(
                    minus
                        .site_energies()
                        .unwrap()
                        .iter()
                        .any(|e| e.is_sign_negative() && *e == 0.0)
                );
                assert_eq!(plus.hamiltonian(), minus.hamiltonian());
                assert_eq!(plus.spectrum(), minus.spectrum());
                for start in [0, 3] {
                    assert_eq!(
                        plus.limiting_distribution(start).unwrap(),
                        minus.limiting_distribution(start).unwrap()
                    );
                    assert_eq!(
                        plus.average_distribution(start, 2.0).unwrap(),
                        minus.average_distribution(start, 2.0).unwrap()
                    );
                    assert_eq!(
                        plus.probabilities(start, 0.8).unwrap(),
                        minus.probabilities(start, 0.8).unwrap()
                    );
                }
            }
        }
        assert_eq!(
            energy_colours(&[0.0, -0.0, 1.0, -0.0, 0.0], 2),
            [1, 1, 0, 1, 1]
        );
        assert_eq!(energy_colours(&[2.0, -1.0, -0.0, 2.0], 0), [0, 1, 2, 3]);
    }

    #[test]
    fn walks_stay_send_and_sync() {
        fn assert_traits<
            T: Send + Sync + Clone + std::panic::UnwindSafe + std::panic::RefUnwindSafe,
        >() {
        }
        assert_traits::<ContinuousWalk>();
    }

    /// The lazy eigendecomposition, first requested by several threads at
    /// once, is computed once and gives the results of a single thread.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn the_lazy_spectrum_is_shared_between_threads() {
        let g = generators::hypercube(7);
        let energies = rough_energies(g.num_nodes());
        let build = || {
            ContinuousWalk::new(&g, Generator::Laplacian, 1.0)
                .unwrap()
                .with_site_energies(&energies)
                .unwrap()
        };
        let serial = build();
        let want: Vec<Vec<f64>> = (0..8)
            .map(|s| serial.limiting_distribution(s).unwrap())
            .collect();
        let walk = build();
        let got: Vec<(Vec<f64>, usize)> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|s| {
                    let walk = &walk;
                    scope.spawn(move || {
                        let limit = walk.limiting_distribution(s).unwrap();
                        (limit, walk.spectrum().unwrap().as_ptr() as usize)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        // One eigendecomposition: every thread sees the same allocation.
        let shared = walk.spectrum().unwrap().as_ptr() as usize;
        for (s, (limit, ptr)) in got.iter().enumerate() {
            assert_eq!(limit, &want[s]);
            assert_eq!(*ptr, shared);
        }
        assert_eq!(walk.spectrum(), serial.spectrum());
    }
}
