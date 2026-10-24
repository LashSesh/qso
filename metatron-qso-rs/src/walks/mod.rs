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
//!   degenerate eigenvalues, beyond `DENSE_LIMIT` through the start node's
//!   cyclic subspace.
//! * [`search`] — Childs–Goldstone spatial search.
//! * [`quotient`] — exact walks and searches reduced by symmetry to a few
//!   dimensions (equitable partitions), for graphs of any size.
//! * [`transfer`] — perfect state transfer (decision and time) and the best
//!   transfer fidelity between two nodes.
//! * [`dephasing`] — open-system walks with site dephasing (Lindblad).
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

pub mod classical;
pub mod dephasing;
pub mod quotient;
pub mod search;
pub mod transfer;

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::linalg::{
    CMatrix, CsrMatrix, KrylovOptions, RMatrix, cyclic_spectrum, eigh_real, expm_multiply,
};

pub use classical::{classical_hitting_times, classical_walk_probabilities};
pub use dephasing::DephasingWalk;
pub use quotient::{Quotient, QuotientSearch, QuotientWalk};
pub use search::{SearchResult, SpatialSearch};
pub use transfer::{StateTransfer, TransferResult};

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

/// Group sorted eigenvalues into degenerate clusters (relative tolerance).
pub(crate) fn eigenspaces(values: &[f64]) -> Vec<std::ops::Range<usize>> {
    let scale = values.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    let tol = 1e-9 * scale;
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

    /// Exact time average `(1/T)∫₀ᵀ P(j, t) dt`.
    pub fn average_distribution(&self, t_max: f64) -> Vec<f64> {
        let (n, m) = self.vectors.shape();
        // P(j,t) = Σ_ab c_a c_b x_a(j) x_b(j) e^{−i(λa−λb)t}
        let kernel = |d: f64| {
            let x = d * t_max;
            if x.abs() < 1e-12 {
                C64::new(1.0, 0.0)
            } else {
                // (1/T)∫ e^{−idt} dt = (1 − e^{−idT})/(idT)
                (C64::new(1.0, 0.0) - C64::from_polar(1.0, -x)) / C64::new(0.0, x)
            }
        };
        let mut avg = vec![0.0; n];
        for a in 0..m {
            for b in 0..m {
                let coeff = self.coeffs[a] * self.coeffs[b];
                if coeff == 0.0 {
                    continue;
                }
                let k = kernel(self.values[a] - self.values[b]).re * coeff;
                for (j, p) in avg.iter_mut().enumerate() {
                    *p += k * self.vectors[(j, a)] * self.vectors[(j, b)];
                }
            }
        }
        avg
    }
}

/// Largest cyclic subspace used for walks beyond [`DENSE_LIMIT`] nodes: at
/// most `DENSE_LIMIT` vectors and `2²⁵` stored amplitudes.
fn cyclic_limit(n: usize) -> usize {
    DENSE_LIMIT.min((1usize << 25) / n.max(1)).max(1)
}

/// Total-variation distance `½ Σ |p − q|`.
pub fn total_variation(p: &[f64], q: &[f64]) -> f64 {
    0.5 * p.iter().zip(q).map(|(a, b)| (a - b).abs()).sum::<f64>()
}

/// Continuous-time quantum walk on a graph.
#[derive(Clone, Debug)]
pub struct ContinuousWalk {
    n: usize,
    gamma: f64,
    generator: Generator,
    hamiltonian: CsrMatrix,
    /// Eigenvalues (ascending) and eigenvectors (columns) for n ≤ DENSE_LIMIT.
    spectral: Option<(Vec<f64>, RMatrix)>,
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
        let spectral = (n <= DENSE_LIMIT).then(|| eigh_real(&hamiltonian.to_dense()));
        Ok(Self {
            n,
            gamma,
            generator,
            hamiltonian,
            spectral,
        })
    }

    pub fn num_nodes(&self) -> usize {
        self.n
    }

    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    pub fn generator(&self) -> Generator {
        self.generator
    }

    /// The (sparse) Hamiltonian.
    pub fn hamiltonian(&self) -> &CsrMatrix {
        &self.hamiltonian
    }

    /// Eigenvalues of `H` (ascending), if computed.
    pub fn spectrum(&self) -> Option<&[f64]> {
        self.spectral.as_ref().map(|(v, _)| v.as_slice())
    }

    fn require_dense(&self) -> Result<&(Vec<f64>, RMatrix)> {
        self.spectral.as_ref().ok_or(QsoError::TooManyQubits {
            what: "dense walk analysis (nodes)",
            max: DENSE_LIMIT,
            requested: self.n,
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
        match &self.spectral {
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
    /// start node's cyclic subspace (see [`cyclic_spectrum`]).
    pub(crate) fn start_spectrum(&self, start: usize) -> Result<StartSpectrum<'_>> {
        self.check_node(start)?;
        match &self.spectral {
            Some((values, vectors)) => Ok(StartSpectrum {
                values: std::borrow::Cow::Borrowed(values),
                coeffs: (0..self.n).map(|k| vectors[(start, k)]).collect(),
                vectors: std::borrow::Cow::Borrowed(vectors),
            }),
            None => {
                let mut v = vec![0.0; self.n];
                v[start] = 1.0;
                let h = &self.hamiltonian;
                let c = cyclic_spectrum(|x, y| h.matvec(x, y), &v, cyclic_limit(self.n))?;
                Ok(StartSpectrum {
                    values: std::borrow::Cow::Owned(c.values),
                    coeffs: c.weights,
                    vectors: std::borrow::Cow::Owned(c.vectors),
                })
            }
        }
    }

    /// Dimension of the smallest invariant subspace containing `|start⟩`:
    /// the number of distinct eigenvalues the walker ever sees.
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
    /// Beyond [`DENSE_LIMIT`] nodes the start node's cyclic subspace is
    /// used, which is exact and small for symmetric graphs (21 dimensions on
    /// the 2²⁰-node hypercube); the call fails if that subspace exceeds
    /// `min(DENSE_LIMIT, 2²⁵/n)` dimensions.
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
        // Cyclic and dense decompositions agree where both exist.
        let small =
            ContinuousWalk::new(&generators::petersen(), Generator::Adjacency, 1.0).unwrap();
        let h = small.hamiltonian();
        let mut v = vec![0.0; 10];
        v[2] = 1.0;
        let c = crate::linalg::cyclic_spectrum(|x, y| h.matvec(x, y), &v, 10).unwrap();
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
        assert!(crate::linalg::cyclic_spectrum(|x, y| h.matvec(x, y), &v, 10).is_err());
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
}
