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
//!   degenerate eigenvalues.
//! * [`search`] — Childs–Goldstone spatial search.
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
pub mod search;

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::linalg::{CMatrix, CsrMatrix, KrylovOptions, RMatrix, eigh_real, expm_multiply};

pub use classical::{classical_hitting_times, classical_walk_probabilities};
pub use dephasing::DephasingWalk;
pub use search::{SearchResult, SpatialSearch};

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

    /// Limiting (infinite-time-averaged) distribution
    /// `P̄(j) = Σ_λ |⟨j|Π_λ|start⟩|²`, where `Π_λ` projects onto the full
    /// eigenspace of `λ` (degeneracies are grouped, which v0.1 omitted).
    pub fn limiting_distribution(&self, start: usize) -> Result<Vec<f64>> {
        self.check_node(start)?;
        let (values, vectors) = self.require_dense()?;
        let mut dist = vec![0.0; self.n];
        for space in eigenspaces(values) {
            for (j, d) in dist.iter_mut().enumerate() {
                let amp: f64 = space
                    .clone()
                    .map(|k| vectors[(j, k)] * vectors[(start, k)])
                    .sum();
                *d += amp * amp;
            }
        }
        Ok(dist)
    }

    /// Exact time average `(1/T)∫₀ᵀ P(j, t) dt` of the node probabilities.
    pub fn average_distribution(&self, start: usize, t_max: f64) -> Result<Vec<f64>> {
        self.check_node(start)?;
        if t_max.is_nan() || t_max <= 0.0 {
            return self.probabilities(start, 0.0);
        }
        let (values, vectors) = self.require_dense()?;
        let n = self.n;
        // P(j,t) = Σ_ab c_a c_b V_ja V_jb e^{−i(λa−λb)t}, with c_k = V_{start,k}
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
        for a in 0..n {
            for b in 0..n {
                let k = kernel(values[a] - values[b]);
                let coeff = vectors[(start, a)] * vectors[(start, b)];
                if coeff == 0.0 {
                    continue;
                }
                for (j, p) in avg.iter_mut().enumerate() {
                    *p += (k * (coeff * vectors[(j, a)] * vectors[(j, b)])).re;
                }
            }
        }
        Ok(avg)
    }

    /// Smallest `T` on the grid `t_max·k/steps` at which the time-averaged
    /// distribution is within total-variation distance `epsilon` of the
    /// limiting distribution (the mixing time of Aharonov et al., 2001).
    pub fn mixing_time(
        &self,
        start: usize,
        epsilon: f64,
        t_max: f64,
        steps: usize,
    ) -> Result<Option<f64>> {
        let limit = self.limiting_distribution(start)?;
        for k in 1..=steps.max(1) {
            let t = t_max * k as f64 / steps.max(1) as f64;
            if total_variation(&self.average_distribution(start, t)?, &limit) <= epsilon {
                return Ok(Some(t));
            }
        }
        Ok(None)
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
