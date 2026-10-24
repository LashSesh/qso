//! Spatial search by continuous-time quantum walk (Childs & Goldstone,
//! Phys. Rev. A 70, 022314, 2004).
//!
//! The walker starts in the uniform superposition `|s⟩` and evolves under
//! `H = γ·L − Σ_{w∈M} |w⟩⟨w|` (or `−γA − Σ|w⟩⟨w|`). At the critical hopping
//! rate the marked nodes are found with high probability after `O(√N)` time
//! on complete graphs, hypercubes and lattices of dimension > 4 — a
//! quadratic speed-up over the `Θ(N)` classical search time.
//!
//! ```
//! use metatron_qso::graph::metatron;
//! use metatron_qso::walks::{Generator, SpatialSearch};
//!
//! // Search for one node of Metatron's Cube (K13) at γ = 1/N.
//! let search = SpatialSearch::new(&metatron::metatron_cube(), &[4], Generator::Laplacian, 1.0 / 13.0).unwrap();
//! let t = std::f64::consts::PI * 13f64.sqrt() / 2.0;
//! assert!((search.success_probability(t) - 1.0).abs() < 1e-10);
//! ```

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::{DENSE_LIMIT, Generator};
use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::linalg::{RMatrix, eigh_real};

/// Outcome of an optimised search run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    pub gamma: f64,
    /// Evolution time at which the success probability peaks.
    pub time: f64,
    /// Probability of measuring a marked node at `time`.
    pub success_probability: f64,
    /// `time / success_probability`: expected total evolution time when
    /// the run is repeated until success.
    pub expected_time: f64,
}

/// Quantum-walk search for a set of marked nodes.
#[derive(Clone, Debug)]
pub struct SpatialSearch {
    n: usize,
    marked: Vec<usize>,
    gamma: f64,
    values: Vec<f64>,
    /// Eigenvector components of the start state, `⟨k|s⟩`.
    overlaps: Vec<f64>,
    vectors: RMatrix,
}

impl SpatialSearch {
    /// Build the search Hamiltonian on `graph` with marked nodes `marked`.
    pub fn new(graph: &Graph, marked: &[usize], generator: Generator, gamma: f64) -> Result<Self> {
        let n = graph.num_nodes();
        if n == 0 || n > DENSE_LIMIT {
            return Err(QsoError::invalid(
                "graph",
                format!("need 1..={DENSE_LIMIT} nodes"),
            ));
        }
        if marked.is_empty() {
            return Err(QsoError::invalid(
                "marked",
                "at least one marked node required",
            ));
        }
        for &w in marked {
            if w >= n {
                return Err(QsoError::NodeOutOfRange {
                    node: w,
                    n_nodes: n,
                });
            }
        }
        if !(gamma.is_finite() && gamma >= 0.0) {
            return Err(QsoError::invalid(
                "gamma",
                "must be finite and non-negative",
            ));
        }
        let mut h = match generator {
            Generator::Laplacian => graph.laplacian_matrix() * gamma,
            Generator::Adjacency => graph.adjacency_matrix() * -gamma,
        };
        let mut marked = marked.to_vec();
        marked.sort_unstable();
        marked.dedup();
        for &w in &marked {
            h[(w, w)] -= 1.0;
        }
        let (values, vectors) = eigh_real(&h);
        let s = 1.0 / (n as f64).sqrt();
        let overlaps = (0..n)
            .map(|k| (0..n).map(|j| vectors[(j, k)] * s).sum())
            .collect();
        Ok(Self {
            n,
            marked,
            gamma,
            values,
            overlaps,
            vectors,
        })
    }

    /// Optimal hopping rate `γ = 1/N` for the complete graph `K_N` with the
    /// Laplacian generator (the large-N limit of [`SpatialSearch::critical_gamma`]).
    pub fn critical_gamma_complete(n: usize) -> f64 {
        1.0 / n as f64
    }

    /// Childs–Goldstone critical hopping rate `γ_c = (1/N)·Σ_{λ_k>0} 1/λ_k`
    /// over the non-zero Laplacian eigenvalues. It is the optimal rate for
    /// vertex-transitive graphs (for regular graphs the adjacency generator
    /// gives the same value, as `L = dI − A`).
    pub fn critical_gamma(graph: &Graph) -> Result<f64> {
        let n = graph.num_nodes();
        if n < 2 || n > DENSE_LIMIT || !graph.is_connected() {
            return Err(QsoError::invalid(
                "graph",
                "need a connected graph with 2..=DENSE_LIMIT nodes",
            ));
        }
        let spectrum = graph.laplacian_spectrum();
        let tol = 1e-9 * spectrum.last().copied().unwrap_or(1.0).max(1.0);
        Ok(spectrum
            .iter()
            .filter(|&&l| l > tol)
            .map(|l| 1.0 / l)
            .sum::<f64>()
            / n as f64)
    }

    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    pub fn marked(&self) -> &[usize] {
        &self.marked
    }

    /// Probability of finding a marked node at time `t`.
    pub fn success_probability(&self, t: f64) -> f64 {
        let phases: Vec<C64> = self
            .values
            .iter()
            .zip(&self.overlaps)
            .map(|(&e, &o)| C64::from_polar(o, -e * t))
            .collect();
        self.marked
            .iter()
            .map(|&w| {
                let amp: C64 = (0..self.n).map(|k| phases[k] * self.vectors[(w, k)]).sum();
                amp.norm_sqr()
            })
            .sum()
    }

    /// Maximise the success probability over `t ∈ [0, t_max]`: grid scan
    /// with `resolution` points followed by golden-section refinement.
    pub fn optimize_time(&self, t_max: f64, resolution: usize) -> SearchResult {
        let resolution = resolution.max(2);
        let dt = t_max / (resolution - 1) as f64;
        let (mut best_t, mut best_p) = (0.0, self.success_probability(0.0));
        for k in 1..resolution {
            let t = k as f64 * dt;
            let p = self.success_probability(t);
            if p > best_p {
                best_t = t;
                best_p = p;
            }
        }
        // Golden-section search on [best_t − dt, best_t + dt].
        let (mut a, mut b) = ((best_t - dt).max(0.0), (best_t + dt).min(t_max));
        let g = (5f64.sqrt() - 1.0) / 2.0;
        for _ in 0..60 {
            let c = b - g * (b - a);
            let d = a + g * (b - a);
            if self.success_probability(c) > self.success_probability(d) {
                b = d;
            } else {
                a = c;
            }
        }
        let t = 0.5 * (a + b);
        let p = self.success_probability(t);
        let (time, success_probability) = if p > best_p { (t, p) } else { (best_t, best_p) };
        SearchResult {
            gamma: self.gamma,
            time,
            success_probability,
            expected_time: if success_probability > 0.0 {
                time / success_probability
            } else {
                f64::INFINITY
            },
        }
    }

    /// Scan hopping rates and return the best [`SearchResult`].
    pub fn scan_gamma(
        graph: &Graph,
        marked: &[usize],
        generator: Generator,
        gammas: &[f64],
        t_max: f64,
        resolution: usize,
    ) -> Result<SearchResult> {
        let mut best: Option<SearchResult> = None;
        for &gamma in gammas {
            let r = SpatialSearch::new(graph, marked, generator, gamma)?
                .optimize_time(t_max, resolution);
            if best
                .as_ref()
                .is_none_or(|b| r.success_probability > b.success_probability)
            {
                best = Some(r);
            }
        }
        best.ok_or_else(|| QsoError::invalid("gammas", "no hopping rates given"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::generators;
    use std::f64::consts::PI;

    #[test]
    fn complete_graph_search_is_exact_at_critical_gamma() {
        for n in [13usize, 64, 100] {
            let g = generators::complete(n);
            let gamma = SpatialSearch::critical_gamma_complete(n);
            let s = SpatialSearch::new(&g, &[n / 3], Generator::Laplacian, gamma).unwrap();
            let t = PI * (n as f64).sqrt() / 2.0;
            assert!((s.success_probability(t) - 1.0).abs() < 1e-10, "N={n}");
            assert!((s.success_probability(0.0) - 1.0 / n as f64).abs() < 1e-12);
            let best = s.optimize_time(2.0 * t, 400);
            assert!((best.time - t).abs() < 1e-4 && best.success_probability > 1.0 - 1e-9);
        }
    }

    #[test]
    fn search_time_scales_as_sqrt_n() {
        // Optimal times on K_N follow π√N/2: quadrupling N doubles the time.
        let t = |n: usize| {
            let g = generators::complete(n);
            SpatialSearch::new(&g, &[0], Generator::Laplacian, 1.0 / n as f64)
                .unwrap()
                .optimize_time(PI * (n as f64).sqrt(), 300)
                .time
        };
        assert!((t(256) / t(64) - 2.0).abs() < 1e-3);
    }

    #[test]
    fn hypercube_search_at_critical_gamma() {
        // Q_d: Laplacian eigenvalues 2r with multiplicity C(d,r), so
        // γ_c = 2^{−d}·Σ_r C(d,r)/(2r). For d = 8 the peak success
        // probability is ≈ 0.755 at t ≈ 27.5 (cross-checked independently
        // with numpy); it tends to 1 only as d → ∞.
        let d = 8;
        let g = generators::hypercube(d);
        let binom =
            |n: u64, k: u64| (1..=k).fold(1.0, |acc, i| acc * (n - k + i) as f64 / i as f64);
        let expected: f64 = (1..=d as u64)
            .map(|r| binom(d as u64, r) / (2.0 * r as f64))
            .sum::<f64>()
            / 256.0;
        let gamma = SpatialSearch::critical_gamma(&g).unwrap();
        assert!((gamma - expected).abs() < 1e-12);
        let at_critical = SpatialSearch::new(&g, &[17], Generator::Adjacency, gamma)
            .unwrap()
            .optimize_time(64.0, 1000);
        assert!(
            (at_critical.success_probability - 0.7546).abs() < 2e-3,
            "{at_critical:?}"
        );
        assert!((at_critical.time - 27.5).abs() < 0.5);
        // No other rate on a fine grid does noticeably better.
        let gammas: Vec<f64> = (0..=40).map(|k| 0.10 + 0.0025 * k as f64).collect();
        let best = SpatialSearch::scan_gamma(&g, &[17], Generator::Adjacency, &gammas, 64.0, 1000)
            .unwrap();
        assert!(best.success_probability < at_critical.success_probability + 2e-3);
        // K_N: γ_c = (N−1)/N².
        let k = generators::complete(13);
        assert!((SpatialSearch::critical_gamma(&k).unwrap() - 12.0 / 169.0).abs() < 1e-12);
    }

    #[test]
    fn wrong_gamma_fails_to_amplify() {
        let n = 64;
        let g = generators::complete(n);
        let s = SpatialSearch::new(&g, &[0], Generator::Laplacian, 10.0 / n as f64).unwrap();
        assert!(s.optimize_time(40.0, 800).success_probability < 0.2);
    }
}
