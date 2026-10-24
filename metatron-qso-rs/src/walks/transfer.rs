//! Quantum state transfer between two nodes.
//!
//! A continuous-time walk has *perfect state transfer* (PST) from `u` to
//! `v` at time `τ` if `|⟨v|e^{−iHτ}|u⟩| = 1`. With the spectral
//! decomposition `H = Σ_θ θ·E_θ`, this happens exactly when (Godsil 2012;
//! Coutinho 2014)
//!
//! 1. `u` and `v` are *strongly cospectral*: `E_θ e_v = σ_θ E_θ e_u` with
//!    `σ_θ = ±1` for every eigenvalue `θ`, and
//! 2. the eigenvalues `θ_0 < θ_1 < …` in the support of `u` (those with
//!    `E_θ e_u ≠ 0`) have differences `θ_k − θ_0 = n_k·g` for integers `n_k`
//!    with no common factor, `n_k` odd exactly when `σ_k ≠ σ_0`.
//!
//! The earliest PST time is then `π/g`. [`StateTransfer`] decides this
//! exactly up to floating-point tolerances (eigenvalue ratios are matched to
//! fractions with denominators up to 10⁴, and every positive answer is
//! confirmed by evaluating the amplitude), and finds the best transfer
//! fidelity within a time window when there is no PST.
//!
//! The analysis runs on the quotient by the coarsest equitable partition in
//! which `u` is a single cell (see [`quotient`](super::quotient)): every
//! amplitude from `u` lives there, PST requires `v` to be a single cell as
//! well, and symmetric graphs reduce to a few dimensions. Implicit families
//! (hypercubes, Hamming and Johnson graphs) are analysed through
//! [`StateTransfer::from_quotient`] without building the graph.
//!
//! ```
//! use metatron_qso::graph::generators;
//! use metatron_qso::walks::{Generator, StateTransfer};
//!
//! // The path on three nodes has PST between its ends at t = π/√2 …
//! let p3 = StateTransfer::new(&generators::path(3), Generator::Adjacency, 1.0, 0, 2).unwrap();
//! let t = p3.perfect_transfer_time().unwrap();
//! assert!((t - std::f64::consts::PI / 2f64.sqrt()).abs() < 1e-9);
//! // … the path on four nodes has none.
//! let p4 = StateTransfer::new(&generators::path(4), Generator::Adjacency, 1.0, 0, 3).unwrap();
//! assert!(p4.strongly_cospectral() && p4.perfect_transfer_time().is_none());
//! ```

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::search::optimize_peak;
use super::{Generator, Quotient, eigenspaces};
use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::linalg::eigh_real;

/// Best transfer found within a time window.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransferResult {
    /// Time of the highest fidelity.
    pub time: f64,
    /// `|⟨v|U(t)|u⟩|²` at that time.
    pub fidelity: f64,
}

/// State-transfer analysis between two nodes (see the module
/// documentation).
#[derive(Clone, Debug)]
pub struct StateTransfer {
    /// Eigenvalues of the reduced Hamiltonian (ascending).
    values: Vec<f64>,
    /// Components of the source and target cell states on the eigenvectors.
    source: Vec<f64>,
    target: Vec<f64>,
    /// `1/√|cell of v|`: the target node's share of its cell state.
    target_scale: f64,
    /// Distinct eigenvalues in the support of the source.
    support: Vec<f64>,
    /// `σ_θ` on the support, if the nodes are strongly cospectral.
    signs: Option<Vec<i8>>,
    pst_time: Option<f64>,
}

/// Best rational approximation `p/q` of `x` with `q ≤ max_den`, if it is
/// within `tol`.
fn rationalize(x: f64, max_den: i64, tol: f64) -> Option<(i64, i64)> {
    let (mut p0, mut q0, mut p1, mut q1) = (0i64, 1i64, 1i64, 0i64);
    let mut r = x;
    for _ in 0..64 {
        let a = r.floor();
        if a.abs() > 1e15 {
            return None;
        }
        let a = a as i64;
        let (p2, q2) = (
            a.checked_mul(p1)?.checked_add(p0)?,
            a.checked_mul(q1)?.checked_add(q0)?,
        );
        if q2 > max_den {
            return None;
        }
        if (x - p2 as f64 / q2 as f64).abs() <= tol * x.abs().max(1.0) {
            return Some((p2, q2));
        }
        (p0, q0, p1, q1) = (p1, q1, p2, q2);
        let frac = r - a as f64;
        if frac == 0.0 {
            return None;
        }
        r = 1.0 / frac;
    }
    None
}

fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

impl StateTransfer {
    /// Analyse transfer from node `u` to node `v ≠ u` of `graph` under
    /// `H = γL` or `H = −γA`.
    pub fn new(
        graph: &Graph,
        generator: Generator,
        gamma: f64,
        u: usize,
        v: usize,
    ) -> Result<Self> {
        let n = graph.num_nodes();
        if v >= n {
            return Err(QsoError::NodeOutOfRange {
                node: v,
                n_nodes: n,
            });
        }
        let quotient = Quotient::with_singletons(graph, &[u])?;
        let cells = quotient.cell_of().expect("explicit graph");
        Self::from_quotient(&quotient, generator, gamma, cells[u], cells[v])
    }

    /// Analyse transfer from the node of the single-node cell `from` to
    /// (a node of) cell `to` of a quotient. If `to` has more than one
    /// node, the amplitudes are those of each of its nodes, which share
    /// the cell's probability equally; PST is then impossible.
    pub fn from_quotient(
        quotient: &Quotient,
        generator: Generator,
        gamma: f64,
        from: usize,
        to: usize,
    ) -> Result<Self> {
        quotient.check_cell(from)?;
        quotient.check_cell(to)?;
        quotient.check_dense()?;
        if quotient.sizes()[from] != 1.0 {
            return Err(QsoError::invalid(
                "from",
                "the source must be a cell with a single node",
            ));
        }
        if from == to {
            return Err(QsoError::invalid("to", "source and target must differ"));
        }
        if !(gamma.is_finite() && gamma > 0.0) {
            return Err(QsoError::invalid(
                "gamma",
                "hopping rate must be positive and finite",
            ));
        }
        let (values, vectors) = eigh_real(&quotient.hamiltonian(generator, gamma));
        let k = values.len();
        let source: Vec<f64> = (0..k).map(|a| vectors[(from, a)]).collect();
        let target: Vec<f64> = (0..k).map(|a| vectors[(to, a)]).collect();
        let single_target = quotient.sizes()[to] == 1.0;
        let mut transfer = Self {
            values,
            source,
            target,
            target_scale: quotient.sizes()[to].sqrt().recip(),
            support: Vec::new(),
            signs: None,
            pst_time: None,
        };
        transfer.analyse(single_target);
        Ok(transfer)
    }

    fn analyse(&mut self, single_target: bool) {
        const TOL: f64 = 1e-8;
        let mut support = Vec::new();
        let mut signs = Vec::new();
        let mut cospectral = single_target;
        for space in eigenspaces(&self.values) {
            // Within an eigenspace, E_θ e_u and E_θ e_v have coordinates
            // source[a] and target[a] on its orthonormal basis.
            let norm_u: f64 = space.clone().map(|a| self.source[a].powi(2)).sum();
            let plus: f64 = space
                .clone()
                .map(|a| (self.target[a] - self.source[a]).powi(2))
                .sum();
            let minus: f64 = space
                .clone()
                .map(|a| (self.target[a] + self.source[a]).powi(2))
                .sum();
            let sign = if plus <= TOL * TOL {
                1
            } else if minus <= TOL * TOL {
                -1
            } else {
                cospectral = false;
                0
            };
            if norm_u > TOL * TOL {
                support.push(
                    self.values[space.start..space.end].iter().sum::<f64>() / space.len() as f64,
                );
                signs.push(sign);
            }
        }
        self.support = support;
        if !cospectral {
            return;
        }
        self.signs = Some(signs.clone());
        self.pst_time = self.pst(&signs);
    }

    /// Earliest PST time from the support and signs, confirmed numerically.
    fn pst(&self, signs: &[i8]) -> Option<f64> {
        let theta = &self.support;
        if theta.len() < 2 {
            return None;
        }
        // Differences D_k = θ_k − θ_0 as rational multiples of D_1.
        let d1 = theta[1] - theta[0];
        let mut fractions = Vec::with_capacity(theta.len() - 1);
        let mut lcm: i64 = 1;
        for &t in &theta[1..] {
            let (p, q) = rationalize((t - theta[0]) / d1, 10_000, 1e-9)?;
            lcm = lcm.checked_mul(q / gcd(lcm, q))?;
            fractions.push((p, q));
        }
        // D_k = (D_1/L)·N_k with integers N_k; divide out their gcd.
        let integers: Vec<i64> = fractions
            .iter()
            .map(|&(p, q)| p.checked_mul(lcm / q))
            .collect::<Option<_>>()?;
        let common = integers.iter().fold(0, |g, &x| gcd(g, x));
        let g = d1 / lcm as f64 * common as f64;
        for (k, &x) in integers.iter().enumerate() {
            let odd = (x / common) % 2 != 0;
            if odd != (signs[k + 1] != signs[0]) {
                return None;
            }
        }
        let tau = std::f64::consts::PI / g.abs();
        (self.fidelity(tau) >= 1.0 - 1e-8).then_some(tau)
    }

    /// `⟨v| e^{−iHt} |u⟩`.
    pub fn amplitude(&self, t: f64) -> C64 {
        let sum: C64 = (0..self.values.len())
            .map(|a| C64::from_polar(self.source[a] * self.target[a], -self.values[a] * t))
            .sum();
        sum * self.target_scale
    }

    /// Transfer fidelity `|⟨v| e^{−iHt} |u⟩|²`.
    pub fn fidelity(&self, t: f64) -> f64 {
        self.amplitude(t).norm_sqr()
    }

    /// Distinct eigenvalues of `H` in the support of `u` (ascending).
    pub fn support(&self) -> &[f64] {
        &self.support
    }

    /// Whether `u` and `v` are strongly cospectral.
    pub fn strongly_cospectral(&self) -> bool {
        self.signs.is_some()
    }

    /// The signs `σ_θ` with `E_θ e_v = σ_θ E_θ e_u` on the support, if `u`
    /// and `v` are strongly cospectral.
    pub fn signs(&self) -> Option<&[i8]> {
        self.signs.as_deref()
    }

    /// Earliest time of perfect state transfer, if there is one.
    pub fn perfect_transfer_time(&self) -> Option<f64> {
        self.pst_time
    }

    /// Highest fidelity over `t ∈ [0, t_max]`: grid scan with `resolution`
    /// points followed by golden-section refinement.
    pub fn best_transfer(&self, t_max: f64, resolution: usize) -> Result<TransferResult> {
        if !(t_max.is_finite() && t_max > 0.0) {
            return Err(QsoError::invalid("t_max", "must be finite and positive"));
        }
        let r = optimize_peak(0.0, t_max, resolution, |t| self.fidelity(t));
        Ok(TransferResult {
            time: r.time,
            fidelity: r.success_probability,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::generators;
    use crate::walks::ContinuousWalk;
    use std::f64::consts::{FRAC_PI_2, PI};

    #[test]
    fn rationals_are_recognised() {
        assert_eq!(rationalize(0.75, 100, 1e-12), Some((3, 4)));
        assert_eq!(rationalize(-2.0, 100, 1e-12), Some((-2, 1)));
        assert_eq!(rationalize(1.0 / 3.0 + 1e-13, 100, 1e-9), Some((1, 3)));
        assert_eq!(rationalize(2f64.sqrt(), 10_000, 1e-12), None);
        assert_eq!(gcd(12, -18), 6);
    }

    #[test]
    fn paths_have_pst_only_up_to_three_nodes() {
        let p2 = StateTransfer::new(&generators::path(2), Generator::Adjacency, 1.0, 0, 1).unwrap();
        assert!((p2.perfect_transfer_time().unwrap() - FRAC_PI_2).abs() < 1e-9);
        // Of the periodic perfect transfers the earliest is reported.
        let best = p2.best_transfer(60.0, 20_000).unwrap();
        assert!((best.time - FRAC_PI_2).abs() < 1e-7 && best.fidelity > 1.0 - 1e-12);
        let p3 = StateTransfer::new(&generators::path(3), Generator::Adjacency, 1.0, 0, 2).unwrap();
        assert!((p3.perfect_transfer_time().unwrap() - PI / 2f64.sqrt()).abs() < 1e-9);
        assert_eq!(p3.signs().unwrap(), &[1, -1, 1]);
        for n in 4..=12 {
            let t = StateTransfer::new(&generators::path(n), Generator::Adjacency, 1.0, 0, n - 1)
                .unwrap();
            assert!(t.strongly_cospectral(), "P{n}");
            assert!(t.perfect_transfer_time().is_none(), "P{n}");
        }
        // Best end-to-end transfer within t ≤ 50 (checked against a scan
        // with 200001 points in numpy). P5 comes close (pretty good state
        // transfer: n + 1 is prime), P8 does not.
        for (n, time, fidelity) in [
            (4, 28.09925, 0.992356742),
            (5, 47.14125, 0.999695639),
            (8, 5.04075, 0.854156303),
        ] {
            let t = StateTransfer::new(&generators::path(n), Generator::Adjacency, 1.0, 0, n - 1)
                .unwrap();
            let best = t.best_transfer(50.0, 20_000).unwrap();
            assert!((best.time - time).abs() < 1e-3, "P{n}: {best:?}");
            assert!(
                best.fidelity >= fidelity - 1e-9 && best.fidelity < fidelity + 1e-6,
                "P{n}"
            );
        }
        // With the Laplacian only P2 has PST.
        let l3 = StateTransfer::new(&generators::path(3), Generator::Laplacian, 1.0, 0, 2).unwrap();
        assert!(l3.perfect_transfer_time().is_none());
        // Inner and end nodes are not cospectral.
        let mid =
            StateTransfer::new(&generators::path(5), Generator::Adjacency, 1.0, 0, 1).unwrap();
        assert!(!mid.strongly_cospectral());
    }

    #[test]
    fn hypercubes_have_antipodal_pst() {
        for d in 1..=6 {
            let g = generators::hypercube(d);
            let antipode = (1 << d) - 1;
            for generator in [Generator::Adjacency, Generator::Laplacian] {
                let t = StateTransfer::new(&g, generator, 1.0, 0, antipode).unwrap();
                assert!(
                    (t.perfect_transfer_time().unwrap() - FRAC_PI_2).abs() < 1e-9,
                    "Q{d}"
                );
            }
            if d >= 2 {
                let t = StateTransfer::new(&g, Generator::Adjacency, 1.0, 0, 1).unwrap();
                assert!(t.perfect_transfer_time().is_none());
            }
        }
        // Q30 from its 31-cell quotient: |amplitude| = |sin t|^30.
        let q = Quotient::hypercube(30).unwrap();
        let t = StateTransfer::from_quotient(&q, Generator::Laplacian, 2.0, 0, 30).unwrap();
        assert!((t.perfect_transfer_time().unwrap() - FRAC_PI_2 / 2.0).abs() < 1e-9);
        assert!((t.amplitude(0.3).norm() - 0.6f64.sin().powi(30)).abs() < 1e-12);
        assert_eq!(t.support().len(), 31);
        // The weight-15 cell holds C(30,15) nodes: no PST into any of them.
        let mid = StateTransfer::from_quotient(&q, Generator::Laplacian, 1.0, 0, 15).unwrap();
        assert!(!mid.strongly_cospectral() && mid.perfect_transfer_time().is_none());
    }

    #[test]
    fn weighted_krawtchouk_chain_has_pst_for_every_length() {
        // Christandl et al. (2004): couplings √(k(N−k)) transfer end to end
        // at t = π/2 for any number of nodes N.
        for n in [2usize, 5, 8, 13] {
            let edges: Vec<(usize, usize, f64)> = (1..n)
                .map(|k| (k - 1, k, ((k * (n - k)) as f64).sqrt()))
                .collect();
            let g = Graph::from_weighted_edges(n, &edges).unwrap();
            let t = StateTransfer::new(&g, Generator::Adjacency, 1.0, 0, n - 1).unwrap();
            assert!(
                (t.perfect_transfer_time().unwrap() - FRAC_PI_2).abs() < 1e-9,
                "N={n}"
            );
        }
    }

    #[test]
    fn amplitudes_match_the_full_propagator() {
        let g = generators::petersen();
        let walk = ContinuousWalk::new(&g, Generator::Laplacian, 0.7).unwrap();
        for v in [1, 2, 7] {
            let t = StateTransfer::new(&g, Generator::Laplacian, 0.7, 0, v).unwrap();
            for time in [0.4, 1.9] {
                let u = walk.propagator(time).unwrap();
                assert!((t.amplitude(time) - u[(v, 0)]).norm() < 1e-12);
            }
            // Vertices of the Petersen graph are not strongly cospectral.
            assert!(!t.strongly_cospectral());
        }
        // Antipodes of C4 and C6 are strongly cospectral; only C4 has PST
        // (on C6 the eigenvalue differences 1, 3, 4 have the wrong parity).
        let c4 =
            StateTransfer::new(&generators::cycle(4), Generator::Adjacency, 1.0, 0, 2).unwrap();
        assert!((c4.perfect_transfer_time().unwrap() - FRAC_PI_2).abs() < 1e-9);
        let c6 =
            StateTransfer::new(&generators::cycle(6), Generator::Adjacency, 1.0, 0, 3).unwrap();
        assert!(c6.strongly_cospectral() && c6.perfect_transfer_time().is_none());
        assert!(StateTransfer::new(&g, Generator::Laplacian, 1.0, 0, 0).is_err());
        assert!(StateTransfer::new(&g, Generator::Laplacian, 1.0, 0, 10).is_err());
    }
}
