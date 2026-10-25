//! QAOA for weighted MaxCut (Farhi, Goldstone & Gutmann, arXiv:1411.4028).
//!
//! The state `|γ,β⟩ = Π_l e^{−iβ_l B} e^{−iγ_l C} |+⟩^{⊗n}` is simulated on
//! the full 2ⁿ-dimensional register: the cost `C` is diagonal (the cut
//! weight of each bit string, precomputed once) and the mixer
//! `B = Σ_q X_q` factorises into `RX(2β)` on every qubit. Gradients are
//! exact (adjoint method specialised to the QAOA structure). The exact
//! optimum is obtained by enumeration, so approximation ratios are true
//! ratios `⟨C⟩ / C_max`.
//!
//! ```
//! use metatron_qso::algorithms::qaoa::{MaxCutQaoa, QaoaConfig};
//! use metatron_qso::graph::generators;
//!
//! // p = 1 on a ring achieves exactly 3/4 (Farhi et al.).
//! let qaoa = MaxCutQaoa::new(&generators::cycle(8), 1).unwrap();
//! let result = qaoa.optimize(&QaoaConfig::default()).unwrap();
//! assert!((result.approximation_ratio.unwrap() - 0.75).abs() < 1e-6);
//! ```

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::linalg::inner;
use crate::optimize::{Method, Objective, Options, minimize};
use crate::parallel::{for_each_mut, map_collect, map_tasks, sum_c64, sum_f64};
use crate::quantum::kernels;
use crate::quantum::plan::{BLOCK_QUBITS, for_each_block};
use crate::quantum::{MAX_QUBITS, StateVector};
use crate::rng::Rng;

/// Optimisation settings for [`MaxCutQaoa::optimize`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QaoaConfig {
    pub method: Method,
    pub options: Options,
    /// Number of random starts (run in parallel).
    pub restarts: usize,
    pub seed: u64,
    /// Optional starting point `[γ…, β…]` used for the first start.
    pub initial: Option<Vec<f64>>,
}

impl Default for QaoaConfig {
    fn default() -> Self {
        Self {
            method: Method::lbfgs(),
            options: Options::default(),
            restarts: 8,
            seed: 0,
            initial: None,
        }
    }
}

/// Optimised QAOA parameters and the statistics of the resulting state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QaoaResult {
    pub p: usize,
    pub gammas: Vec<f64>,
    pub betas: Vec<f64>,
    /// `⟨C⟩` — expected cut weight.
    pub expectation: f64,
    /// Exact maximum cut weight (enumeration).
    pub max_cut: f64,
    /// `⟨C⟩ / C_max` (`None` if `C_max ≤ 0`).
    pub approximation_ratio: Option<f64>,
    /// Probability that a measurement yields an optimal cut.
    pub optimal_probability: f64,
    /// Most likely bit string (bit `q` = side of node `q`) and its weight.
    pub most_likely: usize,
    pub most_likely_cut: f64,
    pub iterations: usize,
    pub evaluations: usize,
    pub converged: bool,
}

/// Largest QAOA depth `p` accepted (`2p` parameters, `2p` layers per
/// evaluation).
pub const MAX_QAOA_DEPTH: usize = 10_000;

fn check_depth(p: usize) -> Result<()> {
    if p == 0 || p > MAX_QAOA_DEPTH {
        return Err(QsoError::invalid(
            "p",
            format!("QAOA depth must lie in 1..={MAX_QAOA_DEPTH}, got {p}"),
        ));
    }
    Ok(())
}

/// QAOA instance for MaxCut on a fixed graph and depth `p`.
#[derive(Clone, Debug)]
pub struct MaxCutQaoa {
    n: usize,
    p: usize,
    cut: Vec<f64>,
    max_cut: f64,
    /// Distinct cut values and the index of each string's value, when there
    /// are at most 2¹⁶ of them (always for unweighted graphs): a cost layer
    /// then looks phases up in a table instead of evaluating `e^{−iγc}`
    /// for every amplitude.
    levels: Option<(Vec<f64>, Vec<u16>)>,
}

/// Distinct values of `cut` and the index of every entry, if there are at
/// most 2¹⁶ distinct values.
fn cut_levels(cut: &[f64]) -> Option<(Vec<f64>, Vec<u16>)> {
    let mut values = cut.to_vec();
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        values.par_sort_unstable_by(f64::total_cmp);
    }
    #[cfg(not(feature = "parallel"))]
    values.sort_unstable_by(f64::total_cmp);
    values.dedup_by(|a, b| a.total_cmp(b).is_eq());
    if values.len() > 1 << 16 {
        return None;
    }
    let index = map_collect(cut.len(), |b| {
        values
            .binary_search_by(|v| v.total_cmp(&cut[b]))
            .expect("every value is listed") as u16
    });
    Some((values, index))
}

/// Phase factor `e^{i·sign·γ·cut[b]}` for every string, through the level
/// table when there is one.
enum Phases<'a> {
    Table(Vec<C64>, &'a [u16]),
    Direct(&'a [f64], f64),
}

impl Phases<'_> {
    #[inline]
    fn at(&self, b: usize) -> C64 {
        match self {
            Phases::Table(table, index) => table[index[b] as usize],
            Phases::Direct(cut, angle) => C64::from_polar(1.0, angle * cut[b]),
        }
    }
}

/// Apply `e^{−iγC}` (if `cost` is given) and then `e^{−iβ Σ X}` (or its
/// inverse for negative β). The phases and the rotations of the qubits
/// below the block size are applied block by block in one pass over
/// memory; the higher qubits follow one pass each.
fn layer(amps: &mut [C64], n: usize, cost: Option<&Phases<'_>>, beta: f64) {
    let (c, s) = (beta.cos(), beta.sin());
    let m = [
        C64::new(c, 0.0),
        C64::new(0.0, -s),
        C64::new(0.0, -s),
        C64::new(c, 0.0),
    ];
    let low = BLOCK_QUBITS.min(n);
    for_each_block(amps, low, |base, block| {
        if let Some(phases) = cost {
            for (i, a) in block.iter_mut().enumerate() {
                *a *= phases.at(base + i);
            }
        }
        for q in 0..low {
            kernels::apply_1q(block, q, 0, &m);
        }
    });
    for q in low..n {
        kernels::apply_1q(amps, q, 0, &m);
    }
}

fn mixer(amps: &mut [C64], n: usize, beta: f64) {
    layer(amps, n, None, beta);
}

/// `⟨λ|Σ_q X_q|φ⟩`: the qubits below the block size block by block, the
/// others in one streaming pass each. The sum is ordered, so the result
/// does not depend on the number of threads.
fn mixer_overlap(lambda: &[C64], phi: &[C64], n: usize) -> C64 {
    let low = BLOCK_QUBITS.min(n);
    let block = 1usize << low;
    let per_block = map_tasks(phi.len() / block, |k| {
        let (l, p) = (
            &lambda[k * block..(k + 1) * block],
            &phi[k * block..(k + 1) * block],
        );
        let mut acc = C64::new(0.0, 0.0);
        for q in 0..low {
            let bit = 1usize << q;
            for i in 0..block {
                acc += l[i].conj() * p[i ^ bit];
            }
        }
        acc
    });
    let mut total: C64 = per_block.into_iter().sum();
    for q in low..n {
        let bit = 1usize << q;
        total += sum_c64(phi.len(), |b| lambda[b].conj() * phi[b ^ bit]);
    }
    total
}

impl MaxCutQaoa {
    /// Build the instance (precomputes the cut weight of all 2ⁿ strings).
    pub fn new(graph: &Graph, p: usize) -> Result<Self> {
        let n = graph.num_nodes();
        if n == 0 || graph.num_edges() == 0 {
            return Err(QsoError::invalid("graph", "MaxCut needs at least one edge"));
        }
        if n > MAX_QUBITS {
            return Err(QsoError::TooManyQubits {
                what: "QAOA",
                max: MAX_QUBITS,
                requested: n,
            });
        }
        check_depth(p)?;
        let edges = graph.edges().to_vec();
        let cut = map_collect(1usize << n, |b| {
            edges
                .iter()
                .filter(|&&(u, v, _)| ((b >> u) ^ (b >> v)) & 1 == 1)
                .map(|&(_, _, w)| w)
                .sum()
        });
        let max_cut = cut.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let levels = cut_levels(&cut);
        Ok(Self {
            n,
            p,
            cut,
            max_cut,
            levels,
        })
    }

    /// The same graph at a different depth (reuses the cut table).
    pub fn with_depth(&self, p: usize) -> Result<Self> {
        check_depth(p)?;
        Ok(Self { p, ..self.clone() })
    }

    pub fn n_qubits(&self) -> usize {
        self.n
    }

    pub fn depth(&self) -> usize {
        self.p
    }

    /// `2p` parameters `[γ₁ … γ_p, β₁ … β_p]`.
    pub fn num_parameters(&self) -> usize {
        2 * self.p
    }

    /// Exact maximum cut weight.
    pub fn max_cut(&self) -> f64 {
        self.max_cut
    }

    /// Weight of the cut encoded by bit string `b`.
    pub fn cut_value(&self, b: usize) -> f64 {
        self.cut[b]
    }

    /// All optimal bit strings.
    pub fn optimal_cuts(&self) -> Vec<usize> {
        let tol = 1e-9 * self.max_cut.abs().max(1.0);
        (0..self.cut.len())
            .filter(|&b| self.max_cut - self.cut[b] <= tol)
            .collect()
    }

    /// Phases `e^{i·angle·cut[b]}`.
    fn phases(&self, angle: f64) -> Phases<'_> {
        match &self.levels {
            Some((values, index)) => Phases::Table(
                values
                    .iter()
                    .map(|&v| C64::from_polar(1.0, angle * v))
                    .collect(),
                index,
            ),
            None => Phases::Direct(&self.cut, angle),
        }
    }

    fn check(&self, params: &[f64]) -> Result<()> {
        if params.len() != 2 * self.p {
            return Err(QsoError::ParameterCount {
                expected: 2 * self.p,
                actual: params.len(),
            });
        }
        Ok(())
    }

    fn evolve(&self, params: &[f64]) -> Vec<C64> {
        let (gammas, betas) = params.split_at(self.p);
        let dim = self.cut.len();
        let mut amps = vec![C64::new(1.0 / (dim as f64).sqrt(), 0.0); dim];
        for l in 0..self.p {
            layer(&mut amps, self.n, Some(&self.phases(-gammas[l])), betas[l]);
        }
        amps
    }

    /// QAOA state for `params`.
    pub fn state(&self, params: &[f64]) -> Result<StateVector> {
        self.check(params)?;
        StateVector::from_amplitudes(self.evolve(params))
    }

    /// Expected cut weight `⟨C⟩`.
    pub fn expectation(&self, params: &[f64]) -> Result<f64> {
        self.check(params)?;
        let amps = self.evolve(params);
        Ok(sum_f64(amps.len(), |b| amps[b].norm_sqr() * self.cut[b]))
    }

    /// `⟨C⟩` and its exact gradient.
    pub fn expectation_and_gradient(&self, params: &[f64]) -> Result<(f64, Vec<f64>)> {
        self.check(params)?;
        let (gammas, betas) = params.split_at(self.p);
        let mut phi = self.evolve(params);
        let cut = &self.cut;
        let mut lambda: Vec<C64> = map_collect(phi.len(), |b| phi[b] * cut[b]);
        let value = inner(&phi, &lambda).re;
        let mut grad = vec![0.0; 2 * self.p];
        let n = self.n;
        for l in (0..self.p).rev() {
            // Mixer: ∂/∂β = 2·Im⟨λ|B|φ⟩ with B = Σ_q X_q.
            let b_phi = mixer_overlap(&lambda, &phi, n);
            grad[self.p + l] = 2.0 * b_phi.im;
            mixer(&mut phi, n, -betas[l]);
            mixer(&mut lambda, n, -betas[l]);
            // Cost: ∂/∂γ = 2·Im⟨λ|C|φ⟩.
            let c_phi = sum_c64(phi.len(), |b| lambda[b].conj() * phi[b] * cut[b]);
            grad[l] = 2.0 * c_phi.im;
            let phases = self.phases(gammas[l]);
            for_each_mut(&mut phi, |b, a| *a *= phases.at(b));
            for_each_mut(&mut lambda, |b, a| *a *= phases.at(b));
        }
        Ok((value, grad))
    }

    /// Full statistics of the state at `params`.
    pub fn analyze(&self, params: &[f64]) -> Result<QaoaResult> {
        self.check(params)?;
        let amps = self.evolve(params);
        let probs: Vec<f64> = amps.iter().map(|a| a.norm_sqr()).collect();
        let expectation = probs.iter().zip(&self.cut).map(|(p, c)| p * c).sum();
        let optimal_probability = self.optimal_cuts().iter().map(|&b| probs[b]).sum();
        let most_likely = (0..probs.len())
            .max_by(|&a, &b| probs[a].total_cmp(&probs[b]))
            .expect("non-empty");
        Ok(QaoaResult {
            p: self.p,
            gammas: params[..self.p].to_vec(),
            betas: params[self.p..].to_vec(),
            expectation,
            max_cut: self.max_cut,
            approximation_ratio: (self.max_cut > 0.0).then(|| expectation / self.max_cut),
            optimal_probability,
            most_likely,
            most_likely_cut: self.cut[most_likely],
            iterations: 0,
            evaluations: 0,
            converged: true,
        })
    }

    /// Maximise `⟨C⟩` from several seeded random starts.
    pub fn optimize(&self, config: &QaoaConfig) -> Result<QaoaResult> {
        crate::optimize::check_restarts(config.restarts)?;
        let objective = NegativeCut { qaoa: self };
        let base = Rng::seed_from_u64(config.seed);
        let p = self.p;
        let runs = map_tasks(config.restarts.max(1), |r| {
            let x0: Vec<f64> = match (&config.initial, r) {
                (Some(init), 0) => init.clone(),
                _ => {
                    let mut rng = base.fork(r as u64);
                    let mut x: Vec<f64> = (0..p)
                        .map(|_| rng.uniform_range(0.0, std::f64::consts::PI))
                        .collect();
                    x.extend((0..p).map(|_| rng.uniform_range(0.0, std::f64::consts::FRAC_PI_2)));
                    x
                }
            };
            self.check(&x0)?;
            minimize(&objective, &x0, &config.method, &config.options)
        });
        let runs: Vec<_> = runs.into_iter().collect::<Result<Vec<_>>>()?;
        let evaluations = runs.iter().map(|r| r.evaluations).sum();
        let best = runs
            .into_iter()
            .min_by(|a, b| a.value.total_cmp(&b.value))
            .expect("at least one run");
        let mut result = self.analyze(&best.x)?;
        result.iterations = best.iterations;
        result.evaluations = evaluations;
        result.converged = best.converged;
        Ok(result)
    }
}

struct NegativeCut<'a> {
    qaoa: &'a MaxCutQaoa,
}

impl Objective for NegativeCut<'_> {
    fn dim(&self) -> usize {
        self.qaoa.num_parameters()
    }

    fn value(&self, x: &[f64]) -> f64 {
        -self.qaoa.expectation(x).unwrap_or(f64::NEG_INFINITY)
    }

    fn value_and_gradient(&self, x: &[f64]) -> Option<(f64, Vec<f64>)> {
        let (v, g) = self.qaoa.expectation_and_gradient(x).ok()?;
        Some((-v, g.into_iter().map(|d| -d).collect()))
    }
}

/// INTERP initialisation (Zhou et al., Phys. Rev. X 10, 021067, 2020):
/// from optimal depth-p parameters `[γ₁…γ_p, β₁…β_p]` build a depth-(p+1)
/// starting point by linear interpolation,
/// `x'_i = ((i−1)/p)·x_{i−1} + ((p−i+1)/p)·x_i`.
pub fn interpolate_parameters(params: &[f64]) -> Result<Vec<f64>> {
    if params.is_empty() || !params.len().is_multiple_of(2) {
        return Err(QsoError::invalid(
            "params",
            "expected [γ₁…γ_p, β₁…β_p] with p ≥ 1",
        ));
    }
    let p = params.len() / 2;
    let interp = |x: &[f64]| -> Vec<f64> {
        (1..=p + 1)
            .map(|i| {
                let prev = if i >= 2 { x[i - 2] } else { 0.0 };
                let cur = if i <= p { x[i - 1] } else { 0.0 };
                ((i - 1) as f64 / p as f64) * prev + ((p + 1 - i) as f64 / p as f64) * cur
            })
            .collect()
    };
    let mut out = interp(&params[..p]);
    out.extend(interp(&params[p..]));
    Ok(out)
}

/// Optimise depths `1..=p_max`, seeding each depth with the INTERP
/// extrapolation of the previous optimum (plus `config.restarts − 1`
/// random starts).
pub fn optimize_layerwise(
    graph: &Graph,
    p_max: usize,
    config: &QaoaConfig,
) -> Result<Vec<QaoaResult>> {
    let base = MaxCutQaoa::new(graph, 1)?;
    let mut results: Vec<QaoaResult> = Vec::with_capacity(p_max);
    for p in 1..=p_max {
        let qaoa = base.with_depth(p)?;
        let mut cfg = config.clone();
        if let Some(prev) = results.last() {
            let mut x = prev.gammas.clone();
            x.extend(&prev.betas);
            cfg.initial = Some(interpolate_parameters(&x)?);
        }
        results.push(qaoa.optimize(&cfg)?);
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::{ansatz, models};
    use crate::graph::{generators, metatron};

    #[test]
    fn phase_tables_match_direct_phases() {
        let mut rng = Rng::seed_from_u64(8);
        let mut edges = Vec::new();
        for u in 0..11 {
            for v in u + 1..11 {
                if rng.uniform() < 0.4 {
                    edges.push((u, v, rng.uniform_range(0.1, 2.0)));
                }
            }
        }
        let g = Graph::from_weighted_edges(11, &edges).unwrap();
        let tabled = MaxCutQaoa::new(&g, 2).unwrap();
        assert!(tabled.levels.is_some());
        let mut direct = tabled.clone();
        direct.levels = None;
        let x = [0.3, 0.9, 0.7, 0.2];
        let (a, ga) = tabled.expectation_and_gradient(&x).unwrap();
        let (b, gb) = direct.expectation_and_gradient(&x).unwrap();
        assert!((a - b).abs() < 1e-12);
        assert!(ga.iter().zip(&gb).all(|(p, q)| (p - q).abs() < 1e-12));
        let many: Vec<f64> = (0..70_000).map(|k| k as f64).collect();
        assert!(cut_levels(&many).is_none());
    }

    #[test]
    fn fast_path_matches_explicit_circuit() {
        let mut rng = Rng::seed_from_u64(4);
        let g = generators::random_regular(8, 3, &mut rng).unwrap();
        let qaoa = MaxCutQaoa::new(&g, 2).unwrap();
        let params = [0.4, 0.9, 0.3, 0.7];
        let fast = qaoa.state(&params).unwrap();
        // The explicit circuit includes the initial Hadamard layer.
        let circuit = ansatz::qaoa_maxcut_circuit(&g, 2).unwrap();
        let explicit = StateVector::from_circuit(&circuit, &params).unwrap();
        assert!((fast.fidelity(&explicit).unwrap() - 1.0).abs() < 1e-12);
        let cost = models::maxcut_cost(&g).unwrap();
        assert!(
            (qaoa.expectation(&params).unwrap() - cost.expectation(&explicit).unwrap()).abs()
                < 1e-10
        );
    }

    #[test]
    fn adjoint_gradient_matches_finite_differences() {
        let g = Graph::from_weighted_edges(
            5,
            &[
                (0, 1, 1.0),
                (1, 2, 0.5),
                (2, 3, 2.0),
                (3, 4, 1.5),
                (4, 0, 0.7),
                (1, 3, 1.2),
            ],
        )
        .unwrap();
        let qaoa = MaxCutQaoa::new(&g, 3).unwrap();
        let x = [0.31, -0.72, 1.05, 0.44, 0.13, -0.9];
        let (v, grad) = qaoa.expectation_and_gradient(&x).unwrap();
        assert!((v - qaoa.expectation(&x).unwrap()).abs() < 1e-12);
        for i in 0..x.len() {
            let mut xp = x;
            let mut xm = x;
            xp[i] += 1e-6;
            xm[i] -= 1e-6;
            let fd = (qaoa.expectation(&xp).unwrap() - qaoa.expectation(&xm).unwrap()) / 2e-6;
            assert!(
                (grad[i] - fd).abs() < 1e-7,
                "param {i}: {} vs {fd}",
                grad[i]
            );
        }
    }

    #[test]
    fn ring_of_disagrees_p1_is_three_quarters() {
        for n in [6, 8, 10] {
            let qaoa = MaxCutQaoa::new(&generators::cycle(n), 1).unwrap();
            assert_eq!(qaoa.max_cut(), n as f64);
            let r = qaoa.optimize(&QaoaConfig::default()).unwrap();
            assert!(
                (r.approximation_ratio.unwrap() - 0.75).abs() < 1e-7,
                "n={n}: {r:?}"
            );
        }
    }

    #[test]
    fn three_regular_triangle_free_p1_is_0_6925() {
        // Farhi et al.: every edge of a triangle-free 3-regular graph
        // contributes 0.6924… at p = 1. The cube graph Q3 is bipartite
        // (C_max = |E|), so the ratio equals that constant.
        let r = MaxCutQaoa::new(&generators::hypercube(3), 1)
            .unwrap()
            .optimize(&QaoaConfig::default())
            .unwrap();
        assert!(
            (r.approximation_ratio.unwrap() - 0.692450).abs() < 2e-6,
            "{r:?}"
        );
    }

    #[test]
    fn layerwise_interp_improves_monotonically() {
        let mut rng = Rng::seed_from_u64(21);
        let g = generators::random_regular(10, 3, &mut rng).unwrap();
        let cfg = QaoaConfig {
            restarts: 4,
            ..Default::default()
        };
        let results = optimize_layerwise(&g, 4, &cfg).unwrap();
        for w in results.windows(2) {
            assert!(w[1].expectation >= w[0].expectation - 1e-9);
        }
        assert!(
            results[3].approximation_ratio.unwrap() > 0.85,
            "{:?}",
            results[3]
        );
    }

    #[test]
    fn metatron_cube_maxcut() {
        // K13: the maximum cut splits 6 | 7 nodes, 42 edges.
        let qaoa = MaxCutQaoa::new(&metatron::metatron_cube(), 1).unwrap();
        assert_eq!(qaoa.max_cut(), 42.0);
        assert_eq!(qaoa.optimal_cuts().len(), 2 * 1716);
        // For K_n the uniform state already gives E = |E|/2 = 39; QAOA must
        // not do worse.
        let r = qaoa.optimize(&QaoaConfig::default()).unwrap();
        assert!(r.expectation >= 39.0 - 1e-9);
    }

    #[test]
    fn interpolation_formula() {
        // p = 1 → 2: [γ] → [γ, γ]·(…)
        let x = interpolate_parameters(&[0.4, 0.2]).unwrap();
        assert!(interpolate_parameters(&[]).is_err());
        assert!(interpolate_parameters(&[0.1, 0.2, 0.3]).is_err());
        assert_eq!(x.len(), 4);
        assert!((x[0] - 0.4).abs() < 1e-15 && (x[1] - 0.4).abs() < 1e-15);
        assert!((x[2] - 0.2).abs() < 1e-15 && (x[3] - 0.2).abs() < 1e-15);
    }

    #[test]
    fn depth_is_bounded() {
        let g = generators::cycle(4);
        assert!(MaxCutQaoa::new(&g, 0).is_err());
        assert!(MaxCutQaoa::new(&g, MAX_QAOA_DEPTH + 1).is_err());
        assert!(MaxCutQaoa::new(&g, usize::MAX).is_err());
        let q = MaxCutQaoa::new(&g, MAX_QAOA_DEPTH).unwrap();
        assert_eq!(q.num_parameters(), 2 * MAX_QAOA_DEPTH);
        assert!(q.with_depth(0).is_err());
        assert!(q.with_depth(usize::MAX / 2 + 1).is_err());
    }
}
