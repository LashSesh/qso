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

use num_complex::{Complex32 as C32, Complex64 as C64};
use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::optimize::{Method, Objective, Options, minimize_at};
use crate::parallel::{
    CHUNK, SyncPtr, for_each_item, for_each_mut, map_collect, map_tasks, sum_c64, sum_f64,
};
use crate::quantum::kernels;
use crate::quantum::plan::{BLOCK_QUBITS, Tiles, for_each_block, y_overlap_im};
use crate::quantum::scalar::Amplitude;
use crate::quantum::{MAX_QUBITS, Precision, StateVector};
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
    cuts: Cuts,
    max_cut: f64,
    precision: Precision,
}

/// The cut weight of every bit string.
#[derive(Clone, Debug)]
enum Cuts {
    /// The distinct values and the index of each string's value, when there
    /// are at most 2¹⁶ of them (always for unweighted graphs): two bytes
    /// per string instead of eight, and a cost layer looks its phases up
    /// in a table instead of evaluating `e^{−iγc}` for every amplitude.
    Levels { values: Vec<f64>, index: Vec<u16> },
    /// Every string's value.
    Values(Vec<f64>),
}

impl Cuts {
    fn len(&self) -> usize {
        match self {
            Cuts::Levels { index, .. } => index.len(),
            Cuts::Values(v) => v.len(),
        }
    }

    #[inline]
    fn at(&self, b: usize) -> f64 {
        match self {
            Cuts::Levels { values, index } => values[index[b] as usize],
            Cuts::Values(v) => v[b],
        }
    }
}

/// Most distinct cut values for which [`Cuts::Levels`] is used.
const MAX_LEVELS: usize = 1 << 16;

/// Sorted distinct values of `a` and `b` (sorted and distinct themselves),
/// or `None` beyond [`MAX_LEVELS`].
fn merge_levels(a: Option<Vec<f64>>, b: Option<Vec<f64>>) -> Option<Vec<f64>> {
    let (a, b) = (a?, b?);
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        let next = match (a.get(i), b.get(j)) {
            (Some(x), Some(y)) if x.total_cmp(y).is_lt() => {
                i += 1;
                *x
            }
            (Some(x), Some(y)) if x.total_cmp(y).is_gt() => {
                j += 1;
                *y
            }
            (Some(x), Some(_)) => {
                i += 1;
                j += 1;
                *x
            }
            (Some(x), None) => {
                i += 1;
                *x
            }
            (None, Some(y)) => {
                j += 1;
                *y
            }
            (None, None) => unreachable!(),
        };
        out.push(next);
        if out.len() > MAX_LEVELS {
            return None;
        }
    }
    Some(out)
}

/// Distinct values of `cut` and the index of every entry, if there are at
/// most 2¹⁶ distinct values. The values are collected chunk by chunk and
/// merged, without a sorted copy of the whole table.
fn cut_levels(cut: &[f64]) -> Option<(Vec<f64>, Vec<u16>)> {
    let distinct = |c: usize| {
        let mut v = cut[c * CHUNK..((c + 1) * CHUNK).min(cut.len())].to_vec();
        v.sort_unstable_by(f64::total_cmp);
        v.dedup_by(|a, b| a.total_cmp(b).is_eq());
        (v.len() <= MAX_LEVELS).then_some(v)
    };
    let chunks = cut.len().div_ceil(CHUNK);
    #[cfg(feature = "parallel")]
    let values = {
        use rayon::prelude::*;
        (0..chunks)
            .into_par_iter()
            .map(distinct)
            .reduce(|| Some(Vec::new()), merge_levels)?
    };
    #[cfg(not(feature = "parallel"))]
    let values = (0..chunks)
        .map(distinct)
        .fold(Some(Vec::new()), merge_levels)?;
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

/// Apply `e^{−iγC}` (if `cost` is given) and then the mixer, in the frame
/// `S^{⊗n}` where it is `e^{−iβ Σ Y}` (or its inverse for negative β).
///
/// `e^{−iβX} = S†·e^{−iβY}·S` with `S = diag(1, i)`, and `S^{⊗n}` is
/// diagonal, so it commutes with the cost layers: between two layers the
/// `S` and `S†` cancel, and the whole state is `S†^{⊗n}` applied to the
/// state evolved with Y rotations from `S^{⊗n}|+⟩`. The Y rotation is a
/// real matrix, which the kernels apply with a quarter of the products of
/// the complex X rotation; ⟨C⟩ and the probabilities do not see `S†^{⊗n}`. The phases and the rotations of the qubits
/// below the block size are applied block by block in one pass over
/// memory; the higher qubits follow in tiles, one pass per group of up to
/// seven ([`Tiles`]).
fn layer<A: Amplitude>(amps: &mut [A], n: usize, cost: Option<&Phases<'_>>, beta: f64) {
    let m = mixer_matrix::<A>(beta);
    let low = BLOCK_QUBITS.min(n);
    for_each_block(amps, low, |base, block| {
        if let Some(phases) = cost {
            for (i, a) in block.iter_mut().enumerate() {
                *a *= A::from_c64(phases.at(base + i));
            }
        }
        for q in 0..low {
            kernels::apply_1q(block, q, 0, &m);
        }
    });
    for tiles in Tiles::groups(n, low) {
        let gates: Vec<(usize, [A; 4])> = tiles.qubits().iter().map(|&q| (q, m)).collect();
        tiles.apply_1q(amps, &gates);
    }
}

/// `Im ⟨λ|Y_q|φ⟩` on a block, `stride = 2^q`, over the contiguous halves
/// of each pair of runs.
fn y_overlap_1q<A: Amplitude>(lambda: &[A], phi: &[A], stride: usize) -> f64 {
    lambda
        .chunks_exact(2 * stride)
        .zip(phi.chunks_exact(2 * stride))
        .map(|(l, p)| {
            let (l0, l1) = l.split_at(stride);
            let (p0, p1) = p.split_at(stride);
            y_overlap_im(l0, l1, p0, p1)
        })
        .sum()
}

/// `e^{−iβY}`, the mixer of one qubit in the frame of [`layer`].
fn mixer_matrix<A: Amplitude>(beta: f64) -> [A; 4] {
    let (c, s) = (beta.cos(), beta.sin());
    [
        C64::new(c, 0.0),
        C64::new(-s, 0.0),
        C64::new(s, 0.0),
        C64::new(c, 0.0),
    ]
    .map(A::from_c64)
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
        let cuts = match cut_levels(&cut) {
            Some((values, index)) => Cuts::Levels { values, index },
            None => Cuts::Values(cut),
        };
        Ok(Self {
            n,
            p,
            cuts,
            max_cut,
            precision: Precision::Double,
        })
    }

    /// The same graph at a different depth (reuses the cut table).
    pub fn with_depth(&self, p: usize) -> Result<Self> {
        check_depth(p)?;
        Ok(Self { p, ..self.clone() })
    }

    /// Simulate in single precision (half the memory of the two state
    /// vectors the gradient needs) or double precision (the default).
    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self
    }

    /// The precision of the simulations.
    pub fn precision(&self) -> Precision {
        self.precision
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
        self.cuts.at(b)
    }

    /// All optimal bit strings.
    pub fn optimal_cuts(&self) -> Vec<usize> {
        let tol = 1e-9 * self.max_cut.abs().max(1.0);
        (0..self.cuts.len())
            .filter(|&b| self.max_cut - self.cuts.at(b) <= tol)
            .collect()
    }

    /// Phases `e^{i·angle·cut[b]}`.
    fn phases(&self, angle: f64) -> Phases<'_> {
        match &self.cuts {
            Cuts::Levels { values, index } => Phases::Table(
                values
                    .iter()
                    .map(|&v| C64::from_polar(1.0, angle * v))
                    .collect(),
                index,
            ),
            Cuts::Values(cut) => Phases::Direct(cut, angle),
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

    /// The state in the frame of [`layer`]: evolved from
    /// `S^{⊗n}|+⟩`, amplitude `i^{|b|}/√2ⁿ`.
    fn evolve<A: Amplitude>(&self, params: &[f64]) -> Vec<A> {
        let (gammas, betas) = params.split_at(self.p);
        let dim = self.cuts.len();
        let r = 1.0 / (dim as f64).sqrt();
        let powers = [
            C64::new(r, 0.0),
            C64::new(0.0, r),
            C64::new(-r, 0.0),
            C64::new(0.0, -r),
        ]
        .map(A::from_c64);
        let mut amps = map_collect(dim, |b| powers[(b.count_ones() % 4) as usize]);
        for l in 0..self.p {
            layer(&mut amps, self.n, Some(&self.phases(-gammas[l])), betas[l]);
        }
        amps
    }

    /// QAOA state for `params`.
    /// (In single precision, the amplitudes widened to double precision.)
    pub fn state(&self, params: &[f64]) -> Result<StateVector> {
        self.check(params)?;
        let mut amps: Vec<C64> = match self.precision {
            Precision::Double => self.evolve(params),
            _ => {
                let amps: Vec<C32> = self.evolve(params);
                map_collect(amps.len(), |b| amps[b].to_c64())
            }
        };
        // Out of the frame: S†^{⊗n}, the phase (−i)^{|b|} (exact).
        for_each_mut(&mut amps, |b, a| {
            *a = match b.count_ones() % 4 {
                0 => *a,
                1 => C64::new(a.im, -a.re),
                2 => -*a,
                _ => C64::new(-a.im, a.re),
            }
        });
        StateVector::from_amplitudes(amps)
    }

    /// Probabilities of the bit strings (in single precision divided by
    /// their sum, which rounding moves away from one).
    fn probabilities(&self, params: &[f64]) -> Vec<f64> {
        match self.precision {
            Precision::Double => {
                let amps: Vec<C64> = self.evolve(params);
                map_collect(amps.len(), |b| amps[b].norm_sqr())
            }
            _ => {
                let amps: Vec<C32> = self.evolve(params);
                let norm2 = sum_f64(amps.len(), |b| amps[b].norm_sqr_f64());
                map_collect(amps.len(), |b| amps[b].norm_sqr_f64() / norm2)
            }
        }
    }

    /// Expected cut weight `⟨C⟩`.
    pub fn expectation(&self, params: &[f64]) -> Result<f64> {
        self.check(params)?;
        match self.precision {
            Precision::Double => {
                let amps: Vec<C64> = self.evolve(params);
                Ok(sum_f64(amps.len(), |b| {
                    amps[b].norm_sqr() * self.cuts.at(b)
                }))
            }
            _ => {
                let amps: Vec<C32> = self.evolve(params);
                let norm2 = sum_f64(amps.len(), |b| amps[b].norm_sqr_f64());
                Ok(sum_f64(amps.len(), |b| amps[b].norm_sqr_f64() * self.cuts.at(b)) / norm2)
            }
        }
    }

    /// `⟨C⟩` and its exact gradient.
    pub fn expectation_and_gradient(&self, params: &[f64]) -> Result<(f64, Vec<f64>)> {
        self.check(params)?;
        match self.precision {
            Precision::Double => Ok(self.adjoint::<C64>(params, false)),
            _ => Ok(self.adjoint::<C32>(params, true)),
        }
    }

    /// The adjoint pass specialised to QAOA, at the amplitudes' precision
    /// (sums in double precision); with `normalise`, value and gradient
    /// divided by the state's norm², which single-precision rounding moves
    /// away from one.
    fn adjoint<A: Amplitude>(&self, params: &[f64], normalise: bool) -> (f64, Vec<f64>) {
        let (gammas, betas) = params.split_at(self.p);
        let mut phi: Vec<A> = self.evolve(params);
        let cuts = &self.cuts;
        let norm2 = sum_f64(phi.len(), |b| phi[b].norm_sqr_f64());
        let mut lambda: Vec<A> =
            map_collect(phi.len(), |b| A::from_c64(phi[b].to_c64() * cuts.at(b)));
        let value = sum_c64(phi.len(), |b| phi[b].to_c64().conj() * lambda[b].to_c64()).re;
        let mut grad = vec![0.0; 2 * self.p];
        let n = self.n;
        let low = BLOCK_QUBITS.min(n);
        let block = 1usize << low;
        let blocks = phi.len() / block;
        let mut parts = vec![(0.0, 0.0); blocks];
        for l in (0..self.p).rev() {
            // Per layer two kinds of pass over φ and λ together. ∂/∂β =
            // 2·Im⟨λ|B|φ⟩ with B = Σ_q Y_q (the mixer's generator in the
            // frame of `layer`), which commutes with every rotation of the
            // mixer: each part of the overlap is taken in the pass that
            // undoes those rotations. The qubits above the block first,
            // tile by tile:
            let m = mixer_matrix::<A>(-betas[l]);
            let mut b_phi = 0.0;
            for tiles in Tiles::groups(n, low) {
                let gates: Vec<(usize, [A; 4])> = tiles.qubits().iter().map(|&q| (q, m)).collect();
                b_phi += tiles.y_overlap_then_apply(&mut lambda, &mut phi, &gates);
            }
            // Then block by block: the rest of the overlap, the low
            // rotations undone, ∂/∂γ = 2·Im⟨λ|C|φ⟩ and the cost phases
            // undone.
            let phases = self.phases(gammas[l]);
            let (pl, pp, pr) = (
                SyncPtr::new(&mut lambda),
                SyncPtr::new(&mut phi),
                SyncPtr::new(&mut parts),
            );
            for_each_item(blocks, phi.len(), 1, |k| {
                let base = k * block;
                // SAFETY: block k of both vectors and slot k of `parts`
                // belong to this task alone.
                let (lam, ph, slot) = unsafe {
                    (
                        std::slice::from_raw_parts_mut(pl.get().add(base), block),
                        std::slice::from_raw_parts_mut(pp.get().add(base), block),
                        &mut *pr.get().add(k),
                    )
                };
                let mut overlap = 0.0;
                for q in 0..low {
                    overlap += y_overlap_1q(lam, ph, 1 << q);
                }
                for q in 0..low {
                    kernels::apply_1q(lam, q, 0, &m);
                    kernels::apply_1q(ph, q, 0, &m);
                }
                // Im(conj(λ)·φ)·C: only the imaginary part is needed.
                let mut cost = 0.0;
                for i in 0..block {
                    let (a, b) = (lam[i].to_c64(), ph[i].to_c64());
                    cost += (a.re * b.im - a.im * b.re) * cuts.at(base + i);
                    let z = A::from_c64(phases.at(base + i));
                    lam[i] *= z;
                    ph[i] *= z;
                }
                *slot = (overlap, cost);
            });
            let (overlap, c_phi) = parts
                .iter()
                .fold((b_phi, 0.0), |(a, b), (x, y)| (a + x, b + y));
            grad[self.p + l] = 2.0 * overlap;
            grad[l] = 2.0 * c_phi;
        }
        if !normalise {
            return (value, grad);
        }
        (value / norm2, grad.into_iter().map(|g| g / norm2).collect())
    }

    /// Full statistics of the state at `params`.
    pub fn analyze(&self, params: &[f64]) -> Result<QaoaResult> {
        self.check(params)?;
        let probs = self.probabilities(params);
        let expectation = probs
            .iter()
            .enumerate()
            .map(|(b, p)| p * self.cuts.at(b))
            .sum();
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
            most_likely_cut: self.cuts.at(most_likely),
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
            minimize_at(
                &objective,
                &x0,
                &config.method,
                &config.options,
                self.precision,
            )
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
    optimize_layerwise_at(graph, p_max, config, Precision::Double)
}

/// [`optimize_layerwise`] at the given precision.
pub fn optimize_layerwise_at(
    graph: &Graph,
    p_max: usize,
    config: &QaoaConfig,
    precision: Precision,
) -> Result<Vec<QaoaResult>> {
    let base = MaxCutQaoa::new(graph, 1)?.with_precision(precision);
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
        assert!(matches!(tabled.cuts, Cuts::Levels { .. }));
        let mut direct = tabled.clone();
        direct.cuts = Cuts::Values((0..1 << 11).map(|b| tabled.cuts.at(b)).collect());
        let x = [0.3, 0.9, 0.7, 0.2];
        let (a, ga) = tabled.expectation_and_gradient(&x).unwrap();
        let (b, gb) = direct.expectation_and_gradient(&x).unwrap();
        assert!((a - b).abs() < 1e-12);
        assert!(ga.iter().zip(&gb).all(|(p, q)| (p - q).abs() < 1e-12));
        let many: Vec<f64> = (0..70_000).map(|k| k as f64).collect();
        assert!(cut_levels(&many).is_none());
    }

    /// Single precision: energy, gradient, state and statistics against
    /// double precision, and an optimisation that reaches the same optimum.
    #[test]
    fn single_precision_matches_double_precision() {
        let mut rng = Rng::seed_from_u64(1271);
        let g = generators::random_regular(16, 3, &mut rng).unwrap();
        let double = MaxCutQaoa::new(&g, 3).unwrap();
        let single = double.clone().with_precision(Precision::Single);
        assert_eq!(single.precision(), Precision::Single);
        let x = [0.4, 0.7, 0.2, 0.6, 0.3, 0.1];
        let (a, ga) = double.expectation_and_gradient(&x).unwrap();
        let (b, gb) = single.expectation_and_gradient(&x).unwrap();
        assert!((a - b).abs() < 1e-5 && (single.expectation(&x).unwrap() - a).abs() < 1e-5);
        for (p, q) in ga.iter().zip(&gb) {
            assert!((p - q).abs() < 1e-4, "{p} vs {q}");
        }
        let fidelity = double
            .state(&x)
            .unwrap()
            .fidelity(&single.state(&x).unwrap())
            .unwrap();
        assert!(1.0 - fidelity < 1e-10, "{fidelity}");
        let (ra, rb) = (double.analyze(&x).unwrap(), single.analyze(&x).unwrap());
        assert!((ra.expectation - rb.expectation).abs() < 1e-5);
        assert!((ra.optimal_probability - rb.optimal_probability).abs() < 1e-6);
        let small = generators::random_regular(10, 3, &mut rng).unwrap();
        let cfg = QaoaConfig::default();
        let (oa, ob) = (
            MaxCutQaoa::new(&small, 2).unwrap().optimize(&cfg).unwrap(),
            MaxCutQaoa::new(&small, 2)
                .unwrap()
                .with_precision(Precision::Single)
                .optimize(&cfg)
                .unwrap(),
        );
        assert!(
            (oa.expectation - ob.expectation).abs() < 1e-5,
            "{oa:?} {ob:?}"
        );
        let la = optimize_layerwise_at(&small, 2, &cfg, Precision::Single).unwrap();
        assert!((la[1].expectation - oa.expectation).abs() < 1e-4);
    }

    /// Single precision stops where its gradient reaches its rounding,
    /// and its line searches treat energies within rounding as equal,
    /// instead of following noise (before: twice the evaluations, and not
    /// converged).
    #[test]
    fn single_precision_optimisation_converges_without_extra_evaluations() {
        let g = generators::random_regular(12, 3, &mut Rng::seed_from_u64(1)).unwrap();
        let cfg = QaoaConfig {
            restarts: 2,
            ..Default::default()
        };
        let double = MaxCutQaoa::new(&g, 4).unwrap().optimize(&cfg).unwrap();
        let single = MaxCutQaoa::new(&g, 4)
            .unwrap()
            .with_precision(Precision::Single)
            .optimize(&cfg)
            .unwrap();
        assert!(single.converged, "{single:?}");
        assert!(
            single.evaluations as f64 <= 1.5 * double.evaluations as f64,
            "{} vs {}",
            single.evaluations,
            double.evaluations
        );
        assert!((single.expectation - double.expectation).abs() < 1e-6 * double.max_cut);
    }

    /// Above the block size (qubits in tiles, the mixer in the Y frame,
    /// the fused backward pass): the state equals the explicit circuit's
    /// and the gradient central differences, in both precisions.
    #[test]
    fn fast_path_matches_explicit_circuit_above_the_block() {
        // Even, for a 3-regular graph (6 qubits under Miri).
        let n = (BLOCK_QUBITS + 4) & !1;
        let g = generators::random_regular(n, 3, &mut Rng::seed_from_u64(28)).unwrap();
        let qaoa = MaxCutQaoa::new(&g, 2).unwrap();
        let params = [0.4, 0.9, 0.3, 0.7];
        let circuit = ansatz::qaoa_maxcut_circuit(&g, 2).unwrap();
        let explicit = StateVector::from_circuit(&circuit, &params).unwrap();
        let fast = qaoa.state(&params).unwrap();
        // Equal up to the global phase of the circuit's RZZ gates.
        assert!(1.0 - fast.fidelity(&explicit).unwrap() < 1e-12);
        let (value, grad) = qaoa.expectation_and_gradient(&params).unwrap();
        let cost = models::maxcut_cost(&g).unwrap();
        assert!((value - cost.expectation(&explicit).unwrap()).abs() < 1e-9);
        let h = 1e-5;
        for k in 0..4 {
            let mut x = params;
            x[k] += h;
            let up = qaoa.expectation(&x).unwrap();
            x[k] -= 2.0 * h;
            let down = qaoa.expectation(&x).unwrap();
            let fd = (up - down) / (2.0 * h);
            assert!(
                (grad[k] - fd).abs() < 1e-6 * value,
                "{k}: {} vs {fd}",
                grad[k]
            );
        }
        let single = qaoa.clone().with_precision(Precision::Single);
        let (v32, g32) = single.expectation_and_gradient(&params).unwrap();
        assert!((v32 - value).abs() < 1e-5 * value);
        for (a, b) in grad.iter().zip(&g32) {
            assert!((a - b).abs() < 1e-4 * value, "{a} vs {b}");
        }
        let f = single.state(&params).unwrap().fidelity(&explicit).unwrap();
        assert!(1.0 - f < 1e-9, "{f}");
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
