//! Transport to a trap on orbitals (crate-internal): the efficiency and
//! mean arrival time of [`super::SymmetricOpenWalk`] and
//! [`super::HypercubeOpenWalk`] by linear solves on the reduced generator
//! (see [`crate::walks::transport`] for the model).
//!
//! The generator `M` of the orbital coefficients is written in orthonormal
//! coordinates, `M̂ = D^{½}MD^{−½}` with `D` the orbital sizes (the
//! hypercube's generator already is), so that GMRES minimises the
//! Hilbert–Schmidt norm and the Hamiltonian part is skew-Hermitian. Then:
//!
//! 1. **Adjoint.** `M̂†â = −F̂` with `F = 2Σ_t κ_t P_t`: `â` holds the
//!    efficiency of every start, `η = Re â[o_ss]` for a fixed start `s`.
//!    With losses, `M̂†â_L = −F̂_loss` gives the population lost. A dark
//!    subspace makes `M̂` singular, but `F` has no component in the kernel
//!    of `𝓛†`, so this solve stays consistent.
//! 2. **Bright start.** Unless every node loses population,
//!    `Π_B = A + A_L` is the projector onto the bright states
//!    (`𝓛†(Π_B) = −(F + F_loss)`), so `q = Π_B e_s` is the start's bright
//!    part and `1 − η − lost` its dark weight. The forward solve starts
//!    from `|q⟩⟨q|`, which keeps it consistent; this works on signed graphs
//!    too, where a closure over orbits of points fails.
//! 3. **Forward.** `M̂x̂ = −x̂₀` gives `X`, and `ητ = ⟨A, X⟩`. A second
//!    solve `𝓛(Y) = −X` is not used: with dark states it fails silently.
//!
//! Both directions are preconditioned on the right by one banded LU of
//! `M̂ − σI` after a reverse Cuthill–McKee ordering, with
//! `σ = max(10⁻²·hopping, 10⁻⁶‖M̂‖∞)`, refactorised at every added
//! dephasing rate (an LU of another rate puts dark components into the
//! solutions). GMRES stops on the backward error
//! `‖b − M̂x‖ ≤ tol·(‖b‖ + ‖M̂‖∞‖x‖)`; one step of iterative refinement
//! follows for the forward solve, and for the adjoint solves when every
//! node loses population (`M̂` nonsingular), so that read-outs on small
//! orbitals keep their relative accuracy and those on large ones their
//! absolute accuracy. Everything is sequential
//! except the row-parallel products, so results do not depend on the thread
//! count.
//!
//! Traces propagate `[x; y]` with `y′ = x` (`OrbitalMap::doubled`), so `y`
//! is the time integral of the populations and the sinks and losses are
//! read off it.

use num_complex::Complex64 as C64;

use super::OrbitalMap;
use crate::error::{QsoError, Result};
use crate::linalg::banded::{BandedLu, reverse_cuthill_mckee};
use crate::linalg::gmres::{GmresEnd, GmresOptions, gmres};
use crate::walks::transport::{
    DephasingOptimum, MAX_TRANSPORT_FACTOR_ENTRIES, MAX_TRANSPORT_WORK, SolveReport,
    TransportOptions, TransportResult, TransportState, arrival_time, check_scan_rates, check_times,
    optimise,
};

#[cfg(test)]
mod tests;

const ZERO: C64 = C64::new(0.0, 0.0);

/// Restart length of the orbital GMRES at most.
const RESTART: usize = 100;

/// Relative breakdown test of the orbital GMRES.
const BREAKDOWN: f64 = 1e-14;

/// `σ` of the preconditioner `M̂ − σI` relative to the hopping.
const SIGMA_HOPPING: f64 = 1e-2;

/// `σ` of the preconditioner relative to `‖M̂‖∞` at least.
const SIGMA_NORM: f64 = 1e-6;

/// The bright part of the start from `Π_B` in plain coordinates: `x̂₀`
/// (orthonormal coordinates of `|q⟩⟨q|`) and `‖q‖²`.
pub(super) type BrightStart<'a> = Box<dyn Fn(&[C64]) -> (Vec<C64>, f64) + 'a>;

/// A transport problem on orbitals.
pub(super) struct OrbitalProblem<'a> {
    /// `M̂`, the generator in orthonormal coordinates.
    pub(super) m: OrbitalMap,
    /// Whether each orbital holds coherences (pairs `a ≠ b`), which an
    /// added uniform dephasing `e` decays at `e`.
    pub(super) coherence: Vec<bool>,
    /// The hopping scale of `σ`: `γ·max|w_e|`.
    pub(super) hopping: f64,
    /// `F̂`: `√|O|·2κ_O` on the diagonal orbitals of traps.
    pub(super) traps: Vec<C64>,
    /// `F̂_loss`: `√|O|·Γ_O` on the diagonal orbitals, if any `Γ > 0`.
    pub(super) losses: Option<Vec<C64>>,
    /// The orbital of `(s, s)` (of size 1).
    pub(super) start: usize,
    /// Whether every node loses population (`Γ₀ = min_j Γ_j > 0`).
    pub(super) all_lose: bool,
    /// Per entry of `TransportResult::trapped`: the orbital and the factor
    /// `f` of `trapped = f·Re x̂_O`.
    pub(super) trapped: Vec<(usize, f64)>,
    /// `(O, Γ_O √|O|)` over the diagonal orbitals with a loss.
    pub(super) lost: Vec<(usize, f64)>,
    /// `√|O|` per orbital.
    pub(super) sqrt_size: Vec<f64>,
    /// `x̂₀` and `‖q‖²` from `Π_B` (see [`BrightStart`]).
    pub(super) bright: BrightStart<'a>,
}

/// Operations and GMRES iterations of one solve so far.
struct Budget {
    spent: f64,
    iterations: usize,
}

/// `‖x‖₂`, summed sequentially.
fn norm2(x: &[C64]) -> f64 {
    x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

/// The refusal past [`MAX_TRANSPORT_WORK`].
fn work_error(iterations: usize) -> QsoError {
    QsoError::invalid(
        "graph",
        format!(
            "transport needs more than {MAX_TRANSPORT_WORK:.1e} operations after {iterations} \
             GMRES iterations; use fewer nodes or more symmetries"
        ),
    )
}

/// `M̂(e) = M̂ − e·diag(coherence)` with its preconditioner.
struct Shifted<'p, 'a> {
    problem: &'p OrbitalProblem<'a>,
    adjoint: &'p OrbitalMap,
    e: f64,
    /// `‖M̂(e)‖∞`.
    norm: f64,
    lu: BandedLu,
}

impl Shifted<'_, '_> {
    /// `y = M̂(e)x` (`adjoint`: `y = M̂(e)†x`; the shift is real).
    fn apply(&self, adjoint: bool, x: &[C64], y: &mut [C64]) {
        let m = if adjoint {
            self.adjoint
        } else {
            &self.problem.m
        };
        m.apply(x, y);
        if self.e != 0.0 {
            for ((o, xi), &c) in y.iter_mut().zip(x).zip(&self.problem.coherence) {
                if c {
                    *o -= xi * self.e;
                }
            }
        }
    }

    /// GMRES on `M̂(e)z = b` (or `M̂(e)†z = b`), named `name` in the
    /// report.
    fn solve(
        &self,
        name: &str,
        adjoint: bool,
        b: &[C64],
        options: &TransportOptions,
        budget: &mut Budget,
    ) -> Result<(Vec<C64>, SolveReport)> {
        let n = b.len();
        let (kl, ku) = self.lu.bandwidth();
        let per = 8.0 * (self.problem.m.nnz() as f64 + 2.0 * (n * (2 * kl + ku + 1)) as f64);
        let opts = GmresOptions {
            tol: options.tolerance,
            restart: n.min(RESTART),
            max_iterations: options.max_iterations,
            work: (per, MAX_TRANSPORT_WORK - budget.spent),
            scale: self.norm,
            breakdown: BREAKDOWN,
        };
        let precondition = |v: &[C64]| -> Vec<C64> {
            let mut z = v.to_vec();
            if adjoint {
                self.lu.solve_adjoint(&mut z);
            } else {
                self.lu.solve(&mut z);
            }
            z
        };
        let apply = |x: &[C64], y: &mut [C64]| self.apply(adjoint, x, y);
        let sol = gmres(n, apply, precondition, b, &opts);
        budget.spent += sol.spent;
        budget.iterations += sol.iterations;
        match sol.end {
            GmresEnd::Converged => {}
            GmresEnd::Work => return Err(work_error(budget.iterations)),
            GmresEnd::Iterations | GmresEnd::Breakdown => {
                return Err(QsoError::NoConvergence {
                    method: "GMRES (orbital transport)",
                    iterations: sol.iterations,
                    residual: sol.residual,
                });
            }
        }
        let mut x = sol.x;
        let mut iterations = sol.iterations;
        // One step of iterative refinement: the backward-error rule measures
        // the residual against ‖M̂‖‖x‖, which the largest orbitals dominate,
        // so a read-out on a small orbital (the start, the trap) can keep a
        // large relative error when it is small (K₄₀₉₆ with loss everywhere:
        // η ≈ 2·10⁻⁶ wrong by 1.4·10⁻⁷ relative; after one step 10⁻¹⁴), and
        // a read-out weighted by √|O| on a large orbital a large absolute
        // one (Q₄₀ with loss on odd weights only and a trap at weight 20:
        // that trap's share wrong by 2.3·10⁻¹⁰, τ by 6·10⁻¹¹ relative; after
        // one step 10⁻¹⁴ and 10⁻¹³). The correction solves for the residual
        // with the same rule, relative to the much smaller residual, and is
        // kept only if it converges. The forward solve is always refined:
        // with dark states `M̂` is singular, but its kernel (stationary
        // states of the dark subspace) holds no trap, loss or bright state,
        // so a kernel component of the correction reaches no read-out. The
        // adjoint solve is refined only when every node loses population
        // (then `M̂` is nonsingular): its kernel holds the conserved dark
        // projectors, which the start's entry sees, and the correction of a
        // residual made of rounding errors could add a large kernel
        // component (it moved η of K₂₀₄₈ without loss from 1/2047 to 0.24).
        let residual = |x: &[C64]| -> Vec<C64> {
            let mut r = vec![ZERO; n];
            apply(x, &mut r);
            r.iter().zip(b).map(|(ax, bi)| bi - ax).collect()
        };
        if self.problem.all_lose || !adjoint {
            let r = residual(&x);
            let opts = GmresOptions {
                work: (per, MAX_TRANSPORT_WORK - budget.spent),
                ..opts
            };
            let correction = gmres(n, apply, precondition, &r, &opts);
            budget.spent += correction.spent;
            budget.iterations += correction.iterations;
            iterations += correction.iterations;
            if correction.end == GmresEnd::Converged && correction.x.iter().all(|d| d.is_finite()) {
                for (xi, d) in x.iter_mut().zip(&correction.x) {
                    *xi += d;
                }
            }
        }
        let bnorm = norm2(b);
        let xnorm = norm2(&x);
        let backward_error =
            norm2(&residual(&x)) / (bnorm + self.norm * xnorm).max(f64::MIN_POSITIVE);
        let condition = if bnorm > 0.0 {
            self.norm * xnorm / bnorm
        } else {
            1.0
        };
        let report = SolveReport {
            name: name.to_string(),
            iterations,
            backward_error,
            condition,
        };
        Ok((x, report))
    }
}

impl<'a> OrbitalProblem<'a> {
    /// The reverse Cuthill–McKee order of the symmetrised pattern of `M̂`.
    pub(super) fn order(&self) -> Vec<u32> {
        let m = &self.m;
        let n = m.indptr.len() - 1;
        let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); n];
        for r in 0..n {
            for &c in &m.cols[m.indptr[r]..m.indptr[r + 1]] {
                if c as usize != r {
                    adjacency[r].push(c);
                    adjacency[c as usize].push(r as u32);
                }
            }
        }
        for a in &mut adjacency {
            a.sort_unstable();
            a.dedup();
        }
        reverse_cuthill_mckee(&adjacency)
    }

    /// `M̂(e)` with the banded LU of `M̂(e) − σI` (`σ` by the module
    /// docs), whose operations count against `budget`.
    fn shifted<'p>(
        &'p self,
        adjoint: &'p OrbitalMap,
        order: &[u32],
        e: f64,
        budget: &mut Budget,
    ) -> Result<Shifted<'p, 'a>> {
        let m = &self.m;
        let n = m.indptr.len() - 1;
        let shift = |r: usize| if self.coherence[r] { e } else { 0.0 };
        let norm = (0..n)
            .map(|r| {
                (m.indptr[r]..m.indptr[r + 1])
                    .map(|k| {
                        let c = m.cols[k] as usize;
                        if c == r {
                            (m.vals[k] - shift(r)).norm()
                        } else {
                            m.vals[k].norm()
                        }
                    })
                    .sum::<f64>()
            })
            .fold(0.0, f64::max);
        let sigma = (SIGMA_HOPPING * self.hopping).max(SIGMA_NORM * norm);
        let lu = BandedLu::new(
            n,
            order.to_vec(),
            |r, row| {
                for k in m.indptr[r]..m.indptr[r + 1] {
                    row.push((m.cols[k], m.vals[k]));
                }
                row.push((r as u32, C64::new(-shift(r) - sigma, 0.0)));
            },
            MAX_TRANSPORT_FACTOR_ENTRIES,
            MAX_TRANSPORT_WORK - budget.spent,
        )?;
        let (kl, ku) = lu.bandwidth();
        budget.spent += 8.0 * n as f64 * kl as f64 * (kl + ku + 1) as f64;
        Ok(Shifted {
            problem: self,
            adjoint,
            e,
            norm,
            lu,
        })
    }

    /// The transport with the uniform rate `e ≥ 0` added to every node's
    /// dephasing; without `want_tau` only the efficiency is computed (one
    /// adjoint solve).
    pub(super) fn solve(
        &self,
        adjoint: &OrbitalMap,
        order: &[u32],
        e: f64,
        want_tau: bool,
        options: &TransportOptions,
    ) -> Result<TransportResult> {
        let n = self.sqrt_size.len();
        let mut budget = Budget {
            spent: 0.0,
            iterations: 0,
        };
        let shifted = self.shifted(adjoint, order, e, &mut budget)?;
        let minus = |f: &[C64]| -> Vec<C64> { f.iter().map(|z| -z).collect() };
        let (a, report_a) =
            shifted.solve("adjoint", true, &minus(&self.traps), options, &mut budget)?;
        let eta_a = a[self.start].re;
        let mut solves = vec![report_a];
        if !want_tau {
            return Ok(self.result(eta_a, None, Vec::new(), 0.0, 0.0, 0.0, e, solves));
        }
        let a_loss = match &self.losses {
            Some(f) => {
                let (x, report) =
                    shifted.solve("adjoint loss", true, &minus(f), options, &mut budget)?;
                solves.push(report);
                Some(x)
            }
            None => None,
        };
        let lost_a = a_loss.as_ref().map_or(0.0, |x| x[self.start].re);
        let dark_raw = 1.0 - eta_a - lost_a;
        let (x0, qn) = if self.all_lose {
            let mut x0 = vec![ZERO; n];
            x0[self.start] = C64::new(1.0, 0.0);
            (x0, 1.0)
        } else if eta_a + lost_a <= options.dark_tolerance {
            // The start is dark: nothing reaches a trap or a loss.
            let error = (eta_a + lost_a).abs();
            let trapped = vec![0.0; self.trapped.len()];
            return Ok(self.result(0.0, None, trapped, 0.0, 1.0, error, e, solves));
        } else {
            let p: Vec<C64> = (0..n)
                .map(|o| {
                    let l = a_loss.as_ref().map_or(ZERO, |x| x[o]);
                    (a[o] + l) / self.sqrt_size[o]
                })
                .collect();
            (self.bright)(&p)
        };
        let (x, report_x) = shifted.solve("forward", false, &minus(&x0), options, &mut budget)?;
        solves.push(report_x);
        let trapped: Vec<f64> = self.trapped.iter().map(|&(o, f)| f * x[o].re).collect();
        let eta_x: f64 = trapped.iter().sum();
        let lost_x: f64 = self.lost.iter().map(|&(o, f)| f * x[o].re).sum();
        let eta_tau: f64 = a.iter().zip(&x).map(|(p, q)| (p.conj() * q).re).sum();
        let dark = if !self.all_lose && dark_raw > options.dark_tolerance {
            dark_raw
        } else {
            0.0
        };
        let mut error =
            (eta_a - eta_x).abs() + (lost_a - lost_x).abs() + (1.0 - eta_a - lost_a - dark).abs();
        if !self.all_lose {
            error += (qn - (eta_a + lost_a)).abs();
        }
        let tau = arrival_time(eta_a, eta_tau, error);
        Ok(self.result(eta_a, tau, trapped, lost_a, dark, error, e, solves))
    }

    #[allow(clippy::too_many_arguments)]
    fn result(
        &self,
        efficiency: f64,
        arrival_time: Option<f64>,
        trapped: Vec<f64>,
        lost: f64,
        dark: f64,
        error: f64,
        e: f64,
        solves: Vec<SolveReport>,
    ) -> TransportResult {
        let condition = solves.iter().map(|s| s.condition).fold(0.0, f64::max);
        TransportResult {
            efficiency,
            arrival_time,
            trapped,
            lost,
            dark,
            added_dephasing: e,
            dimension: self.sqrt_size.len(),
            slowest_decay: None,
            condition,
            error,
            solves,
        }
    }

    /// The transport without added dephasing.
    pub(super) fn transport(&self, options: &TransportOptions) -> Result<TransportResult> {
        let adjoint = self.m.adjoint();
        self.solve(&adjoint, &self.order(), 0.0, true, options)
    }

    /// One result per added rate (one ordering, one LU per rate).
    pub(super) fn scan(
        &self,
        rates: &[f64],
        options: &TransportOptions,
    ) -> Result<Vec<TransportResult>> {
        check_scan_rates(rates)?;
        let adjoint = self.m.adjoint();
        let order = self.order();
        rates
            .iter()
            .map(|&e| self.solve(&adjoint, &order, e, true, options))
            .collect()
    }

    /// The added rate in `[low, high]` of highest efficiency.
    pub(super) fn optimum(
        &self,
        low: f64,
        high: f64,
        options: &TransportOptions,
    ) -> Result<DephasingOptimum> {
        if !(low.is_finite() && high.is_finite() && low > 0.0 && low < high) {
            return Err(QsoError::invalid(
                "range",
                "need 0 < low < high, both finite",
            ));
        }
        let adjoint = self.m.adjoint();
        let order = self.order();
        optimise(
            low,
            high,
            |e| Ok(self.solve(&adjoint, &order, e, false, options)?.efficiency),
            |e| self.solve(&adjoint, &order, e, true, options),
        )
    }
}

/// How a trace reads the propagated `[x; y]`.
pub(super) struct TraceReads {
    /// `(O, f)`: population `f·Re x_O` per reported node (or weight).
    pub(super) populations: Vec<(usize, f64)>,
    /// `(O, f)`: sink population `f·Re y_O` per entry of `trapped`.
    pub(super) trapped: Vec<(usize, f64)>,
    /// `(O, f)`: the loss is `Σ f·Re y_O`.
    pub(super) lost: Vec<(usize, f64)>,
}

/// Propagated populations, sinks and loss at each of `times` from `x0`
/// under `generator` (by `exp_apply` on the doubled map, incrementally
/// between times).
pub(super) fn trace(
    generator: &OrbitalMap,
    x0: &[C64],
    times: &[f64],
    reads: &TraceReads,
) -> Result<Vec<TransportState>> {
    check_times(times)?;
    let doubled = generator.doubled();
    let n = x0.len();
    let mut z = vec![ZERO; 2 * n];
    z[..n].copy_from_slice(x0);
    let mut now = 0.0;
    let mut out = Vec::with_capacity(times.len());
    for &t in times {
        if t > now {
            z = doubled.exp_apply(&z, t - now)?;
            now = t;
        }
        let (x, y) = z.split_at(n);
        out.push(TransportState {
            time: t,
            populations: reads
                .populations
                .iter()
                .map(|&(o, f)| f * x[o].re)
                .collect(),
            trapped: reads.trapped.iter().map(|&(o, f)| f * y[o].re).collect(),
            lost: reads.lost.iter().map(|&(o, f)| f * y[o].re).sum(),
        });
    }
    Ok(out)
}
