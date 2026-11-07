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
//!    too, where a closure over orbits of points fails. When every node
//!    loses population and the smallest loss `Γ₀` is below
//!    `10⁻⁶‖M̂‖∞`, `Γ₀` is a uniform shift (as on the dense path): `Π_B` comes from two more adjoint solves on
//!    `M̂ + Γ₀I` (seeds `F` and the losses above `Γ₀`), and the dark weight
//!    `1 − ‖q‖²`, which only `Γ₀` removes, counts as lost. Starting from
//!    `e_s` instead would put dark components of size `1/Γ₀` into `X`,
//!    which multiply the rounding errors of `A` in `⟨A, X⟩`. A larger `Γ₀`
//!    keeps the start `e_s`.
//! 3. **Forward.** `M̂x̂ = −x̂₀` gives `X`, and `ητ = ⟨A, X⟩`. A second
//!    solve `𝓛(Y) = −X` is not used: with dark states it fails silently.
//!    Instead, a goal-oriented check solves `M̂†b̂ = −â`, which gives
//!    `ητ = ⟨b̂, x̂₀⟩` independently; `τ` is reported only when the two
//!    agree to [`TAU_AGREEMENT`] relative.
//! 4. **Checks.** A dark weight above `dark_tolerance` cannot be told
//!    from a bright state decaying below rounding once
//!    `condition·ε_mach > 10³·dark_tolerance`, nor when it is below
//!    `10³·condition·ε_mach` ([`DARK_CONDITION`]); it is then added to
//!    `error` and `τ` is withheld.
//!
//! Both directions are preconditioned on the right by one banded LU of
//! `M̂ − σI` after a reverse Cuthill–McKee ordering, with
//! `σ = max(10⁻²·hopping, 10⁻⁶‖M̂‖∞)`, refactorised at every added
//! dephasing rate (an LU of another rate puts dark components into the
//! solutions). GMRES stops on the backward error
//! `‖b − M̂x‖ ≤ tol·(‖b‖ + ‖M̂‖∞‖x‖)`; one step of iterative refinement
//! follows for the forward solve, and for the adjoint solves when every
//! node loses population and `Γ₀` is not a uniform shift (`M̂` well
//! nonsingular), so that read-outs on small
//! orbitals keep their relative accuracy and those on large ones their
//! absolute accuracy. The correction may take at most twice the solve's
//! iterations plus [`REFINE_SLACK`] and is kept only if it lowers the
//! residual. Everything is sequential
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
    Budget, DephasingOptimum, MAX_TRANSPORT_FACTOR_ENTRIES, MAX_TRANSPORT_WORK, SolveReport,
    TAU_AGREEMENT, TransportOptions, TransportResult, TransportState, arrival_time,
    check_scan_rates, check_times, optimise,
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

/// The correction of the refinement may take at most twice the solve's
/// iterations plus this many: Q₃₀ with loss 0.05 everywhere needs a few
/// more than the solve, coherent hypercubes needed about 100 more for no
/// gain.
const REFINE_SLACK: usize = 10;

/// `Γ₀` below this times `‖M̂‖∞` is a uniform shift (step 2 of the module
/// docs).
const UNIFORM_SHIFT: f64 = 1e-6;

/// A dark weight above `dark_tolerance` is doubtful (step 4 of the module
/// docs) once `condition·ε_mach` exceeds `dark_tolerance` times this, or
/// when it is below this times `condition·ε_mach` (K₁₆ with loss 10⁻⁶ on
/// 14 nodes: a dark weight of 2.9·10⁻⁸ at condition 3.7·10⁸ was a missed
/// slow mode, and τ came out 46 % short). For the first rule,
/// with genuine dark states `K_N` measured a condition of 3.9·10⁴, 5.7·10⁵
/// and 2.2·10⁶ at N = 256, 1024 and 2048 (growing as N², about 4·10⁷ at
/// 8192), while the slow bright states of `C₃₂`, `C₄₈` and `C₆₄` with site
/// energies `0.7·cos(2πu/N)` show 8.6·10⁹, 1.1·10¹¹ and 5.8·10⁹.
const DARK_CONDITION: f64 = 1e3;

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
    /// `Γ₀ = min_j Γ_j`.
    pub(super) gamma0: f64,
    /// `√|O|·(Γ_O − Γ₀)` on the diagonal orbitals, if any is positive and
    /// every node loses population.
    pub(super) excess: Option<Vec<C64>>,
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
    /// `y = (M̂(e) + lift·I)x` (`adjoint`: `y = (M̂(e) + lift·I)†x`; the
    /// shifts are real).
    fn apply(&self, adjoint: bool, lift: f64, x: &[C64], y: &mut [C64]) {
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
        if lift != 0.0 {
            for (o, xi) in y.iter_mut().zip(x) {
                *o += xi * lift;
            }
        }
    }

    /// GMRES on `(M̂(e) + lift·I)z = b` (or its adjoint), named `name` in
    /// the report, refined once if `refine`. The preconditioner
    /// `M̂(e) − σI` is a shift of the operator for any `lift`, so it keeps
    /// the operator's invariant subspaces.
    #[allow(clippy::too_many_arguments)]
    fn solve(
        &self,
        name: &str,
        adjoint: bool,
        lift: f64,
        refine: bool,
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
        let apply = |x: &[C64], y: &mut [C64]| self.apply(adjoint, lift, x, y);
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
        // with the same rule, relative to the much smaller residual, in at
        // most 2k + REFINE_SLACK iterations after a solve of k (coherent
        // hypercubes needed about 100 more for no gain), and is kept only if
        // it converges and lowers the residual. The forward solve is always refined:
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
        if refine {
            let r = residual(&x);
            let opts = GmresOptions {
                work: (per, MAX_TRANSPORT_WORK - budget.spent),
                max_iterations: (2 * sol.iterations + REFINE_SLACK).min(options.max_iterations),
                ..opts
            };
            let correction = gmres(n, apply, precondition, &r, &opts);
            budget.spent += correction.spent;
            budget.iterations += correction.iterations;
            iterations += correction.iterations;
            if correction.end == GmresEnd::Converged && correction.x.iter().all(|d| d.is_finite()) {
                let refined: Vec<C64> = x.iter().zip(&correction.x).map(|(xi, d)| xi + d).collect();
                if norm2(&residual(&refined)) < norm2(&r) {
                    x = refined;
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
        budget: &mut Budget,
    ) -> Result<TransportResult> {
        let n = self.sqrt_size.len();
        let shifted = self.shifted(adjoint, order, e, budget)?;
        let minus = |f: &[C64]| -> Vec<C64> { f.iter().map(|z| -z).collect() };
        let lose = self.all_lose;
        // Step 2 of the module docs: a small `Γ₀` is a uniform shift. A
        // larger one keeps the start itself, since `M̂ + Γ₀I` can be far
        // worse conditioned than `M̂` (Q₂₀ with loss 0.05 and dephasing 0.1:
        // η's error 3·10⁻¹¹ instead of 2·10⁻¹⁵).
        let route = lose && self.gamma0 < UNIFORM_SHIFT * shifted.norm;
        // `M̂` is nonsingular when every node loses population, and its
        // adjoint solves are refined unless `Γ₀` is so small that `M̂` is
        // nearly singular (K₁₆ at Γ₀ = 10⁻¹²: the refinement moved η by
        // 4·10⁻⁹).
        let refine_adjoint = lose && !route;
        let (a, report_a) = shifted.solve(
            "adjoint",
            true,
            0.0,
            refine_adjoint,
            &minus(&self.traps),
            options,
            budget,
        )?;
        let eta_a = a[self.start].re;
        let mut solves = vec![report_a];
        if !want_tau {
            return Ok(self.result(eta_a, None, Vec::new(), 0.0, 0.0, 0.0, e, solves));
        }
        // On the route of step 2 the loss is read from the forward solve and the
        // bright projection: the adjoint loss solve on the nonsingular but
        // ill-conditioned `M̂` lost 7·10⁻¹⁰ on K₁₆ at Γ₀ = 10⁻¹².
        let a_loss = match &self.losses {
            Some(f) if !route => {
                let (x, report) = shifted.solve(
                    "adjoint loss",
                    true,
                    0.0,
                    refine_adjoint,
                    &minus(f),
                    options,
                    budget,
                )?;
                solves.push(report);
                Some(x)
            }
            _ => None,
        };
        let lost_a = a_loss.as_ref().map_or(0.0, |x| x[self.start].re);
        let dark_raw = 1.0 - eta_a - lost_a;
        // `Π_B` (step 2 of the module docs): on `M̂ + Γ₀I` when every node
        // loses population, which is singular but consistent.
        let lifted = if route {
            let g = self.gamma0;
            let (pa, report) = shifted.solve(
                "adjoint without Γ₀",
                true,
                g,
                false,
                &minus(&self.traps),
                options,
                budget,
            )?;
            solves.push(report);
            let pl = match &self.excess {
                Some(f) => {
                    let (x, report) = shifted.solve(
                        "adjoint loss without Γ₀",
                        true,
                        g,
                        false,
                        &minus(f),
                        options,
                        budget,
                    )?;
                    solves.push(report);
                    Some(x)
                }
                None => None,
            };
            Some((pa, pl))
        } else {
            None
        };
        let (pa, pl) = match &lifted {
            Some((pa, pl)) => (pa, pl.as_ref()),
            None => (&a, a_loss.as_ref()),
        };
        let bright_weight = pa[self.start].re + pl.map_or(0.0, |x| x[self.start].re);
        if (!lose || route) && bright_weight <= options.dark_tolerance {
            // The start is dark: nothing reaches a trap or a lossier node;
            // with loss everywhere, `Γ₀` removes it.
            let trapped = vec![0.0; self.trapped.len()];
            let (lost, dark, error) = if lose {
                (1.0, 0.0, eta_a.abs() + bright_weight.abs())
            } else {
                (0.0, 1.0, bright_weight.abs())
            };
            return Ok(self.result(0.0, None, trapped, lost, dark, error, e, solves));
        }
        let (x0, qn) = if lose && !route {
            let mut x0 = vec![ZERO; n];
            x0[self.start] = C64::new(1.0, 0.0);
            (x0, 1.0)
        } else {
            let p: Vec<C64> = (0..n)
                .map(|o| {
                    let l = pl.map_or(ZERO, |x| x[o]);
                    (pa[o] + l) / self.sqrt_size[o]
                })
                .collect();
            (self.bright)(&p)
        };
        let (x, report_x) =
            shifted.solve("forward", false, 0.0, true, &minus(&x0), options, budget)?;
        solves.push(report_x);
        let trapped: Vec<f64> = self.trapped.iter().map(|&(o, f)| f * x[o].re).collect();
        let eta_x: f64 = trapped.iter().sum();
        let lost_x: f64 = self.lost.iter().map(|&(o, f)| f * x[o].re).sum();
        let eta_tau: f64 = a.iter().zip(&x).map(|(p, q)| (p.conj() * q).re).sum();
        // With loss everywhere the dark weight goes to the uniform loss and
        // stays outside the forward solve's read-outs.
        let (lost, dark) = if route {
            (lost_x + (1.0 - qn), 0.0)
        } else if lose {
            (lost_a, 0.0)
        } else if dark_raw > options.dark_tolerance {
            (lost_a, dark_raw)
        } else {
            (lost_a, 0.0)
        };
        let mut error = (eta_a - eta_x).abs() + (1.0 - eta_a - lost - dark).abs();
        if !lose || route {
            error += (qn - bright_weight).abs();
        }
        if !route {
            error += (lost_a - lost_x).abs();
        }
        // Step 4 of the module docs: a dark weight that may be a bright
        // state decaying below rounding is an error of `η` as well.
        let condition = solves.iter().map(|s| s.condition).fold(0.0, f64::max);
        let rounding = condition * f64::EPSILON;
        let doubtful = dark > 0.0
            && (rounding > DARK_CONDITION * options.dark_tolerance
                || dark < DARK_CONDITION * rounding);
        if doubtful {
            error += dark;
        }
        let mut tau = arrival_time(eta_a, eta_tau, error).filter(|_| !doubtful);
        if tau.is_some() {
            // The goal-oriented check of step 3.
            // A check that does not converge withholds `τ`.
            let time = shifted.solve(
                "adjoint time",
                true,
                0.0,
                refine_adjoint,
                &minus(&a),
                options,
                budget,
            );
            match time {
                Ok((b, report_b)) => {
                    solves.push(report_b);
                    let check: f64 = b.iter().zip(&x0).map(|(p, q)| (p.conj() * q).re).sum();
                    // NaN disagrees too.
                    let agree = (check - eta_tau).abs() <= TAU_AGREEMENT * eta_tau.abs();
                    if !agree {
                        tau = None;
                    }
                }
                Err(QsoError::NoConvergence { .. }) => tau = None,
                Err(e) => return Err(e),
            }
        }
        Ok(self.result(eta_a, tau, trapped, lost, dark, error, e, solves))
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
        let mut budget = Budget::default();
        self.solve(&adjoint, &self.order(), 0.0, true, options, &mut budget)
    }

    /// One result per added rate (one ordering, one LU per rate, one work
    /// limit for all).
    pub(super) fn scan(
        &self,
        rates: &[f64],
        options: &TransportOptions,
    ) -> Result<Vec<TransportResult>> {
        check_scan_rates(rates)?;
        let adjoint = self.m.adjoint();
        let order = self.order();
        let mut budget = Budget::default();
        rates
            .iter()
            .map(|&e| self.solve(&adjoint, &order, e, true, options, &mut budget))
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
        // One work limit for the whole optimisation.
        let budget = std::cell::RefCell::new(Budget::default());
        optimise(
            low,
            high,
            |e| {
                let mut b = budget.borrow_mut();
                Ok(self
                    .solve(&adjoint, &order, e, false, options, &mut b)?
                    .efficiency)
            },
            |e| self.solve(&adjoint, &order, e, true, options, &mut budget.borrow_mut()),
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
