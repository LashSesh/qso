//! The operator of a radical pair at one field, block by block of the
//! system's labels, and its solves: Sylvester equations on the Schur form,
//! GMRES preconditioned by them, and time evolution (see the parent
//! module's docs for the equations).

use std::collections::BTreeMap;

use num_complex::Complex64 as C64;

use super::{PairYields, RadicalPair, Recombination};
use crate::error::{QsoError, Result};
use crate::linalg::CMatrix;
use crate::linalg::dense::{cmul, expm, expm_hermitian, one_norm};
use crate::linalg::gmres::{GmresEnd, GmresOptions, gmres};
use crate::linalg::sylvester::SchurForm;
use crate::parallel::map_tasks;
use crate::spin::{
    MAX_LIOUVILLE_BLOCK, MAX_RELAXATION_MEMORY, MAX_RELAXATION_WORK, SpinSystem, ZERO,
};

const I: C64 = C64::new(0.0, 1.0);

/// Entries between blocks of a start, relative to its largest, that count
/// as zero.
const BETWEEN: f64 = 1e-14;

/// A gap of the Sylvester operator at most this times `‖A_b‖₁` is a state
/// that never decays.
const DARK: f64 = 1e-14;

/// Basis vectors of a GMRES cycle.
const RESTART: usize = 30;

/// Most GMRES iterations.
const MAX_ITERATIONS: usize = 3000;

/// Estimated operations of a Schur form and of a solve on it, per `n³`.
const SCHUR_WORK: f64 = 200.0;
const SYLVESTER_WORK: f64 = 250.0;

/// Estimated operations of one GMRES iteration per `Σ n_b³` (the operator
/// and the preconditioner).
const ITERATION_WORK: f64 = 72.0;

/// A residual `‖𝓛X + ρ₀‖_F/‖ρ₀‖_F` of a result that is accepted as it
/// stands. A backward error of 10⁻¹³ alone also accepts an `X` grown
/// without bound, which is what GMRES returns on a singular operator (a
/// state that never decays under relaxation), and rates beyond the
/// arithmetic leave `X = 0`; a true solution leaves a residual near
/// 10⁻¹⁶ times `‖𝓛‖‖X‖/‖ρ₀‖`.
const RESIDUAL: f64 = 1e-6;

/// A larger residual is accepted with a backward error of at most
/// [`BACKWARD`] and `‖X‖_F` within [`GROWTH`] times `‖ρ₀‖_F/g`, `g` the
/// slowest decay of `A` (twice it, with `k_E`): without jumps
/// `‖X‖_F ≤ ‖ρ₀‖_F (1 + 2 ln κ)/g` for eigenvectors of `A` of condition
/// `κ`. In the laboratory frame at high field `‖𝓛‖ ≈ |γ_e|B` is 10⁹ to
/// 10¹¹ times slow rates, and the residual of an accurate `X` 10⁻⁶ to
/// 10⁻⁴.
const BACKWARD: f64 = 1e-12;
const GROWTH: f64 = 1e3;

/// Refinements of a Sylvester solve, and the backward error that ends them.
const REFINEMENTS: usize = 2;
const REFINED: f64 = 1e3 * f64::EPSILON;

/// Settings of a solve that tests change.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Options {
    /// Take the GMRES path even where Sylvester solves would do.
    pub force_gmres: bool,
    /// Solve in one block of all states.
    pub whole: bool,
}

/// A jump operator embedded in the whole space: `L|s⟩ = κ(s)|τ(s)⟩`.
pub(super) struct Jump {
    pub tau: Vec<usize>,
    pub kappa: Vec<C64>,
}

/// The pair's operator at one field.
pub(super) struct Model {
    /// The states of each block.
    pub blocks: Vec<Vec<usize>>,
    /// The block of each state and its place there.
    pub pos: Vec<(usize, usize)>,
    /// Per block, the place of `π(s)` of each of its states.
    pub swap: Vec<Vec<usize>>,
    /// `A_b = H_b − μ_b − (i/2)(K_b + Γ_b)`.
    pub a: Vec<CMatrix>,
    /// `μ_b`, the mean of the diagonal of `H_b`.
    pub mu: Vec<f64>,
    /// `‖A_b‖₁`.
    pub norms: Vec<f64>,
    /// `σ = i(k_E + c/2)`.
    pub sigma: C64,
    /// The coefficient `c` of the term of Jones, Maeda and Hore (0 for
    /// Haberkorn's).
    pub c: f64,
    pub jumps: Vec<Jump>,
    /// `‖𝓛‖_est = 2 max_b ‖A_b‖₁ + |σ| + c/2 + Σ_j max_s |κ_j(s)|²`.
    pub norm: f64,
}

/// `L` of spin `i` (local matrix `a`, at most one entry per column)
/// embedded.
fn embed_jump(sys: &SpinSystem, i: usize, a: &CMatrix) -> Jump {
    let d = sys.dimension();
    let stride = sys.stride(i);
    let levels = a.nrows();
    let mut tau = Vec::with_capacity(d);
    let mut kappa = Vec::with_capacity(d);
    for s in 0..d {
        let k = (s / stride) % levels;
        debug_assert!(
            (0..levels).filter(|&r| a[(r, k)] != ZERO).count() <= 1,
            "a jump of I₊, I₋ or I_z"
        );
        match (0..levels).find(|&r| a[(r, k)] != ZERO) {
            Some(r) => {
                tau.push(s + r * stride - k * stride);
                kappa.push(a[(r, k)]);
            }
            None => {
                tau.push(s);
                kappa.push(ZERO);
            }
        }
    }
    Jump { tau, kappa }
}

/// `A X − X A† − σX`.
fn sylvester_apply(a: &CMatrix, x: &CMatrix, sigma: C64) -> CMatrix {
    cmul(a, x) - cmul(x, &a.adjoint()) - x * sigma
}

/// `(M + M†)/2`.
fn hermitian_part(m: &CMatrix) -> CMatrix {
    (m + m.adjoint()) * C64::new(0.5, 0.0)
}

fn fro(m: &[C64]) -> f64 {
    m.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

/// The refusal of a start that never decays.
fn never_decays() -> QsoError {
    QsoError::invalid(
        "rates",
        "states that the start reaches (sharing its labels, or through relaxation) never \
         recombine or escape (k_E = 0 and k_S or k_T = 0), or decay more slowly than 10⁻¹⁴ of \
         the largest frequency (slow rates against Larmor frequencies in the laboratory frame); \
         give an escape rate, or use larger rates or the rotating frame",
    )
}

/// The refusal of a result whose residual is too large: `ratio` is
/// `‖𝓛‖_est` over the slowest decay of `A`, `bounded` whether `‖X‖` is
/// within its bound.
fn singular(residual: f64, backward: f64, ratio: f64, bounded: bool, ke: f64) -> QsoError {
    let why = if bounded {
        format!(
            "the solve is not accurate in double precision (residual {residual:.1e} of the \
             start, backward error {backward:.1e}; the operator's largest frequency is \
             {ratio:.1e} times its slowest decay, as Larmor frequencies against slow rates in \
             the laboratory frame at high field): use the rotating frame or larger rates, \
             within the range of the arithmetic"
        )
    } else if ke > 0.0 {
        format!(
            "the pair's equations are singular to rounding (residual {residual:.1e} of the \
             start, the solution beyond its bound): the operator's largest frequency is \
             {ratio:.1e} times its slowest decay; use the rotating frame or larger rates"
        )
    } else {
        format!(
            "the pair's equations are singular to rounding (residual {residual:.1e} of the \
             start, the solution beyond its bound): some state never recombines, escapes or \
             relaxes, or the operator's largest frequency is {ratio:.1e} times its slowest \
             decay; give an escape rate"
        )
    };
    QsoError::invalid("rates", why)
}

impl Model {
    /// The block-diagonal elements `(b, r, c)` as one vector over the
    /// blocks `which` (column-major per block), and the offsets.
    pub(super) fn offsets(&self, which: &[usize]) -> (Vec<usize>, usize) {
        let mut offsets = Vec::with_capacity(which.len());
        let mut n = 0;
        for &b in which {
            offsets.push(n);
            n += self.blocks[b].len() * self.blocks[b].len();
        }
        (offsets, n)
    }

    /// The block `b` of `m`.
    pub(super) fn block_of(&self, m: &CMatrix, b: usize) -> CMatrix {
        let states = &self.blocks[b];
        CMatrix::from_fn(states.len(), states.len(), |r, c| m[(states[r], states[c])])
    }

    /// `m` (block-diagonal) on the blocks `which` as one vector.
    pub(super) fn vector_of(&self, m: &CMatrix, which: &[usize]) -> Vec<C64> {
        which
            .iter()
            .flat_map(|&b| self.block_of(m, b).as_slice().to_vec())
            .collect()
    }

    /// The block-diagonal `d × d` matrix of a vector over the blocks
    /// `which`.
    pub(super) fn matrix_of(&self, v: &[C64], which: &[usize], d: usize) -> CMatrix {
        let mut out = CMatrix::zeros(d, d);
        let mut at = 0;
        for &b in which {
            let states = &self.blocks[b];
            let n = states.len();
            for c in 0..n {
                for r in 0..n {
                    out[(states[r], states[c])] = v[at + c * n + r];
                }
            }
            at += n * n;
        }
        out
    }

    /// The blocks marked in `occupied` and, with jumps, every block their
    /// jumps reach from them, in order.
    fn reachable(&self, occupied: &[bool]) -> Vec<usize> {
        let mut seen = occupied.to_vec();
        let mut queue: Vec<usize> = (0..seen.len()).filter(|&b| seen[b]).collect();
        while let Some(b) = queue.pop() {
            for jump in &self.jumps {
                for &s in &self.blocks[b] {
                    if jump.kappa[s] == ZERO {
                        continue;
                    }
                    let to = self.pos[jump.tau[s]].0;
                    if !seen[to] {
                        seen[to] = true;
                        queue.push(to);
                    }
                }
            }
        }
        (0..seen.len()).filter(|&b| seen[b]).collect()
    }

    /// Whether `m` has no entry between blocks above [`BETWEEN`] of its
    /// largest.
    fn respects(&self, m: &CMatrix) -> bool {
        let largest = m.iter().map(|z| z.norm()).fold(0.0, f64::max);
        let d = m.nrows();
        (0..d).all(|c| {
            (0..d).all(|r| self.pos[r].0 == self.pos[c].0 || m[(r, c)].norm() <= BETWEEN * largest)
        })
    }

    /// `−i(A X − X A† − σX) + (c/2) W X W` on block `b`.
    fn local(&self, b: usize, x: &CMatrix) -> CMatrix {
        let mut y = sylvester_apply(&self.a[b], x, self.sigma) * (-I);
        if self.c != 0.0 {
            let sw = &self.swap[b];
            let n = sw.len();
            let half = 0.5 * self.c;
            for col in 0..n {
                for row in 0..n {
                    y[(row, col)] += x[(sw[row], sw[col])] * half;
                }
            }
        }
        y
    }

    /// `y += Σ_j L_j X L_j†` for `x` over the blocks `which` at `offsets`
    /// (`place[b]`: the position of block `b` among them).
    fn add_jumps(&self, x: &[C64], y: &mut [C64], which: &[usize], place: &[Option<usize>]) {
        let (offsets, _) = self.offsets(which);
        for jump in &self.jumps {
            for (k, &b) in which.iter().enumerate() {
                let states = &self.blocks[b];
                let n = states.len();
                for (col, &sc) in states.iter().enumerate() {
                    let kc = jump.kappa[sc];
                    if kc == ZERO {
                        continue;
                    }
                    let (bt, ic) = self.pos[jump.tau[sc]];
                    let Some(kt) = place[bt] else { continue };
                    let nt = self.blocks[bt].len();
                    let base = offsets[kt] + ic * nt;
                    for (row, &sr) in states.iter().enumerate() {
                        let kr = jump.kappa[sr];
                        if kr == ZERO {
                            continue;
                        }
                        let (br, ir) = self.pos[jump.tau[sr]];
                        debug_assert_eq!(br, bt, "jumps keep the blocks");
                        y[base + ir] += kr * x[offsets[k] + col * n + row] * kc.conj();
                    }
                }
            }
        }
    }

    /// `𝓛` on the blocks `which` (`place[b]`: the position of block `b`
    /// among them), in parallel over the blocks.
    fn apply(&self, x: &[C64], y: &mut [C64], which: &[usize], place: &[Option<usize>]) {
        let (offsets, _) = self.offsets(which);
        let parts = map_tasks(which.len(), |k| {
            let n = self.blocks[which[k]].len();
            let xb = CMatrix::from_column_slice(n, n, &x[offsets[k]..offsets[k] + n * n]);
            self.local(which[k], &xb)
        });
        for (k, part) in parts.iter().enumerate() {
            y[offsets[k]..offsets[k] + part.len()].copy_from_slice(part.as_slice());
        }
        self.add_jumps(x, y, which, place);
    }

    /// The generator `𝓛` on all blocks as a dense matrix over the vector
    /// of [`Model::vector_of`] (every block, in order): per block
    /// `−i(1 ⊗ A_b − Ā_b ⊗ 1) + iσ`, the jumps and the term `(c/2) W X W`.
    pub(super) fn generator_dense(&self) -> CMatrix {
        let all: Vec<usize> = (0..self.blocks.len()).collect();
        let (offsets, n_all) = self.offsets(&all);
        let mut g = CMatrix::zeros(n_all, n_all);
        for (b, states) in self.blocks.iter().enumerate() {
            let n = states.len();
            let off = offsets[b];
            let a = &self.a[b];
            let sw = &self.swap[b];
            for col in 0..n {
                for row in 0..n {
                    let i = off + col * n + row;
                    for l in 0..n {
                        g[(i, off + col * n + l)] += -I * a[(row, l)];
                        g[(i, off + l * n + row)] += I * a[(col, l)].conj();
                    }
                    g[(i, i)] += I * self.sigma;
                    if self.c != 0.0 {
                        g[(i, off + sw[col] * n + sw[row])] += C64::new(0.5 * self.c, 0.0);
                    }
                }
            }
        }
        for jump in &self.jumps {
            for (b, states) in self.blocks.iter().enumerate() {
                let n = states.len();
                for (col, &sc) in states.iter().enumerate() {
                    let kc = jump.kappa[sc];
                    if kc == ZERO {
                        continue;
                    }
                    let (bt, ic) = self.pos[jump.tau[sc]];
                    let nt = self.blocks[bt].len();
                    for (row, &sr) in states.iter().enumerate() {
                        let kr = jump.kappa[sr];
                        if kr == ZERO {
                            continue;
                        }
                        let ir = self.pos[jump.tau[sr]].1;
                        g[(offsets[bt] + ic * nt + ir, offsets[b] + col * n + row)] +=
                            kr * kc.conj();
                    }
                }
            }
        }
        g
    }
}

impl RadicalPair {
    /// The operator at field `b`, in blocks of the labels (one block of all
    /// states if `whole`).
    pub(super) fn model(&self, b: f64, whole: bool) -> Model {
        let sys = &self.system;
        let d = sys.dimension();
        let h = sys.hamiltonian_at(b);
        let pi = self.swap();
        let mut labels = sys.labels();
        let conserves = (0..d).all(|c| (0..d).all(|r| h[(r, c)] == ZERO || labels[r] == labels[c]));
        debug_assert!(conserves, "the frames keep the labels");
        if whole || !conserves {
            labels = vec![Vec::new(); d];
        }
        let mut groups: BTreeMap<Vec<i32>, Vec<usize>> = BTreeMap::new();
        for (s, l) in labels.into_iter().enumerate() {
            groups.entry(l).or_default().push(s);
        }
        let blocks: Vec<Vec<usize>> = groups.into_values().collect();
        let mut pos = vec![(0, 0); d];
        for (k, states) in blocks.iter().enumerate() {
            for (i, &s) in states.iter().enumerate() {
                pos[s] = (k, i);
            }
        }
        let swap: Vec<Vec<usize>> = blocks
            .iter()
            .enumerate()
            .map(|(k, states)| {
                states
                    .iter()
                    .map(|&s| {
                        let (kk, i) = pos[pi[s]];
                        debug_assert_eq!(kk, k, "the swap of the electrons keeps the labels");
                        i
                    })
                    .collect()
            })
            .collect();
        let jumps: Vec<Jump> = sys
            .jumps(b)
            .iter()
            .map(|(i, a)| embed_jump(sys, *i, a))
            .collect();
        let gamma: Vec<f64> = (0..d)
            .map(|s| jumps.iter().map(|j| j.kappa[s].norm_sqr()).sum())
            .collect();
        let [ks, kt, ke] = self.rates;
        let c = match self.recombination {
            Recombination::Haberkorn => 0.0,
            Recombination::JonesMaedaHore => 0.5 * (ks + kt),
        };
        // K = (k_S + k_T)/2 + ((k_T − k_S)/2) W.
        let (k0, kw) = (0.5 * (ks + kt), 0.5 * (kt - ks));
        let mut a = Vec::with_capacity(blocks.len());
        let mut mu = Vec::with_capacity(blocks.len());
        let mut norms = Vec::with_capacity(blocks.len());
        for (k, states) in blocks.iter().enumerate() {
            let n = states.len();
            let m = states.iter().map(|&s| h[(s, s)].re).sum::<f64>() / n as f64;
            let mut ab = CMatrix::from_fn(n, n, |r, col| h[(states[r], states[col])]);
            for (col, &s) in states.iter().enumerate() {
                ab[(col, col)] -= C64::new(m, 0.5 * (k0 + gamma[s]));
                ab[(swap[k][col], col)] -= C64::new(0.0, 0.5 * kw);
            }
            norms.push(one_norm(&ab));
            mu.push(m);
            a.push(ab);
        }
        let sigma = C64::new(0.0, ke + 0.5 * c);
        let largest = norms.iter().copied().fold(0.0, f64::max);
        let norm = 2.0 * largest
            + sigma.norm()
            + 0.5 * c
            + jumps
                .iter()
                .map(|j| j.kappa.iter().map(|z| z.norm_sqr()).fold(0.0, f64::max))
                .sum::<f64>();
        Model {
            blocks,
            pos,
            swap,
            a,
            mu,
            norms,
            sigma,
            c,
            jumps,
            norm,
        }
    }

    /// `X` from the normalised start `rho0` at field `b`, and the yields.
    pub(super) fn solve_at(
        &self,
        rho0: &CMatrix,
        b: f64,
        opts: Options,
    ) -> Result<(PairYields, CMatrix)> {
        let [ks, kt, ke] = self.rates;
        if ks + kt + ke <= 0.0 {
            return Err(QsoError::invalid(
                "rates",
                "without recombination or escape the yields do not exist; RadicalPair::evolve \
                 gives the time evolution",
            ));
        }
        let d = self.system.dimension();
        let mut model = self.model(b, opts.whole);
        if !model.respects(rho0) {
            model = self.model(b, true);
        }
        let largest = rho0.iter().map(|z| z.norm()).fold(0.0, f64::max);
        let occupied: Vec<bool> = model
            .blocks
            .iter()
            .map(|states| {
                states.iter().any(|&r| {
                    states
                        .iter()
                        .any(|&c| rho0[(r, c)].norm() > BETWEEN * largest)
                })
            })
            .collect();
        // Without jumps the occupied blocks; with them every block their
        // jumps reach (an unreachable block may hold states that never
        // decay, which the start never meets).
        let solved = model.reachable(&occupied);
        let direct = model.c == 0.0 && model.jumps.is_empty() && !opts.force_gmres;
        self.check_limits(&model, &solved, direct, b)?;
        let schurs: Vec<SchurForm> =
            map_tasks(solved.len(), |k| SchurForm::new(&model.a[solved[k]]))
                .into_iter()
                .collect::<Result<_>>()?;
        // The slowest decay of the solved blocks, `2·min(−Im λ(A)) + k_E`.
        let mut gap = f64::INFINITY;
        for (k, &blk) in solved.iter().enumerate() {
            let slowest = 2.0 * schurs[k].slowest_decay();
            let threshold = DARK * model.norms[blk];
            // Without jumps, Haberkorn's A decides for both terms (see the
            // module docs); with them the preconditioner needs its gap.
            let dark = model.jumps.is_empty() && slowest + ke <= threshold;
            if dark || slowest + model.sigma.im <= threshold {
                return Err(never_decays());
            }
            gap = gap.min(slowest + ke);
        }
        let (x, backward_error, iterations, residual) = if direct {
            self.sylvester_path(&model, &schurs, &solved, rho0)
        } else {
            self.gmres_path(&model, &schurs, &solved, rho0)?
        };
        // A NaN residual is refused too.
        if residual.is_nan() || residual > RESIDUAL {
            let start = fro(&model.vector_of(rho0, &solved));
            let bounded = gap > 0.0 && fro(&x) <= GROWTH * start / gap;
            if residual.is_nan() || !bounded || backward_error.is_nan() || backward_error > BACKWARD
            {
                return Err(singular(
                    residual,
                    backward_error,
                    model.norm / gap,
                    bounded,
                    ke,
                ));
            }
        }
        let x = model.matrix_of(&x, &solved, d);
        Ok((self.observe(&x, b, backward_error, iterations), x))
    }

    /// Refuse a solve beyond the limits of work and memory, before
    /// anything is factorised.
    fn check_limits(&self, model: &Model, solved: &[usize], direct: bool, b: f64) -> Result<()> {
        let d = self.system.dimension() as f64;
        let cube: f64 = solved
            .iter()
            .map(|&k| (model.blocks[k].len() as f64).powi(3))
            .sum();
        let square: f64 = solved
            .iter()
            .map(|&k| (model.blocks[k].len() as f64).powi(2))
            .sum();
        let work = SYLVESTER_WORK * cube;
        let vectors = if direct { 6.0 } else { (RESTART + 6) as f64 };
        let memory = 16.0 * (vectors * square + 2.0 * square + 2.0 * d * d);
        if work > MAX_RELAXATION_WORK || memory > MAX_RELAXATION_MEMORY {
            return Err(QsoError::invalid(
                "work",
                format!(
                    "the radical pair needs about {work:.1e} operations and {memory:.1e} bytes at \
                     {b} T (limits {MAX_RELAXATION_WORK:.1e} and {MAX_RELAXATION_MEMORY:.1e}); use \
                     the rotating frame or fewer nuclei"
                ),
            ));
        }
        Ok(())
    }

    /// Haberkorn's term without jumps: one Sylvester solve per block,
    /// refined; `X` over `solved`, the largest backward error of a block,
    /// no iterations and the residual relative to the start.
    fn sylvester_path(
        &self,
        model: &Model,
        schurs: &[SchurForm],
        solved: &[usize],
        rho0: &CMatrix,
    ) -> (Vec<C64>, f64, usize, f64) {
        let sigma = model.sigma;
        let parts = map_tasks(solved.len(), |k| {
            let blk = solved[k];
            let a = &model.a[blk];
            let rhs = model.block_of(rho0, blk) * (-I);
            let size = fro(rhs.as_slice());
            let eta = |x: &CMatrix| -> (CMatrix, f64) {
                let r = &rhs - sylvester_apply(a, x, sigma);
                let e = fro(r.as_slice()) / (model.norm * fro(x.as_slice()) + size);
                (r, e)
            };
            let mut x = hermitian_part(&schurs[k].solve(&rhs, sigma));
            let (mut r, mut e) = eta(&x);
            for _ in 0..REFINEMENTS {
                if e.is_nan() || e <= REFINED {
                    break;
                }
                x += hermitian_part(&schurs[k].solve(&r, sigma));
                (r, e) = eta(&x);
            }
            (x, e, fro(r.as_slice()).powi(2), size * size)
        });
        let mut x = Vec::new();
        let mut worst: f64 = 0.0;
        let (mut left, mut start) = (0.0, 0.0);
        for (xb, e, r, s) in parts {
            x.extend_from_slice(xb.as_slice());
            worst = worst.max(e);
            left += r;
            start += s;
        }
        (x, worst, 0, (left / start).sqrt())
    }

    /// The term of Jones, Maeda and Hore or jumps: GMRES over the blocks
    /// `solved`, preconditioned on the right by the Sylvester solves.
    fn gmres_path(
        &self,
        model: &Model,
        schurs: &[SchurForm],
        solved: &[usize],
        rho0: &CMatrix,
    ) -> Result<(Vec<C64>, f64, usize, f64)> {
        let (offsets, n) = model.offsets(solved);
        let mut place = vec![None; model.blocks.len()];
        for (k, &b) in solved.iter().enumerate() {
            place[b] = Some(k);
        }
        let rhs: Vec<C64> = model
            .vector_of(rho0, solved)
            .into_iter()
            .map(|z| -z)
            .collect();
        let apply = |x: &[C64], y: &mut [C64]| model.apply(x, y, solved, &place);
        let sigma = model.sigma;
        let precondition = |v: &[C64]| -> Vec<C64> {
            let parts = map_tasks(solved.len(), |k| {
                let m = model.blocks[solved[k]].len();
                let vb = CMatrix::from_column_slice(m, m, &v[offsets[k]..offsets[k] + m * m]);
                schurs[k].solve(&(vb * I), sigma)
            });
            parts
                .iter()
                .flat_map(|p| p.as_slice().iter().copied())
                .collect()
        };
        let cube: f64 = solved
            .iter()
            .map(|&k| (model.blocks[k].len() as f64).powi(3))
            .sum();
        let opts = GmresOptions {
            tol: 1e-13,
            restart: RESTART,
            max_iterations: MAX_ITERATIONS,
            work: (
                ITERATION_WORK * cube + 16.0 * (model.jumps.len() as f64 + 2.0) * n as f64,
                MAX_RELAXATION_WORK - SCHUR_WORK * cube,
            ),
            scale: model.norm,
            breakdown: 1e-14,
        };
        let out = gmres(n, apply, precondition, &rhs, &opts);
        match out.end {
            GmresEnd::Converged => {}
            GmresEnd::Work => {
                return Err(QsoError::invalid(
                    "work",
                    format!(
                        "the radical pair needs more than {MAX_RELAXATION_WORK:.1e} operations \
                         after {} GMRES iterations; use the rotating frame, fewer nuclei or no \
                         relaxation",
                        out.iterations
                    ),
                ));
            }
            GmresEnd::Breakdown => {
                return Err(QsoError::invalid(
                    "rates",
                    "the pair's equations are singular to rounding: some state never \
                     recombines, escapes or relaxes; give an escape rate",
                ));
            }
            GmresEnd::Iterations => {
                return Err(QsoError::NoConvergence {
                    method: "GMRES (radical pair)",
                    iterations: out.iterations,
                    residual: out.residual,
                });
            }
        }
        // Hermitian blocks, and the backward error of the result.
        let mut x = out.x;
        for (k, &b) in solved.iter().enumerate() {
            let m = model.blocks[b].len();
            let xb = CMatrix::from_column_slice(m, m, &x[offsets[k]..offsets[k] + m * m]);
            x[offsets[k]..offsets[k] + m * m].copy_from_slice(hermitian_part(&xb).as_slice());
        }
        let mut lx = vec![ZERO; n];
        apply(&x, &mut lx);
        let residual: Vec<C64> = lx.iter().zip(&rhs).map(|(l, r)| l - r).collect();
        let eta = fro(&residual) / (model.norm * fro(&x) + fro(&rhs));
        Ok((x, eta, out.iterations, fro(&residual) / fro(&rhs)))
    }

    /// `ρ(t)` from `rho` at field `b` (`t > 0`; see
    /// [`RadicalPair::evolve`]).
    pub(super) fn evolve_at(&self, rho: &CMatrix, t: f64, b: f64) -> Result<CMatrix> {
        let d = self.system.dimension();
        let model = self.model(b, false);
        let [ks, kt, ke] = self.rates;
        if model.c == 0.0 && model.jumps.is_empty() {
            let sizes: Vec<f64> = model.blocks.iter().map(|s| s.len() as f64).collect();
            let work = 240.0 * sizes.iter().map(|n| n.powi(3)).sum::<f64>()
                + 64.0 * d as f64 * sizes.iter().map(|n| n * n).sum::<f64>();
            if work > MAX_RELAXATION_WORK {
                return Err(QsoError::invalid(
                    "work",
                    format!(
                        "the evolution needs about {work:.1e} operations (limit \
                         {MAX_RELAXATION_WORK:.1e})"
                    ),
                ));
            }
            // U_b = e^{−iμ_b t} exp(−iA_b t) e^{−k_E t/2}; A_b is Hermitian
            // without recombination.
            let unitary = ks == 0.0 && kt == 0.0;
            let us = map_tasks(model.blocks.len(), |k| {
                let phase = C64::from_polar((-0.5 * ke * t).exp(), -model.mu[k] * t);
                let u = if unitary {
                    expm_hermitian(&model.a[k], t)
                } else {
                    expm(&(&model.a[k] * C64::new(0.0, -t)))
                };
                u * phase
            });
            let mut out = CMatrix::zeros(d, d);
            for (ka, sa) in model.blocks.iter().enumerate() {
                for (kb, sb) in model.blocks.iter().enumerate() {
                    let sub = CMatrix::from_fn(sa.len(), sb.len(), |r, c| rho[(sa[r], sb[c])]);
                    if sub.iter().all(|z| *z == ZERO) {
                        continue;
                    }
                    let moved = cmul(&cmul(&us[ka], &sub), &us[kb].adjoint());
                    for (r, &i) in sa.iter().enumerate() {
                        for (c, &j) in sb.iter().enumerate() {
                            out[(i, j)] = moved[(r, c)];
                        }
                    }
                }
            }
            return Ok(out);
        }
        if !model.respects(rho) {
            return Err(QsoError::invalid(
                "rho",
                "with relaxation or the term of Jones, Maeda and Hore, ρ must conserve the labels",
            ));
        }
        let all: Vec<usize> = (0..model.blocks.len()).collect();
        let (_, n) = model.offsets(&all);
        let nf = n as f64;
        if n > MAX_LIOUVILLE_BLOCK
            || 240.0 * nf.powi(3) > MAX_RELAXATION_WORK
            || 48.0 * nf * nf > MAX_RELAXATION_MEMORY
        {
            return Err(QsoError::invalid(
                "rho",
                format!(
                    "with relaxation or the term of Jones, Maeda and Hore the evolution needs a \
                     Liouville space of {n} elements (limit {MAX_LIOUVILLE_BLOCK}); solve for the \
                     yields instead"
                ),
            ));
        }
        let g = model.generator_dense() * C64::new(t, 0.0);
        let v = nalgebra::DVector::from_vec(model.vector_of(rho, &all));
        let moved = expm(&g) * v;
        Ok(model.matrix_of(moved.as_slice(), &all, d))
    }
}
