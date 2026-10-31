//! GRAPE on sparse blocks: rf control problems with relaxation on the
//! Liouville generator of the master equation kept as sparse matrices (the
//! blocks of the relaxation of larger spin systems, here one block of all
//! elements with the drift column), applied to vectors only, and reduced to
//! the orbits of declared equivalent spins when the system, the rf channels
//! and the initial state respect them.
//!
//! The members of an ensemble share the matrices: an offset adds the
//! diagonal `−i[2πΔF_z, ·]`, a scaling multiplies the controls. The
//! generator of a step is never formed: its terms are applied in one pass
//! over the rows.
//!
//! On the orbits `O` of the elements the state keeps one coefficient per
//! orbit (the value of each of its elements), the generators become
//! `M_OP = Σ_{c ∈ P} L_{o,c}` read at a representative `o`, and the target
//! pairs with the state through its sums `Σ_{i ∈ O} C_i`: the transfer and
//! its gradient are those of an ordinary linear system on the orbits, with
//! the explicit adjoints of the reduced matrices.

use nalgebra::DVector;
use num_complex::Complex64 as C64;

use super::super::blocks::{Liouville, SYMMETRY_TOLERANCE, Sparse};
use super::super::krylov::{Orbits, SparseBlock, accumulator_bytes, row_dot, rows_per_task};
use super::super::{ONE, SpinSystem, TWO_PI, ZERO};
use super::{
    Control, Direction, Drive, Linear, Member, augment, expmv, frechet, max_abs, substeps,
};
use crate::error::{QsoError, Result};
use crate::linalg::CMatrix;
use crate::parallel::{for_each_block, sum_chunks_c64, threads};

/// Largest `‖hL‖` of a substep (with `‖L‖ = max(‖L‖₁, ‖L‖∞)`): its Taylor
/// series needs about 26 terms, none larger than `e²` times the vector,
/// and once a term falls below the rounding of the sum the rest add at
/// most `e²` times as much. Four times the substep of [`expmv`], at about
/// a third of its products per unit of `‖L‖t`.
const SUBSTEP: f64 = 2.0;

/// Terms of a substep's Taylor series, for the estimates: 9 to 11 were
/// measured on methyl groups with up to eight spins, where the norm bound
/// would allow 26 (the series stops when its terms vanish).
const TAYLOR_TERMS: f64 = 14.0;

/// Gauss–Legendre nodes and weights on `[0, 1]`, ten points: exact for
/// polynomials of degree 19. The integrand `⟨λ(τ), E x(τ)⟩` of a substep
/// with `‖hL‖ ≤ 2` has a 20th derivative of at most `(2‖L‖)^20` times
/// `‖λ‖‖E‖‖x‖`, so the rule's error stays below
/// `h·4^20·(10!)⁴/(21·(20!)³) ≈ 6·10⁻¹⁹·h` of that: below rounding.
const GAUSS: [(f64, f64); 10] = [
    (0.013046735741414128, 0.03333567215434407),
    (0.06746831665550773, 0.0747256745752902),
    (0.16029521585048778, 0.109543181257991),
    (0.2833023029353764, 0.13463335965499826),
    (0.4255628305091844, 0.1477621123573764),
    (0.5744371694908156, 0.1477621123573764),
    (0.7166976970646236, 0.13463335965499826),
    (0.8397047841495122, 0.109543181257991),
    (0.9325316833444923, 0.0747256745752902),
    (0.9869532642585859, 0.03333567215434407),
];

/// Substeps of [`SUBSTEP`] for an exponential of norm `norm`.
fn substeps_of(norm: f64) -> usize {
    (norm / SUBSTEP).ceil().max(1.0) as usize
}

/// Generator products of [`series`] (tests).
#[cfg(test)]
pub(super) static PRODUCTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `e^{hL} v`, calling `each` with every Taylor term `(hL)^k v/k!` (the
/// first `v`), summed until a term falls below 10⁻² of the rounding of the
/// largest entry of the sum.
fn series<G: Linear>(
    g: &G,
    v: &DVector<C64>,
    h: f64,
    mut each: impl FnMut(&DVector<C64>),
) -> DVector<C64> {
    each(v);
    let mut term = v.clone();
    let mut acc = v.clone();
    for k in 1..200 {
        term = g.apply(&term, h / k as f64);
        #[cfg(test)]
        PRODUCTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        acc += &term;
        each(&term);
        if max_abs(&term) <= f64::EPSILON * 1e-2 * max_abs(&acc) {
            break;
        }
    }
    acc
}

/// `e^{tL} x` in substeps of [`SUBSTEP`].
fn propagate<G: Linear>(g: &G, x: &DVector<C64>, t: f64, norm: f64) -> DVector<C64> {
    let s = substeps_of(t * norm);
    let h = t / s as f64;
    let mut v = x.clone();
    for _ in 0..s {
        v = series(g, &v, h, |_| {});
    }
    v
}

/// `e^{hL} v` and its Taylor terms.
fn taylor<G: Linear>(g: &G, v: &DVector<C64>, h: f64) -> (DVector<C64>, Vec<DVector<C64>>) {
    let mut terms = Vec::new();
    let sum = series(g, v, h, |t| terms.push(t.clone()));
    (sum, terms)
}

/// `Σ_k θ^k terms[k]` for every `θ` (Horner), rows in parallel.
fn horner(terms: &[DVector<C64>], thetas: impl Iterator<Item = f64>) -> Vec<DVector<C64>> {
    let n = terms[0].len();
    let rows = 4096;
    thetas
        .map(|theta| {
            let mut out = DVector::zeros(n);
            for_each_block(out.as_mut_slice(), rows, |b, chunk| {
                for (i, o) in chunk.iter_mut().enumerate() {
                    let r = b * rows + i;
                    let mut acc = ZERO;
                    for t in terms.iter().rev() {
                        acc = acc * theta + t[r];
                    }
                    *o = acc;
                }
            });
            out
        })
        .collect()
}

/// `s·⟨λ, E x⟩` of a direction `s·E`, rows in parallel and summed in a
/// fixed order.
fn pair(e: &Scaled<'_>, lambda: &DVector<C64>, x: &DVector<C64>) -> C64 {
    let n = lambda.len();
    let rows = rows_per_task(e.0.nnz(), n);
    let x = x.as_slice();
    let sum = sum_chunks_c64(n.div_ceil(rows), e.0.nnz() + n, |c| {
        let mut acc = ZERO;
        for r in c * rows..((c + 1) * rows).min(n) {
            let (cols, vals) = e.0.row_parts(r);
            acc += lambda[r].conj() * row_dot(cols, vals, x);
        }
        acc
    });
    sum * e.1
}

/// The derivatives `Re ∫₀^Δt ⟨λ(τ), E_j x(τ)⟩ dτ` of the transfer with
/// respect to the controls of one step (`x(τ) = e^{Lτ} x`, `λ(τ) =
/// e^{L†(Δt − τ)} λ`), and the costate at the step's start: per substep of
/// [`SUBSTEP`] the Gauss–Legendre rule of [`GAUSS`], with `x` and `λ` at
/// its nodes summed from the Taylor terms of the substep's exponentials.
/// Exact to rounding like [`frechet`], with two generator products per
/// term (forwards again and backwards) instead of `2 + directions`, and
/// four times longer substeps.
#[allow(clippy::too_many_arguments)]
fn step_gradient(
    forward: &SparseGenerator<'_>,
    backward: &SparseGenerator<'_>,
    directions: &[Scaled<'_>],
    x: &DVector<C64>,
    lambda: &DVector<C64>,
    dt: f64,
    norm: f64,
) -> (Vec<f64>, DVector<C64>) {
    let s = substeps_of(dt * norm);
    let h = dt / s as f64;
    let mut nodes = Vec::with_capacity(s);
    let mut v = x.clone();
    for _ in 0..s {
        let (next, terms) = taylor(forward, &v, h);
        nodes.push(horner(&terms, GAUSS.iter().map(|g| g.0)));
        v = next;
    }
    let mut grad = vec![0.0; directions.len()];
    let mut mu = lambda.clone();
    for at in nodes.iter().rev() {
        let (previous, terms) = taylor(backward, &mu, h);
        let costates = horner(&terms, GAUSS.iter().map(|g| 1.0 - g.0));
        for ((xq, lq), &(_, w)) in at.iter().zip(&costates).zip(&GAUSS) {
            for (g, e) in grad.iter_mut().zip(directions) {
                *g += h * w * pair(e, lq, xq).re;
            }
        }
        mu = previous;
    }
    (grad, mu)
}

/// `−i[A, ·]` on the row-major vectors of `d × d` matrices, with a last
/// element (the trace) it leaves alone, as a sparse matrix.
fn commutator_block(a: &CMatrix) -> SparseBlock {
    let d = a.nrows();
    let mi = C64::new(0.0, -1.0);
    let mut triplets = Vec::new();
    for r in 0..d {
        for c in 0..d {
            let v = a[(r, c)];
            if v == ZERO {
                continue;
            }
            for k in 0..d {
                // (Aρ)_{rk} = Σ_c A_rc ρ_ck; (ρA)_{kc} = Σ_r ρ_kr A_rc.
                triplets.push(((r * d + k) as u32, (c * d + k) as u32, mi * v));
                triplets.push(((k * d + c) as u32, (k * d + r) as u32, -mi * v));
            }
        }
    }
    SparseBlock::from_triplets(d * d + 1, triplets)
}

/// The largest absolute entry of `x`.
fn largest(x: &[C64]) -> f64 {
    x.iter().fold(0.0, |m, z| m.max(z.norm()))
}

/// Nonzero entries of `a`.
fn nonzeros(a: &CMatrix) -> usize {
    a.iter().filter(|z| **z != ZERO).count()
}

/// Whether `a` equals its images under the state permutations `maps`
/// (to 10⁻¹² of its largest entry).
fn invariant(a: &CMatrix, maps: &[Vec<usize>]) -> bool {
    let d = a.nrows();
    let tol = 1e-12 * a.iter().fold(0.0f64, |m, z| m.max(z.norm()));
    maps.iter()
        .all(|m| (0..d).all(|c| (0..d).all(|r| (a[(m[r], m[c])] - a[(r, c)]).norm() <= tol)))
}

/// Whether `a` is exactly Hermitian.
fn hermitian(a: &CMatrix) -> bool {
    let d = a.nrows();
    (0..d).all(|c| (0..d).all(|r| a[(r, c)] == a[(c, r)].conj()))
}

/// `max(‖A‖₁, ‖A‖∞)`: bounds every norm the Taylor series of `A` and `A†`
/// meet.
fn norm_bound(a: &SparseBlock) -> f64 {
    a.norm_one().max(a.norm_inf())
}

/// The images of the augmented elements `a·d + b` (and the trace, last)
/// under state permutations.
fn element_images(maps: &[Vec<usize>], d: usize) -> Vec<Vec<u32>> {
    let n = d * d;
    maps.iter()
        .map(|m| {
            (0..=n)
                .map(|j| {
                    if j == n {
                        n as u32
                    } else {
                        (m[j / d] * d + m[j % d]) as u32
                    }
                })
                .collect()
        })
        .collect()
}

/// `L = base + diag(diagonal) + Σ_j c_j ops_j`, applied to vectors in one
/// pass over the rows.
struct SparseGenerator<'a> {
    base: &'a SparseBlock,
    diagonal: Option<&'a [C64]>,
    ops: &'a [SparseBlock],
    coefficients: Vec<f64>,
}

impl Linear for SparseGenerator<'_> {
    fn apply(&self, x: &DVector<C64>, t: f64) -> DVector<C64> {
        let n = x.len();
        let mut y = DVector::zeros(n);
        let terms: Vec<(&SparseBlock, C64)> = self
            .ops
            .iter()
            .zip(&self.coefficients)
            .filter(|&(_, &c)| c != 0.0)
            .map(|(op, &c)| (op, C64::new(c * t, 0.0)))
            .collect();
        let nnz = self.base.nnz() + terms.iter().map(|(op, _)| op.nnz()).sum::<usize>();
        let rows = rows_per_task(nnz, n);
        let scale = C64::new(t, 0.0);
        let x = x.as_slice();
        for_each_block(y.as_mut_slice(), rows, |k, chunk| {
            for (i, out) in chunk.iter_mut().enumerate() {
                let r = k * rows + i;
                let (cols, vals) = self.base.row_parts(r);
                let mut acc = row_dot(cols, vals, x);
                if let Some(d) = self.diagonal {
                    acc += d[r] * x[r];
                }
                acc *= scale;
                for (op, c) in &terms {
                    let (cols, vals) = op.row_parts(r);
                    acc += c * row_dot(cols, vals, x);
                }
                *out = acc;
            }
        });
        y
    }
}

/// A control direction `s·E_j`.
struct Scaled<'a>(&'a SparseBlock, f64);

impl Direction for Scaled<'_> {
    fn apply_add(&self, x: &DVector<C64>, s: C64, y: &mut DVector<C64>) {
        self.0.apply_add(x.as_slice(), s * self.1, y.as_mut_slice());
    }
}

/// What an rf control problem with relaxation needs on sparse blocks,
/// built once for any number of evaluations.
pub(super) struct SparseControl {
    /// `L₀` of the ideal member with the drift column, and `L₀†`.
    base: SparseBlock,
    base_adjoint: SparseBlock,
    /// `−i[2πF_x, ·]` and `−i[2πF_y, ·]` of every channel, in the order
    /// of the controls.
    ops: Vec<SparseBlock>,
    /// Their adjoints on the orbits; `None`: their negatives (`F_x` and
    /// `F_y` are Hermitian).
    ops_adjoint: Option<Vec<SparseBlock>>,
    /// The diagonal of `−i[2πF_z, ·]` of every channel, for offsets.
    offsets: Vec<Vec<C64>>,
    /// `max(‖·‖₁, ‖·‖∞)` of `L₀` and of every op.
    base_norm: f64,
    op_norms: Vec<f64>,
    /// Whether the elements are reduced to their orbits.
    #[cfg_attr(not(test), allow(dead_code))]
    reduced: bool,
    /// Gradients by Gauss–Legendre nodes of every substep
    /// ([`step_gradient`]), else by the augmented exponentials of
    /// [`frechet`] (tests).
    pub(super) quadrature: bool,
    /// The initial state `[vec Δ; Tr ρ]` and the target it pairs with (on
    /// the orbits: the coefficients and the sums over every orbit).
    start: DVector<C64>,
    target: DVector<C64>,
}

impl SparseControl {
    /// Coefficients propagated: `d² + 1`, or the orbits.
    pub(super) fn size(&self) -> usize {
        self.start.len()
    }

    /// Whether the problem propagates on the orbits of a symmetry.
    #[cfg(test)]
    pub(super) fn reduced(&self) -> bool {
        self.reduced
    }

    /// Entries of `L₀` and of the control directions, and the norm bounds
    /// (tests).
    #[cfg(test)]
    pub(super) fn entries(&self) -> (usize, usize, f64, Vec<f64>) {
        (
            self.base.nnz(),
            self.ops.iter().map(SparseBlock::nnz).sum(),
            self.base_norm,
            self.op_norms.clone(),
        )
    }

    /// Bytes held by the matrices.
    pub(super) fn bytes(&self) -> f64 {
        let blocks = self.base.bytes()
            + self.base_adjoint.bytes()
            + self.ops.iter().map(SparseBlock::bytes).sum::<usize>()
            + self
                .ops_adjoint
                .as_ref()
                .map_or(0, |a| a.iter().map(SparseBlock::bytes).sum());
        blocks as f64 + 16.0 * (self.offsets.len() + 2) as f64 * self.size() as f64
    }

    /// The member's scaling of every control and its diagonal (offsets).
    fn member_terms(&self, c: &Control, m: &Member) -> (Vec<f64>, Option<Vec<C64>>) {
        let scales = (0..c.per_step())
            .map(|j| m.scales.get(j / 2).copied().unwrap_or(1.0))
            .collect();
        let diagonal = m.offsets.iter().any(|&o| o != 0.0).then(|| {
            let mut out = vec![ZERO; self.size()];
            for (z, &o) in self.offsets.iter().zip(&m.offsets) {
                for (x, y) in out.iter_mut().zip(z) {
                    *x += y * o;
                }
            }
            out
        });
        (scales, diagonal)
    }

    /// Estimated operations and memory of one evaluation under the
    /// controls `u` (see [`member_cost`]), with the substeps of every step.
    pub(super) fn cost(&self, c: &Control, u: &[f64]) -> (f64, f64) {
        let members = c.members();
        let p = c.per_step();
        let m = self.size() as f64;
        let ops: f64 = self.ops.iter().map(|o| o.nnz() as f64).sum();
        let entries = self.base.nnz() as f64 + ops + m;
        let mut work = 0.0;
        let mut bytes = 0.0f64;
        for member in &members {
            let (scales, diagonal) = self.member_terms(c, member);
            let diagonal = diagonal.map_or(0.0, |d| largest(&d));
            let mut substeps_total = 0.0;
            let mut most = 0.0f64;
            for step in u.chunks(p) {
                let norm = self.base_norm
                    + diagonal
                    + step
                        .iter()
                        .zip(&scales)
                        .zip(&self.op_norms)
                        .map(|((x, s), n)| (x * s).abs() * n)
                        .sum::<f64>();
                let s = if self.quadrature {
                    substeps_of(norm * c.step())
                } else {
                    substeps(norm * c.step())
                } as f64;
                substeps_total += s;
                most = most.max(s);
            }
            let (w, b) = member_cost(c, m, entries, ops, substeps_total, most, self.quadrature);
            work += w;
            bytes = bytes.max(b);
        }
        let parallel = members.len().min(threads()) as f64;
        (work, self.bytes() + parallel * bytes)
    }

    /// Transfer and gradient of member `m` under the physical controls `u`.
    pub(super) fn member(&self, c: &Control, m: &Member, u: &[f64]) -> (f64, Vec<f64>) {
        let (p, dt) = (c.per_step(), c.step());
        let (scales, diagonal) = self.member_terms(c, m);
        let conjugate: Option<Vec<C64>> = diagonal
            .as_ref()
            .map(|d| d.iter().map(|z| z.conj()).collect());
        let largest = diagonal.as_deref().map_or(0.0, largest);
        let coefficients = |k: usize| -> Vec<f64> {
            u[k * p..(k + 1) * p]
                .iter()
                .zip(&scales)
                .map(|(x, s)| x * s)
                .collect()
        };
        let forward = |k: usize| SparseGenerator {
            base: &self.base,
            diagonal: diagonal.as_deref(),
            ops: &self.ops,
            coefficients: coefficients(k),
        };
        let backward = |k: usize| match &self.ops_adjoint {
            Some(adjoint) => SparseGenerator {
                base: &self.base_adjoint,
                diagonal: conjugate.as_deref(),
                ops: adjoint,
                coefficients: coefficients(k),
            },
            None => SparseGenerator {
                base: &self.base_adjoint,
                diagonal: conjugate.as_deref(),
                ops: &self.ops,
                coefficients: coefficients(k).into_iter().map(|x| -x).collect(),
            },
        };
        let norm = |k: usize| {
            self.base_norm
                + largest
                + coefficients(k)
                    .iter()
                    .zip(&self.op_norms)
                    .map(|(x, n)| x.abs() * n)
                    .sum::<f64>()
        };
        let mut states = Vec::with_capacity(c.steps + 1);
        states.push(self.start.clone());
        for k in 0..c.steps {
            let next = if self.quadrature {
                propagate(&forward(k), &states[k], dt, norm(k))
            } else {
                expmv(&forward(k), &states[k], dt, norm(k))
            };
            states.push(next);
        }
        let value = self.target.dotc(&states[c.steps]).re;
        let directions: Vec<Scaled<'_>> = self
            .ops
            .iter()
            .zip(&scales)
            .map(|(op, &s)| Scaled(op, s))
            .collect();
        let mut grad = vec![0.0; u.len()];
        let mut lambda = self.target.clone();
        for k in (0..c.steps).rev() {
            let n = norm(k);
            let g = if self.quadrature {
                let (g, previous) = step_gradient(
                    &forward(k),
                    &backward(k),
                    &directions,
                    &states[k],
                    &lambda,
                    dt,
                    n,
                );
                lambda = previous;
                g
            } else {
                let g = frechet(&forward(k), &directions, &states[k], &lambda, dt, n);
                lambda = expmv(&backward(k), &lambda, dt, n);
                g
            };
            grad[k * p..(k + 1) * p].copy_from_slice(&g);
        }
        (value, grad)
    }
}

/// Operations and bytes of one member's evaluation with `m` coefficients,
/// `entries` in the generator (the free part, the controls and the
/// diagonal), `ops` in the control directions, `substeps` over all steps
/// and at most `most` in one. With `quadrature`, Taylor series of
/// [`TAYLOR_TERMS`] terms: a product per term for the transfer and two for
/// the gradient (forwards again and backwards), the Horner sums of both at
/// the nodes of [`GAUSS`] and a pairing per node and direction, holding
/// the nodes of every substep of a step; else [`frechet`]'s 16 terms of
/// `2 + directions` products and one backwards. Every member keeps the
/// state of every step.
fn member_cost(
    c: &Control,
    m: f64,
    entries: f64,
    ops: f64,
    substeps: f64,
    most: f64,
    quadrature: bool,
) -> (f64, f64) {
    let (p, q) = (c.per_step() as f64, GAUSS.len() as f64);
    let (work, nodes) = if quadrature {
        let t = TAYLOR_TERMS;
        let products = 8.0 * 3.0 * t * entries;
        let horner = 2.0 * 4.0 * q * t * m;
        let pairs = 8.0 * q * (ops + p * m);
        (substeps * (products + horner + pairs), q * most)
    } else {
        (8.0 * substeps * 16.0 * ((p + 3.0) * entries + ops), 0.0)
    };
    let bytes = 16.0 * m * (c.steps as f64 + 1.0 + nodes + 26.0 + 4.0 * p + 16.0);
    (work, bytes)
}

/// The rf amplitudes an estimate assumes.
#[derive(Clone, Copy, Debug)]
pub(super) enum Amplitudes<'a> {
    /// Every control at most this (Hz).
    Peak(f64),
    /// These controls.
    Controls(&'a [f64]),
}

/// The sparse blocks of an rf control problem, counted without building
/// them.
struct Counted {
    /// Entries of the full `L₀` (with the drift column), of all control
    /// directions, and of the largest.
    base: f64,
    ops: f64,
    largest_op: f64,
    /// Work of building `L₀`.
    build: f64,
    /// Bound on `‖L₀‖`, on the norm of every direction per Hz of rf, and
    /// per Hz of offset.
    base_norm: f64,
    per_rf: Vec<f64>,
    per_offset: f64,
    /// Elements `d² + 1`, and the share of them the orbits keep.
    n: f64,
    share: f64,
    /// Bytes the relaxation holds while `L₀` is built.
    relaxation: f64,
}

impl SpinSystem {
    /// Count the sparse blocks of an rf control problem: their entries
    /// column by column (`exact`: summed, else bounded with the Redfield
    /// part's bound), the norm from the counted rows and the relaxation's
    /// rate bound, and the orbits when the symmetry applies to the
    /// Hamiltonian, the channels and the initial state.
    fn count_sparse(&self, c: &Control, symmetric: bool, exact: bool) -> Result<Counted> {
        let Drive::Rf { targets, .. } = &c.drive else {
            return Err(QsoError::invalid(
                "control",
                "sparse blocks take rf controls",
            ));
        };
        let d = self.dimension();
        let n = d * d + 1;
        let h0 = self.hamiltonian_at(self.field());
        let relaxation = self.relaxation(self.field())?;
        let liouville = Liouville::new(&vec![Vec::new(); d], None);
        let sparse = Sparse::new(&h0);
        let (count, mean, spread) = self.block_estimate(&liouville, 0, &sparse, &relaxation);
        let (redfield, redfield_build) = relaxation.redfield_entries(&liouville, Some(0));
        let base = if exact {
            self.sparse_block_entries(&liouville, 0, &sparse, &relaxation, true) as f64
        } else {
            count as f64 + redfield + n as f64
        };
        let mut ops = 0.0;
        let mut largest_op = 0.0f64;
        let mut per_rf = Vec::with_capacity(2 * targets.len());
        let mut per_offset = 0.0;
        let mut channels = Vec::new();
        for &t in targets {
            let [fx, fy, fz] = self.channel(t)?;
            for f in [&fx, &fy] {
                let entries = 2.0 * d as f64 * nonzeros(f) as f64;
                ops += entries;
                largest_op = largest_op.max(entries);
                per_rf.push(2.0 * TWO_PI * crate::linalg::dense::one_norm(f));
            }
            let z: Vec<f64> = (0..d).map(|s| fz[(s, s)].re).collect();
            let span = z.iter().fold(f64::MIN, |a, &b| a.max(b))
                - z.iter().fold(f64::MAX, |a, &b| a.min(b));
            per_offset += TWO_PI * span;
            channels.extend([fx, fy, fz]);
        }
        let mut share = 1.0;
        if symmetric
            && let Some(maps) = self.symmetry_maps(&h0)
            && channels.iter().all(|f| invariant(f, &maps))
        {
            let orbits = Orbits::new(n, &element_images(&maps, d));
            let start = augment(&c.initial);
            let largest = start
                .iter()
                .take(n - 1)
                .fold(0.0f64, |m, z| m.max(z.norm()));
            if orbits.asymmetry(start.as_slice()) <= SYMMETRY_TOLERANCE * largest {
                share = orbits.len() as f64 / n as f64;
            }
        }
        Ok(Counted {
            base,
            ops,
            largest_op,
            build: 50.0 * count as f64 + redfield_build,
            base_norm: spread + mean.norm() + relaxation.rate,
            per_rf,
            per_offset,
            n: n as f64,
            share,
            relaxation: self.held_bytes(&relaxation),
        })
    }

    /// Estimated operations and memory of one evaluation on sparse blocks
    /// with rf of at most `peak` Hz, without building them (see
    /// [`SpinSystem::sparse_estimate`]).
    pub(super) fn sparse_cost(
        &self,
        c: &Control,
        rf: Amplitudes<'_>,
        symmetric: bool,
    ) -> Result<(f64, f64)> {
        let (work, memory, _) = self.sparse_estimate(c, rf, symmetric)?;
        Ok((work, memory))
    }

    /// Estimated operations and memory of one evaluation on sparse blocks
    /// with the rf amplitudes `rf`, and the coefficients propagated,
    /// without building the blocks: from bounds on their entries, or, when
    /// those bounds exceed a limit, from their entries counted column by
    /// column. The full blocks are built before they are reduced, and the
    /// relaxation is held while `L₀` is built.
    pub(super) fn sparse_estimate(
        &self,
        c: &Control,
        rf: Amplitudes<'_>,
        symmetric: bool,
    ) -> Result<(f64, f64, usize)> {
        let estimate = |counted: &Counted| -> (f64, f64, usize) {
            let members = c.members();
            let scale = members
                .iter()
                .flat_map(|m| m.scales.iter().map(|s| s.abs()))
                .fold(1.0f64, f64::max);
            let offset = members
                .iter()
                .flat_map(|m| m.offsets.iter().map(|o| o.abs()))
                .fold(0.0f64, f64::max);
            let base_norm = counted.base_norm + counted.per_offset * offset;
            let substeps_of_step = |step: &[f64]| -> f64 {
                let norm = base_norm
                    + scale
                        * step
                            .iter()
                            .zip(&counted.per_rf)
                            .map(|(x, n)| x.abs() * n)
                            .sum::<f64>();
                substeps_of(norm * c.step()) as f64
            };
            let (total, most) = match rf {
                Amplitudes::Peak(peak) => {
                    let s = substeps_of_step(&vec![peak; c.per_step()]);
                    (s * c.steps as f64, s)
                }
                Amplitudes::Controls(u) => u
                    .chunks(c.per_step())
                    .map(substeps_of_step)
                    .fold((0.0, 0.0f64), |(t, m), s| (t + s, m.max(s))),
            };
            let share = counted.share;
            let m = (counted.n * share).ceil();
            let (base, ops) = (counted.base * share + m, counted.ops * share + m);
            let (member_work, member_bytes) =
                member_cost(c, m, base + ops + m, ops, total, most, true);
            let build = counted.build + 40.0 * (counted.ops + counted.base);
            let work = member_work * members.len() as f64 + build;
            let reduced = share < 1.0;
            let (full_base, full_ops) = (counted.base, counted.ops);
            // L₀ as columns and as rows, with the relaxation; then the
            // directions one by one; then the reduction next to the full
            // blocks; then what stays (with the adjoints).
            let phases = [
                44.0 * full_base + accumulator_bytes(counted.n as usize) + counted.relaxation,
                20.0 * full_base + 20.0 * full_ops + 24.0 * counted.largest_op,
                if reduced {
                    20.0 * (full_base + full_ops) + 44.0 * (base + ops)
                } else {
                    0.0
                },
                20.0 * (2.0 * base + ops * if reduced { 2.0 } else { 1.0 })
                    + 16.0 * m * (c.per_step() as f64 / 2.0 + 2.0),
            ];
            let parallel = (members.len() as f64).min(threads() as f64);
            let memory = phases.into_iter().fold(0.0, f64::max) + parallel * member_bytes;
            (work, memory, m as usize)
        };
        let counted = self.count_sparse(c, symmetric, false)?;
        let (work, memory, m) = estimate(&counted);
        let (work_limit, memory_limit) =
            (super::super::MAX_RELAXATION_WORK, super::MAX_CONTROL_MEMORY);
        // The bounds on the entries (the Redfield part's, and repeated
        // ones) can be twice their number: count them before refusing,
        // unless even a quarter of the estimate exceeds a limit.
        let within = work <= work_limit && memory <= memory_limit;
        let hopeless = work > 4.0 * work_limit || memory > 4.0 * memory_limit;
        if within || hopeless {
            return Ok((work, memory, m));
        }
        Ok(estimate(&self.count_sparse(c, symmetric, true)?))
    }

    /// The sparse blocks of an rf control problem with relaxation, on the
    /// orbits of the declared symmetries when `symmetric` and the
    /// Hamiltonian, every channel, the relaxation and the initial state
    /// are invariant under them (the target need not be).
    pub(super) fn sparse_control(&self, c: &Control, symmetric: bool) -> Result<SparseControl> {
        let Drive::Rf { targets, .. } = &c.drive else {
            return Err(QsoError::invalid(
                "control",
                "sparse blocks take rf controls",
            ));
        };
        let d = self.dimension();
        let n = d * d;
        let h0 = self.hamiltonian_at(self.field());
        let relaxation = self.relaxation(self.field())?;
        let liouville = Liouville::new(&vec![Vec::new(); d], None);
        let base = self.sparse_block(
            &liouville,
            0,
            &Sparse::new(&h0),
            &relaxation,
            ZERO,
            Some(ONE),
        );
        drop(relaxation);
        let mut ops = Vec::with_capacity(2 * targets.len());
        let mut offsets = Vec::with_capacity(targets.len());
        let mut channels = Vec::with_capacity(3 * targets.len());
        let mut exact = true;
        for &t in targets {
            let [fx, fy, fz] = self.channel(t)?;
            for f in [&fx, &fy] {
                let f = f * C64::new(TWO_PI, 0.0);
                exact &= hermitian(&f);
                ops.push(commutator_block(&f));
            }
            let mut z = vec![ZERO; n + 1];
            for a in 0..d {
                for b in 0..d {
                    z[a * d + b] = C64::new(0.0, -TWO_PI * (fz[(a, a)].re - fz[(b, b)].re));
                }
            }
            offsets.push(z);
            channels.extend([fx, fy, fz]);
        }
        let start = augment(&c.initial);
        let target = DVector::from_iterator(
            n + 1,
            super::vectorise(&c.target)
                .iter()
                .copied()
                .chain(std::iter::once(c.target.trace() / d as f64)),
        );
        let mut orbits = None;
        if symmetric
            && let Some(maps) = self.symmetry_maps(&h0)
            && channels.iter().all(|f| invariant(f, &maps))
        {
            let images = element_images(&maps, d);
            let o = Orbits::new(n + 1, &images);
            let largest = start.iter().take(n).fold(0.0f64, |m, z| m.max(z.norm()));
            if o.len() < n + 1
                && o.asymmetry(start.as_slice()) <= SYMMETRY_TOLERANCE * largest
                && Orbits::invariant(&base, &images, SYMMETRY_TOLERANCE * base.largest())
            {
                orbits = Some(o);
            }
        }
        let Some(o) = orbits else {
            let base_adjoint = base.adjoint();
            let base_norm = norm_bound(&base);
            let op_norms = ops.iter().map(norm_bound).collect();
            let ops_adjoint = (!exact).then(|| ops.iter().map(SparseBlock::adjoint).collect());
            return Ok(SparseControl {
                base,
                base_adjoint,
                ops,
                ops_adjoint,
                offsets,
                base_norm,
                op_norms,
                reduced: false,
                quadrature: true,
                start,
                target,
            });
        };
        // The full blocks go before the adjoints are built.
        let reduced = o.reduce(&base);
        drop(base);
        let base = reduced;
        let ops: Vec<SparseBlock> = ops.into_iter().map(|e| o.reduce(&e)).collect();
        let offsets = offsets
            .iter()
            .map(|z| o.representatives.iter().map(|&r| z[r as usize]).collect())
            .collect();
        let mut sums = vec![ZERO; o.len()];
        for (t, &k) in target.iter().zip(&o.of) {
            sums[k as usize] += t;
        }
        Ok(SparseControl {
            base_adjoint: base.adjoint(),
            base_norm: norm_bound(&base),
            op_norms: ops.iter().map(norm_bound).collect(),
            ops_adjoint: Some(ops.iter().map(SparseBlock::adjoint).collect()),
            base,
            ops,
            offsets,
            start: DVector::from_vec(o.average(start.as_slice())),
            target: DVector::from_vec(sums),
            reduced: true,
            quadrature: true,
        })
    }
}
