//! Sylvester equations `A X − X A† − σX = C` on the complex Schur form
//! (crate-internal), by Bartels and Stewart (Commun. ACM 15, 820 (1972)).
//!
//! One Schur form `A = Q T Q†` serves both sides of the equation, since
//! `A† = Q T† Q†` with `T†` lower triangular in the same basis: with
//! `Y = Q†XQ` the equation becomes `T Y − Y T† − σY = Q†CQ`, solved column
//! by column from the last,
//!
//! ```text
//! (T − (conj T_kk + σ)) y_k = c_k + Σ_{j>k} y_j conj(T_kj),
//! ```
//!
//! which needs `λ_i − conj λ_k − σ ≠ 0` for every pair of eigenvalues. For
//! `Im λ ≤ 0` and `σ = iγ` the smallest of these is `2·min_i(−Im λ_i) + γ`
//! (see [`SchurForm::slowest_decay`]).
//!
//! * The Schur form is nalgebra's, with the deflation tolerance `1e-14` and
//!   LAPACK's budget of `30·max(10, n)` sweeps. Its default tolerance
//!   `f64::EPSILON` stalls on complete graphs from 16 nodes (the subdiagonal
//!   stays at rounding level), and its `Schur::new` has no budget, so it
//!   never returns there. A cyclic shift stalls at any tolerance; one fixed
//!   Householder similarity `U A U`, `U = I − 2vv†` with `v_k ∝ 1/√(k + 1)`,
//!   rescues it. The vector of ones would not: it is an eigenvector of `K_n`
//!   and of every cyclic shift and their adjoints, so it leaves them
//!   unchanged.
//! * The triangular equation is solved by the recursive blocked algorithm of
//!   Jonsson and Kågström (ACM Trans. Math. Softw. 28, 392 (2002)): split
//!   the larger side in halves, solve the half that does not depend on the
//!   other, update the other by a matrix product and recurse, down to
//!   blocks of at most 48 rows and columns, which are solved directly.
//! * Matrices live in planar form ([`Planar`]: real and imaginary parts
//!   apart), and a complex product is four real products (nalgebra's `gemm`
//!   on `f64` views). Products of at least 2²¹ multiply-adds run in column
//!   panels of fixed width 64 in parallel; every entry is computed by the
//!   same calls whatever the thread count, so results are bit-identical on
//!   any number of threads (and without the `parallel` feature).

use std::ops::Range;

use nalgebra::linalg::Schur;
use nalgebra::{Dyn, MatrixView, MatrixViewMut};
use num_complex::Complex64 as C64;

use super::{CMatrix, RMatrix};
use crate::error::{QsoError, Result};

const ZERO: C64 = C64::new(0.0, 0.0);

/// Deflation tolerance of the Schur iteration (relative to neighbouring
/// diagonal entries); it costs no accuracy against `f64::EPSILON`.
const SCHUR_EPS: f64 = 1e-14;

/// Largest block the recursive triangular solver solves directly.
const LEAF: usize = 48;

/// Columns per panel of a parallel product.
const PANEL: usize = 64;

/// Products of at least this many multiply-adds run in panels in parallel.
const PARALLEL_WORK: usize = 1 << 21;

type View<'a> = MatrixView<'a, f64, Dyn, Dyn, Dyn, Dyn>;
type ViewMut<'a> = MatrixViewMut<'a, f64, Dyn, Dyn, Dyn, Dyn>;

/// Sweeps the Schur iteration may take: 30 per eigenvalue, at least 300.
fn schur_budget(n: usize) -> usize {
    30 * n.max(10)
}

/// A complex matrix as its real and imaginary parts, the layout of the
/// products.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Planar {
    /// The real part.
    pub re: RMatrix,
    /// The imaginary part.
    pub im: RMatrix,
}

impl Planar {
    /// The zero matrix of `rows × cols`.
    pub(crate) fn zeros(rows: usize, cols: usize) -> Self {
        Planar {
            re: RMatrix::zeros(rows, cols),
            im: RMatrix::zeros(rows, cols),
        }
    }

    /// The parts of `c`.
    pub(crate) fn from_complex(c: &CMatrix) -> Self {
        Planar {
            re: c.map(|z| z.re),
            im: c.map(|z| z.im),
        }
    }

    /// The complex matrix.
    pub(crate) fn to_complex(&self) -> CMatrix {
        self.re.zip_map(&self.im, C64::new)
    }

    /// Rows.
    pub(crate) fn nrows(&self) -> usize {
        self.re.nrows()
    }

    /// Columns.
    pub(crate) fn ncols(&self) -> usize {
        self.re.ncols()
    }

    /// The Frobenius norm (summed sequentially).
    pub(crate) fn fro(&self) -> f64 {
        self.re
            .iter()
            .zip(self.im.iter())
            .map(|(r, i)| r * r + i * i)
            .sum::<f64>()
            .sqrt()
    }

    /// `self += W·diag(coef)·W†`, the rank-`m` sum `Σ_j coef_j w_j w_j†`
    /// over the columns `w_j` of the `r × m` matrix `W`: `r²m` multiply-adds
    /// as four real products.
    pub(crate) fn add_rank(&mut self, w: &Planar, coef: &[C64]) {
        let (r, m) = (w.nrows(), w.ncols());
        assert_eq!(coef.len(), m, "one coefficient per column");
        assert_eq!((self.nrows(), self.ncols()), (r, r), "square of W's rows");
        let mut scaled = Planar::zeros(r, m);
        for (j, &c) in coef.iter().enumerate() {
            for i in 0..r {
                let z = C64::new(w.re[(i, j)], w.im[(i, j)]) * c;
                scaled.re[(i, j)] = z.re;
                scaled.im[(i, j)] = z.im;
            }
        }
        gemm_planar(
            self.target(0, 0),
            1.0,
            Operand::plain(&scaled),
            Operand::adjoint(w),
            1.0,
        );
    }

    /// `[w_j† Z w_j]_j` for `Z = self` (`r × r`) and the columns `w_j` of
    /// the `r × m` matrix `W`: `Z·W` by four real products, then one
    /// sequential dot product per column.
    pub(crate) fn diag_reads(&self, w: &Planar) -> Vec<C64> {
        let (r, m) = (w.nrows(), w.ncols());
        assert_eq!((self.nrows(), self.ncols()), (r, r), "square of W's rows");
        let mut zw = Planar::zeros(r, m);
        gemm_planar(
            zw.target(0, 0),
            1.0,
            Operand::plain(self),
            Operand::plain(w),
            0.0,
        );
        (0..m)
            .map(|j| {
                let (wr, wi) = (column(&w.re, j), column(&w.im, j));
                let (pr, pi) = (column(&zw.re, j), column(&zw.im, j));
                let mut acc = ZERO;
                for ((wr, wi), (pr, pi)) in wr.iter().zip(wi).zip(pr.iter().zip(pi)) {
                    acc += C64::new(*wr, -wi) * C64::new(*pr, *pi);
                }
                acc
            })
            .collect()
    }

    /// A copy of the block `rows × cols`.
    fn block(&self, rows: Range<usize>, cols: Range<usize>) -> Planar {
        Planar {
            re: self
                .re
                .view((rows.start, cols.start), (rows.len(), cols.len()))
                .into_owned(),
            im: self
                .im
                .view((rows.start, cols.start), (rows.len(), cols.len()))
                .into_owned(),
        }
    }

    /// The parts from entry `(r0, c0)` on, as the target of a product.
    fn target(&mut self, r0: usize, c0: usize) -> Target<'_> {
        let ld = self.re.nrows();
        let at = r0 + c0 * ld;
        Target {
            re: &mut self.re.as_mut_slice()[at..],
            im: &mut self.im.as_mut_slice()[at..],
            ld,
        }
    }
}

/// Column `j` of a matrix (column-major).
fn column(p: &RMatrix, j: usize) -> &[f64] {
    let r = p.nrows();
    &p.as_slice()[j * r..(j + 1) * r]
}

/// A block of a planar matrix, of its transpose or of its adjoint, as an
/// operand of [`gemm_planar`].
#[derive(Clone, Copy)]
struct Operand<'a> {
    re: &'a [f64],
    im: &'a [f64],
    /// Leading dimension of the stored (column-major) matrix.
    ld: usize,
    /// First row and column of the block in the operand (after
    /// transposition), and its size.
    r0: usize,
    c0: usize,
    rows: usize,
    cols: usize,
    /// The operand is the transpose of the stored matrix.
    trans: bool,
    /// Sign of the imaginary part (−1: the complex conjugate).
    sign: f64,
}

impl<'a> Operand<'a> {
    /// The whole matrix `p`.
    fn plain(p: &'a Planar) -> Self {
        Operand {
            re: p.re.as_slice(),
            im: p.im.as_slice(),
            ld: p.nrows(),
            r0: 0,
            c0: 0,
            rows: p.nrows(),
            cols: p.ncols(),
            trans: false,
            sign: 1.0,
        }
    }

    /// The whole of `p†`.
    fn adjoint(p: &'a Planar) -> Self {
        Operand {
            rows: p.ncols(),
            cols: p.nrows(),
            trans: true,
            sign: -1.0,
            ..Self::plain(p)
        }
    }

    /// The block of `rows × cols` from `(r0, c0)` of this operand.
    fn block(self, r0: usize, c0: usize, rows: usize, cols: usize) -> Self {
        debug_assert!(r0 + rows <= self.rows && c0 + cols <= self.cols);
        Operand {
            r0: self.r0 + r0,
            c0: self.c0 + c0,
            rows,
            cols,
            ..self
        }
    }

    /// One part (`self.re` or `self.im`) of the block as a strided view.
    fn view(&self, data: &'a [f64]) -> View<'a> {
        let (at, rs, cs) = if self.trans {
            (self.c0 + self.r0 * self.ld, self.ld, 1)
        } else {
            (self.r0 + self.c0 * self.ld, 1, self.ld)
        };
        View::from_slice_with_strides(&data[at..], self.rows, self.cols, rs, cs)
    }
}

/// The block of a planar matrix that a product writes, from its first entry
/// on (column-major with leading dimension `ld`).
struct Target<'a> {
    re: &'a mut [f64],
    im: &'a mut [f64],
    ld: usize,
}

/// `C ← βC + α·A·B` for planar operands and the target block `C` of
/// `A.rows × B.cols`: four real products, in column panels of [`PANEL`] in
/// parallel from [`PARALLEL_WORK`] multiply-adds on. Every panel runs the
/// same calls on any number of threads, so the result is bit-identical.
/// `C` is not read when `β = 0`.
fn gemm_planar(c: Target<'_>, alpha: f64, a: Operand<'_>, b: Operand<'_>, beta: f64) {
    let (m, k, n) = (a.rows, a.cols, b.cols);
    debug_assert_eq!(k, b.rows);
    if m == 0 || n == 0 {
        return;
    }
    // nalgebra's `gemm` falls back to column-by-column updates once a
    // dimension is at most 5, and there (nalgebra 0.35, `axcpy_uninit`) it
    // takes the length of a column of `A` with a row stride for its number
    // of entries, reading and writing past both columns. A transposed `A`
    // is therefore copied to column-major storage first (`m·k` entries
    // against `m·k·n` multiply-adds; the products are the same).
    let copy;
    let a = if a.trans {
        copy = (a.view(a.re).into_owned(), a.view(a.im).into_owned());
        Operand {
            re: copy.0.as_slice(),
            im: copy.1.as_slice(),
            ld: m,
            r0: 0,
            c0: 0,
            trans: false,
            ..a
        }
    } else {
        a
    };
    // (Saturating: 2048³ passes usize::MAX on 32-bit targets.)
    if m.saturating_mul(k).saturating_mul(n) < PARALLEL_WORK || n <= PANEL {
        product(c, alpha, a, b, beta);
        return;
    }
    let ld = c.ld;
    let (re, im): (&[f64], &[f64]) = (&*c.re, &*c.im);
    // Each panel into buffers of its own, copied back in order.
    let panels = crate::parallel::map_tasks(n.div_ceil(PANEL), |p| {
        let s = p * PANEL;
        let w = PANEL.min(n - s);
        let (mut pr, mut pi) = (vec![0.0; m * w], vec![0.0; m * w]);
        if beta != 0.0 {
            for j in 0..w {
                let at = (s + j) * ld;
                pr[j * m..(j + 1) * m].copy_from_slice(&re[at..at + m]);
                pi[j * m..(j + 1) * m].copy_from_slice(&im[at..at + m]);
            }
        }
        let target = Target {
            re: &mut pr,
            im: &mut pi,
            ld: m,
        };
        product(target, alpha, a, b.block(0, s, k, w), beta);
        (pr, pi)
    });
    for (p, (pr, pi)) in panels.into_iter().enumerate() {
        let s = p * PANEL;
        for j in 0..pr.len() / m {
            let at = (s + j) * ld;
            c.re[at..at + m].copy_from_slice(&pr[j * m..(j + 1) * m]);
            c.im[at..at + m].copy_from_slice(&pi[j * m..(j + 1) * m]);
        }
    }
}

/// [`gemm_planar`] in one call of each of the four real products.
fn product(c: Target<'_>, alpha: f64, a: Operand<'_>, b: Operand<'_>, beta: f64) {
    debug_assert!(!a.trans, "a transposed A is copied by gemm_planar");
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let mut cr = ViewMut::from_slice_with_strides_mut(c.re, m, n, 1, c.ld);
    let mut ci = ViewMut::from_slice_with_strides_mut(c.im, m, n, 1, c.ld);
    if k == 0 {
        if beta == 0.0 {
            cr.fill(0.0);
            ci.fill(0.0);
        } else {
            cr *= beta;
            ci *= beta;
        }
        return;
    }
    let (ar, ai) = (a.view(a.re), a.view(a.im));
    let (br, bi) = (b.view(b.re), b.view(b.im));
    // (ar + i·sa·ai)(br + i·sb·bi)
    cr.gemm(alpha, &ar, &br, beta);
    cr.gemm(-alpha * a.sign * b.sign, &ai, &bi, 1.0);
    ci.gemm(alpha * b.sign, &ar, &bi, beta);
    ci.gemm(alpha * a.sign, &ai, &br, 1.0);
}

/// nalgebra's Schur form `A = Q T Q†` with [`SCHUR_EPS`] and the budget of
/// [`schur_budget`] (never `Schur::new`, which has no budget).
fn attempt(a: &CMatrix) -> Option<(CMatrix, CMatrix)> {
    Schur::try_new(a.clone(), SCHUR_EPS, schur_budget(a.nrows())).map(Schur::unpack)
}

/// The Householder vector of the retry: `v_k ∝ 1/√(k + 1)`, normalised.
fn householder_vector(n: usize) -> Vec<f64> {
    let v: Vec<f64> = (0..n).map(|k| 1.0 / ((k + 1) as f64).sqrt()).collect();
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.into_iter().map(|x| x / norm).collect()
}

/// The complex Schur form `A = Q T Q†` (`Q` unitary, `T` upper triangular)
/// and the Sylvester solves it serves.
#[derive(Clone, Debug)]
pub(crate) struct SchurForm {
    n: usize,
    q: CMatrix,
    t: CMatrix,
    /// `Q` and `T` in planar form, for the products.
    qp: Planar,
    tp: Planar,
    backward_error: f64,
}

impl SchurForm {
    /// The Schur form of the square matrix `a`: nalgebra's iteration with
    /// the tolerance `1e-14` and `30·max(10, n)` sweeps; if that fails, once
    /// more on `U A U` for the Householder reflection `U = I − 2vv†`,
    /// `v_k ∝ 1/√(k + 1)`, with `Q = U Q′` (see the module docs). Entries
    /// below the diagonal of `T` are set to zero (nalgebra leaves them
    /// exactly zero for complex input), and the backward error
    /// `‖A − QTQ†‖_F/‖A‖_F` is measured once.
    ///
    /// Errors: `DimensionMismatch` for a non-square matrix, an invalid
    /// argument for non-finite entries, `NoConvergence` if both attempts
    /// fail.
    pub(crate) fn new(a: &CMatrix) -> Result<Self> {
        let n = a.nrows();
        if a.ncols() != n {
            return Err(QsoError::DimensionMismatch {
                expected: n,
                actual: a.ncols(),
            });
        }
        if a.iter().any(|z| !(z.re.is_finite() && z.im.is_finite())) {
            return Err(QsoError::invalid(
                "matrix",
                "the Schur form needs finite entries",
            ));
        }
        let (q, mut t) = match attempt(a) {
            Some(qt) => qt,
            None => {
                let v = householder_vector(n);
                let (q, t) = attempt(&reflect(a, &v)).ok_or(QsoError::NoConvergence {
                    method: "complex Schur form",
                    iterations: schur_budget(n),
                    residual: f64::NAN,
                })?;
                // U Q′ = Q′ − 2v(vᵀQ′).
                let vq: Vec<C64> = (0..n)
                    .map(|j| (0..n).map(|i| v[i] * q[(i, j)]).sum())
                    .collect();
                (
                    CMatrix::from_fn(n, n, |i, j| q[(i, j)] - 2.0 * v[i] * vq[j]),
                    t,
                )
            }
        };
        for j in 0..n {
            for i in j + 1..n {
                t[(i, j)] = ZERO;
            }
        }
        let mut form = SchurForm {
            n,
            qp: Planar::from_complex(&q),
            tp: Planar::from_complex(&t),
            q,
            t,
            backward_error: 0.0,
        };
        let rebuilt = form.from_schur(&form.tp);
        // Both norms relative to the largest entry, so that their squares
        // neither underflow (entries below about 1e-154 would give a
        // defect of 0) nor overflow.
        let largest = a
            .iter()
            .map(|z| z.re.abs().max(z.im.abs()))
            .fold(0.0, f64::max);
        let defect = rebuilt - a;
        form.backward_error = if largest > 0.0 {
            defect.unscale(largest).norm() / a.unscale(largest).norm()
        } else {
            defect.norm()
        };
        Ok(form)
    }

    /// The dimension `n`.
    pub(crate) fn dim(&self) -> usize {
        self.n
    }

    /// The unitary factor `Q`.
    pub(crate) fn q(&self) -> &CMatrix {
        &self.q
    }

    /// The upper triangular factor `T`.
    pub(crate) fn t(&self) -> &CMatrix {
        &self.t
    }

    /// `‖A − QTQ†‖_F/‖A‖_F` (`‖A − QTQ†‖_F` for `A = 0`).
    pub(crate) fn backward_error(&self) -> f64 {
        self.backward_error
    }

    /// `min_i(−Im T_ii)`, the slowest decay of `exp(−iAt)` (`+∞` for
    /// `n = 0`). For `Im λ ≤ 0` and `σ = iγ`, `2·slowest_decay + γ` is the
    /// smallest `|λ_i − conj λ_j − σ|`, the gap of the Sylvester operator.
    pub(crate) fn slowest_decay(&self) -> f64 {
        (0..self.n)
            .map(|i| -self.t[(i, i)].im)
            .fold(f64::INFINITY, f64::min)
    }

    /// `Q†CQ` (two products).
    pub(crate) fn to_schur(&self, c: &CMatrix) -> Planar {
        let n = self.n;
        assert_eq!(c.shape(), (n, n), "an n × n matrix");
        let c = Planar::from_complex(c);
        let mut w = Planar::zeros(n, n);
        gemm_planar(
            w.target(0, 0),
            1.0,
            Operand::adjoint(&self.qp),
            Operand::plain(&c),
            0.0,
        );
        let mut y = Planar::zeros(n, n);
        gemm_planar(
            y.target(0, 0),
            1.0,
            Operand::plain(&w),
            Operand::plain(&self.qp),
            0.0,
        );
        y
    }

    /// `QYQ†` (two products).
    // The inverse of `to_schur`: a change of basis, not a constructor.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn from_schur(&self, y: &Planar) -> CMatrix {
        let n = self.n;
        assert_eq!((y.nrows(), y.ncols()), (n, n), "an n × n matrix");
        let mut w = Planar::zeros(n, n);
        gemm_planar(
            w.target(0, 0),
            1.0,
            Operand::plain(&self.qp),
            Operand::plain(y),
            0.0,
        );
        let mut x = Planar::zeros(n, n);
        gemm_planar(
            x.target(0, 0),
            1.0,
            Operand::plain(&w),
            Operand::adjoint(&self.qp),
            0.0,
        );
        x.to_complex()
    }

    /// Solve `T Y − Y T† − σY = C` in place (`y` holds `C` on entry and `Y`
    /// on return), by the recursive blocked algorithm. The caller ensures
    /// `T_ii − conj T_kk − σ ≠ 0` (see [`Self::slowest_decay`]).
    pub(crate) fn solve_triangular(&self, y: &mut Planar, sigma: C64) {
        self.solve_blocked(y, sigma, LEAF);
    }

    /// `T Y − Y T† − σY` (two products; for residuals).
    pub(crate) fn apply_triangular(&self, y: &Planar, sigma: C64) -> Planar {
        let n = self.n;
        assert_eq!((y.nrows(), y.ncols()), (n, n), "an n × n matrix");
        let mut p = Planar::zeros(n, n);
        gemm_planar(
            p.target(0, 0),
            1.0,
            Operand::plain(&self.tp),
            Operand::plain(y),
            0.0,
        );
        gemm_planar(
            p.target(0, 0),
            -1.0,
            Operand::plain(y),
            Operand::adjoint(&self.tp),
            1.0,
        );
        for (((pr, pi), yr), yi) in
            p.re.iter_mut()
                .zip(p.im.iter_mut())
                .zip(y.re.iter())
                .zip(y.im.iter())
        {
            *pr -= sigma.re * yr - sigma.im * yi;
            *pi -= sigma.re * yi + sigma.im * yr;
        }
        p
    }

    /// `X` with `A X − X A† − σX = C`: [`Self::to_schur`],
    /// [`Self::solve_triangular`], [`Self::from_schur`].
    pub(crate) fn solve(&self, c: &CMatrix, sigma: C64) -> CMatrix {
        let mut y = self.to_schur(c);
        self.solve_triangular(&mut y, sigma);
        self.from_schur(&y)
    }

    /// [`Self::solve_triangular`] with leaves of at most `leaf` rows and
    /// columns.
    fn solve_blocked(&self, y: &mut Planar, sigma: C64, leaf: usize) {
        let n = self.n;
        assert_eq!((y.nrows(), y.ncols()), (n, n), "an n × n matrix");
        self.solve_block(y, 0..n, 0..n, sigma, leaf.max(1));
    }

    /// The block `rows × cols` of `T Y − Y T† − σY = C`, the contributions
    /// of the rows below and the columns right of it already moved into `C`:
    /// `T_rr Y_rc − Y_rc (T†)_cc − σY_rc = C_rc`.
    fn solve_block(
        &self,
        y: &mut Planar,
        rows: Range<usize>,
        cols: Range<usize>,
        sigma: C64,
        leaf: usize,
    ) {
        let (m, n) = (rows.len(), cols.len());
        if m == 0 || n == 0 {
            return;
        }
        if m <= leaf && n <= leaf {
            self.solve_leaf(y, rows, cols, sigma);
            return;
        }
        if m >= n {
            // T_rr = [T11 T12; 0 T22]: the lower rows first, then
            // C1 −= T12·Y2.
            let h = rows.start + m / 2;
            self.solve_block(y, h..rows.end, cols.clone(), sigma, leaf);
            let y2 = y.block(h..rows.end, cols.clone());
            let t12 = Operand::plain(&self.tp).block(rows.start, h, h - rows.start, rows.end - h);
            gemm_planar(
                y.target(rows.start, cols.start),
                -1.0,
                t12,
                Operand::plain(&y2),
                1.0,
            );
            self.solve_block(y, rows.start..h, cols, sigma, leaf);
        } else {
            // (T†)_cc = [B11 0; B21 B22]: the right columns first, then
            // C1 += Y2·B21.
            let h = cols.start + n / 2;
            self.solve_block(y, rows.clone(), h..cols.end, sigma, leaf);
            let y2 = y.block(rows.clone(), h..cols.end);
            let b21 = Operand::adjoint(&self.tp).block(h, cols.start, cols.end - h, h - cols.start);
            gemm_planar(
                y.target(rows.start, cols.start),
                1.0,
                Operand::plain(&y2),
                b21,
                1.0,
            );
            self.solve_block(y, rows, cols.start..h, sigma, leaf);
        }
    }

    /// A block solved directly, column by column from the last, in complex
    /// arithmetic.
    fn solve_leaf(&self, y: &mut Planar, rows: Range<usize>, cols: Range<usize>, sigma: C64) {
        let (a0, b0) = (rows.start, cols.start);
        let (m, n, ld) = (rows.len(), cols.len(), self.n);
        let t = self.t.as_slice();
        let mut c: Vec<C64> = Vec::with_capacity(m * n);
        for k in 0..n {
            let at = a0 + (b0 + k) * ld;
            let (re, im) = (&y.re.as_slice()[at..at + m], &y.im.as_slice()[at..at + m]);
            c.extend(re.iter().zip(im).map(|(&r, &i)| C64::new(r, i)));
        }
        for k in (0..n).rev() {
            let (left, right) = c.split_at_mut((k + 1) * m);
            let col = &mut left[k * m..];
            // + Σ_{j>k} y_j conj(T[b0 + k, b0 + j]).
            for j in k + 1..n {
                let f = t[(b0 + k) + (b0 + j) * ld].conj();
                if f != ZERO {
                    let yj = &right[(j - k - 1) * m..(j - k) * m];
                    for (r, y) in col.iter_mut().zip(yj) {
                        *r += f * y;
                    }
                }
            }
            let s = t[(b0 + k) * (ld + 1)].conj() + sigma;
            for i in (0..m).rev() {
                let v = col[i] / (t[(a0 + i) * (ld + 1)] - s);
                col[i] = v;
                // Column a0 + i of T above the diagonal, within the block.
                let above = &t[a0 + (a0 + i) * ld..a0 + i + (a0 + i) * ld];
                for (r, t) in col[..i].iter_mut().zip(above) {
                    *r -= v * t;
                }
            }
        }
        for k in 0..n {
            let at = a0 + (b0 + k) * ld;
            let solved = &c[k * m..(k + 1) * m];
            for (dst, z) in y.re.as_mut_slice()[at..at + m].iter_mut().zip(solved) {
                *dst = z.re;
            }
            for (dst, z) in y.im.as_mut_slice()[at..at + m].iter_mut().zip(solved) {
                *dst = z.im;
            }
        }
    }
}

/// `U A U` for the reflection `U = I − 2vvᵀ` (real unit `v`):
/// `A − 2v(vᵀA) − 2(Av)vᵀ + 4(vᵀAv)vvᵀ`, in `O(n²)`.
fn reflect(a: &CMatrix, v: &[f64]) -> CMatrix {
    let n = a.nrows();
    let av: Vec<C64> = (0..n)
        .map(|i| (0..n).map(|j| a[(i, j)] * v[j]).sum())
        .collect();
    let va: Vec<C64> = (0..n)
        .map(|j| (0..n).map(|i| v[i] * a[(i, j)]).sum())
        .collect();
    let vav: C64 = (0..n).map(|i| v[i] * av[i]).sum();
    CMatrix::from_fn(n, n, |i, j| {
        a[(i, j)] - 2.0 * v[i] * va[j] - 2.0 * v[j] * av[i] + 4.0 * v[i] * v[j] * vav
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn random_complex(rows: usize, cols: usize, rng: &mut Rng) -> CMatrix {
        CMatrix::from_fn(rows, cols, |_, _| C64::new(rng.normal(), rng.normal()))
    }

    /// `H − iD` as in transport: `H` real symmetric (random weights on a
    /// random graph, site energies), `D` a positive diagonal, so every
    /// eigenvalue decays: complex symmetric and not normal.
    fn transport_like(n: usize, seed: u64) -> CMatrix {
        let mut rng = Rng::seed_from_u64(seed);
        let mut a = CMatrix::zeros(n, n);
        for i in 0..n {
            for j in i + 1..n {
                if j == i + 1 || rng.uniform() < 4.0 / n as f64 {
                    let w = C64::new(2.0 * rng.uniform() - 1.0, 0.0);
                    a[(i, j)] = w;
                    a[(j, i)] = w;
                }
            }
            a[(i, i)] = C64::new(3.0 * (rng.uniform() - 0.5), -(0.1 + 0.9 * rng.uniform()));
        }
        a
    }

    fn complete(n: usize) -> CMatrix {
        CMatrix::from_fn(n, n, |i, j| if i == j { ZERO } else { C64::new(-1.0, 0.0) })
    }

    fn shift(n: usize) -> CMatrix {
        CMatrix::from_fn(n, n, |i, j| {
            if i == (j + 1) % n {
                C64::new(1.0, 0.0)
            } else {
                ZERO
            }
        })
    }

    fn relative(a: &CMatrix, b: &CMatrix) -> f64 {
        (a - b).norm() / b.norm()
    }

    #[test]
    fn schur_forms_of_complete_graphs_and_cyclic_shifts() {
        let mut trapped = complete(64);
        trapped[(1, 1)] -= C64::new(0.0, 1.0);
        let mut cases = vec![("K16", complete(16), true), ("K64", complete(64), true)];
        cases.push(("K64 + trap", trapped, true));
        // A cyclic shift stalls the first attempt; the retry rescues it.
        for n in [8, 16, 64] {
            cases.push(("shift", shift(n), false));
        }
        for (name, a, first) in cases {
            let n = a.nrows();
            let raw = attempt(&a);
            assert_eq!(raw.is_some(), first, "{name} {n}");
            if let Some((_, t)) = raw {
                // nalgebra's T is exactly upper triangular.
                assert!(
                    (0..n).all(|j| (j + 1..n).all(|i| t[(i, j)] == ZERO)),
                    "{name}"
                );
            }
            let s = SchurForm::new(&a).unwrap();
            assert!(
                s.backward_error() <= 1e-13,
                "{name} {n}: {:e}",
                s.backward_error()
            );
            let gram = s.q().adjoint() * s.q() - CMatrix::identity(n, n);
            assert!(
                gram.norm() <= 1e-13 * n as f64,
                "{name} {n}: {:e}",
                gram.norm()
            );
            assert!((0..n).all(|j| (j + 1..n).all(|i| s.t()[(i, j)] == ZERO)));
            assert_eq!(s.dim(), n);
        }
    }

    #[test]
    fn schur_forms_report_decay_and_refuse_bad_input() {
        let a = CMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![
            C64::new(1.0, -0.5),
            C64::new(2.0, -0.1),
            C64::new(0.0, -0.3),
        ]));
        let s = SchurForm::new(&a).unwrap();
        assert!((s.slowest_decay() - 0.1).abs() < 1e-15);
        assert!(s.backward_error() < 1e-15);
        let zero = SchurForm::new(&CMatrix::zeros(4, 4)).unwrap();
        assert_eq!((zero.backward_error(), zero.slowest_decay()), (0.0, 0.0));
        assert!(matches!(
            SchurForm::new(&CMatrix::zeros(3, 4)),
            Err(QsoError::DimensionMismatch { .. })
        ));
        let mut bad = complete(5);
        bad[(2, 3)] = C64::new(f64::NAN, 0.0);
        assert!(matches!(
            SchurForm::new(&bad),
            Err(QsoError::InvalidArgument { .. })
        ));
    }

    /// `I ⊗ A − conj(A) ⊗ I − σI`: `X ↦ AX − XA† − σX` on column-major
    /// `vec X`.
    fn kronecker(a: &CMatrix, sigma: C64) -> CMatrix {
        let n = a.nrows();
        let mut k = CMatrix::zeros(n * n, n * n);
        for j in 0..n {
            for i in 0..n {
                for l in 0..n {
                    k[(j * n + i, j * n + l)] += a[(i, l)];
                    k[(j * n + i, l * n + i)] -= a[(j, l)].conj();
                }
                k[(j * n + i, j * n + i)] -= sigma;
            }
        }
        k
    }

    #[test]
    fn bartels_stewart_matches_a_kronecker_solve() {
        // The dense LU of the n² × n² Kronecker matrix costs O(n⁶): up to
        // n = 21 it is the reference; at 40 (still one leaf) the residual.
        let mut rng = Rng::seed_from_u64(11);
        for n in [8, 13, 21, 40] {
            // Transport-type and general complex matrices, every
            // eigenvalue in the lower half-plane.
            let general = random_complex(n, n, &mut rng) / C64::new((n as f64).sqrt(), 0.0)
                - CMatrix::identity(n, n) * C64::new(0.0, 1.5);
            for a in [transport_like(n, n as u64), general] {
                let s = SchurForm::new(&a).unwrap();
                let c = random_complex(n, n, &mut rng);
                for sigma in [ZERO, C64::new(0.0, 0.7)] {
                    let x = s.solve(&c, sigma);
                    let back = &a * &x - &x * a.adjoint() - &x * sigma;
                    assert!(relative(&back, &c) <= 1e-13, "{n}");
                    if n > 21 {
                        continue;
                    }
                    let lu = kronecker(&a, sigma).lu();
                    let want = lu
                        .solve(&nalgebra::DVector::from_column_slice(c.as_slice()))
                        .unwrap();
                    let want = CMatrix::from_column_slice(n, n, want.as_slice());
                    let e = relative(&x, &want);
                    assert!(e <= 1e-12, "n = {n}, σ = {sigma}: {e:e}");
                }
            }
        }
    }

    #[test]
    fn the_recursive_solver_equals_the_leaf_solver() {
        // 130 = 2·65: the recursion splits rows and columns down to leaves.
        let n = 130;
        let s = SchurForm::new(&transport_like(n, 5)).unwrap();
        let mut rng = Rng::seed_from_u64(6);
        let c = Planar::from_complex(&random_complex(n, n, &mut rng));
        for sigma in [ZERO, C64::new(0.0, 0.7)] {
            let mut blocked = c.clone();
            s.solve_triangular(&mut blocked, sigma);
            let mut direct = c.clone();
            s.solve_blocked(&mut direct, sigma, usize::MAX);
            let (blocked, direct) = (blocked.to_complex(), direct.to_complex());
            assert!(
                relative(&blocked, &direct) <= 1e-13,
                "{:e}",
                relative(&blocked, &direct)
            );
            // The triangular residual.
            let back = s.apply_triangular(&Planar::from_complex(&blocked), sigma);
            assert!(relative(&back.to_complex(), &c.to_complex()) <= 1e-13);
        }
    }

    #[test]
    fn hermitian_right_hand_sides_give_hermitian_solutions() {
        // C = −i·M with M Hermitian and σ = iγ: X† solves the same equation.
        let n = 60;
        let a = transport_like(n, 9);
        let s = SchurForm::new(&a).unwrap();
        let mut rng = Rng::seed_from_u64(10);
        let m = random_complex(n, n, &mut rng);
        let mut start = CMatrix::zeros(n, n);
        start[(0, 0)] = C64::new(1.0, 0.0);
        for m in [&m + m.adjoint(), start] {
            for gamma in [0.0, 0.7] {
                let x = s.solve(&(m.clone() * C64::new(0.0, -1.0)), C64::new(0.0, gamma));
                let e = relative(&x.adjoint(), &x);
                assert!(e <= 1e-14, "{e:e}");
            }
        }
    }

    #[test]
    fn planar_helpers_match_complex_arithmetic() {
        let mut rng = Rng::seed_from_u64(12);
        let (r, m) = (70, 9);
        let z = random_complex(r, r, &mut rng);
        let w = random_complex(r, m, &mut rng);
        let coef: Vec<C64> = (0..m)
            .map(|_| C64::new(rng.normal(), rng.normal()))
            .collect();
        let (zp, wp) = (Planar::from_complex(&z), Planar::from_complex(&w));
        assert_eq!(zp.to_complex(), z);
        assert!((zp.fro() - z.norm()).abs() <= 1e-13 * z.norm());
        let mut sum = zp.clone();
        sum.add_rank(&wp, &coef);
        let d = CMatrix::from_diagonal(&nalgebra::DVector::from_vec(coef));
        let want = &z + &w * d * w.adjoint();
        assert!(relative(&sum.to_complex(), &want) <= 1e-14);
        let reads = zp.diag_reads(&wp);
        for (j, read) in reads.iter().enumerate() {
            let col = w.column(j);
            let want = (col.adjoint() * &z * col)[(0, 0)];
            assert!((read - want).norm() <= 1e-12 * (1.0 + want.norm()), "{j}");
        }
        // Products of every shape and in panels against nalgebra's complex
        // product, including the adjoint operand and a block target.
        for (rows, inner, cols) in [(3, 0, 4), (1, 5, 1), (7, 3, 130), (150, 150, 150)] {
            let a = random_complex(rows, inner, &mut rng);
            let b = random_complex(cols, inner, &mut rng);
            let mut c = random_complex(rows + 2, cols + 1, &mut rng);
            let mut want = c.clone();
            let block = &a * b.adjoint();
            for j in 0..cols {
                for i in 0..rows {
                    want[(i + 2, j + 1)] = 0.5 * want[(i + 2, j + 1)] - 2.0 * block[(i, j)];
                }
            }
            let (ap, bp) = (Planar::from_complex(&a), Planar::from_complex(&b));
            let mut cp = Planar::from_complex(&c);
            gemm_planar(
                cp.target(2, 1),
                -2.0,
                Operand::plain(&ap),
                Operand::adjoint(&bp),
                0.5,
            );
            c = cp.to_complex();
            assert!(
                (&c - &want).norm() <= 1e-12 * (1.0 + want.norm()),
                "{rows} {inner} {cols}"
            );
        }
    }

    #[test]
    fn small_matrices_and_leaf_boundaries_solve_exactly() {
        // Up to 5 rows or columns nalgebra's product takes a path that must
        // never see a left operand with a row stride (`Q†` in `to_schur`:
        // it used to read and write past the columns); 48 and 49 lie on
        // either side of a leaf.
        let mut rng = Rng::seed_from_u64(14);
        for n in [1, 2, 3, 4, 5, 6, 48, 49] {
            let a = transport_like(n, 20 + n as u64);
            let s = SchurForm::new(&a).unwrap();
            let c = random_complex(n, n, &mut rng);
            for sigma in [ZERO, C64::new(0.0, 0.7)] {
                let x = s.solve(&c, sigma);
                let back = &a * &x - &x * a.adjoint() - &x * sigma;
                assert!(relative(&back, &c) <= 1e-13, "{n}");
                if n <= 6 {
                    let want = kronecker(&a, sigma)
                        .lu()
                        .solve(&nalgebra::DVector::from_column_slice(c.as_slice()))
                        .unwrap();
                    let want = CMatrix::from_column_slice(n, n, want.as_slice());
                    assert!(relative(&x, &want) <= 1e-12, "{n}");
                }
            }
        }
        // Small products with an adjoint on the left.
        for (rows, inner, cols) in [(1, 1, 1), (2, 3, 4), (5, 5, 5), (5, 9, 70), (3, 70, 2)] {
            let a = random_complex(inner, rows, &mut rng);
            let b = random_complex(inner, cols, &mut rng);
            let mut c = Planar::zeros(rows, cols);
            gemm_planar(
                c.target(0, 0),
                1.0,
                Operand::adjoint(&Planar::from_complex(&a)),
                Operand::plain(&Planar::from_complex(&b)),
                0.0,
            );
            let want = a.adjoint() * &b;
            assert!(
                relative(&c.to_complex(), &want) <= 1e-14,
                "{rows} {inner} {cols}"
            );
        }
    }

    #[test]
    fn the_backward_error_of_tiny_matrices_does_not_underflow() {
        // Entries of 1e-160: the squares of the defect's entries (about
        // 1e-176) lie below the smallest number, which made the error 0.
        let a = transport_like(20, 3);
        let unit = SchurForm::new(&a).unwrap().backward_error();
        let tiny = SchurForm::new(&(&a * C64::new(1e-160, 0.0)))
            .unwrap()
            .backward_error();
        assert!(unit > 0.0 && unit <= 1e-13, "{unit:e}");
        assert!(tiny > 0.0 && tiny <= 1e-13, "{tiny:e}");
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn planar_products_do_not_depend_on_the_thread_count() {
        let n = 300;
        let mut rng = Rng::seed_from_u64(13);
        let a = random_complex(n, n, &mut rng) / C64::new((n as f64).sqrt(), 0.0)
            - CMatrix::identity(n, n) * C64::new(0.0, 1.5);
        let s = SchurForm::new(&a).unwrap();
        let c = random_complex(n, n, &mut rng);
        let w = Planar::from_complex(&random_complex(n, 40, &mut rng));
        let coef: Vec<C64> = (0..40).map(|k| C64::new(k as f64, 1.0)).collect();
        let sigma = C64::new(0.0, 0.7);
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let mut y = s.to_schur(&c);
                    s.solve_triangular(&mut y, sigma);
                    let residual = s.apply_triangular(&y, sigma);
                    let mut summed = y.clone();
                    summed.add_rank(&w, &coef);
                    (y.clone(), residual, summed.diag_reads(&w), s.from_schur(&y))
                })
        };
        let one = run(1);
        let four = run(4);
        let bits = |p: &Planar| -> Vec<u64> {
            p.re.iter()
                .chain(p.im.iter())
                .map(|x| x.to_bits())
                .collect()
        };
        let complex_bits = |z: &[C64]| -> Vec<(u64, u64)> {
            z.iter().map(|z| (z.re.to_bits(), z.im.to_bits())).collect()
        };
        assert_eq!(bits(&one.0), bits(&four.0));
        assert_eq!(bits(&one.1), bits(&four.1));
        assert_eq!(complex_bits(&one.2), complex_bits(&four.2));
        assert_eq!(
            complex_bits(one.3.as_slice()),
            complex_bits(four.3.as_slice())
        );
    }
}
