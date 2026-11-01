//! Continuous-time walks with a time-dependent Hamiltonian
//! `H(t) = Σ_j c_j(t)·H_j`.
//!
//! The terms `H_j` are real symmetric matrices on the nodes (Laplacians,
//! adjacency matrices, projectors, potentials) or the projector onto the
//! uniform superposition; the coefficients `c_j(t)` are any real functions
//! of time, as in adiabatic and driven walks.
//!
//! One step of length `h` is the fourth-order commutator-free Magnus
//! integrator with two exponentials (Blanes and Moan 2006; Alvermann and
//! Fehske 2011):
//!
//! ```text
//! U(t + h, t) = exp(−ih(α₁H₁ + α₂H₂)) · exp(−ih(α₂H₁ + α₁H₂)),
//! H_k = H(t + c_k h),  c₁,₂ = 1/2 ∓ √3/6,  α₁,₂ = 1/4 ∓ √3/6,
//! ```
//!
//! each exponential applied by Krylov propagation. Step sizes are chosen by
//! step doubling: a step is accepted when the difference between one step
//! of `h` and two of `h/2`, divided by 15, is below the tolerance per unit
//! time, and the result of the two half steps is kept. The coefficients
//! are sampled inside each step only, so a discontinuous drive should be
//! split into separate evolutions at its jumps, and a short pulse needs
//! [`MagnusOptions::max_step`].
//!
//! ```
//! use metatron_qso::graph::generators;
//! use metatron_qso::walks::driven::{DrivenWalk, MagnusOptions, Term};
//!
//! // Local adiabatic search on K₆₄ (Roland and Cerf 2002).
//! let n = 64;
//! let walk = DrivenWalk::new(n, vec![Term::Uniform, Term::projector(n, &[7]).unwrap()]).unwrap();
//! let (eps, rn) = (0.1, (n as f64 - 1.0).sqrt());
//! let total = n as f64 / (eps * rn) * rn.atan();
//! let s = |t: f64| 0.5 * (1.0 + ((2.0 * eps * rn * t / n as f64 - rn.atan()).tan()) / rn);
//! let psi0 = vec![num_complex::Complex64::new(1.0 / (n as f64).sqrt(), 0.0); n];
//! let r = walk
//!     .evolve(&psi0, 0.0, total, |t| Ok(vec![-(1.0 - s(t)), -s(t)]), &MagnusOptions::default())
//!     .unwrap();
//! assert!(r.state[7].norm_sqr() > 0.98);
//! ```

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::linalg::{CsrMatrix, KrylovOptions, expm_multiply, norm};
use crate::parallel::for_each_mut;

use super::{Generator, Quotient};

/// Steps accepted at the smallest step size before [`DrivenWalk::evolve`]
/// gives up.
const MAX_FORCED_STEPS: usize = 10_000;

/// Largest walk (nodes).
pub const MAX_DRIVEN_NODES: usize = 1 << 24;

/// A term `H_j` of a driven walk.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Term {
    /// A real symmetric matrix.
    Matrix(CsrMatrix),
    /// The projector `|s⟩⟨s|` onto the uniform superposition of all nodes
    /// (dense, applied in `O(n)`).
    Uniform,
}

impl Term {
    /// The Laplacian (`Generator::Laplacian`) or minus the adjacency matrix
    /// (`Generator::Adjacency`) of `graph`, as for static walks.
    pub fn graph(graph: &Graph, generator: Generator) -> Self {
        Term::Matrix(match generator {
            Generator::Laplacian => graph.laplacian_csr(),
            Generator::Adjacency => graph.adjacency_csr().scaled(-1.0),
        })
    }

    /// The projector `Σ_{j ∈ nodes} |j⟩⟨j|`.
    pub fn projector(n: usize, nodes: &[usize]) -> Result<Self> {
        let mut potential = vec![0.0; n];
        for &j in nodes {
            if j >= n {
                return Err(QsoError::NodeOutOfRange {
                    node: j,
                    n_nodes: n,
                });
            }
            potential[j] = 1.0;
        }
        Self::potential(&potential)
    }

    /// The diagonal matrix `Σ_j v_j |j⟩⟨j|`.
    pub fn potential(values: &[f64]) -> Result<Self> {
        if values.iter().any(|v| !v.is_finite()) {
            return Err(QsoError::invalid("potential", "values must be finite"));
        }
        let triplets: Vec<(usize, usize, f64)> =
            values.iter().enumerate().map(|(j, &v)| (j, j, v)).collect();
        CsrMatrix::from_triplets(values.len(), &triplets).map(Term::Matrix)
    }
}

/// Options of [`DrivenWalk::evolve`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MagnusOptions {
    /// Error tolerance for the whole evolution (norm of the state error).
    pub tol: f64,
    /// First step (default: a hundredth of the interval).
    pub initial_step: Option<f64>,
    /// Fixed step instead of step-size control (the last step is shortened
    /// to end on time).
    pub fixed_step: Option<f64>,
    /// Longest step under step-size control. The error estimate samples
    /// the coefficients inside each step only: a jump or a pulse shorter
    /// than a step can go unnoticed, so set this below the shortest
    /// feature of the drive, or split the evolution at discontinuities.
    pub max_step: Option<f64>,
    /// Most steps, accepted and rejected.
    pub max_steps: usize,
}

impl Default for MagnusOptions {
    fn default() -> Self {
        MagnusOptions {
            tol: 1e-8,
            initial_step: None,
            fixed_step: None,
            max_step: None,
            max_steps: 1_000_000,
        }
    }
}

/// The result of [`DrivenWalk::evolve`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DrivenEvolution {
    pub state: Vec<C64>,
    /// Accepted steps.
    pub steps: usize,
    /// Rejected steps.
    pub rejected: usize,
    /// Sum of the error estimates of the accepted steps (0 with a fixed
    /// step).
    pub error_estimate: f64,
    /// Steps accepted at the smallest step size (`10⁻¹²` of the interval)
    /// although their error estimate exceeded the tolerance.
    pub forced: usize,
}

/// A walk with Hamiltonian `H(t) = Σ_j c_j(t)·H_j`.
#[derive(Clone, Debug)]
pub struct DrivenWalk {
    n: usize,
    /// Union of the sparsity patterns of the matrix terms.
    indptr: Vec<usize>,
    indices: Vec<usize>,
    /// Entries of every term on the pattern (`None` for `Uniform`).
    values: Vec<Option<Vec<f64>>>,
}

const ALPHA: [f64; 2] = [
    0.25 - 0.288_675_134_594_812_9, // 1/4 − √3/6
    0.25 + 0.288_675_134_594_812_9,
];
const NODES: [f64; 2] = [
    0.5 - 0.288_675_134_594_812_9, // 1/2 − √3/6
    0.5 + 0.288_675_134_594_812_9,
];

impl DrivenWalk {
    /// A walk on `n` nodes with the given terms (at least one; matrices
    /// `n × n`, symmetric and finite).
    pub fn new(n: usize, terms: Vec<Term>) -> Result<Self> {
        if n == 0 || n > MAX_DRIVEN_NODES {
            return Err(QsoError::invalid(
                "n",
                format!("a driven walk has 1 to {MAX_DRIVEN_NODES} nodes"),
            ));
        }
        if terms.is_empty() {
            return Err(QsoError::invalid("terms", "at least one term"));
        }
        let mut triplets = Vec::new();
        for (j, t) in terms.iter().enumerate() {
            if let Term::Matrix(m) = t {
                if m.dim() != n {
                    return Err(QsoError::DimensionMismatch {
                        expected: n,
                        actual: m.dim(),
                    });
                }
                if !m.is_symmetric() || (0..n).any(|r| m.row(r).any(|(_, v)| !v.is_finite())) {
                    return Err(QsoError::invalid(
                        "terms",
                        format!("term {j} is not a finite symmetric matrix"),
                    ));
                }
                triplets.extend((0..n).flat_map(|r| m.row(r).map(move |(c, _)| (r, c, 1.0))));
            }
        }
        let pattern = CsrMatrix::from_triplets(n, &triplets)?;
        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::with_capacity(pattern.nnz());
        indptr.push(0);
        for r in 0..n {
            indices.extend(pattern.row(r).map(|(c, _)| c));
            indptr.push(indices.len());
        }
        let values = terms
            .iter()
            .map(|t| match t {
                Term::Uniform => None,
                Term::Matrix(m) => {
                    let mut v = vec![0.0; indices.len()];
                    for r in 0..n {
                        let cols = &indices[indptr[r]..indptr[r + 1]];
                        for (c, x) in m.row(r) {
                            let k = cols.binary_search(&c).expect("in the union");
                            v[indptr[r] + k] = x;
                        }
                    }
                    Some(v)
                }
            })
            .collect();
        Ok(Self {
            n,
            indptr,
            indices,
            values,
        })
    }

    pub fn num_nodes(&self) -> usize {
        self.n
    }

    pub fn num_terms(&self) -> usize {
        self.values.len()
    }

    /// `exp(−i Σ_j w_j H_j)·psi`.
    fn exponential(&self, w: &[f64], psi: &[C64], tol: f64) -> Result<Vec<C64>> {
        let mut combined = vec![0.0; self.indices.len()];
        let mut uniform = 0.0;
        for (v, &wj) in self.values.iter().zip(w) {
            match v {
                Some(v) => combined.iter_mut().zip(v).for_each(|(c, x)| *c += wj * x),
                None => uniform += wj,
            }
        }
        let n = self.n;
        let apply = |x: &[C64], y: &mut [C64]| {
            let shift = if uniform == 0.0 {
                C64::new(0.0, 0.0)
            } else {
                x.iter().sum::<C64>() * (uniform / n as f64)
            };
            for_each_mut(y, |r, out| {
                let mut acc = shift;
                for k in self.indptr[r]..self.indptr[r + 1] {
                    acc += x[self.indices[k]] * combined[k];
                }
                *out = acc;
            });
        };
        let opts = KrylovOptions {
            tol,
            ..KrylovOptions::default()
        };
        expm_multiply(apply, psi, 1.0, &opts)
    }

    /// One fourth-order step of length `h` from time `t`.
    fn step<F>(
        &self,
        psi: &[C64],
        t: f64,
        h: f64,
        coefficients: &mut F,
        tol: f64,
    ) -> Result<Vec<C64>>
    where
        F: FnMut(f64) -> Result<Vec<f64>>,
    {
        let c1 = self.coefficients_at(coefficients, t + NODES[0] * h)?;
        let c2 = self.coefficients_at(coefficients, t + NODES[1] * h)?;
        let first: Vec<f64> = c1
            .iter()
            .zip(&c2)
            .map(|(a, b)| h * (ALPHA[1] * a + ALPHA[0] * b))
            .collect();
        let second: Vec<f64> = c1
            .iter()
            .zip(&c2)
            .map(|(a, b)| h * (ALPHA[0] * a + ALPHA[1] * b))
            .collect();
        let mid = self.exponential(&first, psi, tol)?;
        self.exponential(&second, &mid, tol)
    }

    fn coefficients_at<F>(&self, coefficients: &mut F, t: f64) -> Result<Vec<f64>>
    where
        F: FnMut(f64) -> Result<Vec<f64>>,
    {
        let c = coefficients(t)?;
        if c.len() != self.values.len() {
            return Err(QsoError::DimensionMismatch {
                expected: self.values.len(),
                actual: c.len(),
            });
        }
        if c.iter().any(|x| !x.is_finite()) {
            return Err(QsoError::invalid(
                "coefficients",
                format!("the coefficients at t = {t} are not finite"),
            ));
        }
        Ok(c)
    }

    /// Evolve `psi0` from `t0` to `t1` under `H(t) = Σ_j c_j(t)·H_j`, with
    /// `coefficients(t)` returning `[c_0(t), c_1(t), …]`.
    pub fn evolve<F>(
        &self,
        psi0: &[C64],
        t0: f64,
        t1: f64,
        mut coefficients: F,
        options: &MagnusOptions,
    ) -> Result<DrivenEvolution>
    where
        F: FnMut(f64) -> Result<Vec<f64>>,
    {
        if psi0.len() != self.n {
            return Err(QsoError::DimensionMismatch {
                expected: self.n,
                actual: psi0.len(),
            });
        }
        if psi0.iter().any(|z| !z.is_finite()) {
            return Err(QsoError::invalid("psi0", "the state must be finite"));
        }
        if !(t0.is_finite() && t1.is_finite() && t1 >= t0) {
            return Err(QsoError::invalid("t1", "times must be finite with t1 ≥ t0"));
        }
        if !(options.tol.is_finite() && options.tol > 0.0) {
            return Err(QsoError::invalid("tol", "the tolerance must be positive"));
        }
        let span = t1 - t0;
        let mut psi = psi0.to_vec();
        let mut out = DrivenEvolution {
            state: Vec::new(),
            steps: 0,
            rejected: 0,
            error_estimate: 0.0,
            forced: 0,
        };
        if span == 0.0 {
            out.state = psi;
            return Ok(out);
        }
        let scale = norm(psi0).max(f64::MIN_POSITIVE);
        // Krylov errors well below the integrator's: each exponential gets
        // a hundredth of the step's share of the tolerance (the error of a
        // Krylov propagation is absolute, so shares add up over steps).
        let krylov_tol = |h: f64| (1e-2 * options.tol * h / span).max(1e-15);
        let mut t = t0;
        // Next time point, landing exactly on t1 at the end.
        let advance = |t: f64, h: f64| -> Result<f64> {
            if t1 - (t + h) <= 1e-14 * span {
                Ok(t1)
            } else if t + h > t {
                Ok(t + h)
            } else {
                Err(QsoError::invalid(
                    "step",
                    format!("a step of {h} does not advance t = {t} in floating point"),
                ))
            }
        };
        if let Some(h) = options.fixed_step {
            if !(h.is_finite() && h > 0.0) {
                return Err(QsoError::invalid("fixed_step", "the step must be positive"));
            }
            while t < t1 {
                if out.steps >= options.max_steps {
                    return Err(too_many(out.steps, t1 - t));
                }
                let h = h.min(t1 - t);
                let next = advance(t, h)?;
                psi = self.step(&psi, t, h, &mut coefficients, krylov_tol(h))?;
                t = next;
                out.steps += 1;
            }
            out.state = psi;
            return Ok(out);
        }
        let max_step = options.max_step.unwrap_or(span);
        if !(max_step.is_finite() && max_step > 0.0) {
            return Err(QsoError::invalid("max_step", "the step must be positive"));
        }
        let mut h = options.initial_step.unwrap_or(span / 100.0);
        if !(h.is_finite() && h > 0.0) {
            return Err(QsoError::invalid(
                "initial_step",
                "the step must be positive",
            ));
        }
        let h_min = 1e-12 * span;
        while t < t1 {
            if out.steps + out.rejected >= options.max_steps {
                return Err(too_many(out.steps + out.rejected, t1 - t));
            }
            h = h.min(t1 - t).min(max_step).max(h_min.min(t1 - t));
            let next = advance(t, h)?;
            let tol = krylov_tol(h);
            let full = self.step(&psi, t, h, &mut coefficients, tol)?;
            let half = self.step(&psi, t, h / 2.0, &mut coefficients, tol)?;
            let half = self.step(&half, t + h / 2.0, h / 2.0, &mut coefficients, tol)?;
            let diff: f64 = full
                .iter()
                .zip(&half)
                .map(|(a, b)| (a - b).norm_sqr())
                .sum::<f64>()
                .sqrt();
            let err = diff / 15.0 / scale;
            // Below about 100 ulp the estimate is round-off.
            let allowed = (options.tol * h / span).max(100.0 * f64::EPSILON);
            let forced = err > allowed && h <= h_min;
            if err <= allowed || forced {
                psi = half;
                t = next;
                out.steps += 1;
                out.forced += usize::from(forced);
                if out.forced > MAX_FORCED_STEPS {
                    return Err(QsoError::NoConvergence {
                        method: "Magnus integrator (step size at its minimum)",
                        iterations: out.steps + out.rejected,
                        residual: t1 - t,
                    });
                }
                out.error_estimate += err * scale;
            } else {
                out.rejected += 1;
            }
            // Local error ∝ h⁵ against an allowance ∝ h; growth at most
            // doubles the step, so a drive changing on a new time scale is
            // sampled before it is stepped over.
            let factor = if err == 0.0 {
                2.0
            } else {
                (0.9 * (allowed / err).powf(0.25)).clamp(0.2, 2.0)
            };
            // After a forced step, keep the step (it is already the
            // smallest) instead of shrinking it further.
            h = if forced { h_min } else { h * factor };
        }
        out.state = psi;
        Ok(out)
    }
}

/// A term of a [`QuotientDrivenWalk`], on the cells of a [`Quotient`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum QuotientTerm {
    /// The graph's Laplacian or minus its adjacency matrix.
    Graph(Generator),
    /// The projector onto all nodes of the given cells.
    Cells(Vec<usize>),
    /// A potential constant on each cell (one value per cell).
    Potential(Vec<f64>),
    /// The projector onto the uniform superposition of all nodes.
    Uniform,
}

/// A driven walk `H(t) = Σ_j c_j(t)·H_j` whose terms respect an equitable
/// partition ([`Quotient`]): reduced to the cells, so that the walk of a
/// state in the span of the cell states is exact in as many dimensions as
/// there are cells. Adiabatic search on `K_N` or `Q_d` with a marked node
/// runs for any `N` this way.
///
/// ```
/// use metatron_qso::walks::Quotient;
/// use metatron_qso::walks::driven::{MagnusOptions, QuotientDrivenWalk, QuotientTerm};
///
/// // Local adiabatic search (Roland and Cerf 2002) on K_N, N = 2³⁰, in two
/// // dimensions: the node and the others.
/// let n = (1u64 << 30) as f64;
/// let q = Quotient::complete(1 << 30).unwrap();
/// let walk = QuotientDrivenWalk::new(&q, vec![QuotientTerm::Uniform, QuotientTerm::Cells(vec![0])])
///     .unwrap();
/// let (eps, rn) = (0.1, (n - 1.0).sqrt());
/// let total = n / (eps * rn) * rn.atan();
/// let s = |t: f64| 0.5 * (1.0 + ((2.0 * eps * rn * t / n - rn.atan()).tan()) / rn);
/// let r = walk
///     .evolve(&walk.uniform_state(), 0.0, total, |t| Ok(vec![-(1.0 - s(t)), -s(t)]),
///         &MagnusOptions::default())
///     .unwrap();
/// assert!(walk.cell_probabilities(&r.state).unwrap()[0] > 0.98);
/// ```
#[derive(Clone, Debug)]
pub struct QuotientDrivenWalk {
    quotient: Quotient,
    walk: DrivenWalk,
}

impl QuotientDrivenWalk {
    /// The walk with the given terms on the cells of `quotient` (at most
    /// [`super::DENSE_LIMIT`] cells).
    pub fn new(quotient: &Quotient, terms: Vec<QuotientTerm>) -> Result<Self> {
        quotient.check_dense()?;
        let k = quotient.num_cells();
        let matrices = terms
            .into_iter()
            .map(|t| {
                let triplets: Vec<(usize, usize, f64)> = match t {
                    QuotientTerm::Graph(generator) => {
                        let h = quotient.hamiltonian(generator, 1.0);
                        (0..k)
                            .flat_map(|i| (0..k).map(move |j| (i, j)))
                            .filter(|&(i, j)| h[(i, j)] != 0.0)
                            .map(|(i, j)| (i, j, h[(i, j)]))
                            .collect()
                    }
                    QuotientTerm::Cells(cells) => {
                        for &c in &cells {
                            quotient.check_cell(c)?;
                        }
                        cells.iter().map(|&c| (c, c, 1.0)).collect()
                    }
                    QuotientTerm::Potential(v) => {
                        quotient.check_len(v.len())?;
                        if v.iter().any(|x| !x.is_finite()) {
                            return Err(QsoError::invalid("potential", "values must be finite"));
                        }
                        v.iter().enumerate().map(|(c, &x)| (c, c, x)).collect()
                    }
                    QuotientTerm::Uniform => {
                        let s = quotient.uniform_state();
                        (0..k)
                            .flat_map(|i| (0..k).map(move |j| (i, j)))
                            .map(|(i, j)| (i, j, s[i] * s[j]))
                            .collect()
                    }
                };
                // A cell given twice counts once, so the term stays a projector.
                let mut triplets = triplets;
                triplets.sort_by_key(|&(i, j, _)| (i, j));
                triplets.dedup_by_key(|&mut (i, j, _)| (i, j));
                CsrMatrix::from_triplets(k, &triplets).map(Term::Matrix)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(QuotientDrivenWalk {
            quotient: quotient.clone(),
            walk: DrivenWalk::new(k, matrices)?,
        })
    }

    pub fn quotient(&self) -> &Quotient {
        &self.quotient
    }

    /// The uniform superposition of all nodes, in the cell basis.
    pub fn uniform_state(&self) -> Vec<C64> {
        self.quotient
            .uniform_state()
            .into_iter()
            .map(|x| C64::new(x, 0.0))
            .collect()
    }

    /// Evolve the cell amplitudes `c0` from `t0` to `t1`, as
    /// [`DrivenWalk::evolve`].
    pub fn evolve<F>(
        &self,
        c0: &[C64],
        t0: f64,
        t1: f64,
        coefficients: F,
        options: &MagnusOptions,
    ) -> Result<DrivenEvolution>
    where
        F: FnMut(f64) -> Result<Vec<f64>>,
    {
        self.walk.evolve(c0, t0, t1, coefficients, options)
    }

    /// The probability of every cell.
    pub fn cell_probabilities(&self, state: &[C64]) -> Result<Vec<f64>> {
        self.quotient.check_len(state.len())?;
        Ok(state.iter().map(|z| z.norm_sqr()).collect())
    }
}

fn too_many(steps: usize, remaining: f64) -> QsoError {
    QsoError::NoConvergence {
        method: "Magnus integrator",
        iterations: steps,
        residual: remaining,
    }
}

#[cfg(test)]
mod tests;
