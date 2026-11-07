//! The dense transport solver (crate-internal): `X = ∫ρ dt` and
//! `Y = ∫tρ dt` of an open walk on at most 2048 nodes, without integrating
//! in time.
//!
//! 1. **Closure.** The bright space `B` is the smallest space that holds the
//!    traps and the nodes with more than the smallest loss `Γ₀` and is
//!    closed under `H` and under `P_j` for every dephased node it overlaps.
//!    The uniform loss `Γ₀` is a shift of `H_eff`, not a seed. Without
//!    dephasing on every node the closure runs on `H` centred and scaled to
//!    `(H − cI)/s` (otherwise a large common energy offset would make
//!    bright states look dark under the relative tolerance), by two-pass
//!    Gram–Schmidt; with dephasing on every node it is the node span of the
//!    connected components that hold a seed.
//! 2. **Schur form** of
//!    `H′_B = Vᵀ(H − cI)V − i·Vᵀdiag(κ + Γ′/2 + γ/2)V − (i/2)Γ₀I`
//!    (`Γ′ = Γ − Γ₀`; the centre `c` of the diagonal of `H` leaves every
//!    commutator unchanged and keeps a large common energy offset from
//!    costing digits). A uniform added dephasing `e` shifts `H′_B` by
//!    `−(ie/2)I`, so one Schur form serves every added rate.
//! 3. **Split solve** of `𝓛(Z) = −C`, written `𝓢(Z) = C + 𝓙(Z)` with the
//!    coherent and decaying part `𝓢(Z) = i(H′Z − ZH′†) + eZ` (a triangular
//!    Sylvester equation on the Schur form) and the dephasing feed
//!    `𝓙(Z) = Σ_k γ̃_k (q_k†Zq_k) q_kq_k†`: GMRES runs over the populations
//!    `q_k†Zq_k` of the dephased bright nodes only, preconditioned by the
//!    classical (incoherent hopping) rate matrix when every node is
//!    dephased.
//! 4. **Refinement** while the residual `𝓛(X) + ρ₀` is above the tolerance
//!    (at most twice), then the read-outs.

use nalgebra::DVector;
use nalgebra::linalg::LU;
use num_complex::Complex64 as C64;

use super::{MAX_TRANSPORT_WORK, SolveReport, TransportOptions, TransportResult, arrival_time};
use crate::error::{QsoError, Result};
use crate::linalg::dense::cmul;
use crate::linalg::gmres::{GmresEnd, GmresOptions, gmres};
use crate::linalg::sylvester::{Planar, SchurForm};
use crate::linalg::{CMatrix, RMatrix};

const ZERO: C64 = C64::new(0.0, 0.0);

/// Restart length of the population GMRES at most.
const RESTART: usize = 200;

/// Relative breakdown test of the population GMRES.
const BREAKDOWN: f64 = 1e-14;

/// Estimate of `‖I − GΓ̃‖` for the backward-error rule of the population
/// GMRES.
const POPULATION_SCALE: f64 = 2.0;

/// Refinement steps after the first solve at most.
const REFINEMENTS: usize = 2;

/// An open walk as the dense solver sees it.
#[derive(Clone, Debug)]
pub(crate) struct DenseModel {
    /// `H` (real symmetric, site energies on the diagonal).
    pub(crate) h: RMatrix,
    /// Traps `(node, κ)`.
    pub(crate) traps: Vec<(usize, f64)>,
    /// Loss rates `Γ_j`.
    pub(crate) losses: Vec<f64>,
    /// The walk's own dephasing rates `γ_j`.
    pub(crate) dephasing: Vec<f64>,
}

/// Where the walker starts.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Start<'a> {
    /// On one node.
    Node(usize),
    /// In a density matrix (Hermitian, trace 1).
    Density(&'a CMatrix),
}

/// An orthonormal basis `V` (`n × r`) of the bright space.
#[derive(Clone, Debug)]
enum Basis {
    /// Node vectors of these nodes (ascending).
    Nodes(Vec<usize>),
    /// General columns.
    Dense(RMatrix),
}

impl Basis {
    /// The dimension `r`.
    fn dim(&self) -> usize {
        match self {
            Basis::Nodes(nodes) => nodes.len(),
            Basis::Dense(v) => v.ncols(),
        }
    }

    /// `Vᵀe_j`.
    fn row(&self, j: usize) -> Vec<f64> {
        match self {
            Basis::Nodes(nodes) => {
                let mut q = vec![0.0; nodes.len()];
                if let Ok(p) = nodes.binary_search(&j) {
                    q[p] = 1.0;
                }
                q
            }
            Basis::Dense(v) => v.row(j).iter().copied().collect(),
        }
    }

    /// `VᵀAV` for a real `n × n` matrix given by its nonzero rows.
    fn project(&self, rows: &[Vec<(usize, f64)>], n: usize) -> RMatrix {
        match self {
            Basis::Nodes(nodes) => {
                let r = nodes.len();
                let mut out = RMatrix::zeros(r, r);
                for (a, &u) in nodes.iter().enumerate() {
                    for &(v, x) in &rows[u] {
                        if let Ok(b) = nodes.binary_search(&v) {
                            out[(a, b)] = x;
                        }
                    }
                }
                out
            }
            Basis::Dense(v) => {
                let r = v.ncols();
                let mut av = RMatrix::zeros(n, r);
                for c in 0..r {
                    for (u, row) in rows.iter().enumerate() {
                        let mut acc = 0.0;
                        for &(w, x) in row {
                            acc += x * v[(w, c)];
                        }
                        av[(u, c)] = acc;
                    }
                }
                v.transpose() * av
            }
        }
    }

    /// `Vᵀdiag(d)V`.
    fn project_diagonal(&self, d: &[f64]) -> RMatrix {
        match self {
            Basis::Nodes(nodes) => {
                let r = nodes.len();
                let mut out = RMatrix::zeros(r, r);
                for (a, &u) in nodes.iter().enumerate() {
                    out[(a, a)] = d[u];
                }
                out
            }
            Basis::Dense(v) => {
                let mut dv = v.clone();
                for (u, &x) in d.iter().enumerate() {
                    for c in 0..v.ncols() {
                        dv[(u, c)] *= x;
                    }
                }
                v.transpose() * dv
            }
        }
    }

    /// `VᵀρV` for a complex `n × n` matrix.
    fn project_complex(&self, rho: &CMatrix) -> CMatrix {
        match self {
            Basis::Nodes(nodes) => {
                let r = nodes.len();
                CMatrix::from_fn(r, r, |a, b| rho[(nodes[a], nodes[b])])
            }
            Basis::Dense(v) => {
                let vt = v.transpose();
                let re = &vt * rho.map(|z| z.re) * v;
                let im = &vt * rho.map(|z| z.im) * v;
                re.zip_map(&im, C64::new)
            }
        }
    }
}

/// `Σ a_i b_i`, summed in order.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `‖a‖₂`, summed in order.
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Add `w` to the orthonormal `basis` (two passes of Gram–Schmidt) if
/// more than `tolerance` of its norm is left, and queue its index.
fn extend(
    mut w: Vec<f64>,
    basis: &mut Vec<Vec<f64>>,
    queue: &mut std::collections::VecDeque<usize>,
    tolerance: f64,
) {
    let before = norm(&w);
    if before == 0.0 {
        return;
    }
    for _ in 0..2 {
        for b in basis.iter() {
            let d = dot(b, &w);
            for (x, y) in w.iter_mut().zip(b) {
                *x -= d * y;
            }
        }
    }
    let after = norm(&w);
    if after > tolerance * before {
        queue.push_back(basis.len());
        basis.push(w.into_iter().map(|x| x / after).collect());
    }
}

/// The centre `c = (max_j H_jj + min_j H_jj)/2` of the diagonal of `h`.
fn centre(h: &RMatrix) -> f64 {
    let diag = (0..h.nrows()).map(|j| h[(j, j)]);
    0.5 * (diag.clone().fold(f64::NEG_INFINITY, f64::max) + diag.fold(f64::INFINITY, f64::min))
}

/// The nonzero entries of `h` row by row.
fn rows_of(h: &RMatrix) -> Vec<Vec<(usize, f64)>> {
    (0..h.nrows())
        .map(|i| {
            (0..h.ncols())
                .filter(|&j| h[(i, j)] != 0.0)
                .map(|j| (j, h[(i, j)]))
                .collect()
        })
        .collect()
}

/// The bright space of a model (step 1 of the module docs).
#[derive(Clone, Debug)]
pub(crate) struct Closure {
    basis: Basis,
    /// The dephased bright nodes `Δ_B` (ascending).
    delta: Vec<usize>,
    /// The basis is the node span of `delta`, and every node of the model
    /// is dephased.
    every_node_dephased: bool,
    /// The smallest loss `Γ₀`, a shift of `H_eff`.
    gamma0: f64,
    /// Nodes with `Γ′_j = Γ_j − Γ₀ > 0`, with `Γ′_j`.
    excess: Vec<(usize, f64)>,
}

impl Closure {
    /// The bright space of `model` with the dephasing of every node taken
    /// as positive when `all` (the closure of an added uniform rate) or
    /// with the model's own dephasing otherwise.
    pub(crate) fn new(model: &DenseModel, all: bool, dark_tolerance: f64) -> Closure {
        let n = model.h.nrows();
        let gamma0 = model.losses.iter().copied().fold(f64::INFINITY, f64::min);
        let gamma0 = if gamma0.is_finite() { gamma0 } else { 0.0 };
        let excess: Vec<(usize, f64)> = model
            .losses
            .iter()
            .enumerate()
            .filter(|&(_, &g)| g - gamma0 > 0.0)
            .map(|(j, &g)| (j, g - gamma0))
            .collect();
        let mut seeds: Vec<usize> = model.traps.iter().map(|&(t, _)| t).collect();
        seeds.extend(excess.iter().map(|&(j, _)| j));
        seeds.sort_unstable();
        seeds.dedup();
        let dephased: Vec<bool> = model.dephasing.iter().map(|&g| all || g > 0.0).collect();
        let rows = rows_of(&model.h);
        if dephased.iter().all(|&d| d) {
            // Components of the hopping pattern that hold a seed.
            let mut seen = vec![false; n];
            let mut stack = Vec::new();
            for &s in &seeds {
                if !seen[s] {
                    seen[s] = true;
                    stack.push(s);
                }
            }
            while let Some(u) = stack.pop() {
                for &(v, _) in &rows[u] {
                    if v != u && !seen[v] {
                        seen[v] = true;
                        stack.push(v);
                    }
                }
            }
            let nodes: Vec<usize> = (0..n).filter(|&u| seen[u]).collect();
            return Closure {
                basis: Basis::Nodes(nodes.clone()),
                delta: nodes,
                every_node_dephased: true,
                gamma0,
                excess,
            };
        }
        // H centred and scaled: Hs = (H − cI)/s.
        let c = centre(&model.h);
        let mut s = 0.0f64;
        for (i, row) in rows.iter().enumerate() {
            for &(j, x) in row {
                let y = if i == j { x - c } else { x };
                s = s.max(y.abs());
            }
            if !row.iter().any(|&(j, _)| j == i) {
                s = s.max(c.abs());
            }
        }
        if s == 0.0 {
            s = 1.0;
        }
        let apply = |v: &[f64]| -> Vec<f64> {
            rows.iter()
                .enumerate()
                .map(|(i, row)| {
                    let mut acc = 0.0;
                    for &(j, x) in row {
                        acc += x * v[j];
                    }
                    (acc - c * v[i]) / s
                })
                .collect()
        };
        let mut basis: Vec<Vec<f64>> = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        let unit = |j: usize| {
            let mut e = vec![0.0; n];
            e[j] = 1.0;
            e
        };
        for &u in &seeds {
            extend(unit(u), &mut basis, &mut queue, dark_tolerance);
        }
        let mut included = vec![false; n];
        loop {
            while let Some(k) = queue.pop_front() {
                let w = apply(&basis[k]);
                extend(w, &mut basis, &mut queue, dark_tolerance);
            }
            let mut grew = false;
            for j in 0..n {
                if !dephased[j] || included[j] {
                    continue;
                }
                let weight = basis.iter().map(|b| b[j] * b[j]).sum::<f64>().sqrt();
                if weight > dark_tolerance {
                    included[j] = true;
                    extend(unit(j), &mut basis, &mut queue, dark_tolerance);
                    grew = true;
                }
            }
            // Images of vectors added for dephased nodes are still queued.
            if !grew && queue.is_empty() {
                break;
            }
        }
        let r = basis.len();
        if r >= n {
            let delta: Vec<usize> = (0..n).filter(|&j| dephased[j]).collect();
            return Closure {
                basis: Basis::Nodes((0..n).collect()),
                every_node_dephased: false,
                delta,
                gamma0,
                excess,
            };
        }
        let v = RMatrix::from_fn(n, r, |i, k| basis[k][i]);
        Closure {
            basis: Basis::Dense(v),
            delta: (0..n).filter(|&j| included[j]).collect(),
            every_node_dephased: false,
            gamma0,
            excess,
        }
    }

    /// The bright dimension `r`.
    pub(crate) fn dim(&self) -> usize {
        self.basis.dim()
    }

    /// An orthonormal basis (`n × (n − r)`) of the complement of the bright
    /// space: greedy two-pass Gram–Schmidt of `e_1 … e_n` against `V` and
    /// the vectors kept so far, keeping residuals above `1/(2√n)`.
    pub(crate) fn complement(&self, n: usize) -> RMatrix {
        match &self.basis {
            Basis::Nodes(nodes) => {
                let out: Vec<usize> = (0..n).filter(|u| nodes.binary_search(u).is_err()).collect();
                RMatrix::from_fn(n, out.len(), |i, k| if i == out[k] { 1.0 } else { 0.0 })
            }
            Basis::Dense(v) => {
                let want = n - v.ncols();
                let mut cols: Vec<Vec<f64>> = (0..v.ncols())
                    .map(|k| v.column(k).iter().copied().collect())
                    .collect();
                let fixed = cols.len();
                let threshold = 0.5 / (n as f64).sqrt();
                for j in 0..n {
                    if cols.len() - fixed == want {
                        break;
                    }
                    let mut w = vec![0.0; n];
                    w[j] = 1.0;
                    for _ in 0..2 {
                        for b in &cols {
                            let d = dot(b, &w);
                            for (x, y) in w.iter_mut().zip(b) {
                                *x -= d * y;
                            }
                        }
                    }
                    let size = norm(&w);
                    if size > threshold {
                        cols.push(w.into_iter().map(|x| x / size).collect());
                    }
                }
                let kept = &cols[fixed..];
                RMatrix::from_fn(n, kept.len(), |i, k| kept[k][i])
            }
        }
    }
}

/// The classical preconditioner of the population GMRES (only when every
/// node is dephased and the basis is node-aligned): `K̃` (incoherent
/// hopping rates `W_jl = 2H_jl²Λ_jl/((H_jj − H_ll)² + Λ_jl²)` with
/// `Λ_jl = λ_j + λ_l`, `λ_j = κ_j + (Γ_j + γ̃_j)/2`, and the decay
/// `−2λ_j`) and the LU of `A_c = K̃ + diag(γ̃)`; `z = A_c⁻¹(K̃r)`.
struct Classical {
    /// Rows of `K̃` (local indices), diagonal included.
    k: Vec<Vec<(usize, f64)>>,
    lu: LU<f64, nalgebra::Dyn, nalgebra::Dyn>,
}

impl Classical {
    fn apply_k(&self, x: &[f64]) -> DVector<f64> {
        DVector::from_iterator(
            x.len(),
            self.k.iter().map(|row| {
                let mut acc = 0.0;
                for &(l, w) in row {
                    acc += w * x[l];
                }
                acc
            }),
        )
    }

    fn precondition(&self, r: &[C64]) -> Vec<C64> {
        let re: Vec<f64> = r.iter().map(|z| z.re).collect();
        let im: Vec<f64> = r.iter().map(|z| z.im).collect();
        let mut zr = self.apply_k(&re);
        let mut zi = self.apply_k(&im);
        if !(self.lu.solve_mut(&mut zr) && self.lu.solve_mut(&mut zi)) {
            return r.to_vec();
        }
        zr.iter()
            .zip(zi.iter())
            .map(|(&a, &b)| C64::new(a, b))
            .collect()
    }
}

/// What the classical preconditioner needs of the bright nodes (local
/// indices): `H_jj`, `κ_j + Γ_j/2` and the hopping `(a, b, H_ab)`.
#[derive(Clone, Debug)]
struct Hopping {
    diag: Vec<f64>,
    decay: Vec<f64>,
    hops: Vec<(usize, usize, f64)>,
}

/// Operations and GMRES iterations of one transport call so far.
struct Budget {
    spent: f64,
    iterations: usize,
}

/// The generator on the bright space in the basis `V`, for residuals that
/// keep the dephasing of populations and coherences apart.
enum Local {
    /// Node basis: `VᵀHV` and `κ + Γ′/2` per bright node.
    Nodes {
        nodes: Vec<usize>,
        h: CMatrix,
        decay: Vec<f64>,
    },
    /// General basis: `H′_B` and the columns `q_k = Vᵀe_k` over `Δ_B`.
    Dense { h_prime: CMatrix, q: CMatrix },
}

/// The bright space with its Schur form and reads: everything the solves
/// of one closure share, for any added uniform dephasing.
pub(crate) struct Bright {
    closure: Closure,
    schur: SchurForm,
    /// The walk's own dephasing per node.
    dephasing: Vec<f64>,
    traps: Vec<(usize, f64)>,
    /// `U†Vᵀe_t` for the traps, `Γ′` nodes and `Δ_B` (columns in order).
    w_traps: Planar,
    w_excess: Planar,
    w_delta: Planar,
    /// `2·max_i Σ_j |T_ij|`.
    t_norm: f64,
    /// `max_ij |T_ij|`.
    t_max: f64,
    /// Data of the classical preconditioner (every node dephased):
    /// `H_jj` and `κ_j + Γ_j/2` per bright node, and the hopping
    /// `(a, b, H_ab)` between bright nodes (local indices).
    classical: Option<Hopping>,
    /// Use the classical preconditioner where it applies.
    use_classical: bool,
    /// Operations counted as spent before each solve (0; tests of the work
    /// limit set it).
    spent_before: f64,
    /// The generator in the basis `V`.
    local: Local,
}

impl Bright {
    /// The closure of `model` (see [`Closure::new`]), the Schur form of
    /// `H′_B` and the reads.
    pub(crate) fn new(model: &DenseModel, all: bool, options: &TransportOptions) -> Result<Bright> {
        let closure = Closure::new(model, all, options.dark_tolerance);
        let n = model.h.nrows();
        let r = closure.dim();
        let rows = rows_of(&model.h);
        // `VᵀHV − cI`: a real multiple of the identity leaves every
        // commutator unchanged, and centring keeps a large common energy
        // offset (FMO's 12 210 cm⁻¹) from costing digits in the Schur form
        // and the triangular solves.
        let mut re = closure.basis.project(&rows, n);
        let centre = centre(&model.h);
        for a in 0..r {
            re[(a, a)] -= centre;
        }
        let mut excess = vec![0.0; n];
        for &(j, g) in &closure.excess {
            excess[j] = g;
        }
        let mut kappa = vec![0.0; n];
        for &(t, k) in &model.traps {
            kappa[t] = k;
        }
        let d: Vec<f64> = (0..n)
            .map(|j| kappa[j] + 0.5 * excess[j] + 0.5 * model.dephasing[j])
            .collect();
        let im = closure.basis.project_diagonal(&d);
        let shift = 0.5 * closure.gamma0;
        let h_prime = CMatrix::from_fn(r, r, |a, b| {
            let mut z = C64::new(re[(a, b)], -im[(a, b)]);
            if a == b {
                z.im -= shift;
            }
            z
        });
        let schur = SchurForm::new(&h_prime)?;
        let local = match &closure.basis {
            Basis::Nodes(nodes) => Local::Nodes {
                nodes: nodes.clone(),
                h: re.map(|x| C64::new(x, 0.0)),
                decay: nodes.iter().map(|&j| kappa[j] + 0.5 * excess[j]).collect(),
            },
            Basis::Dense(v) => Local::Dense {
                h_prime: h_prime.clone(),
                q: CMatrix::from_fn(r, closure.delta.len(), |a, k| {
                    C64::new(v[(closure.delta[k], a)], 0.0)
                }),
            },
        };
        let u = schur.q();
        let columns = |nodes: &[usize]| -> Planar {
            let mut w = Planar::zeros(r, nodes.len());
            match &closure.basis {
                Basis::Nodes(bright) => {
                    for (c, j) in nodes.iter().enumerate() {
                        if let Ok(p) = bright.binary_search(j) {
                            for a in 0..r {
                                w.re[(a, c)] = u[(p, a)].re;
                                w.im[(a, c)] = -u[(p, a)].im;
                            }
                        }
                    }
                }
                Basis::Dense(v) => {
                    let rows = RMatrix::from_fn(r, nodes.len(), |a, c| v[(nodes[c], a)]);
                    w.re = u.map(|z| z.re).transpose() * &rows;
                    w.im = -(u.map(|z| z.im).transpose() * &rows);
                }
            }
            w
        };
        let trap_nodes: Vec<usize> = model.traps.iter().map(|&(t, _)| t).collect();
        let excess_nodes: Vec<usize> = closure.excess.iter().map(|&(j, _)| j).collect();
        let w_traps = columns(&trap_nodes);
        let w_excess = columns(&excess_nodes);
        let w_delta = columns(&closure.delta);
        let t = schur.t();
        let mut t_norm = 0.0f64;
        let mut t_max = 0.0f64;
        for i in 0..r {
            let mut row = 0.0;
            for j in 0..r {
                row += t[(i, j)].norm();
                t_max = t_max.max(t[(i, j)].norm());
            }
            t_norm = t_norm.max(row);
        }
        let classical = match &closure.basis {
            Basis::Nodes(nodes) if closure.every_node_dephased => {
                let diag: Vec<f64> = nodes.iter().map(|&j| model.h[(j, j)]).collect();
                let decay: Vec<f64> = nodes
                    .iter()
                    .map(|&j| kappa[j] + 0.5 * model.losses[j])
                    .collect();
                let mut hops = Vec::new();
                for (a, &u) in nodes.iter().enumerate() {
                    for &(v, x) in &rows[u] {
                        if v != u
                            && let Ok(b) = nodes.binary_search(&v)
                        {
                            hops.push((a, b, x));
                        }
                    }
                }
                Some(Hopping { diag, decay, hops })
            }
            _ => None,
        };
        Ok(Bright {
            closure,
            schur,
            dephasing: model.dephasing.clone(),
            traps: model.traps.clone(),
            w_traps,
            w_excess,
            w_delta,
            t_norm: 2.0 * t_norm,
            t_max,
            classical,
            use_classical: true,
            spent_before: 0.0,
            local,
        })
    }

    /// Switch the classical preconditioner off (for measurements).
    #[cfg(test)]
    pub(crate) fn without_classical(mut self) -> Self {
        self.use_classical = false;
        self
    }

    /// Count `spent` operations as spent before each solve (to reach the
    /// work limit in tests).
    #[cfg(test)]
    pub(crate) fn with_spent(mut self, spent: f64) -> Self {
        self.spent_before = spent;
        self
    }

    /// The bright dimension `r`.
    pub(crate) fn dim(&self) -> usize {
        self.closure.dim()
    }

    /// `γ̃_k = γ_k + e` over `Δ_B`.
    fn rates(&self, e: f64) -> Vec<f64> {
        self.closure
            .delta
            .iter()
            .map(|&k| self.dephasing[k] + e)
            .collect()
    }

    /// The classical preconditioner for the added rate `e`.
    fn classical(&self, rates: &[f64]) -> Option<Classical> {
        let Hopping { diag, decay, hops } = self.classical.as_ref()?;
        if !self.use_classical || rates.is_empty() {
            return None;
        }
        let m = diag.len();
        let lambda: Vec<f64> = (0..m).map(|a| decay[a] + 0.5 * rates[a]).collect();
        let mut k: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
        let mut out = vec![0.0; m];
        for &(a, b, h) in hops {
            let big = lambda[a] + lambda[b];
            let de = diag[a] - diag[b];
            let w = 2.0 * h * h * big / (de * de + big * big);
            if w.is_finite() && w != 0.0 {
                k[a].push((b, w));
                out[a] += w;
            }
        }
        let mut dense = RMatrix::zeros(m, m);
        for a in 0..m {
            let diagonal = -2.0 * lambda[a] - out[a];
            k[a].push((a, diagonal));
            k[a].sort_by_key(|&(b, _)| b);
            for &(b, w) in &k[a] {
                dense[(a, b)] = w;
            }
            dense[(a, a)] += rates[a];
        }
        let lu = dense.lu();
        if !lu.is_invertible() {
            return None;
        }
        Some(Classical { k, lu })
    }

    /// The shift `σ = ie` of the triangular equations.
    fn sigma(&self, e: f64) -> C64 {
        C64::new(0.0, e)
    }

    /// `𝓢̃⁻¹(C̃)`: `T Y − Y T† − ieY = −iC̃`.
    fn coherent(&self, c: &Planar, e: f64) -> Planar {
        let mut y = Planar {
            re: c.im.clone(),
            im: -&c.re,
        };
        self.schur.solve_triangular(&mut y, self.sigma(e));
        y
    }

    /// `Σ_k coef_k u_k u_k†` over `Δ_B`, times `−i`, through `𝓢̃⁻¹`.
    fn fed(&self, coef: &[C64], e: f64) -> Planar {
        let r = self.dim();
        let mut f = Planar::zeros(r, r);
        let scaled: Vec<C64> = coef.iter().map(|&z| z * C64::new(0.0, -1.0)).collect();
        f.add_rank(&self.w_delta, &scaled);
        self.schur.solve_triangular(&mut f, self.sigma(e));
        f
    }

    /// `Z̃` with `𝓛(Z) = −C` (step 3 of the module docs), and the GMRES
    /// iterations it took.
    fn split(
        &self,
        c: &Planar,
        e: f64,
        rates: &[f64],
        classical: Option<&Classical>,
        options: &TransportOptions,
        budget: &mut Budget,
    ) -> Result<(Planar, usize)> {
        let mut z = self.coherent(c, e);
        let m = rates.len();
        if m == 0 {
            return Ok((z, 0));
        }
        let r = self.dim() as f64;
        let b = z.diag_reads(&self.w_delta);
        let apply = |v: &[C64], out: &mut [C64]| {
            let coef: Vec<C64> = v.iter().zip(rates).map(|(x, g)| x * g).collect();
            let reads = self.fed(&coef, e).diag_reads(&self.w_delta);
            for ((o, x), y) in out.iter_mut().zip(v).zip(reads) {
                *o = x - y;
            }
        };
        let precondition = |v: &[C64]| -> Vec<C64> {
            match classical {
                Some(c) => c.precondition(v),
                None => v.to_vec(),
            }
        };
        let opts = GmresOptions {
            tol: options.tolerance,
            restart: m.min(RESTART),
            max_iterations: options.max_iterations,
            work: (
                8.0 * (r * r * r + 2.0 * r * r * m as f64),
                MAX_TRANSPORT_WORK - budget.spent,
            ),
            scale: POPULATION_SCALE,
            breakdown: BREAKDOWN,
        };
        let sol = gmres(m, apply, precondition, &b, &opts);
        budget.spent += sol.spent;
        budget.iterations += sol.iterations;
        match sol.end {
            GmresEnd::Converged => {}
            GmresEnd::Work => {
                return Err(QsoError::invalid(
                    "graph",
                    format!(
                        "transport needs more than {MAX_TRANSPORT_WORK:.1e} operations after {} \
                         GMRES iterations; use fewer nodes or more symmetries",
                        budget.iterations
                    ),
                ));
            }
            GmresEnd::Iterations | GmresEnd::Breakdown => {
                return Err(QsoError::NoConvergence {
                    method: "GMRES (transport populations)",
                    iterations: sol.iterations,
                    residual: sol.residual,
                });
            }
        }
        let coef: Vec<C64> = sol.x.iter().zip(rates).map(|(x, g)| x * g).collect();
        let fed = self.fed(&coef, e);
        z.re += &fed.re;
        z.im += &fed.im;
        Ok((z, sol.iterations))
    }

    /// `C + 𝓛(X)` in the basis `V`, the residual of `𝓛(X) = −C`. On a node
    /// basis the dephasing acts on the coherences alone, entry by entry, so
    /// the residual does not suffer the cancellation between the decay of
    /// populations in `H′_B` and their refeeding by `𝓙`, which in the Schur
    /// basis costs digits under strong dephasing.
    fn residual(&self, c: &CMatrix, x: &CMatrix, e: f64, rates: &[f64]) -> CMatrix {
        let r = self.dim();
        let minus_i = C64::new(0.0, -1.0);
        match &self.local {
            Local::Nodes { nodes, h, decay } => {
                let commutator = cmul(h, x) - cmul(x, h);
                let gamma0 = self.closure.gamma0;
                let gam: Vec<f64> = nodes.iter().map(|&j| self.dephasing[j] + e).collect();
                CMatrix::from_fn(r, r, |a, b| {
                    let mut rate = decay[a] + decay[b] + gamma0;
                    if a != b {
                        rate += 0.5 * (gam[a] + gam[b]);
                    }
                    c[(a, b)] + minus_i * commutator[(a, b)] - x[(a, b)] * rate
                })
            }
            Local::Dense { h_prime, q } => {
                let commutator = cmul(h_prime, x) - cmul(x, &h_prime.adjoint());
                let mut out = c + commutator * minus_i - x * C64::new(e, 0.0);
                if !rates.is_empty() {
                    let xq = cmul(x, q);
                    let mut scaled = q.clone();
                    for (k, &g) in rates.iter().enumerate() {
                        let read: C64 = (0..r).map(|a| q[(a, k)].conj() * xq[(a, k)]).sum();
                        for a in 0..r {
                            scaled[(a, k)] *= read * g;
                        }
                    }
                    out += cmul(&scaled, &q.adjoint());
                }
                out
            }
        }
    }

    /// `Z̃` with `𝓛(Z) = −C` (`c` in the Schur basis, `c_v` in the basis
    /// `V`), refined while the residual is above `tolerance·‖C‖_F` (at most
    /// twice); with `Z` in the basis `V`, the report and an estimate of the
    /// error of `η` left after the refinement (the last change of `η`,
    /// scaled by the reduction of the residual since).
    #[allow(clippy::too_many_arguments)]
    fn refined(
        &self,
        name: &str,
        c: &Planar,
        c_v: &CMatrix,
        e: f64,
        rates: &[f64],
        classical: Option<&Classical>,
        norm_l: f64,
        options: &TransportOptions,
        budget: &mut Budget,
    ) -> Result<(Planar, CMatrix, SolveReport, f64)> {
        let (mut z, mut iterations) = self.split(c, e, rates, classical, options, budget)?;
        let c_norm = c_v.norm();
        let mut change = 0.0;
        let mut previous = f64::INFINITY;
        let mut step = 0;
        loop {
            let z_v = self.schur.from_schur(&z);
            let res = self.residual(c_v, &z_v, e, rates);
            let res_norm = res.norm();
            // The last correction measured the error of the iterate before
            // it; the residual shrank by `res_norm/previous` since, and so
            // (to first order) did the error.
            if previous > 0.0 && res_norm < previous {
                change *= res_norm / previous;
            }
            previous = res_norm;
            if step == REFINEMENTS || res_norm.is_nan() || res_norm <= options.tolerance * c_norm {
                let z_norm = z_v.norm();
                let size = norm_l * z_norm + c_norm;
                let report = SolveReport {
                    name: name.to_string(),
                    iterations,
                    backward_error: if size > 0.0 { res_norm / size } else { 0.0 },
                    condition: if c_norm > 0.0 {
                        norm_l * z_norm / c_norm
                    } else {
                        0.0
                    },
                };
                return Ok((z, z_v, report, change));
            }
            step += 1;
            let res = self.schur.to_schur(&res);
            let (dz, its) = self.split(&res, e, rates, classical, options, budget)?;
            iterations += its;
            change = self.trapped(&dz).iter().sum();
            z.re += &dz.re;
            z.im += &dz.im;
        }
    }

    /// `2κ_t Re(u_t† Z̃ u_t)` per trap.
    fn trapped(&self, z: &Planar) -> Vec<f64> {
        z.diag_reads(&self.w_traps)
            .iter()
            .zip(&self.traps)
            .map(|(x, &(_, k))| 2.0 * k * x.re)
            .collect()
    }

    /// `ρ̃₀ = U†Vᵀρ₀VU`, `Vᵀρ₀V` and the weight outside the bright space,
    /// `1 − Re tr(Vᵀρ₀V)`.
    fn start(&self, start: &Start<'_>) -> (Planar, CMatrix, f64) {
        let r = self.dim();
        let u = self.schur.q();
        match *start {
            Start::Node(s) => {
                let q = self.closure.basis.row(s);
                let weight = dot(&q, &q);
                let ut: Vec<C64> = (0..r)
                    .map(|a| {
                        let mut acc = ZERO;
                        for (i, &x) in q.iter().enumerate() {
                            if x != 0.0 {
                                acc += u[(i, a)].conj() * x;
                            }
                        }
                        acc
                    })
                    .collect();
                let mut p = Planar::zeros(r, r);
                for b in 0..r {
                    for a in 0..r {
                        let z = ut[a] * ut[b].conj();
                        p.re[(a, b)] = z.re;
                        p.im[(a, b)] = z.im;
                    }
                }
                let plain = CMatrix::from_fn(r, r, |a, b| C64::new(q[a] * q[b], 0.0));
                (p, plain, 1.0 - weight)
            }
            Start::Density(rho) => {
                let c = self.closure.basis.project_complex(rho);
                let trace: f64 = (0..r).map(|a| c[(a, a)].re).sum();
                (self.schur.to_schur(&c), c, 1.0 - trace)
            }
        }
    }

    /// The transport from `start` with the uniform rate `e ≥ 0` added to
    /// every node's dephasing (`e > 0` needs a closure with `all`); without
    /// `want_tau` the second solve is skipped and `arrival_time` is `None`.
    pub(crate) fn solve(
        &self,
        start: &Start<'_>,
        e: f64,
        want_tau: bool,
        options: &TransportOptions,
    ) -> Result<TransportResult> {
        let slowest = 2.0 * self.schur.slowest_decay() + e;
        if slowest.is_nan() || slowest <= 1e-14 * self.t_max {
            return Err(QsoError::invalid(
                "dark_tolerance",
                format!(
                    "a bright state decays at {slowest:.1e}, below rounding: raise dark_tolerance"
                ),
            ));
        }
        let rates = self.rates(e);
        let max_rate = rates.iter().copied().fold(0.0, f64::max);
        let norm_l = self.t_norm + e + max_rate;
        let classical = self.classical(&rates);
        let mut budget = Budget {
            spent: self.spent_before,
            iterations: 0,
        };
        let (rho, rho_v, outside) = self.start(start);
        let (x, x_v, report_x, change) = self.refined(
            "X",
            &rho,
            &rho_v,
            e,
            &rates,
            classical.as_ref(),
            norm_l,
            options,
            &mut budget,
        )?;
        let trapped = self.trapped(&x);
        let efficiency: f64 = trapped.iter().sum();
        let gamma0 = self.closure.gamma0;
        let trace: f64 = (0..self.dim()).map(|a| x.re[(a, a)]).sum();
        let excess: f64 = x
            .diag_reads(&self.w_excess)
            .iter()
            .zip(&self.closure.excess)
            .map(|(z, &(_, g))| g * z.re)
            .sum();
        let mut lost = gamma0 * trace + excess;
        let mut dark = 0.0;
        if gamma0 > 0.0 {
            lost += outside;
        } else {
            dark = outside.max(0.0);
        }
        let error = (efficiency + lost + dark - 1.0).abs() + change.abs();
        let mut solves = vec![
            SolveReport {
                name: "schur".to_string(),
                iterations: 0,
                backward_error: self.schur.backward_error(),
                condition: norm_l / slowest,
            },
            report_x,
        ];
        let mut arrival = None;
        if want_tau {
            let (y, _, report_y, _) = self.refined(
                "Y",
                &x,
                &x_v,
                e,
                &rates,
                classical.as_ref(),
                norm_l,
                options,
                &mut budget,
            )?;
            let eta_tau: f64 = self.trapped(&y).iter().sum();
            arrival = arrival_time(efficiency, eta_tau, error);
            solves.push(report_y);
        }
        let condition = solves.iter().map(|s| s.condition).fold(0.0, f64::max);
        Ok(TransportResult {
            efficiency,
            arrival_time: arrival,
            trapped,
            lost,
            dark,
            added_dephasing: e,
            dimension: self.dim(),
            slowest_decay: Some(slowest),
            condition,
            error,
            solves,
        })
    }
}
