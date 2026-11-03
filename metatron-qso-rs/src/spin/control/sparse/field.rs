//! Field profiles with relaxation on sparse blocks.
//!
//! The generator of a field `B` is `L(B) = −i[H₀ + B·H_Z, ·] + R(B)`, the
//! Hamiltonian affine in `B` and the relaxation `R(B)` (with the drift
//! column towards the thermal state) not. `R` is interpolated once per
//! problem across the profile's range by the Chebyshev polynomial through
//! the Chebyshev–Lobatto nodes `x_j = cos(πj/(N − 1))`, `N` = 5, 9, 17 or 33
//! (the nodes of each level contain the last's), until its two highest
//! coefficients fall below [`FIELD_TOLERANCE`] of the relaxation's largest
//! entry; that size is the interpolant's stated error. Every step then
//! takes `R(B)` and its exact derivative `R'(B)` of the interpolant as
//! weighted sums of the node matrices on their shared pattern, and the
//! gradient with respect to the step's field pairs the costate with
//! `−i[H_Z, ·] + R'(B)` by the Gauss–Legendre quadrature of 1.46.

use nalgebra::DVector;
use num_complex::Complex64 as C64;

use super::super::super::blocks::{Liouville, Sparse};
use super::super::super::krylov::{SparseBlock, SparseFamily};
use super::super::super::{ONE, SpinSystem, ZERO};
use super::super::{Control, Drive, Member, augment};
use super::{
    Budget, GAUSS, SUBSTEP, Scaled, SparseGenerator, commutator_block, member_cost, nonzeros,
    norm_bound, propagate, step_gradient,
};
use crate::error::{QsoError, Result};
use crate::linalg::CMatrix;

/// Relative size of the interpolant's two highest Chebyshev coefficients
/// at which the relaxation over the field is taken as converged.
pub const FIELD_TOLERANCE: f64 = 1e-10;

/// The largest stated error of the interpolant accepted: beyond it, the
/// range is too wide for the relaxation's variation (a narrower range or
/// faster motion helps).
pub const MAX_FIELD_ERROR: f64 = 1e-6;

/// The numbers of Chebyshev–Lobatto nodes tried.
pub(in crate::spin) const LEVELS: [usize; 4] = [5, 9, 17, 33];

/// `(a_j(x), a'_j(x))`: the weights of the node values in the interpolant
/// through `N` Chebyshev–Lobatto nodes and in its derivative,
/// `a_j(x) = Σ_m α_mj T_m(x)`, `α_mj = (2/(N−1)) ε_j ε_m cos(πmj/(N−1))`
/// (`ε = ½` at the ends).
pub(in crate::spin) fn weights(n: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    let mut t = vec![0.0; n];
    let mut dt = vec![0.0; n];
    // T_m and T'_m = m U_{m−1}.
    let mut u = vec![0.0; n];
    t[0] = 1.0;
    if n > 1 {
        t[1] = x;
        u[0] = 1.0;
        dt[1] = 1.0;
    }
    for m in 2..n {
        t[m] = 2.0 * x * t[m - 1] - t[m - 2];
        u[m - 1] = if m == 2 {
            2.0 * x
        } else {
            2.0 * x * u[m - 2] - u[m - 3]
        };
        dt[m] = m as f64 * u[m - 1];
    }
    let edge = |k: usize| if k == 0 || k == n - 1 { 0.5 } else { 1.0 };
    let scale = 2.0 / (n - 1) as f64;
    let mut a = vec![0.0; n];
    let mut da = vec![0.0; n];
    for j in 0..n {
        for m in 0..n {
            let alpha = scale
                * edge(j)
                * edge(m)
                * (std::f64::consts::PI * (m * j) as f64 / (n - 1) as f64).cos();
            a[j] += alpha * t[m];
            da[j] += alpha * dt[m];
        }
    }
    (a, da)
}

/// The weights of the node values in the Chebyshev coefficient `m`.
pub(in crate::spin) fn coefficient(n: usize, m: usize) -> Vec<f64> {
    let edge = |k: usize| if k == 0 || k == n - 1 { 0.5 } else { 1.0 };
    (0..n)
        .map(|j| {
            2.0 / (n - 1) as f64
                * edge(j)
                * edge(m)
                * (std::f64::consts::PI * (m * j) as f64 / (n - 1) as f64).cos()
        })
        .collect()
}

/// What a field-profile problem with relaxation needs on sparse blocks.
pub(in crate::spin::control) struct SparseField {
    /// `−i[H₀, ·]` and `−i[H_Z, ·]` (per tesla) on `[vec Δ; Tr ρ]`, and
    /// their adjoints.
    ops: [SparseBlock; 2],
    ops_adjoint: [SparseBlock; 2],
    op_norms: [f64; 2],
    /// The relaxation at the nodes, on their shared pattern.
    family: SparseFamily,
    nodes: usize,
    /// Bound on `‖R‖` over the nodes.
    relaxation_norm: f64,
    min: f64,
    max: f64,
    /// The interpolant's stated error, relative to the relaxation's
    /// largest entry.
    pub error: f64,
    start: DVector<C64>,
    target: DVector<C64>,
}

impl SparseField {
    /// Coefficients propagated: `d² + 1`.
    pub(in crate::spin::control) fn size(&self) -> usize {
        self.start.len()
    }

    /// Nodes of the interpolant.
    pub(in crate::spin::control) fn nodes(&self) -> usize {
        self.nodes
    }

    /// Bytes held.
    pub(in crate::spin::control) fn bytes(&self) -> f64 {
        (self.family.bytes()
            + self.ops.iter().map(SparseBlock::bytes).sum::<usize>()
            + self
                .ops_adjoint
                .iter()
                .map(SparseBlock::bytes)
                .sum::<usize>()) as f64
            + 32.0 * self.size() as f64
    }

    /// `B` mapped to `[−1, 1]`.
    fn unit(&self, b: f64) -> f64 {
        ((2.0 * b - self.min - self.max) / (self.max - self.min)).clamp(-1.0, 1.0)
    }

    /// `R(B)`, `R(B)†` and `R'(B)` of the interpolant.
    pub(in crate::spin::control) fn relaxation(
        &self,
        b: f64,
    ) -> (SparseBlock, SparseBlock, SparseBlock) {
        let (a, da) = weights(self.nodes, self.unit(b));
        let scale = 2.0 / (self.max - self.min);
        let da: Vec<f64> = da.iter().map(|x| x * scale).collect();
        (
            self.family.combine(&a),
            self.family.combine_adjoint(&a),
            self.family.combine(&da),
        )
    }

    /// Estimated operations and memory of one evaluation under the fields
    /// `u`, with `entries` the relaxation's.
    pub(in crate::spin::control) fn cost(&self, c: &Control, u: &[f64]) -> (f64, f64) {
        let m = self.size() as f64;
        let entries = self.family.nnz() as f64;
        let ops = (self.ops[0].nnz() + self.ops[1].nnz()) as f64;
        let (total, most) = substeps(c, u, self.relaxation_norm, self.op_norms);
        let (work, bytes) = member_cost(c, m, entries + ops + m, entries + ops, total, most, true);
        // Three weighted sums per step, forward and backward.
        let sums = 3.0 * 8.0 * self.nodes as f64 * entries * c.steps as f64;
        (work + sums, self.bytes() + bytes + 3.0 * 20.0 * entries)
    }

    /// Transfer and gradient under the fields `u` (T).
    pub(in crate::spin::control) fn member(
        &self,
        c: &Control,
        u: &[f64],
        budget: &Budget,
    ) -> Result<(f64, Vec<f64>)> {
        let dt = c.step();
        let size = self.size() as f64;
        let entries = self.family.nnz();
        let product = 8.0 * (entries + self.ops[0].nnz() + self.ops[1].nnz()) as f64 + 8.0 * size;
        let sums = 3.0 * 8.0 * self.nodes as f64 * entries as f64;
        let norm =
            |r: &SparseBlock, b: f64| norm_bound(r) + self.op_norms[0] + b.abs() * self.op_norms[1];
        let mut states = Vec::with_capacity(c.steps + 1);
        states.push(self.start.clone());
        for (k, &b) in u.iter().enumerate() {
            let (a, _) = weights(self.nodes, self.unit(b));
            let r = self.family.combine(&a);
            let forward = SparseGenerator {
                base: &r,
                diagonal: None,
                ops: &self.ops,
                coefficients: vec![1.0, b],
            };
            let (next, products) = propagate(&forward, &states[k], dt, norm(&r, b));
            budget.spend(products as f64 * product + sums / 3.0)?;
            states.push(next);
        }
        let value = self.target.dotc(&states[c.steps]).re;
        let mut grad = vec![0.0; u.len()];
        let mut lambda = self.target.clone();
        let pairing = GAUSS.len() as f64 * 8.0 * (2 * entries + self.ops[1].nnz()) as f64;
        for k in (0..c.steps).rev() {
            let b = u[k];
            let (r, r_adjoint, derivative) = self.relaxation(b);
            let forward = SparseGenerator {
                base: &r,
                diagonal: None,
                ops: &self.ops,
                coefficients: vec![1.0, b],
            };
            let backward = SparseGenerator {
                base: &r_adjoint,
                diagonal: None,
                ops: &self.ops_adjoint,
                coefficients: vec![1.0, b],
            };
            let directions = [Scaled(&self.ops[1], 1.0), Scaled(&derivative, 1.0)];
            let (g, previous, (products, substeps)) = step_gradient(
                &forward,
                &backward,
                &directions,
                &states[k],
                &lambda,
                dt,
                norm(&r, b),
            );
            let horner = 4.0 * GAUSS.len() as f64 * size;
            budget
                .spend(products as f64 * (product + horner) + substeps as f64 * pairing + sums)?;
            grad[k] = g[0] + g[1];
            lambda = previous;
        }
        Ok((value, grad))
    }
}

/// The estimated cost of a field profile with relaxation: one evaluation
/// (operations and bytes, the interpolant's first level included) and the
/// first level's build alone.
pub(in crate::spin::control) struct FieldCost {
    pub work: f64,
    pub memory: f64,
    pub build: f64,
    pub build_memory: f64,
}

/// Substeps over all steps under the fields `u`, and the most of one.
fn substeps(c: &Control, u: &[f64], relaxation: f64, ops: [f64; 2]) -> (f64, f64) {
    u.iter().fold((0.0, 0.0f64), |(t, m), &b| {
        let norm = relaxation + ops[0] + b.abs() * ops[1];
        let s = (norm * c.step() / SUBSTEP).ceil();
        let s = if s.is_finite() {
            s.max(1.0)
        } else {
            f64::INFINITY
        };
        (t + s, m.max(s))
    })
}

impl SpinSystem {
    /// The relaxation `[vec Δ; Tr ρ]` block (with the drift column) at
    /// field `b`, without the Hamiltonian.
    fn relaxation_block(&self, liouville: &Liouville, b: f64) -> Result<SparseBlock> {
        let d = self.dimension();
        let relaxation = self.relaxation(b)?;
        let zero = Sparse::new(&CMatrix::zeros(d, d));
        Ok(self.sparse_block(liouville, 0, &zero, &relaxation, ZERO, Some(ONE)))
    }

    /// Estimated operations and memory of one evaluation of a field
    /// profile with relaxation on sparse blocks, with the interpolant at
    /// its first level (five nodes, each a build of the relaxation) and
    /// fields of at most `peak` T (or the fields `u`), without building
    /// anything but the relaxation at the middle of the range, whose
    /// entries are counted column by column.
    pub(in crate::spin::control) fn field_estimate(
        &self,
        c: &Control,
        u: Option<&[f64]>,
    ) -> Result<FieldCost> {
        let Drive::Field { min, max } = c.drive else {
            return Err(QsoError::invalid("control", "a field profile"));
        };
        let d = self.dimension();
        let n = (d * d + 1) as f64;
        let liouville = Liouville::new(&vec![Vec::new(); d], None);
        let middle = 0.5 * (min + max);
        let relaxation = self.relaxation(middle)?;
        let zero = Sparse::new(&CMatrix::zeros(d, d));
        let entries = self.sparse_block_entries(&liouville, 0, &zero, &relaxation, true) as f64;
        let (_, redfield_build) = relaxation.redfield_entries(&liouville, Some(0));
        let (h0, dh) = self.control_hamiltonians(c, &Member::ideal())?;
        let ops = (2 * d * (nonzeros(&h0) + nonzeros(&dh[0]))) as f64;
        let op_norms = [
            2.0 * crate::linalg::dense::one_norm(&h0),
            2.0 * crate::linalg::dense::one_norm(&dh[0]),
        ];
        let fields: Vec<f64> = match u {
            Some(u) => u.to_vec(),
            None => vec![max; c.steps],
        };
        let (total, most) = substeps(c, &fields, relaxation.rate, op_norms);
        let (work, bytes) = member_cost(c, n, entries + ops + n, entries + ops, total, most, true);
        let nodes = LEVELS[0] as f64;
        let builds = nodes * (50.0 * entries + redfield_build) + 8.0 * nodes * entries;
        let sums = 3.0 * 8.0 * nodes * entries * c.steps as f64;
        // The build holds the node blocks (20 bytes per entry) and their
        // family (12 plus 16 per node) at once; an evaluation the family and
        // three combined blocks.
        let held = (12.0 + 36.0 * nodes) * entries + 40.0 * ops + 60.0 * entries;
        let memory = 1.1 * (held + bytes + self.held_bytes(&relaxation));
        let build_memory =
            1.1 * ((12.0 + 36.0 * nodes) * entries + 40.0 * ops + self.held_bytes(&relaxation));
        // The interpolant's levels beyond the first are checked as they
        // are built.
        Ok(FieldCost {
            work: work + builds + sums,
            memory,
            build: builds,
            build_memory,
        })
    }

    /// The sparse blocks of a field profile with relaxation, the
    /// relaxation interpolated over the range (see the module
    /// documentation); refused when the interpolant's stated error exceeds
    /// [`MAX_FIELD_ERROR`] or its next level would exceed `memory` bytes.
    pub(in crate::spin::control) fn sparse_field(
        &self,
        c: &Control,
        memory: f64,
    ) -> Result<SparseField> {
        let Drive::Field { min, max } = c.drive else {
            return Err(QsoError::invalid("control", "a field profile"));
        };
        let d = self.dimension();
        let n = d * d;
        let liouville = Liouville::new(&vec![Vec::new(); d], None);
        let (h0, dh) = self.control_hamiltonians(c, &Member::ideal())?;
        let ops = [commutator_block(&h0), commutator_block(&dh[0])];
        let ops_adjoint = [ops[0].adjoint(), ops[1].adjoint()];
        let op_norms = [norm_bound(&ops[0]), norm_bound(&ops[1])];
        let field = |x: f64| 0.5 * (min + max) + 0.5 * (max - min) * x;
        let angle =
            |j: usize, levels: usize| (std::f64::consts::PI * j as f64 / (levels - 1) as f64).cos();
        let mut blocks: Vec<SparseBlock> = (0..LEVELS[0])
            .map(|j| self.relaxation_block(&liouville, field(angle(j, LEVELS[0]))))
            .collect::<Result<_>>()?;
        // The work of one build, as `field_estimate` counts it.
        let entries = blocks[0].nnz() as f64;
        let (_, redfield_build) = self
            .relaxation(0.5 * (min + max))?
            .redfield_entries(&liouville, Some(0));
        let build = 50.0 * entries + redfield_build;
        let mut built = LEVELS[0] as f64 * build;
        let mut level = 0;
        // Why the levels stopped short of the tolerance, if they did.
        let mut short = None;
        let (family, error) = loop {
            let nodes = LEVELS[level];
            let family = SparseFamily::new(&blocks);
            // The relaxation relative to its largest entry, and the drift
            // column (the last) relative to its own.
            let (largest, largest_drift) = blocks
                .iter()
                .map(|b| b.largest_split(n))
                .fold((0.0f64, 0.0f64), |(a, b), (x, y)| (a.max(x), b.max(y)));
            let error = [nodes - 1, nodes - 2]
                .iter()
                .map(|&m| {
                    let (tail, tail_drift) =
                        family.combine(&coefficient(nodes, m)).largest_split(n);
                    let rest = if largest > 0.0 { tail / largest } else { 0.0 };
                    let drift = if largest_drift > 0.0 {
                        tail_drift / largest_drift
                    } else {
                        0.0
                    };
                    rest.max(drift)
                })
                .fold(0.0, f64::max);
            if error <= FIELD_TOLERANCE || level + 1 == LEVELS.len() {
                break (family, error);
            }
            // The next level holds its blocks and its family at once.
            let next = LEVELS[level + 1];
            let per_block =
                blocks.iter().map(SparseBlock::bytes).sum::<usize>() as f64 / nodes as f64;
            let union = family.nnz() as f64;
            let next_bytes = next as f64 * per_block + union * (12.0 + 16.0 * next as f64);
            let next_work = built + (next - nodes) as f64 * build;
            if next_bytes > memory {
                short = Some(format!(
                    "the next level of {next} fields needs about {:.1} GB (limit {:.0} GB)",
                    next_bytes / 1e9,
                    memory / 1e9
                ));
                break (family, error);
            }
            if next_work > super::super::super::MAX_RELAXATION_WORK {
                short = Some(format!(
                    "the next level of {next} fields needs about {next_work:.1e} operations to \
                     build (limit {:.1e})",
                    super::super::super::MAX_RELAXATION_WORK
                ));
                break (family, error);
            }
            built = next_work;
            // The next level: the old nodes at the even places.
            drop(family);
            let next = LEVELS[level + 1];
            let mut grown = Vec::with_capacity(next);
            let mut old = blocks.into_iter();
            for j in 0..next {
                if j % 2 == 0 {
                    grown.push(old.next().expect("an old node"));
                } else {
                    grown.push(self.relaxation_block(&liouville, field(angle(j, next)))?);
                }
            }
            blocks = grown;
            level += 1;
        };
        if let Some(why) = short {
            return Err(QsoError::invalid(
                "field",
                format!(
                    "the relaxation over {min} to {max} T through {} fields errs by {error:.1e} \
                     (tolerance {FIELD_TOLERANCE:e}), and {why}; use fewer spins or a narrower \
                     range",
                    LEVELS[level]
                ),
            ));
        }
        if error > MAX_FIELD_ERROR {
            return Err(QsoError::invalid(
                "field",
                format!(
                    "the relaxation varies too much over {min} to {max} T to interpolate: the \
                     interpolant through {} fields errs by {error:.1e} of its largest entry (at \
                     most {MAX_FIELD_ERROR:e}); narrow the range",
                    LEVELS[level]
                ),
            ));
        }
        let relaxation_norm = blocks.iter().map(norm_bound).fold(0.0, f64::max);
        drop(blocks);
        let start = augment(&c.initial);
        let target = DVector::from_iterator(
            n + 1,
            super::super::vectorise(&c.target)
                .iter()
                .copied()
                .chain(std::iter::once(c.target.trace() / d as f64)),
        );
        Ok(SparseField {
            ops,
            ops_adjoint,
            op_norms,
            family,
            nodes: LEVELS[level],
            relaxation_norm,
            min,
            max,
            error: error.max(f64::EPSILON),
            start,
            target,
        })
    }

    /// The stated error of the relaxation interpolated over the field range
    /// of a field profile `c` (relative to its largest entry) and the
    /// number of fields it was built from, as the optimiser and
    /// [`SpinSystem::control_transfer`] use it (field profiles with
    /// relaxation; an error without relaxation, or when the stated error
    /// would exceed [`MAX_FIELD_ERROR`]).
    pub fn field_relaxation_error(&self, c: &Control) -> Result<(f64, usize)> {
        self.check_control(c)?;
        if !self.relaxes() {
            return Err(QsoError::invalid(
                "control",
                "the system does not relax: field profiles propagate in Hilbert space",
            ));
        }
        // The interpolant's first level checked before anything is built
        // (nothing is propagated, so the evaluations' cost does not count).
        let cost = self.field_estimate(c, None)?;
        super::super::check_cost(
            "the relaxation over the field",
            cost.build,
            cost.build_memory,
        )?;
        let f = self.sparse_field(c, super::super::MAX_CONTROL_MEMORY)?;
        Ok((f.error, f.nodes()))
    }

    /// The relaxation over the field of a field profile with relaxation, as
    /// the optimiser uses it: its interpolant's stated error, the nodes, and
    /// the largest difference from a direct build at each of `fields`
    /// relative to the largest entry (tests).
    #[cfg(test)]
    pub(in crate::spin::control) fn field_interpolation(
        &self,
        c: &Control,
        fields: &[f64],
    ) -> Result<(f64, usize, Vec<f64>)> {
        let f = self.sparse_field(c, f64::INFINITY)?;
        let d = self.dimension();
        let liouville = Liouville::new(&vec![Vec::new(); d], None);
        let mut out = Vec::new();
        for &b in fields {
            let (r, _, _) = f.relaxation(b);
            let direct = self.relaxation_block(&liouville, b)?;
            let scale = direct.largest().max(1e-300);
            let diff = (r.to_dense() - direct.to_dense())
                .iter()
                .fold(0.0f64, |m, z| m.max(z.norm()));
            out.push(diff / scale);
        }
        Ok((f.error, f.nodes, out))
    }
}
