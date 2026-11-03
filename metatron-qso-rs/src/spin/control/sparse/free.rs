//! Field profiles with Redfield relaxation applied without storing it.
//!
//! The generator of a field `B` on `[vec Δ; Tr ρ]` is
//! `−i[H₀ + B·H_Z, ·] + J(B) − R(B) + Tr ρ · R(B)(ρ_eq − 1/d)`: the
//! coherent part, the `T₁` and `T₂` jumps with their drift `J(B)` as a
//! sparse block, and the Redfield superoperator `R(B)` with its drift
//! applied from its terms at every product (see `RedfieldBlocks::
//! apply_products`), never stored. At every step's field the terms are
//! built anew, so no interpolant over the range is needed; the field
//! derivative `R'(B)` comes from the derivative terms of
//! `SpinSystem::redfield_field` (analytic: the affine operators and the
//! resolvent of the spectral densities), the drift's from
//! `R'(B)(ρ_eq − 1/d) + R(B) ρ_eq'`, and the jumps' and the thermal
//! populations' (closed forms, cheap to build) from Richardson-
//! extrapolated central differences.

use nalgebra::DVector;
use num_complex::Complex64 as C64;

use super::super::super::blocks::{Liouville, Sparse};
use super::super::super::krylov::{SparseBlock, SparseFamily};
use super::super::super::redfield::RedfieldBlocks;
use super::super::super::{ONE, SpinSystem, ZERO};
use super::super::{Control, Drive, Linear, Member, augment};
use super::{
    Budget, GAUSS, Pairing, SUBSTEP, TAYLOR_TERMS, commutator_block, norm_bound, propagate,
    step_gradient,
};
use crate::error::{QsoError, Result};
use crate::linalg::CMatrix;

/// What a field profile with Redfield relaxation needs without storing
/// the relaxation: the coherent parts, the start and the target.
pub(in crate::spin::control) struct FreeField {
    ops: [SparseBlock; 2],
    ops_adjoint: [SparseBlock; 2],
    op_norms: [f64; 2],
    start: DVector<C64>,
    target: DVector<C64>,
    d: usize,
    span: f64,
    /// Operations of one application of `R` and of `R'`, and bytes of the
    /// terms, from the relaxation at the middle of the range.
    apply_work: [f64; 2],
    terms_bytes: f64,
    /// A bound on the generator's relaxation over the range.
    relaxation_norm: f64,
}

/// The generator's field-dependent parts at one field.
struct AtField {
    /// `J(B)` and its adjoint, on `[vec Δ; Tr ρ]` with the drift column.
    jumps: SparseBlock,
    jumps_adjoint: SparseBlock,
    redfield: RedfieldBlocks,
    /// `R(B)(ρ_eq − 1/d)` on `vec Δ`.
    drift: Vec<C64>,
    norm: f64,
}

/// The field derivatives at one field.
struct Derivative {
    jumps: SparseBlock,
    redfield: RedfieldBlocks,
    /// `R'(B)(ρ_eq − 1/d) + R(B) ρ_eq'`.
    drift: Vec<C64>,
}

/// The positions `a·d + b` of every Redfield block's elements.
fn positions(r: &RedfieldBlocks, d: usize) -> Vec<Vec<u32>> {
    r.liouville
        .blocks
        .iter()
        .map(|(_, elements)| elements.iter().map(|&(a, b)| (a * d + b) as u32).collect())
        .collect()
}

/// `out[..d²] += s · R(x[..d²])` (or `R†`), block by block.
fn apply_redfield(
    r: &RedfieldBlocks,
    places: &[Vec<u32>],
    x: &[C64],
    s: C64,
    adjoint: bool,
    out: &mut [C64],
) {
    for (q, p) in places.iter().enumerate() {
        let v: Vec<C64> = p.iter().map(|&i| x[i as usize]).collect();
        r.apply_products(q, &v, s, p, adjoint, out);
    }
}

/// `L = −i[H₀ + B·H_Z, ·] + J − R + drift`, or its adjoint.
struct FreeGenerator<'a> {
    ops: &'a [SparseBlock; 2],
    jumps: &'a SparseBlock,
    at: &'a AtField,
    places: &'a [Vec<u32>],
    b: f64,
    adjoint: bool,
}

impl Linear for FreeGenerator<'_> {
    fn apply(&self, x: &DVector<C64>, t: f64) -> DVector<C64> {
        let n = x.len();
        let d2 = n - 1;
        let mut y = vec![ZERO; n];
        let xs = x.as_slice();
        self.jumps.apply(xs, &mut y);
        self.ops[0].apply_add(xs, ONE, &mut y);
        if self.b != 0.0 {
            self.ops[1].apply_add(xs, C64::new(self.b, 0.0), &mut y);
        }
        apply_redfield(
            &self.at.redfield,
            self.places,
            xs,
            -ONE,
            self.adjoint,
            &mut y[..d2],
        );
        if self.adjoint {
            y[d2] += self
                .at
                .drift
                .iter()
                .zip(xs)
                .map(|(f, v)| f.conj() * v)
                .sum::<C64>();
        } else if xs[d2] != ZERO {
            for (o, f) in y.iter_mut().zip(&self.at.drift) {
                *o += f * xs[d2];
            }
        }
        DVector::from_iterator(n, y.into_iter().map(|v| v * t))
    }
}

/// `E = −i[H_Z, ·] + J' − R' + drift'`, the generator's field derivative.
struct FreeDirection<'a> {
    hz: &'a SparseBlock,
    derivative: &'a Derivative,
    places: &'a [Vec<u32>],
}

impl Pairing for FreeDirection<'_> {
    fn pair(&self, lambda: &DVector<C64>, x: &DVector<C64>) -> C64 {
        let n = x.len();
        let d2 = n - 1;
        let xs = x.as_slice();
        let mut y = vec![ZERO; n];
        self.hz.apply(xs, &mut y);
        self.derivative.jumps.apply_add(xs, ONE, &mut y);
        apply_redfield(
            &self.derivative.redfield,
            self.places,
            xs,
            -ONE,
            false,
            &mut y[..d2],
        );
        if xs[d2] != ZERO {
            for (o, f) in y.iter_mut().zip(&self.derivative.drift) {
                *o += f * xs[d2];
            }
        }
        lambda.iter().zip(&y).map(|(l, v)| l.conj() * v).sum()
    }
}

impl SpinSystem {
    /// `J(B)` on `[vec Δ; Tr ρ]`: the jumps and their drift.
    fn jump_block(&self, liouville: &Liouville, b: f64) -> SparseBlock {
        let d = self.dimension();
        let relaxation = self.jump_relaxation(b);
        let zero = Sparse::new(&CMatrix::zeros(d, d));
        self.sparse_block(liouville, 0, &zero, &relaxation, ZERO, Some(ONE))
    }

    /// `ρ_eq − 1/d` at field `b` (unit trace), as `vec`.
    fn equilibrium_vector(&self, b: f64) -> Vec<C64> {
        let d = self.dimension();
        let mut v = vec![ZERO; d * d];
        for (i, x) in self.thermal_deviation_at(b).into_iter().enumerate() {
            v[i * d + i] = C64::new(x, 0.0);
        }
        v
    }

    /// The generator's parts at field `b`.
    fn at_field(&self, liouville: &Liouville, b: f64) -> Result<(AtField, Vec<Vec<u32>>)> {
        let d = self.dimension();
        // Where the shielding alone relaxes, it vanishes at zero field: the
        // terms of `redfield_field` hold there (and are empty).
        let redfield = match self.redfield(b)? {
            Some(r) => r,
            None => {
                self.redfield_field(b)?
                    .ok_or_else(|| {
                        QsoError::invalid(
                            "control",
                            "matrix-free relaxation needs Redfield relaxation",
                        )
                    })?
                    .0
            }
        };
        let places = positions(&redfield, d);
        let mut drift = vec![ZERO; d * d];
        let eq = self.equilibrium_vector(b);
        apply_redfield(&redfield, &places, &eq, ONE, false, &mut drift);
        let jumps = self.jump_block(liouville, b);
        let jumps_adjoint = jumps.adjoint();
        let norm = norm_bound(&jumps) + redfield.rate();
        Ok((
            AtField {
                jumps,
                jumps_adjoint,
                redfield,
                drift,
                norm,
            },
            places,
        ))
    }

    /// The field derivatives at `b`, with `span` the range for the
    /// differences of the jumps and the thermal populations.
    fn derivative_at(&self, liouville: &Liouville, b: f64, span: f64) -> Result<Derivative> {
        let d = self.dimension();
        let (r, dr) = self.redfield_field(b)?.ok_or_else(|| {
            QsoError::invalid(
                "control",
                "matrix-free relaxation needs Redfield relaxation",
            )
        })?;
        let places = positions(&r, d);
        let h = 1e-2 * b.abs().max(span);
        // Richardson: (4 D(h/2) − D(h))/3 with D(s) the central difference.
        let jumps = {
            let family = SparseFamily::new(&[
                self.jump_block(liouville, b + h / 2.0),
                self.jump_block(liouville, b - h / 2.0),
                self.jump_block(liouville, b + h),
                self.jump_block(liouville, b - h),
            ]);
            let (a, c) = (4.0 / (3.0 * h), 1.0 / (6.0 * h));
            family.combine(&[a, -a, -c, c])
        };
        let eq = self.equilibrium_vector(b);
        let deq: Vec<C64> = {
            let [p1, m1, p2, m2] =
                [b + h / 2.0, b - h / 2.0, b + h, b - h].map(|f| self.equilibrium_vector(f));
            let (a, c) = (4.0 / (3.0 * h), 1.0 / (6.0 * h));
            (0..d * d)
                .map(|i| (p1[i] - m1[i]) * a - (p2[i] - m2[i]) * c)
                .collect()
        };
        let mut drift = vec![ZERO; d * d];
        apply_redfield(&dr, &places, &eq, ONE, false, &mut drift);
        apply_redfield(&r, &places, &deq, ONE, false, &mut drift);
        Ok(Derivative {
            jumps,
            redfield: dr,
            drift,
        })
    }

    /// The parts of a field profile with Redfield relaxation applied
    /// without storing it, and its cost per application from the
    /// relaxation at the middle of the range.
    pub(in crate::spin::control) fn free_field(&self, c: &Control) -> Result<FreeField> {
        let Drive::Field { min, max } = c.drive else {
            return Err(QsoError::invalid("control", "a field profile"));
        };
        let d = self.dimension();
        let (h0, dh) = self.control_hamiltonians(c, &Member::ideal())?;
        let ops = [commutator_block(&h0), commutator_block(&dh[0])];
        let ops_adjoint = [ops[0].adjoint(), ops[1].adjoint()];
        let op_norms = [norm_bound(&ops[0]), norm_bound(&ops[1])];
        let middle = 0.5 * (min + max);
        // The ends first: `redfield` refuses beyond its limits before it
        // builds anything.
        let mut ends = 0.0f64;
        for b in [min, max] {
            if let Some(rb) = self.redfield(b)? {
                ends = ends.max(rb.rate());
            }
        }
        let (r, dr) = self.redfield_field(middle)?.ok_or_else(|| {
            QsoError::invalid(
                "control",
                "matrix-free relaxation needs Redfield relaxation",
            )
        })?;
        let work = |r: &RedfieldBlocks| {
            (0..r.liouville.blocks.len())
                .map(|q| 12.0 * r.products_of(q) as f64)
                .sum::<f64>()
        };
        let apply_work = [work(&r), work(&dr)];
        let scratch = |r: &RedfieldBlocks| {
            r.liouville
                .blocks
                .iter()
                .map(|(_, e)| r.apply_bytes(e.len()))
                .fold(0.0, f64::max)
        };
        let terms_bytes =
            (r.terms_bytes() + dr.terms_bytes()) as f64 + scratch(&r).max(scratch(&dr));
        // The rate bound at the ends and the middle.
        let mut relaxation_norm = r.rate().max(ends);
        let liouville = Liouville::new(&vec![Vec::new(); d], None);
        relaxation_norm += norm_bound(&self.jump_block(&liouville, max));
        let start = augment(&c.initial);
        let target = DVector::from_iterator(
            d * d + 1,
            super::super::vectorise(&c.target)
                .iter()
                .copied()
                .chain(std::iter::once(c.target.trace() / d as f64)),
        );
        Ok(FreeField {
            ops,
            ops_adjoint,
            op_norms,
            start,
            target,
            d,
            span: max - min,
            apply_work,
            terms_bytes,
            relaxation_norm,
        })
    }
}

impl FreeField {
    /// Substeps of a step at field `b`.
    fn substeps(&self, c: &Control, b: f64) -> f64 {
        let norm = self.relaxation_norm + self.op_norms[0] + b.abs() * self.op_norms[1];
        (norm * c.step() / SUBSTEP).ceil().max(1.0)
    }

    /// Estimated operations and bytes of one evaluation under the fields
    /// `u`.
    pub(in crate::spin::control) fn cost(&self, c: &Control, u: &[f64]) -> (f64, f64) {
        let m = (self.d * self.d + 1) as f64;
        let ops = (self.ops[0].nnz() + self.ops[1].nnz()) as f64;
        let product = 8.0 * ops + 8.0 * m + self.apply_work[0];
        let pairing = 8.0 * ops + 8.0 * m + self.apply_work[1];
        let (mut work, mut most) = (0.0, 0.0f64);
        for &b in u {
            let s = self.substeps(c, b);
            // Forward twice and backward once, the pairing at ten nodes per
            // substep, and about six builds of the terms per step (R forward
            // and backward, R' with its two spectral transforms).
            work += s * (3.0 * TAYLOR_TERMS * product + GAUSS.len() as f64 * pairing);
            most = most.max(s);
        }
        work += 6.0 * c.steps as f64 * 50.0 * self.terms_bytes;
        let nodes = GAUSS.len() as f64 * most;
        let bytes = 16.0 * m * (c.steps as f64 + 1.0 + nodes + 30.0)
            + 2.0 * self.terms_bytes
            + 6.0 * 20.0 * ops;
        (work, bytes)
    }

    /// Transfer and gradient under the fields `u` (T).
    pub(in crate::spin::control) fn member(
        &self,
        sys: &SpinSystem,
        c: &Control,
        u: &[f64],
        budget: &Budget,
    ) -> Result<(f64, Vec<f64>)> {
        let dt = c.step();
        let liouville = Liouville::new(&vec![Vec::new(); self.d], None);
        let m = (self.d * self.d + 1) as f64;
        let ops = (self.ops[0].nnz() + self.ops[1].nnz()) as f64;
        let product = 8.0 * ops + 8.0 * m + self.apply_work[0];
        let pairing = 8.0 * ops + 8.0 * m + self.apply_work[1];
        let mut states = Vec::with_capacity(u.len() + 1);
        states.push(self.start.clone());
        for (k, &b) in u.iter().enumerate() {
            let (at, places) = sys.at_field(&liouville, b)?;
            let forward = FreeGenerator {
                ops: &self.ops,
                jumps: &at.jumps,
                at: &at,
                places: &places,
                b,
                adjoint: false,
            };
            let norm = at.norm + self.op_norms[0] + b.abs() * self.op_norms[1];
            let (next, products) = propagate(&forward, &states[k], dt, norm);
            budget.spend(products as f64 * product)?;
            states.push(next);
        }
        let value = self.target.dotc(&states[u.len()]).re;
        let mut grad = vec![0.0; u.len()];
        let mut lambda = self.target.clone();
        for k in (0..u.len()).rev() {
            let b = u[k];
            let (at, places) = sys.at_field(&liouville, b)?;
            let derivative = sys.derivative_at(&liouville, b, self.span)?;
            let forward = FreeGenerator {
                ops: &self.ops,
                jumps: &at.jumps,
                at: &at,
                places: &places,
                b,
                adjoint: false,
            };
            let backward = FreeGenerator {
                ops: &self.ops_adjoint,
                jumps: &at.jumps_adjoint,
                at: &at,
                places: &places,
                b,
                adjoint: true,
            };
            let direction = FreeDirection {
                hz: &self.ops[1],
                derivative: &derivative,
                places: &places,
            };
            let norm = at.norm + self.op_norms[0] + b.abs() * self.op_norms[1];
            let (g, previous, (products, substeps)) = step_gradient(
                &forward,
                &backward,
                &[direction],
                &states[k],
                &lambda,
                dt,
                norm,
            );
            budget.spend(
                products as f64 * product + substeps as f64 * GAUSS.len() as f64 * pairing,
            )?;
            grad[k] = g[0];
            lambda = previous;
        }
        Ok((value, grad))
    }
}
