//! Field profiles of exchange on sparse blocks.
//!
//! Each joint block's generator at field `B` is `C₀ + B·C_Z + R(B)`: the
//! sites' coherent evolution `−i[H₀ + B·H_Z, ·]`, affine in the field, and
//! the rest `R(B)` (relaxation, exchange with the fresh thermal states of
//! its channels, and the drift column of the zero-order block), which is
//! not. As in 1.51, `R` is interpolated once per profile across its range
//! by the Chebyshev polynomial through the Chebyshev–Lobatto nodes (5, 9,
//! 17 or 33), until its two highest coefficients fall below
//! [`FIELD_TOLERANCE`] of its largest entry (and of the drift column's);
//! every step takes `R(B)` and `R'(B)` as weighted sums of the node
//! matrices, and the gradient pairs the costate with `C_Z + R'(B)` by the
//! Gauss–Legendre quadrature of 1.46.

use nalgebra::DVector;
use num_complex::Complex64 as C64;

use super::super::super::blocks::{Relaxation, Sparse};
use super::super::super::control::{
    Budget, GAUSS, LEVELS, SUBSTEP, Scaled, SparseGenerator, TAYLOR_TERMS, coefficient, norm_bound,
    propagate, step_gradient, weights,
};
use super::super::super::krylov::{SparseBlock, SparseFamily};
use super::super::super::{
    FIELD_TOLERANCE, MAX_FIELD_ERROR, MAX_RELAXATION_MEMORY, MAX_RELAXATION_WORK, ZERO,
};
use super::super::{Exchange, Joint, Profile};
use crate::error::{QsoError, Result};
use crate::linalg::CMatrix;

/// One joint block of a field profile: the coherent parts, the
/// interpolated rest, and the block's start and target.
struct FieldBlock {
    /// `C₀` and `C_Z` (per tesla), and their adjoints.
    ops: [SparseBlock; 2],
    ops_adjoint: [SparseBlock; 2],
    op_norms: [f64; 2],
    /// `R` at the nodes, on their shared pattern.
    family: SparseFamily,
    /// Bound on `‖R‖` over the nodes.
    relaxation_norm: f64,
    start: DVector<C64>,
    target: DVector<C64>,
}

/// What a field profile of an exchange needs on sparse blocks, built once
/// for any number of evaluations.
pub(in crate::spin) struct ExchangeField {
    blocks: Vec<FieldBlock>,
    nodes: usize,
    min: f64,
    max: f64,
    /// The interpolant's stated error, relative to the largest entry of
    /// `R` (and of the drift column).
    pub error: f64,
    /// `Σ_s Tr C_s / d_s`: the target's overlap with `c_s/d_s`.
    constant: f64,
}

impl ExchangeField {
    /// Nodes of the interpolant.
    pub fn nodes(&self) -> usize {
        self.nodes
    }

    /// `B` mapped to `[−1, 1]`.
    fn unit(&self, b: f64) -> f64 {
        ((2.0 * b - self.min - self.max) / (self.max - self.min)).clamp(-1.0, 1.0)
    }

    /// Estimated operations of one evaluation under the fields `u` over
    /// steps of `dt`.
    pub fn work(&self, u: &[f64], dt: f64) -> f64 {
        let mut work = 0.0;
        for block in &self.blocks {
            let nnz = (block.family.nnz() + block.ops[0].nnz() + block.ops[1].nnz()) as f64;
            let n = block.start.len() as f64;
            let product = 8.0 * nnz + 8.0 * n;
            let sums = 3.0 * 8.0 * self.nodes as f64 * block.family.nnz() as f64;
            for &b in u {
                let norm = block.relaxation_norm + block.op_norms[0] + b.abs() * block.op_norms[1];
                let s = (norm * dt / SUBSTEP).ceil().max(1.0);
                // Forward twice (the evaluation and the gradient's nodes),
                // backward once, and the pairing at ten nodes.
                work += s * (3.0 * TAYLOR_TERMS * product + GAUSS.len() as f64 * 16.0 * nnz) + sums;
            }
        }
        work
    }

    /// Estimated bytes of one evaluation under the fields `u`: the states
    /// at every step, and a step's states at the quadrature's nodes in
    /// every substep (the largest step's), besides the matrices.
    pub fn memory(&self, u: &[f64], dt: f64) -> f64 {
        let mut held = 0.0f64;
        let mut most = 0.0f64;
        for block in &self.blocks {
            let n = block.start.len() as f64;
            held += block.family.bytes() as f64
                + 2.0 * block.ops.iter().map(SparseBlock::bytes).sum::<usize>() as f64
                + 16.0 * n * (u.len() as f64 + 1.0)
                + 3.0 * 20.0 * block.family.nnz() as f64;
            for &b in u {
                let norm = block.relaxation_norm + block.op_norms[0] + b.abs() * block.op_norms[1];
                let s = (norm * dt / SUBSTEP).ceil().max(1.0);
                most = most.max(16.0 * n * (s * GAUSS.len() as f64 + 30.0));
            }
        }
        held + most
    }

    /// `Σ_s Re Tr(C_s† ρ_s(T))` under the fields `u` (T), steps of `dt`,
    /// and its gradient.
    pub fn transfer(&self, u: &[f64], dt: f64) -> Result<(f64, Vec<f64>)> {
        let budget = Budget::new(MAX_RELAXATION_WORK);
        let mut value = self.constant;
        let mut grad = vec![0.0; u.len()];
        for block in &self.blocks {
            let nnz = block.family.nnz() + block.ops[0].nnz() + block.ops[1].nnz();
            let size = block.start.len() as f64;
            let product = 8.0 * nnz as f64 + 8.0 * size;
            let sums = 3.0 * 8.0 * self.nodes as f64 * block.family.nnz() as f64;
            let norm = |r: &SparseBlock, b: f64| {
                norm_bound(r) + block.op_norms[0] + b.abs() * block.op_norms[1]
            };
            let mut states = Vec::with_capacity(u.len() + 1);
            states.push(block.start.clone());
            for (k, &b) in u.iter().enumerate() {
                let (a, _) = weights(self.nodes, self.unit(b));
                let r = block.family.combine(&a);
                let forward = SparseGenerator {
                    base: &r,
                    diagonal: None,
                    ops: &block.ops,
                    coefficients: vec![1.0, b],
                };
                let (next, products) = propagate(&forward, &states[k], dt, norm(&r, b));
                budget.spend(products as f64 * product + sums / 3.0)?;
                states.push(next);
            }
            value += block.target.dotc(&states[u.len()]).re;
            let mut lambda = block.target.clone();
            let pairing =
                GAUSS.len() as f64 * 8.0 * (2 * block.family.nnz() + block.ops[1].nnz()) as f64;
            let scale = 2.0 / (self.max - self.min);
            for k in (0..u.len()).rev() {
                let b = u[k];
                let (a, da) = weights(self.nodes, self.unit(b));
                let da: Vec<f64> = da.iter().map(|x| x * scale).collect();
                let r = block.family.combine(&a);
                let r_adjoint = block.family.combine_adjoint(&a);
                let derivative = block.family.combine(&da);
                let forward = SparseGenerator {
                    base: &r,
                    diagonal: None,
                    ops: &block.ops,
                    coefficients: vec![1.0, b],
                };
                let backward = SparseGenerator {
                    base: &r_adjoint,
                    diagonal: None,
                    ops: &block.ops_adjoint,
                    coefficients: vec![1.0, b],
                };
                let directions = [Scaled(&block.ops[1], 1.0), Scaled(&derivative, 1.0)];
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
                budget.spend(
                    products as f64 * (product + horner) + substeps as f64 * pairing + sums,
                )?;
                grad[k] += g[0] + g[1];
                lambda = previous;
            }
        }
        Ok((value, grad))
    }
}

impl Exchange {
    /// The coherent parts `−i[H, ·]` of the joint blocks for the sites'
    /// Hamiltonians `h` (no relaxation, no exchange), `m × m` per block
    /// (`m` the block's elements, plus the drift's coordinate for a
    /// zero-order block).
    fn coherent_blocks(&self, joint: &Joint, h: &[CMatrix], sizes: &[usize]) -> Vec<SparseBlock> {
        let mut triplets: Vec<Vec<(u32, u32, C64)>> = vec![Vec::new(); joint.blocks.len()];
        for (s, site) in self.sites.iter().enumerate() {
            let sys = &site.system;
            let none = Relaxation::none(sys.dimension());
            let sparse = Sparse::new(&h[s]);
            sys.generator_entries(&joint.liouvilles[s], &sparse, &none, |k, r, c, v| {
                let (j, offset) = joint.place[s][k];
                triplets[j].push(((offset + r) as u32, (offset + c) as u32, v));
            });
        }
        triplets
            .into_iter()
            .zip(sizes)
            .map(|(t, &m)| SparseBlock::from_triplets(m, t))
            .collect()
    }

    /// `R(B)` of every joint block: the generator without the sites'
    /// Hamiltonians, the drift in the last column of a zero-order block.
    fn rest_blocks(&self, joint: &Joint, b: f64, sizes: &[usize]) -> Result<Vec<SparseBlock>> {
        let parts: Vec<(Sparse, Relaxation)> = self
            .sites
            .iter()
            .map(|site| {
                let d = site.system.dimension();
                Ok((
                    Sparse::new(&CMatrix::zeros(d, d)),
                    site.system.relaxation(b)?,
                ))
            })
            .collect::<Result<_>>()?;
        let mut triplets: Vec<Vec<(u32, u32, C64)>> = vec![Vec::new(); joint.blocks.len()];
        self.terms(joint, b, &parts, |j, r, c, v| {
            triplets[j].push((r as u32, c as u32, v));
        });
        Ok(triplets
            .into_iter()
            .zip(sizes)
            .map(|(t, &m)| SparseBlock::from_triplets(m, t))
            .collect())
    }

    /// The sparse blocks of a field profile, `R` interpolated over the
    /// profile's range (see the module documentation); refused when the
    /// interpolant's stated error exceeds [`MAX_FIELD_ERROR`] or its next
    /// level would exceed the memory or work limits.
    pub(in crate::spin) fn exchange_field(&self, p: &Profile) -> Result<ExchangeField> {
        let joint = self.joint(None);
        let x0 = self.deviations(&p.initial);
        let mut constant = 0.0;
        for (c, site) in p.target.iter().zip(&self.sites) {
            constant += c.trace().re / site.system.dimension() as f64;
        }
        let sizes: Vec<usize> = joint
            .blocks
            .iter()
            .map(|(order, _, n)| n + usize::from(order.iter().all(|&q| q == 0)))
            .collect();
        // Starts and targets per block; blocks the target does not see are
        // left out.
        let mut keep = Vec::new();
        let mut starts = Vec::new();
        let mut targets = Vec::new();
        for ((_, parts, n), &m) in joint.blocks.iter().zip(&sizes) {
            let mut x = DVector::zeros(m);
            let mut l = DVector::zeros(m);
            let mut i = 0;
            for &(s, k, _) in parts {
                let c = self.sites[s].concentration;
                for &(a, b) in &joint.liouvilles[s].blocks[k].1 {
                    x[i] = x0[s][(a, b)];
                    l[i] = p.target[s][(a, b)] / c;
                    i += 1;
                }
            }
            if m > *n {
                x[*n] = C64::new(1.0, 0.0);
            }
            keep.push(l.iter().any(|z| *z != ZERO));
            starts.push(x);
            targets.push(l);
        }
        let h0: Vec<CMatrix> = self
            .sites
            .iter()
            .map(|s| s.system.hamiltonian_at(0.0))
            .collect();
        let hz: Vec<CMatrix> = self
            .sites
            .iter()
            .zip(&h0)
            .map(|(s, h)| s.system.hamiltonian_at(1.0) - h)
            .collect();
        let c0 = self.coherent_blocks(&joint, &h0, &sizes);
        let cz = self.coherent_blocks(&joint, &hz, &sizes);
        let (min, max) = (p.min, p.max);
        let field = |x: f64| 0.5 * (min + max) + 0.5 * (max - min) * x;
        let angle =
            |j: usize, levels: usize| (std::f64::consts::PI * j as f64 / (levels - 1) as f64).cos();
        // The drift's coordinate stays 1: scaled to the deviation, the
        // drift column's sum over the populations would enter the norm
        // bound of the Taylor substeps (and double their number) while
        // the series keep the deviation's digits anyway (the coordinate is
        // constant, its row empty).
        // Node builds per kept block.
        let build = |b: f64| -> Result<Vec<SparseBlock>> {
            Ok(self
                .rest_blocks(&joint, b, &sizes)?
                .into_iter()
                .zip(&keep)
                .filter_map(|(r, &k)| k.then_some(r))
                .collect())
        };
        let mut nodes: Vec<Vec<SparseBlock>> = (0..LEVELS[0])
            .map(|j| build(field(angle(j, LEVELS[0]))))
            .collect::<Result<_>>()?;
        let kept_sizes: Vec<(usize, usize)> = joint
            .blocks
            .iter()
            .zip(&sizes)
            .zip(&keep)
            .filter_map(|(((_, _, n), &m), &k)| k.then_some((*n, m)))
            .collect();
        let entries: f64 = nodes[0].iter().map(|b| b.nnz() as f64).sum();
        let mut built = LEVELS[0] as f64 * 60.0 * entries;
        let mut level = 0;
        let mut short = None;
        let (families, error) = loop {
            let count = LEVELS[level];
            let families: Vec<SparseFamily> = (0..kept_sizes.len())
                .map(|q| {
                    let members: Vec<SparseBlock> = nodes.iter().map(|n| n[q].clone()).collect();
                    SparseFamily::new(&members)
                })
                .collect();
            let mut error = 0.0f64;
            for (q, family) in families.iter().enumerate() {
                let (n, _) = kept_sizes[q];
                let (largest, largest_drift) = nodes
                    .iter()
                    .map(|b| b[q].largest_split(n))
                    .fold((0.0f64, 0.0f64), |(a, b), (x, y)| (a.max(x), b.max(y)));
                for m in [count - 1, count - 2] {
                    let (tail, tail_drift) =
                        family.combine(&coefficient(count, m)).largest_split(n);
                    let rest = if largest > 0.0 { tail / largest } else { 0.0 };
                    let drift = if largest_drift > 0.0 {
                        tail_drift / largest_drift
                    } else {
                        0.0
                    };
                    error = error.max(rest.max(drift));
                }
            }
            if error <= FIELD_TOLERANCE || level + 1 == LEVELS.len() {
                break (families, error);
            }
            let next = LEVELS[level + 1];
            let per_node: f64 = nodes[0].iter().map(|b| b.bytes() as f64).sum();
            let union: f64 = families.iter().map(|f| f.nnz() as f64).sum();
            let next_bytes = next as f64 * per_node + union * (12.0 + 16.0 * next as f64);
            let next_work = built + (next - count) as f64 * 60.0 * entries;
            if next_bytes > MAX_RELAXATION_MEMORY {
                short = Some(format!(
                    "the next level of {next} fields needs about {:.1} GB (limit {:.0} GB)",
                    next_bytes / 1e9,
                    MAX_RELAXATION_MEMORY / 1e9
                ));
                break (families, error);
            }
            if next_work > MAX_RELAXATION_WORK {
                short = Some(format!(
                    "the next level of {next} fields needs about {next_work:.1e} operations to \
                     build (limit {MAX_RELAXATION_WORK:.1e})"
                ));
                break (families, error);
            }
            built = next_work;
            drop(families);
            let mut grown = Vec::with_capacity(next);
            let mut old = nodes.into_iter();
            for j in 0..next {
                if j % 2 == 0 {
                    grown.push(old.next().expect("an old node"));
                } else {
                    grown.push(build(field(angle(j, next)))?);
                }
            }
            nodes = grown;
            level += 1;
        };
        if let Some(why) = short {
            return Err(QsoError::invalid(
                "field",
                format!(
                    "the relaxation and exchange over {min} to {max} T through {} fields err by \
                     {error:.1e} (tolerance {FIELD_TOLERANCE:e}), and {why}; use fewer spins or \
                     a narrower range",
                    LEVELS[level]
                ),
            ));
        }
        if error > MAX_FIELD_ERROR {
            return Err(QsoError::invalid(
                "field",
                format!(
                    "the relaxation and exchange vary too much over {min} to {max} T to \
                     interpolate: the interpolant through {} fields errs by {error:.1e} of its \
                     largest entry (at most {MAX_FIELD_ERROR:e}); narrow the range",
                    LEVELS[level]
                ),
            ));
        }
        let norms: Vec<f64> = (0..kept_sizes.len())
            .map(|q| nodes.iter().map(|n| norm_bound(&n[q])).fold(0.0, f64::max))
            .collect();
        drop(nodes);
        let mut blocks = Vec::new();
        let mut families = families.into_iter();
        let mut norms = norms.into_iter();
        for (j, &k) in keep.iter().enumerate() {
            if !k {
                continue;
            }
            let ops = [c0[j].clone(), cz[j].clone()];
            let ops_adjoint = [ops[0].adjoint(), ops[1].adjoint()];
            let op_norms = [norm_bound(&ops[0]), norm_bound(&ops[1])];
            blocks.push(FieldBlock {
                ops,
                ops_adjoint,
                op_norms,
                family: families.next().expect("a family"),
                relaxation_norm: norms.next().expect("a norm"),
                start: starts[j].clone(),
                target: targets[j].clone(),
            });
        }
        Ok(ExchangeField {
            blocks,
            nodes: LEVELS[level],
            min,
            max,
            error: error.max(f64::EPSILON),
            constant,
        })
    }
}
