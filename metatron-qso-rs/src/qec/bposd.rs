//! Belief propagation with ordered-statistics post-processing (BP+OSD)
//! on a detector error model whose errors may flip any number of
//! detectors.
//!
//! Belief propagation runs on the Tanner graph of the model (detectors
//! as checks, errors as bits) in the log-likelihood domain with the
//! flooding schedule, by the min-sum rule (optionally scaled) or the
//! product-sum rule; it stops as soon as its hard decision explains the
//! detection events. Otherwise ordered-statistics decoding sorts the
//! errors by their posterior log-likelihood ratio, most likely first,
//! takes the first independent columns as an information set, and solves
//! for them (OSD-0). The combination sweep (OSD-CS) also tries every
//! single error outside the set and every pair among the first
//! `osd_order` of them, and keeps the candidate of largest prior
//! likelihood; the exhaustive variant tries every subset of the first
//! `osd_order`. This follows Roffe et al., "Decoding across the quantum
//! LDPC code landscape" (2020), and the `ldpc` package; candidates are
//! compared by their log-likelihood Σ ln((1 − p)/p), where `ldpc` sums
//! ln(1/p).

use super::dem::DetectorErrorModel;
use super::gf2::{self, Basis};
use crate::error::{QsoError, Result};
use serde::{Deserialize, Serialize};

/// Most detectors the decoder accepts: ordered-statistics decoding keeps
/// a dense basis of up to this many vectors of this many bits.
pub const MAX_BPOSD_DETECTORS: usize = 1 << 14;
/// Largest `osd_order` of the exhaustive variant (2^order candidates).
pub const MAX_EXHAUSTIVE_ORDER: usize = 20;
/// Largest `osd_order` of the combination sweep.
pub const MAX_SWEEP_ORDER: usize = 1000;
/// Messages are kept within ±this, so sums stay finite.
const LLR_LIMIT: f64 = 1e100;

/// The rule a check uses for its messages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BpMethod {
    /// Sign product times the smallest incoming magnitude, scaled.
    #[default]
    MinSum,
    /// 2 artanh of the product of tanh(m/2): exact on trees.
    ProductSum,
}

/// The post-processing when belief propagation does not converge.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OsdMethod {
    /// Order 0: the information-set solution only.
    Zero,
    /// Every single error outside the information set, and every pair
    /// among the first `osd_order` of them.
    #[default]
    CombinationSweep,
    /// Every subset of the first `osd_order` errors outside the set.
    Exhaustive,
}

/// Options of [`BpOsdDecoder`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BpOsdOptions {
    /// Iterations of belief propagation; 0 skips it, so that OSD orders
    /// the errors by their prior probabilities.
    pub max_iterations: usize,
    pub method: BpMethod,
    /// Factor of min-sum messages, in (0, 1]; 0 means 1 − 2^(−t) in
    /// iteration t (as in `ldpc`).
    pub scaling: f64,
    pub osd: OsdMethod,
    /// Order of the combination sweep or the exhaustive search.
    pub osd_order: usize,
}

impl Default for BpOsdOptions {
    /// Min-sum with factor 0.625, 50 iterations, OSD-CS of order 7.
    fn default() -> Self {
        BpOsdOptions {
            max_iterations: 50,
            method: BpMethod::MinSum,
            scaling: 0.625,
            osd: OsdMethod::CombinationSweep,
            osd_order: 7,
        }
    }
}

/// What one decoding found.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BpOsdOutcome {
    /// Predicted observable flips.
    pub observables: u64,
    /// Indices (into the model's errors) of the correction, ascending: it
    /// flips exactly the fired detectors.
    pub errors: Vec<usize>,
    /// Whether belief propagation alone explained the detection events.
    pub converged: bool,
    /// Iterations of belief propagation that ran.
    pub iterations: usize,
    /// Σ ln((1 − p)/p) over the correction.
    pub weight: f64,
}

/// A BP+OSD decoder of a detector error model.
#[derive(Clone, Debug)]
pub struct BpOsdDecoder {
    options: BpOsdOptions,
    n_detectors: usize,
    /// Model error of each column (errors of probability 0 and errors
    /// that flip no detector are not columns).
    source: Vec<usize>,
    observables: Vec<u64>,
    /// Prior log-likelihood ratio ln((1 − p)/p) of each column.
    prior: Vec<f64>,
    /// Column `j` has edges `col_start[j]..col_start[j + 1]`, to detectors
    /// `edge_row`.
    col_start: Vec<usize>,
    edge_row: Vec<usize>,
    /// Edges of detector `i`: `row_edges[row_start[i]..row_start[i + 1]]`.
    row_start: Vec<usize>,
    row_edges: Vec<usize>,
    /// Rank of the check matrix.
    rank: usize,
    // Work space.
    to_check: Vec<f64>,
    to_bit: Vec<f64>,
    posterior: Vec<f64>,
    hard: Vec<bool>,
    syndrome: Vec<bool>,
    candidate: Vec<bool>,
    basis: Basis,
}

impl BpOsdDecoder {
    /// The decoder of `dem` (errors of probability 0 and errors that flip
    /// no detector are left out).
    pub fn new(dem: &DetectorErrorModel, options: BpOsdOptions) -> Result<Self> {
        check_options(&options)?;
        let m = dem.n_detectors;
        if m > MAX_BPOSD_DETECTORS {
            return Err(QsoError::invalid(
                "dem",
                format!("{m} detectors; BP+OSD takes at most {MAX_BPOSD_DETECTORS}"),
            ));
        }
        let (mut source, mut observables, mut prior) = (Vec::new(), Vec::new(), Vec::new());
        let mut col_start = vec![0];
        let mut edge_row = Vec::new();
        for (k, e) in dem.errors.iter().enumerate() {
            let p = e.probability;
            if !(0.0..=1.0).contains(&p) {
                return Err(QsoError::invalid(
                    "dem",
                    format!("error {k} has probability {p}"),
                ));
            }
            if let Some(&d) = e.detectors.iter().find(|&&d| d >= m) {
                return Err(QsoError::invalid(
                    "dem",
                    format!("error {k} flips detector {d} of {m}"),
                ));
            }
            if e.detectors.windows(2).any(|w| w[0] >= w[1]) {
                return Err(QsoError::invalid(
                    "dem",
                    format!("the detectors of error {k} must be ascending and distinct"),
                ));
            }
            if p == 0.0 || e.detectors.is_empty() {
                continue;
            }
            source.push(k);
            observables.push(e.observables);
            // ln(1 − p) − ln p stays finite for subnormal p; p = 1 gives
            // −LLR_LIMIT.
            prior.push(((-p).ln_1p() - p.ln()).clamp(-LLR_LIMIT, LLR_LIMIT));
            edge_row.extend_from_slice(&e.detectors);
            col_start.push(edge_row.len());
        }
        let n = source.len();
        let mut row_count = vec![0usize; m + 1];
        for &r in &edge_row {
            row_count[r + 1] += 1;
        }
        let mut row_start = row_count;
        for i in 0..m {
            row_start[i + 1] += row_start[i];
        }
        let mut fill = row_start.clone();
        let mut row_edges = vec![0; edge_row.len()];
        for (e, &r) in edge_row.iter().enumerate() {
            row_edges[fill[r]] = e;
            fill[r] += 1;
        }
        // Rank of the check matrix, for stopping OSD early.
        let mut basis = Basis::new(m, true, m);
        for j in 0..n {
            if basis.rank() == m {
                break;
            }
            basis.insert(&gf2::from_ones(
                m,
                &edge_row[col_start[j]..col_start[j + 1]],
            ));
        }
        let rank = basis.rank();
        basis.clear();
        let edges = edge_row.len();
        Ok(BpOsdDecoder {
            options,
            n_detectors: m,
            source,
            observables,
            prior,
            col_start,
            edge_row,
            row_start,
            row_edges,
            rank,
            to_check: vec![0.0; edges],
            to_bit: vec![0.0; edges],
            posterior: vec![0.0; n],
            hard: vec![false; n],
            syndrome: vec![false; m],
            candidate: vec![false; m],
            basis,
        })
    }

    pub fn options(&self) -> &BpOsdOptions {
        &self.options
    }

    pub fn n_detectors(&self) -> usize {
        self.n_detectors
    }

    /// Predicted observable flips for the fired detectors (a detector
    /// listed twice cancels).
    pub fn decode(&mut self, fired: &[usize]) -> Result<u64> {
        Ok(self.decode_detailed(fired)?.observables)
    }

    /// The correction, and how it was found.
    pub fn decode_detailed(&mut self, fired: &[usize]) -> Result<BpOsdOutcome> {
        let m = self.n_detectors;
        if let Some(&d) = fired.iter().find(|&&d| d >= m) {
            return Err(QsoError::invalid(
                "fired",
                format!("detector {d} does not exist ({m} detectors)"),
            ));
        }
        self.syndrome.fill(false);
        for &d in fired {
            self.syndrome[d] ^= true;
        }
        if !self.syndrome.iter().any(|&s| s) {
            return Ok(BpOsdOutcome {
                observables: 0,
                errors: Vec::new(),
                converged: true,
                iterations: 0,
                weight: 0.0,
            });
        }
        let (iterations, converged) = self.propagate();
        let columns: Vec<usize> = if converged {
            (0..self.source.len()).filter(|&j| self.hard[j]).collect()
        } else {
            if self.options.max_iterations == 0 {
                self.posterior.copy_from_slice(&self.prior);
            }
            self.ordered_statistics()?
        };
        let mut errors: Vec<usize> = columns.iter().map(|&j| self.source[j]).collect();
        errors.sort_unstable();
        Ok(BpOsdOutcome {
            observables: columns.iter().fold(0, |o, &j| o ^ self.observables[j]),
            weight: columns.iter().map(|&j| self.prior[j]).sum(),
            errors,
            converged,
            iterations,
        })
    }

    /// Belief propagation, flooding schedule; returns the iterations run
    /// and whether the hard decision explains the syndrome. Leaves the
    /// posterior and the hard decision of the last iteration.
    fn propagate(&mut self) -> (usize, bool) {
        let n = self.source.len();
        for j in 0..n {
            for e in self.col_start[j]..self.col_start[j + 1] {
                self.to_check[e] = self.prior[j];
            }
        }
        for it in 1..=self.options.max_iterations {
            self.check_messages(it);
            // Posteriors, with the bit-to-check messages as prefix and
            // suffix sums (no subtraction of large values).
            self.candidate.fill(false);
            for j in 0..n {
                let (a, b) = (self.col_start[j], self.col_start[j + 1]);
                let mut sum = self.prior[j];
                for e in a..b {
                    self.to_check[e] = sum;
                    sum += self.to_bit[e];
                }
                let mut suffix = 0.0;
                for e in (a..b).rev() {
                    self.to_check[e] = (self.to_check[e] + suffix).clamp(-LLR_LIMIT, LLR_LIMIT);
                    suffix += self.to_bit[e];
                }
                self.posterior[j] = sum;
                self.hard[j] = sum <= 0.0;
                if self.hard[j] {
                    for e in a..b {
                        self.candidate[self.edge_row[e]] ^= true;
                    }
                }
            }
            if self.candidate == self.syndrome {
                return (it, true);
            }
        }
        (self.options.max_iterations, false)
    }

    fn check_messages(&mut self, iteration: usize) {
        let alpha = if self.options.scaling == 0.0 {
            1.0 - 0.5f64.powi(iteration.min(1000) as i32)
        } else {
            self.options.scaling
        };
        for i in 0..self.n_detectors {
            let edges = &self.row_edges[self.row_start[i]..self.row_start[i + 1]];
            match self.options.method {
                BpMethod::MinSum => {
                    let mut negative = self.syndrome[i];
                    let (mut min1, mut min2, mut arg) = (LLR_LIMIT, LLR_LIMIT, usize::MAX);
                    for &e in edges {
                        let q = self.to_check[e];
                        negative ^= q <= 0.0;
                        let a = q.abs();
                        if a < min1 {
                            (min2, min1, arg) = (min1, a, e);
                        } else if a < min2 {
                            min2 = a;
                        }
                    }
                    for &e in edges {
                        let others = if e == arg { min2 } else { min1 };
                        let sign = negative ^ (self.to_check[e] <= 0.0);
                        let r = alpha * others;
                        self.to_bit[e] = if sign { -r } else { r };
                    }
                }
                BpMethod::ProductSum => {
                    // Forward and backward products of tanh(q/2), so that
                    // no division is needed.
                    let mut product = 1.0;
                    for &e in edges {
                        self.to_bit[e] = product;
                        product *= (self.to_check[e] / 2.0).tanh();
                    }
                    let mut product = 1.0;
                    for &e in edges.iter().rev() {
                        let t = self.to_bit[e] * product;
                        product *= (self.to_check[e] / 2.0).tanh();
                        let r = if t.abs() >= 1.0 {
                            LLR_LIMIT.copysign(t)
                        } else {
                            // 2 artanh t, in the form `ldpc` uses.
                            ((1.0 + t) / (1.0 - t)).ln().clamp(-LLR_LIMIT, LLR_LIMIT)
                        };
                        self.to_bit[e] = if self.syndrome[i] { -r } else { r };
                    }
                }
            }
        }
    }

    /// Ordered-statistics decoding on the current posteriors; returns the
    /// columns of the correction.
    fn ordered_statistics(&mut self) -> Result<Vec<usize>> {
        let m = self.n_detectors;
        let n = self.source.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| self.posterior[a].total_cmp(&self.posterior[b]));
        // Information set: the first independent columns.
        self.basis.clear();
        let mut selected = Vec::with_capacity(self.rank);
        let mut rest = Vec::with_capacity(n.saturating_sub(self.rank));
        for &j in &order {
            if self.basis.rank() < self.rank && self.basis.insert(&self.column(j)).is_some() {
                selected.push(j);
            } else {
                rest.push(j);
            }
        }
        // OSD-0: the syndrome as a sum of members, each a sum of selected
        // columns.
        let mut s: Vec<u64> = vec![0; gf2::words(m)];
        for (i, _) in self.syndrome.iter().enumerate().filter(|(_, s)| **s) {
            gf2::flip(&mut s, i);
        }
        let mut x0 = vec![0u64; self.basis.combo_words()];
        self.basis.reduce(&mut s, Some(&mut x0));
        if s.iter().any(|&w| w != 0) {
            return Err(QsoError::invalid(
                "fired",
                "no set of errors of the model flips exactly these detectors",
            ));
        }
        let weight_of = |x: &[u64]| -> f64 { gf2::ones(x).map(|k| self.prior[selected[k]]).sum() };
        let mut best_flips: Vec<usize> = Vec::new();
        let mut best_x = x0.clone();
        let mut best = weight_of(&x0);
        let order_limit = self.options.osd_order.min(rest.len());
        // Order 0 is OSD-0 for every method, as in `ldpc`.
        let method = if self.options.osd_order == 0 {
            OsdMethod::Zero
        } else {
            self.options.osd
        };
        match method {
            OsdMethod::Zero => {}
            OsdMethod::CombinationSweep => {
                // Single flips of every column outside the set, by the
                // change in weight.
                // Weights are summed afresh for every candidate (not
                // updated by differences), so that equally likely
                // candidates tie exactly and the first is kept.
                let mut x = x0.clone();
                for &t in &rest {
                    let c = self.representation(t);
                    gf2::xor_into(&mut x, &c);
                    let w = weight_of(&x) + self.prior[t];
                    if w < best {
                        best = w;
                        best_flips = vec![t];
                        best_x.copy_from_slice(&x);
                    }
                    gf2::xor_into(&mut x, &c);
                }
                let reps: Vec<Vec<u64>> = rest[..order_limit]
                    .iter()
                    .map(|&t| self.representation(t))
                    .collect();
                for a in 0..order_limit {
                    for b in a + 1..order_limit {
                        let mut x = x0.clone();
                        gf2::xor_into(&mut x, &reps[a]);
                        gf2::xor_into(&mut x, &reps[b]);
                        let w = weight_of(&x) + self.prior[rest[a]] + self.prior[rest[b]];
                        if w < best {
                            best = w;
                            best_flips = vec![rest[a], rest[b]];
                            best_x = x;
                        }
                    }
                }
            }
            OsdMethod::Exhaustive => {
                let reps: Vec<Vec<u64>> = rest[..order_limit]
                    .iter()
                    .map(|&t| self.representation(t))
                    .collect();
                // Subsets in binary order (the first of equally likely
                // candidates is kept, as in `ldpc`).
                for flips in 1u64..1 << order_limit {
                    let mut x = x0.clone();
                    for k in (0..order_limit).filter(|&k| flips >> k & 1 == 1) {
                        gf2::xor_into(&mut x, &reps[k]);
                    }
                    let w = weight_of(&x)
                        + (0..order_limit)
                            .filter(|&k| flips >> k & 1 == 1)
                            .map(|k| self.prior[rest[k]])
                            .sum::<f64>();
                    if w < best {
                        best = w;
                        best_flips = (0..order_limit)
                            .filter(|&k| flips >> k & 1 == 1)
                            .map(|k| rest[k])
                            .collect();
                        best_x = x;
                    }
                }
            }
        }
        let mut columns: Vec<usize> = gf2::ones(&best_x).map(|k| selected[k]).collect();
        columns.extend(best_flips);
        Ok(columns)
    }

    fn column(&self, j: usize) -> Vec<u64> {
        gf2::from_ones(
            self.n_detectors,
            &self.edge_row[self.col_start[j]..self.col_start[j + 1]],
        )
    }

    /// Column `t` as a sum of selected columns: the members whose pivots
    /// it has (the basis is reduced and spans every column).
    fn representation(&self, t: usize) -> Vec<u64> {
        let mut c = vec![0u64; self.basis.combo_words()];
        for &r in &self.edge_row[self.col_start[t]..self.col_start[t + 1]] {
            if let Some(k) = self.basis.pivot_member(r) {
                gf2::xor_into(&mut c, self.basis.combo(k));
            }
        }
        c
    }
}

fn check_options(o: &BpOsdOptions) -> Result<()> {
    if !(0.0..=1.0).contains(&o.scaling) {
        return Err(QsoError::invalid(
            "scaling",
            format!("{} is not in [0, 1] (0: adaptive)", o.scaling),
        ));
    }
    let limit = match o.osd {
        OsdMethod::Zero => usize::MAX,
        OsdMethod::CombinationSweep => MAX_SWEEP_ORDER,
        OsdMethod::Exhaustive => MAX_EXHAUSTIVE_ORDER,
    };
    if o.osd_order > limit {
        return Err(QsoError::invalid(
            "osd_order",
            format!("at most {limit} for this OSD method"),
        ));
    }
    Ok(())
}

impl super::Decoder for BpOsdDecoder {
    fn decode(&mut self, fired: &[usize]) -> Result<u64> {
        BpOsdDecoder::decode(self, fired)
    }
}

#[cfg(test)]
mod tests;
