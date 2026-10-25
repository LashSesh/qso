//! Readout-error mitigation.
//!
//! A readout calibration describes how measured bit strings relate to the
//! prepared ones by a confusion matrix `A[measured][prepared]`. Mitigation
//! applies `A⁻¹` to measured frequencies, giving quasi-probabilities that
//! sum to one but may be slightly negative;
//! [`nearest_probability_distribution`] projects them back.
//!
//! * **Tensored** calibrations assume independent errors per qubit: two
//!   circuits (all zeros, all ones) determine `P(read 1 | 0)` and
//!   `P(read 0 | 1)` for every qubit, and `A = ⊗ A_q`.
//! * **Full** calibrations measure all `2ⁿ` basis states and capture
//!   correlated errors (up to [`MAX_FULL_QUBITS`] qubits).
//!
//! ```
//! use metatron_qso::mitigation::ReadoutCalibration;
//! use metatron_qso::quantum::Counts;
//!
//! let cal = ReadoutCalibration::from_errors(&[(0.1, 0.2)]).unwrap();
//! // 1000 shots of |1⟩ read out as 200 zeros and 800 ones.
//! let counts: Counts = [(0, 200), (1, 800)].into();
//! let q = cal.mitigate(&counts).unwrap();
//! assert!((q[&1] - 1.0).abs() < 1e-12);
//! assert!(q.get(&0).copied().unwrap_or(0.0).abs() < 1e-12);
//! ```

use std::collections::BTreeMap;

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::backend::Backend;
use crate::error::{QsoError, Result};
use crate::quantum::{Circuit, Counts};
use crate::rng::Rng;

/// Largest register for full (correlated) calibrations: 2¹⁰ calibration
/// circuits and a 1024 × 1024 matrix.
pub const MAX_FULL_QUBITS: usize = 10;

/// Largest register whose counts are mitigated (dense 2ⁿ vectors).
pub const MAX_MITIGATED_QUBITS: usize = 24;

/// A readout calibration (see the module documentation).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReadoutCalibration {
    n_qubits: usize,
    model: ReadoutModel,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ReadoutModel {
    /// Per-qubit errors `(P(read 1 | 0), P(read 0 | 1))`.
    Tensored(Vec<(f64, f64)>),
    /// Column-major `2ⁿ × 2ⁿ` confusion matrix `A[measured][prepared]`.
    Full(Vec<f64>),
}

/// Per-qubit inverse confusion matrices, row-major
/// `[[a00, a01], [a10, a11]]` with `A⁻¹[prepared][measured]`.
fn inverse_2x2(p01: f64, p10: f64) -> [f64; 4] {
    // A = [[1 − p01, p10], [p01, 1 − p10]], det = 1 − p01 − p10.
    let det = 1.0 - p01 - p10;
    [(1.0 - p10) / det, -p10 / det, -p01 / det, (1.0 - p01) / det]
}

impl ReadoutCalibration {
    /// Tensored calibration from known per-qubit errors
    /// `(P(read 1 | 0), P(read 0 | 1))`.
    pub fn from_errors(errors: &[(f64, f64)]) -> Result<Self> {
        if errors.len() > MAX_MITIGATED_QUBITS {
            return Err(QsoError::TooManyQubits {
                what: "readout calibration",
                max: MAX_MITIGATED_QUBITS,
                requested: errors.len(),
            });
        }
        for (q, &(p01, p10)) in errors.iter().enumerate() {
            if !(0.0..=1.0).contains(&p01) || !(0.0..=1.0).contains(&p10) {
                return Err(QsoError::invalid(
                    "errors",
                    format!("qubit {q}: readout errors must be probabilities"),
                ));
            }
            if p01 + p10 >= 1.0 - 1e-9 {
                return Err(QsoError::invalid(
                    "errors",
                    format!("qubit {q}: P(1|0) + P(0|1) ≥ 1, the readout carries no information"),
                ));
            }
        }
        Ok(Self {
            n_qubits: errors.len(),
            model: ReadoutModel::Tensored(errors.to_vec()),
        })
    }

    /// Tensored calibration measured on `backend`: `shots` shots each of
    /// `|0…0⟩` and `|1…1⟩`.
    pub fn tensored(
        backend: &dyn Backend,
        n_qubits: usize,
        shots: usize,
        seed: u64,
    ) -> Result<Self> {
        if shots == 0 {
            return Err(QsoError::invalid("shots", "need at least one shot"));
        }
        let base = Rng::seed_from_u64(seed);
        let zeros = backend.run(&Circuit::new(n_qubits), &[], shots, base.fork(0).next_u64())?;
        let mut flip = Circuit::new(n_qubits);
        for q in 0..n_qubits {
            flip.x(q);
        }
        let ones = backend.run(&flip, &[], shots, base.fork(1).next_u64())?;
        let fraction = |counts: &Counts, q: usize, value: usize| {
            counts
                .iter()
                .filter(|&(&b, _)| (b >> q) & 1 == value)
                .map(|(_, &n)| n)
                .sum::<usize>() as f64
                / shots as f64
        };
        let errors: Vec<(f64, f64)> = (0..n_qubits)
            .map(|q| (fraction(&zeros, q, 1), fraction(&ones, q, 0)))
            .collect();
        Self::from_errors(&errors)
    }

    /// Full calibration measured on `backend`: `shots` shots of each of the
    /// `2ⁿ` basis states (at most [`MAX_FULL_QUBITS`] qubits).
    pub fn full(backend: &dyn Backend, n_qubits: usize, shots: usize, seed: u64) -> Result<Self> {
        if n_qubits > MAX_FULL_QUBITS {
            return Err(QsoError::TooManyQubits {
                what: "full readout calibration",
                max: MAX_FULL_QUBITS,
                requested: n_qubits,
            });
        }
        if shots == 0 {
            return Err(QsoError::invalid("shots", "need at least one shot"));
        }
        let dim = 1usize << n_qubits;
        let base = Rng::seed_from_u64(seed);
        let mut a = vec![0.0; dim * dim];
        for x in 0..dim {
            let mut c = Circuit::new(n_qubits);
            for q in 0..n_qubits {
                if (x >> q) & 1 == 1 {
                    c.x(q);
                }
            }
            let counts = backend.run(&c, &[], shots, base.fork(x as u64).next_u64())?;
            for (&b, &n) in &counts {
                a[b + x * dim] = n as f64 / shots as f64;
            }
        }
        Self::from_matrix(n_qubits, a)
    }

    /// Full calibration from a column-major confusion matrix
    /// `A[measured][prepared]` (entry `measured + prepared·2ⁿ`; columns are
    /// probability distributions).
    pub fn from_matrix(n_qubits: usize, matrix: Vec<f64>) -> Result<Self> {
        if n_qubits > MAX_FULL_QUBITS {
            return Err(QsoError::TooManyQubits {
                what: "full readout calibration",
                max: MAX_FULL_QUBITS,
                requested: n_qubits,
            });
        }
        let dim = 1usize << n_qubits;
        if matrix.len() != dim * dim {
            return Err(QsoError::DimensionMismatch {
                expected: dim * dim,
                actual: matrix.len(),
            });
        }
        for col in matrix.chunks(dim) {
            let total: f64 = col.iter().sum();
            if col.iter().any(|&p| !(0.0..=1.0).contains(&p)) || (total - 1.0).abs() > 1e-9 {
                return Err(QsoError::invalid(
                    "matrix",
                    "columns must be probability distributions",
                ));
            }
        }
        let m = DMatrix::from_column_slice(dim, dim, &matrix);
        if m.clone().try_inverse().is_none() {
            return Err(QsoError::invalid("matrix", "confusion matrix is singular"));
        }
        Ok(Self {
            n_qubits,
            model: ReadoutModel::Full(matrix),
        })
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Per-qubit errors `(P(read 1 | 0), P(read 0 | 1))` of a tensored
    /// calibration, or the marginals of a full one.
    pub fn qubit_errors(&self) -> Vec<(f64, f64)> {
        match &self.model {
            ReadoutModel::Tensored(e) => e.clone(),
            ReadoutModel::Full(a) => {
                let dim = 1usize << self.n_qubits;
                (0..self.n_qubits)
                    .map(|q| {
                        // Average over the other prepared bits.
                        let (mut p01, mut p10) = (0.0, 0.0);
                        for x in 0..dim {
                            let wrong: f64 = (0..dim)
                                .filter(|b| (b >> q) & 1 != (x >> q) & 1)
                                .map(|b| a[b + x * dim])
                                .sum();
                            if (x >> q) & 1 == 0 {
                                p01 += wrong;
                            } else {
                                p10 += wrong;
                            }
                        }
                        let half = (dim / 2) as f64;
                        (p01 / half, p10 / half)
                    })
                    .collect()
            }
        }
    }

    fn check_register(&self, counts: &Counts) -> Result<()> {
        if self.n_qubits > MAX_MITIGATED_QUBITS {
            return Err(QsoError::TooManyQubits {
                what: "readout mitigation",
                max: MAX_MITIGATED_QUBITS,
                requested: self.n_qubits,
            });
        }
        if let Some((&b, _)) = counts.iter().next_back()
            && b >> self.n_qubits != 0
        {
            return Err(QsoError::invalid(
                "counts",
                format!(
                    "outcome {b} has more bits than the {} calibrated qubits",
                    self.n_qubits
                ),
            ));
        }
        Ok(())
    }

    /// Apply `A⁻¹` to a dense vector over bit strings.
    fn apply_inverse(&self, v: &mut [f64]) {
        match &self.model {
            ReadoutModel::Tensored(errors) => {
                for (q, &(p01, p10)) in errors.iter().enumerate() {
                    let inv = inverse_2x2(p01, p10);
                    let bit = 1usize << q;
                    for b in 0..v.len() {
                        if b & bit == 0 {
                            let (x, y) = (v[b], v[b | bit]);
                            v[b] = inv[0] * x + inv[1] * y;
                            v[b | bit] = inv[2] * x + inv[3] * y;
                        }
                    }
                }
            }
            ReadoutModel::Full(a) => {
                let dim = v.len();
                let m = DMatrix::from_column_slice(dim, dim, a);
                let x = m
                    .lu()
                    .solve(&nalgebra::DVector::from_column_slice(v))
                    .expect("checked invertible");
                v.copy_from_slice(x.as_slice());
            }
        }
    }

    /// Quasi-probabilities of the prepared bit strings: `A⁻¹` applied to the
    /// measured frequencies (entries that vanish are left out).
    pub fn mitigate(&self, counts: &Counts) -> Result<BTreeMap<usize, f64>> {
        self.check_register(counts)?;
        let shots: usize = counts.values().sum();
        if shots == 0 {
            return Err(QsoError::invalid("counts", "no shots"));
        }
        let mut v = vec![0.0; 1 << self.n_qubits];
        for (&b, &n) in counts {
            v[b] = n as f64 / shots as f64;
        }
        self.apply_inverse(&mut v);
        Ok(v.into_iter()
            .enumerate()
            .filter(|&(_, x)| x.abs() > 1e-15)
            .collect())
    }

    /// Per-outcome weights `w(b)` such that `Σ_b f(b)·w(b)`, with measured
    /// frequencies `f`, is the mitigated expectation of the parity
    /// `(−1)^{|b ∧ mask|}`. For tensored calibrations
    /// `w(b) = Π_{q ∈ mask} (A_q⁻¹[0][b_q] − A_q⁻¹[1][b_q])`.
    pub(crate) fn parity_weights(&self, mask: usize) -> ParityWeights {
        match &self.model {
            ReadoutModel::Tensored(errors) => ParityWeights::Factors(
                errors
                    .iter()
                    .enumerate()
                    .filter(|(q, _)| (mask >> q) & 1 == 1)
                    .map(|(q, &(p01, p10))| {
                        let inv = inverse_2x2(p01, p10);
                        (q, [inv[0] - inv[2], inv[1] - inv[3]])
                    })
                    .collect(),
            ),
            ReadoutModel::Full(a) => {
                // w = A⁻ᵀ s with s(x) = (−1)^{|x ∧ mask|}.
                let dim = 1usize << self.n_qubits;
                let m = DMatrix::from_column_slice(dim, dim, a).transpose();
                let s = nalgebra::DVector::from_fn(dim, |x, _| {
                    if (x & mask).count_ones().is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    }
                });
                let w = m.lu().solve(&s).expect("checked invertible");
                ParityWeights::Dense(w.as_slice().to_vec())
            }
        }
    }
}

/// Weights of a mitigated parity (see
/// [`ReadoutCalibration::parity_weights`]).
#[derive(Clone, Debug)]
pub(crate) enum ParityWeights {
    /// `(qubit, [w if bit 0, w if bit 1])` for every qubit of the mask.
    Factors(Vec<(usize, [f64; 2])>),
    Dense(Vec<f64>),
}

impl ParityWeights {
    pub(crate) fn weight(&self, b: usize) -> f64 {
        match self {
            ParityWeights::Factors(f) => f.iter().map(|&(q, w)| w[(b >> q) & 1]).product(),
            ParityWeights::Dense(w) => w[b],
        }
    }
}

/// The probability distribution closest (in Euclidean norm) to a
/// quasi-probability distribution summing to one (Smolin, Gambetta and
/// Smith 2012).
pub fn nearest_probability_distribution(quasi: &BTreeMap<usize, f64>) -> BTreeMap<usize, f64> {
    let mut entries: Vec<(usize, f64)> = quasi.iter().map(|(&b, &p)| (b, p)).collect();
    // Descending by value.
    entries.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut accumulated = 0.0;
    let mut keep = entries.len();
    while keep > 0 {
        let (_, p) = entries[keep - 1];
        if p + accumulated / keep as f64 >= 0.0 {
            break;
        }
        accumulated += p;
        keep -= 1;
    }
    let shift = if keep > 0 {
        accumulated / keep as f64
    } else {
        0.0
    };
    entries[..keep]
        .iter()
        .map(|&(b, p)| (b, p + shift))
        .filter(|&(_, p)| p > 0.0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::DeviceBackend;
    use crate::noise::{DeviceModel, QubitProperties};

    fn device(errors: &[(f64, f64)]) -> DeviceModel {
        let mut d = DeviceModel::new("readout");
        for (q, &(a, b)) in errors.iter().enumerate() {
            d = d.with_qubit(q, QubitProperties::new().with_readout(a, b));
        }
        d
    }

    #[test]
    fn calibrations_recover_the_errors() {
        let errors = [(0.02, 0.05), (0.1, 0.03), (0.0, 0.2)];
        let backend = DeviceBackend::density_matrix(device(&errors)).unwrap();
        let cal = ReadoutCalibration::tensored(&backend, 3, 200_000, 4).unwrap();
        for (got, want) in cal.qubit_errors().iter().zip(errors) {
            assert!((got.0 - want.0).abs() < 0.003 && (got.1 - want.1).abs() < 0.003);
        }
        let full = ReadoutCalibration::full(&backend, 3, 50_000, 5).unwrap();
        for (got, want) in full.qubit_errors().iter().zip(errors) {
            assert!((got.0 - want.0).abs() < 0.005 && (got.1 - want.1).abs() < 0.005);
        }
        assert!(ReadoutCalibration::from_errors(&[(0.6, 0.5)]).is_err());
    }

    #[test]
    fn mitigation_inverts_the_confusion_matrix() {
        // A GHZ state read out through known errors: the expected
        // frequencies are A·p exactly, so mitigation must return p.
        let errors = [(0.05, 0.1), (0.02, 0.08)];
        let cal = ReadoutCalibration::from_errors(&errors).unwrap();
        let p = [0.5, 0.0, 0.0, 0.5];
        let mut f = p.to_vec();
        for (q, &(p01, p10)) in errors.iter().enumerate() {
            let bit = 1 << q;
            for b in 0..4 {
                if b & bit == 0 {
                    let (x, y) = (f[b], f[b | bit]);
                    f[b] = (1.0 - p01) * x + p10 * y;
                    f[b | bit] = p01 * x + (1.0 - p10) * y;
                }
            }
        }
        let shots = 1_000_000;
        let counts: Counts = f
            .iter()
            .enumerate()
            .map(|(b, &x)| (b, (x * shots as f64).round() as usize))
            .collect();
        let q = cal.mitigate(&counts).unwrap();
        for (b, &want) in p.iter().enumerate() {
            assert!((q.get(&b).copied().unwrap_or(0.0) - want).abs() < 1e-5);
        }
        // The same through the equivalent full matrix.
        let mut a = vec![0.0; 16];
        for x in 0..4 {
            for b in 0..4 {
                a[b + 4 * x] = (0..2)
                    .map(|qq| {
                        let (p01, p10) = errors[qq];
                        match ((x >> qq) & 1, (b >> qq) & 1) {
                            (0, 0) => 1.0 - p01,
                            (0, _) => p01,
                            (_, 0) => p10,
                            _ => 1.0 - p10,
                        }
                    })
                    .product();
            }
        }
        let full = ReadoutCalibration::from_matrix(2, a).unwrap();
        let q2 = full.mitigate(&counts).unwrap();
        for b in 0..4 {
            let x = q.get(&b).copied().unwrap_or(0.0);
            let y = q2.get(&b).copied().unwrap_or(0.0);
            assert!((x - y).abs() < 1e-12);
        }
        // Parity weights reproduce the mitigated ⟨Z0 Z1⟩ for both models.
        for c in [&cal, &full] {
            let w = c.parity_weights(0b11);
            let value: f64 = counts
                .iter()
                .map(|(&b, &n)| n as f64 / shots as f64 * w.weight(b))
                .sum();
            assert!((value - 1.0).abs() < 1e-5, "{value}");
        }
    }

    #[test]
    fn nearest_distribution() {
        let quasi: BTreeMap<usize, f64> = [(0, 0.6), (1, 0.5), (2, -0.1)].into();
        let p = nearest_probability_distribution(&quasi);
        assert!((p[&0] - 0.55).abs() < 1e-15 && (p[&1] - 0.45).abs() < 1e-15);
        assert!(!p.contains_key(&2));
        let quasi: BTreeMap<usize, f64> = [(0, 0.2), (1, 0.8)].into();
        assert_eq!(nearest_probability_distribution(&quasi), quasi);
    }
}
