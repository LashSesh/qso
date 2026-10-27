//! Virtual distillation: error mitigation by purification (Huggins et al.,
//! Phys. Rev. X 11, 041036 (2021); Koczor, Phys. Rev. X 11, 031057
//! (2021)).
//!
//! Noise mixes the ideal state into `ρ`; its dominant eigenvector is
//! usually much closer to it than `ρ`. With `k` copies of `ρ`,
//!
//! ```text
//! ⟨O⟩_k = Tr(ρᵏ O) / Tr(ρᵏ)
//! ```
//!
//! suppresses the other eigenvectors by the `k`-th power of their weights
//! and tends to the dominant eigenvector's value as `k` grows.
//! [`distilled_expectation`] computes it exactly from a density matrix,
//! [`dominant_expectation`] the limit `k → ∞`.
//!
//! On hardware, `Tr(ρᵏ P)` for a Pauli string `P` is the expectation of
//! `S · P⁽¹⁾` in `ρ^{⊗k}`, where `S` shifts the `k` copies cyclically and
//! `P⁽¹⁾` acts on the first. Both factor over the qubits: on the `k`
//! copies of qubit `q`, `U_q = S_q P_q⁽¹⁾` is a `k`-qubit unitary, and the
//! unitaries of different qubits commute. [`DistillationCircuits`]
//! rotates every copy so that `P` becomes a product of `Z`s, then turns
//! each `U_q` into a diagonal matrix by a `k`-qubit gate `V_q†` (its
//! eigenvectors) and measures in the computational basis: a shot that
//! ends in eigenvectors with eigenvalues `λ_q` contributes `Re Π_q λ_q`.
//! One setting per Pauli string estimates `Tr(ρᵏ P)`, one more
//! (`U_q = S_q`) estimates `Tr(ρᵏ)`; the ratio of the two means is the
//! estimate, its standard error from the delta method.

use std::f64::consts::PI;

use crate::C64;
use crate::error::{QsoError, Result};
use crate::estimation::Estimate;
use crate::linalg::{CMatrix, eigh};
use crate::noise::DensityMatrix;
use crate::quantum::{Circuit, Counts, Pauli, PauliString, PauliSum};

/// Most copies (`k`) of the register.
pub const MAX_COPIES: usize = 3;
/// Largest register for [`distilled_expectation`] and
/// [`dominant_expectation`] (dense `2ⁿ × 2ⁿ` matrices).
pub const MAX_DISTILLED_QUBITS: usize = 12;

fn check_copies(copies: usize) -> Result<()> {
    if !(1..=MAX_COPIES).contains(&copies) {
        return Err(QsoError::invalid(
            "copies",
            format!("1 to {MAX_COPIES} copies, got {copies}"),
        ));
    }
    Ok(())
}

fn dense(rho: &DensityMatrix) -> Result<CMatrix> {
    if rho.n_qubits() > MAX_DISTILLED_QUBITS {
        return Err(QsoError::invalid(
            "rho",
            format!(
                "{} qubits exceed the {MAX_DISTILLED_QUBITS} of dense distillation",
                rho.n_qubits()
            ),
        ));
    }
    Ok(rho.to_matrix())
}

fn check_observable(observable: &PauliSum, n: usize) -> Result<()> {
    let needed = observable.min_qubits();
    if needed > n {
        return Err(QsoError::QubitOutOfRange {
            qubit: needed - 1,
            n_qubits: n,
        });
    }
    Ok(())
}

/// `Tr(ρᵏ O) / Tr(ρᵏ)` for `k = copies` (1 to [`MAX_COPIES`]).
pub fn distilled_expectation(
    rho: &DensityMatrix,
    observable: &PauliSum,
    copies: usize,
) -> Result<f64> {
    check_copies(copies)?;
    check_observable(observable, rho.n_qubits())?;
    let r = dense(rho)?;
    let n = rho.n_qubits();
    if copies == 1 {
        return rho.expectation(observable);
    }
    // Tr(ρᵏ P) = Tr(ρ^{k−1} · ρ P): one matrix product at most.
    let left = if copies == 3 { &r * &r } else { r.clone() };
    let trace = product_pauli_trace(&left, &r, PauliString::identity(), n);
    let mut numerator = 0.0;
    for &(c, p) in observable.terms() {
        numerator += c * product_pauli_trace(&left, &r, p, n);
    }
    Ok(numerator / trace)
}

/// `Re Tr(A B P)` for a Pauli string `P` in `O(4ⁿ)`: `P` has one entry
/// per column, `P[a⊕x, a] = phase · (−1)^{|a∧z|}`.
fn product_pauli_trace(a: &CMatrix, b: &CMatrix, p: PauliString, n: usize) -> f64 {
    let dim = 1usize << n;
    let (x, z) = (p.x_mask() as usize, p.z_mask() as usize);
    let y = (x & z).count_ones();
    let phase = [
        C64::new(1.0, 0.0),
        C64::new(0.0, 1.0),
        C64::new(-1.0, 0.0),
        C64::new(0.0, -1.0),
    ][(y % 4) as usize];
    let sum = crate::parallel::sum_c64(dim, |col| {
        let sign = if (col & z).count_ones() % 2 == 1 {
            -1.0
        } else {
            1.0
        };
        let mut v = C64::new(0.0, 0.0);
        for row in 0..dim {
            v += a[(col, row)] * b[(row, col ^ x)];
        }
        v * sign
    });
    (sum * phase).re
}

/// `⟨v|O|v⟩` for the eigenvector `v` of `ρ` with the largest eigenvalue
/// (the limit of [`distilled_expectation`] as the copies grow). An error
/// if the largest eigenvalue is degenerate (to 10⁻¹⁰).
pub fn dominant_expectation(rho: &DensityMatrix, observable: &PauliSum) -> Result<f64> {
    check_observable(observable, rho.n_qubits())?;
    let r = dense(rho)?;
    let (values, vectors) = eigh(&r);
    let dim = values.len();
    if dim > 1 && values[dim - 1] - values[dim - 2] < 1e-10 {
        return Err(QsoError::invalid(
            "rho",
            "the largest eigenvalue is degenerate: no dominant eigenvector",
        ));
    }
    let v: Vec<C64> = (0..dim).map(|i| vectors[(i, dim - 1)]).collect();
    let state = crate::quantum::StateVector::from_amplitudes(v)?;
    observable.expectation(&state)
}

/// One measurement setting: a circuit on `copies · n` qubits and, for
/// every qubit of the register, the eigenvalue belonging to each local
/// outcome of its `copies` qubits.
#[derive(Clone, Debug)]
pub struct DistillationSetting {
    /// Preparation of every copy, basis rotations and the diagonalising
    /// gates; measure all qubits afterwards. Copy `c` of qubit `q` is
    /// qubit `c·n + q`.
    pub circuit: Circuit,
    /// The part after the preparation (rotations and diagonalising gates).
    pub measurement: Circuit,
    /// The Pauli string whose `Tr(ρᵏ P)` this setting estimates (identity:
    /// `Tr(ρᵏ)`).
    pub pauli: PauliString,
    /// `eigenvalues[q][b]`: eigenvalue of `U_q` for local outcome `b`
    /// (copy 0 the most significant bit).
    eigenvalues: Vec<Vec<C64>>,
}

/// The measurement circuits of virtual distillation for one observable,
/// and the estimator that combines their results.
#[derive(Clone, Debug)]
pub struct DistillationCircuits {
    copies: usize,
    n_qubits: usize,
    /// `(coefficient, setting index)` per Pauli string of the observable;
    /// the identity uses the denominator setting.
    terms: Vec<(f64, usize)>,
    /// Setting 0 is the denominator `Tr(ρᵏ)`.
    settings: Vec<DistillationSetting>,
}

/// Matrix of `S · Z⁽⁰⁾` (or `S` alone) on `k` qubits, copy 0 the most
/// significant bit; `S` shifts copy `c` to copy `c + 1`.
fn shift_operator(k: usize, with_z: bool) -> CMatrix {
    let dim = 1usize << k;
    CMatrix::from_fn(dim, dim, |row, col| {
        // bits of `col`: x_0 … x_{k−1} (x_0 most significant).
        let bit = |v: usize, c: usize| (v >> (k - 1 - c)) & 1;
        let mut shifted = 0;
        for c in 0..k {
            // S|x_0, …, x_{k−1}⟩ = |x_{k−1}, x_0, …, x_{k−2}⟩.
            let from = (c + k - 1) % k;
            shifted |= bit(col, from) << (k - 1 - c);
        }
        if shifted != row {
            return C64::new(0.0, 0.0);
        }
        let sign = if with_z && bit(col, 0) == 1 {
            -1.0
        } else {
            1.0
        };
        C64::new(sign, 0.0)
    })
}

/// Eigenvectors (columns) and eigenvalues of a normal matrix, from the
/// Hermitian combination `(U + U†)/2 + c (U − U†)/2i` with a generic `c`.
fn diagonalise_unitary(u: &CMatrix) -> (CMatrix, Vec<C64>) {
    // Eigenvalues are 2k-th roots of unity; any c that keeps
    // Re λ + c Im λ apart for distinct λ separates them.
    let c = 0.3;
    let i = C64::new(0.0, 1.0);
    let h =
        (u + u.adjoint()) * C64::new(0.5, 0.0) + (u - u.adjoint()) * (C64::new(c, 0.0) / (i * 2.0));
    let (_, vectors) = eigh(&h);
    let values = (0..u.nrows())
        .map(|b| {
            let v = vectors.column(b);
            let lambda = (v.adjoint() * u * v)[(0, 0)];
            // Snap to the exact root of unity e^{iπm/6} (k ≤ 3).
            let m = (lambda.arg() / (PI / 6.0)).round();
            C64::from_polar(1.0, m * PI / 6.0)
        })
        .collect();
    (vectors, values)
}

/// Append the unitary `w` on `targets` (`targets[0]` the most significant
/// local bit) as two-level rotations, each a single-qubit gate controlled
/// by the other targets (Gray-code order, so that neighbouring basis states
/// differ in one bit): gates that export to OpenQASM and run anywhere.
fn append_two_level(circuit: &mut Circuit, w: &CMatrix, targets: &[usize]) -> Result<()> {
    let k = targets.len();
    let dim = 1usize << k;
    let gray: Vec<usize> = (0..dim).map(|i| i ^ (i >> 1)).collect();
    // W in Gray order.
    let mut m = CMatrix::from_fn(dim, dim, |r, c| w[(gray[r], gray[c])]);
    // G_m … G_1 W = diag(1, …, 1, e^{iφ}); record the G.
    let mut ops: Vec<(usize, usize, [C64; 4])> = Vec::new();
    for col in 0..dim - 1 {
        for row in (col + 1..dim).rev() {
            let (a, b) = (m[(row - 1, col)], m[(row, col)]);
            if b.norm() < 1e-15 && row > col + 1 {
                continue;
            }
            let norm = (a.norm_sqr() + b.norm_sqr()).sqrt();
            // [[a*, b*], [−b, a]] / norm sends (a, b) to (norm, 0).
            let g = [a.conj() / norm, b.conj() / norm, -b / norm, a / norm];
            for c2 in 0..dim {
                let (x, y) = (m[(row - 1, c2)], m[(row, c2)]);
                m[(row - 1, c2)] = g[0] * x + g[1] * y;
                m[(row, c2)] = g[2] * x + g[3] * y;
            }
            ops.push((row - 1, row, g));
        }
    }
    let phase = m[(dim - 1, dim - 1)];
    // W = G_1† … G_m† · diag(…, e^{iφ}): in time, the phase first.
    let mut apply = |s: usize, t: usize, g: [C64; 4]| -> Result<()> {
        // States s and t differ in one local bit; the others are controls.
        let diff = gray[s] ^ gray[t];
        let bit = diff.trailing_zeros() as usize;
        let target = targets[k - 1 - bit];
        let mut controls = Vec::new();
        let mut flips = Vec::new();
        for (c, &q) in targets.iter().enumerate() {
            let local = k - 1 - c;
            if local == bit {
                continue;
            }
            controls.push(q);
            if (gray[s] >> local) & 1 == 0 {
                flips.push(q);
            }
        }
        // g acts on (state s, state t); order it as (bit 0, bit 1).
        let matrix = if (gray[s] >> bit) & 1 == 0 {
            g.to_vec()
        } else {
            vec![g[3], g[2], g[1], g[0]]
        };
        for &q in &flips {
            circuit.x(q);
        }
        circuit.unitary(matrix, &[target], &controls)?;
        for &q in &flips {
            circuit.x(q);
        }
        Ok(())
    };
    if (phase - C64::new(1.0, 0.0)).norm() > 1e-15 {
        let last = dim - 1;
        apply(
            last - 1,
            last,
            [
                C64::new(1.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(0.0, 0.0),
                phase,
            ],
        )?;
    }
    for &(s, t, g) in ops.iter().rev() {
        // G† = [[g0*, g2*], [g1*, g3*]].
        apply(s, t, [g[0].conj(), g[2].conj(), g[1].conj(), g[3].conj()])?;
    }
    Ok(())
}

impl DistillationCircuits {
    /// The settings for `observable` on the state that `preparation`
    /// prepares from `|0…0⟩`, with `copies` copies (2 or 3; 1 gives the
    /// plain expectation value).
    pub fn new(preparation: &Circuit, observable: &PauliSum, copies: usize) -> Result<Self> {
        check_copies(copies)?;
        let n = preparation.n_qubits();
        if n == 0 {
            return Err(QsoError::invalid("preparation", "the register is empty"));
        }
        check_observable(observable, n)?;
        if copies * n > crate::quantum::MAX_QUBITS {
            return Err(QsoError::invalid(
                "copies",
                format!(
                    "{copies} copies of {n} qubits exceed {} qubits",
                    crate::quantum::MAX_QUBITS
                ),
            ));
        }
        let (plain_vectors, plain_values) = diagonalise_unitary(&shift_operator(copies, false));
        let (z_vectors, z_values) = diagonalise_unitary(&shift_operator(copies, true));
        let build = |p: PauliString| -> Result<DistillationSetting> {
            let mut measurement = Circuit::new(copies * n);
            let mut eigenvalues = Vec::with_capacity(n);
            for q in 0..n {
                let factor = p.get(q);
                for c in 0..copies {
                    match factor {
                        Pauli::X => {
                            measurement.h(c * n + q);
                        }
                        Pauli::Y => {
                            measurement.sdg(c * n + q);
                            measurement.h(c * n + q);
                        }
                        _ => {}
                    }
                }
                let targets: Vec<usize> = (0..copies).map(|c| c * n + q).collect();
                let (v, values) = if factor == Pauli::I {
                    (&plain_vectors, &plain_values)
                } else {
                    (&z_vectors, &z_values)
                };
                if copies > 1 {
                    append_two_level(&mut measurement, &v.adjoint(), &targets)?;
                    eigenvalues.push(values.clone());
                } else {
                    // One copy: measure P itself (Z after the rotation).
                    let z = factor != Pauli::I;
                    eigenvalues.push(vec![
                        C64::new(1.0, 0.0),
                        C64::new(if z { -1.0 } else { 1.0 }, 0.0),
                    ]);
                }
            }
            let mut circuit = Circuit::new(copies * n);
            for c in 0..copies {
                let mapping: Vec<usize> = (0..n).map(|q| c * n + q).collect();
                circuit.append_mapped(preparation, &mapping)?;
            }
            circuit.append(&measurement)?;
            Ok(DistillationSetting {
                circuit,
                measurement,
                pauli: p,
                eigenvalues,
            })
        };
        let mut settings = vec![build(PauliString::identity())?];
        let mut terms = Vec::new();
        for &(c, p) in observable.terms() {
            if p.is_identity() {
                terms.push((c, 0));
                continue;
            }
            let index = match settings.iter().position(|s| s.pauli == p) {
                Some(i) => i,
                None => {
                    settings.push(build(p)?);
                    settings.len() - 1
                }
            };
            terms.push((c, index));
        }
        Ok(Self {
            copies,
            n_qubits: n,
            terms,
            settings,
        })
    }

    pub fn copies(&self) -> usize {
        self.copies
    }

    /// The settings; setting 0 estimates `Tr(ρᵏ)`.
    pub fn settings(&self) -> &[DistillationSetting] {
        &self.settings
    }

    /// `Re Π_q λ_q` for a measured outcome of a setting.
    fn value(&self, setting: &DistillationSetting, outcome: usize) -> f64 {
        let (n, k) = (self.n_qubits, self.copies);
        let mut product = C64::new(1.0, 0.0);
        for q in 0..n {
            let mut local = 0;
            for c in 0..k {
                local = (local << 1) | ((outcome >> (c * n + q)) & 1);
            }
            product *= setting.eigenvalues[q][local];
        }
        product.re
    }

    /// The estimate from the exact outcome probabilities of every setting
    /// (indexed by the measured bit string, qubit 0 least significant).
    pub fn evaluate(&self, probabilities: &[Vec<f64>]) -> Result<f64> {
        self.check_results(probabilities.len())?;
        let outcomes = 1usize << (self.copies * self.n_qubits);
        for p in probabilities {
            if p.len() != outcomes {
                return Err(QsoError::DimensionMismatch {
                    expected: outcomes,
                    actual: p.len(),
                });
            }
            if p.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(QsoError::invalid(
                    "probabilities",
                    "probabilities must be finite and non-negative",
                ));
            }
        }
        let means: Vec<f64> = self
            .settings
            .iter()
            .zip(probabilities)
            .map(|(s, p)| {
                p.iter()
                    .enumerate()
                    .map(|(b, &w)| w * self.value(s, b))
                    .sum()
            })
            .collect();
        let numerator: f64 = self.terms.iter().map(|&(c, i)| c * means[i]).sum();
        if means[0].abs() < 1e-12 {
            return Err(QsoError::invalid(
                "probabilities",
                "the estimate of Tr(ρᵏ) vanishes",
            ));
        }
        Ok(numerator / means[0])
    }

    /// The estimate and its standard error from measured counts of every
    /// setting (delta method on the ratio of means).
    pub fn estimate(&self, counts: &[Counts]) -> Result<Estimate> {
        self.check_results(counts.len())?;
        let mut means = Vec::with_capacity(counts.len());
        let mut variances = Vec::with_capacity(counts.len());
        let mut shots = 0;
        let outcomes = 1usize << (self.copies * self.n_qubits);
        for (index, (s, cnt)) in self.settings.iter().zip(counts).enumerate() {
            if index == 0 && self.copies == 1 {
                // One copy: Tr(ρ) = 1, nothing to measure.
                means.push(1.0);
                variances.push(0.0);
                continue;
            }
            if let Some((&b, _)) = cnt.iter().find(|&(&b, _)| b >= outcomes) {
                return Err(QsoError::invalid(
                    "counts",
                    format!(
                        "outcome {b} has more than {} bits",
                        self.copies * self.n_qubits
                    ),
                ));
            }
            let total: usize = cnt.values().sum();
            if total < 2 {
                return Err(QsoError::invalid(
                    "counts",
                    "every setting needs at least 2 shots",
                ));
            }
            shots += total;
            let mean = cnt
                .iter()
                .map(|(&b, &n)| self.value(s, b) * n as f64)
                .sum::<f64>()
                / total as f64;
            let ss: f64 = cnt
                .iter()
                .map(|(&b, &n)| (self.value(s, b) - mean).powi(2) * n as f64)
                .sum();
            means.push(mean);
            variances.push(ss / (total - 1) as f64 / total as f64);
        }
        // Numerator N = Σ c_i m_i over the Pauli settings (independent of
        // the denominator setting except through identity terms).
        let mut weights = vec![0.0; self.settings.len()];
        for &(c, i) in &self.terms {
            weights[i] += c;
        }
        let numerator: f64 = weights.iter().zip(&means).map(|(w, m)| w * m).sum();
        let d = means[0];
        if d.abs() < 1e-12 {
            return Err(QsoError::invalid(
                "counts",
                "the estimate of Tr(ρᵏ) vanishes",
            ));
        }
        let value = numerator / d;
        // ∂(N/D)/∂m_i = w_i/D (+ for i = 0: −N/D²).
        let mut var = 0.0;
        for (i, &v) in variances.iter().enumerate() {
            let mut grad = weights[i] / d;
            if i == 0 {
                grad -= numerator / (d * d);
            }
            var += grad * grad * v;
        }
        Ok(Estimate {
            value,
            std_error: var.sqrt(),
            samples: shots,
        })
    }

    /// The circuit estimator on `copies` copies of `rho` (exact outcome
    /// probabilities; the preparation is replaced by `rho`).
    pub fn evaluate_on(&self, rho: &DensityMatrix) -> Result<f64> {
        let (n, k) = (self.n_qubits, self.copies);
        if rho.n_qubits() != n {
            return Err(QsoError::DimensionMismatch {
                expected: n,
                actual: rho.n_qubits(),
            });
        }
        let mut probabilities = Vec::with_capacity(self.settings.len());
        for s in &self.settings {
            // Rebuilt per setting rather than cloned: one 4^{kn} copy alive.
            let mut state = tensor_power(rho, k)?;
            state.apply_circuit(&s.measurement, &[], None)?;
            probabilities.push(state.probabilities());
        }
        self.evaluate(&probabilities)
    }

    fn check_results(&self, len: usize) -> Result<()> {
        if len != self.settings.len() {
            return Err(QsoError::DimensionMismatch {
                expected: self.settings.len(),
                actual: len,
            });
        }
        Ok(())
    }
}

/// `ρ^{⊗k}`, copy `c` on qubits `c·n … c·n + n − 1`.
fn tensor_power(rho: &DensityMatrix, k: usize) -> Result<DensityMatrix> {
    let n = rho.n_qubits();
    let total = n * k;
    if total > crate::noise::MAX_DENSITY_QUBITS {
        return Err(QsoError::invalid(
            "rho",
            format!(
                "{k} copies of {n} qubits exceed the {} qubits of a density matrix",
                crate::noise::MAX_DENSITY_QUBITS
            ),
        ));
    }
    let dim = 1usize << total;
    let mask = (1usize << n) - 1;
    let mut data = vec![C64::new(0.0, 0.0); dim * dim];
    for col in 0..dim {
        for row in 0..dim {
            let mut v = C64::new(1.0, 0.0);
            for c in 0..k {
                v *= rho.get((row >> (c * n)) & mask, (col >> (c * n)) & mask);
                if v == C64::new(0.0, 0.0) {
                    break;
                }
            }
            data[row + col * dim] = v;
        }
    }
    DensityMatrix::from_data(total, data)
}

#[cfg(test)]
mod tests;
