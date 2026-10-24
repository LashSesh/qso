//! n-qubit pure states.

use std::collections::BTreeMap;

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::circuit::{Circuit, Instruction};
use super::kernels;
use crate::error::{QsoError, Result};
use crate::linalg::{CMatrix, eigh, inner};
use crate::parallel::{for_each_mut, sum_f64};
use crate::rng::{Rng, sample_cumulative};

/// Largest register a [`StateVector`] may hold (2³⁴ amplitudes = 256 GiB).
pub const MAX_QUBITS: usize = 34;

/// Measurement histogram: basis index → number of shots.
pub type Counts = BTreeMap<usize, usize>;

/// Format basis index `b` as a bit string with qubit `n−1` on the left
/// (the convention used by Qiskit and most papers).
pub fn bitstring(b: usize, n_qubits: usize) -> String {
    (0..n_qubits)
        .rev()
        .map(|q| if (b >> q) & 1 == 1 { '1' } else { '0' })
        .collect()
}

/// A pure state of `n` qubits stored as 2ⁿ complex amplitudes.
///
/// Amplitude `k` belongs to basis state `|b_{n−1} … b_1 b_0⟩` with
/// `k = Σ_q b_q·2^q` (little-endian: qubit 0 is the least significant bit).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "StateVectorData")]
pub struct StateVector {
    n_qubits: usize,
    amps: Vec<C64>,
}

/// Unvalidated serialised form of a [`StateVector`].
#[derive(Deserialize)]
struct StateVectorData {
    n_qubits: usize,
    amps: Vec<C64>,
}

impl TryFrom<StateVectorData> for StateVector {
    type Error = QsoError;

    /// Checks the register size and that there are exactly 2ⁿ finite
    /// amplitudes (the norm is kept as stored).
    fn try_from(data: StateVectorData) -> Result<Self> {
        check_size(data.n_qubits)?;
        if data.amps.len() != 1 << data.n_qubits {
            return Err(QsoError::DimensionMismatch {
                expected: 1 << data.n_qubits,
                actual: data.amps.len(),
            });
        }
        if data
            .amps
            .iter()
            .any(|a| !a.re.is_finite() || !a.im.is_finite())
        {
            return Err(QsoError::invalid("amps", "amplitudes must be finite"));
        }
        Ok(Self {
            n_qubits: data.n_qubits,
            amps: data.amps,
        })
    }
}

fn check_size(n_qubits: usize) -> Result<()> {
    if n_qubits > MAX_QUBITS {
        return Err(QsoError::TooManyQubits {
            what: "StateVector",
            max: MAX_QUBITS,
            requested: n_qubits,
        });
    }
    Ok(())
}

impl StateVector {
    /// `|0…0⟩`.
    ///
    /// # Panics
    /// If `n_qubits > MAX_QUBITS`.
    pub fn zero(n_qubits: usize) -> Self {
        check_size(n_qubits).expect("register too large");
        let mut amps = vec![C64::new(0.0, 0.0); 1 << n_qubits];
        amps[0] = C64::new(1.0, 0.0);
        Self { n_qubits, amps }
    }

    /// Computational basis state `|index⟩`.
    pub fn basis(n_qubits: usize, index: usize) -> Result<Self> {
        check_size(n_qubits)?;
        if index >= 1 << n_qubits {
            return Err(QsoError::DimensionMismatch {
                expected: 1 << n_qubits,
                actual: index + 1,
            });
        }
        let mut amps = vec![C64::new(0.0, 0.0); 1 << n_qubits];
        amps[index] = C64::new(1.0, 0.0);
        Ok(Self { n_qubits, amps })
    }

    /// Uniform superposition `|+⟩^{⊗n}`.
    pub fn plus(n_qubits: usize) -> Self {
        check_size(n_qubits).expect("register too large");
        let dim = 1usize << n_qubits;
        let a = C64::new(1.0 / (dim as f64).sqrt(), 0.0);
        Self {
            n_qubits,
            amps: vec![a; dim],
        }
    }

    /// State from amplitudes (length must be a power of two). The vector is
    /// normalised; a zero vector is rejected.
    pub fn from_amplitudes(amps: Vec<C64>) -> Result<Self> {
        let dim = amps.len();
        if dim == 0 || !dim.is_power_of_two() {
            return Err(QsoError::NotPowerOfTwo(dim));
        }
        let n_qubits = dim.trailing_zeros() as usize;
        check_size(n_qubits)?;
        let mut s = Self { n_qubits, amps };
        s.normalize()?;
        Ok(s)
    }

    /// Haar-random pure state.
    pub fn random(n_qubits: usize, rng: &mut Rng) -> Self {
        check_size(n_qubits).expect("register too large");
        let amps: Vec<C64> = (0..1usize << n_qubits)
            .map(|_| C64::new(rng.normal(), rng.normal()))
            .collect();
        Self::from_amplitudes(amps).expect("Gaussian vector has non-zero norm")
    }

    /// Run `circuit` with parameters `theta` on `|0…0⟩`.
    pub fn from_circuit(circuit: &Circuit, theta: &[f64]) -> Result<Self> {
        check_size(circuit.n_qubits())?;
        let mut s = Self::zero(circuit.n_qubits());
        s.apply_circuit(circuit, theta)?;
        Ok(s)
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Hilbert-space dimension 2ⁿ.
    pub fn dim(&self) -> usize {
        self.amps.len()
    }

    pub fn amplitudes(&self) -> &[C64] {
        &self.amps
    }

    /// Mutable amplitudes. The caller is responsible for keeping the state
    /// normalised (see [`StateVector::normalize`]).
    pub fn amplitudes_mut(&mut self) -> &mut [C64] {
        &mut self.amps
    }

    pub fn into_amplitudes(self) -> Vec<C64> {
        self.amps
    }

    /// `‖ψ‖`.
    pub fn norm(&self) -> f64 {
        sum_f64(self.amps.len(), |i| self.amps[i].norm_sqr()).sqrt()
    }

    /// Rescale to unit norm.
    pub fn normalize(&mut self) -> Result<()> {
        let n = self.norm();
        if n < 1e-300 || !n.is_finite() {
            return Err(QsoError::ZeroNorm);
        }
        let inv = 1.0 / n;
        for_each_mut(&mut self.amps, |_, a| *a *= inv);
        Ok(())
    }

    /// `⟨self|other⟩`.
    pub fn inner(&self, other: &StateVector) -> Result<C64> {
        self.same_register(other)?;
        Ok(inner(&self.amps, &other.amps))
    }

    /// `|⟨self|other⟩|²`.
    pub fn fidelity(&self, other: &StateVector) -> Result<f64> {
        Ok(self.inner(other)?.norm_sqr())
    }

    fn same_register(&self, other: &StateVector) -> Result<()> {
        if self.n_qubits != other.n_qubits {
            return Err(QsoError::DimensionMismatch {
                expected: self.n_qubits,
                actual: other.n_qubits,
            });
        }
        Ok(())
    }

    fn check_qubit(&self, q: usize) -> Result<()> {
        if q >= self.n_qubits {
            return Err(QsoError::QubitOutOfRange {
                qubit: q,
                n_qubits: self.n_qubits,
            });
        }
        Ok(())
    }

    /// Apply a single instruction with parameters `theta`.
    pub fn apply_instruction(&mut self, inst: &Instruction, theta: &[f64]) -> Result<()> {
        inst.validate(self.n_qubits)?;
        for p in &inst.params {
            if let crate::quantum::Param::Symbol(s) = p
                && s.index >= theta.len()
            {
                return Err(QsoError::ParameterCount {
                    expected: s.index + 1,
                    actual: theta.len(),
                });
            }
        }
        if theta.iter().any(|t| !t.is_finite()) {
            return Err(QsoError::invalid("theta", "parameters must be finite"));
        }
        let angles = inst.angles(theta);
        kernels::apply_instruction(&mut self.amps, inst, &angles, false, 0);
        Ok(())
    }

    /// Apply every instruction of `circuit` with parameters `theta`.
    pub fn apply_circuit(&mut self, circuit: &Circuit, theta: &[f64]) -> Result<()> {
        if circuit.n_qubits() > self.n_qubits {
            return Err(QsoError::DimensionMismatch {
                expected: self.n_qubits,
                actual: circuit.n_qubits(),
            });
        }
        circuit.check_parameters(theta)?;
        for inst in circuit.instructions() {
            let angles = inst.angles(theta);
            kernels::apply_instruction(&mut self.amps, inst, &angles, false, 0);
        }
        Ok(())
    }

    /// Apply an arbitrary matrix (row-major, dimension `2^targets.len()`,
    /// `targets[0]` most significant) with optional controls.
    ///
    /// The matrix is not checked for unitarity; non-unitary matrices leave
    /// the state unnormalised.
    pub fn apply_matrix(
        &mut self,
        matrix: &[C64],
        targets: &[usize],
        controls: &[usize],
    ) -> Result<()> {
        let k = targets.len();
        if k == 0 || k > kernels::MAX_DENSE_TARGETS {
            return Err(QsoError::invalid(
                "targets",
                format!(
                    "between 1 and {} targets required",
                    kernels::MAX_DENSE_TARGETS
                ),
            ));
        }
        if matrix.len() != (1 << k) * (1 << k) {
            return Err(QsoError::DimensionMismatch {
                expected: (1 << k) * (1 << k),
                actual: matrix.len(),
            });
        }
        let mut seen = Vec::new();
        for &q in targets.iter().chain(controls) {
            self.check_qubit(q)?;
            if seen.contains(&q) {
                return Err(QsoError::DuplicateQubit {
                    gate: "matrix",
                    qubit: q,
                });
            }
            seen.push(q);
        }
        let mask = controls.iter().fold(0usize, |m, &c| m | (1 << c));
        match k {
            1 => kernels::apply_1q(
                &mut self.amps,
                targets[0],
                mask,
                &matrix.try_into().expect("2x2"),
            ),
            2 => kernels::apply_2q(
                &mut self.amps,
                targets[0],
                targets[1],
                mask,
                &matrix.try_into().expect("4x4"),
            ),
            _ => kernels::apply_kq(&mut self.amps, targets, mask, matrix),
        }
        Ok(())
    }

    /// Multiply amplitude `b` by `exp(−i·angle·diagonal[b])`, i.e. apply
    /// `exp(−i·angle·D)` for a diagonal operator `D` (used by QAOA).
    pub fn apply_diagonal_evolution(&mut self, diagonal: &[f64], angle: f64) -> Result<()> {
        if diagonal.len() != self.dim() {
            return Err(QsoError::DimensionMismatch {
                expected: self.dim(),
                actual: diagonal.len(),
            });
        }
        kernels::apply_diagonal(&mut self.amps, |b| {
            C64::from_polar(1.0, -angle * diagonal[b])
        });
        Ok(())
    }

    /// Probability of every basis state.
    pub fn probabilities(&self) -> Vec<f64> {
        crate::parallel::map_collect(self.amps.len(), |i| self.amps[i].norm_sqr())
    }

    /// Probability that qubit `q` reads 1.
    pub fn probability_one(&self, q: usize) -> Result<f64> {
        self.check_qubit(q)?;
        Ok(sum_f64(self.amps.len(), |i| {
            if (i >> q) & 1 == 1 {
                self.amps[i].norm_sqr()
            } else {
                0.0
            }
        }))
    }

    /// Sample `shots` computational-basis measurements of all qubits.
    pub fn sample(&self, shots: usize, rng: &mut Rng) -> Counts {
        let mut cumulative = Vec::with_capacity(self.amps.len());
        let mut acc = 0.0;
        for a in &self.amps {
            acc += a.norm_sqr();
            cumulative.push(acc);
        }
        let mut counts = Counts::new();
        for _ in 0..shots {
            *counts
                .entry(sample_cumulative(&cumulative, rng))
                .or_insert(0) += 1;
        }
        counts
    }

    /// Projectively measure qubit `q`, collapse the state and return the
    /// outcome.
    pub fn measure(&mut self, q: usize, rng: &mut Rng) -> Result<bool> {
        let p1 = self.probability_one(q)?;
        let outcome = rng.uniform() < p1;
        let keep = if outcome { 1 } else { 0 };
        for_each_mut(&mut self.amps, |i, a| {
            if (i >> q) & 1 != keep {
                *a = C64::new(0.0, 0.0);
            }
        });
        self.normalize()?;
        Ok(outcome)
    }

    /// Measure every qubit, collapse to the outcome and return its index.
    pub fn measure_all(&mut self, rng: &mut Rng) -> usize {
        let outcome = *self.sample(1, rng).keys().next().expect("one sample drawn");
        self.amps.iter_mut().for_each(|a| *a = C64::new(0.0, 0.0));
        self.amps[outcome] = C64::new(1.0, 0.0);
        outcome
    }

    /// Reduced density matrix of `qubits` (traced over all others).
    ///
    /// Row/column index `k` of the result encodes `qubits[j]` in bit `j`.
    pub fn reduced_density_matrix(&self, qubits: &[usize]) -> Result<CMatrix> {
        let mut seen = Vec::new();
        for &q in qubits {
            self.check_qubit(q)?;
            if seen.contains(&q) {
                return Err(QsoError::DuplicateQubit {
                    gate: "subsystem",
                    qubit: q,
                });
            }
            seen.push(q);
        }
        let k = qubits.len();
        if k > 14 {
            return Err(QsoError::TooManyQubits {
                what: "reduced density matrix",
                max: 14,
                requested: k,
            });
        }
        let sub_mask = qubits.iter().fold(0usize, |m, &q| m | (1 << q));
        let env: Vec<usize> = (0..self.n_qubits)
            .filter(|q| sub_mask & (1 << q) == 0)
            .collect();
        let d_sub = 1usize << k;
        let d_env = 1usize << env.len();
        let index = |s: usize, e: usize| {
            let mut i = 0usize;
            for (j, &q) in qubits.iter().enumerate() {
                i |= ((s >> j) & 1) << q;
            }
            for (j, &q) in env.iter().enumerate() {
                i |= ((e >> j) & 1) << q;
            }
            i
        };
        let mut rho = CMatrix::zeros(d_sub, d_sub);
        for e in 0..d_env {
            let column: Vec<C64> = (0..d_sub).map(|s| self.amps[index(s, e)]).collect();
            for r in 0..d_sub {
                for c in 0..d_sub {
                    rho[(r, c)] += column[r] * column[c].conj();
                }
            }
        }
        Ok(rho)
    }

    /// Von Neumann entanglement entropy (in bits) between `qubits` and the
    /// rest of the register.
    pub fn entanglement_entropy(&self, qubits: &[usize]) -> Result<f64> {
        let rho = self.reduced_density_matrix(qubits)?;
        let (values, _) = eigh(&rho);
        Ok(values
            .into_iter()
            .filter(|&p| p > 1e-15)
            .map(|p| -p * p.log2())
            .sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::GateKind;
    use std::f64::consts::PI;

    fn close(a: &[C64], b: &[C64], tol: f64) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).norm() < tol)
    }

    /// Dense reference: build the full 2ⁿ×2ⁿ matrix of an instruction from
    /// its action on basis states, using a naive per-index implementation
    /// independent of the optimised kernels.
    fn reference_apply(n: usize, inst: &Instruction, theta: &[f64], psi: &[C64]) -> Vec<C64> {
        use crate::quantum::gates::{matrix_1q, matrix_2q};
        let angles = inst.angles(theta);
        let m: Vec<C64> = match inst.kind {
            GateKind::Unitary => inst.matrix.as_ref().unwrap().to_vec(),
            k if k.num_targets() == Some(1) => matrix_1q(k, &angles).to_vec(),
            k => matrix_2q(k, &angles).to_vec(),
        };
        let k = inst.targets.len();
        let d = 1usize << k;
        let cmask = inst.control_mask();
        let mut out = vec![C64::new(0.0, 0.0); 1 << n];
        for (b, &amp) in psi.iter().enumerate() {
            if b & cmask != cmask {
                out[b] += amp;
                continue;
            }
            // local column index of b
            let col = (0..k).fold(0, |acc, j| (acc << 1) | ((b >> inst.targets[j]) & 1));
            for row in 0..d {
                let mut target = b;
                for j in 0..k {
                    let bit = (row >> (k - 1 - j)) & 1;
                    target = (target & !(1 << inst.targets[j])) | (bit << inst.targets[j]);
                }
                out[target] += m[row * d + col] * amp;
            }
        }
        out
    }

    #[test]
    fn every_gate_matches_dense_reference() {
        let n = 4;
        let mut rng = Rng::seed_from_u64(3);
        let mut c = Circuit::new(n);
        let th = c.new_parameter();
        c.h(2).x(0).y(3).z(1).s(0).sdg(2).t(3).tdg(1).sx(0).sxdg(3);
        c.rx(1, th)
            .ry(2, 0.4)
            .rz(0, th * 2.0)
            .p(3, -0.8)
            .u(1, 0.3, th, -1.1);
        c.cx(3, 0).cy(0, 2).cz(1, 3).ch(2, 1).swap(0, 3);
        c.crx(0, 1, 0.7).cry(3, 2, th).crz(1, 0, 1.9).cp(2, 0, -0.6);
        c.rxx(0, 2, 0.5).ryy(3, 1, th).rzz(1, 2, 2.2);
        c.ccx(0, 1, 3)
            .cswap(2, 0, 1)
            .mcx(&[0, 1, 2], 3)
            .mcz(&[0, 2, 3]);
        let a = CMatrix::from_fn(8, 8, |_, _| C64::new(rng.normal(), rng.normal()));
        let u = crate::linalg::expm_hermitian(&((&a + a.adjoint()) * C64::new(0.5, 0.0)), 0.37);
        let mut u_rowmajor = Vec::with_capacity(64);
        for r in 0..8 {
            for cc in 0..8 {
                u_rowmajor.push(u[(r, cc)]);
            }
        }
        c.unitary(u_rowmajor, &[3, 0, 2], &[1]).unwrap();
        let theta = [0.917];
        let mut psi = StateVector::random(n, &mut rng);
        for inst in c.instructions() {
            let expected = reference_apply(n, inst, &theta, psi.amplitudes());
            psi.apply_instruction(inst, &theta).unwrap();
            assert!(close(psi.amplitudes(), &expected, 1e-12), "{:?}", inst.kind);
            assert!((psi.norm() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn inverse_circuit_restores_state() {
        let mut c = Circuit::new(3);
        let p = c.new_parameters(2);
        c.h(0)
            .cx(0, 1)
            .u(2, p[0], 0.2, p[1])
            .rzz(0, 2, p[1])
            .crx(1, 2, 0.3)
            .s(1)
            .t(0)
            .sx(2);
        let mut rng = Rng::seed_from_u64(5);
        let psi0 = StateVector::random(3, &mut rng);
        let mut psi = psi0.clone();
        psi.apply_circuit(&c, &[0.4, -1.3]).unwrap();
        psi.apply_circuit(&c.inverse(), &[0.4, -1.3]).unwrap();
        assert!((psi.fidelity(&psi0).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ghz_state_statistics_and_entropy() {
        let n = 5;
        let mut c = Circuit::new(n);
        c.h(0);
        for q in 1..n {
            c.cx(q - 1, q);
        }
        let psi = StateVector::from_circuit(&c, &[]).unwrap();
        let p = psi.probabilities();
        assert!((p[0] - 0.5).abs() < 1e-12 && (p[(1 << n) - 1] - 0.5).abs() < 1e-12);
        assert!((psi.entanglement_entropy(&[0]).unwrap() - 1.0).abs() < 1e-12);
        assert!((psi.entanglement_entropy(&[0, 1, 2]).unwrap() - 1.0).abs() < 1e-12);
        let mut rng = Rng::seed_from_u64(1);
        let counts = psi.sample(10_000, &mut rng);
        assert_eq!(counts.len(), 2);
        let ones = counts[&((1 << n) - 1)] as f64 / 10_000.0;
        assert!((ones - 0.5).abs() < 0.02);
        assert_eq!(bitstring(0b10110, 5), "10110");
    }

    #[test]
    fn measurement_collapses() {
        let mut c = Circuit::new(2);
        c.h(0).cx(0, 1);
        let mut rng = Rng::seed_from_u64(2);
        for _ in 0..20 {
            let mut psi = StateVector::from_circuit(&c, &[]).unwrap();
            let a = psi.measure(0, &mut rng).unwrap();
            let b = psi.measure(1, &mut rng).unwrap();
            assert_eq!(a, b);
            assert!((psi.norm() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn product_state_has_zero_entropy_and_rotations_compose() {
        let mut c = Circuit::new(2);
        c.ry(0, PI / 3.0).ry(0, PI / 6.0).rx(1, PI);
        let psi = StateVector::from_circuit(&c, &[]).unwrap();
        assert!(psi.entanglement_entropy(&[0]).unwrap().abs() < 1e-10);
        // RY(π/2)|0⟩ on q0, RX(π)|0⟩ = −i|1⟩ on q1.
        assert!((psi.probability_one(0).unwrap() - 0.5).abs() < 1e-12);
        assert!((psi.probability_one(1).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn large_register_uses_parallel_kernels_consistently() {
        // 17 qubits crosses the parallel threshold; compare against a
        // reference built gate by gate with the naive implementation.
        let n = 17;
        let mut rng = Rng::seed_from_u64(8);
        let mut c = Circuit::new(n);
        for q in 0..n {
            c.h(q).rz(q, 0.1 * q as f64);
        }
        for q in 0..n - 1 {
            c.cx(q, q + 1);
        }
        c.rxx(0, n - 1, 0.3).ccx(1, 7, 14);
        let mut psi = StateVector::random(n, &mut rng);
        for inst in c.instructions() {
            let expected = reference_apply(n, inst, &[], psi.amplitudes());
            psi.apply_instruction(inst, &[]).unwrap();
            assert!(close(psi.amplitudes(), &expected, 1e-12), "{:?}", inst.kind);
        }
    }
}
