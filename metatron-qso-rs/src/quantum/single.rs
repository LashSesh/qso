//! Pure states in single precision: half the memory of [`StateVector`],
//! so one more qubit fits, with the same kernels and execution plans.
//!
//! Amplitudes are `Complex32`; gate matrices are formed in double
//! precision and rounded once per gate, and every sum (norms,
//! probabilities, expectation values, overlaps) is accumulated in double
//! precision. Rounding moves mostly the norm: over 1000 random gates on 12
//! qubits it drifts by 1.5·10⁻⁶, while the direction stays within
//! 1 − F = 7·10⁻¹³ of the double-precision state (the unit tests measure
//! both). Probabilities and samples are normalised by the actual norm.

use num_complex::{Complex32 as C32, Complex64 as C64};

use super::circuit::{Circuit, Instruction};
use super::kernels;
use super::pauli::PauliSum;
use super::scalar::Amplitude;
use super::state::{Counts, MAX_QUBITS, StateVector};
use crate::error::{QsoError, Result};
use crate::parallel::{for_each_mut, map_collect, sum_c64, sum_f64};
use crate::rng::{Rng, sample_cumulative};

/// Largest register a [`StateVector32`] may hold: one qubit more than
/// [`MAX_QUBITS`] in the same memory.
pub const MAX_QUBITS_SINGLE: usize = MAX_QUBITS + 1;

fn check_size(n_qubits: usize) -> Result<()> {
    if n_qubits > MAX_QUBITS_SINGLE {
        return Err(QsoError::TooManyQubits {
            what: "single-precision state vector",
            max: MAX_QUBITS_SINGLE,
            requested: n_qubits,
        });
    }
    Ok(())
}

/// A pure state of `n` qubits stored as 2ⁿ single-precision amplitudes,
/// with the index convention of [`StateVector`].
#[derive(Clone, Debug, PartialEq)]
pub struct StateVector32 {
    n_qubits: usize,
    amps: Vec<C32>,
}

impl StateVector32 {
    /// `|0…0⟩`.
    pub fn zero(n_qubits: usize) -> Result<Self> {
        check_size(n_qubits)?;
        let mut amps = map_collect(1usize << n_qubits, |_| C32::new(0.0, 0.0));
        amps[0] = C32::new(1.0, 0.0);
        Ok(Self { n_qubits, amps })
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
        let mut amps = map_collect(1usize << n_qubits, |_| C32::new(0.0, 0.0));
        amps[index] = C32::new(1.0, 0.0);
        Ok(Self { n_qubits, amps })
    }

    /// A double-precision state, rounded.
    pub fn from_state(state: &StateVector) -> Self {
        let a = state.amplitudes();
        Self {
            n_qubits: state.n_qubits(),
            amps: map_collect(a.len(), |i| C32::from_c64(a[i])),
        }
    }

    /// The state in double precision (exactly the stored amplitudes; the
    /// norm is kept). Needs twice the memory of `self`.
    pub fn to_state(&self) -> Result<StateVector> {
        if self.n_qubits > MAX_QUBITS {
            return Err(QsoError::TooManyQubits {
                what: "state vector",
                max: MAX_QUBITS,
                requested: self.n_qubits,
            });
        }
        let amps: Vec<C64> = map_collect(self.amps.len(), |i| self.amps[i].to_c64());
        StateVector::from_amplitudes_unnormalized(amps)
    }

    /// Run `circuit` with parameters `theta` on `|0…0⟩`.
    pub fn from_circuit(circuit: &Circuit, theta: &[f64]) -> Result<Self> {
        let mut s = Self::zero(circuit.n_qubits())?;
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

    pub fn amplitudes(&self) -> &[C32] {
        &self.amps
    }

    /// Mutable amplitudes; the caller keeps the state normalised.
    pub fn amplitudes_mut(&mut self) -> &mut [C32] {
        &mut self.amps
    }

    /// `‖ψ‖`, summed in double precision.
    pub fn norm(&self) -> f64 {
        sum_f64(self.amps.len(), |i| self.amps[i].norm_sqr_f64()).sqrt()
    }

    /// Rescale to unit norm.
    pub fn normalize(&mut self) -> Result<()> {
        let n = self.norm();
        if n < 1e-30 || !n.is_finite() {
            return Err(QsoError::ZeroNorm);
        }
        let inv = (1.0 / n) as f32;
        for_each_mut(&mut self.amps, |_, a| *a *= inv);
        Ok(())
    }

    /// `⟨self|other⟩` in double precision.
    pub fn inner(&self, other: &StateVector32) -> Result<C64> {
        self.same_register(other.n_qubits)?;
        Ok(sum_c64(self.amps.len(), |i| {
            self.amps[i].to_c64().conj() * other.amps[i].to_c64()
        }))
    }

    /// `⟨self|other⟩` against a double-precision state.
    pub fn inner_with(&self, other: &StateVector) -> Result<C64> {
        self.same_register(other.n_qubits())?;
        let b = other.amplitudes();
        Ok(sum_c64(self.amps.len(), |i| {
            self.amps[i].to_c64().conj() * b[i]
        }))
    }

    /// `|⟨self|other⟩|² / (‖self‖² ‖other‖²)` against a double-precision
    /// state: the fidelity of the directions, so that the drift of the
    /// norm under rounding does not count.
    pub fn fidelity_with(&self, other: &StateVector) -> Result<f64> {
        let overlap = self.inner_with(other)?.norm_sqr();
        Ok(overlap / (self.norm().powi(2) * other.norm().powi(2)))
    }

    fn same_register(&self, n: usize) -> Result<()> {
        if self.n_qubits != n {
            return Err(QsoError::DimensionMismatch {
                expected: self.n_qubits,
                actual: n,
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

    /// Apply every instruction of `circuit`, fused and blocked as for
    /// [`StateVector::apply_circuit`].
    pub fn apply_circuit(&mut self, circuit: &Circuit, theta: &[f64]) -> Result<()> {
        if circuit.n_qubits() > self.n_qubits {
            return Err(QsoError::DimensionMismatch {
                expected: self.n_qubits,
                actual: circuit.n_qubits(),
            });
        }
        circuit.check_parameters(theta)?;
        super::plan::run(&mut self.amps, circuit, theta);
        Ok(())
    }

    /// Multiply amplitude `b` by `exp(−i·angle·diagonal[b])`.
    pub fn apply_diagonal_evolution(&mut self, diagonal: &[f64], angle: f64) -> Result<()> {
        if diagonal.len() != self.dim() {
            return Err(QsoError::DimensionMismatch {
                expected: self.dim(),
                actual: diagonal.len(),
            });
        }
        kernels::apply_diagonal(&mut self.amps, |b| {
            C32::from_c64(C64::from_polar(1.0, -angle * diagonal[b]))
        });
        Ok(())
    }

    /// Probability of every basis state, in double precision and divided
    /// by the squared norm.
    pub fn probabilities(&self) -> Vec<f64> {
        let total = self.norm().powi(2);
        map_collect(self.amps.len(), |i| self.amps[i].norm_sqr_f64() / total)
    }

    /// Probability that qubit `q` reads 1 (divided by the squared norm).
    pub fn probability_one(&self, q: usize) -> Result<f64> {
        self.check_qubit(q)?;
        let one = sum_f64(self.amps.len(), |i| {
            if (i >> q) & 1 == 1 {
                self.amps[i].norm_sqr_f64()
            } else {
                0.0
            }
        });
        Ok(one / self.norm().powi(2))
    }

    /// `⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩`, accumulated in double precision.
    pub fn expectation(&self, h: &PauliSum) -> Result<f64> {
        if h.min_qubits() > self.n_qubits {
            return Err(QsoError::DimensionMismatch {
                expected: self.n_qubits,
                actual: h.min_qubits(),
            });
        }
        Ok(h.expectation_amplitudes(&self.amps) / self.norm().powi(2))
    }

    /// Sample `shots` computational-basis measurements of all qubits (the
    /// cumulative distribution in double precision, normalised by its
    /// total).
    pub fn sample(&self, shots: usize, rng: &mut Rng) -> Counts {
        let mut cumulative = Vec::with_capacity(self.amps.len());
        let mut acc = 0.0;
        for a in &self.amps {
            acc += a.norm_sqr_f64();
            cumulative.push(acc);
        }
        if acc > 0.0 {
            for c in &mut cumulative {
                *c /= acc;
            }
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
        let keep = usize::from(outcome);
        for_each_mut(&mut self.amps, |i, a| {
            if (i >> q) & 1 != keep {
                *a = C32::new(0.0, 0.0);
            }
        });
        self.normalize()?;
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests;
