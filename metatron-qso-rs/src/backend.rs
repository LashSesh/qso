//! Execution backends and shot-based estimation.
//!
//! A [`Backend`] turns a circuit into measurement counts. Two local
//! backends are provided: the exact [`StatevectorBackend`] and the
//! [`DensityMatrixBackend`] with a [`NoiseModel`]. Real hardware is reached
//! by exporting circuits with [`crate::quantum::to_qasm2`] and submitting
//! them with the vendor's own SDK; QSO deliberately ships no imitation of
//! a cloud client.
//!
//! [`estimate_expectation`] measures a [`PauliSum`] from shots the way a
//! device would: terms are grouped into qubit-wise commuting sets, each set
//! is measured once in its rotated basis, and every term is read off as a
//! parity of measured bits.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::noise::{DensityMatrix, MAX_DENSITY_QUBITS, NoiseModel};
use crate::quantum::{Circuit, Counts, MAX_QUBITS, Pauli, PauliString, PauliSum, StateVector};
use crate::rng::Rng;

/// Something that executes circuits and returns measurement counts.
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;

    /// Largest supported register.
    fn max_qubits(&self) -> usize;

    /// Measure all qubits `shots` times after running `circuit(theta)` on
    /// `|0…0⟩`.
    fn run(&self, circuit: &Circuit, theta: &[f64], shots: usize, seed: u64) -> Result<Counts>;
}

/// Exact pure-state simulation.
#[derive(Clone, Copy, Debug, Default)]
pub struct StatevectorBackend;

impl Backend for StatevectorBackend {
    fn name(&self) -> &str {
        "statevector"
    }

    fn max_qubits(&self) -> usize {
        MAX_QUBITS
    }

    fn run(&self, circuit: &Circuit, theta: &[f64], shots: usize, seed: u64) -> Result<Counts> {
        let state = StateVector::from_circuit(circuit, theta)?;
        Ok(state.sample(shots, &mut Rng::seed_from_u64(seed)))
    }
}

/// Mixed-state simulation with gate and readout noise.
#[derive(Clone, Debug, Default)]
pub struct DensityMatrixBackend {
    pub noise: NoiseModel,
}

impl DensityMatrixBackend {
    pub fn new(noise: NoiseModel) -> Self {
        Self { noise }
    }
}

impl Backend for DensityMatrixBackend {
    fn name(&self) -> &str {
        "density-matrix"
    }

    fn max_qubits(&self) -> usize {
        MAX_DENSITY_QUBITS
    }

    fn run(&self, circuit: &Circuit, theta: &[f64], shots: usize, seed: u64) -> Result<Counts> {
        let mut rho = DensityMatrix::zero(circuit.n_qubits())?;
        rho.apply_circuit(circuit, theta, Some(&self.noise))?;
        Ok(rho.sample(shots, self.noise.readout, &mut Rng::seed_from_u64(seed)))
    }
}

/// A measurement basis (Pauli per qubit) with the terms it measures.
pub type MeasurementGroup = (BTreeMap<usize, Pauli>, Vec<(f64, PauliString)>);

/// Greedily partition the non-identity terms into qubit-wise commuting
/// groups. Returns, per group, the measurement basis (Pauli per qubit) and
/// the member terms.
pub fn measurement_groups(observable: &PauliSum) -> Vec<MeasurementGroup> {
    let mut groups: Vec<MeasurementGroup> = Vec::new();
    for &(c, s) in observable.terms() {
        if s.is_identity() {
            continue;
        }
        let support: Vec<(usize, Pauli)> = (0..s.min_qubits())
            .map(|q| (q, s.get(q)))
            .filter(|&(_, p)| p != Pauli::I)
            .collect();
        let slot = groups.iter_mut().find(|(basis, _)| {
            support
                .iter()
                .all(|(q, p)| basis.get(q).is_none_or(|b| b == p))
        });
        match slot {
            Some((basis, members)) => {
                basis.extend(support.iter().copied());
                members.push((c, s));
            }
            None => groups.push((support.into_iter().collect(), vec![(c, s)])),
        }
    }
    groups
}

/// Estimate `⟨H⟩` for `circuit(theta)` from `shots` measurements per group.
pub fn estimate_expectation(
    backend: &dyn Backend,
    circuit: &Circuit,
    theta: &[f64],
    observable: &PauliSum,
    shots: usize,
    seed: u64,
) -> Result<f64> {
    if shots == 0 {
        return Err(QsoError::invalid("shots", "need at least one shot"));
    }
    if observable.min_qubits() > circuit.n_qubits() {
        return Err(QsoError::QubitOutOfRange {
            qubit: observable.min_qubits() - 1,
            n_qubits: circuit.n_qubits(),
        });
    }
    let base = Rng::seed_from_u64(seed);
    let mut total = observable.constant();
    for (k, (basis, members)) in measurement_groups(observable).into_iter().enumerate() {
        let mut rotated = circuit.clone();
        for (&q, &p) in &basis {
            match p {
                Pauli::X => {
                    rotated.h(q);
                }
                Pauli::Y => {
                    rotated.sdg(q).h(q);
                }
                _ => {}
            }
        }
        let counts = backend.run(&rotated, theta, shots, base.fork(k as u64).next_u64())?;
        for (c, s) in members {
            let mask = (s.x_mask() | s.z_mask()) as usize;
            let parity_sum: i64 = counts
                .iter()
                .map(|(&b, &n)| {
                    if (b & mask).count_ones().is_multiple_of(2) {
                        n as i64
                    } else {
                        -(n as i64)
                    }
                })
                .sum();
            total += c * parity_sum as f64 / shots as f64;
        }
    }
    Ok(total)
}

/// Serializable description of a backend run (for logging results).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunRecord {
    pub backend: String,
    pub shots: usize,
    pub seed: u64,
    pub counts: Counts,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::models;

    #[test]
    fn grouping_is_qubit_wise_commuting() {
        let h: PauliSum = "Z0 Z1 + Z0 + X1 X2 + Y0 + X2 - 0.5".parse().unwrap();
        let groups = measurement_groups(&h);
        for (basis, members) in &groups {
            for (_, s) in members {
                for q in 0..3 {
                    let p = s.get(q);
                    if p != Pauli::I {
                        assert_eq!(basis[&q], p);
                    }
                }
            }
        }
        let n_terms: usize = groups.iter().map(|(_, m)| m.len()).sum();
        assert_eq!(n_terms, 5);
        assert!(groups.len() <= 3);
    }

    #[test]
    fn shot_estimate_converges_to_exact_value() {
        let h2 = models::h2_sto3g();
        let mut c = Circuit::new(2);
        c.ry(0, 0.4).ry(1, -1.1).cx(0, 1).ry(1, 0.3);
        let exact = h2
            .hamiltonian
            .expectation(&StateVector::from_circuit(&c, &[]).unwrap())
            .unwrap();
        let shots = 200_000;
        let est =
            estimate_expectation(&StatevectorBackend, &c, &[], &h2.hamiltonian, shots, 1).unwrap();
        // Standard error ≲ Σ|c|/√shots.
        let bound = 4.0 * h2.hamiltonian.one_norm() / (shots as f64).sqrt();
        assert!(
            (est - exact).abs() < bound,
            "est {est} exact {exact} bound {bound}"
        );
        let again =
            estimate_expectation(&StatevectorBackend, &c, &[], &h2.hamiltonian, shots, 1).unwrap();
        assert_eq!(est, again);
    }

    #[test]
    fn noisy_backend_degrades_bell_correlations() {
        let mut c = Circuit::new(2);
        c.h(0).cx(0, 1);
        let zz: PauliSum = "Z0 Z1".parse().unwrap();
        let ideal = estimate_expectation(&StatevectorBackend, &c, &[], &zz, 10_000, 3).unwrap();
        assert_eq!(ideal, 1.0);
        let noisy = DensityMatrixBackend::new(NoiseModel::depolarizing(0.02, 0.1).unwrap());
        let value = estimate_expectation(&noisy, &c, &[], &zz, 50_000, 3).unwrap();
        assert!(value < 0.95 && value > 0.7, "{value}");
    }
}
