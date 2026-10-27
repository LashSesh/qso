//! Differential testing: one circuit on every simulator that accepts it.
//!
//! [`compare`] runs a circuit on the double-precision state vector (the
//! reference) and on
//!
//! * the single-precision state vector ([`StateVector32`]),
//! * the density-matrix simulator without noise,
//! * one quantum trajectory of a noiseless device,
//! * a matrix-product state whose bond dimension is not limited,
//! * the stabilizer tableau, when every gate is Clifford,
//! * the state vector of the circuit read back from its OpenQASM 2 and
//!   OpenQASM 3 export, when the circuit has one (dense custom unitaries
//!   do not),
//!
//! and reports how far each result lies from the reference. The property
//! tests and the `differential` fuzz target check that every deviation
//! stays below [`TOLERANCE`] ([`SINGLE_TOLERANCE`] for single precision).

use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::mps::{Mps, MpsOptions};
use crate::noise::{DensityMatrix, DeviceModel, TrajectorySimulator};
use crate::quantum::{
    Circuit, PauliSum, Program, StateVector, StateVector32, from_qasm2, to_qasm2, to_qasm3,
};
use crate::rng::Rng;
use crate::stabilizer::StabilizerState;

/// Largest register [`compare`] takes (the density matrix has 4ⁿ entries).
pub const MAX_DIFFERENTIAL_QUBITS: usize = 10;
/// Largest deviation between double-precision simulators.
pub const TOLERANCE: f64 = 1e-9;
/// Largest deviation of single precision (directions, per gate).
pub const SINGLE_TOLERANCE: f64 = 1e-10;

/// How far each simulator's result lies from the double-precision state
/// vector: `1 − F` for states (fidelity of the directions for single
/// precision, of `ρ` with the reference for the density matrix), the
/// largest difference of Pauli expectation values for the tableau.
/// `None` where a simulator does not take the circuit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DifferentialReport {
    pub n_qubits: usize,
    pub gates: usize,
    pub single_precision: f64,
    pub density_matrix: f64,
    /// `|1 − Tr ρ²|` of the noiseless density matrix.
    pub density_matrix_purity: f64,
    pub trajectory: f64,
    pub mps: f64,
    /// `None` unless every gate is Clifford.
    pub stabilizer: Option<f64>,
    /// `None` when the circuit has no OpenQASM 2 (or 3) form.
    pub qasm2: Option<f64>,
    pub qasm3: Option<f64>,
}

impl DifferentialReport {
    /// The simulators whose deviation exceeds its tolerance, with the
    /// deviation.
    pub fn failures(&self) -> Vec<(&'static str, f64)> {
        let single = SINGLE_TOLERANCE * self.gates.max(1) as f64;
        let mut out = Vec::new();
        let mut check = |name: &'static str, value: f64, limit: f64| {
            // NaN fails too.
            if value.is_nan() || value > limit {
                out.push((name, value));
            }
        };
        check("single_precision", self.single_precision, single);
        check("density_matrix", self.density_matrix, TOLERANCE);
        check(
            "density_matrix_purity",
            self.density_matrix_purity,
            TOLERANCE,
        );
        check("trajectory", self.trajectory, TOLERANCE);
        check("mps", self.mps, TOLERANCE);
        if let Some(s) = self.stabilizer {
            check("stabilizer", s, TOLERANCE);
        }
        if let Some(q) = self.qasm2 {
            check("qasm2", q, TOLERANCE);
        }
        if let Some(q) = self.qasm3 {
            check("qasm3", q, TOLERANCE);
        }
        out
    }
}

/// `1 − |⟨a|b⟩|²`, at least 0.
fn infidelity(a: &StateVector, b: &StateVector) -> Result<f64> {
    Ok((1.0 - a.fidelity(b)?).max(0.0))
}

/// Run `circuit(theta)` on every simulator that takes it (see the module
/// documentation).
pub fn compare(circuit: &Circuit, theta: &[f64]) -> Result<DifferentialReport> {
    let n = circuit.n_qubits();
    if n == 0 || n > MAX_DIFFERENTIAL_QUBITS {
        return Err(QsoError::invalid(
            "circuit",
            format!("between 1 and {MAX_DIFFERENTIAL_QUBITS} qubits"),
        ));
    }
    let reference = StateVector::from_circuit(circuit, theta)?;
    let single = StateVector32::from_circuit(circuit, theta)?;
    let single_precision = (1.0 - single.fidelity_with(&reference)?).max(0.0);
    let rho = DensityMatrix::from_circuit(circuit, theta)?;
    let density_matrix = (1.0 - rho.fidelity_with_pure(&reference)?).max(0.0);
    let density_matrix_purity = (1.0 - rho.purity()).abs();
    let trajectories = TrajectorySimulator::new(DeviceModel::new("noiseless"), 1, 0)?;
    let trajectory = infidelity(
        &trajectories.trajectory_state(circuit, theta, 0)?,
        &reference,
    )?;
    let options = MpsOptions {
        max_bond: 1 << n.div_ceil(2),
        cutoff: 0.0,
    };
    let mps = infidelity(
        &Mps::from_circuit(circuit, theta, options)?.to_statevector()?,
        &reference,
    )?;
    let stabilizer = match StabilizerState::from_circuit(circuit, theta) {
        Ok(tableau) => Some(pauli_deviation(&tableau, &reference)?),
        Err(_) => None,
    };
    // An export either fails (no such form) or must read back exactly.
    let qasm2 = match to_qasm2(circuit, theta, false) {
        Ok(text) => Some(infidelity(
            &StateVector::from_circuit(&from_qasm2(&text)?, &[])?,
            &reference,
        )?),
        Err(_) => None,
    };
    let qasm3 = match to_qasm3(circuit, theta, false) {
        Ok(text) => Some(infidelity(
            &StateVector::from_circuit(&Program::from_qasm(&text)?.to_circuit()?, &[])?,
            &reference,
        )?),
        Err(_) => None,
    };
    Ok(DifferentialReport {
        n_qubits: n,
        gates: circuit.len(),
        single_precision,
        density_matrix,
        density_matrix_purity,
        trajectory,
        mps,
        stabilizer,
        qasm2,
        qasm3,
    })
}

/// Largest difference between the tableau's and the state vector's
/// expectation values of Pauli strings: every string on up to 5 qubits,
/// 1024 random ones (fixed seed) beyond.
fn pauli_deviation(tableau: &StabilizerState, reference: &StateVector) -> Result<f64> {
    let n = reference.n_qubits();
    let letters = ['I', 'X', 'Y', 'Z'];
    let strings: Vec<Vec<usize>> = if n <= 5 {
        (0..1usize << (2 * n))
            .map(|k| (0..n).map(|q| (k >> (2 * q)) & 3).collect())
            .collect()
    } else {
        let mut rng = Rng::seed_from_u64(1);
        (0..1024)
            .map(|_| (0..n).map(|_| rng.below(4) as usize).collect())
            .collect()
    };
    let mut worst = 0.0f64;
    for s in strings {
        let text: Vec<String> = s
            .iter()
            .enumerate()
            .filter(|&(_, &l)| l != 0)
            .map(|(q, &l)| format!("{}{q}", letters[l]))
            .collect();
        if text.is_empty() {
            continue;
        }
        let h: PauliSum = text.join(" ").parse()?;
        let a = tableau.expectation_sum(&h)?;
        let b = h.expectation(reference)?;
        worst = worst.max((a - b).abs());
    }
    Ok(worst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bell_and_t_circuits_agree_everywhere() {
        let mut c = Circuit::new(3);
        c.h(0).cx(0, 1).s(2).cz(1, 2).swap(0, 2);
        let r = compare(&c, &[]).unwrap();
        assert!(r.stabilizer.is_some());
        assert!(r.failures().is_empty(), "{r:?}");
        c.t(1).rx(2, 0.3);
        let r = compare(&c, &[]).unwrap();
        assert!(r.stabilizer.is_none());
        assert!(r.failures().is_empty(), "{r:?}");
        assert!(compare(&Circuit::new(11), &[]).is_err());
    }

    #[test]
    fn failures_name_the_simulator() {
        let mut r = compare(&Circuit::new(2), &[]).unwrap();
        r.mps = 0.1;
        r.single_precision = f64::NAN;
        let names: Vec<&str> = r.failures().iter().map(|f| f.0).collect();
        assert_eq!(names, ["single_precision", "mps"]);
    }
}
