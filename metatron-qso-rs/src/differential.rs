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
//!   OpenQASM 3 export, when the exporter has a form for every gate (it
//!   refuses dense unitaries on several qubits and some controlled
//!   two-qubit gates; any other export error is reported),
//!
//! and reports how far each result lies from the reference. States are
//! compared amplitude by amplitude after aligning the global phase, so the
//! measure is linear in the error (an angle off by 10⁻⁸ shows as about
//! 10⁻⁸); norms are reported separately. The property tests and the
//! `differential` fuzz target check every deviation against
//! [`DifferentialReport::failures`].

use serde::{Deserialize, Serialize};

use crate::C64;
use crate::error::{QsoError, Result};
use crate::mps::{Mps, MpsOptions};
use crate::noise::{DensityMatrix, DeviceModel, TrajectorySimulator};
use crate::quantum::{
    Circuit, PauliSum, Program, StateVector, StateVector32, from_qasm2, to_qasm2, to_qasm3,
};
use crate::rng::Rng;
use crate::stabilizer::program::is_clifford_circuit;
use crate::stabilizer::{StabilizerState, clifford_tolerance};

/// Largest register [`compare`] takes (the density matrix has 4ⁿ entries).
pub const MAX_DIFFERENTIAL_QUBITS: usize = 10;
/// Largest deviation between double-precision simulators (amplitudes,
/// density-matrix entries, norms).
pub const TOLERANCE: f64 = 1e-10;
/// Largest deviation of single precision, per gate.
pub const SINGLE_TOLERANCE: f64 = 1e-7;

/// How far each simulator's result lies from the double-precision state
/// vector. States: the largest amplitude difference after aligning the
/// global phase (single precision after normalising); norms: `|‖ψ‖ − 1|`;
/// density matrix: the largest entry of `ρ − |ψ⟩⟨ψ|`; tableau: the largest
/// difference of Pauli expectation values. `None` where a simulator does
/// not take the circuit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DifferentialReport {
    pub n_qubits: usize,
    pub gates: usize,
    pub single_precision: f64,
    pub single_precision_norm: f64,
    pub density_matrix: f64,
    /// `|Tr ρ − 1|`.
    pub density_matrix_trace: f64,
    pub trajectory: f64,
    pub trajectory_norm: f64,
    pub mps: f64,
    /// The distance δ of the states the MPS's truncations dropped (its
    /// own error bound): even an unlimited bond drops singular values below
    /// about 10⁻⁸ of the largest, at the rounding of its Gram matrices.
    pub mps_truncation: f64,
    /// `None` unless every gate is Clifford.
    pub stabilizer: Option<f64>,
    /// How far the tableau may deviate: it applies rotations within
    /// [`clifford_tolerance`] of a Clifford gate as that gate, and each
    /// may move an expectation value by as much.
    pub stabilizer_tolerance: f64,
    /// `None` when the exporter has no form for some gate.
    pub qasm2: Option<f64>,
    pub qasm3: Option<f64>,
}

impl DifferentialReport {
    /// The simulators whose deviation exceeds its tolerance, with the
    /// deviation.
    pub fn failures(&self) -> Vec<(&'static str, f64)> {
        let gates = self.gates.max(1) as f64;
        let single = SINGLE_TOLERANCE * gates;
        let mut out = Vec::new();
        let mut check = |name: &'static str, value: f64, limit: f64| {
            if value.is_nan() || value > limit {
                out.push((name, value));
            }
        };
        check("single_precision", self.single_precision, single);
        check("single_precision_norm", self.single_precision_norm, single);
        check("density_matrix", self.density_matrix, TOLERANCE);
        check("density_matrix_trace", self.density_matrix_trace, TOLERANCE);
        check("trajectory", self.trajectory, TOLERANCE);
        check("trajectory_norm", self.trajectory_norm, TOLERANCE);
        check("mps", self.mps, TOLERANCE + 2.0 * self.mps_truncation);
        if let Some(s) = self.stabilizer {
            check("stabilizer", s, TOLERANCE + self.stabilizer_tolerance);
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

/// The largest `|a_i e^{iφ} − b_i|` with `φ` aligning the global phases
/// (`φ = arg⟨a|b⟩`).
fn distance(a: &[C64], b: &[C64]) -> f64 {
    let overlap: C64 = a.iter().zip(b).map(|(x, y)| x.conj() * y).sum();
    let phase = if overlap.norm() > 0.0 {
        overlap / overlap.norm()
    } else {
        C64::new(1.0, 0.0)
    };
    a.iter()
        .zip(b)
        .map(|(x, y)| (x * phase - y).norm())
        .fold(0.0, f64::max)
}

/// The exporters' refusal of a gate they have no form for (as opposed to
/// a failure).
fn no_form(e: &QsoError) -> bool {
    matches!(e, QsoError::InvalidArgument { name: "circuit", reason } if reason.contains("no OpenQASM"))
}

/// The state read back from an export, or `None` when the exporter has no
/// form for the circuit.
fn read_back(
    export: Result<String>,
    import: impl Fn(&str) -> Result<Circuit>,
    psi: &[C64],
) -> Result<Option<f64>> {
    match export {
        Ok(text) => {
            let back = StateVector::from_circuit(&import(&text)?, &[])?;
            Ok(Some(distance(back.amplitudes(), psi)))
        }
        Err(e) if no_form(&e) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Run `circuit(theta)` on every simulator that takes it (see the module
/// documentation). An error means that a simulator failed on a circuit it
/// should take, or that the circuit is out of range.
pub fn compare(circuit: &Circuit, theta: &[f64]) -> Result<DifferentialReport> {
    let n = circuit.n_qubits();
    if n == 0 || n > MAX_DIFFERENTIAL_QUBITS {
        return Err(QsoError::invalid(
            "circuit",
            format!("between 1 and {MAX_DIFFERENTIAL_QUBITS} qubits"),
        ));
    }
    let reference = StateVector::from_circuit(circuit, theta)?;
    let psi = reference.amplitudes();
    let single = StateVector32::from_circuit(circuit, theta)?;
    let norm32 = single.norm();
    let single_amps: Vec<C64> = single
        .amplitudes()
        .iter()
        .map(|z| C64::new(z.re as f64, z.im as f64) / norm32)
        .collect();
    let rho = DensityMatrix::from_circuit(circuit, theta)?;
    let dim = psi.len();
    let mut density_matrix = 0.0f64;
    let mut trace = C64::new(0.0, 0.0);
    for i in 0..dim {
        for j in 0..dim {
            density_matrix = density_matrix.max((rho.get(i, j) - psi[i] * psi[j].conj()).norm());
        }
        trace += rho.get(i, i);
    }
    let trajectories = TrajectorySimulator::new(DeviceModel::new("noiseless"), 1, 0)?;
    let trajectory = trajectories.trajectory_state(circuit, theta, 0)?;
    let options = MpsOptions {
        max_bond: 1 << n.div_ceil(2),
        ..MpsOptions::exact()
    };
    let mps_state = Mps::from_circuit(circuit, theta, options)?;
    // δ from the bound (1 − δ²/2)².
    let mps_truncation = mps_state
        .fidelity_lower_bound()
        .map_or(f64::NAN, |f| (2.0 * (1.0 - f.sqrt())).max(0.0).sqrt());
    let mps = mps_state.to_statevector()?;
    let stabilizer_tolerance: f64 = circuit
        .instructions()
        .iter()
        .flat_map(|i| i.angles(theta))
        .map(clifford_tolerance)
        .sum();
    let stabilizer = if is_clifford_circuit(circuit, theta) {
        Some(pauli_deviation(
            &StabilizerState::from_circuit(circuit, theta)?,
            &reference,
        )?)
    } else {
        None
    };
    let qasm2 = read_back(to_qasm2(circuit, theta, false), from_qasm2, psi)?;
    let qasm3 = read_back(
        to_qasm3(circuit, theta, false),
        |text| Program::from_qasm(text)?.to_circuit(),
        psi,
    )?;
    Ok(DifferentialReport {
        n_qubits: n,
        gates: circuit.len(),
        single_precision: distance(&single_amps, psi),
        single_precision_norm: (norm32 - 1.0).abs(),
        density_matrix,
        density_matrix_trace: (trace - 1.0).norm(),
        trajectory: distance(trajectory.amplitudes(), psi),
        trajectory_norm: (trajectory.norm() - 1.0).abs(),
        mps: distance(mps.amplitudes(), psi),
        mps_truncation,
        stabilizer,
        stabilizer_tolerance,
        qasm2,
        qasm3,
    })
}

/// Largest difference between the tableau's and the state vector's
/// expectation values: of every stabilizer generator (the state vector
/// must give +1), and of every Pauli string on up to 5 qubits or 1024
/// random ones (fixed seed) beyond.
fn pauli_deviation(tableau: &StabilizerState, reference: &StateVector) -> Result<f64> {
    let n = reference.n_qubits();
    let mut worst = 0.0f64;
    for g in tableau.stabilizers() {
        let (sign, body) = g.split_at(1);
        let sign = if sign == "-" { -1.0 } else { 1.0 };
        let h: PauliSum = body.trim().parse()?;
        worst = worst.max((sign * h.expectation(reference)? - 1.0).abs());
    }
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
        assert!(r.stabilizer.is_some() && r.qasm2.is_some());
        assert!(r.failures().is_empty(), "{r:?}");
        c.t(1).rx(2, 0.3);
        let r = compare(&c, &[]).unwrap();
        assert!(r.stabilizer.is_none());
        assert!(r.failures().is_empty(), "{r:?}");
        assert!(compare(&Circuit::new(11), &[]).is_err());
    }

    #[test]
    fn small_errors_are_seen_linearly() {
        // An angle off by 10⁻⁸ moves amplitudes by about 10⁻⁸, far above
        // the tolerance (1 − F would be 10⁻¹⁶).
        let mut a = Circuit::new(1);
        a.rx(0, 0.3);
        let mut b = Circuit::new(1);
        b.rx(0, 0.3 + 1e-8);
        let (x, y) = (
            StateVector::from_circuit(&a, &[]).unwrap(),
            StateVector::from_circuit(&b, &[]).unwrap(),
        );
        let d = distance(x.amplitudes(), y.amplitudes());
        assert!(d > 4e-9 && d < 1e-8, "{d}");
        // A global phase is not a difference.
        let phased: Vec<C64> = x
            .amplitudes()
            .iter()
            .map(|z| z * C64::from_polar(1.0, 0.7))
            .collect();
        assert!(distance(x.amplitudes(), &phased) < 1e-15);
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
