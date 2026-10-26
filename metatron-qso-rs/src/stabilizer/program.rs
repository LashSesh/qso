//! Clifford circuits and programs on the stabilizer simulator.

use std::collections::BTreeMap;

use super::{StabilizerState, quarter_turns};
use crate::error::{QsoError, Result};
use crate::quantum::program::MAX_SHOT_WORK;
use crate::quantum::{Circuit, GateKind, Instruction, Operation, Program};
use crate::rng::Rng;

/// Whether the stabilizer simulator can apply `inst` (parameters bound to
/// `theta`).
pub fn is_clifford_instruction(inst: &Instruction, theta: &[f64]) -> bool {
    use GateKind::*;
    let a = inst.angles(theta);
    let quarter = |x: f64| quarter_turns(x, "").ok();
    match (inst.kind, inst.controls.len()) {
        (H | X | Y | Z | S | Sdg | SX | SXdg | Swap, 0) => true,
        (X | Y | Z, 1) => true,
        (RX | RY | RZ | P | RXX | RYY | RZZ, 0) => quarter(a[0]).is_some(),
        (U, 0) => a.iter().all(|&x| quarter(x).is_some()),
        (P, 1) => matches!(quarter(a[0]), Some(0 | 2)),
        _ => false,
    }
}

/// Whether every gate of `circuit` (with `theta`) is Clifford.
pub fn is_clifford_circuit(circuit: &Circuit, theta: &[f64]) -> bool {
    circuit
        .instructions()
        .iter()
        .all(|i| is_clifford_instruction(i, theta))
}

fn clifford_ops(ops: &[Operation]) -> bool {
    ops.iter().all(|o| match o {
        Operation::Gate(i) => is_clifford_instruction(i, &[]),
        Operation::If { body, .. } => clifford_ops(body),
        _ => true,
    })
}

/// Whether every gate of `program` is Clifford.
pub fn is_clifford_program(program: &Program) -> bool {
    clifford_ops(program.operations())
}

impl StabilizerState {
    /// Run `circuit(theta)` on `|0…0⟩`.
    pub fn from_circuit(circuit: &Circuit, theta: &[f64]) -> Result<Self> {
        let mut s = StabilizerState::zero(circuit.n_qubits())?;
        s.apply_circuit(circuit, theta)?;
        Ok(s)
    }

    /// Apply `circuit(theta)` (on the first `circuit.n_qubits()` qubits).
    /// The state is unchanged if a gate is not Clifford.
    pub fn apply_circuit(&mut self, circuit: &Circuit, theta: &[f64]) -> Result<()> {
        if circuit.n_qubits() > self.n_qubits() {
            return Err(QsoError::DimensionMismatch {
                expected: self.n_qubits(),
                actual: circuit.n_qubits(),
            });
        }
        if let Some(bad) = circuit
            .instructions()
            .iter()
            .find(|i| !is_clifford_instruction(i, theta))
        {
            return Err(super::not_clifford(bad));
        }
        for inst in circuit.instructions() {
            self.apply_instruction(inst, theta)?;
        }
        Ok(())
    }
}

fn run_ops(
    s: &mut StabilizerState,
    record: &mut u64,
    ops: &[Operation],
    rng: &mut Rng,
) -> Result<()> {
    for op in ops {
        match op {
            Operation::Gate(inst) => s.apply_instruction(inst, &[])?,
            Operation::Measure { qubit, bit } => {
                let v = s.measure(*qubit, rng)?;
                *record = (*record & !(1 << bit)) | (u64::from(v) << bit);
            }
            Operation::Reset { qubit } => s.reset(*qubit, rng)?,
            Operation::If { condition, body } => {
                if condition.holds(*record) {
                    run_ops(s, record, body, rng)?;
                }
            }
        }
    }
    Ok(())
}

/// One shot of a Clifford program: its classical record (bit `k` of the
/// result is bit `k` of the program).
pub fn run_program(program: &Program, rng: &mut Rng) -> Result<u64> {
    // A program without qubits (only bits) runs on a one-qubit dummy.
    let mut s = StabilizerState::zero(program.n_qubits().max(1))?;
    let mut record = 0;
    run_ops(&mut s, &mut record, program.operations(), rng)?;
    Ok(record)
}

/// `shots` seeded samples of a Clifford program's classical record. A
/// program of gates followed by measurements runs once and is then
/// sampled from the outcome distribution; any other program is simulated
/// shot by shot (shots × operations at most 2³⁰).
pub fn sample_program(program: &Program, shots: usize, seed: u64) -> Result<BTreeMap<u64, usize>> {
    if !is_clifford_program(program) {
        return Err(QsoError::invalid(
            "program",
            "has gates outside the Clifford group; the stabilizer simulator cannot run it",
        ));
    }
    let mut rng = Rng::seed_from_u64(seed);
    let mut counts = BTreeMap::new();
    let ops = program.operations();
    let first_measure = ops
        .iter()
        .position(|o| !matches!(o, Operation::Gate(_)))
        .unwrap_or(ops.len());
    let terminal = ops[first_measure..]
        .iter()
        .all(|o| matches!(o, Operation::Measure { .. }));
    if terminal {
        // A program without qubits (only bits) runs on a one-qubit dummy.
    let mut s = StabilizerState::zero(program.n_qubits().max(1))?;
        for op in &ops[..first_measure] {
            if let Operation::Gate(inst) = op {
                s.apply_instruction(inst, &[])?;
            }
        }
        let sampler = s.sampler();
        let pairs: Vec<(usize, usize)> = ops[first_measure..]
            .iter()
            .filter_map(|o| match o {
                Operation::Measure { qubit, bit } => Some((*qubit, *bit)),
                _ => None,
            })
            .collect();
        for _ in 0..shots {
            let v = sampler.sample_words(&mut rng);
            let mut record = 0u64;
            for &(q, b) in &pairs {
                let bit = (v[q / 64] >> (q % 64)) & 1;
                record = (record & !(1 << b)) | (bit << b);
            }
            *counts.entry(record).or_default() += 1;
        }
        return Ok(counts);
    }
    if shots.saturating_mul(program.len().max(1)) > MAX_SHOT_WORK {
        return Err(QsoError::invalid(
            "shots",
            format!("shots × operations exceed {MAX_SHOT_WORK}"),
        ));
    }
    for _ in 0..shots {
        *counts.entry(run_program(program, &mut rng)?).or_default() += 1;
    }
    Ok(counts)
}
