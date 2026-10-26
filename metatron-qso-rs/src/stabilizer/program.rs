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
    let bound = inst.params.iter().all(|p| match p {
        crate::quantum::Param::Symbol(s) => s.index < theta.len(),
        _ => true,
    });
    if !bound || theta.iter().any(|t| !t.is_finite()) {
        return false;
    }
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

/// Most work (shots × operations × qubits²) of shot-by-shot simulation:
/// minutes at most.
pub const MAX_TABLEAU_WORK: u128 = 1 << 36;

/// `shots` seeded samples of a Clifford program's classical record. A
/// program of gates followed by measurements runs once and is then
/// sampled from the distribution of its (at most 64) record bits, at a
/// cost per shot independent of the number of qubits; any other program
/// is simulated shot by shot (shots × operations × qubits² at most
/// [`MAX_TABLEAU_WORK`]).
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
        // The record is a linear function of the outcomes (the last
        // measurement into a bit wins), so it is uniform over the image of
        // the outcome space: at most 64 basis vectors, whatever n is.
        let mut last = BTreeMap::new();
        for op in &ops[first_measure..] {
            if let Operation::Measure { qubit, bit } = op {
                last.insert(*bit, *qubit);
            }
        }
        let record = |v: &[u64]| {
            last.iter()
                .map(|(&b, &q)| ((v[q / 64] >> (q % 64)) & 1) << b)
                .fold(0u64, |acc, x| acc | x)
        };
        let sampler = s.sampler();
        let offset = record(&sampler.offset);
        let mut basis: Vec<u64> = Vec::new();
        for v in &sampler.basis {
            // Reduce against the basis so far; keep what is independent.
            let mut r = record(v);
            for &b in &basis {
                r = r.min(r ^ b);
            }
            if r != 0 {
                basis.push(r);
                basis.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        for _ in 0..shots {
            let bits = rng.next_u64();
            let mut r = offset;
            for (k, &b) in basis.iter().enumerate() {
                if (bits >> k) & 1 == 1 {
                    r ^= b;
                }
            }
            *counts.entry(r).or_default() += 1;
        }
        return Ok(counts);
    }
    // Shot by shot: a measurement costs O(n²), so the work bound counts
    // shots × operations × n².
    let n = program.n_qubits().max(1) as u128;
    let work = (shots as u128) * (program.len().max(1) as u128) * n * n;
    if shots.saturating_mul(program.len().max(1)) > MAX_SHOT_WORK || work > MAX_TABLEAU_WORK {
        return Err(QsoError::invalid(
            "shots",
            format!(
                "shots × operations × qubits² exceeds {MAX_TABLEAU_WORK} (or shots × operations {MAX_SHOT_WORK})"
            ),
        ));
    }
    for _ in 0..shots {
        *counts.entry(run_program(program, &mut rng)?).or_default() += 1;
    }
    Ok(counts)
}
