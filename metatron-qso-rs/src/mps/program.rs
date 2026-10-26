//! OpenQASM programs on matrix-product states.

use std::collections::BTreeMap;

use super::{MAX_SAMPLING_WORK, Mps, MpsOptions};
use crate::error::{QsoError, Result};
use crate::quantum::program::MAX_SHOT_WORK;
use crate::quantum::{Operation, Program};
use crate::rng::Rng;

fn run_ops(s: &mut Mps, record: &mut u64, ops: &[Operation], rng: &mut Rng) -> Result<()> {
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

/// Seeded samples of a program's classical record (bit `k` of a key is bit
/// `k` of the program), and the state's fidelity bound after its gates
/// (see [`Mps::fidelity_lower_bound`]; for a program with mid-circuit
/// measurements, the smallest bound over the shots that has one).
///
/// A program of gates followed by measurements runs once and is then
/// sampled; any other program runs shot by shot from the state after its
/// leading gates.
pub fn sample_program(
    program: &Program,
    shots: usize,
    seed: u64,
    options: MpsOptions,
) -> Result<(BTreeMap<u64, usize>, Option<f64>)> {
    let mut rng = Rng::seed_from_u64(seed);
    let ops = program.operations();
    let first = ops
        .iter()
        .position(|o| !matches!(o, Operation::Gate(_)))
        .unwrap_or(ops.len());
    // A program without qubits (only bits) runs on a one-qubit dummy.
    let mut prefix = Mps::zero(program.n_qubits().max(1), options)?;
    for op in &ops[..first] {
        if let Operation::Gate(inst) = op {
            prefix.apply_instruction(inst, &[])?;
        }
    }
    let mut counts = BTreeMap::new();
    let terminal = ops[first..]
        .iter()
        .all(|o| matches!(o, Operation::Measure { .. }));
    if terminal {
        let bound = prefix.fidelity_lower_bound();
        let mut last = BTreeMap::new();
        for op in &ops[first..] {
            if let Operation::Measure { qubit, bit } = op {
                last.insert(*bit, *qubit);
            }
        }
        for bits in prefix.sample(shots, &mut rng)? {
            let r = last
                .iter()
                .fold(0u64, |acc, (&b, &q)| acc | (u64::from(bits[q]) << b));
            *counts.entry(r).or_default() += 1;
        }
        return Ok((counts, bound));
    }
    let rest = &ops[first..];
    let work = shots as u128 * rest.len().max(1) as u128 * prefix.entries() as u128;
    if shots.saturating_mul(rest.len().max(1)) > MAX_SHOT_WORK || work > MAX_SAMPLING_WORK {
        return Err(QsoError::invalid(
            "shots",
            format!(
                "shot-by-shot simulation needs shots × operations × tensor entries at most {MAX_SAMPLING_WORK}"
            ),
        ));
    }
    let mut bound: Option<f64> = None;
    for _ in 0..shots {
        let mut s = prefix.clone();
        let mut record = 0;
        run_ops(&mut s, &mut record, rest, &mut rng)?;
        if let Some(b) = s.fidelity_lower_bound() {
            bound = Some(bound.map_or(b, |x: f64| x.min(b)));
        }
        *counts.entry(record).or_default() += 1;
    }
    Ok((counts, bound))
}
