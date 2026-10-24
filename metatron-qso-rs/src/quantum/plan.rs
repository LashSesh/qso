//! Execution plans: gate fusion and cache blocking for circuit simulation.
//!
//! Applying a circuit gate by gate streams the whole state vector through
//! memory once per gate. Two transformations reduce that traffic without
//! changing the result beyond rounding:
//!
//! * **Fusion.** Runs of uncontrolled single-qubit gates on the same qubit
//!   are multiplied into one 2×2 matrix (an `RY·RZ` pair of an ansatz layer,
//!   `H·RZ`, …).
//! * **Cache blocking.** A gate is *local* to blocks of `2^c` amplitudes if
//!   its targets lie below qubit `c` (controls may lie anywhere: on a block
//!   they are either all satisfied by the block's high bits or not), or if
//!   it is diagonal (its phase on a block depends on the high bits through
//!   a constant). Consecutive local gates are applied block by block, each
//!   block staying in cache for the whole run of gates, and the blocks in
//!   parallel. Gates that are not local are applied as before, with
//!   parallelism inside the gate.
//!
//! The block size is chosen to fit the per-core L2 cache.

use num_complex::Complex64 as C64;

use super::circuit::{Circuit, Instruction};
use super::gates::{GateKind, matrix_1q};
use super::kernels::{apply_1q, apply_instruction, apply_instruction_masked, apply_phase_on_mask};

/// log₂ of the block size in amplitudes (2¹⁴ amplitudes = 256 KiB).
#[cfg(not(miri))]
pub(crate) const BLOCK_QUBITS: usize = 14;
#[cfg(miri)]
pub(crate) const BLOCK_QUBITS: usize = 3;

/// Bound angles of one instruction (at most three).
#[derive(Clone, Copy, Debug)]
struct Angles {
    values: [f64; 3],
    len: usize,
}

impl Angles {
    fn of(inst: &Instruction, theta: &[f64]) -> Self {
        let mut values = [0.0; 3];
        for (slot, p) in values.iter_mut().zip(&inst.params) {
            *slot = p.value(theta);
        }
        Self {
            values,
            len: inst.params.len().min(3),
        }
    }

    fn get(&self) -> &[f64] {
        &self.values[..self.len]
    }
}

/// One operation of a plan.
#[derive(Clone, Debug)]
enum Op<'a> {
    /// A circuit instruction with bound angles.
    Gate(&'a Instruction, Angles),
    /// A fused run of uncontrolled single-qubit gates.
    Fused { qubit: usize, matrix: [C64; 4] },
}

impl Op<'_> {
    /// Apply with the given control mask (the op's own controls ignored).
    fn apply(&self, amps: &mut [C64], controls: usize) {
        match self {
            Op::Gate(inst, angles) => {
                apply_instruction_masked(amps, inst, angles.get(), false, 0, controls)
            }
            Op::Fused { qubit, matrix } => apply_1q(amps, *qubit, controls, matrix),
        }
    }

    fn apply_full(&self, amps: &mut [C64]) {
        match self {
            Op::Gate(inst, angles) => apply_instruction(amps, inst, angles.get(), false, 0),
            Op::Fused { qubit, matrix } => apply_1q(amps, *qubit, 0, matrix),
        }
    }
}

/// How a local operation acts on one block.
#[derive(Clone, Debug)]
enum Local<'a> {
    /// Targets below the block size: apply `op` with the controls below the
    /// block size to blocks whose index has all `high_controls` set.
    Targets {
        op: Op<'a>,
        low_controls: usize,
        high_controls: usize,
    },
    /// Multiply by `phase` where all bits of `mask` are set.
    Mask { mask: usize, phase: C64 },
    /// Where `controls` are set: `p1` if `bit` is set, else `p0` (RZ).
    Bit {
        controls: usize,
        bit: usize,
        p0: C64,
        p1: C64,
    },
    /// Where `controls` are set: `same` if the bits of `mask` have even
    /// parity, else `diff` (RZZ).
    Parity {
        controls: usize,
        mask: usize,
        same: C64,
        diff: C64,
    },
}

fn mat_mul(a: &[C64; 4], b: &[C64; 4]) -> [C64; 4] {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

/// 2×2 matrix of an uncontrolled single-qubit instruction, if it is one.
fn single_qubit_matrix(inst: &Instruction, angles: &[f64]) -> Option<[C64; 4]> {
    if !inst.controls.is_empty() || inst.targets.len() != 1 {
        return None;
    }
    match inst.kind {
        GateKind::Unitary => inst
            .matrix
            .as_ref()
            .and_then(|m| <[C64; 4]>::try_from(&m[..]).ok()),
        kind if kind.num_targets() == Some(1) => Some(matrix_1q(kind, angles)),
        _ => None,
    }
}

/// Bind parameters and fuse runs of uncontrolled single-qubit gates on
/// qubits `≥ c`. Gates on those qubits are not local to blocks, so each one
/// is a full pass over the state; fusing a run saves passes. Below `c`
/// gates stay separate: inside a block they are cheap and keep their
/// dedicated kernels (real, diagonal, permutation).
fn fuse<'a>(circuit: &'a Circuit, theta: &[f64], c: usize) -> Vec<Op<'a>> {
    type Run<'a> = Option<([C64; 4], Op<'a>, usize)>;
    let n = circuit.n_qubits();
    let mut pending: Vec<Run<'a>> = vec![None; n];
    let mut ops = Vec::with_capacity(circuit.len());
    let flush = |q: usize, pending: &mut Vec<Run<'a>>, ops: &mut Vec<Op<'a>>| {
        if let Some((matrix, first, count)) = pending[q].take() {
            ops.push(if count == 1 {
                first
            } else {
                Op::Fused { qubit: q, matrix }
            });
        }
    };
    for inst in circuit.instructions() {
        let angles = Angles::of(inst, theta);
        if inst.targets[0] >= c
            && let Some(m) = single_qubit_matrix(inst, angles.get())
        {
            let q = inst.targets[0];
            pending[q] = Some(match pending[q].take() {
                None => (m, Op::Gate(inst, angles), 1),
                Some((prev, first, count)) => (mat_mul(&m, &prev), first, count + 1),
            });
            continue;
        }
        for q in inst.qubits() {
            flush(q, &mut pending, &mut ops);
        }
        ops.push(Op::Gate(inst, angles));
    }
    for q in 0..n {
        flush(q, &mut pending, &mut ops);
    }
    ops
}

/// The block form of `op` for blocks of `2^c` amplitudes, if it is local.
fn local<'a>(op: &Op<'a>, c: usize) -> Option<Local<'a>> {
    let low = (1usize << c) - 1;
    let (inst, angles) = match op {
        Op::Fused { qubit, .. } => {
            return (*qubit < c).then(|| Local::Targets {
                op: op.clone(),
                low_controls: 0,
                high_controls: 0,
            });
        }
        Op::Gate(inst, angles) => (*inst, angles.get()),
    };
    let controls = inst.control_mask();
    if inst.targets.iter().all(|&t| t < c) {
        return Some(Local::Targets {
            op: op.clone(),
            low_controls: controls & low,
            high_controls: controls & !low,
        });
    }
    let target_mask = inst.targets.iter().fold(0usize, |m, &t| m | 1 << t);
    match inst.kind {
        GateKind::Z | GateKind::S | GateKind::Sdg | GateKind::T | GateKind::Tdg | GateKind::P => {
            let m = matrix_1q(inst.kind, angles);
            Some(Local::Mask {
                mask: controls | target_mask,
                phase: m[3],
            })
        }
        GateKind::RZ => {
            let m = matrix_1q(GateKind::RZ, angles);
            Some(Local::Bit {
                controls,
                bit: target_mask,
                p0: m[0],
                p1: m[3],
            })
        }
        GateKind::RZZ => {
            let half = angles[0] / 2.0;
            Some(Local::Parity {
                controls,
                mask: target_mask,
                same: C64::from_polar(1.0, -half),
                diff: C64::from_polar(1.0, half),
            })
        }
        _ => None,
    }
}

fn scale(block: &mut [C64], s: C64) {
    block.iter_mut().for_each(|a| *a *= s);
}

/// Apply a local operation to the block starting at global index `base`.
fn apply_local(block: &mut [C64], base: usize, local: &Local<'_>) {
    let low = block.len() - 1;
    match local {
        Local::Targets {
            op,
            low_controls,
            high_controls,
        } => {
            if base & high_controls == *high_controls {
                op.apply(block, *low_controls);
            }
        }
        Local::Mask { mask, phase } => {
            let high = mask & !low;
            if base & high != high {
                return;
            }
            match mask & low {
                0 => scale(block, *phase),
                m => apply_phase_on_mask(block, m, *phase),
            }
        }
        Local::Bit {
            controls,
            bit,
            p0,
            p1,
        } => {
            let high = controls & !low;
            if base & high != high {
                return;
            }
            let low_controls = controls & low;
            if bit & low == 0 {
                let p = if base & bit != 0 { *p1 } else { *p0 };
                match low_controls {
                    0 => scale(block, p),
                    m => apply_phase_on_mask(block, m, p),
                }
            } else {
                let zero = C64::new(0.0, 0.0);
                let q = bit.trailing_zeros() as usize;
                apply_1q(block, q, low_controls, &[*p0, zero, zero, *p1]);
            }
        }
        Local::Parity {
            controls,
            mask,
            same,
            diff,
        } => {
            let high = controls & !low;
            if base & high != high {
                return;
            }
            let odd_high = (base & mask & !low).count_ones() % 2 == 1;
            let (low_mask, low_controls) = (mask & low, controls & low);
            if low_mask == 0 && low_controls == 0 {
                scale(block, if odd_high { *diff } else { *same });
                return;
            }
            for (i, a) in block.iter_mut().enumerate() {
                if i & low_controls == low_controls {
                    let odd = ((i & low_mask).count_ones() % 2 == 1) != odd_high;
                    *a *= if odd { *diff } else { *same };
                }
            }
        }
    }
}

/// Apply a run of local operations block by block, blocks in parallel.
fn apply_run(amps: &mut [C64], c: usize, run: &[Local<'_>]) {
    let block = 1usize << c;
    let apply = |(k, chunk): (usize, &mut [C64])| {
        for local in run {
            apply_local(chunk, k * block, local);
        }
    };
    #[cfg(feature = "parallel")]
    if amps.len() > block {
        use rayon::prelude::*;
        amps.par_chunks_mut(block).enumerate().for_each(apply);
        return;
    }
    amps.chunks_mut(block).enumerate().for_each(apply);
}

/// Apply `circuit` with parameters `theta` (already validated) to `amps`,
/// fusing gates and blocking for cache with blocks of `2^c` amplitudes.
/// Registers of at most `2^c` amplitudes are simulated gate by gate: they
/// fit in cache already.
pub(crate) fn run_with_block(amps: &mut [C64], circuit: &Circuit, theta: &[f64], c: usize) {
    let n = amps.len().trailing_zeros() as usize;
    if n <= c {
        for inst in circuit.instructions() {
            let angles = Angles::of(inst, theta);
            apply_instruction(amps, inst, angles.get(), false, 0);
        }
        return;
    }
    let ops = fuse(circuit, theta, c);
    let mut run: Vec<Local<'_>> = Vec::new();
    for op in &ops {
        match local(op, c) {
            Some(l) => run.push(l),
            None => {
                if !run.is_empty() {
                    apply_run(amps, c, &run);
                    run.clear();
                }
                op.apply_full(amps);
            }
        }
    }
    if !run.is_empty() {
        apply_run(amps, c, &run);
    }
}

/// Call `f(base, block)` for every block of `2^c` amplitudes (`base` is the
/// register index of its first amplitude), blocks in parallel.
pub(crate) fn for_each_block<F>(amps: &mut [C64], c: usize, f: F)
where
    F: Fn(usize, &mut [C64]) + Sync + Send,
{
    let block = 1usize << c;
    #[cfg(feature = "parallel")]
    if amps.len() > block {
        use rayon::prelude::*;
        amps.par_chunks_mut(block)
            .enumerate()
            .for_each(|(k, chunk)| f(k * block, chunk));
        return;
    }
    amps.chunks_mut(block)
        .enumerate()
        .for_each(|(k, chunk)| f(k * block, chunk));
}

/// Apply `circuit` with parameters `theta` (already validated) to `amps`.
pub(crate) fn run(amps: &mut [C64], circuit: &Circuit, theta: &[f64]) {
    run_with_block(amps, circuit, theta, BLOCK_QUBITS);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::StateVector;
    use crate::rng::Rng;

    /// A random circuit over the whole gate set, with random controls.
    fn random_circuit(n: usize, gates: usize, rng: &mut Rng) -> Circuit {
        let mut c = Circuit::new(n);
        let pick = |rng: &mut Rng, k: usize| -> Vec<usize> {
            let mut q: Vec<usize> = (0..n).collect();
            rng.shuffle(&mut q);
            q.truncate(k);
            q
        };
        for _ in 0..gates {
            let a = rng.uniform_range(-3.0, 3.0);
            let kind = if n < 2 { 13 } else { rng.below(14) };
            let controls = if n > 3 && rng.uniform() < 0.3 {
                1 + rng.below(2) as usize
            } else {
                0
            };
            let q = pick(rng, (2 + controls).min(n));
            let (t, cs) = (q[0], &q[q.len().min(2)..]);
            match kind {
                0 => c.controlled(GateKind::H, &[t], cs, &[]).map(|_| ()),
                1 => c.controlled(GateKind::X, &[t], cs, &[]).map(|_| ()),
                2 => c.controlled(GateKind::Y, &[t], cs, &[]).map(|_| ()),
                3 => c.controlled(GateKind::Z, &[t], cs, &[]).map(|_| ()),
                4 => c.controlled(GateKind::T, &[t], cs, &[]).map(|_| ()),
                5 => c
                    .controlled(GateKind::RX, &[t], cs, &[a.into()])
                    .map(|_| ()),
                6 => c
                    .controlled(GateKind::RY, &[t], cs, &[a.into()])
                    .map(|_| ()),
                7 => c
                    .controlled(GateKind::RZ, &[t], cs, &[a.into()])
                    .map(|_| ()),
                8 => c.controlled(GateKind::P, &[t], cs, &[a.into()]).map(|_| ()),
                9 => c
                    .controlled(
                        GateKind::U,
                        &[t],
                        cs,
                        &[a.into(), (0.5 * a).into(), 1.1.into()],
                    )
                    .map(|_| ()),
                10 => c
                    .controlled(GateKind::Swap, &[t, q[1]], cs, &[])
                    .map(|_| ()),
                11 => c
                    .controlled(GateKind::RZZ, &[t, q[1]], cs, &[a.into()])
                    .map(|_| ()),
                12 => c
                    .controlled(GateKind::RXX, &[t, q[1]], cs, &[a.into()])
                    .map(|_| ()),
                _ => {
                    c.h(t).rz(t, a).ry(t, 0.3 * a).sx(t);
                    Ok(())
                }
            }
            .unwrap();
        }
        c
    }

    #[test]
    fn plans_match_gate_by_gate_simulation() {
        let mut rng = Rng::seed_from_u64(17);
        let max_n = if cfg!(miri) { 5 } else { 9 };
        for n in 1..=max_n {
            for c in [1, 2, 3, 5, 14] {
                let circuit = random_circuit(n, 60, &mut rng);
                let psi = StateVector::random(n, &mut rng);
                let mut reference = psi.amplitudes().to_vec();
                for inst in circuit.instructions() {
                    apply_instruction(&mut reference, inst, &inst.angles(&[]), false, 0);
                }
                let mut planned = psi.amplitudes().to_vec();
                run_with_block(&mut planned, &circuit, &[], c);
                for (a, b) in planned.iter().zip(&reference) {
                    assert!((a - b).norm() < 1e-12, "n={n} c={c}");
                }
            }
        }
    }

    #[test]
    fn fusion_merges_single_qubit_runs_above_the_block() {
        let mut c = Circuit::new(2);
        c.h(0).rz(0, 0.3).ry(1, 0.2).cx(0, 1).x(0).t(1).s(1);
        let describe = |ops: &[Op<'_>]| -> Vec<String> {
            ops.iter()
                .map(|o| match o {
                    Op::Gate(inst, _) => inst.kind.name().to_string(),
                    Op::Fused { qubit, .. } => format!("fused{qubit}"),
                })
                .collect()
        };
        // Everything above the block (c = 0): H·RZ fused, RY alone, CX,
        // X alone, T·S fused.
        assert_eq!(
            describe(&fuse(&c, &[], 0)),
            ["fused0", "ry", "x", "x", "fused1"]
        );
        // Qubit 0 inside the block (c = 1): its gates stay separate.
        assert_eq!(
            describe(&fuse(&c, &[], 1)),
            ["h", "rz", "ry", "x", "x", "fused1"]
        );
    }
}
