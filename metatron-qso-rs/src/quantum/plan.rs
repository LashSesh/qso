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
use super::scalar::Amplitude;

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
    fn apply<A: Amplitude>(&self, amps: &mut [A], controls: usize) {
        match self {
            Op::Gate(inst, angles) => {
                apply_instruction_masked(amps, inst, angles.get(), false, 0, controls)
            }
            Op::Fused { qubit, matrix } => {
                apply_1q(amps, *qubit, controls, &matrix.map(A::from_c64))
            }
        }
    }

    fn apply_full<A: Amplitude>(&self, amps: &mut [A]) {
        match self {
            Op::Gate(inst, angles) => apply_instruction(amps, inst, angles.get(), false, 0),
            Op::Fused { qubit, matrix } => apply_1q(amps, *qubit, 0, &matrix.map(A::from_c64)),
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
        Op::Fused { qubit, .. } if *qubit < c => {
            return Some(Local::Targets {
                op: op.clone(),
                low_controls: 0,
                high_controls: 0,
            });
        }
        Op::Fused { qubit, matrix } => {
            // A fused run of diagonal gates (T·S, RZ·P, …) stays a phase.
            let zero = C64::new(0.0, 0.0);
            return (matrix[1] == zero && matrix[2] == zero).then(|| Local::Bit {
                controls: 0,
                bit: 1 << qubit,
                p0: matrix[0],
                p1: matrix[3],
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

fn scale<A: Amplitude>(block: &mut [A], s: C64) {
    A::scale_run(block, A::from_c64(s));
}

/// Apply a local operation to the block starting at global index `base`.
fn apply_local<A: Amplitude>(block: &mut [A], base: usize, local: &Local<'_>) {
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
                m => apply_phase_on_mask(block, m, A::from_c64(*phase)),
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
                    m => apply_phase_on_mask(block, m, A::from_c64(p)),
                }
            } else {
                let q = bit.trailing_zeros() as usize;
                let (p0, p1) = (A::from_c64(*p0), A::from_c64(*p1));
                apply_1q(block, q, low_controls, &[p0, A::ZERO, A::ZERO, p1]);
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
            // The phase is constant on runs below the lowest bit involved.
            let run = 1usize << (low_mask | low_controls).trailing_zeros();
            let (same, diff) = (A::from_c64(*same), A::from_c64(*diff));
            let phase = |i: usize| {
                let odd = ((i & low_mask).count_ones() % 2 == 1) != odd_high;
                if odd { diff } else { same }
            };
            if run < 4 {
                for (i, a) in block.iter_mut().enumerate() {
                    if i & low_controls == low_controls {
                        *a *= phase(i);
                    }
                }
            } else {
                for (j, part) in block.chunks_mut(run).enumerate() {
                    let i = j * run;
                    if i & low_controls == low_controls {
                        A::scale_run(part, phase(i));
                    }
                }
            }
        }
    }
}

/// Apply a run of local operations block by block, blocks in parallel.
fn apply_run<A: Amplitude>(amps: &mut [A], c: usize, run: &[Local<'_>]) {
    let block = 1usize << c;
    let apply = |(k, chunk): (usize, &mut [A])| {
        for local in run {
            apply_local(chunk, k * block, local);
        }
    };
    crate::parallel::for_each_block(amps, block, |k, chunk| apply((k, chunk)));
}

/// Apply `circuit` with parameters `theta` (already validated) to `amps`,
/// fusing gates and blocking for cache with blocks of `2^c` amplitudes.
/// Registers of at most `2^c` amplitudes are simulated gate by gate: they
/// fit in cache already.
pub(crate) fn run_with_block<A: Amplitude>(
    amps: &mut [A],
    circuit: &Circuit,
    theta: &[f64],
    c: usize,
) {
    let n = amps.len().trailing_zeros() as usize;
    if n <= c {
        for inst in circuit.instructions() {
            let angles = Angles::of(inst, theta);
            apply_instruction(amps, inst, angles.get(), false, 0);
        }
        return;
    }
    let ops = fuse(circuit, theta, c);
    // Runs of local operations, as ranges of `ops`; a run of one gains
    // nothing from blocking, and the whole-register kernel streams memory
    // better. Its result is the same up to the sign of zero amplitudes (a
    // controlled phase may go through the general 2×2 kernel there).
    let mut run: Vec<Local<'_>> = Vec::new();
    let mut first = 0;
    let flush = |amps: &mut [A], run: &mut Vec<Local<'_>>, op: &Op<'_>| {
        match run.len() {
            0 => {}
            1 => op.apply_full(amps),
            _ => apply_run(amps, c, run),
        }
        run.clear();
    };
    for (k, op) in ops.iter().enumerate() {
        match local(op, c) {
            Some(l) => {
                if run.is_empty() {
                    first = k;
                }
                run.push(l);
            }
            None => {
                flush(amps, &mut run, &ops[first]);
                op.apply_full(amps);
            }
        }
    }
    if !run.is_empty() {
        flush(amps, &mut run, &ops[first]);
    }
}

/// Call `f(base, block)` for every block of `2^c` amplitudes (`base` is the
/// register index of its first amplitude), blocks in parallel.
pub(crate) fn for_each_block<A, F>(amps: &mut [A], c: usize, f: F)
where
    A: Amplitude,
    F: Fn(usize, &mut [A]) + Sync + Send,
{
    let block = 1usize << c;
    crate::parallel::for_each_block(amps, block, |k, chunk| f(k * block, chunk));
}

/// `Im Σ_i ⟨λ|Y|φ⟩` over pairs of runs (lower runs `l0`, `p0`, upper
/// runs `l1`, `p1`): per pair `Re(conj(λ₁)·φ₀ − conj(λ₀)·φ₁)`, in double
/// precision with four partial sums, combined in a fixed order.
#[inline]
pub(crate) fn y_overlap_im<A: Amplitude>(l0: &[A], l1: &[A], p0: &[A], p1: &[A]) -> f64 {
    let term = |i: usize| {
        let (a0, a1, b0, b1) = (
            l0[i].to_c64(),
            l1[i].to_c64(),
            p0[i].to_c64(),
            p1[i].to_c64(),
        );
        (a1.re * b0.re + a1.im * b0.im) - (a0.re * b1.re + a0.im * b1.im)
    };
    let n = l0.len();
    let mut acc = [0.0f64; 4];
    let full = n - n % 4;
    for i in (0..full).step_by(4) {
        for (k, a) in acc.iter_mut().enumerate() {
            *a += term(i + k);
        }
    }
    let rest: f64 = (full..n).map(term).sum();
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + rest
}

/// log₂ of the contiguous runs of a [`Tiles`] tile, and the most qubits of
/// a group. Measured for QAOA's mixer at 24 qubits (ten qubits above the
/// block), per layer: runs of 2⁷ amplitudes in groups of 7 took 0.117 s,
/// one qubit per pass 0.046 s, runs of 2¹⁰ in groups of 4 0.040 s, runs
/// of 2¹² in groups of 5 0.029 s: long runs keep the prefetchers and the
/// TLB busy with few streams.
#[cfg(not(miri))]
const TILE_RUN_QUBITS: usize = 12;
#[cfg(not(miri))]
const TILE_GROUP: usize = 5;
#[cfg(miri)]
const TILE_RUN_QUBITS: usize = 1;
#[cfg(miri)]
const TILE_GROUP: usize = 2;

// The runs of a tile lie below the block: every qubit a tile pass rotates
// is above them.
const _: () = assert!(TILE_RUN_QUBITS <= BLOCK_QUBITS);

/// Tiles for gates on a group of qubits above the run qubits: a tile
/// holds the 2^g runs of 2^r contiguous amplitudes that differ only in the
/// group's bits, `2^(g + r)` amplitudes in all, so every gate on the group
/// pairs runs inside one tile. Applying the group's gates tile by tile
/// takes one pass over memory instead of one per qubit.
pub(crate) struct Tiles {
    /// The group's qubits, ascending.
    group: Vec<usize>,
    /// Offsets of the runs of a tile (bit j of the run index → group[j]).
    offsets: Vec<usize>,
    /// Bits of the tile index, deposited at these qubits.
    others: Vec<usize>,
    run: usize,
}

impl Tiles {
    /// The tiles for `group` (distinct qubits, all at or above
    /// `TILE_RUN_QUBITS`) on `n` qubits.
    pub(crate) fn new(n: usize, group: &[usize]) -> Self {
        let mut group = group.to_vec();
        group.sort_unstable();
        // A group qubit inside the runs would make two runs of a tile
        // overlap (`apply_tile` relies on this).
        assert!(
            group.iter().all(|&q| q >= TILE_RUN_QUBITS && q < n),
            "tile group {group:?} must lie in {TILE_RUN_QUBITS}..{n}"
        );
        let offsets = (0..1usize << group.len())
            .map(|j| {
                group
                    .iter()
                    .enumerate()
                    .filter(|(b, _)| j >> b & 1 == 1)
                    .map(|(_, q)| 1usize << q)
                    .sum()
            })
            .collect();
        let others = (TILE_RUN_QUBITS..n)
            .filter(|q| !group.contains(q))
            .collect();
        Self {
            group,
            offsets,
            others,
            run: 1 << TILE_RUN_QUBITS,
        }
    }

    /// The qubits `low..n` in groups of up to `TILE_GROUP`.
    pub(crate) fn groups(n: usize, low: usize) -> Vec<Tiles> {
        assert!(
            low >= n || low >= TILE_RUN_QUBITS,
            "qubits below the tiles' runs"
        );
        let qubits: Vec<usize> = (low..n).collect();
        qubits
            .chunks(TILE_GROUP)
            .map(|g| Tiles::new(n, g))
            .collect()
    }

    /// The group's qubits, ascending.
    pub(crate) fn qubits(&self) -> &[usize] {
        &self.group
    }

    fn count(&self) -> usize {
        1 << self.others.len()
    }

    /// First amplitude of tile `t`.
    fn base(&self, t: usize) -> usize {
        self.others
            .iter()
            .enumerate()
            .filter(|(b, _)| t >> b & 1 == 1)
            .map(|(_, q)| 1usize << q)
            .sum()
    }

    /// Bit of the run index for qubit `q` of the group.
    fn bit(&self, q: usize) -> usize {
        1 << self
            .group
            .iter()
            .position(|&g| g == q)
            .expect("qubit of the group")
    }

    /// The gates with the run-index bit of their qubit, and the real parts
    /// of real matrices (applied by the real kernels, as [`apply_1q`]
    /// does).
    fn prepare<A: Amplitude>(&self, gates: &[(usize, [A; 4])]) -> Vec<TileGate<A>> {
        gates
            .iter()
            .map(|(q, m)| TileGate {
                bit: self.bit(*q),
                matrix: *m,
                real: m.iter().all(|&z| z == z.conj()).then(|| m.map(A::re)),
            })
            .collect()
    }

    /// Apply `gates` to the tile at `base` of the vector at `ptr`.
    ///
    /// # Safety
    /// `ptr` points to a vector of the tiles' register, and no other
    /// thread touches this tile.
    unsafe fn apply_tile<A: Amplitude>(&self, ptr: *mut A, base: usize, gates: &[TileGate<A>]) {
        for g in gates {
            for (j, &lo) in self.offsets.iter().enumerate() {
                if j & g.bit != 0 {
                    continue;
                }
                let hi = self.offsets[j | g.bit];
                // SAFETY: `lo` and `hi` are distinct runs of `run`
                // amplitudes of this tile (by the caller, inside the
                // vector and not shared).
                let (a, b) = unsafe {
                    (
                        std::slice::from_raw_parts_mut(ptr.add(base + lo), self.run),
                        std::slice::from_raw_parts_mut(ptr.add(base + hi), self.run),
                    )
                };
                match &g.real {
                    Some(r) => A::mat2_runs_real(a, b, r),
                    None => A::mat2_runs(a, b, &g.matrix),
                }
            }
        }
    }

    /// `Im Σ_{q in group} ⟨λ|Y_q|φ⟩` over the tile at `base`.
    ///
    /// # Safety
    /// Both pointers point to vectors of the tiles' register, and no
    /// other thread writes this tile.
    unsafe fn tile_y_overlap<A: Amplitude>(
        &self,
        lambda: *const A,
        phi: *const A,
        base: usize,
    ) -> f64 {
        let run = |p: *const A, at: usize| {
            // SAFETY: a run of this tile, inside the vector (by the
            // caller), not written meanwhile.
            unsafe { std::slice::from_raw_parts(p.add(at), self.run) }
        };
        let mut acc = 0.0;
        for b in 0..self.group.len() {
            let bit = 1 << b;
            for (j, &lo) in self.offsets.iter().enumerate() {
                if j & bit != 0 {
                    continue;
                }
                let (lo, hi) = (base + lo, base + self.offsets[j | bit]);
                acc += y_overlap_im(run(lambda, lo), run(lambda, hi), run(phi, lo), run(phi, hi));
            }
        }
        acc
    }

    /// Apply the single-qubit `gates` (qubit, matrix), all on the group's
    /// qubits, in order: per amplitude the same arithmetic as gate after
    /// gate.
    pub(crate) fn apply_1q<A: Amplitude>(&self, amps: &mut [A], gates: &[(usize, [A; 4])]) {
        let gates = self.prepare(gates);
        let ptr = crate::parallel::SyncPtr::new(amps);
        crate::parallel::for_each_item(self.count(), amps.len(), 1, |t| {
            // SAFETY: distinct tiles are disjoint.
            unsafe { self.apply_tile(ptr.get(), self.base(t), &gates) }
        });
    }

    /// `Im Σ_{q in group} ⟨λ|Y_q|φ⟩`, summed tile by tile in tile order (so
    /// independent of the thread count).
    #[cfg(test)]
    pub(crate) fn y_overlap<A: Amplitude>(&self, lambda: &[A], phi: &[A]) -> f64 {
        crate::parallel::sum_chunks_f64(self.count(), phi.len(), |t| {
            // SAFETY: reads only, inside both vectors.
            unsafe { self.tile_y_overlap(lambda.as_ptr(), phi.as_ptr(), self.base(t)) }
        })
    }

    /// `Im Σ_{q in group} ⟨λ|Y_q|φ⟩`, then `gates` applied to both,
    /// tile by tile in one pass over the two vectors. (A gate commuting
    /// with `Σ Y` leaves the overlap unchanged: QAOA's mixer.)
    pub(crate) fn y_overlap_then_apply<A: Amplitude>(
        &self,
        lambda: &mut [A],
        phi: &mut [A],
        gates: &[(usize, [A; 4])],
    ) -> f64 {
        let gates = self.prepare(gates);
        let work = phi.len();
        let (l, p) = (
            crate::parallel::SyncPtr::new(lambda),
            crate::parallel::SyncPtr::new(phi),
        );
        crate::parallel::sum_chunks_f64(self.count(), work, |t| {
            let base = self.base(t);
            // SAFETY: distinct tiles are disjoint; each tile is read and
            // then written by this task only.
            unsafe {
                let overlap = self.tile_y_overlap(l.get(), p.get(), base);
                self.apply_tile(l.get(), base, &gates);
                self.apply_tile(p.get(), base, &gates);
                overlap
            }
        })
    }
}

/// A gate of a [`Tiles`] pass.
struct TileGate<A: Amplitude> {
    bit: usize,
    matrix: [A; 4],
    real: Option<[A::Real; 4]>,
}

/// Apply `circuit` with parameters `theta` (already validated) to `amps`.
pub(crate) fn run<A: Amplitude>(amps: &mut [A], circuit: &Circuit, theta: &[f64]) {
    run_with_block(amps, circuit, theta, BLOCK_QUBITS);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::{Param, StateVector};
    use crate::rng::Rng;

    /// A random circuit over the whole gate set (every kind, unitaries on
    /// one to three targets, symbolic angles), with random controls, and
    /// parameters to bind it with.
    fn random_circuit(n: usize, gates: usize, rng: &mut Rng) -> (Circuit, Vec<f64>) {
        let mut c = Circuit::new(n);
        let pick = |rng: &mut Rng, k: usize| -> Vec<usize> {
            let mut q: Vec<usize> = (0..n).collect();
            rng.shuffle(&mut q);
            q.truncate(k);
            q
        };
        let random_unitary = |rng: &mut Rng, k: usize| -> Vec<C64> {
            let d = 1 << k;
            let m =
                crate::linalg::CMatrix::from_fn(d, d, |_, _| C64::new(rng.normal(), rng.normal()));
            let u = m.qr().q();
            (0..d * d).map(|i| u[(i / d, i % d)]).collect()
        };
        use GateKind::*;
        let one = [H, X, Y, Z, S, Sdg, T, Tdg, SX, SXdg, RX, RY, RZ, P, U];
        let two = [Swap, RXX, RYY, RZZ];
        for _ in 0..gates {
            let a = rng.uniform_range(-3.0, 3.0);
            // Half of the angles are symbolic: 0.7·p + offset.
            let angle = |c: &mut Circuit, rng: &mut Rng| -> Param {
                if rng.uniform() < 0.5 {
                    (c.new_parameter() * 0.7 + a).into()
                } else {
                    a.into()
                }
            };
            let controls = if n > 3 && rng.uniform() < 0.3 {
                1 + rng.below(2) as usize
            } else {
                0
            };
            let choice = if n < 2 { 0 } else { rng.below(4) };
            match choice {
                0 | 1 => {
                    let kind = one[rng.below(one.len() as u64) as usize];
                    let q = pick(rng, (1 + controls).min(n));
                    let params: Vec<Param> =
                        (0..kind.num_params()).map(|_| angle(&mut c, rng)).collect();
                    c.controlled(kind, &[q[0]], &q[1..], &params).unwrap();
                }
                2 => {
                    let kind = two[rng.below(two.len() as u64) as usize];
                    let q = pick(rng, (2 + controls).min(n));
                    let params: Vec<Param> =
                        (0..kind.num_params()).map(|_| angle(&mut c, rng)).collect();
                    c.controlled(kind, &[q[0], q[1]], &q[2..], &params).unwrap();
                }
                _ => {
                    let k = 1 + rng.below(3.min(n as u64)) as usize;
                    let q = pick(rng, (k + controls).min(n));
                    let u = random_unitary(rng, k);
                    c.unitary(u, &q[..k], &q[k..]).unwrap();
                }
            }
            if rng.uniform() < 0.1 {
                // A run to fuse.
                let t = rng.below(n as u64) as usize;
                c.h(t).rz(t, a).t(t).s(t).ry(t, 0.3 * a).sx(t);
            }
        }
        let theta = (0..c.num_parameters())
            .map(|_| rng.uniform_range(-3.0, 3.0))
            .collect();
        (c, theta)
    }

    fn check(n: usize, block: usize, rng: &mut Rng) {
        let (circuit, theta) = random_circuit(n, 60, rng);
        let psi = StateVector::random(n, rng);
        let mut reference = psi.amplitudes().to_vec();
        for inst in circuit.instructions() {
            apply_instruction(&mut reference, inst, &inst.angles(&theta), false, 0);
        }
        let mut planned = psi.amplitudes().to_vec();
        run_with_block(&mut planned, &circuit, &theta, block);
        for (a, b) in planned.iter().zip(&reference) {
            assert!((a - b).norm() < 1e-12, "n={n} block={block}");
        }
    }

    /// Tiles apply a group's gates as gate after gate does, bit for bit,
    /// and sum the Y overlaps of the group; for groups of every size, also
    /// with qubits left out between them.
    #[test]
    fn tiles_match_gate_after_gate() {
        use num_complex::Complex32 as C32;
        let mut rng = Rng::seed_from_u64(1280);
        let n = TILE_RUN_QUBITS + 5;
        let random = |rng: &mut Rng| -> Vec<C64> {
            (0..1usize << n)
                .map(|_| C64::new(rng.normal(), rng.normal()))
                .collect()
        };
        let groups: Vec<Vec<usize>> = vec![
            vec![TILE_RUN_QUBITS],
            vec![n - 1],
            (TILE_RUN_QUBITS..n).collect(),
            vec![TILE_RUN_QUBITS + 1, n - 2, n - 1],
            vec![TILE_RUN_QUBITS, TILE_RUN_QUBITS + 2, n - 1],
        ];
        for group in groups {
            let tiles = Tiles::new(n, &group);
            let gates: Vec<(usize, [C64; 4])> = group
                .iter()
                .rev()
                .map(|&q| {
                    // Real matrices (the real kernels) on odd qubits.
                    let im = if q % 2 == 1 { 0.0 } else { 1.0 };
                    let m = [0, 1, 2, 3].map(|_| C64::new(rng.normal(), im * rng.normal()));
                    (q, m)
                })
                .collect();
            let start = random(&mut rng);
            let mut tiled = start.clone();
            tiles.apply_1q(&mut tiled, &gates);
            let mut plain = start.clone();
            for (q, m) in &gates {
                apply_1q(&mut plain, *q, 0, m);
            }
            assert_eq!(tiled, plain, "{group:?}");
            // The same in single precision.
            let start32: Vec<C32> = start.iter().map(|z| C32::from_c64(*z)).collect();
            let gates32: Vec<(usize, [C32; 4])> = gates
                .iter()
                .map(|(q, m)| (*q, m.map(C32::from_c64)))
                .collect();
            let mut tiled32 = start32.clone();
            tiles.apply_1q(&mut tiled32, &gates32);
            let mut plain32 = start32;
            for (q, m) in &gates32 {
                apply_1q(&mut plain32, *q, 0, m);
            }
            assert_eq!(tiled32, plain32, "{group:?}");
            // Im Σ_q ⟨λ|Y_q|φ⟩ directly: (Yφ)_b = ±i·φ_{b⊕q}, + where bit q is 1.
            let lambda = random(&mut rng);
            let direct: C64 = group
                .iter()
                .map(|&q| {
                    (0..start.len())
                        .map(|b| {
                            let sign = if b >> q & 1 == 1 { 1.0 } else { -1.0 };
                            lambda[b].conj() * C64::new(0.0, sign) * start[b ^ (1 << q)]
                        })
                        .sum::<C64>()
                })
                .sum();
            let got = tiles.y_overlap(&lambda, &start);
            assert!(
                (got - direct.im).abs() < 1e-9 * direct.norm().max(1.0),
                "{got} {direct}"
            );
            // In one pass with the gates applied to both vectors.
            let (mut l2, mut p2) = (lambda.clone(), start.clone());
            let fused = tiles.y_overlap_then_apply(&mut l2, &mut p2, &gates);
            assert_eq!(fused, got);
            assert_eq!(p2, plain);
            let mut l3 = lambda.clone();
            tiles.apply_1q(&mut l3, &gates);
            assert_eq!(l2, l3);
        }
        // Groups cover the qubits above the block, a block's worth each.
        let groups = Tiles::groups(BLOCK_QUBITS + 9, BLOCK_QUBITS);
        let covered: Vec<usize> = groups.iter().flat_map(|t| t.qubits().to_vec()).collect();
        assert_eq!(
            covered,
            (BLOCK_QUBITS..BLOCK_QUBITS + 9).collect::<Vec<_>>()
        );
        assert!(groups.iter().all(|t| t.qubits().len() <= TILE_GROUP));
    }

    #[test]
    fn plans_match_gate_by_gate_simulation() {
        let mut rng = Rng::seed_from_u64(17);
        let max_n = if cfg!(miri) { 5 } else { 9 };
        for n in 1..=max_n {
            for block in [1, 2, 3, 5, 14] {
                check(n, block, &mut rng);
            }
        }
        // The production block size on registers just above it.
        if !cfg!(miri) {
            for n in [15, 16] {
                for _ in 0..3 {
                    check(n, BLOCK_QUBITS, &mut rng);
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
        // A fused diagonal run outside the block remains a block-local phase.
        let mut d = Circuit::new(3);
        d.t(2).s(2);
        let ops = fuse(&d, &[], 1);
        assert!(matches!(ops[..], [Op::Fused { qubit: 2, .. }]));
        assert!(matches!(local(&ops[0], 1), Some(Local::Bit { .. })));
        // Qubit 0 inside the block (c = 1): its gates stay separate.
        assert_eq!(
            describe(&fuse(&c, &[], 1)),
            ["h", "rz", "ry", "x", "x", "fused1"]
        );
    }
}
