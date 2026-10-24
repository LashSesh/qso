//! Low-level amplitude kernels shared by the statevector and density-matrix
//! simulators.
//!
//! Index convention: basis index `b = Σ_q bit_q · 2^q` (qubit 0 is the least
//! significant bit). Matrices passed to the kernels are row-major. For a
//! two-qubit matrix acting on `(first, second)` the local basis order is
//! `|x_first x_second⟩`, i.e. `first` is the *more* significant local bit —
//! the textbook convention for `CX(control, target)`.
//!
//! The kernels do not require unitarity: the same code applies derivative
//! matrices (adjoint differentiation) and Kraus superoperators (noise).

use num_complex::Complex64 as C64;

use super::circuit::Instruction;
use super::gates::{GateKind, dmatrix_1q, dmatrix_2q, matrix_1q, matrix_2q};
use crate::parallel::{SyncPtr, for_each_index, for_each_mut, for_each_pair};

/// Insert a zero bit at position `pos` of `k`.
#[inline(always)]
fn insert_zero(k: usize, pos: usize) -> usize {
    let low = k & ((1usize << pos) - 1);
    ((k ^ low) << 1) | low
}

/// Insert zero bits at all positions listed in `sorted` (ascending).
#[inline(always)]
fn insert_zeros(mut k: usize, sorted: &[usize]) -> usize {
    for &p in sorted {
        k = insert_zero(k, p);
    }
    k
}

/// Sorted bit positions touched by a gate.
///
/// # Panics
/// If a position repeats or lies outside a register of `len` amplitudes.
/// The `unsafe` kernels rely on this check (once per gate) for soundness:
/// it guarantees that distinct loop indices address disjoint amplitudes
/// that are in bounds, whatever the caller passed in.
fn sorted_positions(qubits: &[usize], controls: usize, len: usize) -> Vec<usize> {
    let mut pos: Vec<usize> = qubits.to_vec();
    let mut c = controls;
    while c != 0 {
        let p = c.trailing_zeros() as usize;
        pos.push(p);
        c &= c - 1;
    }
    pos.sort_unstable();
    let n_bits = len.trailing_zeros() as usize;
    assert!(
        len.is_power_of_two(),
        "amplitude count must be a power of two"
    );
    assert!(
        pos.windows(2).all(|w| w[0] != w[1]) && pos.last().is_none_or(|&p| p < n_bits),
        "gate qubits {pos:?} must be distinct and below {n_bits}"
    );
    pos
}

/// Apply a 2×2 matrix to `target`, restricted to basis states whose bits in
/// `controls` (a bit mask) are all 1.
pub(crate) fn apply_1q(amps: &mut [C64], target: usize, controls: usize, m: &[C64; 4]) {
    debug_assert!(amps.len().is_power_of_two());
    debug_assert_eq!(controls & (1 << target), 0);
    let m = *m;
    if controls == 0 {
        assert!(
            1usize << target < amps.len(),
            "target {target} outside the register"
        );
        // Contiguous, vectorisable fast path for uncontrolled gates.
        for_each_pair(amps, 1 << target, move |a, b| {
            let (x, y) = (*a, *b);
            *a = m[0] * x + m[1] * y;
            *b = m[2] * x + m[3] * y;
        });
        return;
    }
    let positions = sorted_positions(&[target], controls, amps.len());
    let count = amps.len() >> positions.len();
    let bit = 1usize << target;
    let ptr = SyncPtr::new(amps);
    for_each_index(count, move |k| {
        let i0 = insert_zeros(k, &positions) | controls;
        let i1 = i0 | bit;
        // SAFETY: distinct `k` map to disjoint index pairs (i0, i1).
        unsafe {
            let p = ptr.get();
            let a = *p.add(i0);
            let b = *p.add(i1);
            *p.add(i0) = m[0] * a + m[1] * b;
            *p.add(i1) = m[2] * a + m[3] * b;
        }
    });
}

/// Apply a 4×4 matrix to `(first, second)` restricted to `controls`.
pub(crate) fn apply_2q(
    amps: &mut [C64],
    first: usize,
    second: usize,
    controls: usize,
    m: &[C64; 16],
) {
    debug_assert!(amps.len().is_power_of_two());
    debug_assert_ne!(first, second);
    let positions = sorted_positions(&[first, second], controls, amps.len());
    let count = amps.len() >> positions.len();
    let bf = 1usize << first;
    let bs = 1usize << second;
    let ptr = SyncPtr::new(amps);
    let m = *m;
    for_each_index(count, move |k| {
        let base = insert_zeros(k, &positions) | controls;
        // Local order |x_first x_second⟩ → local index 2·x_first + x_second.
        let idx = [base, base | bs, base | bf, base | bf | bs];
        // SAFETY: distinct `k` map to disjoint index quadruples.
        unsafe {
            let p = ptr.get();
            let v = [
                *p.add(idx[0]),
                *p.add(idx[1]),
                *p.add(idx[2]),
                *p.add(idx[3]),
            ];
            for r in 0..4 {
                let row = &m[4 * r..4 * r + 4];
                *p.add(idx[r]) = row[0] * v[0] + row[1] * v[1] + row[2] * v[2] + row[3] * v[3];
            }
        }
    });
}

/// Pauli X on `target` restricted to `controls` (a pure permutation).
pub(crate) fn apply_x(amps: &mut [C64], target: usize, controls: usize) {
    if controls == 0 {
        assert!(
            1usize << target < amps.len(),
            "target {target} outside the register"
        );
        for_each_pair(amps, 1 << target, std::mem::swap);
        return;
    }
    let positions = sorted_positions(&[target], controls, amps.len());
    let count = amps.len() >> positions.len();
    let bit = 1usize << target;
    let ptr = SyncPtr::new(amps);
    for_each_index(count, move |k| {
        let i0 = insert_zeros(k, &positions) | controls;
        // SAFETY: distinct `k` map to disjoint index pairs.
        unsafe {
            let p = ptr.get();
            std::ptr::swap(p.add(i0), p.add(i0 | bit));
        }
    });
}

/// Swap qubits `a` and `b` on the subspace selected by `controls`.
pub(crate) fn apply_swap(amps: &mut [C64], a: usize, b: usize, controls: usize) {
    let positions = sorted_positions(&[a, b], controls, amps.len());
    let count = amps.len() >> positions.len();
    let ba = 1usize << a;
    let bb = 1usize << b;
    let ptr = SyncPtr::new(amps);
    for_each_index(count, move |k| {
        let base = insert_zeros(k, &positions) | controls;
        // SAFETY: distinct `k` map to disjoint index pairs.
        unsafe {
            let p = ptr.get();
            std::ptr::swap(p.add(base | ba), p.add(base | bb));
        }
    });
}

/// Largest number of targets supported by [`apply_kq`].
pub(crate) const MAX_DENSE_TARGETS: usize = 6;

/// Apply a dense `2^k × 2^k` matrix to `targets` (`targets[0]` is the most
/// significant local bit) restricted to `controls`.
pub(crate) fn apply_kq(amps: &mut [C64], targets: &[usize], controls: usize, m: &[C64]) {
    let k = targets.len();
    assert!(
        k <= MAX_DENSE_TARGETS,
        "at most {MAX_DENSE_TARGETS} targets"
    );
    let d = 1usize << k;
    debug_assert_eq!(m.len(), d * d);
    let positions = sorted_positions(targets, controls, amps.len());
    let count = amps.len() >> positions.len();
    let offsets: Vec<usize> = (0..d)
        .map(|l| {
            (0..k).fold(0usize, |acc, j| {
                if (l >> (k - 1 - j)) & 1 == 1 {
                    acc | (1 << targets[j])
                } else {
                    acc
                }
            })
        })
        .collect();
    let ptr = SyncPtr::new(amps);
    for_each_index(count, move |kk| {
        let base = insert_zeros(kk, &positions) | controls;
        let mut v = [C64::new(0.0, 0.0); 1 << MAX_DENSE_TARGETS];
        // SAFETY: distinct `kk` map to disjoint index sets.
        unsafe {
            let p = ptr.get();
            for (l, off) in offsets.iter().enumerate() {
                v[l] = *p.add(base | off);
            }
            for (r, off) in offsets.iter().enumerate() {
                let row = &m[r * d..(r + 1) * d];
                let mut acc = C64::new(0.0, 0.0);
                for c in 0..d {
                    acc += row[c] * v[c];
                }
                *p.add(base | off) = acc;
            }
        }
    });
}

/// Multiply every amplitude by `phase(index)`.
pub(crate) fn apply_diagonal<F>(amps: &mut [C64], phase: F)
where
    F: Fn(usize) -> C64 + Sync + Send,
{
    for_each_mut(amps, |i, a| *a *= phase(i));
}

/// Multiply amplitudes whose `mask` bits are all set by `phase`.
pub(crate) fn apply_phase_on_mask(amps: &mut [C64], mask: usize, phase: C64) {
    let positions = sorted_positions(&[], mask, amps.len());
    let count = amps.len() >> positions.len();
    let ptr = SyncPtr::new(amps);
    for_each_index(count, move |k| {
        let i = insert_zeros(k, &positions) | mask;
        // SAFETY: distinct `k` map to distinct indices.
        unsafe {
            *ptr.get().add(i) *= phase;
        }
    });
}

/// Apply one circuit instruction with concrete `angles` to raw amplitudes.
///
/// `offset` shifts every qubit index and `conj` applies the complex
/// conjugate matrix; together they let the density-matrix simulator apply
/// `U ⊗ U*` to a vectorised ρ with the same kernels.
pub(crate) fn apply_instruction(
    amps: &mut [C64],
    inst: &Instruction,
    angles: &[f64],
    conj: bool,
    offset: usize,
) {
    let controls = inst
        .controls
        .iter()
        .fold(0usize, |m, &c| m | (1 << (c + offset)));
    let t: Vec<usize> = inst.targets.iter().map(|&q| q + offset).collect();
    let cj = |z: C64| if conj { z.conj() } else { z };
    match inst.kind {
        GateKind::Swap => apply_swap(amps, t[0], t[1], controls),
        GateKind::X => apply_x(amps, t[0], controls),
        GateKind::Z | GateKind::S | GateKind::Sdg | GateKind::T | GateKind::Tdg | GateKind::P => {
            let m = matrix_1q(inst.kind, angles);
            apply_phase_on_mask(amps, controls | (1 << t[0]), cj(m[3]));
        }
        GateKind::RZ if controls == 0 => {
            let m = matrix_1q(GateKind::RZ, angles);
            let (p0, p1) = (cj(m[0]), cj(m[3]));
            let bit = t[0];
            apply_diagonal(amps, move |i| if (i >> bit) & 1 == 1 { p1 } else { p0 });
        }
        GateKind::RZZ => {
            let half = angles[0] / 2.0;
            let same = cj(C64::from_polar(1.0, -half));
            let diff = cj(C64::from_polar(1.0, half));
            let mask = (1usize << t[0]) | (1usize << t[1]);
            apply_diagonal(amps, move |i| {
                if i & controls != controls {
                    C64::new(1.0, 0.0)
                } else if (i & mask).count_ones().is_multiple_of(2) {
                    same
                } else {
                    diff
                }
            });
        }
        GateKind::Unitary => {
            let m = inst
                .matrix
                .as_ref()
                .expect("unitary instruction carries a matrix");
            let m: Vec<C64> = m.iter().map(|&z| cj(z)).collect();
            match t.len() {
                1 => apply_1q(amps, t[0], controls, &m[..].try_into().expect("2x2")),
                2 => apply_2q(amps, t[0], t[1], controls, &m[..].try_into().expect("4x4")),
                _ => apply_kq(amps, &t, controls, &m),
            }
        }
        kind if kind.num_targets() == Some(1) => {
            let m = matrix_1q(kind, angles).map(cj);
            apply_1q(amps, t[0], controls, &m);
        }
        kind => {
            let m = matrix_2q(kind, angles).map(cj);
            apply_2q(amps, t[0], t[1], controls, &m);
        }
    }
}

/// Apply `∂U/∂angle_which` (with the control projector) to `amps`.
///
/// The derivative of a controlled gate vanishes on the uncontrolled
/// subspace, so those amplitudes are zeroed.
pub(crate) fn apply_instruction_derivative(
    amps: &mut [C64],
    inst: &Instruction,
    angles: &[f64],
    which: usize,
) {
    let controls = inst.control_mask();
    match inst.kind.num_targets() {
        Some(1) => apply_1q(
            amps,
            inst.targets[0],
            controls,
            &dmatrix_1q(inst.kind, angles, which),
        ),
        Some(2) => apply_2q(
            amps,
            inst.targets[0],
            inst.targets[1],
            controls,
            &dmatrix_2q(inst.kind, angles),
        ),
        _ => unreachable!("only fixed-arity parametrised gates have derivatives"),
    }
    if controls != 0 {
        for_each_mut(amps, |i, a| {
            if i & controls != controls {
                *a = C64::new(0.0, 0.0);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_zero_places_bits() {
        assert_eq!(insert_zero(0b111, 0), 0b1110);
        assert_eq!(insert_zero(0b111, 1), 0b1101);
        assert_eq!(insert_zero(0b111, 3), 0b0111);
        assert_eq!(insert_zeros(0b11, &[0, 2]), 0b1010);
    }

    #[test]
    fn controlled_x_flips_only_when_control_set() {
        // |q1 q0⟩ = |1 0⟩ (index 2) with control q1, target q0 → |1 1⟩.
        let x = [
            C64::new(0.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
        ];
        let mut amps = vec![C64::new(0.0, 0.0); 4];
        amps[2] = C64::new(1.0, 0.0);
        apply_1q(&mut amps, 0, 0b10, &x);
        assert_eq!(amps[3], C64::new(1.0, 0.0));
        let mut amps = vec![C64::new(0.0, 0.0); 4];
        amps[0] = C64::new(1.0, 0.0);
        apply_1q(&mut amps, 0, 0b10, &x);
        assert_eq!(amps[0], C64::new(1.0, 0.0));
    }
}
