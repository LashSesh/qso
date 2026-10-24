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
use crate::parallel::{CHUNK, SyncPtr, for_each_index, for_each_item, for_each_mut, for_each_pair};

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
        // Contiguous, vectorisable fast paths for uncontrolled gates:
        // diagonal matrices (RZ) scale each half without branching, real
        // matrices (H, RY) need two real products per amplitude instead of
        // two complex ones.
        if m[1] == C64::new(0.0, 0.0) && m[2] == C64::new(0.0, 0.0) {
            let (d0, d1) = (m[0], m[3]);
            for_each_pair(amps, 1 << target, move |a, b| {
                *a *= d0;
                *b *= d1;
            });
            return;
        }
        if m.iter().all(|z| z.im == 0.0) {
            let [a00, a01, a10, a11] = m.map(|z| z.re);
            for_each_pair(amps, 1 << target, move |a, b| {
                let (x, y) = (*a, *b);
                *a = x * a00 + y * a01;
                *b = x * a10 + y * a11;
            });
            return;
        }
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
    if positions.len() == 2 {
        apply_cx(amps, controls, target, &positions);
        return;
    }
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

/// Singly controlled X (`positions` = sorted `[control, target]`, already
/// validated). The amplitudes to exchange form contiguous runs of length
/// `2^min(control, target)`: run `r` starts at `insert_zeros(r·L) | control`
/// and is swapped with the run `2^target` further on. Long runs are split
/// into pieces of at most `CHUNK` elements so that gates on high qubits
/// parallelise as well.
fn apply_cx(amps: &mut [C64], control_mask: usize, target: usize, positions: &[usize]) {
    let run = 1usize << positions[0];
    let piece = run.min(CHUNK);
    let pieces_per_run = run / piece;
    let runs = amps.len() >> (positions.len() + positions[0]);
    let shift = 1usize << target;
    let len = amps.len();
    let ptr = SyncPtr::new(amps);
    for_each_item(runs * pieces_per_run, len, CHUNK / piece, move |w| {
        let (r, k) = (w / pieces_per_run, w % pieces_per_run);
        let start = (insert_zeros(r << positions[0], positions) | control_mask) + k * piece;
        // SAFETY: the start index has the control bit set and the target bit
        // clear, both ranges lie inside one run of the register (positions
        // were validated), and distinct `w` address disjoint ranges.
        unsafe {
            let p = ptr.get();
            std::ptr::swap_nonoverlapping(p.add(start), p.add(start + shift), piece);
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
    if mask.is_power_of_two() {
        // Uncontrolled phase gate: scale the upper half of every block.
        assert!(mask < amps.len(), "qubit outside the register");
        for_each_pair(amps, mask, move |_, b| *b *= phase);
        return;
    }
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
            let m = matrix_1q(GateKind::RZ, angles).map(cj);
            apply_1q(amps, t[0], 0, &m);
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

    /// Reference: X through the generic controlled 2×2 kernel.
    fn reference_cx(amps: &mut [C64], c: usize, t: usize) {
        let x = [
            C64::new(0.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
        ];
        apply_1q(amps, t, 1 << c, &x);
    }

    fn random_state(n: usize, seed: u64) -> Vec<C64> {
        let mut rng = crate::Rng::seed_from_u64(seed);
        (0..1usize << n)
            .map(|_| C64::new(rng.normal(), rng.normal()))
            .collect()
    }

    /// Textbook application of a dense `2^k × 2^k` matrix to `targets`
    /// (`targets[0]` most significant) on basis states with all `controls`
    /// set: safe indexing only, independent of the kernels under test.
    fn reference(amps: &[C64], targets: &[usize], controls: usize, m: &[C64]) -> Vec<C64> {
        let k = targets.len();
        let local = |b: usize| (0..k).fold(0, |acc, j| (acc << 1) | ((b >> targets[j]) & 1));
        let with_local = |b: usize, l: usize| {
            (0..k).fold(b, |acc, j| {
                let bit = 1 << targets[j];
                if (l >> (k - 1 - j)) & 1 == 1 {
                    acc | bit
                } else {
                    acc & !bit
                }
            })
        };
        (0..amps.len())
            .map(|b| {
                if b & controls != controls {
                    return amps[b];
                }
                let r = local(b);
                (0..1 << k)
                    .map(|c| m[r * (1 << k) + c] * amps[with_local(b, c)])
                    .sum()
            })
            .collect()
    }

    fn random_matrix(k: usize, rng: &mut crate::Rng) -> Vec<C64> {
        (0..1 << (2 * k))
            .map(|_| C64::new(rng.normal(), rng.normal()))
            .collect()
    }

    fn assert_close(a: &[C64], b: &[C64], what: &str) {
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).norm() < 1e-12, "{what}: amplitude {i}: {x} vs {y}");
        }
    }

    /// Run `f` on several worker threads, so that the parallel paths of
    /// the kernels really execute concurrently (Miri then checks them for
    /// data races as well as for out-of-bounds and aliasing errors).
    fn on_threads(f: impl FnOnce() + Send) {
        #[cfg(feature = "parallel")]
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(f);
        #[cfg(not(feature = "parallel"))]
        f();
    }

    /// Every `unsafe` kernel against [`reference`] on 3 qubits (serial) and
    /// 7 qubits (parallel under Miri, whose thresholds are lowered so that
    /// 2⁷ amplitudes span several work chunks).
    #[test]
    fn unsafe_kernels_match_the_reference() {
        on_threads(|| {
            let mut rng = crate::Rng::seed_from_u64(2024);
            for n in [3usize, 7] {
                let state = random_state(n, 7 + n as u64);
                let others = |skip: &[usize]| -> Vec<usize> {
                    (0..n).filter(|q| !skip.contains(q)).collect()
                };
                let control_masks = |skip: &[usize], rng: &mut crate::Rng| -> Vec<usize> {
                    let free = others(skip);
                    let pick = |rng: &mut crate::Rng| free[rng.below(free.len() as u64) as usize];
                    let mut masks = vec![0];
                    if !free.is_empty() {
                        masks.push(1 << pick(rng));
                        masks.push((1 << pick(rng)) | (1 << pick(rng)));
                    }
                    masks
                };
                for t in 0..n {
                    let m = random_matrix(1, &mut rng);
                    let m1: [C64; 4] = m.clone().try_into().unwrap();
                    let x = [0.0, 1.0, 1.0, 0.0].map(|v| C64::new(v, 0.0));
                    for c in control_masks(&[t], &mut rng) {
                        let mut a = state.clone();
                        apply_1q(&mut a, t, c, &m1);
                        assert_close(&a, &reference(&state, &[t], c, &m), "apply_1q");
                        let mut a = state.clone();
                        apply_x(&mut a, t, c);
                        assert_close(&a, &reference(&state, &[t], c, &x), "apply_x");
                        let phase = C64::from_polar(1.0, 0.3 + t as f64);
                        let diag = [
                            C64::new(1.0, 0.0),
                            C64::new(0.0, 0.0),
                            C64::new(0.0, 0.0),
                            phase,
                        ];
                        let mut a = state.clone();
                        apply_phase_on_mask(&mut a, c | (1 << t), phase);
                        assert_close(
                            &a,
                            &reference(&state, &[t], c, &diag),
                            "apply_phase_on_mask",
                        );
                    }
                }
                for f in 0..n {
                    for s in others(&[f]) {
                        let m = random_matrix(2, &mut rng);
                        let m2: [C64; 16] = m.clone().try_into().unwrap();
                        let mut swap = vec![C64::new(0.0, 0.0); 16];
                        for (r, c) in [(0, 0), (1, 2), (2, 1), (3, 3)] {
                            swap[4 * r + c] = C64::new(1.0, 0.0);
                        }
                        for c in control_masks(&[f, s], &mut rng) {
                            let mut a = state.clone();
                            apply_2q(&mut a, f, s, c, &m2);
                            assert_close(&a, &reference(&state, &[f, s], c, &m), "apply_2q");
                            let mut a = state.clone();
                            apply_swap(&mut a, f, s, c);
                            assert_close(&a, &reference(&state, &[f, s], c, &swap), "apply_swap");
                        }
                    }
                }
                for _ in 0..6 {
                    let mut targets: Vec<usize> = Vec::new();
                    while targets.len() < 3 {
                        let q = rng.below(n as u64) as usize;
                        if !targets.contains(&q) {
                            targets.push(q);
                        }
                    }
                    let m = random_matrix(3, &mut rng);
                    for c in control_masks(&targets, &mut rng) {
                        let mut a = state.clone();
                        apply_kq(&mut a, &targets, c, &m);
                        assert_close(&a, &reference(&state, &targets, c, &m), "apply_kq");
                    }
                }
            }
        });
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "2¹⁷ amplitudes; covered by unsafe_kernels_match_the_reference"
    )]
    fn block_cx_matches_the_generic_kernel() {
        // n = 17 crosses the parallel threshold and makes runs longer than
        // CHUNK for high qubits.
        for n in [3usize, 6, 17] {
            let state = random_state(n, n as u64);
            for c in 0..n {
                for t in 0..n {
                    if c == t || (n == 17 && (c + t) % 4 != 0 && c.max(t) < 14) {
                        continue;
                    }
                    let mut fast = state.clone();
                    let mut slow = state.clone();
                    apply_x(&mut fast, t, 1 << c);
                    reference_cx(&mut slow, c, t);
                    assert_eq!(fast, slow, "n = {n}, control {c}, target {t}");
                }
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "2¹⁷ amplitudes; safe code")]
    fn real_matrix_path_matches_the_complex_kernel() {
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let real = [
            C64::new(h, 0.0),
            C64::new(h, 0.0),
            C64::new(h, 0.0),
            C64::new(-h, 0.0),
        ];
        for n in [4usize, 17] {
            let state = random_state(n, 99);
            for t in 0..n {
                let mut fast = state.clone();
                apply_1q(&mut fast, t, 0, &real);
                // Reference: the textbook loop over amplitude pairs.
                let mut slow = state.clone();
                let bit = 1usize << t;
                for i in 0..slow.len() {
                    if i & bit == 0 {
                        let (a, b) = (state[i], state[i | bit]);
                        slow[i] = real[0] * a + real[1] * b;
                        slow[i | bit] = real[2] * a + real[3] * b;
                    }
                }
                for (x, y) in fast.iter().zip(&slow) {
                    assert!((x - y).norm() < 1e-14);
                }
            }
        }
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
