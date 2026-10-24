//! Multi-controlled gates in terms of singly controlled gates and Toffolis,
//! without ancilla qubits (Barenco et al., Phys. Rev. A 52, 3457, 1995).
//!
//! * `C^k(X)` with `k − 2` idle ("dirty") qubits: `4(k − 2)` Toffolis
//!   (Lemma 7.2); with a single idle qubit, two such halves (Lemma 7.3 and
//!   Corollary 7.4), `O(k)` Toffolis.
//! * `C^k(U)` for a single-qubit unitary: `C(V)·C^{k−1}(X)·C(V†)·C^{k−1}(X)·
//!   C^{k−1}(V)` with `V² = U` (Lemma 7.5), where the target is the idle
//!   qubit of the `C^{k−1}(X)` gates: `O(k²)` gates in total.
//!
//! Every construction is exact, phases included, so the decomposed gates
//! can themselves be controlled or exported.

use std::sync::Arc;

use num_complex::Complex64 as C64;

use super::circuit::Instruction;
use super::gates::GateKind;

fn gate(kind: GateKind, target: usize, controls: &[usize]) -> Instruction {
    Instruction {
        kind,
        targets: vec![target],
        controls: controls.to_vec(),
        params: Vec::new(),
        matrix: None,
    }
}

fn controlled_unitary(u: [C64; 4], control: usize, target: usize) -> Instruction {
    Instruction {
        kind: GateKind::Unitary,
        targets: vec![target],
        controls: vec![control],
        params: Vec::new(),
        matrix: Some(Arc::new(u.to_vec())),
    }
}

/// A square root `V` (`V² = U`) of a 2×2 unitary.
pub(crate) fn sqrt_unitary(u: &[C64; 4]) -> [C64; 4] {
    // Eigenvalues λ of U; V = (U + s₁s₂·I)/(s₁ + s₂) for roots s_i of λ_i
    // (Cayley–Hamilton), with the branch that keeps s₁ + s₂ away from 0.
    let tr = u[0] + u[3];
    let det = u[0] * u[3] - u[1] * u[2];
    let disc = (tr * tr - 4.0 * det).sqrt();
    let (l1, l2) = ((tr + disc) / 2.0, (tr - disc) / 2.0);
    let s1 = l1.sqrt();
    let mut s2 = l2.sqrt();
    if (s1 + s2).norm() < (s1 - s2).norm() {
        s2 = -s2;
    }
    if (s1 + s2).norm() < 1e-9 {
        // λ₁ = λ₂: U is a multiple of the identity.
        return [s1, C64::new(0.0, 0.0), C64::new(0.0, 0.0), s1];
    }
    let p = s1 * s2;
    let d = s1 + s2;
    [(u[0] + p) / d, u[1] / d, u[2] / d, (u[3] + p) / d]
}

fn adjoint(u: &[C64; 4]) -> [C64; 4] {
    [u[0].conj(), u[2].conj(), u[1].conj(), u[3].conj()]
}

/// `C^k(X)` on `controls → target` with `k − 2` idle qubits `dirty`
/// (Lemma 7.2): Toffolis `T_j = (c_{j+1}, a_{j−1} → a_j)` along the ladder
/// of idle qubits, applied down-up twice.
fn mcx_ladder(controls: &[usize], target: usize, dirty: &[usize], out: &mut Vec<Instruction>) {
    let k = controls.len();
    debug_assert!(k >= 3 && dirty.len() >= k - 2);
    // a[j] for j = 1..=k−2 are idle qubits, a[k−1] is the target.
    let a = |j: usize| if j == k - 1 { target } else { dirty[j - 1] };
    let toffoli = |j: usize| {
        if j == 1 {
            gate(GateKind::X, a(1), &[controls[0], controls[1]])
        } else {
            gate(GateKind::X, a(j), &[controls[j], a(j - 1)])
        }
    };
    let sweep = |top: usize, out: &mut Vec<Instruction>| {
        for j in (1..=top).rev() {
            out.push(toffoli(j));
        }
        for j in 2..=top {
            out.push(toffoli(j));
        }
    };
    sweep(k - 1, out);
    sweep(k - 2, out);
}

/// `C^k(X)` on `controls → target`, using any of `idle` (qubits not among
/// the controls and the target) as dirty ancillas. Needs at least one idle
/// qubit for `k ≥ 3`.
pub(crate) fn mcx(controls: &[usize], target: usize, idle: &[usize], out: &mut Vec<Instruction>) {
    let k = controls.len();
    if k <= 2 {
        out.push(gate(GateKind::X, target, controls));
        return;
    }
    if idle.len() >= k - 2 {
        mcx_ladder(controls, target, idle, out);
        return;
    }
    debug_assert!(!idle.is_empty(), "C^k(X) with k ≥ 3 needs an idle qubit");
    // Split with one idle qubit `a` (Lemma 7.3): the target flips by
    // (g₂ ∧ a) ⊕ (g₂ ∧ (a ⊕ g₁)) = g₁ ∧ g₂, and `a` is restored.
    let a = idle[0];
    let m1 = k.div_ceil(2);
    let (g1, g2) = controls.split_at(m1);
    let mut g2a = g2.to_vec();
    g2a.push(a);
    let mut idle_for_g1: Vec<usize> = g2.to_vec();
    idle_for_g1.push(target);
    for _ in 0..2 {
        mcx(&g2a, target, g1, out);
        mcx(g1, a, &idle_for_g1, out);
    }
}

/// `C^k(U)` for a single-qubit unitary `u` on `controls → target`
/// (Lemma 7.5), using only singly controlled unitaries and Toffolis.
pub(crate) fn mcu(u: [C64; 4], controls: &[usize], target: usize, out: &mut Vec<Instruction>) {
    match controls {
        [] => out.push(Instruction {
            kind: GateKind::Unitary,
            targets: vec![target],
            controls: Vec::new(),
            params: Vec::new(),
            matrix: Some(Arc::new(u.to_vec())),
        }),
        [c] => out.push(controlled_unitary(u, *c, target)),
        _ => {
            let v = sqrt_unitary(&u);
            let (rest, last) = controls.split_at(controls.len() - 1);
            let c = last[0];
            out.push(controlled_unitary(v, c, target));
            mcx(rest, c, &[target], out);
            out.push(controlled_unitary(adjoint(&v), c, target));
            mcx(rest, c, &[target], out);
            mcu(v, rest, target, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::{Circuit, StateVector};
    use crate::rng::Rng;

    fn run(n: usize, gates: &[Instruction], psi: &StateVector) -> StateVector {
        let mut c = Circuit::new(n);
        for g in gates {
            c.try_push(g.clone()).unwrap();
        }
        let mut out = psi.clone();
        out.apply_circuit(&c, &[]).unwrap();
        out
    }

    fn assert_equal(n: usize, expected: &[Instruction], actual: &[Instruction]) {
        let mut rng = Rng::seed_from_u64(7);
        for _ in 0..4 {
            let psi = StateVector::random(n, &mut rng);
            let (a, b) = (run(n, expected, &psi), run(n, actual, &psi));
            let overlap = a.inner(&b).unwrap();
            // Equal including the phase: ⟨a|b⟩ = 1.
            assert!((overlap - C64::new(1.0, 0.0)).norm() < 1e-10, "{overlap}");
        }
    }

    #[test]
    fn square_roots_square_back() {
        let mut rng = Rng::seed_from_u64(1);
        let x = [
            C64::new(0.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
        ];
        let minus_one = [
            C64::new(-1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(-1.0, 0.0),
        ];
        let z = [
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(-1.0, 0.0),
        ];
        let mut cases = vec![x, minus_one, z];
        for _ in 0..20 {
            let m =
                crate::linalg::CMatrix::from_fn(2, 2, |_, _| C64::new(rng.normal(), rng.normal()));
            let q = m.qr().q();
            cases.push([q[(0, 0)], q[(0, 1)], q[(1, 0)], q[(1, 1)]]);
        }
        for u in cases {
            let v = sqrt_unitary(&u);
            let vv = [
                v[0] * v[0] + v[1] * v[2],
                v[0] * v[1] + v[1] * v[3],
                v[2] * v[0] + v[3] * v[2],
                v[2] * v[1] + v[3] * v[3],
            ];
            for (a, b) in vv.iter().zip(&u) {
                assert!((a - b).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn multi_controlled_x_is_exact() {
        for k in 3..=7 {
            let n = k + 1 + (k - 2); // enough idle qubits for the ladder
            let controls: Vec<usize> = (0..k).collect();
            let idle: Vec<usize> = (k + 1..n).collect();
            let mut out = Vec::new();
            mcx(&controls, k, &idle, &mut out);
            assert_eq!(out.len(), 4 * (k - 2));
            assert_equal(n, &[gate(GateKind::X, k, &controls)], &out);
            // One idle qubit only.
            let mut split = Vec::new();
            mcx(&controls, k, &[k + 1], &mut split);
            assert!(split.iter().all(|g| g.controls.len() <= 2));
            assert_equal(k + 2, &[gate(GateKind::X, k, &controls)], &split);
        }
    }

    #[test]
    fn multi_controlled_unitaries_are_exact_without_ancillas() {
        let mut rng = Rng::seed_from_u64(3);
        for k in 0..=6 {
            let n = k + 1;
            let m =
                crate::linalg::CMatrix::from_fn(2, 2, |_, _| C64::new(rng.normal(), rng.normal()));
            let q = m.qr().q();
            let u = [q[(0, 0)], q[(0, 1)], q[(1, 0)], q[(1, 1)]];
            // Controls in scrambled order, target in the middle.
            let target = k / 2;
            let controls: Vec<usize> = (0..n).rev().filter(|&q| q != target).collect();
            let mut out = Vec::new();
            mcu(u, &controls, target, &mut out);
            assert!(out.iter().all(|g| g.controls.len() <= 2));
            let expected = Instruction {
                kind: GateKind::Unitary,
                targets: vec![target],
                controls: controls.clone(),
                params: Vec::new(),
                matrix: Some(Arc::new(u.to_vec())),
            };
            assert_equal(n, &[expected], &out);
        }
    }
}
