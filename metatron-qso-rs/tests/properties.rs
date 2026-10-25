//! Property-based tests on random circuits, observables and graphs.
#![cfg(not(target_family = "wasm"))]

use metatron_qso::algorithms::gradients::{GradientMethod, expectation_and_gradient};
use metatron_qso::prelude::*;
use metatron_qso::rng::Rng as QsoRng;
use proptest::prelude::*;

const N: usize = 4;

/// One random gate: (kind selector, three distinct qubits, angle).
fn gate() -> impl Strategy<Value = (u8, [usize; 3], f64)> {
    // The 24 ordered triples of distinct qubits out of 4.
    let triples: Vec<[usize; 3]> = (0..N)
        .flat_map(|a| (0..N).flat_map(move |b| (0..N).map(move |c| [a, b, c])))
        .filter(|[a, b, c]| a != b && b != c && a != c)
        .collect();
    (0u8..24, prop::sample::select(triples), -6.3f64..6.3)
}

fn build(gates: &[(u8, [usize; 3], f64)], parametrised: bool) -> Circuit {
    let mut c = Circuit::new(N);
    for &(k, [a, b, t], angle) in gates {
        let theta: Param = if parametrised {
            let p = c.new_parameter();
            Param::Symbol(p * 0.7 + 0.1)
        } else {
            angle.into()
        };
        match k {
            0 => c.h(a),
            1 => c.x(a),
            2 => c.y(a),
            3 => c.z(a),
            4 => c.s(a),
            5 => c.t(a),
            6 => c.sx(a),
            7 => c.rx(a, theta),
            8 => c.ry(a, theta),
            9 => c.rz(a, theta),
            10 => c.p(a, theta),
            11 => c.u(a, theta, 0.4, -0.9),
            12 => c.cx(a, b),
            13 => c.cy(a, b),
            14 => c.cz(a, b),
            15 => c.swap(a, b),
            16 => c.crx(a, b, theta),
            17 => c.cry(a, b, theta),
            18 => c.crz(a, b, theta),
            19 => c.cp(a, b, theta),
            20 => c.rxx(a, b, theta),
            21 => c.ryy(a, b, theta),
            22 => c.rzz(a, b, theta),
            _ => c.ccx(a, b, t),
        };
    }
    c
}

fn params_for(c: &Circuit, gates: &[(u8, [usize; 3], f64)]) -> Vec<f64> {
    let angles: Vec<f64> = gates.iter().map(|g| g.2).collect();
    angles[..c.num_parameters()].to_vec()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn circuits_preserve_norm_and_inverse_restores_state(gates in prop::collection::vec(gate(), 1..30), seed in 0u64..1000) {
        let c = build(&gates, false);
        let psi0 = StateVector::random(N, &mut QsoRng::seed_from_u64(seed));
        let mut psi = psi0.clone();
        psi.apply_circuit(&c, &[]).unwrap();
        prop_assert!((psi.norm() - 1.0).abs() < 1e-12);
        psi.apply_circuit(&c.inverse(), &[]).unwrap();
        prop_assert!((psi.fidelity(&psi0).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn qasm_round_trip_preserves_state(gates in prop::collection::vec(gate(), 1..30)) {
        let c = build(&gates, false);
        let parsed = from_qasm2(&to_qasm2(&c, &[], false).unwrap()).unwrap();
        let a = StateVector::from_circuit(&c, &[]).unwrap();
        let b = StateVector::from_circuit(&parsed, &[]).unwrap();
        prop_assert!((a.fidelity(&b).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn density_matrix_matches_statevector(gates in prop::collection::vec(gate(), 1..20)) {
        let c = build(&gates, false);
        let psi = StateVector::from_circuit(&c, &[]).unwrap();
        let rho = DensityMatrix::from_circuit(&c, &[]).unwrap();
        prop_assert!((rho.fidelity_with_pure(&psi).unwrap() - 1.0).abs() < 1e-10);
        prop_assert!((rho.purity() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn adjoint_gradient_matches_finite_differences(gates in prop::collection::vec(gate(), 1..16)) {
        let c = build(&gates, true);
        prop_assume!(c.num_parameters() > 0);
        let theta = params_for(&c, &gates);
        let h: PauliSum = "0.6 Z0 Z1 - 0.3 X2 + 0.8 Y1 Y3 + 0.2 X0 Z2 Y3".parse().unwrap();
        let (e1, g1) = expectation_and_gradient(&c, &h, &theta, None, GradientMethod::Adjoint).unwrap();
        let (e2, g2) = expectation_and_gradient(&c, &h, &theta, None, GradientMethod::FiniteDifference { step: 1e-6 }).unwrap();
        prop_assert!((e1 - e2).abs() < 1e-12);
        for (a, b) in g1.iter().zip(&g2) {
            prop_assert!((a - b).abs() < 1e-6, "adjoint {} vs fd {}", a, b);
        }
    }

    #[test]
    fn automorphism_order_is_invariant_under_relabelling(n in 3usize..9, p in 0.2f64..0.8, seed in 0u64..500) {
        let mut rng = QsoRng::seed_from_u64(seed);
        let g = generators::erdos_renyi(n, p, &mut rng).unwrap();
        let mut perm: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut perm);
        let order = g.automorphism_group().order();
        prop_assert_eq!(g.relabeled(&perm).unwrap().automorphism_group().order(), order);
        // Brute force for small graphs: count permutations preserving edges.
        if n <= 7 {
            let mut count = 0u128;
            let mut idx: Vec<usize> = (0..n).collect();
            permute(&mut idx, 0, &mut |q| {
                if metatron_qso::graph::automorphism::is_automorphism(&g, q) {
                    count += 1;
                }
            });
            prop_assert_eq!(order, Some(count));
        }
    }
}

fn permute(v: &mut Vec<usize>, k: usize, f: &mut impl FnMut(&[usize])) {
    if k == v.len() {
        f(v);
        return;
    }
    for i in k..v.len() {
        v.swap(k, i);
        permute(v, k + 1, f);
        v.swap(k, i);
    }
}
