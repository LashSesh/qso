//! Differential tests: random circuits on every simulator that takes them
//! (see `metatron_qso::differential`), with every gate kind, controls and
//! dense unitaries, and Clifford-only circuits for the tableau.
#![cfg(not(target_family = "wasm"))]

use metatron_qso::differential::compare;
use metatron_qso::prelude::*;
use proptest::prelude::*;

/// One random gate: (kind selector, four qubits — distinct within each
/// gate's use —, angle, seed for dense unitaries).
type Gate = (u8, [usize; 4], f64, u64);

fn gate(n: usize, kinds: u8) -> impl Strategy<Value = Gate> {
    (
        0u8..kinds,
        Just(n).prop_perturb(|n, mut rng| {
            let mut q: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                q.swap(i, rng.random_range(0..=i));
            }
            [q[0], q[1 % n], q[2 % n], q[3 % n]]
        }),
        -6.3f64..6.3,
        any::<u64>(),
    )
}

/// Haar-ish random unitary on `k` qubits from a QR decomposition.
fn unitary(k: usize, seed: u64) -> Vec<C64> {
    let mut rng = metatron_qso::Rng::seed_from_u64(seed);
    let d = 1 << k;
    let m =
        metatron_qso::linalg::CMatrix::from_fn(d, d, |_, _| C64::new(rng.normal(), rng.normal()));
    let u = m.qr().q();
    (0..d * d).map(|i| u[(i / d, i % d)]).collect()
}

fn build(n: usize, gates: &[Gate]) -> Circuit {
    let mut c = Circuit::new(n);
    for &(k, [a, b, t, u], angle, seed) in gates {
        match k {
            // Clifford gates first (selectors 0..9).
            0 => c.h(a),
            1 => c.x(a),
            2 => c.y(a),
            3 => c.z(a),
            4 => c.s(a),
            5 => c.cx(a, b),
            6 => c.cz(a, b),
            7 => c.swap(a, b),
            8 => c.sdg(a),
            9 => c.cy(a, b),
            10 => c.t(a),
            11 => c.sx(a),
            12 => c.rx(a, angle),
            13 => c.ry(a, angle),
            14 => c.rz(a, angle),
            15 => c.p(a, angle),
            16 => c.u(a, angle, 0.4, -0.9),
            17 => c.crx(a, b, angle),
            18 => c.cry(a, b, angle),
            19 => c.crz(a, b, angle),
            20 => c.cp(a, b, angle),
            21 => c.rxx(a, b, angle),
            22 => c.ryy(a, b, angle),
            23 => c.rzz(a, b, angle),
            24 => c.ccx(a, b, t),
            25 => c.cswap(a, b, t),
            26 => c.mcx(&[a, b, t], u),
            27 => c.unitary(unitary(1, seed), &[a], &[]).unwrap(),
            28 => c.unitary(unitary(2, seed), &[a, b], &[]).unwrap(),
            29 => c.unitary(unitary(2, seed), &[a, b], &[t]).unwrap(),
            _ => c.ch(a, b),
        };
    }
    c
}

fn check(c: &Circuit) -> std::result::Result<(), TestCaseError> {
    let r = compare(c, &[]).unwrap();
    prop_assert!(r.failures().is_empty(), "{:?}\n{r:?}", r.failures());
    Ok(())
}

proptest! {
    // 48 cases per test on every change; the scheduled job sets
    // QSO_DIFFERENTIAL_CASES for longer runs.
    #![proptest_config(ProptestConfig {
        cases: std::env::var("QSO_DIFFERENTIAL_CASES").ok().and_then(|v| v.parse().ok()).unwrap_or(48),
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn random_circuits_agree_on_every_simulator(
        (n, gates) in (4usize..7).prop_flat_map(|n| (Just(n), prop::collection::vec(gate(n, 31), 1..40))),
    ) {
        check(&build(n, &gates))?;
    }

    #[test]
    fn clifford_circuits_agree_with_the_tableau(
        gates in prop::collection::vec(gate(5, 10), 1..60),
    ) {
        let c = build(5, &gates);
        let r = compare(&c, &[]).unwrap();
        prop_assert!(r.stabilizer.is_some());
        prop_assert!(r.failures().is_empty(), "{:?}\n{r:?}", r.failures());
    }

    #[test]
    fn small_registers_agree(
        (n, gates) in (1usize..4).prop_flat_map(|n| (Just(n), prop::collection::vec(gate(n, 24), 1..30))),
    ) {
        // One to three qubits: degenerate strides and tails; gates that
        // need more qubits than the register are dropped.
        let gates: Vec<Gate> = gates
            .into_iter()
                .filter(|(k, _, _, _)| {
                let needs = match k { 0..=4 | 8 | 10..=16 => 1, _ => 2 };
                needs <= n
            })
            .collect();
        check(&build(n, &gates))?;
    }
}
