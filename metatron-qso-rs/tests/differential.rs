//! Differential tests: random circuits on every simulator that takes them
//! (see `metatron_qso::differential`): every gate kind, any number of
//! controls, dense unitaries on one to three qubits, symbolic parameters,
//! special angles, one to ten qubits, and Clifford-only circuits for the
//! tableau; plus the disagreements found so far, as regression tests.
#![cfg(not(target_family = "wasm"))]

use metatron_qso::differential::compare;
use metatron_qso::mps::{Mps, MpsOptions};
use metatron_qso::prelude::*;
use metatron_qso::stabilizer::StabilizerState;
use proptest::prelude::*;
use std::f64::consts::{FRAC_PI_2, PI};

/// One random gate: (kind selector, a permutation of the qubits, angle
/// selector, angle, seed).
type Gate = (u8, Vec<usize>, u8, f64, u64);

/// Cases per property test: 256 on every change; the scheduled job sets
/// QSO_DIFFERENTIAL_CASES for longer runs.
fn cases() -> u32 {
    std::env::var("QSO_DIFFERENTIAL_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256)
}

fn gate(n: usize, kinds: u8) -> impl Strategy<Value = Gate> {
    (
        0u8..kinds,
        Just((0..n).collect::<Vec<usize>>()).prop_shuffle(),
        0u8..8,
        -6.3f64..6.3,
        any::<u64>(),
    )
}

/// A random unitary on `k` qubits from a QR decomposition.
fn unitary(k: usize, seed: u64) -> Vec<C64> {
    let mut rng = metatron_qso::Rng::seed_from_u64(seed);
    let d = 1 << k;
    let m =
        metatron_qso::linalg::CMatrix::from_fn(d, d, |_, _| C64::new(rng.normal(), rng.normal()));
    let u = m.qr().q();
    (0..d * d).map(|i| u[(i / d, i % d)]).collect()
}

/// Special angles reach the diagonal and anti-diagonal branches of the
/// decompositions; large ones the rounding of the exporters.
fn angle(selector: u8, a: f64) -> f64 {
    match selector {
        0 => 0.0,
        1 => PI,
        2 => -FRAC_PI_2,
        3 => a * 1000.0,
        _ => a,
    }
}

/// Build the circuit; symbolic angles become parameters whose values are
/// returned as theta.
fn build(n: usize, gates: &[Gate], symbolic: bool) -> (Circuit, Vec<f64>) {
    use GateKind::*;
    let mut c = Circuit::new(n);
    let mut theta = Vec::new();
    for (k, q, sel, a, seed) in gates {
        let value = angle(*sel, *a);
        let p: Param = if symbolic && sel % 2 == 1 {
            let s = c.new_parameter();
            theta.push(value);
            Param::Symbol(s * 1.0)
        } else {
            value.into()
        };
        let (a, b, t) = (q[0], q[1 % n], q[2 % n]);
        let two = n >= 2;
        let three = n >= 3;
        let k = *k % 40;
        match k {
            // Clifford gates first (selectors 0..12).
            0 => c.h(a),
            1 => c.x(a),
            2 => c.y(a),
            3 => c.z(a),
            4 => c.s(a),
            5 => c.sdg(a),
            6 => c.sx(a),
            7 => c.sxdg(a),
            8 if two => c.cx(a, b),
            9 if two => c.cz(a, b),
            10 if two => c.swap(a, b),
            11 if two => c.cy(a, b),
            12 => c.t(a),
            13 => c.tdg(a),
            14 => c.rx(a, p),
            15 => c.ry(a, p),
            16 => c.rz(a, p),
            17 => c.p(a, p),
            18 => c.u(a, p, 0.4, -0.9),
            19 if two => c.crx(a, b, p),
            20 if two => c.cry(a, b, p),
            21 if two => c.crz(a, b, p),
            22 if two => c.cp(a, b, p),
            23 if two => c.rxx(a, b, p),
            24 if two => c.ryy(a, b, p),
            25 if two => c.rzz(a, b, p),
            26 if three => c.ccx(a, b, t),
            27 if three => c.cswap(a, b, t),
            28 if n >= 4 => c.mcx(&q[1..4.min(n)], a),
            29 => c.mcz(&q[..n.min(4)]),
            30 => c.unitary(unitary(1, *seed), &[a], &[]).unwrap(),
            31 if two => c.unitary(unitary(2, *seed), &[a, b], &[]).unwrap(),
            32 if three => c.unitary(unitary(3, *seed), &[a, b, t], &[]).unwrap(),
            33 if two => c.unitary(unitary(1, *seed), &[a], &q[1..n.min(4)]).unwrap(),
            34 if three => c
                .unitary(unitary(2, *seed), &[a, b], &q[2..n.min(5)])
                .unwrap(),
            // Any gate kind under one to three controls.
            35..=39 if n >= 2 => {
                let kinds = [S, T, SX, U, RXX, Swap, Tdg, SXdg, RY, H];
                let kind = kinds[(*seed % kinds.len() as u64) as usize];
                let targets = kind.num_targets().unwrap_or(1);
                let free = n - targets;
                if free == 0 {
                    continue;
                }
                let controls = &q[targets..targets + 1 + (*seed as usize / 7) % free.min(3)];
                let params: Vec<Param> = match kind.num_params() {
                    0 => vec![],
                    1 => vec![p],
                    _ => vec![p, 0.3.into(), (-1.1).into()],
                };
                c.controlled(kind, &q[..targets], controls, &params)
                    .unwrap()
            }
            _ => c.h(a),
        };
    }
    (c, theta)
}

fn check(c: &Circuit, theta: &[f64]) -> std::result::Result<(), TestCaseError> {
    let r = compare(c, theta).unwrap();
    prop_assert!(r.failures().is_empty(), "{:?}\n{r:?}\n{c:?}", r.failures());
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: cases(),
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn random_circuits_agree_on_every_simulator(
        (n, gates) in (1usize..7).prop_flat_map(|n| (Just(n), prop::collection::vec(gate(n, 40), 1..40))),
        symbolic in any::<bool>(),
    ) {
        let (c, theta) = build(n, &gates, symbolic);
        check(&c, &theta)?;
    }

    #[test]
    fn clifford_circuits_agree_with_the_tableau(
        (n, gates) in (1usize..7).prop_flat_map(|n| (Just(n), prop::collection::vec(gate(n, 12), 1..60))),
    ) {
        let (c, _) = build(n, &gates, false);
        let r = compare(&c, &[]).unwrap();
        prop_assert!(r.stabilizer.is_some() && r.qasm2.is_some() && r.qasm3.is_some());
        prop_assert!(r.failures().is_empty(), "{:?}\n{r:?}", r.failures());
    }
}

proptest! {
    // Seven to ten qubits: the density matrix costs 4ⁿ, so fewer cases.
    #![proptest_config(ProptestConfig {
        cases: cases() / 16 + 1,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn large_registers_agree(
        (n, gates) in (7usize..11).prop_flat_map(|n| (Just(n), prop::collection::vec(gate(n, 40), 1..25))),
    ) {
        let (c, theta) = build(n, &gates, true);
        check(&c, &theta)?;
    }
}

#[test]
fn mps_without_cutoff_stays_finite() {
    // Found by the review of 1.21: with cutoff 0 the MPS kept singular
    // values of 10⁻⁴⁰, whose squares broke the next eigendecomposition
    // (a NaN state, an error from `compare`).
    let mut c = Circuit::new(6);
    c.rz(3, 0.375)
        .cz(0, 5)
        .sx(0)
        .x(4)
        .rzz(1, 0, -4.3125)
        .y(3)
        .t(0)
        .sx(3)
        .s(2)
        .rz(3, -4.9375)
        .rz(3, -6.625)
        .rzz(4, 2, -6.8125)
        .cswap(5, 0, 1);
    let r = compare(&c, &[]).unwrap();
    assert!(r.failures().is_empty(), "{r:?}");
    let options = MpsOptions {
        max_bond: 8,
        cutoff: 0.0,
    };
    let mut mps = Mps::from_circuit(&c, &[], options).unwrap();
    assert!(mps.expectation_str("Z0").unwrap().is_finite());
    assert!(mps.to_statevector().is_ok());
}

#[test]
fn the_tableau_refuses_rotations_it_cannot_round() {
    // Found by the review of 1.21: from |θ| ≈ 4·10¹⁴ on, every rotation
    // passed as a Clifford gate.
    for theta in [1e15, 5e14, 1e6, 3.1e18] {
        let mut c = Circuit::new(1);
        c.h(0).rz(0, theta);
        assert!(StabilizerState::from_circuit(&c, &[]).is_err(), "{theta}");
        assert!(compare(&c, &[]).unwrap().stabilizer.is_none());
    }
    // Rotations within 10⁻⁹ of a Clifford gate pass, and their errors add
    // up within the tolerance of `compare`.
    let mut c = Circuit::new(1);
    c.h(0).rz(0, FRAC_PI_2 + 9e-10).rz(0, FRAC_PI_2 + 9e-10);
    let r = compare(&c, &[]).unwrap();
    assert!(r.stabilizer.is_some());
    assert!(r.failures().is_empty(), "{r:?}");
}

#[test]
fn rounded_exports_would_be_seen() {
    // The measure is linear in the error: an angle off by 10⁻⁶ (an export
    // rounded to six digits) gives a deviation of about 10⁻⁷, a thousand
    // times the tolerance.
    let mut a = Circuit::new(2);
    a.rx(0, 0.123456789).cx(0, 1);
    let mut b = Circuit::new(2);
    b.rx(0, 0.123457).cx(0, 1);
    let x = StateVector::from_circuit(&a, &[]).unwrap();
    let y = StateVector::from_circuit(&b, &[]).unwrap();
    let max = x
        .amplitudes()
        .iter()
        .zip(y.amplitudes())
        .map(|(p, q)| (p - q).norm())
        .fold(0.0, f64::max);
    assert!(max > 1e3 * metatron_qso::differential::TOLERANCE, "{max}");
}
