use super::*;
use crate::quantum::{Circuit, StateVector};

/// A random circuit of every gate kind, distant pairs and controls
/// included, on `n` qubits.
fn random_circuit(n: usize, gates: usize, rng: &mut Rng) -> Circuit {
    let mut c = Circuit::new(n);
    let angle = |rng: &mut Rng| rng.uniform_range(-3.0, 3.0);
    for _ in 0..gates {
        let a = rng.below(n as u64) as usize;
        let b = if n > 1 {
            (a + 1 + rng.below(n as u64 - 1) as usize) % n
        } else {
            a
        };
        let others: Vec<usize> = (0..n).filter(|&q| q != a && q != b).collect();
        let kind = match n {
            1 => rng.below(6),
            2 => rng.below(12),
            _ => rng.below(16),
        };
        match kind {
            0 => c.h(a),
            1 => c.t(a),
            2 => c.sx(a),
            3 => c.rx(a, angle(rng)),
            4 => c.ry(a, angle(rng)),
            5 => c.u(a, angle(rng), angle(rng), angle(rng)),
            6 => c.cx(a, b),
            7 => c.cz(a, b),
            8 => c.rzz(a, b, angle(rng)),
            9 => c.rxx(b, a, angle(rng)),
            10 => c.swap(a, b),
            11 => c.crx(a, b, angle(rng)),
            12 | 13 => c.ccx(a, b, others[rng.below(others.len() as u64) as usize]),
            _ => {
                let haar = crate::photonics::haar_unitary(4, rng).unwrap();
                let m: Vec<C64> = (0..16).map(|k| haar[(k / 4, k % 4)]).collect();
                let t = others[rng.below(others.len() as u64) as usize];
                c.unitary(m, &[b, t], &[a]).unwrap()
            }
        };
    }
    c
}

fn exact_entropy(psi: &StateVector, qubits: &[usize]) -> f64 {
    psi.entanglement_entropy(qubits).unwrap()
}

#[test]
fn random_circuits_match_the_state_vector() {
    let mut rng = Rng::seed_from_u64(1);
    for trial in 0..120 {
        let n = 1 + trial % 12;
        let c = random_circuit(n, 6 + 4 * n, &mut rng);
        let psi = StateVector::from_circuit(&c, &[]).unwrap();
        let mut mps = Mps::from_circuit(&c, &[], MpsOptions::exact()).unwrap();
        let phi = mps.to_statevector().unwrap();
        let bound = mps.fidelity_lower_bound().unwrap();
        assert!(bound > 1.0 - 1e-9, "trial {trial}, n = {n}");
        assert!(
            psi.fidelity(&phi).unwrap() >= bound - 1e-12,
            "trial {trial}"
        );
        // Amplitudes agree including the global phase.
        for (k, &a) in psi.amplitudes().iter().enumerate().take(16) {
            let bits: Vec<bool> = (0..n).map(|q| (k >> q) & 1 == 1).collect();
            let got = mps.amplitude(&bits).unwrap();
            // The truncation distance bounds the error, phase included.
            assert!(
                (got - a).norm() <= mps.distance + 1e-12,
                "trial {trial} k {k}: {} vs {} ({} truncations)",
                (got - a).norm(),
                mps.distance,
                mps.truncations
            );
        }
        // Pauli expectation values.
        for _ in 0..4 {
            let ops: Vec<(usize, Pauli)> = (0..n)
                .filter_map(|q| match rng.below(4) {
                    0 => None,
                    1 => Some((q, Pauli::X)),
                    2 => Some((q, Pauli::Y)),
                    _ => Some((q, Pauli::Z)),
                })
                .collect();
            let p = crate::quantum::PauliString::new(&ops).unwrap();
            let h = PauliSum::from_terms([(1.0, p)]);
            let exact = h.expectation(&psi).unwrap();
            assert!((mps.expectation(&ops).unwrap() - exact).abs() < 1e-9);
        }
        // Entanglement across each bond of the site order.
        let mut m = mps.clone();
        for (b, s) in m.bond_entropies().unwrap().iter().enumerate() {
            // Both sides have the same entropy; the smaller is cheaper.
            let side: Vec<usize> = if 2 * (b + 1) <= n {
                (0..=b).collect()
            } else {
                (b + 1..n).collect()
            };
            assert!((s - exact_entropy(&psi, &side)).abs() < 1e-8, "bond {b}");
        }
    }
}

#[test]
fn truncation_is_bounded() {
    // Random circuits on 10 qubits, bond dimension at most 4: the true
    // fidelity respects the bound.
    let mut rng = Rng::seed_from_u64(2);
    let opts = MpsOptions {
        max_bond: 4,
        cutoff: 1e-10,
    };
    let mut truncated = 0;
    for _ in 0..30 {
        let c = random_circuit(10, 40, &mut rng);
        let psi = StateVector::from_circuit(&c, &[]).unwrap();
        let mps = Mps::from_circuit(&c, &[], opts).unwrap();
        assert!(mps.max_bond_dimension() <= 4);
        let f = psi.fidelity(&mps.to_statevector().unwrap()).unwrap();
        let bound = mps.fidelity_lower_bound().unwrap();
        assert!(f >= bound - 1e-12, "fidelity {f} below bound {bound}");
        if mps.truncations() > 0 {
            truncated += 1;
            assert!(mps.discarded_weight() > 0.0);
        }
    }
    assert!(truncated > 20);
}

#[test]
fn sampling_and_measurement_follow_the_born_rule() {
    let mut rng = Rng::seed_from_u64(3);
    let c = random_circuit(5, 30, &mut rng);
    let psi = StateVector::from_circuit(&c, &[]).unwrap();
    let probs = psi.probabilities();
    let mut mps = Mps::from_circuit(&c, &[], MpsOptions::exact()).unwrap();
    let shots = 40_000;
    let samples = mps.sample(shots, &mut rng).unwrap();
    let mut counts = vec![0usize; 32];
    for bits in &samples {
        counts[(0..5).fold(0, |k, q| k | (usize::from(bits[q]) << q))] += 1;
    }
    for (k, &p) in probs.iter().enumerate() {
        let f = counts[k] as f64 / shots as f64;
        assert!((f - p).abs() < 5.0 * (p * (1.0 - p) / shots as f64).sqrt() + 1e-9);
    }
    // Measuring qubit 2 collapses to the conditional state.
    let p1: f64 = probs
        .iter()
        .enumerate()
        .filter(|(k, _)| k >> 2 & 1 == 1)
        .map(|(_, p)| p)
        .sum();
    let mut ones = 0;
    for t in 0..400 {
        let mut m = mps.clone();
        let mut r = Rng::seed_from_u64(100 + t);
        let v = m.measure(2, &mut r).unwrap();
        ones += usize::from(v);
        assert!(
            (m.expectation(&[(2, Pauli::Z)]).unwrap() - if v { -1.0 } else { 1.0 }).abs() < 1e-10
        );
        let phi = m.to_statevector().unwrap();
        let mut proj: Vec<C64> = psi
            .amplitudes()
            .iter()
            .enumerate()
            .map(|(k, &a)| if (k >> 2 & 1 == 1) == v { a } else { ZERO })
            .collect();
        let norm: f64 = proj.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        proj.iter_mut().for_each(|z| *z /= norm);
        let f = phi
            .fidelity(&StateVector::from_amplitudes(proj).unwrap())
            .unwrap();
        assert!((f - 1.0).abs() < 1e-10);
    }
    let f = ones as f64 / 400.0;
    assert!((f - p1).abs() < 5.0 * (p1 * (1.0 - p1) / 400.0).sqrt() + 1e-9);
    // Reset leaves |0⟩.
    let mut m = mps.clone();
    m.reset(4, &mut rng).unwrap();
    assert!((m.expectation(&[(4, Pauli::Z)]).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn ghz_states_have_bond_dimension_two() {
    let n = 2000;
    let mut c = Circuit::new(n);
    c.h(0);
    for q in 1..n {
        c.cx(q - 1, q);
    }
    let mut mps = Mps::from_circuit(&c, &[], MpsOptions::default()).unwrap();
    assert_eq!(mps.max_bond_dimension(), 2);
    assert_eq!(mps.truncations(), 0);
    assert!((mps.expectation_str("Z0 Z1999").unwrap() - 1.0).abs() < 1e-10);
    assert!(mps.expectation_str("Z1000").unwrap().abs() < 1e-10);
    assert!(
        (mps.amplitude(&vec![true; n]).unwrap().norm() - std::f64::consts::FRAC_1_SQRT_2).abs()
            < 1e-10
    );
    let entropies = mps.bond_entropies().unwrap();
    assert!(entropies.iter().all(|s| (s - 1.0).abs() < 1e-10));
    let mut rng = Rng::seed_from_u64(4);
    for bits in mps.sample(20, &mut rng).unwrap() {
        assert!(bits.iter().all(|&b| b == bits[0]));
    }
    // A gate between the ends routes by swaps and keeps the state exact.
    let mut far = Mps::zero(40, MpsOptions::default()).unwrap();
    far.h(0).unwrap();
    far.cx(0, 39).unwrap();
    assert!((far.expectation_str("Z0 Z39").unwrap() - 1.0).abs() < 1e-12);
    assert!((far.expectation_str("X0 X39").unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn invalid_input_is_an_error() {
    assert!(Mps::zero(0, MpsOptions::default()).is_err());
    assert!(Mps::from_circuit(&Circuit::new(0), &[], MpsOptions::default()).is_err());
    assert!(
        Mps::zero(
            2,
            MpsOptions {
                max_bond: 0,
                cutoff: 0.0
            }
        )
        .is_err()
    );
    assert!(
        Mps::zero(
            2,
            MpsOptions {
                max_bond: 4,
                cutoff: 1.5
            }
        )
        .is_err()
    );
    let mut m = Mps::zero(3, MpsOptions::default()).unwrap();
    assert!(m.apply_matrix(&[ONE, ZERO, ZERO, ONE], &[3]).is_err());
    assert!(m.apply_matrix(&[ONE, ONE, ZERO, ONE], &[0]).is_err());
    assert!(m.apply_matrix(&[ONE; 16], &[0, 0]).is_err());
    assert!(m.expectation(&[(0, Pauli::X), (0, Pauli::Z)]).is_err());
    assert!(m.expectation_str("Z5").is_err());
    assert!(m.amplitude(&[false]).is_err());
    let mut rng = Rng::seed_from_u64(0);
    assert!(m.measure(7, &mut rng).is_err());
    let mut big = Circuit::new(10);
    big.mcx(&(0..9).collect::<Vec<_>>(), 9);
    assert!(Mps::from_circuit(&big, &[], MpsOptions::default()).is_err());
    let mut p = Circuit::new(1);
    let t = p.new_parameter();
    p.rx(0, t);
    assert!(Mps::from_circuit(&p, &[], MpsOptions::default()).is_err());
    assert!(
        Mps::zero(30, MpsOptions::default())
            .unwrap()
            .to_statevector()
            .is_err()
    );
    // Angles are checked after evaluation, and before any gate runs.
    let mut huge = Circuit::new(2);
    let t = huge.new_parameter();
    huge.h(0).rx(0, t * 1e300).cx(0, 1);
    let mut m = Mps::zero(2, MpsOptions::default()).unwrap();
    assert!(m.apply_circuit(&huge, &[1e300]).is_err());
    assert!((m.expectation_str("Z0").unwrap() - 1.0).abs() < 1e-15);
    // Samples are bounded by their bits, not only by the work.
    let mut wide = Mps::zero(1 << 16, MpsOptions::default()).unwrap();
    assert!(wide.sample(1 << 13, &mut rng).is_err());
    assert!(wide.sample_counts(1 << 13, &mut rng).is_err());
}

#[test]
fn exact_routing_keeps_the_bound_at_one() {
    // 20 CZ gates between the ends of |+⟩^2000 (the identity in all): every
    // split is exact, so rounding must not add up to a loss of fidelity.
    let n = 2000;
    let mut m = Mps::zero(n, MpsOptions::default()).unwrap();
    for q in 0..n {
        m.h(q).unwrap();
    }
    for _ in 0..20 {
        m.cz(0, n - 1).unwrap();
    }
    assert_eq!(m.truncations(), 0);
    assert_eq!(m.max_bond_dimension(), 1);
    assert!(m.fidelity_lower_bound().unwrap() > 1.0 - 1e-12);
    assert!((m.expectation_str("X0").unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn programs_void_the_bound_and_count_their_bodies() {
    use crate::quantum::Program;
    // A truncation after the first measurement, then more measurements:
    // the exact state depends on the outcomes, so there is no bound.
    let p = Program::from_qasm(
        "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[3]; creg c[3]; h q[0]; \
         measure q[0] -> c[0]; if(c==1) h q[1]; if(c==1) cx q[1],q[2]; \
         measure q[1] -> c[1]; measure q[2] -> c[2];",
    )
    .unwrap();
    let options = MpsOptions {
        max_bond: 1,
        cutoff: 0.0,
    };
    let (counts, bound) = program::sample_program(&p, 200, 1, options).unwrap();
    assert_eq!(counts.values().sum::<usize>(), 200);
    assert_eq!(bound, None);
    let (_, exact) = program::sample_program(&p, 200, 1, MpsOptions::default()).unwrap();
    assert!(exact.unwrap() > 1.0 - 1e-12);
    // The work limit counts the operations inside `if` bodies.
    let body = "x q[0]; ".repeat(2000);
    let p = Program::from_qasm(&format!(
        "OPENQASM 3.0; include \"stdgates.inc\"; qubit[1] q; bit[1] c; \
         c[0] = measure q[0]; if (c[0] == false) {{ {body} }}"
    ))
    .unwrap();
    assert!(program::sample_program(&p, 600_000, 1, MpsOptions::default()).is_err());
}
