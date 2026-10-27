use super::*;

/// A random circuit of every gate kind, on `n` qubits.
fn random_circuit(n: usize, gates: usize, rng: &mut Rng) -> Circuit {
    let mut c = Circuit::new(n);
    for _ in 0..gates {
        let a = rng.below(n as u64) as usize;
        let mut b = rng.below(n as u64) as usize;
        if b == a {
            b = (a + 1) % n;
        }
        let t = rng.uniform_range(-3.0, 3.0);
        match rng.below(12) {
            0 => c.h(a),
            1 => c.rx(a, t),
            2 => c.ry(a, t),
            3 => c.rz(a, t),
            4 => c.cx(a, b),
            5 => c.cz(a, b),
            6 => c.t(a),
            7 => c.s(a),
            8 => c.swap(a, b),
            9 => c.rzz(a, b, t),
            10 => c.crx(a, b, t),
            _ => c.x(a),
        };
    }
    c
}

#[test]
fn single_precision_follows_double_precision() {
    let mut rng = Rng::seed_from_u64(9);
    for (n, gates) in [(3, 50), (8, 400), (12, 1000), (16, 600)] {
        let c = random_circuit(n, gates, &mut rng);
        let exact = StateVector::from_circuit(&c, &[]).unwrap();
        let single = StateVector32::from_circuit(&c, &[]).unwrap();
        let f = single.fidelity_with(&exact).unwrap();
        eprintln!(
            "n = {n}, {gates} gates: 1 − F = {:.2e}, ‖ψ‖ − 1 = {:.2e}",
            1.0 - f,
            single.norm() - 1.0
        );
        // Rounding of about 2⁻²⁴ per gate and amplitude.
        assert!(
            1.0 - f < 1e-10 * gates as f64,
            "n = {n}: 1 − F = {}",
            1.0 - f
        );
        assert!((single.norm() - 1.0).abs() < 1e-5, "{}", single.norm());
        let h: PauliSum = "0.5 Z0 Z1 + 0.3 X0 Y2 - 0.7 X1".parse().unwrap();
        let e64 = h.expectation(&exact).unwrap();
        let e32 = single.expectation(&h).unwrap();
        assert!((e64 - e32).abs() < 1e-5, "{e64} {e32}");
        // Gate by gate, the same kernels.
        let mut step = StateVector32::zero(n).unwrap();
        for inst in c.instructions() {
            step.apply_instruction(inst, &[]).unwrap();
        }
        assert!(1.0 - step.fidelity_with(&exact).unwrap() < 1e-10 * gates as f64);
    }
}

#[test]
fn single_precision_states_convert_sample_and_measure() {
    let mut rng = Rng::seed_from_u64(2);
    let psi = StateVector::random(5, &mut rng);
    let s = StateVector32::from_state(&psi);
    assert!(1.0 - s.fidelity_with(&psi).unwrap() < 1e-12);
    let back = s.to_state().unwrap();
    assert!((back.norm() - 1.0).abs() < 1e-6);
    let p = s.probabilities();
    assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    let counts = s.sample(20_000, &mut rng);
    for (&k, &c) in &counts {
        let expected = p[k] * 20_000.0;
        assert!(
            (c as f64 - expected).abs() < 6.0 * expected.sqrt() + 3.0,
            "{k}"
        );
    }
    let mut m = StateVector32::zero(3).unwrap();
    let mut c = Circuit::new(3);
    c.h(0).cx(0, 1).cx(1, 2);
    m.apply_circuit(&c, &[]).unwrap();
    let first = m.measure(1, &mut rng).unwrap();
    let k = if first { 7 } else { 0 };
    assert!((m.probabilities()[k] - 1.0).abs() < 1e-6);
    // Limits and mismatches.
    assert!(StateVector32::zero(MAX_QUBITS_SINGLE + 1).is_err());
    assert!(StateVector32::basis(2, 4).is_err());
    assert!(m.probability_one(3).is_err());
    let big = Circuit::new(4);
    assert!(m.apply_circuit(&big, &[]).is_err());
    assert!(m.inner_with(&StateVector::zero(2)).is_err());
    assert!(m.apply_diagonal_evolution(&[0.0; 3], 1.0).is_err());
    let mut d = StateVector32::basis(1, 1).unwrap();
    d.apply_diagonal_evolution(&[0.0, 1.0], 0.5).unwrap();
    assert!((d.amplitudes()[1].arg() + 0.5).abs() < 1e-7);
}
