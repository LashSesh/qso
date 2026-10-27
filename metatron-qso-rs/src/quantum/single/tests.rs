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
    // The zero vector reads zeros, as a StateVector does.
    let mut z = StateVector32::zero(2).unwrap();
    z.amplitudes_mut()[0] = C32::new(0.0, 0.0);
    assert!(z.probabilities().iter().all(|&p| p == 0.0));
    assert_eq!(z.probability_one(0).unwrap(), 0.0);
    assert_eq!(z.fidelity_with(&StateVector::zero(2)).unwrap(), 0.0);
    let mut d = StateVector32::basis(1, 1).unwrap();
    d.apply_diagonal_evolution(&[0.0, 1.0], 0.5).unwrap();
    assert!((d.amplitudes()[1].arg() + 0.5).abs() < 1e-7);
}

/// Bit patterns, with every NaN mapped to one pattern (its sign and
/// payload are not specified, and Miri chooses them at random).
fn bits(v: &[C32]) -> Vec<(u32, u32)> {
    let b = |x: f32| {
        if x.is_nan() {
            f32::NAN.to_bits()
        } else {
            x.to_bits()
        }
    };
    v.iter().map(|z| (b(z.re), b(z.im))).collect()
}

#[test]
fn vector_primitives_round_like_the_portable_ones() {
    use crate::quantum::scalar::portable;
    use crate::quantum::simd::single;
    let mut rng = Rng::seed_from_u64(31);
    // Random values with signed zeros, subnormals and large magnitudes
    // mixed in; results are compared as bit patterns.
    let special = [0.0f32, -0.0, 1e-40, -3e38, 1.0];
    let c = |rng: &mut Rng| {
        let part = |rng: &mut Rng| {
            if rng.below(6) == 0 {
                special[rng.below(5) as usize]
            } else {
                rng.normal() as f32
            }
        };
        C32::new(part(rng), part(rng))
    };
    for len in [0usize, 1, 3, 4, 7, 16, 33] {
        let a: Vec<C32> = (0..len).map(|_| c(&mut rng)).collect();
        let b: Vec<C32> = (0..len).map(|_| c(&mut rng)).collect();
        let m: [C32; 4] = std::array::from_fn(|_| c(&mut rng));
        let r: [f32; 4] = std::array::from_fn(|_| rng.normal() as f32);
        let m4: [C32; 16] = std::array::from_fn(|_| c(&mut rng));
        let (mut a1, mut b1, mut a2, mut b2) = (a.clone(), b.clone(), a.clone(), b.clone());
        single::mat2_runs(&mut a1, &mut b1, &m);
        portable::mat2_runs(&mut a2, &mut b2, &m);
        assert_eq!((bits(&a1), bits(&b1)), (bits(&a2), bits(&b2)));
        let (mut a1, mut b1, mut a2, mut b2) = (a.clone(), b.clone(), a.clone(), b.clone());
        single::mat2_runs_real(&mut a1, &mut b1, &r);
        portable::mat2_runs_real(&mut a2, &mut b2, &r);
        assert_eq!((bits(&a1), bits(&b1)), (bits(&a2), bits(&b2)));
        let (mut a1, mut a2) = (a.clone(), a.clone());
        single::scale_run(&mut a1, m[0]);
        portable::scale_run(&mut a2, m[0]);
        assert_eq!(bits(&a1), bits(&a2));
        let mut runs1 = [a.clone(), b.clone(), a.clone(), b.clone()];
        let mut runs2 = runs1.clone();
        let [p, q, s, t] = &mut runs1;
        single::mat4_runs([p, q, s, t], &m4);
        let [p, q, s, t] = &mut runs2;
        portable::mat4_runs([p, q, s, t], &m4);
        assert_eq!(runs1.map(|r| bits(&r)), runs2.map(|r| bits(&r)));
    }
    for stride in [1usize, 2, 4, 8] {
        for blocks in [1usize, 2, 3, 5] {
            let x: Vec<C32> = (0..2 * stride * blocks).map(|_| c(&mut rng)).collect();
            let m: [C32; 4] = std::array::from_fn(|_| c(&mut rng));
            let r: [f32; 4] = std::array::from_fn(|_| rng.normal() as f32);
            let check = |f: &dyn Fn(&mut [C32]), g: &dyn Fn(&mut [C32])| {
                let (mut p, mut q) = (x.clone(), x.clone());
                f(&mut p);
                g(&mut q);
                assert_eq!(bits(&p), bits(&q), "stride {stride}, {blocks} blocks");
            };
            check(&|v| single::mat2_strided(v, stride, &m), &|v| {
                portable::mat2_strided(v, stride, &m)
            });
            check(&|v| single::mat2_strided_real(v, stride, &r), &|v| {
                portable::mat2_strided_real(v, stride, &r)
            });
            check(&|v| single::scale2_strided(v, stride, m[0], m[1]), &|v| {
                portable::scale2_strided(v, stride, m[0], m[1])
            });
            check(&|v| single::scale_upper_strided(v, stride, m[2]), &|v| {
                portable::scale_upper_strided(v, stride, m[2])
            });
        }
    }
}
