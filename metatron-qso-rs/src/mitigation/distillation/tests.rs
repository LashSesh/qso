use super::*;
use crate::Rng;
use crate::noise::NoiseModel;
use crate::photonics::haar_unitary;
use crate::quantum::{MAX_QUBITS, StateVector};

/// `Σ_i p_i |v_i⟩⟨v_i|` with Haar-random orthonormal `v_i`, and the `v_i`.
fn mixed_state(n: usize, weights: &[f64], seed: u64) -> (DensityMatrix, Vec<StateVector>) {
    let dim = 1usize << n;
    let u = haar_unitary(dim, &mut Rng::seed_from_u64(seed)).unwrap();
    let mut rho = DensityMatrix::from_data(n, vec![C64::new(0.0, 0.0); dim * dim]).unwrap();
    let mut vectors = Vec::new();
    for (i, &w) in weights.iter().enumerate() {
        let v = StateVector::from_amplitudes((0..dim).map(|r| u[(r, i)]).collect()).unwrap();
        rho.add_scaled(&DensityMatrix::from_pure(&v).unwrap(), w);
        vectors.push(v);
    }
    (rho, vectors)
}

fn observable() -> PauliSum {
    "0.7 Z0 Z1 - 0.4 X0 Y2 + 0.3 Y1 + 0.2".parse().unwrap()
}

#[test]
fn distilled_values_follow_the_eigenvalue_powers() {
    // Tr(ρᵏO)/Tr(ρᵏ) = Σ p_iᵏ ⟨v_i|O|v_i⟩ / Σ p_iᵏ.
    let weights = [0.6, 0.25, 0.1, 0.05];
    let (rho, vectors) = mixed_state(3, &weights, 3);
    let o = observable();
    let values: Vec<f64> = vectors.iter().map(|v| o.expectation(v).unwrap()).collect();
    for k in 1..=3 {
        let want: f64 = weights
            .iter()
            .zip(&values)
            .map(|(p, v)| p.powi(k as i32) * v)
            .sum::<f64>()
            / weights.iter().map(|p| p.powi(k as i32)).sum::<f64>();
        let got = distilled_expectation(&rho, &o, k).unwrap();
        assert!((got - want).abs() < 1e-12, "k = {k}: {got} vs {want}");
    }
    assert!(
        (distilled_expectation(&rho, &o, 1).unwrap() - rho.expectation(&o).unwrap()).abs() < 1e-12
    );
    // The dominant eigenvector is the limit.
    assert!((dominant_expectation(&rho, &o).unwrap() - values[0]).abs() < 1e-12);
    // A pure state is its own purification.
    let (pure, v) = mixed_state(2, &[1.0], 8);
    let o2: PauliSum = "X0 X1 + 0.5 Z1".parse().unwrap();
    for k in 1..=3 {
        let got = distilled_expectation(&pure, &o2, k).unwrap();
        assert!((got - o2.expectation(&v[0]).unwrap()).abs() < 1e-12);
    }
}

#[test]
fn the_circuit_estimator_equals_the_exact_value() {
    // The settings applied to ρ^{⊗k}: the estimator from exact outcome
    // probabilities equals Tr(ρᵏO)/Tr(ρᵏ) for 1, 2 and 3 copies.
    for (n, k, seed) in [(3, 1, 1), (3, 2, 2), (4, 2, 3), (3, 3, 4), (2, 3, 5)] {
        let (rho, _) = mixed_state(n, &[0.5, 0.3, 0.15, 0.05], seed);
        let o: PauliSum = if n == 2 {
            "0.9 Z0 Z1 + 0.5 X0 X1 - 0.3 Y0 + 0.1".parse().unwrap()
        } else {
            observable()
        };
        let circuits = DistillationCircuits::new(&Circuit::new(n), &o, k).unwrap();
        let got = circuits.evaluate_on(&rho).unwrap();
        let want = distilled_expectation(&rho, &o, k).unwrap();
        assert!(
            (got - want).abs() < 1e-11,
            "n = {n}, k = {k}: {got} vs {want}"
        );
    }
}

/// A noisy preparation: a 3-qubit GHZ-like circuit with rotations.
fn preparation() -> Circuit {
    let mut c = Circuit::new(3);
    c.h(0).cx(0, 1).cx(1, 2).ry(1, 0.3).rz(2, 0.7);
    c
}

#[test]
fn whole_circuits_and_shots_estimate_the_distilled_value() {
    // Running the full circuits (preparation on every copy, noise on every
    // gate of the preparation only) gives the exact distilled value of the
    // noisy state; sampled shots agree within their standard error.
    let prep = preparation();
    let noise = NoiseModel::depolarizing(0.02, 0.05).unwrap();
    let o: PauliSum = "Z0 Z1 + 0.5 X0 X1 X2 - 0.3 Y2".parse().unwrap();
    let mut rho = DensityMatrix::zero(3).unwrap();
    rho.apply_circuit(&prep, &[], Some(&noise)).unwrap();
    let want = distilled_expectation(&rho, &o, 2).unwrap();
    let circuits = DistillationCircuits::new(&prep, &o, 2).unwrap();
    assert_eq!(circuits.settings().len(), 4);
    // Preparation with noise on each copy, then the noiseless measurement
    // part: the same as the product state.
    let mut probabilities = Vec::new();
    for s in circuits.settings() {
        let mut copies = DensityMatrix::zero(6).unwrap();
        for c in 0..2 {
            let mut moved = Circuit::new(6);
            moved
                .append_mapped(&prep, &[3 * c, 3 * c + 1, 3 * c + 2])
                .unwrap();
            copies.apply_circuit(&moved, &[], Some(&noise)).unwrap();
        }
        copies.apply_circuit(&s.measurement, &[], None).unwrap();
        probabilities.push(copies.probabilities());
    }
    let exact = circuits.evaluate(&probabilities).unwrap();
    assert!((exact - want).abs() < 1e-11, "{exact} vs {want}");
    // Shots.
    let mut rng = Rng::seed_from_u64(9);
    let shots = 40_000;
    let counts: Vec<Counts> = probabilities
        .iter()
        .map(|p| {
            let mut acc = 0.0;
            let cumulative: Vec<f64> = p
                .iter()
                .map(|x| {
                    acc += x;
                    acc
                })
                .collect();
            let mut c = Counts::new();
            for _ in 0..shots {
                *c.entry(crate::rng::sample_cumulative(&cumulative, &mut rng))
                    .or_insert(0) += 1;
            }
            c
        })
        .collect();
    let e = circuits.estimate(&counts).unwrap();
    assert!(e.std_error > 0.0 && e.std_error < 0.05, "{e:?}");
    assert!(
        (e.value - want).abs() < 4.0 * e.std_error,
        "{e:?} vs {want}"
    );
    assert_eq!(e.samples, 4 * shots);
}

#[test]
fn distillation_suppresses_depolarising_errors() {
    // The noiseless value of Z0 Z1 + X0 X1 X2 on the GHZ-like state; the
    // error of the noisy value shrinks with every copy.
    let prep = preparation();
    let o: PauliSum = "Z0 Z1 + X0 X1 X2".parse().unwrap();
    let ideal = o
        .expectation(&StateVector::from_circuit(&prep, &[]).unwrap())
        .unwrap();
    let noise = NoiseModel::depolarizing(0.01, 0.04).unwrap();
    let mut rho = DensityMatrix::zero(3).unwrap();
    rho.apply_circuit(&prep, &[], Some(&noise)).unwrap();
    let errors: Vec<f64> = (1..=3)
        .map(|k| (distilled_expectation(&rho, &o, k).unwrap() - ideal).abs())
        .collect();
    assert!(
        errors[1] < 0.2 * errors[0] && errors[2] < 0.5 * errors[1],
        "{errors:?}"
    );
    // Here the ideal state stays an eigenvector of ρ (the only non-Clifford
    // gate comes last), so nothing remains in the limit.
    assert!((dominant_expectation(&rho, &o).unwrap() - ideal).abs() < 1e-12);
    // With the rotation before the entangling gates the dominant
    // eigenvector moves (Koczor's coherent mismatch): the error stops at a
    // floor of about 9·10⁻⁴.
    let mut early = Circuit::new(3);
    early.ry(0, 0.3).h(0).cx(0, 1).cx(1, 2);
    let ideal = o
        .expectation(&StateVector::from_circuit(&early, &[]).unwrap())
        .unwrap();
    let mut rho = DensityMatrix::zero(3).unwrap();
    rho.apply_circuit(&early, &[], Some(&noise)).unwrap();
    let errors: Vec<f64> = (1..=3)
        .map(|k| (distilled_expectation(&rho, &o, k).unwrap() - ideal).abs())
        .collect();
    let floor = (dominant_expectation(&rho, &o).unwrap() - ideal).abs();
    assert!(floor > 5e-4 && floor < 2e-3, "{floor}");
    assert!(
        errors[1] < 0.05 * errors[0] && errors[2] < errors[1],
        "{errors:?}"
    );
    assert!(errors[2] < 2.0 * floor, "{errors:?} {floor}");
}

#[test]
fn coherent_errors_pass_through_unchanged() {
    // An over-rotated but pure state is its own dominant eigenvector:
    // every number of copies returns its (wrong) value.
    let mut ideal = Circuit::new(2);
    ideal.h(0).cx(0, 1);
    let mut over = Circuit::new(2);
    over.ry(0, std::f64::consts::FRAC_PI_2 + 0.3).cx(0, 1);
    let o: PauliSum = "X0 X1 + Z0 Z1".parse().unwrap();
    let want = o
        .expectation(&StateVector::from_circuit(&over, &[]).unwrap())
        .unwrap();
    let truth = o
        .expectation(&StateVector::from_circuit(&ideal, &[]).unwrap())
        .unwrap();
    let mut rho = DensityMatrix::zero(2).unwrap();
    rho.apply_circuit(&over, &[], None).unwrap();
    for k in 1..=3 {
        let got = distilled_expectation(&rho, &o, k).unwrap();
        assert!(
            (got - want).abs() < 1e-12 && (got - truth).abs() > 0.04,
            "{k}: {got}"
        );
    }
}

#[test]
fn diagonalising_gates_are_exportable_two_level_rotations() {
    // The two-level decomposition reproduces random unitaries on 1 to 3
    // qubits (phases included).
    let mut rng = Rng::seed_from_u64(31);
    for k in 1..=3 {
        let dim = 1usize << k;
        let w = haar_unitary(dim, &mut rng).unwrap();
        let targets: Vec<usize> = (0..k).map(|c| 2 * c + 1).collect();
        let mut c = Circuit::new(2 * k);
        append_two_level(&mut c, &w, &targets).unwrap();
        for col in 0..dim {
            // Basis state with local index `col` (targets[0] most significant).
            let mut amps = vec![C64::new(0.0, 0.0); 1 << (2 * k)];
            let index = |local: usize| {
                (0..k)
                    .map(|j| ((local >> (k - 1 - j)) & 1) << targets[j])
                    .sum::<usize>()
            };
            amps[index(col)] = C64::new(1.0, 0.0);
            let mut state = StateVector::from_amplitudes(amps).unwrap();
            state.apply_circuit(&c, &[]).unwrap();
            for row in 0..dim {
                let got = state.amplitudes()[index(row)];
                assert!(
                    (got - w[(row, col)]).norm() < 1e-12,
                    "k = {k} ({row}, {col})"
                );
            }
        }
    }
    // The measurement circuits export to OpenQASM 2 and 3.
    let o: PauliSum = "X0 Y1 + Z0".parse().unwrap();
    let mut prep = Circuit::new(2);
    prep.h(0);
    for k in 2..=3 {
        let vd = DistillationCircuits::new(&prep, &o, k).unwrap();
        for s in vd.settings() {
            assert!(crate::quantum::to_qasm3(&s.circuit, &[], true).is_ok());
            assert!(crate::quantum::to_qasm2(&s.circuit, &[], true).is_ok());
        }
    }
}

#[test]
fn malformed_results_are_rejected() {
    let mut prep = Circuit::new(2);
    prep.h(0).cx(0, 1);
    let o: PauliSum = "Z0 Z1 + 0.5 X0".parse().unwrap();
    let vd = DistillationCircuits::new(&prep, &o, 2).unwrap();
    let n = vd.settings().len();
    assert!(vd.evaluate(&vec![vec![1.0]; n]).is_err());
    assert!(vd.evaluate(&vec![vec![0.0; 16]; n]).is_err());
    let mut bad = vec![0.0; 16];
    bad[0] = f64::NAN;
    assert!(vd.evaluate(&vec![bad; n]).is_err());
    let mut wide = Counts::new();
    wide.insert(0b1011 << 4, 10);
    let fine: Counts = [(0usize, 10usize)].into_iter().collect();
    let mut counts = vec![fine; n];
    counts[1] = wide;
    assert!(vd.estimate(&counts).is_err());
    // Deterministic outcomes give a zero standard error.
    let one: Counts = [(0usize, 100usize)].into_iter().collect();
    let e = DistillationCircuits::new(&Circuit::new(2), &"Z0".parse().unwrap(), 2)
        .unwrap()
        .estimate(&[one.clone(), one])
        .unwrap();
    assert_eq!((e.value, e.std_error), (1.0, 0.0));
}

#[test]
fn invalid_inputs_are_rejected() {
    let rho = DensityMatrix::zero(2).unwrap();
    let o: PauliSum = "Z0".parse().unwrap();
    assert!(distilled_expectation(&rho, &o, 0).is_err());
    assert!(distilled_expectation(&rho, &o, 4).is_err());
    assert!(distilled_expectation(&rho, &"Z2".parse().unwrap(), 2).is_err());
    // |00⟩ has a unique dominant eigenvector; I/4 has none.
    assert!(dominant_expectation(&rho, &o).is_ok());
    assert!(dominant_expectation(&DensityMatrix::maximally_mixed(2).unwrap(), &o).is_err());
    assert!(DistillationCircuits::new(&Circuit::new(0), &o, 2).is_err());
    assert!(DistillationCircuits::new(&Circuit::new(1), &"Z1".parse().unwrap(), 2).is_err());
    assert!(DistillationCircuits::new(&Circuit::new(MAX_QUBITS / 3 + 1), &o, 3).is_err());
    let circuits = DistillationCircuits::new(&Circuit::new(2), &o, 2).unwrap();
    assert!(circuits.evaluate(&[vec![1.0]]).is_err());
    assert!(
        circuits
            .evaluate_on(&DensityMatrix::zero(3).unwrap())
            .is_err()
    );
}
