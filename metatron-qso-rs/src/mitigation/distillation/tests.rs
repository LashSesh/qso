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
    // What remains is the coherent mismatch of the dominant eigenvector.
    let limit = (dominant_expectation(&rho, &o).unwrap() - ideal).abs();
    assert!(errors[2] > limit - 1e-12);
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
