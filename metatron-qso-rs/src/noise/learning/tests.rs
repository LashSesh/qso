use std::collections::BTreeMap;

use super::*;
use crate::noise::{ChannelSpec, GateRule, QubitProperties};
use crate::quantum::StateVector;

fn two_qubit_paulis() -> Vec<PauliString> {
    generators(2, &[(0, 1)])
}

/// A sparse model on qubits 0 and 1 whose rates are equal on Paulis that
/// `cx(0, 1)` exchanges, so its fidelities have the symmetry learning
/// assumes.
fn symmetric_model(seed: u64) -> SparsePauliLindblad {
    let gates = [Gate::Cx(0, 1)];
    let gens = two_qubit_paulis();
    let mut rng = Rng::seed_from_u64(seed);
    let raw: Vec<f64> = gens.iter().map(|_| 0.001 + 0.005 * rng.uniform()).collect();
    let rates = gens
        .iter()
        .zip(&raw)
        .map(|(g, r)| {
            let k = gens
                .iter()
                .position(|h| *h == conjugate(&gates, *g))
                .unwrap();
            (r + raw[k]) / 2.0
        })
        .collect();
    SparsePauliLindblad::new(2, gens, rates).unwrap()
}

/// A device whose `cx` gates carry the model's Pauli channel.
fn device_with(model: &SparsePauliLindblad) -> DeviceModel {
    let letter = |p: Pauli| format!("{p:?}");
    let map: BTreeMap<String, f64> = model
        .error_probabilities()
        .unwrap()
        .into_iter()
        .filter(|(p, _)| !p.is_identity())
        .map(|(p, prob)| (letter(p.get(0)) + &letter(p.get(1)), prob))
        .collect();
    DeviceModel::new("learned")
        .with_rule(GateRule::for_gate("cx").with_channel(ChannelSpec::Pauli(map)))
        .with_idle_noise(false)
}

fn cx_layer(n: usize) -> Circuit {
    let mut layer = Circuit::new(n);
    layer.cx(0, 1);
    layer
}

/// Blocks of random one-qubit rotations.
fn random_blocks(n: usize, count: usize, seed: u64) -> Vec<Circuit> {
    let mut rng = Rng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let mut b = Circuit::new(n);
            for q in 0..n {
                b.ry(q, rng.uniform_range(-3.0, 3.0))
                    .rz(q, rng.uniform_range(-3.0, 3.0));
            }
            b
        })
        .collect()
}

/// `B₀ L B₁ L … B_m` without twirls.
fn joined(blocks: &[Circuit], layer: &Circuit) -> Circuit {
    let mut c = Circuit::new(layer.n_qubits());
    for (i, b) in blocks.iter().enumerate() {
        if i > 0 {
            c.append(layer).unwrap();
        }
        c.append(b).unwrap();
    }
    c
}

fn observable() -> PauliSum {
    PauliSum::from_terms([
        (
            1.0,
            PauliString::new(&[(0, Pauli::Z), (1, Pauli::Z)]).unwrap(),
        ),
        (0.5, PauliString::single(0, Pauli::X)),
        (0.3, PauliString::single(1, Pauli::Z)),
    ])
}

#[test]
fn fidelities_agree_with_the_error_probabilities() {
    let model = symmetric_model(1);
    let probs = model.error_probabilities().unwrap();
    assert!((probs.iter().map(|(_, p)| p).sum::<f64>() - 1.0).abs() < 1e-12);
    for b in two_qubit_paulis() {
        let f: f64 = probs
            .iter()
            .map(|(p, prob)| if p.commutes_with(&b) { *prob } else { -prob })
            .sum();
        assert!((f - model.fidelity(&b)).abs() < 1e-12, "{b}");
    }
    let doubled = model.scaled(2.0).unwrap();
    let b = PauliString::single(0, Pauli::X);
    assert!((doubled.fidelity(&b) - model.fidelity(&b).powi(2)).abs() < 1e-12);
    assert!((model.overhead() - (2.0 * model.rates.iter().sum::<f64>()).exp()).abs() < 1e-12);
}

#[test]
fn conjugation_matches_the_gates() {
    let mut rng = Rng::seed_from_u64(5);
    let mut prep = Circuit::new(2);
    for q in 0..2 {
        prep.ry(q, rng.uniform_range(-3.0, 3.0))
            .rz(q, rng.uniform_range(-3.0, 3.0));
    }
    prep.cx(0, 1).ry(0, 0.4).rx(1, 1.1);
    let psi = StateVector::from_circuit(&prep, &[]).unwrap();
    for (gate, build) in [
        (
            Gate::Cx(0, 1),
            Circuit::cx as fn(&mut Circuit, usize, usize) -> &mut Circuit,
        ),
        (Gate::Cx(1, 0), Circuit::cx),
        (Gate::Cy(0, 1), Circuit::cy),
        (Gate::Cy(1, 0), Circuit::cy),
        (Gate::Cz(0, 1), Circuit::cz),
        (Gate::Swap(0, 1), Circuit::swap),
    ] {
        let (a, b) = pair(gate);
        let mut u = Circuit::new(2);
        build(&mut u, a, b);
        let mut after = psi.clone();
        after.apply_circuit(&u, &[]).unwrap();
        for p in two_qubit_paulis() {
            // ⟨Uψ| U p U† |Uψ⟩ = ⟨ψ|p|ψ⟩, and U p U† = ±conjugate(p).
            let before = PauliSum::from_terms([(1.0, p)]).expectation(&psi).unwrap();
            let q = conjugate(&[gate], p);
            let now = PauliSum::from_terms([(1.0, q)])
                .expectation(&after)
                .unwrap();
            assert!(
                (before.abs() - now.abs()).abs() < 1e-10,
                "{gate:?} {p} → {q}"
            );
        }
    }
}

#[test]
fn nnls_matches_the_constrained_minimum() {
    // Unconstrained solution (1, −1): the constrained minimum has x₂ = 0.
    let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let b = [1.0, -1.0, 0.0];
    let x = nnls(&a, &b);
    // With x₂ = 0: minimise (x₁ − 1)² + x₁², so x₁ = 1/2.
    assert!((x[0] - 0.5).abs() < 1e-12 && x[1] == 0.0, "{x:?}");
    // A feasible unconstrained solution is returned as is.
    let x = nnls(&a, &[1.0, 2.0, 3.0]);
    assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 2.0).abs() < 1e-12);
    // Brute force over a grid agrees on a random problem.
    let mut rng = Rng::seed_from_u64(9);
    let a: Vec<Vec<f64>> = (0..6)
        .map(|_| (0..3).map(|_| rng.uniform_range(-1.0, 1.0)).collect())
        .collect();
    let b: Vec<f64> = (0..6).map(|_| rng.uniform_range(-1.0, 1.0)).collect();
    let x = nnls(&a, &b);
    let cost = |x: &[f64]| -> f64 {
        a.iter()
            .zip(&b)
            .map(|(row, bi)| (row.iter().zip(x).map(|(r, v)| r * v).sum::<f64>() - bi).powi(2))
            .sum()
    };
    assert!(x.iter().all(|v| *v >= 0.0));
    let best = cost(&x);
    let step = 0.01;
    for i in 0..300 {
        for j in 0..300 {
            for k in 0..30 {
                let y = [i as f64 * step, j as f64 * step, k as f64 * 0.1];
                assert!(cost(&y) >= best - 1e-12);
            }
        }
    }
}

#[test]
fn learning_recovers_a_known_model_within_its_standard_errors() {
    let truth = symmetric_model(2);
    let device =
        device_with(&truth).with_default_qubit(QubitProperties::new().with_readout(0.02, 0.04));
    let learned = learn_layer(&cx_layer(3), &device, &LearningOptions::default()).unwrap();
    // Qubits 0..3 with edges (0, 1) and (1, 2): 9 + 18 generators.
    assert_eq!(learned.model.generators.len(), 27);
    let mut worst: f64 = 0.0;
    for ((g, rate), se) in learned
        .model
        .generators
        .iter()
        .zip(&learned.model.rates)
        .zip(&learned.std_errors)
    {
        let expected = truth
            .generators
            .iter()
            .position(|h| h == g)
            .map_or(0.0, |k| truth.rates[k]);
        let z = (rate - expected).abs() / (se + 2e-5);
        worst = worst.max(z);
        assert!(z < 4.0, "{g}: {rate} ± {se}, expected {expected}");
        assert!(*se < 2e-3, "{g}: standard error {se}");
    }
    assert!(worst > 0.0);
    // cx(0, 1) moves X0, Z1, … ; those Paulis are listed as symmetrised.
    assert!(
        learned
            .symmetrised
            .contains(&PauliString::single(0, Pauli::X))
    );
    assert!(
        !learned
            .symmetrised
            .contains(&PauliString::single(0, Pauli::Z))
    );
    assert!(
        !learned
            .symmetrised
            .contains(&PauliString::single(2, Pauli::Y))
    );
    // The fitted products are the true ones.
    for (b, product, se) in &learned.products {
        let want = truth.fidelity(b) * truth.fidelity(&conjugate(&[Gate::Cx(0, 1)], *b));
        assert!(
            (product - want).abs() < 4.0 * se + 1e-4,
            "{b}: {product} vs {want}"
        );
    }
}

#[test]
fn learning_is_reproducible() {
    let device = device_with(&symmetric_model(3));
    let options = LearningOptions {
        depths: vec![2, 8],
        twirls: 2,
        shots: 64,
        ..LearningOptions::default()
    };
    let a = learn_layer(&cx_layer(2), &device, &options).unwrap();
    let b = learn_layer(&cx_layer(2), &device, &options).unwrap();
    assert_eq!(a, b);
}

#[test]
fn error_cancellation_with_the_learned_model_is_unbiased() {
    let truth = symmetric_model(4);
    let device = device_with(&truth);
    let layer = cx_layer(2);
    let learned = learn_layer(&layer, &device, &LearningOptions::default()).unwrap();
    let blocks = random_blocks(2, 4, 11);
    let obs = observable();
    let exact = DensityMatrix::from_circuit(&joined(&blocks, &layer), &[])
        .unwrap()
        .expectation(&obs)
        .unwrap();
    let mut noisy = DensityMatrix::zero(2).unwrap();
    noisy
        .apply_device(&joined(&blocks, &layer), &[], &device)
        .unwrap();
    let noisy = noisy.expectation(&obs).unwrap();
    let pec = pec_layered(&blocks, &layer, &learned.model, &device, &obs, 4000, 7).unwrap();
    let se = pec.estimate.std_error;
    assert_eq!(pec.noisy_operations, 3);
    assert!((pec.overhead - learned.model.overhead().powi(3)).abs() < 1e-12);
    assert!(
        (pec.estimate.value - exact).abs() < 4.0 * se,
        "PEC {} ± {se}, exact {exact}",
        pec.estimate.value
    );
    assert!(
        (noisy - exact).abs() > 6.0 * se,
        "noisy {noisy}, exact {exact}, se {se}"
    );
    // With the true model the estimate is unbiased too.
    let pec = pec_layered(&blocks, &layer, &truth, &device, &obs, 4000, 8).unwrap();
    assert!((pec.estimate.value - exact).abs() < 4.0 * pec.estimate.std_error);
}

#[test]
fn error_amplification_extrapolates_towards_the_exact_value() {
    let truth = symmetric_model(6);
    let device = device_with(&truth);
    let layer = cx_layer(2);
    let blocks = random_blocks(2, 4, 13);
    let obs = observable();
    let exact = DensityMatrix::from_circuit(&joined(&blocks, &layer), &[])
        .unwrap()
        .expectation(&obs)
        .unwrap();
    let zne = pea_layered(
        &blocks,
        &layer,
        &truth,
        &device,
        &obs,
        &[1.0, 2.0, 3.0],
        6000,
        Extrapolation::Richardson,
        3,
    )
    .unwrap();
    let noisy = zne.values[0].value;
    let se = zne.estimate.std_error;
    assert!(
        (zne.estimate.value - exact).abs() < 4.0 * se + 0.1 * (noisy - exact).abs(),
        "PEA {} ± {se}, noisy {noisy}, exact {exact}",
        zne.estimate.value
    );
    assert!((noisy - exact).abs() > 4.0 * se);
    // The values at higher scales are noisier: amplification works.
    assert!((zne.values[2].value - exact).abs() > (noisy - exact).abs());
}

#[test]
fn invalid_inputs_are_rejected() {
    let device = DeviceModel::new("ideal");
    let options = LearningOptions::default();
    let mut h = Circuit::new(2);
    h.h(0);
    assert!(learn_layer(&h, &device, &options).is_err());
    let mut overlap = Circuit::new(3);
    overlap.cx(0, 1).cz(1, 2);
    assert!(learn_layer(&overlap, &device, &options).is_err());
    let odd = LearningOptions {
        depths: vec![2, 3],
        ..LearningOptions::default()
    };
    assert!(learn_layer(&cx_layer(2), &device, &odd).is_err());
    let bad_edges = LearningOptions {
        edges: Some(vec![(0, 2)]),
        ..LearningOptions::default()
    };
    assert!(learn_layer(&cx_layer(2), &device, &bad_edges).is_err());
    assert!(
        SparsePauliLindblad::new(2, vec![PauliString::single(0, Pauli::X)], vec![-0.1]).is_err()
    );
    assert!(
        SparsePauliLindblad::new(
            1,
            vec![
                PauliString::single(0, Pauli::X),
                PauliString::single(0, Pauli::X)
            ],
            vec![0.1, 0.1]
        )
        .is_err()
    );
    let model = symmetric_model(1);
    let obs = observable();
    let blocks = random_blocks(2, 2, 1);
    assert!(pec_layered(&[], &cx_layer(2), &model, &device, &obs, 10, 0).is_err());
    assert!(pec_layered(&blocks, &cx_layer(2), &model, &device, &obs, 0, 0).is_err());
    assert!(
        pea_layered(
            &blocks,
            &cx_layer(2),
            &model,
            &device,
            &obs,
            &[1.0],
            10,
            Extrapolation::Linear,
            0
        )
        .is_err()
    );
    assert!(
        pea_layered(
            &blocks,
            &cx_layer(2),
            &model,
            &device,
            &obs,
            &[0.5, 1.0],
            10,
            Extrapolation::Linear,
            0
        )
        .is_err()
    );
}

#[test]
fn nnls_does_not_depend_on_the_scale_of_rows() {
    let mut rng = Rng::seed_from_u64(21);
    let a: Vec<Vec<f64>> = (0..8)
        .map(|_| (0..6).map(|_| rng.uniform_range(-1.0, 1.0)).collect())
        .collect();
    let b: Vec<f64> = (0..8).map(|_| rng.uniform_range(-1.0, 1.0)).collect();
    let x = nnls(&a, &b);
    assert!(x.iter().filter(|v| **v > 0.0).count() >= 2, "{x:?}");
    for scale in [1e-9, 1e-3, 1e6, 1e12] {
        let sa: Vec<Vec<f64>> = a
            .iter()
            .map(|r| r.iter().map(|v| v * scale).collect())
            .collect();
        let sb: Vec<f64> = b.iter().map(|v| v * scale).collect();
        let y = nnls(&sa, &sb);
        assert!(
            x.iter().zip(&y).all(|(p, q)| (p - q).abs() < 1e-9),
            "{scale}: {y:?}"
        );
    }
    // Rows of very different weight (an exactly measured equation).
    let mut weighted = a.clone();
    weighted[0].iter_mut().for_each(|v| *v *= 1e9);
    let mut wb = b.clone();
    wb[0] *= 1e9;
    let y = nnls(&weighted, &wb);
    assert!(y.iter().all(|v| *v >= 0.0));
    let r0: f64 = weighted[0].iter().zip(&y).map(|(p, q)| p * q).sum::<f64>() - wb[0];
    // The heavy equation is met as well as the constraints allow.
    let x0: f64 = weighted[0].iter().zip(&x).map(|(p, q)| p * q).sum::<f64>() - wb[0];
    assert!(r0.abs() <= x0.abs() + 1e-6);
}

#[test]
fn exactly_measured_generators_do_not_break_the_fit() {
    // No readout errors and idle qubits: the idle generators are measured
    // without any noise, so their weights are far above the others.
    let truth = symmetric_model(2);
    let device = device_with(&truth);
    for (n, seed) in [(3, 1), (4, 2), (5, 1)] {
        let learned = learn_layer(
            &cx_layer(n),
            &device,
            &LearningOptions {
                seed,
                ..LearningOptions::default()
            },
        )
        .unwrap();
        for ((g, rate), se) in learned
            .model
            .generators
            .iter()
            .zip(&learned.model.rates)
            .zip(&learned.std_errors)
        {
            let expected = truth
                .generators
                .iter()
                .position(|h| h == g)
                .map_or(0.0, |k| truth.rates[k]);
            assert!(
                (rate - expected).abs() < 4.0 * se + 1e-6,
                "n {n}, seed {seed}, {g}: {rate} ± {se}, expected {expected}"
            );
        }
    }
}

#[test]
fn deep_noisy_points_do_not_inflate_the_standard_errors() {
    // Depth 36 with few twirls and shots: some bootstrap replicas have
    // non-positive means there.
    let device = device_with(&symmetric_model(2))
        .with_default_qubit(QubitProperties::new().with_readout(0.02, 0.04));
    for seed in [9, 14, 17] {
        let learned = learn_layer(
            &cx_layer(2),
            &device,
            &LearningOptions {
                depths: vec![2, 8, 36],
                twirls: 8,
                shots: 128,
                seed,
                ..LearningOptions::default()
            },
        )
        .unwrap();
        for (b, product, se) in &learned.products {
            assert!(*se < 0.1, "seed {seed}, {b}: {product} ± {se}");
        }
    }
}

#[test]
fn bases_cover_every_coupled_pair() {
    let complete = |n: usize| -> Vec<(usize, usize)> {
        (0..n)
            .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
            .collect()
    };
    // Crown graph on 10 qubits (bipartite; greedy colouring in qubit
    // order needs 5 colours), K4, K5 and a single qubit.
    let crown: Vec<(usize, usize)> = (0..5)
        .flat_map(|i| {
            (0..5)
                .filter(move |&j| j != i)
                .map(move |j| (2 * i, 2 * j + 1))
        })
        .collect();
    for (n, edges, count) in [
        (10, crown, 9),
        (4, complete(4), 9),
        (5, complete(5), 27),
        (13, complete(13), 27),
        (1, vec![], 3),
    ] {
        let b = bases(n, &edges).unwrap();
        assert_eq!(b.len(), count, "{n} qubits");
        for &(p, q) in &edges {
            let mut seen: Vec<(Pauli, Pauli)> = b.iter().map(|row| (row[p], row[q])).collect();
            seen.sort_by_key(|&(x, y)| (x as u8, y as u8));
            seen.dedup();
            assert_eq!(seen.len(), 9, "{p}–{q}");
        }
        for g in generators(n, &edges) {
            assert!(b.iter().any(|row| measurable(&g, row)), "{g}");
        }
    }
    assert!(bases(14, &complete(14)).is_err());
}

#[test]
fn invalid_models_are_rejected_by_the_mitigation() {
    let layer = cx_layer(2);
    let device = DeviceModel::new("ideal");
    let obs = observable();
    let blocks = random_blocks(2, 2, 3);
    let mut short = symmetric_model(1);
    short.rates.truncate(3);
    assert!(pec_layered(&blocks, &layer, &short, &device, &obs, 10, 0).is_err());
    let mut negative = symmetric_model(1);
    negative.rates[0] = -5.0;
    assert!(pec_layered(&blocks, &layer, &negative, &device, &obs, 10, 0).is_err());
    assert!(
        pea_layered(
            &blocks,
            &layer,
            &negative,
            &device,
            &obs,
            &[1.0, 2.0],
            10,
            Extrapolation::Linear,
            0
        )
        .is_err()
    );
    let huge = symmetric_model(1).scaled(1e6).unwrap();
    assert!(pec_layered(&blocks, &layer, &huge, &device, &obs, 10, 0).is_err());
    let options = |twirls, shots, depths: Vec<usize>| LearningOptions {
        twirls,
        shots,
        depths,
        ..LearningOptions::default()
    };
    for o in [
        options(1, 256, vec![2, 4]),
        options(16, 0, vec![2, 4]),
        options(16, MAX_LEARNING_SHOTS + 1, vec![2, 4]),
        options(16, 256, vec![4, 4]),
        options(usize::MAX, 256, vec![2, 4]),
    ] {
        assert!(learn_layer(&layer, &device, &o).is_err(), "{o:?}");
    }
}
