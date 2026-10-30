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

#[test]
fn nnls_on_the_normal_equations_agrees_with_the_dense_solver() {
    let mut rng = Rng::seed_from_u64(21);
    for (m, k) in [(12, 8), (40, 30), (90, 90), (150, 120)] {
        let a: Vec<Vec<f64>> = (0..m)
            .map(|_| (0..k).map(|_| rng.uniform_range(-1.0, 1.0)).collect())
            .collect();
        // A right-hand side whose unconstrained solution has both signs.
        let truth: Vec<f64> = (0..k).map(|_| rng.uniform_range(-1.0, 1.0)).collect();
        let b: Vec<f64> = a
            .iter()
            .map(|row| {
                row.iter().zip(&truth).map(|(r, t)| r * t).sum::<f64>() + 0.01 * rng.uniform()
            })
            .collect();
        let dense = nnls_dense(&a, &b);
        let gram = nnls_gram(&a, &b);
        let scale = dense.iter().copied().fold(0.0, f64::max);
        assert!(
            dense.iter().filter(|v| **v == 0.0).count() > k / 4,
            "{m}×{k}"
        );
        for (d, g) in dense.iter().zip(&gram) {
            assert!(
                *g >= 0.0 && (d - g).abs() < 1e-8 * scale,
                "{m}×{k}: {d} vs {g}"
            );
        }
    }
    // A column that is the sum of two others, with a solution that wants
    // all three positive: the Gram matrix of the support has no Cholesky
    // factor, and the dense path answers.
    let a: Vec<Vec<f64>> = (0..80)
        .map(|i| {
            let mut row: Vec<f64> = (0..70).map(|j| ((i * j + j) as f64 * 0.37).cos()).collect();
            row[2] = row[0] + row[1];
            row
        })
        .collect();
    let b: Vec<f64> = a
        .iter()
        .enumerate()
        .map(|(i, row)| 0.5 * row.iter().sum::<f64>() + 0.01 * (i as f64).sin())
        .collect();
    let x = nnls(&a, &b);
    let y = nnls_dense(&a, &b);
    let residual = |x: &[f64]| -> f64 {
        a.iter()
            .zip(&b)
            .map(|(row, bi)| (row.iter().zip(x).map(|(r, v)| r * v).sum::<f64>() - bi).powi(2))
            .sum()
    };
    assert!((residual(&x) - residual(&y)).abs() < 1e-9 * residual(&y).max(1.0));
}

/// A device with Pauli noise of several kinds on four qubits: two-qubit
/// depolarising on `cx` and `cz`, a Pauli channel and a ZZ crosstalk on a
/// spectator, one-qubit depolarising on `h` and readout errors.
fn pauli_device() -> DeviceModel {
    let map = |pairs: &[(&str, f64)]| -> BTreeMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    };
    DeviceModel::new("pauli")
        .with_rule(
            GateRule::for_gate("cx")
                .with_depolarizing(0.03)
                .with_spectator(&[0, 2], ChannelSpec::Pauli(map(&[("ZZ", 0.02)]))),
        )
        .with_rule(
            GateRule::for_gate("cz")
                .with_depolarizing(0.02)
                .with_channel(ChannelSpec::Pauli(map(&[("XI", 0.01), ("ZY", 0.015)]))),
        )
        .with_rule(GateRule::for_gate("h").with_depolarizing(0.01))
        .with_default_qubit(QubitProperties::new().with_readout(0.02, 0.05))
        .with_idle_noise(false)
}

fn two_gate_layer() -> Circuit {
    let mut layer = Circuit::new(4);
    layer.cx(0, 1).cz(2, 3);
    layer
}

#[test]
fn pauli_frames_sample_the_density_matrix_distribution() {
    let device = pauli_device();
    let layer = two_gate_layer();
    let gates = layer_gates(&layer, 4, "test").unwrap();
    let mut rng = Rng::seed_from_u64(3);
    // A learning circuit: Y, X, Z, X basis, four twirled layers, readout
    // flips on qubits 1 and 2.
    let mut c = Circuit::new(4);
    c.h(0).s(0).h(1).h(3);
    for _ in 0..4 {
        push_twirled_layer(&mut c, &layer, &gates, &mut rng);
    }
    c.sdg(0).h(0).h(1).h(3).x(1).x(2);
    let shots = 400_000;
    let program = Program::compile(&c, &[], &device).unwrap();
    let readout = device.readout_errors(4);
    let mut counts: BTreeMap<u64, usize> = BTreeMap::new();
    let mut chunks = 0;
    frame_shots(&program, shots, &readout, &mut rng, |bits| {
        chunks += 1;
        for &b in bits {
            *counts.entry(b).or_insert(0) += 1;
        }
    })
    .unwrap();
    assert_eq!(chunks, shots.div_ceil(FRAME_CHUNK));
    assert_eq!(counts.values().sum::<usize>(), shots);
    let mut rho = DensityMatrix::zero(4).unwrap();
    rho.apply_device(&c, &[], &device).unwrap();
    let ideal = rho.probabilities();
    let mut worst: f64 = 0.0;
    for y in 0..16usize {
        let p: f64 = (0..16usize)
            .map(|x| {
                ideal[x]
                    * (0..4)
                        .map(|q| {
                            let (p01, p10) = readout[q];
                            match ((x >> q) & 1, (y >> q) & 1) {
                                (0, 0) => 1.0 - p01,
                                (0, 1) => p01,
                                (1, 0) => p10,
                                _ => 1.0 - p10,
                            }
                        })
                        .product::<f64>()
            })
            .sum();
        let f = counts.get(&(y as u64)).copied().unwrap_or(0) as f64 / shots as f64;
        let se = (p * (1.0 - p) / shots as f64).sqrt() + 1e-6;
        worst = worst.max((f - p).abs() / se);
    }
    assert!(worst < 4.5, "worst deviation {worst} standard errors");
    // A non-Clifford gate cannot carry a frame.
    let mut t = Circuit::new(1);
    t.t(0);
    let program = Program::compile(&t, &[], &DeviceModel::new("ideal")).unwrap();
    assert!(frame_shots(&program, 4, &[], &mut rng, |_| {}).is_err());
}

#[test]
fn frames_and_density_matrices_learn_the_same_model() {
    let device = pauli_device();
    let layer = two_gate_layer();
    let options = LearningOptions {
        edges: Some(vec![(0, 1), (2, 3), (0, 2)]),
        ..LearningOptions::default()
    };
    let dm = learn_layer_with(&layer, &device, &options, LearningSimulator::DensityMatrix).unwrap();
    let frames = learn_layer_with(&layer, &device, &options, LearningSimulator::Frames).unwrap();
    assert_eq!(dm.model.generators, frames.model.generators);
    let mut worst: f64 = 0.0;
    for k in 0..dm.model.rates.len() {
        let se = dm.std_errors[k].hypot(frames.std_errors[k]) + 2e-5;
        worst = worst.max((dm.model.rates[k] - frames.model.rates[k]).abs() / se);
    }
    assert!(worst < 4.5, "worst difference {worst} standard errors");
    // The crosstalk ZZ on (0, 2): λ = −ln(1 − 2·0.02)/2.
    let zz = PauliString::new(&[(0, Pauli::Z), (2, Pauli::Z)]).unwrap();
    let k = frames
        .model
        .generators
        .iter()
        .position(|g| *g == zz)
        .unwrap();
    let want = -(1.0f64 - 0.04).ln() / 2.0;
    assert!(
        (frames.model.rates[k] - want).abs() < 4.0 * frames.std_errors[k] + 2e-5,
        "{} ± {}, expected {want}",
        frames.model.rates[k],
        frames.std_errors[k]
    );
    // Auto picks density matrices here: the same result.
    let auto = learn_layer(&layer, &device, &options).unwrap();
    assert_eq!(auto, dm);
}

#[test]
fn learning_and_mitigation_run_above_ten_qubits() {
    let truth = symmetric_model(31);
    let device = device_with(&truth);
    let layer = cx_layer(12);
    let options = LearningOptions::default();
    // Exact density matrices stop at ten qubits; frames go on.
    assert!(learn_layer_with(&layer, &device, &options, LearningSimulator::DensityMatrix).is_err());
    let learned = learn_layer(&layer, &device, &options).unwrap();
    assert_eq!(learned.model.generators.len(), 12 * 3 + 11 * 9);
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
            (rate - expected).abs() < 5.0 * se + 3e-5,
            "{g}: {rate} ± {se}, expected {expected}"
        );
    }
    // Error cancellation and amplification on trajectories.
    let blocks = random_blocks(12, 4, 17);
    let obs = observable();
    let exact = StateVector::from_circuit(&joined(&blocks, &layer), &[])
        .and_then(|psi| obs.expectation(&psi))
        .unwrap();
    let noisy = TrajectorySimulator::new(device.clone(), 8000, 5)
        .unwrap()
        .expectation(&joined(&blocks, &layer), &[], &obs)
        .unwrap();
    let pec = pec_layered(&blocks, &layer, &learned.model, &device, &obs, 8000, 8).unwrap();
    let se = pec.estimate.std_error;
    assert!(
        (pec.estimate.value - exact).abs() < 4.0 * se,
        "PEC {} ± {se}, exact {exact}",
        pec.estimate.value
    );
    assert!(
        (noisy.value - exact).abs() > 4.0 * (se + noisy.std_error),
        "noisy {} ± {}, exact {exact}",
        noisy.value,
        noisy.std_error
    );
    let zne = pea_layered(
        &blocks,
        &layer,
        &learned.model,
        &device,
        &obs,
        &[1.0, 2.0],
        8000,
        Extrapolation::Linear,
        9,
    )
    .unwrap();
    assert!((zne.values[1].value - exact).abs() > (zne.values[0].value - exact).abs());
    // Matrix-product samples agree; exact ones need no truncation.
    let mps = pec_layered_with(
        &blocks,
        &layer,
        &learned.model,
        &device,
        &obs,
        4000,
        8,
        LayeredSimulator::Mps(MpsOptions::exact()),
    )
    .unwrap();
    assert!(
        (mps.estimate.value - exact).abs() < 4.0 * mps.estimate.std_error,
        "MPS PEC {:?}, exact {exact}",
        mps.estimate
    );
    // Density matrices stop at ten qubits, frames and matrix-product
    // states at 64.
    assert!(
        pec_layered_with(
            &blocks,
            &layer,
            &learned.model,
            &device,
            &obs,
            10,
            1,
            LayeredSimulator::DensityMatrix
        )
        .is_err()
    );
    assert!(learn_layer(&cx_layer(65), &device, &options).is_err());
    let model = SparsePauliLindblad::new(64, Vec::new(), Vec::new()).unwrap();
    let blocks = random_blocks(64, 2, 1);
    let r = pec_layered(&blocks, &cx_layer(64), &model, &device, &obs, 4, 1).unwrap();
    assert!(r.estimate.value.is_finite());
}

#[test]
fn matrix_product_samples_draw_pauli_and_unitary_terms() {
    // Depolarising and a coherent rotation on cx: the mean over drawn
    // terms is the density matrix value.
    let device = DeviceModel::new("mixed")
        .with_rule(
            GateRule::for_gate("cx")
                .with_depolarizing(0.1)
                .with_channel(ChannelSpec::Rotation {
                    pauli: "ZX".into(),
                    angle: 0.3,
                }),
        )
        .with_idle_noise(false);
    let blocks = random_blocks(3, 3, 4);
    let mut layer = Circuit::new(3);
    layer.cx(1, 2);
    let c = joined(&blocks, &layer);
    let obs = observable();
    let mut rho = DensityMatrix::zero(3).unwrap();
    rho.apply_device(&c, &[], &device).unwrap();
    let want = rho.expectation(&obs).unwrap();
    let base = Rng::seed_from_u64(2);
    let values: Vec<f64> = (0..6000)
        .map(|s| {
            let mut rng = base.fork(s);
            mps_sample(&c, &device, &obs, MpsOptions::exact(), true, &mut rng).unwrap()
        })
        .collect();
    let e = Estimate::from_samples(&values);
    assert!(
        (e.value - want).abs() < 4.0 * e.std_error,
        "{e:?} vs {want}"
    );
    // Damping has no such terms; truncation is refused when strict.
    let damped = DeviceModel::new("damped")
        .with_rule(GateRule::for_gate("cx").with_channel(ChannelSpec::AmplitudeDamping(0.1)))
        .with_idle_noise(false);
    let mut rng = Rng::seed_from_u64(1);
    let err = mps_sample(&c, &damped, &obs, MpsOptions::exact(), true, &mut rng).unwrap_err();
    assert!(err.to_string().contains("mixtures of unitaries"), "{err}");
    let mut wide = Circuit::new(6);
    for q in 0..6 {
        wide.ry(q, 0.9);
    }
    for q in 0..5 {
        wide.cx(q, q + 1).ry(q + 1, 0.7);
    }
    wide.cx(0, 5).rz(5, 0.4).cx(5, 0);
    let tiny = MpsOptions {
        max_bond: 1,
        cutoff: 0.0,
    };
    let ideal = DeviceModel::new("ideal");
    assert!(mps_sample(&wide, &ideal, &obs, tiny, true, &mut rng).is_err());
    assert!(mps_sample(&wide, &ideal, &obs, tiny, false, &mut rng).is_ok());
}

#[test]
fn frames_refuse_slow_noise() {
    let device = pauli_device().with_default_qubit(
        QubitProperties::new().with_detuning(crate::noise::Spread::Gaussian { sigma: 0.01 }),
    );
    let options = LearningOptions::default();
    let err = learn_layer_with(
        &two_gate_layer(),
        &device,
        &options,
        LearningSimulator::Frames,
    )
    .unwrap_err();
    assert!(err.to_string().contains("slow noise"), "{err}");
    // Density matrices refuse it too.
    assert!(learn_layer(&two_gate_layer(), &device, &options).is_err());
}

#[test]
fn frame_gates_conjugate_like_the_gates() {
    // For every Clifford gate other than a Pauli and every two-qubit Pauli
    // p: the frame gate maps p to q with U p U† = ±q, checked by
    // expectations on a random state. Pauli gates multiply the frame.
    for (build, x0, z0) in [
        (Circuit::x as fn(&mut Circuit, usize) -> &mut Circuit, 1, 0),
        (Circuit::y, 1, 1),
        (Circuit::z, 0, 1),
    ] {
        let mut u = Circuit::new(2);
        build(&mut u, 0);
        let (mut x, mut z) = ([0b10u64], [0b11u64]);
        frame_gate(&u.instructions()[0], &mut x, &mut z).unwrap();
        assert_eq!((x[0], z[0]), (0b10 ^ x0, 0b11 ^ z0));
    }
    let mut rng = Rng::seed_from_u64(8);
    let mut prep = Circuit::new(2);
    for q in 0..2 {
        prep.ry(q, rng.uniform_range(-3.0, 3.0))
            .rz(q, rng.uniform_range(-3.0, 3.0));
    }
    prep.cx(0, 1).ry(0, 0.4).rx(1, 1.1).rz(0, 0.9);
    let psi = StateVector::from_circuit(&prep, &[]).unwrap();
    let gates: Vec<fn(&mut Circuit) -> &mut Circuit> = vec![
        |c| c.h(0),
        |c| c.h(1),
        |c| c.s(0),
        |c| c.sdg(1),
        |c| c.sx(0),
        |c| c.sxdg(1),
        |c| c.cx(0, 1),
        |c| c.cx(1, 0),
        |c| c.cz(0, 1),
        |c| c.cy(0, 1),
        |c| c.cy(1, 0),
        |c| c.swap(0, 1),
        |c| c.id(1),
    ];
    for build in gates {
        let mut u = Circuit::new(2);
        build(&mut u);
        let inst = &u.instructions()[0];
        let mut after = psi.clone();
        after.apply_circuit(&u, &[]).unwrap();
        for p in two_qubit_paulis().into_iter().chain([
            PauliString::single(0, Pauli::Y),
            PauliString::single(1, Pauli::X),
        ]) {
            let (mut x, mut z) = ([p.x_mask()], [p.z_mask()]);
            frame_gate(inst, &mut x, &mut z).unwrap();
            let q = PauliString::from_masks(x[0], z[0]);
            let before = PauliSum::from_terms([(1.0, p)]).expectation(&psi).unwrap();
            let now = PauliSum::from_terms([(1.0, q)])
                .expectation(&after)
                .unwrap();
            assert!(
                (before.abs() - now.abs()).abs() < 1e-10,
                "{}: {p} → {q}",
                inst.name()
            );
        }
    }
}

#[test]
fn trajectories_and_density_matrices_sample_the_same_mitigation() {
    let two = symmetric_model(12);
    let device = device_with(&two);
    let truth = SparsePauliLindblad::new(3, two.generators.clone(), two.rates.clone()).unwrap();
    let layer = cx_layer(3);
    let mut blocks = random_blocks(3, 3, 5);
    blocks[1].cz(1, 2);
    let obs = observable();
    let exact = StateVector::from_circuit(&joined(&blocks, &layer), &[])
        .and_then(|psi| obs.expectation(&psi))
        .unwrap();
    let mut values = Vec::new();
    for simulator in [
        LayeredSimulator::DensityMatrix,
        LayeredSimulator::Trajectories,
        LayeredSimulator::Mps(MpsOptions::exact()),
    ] {
        let r =
            pec_layered_with(&blocks, &layer, &truth, &device, &obs, 4000, 2, simulator).unwrap();
        assert!(
            (r.estimate.value - exact).abs() < 4.0 * r.estimate.std_error,
            "{simulator:?}: {:?} vs {exact}",
            r.estimate
        );
        values.push(r.estimate.value);
    }
    // Same samples (same Paulis), different simulators of each: the
    // density matrix is exact, the others its unbiased estimates.
    assert!(values[0] != values[1] && values[1] != values[2]);
}

#[test]
fn learning_runs_on_64_qubits() {
    let truth = symmetric_model(13);
    let device = device_with(&truth);
    let mut layer = cx_layer(64);
    layer.cx(62, 63);
    let options = LearningOptions {
        depths: vec![2, 6],
        twirls: 2,
        shots: 64,
        ..LearningOptions::default()
    };
    let learned = learn_layer(&layer, &device, &options).unwrap();
    assert_eq!(learned.model.generators.len(), 64 * 3 + 63 * 9);
    // Qubit 63 is measured: the noise of cx(62, 63) shows up there.
    let x63 = PauliString::single(63, Pauli::X);
    let k = learned
        .model
        .generators
        .iter()
        .position(|g| *g == x63)
        .unwrap();
    assert!(learned.model.rates[k].is_finite());
    assert!(learned.model.rates.iter().any(|r| *r > 0.0));
}

#[test]
fn auto_chooses_strict_matrix_product_samples_above_24_qubits() {
    let n = 26;
    let blocks = random_blocks(n, 2, 1);
    let model = SparsePauliLindblad::new(n, Vec::new(), Vec::new()).unwrap();
    let obs = observable();
    let resolve = |layer: &Circuit, blocks: &[Circuit], simulator| {
        check_layered(blocks, layer, &model, &obs, 10, simulator).map(|(_, s)| s)
    };
    let auto = resolve(&cx_layer(n), &blocks, LayeredSimulator::Auto).unwrap();
    assert!(matches!(auto, Sampler::Mps(o, true) if o == MpsOptions::default()));
    let tiny = MpsOptions {
        max_bond: 4,
        cutoff: 0.0,
    };
    let explicit = resolve(&cx_layer(n), &blocks, LayeredSimulator::Mps(tiny)).unwrap();
    assert!(matches!(explicit, Sampler::Mps(o, false) if o == tiny));
    let small = random_blocks(12, 2, 1);
    let model12 = SparsePauliLindblad::new(12, Vec::new(), Vec::new()).unwrap();
    let r = check_layered(
        &small,
        &cx_layer(12),
        &model12,
        &obs,
        10,
        LayeredSimulator::Auto,
    );
    assert!(matches!(r, Ok((_, Sampler::Trajectory))));
    // Trajectory limits name the simulator.
    let err = resolve(&cx_layer(n), &blocks, LayeredSimulator::Trajectories).unwrap_err();
    assert!(err.to_string().contains("trajectories"), "{err}");
    // A device the samples cannot draw from fails at once, not after
    // every sample.
    let damped = DeviceModel::new("damped")
        .with_rule(GateRule::for_gate("cx").with_channel(ChannelSpec::AmplitudeDamping(0.1)))
        .with_idle_noise(false);
    let err = pec_layered(&blocks, &cx_layer(n), &model, &damped, &obs, 200_000, 1).unwrap_err();
    assert!(err.to_string().contains("mixtures of unitaries"), "{err}");
}
