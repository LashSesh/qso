use super::*;
use crate::rng::Rng;
use crate::stabilizer::StabilizerState;

/// Reference: the same noisy circuit, shot by shot on the tableau
/// simulator, with explicitly sampled Pauli errors. Returns detection
/// events and observable flips relative to a noiseless run.
fn tableau_samples(c: &DetectorCircuit, shots: usize, seed: u64) -> (Vec<Vec<bool>>, Vec<u64>) {
    fn run(c: &DetectorCircuit, rng: &mut Rng, noisy: bool) -> (Vec<bool>, u64) {
        let mut s = StabilizerState::zero(c.n_qubits()).unwrap();
        let mut rec = Vec::new();
        let mut det = Vec::new();
        let mut obs = 0u64;
        let pauli = |s: &mut StabilizerState, q: usize, x: bool, z: bool| {
            match (x, z) {
                (true, false) => drop(s.x(q).unwrap()),
                (true, true) => drop(s.y(q).unwrap()),
                (false, true) => drop(s.z(q).unwrap()),
                _ => {}
            };
        };
        for op in c.operations() {
            match op {
                QecOp::Gate(i) => s.apply_instruction(i, &[]).unwrap(),
                QecOp::Measure { qubit, flip } => {
                    let v = s.measure(*qubit, rng).unwrap();
                    rec.push(v ^ (noisy && rng.uniform() < *flip));
                }
                QecOp::Reset { qubit } => s.reset(*qubit, rng).unwrap(),
                QecOp::Noise { noise, qubits } if noisy => match *noise {
                    Noise::XError(p) | Noise::YError(p) | Noise::ZError(p) => {
                        let (x, z) = match noise {
                            Noise::XError(_) => (true, false),
                            Noise::YError(_) => (true, true),
                            _ => (false, true),
                        };
                        for &q in qubits {
                            if rng.uniform() < p {
                                pauli(&mut s, q, x, z);
                            }
                        }
                    }
                    Noise::Depolarize1(p) => {
                        for &q in qubits {
                            if rng.uniform() < p {
                                let k = rng.below(3) + 1;
                                pauli(&mut s, q, k & 1 == 1, k & 2 == 2);
                            }
                        }
                    }
                    Noise::Depolarize2(p) => {
                        for pair in qubits.chunks(2) {
                            if rng.uniform() < p {
                                let k = rng.below(15) + 1;
                                pauli(&mut s, pair[0], k & 1 == 1, k & 2 == 2);
                                pauli(&mut s, pair[1], k & 4 == 4, k & 8 == 8);
                            }
                        }
                    }
                },
                QecOp::Noise { .. } => {}
                QecOp::Detector { records } => {
                    det.push(records.iter().fold(false, |a, &r| a ^ rec[r]));
                }
                QecOp::Observable { index, records } => {
                    if records.iter().fold(false, |a, &r| a ^ rec[r]) {
                        obs ^= 1 << index;
                    }
                }
            }
        }
        (det, obs)
    }
    let mut rng = Rng::seed_from_u64(seed);
    let (ref_det, ref_obs) = run(c, &mut rng, false);
    (0..shots)
        .map(|_| {
            let (d, o) = run(c, &mut rng, true);
            (
                d.iter().zip(&ref_det).map(|(a, b)| a ^ b).collect(),
                o ^ ref_obs,
            )
        })
        .unzip()
}

/// A random Clifford circuit U on `n` qubits, every kind of noise on
/// random qubits, then U† and noisy measurements of every qubit: each
/// result is 0 without noise, so each is a detector.
fn random_noisy_circuit(n: usize, seed: u64) -> DetectorCircuit {
    let mut rng = Rng::seed_from_u64(seed);
    let mut u = crate::quantum::Circuit::new(n);
    for _ in 0..30 {
        let a = rng.below(n as u64) as usize;
        let b = (a + 1 + rng.below(n as u64 - 1) as usize) % n;
        match rng.below(6) {
            0 => u.h(a),
            1 => u.s(a),
            2 => u.sx(a),
            3 => u.cx(a, b),
            4 => u.cz(a, b),
            _ => u.cy(a, b),
        };
    }
    let mut c = DetectorCircuit::new(n).unwrap();
    c.append_circuit(&u, &[]).unwrap();
    for k in 0..8 {
        let a = rng.below(n as u64) as usize;
        let b = (a + 1 + rng.below(n as u64 - 1) as usize) % n;
        let noise = match k % 5 {
            0 => Noise::XError(0.05),
            1 => Noise::YError(0.04),
            2 => Noise::ZError(0.06),
            3 => Noise::Depolarize1(0.08),
            _ => Noise::Depolarize2(0.1),
        };
        let qs = if k % 5 == 4 { vec![a, b] } else { vec![a] };
        c.noise(noise, &qs).unwrap();
        // More gates between the noise.
        let mut v = crate::quantum::Circuit::new(n);
        v.h(b).cx(a, b);
        c.append_circuit(&v, &[]).unwrap();
        c.append_circuit(&v.inverse(), &[]).unwrap();
    }
    c.append_circuit(&u.inverse(), &[]).unwrap();
    let m: Vec<usize> = (0..n).map(|q| c.measure(q, 0.02).unwrap()).collect();
    for &r in &m {
        c.detector(&[r]).unwrap();
    }
    c.observable(0, &[m[0], m[n - 1]]).unwrap();
    c.observable(1, &[m[1]]).unwrap();
    c
}

fn circuits() -> Vec<(&'static str, DetectorCircuit)> {
    vec![
        ("random 4 qubits", random_noisy_circuit(4, 1)),
        ("random 6 qubits", random_noisy_circuit(6, 2)),
        (
            "repetition d=3",
            repetition_code_memory(3, 2, &CircuitNoise::uniform(0.03)).unwrap(),
        ),
        (
            "repetition d=5",
            repetition_code_memory(5, 3, &CircuitNoise::uniform(0.02)).unwrap(),
        ),
        (
            "surface d=3",
            surface_code_memory(3, 2, &CircuitNoise::uniform(0.01)).unwrap(),
        ),
    ]
}

/// Agreement of two frequencies within 5 standard errors.
fn close(a: f64, na: usize, b: f64, nb: usize) -> bool {
    let p = (a * na as f64 + b * nb as f64) / (na + nb) as f64;
    let se = (p * (1.0 - p) * (1.0 / na as f64 + 1.0 / nb as f64)).sqrt();
    (a - b).abs() <= 5.0 * se + 1e-9
}

#[test]
fn frame_sampling_matches_the_tableau() {
    for (name, c) in circuits() {
        let (nf, nt) = (100_000, 3000);
        let frames = sample_detectors(&c, nf, 1).unwrap();
        let (tab_det, tab_obs) = tableau_samples(&c, nt, 2);
        for d in 0..c.n_detectors() {
            let f = frames.count(d) as f64 / nf as f64;
            let t = tab_det.iter().filter(|s| s[d]).count() as f64 / nt as f64;
            assert!(
                close(f, nf, t, nt),
                "{name} detector {d}: frames {f}, tableau {t}"
            );
        }
        let f = (0..nf).filter(|&s| frames.observable_flips(s) != 0).count() as f64 / nf as f64;
        let t = tab_obs.iter().filter(|&&o| o != 0).count() as f64 / nt as f64;
        assert!(
            close(f, nf, t, nt),
            "{name} observable: frames {f}, tableau {t}"
        );
    }
}

#[test]
fn the_error_model_predicts_detector_rates() {
    for (name, c) in circuits() {
        let dem = detector_error_model(&c, false).unwrap();
        let shots = 400_000;
        let samples = sample_detectors(&c, shots, 3).unwrap();
        for d in 0..c.n_detectors() {
            // P(odd number of independent errors) = (1 − Π(1 − 2p))/2.
            let prod: f64 = dem
                .errors
                .iter()
                .filter(|e| e.detectors.contains(&d))
                .map(|e| 1.0 - 2.0 * e.probability)
                .product();
            let predicted = (1.0 - prod) / 2.0;
            let observed = samples.count(d) as f64 / shots as f64;
            let se = (predicted * (1.0 - predicted) / shots as f64).sqrt();
            assert!(
                (observed - predicted).abs() < 5.0 * se + 1e-9,
                "{name} detector {d}: predicted {predicted}, observed {observed}"
            );
        }
    }
}

/// Exact logical error rate of the union-find decoder on the repetition
/// code with independent bit flips (probability q per data qubit), by
/// enumerating every error pattern.
fn exact_repetition_rate(d: usize, q: f64) -> f64 {
    let c = repetition_code_memory(d, 1, &CircuitNoise::data_only(0.3)).unwrap();
    let dem = detector_error_model(&c, true).unwrap();
    let mut decoder = UnionFindDecoder::new(MatchingGraph::from_dem(&dem).unwrap());
    // One mechanism per data qubit (X or Y before the round).
    assert_eq!(dem.errors.len(), d);
    let mut rate = 0.0;
    for pattern in 0u32..(1 << d) {
        let mut fired: Vec<usize> = Vec::new();
        let mut obs = 0u64;
        for (k, e) in dem.errors.iter().enumerate() {
            if (pattern >> k) & 1 == 1 {
                for &det in &e.detectors {
                    if let Some(i) = fired.iter().position(|&x| x == det) {
                        fired.remove(i);
                    } else {
                        fired.push(det);
                    }
                }
                obs ^= e.observables;
            }
        }
        let predicted = if fired.is_empty() {
            0
        } else {
            decoder.decode(&fired).unwrap()
        };
        if predicted != obs {
            let w = pattern.count_ones() as i32;
            rate += q.powi(w) * (1.0 - q).powi(d as i32 - w);
        }
    }
    rate
}

#[test]
fn repetition_code_decoding_is_majority_vote() {
    for d in [3usize, 5, 7, 9] {
        for q in [0.01f64, 0.1, 0.3] {
            // Majority vote fails when more than half of the bits flip.
            let majority: f64 = (d.div_ceil(2)..=d)
                .map(|k| {
                    let binom: f64 = (0..k).map(|i| (d - i) as f64 / (i + 1) as f64).product();
                    binom * q.powi(k as i32) * (1.0 - q).powi((d - k) as i32)
                })
                .sum();
            let exact = exact_repetition_rate(d, q);
            assert!(
                (exact - majority).abs() < 1e-12,
                "d={d} q={q}: {exact} vs {majority}"
            );
        }
    }
    // Sampling, decoding and counting reproduce it: depolarizing p flips a
    // bit (X or Y) with probability 2p/3.
    let (d, p) = (5, 0.15);
    let c = repetition_code_memory(d, 1, &CircuitNoise::data_only(p)).unwrap();
    let r = logical_error_rate(&c, 200_000, 4).unwrap();
    let exact = exact_repetition_rate(d, 2.0 * p / 3.0);
    assert!(
        (r.rate - exact).abs() < 5.0 * r.standard_error,
        "{} vs {exact}",
        r.rate
    );
}

#[test]
fn every_single_fault_is_corrected() {
    for (name, c) in [
        (
            "repetition d=5",
            repetition_code_memory(5, 4, &CircuitNoise::uniform(0.001)).unwrap(),
        ),
        (
            "surface d=3",
            surface_code_memory(3, 3, &CircuitNoise::uniform(0.001)).unwrap(),
        ),
        (
            "surface d=5",
            surface_code_memory(5, 3, &CircuitNoise::uniform(0.001)).unwrap(),
        ),
    ] {
        let dem = detector_error_model(&c, true).unwrap();
        let graph = MatchingGraph::from_dem(&dem).unwrap();
        assert!(graph.undetectable().is_empty(), "{name}");
        let mut decoder = UnionFindDecoder::new(graph);
        for e in &dem.errors {
            assert_eq!(
                decoder.decode(&e.detectors).unwrap(),
                e.observables,
                "{name}: {e:?}"
            );
        }
    }
}

#[test]
fn every_pair_of_faults_is_corrected_at_distance_five() {
    let c = surface_code_memory(5, 2, &CircuitNoise::uniform(0.001)).unwrap();
    let dem = detector_error_model(&c, true).unwrap();
    let mut decoder = UnionFindDecoder::new(MatchingGraph::from_dem(&dem).unwrap());
    let errs = &dem.errors;
    for i in 0..errs.len() {
        for j in i + 1..errs.len() {
            let mut fired = errs[i].detectors.clone();
            for &d in &errs[j].detectors {
                if let Some(k) = fired.iter().position(|&x| x == d) {
                    fired.remove(k);
                } else {
                    fired.push(d);
                }
            }
            let obs = errs[i].observables ^ errs[j].observables;
            let predicted = if fired.is_empty() {
                0
            } else {
                decoder.decode(&fired).unwrap()
            };
            assert_eq!(predicted, obs, "{:?} + {:?}", errs[i], errs[j]);
        }
    }
}

#[test]
fn invalid_input_is_an_error() {
    assert!(DetectorCircuit::new(0).is_err());
    let mut c = DetectorCircuit::new(2).unwrap();
    assert!(c.measure(2, 0.0).is_err());
    assert!(c.measure(0, 1.5).is_err());
    assert!(c.noise(Noise::Depolarize1(0.8), &[0]).is_err());
    assert!(c.noise(Noise::Depolarize2(0.1), &[0]).is_err());
    assert!(c.noise(Noise::Depolarize2(0.1), &[1, 1]).is_err());
    assert!(c.noise(Noise::XError(f64::NAN), &[0]).is_err());
    assert!(c.detector(&[0]).is_err());
    let mut t = crate::quantum::Circuit::new(2);
    t.t(0);
    assert!(c.gate(t.instructions()[0].clone()).is_err());
    assert!(c.observable(64, &[]).is_err());
    // A random measurement as a detector is rejected when sampling.
    let mut r = DetectorCircuit::new(1).unwrap();
    r.h(0).unwrap();
    let m = r.measure(0, 0.0).unwrap();
    r.detector(&[m]).unwrap();
    assert!(sample_detectors(&r, 10, 0).is_err());
    // Distances and rounds.
    assert!(surface_code_memory(4, 1, &CircuitNoise::default()).is_err());
    assert!(repetition_code_memory(1, 1, &CircuitNoise::default()).is_err());
    assert!(repetition_code_memory(3, 0, &CircuitNoise::default()).is_err());
    // A decoder cannot explain a detector without edges.
    let c = repetition_code_memory(3, 1, &CircuitNoise::default()).unwrap();
    let dem = detector_error_model(&c, true).unwrap();
    assert!(dem.errors.is_empty());
    let mut decoder = UnionFindDecoder::new(MatchingGraph::from_dem(&dem).unwrap());
    assert!(decoder.decode(&[0]).is_err());
    assert!(decoder.decode(&[99]).is_err());
}

#[test]
fn logical_error_rate_streams_the_samples_of_one_call() {
    // More shots than one block, so the streaming path runs twice.
    let c = repetition_code_memory(3, 2, &CircuitNoise::uniform(0.05)).unwrap();
    let shots = 64 * frame::batch_shots(&c) + 1000;
    let r = logical_error_rate(&c, shots, 9).unwrap();
    let samples = sample_detectors(&c, shots, 9).unwrap();
    let dem = detector_error_model(&c, true).unwrap();
    let mut decoder = UnionFindDecoder::new(MatchingGraph::from_dem(&dem).unwrap());
    let errors = (0..shots)
        .filter(|&s| {
            let fired = samples.fired_detectors(s);
            let p = if fired.is_empty() {
                0
            } else {
                decoder.decode(&fired).unwrap()
            };
            p != samples.observable_flips(s)
        })
        .count();
    assert_eq!(r.errors, errors);
    assert_eq!(r.shots, shots);
}

#[test]
fn appended_circuits_and_the_stim_export() {
    let mut t = crate::quantum::Circuit::new(2);
    t.h(0).cx(0, 1);
    let theta = t.new_parameter();
    t.rz(1, theta);
    let mut c = DetectorCircuit::new(3).unwrap();
    c.append_circuit(&t, &[std::f64::consts::FRAC_PI_2])
        .unwrap();
    assert!(c.append_circuit(&t, &[0.3]).is_err());
    assert_eq!(c.operations().len(), 3);
    let m0 = c.measure(0, 0.01).unwrap();
    let m1 = c.measure(1, 0.0).unwrap();
    c.noise(Noise::Depolarize2(0.02), &[0, 2]).unwrap();
    c.detector(&[m0, m1]).unwrap();
    c.observable(0, &[m1]).unwrap();
    let text = c.to_stim().unwrap();
    assert_eq!(
        text,
        "H 0\nCX 0 1\nS 1\nM(0.01) 0\nM 1\nDEPOLARIZE2(0.02) 0 2\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
    );
    let mut big = crate::quantum::Circuit::new(4);
    big.h(3);
    assert!(c.append_circuit(&big, &[]).is_err());
}

#[test]
fn limits_fail_early_with_an_error() {
    // Error models need deterministic detectors, as sampling does.
    let mut r = DetectorCircuit::new(1).unwrap();
    r.noise(Noise::XError(0.1), &[0]).unwrap();
    r.h(0).unwrap();
    let m = r.measure(0, 0.0).unwrap();
    r.detector(&[m]).unwrap();
    assert!(detector_error_model(&r, false).is_err());
    // An error that flips every one of 100 000 detectors, 2000 times
    // before merging: too many signature entries.
    let n = 2001;
    let mut c = DetectorCircuit::new(n).unwrap();
    c.noise(Noise::XError(0.01), &(1..n).collect::<Vec<_>>())
        .unwrap();
    for q in 1..n {
        c.cx(q, 0).unwrap();
    }
    let m = c.measure(0, 0.0).unwrap();
    for _ in 0..100_000 {
        c.detector(&[m]).unwrap();
    }
    assert!(detector_error_model(&c, false).is_err());
    // Two million noisy measurements: too much propagation work.
    let mut w = DetectorCircuit::new(1).unwrap();
    for _ in 0..2_000_000 {
        w.measure(0, 0.01).unwrap();
    }
    w.detector(&[0]).unwrap();
    assert!(detector_error_model(&w, false).is_err());
    // Memory experiments beyond the operation limit fail before building.
    let start = std::time::Instant::now();
    assert!(surface_code_memory(101, 10_000, &CircuitNoise::uniform(0.001)).is_err());
    assert!(start.elapsed().as_secs_f64() < 2.0);
    // Matching graphs check their detectors.
    let bad = DetectorErrorModel {
        n_detectors: 3,
        n_observables: 0,
        errors: vec![ErrorMechanism {
            probability: 0.1,
            detectors: vec![0, 5],
            observables: 0,
        }],
    };
    assert!(MatchingGraph::from_dem(&bad).is_err());
    // Tiny probabilities are exported with an exponent.
    let mut t = DetectorCircuit::new(1).unwrap();
    t.noise(Noise::XError(1e-62), &[0]).unwrap();
    assert_eq!(t.to_stim().unwrap(), "X_ERROR(1e-62) 0\n");
}

#[test]
fn edges_more_likely_than_not_are_grown_at_once() {
    // A model with an edge of p > 1/2: decoding still terminates and
    // explains the syndrome.
    let dem = DetectorErrorModel {
        n_detectors: 2,
        n_observables: 1,
        errors: vec![
            ErrorMechanism {
                probability: 0.7,
                detectors: vec![0, 1],
                observables: 1,
            },
            ErrorMechanism {
                probability: 0.01,
                detectors: vec![0],
                observables: 0,
            },
            ErrorMechanism {
                probability: 0.01,
                detectors: vec![1],
                observables: 0,
            },
        ],
    };
    let mut decoder = UnionFindDecoder::new(MatchingGraph::from_dem(&dem).unwrap());
    assert_eq!(decoder.decode(&[0, 1]).unwrap(), 1);
    assert_eq!(decoder.decode(&[0]).unwrap(), 0);
}

#[test]
fn weighted_growth_corrects_random_low_weight_errors() {
    // The syndrome of three errors at distance 7 that step-wise growth
    // misdecoded (edges [43, 47], [38, 43] and [2, 30], no observable).
    let c = surface_code_memory(7, 7, &CircuitNoise::uniform(0.001)).unwrap();
    let dem = detector_error_model(&c, true).unwrap();
    let graph = MatchingGraph::from_dem(&dem).unwrap();
    let (nd, edges) = (graph.n_detectors(), graph.edges().to_vec());
    let mut decoder = UnionFindDecoder::new(graph);
    assert_eq!(decoder.decode(&[2, 30, 38, 47]).unwrap(), 0);
    // 20 000 random sets of (d − 1)/2 = 3 errors are all corrected.
    let mut rng = Rng::seed_from_u64(5);
    for _ in 0..20_000 {
        let mut fired = vec![false; nd + 1];
        let mut flips = 0;
        for _ in 0..3 {
            let e = &edges[rng.below(edges.len() as u64) as usize];
            fired[e.a] ^= true;
            fired[e.b] ^= true;
            flips ^= e.observables;
        }
        let f: Vec<usize> = (0..nd).filter(|&d| fired[d]).collect();
        let predicted = if f.is_empty() {
            0
        } else {
            decoder.decode(&f).unwrap()
        };
        assert_eq!(predicted, flips, "{f:?}");
    }
}

#[test]
fn matching_finds_the_lightest_correction() {
    // Random small error models: for every set of fired detectors, the
    // correction weighs as little as the lightest set of errors with that
    // syndrome (enumerated), and predicts the observables of one of them.
    let mut rng = Rng::seed_from_u64(7);
    for trial in 0..300 {
        let nd = 1 + trial % 7;
        let ne = 1 + rng.below(11) as usize;
        let with_boundary = trial % 4 != 0;
        let errors: Vec<ErrorMechanism> = (0..ne)
            .map(|_| {
                let a = rng.below(nd as u64) as usize;
                let mut b = rng.below(nd as u64 + u64::from(with_boundary)) as usize;
                if b == a {
                    b = if with_boundary { nd } else { (a + 1) % nd };
                }
                let mut detectors = vec![a];
                if b < nd && b != a {
                    detectors.push(b);
                }
                detectors.sort_unstable();
                ErrorMechanism {
                    probability: 0.001 + 0.45 * rng.uniform(),
                    detectors,
                    observables: rng.below(4),
                }
            })
            .collect();
        let dem = DetectorErrorModel {
            n_detectors: nd,
            n_observables: 2,
            errors,
        };
        let mut decoder = MatchingDecoder::new(MatchingGraph::from_dem(&dem).unwrap());
        let edges = decoder.graph().edges().to_vec();
        // Lightest weight and its observables, per syndrome.
        let mut best: Vec<Option<(i64, Vec<u64>)>> = vec![None; 1 << nd];
        for subset in 0usize..1 << edges.len() {
            let (mut syndrome, mut weight, mut obs) = (0usize, 0i64, 0u64);
            for (k, e) in edges.iter().enumerate() {
                if subset & (1 << k) != 0 {
                    syndrome ^= 1 << e.a;
                    if e.b < nd {
                        syndrome ^= 1 << e.b;
                    }
                    weight += decoder.weight[k];
                    obs ^= e.observables;
                }
            }
            match &mut best[syndrome] {
                Some((w, o)) if *w == weight => o.push(obs),
                Some((w, _)) if *w < weight => {}
                slot => *slot = Some((weight, vec![obs])),
            }
        }
        for (syndrome, b) in best.iter().enumerate() {
            let fired: Vec<usize> = (0..nd).filter(|&d| syndrome & (1 << d) != 0).collect();
            match (b, decoder.decode_weighted(&fired)) {
                (None, r) => assert!(r.is_err(), "trial {trial}: {fired:?} has no correction"),
                (Some((w, obs)), r) => {
                    let (prediction, weight) = r.unwrap();
                    assert_eq!(weight, *w, "trial {trial}: {fired:?} in {edges:?}");
                    assert!(obs.contains(&prediction), "trial {trial}: {fired:?}");
                }
            }
        }
    }
}

#[test]
fn matching_corrects_every_fault_up_to_half_the_distance() {
    for (name, c) in [
        (
            "repetition d=5",
            repetition_code_memory(5, 4, &CircuitNoise::uniform(0.001)).unwrap(),
        ),
        (
            "surface d=3",
            surface_code_memory(3, 3, &CircuitNoise::uniform(0.001)).unwrap(),
        ),
        (
            "surface d=5",
            surface_code_memory(5, 3, &CircuitNoise::uniform(0.001)).unwrap(),
        ),
    ] {
        let dem = detector_error_model(&c, true).unwrap();
        let mut decoder = MatchingDecoder::new(MatchingGraph::from_dem(&dem).unwrap());
        for e in &dem.errors {
            assert_eq!(
                decoder.decode(&e.detectors).unwrap(),
                e.observables,
                "{name}: {e:?}"
            );
        }
    }
    // Every pair of faults at distance 5.
    let c = surface_code_memory(5, 2, &CircuitNoise::uniform(0.001)).unwrap();
    let dem = detector_error_model(&c, true).unwrap();
    let mut decoder = MatchingDecoder::new(MatchingGraph::from_dem(&dem).unwrap());
    let errs = &dem.errors;
    for i in 0..errs.len() {
        for j in i + 1..errs.len() {
            let mut fired = errs[i].detectors.clone();
            fired.extend(&errs[j].detectors);
            assert_eq!(
                decoder.decode(&fired).unwrap(),
                errs[i].observables ^ errs[j].observables,
                "{:?} + {:?}",
                errs[i],
                errs[j]
            );
        }
    }
    // Random sets of three faults at distance 7.
    let c = surface_code_memory(7, 7, &CircuitNoise::uniform(0.001)).unwrap();
    let dem = detector_error_model(&c, true).unwrap();
    let graph = MatchingGraph::from_dem(&dem).unwrap();
    let edges = graph.edges().to_vec();
    let mut decoder = MatchingDecoder::new(graph);
    let mut rng = Rng::seed_from_u64(5);
    for _ in 0..20_000 {
        let mut fired = Vec::new();
        let mut flips = 0;
        for _ in 0..3 {
            let e = &edges[rng.below(edges.len() as u64) as usize];
            fired.push(e.a);
            if e.b < dem.n_detectors {
                fired.push(e.b);
            }
            flips ^= e.observables;
        }
        assert_eq!(decoder.decode(&fired).unwrap(), flips, "{fired:?}");
    }
}

#[test]
fn matching_fails_less_often_than_union_find() {
    let c = surface_code_memory(5, 5, &CircuitNoise::uniform(0.006)).unwrap();
    let uf = logical_error_rate_with(&c, 20_000, 3, DecoderKind::UnionFind).unwrap();
    let mwpm = logical_error_rate_with(&c, 20_000, 3, DecoderKind::Matching).unwrap();
    assert_eq!(uf, logical_error_rate(&c, 20_000, 3).unwrap());
    assert!(
        mwpm.errors > 0 && mwpm.errors < uf.errors,
        "{mwpm:?} vs {uf:?}"
    );
}

#[test]
fn matching_without_the_distance_cache_agrees() {
    // Beyond the cache, bounded searches find the same pairs.
    let c = surface_code_memory(5, 5, &CircuitNoise::uniform(0.01)).unwrap();
    let dem = detector_error_model(&c, true).unwrap();
    let graph = MatchingGraph::from_dem(&dem).unwrap();
    let mut cached = MatchingDecoder::new(graph.clone());
    let mut searching = MatchingDecoder::new(graph);
    searching.cache_capacity = 0;
    let samples = sample_detectors(&c, 2000, 9).unwrap();
    for fired in samples.fired_by_shot() {
        let (p, w) = cached.decode_weighted(&fired).unwrap();
        let (q, v) = searching.decode_weighted(&fired).unwrap();
        assert_eq!((p, w), (q, v), "{fired:?}");
    }
}
