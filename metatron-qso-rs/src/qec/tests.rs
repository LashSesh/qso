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

fn circuits() -> Vec<(&'static str, DetectorCircuit)> {
    vec![
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
