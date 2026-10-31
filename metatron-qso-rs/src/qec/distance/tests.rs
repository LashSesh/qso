use super::*;
use crate::qec::{
    CircuitNoise, CssCode, MemoryBasis, css_code_memory, detector_error_model,
    repetition_code_memory, surface_code_memory,
};

fn error(detectors: &[usize], observables: u64) -> ErrorMechanism {
    ErrorMechanism {
        probability: 0.01,
        detectors: detectors.to_vec(),
        observables,
    }
}

fn model(n: usize, errors: Vec<ErrorMechanism>) -> DetectorErrorModel {
    DetectorErrorModel {
        n_detectors: n,
        n_observables: 2,
        errors,
    }
}

/// The witness is an undetectable logical error of the model.
fn check_witness(dem: &DetectorErrorModel, d: &CircuitDistance) {
    let Some(w) = &d.witness else { return };
    let mut fired = vec![false; dem.n_detectors];
    let mut obs = 0;
    for &k in w {
        for &x in &dem.errors[k].detectors {
            fired[x] ^= true;
        }
        obs ^= dem.errors[k].observables;
    }
    assert!(!fired.iter().any(|&f| f), "the witness fires detectors");
    assert_ne!(obs, 0);
    assert_eq!(obs, d.observables);
    assert!(w.windows(2).all(|p| p[0] < p[1]));
}

/// The least weight of an undetectable logical error, by trying every
/// subset.
fn brute_force(dem: &DetectorErrorModel) -> Option<usize> {
    let m = dem.errors.len();
    assert!(m <= 16);
    (1u32..1 << m)
        .filter(|set| {
            let mut fired = vec![false; dem.n_detectors];
            let mut obs = 0;
            for k in (0..m).filter(|k| set >> k & 1 == 1) {
                for &x in &dem.errors[k].detectors {
                    fired[x] ^= true;
                }
                obs ^= dem.errors[k].observables;
            }
            obs != 0 && !fired.iter().any(|&f| f)
        })
        .map(|set| set.count_ones() as usize)
        .min()
}

fn random_model(rng: &mut Rng, n: usize, m: usize, widest: usize) -> DetectorErrorModel {
    let errors = (0..m)
        .map(|_| {
            let k = 1 + rng.below(widest as u64) as usize;
            let mut dets: Vec<usize> = (0..k).map(|_| rng.below(n as u64) as usize).collect();
            dets.sort_unstable();
            dets.dedup();
            let obs = if rng.below(4) == 0 {
                1 + rng.below(3)
            } else {
                0
            };
            error(&dets, obs)
        })
        .collect();
    model(n, errors)
}

#[test]
fn graph_search_matches_brute_force() {
    let mut rng = Rng::seed_from_u64(1);
    for case in 0..300 {
        let dem = random_model(&mut rng, 6, 4 + case % 11, 2);
        let expected = brute_force(&dem);
        let d = graphlike_distance(&dem).unwrap();
        check_witness(&dem, &d);
        assert_eq!(d.upper_bound(), expected, "case {case}: {dem:?}");
        assert_eq!(d.distance(), expected);
        let c = circuit_distance(&dem, &DistanceOptions::default()).unwrap();
        assert_eq!(c.distance(), expected);
        if expected.is_none() {
            assert_eq!(d.lower_bound, usize::MAX);
        }
    }
}

#[test]
fn exhaustive_search_matches_brute_force() {
    let mut rng = Rng::seed_from_u64(2);
    for case in 0..300 {
        let dem = random_model(&mut rng, 7, 4 + case % 12, 4);
        let expected = brute_force(&dem);
        // Without BP+OSD: the exhaustive search alone.
        let options = DistanceOptions {
            bposd_trials: 0,
            max_weight: 16,
            ..DistanceOptions::default()
        };
        let d = circuit_distance(&dem, &options).unwrap();
        check_witness(&dem, &d);
        assert_eq!(d.distance(), expected, "case {case}: {dem:?}");
        if expected.is_none() {
            assert_eq!(d.lower_bound, usize::MAX);
        }
        // With BP+OSD: an upper bound, never below the distance.
        let b = circuit_distance(&dem, &DistanceOptions::default()).unwrap();
        check_witness(&dem, &b);
        assert_eq!(b.distance(), expected);
    }
}

#[test]
fn limits_give_honest_bounds() {
    let dem = detector_error_model(
        &css_code_memory(
            &CssCode::color_code(5).unwrap(),
            2,
            &CircuitNoise::uniform(0.001),
            MemoryBasis::Z,
        )
        .unwrap(),
        false,
    )
    .unwrap();
    // No search: the bounds are what BP+OSD and the graph-like errors give.
    let none = DistanceOptions {
        max_weight: 1,
        bposd_trials: 1,
        ..DistanceOptions::default()
    };
    let d = circuit_distance(&dem, &none).unwrap();
    check_witness(&dem, &d);
    assert_eq!(d.lower_bound, 2);
    assert!(d.upper_bound().unwrap() <= 5);
    // Too little work to finish weight 2.
    let starved = DistanceOptions {
        max_work: 10,
        ..DistanceOptions::default()
    };
    let d = circuit_distance(&dem, &starved).unwrap();
    assert_eq!(d.lower_bound, 2);
    assert!(d.distance().is_none() || d.upper_bound() == Some(2));
}

#[test]
fn memory_experiments_have_their_distance() {
    // Repetition code: graph-like, exact.
    for d in [3, 5, 7] {
        let c = repetition_code_memory(d, d, &CircuitNoise::uniform(0.001)).unwrap();
        let dem = detector_error_model(&c, false).unwrap();
        let r = circuit_distance(&dem, &DistanceOptions::default()).unwrap();
        check_witness(&dem, &r);
        assert_eq!(r.distance(), Some(d));
    }
    // Rotated surface code: the decomposed model is a graph whose
    // distance is d; the whole model (Y errors flip four detectors) has
    // distance d as well, proven by the exhaustive search.
    for d in [3, 5] {
        for basis in [MemoryBasis::Z, MemoryBasis::X] {
            let c =
                crate::qec::surface_code_memory_with(d, d, &CircuitNoise::uniform(0.001), basis)
                    .unwrap();
            let graph = detector_error_model(&c, true).unwrap();
            let g = graphlike_distance(&graph).unwrap();
            check_witness(&graph, &g);
            assert_eq!(g.distance(), Some(d));
            let whole = detector_error_model(&c, false).unwrap();
            let r = circuit_distance(&whole, &DistanceOptions::default()).unwrap();
            check_witness(&whole, &r);
            assert_eq!(r.distance(), Some(d), "surface d = {d}");
        }
    }
    let _ = surface_code_memory;
}

#[test]
fn code_capacity_distance_of_codes() {
    // One noisy round of data errors: the circuit distance is the code's
    // distance (Z basis: X errors).
    let cases = [
        (CssCode::color_code(3).unwrap(), 3),
        (CssCode::color_code(5).unwrap(), 5),
        (CssCode::bivariate_bicycle_code(72).unwrap().0, 6),
    ];
    for (code, d) in cases {
        let c = css_code_memory(&code, 1, &CircuitNoise::data_only(0.01), MemoryBasis::Z).unwrap();
        let dem = detector_error_model(&c, false).unwrap();
        let r = circuit_distance(&dem, &DistanceOptions::default()).unwrap();
        check_witness(&dem, &r);
        assert_eq!(r.distance(), Some(d), "[[{}, {}]]", code.n(), code.k());
    }
}

#[test]
fn trivial_and_invalid_models() {
    // An undetectable error flipping an observable: distance 1.
    let dem = model(2, vec![error(&[0, 1], 0), error(&[], 2)]);
    let d = circuit_distance(&dem, &DistanceOptions::default()).unwrap();
    assert_eq!(
        (d.distance(), d.witness.clone(), d.observables),
        (Some(1), Some(vec![1]), 2)
    );
    assert_eq!(graphlike_distance(&dem).unwrap().distance(), Some(1));
    // No observable: no logical error.
    let dem = model(2, vec![error(&[0, 1], 0), error(&[0], 0)]);
    for d in [
        circuit_distance(&dem, &DistanceOptions::default()).unwrap(),
        graphlike_distance(&dem).unwrap(),
    ] {
        assert_eq!((d.lower_bound, d.witness), (usize::MAX, None));
    }
    // Errors of probability zero never happen.
    let mut e = error(&[], 1);
    e.probability = 0.0;
    let dem = model(1, vec![e, error(&[0], 1), error(&[0], 0)]);
    assert_eq!(
        circuit_distance(&dem, &DistanceOptions::default())
            .unwrap()
            .witness,
        Some(vec![1, 2])
    );
    // Malformed models are errors.
    for bad in [
        model(1, vec![error(&[1], 1)]),
        model(2, vec![error(&[1, 0], 1)]),
        model(
            2,
            vec![ErrorMechanism {
                probability: 1.5,
                ..error(&[0], 1)
            }],
        ),
    ] {
        assert!(circuit_distance(&bad, &DistanceOptions::default()).is_err());
        assert!(graphlike_distance(&bad).is_err());
    }
}

#[test]
fn bounds_below_the_distance_are_not_taken_for_none() {
    // A hyperedge with an observable and three single-detector errors:
    // distance 4, no graph-like logical error; with the search stopped at
    // weight 3 and no BP+OSD the bound is 4, not "none".
    let dem = model(
        3,
        vec![
            error(&[0, 1, 2], 1),
            error(&[0], 0),
            error(&[1], 0),
            error(&[2], 0),
        ],
    );
    let stopped = DistanceOptions {
        max_weight: 3,
        bposd_trials: 0,
        ..DistanceOptions::default()
    };
    let d = circuit_distance(&dem, &stopped).unwrap();
    assert_eq!((d.lower_bound, d.witness.clone()), (4, None));
    let full = DistanceOptions {
        bposd_trials: 0,
        ..DistanceOptions::default()
    };
    assert_eq!(circuit_distance(&dem, &full).unwrap().distance(), Some(4));
    // On random models with hyperedges: a stopped search never claims
    // that nothing exists when something does, and its bound holds.
    let mut rng = Rng::seed_from_u64(3);
    for case in 0..400 {
        let dem = random_model(&mut rng, 7, 4 + case % 12, 4);
        let expected = brute_force(&dem);
        for max_weight in [1, 2, 3] {
            let d = circuit_distance(
                &dem,
                &DistanceOptions {
                    max_weight,
                    bposd_trials: 0,
                    ..DistanceOptions::default()
                },
            )
            .unwrap();
            check_witness(&dem, &d);
            match expected {
                None => assert_eq!(d.lower_bound, usize::MAX, "case {case}"),
                Some(e) => {
                    assert!(d.lower_bound <= e, "case {case}: {d:?} vs {e}");
                    assert!(d.upper_bound().is_none_or(|u| u >= e));
                }
            }
        }
    }
    // Too many BP+OSD searches are refused.
    let many = DistanceOptions {
        bposd_trials: MAX_BPOSD_TRIALS + 1,
        ..DistanceOptions::default()
    };
    assert!(circuit_distance(&dem, &many).is_err());
}

#[test]
fn every_exact_method_matches_brute_force() {
    let mut rng = Rng::seed_from_u64(42);
    let methods = [
        DistanceMethod::Auto,
        DistanceMethod::Search,
        DistanceMethod::BrouwerZimmermann,
    ];
    for _ in 0..300 {
        let n = 1 + rng.below(8) as usize;
        let m = 1 + rng.below(15) as usize;
        let dem = random_model(&mut rng, n, m, 4);
        let want = brute_force(&dem);
        for method in methods {
            let options = ExactDistanceOptions::default()
                .with_method(method)
                .with_bposd_trials(0);
            let r = exact_circuit_distance(&dem, &options).unwrap();
            check_witness(&dem, &r);
            match want {
                Some(d) => assert_eq!(r.distance(), Some(d), "{method:?}"),
                None => assert_eq!(r.lower_bound, usize::MAX, "{method:?}"),
            }
        }
    }
}

#[test]
fn exact_distances_of_codes() {
    use super::zimmermann::tests::{classical, golay, reed_muller};
    // Classical codes, by either method.
    for (name, (rows, n), d) in [
        ("Golay [23, 12]", golay(false), 7),
        ("Golay [24, 12]", golay(true), 8),
        ("RM(2, 6) [64, 22]", reed_muller(2, 6), 16),
    ] {
        let dem = classical(&rows, n);
        for method in [DistanceMethod::Auto, DistanceMethod::BrouwerZimmermann] {
            let options = ExactDistanceOptions::default().with_method(method);
            let r = exact_circuit_distance(&dem, &options).unwrap();
            check_witness(&dem, &r);
            assert_eq!(r.distance(), Some(d), "{name}, {method:?}");
        }
    }
    // Code capacity: one round of data errors (Z basis: X errors).
    let cases = [
        (CssCode::color_code(7).unwrap(), 7),
        (CssCode::bivariate_bicycle_code(72).unwrap().0, 6),
        (CssCode::bivariate_bicycle_code(144).unwrap().0, 12),
    ];
    for (code, d) in cases {
        let c = css_code_memory(&code, 1, &CircuitNoise::data_only(0.01), MemoryBasis::Z).unwrap();
        let dem = detector_error_model(&c, false).unwrap();
        let r = exact_circuit_distance(&dem, &ExactDistanceOptions::default()).unwrap();
        check_witness(&dem, &r);
        assert_eq!(r.distance(), Some(d), "[[{}, {}]]", code.n(), code.k());
    }
}

#[test]
fn exact_circuit_distances_and_time_limits() {
    // Circuit noise: the colour code's hook errors halve its distance; the
    // bicycle code's depth-8 cycle keeps 6.
    let colour = css_code_memory(
        &CssCode::color_code(5).unwrap(),
        2,
        &CircuitNoise::uniform(0.001),
        MemoryBasis::Z,
    )
    .unwrap();
    let dem = detector_error_model(&colour, false).unwrap();
    for method in [DistanceMethod::Auto, DistanceMethod::Search] {
        let options = ExactDistanceOptions::default().with_method(method);
        let r = exact_circuit_distance(&dem, &options).unwrap();
        check_witness(&dem, &r);
        assert_eq!(r.distance(), Some(3), "{method:?}");
    }
    let (code, _) = CssCode::bivariate_bicycle_code(72).unwrap();
    let c = crate::qec::css_code_memory_with(
        &code,
        2,
        &CircuitNoise::uniform(0.001),
        MemoryBasis::Z,
        crate::qec::CssSchedule::Bicycle,
    )
    .unwrap();
    let dem = detector_error_model(&c, false).unwrap();
    let r = exact_circuit_distance(&dem, &ExactDistanceOptions::default()).unwrap();
    check_witness(&dem, &r);
    assert_eq!(r.distance(), Some(6));
    // A time limit returns honest bounds at once.
    let (code, _) = CssCode::bivariate_bicycle_code(144).unwrap();
    let c = crate::qec::css_code_memory_with(
        &code,
        2,
        &CircuitNoise::uniform(0.001),
        MemoryBasis::Z,
        crate::qec::CssSchedule::Bicycle,
    )
    .unwrap();
    let dem = detector_error_model(&c, false).unwrap();
    let start = std::time::Instant::now();
    let options = ExactDistanceOptions::default()
        .with_max_seconds(0.5)
        .with_bposd_trials(1);
    let r = exact_circuit_distance(&dem, &options).unwrap();
    assert!(start.elapsed().as_secs_f64() < 10.0);
    check_witness(&dem, &r);
    assert!(r.lower_bound >= 2 && r.upper_bound().is_some_and(|u| u >= r.lower_bound));
    // Invalid limits.
    for bad in [f64::NAN, -1.0] {
        assert!(exact_circuit_distance(&dem, &options.with_max_seconds(bad)).is_err());
    }
}

#[test]
fn witnesses_are_explained_by_circuit_faults() {
    // Every error of a circuit's model comes from faults whose
    // probabilities combine to its own, and the colour code's lightest
    // logical error is three faults of the circuit.
    let colour = css_code_memory(
        &CssCode::color_code(3).unwrap(),
        2,
        &CircuitNoise::uniform(0.002),
        MemoryBasis::Z,
    )
    .unwrap();
    let dem = detector_error_model(&colour, false).unwrap();
    let explained = crate::qec::explain_errors(&colour, &dem.errors).unwrap();
    for (e, faults) in dem.errors.iter().zip(&explained) {
        assert!(!faults.is_empty());
        let p = faults.iter().fold(0.0, |acc, f| {
            acc * (1.0 - f.probability) + f.probability * (1.0 - acc)
        });
        assert!(
            (p - e.probability).abs() < 1e-15,
            "{p} vs {}",
            e.probability
        );
        for f in faults {
            assert!(matches!(
                colour.operations()[f.operation],
                crate::qec::QecOp::Noise { .. } | crate::qec::QecOp::Measure { .. }
            ));
            assert!(!f.paulis.is_empty());
            assert!(f.paulis.iter().all(|&(_, c)| "XYZM".contains(c)));
        }
    }
    let r = exact_circuit_distance(&dem, &ExactDistanceOptions::default()).unwrap();
    let w = r.witness.unwrap();
    let chosen: Vec<ErrorMechanism> = w.iter().map(|&k| dem.errors[k].clone()).collect();
    let faults = crate::qec::explain_errors(&colour, &chosen).unwrap();
    assert_eq!(faults.len(), r.lower_bound);
    assert!(faults.iter().all(|f| !f.is_empty()));
    // An error the circuit cannot make has no fault.
    let foreign = ErrorMechanism {
        probability: 0.1,
        detectors: vec![0, 1, 2, 3, 4, 5, 6],
        observables: 1,
    };
    assert!(crate::qec::explain_errors(&colour, &[foreign]).unwrap()[0].is_empty());
}

#[test]
fn exact_distance_ends_when_nothing_is_logical() {
    // Beyond the detectors where linear algebra decides, and without a
    // time limit: the search stops once its weight passes the number of
    // errors, and at once when no error flips an observable.
    let n = MAX_EXISTENCE_DETECTORS + 10;
    let unlimited = ExactDistanceOptions::default().with_max_seconds(f64::INFINITY);
    for method in [
        DistanceMethod::Auto,
        DistanceMethod::Search,
        DistanceMethod::BrouwerZimmermann,
    ] {
        for errors in [
            vec![error(&[0, 1, 2], 1), error(&[0], 0)],
            vec![error(&[0, 1, 2], 0), error(&[0], 0), error(&[1, 2], 0)],
        ] {
            let dem = model(n, errors);
            let r = exact_circuit_distance(&dem, &unlimited.with_method(method)).unwrap();
            assert_eq!((r.lower_bound, r.witness), (usize::MAX, None), "{method:?}");
        }
    }
    // Disjoint triples of detectors: no error can follow another, so the
    // search runs out of errors to add at once, with or without a limit
    // on its work.
    let triples = model(
        n,
        (0..1000)
            .map(|i| error(&[3 * i, 3 * i + 1, 3 * i + 2], u64::from(i % 2 == 0)))
            .collect(),
    );
    let r = exact_circuit_distance(&triples, &unlimited.with_bposd_trials(0)).unwrap();
    assert_eq!((r.lower_bound, r.witness.clone()), (usize::MAX, None));
    let options = DistanceOptions {
        bposd_trials: 0,
        ..DistanceOptions::default()
    };
    let r = circuit_distance(&triples, &options).unwrap();
    assert_eq!((r.lower_bound, r.witness), (usize::MAX, None));
}

/// Errors that each flip `width` distinct random detectors of `n`, a
/// quarter of them an observable.
fn wide_model(rng: &mut Rng, n: usize, m: usize, width: usize) -> DetectorErrorModel {
    let errors = (0..m)
        .map(|_| {
            let mut dets: Vec<usize> = Vec::new();
            while dets.len() < width {
                let d = rng.below(n as u64) as usize;
                if !dets.contains(&d) {
                    dets.push(d);
                }
            }
            dets.sort_unstable();
            error(&dets, u64::from(rng.below(4) == 0))
        })
        .collect();
    model(n, errors)
}

#[cfg(feature = "parallel")]
#[test]
fn bounds_and_witnesses_do_not_depend_on_the_threads() {
    // The work limit stops the search at different points on different
    // numbers of threads; the answer is that of the search in index
    // order on any number. The colour code of distance 7 over two rounds
    // has circuit distance 4.
    let c = css_code_memory(
        &CssCode::color_code(7).unwrap(),
        2,
        &CircuitNoise::uniform(0.001),
        MemoryBasis::Z,
    )
    .unwrap();
    let dem = detector_error_model(&c, false).unwrap();
    let on = |threads: usize, max_work: u64| {
        let options = DistanceOptions {
            max_work,
            bposd_trials: 0,
            ..DistanceOptions::default()
        };
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| circuit_distance(&dem, &options).unwrap())
    };
    let mut exact = Vec::new();
    for max_work in [1 << 6, 1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20] {
        let one = on(1, max_work);
        check_witness(&dem, &one);
        for threads in [2, 3, 4] {
            assert_eq!(on(threads, max_work), one, "{threads} threads, {max_work}");
        }
        exact.push(one.distance().is_some());
    }
    // 64 sets do not cover the first errors; 2²⁰ finish the search.
    assert!(!exact[0] && exact[exact.len() - 1], "{exact:?}");
    assert_eq!(on(1, 1 << 20).distance(), Some(4));
    // Without a limit the exact distance agrees.
    let unlimited = ExactDistanceOptions::default()
        .with_method(DistanceMethod::Search)
        .with_bposd_trials(0);
    for threads in [1, 4] {
        let r = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| exact_circuit_distance(&dem, &unlimited).unwrap());
        assert_eq!(r, on(1, 1 << 20));
    }
}

#[test]
fn the_enumeration_keeps_the_time_limit() {
    // Level 2 alone is 7·10⁷ sums of 188 words; the deadline stops it.
    let mut rng = Rng::seed_from_u64(7);
    let dem = wide_model(&mut rng, 300, 12_000, 6);
    let options = ExactDistanceOptions::default()
        .with_method(DistanceMethod::BrouwerZimmermann)
        .with_max_seconds(0.3)
        .with_bposd_trials(0);
    let start = std::time::Instant::now();
    let r = exact_circuit_distance(&dem, &options).unwrap();
    let seconds = start.elapsed().as_secs_f64();
    assert!(seconds < 2.0, "{seconds} s");
    assert!(r.lower_bound >= 2 && r.distance().is_none(), "{r:?}");
}
