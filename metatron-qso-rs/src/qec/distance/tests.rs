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
