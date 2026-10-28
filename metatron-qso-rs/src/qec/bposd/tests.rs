use super::*;
use crate::qec::{
    CircuitNoise, CssCode, ErrorMechanism, MatchingDecoder, MatchingGraph, MemoryBasis,
    css_code_memory, detector_error_model, sample_detectors, surface_code_memory,
};
use crate::rng::Rng;

/// A random model: `nd` detectors, `ne` errors on one to four detectors.
fn random_dem(rng: &mut Rng, nd: usize, ne: usize) -> DetectorErrorModel {
    let errors = (0..ne)
        .map(|_| {
            let k = 1 + rng.below(4.min(nd as u64)) as usize;
            let mut detectors: Vec<usize> = Vec::new();
            while detectors.len() < k {
                let d = rng.below(nd as u64) as usize;
                if !detectors.contains(&d) {
                    detectors.push(d);
                }
            }
            detectors.sort_unstable();
            ErrorMechanism {
                probability: 0.001 + 0.3 * rng.uniform(),
                detectors,
                observables: rng.below(4),
            }
        })
        .collect();
    DetectorErrorModel {
        n_detectors: nd,
        n_observables: 2,
        errors,
    }
}

/// The detectors a set of model errors flips.
fn syndrome_of(dem: &DetectorErrorModel, errors: &[usize]) -> Vec<usize> {
    let mut s = vec![false; dem.n_detectors];
    for &e in errors {
        for &d in &dem.errors[e].detectors {
            s[d] ^= true;
        }
    }
    (0..dem.n_detectors).filter(|&d| s[d]).collect()
}

fn llr(p: f64) -> f64 {
    (1.0 - p).ln() - p.ln()
}

#[test]
fn exhaustive_osd_finds_the_most_likely_correction() {
    // Without belief propagation and with every non-pivot column in the
    // search, OSD enumerates the whole coset of solutions.
    let mut rng = Rng::seed_from_u64(31);
    for trial in 0..150 {
        let nd = 1 + trial % 7;
        let ne = 1 + rng.below(13) as usize;
        let dem = random_dem(&mut rng, nd, ne);
        let options = BpOsdOptions {
            max_iterations: 0,
            osd: OsdMethod::Exhaustive,
            osd_order: MAX_EXHAUSTIVE_ORDER,
            ..Default::default()
        };
        let mut decoder = BpOsdDecoder::new(&dem, options).unwrap();
        // Lightest weight per syndrome, by enumeration.
        let mut best: Vec<Option<f64>> = vec![None; 1 << nd];
        for subset in 0usize..1 << ne {
            let set: Vec<usize> = (0..ne).filter(|&k| subset >> k & 1 == 1).collect();
            let s = syndrome_of(&dem, &set).iter().fold(0, |a, &d| a | 1 << d);
            let w: f64 = set.iter().map(|&k| llr(dem.errors[k].probability)).sum();
            if best[s].is_none_or(|b| w < b) {
                best[s] = Some(w);
            }
        }
        for (s, b) in best.iter().enumerate() {
            let fired: Vec<usize> = (0..nd).filter(|&d| s >> d & 1 == 1).collect();
            match (b, decoder.decode_detailed(&fired)) {
                (None, r) => assert!(r.is_err(), "trial {trial}: {fired:?}"),
                (Some(w), r) => {
                    let out = r.unwrap();
                    assert_eq!(syndrome_of(&dem, &out.errors), fired, "trial {trial}");
                    assert!(
                        (out.weight - w).abs() < 1e-9,
                        "trial {trial}: {} vs {w}",
                        out.weight
                    );
                    let obs = out
                        .errors
                        .iter()
                        .fold(0, |o, &e| o ^ dem.errors[e].observables);
                    assert_eq!(obs, out.observables);
                }
            }
        }
    }
}

#[test]
fn every_variant_explains_the_detection_events() {
    let mut rng = Rng::seed_from_u64(7);
    let variants = [
        BpOsdOptions::default(),
        BpOsdOptions {
            method: BpMethod::ProductSum,
            osd: OsdMethod::Zero,
            ..Default::default()
        },
        BpOsdOptions {
            scaling: 0.0,
            osd_order: 3,
            ..Default::default()
        },
        BpOsdOptions {
            max_iterations: 3,
            osd: OsdMethod::Exhaustive,
            osd_order: 4,
            ..Default::default()
        },
    ];
    for trial in 0..40 {
        let nd = 10 + rng.below(40) as usize;
        let ne = nd + rng.below(3 * nd as u64) as usize;
        let dem = random_dem(&mut rng, nd, ne);
        for options in variants {
            let mut decoder = BpOsdDecoder::new(&dem, options).unwrap();
            for _ in 0..20 {
                let set: Vec<usize> = (0..ne).filter(|_| rng.below(8) == 0).collect();
                let fired = syndrome_of(&dem, &set);
                let out = decoder.decode_detailed(&fired).unwrap();
                assert_eq!(
                    syndrome_of(&dem, &out.errors),
                    fired,
                    "trial {trial} {options:?}"
                );
                assert!(out.iterations <= options.max_iterations);
            }
        }
    }
}

#[test]
fn osd_sweeps_never_do_worse_than_order_zero() {
    let mut rng = Rng::seed_from_u64(11);
    for _ in 0..30 {
        let nd = 20 + rng.below(20) as usize;
        let ne = 2 * nd + rng.below(nd as u64) as usize;
        let dem = random_dem(&mut rng, nd, ne);
        let base = BpOsdOptions {
            max_iterations: 0,
            ..Default::default()
        };
        let mut zero = BpOsdDecoder::new(
            &dem,
            BpOsdOptions {
                osd: OsdMethod::Zero,
                ..base
            },
        )
        .unwrap();
        let mut sweep = BpOsdDecoder::new(
            &dem,
            BpOsdOptions {
                osd_order: 10,
                ..base
            },
        )
        .unwrap();
        let mut exhaustive = BpOsdDecoder::new(
            &dem,
            BpOsdOptions {
                osd: OsdMethod::Exhaustive,
                osd_order: 8,
                ..base
            },
        )
        .unwrap();
        for _ in 0..20 {
            let set: Vec<usize> = (0..ne).filter(|_| rng.below(6) == 0).collect();
            let fired = syndrome_of(&dem, &set);
            let w0 = zero.decode_detailed(&fired).unwrap().weight;
            let ws = sweep.decode_detailed(&fired).unwrap().weight;
            let we = exhaustive.decode_detailed(&fired).unwrap().weight;
            assert!(ws <= w0 + 1e-9 && we <= w0 + 1e-9, "{w0} {ws} {we}");
        }
    }
}

#[test]
fn bposd_rejects_bad_input_and_is_reusable() {
    let dem = DetectorErrorModel {
        n_detectors: 3,
        n_observables: 1,
        errors: vec![
            ErrorMechanism {
                probability: 0.1,
                detectors: vec![0, 1],
                observables: 1,
            },
            ErrorMechanism {
                probability: 0.0,
                detectors: vec![2],
                observables: 0,
            },
            ErrorMechanism {
                probability: 0.2,
                detectors: vec![],
                observables: 1,
            },
        ],
    };
    let mut d = BpOsdDecoder::new(&dem, BpOsdOptions::default()).unwrap();
    assert_eq!(d.decode(&[0, 1]).unwrap(), 1);
    assert!(d.decode(&[0]).is_err());
    assert!(d.decode(&[2]).is_err());
    assert!(d.decode(&[3]).is_err());
    assert_eq!(d.decode(&[0, 0]).unwrap(), 0);
    assert_eq!(d.decode(&[1, 0, 0, 0]).unwrap(), 1);
    let out = d.decode_detailed(&[1, 0]).unwrap();
    assert_eq!(out.errors, vec![0]);
    assert!(out.converged);
    // Bad models and options.
    let bad = |errors: Vec<ErrorMechanism>| DetectorErrorModel {
        n_detectors: 2,
        n_observables: 1,
        errors,
    };
    let e = |p: f64, d: Vec<usize>| ErrorMechanism {
        probability: p,
        detectors: d,
        observables: 0,
    };
    assert!(BpOsdDecoder::new(&bad(vec![e(1.5, vec![0])]), BpOsdOptions::default()).is_err());
    assert!(BpOsdDecoder::new(&bad(vec![e(0.1, vec![2])]), BpOsdOptions::default()).is_err());
    assert!(BpOsdDecoder::new(&bad(vec![e(0.1, vec![1, 0])]), BpOsdOptions::default()).is_err());
    let dem = bad(vec![e(0.1, vec![0])]);
    for options in [
        BpOsdOptions {
            scaling: 1.5,
            ..Default::default()
        },
        BpOsdOptions {
            osd: OsdMethod::Exhaustive,
            osd_order: 21,
            ..Default::default()
        },
        BpOsdOptions {
            osd_order: MAX_SWEEP_ORDER + 1,
            ..Default::default()
        },
    ] {
        assert!(BpOsdDecoder::new(&dem, options).is_err(), "{options:?}");
    }
}

/// Sets of at most `t` (1 or 2) single-qubit X errors before one round,
/// as detection events of `code`'s Z-basis memory experiment with data
/// noise only, that the decoder gets wrong.
fn low_weight_failures(code: &CssCode, t: usize, options: BpOsdOptions) -> Vec<Vec<usize>> {
    let c = css_code_memory(code, 1, &CircuitNoise::data_only(0.01), MemoryBasis::Z).unwrap();
    let dem = detector_error_model(&c, false).unwrap();
    let mut decoder = BpOsdDecoder::new(&dem, options).unwrap();
    let n = code.n();
    let mut sets: Vec<Vec<usize>> = (0..n).map(|q| vec![q]).collect();
    if t >= 2 {
        for a in 0..n {
            for b in a + 1..n {
                sets.push(vec![a, b]);
            }
        }
    }
    let odd = |support: &[usize], set: &[usize]| {
        support.iter().filter(|q| set.contains(q)).count() % 2 == 1
    };
    sets.into_iter()
        .filter(|set| {
            // First-round detectors flip with the Z checks that see an odd
            // number of the errors.
            let fired: Vec<usize> = (0..code.z_checks().len())
                .filter(|&i| odd(&code.z_checks()[i], set))
                .collect();
            let flips = (0..code.k())
                .filter(|&i| odd(&code.z_logicals()[i], set))
                .fold(0u64, |o, i| o | 1 << i);
            decoder.decode(&fired).unwrap() != flips
        })
        .collect()
}

#[test]
fn bposd_corrects_low_weight_errors_on_colour_and_bicycle_codes() {
    let bb72 = CssCode::bivariate_bicycle_code(72).unwrap().0;
    for options in [
        BpOsdOptions::default(),
        BpOsdOptions {
            method: BpMethod::ProductSum,
            ..Default::default()
        },
        BpOsdOptions {
            scaling: 0.0,
            ..Default::default()
        },
    ] {
        for (code, t) in [
            (CssCode::color_code(5).unwrap(), 2),
            (CssCode::color_code(7).unwrap(), 2),
            (bb72.clone(), 2),
        ] {
            let failures = low_weight_failures(&code, t, options);
            assert!(
                failures.is_empty(),
                "n = {}, {options:?}: {failures:?}",
                code.n()
            );
        }
        // On the Steane code an error on the qubit of all three checks
        // fires every detector, and belief propagation converges in its
        // first iteration to the four errors on the other qubits of two or
        // three checks, a logical error, as `ldpc` does with the same
        // options.
        let steane = CssCode::color_code(3).unwrap();
        assert_eq!(low_weight_failures(&steane, 1, options), vec![vec![3]]);
    }
    // Plain min-sum leaves that case to OSD, which corrects it.
    let plain = BpOsdOptions {
        scaling: 1.0,
        ..Default::default()
    };
    assert!(low_weight_failures(&CssCode::color_code(3).unwrap(), 1, plain).is_empty());
}
#[test]
fn bposd_is_close_to_matching_on_surface_codes() {
    // On a matching graph the matching decoder is optimal; BP+OSD's
    // corrections are never lighter, and mostly as light.
    let c = surface_code_memory(5, 5, &CircuitNoise::uniform(0.005)).unwrap();
    let dem = detector_error_model(&c, true).unwrap();
    let mut matching = MatchingDecoder::new(MatchingGraph::from_dem(&dem).unwrap());
    let mut bposd = BpOsdDecoder::new(&dem, BpOsdOptions::default()).unwrap();
    let shots = sample_detectors(&c, 300, 4).unwrap();
    let (mut same, mut total) = (0, 0);
    for fired in shots.fired_by_shot() {
        if fired.is_empty() {
            continue;
        }
        let (_, w) = matching.decode_with_weight(&fired).unwrap();
        let out = bposd.decode_detailed(&fired).unwrap();
        assert_eq!(syndrome_of(&dem, &out.errors), {
            let mut f = fired.clone();
            f.sort_unstable();
            f.dedup();
            f
        });
        // (Matching rounds its weights to multiples of about 1e-5.)
        assert!(out.weight >= w - 1e-3, "{} < {w}", out.weight);
        total += 1;
        if out.weight < w + 1e-3 {
            same += 1;
        }
    }
    assert!(same * 10 >= total * 9, "{same} of {total}");
}

#[test]
fn sizes_are_checked_before_building() {
    use crate::qec::DecoderChoice;
    // More detectors than BP+OSD takes: an error, not a panic.
    let dem = DetectorErrorModel {
        n_detectors: MAX_BPOSD_DETECTORS + 1,
        n_observables: 1,
        errors: vec![ErrorMechanism {
            probability: 0.1,
            detectors: vec![0, 1],
            observables: 1,
        }],
    };
    let graph = MatchingGraph::from_dem(&dem).unwrap();
    assert!(DecoderChoice::BpOsd.build(graph.clone()).is_err());
    assert!(DecoderChoice::Matching.build(graph).is_ok());
    let huge = DetectorErrorModel {
        n_detectors: usize::MAX,
        ..dem
    };
    assert!(MatchingGraph::from_dem(&huge).is_err());
    // A sweep of order 0 is OSD-0 (as in ldpc).
    let mut rng = Rng::seed_from_u64(4);
    let dem = random_dem(&mut rng, 20, 60);
    let base = BpOsdOptions {
        max_iterations: 0,
        osd_order: 0,
        ..Default::default()
    };
    let mut sweep = BpOsdDecoder::new(&dem, base).unwrap();
    let mut zero = BpOsdDecoder::new(
        &dem,
        BpOsdOptions {
            osd: OsdMethod::Zero,
            ..base
        },
    )
    .unwrap();
    for _ in 0..50 {
        let set: Vec<usize> = (0..60).filter(|_| rng.below(5) == 0).collect();
        let fired = syndrome_of(&dem, &set);
        assert_eq!(
            sweep.decode_detailed(&fired).unwrap(),
            zero.decode_detailed(&fired).unwrap()
        );
    }
}
