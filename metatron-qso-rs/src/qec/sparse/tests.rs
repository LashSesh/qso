use super::*;
use crate::qec::{
    CircuitNoise, DetectorErrorModel, ErrorMechanism, MatchingDecoder, detector_error_model,
    repetition_code_memory, sample_detectors, surface_code_memory,
};
use crate::rng::Rng;

/// A random error model: `nd` detectors, `ne` errors on one or two
/// detectors (or one and the boundary), random probabilities.
fn random_dem(rng: &mut Rng, nd: usize, ne: usize, with_boundary: bool) -> DetectorErrorModel {
    let errors = (0..ne)
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
    DetectorErrorModel {
        n_detectors: nd,
        n_observables: 2,
        errors,
    }
}

#[test]
fn sparse_matching_finds_the_lightest_correction() {
    // Every syndrome of small random models: the weight of the lightest
    // set of errors (enumerated), observables of one of them.
    let mut rng = Rng::seed_from_u64(17);
    for trial in 0..400 {
        let nd = 1 + trial % 8;
        let ne = 1 + rng.below(12) as usize;
        let dem = random_dem(&mut rng, nd, ne, trial % 4 != 0);
        let graph = MatchingGraph::from_dem(&dem).unwrap();
        let exact = MatchingDecoder::new(graph.clone());
        let mut sparse = SparseMatchingDecoder::new(graph);
        let edges = exact.graph().edges().to_vec();
        let mut best: Vec<Option<(i64, Vec<u64>)>> = vec![None; 1 << nd];
        for subset in 0usize..1 << edges.len() {
            let (mut syndrome, mut weight, mut obs) = (0usize, 0i64, 0u64);
            for (k, e) in edges.iter().enumerate() {
                if subset & (1 << k) != 0 {
                    syndrome ^= 1 << e.a;
                    if e.b < nd {
                        syndrome ^= 1 << e.b;
                    }
                    weight += exact.weight[k];
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
            match (b, sparse.decode_weighted(&fired)) {
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

/// Shot by shot, the same weight as the exact decoder; returns the
/// decoder's event counts.
fn agree_on_samples(dem: &DetectorErrorModel, shots: Vec<Vec<usize>>, label: &str) -> [usize; 4] {
    let graph = MatchingGraph::from_dem(dem).unwrap();
    let mut exact = MatchingDecoder::new(graph.clone());
    let mut sparse = SparseMatchingDecoder::new(graph);
    for fired in shots {
        let a = exact.decode_weighted(&fired);
        let b = sparse.decode_weighted(&fired);
        match (a, b) {
            (Ok((_, w)), Ok((_, v))) => assert_eq!(w, v, "{label}: {fired:?}"),
            (Err(_), Err(_)) => {}
            (a, b) => panic!("{label}: {fired:?}: {a:?} vs {b:?}"),
        }
    }
    sparse.counts
}

#[test]
fn sparse_matching_agrees_with_the_exact_decoder_on_memory_experiments() {
    for (d, p) in [(3, 0.01), (5, 0.005), (5, 0.02), (7, 0.01), (9, 0.004)] {
        let c = surface_code_memory(d, d, &CircuitNoise::uniform(p)).unwrap();
        let dem = detector_error_model(&c, true).unwrap();
        let shots = sample_detectors(&c, 600, d as u64).unwrap().fired_by_shot();
        let _ = agree_on_samples(&dem, shots, &format!("surface d={d} p={p}"));
    }
    let c = repetition_code_memory(11, 11, &CircuitNoise::uniform(0.03)).unwrap();
    let dem = detector_error_model(&c, true).unwrap();
    let shots = sample_detectors(&c, 1000, 3).unwrap().fired_by_shot();
    let _ = agree_on_samples(&dem, shots, "repetition d=11");
}

#[test]
fn sparse_matching_agrees_on_dense_random_syndromes() {
    // Many detection events on random graphs with and without a boundary:
    // trees, blossoms inside blossoms and shattering blossoms.
    let mut rng = Rng::seed_from_u64(23);
    let mut counts = [0usize; 4];
    for trial in 0..60 {
        let nd = 20 + rng.below(40) as usize;
        let ne = nd + rng.below(3 * nd as u64) as usize;
        let dem = random_dem(&mut rng, nd, ne, trial % 3 != 0);
        let shots = (0..40)
            .map(|_| {
                let k = 2 + rng.below(nd as u64 - 1) as usize;
                (0..k).map(|_| rng.below(nd as u64) as usize).collect()
            })
            .collect();
        let c = agree_on_samples(&dem, shots, &format!("random trial {trial}"));
        (0..4).for_each(|k| counts[k] += c[k]);
    }
    // A cycle with equal weights: odd cycles of events form blossoms.
    for n in [5usize, 7, 9, 15] {
        let errors = (0..n)
            .map(|i| ErrorMechanism {
                probability: 0.1,
                detectors: {
                    let mut d = vec![i, (i + 1) % n];
                    d.sort_unstable();
                    d
                },
                observables: u64::from(i == 0),
            })
            .collect();
        let dem = DetectorErrorModel {
            n_detectors: n,
            n_observables: 1,
            errors,
        };
        let mut shots = Vec::new();
        for mask in 0u32..1 << n {
            if mask.count_ones() % 2 == 0 && shots.len() < 3000 {
                shots.push((0..n).filter(|&i| mask >> i & 1 == 1).collect());
            }
        }
        let c = agree_on_samples(&dem, shots, &format!("cycle {n}"));
        (0..4).for_each(|k| counts[k] += c[k]);
    }
    // Every operation of the algorithm ran.
    eprintln!("blossoms, shattered, zero-radius, grown: {counts:?}");
    assert!(counts.iter().all(|&c| c > 50), "{counts:?}");
}

#[test]
fn sparse_matching_rejects_impossible_syndromes() {
    // Two detectors joined by one error, no boundary: one event alone has
    // no explanation.
    let dem = DetectorErrorModel {
        n_detectors: 3,
        n_observables: 1,
        errors: vec![ErrorMechanism {
            probability: 0.1,
            detectors: vec![0, 1],
            observables: 1,
        }],
    };
    let mut d = SparseMatchingDecoder::new(MatchingGraph::from_dem(&dem).unwrap());
    assert!(d.decode(&[0]).is_err());
    assert!(d.decode(&[2]).is_err());
    assert!(d.decode(&[3]).is_err());
    assert_eq!(d.decode(&[0, 1]).unwrap(), 1);
    // Listed twice cancels; three times counts once.
    assert_eq!(d.decode(&[0, 0]).unwrap(), 0);
    assert_eq!(d.decode(&[0, 0, 0, 1]).unwrap(), 1);
    // The decoder is reusable after an error.
    assert_eq!(d.decode(&[1, 0]).unwrap(), 1);
}
