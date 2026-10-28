use super::*;
use crate::qec::{detector_error_model, sample_detectors};

/// Checks, logicals and their pairing are those of a CSS code.
fn assert_valid(code: &CssCode) {
    let n = code.n();
    let vecs =
        |s: &[Vec<usize>]| -> Vec<Vec<u64>> { s.iter().map(|c| gf2::from_ones(n, c)).collect() };
    let (hx, hz) = (vecs(code.x_checks()), vecs(code.z_checks()));
    let (xl, zl) = (vecs(code.x_logicals()), vecs(code.z_logicals()));
    let rank = |rows: &[Vec<u64>]| {
        let mut b = Basis::new(n, false, 0);
        rows.iter().filter(|r| b.insert(r).is_some()).count()
    };
    assert_eq!(code.k(), n - rank(&hx) - rank(&hz));
    assert_eq!(xl.len(), zl.len());
    for x in &hx {
        assert!(hz.iter().chain(&zl).all(|z| !gf2::dot(x, z)));
    }
    for z in &hz {
        assert!(xl.iter().all(|x| !gf2::dot(x, z)));
    }
    for (i, x) in xl.iter().enumerate() {
        for (j, z) in zl.iter().enumerate() {
            assert_eq!(gf2::dot(x, z), i == j, "logicals {i} and {j}");
        }
    }
    // The logicals are independent of the checks.
    let mut all = hz.clone();
    all.extend(zl.iter().cloned());
    assert_eq!(rank(&all), rank(&hz) + zl.len());
}

/// Weight of the lightest Z logical (commutes with the X checks, outside
/// the span of the Z checks), searched up to `max` qubits.
fn z_distance(code: &CssCode, max: usize) -> Option<usize> {
    let n = code.n();
    let hx: Vec<Vec<u64>> = code
        .x_checks()
        .iter()
        .map(|c| gf2::from_ones(n, c))
        .collect();
    let mut span = Basis::new(n, false, 0);
    for c in code.z_checks() {
        span.insert(&gf2::from_ones(n, c));
    }
    fn subsets(
        n: usize,
        w: usize,
        start: usize,
        cur: &mut Vec<usize>,
        f: &mut dyn FnMut(&[usize]) -> bool,
    ) -> bool {
        if cur.len() == w {
            return f(cur);
        }
        for q in start..n {
            cur.push(q);
            if subsets(n, w, q + 1, cur, f) {
                return true;
            }
            cur.pop();
        }
        false
    }
    (1..=max).find(|&w| {
        subsets(n, w, 0, &mut Vec::new(), &mut |s| {
            let v = gf2::from_ones(n, s);
            hx.iter().all(|x| !gf2::dot(x, &v)) && !span.contains(&v)
        })
    })
}

#[test]
fn colour_codes_have_their_parameters() {
    for (d, n) in [(3, 7), (5, 19), (7, 37), (9, 61), (21, 331)] {
        let code = CssCode::color_code(d).unwrap();
        assert_eq!(code.n(), n);
        assert_eq!(code.k(), 1);
        assert_eq!(code.x_checks(), code.z_checks());
        assert_eq!(code.x_checks().len(), (n - 1) / 2);
        assert!(code.x_checks().iter().all(|c| c.len() == 4 || c.len() == 6));
        assert_valid(&code);
    }
    // The distance, by search.
    for d in [3, 5] {
        let code = CssCode::color_code(d).unwrap();
        assert_eq!(z_distance(&code, d), Some(d), "d = {d}");
    }
    for d in [0, 1, 2, 4, 103] {
        assert!(CssCode::color_code(d).is_err());
    }
}

#[test]
fn bivariate_bicycle_codes_have_their_parameters() {
    let (a, b) = ([(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]);
    // Bravyi et al. (2024), table 3.
    for (l, m, a, b, n, k) in [
        (6, 6, a.to_vec(), b.to_vec(), 72, 12),
        (
            15,
            3,
            vec![(9, 0), (0, 1), (0, 2)],
            vec![(0, 0), (2, 0), (7, 0)],
            90,
            8,
        ),
        (9, 6, a.to_vec(), b.to_vec(), 108, 8),
        (12, 6, a.to_vec(), b.to_vec(), 144, 12),
        (12, 12, vec![(3, 0), (0, 2), (0, 7)], b.to_vec(), 288, 12),
    ] {
        let code = CssCode::bivariate_bicycle(l, m, &a, &b).unwrap();
        assert_eq!((code.n(), code.k()), (n, k), "l = {l}, m = {m}");
        assert!(
            code.x_checks()
                .iter()
                .chain(code.z_checks())
                .all(|c| c.len() == 6)
        );
        assert_valid(&code);
    }
    for (n, k, d) in [
        (72, 12, 6),
        (90, 8, 10),
        (108, 8, 10),
        (144, 12, 12),
        (288, 12, 18),
    ] {
        let (code, distance) = CssCode::bivariate_bicycle_code(n).unwrap();
        assert_eq!((code.n(), code.k(), distance), (n, k, d));
    }
    assert!(CssCode::bivariate_bicycle_code(100).is_err());
    // Equal monomials cancel; a zero polynomial leaves no checks.
    let code = CssCode::bivariate_bicycle(3, 3, &[(1, 0), (1, 0)], &[(0, 0)]).unwrap();
    assert!(code.x_checks().iter().all(|c| c.len() == 1));
    assert!(CssCode::bivariate_bicycle(0, 3, &a, &b).is_err());
    assert!(CssCode::bivariate_bicycle(100, 100, &a, &b).is_err());
}

#[test]
fn css_codes_check_their_input() {
    // The Steane code from its Hamming checks.
    let h = vec![vec![3, 4, 5, 6], vec![1, 2, 5, 6], vec![0, 2, 4, 6]];
    let steane = CssCode::new(7, h.clone(), h.clone()).unwrap();
    assert_eq!(steane.k(), 1);
    assert_valid(&steane);
    assert_eq!(z_distance(&steane, 3), Some(3));
    // Anticommuting checks, repeated or foreign qubits, empty checks.
    assert!(CssCode::new(2, vec![vec![0]], vec![vec![0, 1]]).is_err());
    assert!(CssCode::new(3, vec![vec![0, 0]], vec![]).is_err());
    assert!(CssCode::new(3, vec![vec![3]], vec![]).is_err());
    assert!(CssCode::new(3, vec![vec![]], vec![]).is_err());
    assert!(CssCode::new(0, vec![], vec![]).is_err());
    // No checks at all: every qubit is logical.
    let free = CssCode::new(3, vec![], vec![]).unwrap();
    assert_eq!(free.k(), 3);
    assert_valid(&free);
}

#[test]
fn css_memory_experiments_are_deterministic_without_noise() {
    let codes = [
        CssCode::color_code(5).unwrap(),
        CssCode::bivariate_bicycle(6, 6, &[(3, 0), (0, 1), (0, 2)], &[(0, 3), (1, 0), (2, 0)])
            .unwrap(),
    ];
    for code in &codes {
        for basis in [MemoryBasis::Z, MemoryBasis::X] {
            let rounds = 3;
            let c = css_code_memory(code, rounds, &CircuitNoise::default(), basis).unwrap();
            let checks = match basis {
                MemoryBasis::Z => code.z_checks().len(),
                MemoryBasis::X => code.x_checks().len(),
            };
            assert_eq!(c.n_detectors(), (rounds + 1) * checks);
            assert_eq!(c.n_observables(), code.k());
            let s = sample_detectors(&c, 64, 1).unwrap();
            assert!(s.fired_by_shot().iter().all(|f| f.is_empty()));
            assert!((0..64).all(|k| s.observable_flips(k) == 0));
            // With noise, the model exists and has errors that flip three
            // or more detectors.
            let noisy =
                css_code_memory(code, rounds, &CircuitNoise::uniform(0.001), basis).unwrap();
            let dem = detector_error_model(&noisy, false).unwrap();
            assert!(dem.errors.iter().any(|e| e.detectors.len() >= 3));
        }
    }
    assert!(css_code_memory(&codes[0], 0, &CircuitNoise::default(), MemoryBasis::Z).is_err());
    let no_logical = CssCode::new(1, vec![], vec![vec![0]]).unwrap();
    assert!(css_code_memory(&no_logical, 1, &CircuitNoise::default(), MemoryBasis::Z).is_err());
}

#[test]
fn cnot_layers_touch_each_qubit_once() {
    let code = CssCode::color_code(7).unwrap();
    let l = layers(code.z_checks(), code.n());
    let total: usize = code.z_checks().iter().map(Vec::len).sum();
    assert_eq!(l.iter().map(Vec::len).sum::<usize>(), total);
    for layer in &l {
        let mut qs: Vec<usize> = layer.iter().map(|&(_, q)| q).collect();
        let mut cs: Vec<usize> = layer.iter().map(|&(c, _)| c).collect();
        qs.sort_unstable();
        cs.sort_unstable();
        assert!(qs.windows(2).all(|w| w[0] < w[1]));
        assert!(cs.windows(2).all(|w| w[0] < w[1]));
    }
    // Greedy colouring needs at most 2Δ − 1 layers (Δ = 6 here).
    assert!(l.len() <= 11, "{}", l.len());
}

#[test]
fn bicycle_exponents_reduce_and_sizes_are_bounded() {
    // x^(2⁶⁴ − 1) = x³ on Z_6: the [[72, 12]] code.
    let b = [(0, 3), (1, 0), (2, 0)];
    let huge = CssCode::bivariate_bicycle(6, 6, &[(usize::MAX, 0), (0, 1), (0, 2 + 6 * 1000)], &b)
        .unwrap();
    let plain = CssCode::bivariate_bicycle(6, 6, &[(3, 0), (0, 1), (0, 2)], &b).unwrap();
    assert_eq!(huge, plain);
    assert!(CssCode::new(MAX_CSS_QUBITS + 1, vec![], vec![]).is_err());
}

#[test]
fn the_bicycle_cycle_keeps_the_distance() {
    use crate::qec::{
        CssSchedule, DistanceOptions, circuit_distance, css_code_memory_with, detector_error_model,
    };
    // [[72, 12, 6]], three rounds of circuit noise: with the depth-8 cycle
    // of Bravyi et al. the lightest undetectable logical error has 6 faults
    // (proven: the exhaustive search excludes 5), as the paper's bound
    // dcirc ≤ 6 allows; the greedy schedule loses two to hook errors.
    let (code, _) = CssCode::bivariate_bicycle_code(72).unwrap();
    let noise = CircuitNoise::uniform(0.001);
    for basis in [MemoryBasis::Z, MemoryBasis::X] {
        for (schedule, d) in [(CssSchedule::Bicycle, 6), (CssSchedule::Greedy, 4)] {
            let c = css_code_memory_with(&code, 3, &noise, basis, schedule).unwrap();
            let dem = detector_error_model(&c, false).unwrap();
            let r = circuit_distance(&dem, &DistanceOptions::default()).unwrap();
            assert_eq!(r.distance(), Some(d), "{basis:?} {schedule:?}");
        }
    }
    // Only bicycle codes with three monomials per polynomial have the cycle.
    let colour = CssCode::color_code(3).unwrap();
    assert!(
        css_code_memory_with(&colour, 2, &noise, MemoryBasis::Z, CssSchedule::Bicycle).is_err()
    );
    let two = CssCode::bivariate_bicycle(3, 3, &[(1, 0), (0, 1)], &[(0, 0), (1, 1)]);
    if let Ok(two) = two {
        if two.k() > 0 {
            assert!(
                css_code_memory_with(&two, 2, &noise, MemoryBasis::Z, CssSchedule::Bicycle)
                    .is_err()
            );
        }
    }
}
