use super::super::{Budget, Error, errors};
use super::*;
use crate::qec::dem::{DetectorErrorModel, ErrorMechanism};
use crate::qec::gf2;
use crate::rng::Rng;

/// The generator matrices of a model's errors, if they were built.
fn built(n: usize, errs: &[Error]) -> Option<Zimmermann> {
    match Zimmermann::build(n, errs, &Budget::new(u64::MAX, None)) {
        Enumeration::Ready(z) => Some(z),
        _ => None,
    }
}

/// The lightest logical word, level by level until the bound reaches it
/// (`None`: no logical word).
fn minimum(dem: &DetectorErrorModel) -> Option<usize> {
    let errs = errors(dem).unwrap();
    let budget = Budget::new(u64::MAX, None);
    let z = built(dem.n_detectors, &errs)?;
    let mut best = usize::MAX;
    for w in 1..=z.dimension() {
        let level = z.level(w, best, &budget);
        assert!(level.complete);
        if let Some(set) = level.found {
            best = set.len();
        }
        if z.bound(w) >= best {
            break;
        }
    }
    (best != usize::MAX).then_some(best)
}

/// The detector error model of the classical code spanned by `rows` (of
/// `n` bits): its parity checks as detectors, and one observable per
/// column of an information set, so that every nonzero codeword is
/// logical.
pub(crate) fn classical(rows: &[Vec<u64>], n: usize) -> DetectorErrorModel {
    let checks = gf2::kernel(rows, n);
    // An information set: columns independent in the generator matrix.
    let mut basis = gf2::Basis::new(rows.len(), false, 0);
    let mut info = Vec::new();
    for c in 0..n {
        let column: Vec<usize> = (0..rows.len()).filter(|&r| gf2::get(&rows[r], c)).collect();
        if basis.insert(&gf2::from_ones(rows.len(), &column)).is_some() {
            info.push(c);
        }
    }
    assert!(info.len() <= 64);
    let errors = (0..n)
        .map(|c| ErrorMechanism {
            probability: 0.01,
            detectors: (0..checks.len())
                .filter(|&i| gf2::get(&checks[i], c))
                .collect(),
            observables: info.iter().position(|&i| i == c).map_or(0, |j| 1 << j),
        })
        .collect();
    DetectorErrorModel {
        n_detectors: checks.len(),
        n_observables: info.len(),
        errors,
    }
}

/// The binary Golay code [23, 12, 7] from its generator polynomial
/// `1 + x² + x⁴ + x⁵ + x⁶ + x¹⁰ + x¹¹`, or with a parity bit [24, 12, 8].
pub(crate) fn golay(extended: bool) -> (Vec<Vec<u64>>, usize) {
    let g = [0usize, 2, 4, 5, 6, 10, 11];
    let n = if extended { 24 } else { 23 };
    let rows = (0..12)
        .map(|s| {
            let mut ones: Vec<usize> = g.iter().map(|&e| e + s).collect();
            if extended {
                ones.push(23);
            }
            gf2::from_ones(n, &ones)
        })
        .collect();
    (rows, n)
}

/// The Reed–Muller code RM(r, m): the values of the monomials of degree
/// at most `r` on the points of F₂^m.
pub(crate) fn reed_muller(r: u32, m: u32) -> (Vec<Vec<u64>>, usize) {
    let n = 1usize << m;
    let rows = (0u32..1 << m)
        .filter(|mono| mono.count_ones() <= r)
        .map(|mono| {
            let ones: Vec<usize> = (0..n).filter(|&x| x as u32 & mono == mono).collect();
            gf2::from_ones(n, &ones)
        })
        .collect();
    (rows, n)
}

#[test]
fn classical_codes_have_their_minimum_distance() {
    for (name, (rows, n), d) in [
        ("Golay [23, 12]", golay(false), 7),
        ("Golay [24, 12]", golay(true), 8),
        ("RM(1, 5) [32, 6]", reed_muller(1, 5), 16),
        ("RM(2, 5) [32, 16]", reed_muller(2, 5), 8),
        ("RM(2, 6) [64, 22]", reed_muller(2, 6), 16),
        ("RM(3, 6) [64, 42]", reed_muller(3, 6), 8),
    ] {
        let dem = classical(&rows, n);
        assert_eq!(minimum(&dem), Some(d), "{name}");
    }
}

#[test]
fn information_sets_are_disjoint_and_span_the_code() {
    let mut rng = Rng::seed_from_u64(3);
    for _ in 0..50 {
        let n = 2 + rng.below(30) as usize;
        let m = 1 + rng.below(90) as usize;
        let errors_list: Vec<ErrorMechanism> = (0..m)
            .map(|_| {
                let mut d: Vec<usize> = (0..1 + rng.below(4) as usize)
                    .map(|_| rng.below(n as u64) as usize)
                    .collect();
                d.sort_unstable();
                d.dedup();
                ErrorMechanism {
                    probability: 0.01,
                    detectors: d,
                    observables: rng.below(4),
                }
            })
            .collect();
        let dem = DetectorErrorModel {
            n_detectors: n,
            n_observables: 2,
            errors: errors_list,
        };
        let errs = errors(&dem).unwrap();
        let Some(z) = built(n, &errs) else {
            continue;
        };
        let (k, words) = (z.k, z.words);
        let mut seen = vec![false; errs.len()];
        for g in &z.matrices {
            assert_eq!(g.rows.len(), k * words);
            // Every row is undetectable, with the observables it flips.
            for r in 0..k {
                let row = &g.rows[r * words..(r + 1) * words];
                let mut fired = vec![false; n];
                let mut obs = 0;
                for c in gf2::ones(row) {
                    for &d in &errs[c].detectors {
                        fired[d] ^= true;
                    }
                    obs ^= errs[c].observables;
                }
                assert!(!fired.iter().any(|&f| f));
                assert_eq!(obs, g.observables[r]);
            }
            // The rows are independent (they span the code), and the
            // matrix is systematic on its information set, which no other
            // matrix shares.
            let mut basis = gf2::Basis::new(errs.len(), false, 0);
            for r in 0..k {
                assert!(basis.insert(&g.rows[r * words..(r + 1) * words]).is_some());
            }
            assert_eq!(g.columns.len(), g.rank);
            for (i, &c) in g.columns.iter().enumerate() {
                assert!(!seen[c]);
                seen[c] = true;
                for r in 0..k {
                    let bit = gf2::get(&g.rows[r * words..(r + 1) * words], c);
                    assert_eq!(bit, r == i);
                }
            }
        }
        assert_eq!(z.matrices[0].rank, k);
    }
}

#[test]
fn the_enumeration_matches_brute_force() {
    let mut rng = Rng::seed_from_u64(11);
    for _ in 0..300 {
        let n = 1 + rng.below(7) as usize;
        let m = 1 + rng.below(14) as usize;
        let errs: Vec<ErrorMechanism> = (0..m)
            .map(|_| {
                let mut d: Vec<usize> = (0..1 + rng.below(4) as usize)
                    .map(|_| rng.below(n as u64) as usize)
                    .collect();
                d.sort_unstable();
                d.dedup();
                ErrorMechanism {
                    probability: 0.01,
                    detectors: d,
                    observables: if rng.below(3) == 0 {
                        1 + rng.below(3)
                    } else {
                        0
                    },
                }
            })
            .collect();
        let dem = DetectorErrorModel {
            n_detectors: n,
            n_observables: 2,
            errors: errs,
        };
        // Brute force over subsets of the distinct errors.
        let distinct = errors(&dem).unwrap();
        let want = (1u32..1 << distinct.len())
            .filter(|set| {
                let mut fired = vec![false; n];
                let mut obs = 0;
                for (k, e) in distinct.iter().enumerate() {
                    if set >> k & 1 == 1 {
                        for &d in &e.detectors {
                            fired[d] ^= true;
                        }
                        obs ^= e.observables;
                    }
                }
                obs != 0 && !fired.iter().any(|&f| f)
            })
            .map(|set| set.count_ones() as usize)
            .min();
        assert_eq!(minimum(&dem), want);
    }
}

#[test]
fn bounds_grow_with_the_ranks_of_the_information_sets() {
    // Golay [24, 12]: two disjoint information sets of rank 12, so after
    // level w no word lighter than 2(w + 1) is left.
    let (rows, n) = golay(true);
    let dem = classical(&rows, n);
    let z = built(dem.n_detectors, &errors(&dem).unwrap()).unwrap();
    assert_eq!(z.dimension(), 12);
    assert_eq!(z.matrices.len(), 2);
    assert!(z.matrices.iter().all(|g| g.rank == 12));
    assert_eq!(z.bound(3), 8);
    assert_eq!(z.bound(12), usize::MAX);
    assert!(z.cost(3) > z.cost(2));
}

/// A model from detector lists and observables.
fn model(n: usize, errors: &[(&[usize], u64)]) -> DetectorErrorModel {
    DetectorErrorModel {
        n_detectors: n,
        n_observables: 1,
        errors: errors
            .iter()
            .map(|&(d, o)| ErrorMechanism {
                probability: 0.01,
                detectors: d.to_vec(),
                observables: o,
            })
            .collect(),
    }
}

#[test]
fn deficient_information_sets_do_not_skip_light_words() {
    // Dimension 8, a second information set of rank 6: the lightest
    // logical word (3 errors) is a single row of the second matrix that
    // vanishes on its information set. Enumerating that matrix only from
    // level 2 on, as sums of exactly two rows, missed it, and the bound of
    // level 2 (3 + 1) stopped the search at a word of 4.
    let dem = model(
        6,
        &[
            (&[1, 2], 0),
            (&[0, 1, 2, 3], 0),
            (&[0, 1, 3], 1),
            (&[2, 3, 4, 5], 0),
            (&[0, 2, 3, 4], 0),
            (&[4, 5], 0),
            (&[0, 2, 3, 5], 0),
            (&[2, 3, 4], 0),
            (&[2, 4], 0),
            (&[0, 1, 2], 1),
            (&[2], 1),
            (&[0, 1, 4, 5], 0),
            (&[2, 3, 5], 0),
            (&[4], 0),
        ],
    );
    let z = built(dem.n_detectors, &errors(&dem).unwrap()).unwrap();
    assert_eq!(z.dimension(), 8);
    assert_eq!(z.matrices[1].rank, 6);
    assert_eq!(z.bound(2), 4);
    assert_eq!(minimum(&dem), Some(3));
}

#[test]
fn every_level_meets_the_words_below_its_bound() {
    // Codes of high rate with heavy columns, whose further information
    // sets are mostly deficient: after every level, the lightest logical
    // word (from all 2^k words) has been met or weighs at least the bound.
    let mut rng = Rng::seed_from_u64(2);
    let (mut checked, mut deficient) = (0, 0);
    for _ in 0..1500 {
        let n = 3 + rng.below(8) as usize;
        let m = n + 4 + rng.below(10) as usize;
        let errs: Vec<ErrorMechanism> = (0..m)
            .map(|_| {
                let mut d: Vec<usize> = (0..1 + rng.below(5) as usize)
                    .map(|_| rng.below(n as u64) as usize)
                    .collect();
                d.sort_unstable();
                d.dedup();
                ErrorMechanism {
                    probability: 0.01,
                    detectors: d,
                    observables: u64::from(rng.below(3) == 0),
                }
            })
            .collect();
        let dem = DetectorErrorModel {
            n_detectors: n,
            n_observables: 1,
            errors: errs,
        };
        let distinct = errors(&dem).unwrap();
        if distinct.iter().any(|e| e.detectors.is_empty()) {
            continue;
        }
        let Some(z) = built(n, &distinct) else {
            continue;
        };
        if z.k > 12 {
            continue;
        }
        let (g, words) = (&z.matrices[0], z.words);
        let Some(want) = (1u32..1 << z.k)
            .filter_map(|u| {
                let mut sum = vec![0u64; words];
                let mut obs = 0;
                for r in (0..z.k).filter(|&r| u >> r & 1 == 1) {
                    for (s, x) in sum.iter_mut().zip(&g.rows[r * words..(r + 1) * words]) {
                        *s ^= x;
                    }
                    obs ^= g.observables[r];
                }
                (obs != 0).then(|| sum.iter().map(|x| x.count_ones() as usize).sum::<usize>())
            })
            .min()
        else {
            continue;
        };
        checked += 1;
        deficient += usize::from(z.matrices.iter().any(|g| g.rank < z.k));
        let budget = Budget::new(u64::MAX, None);
        let mut best = usize::MAX;
        for w in 1..=z.k {
            let level = z.level(w, best, &budget);
            assert!(level.complete);
            if let Some(set) = level.found {
                best = set.len();
            }
            assert!(
                best == want || z.bound(w) <= want,
                "level {w}: best {best}, bound {}, lightest {want}",
                z.bound(w)
            );
        }
        assert_eq!(best, want);
    }
    assert!(checked > 1400 && deficient > 1300, "{checked}, {deficient}");
}
