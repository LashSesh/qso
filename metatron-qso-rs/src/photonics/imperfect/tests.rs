use std::collections::HashMap;

use super::*;
use crate::photonics::{beam_splitter, haar_unitary};

/// Output distribution of photons with internal states, simulated in the
/// Fock space of `m × d` modes (internal states as extra modes the
/// interferometer does not touch), summed over the internal states.
fn brute_force(u: &CMatrix, modes: &[usize], states: &[Vec<C64>]) -> HashMap<Vec<usize>, f64> {
    let m = u.nrows();
    let d = states[0].len();
    type Poly = HashMap<Vec<(usize, usize)>, C64>;
    let multiply = |poly: &Poly, terms: &[((usize, usize), C64)]| -> Poly {
        let mut out: Poly = HashMap::new();
        for (mono, c) in poly {
            for &(op, w) in terms {
                let mut next = mono.clone();
                next.push(op);
                next.sort_unstable();
                *out.entry(next).or_insert(C64::new(0.0, 0.0)) += c * w;
            }
        }
        out
    };
    let norm2 = |poly: &Poly| -> f64 {
        poly.iter()
            .map(|(mono, c)| {
                let mut counts: HashMap<(usize, usize), usize> = HashMap::new();
                for &op in mono {
                    *counts.entry(op).or_insert(0) += 1;
                }
                c.norm_sqr() * counts.values().map(|&v| factorial(v)).product::<f64>()
            })
            .sum()
    };
    let mut input: Poly = HashMap::from([(Vec::new(), C64::new(1.0, 0.0))]);
    let mut output = input.clone();
    for (&s, psi) in modes.iter().zip(states) {
        let terms_in: Vec<_> = (0..d).map(|k| ((s, k), psi[k])).collect();
        input = multiply(&input, &terms_in);
        let terms_out: Vec<_> = (0..m)
            .flat_map(|i| (0..d).map(move |k| (i, k)))
            .map(|(i, k)| ((i, k), u[(i, s)] * psi[k]))
            .collect();
        output = multiply(&output, &terms_out);
    }
    let norm = norm2(&input);
    let mut dist: HashMap<Vec<usize>, f64> = HashMap::new();
    for (mono, c) in &output {
        let mut counts: HashMap<(usize, usize), usize> = HashMap::new();
        for &op in mono {
            *counts.entry(op).or_insert(0) += 1;
        }
        let mut t = vec![0; m];
        for (&(i, _), &v) in &counts {
            t[i] += v;
        }
        let p = c.norm_sqr() * counts.values().map(|&v| factorial(v)).product::<f64>() / norm;
        *dist.entry(t).or_insert(0.0) += p;
    }
    dist
}

fn random_state(d: usize, rng: &mut Rng) -> Vec<C64> {
    let v: Vec<C64> = (0..d)
        .map(|_| C64::new(rng.normal(), rng.normal()))
        .collect();
    let n = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    v.into_iter().map(|z| z / n).collect()
}

#[test]
fn partial_distinguishability_matches_internal_modes() {
    let mut rng = Rng::seed_from_u64(4);
    let u = haar_unitary(3, &mut rng).unwrap();
    for input in [
        vec![1, 1, 1],
        vec![2, 1, 0],
        vec![0, 1, 2],
        vec![0, 0, 3],
        vec![1, 1, 2],
    ] {
        let modes: Vec<usize> = input
            .iter()
            .enumerate()
            .flat_map(|(j, &s)| std::iter::repeat_n(j, s))
            .collect();
        let states: Vec<Vec<C64>> = modes.iter().map(|_| random_state(3, &mut rng)).collect();
        let sampler = BosonSampler::new(u.clone(), &input).unwrap();
        let imperfect = ImperfectBosonSampler::new(sampler)
            .with_internal_states(&states)
            .unwrap();
        let want = brute_force(&u, &modes, &states);
        for (t, p) in &want {
            let got = imperfect.probability(t).unwrap();
            assert!((got - p).abs() < 1e-13, "{input:?} {t:?}: {got} vs {p}");
        }
        let total: f64 = imperfect.distribution().unwrap().probabilities.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
    }
}

#[test]
fn the_limits_are_the_ideal_samplers() {
    let mut rng = Rng::seed_from_u64(9);
    let u = haar_unitary(4, &mut rng).unwrap();
    let sampler = BosonSampler::new(u, &[1, 1, 0, 1]).unwrap();
    let ideal = ImperfectBosonSampler::new(sampler.clone());
    let one = ImperfectBosonSampler::new(sampler.clone())
        .with_indistinguishability(1.0)
        .unwrap();
    let zero = ImperfectBosonSampler::new(sampler.clone())
        .with_indistinguishability(0.0)
        .unwrap();
    // Nearly identical photons take the general formula and approach the ideal.
    let almost = ImperfectBosonSampler::new(sampler.clone())
        .with_indistinguishability(1.0 - 1e-9)
        .unwrap();
    for t in occupation_patterns(3, 4).unwrap() {
        let p = sampler.probability(&t).unwrap();
        assert!((ideal.probability(&t).unwrap() - p).abs() < 1e-14);
        assert!((one.probability(&t).unwrap() - p).abs() < 1e-14);
        assert!((almost.probability(&t).unwrap() - p).abs() < 1e-8);
        let q = sampler.probability_distinguishable(&t).unwrap();
        assert!((zero.probability(&t).unwrap() - q).abs() < 1e-14);
    }
    // Patterns with fewer photons are impossible without loss.
    assert_eq!(ideal.probability(&[1, 1, 0, 0]).unwrap(), 0.0);
}

#[test]
fn hong_ou_mandel_visibility_is_the_overlap() {
    // Two photons on a 50:50 beam splitter: P(1, 1) = (1 − |⟨ψ₁|ψ₂⟩|²)/2.
    let sampler =
        BosonSampler::new(beam_splitter(std::f64::consts::FRAC_PI_4, 0.0), &[1, 1]).unwrap();
    for x in [0.0, 0.25, 0.5, 0.9, 1.0] {
        let s = ImperfectBosonSampler::new(sampler.clone())
            .with_indistinguishability(x)
            .unwrap();
        assert!(
            (s.probability(&[1, 1]).unwrap() - (1.0 - x) / 2.0).abs() < 1e-14,
            "{x}"
        );
    }
    // A complex overlap: only its modulus matters.
    let g = CMatrix::from_row_slice(
        2,
        2,
        &[
            C64::new(1.0, 0.0),
            C64::new(0.3, 0.4),
            C64::new(0.3, -0.4),
            C64::new(1.0, 0.0),
        ],
    );
    let s = ImperfectBosonSampler::new(sampler).with_gram(g).unwrap();
    assert!((s.probability(&[1, 1]).unwrap() - (1.0 - 0.25) / 2.0).abs() < 1e-14);
}

/// `u` on `2m` modes, with every input mode `j` (`at_input`) or output mode
/// `j` coupled to environment mode `m + j` by a beam splitter of
/// transmission `losses[j]`.
fn with_environment(u: &CMatrix, losses: &[f64], at_input: bool) -> CMatrix {
    let m = u.nrows();
    let mut big = CMatrix::identity(2 * m, 2 * m);
    big.view_mut((0, 0), (m, m)).copy_from(u);
    let mut bs = CMatrix::identity(2 * m, 2 * m);
    for (j, &e) in losses.iter().enumerate() {
        let (t, r) = (e.sqrt(), (1.0 - e).sqrt());
        bs[(j, j)] = C64::new(t, 0.0);
        bs[(j, j + m)] = C64::new(-r, 0.0);
        bs[(j + m, j)] = C64::new(r, 0.0);
        bs[(j + m, j + m)] = C64::new(t, 0.0);
    }
    if at_input { &big * &bs } else { &bs * &big }
}

#[test]
fn losses_match_an_interferometer_with_environment_modes() {
    let mut rng = Rng::seed_from_u64(12);
    let u = haar_unitary(3, &mut rng).unwrap();
    let input = [1, 2, 0];
    let losses = [0.7, 0.4, 0.9];
    for at_input in [true, false] {
        let sampler = BosonSampler::new(u.clone(), &input).unwrap();
        let lossy = if at_input {
            ImperfectBosonSampler::new(sampler)
                .with_transmission(&losses)
                .unwrap()
        } else {
            ImperfectBosonSampler::new(sampler)
                .with_efficiency(&losses)
                .unwrap()
        };
        let big_input = [input[0], input[1], input[2], 0, 0, 0];
        let big = BosonSampler::new(with_environment(&u, &losses, at_input), &big_input).unwrap();
        let mut marginal: HashMap<Vec<usize>, f64> = HashMap::new();
        for t in occupation_patterns(3, 6).unwrap() {
            *marginal.entry(t[..3].to_vec()).or_insert(0.0) += big.probability(&t).unwrap();
        }
        for (t, p) in &marginal {
            let got = lossy.probability(t).unwrap();
            assert!((got - p).abs() < 1e-13, "{at_input} {t:?}: {got} vs {p}");
        }
        let d = lossy.distribution().unwrap();
        assert!((d.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(d.patterns.len(), 1 + 3 + 6 + 10);
    }
}

#[test]
fn partial_distinguishability_with_loss_and_sampling() {
    let mut rng = Rng::seed_from_u64(2);
    let u = haar_unitary(3, &mut rng).unwrap();
    let sampler = BosonSampler::new(u, &[1, 1, 1]).unwrap();
    let s = ImperfectBosonSampler::new(sampler)
        .with_indistinguishability(0.6)
        .unwrap()
        .with_transmission(&[0.9, 0.8, 0.95])
        .unwrap()
        .with_efficiency(&[0.85, 0.9, 0.8])
        .unwrap();
    let d = s.distribution().unwrap();
    assert!((d.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    let shots = 20_000;
    let samples = s.sample(shots, &mut Rng::seed_from_u64(3)).unwrap();
    for (t, &p) in d.patterns.iter().zip(&d.probabilities).take(8) {
        let f = samples.iter().filter(|x| *x == t).count() as f64 / shots as f64;
        let se = (p * (1.0 - p) / shots as f64).sqrt();
        assert!((f - p).abs() < 5.0 * se + 1e-4, "{t:?}: {f} vs {p}");
    }
}

#[test]
fn invalid_imperfections_are_rejected() {
    let sampler = BosonSampler::new(beam_splitter(0.3, 0.1), &[1, 1]).unwrap();
    let base = ImperfectBosonSampler::new(sampler);
    let one = C64::new(1.0, 0.0);
    assert!(base.clone().with_gram(CMatrix::identity(3, 3)).is_err());
    let not_hermitian =
        CMatrix::from_row_slice(2, 2, &[one, C64::new(0.5, 0.0), C64::new(0.1, 0.0), one]);
    assert!(base.clone().with_gram(not_hermitian).is_err());
    let not_psd =
        CMatrix::from_row_slice(2, 2, &[one, C64::new(1.5, 0.0), C64::new(1.5, 0.0), one]);
    assert!(base.clone().with_gram(not_psd).is_err());
    let diag = CMatrix::from_row_slice(
        2,
        2,
        &[
            C64::new(2.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            one,
        ],
    );
    assert!(base.clone().with_gram(diag).is_err());
    assert!(base.clone().with_indistinguishability(1.5).is_err());
    assert!(base.clone().with_transmission(&[0.5]).is_err());
    assert!(base.clone().with_efficiency(&[0.5, -0.1]).is_err());
    assert!(
        base.clone()
            .with_internal_states(&[vec![one], vec![C64::new(0.0, 0.0)]])
            .is_err()
    );
    assert!(base.probability(&[1]).is_err());
    let nine = BosonSampler::new(CMatrix::identity(9, 9), &[1; 9]).unwrap();
    let partial = ImperfectBosonSampler::new(nine)
        .with_indistinguishability(0.5)
        .unwrap();
    assert!(partial.probability(&[1; 9]).is_err());
}
