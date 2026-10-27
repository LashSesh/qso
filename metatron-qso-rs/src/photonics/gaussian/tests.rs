use super::*;
use crate::photonics::haar_unitary;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn factorial(k: usize) -> f64 {
    (1..=k).map(|i| i as f64).product()
}

#[test]
fn single_mode_states_have_their_closed_form_statistics() {
    // Squeezed vacuum: p(2k) = tanh^{2k} r · (2k)! / (4^k k!² cosh r).
    let r: f64 = 0.8;
    let mut s = GaussianState::vacuum(1).unwrap();
    s.squeeze(0, r, 0.4).unwrap();
    assert!(close(s.mean_photon_numbers()[0], r.sinh().powi(2), 1e-12));
    for n in 0..12 {
        let want = if n % 2 == 1 {
            0.0
        } else {
            let k = n / 2;
            r.tanh().powi(n as i32) * factorial(n)
                / (4f64.powi(k as i32) * factorial(k).powi(2) * r.cosh())
        };
        assert!(close(s.probability(&[n]).unwrap(), want, 1e-13), "{n}");
    }
    // Coherent state: Poisson with mean |α|².
    let alpha = C64::new(0.9, -0.6);
    let mut c = GaussianState::vacuum(1).unwrap();
    c.displace(0, alpha).unwrap();
    let mean = alpha.norm_sqr();
    for n in 0..12 {
        let want = (-mean).exp() * mean.powi(n as i32) / factorial(n);
        assert!(close(c.probability(&[n]).unwrap(), want, 1e-13), "{n}");
    }
    // Thermal state: geometric.
    let t = GaussianState::thermal(&[1.7]).unwrap();
    for n in 0..12 {
        let want = 1.7f64.powi(n as i32) / 2.7f64.powi(n as i32 + 1);
        assert!(close(t.probability(&[n]).unwrap(), want, 1e-13), "{n}");
    }
    assert!(close(
        t.click_probability(&[false]).unwrap(),
        1.0 / 2.7,
        1e-13
    ));
    assert!(close(
        t.click_probability(&[true]).unwrap(),
        1.7 / 2.7,
        1e-13
    ));
}

#[test]
fn two_mode_squeezing_correlates_photon_numbers() {
    // Squeezing two modes oppositely and mixing them on a 50:50 beam
    // splitter gives the two-mode squeezed vacuum: p(n, n) = tanh^{2n} r / cosh² r.
    let r: f64 = 0.6;
    let mut s = GaussianState::vacuum(2).unwrap();
    s.squeeze(0, r, 0.0).unwrap();
    s.squeeze(1, -r, 0.0).unwrap();
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let bs = CMatrix::from_row_slice(
        2,
        2,
        &[
            C64::new(h, 0.0),
            C64::new(h, 0.0),
            C64::new(h, 0.0),
            C64::new(-h, 0.0),
        ],
    );
    s.interferometer(&bs, &[0, 1]).unwrap();
    for n in 0..6 {
        for m in 0..6 {
            let want = if n == m {
                r.tanh().powi(2 * n as i32) / r.cosh().powi(2)
            } else {
                0.0
            };
            assert!(
                close(s.probability(&[n, m]).unwrap(), want, 1e-13),
                "{n} {m}"
            );
        }
    }
}

#[test]
fn loss_is_binomial_thinning() {
    // Loss η on a squeezed state: p'(k) = Σ_n p(n) C(n, k) η^k (1−η)^{n−k}
    // (r = 0.3: the terms beyond 20 photons are below 10⁻¹¹).
    let (r, eta) = (0.3, 0.6);
    let mut s = GaussianState::vacuum(1).unwrap();
    s.squeeze(0, r, 1.1).unwrap();
    let mut lossy = s.clone();
    lossy.loss(0, eta, 0.0).unwrap();
    let ideal: Vec<f64> = (0..=MAX_GAUSSIAN_PHOTONS)
        .map(|n| s.probability(&[n]).unwrap())
        .collect();
    let binom = |n: usize, k: usize| factorial(n) / (factorial(k) * factorial(n - k));
    for k in 0..8 {
        let want: f64 = (k..=MAX_GAUSSIAN_PHOTONS)
            .map(|n| ideal[n] * binom(n, k) * eta.powi(k as i32) * (1.0 - eta).powi((n - k) as i32))
            .sum();
        assert!(close(lossy.probability(&[k]).unwrap(), want, 1e-12), "{k}");
    }
    assert!(close(
        lossy.mean_photon_numbers()[0],
        eta * s.mean_photon_numbers()[0],
        1e-12
    ));
}

#[test]
fn probabilities_sum_to_one_and_interferometers_preserve_photons() {
    let mut rng = Rng::seed_from_u64(5);
    let mut s = GaussianState::vacuum(3).unwrap();
    for (i, r) in [0.3, 0.2, 0.25].into_iter().enumerate() {
        s.squeeze(i, r, 0.5 * i as f64).unwrap();
    }
    s.displace(1, C64::new(0.2, 0.1)).unwrap();
    let u = haar_unitary(3, &mut rng).unwrap();
    let before: f64 = s.mean_photon_numbers().iter().sum();
    s.interferometer(&u, &[0, 1, 2]).unwrap();
    s.loss(2, 0.8, 0.0).unwrap();
    let mut total = 0.0;
    let mut mean = 0.0;
    for a in 0..13 {
        for b in 0..13 - a {
            for c in 0..13 - a - b {
                let p = s.probability(&[a, b, c]).unwrap();
                total += p;
                mean += p * (a + b + c) as f64;
            }
        }
    }
    // Up to 12 photons; the rest of the distribution is below 10⁻⁷.
    assert!(total > 1.0 - 1e-7 && total <= 1.0 + 1e-12, "{total}");
    let after: f64 = s.mean_photon_numbers().iter().sum();
    assert!(after < before);
    assert!(close(mean, after, 1e-6), "{mean} {after}");
    // Click probabilities of all patterns sum to one.
    let mut z = s.clone();
    z = GaussianState::from_parts(z.cov().clone(), vec![0.0; 6]).unwrap();
    let mut clicks = 0.0;
    for m in 0..8u32 {
        let pattern: Vec<bool> = (0..3).map(|i| (m >> i) & 1 == 1).collect();
        clicks += z.click_probability(&pattern).unwrap();
    }
    assert!(close(clicks, 1.0, 1e-12), "{clicks}");
    // No click = vacuum projection: p(no click) = p(0, 0, 0).
    assert!(close(
        z.click_probability(&[false; 3]).unwrap(),
        z.probability(&[0, 0, 0]).unwrap(),
        1e-13
    ));
}

#[test]
fn sampling_follows_the_probabilities() {
    let mut s = GaussianState::vacuum(2).unwrap();
    s.squeeze(0, 0.5, 0.0).unwrap();
    s.squeeze(1, 0.3, 0.7).unwrap();
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let bs = CMatrix::from_row_slice(
        2,
        2,
        &[
            C64::new(h, 0.0),
            C64::new(0.0, h),
            C64::new(0.0, h),
            C64::new(h, 0.0),
        ],
    );
    s.interferometer(&bs, &[0, 1]).unwrap();
    let mut rng = Rng::seed_from_u64(11);
    let shots = 20_000;
    let samples = s.sample(shots, 12, &mut rng).unwrap();
    for pattern in [[0usize, 0], [1, 1], [2, 0], [0, 2], [1, 3]] {
        let p = s.probability(&pattern).unwrap();
        let f = samples.iter().filter(|x| x[..] == pattern[..]).count() as f64 / shots as f64;
        let se = (p * (1.0 - p) / shots as f64).sqrt();
        assert!((f - p).abs() < 5.0 * se + 1e-4, "{pattern:?}: {f} vs {p}");
    }
    let clicks = s.sample_clicks(shots, &mut rng).unwrap();
    for pattern in [[false, false], [true, true], [true, false]] {
        let p = s.click_probability(&pattern).unwrap();
        let f = clicks.iter().filter(|x| x[..] == pattern[..]).count() as f64 / shots as f64;
        let se = (p * (1.0 - p) / shots as f64).sqrt();
        assert!((f - p).abs() < 5.0 * se, "{pattern:?}: {f} vs {p}");
    }
    // Same seed, same samples.
    let a = s.sample(50, 12, &mut Rng::seed_from_u64(1)).unwrap();
    let b = s.sample(50, 12, &mut Rng::seed_from_u64(1)).unwrap();
    assert_eq!(a, b);
}

#[test]
fn invalid_states_and_inputs_are_rejected() {
    // Violates the uncertainty principle.
    let cov = RMatrix::from_diagonal_element(2, 2, 0.5);
    assert!(GaussianState::from_parts(cov, vec![0.0; 2]).is_err());
    assert!(GaussianState::from_parts(RMatrix::identity(3, 3), vec![0.0; 3]).is_err());
    assert!(GaussianState::from_parts(RMatrix::identity(2, 2), vec![0.0; 3]).is_err());
    assert!(GaussianState::vacuum(0).is_err());
    assert!(GaussianState::thermal(&[-1.0]).is_err());
    let mut s = GaussianState::vacuum(2).unwrap();
    assert!(s.squeeze(2, 0.1, 0.0).is_err());
    assert!(s.squeeze(0, f64::NAN, 0.0).is_err());
    assert!(s.loss(0, 1.5, 0.0).is_err());
    let not_unitary = CMatrix::from_element(2, 2, C64::new(1.0, 0.0));
    assert!(s.interferometer(&not_unitary, &[0, 1]).is_err());
    assert!(s.interferometer(&CMatrix::identity(2, 2), &[0, 0]).is_err());
    assert!(s.probability(&[1]).is_err());
    assert!(s.probability(&[MAX_GAUSSIAN_PHOTONS, 1]).is_err());
    s.displace(0, C64::new(0.1, 0.0)).unwrap();
    assert!(s.click_probability(&[true, false]).is_err());
    // A sample beyond max_photons is an error, not a truncated sample.
    let mut big = GaussianState::vacuum(1).unwrap();
    big.squeeze(0, 2.0, 0.0).unwrap();
    assert!(big.sample(100, 2, &mut Rng::seed_from_u64(0)).is_err());
}

#[test]
fn strongly_squeezed_and_large_states_keep_their_accuracy() {
    // Opposite squeezing r = 5 on a beam splitter: two-mode squeezed
    // vacuum, p(n, n) = tanh^{2n} r / cosh² r. A = X(I − Q⁻¹)* is not
    // exactly symmetric here.
    let r: f64 = 5.0;
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let bs = CMatrix::from_row_slice(
        2,
        2,
        &[
            C64::new(h, 0.0),
            C64::new(h, 0.0),
            C64::new(h, 0.0),
            C64::new(-h, 0.0),
        ],
    );
    let mut s = GaussianState::vacuum(2).unwrap();
    s.squeeze(0, r, 0.0).unwrap();
    s.squeeze(1, -r, 0.0).unwrap();
    s.interferometer(&bs, &[0, 1]).unwrap();
    for k in 0..4 {
        let want = r.tanh().powi(2 * k) / r.cosh().powi(2);
        let got = s.probability(&[k as usize, k as usize]).unwrap();
        assert!((got - want).abs() < 1e-9 * want, "{k}: {got} vs {want}");
    }
    // Equal squeezing and random interferometers no longer fail the
    // symmetry check.
    let mut rng = Rng::seed_from_u64(8);
    for _ in 0..20 {
        let mut s = GaussianState::vacuum(4).unwrap();
        for i in 0..4 {
            s.squeeze(i, 5.0, 0.3 * i as f64).unwrap();
        }
        s.interferometer(&haar_unitary(4, &mut rng).unwrap(), &[0, 1, 2, 3])
            .unwrap();
        assert!(s.probability(&[1, 1, 0, 2]).unwrap() >= 0.0);
    }
    // Squeezed vacuum r = 3 at 20 photons (pure: the n × n loop hafnian;
    // 4·10⁻¹² relative, Kan's formula on the 2n × 2n matrix gave 2·10⁻⁹).
    let r: f64 = 3.0;
    let mut sq = GaussianState::vacuum(1).unwrap();
    sq.squeeze(0, r, 0.0).unwrap();
    let want =
        r.tanh().powi(20) * factorial(20) / (4f64.powi(10) * factorial(10).powi(2) * r.cosh());
    let got = sq.probability(&[20]).unwrap();
    assert!((got - want).abs() < 1e-11 * want, "{got} vs {want}");
    // 300 modes squeezed at r = 2: det Q ≈ e^{794} overflows, the
    // probability of the vacuum is Π 1/cosh r ≈ 10^{−173}.
    let mut big = GaussianState::vacuum(300).unwrap();
    for i in 0..300 {
        big.squeeze(i, 2.0, 0.0).unwrap();
    }
    let want = (-300.0 * 2f64.cosh().ln()).exp();
    let got = big.probability(&[0; 300]).unwrap();
    assert!((got - want).abs() < 1e-9 * want, "{got} vs {want}");
    assert!((big.click_probability(&[false; 300]).unwrap() - want).abs() < 1e-9 * want);
}

#[test]
fn covariance_matrices_are_symmetrised_and_checked_independently_of_their_scale() {
    // Asymmetric within the tolerance: accepted and symmetrised.
    let mut s = GaussianState::vacuum(2).unwrap();
    s.squeeze(0, 1.0, 0.0).unwrap();
    s.squeeze(1, 0.5, 0.0).unwrap();
    s.interferometer(
        &haar_unitary(2, &mut Rng::seed_from_u64(1)).unwrap(),
        &[0, 1],
    )
    .unwrap();
    let mut cov = s.cov().clone();
    cov[(0, 1)] += 0.9e-12;
    let t = GaussianState::from_parts(cov, vec![0.0; 4]).unwrap();
    assert!((t.probability(&[1, 1]).unwrap() - s.probability(&[1, 1]).unwrap()).abs() < 1e-11);
    // A squeezed state slightly below the uncertainty limit (det V =
    // 0.999) is rejected, while squeezed states round-trip (up to r = 8,
    // where the rounded matrix has det V = 0.9988).
    for r in [4.0, 8.0] {
        let mut sq = GaussianState::vacuum(1).unwrap();
        sq.squeeze(0, r, 0.7).unwrap();
        assert!(GaussianState::from_parts(sq.cov().clone(), vec![0.0; 2]).is_ok());
    }
    let mut diag = GaussianState::vacuum(1).unwrap();
    diag.squeeze(0, 4.0, 0.0).unwrap();
    let mut bad = diag.cov().clone();
    bad[(1, 1)] *= 0.999;
    assert!(GaussianState::from_parts(bad, vec![0.0; 2]).is_err());
    // Repeated modes do not make a reduced state.
    assert!(s.reduced(&[0, 0]).is_err());
    // Round-off residues in the means count as zero for clicks.
    let mut z = GaussianState::from_parts(s.cov().clone(), vec![1e-17, 0.0, 0.0, -1e-17]).unwrap();
    assert!(z.click_probability(&[true, false]).is_ok());
    z.displace(0, C64::new(0.1, 0.0)).unwrap();
    assert!(z.click_probability(&[true, false]).is_err());
}
