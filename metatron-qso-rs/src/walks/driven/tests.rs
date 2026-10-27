use super::*;
use crate::graph::generators;
use crate::walks::ContinuousWalk;

fn localized(n: usize, j: usize) -> Vec<C64> {
    let mut psi = vec![C64::new(0.0, 0.0); n];
    psi[j] = C64::new(1.0, 0.0);
    psi
}

fn distance(a: &[C64], b: &[C64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm_sqr())
        .sum::<f64>()
        .sqrt()
}

/// A driven walk on the cycle C₈ with a moving potential.
fn driven_cycle() -> (DrivenWalk, impl FnMut(f64) -> Result<Vec<f64>>) {
    let g = generators::cycle(8);
    let walk = DrivenWalk::new(
        8,
        vec![
            Term::graph(&g, Generator::Laplacian),
            Term::potential(&[1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0, 0.5]).unwrap(),
            Term::Uniform,
        ],
    )
    .unwrap();
    let coefficients = |t: f64| Ok(vec![1.0 + 0.5 * (2.0 * t).sin(), 3.0 * (1.3 * t).cos(), t]);
    (walk, coefficients)
}

#[test]
fn a_static_hamiltonian_matches_the_continuous_walk() {
    let g = generators::hypercube(4);
    let walk = DrivenWalk::new(16, vec![Term::graph(&g, Generator::Laplacian)]).unwrap();
    let r = walk
        .evolve(
            &localized(16, 0),
            0.0,
            2.5,
            |_| Ok(vec![0.7]),
            &MagnusOptions::default(),
        )
        .unwrap();
    let exact = ContinuousWalk::new(&g, Generator::Laplacian, 0.7)
        .unwrap()
        .evolve(&localized(16, 0), 2.5)
        .unwrap();
    assert!(distance(&r.state, &exact) < 1e-9);
    assert_eq!(r.rejected, 0);
}

#[test]
fn the_integrator_converges_with_fourth_order() {
    let (walk, mut c) = driven_cycle();
    let psi0 = localized(8, 2);
    let run = |h: f64, c: &mut dyn FnMut(f64) -> Result<Vec<f64>>| {
        let opts = MagnusOptions {
            fixed_step: Some(h),
            tol: 1e-12,
            ..MagnusOptions::default()
        };
        walk.evolve(&psi0, 0.0, 2.0, c, &opts).unwrap().state
    };
    let reference = run(1.0 / 512.0, &mut c);
    let errors: Vec<f64> = [0.2, 0.1, 0.05]
        .iter()
        .map(|&h| distance(&run(h, &mut c), &reference))
        .collect();
    for w in errors.windows(2) {
        let ratio = w[0] / w[1];
        assert!((13.0..19.0).contains(&ratio), "{errors:?}");
    }
}

#[test]
fn step_size_control_meets_the_tolerance() {
    let (walk, mut c) = driven_cycle();
    let psi0 = localized(8, 0);
    let reference = walk
        .evolve(
            &psi0,
            0.0,
            3.0,
            &mut c,
            &MagnusOptions {
                fixed_step: Some(1.0 / 1024.0),
                tol: 1e-13,
                ..MagnusOptions::default()
            },
        )
        .unwrap()
        .state;
    for tol in [1e-4, 1e-7, 1e-10] {
        let opts = MagnusOptions {
            tol,
            ..MagnusOptions::default()
        };
        let r = walk.evolve(&psi0, 0.0, 3.0, &mut c, &opts).unwrap();
        let err = distance(&r.state, &reference);
        assert!(err < tol, "tol {tol}: error {err}");
        assert!((norm(&r.state) - 1.0).abs() < 1e-10);
        assert!(r.error_estimate < tol);
    }
}

#[test]
fn local_adiabatic_search_beats_the_linear_schedule() {
    // Roland and Cerf (2002): H(s) = (1 − s)(I − |s⟩⟨s|) + s(I − |m⟩⟨m|)
    // with ds/dt = ε·g(s)², g² = 1 − 4(1 − 1/N)s(1 − s), takes
    // T = N/(ε√(N−1))·arctan√(N−1) ≈ (π/2ε)√N. The identity is dropped.
    let n = 256;
    let marked = 3;
    let walk = DrivenWalk::new(
        n,
        vec![Term::Uniform, Term::projector(n, &[marked]).unwrap()],
    )
    .unwrap();
    let eps = 0.2;
    let rn = (n as f64 - 1.0).sqrt();
    let total = n as f64 / (eps * rn) * rn.atan();
    assert!((total / (n as f64).sqrt() - std::f64::consts::FRAC_PI_2 / eps).abs() < 0.5);
    let local = |t: f64| 0.5 * (1.0 + (2.0 * eps * rn * t / n as f64 - rn.atan()).tan() / rn);
    let psi0 = vec![C64::new(1.0 / (n as f64).sqrt(), 0.0); n];
    let opts = MagnusOptions::default();
    let success = |schedule: &dyn Fn(f64) -> f64| {
        let r = walk
            .evolve(
                &psi0,
                0.0,
                total,
                |t| Ok(vec![-(1.0 - schedule(t)), -schedule(t)]),
                &opts,
            )
            .unwrap();
        r.state[marked].norm_sqr()
    };
    let p_local = success(&local);
    let p_linear = success(&|t: f64| t / total);
    // An independent integration of the two-dimensional subspace (SciPy,
    // 20 000 midpoint steps) gives 0.996694 and 0.310618.
    assert!((p_local - 0.996694).abs() < 1e-5, "{p_local}");
    assert!((p_linear - 0.310618).abs() < 1e-5, "{p_linear}");
    assert!(p_local > 1.0 - eps * eps);
    assert!((local(0.0)).abs() < 1e-12 && (local(total) - 1.0).abs() < 1e-12);
}

#[test]
fn invalid_inputs_are_rejected() {
    let g = generators::cycle(4);
    assert!(DrivenWalk::new(4, vec![]).is_err());
    assert!(DrivenWalk::new(5, vec![Term::graph(&g, Generator::Laplacian)]).is_err());
    let skew = CsrMatrix::from_triplets(2, &[(0, 1, 1.0)]).unwrap();
    assert!(DrivenWalk::new(2, vec![Term::Matrix(skew)]).is_err());
    assert!(Term::projector(4, &[4]).is_err());
    assert!(Term::potential(&[f64::NAN]).is_err());
    let walk = DrivenWalk::new(4, vec![Term::graph(&g, Generator::Laplacian)]).unwrap();
    let psi = localized(4, 0);
    let opts = MagnusOptions::default();
    assert!(
        walk.evolve(&psi, 0.0, 1.0, |_| Ok(vec![1.0, 2.0]), &opts)
            .is_err()
    );
    assert!(
        walk.evolve(&psi, 0.0, 1.0, |_| Ok(vec![f64::NAN]), &opts)
            .is_err()
    );
    assert!(
        walk.evolve(&psi, 1.0, 0.0, |_| Ok(vec![1.0]), &opts)
            .is_err()
    );
    assert!(
        walk.evolve(&psi[..3], 0.0, 1.0, |_| Ok(vec![1.0]), &opts)
            .is_err()
    );
    let failing = |_: f64| Err(QsoError::invalid("c", "boom"));
    assert!(walk.evolve(&psi, 0.0, 1.0, failing, &opts).is_err());
    let r = walk
        .evolve(&psi, 1.0, 1.0, |_| Ok(vec![1.0]), &opts)
        .unwrap();
    assert_eq!(r.state, psi);
}

#[test]
fn tight_tolerances_and_stiff_walks_still_finish() {
    let (walk, mut c) = driven_cycle();
    let psi0 = localized(8, 1);
    let reference = walk
        .evolve(
            &psi0,
            0.0,
            2.0,
            &mut c,
            &MagnusOptions {
                fixed_step: Some(1.0 / 2048.0),
                tol: 1e-14,
                ..MagnusOptions::default()
            },
        )
        .unwrap()
        .state;
    for tol in [1e-12, 1e-14, 1e-16] {
        let opts = MagnusOptions {
            tol,
            ..MagnusOptions::default()
        };
        let r = walk.evolve(&psi0, 0.0, 2.0, &mut c, &opts).unwrap();
        assert!(distance(&r.state, &reference) < 1e-11, "tol {tol}");
        assert!(r.steps < 20_000, "tol {tol}: {} steps", r.steps);
    }
    // A cycle of 200 nodes with ‖H‖ ≈ 30.
    let g = generators::cycle(200);
    let stiff = DrivenWalk::new(200, vec![Term::graph(&g, Generator::Laplacian)]).unwrap();
    let opts = MagnusOptions {
        tol: 1e-12,
        ..MagnusOptions::default()
    };
    let r = stiff
        .evolve(
            &localized(200, 0),
            0.0,
            1.0,
            |t| Ok(vec![7.5 * (1.0 + t)]),
            &opts,
        )
        .unwrap();
    assert!((norm(&r.state) - 1.0).abs() < 1e-10);
    // A coefficient with an infinite derivative at t = 0.7.
    let r = walk
        .evolve(
            &psi0,
            0.0,
            1.4,
            |t| Ok(vec![1.0, (t - 0.7).abs().sqrt(), 0.0]),
            &MagnusOptions::default(),
        )
        .unwrap();
    assert!(r.forced < 100);
}

#[test]
fn max_step_resolves_pulses_and_jumps_are_split() {
    let (walk, _) = driven_cycle();
    let psi0 = localized(8, 0);
    // A Gaussian pulse of width 0.01 at t = 0.83 on the potential.
    let pulse = |t: f64| Ok(vec![1.0, 30.0 * (-((t - 0.83) / 0.01).powi(2)).exp(), 0.0]);
    let fine = MagnusOptions {
        fixed_step: Some(1e-4),
        tol: 1e-12,
        ..MagnusOptions::default()
    };
    let reference = walk.evolve(&psi0, 0.0, 2.0, pulse, &fine).unwrap().state;
    let opts = MagnusOptions {
        max_step: Some(0.005),
        ..MagnusOptions::default()
    };
    let r = walk.evolve(&psi0, 0.0, 2.0, pulse, &opts).unwrap();
    assert!(distance(&r.state, &reference) < 1e-7);
    // A jump at t = 0.5, integrated in two pieces.
    let jump = |t: f64| Ok(vec![1.0, if t < 0.5 { 0.0 } else { 2.0 }, 0.0]);
    let before = |_: f64| Ok(vec![1.0, 0.0, 0.0]);
    let after = |_: f64| Ok(vec![1.0, 2.0, 0.0]);
    let defaults = MagnusOptions::default();
    let mid = walk
        .evolve(&psi0, 0.0, 0.5, before, &defaults)
        .unwrap()
        .state;
    let split = walk.evolve(&mid, 0.5, 2.0, after, &defaults).unwrap().state;
    let reference = walk
        .evolve(
            &psi0,
            0.0,
            2.0,
            jump,
            &MagnusOptions {
                fixed_step: Some(1.0 / 1024.0),
                ..fine.clone()
            },
        )
        .unwrap()
        .state;
    // 0.5 is a multiple of the fixed step, so the reference is exact too.
    assert!(distance(&split, &reference) < 1e-8);
    assert!(
        walk.evolve(
            &psi0,
            0.0,
            1.0,
            before,
            &MagnusOptions {
                max_step: Some(0.0),
                ..defaults.clone()
            }
        )
        .is_err()
    );
    // Steps that do not advance time in floating point are an error.
    let far = MagnusOptions {
        fixed_step: Some(1e-5),
        ..MagnusOptions::default()
    };
    assert!(walk.evolve(&psi0, 1e12, 1e12 + 1e-3, before, &far).is_err());
}
