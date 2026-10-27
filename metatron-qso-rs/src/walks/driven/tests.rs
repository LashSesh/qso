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
