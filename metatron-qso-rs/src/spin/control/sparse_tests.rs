//! Optimal control on sparse blocks: the sparse generator against the
//! dense one up to six spins, central differences beyond, the orbits of
//! equivalent spins against the full space, the unitary bound without
//! relaxation, and optimised pulses propagated afterwards.

use super::sparse::Amplitudes;
use super::*;
use crate::spin::{Isotope, axial_csa};

/// Value and gradient on `path`.
fn on(sys: &SpinSystem, c: &Control, u: &[f64], path: Path) -> (f64, Vec<f64>) {
    let (value, grad, _) = sys.control_members_on(c, u, path).unwrap();
    (value, grad)
}

const SPARSE: Path = Path::Sparse { symmetric: true };
const FULL: Path = Path::Sparse { symmetric: false };

/// Relative differences of the values (against `scale`) and of the
/// gradients (against the largest entry of `b`'s).
fn differ(a: &(f64, Vec<f64>), b: &(f64, Vec<f64>), scale: f64) -> (f64, f64) {
    let largest = b.1.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    assert!(largest > 0.0);
    let grad =
        a.1.iter()
            .zip(&b.1)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
    ((a.0 - b.0).abs() / scale, grad / largest)
}

/// `n` protons with shifts over 2 ppm, couplings to the next two, `T₁`
/// and `T₂`.
fn protons(n: usize) -> SpinSystem {
    let mut sys = SpinSystem::new(9.4);
    for i in 0..n {
        let k = sys.add_spin(Isotope::H1, 1.0 + 0.4 * i as f64).unwrap();
        sys.set_relaxation(k, Some(1.5 + 0.2 * i as f64), Some(0.4 + 0.05 * i as f64))
            .unwrap();
    }
    for i in 0..n - 1 {
        sys.couple(i, i + 1, 7.0 + i as f64).unwrap();
        if i + 2 < n {
            sys.couple(i, i + 2, -1.5).unwrap();
        }
    }
    sys
}

/// A ¹³C methyl group (carbon 0, protons 1–3 about the z axis) with
/// `extra` protons on the axis below it: Redfield relaxation (dipolar
/// couplings, the carbon's shielding anisotropy), `T₁` and `T₂` of the
/// protons, threefold symmetric and declared so.
fn methyl(extra: usize) -> SpinSystem {
    let mut sys = SpinSystem::new(9.4);
    sys.add_spin(Isotope::C13, 20.0).unwrap();
    sys.set_position(0, [0.0, 0.0, 0.0]).unwrap();
    for k in 0..3 {
        let theta = 2.0 * std::f64::consts::PI * k as f64 / 3.0;
        let i = sys.add_spin(Isotope::H1, 1.2).unwrap();
        sys.set_position(i, [1.03 * theta.cos(), 1.03 * theta.sin(), 0.36])
            .unwrap();
        sys.couple(0, i, 125.0).unwrap();
        sys.set_relaxation(i, Some(2.0), Some(0.8)).unwrap();
    }
    for (i, j) in [(1, 2), (2, 3), (1, 3)] {
        sys.couple(i, j, -12.0).unwrap();
    }
    for e in 0..extra {
        let i = sys.add_spin(Isotope::H1, 2.0 + 0.6 * e as f64).unwrap();
        sys.set_position(i, [0.0, 0.0, -1.5 - 1.2 * e as f64])
            .unwrap();
        for m in 1..4 {
            sys.couple(m, i, 3.0 + e as f64).unwrap();
        }
        if e > 0 {
            sys.couple(i - 1, i, 6.5).unwrap();
        }
        sys.set_relaxation(i, Some(1.0), Some(0.5)).unwrap();
    }
    sys.set_csa(0, Some(axial_csa(30.0, [0.0, 0.0, 1.0]).unwrap()))
        .unwrap();
    let mut sys = sys.with_correlation_time(Some(5e-11)).unwrap();
    let mut cycle: Vec<usize> = (0..sys.spins().len()).collect();
    cycle[1..4].copy_from_slice(&[2, 3, 1]);
    sys.set_symmetries(vec![cycle]).unwrap();
    sys
}

fn channels() -> Vec<Target> {
    vec![Target::Isotope(Isotope::H1), Target::Isotope(Isotope::C13)]
}

/// Two members with offsets and rf errors on two channels.
fn ensemble() -> Vec<Member> {
    vec![
        Member::ideal(),
        Member {
            weight: 0.5,
            scales: vec![0.9, 1.1],
            offsets: vec![40.0, -25.0],
        },
    ]
}

/// Value and gradient on sparse blocks with the derivatives of
/// [`frechet`] instead of the Gauss–Legendre nodes.
fn augmented(sys: &SpinSystem, c: &Control, u: &[f64], symmetric: bool) -> (f64, Vec<f64>) {
    let mut blocks = sys.sparse_control(c, symmetric).unwrap();
    blocks.quadrature = false;
    let (value, grad, _) = sys
        .evaluate_control(c, u, &Prepared::Sparse(Box::new(blocks)))
        .unwrap();
    (value, grad)
}

#[test]
fn sparse_gradients_equal_the_dense_ones_up_to_six_spins() {
    // The Gauss–Legendre gradients on the orbits and in full, and those of
    // the augmented exponentials on sparse blocks, against the dense
    // generator of 1.36.
    let check = |name: &str, sys: &SpinSystem, c: &Control, u: &[f64]| {
        let dense = on(sys, c, u, Path::Dense);
        let scale = dense.0.abs();
        let mut results = vec![
            ("orbits".to_string(), on(sys, c, u, SPARSE)),
            ("full".to_string(), on(sys, c, u, FULL)),
        ];
        results.push(("augmented".to_string(), augmented(sys, c, u, true)));
        for (what, got) in results {
            let (dv, dg) = differ(&got, &dense, scale);
            eprintln!("{name} {what}: value {dv:.1e}, gradient {dg:.1e}");
            assert!(dv < 1e-11 && dg < 1e-10, "{name} {what}: {dv:e} {dg:e}");
        }
    };
    // A ¹H–¹³C pair with T₁ and T₂, offsets and rf errors.
    let mut pair = SpinSystem::new(9.4);
    pair.add_spin(Isotope::H1, 0.3).unwrap();
    pair.add_spin(Isotope::C13, 0.0).unwrap();
    pair.couple(0, 1, 140.0).unwrap();
    pair.set_relaxation(0, Some(0.02), Some(0.01)).unwrap();
    pair.set_relaxation(1, Some(0.05), None).unwrap();
    let mut c = Control::rf(
        channels(),
        None,
        4e-3,
        8,
        pair.iz(0).unwrap(),
        pair.iz(1).unwrap(),
    );
    c.ensemble = ensemble();
    let u = c.random_start(3, 400.0);
    check("pair", &pair, &c, &u);
    // Redfield relaxation from the thermal state at 50 mK, so that the
    // drift towards equilibrium matters.
    let mut red = SpinSystem::new(9.4);
    red.add_spin(Isotope::H1, 1.0).unwrap();
    red.add_spin(Isotope::C13, 0.0).unwrap();
    red.couple(0, 1, 140.0).unwrap();
    red.set_position(0, [0.0, 0.0, 0.0]).unwrap();
    red.set_position(1, [1.09, 0.0, 0.0]).unwrap();
    let red = red
        .with_correlation_time(Some(1e-9))
        .unwrap()
        .with_temperature(0.05)
        .unwrap();
    let mut thermal = Control::rf(
        channels(),
        None,
        2e-3,
        6,
        red.thermal_state().unwrap(),
        red.iz(1).unwrap(),
    );
    thermal.ensemble = ensemble();
    check("redfield", &red, &thermal, &thermal.random_start(2, 400.0));
    // A spin 1.
    let mut deut = SpinSystem::new(9.4);
    deut.add_spin(Isotope::H2, 0.0).unwrap();
    deut.add_spin(Isotope::C13, 0.0).unwrap();
    deut.couple(0, 1, 20.0).unwrap();
    deut.set_relaxation(0, Some(0.01), None).unwrap();
    let deut = deut.with_temperature(0.01).unwrap();
    let spin1 = Control::rf(
        vec![Target::Isotope(Isotope::H2), Target::Isotope(Isotope::C13)],
        None,
        5e-3,
        6,
        deut.thermal_state().unwrap(),
        deut.iz(1).unwrap(),
    );
    check("spin 1", &deut, &spin1, &spin1.random_start(3, 300.0));
    // A methyl group with a proton below it (five spins), from the
    // thermal state onto the carbon: on the orbits and in full.
    let five = methyl(1);
    let mut cold = five.clone().with_temperature(0.05).unwrap();
    cold.set_symmetries(five.symmetries().to_vec()).unwrap();
    let mut c = Control::rf(
        channels(),
        Some(3000.0),
        2e-4,
        2,
        cold.thermal_state().unwrap(),
        cold.iz(0).unwrap(),
    );
    c.ensemble = ensemble();
    let u = c.random_start(4, 1000.0);
    check("methyl", &cold, &c, &u);
    // Six coupled protons with T₁ and T₂, one channel.
    let six = protons(6);
    let mut fz = six.iz(0).unwrap();
    for i in 1..6 {
        fz += six.iz(i).unwrap();
    }
    let c = Control::rf(
        vec![Target::Isotope(Isotope::H1)],
        Some(2000.0),
        4e-5,
        2,
        fz,
        six.iz(5).unwrap(),
    );
    check("six protons", &six, &c, &c.random_start(5, 1000.0));
}

#[test]
fn small_deviations_keep_their_digits() {
    // Without a drift towards equilibrium (T₂ alone) the transfer is linear
    // in the deviation: from 1/d + ε·F_z it is ε times that from F_z, to
    // the rounding of the deviation however small ε is against the trace
    // (ε = 2⁻⁴⁰, so that 1/d + ε·F_z holds the deviation exactly). Series
    // that stopped relative to the trace lost digits in proportion.
    let mut sys = SpinSystem::new(9.4);
    sys.add_spin(Isotope::C13, 20.0).unwrap();
    for _ in 0..3 {
        let i = sys.add_spin(Isotope::H1, 1.2).unwrap();
        sys.couple(0, i, 125.0).unwrap();
        sys.set_relaxation(i, None, Some(0.01)).unwrap();
    }
    sys.set_relaxation(0, None, Some(0.02)).unwrap();
    sys.set_symmetries(vec![vec![0, 2, 3, 1]]).unwrap();
    let d = sys.dimension();
    let fz = sys.fz(Isotope::H1).unwrap();
    let eps = 2f64.powi(-40);
    let mixed = CMatrix::identity(d, d) * C64::new(1.0 / d as f64, 0.0) + &fz * C64::new(eps, 0.0);
    let target = sys.iz(0).unwrap();
    let plain = Control::rf(channels(), None, 2e-3, 4, fz, target.clone());
    let small = Control::rf(channels(), None, 2e-3, 4, mixed, target);
    let u = plain.random_start(3, 1500.0);
    for path in [SPARSE, FULL, Path::Dense] {
        let (value, grad) = on(&sys, &plain, &u, path);
        let (v, g) = on(&sys, &small, &u, path);
        let largest = grad.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let dv = (v - eps * value).abs() / (eps * value.abs());
        let dg = g
            .iter()
            .zip(&grad)
            .map(|(a, b)| (a - eps * b).abs())
            .fold(0.0, f64::max)
            / (eps * largest);
        eprintln!("{path:?}: value {dv:.1e}, gradient {dg:.1e}");
        assert!(dv < 1e-12 && dg < 1e-11, "{path:?}: {dv:e} {dg:e}");
    }
}

/// `(f(u + hv) − f(u − hv))/2h` against `∇f·v` for a seeded direction.
fn directional(
    sys: &SpinSystem,
    c: &Control,
    u: &[f64],
    path: Path,
    seed: u64,
    h: f64,
) -> (f64, f64) {
    let mut rng = crate::rng::Rng::seed_from_u64(seed);
    let v: Vec<f64> = (0..u.len()).map(|_| 2.0 * rng.uniform() - 1.0).collect();
    let (_, grad) = on(sys, c, u, path);
    let shifted = |s: f64| {
        let p: Vec<f64> = u.iter().zip(&v).map(|(a, b)| a + s * b).collect();
        on(sys, c, &p, path).0
    };
    let fd = (shifted(h) - shifted(-h)) / (2.0 * h);
    let exact: f64 = grad.iter().zip(&v).map(|(a, b)| a * b).sum();
    (exact, fd)
}

#[test]
fn sparse_gradients_agree_with_central_differences_beyond_six_spins() {
    // A methyl group with three protons below it (seven spins, 16 385
    // coefficients, 6145 orbits), relaxing, from the thermal state at
    // 50 mK onto the carbon, with an ensemble: along random directions
    // and for single controls.
    let seven = methyl(3);
    let mut cold = seven.clone().with_temperature(0.05).unwrap();
    cold.set_symmetries(seven.symmetries().to_vec()).unwrap();
    let mut c = Control::rf(
        channels(),
        Some(3000.0),
        1e-3,
        3,
        cold.thermal_state().unwrap(),
        cold.iz(0).unwrap(),
    );
    c.ensemble = ensemble();
    let u = c.random_start(7, 1500.0);
    let blocks = cold.sparse_control(&c, true).unwrap();
    assert!(blocks.reduced());
    assert_eq!(blocks.size(), 6145);
    for seed in 0..2 {
        let (exact, fd) = directional(&cold, &c, &u, SPARSE, seed, 1e-2);
        eprintln!("direction {seed}: {exact:e} vs {fd:e}");
        assert!((exact - fd).abs() < 1e-7 * exact.abs(), "{exact} vs {fd}");
    }
    let (_, grad) = on(&cold, &c, &u, SPARSE);
    let largest = grad.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    for i in [0, 5, 9] {
        let mut p = u.clone();
        p[i] += 1e-2;
        let up = on(&cold, &c, &p, SPARSE).0;
        p[i] -= 2e-2;
        let down = on(&cold, &c, &p, SPARSE).0;
        let fd = (up - down) / 2e-2;
        assert!(
            (grad[i] - fd).abs() < 1e-7 * largest,
            "{i}: {} vs {fd}",
            grad[i]
        );
    }
}

#[test]
fn orbits_give_the_full_transfer_and_gradient() {
    // Five spins (1025 coefficients, 385 orbits): a target that breaks
    // the symmetry still pairs with the orbits; an initial state or a
    // channel that breaks it propagates in full.
    let sys = methyl(1);
    let thermal = sys.thermal_state().unwrap();
    let mut c = Control::rf(
        channels(),
        Some(3000.0),
        3e-3,
        6,
        thermal.clone(),
        sys.iz(1).unwrap(),
    );
    c.ensemble = ensemble();
    let u = c.random_start(1, 1500.0);
    let reduced = sys.sparse_control(&c, true).unwrap();
    assert!(reduced.reduced());
    assert_eq!(
        (
            reduced.size(),
            sys.sparse_control(&c, false).unwrap().size()
        ),
        (385, 1025)
    );
    let full = on(&sys, &c, &u, FULL);
    let (dv, dg) = differ(&on(&sys, &c, &u, SPARSE), &full, full.0.abs());
    eprintln!("orbits against full: {dv:.1e} {dg:.1e}");
    assert!(dv < 1e-12 && dg < 1e-11, "{dv:e} {dg:e}");
    let mut broken = c.clone();
    broken.initial = sys.iz(1).unwrap();
    assert!(!sys.sparse_control(&broken, true).unwrap().reduced());
    let mut one = c.clone();
    one.drive = Drive::Rf {
        targets: vec![Target::Spin(1), Target::Isotope(Isotope::C13)],
        limit: Some(3000.0),
    };
    assert!(!sys.sparse_control(&one, true).unwrap().reduced());
    one.duration = 2e-4;
    one.steps = 2;
    one.ensemble = Vec::new();
    let short = &u[..8];
    let dense = on(&sys, &one, short, Path::Dense);
    let (dv, dg) = differ(&on(&sys, &one, short, SPARSE), &dense, dense.0.abs());
    assert!(dv < 1e-11 && dg < 1e-10, "{dv:e} {dg:e}");
}

/// The best of `seeds` optimisations on `path` from random starts.
fn best_on(
    sys: &SpinSystem,
    c: &Control,
    path: Path,
    seeds: u64,
    scale: f64,
    iterations: usize,
) -> Optimized {
    let options = Options {
        max_iterations: iterations,
        g_tol: 1e-10,
        ..Options::default()
    };
    (0..seeds)
        .map(|s| {
            sys.optimize_control_on(c, &c.random_start(s, scale), &options, path)
                .unwrap()
        })
        .max_by(|a, b| a.transfer.total_cmp(&b.transfer))
        .unwrap()
}

#[test]
fn without_relaxation_sparse_blocks_reach_the_unitary_bound() {
    // The problems of 1.36 on sparse blocks: ¹H polarisation onto ¹³C in
    // a pair and in I₂S, where Hilbert space reaches the bound.
    let mut pair = SpinSystem::new(9.4);
    pair.add_spin(Isotope::H1, 0.0).unwrap();
    pair.add_spin(Isotope::C13, 0.0).unwrap();
    pair.couple(0, 1, 140.0).unwrap();
    let mut i2s = SpinSystem::new(9.4);
    i2s.add_spin(Isotope::H1, 0.0).unwrap();
    i2s.add_spin(Isotope::H1, 0.0).unwrap();
    i2s.add_spin(Isotope::C13, 0.0).unwrap();
    i2s.couple(0, 2, 140.0).unwrap();
    i2s.couple(1, 2, 140.0).unwrap();
    i2s.set_symmetries(vec![vec![1, 0, 2]]).unwrap();
    for (sys, limit, duration, steps, iterations) in [
        (&pair, 5000.0, 1.2 / 140.0, 40, 400),
        (&i2s, 10_000.0, 2.0 / 140.0, 80, 1000),
    ] {
        let rho = sys.thermal_state().unwrap();
        let carbon = sys.spins().len() - 1;
        let sz = sys.iz(carbon).unwrap();
        let bound = unitary_bound(&rho, &sz).unwrap();
        let c = Control::rf(channels(), Some(limit), duration, steps, rho, sz);
        let got = best_on(sys, &c, SPARSE, 3, 500.0, iterations);
        eprintln!(
            "{} spins: {:.9} of the bound",
            sys.spins().len(),
            got.transfer / bound
        );
        assert!(got.transfer <= bound * (1.0 + 1e-9));
        assert!(
            got.transfer >= (1.0 - 1e-6) * bound,
            "{} vs {bound}",
            got.transfer
        );
        // Hilbert space gives the same transfer and gradient at a random
        // start (to the rounding of ρ itself, which Hilbert space
        // propagates, about 10⁻¹¹ of a thermal deviation).
        let u = c.random_start(9, 1000.0);
        let hilbert = on(sys, &c, &u, Path::Hilbert);
        let (dv, dg) = differ(&on(sys, &c, &u, SPARSE), &hilbert, bound);
        assert!(dv < 1e-9 && dg < 1e-9, "{dv:e} {dg:e}");
    }
}

#[test]
fn optimised_pulses_propagate_to_the_reported_transfer() {
    // Seven spins with relaxation, optimised on the orbits, propagated
    // afterwards by evolve (Krylov steps), and a pair (dense
    // exponentials).
    let seven = methyl(3);
    let rho = seven.thermal_state().unwrap();
    let target = seven.iz(0).unwrap();
    let c = Control::rf(
        channels(),
        Some(2000.0),
        2e-3,
        8,
        rho.clone(),
        target.clone(),
    );
    let options = Options {
        max_iterations: 10,
        ..Options::default()
    };
    let start = c.random_start(2, 1000.0);
    let got = seven.optimize_control(&c, &start, &options).unwrap();
    let (first, _) = seven.control_transfer(&c, &start).unwrap();
    eprintln!("seven spins: {first:e} → {:e}", got.transfer);
    assert!(got.transfer > first);
    let after = seven
        .evolve(&rho, &seven.control_events(&c, &got.controls).unwrap())
        .unwrap();
    let want = seven.expectation(&after, &target).unwrap().re;
    eprintln!(
        "replayed by evolve: {:.1e}",
        (got.transfer - want).abs() / want.abs()
    );
    assert!(
        (got.transfer - want).abs() < 1e-9 * want.abs(),
        "{} vs {want}",
        got.transfer
    );
    let mut pair = SpinSystem::new(9.4);
    pair.add_spin(Isotope::H1, 0.0).unwrap();
    pair.add_spin(Isotope::C13, 0.0).unwrap();
    pair.couple(0, 1, 140.0).unwrap();
    pair.set_relaxation(0, Some(0.05), Some(0.02)).unwrap();
    pair.set_relaxation(1, Some(0.5), Some(0.02)).unwrap();
    let rho = pair.thermal_state().unwrap();
    let sz = pair.iz(1).unwrap();
    let c = Control::rf(
        channels(),
        Some(5000.0),
        1.2 / 140.0,
        40,
        rho.clone(),
        sz.clone(),
    );
    let got = pair
        .optimize_control(&c, &c.random_start(0, 500.0), &options)
        .unwrap();
    let after = pair
        .evolve(&rho, &pair.control_events(&c, &got.controls).unwrap())
        .unwrap();
    let want = pair.expectation(&after, &sz).unwrap().re;
    assert!((got.transfer - want).abs() < 1e-10 * want.abs());
}

#[test]
fn large_problems_are_estimated_before_anything_is_built() {
    // Seven relaxing protons take rf controls now; field profiles stay
    // within six spins.
    let seven = protons(7);
    let rf = Control::rf(
        vec![Target::Isotope(Isotope::H1)],
        Some(1000.0),
        1e-3,
        4,
        seven.iz(0).unwrap(),
        seven.iz(1).unwrap(),
    );
    assert!(seven.control_transfer(&rf, &[0.0; 8]).is_ok());
    let field = Control::field(
        9.0,
        9.4,
        1e-3,
        2,
        seven.iz(0).unwrap(),
        seven.iz(1).unwrap(),
    );
    let err = seven.control_transfer(&field, &[9.2, 9.3]).unwrap_err();
    assert!(err.to_string().contains("Liouville"), "{err}");
    // Ten relaxing protons over 10⁵ steps: refused from the bounds alone,
    // in about a second (69 s when the entries were counted first; the
    // bound leaves room for a loaded machine running the other tests).
    let ten = protons(10);
    let long = Control::rf(
        vec![Target::Isotope(Isotope::H1)],
        Some(1000.0),
        1.0,
        100_000,
        ten.iz(0).unwrap(),
        ten.iz(1).unwrap(),
    );
    let clock = std::time::Instant::now();
    let err = ten
        .control_transfer(&long, &vec![0.0; 200_000])
        .unwrap_err();
    assert!(err.to_string().contains("needs about"), "{err}");
    let seconds = clock.elapsed().as_secs_f64();
    assert!(seconds < 20.0, "{seconds} s");
    // The estimate before building against the one from the blocks.
    for (sys, c) in [
        (
            methyl(1),
            Control::rf(
                channels(),
                Some(3000.0),
                3e-3,
                5,
                methyl(1).thermal_state().unwrap(),
                methyl(1).iz(0).unwrap(),
            ),
        ),
        (
            protons(5),
            Control::rf(
                vec![Target::Isotope(Isotope::H1)],
                Some(2000.0),
                1e-3,
                4,
                protons(5).iz(0).unwrap(),
                protons(5).iz(1).unwrap(),
            ),
        ),
    ] {
        let (work, memory) = sys.sparse_cost(&c, Amplitudes::Peak(3000.0), true).unwrap();
        let blocks = sys.sparse_control(&c, true).unwrap();
        let (exact_work, exact_memory) = blocks.cost(&c, &vec![3000.0; c.len()]);
        eprintln!(
            "estimate {work:.2e} ops {memory:.2e} B, from the blocks {exact_work:.2e} ops \
             {exact_memory:.2e} B"
        );
        assert!(work >= 0.9 * exact_work && work <= 3.0 * exact_work);
        assert!(memory >= exact_memory && memory <= 4.0 * exact_memory);
    }
}

/// Seconds per evaluation of methyl groups with 1–4 protons below them
/// on each path (run with `--ignored --nocapture`).
#[test]
#[ignore]
fn measure_evaluations() {
    for extra in 1..=4 {
        let sys = methyl(extra);
        let n = sys.spins().len();
        let c = Control::rf(
            channels(),
            Some(5000.0),
            1e-3,
            10,
            sys.thermal_state().unwrap(),
            sys.iz(0).unwrap(),
        );
        let u = c.random_start(1, 2500.0);
        for path in [SPARSE, FULL, Path::Dense] {
            let clock = std::time::Instant::now();
            let blocks =
                (path != Path::Dense).then(|| sys.sparse_control(&c, path == SPARSE).unwrap());
            let build = clock.elapsed().as_secs_f64();
            let estimate = sys.path_cost(&c, Amplitudes::Peak(5000.0), path);
            let clock = std::time::Instant::now();
            let result = sys.control_members_on(&c, &u, path);
            let total = clock.elapsed().as_secs_f64();
            eprintln!(
                "{n} spins {path:?}: size {:?} MB {:?}, build {build:.2} s, evaluation {total:.2} s, \
                 estimate {:?}, from the blocks {:?}, value {:?}",
                blocks.as_ref().map(|b| b.size()),
                blocks.as_ref().map(|b| (b.bytes() / 1e6).round()),
                estimate
                    .as_ref()
                    .map(|(w, m)| (format!("{w:.2e}"), format!("{:.2}", m / 1e9))),
                blocks.as_ref().map(|b| {
                    let (w, m) = b.cost(&c, &vec![5000.0; c.len()]);
                    (format!("{w:.2e}"), format!("{:.2}", m / 1e9))
                }),
                result.map(|r| r.0)
            );
            if let Some(b) = &blocks {
                let (base, ops, base_norm, op_norms) = b.entries();
                super::sparse::PRODUCTS.store(0, std::sync::atomic::Ordering::Relaxed);
                let clock = std::time::Instant::now();
                let r = sys.evaluate_control(
                    &c,
                    &u,
                    &Prepared::Sparse(Box::new(sys.sparse_control(&c, path == SPARSE).unwrap())),
                );
                let time = clock.elapsed().as_secs_f64();
                let products = super::sparse::PRODUCTS.load(std::sync::atomic::Ordering::Relaxed);
                eprintln!(
                    "    entries {base} + {ops}, norms {base_norm:.3e} {op_norms:.3?}, products {products}, \
                     {time:.2} s, cost of u {:.2e}, ok {}",
                    b.cost(&c, &u).0,
                    r.is_ok()
                );
            }
            if let Some(mut b) = blocks {
                b.quadrature = false;
                let clock = std::time::Instant::now();
                let r = sys.evaluate_control(&c, &u, &Prepared::Sparse(Box::new(b)));
                eprintln!(
                    "    augmented exponentials: {:.2} s, value {:?}",
                    clock.elapsed().as_secs_f64(),
                    r.map(|r| r.0)
                );
            }
        }
    }
}

/// Refocused INEPT for a CH₃ group with ideal pulses: `τ = 1/(4J)`, and a
/// refocusing period `Δ` with `tan²(πJΔ) = 1/2`, the optimum for three
/// protons.
fn refocused_inept(j: f64) -> (Vec<Event>, f64) {
    let (h, cc) = (Target::Isotope(Isotope::H1), Target::Isotope(Isotope::C13));
    let tau = 1.0 / (4.0 * j);
    let delta = (0.5f64.sqrt()).atan() / (std::f64::consts::PI * j);
    let events = vec![
        Event::pulse(h, 90.0, 0.0),
        Event::Delay(tau),
        Event::pulse(h, 180.0, 0.0),
        Event::pulse(cc, 180.0, 0.0),
        Event::Delay(tau),
        Event::pulse(h, 90.0, 90.0),
        Event::pulse(cc, 90.0, 0.0),
        Event::Delay(delta / 2.0),
        Event::pulse(h, 180.0, 0.0),
        Event::pulse(cc, 180.0, 0.0),
        Event::Delay(delta / 2.0),
        Event::pulse(cc, 90.0, 270.0),
    ];
    (events, 2.0 * tau + delta)
}

/// GRAPE on seven and eight relaxing spins against refocused INEPT of the
/// same duration (run with `--ignored --nocapture`).
#[test]
#[ignore]
fn measure_optimisation() {
    for extra in [3, 4] {
        let sys = methyl(extra);
        let n = sys.spins().len();
        let rho = sys.thermal_state().unwrap();
        let target = sys.iz(0).unwrap();
        let own = sys.expectation(&rho, &target).unwrap().re;
        let bound = unitary_bound(&rho, &target).unwrap();
        let (events, duration) = refocused_inept(125.0);
        let clock = std::time::Instant::now();
        let after = sys.evolve(&rho, &events).unwrap();
        let inept = sys.expectation(&after, &target).unwrap().re;
        eprintln!(
            "{n} spins: own {own:.4e}, bound {:.3} of own, refocused INEPT ({:.2} ms) {:.3} of own \
             ({:.1} s)",
            bound / own,
            duration * 1e3,
            inept / own,
            clock.elapsed().as_secs_f64()
        );
        let steps = 50;
        let c = Control::rf(
            channels(),
            Some(5000.0),
            duration,
            steps,
            rho.clone(),
            target.clone(),
        );
        let cost = sys.control_cost(&c, 5000.0).unwrap();
        eprintln!("    cost at 5 kHz: {cost:?}");
        let iterations = if extra == 3 { 60 } else { 4 };
        let options = Options {
            max_iterations: iterations,
            ..Options::default()
        };
        let start = c.random_start(1, 1000.0);
        let clock = std::time::Instant::now();
        match sys.optimize_control(&c, &start, &options) {
            Ok(got) => {
                let time = clock.elapsed().as_secs_f64();
                let replay = sys
                    .evolve(&rho, &sys.control_events(&c, &got.controls).unwrap())
                    .unwrap();
                let want = sys.expectation(&replay, &target).unwrap().re;
                eprintln!(
                    "    GRAPE {} iterations, {} evaluations in {time:.0} s ({:.1} s each): {:.3} of own, \
                     {:.3} of the bound; evolve differs by {:.1e}",
                    got.iterations,
                    got.evaluations,
                    time / got.evaluations as f64,
                    got.transfer / own,
                    got.transfer / bound,
                    (got.transfer - want).abs() / want.abs()
                );
            }
            Err(e) => eprintln!("    GRAPE refused: {e}"),
        }
    }
}
