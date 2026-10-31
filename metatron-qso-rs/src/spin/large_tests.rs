//! Relaxation of larger spin systems: sparse blocks propagated by Krylov
//! steps, the Redfield superoperator built column by column, and the
//! reduction by equivalent spins.

use super::blocks::Method;
use super::*;

/// The largest entry of `x − y` against the largest of `y − Tr(y)/d`.
fn relative(x: &CMatrix, y: &CMatrix) -> f64 {
    let d = y.nrows();
    let dev = y - CMatrix::identity(d, d) * (y.trace() / d as f64);
    let scale = dev.iter().fold(0.0f64, |m, z| m.max(z.norm()));
    (x - y).iter().fold(0.0f64, |m, z| m.max(z.norm())) / scale
}

/// `n` protons with shifts over 3 ppm, couplings to the next two, `T₁` and
/// `T₂`.
fn protons(n: usize, b: f64) -> SpinSystem {
    let mut sys = SpinSystem::new(b);
    for i in 0..n {
        let k = sys.add_spin(Isotope::H1, 1.0 + 0.43 * i as f64).unwrap();
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
/// `extra` protons on the axis below it, Redfield relaxation (dipolar
/// couplings and the carbon's axial shielding anisotropy) and `T₁`, `T₂`:
/// threefold symmetric, declared by the cycle of the protons.
fn methyl(extra: usize, b: f64, tau: f64) -> SpinSystem {
    let mut sys = SpinSystem::new(b);
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
        sys.set_relaxation(i, Some(1.0), Some(0.5)).unwrap();
    }
    sys.set_csa(0, Some(axial_csa(30.0, [0.0, 0.0, 1.0]).unwrap()))
        .unwrap();
    let mut sys = sys.with_correlation_time(Some(tau)).unwrap();
    let mut cycle: Vec<usize> = (0..sys.spins().len()).collect();
    cycle[1..4].copy_from_slice(&[2, 3, 1]);
    sys.set_symmetries(vec![cycle]).unwrap();
    sys
}

/// A ¹³C methylene group (protons 1 and 2 mirrored in x) with a proton
/// and a carbon in the mirror plane: symmetric under swapping 1 and 2.
fn methylene(b: f64, tau: f64) -> SpinSystem {
    let mut sys = SpinSystem::new(b);
    let places = [
        (Isotope::C13, 30.0, [0.0, 0.0, 0.0]),
        (Isotope::H1, 2.1, [0.89, 0.0, 0.63]),
        (Isotope::H1, 2.1, [-0.89, 0.0, 0.63]),
        (Isotope::H1, 4.2, [0.0, 1.2, -1.4]),
        (Isotope::C13, 170.0, [0.0, -1.3, -0.8]),
    ];
    for (k, (x, shift, p)) in places.into_iter().enumerate() {
        sys.add_spin(x, shift).unwrap();
        sys.set_position(k, p).unwrap();
    }
    for (i, j, jh) in [
        (0, 1, 140.0),
        (0, 2, 140.0),
        (1, 3, 6.5),
        (2, 3, 6.5),
        (1, 2, -14.0),
    ] {
        sys.couple(i, j, jh).unwrap();
    }
    sys.couple(0, 4, 55.0).unwrap();
    sys.set_relaxation(3, Some(3.0), Some(1.0)).unwrap();
    let mut sys = sys.with_correlation_time(Some(tau)).unwrap();
    sys.set_symmetries(vec![vec![0, 2, 1, 3, 4]]).unwrap();
    sys
}

fn pulsed(sys: &SpinSystem) -> CMatrix {
    sys.evolve(
        &sys.thermal_state().unwrap(),
        &[Event::pulse(Target::Isotope(Isotope::H1), 90.0, 30.0)],
    )
    .unwrap()
}

#[test]
fn krylov_steps_equal_the_dense_exponentials_up_to_six_spins() {
    // Phenomenological relaxation of six protons, Redfield relaxation of a
    // methyl group with a proton below (five spins) in both frames, and a
    // shaped pulse (one block of all elements): to 10⁻¹⁰ of the deviation.
    let rf = |sys: &SpinSystem| {
        sys.hamiltonian().unwrap() + sys.fx(Isotope::H1).unwrap() * C64::new(TWO_PI * 2e3, 0.0)
    };
    // Five protons in a chain with dipolar relaxation, in the laboratory
    // frame (one isotope, so its blocks hold no Larmor spread).
    let mut chain = protons(5, 9.4).with_frame(Frame::Laboratory);
    for i in 0..5 {
        chain
            .set_position(i, [1.8 * i as f64, 0.5 * (i % 2) as f64, 0.0])
            .unwrap();
    }
    let chain = chain.with_correlation_time(Some(5e-11)).unwrap();
    let cases: Vec<(SpinSystem, f64, bool)> = vec![
        (protons(6, 2.0), 0.3, true),
        (protons(6, 9.4), 0.05, true),
        (methyl(1, 9.4, 3e-11), 0.2, true),
        (chain, 0.02, true),
        (methyl(0, 9.4, 3e-11), 2e-4, false),
    ];
    for (sys, t, conserving) in cases {
        let h = if conserving {
            sys.hamiltonian().unwrap()
        } else {
            rf(&sys)
        };
        let relaxation = sys.relaxation(sys.field()).unwrap();
        for rho in [pulsed(&sys), sys.identity_state()] {
            let run = |m| {
                sys.relax_planned(&rho, &h, t, conserving, &relaxation, Some(m))
                    .unwrap()
            };
            let (dense, krylov) = (run(Method::Dense), run(Method::Krylov));
            let err = relative(&krylov, &dense);
            assert!(err < 1e-10, "{} spins, t = {t}: {err:e}", sys.spins().len());
        }
    }
}

#[test]
fn seven_protons_relax_as_dormand_prince_steps_and_beyond() {
    // Seven protons: the coherences' blocks hold up to 3432 elements,
    // beyond a dense exponential. Krylov steps equal Dormand–Prince steps
    // over 2 ms, and a delay of one second, which neither of those could
    // afford, runs by Krylov steps and decays as T₁ and T₂ require.
    let sys = protons(7, 1.0);
    let h = sys.hamiltonian().unwrap();
    let relaxation = sys.relaxation(1.0).unwrap();
    let rho = pulsed(&sys);
    let run = |m, t| {
        sys.relax_planned(&rho, &h, t, true, &relaxation, Some(m))
            .unwrap()
    };
    let err = relative(&run(Method::Krylov, 2e-3), &run(Method::Steps, 2e-3));
    assert!(err < 1e-9, "{err:e}");
    let sizes = sys.planned_sizes(&rho).unwrap();
    assert_eq!(sizes.iter().map(|s| s.0).max(), Some(3433));
    let later = sys.evolve(&rho, &[Event::Delay(1.0)]).unwrap();
    // ⟨I_z⟩ of each spin returns towards equilibrium; ⟨F₊⟩ decays by about
    // e^{−t/T₂} (couplings spread it, so only a bound).
    let thermal = sys.thermal_state().unwrap();
    let fp = sys.fplus(Isotope::H1).unwrap();
    let before = sys.expectation(&rho, &fp).unwrap().norm();
    let after = sys.expectation(&later, &fp).unwrap().norm();
    assert!(
        after < before * (-1.0f64 / 0.7).exp(),
        "{after} vs {before}"
    );
    for i in 0..7 {
        let iz = sys.iz(i).unwrap();
        let (z0, z1, zeq) = (
            sys.expectation(&rho, &iz).unwrap().re,
            sys.expectation(&later, &iz).unwrap().re,
            sys.expectation(&thermal, &iz).unwrap().re,
        );
        assert!((z1 - zeq).abs() < (z0 - zeq).abs(), "spin {i}");
    }
    // The free induction decay by Krylov steps, against the delays.
    let fid = sys.fid(&rho, Isotope::H1, 1e-3, 3).unwrap();
    for (k, s) in fid.iter().enumerate() {
        let at = run(Method::Krylov, k as f64 * 1e-3);
        let want = sys.expectation(&at, &fp).unwrap();
        assert!(
            (s - want).norm() < 1e-10 * want.norm(),
            "{k}: {s} vs {want}"
        );
    }
}

#[test]
fn the_free_induction_decay_agrees_between_methods() {
    // Six protons and a methyl group with Redfield relaxation: dense,
    // Krylov and Dormand–Prince propagation of the decay agree.
    for sys in [protons(6, 2.0), methyl(2, 9.4, 3e-11)] {
        let rho = pulsed(&sys);
        let run = |m| {
            sys.relaxing_fid_planned(&rho, Isotope::H1, 2e-4, 6, Some(m))
                .unwrap()
        };
        let (dense, krylov, steps) = (run(Method::Dense), run(Method::Krylov), run(Method::Steps));
        let scale = dense[0].norm();
        for k in 0..6 {
            assert!((krylov[k] - dense[k]).norm() < 1e-10 * scale, "{k}");
            assert!((steps[k] - dense[k]).norm() < 1e-8 * scale, "{k}");
        }
    }
}

#[test]
fn equivalent_spins_reduce_the_blocks_exactly() {
    // A methyl group with three protons on its axis (seven spins) and a
    // methylene group: on the orbits of the symmetry the deviation
    // propagates as in full, in blocks about 8/3 and 8/5 times smaller
    // (the fixed elements of a threefold cycle are 1/16 of all, of a
    // swap 1/4).
    for (sys, t, factor) in [
        (methyl(3, 9.4, 3e-11), 0.05, 2.4),
        (methylene(9.4, 5e-11), 0.2, 1.5),
    ] {
        let h = sys.hamiltonian().unwrap();
        let relaxation = sys.relaxation(sys.field()).unwrap();
        let rho = pulsed(&sys);
        let full = sys
            .relax_planned(&rho, &h, t, true, &relaxation, Some(Method::Full))
            .unwrap();
        let reduced = sys.relax_with(&rho, &h, t, true, &relaxation).unwrap();
        let err = relative(&reduced, &full);
        assert!(err < 1e-10, "{err:e}");
        let sizes = sys.planned_sizes(&rho).unwrap();
        let (all, kept): (usize, usize) = sizes.iter().fold((0, 0), |(a, k), s| (a + s.0, k + s.1));
        assert!(all as f64 > factor * kept as f64, "{all} → {kept}");
        let fid = sys.fid(&rho, Isotope::H1, 1e-4, 4).unwrap();
        let fp = sys.fplus(Isotope::H1).unwrap();
        for (k, s) in fid.iter().enumerate() {
            let at = sys
                .relax_planned(
                    &rho,
                    &h,
                    k as f64 * 1e-4,
                    true,
                    &relaxation,
                    Some(Method::Full),
                )
                .unwrap();
            let want = sys.expectation(&at, &fp).unwrap();
            assert!(
                (s - want).norm() < 1e-10 * want.norm().max(fid[0].norm()),
                "{k}"
            );
        }
    }
    // A pulse on one methyl proton breaks the symmetry of the state: it
    // propagates in full, and still correctly.
    let sys = methyl(1, 9.4, 3e-11);
    let rho = sys
        .evolve(
            &sys.thermal_state().unwrap(),
            &[Event::pulse(Target::Spin(1), 90.0, 0.0)],
        )
        .unwrap();
    let sizes = sys.planned_sizes(&rho).unwrap();
    assert!(sizes.iter().all(|s| s.0 == s.1));
    let h = sys.hamiltonian().unwrap();
    let relaxation = sys.relaxation(9.4).unwrap();
    let full = sys
        .relax_planned(&rho, &h, 0.1, true, &relaxation, Some(Method::Full))
        .unwrap();
    let got = sys.relax_with(&rho, &h, 0.1, true, &relaxation).unwrap();
    assert!(relative(&got, &full) < 1e-12);
}

#[test]
fn declared_symmetries_are_checked() {
    let mut sys = methyl(1, 9.4, 3e-11);
    let n = sys.spins().len();
    let good = sys.symmetries().to_vec();
    // Not a permutation; carbon onto a proton; protons of other shifts.
    for bad in [
        vec![0, 1, 1, 3, 4],
        vec![1, 0, 2, 3, 4],
        vec![0, 1, 2, 4, 3],
        vec![0; 3],
    ] {
        assert!(sys.set_symmetries(vec![bad]).is_err());
        assert_eq!(sys.symmetries(), good.as_slice());
    }
    // A proton moved off the axis breaks the threefold symmetry of the
    // dipolar couplings; a coupling to one methyl proton breaks it too.
    let mut moved = sys.clone();
    moved.set_position(4, [0.3, 0.0, -1.5]).unwrap();
    assert!(moved.validate().is_err());
    let mut coupled = sys.clone();
    coupled.couple(1, 4, 9.0).unwrap();
    assert!(coupled.thermal_state().is_err());
    // The symmetries survive JSON and a new spin, which stays put.
    let back = SpinSystem::from_json(&sys.to_json()).unwrap();
    assert_eq!(back.symmetries(), good.as_slice());
    sys.add_spin(Isotope::N15, 0.0).unwrap();
    assert_eq!(sys.symmetries()[0].len(), n + 1);
    assert_eq!(sys.symmetries()[0][n], n);
    // Without positions the symmetry of shifts, couplings and relaxation
    // times is enough.
    let mut plain = SpinSystem::new(9.4);
    for shift in [1.0, 2.0, 1.0] {
        plain.add_spin(Isotope::H1, shift).unwrap();
    }
    plain.couple(0, 1, 7.0).unwrap();
    plain.couple(1, 2, 7.0).unwrap();
    plain.set_symmetries(vec![vec![2, 1, 0]]).unwrap();
    plain.set_relaxation(0, Some(1.0), None).unwrap();
    assert!(plain.validate().is_err());
    plain.set_relaxation(2, Some(1.0), None).unwrap();
    plain.validate().unwrap();
    plain.couple(1, 2, 6.0).unwrap();
    assert!(plain.validate().is_err());
}

#[test]
fn thermal_equilibrium_stays_put() {
    // Seven protons with distinct shifts and no couplings among them (so
    // that H commutes with the thermal state), T₁, T₂ and Redfield
    // relaxation: five seconds keep equilibrium to 10⁻¹⁰ of its deviation
    // from the maximally mixed state, in one Krylov step: its deviation,
    // parts in 10⁵ of 1/d, is only known to about 10⁻¹¹, and moves no
    // more than that.
    let mut sys = SpinSystem::new(9.4);
    for i in 0..7 {
        sys.add_spin(Isotope::H1, 1.0 + 0.5 * i as f64).unwrap();
        sys.set_position(i, [1.8 * i as f64, 0.4 * (i % 3) as f64, 0.0])
            .unwrap();
        sys.set_relaxation(i, Some(2.0 + 0.3 * i as f64), Some(1.0))
            .unwrap();
    }
    let sys = sys.with_correlation_time(Some(2e-11)).unwrap();
    let thermal = sys.thermal_state().unwrap();
    let start = std::time::Instant::now();
    let later = sys.evolve(&thermal, &[Event::Delay(5.0)]).unwrap();
    let err = relative(&later, &thermal);
    assert!(err < 1e-10, "{err:e}");
    assert!(start.elapsed().as_secs_f64() < 10.0);
}

#[test]
fn large_problems_are_refused_from_the_estimates() {
    // Eleven protons with Redfield relaxation: the superoperator's terms
    // alone exceed the memory limit. Nine protons relaxing for 100 s
    // exceed the work limit. Both are refused at once.
    let start = std::time::Instant::now();
    let mut sys = SpinSystem::new(9.4);
    for i in 0..11 {
        sys.add_spin(Isotope::H1, 0.3 * i as f64).unwrap();
        sys.set_position(i, [1.7 * i as f64, 0.0, 0.0]).unwrap();
    }
    let sys = sys.with_correlation_time(Some(1e-11)).unwrap();
    let err = sys
        .relaxation_rates(&[sys.iz(0).unwrap()])
        .unwrap_err()
        .to_string();
    assert!(err.contains("GB"), "{err}");
    let nine = protons(9, 9.4);
    let err = nine
        .evolve(&pulsed(&nine), &[Event::Delay(100.0)])
        .unwrap_err()
        .to_string();
    assert!(err.contains("operations"), "{err}");
    assert!(start.elapsed().as_secs_f64() < 20.0);
}

/// Timings for the guide (`cargo test --release -- --ignored --nocapture
/// timings_of_larger_systems`).
#[test]
#[ignore]
fn timings_of_larger_systems() {
    let time = |label: &str, f: &mut dyn FnMut()| {
        let start = std::time::Instant::now();
        f();
        println!("{label}: {:.2} s", start.elapsed().as_secs_f64());
    };
    for (n, b, t) in [
        (5, 9.4, 1.0),
        (6, 9.4, 1.0),
        (7, 9.4, 1.0),
        (8, 1.0, 1.0),
        (8, 9.4, 0.1),
    ] {
        let sys = protons(n, b);
        let rho = pulsed(&sys);
        time(&format!("{n} protons, T1/T2, {b} T, {t} s"), &mut || {
            sys.evolve(&rho, &[Event::Delay(t)]).unwrap();
        });
    }
    let mut chain = protons(7, 9.4);
    for i in 0..7 {
        chain
            .set_position(i, [1.8 * i as f64, 0.5 * (i % 2) as f64, 0.0])
            .unwrap();
    }
    let chain = chain.with_correlation_time(Some(5e-11)).unwrap();
    time("7 protons, Redfield terms", &mut || {
        chain.relaxation(9.4).unwrap();
    });
    let rho = pulsed(&chain);
    for t in [0.01, 0.1] {
        time(&format!("7 protons, Redfield, 9.4 T, {t} s"), &mut || {
            chain.evolve(&rho, &[Event::Delay(t)]).unwrap();
        });
    }
    let methyl7 = methyl(3, 9.4, 3e-11);
    let h = methyl7.hamiltonian().unwrap();
    let relaxation = methyl7.relaxation(9.4).unwrap();
    let rho = pulsed(&methyl7);
    for (label, force) in [("full", Some(Method::Full)), ("on orbits", None)] {
        time(
            &format!("methyl + 3 protons, Redfield, 0.1 s, {label}"),
            &mut || {
                methyl7
                    .relax_planned(&rho, &h, 0.1, true, &relaxation, force)
                    .unwrap();
            },
        );
    }
    println!("sizes {:?}", methyl7.planned_sizes(&rho).unwrap());
}

/// The accuracies quoted in the guide and the README (`cargo test
/// --release -- --ignored --nocapture accuracies_of_larger_systems`).
#[test]
#[ignore]
fn accuracies_of_larger_systems() {
    let mut chain = protons(5, 9.4).with_frame(Frame::Laboratory);
    for i in 0..5 {
        chain
            .set_position(i, [1.8 * i as f64, 0.5 * (i % 2) as f64, 0.0])
            .unwrap();
    }
    let chain = chain.with_correlation_time(Some(5e-11)).unwrap();
    let mut worst = 0.0f64;
    for (sys, t) in [
        (protons(6, 2.0), 0.3),
        (protons(6, 9.4), 0.05),
        (methyl(1, 9.4, 3e-11), 0.2),
        (chain, 0.02),
    ] {
        let h = sys.hamiltonian().unwrap();
        let relaxation = sys.relaxation(sys.field()).unwrap();
        let rho = pulsed(&sys);
        let run = |m| {
            sys.relax_planned(&rho, &h, t, true, &relaxation, Some(m))
                .unwrap()
        };
        let err = relative(&run(Method::Krylov), &run(Method::Dense));
        println!(
            "{} spins, {t} s: Krylov vs dense {err:.1e}",
            sys.spins().len()
        );
        worst = worst.max(err);
    }
    println!("worst Krylov vs dense: {worst:.1e}");
    let sys = protons(7, 1.0);
    let h = sys.hamiltonian().unwrap();
    let relaxation = sys.relaxation(1.0).unwrap();
    let rho = pulsed(&sys);
    let run = |m| {
        sys.relax_planned(&rho, &h, 2e-3, true, &relaxation, Some(m))
            .unwrap()
    };
    println!(
        "7 protons, 2 ms: Krylov vs Dormand–Prince {:.1e}",
        relative(&run(Method::Krylov), &run(Method::Steps))
    );
    for (sys, t) in [(methyl(3, 9.4, 3e-11), 0.05), (methylene(9.4, 5e-11), 0.2)] {
        let h = sys.hamiltonian().unwrap();
        let relaxation = sys.relaxation(sys.field()).unwrap();
        let rho = pulsed(&sys);
        let full = sys
            .relax_planned(&rho, &h, t, true, &relaxation, Some(Method::Full))
            .unwrap();
        let reduced = sys.relax_with(&rho, &h, t, true, &relaxation).unwrap();
        let sizes = sys.planned_sizes(&rho).unwrap();
        let (all, kept): (usize, usize) = sizes.iter().fold((0, 0), |(a, k), s| (a + s.0, k + s.1));
        println!(
            "{} spins: orbits vs full {:.1e}, {all} → {kept} coefficients",
            sys.spins().len(),
            relative(&reduced, &full)
        );
    }
    let mut sys = SpinSystem::new(9.4);
    for i in 0..7 {
        sys.add_spin(Isotope::H1, 1.0 + 0.5 * i as f64).unwrap();
        sys.set_position(i, [1.8 * i as f64, 0.4 * (i % 3) as f64, 0.0])
            .unwrap();
        sys.set_relaxation(i, Some(2.0 + 0.3 * i as f64), Some(1.0))
            .unwrap();
    }
    let sys = sys.with_correlation_time(Some(2e-11)).unwrap();
    let thermal = sys.thermal_state().unwrap();
    let later = sys.evolve(&thermal, &[Event::Delay(5.0)]).unwrap();
    println!("equilibrium after 5 s: {:.1e}", relative(&later, &thermal));
}
