//! Field profiles with Redfield relaxation applied without storing it
//! (1.55), against the dense generator and the interpolant of 1.51.

use super::field_tests::{relaxing, sheath};
use super::*;

#[test]
fn matrix_free_field_profiles_equal_the_stored_ones() {
    // The relaxation built at every step's field from its terms, with its
    // analytic derivative, against the dense generator of 1.36 (the
    // derivative by extrapolated differences) and the interpolant of 1.51,
    // for a random profile of eight steps between 1 and 100 µT where the
    // spectral densities disperse, at 1 K (the thermal drift counts).
    let sys = relaxing();
    let start = sys.parahydrogen_state(0, 1, 1.0).unwrap();
    let c = Control::field(1e-6, 1e-4, 0.02, 8, start, sys.iz(2).unwrap());
    let u = c.random_start(7, 0.0);
    let (v_free, g_free, _) = sys.control_members_on(&c, &u, Path::FreeField).unwrap();
    let (v_dense, g_dense, _) = sys.control_members_on(&c, &u, Path::Dense).unwrap();
    let (v_sparse, g_sparse, _) = sys.control_members_on(&c, &u, Path::SparseField).unwrap();
    for v in [v_dense, v_sparse] {
        assert!(
            (v_free - v).abs() < 1e-10 * v.abs().max(1e-3),
            "{v_free} vs {v}"
        );
    }
    let scale = g_dense.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    for (k, ((a, b), s)) in g_free.iter().zip(&g_dense).zip(&g_sparse).enumerate() {
        assert!((a - b).abs() < 1e-7 * scale, "{k}: {a} vs dense {b}");
        assert!((a - s).abs() < 1e-7 * scale, "{k}: {a} vs sparse {s}");
    }
}

#[test]
fn beyond_the_interpolant_profiles_apply_the_relaxation_from_its_terms() {
    // Six spins of SABRE-SHEATH with dipolar Redfield relaxation and a
    // field profile whose interpolant needs more memory than the limit
    // allows (forced here by its path): the gradient along a direction
    // against Richardson-extrapolated central differences, and replayed by
    // evolve.
    let (sys, lac) = sheath(3, true);
    let start = sys.parahydrogen_state(0, 1, 1.0).unwrap();
    let c = Control::field(0.0, 2.0 * lac, 0.1, 4, start.clone(), sys.iz(2).unwrap());
    let u = [0.6 * lac, 1.1 * lac, 0.9 * lac, 1.4 * lac];
    let (value, grad, _) = sys.control_members_on(&c, &u, Path::FreeField).unwrap();
    let events = sys.control_events(&c, &u).unwrap();
    let end = sys.evolve(&start, &events).unwrap();
    let want = sys.expectation(&end, &sys.iz(2).unwrap()).unwrap().re;
    assert!(
        (value - want).abs() < 1e-9 * want.abs(),
        "{value} vs {want}"
    );
    let dir = [1.0, -0.4, 0.7, 0.2];
    let along = |h: f64| {
        let up: Vec<f64> = u.iter().zip(&dir).map(|(b, d)| b + h * d).collect();
        let down: Vec<f64> = u.iter().zip(&dir).map(|(b, d)| b - h * d).collect();
        let f = |x: &[f64]| sys.control_members_on(&c, x, Path::FreeField).unwrap().0;
        (f(&up) - f(&down)) / (2.0 * h)
    };
    let h = 1e-3 * lac;
    let fd = (4.0 * along(h / 2.0) - along(h)) / 3.0;
    let exact: f64 = grad.iter().zip(&dir).map(|(g, d)| g * d).sum();
    assert!((fd - exact).abs() < 1e-5 * exact.abs(), "{fd} vs {exact}");
}

#[test]
fn profiles_past_the_interpolants_memory_take_the_matrix_free_path() {
    // Eight spins of SABRE-SHEATH with Redfield relaxation: the nodes of
    // the interpolant would need 9.4 GB; the problem is costed on the
    // matrix-free path instead of refused. A field profile with only the
    // shielding relaxing passes zero field, where it vanishes.
    let (sys, lac) = sheath(5, true);
    let start = sys.parahydrogen_state(0, 1, 1.0).unwrap();
    let c = Control::field(0.0, 2.0 * lac, 0.04, 2, start, sys.iz(2).unwrap());
    let cost = sys.control_cost(&c, 2.0 * lac).unwrap();
    assert_eq!(cost.method, "matrix-free");
    assert!(cost.bytes < 1e9, "{}", cost.bytes);
    let mut csa = SpinSystem::new(1e-3).with_frame(Frame::Laboratory);
    csa.add_spin(crate::spin::Isotope::H1, 1.0).unwrap();
    csa.add_spin(crate::spin::Isotope::C13, 60.0).unwrap();
    csa.couple(0, 1, 140.0).unwrap();
    csa.set_csa(
        1,
        Some(crate::spin::axial_csa(80.0, [0.0, 0.0, 1.0]).unwrap()),
    )
    .unwrap();
    csa.set_relaxation(1, Some(2.0), None).unwrap();
    let csa = csa.with_correlation_time(Some(3e-9)).unwrap();
    let start = csa.thermal_state().unwrap();
    let c = Control::field(0.0, 1e-3, 0.002, 3, start, csa.iz(1).unwrap());
    let u = [5e-4, 0.0, 1e-3];
    let (free, _, _) = csa.control_members_on(&c, &u, Path::FreeField).unwrap();
    let (dense, _, _) = csa.control_members_on(&c, &u, Path::Dense).unwrap();
    assert!(
        (free - dense).abs() < 1e-10 * dense.abs(),
        "{free} vs {dense}"
    );
}

/// SABRE-SHEATH with eight spins and dipolar Redfield relaxation (the
/// interpolant of 1.51 would need 9.4 GB): the cost of ten steps over
/// 0.2 s, and two steps over 0.02 s evaluated and replayed by `evolve`,
/// for the documentation (`cargo test --release -- --ignored
/// measure_free_field --nocapture`).
#[test]
#[ignore]
fn measure_free_field() {
    let (sys, lac) = sheath(5, true);
    let start = sys.parahydrogen_state(0, 1, 1.0).unwrap();
    for (steps, duration) in [(10usize, 0.2), (2, 0.02)] {
        let c = Control::field(
            0.0,
            2.0 * lac,
            duration,
            steps,
            start.clone(),
            sys.iz(2).unwrap(),
        );
        let cost = sys.control_cost(&c, 2.0 * lac).unwrap();
        let u = vec![lac; steps];
        let clock = std::time::Instant::now();
        let result = sys.control_transfer(&c, &u);
        let t = clock.elapsed().as_secs_f64();
        let line = match result {
            Ok((value, grad)) => {
                let events = sys.control_events(&c, &u).unwrap();
                let end = sys.evolve(&start, &events).unwrap();
                let want = sys.expectation(&end, &sys.iz(2).unwrap()).unwrap().re;
                format!(
                    "{t:.1} s, ⟨N_z⟩ = {value:.6e} (replayed {want:.6e}), first gradient component {:.3e}",
                    grad[0]
                )
            }
            Err(e) => format!("refused: {e}"),
        };
        eprintln!(
            "{} spins, {steps} steps over {duration} s: {} path, {:.1e} operations and {:.2} GB \
             estimated; {line}",
            sys.spins().len(),
            cost.method,
            cost.operations,
            cost.bytes / 1e9
        );
    }
}
