//! Field profiles with relaxation on sparse blocks (1.51).

use super::*;
use crate::optimize::Options;
use crate::spin::Isotope;

/// ¹H–¹H–¹³C in the laboratory frame with J couplings, `T₁` of the carbon
/// and Redfield relaxation of the dipolar pairs and the carbon's shielding
/// (τ_c = 3·10⁻⁵ s, so that the spectral densities disperse between 1
/// and 100 µT), at 1 K so that the thermal drift counts.
fn relaxing() -> SpinSystem {
    let mut sys = SpinSystem::new(5e-5).with_frame(Frame::Laboratory);
    sys.add_spin(Isotope::H1, 2.0).unwrap();
    sys.add_spin(Isotope::H1, 4.0).unwrap();
    sys.add_spin(Isotope::C13, 170.0).unwrap();
    sys.couple(0, 1, 7.2).unwrap();
    sys.couple(0, 2, 5.8).unwrap();
    sys.couple(1, 2, -1.4).unwrap();
    sys.set_position(0, [5.0, 0.0, 0.0]).unwrap();
    sys.set_position(1, [-1.8, 4.6, 0.0]).unwrap();
    sys.set_position(2, [0.0, 0.0, 0.0]).unwrap();
    sys.set_csa(
        2,
        Some(crate::spin::axial_csa(60.0, [0.0, 0.0, 1.0]).unwrap()),
    )
    .unwrap();
    sys.set_relaxation(2, Some(2.0), None).unwrap();
    sys.with_correlation_time(Some(3e-5))
        .unwrap()
        .with_temperature(1.0)
        .unwrap()
}

#[test]
fn the_relaxation_over_the_field_equals_direct_builds_within_its_error() {
    let sys = relaxing();
    let start = sys.parahydrogen_state(0, 1, 1.0).unwrap();
    let c = Control::field(1e-6, 1e-4, 0.02, 4, start, sys.iz(2).unwrap());
    let fields = [1e-6, 5.5e-6, 1.9e-5, 4e-5, 6.17e-5, 9.5e-5, 1e-4];
    let (error, nodes, diffs) = sys.field_interpolation(&c, &fields).unwrap();
    assert!(error <= FIELD_TOLERANCE, "{error:e} with {nodes} nodes");
    for (b, d) in fields.iter().zip(&diffs) {
        assert!(
            *d <= 10.0 * error + 1e-14,
            "{b} T: {d:e} vs stated {error:e}"
        );
    }
    assert_eq!(sys.field_relaxation_error(&c).unwrap(), (error, nodes));
    // Slow motion over a wide range does not converge within the nodes: the
    // stated error is above the limit, and the problem is refused.
    let slow = sys.clone().with_correlation_time(Some(1e-7)).unwrap();
    let wide = Control::field(0.0, 9.4, 0.1, 4, c.initial.clone(), c.target.clone());
    let err = slow.field_relaxation_error(&wide).unwrap_err();
    assert!(err.to_string().contains("narrow the range"), "{err}");
    // Without relaxation there is nothing to interpolate.
    let mut quiet = SpinSystem::new(1.0).with_frame(Frame::Laboratory);
    quiet.add_spin(Isotope::H1, 1.0).unwrap();
    let q = Control::field(0.0, 1.0, 0.1, 2, quiet.iz(0).unwrap(), quiet.iz(0).unwrap());
    assert!(quiet.field_relaxation_error(&q).is_err());
}

#[test]
fn sparse_field_gradients_equal_the_dense_ones() {
    // The dense generator of 1.36 with the relaxation's derivative by
    // extrapolated differences against the interpolated one on sparse
    // blocks, for a random profile of eight steps.
    let sys = relaxing();
    let start = sys.parahydrogen_state(0, 1, 1.0).unwrap();
    let c = Control::field(1e-6, 1e-4, 0.02, 8, start, sys.iz(2).unwrap());
    let u = c.random_start(7, 0.0);
    let (v_sparse, g_sparse, _) = sys.control_members_on(&c, &u, Path::SparseField).unwrap();
    let (v_dense, g_dense, _) = sys.control_members_on(&c, &u, Path::Dense).unwrap();
    assert!(
        (v_sparse - v_dense).abs() < 1e-10 * v_dense.abs().max(1e-3),
        "{v_sparse} vs {v_dense}"
    );
    let scale = g_dense.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    for (k, (a, b)) in g_sparse.iter().zip(&g_dense).enumerate() {
        assert!((a - b).abs() < 1e-7 * scale, "{k}: {a} vs {b}");
    }
    assert_eq!(sys.control_cost(&c, 2.0).unwrap().method, "sparse");
}

#[test]
fn seven_relaxing_spins_follow_central_differences() {
    // Six protons and a ¹³C with T₁ and T₂ in the laboratory frame between
    // 5 and 20 µT (where the Zeeman difference of ¹H and ¹³C is comparable
    // to the J couplings), from the singlet of the first two to the
    // carbon: three steps of 2 ms, against Richardson-extrapolated central
    // differences of 10⁻⁹ T along a direction.
    let mut sys = SpinSystem::new(1e-5).with_frame(Frame::Laboratory);
    for k in 0..7 {
        let isotope = if k == 5 { Isotope::C13 } else { Isotope::H1 };
        sys.add_spin(isotope, 1.0 + 0.7 * k as f64).unwrap();
        sys.set_relaxation(k, Some(3.0 + k as f64), Some(1.5))
            .unwrap();
    }
    for k in 0..6 {
        sys.couple(k, k + 1, 7.0 - 0.5 * k as f64).unwrap();
    }
    sys.couple(1, 5, 140.0).unwrap();
    let start = sys.parahydrogen_state(0, 1, 1.0).unwrap();
    let c = Control::field(5e-6, 2e-5, 0.006, 3, start, sys.iz(5).unwrap());
    let u = [7.1e-6, 1.83e-5, 1.12e-5];
    let v = [0.3, -0.8, 0.5];
    let (_, grad) = sys.control_transfer(&c, &u).unwrap();
    let along: f64 = grad.iter().zip(&v).map(|(g, v)| g * v).sum();
    let h = 1e-9;
    let at = |x: f64| {
        let p: Vec<f64> = u.iter().zip(&v).map(|(u, v)| u + x * v).collect();
        sys.control_transfer(&c, &p).unwrap().0
    };
    let fd = (8.0 * (at(h) - at(-h)) - (at(2.0 * h) - at(-2.0 * h))) / (12.0 * h);
    assert!((along - fd).abs() < 1e-6 * fd.abs(), "{along} vs {fd}");
    assert!(fd.abs() > 1e2, "{fd}");
}

#[test]
fn an_optimised_profile_replays_to_its_transfer() {
    let sys = relaxing();
    let start = sys.parahydrogen_state(0, 1, 1.0).unwrap();
    let mut c = Control::field(1e-6, 1e-4, 0.02, 6, start.clone(), sys.iz(2).unwrap());
    c.smoothness = 1e-3;
    let options = Options {
        max_iterations: 15,
        ..Options::default()
    };
    let out = sys
        .optimize_control(&c, &c.random_start(3, 0.0), &options)
        .unwrap();
    let events = sys.control_events(&c, &out.controls).unwrap();
    let rho = sys.evolve(&start, &events).unwrap();
    let replayed = (sys.iz(2).unwrap().adjoint() * rho).trace().re;
    assert!(
        (replayed - out.transfer).abs() < 1e-9 * out.transfer.abs().max(1e-3),
        "{replayed} vs {}",
        out.transfer
    );
}

#[test]
fn sabre_sheath_transfer_peaks_at_the_anti_crossing() {
    // Two hydrides and a ¹⁵N (J_HH = −7, J_H₁N = −2 Hz) in the laboratory
    // frame, the ¹⁵N relaxing (T₁ = 30 s): from the singlet, ⟨N_z⟩ after
    // 0.5 s at a constant field peaks where ν_N − ν_H = ±(J_HH − J_HN/4)
    // (the anti-crossing |S α_N⟩ ↔ |T₊ β_N⟩ of the three-spin model).
    let nu = |x: Isotope, shift: f64| -x.larmor(1.0) * (1.0 + shift * 1e-6);
    let slope = nu(Isotope::N15, 250.0) - nu(Isotope::H1, -22.0);
    let lac = ((-7.0 + 2.0 / 4.0) / slope).abs();
    let mut sys = SpinSystem::new(lac).with_frame(Frame::Laboratory);
    sys.add_spin(Isotope::H1, -22.0).unwrap();
    sys.add_spin(Isotope::H1, -22.0).unwrap();
    sys.add_spin(Isotope::N15, 250.0).unwrap();
    sys.couple(0, 1, -7.0).unwrap();
    sys.couple(0, 2, -2.0).unwrap();
    sys.set_relaxation(2, Some(30.0), None).unwrap();
    let start = sys.parahydrogen_state(0, 1, 1.0).unwrap();
    let c = Control::field(0.2 * lac, 2.0 * lac, 0.5, 1, start, sys.iz(2).unwrap());
    let fields: Vec<f64> = (0..=60).map(|k| lac * (0.4 + k as f64 / 50.0)).collect();
    let transfer: Vec<f64> = fields
        .iter()
        .map(|&b| sys.control_transfer(&c, &[b]).unwrap().0.abs())
        .collect();
    let (best, peak) = transfer
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    assert!(
        (fields[best] / lac - 1.0).abs() < 0.03,
        "{} vs {lac}",
        fields[best]
    );
    assert!(
        *peak > 2.0 * transfer[0] && *peak > 2.0 * transfer[60],
        "{peak}"
    );
}

/// SABRE-SHEATH on the free substrate's complex: two hydrides, the ¹⁵N of
/// a bound pyridine and `protons` of its ring protons, in the laboratory
/// frame at microtesla fields, every spin with `T₁` and `T₂`, and with
/// `redfield` Redfield relaxation of the dipolar pairs (τ_c = 10⁻¹⁰ s).
fn sheath(protons: usize, redfield: bool) -> (SpinSystem, f64) {
    let nu = |x: Isotope, shift: f64| -x.larmor(1.0) * (1.0 + shift * 1e-6);
    let slope = nu(Isotope::N15, 250.0) - nu(Isotope::H1, -22.0);
    let lac = ((-7.0 + 2.0 / 4.0) / slope).abs();
    let mut sys = SpinSystem::new(lac).with_frame(Frame::Laboratory);
    sys.add_spin(Isotope::H1, -22.0).unwrap();
    sys.add_spin(Isotope::H1, -22.0).unwrap();
    sys.add_spin(Isotope::N15, 250.0).unwrap();
    sys.couple(0, 1, -7.0).unwrap();
    sys.couple(0, 2, -2.0).unwrap();
    sys.couple(1, 2, 0.5).unwrap();
    sys.set_position(0, [1.6, 0.0, 0.0]).unwrap();
    sys.set_position(1, [0.0, 1.6, 0.0]).unwrap();
    sys.set_position(2, [0.0, 0.0, 2.1]).unwrap();
    for k in 0..protons {
        let s = sys.add_spin(Isotope::H1, 7.0 + 0.5 * k as f64).unwrap();
        let angle = std::f64::consts::PI * k as f64 / 3.0;
        sys.set_position(s, [2.2 * angle.cos(), 2.2 * angle.sin(), 4.0])
            .unwrap();
        sys.couple(2, s, -10.0 + 2.0 * k as f64).unwrap();
        if k > 0 {
            sys.couple(s - 1, s, 7.5).unwrap();
        }
    }
    for k in 0..sys.spins().len() {
        sys.set_relaxation(k, Some(if k == 2 { 30.0 } else { 3.0 }), Some(1.0))
            .unwrap();
    }
    let sys = sys
        .with_correlation_time(redfield.then_some(1e-10))
        .unwrap()
        .with_temperature(300.0)
        .unwrap();
    (sys, lac)
}

#[test]
#[ignore]
fn measure_field_profiles() {
    // Eight spins with Redfield relaxation need about 5.6 GB (five node
    // matrices of the dipolar relaxation), over the limit: measured with
    // T₁ and T₂ alone.
    for (protons, duration, redfield) in [(4, 0.5, true), (5, 0.2, false)] {
        let (sys, lac) = sheath(protons, redfield);
        let n = sys.spins().len();
        let start = sys.parahydrogen_state(0, 1, 1.0).unwrap();
        let target = sys.iz(2).unwrap();
        let steps = 10;
        let c = Control::field(
            0.0,
            2.0 * lac,
            duration,
            steps,
            start.clone(),
            target.clone(),
        );
        let cost = sys.control_cost(&c, 2.0 * lac).unwrap();
        let clock = std::time::Instant::now();
        let (error, nodes) = sys.field_relaxation_error(&c).unwrap();
        let build = clock.elapsed().as_secs_f64();
        let constant = vec![lac; steps];
        let clock = std::time::Instant::now();
        let (at_lac, _) = sys.control_transfer(&c, &constant).unwrap();
        let evaluation = clock.elapsed().as_secs_f64();
        eprintln!(
            "{n} spins (Redfield {redfield}, {duration} s): lac {:.3} µT, estimate {:.2e} operations {:.2} GB, interpolant {nodes} \
             nodes, error {error:.1e} ({build:.1} s), one evaluation {evaluation:.1} s, \
             ⟨N_z⟩ at the anti-crossing {at_lac:.4}",
            lac * 1e6,
            cost.operations,
            cost.bytes / 1e9
        );
        if protons > 4 {
            continue;
        }
        let options = Options {
            max_iterations: 5,
            ..Options::default()
        };
        let clock = std::time::Instant::now();
        let out = sys.optimize_control(&c, &constant, &options).unwrap();
        let events = sys.control_events(&c, &out.controls).unwrap();
        let rho = sys.evolve(&start, &events).unwrap();
        let replayed = (target.adjoint() * rho).trace().re;
        eprintln!(
            "  optimised: {:.4} after {} iterations ({} evaluations, {:.0} s), replayed {:.2e}",
            out.transfer,
            out.iterations,
            out.evaluations,
            clock.elapsed().as_secs_f64(),
            (replayed - out.transfer).abs()
        );
    }
}
