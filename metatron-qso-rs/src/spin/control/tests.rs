use super::*;
use crate::spin::Isotope;

/// A ¹H–¹³C pair on resonance (rotating frame, 9.4 T), `J` Hz.
fn pair(j: f64) -> SpinSystem {
    let mut sys = SpinSystem::new(9.4);
    sys.add_spin(Isotope::H1, 0.0).unwrap();
    sys.add_spin(Isotope::C13, 0.0).unwrap();
    sys.couple(0, 1, j).unwrap();
    sys
}

fn channels() -> Vec<Target> {
    vec![Target::Isotope(Isotope::H1), Target::Isotope(Isotope::C13)]
}

/// Central differences of the transfer, step `h`.
fn differences(sys: &SpinSystem, c: &Control, u: &[f64], h: f64) -> Vec<f64> {
    (0..u.len())
        .map(|i| {
            let mut p = u.to_vec();
            p[i] += h;
            let up = sys.control_transfer(c, &p).unwrap().0;
            p[i] -= 2.0 * h;
            let down = sys.control_transfer(c, &p).unwrap().0;
            (up - down) / (2.0 * h)
        })
        .collect()
}

fn assert_gradient(sys: &SpinSystem, c: &Control, u: &[f64], h: f64, tol: f64) {
    let (_, grad) = sys.control_transfer(c, u).unwrap();
    let fd = differences(sys, c, u, h);
    let scale = fd.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    assert!(scale > 0.0);
    for (i, (a, b)) in grad.iter().zip(&fd).enumerate() {
        assert!(
            (a - b).abs() <= tol * scale,
            "{i}: {a} vs {b} (scale {scale})"
        );
    }
}

#[test]
fn gradients_agree_with_finite_differences() {
    // rf on both channels with offsets and rf errors, without and with
    // relaxation.
    let mut sys = pair(140.0);
    sys.spins[0].shift = 0.3;
    let iz = sys.iz(0).unwrap();
    let sz = sys.iz(1).unwrap();
    let mut c = Control::rf(channels(), None, 4e-3, 8, iz.clone(), sz.clone());
    c.ensemble = vec![
        Member::ideal(),
        Member {
            weight: 0.5,
            scales: vec![0.9, 1.1],
            offsets: vec![40.0, -25.0],
        },
    ];
    let u = c.random_start(3, 400.0);
    assert_gradient(&sys, &c, &u, 1e-3, 1e-7);
    sys.set_relaxation(0, Some(0.02), Some(0.01)).unwrap();
    sys.set_relaxation(1, Some(0.05), None).unwrap();
    assert_gradient(&sys, &c, &u, 1e-2, 1e-8);
    // A thermal state (at 50 mK, so that differences resolve the drift
    // towards equilibrium against the identity).
    let cold = sys.clone().with_temperature(0.05).unwrap();
    let mut thermal = c.clone();
    thermal.initial = cold.thermal_state().unwrap();
    assert_gradient(&cold, &thermal, &u, 1e-2, 1e-8);
    // Redfield relaxation (two protons and a carbon tumbling) and a spin 1.
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
    thermal.ensemble = c.ensemble.clone();
    assert_gradient(&red, &thermal, &thermal.random_start(2, 400.0), 1e-2, 1e-8);
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
    assert_gradient(&deut, &spin1, &spin1.random_start(3, 300.0), 1e-2, 1e-8);
    // Field profiles in the laboratory frame, without and with relaxation.
    let mut lab = SpinSystem::new(0.0).with_frame(Frame::Laboratory);
    lab.add_spin(Isotope::H1, 2.0).unwrap();
    lab.add_spin(Isotope::H1, 4.0).unwrap();
    lab.add_spin(Isotope::C13, 170.0).unwrap();
    lab.couple(0, 1, 7.2).unwrap();
    lab.couple(0, 2, 5.8).unwrap();
    lab.couple(1, 2, -1.4).unwrap();
    let start = lab.parahydrogen_state(0, 1, 1.0).unwrap();
    let f = Control::field(0.0, 2e-6, 0.2, 6, start, lab.iz(2).unwrap());
    let b = f.random_start(5, 0.0);
    assert_gradient(&lab, &f, &b, 1e-10, 1e-6);
    lab.set_relaxation(2, Some(0.5), None).unwrap();
    assert_gradient(&lab, &f, &b, 1e-10, 1e-6);
}

#[test]
fn the_transfer_is_what_evolve_gives() {
    let mut sys = pair(140.0);
    let c = Control::rf(
        channels(),
        Some(2000.0),
        3e-3,
        12,
        sys.thermal_state().unwrap(),
        sys.iz(1).unwrap(),
    );
    let u = c.random_start(1, 1500.0);
    for relax in [false, true] {
        if relax {
            sys.set_relaxation(0, Some(0.05), Some(0.02)).unwrap();
        }
        let (value, _) = sys.control_transfer(&c, &u).unwrap();
        let rho = sys
            .evolve(&c.initial, &sys.control_events(&c, &u).unwrap())
            .unwrap();
        let want = sys.expectation(&rho, &c.target).unwrap().re;
        assert!(
            (value - want).abs() < 1e-10 * want.abs().max(1e-6),
            "{value} vs {want}"
        );
    }
}

/// The best of `seeds` optimisations from random starts of `scale` Hz.
fn best(sys: &SpinSystem, c: &Control, seeds: u64, scale: f64, iterations: usize) -> Optimized {
    let options = Options {
        max_iterations: iterations,
        g_tol: 1e-10,
        ..Options::default()
    };
    (0..seeds)
        .map(|s| {
            sys.optimize_control(c, &c.random_start(s, scale), &options)
                .unwrap()
        })
        .max_by(|a, b| a.transfer.total_cmp(&b.transfer))
        .unwrap()
}

#[test]
fn optimised_transfer_reaches_the_unitary_bound() {
    // Thermal ¹H polarisation onto ¹³C: the bound (Glaser et al. 1998)
    // is the proton's polarisation, γ_H/γ_C times carbon's own.
    let sys = pair(140.0);
    let rho = sys.thermal_state().unwrap();
    let sz = sys.iz(1).unwrap();
    let bound = unitary_bound(&rho, &sz).unwrap();
    let own = sys.expectation(&rho, &sz).unwrap().re;
    assert!((bound / own - 3.976).abs() < 0.01, "{}", bound / own);
    let c = Control::rf(
        channels(),
        Some(5000.0),
        1.2 / 140.0,
        40,
        rho.clone(),
        sz.clone(),
    );
    let got = best(&sys, &c, 3, 500.0, 400);
    assert!(got.transfer <= bound * (1.0 + 1e-9));
    assert!(
        got.transfer >= (1.0 - 1e-6) * bound,
        "{} vs {bound}",
        got.transfer
    );
    // Two protons and a carbon (I₂S): 4.48 times carbon's own polarisation.
    let mut sys = SpinSystem::new(9.4);
    sys.add_spin(Isotope::H1, 0.0).unwrap();
    sys.add_spin(Isotope::H1, 0.0).unwrap();
    sys.add_spin(Isotope::C13, 0.0).unwrap();
    sys.couple(0, 2, 140.0).unwrap();
    sys.couple(1, 2, 140.0).unwrap();
    let rho = sys.thermal_state().unwrap();
    let sz = sys.iz(2).unwrap();
    let bound = unitary_bound(&rho, &sz).unwrap();
    let own = sys.expectation(&rho, &sz).unwrap().re;
    assert!((bound / own - 4.476).abs() < 0.01, "{}", bound / own);
    let c = Control::rf(
        channels(),
        Some(10_000.0),
        2.0 / 140.0,
        80,
        rho.clone(),
        sz.clone(),
    );
    let got = best(&sys, &c, 3, 500.0, 1000);
    assert!(got.transfer <= bound * (1.0 + 1e-9));
    assert!(
        got.transfer >= (1.0 - 1e-6) * bound,
        "{} vs {bound}",
        got.transfer
    );
    // With relaxation the optimum stays below the bound, and evolve
    // reproduces it.
    let mut relaxing = pair(140.0);
    relaxing.set_relaxation(0, Some(0.05), Some(0.02)).unwrap();
    relaxing.set_relaxation(1, Some(0.5), Some(0.02)).unwrap();
    let rho = relaxing.thermal_state().unwrap();
    let sz = relaxing.iz(1).unwrap();
    let bound = unitary_bound(&rho, &sz).unwrap();
    let c = Control::rf(
        channels(),
        Some(5000.0),
        1.2 / 140.0,
        40,
        rho.clone(),
        sz.clone(),
    );
    let got = best(&relaxing, &c, 2, 500.0, 300);
    eprintln!("relaxing: {} of the bound", got.transfer / bound);
    assert!(
        got.transfer < 0.95 * bound && got.transfer > 0.5 * bound,
        "{}",
        got.transfer / bound
    );
    let after = relaxing
        .evolve(&rho, &relaxing.control_events(&c, &got.controls).unwrap())
        .unwrap();
    let want = relaxing.expectation(&after, &sz).unwrap().re;
    assert!((got.transfer - want).abs() < 1e-10 * want.abs());
}

#[test]
fn robust_pulses_keep_their_transfer_where_inept_does_not() {
    // ¹H → ¹³C polarisation with the rf amplitudes of both channels off
    // by up to 20 %: refocused INEPT transfers it fully only at the
    // nominal amplitude, a pulse optimised for the range everywhere.
    let sys = pair(140.0);
    let iz = sys.iz(0).unwrap();
    let sz = sys.iz(1).unwrap();
    let scales = [0.8, 0.9, 1.0, 1.1, 1.2];
    let inept = |s: f64| -> f64 {
        let (h, cc) = (Target::Isotope(Isotope::H1), Target::Isotope(Isotope::C13));
        let tau = 1.0 / (4.0 * 140.0);
        let p = |t, a: f64, ph| Event::pulse(t, a * s, ph);
        let events = vec![
            p(h, 90.0, 0.0),
            Event::Delay(tau),
            p(h, 180.0, 0.0),
            p(cc, 180.0, 0.0),
            Event::Delay(tau),
            p(h, 90.0, 90.0),
            p(cc, 90.0, 0.0),
            Event::Delay(tau),
            p(h, 180.0, 0.0),
            p(cc, 180.0, 0.0),
            Event::Delay(tau),
            p(cc, 90.0, 270.0),
        ];
        let rho = sys.evolve(&iz, &events).unwrap();
        sys.expectation(&rho, &sz).unwrap().re
    };
    let classic = scales.map(inept);
    assert!((classic[2] - 1.0).abs() < 1e-12);
    assert!(classic[0] < 0.1 && classic[4] < 0.3, "{classic:?}");
    let mut c = Control::rf(channels(), Some(10_000.0), 1.5 / 140.0, 60, iz, sz);
    c.ensemble = scales
        .iter()
        .map(|&s| Member {
            weight: 1.0,
            scales: vec![s, s],
            offsets: vec![],
        })
        .collect();
    let got = best(&sys, &c, 3, 500.0, 1000);
    assert!(got.members.iter().all(|&x| x > 0.9999), "{:?}", got.members);
    // Between and slightly beyond the training amplitudes as well.
    for k in 0..=50 {
        let s = 0.75 + 0.01 * k as f64;
        c.ensemble = vec![Member {
            weight: 1.0,
            scales: vec![s, s],
            offsets: vec![],
        }];
        let (value, _) = sys.control_transfer(&c, &got.controls).unwrap();
        assert!(value > 0.9998, "{s}: {value}");
    }
}

#[test]
fn a_field_profile_reaches_the_best_possible_transfer_in_two_seconds() {
    // Parahydrogen order to ¹³C (AA'X, as in the field-cycling test of
    // 1.35): the adiabatic passage of 50 s gives −0.8128, a linear sweep of
    // 2 s −0.25. A field conserves Σ I_z, so no field profile can beat the
    // unitary bound within each block of total M; an optimised profile of
    // 2 s and 40 steps, started at zero field where the order is formed,
    // reaches that bound.
    let mut three = SpinSystem::new(0.0).with_frame(Frame::Laboratory);
    three.add_spin(Isotope::H1, 2.0).unwrap();
    three.add_spin(Isotope::H1, 4.0).unwrap();
    three.add_spin(Isotope::C13, 170.0).unwrap();
    three.couple(0, 1, 7.2).unwrap();
    three.couple(0, 2, 5.8).unwrap();
    three.couple(1, 2, -1.4).unwrap();
    let start = three
        .time_average(&three.parahydrogen_state(0, 1, 1.0).unwrap())
        .unwrap();
    let minus_p = three.iz(2).unwrap() * C64::new(-2.0, 0.0);
    let mut bound = 0.0;
    for m in [-3, -1, 1, 3] {
        let states: Vec<usize> = (0..three.dimension())
            .filter(|&s| {
                (0..3)
                    .map(|i| (2.0 * three.m(i, s)).round() as i32)
                    .sum::<i32>()
                    == m
            })
            .collect();
        let sub = |op: &CMatrix| {
            CMatrix::from_fn(states.len(), states.len(), |r, c| {
                op[(states[r], states[c])]
            })
        };
        bound += unitary_bound(&sub(&start), &sub(&minus_p)).unwrap();
    }
    assert!((bound - 0.8128).abs() < 1e-4, "{bound}");
    let linear = three
        .evolve(
            &start,
            &[Event::Sweep {
                to: 5e-5,
                duration: 2.0,
                steps: 2000,
            }],
        )
        .unwrap();
    let swept = three.expectation(&linear, &minus_p).unwrap().re;
    assert!(swept < 0.3, "{swept}");
    let f = Control::field(0.0, 5e-5, 2.0, 40, start.clone(), minus_p.clone());
    let options = Options {
        max_iterations: 1000,
        ..Options::default()
    };
    let got = three.optimize_control(&f, &[0.0; 40], &options).unwrap();
    assert!(
        got.transfer <= bound + 1e-9 && got.transfer > bound - 1e-6,
        "{} vs {bound}",
        got.transfer
    );
    let after = three
        .evolve(&start, &three.control_events(&f, &got.controls).unwrap())
        .unwrap();
    let want = three.expectation(&after, &minus_p).unwrap().re;
    assert!((got.transfer - want).abs() < 1e-10);
}

#[test]
fn control_input_is_checked() {
    let sys = pair(140.0);
    let iz = sys.iz(0).unwrap();
    let ok = Control::rf(channels(), Some(1000.0), 1e-3, 4, iz.clone(), iz.clone());
    let u = ok.random_start(0, 100.0);
    assert!(sys.control_transfer(&ok, &u).is_ok());
    assert!(sys.control_transfer(&ok, &u[1..]).is_err());
    let mut bad = ok.clone();
    bad.steps = 0;
    assert!(sys.control_transfer(&bad, &[]).is_err());
    let mut bad = ok.clone();
    bad.ensemble = vec![Member {
        weight: 1.0,
        scales: vec![1.0],
        offsets: vec![],
    }];
    assert!(sys.control_transfer(&bad, &u).is_err());
    let mut bad = ok.clone();
    bad.drive = Drive::Rf {
        targets: vec![Target::Spin(5)],
        limit: None,
    };
    assert!(sys.control_transfer(&bad, &u[..8]).is_err());
    let lab = pair(140.0).with_frame(Frame::Laboratory);
    assert!(lab.control_transfer(&ok, &u).is_err());
    let field = Control::field(1.0, 0.5, 1.0, 2, iz.clone(), iz.clone());
    assert!(sys.control_transfer(&field, &[0.7, 0.8]).is_err());
    let field = Control::field(0.0, 1.0, 1.0, 2, iz.clone(), iz);
    assert!(sys.control_transfer(&field, &[0.5, 2.0]).is_err());
    // Penalties: power and roughness with their gradient.
    let mut p = ok.clone();
    p.power = 1e-6;
    p.smoothness = 1e-4;
    let (value, grad) = p.penalty(&u);
    for i in 0..u.len() {
        let mut up = u.clone();
        up[i] += 1e-3;
        let mut down = u.clone();
        down[i] -= 1e-3;
        let fd = (p.penalty(&up).0 - p.penalty(&down).0) / 2e-3;
        assert!((fd - grad[i]).abs() < 1e-8 * (1.0 + fd.abs()));
    }
    assert!(value > 0.0);
    // Bounded controls stay inside their limit, and the maps invert.
    for v in [[1e6, -3e6], [1570.0, 0.0], [3.0, 4.0]] {
        let (x, _) = ok.physical(&[v[0], v[1], 0.0, 0.0]);
        assert!(x[0].hypot(x[1]) <= 1000.0 * (1.0 + 1e-12));
    }
    let back = ok.physical(&ok.free(&u)).0;
    assert!(back.iter().zip(&u).all(|(a, b)| (a - b).abs() < 1e-9));
    // Starts beyond a limit are refused, on it moved inside.
    let opts = Options {
        max_iterations: 2,
        ..Options::default()
    };
    let mut beyond = u.clone();
    beyond[0] = 1500.0;
    assert!(sys.optimize_control(&ok, &beyond, &opts).is_err());
    let inside = ok
        .inside(&[
            1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])
        .unwrap();
    assert!((inside[0] - 980.0).abs() < 1e-9);
    // Non-finite operators and weights that overflow are refused.
    let mut nan = ok.clone();
    nan.initial[(0, 0)] = C64::new(f64::NAN, 0.0);
    assert!(sys.control_transfer(&nan, &u).is_err());
    let mut huge = ok.clone();
    huge.ensemble = vec![
        Member {
            weight: 1e308,
            scales: vec![],
            offsets: vec![],
        };
        2
    ];
    assert!(sys.control_transfer(&huge, &u).is_err());
    // Problems too large for memory are refused before anything is
    // allocated: seven relaxing spins in Liouville space, and a million
    // steps of eight spins without relaxation.
    let mut seven = SpinSystem::new(9.4);
    for k in 0..7 {
        seven.add_spin(Isotope::H1, k as f64).unwrap();
        seven.set_relaxation(k, Some(1.0), None).unwrap();
    }
    let big = Control::rf(
        vec![Target::Isotope(Isotope::H1)],
        Some(1000.0),
        1e-3,
        4,
        seven.iz(0).unwrap(),
        seven.iz(1).unwrap(),
    );
    let err = seven.control_transfer(&big, &[0.0; 8]).unwrap_err();
    assert!(err.to_string().contains("Liouville"), "{err}");
    let mut eight = SpinSystem::new(9.4);
    for k in 0..8 {
        eight.add_spin(Isotope::H1, k as f64).unwrap();
    }
    let long = Control::rf(
        vec![Target::Isotope(Isotope::H1)],
        None,
        1.0,
        1 << 20,
        eight.iz(0).unwrap(),
        eight.iz(1).unwrap(),
    );
    let err = eight
        .control_transfer(&long, &vec![0.0; 2 << 20])
        .unwrap_err();
    assert!(err.to_string().contains("needs about"), "{err}");
    // The multi-channel event survives JSON.
    let events = sys.control_events(&ok, &u).unwrap();
    let text = serde_json::to_string(&events).unwrap();
    let back: Vec<Event> = serde_json::from_str(&text).unwrap();
    assert_eq!(back, events);
}

#[test]
fn optimisation_leaves_bounds_and_mixed_starts() {
    // Every rf pair on its limit: the start is moved inside and the
    // optimum (complete ¹H → ¹³C transfer) found.
    let sys = pair(140.0);
    let (iz, sz) = (sys.iz(0).unwrap(), sys.iz(1).unwrap());
    let c = Control::rf(channels(), Some(2000.0), 1.5 / 140.0, 40, iz, sz.clone());
    let options = Options {
        max_iterations: 1000,
        ..Options::default()
    };
    // (Random phases: constant x pulses on both channels are a stationary
    // point by symmetry.)
    let edge: Vec<f64> = c
        .random_start(4, 1000.0)
        .chunks(2)
        .flat_map(|p| {
            let r = p[0].hypot(p[1]);
            [2000.0 * p[0] / r, 2000.0 * p[1] / r]
        })
        .collect();
    let got = sys.optimize_control(&c, &edge, &options).unwrap();
    assert!(got.transfer > 0.999, "{}", got.transfer);
    // From the identity, relaxation builds carbon polarisation; to make it
    // negative the optimiser must invert it, which a start with tiny
    // transfer must not hide.
    let mut relaxing = pair(140.0);
    relaxing.set_relaxation(1, Some(0.02), None).unwrap();
    let mixed = relaxing.identity_state();
    let c = Control::rf(
        channels(),
        Some(2000.0),
        0.02,
        20,
        mixed,
        sz * C64::new(-1.0, 0.0),
    );
    let start = c.random_start(1, 100.0);
    let (first, _) = relaxing.control_transfer(&c, &start).unwrap();
    let got = relaxing.optimize_control(&c, &start, &options).unwrap();
    assert!(
        first < 0.0 && got.transfer > 0.5 * first.abs(),
        "{first} → {}",
        got.transfer
    );
}

#[test]
fn liouville_transfers_keep_their_precision_at_low_polarisation() {
    // Rotating frame without offsets: the field enters only through the
    // thermal populations, linearly at room temperature, so the transfer
    // is `Σ_k g_k B_k` exactly and doubles with the fields.
    let mut s = pair(140.0);
    s.set_relaxation(0, Some(0.1), None).unwrap();
    s.set_relaxation(1, Some(0.3), Some(0.1)).unwrap();
    let target = s.iz(0).unwrap() + s.iz(1).unwrap();
    for top in [1e-6, 1e-3, 9.0] {
        let f = Control::field(0.0, 2.0 * top, 0.3, 6, s.identity_state(), target.clone());
        let b: Vec<f64> = (0..6).map(|k| top * (0.3 + 0.1 * k as f64)).collect();
        let (v, g) = s.control_transfer(&f, &b).unwrap();
        let euler: f64 = g.iter().zip(&b).map(|(a, x)| a * x).sum();
        assert!((euler / v - 1.0).abs() < 1e-7, "{top}: {euler} vs {v}");
        let twice: Vec<f64> = b.iter().map(|x| 2.0 * x).collect();
        let (w, _) = s.control_transfer(&f, &twice).unwrap();
        assert!((w / (2.0 * v) - 1.0).abs() < 1e-7, "{top}: {w} vs {v}");
    }
}

/// `η*(t)` of the in-phase transfer `S⁻ → I⁻` (Khaneja, Brockett and
/// Glaser 2001, theorem 7): `sin(πJa) sin(πJb)` with `a + 2b = t`,
/// `tan(πJa) = 2 tan(πJb)`, for `t ≤ 3/(2J)`.
fn in_phase_limit(j: f64, t: f64) -> f64 {
    let x = j * t;
    assert!(x <= 1.25, "the root below lies in (0, π/2) up to JT = 1.25");
    // Solve tan(π(x − 2b)) = 2 tan(πb) by bisection, with π(x − 2b) and
    // πb in (0, π/2).
    let f = |b: f64| {
        (std::f64::consts::PI * (x - 2.0 * b)).tan() - 2.0 * (std::f64::consts::PI * b).tan()
    };
    let (mut lo, mut hi) = (((x - 0.5) / 2.0).max(0.0) + 1e-12, x / 2.0 - 1e-12);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(lo) * f(mid) <= 0.0 {
            hi = mid
        } else {
            lo = mid
        }
    }
    let b = 0.5 * (lo + hi);
    let a = x - 2.0 * b;
    (std::f64::consts::PI * a).sin() * (std::f64::consts::PI * b).sin()
}

#[test]
fn transfer_follows_the_time_optimal_limits() {
    let j = 100.0;
    let sys = pair(j);
    let sqrt2 = C64::new(std::f64::consts::SQRT_2, 0.0);
    let im = |i: usize| sys.ix(i).unwrap() - sys.iy(i).unwrap() * C64::new(0.0, 1.0);
    let antiphase = sys.iz(0).unwrap() * im(1) * sqrt2;
    let i_minus = im(0) * C64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
    let s_minus = im(1) * C64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
    // Antiphase 2I_zS⁻ → I⁻: at most sin(πJT/2) for T ≤ 1/J, one after.
    for (x, steps) in [(0.5, 100), (0.8, 120), (1.2, 100)] {
        let t = x / j;
        let c = Control::rf(
            channels(),
            Some(20_000.0),
            t,
            steps,
            antiphase.clone(),
            i_minus.clone(),
        );
        let got = best(&sys, &c, 3, 1000.0, 1500);
        let limit = (std::f64::consts::PI * f64::min(x, 1.0) / 2.0).sin();
        eprintln!("antiphase JT = {x}: {} vs {limit}", got.transfer);
        assert!(got.transfer <= limit + 1e-9 && got.transfer >= 0.993 * limit);
    }
    // In-phase S⁻ → I⁻: one only at 3/(2J).
    for (x, steps) in [(0.8, 120), (1.2, 120), (1.7, 100)] {
        let t = x / j;
        let c = Control::rf(
            channels(),
            Some(20_000.0),
            t,
            steps,
            s_minus.clone(),
            i_minus.clone(),
        );
        let got = best(&sys, &c, 3, 1000.0, 1500);
        let limit = if x >= 1.5 { 1.0 } else { in_phase_limit(j, t) };
        eprintln!("in-phase JT = {x}: {} vs {limit}", got.transfer);
        assert!(got.transfer <= limit + 1e-9 && got.transfer >= 0.999 * limit);
    }
}
