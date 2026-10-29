use super::*;
use crate::spin::{Event, Frame, Isotope, Target};

/// `J(ω) = τ/(1 + ω²τ²)`.
fn j(w: f64, tau: f64) -> f64 {
    tau / (1.0 + w * w * tau * tau)
}

/// Two spins at distance `r` (Å) along x, tumbling with `tau`.
fn pair(x: Isotope, y: Isotope, r: f64, b: f64, tau: f64, frame: Frame) -> SpinSystem {
    let mut sys = SpinSystem::new(b).with_frame(frame);
    let i = sys.add_spin(x, 1.0).unwrap();
    let k = sys.add_spin(y, 3.0).unwrap();
    sys.set_position(i, [0.0, 0.0, 0.0]).unwrap();
    sys.set_position(k, [r, 0.0, 0.0]).unwrap();
    sys.with_correlation_time(Some(tau)).unwrap()
}

/// Rates `[ρ_I, σ_IS, R2_I]` of Solomon (Phys. Rev. 99, 559 (1955)) for
/// dipolar spins ½ at angular frequencies `wi`, `ws` (signed).
fn solomon(b: f64, wi: f64, ws: f64, tau: f64) -> [f64; 3] {
    let k = b * b / 10.0;
    [
        k * (j(wi - ws, tau) + 3.0 * j(wi, tau) + 6.0 * j(wi + ws, tau)),
        k * (6.0 * j(wi + ws, tau) - j(wi - ws, tau)),
        k / 2.0
            * (4.0 * j(0.0, tau)
                + j(wi - ws, tau)
                + 3.0 * j(wi, tau)
                + 6.0 * j(ws, tau)
                + 6.0 * j(wi + ws, tau)),
    ]
}

fn larmor(sys: &SpinSystem, i: usize) -> f64 {
    let s = &sys.spins()[i];
    -s.isotope.gamma() * sys.field() * (1.0 + s.shift * 1e-6)
}

#[test]
fn dipolar_pairs_follow_solomon() {
    for (x, y, r) in [
        (Isotope::H1, Isotope::H1, 1.78),
        (Isotope::C13, Isotope::H1, 1.09),
    ] {
        for tau in [1e-12, 1e-10, 3e-9] {
            for frame in [Frame::Rotating, Frame::Laboratory] {
                let sys = pair(x, y, r, 9.4, tau, frame);
                let b = sys.dipolar_coupling(0, 1).unwrap();
                let want = solomon(b, larmor(&sys, 0), larmor(&sys, 1), tau);
                let ops = [
                    sys.iz(0).unwrap(),
                    sys.iz(1).unwrap(),
                    sys.iplus(0).unwrap(),
                ];
                let g = sys.relaxation_rates(&ops).unwrap();
                let scale = want[0];
                for (got, want) in [
                    (g[(0, 0)], want[0]),
                    (g[(0, 1)], want[1]),
                    (g[(2, 2)], want[2]),
                ] {
                    assert!(
                        (got - want).abs() < 1e-9 * scale,
                        "{x:?}-{y:?} τ={tau} {frame:?}: {got} vs {want}"
                    );
                }
            }
        }
    }
    // Like spins: the total magnetisation relaxes at (3/10) b² (J(ω) +
    // 4J(2ω)) (Bloembergen, Purcell and Pound), 3b²τ/2 when ωτ ≪ 1.
    let sys = pair(Isotope::H1, Isotope::H1, 1.78, 9.4, 1e-12, Frame::Rotating);
    let b = sys.dipolar_coupling(0, 1).unwrap();
    let total = sys.fz(Isotope::H1).unwrap();
    let rate = sys.relaxation_rates(&[total]).unwrap()[(0, 0)];
    let w = Isotope::H1.gamma() * 9.4;
    let bpp = 0.3 * b * b * (j(w, 1e-12) + 4.0 * j(2.0 * w, 1e-12));
    assert!((rate / bpp - 1.0).abs() < 1e-5, "{rate} vs {bpp}");
    assert!((rate / (1.5 * b * b * 1e-12) - 1.0).abs() < 1e-4);
}

#[test]
fn the_overhauser_enhancement_of_carbon_is_one_plus_half_the_gamma_ratio() {
    // Extreme narrowing (ωτ ≈ 2·10⁻³): η = 1 + (σ/ρ_C)(γ_H/γ_C) →
    // 1 + γ_H/(2γ_C) ≈ 2.99.
    let sys = pair(Isotope::C13, Isotope::H1, 1.09, 9.4, 1e-12, Frame::Rotating);
    let g = sys
        .relaxation_rates(&[sys.iz(0).unwrap(), sys.iz(1).unwrap()])
        .unwrap();
    let ratio = Isotope::H1.gamma() / Isotope::C13.gamma();
    let eta = 1.0 + g[(0, 1)] / g[(0, 0)] * ratio;
    assert!((eta - (1.0 + ratio / 2.0)).abs() < 1e-4, "{eta}");
    assert!((eta - 2.988).abs() < 1e-3);
    // The transient: proton inverted, carbon follows Solomon's equations.
    let tau = 2e-10;
    let sys = pair(Isotope::C13, Isotope::H1, 1.09, 9.4, tau, Frame::Rotating);
    let b = sys.dipolar_coupling(0, 1).unwrap();
    let (wc, wh) = (larmor(&sys, 0), larmor(&sys, 1));
    let [rc, s, _] = solomon(b, wc, wh, tau);
    let [rh, _, _] = solomon(b, wh, wc, tau);
    let eq = sys.thermal_state().unwrap();
    let (c0, h0) = (
        sys.polarization(&eq, 0).unwrap(),
        sys.polarization(&eq, 1).unwrap(),
    );
    let inverted = sys
        .evolve(
            &eq,
            &[Event::pulse(Target::Isotope(Isotope::H1), 180.0, 0.0)],
        )
        .unwrap();
    // d/dt (c − c0, h − h0) = −[[ρ_C, σ], [σ, ρ_H]] (c − c0, h − h0).
    let m = nalgebra::Matrix2::new(rc, s, s, rh);
    let eig = m.symmetric_eigen();
    for t in [0.5, 2.0, 6.0] {
        let late = sys.evolve(&inverted, &[Event::Delay(t)]).unwrap();
        let decay = nalgebra::Matrix2::from_diagonal(&eig.eigenvalues.map(|l| (-l * t).exp()));
        let p = eig.eigenvectors * decay * eig.eigenvectors.transpose();
        let start = nalgebra::Vector2::new(0.0, -2.0 * h0);
        let want = p * start;
        let got = sys.polarization(&late, 0).unwrap() - c0;
        assert!(
            (got - want[0]).abs() < 1e-7 * c0.abs(),
            "{t}: {got} vs {}",
            want[0]
        );
        let got = sys.polarization(&late, 1).unwrap() - h0;
        assert!(
            (got - want[1]).abs() < 1e-7 * h0.abs(),
            "{t}: {got} vs {}",
            want[1]
        );
    }
}

#[test]
fn shielding_anisotropy_relaxes_with_the_square_of_the_field() {
    // R₁ = (ω₀²/5)(σ:σ) J(ω) and R₂ = (ω₀²/30)(σ:σ)(4J(0) + 3J(ω)) for
    // the traceless symmetric part σ, ω₀ = γB and ω = γB(1 + δ) the
    // Larmor frequency; for an axial tensor σ:σ = (2/3)Δσ².
    let tensor = [[40.0, 15.0, -7.0], [5.0, -60.0, 22.0], [-7.0, 30.0, 35.0]];
    let aniso = anisotropic(&tensor);
    let contraction: f64 = aniso.iter().flatten().map(|x| x * x).sum::<f64>() * 1e-12;
    let mut rates = Vec::new();
    for b in [4.7, 9.4, 18.8] {
        for tau in [1e-12, 1e-9] {
            let mut sys = SpinSystem::new(b);
            sys.add_spin(Isotope::C13, 170.0).unwrap();
            sys.set_csa(0, Some(tensor)).unwrap();
            let sys = sys.with_correlation_time(Some(tau)).unwrap();
            let g = sys
                .relaxation_rates(&[sys.iz(0).unwrap(), sys.iplus(0).unwrap()])
                .unwrap();
            let w0 = Isotope::C13.gamma() * b;
            let w = w0 * (1.0 + 170e-6);
            let r1 = w0 * w0 / 5.0 * contraction * j(w, tau);
            let r2 = w0 * w0 / 30.0 * contraction * (4.0 * j(0.0, tau) + 3.0 * j(w, tau));
            assert!(
                (g[(0, 0)] / r1 - 1.0).abs() < 1e-9,
                "{b} {tau}: {} vs {r1}",
                g[(0, 0)]
            );
            assert!(
                (g[(1, 1)] / r2 - 1.0).abs() < 1e-9,
                "{b} {tau}: {} vs {r2}",
                g[(1, 1)]
            );
            if tau == 1e-12 {
                rates.push(g[(0, 0)]);
            }
        }
    }
    // Four times per doubling of the field, in extreme narrowing.
    assert!((rates[1] / rates[0] - 4.0).abs() < 1e-3 && (rates[2] / rates[1] - 4.0).abs() < 1e-3);
    // An axial tensor of anisotropy Δσ: R₁ = (2/15) ω² Δσ² J(ω).
    let mut sys = SpinSystem::new(9.4);
    sys.add_spin(Isotope::C13, 170.0).unwrap();
    sys.set_csa(0, Some(axial_csa(150.0, [1.0, 2.0, 0.5]).unwrap()))
        .unwrap();
    let sys = sys.with_correlation_time(Some(5e-11)).unwrap();
    let w = Isotope::C13.gamma() * 9.4;
    let want = 2.0 / 15.0 * (w * 150e-6).powi(2) * j(w * (1.0 + 170e-6), 5e-11);
    let got = sys.relaxation_rates(&[sys.iz(0).unwrap()]).unwrap()[(0, 0)];
    assert!((got / want - 1.0).abs() < 1e-9, "{got} vs {want}");
}

#[test]
fn singlet_order_is_immune_to_the_pair_s_own_dipolar_coupling() {
    // Two equivalent protons (Carravetta and Levitt 2004): the singlet
    // order I₁·I₂ does not relax by their dipolar coupling, while their
    // magnetisation does.
    let mut sys = SpinSystem::new(9.4);
    for k in 0..2 {
        sys.add_spin(Isotope::H1, 4.0).unwrap();
        sys.set_position(k, [1.8 * k as f64, 0.0, 0.0]).unwrap();
    }
    sys.couple(0, 1, 7.0).unwrap();
    let sys = sys.with_correlation_time(Some(5e-11)).unwrap();
    let singlet = sys.dot(0, 1).unwrap();
    let g = sys
        .relaxation_rates(&[singlet.clone(), sys.fz(Isotope::H1).unwrap()])
        .unwrap();
    assert!(g[(1, 1)] > 0.01);
    assert!(g[(0, 0)].abs() < 1e-12 * g[(1, 1)], "{}", g[(0, 0)]);
    // A third proton nearby, or shielding anisotropy, does relax it.
    let mut third = sys.clone();
    let k = third.add_spin(Isotope::H1, 2.0).unwrap();
    third.set_position(k, [0.9, 2.5, 0.0]).unwrap();
    let ops = [third.dot(0, 1).unwrap()];
    assert!(third.relaxation_rates(&ops).unwrap()[(0, 0)] > 1e-3 * g[(1, 1)]);
    let mut csa = sys.clone();
    csa.set_csa(0, Some(axial_csa(10.0, [0.0, 0.0, 1.0]).unwrap()))
        .unwrap();
    assert!(
        csa.relaxation_rates(std::slice::from_ref(&singlet))
            .unwrap()[(0, 0)]
            > 0.0
    );
    // The singlet population lives on while the magnetisation relaxes.
    let rho = sys.parahydrogen_state(0, 1, 1.0).unwrap();
    let late = sys.evolve(&rho, &[Event::Delay(20.0 / g[(1, 1)])]).unwrap();
    let before = sys.expectation(&rho, &singlet).unwrap().re;
    let after = sys.expectation(&late, &singlet).unwrap().re;
    assert!((after / before - 1.0).abs() < 1e-6, "{before} → {after}");
}

#[test]
fn redfield_relaxation_ends_in_thermal_equilibrium() {
    // ¹H–¹³C–¹H with dipolar couplings, CSA and their cross-correlations,
    // at 4 K so that the polarisations are 10⁻³: every state relaxes to
    // the thermal one, exactly in the rotating frame. In the laboratory
    // frame the flip-flop terms between ¹H and ¹³C mix their levels by
    // J/(ν_H − ν_C) ≈ 5·10⁻⁷, which shifts the steady state that much.
    for frame in [Frame::Rotating, Frame::Laboratory] {
        let mut sys = SpinSystem::new(9.4)
            .with_temperature(4.0)
            .unwrap()
            .with_frame(frame);
        let shifts = [(Isotope::H1, 2.0), (Isotope::C13, 30.0), (Isotope::H1, 2.5)];
        let places = [[1.09, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.36, 1.03, 0.0]];
        for (k, ((x, shift), p)) in shifts.into_iter().zip(places).enumerate() {
            sys.add_spin(x, shift).unwrap();
            sys.set_position(k, p).unwrap();
        }
        sys.set_csa(1, Some(axial_csa(40.0, [0.0, 0.0, 1.0]).unwrap()))
            .unwrap();
        sys.couple(0, 1, 140.0).unwrap();
        sys.couple(1, 2, 135.0).unwrap();
        let sys = sys.with_correlation_time(Some(1e-10)).unwrap();
        let eq = sys.thermal_state().unwrap();
        let slowest = sys.relaxation_rates(&[sys.iz(1).unwrap()]).unwrap()[(0, 0)];
        let late = sys
            .evolve(&sys.identity_state(), &[Event::Delay(60.0 / slowest)])
            .unwrap();
        let scale = (&eq - sys.identity_state()).norm();
        let tol = if frame == Frame::Rotating { 1e-8 } else { 2e-6 };
        assert!(
            (&late - &eq).norm() < tol * scale,
            "{frame:?}: {}",
            (&late - &eq).norm() / scale
        );
    }
}

#[test]
fn deuteration_slows_dipolar_relaxation_of_carbon_sixteenfold() {
    // Extreme narrowing: R₁ ∝ γ_S² S(S + 1), so ¹H → ²H divides it by
    // (γ_H/γ_D)² · (3/4)/2 ≈ 15.9.
    let rate = |x| {
        let sys = pair(Isotope::C13, x, 1.09, 9.4, 1e-12, Frame::Rotating);
        sys.relaxation_rates(&[sys.iz(0).unwrap()]).unwrap()[(0, 0)]
    };
    let want = (Isotope::H1.gamma() / Isotope::H2.gamma()).powi(2) * 0.75 / 2.0;
    let got = rate(Isotope::H1) / rate(Isotope::H2);
    assert!((got / want - 1.0).abs() < 1e-4, "{got} vs {want}");
    assert!((got - 15.9).abs() < 0.05);
}

#[test]
fn both_frames_relax_alike_and_input_is_checked() {
    let rot = pair(Isotope::C13, Isotope::H1, 1.09, 9.4, 3e-10, Frame::Rotating);
    let lab = rot.clone().with_frame(Frame::Laboratory);
    let events = [
        Event::pulse(Target::Isotope(Isotope::H1), 180.0, 0.0),
        Event::Delay(0.7),
    ];
    let a = rot.evolve(&rot.thermal_state().unwrap(), &events).unwrap();
    let b = lab.evolve(&lab.thermal_state().unwrap(), &events).unwrap();
    for i in 0..2 {
        let (p, q) = (
            rot.polarization(&a, i).unwrap(),
            lab.polarization(&b, i).unwrap(),
        );
        assert!((p - q).abs() < 1e-7 * p.abs(), "{i}: {p} vs {q}");
    }
    // Input.
    let mut sys = rot.clone();
    assert!(sys.set_position(1, [0.0, 0.0, 0.0]).is_err());
    assert_eq!(sys.spins()[1].position, Some([1.09, 0.0, 0.0]));
    assert!(sys.set_position(0, [f64::NAN, 0.0, 0.0]).is_err());
    assert!(sys.set_csa(0, Some([[f64::INFINITY; 3]; 3])).is_err());
    assert!(sys.clone().with_correlation_time(Some(-1.0)).is_err());
    assert!(sys.dipolar_coupling(0, 0).is_err());
    assert!(axial_csa(10.0, [0.0; 3]).is_err());
    // JSON keeps positions, tensors and the correlation time.
    sys.set_csa(0, Some(axial_csa(80.0, [0.0, 1.0, 0.0]).unwrap()))
        .unwrap();
    let back = SpinSystem::from_json(&sys.to_json()).unwrap();
    assert_eq!(back, sys);
    // Seven relaxing spins in the laboratory frame have blocks of 3432.
    let mut big = SpinSystem::new(9.4).with_frame(Frame::Laboratory);
    for k in 0..7 {
        big.add_spin(Isotope::H1, 0.0).unwrap();
        big.set_position(k, [1.8 * k as f64, 0.0, 0.0]).unwrap();
    }
    let big = big.with_correlation_time(Some(1e-10)).unwrap();
    let err = big.relaxation_rates(&[big.iz(0).unwrap()]).unwrap_err();
    assert!(format!("{err}").contains("Redfield"), "{err}");
}
