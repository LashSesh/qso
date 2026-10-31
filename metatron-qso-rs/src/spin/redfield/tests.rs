use super::*;
use crate::spin::{Event, Frame, Isotope, Target};

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

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
    // Two equivalent protons: the total magnetisation relaxes at
    // (3/10) b² (J(ω) + 4J(2ω)) (Bloembergen, Purcell and Pound),
    // 3b²τ/2 when ωτ ≪ 1.
    for tau in [1e-12, 1e-9] {
        let mut sys = SpinSystem::new(9.4);
        for k in 0..2 {
            sys.add_spin(Isotope::H1, 1.0).unwrap();
            sys.set_position(k, [1.78 * k as f64, 0.0, 0.0]).unwrap();
        }
        let sys = sys.with_correlation_time(Some(tau)).unwrap();
        let b = sys.dipolar_coupling(0, 1).unwrap();
        let total = sys.fz(Isotope::H1).unwrap();
        let rate = sys.relaxation_rates(&[total]).unwrap()[(0, 0)];
        let w = Isotope::H1.gamma() * 9.4 * (1.0 + 1e-6);
        let bpp = 0.3 * b * b * (j(w, tau) + 4.0 * j(2.0 * w, tau));
        assert!((rate / bpp - 1.0).abs() < 1e-9, "{rate} vs {bpp}");
        if tau == 1e-12 {
            assert!((rate / (1.5 * b * b * tau) - 1.0).abs() < 1e-4);
        }
    }
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
    // Seven relaxing spins have blocks of 3432 elements, beyond a dense
    // superoperator; with one isotope both frames keep all of R.
    let mut big = SpinSystem::new(9.4).with_frame(Frame::Laboratory);
    for k in 0..7 {
        big.add_spin(Isotope::H1, 0.0).unwrap();
        big.set_position(k, [1.8 * k as f64, 0.0, 0.0]).unwrap();
    }
    let big = big.with_correlation_time(Some(1e-10)).unwrap();
    let ops = [
        big.iz(0).unwrap(),
        big.iz(3).unwrap(),
        big.iplus(1).unwrap(),
    ];
    let lab = big.relaxation_rates(&ops).unwrap();
    let rot = big
        .clone()
        .with_frame(Frame::Rotating)
        .relaxation_rates(&ops)
        .unwrap();
    assert!(lab[(0, 0)] > 0.0 && lab[(2, 2)] > lab[(0, 0)]);
    assert!((&lab - &rot).abs().max() < 1e-12 * lab.abs().max());
}

#[test]
fn long_delays_stay_finite() {
    // A C–H pair relaxes towards equilibrium for twenty minutes, and a
    // singlet of three protons is stored for twice its lifetime: the
    // block shifts may not trade an overflow for an underflow.
    let sys = pair(Isotope::C13, Isotope::H1, 1.09, 9.4, 3e-10, Frame::Rotating);
    let eq = sys.thermal_state().unwrap();
    let inverted = sys
        .evolve(
            &eq,
            &[Event::pulse(Target::Isotope(Isotope::H1), 180.0, 0.0)],
        )
        .unwrap();
    for frame in [Frame::Rotating, Frame::Laboratory] {
        let s = sys.clone().with_frame(frame);
        let late = s.evolve(&inverted, &[Event::Delay(1200.0)]).unwrap();
        assert!(
            late.iter().all(|z| z.re.is_finite() && z.im.is_finite()),
            "{frame:?}"
        );
        let scale = (&eq - s.identity_state()).norm();
        assert!((&late - &eq).norm() < 2e-6 * scale, "{frame:?}");
    }
    let mut three = SpinSystem::new(9.4);
    for (k, p) in [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [0.9, 3.5, 0.0]]
        .into_iter()
        .enumerate()
    {
        three.add_spin(Isotope::H1, 4.0).unwrap();
        three.set_position(k, p).unwrap();
    }
    three.couple(0, 1, 7.0).unwrap();
    let three = three.with_correlation_time(Some(2e-11)).unwrap();
    let singlet = three.dot(0, 1).unwrap();
    let rate = three
        .relaxation_rates(std::slice::from_ref(&singlet))
        .unwrap()[(0, 0)];
    let rho = three.parahydrogen_state(0, 1, 1.0).unwrap();
    let late = three.evolve(&rho, &[Event::Delay(2.0 / rate)]).unwrap();
    assert!(late.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
    let kept = three.expectation(&late, &singlet).unwrap().re
        / three.expectation(&rho, &singlet).unwrap().re;
    assert!(kept > 0.0 && kept < 0.5, "{kept}");
}

#[test]
fn shielding_and_dipolar_coupling_interfere_with_the_documented_sign() {
    // ¹⁵N–¹H with an axial shielding tensor along the bond: the
    // cross-correlated rate between N_x and 2N_xH_z (TROSY) is
    // (A_CSA : A_DD)/30 · (4J(0) + 3J(ω_N)) with A_CSA = γ_N B σ and
    // A_DD = b(1 − 3nnᵀ), so it changes sign with σ.
    let (b0, tau) = (18.8, 1e-8);
    for anisotropy in [-160.0, 160.0] {
        let mut sys = SpinSystem::new(b0);
        let n = sys.add_spin(Isotope::N15, 120.0).unwrap();
        let h = sys.add_spin(Isotope::H1, 8.3).unwrap();
        sys.set_position(n, [0.0, 0.0, 0.0]).unwrap();
        sys.set_position(h, [1.02, 0.0, 0.0]).unwrap();
        let sigma = axial_csa(anisotropy, [1.0, 0.0, 0.0]).unwrap();
        sys.set_csa(n, Some(sigma)).unwrap();
        let sys = sys.with_correlation_time(Some(tau)).unwrap();
        let b = sys.dipolar_coupling(n, h).unwrap();
        let w = Isotope::N15.gamma() * b0;
        let mut contraction = 0.0;
        for (x, row) in sigma.iter().enumerate() {
            for (y, s) in row.iter().enumerate() {
                let delta = if x == y { 1.0 } else { 0.0 };
                let along = if x == 0 && y == 0 { 1.0 } else { 0.0 };
                contraction += w * 1e-6 * s * b * (delta - 3.0 * along);
            }
        }
        let wn = w * (1.0 + 120e-6);
        let want = contraction / 30.0 * (4.0 * j(0.0, tau) + 3.0 * j(wn, tau));
        let nx = sys.ix(n).unwrap();
        let anti = &nx * &sys.iz(h).unwrap() * C64::new(2.0, 0.0);
        let g = sys.relaxation_rates(&[nx, anti]).unwrap();
        assert!(
            (g[(0, 1)] / want - 1.0).abs() < 1e-9,
            "{anisotropy}: {} vs {want}",
            g[(0, 1)]
        );
        // σ:(1 − 3nnᵀ) = −2Δσ along the bond, and b·γ_N B ∝ γ_N² γ_H > 0.
        assert!(want.abs() > 1.0 && want.signum() == -anisotropy.signum());
    }
}

#[test]
fn redfield_runs_through_every_path() {
    // ¹³C–¹H–¹H with CSA and T₁/T₂ on top: an FID equals propagation to
    // each point, a shaped pulse and short delays (Dormand–Prince) equal
    // the exponential of the full superoperator, field jumps change the
    // relaxation with the field, and the rotating frame keeps exactly the
    // laboratory-frame entries that conserve every isotope's order.
    let mut sys = SpinSystem::new(9.4);
    let shifts = [(Isotope::C13, 30.0), (Isotope::H1, 2.0), (Isotope::H1, 2.5)];
    let places = [[0.0, 0.0, 0.0], [1.09, 0.0, 0.0], [-0.36, 1.03, 0.0]];
    for (k, ((x, shift), p)) in shifts.into_iter().zip(places).enumerate() {
        sys.add_spin(x, shift).unwrap();
        sys.set_position(k, p).unwrap();
    }
    sys.set_csa(0, Some(axial_csa(60.0, [0.0, 0.0, 1.0]).unwrap()))
        .unwrap();
    sys.set_relaxation(1, Some(3.0), Some(1.0)).unwrap();
    sys.couple(0, 1, 140.0).unwrap();
    sys.couple(1, 2, -12.0).unwrap();
    let sys = sys.with_correlation_time(Some(1e-9)).unwrap();
    let rho = sys
        .evolve(
            &sys.thermal_state().unwrap(),
            &[Event::pulse(Target::Isotope(Isotope::H1), 90.0, 0.0)],
        )
        .unwrap();
    let fp = sys.fplus(Isotope::H1).unwrap();
    let fid = sys.fid(&rho, Isotope::H1, 2e-3, 4).unwrap();
    for (k, s) in fid.iter().enumerate() {
        let at = sys.evolve(&rho, &[Event::Delay(k as f64 * 2e-3)]).unwrap();
        let want = sys.expectation(&at, &fp).unwrap();
        assert!(
            (s - want).norm() < 1e-11 * want.norm(),
            "{k}: {s} vs {want}"
        );
    }
    // The full superoperator of the master equation from the blocks.
    let d = 8;
    let relaxation = sys.relaxation(9.4).unwrap();
    let full = |h: &CMatrix| -> CMatrix {
        let mut l = CMatrix::zeros(d * d, d * d);
        for c in 0..d * d {
            let mut x = CMatrix::zeros(d, d);
            x[(c % d, c / d)] = C64::new(1.0, 0.0);
            let y = (h * &x - &x * h) * C64::new(0.0, -1.0) + relaxation.dissipate(&x);
            for r in 0..d * d {
                l[(r, c)] = y[(r % d, r / d)];
            }
        }
        l
    };
    let reference = |h: &CMatrix, t: f64| -> CMatrix {
        let l = full(h);
        let n = d * d;
        let mut aug = CMatrix::zeros(n + 1, n + 1);
        aug.view_mut((0, 0), (n, n))
            .copy_from(&(&l * C64::new(t, 0.0)));
        let drift = CMatrix::from_column_slice(n, 1, relaxation.drift.as_slice());
        aug.view_mut((0, n), (n, 1))
            .copy_from(&(drift * C64::new(t, 0.0)));
        let p = aug.exp();
        let dev = &rho - sys.identity_state();
        let mut v = p.view((0, n), (n, 1)).clone_owned();
        v += p.view((0, 0), (n, n)) * CMatrix::from_column_slice(n, 1, dev.as_slice());
        CMatrix::from_column_slice(d, d, v.as_slice())
    };
    let h = sys.hamiltonian().unwrap();
    let dev = |m: &CMatrix| m - sys.identity_state();
    for (hh, t, conserving) in [
        (h.clone(), 0.05, true),
        (h.clone(), 2e-6, true),
        (
            &h + sys.fx(Isotope::H1).unwrap() * C64::new(TWO_PI * 5e3, 0.0),
            3e-5,
            false,
        ),
    ] {
        let want = reference(&hh, t);
        let got = dev(&sys
            .relax_with(&rho, &hh, t, conserving, &relaxation)
            .unwrap());
        let scale = want.iter().fold(0.0f64, |m, z| m.max(z.norm()));
        let err = (&got - &want).iter().fold(0.0f64, |m, z| m.max(z.norm())) / scale;
        assert!(err < 1e-9, "t = {t}: {err}");
    }
    // A field jump: relaxation at the new field, as for a system there.
    let low = sys.clone().with_field(1.0).unwrap();
    let jumped = sys
        .evolve(&rho, &[Event::Field(1.0), Event::Delay(0.3)])
        .unwrap();
    let direct = low.evolve(&rho, &[Event::Delay(0.3)]).unwrap();
    assert!(
        (&jumped - &direct).norm() < 1e-14,
        "{}",
        (&jumped - &direct).norm()
    );
    // The rotating-frame projection.
    let lab = sys.clone().with_frame(Frame::Laboratory);
    let (rot_r, lab_r) = (
        sys.redfield(9.4).unwrap().unwrap(),
        lab.redfield(9.4).unwrap().unwrap(),
    );
    // Every entry of the rotating-frame blocks, and whether it is kept.
    let entries = |r: &RedfieldBlocks| {
        let mut m = std::collections::BTreeMap::new();
        for (k, (_, elements)) in r.liouville.blocks.iter().enumerate() {
            let block = r.block(k).to_dense();
            for (j, &x) in elements.iter().enumerate() {
                for (i, &y) in elements.iter().enumerate() {
                    m.insert((y, x), block[(i, j)]);
                }
            }
        }
        m
    };
    let (rot_e, lab_e) = (entries(&rot_r), entries(&lab_r));
    let mut dropped = 0.0f64;
    for (key, v) in &lab_e {
        match rot_e.get(key) {
            Some(w) => assert!((v - w).norm() < 1e-12 * v.norm().max(1.0), "{key:?}"),
            None => dropped = dropped.max(v.norm()),
        }
    }
    assert!(rot_e.keys().all(|k| lab_e.contains_key(k)));
    assert!(dropped > 1e-3, "{dropped}");
}

#[test]
fn redfield_refuses_slow_motion_and_impossible_input() {
    // |H₁|·τ_c ≫ 1 is outside Redfield theory: a C–H pair at 1 ms, or two
    // spins 10⁻⁶⁰ Å apart.
    let slow = pair(Isotope::C13, Isotope::H1, 1.09, 9.4, 1e-3, Frame::Rotating);
    let err = slow.relaxation_rates(&[slow.iz(0).unwrap()]).unwrap_err();
    assert!(format!("{err}").contains("Redfield theory"), "{err}");
    assert!(
        slow.evolve(&slow.thermal_state().unwrap(), &[Event::Delay(1.0)])
            .is_err()
    );
    let close = pair(Isotope::H1, Isotope::H1, 1e-60, 9.4, 1e-11, Frame::Rotating);
    assert!(close.relaxation_rates(&[close.iz(0).unwrap()]).is_err());
    // A large protein (τ_c = 50 ns) is still within it.
    let protein = pair(Isotope::N15, Isotope::H1, 1.02, 18.8, 5e-8, Frame::Rotating);
    assert!(protein.relaxation_rates(&[protein.iz(0).unwrap()]).is_ok());
    // Nothing relaxes: the rates are +0, not −0.
    let still = SpinSystem::new(9.4)
        .with_correlation_time(Some(1e-11))
        .unwrap();
    let mut still = still;
    still.add_spin(Isotope::H1, 0.0).unwrap();
    let g = still.relaxation_rates(&[still.iz(0).unwrap()]).unwrap();
    assert!(g[(0, 0)] == 0.0 && g[(0, 0)].is_sign_positive());
}

/// `Y(X)_mn = X_mn J(ω_mn)` in the eigenbasis of the laboratory `H₀`,
/// densely.
fn dense_spectral(x: &CMatrix, blocks: &[Eigenblock], tau: f64) -> CMatrix {
    let d = x.nrows();
    let mut out = CMatrix::zeros(d, d);
    for a in blocks {
        for b in blocks {
            let sub = CMatrix::from_fn(a.states.len(), b.states.len(), |r, c| {
                x[(a.states[r], b.states[c])]
            });
            let mut t = a.vectors.adjoint() * sub * &b.vectors;
            for (r, &ea) in a.values.iter().enumerate() {
                for (c, &eb) in b.values.iter().enumerate() {
                    t[(r, c)] *= j((a.shift - b.shift) + ea - eb, tau);
                }
            }
            let back = &a.vectors * t * b.vectors.adjoint();
            for (r, &i) in a.states.iter().enumerate() {
                for (c, &k) in b.states.iter().enumerate() {
                    out[(i, k)] += back[(r, c)];
                }
            }
        }
    }
    out
}

#[test]
fn the_sparse_superoperator_equals_the_dense_formulas() {
    // R(|a⟩⟨b|) = Σ w (𝒬Y|a⟩⟨b| − 𝒬|a⟩⟨b|Y − Y|a⟩⟨b|𝒬 + |a⟩⟨b|Y𝒬) from
    // dense operators (full-space products, the old construction), read
    // on every block of both frames: ¹³C, two protons and ¹⁵N with
    // positions, shielding anisotropies and strong coupling.
    let mut base = SpinSystem::new(1.5);
    let spins = [
        (Isotope::C13, 40.0, [0.0, 0.0, 0.0]),
        (Isotope::H1, 2.0, [1.09, 0.0, 0.0]),
        (Isotope::H1, 2.02, [-0.36, 1.03, 0.0]),
        (Isotope::N15, 120.0, [-0.5, -0.8, 1.1]),
    ];
    for (k, (x, shift, p)) in spins.into_iter().enumerate() {
        base.add_spin(x, shift).unwrap();
        base.set_position(k, p).unwrap();
    }
    base.set_csa(0, Some(axial_csa(80.0, [0.3, 0.1, 1.0]).unwrap()))
        .unwrap();
    base.set_csa(3, Some(axial_csa(-150.0, [1.0, 0.0, 0.2]).unwrap()))
        .unwrap();
    base.couple(1, 2, 12.0).unwrap();
    base.couple(0, 1, 140.0).unwrap();
    base.couple(0, 3, 15.0).unwrap();
    let base = base.with_correlation_time(Some(2e-10)).unwrap();
    for frame in [Frame::Rotating, Frame::Laboratory] {
        let sys = base.clone().with_frame(frame);
        let b = sys.field();
        let r = sys.redfield(b).unwrap().unwrap();
        let d = sys.dimension();
        let ops: Vec<[CMatrix; 3]> = (0..4)
            .map(|i| [sys.ix(i).unwrap(), sys.iy(i).unwrap(), sys.iz(i).unwrap()])
            .collect();
        let pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
        let half = C64::new(0.5, 0.0);
        let third = C64::new(1.0 / 3.0, 0.0);
        let tensor = |c: Coupling, (x, y): (usize, usize)| -> CMatrix {
            match c {
                Coupling::Dipolar(i, k) => {
                    let dot =
                        &ops[i][0] * &ops[k][0] + &ops[i][1] * &ops[k][1] + &ops[i][2] * &ops[k][2];
                    let mut q = (&ops[i][x] * &ops[k][y] + &ops[i][y] * &ops[k][x]) * half;
                    if x == y {
                        q -= dot * third;
                    }
                    q
                }
                Coupling::Shielding(i) => {
                    let mut q = CMatrix::zeros(d, d);
                    if y == 2 {
                        q += &ops[i][x] * half;
                    }
                    if x == 2 {
                        q += &ops[i][y] * half;
                    }
                    if x == y {
                        q -= &ops[i][2] * third;
                    }
                    q
                }
            }
        };
        let interactions = sys.interactions(b);
        let mut lab = sys.clone();
        lab.frame = Frame::Laboratory;
        let eigen = lab.eigenblocks(&lab.hamiltonian_at(b));
        let mut terms = Vec::new();
        for e in &basis() {
            for &(x, y) in &pairs {
                let mut q = CMatrix::zeros(d, d);
                for (a, c) in &interactions {
                    let alpha: f64 = (0..3)
                        .map(|u| (0..3).map(|v| a[u][v] * e[u][v]).sum::<f64>())
                        .sum();
                    q += tensor(*c, (x, y)) * C64::new(alpha, 0.0);
                }
                let yq = dense_spectral(&q, &eigen, 2e-10);
                terms.push((q, yq, if x == y { 1.0 } else { 2.0 } / 5.0));
            }
        }
        let mut largest = 0.0f64;
        let mut worst = 0.0f64;
        for (k, (_, elements)) in r.liouville.blocks.iter().enumerate() {
            let block = r.block(k).to_dense();
            for (jj, &(a, bb)) in elements.iter().enumerate() {
                let mut x = CMatrix::zeros(d, d);
                x[(a, bb)] = C64::new(1.0, 0.0);
                let mut want = CMatrix::zeros(d, d);
                for (q, yq, w) in &terms {
                    want +=
                        (q * yq * &x - q * &x * yq - yq * &x * q + &x * yq * q) * C64::new(*w, 0.0);
                }
                for (i, &(c, e)) in elements.iter().enumerate() {
                    largest = largest.max(want[(c, e)].norm());
                    worst = worst.max((block[(i, jj)] - want[(c, e)]).norm());
                }
            }
        }
        assert!(
            worst < 1e-12 * largest,
            "{frame:?}: {worst:e} of {largest:e}"
        );
        // Column by column without keeping blocks, as the drift is built.
        let deviation = sys.thermal_state().unwrap() - sys.identity_state();
        let (mut kept, mut fresh) = (CMatrix::zeros(d, d), CMatrix::zeros(d, d));
        r.apply(&deviation, &mut kept, 1.0);
        r.image(&deviation, &mut fresh, 1.0);
        assert!((&kept - &fresh).norm() < 1e-12 * kept.norm());
    }
}
