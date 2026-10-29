use super::*;

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

fn close(a: C64, b: C64, tol: f64) -> bool {
    (a - b).norm() <= tol
}

/// Thermal polarisation `tanh(x/2)` of a spin ½ (x = ħγB/kT).
fn thermal_p(sys: &SpinSystem, i: usize) -> f64 {
    (sys.boltzmann(i, sys.field()) / 2.0).tanh()
}

fn commutator(a: &CMatrix, b: &CMatrix) -> CMatrix {
    a * b - b * a
}

#[test]
fn spin_operators_obey_the_angular_momentum_algebra() {
    let mut sys = SpinSystem::new(1.0);
    let h = sys.add_spin(Isotope::H1, 0.0).unwrap();
    let d = sys.add_spin(Isotope::H2, 0.0).unwrap();
    assert_eq!(sys.dimension(), 6);
    for i in [h, d] {
        let (x, y, z) = (sys.ix(i).unwrap(), sys.iy(i).unwrap(), sys.iz(i).unwrap());
        let diff = commutator(&x, &y) - &z * C64::new(0.0, 1.0);
        assert!(diff.norm() < 1e-14);
        // I² = I(I + 1).
        let s = sys.spins()[i].isotope.spin();
        let sq = &x * &x + &y * &y + &z * &z;
        let id = CMatrix::identity(6, 6) * C64::new(s * (s + 1.0), 0.0);
        assert!((sq - id).norm() < 1e-13);
    }
    // Operators of different spins commute.
    assert!(commutator(&sys.ix(h).unwrap(), &sys.iy(d).unwrap()).norm() < 1e-15);
    // The Hamiltonian is Hermitian and conserves F_z in the rotating frame.
    sys.couple(h, d, 3.0).unwrap();
    let ham = sys.hamiltonian().unwrap();
    assert!((&ham - ham.adjoint()).norm() < 1e-9);
}

#[test]
fn thermal_states_follow_boltzmann() {
    let mut sys = SpinSystem::new(9.4).with_temperature(4.2).unwrap();
    let h = sys.add_spin(Isotope::H1, 0.0).unwrap();
    let n = sys.add_spin(Isotope::N15, 0.0).unwrap();
    let d = sys.add_spin(Isotope::H2, 0.0).unwrap();
    let rho = sys.thermal_state().unwrap();
    assert!((rho.trace().re - 1.0).abs() < 1e-14);
    // Spin ½: P = tanh(ħγB/2kT); ¹⁵N (γ < 0) is polarised against the
    // field.
    let x = HBAR * Isotope::H1.gamma() * 9.4 / (BOLTZMANN * 4.2);
    assert!((sys.polarization(&rho, h).unwrap() - (x / 2.0).tanh()).abs() < 1e-14);
    assert!(sys.polarization(&rho, n).unwrap() < 0.0);
    // Spin 1: ⟨I_z⟩ = (e^x − e^{−x})/(e^x + 1 + e^{−x}).
    let y = HBAR * Isotope::H2.gamma() * 9.4 / (BOLTZMANN * 4.2);
    let want = (y.exp() - (-y).exp()) / (y.exp() + 1.0 + (-y).exp());
    assert!((sys.polarization(&rho, d).unwrap() - want).abs() < 1e-14);
}

#[test]
fn free_induction_decay_and_spectrum_of_one_spin() {
    // ¹H at 2 ppm, 9.4 T, T₂ = 0.05 s: s(t) = −(iP/2) e^{iΩt − t/T₂}.
    let mut sys = SpinSystem::new(9.4);
    let h = sys.add_spin(Isotope::H1, 2.0).unwrap();
    sys.set_relaxation(h, Some(0.2), Some(0.05)).unwrap();
    let p = thermal_p(&sys, h);
    let omega = -Isotope::H1.gamma() * 9.4 * 2.0e-6;
    let rho = sys
        .evolve(
            &sys.thermal_state().unwrap(),
            &[Event::pulse(Target::Isotope(Isotope::H1), 90.0, 0.0)],
        )
        .unwrap();
    let dwell = 1e-4;
    let fid = sys.fid(&rho, Isotope::H1, dwell, 512).unwrap();
    for (k, s) in fid.iter().enumerate() {
        let t = k as f64 * dwell;
        let want = C64::new(0.0, -p / 2.0) * C64::from_polar((-t / 0.05).exp(), omega * t);
        // The state is 1/2 plus a deviation of order P ≈ 3·10⁻⁵; the
        // propagator is accurate to about 10⁻¹³ absolute.
        assert!(close(*s, want, 1e-14), "{k}: {s} vs {want}");
    }
    // The spectrum peaks at Ω/2π, which maps back to 2 ppm.
    let spec = Spectrum::from_fid(&fid, dwell, 0.0).unwrap();
    let (best, _) = spec
        .values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.norm().total_cmp(&b.1.norm()))
        .unwrap();
    let resolution = 1.0 / (spec.frequencies.len() as f64 * dwell);
    assert!((spec.frequencies[best] - omega / TWO_PI).abs() <= resolution);
    let lines = sys
        .lines(
            &sys.evolve(
                &sys.thermal_state().unwrap(),
                &[Event::pulse(Target::Isotope(Isotope::H1), 90.0, 0.0)],
            )
            .unwrap(),
            Isotope::H1,
        )
        .unwrap();
    assert_eq!(lines.len(), 1);
    assert!(
        (lines[0].ppm - 2.0).abs() < 1e-9 && (lines[0].frequency - omega / TWO_PI).abs() < 1e-6
    );
    assert!(close(lines[0].amplitude, C64::new(0.0, -p / 2.0), 1e-14));
    // The laboratory frame puts the line at the Larmor frequency.
    let lab = sys.clone().with_frame(Frame::Laboratory);
    let rho = lab
        .evolve(
            &lab.thermal_state().unwrap(),
            &[Event::pulse(Target::Isotope(Isotope::H1), 90.0, 0.0)],
        )
        .unwrap();
    let l = lab.lines(&rho, Isotope::H1).unwrap();
    assert!((l[0].ppm - 2.0).abs() < 1e-6, "{}", l[0].ppm);
    assert!((l[0].frequency + Isotope::H1.larmor(9.4) * (1.0 + 2e-6)).abs() < 1e-3);
}

#[test]
fn spin_echoes_refocus_offsets_but_not_couplings() {
    // One spin: the echo at 2τ is −(iP/2) e^{−2τ/T₂} whatever its offset.
    for shift in [-3.0, 0.5, 7.0] {
        let mut sys = SpinSystem::new(11.7);
        let h = sys.add_spin(Isotope::H1, shift).unwrap();
        sys.set_relaxation(h, Some(1.0), Some(0.3)).unwrap();
        let p = thermal_p(&sys, h);
        let tau = 0.037;
        let rho = sys
            .evolve(&sys.thermal_state().unwrap(), &spin_echo(Isotope::H1, tau))
            .unwrap();
        let s = sys
            .expectation(&rho, &sys.fplus(Isotope::H1).unwrap())
            .unwrap();
        let want = C64::new(0.0, -p / 2.0 * (-2.0 * tau / 0.3).exp());
        assert!(close(s, want, 1e-11 * p), "{shift}: {s} vs {want}");
    }
    // Two weakly coupled protons: the in-phase echo is modulated by
    // cos(2πJτ).
    let mut sys = SpinSystem::new(14.1);
    let a = sys.add_spin(Isotope::H1, 1.0).unwrap();
    let b = sys.add_spin(Isotope::H1, 11.0).unwrap();
    let j = 7.0;
    sys.couple(a, b, j).unwrap();
    let p = thermal_p(&sys, a);
    for tau in [0.01, 0.03, 0.05] {
        let rho = sys
            .evolve(&sys.thermal_state().unwrap(), &spin_echo(Isotope::H1, tau))
            .unwrap();
        let s = sys
            .expectation(&rho, &sys.fplus(Isotope::H1).unwrap())
            .unwrap();
        let want = -p * (TWO_PI * j * tau).cos();
        // Strong-coupling corrections are of order J/Δν ≈ 10⁻³.
        assert!(
            (s.im - want).abs() < 2e-3 * p && s.re.abs() < 2e-3 * p,
            "{tau}: {s} vs {want}"
        );
    }
}

#[test]
fn relaxation_recovers_and_reaches_equilibrium() {
    let mut sys = SpinSystem::new(7.0);
    let h = sys.add_spin(Isotope::H1, 1.0).unwrap();
    let c = sys.add_spin(Isotope::C13, 30.0).unwrap();
    sys.couple(h, c, 145.0).unwrap();
    sys.set_relaxation(h, Some(1.5), Some(0.4)).unwrap();
    sys.set_relaxation(c, Some(4.0), None).unwrap();
    let eq = sys.thermal_state().unwrap();
    let p = thermal_p(&sys, h);
    // Inversion recovery: P(t) = P₀(1 − 2e^{−t/T₁}) before the read pulse.
    for t in [0.1, 0.7, 2.0] {
        let rho = sys
            .evolve(
                &eq,
                &[
                    Event::pulse(Target::Isotope(Isotope::H1), 180.0, 0.0),
                    Event::Delay(t),
                ],
            )
            .unwrap();
        let got = sys.polarization(&rho, h).unwrap();
        let want = p * (1.0 - 2.0 * (-t / 1.5).exp());
        assert!((got - want).abs() < 1e-12, "{t}: {got} vs {want}");
    }
    // From anything, back to equilibrium.
    let mixed = sys.identity_state();
    let late = sys.evolve(&mixed, &[Event::Delay(80.0)]).unwrap();
    assert!((&late - &eq).norm() < 1e-10, "{}", (&late - &eq).norm());
    assert!(p > 1e-5);
}

#[test]
fn superoperator_and_dormand_prince_agree() {
    // Four spins (d = 16): the superoperator and Dormand–Prince steps on
    // the same relaxing system.
    let mut sys = SpinSystem::new(1.0);
    for (k, shift) in [0.0, 1.0, 2.5, 4.0].into_iter().enumerate() {
        let i = sys.add_spin(Isotope::H1, shift).unwrap();
        sys.set_relaxation(i, Some(0.5 + 0.1 * k as f64), Some(0.2))
            .unwrap();
    }
    for i in 0..3 {
        sys.couple(i, i + 1, 3.0 + i as f64).unwrap();
    }
    let rho = sys
        .evolve(
            &sys.thermal_state().unwrap(),
            &[Event::pulse(Target::Isotope(Isotope::H1), 70.0, 30.0)],
        )
        .unwrap();
    let h = sys.hamiltonian().unwrap();
    let jumps = sys.jumps(1.0);
    let t = 0.03;
    let exact = {
        let p = (sys.superoperator(&h, &jumps).unwrap() * C64::new(t, 0.0)).exp();
        let v = &p * CMatrix::from_column_slice(rho.len(), 1, rho.as_slice());
        CMatrix::from_column_slice(16, 16, v.as_slice())
    };
    let steps = sys.dormand_prince(&rho, &h, &jumps, t).unwrap();
    // Local errors of 10⁻¹² per step add up over the steps.
    assert!(
        (&exact - &steps).norm() < 1e-10,
        "{}",
        (&exact - &steps).norm()
    );
    assert!((exact.trace().re - 1.0).abs() < 1e-12);
}

#[test]
fn inept_gains_the_ratio_of_gyromagnetic_ratios() {
    let mut sys = SpinSystem::new(9.4);
    let h = sys.add_spin(Isotope::H1, 4.0).unwrap();
    let c = sys.add_spin(Isotope::C13, 60.0).unwrap();
    let j = 140.0;
    sys.couple(h, c, j).unwrap();
    let eq = sys.thermal_state().unwrap();
    let direct = sys
        .lines(
            &sys.evolve(
                &eq,
                &[Event::pulse(Target::Isotope(Isotope::C13), 90.0, 0.0)],
            )
            .unwrap(),
            Isotope::C13,
        )
        .unwrap();
    // Phase-cycled INEPT: the two steps' lines, weighted by the receiver.
    let mut cycled: Vec<(f64, C64)> = Vec::new();
    for (events, receiver) in inept_cycle(Isotope::H1, Isotope::C13, j) {
        let w = C64::from_polar(0.5, -f64::to_radians(receiver));
        for l in sys
            .lines(&sys.evolve(&eq, &events).unwrap(), Isotope::C13)
            .unwrap()
        {
            match cycled
                .iter_mut()
                .find(|(f, _)| (f - l.frequency).abs() < 1e-6)
            {
                Some((_, a)) => *a += l.amplitude * w,
                None => cycled.push((l.frequency, l.amplitude * w)),
            }
        }
    }
    assert_eq!((direct.len(), cycled.len()), (2, 2));
    let ratio = thermal_p(&sys, h) / thermal_p(&sys, c);
    // The shifts change the ratio by parts in 10⁵.
    assert!((ratio / (Isotope::H1.gamma() / Isotope::C13.gamma()) - 1.0).abs() < 1e-4);
    assert!((ratio - 3.976).abs() < 1e-3);
    // Antiphase lines, each γ_H/γ_C times a line of direct excitation.
    assert!((cycled[0].1 + cycled[1].1).norm() < 1e-9 * cycled[0].1.norm());
    // The two direct lines differ by the proton populations (1 ± P_H).
    let mean_direct = (direct[0].amplitude.norm() + direct[1].amplitude.norm()) / 2.0;
    assert!((cycled[0].1.norm() / mean_direct / ratio - 1.0).abs() < 1e-8);
    // The FID of the cycle carries the same lines.
    let fid = sys
        .phase_cycle(
            &eq,
            &inept_cycle(Isotope::H1, Isotope::C13, j),
            Isotope::C13,
            1e-4,
            8,
        )
        .unwrap();
    let want: C64 = cycled.iter().map(|(_, a)| *a).sum();
    assert!(close(fid[0], want, 1e-12 * want.norm().max(1e-6)));
}

#[test]
fn pasadena_and_altadena_give_their_line_patterns() {
    // Two protons at 1 and 11 ppm, 14.1 T, J = 7 Hz.
    let mut sys = SpinSystem::new(14.1);
    let a = sys.add_spin(Isotope::H1, 1.0).unwrap();
    let b = sys.add_spin(Isotope::H1, 11.0).unwrap();
    let j = 7.0;
    sys.couple(a, b, j).unwrap();
    let nu = |i: usize| -Isotope::H1.larmor(14.1) * sys.spins()[i].shift * 1e-6;
    let find = |lines: &[Line], f: f64| -> C64 {
        lines
            .iter()
            .find(|l| (l.frequency - f).abs() < 0.05)
            .map_or(ZERO, |l| l.amplitude)
    };
    // PASADENA: the singlet averaged at high field, a θ pulse: the line of
    // spin i with its partner in m_j is i·sinθ·cosθ·m_j/2 at ν_i + J m_j.
    let para = sys
        .time_average(&sys.parahydrogen_state(a, b, 1.0).unwrap())
        .unwrap();
    for theta in [20.0f64, 45.0, 70.0] {
        let (s, c) = (theta.to_radians().sin(), theta.to_radians().cos());
        let rho = sys
            .evolve(
                &para,
                &[Event::pulse(Target::Isotope(Isotope::H1), theta, 0.0)],
            )
            .unwrap();
        let lines = sys.lines(&rho, Isotope::H1).unwrap();
        for (i, other) in [(a, b), (b, a)] {
            let _ = other;
            for m in [0.5, -0.5] {
                let got = find(&lines, nu(i) + j * m);
                let want = C64::new(0.0, s * c * m / 2.0);
                assert!(
                    close(got, want, 3e-3 * 0.25),
                    "PASADENA {theta}: {got} vs {want}"
                );
            }
        }
        // No net magnetisation.
        let net: C64 = lines.iter().map(|l| l.amplitude).sum();
        assert!(net.norm() < 1e-10);
    }
    // ALTADENA: the singlet at zero field, carried slowly to 1 T (where
    // Δν = 426 Hz ≫ J, so the eigenstates are |αβ⟩ and |βα⟩ to 10⁻⁴) and
    // then into the magnet, ends in the lower of |αβ⟩ and |βα⟩.
    let zero = sys.clone().with_field(0.0).unwrap();
    let start = zero.parahydrogen_state(a, b, 1.0).unwrap();
    let carried = zero
        .evolve(
            &start,
            &[
                Event::Sweep {
                    to: 1.0,
                    duration: 20.0,
                    steps: 20_000,
                },
                Event::Field(14.1),
            ],
        )
        .unwrap();
    let h = sys.hamiltonian().unwrap();
    // Basis 1 = |αβ⟩, 2 = |βα⟩.
    let lower = if h[(1, 1)].re < h[(2, 2)].re { 1 } else { 2 };
    assert!(
        carried[(lower, lower)].re > 0.995,
        "{}",
        carried[(lower, lower)].re
    );
    let (up, down) = if lower == 1 { (a, b) } else { (b, a) };
    let theta = 30.0f64;
    let (s, c) = (theta.to_radians().sin(), theta.to_radians().cos());
    // The small zero-quantum coherence the field jump leaves dephases
    // before the pulse, as hydrogenation times differ.
    let settled = sys.time_average(&carried).unwrap();
    let rho = sys
        .evolve(
            &settled,
            &[Event::pulse(Target::Isotope(Isotope::H1), theta, 0.0)],
        )
        .unwrap();
    let lines = sys.lines(&rho, Isotope::H1).unwrap();
    // Spin in α: i(−s/4 + cs m/2); spin in β: i(s/4 + cs m/2), m of the
    // partner.
    for m in [0.5, -0.5] {
        let got = find(&lines, nu(up) + j * m);
        let want = C64::new(0.0, -s / 4.0 + c * s * m / 2.0);
        assert!(
            close(got, want, 0.01 * s),
            "ALTADENA α {m}: {got} vs {want}"
        );
        let got = find(&lines, nu(down) + j * m);
        let want = C64::new(0.0, s / 4.0 + c * s * m / 2.0);
        assert!(
            close(got, want, 0.01 * s),
            "ALTADENA β {m}: {got} vs {want}"
        );
    }
}

/// Fields of the level anti-crossings `|S₀ α⟩ ↔ |T₊ β⟩` and
/// `|S₀ β⟩ ↔ |T₋ α⟩` of hydrides A, B and substrate S (J_BS = 0):
/// `ν_S − ν_H = ±(J_AB − J_AS/4)`, `ν` linear in the field.
fn lac_fields(sys: &SpinSystem, h: usize, s: usize, j_ab: f64, j_as: f64) -> Vec<f64> {
    let nu = |i: usize, b: f64| {
        let sp = &sys.spins()[i];
        match sys.frame() {
            Frame::Rotating => -sp.isotope.larmor(b) * sp.shift * 1e-6,
            Frame::Laboratory => -sp.isotope.larmor(b) * (1.0 + sp.shift * 1e-6),
        }
    };
    let k = nu(s, 1.0) - nu(h, 1.0);
    [j_ab - j_as / 4.0, -(j_ab - j_as / 4.0)]
        .into_iter()
        .map(|r| r / k)
        .filter(|b| *b > 0.0)
        .collect()
}

/// Polarisation of the substrate after a long, random contact with the
/// catalyst at field `b`.
fn sabre_transfer(sys: &SpinSystem, b: f64) -> f64 {
    let at = sys.clone().with_field(b).unwrap();
    let rho = at
        .time_average(&at.parahydrogen_state(0, 1, 1.0).unwrap())
        .unwrap();
    at.polarization(&rho, 2).unwrap()
}

#[test]
fn sabre_transfers_at_the_level_anti_crossing() {
    // Hydrides at −22 ppm, substrate proton at 8 ppm: about 5.7 mT.
    let mut sys = SpinSystem::new(0.0);
    sys.add_spin(Isotope::H1, -22.0).unwrap();
    sys.add_spin(Isotope::H1, -22.0).unwrap();
    sys.add_spin(Isotope::H1, 8.0).unwrap();
    let (j_ab, j_as) = (-7.0, 1.2);
    sys.couple(0, 1, j_ab).unwrap();
    sys.couple(0, 2, j_as).unwrap();
    // Substrate ¹⁵N at 250 ppm in the laboratory frame: about 0.15 μT.
    let mut sheath = SpinSystem::new(0.0).with_frame(Frame::Laboratory);
    sheath.add_spin(Isotope::H1, -22.0).unwrap();
    sheath.add_spin(Isotope::H1, -22.0).unwrap();
    sheath.add_spin(Isotope::N15, 250.0).unwrap();
    let (n_ab, n_as) = (-7.0, -2.0);
    sheath.couple(0, 1, n_ab).unwrap();
    sheath.couple(0, 2, n_as).unwrap();
    for (sys, j_ab, j_as) in [(&sys, j_ab, j_as), (&sheath, n_ab, n_as)] {
        let lac = lac_fields(sys, 0, 2, j_ab, j_as);
        assert_eq!(lac.len(), 1, "{lac:?}");
        let lac = lac[0];
        let grid: Vec<f64> = (0..=400).map(|k| lac * (0.5 + k as f64 / 400.0)).collect();
        let transfer: Vec<f64> = grid.iter().map(|&b| sabre_transfer(sys, b).abs()).collect();
        let (best, peak) = transfer
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.total_cmp(y.1))
            .unwrap();
        assert!(
            (grid[best] / lac - 1.0).abs() < 0.02,
            "peak at {} T, LAC at {lac} T",
            grid[best]
        );
        // At half and one and a half times that field, the transfer is
        // much weaker.
        assert!(*peak > 0.4 && transfer[0] < 0.2 * peak && transfer[400] < 0.2 * peak);
    }
}

#[test]
fn shaped_pulses_and_sweeps_are_consistent() {
    let mut sys = SpinSystem::new(9.4);
    let h = sys.add_spin(Isotope::H1, 0.0).unwrap();
    let c = sys.add_spin(Isotope::C13, 0.0).unwrap();
    sys.couple(h, c, 0.0).unwrap();
    let eq = sys.thermal_state().unwrap();
    // On resonance, a rectangular pulse of ν₁ for τ is a hard pulse of
    // 360°·ν₁τ, also split into pieces.
    let shaped = Event::Shaped {
        target: Target::Isotope(Isotope::H1),
        segments: vec![
            Segment {
                duration: 5e-6,
                amplitude: 25_000.0,
                phase: 30.0,
            },
            Segment {
                duration: 5e-6,
                amplitude: 25_000.0,
                phase: 30.0,
            },
        ],
    };
    let a = sys.evolve(&eq, &[shaped]).unwrap();
    let b = sys
        .evolve(
            &eq,
            &[Event::pulse(Target::Isotope(Isotope::H1), 90.0, 30.0)],
        )
        .unwrap();
    assert!((&a - &b).norm() < 1e-14);
    // A selective pulse on one spin equals the isotope pulse when it is
    // the only one.
    let s = sys
        .evolve(&eq, &[Event::pulse(Target::Spin(h), 90.0, 30.0)])
        .unwrap();
    assert!((&s - &b).norm() < 1e-15);
    // A sweep ending where it starts, in one piece, is a delay.
    let d = sys.evolve(&b, &[Event::Delay(0.01)]).unwrap();
    let w = sys
        .evolve(
            &b,
            &[Event::Sweep {
                to: 9.4,
                duration: 0.01,
                steps: 1,
            }],
        )
        .unwrap();
    assert!((&d - &w).norm() < 1e-14);
    // Shaped pulses need the rotating frame; pulses need their isotope.
    let lab = sys.clone().with_frame(Frame::Laboratory);
    assert!(
        lab.evolve(
            &eq,
            &[Event::Shaped {
                target: Target::Spin(h),
                segments: vec![]
            }]
        )
        .is_err()
    );
    assert!(
        sys.evolve(
            &eq,
            &[Event::pulse(Target::Isotope(Isotope::N15), 90.0, 0.0)]
        )
        .is_err()
    );
}

#[test]
fn deuterium_relaxes_to_its_thermal_state() {
    let mut sys = SpinSystem::new(3.0).with_temperature(1.5).unwrap();
    let d = sys.add_spin(Isotope::H2, 0.0).unwrap();
    let h = sys.add_spin(Isotope::H1, 2.0).unwrap();
    sys.couple(d, h, 1.0).unwrap();
    sys.set_relaxation(d, Some(0.3), Some(0.2)).unwrap();
    sys.set_relaxation(h, Some(0.6), None).unwrap();
    let eq = sys.thermal_state().unwrap();
    let late = sys
        .evolve(&sys.identity_state(), &[Event::Delay(30.0)])
        .unwrap();
    assert!((&late - &eq).norm() < 1e-9);
    // A spin-1 pulse of 360° is the identity, of 180° inverts I_z.
    let p = sys.polarization(&eq, d).unwrap();
    let flip = sys
        .evolve(&eq, &[Event::pulse(Target::Spin(d), 180.0, 0.0)])
        .unwrap();
    assert!((sys.polarization(&flip, d).unwrap() + p).abs() < 1e-14);
}

#[test]
fn coupling_graphs_and_invalid_input() {
    let mut sys = SpinSystem::new(1.0);
    for _ in 0..4 {
        sys.add_spin(Isotope::H1, 0.0).unwrap();
    }
    sys.couple(0, 1, 7.0)
        .unwrap()
        .couple(1, 2, -3.0)
        .unwrap()
        .couple(2, 3, 1.0)
        .unwrap();
    sys.couple(2, 3, 0.0).unwrap();
    let g = sys.coupling_graph().unwrap();
    assert_eq!(g.num_edges(), 2);
    assert!(sys.couple(0, 0, 1.0).is_err());
    assert!(sys.couple(0, 9, 1.0).is_err());
    assert!(sys.set_relaxation(0, Some(1.0), Some(3.0)).is_err());
    assert!(sys.set_relaxation(0, Some(-1.0), None).is_err());
    assert!(sys.clone().with_field(-1.0).is_err());
    assert!(sys.clone().with_temperature(0.0).is_err());
    assert!(
        sys.evolve(&sys.identity_state(), &[Event::Delay(-1.0)])
            .is_err()
    );
    assert!(
        sys.fid(&sys.identity_state(), Isotope::H1, 0.0, 10)
            .is_err()
    );
    assert!(Isotope::from_name("17O").is_err());
    assert_eq!(Isotope::from_name("c13").unwrap(), Isotope::C13);
    let mut big = SpinSystem::new(1.0);
    for _ in 0..11 {
        big.add_spin(Isotope::H1, 0.0).unwrap();
    }
    assert!(big.add_spin(Isotope::H1, 0.0).is_err());
    // JSON round trip, and its validation.
    let text = serde_json::to_string(&sys).unwrap();
    let back: SpinSystem = serde_json::from_str(&text).unwrap();
    assert_eq!(back, sys);
    let bad = text.replace("\"field\":1.0", "\"field\":-1.0");
    let bad: SpinSystem = serde_json::from_str(&bad).unwrap();
    assert!(bad.validate().is_err());
}
