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

/// `ρ − Tr(ρ)/d`.
fn deviation(rho: &CMatrix) -> CMatrix {
    let d = rho.nrows();
    rho - CMatrix::identity(d, d) * (rho.trace() / d as f64)
}

fn largest(m: &CMatrix) -> f64 {
    m.iter().fold(0.0, |x, z| x.max(z.norm()))
}

/// The Lindblad superoperator on column-stacked `vec(ρ)`, from Kronecker
/// products (`vec(AρB) = (Bᵀ ⊗ A) vec(ρ)`): independent of the blocks.
fn full_superoperator(sys: &SpinSystem, h: &CMatrix, jumps: &[(usize, CMatrix)]) -> CMatrix {
    let d = sys.dimension();
    let id = CMatrix::identity(d, d);
    let mut l = (id.kronecker(h) - h.transpose().kronecker(&id)) * C64::new(0.0, -1.0);
    for (i, op) in jumps {
        let a = sys.embed(*i, op).unwrap();
        let ada = a.adjoint() * &a;
        l += a.conjugate().kronecker(&a);
        l -= id.kronecker(&ada) * C64::new(0.5, 0.0);
        l -= ada.transpose().kronecker(&id) * C64::new(0.5, 0.0);
    }
    l
}

/// `Δ(t)` from the full superoperator for `h`, augmented by `L(1/d)`,
/// with nalgebra's exponential.
fn reference_deviation(sys: &SpinSystem, h: &CMatrix, rho: &CMatrix, t: f64, b: f64) -> CMatrix {
    let d = sys.dimension();
    let l = full_superoperator(sys, h, &sys.jumps(b));
    let mixed = CMatrix::identity(d, d) * C64::new(1.0 / d as f64, 0.0);
    let drift = &l * CMatrix::from_column_slice(d * d, 1, mixed.as_slice());
    let n = d * d;
    let mut aug = CMatrix::zeros(n + 1, n + 1);
    aug.view_mut((0, 0), (n, n))
        .copy_from(&(&l * C64::new(t, 0.0)));
    aug.view_mut((0, n), (n, 1))
        .copy_from(&(drift * (rho.trace() * t)));
    let p = aug.exp();
    let dev = deviation(rho);
    let mut v = p.view((0, n), (n, 1)).clone_owned();
    v += p.view((0, 0), (n, n)) * CMatrix::from_column_slice(n, 1, dev.as_slice());
    CMatrix::from_column_slice(d, d, v.as_slice())
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
fn relaxation_propagators_agree_at_thermal_polarisation() {
    // Four protons at 9.4 T and room temperature (P ≈ 3·10⁻⁵, so the
    // state is 1/16 plus a deviation of 10⁻⁶): the block exponentials,
    // Dormand–Prince steps and nalgebra's exponential of the full
    // superoperator agree relative to the deviation.
    let mut sys = SpinSystem::new(9.4);
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
    assert!(largest(&deviation(&rho)) < 1e-5);
    let h = sys.hamiltonian().unwrap();
    let t = 0.05;
    let reference = reference_deviation(&sys, &h, &rho, t, 9.4);
    let scale = largest(&reference);
    let blocks = deviation(&sys.relax(&rho, &h, t, 9.4, true).unwrap());
    let err = largest(&(&blocks - &reference)) / scale;
    assert!(err < 1e-11, "blocks: {err}");
    let relaxation = sys.relaxation(9.4).unwrap();
    let steps = sys
        .dormand_prince(
            &deviation(&rho),
            &blocks::Sparse::new(&h),
            &relaxation,
            &relaxation.drift,
            t,
        )
        .unwrap();
    let err = largest(&(&steps - &reference)) / scale;
    assert!(err < 1e-7, "Dormand–Prince: {err}");
    // The trace stays 1 and the state Hermitian.
    let full = sys.relax(&rho, &h, t, 9.4, true).unwrap();
    assert!((full.trace() - ONE).norm() < 1e-14);
    assert!(largest(&(&full - full.adjoint())) < 1e-10 * scale);
    // During a shaped pulse the field mixes the coherence orders: one
    // block, here by Dormand–Prince steps.
    let rf = &h + sys.fx(Isotope::H1).unwrap() * C64::new(TWO_PI * 5e3, 0.0);
    let t = 2e-5;
    let reference = reference_deviation(&sys, &rf, &rho, t, 9.4);
    let pulsed = deviation(&sys.relax(&rho, &rf, t, 9.4, false).unwrap());
    let err = largest(&(&pulsed - &reference)) / largest(&reference);
    assert!(err < 1e-9, "shaped: {err}");
    // The free induction decay steps the −1 coherences by one block
    // exponential: as the full propagation to each point.
    let fp = sys.fplus(Isotope::H1).unwrap();
    let fid = sys.fid(&rho, Isotope::H1, 1e-3, 4).unwrap();
    for (k, s) in fid.iter().enumerate() {
        let at = deviation(&reference_deviation(&sys, &h, &rho, k as f64 * 1e-3, 9.4));
        let want = sys.expectation(&at, &fp).unwrap();
        assert!(
            (s - want).norm() < 1e-10 * want.norm(),
            "{k}: {s} vs {want}"
        );
    }
}

#[test]
fn the_laboratory_frame_propagates_as_the_rotating_one() {
    // One isotope: the rotating frame is exact, so polarisations and the
    // magnitudes of the lines agree with the laboratory frame, where the
    // Zeeman frequencies are 10⁵ times the offsets. (Pulse phases refer to
    // the laboratory axes there, so a later pulse's phase differs between
    // the frames: one pulse, then relaxation.)
    let mut sys = SpinSystem::new(9.4);
    let a = sys.add_spin(Isotope::H1, 1.0).unwrap();
    let b = sys.add_spin(Isotope::H1, 3.0).unwrap();
    sys.couple(a, b, 7.0).unwrap();
    sys.set_relaxation(a, Some(1.0), Some(0.3)).unwrap();
    sys.set_relaxation(b, Some(2.0), Some(0.5)).unwrap();
    let lab = sys.clone().with_frame(Frame::Laboratory);
    let events = [
        Event::pulse(Target::Spin(b), 120.0, 45.0),
        Event::Delay(0.03),
    ];
    let rot = sys.evolve(&sys.thermal_state().unwrap(), &events).unwrap();
    let lr = lab.evolve(&lab.thermal_state().unwrap(), &events).unwrap();
    for i in [a, b] {
        let (p, q) = (
            sys.polarization(&rot, i).unwrap(),
            lab.polarization(&lr, i).unwrap(),
        );
        assert!((p - q).abs() < 1e-7 * p.abs(), "{i}: {p} vs {q}");
    }
    // A free induction decay: equal magnitudes, the laboratory one at the
    // Larmor frequency.
    let (lines, lab_lines) = (
        sys.lines(&rot, Isotope::H1).unwrap(),
        lab.lines(&lr, Isotope::H1).unwrap(),
    );
    assert_eq!((lines.len(), lab_lines.len()), (4, 4));
    let larmor = -Isotope::H1.larmor(9.4);
    for (x, y) in lines.iter().zip(&lab_lines) {
        assert!(
            (y.frequency - larmor - x.frequency).abs() < 1e-4,
            "{} {}",
            x.frequency,
            y.frequency
        );
        assert!((x.ppm - y.ppm).abs() < 1e-6);
        let (u, v) = (x.amplitude.norm(), y.amplitude.norm());
        assert!((u - v).abs() < 1e-7 * u, "{u} vs {v}");
    }
}

#[test]
fn hopeless_relaxation_is_refused_early() {
    // Seven spins in the laboratory frame at 9.4 T: blocks of 3432
    // elements and Zeeman frequencies of 10⁹ rad/s.
    let mut sys = SpinSystem::new(9.4).with_frame(Frame::Laboratory);
    for k in 0..7 {
        let i = sys.add_spin(Isotope::H1, k as f64).unwrap();
        sys.set_relaxation(i, Some(1.0), None).unwrap();
    }
    let start = std::time::Instant::now();
    let err = sys
        .evolve(&sys.thermal_state().unwrap(), &[Event::Delay(1.0)])
        .unwrap_err();
    assert!(format!("{err}").contains("operations"), "{err}");
    assert!(start.elapsed().as_secs_f64() < 5.0);
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
    for (events, receiver) in inept_cycle(Isotope::H1, Isotope::C13, j).unwrap() {
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
            &inept_cycle(Isotope::H1, Isotope::C13, j).unwrap(),
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
    // Invalid times leave the spin as it was, and the system valid.
    sys.set_relaxation(0, Some(2.0), Some(1.0)).unwrap();
    assert!(sys.set_relaxation(0, Some(1.0), Some(3.0)).is_err());
    assert!(sys.set_relaxation(0, Some(-1.0), None).is_err());
    assert_eq!(
        (sys.spins()[0].t1, sys.spins()[0].t2),
        (Some(2.0), Some(1.0))
    );
    sys.validate().unwrap();
    assert!(sys.clone().with_field(-1.0).is_err());
    assert!(sys.clone().with_temperature(0.0).is_err());
    let name = |e: QsoError| match e {
        QsoError::InvalidArgument { name, .. } => name,
        other => panic!("{other}"),
    };
    let err = sys
        .evolve(&sys.identity_state(), &[Event::Delay(-1.0)])
        .unwrap_err();
    assert_eq!(name(err), "delay");
    let err = sys
        .fid(&sys.identity_state(), Isotope::H1, 0.0, 10)
        .unwrap_err();
    assert_eq!(name(err), "fid");
    // INEPT needs two isotopes and a coupling.
    for (x, j) in [
        (Isotope::H1, 140.0),
        (Isotope::C13, 0.0),
        (Isotope::C13, f64::NAN),
    ] {
        assert_eq!(name(inept(Isotope::H1, x, j).unwrap_err()), "inept");
        assert!(inept_cycle(Isotope::H1, x, j).is_err());
    }
    // Events and segments reject unknown fields.
    let ok = r#"[{"pulse": {"target": {"isotope": "1H"}, "angle": 90.0, "phase": 0.0}},
                 {"shaped": {"target": {"spin": 0}, "segments": [{"duration": 1e-6, "amplitude": 1e3, "phase": 0.0}]}}]"#;
    assert_eq!(serde_json::from_str::<Vec<Event>>(ok).unwrap().len(), 2);
    for bad in [
        r#"[{"pulse": {"target": {"isotope": "1H"}, "angle": 90.0, "phase": 0.0, "power": 1.0}}]"#,
        r#"[{"shaped": {"target": {"spin": 0}, "segments": [{"duration": 1e-6, "amplitude": 1e3, "phase": 0.0, "x": 1}]}}]"#,
        r#"[{"sweep": {"to": 1.0, "duration": 1.0, "steps": 10, "shape": "linear"}}]"#,
    ] {
        assert!(serde_json::from_str::<Vec<Event>>(bad).is_err(), "{bad}");
    }
    // The field after a sequence.
    let seq = [
        Event::Delay(1.0),
        Event::Field(2.0),
        Event::Sweep {
            to: 0.5,
            duration: 1.0,
            steps: 3,
        },
    ];
    assert_eq!(sys.field_after(&seq), 0.5);
    assert_eq!(sys.field_after(&seq[..2]), 2.0);
    assert_eq!(sys.field_after(&[]), 1.0);
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

#[test]
fn time_averages_do_not_depend_on_the_frame() {
    // ¹H and ¹³C on their carriers: in the rotating frame |αα⟩ and |ββ⟩
    // have the same energy, but their coherence oscillates at the sum of
    // the Larmor frequencies, so the average removes it in both frames.
    let mut sys = SpinSystem::new(9.4);
    let h = sys.add_spin(Isotope::H1, 0.0).unwrap();
    let c = sys.add_spin(Isotope::C13, 0.0).unwrap();
    sys.couple(h, c, 140.0).unwrap();
    let lab = sys.clone().with_frame(Frame::Laboratory);
    // Populations, a double-quantum coherence |αα⟩⟨ββ⟩ and a zero-quantum
    // one |αβ⟩⟨βα⟩ between the isotopes.
    let mut rho = sys.identity_state();
    for (k, p) in [3e-3, 1e-3, -1e-3, -3e-3].into_iter().enumerate() {
        rho[(k, k)] += C64::new(p, 0.0);
    }
    for (r, c, v) in [(0, 3, C64::new(1e-3, 2e-4)), (1, 2, C64::new(-5e-4, 1e-3))] {
        rho[(r, c)] = v;
        rho[(c, r)] = v.conj();
    }
    let a = sys.time_average(&rho).unwrap();
    let b = lab.time_average(&rho).unwrap();
    // The laboratory frame's flip-flop terms mix |αβ⟩ and |βα⟩ by
    // J/(ν_H − ν_C) ≈ 5·10⁻⁷.
    let scale = largest(&deviation(&rho));
    assert!(
        largest(&(&a - &b)) < 1e-5 * scale,
        "{}",
        largest(&(&a - &b)) / scale
    );
    assert!(a[(0, 3)].norm() < 1e-20 && a[(1, 2)].norm() < 1e-20);
    for k in 0..4 {
        assert!((a[(k, k)] - rho[(k, k)]).norm() < 1e-18);
    }
    // Two equivalent uncoupled protons on resonance (H = 0 in the
    // rotating frame): the zero-quantum coherence between the degenerate
    // |αβ⟩ and |βα⟩ stays, the single-quantum one precesses at the Larmor
    // frequency and goes.
    let mut pair = SpinSystem::new(9.4);
    pair.add_spin(Isotope::H1, 0.0).unwrap();
    pair.add_spin(Isotope::H1, 0.0).unwrap();
    let mut rho = pair.identity_state();
    let (zq, sq) = (C64::new(2e-3, 1e-3), C64::new(1e-3, 0.0));
    rho[(1, 2)] = zq;
    rho[(2, 1)] = zq.conj();
    rho[(0, 1)] = sq;
    rho[(1, 0)] = sq;
    let avg = pair.time_average(&rho).unwrap();
    let mut want = pair.identity_state();
    want[(1, 2)] = zq;
    want[(2, 1)] = zq.conj();
    assert!((&avg - &want).norm() < 1e-18, "{}", (&avg - &want).norm());
}

#[test]
fn lines_ignore_rounding() {
    let mut sys = SpinSystem::new(14.1);
    let a = sys.add_spin(Isotope::H1, 1.0).unwrap();
    let b = sys.add_spin(Isotope::H1, 2.2).unwrap();
    let c = sys.add_spin(Isotope::C13, 40.0).unwrap();
    sys.couple(a, b, 7.0).unwrap();
    sys.couple(a, c, 130.0).unwrap();
    // No coherence, no line.
    assert!(
        sys.lines(&sys.identity_state(), Isotope::H1)
            .unwrap()
            .is_empty()
    );
    assert!(
        sys.lines(&sys.thermal_state().unwrap(), Isotope::H1)
            .unwrap()
            .is_empty()
    );
    // Four 90° pulses leave rounding of 10⁻¹⁷ in the coherences.
    let full = vec![Event::pulse(Target::Isotope(Isotope::H1), 90.0, 0.0); 4];
    let back = sys.evolve(&sys.thermal_state().unwrap(), &full).unwrap();
    assert!(sys.lines(&back, Isotope::H1).unwrap().is_empty());
    // The laboratory frame gives the same lines as the rotating one, at
    // the Larmor frequency.
    let one = [Event::pulse(Target::Isotope(Isotope::H1), 90.0, 0.0)];
    let rot = sys
        .lines(
            &sys.evolve(&sys.thermal_state().unwrap(), &one).unwrap(),
            Isotope::H1,
        )
        .unwrap();
    let lab = sys.clone().with_frame(Frame::Laboratory);
    let lines = lab
        .lines(
            &lab.evolve(&lab.thermal_state().unwrap(), &one).unwrap(),
            Isotope::H1,
        )
        .unwrap();
    assert_eq!((rot.len(), lines.len()), (8, 8));
    for (x, y) in rot.iter().zip(&lines) {
        // Strong coupling between ¹H and ¹³C is 10⁻⁷ of the lines.
        assert!((x.ppm - y.ppm).abs() < 1e-5, "{} {}", x.ppm, y.ppm);
        assert!((x.amplitude - y.amplitude).norm() < 1e-5 * x.amplitude.norm());
    }
}

#[test]
fn thermal_states_survive_extreme_fields_and_temperatures() {
    let mut sys = SpinSystem::new(20.0).with_temperature(1e-6).unwrap();
    let h = sys.add_spin(Isotope::H1, 0.0).unwrap();
    let n = sys.add_spin(Isotope::N15, 0.0).unwrap();
    let d = sys.add_spin(Isotope::H2, 0.0).unwrap();
    let rho = sys.thermal_state().unwrap();
    assert!(rho.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
    assert!((rho.trace().re - 1.0).abs() < 1e-14);
    assert!((sys.polarization(&rho, h).unwrap() - 1.0).abs() < 1e-14);
    assert!((sys.polarization(&rho, n).unwrap() + 1.0).abs() < 1e-14);
    assert!((sys.polarization(&rho, d).unwrap() - 1.0).abs() < 1e-14);
}

#[test]
fn the_fourier_transform_is_accurate_at_many_points() {
    // A tone off the grid, 2¹⁵ points: the transform against direct sums.
    let n = 1 << 15;
    let fid: Vec<C64> = (0..n)
        .map(|k| C64::from_polar(1.0, 0.123_456 * k as f64))
        .collect();
    let spec = Spectrum::from_fid(&fid, 1.0, 0.0).unwrap();
    let m = spec.values.len();
    for j in [0usize, 777, m / 2 + 1234, m - 1] {
        let f = spec.frequencies[j];
        let want: C64 = fid
            .iter()
            .enumerate()
            .map(|(k, s)| {
                s * C64::from_polar(if k == 0 { 0.5 } else { 1.0 }, -TWO_PI * f * k as f64)
            })
            .sum();
        assert!((spec.values[j] - want).norm() < 1e-9 * (n as f64), "{j}");
    }
}

#[test]
fn spin_one_relaxes_with_t1_and_t2_at_high_temperature() {
    // ²H on resonance at 10¹⁵ K (k↑ = k↓ to 3·10⁻¹⁹): ⟨I_x⟩ decays
    // exactly with T₂ and ⟨I_z⟩ with T₁, as for spin ½.
    let mut sys = SpinSystem::new(1.0).with_temperature(1e15).unwrap();
    let d = sys.add_spin(Isotope::H2, 0.0).unwrap();
    sys.set_relaxation(d, Some(0.3), Some(0.2)).unwrap();
    for (op, time) in [(sys.ix(d).unwrap(), 0.2), (sys.iz(d).unwrap(), 0.3)] {
        let rho = sys.identity_state() + &op * C64::new(0.01, 0.0);
        for t in [0.05, 0.4, 1.0] {
            let late = sys.evolve(&rho, &[Event::Delay(t)]).unwrap();
            // Tr(I_x²) = Tr(I_z²) = 2 for spin 1.
            let want = 0.02 * (-t / time).exp();
            let got = sys.expectation(&late, &op).unwrap().re;
            // Rounding relative to the initial 0.02.
            assert!((got - want).abs() < 1e-11 * 0.02, "{t}: {got} vs {want}");
        }
    }
}
