use super::*;

const PI: f64 = std::f64::consts::PI;

fn on(nv: &NvCenter, rabi: Option<f64>) -> Microwave {
    Microwave::resonant(nv, -1, rabi)
}

#[test]
fn odmr_lines_lie_at_d_plus_minus_gamma_b_split_by_the_nitrogen() {
    // In the secular approximation the lines are D + m_s γB + m_s A m_I,
    // whatever the quadrupole and nuclear Zeeman terms (equal m_I).
    for nitrogen in [Nitrogen::n14(), Nitrogen::n15()] {
        let nv = NvCenter::new(0.01).with_nitrogen(nitrogen);
        let lines = nv.lines().unwrap();
        let spins: Vec<f64> = match nitrogen.isotope {
            NitrogenIsotope::N14 => vec![1.0, 0.0, -1.0],
            NitrogenIsotope::N15 => vec![0.5, -0.5],
        };
        let mut want: Vec<f64> = [1.0, -1.0]
            .iter()
            .flat_map(|&ms| {
                spins.iter().map(move |&mi| {
                    NV_ZERO_FIELD_SPLITTING
                        + ms * NV_GYROMAGNETIC_RATIO * 0.01
                        + ms * nitrogen.parallel * mi
                })
            })
            .collect();
        want.sort_by(f64::total_cmp);
        assert_eq!(lines.len(), want.len(), "{lines:?}");
        for ((f, w), x) in lines.iter().map(|l| (l.0, l.1)).zip(&want) {
            assert!((f - x).abs() < 1e-3, "{f} vs {x}");
            assert!((w - lines[0].1).abs() < 1e-12);
        }
    }
    // Pulsed ODMR: a π pulse of 0.3 MHz flips one nitrogen state in three
    // at each ¹⁴N line, so the bright population dips to 2/3 there.
    let nv = NvCenter::new(0.01).with_nitrogen(Nitrogen::n14());
    let centre = NV_ZERO_FIELD_SPLITTING - NV_GYROMAGNETIC_RATIO * 0.01;
    let freqs: Vec<f64> = (0..=500).map(|k| centre - 5e6 + 2e4 * k as f64).collect();
    let rabi = 0.3e6;
    let p0 = nv.odmr(&freqs, rabi, 1.0 / (2.0 * rabi)).unwrap();
    let dips: Vec<usize> = (1..p0.len() - 1)
        .filter(|&k| p0[k] < p0[k - 1] && p0[k] <= p0[k + 1] && p0[k] < 0.8)
        .collect();
    assert_eq!(dips.len(), 3, "{dips:?}");
    // Ascending: m_I = −1, 0, 1 (A_N < 0 and m_s = −1).
    for (&k, mi) in dips.iter().zip([-1.0, 0.0, 1.0]) {
        let want = centre - Nitrogen::n14().parallel * mi;
        assert!((freqs[k] - want).abs() <= 2e4, "{} vs {want}", freqs[k]);
        assert!((p0[k] - 2.0 / 3.0).abs() < 0.02, "{}", p0[k]);
    }
    // ODMR is one finite pulse at each frequency.
    let nv = nv.with_carbon(60e3, 30e3);
    for f in [centre - 2.2e6, centre, centre + 0.7e6] {
        let mw = Microwave {
            frequency: f,
            transition: -1,
            rabi: Some(rabi),
        };
        let start = nv.initial_state().unwrap();
        let want = nv.bright(&dense(&nv, &start, &[pulse(180.0, 0.0)], &mw, 0.0));
        let got = nv.odmr(&[f], rabi, 1.0 / (2.0 * rabi)).unwrap()[0];
        assert!((got - want).abs() < 1e-10, "{got} vs {want}");
    }
    // At zero field the nuclear levels of m_s = 0 are degenerate: the lines
    // of both transitions merge at D ± A/2, A = √(A_∥² + A_⊥²), half each.
    let lines = NvCenter::new(0.0).with_carbon(60e3, 30e3).lines().unwrap();
    let a = 60e3f64.hypot(30e3);
    assert_eq!(lines.len(), 2, "{lines:?}");
    for (line, want) in lines.iter().zip([-a / 2.0, a / 2.0]) {
        assert!(
            (line.0 - NV_ZERO_FIELD_SPLITTING - want).abs() < 1e-3,
            "{lines:?}"
        );
        assert!((line.1 - 0.5).abs() < 1e-12, "{lines:?}");
    }
    // Above B = D/γ_e the line of m = −1 is negative: D − γ_e B.
    let high = NvCenter::new(0.15);
    let lines = high.lines().unwrap();
    let want = NV_ZERO_FIELD_SPLITTING - NV_GYROMAGNETIC_RATIO * 0.15;
    assert!(want < 0.0 && (lines[0].0 - want).abs() < 1e-3, "{lines:?}");
    assert!((Microwave::resonant(&high, -1, None).frequency - want).abs() < 1e-3);
    let p = high.odmr(&[want, -want], 1e6, 0.5e-6).unwrap();
    // (Off resonance by gigahertz, the other levels respond at 10⁻⁸.)
    assert!(p[0] < 1e-6 && p[1] > 1.0 - 1e-6, "{p:?}");
}

/// The dense propagation that the blocks replace: full Hamiltonian and
/// pulses, with a static field `bias` (Hz) along the axis throughout.
fn dense(
    nv: &NvCenter,
    rho: &CMatrix,
    events: &[NvEvent],
    drive: &Microwave,
    bias: f64,
) -> CMatrix {
    let sz = nv.embed(0, &NvCenter::electron()[0]);
    let h0 = nv.hamiltonian(drive.frequency).unwrap() + &sz * C64::new(TWO_PI * bias, 0.0);
    let mut rho = rho.clone();
    for event in NvCenter::centred(events, drive).unwrap() {
        let u = match event {
            NvEvent::Free(t) => expm_hermitian(&h0, t),
            NvEvent::Pulse { angle, phase } => match drive.rabi {
                None => nv.embed(0, &NvCenter::ideal_pulse(drive.transition, angle, phase)),
                Some(rabi) => {
                    let (duration, phase) = finite(angle, phase, rabi);
                    expm_hermitian(&(&h0 + nv.drive(rabi, phase)), duration)
                }
            },
        };
        rho = &u * rho * u.adjoint();
    }
    rho
}

#[test]
fn blocks_match_the_dense_propagation() {
    // ¹⁵N with three carbons: arbitrary states, also non-Hermitian ones.
    let nv = NvCenter::new(0.012)
        .with_nitrogen(Nitrogen::n15())
        .with_carbon(90e3, -10e3)
        .with_carbon(-20e3, 45e3)
        .with_carbon(5e3, 2e3);
    let d = nv.dimension();
    let mut rng = Rng::seed_from_u64(9);
    let x = CMatrix::from_fn(d, d, |_, _| C64::new(rng.normal(), rng.normal()));
    let events = [
        pulse(90.0, 30.0),
        NvEvent::Free(0.4e-6),
        pulse(180.0, 0.0),
        NvEvent::Free(0.1e-6),
        NvEvent::Free(0.3e-6),
        pulse(-45.0, 250.0),
    ];
    for transition in [-1, 1] {
        for rabi in [None, Some(8e6)] {
            let mw = Microwave::resonant(&nv, transition, rabi);
            let want = dense(&nv, &x, &events, &mw, 0.0);
            let got = nv.evolve(&x, &events, &mw).unwrap();
            let worst = (got - &want).iter().fold(0.0f64, |m, z| m.max(z.norm()));
            assert!(worst < 1e-10, "{transition} {rabi:?}: {worst}");
        }
    }
    let nv = NvCenter::new(0.031)
        .with_nitrogen(Nitrogen::n14())
        .with_carbon(60e3, 30e3)
        .with_carbon(-20e3, 45e3);
    let d = nv.dimension();
    let mut rng = Rng::seed_from_u64(5);
    let a = CMatrix::from_fn(d, d, |_, _| C64::new(rng.normal(), rng.normal()));
    let rho = &a * a.adjoint();
    let rho = &rho / rho.trace();
    let events = [
        pulse(90.0, 0.0),
        NvEvent::Free(0.3e-6),
        pulse(180.0, 90.0),
        NvEvent::Free(0.6e-6),
        pulse(-70.0, 20.0),
        NvEvent::Free(0.2e-6),
        pulse(180.0, 90.0),
        NvEvent::Free(0.3e-6),
        pulse(90.0, 180.0),
    ];
    let noise = FieldNoise::Telegraph {
        amplitude: 2e5,
        rate: 0.0,
    };
    for transition in [-1, 1] {
        for rabi in [None, Some(12e6)] {
            let mut mw = Microwave::resonant(&nv, transition, rabi);
            mw.frequency += 0.3e6;
            let want = dense(&nv, &rho, &events, &mw, 0.0);
            let got = nv.evolve(&rho, &events, &mw).unwrap();
            let worst = (got - &want).iter().fold(0.0f64, |m, z| m.max(z.norm()));
            assert!(worst < 1e-10, "{transition} {rabi:?}: {worst}");
            // Static telegraph noise: each trajectory sees +a or −a
            // throughout, during finite pulses too.
            let start = nv.initial_state().unwrap();
            let plus = nv.bright(&dense(&nv, &start, &events, &mw, 2e5));
            let minus = nv.bright(&dense(&nv, &start, &events, &mw, -2e5));
            assert!((plus - minus).abs() > 1e-3, "{plus} {minus}");
            let mut seen = [false; 2];
            for seed in 0..12 {
                let (p, _) = nv.signal_with_noise(&events, &mw, &noise, 1, seed).unwrap();
                let k = usize::from((p - plus).abs() > 1e-10);
                assert!(
                    (p - [plus, minus][k]).abs() < 1e-10,
                    "{p} vs {plus}, {minus}"
                );
                seen[k] = true;
            }
            assert!(seen[0] && seen[1]);
        }
    }
}

/// The `m_s = 0` population of a bare centre after `events` in the
/// laboratory frame: a linearly polarised carrier `√2·2πΩ cos(2πft + φ)
/// S_x`, no rotating-wave approximation, the clock starting at the first
/// event; free periods exact, pulses by exponential midpoint steps in the
/// interaction picture of the static Hamiltonian, 64 per carrier period.
fn laboratory(nv: &NvCenter, events: &[NvEvent], drive: &Microwave) -> f64 {
    let rabi = drive.rabi.unwrap();
    let energy: Vec<f64> = [1.0, 0.0, -1.0]
        .iter()
        .map(|m| TWO_PI * (nv.zero_field_splitting * m * m + nv.gyromagnetic_ratio * nv.field * m))
        .collect();
    let sx = NvCenter::electron()[1].clone();
    let omega = TWO_PI * drive.frequency;
    // Interaction picture: ψ_I(t) = e^{iH₀t} ψ(t), populations equal.
    let mut psi = nalgebra::DVector::from_vec(vec![ZERO, C64::new(1.0, 0.0), ZERO]);
    let mut t = 0.0;
    for event in NvCenter::centred(events, drive).unwrap() {
        match event {
            NvEvent::Free(dt) => t += dt,
            NvEvent::Pulse { angle, phase } => {
                let (duration, phase) = finite(angle, phase, rabi);
                let steps = (duration * drive.frequency.abs() * 64.0).ceil().max(1.0) as usize;
                let dt = duration / steps as f64;
                for k in 0..steps {
                    let tm = t + (k as f64 + 0.5) * dt;
                    let v = std::f64::consts::SQRT_2
                        * TWO_PI
                        * rabi
                        * (omega * tm + phase.to_radians()).cos();
                    let h = CMatrix::from_fn(3, 3, |a, b| {
                        sx[(a, b)] * v * C64::from_polar(1.0, (energy[a] - energy[b]) * tm)
                    });
                    psi = expm_hermitian(&h, dt) * psi;
                }
                t += duration;
            }
        }
    }
    psi[1].norm_sqr()
}

#[test]
fn the_drive_matches_a_linearly_polarised_carrier_in_the_laboratory() {
    // At zero and low field a carrier drives both transitions with one
    // phase: phase-shifted sequences (Hahn, XY-8) test the convention.
    // Rotating-wave errors are about Ω/f ≈ 2·10⁻³ here.
    for (b, transition) in [(0.0, -1), (0.5e-3, -1), (2e-3, 1)] {
        let nv = NvCenter::new(b);
        let mw = Microwave::resonant(&nv, transition, Some(5e6));
        for events in [
            hahn(0.3e-6),
            xy8(1, 0.1e-6),
            vec![pulse(90.0, 0.0), NvEvent::Free(0.2e-6), pulse(90.0, 90.0)],
        ] {
            let lab = laboratory(&nv, &events, &mw);
            let got = nv.signal(&events, &mw).unwrap();
            assert!((got - lab).abs() < 0.01, "{b} {transition}: {got} vs {lab}");
        }
    }
}

#[test]
fn ideal_pulses_are_the_limit_of_selective_finite_ones() {
    // At 1 T the transitions are 56 GHz apart: 30 MHz pulses are
    // selective and short, for both transitions, any angle and phase.
    let nv = NvCenter::new(1.0);
    let events = [
        pulse(90.0, 0.0),
        NvEvent::Free(0.1e-6),
        pulse(-60.0, 135.0),
        NvEvent::Free(0.2e-6),
        pulse(270.0, 200.0),
        NvEvent::Free(0.1e-6),
        pulse(45.0, 90.0),
    ];
    for transition in [-1, 1] {
        let ideal = Microwave::resonant(&nv, transition, None);
        let finite = Microwave {
            rabi: Some(30e6),
            ..ideal
        };
        let rho = nv.initial_state().unwrap();
        let (a, b) = (
            nv.evolve(&rho, &events, &ideal).unwrap(),
            nv.evolve(&rho, &events, &finite).unwrap(),
        );
        let worst = (a - b).iter().fold(0.0f64, |m, z| m.max(z.norm()));
        assert!(worst < 2e-3, "{transition}: {worst}");
    }
}

#[test]
fn noise_during_finite_pulses_matches_finer_pieces() {
    // Correlation time 20 ns, π pulses of 100 ns: the noise changes during
    // a pulse. Against a reference with 128 pieces per pulse.
    let nv = NvCenter::new(0.4);
    let mw = on(&nv, Some(5e6));
    let noise = FieldNoise::OrnsteinUhlenbeck {
        sigma: 1e6,
        correlation_time: 20e-9,
    };
    let events = hahn(0.3e-6);
    let (mean, err) = nv
        .signal_with_noise(&events, &mw, &noise, 20000, 1)
        .unwrap();
    let h0 = nv.hamiltonian(mw.frequency).unwrap();
    let sz = NvCenter::electron()[0].clone();
    let centred = NvCenter::centred(&events, &mw).unwrap();
    let root = Rng::seed_from_u64(77);
    let values: Vec<f64> = crate::parallel::map_tasks(20000, |k| {
        let mut rng = root.fork(k as u64);
        let mut path = NoisePath::new(&noise, &mut rng);
        let mut rho = nv.initial_state().unwrap();
        for event in &centred {
            let u = match *event {
                NvEvent::Free(t) => {
                    let phase = TWO_PI * path.integral(t, &ou_steps(&noise, t), &mut rng);
                    expm_hermitian(&(&h0 + &sz * C64::new(phase / t, 0.0)), t)
                }
                NvEvent::Pulse { angle, phase } => {
                    let (duration, phase) = finite(angle, phase, 5e6);
                    let piece = duration / 128.0;
                    let steps = ou_steps(&noise, piece);
                    let h = &h0 + nv.drive(5e6, phase);
                    let mut u = CMatrix::identity(3, 3);
                    for _ in 0..128 {
                        let b = path.integral(piece, &steps, &mut rng) / piece;
                        u = expm_hermitian(&(&h + &sz * C64::new(TWO_PI * b, 0.0)), piece) * u;
                    }
                    u
                }
            };
            rho = &u * rho * u.adjoint();
        }
        nv.bright(&rho)
    });
    let n = values.len() as f64;
    let want = values.iter().sum::<f64>() / n;
    let spread = (values.iter().map(|x| (x - want).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt();
    assert!(
        (mean - want).abs() < 4.0 * (err * err + spread * spread).sqrt(),
        "{mean} ± {err} vs {want} ± {spread}"
    );
    // The noise matters here: without it the echo returns to one.
    assert!(want < 0.9, "{want}");
}

/// `M` of Taminiau et al. (PRL 109, 137602 (2012), supplement eqs. 11–13):
/// nuclear Hamiltonians `ω_L I_z` (m_s = 0) and `(A + ω_L) I_z + B I_x`
/// (m_s = −1), N π pulses spaced 2τ.
fn taminiau(omega_l: f64, a: f64, b: f64, tau: f64, n: usize) -> f64 {
    let w = ((a + omega_l).powi(2) + b * b).sqrt();
    let (mz, mx) = ((a + omega_l) / w, b / w);
    let (alpha, beta) = (w * tau, omega_l * tau);
    let cos_phi = alpha.cos() * beta.cos() - mz * alpha.sin() * beta.sin();
    let phi = cos_phi.clamp(-1.0, 1.0).acos();
    let one_minus = mx * mx * (1.0 - alpha.cos()) * (1.0 - beta.cos()) / (1.0 + cos_phi);
    1.0 - one_minus * (n as f64 * phi / 2.0).sin().powi(2)
}

#[test]
fn cpmg_follows_taminiau_for_a_weakly_coupled_carbon() {
    // One ¹³C (A_∥ = 60 kHz, A_⊥ = 30 kHz) at 40.3 mT, ¹⁴N present (its
    // static shift is refocused): ideal CPMG on the 0 ↔ −1 transition.
    let (b, a_par, a_perp) = (0.0403, 60e3, 30e3);
    let nv = NvCenter::new(b)
        .with_nitrogen(Nitrogen::n14())
        .with_carbon(a_par, a_perp);
    let omega_l = -TWO_PI * Isotope::C13.gamma() / TWO_PI * b;
    let (a, bb) = (-TWO_PI * a_par, -TWO_PI * a_perp);
    let mw = on(&nv, None);
    let mut worst = 0.0f64;
    let taus: Vec<f64> = (0..1360).map(|k| 0.4e-6 + 0.005e-6 * k as f64).collect();
    let mut values = Vec::new();
    for &tau in &taus {
        let p0 = nv.signal(&cpmg(64, tau), &mw).unwrap();
        let want = (1.0 + taminiau(omega_l, a, bb, tau, 64)) / 2.0;
        worst = worst.max((p0 - want).abs());
        values.push(p0);
    }
    assert!(worst < 1e-10, "{worst}");
    // At high field the resonances lie near τ_k = (2k − 1)π/(2ω_L + A)
    // (ω_L and A of one sign here): each has a deep dip within 1.5 %.
    for k in 1..=6 {
        let want = (2 * k - 1) as f64 * PI / (2.0 * omega_l.abs() + TWO_PI * a_par);
        let (at, low) = taus
            .iter()
            .zip(&values)
            .filter(|(t, _)| (**t / want - 1.0).abs() < 0.03)
            .min_by(|x, y| x.1.total_cmp(y.1))
            .map(|(t, v)| (*t, *v))
            .unwrap();
        assert!(
            (at / want - 1.0).abs() < 0.015 && low < 0.55,
            "{k}: {at} ({low}) vs {want}"
        );
    }
    // Finite pulses of 20 MHz (free periods between pulse centres): the
    // ±2.14 MHz ¹⁴N detuning tilts each π pulse by ~6°. XY-8 compensates
    // that for every state, so its first three resonances stay at the same
    // τ within 10 ns and as deep within 0.03; CPMG compensates it only for
    // coherence along its pulse axis, not for the electron–carbon terms the
    // resonance builds up, and loses most of the contrast (Gullion et al.,
    // J. Magn. Reson. 89, 479 (1990)).
    let finite = on(&nv, Some(20e6));
    for k in 1..=3 {
        let want = (2 * k - 1) as f64 * PI / (2.0 * omega_l.abs() + TWO_PI * a_par);
        let lowest = |events: &dyn Fn(f64) -> Vec<NvEvent>, drive: &Microwave| {
            taus.iter()
                .copied()
                .filter(|t| (t / want - 1.0).abs() < 0.03)
                .map(|t| (t, nv.signal(&events(t), drive).unwrap()))
                .min_by(|x, y| x.1.total_cmp(&y.1))
                .unwrap()
        };
        let ideal = lowest(&|t| cpmg(64, t), &mw);
        let (x, c) = (
            lowest(&|t| xy8(8, t), &finite),
            lowest(&|t| cpmg(64, t), &finite),
        );
        assert!(
            (ideal.0 - x.0).abs() <= 10e-9 && (ideal.1 - x.1).abs() < 0.03,
            "{k}: {ideal:?} vs {x:?}"
        );
        assert!(c.1 > ideal.1 + 0.3, "{k}: {ideal:?} vs {c:?}");
    }
    // Without the carbon both keep the electron coherence.
    let bare = NvCenter::new(b).with_nitrogen(Nitrogen::n14());
    let drive = on(&bare, Some(20e6));
    for events in [cpmg(64, 0.54e-6), xy8(8, 0.54e-6)] {
        let p0 = bare.signal(&events, &drive).unwrap();
        assert!(p0 > 0.99, "{p0}");
    }
}

/// Kubo–Anderson: `⟨e^{iν∫s}⟩` of a symmetric telegraph process of flip
/// rate `γ` over `t` (as in the tests of 1.30).
fn telegraph_free(nu: f64, gamma: f64, t: f64) -> f64 {
    let mu2 = gamma * gamma - nu * nu;
    let (c, s) = if mu2 > 0.0 {
        let mu = mu2.sqrt();
        ((mu * t).cosh(), gamma * (mu * t).sinh() / mu)
    } else {
        let w = (-mu2).sqrt();
        ((w * t).cos(), gamma * (w * t).sin() / w)
    };
    (-gamma * t).exp() * (c + s)
}

/// The Hahn echo of the same process over a total time `t`.
fn telegraph_echo(nu: f64, gamma: f64, t: f64) -> f64 {
    use num_complex::Complex64 as Z;
    let half = t / 2.0;
    let step = |f: [Z; 2], sign: f64| -> [Z; 2] {
        let mu = Z::new(gamma * gamma - nu * nu, 0.0).sqrt();
        let (ch, sh) = ((mu * half).cosh(), (mu * half).sinh());
        let k = if mu.norm() < 1e-14 {
            Z::new(half, 0.0)
        } else {
            sh / mu
        };
        let i = Z::new(0.0, nu * sign);
        let b = [[i, Z::new(gamma, 0.0)], [Z::new(gamma, 0.0), -i]];
        let decay = (-gamma * half).exp();
        let mut out = [Z::new(0.0, 0.0); 2];
        for r in 0..2 {
            out[r] = ch * f[r];
            for c in 0..2 {
                out[r] += k * b[r][c] * f[c];
            }
            out[r] *= decay;
        }
        out
    };
    let f = step([Z::new(0.5, 0.0); 2], 1.0);
    let f = step(f, -1.0);
    (f[0] + f[1]).re
}

#[test]
fn gaussian_noise_follows_the_filter_function() {
    // Ornstein–Uhlenbeck noise of 30 kHz and 3 μs on a bare centre:
    // exact trajectories against e^{−χ}, and χ against the frequency
    // integral of the spectrum times the filter function.
    let nv = NvCenter::new(0.02);
    let mw = on(&nv, None);
    let noise = FieldNoise::OrnsteinUhlenbeck {
        sigma: 30e3,
        correlation_time: 3e-6,
    };
    for events in [ramsey(2e-6), hahn(4e-6), cpmg(4, 2e-6), xy8(1, 1e-6)] {
        let chi = decay_exponent(&events, &noise).unwrap();
        // ∫ S |Y|² dω/(2π) by the trapezoid rule on ω = tan(u)/τ_c.
        let n = 200_000;
        let mut integral = 0.0;
        for k in 1..n {
            let u = -PI / 2.0 + PI * k as f64 / n as f64;
            let omega = u.tan() / 3e-6;
            let jac = (1.0 + u.tan().powi(2)) / 3e-6 * PI / n as f64;
            integral += noise.spectrum(omega) * filter_function(&events, omega).unwrap() * jac;
        }
        let from_spectrum = 0.5 * TWO_PI * TWO_PI * integral / TWO_PI;
        assert!(
            (chi / from_spectrum - 1.0).abs() < 1e-5,
            "{chi} vs {from_spectrum}"
        );
        // Both transitions: the noise couples through S_z, m = ∓1 alike.
        for (mw, seed) in [(mw, 3), (Microwave::resonant(&nv, 1, None), 4)] {
            let (mean, err) = nv
                .signal_with_noise(&events, &mw, &noise, 20000, seed)
                .unwrap();
            let want = (1.0 + (-chi).exp()) / 2.0;
            assert!(
                (mean - want).abs() < 4.0 * err,
                "{}: {mean} ± {err} vs {want}",
                mw.transition
            );
        }
        assert!(chi > 0.05, "{chi}");
    }
    // Ramsey in closed form: χ = (2π)²σ²τ_c²(T/τ_c − 1 + e^{−T/τ_c}).
    let chi = decay_exponent(&ramsey(2e-6), &noise).unwrap();
    let x: f64 = 2e-6 / 3e-6;
    let want = TWO_PI * TWO_PI * 30e3f64.powi(2) * 9e-12 * (x - 1.0 + (-x).exp());
    assert!((chi / want - 1.0).abs() < 1e-12, "{chi} vs {want}");
}

#[test]
fn telegraph_noise_follows_kubo_and_anderson() {
    // Fast and slow telegraph noise: exact trajectories against the
    // Kubo–Anderson free decay and echo; the Gaussian e^{−χ} fails for
    // slow noise.
    let nv = NvCenter::new(0.02);
    let mw = on(&nv, None);
    for (amplitude, rate) in [(50e3, 2e6), (100e3, 1e5)] {
        let noise = FieldNoise::Telegraph { amplitude, rate };
        let nu = TWO_PI * amplitude;
        for (events, total, echo) in [(ramsey(3e-6), 3e-6, false), (hahn(2e-6), 4e-6, true)] {
            let want = if echo {
                telegraph_echo(nu, rate, total)
            } else {
                telegraph_free(nu, rate, total)
            };
            let (mean, err) = nv
                .signal_with_noise(&events, &mw, &noise, 20000, 5)
                .unwrap();
            let want = (1.0 + want) / 2.0;
            assert!(
                (mean - want).abs() < 4.0 * err,
                "{rate} {echo}: {mean} ± {err} vs {want}"
            );
        }
    }
    let slow = FieldNoise::Telegraph {
        amplitude: 100e3,
        rate: 1e5,
    };
    let gaussian = (1.0 + (-decay_exponent(&ramsey(3e-6), &slow).unwrap()).exp()) / 2.0;
    let exact = (1.0 + telegraph_free(TWO_PI * 100e3, 1e5, 3e-6)) / 2.0;
    assert!((gaussian - exact).abs() > 0.05, "{gaussian} vs {exact}");
}

#[test]
fn pink_noise_has_a_one_over_f_spectrum_and_gaussian_decay() {
    let noise = FieldNoise::Pink {
        sigma: 40e3,
        lowest: 1e3,
        highest: 1e7,
        fluctuators: 40,
    };
    let (w1, w2) = (TWO_PI * 1e4, TWO_PI * 1e6);
    let slope = (noise.spectrum(w2) / noise.spectrum(w1)).ln() / (w2 / w1).ln();
    assert!((slope + 1.0).abs() < 0.05, "{slope}");
    let nv = NvCenter::new(0.02);
    let mw = on(&nv, None);
    for events in [hahn(3e-6), cpmg(8, 0.5e-6)] {
        let chi = decay_exponent(&events, &noise).unwrap();
        let (mean, err) = nv
            .signal_with_noise(&events, &mw, &noise, 20000, 9)
            .unwrap();
        let want = (1.0 + (-chi).exp()) / 2.0;
        assert!((mean - want).abs() < 4.0 * err, "{mean} ± {err} vs {want}");
    }
}

#[test]
fn nv_input_is_checked() {
    let nv = NvCenter::new(0.01);
    let mut mw = on(&nv, None);
    mw.transition = 0;
    assert!(nv.signal(&ramsey(1e-6), &mw).is_err());
    let mw = on(&nv, Some(-1.0));
    assert!(nv.signal(&ramsey(1e-6), &mw).is_err());
    assert!(nv.signal(&[NvEvent::Free(-1.0)], &on(&nv, None)).is_err());
    let mut big = NvCenter::new(0.01).with_nitrogen(Nitrogen::n14());
    for _ in 0..7 {
        big = big.with_carbon(1e4, 1e3);
    }
    assert!(big.lines().is_err());
    assert!(filter_function(&[NvEvent::Free(1e-6)], 1.0).is_err());
    let bad = FieldNoise::OrnsteinUhlenbeck {
        sigma: 1.0,
        correlation_time: 0.0,
    };
    assert!(decay_exponent(&hahn(1e-6), &bad).is_err());
    let nv = NvCenter::new(0.01)
        .with_nitrogen(Nitrogen::n15())
        .with_carbon(1e4, 2e3);
    assert_eq!(NvCenter::from_json(&nv.to_json()).unwrap(), nv);
    assert!(NvCenter::from_json(&big.to_json()).is_err());
    assert!(NvCenter::from_json("{\"field\": 1}").is_err());
    // The dimension saturates instead of wrapping: 64 carbons are refused.
    let mut huge = NvCenter::new(0.01);
    for _ in 0..64 {
        huge = huge.with_carbon(1e3, 0.0);
    }
    assert_eq!(huge.dimension(), usize::MAX);
    assert!(huge.validate().is_err() && huge.initial_state().is_err());
    assert!(huge.signal(&ramsey(1e-6), &on(&huge, None)).is_err());
    // Filter functions take angles modulo 360° and check the periods.
    let noise = FieldNoise::OrnsteinUhlenbeck {
        sigma: 1e5,
        correlation_time: 1e-6,
    };
    let odd = [
        pulse(-90.0, 0.0),
        NvEvent::Free(1e-6),
        pulse(540.0, 90.0),
        NvEvent::Free(0.5e-6),
        pulse(360.0, 0.0),
        NvEvent::Free(0.5e-6),
        pulse(270.0, 0.0),
    ];
    let (a, b) = (
        decay_exponent(&odd, &noise).unwrap(),
        decay_exponent(&hahn(1e-6), &noise).unwrap(),
    );
    assert!((a - b).abs() < 1e-15 * b, "{a} vs {b}");
    for bad in [-2e-6, f64::NAN] {
        let events = [pulse(90.0, 0.0), NvEvent::Free(bad), pulse(90.0, 0.0)];
        assert!(decay_exponent(&events, &noise).is_err());
        assert!(filter_function(&events, 1.0).is_err());
    }
    assert!(
        decay_exponent(
            &[pulse(90.0, 0.0), pulse(100.0, 0.0), pulse(90.0, 0.0)],
            &noise
        )
        .is_err()
    );
    // Static telegraph noise: a delta spectrum, and χ = (2πaτ)²/2 for
    // Ramsey however slow the flips (no underflow).
    let frozen = FieldNoise::Telegraph {
        amplitude: 1e5,
        rate: 0.0,
    };
    assert!(frozen.spectrum(0.0).is_infinite() && frozen.spectrum(1.0) == 0.0);
    for rate in [0.0, 1e-200] {
        let noise = FieldNoise::Telegraph {
            amplitude: 1e5,
            rate,
        };
        let chi = decay_exponent(&ramsey(1e-6), &noise).unwrap();
        let want = 0.5 * (TWO_PI * 1e5 * 1e-6f64).powi(2);
        assert!((chi - want).abs() < 1e-12 * want, "{chi}");
        assert!(decay_exponent(&hahn(1e-6), &noise).unwrap().abs() < 1e-12);
    }
    // Adjacent free periods merge before pulses are centred.
    let finite = on(&nv, Some(10e6));
    let split = [
        pulse(90.0, 0.0),
        NvEvent::Free(0.0),
        NvEvent::Free(1e-6),
        pulse(90.0, 180.0),
    ];
    let (a, b) = (
        nv.signal(&split, &finite).unwrap(),
        nv.signal(&ramsey(1e-6), &finite).unwrap(),
    );
    assert!((a - b).abs() < 1e-14, "{a} vs {b}");
    // One trajectory has no standard error.
    let (_, err) = nv
        .signal_with_noise(&ramsey(1e-6), &on(&nv, None), &noise, 1, 0)
        .unwrap();
    assert!(err.is_nan());
    // A finite pulse of negative angle is the positive one about the
    // opposite axis (forward in time), also under noise.
    let finite = on(&nv, Some(10e6));
    let noise = FieldNoise::OrnsteinUhlenbeck {
        sigma: 1e5,
        correlation_time: 1e-6,
    };
    let minus = [pulse(-60.0, 30.0), NvEvent::Free(1e-6), pulse(90.0, 0.0)];
    let plus = [pulse(60.0, 210.0), NvEvent::Free(1e-6), pulse(90.0, 0.0)];
    let (a, b) = (
        nv.signal(&minus, &finite).unwrap(),
        nv.signal(&plus, &finite).unwrap(),
    );
    assert!((a - b).abs() < 1e-12 && (a - 1.0).abs() > 0.1, "{a} vs {b}");
    let (a, b) = (
        nv.signal_with_noise(&minus, &finite, &noise, 50, 3)
            .unwrap(),
        nv.signal_with_noise(&plus, &finite, &noise, 50, 3).unwrap(),
    );
    assert!((a.0 - b.0).abs() < 1e-12, "{a:?} vs {b:?}");
    // Pulses of zero angle do nothing, also under noise.
    let zero = [pulse(0.0, 0.0), NvEvent::Free(1e-6)];
    let (p, _) = nv.signal_with_noise(&zero, &finite, &noise, 5, 1).unwrap();
    assert!((p - 1.0).abs() < 1e-12, "{p}");
}
