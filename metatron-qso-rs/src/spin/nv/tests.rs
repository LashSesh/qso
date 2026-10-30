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
            let start = nv.initial_state();
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
        let (mean, err) = nv.signal_with_noise(&events, &mw, &noise, 4000, 3).unwrap();
        let want = (1.0 + (-chi).exp()) / 2.0;
        assert!(
            (mean - want).abs() < 4.0 * err + 1e-3,
            "{mean} ± {err} vs {want}"
        );
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
            let (mean, err) = nv.signal_with_noise(&events, &mw, &noise, 4000, 5).unwrap();
            let want = (1.0 + want) / 2.0;
            assert!(
                (mean - want).abs() < 4.0 * err + 1e-3,
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
        let (mean, err) = nv.signal_with_noise(&events, &mw, &noise, 4000, 9).unwrap();
        let want = (1.0 + (-chi).exp()) / 2.0;
        assert!(
            (mean - want).abs() < 4.0 * err + 1e-3,
            "{mean} ± {err} vs {want}"
        );
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
