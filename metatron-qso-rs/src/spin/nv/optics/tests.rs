use super::*;
use crate::rng::Rng;
use crate::spin::Isotope;
use crate::spin::nv::NvCenter;

fn quiet(field: [f64; 3]) -> OpticalNv {
    let mut nv = OpticalNv::new(field);
    nv.spin_lattice = 0.0;
    nv.dephasing = 0.0;
    nv
}

#[test]
fn the_steady_state_solves_the_rate_equations() {
    // Along the axis and without a microwave the populations obey rate
    // equations with a closed-form steady state (±1 alike):
    //   e₀ = p g₀/(k_r + k₄₇),  e₁ = p g₁/(k_r + k₅₇),
    //   s = p k₅₇ g₁/((k_r + k₅₇) k₇₂),  g₀ = k₇₁ (k_r + k₄₇) s/(p k₄₇).
    let r = Photophysics::tetienne();
    for pump in [1e5, 5e6, 2e8] {
        let nv = quiet([0.0, 0.0, 0.01]);
        let rho = nv.steady_state(pump, None).unwrap();
        let (kr, k47, k57, k71, k72) = (
            r.radiative,
            r.isc_zero,
            r.isc_plus_minus,
            r.singlet_zero,
            r.singlet_plus_minus,
        );
        let g1 = 1.0;
        let s = pump * k57 * g1 / ((kr + k57) * k72);
        let g0 = k71 * (kr + k47) * s / (pump * k47);
        let (e0, e1) = (pump * g0 / (kr + k47), pump * g1 / (kr + k57));
        let total = g0 + 2.0 * g1 + e0 + 2.0 * e1 + s;
        // Ground ascending at 10 mT: 0, −1, +1; excited likewise.
        let want = [g0, g1, g1, e0, e1, e1, s].map(|x| x / total);
        for (k, w) in want.iter().enumerate() {
            assert!(
                (rho[(k, k)].re - w).abs() < 1e-12,
                "pump {pump}, level {k}: {} vs {w}",
                rho[(k, k)].re
            );
        }
        let off: f64 = (0..7)
            .flat_map(|a| (0..7).map(move |b| (a, b)))
            .filter(|(a, b)| a != b)
            .map(|(a, b)| rho[(a, b)].norm())
            .fold(0.0, f64::max);
        assert!(off < 1e-14);
        // After the laser: the excited levels and the singlet decay with
        // their branching ratios; the polarisation does not depend on the
        // pump.
        let to_zero = k71 / (k71 + 2.0 * k72);
        let polarised = (g0
            + e0 * (kr + k47 * to_zero) / (kr + k47)
            + 2.0 * e1 * k57 * to_zero / (kr + k57)
            + s * to_zero)
            / total;
        let p = nv.polarisation(pump, f64::INFINITY).unwrap();
        assert!((p - polarised).abs() < 1e-10, "{p} vs {polarised}");
        // From the unpolarised state a long pulse reaches it.
        assert!((nv.polarisation(pump, 1e-3).unwrap() - p).abs() < 1e-6);
    }
}

/// Eq. (17) of Dréau et al.: the steady-state population of |1⟩.
fn dreau_population(m: &TwoLevelOdmr, detuning: f64, rabi: f64) -> f64 {
    let (g1, gp) = (m.spin_lattice, m.polarisation);
    let g2 = m.dephasing + m.coherence;
    let a = 2.0 * g1 + gp;
    let num = g1 / a * (detuning * detuning + g2 * g2) + g2 * rabi * rabi / (2.0 * a);
    num / (detuning * detuning + g2 * g2 + g2 * rabi * rabi / a)
}

#[test]
fn the_reduced_model_follows_dreau_et_al() {
    // At saturation s = 1 with the rates of Dréau et al.: Γ_p = Γ_p^∞/2,
    // Γ_c = Γ_c^∞/2, Γ₁ = 10³, Γ₂* = 2·10⁵ s⁻¹, Θ = (α − β)/2α = 0.2.
    let m = TwoLevelOdmr {
        polarisation: 2.5e6,
        coherence: 4e7,
        spin_lattice: 1e3,
        dephasing: 2e5,
        bright: 2.5e5,
        dark: 1.5e5,
    };
    for rabi in [1e6, 2.5e6, 2e7] {
        for detuning in [0.0, 1e6, -3e7, 2e8] {
            let p = m.population(detuning, rabi).unwrap();
            let want = dreau_population(&m, detuning, rabi);
            assert!((p - want).abs() < 1e-12, "{p} vs {want}");
        }
        let g2 = m.dephasing + m.coherence;
        let a = 2.0 * m.spin_lattice + m.polarisation;
        let width = (g2 * g2 + rabi * rabi * g2 / a).sqrt() / std::f64::consts::PI;
        let line = m.line(20.0 * TWO_PI * width, 8001, rabi).unwrap();
        // Contrast (their appendix) and full width (Hz) of eq. (17).
        let (al, be) = (m.bright, m.dark);
        let contrast = 0.5 * (al - be) * m.polarisation
            / ((al + be) * m.spin_lattice + al * m.polarisation)
            * rabi
            * rabi
            / (rabi * rabi + g2 * a);
        assert!(
            (line.contrast / contrast - 1.0).abs() < 1e-9,
            "{} vs {contrast}",
            line.contrast
        );
        assert!(
            (line.linewidth / width - 1.0).abs() < 1e-5,
            "{} vs {width}",
            line.linewidth
        );
        // The shot-noise limit of a Lorentzian, 4/(3√3) Δν/(γ C √R).
        let eta = 4.0 / (3.0 * 3f64.sqrt()) * width
            / (NV_GYROMAGNETIC_RATIO * contrast * line.off_resonance.sqrt());
        assert!(
            (line.sensitivity / eta - 1.0).abs() < 1e-4,
            "{} vs {eta}",
            line.sensitivity
        );
    }
}

#[test]
fn the_seven_level_line_is_a_power_broadened_lorentzian() {
    let nv = OpticalNv::new([0.0, 0.0, 0.005]);
    let [(f, strength), _] = nv.lines().unwrap();
    assert!((strength - 0.5).abs() < 1e-12);
    let pump = 5e6;
    let mut widths = Vec::new();
    for rabi in [0.2e6, 0.5e6, 1.0e6] {
        let line = nv.cw_line(f, 20e6, 801, rabi, pump).unwrap();
        // The dip sits on the line.
        let (k, _) = line
            .rates
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        assert!((line.frequencies[k] - f).abs() <= 25e3 + 1.0);
        assert!(
            line.contrast > 0.01 && line.contrast < 0.35,
            "{}",
            line.contrast
        );
        // A Lorentzian: the slope and the width give the same sensitivity.
        let eta = 4.0 / (3.0 * 3f64.sqrt()) * line.linewidth
            / (NV_GYROMAGNETIC_RATIO * line.contrast * line.off_resonance.sqrt());
        assert!(
            (line.sensitivity / eta - 1.0).abs() < 0.02,
            "{} vs {eta}",
            line.sensitivity
        );
        widths.push(line.linewidth);
    }
    assert!(widths[0] < widths[1] && widths[1] < widths[2], "{widths:?}");
}

/// The roots of λ³ − 2Dλ² + (D² − b²)λ + D b_⊥² = 0, trigonometrically.
fn cubic_roots(d: f64, b: f64, perp: f64) -> [f64; 3] {
    let (a2, a1, a0) = (-2.0 * d, d * d - b * b, d * perp * perp);
    let p = a1 - a2 * a2 / 3.0;
    let q = 2.0 * a2.powi(3) / 27.0 - a2 * a1 / 3.0 + a0;
    let m = 2.0 * (-p / 3.0).sqrt();
    let theta = (3.0 * q / (p * m)).clamp(-1.0, 1.0).acos() / 3.0;
    let mut r: [f64; 3] =
        std::array::from_fn(|k| m * (theta - TWO_PI * k as f64 / 3.0).cos() - a2 / 3.0);
    r.sort_by(f64::total_cmp);
    r
}

#[test]
fn levels_in_any_field_are_the_roots_of_the_cubic() {
    let mut rng = Rng::seed_from_u64(4);
    for _ in 0..200 {
        let size = 10f64.powf(rng.uniform_range(-4.0, -0.7));
        let v = [rng.normal(), rng.normal(), rng.normal()];
        let n = norm3(v);
        let field = v.map(|x| x / n * size);
        let nv = OpticalNv::new(field);
        for (levels, d) in [
            (nv.ground_levels().unwrap().0, nv.zero_field_splitting),
            (nv.excited_levels().unwrap().0, nv.excited_splitting),
        ] {
            let g = nv.gyromagnetic_ratio;
            let roots = cubic_roots(d, g * size, g * field[0].hypot(field[1]));
            for (a, b) in levels.iter().zip(roots) {
                assert!(
                    (a - b).abs() < 1e-12 * d.max(a.abs()),
                    "{a} vs {b} at {field:?}"
                );
            }
        }
    }
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[test]
fn along_the_axis_it_is_the_centre_of_1_37() {
    for bz in [0.0, 0.004, 0.05, 0.15] {
        let optical = OpticalNv::new([0.0, 0.0, bz]);
        let nv = NvCenter::new(bz);
        let ours = optical.ground_hamiltonian() * C64::new(TWO_PI, 0.0);
        let theirs = nv.hamiltonian(0.0).unwrap();
        let diff = ours
            .iter()
            .zip(theirs.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        assert!(diff < 1e-3, "{diff}");
        let lines = optical.lines().unwrap();
        let mut theirs = nv.lines().unwrap();
        theirs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut ours: Vec<(f64, f64)> = lines.to_vec();
        ours.sort_by(|a, b| a.0.total_cmp(&b.0));
        if bz == 0.0 {
            // Degenerate at zero field: one line of strength 1.
            assert_eq!(theirs.len(), 1);
            assert!((ours[0].0 - theirs[0].0).abs() < 1e-3);
            continue;
        }
        for (a, b) in ours.iter().zip(&theirs) {
            assert!(
                (a.0 - b.0).abs() < 1e-3 && (a.1 - b.1).abs() < 1e-12,
                "{a:?} vs {b:?}"
            );
        }
    }
}

#[test]
fn the_field_vector_from_the_four_orientations() {
    let mut rng = Rng::seed_from_u64(9);
    let (d, g) = (NV_ZERO_FIELD_SPLITTING, NV_GYROMAGNETIC_RATIO);
    for _ in 0..50 {
        let size = rng.uniform_range(1e-3, 3e-2);
        let v = [rng.normal(), rng.normal(), rng.normal()];
        let n = norm3(v);
        let field = v.map(|x| x / n * size);
        let lines = ensemble_lines(field, d, g).unwrap();
        // Each orientation: |B_∥| and B_⊥ exactly from its two lines.
        for (k, axis) in nv_orientations().iter().enumerate() {
            let (along, across) = split(field, *axis);
            let (a, c) = axial_and_transverse(lines[k], d, g).unwrap();
            assert!(
                (a - along.abs()).abs() < 1e-11 && (c - across).abs() < 1e-11,
                "{a} {c} vs {along} {across}"
            );
        }
        // In either order within each pair.
        let shuffled: [[f64; 2]; 4] = std::array::from_fn(|k| {
            if k % 2 == 0 {
                [lines[k][1], lines[k][0]]
            } else {
                lines[k]
            }
        });
        let (back, rms) = field_from_lines(shuffled, d, g).unwrap();
        assert!(rms < 1e-3, "{rms}");
        let sign = if field[0] + field[1] + field[2] >= 0.0 {
            1.0
        } else {
            -1.0
        };
        for c in 0..3 {
            assert!(
                (back[c] - sign * field[c]).abs() < 1e-11,
                "{back:?} vs {field:?}"
            );
        }
    }
}

#[test]
fn the_pulsed_readout() {
    // From the state the laser leaves (75 % in m_s = 0 at low power with
    // these rates, less at high power) the bright state fluoresces about
    // 30 % more than the dark one during the first few hundred
    // nanoseconds, until the laser repolarises both; the best window
    // shortens with the power.
    let nv = OpticalNv::new([0.0, 0.0, 0.01]);
    let mut windows = Vec::new();
    for fraction in [0.05, 0.2, 1.0, 3.0] {
        let r = nv
            .pulsed_readout(fraction * nv.rates.radiative, 3e-6, 3000, -1)
            .unwrap();
        assert_eq!(r.bright[0], 0.0);
        let peak = r
            .bright
            .iter()
            .zip(&r.dark)
            .skip(1)
            .map(|(b, d)| 1.0 - d / b)
            .fold(0.0, f64::max);
        assert!((0.25..0.4).contains(&peak), "{peak}");
        let last = r.bright.len() - 1;
        if fraction >= 1.0 {
            assert!(((r.bright[last] - r.dark[last]) / r.bright[last]).abs() < 1e-3);
        }
        let c = r.readout_contrast();
        assert!((0.01..0.05).contains(&c), "{c}");
        windows.push(r.window);
    }
    assert!(windows.windows(2).all(|w| w[1] < w[0]), "{windows:?}");
    assert!((200e-9..500e-9).contains(&windows[3]), "{windows:?}");
    // With Γ₁ = 0 and a weak laser the polarisation tends to the ratio of
    // the closed form, 0.749.
    let p = quiet([0.0, 0.0, 0.01])
        .polarisation(1e4, f64::INFINITY)
        .unwrap();
    assert!((p - 0.749).abs() < 1e-3, "{p}");
}

#[test]
fn the_best_ramsey_time() {
    // |L| = e^{−(τ/T₂*)²}: without overhead the optimum is τ = T₂*/2 and
    // η = e^{1/4} √(2/T₂*)/(2π γ_e C).
    let (t2, c) = (3e-6, 0.03);
    let times: Vec<f64> = (1..=20000).map(|k| k as f64 * 1e-10).collect();
    let coherences: Vec<f64> = times.iter().map(|t| (-(t / t2).powi(2)).exp()).collect();
    let (tau, eta) = best_sensitivity(&times, &coherences, c, 0.0, false).unwrap();
    assert!((readout_contrast(0.03, 0.02).unwrap() - 1001f64.powf(-0.5)).abs() < 1e-15);
    assert!((tau - t2 / 2.0).abs() < 2e-10);
    let want = 0.25f64.exp() * (2.0 / t2).sqrt() / (TWO_PI * NV_GYROMAGNETIC_RATIO * c);
    assert!((eta / want - 1.0).abs() < 1e-6, "{eta} vs {want}");
    let echo = best_sensitivity(&times, &coherences, c, 0.0, true).unwrap();
    assert!((echo.1 / eta - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
}

#[test]
fn rejects_invalid_input() {
    let nv = OpticalNv::new([0.0, 0.0, 0.01]);
    assert!(nv.steady_state(-1.0, None).is_err());
    assert!(nv.steady_state(1e6, Some((f64::NAN, 1e6))).is_err());
    assert!(nv.cw_line(2.6e9, 0.0, 11, 1e6, 1e6).is_err());
    assert!(nv.pulsed_readout(1e6, 1e-6, 0, -1).is_err());
    assert!(nv.pulsed_readout(1e6, 1e-6, 10, 0).is_err());
    let mut bad = nv.clone();
    bad.collection = 0.0;
    assert!(bad.lines().is_err());
    bad = nv.clone();
    bad.rates.radiative = 0.0;
    assert!(bad.steady_state(1e6, None).is_err());
    assert!(ramsey_sensitivity(1e-6, 0.5, 0.0, 0.0).is_err());
    assert!(ramsey_sensitivity(0.0, 0.5, 0.1, 0.0).is_err());
    assert_eq!(
        ramsey_sensitivity(1e-6, 0.0, 0.1, 0.0).unwrap(),
        f64::INFINITY
    );
    assert!(best_sensitivity(&[1e-6], &[], 0.1, 0.0, false).is_err());
    assert!(best_sensitivity(&[1e-6, 2e-6], &[0.0, 0.0], 0.1, 0.0, false).is_err());
    assert!(ramsey_sensitivity(1e-6, f64::NAN, 0.1, 0.0).is_err());
    assert!(ramsey_sensitivity(1e-6, 5.0, 0.1, 0.0).is_err());
    assert!(readout_contrast(-1.0, 0.0).is_err());
    assert!(axial_and_transverse([f64::NAN, 1.0], 2.87e9, 28e9).is_err());
    assert!(axial_and_transverse([1.0, 2.0], f64::INFINITY, 28e9).is_err());
    assert!(ensemble_lines([0.0, 0.0, 0.001], 2.87e9, f64::INFINITY).is_err());
    // Limits that used to hang, abort or return garbage.
    assert!(nv.cw_line(2.6e9, 1e6, 1_000_000_000, 1e6, 1e6).is_err());
    assert!(nv.cw_line(2.6e9, 1e6, 11, 0.0, 1e6).is_err());
    assert!(nv.pulsed_readout(1e7, 1e300, 1, -1).is_err());
    assert!(nv.polarisation(1e7, 1e300).is_err());
    let rho = nv.steady_state(1e6, None).unwrap();
    assert!(nv.evolve(&rho, 1e10, 1e6, None).is_err());
    let mut still = nv.clone();
    still.spin_lattice = 0.0;
    assert!(still.steady_state(0.0, Some((2.6e9, 1e6))).is_err());
    let m = TwoLevelOdmr {
        polarisation: 1e6,
        coherence: 1e7,
        spin_lattice: 0.0,
        dephasing: 0.0,
        bright: 1.0,
        dark: 0.5,
    };
    assert!(m.population(f64::NAN, 1e6).is_err());
    assert!(m.line(1e6, 1_000_000_000, 1e6).is_err());
}

#[test]
fn spin_lattice_relaxation_in_the_rate_equations() {
    // Along the axis with Γ₁ the populations still obey rate equations,
    // solved here independently.
    let mut nv = quiet([0.0, 0.0, 0.01]);
    nv.spin_lattice = 5e4;
    let pump = 2e6;
    let rho = nv.steady_state(pump, None).unwrap();
    let r = nv.rates;
    // Levels: ground 0, −1, +1; excited 0, −1, +1; singlet.
    let mut k = nalgebra::DMatrix::<f64>::zeros(7, 7);
    for (g, e) in [(0, 3), (1, 4), (2, 5)] {
        k[(g, e)] = pump;
        k[(e, g)] = r.radiative;
    }
    k[(3, 6)] = r.isc_zero;
    k[(4, 6)] = r.isc_plus_minus;
    k[(5, 6)] = r.isc_plus_minus;
    k[(6, 0)] = r.singlet_zero;
    k[(6, 1)] = r.singlet_plus_minus;
    k[(6, 2)] = r.singlet_plus_minus;
    for (a, b) in [(0, 1), (1, 0), (0, 2), (2, 0)] {
        k[(a, b)] = 5e4;
    }
    let mut a = k.transpose() - nalgebra::DMatrix::from_diagonal(&k.column_sum());
    for c in 0..7 {
        a[(0, c)] = 1.0;
    }
    let mut rhs = nalgebra::DVector::zeros(7);
    rhs[0] = 1.0;
    let want = a.lu().solve(&rhs).unwrap();
    for i in 0..7 {
        assert!(
            (rho[(i, i)].re - want[i]).abs() < 1e-11,
            "{i}: {} vs {}",
            rho[(i, i)].re,
            want[i]
        );
    }
}

#[test]
fn a_pi_pulse_matches_the_nv_centre_of_1_37() {
    // The microwave's normalisation and detuning: without the laser and
    // relaxation, a pulse of 1/(2Ω) on resonance equals the finite π pulse
    // of NvCenter (both transitions driven, the other far detuned).
    for (field, transition) in [(0.005, -1i8), (0.02, 1), (0.15, -1)] {
        let optical = quiet([0.0, 0.0, field]);
        let lines = optical.lines().unwrap();
        let f = if transition < 0 {
            lines[0].0
        } else {
            lines[1].0
        };
        let rabi = 1e6;
        let l = optical.levels().unwrap();
        let mut rho = CMatrix::zeros(7, 7);
        rho[(l.zero, l.zero)] = C64::new(1.0, 0.0);
        let nv = NvCenter::new(field);
        let drive = crate::spin::nv::Microwave::resonant(&nv, transition, Some(rabi));
        for angle in [90.0, 180.0, 270.0] {
            let out = optical
                .evolve(&rho, angle / 360.0 / rabi, 0.0, Some((f, rabi)))
                .unwrap();
            let theirs = nv
                .signal(
                    &[crate::spin::nv::NvEvent::Pulse { angle, phase: 0.0 }],
                    &drive,
                )
                .unwrap();
            assert!(
                (out[(l.zero, l.zero)].re - theirs).abs() < 1e-9,
                "{field} {angle}: {} vs {theirs}",
                out[(l.zero, l.zero)].re
            );
        }
    }
}

#[test]
fn the_width_at_low_power_is_set_by_the_dephasing() {
    // With a weak laser and microwave the line is a Lorentzian of full
    // width Γ₂/π, Γ₂ = Γ₂* + 3Γ₁/2 + pump (the jumps out of both levels).
    let nv = OpticalNv::new([0.0, 0.0, 0.005]);
    let pump = 1e3;
    let gamma2 = nv.dephasing + 1.5 * nv.spin_lattice + pump;
    let width = gamma2 / std::f64::consts::PI;
    let [(f, _), _] = nv.lines().unwrap();
    let line = nv.cw_line(f, 20.0 * width, 4001, 200.0, pump).unwrap();
    assert!(
        (line.linewidth / width - 1.0).abs() < 0.01,
        "{} vs {width}",
        line.linewidth
    );
}

#[test]
fn lines_across_the_axis_are_distinct() {
    // A field across the axis: the levels closest to ±1 are (|+1⟩ ± |−1⟩)/√2
    // mixtures; S_x drives one of them only.
    let nv = OpticalNv::new([0.01, 0.0, 0.0]);
    let (e, _) = nv.ground_levels().unwrap();
    let lines = nv.lines().unwrap();
    assert!((lines[0].0 - lines[1].0).abs() > 1e6, "{lines:?}");
    let mut gaps = [e[1] - e[0], e[2] - e[0]];
    gaps.sort_by(f64::total_cmp);
    let mut ours = [lines[0].0, lines[1].0];
    ours.sort_by(f64::total_cmp);
    assert!((ours[0] - gaps[0]).abs() < 1e-3 && (ours[1] - gaps[1]).abs() < 1e-3);
    let strengths = [lines[0].1, lines[1].1];
    assert!(
        strengths.iter().any(|s| *s > 0.4) && strengths.iter().any(|s| *s < 1e-6),
        "{strengths:?}"
    );
    let a = nv.pulsed_readout(65.9e6, 2e-6, 2000, -1).unwrap();
    let b = nv.pulsed_readout(65.9e6, 2e-6, 2000, 1).unwrap();
    assert!(a.dark != b.dark);
}

#[test]
fn degenerate_fields_are_continuous() {
    // The dephasing acts on S_z, not on the eigenbasis, so a nanotesla
    // across the axis changes nothing at zero field, nor at the excited
    // state's anticrossing near 51 mT.
    let width = |field: [f64; 3]| {
        let nv = OpticalNv::new(field);
        let [(f, _), _] = nv.lines().unwrap();
        nv.cw_line(f, 2e6, 801, 2e4, 1e5).unwrap().linewidth
    };
    let w0 = width([0.0, 0.0, 0.0]);
    for field in [[1e-9, 0.0, 0.0], [0.0, 0.0, 1e-9], [0.0, 1e-9, 1e-9]] {
        assert!(
            (width(field) / w0 - 1.0).abs() < 1e-3,
            "{field:?}: {} vs {w0}",
            width(field)
        );
    }
    let crossing = NV_EXCITED_SPLITTING / NV_GYROMAGNETIC_RATIO;
    let glow = |field: [f64; 3]| {
        let nv = OpticalNv::new(field);
        (
            nv.fluorescence(&nv.steady_state(5e6, None).unwrap()),
            nv.polarisation(5e6, f64::INFINITY).unwrap(),
        )
    };
    let (a, pa) = glow([0.0, 0.0, crossing]);
    let (b, pb) = glow([1e-9, 0.0, crossing]);
    assert!(
        (a / b - 1.0).abs() < 1e-6 && (pa - pb).abs() < 1e-6,
        "{a} {b} {pa} {pb}"
    );
}

#[test]
fn split_levels_follow_the_rates_of_tetienne_et_al() {
    // Away from anticrossings the eigenbasis rate equations of Tetienne et
    // al., k_ij = Σ_pq |⟨i|p⟩|²|⟨j|q⟩|² k⁰_pq, give the populations (to the
    // ratio of rates to splittings).
    let b = 0.02;
    let theta = 30f64.to_radians();
    let mut nv = quiet([b * theta.sin(), 0.0, b * theta.cos()]);
    nv.spin_lattice = 0.0;
    let pump = 5e6;
    let rho = nv.steady_state(pump, None).unwrap();
    let (_, vg) = nv.ground_levels().unwrap();
    let (_, ve) = nv.excited_levels().unwrap();
    let (wg, we) = (weights(&vg), weights(&ve));
    let r = nv.rates;
    let isc = [r.isc_plus_minus, r.isc_zero, r.isc_plus_minus];
    let back = [r.singlet_plus_minus, r.singlet_zero, r.singlet_plus_minus];
    let mut k = nalgebra::DMatrix::<f64>::zeros(7, 7);
    for i in 0..3 {
        for j in 0..3 {
            let o: f64 = (0..3).map(|p| wg[i][p] * we[j][p]).sum();
            k[(i, 3 + j)] = pump * o;
            k[(3 + j, i)] = r.radiative * o;
        }
        k[(3 + i, 6)] = (0..3).map(|p| we[i][p] * isc[p]).sum();
        k[(6, i)] = (0..3).map(|p| wg[i][p] * back[p]).sum();
    }
    let mut a = k.transpose() - nalgebra::DMatrix::from_diagonal(&k.column_sum());
    for c in 0..7 {
        a[(0, c)] = 1.0;
    }
    let mut rhs = nalgebra::DVector::zeros(7);
    rhs[0] = 1.0;
    let want = a.lu().solve(&rhs).unwrap();
    for i in 0..7 {
        assert!(
            (rho[(i, i)].re - want[i]).abs() < 1e-3,
            "{i}: {} vs {}",
            rho[(i, i)].re,
            want[i]
        );
    }
}

#[test]
fn the_shift_of_a_tilted_line() {
    // The sensitivity counts how fast the line moves with a field along the
    // axis: γ_e along the axis, less when the field is tilted.
    for field in [[0.0, 0.0, 0.005], [0.03, 0.0, 0.005], [0.01, 0.0, 1e-4]] {
        let nv = OpticalNv::new(field);
        let [(f, _), _] = nv.lines().unwrap();
        let line = nv.cw_line(f, 20e6, 101, 1e6, 5e6).unwrap();
        let h = 1e-7;
        let moved = |dz: f64| {
            OpticalNv::new([field[0], field[1], field[2] + dz])
                .lines()
                .unwrap()[0]
                .0
        };
        let numeric = (moved(h) - moved(-h)) / (2.0 * h);
        assert!(
            (line.shift / numeric - 1.0).abs() < 1e-4,
            "{field:?}: {} vs {numeric}",
            line.shift
        );
    }
}

#[test]
fn ensembles_in_awkward_fields() {
    let (d, g) = (NV_ZERO_FIELD_SPLITTING, NV_GYROMAGNETIC_RATIO);
    // Across two of the axes: [1, −1, 0].
    let field = [0.005 / 2f64.sqrt(), -0.005 / 2f64.sqrt(), 0.0];
    let lines = ensemble_lines(field, d, g).unwrap();
    for pair in &lines {
        assert!(pair[0] != pair[1]);
    }
    let (back, rms) = field_from_lines(lines, d, g).unwrap();
    let sign = if field[0] + field[1] + field[2] >= 0.0 {
        1.0
    } else {
        -1.0
    };
    assert!(rms < 1e-3);
    // Two axes stand across the field: their lines depend on B·n only
    // quadratically, so the field is fixed to about √ε |B| (10⁻¹⁰ T).
    for c in 0..3 {
        assert!((back[c] - sign * field[c]).abs() < 1e-9, "{back:?}");
    }
    // Along a cube axis all four orientations see the same lines: the
    // result fits them and has the right size, whichever cube axis it is.
    let lines = ensemble_lines([0.004, 0.0, 0.0], d, g).unwrap();
    let (back, rms) = field_from_lines(lines, d, g).unwrap();
    assert!(
        rms < 1e-3 && (norm3(back) - 0.004).abs() < 1e-11,
        "{back:?} {rms}"
    );
}

#[test]
fn transverse_fields_quench_the_fluorescence_and_the_contrast() {
    // Tetienne et al.: a field across the axis mixes the spin states, so
    // the excited ±1 character spreads over all levels; the fluorescence
    // and the ODMR contrast fall with it.
    let pump = 0.2 * Photophysics::tetienne().radiative;
    let mut previous = (f64::INFINITY, f64::INFINITY);
    for across in [0.0, 0.01, 0.03, 0.06] {
        let nv = OpticalNv::new([across, 0.0, 0.005]);
        let rate = nv.fluorescence(&nv.steady_state(pump, None).unwrap());
        let [(f, _), _] = nv.lines().unwrap();
        let line = nv.cw_line(f, 30e6, 301, 1e6, pump).unwrap();
        assert!(
            rate < previous.0 && line.contrast < previous.1,
            "{across}: {rate} {}",
            line.contrast
        );
        previous = (rate, line.contrast);
    }
    assert!(
        previous.1
            < 0.5
                * OpticalNv::new([0.0, 0.0, 0.005])
                    .cw_line(
                        OpticalNv::new([0.0, 0.0, 0.005]).lines().unwrap()[0].0,
                        30e6,
                        301,
                        1e6,
                        pump
                    )
                    .unwrap()
                    .contrast
    );
}

#[test]
fn echo_sensitivity_with_the_bath() {
    // A single centre in natural diamond at 50 mT: the Hahn echo of its
    // ¹³C bath (1.39), the readout of the seven-level model with 1 % of the
    // photons, 1 μs of initialisation: tens of nT/√Hz for AC fields, as
    // reported for single NV centres (Maze et al., Nature 455, 644 (2008)).
    use crate::spin::nv::bath::{Cce, NATURAL_C13_ABUNDANCE, SpinBath, diamond_bath};
    use crate::spin::nv::hahn;
    let field = 0.05;
    let nv = OpticalNv::new([0.0, 0.0, field]);
    let r = nv
        .pulsed_readout(nv.rates.radiative, 2e-6, 2000, -1)
        .unwrap();
    let c = r.readout_contrast();
    let bath = SpinBath::new(
        field,
        -1,
        diamond_bath(NATURAL_C13_ABUNDANCE, 4e-9, 1000).unwrap(),
    );
    let larmor = 1.0 / (Isotope::C13.gamma() / TWO_PI * field);
    let taus: Vec<f64> = (1..=100).map(|k| (5 * k) as f64 * larmor).collect();
    let l = bath
        .coherence(
            &taus.iter().map(|&t| hahn(t)).collect::<Vec<_>>(),
            &Cce::default(),
        )
        .unwrap();
    let totals: Vec<f64> = taus.iter().map(|t| 2.0 * t).collect();
    let coherences: Vec<f64> = l.iter().map(|z| z.re).collect();
    let (t, eta) = best_sensitivity(&totals, &coherences, c, 1e-6 + r.window, true).unwrap();
    assert!((1e-8..6e-8).contains(&eta), "{eta} at {t}");
    assert!((0.2e-3..1.2e-3).contains(&t), "{t}");
}
