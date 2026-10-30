use super::*;
use crate::spin::Spectrum;

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// One proton in two sites with shifts `da`, `db` (ppm) and populations
/// `pa`, `pb`, exchanging A → B at `kab` (B → A at `kab·pa/pb`), with
/// transverse relaxation `t2` in both.
fn two_sites(da: f64, db: f64, pa: f64, pb: f64, kab: f64, t2: Option<f64>) -> Exchange {
    let mut x = Exchange::new();
    for (shift, p) in [(da, pa), (db, pb)] {
        let mut s = SpinSystem::new(9.4);
        s.add_spin(Isotope::H1, shift).unwrap();
        if let Some(t2) = t2 {
            s.set_relaxation(0, Some(10.0), Some(t2)).unwrap();
        }
        x.add_site(s, p).unwrap();
    }
    for (from, to, rate) in [(0, 1, kab), (1, 0, kab * pa / pb)] {
        x.add_channel(Channel {
            from,
            to,
            rate,
            carried: vec![(0, 0)],
            fresh: Fresh::Thermal,
        })
        .unwrap();
    }
    x
}

fn offset(shift: f64) -> f64 {
    -Isotope::H1.gamma() * 9.4 * shift * 1e-6
}

#[test]
fn two_site_exchange_follows_the_bloch_mcconnell_equations() {
    // dm/dt = [[iΩ_A − k_AB − R₂, k_BA], [k_AB, iΩ_B − k_BA − R₂]] m for
    // the transverse magnetisation m_s = c_s Tr(ρ_s F₊) of each site.
    for (pa, kab, t2) in [
        (0.5, 30.0, None),
        (0.7, 400.0, Some(0.05)),
        (0.2, 5.0, Some(0.2)),
    ] {
        let pb = 1.0 - pa;
        let x = two_sites(1.0, 1.4, pa, pb, kab, t2);
        let start = x
            .evolve(
                &x.thermal_state().unwrap(),
                &[Event::pulse(Target::Isotope(Isotope::H1), 90.0, 0.0)],
            )
            .unwrap();
        let dwell = 2e-4;
        let fid = x.fid(&start, Isotope::H1, dwell, 400).unwrap();
        let kba = kab * pa / pb;
        let r2 = t2.map_or(0.0, |t| 1.0 / t);
        let a = nalgebra::Matrix2::new(
            C64::new(-kab - r2, offset(1.0)),
            C64::new(kba, 0.0),
            C64::new(kab, 0.0),
            C64::new(-kba - r2, offset(1.4)),
        );
        let m0 = nalgebra::Vector2::from_fn(|s, _| {
            let f = x.sites()[s].system.fplus(Isotope::H1).unwrap();
            x.sites()[s].system.expectation(&start[s], &f).unwrap() * x.sites()[s].concentration
        });
        let scale = m0.norm();
        for (k, s) in fid.iter().enumerate() {
            let m = (a * C64::new(k as f64 * dwell, 0.0)).exp() * m0;
            let want = m[0] + m[1];
            assert!(
                (s - want).norm() < 1e-10 * scale,
                "{pa} {kab} {k}: {s} vs {want}"
            );
        }
    }
}

#[test]
fn lines_coalesce_at_pi_delta_nu_over_root_two() {
    // Symmetric exchange between sites Δν apart: two lines below
    // k = πΔν/√2, one above.
    let (da, db) = (1.0, 1.25);
    let dnu = (offset(db) - offset(da)).abs() / TWO_PI;
    let kc = std::f64::consts::PI * dnu / 2f64.sqrt();
    let centre = (offset(da) + offset(db)) / 2.0 / TWO_PI;
    for (factor, two_lines) in [(0.85, true), (1.15, false)] {
        let x = two_sites(da, db, 0.5, 0.5, factor * kc, None);
        let start = x
            .evolve(
                &x.thermal_state().unwrap(),
                &[Event::pulse(Target::Isotope(Isotope::H1), 90.0, 0.0)],
            )
            .unwrap();
        let dwell = 1.0 / (40.0 * dnu);
        let fid = x.fid(&start, Isotope::H1, dwell, 1 << 14).unwrap();
        let spec = Spectrum::from_fid(&fid, dwell, 0.0).unwrap();
        // Absorption (−Im, as the signal starts as −iP/2) near the centre
        // and at the sites.
        let value = |f: f64| {
            let (j, _) = spec
                .frequencies
                .iter()
                .enumerate()
                .min_by(|a, b| (a.1 - f).abs().total_cmp(&(b.1 - f).abs()))
                .unwrap();
            -spec.values[j].im
        };
        let middle = value(centre);
        let side = value(centre + 0.35 * dnu).max(value(centre - 0.35 * dnu));
        assert_eq!(
            side > middle,
            two_lines,
            "k = {factor}·k_c: {side} vs {middle}"
        );
    }
}

#[test]
fn exchange_keeps_thermal_equilibrium_and_renews_states() {
    // Exchange between thermal sites of equal shift leaves them thermal
    // (at different shifts their Boltzmann factors differ by parts in 10⁶,
    // and exchange mixes them).
    let x = two_sites(1.0, 1.0, 0.3, 0.7, 50.0, Some(0.1));
    let eq = x.thermal_state().unwrap();
    let late = x.evolve(&eq, &[Event::Delay(0.3)]).unwrap();
    for (a, b) in late.iter().zip(&eq) {
        assert!((a - b).norm() < 1e-14, "{}", (a - b).norm());
    }
    let steady = x.steady_state().unwrap();
    for (a, b) in steady.iter().zip(&eq) {
        assert!((a - b).norm() < 1e-14);
    }
    // A site renewed with fresh thermal molecules at rate k: polarisation
    // and coherence relax as e^{−kt}.
    let mut sys = SpinSystem::new(9.4);
    sys.add_spin(Isotope::C13, 0.0).unwrap();
    let mut renew = Exchange::new();
    renew.add_site(sys.clone(), 1.0).unwrap();
    let k = 3.0;
    renew
        .add_channel(Channel {
            from: 0,
            to: 0,
            rate: k,
            carried: vec![],
            fresh: Fresh::Thermal,
        })
        .unwrap();
    let p0 = sys.polarization(&sys.thermal_state().unwrap(), 0).unwrap();
    let inverted = renew
        .evolve(
            &renew.thermal_state().unwrap(),
            &[Event::pulse(Target::Isotope(Isotope::C13), 180.0, 0.0)],
        )
        .unwrap();
    let tilted = renew
        .evolve(
            &renew.thermal_state().unwrap(),
            &[Event::pulse(Target::Isotope(Isotope::C13), 90.0, 0.0)],
        )
        .unwrap();
    let fp = sys.fplus(Isotope::C13).unwrap();
    let s0 = sys.expectation(&tilted[0], &fp).unwrap();
    for t in [0.1, 0.5, 2.0] {
        let p = renew.polarization(&renew.evolve(&inverted, &[Event::Delay(t)]).unwrap(), 0, 0);
        let want = p0 * (1.0 - 2.0 * (-k * t).exp());
        // The states come back normalised, entries about 1/2: rounding is
        // 10⁻¹⁶ against P ≈ 8·10⁻⁶.
        assert!((p.unwrap() - want).abs() < 1e-10 * p0, "{t}");
        let s = sys
            .expectation(&renew.evolve(&tilted, &[Event::Delay(t)]).unwrap()[0], &fp)
            .unwrap();
        assert!((s - s0 * (-k * t).exp()).norm() < 1e-12 * s0.norm(), "{t}");
    }
}

#[test]
fn slow_renewal_with_parahydrogen_approaches_the_time_average() {
    // Parahydrogen renewed slowly on two protons: the steady state tends
    // to the time average of the singlet, as k → 0 (1.33).
    let mut sys = SpinSystem::new(14.1);
    sys.add_spin(Isotope::H1, 1.0).unwrap();
    sys.add_spin(Isotope::H1, 5.0).unwrap();
    sys.couple(0, 1, 7.0).unwrap();
    let mut x = Exchange::new();
    x.add_site(sys.clone(), 1.0).unwrap();
    x.add_channel(Channel {
        from: 0,
        to: 0,
        rate: 0.05,
        carried: vec![],
        fresh: Fresh::Parahydrogen {
            spins: [0, 1],
            fraction: 1.0,
        },
    })
    .unwrap();
    let steady = x.steady_state().unwrap();
    let average = sys
        .time_average(&sys.parahydrogen_state(0, 1, 1.0).unwrap())
        .unwrap();
    // The remaining coherence oscillates at 2πΔν ≈ 2.4·10⁴ rad/s, so it is
    // k/Δω ≈ 2·10⁻⁶ of the singlet's.
    let err = (&steady[0] - &average).norm();
    assert!(err < 1e-5, "{err}");
    // Populations in the eigenbasis are conserved by free evolution, so
    // in the product basis they differ only by that coherence times the
    // mixing of the states (J/Δν ≈ 10⁻³).
    for i in 0..4 {
        assert!((steady[0][(i, i)] - average[(i, i)]).norm() < 1e-8);
    }
}

/// SABRE with two hydrides and a ¹⁵N substrate in the laboratory frame:
/// complex (H_a, H_b, N) and free substrate (N), bound for `1/kd`, the
/// hydrides renewed from parahydrogen at `kh`.
fn sabre(field: f64, kd: f64, kh: f64) -> Exchange {
    let mut complex = SpinSystem::new(field).with_frame(Frame::Laboratory);
    complex.add_spin(Isotope::H1, -22.0).unwrap();
    complex.add_spin(Isotope::H1, -22.0).unwrap();
    complex.add_spin(Isotope::N15, 250.0).unwrap();
    complex.couple(0, 1, -7.0).unwrap();
    complex.couple(0, 2, -2.0).unwrap();
    let mut free = SpinSystem::new(field).with_frame(Frame::Laboratory);
    free.add_spin(Isotope::N15, 310.0).unwrap();
    free.set_relaxation(0, Some(30.0), None).unwrap();
    let (cc, cf) = (1.0, 10.0);
    let mut x = Exchange::new();
    x.add_site(complex, cc).unwrap();
    x.add_site(free, cf).unwrap();
    let para = Fresh::Parahydrogen {
        spins: [0, 1],
        fraction: 1.0,
    };
    for c in [
        Channel {
            from: 0,
            to: 1,
            rate: kd,
            carried: vec![(2, 0)],
            fresh: Fresh::Thermal,
        },
        Channel {
            from: 1,
            to: 0,
            rate: kd * cc / cf,
            carried: vec![(0, 2)],
            fresh: para,
        },
        Channel {
            from: 0,
            to: 0,
            rate: kh,
            carried: vec![(2, 2)],
            fresh: para,
        },
    ] {
        x.add_channel(c).unwrap();
    }
    x
}

#[test]
fn sabre_polarises_the_free_substrate_at_the_level_anti_crossing() {
    // Anti-crossing |S α_N⟩ ↔ |T₊ β_N⟩: ν_N − ν_H = ±(J_AB − J_AN/4) in
    // the laboratory frame (ν = −γB(1 + δ)/2π), about 0.14 μT.
    let nu = |x: Isotope, shift: f64| -x.larmor(1.0) * (1.0 + shift * 1e-6);
    let slope = nu(Isotope::N15, 250.0) - nu(Isotope::H1, -22.0);
    let target = -7.0 - (-2.0) / 4.0;
    let lac = (target / slope).abs();
    let fields: Vec<f64> = (0..=120).map(|k| lac * (0.4 + k as f64 / 100.0)).collect();
    let polarisation: Vec<f64> = fields
        .iter()
        .map(|&b| {
            let x = sabre(b, 2.0, 0.5);
            x.polarization(&x.steady_state().unwrap(), 1, 0).unwrap()
        })
        .collect();
    let (best, peak) = polarisation
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .unwrap();
    assert!(
        (fields[best] / lac - 1.0).abs() < 0.03,
        "{} vs {lac}",
        fields[best]
    );
    // About 74 % at the anti-crossing, about a third of that at 0.4 and
    // 1.6 times its field (the free substrate keeps what it receives for
    // T₁ = 30 s).
    assert!(peak.abs() > 0.5, "{peak}");
    assert!(polarisation[0].abs() < 0.5 * peak.abs() && polarisation[120].abs() < 0.5 * peak.abs());
    // The polarisation builds up from thermal towards that steady state.
    let x = sabre(lac, 2.0, 0.5);
    let steady = x.polarization(&x.steady_state().unwrap(), 1, 0).unwrap();
    let mut last = 0.0f64;
    for t in [1.0, 5.0, 20.0] {
        let p = x
            .polarization(
                &x.evolve(&x.thermal_state().unwrap(), &[Event::Delay(t)])
                    .unwrap(),
                1,
                0,
            )
            .unwrap();
        assert!(
            p.abs() > last.abs() && p.abs() <= steady.abs() * (1.0 + 1e-9),
            "{t}: {p}"
        );
        last = p;
    }
    let late = x
        .evolve(&x.thermal_state().unwrap(), &[Event::Delay(400.0)])
        .unwrap();
    let p = x.polarization(&late, 1, 0).unwrap();
    assert!((p / steady - 1.0).abs() < 1e-6, "{p} vs {steady}");
}

#[test]
fn steady_states_hold_in_the_laboratory_frame_at_high_field() {
    // Zero-quantum frequencies of the laboratory frame exceed the rates by
    // 10⁸ and more; the steady state must not depend on that scale. Two
    // sites exchanging both spins relax to the thermal state.
    for b in [0.1, 1.0, 9.4] {
        for frame in [Frame::Laboratory, Frame::Rotating] {
            let mut s = SpinSystem::new(b).with_frame(frame);
            s.add_spin(Isotope::H1, 1.0).unwrap();
            s.add_spin(Isotope::C13, 30.0).unwrap();
            s.couple(0, 1, 140.0).unwrap();
            s.set_relaxation(0, Some(2.0), None).unwrap();
            s.set_relaxation(1, Some(5.0), None).unwrap();
            let thermal = s.thermal_state().unwrap();
            let mut x = Exchange::new();
            x.add_site(s.clone(), 1.0).unwrap();
            x.add_site(s.clone(), 3.0).unwrap();
            for (from, to, rate) in [(0, 1, 3.0), (1, 0, 1.0)] {
                x.add_channel(Channel {
                    from,
                    to,
                    rate,
                    carried: vec![(0, 0), (1, 1)],
                    fresh: Fresh::Thermal,
                })
                .unwrap();
            }
            let steady = x.steady_state().unwrap();
            for site in 0..2 {
                for spin in 0..2 {
                    let p = x.polarization(&steady, site, spin).unwrap();
                    let want = s.polarization(&thermal, spin).unwrap();
                    assert!(
                        (p / want - 1.0).abs() < 1e-6,
                        "{b} {frame:?}: {p} vs {want}"
                    );
                }
            }
        }
    }
    // SABRE at 9.4 T: the laboratory frame's steady state is the rotating
    // frame's and stays.
    let lab = sabre(9.4, 2.0, 0.5);
    let steady = lab.steady_state().unwrap();
    let p = lab.polarization(&steady, 1, 0).unwrap();
    let mut rot = lab.clone();
    for site in &mut rot.sites {
        site.system = site.system.clone().with_frame(Frame::Rotating);
    }
    let want = rot
        .polarization(&rot.steady_state().unwrap(), 1, 0)
        .unwrap();
    assert!((p / want - 1.0).abs() < 1e-6, "{p} vs {want}");
    let later = lab.evolve(&steady, &[Event::Delay(1.0)]).unwrap();
    let q = lab.polarization(&later, 1, 0).unwrap();
    assert!((q / p - 1.0).abs() < 1e-6, "{q} vs {p}");
    // Without relaxation of the free substrate nothing is unique.
    let mut none = sabre(9.4, 2.0, 0.5);
    none.sites[1].system.set_relaxation(0, None, None).unwrap();
    assert!(none.steady_state().is_err());
}

#[test]
fn long_sweeps_are_refused_before_they_start() {
    // Four relaxing protons in the laboratory frame: a block of 70
    // elements per step, so a million steps are far too many.
    let mut s = SpinSystem::new(1e-6).with_frame(Frame::Laboratory);
    for k in 0..4 {
        s.add_spin(Isotope::H1, k as f64).unwrap();
        s.set_relaxation(k, Some(2.0), None).unwrap();
    }
    s.couple(0, 1, 7.0).unwrap();
    let mut x = Exchange::new();
    x.add_site(s, 1.0).unwrap();
    let start = x.thermal_state().unwrap();
    let sweep = |steps| Event::Sweep {
        to: 1e-3,
        duration: 10.0,
        steps,
    };
    let err = x.evolve(&start, &[sweep(1 << 20)]).unwrap_err();
    assert!(err.to_string().contains("operations"), "{err}");
    assert!(x.evolve(&start, &[sweep(100)]).is_ok());
}

#[test]
fn exchange_input_is_checked() {
    let mut x = two_sites(1.0, 2.0, 0.5, 0.5, 10.0, None);
    // Unbalanced: site 0 loses more than it gains.
    x.channels[1].rate = 5.0;
    assert!(x.validate().is_err());
    let mut x = two_sites(1.0, 2.0, 0.5, 0.5, 10.0, None);
    let mut c13 = SpinSystem::new(9.4);
    c13.add_spin(Isotope::C13, 0.0).unwrap();
    let c = x.add_site(c13, 1.0).unwrap();
    assert!(
        x.add_channel(Channel {
            from: 0,
            to: c,
            rate: 1.0,
            carried: vec![(0, 0)],
            fresh: Fresh::Thermal
        })
        .is_err()
    );
    assert!(
        x.add_channel(Channel {
            from: 0,
            to: 1,
            rate: 1.0,
            carried: vec![(0, 0)],
            fresh: Fresh::Parahydrogen {
                spins: [0, 1],
                fraction: 1.0
            }
        })
        .is_err()
    );
    let mut other = SpinSystem::new(1.0);
    other.add_spin(Isotope::H1, 0.0).unwrap();
    let mut y = two_sites(1.0, 2.0, 0.5, 0.5, 10.0, None);
    y.add_site(other, 1.0).unwrap();
    assert!(y.validate().is_err());
    assert!(Exchange::new().validate().is_err());
    // JSON round trip.
    let s = sabre(1e-6, 2.0, 0.5);
    assert_eq!(Exchange::from_json(&s.to_json()).unwrap(), s);
}
