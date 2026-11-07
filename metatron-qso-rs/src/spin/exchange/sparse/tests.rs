//! The sparse joint blocks against the dense blocks of 1.35.

use super::super::*;
use super::{GMRES_ITERATIONS, GMRES_RESTART, GMRES_TOLERANCE, gmres, norm2};
use crate::spin::{Event, Frame, Isotope, MAX_RELAXATION_WORK, SpinSystem, Target};

/// SABRE with `ring` protons beside the ¹⁵N: the complex (two hydrides,
/// the bound ¹⁵N and its ring protons) and the free substrate (¹⁵N and its
/// ring protons), every spin relaxing, the hydrides renewed from
/// parahydrogen.
pub(crate) fn sabre_ring(field: f64, frame: Frame, ring: usize) -> Exchange {
    let mut complex = SpinSystem::new(field).with_frame(frame);
    complex.add_spin(Isotope::H1, -22.0).unwrap();
    complex.add_spin(Isotope::H1, -22.5).unwrap();
    complex.add_spin(Isotope::N15, 250.0).unwrap();
    complex.couple(0, 1, -7.0).unwrap();
    complex.couple(0, 2, -20.0).unwrap();
    complex.couple(1, 2, 2.0).unwrap();
    let mut free = SpinSystem::new(field).with_frame(frame);
    free.add_spin(Isotope::N15, 310.0).unwrap();
    for k in 0..ring {
        complex.add_spin(Isotope::H1, 8.0 + 0.4 * k as f64).unwrap();
        free.add_spin(Isotope::H1, 8.5 + 0.3 * k as f64).unwrap();
        complex.couple(2, 3 + k, -10.0 + k as f64).unwrap();
        free.couple(0, 1 + k, -11.0 + k as f64).unwrap();
        if k > 0 {
            complex.couple(2 + k, 3 + k, 5.0).unwrap();
            free.couple(k, 1 + k, 5.5).unwrap();
        }
    }
    for k in 0..complex.spins().len() {
        complex.set_relaxation(k, Some(3.0), Some(1.0)).unwrap();
    }
    for k in 0..free.spins().len() {
        free.set_relaxation(k, Some(if k == 0 { 30.0 } else { 5.0 }), Some(2.0))
            .unwrap();
    }
    let (cc, cf) = (1.0, 10.0);
    let mut x = Exchange::new();
    x.add_site(complex, cc).unwrap();
    x.add_site(free, cf).unwrap();
    let para = Fresh::Parahydrogen {
        spins: [0, 1],
        fraction: 0.9,
    };
    let bound: Vec<(usize, usize)> = (0..=ring).map(|k| (2 + k, k)).collect();
    let freed: Vec<(usize, usize)> = (0..=ring).map(|k| (k, 2 + k)).collect();
    let kept: Vec<(usize, usize)> = (2..3 + ring).map(|k| (k, k)).collect();
    for c in [
        Channel {
            from: 0,
            to: 1,
            rate: 2.0,
            carried: bound,
            fresh: Fresh::Thermal,
        },
        Channel {
            from: 1,
            to: 0,
            rate: 2.0 * cc / cf,
            carried: freed,
            fresh: para,
        },
        Channel {
            from: 0,
            to: 0,
            rate: 0.5,
            carried: kept,
            fresh: para,
        },
    ] {
        x.add_channel(c).unwrap();
    }
    x
}

/// The largest difference of the states' deviations from `1/d`, relative
/// to the largest deviation.
fn difference(a: &[CMatrix], b: &[CMatrix]) -> f64 {
    let mut worst = 0.0f64;
    let mut size = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let d = x.nrows();
        for r in 0..d {
            for c in 0..d {
                let shift = if r == c { 1.0 / d as f64 } else { 0.0 };
                worst = worst.max((x[(r, c)] - y[(r, c)]).norm());
                size = size.max((x[(r, c)] - shift).norm());
            }
        }
    }
    worst / size
}

#[test]
fn krylov_blocks_equal_dense_blocks() {
    // Fields where Krylov steps are affordable: microtesla in the
    // laboratory frame (at millitesla its coherences precess at the Larmor
    // differences, and the dense exponentials are cheaper), and the
    // rotating frame for short delays.
    for (field, frame, ring, delay) in [
        (0.14e-6, Frame::Laboratory, 1, 0.3),
        (20e-6, Frame::Laboratory, 2, 0.3),
        (9.4, Frame::Rotating, 2, 0.02),
    ] {
        let x = sabre_ring(field, frame, ring);
        let start = x.thermal_state().unwrap();
        let mut events = vec![Event::Delay(delay)];
        if frame == Frame::Rotating {
            events.push(Event::Pulse {
                target: Target::Isotope(Isotope::N15),
                angle: 63.0,
                phase: 17.0,
            });
            events.push(Event::Delay(0.01));
        } else {
            events.push(Event::Sweep {
                to: 2.0 * field,
                duration: 0.2,
                steps: 4,
            });
        }
        let dense = x.evolve_with(&start, &events, Propagation::Dense).unwrap();
        let krylov = x.evolve_with(&start, &events, Propagation::Krylov).unwrap();
        let auto = x.evolve(&start, &events).unwrap();
        let e = difference(&dense, &krylov);
        assert!(e < 1e-9, "{field} {frame:?} {ring}: {e:e}");
        assert!(difference(&dense, &auto) < 1e-9);
        // Traces stay 1: the amounts.
        for rho in &krylov {
            assert!((rho.trace() - 1.0).norm() < 1e-12);
        }
        // Steady states: GMRES against the dense solve.
        let a = x.steady_state_with(Propagation::Dense).unwrap();
        let b = x.steady_state_with(Propagation::Krylov).unwrap();
        let e = difference(&a, &b);
        assert!(e < 1e-10, "steady {field} {frame:?} {ring}: {e:e}");
        if frame == Frame::Rotating {
            let tipped = x
                .evolve(
                    &a,
                    &[Event::Pulse {
                        target: Target::Isotope(Isotope::H1),
                        angle: 90.0,
                        phase: 0.0,
                    }],
                )
                .unwrap();
            let fd = x
                .fid_with(&tipped, Isotope::H1, 1e-4, 64, Propagation::Dense)
                .unwrap();
            let fk = x
                .fid_with(&tipped, Isotope::H1, 1e-4, 64, Propagation::Krylov)
                .unwrap();
            let size = fd.iter().fold(0.0f64, |m, z| m.max(z.norm()));
            let worst = fd
                .iter()
                .zip(&fk)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).norm()));
            assert!(worst < 1e-9 * size, "fid {worst:e} of {size:e}");
        }
    }
}

#[test]
fn beyond_the_dense_limit_the_steady_state_is_reached() {
    // Four ring protons in the laboratory frame: the complex has 7 spins,
    // a zero-order block of 3432 elements beside the substrate's 252,
    // beyond the dense limit of 2048.
    let x = sabre_ring(0.14e-6, Frame::Laboratory, 4);
    let joint = x.joint(None);
    let largest = joint.blocks.iter().map(|(_, _, n)| *n).max().unwrap();
    assert!(largest > crate::spin::MAX_LIOUVILLE_BLOCK, "{largest}");
    assert!(x.steady_state_with(Propagation::Dense).is_err());
    let steady = x.steady_state().unwrap();
    for rho in &steady {
        assert!((rho.trace() - 1.0).norm() < 1e-12);
    }
    // The steady state is stationary under the evolution, and thermal
    // states approach it.
    let later = x.evolve(&steady, &[Event::Delay(2.0)]).unwrap();
    let e = difference(&steady, &later);
    assert!(e < 1e-8, "{e:e}");
    let p = x.polarization(&steady, 1, 0).unwrap();
    assert!(p.abs() > 1e-3, "{p}");
    let mut state = x.thermal_state().unwrap();
    let mut last = 0.0f64;
    for _ in 0..3 {
        state = x.evolve(&state, &[Event::Delay(4.0)]).unwrap();
        let q = x.polarization(&state, 1, 0).unwrap();
        assert!(q.abs() > last.abs() && q.abs() < p.abs(), "{q} vs {p}");
        last = q;
    }
}

/// A profile from thermal towards the free substrate's ¹⁵N polarisation.
fn profile(x: &Exchange, max: f64, duration: f64, steps: usize) -> Profile {
    let mut target: Vec<CMatrix> = x
        .sites()
        .iter()
        .map(|s| CMatrix::zeros(s.system.dimension(), s.system.dimension()))
        .collect();
    target[1] = x.sites()[1].system.iz(0).unwrap() * C64::new(2.0, 0.0);
    Profile {
        min: 0.0,
        max,
        duration,
        steps,
        initial: x.thermal_state().unwrap(),
        target,
        smoothness: 0.0,
    }
}

#[test]
fn field_profiles_on_sparse_blocks_equal_the_dense_blocks() {
    // Two ring protons, every spin relaxing, between 0 and 0.4 μT.
    let x = sabre_ring(1.4e-7, Frame::Laboratory, 2);
    let p = profile(&x, 4e-7, 2.0, 6);
    let fields: Vec<f64> = (0..6).map(|k| 0.5e-7 + 0.5e-7 * k as f64).collect();
    let (value, grad) = x.profile_transfer(&p, &fields).unwrap();
    let (dense, dense_grad) = x.profile_transfer_dense(&p, &fields).unwrap();
    assert!((value - dense).abs() < 1e-10, "{value} vs {dense}");
    let scale = dense_grad.iter().fold(0.0f64, |m, g| m.max(g.abs()));
    for (a, b) in grad.iter().zip(&dense_grad) {
        assert!((a - b).abs() < 1e-6 * scale, "{a} vs {b}");
    }
    // Replayed by evolve.
    let events: Vec<Event> = fields
        .iter()
        .flat_map(|&b| [Event::Field(b), Event::Delay(2.0 / 6.0)])
        .collect();
    let want = x
        .polarization(&x.evolve(&p.initial, &events).unwrap(), 1, 0)
        .unwrap();
    assert!((value - want).abs() < 1e-10, "{value} vs {want}");
    let (error, nodes) = x.profile_relaxation_error(&p).unwrap();
    assert!(error < 1e-10 && nodes >= 5, "{error} {nodes}");
}

#[test]
fn field_profiles_beyond_the_dense_limit_follow_central_differences() {
    // Four ring protons: a zero-order block of 3684 elements.
    let x = sabre_ring(1.4e-7, Frame::Laboratory, 4);
    let p = profile(&x, 3e-7, 0.4, 4);
    assert!(x.profile_transfer_dense(&p, &[1e-7; 4]).is_err());
    let fields = [0.8e-7, 1.2e-7, 1.6e-7, 1.0e-7];
    let (value, grad) = x.profile_transfer(&p, &fields).unwrap();
    let events: Vec<Event> = fields
        .iter()
        .flat_map(|&b| [Event::Field(b), Event::Delay(0.1)])
        .collect();
    let want = x
        .polarization(&x.evolve(&p.initial, &events).unwrap(), 1, 0)
        .unwrap();
    assert!(
        (value - want).abs() < 1e-10 * want.abs().max(1e-6),
        "{value} vs {want}"
    );
    // Along a direction, Richardson-extrapolated central differences.
    let dir = [1.0, -0.5, 0.25, 0.7];
    let along = |h: f64| {
        let up: Vec<f64> = fields.iter().zip(&dir).map(|(b, d)| b + h * d).collect();
        let down: Vec<f64> = fields.iter().zip(&dir).map(|(b, d)| b - h * d).collect();
        (x.profile_transfer(&p, &up).unwrap().0 - x.profile_transfer(&p, &down).unwrap().0)
            / (2.0 * h)
    };
    let h = 2e-9;
    let fd = (4.0 * along(h / 2.0) - along(h)) / 3.0;
    let exact: f64 = grad.iter().zip(&dir).map(|(g, d)| g * d).sum();
    assert!((fd - exact).abs() < 1e-6 * exact.abs(), "{fd} vs {exact}");
}

/// SABRE with seven to nine spins in the complex, for the documentation
/// (`cargo test --release -- --ignored measure_sparse_exchange --nocapture`).
#[test]
#[ignore]
fn measure_sparse_exchange() {
    for ring in [4usize, 5, 6] {
        let x = sabre_ring(1.4e-7, Frame::Laboratory, ring);
        let joint = x.joint(None);
        let zero = joint
            .blocks
            .iter()
            .filter(|(order, _, _)| order.iter().all(|&q| q == 0))
            .map(|(_, _, n)| *n)
            .sum::<usize>();
        let total: usize = joint.blocks.iter().map(|(_, _, n)| *n).sum();
        let clock = std::time::Instant::now();
        let steady = x.steady_state().unwrap();
        let t_steady = clock.elapsed().as_secs_f64();
        let p = x.polarization(&steady, 1, 0).unwrap();
        let clock = std::time::Instant::now();
        let later = x
            .evolve(&x.thermal_state().unwrap(), &[Event::Delay(1.0)])
            .unwrap();
        let t_delay = clock.elapsed().as_secs_f64();
        let q = x.polarization(&later, 1, 0).unwrap();
        let prof = profile(&x, 3e-7, 1.0, 5);
        let clock = std::time::Instant::now();
        let field = x.exchange_field(&prof).unwrap();
        let t_build = clock.elapsed().as_secs_f64();
        let clock = std::time::Instant::now();
        let (value, _) = x
            .field_transfer(&field, &prof, &[1.0e-7, 1.2e-7, 1.4e-7, 1.6e-7, 1.4e-7])
            .unwrap();
        let t_eval = clock.elapsed().as_secs_f64();
        eprintln!(
            "complex of {} spins, substrate of {}: zero-order block {zero}, all blocks {total}; \
             steady state {t_steady:.1} s (P_N = {p:.4}); 1 s from thermal {t_delay:.1} s \
             (P_N = {q:.3e}); profile of 5 steps over 1 s: interpolant {t_build:.1} s ({} \
             fields, error {:.1e}), evaluation with gradient {t_eval:.1} s (⟨2N_z⟩ = {value:.3e})",
            3 + ring,
            1 + ring,
            field.nodes(),
            field.error
        );
    }
}

/// One ¹H that relaxes beside an uncoupled ¹³C, with or without its
/// relaxation, at `field` in the rotating frame.
fn pair_site(c13_relaxes: bool) -> SpinSystem {
    let mut s = SpinSystem::new(9.4);
    s.add_spin(Isotope::H1, 1.0).unwrap();
    s.add_spin(Isotope::C13, 30.0).unwrap();
    s.set_relaxation(0, Some(2.0), Some(1.0)).unwrap();
    if c13_relaxes {
        s.set_relaxation(1, Some(5.0), Some(1.0)).unwrap();
    }
    s
}

#[test]
fn steady_states_that_are_not_unique_are_refused() {
    // A spin that neither relaxes nor exchanges keeps any polarisation:
    // the dense solve refuses, and so must GMRES.
    for relaxes in [false, true] {
        let mut x = Exchange::new();
        x.add_site(pair_site(relaxes), 1.0).unwrap();
        let dense = x.steady_state_with(Propagation::Dense);
        let krylov = x.steady_state_with(Propagation::Krylov);
        assert_eq!(dense.is_ok(), relaxes);
        assert_eq!(krylov.is_ok(), relaxes, "{relaxes}: {krylov:?}");
        let mut lone = Exchange::new();
        let mut s = SpinSystem::new(9.4);
        s.add_spin(Isotope::H1, 1.0).unwrap();
        lone.add_site(s, 1.0).unwrap();
        assert!(lone.steady_state_with(Propagation::Krylov).is_err());
    }
    // Beyond the dense limit: SABRE beside a site whose ¹³C is left alone.
    let mut x = sabre_ring(1.4e-7, Frame::Laboratory, 4);
    let mut s = SpinSystem::new(1.4e-7).with_frame(Frame::Laboratory);
    s.add_spin(Isotope::C13, 30.0).unwrap();
    x.add_site(s, 1.0).unwrap();
    assert!(x.steady_state().is_err());
}

#[test]
fn steady_states_hold_where_the_trace_terms_cancel_the_chemistry() {
    // Two one-proton sites exchanging at k = 2 each way, T₁ tuned so that
    // each site's fastest rate is 2k: with ρ_s of the other sign the trace
    // equations would be singular.
    let site = || {
        let mut s = SpinSystem::new(9.4);
        s.add_spin(Isotope::H1, 1.0).unwrap();
        s.set_relaxation(0, Some(0.25000000000085665), None)
            .unwrap();
        s
    };
    let mut x = Exchange::new();
    x.add_site(site(), 1.0).unwrap();
    x.add_site(site(), 1.0).unwrap();
    for (from, to) in [(0, 1), (1, 0)] {
        x.add_channel(Channel {
            from,
            to,
            rate: 2.0,
            carried: vec![(0, 0)],
            fresh: Fresh::Thermal,
        })
        .unwrap();
    }
    let a = x.steady_state_with(Propagation::Dense).unwrap();
    let b = x.steady_state_with(Propagation::Krylov).unwrap();
    let e = difference(&a, &b);
    assert!(e < 1e-10, "{e:e}");
}

#[test]
fn results_do_not_depend_on_the_unit_of_the_concentrations() {
    // The normalised dynamics are the same for concentrations of any
    // unit; neither the results nor the estimates may change.
    let base = sabre_ring(1.4e-7, Frame::Laboratory, 1);
    let start = base.thermal_state().unwrap();
    let events = [Event::Delay(0.3)];
    let want = base
        .evolve_with(&start, &events, Propagation::Krylov)
        .unwrap();
    for unit in [1e-9, 1e6, 1e12] {
        let mut x = base.clone();
        for site in &mut x.sites {
            site.concentration *= unit;
        }
        for method in [Propagation::Dense, Propagation::Krylov] {
            let got = x.evolve_with(&start, &events, method).unwrap();
            let e = difference(&want, &got);
            assert!(e < 1e-9, "{unit} {method:?}: {e:e}");
        }
        let steady = x.steady_state_with(Propagation::Krylov).unwrap();
        let e = difference(&base.steady_state().unwrap(), &steady);
        assert!(e < 1e-10, "{unit}: {e:e}");
    }
}

#[test]
fn field_profiles_with_field_dependent_relaxation() {
    // From 0 to 1 mT the fresh thermal states and the relaxation's drift
    // change with the field: the interpolant's derivative carries part of
    // the gradient. Against the replay, the dense blocks' gradient (their
    // field derivative by differences) and Richardson-extrapolated central
    // differences, with steps below the 5·10⁻⁷ T over which the Larmor
    // phase of 0.05 s turns once.
    let x = sabre_ring(1e-4, Frame::Laboratory, 0);
    let p = profile(&x, 1e-3, 0.2, 4);
    let fields = [2e-4, 6e-4, 9e-4, 4e-4];
    let (value, grad) = x.profile_transfer(&p, &fields).unwrap();
    // The transfer, 4·10⁻¹⁰, beside parahydrogen order of 1 in the state:
    // replays agree to rounding of the deviation.
    let events: Vec<Event> = fields
        .iter()
        .flat_map(|&b| [Event::Field(b), Event::Delay(0.05)])
        .collect();
    for method in [Propagation::Dense, Propagation::Krylov] {
        let end = x.evolve_with(&p.initial, &events, method).unwrap();
        let want = x.polarization(&end, 1, 0).unwrap();
        assert!(
            (value - want).abs() < 1e-14,
            "{method:?}: {value} vs {want}"
        );
    }
    let (_, dense) = x.profile_transfer_dense(&p, &fields).unwrap();
    let scale = dense.iter().fold(0.0f64, |m, g| m.max(g.abs()));
    for (a, b) in grad.iter().zip(&dense) {
        assert!((a - b).abs() < 1e-8 * scale, "{a} vs {b}");
    }
    let dir = [0.3, -1.0, 0.6, 0.8];
    let along = |h: f64| {
        let up: Vec<f64> = fields.iter().zip(&dir).map(|(b, d)| b + h * d).collect();
        let down: Vec<f64> = fields.iter().zip(&dir).map(|(b, d)| b - h * d).collect();
        (x.profile_transfer(&p, &up).unwrap().0 - x.profile_transfer(&p, &down).unwrap().0)
            / (2.0 * h)
    };
    let h = 4e-9;
    let fd = (4.0 * along(h / 2.0) - along(h)) / 3.0;
    let exact: f64 = grad.iter().zip(&dir).map(|(g, d)| g * d).sum();
    assert!((fd - exact).abs() < 1e-5 * exact.abs(), "{fd} vs {exact}");
}

/// 1.54's `gmres` as it was before the iteration moved to
/// `linalg::gmres`, verbatim, for the comparison below.
fn gmres_154(
    n: usize,
    apply: impl Fn(&[C64], &mut [C64]),
    precondition: impl Fn(&[C64]) -> Vec<C64>,
    b: &[C64],
    room: usize,
    (per, limit): (f64, f64),
) -> Result<Option<Vec<C64>>> {
    let target = GMRES_TOLERANCE * norm2(b);
    let mut x = vec![ZERO; n];
    if target == 0.0 {
        return Ok(Some(x));
    }
    let restart = room.max(GMRES_RESTART).min(n.max(1));
    let mut spent = 0.0;
    let mut iterations = 0;
    let mut w = vec![ZERO; n];
    loop {
        apply(&x, &mut w);
        let r: Vec<C64> = b.iter().zip(&w).map(|(b, ax)| b - ax).collect();
        let beta = norm2(&r);
        if beta <= target {
            return Ok(Some(x));
        }
        if iterations >= GMRES_ITERATIONS {
            return Ok(None);
        }
        let mut basis: Vec<Vec<C64>> = vec![r.iter().map(|v| v / beta).collect()];
        let mut h: Vec<Vec<C64>> = Vec::new();
        let mut rotations: Vec<(f64, C64)> = Vec::new();
        let mut g = vec![ZERO; restart + 1];
        g[0] = C64::new(beta, 0.0);
        let mut k = 0;
        while k < restart && iterations < GMRES_ITERATIONS {
            iterations += 1;
            spent += per + 16.0 * ((k + 1) * n) as f64;
            if spent > limit {
                return Err(QsoError::invalid(
                    "steady state",
                    format!(
                        "GMRES needs more than {:.1e} operations after {iterations} \
                         iterations; use fewer spins",
                        MAX_RELAXATION_WORK
                    ),
                ));
            }
            let z = precondition(&basis[k]);
            apply(&z, &mut w);
            let mut column = vec![ZERO; k + 2];
            for (i, v) in basis.iter().enumerate() {
                let hij: C64 = v.iter().zip(&w).map(|(a, b)| a.conj() * b).sum();
                column[i] = hij;
                for (x, b) in w.iter_mut().zip(v) {
                    *x -= hij * b;
                }
            }
            let s = norm2(&w);
            column[k + 1] = C64::new(s, 0.0);
            for (i, &(c, sn)) in rotations.iter().enumerate() {
                let (p, q) = (column[i], column[i + 1]);
                column[i] = p * c + sn * q;
                column[i + 1] = -sn.conj() * p + q * c;
            }
            let (p, q) = (column[k], column[k + 1]);
            let r = (p.norm_sqr() + q.norm_sqr()).sqrt();
            let (c, sn) = if r == 0.0 {
                (1.0, ZERO)
            } else if p.norm() == 0.0 {
                (0.0, C64::new(1.0, 0.0))
            } else {
                (p.norm() / r, (p / p.norm()) * q.conj() / r)
            };
            column[k] = p * c + sn * q;
            column[k + 1] = ZERO;
            g[k + 1] = -sn.conj() * g[k];
            g[k] *= c;
            rotations.push((c, sn));
            h.push(column);
            k += 1;
            if g[k].norm() <= target || s == 0.0 {
                break;
            }
            basis.push(w.iter().map(|v| v / s).collect());
        }
        // y from the triangle, x += M⁻¹ V y.
        let mut y = vec![ZERO; k];
        for i in (0..k).rev() {
            let mut acc = g[i];
            for j in i + 1..k {
                acc -= h[j][i] * y[j];
            }
            if h[i][i].norm() == 0.0 {
                return Ok(None);
            }
            y[i] = acc / h[i][i];
        }
        let mut update = vec![ZERO; n];
        for (v, yi) in basis.iter().zip(&y) {
            for (u, b) in update.iter_mut().zip(v) {
                *u += yi * b;
            }
        }
        for (xi, d) in x.iter_mut().zip(precondition(&update)) {
            *xi += d;
        }
    }
}

/// A sparse non-normal operator: a convection–diffusion stencil on a ring
/// with a complex shift, plus two random couplings per row.
struct Ring {
    rows: Vec<Vec<(usize, C64)>>,
}

impl Ring {
    fn new(n: usize, seed: u64, shift: C64, convection: f64) -> Self {
        let mut rng = crate::rng::Rng::seed_from_u64(seed);
        let rows = (0..n)
            .map(|i| {
                let mut row = vec![
                    (i, C64::new(2.0, 0.0) + shift),
                    ((i + 1) % n, C64::new(-1.0 + convection, 0.0)),
                    ((i + n - 1) % n, C64::new(-1.0 - convection, 0.0)),
                ];
                for _ in 0..2 {
                    let j = rng.below(n as u64) as usize;
                    row.push((j, C64::new(0.2 * rng.normal(), 0.2 * rng.normal())));
                }
                row
            })
            .collect();
        Ring { rows }
    }

    fn apply(&self, x: &[C64], y: &mut [C64]) {
        for (yi, row) in y.iter_mut().zip(&self.rows) {
            *yi = row.iter().map(|&(j, v)| v * x[j]).sum();
        }
    }

    fn diagonal(&self) -> Vec<C64> {
        (0..self.rows.len())
            .map(|i| self.rows[i].iter().filter(|e| e.0 == i).map(|e| e.1).sum())
            .collect()
    }
}

#[test]
fn gmres_through_linalg_is_bit_identical_to_1_54() {
    type Outcome = Result<Option<Vec<C64>>>;
    let same = |a: &Outcome, b: &Outcome| match (a, b) {
        (Ok(Some(x)), Ok(Some(y))) => {
            x.len() == y.len()
                && x.iter().zip(y).all(|(p, q)| {
                    p.re.to_bits() == q.re.to_bits() && p.im.to_bits() == q.im.to_bits()
                })
        }
        (Ok(None), Ok(None)) => true,
        (Err(e), Err(f)) => e == f,
        _ => false,
    };
    let kind = |a: &Outcome| match a {
        Ok(Some(_)) => "converged",
        Ok(None) => "none",
        Err(_) => "error",
    };
    // Converging runs with and without the Jacobi preconditioner (2 to 199
    // restart cycles, 99 to 11 922 iterations), one with a larger room, and
    // the same runs under a work limit that bites.
    for (n, seed, shift, convection, room, jacobi) in [
        (200, 1, C64::new(0.3, 0.1), 0.2, 0, true),
        (200, 2, C64::new(0.05, 0.0), 0.4, 0, false),
        (200, 3, C64::new(0.01, 0.02), 0.6, 0, true),
        (1000, 3, C64::new(0.1, 0.02), 0.6, 0, true),
        (1000, 4, C64::new(0.1, 0.02), 0.6, 300, true),
        (5000, 5, C64::new(0.2, 0.0), 0.3, 0, true),
    ] {
        let op = Ring::new(n, seed, shift, convection);
        let d = op.diagonal();
        let mut rng = crate::rng::Rng::seed_from_u64(seed + 100);
        let b: Vec<C64> = (0..n)
            .map(|_| C64::new(rng.normal(), rng.normal()))
            .collect();
        let pre = |r: &[C64]| -> Vec<C64> {
            if jacobi {
                r.iter().zip(&d).map(|(x, d)| x / d).collect()
            } else {
                r.to_vec()
            }
        };
        let work = (8.0 * 5.0 * n as f64, MAX_RELAXATION_WORK);
        let old = gmres_154(n, |x, y| op.apply(x, y), pre, &b, room, work);
        let new = gmres(n, |x, y| op.apply(x, y), pre, &b, room, work);
        assert!(
            same(&old, &new),
            "n = {n}, seed {seed}: {} / {}",
            kind(&old),
            kind(&new)
        );
        assert_eq!(kind(&new), "converged", "n = {n}, seed {seed}");
        let tight = (work.0, 30.0 * work.0);
        let old = gmres_154(n, |x, y| op.apply(x, y), pre, &b, room, tight);
        let new = gmres(n, |x, y| op.apply(x, y), pre, &b, room, tight);
        assert!(same(&old, &new), "tight, n = {n}");
        assert_eq!(kind(&new), "error");
    }
    // The iterations run out on a nearly singular operator (334 cycles).
    let n = 200;
    let op = Ring::new(n, 9, C64::new(1e-9, 0.0), 0.0);
    let mut rng = crate::rng::Rng::seed_from_u64(109);
    let b: Vec<C64> = (0..n)
        .map(|_| C64::new(rng.normal(), rng.normal()))
        .collect();
    let identity = |r: &[C64]| r.to_vec();
    let unlimited = (1.0, f64::INFINITY);
    let old = gmres_154(n, |x, y| op.apply(x, y), identity, &b, 0, unlimited);
    let new = gmres(n, |x, y| op.apply(x, y), identity, &b, 0, unlimited);
    assert!(same(&old, &new) && kind(&new) == "none", "{}", kind(&new));
    // Breakdown on the zero operator, and b = 0.
    let ones = [C64::new(1.0, 0.0); 5];
    let zero = |_: &[C64], y: &mut [C64]| y.fill(ZERO);
    let old = gmres_154(5, zero, identity, &ones, 0, (1.0, 1e12));
    let new = gmres(5, zero, identity, &ones, 0, (1.0, 1e12));
    assert!(same(&old, &new) && kind(&new) == "none");
    let copy = |x: &[C64], y: &mut [C64]| y.copy_from_slice(x);
    let old = gmres_154(3, copy, identity, &[ZERO; 3], 0, (1.0, 1e12));
    let new = gmres(3, copy, identity, &[ZERO; 3], 0, (1.0, 1e12));
    assert!(same(&old, &new) && kind(&new) == "converged");
}

// TEMPORARY (verification): random comparison of the wrapper with 1.54.
#[test]
fn tmp_gmres_wrapper_random_bit_identity() {
    use std::cell::Cell;
    type Outcome = Result<Option<Vec<C64>>>;
    let bits = |a: &Outcome| -> String {
        match a {
            Ok(Some(x)) => format!(
                "some {:?}",
                x.iter()
                    .map(|z| (z.re.to_bits(), z.im.to_bits()))
                    .collect::<Vec<_>>()
            ),
            Ok(None) => "none".into(),
            Err(e) => format!("err {e}"),
        }
    };
    let mut rng = crate::rng::Rng::seed_from_u64(20261107);
    let mut tally = std::collections::BTreeMap::<String, usize>::new();
    for case in 0..1500u64 {
        let n = match rng.below(10) {
            0 => 1 + rng.below(3) as usize,
            1 => 400 + rng.below(700) as usize,
            _ => 2 + rng.below(120) as usize,
        };
        let kind = rng.below(9);
        // Dense coefficients for kind 1.
        let dense: Vec<C64> = if kind == 1 && n <= 150 {
            (0..n * n)
                .map(|_| C64::new(rng.normal(), rng.normal()) / (n as f64).sqrt())
                .collect()
        } else {
            Vec::new()
        };
        let shift = match rng.below(4) {
            0 => C64::new(1e-9, 0.0),
            1 => C64::new(rng.normal(), rng.normal()),
            2 => ZERO,
            _ => C64::new(0.05 * rng.uniform(), 0.02 * rng.normal()),
        };
        let ring = Ring::new(n, case, shift, rng.uniform());
        let levels: Vec<f64> = (0..n)
            .map(|i| ((i * 7 + case as usize) % 4) as f64)
            .collect();
        let poison = rng.below(40) == 0;
        let apply = |x: &[C64], y: &mut [C64]| match kind {
            0 | 8 => ring.apply(x, y),
            1 if !dense.is_empty() => {
                for i in 0..n {
                    let mut acc = shift * x[i];
                    for j in 0..n {
                        acc += dense[i + j * n] * x[j];
                    }
                    y[i] = acc;
                }
            }
            2 => {
                for i in 0..n {
                    y[i] = levels[i] * x[i];
                }
            }
            3 => y.fill(ZERO),
            4 => {
                for i in 0..n {
                    y[i] = if i + 1 < n { x[i + 1] } else { ZERO };
                }
            }
            5 => y.copy_from_slice(x),
            6 => {
                ring.apply(x, y);
                if poison {
                    y[0] = C64::new(f64::NAN, 0.0);
                }
            }
            _ => ring.apply(x, y),
        };
        let diag = ring.diagonal();
        let scaling: Vec<C64> = (0..n)
            .map(|_| C64::new(0.5 + rng.uniform(), 0.3 * rng.normal()))
            .collect();
        let pre_kind = rng.below(5);
        let pre = |r: &[C64]| -> Vec<C64> {
            match pre_kind {
                0 | 1 => r.to_vec(),
                2 => r.iter().zip(&diag).map(|(x, d)| x / d).collect(),
                3 => r.iter().zip(&scaling).map(|(x, d)| x * d).collect(),
                _ => {
                    if kind == 7 {
                        vec![ZERO; r.len()]
                    } else {
                        r.iter().zip(&scaling).map(|(x, d)| x / d).collect()
                    }
                }
            }
        };
        let b: Vec<C64> = match rng.below(9) {
            0 => vec![ZERO; n],
            1 => {
                let mut b = vec![ZERO; n];
                b[rng.below(n as u64) as usize] = C64::new(1.0, -0.5);
                b
            }
            2 => (0..n)
                .map(|_| C64::new(rng.normal(), rng.normal()) * 1e-200)
                .collect(),
            3 => (0..n)
                .map(|_| C64::new(rng.normal(), rng.normal()) * 1e200)
                .collect(),
            4 => (0..n)
                .map(|_| C64::new(5e-324 * rng.below(3) as f64, 0.0))
                .collect(),
            _ => (0..n)
                .map(|_| C64::new(rng.normal(), rng.normal()))
                .collect(),
        };
        let room = match rng.below(4) {
            0 => 0,
            1 => 1 + rng.below(2 * n as u64) as usize,
            2 => 10 * n,
            _ => rng.below(70) as usize,
        };
        let per = 8.0 * 5.0 * n as f64;
        let limit = match rng.below(5) {
            0 => f64::INFINITY,
            1 => per * (1.0 + 50.0 * rng.uniform()),
            2 => 0.0,
            3 => MAX_RELAXATION_WORK,
            _ => per * 2000.0 * rng.uniform() + 16.0 * (n * 60 * 40) as f64,
        };
        let calls = [Cell::new(0usize), Cell::new(0usize)];
        let pcalls = [Cell::new(0usize), Cell::new(0usize)];
        let old = gmres_154(
            n,
            |x, y| {
                calls[0].set(calls[0].get() + 1);
                apply(x, y)
            },
            |r| {
                pcalls[0].set(pcalls[0].get() + 1);
                pre(r)
            },
            &b,
            room,
            (per, limit),
        );
        let new = gmres(
            n,
            |x, y| {
                calls[1].set(calls[1].get() + 1);
                apply(x, y)
            },
            |r| {
                pcalls[1].set(pcalls[1].get() + 1);
                pre(r)
            },
            &b,
            room,
            (per, limit),
        );
        assert_eq!(bits(&old), bits(&new), "case {case}: n {n}, kind {kind}");
        assert_eq!(calls[0].get(), calls[1].get(), "case {case}");
        assert_eq!(pcalls[0].get(), pcalls[1].get(), "case {case}");
        let label = match &new {
            Ok(Some(x)) if x.iter().any(|z| z.re.is_nan() || z.im.is_nan()) => "some-nan",
            Ok(Some(_)) => "some",
            Ok(None) => "none",
            Err(_) => "err",
        };
        *tally.entry(format!("kind {kind} {label}")).or_default() += 1;
    }
    println!("{tally:#?}");
}
