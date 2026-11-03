//! The sparse joint blocks against the dense blocks of 1.35.

use super::super::*;
use crate::spin::{Event, Frame, Isotope, SpinSystem, Target};

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
        assert!(e < 1e-8, "steady {field} {frame:?} {ring}: {e:e}");
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
