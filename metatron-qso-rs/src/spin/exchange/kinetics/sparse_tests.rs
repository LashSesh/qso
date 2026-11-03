//! Reactions on sparse blocks against the dense blocks of 1.50, a model
//! beyond them against its spectator-free core, and a Monte Carlo of
//! reaction events.

use super::*;
use crate::rng::Rng;
use crate::spin::{Frame, Isotope};

const S: fn(usize) -> Participant = Participant::Site;
const P: fn(usize) -> Participant = Participant::Species;

/// PHIP: a substrate (¹³C and a proton, and `spectators` uncoupled
/// relaxing protons) takes up parahydrogen into a product with two
/// hydrides, every substrate spin carried: `S + H₂ → P` at `k = 2` with
/// `second`, else `S → P` at `k = 1`, the parahydrogen implied (the
/// coherences carried then need no steps of `‖hM‖ ≤ ¼`).
fn phip(field: f64, frame: Frame, spectators: usize, second: bool) -> (Reactions, usize, usize) {
    let mut substrate = SpinSystem::new(field).with_frame(frame);
    substrate.add_spin(Isotope::C13, 160.0).unwrap();
    substrate.add_spin(Isotope::H1, 6.0).unwrap();
    substrate.couple(0, 1, 160.0).unwrap();
    let mut product = SpinSystem::new(field).with_frame(frame);
    product.add_spin(Isotope::H1, 1.2).unwrap();
    product.add_spin(Isotope::H1, 1.6).unwrap();
    product.add_spin(Isotope::C13, 30.0).unwrap();
    product.add_spin(Isotope::H1, 2.0).unwrap();
    product.couple(0, 1, 7.0).unwrap();
    product.couple(0, 2, 5.0).unwrap();
    product.couple(1, 2, 130.0).unwrap();
    product.couple(2, 3, 125.0).unwrap();
    product.couple(1, 3, 3.0).unwrap();
    for k in 0..spectators {
        substrate.add_spin(Isotope::H1, 3.0 + k as f64).unwrap();
        product.add_spin(Isotope::H1, 3.5 + k as f64).unwrap();
    }
    for k in 0..substrate.spins().len() {
        substrate.set_relaxation(k, Some(4.0), Some(2.0)).unwrap();
    }
    for k in 0..product.spins().len() {
        product.set_relaxation(k, Some(3.0), Some(1.0)).unwrap();
    }
    let mut r = Reactions::new();
    let s = r.add_site(substrate, 1.0).unwrap();
    let p = r.add_site(product, 0.0).unwrap();
    let mut carried = vec![(0, 2), (1, 3)];
    carried.extend((0..spectators).map(|k| (2 + k, 4 + k)));
    let (reactants, rate) = if second {
        let h2 = r.add_species("H2", 0.5).unwrap();
        (vec![S(s), P(h2)], 2.0)
    } else {
        (vec![S(s)], 1.0)
    };
    r.add_reaction(Reaction {
        reactants,
        products: vec![S(p)],
        rate,
        carried,
        fresh: Fresh::Parahydrogen {
            spins: [0, 1],
            fraction: 1.0,
        },
    })
    .unwrap();
    (r, s, p)
}

/// The largest difference of two sets of states, relative to the largest
/// deviation from `Tr σ/d`.
fn difference(a: &[CMatrix], b: &[CMatrix]) -> f64 {
    let (mut worst, mut size) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let d = x.nrows();
        let mean = x.trace() / d as f64;
        for r in 0..d {
            for c in 0..d {
                worst = worst.max((x[(r, c)] - y[(r, c)]).norm());
                let shift = if r == c { mean } else { C64::new(0.0, 0.0) };
                size = size.max((x[(r, c)] - shift).norm());
            }
        }
    }
    worst / size
}

#[test]
fn krylov_steps_equal_dense_steps() {
    for (field, frame, second, events) in [
        (
            5e-6,
            Frame::Laboratory,
            true,
            vec![
                Event::Delay(0.4),
                Event::Sweep {
                    to: 2e-6,
                    duration: 0.1,
                    steps: 3,
                },
            ],
        ),
        (
            9.4,
            Frame::Rotating,
            false,
            vec![
                Event::Delay(0.05),
                Event::Pulse {
                    target: Target::Isotope(Isotope::C13),
                    angle: 40.0,
                    phase: 11.0,
                },
                Event::Delay(0.01),
            ],
        ),
    ] {
        let (r, _, p) = phip(field, frame, 0, second);
        let start = r.thermal_state().unwrap();
        let dense = r.evolve_with(&start, &events, Propagation::Dense).unwrap();
        let krylov = r.evolve_with(&start, &events, Propagation::Krylov).unwrap();
        let e = difference(&dense, &krylov);
        assert!(e < 1e-9, "{field} {frame:?}: {e:e}");
        // The product's trace is the amount formed: c_S(0) − c_S(T).
        let amounts = r.amounts(events.iter().map(duration).sum()).unwrap();
        assert!((krylov[p].trace().re - amounts[p]).abs() < 1e-12);
    }
}

fn duration(e: &Event) -> f64 {
    match e {
        Event::Delay(t) => *t,
        Event::Sweep { duration, .. } => *duration,
        _ => 0.0,
    }
}

#[test]
fn spectators_beyond_the_dense_limit_leave_the_core_alone() {
    // Three uncoupled protons that relax to the thermal state and arrive
    // thermal stay thermal and uncorrelated: the core's polarisations are
    // those of the model without them. With them the product has seven
    // spins, a zero-order block of 3432 elements in the laboratory frame.
    let field = 5e-6;
    let (big, s, p) = phip(field, Frame::Laboratory, 3, false);
    let (small, _, _) = phip(field, Frame::Laboratory, 0, false);
    let events = [Event::Delay(0.3)];
    let a = big.evolve(&big.thermal_state().unwrap(), &events).unwrap();
    let b = small
        .evolve(&small.thermal_state().unwrap(), &events)
        .unwrap();
    for (site, spins) in [(s, 2), (p, 4)] {
        let sys_big = &big.sites()[site].system;
        let sys_small = &small.sites()[site].system;
        let (ta, tb) = (a[site].trace().re, b[site].trace().re);
        assert!((ta - tb).abs() < 1e-12, "{ta} vs {tb}");
        for k in 0..spins {
            let pa = sys_big
                .polarization(&(&a[site] * C64::new(1.0 / ta, 0.0)), k)
                .unwrap();
            let pb = sys_small
                .polarization(&(&b[site] * C64::new(1.0 / tb, 0.0)), k)
                .unwrap();
            assert!(
                (pa - pb).abs() < 1e-9 * pb.abs().max(1e-3),
                "site {site} spin {k}: {pa} vs {pb}"
            );
        }
    }
    let largest = big
        .base(field)
        .unwrap()
        .joint
        .blocks
        .iter()
        .map(|(_, _, n)| *n)
        .max()
        .unwrap();
    assert!(largest > crate::spin::MAX_LIOUVILLE_BLOCK, "{largest}");
}

/// `Tr₁ ρ` of a two-spin state (spin 0 kept).
fn keep_first(sys: &SpinSystem, rho: &CMatrix) -> CMatrix {
    let (s0, s1) = (sys.stride(0), sys.stride(1));
    CMatrix::from_fn(2, 2, |a, b| {
        (0..2)
            .map(|k| rho[(a * s0 + k * s1, b * s0 + k * s1)])
            .sum()
    })
}

/// `ρ₀ ⊗ ρ₁` on a two-spin system.
fn join(sys: &SpinSystem, first: &CMatrix, second: &CMatrix) -> CMatrix {
    let (s0, s1) = (sys.stride(0), sys.stride(1));
    let mut out = CMatrix::zeros(4, 4);
    for a in 0..2 {
        for b in 0..2 {
            for c in 0..2 {
                for e in 0..2 {
                    out[(a * s0 + c * s1, b * s0 + e * s1)] = first[(a, b)] * second[(c, e)];
                }
            }
        }
    }
    out
}

#[test]
fn reaction_events_sampled_one_by_one_agree() {
    // A → B at k = 3: each molecule reacts at an exponential time τ, its
    // spin 0 carried into B and B's spin 1 fresh and thermal, then
    // evolves in B until T. The average over molecules is the ensemble the
    // rate equations follow.
    let mut a_sys = SpinSystem::new(9.4);
    a_sys.add_spin(Isotope::H1, 1.0).unwrap();
    a_sys.add_spin(Isotope::H1, 2.5).unwrap();
    a_sys.couple(0, 1, 12.0).unwrap();
    a_sys.set_relaxation(0, Some(2.0), Some(0.5)).unwrap();
    a_sys.set_relaxation(1, Some(3.0), Some(0.8)).unwrap();
    let mut b_sys = SpinSystem::new(9.4);
    b_sys.add_spin(Isotope::H1, 1.8).unwrap();
    b_sys.add_spin(Isotope::H1, 4.0).unwrap();
    b_sys.couple(0, 1, 6.0).unwrap();
    b_sys.set_relaxation(0, Some(1.5), Some(0.4)).unwrap();
    b_sys.set_relaxation(1, Some(1.0), Some(0.3)).unwrap();
    let (k, total) = (3.0, 0.6);
    let mut r = Reactions::new();
    let a = r.add_site(a_sys.clone(), 1.0).unwrap();
    let b = r.add_site(b_sys.clone(), 0.0).unwrap();
    r.add_reaction(Reaction {
        reactants: vec![S(a)],
        products: vec![S(b)],
        rate: k,
        carried: vec![(0, 0)],
        fresh: Fresh::Thermal,
    })
    .unwrap();
    // Start from the first proton's transverse state.
    let tipped = a_sys
        .evolve(
            &a_sys.thermal_state().unwrap(),
            &[Event::Pulse {
                target: Target::Spin(0),
                angle: 90.0,
                phase: 0.0,
            }],
        )
        .unwrap();
    let start = [tipped.clone(), CMatrix::zeros(4, 4)];
    let exact = r.evolve(&start, &[Event::Delay(total)]).unwrap();
    // The fresh spin: thermal populations of B's spin 1 alone.
    let fresh = {
        let p = b_sys.populations(1, 9.4);
        CMatrix::from_fn(2, 2, |i, j| {
            if i == j {
                C64::new(p[i], 0.0)
            } else {
                C64::new(0.0, 0.0)
            }
        })
    };
    // Observables of B: ⟨I₀ˣ⟩, ⟨I₀ʸ⟩ (the carried coherence) and the
    // deviation of ⟨I₁ᶻ⟩ (the fresh spin's relaxation).
    let (ix, iy, iz) = (
        b_sys.ix(0).unwrap(),
        b_sys.iy(0).unwrap(),
        b_sys.iz(1).unwrap(),
    );
    let observe = |rho: &CMatrix| -> [f64; 3] {
        [
            (rho * &ix).trace().re,
            (rho * &iy).trace().re,
            (rho * &iz).trace().re,
        ]
    };
    let want = observe(&exact[b]);
    let mut rng = Rng::seed_from_u64(7);
    let samples = 4000;
    let mut sums = [0.0f64; 3];
    let mut squares = [0.0f64; 3];
    for _ in 0..samples {
        let tau = -rng.uniform().max(1e-300).ln() / k;
        let mut v = [0.0; 3];
        if tau < total {
            let at = a_sys.evolve(&tipped, &[Event::Delay(tau)]).unwrap();
            let moved = join(&b_sys, &keep_first(&a_sys, &at), &fresh);
            let end = b_sys.evolve(&moved, &[Event::Delay(total - tau)]).unwrap();
            v = observe(&end);
        }
        for i in 0..3 {
            sums[i] += v[i];
            squares[i] += v[i] * v[i];
        }
    }
    for i in 0..3 {
        let mean = sums[i] / samples as f64;
        let error = ((squares[i] / samples as f64 - mean * mean) / samples as f64).sqrt();
        assert!(
            (mean - want[i]).abs() < 4.0 * error,
            "{i}: {mean} ± {error} vs {}",
            want[i]
        );
        assert!(error < 0.05 * want[i].abs().max(1e-3), "{i}: {error}");
    }
}
