//! Tests of milestone 1.45: the generalised cluster-correlation expansion.

use num_complex::Complex64 as C64;

use super::*;
use crate::spin::nv::bath::{Cce, diamond_bath};
use crate::spin::nv::{cpmg, hahn, ramsey};

/// Four ¹³C close together (every pair coupled) and two farther out.
fn compact() -> Vec<[f64; 3]> {
    [
        [0.6, 0.2, 0.4],
        [0.9, 0.1, 0.1],
        [0.5, 0.6, 0.0],
        [0.8, 0.5, 0.5],
        [1.4, -0.3, 0.9],
        [-0.8, 0.9, 1.1],
    ]
    .iter()
    .map(|p| p.map(|x| x * 1e-9))
    .collect()
}

fn bath(field: [f64; 3], count: usize) -> GeneralisedBath {
    let spins = compact()[..count]
        .iter()
        .map(|&p| FullSpin::point_dipole(p).unwrap())
        .collect();
    GeneralisedBath::new(field, -1, spins)
}

fn sequences() -> Vec<Vec<NvEvent>> {
    vec![hahn(0.2e-3), hahn(0.6e-3), cpmg(2, 0.15e-3), ramsey(20e-6)]
}

#[test]
fn the_expansion_at_full_order_is_exact() {
    // Four nuclei, every pair coupled, in a tilted field: the expansion to
    // fourth order telescopes to the exact propagation, for ideal pulses in
    // either frame and for finite pulses.
    let mut b = bath([0.004, -0.002, 0.012], 4);
    for dressed in [false, true] {
        b.dressed = dressed;
        for pulses in [
            Pulses::Ideal,
            Pulses::Finite { rabi: 15e6 },
            Pulses::Shaped {
                rabi: 20e6,
                envelope: gaussian(8),
            },
            Pulses::Laboratory,
        ] {
            let cce = GeneralisedCce {
                order: 4,
                cutoff: f64::INFINITY,
                pulses: pulses.clone(),
                ..Default::default()
            };
            let seqs = sequences();
            let exact = b.exact(&seqs, f64::INFINITY, &pulses).unwrap();
            let coherence = b.coherence(&seqs, &cce).unwrap();
            let signal = b.signal(&seqs, &cce).unwrap();
            for (k, (l, s)) in exact.iter().enumerate() {
                assert!(
                    (coherence[k] - l).norm() < 1e-10,
                    "{dressed} {pulses:?} {k}: {} vs {l}",
                    coherence[k]
                );
                assert!((signal[k] - s).abs() < 1e-10, "{pulses:?} {k}");
            }
            // Lower orders differ, but not by much for a bath this small.
            let second = b
                .coherence(&seqs, &GeneralisedCce { order: 2, ..cce })
                .unwrap();
            assert!(
                second
                    .iter()
                    .zip(&exact)
                    .any(|(a, (l, _))| (a - l).norm() > 1e-12)
            );
        }
    }
}

/// A Gaussian envelope of `k` segments over ±2σ, with phase 0.
fn gaussian(k: usize) -> Vec<(f64, f64)> {
    (0..k)
        .map(|j| {
            let x = -2.0 + 4.0 * (j as f64 + 0.5) / k as f64;
            ((-x * x / 2.0).exp(), 0.0)
        })
        .collect()
}

#[test]
fn without_nuclei_the_qubit_returns() {
    // The electron alone: a Hahn echo returns it to α with ideal pulses;
    // finite pulses lose a little to the off-resonant transition.
    let b = bath([0.0, 0.0, 0.05], 0);
    let cce = GeneralisedCce {
        order: 1,
        ..Default::default()
    };
    let seqs = vec![hahn(0.3e-3), cpmg(4, 0.1e-3)];
    for s in b.signal(&seqs, &cce).unwrap() {
        assert!((s - 1.0).abs() < 1e-12, "{s}");
    }
    for l in b.coherence(&seqs, &cce).unwrap() {
        assert!((l - C64::new(1.0, 0.0)).norm() < 1e-12);
    }
    for pulses in [
        Pulses::Finite { rabi: 20e6 },
        Pulses::Shaped {
            rabi: 30e6,
            envelope: gaussian(16),
        },
    ] {
        let finite = GeneralisedCce {
            pulses,
            ..cce.clone()
        };
        for s in b.signal(&seqs, &finite).unwrap() {
            assert!(s > 0.99 && s <= 1.0 + 1e-12, "{s}");
        }
    }
}

#[test]
fn shaped_pulses() {
    let b = bath([0.0, 0.0, 0.05], 3);
    let seqs = vec![hahn(0.2e-3), cpmg(2, 0.1e-3)];
    let run = |pulses: Pulses| b.exact(&seqs, f64::INFINITY, &pulses).unwrap();
    // A rectangular envelope cut into segments is the rectangular pulse;
    // amplitude 1/2 at twice the Rabi frequency too.
    let finite = run(Pulses::Finite { rabi: 10e6 });
    for pulses in [
        Pulses::Shaped {
            rabi: 10e6,
            envelope: vec![(1.0, 0.0); 5],
        },
        Pulses::Shaped {
            rabi: 20e6,
            envelope: vec![(0.5, 0.0); 3],
        },
    ] {
        for ((a, s), (b, t)) in run(pulses).iter().zip(&finite) {
            assert!((a - b).norm() < 1e-10 && (s - t).abs() < 1e-10);
        }
    }
    // A phase of 180° on the envelope is a phase of 180° on every pulse but
    // the last (which reads out ideally): the same signal.
    let flipped = run(Pulses::Shaped {
        rabi: 10e6,
        envelope: vec![(1.0, 180.0)],
    });
    let turned: Vec<Vec<NvEvent>> = seqs
        .iter()
        .map(|seq| {
            let last = seq.len() - 1;
            seq.iter()
                .enumerate()
                .map(|(k, e)| match *e {
                    NvEvent::Pulse { angle, phase } if k < last => NvEvent::Pulse {
                        angle,
                        phase: phase + 180.0,
                    },
                    other => other,
                })
                .collect()
        })
        .collect();
    let opposite = b
        .exact(&turned, f64::INFINITY, &Pulses::Finite { rabi: 10e6 })
        .unwrap();
    for ((_, s), (_, t)) in flipped.iter().zip(&opposite) {
        assert!((s - t).abs() < 1e-10, "{s} {t}");
    }
    // A Gaussian pulse differs from the rectangular one of equal area.
    let shaped = run(Pulses::Shaped {
        rabi: 10e6,
        envelope: gaussian(12),
    });
    assert!(
        shaped
            .iter()
            .zip(&finite)
            .any(|((_, s), (_, t))| (s - t).abs() > 1e-9)
    );
    // Invalid envelopes.
    for envelope in [
        vec![],
        vec![(-0.1, 0.0)],
        vec![(0.0, 0.0), (0.0, 0.0)],
        vec![(1.0, f64::NAN)],
        vec![(f64::INFINITY, 0.0)],
        vec![(1.0, 0.0); MAX_ENVELOPE_SEGMENTS + 1],
    ] {
        let pulses = Pulses::Shaped {
            rabi: 10e6,
            envelope,
        };
        assert!(pulses.validate().is_err());
        assert!(b.exact(&seqs, f64::INFINITY, &pulses).is_err());
    }
}

#[test]
fn the_point_dipole_tensor() {
    let p = [0.7e-9, -0.4e-9, 0.9e-9];
    let full = FullSpin::point_dipole(p).unwrap();
    let row = BathSpin::point_dipole(p).unwrap().hyperfine;
    for c in 0..3 {
        assert!((full.hyperfine[2][c] - row[c]).abs() < 1e-9 * row[2].abs());
        for r in 0..3 {
            assert!((full.hyperfine[r][c] - full.hyperfine[c][r]).abs() < 1e-9);
        }
    }
    let trace: f64 = (0..3).map(|c| full.hyperfine[c][c]).sum();
    assert!(trace.abs() < 1e-9);
    assert!(FullSpin::point_dipole([0.0; 3]).is_err());
}

#[test]
fn secular_couplings_give_the_secular_expansion() {
    // With the hyperfine tensors cut to their row A_z· and the field along
    // the axis, the electron never leaves its branch between pulses: every
    // order of the generalised expansion is that of 1.39 (to the rounding
    // of D·t, about 10⁷ rad, in the laboratory frame).
    let field = 0.05;
    let spins: Vec<BathSpin> = compact()
        .iter()
        .map(|&p| BathSpin::point_dipole(p).unwrap())
        .collect();
    let secular = SpinBath::new(field, -1, spins);
    let mut general = GeneralisedBath::from_bath(&secular).unwrap();
    for s in &mut general.spins {
        s.hyperfine[0] = [0.0; 3];
        s.hyperfine[1] = [0.0; 3];
    }
    let seqs = vec![hahn(0.2e-3), hahn(0.8e-3), cpmg(2, 0.3e-3)];
    for order in [1, 2, 3] {
        let old = secular
            .coherence(
                &seqs,
                &Cce {
                    order,
                    cutoff: 8e-10,
                },
            )
            .unwrap();
        let new = general
            .coherence(
                &seqs,
                &GeneralisedCce {
                    order,
                    ..Default::default()
                },
            )
            .unwrap();
        for (a, b) in old.iter().zip(&new) {
            assert!((a - b).norm() < 1e-8, "order {order}: {a} vs {b}");
        }
    }
    let old = secular.exact_coherence(&seqs, f64::INFINITY).unwrap();
    let new = general.exact(&seqs, f64::INFINITY, &Pulses::Ideal).unwrap();
    for (a, (b, _)) in old.iter().zip(&new) {
        assert!((a - b).norm() < 1e-8, "exact: {a} vs {b}");
    }
}

#[test]
fn non_secular_couplings_fade_with_the_field() {
    // With the full tensors the electron mixes with the nuclei to second
    // order, A²/(D ± γ_e B): the coherence departs from the secular model
    // by less at higher fields.
    let departure = |field: f64| -> f64 {
        let spins: Vec<BathSpin> = compact()
            .iter()
            .map(|&p| BathSpin::point_dipole(p).unwrap())
            .collect();
        let secular = SpinBath::new(field, -1, spins);
        let general = GeneralisedBath::from_bath(&secular).unwrap();
        let seqs = vec![hahn(0.2e-3), hahn(0.8e-3)];
        let old = secular.exact_coherence(&seqs, f64::INFINITY).unwrap();
        let new = general.exact(&seqs, f64::INFINITY, &Pulses::Ideal).unwrap();
        old.iter()
            .zip(&new)
            .map(|(a, (b, _))| (a - b).norm())
            .fold(0.0, f64::max)
    };
    let (low, high) = (departure(0.5), departure(2.0));
    assert!(low < 1e-4 && high < low / 2.0, "{low} {high}");
}

#[test]
fn finite_pulses_approach_ideal_ones() {
    // The error of finite pulses against ideal ones falls with the Rabi
    // frequency, at 0.3 T where the other transition lies 17 GHz away (far
    // stronger pulses drive it, and the error rises again).
    let b = bath([0.0, 0.0, 0.3], 3);
    let seqs = vec![hahn(0.25e-3), cpmg(2, 0.12e-3)];
    let ideal = b.exact(&seqs, f64::INFINITY, &Pulses::Ideal).unwrap();
    let errors: Vec<f64> = [1e6, 4e6, 16e6]
        .iter()
        .map(|&rabi| {
            let finite = b
                .exact(&seqs, f64::INFINITY, &Pulses::Finite { rabi })
                .unwrap();
            finite
                .iter()
                .zip(&ideal)
                .map(|((_, a), (_, b))| (a - b).abs())
                .fold(0.0, f64::max)
        })
        .collect();
    assert!(errors[0] > errors[1] && errors[1] > errors[2], "{errors:?}");
    assert!(errors[2] < 1e-4, "{errors:?}");
}

#[test]
fn ensembles_and_invalid_input() {
    let seqs = vec![hahn(0.2e-3)];
    let cce = GeneralisedCce::default();
    let runs = generalised_ensemble_signal([0.0, 0.0, 0.05], -1, 0.0107, 1.5e-9, 3, 7, &seqs, &cce)
        .unwrap();
    assert_eq!(runs.len(), 1);
    assert!(runs[0].0 > 0.0 && runs[0].0 <= 1.0 && runs[0].1.is_finite());
    assert!(
        generalised_ensemble_signal([0.0, 0.0, 0.05], -1, 0.0107, 1.5e-9, 0, 7, &seqs, &cce)
            .is_err()
    );
    // The baths of 1.39 at the same seed.
    let spins = diamond_bath(0.0107, 1.5e-9, 3).unwrap();
    assert!(GeneralisedBath::from_bath(&SpinBath::new(0.05, -1, spins)).is_ok());
    let b = bath([0.0, 0.0, 0.05], 3);
    assert!(b.coherence(&[vec![NvEvent::Free(1e-6)]], &cce).is_err());
    assert!(
        b.coherence(
            &seqs,
            &GeneralisedCce {
                order: 9,
                ..cce.clone()
            }
        )
        .is_err()
    );
    assert!(
        b.coherence(
            &seqs,
            &GeneralisedCce {
                pulses: Pulses::Finite { rabi: -1.0 },
                ..cce
            }
        )
        .is_err()
    );
    let mut bad = b.clone();
    bad.field = [f64::NAN, 0.0, 0.0];
    assert!(bad.validate().is_err());
    let mut bad = b.clone();
    bad.transition = 0;
    assert!(bad.validate().is_err());
    assert!(
        bath([0.0, 0.0, 0.05], 6)
            .exact(&seqs, f64::INFINITY, &Pulses::Ideal)
            .is_ok()
    );
    let many = GeneralisedBath::new(
        [0.0, 0.0, 0.05],
        -1,
        (0..9)
            .map(|k| FullSpin::point_dipole([1e-9 + 2e-10 * k as f64, 0.3e-9, 0.5e-9]).unwrap())
            .collect(),
    );
    assert!(many.exact(&seqs, 8e-10, &Pulses::Ideal).is_err());
}

#[test]
fn baths_of_the_lattice_and_given_couplings() {
    // The nuclei of 1.39's baths at the same seed, nested in the radius.
    let ours = GeneralisedBath::diamond([0.0, 0.0, 0.05], -1, 0.0107, 2e-9, 11).unwrap();
    let theirs = diamond_bath(0.0107, 2e-9, 11).unwrap();
    assert_eq!(ours.spins.len(), theirs.len());
    for (a, b) in ours.spins.iter().zip(&theirs) {
        assert_eq!(a.position, b.position);
        assert_eq!(a.hyperfine[2], b.hyperfine);
    }
    let inner = ours.clone().without_spins_within(1e-9);
    assert!(inner.spins.len() < ours.spins.len());
    assert!(
        inner
            .spins
            .iter()
            .all(|s| s.position.iter().map(|x| x * x).sum::<f64>() >= 1e-18)
    );
    // A coupling given for a nucleus next to the vacancy keeps its row; the
    // secular expansion of 1.39 follows it.
    let mut spins: Vec<BathSpin> = compact()
        .iter()
        .map(|&p| BathSpin::point_dipole(p).unwrap())
        .collect();
    spins[0].hyperfine = [2.0e5, -1.0e5, 1.3e6];
    let secular = SpinBath::new(0.05, -1, spins);
    let mut general = GeneralisedBath::from_bath(&secular).unwrap();
    assert_eq!(general.spins[0].hyperfine[2], [2.0e5, -1.0e5, 1.3e6]);
    for s in &mut general.spins {
        s.hyperfine[0] = [0.0; 3];
        s.hyperfine[1] = [0.0; 3];
    }
    let seqs = vec![hahn(0.1e-3), hahn(0.4e-3)];
    let old = secular.exact_coherence(&seqs, f64::INFINITY).unwrap();
    let new = general.exact(&seqs, f64::INFINITY, &Pulses::Ideal).unwrap();
    for (a, (b, _)) in old.iter().zip(&new) {
        assert!((a - b).norm() < 1e-8, "{a} vs {b}");
    }
}

#[test]
fn dressed_levels() {
    // A nucleus 0.25 nm from the vacancy (A_zz = −0.40 MHz) and seven
    // farther out. In the electron's bare frame its first-order mixing
    // (A/Δ) makes the populations oscillate at the electron's frequencies,
    // which the other nuclei dephase and the expansion cannot follow: the
    // error stalls near 10⁻⁵. In the dressed frame it falls with the order.
    let mut positions = compact();
    positions.push([-0.5e-9, -0.4e-9, 0.6e-9]);
    positions.insert(0, [0.2e-9, 0.1e-9, 0.12e-9]);
    let spins: Vec<FullSpin> = positions
        .iter()
        .map(|&p| FullSpin::point_dipole(p).unwrap())
        .collect();
    let seqs: Vec<Vec<NvEvent>> = [20e-6, 60e-6, 120e-6, 200e-6, 400e-6]
        .iter()
        .map(|&t| hahn(t / 2.0))
        .collect();
    let mut b = GeneralisedBath::new([0.0, 0.0, 0.05], -1, spins);
    let errors = |b: &GeneralisedBath| -> (Vec<f64>, Vec<(C64, f64)>) {
        let exact = b.exact(&seqs, f64::INFINITY, &Pulses::Ideal).unwrap();
        let errors = (2..=4)
            .map(|order| {
                let cce = GeneralisedCce {
                    order,
                    cutoff: f64::INFINITY,
                    ..Default::default()
                };
                b.coherence(&seqs, &cce)
                    .unwrap()
                    .iter()
                    .zip(&exact)
                    .map(|(a, (l, _))| (a - l).norm())
                    .fold(0.0, f64::max)
            })
            .collect();
        (errors, exact)
    };
    let (bare, bare_exact) = errors(&b);
    b.dressed = true;
    let (dressed, dressed_exact) = errors(&b);
    assert!(bare[2] > 5e-6, "{bare:?}");
    assert!(
        dressed[0] > dressed[1] && dressed[1] > dressed[2],
        "{dressed:?}"
    );
    assert!(
        dressed[2] < 1e-7 && dressed[2] < bare[2] / 100.0,
        "{dressed:?} {bare:?}"
    );
    // The two frames differ by the mixing, A/Δ ≈ 3·10⁻⁴.
    for ((a, s), (c, t)) in bare_exact.iter().zip(&dressed_exact) {
        assert!(
            (a - c).norm() < 1e-3 && (s - t).abs() < 1e-3,
            "{a} {c} {s} {t}"
        );
    }
    // Secular couplings and a field along the axis mix nothing: the frames
    // agree.
    let mut secular = bath([0.0, 0.0, 0.05], 4);
    for s in &mut secular.spins {
        s.hyperfine[0] = [0.0; 3];
        s.hyperfine[1] = [0.0; 3];
    }
    let cce = GeneralisedCce {
        order: 2,
        ..Default::default()
    };
    let plain = secular.coherence(&seqs, &cce).unwrap();
    secular.dressed = true;
    for (a, b) in secular.coherence(&seqs, &cce).unwrap().iter().zip(&plain) {
        assert!((a - b).norm() < 1e-10, "{a} {b}");
    }
    // At the crossing of m_s = 0 and −1 (D/γ_e = 0.1024 T) the levels mix:
    // the dressed frame does not exist, the bare one does.
    let mut crossing = GeneralisedBath::new([0.0, 0.0, 0.1024], -1, b.spins[..4].to_vec());
    crossing.dressed = true;
    assert!(crossing.coherence(&seqs, &cce).is_err());
    assert!(
        crossing
            .exact(&seqs, f64::INFINITY, &Pulses::Ideal)
            .is_err()
    );
    crossing.dressed = false;
    assert!(crossing.exact(&seqs, f64::INFINITY, &Pulses::Ideal).is_ok());
}

#[test]
fn finite_pulses_on_a_natural_bath() {
    // Finite pulses leave the third level a little population; its elements
    // in the electron alone are tiny and their ratios unstable, so the
    // expansion keeps them out: every order and Rabi frequency stays finite.
    // Sampled bath states put the detuning by all nuclei into every
    // cluster; their estimate stays close to ideal pulses.
    let spins = diamond_bath(0.0107, 2e-9, 1).unwrap();
    let general = GeneralisedBath::from_bath(&SpinBath::new(0.05, -1, spins))
        .unwrap()
        .without_spins_within(5e-10);
    let seqs = vec![cpmg(8, 10e-6), cpmg(16, 10e-6)];
    let ideal = general.signal(&seqs, &GeneralisedCce::default()).unwrap();
    for rabi in [2e6, 10e6, 50e6] {
        for order in [1, 2] {
            let cce = GeneralisedCce {
                order,
                pulses: Pulses::Finite { rabi },
                ..Default::default()
            };
            assert!(
                general
                    .signal(&seqs, &cce)
                    .unwrap()
                    .iter()
                    .all(|s| s.is_finite())
            );
        }
        let sampled = GeneralisedCce {
            pulses: Pulses::Finite { rabi },
            bath_states: 100,
            seed: 2,
            ..Default::default()
        };
        for (e, t) in general
            .estimate(&seqs, &sampled)
            .unwrap()
            .iter()
            .zip(&ideal)
        {
            assert!(
                (e.signal - t).abs() < 5.0 * e.signal_error + 2e-3,
                "{rabi}: {e:?} vs {t}"
            );
        }
    }
}

#[test]
fn sampled_bath_states_are_exact_at_full_order() {
    // In each sampled product state the expansion telescopes to the exact
    // propagation of that state (the mean field of the nuclei outside a
    // cluster vanishes for the whole bath).
    let b = bath([0.003, -0.001, 0.02], 4);
    let seqs = vec![hahn(0.3e-3), ramsey(30e-6), cpmg(2, 0.2e-3)];
    for pulses in [Pulses::Ideal, Pulses::Finite { rabi: 20e6 }] {
        let cce = GeneralisedCce {
            order: 4,
            cutoff: f64::INFINITY,
            pulses: pulses.clone(),
            bath_states: 12,
            seed: 5,
        };
        let parsed = b.parse(&seqs, &pulses).unwrap();
        let (totals, _) = b.expanded(&parsed, &cce).unwrap();
        let fields = b.bath_fields(f64::INFINITY).unwrap();
        let states = b.bath_states(12, 5);
        let electron = b.electron();
        let all = [0, 1, 2, 3];
        for (m, total) in totals.iter().enumerate() {
            let exact = b
                .cluster_states(
                    &all,
                    &parsed,
                    f64::INFINITY,
                    &pulses,
                    &electron,
                    b.frame_frequency(),
                    Some(&fields.mean_field(&states[m], &all)),
                )
                .unwrap();
            for (k, seq) in parsed.iter().enumerate() {
                let (s, t) = (
                    GeneralisedBath::readout(&electron, &total[k], seq.last),
                    GeneralisedBath::readout(&electron, &exact[k], seq.last),
                );
                assert!(
                    (s - t).abs() < 1e-10,
                    "{pulses:?} state {m} sequence {k}: {s} vs {t}"
                );
            }
        }
        // The states are drawn ±½ along the field.
        assert!(states.iter().flatten().all(|&x| x == 0.5 || x == -0.5));
        assert!(states.iter().flatten().any(|&x| x > 0.0));
        assert!(states.iter().flatten().any(|&x| x < 0.0));
    }
}

#[test]
fn sampled_states_converge_with_finite_pulses() {
    // A finite pulse depends on the electron's detuning by all nuclei
    // together: the expansion of unpolarised nuclei moves away from the
    // exact result as the order grows, while that of sampled states, each
    // cluster in the field of the others, stays within its standard error.
    let b = bath([0.0, 0.0, 0.05], 6);
    let seqs = vec![cpmg(8, 10e-6), cpmg(16, 10e-6)];
    let pulses = Pulses::Finite { rabi: 50e6 };
    let exact = b.exact(&seqs, f64::INFINITY, &pulses).unwrap();
    let error = |cce: &GeneralisedCce| -> (f64, f64) {
        let e = b.estimate(&seqs, cce).unwrap();
        let err = e
            .iter()
            .zip(&exact)
            .map(|(a, (_, s))| (a.signal - s).abs())
            .fold(0.0, f64::max);
        (err, e.iter().map(|a| a.signal_error).fold(0.0, f64::max))
    };
    let unpolarised = |order| {
        error(&GeneralisedCce {
            order,
            cutoff: f64::INFINITY,
            pulses: pulses.clone(),
            ..Default::default()
        })
        .0
    };
    let (first, third) = (unpolarised(1), unpolarised(3));
    assert!(third > 3.0 * first && third > 5e-3, "{first} {third}");
    for order in [1, 2, 3] {
        let (err, sd) = error(&GeneralisedCce {
            order,
            cutoff: f64::INFINITY,
            pulses: pulses.clone(),
            bath_states: 400,
            seed: 3,
        });
        assert!(
            sd > 0.0 && err < 5.0 * sd + 2e-4,
            "order {order}: {err} ± {sd}"
        );
        assert!(err < third / 5.0, "order {order}: {err} vs {third}");
    }
}

#[test]
fn finite_pulses_in_tilted_fields() {
    // A transverse field changes how a drive along x couples the qubit and
    // the phases of the electron's eigenstates; finite pulses are scaled to
    // turn the qubit at their Rabi frequency about the axes of the ideal
    // pulses, so the electron alone returns as with ideal ones. (At 70° the
    // other transition lies 0.38 GHz away, and 20 MHz pulses drive it by
    // 2 %; 5 MHz pulses do not.)
    let seqs = vec![
        hahn(50e-6),
        cpmg(4, 20e-6),
        crate::spin::nv::xy8(1, 20e-6),
        ramsey(1e-6),
    ];
    for (degrees, rabi) in [(0.0f64, 20e6), (30.0, 20e6), (45.0, 20e6), (70.0, 5e6)] {
        let a = degrees.to_radians();
        let b = GeneralisedBath::new([0.02 * a.sin(), 0.0, 0.02 * a.cos()], -1, vec![]);
        let ideal = b.exact(&seqs, f64::INFINITY, &Pulses::Ideal).unwrap();
        let finite = b
            .exact(&seqs, f64::INFINITY, &Pulses::Finite { rabi })
            .unwrap();
        for ((_, s), (_, t)) in finite.iter().zip(&ideal) {
            assert!((s - t).abs() < 2e-3, "{degrees}°: {s} vs {t}");
        }
    }
}
