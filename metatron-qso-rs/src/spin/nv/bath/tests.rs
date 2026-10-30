use super::*;
use crate::spin::nv::{Microwave, cpmg, hahn, ramsey, xy8};

fn spin(position: [f64; 3]) -> BathSpin {
    BathSpin::point_dipole(position).unwrap()
}

/// A few ¹³C close together (0.3 to 0.6 nm from each other).
fn compact(n: usize) -> Vec<BathSpin> {
    let points = [
        [0.6, 0.2, 0.4],
        [0.9, 0.1, 0.1],
        [0.5, 0.6, 0.0],
        [0.8, 0.5, 0.5],
        [0.4, 0.3, 0.8],
        [1.1, 0.4, 0.3],
        [0.7, 0.8, 0.6],
        [1.0, 0.7, 0.0],
    ];
    points[..n]
        .iter()
        .map(|p| spin(p.map(|x| x * 1e-9)))
        .collect()
}

fn sequences(tau: f64) -> Vec<Vec<NvEvent>> {
    vec![
        ramsey(tau),
        hahn(tau),
        cpmg(4, tau / 4.0),
        xy8(1, tau / 8.0),
        // Arbitrary end pulses and phases.
        vec![
            NvEvent::Pulse {
                angle: 70.0,
                phase: 30.0,
            },
            NvEvent::Free(0.3 * tau),
            NvEvent::Pulse {
                angle: 180.0,
                phase: 45.0,
            },
            NvEvent::Free(0.5 * tau),
            NvEvent::Pulse {
                angle: -180.0,
                phase: -20.0,
            },
            NvEvent::Free(0.2 * tau),
            NvEvent::Pulse {
                angle: 110.0,
                phase: -60.0,
            },
        ],
    ]
}

#[test]
fn without_couplings_it_is_the_nv_centre_of_1_37() {
    // Three ¹³C without dipolar couplings: the expansion is exact at first
    // order and must reproduce the full propagation of NvCenter, whose
    // A_⊥ lies along x (the azimuth of each nucleus is irrelevant then).
    let spins = vec![
        BathSpin {
            position: [1e-9, 0.0, 0.0],
            hyperfine: [30e3, -20e3, 60e3],
        },
        BathSpin {
            position: [0.0, 1e-9, 0.0],
            hyperfine: [-5e3, 12e3, -25e3],
        },
        BathSpin {
            position: [0.0, 0.0, 1e-9],
            hyperfine: [80e3, 0.0, 15e3],
        },
    ];
    for field in [0.003, 0.03, 0.2] {
        for transition in [-1i8, 1] {
            let bath = SpinBath::new(field, transition, spins.clone());
            let mut nv = NvCenter::new(field);
            for s in &spins {
                let [x, y, z] = s.hyperfine;
                nv = nv.with_carbon(z, x.hypot(y));
            }
            let drive = Microwave::resonant(&nv, transition, None);
            for tau in [3e-6, 17e-6, 40e-6] {
                let seqs = sequences(tau);
                let ours = bath
                    .signal(
                        &seqs,
                        &Cce {
                            order: 1,
                            cutoff: 0.0,
                        },
                    )
                    .unwrap();
                for (s, x) in seqs.iter().zip(&ours) {
                    let want = nv.signal(s, &drive).unwrap();
                    assert!(
                        (x - want).abs() < 1e-10,
                        "B = {field}, m = {transition}: {x} vs {want}"
                    );
                }
            }
        }
    }
}

/// `Σ_i c_i ⊗` the 2 × 2 `op` on spin `i` of `k`, by Kronecker products
/// (spin 0 the least significant).
fn embed(k: usize, i: usize, op: &CMatrix) -> CMatrix {
    let id = CMatrix::identity(2, 2);
    (0..k).rev().fold(CMatrix::identity(1, 1), |acc, j| {
        acc.kronecker(if j == i { op } else { &id })
    })
}

/// The full propagation of the electron qubit `(|m⟩, |0⟩)` and the bath,
/// written independently of the module: `H = |m⟩⟨m| ⊗ H_m + |0⟩⟨0| ⊗
/// H_0`, pulses as rotations of the qubit.
fn dense_signal(bath: &SpinBath, events: &[NvEvent]) -> f64 {
    let k = bath.spins.len();
    let d = 1usize << k;
    let c = |x: f64| C64::new(x, 0.0);
    let sx = CMatrix::from_row_slice(2, 2, &[c(0.0), c(0.5), c(0.5), c(0.0)]);
    let sy = CMatrix::from_row_slice(
        2,
        2,
        &[c(0.0), C64::new(0.0, -0.5), C64::new(0.0, 0.5), c(0.0)],
    );
    let sz = CMatrix::from_row_slice(2, 2, &[c(0.5), c(0.0), c(0.0), c(-0.5)]);
    let ops = [sx, sy, sz];
    let gc = Isotope::C13.gamma() / TWO_PI;
    let mut base = CMatrix::zeros(d, d);
    for i in 0..k {
        base += embed(k, i, &ops[2]) * c(-TWO_PI * gc * bath.field);
        for j in i + 1..k {
            let (p, q) = (bath.spins[i].position, bath.spins[j].position);
            let v = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
            let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let dd = MU0_OVER_4PI * gc * gc * TWO_PI * HBAR / r.powi(3);
            for a in 0..3 {
                for b in 0..3 {
                    let w = f64::from(u8::from(a == b)) - 3.0 * v[a] * v[b] / (r * r);
                    base += embed(k, i, &ops[a]) * embed(k, j, &ops[b]) * c(TWO_PI * dd * w);
                }
            }
        }
    }
    let mut hm = base.clone();
    for i in 0..k {
        for a in 0..3 {
            hm += embed(k, i, &ops[a])
                * c(TWO_PI * f64::from(bath.transition) * bath.spins[i].hyperfine[a]);
        }
    }
    let pm = CMatrix::from_row_slice(2, 2, &[c(1.0), c(0.0), c(0.0), c(0.0)]);
    let p0 = CMatrix::from_row_slice(2, 2, &[c(0.0), c(0.0), c(0.0), c(1.0)]);
    let h = pm.kronecker(&hm) + p0.kronecker(&base);
    let mut rho = p0.kronecker(&(CMatrix::identity(d, d) * c(1.0 / d as f64)));
    for e in events {
        let u = match *e {
            NvEvent::Free(t) => crate::linalg::expm_hermitian(&h, t),
            NvEvent::Pulse { angle, phase } => {
                let g = CMatrix::from_row_slice(
                    2,
                    2,
                    &[
                        c(0.0),
                        C64::from_polar(1.0, -phase.to_radians()),
                        C64::from_polar(1.0, phase.to_radians()),
                        c(0.0),
                    ],
                );
                crate::linalg::expm_hermitian(&g, angle.to_radians() / 2.0)
                    .kronecker(&CMatrix::identity(d, d))
            }
        };
        rho = &u * &rho * u.adjoint();
    }
    (d..2 * d).map(|i| rho[(i, i)].re).sum()
}

#[test]
fn the_exact_propagation_matches_an_independent_dense_one() {
    for transition in [-1i8, 1] {
        let bath = SpinBath::new(0.02, transition, compact(4));
        for tau in [20e-6, 150e-6] {
            let seqs = sequences(tau);
            let l = bath.exact_coherence(&seqs, f64::INFINITY).unwrap();
            for ((s, l), b) in seqs.iter().zip(&l).zip(branches_of(&seqs)) {
                let want = dense_signal(&bath, s);
                let got = b.signal(*l);
                assert!((got - want).abs() < 1e-10, "{got} vs {want}");
            }
        }
    }
}

fn branches_of(seqs: &[Vec<NvEvent>]) -> Vec<Branches> {
    seqs.iter().map(|s| branches(s).unwrap()).collect()
}

#[test]
fn standard_sequences_read_out_half_of_one_plus_the_coherence() {
    for s in [ramsey(1e-6), hahn(1e-6), cpmg(3, 1e-6), xy8(2, 1e-6)] {
        let b = branches(&s).unwrap();
        for l in [C64::new(0.3, 0.4), C64::new(-0.7, 0.1)] {
            assert!((b.signal(l) - (1.0 + l.re) / 2.0).abs() < 1e-15);
        }
    }
}

#[test]
fn the_expansion_is_exact_at_the_size_of_the_bath() {
    for n in [6, 8] {
        let bath = SpinBath::new(0.01, -1, compact(n));
        let seqs: Vec<Vec<NvEvent>> = [0.2e-3, 0.6e-3, 1.5e-3]
            .iter()
            .flat_map(|&t| [hahn(t / 2.0), cpmg(2, t / 4.0)])
            .collect();
        let exact = bath.exact_coherence(&seqs, f64::INFINITY).unwrap();
        let mut errors = Vec::new();
        for order in 1..=n {
            let l = bath
                .coherence(
                    &seqs,
                    &Cce {
                        order,
                        cutoff: f64::INFINITY,
                    },
                )
                .unwrap();
            errors.push(
                l.iter()
                    .zip(&exact)
                    .map(|(a, b)| (a - b).norm())
                    .fold(0.0, f64::max),
            );
        }
        assert!(errors[n - 1] < 1e-10, "n = {n}: {errors:?}");
        // The truncations differ from it: the test is not trivial.
        assert!(errors[0] > 1e-4, "n = {n}: {errors:?}");
        assert!(errors[n - 2] > 1e-12, "n = {n}: {errors:?}");
    }
}

#[test]
fn clusters_that_are_not_connected_factorise() {
    // Two groups of three far apart: with a cutoff that separates them,
    // the expansion to third order is exact.
    let mut spins = compact(3);
    spins.extend(
        compact(3)
            .into_iter()
            .map(|s| spin([s.position[0] + 5e-9, s.position[1], s.position[2]])),
    );
    let bath = SpinBath::new(0.01, -1, spins);
    let cutoff = 1e-9;
    assert_eq!(bath.clusters(3, cutoff).unwrap().len(), 6 + 6 + 2);
    let seqs = vec![hahn(0.3e-3), hahn(0.8e-3)];
    let exact = bath.exact_coherence(&seqs, cutoff).unwrap();
    let l = bath.coherence(&seqs, &Cce { order: 3, cutoff }).unwrap();
    for (a, b) in l.iter().zip(&exact) {
        assert!((a - b).norm() < 1e-12, "{a} vs {b}");
    }
}

#[test]
fn the_lattice_and_the_point_dipole() {
    let a = DIAMOND_LATTICE_CONSTANT;
    let nearest = a * 3f64.sqrt() / 4.0;
    let sites = diamond_sites(2e-9).unwrap();
    // 8 atoms per a³, less the vacancy and the nitrogen.
    let expected = 8.0 * 4.0 / 3.0 * std::f64::consts::PI * (2e-9 / a).powi(3) - 2.0;
    assert!(
        (sites.len() as f64 / expected - 1.0).abs() < 0.03,
        "{} vs {expected}",
        sites.len()
    );
    let first: Vec<&[f64; 3]> = sites
        .iter()
        .filter(|p| norm(**p) < nearest * 1.01)
        .collect();
    assert_eq!(first.len(), 3);
    for p in &first {
        assert!((norm(**p) - nearest).abs() < 1e-15);
        // The three carbons next to the vacancy, 109.47° from the axis.
        assert!((p[2] / nearest + 1.0 / 3.0).abs() < 1e-12);
    }
    let min = sites
        .iter()
        .enumerate()
        .flat_map(|(i, p)| {
            sites[i + 1..]
                .iter()
                .map(move |q| norm([p[0] - q[0], p[1] - q[1], p[2] - q[2]]))
        })
        .fold(f64::INFINITY, f64::min);
    assert!((min - nearest).abs() < 1e-15);
    // On the axis A_zz = 2K, at right angles −K, K = 19.89 kHz at 1 nm.
    let k = hyperfine_scale() / 1e-27;
    assert!((k - 19_885.0).abs() < 5.0, "{k}");
    let on = spin([0.0, 0.0, 1e-9]).hyperfine;
    assert!((on[2] - 2.0 * k).abs() < 1e-9 && on[0] == 0.0 && on[1] == 0.0);
    let off = spin([1e-9, 0.0, 0.0]).hyperfine;
    assert!((off[2] + k).abs() < 1e-9);
    // The abundance.
    let all = diamond_bath(1.0, 2e-9, 1).unwrap();
    assert_eq!(all.len(), sites.len());
    let some = diamond_bath(0.1, 2e-9, 7).unwrap();
    let mean = 0.1 * sites.len() as f64;
    assert!((some.len() as f64 - mean).abs() < 4.0 * (mean * 0.9).sqrt());
}

#[test]
fn gaussian_noise_multiplies_the_coherence() {
    let bath = SpinBath::new(0.05, -1, compact(3));
    let noise = FieldNoise::OrnsteinUhlenbeck {
        sigma: 20e3,
        correlation_time: 30e-6,
    };
    let seqs = vec![hahn(20e-6), cpmg(4, 10e-6)];
    let cce = Cce {
        order: 3,
        cutoff: f64::INFINITY,
    };
    let l = bath.coherence(&seqs, &cce).unwrap();
    let with = bath.signal_with_noise(&seqs, &cce, &noise, 0, 0).unwrap();
    for ((s, l), (x, err)) in seqs.iter().zip(&l).zip(&with) {
        let want = (1.0 + (l * (-decay_exponent(s, &noise).unwrap()).exp()).re) / 2.0;
        assert!((x - want).abs() < 1e-14 && *err == 0.0);
    }
    // Without a bath it is the noise alone, as sampled by NvCenter.
    let empty = SpinBath::new(0.05, -1, Vec::new());
    let nv = NvCenter::new(0.05);
    let drive = Microwave::resonant(&nv, -1, None);
    let ours = empty.signal_with_noise(&seqs, &cce, &noise, 0, 0).unwrap();
    for (s, (x, _)) in seqs.iter().zip(&ours) {
        let (y, e) = nv.signal_with_noise(s, &drive, &noise, 4000, 5).unwrap();
        assert!((x - y).abs() < 4.0 * e, "{x} vs {y} ± {e}");
    }
}

#[test]
fn telegraph_noise_is_sampled() {
    let noise = FieldNoise::Telegraph {
        amplitude: 30e3,
        rate: 2e4,
    };
    let empty = SpinBath::new(0.05, 1, Vec::new());
    let nv = NvCenter::new(0.05);
    let drive = Microwave::resonant(&nv, 1, None);
    let seqs = vec![hahn(15e-6), ramsey(8e-6)];
    let ours = empty
        .signal_with_noise(&seqs, &Cce::default(), &noise, 4000, 11)
        .unwrap();
    for (s, (x, e)) in seqs.iter().zip(&ours) {
        let (y, f) = nv.signal_with_noise(s, &drive, &noise, 4000, 3).unwrap();
        assert!((x - y).abs() < 4.0 * e.hypot(f), "{x} ± {e} vs {y} ± {f}");
    }
    // With a bath the sampled factor multiplies its coherence: the mean
    // over paths of Re(L e^{−iφ}) for a real L.
    let bath = SpinBath::new(0.05, 1, compact(2));
    let l = bath.coherence(&seqs[..1], &Cce::default()).unwrap()[0];
    let (with, e) = bath
        .signal_with_noise(&seqs[..1], &Cce::default(), &noise, 4000, 11)
        .unwrap()[0];
    let (alone, _) = ours[0];
    // Same seed, same paths: the real part of the factor is shared, its
    // imaginary part averages to zero (the telegraph is symmetric).
    let factor = 2.0 * alone - 1.0;
    assert!(
        (with - (1.0 + l.re * factor) / 2.0).abs() < 4.0 * e,
        "{with} ± {e}"
    );
}

#[test]
fn rejects_invalid_input() {
    let bath = SpinBath::new(0.01, -1, compact(3));
    let cce = Cce::default();
    let bad = vec![vec![
        NvEvent::Pulse {
            angle: 90.0,
            phase: 0.0,
        },
        NvEvent::Pulse {
            angle: 90.0,
            phase: 0.0,
        },
        NvEvent::Pulse {
            angle: 90.0,
            phase: 0.0,
        },
    ]];
    assert!(bath.coherence(&bad, &cce).is_err());
    assert!(bath.coherence(&[vec![NvEvent::Free(1e-6)]], &cce).is_err());
    assert!(bath.coherence(&[ramsey(-1.0)], &cce).is_err());
    assert!(
        bath.coherence(
            &[hahn(1e-6)],
            &Cce {
                order: 0,
                cutoff: 1e-9
            }
        )
        .is_err()
    );
    assert!(
        bath.coherence(
            &[hahn(1e-6)],
            &Cce {
                order: 2,
                cutoff: f64::NAN
            }
        )
        .is_err()
    );
    assert!(
        SpinBath::new(0.01, 0, compact(2))
            .coherence(&[hahn(1e-6)], &cce)
            .is_err()
    );
    let mut twice = compact(2);
    twice.push(twice[0]);
    assert!(
        SpinBath::new(0.01, -1, twice)
            .coherence(&[hahn(1e-6)], &cce)
            .is_err()
    );
    let many = SpinBath::new(0.01, -1, diamond_bath(1.0, 0.5e-9, 1).unwrap());
    assert!(many.spins.len() > MAX_EXACT_BATH);
    assert!(many.exact_coherence(&[hahn(1e-6)], 1e-9).is_err());
    assert!(BathSpin::point_dipole([0.0; 3]).is_err());
    assert!(diamond_bath(0.0, 1e-9, 1).is_err());
    assert!(diamond_bath(0.5, 1.0, 1).is_err());
    assert!(ensemble_signal(0.01, -1, 0.01, 1e-9, 0, 1, &[hahn(1e-6)], &cce).is_err());
}
