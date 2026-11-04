//! Reactions of two spin-carrying reactants: products formed in the product
//! of their states and split by partial traces, against 1.50 with a partner
//! held in its state, the equations written out and integrated explicitly,
//! molecules reacting at random, the closed-form kinetics of `A + B ⇌ C`,
//! conservation laws and a steady state found by Newton's method.

use super::*;
use crate::rng::Rng;
use crate::spin::{Frame, Isotope};
use nalgebra::DVector;

const S: fn(usize) -> Participant = Participant::Site;
const P: fn(usize) -> Participant = Participant::Species;

fn reaction(reactants: &[Participant], products: &[Participant], rate: f64) -> Reaction {
    Reaction {
        reactants: reactants.to_vec(),
        products: products.to_vec(),
        rate,
        carried: Vec::new(),
        fresh: Fresh::Thermal,
    }
}

fn route(reactant: usize, product: usize, spins: &[(usize, usize)]) -> Route {
    Route {
        reactant,
        product,
        spins: spins.to_vec(),
    }
}

/// A system at 1 T in the rotating frame: `(isotope, shift, T₁, T₂)` per
/// spin (no relaxation for `T₁ = 0`) and couplings (Hz).
fn system(spins: &[(Isotope, f64, f64, f64)], couplings: &[(usize, usize, f64)]) -> SpinSystem {
    let mut sys = SpinSystem::new(1.0);
    for &(x, shift, t1, t2) in spins {
        let i = sys.add_spin(x, shift).unwrap();
        if t1 > 0.0 {
            sys.set_relaxation(i, Some(t1), Some(t2)).unwrap();
        }
    }
    for &(i, j, hz) in couplings {
        sys.couple(i, j, hz).unwrap();
    }
    sys
}

fn levels(sys: &SpinSystem) -> Vec<usize> {
    sys.spins().iter().map(|s| s.isotope.levels()).collect()
}

/// The levels of every spin in basis state `a`.
fn digits(levels: &[usize], mut a: usize) -> Vec<usize> {
    let mut out = vec![0; levels.len()];
    for i in (0..levels.len()).rev() {
        out[i] = a % levels[i];
        a /= levels[i];
    }
    out
}

/// `Tr` over the spins of `sys` not in `keep`, the kept ones in the order
/// of `keep`.
fn partial(sys: &SpinSystem, rho: &CMatrix, keep: &[usize]) -> CMatrix {
    let lv = levels(sys);
    let dk: usize = keep.iter().map(|&i| lv[i]).product();
    let index = |dg: &[usize]| keep.iter().fold(0, |acc, &i| acc * lv[i] + dg[i]);
    let mut out = CMatrix::zeros(dk, dk);
    for a in 0..sys.dimension() {
        let da = digits(&lv, a);
        for b in 0..sys.dimension() {
            let db = digits(&lv, b);
            if (0..lv.len()).any(|i| !keep.contains(&i) && da[i] != db[i]) {
                continue;
            }
            out[(index(&da), index(&db))] += rho[(a, b)];
        }
    }
    out
}

/// The matrix of `sys` that is the product of `parts`, each a matrix on
/// the spins it lists (together every spin once).
fn embed(sys: &SpinSystem, parts: &[(CMatrix, Vec<usize>)]) -> CMatrix {
    let lv = levels(sys);
    let d = sys.dimension();
    CMatrix::from_fn(d, d, |a, b| {
        let (da, db) = (digits(&lv, a), digits(&lv, b));
        parts.iter().fold(C64::new(1.0, 0.0), |v, (m, spins)| {
            let ia = spins.iter().fold(0, |acc, &i| acc * lv[i] + da[i]);
            let ib = spins.iter().fold(0, |acc, &i| acc * lv[i] + db[i]);
            v * m[(ia, ib)]
        })
    })
}

/// `dσ/dt` written out: each site's evolution and relaxation (its
/// generator, from the base), the losses `k Π_other c · σ`, and each
/// product site's gain `k Π_unrouted c · ⊗_routed Tr(σ)`, or the velocity
/// times its thermal or parahydrogen state when no spin is routed into it.
/// Amounts are the traces; sites only.
fn derivative(r: &Reactions) -> impl Fn(&[CMatrix]) -> Vec<CMatrix> + '_ {
    let base = r.base(r.field()).unwrap();
    let ones: Vec<f64> = std::iter::once(1.0)
        .chain(base.members.iter().map(|_| 0.0))
        .collect();
    let g: Vec<CMatrix> = base
        .family
        .iter()
        .map(|f| f.combine(&ones).to_dense())
        .collect();
    let lists: Vec<Vec<(usize, usize, usize)>> = base
        .joint
        .blocks
        .iter()
        .map(|(_, parts, _)| {
            let mut e = Vec::new();
            for &(s, k, _) in parts {
                for &(a, b) in &base.joint.liouvilles[s].blocks[k].1 {
                    e.push((s, a, b));
                }
            }
            e
        })
        .collect();
    let drift: Vec<Vec<(usize, usize, C64)>> = base.drift.clone();
    move |sig: &[CMatrix]| {
        let c: Vec<f64> = sig.iter().map(|m| m.trace().re).collect();
        let x = deviations(sig, &c);
        let mut out: Vec<CMatrix> = sig
            .iter()
            .map(|m| CMatrix::zeros(m.nrows(), m.ncols()))
            .collect();
        for (j, list) in lists.iter().enumerate() {
            let v = DVector::from_iterator(list.len(), list.iter().map(|&(s, a, b)| x[s][(a, b)]));
            let y = &g[j] * v;
            for (k, &(s, a, b)) in list.iter().enumerate() {
                out[s][(a, b)] += y[k];
            }
        }
        for (s, here) in drift.iter().enumerate() {
            for &(j, i, v) in here {
                let (s2, a, b) = lists[j][i];
                assert_eq!(s, s2);
                out[s][(a, b)] += v * c[s];
            }
        }
        let amount = |p: &Participant| match *p {
            Participant::Site(s) => c[s],
            Participant::Species(_) => panic!("sites only"),
        };
        for (k, reaction) in r.reactions.iter().enumerate() {
            let v = reaction
                .reactants
                .iter()
                .fold(reaction.rate, |v, p| v * amount(p));
            for (q, p) in reaction.reactants.iter().enumerate() {
                let Participant::Site(s) = *p else { continue };
                let kappa = reaction
                    .reactants
                    .iter()
                    .enumerate()
                    .filter(|&(o, _)| o != q)
                    .fold(reaction.rate, |v, (_, p)| v * amount(p));
                out[s] -= &sig[s] * C64::new(kappa, 0.0);
            }
            for (pp, p) in reaction.products.iter().enumerate() {
                let Participant::Site(target) = *p else {
                    continue;
                };
                let sys = &r.sites[target].system;
                let routes: Vec<&Route> = r.routes(k).iter().filter(|x| x.product == pp).collect();
                if routes.is_empty() {
                    let fresh = match reaction.fresh {
                        Fresh::Thermal => sys.thermal_state().unwrap(),
                        Fresh::Parahydrogen {
                            spins: [i, j],
                            fraction,
                        } => {
                            assert_eq!(sys.spins().len(), 2);
                            sys.parahydrogen_state(i, j, fraction).unwrap()
                        }
                    };
                    out[target] += fresh * C64::new(v, 0.0);
                    continue;
                }
                let mut parts = Vec::new();
                let mut scale = reaction.rate;
                for (q, rp) in reaction.reactants.iter().enumerate() {
                    match routes.iter().find(|x| x.reactant == q) {
                        Some(route) => {
                            let Participant::Site(s) = *rp else {
                                unreachable!()
                            };
                            let keep: Vec<usize> = route.spins.iter().map(|&(i, _)| i).collect();
                            let onto: Vec<usize> = route.spins.iter().map(|&(_, j)| j).collect();
                            parts.push((partial(&r.sites[s].system, &sig[s], &keep), onto));
                        }
                        None => scale *= amount(rp),
                    }
                }
                let routed: usize = parts.iter().map(|(_, o)| o.len()).sum();
                assert_eq!(routed, sys.spins().len(), "every product spin routed");
                out[target] += embed(sys, &parts) * C64::new(scale, 0.0);
            }
        }
        out
    }
}

fn axpy(a: &[CMatrix], b: &[CMatrix], h: f64) -> Vec<CMatrix> {
    a.iter()
        .zip(b)
        .map(|(x, y)| x + y * C64::new(h, 0.0))
        .collect()
}

/// σ after `t` by `steps` steps of the classical Runge–Kutta method on
/// [`derivative`].
fn explicit(r: &Reactions, sigma: &[CMatrix], t: f64, steps: usize) -> Vec<CMatrix> {
    let f = derivative(r);
    let dt = t / steps as f64;
    let mut s = sigma.to_vec();
    for _ in 0..steps {
        let k1 = f(&s);
        let k2 = f(&axpy(&s, &k1, dt / 2.0));
        let k3 = f(&axpy(&s, &k2, dt / 2.0));
        let k4 = f(&axpy(&s, &k3, dt));
        for i in 0..s.len() {
            s[i] += (&k1[i] + &k2[i] * C64::new(2.0, 0.0) + &k3[i] * C64::new(2.0, 0.0) + &k4[i])
                * C64::new(dt / 6.0, 0.0);
        }
    }
    s
}

/// The largest difference of two sets of states relative to the largest
/// deviation from `c/d` of `want`.
fn relative(got: &[CMatrix], want: &[CMatrix]) -> f64 {
    let c: Vec<f64> = want.iter().map(|m| m.trace().re).collect();
    let scale = deviations(want, &c)
        .iter()
        .map(|x| x.norm())
        .fold(0.0, f64::max);
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max)
        / scale
}

/// `1/d + Σ w_k O_k` of `sys`.
fn state(sys: &SpinSystem, parts: &[(CMatrix, f64)]) -> CMatrix {
    let mut rho = sys.identity_state();
    for (op, w) in parts {
        rho += op * C64::new(*w, 0.0);
    }
    rho
}

/// A (¹H) + B (¹³C, ¹H) ⇌ C with relaxation everywhere: both reactants
/// carry coherences of order zero and not, at frequencies of a few hertz.
fn exchange_model() -> (Reactions, Vec<CMatrix>) {
    let a = system(&[(Isotope::H1, 0.1, 2.0, 1.0)], &[]);
    let b = system(
        &[(Isotope::C13, -0.4, 1.5, 0.8), (Isotope::H1, 0.3, 2.5, 1.2)],
        &[(0, 1, 8.0)],
    );
    let c = system(
        &[
            (Isotope::H1, 0.15, 1.2, 0.6),
            (Isotope::C13, -0.3, 1.8, 0.9),
            (Isotope::H1, 0.25, 2.0, 0.7),
        ],
        &[(0, 1, 4.0), (1, 2, 8.0), (0, 2, 3.0)],
    );
    let mut r = Reactions::new();
    let ia = r.add_site(a.clone(), 1.0).unwrap();
    let ib = r.add_site(b.clone(), 0.8).unwrap();
    let ic = r.add_site(c.clone(), 0.1).unwrap();
    r.add_routed_reaction(
        reaction(&[S(ia), S(ib)], &[S(ic)], 3.0),
        vec![route(0, 0, &[(0, 0)]), route(1, 0, &[(0, 1), (1, 2)])],
    )
    .unwrap();
    r.add_routed_reaction(
        reaction(&[S(ic)], &[S(ia), S(ib)], 1.5),
        vec![route(0, 0, &[(0, 0)]), route(0, 1, &[(1, 0), (2, 1)])],
    )
    .unwrap();
    let rho_a = state(&a, &[(a.ix(0).unwrap(), 0.4), (a.iz(0).unwrap(), 0.1)]);
    let zq = b.iplus(0).unwrap() * b.iplus(1).unwrap().adjoint();
    let rho_b = state(
        &b,
        &[
            (b.iz(0).unwrap(), 0.2),
            (b.iy(1).unwrap(), 0.15),
            (&zq + zq.adjoint(), 0.05),
        ],
    );
    let rho_c = state(&c, &[(c.ix(0).unwrap(), 0.1), (c.iz(1).unwrap(), 0.2)]);
    let sigma = vec![
        rho_a * C64::new(1.0, 0.0),
        rho_b * C64::new(0.8, 0.0),
        rho_c * C64::new(0.1, 0.0),
    ];
    (r, sigma)
}

#[test]
fn a_partner_held_in_its_state_equals_the_reactions_of_1_50() {
    // A + H₂ → C with H₂ a site in the parahydrogen state (no relaxation,
    // equivalent protons: it stays so, through pulses too) equals 1.50's
    // A + H₂ with H₂ a species and its spins fresh parahydrogen.
    let a = system(&[(Isotope::H1, 0.5, 3.0, 1.0)], &[]);
    let h2 = system(
        &[(Isotope::H1, 0.2, 0.0, 0.0), (Isotope::H1, 0.2, 0.0, 0.0)],
        &[(0, 1, 5.0)],
    );
    let c = system(
        &[
            (Isotope::H1, 0.6, 2.0, 0.7),
            (Isotope::H1, 0.1, 2.0, 0.7),
            (Isotope::H1, 0.4, 2.0, 0.7),
        ],
        &[(0, 1, 7.0), (0, 2, 3.0), (1, 2, -2.0)],
    );
    let fraction = 0.9;
    let rho_a = state(&a, &[(a.ix(0).unwrap(), 0.3), (a.iy(0).unwrap(), 0.1)]);
    let para = h2.parahydrogen_state(0, 1, fraction).unwrap();
    let events = [
        Event::Delay(0.3),
        Event::pulse(Target::Isotope(Isotope::H1), 90.0, 0.0),
        Event::Delay(0.2),
    ];

    let mut routed = Reactions::new();
    let ia = routed.add_site(a.clone(), 1.0).unwrap();
    let ih = routed.add_site(h2.clone(), 0.8).unwrap();
    let ic = routed.add_site(c.clone(), 0.0).unwrap();
    routed
        .add_routed_reaction(
            reaction(&[S(ia), S(ih)], &[S(ic)], 2.5),
            vec![route(0, 0, &[(0, 0)]), route(1, 0, &[(0, 1), (1, 2)])],
        )
        .unwrap();
    let got = routed
        .evolve(
            &[
                rho_a.clone(),
                &para * C64::new(0.8, 0.0),
                CMatrix::zeros(8, 8),
            ],
            &events,
        )
        .unwrap();

    let mut fixed = Reactions::new();
    let ja = fixed.add_site(a.clone(), 1.0).unwrap();
    let jc = fixed.add_site(c.clone(), 0.0).unwrap();
    let g = fixed.add_species("H2", 0.8).unwrap();
    let mut rx = reaction(&[S(ja), P(g)], &[S(jc)], 2.5);
    rx.carried = vec![(0, 0)];
    rx.fresh = Fresh::Parahydrogen {
        spins: [1, 2],
        fraction,
    };
    fixed.add_reaction(rx).unwrap();
    let want = fixed
        .evolve(&[rho_a, CMatrix::zeros(8, 8)], &events)
        .unwrap();
    let err = relative(&[got[0].clone(), got[2].clone()], &want);
    assert!(err < 1e-9, "routed against 1.50: {err:.2e}");
    // The hydrogen stayed parahydrogen.
    let left = routed.amounts(0.5).unwrap()[1];
    let held = (&got[1] - &para * C64::new(left, 0.0)).norm();
    assert!(held < 1e-10, "{held:.2e}");

    // A (¹H, laboratory frame at 50 µT) + B (one ¹³C, thermal with T₁:
    // it stays thermal) → C equals 1.50 with B a species and its spin
    // fresh and thermal.
    let lab = |s: SpinSystem| s.with_field(50e-6).unwrap().with_frame(Frame::Laboratory);
    let a = lab(system(&[(Isotope::H1, 1.0, 2.0, 1.0)], &[]));
    let b = lab(system(&[(Isotope::C13, 0.0, 1.0, 1.0)], &[]));
    let c = lab(system(
        &[(Isotope::H1, 2.0, 1.5, 0.8), (Isotope::C13, 0.0, 2.0, 1.0)],
        &[(0, 1, 140.0)],
    ));
    let rho_a = state(&a, &[(a.ix(0).unwrap(), 0.4)]);
    let mut routed = Reactions::new();
    let ia = routed.add_site(a.clone(), 1.0).unwrap();
    let ib = routed.add_site(b.clone(), 0.6).unwrap();
    let ic = routed.add_site(c.clone(), 0.0).unwrap();
    routed
        .add_routed_reaction(
            reaction(&[S(ia), S(ib)], &[S(ic)], 3.0),
            vec![route(0, 0, &[(0, 0)]), route(1, 0, &[(0, 1)])],
        )
        .unwrap();
    let thermal = b.thermal_state().unwrap();
    let got = routed
        .evolve(
            &[
                rho_a.clone(),
                &thermal * C64::new(0.6, 0.0),
                CMatrix::zeros(4, 4),
            ],
            &[Event::Delay(0.4)],
        )
        .unwrap();
    let mut fixed = Reactions::new();
    let ja = fixed.add_site(a, 1.0).unwrap();
    let jc = fixed.add_site(c, 0.0).unwrap();
    let g = fixed.add_species("B", 0.6).unwrap();
    let mut rx = reaction(&[S(ja), P(g)], &[S(jc)], 3.0);
    rx.carried = vec![(0, 0)];
    fixed.add_reaction(rx).unwrap();
    let want = fixed
        .evolve(&[rho_a, CMatrix::zeros(4, 4)], &[Event::Delay(0.4)])
        .unwrap();
    let err = relative(&[got[0].clone(), got[2].clone()], &want);
    assert!(err < 1e-9, "thermal partner against 1.50: {err:.2e}");
}

#[test]
fn products_of_two_changing_states_follow_the_equations_written_out() {
    let (r, sigma) = exchange_model();
    assert_eq!(r.routes(0).len(), 2, "the first reaction keeps its routes");
    let t = 0.5;
    let want = explicit(&r, &sigma, t, 2000);
    let got = r.evolve(&sigma, &[Event::Delay(t)]).unwrap();
    let err = relative(&got, &want);
    assert!(err < 1e-7, "A + B ⇌ C: {err:.2e}");
    // On sparse blocks by Krylov steps alike.
    let krylov = r
        .evolve_with(&sigma, &[Event::Delay(t)], Propagation::Krylov)
        .unwrap();
    let err = relative(&krylov, &got);
    assert!(err < 1e-8, "Krylov against dense: {err:.2e}");
    // Traces are the amounts.
    let amounts = r.amounts(t).unwrap();
    for (m, c) in got.iter().zip(&amounts) {
        assert!((m.trace().re - c).abs() < 1e-12, "{} vs {c}", m.trace().re);
    }

    // 2A ⇌ D: two molecules of one site, each carrying its spin.
    let a = system(&[(Isotope::H1, 0.2, 2.0, 1.0)], &[]);
    let d = system(
        &[(Isotope::H1, 0.1, 1.5, 0.7), (Isotope::H1, -0.2, 1.5, 0.7)],
        &[(0, 1, 6.0)],
    );
    let mut r = Reactions::new();
    let ia = r.add_site(a.clone(), 1.5).unwrap();
    let id = r.add_site(d.clone(), 0.0).unwrap();
    r.add_routed_reaction(
        reaction(&[S(ia), S(ia)], &[S(id)], 1.2),
        vec![route(0, 0, &[(0, 0)]), route(1, 0, &[(0, 1)])],
    )
    .unwrap();
    r.add_routed_reaction(
        reaction(&[S(id)], &[S(ia), S(ia)], 0.8),
        vec![route(0, 0, &[(0, 0)]), route(0, 1, &[(1, 0)])],
    )
    .unwrap();
    let sigma = vec![
        state(&a, &[(a.ix(0).unwrap(), 0.35), (a.iz(0).unwrap(), 0.1)]) * C64::new(1.5, 0.0),
        CMatrix::zeros(4, 4),
    ];
    let want = explicit(&r, &sigma, t, 2000);
    let got = r.evolve(&sigma, &[Event::Delay(t)]).unwrap();
    let err = relative(&got, &want);
    assert!(err < 1e-7, "2A ⇌ D: {err:.2e}");
}

#[test]
fn amounts_of_a_plus_b_equilibrium_follow_the_closed_form() {
    let (a0, b0, kf, kr, t) = (1.2, 0.7, 1.7, 0.6, 0.9);
    let one = || system(&[(Isotope::H1, 0.0, 0.0, 0.0)], &[]);
    let two = system(
        &[(Isotope::H1, 0.0, 0.0, 0.0), (Isotope::H1, 0.0, 0.0, 0.0)],
        &[],
    );
    let mut r = Reactions::new();
    let a = r.add_site(one(), a0).unwrap();
    let b = r.add_site(one(), b0).unwrap();
    let c = r.add_site(two, 0.0).unwrap();
    r.add_routed_reaction(
        reaction(&[S(a), S(b)], &[S(c)], kf),
        vec![route(0, 0, &[(0, 0)]), route(1, 0, &[(0, 1)])],
    )
    .unwrap();
    r.add_routed_reaction(
        reaction(&[S(c)], &[S(a), S(b)], kr),
        vec![route(0, 0, &[(0, 0)]), route(0, 1, &[(1, 0)])],
    )
    .unwrap();
    // x = [C] solves ẋ = k_f (x − x₋)(x − x₊), x(0) = 0.
    let (p, q) = (kf * (a0 + b0) + kr, kf * a0 * b0);
    let root = (p * p - 4.0 * kf * q).sqrt();
    let (lo, hi) = ((p - root) / (2.0 * kf), (p + root) / (2.0 * kf));
    let e = (-kf * (hi - lo) * t).exp();
    let x = lo * hi * (1.0 - e) / (hi - lo * e);
    let got = r.amounts(t).unwrap();
    assert!((got[2] - x).abs() < 1e-11, "{} vs {x}", got[2]);
    assert!((got[0] - (a0 - x)).abs() < 1e-11 && (got[1] - (b0 - x)).abs() < 1e-11);
}

#[test]
fn magnetisation_and_atoms_are_conserved_without_relaxation() {
    // The exchange model without relaxation: the rotating frame keeps each
    // isotope's F_z, and the reactions move it between sites whole.
    let (r0, sigma) = exchange_model();
    let mut r = r0.clone();
    for site in &mut r.sites {
        let mut sys = SpinSystem::new(site.system.field());
        for spin in site.system.spins() {
            sys.add_spin(spin.isotope, spin.shift).unwrap();
        }
        for &(i, j, hz) in site.system.couplings() {
            sys.couple(i, j, hz).unwrap();
        }
        site.system = sys;
    }
    let got = r.evolve(&sigma, &[Event::Delay(0.6)]).unwrap();
    for x in [Isotope::H1, Isotope::C13] {
        let total = |states: &[CMatrix]| -> C64 {
            states
                .iter()
                .zip(&r.sites)
                .map(|(m, s)| {
                    if s.system.isotope_list().contains(&x) {
                        (m * s.system.fz(x).unwrap()).trace()
                    } else {
                        ZERO
                    }
                })
                .sum()
        };
        let (before, after) = (total(&sigma), total(&got));
        assert!(
            (before - after).norm() < 1e-9 * before.norm(),
            "{}: {before} → {after}",
            x.name()
        );
    }
    let tr: Vec<f64> = got.iter().map(|m| m.trace().re).collect();
    assert!((tr[0] + tr[2] - 1.1).abs() < 1e-12 && (tr[1] + tr[2] - 0.9).abs() < 1e-12);
}

#[test]
fn molecules_reacting_at_random_agree_within_their_standard_error() {
    // A (¹H) + B (¹³C) ⇌ C (¹H–¹³C, J = 30 Hz): molecules with their own
    // states, reacting at random times (Gillespie), each evolved alone
    // between its reactions; a product forms in ρ_A ⊗ ρ_B of a random
    // pair and splits into the partial traces of its state.
    let a = system(&[(Isotope::H1, 0.3, 2.0, 0.8)], &[]);
    let b = system(&[(Isotope::C13, -1.0, 3.0, 1.0)], &[]);
    let c = system(
        &[(Isotope::H1, 0.5, 1.5, 0.5), (Isotope::C13, 0.6, 2.0, 0.7)],
        &[(0, 1, 30.0)],
    );
    let (kf, kr, t) = (2.0, 1.0, 0.8);
    let mut r = Reactions::new();
    let ia = r.add_site(a.clone(), 1.0).unwrap();
    let ib = r.add_site(b.clone(), 1.0).unwrap();
    let ic = r.add_site(c.clone(), 0.0).unwrap();
    r.add_routed_reaction(
        reaction(&[S(ia), S(ib)], &[S(ic)], kf),
        vec![route(0, 0, &[(0, 0)]), route(1, 0, &[(0, 1)])],
    )
    .unwrap();
    r.add_routed_reaction(
        reaction(&[S(ic)], &[S(ia), S(ib)], kr),
        vec![route(0, 0, &[(0, 0)]), route(0, 1, &[(1, 0)])],
    )
    .unwrap();
    let rho_a = state(&a, &[(a.ix(0).unwrap(), 0.45)]);
    let rho_b = state(&b, &[(b.iz(0).unwrap(), 0.3)]);
    let exact = r
        .evolve(
            &[rho_a.clone(), rho_b.clone(), CMatrix::zeros(4, 4)],
            &[Event::Delay(t)],
        )
        .unwrap();
    let zz = c.iz(0).unwrap() * c.iz(1).unwrap() * C64::new(2.0, 0.0);
    let observables: Vec<(usize, CMatrix)> = vec![
        (0, a.ix(0).unwrap()),
        (0, a.iy(0).unwrap()),
        (1, b.iz(0).unwrap()),
        (2, c.ix(0).unwrap()),
        (2, c.iy(0).unwrap()),
        (2, c.iz(1).unwrap()),
        (2, zz),
    ];
    let systems = [a, b, c];
    let volume = 200.0;
    let runs = 16;
    let mut estimates = vec![Vec::new(); observables.len()];
    for run in 0..runs {
        let mut rng = Rng::seed_from_u64(1590 + run);
        // Per site, the molecules' states and the time each was last
        // evolved to.
        let mut molecules: Vec<Vec<(CMatrix, f64)>> = vec![
            vec![(rho_a.clone(), 0.0); 200],
            vec![(rho_b.clone(), 0.0); 200],
            Vec::new(),
        ];
        let mut now = 0.0;
        let catch_up = |m: &mut (CMatrix, f64), sys: &SpinSystem, to: f64| {
            if to > m.1 {
                m.0 = sys.evolve(&m.0, &[Event::Delay(to - m.1)]).unwrap();
                m.1 = to;
            }
        };
        loop {
            let (na, nb, nc) = (
                molecules[0].len() as f64,
                molecules[1].len() as f64,
                molecules[2].len() as f64,
            );
            let forward = kf * na * nb / volume;
            let total = forward + kr * nc;
            now += -(1.0 - rng.uniform()).ln() / total;
            if now >= t {
                break;
            }
            if rng.uniform() * total < forward {
                let i = rng.below(na as u64) as usize;
                let j = rng.below(nb as u64) as usize;
                let mut x = molecules[0].swap_remove(i);
                let mut y = molecules[1].swap_remove(j);
                catch_up(&mut x, &systems[0], now);
                catch_up(&mut y, &systems[1], now);
                molecules[2].push((x.0.kronecker(&y.0), now));
            } else {
                let k = rng.below(nc as u64) as usize;
                let mut z = molecules[2].swap_remove(k);
                catch_up(&mut z, &systems[2], now);
                molecules[0].push((partial(&systems[2], &z.0, &[0]), now));
                molecules[1].push((partial(&systems[2], &z.0, &[1]), now));
            }
        }
        let mut sums: Vec<CMatrix> = systems
            .iter()
            .map(|s| CMatrix::zeros(s.dimension(), s.dimension()))
            .collect();
        for (site, list) in molecules.iter_mut().enumerate() {
            for m in list.iter_mut() {
                catch_up(m, &systems[site], t);
                sums[site] += &m.0;
            }
        }
        for (k, (site, op)) in observables.iter().enumerate() {
            estimates[k].push((&sums[*site] * op).trace().re / volume);
        }
    }
    for (k, (site, op)) in observables.iter().enumerate() {
        let want = (&exact[*site] * op).trace().re;
        let n = runs as f64;
        let mean = estimates[k].iter().sum::<f64>() / n;
        let var = estimates[k].iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let se = (var / n).sqrt();
        assert!(
            (mean - want).abs() < 4.0 * se + 1e-3,
            "observable {k}: molecules {mean:.4} ± {se:.4}, equations {want:.4}"
        );
    }
}

#[test]
fn the_steady_state_equals_newtons_solution() {
    // H₂ + S ⇌ C with the hydrogen renewed from parahydrogen, at chemical
    // equilibrium, relaxation everywhere: the state after 40 s against
    // Newton's method on the equations written out.
    let h2 = system(
        &[(Isotope::H1, 0.2, 0.8, 0.5), (Isotope::H1, 0.2, 0.8, 0.5)],
        &[(0, 1, 5.0)],
    );
    let s = system(&[(Isotope::C13, -0.5, 1.0, 0.5)], &[]);
    let c = system(
        &[
            (Isotope::H1, 0.4, 0.7, 0.4),
            (Isotope::H1, -0.3, 0.7, 0.4),
            (Isotope::C13, 0.2, 0.9, 0.5),
        ],
        &[(0, 1, -7.0), (0, 2, 40.0), (1, 2, 10.0)],
    );
    let (kf, kr) = (2.0, 4.0);
    let mut r = Reactions::new();
    let ih = r.add_site(h2, 1.0).unwrap();
    let is = r.add_site(s, 0.5).unwrap();
    let ic = r.add_site(c, kf * 0.5 / kr).unwrap();
    r.add_routed_reaction(
        reaction(&[S(ih), S(is)], &[S(ic)], kf),
        vec![route(0, 0, &[(0, 0), (1, 1)]), route(1, 0, &[(0, 2)])],
    )
    .unwrap();
    r.add_routed_reaction(
        reaction(&[S(ic)], &[S(ih), S(is)], kr),
        vec![route(0, 0, &[(0, 0), (1, 1)]), route(0, 1, &[(2, 0)])],
    )
    .unwrap();
    let mut renew = reaction(&[S(ih)], &[S(ih)], 0.7);
    renew.fresh = Fresh::Parahydrogen {
        spins: [0, 1],
        fraction: 1.0,
    };
    r.add_reaction(renew).unwrap();
    let start = r.thermal_state().unwrap();
    let got = r.evolve(&start, &[Event::Delay(40.0)]).unwrap();

    // Newton on dσ/dt = 0 (its Jacobian has no eigenvalue with a positive
    // real part: the fixed point is stable), one population of H₂ and of S replaced by their
    // traces (the atoms of H₂ + C and of S + C are conserved). The
    // equations are quadratic, so central differences give the Jacobian
    // exactly.
    let f = derivative(&r);
    let dims: Vec<usize> = start.iter().map(|m| m.nrows()).collect();
    let flatten = |sig: &[CMatrix]| -> DVector<C64> {
        DVector::from_iterator(
            dims.iter().map(|d| d * d).sum(),
            sig.iter()
                .flat_map(|m| m.transpose().iter().copied().collect::<Vec<_>>()),
        )
    };
    let unflatten = |v: &DVector<C64>| -> Vec<CMatrix> {
        let mut out = Vec::new();
        let mut at = 0;
        for &d in &dims {
            out.push(CMatrix::from_fn(d, d, |i, j| v[at + i * d + j]));
            at += d * d;
        }
        out
    };
    let amounts = [1.0, 0.5];
    let equations = |v: &DVector<C64>| -> DVector<C64> {
        let sig = unflatten(v);
        let mut out = flatten(&f(&sig));
        let mut at = 0;
        for (s, &want) in amounts.iter().enumerate() {
            out[at] = sig[s].trace() - C64::new(want, 0.0);
            at += dims[s] * dims[s];
        }
        out
    };
    let mut v = flatten(&start);
    let n = v.len();
    for _ in 0..8 {
        let residual = equations(&v);
        let mut jacobian = CMatrix::zeros(n, n);
        for k in 0..n {
            let mut up = v.clone();
            let mut down = v.clone();
            up[k] += C64::new(1e-3, 0.0);
            down[k] -= C64::new(1e-3, 0.0);
            let column = (equations(&up) - equations(&down)) / C64::new(2e-3, 0.0);
            jacobian.set_column(k, &column);
        }
        let step = jacobian.lu().solve(&residual).expect("a regular Jacobian");
        v -= step;
    }
    assert!(equations(&v).norm() < 1e-13, "{:.2e}", equations(&v).norm());
    let want = unflatten(&v);
    let err = relative(&got, &want);
    assert!(err < 1e-12, "evolved against Newton: {err:.2e}");
    // Steady chemistry over 1000 s more: the traces of the deviations,
    // which C → H₂ + S would feed twice, stay zero.
    let later = r.evolve(&got, &[Event::Delay(1000.0)]).unwrap();
    let err = relative(&later, &want);
    assert!(err < 1e-12, "after 1000 s more: {err:.2e}");
    // Not a trivial state: the complex holds order from the singlet.
    let sys = &r.sites[ic].system;
    let zz = sys.iz(0).unwrap() * sys.iz(1).unwrap() * C64::new(4.0, 0.0);
    let order = (&want[ic] * zz).trace().re / want[ic].trace().re;
    assert!(order < -0.05, "{order}");
}

#[test]
fn routes_are_checked_and_survive_json() {
    let one = system(&[(Isotope::H1, 0.0, 0.0, 0.0)], &[]);
    let carbon = system(&[(Isotope::C13, 0.0, 0.0, 0.0)], &[]);
    let two = system(
        &[(Isotope::H1, 0.0, 0.0, 0.0), (Isotope::H1, 0.0, 0.0, 0.0)],
        &[],
    );
    let mut r = Reactions::new();
    let a = r.add_site(one.clone(), 1.0).unwrap();
    let b = r.add_site(one, 1.0).unwrap();
    let c = r.add_site(two, 0.0).unwrap();
    let k = r.add_site(carbon, 1.0).unwrap();
    let g = r.add_species("G", 1.0).unwrap();
    let ab = || reaction(&[S(a), S(b)], &[S(c)], 1.0);
    for (rx, routes) in [
        // Positions out of range, species, no spins.
        (ab(), vec![route(2, 0, &[(0, 0)])]),
        (ab(), vec![route(0, 1, &[(0, 0)])]),
        (
            reaction(&[S(a), P(g)], &[S(c)], 1.0),
            vec![route(1, 0, &[(0, 0)])],
        ),
        (
            reaction(&[S(a)], &[S(c), P(g)], 1.0),
            vec![route(0, 1, &[(0, 0)])],
        ),
        (ab(), vec![route(0, 0, &[])]),
        // The same reactant and product twice, a spin twice, out of range.
        (ab(), vec![route(0, 0, &[(0, 0)]), route(0, 0, &[(0, 1)])]),
        (ab(), vec![route(0, 0, &[(0, 0)]), route(1, 0, &[(0, 0)])]),
        (ab(), vec![route(0, 0, &[(0, 0), (0, 1)])]),
        (ab(), vec![route(0, 0, &[(1, 0)])]),
        (ab(), vec![route(0, 0, &[(0, 2)])]),
        // Isotopes differ.
        (
            reaction(&[S(k), S(b)], &[S(c)], 1.0),
            vec![route(0, 0, &[(0, 0)])],
        ),
    ] {
        assert!(r.add_routed_reaction(rx, routes).is_err());
    }
    // A spin of a reactant into two products.
    let split = reaction(&[S(c)], &[S(a), S(b)], 1.0);
    assert!(
        r.add_routed_reaction(
            split.clone(),
            vec![route(0, 0, &[(0, 0)]), route(0, 1, &[(0, 0)])]
        )
        .is_err()
    );
    // Routes and `carried` together, and parahydrogen on a routed spin.
    let mut both = ab();
    both.carried = vec![(0, 0)];
    assert!(
        r.add_routed_reaction(both, vec![route(1, 0, &[(0, 1)])])
            .is_err()
    );
    let mut para = reaction(&[S(a)], &[S(c)], 1.0);
    para.fresh = Fresh::Parahydrogen {
        spins: [0, 1],
        fraction: 1.0,
    };
    assert!(
        r.add_routed_reaction(para.clone(), vec![route(0, 0, &[(0, 0)])])
            .is_err()
    );
    r.add_routed_reaction(para, Vec::new()).unwrap();
    // Plain and routed reactions keep their routes in step.
    r.add_routed_reaction(ab(), vec![route(0, 0, &[(0, 0)]), route(1, 0, &[(0, 1)])])
        .unwrap();
    r.add_reaction(reaction(&[S(c)], &[S(a), S(b)], 0.5))
        .unwrap();
    r.add_routed_reaction(split, vec![route(0, 0, &[(0, 0)]), route(0, 1, &[(1, 0)])])
        .unwrap();
    assert!(r.routes(0).is_empty() && r.routes(2).is_empty());
    assert_eq!(r.routes(1).len(), 2);
    assert_eq!(r.routes(3)[1], route(0, 1, &[(1, 0)]));
    let back = Reactions::from_json(&r.to_json()).unwrap();
    assert_eq!(back, r);
    // Without routes, no routes in the JSON; a wrong count is refused.
    let mut plain = Reactions::new();
    plain
        .add_site(system(&[(Isotope::H1, 0.0, 0.0, 0.0)], &[]), 1.0)
        .unwrap();
    assert!(!plain.to_json().contains("routes"));
    let wrong = r
        .to_json()
        .replacen("\"routes\": [\n    [],", "\"routes\": [", 1);
    assert_ne!(wrong, r.to_json());
    assert!(Reactions::from_json(&wrong).is_err());
}

/// SABRE with hydrogen and substrate pools that both carry spins, and PHIP
/// of a ¹³C-labelled substrate from a hydrogen pool whose singlet decays,
/// for the documentation (`cargo test --release -- --ignored
/// measure_two_reactants --nocapture`).
#[test]
#[ignore]
fn measure_two_reactants() {
    let polarization = |sys: &SpinSystem, sigma: &CMatrix, spin: usize| {
        sys.polarization(&(sigma * C64::new(1.0 / sigma.trace().re, 0.0)), spin)
            .unwrap()
    };
    // The singlet population of two spins ½ of `sys`.
    let singlet = |sys: &SpinSystem, sigma: &CMatrix, i: usize, j: usize| {
        let dot = sys.ix(i).unwrap() * sys.ix(j).unwrap()
            + sys.iy(i).unwrap() * sys.iy(j).unwrap()
            + sys.iz(i).unwrap() * sys.iz(j).unwrap();
        let rho = sigma * C64::new(1.0 / sigma.trace().re, 0.0);
        0.25 - (&rho * dot).trace().re
    };

    // SABRE at the anti-crossing (1.50's complex: two hydrides and the
    // bound ¹⁵N), the free ¹⁵N with T₁ = 30 s. 1.50: the substrate leaves
    // (2 s⁻¹) and binds the free catalyst, whose hydrides arrive fresh as
    // parahydrogen, and the hydrides are renewed (0.5 s⁻¹). 1.59: the
    // complex swaps its substrate with a free one (C + S → C + S, 2 s⁻¹
    // per complex) keeping its hydrides, and its hydrides with a free H₂
    // (C + H₂ → C + H₂, 0.5 s⁻¹ per complex), the released hydrides going
    // into the pool; the pool (3 per catalyst) is renewed from the gas at
    // k_g.
    let nu = |x: Isotope, shift: f64| -x.larmor(1.0) * (1.0 + shift * 1e-6);
    let lac = ((-7.0 + 0.5) / (nu(Isotope::N15, 250.0) - nu(Isotope::H1, -22.0))).abs();
    let lab = |s: SpinSystem| s.with_field(lac).unwrap().with_frame(Frame::Laboratory);
    let complex = lab(system(
        &[
            (Isotope::H1, -22.0, 0.0, 0.0),
            (Isotope::H1, -22.0, 0.0, 0.0),
            (Isotope::N15, 250.0, 0.0, 0.0),
        ],
        &[(0, 1, -7.0), (0, 2, -2.0)],
    ));
    let free = lab(system(&[(Isotope::N15, 310.0, 30.0, 30.0)], &[]));
    let hydrogen = lab(system(
        &[(Isotope::H1, 4.5, 0.0, 0.0), (Isotope::H1, 4.5, 0.0, 0.0)],
        &[],
    ));
    let para = Fresh::Parahydrogen {
        spins: [0, 1],
        fraction: 1.0,
    };
    let times = [10.0, 30.0, 60.0];
    {
        let mut r = Reactions::new();
        let c = r.add_site(complex.clone(), 1.0).unwrap();
        let f = r.add_site(free.clone(), 10.0).unwrap();
        let cat = r.add_species("catalyst", 0.0).unwrap();
        let mut off = reaction(&[S(c)], &[S(f), P(cat)], 2.0);
        off.carried = vec![(2, 0)];
        let mut on = reaction(&[S(f), P(cat)], &[S(c)], 200.0);
        on.carried = vec![(0, 2)];
        on.fresh = para;
        let mut renew = reaction(&[S(c)], &[S(c)], 0.5);
        renew.carried = vec![(2, 2)];
        renew.fresh = para;
        for rx in [off, on, renew] {
            r.add_reaction(rx).unwrap();
        }
        let clock = std::time::Instant::now();
        let mut sigma = r.thermal_state().unwrap();
        let mut line = String::new();
        let mut now = 0.0;
        for t in times {
            sigma = r
                .advanced(now)
                .unwrap()
                .evolve(&sigma, &[Event::Delay(t - now)])
                .unwrap();
            now = t;
            line += &format!(
                " {t:.0} s {:.1} %;",
                100.0 * polarization(&free, &sigma[f], 0)
            );
        }
        eprintln!(
            "SABRE 1.50 (fresh hydrides on binding):{line} in {:.1} s",
            clock.elapsed().as_secs_f64()
        );
    }
    for gas in [f64::INFINITY, 1.0, 0.1, 0.0] {
        let mut r = Reactions::new();
        let c = r.add_site(complex.clone(), 1.0).unwrap();
        let f = r.add_site(free.clone(), 10.0).unwrap();
        let h = r.add_site(hydrogen.clone(), 3.0).unwrap();
        r.add_routed_reaction(
            reaction(&[S(c), S(f)], &[S(c), S(f)], 2.0 / 10.0),
            vec![
                route(0, 0, &[(0, 0), (1, 1)]),
                route(1, 0, &[(0, 2)]),
                route(0, 1, &[(2, 0)]),
            ],
        )
        .unwrap();
        if gas.is_finite() {
            r.add_routed_reaction(
                reaction(&[S(c), S(h)], &[S(c), S(h)], 0.5 / 3.0),
                vec![
                    route(0, 0, &[(2, 2)]),
                    route(1, 0, &[(0, 0), (1, 1)]),
                    route(0, 1, &[(0, 0), (1, 1)]),
                ],
            )
            .unwrap();
            if gas > 0.0 {
                let mut renew = reaction(&[S(h)], &[S(h)], gas);
                renew.fresh = para;
                r.add_reaction(renew).unwrap();
            }
        } else {
            // A pool held in parahydrogen: the hydrides renewed fresh.
            let mut renew = reaction(&[S(c)], &[S(c)], 0.5);
            renew.carried = vec![(2, 2)];
            renew.fresh = para;
            r.add_reaction(renew).unwrap();
        }
        let clock = std::time::Instant::now();
        let mut sigma = r.thermal_state().unwrap();
        sigma[h] = hydrogen.parahydrogen_state(0, 1, 1.0).unwrap() * C64::new(3.0, 0.0);
        let mut line = String::new();
        let mut now = 0.0;
        for t in times {
            sigma = r.evolve(&sigma, &[Event::Delay(t - now)]).unwrap();
            now = t;
            line += &format!(
                " {t:.0} s {:.1} % (pool singlet {:.3});",
                100.0 * polarization(&free, &sigma[f], 0),
                singlet(&hydrogen, &sigma[h], 0, 1)
            );
        }
        eprintln!(
            "SABRE 1.59, hydrides kept, gas exchange {gas} /s:{line} in {:.1} s",
            clock.elapsed().as_secs_f64()
        );
    }

    // PHIP at 1 μT: S (a ¹³C) + H₂ → P (two hydrides and the ¹³C) at
    // k = 2 with 0.6 H₂ per substrate, the pool's protons relaxing with
    // T₁ = T₂ = T_c (the singlet decaying to normal hydrogen), against H₂ a
    // species with fresh parahydrogen (1.50).
    let low = |s: SpinSystem| s.with_field(1e-6).unwrap().with_frame(Frame::Laboratory);
    let substrate = low(system(&[(Isotope::C13, 30.0, 10.0, 10.0)], &[]));
    let product = low(system(
        &[
            (Isotope::H1, 1.2, 3.0, 3.0),
            (Isotope::H1, 1.6, 3.0, 3.0),
            (Isotope::C13, 30.0, 10.0, 10.0),
        ],
        &[(0, 1, 7.0), (0, 2, 5.0), (1, 2, 130.0)],
    ));
    let t = 2.0;
    {
        let mut r = Reactions::new();
        let s = r.add_site(substrate.clone(), 1.0).unwrap();
        let p = r.add_site(product.clone(), 0.0).unwrap();
        let g = r.add_species("H2", 0.6).unwrap();
        let mut rx = reaction(&[S(s), P(g)], &[S(p)], 2.0);
        rx.carried = vec![(0, 2)];
        rx.fresh = para;
        r.add_reaction(rx).unwrap();
        let clock = std::time::Instant::now();
        let out = r
            .evolve(&r.thermal_state().unwrap(), &[Event::Delay(t)])
            .unwrap();
        eprintln!(
            "PHIP 1.50 (fresh parahydrogen): product ¹³C {:.2} % after {t} s, formed {:.3}, in {:.1} s",
            100.0 * polarization(&product, &out[p], 2),
            out[p].trace().re,
            clock.elapsed().as_secs_f64()
        );
    }
    for t_c in [f64::INFINITY, 10.0, 2.0, 0.5] {
        let relax = if t_c.is_finite() { t_c } else { 0.0 };
        let pool = low(system(
            &[
                (Isotope::H1, 4.5, relax, relax),
                (Isotope::H1, 4.5, relax, relax),
            ],
            &[],
        ));
        let mut r = Reactions::new();
        let s = r.add_site(substrate.clone(), 1.0).unwrap();
        let p = r.add_site(product.clone(), 0.0).unwrap();
        let h = r.add_site(pool.clone(), 0.6).unwrap();
        r.add_routed_reaction(
            reaction(&[S(s), S(h)], &[S(p)], 2.0),
            vec![route(0, 0, &[(0, 2)]), route(1, 0, &[(0, 0), (1, 1)])],
        )
        .unwrap();
        let mut start = r.thermal_state().unwrap();
        start[h] = pool.parahydrogen_state(0, 1, 1.0).unwrap() * C64::new(0.6, 0.0);
        let clock = std::time::Instant::now();
        let out = r.evolve(&start, &[Event::Delay(t)]).unwrap();
        eprintln!(
            "PHIP 1.59, pool T_c = {t_c} s: product ¹³C {:.2} % after {t} s (pool singlet {:.3}), in {:.1} s",
            100.0 * polarization(&product, &out[p], 2),
            singlet(&pool, &out[h], 0, 1),
            clock.elapsed().as_secs_f64()
        );
    }
}
