//! Reaction kinetics with spins against closed forms.

use super::*;
use crate::spin::{BOLTZMANN, Exchange, Frame, HBAR, Isotope};

fn proton() -> SpinSystem {
    let mut sys = SpinSystem::new(14.1);
    sys.add_spin(Isotope::H1, 5.0).unwrap();
    sys
}

/// Two protons at 1 and 11 ppm, J = 7 Hz, at 14.1 T.
fn pair() -> SpinSystem {
    let mut sys = SpinSystem::new(14.1);
    let a = sys.add_spin(Isotope::H1, 1.0).unwrap();
    let b = sys.add_spin(Isotope::H1, 11.0).unwrap();
    sys.couple(a, b, 7.0).unwrap();
    sys
}

fn singlet() -> Fresh {
    Fresh::Parahydrogen {
        spins: [0, 1],
        fraction: 1.0,
    }
}

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

#[test]
fn amounts_follow_the_closed_forms() {
    let t = 0.7;
    // A → B, A ⇌ B, A + B → C with unequal amounts, 2A → C.
    let mut r = Reactions::new();
    r.add_site(proton(), 1.0).unwrap();
    let a = r.add_species("A", 2.0).unwrap();
    let b = r.add_species("B", 0.0).unwrap();
    r.add_reaction(reaction(&[P(a)], &[P(b)], 1.3)).unwrap();
    let got = r.amounts(t).unwrap();
    let want = 2.0 * (-1.3 * t).exp();
    assert!((got[1] - want).abs() < 1e-11 && (got[2] - (2.0 - want)).abs() < 1e-11);

    let mut r = Reactions::new();
    r.add_site(proton(), 1.0).unwrap();
    let a = r.add_species("A", 2.0).unwrap();
    let b = r.add_species("B", 0.5).unwrap();
    r.add_reaction(reaction(&[P(a)], &[P(b)], 1.3)).unwrap();
    r.add_reaction(reaction(&[P(b)], &[P(a)], 0.4)).unwrap();
    let got = r.amounts(t).unwrap();
    let eq = 2.5 * 0.4 / 1.7;
    let want = eq + (2.0 - eq) * (-1.7 * t).exp();
    assert!((got[1] - want).abs() < 1e-11, "{} vs {want}", got[1]);

    let mut r = Reactions::new();
    r.add_site(proton(), 1.0).unwrap();
    let (a0, b0, k) = (2.0, 0.8, 1.9);
    let a = r.add_species("A", a0).unwrap();
    let b = r.add_species("B", b0).unwrap();
    let c = r.add_species("C", 0.0).unwrap();
    r.add_reaction(reaction(&[P(a), P(b)], &[P(c)], k)).unwrap();
    let got = r.amounts(t).unwrap();
    let e = ((a0 - b0) * k * t).exp();
    let x = a0 * b0 * (e - 1.0) / (a0 * e - b0);
    assert!((got[3] - x).abs() < 1e-11, "{} vs {x}", got[3]);
    assert!((got[1] - (a0 - x)).abs() < 1e-11 && (got[2] - (b0 - x)).abs() < 1e-11);

    let mut r = Reactions::new();
    r.add_site(proton(), 1.0).unwrap();
    let a = r.add_species("A", 1.5).unwrap();
    let c = r.add_species("C", 0.0).unwrap();
    r.add_reaction(reaction(&[P(a), P(a)], &[P(c)], 0.9))
        .unwrap();
    let got = r.amounts(t).unwrap();
    let want = 1.5 / (1.0 + 2.0 * 0.9 * 1.5 * t);
    assert!((got[1] - want).abs() < 1e-11);
    assert!((got[2] - (1.5 - want) / 2.0).abs() < 1e-11);
}

#[test]
fn coherent_formation_follows_its_closed_form() {
    // Parahydrogen adds to A (no spins) at rate k into a pair of protons
    // in the singlet. In the eigenbasis of the product's H the element
    // mn of σ at T is c₀ ρ_mn k e^{−iω T}(1 − e^{−(k − iω)T})/(k − iω),
    // ω = ω_m − ω_n: averaged away for k ≪ |ω| (PASADENA), the coherent
    // singlet of 1.33 for k ≫ |ω|.
    let product = pair();
    let h = product.hamiltonian().unwrap();
    let eigen = h.clone().symmetric_eigen();
    let v = &eigen.eigenvectors;
    let rho0 = product.parahydrogen_state(0, 1, 1.0).unwrap();
    let in_eigen = v.adjoint() * &rho0 * v;
    let total_t = 2e-3;
    for k in [1e2, 1e4, 1e6, 1e8] {
        let mut r = Reactions::new();
        let p = r.add_site(product.clone(), 0.0).unwrap();
        let a = r.add_species("A", 1.0).unwrap();
        let mut rx = reaction(&[P(a)], &[S(p)], k);
        rx.fresh = singlet();
        r.add_reaction(rx).unwrap();
        let start = vec![CMatrix::zeros(4, 4)];
        let got = r.evolve(&start, &[Event::Delay(total_t)]).unwrap();
        let mut want = CMatrix::zeros(4, 4);
        for m in 0..4 {
            for n in 0..4 {
                let w = eigen.eigenvalues[m] - eigen.eigenvalues[n];
                let z = C64::new(k, -w);
                let f = C64::new(k, 0.0)
                    * (C64::new(0.0, -w * total_t)).exp()
                    * (C64::new(1.0, 0.0) - (-z * total_t).exp())
                    / z;
                want[(m, n)] = in_eigen[(m, n)] * f;
            }
        }
        let want = v * want * v.adjoint();
        let worst = (&got[0] - &want)
            .iter()
            .fold(0.0f64, |m, z| m.max(z.norm()));
        assert!(worst < 1e-10, "k = {k}: {worst:e}");
        assert!((got[0].trace().re - (1.0 - (-k * total_t).exp())).abs() < 1e-10);
    }
    // k = 10⁸ ≫ |ω| = 3.8·10⁴ rad/s: the singlet of 1.33 evolved for T, to
    // |ω|/k = 3.8·10⁻⁴.
    let mut r = Reactions::new();
    let p = r.add_site(product.clone(), 0.0).unwrap();
    let a = r.add_species("A", 1.0).unwrap();
    let mut rx = reaction(&[P(a)], &[S(p)], 1e8);
    rx.fresh = singlet();
    r.add_reaction(rx).unwrap();
    let got = r
        .evolve(&[CMatrix::zeros(4, 4)], &[Event::Delay(total_t)])
        .unwrap();
    let coherent = product.evolve(&rho0, &[Event::Delay(total_t)]).unwrap();
    let worst = (&got[0] - &coherent)
        .iter()
        .fold(0.0f64, |m, z| m.max(z.norm()));
    assert!(worst < 3.8e4 / 1e8, "{worst:e}");
}

#[test]
fn accumulated_order_with_relaxation_is_a_convolution() {
    // A → P at rate k into the singlet; T₁ of both protons at a
    // temperature where the thermal state is 1/4 (10⁻¹⁰ polarisation), so
    // ⟨4I_zS_z⟩ decays at R = 2/T₁: c₀ k (e^{−Rt} − e^{−kt})/(k − R), and
    // −1 per molecule formed.
    let mut product = pair().with_temperature(1e7).unwrap();
    for i in 0..2 {
        product.set_relaxation(i, Some(0.8), Some(0.05)).unwrap();
    }
    let zz = &product.iz(0).unwrap() * &product.iz(1).unwrap() * C64::new(4.0, 0.0);
    let (k, c0) = (3.0, 2.0);
    let rr = 2.0 / 0.8;
    let mut r = Reactions::new();
    let p = r.add_site(product, 0.0).unwrap();
    let a = r.add_species("A", c0).unwrap();
    let mut rx = reaction(&[P(a)], &[S(p)], k);
    rx.fresh = singlet();
    r.add_reaction(rx).unwrap();
    let mut t = 0.0;
    for dt in [0.1, 0.4, 1.0] {
        let next = r
            .evolve(&[CMatrix::zeros(4, 4)], &[Event::Delay(t + dt)])
            .unwrap();
        t += dt;
        let got = (&next[0] * &zz).trace().re;
        let want = -c0 * k * ((-rr * t).exp() - (-k * t).exp()) / (k - rr);
        assert!(
            (got - want).abs() < 1e-10 * want.abs(),
            "t = {t}: {got} vs {want}"
        );
    }
}

#[test]
fn steady_reactions_are_the_exchange_of_1_35() {
    // A ⇌ B carrying a proton between two sites at steady amounts, against
    // the Bloch–McConnell exchange.
    let mut site_a = proton();
    site_a.set_relaxation(0, Some(1.5), Some(0.3)).unwrap();
    let mut site_b = SpinSystem::new(14.1);
    site_b.add_spin(Isotope::H1, 6.0).unwrap();
    site_b.set_relaxation(0, Some(2.0), Some(0.5)).unwrap();
    let (ca, cb, kf) = (1.0, 2.5, 4.0);
    let kb = kf * ca / cb;
    let mut ex = Exchange::new();
    ex.add_site(site_a.clone(), ca).unwrap();
    ex.add_site(site_b.clone(), cb).unwrap();
    for (from, to, rate) in [(0, 1, kf), (1, 0, kb)] {
        ex.add_channel(Channel {
            from,
            to,
            rate,
            carried: vec![(0, 0)],
            fresh: Fresh::Thermal,
        })
        .unwrap();
    }
    let mut r = Reactions::new();
    r.add_site(site_a.clone(), ca).unwrap();
    r.add_site(site_b.clone(), cb).unwrap();
    for (from, to, rate) in [(0, 1, kf), (1, 0, kb)] {
        let mut rx = reaction(&[S(from)], &[S(to)], rate);
        rx.carried = vec![(0, 0)];
        r.add_reaction(rx).unwrap();
    }
    let pulse = Event::pulse(Target::Isotope(Isotope::H1), 90.0, 0.0);
    let events = [pulse, Event::Delay(0.37)];
    let start = ex.thermal_state().unwrap();
    let want = ex.evolve(&start, &events).unwrap();
    let begin = r.thermal_state().unwrap();
    let got = r.evolve(&begin, &events).unwrap();
    for (s, c) in [(0, ca), (1, cb)] {
        let worst = (&got[s] - &want[s] * C64::new(c, 0.0))
            .iter()
            .fold(0.0f64, |m, z| m.max(z.norm()));
        assert!(worst < 1e-12, "site {s}: {worst:e}");
    }
}

#[test]
fn carried_spins_keep_their_state_through_a_second_order_reaction() {
    // A substrate with a polarised ¹³C (p = 0.3) takes up parahydrogen
    // (depleted: less H₂ than substrate) by A + H₂ → P. ⟨C_z⟩ of the
    // product is p/2 times its amount, of the substrate p/2 times its own,
    // as C_z commutes with the weakly coupled Hamiltonian.
    let mut substrate = SpinSystem::new(9.4);
    substrate.add_spin(Isotope::C13, 170.0).unwrap();
    let mut product = SpinSystem::new(9.4);
    let c = product.add_spin(Isotope::C13, 172.0).unwrap();
    let h1 = product.add_spin(Isotope::H1, 2.0).unwrap();
    let h2 = product.add_spin(Isotope::H1, 4.5).unwrap();
    product.couple(c, h1, 7.0).unwrap();
    product.couple(h1, h2, 7.5).unwrap();
    let mut r = Reactions::new();
    let a = r.add_site(substrate.clone(), 1.0).unwrap();
    let p = r.add_site(product.clone(), 0.0).unwrap();
    let h2_gas = r.add_species("H2", 0.6).unwrap();
    r.add_reaction(Reaction {
        reactants: vec![S(a), P(h2_gas)],
        products: vec![S(p)],
        rate: 5.0,
        carried: vec![(0, 0)],
        fresh: Fresh::Parahydrogen {
            spins: [1, 2],
            fraction: 1.0,
        },
    })
    .unwrap();
    let polarised = {
        let mut rho = CMatrix::identity(2, 2) * C64::new(0.5, 0.0);
        rho += substrate.iz(0).unwrap() * C64::new(0.3, 0.0);
        rho
    };
    let start = vec![polarised, CMatrix::zeros(8, 8)];
    let events = [Event::Delay(0.5)];
    let got = r.evolve(&start, &events).unwrap();
    let amounts = r.amounts(0.5).unwrap();
    let cz_p = (&got[p] * &product.iz(c).unwrap()).trace().re;
    let cz_a = (&got[a] * &substrate.iz(0).unwrap()).trace().re;
    assert!(
        (cz_p - 0.15 * amounts[p]).abs() < 1e-9,
        "{cz_p} vs {}",
        0.15 * amounts[p]
    );
    assert!((cz_a - 0.15 * amounts[a]).abs() < 1e-9);
    for (s, sigma) in got.iter().enumerate() {
        assert!((sigma.trace().re - amounts[s]).abs() < 1e-12);
    }
    // The second-order closed form, a₀ = 1, b₀ = 0.6, k = 5, t = 0.5.
    let e = ((1.0f64 - 0.6) * 5.0 * 0.5).exp();
    let x = 0.6 * (e - 1.0) / (e - 0.6);
    assert!((amounts[p] - x).abs() < 1e-11, "{} vs {x}", amounts[p]);
}

#[test]
fn thermal_states_stay_thermal_and_traces_follow_through_events() {
    // Thermal fresh spins and amounts that change: every state stays its
    // amount times the thermal state, through a sweep.
    let mut a_sys = proton();
    a_sys.set_relaxation(0, Some(1.0), Some(0.5)).unwrap();
    let mut b_sys = SpinSystem::new(14.1);
    b_sys.add_spin(Isotope::H1, 3.0).unwrap();
    b_sys.set_relaxation(0, Some(1.0), Some(0.5)).unwrap();
    let mut r = Reactions::new();
    let a = r.add_site(a_sys.clone(), 1.0).unwrap();
    let b = r.add_site(b_sys.clone(), 0.2).unwrap();
    r.add_reaction(reaction(&[S(a)], &[S(b)], 2.0)).unwrap();
    let start = r.thermal_state().unwrap();
    let got = r.evolve(&start, &[Event::Delay(0.6)]).unwrap();
    let amounts = r.amounts(0.6).unwrap();
    for (s, sys) in [(a, &a_sys), (b, &b_sys)] {
        let want = sys.thermal_state().unwrap() * C64::new(amounts[s], 0.0);
        let worst = (&got[s] - &want)
            .iter()
            .fold(0.0f64, |m, z| m.max(z.norm()));
        assert!(worst < 1e-13, "{s}: {worst:e}");
    }
    let swept = r
        .evolve(
            &start,
            &[
                Event::pulse(Target::Isotope(Isotope::H1), 30.0, 0.0),
                Event::Sweep {
                    to: 1.0,
                    duration: 0.3,
                    steps: 30,
                },
            ],
        )
        .unwrap();
    let amounts = r.amounts(0.3).unwrap();
    for (s, sigma) in swept.iter().enumerate() {
        assert!((sigma.trace().re - amounts[s]).abs() < 1e-12);
    }
}

#[test]
fn reactions_input_is_checked_and_survives_json() {
    let mut r = Reactions::new();
    assert!(r.add_site(proton(), -1.0).is_err());
    let p = r.add_site(pair(), 0.0).unwrap();
    let a = r.add_species("A", 1.0).unwrap();
    assert!(r.add_species("B", f64::NAN).is_err());
    for bad in [
        reaction(&[], &[S(p)], 1.0),
        reaction(&[P(a), P(a), P(a)], &[S(p)], 1.0),
        reaction(&[P(a)], &[S(p), S(p), S(p)], 1.0),
        reaction(&[P(5)], &[S(p)], 1.0),
        reaction(&[P(a)], &[S(p)], -1.0),
    ] {
        assert!(r.add_reaction(bad).is_err());
    }
    let mut carried = reaction(&[P(a)], &[S(p)], 1.0);
    carried.carried = vec![(0, 0)];
    assert!(r.add_reaction(carried).is_err());
    let mut fresh = reaction(&[P(a)], &[], 1.0);
    fresh.fresh = singlet();
    assert!(r.add_reaction(fresh).is_err());
    let mut ok = reaction(&[P(a)], &[S(p)], 2.0);
    ok.fresh = singlet();
    r.add_reaction(ok).unwrap();
    // Evolution in two pieces equals one.
    let start = vec![CMatrix::zeros(4, 4)];
    let once = r.evolve(&start, &[Event::Delay(0.6)]).unwrap();
    let half = r.evolve(&start, &[Event::Delay(0.25)]).unwrap();
    let twice = r
        .advanced(0.25)
        .unwrap()
        .evolve(&half, &[Event::Delay(0.35)])
        .unwrap();
    assert!((&once[0] - &twice[0]).norm() < 1e-11);
    // States must carry the initial amounts.
    assert!(r.evolve(&[CMatrix::identity(4, 4)], &[]).is_err());
    let back = Reactions::from_json(&r.to_json()).unwrap();
    assert_eq!(back, r);
    assert!(Reactions::new().amounts(1.0).is_err());
    // Species need names of their own, also in JSON.
    assert!(r.add_species("A", 1.0).is_err());
    let twice = r.to_json().replace(
        "\"species\": [\n    {\n      \"name\": \"A\",\n      \"amount\": 1.0\n    }\n  ]",
        "\"species\": [{\"name\": \"A\", \"amount\": 1.0}, {\"name\": \"A\", \"amount\": 2.0}]",
    );
    assert_ne!(twice, r.to_json());
    assert!(Reactions::from_json(&twice).is_err());
    // A site's trace is checked against its own amount, not all amounts.
    let mut big = Reactions::new();
    big.add_site(proton(), 1.0).unwrap();
    big.add_species("H2", 1e4).unwrap();
    let mut off = proton().thermal_state().unwrap();
    off[(0, 0)] += C64::new(5e-6, 0.0);
    assert!(big.evolve(&[off], &[]).is_err());
    // Amounts that overflow are refused, not returned as infinities.
    let mut hot = Reactions::new();
    hot.add_site(proton(), 1.0).unwrap();
    let a = hot.add_species("A", 1e160).unwrap();
    let b = hot.add_species("B", 0.0).unwrap();
    hot.add_reaction(reaction(&[P(a), P(a)], &[P(b)], 1.0))
        .unwrap();
    let err = hot.amounts(1.0).unwrap_err();
    assert!(err.to_string().contains("overflow"), "{err}");
}

#[test]
fn carried_coherence_follows_a_changing_rate() {
    // S (a proton at 1 ppm, T₂ = 1 s) meets H₂ (0.7 per S) at k = 4 and
    // becomes P (a proton at 3 ppm, T₂ = 0.5 s), the spin carried: P's
    // coherence is σ₀ ∫ e^{λ_P(T−s)} e^{λ_S s} (−ċ_S(s)/c_S(0)) ds with λ
    // each site's own coherence rate (the S coherence decays as c_S(s)
    // besides). The pseudo-rate k c_H₂ falls as H₂ is used, while the
    // coherences of S and P differ by 2 ppm (5·10³ rad/s at 9.4 T):
    // frozen at each step's midpoint it erred by 1.5 % with steps that
    // followed the kinetics alone; within h‖M‖ ≤ 1/4 by 10⁻¹² of the
    // coherence carried (0.25).
    let site = |shift: f64, t2: f64| {
        let mut sys = SpinSystem::new(9.4);
        sys.add_spin(Isotope::H1, shift).unwrap();
        sys.set_relaxation(0, None, Some(t2)).unwrap();
        sys
    };
    let (s_sys, p_sys) = (site(1.0, 1.0), site(3.0, 0.5));
    // Each site's own rate of the coherence ρ₀₁, from a short delay.
    let rate = |sys: &SpinSystem| {
        let mut rho = CMatrix::identity(2, 2) * C64::new(0.5, 0.0);
        rho[(0, 1)] = C64::new(0.25, 0.0);
        rho[(1, 0)] = C64::new(0.25, 0.0);
        let tau = 1e-5;
        let after = sys.evolve(&rho, &[Event::Delay(tau)]).unwrap();
        (after[(0, 1)] / rho[(0, 1)]).ln() / tau
    };
    let (ls, lp) = (rate(&s_sys), rate(&p_sys));
    let (a0, b0, k, total) = (1.0, 0.7, 4.0, 0.5);
    let mut r = Reactions::new();
    let s = r.add_site(s_sys, a0).unwrap();
    let p = r.add_site(p_sys, 0.0).unwrap();
    let h2 = r.add_species("H2", b0).unwrap();
    let mut rx = reaction(&[S(s), P(h2)], &[S(p)], k);
    rx.carried = vec![(0, 0)];
    r.add_reaction(rx).unwrap();
    let mut start = CMatrix::identity(2, 2) * C64::new(0.5, 0.0);
    start[(0, 1)] = C64::new(0.25, 0.0);
    start[(1, 0)] = C64::new(0.25, 0.0);
    let got = r
        .evolve(&[start, CMatrix::zeros(2, 2)], &[Event::Delay(total)])
        .unwrap();
    // c_S for A + B → P with unequal amounts, and −ċ_S = k c_S c_B.
    let c_s = |t: f64| {
        let e = ((a0 - b0) * k * t).exp();
        (a0 - b0) * a0 * e / (a0 * e - b0)
    };
    let loss = |t: f64| {
        let c = c_s(t);
        k * c * (c - (a0 - b0))
    };
    // Composite five-point Gauss–Legendre over 20 000 panels.
    let (u, v) = (0.538_469_310_105_683_1, 0.906_179_845_938_664);
    let nodes = [-v, -u, 0.0, u, v];
    let (wu, wv) = (0.478_628_670_499_366_5, 0.236_926_885_056_189_1);
    let weights = [wv, wu, 0.568_888_888_888_888_9, wu, wv];
    let panels = 20_000;
    let width = total / panels as f64;
    let mut want = C64::new(0.0, 0.0);
    for q in 0..panels {
        for (x, w) in nodes.iter().zip(&weights) {
            let t = width * (q as f64 + 0.5 * (x + 1.0));
            want += (lp * (total - t) + ls * t).exp() * (loss(t) / a0) * (0.5 * w * width);
        }
    }
    want *= C64::new(0.25, 0.0);
    let error = (got[p][(0, 1)] - want).norm();
    assert!(error < 4e-12, "{} vs {want}: {error:e}", got[p][(0, 1)]);
    // S's own coherence decays with c_S, its trace is c_S.
    let own = C64::new(0.25, 0.0) * (ls * total).exp() * (c_s(total) / a0);
    assert!((got[s][(0, 1)] - own).norm() < 1e-10 * own.norm());
    assert!((got[p].trace().re - (a0 - c_s(total))).abs() < 1e-12);
}

#[test]
fn steady_chemistry_takes_few_steps_and_stiff_kinetics_are_refused() {
    // A ⇌ B at 10⁵ s⁻¹ each way, the spin carried, amounts at
    // equilibrium: the coefficients are constant, so one step, equal to
    // the exchange of 1.35.
    let (a_sys, b_sys) = {
        let mut a = proton();
        a.set_relaxation(0, Some(1.0), Some(0.3)).unwrap();
        let mut b = SpinSystem::new(14.1);
        b.add_spin(Isotope::H1, 2.0).unwrap();
        b.set_relaxation(0, Some(2.0), Some(0.5)).unwrap();
        (a, b)
    };
    let mut r = Reactions::new();
    let a = r.add_site(a_sys.clone(), 1.0).unwrap();
    let b = r.add_site(b_sys.clone(), 1.0).unwrap();
    for (from, to) in [(a, b), (b, a)] {
        let mut rx = reaction(&[S(from)], &[S(to)], 1e5);
        rx.carried = vec![(0, 0)];
        r.add_reaction(rx).unwrap();
    }
    let mut x = Exchange::new();
    x.add_site(a_sys.clone(), 1.0).unwrap();
    x.add_site(b_sys, 1.0).unwrap();
    x.add_channel(Channel {
        from: 0,
        to: 1,
        rate: 1e5,
        carried: vec![(0, 0)],
        fresh: Fresh::Thermal,
    })
    .unwrap();
    x.add_channel(Channel {
        from: 1,
        to: 0,
        rate: 1e5,
        carried: vec![(0, 0)],
        fresh: Fresh::Thermal,
    })
    .unwrap();
    let mut start = a_sys.thermal_state().unwrap();
    start[(0, 1)] = C64::new(0.2, 0.1);
    start[(1, 0)] = C64::new(0.2, -0.1);
    let states = [start.clone(), b_sys_thermal()];
    let clock = std::time::Instant::now();
    let got = r.evolve(&states, &[Event::Delay(1.0)]).unwrap();
    assert!(clock.elapsed().as_secs_f64() < 5.0);
    let want = x.evolve(&states, &[Event::Delay(1.0)]).unwrap();
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).norm() < 1e-12, "{:e}", (g - w).norm());
    }
    // Stiff kinetics (10⁹ s⁻¹ over 10 s) are refused, quickly.
    let mut stiff = Reactions::new();
    stiff.add_site(proton(), 1.0).unwrap();
    let a = stiff.add_species("A", 1.0).unwrap();
    let b = stiff.add_species("B", 0.0).unwrap();
    stiff.add_reaction(reaction(&[P(a)], &[P(b)], 1e9)).unwrap();
    stiff.add_reaction(reaction(&[P(b)], &[P(a)], 1e9)).unwrap();
    let clock = std::time::Instant::now();
    let err = stiff.amounts(10.0).unwrap_err();
    assert!(err.to_string().contains("stiff"), "{err}");
    assert!(clock.elapsed().as_secs_f64() < 20.0);
}

fn b_sys_thermal() -> CMatrix {
    let mut b = SpinSystem::new(14.1);
    b.add_spin(Isotope::H1, 2.0).unwrap();
    b.thermal_state().unwrap()
}

#[test]
fn long_sweeps_are_refused_before_they_start() {
    // Four coupled protons formed from A, the field swept in 200 000
    // steps: refused before the first step, as Exchange refuses it.
    let mut product = SpinSystem::new(1.0);
    for k in 0..4 {
        product.add_spin(Isotope::H1, 1.0 + k as f64).unwrap();
        product.set_relaxation(k, Some(2.0), Some(1.0)).unwrap();
    }
    for k in 0..3 {
        product.couple(k, k + 1, 7.0).unwrap();
    }
    let mut r = Reactions::new();
    let p = r.add_site(product, 0.0).unwrap();
    let a = r.add_species("A", 1.0).unwrap();
    r.add_reaction(reaction(&[P(a)], &[S(p)], 3.0)).unwrap();
    let clock = std::time::Instant::now();
    let err = r
        .evolve(
            &[CMatrix::zeros(16, 16)],
            &[Event::Sweep {
                to: 1.0,
                duration: 1.0,
                steps: 200_000,
            }],
        )
        .unwrap_err();
    assert!(err.to_string().contains("operations"), "{err}");
    assert!(clock.elapsed().as_secs_f64() < 30.0);
}

#[test]
fn pulses_rotate_sites_without_amount() {
    // A traceless deviation on a site of amount zero turns like any other.
    let mut r = Reactions::new();
    r.add_site(proton(), 0.0).unwrap();
    let mut z = CMatrix::zeros(2, 2);
    z[(0, 0)] = C64::new(0.1, 0.0);
    z[(1, 1)] = C64::new(-0.1, 0.0);
    let got = r
        .evolve(
            &[z.clone()],
            &[Event::pulse(Target::Isotope(Isotope::H1), 180.0, 0.0)],
        )
        .unwrap();
    assert!((&got[0] + &z).norm() < 1e-14, "{}", got[0]);
}

/// PASADENA, ALTADENA and SABRE as they run, for the documentation
/// (`cargo test --release -- --ignored measure_reactions --nocapture`).
#[test]
#[ignore]
fn measure_reactions() {
    use crate::spin::Frame;
    let clock = std::time::Instant::now();
    // PASADENA at 9.4 T: A + H₂ → P, the product's protons with T₁ = 2 s,
    // T₂ = 0.5 s; −⟨4I_zS_z⟩ per molecule of A, parahydrogen in excess
    // (10 per A) or depleted (0.8 per A).
    let mut product = SpinSystem::new(9.4);
    product.add_spin(Isotope::H1, 1.2).unwrap();
    product.add_spin(Isotope::H1, 2.9).unwrap();
    product.couple(0, 1, 7.2).unwrap();
    for i in 0..2 {
        product.set_relaxation(i, Some(2.0), Some(0.5)).unwrap();
    }
    let zz = &product.iz(0).unwrap() * &product.iz(1).unwrap() * C64::new(4.0, 0.0);
    for (label, h2, k) in [("excess", 10.0, 0.2), ("depleted", 0.8, 2.5)] {
        let mut r = Reactions::new();
        let p = r.add_site(product.clone(), 0.0).unwrap();
        let a = r.add_species("A", 1.0).unwrap();
        let g = r.add_species("H2", h2).unwrap();
        let mut rx = reaction(&[P(a), P(g)], &[S(p)], k);
        rx.fresh = singlet();
        r.add_reaction(rx).unwrap();
        let mut sigma = vec![CMatrix::zeros(4, 4)];
        let mut best = (0.0, 0.0);
        let mut line = String::new();
        for step in 1..=40 {
            sigma = r_step(&r, &sigma, step as f64 * 0.25, 0.25);
            let s = -(&sigma[0] * &zz).trace().re;
            if s > best.1 {
                best = (step as f64 * 0.25, s);
            }
            if step % 8 == 0 {
                line += &format!(" t={:.0} s {s:.3};", step as f64 * 0.25);
            }
        }
        let formed = r.amounts(10.0).unwrap()[0];
        eprintln!(
            "PASADENA, H₂ {label}: signal peaks at {:.2} s at {:.3} (formed then {:.3}, \
             at 10 s {formed:.3});{line}",
            best.0,
            best.1,
            r.amounts(best.0).unwrap()[0]
        );
    }
    // ALTADENA: the reaction runs while the field rises from 10 mT to 1 T
    // in 2 s; then 9.4 T. Population of the lower of |αβ⟩, |βα⟩ minus the
    // other, per molecule formed.
    let mut low = SpinSystem::new(0.01);
    low.add_spin(Isotope::H1, 1.2).unwrap();
    low.add_spin(Isotope::H1, 2.9).unwrap();
    low.couple(0, 1, 7.2).unwrap();
    for k in [0.3, 1.0, 3.0, 30.0] {
        let mut r = Reactions::new();
        let p = r.add_site(low.clone(), 0.0).unwrap();
        let a = r.add_species("A", 1.0).unwrap();
        let mut rx = reaction(&[P(a)], &[S(p)], k);
        rx.fresh = singlet();
        r.add_reaction(rx).unwrap();
        let out = r
            .evolve(
                &[CMatrix::zeros(4, 4)],
                &[
                    Event::Sweep {
                        to: 1.0,
                        duration: 2.0,
                        steps: 400,
                    },
                    Event::Field(9.4),
                ],
            )
            .unwrap();
        let formed = out[0].trace().re;
        let diff = (out[0][(2, 2)].re - out[0][(1, 1)].re) / formed;
        eprintln!(
            "ALTADENA, k = {k} /s: formed {formed:.3}, population difference |βα⟩ − |αβ⟩ {diff:.3}"
        );
    }
    // SABRE (the 1.35 model at its anti-crossing) with a free catalyst that
    // binds the substrate (k_a) and is deactivated (k_x): the free ¹⁵N's
    // polarisation over two minutes.
    let nu = |x: Isotope, shift: f64| -x.larmor(1.0) * (1.0 + shift * 1e-6);
    let lac = ((-7.0 + 0.5) / (nu(Isotope::N15, 250.0) - nu(Isotope::H1, -22.0))).abs();
    let mut complex = SpinSystem::new(lac).with_frame(Frame::Laboratory);
    complex.add_spin(Isotope::H1, -22.0).unwrap();
    complex.add_spin(Isotope::H1, -22.0).unwrap();
    complex.add_spin(Isotope::N15, 250.0).unwrap();
    complex.couple(0, 1, -7.0).unwrap();
    complex.couple(0, 2, -2.0).unwrap();
    let mut free = SpinSystem::new(lac).with_frame(Frame::Laboratory);
    free.add_spin(Isotope::N15, 310.0).unwrap();
    free.set_relaxation(0, Some(30.0), None).unwrap();
    for kx in [0.0, 5.0] {
        let mut r = Reactions::new();
        let c = r.add_site(complex.clone(), 1.0).unwrap();
        let f = r.add_site(free.clone(), 10.0).unwrap();
        let cat = r.add_species("catalyst", 0.0).unwrap();
        let dead = r.add_species("dead", 0.0).unwrap();
        let mut off = reaction(&[S(c)], &[S(f), P(cat)], 2.0);
        off.carried = vec![(2, 0)];
        let mut on = reaction(&[S(f), P(cat)], &[S(c)], 200.0);
        on.carried = vec![(0, 2)];
        on.fresh = singlet();
        let mut renew = reaction(&[S(c)], &[S(c)], 0.5);
        renew.carried = vec![(2, 2)];
        renew.fresh = singlet();
        for rx in [off, on, renew, reaction(&[P(cat)], &[P(dead)], kx)] {
            r.add_reaction(rx).unwrap();
        }
        let mut sigma = r.thermal_state().unwrap();
        let iz = free.iz(0).unwrap();
        let mut line = String::new();
        let mut now = 0.0;
        for t in [10.0, 30.0, 60.0, 120.0] {
            sigma = r_step(&r, &sigma, t, t - now);
            now = t;
            let amounts = r.amounts(t).unwrap();
            let pol = 2.0 * (&sigma[f] * &iz).trace().re / amounts[f];
            line += &format!(
                " {t:.0} s: {:.1} % (complex {:.3});",
                100.0 * pol,
                amounts[c]
            );
        }
        eprintln!("SABRE, k_x = {kx} /s:{line}");
    }
    eprintln!("all in {:.1} s", clock.elapsed().as_secs_f64());
}

/// The states at `t` from those at `t − dt`, by the reactions advanced to
/// `t − dt`.
fn r_step(r: &Reactions, sigma: &[CMatrix], t: f64, dt: f64) -> Vec<CMatrix> {
    r.advanced(t - dt)
        .unwrap()
        .evolve(sigma, &[Event::Delay(dt)])
        .unwrap()
}

/// `Tr(σ I_z)` of spin `i`.
fn iz_of(sys: &SpinSystem, sigma: &CMatrix, i: usize) -> f64 {
    (sigma * sys.iz(i).unwrap()).trace().re
}

fn worst(a: &CMatrix, b: &CMatrix) -> f64 {
    (a - b).iter().fold(0.0f64, |m, z| m.max(z.norm()))
}

#[test]
fn a_source_is_a_reaction_of_a_catalyst() {
    // A → P + A at k forms P at k·c_A with fresh spins: a source of that
    // rate in the fresh state, thermal or parahydrogen. (P starts with an
    // amount: from none the reaction's steps would not follow P's
    // consumption, while a source's do.)
    let (k, ca, kd, c0) = (0.8, 1.5, 0.3, 0.4);
    let mut product = pair();
    for i in 0..2 {
        product.set_relaxation(i, Some(2.0), Some(1.0)).unwrap();
    }
    for (fresh, state) in [
        (Fresh::Thermal, product.thermal_state().unwrap()),
        (singlet(), product.parahydrogen_state(0, 1, 1.0).unwrap()),
    ] {
        let mut by_reaction = Reactions::new();
        let p = by_reaction.add_site(product.clone(), c0).unwrap();
        let a = by_reaction.add_species("A", ca).unwrap();
        let d = by_reaction.add_species("D", 0.0).unwrap();
        let mut rx = reaction(&[P(a)], &[S(p), P(a)], k);
        rx.fresh = fresh;
        by_reaction.add_reaction(rx).unwrap();
        by_reaction
            .add_reaction(reaction(&[S(p)], &[P(d)], kd))
            .unwrap();
        let mut by_source = Reactions::new();
        let p2 = by_source.add_site(product.clone(), c0).unwrap();
        by_source.add_species("A", ca).unwrap();
        let d2 = by_source.add_species("D", 0.0).unwrap();
        assert_eq!(by_source.add_source(p2, k * ca, &state).unwrap(), 0);
        by_source
            .add_reaction(reaction(&[S(p2)], &[P(d2)], kd))
            .unwrap();
        let start = vec![product.thermal_state().unwrap() * C64::new(c0, 0.0)];
        for t in [0.2, 1.5] {
            let x = by_reaction.evolve(&start, &[Event::Delay(t)]).unwrap();
            let y = by_source.evolve(&start, &[Event::Delay(t)]).unwrap();
            assert!(
                worst(&x[0], &y[0]) <= 1e-12,
                "{fresh:?} {t}: {:e}",
                worst(&x[0], &y[0])
            );
            let (ax, ay) = (
                by_reaction.amounts(t).unwrap(),
                by_source.amounts(t).unwrap(),
            );
            for (u, v) in ax.iter().zip(&ay) {
                assert!((u - v).abs() <= 1e-12, "{fresh:?} {t}: {u} vs {v}");
            }
        }
    }
}

#[test]
fn formation_decay_and_relaxation_follow_their_closed_form() {
    // P formed at F with ⟨I_z⟩ = p_s per molecule, consumed at k_d, T₁:
    // c = (F/k_d)(1 − e^{−k_d t}) and
    // m = c·m_eq + F(p_s − m_eq)/(k_d + R₁)·(1 − e^{−(k_d + R₁)t}).
    let (f, ps, kd, t1) = (2e-3, 0.15, 0.5, 2.0);
    let mut sys = SpinSystem::new(9.4).with_temperature(300.0).unwrap();
    let h = sys.add_spin(Isotope::H1, 0.0).unwrap();
    sys.set_relaxation(h, Some(t1), None).unwrap();
    let state =
        CMatrix::identity(2, 2) * C64::new(0.5, 0.0) + sys.iz(h).unwrap() * C64::new(2.0 * ps, 0.0);
    let mut r = Reactions::new();
    let p = r.add_site(sys.clone(), 0.0).unwrap();
    let d = r.add_species("D", 0.0).unwrap();
    r.add_source(p, f, &state).unwrap();
    r.add_reaction(reaction(&[S(p)], &[P(d)], kd)).unwrap();
    let x = HBAR * Isotope::H1.gamma() * 9.4 / (BOLTZMANN * 300.0);
    let meq = 0.5 * (x / 2.0).tanh();
    let r1 = 1.0 / t1;
    for t in [0.5, 2.0, 8.0] {
        let sigma = r
            .evolve(&[CMatrix::zeros(2, 2)], &[Event::Delay(t)])
            .unwrap();
        let c = f / kd * (1.0 - (-kd * t).exp());
        let m = c * meq + f * (ps - meq) / (kd + r1) * (1.0 - (-(kd + r1) * t).exp());
        let amount = sigma[0].trace().re;
        assert!((amount / c - 1.0).abs() <= 1e-12, "{t}: {amount} vs {c}");
        assert!((r.amounts(t).unwrap()[0] / c - 1.0).abs() <= 1e-12);
        let got = iz_of(&sys, &sigma[0], h);
        assert!((got / m - 1.0).abs() <= 1e-12, "{t}: {got} vs {m}");
    }
}

#[test]
fn sources_keep_or_drop_coherences_by_frame() {
    // ¹H–¹³C with a zero-quantum coherence in the source: total order 0.
    let make = |frame: Frame| {
        let mut sys = SpinSystem::new(1e-3).with_frame(frame);
        let a = sys.add_spin(Isotope::H1, 2.0).unwrap();
        let b = sys.add_spin(Isotope::C13, 30.0).unwrap();
        sys.couple(a, b, 140.0).unwrap();
        sys
    };
    let mut state = CMatrix::identity(4, 4) * C64::new(0.25, 0.0);
    // |αβ⟩ = 1, |βα⟩ = 2.
    state[(1, 2)] = C64::new(0.05, 0.02);
    state[(2, 1)] = C64::new(0.05, -0.02);
    let f = 3.0;
    // Laboratory frame: F∫₀ᵗ e^{𝓛s}(ρ − 1/4) ds + (Ft/4)·1 in the
    // eigenbasis of H.
    let lab = make(Frame::Laboratory);
    let h = lab.hamiltonian().unwrap();
    let eigen = h.clone().symmetric_eigen();
    let v = &eigen.eigenvectors;
    let dev = &state - CMatrix::identity(4, 4) * C64::new(0.25, 0.0);
    let in_eigen = v.adjoint() * &dev * v;
    let mut r = Reactions::new();
    let p = r.add_site(lab.clone(), 0.0).unwrap();
    r.add_source(p, f, &state).unwrap();
    for t in [1e-4, 1e-3] {
        let got = r
            .evolve(&[CMatrix::zeros(4, 4)], &[Event::Delay(t)])
            .unwrap();
        let mut want = CMatrix::zeros(4, 4);
        for m in 0..4 {
            for n in 0..4 {
                let w = eigen.eigenvalues[m] - eigen.eigenvalues[n];
                let integral = if w == 0.0 {
                    C64::new(t, 0.0)
                } else {
                    (C64::new(1.0, 0.0) - C64::new(0.0, -w * t).exp()) / C64::new(0.0, w)
                };
                want[(m, n)] = in_eigen[(m, n)] * integral * f;
            }
        }
        let want = v * want * v.adjoint() + CMatrix::identity(4, 4) * C64::new(f * t / 4.0, 0.0);
        assert!(
            worst(&got[0], &want) <= 1e-12,
            "{t}: {:e}",
            worst(&got[0], &want)
        );
    }
    // Rotating frame: the coherence (orders ±1 of each isotope) is dropped,
    // bit for bit.
    let rot = make(Frame::Rotating);
    let mut plain = state.clone();
    plain[(1, 2)] = ZERO;
    plain[(2, 1)] = ZERO;
    let run = |rho: &CMatrix| {
        let mut r = Reactions::new();
        let p = r.add_site(rot.clone(), 0.0).unwrap();
        r.add_source(p, f, rho).unwrap();
        r.evolve(&[CMatrix::zeros(4, 4)], &[Event::Delay(1e-3)])
            .unwrap()
    };
    assert_eq!(run(&state), run(&plain));
    // Laboratory frame: a single-quantum element is refused.
    let mut single = CMatrix::identity(4, 4) * C64::new(0.25, 0.0);
    single[(0, 1)] = C64::new(0.05, 0.0);
    single[(1, 0)] = C64::new(0.05, 0.0);
    let mut r = Reactions::new();
    let p = r.add_site(lab, 0.0).unwrap();
    assert!(r.add_source(p, f, &single).is_err());
}

#[test]
fn sources_are_checked_stored_and_survive_json() {
    let mut r = Reactions::new();
    let p = r.add_site(pair(), 1.0).unwrap();
    let a = r.add_species("A", 1.0).unwrap();
    r.add_reaction(reaction(&[S(p)], &[P(a)], 0.7)).unwrap();
    let before = r.to_json();
    assert!(!before.contains("sources"));
    let thermal = pair().thermal_state().unwrap();
    let start = vec![thermal.clone()];
    // A source of rate zero changes nothing, bit for bit.
    let plain = r.evolve(&start, &[Event::Delay(0.4)]).unwrap();
    let mut zero = r.clone();
    zero.add_source(p, 0.0, &pair().parahydrogen_state(0, 1, 1.0).unwrap())
        .unwrap();
    assert_eq!(zero.evolve(&start, &[Event::Delay(0.4)]).unwrap(), plain);
    assert_eq!(zero.amounts(0.4).unwrap(), r.amounts(0.4).unwrap());
    // Pre-1.65 JSON parses; a source survives JSON, stored normalised.
    assert_eq!(Reactions::from_json(&before).unwrap(), r);
    let unnormalised = thermal.clone() * C64::new(1.0 + 1e-10, 0.0);
    assert!(r.add_source(p, 0.5, &unnormalised).is_ok());
    let sources = r.sources();
    assert_eq!(sources.len(), 1);
    assert_eq!((sources[0].0, sources[0].1), (p, 0.5));
    assert!(worst(&sources[0].2, &thermal) <= 1e-15);
    assert!((sources[0].2.trace().re - 1.0).abs() <= 1e-15);
    let back = Reactions::from_json(&r.to_json()).unwrap();
    assert_eq!(back, r);
    assert!(r.to_json().contains("\"sources\""));
    // The checks.
    let invalid = |r: Result<usize>| match r {
        Err(QsoError::InvalidArgument { name, .. }) => assert_eq!(name, "source"),
        other => panic!("expected an invalid source, got {other:?}"),
    };
    let mut x = r.clone();
    invalid(x.add_source(5, 1.0, &thermal));
    invalid(x.add_source(p, -1.0, &thermal));
    invalid(x.add_source(p, f64::NAN, &thermal));
    invalid(x.add_source(p, 1.0, &CMatrix::identity(2, 2)));
    // A valid state flattened to 1 × 16 or 2 × 8 (the right number of
    // entries) is refused, not a panic in the trace.
    let rows: Vec<C64> = thermal.transpose().iter().copied().collect();
    invalid(x.add_source(p, 1.0, &CMatrix::from_row_slice(1, 16, &rows)));
    invalid(x.add_source(p, 1.0, &CMatrix::from_row_slice(2, 8, &rows)));
    invalid(x.add_source(5, 1.0, &CMatrix::from_row_slice(1, 16, &rows)));
    invalid(x.add_source(p, 1.0, &(thermal.clone() * C64::new(2.0, 0.0))));
    let mut nan = thermal.clone();
    nan[(1, 1)] = C64::new(f64::NAN, 0.0);
    invalid(x.add_source(p, 1.0, &nan));
    let mut skew = thermal.clone();
    skew[(0, 1)] = C64::new(0.1, 0.0);
    invalid(x.add_source(p, 1.0, &skew));
    let mut negative = CMatrix::zeros(4, 4);
    negative[(0, 0)] = C64::new(1.5, 0.0);
    negative[(1, 1)] = C64::new(-0.5, 0.0);
    invalid(x.add_source(p, 1.0, &negative));
    // A source edited into JSON is checked too.
    let json = r.to_json().replace("\"rate\": 0.5", "\"rate\": -0.5");
    assert_ne!(json, r.to_json());
    assert!(Reactions::from_json(&json).is_err());
}

#[test]
fn dense_and_krylov_propagation_agree_with_sources() {
    let mut product = pair();
    product.set_relaxation(0, Some(1.0), Some(0.3)).unwrap();
    let mut r = Reactions::new();
    let p = r.add_site(product.clone(), 0.2).unwrap();
    let d = r.add_species("D", 0.0).unwrap();
    r.add_source(p, 0.4, &product.parahydrogen_state(0, 1, 1.0).unwrap())
        .unwrap();
    r.add_reaction(reaction(&[S(p)], &[P(d)], 0.9)).unwrap();
    let start = vec![product.thermal_state().unwrap() * C64::new(0.2, 0.0)];
    let events = [
        Event::Delay(0.3),
        Event::pulse(crate::spin::Target::Isotope(Isotope::H1), 90.0, 0.0),
        Event::Delay(0.2),
    ];
    let dense = r.evolve_with(&start, &events, Propagation::Dense).unwrap();
    let krylov = r.evolve_with(&start, &events, Propagation::Krylov).unwrap();
    assert!(worst(&dense[0], &krylov[0]) <= 1e-11);
}

#[test]
fn a_radical_pair_product_forms_at_its_yield() {
    use crate::spin::{Birth, FREE_ELECTRON_G, Product, RadicalPair};
    // The singlet product of a two-proton pair (laboratory frame, 1 mT)
    // onto a two-proton site at 9.4 T, formed at G·Φ_S and consumed at k_d.
    let mut sys = SpinSystem::new(1e-3).with_frame(Frame::Laboratory);
    let e1 = sys
        .add_spin(Isotope::Electron, 1e-3 / FREE_ELECTRON_G * 1e6)
        .unwrap();
    let e2 = sys.add_electron(FREE_ELECTRON_G).unwrap();
    let h1 = sys.add_spin(Isotope::H1, 0.0).unwrap();
    let h2 = sys.add_spin(Isotope::H1, 0.0).unwrap();
    sys.hyperfine(e1, h1, 0.8).unwrap();
    sys.hyperfine(e2, h2, -0.35).unwrap();
    sys.exchange_coupling(e1, e2, -0.05).unwrap();
    let pair = RadicalPair::new(sys)
        .unwrap()
        .with_rates(3e7, 8e6, 2e6)
        .unwrap();
    let solution = pair.solve(Birth::Singlet).unwrap();
    let phi = solution.yields().singlet;
    let mut site = SpinSystem::new(9.4).with_temperature(300.0).unwrap();
    site.add_spin(Isotope::H1, 1.0).unwrap();
    site.add_spin(Isotope::H1, 3.0).unwrap();
    let t1 = 2.0;
    for i in 0..2 {
        site.set_relaxation(i, Some(t1), None).unwrap();
    }
    let state = solution
        .product_on(Product::Singlet, &site, &[(h1, 0), (h2, 1)], Fresh::Thermal)
        .unwrap();
    let (g, kd) = (1e-3, 0.5);
    let mut r = Reactions::new();
    let p = r.add_site(site.clone(), 0.0).unwrap();
    let d = r.add_species("D", 0.0).unwrap();
    r.add_source(p, g * phi, &state).unwrap();
    r.add_reaction(reaction(&[S(p)], &[P(d)], kd)).unwrap();
    let x = HBAR * Isotope::H1.gamma() * 9.4 / (BOLTZMANN * 300.0);
    let r1 = 1.0 / t1;
    for t in [0.5, 4.0] {
        let sigma = r
            .evolve(&[CMatrix::zeros(4, 4)], &[Event::Delay(t)])
            .unwrap();
        let f = g * phi;
        let c = f / kd * (1.0 - (-kd * t).exp());
        assert!((sigma[0].trace().re / c - 1.0).abs() <= 1e-12);
        for (j, shift) in [(0, 1.0), (1, 3.0)] {
            let meq = 0.5 * (x * (1.0 + shift * 1e-6) / 2.0).tanh();
            let ps = iz_of(&site, &state, j);
            let m = c * meq + f * (ps - meq) / (kd + r1) * (1.0 - (-(kd + r1) * t).exp());
            let got = iz_of(&site, &sigma[0], j);
            assert!((got / m - 1.0).abs() <= 1e-11, "{t} {j}: {got} vs {m}");
        }
    }
}
