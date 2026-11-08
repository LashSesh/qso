//! Births, closed forms, solves against Liouville space, sums, CIDNP,
//! products and time evolution of radical pairs.

use super::solve::Options;
use super::*;
use crate::spin::{Event, FREE_ELECTRON_G, Frame, Fresh, Isotope, SpinSystem, Target};

/// `|γ_e|` (rad s⁻¹ T⁻¹).
const GAMMA_E: f64 = 1.760_859_630_23e11;

const I: C64 = C64::new(0.0, 1.0);

fn system(b: f64, frame: Frame) -> SpinSystem {
    SpinSystem::new(b).with_frame(frame)
}

/// An electron of g = g_e + `dg`, its shift written as the references
/// write it, `dg/g_e·10⁶`.
fn electron(sys: &mut SpinSystem, dg: f64) -> usize {
    sys.add_spin(Isotope::Electron, dg / FREE_ELECTRON_G * 1e6)
        .unwrap()
}

fn nucleus(sys: &mut SpinSystem, isotope: Isotope) -> usize {
    sys.add_spin(isotope, 0.0).unwrap()
}

/// Two protons in the laboratory frame at 1 mT (11.3, item 1): ¹H 0.8 mT
/// on radical 1, ¹H −0.35 mT on radical 2, Δg = 10⁻³, exchange −0.05 mT,
/// `k_S = 3·10⁷`, `k_T = 8·10⁶`, `k_E = 2·10⁶`.
fn two_protons(frame: Frame, b: f64) -> RadicalPair {
    let mut sys = system(b, frame);
    let e1 = electron(&mut sys, 1e-3);
    let e2 = electron(&mut sys, 0.0);
    let h1 = nucleus(&mut sys, Isotope::H1);
    let h2 = nucleus(&mut sys, Isotope::H1);
    sys.hyperfine(e1, h1, 0.8).unwrap();
    sys.hyperfine(e2, h2, -0.35).unwrap();
    sys.exchange_coupling(e1, e2, -0.05).unwrap();
    RadicalPair::new(sys)
        .unwrap()
        .with_rates(3e7, 8e6, 2e6)
        .unwrap()
}

/// The pair of 11.2, item 4: rotating frame at 0.5 T, Δg = 1.5·10⁻³;
/// ¹H 1.2 mT and ¹H −0.4 mT on radical 1, ¹⁴N 0.6 mT on radical 2;
/// exchange 0.07 mT; `k_S = 4·10⁷`, `k_T = 1.5·10⁷`, `k_E = 3·10⁶`.
fn high_field() -> RadicalPair {
    let mut sys = system(0.5, Frame::Rotating);
    let e1 = electron(&mut sys, 1.5e-3);
    let e2 = electron(&mut sys, 0.0);
    let a = nucleus(&mut sys, Isotope::H1);
    let b = nucleus(&mut sys, Isotope::H1);
    let n = nucleus(&mut sys, Isotope::N14);
    sys.hyperfine(e1, a, 1.2).unwrap();
    sys.hyperfine(e1, b, -0.4).unwrap();
    sys.hyperfine(e2, n, 0.6).unwrap();
    sys.exchange_coupling(e1, e2, 0.07).unwrap();
    RadicalPair::new(sys)
        .unwrap()
        .with_rates(4e7, 1.5e7, 3e6)
        .unwrap()
}

/// Two electrons, ¹⁴N and two ¹H (48 states).
fn nitrogen_pair(b: f64, frame: Frame) -> SpinSystem {
    let mut sys = system(b, frame);
    let e1 = sys.add_electron(FREE_ELECTRON_G + 2e-3).unwrap();
    let e2 = sys.add_electron(FREE_ELECTRON_G).unwrap();
    let n = nucleus(&mut sys, Isotope::N14);
    let h1 = nucleus(&mut sys, Isotope::H1);
    let h2 = nucleus(&mut sys, Isotope::H1);
    sys.hyperfine(e1, n, 0.514).unwrap();
    sys.hyperfine(e2, h1, 1.6045).unwrap();
    sys.hyperfine(e2, h2, -0.5983).unwrap();
    sys
}

fn largest(m: &CMatrix) -> f64 {
    m.iter().map(|z| z.norm()).fold(0.0, f64::max)
}

/// `max|a − b| / max|b|`.
fn relative(a: &CMatrix, b: &CMatrix) -> f64 {
    largest(&(a - b)) / largest(b)
}

fn sum(y: &PairYields) -> f64 {
    y.singlet + y.triplet + y.escape
}

fn yields(pair: &RadicalPair, birth: Birth) -> PairYields {
    pair.solve(birth).unwrap().yields().clone()
}

fn assert_invalid<T: std::fmt::Debug>(r: Result<T>, name: &str) {
    match r {
        Err(QsoError::InvalidArgument { name: got, .. }) => assert_eq!(got, name),
        other => panic!("expected an invalid `{name}`, got {other:?}"),
    }
}

const BIRTHS: [Birth; 6] = [
    Birth::Singlet,
    Birth::TripletZero,
    Birth::TripletPlus,
    Birth::TripletMinus,
    Birth::Triplet,
    Birth::Uncorrelated,
];

/// The pair's Liouvillian on column-stacked `vec ρ` from Kronecker products
/// of its Hamiltonian, projectors and the system's jumps
/// (`vec(AρB) = (Bᵀ ⊗ A) vec ρ`): independent of the blocks and of the
/// Sylvester form.
fn liouvillian(pair: &RadicalPair) -> CMatrix {
    let sys = pair.system();
    let h = sys.hamiltonian().unwrap();
    let d = h.nrows();
    let id = CMatrix::identity(d, d);
    let ps = pair.projector(Birth::Singlet);
    let pt = pair.projector(Birth::Triplet);
    let [ks, kt, ke] = pair.rates();
    let k = &ps * C64::new(ks, 0.0) + &pt * C64::new(kt, 0.0);
    let mut l = (id.kronecker(&h) - h.transpose().kronecker(&id)) * (-I)
        - (id.kronecker(&k) + k.transpose().kronecker(&id)) * C64::new(0.5, 0.0)
        - CMatrix::identity(d * d, d * d) * C64::new(ke, 0.0);
    if pair.recombination() == Recombination::JonesMaedaHore {
        let c = 0.5 * (ks + kt);
        l -= (pt.transpose().kronecker(&ps) + ps.transpose().kronecker(&pt)) * C64::new(c, 0.0);
    }
    for (i, op) in sys.jumps(sys.field()) {
        let a = sys.embed(i, &op).unwrap();
        let ada = a.adjoint() * &a;
        l += a.conjugate().kronecker(&a);
        l -= (id.kronecker(&ada) + ada.transpose().kronecker(&id)) * C64::new(0.5, 0.0);
    }
    l
}

/// `X` with `𝓛X = −ρ₀` by LU of [`liouvillian`].
fn liouville_x(pair: &RadicalPair, rho0: &CMatrix) -> CMatrix {
    let d = rho0.nrows();
    let b = nalgebra::DVector::from_iterator(d * d, rho0.iter().map(|z| -z));
    let x = liouvillian(pair).lu().solve(&b).unwrap();
    CMatrix::from_column_slice(d, d, x.as_slice())
}

#[test]
fn births_are_projectors_and_match_parahydrogen() {
    // The electrons at indices 0 and 3, not adjacent.
    let mut sys = system(0.1, Frame::Laboratory);
    let e1 = sys.add_electron(FREE_ELECTRON_G).unwrap();
    let h = nucleus(&mut sys, Isotope::H1);
    let n = nucleus(&mut sys, Isotope::N14);
    let e2 = sys.add_electron(2.003).unwrap();
    sys.hyperfine(e2, h, 0.3).unwrap();
    sys.hyperfine(e1, n, 0.2).unwrap();
    let pair = RadicalPair::new(sys.clone()).unwrap();
    assert_eq!(pair.electrons(), [0, 3]);
    assert_eq!(pair.nuclei(), vec![1, 2]);
    let s = pair.birth_state(Birth::Singlet);
    let t = pair.birth_state(Birth::Triplet);
    assert!(largest(&(&s - sys.parahydrogen_state(e1, e2, 1.0).unwrap())) <= 1e-15);
    assert!(largest(&(&t - sys.parahydrogen_state(e1, e2, 0.0).unwrap())) <= 1e-15);
    let mean = (pair.birth_state(Birth::Singlet)
        + pair.birth_state(Birth::TripletZero)
        + pair.birth_state(Birth::TripletPlus)
        + pair.birth_state(Birth::TripletMinus))
        * C64::new(0.25, 0.0);
    let uncorrelated = pair.birth_state(Birth::Uncorrelated);
    assert!(largest(&(&mean - &uncorrelated)) <= 1e-15);
    assert!(largest(&(&uncorrelated - sys.identity_state())) <= 1e-15);
    let d = sys.dimension() as f64;
    for (birth, trace) in BIRTHS.into_iter().zip([0.25, 0.25, 0.25, 0.25, 0.75, 1.0]) {
        let p = pair.projector(birth);
        assert!(largest(&(&p * &p - &p)) <= 1e-15, "{birth:?}");
        assert!(largest(&(&p - p.adjoint())) == 0.0, "{birth:?}");
        assert!((p.trace().re - trace * d).abs() <= 1e-12, "{birth:?}");
        assert!((pair.birth_state(birth).trace().re - 1.0).abs() <= 1e-15);
    }
    // P_T₀ + P_T₊ + P_T₋ = P_T, and P_S + P_T = 1.
    let sum = pair.projector(Birth::TripletZero)
        + pair.projector(Birth::TripletPlus)
        + pair.projector(Birth::TripletMinus);
    assert!(largest(&(&sum - pair.projector(Birth::Triplet))) <= 1e-15);
    let one = pair.projector(Birth::Singlet) + pair.projector(Birth::Triplet);
    assert!(largest(&(one - CMatrix::identity(24, 24))) <= 1e-15);
}

/// `Δω = H[αβ, αβ] − H[βα, βα]` of a pair without nuclei (electrons 0, 1).
fn delta_omega(pair: &RadicalPair) -> f64 {
    let h = pair.system().hamiltonian().unwrap();
    h[(1, 1)].re - h[(2, 2)].re
}

#[test]
fn a_g_difference_at_high_field_gives_the_closed_form() {
    let make = |frame: Frame, b: f64, g1: f64, g2: f64| {
        let mut sys = system(b, frame);
        sys.add_electron(g1).unwrap();
        sys.add_electron(g2).unwrap();
        RadicalPair::new(sys).unwrap()
    };
    let pair = make(Frame::Rotating, 0.3, 2.0031, 2.0027);
    let dw = delta_omega(&pair);
    assert!(
        (dw / 1.055_292_006_461_082_6e7 - 1.0).abs() <= 1e-12,
        "{dw}"
    );
    let mu_b = 9.274_010_078_3e-24;
    let want = (2.0031 - 2.0027) * mu_b * 0.3 / crate::spin::HBAR;
    assert!((dw / want - 1.0).abs() <= 1e-9);
    let closed = |k: f64, dw: f64| 0.5 + 0.5 * k * k / (k * k + dw * dw);
    for (k, want) in [(dw, 0.75), (3e7, 0.944_943_606_620_163_1)] {
        let p = pair.clone().with_rates(k, k, 0.0).unwrap();
        let y = yields(&p, Birth::Singlet);
        assert!((y.singlet - want).abs() <= 1e-13, "{k}: {}", y.singlet);
        assert!((closed(k, dw) - want).abs() <= 1e-15);
        assert!((sum(&y) - 1.0).abs() <= 1e-13);
        let t0 = yields(&p, Birth::TripletZero);
        assert!((t0.singlet - (1.0 - closed(k, dw))).abs() <= 1e-13);
        // The laboratory frame, Δω read from its Hamiltonian.
        let lab = make(Frame::Laboratory, 0.3, 2.0031, 2.0027)
            .with_rates(k, k, 0.0)
            .unwrap();
        let dw_lab = delta_omega(&lab);
        let y = yields(&lab, Birth::Singlet);
        assert!((y.singlet - closed(k, dw_lab)).abs() <= 1e-13);
        assert!((y.singlet - want).abs() <= 1e-12);
    }
    for dg in [1e-3, 3e-3, -2e-4] {
        for b in [0.35, 9.4] {
            for k in [1e7, 1e8, 1e9] {
                let p = make(Frame::Rotating, b, FREE_ELECTRON_G + dg, FREE_ELECTRON_G)
                    .with_rates(k, k, 0.0)
                    .unwrap();
                let y = yields(&p, Birth::Singlet);
                let want = closed(k, delta_omega(&p));
                assert!((y.singlet - want).abs() <= 1e-13, "{dg} {b} {k}");
            }
        }
    }
}

#[test]
fn one_nucleus_at_zero_field_gives_the_closed_forms() {
    let pair = |isotope: Isotope, k: f64| {
        let mut sys = system(0.0, Frame::Laboratory);
        let e1 = sys.add_electron(FREE_ELECTRON_G).unwrap();
        sys.add_electron(FREE_ELECTRON_G).unwrap();
        let n = nucleus(&mut sys, isotope);
        sys.hyperfine(e1, n, 1.0).unwrap();
        RadicalPair::new(sys)
            .unwrap()
            .with_rates(k, k, 0.0)
            .unwrap()
    };
    let a = GAMMA_E * 1e-3;
    let table = [
        (Isotope::H1, 0.7, 0.6625),
        (Isotope::N14, 0.6, 49.0 / 85.0),
        (Isotope::H2, 0.6, 49.0 / 85.0),
    ];
    for (isotope, half, third) in table {
        for (f, want) in [(0.5, half), (1.0 / 3.0, third)] {
            let y = yields(&pair(isotope, f * a), Birth::Singlet);
            assert!(
                (y.singlet - want).abs() <= 1e-13,
                "{isotope:?} {f}: {}",
                y.singlet
            );
            assert!((sum(&y) - 1.0).abs() <= 1e-13);
        }
        // 1 − c + c·k²/(k² + (I + ½)²a²), c = 2I(I + 1)/(2I + 1)².
        let spin = isotope.spin();
        let c = 2.0 * spin * (spin + 1.0) / (2.0 * spin + 1.0).powi(2);
        for f in [0.2, 3.0] {
            let k = f * a;
            let want = 1.0 - c + c * k * k / (k * k + (spin + 0.5).powi(2) * a * a);
            let y = yields(&pair(isotope, k), Birth::Singlet);
            assert!((y.singlet - want).abs() <= 1e-13, "{isotope:?} {f}");
        }
    }
}

#[test]
fn every_nuclear_state_at_high_field_gives_the_closed_form() {
    let pair = high_field();
    let sys = pair.system();
    let h = sys.hamiltonian().unwrap();
    // ω₁ − ω₂ from the Hamiltonian (no nuclei flipped: states 0b01·… and
    // 0b10·… of the electrons with the nuclei at their first level).
    let block = sys.dimension() / 4;
    let dw0 = h[(block, block)].re - h[(2 * block, 2 * block)].re;
    let coupling = |i: usize, j: usize| {
        sys.couplings()
            .iter()
            .find(|c| (c.0, c.1) == (i, j))
            .unwrap()
            .2
            * 2.0
            * std::f64::consts::PI
    };
    let (aa, ab, an) = (coupling(0, 2), coupling(0, 3), coupling(1, 4));
    // H = −2J S₁·S₂: the coupling is −2J.
    let d = -coupling(0, 1);
    // The hyperfine shifts of the S–T₀ splitting also enter the diagonal
    // read above: take them off (nuclei at m = ½, ½, 1).
    let dw0 = dw0 - (aa * 0.5 + ab * 0.5 - an * 1.0);
    let [ks, kt, ke] = pair.rates();
    let (s, t) = (ks + ke, kt + ke);
    let mut forms = [[0.0; 3]; 2];
    for ma in [0.5, -0.5] {
        for mb in [0.5, -0.5] {
            for mn in [1.0, 0.0, -1.0] {
                let w = dw0 + aa * ma + ab * mb - an * mn;
                let den = (s + t).powi(2) * (w * w + s * t) + 4.0 * d * d * s * t;
                let mixed = w * w * (s + t) / den;
                let ss = (w * w * (s + t) + t * ((s + t).powi(2) + 4.0 * d * d)) / den;
                let tt = (w * w * (s + t) + s * ((s + t).powi(2) + 4.0 * d * d)) / den;
                forms[0][0] += ks * ss / 12.0;
                forms[0][1] += kt * mixed / 12.0;
                forms[0][2] += ke * (ss + mixed) / 12.0;
                forms[1][0] += ks * mixed / 12.0;
                forms[1][1] += kt * tt / 12.0;
                forms[1][2] += ke * (mixed + tt) / 12.0;
            }
        }
    }
    let table = [
        (
            Birth::Singlet,
            [
                0.719_876_116_257_715_1,
                0.188_444_312_519_130_1,
                0.091_679_571_223_154_66,
            ],
        ),
        (
            Birth::TripletZero,
            [
                0.502_518_166_717_680_3,
                0.383_160_808_982_078,
                0.114_321_024_300_241_62,
            ],
        ),
    ];
    for (k, (birth, want)) in table.into_iter().enumerate() {
        let y = yields(&pair, birth);
        for (got, (w, form)) in [y.singlet, y.triplet, y.escape]
            .into_iter()
            .zip(want.iter().zip(forms[k]))
        {
            assert!((got - w).abs() <= 1e-13, "{birth:?}: {got} vs {w}");
            assert!(
                (form - w).abs() <= 1e-13,
                "{birth:?}: formula {form} vs {w}"
            );
        }
    }
    for birth in [Birth::TripletPlus, Birth::TripletMinus] {
        let y = yields(&pair, birth);
        assert!(y.singlet.abs() <= 1e-15, "{}", y.singlet);
        assert!((y.triplet - 5.0 / 6.0).abs() <= 1e-15);
        assert!((y.escape - 1.0 / 6.0).abs() <= 1e-15);
    }
}

#[test]
fn starts_that_never_decay_are_refused() {
    let one_proton = |b: f64, frame: Frame, dg: f64| {
        let mut sys = system(b, frame);
        let e1 = sys.add_electron(FREE_ELECTRON_G + dg).unwrap();
        sys.add_electron(FREE_ELECTRON_G).unwrap();
        let h = nucleus(&mut sys, Isotope::H1);
        sys.hyperfine(e1, h, 1.0).unwrap();
        RadicalPair::new(sys).unwrap()
    };
    let zero = one_proton(0.0, Frame::Laboratory, 0.0);
    for model in [Recombination::Haberkorn, Recombination::JonesMaedaHore] {
        let p = zero
            .clone()
            .with_rates(1e7, 0.0, 0.0)
            .unwrap()
            .with_recombination(model);
        for birth in BIRTHS {
            assert_invalid(p.solve(birth), "rates");
        }
        let escaping = p.with_rates(1e7, 0.0, 1e5).unwrap();
        for birth in BIRTHS {
            let y = yields(&escaping, birth);
            assert!((sum(&y) - 1.0).abs() <= 1e-12, "{model:?} {birth:?}");
        }
    }
    // At high field in the rotating frame a singlet-born pair never meets
    // its T± states: accepted, and every pair recombines as a singlet.
    let high = one_proton(0.35, Frame::Rotating, 2e-3)
        .with_rates(1e7, 0.0, 0.0)
        .unwrap();
    let y = yields(&high, Birth::Singlet);
    assert!((y.singlet - 1.0).abs() <= 1e-12, "{}", y.singlet);
    assert_invalid(high.solve(Birth::TripletPlus), "rates");
}

#[test]
fn states_that_relaxation_leaves_dark_are_refused() {
    // Zero field, one proton, k_S only and electron dephasing only: the
    // stretched T₊ state |ααα⟩ is stationary (a 1 × 1 block on which 𝓛 is
    // exactly zero, though the preconditioner's gap is not). GMRES returned
    // an X grown without bound with a backward error of 4·10⁻¹⁷ (Φ_S =
    // −1.19); the residual check refuses it.
    let mut sys = system(0.0, Frame::Laboratory);
    let e1 = sys.add_electron(FREE_ELECTRON_G).unwrap();
    let e2 = sys.add_electron(FREE_ELECTRON_G).unwrap();
    let h = nucleus(&mut sys, Isotope::H1);
    sys.hyperfine(e1, h, 1.0).unwrap();
    for e in [e1, e2] {
        sys.set_relaxation(e, None, Some(1e-6)).unwrap();
    }
    let pair = RadicalPair::new(sys)
        .unwrap()
        .with_rates(1e7, 0.0, 0.0)
        .unwrap();
    for birth in [Birth::Triplet, Birth::TripletPlus, Birth::Uncorrelated] {
        assert_invalid(pair.solve(birth), "rates");
    }
    // The singlet-born pair never reaches |ααα⟩: every pair recombines.
    let y = yields(&pair, Birth::Singlet);
    assert!((y.singlet - 1.0).abs() <= 1e-12, "{}", y.singlet);
    // An escape rate is the remedy.
    let escaping = pair.with_rates(1e7, 0.0, 1e5).unwrap();
    let y = yields(&escaping, Birth::Triplet);
    assert!((sum(&y) - 1.0).abs() <= 1e-12);
    // Dephasing of a spin-1 nucleus leaves its m = 0 states alone: in the
    // rotating frame T±|m = 0⟩ never decays with k_S alone, in blocks a
    // singlet start never reaches (the jumps keep the labels). Only the
    // blocks the start reaches are solved: accepted, with every pair
    // recombining (it was refused with T₂ alone, accepted with T₁).
    for (t1, t2) in [
        (None, Some(1e-4)),
        (Some(1e-3), None),
        (Some(1e-3), Some(1e-4)),
    ] {
        let mut sys = system(0.3, Frame::Rotating);
        let e1 = sys.add_electron(2.0031).unwrap();
        let e2 = sys.add_electron(2.0026).unwrap();
        let h = nucleus(&mut sys, Isotope::H1);
        let n = nucleus(&mut sys, Isotope::N14);
        sys.hyperfine(e1, h, 0.8).unwrap();
        sys.hyperfine(e2, n, 0.5).unwrap();
        sys.set_relaxation(n, t1, t2).unwrap();
        let pair = RadicalPair::new(sys)
            .unwrap()
            .with_rates(1e8, 0.0, 0.0)
            .unwrap();
        let y = yields(&pair, Birth::Singlet);
        assert!(
            (y.singlet - 1.0).abs() <= 1e-12,
            "{t1:?} {t2:?}: {}",
            y.singlet
        );
        // A triplet start reaches those states and is refused.
        assert_invalid(pair.solve(Birth::TripletPlus), "rates");
    }
    // Rates beyond the arithmetic: the Sylvester path returned X = 0 (all
    // yields 0, backward error 1).
    let mut sys = system(1e-3, Frame::Laboratory);
    let e1 = sys.add_electron(FREE_ELECTRON_G).unwrap();
    sys.add_electron(FREE_ELECTRON_G).unwrap();
    let h = nucleus(&mut sys, Isotope::H1);
    sys.hyperfine(e1, h, 1.0).unwrap();
    let pair = RadicalPair::new(sys).unwrap();
    for model in [Recombination::Haberkorn, Recombination::JonesMaedaHore] {
        let p = pair
            .clone()
            .with_rates(1e300, 1e300, 1e300)
            .unwrap()
            .with_recombination(model);
        assert!(p.solve(Birth::Singlet).is_err(), "{model:?}");
    }
}

/// The exponential model (`k_S = k_T = k`, `A` normal) by the
/// eigenvectors of `H`: `Φ_S = Σ_ij k/(k + k_E + i(ε_i − ε_j)) p_ji r_ij`
/// with `p` and `r` the projector and the start in that basis.
fn exponential_singlet_yield(pair: &RadicalPair, birth: Birth) -> f64 {
    let [k, _, ke] = pair.rates();
    let (e, v) = crate::linalg::eigh(&pair.system().hamiltonian().unwrap());
    let r = v.adjoint() * pair.birth_state(birth) * &v;
    let p = v.adjoint() * pair.projector(Birth::Singlet) * &v;
    let n = e.len();
    let mut total = ZERO;
    for i in 0..n {
        for j in 0..n {
            total += p[(j, i)] * r[(i, j)] * k / C64::new(k + ke, e[i] - e[j]);
        }
    }
    total.re
}

#[test]
fn slow_rates_in_the_laboratory_frame_at_high_field_are_solved() {
    // ‖𝓛‖ ≈ 2|γ_e|B is 10¹¹ times rates of 10 s⁻¹ at 3 T: the residual of an
    // accurate X is near 10⁻⁵, refused by a fixed bound of 10⁻⁶ before; the
    // bound now scales with the slowest decay.
    let make = |b: f64| {
        let mut sys = system(b, Frame::Laboratory);
        let e1 = electron(&mut sys, 2e-3);
        let e2 = electron(&mut sys, 0.0);
        let h = nucleus(&mut sys, Isotope::H1);
        let n = nucleus(&mut sys, Isotope::N14);
        sys.hyperfine(e1, h, 0.8).unwrap();
        sys.hyperfine(e2, n, 0.5).unwrap();
        RadicalPair::new(sys).unwrap()
    };
    for b in [0.05, 1.0, 3.0] {
        for (k, ke) in [(10.0, 0.0), (10.0, 10.0), (100.0, 0.0)] {
            let pair = make(b).with_rates(k, k, ke).unwrap();
            for birth in [Birth::Singlet, Birth::Triplet, Birth::TripletPlus] {
                let y = yields(&pair, birth);
                let want = exponential_singlet_yield(&pair, birth);
                // 1.7·10⁻⁹ at most (3 T, k = 100 s⁻¹) against the same
                // formula in 50 digits, where the reference here errs by
                // 2·10⁻¹⁶: Schur's rounding of order ε‖A‖ divided by k.
                assert!(
                    (y.singlet - want).abs() <= 1e-8,
                    "{b} {k} {birth:?}: {} vs {want}",
                    y.singlet
                );
                assert!(
                    (sum(&y) - 1.0).abs() <= 1e-8,
                    "{b} {k} {birth:?}: {:e}",
                    sum(&y) - 1.0
                );
            }
        }
        // k = 0.1 s⁻¹ leaves a residual near 10⁻³ and errs by 2·10⁻⁸ (50
        // digits, 3 T).
        let slower = make(b).with_rates(0.1, 0.1, 0.0).unwrap();
        let y = yields(&slower, Birth::Singlet);
        let want = exponential_singlet_yield(&slower, Birth::Singlet);
        assert!(
            (y.singlet - want).abs() <= 1e-7,
            "{b}: {} vs {want}",
            y.singlet
        );
        // Unequal rates, k_T and k_E slow: the sums.
        let pair = make(b).with_rates(1e4, 10.0, 10.0).unwrap();
        for birth in [Birth::Singlet, Birth::Triplet] {
            let y = yields(&pair, birth);
            assert!((sum(&y) - 1.0).abs() <= 1e-9, "{b} {birth:?}");
            assert!(
                [y.singlet, y.triplet, y.escape]
                    .iter()
                    .all(|v| (-1e-9..=1.0 + 1e-9).contains(v))
            );
        }
    }
    // Rates of 10⁻³ s⁻¹ at 3 T decay at less than 10⁻¹⁴ of the largest
    // frequency: refused, with the frame as a remedy.
    let tiny = make(3.0).with_rates(1e-3, 1e-3, 0.0).unwrap();
    match tiny.solve(Birth::Singlet) {
        Err(QsoError::InvalidArgument { name, reason }) => {
            assert_eq!(name, "rates");
            assert!(reason.contains("rotating frame"), "{reason}");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn bad_pairs_and_starts_are_refused() {
    let mut one = system(0.0, Frame::Laboratory);
    one.add_electron(FREE_ELECTRON_G).unwrap();
    nucleus(&mut one, Isotope::H1);
    assert_invalid(RadicalPair::new(one.clone()), "system");
    let mut three = one.clone();
    three.add_electron(2.0).unwrap();
    three.add_electron(2.0).unwrap();
    assert_invalid(RadicalPair::new(three), "system");
    let mut redfield = one.clone();
    redfield.add_electron(2.0).unwrap();
    let redfield = redfield.with_correlation_time(Some(1e-10)).unwrap();
    assert_invalid(RadicalPair::new(redfield), "system");
    let pair = two_protons(Frame::Laboratory, 1e-3);
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        assert_invalid(pair.clone().with_rates(bad, 0.0, 0.0), "rates");
        assert_invalid(pair.clone().with_rates(0.0, 0.0, bad), "rates");
    }
    assert_invalid(pair.clone().with_field(-1.0), "field");
    let idle = pair.clone().with_rates(0.0, 0.0, 0.0).unwrap();
    assert_invalid(idle.solve(Birth::Singlet), "rates");
    let d = pair.system().dimension();
    let mut skew = pair.birth_state(Birth::Singlet);
    skew[(0, 1)] += C64::new(0.1, 0.0);
    assert_invalid(pair.solve_from(&skew), "rho0");
    assert_invalid(pair.solve_from(&CMatrix::zeros(d, d)), "rho0");
    let mut nan = pair.birth_state(Birth::Singlet);
    nan[(3, 3)] = C64::new(f64::NAN, 0.0);
    assert_invalid(pair.solve_from(&nan), "rho0");
    assert!(matches!(
        pair.solve_from(&CMatrix::zeros(3, 3)),
        Err(QsoError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        pair.evolve(&CMatrix::zeros(3, 3), 1e-9),
        Err(QsoError::DimensionMismatch { .. })
    ));
    assert_invalid(pair.evolve(&pair.birth_state(Birth::Singlet), -1.0), "t");
    assert_invalid(pair.field_effect(Birth::Singlet, &[0.0, f64::NAN]), "field");
    // Any positive trace is normalised.
    let twice = pair.birth_state(Birth::Singlet) * C64::new(2.0, 0.0);
    assert_eq!(
        pair.solve_from(&twice).unwrap().yields(),
        pair.solve(Birth::Singlet).unwrap().yields()
    );
}

#[test]
fn sylvester_solves_equal_liouville_solves() {
    let pair = two_protons(Frame::Laboratory, 1e-3);
    let table = [
        (
            Birth::Singlet,
            [
                0.694_062_627_181_530_1,
                0.207_733_224_805_094_85,
                0.098_204_148_013_375_72,
            ],
        ),
        (
            Birth::TripletZero,
            [
                0.528_242_815_196_147_1,
                0.349_232_797_699_288_33,
                0.122_524_387_104_565_2,
            ],
        ),
        (
            Birth::TripletPlus,
            [
                0.111_067_326_940_132_09,
                0.705_222_547_677_753_7,
                0.183_710_125_382_113_9,
            ],
        ),
        (
            Birth::TripletMinus,
            [
                0.139_689_450_882_826_67,
                0.680_798_335_246_654_1,
                0.179_512_213_870_518_63,
            ],
        ),
        (
            Birth::Triplet,
            [
                0.259_666_531_006_368_66,
                0.578_417_893_541_231_9,
                0.161_915_575_452_399_24,
            ],
        ),
    ];
    for (birth, want) in table {
        let solution = pair.solve(birth).unwrap();
        let x = solution.integral();
        let oracle = liouville_x(&pair, &pair.birth_state(birth));
        let e = relative(x, &oracle);
        assert!(e <= 1e-13, "{birth:?}: {e:e}");
        let y = solution.yields();
        assert_eq!(y.iterations, 0);
        assert!(y.backward_error <= 1e-14, "{:e}", y.backward_error);
        for (got, w) in [y.singlet, y.triplet, y.escape].into_iter().zip(want) {
            assert!((got - w).abs() <= 1e-13, "{birth:?}: {got} vs {w}");
        }
    }
    let solution = pair.solve(Birth::Singlet).unwrap();
    let cidnp = &solution.yields().cidnp;
    let want = [
        [
            -0.008_415_589_688_610_586,
            0.013_045_661_244_178_282,
            0.002_700_375_998_470_537,
        ],
        [
            0.011_587_716_401_277_204,
            -0.008_119_976_320_934_804,
            -0.001_257_479_653_481_890_7,
        ],
    ];
    let sys = pair.system();
    let h = sys.hamiltonian().unwrap();
    let x = solution.integral();
    for (j, (got, want)) in pair.nuclei().into_iter().zip(cidnp.iter().zip(want)) {
        for p in 0..3 {
            assert!((got[p] - want[p]).abs() <= 1e-14, "{j} {p}: {}", got[p]);
        }
        // Σ_P ⟨I_z⟩_P = ⟨I_z⟩₀ − i Tr(X [I_z, H]), not zero here.
        let iz = sys.iz(j).unwrap();
        let comm = &iz * &h - &h * &iz;
        let t = (x * comm).trace() * (-I);
        let total: f64 = got.iter().sum();
        assert!((total - t.re).abs() <= 1e-15, "{j}: {total} vs {}", t.re);
        assert!(total.abs() > 1e-6);
    }
}

#[test]
fn yields_sum_to_one_on_both_paths() {
    let relaxations = [None, Some(298.15), Some(1.0)];
    for (frame, b) in [(Frame::Laboratory, 1e-3), (Frame::Rotating, 0.35)] {
        for temperature in relaxations {
            let mut pair = two_protons(frame, b);
            if let Some(kelvin) = temperature {
                let mut sys = pair.system().clone().with_temperature(kelvin).unwrap();
                for e in pair.electrons() {
                    sys.set_relaxation(e, Some(200e-9), Some(100e-9)).unwrap();
                }
                pair = RadicalPair::new(sys)
                    .unwrap()
                    .with_rates(3e7, 8e6, 2e6)
                    .unwrap();
            }
            for model in [Recombination::Haberkorn, Recombination::JonesMaedaHore] {
                let p = pair.clone().with_recombination(model);
                let direct = temperature.is_none() && model == Recombination::Haberkorn;
                for birth in BIRTHS {
                    let y = yields(&p, birth);
                    let tol = if direct { 1e-13 } else { 1e-12 };
                    let e = (sum(&y) - 1.0).abs();
                    assert!(
                        e <= tol,
                        "{frame:?} {temperature:?} {model:?} {birth:?}: {e:e}"
                    );
                    assert_eq!(y.iterations == 0, direct);
                    if direct {
                        let rho0 = p.birth_state(birth);
                        let forced = Options {
                            force_gmres: true,
                            whole: false,
                        };
                        let g = p.solve_with(&rho0, b, forced).unwrap();
                        assert!(g.yields().iterations > 0);
                        let e = relative(g.integral(), p.solve(birth).unwrap().integral());
                        assert!(e <= 1e-12, "{frame:?} {birth:?}: {e:e}");
                        for (a, c) in [
                            (g.yields().singlet, y.singlet),
                            (g.yields().triplet, y.triplet),
                            (g.yields().escape, y.escape),
                        ] {
                            assert!((a - c).abs() <= 1e-12);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn relaxation_and_the_jones_maeda_hore_term_match_liouville_space() {
    let base = two_protons(Frame::Laboratory, 1e-3);
    let mut sys = base.system().clone();
    for e in base.electrons() {
        sys.set_relaxation(e, Some(200e-9), Some(100e-9)).unwrap();
    }
    // A relaxing proton too.
    sys.set_relaxation(2, Some(1e-6), Some(5e-7)).unwrap();
    let relaxing = RadicalPair::new(sys)
        .unwrap()
        .with_rates(3e7, 8e6, 2e6)
        .unwrap();
    let b = relaxing.system().field();
    for pair in [
        relaxing.clone(),
        relaxing
            .clone()
            .with_recombination(Recombination::JonesMaedaHore),
        base.clone()
            .with_recombination(Recombination::JonesMaedaHore),
    ] {
        let model = pair.model(b, false);
        let g = model.generator_dense();
        let all: Vec<usize> = (0..model.blocks.len()).collect();
        for birth in BIRTHS {
            let rho0 = pair.birth_state(birth);
            let solution = pair.solve(birth).unwrap();
            let y = solution.yields();
            assert!(y.iterations > 0);
            assert!(y.backward_error <= 1e-12, "{:e}", y.backward_error);
            // generator_dense with LU.
            let rhs = nalgebra::DVector::from_iterator(
                g.nrows(),
                model.vector_of(&rho0, &all).into_iter().map(|z| -z),
            );
            let v = g.clone().lu().solve(&rhs).unwrap();
            let x = model.matrix_of(v.as_slice(), &all, rho0.nrows());
            let lu = pair.observe(&x, b, 0.0, 0);
            for (a, c) in [
                (y.singlet, lu.singlet),
                (y.triplet, lu.triplet),
                (y.escape, lu.escape),
            ] {
                assert!((a - c).abs() <= 1e-12, "{birth:?}: {a} vs {c}");
            }
            // And the Kronecker oracle.
            let e = relative(solution.integral(), &liouville_x(&pair, &rho0));
            assert!(e <= 1e-12, "{birth:?}: {e:e}");
            assert!(relative(&x, &liouville_x(&pair, &rho0)) <= 1e-12);
        }
    }
}

#[test]
fn without_rates_the_pair_evolves_as_the_spin_system() {
    for (b, frame) in [(1e-3, Frame::Laboratory), (0.3, Frame::Rotating)] {
        let sys = nitrogen_pair(b, frame);
        let pair = RadicalPair::new(sys.clone()).unwrap();
        let rho = pair.birth_state(Birth::Singlet);
        let mine = pair.evolve(&rho, 50e-9).unwrap();
        let theirs = sys.evolve(&rho, &[Event::Delay(50e-9)]).unwrap();
        let e = largest(&(&mine - &theirs));
        assert!(e <= 1e-13, "{frame:?}: {e:e}");
        // With electron relaxation: the dense Liouville path.
        let mut relaxing = sys.clone().with_temperature(300.0).unwrap();
        for e in pair.electrons() {
            relaxing
                .set_relaxation(e, Some(1e-6), Some(0.4e-6))
                .unwrap();
        }
        let pair = RadicalPair::new(relaxing.clone()).unwrap();
        let mine = pair.evolve(&rho, 50e-9).unwrap();
        let theirs = relaxing.evolve(&rho, &[Event::Delay(50e-9)]).unwrap();
        let e = largest(&(&mine - &theirs));
        assert!(e <= 1e-11, "{frame:?} relaxing: {e:e}");
    }
}

#[test]
fn the_evolution_integrates_to_the_yields() {
    // k = 10⁸: steps of 10⁻¹⁰ s resolve the hyperfine frequencies
    // (hω ≈ 0.03, so Simpson's rule errs by about (hω)⁴/180).
    let k = 1e8;
    let pair = two_protons(Frame::Laboratory, 1e-3)
        .with_rates(k, k, 0.0)
        .unwrap();
    let b = pair.system().field();
    let rho0 = pair.birth_state(Birth::Singlet);
    // Against the exponential of the dense generator.
    let model = pair.model(b, false);
    let all: Vec<usize> = (0..model.blocks.len()).collect();
    let t = 3e-9;
    let g = model.generator_dense() * C64::new(t, 0.0);
    let v =
        crate::linalg::dense::expm(&g) * nalgebra::DVector::from_vec(model.vector_of(&rho0, &all));
    let want = model.matrix_of(v.as_slice(), &all, rho0.nrows());
    let got = pair.evolve(&rho0, t).unwrap();
    assert!(largest(&(&got - &want)) <= 1e-13);
    // Simpson's rule for k_S ∫ Tr(P_S ρ(t)) dt over 40/k.
    let ps = pair.projector(Birth::Singlet);
    let points = 4001;
    let h = 40.0 / k / (points - 1) as f64;
    let mut integral = 0.0;
    let mut last_trace = f64::INFINITY;
    for i in 0..points {
        let rho = pair.evolve(&rho0, i as f64 * h).unwrap();
        let trace = rho.trace().re;
        assert!(trace <= last_trace + 1e-15);
        last_trace = trace;
        let w = if i == 0 || i == points - 1 {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        integral += w * (&ps * &rho).trace().re;
    }
    let simpson = k * integral * h / 3.0;
    let y = yields(&pair, Birth::Singlet);
    assert!(
        (simpson - y.singlet).abs() <= 1e-7,
        "{simpson} vs {}",
        y.singlet
    );
}

#[test]
fn the_frames_agree_at_high_field() {
    // Both frames against an independent Liouville-space build (numpy,
    // QSO's conventions). They differ by 3.9·10⁻⁶, the size of the
    // flip-flop terms the rotating frame drops: about 11 Σ_n (a_n/ω_e)²
    // here.
    let k = 1e8;
    let pair = |frame| {
        RadicalPair::new(nitrogen_pair(3.0, frame))
            .unwrap()
            .with_rates(k, k, 0.0)
            .unwrap()
    };
    let lab = yields(&pair(Frame::Laboratory), Birth::Singlet).singlet;
    let rot = yields(&pair(Frame::Rotating), Birth::Singlet).singlet;
    assert!((lab - 0.524_084_992_454_291_3).abs() <= 1e-12, "{lab}");
    assert!((rot - 0.524_081_083_879_154_5).abs() <= 1e-13, "{rot}");
    let w = GAMMA_E * 3.0;
    let flip_flops: f64 = [0.514, 1.6045, -0.5983]
        .iter()
        .map(|a| (a * GAMMA_E * 1e-3 / w).powi(2))
        .sum();
    assert!((lab - rot).abs() <= 20.0 * flip_flops, "{:e}", lab - rot);
}

#[test]
fn starts_with_coherences_between_labels_take_one_block() {
    let pair = two_protons(Frame::Laboratory, 1e-3);
    let sys = pair.system();
    // Transverse coherence of the first proton.
    let rho0 = pair.birth_state(Birth::Singlet) + sys.ix(2).unwrap() * C64::new(0.01, 0.0);
    let solution = pair.solve_from(&rho0).unwrap();
    let e = relative(solution.integral(), &liouville_x(&pair, &rho0));
    assert!(e <= 1e-13, "{e:e}");
    // A block-diagonal start forced into one block.
    let b = sys.field();
    for model in [Recombination::Haberkorn, Recombination::JonesMaedaHore] {
        let p = pair.clone().with_recombination(model);
        let rho0 = p.birth_state(Birth::Singlet);
        let whole = Options {
            force_gmres: false,
            whole: true,
        };
        let one = p.solve_with(&rho0, b, whole).unwrap();
        let blocks = p.solve(Birth::Singlet).unwrap();
        let e = relative(one.integral(), blocks.integral());
        assert!(e <= 1e-13, "{model:?}: {e:e}");
    }
}

#[cfg(feature = "parallel")]
#[test]
fn field_effects_do_not_depend_on_the_thread_count() {
    let mut sys = nitrogen_pair(0.0, Frame::Laboratory);
    sys.set_relaxation(0, Some(1e-6), Some(5e-7)).unwrap();
    let relaxing = RadicalPair::new(sys)
        .unwrap()
        .with_rates(2e7, 5e6, 1e6)
        .unwrap();
    let fields: Vec<f64> = (0..8).map(|k| k as f64 * 0.7e-3).collect();
    let plain = RadicalPair::new(nitrogen_pair(0.0, Frame::Laboratory))
        .unwrap()
        .with_rates(2e7, 5e6, 1e6)
        .unwrap();
    for pair in [
        plain.clone(),
        plain.with_recombination(Recombination::JonesMaedaHore),
        relaxing,
    ] {
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| pair.field_effect(Birth::Singlet, &fields).unwrap())
        };
        let (one, four) = (run(1), run(4));
        let bits = |ys: &[PairYields]| -> Vec<u64> {
            ys.iter()
                .flat_map(|y| {
                    let mut v = vec![
                        y.field.to_bits(),
                        y.singlet.to_bits(),
                        y.triplet.to_bits(),
                        y.escape.to_bits(),
                        y.backward_error.to_bits(),
                        y.iterations as u64,
                    ];
                    v.extend(y.cidnp.iter().flatten().map(|x| x.to_bits()));
                    v
                })
                .collect()
        };
        assert_eq!(bits(&one), bits(&four));
        assert_eq!(one.len(), 8);
    }
}

#[test]
fn products_cancel_nucleus_by_nucleus_in_the_rotating_frame() {
    let pair = high_field();
    for model in [Recombination::Haberkorn, Recombination::JonesMaedaHore] {
        let p = pair.clone().with_recombination(model);
        for birth in [
            Birth::Singlet,
            Birth::TripletZero,
            Birth::Triplet,
            Birth::Uncorrelated,
        ] {
            let y = yields(&p, birth);
            assert_eq!(y.cidnp.len(), 3);
            // GMRES stops at a backward error of 10⁻¹³: its sums cancel to
            // the 2·10⁻¹⁵ that leaves (of polarisations near 10⁻²).
            let tol = if model == Recombination::Haberkorn {
                1e-15
            } else {
                1e-14
            };
            for (j, c) in y.cidnp.iter().enumerate() {
                let total: f64 = c.iter().sum();
                assert!(total.abs() <= tol, "{model:?} {birth:?} {j}: {total:e}");
            }
        }
    }
}

#[test]
fn in_the_laboratory_frame_the_products_keep_the_total_angular_momentum() {
    // `[F_z, H] = [F_z, P_S] = 0` and `F_z = W F_z W`, so without
    // relaxation `Σ_P k_P Tr(F_z P_P X) = Tr(F_z ρ₀)` for both terms, with
    // the electrons' `S_z`; the nuclei's share alone moves with S–T±
    // mixing and does not cancel.
    let pair = two_protons(Frame::Laboratory, 1e-3);
    let sys = pair.system();
    let fz = (0..sys.spins().len())
        .map(|i| sys.iz(i).unwrap())
        .fold(CMatrix::zeros(16, 16), |a, b| a + b);
    let [e1, e2] = pair.electrons();
    let sz = sys.iz(e1).unwrap() + sys.iz(e2).unwrap();
    for model in [Recombination::Haberkorn, Recombination::JonesMaedaHore] {
        let p = pair.clone().with_recombination(model);
        for birth in BIRTHS {
            let solution = p.solve(birth).unwrap();
            let start = (&fz * p.birth_state(birth)).trace().re;
            let (mut total, mut nuclear, mut electrons) = (0.0, 0.0, 0.0);
            for (k, which) in [Product::Singlet, Product::Triplet, Product::Escape]
                .into_iter()
                .enumerate()
            {
                let m = solution.weighted(which);
                total += (&fz * &m).trace().re;
                electrons += (&sz * &m).trace().re;
                nuclear += solution.yields().cidnp.iter().map(|c| c[k]).sum::<f64>();
            }
            // GMRES (Jones, Maeda and Hore) stops at a backward error of
            // 10⁻¹³.
            let tol = if model == Recombination::Haberkorn {
                1e-14
            } else {
                1e-13
            };
            assert!(
                (total - start).abs() <= tol,
                "{model:?} {birth:?}: {total} vs {start}"
            );
            assert!((nuclear + electrons - start).abs() <= tol);
            if birth == Birth::Singlet {
                assert!(nuclear.abs() > 1e-4, "{model:?}: {nuclear:e}");
            }
        }
    }
    // With k_S = k_T and exchange, the escape product is polarised in the
    // laboratory frame (not in the rotating frame).
    let equal = pair.with_rates(2e7, 2e7, 1e6).unwrap();
    let y = yields(&equal, Birth::Singlet);
    assert!(y.cidnp.iter().all(|c| c[2].abs() > 1e-5), "{:?}", y.cidnp);
}

#[test]
fn polarisations_follow_kapteins_rule() {
    let pair = |dg: f64, a: f64, ks: f64, kt: f64| {
        let mut sys = system(0.35, Frame::Rotating);
        let e1 = electron(&mut sys, dg);
        electron(&mut sys, 0.0);
        let h = nucleus(&mut sys, Isotope::H1);
        sys.hyperfine(e1, h, a).unwrap();
        RadicalPair::new(sys)
            .unwrap()
            .with_rates(ks, kt, 1e7)
            .unwrap()
    };
    let mut smallest = f64::INFINITY;
    for (birth, mu) in [(Birth::Singlet, -1.0), (Birth::Triplet, 1.0)] {
        for dg in [2e-3, -2e-3] {
            for a in [1.0, -1.0] {
                let y = yields(&pair(dg, a, 1e8, 3e7), birth);
                let c = y.cidnp[0];
                for (p, eps) in [1.0, -1.0, -1.0].into_iter().enumerate() {
                    let sign = mu * eps * dg.signum() * a.signum();
                    assert_eq!(c[p].signum(), sign, "{birth:?} {dg} {a} {p}: {}", c[p]);
                    assert!(c[p].abs() >= 1e-3);
                    smallest = smallest.min(c[p].abs());
                }
                assert!(c.iter().sum::<f64>().abs() <= 1e-15);
                // Equal recombination rates: no polarisation escapes.
                let y = yields(&pair(dg, a, 1e8, 1e8), birth);
                assert!(y.cidnp[0][2].abs() <= 1e-17, "{:e}", y.cidnp[0][2]);
            }
        }
    }
    assert!(smallest > 6e-3, "{smallest:e}");
}

/// A system of the pair's nuclei alone, in their order.
fn nuclear_copy(pair: &RadicalPair, b: f64) -> SpinSystem {
    let mut sys = SpinSystem::new(b).with_frame(pair.system().frame());
    for j in pair.nuclei() {
        nucleus(&mut sys, pair.system().spins()[j].isotope);
    }
    sys
}

#[test]
fn products_are_nuclear_states_of_their_yields() {
    let pair = two_protons(Frame::Laboratory, 1e-3);
    let solution = pair.solve(Birth::Singlet).unwrap();
    let y = solution.yields();
    let products = [Product::Singlet, Product::Triplet, Product::Escape];
    let phis = [y.singlet, y.triplet, y.escape];
    let mut total = CMatrix::zeros(4, 4);
    for (p, phi) in products.into_iter().zip(phis) {
        let sigma = solution.product(p);
        assert_eq!(sigma.shape(), (4, 4));
        assert!(largest(&(&sigma - sigma.adjoint())) == 0.0);
        let (values, _) = crate::linalg::eigh(&sigma);
        assert!(values[0] >= -1e-15, "{p:?}: {}", values[0]);
        assert!((sigma.trace().re - phi).abs() <= 1e-14);
        total += sigma;
    }
    assert!((total.trace().re - 1.0).abs() <= 1e-13);
    // Σ_P product = Tr_e((K + k_E) X), from the integral and the projectors.
    let [ks, kt, ke] = pair.rates();
    let k = pair.projector(Birth::Singlet) * C64::new(ks, 0.0)
        + pair.projector(Birth::Triplet) * C64::new(kt + ke, 0.0)
        + pair.projector(Birth::Singlet) * C64::new(ke, 0.0);
    let m = k * solution.integral();
    // Electrons 0 and 1 are the most significant digits: Tr_e sums the
    // four diagonal blocks of 4 × 4.
    let mut want = CMatrix::zeros(4, 4);
    for e in 0..4 {
        want += m.view((4 * e, 4 * e), (4, 4));
    }
    assert!(largest(&(&total - &want)) <= 1e-14);
    // Onto a copy of the nuclei, every spin carried.
    let copy = nuclear_copy(&pair, 9.4);
    let routes = [(2, 0), (3, 1)];
    for (p, phi) in products.into_iter().zip(phis) {
        let on = solution
            .product_on(p, &copy, &routes, Fresh::Thermal)
            .unwrap();
        let want = solution.product(p).unscale(phi);
        assert!(largest(&(&on - &want)) <= 1e-15, "{p:?}");
        for (k, j) in pair.nuclei().into_iter().enumerate() {
            let pol = copy.polarization(&on, k).unwrap() * 0.5;
            let c = y.cidnp[k][p as usize] / phi;
            assert!((pol - c).abs() <= 1e-13, "{p:?} {j}: {pol} vs {c}");
        }
    }
    // An extra target spin arrives thermal.
    let mut extra = copy.clone();
    let c = nucleus(&mut extra, Isotope::C13);
    let on = solution
        .product_on(Product::Singlet, &extra, &routes, Fresh::Thermal)
        .unwrap();
    let back = CMatrix::from_fn(4, 4, |r, col| {
        on[(2 * r, 2 * col)] + on[(2 * r + 1, 2 * col + 1)]
    });
    let want = solution.product(Product::Singlet).unscale(y.singlet);
    assert!(largest(&(&back - &want)) <= 1e-15);
    let thermal = extra.thermal_state().unwrap();
    let pc = extra.polarization(&on, c).unwrap();
    assert!((pc - extra.polarization(&thermal, c).unwrap()).abs() <= 1e-15);
    // Parahydrogen on two uncarried target protons.
    let mut hydro = copy.clone();
    let p1 = nucleus(&mut hydro, Isotope::H1);
    let p2 = nucleus(&mut hydro, Isotope::H1);
    let fresh = Fresh::Parahydrogen {
        spins: [p1, p2],
        fraction: 1.0,
    };
    let on = solution
        .product_on(Product::Singlet, &hydro, &routes, fresh)
        .unwrap();
    // Reduced to the fresh pair (trace over the first two spins).
    let pair_part = CMatrix::from_fn(4, 4, |r, col| {
        (0..4).map(|u| on[(4 * u + r, 4 * u + col)]).sum()
    });
    let para = hydro.parahydrogen_state(p1, p2, 1.0).unwrap();
    let para_part = CMatrix::from_fn(4, 4, |r, col| {
        (0..4).map(|u| para[(4 * u + r, 4 * u + col)]).sum()
    });
    assert!(largest(&(&pair_part - &para_part)) <= 1e-15);
    // Refusals.
    let no_escape = pair.clone().with_rates(3e7, 8e6, 0.0).unwrap();
    let s = no_escape.solve(Birth::Singlet).unwrap();
    assert_invalid(
        s.product_on(Product::Escape, &copy, &routes, Fresh::Thermal),
        "product",
    );
    let mut carbon = SpinSystem::new(9.4).with_frame(Frame::Laboratory);
    nucleus(&mut carbon, Isotope::C13);
    nucleus(&mut carbon, Isotope::H1);
    assert_invalid(
        solution.product_on(Product::Singlet, &carbon, &[(2, 0)], Fresh::Thermal),
        "channel",
    );
    assert_invalid(
        solution.product_on(Product::Singlet, &copy, &[(0, 0)], Fresh::Thermal),
        "channel",
    );
    assert_invalid(
        solution.product_on(Product::Singlet, &copy, &[(2, 0), (2, 1)], Fresh::Thermal),
        "channel",
    );
    assert_invalid(
        solution.product_on(Product::Singlet, &copy, &[(2, 0), (3, 0)], Fresh::Thermal),
        "channel",
    );
    assert_invalid(
        solution.product_on(Product::Singlet, pair.system(), &[(2, 2)], Fresh::Thermal),
        "sites",
    );
}

#[test]
fn a_pair_without_nuclei_has_scalar_products() {
    let mut sys = system(0.3, Frame::Rotating);
    sys.add_electron(2.0031).unwrap();
    sys.add_electron(2.0027).unwrap();
    let pair = RadicalPair::new(sys)
        .unwrap()
        .with_rates(1e7, 2e7, 1e6)
        .unwrap();
    let solution = pair.solve(Birth::Singlet).unwrap();
    let s = solution.product(Product::Singlet);
    assert_eq!(s.shape(), (1, 1));
    assert!((s[(0, 0)].re - solution.yields().singlet).abs() <= 1e-15);
    assert!(solution.yields().cidnp.is_empty());
}

#[test]
fn products_handed_to_a_spin_system_show_kapteins_signs() {
    // The singlet product of the high-field pair onto ¹H, ¹H and ¹⁴N at
    // 9.4 T, then a 90° pulse on the protons: each proton's line has the
    // sign of Kaptein's rule against a thermal (absorptive) line.
    let pair = high_field();
    let solution = pair.solve(Birth::Singlet).unwrap();
    let mut target = SpinSystem::new(9.4);
    nucleus(&mut target, Isotope::H1);
    target.add_spin(Isotope::H1, 7.0).unwrap();
    nucleus(&mut target, Isotope::N14);
    let rho = solution
        .product_on(
            Product::Singlet,
            &target,
            &[(2, 0), (3, 1), (4, 2)],
            Fresh::Thermal,
        )
        .unwrap();
    let pulse = [Event::pulse(Target::Isotope(Isotope::H1), 90.0, 0.0)];
    let lines = |rho: &CMatrix| {
        target
            .lines(&target.evolve(rho, &pulse).unwrap(), Isotope::H1)
            .unwrap()
    };
    let product = lines(&rho);
    let thermal = lines(&target.thermal_state().unwrap());
    assert_eq!(product.len(), 2);
    assert_eq!(thermal.len(), 2);
    // Singlet precursor (μ = −1), singlet product (ε = +1), Δg > 0 for
    // radical 1: the sign is −sgn(a); a = 1.2 mT (0 ppm), −0.4 mT (7 ppm).
    for (ppm, sign) in [(0.0, -1.0), (7.0, 1.0)] {
        let find = |ls: &[crate::spin::Line]| {
            ls.iter()
                .find(|l| (l.ppm - ppm).abs() < 0.1)
                .unwrap()
                .amplitude
        };
        let relative_sign = (find(&product) * find(&thermal).conj()).re.signum();
        assert_eq!(relative_sign, sign, "{ppm}");
    }
}

#[test]
fn the_order_of_the_spins_does_not_matter() {
    // The same pair with its spins as e₁, e₂, ¹⁴N, ¹H and as ¹H, e₁, ¹⁴N,
    // e₂ (electrons between and after nuclei): the same yields, CIDNP in the
    // order of the nuclei, and products equal after reordering, each the
    // partial trace of `k_P P_P X` taken here by hand.
    let make = |order: [usize; 4]| {
        // order[k]: the spin at index k (0 e₁, 1 e₂, 2 ¹⁴N, 3 ¹H).
        let mut sys = system(2e-3, Frame::Laboratory);
        let mut at = [0; 4];
        for &o in &order {
            at[o] = match o {
                0 => sys.add_electron(FREE_ELECTRON_G + 1e-3).unwrap(),
                1 => sys.add_electron(FREE_ELECTRON_G).unwrap(),
                2 => nucleus(&mut sys, Isotope::N14),
                _ => nucleus(&mut sys, Isotope::H1),
            };
        }
        sys.hyperfine(at[0], at[2], 0.5).unwrap();
        sys.hyperfine(at[1], at[3], -0.9).unwrap();
        sys.exchange_coupling(at[0], at[1], 0.03).unwrap();
        RadicalPair::new(sys)
            .unwrap()
            .with_rates(2e7, 7e6, 1e6)
            .unwrap()
    };
    let (a, b) = (make([0, 1, 2, 3]), make([3, 0, 2, 1]));
    assert_eq!(b.electrons(), [1, 3]);
    assert_eq!(b.nuclei(), vec![0, 2]);
    for model in [Recombination::Haberkorn, Recombination::JonesMaedaHore] {
        let (a, b) = (
            a.clone().with_recombination(model),
            b.clone().with_recombination(model),
        );
        let (sa, sb) = (
            a.solve(Birth::Singlet).unwrap(),
            b.solve(Birth::Singlet).unwrap(),
        );
        let (ya, yb) = (sa.yields(), sb.yields());
        for (u, v) in [
            (ya.singlet, yb.singlet),
            (ya.triplet, yb.triplet),
            (ya.escape, yb.escape),
        ] {
            assert!((u - v).abs() <= 1e-13, "{model:?}: {u} vs {v}");
        }
        for p in 0..3 {
            assert!((ya.cidnp[0][p] - yb.cidnp[1][p]).abs() <= 1e-14);
            assert!((ya.cidnp[1][p] - yb.cidnp[0][p]).abs() <= 1e-14);
        }
        let [ks, kt, ke] = b.rates();
        for which in [Product::Singlet, Product::Triplet, Product::Escape] {
            // a: ¹⁴N ⊗ ¹H; b: ¹H ⊗ ¹⁴N.
            let (pa, pb) = (sa.product(which), sb.product(which));
            let swapped =
                CMatrix::from_fn(6, 6, |r, c| pb[((r % 2) * 3 + r / 2, (c % 2) * 3 + c / 2)]);
            assert!(largest(&(&pa - &swapped)) <= 1e-14, "{model:?} {which:?}");
            let weight = match which {
                Product::Singlet => b.projector(Birth::Singlet) * C64::new(ks, 0.0),
                Product::Triplet => b.projector(Birth::Triplet) * C64::new(kt, 0.0),
                Product::Escape => CMatrix::identity(24, 24) * C64::new(ke, 0.0),
            };
            let m = weight * sb.integral();
            // b's basis: ¹H (2) e₁ (2) ¹⁴N (3) e₂ (2).
            let at = |h: usize, e1: usize, n: usize, e2: usize| h * 12 + e1 * 6 + n * 2 + e2;
            let want = CMatrix::from_fn(6, 6, |r, c| {
                let mut z = C64::new(0.0, 0.0);
                for e1 in 0..2 {
                    for e2 in 0..2 {
                        z += m[(at(r / 3, e1, r % 3, e2), at(c / 3, e1, c % 3, e2))];
                    }
                }
                z
            });
            assert!(largest(&(&pb - &want)) <= 1e-14, "{model:?} {which:?}");
        }
    }
}

#[test]
fn the_evolution_with_relaxation_and_the_jones_maeda_hore_term_matches_liouville_space() {
    let mut sys = system(1e-3, Frame::Laboratory)
        .with_temperature(5.0)
        .unwrap();
    let e1 = sys.add_electron(FREE_ELECTRON_G + 1e-3).unwrap();
    let e2 = sys.add_electron(FREE_ELECTRON_G).unwrap();
    let h = nucleus(&mut sys, Isotope::H1);
    sys.hyperfine(e1, h, 0.8).unwrap();
    sys.set_relaxation(e1, Some(3e-7), Some(1e-7)).unwrap();
    sys.set_relaxation(e2, Some(5e-7), Some(4e-7)).unwrap();
    let pair = RadicalPair::new(sys)
        .unwrap()
        .with_rates(3e7, 8e6, 2e6)
        .unwrap()
        .with_recombination(Recombination::JonesMaedaHore);
    let rho0 = pair.birth_state(Birth::Singlet);
    let t = 4e-8;
    let got = pair.evolve(&rho0, t).unwrap();
    let v = crate::linalg::dense::expm(&(liouvillian(&pair) * C64::new(t, 0.0)))
        * nalgebra::DVector::from_column_slice(rho0.as_slice());
    let want = CMatrix::from_column_slice(8, 8, v.as_slice());
    let e = largest(&(&got - &want));
    assert!(e <= 1e-13, "{e:e}");
}
