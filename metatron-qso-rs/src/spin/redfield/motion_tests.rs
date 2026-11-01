//! Motion beyond isotropic tumbling against closed forms.

use super::motion::diffusion_operator;
use super::*;
use crate::spin::{Event, Frame, Isotope};

/// `L(ω, τ) = τ/(1 + ω²τ²)`.
fn lor(w: f64, tau: f64) -> f64 {
    tau / (1.0 + w * w * tau * tau)
}

fn unit(theta: f64, phi: f64) -> [f64; 3] {
    [
        theta.sin() * phi.cos(),
        theta.sin() * phi.sin(),
        theta.cos(),
    ]
}

fn diag(x: f64, y: f64, z: f64) -> [[f64; 3]; 3] {
    [[x, 0.0, 0.0], [0.0, y, 0.0], [0.0, 0.0, z]]
}

/// The normalised spectral density of Woessner for a vector with direction
/// cosines `(l, m, n)` in the frame of `D = diag(dx, dy, dz)`:
/// `Σ_i A_i τ_i/(1 + ω²τ_i²)` (Woessner 1962; the five-term form, e.g.
/// Cavanagh et al., Protein NMR Spectroscopy, ch. 8).
fn woessner(dir: [f64; 3], d: [f64; 3], tau_e: Option<(f64, f64)>) -> impl Fn(f64) -> f64 {
    let [l, m, n] = dir;
    let [dx, dy, dz] = d;
    let mean = (dx + dy + dz) / 3.0;
    let l2 = (dx * dy + dx * dz + dy * dz) / 3.0;
    let root = (mean * mean - l2).sqrt();
    let rates = [
        4.0 * dx + dy + dz,
        dx + 4.0 * dy + dz,
        dx + dy + 4.0 * dz,
        6.0 * mean + 6.0 * root,
        6.0 * mean - 6.0 * root,
    ];
    let mut a = [
        3.0 * m * m * n * n,
        3.0 * l * l * n * n,
        3.0 * l * l * m * m,
        0.0,
        0.0,
    ];
    let dd = (3.0 * (l.powi(4) + m.powi(4) + n.powi(4)) - 1.0) / 4.0;
    if root > 1e-12 * mean {
        let delta = |di: f64| (di - mean) / root;
        let e = (delta(dx) * (3.0 * l.powi(4) + 6.0 * m * m * n * n - 1.0)
            + delta(dy) * (3.0 * m.powi(4) + 6.0 * l * l * n * n - 1.0)
            + delta(dz) * (3.0 * n.powi(4) + 6.0 * l * l * m * m - 1.0))
            / 12.0;
        a[3] = dd - e;
        a[4] = dd + e;
    } else {
        a[3] = dd;
        a[4] = 0.0;
    }
    move |w: f64| {
        let (s2, te) = tau_e.unwrap_or((1.0, 1.0));
        (0..5)
            .map(|i| {
                let fast = rates[i] + 1.0 / te;
                a[i] * (s2 * lor(w, 1.0 / rates[i]) + (1.0 - s2) * lor(w, 1.0 / fast))
            })
            .sum()
    }
}

/// The rates of axially symmetric diffusion and the coefficients of two
/// axial tensors along `u` and `v`: `A₁ = P₂P₂`, `A₂ = 3 s c s' c'
/// cos Δφ`, `A₃ = (3/4) s² s'² cos 2Δφ` (θ from the symmetry axis z).
fn axial_cross(u: [f64; 3], v: [f64; 3], par: f64, perp: f64) -> impl Fn(f64) -> f64 {
    let polar = |x: [f64; 3]| (x[2].acos(), x[1].atan2(x[0]));
    let ((t1, p1), (t2, p2)) = (polar(u), polar(v));
    let a = [
        (3.0 * t1.cos().powi(2) - 1.0) * (3.0 * t2.cos().powi(2) - 1.0) / 4.0,
        3.0 * t1.sin() * t1.cos() * t2.sin() * t2.cos() * (p1 - p2).cos(),
        0.75 * t1.sin().powi(2) * t2.sin().powi(2) * (2.0 * (p1 - p2)).cos(),
    ];
    let rates = [6.0 * perp, 5.0 * perp + par, 2.0 * perp + 4.0 * par];
    move |w: f64| (0..3).map(|i| a[i] * lor(w, 1.0 / rates[i])).sum()
}

#[test]
fn the_diffusion_operator_has_woessner_rates() {
    // Isotropic: 6D on every traceless tensor.
    let m = diffusion_operator(&diag(2.0, 2.0, 2.0));
    for mu in 0..5 {
        for nu in 0..5 {
            let want = if mu == nu { 12.0 } else { 0.0 };
            assert!((m[(mu, nu)] - want).abs() < 1e-12, "{mu} {nu}");
        }
    }
    // Fully anisotropic, also in a rotated frame (same eigenvalues).
    let (dx, dy, dz): (f64, f64, f64) = (1.3e7, 2.1e7, 4.4e7);
    let mean = (dx + dy + dz) / 3.0;
    let l2 = (dx * dy + dx * dz + dy * dz) / 3.0;
    let root = (mean * mean - l2).sqrt();
    let mut want = vec![
        4.0 * dx + dy + dz,
        dx + 4.0 * dy + dz,
        dx + dy + 4.0 * dz,
        6.0 * mean + 6.0 * root,
        6.0 * mean - 6.0 * root,
    ];
    want.sort_by(f64::total_cmp);
    let (c, s) = (0.3f64.cos(), 0.3f64.sin());
    let r = [
        [c, -s, 0.0],
        [s * 0.6, c * 0.6, 0.8],
        [-s * 0.8, -c * 0.8, 0.6],
    ];
    let mut rotated = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            rotated[a][b] =
                r[a][0] * r[b][0] * dx + r[a][1] * r[b][1] * dy + r[a][2] * r[b][2] * dz;
        }
    }
    for d in [diag(dx, dy, dz), rotated] {
        let mut got: Vec<f64> = diffusion_operator(&d)
            .symmetric_eigen()
            .eigenvalues
            .iter()
            .copied()
            .collect();
        got.sort_by(f64::total_cmp);
        for (g, w) in got.iter().zip(&want) {
            assert!((g / w - 1.0).abs() < 1e-12, "{g} vs {w}");
        }
    }
}

/// ¹⁵N at the origin and ¹H at 1.02 Å along `u`, with an axial shielding
/// anisotropy of −160 ppm along `v`.
fn nh(b0: f64, u: [f64; 3], v: [f64; 3]) -> SpinSystem {
    let mut sys = SpinSystem::new(b0);
    let n = sys.add_spin(Isotope::N15, 120.0).unwrap();
    let h = sys.add_spin(Isotope::H1, 8.3).unwrap();
    sys.set_position(n, [0.0; 3]).unwrap();
    sys.set_position(h, u.map(|x| 1.02 * x)).unwrap();
    sys.set_csa(n, Some(axial_csa(-160.0, v).unwrap())).unwrap();
    sys
}

fn larmor(sys: &SpinSystem, i: usize) -> f64 {
    let s = &sys.spins()[i];
    -s.isotope.gamma() * sys.field() * (1.0 + s.shift * 1e-6)
}

/// `R₁`, `σ_NH` and `R₂` of ¹⁵N from the normalised spectral densities of
/// the dipolar coupling (`jd`) and of the shielding (`jc`).
fn nh_rates(sys: &SpinSystem, jd: &dyn Fn(f64) -> f64, jc: &dyn Fn(f64) -> f64) -> [f64; 3] {
    let b = sys.dipolar_coupling(0, 1).unwrap();
    let (wn, wh) = (larmor(sys, 0), larmor(sys, 1));
    let w0 = Isotope::N15.gamma() * sys.field();
    let c2 = w0 * w0 * (2.0 / 3.0) * 160e-6 * 160e-6;
    let k = b * b / 10.0;
    [
        k * (jd(wn - wh) + 3.0 * jd(wn) + 6.0 * jd(wn + wh)) + c2 / 5.0 * jc(wn),
        k * (6.0 * jd(wn + wh) - jd(wn - wh)),
        k / 2.0 * (4.0 * jd(0.0) + jd(wn - wh) + 3.0 * jd(wn) + 6.0 * jd(wh) + 6.0 * jd(wn + wh))
            + c2 / 30.0 * (4.0 * jc(0.0) + 3.0 * jc(wn)),
    ]
}

fn qso_rates(sys: &SpinSystem) -> [f64; 3] {
    let ops = [
        sys.iz(0).unwrap(),
        sys.iz(1).unwrap(),
        sys.iplus(0).unwrap(),
    ];
    let g = sys.relaxation_rates(&ops).unwrap();
    [g[(0, 0)], g[(0, 1)], g[(2, 2)]]
}

fn assert_rates(got: [f64; 3], want: [f64; 3], what: &str) {
    for (g, w) in got.iter().zip(&want) {
        assert!((g / w - 1.0).abs() < 1e-9, "{what}: {got:?} vs {want:?}");
    }
}

const DD: Interaction = Interaction::Dipolar(0, 1);
const CSA: Interaction = Interaction::Shielding(0);

#[test]
fn nh_rates_follow_the_closed_forms_of_every_motion() {
    let b0 = 18.8;
    let u = unit(0.7, 0.4);
    let v = unit(0.95, 0.1);
    let tau_c = 8e-9;
    // Isotropic, rigid and Lipari–Szabo (S² = 0.85, τ_e = 50 ps) on both
    // interactions together; the extended form (S_f² = 0.9, τ_f = 20 ps,
    // S_s² = 0.8, τ_s = 1.5 ns).
    let iso = |s2f: f64, tf: f64, s2s: f64, ts: f64| {
        move |w: f64| {
            let s2 = s2f * s2s;
            s2 * lor(w, tau_c)
                + (1.0 - s2f) * lor(w, 1.0 / (1.0 / tau_c + 1.0 / tf))
                + (s2f - s2) * lor(w, 1.0 / (1.0 / tau_c + 1.0 / ts))
        }
    };
    for (motion, label) in [
        (None, "rigid"),
        (
            Some(ModelFree::lipari_szabo(0.85, 5e-11).unwrap()),
            "Lipari–Szabo",
        ),
        (
            Some(ModelFree::extended(0.9, 2e-11, 0.8, 1.5e-9).unwrap()),
            "extended",
        ),
    ] {
        let mut sys = nh(b0, u, v).with_correlation_time(Some(tau_c)).unwrap();
        sys.set_internal_motion(&[DD, CSA], motion).unwrap();
        let (s2f, tf, s2s, ts) = motion.map_or((1.0, 1.0, 1.0, 1.0), |m| m.parameters());
        let j = iso(s2f, tf, s2s, ts);
        assert_rates(qso_rates(&sys), nh_rates(&sys, &j, &j), label);
        for w in [0.0, 5e8, 3e9] {
            for (k, l) in [(DD, DD), (CSA, CSA)] {
                let got = sys.spectral_density(k, l, w).unwrap();
                assert!((got / j(w) - 1.0).abs() < 1e-12, "{label} {w}");
            }
        }
    }
    // Axially symmetric diffusion (D_∥/D_⊥ = 2, about z), rigid: Woessner's
    // three terms for each interaction.
    let (par, perp) = (3.0e7, 1.5e7);
    let sys = nh(b0, u, v)
        .with_diffusion(Some(axial_diffusion(par, perp, [0.0, 0.0, 1.0]).unwrap()))
        .unwrap();
    let jd = axial_cross(u, u, par, perp);
    let jc = axial_cross(v, v, par, perp);
    assert_rates(qso_rates(&sys), nh_rates(&sys, &jd, &jc), "axial");
    for w in [0.0, 7e8] {
        let got = sys.spectral_density(DD, CSA, w).unwrap();
        let want = axial_cross(u, v, par, perp)(w);
        assert!(
            (got - want).abs() < 1e-12 * jd(0.0),
            "cross {w}: {got} vs {want}"
        );
    }
    // Fully anisotropic diffusion with Lipari–Szabo internal motion of the
    // bond alone (the shielding rigid): Woessner's five terms.
    let d = [1.3e7, 2.1e7, 4.4e7];
    let mut sys = nh(b0, u, v)
        .with_diffusion(Some(diag(d[0], d[1], d[2])))
        .unwrap();
    let ls = ModelFree::lipari_szabo(0.8, 8e-11).unwrap();
    sys.set_internal_motion(&[DD], Some(ls)).unwrap();
    let jd = woessner(u, d, Some((0.8, 8e-11)));
    let jc = woessner(v, d, None);
    assert_rates(qso_rates(&sys), nh_rates(&sys, &jd, &jc), "anisotropic");
}

/// The TROSY cross-correlated rate `η_xy = Γ(N_x, 2N_xH_z)` of Goldman (J.
/// Magn. Reson. 60, 437 (1984)): `η = −(1/15) b ω_N Δσ P₂(cos β) (4J(0) +
/// 3J(ω_N))` for isotropic tumbling, with `J` the normalised spectral
/// density and `β` the angle between the bond and the shielding axis.
#[test]
fn the_trosy_rate_follows_goldman() {
    let (b0, tau) = (18.8, 1e-8);
    let u = [0.0, 0.0, 1.0];
    let magic = (1.0f64 / 3.0).sqrt().acos();
    for beta in [0.0, 0.3, magic, 1.2, std::f64::consts::FRAC_PI_2] {
        let v = unit(beta, 0.0);
        let sys = nh(b0, u, v).with_correlation_time(Some(tau)).unwrap();
        let b = sys.dipolar_coupling(0, 1).unwrap();
        let w0 = Isotope::N15.gamma() * b0;
        let wn = larmor(&sys, 0);
        let p2 = (3.0 * beta.cos().powi(2) - 1.0) / 2.0;
        let eta =
            -(1.0 / 15.0) * b * w0 * (-160e-6) * p2 * (4.0 * lor(0.0, tau) + 3.0 * lor(wn, tau));
        let nx = sys.ix(0).unwrap();
        let anti = &nx * &sys.iz(1).unwrap() * C64::new(2.0, 0.0);
        let g = sys.relaxation_rates(&[nx.clone(), anti.clone()]).unwrap();
        let scale = (b * w0 * 160e-6 * tau).abs();
        assert!(
            (g[(0, 1)] - eta).abs() < 1e-9 * scale,
            "β = {beta}: {} vs {eta}",
            g[(0, 1)]
        );
        // The two lines of the ¹⁵N doublet relax at R₂ ± η: they differ by
        // 2η, which vanishes at the magic angle.
        let plus = sys.iplus(0).unwrap();
        let half = C64::new(0.5, 0.0);
        let hz = sys.iz(1).unwrap();
        let alpha = &plus * half + &plus * &hz;
        let beta_line = &plus * half - &plus * &hz;
        let r = sys.relaxation_rates(&[alpha, beta_line]).unwrap();
        assert!(
            ((r[(0, 0)] - r[(1, 1)]) - 2.0 * eta).abs() < 1e-9 * scale,
            "β = {beta}"
        );
        if beta == magic {
            assert!(eta.abs() < 1e-12 * scale);
        }
    }
    // In axially symmetric diffusion the cross-correlation follows the
    // cross spectral density of the two axes.
    let (par, perp) = (4e7, 1.6e7);
    let (u, v) = (unit(0.5, 0.2), unit(0.8, 1.1));
    let sys = nh(b0, u, v)
        .with_diffusion(Some(axial_diffusion(par, perp, [0.0, 0.0, 1.0]).unwrap()))
        .unwrap();
    let b = sys.dipolar_coupling(0, 1).unwrap();
    let w0 = Isotope::N15.gamma() * b0;
    let wn = larmor(&sys, 0);
    let j = axial_cross(u, v, par, perp);
    // |A_D||A_C|/30 · (4J(0) + 3J(ω_N)) with |A_D| = √6 b, |A_C| = √(2/3)|ω₀Δσ|,
    // and the sign of A_D : A_C (= −b ω₀ Δσ · 3·P₂-weighted in J).
    let norms = 6f64.sqrt() * b * (2.0f64 / 3.0).sqrt() * (w0 * 160e-6).abs();
    // A_D = b(1 − 3uuᵀ) and A_C ∝ Δσ(3vvᵀ − 1): the normalised tensors are
    // −(3uuᵀ − 1)/√6 and sign(ω₀Δσ)(3vvᵀ − 1)/√6.
    let sign = -(w0 * -160e-6).signum();
    let eta = sign * norms / 30.0 * (4.0 * j(0.0) + 3.0 * j(wn));
    let nx = sys.ix(0).unwrap();
    let anti = &nx * &sys.iz(1).unwrap() * C64::new(2.0, 0.0);
    let g = sys.relaxation_rates(&[nx, anti]).unwrap();
    assert!(
        (g[(0, 1)] / eta - 1.0).abs() < 1e-9,
        "{} vs {eta}",
        g[(0, 1)]
    );
}

#[test]
fn isotropic_diffusion_and_rigid_motion_are_isotropic_tumbling() {
    // D = d·1 is τ_c = 1/(6d), and S² = 1 is no internal motion: every
    // rate of a ¹H–¹³C–¹H system with shielding equal to 1.34's.
    let mut sys = SpinSystem::new(14.1);
    let shifts = [(Isotope::H1, 2.0), (Isotope::C13, 30.0), (Isotope::H1, 2.5)];
    let places = [[1.09, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.36, 1.03, 0.2]];
    for (k, ((x, shift), p)) in shifts.into_iter().zip(places).enumerate() {
        sys.add_spin(x, shift).unwrap();
        sys.set_position(k, p).unwrap();
    }
    sys.set_csa(1, Some(axial_csa(40.0, [0.3, 0.1, 1.0]).unwrap()))
        .unwrap();
    let tau = 2e-10;
    let reference = sys.clone().with_correlation_time(Some(tau)).unwrap();
    let d = 1.0 / (6.0 * tau);
    let diffusing = sys.clone().with_diffusion(Some(diag(d, d, d))).unwrap();
    let mut rigid = reference.clone();
    rigid
        .set_internal_motion(
            &[Interaction::Dipolar(1, 0), Interaction::Shielding(1)],
            Some(ModelFree::lipari_szabo(1.0, 3e-11).unwrap()),
        )
        .unwrap();
    let ops: Vec<CMatrix> = (0..3)
        .flat_map(|k| [reference.iz(k).unwrap(), reference.iplus(k).unwrap()])
        .chain([&reference.iz(0).unwrap() * &reference.iz(1).unwrap()])
        .collect();
    let want = reference.relaxation_rates(&ops).unwrap();
    let largest = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    for (sys, what) in [(&diffusing, "diffusion"), (&rigid, "S² = 1")] {
        let got = sys.relaxation_rates(&ops).unwrap();
        let worst = (&got - &want).iter().fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(worst < 1e-12 * largest, "{what}: {worst:e}");
    }
    assert_eq!(diffusing.correlation_time(), None);
    assert_eq!(
        diffusing
            .clone()
            .with_correlation_time(Some(tau))
            .unwrap()
            .diffusion(),
        None
    );
}

#[test]
fn spectral_densities_are_the_transforms_of_the_correlation_functions() {
    // J_kl(ω) = ∫₀^∞ C_kl(τ) cos ωτ dτ, checked by quadrature, and
    // ∫₀^∞ J dω = (π/2) C(0); C_kk(0) = 1.
    let (u, v) = (unit(0.7, 0.4), unit(1.3, 2.0));
    let mut sys = nh(14.1, u, v)
        .with_diffusion(Some(diag(1.0e7, 1.7e7, 3.1e7)))
        .unwrap();
    sys.set_internal_motion(
        &[DD],
        Some(ModelFree::extended(0.9, 3e-11, 0.85, 1e-9).unwrap()),
    )
    .unwrap();
    // Five-point Gauss–Legendre on steps of 2 ps up to 0.2 ns (the fast
    // internal motion, 1/τ_f = 3·10¹⁰ s⁻¹) and of 50 ps up to 0.6 µs (C
    // has decayed by e⁻³⁸ there).
    let nodes = [
        (-0.906_179_845_938_664, 0.236_926_885_056_189_1),
        (-0.538_469_310_105_683_1, 0.478_628_670_499_366_5),
        (0.0, 0.568_888_888_888_888_9),
        (0.538_469_310_105_683_1, 0.478_628_670_499_366_5),
        (0.906_179_845_938_664, 0.236_926_885_056_189_1),
    ];
    let integrate = |f: &dyn Fn(f64) -> f64| {
        let mut total = 0.0;
        for (start, end, h) in [(0.0f64, 2e-10f64, 2e-12f64), (2e-10, 6e-7, 5e-11)] {
            let n = ((end - start) / h).round() as usize;
            for i in 0..n {
                let mid = start + (i as f64 + 0.5) * h;
                total += nodes
                    .iter()
                    .map(|&(x, w)| w * f(mid + x * h / 2.0))
                    .sum::<f64>()
                    * h
                    / 2.0;
            }
        }
        total
    };
    for (k, l) in [(DD, DD), (DD, CSA), (CSA, CSA)] {
        let c0 = sys.correlation_function(k, l, 0.0).unwrap();
        if k == l {
            assert!((c0 - 1.0).abs() < 1e-14);
        }
        for w in [0.0, 1.5e9] {
            let f = |t: f64| sys.correlation_function(k, l, t).unwrap() * (w * t).cos();
            let integral = integrate(&f);
            let j = sys.spectral_density(k, l, w).unwrap();
            let scale = sys.spectral_density(k, k, 0.0).unwrap();
            assert!(
                (integral - j).abs() < 1e-9 * scale,
                "{k:?} {l:?} {w}: {integral} vs {j}"
            );
        }
    }
    // The internal correlation function of the extended model.
    let m = ModelFree::extended(0.9, 3e-11, 0.85, 1e-9).unwrap();
    assert!((m.correlation(0.0) - 1.0).abs() < 1e-15);
    assert!((m.correlation(1.0) - m.order_parameter()).abs() < 1e-15);
}

#[test]
fn anisotropic_motion_keeps_the_thermal_state() {
    // ¹H–¹³C–¹H at 4 K with anisotropic diffusion and internal motion of
    // one C–H bond: every state relaxes to the thermal one.
    let mut sys = SpinSystem::new(9.4).with_temperature(4.0).unwrap();
    let shifts = [(Isotope::H1, 2.0), (Isotope::C13, 30.0), (Isotope::H1, 2.5)];
    let places = [[1.09, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.36, 1.03, 0.0]];
    for (k, ((x, shift), p)) in shifts.into_iter().zip(places).enumerate() {
        sys.add_spin(x, shift).unwrap();
        sys.set_position(k, p).unwrap();
    }
    sys.set_csa(1, Some(axial_csa(40.0, [0.0, 0.0, 1.0]).unwrap()))
        .unwrap();
    sys.couple(0, 1, 140.0).unwrap();
    sys.couple(1, 2, 135.0).unwrap();
    let mut sys = sys
        .with_diffusion(Some(axial_diffusion(2e9, 8e8, [1.0, 1.0, 0.0]).unwrap()))
        .unwrap();
    sys.set_internal_motion(
        &[Interaction::Dipolar(0, 1)],
        Some(ModelFree::lipari_szabo(0.7, 2e-11).unwrap()),
    )
    .unwrap();
    let eq = sys.thermal_state().unwrap();
    let slowest = sys.relaxation_rates(&[sys.iz(1).unwrap()]).unwrap()[(0, 0)];
    let late = sys
        .evolve(&sys.identity_state(), &[Event::Delay(60.0 / slowest)])
        .unwrap();
    let scale = (&eq - sys.identity_state()).norm();
    assert!((&late - &eq).norm() < 1e-8 * scale);
}

#[test]
fn symmetries_must_keep_the_motion() {
    // Two equivalent protons on a ¹³C: the exchange of the protons is a
    // symmetry under isotropic tumbling and under diffusion about the
    // bisector, not under diffusion about one bond.
    let build = |axis: [f64; 3]| {
        let mut sys = SpinSystem::new(9.4);
        sys.add_spin(Isotope::C13, 30.0).unwrap();
        sys.add_spin(Isotope::H1, 2.0).unwrap();
        sys.add_spin(Isotope::H1, 2.0).unwrap();
        sys.set_position(0, [0.0; 3]).unwrap();
        sys.set_position(1, [0.9, 0.7, 0.0]).unwrap();
        sys.set_position(2, [-0.9, 0.7, 0.0]).unwrap();
        sys.couple(0, 1, 125.0).unwrap();
        sys.couple(0, 2, 125.0).unwrap();
        sys.with_diffusion(Some(axial_diffusion(3e9, 1e9, axis).unwrap()))
            .unwrap()
    };
    let swap = [0, 2, 1];
    assert!(build([0.0, 1.0, 0.0]).redfield_invariant(&swap));
    assert!(!build([0.9, 0.7, 0.0]).redfield_invariant(&swap));
    // Internal motion of one bond only breaks it too, of both keeps it.
    let mut one = build([0.0, 1.0, 0.0]);
    let ls = ModelFree::lipari_szabo(0.8, 2e-11).unwrap();
    one.set_internal_motion(&[Interaction::Dipolar(0, 1)], Some(ls))
        .unwrap();
    assert!(!one.redfield_invariant(&swap));
    let mut both = one.clone();
    both.set_internal_motion(&[Interaction::Dipolar(0, 2)], Some(ls))
        .unwrap();
    assert!(both.redfield_invariant(&swap));
}

#[test]
fn motion_input_is_checked_and_survives_json() {
    let mut sys = nh(14.1, unit(0.7, 0.4), unit(0.9, 0.1));
    assert!(sys.spectral_density(DD, DD, 0.0).is_err());
    for bad in [
        diag(1e7, 1e7, -1e7),
        diag(1e7, f64::NAN, 1e7),
        [[1e7, 1e6, 0.0], [0.0, 1e7, 0.0], [0.0, 0.0, 1e7]],
    ] {
        assert!(sys.clone().with_diffusion(Some(bad)).is_err());
    }
    assert!(axial_diffusion(0.0, 1e7, [0.0, 0.0, 1.0]).is_err());
    assert!(axial_diffusion(1e7, 1e7, [0.0; 3]).is_err());
    for (s2, tau) in [
        (1.2, 1e-11),
        (-0.1, 1e-11),
        (0.5, 0.0),
        (0.5, f64::INFINITY),
    ] {
        assert!(ModelFree::lipari_szabo(s2, tau).is_err());
    }
    let ls = ModelFree::lipari_szabo(0.8, 5e-11).unwrap();
    assert!(
        sys.set_internal_motion(&[Interaction::Dipolar(0, 5)], Some(ls))
            .is_err()
    );
    assert!(
        sys.set_internal_motion(&[Interaction::Dipolar(1, 1)], Some(ls))
            .is_err()
    );
    sys.set_internal_motion(&[Interaction::Dipolar(1, 0), CSA], Some(ls))
        .unwrap();
    assert_eq!(sys.internal_motion(DD), Some(ls));
    // An interaction moves to the latest group; None removes it.
    let other = ModelFree::extended(0.9, 2e-11, 0.9, 1e-9).unwrap();
    sys.set_internal_motion(&[CSA], Some(other)).unwrap();
    assert_eq!(sys.internal_motion(CSA), Some(other));
    sys.set_internal_motion(&[DD], None).unwrap();
    assert_eq!(sys.internal_motion(DD), None);
    let sys = sys
        .with_diffusion(Some(axial_diffusion(3e7, 1.5e7, [0.0, 0.0, 1.0]).unwrap()))
        .unwrap();
    assert!(
        sys.spectral_density(Interaction::Shielding(1), CSA, 0.0)
            .is_err()
    );
    let back = SpinSystem::from_json(&sys.to_json()).unwrap();
    assert_eq!(back, sys);
    // A correlation time and a diffusion tensor together are refused.
    let text = sys.to_json().replacen(
        "\"diffusion\"",
        "\"correlation_time\": 1e-9, \"diffusion\"",
        1,
    );
    assert!(SpinSystem::from_json(&text).is_err());
    // Motion too slow for Redfield theory is refused.
    let slow = nh(14.1, unit(0.7, 0.4), unit(0.9, 0.1))
        .with_diffusion(Some(diag(1e3, 1e3, 2e3)))
        .unwrap();
    assert!(slow.relaxation_rates(&[slow.iz(0).unwrap()]).is_err());
}

#[test]
fn diffusion_runs_through_the_frames() {
    // The laboratory frame gives the same ¹⁵N rates as the rotating one.
    let make = |frame: Frame| {
        let mut sys = nh(14.1, unit(0.7, 0.4), unit(0.9, 0.1))
            .with_frame(frame)
            .with_diffusion(Some(diag(1.0e7, 1.7e7, 3.1e7)))
            .unwrap();
        sys.set_internal_motion(&[DD], Some(ModelFree::lipari_szabo(0.85, 4e-11).unwrap()))
            .unwrap();
        sys
    };
    let a = qso_rates(&make(Frame::Rotating));
    let b = qso_rates(&make(Frame::Laboratory));
    assert_rates(a, b, "frames");
}

/// Rates for the documentation (`cargo test --release -- --ignored
/// measure_motion --nocapture`).
#[test]
#[ignore]
fn measure_motion() {
    // ¹⁵N–¹H of a small protein (τ_c = 4.1 ns; D_∥/D_⊥ = 1.2 at the same
    // mean rate; S² = 0.85, τ_e = 50 ps), the bond at 40° from the long
    // axis and the shielding 17° from the bond.
    let u = unit(40f64.to_radians(), 0.0);
    let v = unit(57f64.to_radians(), 0.0);
    let tau_c = 4.1e-9;
    let mean = 1.0 / (6.0 * tau_c);
    let (perp, par) = (3.0 * mean / 3.2, 1.2 * 3.0 * mean / 3.2);
    for b0 in [14.1, 18.8] {
        for (label, diffusion, ls) in [
            ("isotropic, rigid", false, false),
            ("isotropic, Lipari–Szabo", false, true),
            ("axial, rigid", true, false),
            ("axial, Lipari–Szabo", true, true),
        ] {
            let base = nh(b0, u, v);
            let mut sys = if diffusion {
                base.with_diffusion(Some(axial_diffusion(par, perp, [0.0, 0.0, 1.0]).unwrap()))
                    .unwrap()
            } else {
                base.with_correlation_time(Some(tau_c)).unwrap()
            };
            if ls {
                sys.set_internal_motion(
                    &[DD, CSA],
                    Some(ModelFree::lipari_szabo(0.85, 5e-11).unwrap()),
                )
                .unwrap();
            }
            let [r1, sigma, r2] = qso_rates(&sys);
            let noe = 1.0 + sigma / r1 * Isotope::H1.gamma() / Isotope::N15.gamma();
            let nx = sys.ix(0).unwrap();
            let anti = &nx * &sys.iz(1).unwrap() * C64::new(2.0, 0.0);
            let eta = sys.relaxation_rates(&[nx, anti]).unwrap()[(0, 1)];
            eprintln!(
                "{b0} T {label}: R1 {r1:.3} s⁻¹, R2 {r2:.3} s⁻¹, NOE {noe:.3}, η_xy {eta:.3} s⁻¹, \
                 TROSY line {:.3} s⁻¹",
                r2 - eta.abs()
            );
        }
    }
    // Cost: six protons on a ring with anisotropic diffusion and three
    // groups of internal motion against isotropic tumbling.
    let mut ring = SpinSystem::new(14.1);
    for k in 0..6 {
        let a = k as f64 * std::f64::consts::PI / 3.0;
        ring.add_spin(Isotope::H1, 1.0 + 0.3 * k as f64).unwrap();
        ring.set_position(k, [2.5 * a.cos(), 2.5 * a.sin(), 0.3 * (k % 2) as f64])
            .unwrap();
    }
    let ops: Vec<CMatrix> = (0..6).map(|k| ring.iz(k).unwrap()).collect();
    let iso = ring.clone().with_correlation_time(Some(2e-10)).unwrap();
    let clock = std::time::Instant::now();
    iso.relaxation_rates(&ops).unwrap();
    let t_iso = clock.elapsed().as_secs_f64();
    let mut aniso = ring.with_diffusion(Some(diag(5e8, 8e8, 1.4e9))).unwrap();
    for g in 0..3 {
        aniso
            .set_internal_motion(
                &[Interaction::Dipolar(2 * g, 2 * g + 1)],
                Some(ModelFree::extended(0.9, 2e-11, 0.8, 5e-10).unwrap()),
            )
            .unwrap();
    }
    let interactions = aniso.interactions(1.0);
    let modes = aniso.modes(&interactions).len();
    let clock = std::time::Instant::now();
    aniso.relaxation_rates(&ops).unwrap();
    let t_aniso = clock.elapsed().as_secs_f64();
    eprintln!(
        "six protons, 15 dipolar couplings: isotropic {t_iso:.2} s (5 modes), anisotropic with \
         three internal motions {t_aniso:.2} s ({modes} modes)"
    );
}

#[test]
fn a_rotated_frame_gives_the_same_rates() {
    // Positions, shielding axis and diffusion tensor rotated together: the
    // diffusion tensor is no longer diagonal, the rates are unchanged.
    let (c, s) = (0.3f64.cos(), 0.3f64.sin());
    let r = [
        [c, -s, 0.0],
        [s * 0.6, c * 0.6, 0.8],
        [-s * 0.8, -c * 0.8, 0.6],
    ];
    let turn =
        |x: [f64; 3]| -> [f64; 3] { [0, 1, 2].map(|a| (0..3).map(|b| r[a][b] * x[b]).sum()) };
    let d = [1.3e7, 2.1e7, 4.4e7];
    let mut rotated_d = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            rotated_d[a][b] = (0..3).map(|k| r[a][k] * r[b][k] * d[k]).sum();
        }
    }
    let (u, v) = (unit(0.7, 0.4), unit(1.3, 2.0));
    let ls = ModelFree::lipari_szabo(0.8, 6e-11).unwrap();
    let mut plain = nh(18.8, u, v)
        .with_diffusion(Some(diag(d[0], d[1], d[2])))
        .unwrap();
    plain.set_internal_motion(&[DD], Some(ls)).unwrap();
    let mut turned = nh(18.8, turn(u), turn(v))
        .with_diffusion(Some(rotated_d))
        .unwrap();
    turned.set_internal_motion(&[DD], Some(ls)).unwrap();
    let ops = |sys: &SpinSystem| {
        let nx = sys.ix(0).unwrap();
        let anti = &nx * &sys.iz(1).unwrap() * C64::new(2.0, 0.0);
        vec![sys.iz(0).unwrap(), sys.iz(1).unwrap(), nx, anti]
    };
    let a = plain.relaxation_rates(&ops(&plain)).unwrap();
    let b = turned.relaxation_rates(&ops(&turned)).unwrap();
    let largest = a.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let worst = (&a - &b).iter().fold(0.0f64, |m, x| m.max(x.abs()));
    assert!(worst < 1e-10 * largest, "{worst:e}");
}
