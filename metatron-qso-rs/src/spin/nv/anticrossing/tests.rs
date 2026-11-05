//! The NV centre with its nuclei near the anticrossings: decoupled nuclei
//! against 1.40, the levels against closed forms, the microwave in the
//! laboratory frame against 1.40's rotating-wave model and against a
//! direct propagation over the period, the nuclear polarisation at the
//! excited-state anticrossing and the fluorescence at the ground-state
//! one.

use super::*;
use crate::linalg::dense::expm;
use crate::spin::nv::optics::Photophysics;

const KR: f64 = 65.9e6;

/// ¹⁵N with no hyperfine coupling: the nucleus only relaxes.
fn decoupled(field: [f64; 3]) -> NuclearOpticalNv {
    let mut n = NuclearSpin::nitrogen_15();
    n.ground = [[0.0; 3]; 3];
    n.excited = [[0.0; 3]; 3];
    let mut nv = NuclearOpticalNv::new(field, n);
    nv.nuclear_relaxation = 1e3;
    nv
}

#[test]
fn decoupled_nuclei_leave_the_optics_of_1_40() {
    for field in [
        [0.0, 0.0, 0.005],
        [0.003, 0.001, 0.02],
        [0.0, 0.0, 0.051],
        [0.0, 0.0, 0.1024],
    ] {
        let nv = decoupled(field);
        for pump in [0.02 * KR, KR] {
            let ours = nv.steady_state(pump, None).unwrap();
            let theirs = nv.electron.steady_state(pump, None).unwrap();
            let want = nv.electron.fluorescence(&theirs);
            assert!(
                (ours.fluorescence - want).abs() < 1e-10 * want,
                "{field:?}: {} vs {want}",
                ours.fluorescence
            );
            assert!(ours.polarisation[0].abs() < 1e-12);
        }
    }
}

#[test]
fn levels_follow_the_two_level_blocks_of_nitrogen_15() {
    // Along the axis S·A·I with A = diag(A⊥, A⊥, A∥) keeps m_s + m_I: the
    // singles |+1, +½⟩, |−1, −½⟩ and the pairs {|0, +½⟩, |+1, −½⟩},
    // {|−1, +½⟩, |0, −½⟩} coupled by A⊥/√2.
    for b in [0.03, 0.051, 0.1024] {
        let nv = NuclearOpticalNv::new([0.0, 0.0, b], NuclearSpin::nitrogen_15());
        let n = &nv.nuclei[0];
        for (d, a, levels) in [
            (
                nv.electron.zero_field_splitting,
                n.ground,
                nv.ground_levels().unwrap().0,
            ),
            (
                nv.electron.excited_splitting,
                n.excited,
                nv.excited_levels().unwrap().0,
            ),
        ] {
            let (par, perp) = (a[2][2], a[0][0]);
            let e = |ms: f64, mi: f64| {
                d * ms * ms + nv.electron.gyromagnetic_ratio * b * ms
                    - n.gyromagnetic_ratio * b * mi
                    + par * ms * mi
            };
            let pair = |x: f64, y: f64| {
                let (m, h) = ((x + y) / 2.0, (x - y) / 2.0);
                let r = (h * h + perp * perp / 2.0).sqrt();
                [m - r, m + r]
            };
            let mut want = vec![e(1.0, 0.5), e(-1.0, -0.5)];
            want.extend(pair(e(0.0, 0.5), e(1.0, -0.5)));
            want.extend(pair(e(-1.0, 0.5), e(0.0, -0.5)));
            want.sort_by(f64::total_cmp);
            for (got, want) in levels.iter().zip(&want) {
                assert!((got - want).abs() < 1e-6, "{b}: {got} vs {want}");
            }
        }
    }
}

#[test]
fn the_microwave_in_the_laboratory_equals_the_rotating_wave_model_where_it_holds() {
    // 5 mT along the axis, the nucleus decoupled, Ω = 0.5 MHz: the
    // counter-rotating terms shift the line by Ω²/4f ~ 20 Hz.
    let nv = decoupled([0.0, 0.0, 0.005]);
    let [(minus, _), _] = nv.electron.lines().unwrap();
    let f0 = minus.abs();
    let freqs: Vec<f64> = [-2e6, -0.4e6, 0.0, 0.4e6, 2e6]
        .iter()
        .map(|d| f0 + d)
        .collect();
    let pump = 0.05 * KR;
    let ours = nv.cw_odmr(&freqs, 0.5e6, pump).unwrap();
    let theirs = nv.electron.cw_odmr(&freqs, 0.5e6, pump).unwrap();
    for (a, b) in ours.iter().zip(&theirs) {
        assert!((a - b).abs() < 1e-5 * b, "{a} vs {b}");
    }
    // The dip is there.
    assert!(ours[2] < 0.97 * ours[0], "{ours:?}");
}

/// The period average of the fluorescence of `nv` (nucleus decoupled)
/// under `√2·2πΩ cos(2πft) S_x` by propagating the seven electron levels
/// over the period in `steps` exponential midpoint steps from their
/// periodic state, the fixed point of the period's map.
fn propagated(nv: &NuclearOpticalNv, pump: f64, f: f64, rabi: f64, steps: usize) -> f64 {
    let (model, l) = nv.electron.model(pump, None).unwrap();
    let l0 = model.superoperator();
    let [_, _, _, sx, _] = spin_ops(3);
    let mut h1 = CMatrix::zeros(7, 7);
    let s = l.vg.adjoint() * sx * &l.vg;
    for i in 0..3 {
        for j in 0..3 {
            h1[(i, j)] = s[(i, j)] * (TWO_PI * rabi * std::f64::consts::SQRT_2);
        }
    }
    let l1 = Lindblad {
        n: 7,
        h: h1,
        ops: Vec::new(),
    }
    .superoperator();
    let dt = 1.0 / f / steps as f64;
    let maps: Vec<CMatrix> = (0..steps)
        .map(|k| {
            let t = (k as f64 + 0.5) * dt;
            let c = (TWO_PI * f * t).cos();
            expm(&((&l0 + &l1 * C64::new(c, 0.0)) * C64::new(dt, 0.0)))
        })
        .collect();
    let mut period = CMatrix::identity(49, 49);
    for m in &maps {
        period = m * period;
    }
    // (P − 1) x = 0 with unit trace.
    let mut a = period - CMatrix::identity(49, 49);
    for c in 0..49 {
        a[(0, c)] = C64::new(0.0, 0.0);
    }
    for k in 0..7 {
        a[(0, k + 7 * k)] = C64::new(1.0, 0.0);
    }
    let mut rhs = nalgebra::DVector::from_element(49, C64::new(0.0, 0.0));
    rhs[0] = C64::new(1.0, 0.0);
    let mut x = a.full_piv_lu().solve(&rhs).unwrap();
    let excited = |x: &nalgebra::DVector<C64>| (3..6).map(|j| x[j + 7 * j].re).sum::<f64>();
    let mut total = 0.0;
    for m in &maps {
        let next = m * &x;
        total += 0.5 * (excited(&x) + excited(&next));
        x = next;
    }
    nv.electron.collection * nv.electron.rates.radiative * total / steps as f64
}

#[test]
fn the_microwave_in_the_laboratory_equals_a_propagation_over_its_period() {
    // 100 mT along the axis: the transition 0 ↔ −1 at 67 MHz, driven at
    // Ω = 15 MHz, where the rotating-wave approximation fails.
    let nv = decoupled([0.0, 0.0, 0.1]);
    let [(minus, _), _] = nv.electron.lines().unwrap();
    let pump = 0.1 * KR;
    let rabi = 15e6;
    for f in [minus.abs(), minus.abs() + 8e6] {
        let floquet = nv.steady_state(pump, Some((f, rabi))).unwrap();
        let (a, b) = (
            propagated(&nv, pump, f, rabi, 256),
            propagated(&nv, pump, f, rabi, 512),
        );
        let want = (4.0 * b - a) / 3.0;
        assert!(
            (a - b).abs() < 1e-4 * b,
            "the propagation has not converged: {a} {b}"
        );
        assert!(
            (floquet.fluorescence - want).abs() < 1e-7 * want,
            "{f}: Floquet {} against {want} ({} harmonics)",
            floquet.fluorescence,
            floquet.harmonics
        );
        let rwa = nv.electron.cw_odmr(&[f], rabi, pump).unwrap()[0];
        assert!(
            (rwa - want).abs() > 1e-3 * want,
            "the rotating-wave model holds here after all: {rwa} vs {want}"
        );
    }
}

#[test]
fn anticrossing_input_is_checked_and_survives_json() {
    let mut nv = NuclearOpticalNv::new([0.0, 0.0, 0.05], NuclearSpin::nitrogen_14());
    assert_eq!(nv.dimension(), 21);
    assert!(nv.steady_state(-1.0, None).is_err());
    assert!(nv.steady_state(KR, Some((-1.0, 1e6))).is_err());
    assert!(nv.steady_state(KR, Some((1e9, f64::NAN))).is_err());
    nv.nuclear_relaxation = -1.0;
    assert!(nv.validate().is_err());
    nv.nuclear_relaxation = 0.0;
    let back: NuclearOpticalNv =
        serde_json::from_str(&serde_json::to_string(&nv).unwrap()).unwrap();
    assert_eq!(back, nv);
    let mut bad = nv.clone();
    bad.nuclei[0].levels = 4;
    assert!(bad.validate().is_err());
    let mut three = nv.clone();
    three
        .nuclei
        .push(NuclearSpin::carbon_13([[0.0; 3]; 3], [[0.0; 3]; 3]));
    three
        .nuclei
        .push(NuclearSpin::carbon_13([[0.0; 3]; 3], [[0.0; 3]; 3]));
    assert!(three.validate().is_err());
    assert!(nv.nuclear_polarisation(&CMatrix::zeros(21, 21), 1).is_err());
    // A zero Rabi frequency is no microwave.
    let off = nv.steady_state(0.1 * KR, None).unwrap();
    let zero = nv.steady_state(0.1 * KR, Some((2.8e9, 0.0))).unwrap();
    assert!((&off.rho - &zero.rho).norm() < 1e-14);
}

#[test]
#[ignore]
fn explore() {
    // Duarte et al.'s ¹⁵N (their A⊥ is twice the coefficient of
    // S_xI_x + S_yI_y) with Gupta's rates and T₁ of the nucleus 100 ms.
    let mut n = NuclearSpin::nitrogen_15();
    n.ground = axial(3.9e6, 3.4e6);
    n.excited = axial(-38.5e6, -58.1e6);
    for (name, rates) in [
        ("gupta", Photophysics::gupta()),
        ("tetienne", Photophysics::tetienne()),
    ] {
        for pump in [4e6, 70e6] {
            let mut line = String::new();
            for b in [0.02, 0.035, 0.044, 0.048, 0.050, 0.051, 0.052, 0.055, 0.07] {
                let mut nv = NuclearOpticalNv::new([0.0, 0.0, b], n.clone());
                nv.electron.rates = rates;
                nv.nuclear_relaxation = 10.0;
                let s = nv.steady_state(pump, None).unwrap();
                line += &format!(
                    " {:.0} mT {:.3}/{:.3};",
                    b * 1e3,
                    s.polarisation[0],
                    s.polarisation_zero[0]
                );
            }
            eprintln!("{name} k = {pump:.0e}:{line}");
        }
    }
}
