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
            assert!(
                ours.polarisation[0].abs() < 1e-8,
                "{}",
                ours.polarisation[0]
            );
        }
    }
}

#[test]
fn a_steady_state_that_is_not_unique_is_refused() {
    // Without hyperfine couplings and nuclear relaxation the nucleus keeps
    // whatever population it had: every mixture is stationary.
    for field in [[0.0, 0.0, 0.03], [0.002, 0.0, 0.03]] {
        let mut nv = decoupled(field);
        nv.nuclear_relaxation = 0.0;
        assert!(nv.steady_state(0.1 * KR, None).is_err(), "{field:?}");
        let line = nv.electron.lines().unwrap()[0].0;
        assert!(
            nv.steady_state(0.1 * KR, Some((line.abs(), 1e6))).is_err(),
            "{field:?}"
        );
        // Couplings along the axis only conserve I_z in a field along it;
        // across it the nucleus is quantised differently in each level.
        let mut axial = nv.clone();
        axial.nuclei[0].ground = [[0.0; 3], [0.0; 3], [0.0, 0.0, 3e6]];
        axial.nuclei[0].excited = [[0.0; 3], [0.0; 3], [0.0, 0.0, 60e6]];
        assert_eq!(
            axial.steady_state(0.1 * KR, None).is_err(),
            field[0] == 0.0,
            "{field:?}"
        );
        // The slowest legitimate rate: 1 per second against gigahertz.
        nv.nuclear_relaxation = 1.0;
        let p = nv.steady_state(0.1 * KR, None).unwrap().polarisation[0];
        assert!(p.abs() < 1e-6, "{field:?}: {p}");
    }
}

#[test]
fn the_electron_with_its_nuclei_stays_near_1_40_away_from_the_anticrossings() {
    // The hyperfine couplings mix the excited levels by about (A_e/Δ)²:
    // 0.2 to 0.7 % of the fluorescence between 5 and 70 mT.
    for b in [0.005, 0.02, 0.07] {
        for nitrogen in [NuclearSpin::nitrogen_15(), NuclearSpin::nitrogen_14()] {
            let mut nv = NuclearOpticalNv::new([0.0, 0.0, b], nitrogen);
            nv.nuclear_relaxation = 10.0;
            let got = nv.steady_state(0.1 * KR, None).unwrap().fluorescence;
            let want = nv
                .electron
                .fluorescence(&nv.electron.steady_state(0.1 * KR, None).unwrap());
            assert!((got - want).abs() < 1e-2 * want, "{b}: {got} vs {want}");
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
    // 101.7 mT along the axis: the transition 0 ↔ −1 at 20 MHz, driven at
    // Ω = 15 MHz, where the rotating-wave approximation fails (the
    // counter-rotating field shifts the line by about Ω²/4f = 3 MHz).
    let nv = decoupled([0.0, 0.0, 0.1017]);
    let [(minus, _), _] = nv.electron.lines().unwrap();
    let pump = 0.1 * KR;
    let rabi = 15e6;
    for f in [minus.abs(), minus.abs() + 5e6] {
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
    for wrong in [CMatrix::zeros(14, 14), CMatrix::zeros(21, 14)] {
        assert!(nv.fluorescence(&wrong).is_err());
        assert!(nv.nuclear_polarisation(&wrong, 0).is_err());
        assert!(nv.nuclear_polarisation_in(&wrong, 0, 1).is_err());
    }
    assert!(nv.field_sensitivity(KR, [0.0, 0.0, 1.0], 0.0).is_err());
    assert!(nv.field_sensitivity(KR, [0.0; 3], 1e-6).is_err());
    // Scans of 42 levels beyond the budget are refused before any work.
    let big = three.nuclei.pop().map(|_| three).unwrap();
    assert_eq!(big.dimension(), 42);
    assert!(big.cw_odmr(&[2.8e9; 20], 1e6, KR).is_err());
    assert!(big.field_scan(&[[0.0, 0.0, 0.05]; 1000], KR).is_err());
    // A zero Rabi frequency is no microwave.
    let off = nv.steady_state(0.1 * KR, None).unwrap();
    let zero = nv.steady_state(0.1 * KR, Some((2.8e9, 0.0))).unwrap();
    assert!((&off.rho - &zero.rho).norm() < 1e-14);
}

/// Duarte et al.'s ¹⁵N (Phys. Rev. B 103, 195443 (2021), table 2: their
/// `A⊥` is twice the coefficient of `S_xI_x + S_yI_y`) with the rates of
/// Gupta et al. or Tetienne et al. and a nuclear T₁ of 100 ms. Their
/// excited coupling has the sign opposite to the default (only its size is
/// measured); the nuclear Zeeman term stays the physical `−γ B·I`.
fn duarte(b: f64, rates: Photophysics) -> NuclearOpticalNv {
    let mut n = NuclearSpin::nitrogen_15();
    n.ground = axial(3.9e6, 3.4e6);
    n.excited = axial(-38.5e6, -58.1e6);
    let mut nv = NuclearOpticalNv::new([0.0, 0.0, b], n);
    nv.electron.rates = rates;
    nv.nuclear_relaxation = 10.0;
    nv
}

#[test]
fn nitrogen_15_polarises_at_the_excited_anticrossing() {
    // Duarte et al. find about 95 % with Gupta's rates (experiment: 96 %,
    // and 98 % at 50 mT, 95 % at 44 mT in Jacques et al. 2009), almost
    // independent of the laser.
    let p =
        |b: f64, rates: Photophysics, pump: f64| duarte(b, rates).steady_state(pump, None).unwrap();
    let at = p(0.051, Photophysics::gupta(), 4e6);
    let (total, zero) = (at.polarisation[0], at.polarisation_zero[0]);
    assert!((0.94..0.96).contains(&total), "{total}");
    assert!((zero - total).abs() < 0.005, "{zero} vs {total}");
    let strong = p(0.051, Photophysics::gupta(), 70e6).polarisation[0];
    assert!((strong - total).abs() < 0.005, "{strong} vs {total}");
    let off = p(0.044, Photophysics::gupta(), 4e6).polarisation[0];
    assert!((0.93..total).contains(&off), "{off}");
    let low = p(0.02, Photophysics::gupta(), 4e6).polarisation[0];
    assert!(low < 0.7, "{low}");
    // Tetienne's singlet returns to m_s = ±1 almost as often as to 0: the
    // electron is polarised to 0.75, and |−1, ↑⟩ undoes the nucleus at the
    // anticrossing.
    let tetienne = p(0.051, Photophysics::tetienne(), 4e6).polarisation[0];
    assert!((0.79..0.82).contains(&tetienne), "{tetienne}");
}

#[test]
fn a_field_across_the_axis_dims_the_ground_anticrossing() {
    // ¹⁴N, 0.1 mT across the axis: the fluorescence dips where m_s = 0
    // and −1 cross (D/γ_e = 102.4 mT); along the axis the nucleus keeps
    // the crossing partners equally populated and nothing shows.
    let pump = 0.1 * KR;
    let crossing = NV_ZERO_FIELD_SPLITTING_OF(&NuclearOpticalNv::new(
        [0.0, 0.0, 0.1],
        NuclearSpin::nitrogen_14(),
    ));
    let fields: Vec<[f64; 3]> = [-0.002, -0.0005, 0.0, 0.0005, 0.002]
        .iter()
        .map(|d| [1e-4, 0.0, crossing + d])
        .collect();
    let nv = NuclearOpticalNv::new([1e-4, 0.0, crossing], NuclearSpin::nitrogen_14());
    let scan = nv.field_scan(&fields, pump).unwrap();
    let f: Vec<f64> = scan.iter().map(|x| x.0).collect();
    assert!(f[2] < 0.8 * f[0] && f[2] < 0.8 * f[4], "{f:?}");
    assert!(f[1] < f[0] && f[3] < f[4], "{f:?}");
    let along = NuclearOpticalNv::new([0.0, 0.0, crossing], NuclearSpin::nitrogen_14());
    let aligned = along
        .field_scan(&[[0.0, 0.0, crossing - 0.002], [0.0, 0.0, crossing]], pump)
        .unwrap();
    assert!((aligned[1].0 - aligned[0].0).abs() < 1e-4 * aligned[0].0);
    // The dip's slope makes a magnetometer without microwaves.
    let (eta, rate, slope) = nv.field_sensitivity(pump, [0.0, 0.0, 1.0], 1e-6).unwrap();
    assert!(eta.is_finite() && eta > 0.0 && rate > 0.0 && slope.is_finite());
}

/// `D/γ_e` of the centre (T).
#[allow(non_snake_case)]
fn NV_ZERO_FIELD_SPLITTING_OF(nv: &NuclearOpticalNv) -> f64 {
    nv.electron.zero_field_splitting / nv.electron.gyromagnetic_ratio
}

/// The anticrossings measured for the documentation (`cargo test --release
/// -- --ignored measure_anticrossings --nocapture`).
#[test]
#[ignore]
fn measure_anticrossings() {
    let clock = std::time::Instant::now();
    // ¹⁵N polarisation against the field (along the axis), Γ_n = 10/s.
    let fields = [0.02, 0.035, 0.044, 0.048, 0.051, 0.055, 0.07, 0.1];
    for (name, rates, jacques) in [
        ("Gupta, Duarte's hyperfine", Photophysics::gupta(), false),
        (
            "Tetienne, Duarte's hyperfine",
            Photophysics::tetienne(),
            false,
        ),
        (
            "Gupta, Jacques's isotropic 60 MHz",
            Photophysics::gupta(),
            true,
        ),
    ] {
        let mut line = String::new();
        for &b in &fields {
            let mut nv = duarte(b, rates);
            if jacques {
                nv.nuclei[0] = NuclearSpin::nitrogen_15();
            }
            let s = nv.steady_state(4e6, None).unwrap();
            line += &format!(" {:.0} mT {:.3};", b * 1e3, s.polarisation_zero[0]);
        }
        eprintln!("15N in m_s = 0, {name}:{line}");
    }
    let mut line = String::new();
    for &b in &fields {
        let mut nv = NuclearOpticalNv::new([0.0, 0.0, b], NuclearSpin::nitrogen_14());
        nv.electron.rates = Photophysics::gupta();
        nv.nuclear_relaxation = 10.0;
        let s = nv.steady_state(4e6, None).unwrap();
        line += &format!(" {:.0} mT {:.3};", b * 1e3, s.polarisation_zero[0]);
    }
    eprintln!("14N in m_s = 0, Gupta:{line}");
    // The ground-state anticrossing with ¹⁴N: the dip against the field
    // across the axis.
    let pump = 0.1 * KR;
    let probe = NuclearOpticalNv::new([0.0, 0.0, 0.1], NuclearSpin::nitrogen_14());
    let crossing = NV_ZERO_FIELD_SPLITTING_OF(&probe);
    for across in [0.0, 2e-5, 5e-5, 1e-4, 2e-4] {
        let nv = NuclearOpticalNv::new([across, 0.0, crossing], NuclearSpin::nitrogen_14());
        let grid: Vec<[f64; 3]> = (0..=200)
            .map(|k| [across, 0.0, crossing - 1e-3 + 1e-5 * k as f64])
            .collect();
        let scan = nv.field_scan(&grid, pump).unwrap();
        let rates: Vec<f64> = scan.iter().map(|x| x.0).collect();
        let base = rates[0].max(rates[200]);
        let (k, low) =
            rates.iter().enumerate().fold(
                (0, f64::INFINITY),
                |m, (k, &r)| if r < m.1 { (k, r) } else { m },
            );
        let half = base - (base - low) / 2.0;
        let width = rates.iter().filter(|&&r| r < half).count() as f64 * 1e-5;
        // The sensitivity where the slope of the scan is steepest, with
        // the rate there (not at the bottom of the dip).
        let steep = (0..200)
            .max_by(|&a, &b| {
                let s = |j: usize| (rates[j + 1] - rates[j]).abs();
                s(a).total_cmp(&s(b))
            })
            .unwrap();
        let mut at = nv.clone();
        at.electron.field[2] = crossing - 1e-3 + 1e-5 * (steep as f64 + 0.5);
        let (eta, _, _) = at.field_sensitivity(pump, [0.0, 0.0, 1.0], 1e-6).unwrap();
        eprintln!(
            "GSLAC, {:.0} μT across: dip {:.1} % at {:.3} mT, width {:.0} μT, \
             √R/|dR/dB| at the steepest point {:.2e} T/√Hz",
            across * 1e6,
            100.0 * (base - low) / base,
            grid[k][2] * 1e3,
            width * 1e6,
            eta
        );
    }
    // The microwave near the ground-state anticrossing (101.7 mT, the
    // line at 20 MHz): the laboratory frame against the rotating-wave
    // model, Ω = 15 MHz.
    let nv = decoupled([0.0, 0.0, 0.1017]);
    let [(minus, _), _] = nv.electron.lines().unwrap();
    let f0 = minus.abs();
    let freqs: Vec<f64> = (0..=60).map(|k| f0 - 15e6 + 0.5e6 * k as f64).collect();
    let t = std::time::Instant::now();
    let lab = nv.cw_odmr(&freqs, 15e6, pump).unwrap();
    let lab_time = t.elapsed().as_secs_f64();
    let rwa = nv.electron.cw_odmr(&freqs, 15e6, pump).unwrap();
    let dip = |r: &[f64]| {
        let k = r
            .iter()
            .enumerate()
            .fold(
                (0, f64::INFINITY),
                |m, (k, &x)| if x < m.1 { (k, x) } else { m },
            )
            .0;
        freqs[k]
    };
    eprintln!(
        "101.7 mT, line {:.2} MHz, Ω = 15 MHz: dip at {:.1} MHz in the laboratory, {:.1} MHz rotating wave; 61 points in {lab_time:.1} s",
        f0 / 1e6,
        dip(&lab) / 1e6,
        dip(&rwa) / 1e6
    );
    // Costs.
    for (name, nv) in [
        (
            "15N",
            NuclearOpticalNv::new([0.0, 0.0, 0.051], NuclearSpin::nitrogen_15()),
        ),
        (
            "14N",
            NuclearOpticalNv::new([0.0, 0.0, 0.051], NuclearSpin::nitrogen_14()),
        ),
        ("14N and 13C", {
            let mut nv = NuclearOpticalNv::new([0.0, 0.0, 0.051], NuclearSpin::nitrogen_14());
            let c = [[13e6, 0.0, 0.0], [0.0, 13e6, 0.0], [0.0, 0.0, 15e6]];
            nv.nuclei.push(NuclearSpin::carbon_13(c, c));
            nv.nuclear_relaxation = 10.0;
            nv
        }),
    ] {
        let t = std::time::Instant::now();
        let s = nv.steady_state(pump, None).unwrap();
        let steady = t.elapsed().as_secs_f64();
        let [(_, _), (plus, _)] = nv.electron.lines().unwrap();
        let t = std::time::Instant::now();
        let driven = nv.steady_state(pump, Some((plus.abs(), 1e6))).unwrap();
        eprintln!(
            "{name} ({} levels): steady state {steady:.2} s (P = {:.3}), with a microwave {:.2} s ({} harmonics)",
            nv.dimension(),
            s.polarisation[0],
            t.elapsed().as_secs_f64(),
            driven.harmonics
        );
    }
    eprintln!("all in {:.1} s", clock.elapsed().as_secs_f64());
}
