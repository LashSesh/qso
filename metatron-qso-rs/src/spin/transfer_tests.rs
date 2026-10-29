//! Spin-order transfer and storage: M2S and S2M, field cycling of
//! parahydrogen, singlet storage under a spin lock.

use super::*;

/// Two protons of a near-equivalent pair: coupling `j`, shift difference
/// `dnu` (Hz) at field `b`.
fn pair(b: f64, j: f64, dnu: f64) -> SpinSystem {
    let mut sys = SpinSystem::new(b);
    let shift = dnu / Isotope::H1.larmor(b) * 1e6;
    sys.add_spin(Isotope::H1, 0.0).unwrap();
    sys.add_spin(Isotope::H1, shift).unwrap();
    sys.couple(0, 1, j).unwrap();
    sys
}

/// `ρ` with only its singlet order `I₁·I₂` kept (all else relaxed or
/// dephased during storage).
fn singlet_order_only(sys: &SpinSystem, rho: &CMatrix) -> CMatrix {
    let q = sys.dot(0, 1).unwrap();
    let c = sys.expectation(rho, &q).unwrap().re / sys.expectation(&q, &q).unwrap().re;
    sys.identity_state() + q * C64::new(c, 0.0)
}

#[test]
fn m2s_and_s2m_reach_two_thirds() {
    // Pileio, Carravetta and Levitt: storing magnetisation as singlet
    // order and reading it back returns at most 2/3 of it (at high
    // temperature). Their ¹⁵N₂O pair: J = 8.21 Hz, Δν ≈ 0.78 Hz.
    for (j, dnu) in [(8.21, 0.78), (15.0, 2.0), (7.0, 0.3)] {
        let sys = pair(1.0, j, dnu);
        let rho = sys.thermal_state().unwrap();
        let fz = sys.fz(Isotope::H1).unwrap();
        let m0 = sys.expectation(&rho, &fz).unwrap().re;
        let stored = singlet_order_only(
            &sys,
            &sys.evolve(&rho, &m2s(Isotope::H1, j, dnu).unwrap())
                .unwrap(),
        );
        let back = sys
            .evolve(&stored, &s2m(Isotope::H1, j, dnu).unwrap())
            .unwrap();
        let efficiency = (sys.expectation(&back, &fz).unwrap().re / m0).abs();
        assert!(efficiency <= 2.0 / 3.0 + 1e-9, "{j} {dnu}: {efficiency}");
        assert!(efficiency > 0.97 * 2.0 / 3.0, "{j} {dnu}: {efficiency}");
    }
    assert!(m2s(Isotope::H1, 1.0, 2.0).is_err());
    assert!(m2s(Isotope::H1, 1.0, 0.0).is_err());
}

/// Eigenvalues and eigenvectors of `h` restricted to states with total
/// `2M` equal to `m`, ascending.
fn subspace(sys: &SpinSystem, h: &CMatrix, m: i32) -> (Vec<usize>, Vec<f64>, CMatrix) {
    let n = sys.spins().len();
    let states: Vec<usize> = (0..sys.dimension())
        .filter(|&s| {
            (0..n)
                .map(|i| (2.0 * sys.m(i, s)).round() as i32)
                .sum::<i32>()
                == m
        })
        .collect();
    let sub = CMatrix::from_fn(states.len(), states.len(), |r, c| h[(states[r], states[c])]);
    let (e, v) = crate::linalg::eigh(&sub);
    (states, e, v)
}

#[test]
fn field_cycling_moves_parahydrogen_order_to_carbon() {
    // Two protons from parahydrogen and a ¹³C (AA'X): the singlet, formed
    // at zero field and averaged there, is carried adiabatically to 50 μT
    // (Jóhannesson, Axelsson and Karlsson 2004). Sudden-adiabatic theory:
    // the populations of the zero-field eigenstates go to the eigenstates
    // of the same rank and total M at the final field.
    let mut sys = SpinSystem::new(0.0).with_frame(Frame::Laboratory);
    sys.add_spin(Isotope::H1, 2.0).unwrap();
    sys.add_spin(Isotope::H1, 4.0).unwrap();
    sys.add_spin(Isotope::C13, 170.0).unwrap();
    sys.couple(0, 1, 7.2).unwrap();
    sys.couple(0, 2, 5.8).unwrap();
    sys.couple(1, 2, -1.4).unwrap();
    let start = sys
        .time_average(&sys.parahydrogen_state(0, 1, 1.0).unwrap())
        .unwrap();
    let end = 5e-5;
    let (h0, h1) = (sys.hamiltonian_at(0.0), sys.hamiltonian_at(end));
    let sz = sys.iz(2).unwrap();
    let mut want = 0.0;
    for m in [-3, -1, 1, 3] {
        let (states, _, v0) = subspace(&sys, &h0, m);
        let (_, _, v1) = subspace(&sys, &h1, m);
        let sub = |op: &CMatrix| {
            CMatrix::from_fn(states.len(), states.len(), |r, c| {
                op[(states[r], states[c])]
            })
        };
        let p = v0.adjoint() * sub(&start) * &v0;
        let z = v1.adjoint() * sub(&sz) * &v1;
        for k in 0..states.len() {
            want += 2.0 * p[(k, k)].re * z[(k, k)].re;
        }
    }
    assert!(want.abs() > 0.1, "{want}");
    // The simulation: slowly through the ultra-low fields, where the
    // levels of one M come within 1.75 Hz (about 15 nT), then faster.
    let carried = sys
        .evolve(
            &start,
            &[
                Event::Sweep {
                    to: 1e-6,
                    duration: 40.0,
                    steps: 8000,
                },
                Event::Sweep {
                    to: end,
                    duration: 10.0,
                    steps: 2000,
                },
            ],
        )
        .unwrap();
    let got = sys.polarization(&carried, 2).unwrap();
    assert!((got - want).abs() < 5e-4, "{got} vs {want}");
    // A sudden jump instead transfers nothing to carbon.
    let jumped = sys.evolve(&start, &[Event::Field(end)]).unwrap();
    assert!(sys.polarization(&jumped, 2).unwrap().abs() < 1e-12);
}

#[test]
fn a_spin_lock_stores_singlet_order_for_its_redfield_lifetime() {
    // Near-equivalent protons 1.8 Å apart with a third proton nearby, at
    // 9.4 T: under a 1 kHz spin lock their singlet order decays at the
    // Redfield rate of I₁·I₂, many times slower than their magnetisation.
    let mut sys = pair(9.4, 15.0, 4.0);
    let k = sys.add_spin(Isotope::H1, 3.0).unwrap();
    for (i, p) in [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [0.9, 2.6, 0.0]]
        .into_iter()
        .enumerate()
    {
        sys.set_position(i, p).unwrap();
    }
    let _ = k;
    let sys = sys.with_correlation_time(Some(2e-11)).unwrap();
    let q = sys.dot(0, 1).unwrap();
    let fz = sys.iz(0).unwrap() + sys.iz(1).unwrap();
    let g = sys.relaxation_rates(&[q.clone(), fz]).unwrap();
    let (rate_s, rate_1) = (g[(0, 0)], g[(1, 1)]);
    assert!(rate_1 / rate_s > 3.0, "T_S/T_1 = {}", rate_1 / rate_s);
    let rho = sys.parahydrogen_state(0, 1, 1.0).unwrap();
    let s0 = sys.expectation(&rho, &q).unwrap().re;
    let t = 0.5 / rate_s;
    let locked = sys
        .evolve(&rho, &[spin_lock(Isotope::H1, 1000.0, t, 0.0)])
        .unwrap();
    let decay = sys.expectation(&locked, &q).unwrap().re / s0;
    let want = (-rate_s * t).exp();
    assert!((decay / want - 1.0).abs() < 2e-3, "{decay} vs {want}");
    // Without the lock the shift difference mixes the singlet with the
    // faster-relaxing triplet.
    let free = sys.evolve(&rho, &[Event::Delay(t)]).unwrap();
    let unlocked = sys.expectation(&free, &q).unwrap().re / s0;
    assert!(unlocked < 0.95 * want, "{unlocked} vs {want}");
}
