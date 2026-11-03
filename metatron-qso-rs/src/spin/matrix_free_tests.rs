//! Redfield relaxation applied from its terms without storing the
//! superoperator, against the stored blocks of 1.41.

use super::blocks::Method;
use super::*;

/// The largest entry of `x − y` against the largest of `y − Tr(y)/d`.
fn relative(x: &CMatrix, y: &CMatrix) -> f64 {
    let d = y.nrows();
    let dev = y - CMatrix::identity(d, d) * (y.trace() / d as f64);
    let scale = dev.iter().fold(0.0f64, |m, z| m.max(z.norm()));
    (x - y).iter().fold(0.0f64, |m, z| m.max(z.norm())) / scale
}

/// `n` protons on a zigzag chain (1.8 Å apart) with dipolar Redfield
/// relaxation at `tau`, a carbon with shielding anisotropy at its end,
/// couplings to the neighbours.
pub(super) fn chain(n: usize, b: f64, tau: f64, frame: Frame) -> SpinSystem {
    let mut sys = SpinSystem::new(b).with_frame(frame);
    for i in 0..n {
        sys.add_spin(Isotope::H1, 1.0 + 0.37 * i as f64).unwrap();
        sys.set_position(i, [1.8 * i as f64, 0.6 * (i % 2) as f64, 0.1 * i as f64])
            .unwrap();
        if i > 0 {
            sys.couple(i - 1, i, 7.0 - 0.5 * i as f64).unwrap();
        }
    }
    let c = sys.add_spin(Isotope::C13, 40.0).unwrap();
    sys.set_position(c, [-1.1, 0.4, 0.2]).unwrap();
    sys.couple(0, c, 140.0).unwrap();
    sys.set_csa(c, Some(redfield::axial_csa(40.0, [0.3, 0.1, 1.0]).unwrap()))
        .unwrap();
    sys.with_correlation_time(Some(tau)).unwrap()
}

fn pulsed(sys: &SpinSystem) -> CMatrix {
    sys.evolve(
        &sys.thermal_state().unwrap(),
        &[Event::pulse(Target::Isotope(Isotope::H1), 70.0, 30.0)],
    )
    .unwrap()
}

#[test]
fn matrix_free_relaxation_equals_the_stored_blocks() {
    // Rotating and laboratory frames, at high and low field (where Y(𝒬)
    // fills the label blocks), a state with coherences and the identity
    // (the drift alone), and a Hamiltonian that mixes the orders (one
    // block of all elements).
    let cases = [
        (chain(4, 9.4, 3e-11, Frame::Rotating), 0.05, true),
        (chain(4, 1e-3, 3e-11, Frame::Laboratory), 0.05, true),
        (chain(3, 2.0, 1e-10, Frame::Laboratory), 0.02, true),
        (chain(3, 9.4, 3e-11, Frame::Rotating), 2e-4, false),
    ];
    for (sys, t, conserving) in cases {
        let h = if conserving {
            sys.hamiltonian().unwrap()
        } else {
            sys.hamiltonian().unwrap() + sys.fx(Isotope::H1).unwrap() * C64::new(TWO_PI * 2e3, 0.0)
        };
        let relaxation = sys.relaxation(sys.field()).unwrap();
        for rho in [pulsed(&sys), sys.identity_state()] {
            let run = |m| {
                sys.relax_planned(&rho, &h, t, conserving, &relaxation, Some(m))
                    .unwrap()
            };
            let stored = run(Method::Krylov);
            let free = run(Method::MatrixFree);
            // Both are Krylov steps to the same tolerance (in the
            // laboratory frame at 2 T they differ from the dense
            // exponentials by 2·10⁻⁹): the matrix-free ones agree with the
            // stored and are no further from the dense.
            let err = relative(&free, &stored);
            assert!(
                err < 1e-9,
                "{} spins at {} T, t = {t}: {err:e}",
                sys.spins().len(),
                sys.field()
            );
            if conserving {
                let dense = run(Method::Dense);
                let (a, b) = (relative(&free, &dense), relative(&stored, &dense));
                assert!(a <= 1.1 * b + 1e-12, "{} T: {a:e} vs {b:e}", sys.field());
            }
        }
    }
}

#[test]
fn matrix_free_decays_equal_the_stored_blocks() {
    let sys = chain(4, 9.4, 3e-11, Frame::Rotating);
    let rho = pulsed(&sys);
    let run = |m| {
        sys.relaxing_fid_planned(&rho, Isotope::H1, 2e-4, 8, Some(m))
            .unwrap()
    };
    let (stored, free) = (run(Method::Krylov), run(Method::MatrixFree));
    let scale = stored[0].norm();
    for k in 0..8 {
        assert!((free[k] - stored[k]).norm() < 1e-10 * scale, "{k}");
    }
}

/// Chains of seven to nine protons and a carbon with dipolar Redfield
/// relaxation: 0.1 s at 10 µT in the laboratory frame from a 70° pulse
/// (field cycling), and 1 s at 9.4 T in the rotating frame after
/// inversion (populations), for the documentation (`cargo test --release
/// -- --ignored measure_matrix_free --nocapture`).
#[test]
#[ignore]
fn measure_matrix_free() {
    for (b, frame, angle) in [
        (1e-5, Frame::Laboratory, 70.0),
        (9.4, Frame::Rotating, 180.0),
    ] {
        for (n, t) in [(7usize, 0.1), (8, 0.01), (9, 0.001)] {
            let sys = chain(n, b, 3e-11, frame);
            let clock = std::time::Instant::now();
            let relaxation = sys.relaxation(sys.field()).unwrap();
            let built = clock.elapsed().as_secs_f64();
            let liouville = blocks::Liouville::new(&sys.labels(), None);
            let stored: f64 = (0..liouville.blocks.len())
                .map(|k| relaxation.redfield_entries(&liouville, Some(k)).0)
                .sum();
            let terms = sys.held_bytes(&relaxation);
            let rho = sys
                .evolve(
                    &sys.thermal_state().unwrap(),
                    &[Event::pulse(Target::Isotope(Isotope::H1), angle, 0.0)],
                )
                .unwrap();
            let clock = std::time::Instant::now();
            let after = sys.evolve(&rho, &[Event::Delay(t)]);
            let elapsed = clock.elapsed().as_secs_f64();
            eprintln!(
                "{} spins at {b} T: terms {built:.1} s, {:.2} GB held; the stored \
                 superoperator about {stored:.1e} entries ({:.1} GB); {t} s: {elapsed:.1} s, {}",
                n + 1,
                terms / 1e9,
                stored * 20.0 / 1e9,
                match after {
                    Ok(r) => format!(
                        "⟨I₀z⟩ = {:.6e}",
                        sys.expectation(&r, &sys.iz(0).unwrap()).unwrap().re
                    ),
                    Err(e) => format!("refused: {e}"),
                }
            );
        }
    }
}

/// `R(x)` (or `R†(x)`) of a whole `d × d` matrix, block by block, from the
/// terms.
fn apply_all(r: &redfield::RedfieldBlocks, x: &CMatrix, adjoint: bool) -> CMatrix {
    let d = x.nrows();
    let mut out = vec![C64::new(0.0, 0.0); d * d];
    for (q, (_, elements)) in r.liouville.blocks.iter().enumerate() {
        let v: Vec<C64> = elements.iter().map(|&(a, b)| x[(a, b)]).collect();
        let positions: Vec<u32> = elements.iter().map(|&(a, b)| (a * d + b) as u32).collect();
        r.apply_products(q, &v, C64::new(1.0, 0.0), &positions, adjoint, &mut out);
    }
    CMatrix::from_fn(d, d, |a, b| out[a * d + b])
}

/// A pseudo-random matrix.
fn random(d: usize, seed: u64) -> CMatrix {
    let mut rng = crate::rng::Rng::seed_from_u64(seed);
    CMatrix::from_fn(d, d, |_, _| C64::new(rng.normal(), rng.normal()))
}

#[test]
fn the_field_derivative_of_redfield_relaxation_is_exact() {
    // R(B) from the affine operators equals the stored blocks of 1.41; its
    // adjoint is the adjoint; R'(B) equals Richardson-extrapolated central
    // differences, at high field, at low field where the eigenbasis mixes
    // the spins, and at zero field (the shielding then contributes only to
    // the derivative).
    for (b, frame) in [
        (9.4, Frame::Rotating),
        (2e-3, Frame::Laboratory),
        (3e-5, Frame::Laboratory),
        (0.0, Frame::Laboratory),
    ] {
        let sys = chain(3, b, 3e-11, frame);
        let d = sys.dimension();
        let x = random(d, 3);
        let z = random(d, 5);
        let (r, dr) = sys.redfield_field(b).unwrap().unwrap();
        let stored = sys.redfield(b).unwrap().unwrap();
        let mut want = CMatrix::zeros(d, d);
        stored.apply(&x, &mut want, 1.0);
        let got = apply_all(&r, &x, false);
        let size = want.iter().fold(0.0f64, |m, v| m.max(v.norm()));
        let e = (&got - &want).iter().fold(0.0f64, |m, v| m.max(v.norm())) / size;
        assert!(e < 1e-12, "{b}: R {e:e}");
        // ⟨z, R x⟩ = ⟨R† z, x⟩.
        let lhs = (z.adjoint() * &got).trace();
        let rhs = (apply_all(&r, &z, true).adjoint() * &x).trace();
        assert!(
            (lhs - rhs).norm() < 1e-12 * lhs.norm().max(size),
            "{b}: {lhs} vs {rhs}"
        );
        // The derivative.
        // Steps below 10⁻⁵ T lose more to rounding than they gain.
        let span = 1e-2 * b.abs().max(1e-3);
        let at = |f: f64| {
            let mut m = CMatrix::zeros(d, d);
            sys.redfield(f).unwrap().unwrap().apply(&x, &mut m, 1.0);
            m
        };
        let central = |h: f64| (at(b + h) - at(b - h)) * C64::new(0.5 / h, 0.0);
        let fd = central(span / 2.0) * C64::new(4.0 / 3.0, 0.0)
            - central(span) * C64::new(1.0 / 3.0, 0.0);
        let exact = apply_all(&dr, &x, false);
        let scale = fd.iter().fold(0.0f64, |m, v| m.max(v.norm()));
        let e = (&exact - &fd).iter().fold(0.0f64, |m, v| m.max(v.norm())) / scale;
        assert!(e < 1e-6, "{b}: R' {e:e} of {scale:e}");
    }
}
