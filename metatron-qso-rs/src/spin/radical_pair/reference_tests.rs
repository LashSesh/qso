//! Pinned reference values of radical pairs (recomputed independently in
//! Liouville space or by the eigenbasis formula of the exponential model,
//! with QSO's gyromagnetic ratios), and the measurements of 1.65.

use super::*;
use crate::spin::{FREE_ELECTRON_G, Frame, Isotope, SpinSystem};

/// The flavin–tryptophan pair of `examples/spins/flavin_tryptophan.json`:
/// laboratory frame, g = g_e, ¹⁴N and two ¹H on the flavin, three ¹H on
/// the tryptophan.
fn flavin_tryptophan() -> SpinSystem {
    SpinSystem::from_json(include_str!(
        "../../../examples/spins/flavin_tryptophan.json"
    ))
    .unwrap()
}

#[test]
fn the_flavin_tryptophan_pair_gives_its_mary_curve() {
    let sys = flavin_tryptophan();
    assert_eq!(sys.dimension(), 384);
    let table = [
        (0.0, 0.344_285_671_543_554_05, 0.519_240_322_424_056_7),
        (0.1, 0.344_086_564_190_344_1, 0.519_325_464_010_282_5),
        (1.0, 0.378_262_108_028_472_93, 0.540_386_853_520_068_2),
        (2.0, 0.464_146_500_509_986_07, 0.603_475_801_613_570_1),
        (5.0, 0.523_848_440_349_130_7, 0.685_247_099_761_334_5),
        (50.0, 0.534_489_461_607_385_4, 0.712_057_019_151_788_8),
    ];
    let fields: Vec<f64> = table.iter().map(|r| r.0 * 1e-3).collect();
    for (column, k) in [(1, 2e7), (2, 1e8)] {
        let pair = RadicalPair::new(sys.clone())
            .unwrap()
            .with_rates(k, k, 0.0)
            .unwrap();
        let curve = pair.field_effect(Birth::Singlet, &fields).unwrap();
        for (row, y) in table.iter().zip(&curve) {
            let want = if column == 1 { row.1 } else { row.2 };
            assert!(
                (y.singlet - want).abs() <= 1e-12,
                "k = {k:e}, {} mT: {} vs {want}",
                row.0,
                y.singlet
            );
            assert!((y.singlet + y.triplet - 1.0).abs() <= 1e-12);
            assert_eq!(y.field, row.0 * 1e-3);
        }
        if column == 1 {
            // The low-field effect: a dip below the zero-field yield.
            assert!(curve[1].singlet < curve[0].singlet);
            let triplet = pair.field_effect(Birth::Triplet, &[0.0, 50e-3]).unwrap();
            assert!((triplet[0].singlet - 0.218_571_442_818_815_35).abs() <= 1e-12);
            assert!((triplet[1].singlet - 0.155_170_179_464_204_75).abs() <= 1e-12);
        }
    }
}

/// N5 of the flavin (¹⁴N, 0.514 mT) on radical 1, Hβ1 and H1 of the
/// tryptophan on radical 2, laboratory frame at 1 mT; `k_S = 2·10⁷`,
/// `k_T = 5·10⁶`, `k_E = 10⁶`; electron `T₁` 200 ns and `T₂` 100 ns when
/// `temperature` is given.
fn three_nuclei(model: Recombination, temperature: Option<f64>) -> RadicalPair {
    let mut sys = SpinSystem::new(1e-3).with_frame(Frame::Laboratory);
    let e1 = sys.add_electron(FREE_ELECTRON_G).unwrap();
    let e2 = sys.add_electron(FREE_ELECTRON_G).unwrap();
    let n = sys.add_spin(Isotope::N14, 0.0).unwrap();
    let h1 = sys.add_spin(Isotope::H1, 0.0).unwrap();
    let h2 = sys.add_spin(Isotope::H1, 0.0).unwrap();
    sys.hyperfine(e1, n, 0.514_140_613_991_168_1).unwrap();
    sys.hyperfine(e2, h1, 1.6045).unwrap();
    sys.hyperfine(e2, h2, -0.5983).unwrap();
    if let Some(kelvin) = temperature {
        sys = sys.with_temperature(kelvin).unwrap();
        for e in [e1, e2] {
            sys.set_relaxation(e, Some(200e-9), Some(100e-9)).unwrap();
        }
    }
    RadicalPair::new(sys)
        .unwrap()
        .with_rates(2e7, 5e6, 1e6)
        .unwrap()
        .with_recombination(model)
}

#[test]
fn three_nuclei_give_the_liouville_references() {
    let cases = [
        (
            Recombination::Haberkorn,
            None,
            [
                0.609_072_558_439_314_3,
                0.300_394_844_698_933_2,
                0.090_532_596_861_752_37,
            ],
            1e-13,
        ),
        (
            Recombination::JonesMaedaHore,
            None,
            [
                0.594_864_731_399_94,
                0.312_826_693_358_385_75,
                0.092_308_575_241_674_16,
            ],
            1e-12,
        ),
        (
            Recombination::Haberkorn,
            Some(1e9),
            [
                0.566_316_116_824_479,
                0.337_806_731_111_914_3,
                0.095_877_152_063_606_81,
            ],
            1e-12,
        ),
        (
            Recombination::JonesMaedaHore,
            Some(1e9),
            [
                0.564_036_041_299_369_5,
                0.339_801_797_196_384_9,
                0.096_162_161_504_245_47,
            ],
            1e-12,
        ),
        (
            Recombination::Haberkorn,
            Some(298.15),
            [
                0.566_316_116_840_549_9,
                0.337_806_731_097_852_14,
                0.095_877_152_061_597_93,
            ],
            1e-12,
        ),
    ];
    let mut singlets = Vec::new();
    for (model, temperature, want, tol) in cases {
        let pair = three_nuclei(model, temperature);
        let y = pair.solve(Birth::Singlet).unwrap().yields().clone();
        for (got, w) in [y.singlet, y.triplet, y.escape].into_iter().zip(want) {
            assert!(
                (got - w).abs() <= tol,
                "{model:?} {temperature:?}: {got} vs {w} ({:e})",
                got - w
            );
        }
        singlets.push(y.singlet);
        if model == Recombination::Haberkorn && temperature.is_none() {
            let singlet = [
                0.003_290_713_325_497_38,
                0.004_600_285_239_359_125,
                -0.004_874_698_935_651_04,
            ];
            let triplet = [
                -0.002_725_868_586_224_34,
                -0.002_037_280_259_163_62,
                0.003_102_596_178_892_12,
            ];
            for (j, c) in y.cidnp.iter().enumerate() {
                assert!((c[0] - singlet[j]).abs() <= 1e-14, "{j}: {}", c[0]);
                assert!((c[1] - triplet[j]).abs() <= 1e-14, "{j}: {}", c[1]);
            }
        }
    }
    // The thermal bias of the electrons at 298.15 K against 10⁹ K moves
    // Φ_S by 1.607·10⁻¹¹, resolved at the tolerance.
    let bias = singlets[4] - singlets[2];
    assert!((bias - 1.607e-11).abs() <= 1e-12, "{bias:e}");
}

/// A pair of two electrons, one ¹⁴N on radical 1 and `protons` ¹H shared
/// between the radicals (laboratory frame unless `rotating`).
fn measured_pair(protons: usize, b: f64, frame: Frame) -> RadicalPair {
    let mut sys = SpinSystem::new(b).with_frame(frame);
    let e1 = sys.add_electron(FREE_ELECTRON_G + 1e-3).unwrap();
    let e2 = sys.add_electron(FREE_ELECTRON_G).unwrap();
    let n = sys.add_spin(Isotope::N14, 0.0).unwrap();
    sys.hyperfine(e1, n, 0.514).unwrap();
    for k in 0..protons {
        let h = sys.add_spin(Isotope::H1, 0.0).unwrap();
        let a = 1.6 * (0.8f64).powi(k as i32) * if k % 2 == 0 { 1.0 } else { -1.0 };
        sys.hyperfine(if k % 2 == 0 { e2 } else { e1 }, h, a)
            .unwrap();
    }
    RadicalPair::new(sys)
        .unwrap()
        .with_rates(2e7, 2e7, 0.0)
        .unwrap()
}

/// Seconds of one call.
fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let start = std::time::Instant::now();
    let out = f();
    (out, start.elapsed().as_secs_f64())
}

/// The measurements of 1.65 (run with `--ignored --nocapture` in release,
/// with `RAYON_NUM_THREADS` 4 and 1).
#[test]
#[ignore]
fn measure_radical_pairs() {
    // Flavin–tryptophan, one field and a sweep of 100.
    let ft = RadicalPair::new(flavin_tryptophan())
        .unwrap()
        .with_rates(2e7, 2e7, 0.0)
        .unwrap();
    let (y, s) = timed(|| ft.solve(Birth::Singlet).unwrap().yields().clone());
    println!(
        "flavin-tryptophan 384, one field: {:.1} ms (η {:.1e})",
        s * 1e3,
        y.backward_error
    );
    let fields: Vec<f64> = (0..100).map(|k| k as f64 * 0.5e-3).collect();
    let (_, s) = timed(|| ft.field_effect(Birth::Singlet, &fields).unwrap());
    println!(
        "flavin-tryptophan 384, 100 fields: {:.1} ms per field",
        s * 10.0
    );
    // Sizes 96 to 1536 by the Sylvester path.
    for protons in [3, 5, 7] {
        for (b, frame) in [
            (0.0, Frame::Laboratory),
            (1e-3, Frame::Laboratory),
            (50e-3, Frame::Laboratory),
            (0.35, Frame::Rotating),
        ] {
            let pair = measured_pair(protons, b, frame);
            let d = pair.system().dimension();
            let largest = pair
                .model(b, false)
                .blocks
                .iter()
                .map(Vec::len)
                .max()
                .unwrap();
            let (y, s) = timed(|| pair.solve(Birth::Singlet).unwrap().yields().clone());
            println!(
                "Sylvester {d} ({frame:?}, {b} T): {s:.3} s, largest block {largest}, η {:.1e}, Φ_S {:.6}",
                y.backward_error, y.singlet
            );
        }
    }
    // 2e + 9 ¹H (2048).
    let mut sys = SpinSystem::new(1e-3).with_frame(Frame::Laboratory);
    let e1 = sys.add_electron(FREE_ELECTRON_G).unwrap();
    let e2 = sys.add_electron(FREE_ELECTRON_G).unwrap();
    for k in 0..9 {
        let h = sys.add_spin(Isotope::H1, 0.0).unwrap();
        sys.hyperfine(if k % 2 == 0 { e1 } else { e2 }, h, 1.5 * (0.8f64).powi(k))
            .unwrap();
    }
    let big = RadicalPair::new(sys)
        .unwrap()
        .with_rates(2e7, 2e7, 0.0)
        .unwrap();
    let (y, s) = timed(|| big.solve(Birth::Singlet).unwrap().yields().clone());
    println!("Sylvester 2048: {s:.3} s, η {:.1e}", y.backward_error);
    // GMRES: the term of Jones, Maeda and Hore, and electron relaxation.
    for (name, pair) in [
        ("384", ft.clone()),
        ("1536", measured_pair(7, 1e-3, Frame::Laboratory)),
    ] {
        let jmh = pair
            .clone()
            .with_recombination(Recombination::JonesMaedaHore);
        let (y, s) = timed(|| jmh.solve(Birth::Singlet).unwrap().yields().clone());
        println!(
            "GMRES JMH {name}: {} iterations, {s:.3} s, η {:.1e}",
            y.iterations, y.backward_error
        );
        for ke in [0.0, 1e6] {
            let mut sys = pair.system().clone();
            for e in pair.electrons() {
                sys.set_relaxation(e, Some(1e-6), Some(0.5e-6)).unwrap();
            }
            let relaxing = RadicalPair::new(sys)
                .unwrap()
                .with_rates(2e7, 2e7, ke)
                .unwrap();
            let (r, s) = timed(|| relaxing.solve(Birth::Singlet));
            match r {
                Ok(sol) => println!(
                    "GMRES relaxation {name}, k_E {ke:e}: {} iterations, {s:.3} s, η {:.1e}",
                    sol.yields().iterations,
                    sol.yields().backward_error
                ),
                Err(e) => {
                    println!("GMRES relaxation {name}, k_E {ke:e}: refused after {s:.3} s: {e}")
                }
            }
        }
    }
    // The whole path (a start with nuclear coherence).
    for protons in [5, 7] {
        let pair = measured_pair(protons, 1e-3, Frame::Laboratory);
        let d = pair.system().dimension();
        let n = pair.nuclei()[1];
        let rho0 =
            pair.birth_state(Birth::Singlet) + pair.system().ix(n).unwrap() * C64::new(1e-3, 0.0);
        let (r, s) = timed(|| pair.solve_from(&rho0));
        match r {
            Ok(sol) => println!(
                "whole path {d}: {s:.3} s, η {:.1e}",
                sol.yields().backward_error
            ),
            Err(e) => println!("whole path {d}: refused after {s:.3} s: {e}"),
        }
    }
    // Products and evolution at 1536.
    let pair = measured_pair(7, 1e-3, Frame::Laboratory);
    let solution = pair.solve(Birth::Singlet).unwrap();
    let mut target = SpinSystem::new(9.4);
    for j in pair.nuclei() {
        target
            .add_spin(pair.system().spins()[j].isotope, 0.0)
            .unwrap();
    }
    target.add_spin(Isotope::C13, 0.0).unwrap();
    let routes: Vec<(usize, usize)> = pair.nuclei().into_iter().zip(0..).collect();
    let (_, s) = timed(|| {
        solution
            .product_on(
                Product::Singlet,
                &target,
                &routes,
                crate::spin::Fresh::Thermal,
            )
            .unwrap()
    });
    println!("product_on 1536 → 512: {s:.3} s");
    let rho = pair.birth_state(Birth::Singlet);
    let (_, s) = timed(|| pair.evolve(&rho, 1e-8).unwrap());
    println!("evolve 1536 (blocks of H): {s:.3} s");
    let mut sys = measured_pair(3, 1e-3, Frame::Laboratory).system().clone();
    for e in [0, 1] {
        sys.set_relaxation(e, Some(1e-6), Some(0.5e-6)).unwrap();
    }
    let relaxing = RadicalPair::new(sys)
        .unwrap()
        .with_rates(2e7, 2e7, 0.0)
        .unwrap();
    let b = relaxing.system().field();
    let model = relaxing.model(b, false);
    let n: usize = model.blocks.iter().map(|s| s.len() * s.len()).sum();
    let rho = relaxing.birth_state(Birth::Singlet);
    let (_, s) = timed(|| relaxing.evolve(&rho, 1e-8).unwrap());
    println!("evolve with relaxation, Liouville space {n}: {s:.3} s");
    // Reactions with and without a source (four protons at 9.4 T).
    let mut site = SpinSystem::new(9.4);
    for k in 0..4 {
        let h = site.add_spin(Isotope::H1, k as f64).unwrap();
        site.set_relaxation(h, Some(2.0), Some(1.0)).unwrap();
    }
    let solution = pair.solve(Birth::Singlet).unwrap();
    let routes: Vec<(usize, usize)> = pair.nuclei()[1..5].iter().copied().zip(0..).collect();
    let state = solution
        .product_on(
            Product::Singlet,
            &site,
            &routes,
            crate::spin::Fresh::Thermal,
        )
        .unwrap();
    for with_source in [false, true] {
        let mut r = crate::spin::Reactions::new();
        let p = r.add_site(site.clone(), 0.5).unwrap();
        let d = r.add_species("D", 0.0).unwrap();
        r.add_reaction(crate::spin::Reaction {
            reactants: vec![crate::spin::Participant::Site(p)],
            products: vec![crate::spin::Participant::Species(d)],
            rate: 0.5,
            carried: Vec::new(),
            fresh: crate::spin::Fresh::Thermal,
        })
        .unwrap();
        if with_source {
            r.add_source(p, 0.1, &state).unwrap();
        }
        let start = vec![site.thermal_state().unwrap() * C64::new(0.5, 0.0)];
        let (_, s) = timed(|| {
            r.evolve(&start, &[crate::spin::Event::Delay(10.0)])
                .unwrap()
        });
        println!("reactions over 10 s, source {with_source}: {s:.3} s");
    }
}
