use super::*;

/// PySCF's fingerprint `Σ_k a_k cos k` over the row-major entries.
fn fingerprint(values: impl Iterator<Item = f64>) -> f64 {
    values.enumerate().map(|(k, v)| v * (k as f64).cos()).sum()
}

const H2O: &str = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692";
const NH3: &str =
    "N 0.1 -0.05 0.12; H 0.9377 -0.3816 0.0; H -0.4689 -0.1 0.8121; H -0.4689 0.7 -0.8121";

#[test]
fn integrals_match_pyscf() {
    // (atoms, basis, functions, fingerprints of S, T, V, (ij|kl), E_nuc)
    // from PySCF 2.14 (`lib.fp(mol.intor(...))`, `cart=True`), with the
    // geometry given in bohr (PySCF converts ångström with an older bohr).
    let cases: [(&str, Basis, usize, [f64; 5]); 5] = [
        (
            "H 0 0 0; H 0 0 0.735",
            Basis::Sto3g,
            2,
            [
                0.09234067410222924,
                0.03739762049027994,
                -0.1684902171638355,
                0.4547041025986293,
                0.7199689944258503,
            ],
        ),
        (
            "Li 0 0 0; H 0 0 1.595",
            Basis::SixThirtyOneG,
            11,
            [
                2.994700946305486,
                4.139410865927713,
                -10.821682067201497,
                7.069853179571066,
                0.9953176380620689,
            ],
        ),
        (
            H2O,
            Basis::Sto3g,
            7,
            [
                1.3598018385799495,
                29.183974516206543,
                -75.67223177172457,
                3.0232344197428724,
                9.189533762639684,
            ],
        ),
        (
            H2O,
            Basis::SixThirtyOneG,
            13,
            [
                0.22447320186859715,
                29.587024641562255,
                -72.81683514962143,
                4.424494716586154,
                9.189533762639684,
            ],
        ),
        (
            NH3,
            Basis::SixThirtyOneG,
            15,
            [
                2.5244055798721896,
                24.19577242899616,
                -51.728622677280256,
                -3.385940409187806,
                11.884381477287532,
            ],
        ),
    ];
    for (atoms, basis, n, want) in cases {
        let mol = Molecule::from_atoms(atoms, 0, 1).unwrap();
        let ints = AoIntegrals::compute(&mol, basis).unwrap();
        assert_eq!(ints.n_functions(), n, "{atoms}");
        assert!(ints.belongs_to(&mol) && ints.basis() == basis);
        let got = [
            fingerprint(ints.overlap.iter().copied()),
            fingerprint(ints.kinetic.iter().copied()),
            fingerprint(ints.nuclear.iter().copied()),
            fingerprint(ints.eri.iter().copied()),
            mol.nuclear_repulsion(),
        ];
        for (g, w) in got.iter().zip(&want) {
            assert!(
                (g - w).abs() < 1e-12 * w.abs().max(1.0),
                "{atoms} {basis:?}: {g} vs {w}"
            );
        }
        for i in 0..n {
            assert!((ints.overlap(i, i) - 1.0).abs() < 1e-12);
        }
    }
    // An integral of s functions against a 40-digit evaluation of its
    // closed form: (N 2s, H₂ 1s | N 1s, N 1s) in NH₃/6-31G.
    let mol = Molecule::from_atoms(NH3, 0, 1).unwrap();
    let ints = AoIntegrals::compute(&mol, Basis::SixThirtyOneG).unwrap();
    assert!((ints.eri(1, 11, 0, 0) - 0.320_765_938_400_474_3).abs() < 1e-15);
}

#[test]
fn boys_function_matches_its_series_and_asymptotics() {
    // F_0(T) = ½ √(π/T) erf(√T); erf from its series, F_n by quadrature.
    for &t in &[0.0, 1e-9, 0.3, 2.0, 11.5, 35.0, 59.9, 60.1, 120.0] {
        let mut f = [0.0; 9];
        boys(&mut f, t);
        for (n, &value) in f.iter().enumerate() {
            // Simpson quadrature of ∫₀¹ x^{2n} e^{−T x²} dx.
            let m = 20_000;
            let h = 1.0 / m as f64;
            let g = |x: f64| x.powi(2 * n as i32) * (-t * x * x).exp();
            let mut s = g(0.0) + g(1.0);
            for k in 1..m {
                s += if k % 2 == 1 { 4.0 } else { 2.0 } * g(k as f64 * h);
            }
            let want = s * h / 3.0;
            assert!(
                (value - want).abs() < 1e-12 * want.max(1e-300) + 1e-15,
                "F_{n}({t}) = {value} vs {want}"
            );
        }
    }
}

#[test]
fn molecules_are_parsed_and_checked() {
    let h2o = Molecule::from_atoms("o 0 0 0\nH 0 0.757 0.587; h 0 -0.757 0.587", 0, 1).unwrap();
    assert_eq!(h2o.charges(), &[8, 1, 1]);
    assert_eq!(h2o.n_electrons(), 10);
    assert_eq!(h2o.n_alpha_beta(), (5, 5));
    assert!((h2o.positions()[1][1] - 0.757 * BOHR_PER_ANGSTROM).abs() < 1e-15);
    let oh = Molecule::from_atoms("O 0 0 0; H 0 0 0.97", 0, 2).unwrap();
    assert_eq!(oh.n_alpha_beta(), (5, 4));
    let cation = Molecule::from_atoms("H 0 0 0; H 0 0 1", 1, 2).unwrap();
    assert_eq!(cation.n_alpha_beta(), (1, 0));
    assert!(Molecule::from_atoms("O 0 0 0; H 0 0 0.97", 0, 1).is_err());
    assert!(Molecule::from_atoms("H 0 0 0; H 0 0 1", 0, 0).is_err());
    assert!(Molecule::from_atoms("H 0 0 0; H 0 0 1", 3, 1).is_err());
    assert!(Molecule::from_atoms("K 0 0 0", 1, 1).is_err());
    assert!(Molecule::from_atoms("H 0 0", 0, 2).is_err());
    assert!(Molecule::from_atoms("H 0 0 x", 0, 2).is_err());
    assert!(Molecule::from_atoms("H 0 0 0; H 0 0 0", 0, 1).is_err());
    assert!(Molecule::from_atoms("", 0, 1).is_err());
    assert!(Molecule::from_atoms("H 0 0 inf", 0, 2).is_err());
    assert_eq!(Basis::from_name("STO-3G").unwrap(), Basis::Sto3g);
    assert_eq!(Basis::from_name("6-31g").unwrap(), Basis::SixThirtyOneG);
    assert_eq!(Basis::from_name("cc-pVDZ").unwrap(), Basis::CcPvdz);
    assert_eq!(
        Basis::from_name("6-31G(d)").unwrap(),
        Basis::SixThirtyOneGStar
    );
    for b in [
        Basis::Sto3g,
        Basis::SixThirtyOneG,
        Basis::SixThirtyOneGStar,
        Basis::CcPvdz,
    ] {
        assert_eq!(Basis::from_name(b.name()).unwrap(), b);
    }
    assert!(Basis::from_name("cc-pvtz").is_err());
}

#[test]
fn integrals_have_their_symmetries() {
    let mol = Molecule::from_atoms(NH3, 0, 1).unwrap();
    let ints = AoIntegrals::compute(&mol, Basis::Sto3g).unwrap();
    let n = ints.n_functions();
    assert_eq!(ints.atoms(), &[0, 0, 0, 0, 0, 1, 2, 3]);
    assert_eq!(ints.angular_momenta(), &[0, 0, 1, 1, 1, 0, 0, 0]);
    for i in 0..n {
        for j in 0..n {
            assert_eq!(ints.overlap(i, j), ints.overlap(j, i));
            assert_eq!(ints.core(i, j), ints.kinetic(i, j) + ints.nuclear(i, j));
            for k in 0..n {
                for l in 0..n {
                    let v = ints.eri(i, j, k, l);
                    assert_eq!(v, ints.eri(j, i, l, k));
                    assert_eq!(v, ints.eri(k, l, i, j));
                }
            }
        }
    }
    // Translation leaves every integral unchanged (to rounding).
    let shifted = Molecule::new(
        mol.charges().to_vec(),
        mol.positions()
            .iter()
            .map(|r| [r[0] + 3.1, r[1] - 1.7, r[2] + 0.4])
            .collect(),
        0,
        1,
    )
    .unwrap();
    let moved = AoIntegrals::compute(&shifted, Basis::Sto3g).unwrap();
    for (a, b) in ints.eri.iter().zip(&moved.eri) {
        assert!((a - b).abs() < 1e-11);
    }
    for (a, b) in ints.nuclear.iter().zip(&moved.nuclear) {
        assert!((a - b).abs() < 1e-11);
    }
}

#[test]
fn limits_and_deserialisation_are_checked() {
    // C₅H₁₂ in 6-31G has 69 basis functions, more than the maximum.
    let pentane = Molecule::new(
        [6; 5].into_iter().chain([1; 12]).collect(),
        (0..17)
            .map(|k| {
                [
                    1.4 * k as f64,
                    if k % 2 == 0 { 0.0 } else { 1.0 },
                    0.3 * k as f64,
                ]
            })
            .collect(),
        0,
        1,
    )
    .unwrap();
    let err = AoIntegrals::compute(&pentane, Basis::SixThirtyOneG).unwrap_err();
    assert!(err.to_string().contains("exceed the maximum 64"), "{err}");
    // Deserialisation validates like the constructor.
    let ok: Molecule = serde_json::from_str(
        r#"{"charges":[1,1],"positions":[[0,0,0],[0,0,1.4]],"charge":0,"multiplicity":1}"#,
    )
    .unwrap();
    assert_eq!(ok.n_alpha_beta(), (1, 1));
    for bad in [
        r#"{"charges":[19],"positions":[[0,0,0]],"charge":0,"multiplicity":2}"#,
        r#"{"charges":[1,1],"positions":[[0,0,0],[0,0,1.4]],"charge":0,"multiplicity":0}"#,
        r#"{"charges":[1,1],"positions":[[0,0,0],[0,0,0]],"charge":0,"multiplicity":1}"#,
    ] {
        assert!(serde_json::from_str::<Molecule>(bad).is_err(), "{bad}");
    }
}

#[test]
fn f_and_g_shells_match_pyscf() {
    // O with s, p, d, f and g shells, H with s and f, at (0.3, 0.4, 1.8)
    // bohr: PySCF 2.14's integrals (its Cartesian components rescaled to
    // norm one), summed in absolute value and at single entries.
    static O: [(f64, f64); 2] = [(5.0, 0.6), (1.0, 0.5)];
    static H: [(f64, f64); 2] = [(1.5, 0.5), (0.3, 0.6)];
    static P: [(f64, f64); 1] = [(2.0, 1.0)];
    static D: [(f64, f64); 1] = [(1.2, 1.0)];
    static F: [(f64, f64); 1] = [(0.9, 1.0)];
    static G: [(f64, f64); 1] = [(0.7, 1.0)];
    static HF: [(f64, f64); 1] = [(0.6, 1.0)];
    let shells = |z: u32| -> Vec<ShellData> {
        if z == 8 {
            vec![(0, &O), (1, &P), (2, &D), (3, &F), (4, &G)]
        } else {
            vec![(0, &H), (3, &HF)]
        }
    };
    let mol = Molecule::new(vec![8, 1], vec![[0.0; 3], [0.3, 0.4, 1.8]], -1, 1).unwrap();
    let sum = |n: usize, f: &dyn Fn(usize, usize) -> f64| -> f64 {
        (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| f(i, j).abs())
            .sum()
    };
    for (harmonics, n, s_sum, h_sum, eri_sum) in [
        (
            Harmonics::Spherical,
            33,
            59.786700494518165,
            234.69935673141197,
            3803.304395263438,
        ),
        (
            Harmonics::Cartesian,
            46,
            191.7366215078348,
            1008.1530870184897,
            32513.07582190359,
        ),
    ] {
        let ints = AoIntegrals::from_shells(&mol, Basis::Sto3g, harmonics, &shells).unwrap();
        assert_eq!(ints.n_functions(), n);
        assert!((sum(n, &|i, j| ints.overlap(i, j)) - s_sum).abs() < 1e-11 * s_sum);
        assert!((sum(n, &|i, j| ints.core(i, j)) - h_sum).abs() < 1e-11 * h_sum);
        let e: f64 = ints.eri_tensor().iter().map(|v| v.abs()).sum();
        assert!((e - eri_sum).abs() < 1e-11 * eri_sum, "{e} vs {eri_sum}");
        if harmonics == Harmonics::Spherical {
            for ((i, j, k, l), v) in [
                ((0, 5, 9, 30), 0.0010527254138130967),
                ((12, 13, 20, 25), 0.0012704485597235984),
                ((24, 27, 26, 31), -0.003320832750801069),
            ] {
                assert!((ints.eri(i, j, k, l) - v).abs() < 1e-13, "({i}{j}|{k}{l})");
            }
            assert!((ints.core(7, 22) - -0.013020531670640087).abs() < 1e-13);
        } else {
            assert!((ints.core(7, 22) - -0.002719710180330567).abs() < 1e-13);
        }
    }
}

#[test]
fn solid_harmonics_have_pyscfs_signs() {
    // PySCF's cart2sph(2) and (3), columns m = −l..l, scaled to unit
    // leading coefficient: same signs and ratios.
    let d = [
        (-2, vec![(1, 1.0)]),
        (0, vec![(0, -0.5), (3, -0.5), (5, 1.0)]),
        (2, vec![(0, 1.0), (3, -1.0)]),
    ];
    for (m, entries) in d {
        let c = solid_harmonic(2, m);
        let scale = entries[entries.len() - 1].1 / c[entries[entries.len() - 1].0];
        for (k, v) in entries {
            assert!((c[k] * scale - v).abs() < 1e-12, "d m = {m}");
        }
    }
    // f, m = 1: x(4z² − x² − y²) over xxx, xyy, xzz.
    let c = solid_harmonic(3, 1);
    let powers = cartesian_powers(3);
    let at = |p: [u32; 3]| c[powers.iter().position(|q| *q == p).unwrap()];
    assert!((at([1, 0, 2]) / at([3, 0, 0]) - -4.0).abs() < 1e-12);
    assert!((at([1, 2, 0]) / at([3, 0, 0]) - 1.0).abs() < 1e-12);
    assert!(at([1, 0, 2]) > 0.0);
    // f, m = −3: 3x²y − y³.
    let c = solid_harmonic(3, -3);
    let at = |p: [u32; 3]| c[powers.iter().position(|q| *q == p).unwrap()];
    assert!((at([2, 1, 0]) / at([0, 3, 0]) - -3.0).abs() < 1e-12);
    assert!(at([2, 1, 0]) > 0.0);
}
