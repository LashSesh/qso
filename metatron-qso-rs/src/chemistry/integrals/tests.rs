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
    assert!(Molecule::from_atoms("Na 0 0 0", 1, 1).is_err());
    assert!(Molecule::from_atoms("H 0 0", 0, 2).is_err());
    assert!(Molecule::from_atoms("H 0 0 x", 0, 2).is_err());
    assert!(Molecule::from_atoms("H 0 0 0; H 0 0 0", 0, 1).is_err());
    assert!(Molecule::from_atoms("", 0, 1).is_err());
    assert!(Molecule::from_atoms("H 0 0 inf", 0, 2).is_err());
    assert_eq!(Basis::from_name("STO-3G").unwrap(), Basis::Sto3g);
    assert_eq!(Basis::from_name("6-31g").unwrap(), Basis::SixThirtyOneG);
    assert!(Basis::from_name("cc-pvdz").is_err());
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
        r#"{"charges":[11],"positions":[[0,0,0]],"charge":0,"multiplicity":2}"#,
        r#"{"charges":[1,1],"positions":[[0,0,0],[0,0,1.4]],"charge":0,"multiplicity":0}"#,
        r#"{"charges":[1,1],"positions":[[0,0,0],[0,0,0]],"charge":0,"multiplicity":1}"#,
    ] {
        assert!(serde_json::from_str::<Molecule>(bad).is_err(), "{bad}");
    }
}
