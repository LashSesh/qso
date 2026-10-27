use super::*;
use crate::chemistry::integrals::Basis;

const H2: &str = "H 0 0 0; H 0 0 0.735";
const LIH: &str = "Li 0 0 0; H 0 0 1.595";
const BEH2: &str = "Be 0 0 0; H 0 0 1.326; H 0 0 -1.326";
const H2O: &str = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692";

fn setup(atoms: &str, basis: Basis, charge: i32, multiplicity: u32) -> (Molecule, AoIntegrals) {
    let mol = Molecule::from_atoms(atoms, charge, multiplicity).unwrap();
    let ints = AoIntegrals::compute(&mol, basis).unwrap();
    (mol, ints)
}

#[test]
fn restricted_energies_match_pyscf() {
    // PySCF 2.14, `scf.RHF(mol)` with conv_tol 1e-13.
    let cases = [
        (H2, Basis::Sto3g, -1.116998996754004),
        (H2, Basis::SixThirtyOneG, -1.126809358128038),
        (LIH, Basis::Sto3g, -7.8620238601271195),
        (LIH, Basis::SixThirtyOneG, -7.979268948423271),
        (BEH2, Basis::Sto3g, -15.560334935987303),
        (BEH2, Basis::SixThirtyOneG, -15.75932887792355),
        (H2O, Basis::Sto3g, -74.96302313846292),
        (H2O, Basis::SixThirtyOneG, -75.983974472722),
    ];
    for (atoms, basis, want) in cases {
        let (mol, ints) = setup(atoms, basis, 0, 1);
        let hf = rhf(&mol, &ints, &ScfOptions::default()).unwrap();
        assert!(
            (hf.energy - want).abs() < 1e-9,
            "{atoms} {basis:?}: {} vs {want}",
            hf.energy
        );
        assert!(hf.is_restricted());
        assert!(hf.s_squared().abs() < 1e-12);
        // Orbitals are orthonormal in the overlap metric.
        let c = &hf.coefficients[0];
        let ctsc = c.transpose() * &hf.overlap * c;
        for i in 0..c.ncols() {
            for j in 0..c.ncols() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((ctsc[(i, j)] - want).abs() < 1e-10);
            }
        }
        let eps = &hf.orbital_energies[0];
        assert!(eps.windows(2).all(|w| w[0] <= w[1]));
        // UHF on a closed shell converges to the same determinant.
        let u = uhf(&mol, &ints, &ScfOptions::default()).unwrap();
        assert!((u.energy - hf.energy).abs() < 1e-9);
    }
}

#[test]
fn unrestricted_energies_and_spin_contamination_match_pyscf() {
    // PySCF 2.14, `scf.UHF(mol)`: energy and ⟨S²⟩.
    let cases = [
        (
            "O 0 0 0; H 0 0 0.97",
            Basis::SixThirtyOneG,
            0,
            2,
            -75.36316824957647,
            0.7537742361406874,
        ),
        (
            "C 0 0 0.1; H 0 0.99 -0.5; H 0 -0.99 -0.5",
            Basis::SixThirtyOneG,
            0,
            3,
            -38.89981456758459,
            2.0173572133731907,
        ),
        (
            H2O,
            Basis::Sto3g,
            1,
            2,
            -74.65589871362081,
            0.7552020403737494,
        ),
        (
            "Li 0 0 0",
            Basis::SixThirtyOneG,
            0,
            2,
            -7.431235811083672,
            0.7500006790934602,
        ),
    ];
    for (atoms, basis, charge, mult, energy, s2) in cases {
        let (mol, ints) = setup(atoms, basis, charge, mult);
        let hf = uhf(&mol, &ints, &ScfOptions::default()).unwrap();
        assert!(
            (hf.energy - energy).abs() < 1e-9,
            "{atoms}: {} vs {energy}",
            hf.energy
        );
        assert!(
            (hf.s_squared() - s2).abs() < 1e-7,
            "{atoms}: {} vs {s2}",
            hf.s_squared()
        );
        assert!(!hf.is_restricted());
        assert!(rhf(&mol, &ints, &ScfOptions::default()).is_err());
        assert!(hf.molecular_integrals(&ints, 0, None).is_err());
    }
}

#[test]
fn molecular_orbital_hamiltonians_reproduce_hartree_fock_fci_and_casci() {
    // The FCIDUMP files in examples/molecules were written by PySCF from
    // the same RHF orbitals; their FCI energies are in reference.json.
    for (atoms, file, fci) in [
        (
            H2,
            include_str!("../../../examples/molecules/h2_sto3g.fcidump"),
            -1.1373060357534004,
        ),
        (
            LIH,
            include_str!("../../../examples/molecules/lih_sto3g.fcidump"),
            -7.882401932290221,
        ),
        (
            BEH2,
            include_str!("../../../examples/molecules/beh2_sto3g.fcidump"),
            -15.595182356661681,
        ),
    ] {
        let (mol, ints) = setup(atoms, Basis::Sto3g, 0, 1);
        let hf = rhf(&mol, &ints, &ScfOptions::default()).unwrap();
        let mo = hf.molecular_integrals(&ints, 0, None).unwrap();
        let reference = MolecularIntegrals::from_fcidump(file).unwrap();
        assert_eq!(mo.n_orbitals(), reference.n_orbitals());
        assert!((mo.hartree_fock_energy() - hf.energy).abs() < 1e-10);
        assert!((mo.hartree_fock_energy() - reference.hartree_fock_energy()).abs() < 1e-8);
        let e = mo.fci_energies(1).unwrap()[0];
        assert!((e - fci).abs() < 1e-8, "{atoms}: {e} vs {fci}");
        assert!((e - reference.fci_energies(1).unwrap()[0]).abs() < 1e-8);
    }
    // PySCF `fci.FCI(mf)` and `mcscf.CASCI(mf, n_active, n_electrons)`.
    let cases = [
        (LIH, Basis::SixThirtyOneG, 0, None, -7.998276133495167),
        (LIH, Basis::SixThirtyOneG, 1, Some(5), -7.9816802590176685),
        (H2O, Basis::Sto3g, 0, None, -75.01257824109207),
        (H2O, Basis::Sto3g, 1, Some(6), -75.01250015395725),
        (H2O, Basis::SixThirtyOneG, 1, Some(8), -76.02472563260908),
    ];
    for (atoms, basis, frozen, active, want) in cases {
        let (mol, ints) = setup(atoms, basis, 0, 1);
        let hf = rhf(&mol, &ints, &ScfOptions::default()).unwrap();
        let mo = hf.molecular_integrals(&ints, frozen, active).unwrap();
        // Freezing doubly occupied orbitals keeps the determinant energy.
        assert!((mo.hartree_fock_energy() - hf.energy).abs() < 1e-10);
        let e = mo.fci_energies(1).unwrap()[0];
        assert!(
            (e - want).abs() < 1e-8,
            "{atoms} {basis:?} {frozen} {active:?}: {e} vs {want}"
        );
    }
}

#[test]
fn invalid_inputs_are_rejected() {
    let (mol, ints) = setup(H2O, Basis::Sto3g, 0, 1);
    let hf = rhf(&mol, &ints, &ScfOptions::default()).unwrap();
    assert!(hf.molecular_integrals(&ints, 6, None).is_err());
    assert!(hf.molecular_integrals(&ints, 1, Some(3)).is_err());
    assert!(hf.molecular_integrals(&ints, 1, Some(7)).is_err());
    assert!(hf.molecular_integrals(&ints, 0, Some(0)).is_err());
    let bad = ScfOptions {
        energy_tolerance: 0.0,
        ..ScfOptions::default()
    };
    assert!(rhf(&mol, &ints, &bad).is_err());
    let few = ScfOptions {
        max_iterations: 2,
        ..ScfOptions::default()
    };
    assert!(matches!(
        rhf(&mol, &ints, &few),
        Err(QsoError::NoConvergence { .. })
    ));
    // Too many electrons of one spin for the basis.
    let (he, he_ints) = setup("He 0 0 0", Basis::Sto3g, 0, 3);
    assert!(uhf(&he, &he_ints, &ScfOptions::default()).is_err());
}
