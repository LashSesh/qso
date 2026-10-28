use super::*;
use crate::chemistry::integrals::Basis;

const H2: &str = "H 0 0 0; H 0 0 0.735";
const LIH: &str = "Li 0 0 0; H 0 0 1.595";
const BEH2: &str = "Be 0 0 0; H 0 0 1.326; H 0 0 -1.326";
const H2O: &str = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692";
const N2: &str = "N 0 0 0; N 0 0 1.0977";

fn setup(atoms: &str, basis: Basis, charge: i32, multiplicity: u32) -> (Molecule, AoIntegrals) {
    let mol = Molecule::from_atoms(atoms, charge, multiplicity).unwrap();
    let ints = AoIntegrals::compute(&mol, basis).unwrap();
    (mol, ints)
}

#[test]
fn restricted_energies_match_pyscf() {
    // PySCF 2.14, `scf.RHF(mol)` with conv_tol 1e-13, geometries in bohr.
    // N₂: the core-Hamiltonian guess ends 0.73 Ha (STO-3G) above.
    let cases = [
        (N2, Basis::Sto3g, -107.4958933078439),
        (N2, Basis::SixThirtyOneG, -108.86776337590582),
        (H2, Basis::Sto3g, -1.116998996752976),
        (H2, Basis::SixThirtyOneG, -1.1268093581278664),
        (LIH, Basis::Sto3g, -7.862023860125529),
        (LIH, Basis::SixThirtyOneG, -7.97926894842384),
        (BEH2, Basis::Sto3g, -15.560334935984908),
        (BEH2, Basis::SixThirtyOneG, -15.759328877923995),
        (H2O, Basis::Sto3g, -74.9630231384672),
        (H2O, Basis::SixThirtyOneG, -75.98397447272153),
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
            -75.36316824957633,
            0.7537742361413269,
        ),
        (
            "C 0 0 0.1; H 0 0.99 -0.5; H 0 -0.99 -0.5",
            Basis::SixThirtyOneG,
            0,
            3,
            -38.89981456757777,
            2.017357213374883,
        ),
        (
            H2O,
            Basis::Sto3g,
            1,
            2,
            -74.65589871363201,
            0.7552020403744817,
        ),
        (
            "Li 0 0 0",
            Basis::SixThirtyOneG,
            0,
            2,
            -7.4312358110836705,
            0.7500006790934605,
        ),
        // NH₂ (the core guess ends 0.1 Ha above) and NO (DIIS from the
        // core guess did not converge).
        (
            "N 0 0 0; H 0 0.8 0.6; H 0 -0.8 0.6",
            Basis::Sto3g,
            0,
            2,
            -54.8319833767935,
            0.7565683102942997,
        ),
        (
            "N 0 0 0; O 0 0 1.151",
            Basis::Sto3g,
            0,
            2,
            -127.53046550455346,
            0.965952258993954,
        ),
    ];
    // ⟨S²⟩ follows the orbitals, which converge more slowly than the
    // energy (PySCF with conv_tol_grad 1e-10 for NO).
    let tight = ScfOptions {
        gradient_tolerance: 1e-10,
        ..ScfOptions::default()
    };
    for (atoms, basis, charge, mult, energy, s2) in cases {
        let (mol, ints) = setup(atoms, basis, charge, mult);
        let hf = uhf(&mol, &ints, &tight).unwrap();
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
            -1.1373060357533946,
        ),
        (
            LIH,
            include_str!("../../../examples/molecules/lih_sto3g.fcidump"),
            -7.882401932289461,
        ),
        (
            BEH2,
            include_str!("../../../examples/molecules/beh2_sto3g.fcidump"),
            -15.595182356661152,
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
        (LIH, Basis::SixThirtyOneG, 0, None, -7.998276133496047),
        (LIH, Basis::SixThirtyOneG, 1, Some(5), -7.981680259018377),
        (H2O, Basis::Sto3g, 0, None, -75.01257824110033),
        (H2O, Basis::Sto3g, 1, Some(6), -75.01250015396552),
        (H2O, Basis::SixThirtyOneG, 1, Some(8), -76.02472563261222),
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

#[test]
fn guesses_and_mismatched_integrals() {
    // The core-Hamiltonian guess converges to an excited N₂ determinant;
    // the atomic guess to the ground state.
    let (mol, ints) = setup(N2, Basis::Sto3g, 0, 1);
    let core = ScfOptions {
        guess: Guess::Core,
        ..ScfOptions::default()
    };
    let excited = rhf(&mol, &ints, &core).unwrap().energy;
    let ground = rhf(&mol, &ints, &ScfOptions::default()).unwrap().energy;
    assert!(excited > ground + 0.5, "{excited} {ground}");
    // Integrals of another molecule, or of another calculation.
    let (h2, h2_ints) = setup(H2, Basis::Sto3g, 0, 1);
    assert!(rhf(&mol, &h2_ints, &ScfOptions::default()).is_err());
    let hf = rhf(&h2, &h2_ints, &ScfOptions::default()).unwrap();
    assert!(hf.molecular_integrals(&ints, 0, None).is_err());
    let (_, h2_631) = setup(H2, Basis::SixThirtyOneG, 0, 1);
    assert!(hf.molecular_integrals(&h2_631, 0, None).is_err());
    // Orbital energies belong to the converged Fock matrix: F C = S C ε.
    let hf = rhf(&mol, &ints, &ScfOptions::default()).unwrap();
    let n = ints.n_functions();
    let s = RMatrix::from_fn(n, n, |i, j| ints.overlap(i, j));
    let c = &hf.coefficients[0];
    let occ = density(c, 7) * 2.0;
    let d = [&occ * 0.5, &occ * 0.5];
    let h = RMatrix::from_fn(n, n, |i, j| ints.core(i, j));
    let f = fock_matrices(&ints, &h, &d, true).remove(0);
    let residual = &f * c
        - &s * c
            * RMatrix::from_diagonal(&nalgebra::DVector::from_vec(hf.orbital_energies[0].clone()));
    assert!(residual.amax() < 1e-7, "{}", residual.amax());
}

#[test]
fn oscillating_open_shells_converge_with_a_level_shift() {
    // CN in STO-3G: DIIS oscillates (PySCF does not converge from its
    // guesses either); the level-shifted fallback reaches a stationary
    // point below every PySCF iterate (−90.989 Ha).
    let (mol, ints) = setup("C 0 0 0; N 0 0 1.17", Basis::Sto3g, 0, 2);
    let hf = uhf(&mol, &ints, &ScfOptions::default()).unwrap();
    assert!(
        (hf.energy - -91.02103183743569).abs() < 1e-8,
        "{}",
        hf.energy
    );
    let plain = ScfOptions {
        level_shift: 0.0,
        diis_space: 8,
        max_iterations: 300,
        ..ScfOptions::default()
    };
    assert!(scf(&mol, &ints, &plain, Kind::Unrestricted).is_err());
    let bad = ScfOptions {
        level_shift: -1.0,
        ..ScfOptions::default()
    };
    assert!(uhf(&mol, &ints, &bad).is_err());
}
