import numpy as np
import pytest

import metatron_qso as q

H2O = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"


def test_molecule_and_integrals():
    mol = q.Molecule(H2O)
    assert mol.charges == [8, 1, 1]
    assert mol.n_electrons == 10 and mol.n_alpha_beta == (5, 5)
    assert mol.positions.shape == (3, 3)
    # PySCF 2.14 on the same geometry in bohr.
    assert abs(mol.nuclear_repulsion() - 9.189533762639684) < 1e-12
    ints = q.AoIntegrals(mol, "sto-3g")
    n = ints.n_functions
    assert n == 7
    assert np.allclose(np.diag(ints.overlap), 1.0)
    assert np.allclose(ints.core, ints.kinetic + ints.nuclear)
    eri = ints.eri
    assert eri.shape == (n,) * 4
    assert np.allclose(eri, eri.transpose(1, 0, 3, 2))
    assert np.allclose(eri, eri.transpose(2, 3, 0, 1))
    assert ints.atoms == [0, 0, 0, 0, 0, 1, 2]
    assert ints.angular_momenta == [0, 0, 1, 1, 1, 0, 0]


def test_hartree_fock_and_full_ci():
    # PySCF 2.14 references.
    hf = q.hartree_fock(q.Molecule(H2O), "6-31g")
    assert hf.restricted
    assert abs(hf.energy - -75.98397447272153) < 1e-10
    assert abs(hf.s_squared()) < 1e-12
    c = hf.coefficients[0]
    s = hf.integrals.overlap
    assert np.allclose(c.T @ s @ c, np.eye(c.shape[1]), atol=1e-10)
    mo = hf.molecular_integrals(frozen=1, active=8)
    assert abs(mo.hartree_fock_energy() - hf.energy) < 1e-10
    assert abs(mo.fci_energies()[0] - -76.02472563261222) < 1e-9
    # The integrals are shared, and can be passed in again.
    assert hf.integrals is hf.integrals
    again = q.hartree_fock(q.Molecule(H2O), "6-31g", integrals=hf.integrals)
    assert again.energy == hf.energy and again.integrals is hf.integrals
    oh = q.hartree_fock(q.Molecule("O 0 0 0; H 0 0 0.97", multiplicity=2), "6-31g", unrestricted=True)
    assert not oh.restricted and len(oh.orbital_energies) == 2
    assert abs(oh.energy - -75.36316824957633) < 1e-10
    assert abs(oh.s_squared() - 0.7537742361406874) < 1e-7


def test_invalid_input():
    with pytest.raises(ValueError):
        q.Molecule("K 0 0 0")
    with pytest.raises(ValueError):
        q.Molecule("O 0 0 0; H 0 0 0.97")  # odd electrons, singlet
    with pytest.raises(ValueError):
        q.AoIntegrals(q.Molecule("H 0 0 0; H 0 0 0.7"), "cc-pvtz")
    oh = q.Molecule("O 0 0 0; H 0 0 0.97", multiplicity=2)
    with pytest.raises(ValueError):
        q.hartree_fock(oh)
    with pytest.raises(ValueError):
        q.hartree_fock(oh, unrestricted=True).molecular_integrals()
    with pytest.raises(ValueError):
        q.hartree_fock(q.Molecule(H2O), max_iterations=2)


def test_guesses_and_level_shift():
    n2 = q.Molecule("N 0 0 0; N 0 0 1.0977")
    ground = q.hartree_fock(n2).energy
    assert abs(ground - -107.4958933078439) < 1e-10
    assert q.hartree_fock(n2, guess="core").energy > ground + 0.5
    shifted = q.hartree_fock(n2, level_shift=0.3, diis_space=0)
    assert abs(shifted.energy - ground) < 1e-9
    with pytest.raises(ValueError):
        q.hartree_fock(n2, guess="huckel")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        q.hartree_fock(n2, integrals=q.AoIntegrals(n2, "6-31g"))
    with pytest.raises(ValueError):
        q.hartree_fock(q.Molecule("H 0 0 0; H 0 0 0.7"), integrals=q.AoIntegrals(n2))


def test_d_functions_and_open_shells() -> None:
    # PySCF 2.14 references, given the geometries in bohr as QSO converts
    # them (see the Rust tests).
    h2o = q.Molecule(H2O)
    assert abs(q.hartree_fock(h2o, "cc-pvdz").energy - -76.02677205339262) < 1e-8
    cart = q.hartree_fock(h2o, "6-31g*", harmonics="cartesian")
    assert abs(cart.energy - -76.01050498825595) < 1e-8
    assert q.AoIntegrals(h2o, "6-31g*").n_functions == 18
    assert q.AoIntegrals(h2o, "6-31g*", "cartesian").n_functions == 19
    o2 = q.Molecule("O 0 0 0; O 0 0 1.2075", multiplicity=3)
    rohf = q.hartree_fock(o2, "cc-pvdz", restricted_open=True)
    assert abs(rohf.energy - -149.60808446615613) < 1e-8
    casci = rohf.molecular_integrals(frozen=2, active=8).fci_energies(1)[0]
    assert abs(casci - -149.68777965956423) < 1e-8
    uhf = q.hartree_fock(o2, "cc-pvdz", unrestricted=True)
    assert uhf.energy < rohf.energy
    hcl = q.hartree_fock(q.Molecule("H 0 0 0; Cl 0 0 1.2746"), "cc-pvdz")
    assert abs(hcl.energy - -460.08944519174247) < 1e-8
    with pytest.raises(ValueError):
        q.hartree_fock(o2, unrestricted=True, restricted_open=True)
    with pytest.raises(ValueError):
        q.AoIntegrals(h2o, "6-31g*", "pure")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        q.hartree_fock(h2o, "6-31g*", integrals=q.AoIntegrals(h2o, "6-31g*", "cartesian"))
