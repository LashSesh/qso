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
        q.Molecule("Na 0 0 0")
    with pytest.raises(ValueError):
        q.Molecule("O 0 0 0; H 0 0 0.97")  # odd electrons, singlet
    with pytest.raises(ValueError):
        q.AoIntegrals(q.Molecule("H 0 0 0; H 0 0 0.7"), "cc-pvdz")
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
