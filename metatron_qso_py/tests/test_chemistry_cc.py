"""MP2, CCSD and CCSD(T) references (1.62)."""

import pytest

import metatron_qso as q

H2O = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"


def test_closed_shell_matches_pyscf() -> None:
    # PySCF 2.14: water in STO-3G, one frozen core orbital.
    hf = q.hartree_fock(q.Molecule(H2O), "sto-3g", energy_tolerance=1e-12, gradient_tolerance=1e-9)
    pt = hf.mp2(frozen=1)
    assert pt["correlation"] == pytest.approx(-0.03544594119987926, abs=1e-8)
    assert pt["energy"] == pytest.approx(hf.energy + pt["correlation"], abs=1e-12)
    assert pt["same_spin"] + pt["opposite_spin"] == pytest.approx(pt["correlation"], abs=1e-12)
    cc = hf.ccsd_t(frozen=1)
    assert not cc["ccsd"]["spin_orbitals"]
    assert cc["ccsd"]["correlation"] == pytest.approx(-0.04936022397391921, abs=1e-8)
    assert cc["triples"] == pytest.approx(-6.748571270771238e-05, abs=1e-8)
    assert cc["ccsd"]["t1_diagnostic"] == pytest.approx(0.00483735094111411, abs=1e-6)
    assert cc["energy"] == pytest.approx(cc["ccsd"]["energy"] + cc["triples"], abs=1e-12)
    assert hf.ccsd(frozen=1) == cc["ccsd"]


def test_open_shell_matches_pyscf() -> None:
    # PySCF 2.14: OH in 6-31G, UHF and ROHF.
    oh = q.Molecule("O 0 0 0; H 0 0 0.97", multiplicity=2)
    u = q.hartree_fock(oh, "6-31g", unrestricted=True, energy_tolerance=1e-12, gradient_tolerance=1e-9)
    assert u.mp2()["correlation"] == pytest.approx(-0.08918054448159576, abs=1e-8)
    cc = u.ccsd_t()
    assert cc["ccsd"]["spin_orbitals"]
    assert cc["ccsd"]["correlation"] == pytest.approx(-0.09882768680936423, abs=1e-8)
    assert cc["triples"] == pytest.approx(-0.0005574953546378659, abs=1e-8)
    ro = q.hartree_fock(oh, "6-31g", restricted_open=True, energy_tolerance=1e-12, gradient_tolerance=1e-9)
    assert ro.ccsd()["correlation"] == pytest.approx(-0.10014818757154079, abs=1e-8)
    # ROHF orbitals of an open shell are not canonical.
    with pytest.raises(ValueError):
        ro.mp2()
    with pytest.raises(ValueError):
        ro.ccsd_t()


def test_two_electrons_are_full_ci() -> None:
    hf = q.hartree_fock(q.Molecule("H 0 0 0; H 0 0 0.735"), "6-31g")
    fci = hf.molecular_integrals().fci_energies()[0]
    cc = hf.ccsd_t()
    assert cc["ccsd"]["energy"] == pytest.approx(fci, abs=1e-9)
    assert abs(cc["triples"]) < 1e-12


def test_bad_input() -> None:
    hf = q.hartree_fock(q.Molecule(H2O), "sto-3g")
    with pytest.raises(ValueError):
        hf.ccsd(frozen=6)
    with pytest.raises(ValueError):
        hf.ccsd(max_iterations=2)
    with pytest.raises(ValueError):
        hf.ccsd(diis_space=0, energy_tolerance=-1.0)
