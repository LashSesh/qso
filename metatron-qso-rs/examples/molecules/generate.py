"""Regenerate the FCIDUMP files and the reference energies in this folder.

Requires PySCF (only for this script; QSO does not depend on it):

    pip install pyscf==2.14.0
    python generate.py

Every molecule is computed in the STO-3G basis with restricted Hartree-Fock
orbitals and no frozen core. `reference.json` records the Hartree-Fock energy
and the lowest full-CI energies of the neutral molecule with Sz = 0 (the
determinant space QSO diagonalises), computed by PySCF's FCI solver.
"""

import json

import numpy as np
from pyscf import ao2mo, fci, gto, scf
from pyscf.tools import fcidump

MOLECULES = {
    "h2_sto3g": "H 0 0 0; H 0 0 0.735",
    "lih_sto3g": "Li 0 0 0; H 0 0 1.595",
    "beh2_sto3g": "Be 0 0 0; H 0 0 1.326; H 0 0 -1.326",
}
ROOTS = 6

reference = {}
for name, atoms in MOLECULES.items():
    mol = gto.M(atom=atoms, basis="sto-3g", unit="angstrom", symmetry=False)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    fcidump.from_scf(mf, f"{name}.fcidump", tol=1e-15)
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1e-12
    energies, _ = solver.kernel(h1, eri, norb, mol.nelec, nroots=ROOTS, ecore=mol.energy_nuc())
    reference[name] = {
        "geometry_angstrom": atoms,
        "spatial_orbitals": norb,
        "electrons": list(mol.nelec),
        "nuclear_repulsion": mol.energy_nuc(),
        "hartree_fock": mf.e_tot,
        "fci": [float(e) for e in np.atleast_1d(energies)],
    }

with open("reference.json", "w") as f:
    json.dump(reference, f, indent=2)
    f.write("\n")
