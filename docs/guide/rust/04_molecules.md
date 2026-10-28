# Molecules

`metatron-qso-rs/examples/molecules.rs` reads molecular integrals from
FCIDUMP files, maps them to qubits in several ways, and compares
variational ansätze, excited-state methods and product formulas with full
configuration interaction. The FCIDUMP files and the reference energies in
`examples/molecules/` were written by PySCF; QSO can also compute the
integrals itself (see [Electronic structure](../chemistry.md)).

## Qubit encodings

`MolecularIntegrals::from_fcidump` reads the one- and two-electron
integrals of LiH in STO-3G (6 spatial orbitals, 12 spin orbitals, two
electrons of each spin). Jordan–Wigner, parity and Bravyi–Kitaev give
different Hamiltonians on 12 qubits; the parity mapping with the
two-qubit reduction removes two qubits by the conservation of the
numbers of up and down electrons. `fci_energies_with` diagonalises each
qubit Hamiltonian in the span of the encoded determinants: all four
encodings must give the same lowest energy, which checks the mappings.

{{example molecules.rs:encodings}}

## Ansätze

Starting from Hartree–Fock (all parameters zero), VQE with UCCSD, with
one layer of k-UpCCGSD, and ADAPT-VQE, which grows its circuit one
UCCSD excitation at a time, chosen by the largest energy gradient. The
table gives each energy's error against full CI.

{{example molecules.rs:ansatze}}

## Excited states

Variational quantum deflation (VQD) finds the lowest four states of H₂
on two qubits one after the other, each optimisation penalising overlap
with the states already found.

{{example molecules.rs:excited}}

## Time evolution

`trotter_circuit` approximates exp(−iHt) with product formulas of order
1, 2 and 4. The program applies each circuit to the four basis states,
compares the columns with the exact unitary (`expm_hermitian` of the
dense matrix, identity term removed since it only adds a global phase)
and prints the largest column error next to the commutator bound of
`trotter_error_bound`, which must be above it.

{{example molecules.rs:dynamics}}

## Output

{{output molecules}}
