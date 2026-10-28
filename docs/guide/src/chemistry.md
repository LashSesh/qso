# Electronic structure

QSO's chemistry chain starts from the molecular integrals. From them it
builds the Hamiltonian, maps it to qubits and prepares the Hartree–Fock
state. Full configuration interaction (CI) gives the exact reference
energy, and UCCSD, k-UpCCGSD and ADAPT-VQE give variational estimates.
The integrals come either from a molecule's geometry, through QSO's own
Gaussian integrals and Hartree–Fock, or from FCIDUMP files, which
quantum-chemistry codes such as PySCF, Psi4 or Molpro write.

## Fermionic operators

`FermionOperator` parses sums of products of ladder operators: `p^`
creates and `p` annihilates a fermion in mode `p`. `normal_ordered()`
applies the anticommutation relations and brings each product into a
canonical form, so equal operators compare equal:

```python
import metatron_qso as q

a, a_dag = q.FermionOperator("[0]"), q.FermionOperator("[0^]")
assert (a * a_dag + a_dag * a).normal_ordered() == q.FermionOperator("1 []")
hop = q.FermionOperator("-1 [0^ 1] - 1 [1^ 0]")
print(hop.to_qubit("jordan_wigner").to_pauli_sum())   # -0.5 X0 X1 - 0.5 Y0 Y1
```

## Mappings

The three mappings store the occupation vector `n` as qubits `b = A n
(mod 2)`:

| mapping | qubit j holds | weight of `a_j†` |
|---|---|---|
| `jordan_wigner` | `n_j` | up to n |
| `parity` | `n_0 ⊕ … ⊕ n_j` | up to n |
| `bravyi_kitaev` | the parity of a Fenwick-tree range ending at j | O(log n) |

Every transform `A` is a CNOT circuit. It therefore maps Pauli strings to
Pauli strings, and LiH has 631 terms under each mapping. Spin orbitals are
in block order: modes `0 … m−1` are spin up, modes `m … 2m−1` spin down.
Under the parity mapping, two qubits then hold the parity of the spin-up
electrons and the total parity. The **two-qubit reduction** replaces both
by their known values and removes them.

## Molecules

`MolecularIntegrals.from_fcidump` (or `from_file`) reads restricted
integrals. The integrals below are those of H₂ at 0.735 Å in STO-3G. The
folder `metatron-qso-rs/examples/molecules/` contains H₂, LiH and BeH₂,
the script that produced them with PySCF, and PySCF's reference energies.

```python
import metatron_qso as q

H2 = """ &FCI NORB=2,NELEC=2,MS2=0,
 ORBSYM=1,1, ISYM=1,
 &END
 0.6757101548035163 1 1 1 1
 0.6645817302552969 1 1 2 2
 0.1809311997842314 2 1 2 1
 0.6645817302552965 2 2 1 1
 0.6985737227320176 2 2 2 2
 -1.25633907300325 1 1 0 0
 -0.4718960072811418 2 2 0 0
 0.7199689944489797 0 0 0 0
"""
mol = q.MolecularIntegrals.from_fcidump(H2)
h = mol.qubit_hamiltonian("parity", two_qubit_reduction=True)   # 2 qubits
hf = mol.hartree_fock_circuit("parity", two_qubit_reduction=True)
print(mol.hartree_fock_energy(), mol.fci_energies(4))
assert abs(h.expectation(q.simulate(hf)) - mol.hartree_fock_energy()) < 1e-12
assert abs(h.ground_state_energy() - mol.fci_energies()[0]) < 1e-10
```

`fci_energies(k)` returns the lowest `k` levels of the Hamiltonian among
all determinants with the given numbers of spin-up and spin-down
electrons. Up to 1024 determinants it diagonalises the matrix of the
Slater–Condon rules. Above that, up to 2¹⁸ determinants and 32 levels, it
uses Davidson's method on `H c` computed from the integrals without the
matrix (the string-driven algorithm of Knowles and Handy). Water in 6-31G
with a frozen core in 12 orbitals (245 025 determinants) takes 36 s and
0.7 GB for the ground state.

The tests check the whole chain against PySCF for H₂, LiH and BeH₂ in
STO-3G:

- Hartree–Fock energies agree to 10⁻⁹ Ha.
- The determinant-space spectra agree under Jordan–Wigner, parity,
  Bravyi–Kitaev and the reduced parity encoding.
- Their lowest roots equal PySCF's full-CI energies to 10⁻⁸ Ha.

## Molecules from their geometry

`Molecule` takes the atoms in ångström, the total charge and the spin
multiplicity. `AoIntegrals` evaluates the overlap, kinetic,
nuclear-attraction and electron-repulsion integrals over contracted
Gaussians by the McMurchie–Davidson scheme, for shells up to g:

- **Basis sets.** STO-3G, 6-31G, 6-31G* (d functions on Li to Ar) and
  cc-pVDZ (p functions on H and He, d on Li to Ar), for hydrogen to argon.
  The data come from PySCF's copy of the Basis Set Exchange
  (`scripts/generate_basis_data.py`).
- **Limits.** At most 64 basis functions (and 80 Cartesian components
  behind them).
- **Order.** Functions come in PySCF's order.
- **Harmonics.** d and higher shells are spherical by default: real
  solid harmonics, `m = −l, …, l`, with PySCF's signs. With
  `harmonics="cartesian"` they are the monomials `xx, xy, xz, yy, yz, zz`,
  each normalised to one; PySCF leaves its Cartesian d components with
  norms 4π/5 and 4π/15, so its integrals differ from QSO's by these
  factors.

`hartree_fock` solves the restricted (closed-shell), unrestricted
(`unrestricted=True`) or restricted open-shell (`restricted_open=True`)
Hartree–Fock equations with DIIS. ROHF diagonalises Roothaan's
effective Fock matrix with PySCF's blocks: ½(F_α + F_β) within the
closed, open and virtual spaces and between closed and virtual orbitals,
F_β between closed and open, F_α between open and virtual. Its energy
does not depend on this choice; its orbital energies do. The iterations start from
a superposition of atomic densities: every atom's spherically averaged
Hartree–Fock density in the same basis, like PySCF's `atom` guess.
Hartree–Fock finds a stationary point near its start, not always the
lowest one. Starting from the core Hamiltonian (`guess="core"`) ends
0.73 Ha above the ground state of N₂ in STO-3G. When DIIS does not
converge, a second run without DIIS shifts the virtual orbitals by
0.5 Ha (`level_shift`) while the gradient is large; this settles CN, where
PySCF does not converge. A calculation that still does not converge
raises an error. Bases with nearly dependent functions (atoms closer than
0.01 Å) leave rounding noise above the default gradient tolerance.
`integrals=` reuses integrals computed before.

`molecular_integrals(frozen, active)` transforms the integrals to the
restricted orbitals. The lowest `frozen` orbitals stay doubly occupied and
move into the core energy; `active` orbitals are kept. The result is a
`MolecularIntegrals`, so everything above applies, and `to_fcidump()`
writes it. For an open shell this needs the restricted orbitals of ROHF
(UHF's two sets of orbitals give no single Hamiltonian):

```python
import metatron_qso as q

water = q.Molecule("O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692")
hf = q.hartree_fock(water, "6-31g")
mo = hf.molecular_integrals(frozen=1, active=8)        # 16 qubits
print(hf.energy, mo.fci_energies()[0])                 # −75.98397, −76.02473
oh = q.hartree_fock(q.Molecule("O 0 0 0; H 0 0 0.97", multiplicity=2),
                    "6-31g", unrestricted=True)
print(oh.energy, oh.s_squared())                       # −75.36317, 0.75377
o2 = q.hartree_fock(q.Molecule("O 0 0 0; O 0 0 1.2075", multiplicity=3),
                    "cc-pvdz", restricted_open=True)
cas = o2.molecular_integrals(frozen=2, active=8)        # 12 electrons in 8 orbitals
print(o2.energy, cas.fci_energies()[0])                # −149.60808, −149.68778
```

The tests compare with PySCF 2.14:

- On the same geometries (in bohr), fingerprints of all integrals of H₂,
  LiH, H₂O and NH₃ agree to 10⁻¹² relative, and in the cross-validation
  every integral of H₂O, NH₃, CO and F₂ agrees to 1.4·10⁻¹³.
- Restricted energies of H₂, LiH, BeH₂, H₂O and N₂ in both bases, and
  unrestricted energies of OH, triplet CH₂, H₂O⁺, Li, NH₂ and NO, agree to
  10⁻¹⁰ Ha, ⟨S²⟩ to 10⁻⁷.
- Full CI and frozen-core active spaces (CASCI) agree to 10⁻⁸ Ha, and the
  FCIDUMP files of H₂, LiH and BeH₂ are reproduced from their geometries.
- With d functions and through the third row, RHF of H₂O (cc-pVDZ, and
  6-31G* with Cartesian shells), NH₃ (6-31G*), HCl and H₂S (cc-pVDZ), UHF
  and ROHF of triplet O₂ (cc-pVDZ), and ROHF of NH₂ (6-31G*) agree to
  10⁻⁹ Ha. CASCI in the ROHF orbitals of O₂ (12 electrons in 8 orbitals)
  and NH₂ agrees to 10⁻⁹ Ha.
- Integrals of f and g shells agree with PySCF (sums over every
  integral to 10⁻¹¹ relative, single integrals to 10⁻¹³), spherical and
  Cartesian.
- The cross-validation compares every integral of H₂O, NH₃, CO and F₂ in
  all four bases, of HCl, H₂S, SiH₄ and NaF, and of two molecules with
  Cartesian shells: the largest difference is 2.6·10⁻¹³. RHF, UHF, ROHF
  and CASCI energies, ROHF ones up to 12 active electrons, agree to
  2.4·10⁻¹⁰ Ha.

PySCF converts ångström with the CODATA 2010 bohr, QSO with CODATA 2018:
the same input in ångström gives geometries 3·10⁻¹¹ apart (relative) and
energies about 10⁻¹⁰ Ha apart, so the comparisons give PySCF QSO's
positions in bohr.

## UCCSD, k-UpCCGSD and ADAPT-VQE

An excitation `T` moves electrons between spin orbitals, and
`exp(θ(T − T†))` is unitary. Mapped to qubits, `T − T†` is a sum of Pauli
strings that commute with each other. The circuit therefore applies one
Pauli rotation per string, all sharing the parameter θ, and the product is
exact. All three ansätze start from the Hartree–Fock state, so θ = 0 gives
the Hartree–Fock energy.

- `uccsd` uses every spin-conserving single and double excitation from
  occupied to virtual orbitals.
- `k_upccgsd` stacks `k` layers of generalised pair doubles and
  spin-adapted singles between all orbital pairs.
- `adapt_vqe` grows the circuit from a pool of excitations. Each step adds
  the operator with the largest energy gradient `2 Re⟨Hψ|τψ⟩`, then
  optimises all parameters again.

```python
import numpy as np
import metatron_qso as q

H2 = """ &FCI NORB=2,NELEC=2,MS2=0, &END
 0.6757101548035163 1 1 1 1
 0.6645817302552969 1 1 2 2
 0.1809311997842314 2 1 2 1
 0.6645817302552965 2 2 1 1
 0.6985737227320176 2 2 2 2
 -1.25633907300325 1 1 0 0
 -0.4718960072811418 2 2 0 0
 0.7199689944489797 0 0 0 0
"""
mol = q.MolecularIntegrals.from_fcidump(H2)
fci = mol.fci_energies()[0]
ansatz = q.uccsd(mol, "bravyi_kitaev")
h = mol.qubit_hamiltonian("bravyi_kitaev")
vqe = q.run_vqe(h, ansatz, initial_parameters=np.zeros(ansatz.num_parameters))
adapt = q.adapt_vqe(mol, "parity", two_qubit_reduction=True)
print(vqe["energy"] - fci, adapt["operators"], adapt["energy"] - fci)
```

The example `molecules` does the same for LiH. It uses 10 qubits (parity
mapping with the two-qubit reduction). The output below is measured; the
full-CI energy is −7.882401932 Ha:

| method | parameters | error vs full CI (Ha) |
|---|---:|---:|
| Hartree–Fock | 0 | 2.04·10⁻² |
| UCCSD | 92 | 1.06·10⁻⁵ |
| 1-UpCCGSD | 30 | 1.24·10⁻⁴ |
| ADAPT-VQE | 30 | 1.10·10⁻⁵ |

## In Rust and on the command line

The same functionality is in `metatron_qso::chemistry`:

```rust
use metatron_qso::algorithms::Vqe;
use metatron_qso::chemistry::{Mapping, MolecularIntegrals, QubitEncoding, uccsd};

fn main() -> metatron_qso::Result<()> {
    let text = " &FCI NORB=2,NELEC=2,MS2=0, &END
 0.6757101548035163 1 1 1 1
 0.6645817302552969 1 1 2 2
 0.1809311997842314 2 1 2 1
 0.6645817302552965 2 2 1 1
 0.6985737227320176 2 2 2 2
 -1.25633907300325 1 1 0 0
 -0.4718960072811418 2 2 0 0
 0.7199689944489797 0 0 0 0
";
    let mol = MolecularIntegrals::from_fcidump(text)?;
    let encoding = QubitEncoding::new(Mapping::JordanWigner);
    let ansatz = uccsd(&mol, encoding)?;
    let p = ansatz.num_parameters();
    let r = Vqe::new(mol.qubit_hamiltonian(encoding)?, ansatz)?
        .with_initial_parameters(vec![0.0; p])?
        .run()?;
    assert!((r.energy - mol.fci_energies(1)?[0]).abs() < 1e-8);
    Ok(())
}
```

```text
$ qso molecule examples/molecules/lih_sto3g.fcidump --mapping parity --two-qubit-reduction --roots 2
6 orbitals, 2 + 2 electrons; 10 qubits (parity, two-qubit reduction), 631 Pauli terms
Hartree–Fock   -7.8620238601
full CI        -7.8824019323  -7.7664184751
UCCSD          -7.8823912923  (92 parameters, error vs full CI 1.06e-5)
```

`qso scf` runs Hartree–Fock from a geometry and, with `--fci`, full CI in
an active space; `--fcidump` writes the active-space Hamiltonian for
`qso molecule`:

```text
$ qso scf --atoms "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692" --basis 6-31g --frozen 1 --active 8 --fci
13 basis functions, 5 + 5 electrons
RHF   -75.9839744727  (13 iterations, <S²> = 0.000000)
full CI (8 orbitals, 4 + 4 electrons)   -76.0247256325  (correlation -0.0407511598)
```
