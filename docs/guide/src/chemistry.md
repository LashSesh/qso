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
- **Limits.** At most 64 basis functions. The repulsion integrals are
  computed per quartet of shells, the columns of a general contraction
  together; 64 functions with three chlorines in cc-pVDZ take about 3 s
  on four cores and 0.35 GB (`benchmark_report`).
- **Screening.** A quartet of shells is skipped when the Schwarz bound
  `√((ab|ab)(cd|cd))` of all its integrals is below `screening` (10⁻¹²
  Hartree by default; `screening=0` computes everything).
  `screened_quartets` counts the quartets skipped and computed. For a
  chain of 32 hydrogens 1 Å apart in 6-31G (64 functions) screening skips
  86 % of the quartets and cuts the time from 1.5 s to 0.2 s; the RHF
  energy moves by 5·10⁻¹³ Ha and still agrees with PySCF to 2·10⁻¹¹ Ha.
  Compact molecules gain less: butane in 6-31G, 5 % of the quartets.
  Every skipped Cartesian integral is below the threshold; spherical
  functions combine Cartesian ones and stay within a small multiple.
- **Order.** Functions come in PySCF's order.
- **Harmonics.** d and higher shells are spherical by default: real
  solid harmonics, `m = −l, …, l`, with PySCF's signs. With
  `harmonics="cartesian"` they are the monomials `xx, xy, xz, yy, yz, zz`,
  each normalised to one; PySCF leaves its Cartesian d components with
  squared norms 4π/5 (xx) and 4π/15 (xy), so its integrals differ from
  QSO's by the square roots of these factors, one per index. 6-31G* is
  defined, and run by Gaussian by default, with Cartesian d shells: use
  `harmonics="cartesian"` to compare with such results.

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
- Integrals of f and g shells (a custom basis on OH⁻) agree with PySCF:
  its fingerprints of the overlap, core Hamiltonian and repulsion
  integrals (Σ_k a_k cos k, so signs and order count) to 10⁻¹⁰, sums of
  absolute values to 10⁻¹¹ relative, single integrals to 10⁻¹³, spherical
  and Cartesian.
- ROHF of ClO, PH₂ and triplet SO (Cartesian d shells for the last two)
  agrees to 10⁻⁹ Ha, also with a level shift; ROHF of a closed shell is
  RHF.
- The cross-validation compares every integral of H₂O, NH₃, CO and F₂ in
  all four bases, of HCl, H₂S, SiH₄ and NaF, and of two molecules with
  Cartesian shells: the largest difference is 2.6·10⁻¹³. RHF, UHF, ROHF
  and CASCI energies, ROHF ones up to 12 active electrons, agree to
  2.4·10⁻¹⁰ Ha.

PySCF converts ångström with the CODATA 2010 bohr, QSO with CODATA 2018:
the same input in ångström gives geometries 3·10⁻¹¹ apart (relative) and
energies about 10⁻¹⁰ Ha apart, so the comparisons give PySCF QSO's
positions in bohr.

## Correlation beyond Hartree–Fock

Full CI stops at 32 active orbitals and 2¹⁸ determinants. For whole
molecules a `HartreeFock` gives three references, each with the lowest
`frozen` orbitals kept doubly occupied (a frozen core):

- **`mp2(frozen)`**: second-order Møller–Plesset energies in the
  canonical orbitals of RHF or UHF, with their same-spin and
  opposite-spin parts. ROHF of an open shell has no canonical orbitals
  and is refused.
- **`ccsd(frozen)`**: coupled-cluster singles and doubles. A closed shell
  in restricted orbitals is solved in spatial orbitals; UHF and ROHF of
  open shells in spin orbitals, with the equations of Stanton, Gauss,
  Watts and Bartlett, which take the whole Fock matrix and so hold for
  ROHF's orbitals too. The amplitudes start from MP2's and are iterated
  with DIIS (`diis_space` amplitude vectors) until the energy changes by
  less than `energy_tolerance` (10⁻¹⁰ Ha) and the amplitudes by less than
  `amplitude_tolerance` (10⁻⁸, the norm of the step). The T1 diagnostic
  of Lee and Taylor, `√(Σ (t_i^a)² / N)` over `N` correlated electrons,
  flags references that CCSD describes poorly (above about 0.02).
- **`ccsd_t(frozen)`**: CCSD and the perturbative triples (T) of
  Raghavachari et al., for canonical references: RHF of a closed shell or
  UHF. ROHF of an open shell would need semicanonical orbitals and is
  refused.

```python
import metatron_qso as q

water = q.Molecule("O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692")
hf = q.hartree_fock(water, "cc-pvdz")
pt = hf.mp2(frozen=1)
cc = hf.ccsd_t(frozen=1)
print(pt["correlation"], pt["same_spin"])              # −0.20167, −0.05072
print(cc["ccsd"]["correlation"], cc["triples"])        # −0.21123, −0.00304
print(cc["energy"], cc["ccsd"]["t1_diagnostic"])       # −76.24104, 0.0059
oh = q.Molecule("O 0 0 0; H 0 0 0.97", multiplicity=2)
uhf = q.hartree_fock(oh, "6-31g", unrestricted=True)
print(uhf.ccsd_t()["triples"])                         # −0.000557, in spin orbitals
rohf = q.hartree_fock(oh, "6-31g", restricted_open=True)
print(rohf.ccsd()["correlation"])                      # −0.10015
```

The integrals are transformed to the molecular orbitals in full by four
quarter transformations, each a matrix product. A closed shell keeps
`m⁴` of them for `m` correlated orbitals (64 functions: 134 MB). Spin
orbitals keep their whole antisymmetrised tensor and at most 100 of them
are correlated (`MAX_SPIN_ORBITALS`; 50 spatial orbitals). Everything runs
on one thread.

The tests and the cross-validation compare with PySCF 2.14 (`mp.MP2`,
`cc.CCSD`, `ccsd_t` and `uccsd_t`):

- MP2, CCSD and (T) of water, LiH and N₂ in STO-3G, 6-31G and cc-pVDZ,
  with and without frozen cores, UHF MP2, CCSD and (T) and ROHF CCSD of
  OH (6-31G, cc-pVDZ), NH₂ (6-31G*) and triplet O₂ (cc-pVDZ, two frozen
  orbitals) agree to 2.3·10⁻¹⁰ Ha, the T1 diagnostics of the closed
  shells to 3.4·10⁻⁹ (the tests hold them to 10⁻⁸).
- The cross-validation adds water (6-31G*), NH₃ and CO (cc-pVDZ), HF
  (6-31G*), HCl (6-31G, five frozen orbitals) and, in UHF and ROHF,
  triplet CH₂ and NO (6-31G): the largest difference is 2.1·10⁻¹⁰ Ha.
  MP2 depends on the orbitals at first order: PySCF's UHF of NO stops
  unconverged after its default 50 cycles, and its MP2 is then 7·10⁻⁹ Ha
  off, so the comparison lets it run to 10⁻¹⁰ in the gradient.
- CCSD is full CI for two electrons (H₂ in 6-31G and cc-pVDZ, HeH⁺) to
  10⁻⁹ Ha, from RHF and UHF, and (T) vanishes there.
- Two N₂ molecules 100 Å apart (STO-3G) have twice the correlation
  energy of one to 10⁻⁹ Ha in MP2 and CCSD and to 10⁻¹⁰ in (T) (size
  consistency).
- The spatial and spin-orbital equations give the same CCSD and (T) for
  closed shells, and UHF of a closed shell gives RHF's.
- The first CCSD energy, from the MP2 amplitudes, is the MP2 energy.

`qso scf` takes `--mp2`, `--ccsd` and `--ccsd-t`, with `--frozen`:

```text
$ qso scf --atoms "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692" --basis cc-pvdz --frozen 1 --mp2 --ccsd-t
24 basis functions, 5 + 5 electrons
RHF   -76.0267720534  (12 iterations, <S²> = 0.000000)
MP2   -76.2284380332  (correlation -0.2016659798: same spin -0.0507168938, opposite spin -0.1509490859)
CCSD  -76.2380047126  (correlation -0.2112326592, 13 iterations, T1 diagnostic 0.0059)
CCSD(T)   -76.2410412034  (triples -0.0030364908)
```

In Rust the functions take the integrals beside the Hartree–Fock result:

```rust
use metatron_qso::chemistry::cc::{CcOptions, ccsd, ccsd_t, mp2};
use metatron_qso::chemistry::integrals::{AoIntegrals, Basis, Molecule};
use metatron_qso::chemistry::scf::{ScfOptions, rhf, rohf};

fn main() -> metatron_qso::Result<()> {
    let water = Molecule::from_atoms("O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", 0, 1)?;
    let ints = AoIntegrals::compute(&water, Basis::CcPvdz)?;
    let hf = rhf(&water, &ints, &ScfOptions::default())?;
    let pt = mp2(&hf, &ints, 1)?;
    let cc = ccsd_t(&hf, &ints, 1, &CcOptions::default())?;
    println!("{} {} {}", pt.correlation, cc.ccsd.correlation, cc.triples);
    let oh = Molecule::from_atoms("O 0 0 0; H 0 0 0.97", 0, 2)?;
    let ints = AoIntegrals::compute(&oh, Basis::SixThirtyOneG)?;
    let ro = rohf(&oh, &ints, &ScfOptions::default())?;
    let open = ccsd(&ro, &ints, 0, &CcOptions::default())?;
    assert!(open.spin_orbitals);
    Ok(())
}
```

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
