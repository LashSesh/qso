# Electronic structure

QSO's chemistry chain starts from the molecular integrals. From them it
builds the Hamiltonian, maps it to qubits and prepares the Hartree–Fock
state. Full configuration interaction (CI) gives the exact reference
energy, and UCCSD, k-UpCCGSD and ADAPT-VQE give variational estimates.
QSO computes no integrals itself. It reads them from FCIDUMP files, which
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

`fci_energies(k)` diagonalises the Hamiltonian among all determinants
with the given numbers of spin-up and spin-down electrons. Up to 4096
determinants it returns the lowest `k` levels. Above that it returns the
ground state only, computed by Lanczos.

The tests check the whole chain against PySCF for H₂, LiH and BeH₂ in
STO-3G:

- Hartree–Fock energies agree to 10⁻⁹ Ha.
- The determinant-space spectra agree under Jordan–Wigner, parity,
  Bravyi–Kitaev and the reduced parity encoding.
- Their lowest roots equal PySCF's full-CI energies to 10⁻⁸ Ha.

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
