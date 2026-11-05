//! Electronic structure on qubits.
//!
//! * [`fermion`] — fermionic operators with normal ordering
//! * [`mapping`] — Jordan–Wigner, parity and Bravyi–Kitaev mappings and
//!   the two-qubit reduction
//! * [`integrals`] — molecules, STO-3G and 6-31G basis sets and their
//!   integrals over Gaussian functions
//! * [`scf`] — restricted and unrestricted Hartree–Fock and the
//!   transformation to molecular orbitals
//! * [`molecule`] — molecular integrals from FCIDUMP files, qubit
//!   Hamiltonians, Hartree–Fock states and full CI
//! * [`cc`] — MP2, CCSD and CCSD(T) references beyond full CI
//! * [`ansatz`] — UCCSD and k-UpCCGSD circuits
//! * [`adapt`] — ADAPT-VQE

pub mod adapt;
pub mod ansatz;
mod basis_data;
pub mod cc;
pub mod fermion;
pub mod integrals;
pub mod mapping;
pub mod molecule;
pub mod scf;

pub use adapt::{AdaptOptions, AdaptResult, adapt_vqe, adapt_vqe_molecule};
pub use ansatz::{Excitation, k_upccgsd, uccsd};
pub use cc::{CANONICAL_TOLERANCE, CcOptions, Ccsd, CcsdT, Mp2, ccsd, ccsd_t, mp2};
pub use fermion::{FermionOperator, Ladder};
pub use integrals::{AoIntegrals, Basis, Harmonics, Molecule};
pub use mapping::{Mapping, QubitEncoding, taper, two_qubit_reduction};
pub use molecule::MolecularIntegrals;
pub use scf::{Guess, ScfOptions, ScfResult, rhf, rohf, uhf};
