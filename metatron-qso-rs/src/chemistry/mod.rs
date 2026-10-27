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
//! * [`ansatz`] — UCCSD and k-UpCCGSD circuits
//! * [`adapt`] — ADAPT-VQE

pub mod adapt;
pub mod ansatz;
mod basis_data;
pub mod fermion;
pub mod integrals;
pub mod mapping;
pub mod molecule;
pub mod scf;

pub use adapt::{AdaptOptions, AdaptResult, adapt_vqe, adapt_vqe_molecule};
pub use ansatz::{Excitation, k_upccgsd, uccsd};
pub use fermion::{FermionOperator, Ladder};
pub use integrals::{AoIntegrals, Basis, Molecule};
pub use mapping::{Mapping, QubitEncoding, taper, two_qubit_reduction};
pub use molecule::MolecularIntegrals;
pub use scf::{Guess, ScfOptions, ScfResult, rhf, uhf};
