//! Electronic structure on qubits.
//!
//! * [`fermion`] — fermionic operators with normal ordering
//! * [`mapping`] — Jordan–Wigner, parity and Bravyi–Kitaev mappings and
//!   the two-qubit reduction
//! * [`molecule`] — molecular integrals from FCIDUMP files, qubit
//!   Hamiltonians, Hartree–Fock states and full CI
//! * [`ansatz`] — UCCSD and k-UpCCGSD circuits
//! * [`adapt`] — ADAPT-VQE

pub mod adapt;
pub mod ansatz;
pub mod fermion;
pub mod mapping;
pub mod molecule;

pub use adapt::{AdaptOptions, AdaptResult, adapt_vqe, adapt_vqe_molecule};
pub use ansatz::{Excitation, k_upccgsd, uccsd};
pub use fermion::{FermionOperator, Ladder};
pub use mapping::{Mapping, QubitEncoding, taper, two_qubit_reduction};
pub use molecule::MolecularIntegrals;
