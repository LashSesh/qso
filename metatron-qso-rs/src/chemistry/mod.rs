//! Electronic structure on qubits.
//!
//! * [`fermion`] — fermionic operators with normal ordering
//! * [`mapping`] — Jordan–Wigner, parity and Bravyi–Kitaev mappings and
//!   the two-qubit reduction
//! * [`molecule`] — molecular integrals from FCIDUMP files, qubit
//!   Hamiltonians, Hartree–Fock states and full CI

pub mod fermion;
pub mod mapping;
pub mod molecule;

pub use fermion::{FermionOperator, Ladder};
pub use mapping::{Mapping, QubitEncoding, taper, two_qubit_reduction};
pub use molecule::MolecularIntegrals;
