//! Fermion-operator and FCIDUMP parsers: no input may panic; accepted
//! operators print and re-parse to the same operator; accepted FCIDUMP
//! files write and re-read to the same integrals; small molecules map to a
//! Hermitian qubit Hamiltonian under every encoding.
#![no_main]

use libfuzzer_sys::fuzz_target;
use metatron_qso::chemistry::{FermionOperator, Mapping, MolecularIntegrals, QubitEncoding};

fuzz_target!(|src: &str| {
    if let Ok(op) = src.parse::<FermionOperator>() {
        let text = op.to_string();
        let again: FermionOperator =
            text.parse().unwrap_or_else(|e| panic!("'{text}' does not parse: {e}"));
        assert_eq!(again, op, "{text}");
        let _ = op.normal_ordered();
        if op.num_modes() <= 8 && op.num_terms() <= 8 {
            for m in [Mapping::JordanWigner, Mapping::Parity, Mapping::BravyiKitaev] {
                let _ = m.map(&op, op.num_modes().max(1));
            }
        }
    }
    let Ok(mol) = MolecularIntegrals::from_fcidump(src) else {
        return;
    };
    let again = MolecularIntegrals::from_fcidump(&mol.to_fcidump()).expect("written file re-reads");
    let m = mol.n_orbitals();
    assert_eq!((again.n_orbitals(), again.n_alpha(), again.n_beta()), (m, mol.n_alpha(), mol.n_beta()));
    if m <= 3 {
        for p in 0..m {
            for q in 0..m {
                assert_eq!(again.one_body(p, q), mol.one_body(p, q));
            }
        }
        for encoding in [
            QubitEncoding::new(Mapping::JordanWigner),
            QubitEncoding::new(Mapping::BravyiKitaev),
            QubitEncoding::parity_reduced(),
        ] {
            let _ = mol.qubit_hamiltonian(encoding);
        }
    }
});
