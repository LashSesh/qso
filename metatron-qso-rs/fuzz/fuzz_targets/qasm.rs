//! OpenQASM 2 import: no input may panic, and every accepted program must
//! export and re-import to an equivalent circuit (the exporter decomposes
//! gates outside `qelib1.inc`, up to a global phase).
#![no_main]

use libfuzzer_sys::fuzz_target;
use metatron_qso::Rng;
use metatron_qso::quantum::{StateVector, from_qasm2, to_qasm2};

fuzz_target!(|src: &str| {
    let Ok(circuit) = from_qasm2(src) else {
        return;
    };
    let text = to_qasm2(&circuit, &[], false).expect("imported circuits have no free parameters");
    let again = from_qasm2(&text).expect("exported programs import");
    assert_eq!(again.n_qubits(), circuit.n_qubits());
    // The exporter adds and halves angles; beyond ~10⁶ rad that rounding
    // shows up in the state, so equivalence is checked for moderate angles.
    let moderate = circuit
        .instructions()
        .iter()
        .all(|i| i.angles(&[]).iter().all(|a| a.abs() <= 1e6));
    if circuit.n_qubits() <= 8 {
        let mut rng = Rng::seed_from_u64(1);
        for _ in 0..2 {
            let input = StateVector::random(circuit.n_qubits(), &mut rng);
            let (mut a, mut b) = (input.clone(), input);
            a.apply_circuit(&circuit, &[]).expect("imported circuits simulate");
            b.apply_circuit(&again, &[]).expect("re-imported circuits simulate");
            assert!((a.norm() - 1.0).abs() < 1e-9, "norm {}", a.norm());
            let f = a.fidelity(&b).unwrap();
            assert!(!moderate || f > 1.0 - 1e-9, "fidelity {f} after export of\n{text}");
        }
    }
});
