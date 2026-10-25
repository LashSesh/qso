//! OpenQASM 2 and 3 programs: no input may panic, and every accepted
//! program small enough to simulate must have a normalised outcome
//! distribution that its OpenQASM 3 (and, when possible, OpenQASM 2)
//! export reproduces.
#![no_main]

use std::collections::BTreeMap;

use libfuzzer_sys::fuzz_target;
use metatron_qso::quantum::{Operation, Program};

fn close(a: &BTreeMap<u64, f64>, b: &BTreeMap<u64, f64>) -> bool {
    let keys: std::collections::BTreeSet<_> = a.keys().chain(b.keys()).collect();
    keys.into_iter().all(|k| {
        (a.get(k).copied().unwrap_or(0.0) - b.get(k).copied().unwrap_or(0.0)).abs() < 1e-7
    })
}

fn moderate(ops: &[Operation]) -> bool {
    ops.iter().all(|op| match op {
        Operation::Gate(i) => i.angles(&[]).iter().all(|a| a.abs() <= 1e6),
        Operation::If { body, .. } => moderate(body),
        _ => true,
    })
}

fuzz_target!(|src: &str| {
    let Ok(program) = Program::from_qasm(src) else {
        return;
    };
    // Exports must always succeed for OpenQASM 3 and re-import.
    let v3 = program.to_qasm3().expect("every program exports to OpenQASM 3");
    let again = Program::from_qasm(&v3).expect("exported programs import");
    assert_eq!(
        (again.n_qubits(), again.n_bits()),
        (program.n_qubits(), program.n_bits())
    );
    // The exporter adds and halves angles; beyond ~10⁶ rad that rounding
    // shows up in the state, so equivalence is checked for moderate angles.
    let moderate = moderate(program.operations());
    if program.n_qubits() > 8 || program.len() > 400 {
        return;
    }
    let Ok(dist) = program.probabilities() else {
        return;
    };
    let total: f64 = dist.values().sum();
    assert!((total - 1.0).abs() < 1e-9, "total probability {total}");
    if !moderate {
        return;
    }
    let d3 = again.probabilities().expect("the export simulates like the original");
    assert!(close(&dist, &d3), "OpenQASM 3 export changed the distribution:\n{v3}");
    if let Ok(v2) = program.to_qasm2() {
        let d2 = Program::from_qasm(&v2)
            .expect("exported programs import")
            .probabilities()
            .expect("the export simulates like the original");
        assert!(close(&dist, &d2), "OpenQASM 2 export changed the distribution:\n{v2}");
    }
});
