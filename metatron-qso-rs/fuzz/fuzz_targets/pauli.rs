//! Pauli-sum parsers: no input may panic, every accepted sum must print
//! and re-parse to the same sum, and the wide operator of Pauli
//! propagation (16 384 qubits) must agree with `PauliSum` where both
//! accept the text.
#![no_main]

use libfuzzer_sys::fuzz_target;
use metatron_qso::propagation::PauliOperator;
use metatron_qso::quantum::PauliSum;

fuzz_target!(|src: &str| {
    let sum = src.parse::<PauliSum>();
    if let Ok(sum) = &sum {
        let text = sum.to_string();
        let again: PauliSum = text
            .parse()
            .unwrap_or_else(|e| panic!("'{text}' does not parse: {e}"));
        assert_eq!(again.num_terms(), sum.num_terms(), "{text}");
        for ((a, s), (b, t)) in again.terms().iter().zip(sum.terms()) {
            assert_eq!(s, t, "{text}");
            assert!(a == b || (a - b).abs() <= 1e-12 * b.abs(), "{a} vs {b} in '{text}'");
        }
    }
    let Ok(op) = PauliOperator::parse(16384, src) else {
        return;
    };
    // Display prints the shortest round-trip form, so the operator comes
    // back exactly.
    let text = op.to_string();
    let again = PauliOperator::parse(16384, &text)
        .unwrap_or_else(|e| panic!("'{text}' does not parse as an operator: {e}"));
    assert_eq!(again, op, "{text}");
    if let Ok(sum) = sum {
        assert_eq!(op.to_sum().expect("a PauliSum lies on 64 qubits"), sum, "{src}");
    }
});
