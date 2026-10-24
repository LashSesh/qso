//! Pauli-sum parser: no input may panic, and every accepted sum must print
//! and re-parse to the same sum.
#![no_main]

use libfuzzer_sys::fuzz_target;
use metatron_qso::quantum::PauliSum;

fuzz_target!(|src: &str| {
    let Ok(sum) = src.parse::<PauliSum>() else {
        return;
    };
    let text = sum.to_string();
    let again: PauliSum = text.parse().unwrap_or_else(|e| panic!("'{text}' does not parse: {e}"));
    assert_eq!(again.num_terms(), sum.num_terms(), "{text}");
    for ((a, s), (b, t)) in again.terms().iter().zip(sum.terms()) {
        assert_eq!(s, t, "{text}");
        assert!(a == b || (a - b).abs() <= 1e-12 * b.abs(), "{a} vs {b} in '{text}'");
    }
});
