//! Stim circuit import: no input may panic, and every accepted circuit
//! exports to text that reads back to the same text.
#![no_main]

use libfuzzer_sys::fuzz_target;
use metatron_qso::qec::DetectorCircuit;

fuzz_target!(|src: &str| {
    let Ok(c) = DetectorCircuit::from_stim(src) else {
        return;
    };
    let text = c.to_stim().expect("imported circuits export");
    let back = DetectorCircuit::from_stim(&text).expect("exported circuits import");
    assert_eq!(back.to_stim().unwrap(), text);
});
