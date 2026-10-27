//! Differential fuzzing: a circuit decoded from the input runs on every
//! simulator that takes it (`metatron_qso::differential::compare`), and
//! every result must match the double-precision state vector.
//!
//! Input: the first byte picks the register (1 to 6 qubits), then every
//! five bytes one gate: kind, three qubit bytes, angle byte.
#![no_main]

use libfuzzer_sys::fuzz_target;
use metatron_qso::differential::compare;
use metatron_qso::quantum::Circuit;

fuzz_target!(|data: &[u8]| {
    let Some((&first, rest)) = data.split_first() else {
        return;
    };
    let n = 1 + first as usize % 6;
    let mut c = Circuit::new(n);
    for g in rest.chunks_exact(5).take(64) {
        let a = g[1] as usize % n;
        let b = g[2] as usize % n;
        let t = g[3] as usize % n;
        let angle = (g[4] as f64 - 128.0) / 16.0;
        let two = a != b;
        let three = two && t != a && t != b;
        match g[0] % 24 {
            0 => {
                c.h(a);
            }
            1 => {
                c.x(a);
            }
            2 => {
                c.y(a);
            }
            3 => {
                c.s(a);
            }
            4 => {
                c.t(a);
            }
            5 => {
                c.sx(a);
            }
            6 => {
                c.rx(a, angle);
            }
            7 => {
                c.ry(a, angle);
            }
            8 => {
                c.rz(a, angle);
            }
            9 => {
                c.u(a, angle, 0.5 * angle, -0.3);
            }
            10 if two => {
                c.cx(a, b);
            }
            11 if two => {
                c.cz(a, b);
            }
            12 if two => {
                c.swap(a, b);
            }
            13 if two => {
                c.cy(a, b);
            }
            14 if two => {
                c.crx(a, b, angle);
            }
            15 if two => {
                c.cp(a, b, angle);
            }
            16 if two => {
                c.rxx(a, b, angle);
            }
            17 if two => {
                c.ryy(a, b, angle);
            }
            18 if two => {
                c.rzz(a, b, angle);
            }
            19 if three => {
                c.ccx(a, b, t);
            }
            20 if three => {
                c.cswap(a, b, t);
            }
            21 if two => {
                c.ch(a, b);
            }
            22 => {
                c.sdg(a);
            }
            _ => {
                c.z(a);
            }
        }
    }
    let report = compare(&c, &[]).expect("every decoded circuit runs");
    assert!(report.failures().is_empty(), "{:?}\n{report:?}", report.failures());
    // Every gate decoded here has an OpenQASM 2 and 3 form.
    assert!(report.qasm2.is_some() && report.qasm3.is_some(), "{report:?}");
});
