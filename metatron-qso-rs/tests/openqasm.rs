//! OpenQASM 2 and 3: the example programs of the OpenQASM project
//! (`tests/data/openqasm/`), their results and their round trips.

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::path::Path;

use metatron_qso::quantum::{Program, from_qasm2};

fn read(version: &str, name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/openqasm")
        .join(version)
        .join(format!("{name}.qasm"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn dist(p: &Program) -> BTreeMap<u64, f64> {
    p.probabilities().unwrap()
}

fn same(a: &BTreeMap<u64, f64>, b: &BTreeMap<u64, f64>) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|((ka, pa), (kb, pb))| ka == kb && (pa - pb).abs() < 1e-10)
}

/// Value of the bits of `record` from `offset` on, `size` wide.
fn field(record: u64, offset: usize, size: usize) -> u64 {
    (record >> offset) & ((1u64 << size) - 1)
}

/// Exporting to OpenQASM 3 (and to OpenQASM 2 where possible) and reading
/// back gives the same outcome distribution.
fn round_trips(p: &Program) {
    let d = dist(p);
    let v3 = Program::from_qasm(&p.to_qasm3().unwrap()).unwrap();
    assert!(
        same(&d, &dist(&v3)),
        "OpenQASM 3 round trip changed\n{}",
        p.to_qasm3().unwrap()
    );
    if let Ok(text) = p.to_qasm2() {
        let v2 = Program::from_qasm(&text).unwrap();
        assert!(
            same(&d, &dist(&v2)),
            "OpenQASM 2 round trip changed\n{text}"
        );
    }
}

#[test]
fn openqasm2_examples_give_their_known_results() {
    // Ripple-carry adder: 1 + 15 = 16 (b = 0000, carry 1).
    let adder = Program::from_qasm(&read("v2", "adder")).unwrap();
    assert_eq!(
        dist(&adder).into_iter().collect::<Vec<_>>(),
        vec![(16, 1.0)]
    );
    round_trips(&adder);
    // 8-bit adder: the program's inputs a = 1, b = 15 give 16 (carry 0).
    let big = Program::from_qasm(&read("v2", "bigadder")).unwrap();
    let d = dist(&big);
    assert_eq!(d.len(), 1);
    // Phase estimation of e^{i·3π/8}: 3/16 = 0.0011₂, exact in four bits.
    let pea = Program::from_qasm(&read("v2", "pea_3_pi_8")).unwrap();
    let ipea = Program::from_qasm(&read("v2", "ipea_3_pi_8")).unwrap();
    for p in [&pea, &ipea] {
        let d = dist(p);
        assert_eq!(d.len(), 1, "{d:?}");
        let (&value, &prob) = d.iter().next().unwrap();
        assert!((prob - 1.0).abs() < 1e-10);
        // Bit k of the estimate is the k-th binary digit counted from the
        // least significant measured digit 1/16.
        let digits: f64 = (0..4)
            .map(|k| ((value >> k) & 1) as f64 / 2f64.powi(4 - k))
            .sum();
        assert!(
            (digits * 2.0 * PI - 3.0 * PI / 8.0).abs() < 1e-12,
            "{value:04b}"
        );
        round_trips(p);
    }
    // Teleportation: the third bit is distributed like a measurement of
    // u3(0.3, 0.2, 0.1)|0⟩, P(1) = sin²(0.15), whatever the first two say.
    for name in ["teleport", "teleportv2"] {
        let p = Program::from_qasm(&read("v2", name)).unwrap();
        let p1: f64 = dist(&p)
            .iter()
            .filter(|(r, _)| (*r >> 2) & 1 == 1)
            .map(|(_, q)| q)
            .sum();
        assert!((p1 - 0.15f64.sin().powi(2)).abs() < 1e-12, "{name}");
        round_trips(&p);
    }
    // W state: 001, 010, 100 with probability ≈ 1/3 (u3(1.91063) rounds
    // 2·arccos(1/√3) = 1.910633).
    let w = Program::from_qasm(&read("v2", "W-state")).unwrap();
    let d = dist(&w);
    assert_eq!(d.keys().copied().collect::<Vec<_>>(), vec![1, 2, 4]);
    assert!(d.values().all(|p| (p - 1.0 / 3.0).abs() < 1e-5));
    round_trips(&w);
    // Repetition code: the syndrome 01 locates the error on q[0]; after
    // the correction all data qubits read 0.
    let qec = Program::from_qasm(&read("v2", "qec")).unwrap();
    let d = dist(&qec);
    assert_eq!(d.len(), 1);
    let record = *d.keys().next().unwrap();
    assert_eq!((field(record, 0, 3), field(record, 3, 2)), (0, 1));
    round_trips(&qec);
    // QFT of |0101⟩: uniform over 16 outcomes; randomized benchmarking
    // sequence: returns to 00; qpt: H|0⟩ measured.
    let qft = Program::from_qasm(&read("v2", "qft")).unwrap();
    let d = dist(&qft);
    assert_eq!(d.len(), 16);
    assert!(d.values().all(|p| (p - 1.0 / 16.0).abs() < 1e-12));
    round_trips(&qft);
    let rb = Program::from_qasm(&read("v2", "rb")).unwrap();
    assert_eq!(dist(&rb).keys().copied().collect::<Vec<_>>(), vec![0]);
    round_trips(&rb);
    let qpt = Program::from_qasm(&read("v2", "qpt")).unwrap();
    assert!(dist(&qpt).values().all(|p| (p - 0.5).abs() < 1e-12));
    // Inverse QFT with measurement and feed-forward, two versions.
    let i1 = Program::from_qasm(&read("v2", "inverseqft1")).unwrap();
    let i2 = Program::from_qasm(&read("v2", "inverseqft2")).unwrap();
    assert!(same(&dist(&i1), &dist(&i2)));
    round_trips(&i1);
    round_trips(&i2);
    // The repository's two invalid examples are errors.
    for name in ["invalid_gate_no_found", "invalid_missing_semicolon"] {
        assert!(Program::from_qasm(&read("v2", name)).is_err(), "{name}");
    }
    // from_qasm2 reads the unitary ones as circuits.
    assert_eq!(from_qasm2(&read("v2", "adder")).unwrap().n_qubits(), 10);
}

#[test]
fn openqasm3_examples_in_the_subset_agree_with_their_openqasm2_versions() {
    for name in ["teleport", "inverseqft1", "inverseqft2", "qft", "rb", "qpt"] {
        let v3 = Program::from_qasm(&read("v3", name)).unwrap();
        let v2 = Program::from_qasm(&read("v2", name)).unwrap();
        // Same registers in the same order in both versions.
        assert!(same(&dist(&v3), &dist(&v2)), "{name}");
        round_trips(&v3);
    }
    // The other examples use features outside the subset (loops,
    // subroutines, classical types, calibrations, timing) and are rejected
    // with an error naming the feature, never mis-read.
    for name in [
        "adder",
        "alignment",
        "arrays",
        "cphase",
        "dd",
        "defcal",
        "gateteleport",
        "ipe",
        "msd",
        "qec",
        "rus",
        "scqec",
        "t1",
        "varteleport",
        "vqe",
    ] {
        assert!(Program::from_qasm(&read("v3", name)).is_err(), "{name}");
    }
}
