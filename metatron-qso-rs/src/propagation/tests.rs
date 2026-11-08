//! Unit tests of Pauli propagation: the operator type, the tables of gates
//! and channels against dense matrices, and small noisy circuits against
//! closed forms and density-matrix values.

use std::collections::BTreeMap;
use std::f64::consts::{FRAC_PI_2, PI};
use std::sync::Arc;

use num_complex::Complex64 as C64;

use super::compile::{Op, compile_instruction, compile_layers, instruction_table};
use super::transfer::{LocalMap, SNAP, TableCache, channel_table};
use super::*;
use crate::linalg::CMatrix;
use crate::noise::learning::SparsePauliLindblad;
use crate::photonics::haar_unitary;
use crate::quantum::{GateKind, Instruction, Param, Pauli, PauliString, PauliSum, StateVector};
use crate::rng::Rng;

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

/// Dense single-qubit Pauli.
fn sigma(p: usize) -> CMatrix {
    let (o, z, i) = (C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 1.0));
    match p {
        0 => CMatrix::from_row_slice(2, 2, &[o, z, z, o]),
        1 => CMatrix::from_row_slice(2, 2, &[z, o, o, z]),
        2 => CMatrix::from_row_slice(2, 2, &[z, -i, i, z]),
        _ => CMatrix::from_row_slice(2, 2, &[o, z, z, -o]),
    }
}

/// Dense local string `b = xbits | zbits << k` as a Kronecker product,
/// the op's qubit 0 the most significant factor.
fn dense_string(b: usize, k: usize) -> CMatrix {
    let letter = |j: usize| match ((b >> j) & 1, (b >> (k + j)) & 1) {
        (0, 0) => 0,
        (1, 0) => 1,
        (1, 1) => 2,
        _ => 3,
    };
    let mut m = sigma(letter(0));
    for j in 1..k {
        m = m.kronecker(&sigma(letter(j)));
    }
    m
}

/// The matrix of a local instruction (controls `0..c`, targets after)
/// from the state-vector simulator: column `m` is the image of basis
/// state `m`, local qubit `j` at matrix bit `k − 1 − j`.
fn dense_gate(inst: &Instruction, k: usize) -> CMatrix {
    let d = 1usize << k;
    let reverse = |s: usize| (0..k).fold(0, |acc, j| acc | (((s >> j) & 1) << (k - 1 - j)));
    let mut u = CMatrix::zeros(d, d);
    for m in 0..d {
        let mut psi = StateVector::basis(k, reverse(m)).unwrap();
        psi.apply_instruction(inst, &[]).unwrap();
        for (s, a) in psi.amplitudes().iter().enumerate() {
            u[(reverse(s), m)] = *a;
        }
    }
    u
}

/// Reference coefficients `c_a = Tr(P_a M)/d = Σ_ij (P_a)_ij M_ji / d`
/// of a dense operator.
fn coefficients(m: &CMatrix, k: usize) -> Vec<C64> {
    let d = 1usize << k;
    (0..1usize << (2 * k))
        .map(|a| {
            let p = dense_string(a, k);
            let mut tr = C64::new(0.0, 0.0);
            for i in 0..d {
                for j in 0..d {
                    tr += p[(i, j)] * m[(j, i)];
                }
            }
            tr / d as f64
        })
        .collect()
}

/// Compare every column of `map` with the Heisenberg images `heis(P_b)`:
/// kept entries to `tol`, dropped ones below the snap and their sum equal
/// to the residual.
fn check_table(map: &LocalMap, k: usize, tol: f64, heis: impl Fn(&CMatrix) -> CMatrix) {
    for b in 0..1usize << (2 * k) {
        let reference = coefficients(&heis(&dense_string(b, k)), k);
        let images = map.images(b);
        let mut dropped = 0.0;
        for (a, c) in reference.iter().enumerate() {
            assert!(c.im.abs() <= tol, "b = {b}, a = {a}: imaginary {}", c.im);
            match images.iter().find(|&&(i, _)| i as usize == a) {
                Some(&(_, f)) => assert!((f - c.re).abs() <= tol, "b = {b}, a = {a}: {f} vs {c}"),
                None => {
                    assert!(c.re.abs() <= SNAP + tol, "b = {b}, a = {a}: dropped {c}");
                    dropped += c.re.abs();
                }
            }
        }
        if b != 0 {
            assert!(
                (dropped - map.residual(b)).abs() <= tol,
                "b = {b}: dropped {dropped} vs residual {}",
                map.residual(b)
            );
        }
    }
}

/// `U† P U` for a dense unitary.
fn heisenberg_gate(u: &CMatrix) -> impl Fn(&CMatrix) -> CMatrix + '_ {
    move |p| u.adjoint() * p * u
}

fn local(kind: GateKind, controls: usize, targets: usize, params: &[f64]) -> Instruction {
    Instruction {
        kind,
        targets: (controls..controls + targets).collect(),
        controls: (0..controls).collect(),
        params: params.iter().map(|&a| Param::Value(a)).collect(),
        matrix: None,
    }
}

fn table(inst: &Instruction) -> Arc<LocalMap> {
    instruction_table(inst, &inst.angles(&[]), &mut TableCache::default()).unwrap()
}

/// Serial Heisenberg propagation of `observable` through `ops` (time
/// order) in a sorted map, the reference engine of these tests.
fn heisenberg(ops: &[Op], observable: &PauliOperator) -> BTreeMap<Vec<u64>, f64> {
    let w = observable.words();
    let (keys, coefs) = observable.rows();
    let mut current: BTreeMap<Vec<u64>, f64> = coefs
        .iter()
        .enumerate()
        .map(|(i, &c)| (keys[2 * w * i..2 * w * (i + 1)].to_vec(), c))
        .collect();
    for op in ops.iter().rev() {
        let mut next: BTreeMap<Vec<u64>, f64> = BTreeMap::new();
        for (key, &c) in &current {
            match op {
                Op::Local { qubits, map } => {
                    let k = qubits.len();
                    let mut b = 0usize;
                    for (j, &q) in qubits.iter().enumerate() {
                        let (wq, bit) = (q / 64, q % 64);
                        b |= (((key[wq] >> bit) & 1) as usize) << j;
                        b |= (((key[w + wq] >> bit) & 1) as usize) << (k + j);
                    }
                    for &(image, f) in map.images(b) {
                        let mut out = key.clone();
                        for (j, &q) in qubits.iter().enumerate() {
                            let (wq, bit) = (q / 64, q % 64);
                            let xb = ((image as u64) >> j) & 1;
                            let zb = ((image as u64) >> (k + j)) & 1;
                            out[wq] = (out[wq] & !(1 << bit)) | (xb << bit);
                            out[w + wq] = (out[w + wq] & !(1 << bit)) | (zb << bit);
                        }
                        *next.entry(out).or_insert(0.0) += c * f;
                    }
                }
                Op::Diag(d) => *next.entry(key.clone()).or_insert(0.0) += c * d.fidelity(key),
            }
        }
        next.retain(|_, c| *c != 0.0);
        current = next;
    }
    current
}

/// `⟨0…0| O |0…0⟩` of a propagated table: the coefficients of the strings
/// without X parts.
fn value_at_zero(table: &BTreeMap<Vec<u64>, f64>, words: usize) -> f64 {
    table
        .iter()
        .filter(|(key, _)| key[..words].iter().all(|&x| x == 0))
        .map(|(_, c)| c)
        .sum()
}

/// `⟨0…0| O(0) |0…0⟩` by [`propagate`], checked against the serial
/// reference engine [`heisenberg`] (string for string).
fn run(layers: &[Layer], observable: &PauliOperator) -> f64 {
    let n = observable.n_qubits();
    let ops = compile_layers(layers, n, &mut TableCache::default()).unwrap();
    let reference = heisenberg(&ops, observable);
    let expected = value_at_zero(&reference, observable.words());
    let result = propagate(
        layers,
        observable,
        None,
        &PropagationOptions::default().with_operator(true),
    )
    .unwrap();
    assert!(
        (result.value - expected).abs() <= 1e-12,
        "engine {} vs reference {expected}",
        result.value
    );
    assert!(
        result.truncation_bound <= 1e-12,
        "{}",
        result.truncation_bound
    );
    let operator = result.operator.unwrap();
    let w = observable.words();
    let (keys, coefs) = operator.rows();
    let mut engine: BTreeMap<Vec<u64>, f64> = BTreeMap::new();
    for (i, &c) in coefs.iter().enumerate() {
        engine.insert(keys[2 * w * i..2 * w * (i + 1)].to_vec(), c);
    }
    for (key, &c) in &reference {
        let e = engine.get(key).copied().unwrap_or(0.0);
        assert!(
            (e - c).abs() <= 1e-12,
            "{key:?}: engine {e} vs reference {c}"
        );
    }
    for (key, &c) in &engine {
        assert!(
            reference.contains_key(key) || c.abs() <= 1e-12,
            "{key:?}: engine {c}, absent from the reference"
        );
    }
    result.value
}

fn circuit(n: usize, build: impl FnOnce(&mut Circuit)) -> Layer {
    let mut c = Circuit::new(n);
    build(&mut c);
    Layer::gates(c)
}

// ---------------------------------------------------------------------
// 1. PauliOperator
// ---------------------------------------------------------------------

#[test]
fn operator_text_round_trips_on_every_register_size() {
    for n in [1usize, 64, 65, 200, 16384] {
        let last = n - 1;
        let text = if n == 1 {
            "0.5*Z0 - 1.25 X0 + 3e-7*Y0 + 2 + 0.25 Y0 - I".to_string()
        } else {
            let mid = n / 2;
            format!(
                "0.5*Z0 Z{last} - 1.25 X{mid} Y{last} + 3e-7*Y{mid} + 2 - 0.5 Z{last} Z0 \
                 + 1e-300 X{last} - 7.5e12*Y0*X{mid}*Z{last}"
            )
        };
        let op = PauliOperator::parse(n, &text).unwrap();
        let printed = op.to_string();
        let again = PauliOperator::parse(n, &printed).unwrap();
        assert_eq!(again, op, "n = {n}: '{printed}'");
        assert_eq!(again.to_string(), printed);
        // Canonical order: rows ascending by (x words, z words).
        let (keys, _) = op.rows();
        let r = 2 * op.words();
        assert!((1..op.len()).all(|i| keys[(i - 1) * r..i * r] < keys[i * r..(i + 1) * r]));
        if n == 1 {
            // 0.5 Z0 + 1 (2 − I) − 1.25 X0 + (3e-7 + 0.25) Y0.
            assert_eq!(op.len(), 4);
            assert_eq!(op.coefficient(&[]).unwrap(), 1.0);
            assert_eq!(op.coefficient(&[(0, Pauli::Y)]).unwrap(), 3e-7 + 0.25);
        } else {
            // Z0 Z_last cancels exactly.
            assert_eq!(op.len(), 5, "n = {n}: {op}");
            assert_eq!(
                op.coefficient(&[(0, Pauli::Z), (last, Pauli::Z)]).unwrap(),
                0.0
            );
            assert_eq!(op.coefficient(&[]).unwrap(), 2.0);
            assert_eq!(op.max_weight(), 3);
        }
    }
}

#[test]
fn operator_prints_in_canonical_order() {
    let op = PauliOperator::parse(200, "0.5*Z0 Z199 + X120 - 1.05 + 0.25 X120").unwrap();
    assert_eq!(op.to_string(), "-1.05 + 0.5 Z0 Z199 + 1.25 X120");
    assert_eq!(op.len(), 3);
    assert!((op.one_norm() - 2.8).abs() < 1e-15);
    assert_eq!(op.max_weight(), 2);
    let terms: Vec<_> = op.terms().collect();
    assert_eq!(terms[0], (-1.05, vec![]));
    assert_eq!(terms[1], (0.5, vec![(0, Pauli::Z), (199, Pauli::Z)]));
    assert_eq!(terms[2], (1.25, vec![(120, Pauli::X)]));
    assert_eq!(PauliOperator::new(3).unwrap().to_string(), "0");
}

#[test]
fn operator_matches_pauli_sum_up_to_64_qubits() {
    let sum: PauliSum = "0.5*Z0 Z1 - 1.2 X0 + 0.3*Y2*Y3 - 1.05 + 2 X63 Z5 - 0.75 Y1 X0"
        .parse()
        .unwrap();
    for n in [64, 70, 1000] {
        let op = PauliOperator::from_sum(n, &sum).unwrap();
        assert_eq!(op.to_sum().unwrap(), sum);
        assert_eq!(op.to_string(), sum.to_string());
        assert_eq!(PauliOperator::parse(n, &sum.to_string()).unwrap(), op);
        assert_eq!(
            PauliOperator::parse(n, &sum.to_string()).unwrap().len(),
            sum.num_terms()
        );
        assert_eq!(op.one_norm(), sum.one_norm());
    }
    // Merging in the order given, as PauliSum.
    let text = "0.1 X0 + 0.2 X0 + 0.3 Z1 - 0.3 Z1 + 1e-17 X0";
    let sum: PauliSum = text.parse().unwrap();
    let op = PauliOperator::parse(5, text).unwrap();
    assert_eq!(op.to_sum().unwrap(), sum);
    assert_eq!(op.len(), 1);
    // A term beyond qubit 63 has no PauliSum.
    let wide = PauliOperator::parse(65, "X64").unwrap();
    assert!(matches!(
        wide.to_sum(),
        Err(QsoError::QubitOutOfRange { qubit: 64, .. })
    ));
    assert!(PauliOperator::from_sum(3, &sum).is_ok());
    let big: PauliSum = "Z5".parse().unwrap();
    assert!(matches!(
        PauliOperator::from_sum(5, &big),
        Err(QsoError::QubitOutOfRange {
            qubit: 5,
            n_qubits: 5
        })
    ));
}

#[test]
fn operator_terms_merge_and_cancel() {
    let mut op = PauliOperator::new(130).unwrap();
    op.add_term(0.5, &[(129, Pauli::X), (3, Pauli::Z)]).unwrap();
    op.add_term(1.5, &[]).unwrap();
    op.add_term(-0.25, &[(3, Pauli::Z), (129, Pauli::X), (7, Pauli::I)])
        .unwrap();
    assert_eq!(op.len(), 2);
    assert_eq!(
        op.coefficient(&[(3, Pauli::Z), (129, Pauli::X)]).unwrap(),
        0.25
    );
    op.add_term(-0.25, &[(3, Pauli::Z), (129, Pauli::X)])
        .unwrap();
    op.add_term(0.0, &[(4, Pauli::Y)]).unwrap();
    assert_eq!(op.len(), 1);
    assert_eq!(op.to_string(), "1.5");
    op.add_term(-1.5, &[]).unwrap();
    assert!(op.is_empty());
    assert_eq!(op.max_weight(), 0);
    assert_eq!(op.one_norm(), 0.0);
    assert_eq!(op, PauliOperator::new(130).unwrap());
}

#[test]
fn operator_errors_name_the_problem() {
    for n in [0, MAX_PROPAGATION_QUBITS + 1] {
        assert!(matches!(
            PauliOperator::new(n),
            Err(QsoError::TooManyQubits { what: "Pauli propagation", requested, .. }) if requested == n
        ));
        assert!(matches!(
            PauliOperator::parse(n, "Z0"),
            Err(QsoError::TooManyQubits { .. })
        ));
    }
    assert!(matches!(
        PauliOperator::parse(10, "X10"),
        Err(QsoError::QubitOutOfRange {
            qubit: 10,
            n_qubits: 10
        })
    ));
    assert!(matches!(
        PauliOperator::parse(10, "X1 Z1"),
        Err(QsoError::DuplicateQubit {
            gate: "Pauli string",
            qubit: 1
        })
    ));
    for bad in [
        "X1 +",
        "X",
        "2 * * Z1",
        "Q3",
        "X99999999999999999999999",
        "1e999 X0",
    ] {
        assert!(
            matches!(PauliOperator::parse(10, bad), Err(QsoError::Parse(_))),
            "{bad}"
        );
    }
    assert!(matches!(
        PauliOperator::parse(10, "1e308 X0 + 1e308 X0"),
        Err(QsoError::InvalidArgument {
            name: "coefficient",
            ..
        })
    ));
    let mut op = PauliOperator::new(4).unwrap();
    assert!(op.add_term(f64::NAN, &[(0, Pauli::X)]).is_err());
    assert!(op.add_term(1.0, &[(4, Pauli::X)]).is_err());
    op.add_term(1e308, &[(0, Pauli::X)]).unwrap();
    assert!(op.add_term(1e308, &[(0, Pauli::X)]).is_err());
    assert_eq!(op.coefficient(&[(0, Pauli::X)]).unwrap(), 1e308);
    assert!(op.coefficient(&[(0, Pauli::X), (0, Pauli::Z)]).is_err());
}

#[test]
fn lindblad_rates_land_on_the_mapped_qubits() {
    let model = SparsePauliLindblad::new(
        2,
        vec![
            PauliString::new(&[(0, Pauli::X)]).unwrap(),
            PauliString::new(&[(0, Pauli::Z), (1, Pauli::Y)]).unwrap(),
            PauliString::new(&[(1, Pauli::Z)]).unwrap(),
        ],
        vec![0.1, 0.02, 0.0],
    )
    .unwrap();
    let op = PauliOperator::from_lindblad(&model, &[100, 7], 128).unwrap();
    assert_eq!(op.len(), 2);
    assert_eq!(op.coefficient(&[(100, Pauli::X)]).unwrap(), 0.1);
    assert_eq!(
        op.coefficient(&[(100, Pauli::Z), (7, Pauli::Y)]).unwrap(),
        0.02
    );
    assert!(matches!(
        PauliOperator::from_lindblad(&model, &[3, 3], 8),
        Err(QsoError::DuplicateQubit {
            gate: "qubit map",
            qubit: 3
        })
    ));
    assert!(matches!(
        PauliOperator::from_lindblad(&model, &[3], 8),
        Err(QsoError::DimensionMismatch {
            expected: 2,
            actual: 1
        })
    ));
    assert!(matches!(
        PauliOperator::from_lindblad(&model, &[3, 8], 8),
        Err(QsoError::QubitOutOfRange {
            qubit: 8,
            n_qubits: 8
        })
    ));
}

// ---------------------------------------------------------------------
// 2. Clifford tables
// ---------------------------------------------------------------------

#[test]
fn clifford_tables_match_dense_conjugation_on_every_string() {
    use GateKind::*;
    let cases = [
        local(H, 0, 1, &[]),
        local(X, 0, 1, &[]),
        local(Y, 0, 1, &[]),
        local(Z, 0, 1, &[]),
        local(S, 0, 1, &[]),
        local(Sdg, 0, 1, &[]),
        local(SX, 0, 1, &[]),
        local(SXdg, 0, 1, &[]),
        local(Swap, 0, 2, &[]),
        local(X, 1, 1, &[]),
        local(Y, 1, 1, &[]),
        local(Z, 1, 1, &[]),
    ];
    for inst in &cases {
        let k = inst.controls.len() + inst.targets.len();
        let map = table(inst);
        assert!(map.is_permutation(), "{}", inst.name());
        let u = dense_gate(inst, k);
        for b in 0..1usize << (2 * k) {
            let images = map.images(b);
            assert_eq!(images.len(), 1, "{} b = {b}", inst.name());
            let (image, f) = images[0];
            assert!(f == 1.0 || f == -1.0);
            assert_eq!(map.residual(b), 0.0);
            let reference = coefficients(&(u.adjoint() * dense_string(b, k) * &u), k);
            for (a, c) in reference.iter().enumerate() {
                let expected = if a == image as usize { f } else { 0.0 };
                assert!(
                    (c - C64::new(expected, 0.0)).norm() < 1e-12,
                    "{} b = {b} a = {a}: {c} vs {expected}",
                    inst.name()
                );
            }
        }
    }
    // Signs as Stim's PauliString.before (C† P C).
    assert_eq!(table(&local(S, 0, 1, &[])).images(0b01), &[(0b11, -1.0)]); // X → −Y
    assert_eq!(table(&local(SX, 0, 1, &[])).images(0b11), &[(0b10, -1.0)]); // Y → −Z
    let cz = table(&local(Z, 1, 1, &[]));
    assert_eq!(cz.images(0b0011), &[(0b1111, 1.0)]); // XX → +YY
    assert!(cz.grows());
    assert!(!table(&local(H, 0, 1, &[])).grows());
}

// ---------------------------------------------------------------------
// 3. Rotation tables
// ---------------------------------------------------------------------

#[test]
fn rotation_tables_match_dense_conjugation_and_snap_into_the_residual() {
    use GateKind::*;
    let angles = [FRAC_PI_2, -FRAC_PI_2, PI, 0.7, 1e-15];
    let mut cases: Vec<(Instruction, f64)> = Vec::new();
    for &theta in &angles {
        for kind in [RX, RY, RZ, P] {
            cases.push((local(kind, 0, 1, &[theta]), theta));
        }
        for kind in [RXX, RYY, RZZ] {
            cases.push((local(kind, 0, 2, &[theta]), theta));
        }
    }
    cases.push((local(T, 0, 1, &[]), PI / 4.0));
    cases.push((local(Tdg, 0, 1, &[]), -PI / 4.0));
    for (inst, theta) in &cases {
        let k = inst.targets.len();
        let map = table(inst);
        let u = dense_gate(inst, k);
        check_table(&map, k, 1e-15, heisenberg_gate(&u));
        // Every anticommuting string has two images unless one branch is
        // snapped; the snapped branch is then exactly the residual.
        let (cos, sin) = (theta.cos().abs(), theta.sin().abs());
        let clifford = cos <= SNAP || sin <= SNAP;
        assert_eq!(map.is_permutation(), clifford, "{} {theta}", inst.name());
        // (check_table has matched the residuals with the dense branches.)
        for b in 1..1usize << (2 * k) {
            let r = map.residual(b);
            let snapped = if cos <= SNAP { cos } else { sin };
            assert!(
                r == 0.0 || (clifford && r == snapped),
                "{} {theta} b = {b}: {r}",
                inst.name()
            );
        }
        // Two-qubit rotations grow strings through their sin branch.
        assert_eq!(map.grows(), k == 2 && sin > SNAP, "{} {theta}", inst.name());
    }
    // RX(π/2): Z → −Y exactly... as R†ZR = cos θ Z + sin θ Y.
    let rx = table(&local(RX, 0, 1, &[FRAC_PI_2]));
    assert_eq!(rx.images(0b10), &[(0b11, 1.0)]);
    assert_eq!(rx.residual(0b10), FRAC_PI_2.cos());
}

// ---------------------------------------------------------------------
// 4. Transfer matrices
// ---------------------------------------------------------------------

#[test]
fn transfer_tables_match_dense_conjugation() {
    use GateKind::*;
    let mut cases = vec![
        local(H, 1, 1, &[]),
        local(S, 1, 1, &[]),
        local(T, 1, 1, &[]),
        local(SX, 1, 1, &[]),
        local(U, 1, 1, &[0.3, -0.5, 0.7]),
        local(U, 0, 1, &[0.3, -0.5, 0.7]),
        local(RZ, 1, 1, &[0.9]),
        local(RX, 1, 1, &[-0.4]),
        local(RY, 1, 1, &[1.3]),
        local(P, 1, 1, &[1.1]),
        local(P, 1, 1, &[PI]),
        local(X, 2, 1, &[]),
        local(Z, 2, 1, &[]),
        local(Swap, 1, 2, &[]),
        local(X, 3, 1, &[]),
        local(RZZ, 1, 2, &[0.6]),
        local(RZ, 2, 1, &[0.8]),
    ];
    let mut rng = Rng::seed_from_u64(164);
    for (targets, controls) in [(1, 0), (2, 0), (3, 0), (4, 0), (1, 1), (2, 1), (1, 3)] {
        let d = 1usize << targets;
        let u = haar_unitary(d, &mut rng).unwrap();
        let matrix: Vec<C64> = (0..d * d).map(|i| u[(i / d, i % d)]).collect();
        let mut inst = local(Unitary, controls, targets, &[]);
        inst.matrix = Some(Arc::new(matrix));
        cases.push(inst);
    }
    for inst in &cases {
        let k = inst.controls.len() + inst.targets.len();
        let map = table(inst);
        let u = dense_gate(inst, k);
        check_table(&map, k, 1e-14, heisenberg_gate(&u));
    }
    // CP(π) = CZ: a signed permutation after snapping.
    assert!(table(&local(P, 1, 1, &[PI])).is_permutation());
    // CCX permutes strings up to four images.
    let ccx = table(&local(X, 2, 1, &[]));
    assert!(!ccx.is_permutation());
    assert!((0..64).all(|b| ccx.images(b).len() <= 4));
}

#[test]
fn gates_beyond_four_qubits_are_refused_and_identities_skipped() {
    let mut c = Circuit::new(6);
    c.mcx(&[0, 1, 2, 3], 4);
    let inst = &c.instructions()[0];
    let err = compile_instruction(inst, &[], 6, &mut TableCache::default()).unwrap_err();
    assert!(
        matches!(&err, QsoError::InvalidArgument { name: "circuit", reason } if reason.contains("acts on 5 qubits")),
        "{err}"
    );
    let mut c = Circuit::new(2);
    c.id(1);
    let op = compile_instruction(&c.instructions()[0], &[], 2, &mut TableCache::default());
    assert!(op.unwrap().is_none());
}

#[test]
fn tables_are_shared_by_key_and_independent_of_the_qubits() {
    let mut c = Circuit::new(5);
    c.rz(0, 0.3).rz(4, 0.3).rz(4, 0.4).cx(0, 1).cx(3, 2).h(2);
    let mut cache = TableCache::default();
    let ops = compile_layers(&[Layer::gates(c)], 5, &mut cache).unwrap();
    assert_eq!(ops.len(), 6);
    assert_eq!(cache.len(), 4);
    let maps: Vec<&Arc<LocalMap>> = ops
        .iter()
        .map(|op| match op {
            Op::Local { map, .. } => map,
            Op::Diag(_) => unreachable!(),
        })
        .collect();
    assert!(Arc::ptr_eq(maps[0], maps[1]) && !Arc::ptr_eq(maps[1], maps[2]));
    assert!(Arc::ptr_eq(maps[3], maps[4]));
    match &ops[4] {
        Op::Local { qubits, .. } => assert_eq!(qubits, &[3, 2]),
        Op::Diag(_) => unreachable!(),
    }
}

// ---------------------------------------------------------------------
// Channel tables
// ---------------------------------------------------------------------

#[test]
fn channel_tables_match_the_dense_adjoint() {
    let mut rng = Rng::seed_from_u64(1640);
    // A random channel on two qubits: a unitary on them and one
    // environment qubit (least significant), started in |0⟩.
    let v = haar_unitary(8, &mut rng).unwrap();
    let kraus: Vec<Vec<C64>> = (0..2)
        .map(|e| {
            (0..16)
                .map(|i| v[(((i / 4) << 1) | e, (i % 4) << 1)])
                .collect()
        })
        .collect();
    let channels = [
        Channel::amplitude_damping(0.2).unwrap(),
        Channel::thermal_relaxation(50.0, 70.0, 10.0).unwrap(),
        Channel::depolarizing(0.1).unwrap(),
        Channel::depolarizing_2q(0.06).unwrap(),
        Channel::from_kraus("random", kraus).unwrap(),
        Channel::pauli(&[("XZY", 0.1), ("IIZ", 0.05)]).unwrap(),
    ];
    for channel in &channels {
        let k = channel.n_qubits();
        let d = 1usize << k;
        let map = channel_table(channel);
        let ops: Vec<CMatrix> = channel
            .kraus_operators()
            .iter()
            .map(|m| CMatrix::from_row_slice(d, d, m))
            .collect();
        check_table(&map, k, 1e-14, |p| {
            ops.iter()
                .fold(CMatrix::zeros(d, d), |acc, kr| acc + kr.adjoint() * p * kr)
        });
        let pauli = channel.pauli_probabilities().is_some();
        assert_eq!(map.is_permutation(), pauli, "{}", channel.name());
        // Products of one-qubit channels and Pauli channels keep supports;
        // a generic two-qubit channel spreads Z0 onto qubit 1.
        assert_eq!(
            map.grows(),
            channel.name() == "random",
            "{}",
            channel.name()
        );
    }
    // Amplitude damping maps Z to (1 − γ)Z + γI and X to √(1 − γ) X.
    let ad = channel_table(&channels[0]);
    assert_eq!(ad.images(0b10).len(), 2);
    assert_eq!(ad.images(0b10)[0].0, 0);
    assert!((ad.images(0b10)[0].1 - 0.2).abs() < 1e-15);
    assert!((ad.images(0b10)[1].1 - 0.8).abs() < 1e-15);
    assert_eq!(ad.images(0b01).len(), 1);
    assert!((ad.images(0b01)[0].1 - 0.8f64.sqrt()).abs() < 1e-15);
}

// ---------------------------------------------------------------------
// 5–6. Small noisy circuits
// ---------------------------------------------------------------------

fn z(q: usize, n: usize) -> PauliOperator {
    PauliOperator::parse(n, &format!("Z{q}")).unwrap()
}

#[test]
fn rotation_then_amplitude_damping() {
    let layers = [
        circuit(1, |c| {
            c.ry(0, 0.7);
        }),
        Layer::channel(Channel::amplitude_damping(0.2).unwrap(), &[0]),
    ];
    let value = run(&layers, &z(0, 1));
    assert!((value - 0.8118737498275908).abs() <= 1e-15, "{value}");
    assert!((value - (0.8 * 0.7f64.cos() + 0.2)).abs() <= 1e-15);
}

#[test]
fn damping_relaxation_and_depolarising_match_closed_forms() {
    let ad = Layer::channel(Channel::amplitude_damping(0.3).unwrap(), &[0]);
    let x = circuit(1, |c| {
        c.x(0);
    });
    let value = run(&[x.clone(), ad], &z(0, 1));
    assert!((value + 0.4).abs() <= 1e-15, "{value}");

    let relax = Layer::channel(Channel::thermal_relaxation(50.0, 70.0, 10.0).unwrap(), &[0]);
    let h = circuit(1, |c| {
        c.h(0);
    });
    let xobs = PauliOperator::parse(1, "X0").unwrap();
    let value = run(&[h, relax.clone()], &xobs);
    assert!((value - 0.8668778997501816).abs() <= 1e-15, "{value}");
    assert!((value - (-1.0f64 / 7.0).exp()).abs() <= 1e-15);
    let value = run(&[x, relax], &z(0, 1));
    assert!((value + 0.6374615061559636).abs() <= 1e-15, "{value}");
    assert!((value - (1.0 - 2.0 * (-0.2f64).exp())).abs() <= 1e-15);

    // On a wide register: the Bell pair on qubits 70 and 3.
    let n = 130;
    let bell = circuit(n, |c| {
        c.h(70).cx(70, 3);
    });
    let depol = Layer::channel(Channel::depolarizing_2q(0.06).unwrap(), &[70, 3]);
    let xx = PauliOperator::parse(n, "X70 X3").unwrap();
    let value = run(&[bell, depol], &xx);
    assert!((value - 0.936).abs() <= 1e-15, "{value}");
}

#[test]
fn pauli_lindblad_layers_scale_as_density_matrices() {
    // Reference values: NumPy density matrices (dense_refs.py of the
    // design notes).
    let rates = PauliOperator::parse(3, "0.1 X0 + 0.05 Z0 Z1 + 0.02 Y1 + 0.03 X1 Z2").unwrap();
    let before = circuit(3, |c| {
        c.h(0).cx(0, 1).ry(2, 0.7);
    });
    let after = circuit(3, |c| {
        c.rx(1, 0.4).cx(1, 2);
    });
    let ad = Layer::channel(Channel::amplitude_damping(0.2).unwrap(), &[2]);
    let obs = PauliOperator::parse(3, "Z0 Z1 + 0.5 X0 + 0.25 X1 X2 - 0.3 Z2").unwrap();
    let value = |factors: &[f64]| {
        let mut layers = vec![before.clone()];
        layers.extend(
            factors
                .iter()
                .map(|&a| Layer::pauli_lindblad(rates.clone(), a)),
        );
        layers.extend([after.clone(), ad.clone()]);
        run(&layers, &obs)
    };
    for (factors, expected) in [
        (&[0.0][..], 0.8610609940028848),
        (&[1.0], 0.6223387667165513),
        (&[2.5], 0.37507840660135505),
        (&[-1.0], 1.1833022950695018),
        (&[-0.5], 1.0101202024739016),
        (&[1.0, -1.0], 0.8610609940028842),
    ] {
        let v = value(factors);
        assert!(
            (v - expected).abs() <= 1e-14,
            "{factors:?}: {v} vs {expected}"
        );
    }
    let ops = compile_layers(
        &[Layer::pauli_lindblad(rates.clone(), -1.0)],
        3,
        &mut TableCache::default(),
    )
    .unwrap();
    match &ops[0] {
        Op::Diag(d) => {
            assert_eq!(d.len(), 4);
            assert!((d.gamma() - 0.4f64.exp()).abs() < 1e-15);
            assert_eq!(d.support(1), vec![0b111]);
        }
        Op::Local { .. } => unreachable!(),
    }
}

#[test]
fn layer_errors() {
    let mut cache = TableCache::default();
    let wide = circuit(5, |c| {
        c.h(4);
    });
    assert!(matches!(
        compile_layers(&[wide], 4, &mut cache),
        Err(QsoError::DimensionMismatch {
            expected: 4,
            actual: 5
        })
    ));
    let mut c = Circuit::new(2);
    let t = c.new_parameter();
    c.rx(0, t);
    assert!(matches!(
        compile_layers(&[Layer::gates(c.clone())], 2, &mut cache),
        Err(QsoError::ParameterCount {
            expected: 1,
            actual: 0
        })
    ));
    assert!(
        compile_layers(
            &[Layer::Gates {
                circuit: c,
                theta: vec![0.3]
            }],
            2,
            &mut cache
        )
        .is_ok()
    );
    let ad = Channel::amplitude_damping(0.1).unwrap();
    for (qubits, ok) in [(&[0usize][..], true), (&[2], false), (&[0, 1], false)] {
        let r = compile_layers(&[Layer::channel(ad.clone(), qubits)], 2, &mut cache);
        assert_eq!(r.is_ok(), ok, "{qubits:?}");
    }
    let dep = Channel::depolarizing_2q(0.1).unwrap();
    assert!(matches!(
        compile_layers(&[Layer::channel(dep, &[1, 1])], 2, &mut cache),
        Err(QsoError::DuplicateQubit {
            gate: "channel",
            qubit: 1
        })
    ));
    let rates = PauliOperator::parse(2, "0.1 X0").unwrap();
    assert!(matches!(
        compile_layers(
            &[Layer::pauli_lindblad(rates.clone(), f64::NAN)],
            2,
            &mut cache
        ),
        Err(QsoError::InvalidArgument { name: "factor", .. })
    ));
    assert!(matches!(
        compile_layers(&[Layer::pauli_lindblad(rates, 1.0)], 3, &mut cache),
        Err(QsoError::DimensionMismatch {
            expected: 3,
            actual: 2
        })
    ));
    for bad in ["0.1 X0 + 0.2", "-0.1 X0"] {
        let rates = PauliOperator::parse(2, bad).unwrap();
        assert!(matches!(
            compile_layers(&[Layer::pauli_lindblad(rates, 1.0)], 2, &mut cache),
            Err(QsoError::InvalidArgument { name: "rates", .. })
        ));
    }
    // A non-unitary matrix pushed past validation is refused.
    let mut c = Circuit::new(1);
    c.push_trusted(Instruction {
        kind: GateKind::Unitary,
        targets: vec![0],
        controls: vec![],
        params: vec![],
        matrix: Some(Arc::new(vec![
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.5, 0.0),
        ])),
    });
    assert!(matches!(
        compile_layers(&[Layer::gates(c)], 1, &mut cache),
        Err(QsoError::InvalidArgument { name: "matrix", .. })
    ));
}

#[test]
fn options_check_their_ranges() {
    assert!(PropagationOptions::default().validate().is_ok());
    let o = PropagationOptions::default()
        .with_min_coefficient(1e-3)
        .with_max_weight(3)
        .with_max_terms(10)
        .with_operator(true);
    assert_eq!(
        (
            o.min_coefficient,
            o.max_weight,
            o.max_terms,
            o.keep_operator
        ),
        (1e-3, Some(3), 10, true)
    );
    assert!(o.validate().is_ok());
    for bad in [
        PropagationOptions::default().with_min_coefficient(-1.0),
        PropagationOptions::default().with_min_coefficient(f64::INFINITY),
        PropagationOptions::default().with_max_terms(0),
        PropagationOptions::default().with_max_terms(MAX_PROPAGATION_TERMS + 1),
    ] {
        assert!(bad.validate().is_err(), "{bad:?}");
    }
    let r = PropagationResult {
        value: 0.25,
        truncation_bound: 0.0,
        terms: 1,
        peak_terms: 1,
        dropped_terms: 0,
        operator: None,
    };
    assert_eq!(r.estimate().value, 0.25);
    assert_eq!(r.estimate().std_error, 0.0);
}

// ---------------------------------------------------------------------
// Tables on register qubits against the simulators
// ---------------------------------------------------------------------

/// A random instruction of the whole gate set (controls included, at
/// most four qubits) on distinct random qubits of `n`; angles are random
/// or multiples of π/2, so that some tables are snapped.
fn random_instruction(n: usize, rng: &mut Rng) -> Instruction {
    use GateKind::*;
    let kinds = [
        H, X, Y, Z, S, Sdg, T, Tdg, SX, SXdg, RX, RY, RZ, P, U, Swap, RXX, RYY, RZZ, Unitary,
    ];
    let kind = kinds[rng.below(kinds.len() as u64) as usize];
    let targets = match kind.num_targets() {
        Some(t) => t,
        None => 1 + rng.below(3.min(n) as u64) as usize,
    };
    let controls =
        rng.below((MAX_TRANSFER_QUBITS - targets + 1).min(n - targets + 1) as u64) as usize;
    let mut qubits: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut qubits);
    let angle = |rng: &mut Rng| {
        if rng.below(3) == 0 {
            (rng.below(8) as f64 - 4.0) * FRAC_PI_2
        } else {
            rng.uniform_range(-PI, PI)
        }
    };
    let params = (0..kind.num_params())
        .map(|_| Param::Value(angle(rng)))
        .collect();
    let matrix = (kind == Unitary).then(|| {
        let d = 1usize << targets;
        let u = haar_unitary(d, rng).unwrap();
        Arc::new((0..d * d).map(|i| u[(i / d, i % d)]).collect::<Vec<C64>>())
    });
    Instruction {
        kind,
        targets: qubits[controls..controls + targets].to_vec(),
        controls: qubits[..controls].to_vec(),
        params,
        matrix,
    }
}

/// A random sum of a few Pauli strings on `n ≤ 64` qubits.
fn random_observable(n: usize, terms: usize, rng: &mut Rng) -> PauliSum {
    let letters = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z];
    let terms: Vec<(f64, PauliString)> = (0..terms)
        .map(|_| {
            let factors: Vec<(usize, Pauli)> = (0..n)
                .map(|q| (q, letters[rng.below(4) as usize]))
                .collect();
            (
                rng.uniform_range(-1.0, 1.0),
                PauliString::new(&factors).unwrap(),
            )
        })
        .collect();
    PauliSum::from_terms(terms)
}

#[test]
fn random_circuits_on_register_qubits_match_the_state_vector() {
    let mut rng = Rng::seed_from_u64(16_401);
    for trial in 0..40 {
        let n = 2 + trial % 5;
        let mut c = Circuit::new(n);
        for _ in 0..25 {
            c.try_push(random_instruction(n, &mut rng)).unwrap();
        }
        let sum = random_observable(n, 6, &mut rng);
        let psi = StateVector::from_circuit(&c, &[]).unwrap();
        let expected = sum.expectation(&psi).unwrap();
        let value = run(
            &[Layer::gates(c)],
            &PauliOperator::from_sum(n, &sum).unwrap(),
        );
        assert!(
            (value - expected).abs() <= 1e-12,
            "trial {trial}: {value} vs {expected}"
        );
    }
}

/// A random channel on `k` qubits: a Haar unitary on them and one
/// environment qubit (least significant) started in `|0⟩`.
fn random_channel(k: usize, rng: &mut Rng) -> Channel {
    let d = 1usize << k;
    let v = haar_unitary(2 * d, rng).unwrap();
    let kraus = (0..2)
        .map(|e| {
            (0..d * d)
                .map(|i| v[(((i / d) << 1) | e, (i % d) << 1)])
                .collect()
        })
        .collect();
    Channel::from_kraus("random", kraus).unwrap()
}

#[test]
fn channels_on_register_qubits_match_the_density_matrix() {
    let mut rng = Rng::seed_from_u64(16_402);
    for trial in 0..12 {
        let n = 4 + trial % 2;
        let mut prepare = Circuit::new(n);
        for _ in 0..12 {
            prepare.try_push(random_instruction(n, &mut rng)).unwrap();
        }
        let mut rho = crate::noise::DensityMatrix::from_circuit(&prepare, &[]).unwrap();
        let mut layers = vec![Layer::gates(prepare)];
        for step in 0..6 {
            let channel = match step {
                0 => Channel::amplitude_damping(0.3).unwrap(),
                1 => Channel::thermal_relaxation(50.0, 30.0, 12.0).unwrap(),
                2 => Channel::pauli(&[("XZY", 0.1), ("IYZ", 0.05)]).unwrap(),
                3 => Channel::rotation("XZ", 0.4).unwrap(),
                _ => random_channel(1 + rng.below(4) as usize, &mut rng),
            };
            let mut qubits: Vec<usize> = (0..n).collect();
            rng.shuffle(&mut qubits);
            qubits.truncate(channel.n_qubits());
            rho.apply_channel(&channel, &qubits).unwrap();
            layers.push(Layer::channel(channel, &qubits));
            let mut c = Circuit::new(n);
            c.try_push(random_instruction(n, &mut rng)).unwrap();
            rho.apply_circuit(&c, &[], None).unwrap();
            layers.push(Layer::gates(c));
        }
        let sum = random_observable(n, 6, &mut rng);
        let expected = rho.expectation(&sum).unwrap();
        let value = run(&layers, &PauliOperator::from_sum(n, &sum).unwrap());
        assert!(
            (value - expected).abs() <= 1e-12,
            "trial {trial}: {value} vs {expected}"
        );
    }
}

// ---------------------------------------------------------------------
// 7–15. The engine: schedule, light cone, passes, truncation, readout
// ---------------------------------------------------------------------

fn opts() -> PropagationOptions {
    PropagationOptions::default()
}

#[test]
fn mid_circuit_pauli_lindblad_layer_through_the_engine() {
    // §6.1 row 7 directly through `propagate`, with an explicit `|0⟩`
    // product state as well (Bloch vector (0, 0, 1)).
    let rates = PauliOperator::parse(3, "0.1 X0 + 0.05 Z0 Z1 + 0.02 Y1 + 0.03 X1 Z2").unwrap();
    let before = circuit(3, |c| {
        c.h(0).cx(0, 1).ry(2, 0.7);
    });
    let after = circuit(3, |c| {
        c.rx(1, 0.4).cx(1, 2);
    });
    let ad = Layer::channel(Channel::amplitude_damping(0.2).unwrap(), &[2]);
    let obs = PauliOperator::parse(3, "Z0 Z1 + 0.5 X0 + 0.25 X1 X2 - 0.3 Z2").unwrap();
    let zero = [[0.0, 0.0, 1.0]; 3];
    for (factors, expected) in [
        (&[0.0][..], 0.8610609940028848),
        (&[1.0], 0.6223387667165513),
        (&[2.5], 0.37507840660135505),
        (&[-1.0], 1.1833022950695018),
        (&[-0.5], 1.0101202024739016),
        (&[1.0, -1.0], 0.8610609940028842),
    ] {
        let mut layers = vec![before.clone()];
        layers.extend(
            factors
                .iter()
                .map(|&a| Layer::pauli_lindblad(rates.clone(), a)),
        );
        layers.extend([after.clone(), ad.clone()]);
        for state in [None, Some(&zero[..])] {
            let r = propagate(&layers, &obs, state, &opts()).unwrap();
            assert!(
                (r.value - expected).abs() <= 1e-14,
                "{factors:?}: {} vs {expected}",
                r.value
            );
            assert!(r.truncation_bound <= 1e-14);
            assert_eq!(r.dropped_terms, 0);
        }
    }
}

#[test]
fn inverted_layers_amplify_the_bound_only_on_the_light_cone() {
    // Dropping Z0 before an inverted X0 (λ = 0.1) makes the error e^{0.2}:
    // the bound is tight, and the generator X1 outside the cone of Z0 does
    // not enter it (e^{0.8} with the full rates).
    let rates = PauliOperator::parse(2, "0.1 X0 + 0.3 X1").unwrap();
    let layers = [Layer::pauli_lindblad(rates, -1.0)];
    let z0 = PauliOperator::parse(2, "Z0").unwrap();
    let exact = propagate(&layers, &z0, None, &opts()).unwrap();
    assert!(
        (exact.value - 0.2f64.exp()).abs() <= 1e-15,
        "{}",
        exact.value
    );
    let cut = propagate(&layers, &z0, None, &opts().with_min_coefficient(2.0)).unwrap();
    assert_eq!(cut.value, 0.0);
    assert_eq!((cut.dropped_terms, cut.terms), (1, 0));
    assert!(
        (cut.truncation_bound - 1.2214027581601699).abs() <= 1e-15,
        "{}",
        cut.truncation_bound
    );
    assert!(((exact.value - cut.value).abs() - cut.truncation_bound).abs() <= 1e-15);
}

#[test]
fn light_cone_follows_the_letters_strings_can_carry() {
    // A gate that commutes with every string it can meet is removed, and
    // a generator keeps only the factors that some possible letter
    // anticommutes with: Z0 passes CZ(0, 1) unchanged, so of the rates
    // 0.1·X0 X1 + 0.2·Z0 X1 only 0.1·X0 acts, and the bound of dropping
    // Z0 is e^{0.2} (the error), not e^{0.6}.
    let rates = PauliOperator::parse(2, "0.1 X0 X1 + 0.2 Z0 X1").unwrap();
    let cz = circuit(2, |c| {
        c.cz(0, 1);
    });
    let layers = [Layer::pauli_lindblad(rates, -1.0), cz];
    let z0 = PauliOperator::parse(2, "Z0").unwrap();
    let exact = propagate(&layers, &z0, None, &opts()).unwrap();
    assert!(
        (exact.value - 0.2f64.exp()).abs() <= 1e-15,
        "{}",
        exact.value
    );
    let cut = propagate(&layers, &z0, None, &opts().with_min_coefficient(2.0)).unwrap();
    assert!(
        (cut.truncation_bound - 0.2f64.exp()).abs() <= 1e-15,
        "{}",
        cut.truncation_bound
    );

    // Kicked Ising on heavy-hex(7): the light cone of Z57 has 1, 3, 5, 9,
    // 13, 20 and 26 qubits after 1–7 steps (Aer's reduced circuits), not
    // the 5 … 101 qubits of every gate touching it.
    let data: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/data/heavy_hex_7.json")).unwrap();
    let n = data["n_qubits"].as_u64().unwrap() as usize;
    let obs = PauliOperator::parse(n, "Z57").unwrap();
    for (steps, (cone, gates)) in [
        (1, 1),
        (3, 6),
        (5, 15),
        (9, 32),
        (13, 57),
        (20, 97),
        (26, 149),
    ]
    .into_iter()
    .enumerate()
    {
        let mut c = Circuit::new(n);
        for _ in 0..=steps {
            for q in 0..n {
                c.rx(q, 0.3);
            }
            for class in data["colours"].as_array().unwrap() {
                for e in class.as_array().unwrap() {
                    let (a, b) = (e[0].as_u64().unwrap(), e[1].as_u64().unwrap());
                    c.rzz(a as usize, b as usize, -FRAC_PI_2);
                }
            }
        }
        let mut cache = TableCache::default();
        let ops = compile_layers(&[Layer::gates(c)], n, &mut cache).unwrap();
        let plan = super::compile::plan(ops, &obs).unwrap();
        let kept: usize = plan
            .passes
            .iter()
            .flat_map(|p| p.levels.iter().map(|l| l.locals.len()))
            .sum();
        assert_eq!(
            (plan.labels.len(), kept, plan.words),
            (cone, gates, 1),
            "{} steps",
            steps + 1
        );
    }
}

/// `ρ ↦ Π_k (w_k ρ + (1 − w_k) P_k ρ P_k)`, `w_k = (1 + e^{−2αλ_k})/2`, on
/// a density matrix for any sign of `α` (a linear map, not a channel for
/// `α < 0`).
fn apply_pauli_lindblad(rho: &mut crate::noise::DensityMatrix, rates: &PauliOperator, alpha: f64) {
    let n = rates.n_qubits();
    for (lambda, factors) in rates.terms() {
        let w = 0.5 * (1.0 + (-2.0 * alpha * lambda).exp());
        let mut flip = Circuit::new(n);
        for &(q, p) in &factors {
            match p {
                Pauli::X => flip.x(q),
                Pauli::Y => flip.y(q),
                Pauli::Z => flip.z(q),
                Pauli::I => unreachable!("no identity factors"),
            };
        }
        let mut flipped = rho.clone();
        flipped.apply_circuit(&flip, &[], None).unwrap();
        rho.scale(w);
        rho.add_scaled(&flipped, 1.0 - w);
    }
}

#[test]
fn inverted_layers_match_the_density_matrix_and_the_bound_holds() {
    // Random noisy circuits with Pauli–Lindblad layers at α < 0, against
    // a density matrix that applies the inverted layers as linear maps;
    // truncated runs stay within their bound.
    let mut rng = Rng::seed_from_u64(16_413);
    let mut cut_cases = 0;
    for trial in 0..12 {
        let n = 3 + trial % 4;
        let (rho0, bloch) = product_state(n, trial % 2 == 1, &mut rng);
        let (mut rho, state) = if trial % 3 == 0 {
            (crate::noise::DensityMatrix::zero(n).unwrap(), None)
        } else {
            (rho0, Some(bloch))
        };
        let alphas = [-1.0, -0.5, -2.0];
        let mut rates = PauliOperator::new(n).unwrap();
        let letters = [Pauli::X, Pauli::Y, Pauli::Z];
        for _ in 0..4 {
            let a = rng.below(n as u64) as usize;
            let b = (a + 1 + rng.below(n as u64 - 1) as usize) % n;
            let mut factors = vec![(a, letters[rng.below(3) as usize])];
            if rng.below(2) == 0 {
                factors.push((b, letters[rng.below(3) as usize]));
            }
            rates
                .add_term(rng.uniform_range(0.01, 0.08), &factors)
                .unwrap();
        }
        let mut layers = Vec::new();
        for step in 0..6 {
            let mut c = Circuit::new(n);
            for _ in 0..n {
                c.try_push(random_instruction(n, &mut rng)).unwrap();
            }
            rho.apply_circuit(&c, &[], None).unwrap();
            layers.push(Layer::gates(c));
            let (ch, qubits) = any_channel(n, &mut rng);
            rho.apply_channel(&ch, &qubits).unwrap();
            layers.push(Layer::channel(ch, &qubits));
            let alpha = alphas[(trial + step) % 3];
            apply_pauli_lindblad(&mut rho, &rates, alpha);
            layers.push(Layer::pauli_lindblad(rates.clone(), alpha));
        }
        let sum = random_observable(n, 5, &mut rng);
        let expected = rho.expectation(&sum).unwrap();
        let obs = PauliOperator::from_sum(n, &sum).unwrap();
        let exact = propagate(&layers, &obs, state.as_deref(), &opts()).unwrap();
        assert!(
            (exact.value - expected).abs() <= 1e-11 * expected.abs().max(1.0),
            "trial {trial}: {} vs density matrix {expected}",
            exact.value
        );
        for tau in [1e-3, 1e-2, 5e-2, 0.2] {
            for w in [None, Some(1), Some(2)] {
                let mut o = opts().with_min_coefficient(tau);
                if let Some(w) = w {
                    o = o.with_max_weight(w);
                }
                let r = propagate(&layers, &obs, state.as_deref(), &o).unwrap();
                let err = (r.value - expected).abs();
                assert!(
                    err <= r.truncation_bound + 1e-11,
                    "trial {trial}, τ = {tau}, w = {w:?}: error {err:e} > bound {:e}",
                    r.truncation_bound
                );
                cut_cases += (r.dropped_terms > 0) as usize;
            }
        }
    }
    assert!(cut_cases > 12 * 12 / 2, "{cut_cases} truncated");
}

/// A product state of Bloch vectors `(sin θ cos φ, sin θ sin φ, cos θ)·s`
/// prepared on a density matrix by `RY(θ)`, `RZ(φ)` and a depolarising
/// channel that shrinks the vector by `s = 1 − 4p/3`.
fn product_state(
    n: usize,
    mixed: bool,
    rng: &mut Rng,
) -> (crate::noise::DensityMatrix, Vec<[f64; 3]>) {
    let mut prep = Circuit::new(n);
    let mut bloch = Vec::with_capacity(n);
    let mut shrink = Vec::with_capacity(n);
    for q in 0..n {
        let theta = rng.uniform_range(0.0, PI);
        let phi = rng.uniform_range(-PI, PI);
        prep.ry(q, theta).rz(q, phi);
        let p = if mixed {
            rng.uniform_range(0.0, 0.5)
        } else {
            0.0
        };
        let s = 1.0 - 4.0 * p / 3.0;
        bloch.push([
            s * theta.sin() * phi.cos(),
            s * theta.sin() * phi.sin(),
            s * theta.cos(),
        ]);
        shrink.push(p);
    }
    let mut rho = crate::noise::DensityMatrix::from_circuit(&prep, &[]).unwrap();
    for (q, &p) in shrink.iter().enumerate() {
        if p > 0.0 {
            rho.apply_channel(&Channel::depolarizing(p).unwrap(), &[q])
                .unwrap();
        }
    }
    (rho, bloch)
}

/// A random channel of every kind the tests know, on random qubits.
fn any_channel(n: usize, rng: &mut Rng) -> (Channel, Vec<usize>) {
    let channel = match rng.below(9) {
        0 => Channel::amplitude_damping(rng.uniform_range(0.0, 0.4)).unwrap(),
        1 => Channel::thermal_relaxation(50.0, 30.0, rng.uniform_range(1.0, 20.0)).unwrap(),
        2 => Channel::depolarizing(rng.uniform_range(0.0, 0.2)).unwrap(),
        3 => Channel::depolarizing_2q(rng.uniform_range(0.0, 0.2)).unwrap(),
        4 => Channel::phase_damping(rng.uniform_range(0.0, 0.3)).unwrap(),
        5 => Channel::pauli(&[("XZ", 0.05), ("YY", 0.03), ("IZ", 0.02)]).unwrap(),
        6 => Channel::rotation("ZX", rng.uniform_range(-0.5, 0.5)).unwrap(),
        7 => Channel::bit_flip(rng.uniform_range(0.0, 0.2)).unwrap(),
        _ => random_channel(1 + rng.below(3) as usize, rng),
    };
    let mut qubits: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut qubits);
    qubits.truncate(channel.n_qubits());
    (channel, qubits)
}

#[test]
fn truncation_bound_holds_on_random_noisy_circuits() {
    let mut rng = Rng::seed_from_u64(16_409);
    let mut cases = 0;
    let mut cut_cases = 0;
    for trial in 0..24 {
        let n = 3 + trial % 5;
        let state_kind = trial % 3;
        let (mut rho, bloch) = product_state(n, state_kind == 2, &mut rng);
        let state = (state_kind != 0).then_some(bloch);
        if state_kind == 0 {
            rho = crate::noise::DensityMatrix::zero(n).unwrap();
        }
        let alpha = [0.0, 1.0, 2.0, -1.0][trial % 4];
        let rates = PauliOperator::parse(
            n,
            &format!(
                "0.05 X0 + 0.02 Z{} + 0.03 Y1 Z{} + 0.01 X{}",
                n - 1,
                n - 1,
                n / 2
            ),
        )
        .unwrap();
        let mut layers = Vec::new();
        for step in 0..8 {
            let mut c = Circuit::new(n);
            for _ in 0..n {
                c.try_push(random_instruction(n, &mut rng)).unwrap();
            }
            rho.apply_circuit(&c, &[], None).unwrap();
            layers.push(Layer::gates(c));
            if step % 3 == 1 {
                // The Pauli–Lindblad layer as density-matrix channels
                // (α ≥ 0) or as the linear map of the inverted layer on
                // the density matrix (α < 0).
                layers.push(Layer::pauli_lindblad(rates.clone(), alpha));
                if alpha < 0.0 {
                    apply_pauli_lindblad(&mut rho, &rates, alpha);
                } else {
                    let (keys, coefs) = rates.rows();
                    let w = rates.words();
                    for (i, &lambda) in coefs.iter().enumerate() {
                        let key = &keys[2 * w * i..2 * w * (i + 1)];
                        let mut label = String::new();
                        let mut qubits = Vec::new();
                        for q in 0..n {
                            let (x, z) = ((key[0] >> q) & 1, (key[w] >> q) & 1);
                            let p = match (x, z) {
                                (0, 0) => continue,
                                (1, 0) => 'X',
                                (0, 1) => 'Z',
                                _ => 'Y',
                            };
                            label.push(p);
                            qubits.push(q);
                        }
                        let flip = 0.5 * (1.0 - (-2.0 * alpha * lambda).exp());
                        if flip > 0.0 {
                            let ch = Channel::pauli(&[(label.as_str(), flip)]).unwrap();
                            rho.apply_channel(&ch, &qubits).unwrap();
                        }
                    }
                }
            } else {
                let (ch, qubits) = any_channel(n, &mut rng);
                rho.apply_channel(&ch, &qubits).unwrap();
                layers.push(Layer::channel(ch, &qubits));
            }
        }
        let sum = random_observable(n, 5, &mut rng);
        let obs = PauliOperator::from_sum(n, &sum).unwrap();
        let exact = propagate(&layers, &obs, state.as_deref(), &opts()).unwrap();
        // Review: the reference is the density matrix for every α (it
        // was the untruncated propagation itself for α < 0).
        let expected = rho.expectation(&sum).unwrap();
        assert!(
            (exact.value - expected).abs() <= 1e-12,
            "trial {trial}: {} vs density matrix {expected}",
            exact.value
        );
        for tau in [1e-3, 1e-2, 5e-2] {
            for w in [None, Some(2), Some(3)] {
                let mut o = opts().with_min_coefficient(tau);
                if let Some(w) = w {
                    o = o.with_max_weight(w);
                }
                let r = propagate(&layers, &obs, state.as_deref(), &o).unwrap();
                let err = (r.value - expected).abs();
                assert!(
                    err <= r.truncation_bound + 1e-14,
                    "trial {trial}, τ = {tau}, w = {w:?}: error {err:e} > bound {:e}",
                    r.truncation_bound
                );
                cases += 1;
                cut_cases += (r.dropped_terms > 0) as usize;
            }
        }
    }
    assert_eq!(cases, 24 * 9);
    assert!(
        cut_cases > cases / 2,
        "{cut_cases} of {cases} cases truncated"
    );
}

#[test]
fn exact_cancellations_leave_the_input_string_for_string() {
    // CH·CH = I: the second pass cancels every string the first one made.
    let mut c = Circuit::new(2);
    c.ch(0, 1).ch(0, 1);
    let obs = PauliOperator::parse(2, "Z0 Z1 + X1").unwrap();
    let r = propagate(&[Layer::gates(c)], &obs, None, &opts().with_operator(true)).unwrap();
    let op = r.operator.unwrap();
    assert_eq!(op.len(), obs.len(), "{op}");
    for ((a, fa), (b, fb)) in op.terms().zip(obs.terms()) {
        assert_eq!(fa, fb);
        assert!((a - b).abs() <= 1e-15, "{a} vs {b}");
    }
    assert!(r.truncation_bound <= 1e-14, "{}", r.truncation_bound);
    assert_eq!(r.terms, 2);
}

/// A random circuit of `layers` layers on `n` qubits: in each layer about
/// `n/2` random instructions of the whole gate set; with `clifford`, only
/// Clifford gates except `rotations` random rotations.
fn random_circuit(
    n: usize,
    layers: usize,
    clifford: bool,
    rotations: usize,
    rng: &mut Rng,
) -> Circuit {
    let mut c = Circuit::new(n);
    let total = layers * n.div_ceil(2);
    let mut rotation_at: Vec<usize> = (0..total).collect();
    rng.shuffle(&mut rotation_at);
    rotation_at.truncate(rotations);
    for g in 0..total {
        if !clifford || rotation_at.contains(&g) {
            c.try_push(random_instruction(n, rng)).unwrap();
            continue;
        }
        let a = rng.below(n as u64) as usize;
        let b = (a + 1 + rng.below(n as u64 - 1) as usize) % n;
        match rng.below(7) {
            0 => c.h(a),
            1 => c.s(a),
            2 => c.sx(a),
            3 => c.cx(a, b),
            4 => c.cz(a, b),
            5 => c.swap(a, b),
            _ => c.sdg(a),
        };
    }
    c
}

/// A random sum of strings of weight at most `max_weight` on `n` qubits.
fn sparse_observable(n: usize, terms: usize, max_weight: usize, rng: &mut Rng) -> PauliSum {
    let letters = [Pauli::X, Pauli::Y, Pauli::Z];
    let terms: Vec<(f64, PauliString)> = (0..terms)
        .map(|_| {
            let mut qubits: Vec<usize> = (0..n).collect();
            rng.shuffle(&mut qubits);
            let w = 1 + rng.below(max_weight as u64) as usize;
            let factors: Vec<(usize, Pauli)> = qubits[..w]
                .iter()
                .map(|&q| (q, letters[rng.below(3) as usize]))
                .collect();
            (
                rng.uniform_range(-1.0, 1.0),
                PauliString::new(&factors).unwrap(),
            )
        })
        .collect();
    PauliSum::from_terms(terms)
}

#[test]
fn larger_random_circuits_match_the_state_vector() {
    let mut rng = Rng::seed_from_u64(16_411);
    let check = |c: &Circuit, sum: &PauliSum, label: &str| {
        let n = c.n_qubits();
        let psi = StateVector::from_circuit(c, &[]).unwrap();
        let expected = sum.expectation(&psi).unwrap();
        let obs = PauliOperator::from_sum(n, sum).unwrap();
        let r = propagate(&[Layer::gates(c.clone())], &obs, None, &opts()).unwrap();
        assert!(
            (r.value - expected).abs() <= 1e-12,
            "{label}: {} vs {expected} ({} strings)",
            r.value,
            r.terms
        );
    };
    // Depth 20 on up to 8 qubits, the whole gate set.
    for n in [3, 5, 8] {
        let c = random_circuit(n, 20, false, 0, &mut rng);
        let sum = random_observable(n, 4, &mut rng);
        check(&c, &sum, &format!("{n} qubits, depth 20"));
    }
    // Depth 20 on 12 qubits: Cliffords and 6 other gates.
    for _ in 0..2 {
        let c = random_circuit(12, 20, true, 6, &mut rng);
        let sum = sparse_observable(12, 4, 3, &mut rng);
        check(&c, &sum, "12 qubits, depth 20, 6 rotations");
    }
    // 20 qubits: 20 gates of the whole gate set (on up to four random
    // qubits each), or Cliffords with 6 rotations at depth 20.
    let c = random_circuit(20, 2, false, 0, &mut rng);
    let sum = sparse_observable(20, 2, 2, &mut rng);
    check(&c, &sum, "20 qubits, 20 gates");
    let c = random_circuit(20, 20, true, 6, &mut rng);
    let sum = sparse_observable(20, 4, 3, &mut rng);
    check(&c, &sum, "20 qubits, depth 20, 6 rotations");
}

/// A device with every Markovian kind of noise: idle `T1`/`T2`, gate
/// durations, depolarising errors, Pauli and Kraus channels, spectators
/// and a fixed over-rotation.
fn noisy_device() -> DeviceModel {
    use crate::noise::{ChannelSpec, GateRule, QubitProperties};
    let kraus = random_channel(1, &mut Rng::seed_from_u64(7));
    DeviceModel::new("every kind")
        .with_default_qubit(QubitProperties::new().with_t1_t2(80.0, 60.0))
        .with_qubit(1, QubitProperties::new().with_t1_t2(40.0, 50.0))
        .with_rule(
            GateRule::for_gate("cx")
                .with_duration(3.0)
                .with_depolarizing(0.01)
                .with_channel(ChannelSpec::Pauli(
                    [("ZZ".to_string(), 0.01), ("XI".to_string(), 0.02)]
                        .into_iter()
                        .collect(),
                ))
                .with_spectator(&[2], ChannelSpec::PhaseFlip(0.03)),
        )
        .with_rule(
            GateRule::for_gate("rx")
                .with_duration(1.0)
                .with_over_rotation(0.02)
                .with_channel(ChannelSpec::from(&kraus)),
        )
        .with_rule(
            GateRule::any()
                .with_num_qubits(1)
                .with_duration(0.5)
                .with_channel(ChannelSpec::AmplitudeDamping(0.01)),
        )
}

#[test]
fn device_layers_match_the_density_matrix() {
    let device = noisy_device();
    let mut rng = Rng::seed_from_u64(16_412);
    for (n, gates) in [(4, 30), (6, 30), (8, 24), (10, 16), (12, 8)] {
        let mut c = Circuit::new(n);
        for _ in 0..gates {
            let a = rng.below(n as u64) as usize;
            let b = (a + 1 + rng.below(n as u64 - 1) as usize) % n;
            match rng.below(5) {
                0 => c.h(a),
                1 => c.rx(a, rng.uniform_range(-PI, PI)),
                2 => c.ry(a, rng.uniform_range(-PI, PI)),
                3 => c.t(a),
                _ => c.cx(a, b),
            };
        }
        let layers = device_layers(&c, &[], &device).unwrap();
        let mut rho = crate::noise::DensityMatrix::zero(n).unwrap();
        rho.apply_device(&c, &[], &device).unwrap();
        let sum = sparse_observable(n, 5, 3, &mut rng);
        let expected = rho.expectation(&sum).unwrap();
        let r = propagate(
            &layers,
            &PauliOperator::from_sum(n, &sum).unwrap(),
            None,
            &opts(),
        )
        .unwrap();
        assert!(
            (r.value - expected).abs() <= 1e-12,
            "{n} qubits: {} vs {expected}",
            r.value
        );
    }
}

#[test]
fn device_entries_beyond_the_simulators_on_this_path_only() {
    let mut device = DeviceModel::new("wide")
        .with_rule(crate::noise::GateRule::for_gate("cx").with_depolarizing(0.02));
    for q in 0..40 {
        device = device.with_qubit(
            q,
            crate::noise::QubitProperties::new().with_t1_t2(100.0 + q as f64, 80.0),
        );
    }
    assert!(device.validate().is_err());
    let mut c = Circuit::new(40);
    c.h(0);
    for q in 0..39 {
        c.cx(q, q + 1);
    }
    let layers = device_layers(&c, &[], &device).unwrap();
    assert!(layers.len() > 1);
    let obs = PauliOperator::parse(40, "Z38 Z39").unwrap();
    let r = propagate(&layers, &obs, None, &opts()).unwrap();
    // Two-qubit depolarising after each CX: ZZ on the last pair decays.
    assert!(r.value > 0.9 && r.value < 1.0, "{}", r.value);

    let json = {
        let mut d = DeviceModel::new("heavy hex").with_num_qubits(115);
        for q in 0..115 {
            d = d.with_qubit(
                q,
                crate::noise::QubitProperties::new().with_t1_t2(100.0, 90.0),
            );
        }
        d.to_json()
    };
    assert!(DeviceModel::from_json(&json).is_err());
    let d = device_from_json(&json).unwrap();
    assert_eq!(d.qubits.len(), 115);
    assert!(matches!(device_from_json("{"), Err(QsoError::Parse(_))));
    let too_many = {
        let mut d = DeviceModel::new("too wide");
        d.qubits = vec![crate::noise::QubitProperties::new(); MAX_PROPAGATION_QUBITS + 1];
        d.to_json()
    };
    assert!(device_from_json(&too_many).is_err());
}

/// One-dimensional kicked Ising chain of `n` qubits with Pauli–Lindblad
/// noise after the ZZ layers: about 10⁵ strings after a few steps.
#[cfg(feature = "parallel")]
fn kicked_chain(n: usize, steps: usize, theta: f64, alpha: f64) -> Vec<Layer> {
    let mut rates = PauliOperator::new(n).unwrap();
    for q in 0..n {
        rates
            .add_term(1e-3, &[(q, Pauli::X)])
            .unwrap()
            .add_term(2e-3, &[(q, Pauli::Z)])
            .unwrap();
    }
    let mut layers = Vec::new();
    for _ in 0..steps {
        let mut c = Circuit::new(n);
        for q in 0..n {
            c.rx(q, theta);
        }
        for start in [0, 1] {
            for q in (start..n - 1).step_by(2) {
                c.rzz(q, q + 1, -FRAC_PI_2);
            }
        }
        layers.push(Layer::gates(c));
        layers.push(Layer::pauli_lindblad(rates.clone(), alpha));
    }
    layers
}

#[cfg(feature = "parallel")]
#[test]
fn results_are_bit_identical_on_any_number_of_threads() {
    let n = 24;
    let layers = kicked_chain(n, 7, 0.6, -0.5);
    let obs = PauliOperator::parse(n, "Z12 + 0.5 X11 X12").unwrap();
    let state: Vec<[f64; 3]> = (0..n)
        .map(|q| {
            let t = 0.1 * q as f64;
            [0.3 * t.sin(), 0.2 * t.cos(), 0.9]
        })
        .collect();
    let options = [
        opts().with_operator(true),
        opts().with_operator(true).with_min_coefficient(1e-5),
    ];
    let mut reference: Vec<PropagationResult> = Vec::new();
    for threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for (i, o) in options.iter().enumerate() {
            let r = pool
                .install(|| propagate(&layers, &obs, Some(&state), o))
                .unwrap();
            if threads == 1 {
                // Several batches of a splitting pass are exercised.
                assert!(
                    r.peak_terms > crate::parallel::CHUNK * super::table::BATCH_CHUNKS,
                    "{}",
                    r.peak_terms
                );
                reference.push(r);
                continue;
            }
            let e = &reference[i];
            assert_eq!(r.value.to_bits(), e.value.to_bits(), "{threads} threads");
            assert_eq!(r.truncation_bound.to_bits(), e.truncation_bound.to_bits());
            assert_eq!(
                (r.terms, r.peak_terms, r.dropped_terms),
                (e.terms, e.peak_terms, e.dropped_terms)
            );
            let (a, b) = (r.operator.as_ref().unwrap(), e.operator.as_ref().unwrap());
            assert_eq!(a.rows().0, b.rows().0);
            assert!(
                a.rows()
                    .1
                    .iter()
                    .zip(b.rows().1)
                    .all(|(x, y)| x.to_bits() == y.to_bits())
            );
        }
    }
}

#[test]
fn propagation_errors_name_the_problem() {
    let invalid = |r: Result<PropagationResult>, name: &str| match r {
        Err(QsoError::InvalidArgument { name: got, .. }) => assert_eq!(got, name),
        other => panic!("expected {name}: {other:?}"),
    };
    let z0 = PauliOperator::parse(3, "Z0").unwrap();
    // Registers.
    assert!(matches!(
        PauliOperator::new(0),
        Err(QsoError::TooManyQubits {
            what: "Pauli propagation",
            ..
        })
    ));
    assert!(matches!(
        PauliOperator::new(MAX_PROPAGATION_QUBITS + 1),
        Err(QsoError::TooManyQubits { .. })
    ));
    assert!(matches!(
        PauliOperator::parse(3, "Z3"),
        Err(QsoError::QubitOutOfRange { .. })
    ));
    assert!(matches!(
        PauliOperator::parse(3, "Z1 X1"),
        Err(QsoError::DuplicateQubit {
            gate: "Pauli string",
            ..
        })
    ));
    assert!(matches!(
        PauliOperator::parse(3, "Z0 +* X1"),
        Err(QsoError::Parse(_))
    ));
    // Layers.
    let wide = circuit(4, |c| {
        c.h(3);
    });
    assert!(matches!(
        propagate(&[wide], &z0, None, &opts()),
        Err(QsoError::DimensionMismatch { .. })
    ));
    let mut c = Circuit::new(3);
    let t = c.new_parameter();
    c.rx(0, t);
    assert!(matches!(
        propagate(&[Layer::gates(c)], &z0, None, &opts()),
        Err(QsoError::ParameterCount { .. })
    ));
    let ad = Channel::amplitude_damping(0.1).unwrap();
    assert!(matches!(
        propagate(&[Layer::channel(ad.clone(), &[3])], &z0, None, &opts()),
        Err(QsoError::QubitOutOfRange { .. })
    ));
    assert!(matches!(
        propagate(&[Layer::channel(ad, &[0, 1])], &z0, None, &opts()),
        Err(QsoError::DimensionMismatch { .. })
    ));
    let dep = Channel::depolarizing_2q(0.1).unwrap();
    assert!(matches!(
        propagate(&[Layer::channel(dep, &[2, 2])], &z0, None, &opts()),
        Err(QsoError::DuplicateQubit { .. })
    ));
    let mut c = Circuit::new(5);
    c.try_push(Instruction {
        kind: GateKind::X,
        targets: vec![4],
        controls: vec![0, 1, 2, 3],
        params: vec![],
        matrix: None,
    })
    .unwrap();
    let z4 = PauliOperator::parse(5, "Z4").unwrap();
    invalid(propagate(&[Layer::gates(c)], &z4, None, &opts()), "circuit");
    // Rates and factors.
    for (text, n) in [("0.1 X0 + 0.2", 3), ("-0.1 X0", 3)] {
        let rates = PauliOperator::parse(n, text).unwrap();
        invalid(
            propagate(&[Layer::pauli_lindblad(rates, 1.0)], &z0, None, &opts()),
            "rates",
        );
    }
    let rates = PauliOperator::parse(3, "0.1 X0").unwrap();
    invalid(
        propagate(
            &[Layer::pauli_lindblad(rates.clone(), f64::INFINITY)],
            &z0,
            None,
            &opts(),
        ),
        "factor",
    );
    let other = PauliOperator::parse(4, "0.1 X0").unwrap();
    assert!(matches!(
        propagate(&[Layer::pauli_lindblad(other, 1.0)], &z0, None, &opts()),
        Err(QsoError::DimensionMismatch { .. })
    ));
    // Γ beyond floating point.
    invalid(
        propagate(&[Layer::pauli_lindblad(rates, -1e5)], &z0, None, &opts()),
        "factor",
    );
    // States.
    assert!(matches!(
        propagate(&[], &z0, Some(&[[0.0, 0.0, 1.0]; 2]), &opts()),
        Err(QsoError::DimensionMismatch { .. })
    ));
    for bad in [[0.8, 0.0, 0.8], [f64::NAN, 0.0, 0.0]] {
        invalid(
            propagate(&[], &z0, Some(&[bad, [0.0; 3], [0.0; 3]]), &opts()),
            "state",
        );
    }
    let unit = std::f64::consts::FRAC_1_SQRT_2;
    let r = propagate(
        &[],
        &z0,
        Some(&[[unit, 0.0, unit], [0.0; 3], [0.0; 3]]),
        &opts(),
    )
    .unwrap();
    assert!((r.value - unit).abs() < 1e-16);
    // Options.
    invalid(
        propagate(&[], &z0, None, &opts().with_min_coefficient(-1.0)),
        "min_coefficient",
    );
    invalid(
        propagate(&[], &z0, None, &opts().with_max_terms(0)),
        "max_terms",
    );
    // Too many strings held.
    let mut c = Circuit::new(3);
    for q in 0..3 {
        c.rx(q, 0.3).ry(q, 0.4);
    }
    let all = PauliOperator::parse(3, "Z0 Z1 Z2").unwrap();
    invalid(
        propagate(
            &[Layer::gates(c.clone())],
            &all,
            None,
            &opts().with_max_terms(4),
        ),
        "max_terms",
    );
    assert!(propagate(&[Layer::gates(c)], &all, None, &opts().with_max_terms(64)).is_ok());
    let two = PauliOperator::parse(3, "Z0 + Z1").unwrap();
    invalid(
        propagate(&[], &two, None, &opts().with_max_terms(1)),
        "max_terms",
    );
    // Devices with slow noise.
    let slow = DeviceModel::new("slow").with_default_qubit(
        crate::noise::QubitProperties::new()
            .with_detuning(crate::noise::Spread::Gaussian { sigma: 0.1 }),
    );
    let mut c = Circuit::new(2);
    c.h(0);
    match device_layers(&c, &[], &slow) {
        Err(QsoError::InvalidArgument { name, .. }) => assert_eq!(name, "device"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn images_waiting_to_be_merged_count_towards_max_terms() {
    // RX(0.3) maps Y0 + Z0 to four images that merge into two strings:
    // the images are held at once, so three strings are too few, and a
    // chunk cannot outgrow the limit before its batch is merged.
    let rx = circuit(1, |c| {
        c.rx(0, 0.3);
    });
    let obs = PauliOperator::parse(1, "Y0 + Z0").unwrap();
    match propagate(
        std::slice::from_ref(&rx),
        &obs,
        None,
        &opts().with_max_terms(3),
    ) {
        Err(QsoError::InvalidArgument { name, .. }) => assert_eq!(name, "max_terms"),
        other => panic!("{other:?}"),
    }
    let r = propagate(&[rx], &obs, None, &opts().with_max_terms(4)).unwrap();
    assert_eq!((r.terms, r.peak_terms), (2, 2));
    // Many strings that each split a little: the chunk's images exceed
    // the limit long before a batch is complete.
    let n = 12;
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.rx(q, 0.3);
    }
    let mut obs = PauliOperator::new(n).unwrap();
    let mut rng = Rng::seed_from_u64(16_414);
    for _ in 0..3000 {
        let factors: Vec<(usize, Pauli)> = (0..n)
            .filter(|_| rng.below(3) == 0)
            .map(|q| (q, Pauli::Z))
            .collect();
        obs.add_term(1.0, &factors).unwrap();
    }
    let held = obs.len();
    match propagate(
        &[Layer::gates(c)],
        &obs,
        None,
        &opts().with_max_terms(held + 10),
    ) {
        Err(QsoError::InvalidArgument { name, reason }) => {
            assert_eq!(name, "max_terms");
            assert!(reason.contains("pass 1"), "{reason}");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn light_cone_and_identity_observables() {
    // Ops off the light cone are removed: a huge register with a few
    // gates costs nothing, and the identity term survives every layer.
    let n = MAX_PROPAGATION_QUBITS;
    let mut c = Circuit::new(n);
    c.h(9000).cx(9000, 16_000).rx(5, 0.3).ry(16_383, 0.2);
    let ad = Layer::channel(Channel::amplitude_damping(0.3).unwrap(), &[16_000]);
    let obs = PauliOperator::parse(n, "0.25 + X9000 X16000 + Z5").unwrap();
    let r = propagate(
        &[Layer::gates(c), ad],
        &obs,
        None,
        &opts().with_operator(true),
    )
    .unwrap();
    // ⟨X X⟩ on the Bell pair after damping the target: √(1 − γ).
    let expected = 0.25 + 0.7f64.sqrt() + 0.3f64.cos();
    assert!((r.value - expected).abs() <= 1e-14, "{}", r.value);
    let op = r.operator.unwrap();
    assert_eq!(op.n_qubits(), n);
    assert!((op.coefficient(&[]).unwrap() - 0.25).abs() < 1e-16);
    // The zero observable.
    let zero = PauliOperator::new(4).unwrap();
    let r = propagate(&[], &zero, None, &opts()).unwrap();
    assert_eq!((r.value, r.terms), (0.0, 0));
}

#[test]
fn snapped_images_keep_inverted_layers_in_the_bound() {
    // Review regression: RXX(ε) with sin ε ≤ 10⁻¹⁴ snaps the branch
    // sin ε·Y0 X1 into the bound, but the exact map carries it on through
    // the inverted layers on Z1 (time order: the layers, then the gate),
    // which multiply it by e^{2λ} each. The light cone must count those
    // layers: before the fix they were cut off and the bound stayed at
    // ε, up to 10¹³ times below the error.
    let eps = 9e-15f64;
    let mut gate = Circuit::new(2);
    gate.rxx(0, 1, eps);
    let z0 = PauliOperator::parse(2, "Z0").unwrap();
    // ⟨Y0⟩ = ⟨X1⟩ = 1, so ⟨Z0⟩ = 0 and the exact value is
    // sin ε·e^{2λ·reps} (R† Z0 R = cos ε·Z0 + sin ε·Y0 X1).
    let state = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
    for (lambda, reps) in [(15.0, 1), (1.0, 10), (0.1, 40)] {
        let rates = PauliOperator::parse(2, &format!("{lambda}*Z1")).unwrap();
        let mut layers = vec![Layer::pauli_lindblad(rates, -1.0); reps];
        layers.push(Layer::gates(gate.clone()));
        let exact = eps.sin() * (2.0 * lambda * reps as f64).exp();
        for o in [opts(), opts().with_min_coefficient(2.0)] {
            let r = propagate(&layers, &z0, Some(&state), &o).unwrap();
            assert_eq!(r.value, 0.0);
            let err = (exact - r.value).abs();
            assert!(
                err <= r.truncation_bound * (1.0 + 1e-12),
                "λ = {lambda}, {reps} layers, τ = {}: error {err:e} > bound {:e}",
                o.min_coefficient,
                r.truncation_bound
            );
        }
    }
    // Without the gate's branch nothing is snapped and the bound is 0.
    let rates = PauliOperator::parse(2, "1*Z1").unwrap();
    let r = propagate(&[Layer::pauli_lindblad(rates, -1.0)], &z0, None, &opts()).unwrap();
    assert_eq!((r.value, r.truncation_bound), (1.0, 0.0));
}

#[test]
fn overflow_is_an_error_not_a_value() {
    // Review regression: Γ = e^{600} is finite, but times the observable's
    // 1-norm it is not; the run used to return inf, NaN or a value with
    // every string dropped as a "cancellation".
    let name = |r: Result<PropagationResult>| match r {
        Err(QsoError::InvalidArgument { name, .. }) => name,
        other => panic!("expected an error: {other:?}"),
    };
    let rates = PauliOperator::parse(1, "300*X0").unwrap();
    let big = PauliOperator::parse(1, "1e100*Z0").unwrap();
    let hh = circuit(1, |c| {
        c.h(0).h(0);
    });
    let ry = circuit(1, |c| {
        c.ry(0, 0.3);
    });
    for gates in [hh, ry] {
        let layers = [gates, Layer::pauli_lindblad(rates.clone(), -1.0)];
        for o in [opts(), opts().with_operator(true)] {
            assert_eq!(name(propagate(&layers, &big, None, &o)), "factor");
        }
        // The same layers on a unit observable are fine.
        let r = propagate(&layers, &z(0, 1), None, &opts()).unwrap();
        assert!(r.value.is_finite() && r.truncation_bound.is_finite());
    }
    // A 1-norm beyond floating point without inverted layers.
    let huge = PauliOperator::parse(2, "1e308*Z0 + 1e308*Z1").unwrap();
    assert_eq!(name(propagate(&[], &huge, None, &opts())), "observable");
}

#[test]
fn empty_sums_read_as_positive_zero() {
    // Review regression: a sum over no strings is +0, not −0.
    let zero = PauliOperator::new(3).unwrap();
    let r = propagate(&[], &zero, None, &opts()).unwrap();
    assert_eq!(r.value.to_bits(), 0f64.to_bits());
    assert_eq!(r.estimate().value.to_bits(), 0f64.to_bits());
    let rates = PauliOperator::parse(2, "0.1*X0 + 0.3*X1").unwrap();
    let cut = propagate(
        &[Layer::pauli_lindblad(rates, -1.0)],
        &z(0, 2),
        None,
        &opts().with_min_coefficient(2.0),
    )
    .unwrap();
    assert_eq!((cut.value.to_bits(), cut.terms), (0f64.to_bits(), 0));
}

#[test]
fn operators_from_terms_equal_the_parsed_text() {
    let terms = vec![
        (0.5, vec![(0, Pauli::Z), (70, Pauli::X)]),
        (-1.0, vec![(3, Pauli::Y)]),
        (0.25, vec![(70, Pauli::X), (0, Pauli::Z)]),
        (2.0, vec![]),
        (1.0, vec![(3, Pauli::Y), (5, Pauli::I)]),
    ];
    let op = PauliOperator::from_terms(80, terms.clone()).unwrap();
    assert_eq!(op, PauliOperator::parse(80, "0.75*Z0 X70 + 2").unwrap());
    let mut one_by_one = PauliOperator::new(80).unwrap();
    for (c, f) in &terms {
        one_by_one.add_term(*c, f).unwrap();
    }
    assert_eq!(op, one_by_one);
    assert!(matches!(
        PauliOperator::from_terms(4, [(1.0, [(4, Pauli::X)])]),
        Err(QsoError::QubitOutOfRange { .. })
    ));
    assert!(matches!(
        PauliOperator::from_terms(4, [(1.0, [(1, Pauli::X), (1, Pauli::Z)])]),
        Err(QsoError::DuplicateQubit { .. })
    ));
    assert!(matches!(
        PauliOperator::from_terms(4, [(f64::NAN, [(1, Pauli::X)])]),
        Err(QsoError::InvalidArgument { .. })
    ));
    // 16384 terms on 16384 qubits at once (term by term this is quadratic).
    let n = MAX_PROPAGATION_QUBITS;
    let wide = PauliOperator::from_terms(n, (0..n).map(|q| (1.0, [(q, Pauli::Z)]))).unwrap();
    assert_eq!((wide.len(), wide.max_weight()), (n, 1));
}

#[test]
fn programs_propagate_with_final_measurements_only() {
    // Final measurements are left out; a measurement followed by a gate on
    // its qubit (which dephases it), resets and conditions are refused.
    let text = |body: &str| {
        format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\n{body}")
    };
    let value = |body: &str| -> Result<f64> {
        let c = program_circuit(&Program::from_qasm(&text(body))?)?;
        Ok(propagate(&[Layer::gates(c)], &z(0, 2), None, &opts())?.value)
    };
    assert_eq!(value("x q[0];\nmeasure q -> c;").unwrap(), -1.0);
    assert_eq!(
        value("x q[0];\nmeasure q[1] -> c[1];\nh q[0];\nh q[0];").unwrap(),
        -1.0
    );
    for body in [
        "h q[0];\nmeasure q[0] -> c[0];\nh q[0];",
        "x q[0];\nreset q[0];",
        "measure q[1] -> c[1];\nif (c == 1) x q[0];",
    ] {
        assert!(
            matches!(value(body), Err(QsoError::InvalidArgument { name, .. }) if name == "program"),
            "{body}"
        );
    }
}

// ---------------------------------------------------------------------
// Measurements (run with `--ignored --nocapture` in release)
// ---------------------------------------------------------------------

/// Peak resident memory since the last reset, in MB (Linux only).
#[cfg(feature = "parallel")]
fn peak_mb(reset: bool) -> f64 {
    if reset {
        let _ = std::fs::write("/proc/self/clear_refs", "5");
    }
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
        })
        .map_or(f64::NAN, |kb| kb / 1024.0)
}

#[cfg(feature = "parallel")]
fn heavy_hex_kicked_ising(steps: usize, theta: f64) -> (usize, Vec<Layer>) {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/data/heavy_hex_7.json")).unwrap();
    let n = data["n_qubits"].as_u64().unwrap() as usize;
    let mut c = Circuit::new(n);
    for _ in 0..steps {
        for q in 0..n {
            c.rx(q, theta);
        }
        for class in data["colours"].as_array().unwrap() {
            for e in class.as_array().unwrap() {
                c.rzz(
                    e[0].as_u64().unwrap() as usize,
                    e[1].as_u64().unwrap() as usize,
                    -FRAC_PI_2,
                );
            }
        }
    }
    (n, vec![Layer::gates(c)])
}

#[cfg(feature = "parallel")]
#[test]
#[ignore]
fn measure_kicked_ising() {
    use std::time::Instant;
    let pool = |t: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap()
    };
    // `QSO_KIM_CASE=θ,steps,τ` (θ as 0.3, pi4 or 1) runs one truncated
    // configuration only: its peak memory is then that of a fresh process,
    // not inflated by what the allocator kept from an earlier run.
    let only: Option<(f64, usize, f64)> = std::env::var("QSO_KIM_CASE").ok().map(|v| {
        let p: Vec<&str> = v.split(',').collect();
        let theta = match p[0] {
            "pi4" => std::f64::consts::FRAC_PI_4,
            t => t.parse().unwrap(),
        };
        (theta, p[1].parse().unwrap(), p[2].parse().unwrap())
    });
    // 1 and 5. Exact, five steps: time, strings, memory, bit identity.
    for theta in [0.3, std::f64::consts::FRAC_PI_4, 1.0] {
        if only.is_some() {
            break;
        }
        let (n, layers) = heavy_hex_kicked_ising(5, theta);
        let z = PauliOperator::parse(n, "Z57").unwrap();
        let mut first: Option<PropagationResult> = None;
        for threads in [1, 2, 4] {
            let p = pool(threads);
            let _ = peak_mb(true);
            let mut best = f64::INFINITY;
            let mut r = None;
            for _ in 0..5 {
                let t = Instant::now();
                let out = p.install(|| propagate(&layers, &z, None, &opts())).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
                r = Some(out);
            }
            let r = r.unwrap();
            let same = first.as_ref().is_none_or(|f| {
                f.value.to_bits() == r.value.to_bits()
                    && f.truncation_bound.to_bits() == r.truncation_bound.to_bits()
                    && f.terms == r.terms
            });
            eprintln!(
                "θ = {theta:.4}, 5 steps, {threads} threads: {:.1} ms (best of 5), value {:.16}, {} strings, peak {} strings, bound {:.1e}, peak RSS {:.0} MB, bit-identical {same}",
                best * 1e3,
                r.value,
                r.terms,
                r.peak_terms,
                r.truncation_bound,
                peak_mb(false)
            );
            first.get_or_insert(r);
        }
    }
    // 2. Truncated, six and seven steps.
    let aer = [
        (0.3, [0.9512884369194023, 0.9559990282385086]),
        (
            std::f64::consts::FRAC_PI_4,
            [0.3744370639324195, 0.3790864877280415],
        ),
        (1.0, [0.11553327362744006, 0.09304591906752682]),
    ];
    let p4 = pool(4);
    let p1 = pool(1);
    for (theta, values) in aer {
        for (i, steps) in [6, 7].into_iter().enumerate() {
            let (n, layers) = heavy_hex_kicked_ising(steps, theta);
            let z = PauliOperator::parse(n, "Z57").unwrap();
            for tau in [1e-3, 1e-4, 1e-5] {
                if only.is_some_and(|c| c != (theta, steps, tau)) {
                    continue;
                }
                let o = opts().with_min_coefficient(tau);
                let _ = peak_mb(true);
                let t = Instant::now();
                let r = p4.install(|| propagate(&layers, &z, None, &o));
                let t4 = t.elapsed().as_secs_f64();
                let rss = peak_mb(false);
                match r {
                    Ok(r) => {
                        let t = Instant::now();
                        let r1 = p1.install(|| propagate(&layers, &z, None, &o)).unwrap();
                        let t1 = t.elapsed().as_secs_f64();
                        assert_eq!(r1.value.to_bits(), r.value.to_bits());
                        let err = (r.value - values[i]).abs();
                        eprintln!(
                            "θ = {theta:.4}, {steps} steps, τ = {tau:.0e}: value {:.10}, error {err:.2e}, bound {:.3e}, within {}, {} strings (peak {}), {} dropped, {:.1} ms on 1 thread, {:.1} ms on 4, peak RSS {rss:.0} MB",
                            r.value,
                            r.truncation_bound,
                            err <= r.truncation_bound,
                            r.terms,
                            r.peak_terms,
                            r.dropped_terms,
                            t1 * 1e3,
                            t4 * 1e3,
                        );
                    }
                    Err(e) => eprintln!("θ = {theta:.4}, {steps} steps, τ = {tau:.0e}: {e}"),
                }
            }
        }
    }
}

/// `count` random strings on `n` qubits: dense (every letter uniform) or
/// `I`/`Z` at random with one `X`/`Y` (as the design's scratch benchmark).
#[cfg(feature = "parallel")]
fn synthetic(n: usize, count: usize, dense: bool, rng: &mut Rng) -> PauliOperator {
    let w = n.div_ceil(64);
    let top = if n.is_multiple_of(64) {
        u64::MAX
    } else {
        (1u64 << (n % 64)) - 1
    };
    let mut keys = Vec::with_capacity(count * 2 * w);
    let mut coefs = Vec::with_capacity(count);
    for _ in 0..count {
        let mut key = vec![0u64; 2 * w];
        if dense {
            key.iter_mut().for_each(|k| *k = rng.next_u64());
        } else {
            key[w..].iter_mut().for_each(|k| *k = rng.next_u64());
            let q = rng.below(n as u64) as usize;
            key[..w].iter_mut().for_each(|k| *k = 0);
            key[q / 64] |= 1 << (q % 64);
        }
        key[w - 1] &= top;
        key[2 * w - 1] &= top;
        keys.extend(key);
        coefs.push(rng.uniform_range(-0.5, 0.5));
    }
    PauliOperator::from_rows(n, keys, coefs).unwrap()
}

#[cfg(feature = "parallel")]
#[test]
#[ignore]
fn measure_synthetic_tables() {
    use std::time::Instant;
    let count = 1_000_000;
    for n in [64usize, 127, 1024] {
        let mut rng = Rng::seed_from_u64(1);
        let dense = synthetic(n, count, true, &mut rng);
        let flips = synthetic(n, count, false, &mut rng);
        let mut cx = Circuit::new(n);
        for i in 0..n / 2 {
            cx.cx(2 * i, 2 * i + 1);
        }
        let mut rz = Circuit::new(n);
        for q in 0..n {
            rz.rz(q, 0.1 + 0.01 * q as f64);
        }
        for (name, layer, obs) in [
            ("CX brickwork, in place", Layer::gates(cx), &dense),
            ("RZ layer, split", Layer::gates(rz), &flips),
        ] {
            let ops = compile_layers(&[layer], n, &mut TableCache::default()).unwrap();
            let plan = super::compile::plan(ops, obs).unwrap();
            for threads in [1, 2, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let mut best = f64::INFINITY;
                let mut terms = 0;
                for _ in 0..3 {
                    let t = Instant::now();
                    let out = pool
                        .install(|| super::table::run(&plan, None, &opts()))
                        .unwrap();
                    best = best.min(t.elapsed().as_secs_f64());
                    terms = out.coefs.len();
                }
                eprintln!(
                    "n = {n}, {name}, {threads} threads: {:.1} ms (best of 3), {count} -> {terms} strings, {} passes",
                    best * 1e3,
                    plan.passes.len()
                );
            }
        }
    }
}
