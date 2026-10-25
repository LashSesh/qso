//! OpenQASM 2.0 import and export.
//!
//! Export targets the gate set of the original `qelib1.inc` (Cross et al.,
//! arXiv:1707.03429) so that the output loads in every OpenQASM 2 consumer
//! (Qiskit, Cirq, tket, hardware vendors). Gates outside that set are
//! decomposed exactly (e.g. `rzz → cx·rz·cx`, `swap → 3 cx`); controlled
//! single-qubit gates without a native controlled form are decomposed into
//! `u1`, `u3` and `cx`. `cu3` is never emitted because the original and
//! Qiskit's `qelib1.inc` define it differently (by a controlled phase); the
//! exported circuits are exact under both. Gates with more controls than
//! `qelib1.inc` offers (multi-controlled X, Z, single-qubit unitaries and
//! swaps with any number of controls) are decomposed without ancillas into
//! singly controlled gates and Toffolis, `O(k²)` gates for `k` controls
//! (Barenco et al. 1995). Uncontrolled gates are exported up to a global
//! phase, which OpenQASM 2 cannot express. When *importing*
//! `cu3`, Qiskit's definition (exact controlled-U3) is assumed.
//!
//! Import ([`from_qasm2`]) understands the same set plus the extensions
//! shipped with Qiskit's `qelib1.inc` (`sx`, `sxdg`, `p`, `cp`, `u`,
//! `swap`, `cswap`, `crx`, `cry`, `rxx`, `rzz`, `id`, `u2`), custom `gate`
//! definitions, several quantum registers, register broadcasting and
//! arithmetic parameter expressions with `pi`. `measure` and `barrier` are
//! skipped; `reset` and classical conditions are rejected, as are programs
//! beyond [`MAX_QASM_QUBITS`] qubits or [`MAX_QASM_INSTRUCTIONS`] gates and
//! parameters that are not finite. Programs with measurements, resets and
//! conditions are read by [`Program::from_qasm`](super::Program::from_qasm),
//! which also reads OpenQASM 3.

use std::f64::consts::PI;
use std::fmt::Write as _;

use num_complex::Complex64 as C64;

use super::circuit::{Circuit, Instruction};
use super::decompose;
use super::gates::{GateKind, matrix_1q};
use crate::error::{QsoError, Result};

/// Largest total size of the quantum registers of an imported program.
pub const MAX_QASM_QUBITS: usize = 1 << 14;

/// Largest number of gates an imported program may expand to (after
/// register broadcasting).
pub const MAX_QASM_INSTRUCTIONS: usize = 1 << 22;

pub(crate) fn real(v: f64) -> String {
    let s = format!("{v:?}");
    if s.contains('e') && !s.contains('.') {
        s.replacen('e', ".0e", 1)
    } else {
        s
    }
}

/// ZYZ decomposition `M = e^{iα}·U3(θ, φ, λ)` of a 2×2 unitary.
pub(crate) fn zyz(m: &[C64; 4]) -> (f64, f64, f64, f64) {
    // det M = e^{2iα}·det U3 = e^{2iα}·e^{i(φ+λ)}.
    let theta = 2.0 * m[2].norm().atan2(m[0].norm());
    let (alpha, phi, lambda);
    if m[0].norm() > 1e-12 && m[2].norm() > 1e-12 {
        // m00 = e^{iα} cos, m10 = e^{i(α+φ)} sin, m01 = −e^{i(α+λ)} sin
        alpha = m[0].arg();
        phi = m[2].arg() - alpha;
        lambda = (-m[1]).arg() - alpha;
    } else if m[0].norm() > 1e-12 {
        // Diagonal: only φ+λ is determined; put it all in λ.
        alpha = m[0].arg();
        phi = 0.0;
        lambda = m[3].arg() - alpha;
    } else {
        // Anti-diagonal: only φ−λ is determined; set λ = 0.
        // m10 = e^{i(α+φ)}, m01 = −e^{i(α+λ)} = −e^{iα}
        alpha = (-m[1]).arg();
        phi = m[2].arg() - alpha;
        lambda = 0.0;
    }
    (theta, phi, lambda, alpha)
}

/// Controlled `e^{iα}·U3(θ, φ, λ)` from uncontrolled `u1`, `u3` and `cx`.
///
/// `cu3` itself is avoided: the original `qelib1.inc` defines it without
/// the `u1((λ+φ)/2)` on the control (a controlled global phase), Qiskit's
/// with it. Built from uncontrolled gates the circuit is exact under both
/// definitions (they differ only by global phases of uncontrolled gates).
fn emit_controlled_u3(
    out: &mut String,
    theta: f64,
    phi: f64,
    lambda: f64,
    alpha: f64,
    c: usize,
    t: usize,
) {
    let _ = writeln!(out, "u1({}) q[{c}];", real((lambda + phi) / 2.0 + alpha));
    let _ = writeln!(out, "u1({}) q[{t}];", real((lambda - phi) / 2.0));
    let _ = writeln!(out, "cx q[{c}],q[{t}];");
    let _ = writeln!(
        out,
        "u3({},0,{}) q[{t}];",
        real(-theta / 2.0),
        real(-(phi + lambda) / 2.0)
    );
    let _ = writeln!(out, "cx q[{c}],q[{t}];");
    let _ = writeln!(out, "u3({},{},0) q[{t}];", real(theta / 2.0), real(phi));
}

fn emit_1q_controlled(out: &mut String, m: &[C64; 4], c: usize, t: usize) {
    let (theta, phi, lambda, alpha) = zyz(m);
    emit_controlled_u3(out, theta, phi, lambda, alpha, c, t);
}

/// Gates with more controls than qelib1 offers, as singly controlled gates
/// and Toffolis (see [`decompose`](super::decompose)); `None` if the
/// instruction is not of that kind.
fn decompose_controls(
    inst: &Instruction,
    angles: &[f64],
    n_qubits: usize,
) -> Option<Vec<Instruction>> {
    use GateKind::*;
    let (t, c) = (&inst.targets, &inst.controls);
    let native = matches!((inst.kind, c.len()), (X | Z, 2));
    if c.len() < 2 || native {
        return None;
    }
    let idle: Vec<usize> = (0..n_qubits)
        .filter(|q| !c.contains(q) && !t.contains(q))
        .collect();
    let mut out = Vec::new();
    match inst.kind {
        Swap => {
            // CSWAP^k = CX(b→a) · C^{k+1}X(controls, a → b) · CX(b→a).
            let (a, b) = (t[0], t[1]);
            let cx = |out: &mut Vec<Instruction>| {
                out.push(Instruction {
                    kind: X,
                    targets: vec![a],
                    controls: vec![b],
                    params: Vec::new(),
                    matrix: None,
                })
            };
            cx(&mut out);
            let mut controls = c.clone();
            controls.push(a);
            if idle.is_empty() {
                decompose::mcu(matrix_1q(X, &[]), &controls, b, &mut out);
            } else {
                decompose::mcx(&controls, b, &idle, &mut out);
            }
            cx(&mut out);
        }
        X if !idle.is_empty() => decompose::mcx(c, t[0], &idle, &mut out),
        Unitary if t.len() == 1 => {
            let m: [C64; 4] = inst.matrix.as_ref().expect("matrix")[..]
                .try_into()
                .expect("2x2");
            decompose::mcu(m, c, t[0], &mut out);
        }
        kind if kind.num_targets() == Some(1) => {
            decompose::mcu(matrix_1q(kind, angles), c, t[0], &mut out);
        }
        _ => return None,
    }
    Some(out)
}

pub(crate) fn emit_instruction(
    out: &mut String,
    inst: &Instruction,
    angles: &[f64],
    n_qubits: usize,
) -> Result<()> {
    use GateKind::*;
    if let Some(gates) = decompose_controls(inst, angles, n_qubits) {
        for g in &gates {
            emit_instruction(out, g, &g.angles(&[]), n_qubits)?;
        }
        return Ok(());
    }
    let t = &inst.targets;
    let c = &inst.controls;
    let unsupported = || {
        QsoError::invalid(
            "circuit",
            format!(
                "{} with {} control(s) has no OpenQASM 2 representation in qelib1",
                inst.kind.name(),
                c.len()
            ),
        )
    };
    match (inst.kind, c.len()) {
        (H | X | Y | Z | S | Sdg | T | Tdg, 0) => {
            let _ = writeln!(out, "{} q[{}];", inst.kind.name(), t[0]);
        }
        (RX | RY | RZ, 0) => {
            let _ = writeln!(
                out,
                "{}({}) q[{}];",
                inst.kind.name(),
                real(angles[0]),
                t[0]
            );
        }
        (P, 0) => {
            let _ = writeln!(out, "u1({}) q[{}];", real(angles[0]), t[0]);
        }
        (U, 0) => {
            let _ = writeln!(
                out,
                "u3({},{},{}) q[{}];",
                real(angles[0]),
                real(angles[1]),
                real(angles[2]),
                t[0]
            );
        }
        (SX | SXdg, 0) => {
            let a = if inst.kind == SX { PI / 2.0 } else { -PI / 2.0 };
            let _ = writeln!(out, "rx({}) q[{}];", real(a), t[0]);
        }
        (Unitary, 0) if t.len() == 1 => {
            let m: [C64; 4] = inst.matrix.as_ref().expect("matrix")[..]
                .try_into()
                .expect("2x2");
            let (theta, phi, lambda, _) = zyz(&m);
            let _ = writeln!(
                out,
                "u3({},{},{}) q[{}];",
                real(theta),
                real(phi),
                real(lambda),
                t[0]
            );
        }
        (X, 1) => {
            let _ = writeln!(out, "cx q[{}],q[{}];", c[0], t[0]);
        }
        (Y, 1) => {
            let _ = writeln!(out, "cy q[{}],q[{}];", c[0], t[0]);
        }
        (Z, 1) => {
            let _ = writeln!(out, "cz q[{}],q[{}];", c[0], t[0]);
        }
        (H, 1) => {
            let _ = writeln!(out, "ch q[{}],q[{}];", c[0], t[0]);
        }
        (RZ, 1) => {
            let _ = writeln!(out, "crz({}) q[{}],q[{}];", real(angles[0]), c[0], t[0]);
        }
        (P, 1) => {
            let _ = writeln!(out, "cu1({}) q[{}],q[{}];", real(angles[0]), c[0], t[0]);
        }
        (U, 1) => emit_controlled_u3(out, angles[0], angles[1], angles[2], 0.0, c[0], t[0]),
        (S | Sdg | T | Tdg | SX | SXdg | RX | RY, 1) => {
            emit_1q_controlled(out, &matrix_1q(inst.kind, angles), c[0], t[0]);
        }
        (Unitary, 1) if t.len() == 1 => {
            let m: [C64; 4] = inst.matrix.as_ref().expect("matrix")[..]
                .try_into()
                .expect("2x2");
            emit_1q_controlled(out, &m, c[0], t[0]);
        }
        (X, 2) => {
            let _ = writeln!(out, "ccx q[{}],q[{}],q[{}];", c[0], c[1], t[0]);
        }
        (Z, 2) => {
            let _ = writeln!(out, "h q[{}];", t[0]);
            let _ = writeln!(out, "ccx q[{}],q[{}],q[{}];", c[0], c[1], t[0]);
            let _ = writeln!(out, "h q[{}];", t[0]);
        }
        (Swap, 0) => {
            let (a, b) = (t[0], t[1]);
            let _ = writeln!(
                out,
                "cx q[{a}],q[{b}];\ncx q[{b}],q[{a}];\ncx q[{a}],q[{b}];"
            );
        }
        (Swap, 1) => {
            let (a, b) = (t[0], t[1]);
            let _ = writeln!(
                out,
                "cx q[{b}],q[{a}];\nccx q[{}],q[{a}],q[{b}];\ncx q[{b}],q[{a}];",
                c[0]
            );
        }
        (RZZ | RXX | RYY, 0) => {
            let (a, b) = (t[0], t[1]);
            let (pre, post) = match inst.kind {
                RZZ => (String::new(), String::new()),
                RXX => (
                    format!("h q[{a}];\nh q[{b}];\n"),
                    format!("h q[{a}];\nh q[{b}];\n"),
                ),
                _ => (
                    format!("rx({0}) q[{a}];\nrx({0}) q[{b}];\n", real(PI / 2.0)),
                    format!("rx({0}) q[{a}];\nrx({0}) q[{b}];\n", real(-PI / 2.0)),
                ),
            };
            let _ = write!(
                out,
                "{pre}cx q[{a}],q[{b}];\nrz({}) q[{b}];\ncx q[{a}],q[{b}];\n{post}",
                real(angles[0])
            );
        }
        _ => return Err(unsupported()),
    }
    Ok(())
}

/// Export `circuit` (with parameters `theta` bound) as OpenQASM 2.0.
///
/// Qubit `q` of the circuit is `q[q]` of a single register. With
/// `measure_all` every qubit is measured into a classical register `c`.
pub fn to_qasm2(circuit: &Circuit, theta: &[f64], measure_all: bool) -> Result<String> {
    circuit.check_parameters(theta)?;
    let n = circuit.n_qubits();
    let mut out = String::from("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n");
    let _ = writeln!(out, "qreg q[{n}];");
    if measure_all {
        let _ = writeln!(out, "creg c[{n}];");
    }
    for inst in circuit.instructions() {
        emit_instruction(&mut out, inst, &inst.angles(theta), n)?;
    }
    if measure_all {
        out.push_str("measure q -> c;\n");
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parse an OpenQASM 2.0 program into a [`Circuit`] (no free parameters).
///
/// Quantum registers are concatenated in declaration order. Gate
/// definitions are expanded; measurements are dropped and `reset` and
/// `if` are errors — read such programs with
/// [`Program::from_qasm`](super::Program::from_qasm).
pub fn from_qasm2(src: &str) -> Result<Circuit> {
    let (program, version) = super::openqasm::parse(src)?;
    if version != 2 || !src.contains("OPENQASM") {
        return Err(QsoError::Parse(
            "missing 'OPENQASM 2.0;' header (use Program::from_qasm for OpenQASM 3)".into(),
        ));
    }
    program.gates_skipping_measurements()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::StateVector;
    use crate::rng::Rng;

    fn equal_up_to_phase(a: &StateVector, b: &StateVector) -> bool {
        (a.fidelity(b).unwrap() - 1.0).abs() < 1e-10
    }

    #[test]
    fn expression_evaluator() {
        let angle = |e: &str| -> Result<f64> {
            let c = from_qasm2(&format!("OPENQASM 2.0; qreg q[1]; rz({e}) q[0];"))?;
            Ok(c.instructions()[0].angles(&[])[0])
        };
        assert!((angle("pi/2").unwrap() - PI / 2.0).abs() < 1e-15);
        assert!((angle("-3*pi/4 + 1e-1").unwrap() - (-3.0 * PI / 4.0 + 0.1)).abs() < 1e-15);
        assert!((angle("2^3 - sqrt(4)").unwrap() - 6.0).abs() < 1e-15);
        assert!((angle("cos(0)+-(1)").unwrap()).abs() < 1e-15);
        assert!((angle("-2^2").unwrap() + 4.0).abs() < 1e-15);
        assert!((angle("2^-1").unwrap() - 0.5).abs() < 1e-15);
        assert!(angle("pi pi").is_err());
        assert!(angle("foo").is_err());
    }

    #[test]
    fn export_import_round_trip_preserves_state() {
        let n = 4;
        let mut c = Circuit::new(n);
        let th = c.new_parameter();
        c.h(0).x(1).y(2).z(3).s(0).sdg(1).t(2).tdg(3).sx(0).sxdg(1);
        c.rx(2, th)
            .ry(3, 0.4)
            .rz(0, -0.3)
            .p(1, 1.1)
            .u(2, 0.3, 0.9, -1.4);
        c.cx(0, 1)
            .cy(1, 2)
            .cz(2, 3)
            .ch(3, 0)
            .swap(1, 3)
            .cswap(0, 2, 3);
        c.crx(0, 3, 0.7).cry(1, 0, th).crz(2, 1, 1.9).cp(3, 2, -0.6);
        c.rxx(0, 2, 0.5)
            .ryy(3, 1, th)
            .rzz(1, 2, 2.2)
            .ccx(0, 1, 3)
            .mcz(&[1, 2, 3]);
        c.controlled(GateKind::S, &[2], &[0], &[]).unwrap();
        c.controlled(GateKind::SX, &[1], &[3], &[]).unwrap();
        c.controlled(
            GateKind::U,
            &[0],
            &[2],
            &[0.2.into(), 0.5.into(), (-0.7).into()],
        )
        .unwrap();
        let theta = [0.917];
        let qasm = to_qasm2(&c, &theta, true).unwrap();
        let parsed = from_qasm2(&qasm).unwrap();
        let mut rng = Rng::seed_from_u64(4);
        for _ in 0..3 {
            let psi0 = StateVector::random(n, &mut rng);
            let mut a = psi0.clone();
            a.apply_circuit(&c, &theta).unwrap();
            let mut b = psi0.clone();
            b.apply_circuit(&parsed, &[]).unwrap();
            assert!(equal_up_to_phase(&a, &b));
        }
    }

    #[test]
    fn multi_controlled_gates_export_exactly() {
        let mut rng = Rng::seed_from_u64(12);
        let m = crate::linalg::CMatrix::from_fn(2, 2, |_, _| C64::new(rng.normal(), rng.normal()));
        let q = m.qr().q();
        let u = vec![q[(0, 0)], q[(0, 1)], q[(1, 0)], q[(1, 1)]];
        for n in [4usize, 5, 7] {
            let mut c = Circuit::new(n);
            for k in 0..n {
                c.ry(k, 0.3 + 0.2 * k as f64).rz(k, 0.1 * k as f64);
            }
            let all: Vec<usize> = (0..n).collect();
            c.mcx(&all[..n - 1], n - 1); // no idle qubit
            c.mcx(&all[1..n - 1], 0); // one idle qubit
            c.mcz(&all);
            c.controlled(GateKind::RY, &[1], &[0, 2, 3], &[0.7.into()])
                .unwrap();
            c.controlled(GateKind::H, &[n - 1], &all[..n - 1], &[])
                .unwrap();
            c.unitary(u.clone(), &[2], &[3, 0, 1]).unwrap();
            c.controlled(GateKind::Swap, &[0, n - 1], &all[1..n - 1], &[])
                .unwrap();
            let qasm = to_qasm2(&c, &[], false).unwrap();
            let parsed = from_qasm2(&qasm).unwrap();
            assert!(parsed.instructions().iter().all(|g| g.controls.len() <= 2));
            for _ in 0..3 {
                let psi0 = StateVector::random(n, &mut rng);
                let mut a = psi0.clone();
                a.apply_circuit(&c, &[]).unwrap();
                let mut b = psi0.clone();
                b.apply_circuit(&parsed, &[]).unwrap();
                assert!(equal_up_to_phase(&a, &b), "n={n}");
            }
        }
    }

    #[test]
    fn controlled_unitary_export_keeps_relative_phase() {
        // Controlled global-phase-carrying gates must be exact, not just up to phase.
        let mut rng = Rng::seed_from_u64(9);
        let a = crate::linalg::CMatrix::from_fn(2, 2, |_, _| C64::new(rng.normal(), rng.normal()));
        let u = crate::linalg::expm_hermitian(&((&a + a.adjoint()) * C64::new(0.5, 0.0)), 1.3);
        let m = vec![u[(0, 0)], u[(0, 1)], u[(1, 0)], u[(1, 1)]];
        let mut c = Circuit::new(2);
        c.h(0).unitary(m, &[1], &[0]).unwrap();
        let parsed = from_qasm2(&to_qasm2(&c, &[], false).unwrap()).unwrap();
        let psi0 = StateVector::random(2, &mut rng);
        let mut x = psi0.clone();
        x.apply_circuit(&c, &[]).unwrap();
        let mut y = psi0;
        y.apply_circuit(&parsed, &[]).unwrap();
        assert!(equal_up_to_phase(&x, &y));
    }

    #[test]
    fn parses_registers_broadcasting_and_comments() {
        let src = r#"
            OPENQASM 2.0;
            include "qelib1.inc"; // standard library
            qreg a[2];
            qreg b[2];
            creg m[4];
            h a;            // broadcast
            cx a, b;        // pairwise broadcast
            u2(0, pi) b[1];
            barrier a, b;
            measure a -> m[0:1];
        "#;
        let c = from_qasm2(src).unwrap();
        assert_eq!(c.n_qubits(), 4);
        assert_eq!(c.len(), 5);
        let psi = StateVector::from_circuit(&c, &[]).unwrap();
        // a = |++>, b copies a, then u2(0,π) = H on b[1]
        assert!((psi.norm() - 1.0).abs() < 1e-12);
        assert!(from_qasm2("qreg q[1]; h q;").is_err());
        // Gate definitions are expanded where they are used.
        let g = from_qasm2("OPENQASM 2.0; qreg q[2]; gate foo(t) a, b { h a; cx a, b; rz(t/2) b; } foo(pi) q[1], q[0];").unwrap();
        assert_eq!(g.len(), 3);
        assert_eq!(g.instructions()[1].controls, vec![1]);
        assert!(from_qasm2("OPENQASM 2.0; qreg q[1]; gate foo a { foo a; } foo q[0];").is_err());
        assert!(from_qasm2("OPENQASM 2.0; qreg q[1]; opaque foo a;").is_err());
        assert!(from_qasm2("OPENQASM 2.0; qreg q[1]; creg c[1]; reset q[0];").is_err());
        assert!(from_qasm2("OPENQASM 2.0; qreg q[1]; bogus q[0];").is_err());
    }

    /// Inputs found by the `qasm` fuzz target (and their relatives): each
    /// must be rejected with an error, not a panic or a stack overflow.
    #[test]
    fn malformed_programs_are_errors() {
        let header = "OPENQASM 2.0;\n";
        let deep = format!("{}1{}", "(".repeat(10_000), ")".repeat(10_000));
        let minus = format!("{}1", "-".repeat(10_000));
        let sines = format!("{}1{}", "sin(".repeat(10_000), ")".repeat(10_000));
        for body in [
            "]\u{6}[".to_string(),
            "qreg q]3[;".to_string(),
            "qreg q[2]x;".to_string(),
            "qreg [2];".to_string(),
            "qreg q[2]; qreg q[3];".to_string(),
            "qreg q[2]; h q]0[;".to_string(),
            "qreg q[2]; h q[0]];".to_string(),
            format!("qreg a[{}]; qreg b[2];", usize::MAX),
            format!("qreg q[{}];", MAX_QASM_QUBITS + 1),
            format!("qreg q[1]; rx({deep}) q[0];"),
            format!("qreg q[1]; rx({minus}) q[0];"),
            format!("qreg q[1]; rx({sines}) q[0];"),
            "qreg q[1]; rx(1/0) q[0];".to_string(),
            "qreg q[1]; rx(ln(0)) q[0];".to_string(),
            "qreg q[1]; rx(sqrt(-1)) q[0];".to_string(),
            "qreg q[1]; rx(1e999) q[0];".to_string(),
        ] {
            assert!(from_qasm2(&format!("{header}{body}")).is_err(), "{body:?}");
        }
        // Nesting within the limit still evaluates.
        let ok = format!(
            "{header}qreg q[1]; rx({}pi{}) q[0];",
            "(".repeat(60),
            ")".repeat(60)
        );
        assert!(from_qasm2(&ok).is_ok());
        // Broadcasting is bounded by the instruction limit.
        let wide = format!(
            "{header}qreg q[{MAX_QASM_QUBITS}];{}",
            " h q;".repeat(MAX_QASM_INSTRUCTIONS / MAX_QASM_QUBITS + 1)
        );
        assert!(from_qasm2(&wide).is_err());
    }
}
