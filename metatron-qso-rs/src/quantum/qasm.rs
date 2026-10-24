//! OpenQASM 2.0 import and export.
//!
//! Export targets the gate set of the original `qelib1.inc` (Cross et al.,
//! arXiv:1707.03429) so that the output loads in every OpenQASM 2 consumer
//! (Qiskit, Cirq, tket, hardware vendors). Gates outside that set are
//! decomposed exactly (e.g. `rzz → cx·rz·cx`, `swap → 3 cx`); controlled
//! single-qubit gates without a native controlled form are decomposed into
//! `u1`, `u3` and `cx`. `cu3` is never emitted because the original and
//! Qiskit's `qelib1.inc` define it differently (by a controlled phase); the
//! exported circuits are exact under both. Uncontrolled gates are exported
//! up to a global phase, which OpenQASM 2 cannot express. When *importing*
//! `cu3`, Qiskit's definition (exact controlled-U3) is assumed.
//!
//! Import understands the same set plus the extensions shipped with Qiskit's
//! `qelib1.inc` (`sx`, `sxdg`, `p`, `cp`, `u`, `swap`, `cswap`, `crx`,
//! `cry`, `rxx`, `rzz`, `id`, `u2`), several quantum registers, register
//! broadcasting and arithmetic parameter expressions with `pi`.
//! `measure` and `barrier` are skipped; custom `gate` definitions,
//! classical control and `reset` are rejected with an error.

use std::f64::consts::PI;
use std::fmt::Write as _;

use num_complex::Complex64 as C64;

use super::circuit::{Circuit, Instruction};
use super::gates::{GateKind, matrix_1q};
use crate::error::{QsoError, Result};

fn real(v: f64) -> String {
    let s = format!("{v:?}");
    if s.contains('e') && !s.contains('.') {
        s.replacen('e', ".0e", 1)
    } else {
        s
    }
}

/// ZYZ decomposition `M = e^{iα}·U3(θ, φ, λ)` of a 2×2 unitary.
fn zyz(m: &[C64; 4]) -> (f64, f64, f64, f64) {
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

fn emit_instruction(out: &mut String, inst: &Instruction, angles: &[f64]) -> Result<()> {
    use GateKind::*;
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
        emit_instruction(&mut out, inst, &inst.angles(theta))?;
    }
    if measure_all {
        out.push_str("measure q -> c;\n");
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Recursive-descent evaluator for OpenQASM parameter expressions.
struct Expr<'a> {
    s: &'a [u8],
    pos: usize,
}

impl<'a> Expr<'a> {
    fn eval(src: &'a str) -> Result<f64> {
        let mut e = Expr {
            s: src.as_bytes(),
            pos: 0,
        };
        let v = e.sum()?;
        e.ws();
        if e.pos != e.s.len() {
            return Err(QsoError::Parse(format!(
                "trailing input in expression '{src}'"
            )));
        }
        Ok(v)
    }

    fn ws(&mut self) {
        while self.pos < self.s.len() && self.s[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.s.get(self.pos).copied()
    }

    fn sum(&mut self) -> Result<f64> {
        let mut v = self.product()?;
        while let Some(op @ (b'+' | b'-')) = self.peek() {
            self.pos += 1;
            let rhs = self.product()?;
            v = if op == b'+' { v + rhs } else { v - rhs };
        }
        Ok(v)
    }

    fn product(&mut self) -> Result<f64> {
        let mut v = self.unary()?;
        while let Some(op @ (b'*' | b'/')) = self.peek() {
            self.pos += 1;
            let rhs = self.unary()?;
            v = if op == b'*' { v * rhs } else { v / rhs };
        }
        Ok(v)
    }

    /// `^` binds tighter than unary minus and is right-associative:
    /// `-2^2 = −4`, `2^-1 = 0.5`, `2^3^2 = 2^9`.
    fn power(&mut self) -> Result<f64> {
        let base = self.atom()?;
        if self.peek() == Some(b'^') {
            self.pos += 1;
            let exp = self.unary()?;
            return Ok(base.powf(exp));
        }
        Ok(base)
    }

    fn unary(&mut self) -> Result<f64> {
        match self.peek() {
            Some(b'-') => {
                self.pos += 1;
                Ok(-self.unary()?)
            }
            Some(b'+') => {
                self.pos += 1;
                self.unary()
            }
            _ => self.power(),
        }
    }

    fn atom(&mut self) -> Result<f64> {
        match self.peek() {
            Some(b'(') => {
                self.pos += 1;
                let v = self.sum()?;
                if self.peek() != Some(b')') {
                    return Err(QsoError::Parse("missing ')' in expression".into()));
                }
                self.pos += 1;
                Ok(v)
            }
            Some(c) if c.is_ascii_digit() || c == b'.' => {
                let start = self.pos;
                while self.pos < self.s.len() {
                    let d = self.s[self.pos];
                    let prev = if self.pos > start {
                        self.s[self.pos - 1]
                    } else {
                        0
                    };
                    let mantissa = d.is_ascii_digit() || d == b'.' || d == b'e' || d == b'E';
                    let exponent_sign = (d == b'+' || d == b'-') && (prev == b'e' || prev == b'E');
                    if mantissa || exponent_sign {
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                let text = std::str::from_utf8(&self.s[start..self.pos]).expect("ascii");
                text.parse()
                    .map_err(|_| QsoError::Parse(format!("invalid number '{text}'")))
            }
            Some(c) if c.is_ascii_alphabetic() => {
                let start = self.pos;
                while self.pos < self.s.len() && self.s[self.pos].is_ascii_alphanumeric() {
                    self.pos += 1;
                }
                let name = std::str::from_utf8(&self.s[start..self.pos]).expect("ascii");
                if name == "pi" {
                    return Ok(PI);
                }
                let f: fn(f64) -> f64 = match name {
                    "sin" => f64::sin,
                    "cos" => f64::cos,
                    "tan" => f64::tan,
                    "exp" => f64::exp,
                    "ln" => f64::ln,
                    "sqrt" => f64::sqrt,
                    _ => return Err(QsoError::Parse(format!("unknown identifier '{name}'"))),
                };
                if self.peek() != Some(b'(') {
                    return Err(QsoError::Parse(format!("expected '(' after {name}")));
                }
                let arg = self.atom()?;
                Ok(f(arg))
            }
            _ => Err(QsoError::Parse("unexpected end of expression".into())),
        }
    }
}

/// Qubit operand: a single qubit or a whole register (for broadcasting).
enum Operand {
    Qubit(usize),
    Register(usize, usize),
}

fn split_top_level(s: &str, sep: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            c if c == sep && depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

/// Parse an OpenQASM 2.0 program into a [`Circuit`] (no free parameters).
///
/// Quantum registers are concatenated in declaration order.
pub fn from_qasm2(src: &str) -> Result<Circuit> {
    let mut cleaned = String::with_capacity(src.len());
    for line in src.lines() {
        let code = line.split("//").next().unwrap_or("");
        cleaned.push_str(code);
        cleaned.push('\n');
    }
    if cleaned.contains('{') {
        return Err(QsoError::Parse(
            "custom gate definitions are not supported".into(),
        ));
    }
    let mut registers: Vec<(String, usize, usize)> = Vec::new(); // name, offset, size
    let mut n_qubits = 0usize;
    let mut pending: Vec<(String, Vec<f64>, Vec<Operand>)> = Vec::new();
    let mut saw_header = false;

    let lookup = |registers: &[(String, usize, usize)], operand: &str| -> Result<Operand> {
        let operand = operand.trim();
        if let Some(open) = operand.find('[') {
            let name = operand[..open].trim();
            let close = operand
                .find(']')
                .ok_or_else(|| QsoError::Parse(format!("missing ']' in '{operand}'")))?;
            let idx: usize = operand[open + 1..close]
                .trim()
                .parse()
                .map_err(|_| QsoError::Parse(format!("invalid index in '{operand}'")))?;
            let (_, off, size) = registers
                .iter()
                .find(|(n, _, _)| n == name)
                .ok_or_else(|| QsoError::Parse(format!("unknown register '{name}'")))?;
            if idx >= *size {
                return Err(QsoError::Parse(format!(
                    "index {idx} out of range for '{name}'"
                )));
            }
            Ok(Operand::Qubit(off + idx))
        } else {
            let (_, off, size) = registers
                .iter()
                .find(|(n, _, _)| n == operand)
                .ok_or_else(|| QsoError::Parse(format!("unknown register '{operand}'")))?;
            Ok(Operand::Register(*off, *size))
        }
    };

    for raw in cleaned.split(';') {
        let stmt = raw.trim();
        if stmt.is_empty() {
            continue;
        }
        let keyword = stmt.split_whitespace().next().unwrap_or("");
        match keyword {
            "OPENQASM" => {
                let version = stmt["OPENQASM".len()..].trim();
                if !version.starts_with('2') {
                    return Err(QsoError::Parse(format!(
                        "unsupported OpenQASM version {version}"
                    )));
                }
                saw_header = true;
            }
            "include" | "creg" | "barrier" | "measure" => {}
            "qreg" => {
                let decl = stmt["qreg".len()..].trim();
                let open = decl
                    .find('[')
                    .ok_or_else(|| QsoError::Parse(format!("malformed qreg '{stmt}'")))?;
                let close = decl
                    .find(']')
                    .ok_or_else(|| QsoError::Parse(format!("malformed qreg '{stmt}'")))?;
                let name = decl[..open].trim().to_string();
                let size: usize = decl[open + 1..close]
                    .trim()
                    .parse()
                    .map_err(|_| QsoError::Parse(format!("invalid size in '{stmt}'")))?;
                registers.push((name, n_qubits, size));
                n_qubits += size;
            }
            "reset" | "if" | "opaque" | "gate" => {
                return Err(QsoError::Parse(format!("'{keyword}' is not supported")));
            }
            _ => {
                // Gate application: name[(params)] operands
                let name_end = stmt
                    .find(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
                    .unwrap_or(stmt.len());
                let rest = &stmt[name_end..];
                let (head, operands) = if rest.trim_start().starts_with('(') {
                    // Parameter list: find the matching ')' by depth.
                    let open = name_end + (rest.len() - rest.trim_start().len());
                    let mut depth = 0usize;
                    let mut close = None;
                    for (i, ch) in stmt[open..].char_indices() {
                        match ch {
                            '(' => depth += 1,
                            ')' => {
                                depth -= 1;
                                if depth == 0 {
                                    close = Some(open + i);
                                    break;
                                }
                            }
                            _ => {}
                        }
                    }
                    let close = close.ok_or_else(|| {
                        QsoError::Parse(format!("unbalanced parentheses in '{stmt}'"))
                    })?;
                    (&stmt[..=close], &stmt[close + 1..])
                } else {
                    if name_end == stmt.len() {
                        return Err(QsoError::Parse(format!("malformed statement '{stmt}'")));
                    }
                    (&stmt[..name_end], rest)
                };
                let (name, params) = match head.find('(') {
                    Some(open) => {
                        let inner = &head[open + 1..head.len() - 1];
                        let values = if inner.trim().is_empty() {
                            Vec::new()
                        } else {
                            split_top_level(inner, ',')
                                .into_iter()
                                .map(Expr::eval)
                                .collect::<Result<Vec<f64>>>()?
                        };
                        (head[..open].trim().to_string(), values)
                    }
                    None => (head.trim().to_string(), Vec::new()),
                };
                let ops = operands
                    .split(',')
                    .map(|o| lookup(&registers, o))
                    .collect::<Result<Vec<Operand>>>()?;
                pending.push((name, params, ops));
            }
        }
    }
    if !saw_header {
        return Err(QsoError::Parse("missing 'OPENQASM 2.0;' header".into()));
    }

    let mut circuit = Circuit::new(n_qubits);
    for (name, params, ops) in pending {
        // Broadcast register operands.
        let width = ops
            .iter()
            .filter_map(|o| match o {
                Operand::Register(_, s) => Some(*s),
                Operand::Qubit(_) => None,
            })
            .max();
        let reps = width.unwrap_or(1);
        for k in 0..reps {
            let qubits: Vec<usize> = ops
                .iter()
                .map(|o| match o {
                    Operand::Qubit(q) => Ok(*q),
                    Operand::Register(off, size) if *size == reps => Ok(off + k),
                    Operand::Register(..) => Err(QsoError::Parse(format!(
                        "register size mismatch in broadcast of '{name}'"
                    ))),
                })
                .collect::<Result<Vec<usize>>>()?;
            apply_named(&mut circuit, &name, &params, &qubits)?;
        }
    }
    Ok(circuit)
}

fn apply_named(c: &mut Circuit, name: &str, p: &[f64], q: &[usize]) -> Result<()> {
    let arity = |n_q: usize, n_p: usize| -> Result<()> {
        if q.len() != n_q || p.len() != n_p {
            return Err(QsoError::Parse(format!(
                "'{name}' expects {n_q} qubit(s) and {n_p} parameter(s), got {} and {}",
                q.len(),
                p.len()
            )));
        }
        Ok(())
    };
    let gate =
        |kind: GateKind, targets: &[usize], controls: &[usize], params: &[f64]| Instruction {
            kind,
            targets: targets.to_vec(),
            controls: controls.to_vec(),
            params: params.iter().map(|&v| v.into()).collect(),
            matrix: None,
        };
    use GateKind::*;
    let inst = match name {
        "id" | "u0" => {
            arity(1, if name == "u0" { 1 } else { 0 })?;
            return Ok(());
        }
        "h" | "x" | "y" | "z" | "s" | "sdg" | "t" | "tdg" | "sx" | "sxdg" => {
            arity(1, 0)?;
            let kind = match name {
                "h" => H,
                "x" => X,
                "y" => Y,
                "z" => Z,
                "s" => S,
                "sdg" => Sdg,
                "t" => T,
                "tdg" => Tdg,
                "sx" => SX,
                _ => SXdg,
            };
            gate(kind, &q[..1], &[], &[])
        }
        "rx" | "ry" | "rz" | "u1" | "p" => {
            arity(1, 1)?;
            let kind = match name {
                "rx" => RX,
                "ry" => RY,
                "rz" => RZ,
                _ => P,
            };
            gate(kind, &q[..1], &[], p)
        }
        "u3" | "u" | "U" => {
            arity(1, 3)?;
            gate(U, &q[..1], &[], p)
        }
        "u2" => {
            arity(1, 2)?;
            gate(U, &q[..1], &[], &[PI / 2.0, p[0], p[1]])
        }
        "cx" | "CX" | "cy" | "cz" | "ch" => {
            arity(2, 0)?;
            let kind = match name {
                "cy" => Y,
                "cz" => Z,
                "ch" => H,
                _ => X,
            };
            gate(kind, &q[1..2], &q[..1], &[])
        }
        "crx" | "cry" | "crz" | "cu1" | "cp" => {
            arity(2, 1)?;
            let kind = match name {
                "crx" => RX,
                "cry" => RY,
                "crz" => RZ,
                _ => P,
            };
            gate(kind, &q[1..2], &q[..1], p)
        }
        "cu3" => {
            arity(2, 3)?;
            gate(U, &q[1..2], &q[..1], p)
        }
        "ccx" => {
            arity(3, 0)?;
            gate(X, &q[2..3], &q[..2], &[])
        }
        "swap" => {
            arity(2, 0)?;
            gate(Swap, q, &[], &[])
        }
        "cswap" => {
            arity(3, 0)?;
            gate(Swap, &q[1..3], &q[..1], &[])
        }
        "rxx" | "ryy" | "rzz" => {
            arity(2, 1)?;
            let kind = match name {
                "rxx" => RXX,
                "ryy" => RYY,
                _ => RZZ,
            };
            gate(kind, q, &[], p)
        }
        _ => return Err(QsoError::Parse(format!("unsupported gate '{name}'"))),
    };
    c.try_push(inst)?;
    Ok(())
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
        assert!((Expr::eval("pi/2").unwrap() - PI / 2.0).abs() < 1e-15);
        assert!((Expr::eval("-3*pi/4 + 1e-1").unwrap() - (-3.0 * PI / 4.0 + 0.1)).abs() < 1e-15);
        assert!((Expr::eval("2^3 - sqrt(4)").unwrap() - 6.0).abs() < 1e-15);
        assert!((Expr::eval("cos(0)+-(1)").unwrap()).abs() < 1e-15);
        assert!(Expr::eval("pi pi").is_err());
        assert!(Expr::eval("foo").is_err());
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
        assert!(from_qasm2("OPENQASM 2.0; qreg q[1]; gate foo a { h a; }").is_err());
        assert!(from_qasm2("OPENQASM 2.0; qreg q[1]; bogus q[0];").is_err());
    }
}
