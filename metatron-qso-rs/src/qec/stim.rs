//! Detector circuits in Stim's text format: export and import.

use std::fmt::Write;

use super::circuit::{DetectorCircuit, MAX_QEC_OPERATIONS, MAX_QEC_QUBITS, Noise, QecOp};
use crate::error::{QsoError, Result};
use crate::quantum::{GateKind, Instruction, Param};
use crate::stabilizer::{CliffordTarget, apply_clifford};

/// Writes each Clifford gate as Stim's H, S, S_DAG, X, Y, Z, CX and SWAP.
struct Writer<'a>(&'a mut String);

impl Writer<'_> {
    fn line(&mut self, gate: &str, qubits: &[usize]) {
        self.0.push_str(gate);
        for q in qubits {
            let _ = write!(self.0, " {q}");
        }
        self.0.push('\n');
    }
}

impl CliffordTarget for Writer<'_> {
    fn h(&mut self, q: usize) {
        self.line("H", &[q]);
    }

    fn s(&mut self, q: usize) {
        self.line("S", &[q]);
    }

    fn sdg(&mut self, q: usize) {
        self.line("S_DAG", &[q]);
    }

    fn x(&mut self, q: usize) {
        self.line("X", &[q]);
    }

    fn y(&mut self, q: usize) {
        self.line("Y", &[q]);
    }

    fn z(&mut self, q: usize) {
        self.line("Z", &[q]);
    }

    fn cx(&mut self, c: usize, t: usize) {
        self.line("CX", &[c, t]);
    }

    fn swap(&mut self, a: usize, b: usize) {
        self.line("SWAP", &[a, b]);
    }
}

fn records(out: &mut String, recs: &[usize], measured: usize) {
    for &r in recs {
        let _ = write!(out, " rec[-{}]", measured - r);
    }
    out.push('\n');
}

impl DetectorCircuit {
    /// The circuit in Stim's text format (Gidney, Quantum 5, 497 (2021)),
    /// one operation per line. Gates are written as H, S, S_DAG, X, Y, Z,
    /// CX and SWAP; measurement records are relative (`rec[-k]`), as Stim
    /// requires. Probabilities are written exactly, as the shortest
    /// decimal that reads back to the same number (with an exponent for
    /// very small ones).
    pub fn to_stim(&self) -> Result<String> {
        let mut out = String::new();
        let mut measured = 0;
        for op in self.operations() {
            match op {
                QecOp::Gate(inst) => apply_clifford(&mut Writer(&mut out), inst, &[])?,
                QecOp::Measure { qubit, flip } => {
                    if *flip > 0.0 {
                        let _ = writeln!(out, "M({flip:?}) {qubit}");
                    } else {
                        let _ = writeln!(out, "M {qubit}");
                    }
                    measured += 1;
                }
                QecOp::Reset { qubit } => {
                    let _ = writeln!(out, "R {qubit}");
                }
                QecOp::Noise { noise, qubits } => {
                    let name = match noise {
                        Noise::XError(_) => "X_ERROR",
                        Noise::YError(_) => "Y_ERROR",
                        Noise::ZError(_) => "Z_ERROR",
                        Noise::Depolarize1(_) => "DEPOLARIZE1",
                        Noise::Depolarize2(_) => "DEPOLARIZE2",
                    };
                    Writer(&mut out).line(&format!("{name}({:?})", noise.probability()), qubits);
                }
                QecOp::Detector { records: recs } => {
                    out.push_str("DETECTOR");
                    records(&mut out, recs, measured);
                }
                QecOp::Observable {
                    index,
                    records: recs,
                } => {
                    let _ = write!(out, "OBSERVABLE_INCLUDE({index})");
                    records(&mut out, recs, measured);
                }
            }
        }
        Ok(out)
    }
}

// ---- import ---------------------------------------------------------------

/// A target of a Stim instruction.
#[derive(Clone, Copy, Debug)]
enum Target {
    /// A qubit (`!q`, an inverted result, reads the same here: detection
    /// events are relative to the noiseless circuit).
    Qubit(usize),
    /// `rec[-k]`.
    Record(usize),
}

#[derive(Debug)]
enum Item {
    Instruction {
        name: String,
        args: Vec<f64>,
        targets: Vec<Target>,
        line: usize,
    },
    Repeat {
        count: u64,
        body: Vec<Item>,
    },
}

fn parse_error(line: usize, what: impl std::fmt::Display) -> QsoError {
    QsoError::Parse(format!("Stim circuit, line {line}: {what}"))
}

/// Instructions without effect here.
const IGNORED: [&str; 4] = ["TICK", "QUBIT_COORDS", "SHIFT_COORDS", "I"];

/// Every instruction the importer reads.
const SUPPORTED: [&str; 41] = [
    "TICK",
    "QUBIT_COORDS",
    "SHIFT_COORDS",
    "I",
    "DETECTOR",
    "OBSERVABLE_INCLUDE",
    "H",
    "H_XZ",
    "S",
    "SQRT_Z",
    "S_DAG",
    "SQRT_Z_DAG",
    "X",
    "Y",
    "Z",
    "SQRT_X",
    "SQRT_X_DAG",
    "SQRT_Y",
    "SQRT_Y_DAG",
    "CX",
    "CNOT",
    "ZCX",
    "CY",
    "ZCY",
    "CZ",
    "ZCZ",
    "XCZ",
    "YCZ",
    "SWAP",
    "X_ERROR",
    "Y_ERROR",
    "Z_ERROR",
    "DEPOLARIZE1",
    "DEPOLARIZE2",
    "R",
    "RZ",
    "RX",
    "RY",
    "M",
    "MZ",
    "MX",
];
/// The rest of the measurements and resets.
const SUPPORTED_MEASUREMENTS: [&str; 5] = ["MY", "MR", "MRZ", "MRX", "MRY"];

/// Operations of the detector circuit an instruction becomes (an upper
/// bound, for the size check before expansion).
fn cost(name: &str, targets: usize) -> u64 {
    let per = match name {
        "MY" | "MRY" => 6,
        "MX" | "MRX" | "RY" => 4,
        "RX" | "MR" | "MRZ" => 2,
        _ => 1,
    };
    per * targets.max(1) as u64
}

fn parse_items(
    lines: &mut std::iter::Peekable<impl Iterator<Item = (usize, String)>>,
    nested: bool,
) -> Result<Vec<Item>> {
    let mut items = Vec::new();
    while let Some((line, text)) = lines.next() {
        let text = text.split('#').next().unwrap_or("").trim().to_string();
        if text.is_empty() {
            continue;
        }
        if text == "}" {
            if nested {
                return Ok(items);
            }
            return Err(parse_error(line, "unmatched }"));
        }
        // Name, optional (arguments), targets.
        let (head, rest) = match text.find(|c: char| c == '(' || c.is_whitespace()) {
            Some(i) => text.split_at(i),
            None => (text.as_str(), ""),
        };
        let name = head.to_ascii_uppercase();
        let mut rest = rest.trim_start();
        let mut args = Vec::new();
        if let Some(inner) = rest.strip_prefix('(') {
            let close = inner
                .find(')')
                .ok_or_else(|| parse_error(line, "missing )"))?;
            for a in inner[..close]
                .split(',')
                .map(str::trim)
                .filter(|a| !a.is_empty())
            {
                let v: f64 = a
                    .parse()
                    .map_err(|_| parse_error(line, format!("{a:?} is not a number")))?;
                if !v.is_finite() {
                    return Err(parse_error(line, format!("{a} is not finite")));
                }
                args.push(v);
            }
            rest = inner[close + 1..].trim_start();
        }
        if name == "REPEAT" {
            let body = rest
                .strip_suffix('{')
                .ok_or_else(|| parse_error(line, "REPEAT needs a block: REPEAT N {"))?;
            let count: u64 = body
                .trim()
                .parse()
                .map_err(|_| parse_error(line, "REPEAT needs a count"))?;
            let body = parse_items(lines, true)?;
            items.push(Item::Repeat { count, body });
            continue;
        }
        if !SUPPORTED.contains(&name.as_str()) && !SUPPORTED_MEASUREMENTS.contains(&name.as_str()) {
            return Err(parse_error(
                line,
                format!("the Stim instruction {name} is not supported"),
            ));
        }
        let mut targets = Vec::new();
        for t in rest.split_whitespace() {
            let target = if let Some(r) = t.strip_prefix("rec[-").and_then(|r| r.strip_suffix(']'))
            {
                let k: usize = r
                    .parse()
                    .map_err(|_| parse_error(line, format!("{t} is not a record")))?;
                if k == 0 {
                    return Err(parse_error(line, "rec[-0] does not exist"));
                }
                Target::Record(k)
            } else {
                // Parsed as u64, so that the limit, not the width of usize,
                // decides on every platform.
                let q: u64 = t.strip_prefix('!').unwrap_or(t).parse().map_err(|_| {
                    parse_error(line, format!("{name} target {t} is not supported"))
                })?;
                if q >= MAX_QEC_QUBITS as u64 {
                    return Err(parse_error(
                        line,
                        format!(
                            "qubit {q} is beyond the largest register ({MAX_QEC_QUBITS} qubits)"
                        ),
                    ));
                }
                Target::Qubit(q as usize)
            };
            targets.push(target);
        }
        items.push(Item::Instruction {
            name,
            args,
            targets,
            line,
        });
    }
    if nested {
        return Err(QsoError::Parse(
            "Stim circuit: missing } of a REPEAT block".into(),
        ));
    }
    Ok(items)
}

/// Operations after expansion, and the largest qubit.
fn measure_items(items: &[Item]) -> (u64, Option<usize>) {
    let mut ops = 0u64;
    let mut max = None;
    for item in items {
        match item {
            Item::Instruction { name, targets, .. } => {
                if !IGNORED.contains(&name.as_str()) {
                    ops = ops.saturating_add(cost(name, targets.len()));
                }
                for t in targets {
                    if let Target::Qubit(q) = t {
                        max = max.max(Some(*q));
                    }
                }
            }
            Item::Repeat { count, body } => {
                let (o, m) = measure_items(body);
                ops = ops.saturating_add(o.saturating_mul(*count));
                if *count > 0 {
                    max = max.max(m);
                }
            }
        }
    }
    (ops, max)
}

struct Builder {
    c: DetectorCircuit,
}

impl Builder {
    fn gate(
        &mut self,
        kind: GateKind,
        target: usize,
        control: Option<usize>,
        angle: Option<f64>,
    ) -> Result<()> {
        self.c.gate(Instruction {
            kind,
            targets: vec![target],
            controls: control.into_iter().collect(),
            params: angle.into_iter().map(Param::Value).collect(),
            matrix: None,
        })?;
        Ok(())
    }

    fn records(&self, targets: &[Target], line: usize) -> Result<Vec<usize>> {
        let measured = self.c.n_measurements();
        targets
            .iter()
            .map(|t| match *t {
                Target::Record(k) if k <= measured => Ok(measured - k),
                Target::Record(k) => Err(parse_error(
                    line,
                    format!("rec[-{k}] before {measured} measurements"),
                )),
                Target::Qubit(_) => Err(parse_error(line, "expected measurement records")),
            })
            .collect()
    }

    fn emit(&mut self, items: &[Item]) -> Result<()> {
        for item in items {
            match item {
                // A block without operations (only TICKs) is skipped,
                // whatever its count.
                Item::Repeat { count, body } if measure_items(body).0 > 0 => {
                    for _ in 0..*count {
                        self.emit(body)?;
                    }
                }
                Item::Repeat { .. } => {}
                Item::Instruction {
                    name,
                    args,
                    targets,
                    line,
                } => self.instruction(name, args, targets, *line)?,
            }
        }
        Ok(())
    }

    fn instruction(
        &mut self,
        name: &str,
        args: &[f64],
        targets: &[Target],
        line: usize,
    ) -> Result<()> {
        use GateKind::*;
        let fail = |e: QsoError| parse_error(line, format!("{name}: {e}"));
        if IGNORED.contains(&name) {
            return Ok(());
        }
        match name {
            "DETECTOR" => {
                let r = self.records(targets, line)?;
                self.c.detector(&r).map_err(fail)?;
                return Ok(());
            }
            "OBSERVABLE_INCLUDE" => {
                let index = match args {
                    [k] if *k >= 0.0 && k.fract() == 0.0 => *k as usize,
                    _ => return Err(parse_error(line, "OBSERVABLE_INCLUDE needs one index")),
                };
                let r = self.records(targets, line)?;
                self.c.observable(index, &r).map_err(fail)?;
                return Ok(());
            }
            _ => {}
        }
        let qubits: Vec<usize> = targets
            .iter()
            .map(|t| match *t {
                Target::Qubit(q) => Ok(q),
                Target::Record(_) => Err(parse_error(
                    line,
                    format!("{name} with measurement records (classical control) is not supported"),
                )),
            })
            .collect::<Result<_>>()?;
        let p = match args {
            [] => 0.0,
            [p] => *p,
            _ => {
                return Err(parse_error(
                    line,
                    format!("{name} takes at most one argument"),
                ));
            }
        };
        let one = |b: &mut Self, kind: GateKind, angle: Option<f64>| -> Result<()> {
            for &q in &qubits {
                b.gate(kind, q, None, angle).map_err(fail)?;
            }
            Ok(())
        };
        let pairs =
            |b: &mut Self, f: &dyn Fn(&mut Self, usize, usize) -> Result<()>| -> Result<()> {
                if !qubits.len().is_multiple_of(2) {
                    return Err(parse_error(line, format!("{name} needs pairs of qubits")));
                }
                for pair in qubits.chunks(2) {
                    f(b, pair[0], pair[1]).map_err(fail)?;
                }
                Ok(())
            };
        let half = std::f64::consts::FRAC_PI_2;
        let noise = |b: &mut Self, n: Noise| -> Result<()> {
            b.c.noise(n, &qubits).map_err(fail)?;
            Ok(())
        };
        match name {
            "H" | "H_XZ" => one(self, H, None)?,
            "S" | "SQRT_Z" => one(self, S, None)?,
            "S_DAG" | "SQRT_Z_DAG" => one(self, Sdg, None)?,
            "X" => one(self, X, None)?,
            "Y" => one(self, Y, None)?,
            "Z" => one(self, Z, None)?,
            "SQRT_X" => one(self, SX, None)?,
            "SQRT_X_DAG" => one(self, SXdg, None)?,
            "SQRT_Y" => one(self, RY, Some(half))?,
            "SQRT_Y_DAG" => one(self, RY, Some(-half))?,
            "CX" | "CNOT" | "ZCX" => pairs(self, &|b, c, t| b.gate(X, t, Some(c), None))?,
            "CY" | "ZCY" => pairs(self, &|b, c, t| b.gate(Y, t, Some(c), None))?,
            "CZ" | "ZCZ" => pairs(self, &|b, c, t| b.gate(Z, t, Some(c), None))?,
            "XCZ" => pairs(self, &|b, t, c| b.gate(X, t, Some(c), None))?,
            "YCZ" => pairs(self, &|b, t, c| b.gate(Y, t, Some(c), None))?,
            "SWAP" => pairs(self, &|b, x, y| {
                b.c.gate(Instruction {
                    kind: Swap,
                    targets: vec![x, y],
                    controls: vec![],
                    params: vec![],
                    matrix: None,
                })?;
                Ok(())
            })?,
            "X_ERROR" => noise(self, Noise::XError(p))?,
            "Y_ERROR" => noise(self, Noise::YError(p))?,
            "Z_ERROR" => noise(self, Noise::ZError(p))?,
            "DEPOLARIZE1" => noise(self, Noise::Depolarize1(p))?,
            "DEPOLARIZE2" => noise(self, Noise::Depolarize2(p))?,
            "R" | "RZ" | "RX" | "RY" | "M" | "MZ" | "MX" | "MY" | "MR" | "MRZ" | "MRX" | "MRY" => {
                // Other bases rotate to Z and back: X by H, Y by S† then H.
                let basis = name
                    .trim_start_matches(['M', 'R'])
                    .chars()
                    .next()
                    .unwrap_or('Z');
                let measure = name.starts_with('M');
                let reset = name.starts_with('R') || name.starts_with("MR");
                for &q in &qubits {
                    if measure {
                        match basis {
                            'X' => self.gate(H, q, None, None),
                            'Y' => self
                                .gate(Sdg, q, None, None)
                                .and_then(|_| self.gate(H, q, None, None)),
                            _ => Ok(()),
                        }
                        .map_err(fail)?;
                        self.c.measure(q, p).map_err(fail)?;
                        if reset {
                            self.c.reset(q).map_err(fail)?;
                        }
                    } else {
                        self.c.reset(q).map_err(fail)?;
                    }
                    match basis {
                        'X' => self.gate(H, q, None, None),
                        'Y' => self
                            .gate(H, q, None, None)
                            .and_then(|_| self.gate(S, q, None, None)),
                        _ => Ok(()),
                    }
                    .map_err(fail)?;
                }
            }
            other => {
                return Err(parse_error(
                    line,
                    format!("the Stim instruction {other} is not supported"),
                ));
            }
        }
        Ok(())
    }
}

impl DetectorCircuit {
    /// A circuit in Stim's text format: the Clifford gates H, S, S_DAG,
    /// X, Y, Z, SQRT_X(_DAG), SQRT_Y(_DAG), CX, CY, CZ, XCZ, YCZ and SWAP;
    /// measurements and resets in the X, Y and Z bases (M, MX, MY, R, RX,
    /// RY, MR, MRX, MRY, with an optional flip probability); X_ERROR,
    /// Y_ERROR, Z_ERROR, DEPOLARIZE1 and DEPOLARIZE2; DETECTOR,
    /// OBSERVABLE_INCLUDE and REPEAT blocks. TICK and coordinates are
    /// ignored. The register has as many qubits as the largest target
    /// needs. Other instructions (feedback, MPP, correlated or heralded
    /// errors, PAULI_CHANNEL) are an error that names them.
    pub fn from_stim(text: &str) -> Result<DetectorCircuit> {
        let mut lines = text
            .lines()
            .enumerate()
            .map(|(i, l)| (i + 1, l.to_string()))
            .peekable();
        let items = parse_items(&mut lines, false)?;
        let (ops, max) = measure_items(&items);
        if ops > MAX_QEC_OPERATIONS as u64 {
            return Err(QsoError::invalid(
                "circuit",
                format!("the circuit expands to more than {MAX_QEC_OPERATIONS} operations"),
            ));
        }
        let n = max.map_or(1, |q| q + 1);
        let mut b = Builder {
            c: DetectorCircuit::new(n)?,
        };
        b.emit(&items)?;
        Ok(b.c)
    }
}
