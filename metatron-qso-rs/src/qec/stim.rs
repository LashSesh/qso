//! Export of detector circuits in Stim's text format.

use std::fmt::Write;

use super::circuit::{DetectorCircuit, Noise, QecOp};
use crate::error::Result;
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
