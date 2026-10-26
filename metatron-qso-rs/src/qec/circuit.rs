//! Noisy Clifford circuits with measurement records, detectors and logical
//! observables.

use crate::error::{QsoError, Result};
use crate::quantum::{Circuit, GateKind, Instruction};
use crate::stabilizer::program::{is_clifford_circuit, is_clifford_instruction};
use crate::stabilizer::validate_instruction;

/// Largest register of a detector circuit.
pub const MAX_QEC_QUBITS: usize = 1 << 16;
/// Most operations of a detector circuit.
pub const MAX_QEC_OPERATIONS: usize = 1 << 24;
/// Most logical observables (they are reported as a 64-bit mask).
pub const MAX_OBSERVABLES: usize = 64;

/// A Pauli noise channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Noise {
    /// X with probability p.
    XError(f64),
    /// Y with probability p.
    YError(f64),
    /// Z with probability p.
    ZError(f64),
    /// X, Y or Z, each with probability p/3 (p ≤ 3/4).
    Depolarize1(f64),
    /// One of the 15 non-identity two-qubit Paulis, each with probability
    /// p/15 (p ≤ 15/16), on consecutive pairs of the qubits.
    Depolarize2(f64),
}

impl Noise {
    pub fn probability(&self) -> f64 {
        match *self {
            Noise::XError(p)
            | Noise::YError(p)
            | Noise::ZError(p)
            | Noise::Depolarize1(p)
            | Noise::Depolarize2(p) => p,
        }
    }

    fn max_probability(&self) -> f64 {
        match self {
            Noise::Depolarize1(_) => 0.75,
            Noise::Depolarize2(_) => 15.0 / 16.0,
            _ => 1.0,
        }
    }
}

/// One operation of a [`DetectorCircuit`].
#[derive(Clone, Debug, PartialEq)]
pub enum QecOp {
    /// A Clifford gate.
    Gate(Instruction),
    /// Measure in the Z basis; the result is reported flipped with
    /// probability `flip`. Appends one bit to the measurement record.
    Measure { qubit: usize, flip: f64 },
    /// Reset to `|0⟩`.
    Reset { qubit: usize },
    /// Pauli noise on the qubits.
    Noise { noise: Noise, qubits: Vec<usize> },
    /// A detector: the parity of these measurement records, which must be
    /// deterministic without noise.
    Detector { records: Vec<usize> },
    /// Add the parity of these records to logical observable `index`.
    Observable { index: usize, records: Vec<usize> },
}

/// A Clifford circuit with Pauli noise, a measurement record, detectors
/// and logical observables, in the manner of Stim (Gidney, Quantum 5, 497
/// (2021)).
///
/// Detectors are parities of measurements that are deterministic in the
/// noiseless circuit; samplers report whether noise flipped them
/// (detection events), and whether it flipped each logical observable.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DetectorCircuit {
    n: usize,
    ops: Vec<QecOp>,
    measurements: usize,
    detectors: usize,
    observables: usize,
}

fn check_probability(p: f64, max: f64) -> Result<()> {
    if !(0.0..=max).contains(&p) {
        return Err(QsoError::invalid(
            "probability",
            format!("{p} is not a probability in [0, {max}]"),
        ));
    }
    Ok(())
}

impl DetectorCircuit {
    /// An empty circuit on `n` qubits, all starting in `|0⟩`.
    pub fn new(n: usize) -> Result<Self> {
        if n == 0 || n > MAX_QEC_QUBITS {
            return Err(QsoError::TooManyQubits {
                what: "detector circuits (1 or more)",
                max: MAX_QEC_QUBITS,
                requested: n,
            });
        }
        Ok(DetectorCircuit {
            n,
            ..Default::default()
        })
    }

    pub fn n_qubits(&self) -> usize {
        self.n
    }

    pub fn operations(&self) -> &[QecOp] {
        &self.ops
    }

    pub fn n_measurements(&self) -> usize {
        self.measurements
    }

    pub fn n_detectors(&self) -> usize {
        self.detectors
    }

    pub fn n_observables(&self) -> usize {
        self.observables
    }

    fn push(&mut self, op: QecOp) -> Result<()> {
        if self.ops.len() >= MAX_QEC_OPERATIONS {
            return Err(QsoError::invalid(
                "circuit",
                format!("more than {MAX_QEC_OPERATIONS} operations"),
            ));
        }
        self.ops.push(op);
        Ok(())
    }

    fn check(&self, q: usize) -> Result<()> {
        if q >= self.n {
            return Err(QsoError::QubitOutOfRange {
                qubit: q,
                n_qubits: self.n,
            });
        }
        Ok(())
    }

    /// Append a Clifford gate (parameters must be numbers).
    pub fn gate(&mut self, inst: Instruction) -> Result<&mut Self> {
        validate_instruction(&inst, &[], self.n)?;
        if !is_clifford_instruction(&inst, &[]) {
            return Err(QsoError::invalid(
                "gate",
                format!("{} is not a Clifford gate", inst.kind.name()),
            ));
        }
        self.push(QecOp::Gate(inst))?;
        Ok(self)
    }

    /// Append every gate of a Clifford circuit (parameters bound to
    /// `theta`) on the first qubits of this one. Nothing is appended if
    /// any gate is not Clifford.
    pub fn append_circuit(&mut self, circuit: &Circuit, theta: &[f64]) -> Result<&mut Self> {
        if circuit.n_qubits() > self.n {
            return Err(QsoError::invalid(
                "circuit",
                format!(
                    "{} qubits do not fit a detector circuit of {}",
                    circuit.n_qubits(),
                    self.n
                ),
            ));
        }
        let bound = circuit.bind(theta)?;
        if !is_clifford_circuit(&bound, &[]) {
            return Err(QsoError::invalid(
                "circuit",
                "every gate must be a Clifford gate",
            ));
        }
        if self.ops.len() + bound.instructions().len() > MAX_QEC_OPERATIONS {
            return Err(QsoError::invalid(
                "circuit",
                format!("more than {MAX_QEC_OPERATIONS} operations"),
            ));
        }
        for inst in bound.instructions() {
            self.gate(inst.clone())?;
        }
        Ok(self)
    }

    fn simple(
        &mut self,
        kind: GateKind,
        targets: &[usize],
        controls: &[usize],
    ) -> Result<&mut Self> {
        self.gate(Instruction {
            kind,
            targets: targets.to_vec(),
            controls: controls.to_vec(),
            params: vec![],
            matrix: None,
        })
    }

    pub fn h(&mut self, q: usize) -> Result<&mut Self> {
        self.simple(GateKind::H, &[q], &[])
    }

    pub fn s(&mut self, q: usize) -> Result<&mut Self> {
        self.simple(GateKind::S, &[q], &[])
    }

    pub fn x(&mut self, q: usize) -> Result<&mut Self> {
        self.simple(GateKind::X, &[q], &[])
    }

    pub fn z(&mut self, q: usize) -> Result<&mut Self> {
        self.simple(GateKind::Z, &[q], &[])
    }

    pub fn cx(&mut self, control: usize, target: usize) -> Result<&mut Self> {
        self.simple(GateKind::X, &[target], &[control])
    }

    pub fn cz(&mut self, a: usize, b: usize) -> Result<&mut Self> {
        self.simple(GateKind::Z, &[b], &[a])
    }

    /// Measure `qubit`, reporting a flipped result with probability
    /// `flip`; returns the index of the result in the measurement record.
    pub fn measure(&mut self, qubit: usize, flip: f64) -> Result<usize> {
        self.check(qubit)?;
        check_probability(flip, 1.0)?;
        self.push(QecOp::Measure { qubit, flip })?;
        self.measurements += 1;
        Ok(self.measurements - 1)
    }

    pub fn reset(&mut self, qubit: usize) -> Result<&mut Self> {
        self.check(qubit)?;
        self.push(QecOp::Reset { qubit })?;
        Ok(self)
    }

    /// Pauli noise on `qubits` (pairs of them for [`Noise::Depolarize2`]).
    pub fn noise(&mut self, noise: Noise, qubits: &[usize]) -> Result<&mut Self> {
        check_probability(noise.probability(), noise.max_probability())?;
        for &q in qubits {
            self.check(q)?;
        }
        if let Noise::Depolarize2(_) = noise {
            if !qubits.len().is_multiple_of(2) {
                return Err(QsoError::invalid(
                    "qubits",
                    "two-qubit noise needs an even number of qubits",
                ));
            }
            if qubits.chunks(2).any(|p| p[0] == p[1]) {
                return Err(QsoError::invalid(
                    "qubits",
                    "two-qubit noise acts on two different qubits",
                ));
            }
        }
        if noise.probability() > 0.0 && !qubits.is_empty() {
            self.push(QecOp::Noise {
                noise,
                qubits: qubits.to_vec(),
            })?;
        }
        Ok(self)
    }

    fn check_records(&self, records: &[usize]) -> Result<()> {
        if let Some(&r) = records.iter().find(|&&r| r >= self.measurements) {
            return Err(QsoError::invalid(
                "records",
                format!(
                    "record {r} does not exist yet ({} measurements so far)",
                    self.measurements
                ),
            ));
        }
        Ok(())
    }

    /// Declare a detector on these measurement records; returns its index.
    pub fn detector(&mut self, records: &[usize]) -> Result<usize> {
        self.check_records(records)?;
        self.push(QecOp::Detector {
            records: records.to_vec(),
        })?;
        self.detectors += 1;
        Ok(self.detectors - 1)
    }

    /// Add these measurement records to logical observable `index`.
    pub fn observable(&mut self, index: usize, records: &[usize]) -> Result<&mut Self> {
        if index >= MAX_OBSERVABLES {
            return Err(QsoError::invalid(
                "observable",
                format!("observable indices are below {MAX_OBSERVABLES}"),
            ));
        }
        self.check_records(records)?;
        self.push(QecOp::Observable {
            index,
            records: records.to_vec(),
        })?;
        self.observables = self.observables.max(index + 1);
        Ok(self)
    }

    /// The circuit without noise (measurement flips set to zero).
    pub fn without_noise(&self) -> DetectorCircuit {
        let ops = self
            .ops
            .iter()
            .filter(|op| !matches!(op, QecOp::Noise { .. }))
            .map(|op| match op {
                QecOp::Measure { qubit, .. } => QecOp::Measure {
                    qubit: *qubit,
                    flip: 0.0,
                },
                other => other.clone(),
            })
            .collect();
        DetectorCircuit {
            ops,
            ..self.clone()
        }
    }
}
