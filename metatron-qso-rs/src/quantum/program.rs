//! Programs with mid-circuit measurements, resets and classical
//! conditions — the dynamic part of OpenQASM.
//!
//! A [`Program`] is a list of [`Operation`]s on `n_qubits` qubits and
//! `n_bits` classical bits: gates, measurements into bits, resets to |0⟩,
//! and blocks executed when a classical condition holds. Registers only
//! name ranges of qubits and bits; they matter for printing and for
//! conditions on whole registers.
//!
//! [`Program::probabilities`] returns the exact distribution of the
//! classical bits. A program whose measurements all come after the last
//! gate on their qubit, without resets or conditions, needs one
//! simulation; otherwise every measurement splits the computation into
//! branches, which is exact but grows with the number of branches with
//! non-zero probability (at most [`MAX_BRANCHES`]). [`Program::sample`]
//! samples shots from that distribution, or simulates shot by shot when it
//! has too many branches.
//!
//! ```
//! use metatron_qso::quantum::Program;
//!
//! // Teleport RY(0.8)|0⟩ from qubit 0 to qubit 2 and measure it.
//! let p = Program::from_qasm(
//!     "OPENQASM 3.0; include \"stdgates.inc\";
//!      qubit[3] q; bit[3] c;
//!      ry(0.8) q[0]; h q[1]; cx q[1], q[2];
//!      cx q[0], q[1]; h q[0];
//!      c[0] = measure q[0]; c[1] = measure q[1];
//!      if (c[1] == 1) x q[2];
//!      if (c[0] == 1) z q[2];
//!      c[2] = measure q[2];",
//! ).unwrap();
//! let dist = p.probabilities().unwrap();
//! let p1: f64 = dist.iter().filter(|(b, _)| *b >> 2 & 1 == 1).map(|(_, p)| p).sum();
//! assert!((p1 - (0.4f64).sin().powi(2)).abs() < 1e-12);
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::circuit::{Circuit, Instruction};
use super::state::{MAX_QUBITS, StateVector};
use crate::error::{QsoError, Result};
use crate::rng::Rng;

/// A classical record as a bit string of `n_bits` digits, bit 0 on the
/// right (like [`bitstring`](super::bitstring), but for 64-bit records on
/// every platform).
pub fn record_bitstring(record: u64, n_bits: usize) -> String {
    (0..n_bits)
        .rev()
        .map(|k| {
            if k < 64 && (record >> k) & 1 == 1 {
                '1'
            } else {
                '0'
            }
        })
        .collect()
}

/// Most measurement branches [`Program::probabilities`] follows.
pub const MAX_BRANCHES: usize = 1 << 12;
/// Most amplitudes held over all branches at once.
pub const MAX_BRANCH_AMPLITUDES: usize = 1 << 26;
/// Most shots × operations of shot-by-shot simulation.
pub const MAX_SHOT_WORK: usize = 1 << 30;
/// Most classical bits (outcomes are 64-bit masks).
pub const MAX_BITS: usize = 64;
/// Most qubits a program may declare; simulating it needs at most
/// [`MAX_QUBITS`].
pub const MAX_PROGRAM_QUBITS: usize = 1 << 14;

/// A named range of qubits or bits.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Register {
    pub name: String,
    pub offset: usize,
    pub size: usize,
}

/// Classical condition: the bits `bits` (least significant first) read as
/// an integer equal (or, with `equal = false`, differ from) `value`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Condition {
    pub bits: Vec<usize>,
    pub value: u64,
    pub equal: bool,
}

impl Condition {
    fn holds(&self, record: u64) -> bool {
        let v = self
            .bits
            .iter()
            .enumerate()
            .fold(0u64, |acc, (k, &b)| acc | (((record >> b) & 1) << k));
        (v == self.value) == self.equal
    }
}

/// One step of a [`Program`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Operation {
    Gate(Instruction),
    Measure {
        qubit: usize,
        bit: usize,
    },
    Reset {
        qubit: usize,
    },
    If {
        condition: Condition,
        body: Vec<Operation>,
    },
}

/// Qubits, bits, registers and operations of a dynamic circuit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Program {
    n_qubits: usize,
    n_bits: usize,
    qubit_registers: Vec<Register>,
    bit_registers: Vec<Register>,
    operations: Vec<Operation>,
}

impl Program {
    /// Empty program with one register `q` of `n_qubits` qubits and one
    /// register `c` of `n_bits` bits (if any).
    pub fn new(n_qubits: usize, n_bits: usize) -> Result<Self> {
        let mut p = Self::with_registers(Vec::new(), Vec::new())?;
        if n_qubits > 0 {
            p.add_qubit_register("q", n_qubits)?;
        }
        if n_bits > 0 {
            p.add_bit_register("c", n_bits)?;
        }
        Ok(p)
    }

    pub(crate) fn with_registers(qubits: Vec<Register>, bits: Vec<Register>) -> Result<Self> {
        let mut p = Self {
            n_qubits: 0,
            n_bits: 0,
            qubit_registers: Vec::new(),
            bit_registers: Vec::new(),
            operations: Vec::new(),
        };
        for r in qubits {
            p.add_qubit_register(&r.name, r.size)?;
        }
        for r in bits {
            p.add_bit_register(&r.name, r.size)?;
        }
        Ok(p)
    }

    /// Append a quantum register; its qubits follow the existing ones.
    pub fn add_qubit_register(&mut self, name: &str, size: usize) -> Result<&Register> {
        if self
            .qubit_registers
            .iter()
            .chain(&self.bit_registers)
            .any(|r| r.name == name)
        {
            return Err(QsoError::invalid(
                "register",
                format!("'{name}' declared twice"),
            ));
        }
        let total = self
            .n_qubits
            .checked_add(size)
            .filter(|&t| t <= MAX_PROGRAM_QUBITS);
        let Some(total) = total else {
            return Err(QsoError::TooManyQubits {
                what: "program",
                max: MAX_PROGRAM_QUBITS,
                requested: self.n_qubits.saturating_add(size),
            });
        };
        self.qubit_registers.push(Register {
            name: name.to_string(),
            offset: self.n_qubits,
            size,
        });
        self.n_qubits = total;
        Ok(self.qubit_registers.last().expect("just pushed"))
    }

    /// Append a classical register; its bits follow the existing ones.
    pub fn add_bit_register(&mut self, name: &str, size: usize) -> Result<&Register> {
        if self
            .qubit_registers
            .iter()
            .chain(&self.bit_registers)
            .any(|r| r.name == name)
        {
            return Err(QsoError::invalid(
                "register",
                format!("'{name}' declared twice"),
            ));
        }
        let total = self.n_bits.checked_add(size).filter(|&t| t <= MAX_BITS);
        let Some(total) = total else {
            return Err(QsoError::invalid(
                "register",
                format!("programs may have at most {MAX_BITS} classical bits"),
            ));
        };
        self.bit_registers.push(Register {
            name: name.to_string(),
            offset: self.n_bits,
            size,
        });
        self.n_bits = total;
        Ok(self.bit_registers.last().expect("just pushed"))
    }

    /// The program of `circuit` (parameters bound to `theta`), with every
    /// qubit measured into the bit of the same index if `measure_all`.
    pub fn from_circuit(circuit: &Circuit, theta: &[f64], measure_all: bool) -> Result<Self> {
        let bound = circuit.bind(theta)?;
        let n = circuit.n_qubits();
        let mut p = Self::new(n, if measure_all { n } else { 0 })?;
        for inst in bound.instructions() {
            p.push(Operation::Gate(inst.clone()))?;
        }
        if measure_all {
            for q in 0..n {
                p.push(Operation::Measure { qubit: q, bit: q })?;
            }
        }
        Ok(p)
    }

    fn validate(&self, op: &Operation) -> Result<()> {
        match op {
            Operation::Gate(inst) => {
                inst.validate(self.n_qubits)?;
                if inst.is_parametrised() {
                    return Err(QsoError::invalid(
                        "program",
                        "gates in programs must have numeric angles",
                    ));
                }
            }
            Operation::Measure { qubit, bit } => {
                self.check_qubit(*qubit)?;
                self.check_bit(*bit)?;
            }
            Operation::Reset { qubit } => self.check_qubit(*qubit)?,
            Operation::If { condition, body } => {
                if condition.bits.is_empty() || condition.bits.len() > 64 {
                    return Err(QsoError::invalid("condition", "needs 1 to 64 bits"));
                }
                for &b in &condition.bits {
                    self.check_bit(b)?;
                }
                if condition.bits.len() < 64 && condition.value >> condition.bits.len() != 0 {
                    return Err(QsoError::invalid(
                        "condition",
                        format!(
                            "value {} does not fit into {} bits",
                            condition.value,
                            condition.bits.len()
                        ),
                    ));
                }
                for o in body {
                    self.validate(o)?;
                }
            }
        }
        Ok(())
    }

    fn check_simulable(&self) -> Result<()> {
        if self.n_qubits > MAX_QUBITS {
            return Err(QsoError::TooManyQubits {
                what: "program simulation",
                max: MAX_QUBITS,
                requested: self.n_qubits,
            });
        }
        Ok(())
    }

    fn check_qubit(&self, q: usize) -> Result<()> {
        if q >= self.n_qubits {
            return Err(QsoError::QubitOutOfRange {
                qubit: q,
                n_qubits: self.n_qubits,
            });
        }
        Ok(())
    }

    fn check_bit(&self, b: usize) -> Result<()> {
        if b >= self.n_bits {
            return Err(QsoError::invalid(
                "bit",
                format!("bit {b} out of range for {} bits", self.n_bits),
            ));
        }
        Ok(())
    }

    /// Append a validated operation.
    pub fn push(&mut self, op: Operation) -> Result<&mut Self> {
        self.validate(&op)?;
        self.operations.push(op);
        Ok(self)
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    pub fn n_bits(&self) -> usize {
        self.n_bits
    }

    pub fn qubit_registers(&self) -> &[Register] {
        &self.qubit_registers
    }

    pub fn bit_registers(&self) -> &[Register] {
        &self.bit_registers
    }

    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }

    /// Number of operations, counting the bodies of conditions.
    pub fn len(&self) -> usize {
        fn count(ops: &[Operation]) -> usize {
            ops.iter()
                .map(|o| match o {
                    Operation::If { body, .. } => 1 + count(body),
                    _ => 1,
                })
                .sum()
        }
        count(&self.operations)
    }

    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Whether the program has only gates.
    pub fn is_unitary(&self) -> bool {
        self.operations
            .iter()
            .all(|o| matches!(o, Operation::Gate(_)))
    }

    /// The gates as a [`Circuit`]; an error if the program measures,
    /// resets or branches.
    pub fn to_circuit(&self) -> Result<Circuit> {
        if !self.is_unitary() {
            return Err(QsoError::invalid(
                "program",
                "has measurements, resets or conditions; use the unitary part explicitly",
            ));
        }
        self.gates_skipping_measurements()
    }

    /// The gates, with measurements dropped (resets and conditions are
    /// errors) — the reading of OpenQASM 2 programs by `from_qasm2`.
    pub(crate) fn gates_skipping_measurements(&self) -> Result<Circuit> {
        let mut c = Circuit::new(self.n_qubits);
        for op in &self.operations {
            match op {
                Operation::Gate(inst) => {
                    c.try_push(inst.clone())?;
                }
                Operation::Measure { .. } => {}
                Operation::Reset { .. } => {
                    return Err(QsoError::Parse("'reset' is not supported here".into()));
                }
                Operation::If { .. } => {
                    return Err(QsoError::Parse(
                        "classical conditions are not supported here".into(),
                    ));
                }
            }
        }
        Ok(c)
    }

    /// If the program is gates followed by measurements of qubits no gate
    /// touches afterwards: the gates and the `(qubit, bit)` pairs.
    fn terminal_form(&self) -> Option<(Circuit, Vec<(usize, usize)>)> {
        let mut c = Circuit::new(self.n_qubits);
        let mut measured: Vec<(usize, usize)> = Vec::new();
        for op in &self.operations {
            match op {
                Operation::Gate(inst) => {
                    if inst.qubits().any(|q| measured.iter().any(|m| m.0 == q)) {
                        return None;
                    }
                    c.try_push(inst.clone()).ok()?;
                }
                Operation::Measure { qubit, bit } => measured.push((*qubit, *bit)),
                _ => return None,
            }
        }
        Some((c, measured))
    }

    /// Exact probabilities of the classical records (bit `k` of the key is
    /// bit `k`; unwritten bits are 0). Probabilities below 10⁻¹⁵ are
    /// dropped.
    pub fn probabilities(&self) -> Result<BTreeMap<u64, f64>> {
        self.check_simulable()?;
        let mut out: BTreeMap<u64, f64> = BTreeMap::new();
        if let Some((c, measured)) = self.terminal_form() {
            let psi = StateVector::from_circuit(&c, &[])?;
            for (b, p) in psi.probabilities().into_iter().enumerate() {
                if p <= 0.0 {
                    continue;
                }
                let mut record = 0u64;
                for &(q, bit) in &measured {
                    let v = ((b >> q) & 1) as u64;
                    record = (record & !(1 << bit)) | (v << bit);
                }
                *out.entry(record).or_default() += p;
            }
        } else {
            let start = Branch {
                weight: 1.0,
                state: StateVector::zero(self.n_qubits),
                record: 0,
            };
            let branches = run_branches(vec![start], &self.operations, self.n_qubits)?;
            for b in branches {
                *out.entry(b.record).or_default() += b.weight;
            }
        }
        out.retain(|_, p| *p > 1e-15);
        Ok(out)
    }

    /// `shots` records sampled with a seeded generator: from the exact
    /// distribution when it has at most [`MAX_BRANCHES`] branches,
    /// otherwise by simulating every shot.
    pub fn sample(&self, shots: usize, seed: u64) -> Result<BTreeMap<u64, usize>> {
        self.check_simulable()?;
        let mut rng = Rng::seed_from_u64(seed);
        let mut counts: BTreeMap<u64, usize> = BTreeMap::new();
        match self.probabilities() {
            Ok(dist) => {
                let entries: Vec<(u64, f64)> = dist.into_iter().collect();
                let total: f64 = entries.iter().map(|e| e.1).sum();
                let mut cumulative = Vec::with_capacity(entries.len());
                let mut acc = 0.0;
                for e in &entries {
                    acc += e.1 / total;
                    cumulative.push(acc);
                }
                for _ in 0..shots {
                    let u = rng.uniform();
                    let k = cumulative
                        .partition_point(|&x| x <= u)
                        .min(entries.len() - 1);
                    *counts.entry(entries[k].0).or_default() += 1;
                }
            }
            Err(QsoError::InvalidArgument {
                name: "branches", ..
            }) => {
                if shots.saturating_mul(self.len().max(1)) > MAX_SHOT_WORK {
                    return Err(QsoError::invalid(
                        "shots",
                        format!("shots × operations exceed {MAX_SHOT_WORK}"),
                    ));
                }
                for _ in 0..shots {
                    let mut psi = StateVector::zero(self.n_qubits);
                    let mut record = 0u64;
                    run_shot(&mut psi, &mut record, &self.operations, &mut rng)?;
                    *counts.entry(record).or_default() += 1;
                }
            }
            Err(e) => return Err(e),
        }
        Ok(counts)
    }
}

struct Branch {
    weight: f64,
    state: StateVector,
    record: u64,
}

/// Probability that qubit `q` is |1⟩.
fn p_one(psi: &StateVector, q: usize) -> f64 {
    psi.amplitudes()
        .iter()
        .enumerate()
        .filter(|(b, _)| (b >> q) & 1 == 1)
        .map(|(_, a)| a.norm_sqr())
        .sum::<f64>()
        .clamp(0.0, 1.0)
}

/// Project qubit `q` onto `outcome` and renormalise by `p` (its
/// probability).
fn project(psi: &mut StateVector, q: usize, outcome: bool, p: f64) {
    let s = 1.0 / p.sqrt();
    for (b, a) in psi.amplitudes_mut().iter_mut().enumerate() {
        if ((b >> q) & 1 == 1) == outcome {
            *a *= s;
        } else {
            *a = num_complex::Complex64::new(0.0, 0.0);
        }
    }
}

fn flip(psi: &mut StateVector, q: usize) {
    let amps = psi.amplitudes_mut();
    let bit = 1usize << q;
    for b in 0..amps.len() {
        if b & bit == 0 {
            amps.swap(b, b | bit);
        }
    }
}

fn run_branches(mut branches: Vec<Branch>, ops: &[Operation], n: usize) -> Result<Vec<Branch>> {
    let too_many = || {
        QsoError::invalid(
            "branches",
            format!(
                "more than {MAX_BRANCHES} measurement branches (or {MAX_BRANCH_AMPLITUDES} amplitudes); sample shot by shot instead"
            ),
        )
    };
    for op in ops {
        match op {
            Operation::Gate(inst) => {
                for b in &mut branches {
                    b.state.apply_instruction(inst, &[])?;
                }
            }
            Operation::Measure { qubit, bit } => {
                branches = split(branches, *qubit, Some(*bit), false, n).ok_or_else(too_many)?;
            }
            Operation::Reset { qubit } => {
                branches = split(branches, *qubit, None, true, n).ok_or_else(too_many)?;
            }
            Operation::If { condition, body } => {
                let (hit, miss): (Vec<Branch>, Vec<Branch>) = branches
                    .into_iter()
                    .partition(|b| condition.holds(b.record));
                let mut hit = run_branches(hit, body, n)?;
                hit.extend(miss);
                branches = hit;
            }
        }
        if branches.len() > MAX_BRANCHES
            || branches.len().saturating_mul(1usize << n) > MAX_BRANCH_AMPLITUDES
        {
            return Err(too_many());
        }
    }
    Ok(branches)
}

/// Split every branch on the value of `qubit`: record it in `bit`, and for
/// a reset flip the |1⟩ branch back to |0⟩.
/// Split every branch by the outcome of measuring `qubit`; `None` if the
/// result would exceed [`MAX_BRANCHES`] or [`MAX_BRANCH_AMPLITUDES`] (checked
/// before any state is copied).
fn split(
    branches: Vec<Branch>,
    qubit: usize,
    bit: Option<usize>,
    reset: bool,
    n: usize,
) -> Option<Vec<Branch>> {
    let p_ones: Vec<f64> = branches.iter().map(|b| p_one(&b.state, qubit)).collect();
    let count: usize = p_ones
        .iter()
        .map(|&p1| if p1 > 1e-15 && 1.0 - p1 > 1e-15 { 2 } else { 1 })
        .sum();
    if count > MAX_BRANCHES || count.saturating_mul(1usize << n) > MAX_BRANCH_AMPLITUDES {
        return None;
    }
    let mut out = Vec::with_capacity(count);
    for (b, p1) in branches.into_iter().zip(p_ones) {
        let p0 = 1.0 - p1;
        let set = |record: u64, v: bool| match bit {
            Some(k) => (record & !(1 << k)) | (u64::from(v) << k),
            None => record,
        };
        if p1 > 1e-15 && p0 > 1e-15 {
            let mut one = b.state.clone();
            project(&mut one, qubit, true, p1);
            if reset {
                flip(&mut one, qubit);
            }
            let mut zero = b.state;
            project(&mut zero, qubit, false, p0);
            out.push(Branch {
                weight: b.weight * p0,
                state: zero,
                record: set(b.record, false),
            });
            out.push(Branch {
                weight: b.weight * p1,
                state: one,
                record: set(b.record, true),
            });
        } else {
            let outcome = p1 > 1e-15;
            let mut state = b.state;
            if outcome && reset {
                flip(&mut state, qubit);
            }
            out.push(Branch {
                weight: b.weight,
                state,
                record: set(b.record, outcome),
            });
        }
    }
    Some(out)
}

fn run_shot(
    psi: &mut StateVector,
    record: &mut u64,
    ops: &[Operation],
    rng: &mut Rng,
) -> Result<()> {
    for op in ops {
        match op {
            Operation::Gate(inst) => psi.apply_instruction(inst, &[])?,
            Operation::Measure { qubit, bit } => {
                let v = psi.measure(*qubit, rng)?;
                *record = (*record & !(1 << bit)) | (u64::from(v) << bit);
            }
            Operation::Reset { qubit } => {
                if psi.measure(*qubit, rng)? {
                    flip(psi, *qubit);
                }
            }
            Operation::If { condition, body } => {
                if condition.holds(*record) {
                    run_shot(psi, record, body, rng)?;
                }
            }
        }
    }
    Ok(())
}
