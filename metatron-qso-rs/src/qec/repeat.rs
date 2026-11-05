//! Detector circuits that keep their repeated blocks, as Stim's `REPEAT`.
//!
//! A memory experiment of 10⁵ rounds repeats one round 10⁵ times. A
//! [`RepeatedCircuit`] stores the round once, with its measurement records
//! counted back from the operation that reads them (Stim's `rec[-k]`), so
//! that every iteration reads the same relative records; nothing is
//! expanded until an operation needs it, and then only as it runs.
//!
//! * Sampling runs the operations in order and keeps only the
//!   measurement records a detector can still reach.
//! * The detector error model comes from one pass backwards (see
//!   [`super::dem`]): through a block it carries, for every qubit, what an
//!   X or a Z there would flip. Once the state at the start of an iteration
//!   is that at its end with every detector and record shifted by one
//!   iteration's, each earlier iteration repeats the last one's errors
//!   shifted, and the pass jumps to the start of the block, adding the
//!   errors of the iterations it skipped, shifted, without propagating
//!   them (as Stim folds loops). A memory experiment repeats after two or
//!   three rounds; the model of 10⁵ rounds then costs its errors' count.

use std::fmt::Write;

use super::circuit::{
    DetectorCircuit, MAX_QEC_OPERATIONS, Noise, QecOp, check_gate, check_noise, check_observable,
    check_probability, check_qubit,
};
use super::dem::{CorrelatedErrorModel, DetectorErrorModel};
use super::frame::{self, DetectorSamples, MAX_SAMPLED_EVENTS, Walk};
use super::stim::{self, Sink};
use crate::error::{QsoError, Result};
use crate::quantum::Instruction;
use crate::rng::Rng;

/// Most operations of a repeated circuit expanded (which also bounds its
/// measurements and detectors).
pub const MAX_REPEATED_OPERATIONS: u64 = 1 << 31;

/// A block of a [`RepeatedCircuit`].
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Block {
    /// Operations whose records count back from the operation (1: the
    /// last measurement before it).
    Ops(Vec<QecOp>),
    Repeat(Box<Repeat>),
}

/// A block repeated `count` times.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Repeat {
    pub(super) count: usize,
    pub(super) body: Vec<Block>,
    /// Measurements, detectors and operations of one iteration.
    pub(super) measurements: usize,
    pub(super) detectors: usize,
    pub(super) operations: u64,
}

/// A detector circuit with repeated blocks, from Stim's text format
/// ([`RepeatedCircuit::from_stim`]) or the memory experiments
/// ([`super::repetition_code_memory_repeated`] and its siblings): the same
/// operations as [`DetectorCircuit`], but a block of `REPEAT n` is stored
/// once (see the module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct RepeatedCircuit {
    n: usize,
    pub(super) blocks: Vec<Block>,
    measurements: usize,
    detectors: usize,
    observables: usize,
    operations: u64,
    stored: usize,
    /// The furthest any detector or observable reaches back.
    lookback: usize,
}

/// Builds a repeated circuit operation by operation, blocks opened and
/// closed around them.
pub(super) struct Builder {
    n: usize,
    /// Open blocks, the whole circuit first: their blocks so far, count,
    /// and the measurements, detectors and operations when they opened.
    stack: Vec<(Vec<Block>, usize, (usize, usize, u64))>,
    measured: usize,
    detectors: usize,
    observables: usize,
    operations: u64,
    stored: usize,
    lookback: usize,
}

impl Builder {
    pub(super) fn new(n: usize) -> Result<Self> {
        // The register's limits.
        DetectorCircuit::new(n)?;
        Ok(Builder {
            n,
            stack: vec![(Vec::new(), 1, (0, 0, 0))],
            measured: 0,
            detectors: 0,
            observables: 0,
            operations: 0,
            stored: 0,
            lookback: 0,
        })
    }

    fn check_lookbacks(&self, lookbacks: &[usize]) -> Result<()> {
        if let Some(&k) = lookbacks.iter().find(|&&k| k == 0 || k > self.measured) {
            return Err(QsoError::invalid(
                "records",
                format!(
                    "a record {k} measurements back does not exist ({} measurements so far)",
                    self.measured
                ),
            ));
        }
        Ok(())
    }

    /// Append a checked operation (records as lookbacks).
    fn push(&mut self, op: QecOp) -> Result<()> {
        if self.stored >= MAX_QEC_OPERATIONS {
            return Err(QsoError::invalid(
                "circuit",
                format!("more than {MAX_QEC_OPERATIONS} operations stored"),
            ));
        }
        if self.operations >= MAX_REPEATED_OPERATIONS {
            return Err(QsoError::invalid(
                "circuit",
                format!("more than {MAX_REPEATED_OPERATIONS} operations expanded"),
            ));
        }
        match &op {
            QecOp::Measure { .. } => self.measured += 1,
            QecOp::Detector { records } => {
                self.detectors += 1;
                self.lookback = self
                    .lookback
                    .max(records.iter().copied().max().unwrap_or(0));
            }
            QecOp::Observable { index, records } => {
                self.observables = self.observables.max(index + 1);
                self.lookback = self
                    .lookback
                    .max(records.iter().copied().max().unwrap_or(0));
            }
            _ => {}
        }
        self.stored += 1;
        self.operations += 1;
        let blocks = &mut self.stack.last_mut().expect("the whole circuit").0;
        match blocks.last_mut() {
            Some(Block::Ops(ops)) => ops.push(op),
            _ => blocks.push(Block::Ops(vec![op])),
        }
        Ok(())
    }

    pub(super) fn begin(&mut self, count: usize) -> Result<()> {
        if count == 0 {
            return Err(QsoError::invalid("repeat", "a block repeats at least once"));
        }
        self.stack.push((
            Vec::new(),
            count,
            (self.measured, self.detectors, self.operations),
        ));
        Ok(())
    }

    pub(super) fn end(&mut self) -> Result<()> {
        if self.stack.len() < 2 {
            return Err(QsoError::invalid("repeat", "no block is open"));
        }
        let (body, count, (m, d, o)) = self.stack.pop().expect("an open block");
        let (dm, dd, dops) = (self.measured - m, self.detectors - d, self.operations - o);
        let total = |start: u64, per: u64| -> Option<u64> {
            per.checked_mul(count as u64)?.checked_add(start)
        };
        let operations = total(o, dops).filter(|&t| t <= MAX_REPEATED_OPERATIONS);
        let Some(operations) = operations else {
            return Err(QsoError::invalid(
                "circuit",
                format!("more than {MAX_REPEATED_OPERATIONS} operations expanded"),
            ));
        };
        self.operations = operations;
        // Measurements and detectors are operations, so they fit too.
        self.measured = m + dm * count;
        self.detectors = d + dd * count;
        self.stack
            .last_mut()
            .expect("the whole circuit")
            .0
            .push(Block::Repeat(Box::new(Repeat {
                count,
                body,
                measurements: dm,
                detectors: dd,
                operations: dops,
            })));
        Ok(())
    }

    pub(super) fn finish(mut self) -> Result<RepeatedCircuit> {
        if self.stack.len() != 1 {
            return Err(QsoError::invalid("repeat", "a block is still open"));
        }
        let (blocks, ..) = self.stack.pop().expect("the whole circuit");
        Ok(RepeatedCircuit {
            n: self.n,
            blocks,
            measurements: self.measured,
            detectors: self.detectors,
            observables: self.observables,
            operations: self.operations,
            stored: self.stored,
            lookback: self.lookback,
        })
    }

    /// Append `op` of a flat circuit, `measured` measurements before it,
    /// its records turned into lookbacks.
    pub(super) fn push_flat(&mut self, op: &QecOp, measured: usize) -> Result<()> {
        let back =
            |records: &[usize]| -> Vec<usize> { records.iter().map(|&r| measured - r).collect() };
        match op {
            QecOp::Detector { records } => self.detector(&back(records)),
            QecOp::Observable { index, records } => self.observable(*index, &back(records)),
            other => self.push(other.clone()),
        }
    }
}

impl Sink for Builder {
    fn gate(&mut self, inst: Instruction) -> Result<()> {
        check_gate(self.n, &inst)?;
        self.push(QecOp::Gate(inst))
    }

    fn measure(&mut self, qubit: usize, flip: f64) -> Result<()> {
        check_qubit(self.n, qubit)?;
        check_probability(flip, 1.0)?;
        self.push(QecOp::Measure { qubit, flip })
    }

    fn reset(&mut self, qubit: usize) -> Result<()> {
        check_qubit(self.n, qubit)?;
        self.push(QecOp::Reset { qubit })
    }

    fn noise(&mut self, noise: Noise, qubits: &[usize]) -> Result<()> {
        check_noise(self.n, &noise, qubits)?;
        // As DetectorCircuit::noise: noise that cannot happen is left out.
        if noise.probability() > 0.0 && !qubits.is_empty() {
            self.push(QecOp::Noise {
                noise,
                qubits: qubits.to_vec(),
            })?;
        }
        Ok(())
    }

    fn detector(&mut self, lookbacks: &[usize]) -> Result<()> {
        self.check_lookbacks(lookbacks)?;
        self.push(QecOp::Detector {
            records: lookbacks.to_vec(),
        })
    }

    fn observable(&mut self, index: usize, lookbacks: &[usize]) -> Result<()> {
        check_observable(index)?;
        self.check_lookbacks(lookbacks)?;
        self.push(QecOp::Observable {
            index,
            records: lookbacks.to_vec(),
        })
    }

    fn measured(&self) -> usize {
        self.measured
    }

    fn begin_repeat(&mut self, count: u64) -> Result<bool> {
        let count = usize::try_from(count)
            .ok()
            .filter(|&c| c as u64 <= MAX_REPEATED_OPERATIONS)
            .ok_or_else(|| {
                QsoError::invalid(
                    "circuit",
                    format!(
                        "REPEAT {count} expands to more than {MAX_REPEATED_OPERATIONS} operations"
                    ),
                )
            })?;
        self.begin(count)?;
        Ok(true)
    }

    fn end_repeat(&mut self) -> Result<()> {
        self.end()
    }
}

/// Calls `f(op, records)` for every operation of `blocks` in order, the
/// records of detectors and observables as indices into the whole record.
fn walk_blocks(
    blocks: &[Block],
    measured: &mut usize,
    records: &mut Vec<usize>,
    f: &mut dyn FnMut(&QecOp, &[usize]) -> Result<()>,
) -> Result<()> {
    for block in blocks {
        match block {
            Block::Ops(ops) => {
                for op in ops {
                    records.clear();
                    match op {
                        QecOp::Detector { records: back }
                        | QecOp::Observable { records: back, .. } => {
                            records.extend(back.iter().map(|&k| *measured - k));
                        }
                        QecOp::Measure { .. } => *measured += 1,
                        _ => {}
                    }
                    f(op, records)?;
                }
            }
            Block::Repeat(r) => {
                for _ in 0..r.count {
                    walk_blocks(&r.body, measured, records, f)?;
                }
            }
        }
    }
    Ok(())
}

impl Walk for RepeatedCircuit {
    fn n_qubits(&self) -> usize {
        self.n
    }

    fn n_detectors(&self) -> usize {
        self.detectors
    }

    fn n_observables(&self) -> usize {
        self.observables
    }

    fn ring(&self) -> usize {
        self.lookback.max(1)
    }

    fn walk(&self, f: &mut dyn FnMut(&QecOp, &[usize]) -> Result<()>) -> Result<()> {
        walk_blocks(&self.blocks, &mut 0, &mut Vec::new(), f)
    }
}

fn write_blocks(out: &mut String, blocks: &[Block], depth: usize) -> Result<()> {
    let indent = "    ".repeat(depth);
    for block in blocks {
        match block {
            Block::Ops(ops) => {
                for op in ops {
                    stim::write_op(out, op, &indent, &|k| k)?;
                }
            }
            Block::Repeat(r) => {
                let _ = writeln!(out, "{indent}REPEAT {} {{", r.count);
                write_blocks(out, &r.body, depth + 1)?;
                let _ = writeln!(out, "{indent}}}");
            }
        }
    }
    Ok(())
}

impl RepeatedCircuit {
    /// A circuit in Stim's text format with its `REPEAT` blocks kept (the
    /// instructions of [`DetectorCircuit::from_stim`]). Up to
    /// [`MAX_QEC_OPERATIONS`] operations are stored and
    /// [`MAX_REPEATED_OPERATIONS`] expanded.
    pub fn from_stim(text: &str) -> Result<RepeatedCircuit> {
        let parsed = stim::parse(text)?;
        stim::emit(&parsed, Builder::new(parsed.n)?)?.finish()
    }

    /// A flat circuit as a repeated one without blocks.
    pub fn from_circuit(circuit: &DetectorCircuit) -> Result<RepeatedCircuit> {
        let mut b = Builder::new(circuit.n_qubits())?;
        let mut measured = 0;
        for op in circuit.operations() {
            b.push_flat(op, measured)?;
            if let QecOp::Measure { .. } = op {
                measured += 1;
            }
        }
        b.finish()
    }

    /// The operations `body` of a flat circuit repeated `count` times
    /// between those before and after it; their records turned into
    /// lookbacks, so that each iteration reads the records the body read
    /// relative to it.
    pub(super) fn from_parts(
        circuit: &DetectorCircuit,
        body: std::ops::Range<usize>,
        count: usize,
    ) -> Result<RepeatedCircuit> {
        let ops = circuit.operations();
        if body.start >= body.end || body.end > ops.len() {
            return Err(QsoError::invalid(
                "body",
                format!("operations {body:?} of {}", ops.len()),
            ));
        }
        let mut b = Builder::new(circuit.n_qubits())?;
        let mut measured = 0;
        for (i, op) in ops.iter().enumerate() {
            if i == body.start {
                b.begin(count)?;
            }
            b.push_flat(op, measured)?;
            if let QecOp::Measure { .. } = op {
                measured += 1;
            }
            if i + 1 == body.end {
                b.end()?;
            }
        }
        b.finish()
    }

    /// The circuit in Stim's text format, its blocks as `REPEAT` (the
    /// operations written as [`DetectorCircuit::to_stim`] writes them).
    pub fn to_stim(&self) -> Result<String> {
        let mut out = String::new();
        write_blocks(&mut out, &self.blocks, 0)?;
        Ok(out)
    }

    /// Every block expanded, if it fits a [`DetectorCircuit`].
    pub fn flatten(&self) -> Result<DetectorCircuit> {
        if self.operations > MAX_QEC_OPERATIONS as u64 {
            return Err(QsoError::invalid(
                "circuit",
                format!(
                    "{} operations expanded do not fit a detector circuit ({MAX_QEC_OPERATIONS})",
                    self.operations
                ),
            ));
        }
        let mut c = DetectorCircuit::new(self.n)?;
        self.walk(&mut |op, records| {
            match op {
                QecOp::Gate(inst) => {
                    c.gate(inst.clone())?;
                }
                QecOp::Measure { qubit, flip } => {
                    c.measure(*qubit, *flip)?;
                }
                QecOp::Reset { qubit } => {
                    c.reset(*qubit)?;
                }
                QecOp::Noise { noise, qubits } => {
                    c.noise(*noise, qubits)?;
                }
                QecOp::Detector { .. } => {
                    c.detector(records)?;
                }
                QecOp::Observable { index, .. } => {
                    c.observable(*index, records)?;
                }
            }
            Ok(())
        })?;
        Ok(c)
    }

    pub fn n_qubits(&self) -> usize {
        self.n
    }

    /// Measurements, expanded.
    pub fn n_measurements(&self) -> usize {
        self.measurements
    }

    /// Detectors, expanded.
    pub fn n_detectors(&self) -> usize {
        self.detectors
    }

    pub fn n_observables(&self) -> usize {
        self.observables
    }

    /// Operations, expanded.
    pub fn n_operations(&self) -> u64 {
        self.operations
    }

    /// Operations stored (each block once).
    pub fn n_stored_operations(&self) -> usize {
        self.stored
    }

    /// The layer of every detector, as [`DetectorCircuit::detector_layers`]
    /// gives it for the expanded circuit.
    pub fn detector_layers(&self) -> Vec<usize> {
        let mut layers = Vec::with_capacity(self.detectors);
        let mut layer = 0;
        let mut previous_was_detector = false;
        let _ = self.walk(&mut |op, _| {
            match op {
                QecOp::Detector { .. } => {
                    if !previous_was_detector && !layers.is_empty() {
                        layer += 1;
                    }
                    layers.push(layer);
                    previous_was_detector = true;
                }
                QecOp::Observable { .. } => {}
                _ => previous_was_detector = false,
            }
            Ok(())
        });
        layers
    }

    /// The detector error model of the expanded circuit, as
    /// [`super::detector_error_model`] gives it (up to the rounding of
    /// merged probabilities), by one pass backwards that folds the blocks
    /// (see the module docs).
    pub fn detector_error_model(&self, decompose: bool) -> Result<DetectorErrorModel> {
        super::dem::repeated_error_model(self, decompose)
    }

    /// The correlated model of the expanded circuit, as
    /// [`super::correlated_error_model`] gives it, folded the same way.
    pub fn correlated_error_model(&self) -> Result<CorrelatedErrorModel> {
        super::dem::repeated_correlated_model(self)
    }

    /// Detection events and observable flips of `shots` shots, as
    /// [`super::sample_detectors`] samples the expanded circuit (the same
    /// shots for the same seed while a batch of shots fits both); only the
    /// measurement records a detector can still reach are kept.
    pub fn sample(&self, shots: usize, seed: u64) -> Result<DetectorSamples> {
        if (shots as u64).saturating_mul(self.detectors as u64 + 64) > MAX_SAMPLED_EVENTS {
            return Err(QsoError::invalid(
                "shots",
                format!("shots × (detectors + 64) must be at most {MAX_SAMPLED_EVENTS}"),
            ));
        }
        frame::check_deterministic(self)?;
        frame::sample_with(self, shots, &mut Rng::seed_from_u64(seed))
    }
}

#[cfg(test)]
mod tests;
