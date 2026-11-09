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

/// Operations of a detector circuit beyond [`QecOp`], one variant per Stim
/// instruction; more will be added (Pauli channels, correlated errors, MPP,
/// feedback). Built by the methods of [`DetectorCircuit`] (such as
/// [`DetectorCircuit::heralded_erase`]) and listed in order with every
/// other operation by [`DetectorCircuit::steps`].
///
/// The heralded channels add one measurement record per target (its
/// herald), as Stim does; a later variant may add none (a Pauli channel)
/// or one per product (MPP), and `n_records` says how many. Erasure stays
/// in the qubit space: an erased qubit is replaced by a maximally mixed
/// one, nothing leaks.
///
/// Adding a variant: its builder and check (`check_extended`);
/// `n_records` and `noiseless`; `frame::run_extended`;
/// `dem::Folder::extended`; the Stim tables (`stim::extended_instruction`,
/// `stim::write_extended`); the fuzz builder; tests against Stim term by
/// term and against its sampler. Inside the crate every `match` on
/// `ExtendedOp` or [`CircuitStep`] is exhaustive, so the compiler lists
/// every place that must learn a new variant.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ExtendedOp {
    /// Stim's `HERALDED_ERASE(p)`: for each target in order, with
    /// probability `p` the qubit is replaced by a maximally mixed one (I,
    /// X, Y or Z with p/4 each) and its herald, a new measurement record,
    /// reads 1; otherwise the herald reads 0. Heralds are never flipped or
    /// inverted.
    #[non_exhaustive]
    HeraldedErase {
        probability: f64,
        qubits: Vec<usize>,
    },
    /// Stim's `HERALDED_PAULI_CHANNEL_1(pI, pX, pY, pZ)`: for each target,
    /// disjointly, the herald reads 1 together with I, X, Y or Z with these
    /// probabilities (Stim's argument order); with 1 − Σ neither herald nor
    /// error.
    #[non_exhaustive]
    HeraldedPauliChannel1 {
        probabilities: [f64; 4],
        qubits: Vec<usize>,
    },
}

impl ExtendedOp {
    /// The qubits it acts on, in order (repeats allowed).
    pub(crate) fn qubits(&self) -> &[usize] {
        match self {
            ExtendedOp::HeraldedErase { qubits, .. }
            | ExtendedOp::HeraldedPauliChannel1 { qubits, .. } => qubits,
        }
    }

    /// Measurement records it adds (one herald per target).
    pub(crate) fn n_records(&self) -> usize {
        match self {
            ExtendedOp::HeraldedErase { qubits, .. }
            | ExtendedOp::HeraldedPauliChannel1 { qubits, .. } => qubits.len(),
        }
    }

    /// The same operation with probability 0 (its records kept).
    pub(crate) fn noiseless(&self) -> ExtendedOp {
        match self {
            ExtendedOp::HeraldedErase { qubits, .. } => ExtendedOp::HeraldedErase {
                probability: 0.0,
                qubits: qubits.clone(),
            },
            ExtendedOp::HeraldedPauliChannel1 { qubits, .. } => ExtendedOp::HeraldedPauliChannel1 {
                probabilities: [0.0; 4],
                qubits: qubits.clone(),
            },
        }
    }
}

/// Stim's tolerance on the sum of a heralded Pauli channel's probabilities.
const PROBABILITY_SUM_TOLERANCE: f64 = 1e-7;

/// An extended operation with valid probabilities on qubits of a register
/// of `n`.
pub(super) fn check_extended(n: usize, op: &ExtendedOp) -> Result<()> {
    match op {
        ExtendedOp::HeraldedErase { probability, .. } => check_probability(*probability, 1.0)?,
        ExtendedOp::HeraldedPauliChannel1 { probabilities, .. } => {
            for &p in probabilities {
                check_probability(p, 1.0)?;
            }
            let [pi, px, py, pz] = *probabilities;
            let s = ((pi + px) + py) + pz;
            if s > 1.0 + PROBABILITY_SUM_TOLERANCE {
                return Err(QsoError::invalid(
                    "probability",
                    format!(
                        "the probabilities of HERALDED_PAULI_CHANNEL_1 add up to {s}, more than 1"
                    ),
                ));
            }
        }
    }
    for &q in op.qubits() {
        check_qubit(n, q)?;
    }
    Ok(())
}

/// One operation of a detector circuit, in circuit order
/// ([`DetectorCircuit::steps`]).
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum CircuitStep<'a> {
    /// An operation of [`DetectorCircuit::operations`].
    Qec(&'a QecOp),
    /// An extended operation (heralded noise).
    Extended(&'a ExtendedOp),
}

/// The operations of a circuit in order ([`DetectorCircuit::steps`]).
#[derive(Clone, Debug)]
pub struct Steps<'a> {
    ops: &'a [QecOp],
    extended: &'a [(usize, ExtendedOp)],
    /// Next operation and extended operation from the front, and the ends
    /// from the back.
    i: usize,
    j: usize,
    i_end: usize,
    j_end: usize,
}

impl<'a> Iterator for Steps<'a> {
    type Item = CircuitStep<'a>;

    fn next(&mut self) -> Option<CircuitStep<'a>> {
        let ext_first =
            self.j < self.j_end && (self.i == self.i_end || self.extended[self.j].0 <= self.i);
        if ext_first {
            self.j += 1;
            Some(CircuitStep::Extended(&self.extended[self.j - 1].1))
        } else if self.i < self.i_end {
            self.i += 1;
            Some(CircuitStep::Qec(&self.ops[self.i - 1]))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = (self.i_end - self.i) + (self.j_end - self.j);
        (n, Some(n))
    }
}

impl DoubleEndedIterator for Steps<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let ext_last = self.j < self.j_end
            && (self.i == self.i_end || self.extended[self.j_end - 1].0 >= self.i_end);
        if ext_last {
            self.j_end -= 1;
            Some(CircuitStep::Extended(&self.extended[self.j_end].1))
        } else if self.i < self.i_end {
            self.i_end -= 1;
            Some(CircuitStep::Qec(&self.ops[self.i_end]))
        } else {
            None
        }
    }
}

impl ExactSizeIterator for Steps<'_> {}

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
    /// Extended operations, each before `ops[position]` (at the end for
    /// `ops.len()`); equal positions in the order they were added.
    extended: Vec<(usize, ExtendedOp)>,
    measurements: usize,
    detectors: usize,
    observables: usize,
}

pub(super) fn check_probability(p: f64, max: f64) -> Result<()> {
    if !(0.0..=max).contains(&p) {
        return Err(QsoError::invalid(
            "probability",
            format!("{p} is not a probability in [0, {max}]"),
        ));
    }
    Ok(())
}

/// `q` is a qubit of a register of `n`.
pub(super) fn check_qubit(n: usize, q: usize) -> Result<()> {
    if q >= n {
        return Err(QsoError::QubitOutOfRange {
            qubit: q,
            n_qubits: n,
        });
    }
    Ok(())
}

/// A Clifford gate on a register of `n` with numbers for parameters.
pub(super) fn check_gate(n: usize, inst: &Instruction) -> Result<()> {
    validate_instruction(inst, &[], n)?;
    if !is_clifford_instruction(inst, &[]) {
        return Err(QsoError::invalid(
            "gate",
            format!("{} is not a Clifford gate", inst.kind.name()),
        ));
    }
    Ok(())
}

/// Noise of a valid probability on qubits of a register of `n` (pairs of
/// different qubits for two-qubit noise).
pub(super) fn check_noise(n: usize, noise: &Noise, qubits: &[usize]) -> Result<()> {
    check_probability(noise.probability(), noise.max_probability())?;
    for &q in qubits {
        check_qubit(n, q)?;
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
    Ok(())
}

/// An observable index below [`MAX_OBSERVABLES`].
pub(super) fn check_observable(index: usize) -> Result<()> {
    if index >= MAX_OBSERVABLES {
        return Err(QsoError::invalid(
            "observable",
            format!("observable indices are below {MAX_OBSERVABLES}"),
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

    /// The [`QecOp`]s in order; a circuit with extended operations
    /// interleaves them (see [`DetectorCircuit::steps`]).
    pub fn operations(&self) -> &[QecOp] {
        &self.ops
    }

    /// Every operation in order, the extended ones included
    /// ([`DetectorCircuit::operations`] lists the [`QecOp`]s only).
    pub fn steps(&self) -> Steps<'_> {
        Steps {
            ops: &self.ops,
            extended: &self.extended,
            i: 0,
            j: 0,
            i_end: self.ops.len(),
            j_end: self.extended.len(),
        }
    }

    /// `operations().len()` plus the extended operations.
    pub fn n_steps(&self) -> usize {
        self.ops.len() + self.extended.len()
    }

    /// Whether the circuit has extended operations ([`ExtendedOp`]: today
    /// heralded noise), which some functions refuse.
    pub fn has_extended(&self) -> bool {
        !self.extended.is_empty()
    }

    /// Measurement records, heralds included (as Stim's
    /// `num_measurements`).
    pub fn n_measurements(&self) -> usize {
        self.measurements
    }

    pub fn n_detectors(&self) -> usize {
        self.detectors
    }

    pub fn n_observables(&self) -> usize {
        self.observables
    }

    /// The layer (round) of every detector: detectors declared one after
    /// another, with no other operation between them (observables
    /// aside), form one layer, as each round of a memory experiment
    /// declares its detectors after its measurements. Layers are
    /// numbered from 0 in the order of the circuit, so they never
    /// decrease with the detector index.
    pub fn detector_layers(&self) -> Vec<usize> {
        let mut layers = Vec::with_capacity(self.detectors);
        let mut layer = 0;
        let mut previous_was_detector = false;
        for step in self.steps() {
            match step {
                CircuitStep::Qec(QecOp::Detector { .. }) => {
                    if !previous_was_detector && !layers.is_empty() {
                        layer += 1;
                    }
                    layers.push(layer);
                    previous_was_detector = true;
                }
                CircuitStep::Qec(QecOp::Observable { .. }) => {}
                CircuitStep::Qec(_) | CircuitStep::Extended(_) => previous_was_detector = false,
            }
        }
        layers
    }

    fn check_room(&self) -> Result<()> {
        if self.n_steps() >= MAX_QEC_OPERATIONS {
            return Err(QsoError::invalid(
                "circuit",
                format!("more than {MAX_QEC_OPERATIONS} operations"),
            ));
        }
        Ok(())
    }

    fn push(&mut self, op: QecOp) -> Result<()> {
        self.check_room()?;
        self.ops.push(op);
        Ok(())
    }

    /// Append a checked extended operation; returns its records.
    pub(super) fn push_extended(&mut self, op: ExtendedOp) -> Result<Vec<usize>> {
        if op.n_records() == 0 {
            return Ok(Vec::new());
        }
        self.check_room()?;
        let first = self.measurements;
        self.measurements += op.n_records();
        self.extended.push((self.ops.len(), op));
        Ok((first..self.measurements).collect())
    }

    /// Stim's `HERALDED_ERASE(probability)` on `qubits` (repeats allowed):
    /// each target, in order, is erased with this probability (replaced
    /// by a maximally mixed qubit: I, X, Y or Z with a quarter of it each)
    /// and its herald, a new measurement record, reads whether it was.
    /// Returns the herald's record of each target. It is kept at
    /// probability 0 (its records exist, as Stim's `MPAD 0`); without
    /// qubits nothing is added.
    pub fn heralded_erase(&mut self, probability: f64, qubits: &[usize]) -> Result<Vec<usize>> {
        let op = ExtendedOp::HeraldedErase {
            probability,
            qubits: qubits.to_vec(),
        };
        check_extended(self.n, &op)?;
        self.push_extended(op)
    }

    /// Stim's `HERALDED_PAULI_CHANNEL_1` on `qubits`: for each target the
    /// herald fires together with I, X, Y or Z with `probabilities` =
    /// (pI, pX, pY, pZ) (disjointly; their sum at most 1, with Stim's
    /// tolerance of 10⁻⁷). Returns the herald's record of each target.
    pub fn heralded_pauli_channel_1(
        &mut self,
        probabilities: [f64; 4],
        qubits: &[usize],
    ) -> Result<Vec<usize>> {
        let op = ExtendedOp::HeraldedPauliChannel1 {
            probabilities,
            qubits: qubits.to_vec(),
        };
        check_extended(self.n, &op)?;
        self.push_extended(op)
    }

    fn check(&self, q: usize) -> Result<()> {
        check_qubit(self.n, q)
    }

    /// Append a Clifford gate (parameters must be numbers).
    pub fn gate(&mut self, inst: Instruction) -> Result<&mut Self> {
        check_gate(self.n, &inst)?;
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
        if self.n_steps() + bound.instructions().len() > MAX_QEC_OPERATIONS {
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
        check_noise(self.n, &noise, qubits)?;
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
        check_observable(index)?;
        self.check_records(records)?;
        self.push(QecOp::Observable {
            index,
            records: records.to_vec(),
        })?;
        self.observables = self.observables.max(index + 1);
        Ok(self)
    }

    /// The circuit without noise (measurement flips set to zero; extended
    /// operations kept with probability 0, so that their records stay).
    pub fn without_noise(&self) -> DetectorCircuit {
        // Noise operations before each position, to move the extended
        // operations with the rest.
        let mut noise_before = Vec::with_capacity(self.ops.len() + 1);
        let mut count = 0;
        for op in &self.ops {
            noise_before.push(count);
            count += matches!(op, QecOp::Noise { .. }) as usize;
        }
        noise_before.push(count);
        let extended = self
            .extended
            .iter()
            .map(|(pos, op)| (pos - noise_before[*pos], op.noiseless()))
            .collect();
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
            n: self.n,
            ops,
            extended,
            measurements: self.measurements,
            detectors: self.detectors,
            observables: self.observables,
        }
    }
}
