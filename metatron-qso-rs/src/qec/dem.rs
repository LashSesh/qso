//! Detector error models: which detectors and observables each error
//! flips, and how likely it is.
//!
//! Every noise channel is written as independent Pauli errors (exactly:
//! a depolarizing channel equals independent X, Y and Z errors, or the 15
//! two-qubit Paulis, with adjusted probabilities). What each error flips
//! comes from one pass backwards through the noiseless circuit, which
//! carries for every qubit the detectors and observables an X or a Z there
//! would flip (the transpose of propagating Pauli frames forwards), in
//! time linear in the circuit; errors with the same effect are merged.

use std::collections::{BTreeMap, HashMap};

use super::circuit::{DetectorCircuit, Noise, QecOp};
#[cfg(test)]
use super::frame::Frames;
use super::repeat::{Block, Repeat, RepeatedCircuit};
use crate::error::{QsoError, Result};
use crate::quantum::Instruction;
use crate::stabilizer::{CliffordTarget, apply_clifford};

/// Most elementary error components a model propagates.
pub const MAX_COMPONENTS: usize = 1 << 22;
/// Most detector entries over all propagated components (1 GiB).
pub const MAX_SIGNATURE_ENTRIES: usize = 1 << 27;
/// Most entries of effects the pass backwards through a circuit touches
/// (each XOR of two effects counts their lengths; about fifteen seconds at
/// the limit).
pub const MAX_MODEL_WORK: u64 = 1 << 34;

/// One independent error mechanism of a detector error model.
#[derive(Clone, Debug, PartialEq)]
pub struct ErrorMechanism {
    pub probability: f64,
    /// Detectors it flips, ascending.
    pub detectors: Vec<usize>,
    /// Observables it flips (bit `i`: observable `i`).
    pub observables: u64,
}

/// Most detectors of a model: a circuit has at most one per operation.
pub const MAX_MODEL_DETECTORS: usize = super::circuit::MAX_QEC_OPERATIONS;

/// Independent error mechanisms of a circuit.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectorErrorModel {
    pub n_detectors: usize,
    pub n_observables: usize,
    pub errors: Vec<ErrorMechanism>,
}

/// An elementary component: X or Z on a qubit at a noise operation, or a
/// measurement flip. Ordered by operation first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Component {
    op: usize,
    qubit: usize,
    kind: Kind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Kind {
    X,
    Z,
    Flip,
}

/// Effect of an error: flipped detectors (sorted) and observables.
pub(super) type Signature = (Vec<usize>, u64);

/// What receives each channel: its place, its basis errors' effects and
/// the probabilities of their combinations.
pub(super) type ChannelSink<'a> = dyn FnMut(&Channel, &[Signature], &[f64]) -> Result<()> + 'a;

fn xor(a: &Signature, b: &Signature) -> Signature {
    let mut d = Vec::with_capacity(a.0.len() + b.0.len());
    let (mut i, mut j) = (0, 0);
    while i < a.0.len() || j < b.0.len() {
        match (a.0.get(i), b.0.get(j)) {
            (Some(&x), Some(&y)) if x == y => {
                i += 1;
                j += 1;
            }
            (Some(&x), Some(&y)) if x < y => {
                d.push(x);
                i += 1;
            }
            (Some(&x), None) => {
                d.push(x);
                i += 1;
            }
            (_, Some(&y)) => {
                d.push(y);
                j += 1;
            }
            (None, None) => break,
        }
    }
    (d, a.1 ^ b.1)
}

/// Probability of each of k independent errors whose combination is a
/// depolarizing channel of strength p on `qubits` qubits (k = 4^q − 1).
fn independent(p: f64, qubits: u32) -> f64 {
    let k = 4f64.powi(qubits as i32) - 1.0;
    let d = 4f64.powi(qubits as i32);
    // 1 − 2q = (1 − d p / k)^(1 / (d / 2)) (Stim's conversion).
    0.5 - 0.5 * (1.0 - d * p / k).powf(2.0 / d)
}

/// The detector error model of `circuit`; its detectors and observables
/// must be deterministic without noise, as for
/// [`sample_detectors`](super::sample_detectors). With `decompose`, every
/// error is split into graph-like parts (each flipping at most two
/// detectors) as Stim decomposes errors (see [`correlated_error_model`]),
/// so that the model is a matching graph; every part keeps the error's
/// probability, which overestimates correlated errors slightly, as
/// decoders do.
pub fn detector_error_model(
    circuit: &DetectorCircuit,
    decompose: bool,
) -> Result<DetectorErrorModel> {
    let mut merged: BTreeMap<Signature, f64> = BTreeMap::new();
    each_channel(circuit, |_, basis, probabilities| {
        if !decompose {
            for (k, &p) in probabilities.iter().enumerate().skip(1) {
                let total = combination(basis, k);
                if p > 0.0 && !is_empty(&total) {
                    merge(&mut merged, total, p);
                }
            }
            return Ok(());
        }
        for (parts, p) in decomposed(basis, probabilities)? {
            for part in parts {
                merge(&mut merged, part, p);
            }
        }
        Ok(())
    })?;
    let errors = merged
        .into_iter()
        .map(|((detectors, observables), probability)| ErrorMechanism {
            probability,
            detectors,
            observables,
        })
        .collect();
    Ok(DetectorErrorModel {
        n_detectors: circuit.n_detectors(),
        n_observables: circuit.n_observables(),
        errors,
    })
}

/// Combine an independent error of probability `p` into the error with the
/// same effect.
fn merge<K: Ord>(merged: &mut BTreeMap<K, f64>, key: K, p: f64) {
    let e = merged.entry(key).or_insert(0.0);
    *e = *e * (1.0 - p) + p * (1.0 - *e);
}

fn is_empty(s: &Signature) -> bool {
    s.0.is_empty() && s.1 == 0
}

/// The effect of the basis errors in the bits of `k`.
fn combination(basis: &[Signature], k: usize) -> Signature {
    basis
        .iter()
        .enumerate()
        .filter(|(i, _)| k >> i & 1 == 1)
        .fold((Vec::new(), 0u64), |acc, (_, s)| xor(&acc, s))
}

/// The graph-like parts of every error of a channel with these basis
/// errors (in Stim's order) and their probabilities, by combination of
/// the basis errors; errors with no effect or probability 0 are left out.
///
/// Within the channel, as in Stim's error analysis: combinations that flip
/// one detector are parts; so are combinations that flip two detectors,
/// one of which no single-detector combination flips ("irreducible
/// pairs"). Every other combination is written as at most two irreducible
/// pairs whose remainder single-detector combinations flip, and finished
/// with single-detector combinations. A combination that cannot be written
/// so (Stim decomposes it later against the whole model's graph-like
/// errors), or whose parts do not add up to it, is split into its basis
/// errors.
fn decomposed(basis: &[Signature], probabilities: &[f64]) -> Result<Vec<(Vec<Signature>, f64)>> {
    let s = basis.len();
    let combos: Vec<Signature> = (0..1usize << s).map(|k| combination(basis, k)).collect();
    // Detectors of the channel as bits.
    let mut involved: Vec<usize> = basis.iter().flat_map(|b| b.0.iter().copied()).collect();
    involved.sort_unstable();
    involved.dedup();
    let wide = involved.len() > 64;
    let mask = |sig: &Signature| -> u64 {
        sig.0.iter().fold(0u64, |m, d| {
            m | 1 << involved.binary_search(d).expect("an involved detector")
        })
    };
    let masks: Vec<u64> = if wide {
        Vec::new()
    } else {
        combos.iter().map(mask).collect()
    };
    let count = |k: usize| combos[k].0.len();
    let mut singles = 0u64;
    let mut solved = vec![false; combos.len()];
    let mut pairs = Vec::new();
    if !wide {
        for k in 1..combos.len() {
            if count(k) == 1 {
                singles |= masks[k];
                solved[k] = true;
            }
        }
        for k in 1..combos.len() {
            if count(k) == 2 && masks[k] & !singles != 0 {
                pairs.push(k);
                solved[k] = true;
            }
        }
    }
    // The parts of an unsolved combination, if Stim's rules find them.
    let split = |k: usize| -> Option<Vec<Signature>> {
        let goal = masks[k];
        let mut parts = Vec::new();
        let mut remnants = goal;
        if goal & !singles != 0 {
            let one = pairs
                .iter()
                .find(|&&j| goal & masks[j] == masks[j] && goal & !(singles | masks[j]) == 0);
            if let Some(&j) = one {
                parts.push(combos[j].clone());
                remnants = goal & !masks[j];
            } else {
                let mut two = None;
                'search: for (i1, &j1) in pairs.iter().enumerate() {
                    for &j2 in &pairs[i1 + 1..] {
                        let (m1, m2) = (masks[j1], masks[j2]);
                        if m1 & m2 == 0 && goal & !(singles | m1 | m2) == 0 {
                            two = Some((j1, j2));
                            break 'search;
                        }
                    }
                }
                let (j1, j2) = two?;
                parts.push(combos[j1].clone());
                parts.push(combos[j2].clone());
                remnants = goal & !(masks[j1] | masks[j2]);
            }
        }
        for (j, m) in masks.iter().enumerate() {
            if remnants == 0 {
                break;
            }
            if count(j) == 1 && m & !remnants == 0 {
                remnants &= !m;
                parts.push(combos[j].clone());
            }
        }
        Some(parts)
    };
    // A channel of one error (a Y error) is no family of combinations: as
    // in Stim, it stays whole when it is graph-like.
    let single = probabilities.iter().filter(|&&p| p > 0.0).count() == 1;
    let mut out = Vec::new();
    for (k, &p) in probabilities.iter().enumerate().skip(1) {
        if p == 0.0 || is_empty(&combos[k]) {
            continue;
        }
        // Stim's rules can leave parts whose effects do not add up to the
        // error's (an observable lost, a detector added); those errors are
        // split into their basis errors instead.
        let adds_up = |parts: &Vec<Signature>| {
            parts.iter().fold((Vec::new(), 0u64), |acc, s| xor(&acc, s)) == combos[k]
        };
        let parts = if count(k) == 0 || (!wide && solved[k]) || (single && count(k) <= 2) {
            vec![combos[k].clone()]
        } else if let Some(parts) = (!wide).then(|| split(k)).flatten().filter(adds_up) {
            parts
        } else if count(k) <= 2 {
            vec![combos[k].clone()]
        } else {
            // Split into the basis errors.
            let parts: Vec<Signature> = (0..s)
                .filter(|i| k >> i & 1 == 1)
                .map(|i| basis[i].clone())
                .filter(|b| !is_empty(b))
                .collect();
            if let Some(b) = parts.iter().find(|b| b.0.len() > 2) {
                return Err(QsoError::invalid(
                    "circuit",
                    format!(
                        "an error flips {} detectors even after splitting into single-qubit X and Z parts; the model is not a matching graph (decode it with BP+OSD)",
                        b.0.len()
                    ),
                ));
            }
            parts
        };
        out.push((parts, p));
    }
    Ok(out)
}

/// An error of a decomposed model and the graph-like parts it was split
/// into, which happen together.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrelatedError {
    pub probability: f64,
    /// Detectors (ascending, at most two) and observables of each part.
    pub parts: Vec<(Vec<usize>, u64)>,
}

/// A decomposed detector error model that remembers which parts belong to
/// the same error, for correlated decoding.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrelatedErrorModel {
    pub n_detectors: usize,
    pub n_observables: usize,
    pub errors: Vec<CorrelatedError>,
}

impl CorrelatedErrorModel {
    /// The decomposed model: every part an independent error (equal to
    /// `detector_error_model(circuit, true)` for the model of a circuit, up
    /// to rounding).
    pub fn to_dem(&self) -> DetectorErrorModel {
        let mut merged: BTreeMap<Signature, f64> = BTreeMap::new();
        for e in &self.errors {
            for part in &e.parts {
                if !is_empty(part) {
                    merge(&mut merged, part.clone(), e.probability);
                }
            }
        }
        DetectorErrorModel {
            n_detectors: self.n_detectors,
            n_observables: self.n_observables,
            errors: merged
                .into_iter()
                .map(|((detectors, observables), probability)| ErrorMechanism {
                    probability,
                    detectors,
                    observables,
                })
                .collect(),
        }
    }
}

/// The decomposed detector error model of `circuit` as in
/// [`detector_error_model`] with `decompose`, keeping for every error the
/// graph-like parts it was split into. The decomposition follows Stim's
/// within each noise channel (a depolarizing channel's X and Z errors on
/// its qubits and their combinations): combinations that flip one
/// detector, or two of which one no such combination flips, are parts; the
/// other combinations are written with at most two such pairs and
/// single-detector combinations (a Y error is such a combination). A
/// combination it cannot write so (Stim then decomposes against the
/// graph-like errors of the whole model), or whose parts would not add up
/// to it, is split into its single-qubit X and Z parts. Errors with the
/// same parts are merged.
pub fn correlated_error_model(circuit: &DetectorCircuit) -> Result<CorrelatedErrorModel> {
    let mut merged: BTreeMap<Vec<Signature>, f64> = BTreeMap::new();
    each_channel(circuit, |_, basis, probabilities| {
        for (mut parts, p) in decomposed(basis, probabilities)? {
            parts.sort();
            merge(&mut merged, parts, p);
        }
        Ok(())
    })?;
    Ok(CorrelatedErrorModel {
        n_detectors: circuit.n_detectors(),
        n_observables: circuit.n_observables(),
        errors: merged
            .into_iter()
            .map(|(parts, probability)| CorrelatedError { probability, parts })
            .collect(),
    })
}

/// A fault of a circuit: one independent error of a noise channel, or a
/// flipped measurement.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct CircuitFault {
    /// The noise or measurement operation (an index into
    /// [`DetectorCircuit::operations`]).
    pub operation: usize,
    /// The Pauli error on each qubit it touches (`'X'`, `'Y'`, `'Z'`), or
    /// `'M'` for a flipped measurement of that qubit.
    pub paulis: Vec<(usize, char)>,
    /// Its probability as an independent error.
    pub probability: f64,
}

/// Where errors of the circuit's (undecomposed) detector error model come
/// from: for every error given, the faults of the circuit whose effect is
/// exactly its detectors and observables, in circuit order (several when
/// faults are equivalent; none for an error the circuit cannot produce).
/// With the witness of a circuit distance this gives the lightest logical
/// error as circuit locations.
pub fn explain_errors(
    circuit: &DetectorCircuit,
    errors: &[ErrorMechanism],
) -> Result<Vec<Vec<CircuitFault>>> {
    let mut wanted: BTreeMap<Signature, Vec<usize>> = BTreeMap::new();
    for (i, e) in errors.iter().enumerate() {
        wanted
            .entry((e.detectors.clone(), e.observables))
            .or_default()
            .push(i);
    }
    let mut out = vec![Vec::new(); errors.len()];
    each_channel(circuit, |channel, basis, probabilities| {
        for (k, &p) in probabilities.iter().enumerate().skip(1) {
            if p == 0.0 {
                continue;
            }
            let Some(targets) = wanted.get(&combination(basis, k)) else {
                continue;
            };
            // The Pauli on each qubit from its Z and X parts.
            let mut paulis: Vec<(usize, char)> = Vec::new();
            for (b, &(qubit, kind)) in channel.parts.iter().enumerate() {
                if k >> b & 1 == 0 {
                    continue;
                }
                let letter = match kind {
                    Kind::X => 'X',
                    Kind::Z => 'Z',
                    Kind::Flip => 'M',
                };
                match paulis.iter_mut().find(|(q, _)| *q == qubit) {
                    Some((_, c)) if (*c, letter) == ('Z', 'X') || (*c, letter) == ('X', 'Z') => {
                        *c = 'Y'
                    }
                    _ => paulis.push((qubit, letter)),
                }
            }
            let fault = CircuitFault {
                operation: channel.op,
                paulis,
                probability: p,
            };
            for &t in targets {
                out[t].push(fault.clone());
            }
        }
        Ok(())
    })?;
    Ok(out)
}

/// Where a noise channel acts: its operation and, for each basis error,
/// the qubit and kind.
pub(super) struct Channel {
    pub(super) op: usize,
    pub(super) parts: Vec<(usize, Kind)>,
}

/// Calls `f(channel, basis, probabilities)` for every noise channel of the
/// circuit: the effects of its basis errors in Stim's order (for a
/// depolarizing channel Z then X on each qubit, as for a Y error; a
/// measurement flip or a single-qubit X or Z error is one basis error) and
/// the probability of each combination of them (index: bit `i` for basis
/// error `i`), as independent errors.
fn each_channel(
    circuit: &DetectorCircuit,
    mut f: impl FnMut(&Channel, &[Signature], &[f64]) -> Result<()>,
) -> Result<()> {
    super::frame::check_deterministic(circuit)?;
    // Elementary components.
    let mut components: Vec<Component> = Vec::new();
    for (i, op) in circuit.operations().iter().enumerate() {
        match op {
            QecOp::Noise { qubits, .. } => {
                for &qubit in qubits {
                    for kind in [Kind::X, Kind::Z] {
                        components.push(Component { op: i, qubit, kind });
                    }
                }
            }
            QecOp::Measure { flip, .. } if *flip > 0.0 => {
                components.push(Component {
                    op: i,
                    qubit: 0,
                    kind: Kind::Flip,
                });
            }
            _ => {}
        }
        if components.len() > MAX_COMPONENTS {
            return Err(QsoError::invalid(
                "circuit",
                format!("more than {MAX_COMPONENTS} error components"),
            ));
        }
    }
    components.sort();
    components.dedup();
    let signatures = propagate(circuit, &components)?;
    let sig = |c: Component| -> Signature {
        signatures[components
            .binary_search(&c)
            .expect("every component was propagated")]
        .clone()
    };
    for (i, op) in circuit.operations().iter().enumerate() {
        match op {
            QecOp::Noise { noise, qubits } => {
                let x = |qubit: usize| {
                    sig(Component {
                        op: i,
                        qubit,
                        kind: Kind::X,
                    })
                };
                let z = |qubit: usize| {
                    sig(Component {
                        op: i,
                        qubit,
                        kind: Kind::Z,
                    })
                };
                noise_channels(i, *noise, qubits, &x, &z, &mut f)?;
            }
            QecOp::Measure { flip, qubit } if *flip > 0.0 => {
                let s = sig(Component {
                    op: i,
                    qubit: 0,
                    kind: Kind::Flip,
                });
                flip_channel(i, *qubit, *flip, s, &mut f)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// The channels of noise operation `i` on `qubits`, whose X and Z errors
/// have the effects `x(q)` and `z(q)` (see [`each_channel`]).
pub(super) fn noise_channels(
    i: usize,
    noise: Noise,
    qubits: &[usize],
    x: &dyn Fn(usize) -> Signature,
    z: &dyn Fn(usize) -> Signature,
    f: &mut ChannelSink<'_>,
) -> Result<()> {
    let at = |parts: Vec<(usize, Kind)>| Channel { op: i, parts };
    let zx = |q: usize| vec![(q, Kind::Z), (q, Kind::X)];
    match noise {
        Noise::XError(p) => qubits
            .iter()
            .try_for_each(|&q| f(&at(vec![(q, Kind::X)]), &[x(q)], &[0.0, p])),
        Noise::ZError(p) => qubits
            .iter()
            .try_for_each(|&q| f(&at(vec![(q, Kind::Z)]), &[z(q)], &[0.0, p])),
        // As one error over the basis Z, X (so that it splits into its X
        // and Z parts like a depolarizing channel's).
        Noise::YError(p) => qubits
            .iter()
            .try_for_each(|&q| f(&at(zx(q)), &[z(q), x(q)], &[0.0, 0.0, 0.0, p])),
        Noise::Depolarize1(p) => {
            let r = independent(p, 1);
            qubits
                .iter()
                .try_for_each(|&q| f(&at(zx(q)), &[z(q), x(q)], &[0.0, r, r, r]))
        }
        Noise::Depolarize2(p) => {
            let r = independent(p, 2);
            let mut probabilities = [r; 16];
            probabilities[0] = 0.0;
            for pair in qubits.chunks(2) {
                let mut parts = zx(pair[0]);
                parts.extend(zx(pair[1]));
                f(
                    &at(parts),
                    &[z(pair[0]), x(pair[0]), z(pair[1]), x(pair[1])],
                    &probabilities,
                )?;
            }
            Ok(())
        }
    }
}

/// The channel of a measurement of `qubit` at operation `i` flipped with
/// probability `flip`, whose flip has effect `s`.
pub(super) fn flip_channel(
    i: usize,
    qubit: usize,
    flip: f64,
    s: Signature,
    f: &mut ChannelSink<'_>,
) -> Result<()> {
    f(
        &Channel {
            op: i,
            parts: vec![(qubit, Kind::Flip)],
        },
        &[s],
        &[0.0, flip],
    )
}

/// XOR `b` into `a`, counting the entries touched in `work`.
pub(super) fn xor_into(a: &mut Signature, b: &Signature, work: &mut u64) {
    if b.0.is_empty() {
        *work += 1;
        a.1 ^= b.1;
    } else if a.0.is_empty() {
        *work += b.0.len() as u64;
        a.0.clone_from(&b.0);
        a.1 ^= b.1;
    } else {
        *work += (a.0.len() + b.0.len()) as u64;
        *a = xor(a, b);
    }
}

/// One step of a Clifford gate's action on Pauli frames.
#[derive(Clone, Copy, Debug)]
pub(super) enum Step {
    H(usize),
    S(usize),
    Cx(usize, usize),
    Swap(usize, usize),
}

/// The steps a gate takes on a frame (signs aside), in order.
#[derive(Default)]
struct Steps(Vec<Step>);

impl CliffordTarget for Steps {
    fn h(&mut self, q: usize) {
        self.0.push(Step::H(q));
    }

    fn s(&mut self, q: usize) {
        self.0.push(Step::S(q));
    }

    // On frames S† acts as S, and Pauli gates only change signs.
    fn sdg(&mut self, q: usize) {
        self.0.push(Step::S(q));
    }

    fn x(&mut self, _: usize) {}

    fn y(&mut self, _: usize) {}

    fn z(&mut self, _: usize) {}

    fn cx(&mut self, c: usize, t: usize) {
        self.0.push(Step::Cx(c, t));
    }

    fn swap(&mut self, a: usize, b: usize) {
        self.0.push(Step::Swap(a, b));
    }
}

/// What an error at the current point of a pass backwards through a
/// circuit flips: on each qubit the effect of an X and of a Z there, and
/// the effect of flipping each measurement not yet passed whose detectors
/// or observables were. Every operation acts as the transpose of its
/// action on Pauli frames (a gate's steps in reverse; a measurement adds
/// its record's effect to the X of its qubit and clears the Z, as frames
/// drop the Z of a measured qubit; a reset clears both), so the effects are
/// exactly those of propagating each error forwards, in one pass.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Tracker {
    pub(super) x: Vec<Signature>,
    pub(super) z: Vec<Signature>,
    /// Effects of flipping measurement records not yet passed, their
    /// detectors descending (detectors are passed in descending order, so
    /// each is appended).
    pub(super) pending: HashMap<usize, Signature>,
    /// Entries touched so far.
    pub(super) work: u64,
}

impl Tracker {
    pub(super) fn new(n: usize) -> Self {
        Tracker {
            x: vec![(Vec::new(), 0); n],
            z: vec![(Vec::new(), 0); n],
            pending: HashMap::new(),
            work: 0,
        }
    }

    pub(super) fn gate(&mut self, inst: &Instruction, steps: &mut Vec<Step>) -> Result<()> {
        let mut record = Steps(std::mem::take(steps));
        record.0.clear();
        apply_clifford(&mut record, inst, &[])?;
        for step in record.0.iter().rev() {
            match *step {
                Step::H(q) => std::mem::swap(&mut self.x[q], &mut self.z[q]),
                Step::S(q) => {
                    let z = std::mem::take(&mut self.z[q]);
                    xor_into(&mut self.x[q], &z, &mut self.work);
                    self.z[q] = z;
                }
                Step::Cx(c, t) => {
                    let xt = std::mem::take(&mut self.x[t]);
                    xor_into(&mut self.x[c], &xt, &mut self.work);
                    self.x[t] = xt;
                    let zc = std::mem::take(&mut self.z[c]);
                    xor_into(&mut self.z[t], &zc, &mut self.work);
                    self.z[c] = zc;
                }
                Step::Swap(a, b) => {
                    self.x.swap(a, b);
                    self.z.swap(a, b);
                }
            }
        }
        *steps = record.0;
        Ok(())
    }

    /// Passes the measurement of `qubit` into `record`: returns the effect
    /// of flipping it.
    pub(super) fn measure(&mut self, qubit: usize, record: usize) -> Signature {
        let mut s = self.pending.remove(&record).unwrap_or_default();
        s.0.reverse();
        xor_into(&mut self.x[qubit], &s, &mut self.work);
        self.z[qubit] = (Vec::new(), 0);
        s
    }

    pub(super) fn reset(&mut self, qubit: usize) {
        self.x[qubit] = (Vec::new(), 0);
        self.z[qubit] = (Vec::new(), 0);
    }

    /// Passes detector `id`, below every detector passed before.
    pub(super) fn detector(&mut self, id: usize, records: &[usize]) {
        for &r in records {
            let e = &mut self.pending.entry(r).or_default().0;
            debug_assert!(e.last().is_none_or(|&d| d >= id));
            // A record listed twice cancels.
            if e.last() == Some(&id) {
                e.pop();
            } else {
                e.push(id);
            }
            self.work += 1;
        }
    }

    pub(super) fn observable(&mut self, index: usize, records: &[usize]) {
        for &r in records {
            self.pending.entry(r).or_default().1 ^= 1 << index;
            self.work += 1;
        }
    }
}

impl Tracker {
    /// Whether this state is `before` with every detector `dd` and every
    /// record `dm` lower (records whose effect is empty left aside).
    fn is_shift_of(&self, before: &Tracker, dd: usize, dm: usize) -> bool {
        let same = |a: &Signature, b: &Signature| {
            a.1 == b.1 && a.0.len() == b.0.len() && a.0.iter().zip(&b.0).all(|(x, y)| x + dd == *y)
        };
        let live = |p: &HashMap<usize, Signature>| p.values().filter(|s| !is_empty(s)).count();
        self.x.iter().zip(&before.x).all(|(a, b)| same(a, b))
            && self.z.iter().zip(&before.z).all(|(a, b)| same(a, b))
            && live(&self.pending) == live(&before.pending)
            && before
                .pending
                .iter()
                .filter(|(_, s)| !is_empty(s))
                .all(|(&r, s)| r >= dm && self.pending.get(&(r - dm)).is_some_and(|a| same(a, s)))
    }

    /// Every detector `dd` and every record `dm` lower.
    fn shift_down(&mut self, dd: usize, dm: usize) -> Result<()> {
        let lower = |s: &mut Signature| -> Result<()> {
            for d in &mut s.0 {
                *d = d.checked_sub(dd).ok_or_else(|| {
                    QsoError::invalid("circuit", "a repeated block shifted below detector 0")
                })?;
            }
            Ok(())
        };
        for s in self.x.iter_mut().chain(self.z.iter_mut()) {
            lower(s)?;
        }
        let mut pending = HashMap::with_capacity(self.pending.len());
        for (r, mut s) in std::mem::take(&mut self.pending) {
            if is_empty(&s) {
                continue;
            }
            lower(&mut s)?;
            let r = r.checked_sub(dm).ok_or_else(|| {
                QsoError::invalid("circuit", "a repeated block shifted below record 0")
            })?;
            pending.insert(r, s);
        }
        self.pending = pending;
        Ok(())
    }
}

/// A key of a merged model whose detectors can be shifted.
pub(super) trait Key: Ord + Clone {
    /// The key with every detector `by` lower.
    fn lowered(&self, by: usize) -> Self;
    /// Detector entries it holds.
    fn entries(&self) -> usize;
}

impl Key for Signature {
    fn lowered(&self, by: usize) -> Self {
        (self.0.iter().map(|d| d - by).collect(), self.1)
    }

    fn entries(&self) -> usize {
        self.0.len()
    }
}

impl Key for Vec<Signature> {
    fn lowered(&self, by: usize) -> Self {
        self.iter().map(|s| s.lowered(by)).collect()
    }

    fn entries(&self) -> usize {
        self.iter().map(|s| s.0.len()).sum()
    }
}

/// What one channel adds to a merged model: keys and probabilities.
pub(super) type Contribute<'a, K> =
    dyn FnMut(&Channel, &[Signature], &[f64], &mut Vec<(K, f64)>) -> Result<()> + 'a;

/// The pass backwards through a repeated circuit that folds its blocks.
struct Folder<'a, K: Key> {
    t: Tracker,
    steps: Vec<Step>,
    /// Measurements, detectors and operations before the current point.
    m: usize,
    d: usize,
    op: u64,
    merged: BTreeMap<K, f64>,
    /// The contributions of the current iteration of every open block.
    open: Vec<BTreeMap<K, f64>>,
    scratch: Vec<(K, f64)>,
    entries: usize,
    contribute: &'a mut Contribute<'a, K>,
}

impl<K: Key> Folder<'_, K> {
    fn add(&mut self, key: K, p: f64) -> Result<()> {
        for o in &mut self.open {
            merge(o, key.clone(), p);
        }
        let n = key.entries();
        let e = self.merged.entry(key);
        if let std::collections::btree_map::Entry::Vacant(_) = e {
            self.entries += n;
            if self.entries > MAX_SIGNATURE_ENTRIES {
                return Err(QsoError::invalid(
                    "circuit",
                    format!(
                        "the model's errors flip more than {MAX_SIGNATURE_ENTRIES} detectors in all"
                    ),
                ));
            }
        }
        let v = e.or_insert(0.0);
        *v = *v * (1.0 - p) + p * (1.0 - *v);
        Ok(())
    }

    fn channel(
        &mut self,
        channel: &Channel,
        basis: &[Signature],
        probabilities: &[f64],
    ) -> Result<()> {
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.clear();
        (self.contribute)(channel, basis, probabilities, &mut scratch)?;
        for (k, p) in scratch.drain(..) {
            self.add(k, p)?;
        }
        self.scratch = scratch;
        Ok(())
    }

    fn op(&mut self, op: &QecOp) -> Result<()> {
        self.op -= 1;
        let i = self.op as usize;
        match op {
            QecOp::Gate(inst) => self.t.gate(inst, &mut self.steps)?,
            QecOp::Measure { qubit, flip } => {
                self.m -= 1;
                let s = self.t.measure(*qubit, self.m);
                if *flip > 0.0 {
                    let mut add = |c: &Channel, b: &[Signature], p: &[f64]| self.channel(c, b, p);
                    flip_channel(i, *qubit, *flip, s, &mut add)?;
                }
            }
            QecOp::Reset { qubit } => self.t.reset(*qubit),
            QecOp::Noise { noise, qubits } => {
                // The effects are read while contributions are added.
                let (x, z) = (std::mem::take(&mut self.t.x), std::mem::take(&mut self.t.z));
                let mut add = |c: &Channel, b: &[Signature], p: &[f64]| self.channel(c, b, p);
                let done = noise_channels(
                    i,
                    *noise,
                    qubits,
                    &|q| x[q].clone(),
                    &|q| z[q].clone(),
                    &mut add,
                );
                self.t.x = x;
                self.t.z = z;
                done?;
            }
            QecOp::Detector { records } => {
                self.d -= 1;
                let r: Vec<usize> = records.iter().map(|&k| self.m - k).collect();
                self.t.detector(self.d, &r);
            }
            QecOp::Observable { index, records } => {
                let r: Vec<usize> = records.iter().map(|&k| self.m - k).collect();
                self.t.observable(*index, &r);
            }
        }
        if self.t.work > MAX_MODEL_WORK {
            return Err(QsoError::invalid(
                "circuit",
                format!("the error model needs more than {MAX_MODEL_WORK} operations"),
            ));
        }
        Ok(())
    }

    fn blocks(&mut self, blocks: &[Block]) -> Result<()> {
        for block in blocks.iter().rev() {
            match block {
                Block::Ops(ops) => {
                    for op in ops.iter().rev() {
                        self.op(op)?;
                    }
                }
                Block::Repeat(r) => self.repeat(r)?,
            }
        }
        Ok(())
    }

    /// The iterations of `r` from the last; once an iteration's start is
    /// its end shifted by one iteration, the earlier ones are its
    /// contributions shifted.
    fn repeat(&mut self, r: &Repeat) -> Result<()> {
        self.open.push(BTreeMap::new());
        let mut j = r.count;
        while j > 0 {
            j -= 1;
            self.open.last_mut().expect("this block").clear();
            let before = (j > 0).then(|| Tracker {
                work: 0,
                ..self.t.clone()
            });
            self.blocks(&r.body)?;
            let Some(before) = before else { break };
            if !self.t.is_shift_of(&before, r.detectors, r.measurements) {
                continue;
            }
            let last = self.open.pop().expect("this block");
            for i in 1..=j {
                let by = i * r.detectors;
                for (k, &p) in &last {
                    self.add(k.lowered(by), p)?;
                }
            }
            self.t.shift_down(j * r.detectors, j * r.measurements)?;
            self.m -= j * r.measurements;
            self.d -= j * r.detectors;
            self.op -= j as u64 * r.operations;
            return Ok(());
        }
        self.open.pop();
        Ok(())
    }
}

/// The merged contributions of every channel of a repeated circuit, by one
/// pass backwards that folds its blocks (see [`super::repeat`]).
pub(super) fn fold<'a, K: Key>(
    circuit: &RepeatedCircuit,
    contribute: &'a mut Contribute<'a, K>,
) -> Result<BTreeMap<K, f64>> {
    if circuit.n_detectors() > MAX_MODEL_DETECTORS {
        return Err(QsoError::invalid(
            "circuit",
            format!(
                "{} detectors; a model holds at most {MAX_MODEL_DETECTORS}",
                circuit.n_detectors()
            ),
        ));
    }
    super::frame::check_deterministic(circuit)?;
    let mut f = Folder {
        t: Tracker::new(circuit.n_qubits()),
        steps: Vec::new(),
        m: circuit.n_measurements(),
        d: circuit.n_detectors(),
        op: circuit.n_operations(),
        merged: BTreeMap::new(),
        open: Vec::new(),
        scratch: Vec::new(),
        entries: 0,
        contribute,
    };
    f.blocks(&circuit.blocks)?;
    debug_assert_eq!((f.m, f.d, f.op), (0, 0, 0));
    Ok(f.merged)
}

/// The contributions of a channel to the model of [`detector_error_model`].
pub(super) fn model_contributions(decompose: bool) -> Box<Contribute<'static, Signature>> {
    Box::new(move |_, basis, probabilities, out| {
        if !decompose {
            for (k, &p) in probabilities.iter().enumerate().skip(1) {
                let total = combination(basis, k);
                if p > 0.0 && !is_empty(&total) {
                    out.push((total, p));
                }
            }
            return Ok(());
        }
        for (parts, p) in decomposed(basis, probabilities)? {
            for part in parts {
                out.push((part, p));
            }
        }
        Ok(())
    })
}

/// The detector error model of a repeated circuit, as
/// [`detector_error_model`] gives it for the expanded circuit (up to the
/// rounding of the merged probabilities).
pub(super) fn repeated_error_model(
    circuit: &RepeatedCircuit,
    decompose: bool,
) -> Result<DetectorErrorModel> {
    let mut contribute = model_contributions(decompose);
    let merged = fold::<Signature>(circuit, contribute.as_mut())?;
    Ok(DetectorErrorModel {
        n_detectors: circuit.n_detectors(),
        n_observables: circuit.n_observables(),
        errors: merged
            .into_iter()
            .map(|((detectors, observables), probability)| ErrorMechanism {
                probability,
                detectors,
                observables,
            })
            .collect(),
    })
}

/// The correlated model of a repeated circuit, as
/// [`correlated_error_model`] gives it for the expanded circuit.
pub(super) fn repeated_correlated_model(circuit: &RepeatedCircuit) -> Result<CorrelatedErrorModel> {
    let mut contribute = |_: &Channel,
                          basis: &[Signature],
                          probabilities: &[f64],
                          out: &mut Vec<(Vec<Signature>, f64)>|
     -> Result<()> {
        for (mut parts, p) in decomposed(basis, probabilities)? {
            parts.sort();
            out.push((parts, p));
        }
        Ok(())
    };
    let merged = fold::<Vec<Signature>>(circuit, &mut contribute)?;
    Ok(CorrelatedErrorModel {
        n_detectors: circuit.n_detectors(),
        n_observables: circuit.n_observables(),
        errors: merged
            .into_iter()
            .map(|(parts, probability)| CorrelatedError { probability, parts })
            .collect(),
    })
}

/// The effect of each component (sorted by operation), by one pass
/// backwards through the circuit (see [`Tracker`]).
fn propagate(circuit: &DetectorCircuit, components: &[Component]) -> Result<Vec<Signature>> {
    let mut t = Tracker::new(circuit.n_qubits());
    let mut steps = Vec::new();
    let mut out: Vec<Signature> = vec![(Vec::new(), 0); components.len()];
    let (mut m, mut d) = (circuit.n_measurements(), circuit.n_detectors());
    let mut next = components.len();
    let mut entries = 0usize;
    for (i, op) in circuit.operations().iter().enumerate().rev() {
        match op {
            QecOp::Gate(inst) => t.gate(inst, &mut steps)?,
            QecOp::Measure { qubit, .. } => {
                m -= 1;
                let s = t.measure(*qubit, m);
                while next > 0 && components[next - 1].op == i {
                    next -= 1;
                    entries += s.0.len();
                    out[next] = s.clone();
                }
            }
            QecOp::Reset { qubit } => t.reset(*qubit),
            QecOp::Noise { .. } => {
                while next > 0 && components[next - 1].op == i {
                    next -= 1;
                    let c = components[next];
                    let s = match c.kind {
                        Kind::X => &t.x[c.qubit],
                        _ => &t.z[c.qubit],
                    };
                    entries += s.0.len();
                    out[next] = s.clone();
                }
            }
            QecOp::Detector { records } => {
                d -= 1;
                t.detector(d, records);
            }
            QecOp::Observable { index, records } => t.observable(*index, records),
        }
        if t.work > MAX_MODEL_WORK {
            return Err(QsoError::invalid(
                "circuit",
                format!("the error model needs more than {MAX_MODEL_WORK} operations"),
            ));
        }
        if entries > MAX_SIGNATURE_ENTRIES {
            return Err(QsoError::invalid(
                "circuit",
                format!(
                    "the errors flip more than {MAX_SIGNATURE_ENTRIES} detectors in all before merging"
                ),
            ));
        }
    }
    Ok(out)
}

/// Words of frame and record memory one propagation pass may use.
#[cfg(test)]
const PASS_MEMORY_WORDS: usize = 1 << 24;

/// The effect of each component (sorted by operation), up to 64 words of
/// 64 components per pass forwards through the circuit: the reference of
/// [`propagate`], quadratic in the rounds of an experiment.
#[cfg(test)]
fn propagate_forward(
    circuit: &DetectorCircuit,
    components: &[Component],
) -> Result<Vec<Signature>> {
    let n = circuit.n_qubits();
    let (n_det, n_obs) = (circuit.n_detectors(), circuit.n_observables());
    let per_word = circuit.n_measurements() + 2 * n + n_det + n_obs;
    let max_words = (PASS_MEMORY_WORDS / per_word.max(1)).clamp(1, 64);
    // Each pass runs from its first component to the end of the circuit.
    let ops = circuit.operations().len() as u64;
    let work = components
        .chunks(64 * max_words)
        .map(|chunk| (ops - chunk[0].op as u64).saturating_mul(chunk.len().div_ceil(64) as u64))
        .fold(0u64, u64::saturating_add);
    if work > MAX_MODEL_WORK {
        return Err(QsoError::invalid(
            "circuit",
            format!(
                "the error model needs about {work} word operations, more than {MAX_MODEL_WORK}"
            ),
        ));
    }
    // Measurements and detectors before each operation: a pass starts at
    // its first component, before which its frames, records and detectors
    // are zero.
    let mut before = Vec::with_capacity(circuit.operations().len() + 1);
    let (mut m_count, mut d_count) = (0usize, 0usize);
    for op in circuit.operations() {
        before.push((m_count, d_count));
        match op {
            QecOp::Measure { .. } => m_count += 1,
            QecOp::Detector { .. } => d_count += 1,
            _ => {}
        }
    }
    before.push((m_count, d_count));
    let mut entries = 0usize;
    let mut out: Vec<Signature> = Vec::with_capacity(components.len());
    for chunk in components.chunks(64 * max_words) {
        let w = chunk.len().div_ceil(64);
        let first = chunk[0].op;
        let mut f = Frames::new(n, w);
        let mut records: Vec<u64> = Vec::with_capacity(circuit.n_measurements() * w);
        records.resize(before[first].0 * w, 0);
        let mut det: Vec<u64> = Vec::with_capacity(n_det * w);
        det.resize(before[first].1 * w, 0);
        let mut obs = vec![0u64; n_obs * w];
        let mut next = 0;
        for (i, op) in circuit.operations().iter().enumerate().skip(first) {
            match op {
                // Frames are zero before the first component.
                QecOp::Gate(inst) if i > first => apply_clifford(&mut f, inst, &[])?,
                QecOp::Gate(_) => {}
                QecOp::Noise { .. } => {
                    while next < chunk.len() && chunk[next].op == i {
                        let c = chunk[next];
                        let v = if c.kind == Kind::X {
                            &mut f.x
                        } else {
                            &mut f.z
                        };
                        v[c.qubit * w + next / 64] ^= 1 << (next % 64);
                        next += 1;
                    }
                }
                QecOp::Measure { qubit, .. } => {
                    let start = records.len();
                    records.extend_from_slice(&f.x[qubit * w..(qubit + 1) * w]);
                    while next < chunk.len() && chunk[next].op == i {
                        records[start + next / 64] ^= 1 << (next % 64);
                        next += 1;
                    }
                    // The qubit is now in a Z eigenstate: Z errors on it
                    // have no effect.
                    f.z[qubit * w..(qubit + 1) * w].fill(0);
                }
                QecOp::Reset { qubit } => {
                    f.x[qubit * w..(qubit + 1) * w].fill(0);
                    f.z[qubit * w..(qubit + 1) * w].fill(0);
                }
                QecOp::Detector { records: recs } => {
                    for k in 0..w {
                        det.push(recs.iter().fold(0, |a, &r| a ^ records[r * w + k]));
                    }
                }
                QecOp::Observable {
                    index,
                    records: recs,
                } => {
                    for k in 0..w {
                        obs[index * w + k] ^= recs.iter().fold(0, |a, &r| a ^ records[r * w + k]);
                    }
                }
            }
        }
        // Signatures: the set bits of each detector and observable row.
        let base = out.len();
        out.resize(base + chunk.len(), (Vec::new(), 0));
        entries += det.iter().map(|b| b.count_ones() as usize).sum::<usize>();
        if entries > MAX_SIGNATURE_ENTRIES {
            return Err(QsoError::invalid(
                "circuit",
                format!(
                    "the errors flip more than {MAX_SIGNATURE_ENTRIES} detectors in all before merging"
                ),
            ));
        }
        for d in 0..n_det {
            for k in 0..w {
                let mut bits = det[d * w + k];
                while bits != 0 {
                    out[base + 64 * k + bits.trailing_zeros() as usize]
                        .0
                        .push(d);
                    bits &= bits - 1;
                }
            }
        }
        for i in 0..n_obs {
            for k in 0..w {
                let mut bits = obs[i * w + k];
                while bits != 0 {
                    out[base + 64 * k + bits.trailing_zeros() as usize].1 |= 1 << i;
                    bits &= bits - 1;
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod backward_tests {
    use super::*;
    use crate::qec::codes::{
        CircuitNoise, MemoryBasis, repetition_code_memory, surface_code_memory_with,
    };
    use crate::qec::css::{CssCode, css_code_memory};
    use crate::rng::Rng;

    fn components(circuit: &DetectorCircuit) -> Vec<Component> {
        let mut out = Vec::new();
        for (i, op) in circuit.operations().iter().enumerate() {
            match op {
                QecOp::Noise { qubits, .. } => {
                    for &qubit in qubits {
                        for kind in [Kind::X, Kind::Z] {
                            out.push(Component { op: i, qubit, kind });
                        }
                    }
                }
                QecOp::Measure { flip, .. } if *flip > 0.0 => out.push(Component {
                    op: i,
                    qubit: 0,
                    kind: Kind::Flip,
                }),
                _ => {}
            }
        }
        out.sort();
        out.dedup();
        out
    }

    fn same(circuit: &DetectorCircuit) {
        let c = components(circuit);
        let back = propagate(circuit, &c).unwrap();
        let forward = propagate_forward(circuit, &c).unwrap();
        assert_eq!(back, forward);
    }

    /// A random noisy Clifford circuit on `n` qubits with detectors and
    /// observables on random records (deterministic or not: the effects
    /// of errors are linear either way).
    fn random_circuit(n: usize, ops: usize, rng: &mut Rng) -> DetectorCircuit {
        use crate::quantum::{GateKind, Param};
        let mut c = DetectorCircuit::new(n).unwrap();
        let q = |rng: &mut Rng| rng.below(n as u64) as usize;
        for _ in 0..ops {
            let a = q(rng);
            let b = (a + 1 + rng.below(n as u64 - 1) as usize) % n;
            match rng.below(14) {
                0 => {
                    c.h(a).unwrap();
                }
                1 => {
                    c.s(a).unwrap();
                }
                2 => {
                    c.cx(a, b).unwrap();
                }
                3 => {
                    c.cz(a, b).unwrap();
                }
                4 => {
                    let kind = [GateKind::Sdg, GateKind::SX, GateKind::SXdg, GateKind::Y]
                        [rng.below(4) as usize];
                    c.gate(Instruction {
                        kind,
                        targets: vec![a],
                        controls: vec![],
                        params: vec![],
                        matrix: None,
                    })
                    .unwrap();
                }
                5 => {
                    c.gate(Instruction {
                        kind: GateKind::RY,
                        targets: vec![a],
                        controls: vec![],
                        params: vec![Param::Value(std::f64::consts::FRAC_PI_2)],
                        matrix: None,
                    })
                    .unwrap();
                }
                6 => {
                    let kind = [GateKind::Swap, GateKind::Y][rng.below(2) as usize];
                    let inst = if kind == GateKind::Swap {
                        Instruction {
                            kind,
                            targets: vec![a, b],
                            controls: vec![],
                            params: vec![],
                            matrix: None,
                        }
                    } else {
                        Instruction {
                            kind,
                            targets: vec![b],
                            controls: vec![a],
                            params: vec![],
                            matrix: None,
                        }
                    };
                    c.gate(inst).unwrap();
                }
                7 | 8 => {
                    c.measure(a, if rng.below(2) == 0 { 0.0 } else { 0.01 })
                        .unwrap();
                }
                9 => {
                    c.reset(a).unwrap();
                }
                10 => {
                    let noise = [
                        Noise::XError(0.01),
                        Noise::YError(0.02),
                        Noise::ZError(0.03),
                        Noise::Depolarize1(0.01),
                    ][rng.below(4) as usize];
                    c.noise(noise, &[a]).unwrap();
                }
                11 => {
                    c.noise(Noise::Depolarize2(0.01), &[a, b]).unwrap();
                }
                _ => {
                    let m = c.n_measurements();
                    if m > 0 {
                        let k = 1 + rng.below(3) as usize;
                        let recs: Vec<usize> =
                            (0..k).map(|_| rng.below(m as u64) as usize).collect();
                        if rng.below(4) == 0 {
                            c.observable(rng.below(3) as usize, &recs).unwrap();
                        } else {
                            c.detector(&recs).unwrap();
                        }
                    }
                }
            }
        }
        c
    }

    #[test]
    fn the_pass_backwards_equals_frames_propagated_forwards() {
        let noise = CircuitNoise::uniform(1e-3);
        same(&repetition_code_memory(5, 7, &noise).unwrap());
        for basis in [MemoryBasis::Z, MemoryBasis::X] {
            same(&surface_code_memory_with(3, 4, &noise, basis).unwrap());
            same(&surface_code_memory_with(5, 3, &noise, basis).unwrap());
        }
        let colour = CssCode::color_code(5).unwrap();
        same(&css_code_memory(&colour, 3, &noise, MemoryBasis::Z).unwrap());
        let (bb, _) = CssCode::bivariate_bicycle_code(72).unwrap();
        same(&css_code_memory(&bb, 2, &noise, MemoryBasis::X).unwrap());
        let mut rng = Rng::seed_from_u64(61);
        for trial in 0..200 {
            let n = 2 + trial % 6;
            same(&random_circuit(n, 20 + 3 * trial, &mut rng));
        }
    }
}
