//! Detector error models: which detectors and observables each error
//! flips, and how likely it is.
//!
//! Every noise channel is written as independent Pauli errors (exactly:
//! a depolarizing channel equals independent X, Y and Z errors, or the 15
//! two-qubit Paulis, with adjusted probabilities). Each error is
//! propagated through the noiseless circuit as a Pauli frame, 64 errors
//! per machine word and up to 4096 per pass; errors with the same effect are merged.

use std::collections::BTreeMap;

use super::circuit::{DetectorCircuit, Noise, QecOp};
use super::frame::Frames;
use crate::error::{QsoError, Result};
use crate::stabilizer::apply_clifford;

/// Most elementary error components a model propagates.
pub const MAX_COMPONENTS: usize = 1 << 22;
/// Most detector entries over all propagated components (1 GiB).
pub const MAX_SIGNATURE_ENTRIES: usize = 1 << 27;
/// Most word operations of propagation, passes × operations × words
/// (about a minute and a half at the limit).
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
enum Kind {
    X,
    Z,
    Flip,
}

/// Effect of an error: flipped detectors (sorted) and observables.
type Signature = (Vec<usize>, u64);

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
    each_channel(circuit, |basis, probabilities| {
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
/// errors) is split into its basis errors.
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
    let mut out = Vec::new();
    for (k, &p) in probabilities.iter().enumerate().skip(1) {
        if p == 0.0 || is_empty(&combos[k]) {
            continue;
        }
        let parts = if count(k) == 0 || (!wide && solved[k]) {
            vec![combos[k].clone()]
        } else if let Some(parts) = (!wide).then(|| split(k)).flatten() {
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
                        "an error flips {} detectors even after splitting into single-qubit X and Z parts; the model is not a matching graph",
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
/// single-detector combinations. A combination it cannot write so (Stim
/// then decomposes against the graph-like errors of the whole model) is
/// split into its single-qubit X and Z parts. Errors with the same parts
/// are merged.
pub fn correlated_error_model(circuit: &DetectorCircuit) -> Result<CorrelatedErrorModel> {
    let mut merged: BTreeMap<Vec<Signature>, f64> = BTreeMap::new();
    each_channel(circuit, |basis, probabilities| {
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

/// Calls `f(basis, probabilities)` for every noise channel of the circuit:
/// the effects of its basis errors in Stim's order (for a depolarizing
/// channel Z then X on each qubit; a measurement flip, a single-qubit X, Y
/// or Z error is one basis error) and the probability of each combination
/// of them (index: bit `i` for basis error `i`), as independent errors.
fn each_channel(
    circuit: &DetectorCircuit,
    mut f: impl FnMut(&[Signature], &[f64]) -> Result<()>,
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
                match *noise {
                    Noise::XError(p) => qubits.iter().try_for_each(|&q| f(&[x(q)], &[0.0, p]))?,
                    Noise::ZError(p) => qubits.iter().try_for_each(|&q| f(&[z(q)], &[0.0, p]))?,
                    Noise::YError(p) => qubits
                        .iter()
                        .try_for_each(|&q| f(&[xor(&x(q), &z(q))], &[0.0, p]))?,
                    Noise::Depolarize1(p) => {
                        let r = independent(p, 1);
                        for &q in qubits {
                            f(&[z(q), x(q)], &[0.0, r, r, r])?;
                        }
                    }
                    Noise::Depolarize2(p) => {
                        let r = independent(p, 2);
                        let mut probabilities = [r; 16];
                        probabilities[0] = 0.0;
                        for pair in qubits.chunks(2) {
                            f(
                                &[z(pair[0]), x(pair[0]), z(pair[1]), x(pair[1])],
                                &probabilities,
                            )?;
                        }
                    }
                }
            }
            QecOp::Measure { flip, .. } if *flip > 0.0 => {
                f(
                    &[sig(Component {
                        op: i,
                        qubit: 0,
                        kind: Kind::Flip,
                    })],
                    &[0.0, *flip],
                )?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Words of frame and record memory one propagation pass may use.
const PASS_MEMORY_WORDS: usize = 1 << 24;

/// The effect of each component (sorted by operation), up to 64 words of
/// 64 components per pass through the circuit.
fn propagate(circuit: &DetectorCircuit, components: &[Component]) -> Result<Vec<Signature>> {
    let n = circuit.n_qubits();
    let (n_det, n_obs) = (circuit.n_detectors(), circuit.n_observables());
    let per_word = circuit.n_measurements() + 2 * n + n_det + n_obs;
    let max_words = (PASS_MEMORY_WORDS / per_word.max(1)).clamp(1, 64);
    let passes = components.len().div_ceil(64 * max_words) as u64;
    let work = passes
        .saturating_mul(circuit.operations().len() as u64)
        .saturating_mul(max_words.min(components.len().div_ceil(64)) as u64);
    if work > MAX_MODEL_WORK {
        return Err(QsoError::invalid(
            "circuit",
            format!(
                "the error model needs about {work} word operations, more than {MAX_MODEL_WORK}"
            ),
        ));
    }
    let mut entries = 0usize;
    let mut out: Vec<Signature> = Vec::with_capacity(components.len());
    for chunk in components.chunks(64 * max_words) {
        let w = chunk.len().div_ceil(64);
        let first = chunk[0].op;
        let mut f = Frames::new(n, w);
        let mut records: Vec<u64> = Vec::with_capacity(circuit.n_measurements() * w);
        let mut det: Vec<u64> = Vec::with_capacity(n_det * w);
        let mut obs = vec![0u64; n_obs * w];
        let mut next = 0;
        for (i, op) in circuit.operations().iter().enumerate() {
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
