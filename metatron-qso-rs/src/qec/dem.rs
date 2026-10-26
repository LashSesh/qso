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

/// One independent error mechanism of a detector error model.
#[derive(Clone, Debug, PartialEq)]
pub struct ErrorMechanism {
    pub probability: f64,
    /// Detectors it flips, ascending.
    pub detectors: Vec<usize>,
    /// Observables it flips (bit `i`: observable `i`).
    pub observables: u64,
}

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

/// The detector error model of `circuit`. With `decompose`, an error that
/// flips more than two detectors is split into its single-qubit X and Z
/// parts (each must flip at most two), so that the model is a matching
/// graph; this keeps every part's probability and so overestimates
/// correlated errors slightly, as decoders do.
pub fn detector_error_model(
    circuit: &DetectorCircuit,
    decompose: bool,
) -> Result<DetectorErrorModel> {
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
    let sig = |c: Component| -> &Signature {
        &signatures[components
            .binary_search(&c)
            .expect("every component was propagated")]
    };

    // Independent error terms and their effects.
    let mut merged: BTreeMap<Signature, f64> = BTreeMap::new();
    let mut add = |s: Signature, p: f64| {
        if s.0.is_empty() && s.1 == 0 {
            return;
        }
        let e = merged.entry(s).or_insert(0.0);
        *e = *e * (1.0 - p) + p * (1.0 - *e);
    };
    let mut term = |parts: Vec<Component>, p: f64| -> Result<()> {
        let total = parts
            .iter()
            .fold((Vec::new(), 0u64), |acc, &c| xor(&acc, sig(c)));
        if !decompose || total.0.len() <= 2 {
            add(total, p);
            return Ok(());
        }
        for c in parts {
            let s = sig(c).clone();
            if s.0.len() > 2 {
                return Err(QsoError::invalid(
                    "circuit",
                    format!(
                        "an error flips {} detectors even after splitting into single-qubit X and Z parts; the model is not a matching graph",
                        s.0.len()
                    ),
                ));
            }
            add(s, p);
        }
        Ok(())
    };
    for (i, op) in circuit.operations().iter().enumerate() {
        match op {
            QecOp::Noise { noise, qubits } => {
                let xq = |qubit: usize| Component {
                    op: i,
                    qubit,
                    kind: Kind::X,
                };
                let zq = |qubit: usize| Component {
                    op: i,
                    qubit,
                    kind: Kind::Z,
                };
                match *noise {
                    Noise::XError(p) => qubits.iter().try_for_each(|&q| term(vec![xq(q)], p))?,
                    Noise::ZError(p) => qubits.iter().try_for_each(|&q| term(vec![zq(q)], p))?,
                    Noise::YError(p) => qubits
                        .iter()
                        .try_for_each(|&q| term(vec![xq(q), zq(q)], p))?,
                    Noise::Depolarize1(p) => {
                        let r = independent(p, 1);
                        for &q in qubits {
                            term(vec![xq(q)], r)?;
                            term(vec![zq(q)], r)?;
                            term(vec![xq(q), zq(q)], r)?;
                        }
                    }
                    Noise::Depolarize2(p) => {
                        let r = independent(p, 2);
                        for pair in qubits.chunks(2) {
                            for k in 1..16u32 {
                                let mut parts = Vec::new();
                                if k & 1 != 0 {
                                    parts.push(xq(pair[0]));
                                }
                                if k & 2 != 0 {
                                    parts.push(zq(pair[0]));
                                }
                                if k & 4 != 0 {
                                    parts.push(xq(pair[1]));
                                }
                                if k & 8 != 0 {
                                    parts.push(zq(pair[1]));
                                }
                                term(parts, r)?;
                            }
                        }
                    }
                }
            }
            QecOp::Measure { flip, .. } if *flip > 0.0 => {
                term(
                    vec![Component {
                        op: i,
                        qubit: 0,
                        kind: Kind::Flip,
                    }],
                    *flip,
                )?;
            }
            _ => {}
        }
    }
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

/// Words of frame and record memory one propagation pass may use.
const PASS_MEMORY_WORDS: usize = 1 << 24;

/// The effect of each component (sorted by operation), up to 64 words of
/// 64 components per pass through the circuit.
fn propagate(circuit: &DetectorCircuit, components: &[Component]) -> Result<Vec<Signature>> {
    let n = circuit.n_qubits();
    let (n_det, n_obs) = (circuit.n_detectors(), circuit.n_observables());
    let per_word = circuit.n_measurements() + 2 * n + n_det + n_obs;
    let max_words = (PASS_MEMORY_WORDS / per_word.max(1)).clamp(1, 64);
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
