//! Detector error models: which detectors and observables each error
//! flips, and how likely it is.
//!
//! Every noise channel is written as independent Pauli errors (exactly:
//! a depolarizing channel equals independent X, Y and Z errors, or the 15
//! two-qubit Paulis, with adjusted probabilities). Each error is
//! propagated through the noiseless circuit as a Pauli frame, 64 errors
//! per machine word; errors with the same effect are merged.

use std::collections::{BTreeMap, HashMap};

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
/// measurement flip.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Component {
    Pauli { op: usize, qubit: usize, x: bool },
    Flip { op: usize },
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
                for &q in qubits {
                    components.push(Component::Pauli {
                        op: i,
                        qubit: q,
                        x: true,
                    });
                    components.push(Component::Pauli {
                        op: i,
                        qubit: q,
                        x: false,
                    });
                }
            }
            QecOp::Measure { flip, .. } if *flip > 0.0 => {
                components.push(Component::Flip { op: i });
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
    let index: HashMap<Component, usize> = components
        .iter()
        .enumerate()
        .map(|(k, c)| (*c, k))
        .collect();
    let sig = |c: Component| -> &Signature { &signatures[index[&c]] };

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
                let xq = |q: usize| Component::Pauli {
                    op: i,
                    qubit: q,
                    x: true,
                };
                let zq = |q: usize| Component::Pauli {
                    op: i,
                    qubit: q,
                    x: false,
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
                term(vec![Component::Flip { op: i }], *flip)?;
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

/// The effect of each component, 64 components per pass.
fn propagate(circuit: &DetectorCircuit, components: &[Component]) -> Result<Vec<Signature>> {
    let n = circuit.n_qubits();
    let mut out = Vec::with_capacity(components.len());
    for chunk in components.chunks(64) {
        // Components of this chunk by operation index.
        let mut at: HashMap<usize, Vec<(usize, Component)>> = HashMap::new();
        for (lane, c) in chunk.iter().enumerate() {
            let op = match c {
                Component::Pauli { op, .. } | Component::Flip { op } => *op,
            };
            at.entry(op).or_default().push((lane, *c));
        }
        let mut f = Frames::new(n, 1);
        let mut records: Vec<u64> = Vec::with_capacity(circuit.n_measurements());
        let mut det: Vec<u64> = Vec::with_capacity(circuit.n_detectors());
        let mut obs = vec![0u64; circuit.n_observables()];
        for (i, op) in circuit.operations().iter().enumerate() {
            let here = at.get(&i);
            match op {
                QecOp::Gate(inst) => apply_clifford(&mut f, inst, &[])?,
                QecOp::Noise { .. } => {
                    for &(lane, c) in here.into_iter().flatten() {
                        if let Component::Pauli { qubit, x, .. } = c {
                            let v = if x { &mut f.x } else { &mut f.z };
                            v[qubit] ^= 1 << lane;
                        }
                    }
                }
                QecOp::Measure { qubit, .. } => {
                    let mut bits = f.x[*qubit];
                    for &(lane, _) in here.into_iter().flatten() {
                        bits ^= 1 << lane;
                    }
                    records.push(bits);
                    // The qubit is now in a Z eigenstate: Z errors on it
                    // have no effect.
                    f.z[*qubit] = 0;
                }
                QecOp::Reset { qubit } => {
                    f.x[*qubit] = 0;
                    f.z[*qubit] = 0;
                }
                QecOp::Detector { records: recs } => {
                    det.push(recs.iter().fold(0, |a, &r| a ^ records[r]));
                }
                QecOp::Observable {
                    index,
                    records: recs,
                } => {
                    obs[*index] ^= recs.iter().fold(0, |a, &r| a ^ records[r]);
                }
            }
        }
        for lane in 0..chunk.len() {
            let detectors = (0..det.len())
                .filter(|&d| (det[d] >> lane) & 1 == 1)
                .collect();
            let observables = obs
                .iter()
                .enumerate()
                .filter(|(_, o)| (**o >> lane) & 1 == 1)
                .fold(0u64, |m, (i, _)| m | (1 << i));
            out.push((detectors, observables));
        }
    }
    Ok(out)
}
