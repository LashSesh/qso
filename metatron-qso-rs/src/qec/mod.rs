//! Quantum error-correction experiments.
//!
//! * [`DetectorCircuit`]: Clifford circuits with Pauli noise, measurement
//!   records, detectors and logical observables;
//! * [`sample_detectors`]: Pauli-frame sampling, 64 shots per machine word;
//! * [`detector_error_model`]: the independent errors of a circuit and the
//!   detectors and observables each flips;
//! * [`MatchingGraph`] and [`UnionFindDecoder`]: decoding by weighted
//!   union-find with peeling;
//! * [`repetition_code_memory`] and [`surface_code_memory`]: memory
//!   experiments with circuit-level noise;
//! * [`logical_error_rate`]: sample, decode and count.

pub mod circuit;
pub mod codes;
pub mod decoder;
pub mod dem;
pub mod frame;

pub use circuit::{DetectorCircuit, MAX_OBSERVABLES, Noise, QecOp};
pub use codes::{CircuitNoise, repetition_code_memory, surface_code_memory};
pub use decoder::{MatchingEdge, MatchingGraph, UnionFindDecoder};
pub use dem::{DetectorErrorModel, ErrorMechanism, detector_error_model};
pub use frame::{DetectorSamples, sample_detectors};

use crate::error::Result;
use serde::{Deserialize, Serialize};

/// Logical failures of a decoder on sampled shots.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogicalErrorRate {
    pub shots: usize,
    /// Shots where the predicted observable flips differ from the actual.
    pub errors: usize,
    pub rate: f64,
    /// Binomial standard error of `rate`.
    pub standard_error: f64,
}

/// Sample `shots` shots of `circuit`, decode each with the union-find
/// decoder of its (decomposed) detector error model, and count the shots
/// whose observables it predicts wrongly.
pub fn logical_error_rate(
    circuit: &DetectorCircuit,
    shots: usize,
    seed: u64,
) -> Result<LogicalErrorRate> {
    let dem = detector_error_model(circuit, true)?;
    let mut decoder = UnionFindDecoder::new(MatchingGraph::from_dem(&dem)?);
    let samples = sample_detectors(circuit, shots, seed)?;
    let mut errors = 0;
    for s in 0..shots {
        let fired = samples.fired_detectors(s);
        let predicted = if fired.is_empty() {
            0
        } else {
            decoder.decode(&fired)?
        };
        if predicted != samples.observable_flips(s) {
            errors += 1;
        }
    }
    let rate = if shots == 0 {
        0.0
    } else {
        errors as f64 / shots as f64
    };
    let standard_error = if shots == 0 {
        0.0
    } else {
        (rate * (1.0 - rate) / shots as f64).sqrt()
    };
    Ok(LogicalErrorRate {
        shots,
        errors,
        rate,
        standard_error,
    })
}

#[cfg(test)]
mod tests;
