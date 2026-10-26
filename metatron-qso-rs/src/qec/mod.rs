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
//! * [`logical_error_rate`]: sample, decode and count;
//! * [`DetectorCircuit::to_stim`]: export in Stim's text format.

pub mod circuit;
pub mod codes;
pub mod decoder;
pub mod dem;
pub mod frame;
mod stim;

pub use circuit::{DetectorCircuit, MAX_OBSERVABLES, Noise, QecOp};
pub use codes::{CircuitNoise, repetition_code_memory, surface_code_memory};
pub use decoder::{MatchingEdge, MatchingGraph, UnionFindDecoder};
pub use dem::{DetectorErrorModel, ErrorMechanism, detector_error_model};
pub use frame::{DetectorSamples, sample_detectors};

use crate::error::Result;
use crate::rng::Rng;
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
/// whose observables it predicts wrongly. Shots are sampled in blocks, so
/// memory does not grow with `shots`; the shots are those of
/// [`sample_detectors`] with the same seed.
pub fn logical_error_rate(
    circuit: &DetectorCircuit,
    shots: usize,
    seed: u64,
) -> Result<LogicalErrorRate> {
    let dem = detector_error_model(circuit, true)?;
    let mut decoder = UnionFindDecoder::new(MatchingGraph::from_dem(&dem)?);
    frame::check_deterministic(circuit)?;
    let mut rng = Rng::seed_from_u64(seed);
    let block = 64 * frame::BATCH_SHOTS;
    let mut errors = 0;
    let mut done = 0;
    while done < shots {
        let k = (shots - done).min(block);
        let samples = frame::sample_with(circuit, k, &mut rng)?;
        for (s, fired) in samples.fired_by_shot().into_iter().enumerate() {
            let predicted = if fired.is_empty() {
                0
            } else {
                decoder.decode(&fired)?
            };
            if predicted != samples.observable_flips(s) {
                errors += 1;
            }
        }
        done += k;
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
