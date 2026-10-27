//! Quantum error-correction experiments.
//!
//! * [`DetectorCircuit`]: Clifford circuits with Pauli noise, measurement
//!   records, detectors and logical observables;
//! * [`sample_detectors`]: Pauli-frame sampling, 64 shots per machine word;
//! * [`detector_error_model`]: the independent errors of a circuit and the
//!   detectors and observables each flips;
//! * [`MatchingGraph`], [`UnionFindDecoder`], [`SparseMatchingDecoder`]
//!   and [`MatchingDecoder`]: decoding by weighted union-find with
//!   peeling, or by exact minimum-weight perfect matching, grown on the
//!   detector graph or computed on the complete graph of detection events;
//! * [`repetition_code_memory`] and [`surface_code_memory`]
//!   ([`surface_code_memory_with`] for the X basis): memory experiments
//!   with circuit-level noise;
//! * [`logical_error_rate`] and [`logical_error_rate_with`]: sample,
//!   decode and count;
//! * [`DetectorCircuit::to_stim`]: export in Stim's text format.

pub mod circuit;
pub mod codes;
pub mod decoder;
pub mod dem;
pub mod frame;
pub mod matching;
pub mod sparse;
mod stim;

pub use circuit::{DetectorCircuit, MAX_OBSERVABLES, Noise, QecOp};
pub use codes::{
    CircuitNoise, MemoryBasis, repetition_code_memory, surface_code_memory,
    surface_code_memory_with,
};
pub use decoder::{MatchingEdge, MatchingGraph, UnionFindDecoder};
pub use dem::{DetectorErrorModel, ErrorMechanism, detector_error_model};
pub use frame::{DetectorSamples, sample_detectors};
pub use matching::{MAX_MATCHING_PAIRS, MatchingDecoder};
pub use sparse::SparseMatchingDecoder;

use crate::error::Result;
use crate::rng::Rng;
use serde::{Deserialize, Serialize};

/// A decoder of detection events.
pub trait Decoder {
    /// Predicted observable flips for the fired detectors.
    fn decode(&mut self, fired: &[usize]) -> Result<u64>;
}

impl Decoder for UnionFindDecoder {
    fn decode(&mut self, fired: &[usize]) -> Result<u64> {
        UnionFindDecoder::decode(self, fired)
    }
}

impl Decoder for MatchingDecoder {
    fn decode(&mut self, fired: &[usize]) -> Result<u64> {
        MatchingDecoder::decode(self, fired)
    }
}

impl Decoder for SparseMatchingDecoder {
    fn decode(&mut self, fired: &[usize]) -> Result<u64> {
        SparseMatchingDecoder::decode(self, fired)
    }
}

/// The decoders [`logical_error_rate_with`] can use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DecoderKind {
    /// Weighted union-find ([`UnionFindDecoder`]): almost linear time.
    #[default]
    UnionFind,
    /// Minimum-weight perfect matching by region growth
    /// ([`SparseMatchingDecoder`]; the same weights as
    /// [`MatchingDecoder`]).
    Matching,
}

impl DecoderKind {
    /// The decoder of a matching graph.
    pub fn build(self, graph: MatchingGraph) -> Box<dyn Decoder + Send> {
        match self {
            DecoderKind::UnionFind => Box::new(UnionFindDecoder::new(graph)),
            DecoderKind::Matching => Box::new(SparseMatchingDecoder::new(graph)),
        }
    }
}

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
/// whose observables it predicts wrongly. Shots are sampled in blocks of at
/// most 65 536 shots and 1 GiB of results, so memory does not grow with
/// `shots`; the shots are those of [`sample_detectors`] with the same
/// seed.
pub fn logical_error_rate(
    circuit: &DetectorCircuit,
    shots: usize,
    seed: u64,
) -> Result<LogicalErrorRate> {
    logical_error_rate_with(circuit, shots, seed, DecoderKind::UnionFind)
}

/// [`logical_error_rate`] with the given decoder; the shots are the same
/// for every decoder.
pub fn logical_error_rate_with(
    circuit: &DetectorCircuit,
    shots: usize,
    seed: u64,
    decoder: DecoderKind,
) -> Result<LogicalErrorRate> {
    let dem = detector_error_model(circuit, true)?;
    let mut decoder = decoder.build(MatchingGraph::from_dem(&dem)?);
    frame::check_deterministic(circuit)?;
    let mut rng = Rng::seed_from_u64(seed);
    // Blocks of whole batches, with at most MAX_SAMPLED_EVENTS bits of
    // results, so the stream equals one call.
    let batch = frame::batch_shots(circuit);
    let per_shot = circuit.n_detectors() as u64 + 64;
    let block = batch * ((frame::MAX_SAMPLED_EVENTS / per_shot) as usize / batch).clamp(1, 64);
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
