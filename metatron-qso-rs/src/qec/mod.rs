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
//! * [`CorrelatedMatchingDecoder`]: two-pass matching that reweights the
//!   graph with the correlations of decomposed errors
//!   ([`correlated_error_model`]);
//! * [`BpOsdDecoder`]: belief propagation with ordered-statistics
//!   post-processing, for models whose errors flip more than two
//!   detectors (colour codes, quantum LDPC codes);
//! * [`repetition_code_memory`] and [`surface_code_memory`]
//!   ([`surface_code_memory_with`] for the X basis): memory experiments
//!   with circuit-level noise;
//! * [`CssCode`] (triangular colour codes, bivariate bicycle codes, or any
//!   checks) and [`css_code_memory`]: memory experiments for CSS codes;
//! * [`logical_error_rate`], [`logical_error_rate_with`] and
//!   [`logical_error_rate_using`]: sample, decode and count;
//! * [`circuit_distance`] and [`graphlike_distance`]: the fewest errors
//!   that flip a logical observable undetected;
//! * [`splitting_failure_rate`], [`leading_failures`] and
//!   [`enumerated_failure_rate`]: logical failure rates far below the
//!   reach of sampling (splitting), their leading term, and exact
//!   references for small models;
//! * [`WindowDecoder`] and [`WindowStream`]: decoding in sliding windows
//!   of rounds, by matching or BP+OSD, for experiments of thousands of
//!   rounds and for detection events given round by round, and in
//!   parallel windows ([`WindowDecoder::parallel`]) on several cores;
//! * [`RepeatedCircuit`]: circuits that keep their repeated blocks (from
//!   Stim's `REPEAT` and [`surface_code_memory_repeated`] and its
//!   siblings), sampled as they run, their error models folded
//!   ([`logical_error_rate_repeated`]);
//! * [`DetectorCircuit::to_stim`]: export in Stim's text format.

pub mod bposd;
pub mod circuit;
pub mod codes;
pub mod correlated;
pub mod css;
pub mod decoder;
pub mod dem;
pub mod distance;
pub mod frame;
mod gf2;
pub mod matching;
pub mod repeat;
pub mod sparse;
pub mod splitting;
mod stim;
pub mod window;

pub use bposd::{
    BpMethod, BpOsdDecoder, BpOsdOptions, BpOsdOutcome, MAX_BPOSD_DETECTORS, OsdMethod,
};
pub use circuit::{DetectorCircuit, MAX_OBSERVABLES, Noise, QecOp};
pub use codes::{
    CircuitNoise, MAX_REPEATED_ROUNDS, MemoryBasis, repetition_code_memory,
    repetition_code_memory_repeated, surface_code_memory, surface_code_memory_repeated,
    surface_code_memory_with,
};
pub use correlated::CorrelatedMatchingDecoder;
pub use css::{
    COLOR_DIRECTIONS, CssCode, CssSchedule, MAX_CSS_QUBITS, css_code_memory,
    css_code_memory_repeated, css_code_memory_with,
};
pub use decoder::{MatchingEdge, MatchingGraph, UnionFindDecoder};
pub use dem::{
    CircuitFault, CorrelatedError, CorrelatedErrorModel, DetectorErrorModel, ErrorMechanism,
    MAX_MODEL_DETECTORS, correlated_error_model, detector_error_model, explain_errors,
};
pub use distance::{
    CircuitDistance, DistanceMethod, DistanceOptions, ExactDistanceOptions, MAX_SYMMETRY_DETECTORS,
    circuit_distance, error_model_symmetries, exact_circuit_distance,
    exact_circuit_distance_with_symmetries, graphlike_distance,
};
pub use frame::{DetectorSamples, sample_detectors};
pub use matching::{MAX_MATCHING_PAIRS, MatchingDecoder};
pub use repeat::{MAX_REPEATED_OPERATIONS, RepeatedCircuit};
pub use sparse::SparseMatchingDecoder;
pub use splitting::{
    EnumeratedFailureRate, LeadingFailures, MAX_SPLITTING_DECODES, MAX_SPLITTING_RATES,
    SplittingEstimate, SplittingOptions, SplittingRate, enumerated_failure_rate, leading_failures,
    splitting_failure_rate,
};
pub use window::{
    ParallelWindowOptions, StreamState, WindowDecoder, WindowMethod, WindowOptions, WindowStream,
};

use crate::error::{QsoError, Result};
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

impl Decoder for CorrelatedMatchingDecoder {
    fn decode(&mut self, fired: &[usize]) -> Result<u64> {
        CorrelatedMatchingDecoder::decode(self, fired)
    }
}

/// The decoders of 1.0 (see [`DecoderChoice`] for all of them).
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

impl From<DecoderKind> for DecoderChoice {
    fn from(kind: DecoderKind) -> Self {
        match kind {
            DecoderKind::UnionFind => DecoderChoice::UnionFind,
            DecoderKind::Matching => DecoderChoice::Matching,
        }
    }
}

/// The decoders [`logical_error_rate_using`] can use (more may be added).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum DecoderChoice {
    /// Weighted union-find ([`UnionFindDecoder`]): almost linear time.
    #[default]
    UnionFind,
    /// Minimum-weight perfect matching by region growth
    /// ([`SparseMatchingDecoder`]; the same weights as
    /// [`MatchingDecoder`]).
    Matching,
    /// Two-pass matching with the correlations of the decomposed model
    /// ([`CorrelatedMatchingDecoder`]); for circuits only.
    CorrelatedMatching,
    /// BP+OSD ([`BpOsdDecoder`] with its default options); for a circuit
    /// ([`DecoderChoice::build_for`]) on the model without decomposition.
    BpOsd,
    /// Matching in sliding windows of `window` rounds committing `commit`
    /// ([`WindowDecoder`]); for circuits only, whose rounds are
    /// [`DetectorCircuit::detector_layers`].
    WindowMatching { window: usize, commit: usize },
    /// BP+OSD (default options) in sliding windows ([`WindowDecoder`]) on
    /// the model without decomposition; for circuits only.
    WindowBpOsd { window: usize, commit: usize },
    /// Matching in parallel windows ([`WindowDecoder::parallel`]) that
    /// commit `commit` rounds with `buffer` on each side; for circuits only.
    ParallelWindowMatching { commit: usize, buffer: usize },
    /// BP+OSD (default options) in parallel windows on the model without
    /// decomposition; for circuits only.
    ParallelWindowBpOsd { commit: usize, buffer: usize },
}

/// What a decoder of a circuit is built from: its models and the rounds
/// of its detectors.
trait Models {
    fn model(&self, decompose: bool) -> Result<DetectorErrorModel>;
    fn correlated(&self) -> Result<CorrelatedErrorModel>;
    fn layers(&self) -> Vec<usize>;
}

impl Models for DetectorCircuit {
    fn model(&self, decompose: bool) -> Result<DetectorErrorModel> {
        detector_error_model(self, decompose)
    }

    fn correlated(&self) -> Result<CorrelatedErrorModel> {
        correlated_error_model(self)
    }

    fn layers(&self) -> Vec<usize> {
        self.detector_layers()
    }
}

impl Models for RepeatedCircuit {
    fn model(&self, decompose: bool) -> Result<DetectorErrorModel> {
        self.detector_error_model(decompose)
    }

    fn correlated(&self) -> Result<CorrelatedErrorModel> {
        self.correlated_error_model()
    }

    fn layers(&self) -> Vec<usize> {
        self.detector_layers()
    }
}

impl DecoderChoice {
    /// The decoder of a matching graph (BP+OSD on the graph's errors; it
    /// takes at most [`MAX_BPOSD_DETECTORS`]
    /// detectors). A graph does not know which edges are parts of one
    /// error, so correlated matching needs [`DecoderChoice::build_for`].
    pub fn build(self, graph: MatchingGraph) -> Result<Box<dyn Decoder + Send>> {
        Ok(match self {
            DecoderChoice::UnionFind => DecoderKind::UnionFind.build(graph),
            DecoderChoice::Matching => DecoderKind::Matching.build(graph),
            DecoderChoice::CorrelatedMatching => {
                return Err(QsoError::invalid(
                    "decoder",
                    "correlated matching needs the decomposed model of a circuit (build_for)",
                ));
            }
            DecoderChoice::BpOsd => {
                Box::new(BpOsdDecoder::new(&graph.to_dem(), BpOsdOptions::default())?)
            }
            DecoderChoice::WindowMatching { .. }
            | DecoderChoice::WindowBpOsd { .. }
            | DecoderChoice::ParallelWindowMatching { .. }
            | DecoderChoice::ParallelWindowBpOsd { .. } => {
                return Err(QsoError::invalid(
                    "decoder",
                    "windows need the rounds of a circuit's detectors (build_for)",
                ));
            }
        })
    }

    /// The decoder of a circuit's detector error model: decomposed into a
    /// matching graph for union-find and matching (also in windows), with
    /// the parts of each error for correlated matching, whole for BP+OSD
    /// (also in windows).
    pub fn build_for(self, circuit: &DetectorCircuit) -> Result<Box<dyn Decoder + Send>> {
        self.build_from(circuit)
    }

    /// [`DecoderChoice::build_for`] a circuit with repeated blocks, from
    /// its folded models.
    pub fn build_for_repeated(self, circuit: &RepeatedCircuit) -> Result<Box<dyn Decoder + Send>> {
        self.build_from(circuit)
    }

    fn build_from(self, circuit: &dyn Models) -> Result<Box<dyn Decoder + Send>> {
        let bposd = WindowMethod::BpOsd(BpOsdOptions::default());
        Ok(match self {
            DecoderChoice::BpOsd => Box::new(BpOsdDecoder::new(
                &circuit.model(false)?,
                BpOsdOptions::default(),
            )?),
            DecoderChoice::CorrelatedMatching => {
                Box::new(CorrelatedMatchingDecoder::new(&circuit.correlated()?)?)
            }
            DecoderChoice::WindowMatching { window, commit } => Box::new(WindowDecoder::new(
                &circuit.model(true)?,
                &circuit.layers(),
                WindowOptions { window, commit },
                WindowMethod::Matching,
            )?),
            DecoderChoice::WindowBpOsd { window, commit } => Box::new(WindowDecoder::new(
                &circuit.model(false)?,
                &circuit.layers(),
                WindowOptions { window, commit },
                bposd,
            )?),
            DecoderChoice::ParallelWindowMatching { commit, buffer } => {
                Box::new(WindowDecoder::parallel(
                    &circuit.model(true)?,
                    &circuit.layers(),
                    ParallelWindowOptions { commit, buffer },
                    WindowMethod::Matching,
                )?)
            }
            DecoderChoice::ParallelWindowBpOsd { commit, buffer } => {
                Box::new(WindowDecoder::parallel(
                    &circuit.model(false)?,
                    &circuit.layers(),
                    ParallelWindowOptions { commit, buffer },
                    bposd,
                )?)
            }
            _ => self.build(MatchingGraph::from_dem(&circuit.model(true)?)?)?,
        })
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
    logical_error_rate_using(circuit, shots, seed, decoder.into())
}

/// [`logical_error_rate`] with any decoder (of the model
/// [`DecoderChoice::build_for`] gives); the shots are the same for every
/// decoder.
pub fn logical_error_rate_using(
    circuit: &DetectorCircuit,
    shots: usize,
    seed: u64,
    decoder: DecoderChoice,
) -> Result<LogicalErrorRate> {
    let mut decoder = decoder.build_for(circuit)?;
    count_failures(circuit, decoder.as_mut(), shots, seed)
}

/// [`logical_error_rate_using`] for a circuit with repeated blocks: its
/// shots sampled as they run, decoded by `decoder` built from its folded
/// models ([`DecoderChoice::build_for_repeated`]).
pub fn logical_error_rate_repeated(
    circuit: &RepeatedCircuit,
    shots: usize,
    seed: u64,
    decoder: DecoderChoice,
) -> Result<LogicalErrorRate> {
    let mut decoder = decoder.build_for_repeated(circuit)?;
    count_failures(circuit, decoder.as_mut(), shots, seed)
}

/// Sample `shots` shots in blocks, decode each, and count the failures.
fn count_failures(
    circuit: &dyn frame::Walk,
    decoder: &mut (dyn Decoder + Send),
    shots: usize,
    seed: u64,
) -> Result<LogicalErrorRate> {
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
