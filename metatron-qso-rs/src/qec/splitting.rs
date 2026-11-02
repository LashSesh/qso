//! Logical failure rates below the reach of sampling.
//!
//! A detector error model is a set of independent error mechanisms with
//! probabilities `p_i`; a configuration `E` (a set of mechanisms) has
//! probability `P(E) = Π_{i∈E} p_i Π_{i∉E} (1 − p_i)` and fails when the
//! decoder's prediction from its fired detectors differs from the
//! observables it flips. The failure probability is `Σ_E P(E)·fail(E)`.
//!
//! * [`splitting_failure_rate`]: the splitting method of Bravyi and Vargo
//!   (Phys. Rev. A 88, 062308 (2013)). The failure probability is sampled
//!   directly at a start rate, then carried down a sequence of rates by
//!   ratios: at each rate, Metropolis chains sample the failing
//!   configurations with weight `P(E)`, and the ratio of the failure
//!   probabilities of consecutive rates follows from the chains of both
//!   by Bennett's acceptance ratio, `Z₁/Z₀ = C ⟨σ(x − ln C)⟩₀ / ⟨σ(ln C −
//!   x)⟩₁` with `x = ln(q₁/q₀)` and `ln C` iterated to the estimate. The
//!   decoder of each rate is the one built for that rate, so a
//!   configuration counts as failing at a rate when that rate's decoder
//!   fails on it. Standard errors come from the spread of independent
//!   chains.
//! * [`leading_failures`]: the failing configurations of the least
//!   weight `w = ⌈d/2⌉`, from the subsets of `w` errors of every
//!   undetectable logical error of at most `2w` errors: their number and
//!   their probability, the leading term `A·p^w` of the failure
//!   probability.
//! * [`enumerated_failure_rate`]: every configuration up to a weight,
//!   decoded, with an upper bound on the probability of the heavier ones
//!   (the reference for small models).

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::{
    Decoder, DecoderChoice, DetectorCircuit, DetectorErrorModel, LogicalErrorRate,
    detector_error_model,
};
use crate::error::{QsoError, Result};
use crate::parallel::map_tasks;
use crate::rng::Rng;

/// Most rates of one splitting run.
pub const MAX_SPLITTING_RATES: usize = 256;

/// Most decodes of one splitting run, or of one enumeration.
pub const MAX_SPLITTING_DECODES: u64 = 1 << 32;

/// The largest factor between consecutive rates.
const LARGEST_STEP: f64 = 4.0;

/// Options of [`splitting_failure_rate`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SplittingOptions {
    /// Shots sampled at the start rate.
    pub start_shots: usize,
    /// Independent chains at each rate (run in parallel, at least 2); the
    /// standard errors come from their spread.
    pub chains: usize,
    /// Metropolis steps of each chain at each rate, after `burn_in`.
    pub steps: usize,
    /// Steps discarded at the start of each chain at each rate: each
    /// starts from a configuration of the last rate, heavier than the new
    /// rate's typical ones, and too short a burn-in biases the ratio
    /// downwards.
    pub burn_in: usize,
    pub seed: u64,
}

impl Default for SplittingOptions {
    /// 10⁵ start shots, 16 chains of 20 000 steps after 5000 of burn-in.
    fn default() -> Self {
        SplittingOptions {
            start_shots: 100_000,
            chains: 16,
            steps: 20_000,
            burn_in: 5000,
            seed: 0,
        }
    }
}

/// One rate of a splitting run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SplittingRate {
    /// The noise strength the circuit was built with.
    pub p: f64,
    /// Failure probability at `p` over that at the previous rate.
    pub ratio: f64,
    /// Standard error of `ln ratio`.
    pub log_ratio_error: f64,
    /// Accepted moves over proposed moves of the chains at `p`.
    pub acceptance: f64,
    /// Mean and standard deviation of the number of errors of the
    /// configurations the chains sampled.
    pub mean_weight: f64,
    pub weight_spread: f64,
}

/// The failure probability at the target rate of a splitting run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SplittingEstimate {
    /// The target noise strength.
    pub p: f64,
    /// The logical failure probability per shot at `p`.
    pub rate: f64,
    /// `rate` times `log_standard_error` (the estimate is a product of
    /// ratios, so its error is relative).
    pub standard_error: f64,
    /// Standard error of `ln rate`.
    pub log_standard_error: f64,
    /// Direct sampling at the start rate.
    pub start: LogicalErrorRate,
    /// The rates after the start, the last the target.
    pub rates: Vec<SplittingRate>,
    /// Decodes of the whole run.
    pub decodes: u64,
}

/// The mechanisms of a model, the same at every rate.
struct Structure {
    n_detectors: usize,
    detectors: Vec<Vec<u32>>,
    observables: Vec<u64>,
    /// Mechanisms that share a detector with each.
    neighbours: Vec<Vec<u32>>,
}

impl Structure {
    fn new(dem: &DetectorErrorModel) -> Self {
        let detectors: Vec<Vec<u32>> = dem
            .errors
            .iter()
            .map(|e| e.detectors.iter().map(|&d| d as u32).collect())
            .collect();
        let mut by_detector: Vec<Vec<u32>> = vec![Vec::new(); dem.n_detectors];
        for (k, d) in detectors.iter().enumerate() {
            for &x in d {
                by_detector[x as usize].push(k as u32);
            }
        }
        let neighbours = detectors
            .iter()
            .enumerate()
            .map(|(k, d)| {
                let mut out: Vec<u32> = d
                    .iter()
                    .flat_map(|&x| by_detector[x as usize].iter().copied())
                    .filter(|&j| j as usize != k)
                    .collect();
                out.sort_unstable();
                out.dedup();
                out
            })
            .collect();
        Structure {
            n_detectors: dem.n_detectors,
            detectors,
            observables: dem.errors.iter().map(|e| e.observables).collect(),
            neighbours,
        }
    }

    fn matches(&self, dem: &DetectorErrorModel) -> bool {
        dem.n_detectors == self.n_detectors
            && dem.errors.len() == self.detectors.len()
            && dem
                .errors
                .iter()
                .zip(&self.detectors)
                .zip(&self.observables)
                .all(|((e, d), &o)| {
                    e.observables == o
                        && e.detectors.len() == d.len()
                        && e.detectors.iter().zip(d).all(|(&a, &b)| a == b as usize)
                })
    }

    /// The fired detectors (ascending) and the observables of `set`.
    fn syndrome(&self, set: &[u32]) -> (Vec<usize>, u64) {
        let mut all: Vec<u32> = set
            .iter()
            .flat_map(|&k| self.detectors[k as usize].iter().copied())
            .collect();
        all.sort_unstable();
        let mut fired = Vec::with_capacity(all.len());
        let mut i = 0;
        while i < all.len() {
            let mut j = i;
            while j < all.len() && all[j] == all[i] {
                j += 1;
            }
            if (j - i) % 2 == 1 {
                fired.push(all[i] as usize);
            }
            i = j;
        }
        let observables = set
            .iter()
            .fold(0u64, |o, &k| o ^ self.observables[k as usize]);
        (fired, observables)
    }

    /// Whether `decoder` fails on `set`.
    fn fails(&self, decoder: &mut (dyn Decoder + Send), set: &[u32]) -> Result<bool> {
        let (fired, observables) = self.syndrome(set);
        let predicted = if fired.is_empty() {
            0
        } else {
            decoder.decode(&fired)?
        };
        Ok(predicted != observables)
    }
}

/// The probabilities of a model at one rate.
struct Rate {
    p: f64,
    circuit: DetectorCircuit,
    /// `ln(p_i/(1 − p_i))`.
    log_odds: Vec<f64>,
    /// `Σ_i ln(1 − p_i)`.
    log_none: f64,
    probabilities: Vec<f64>,
}

impl Rate {
    fn new(p: f64, circuit: DetectorCircuit, dem: &DetectorErrorModel) -> Result<Self> {
        let probabilities: Vec<f64> = dem.errors.iter().map(|e| e.probability).collect();
        if probabilities.iter().any(|&q| !(q > 0.0 && q < 1.0)) {
            return Err(QsoError::invalid(
                "circuit",
                format!("every error of the model at {p} needs a probability in (0, 1)"),
            ));
        }
        Ok(Rate {
            p,
            circuit,
            log_odds: probabilities
                .iter()
                .map(|&q| (q / (1.0 - q)).ln())
                .collect(),
            log_none: probabilities.iter().map(|&q| (-q).ln_1p()).sum(),
            probabilities,
        })
    }

    /// `ln P(set)`.
    fn log_probability(&self, set: &[u32]) -> f64 {
        self.log_none + set.iter().map(|&k| self.log_odds[k as usize]).sum::<f64>()
    }

    fn decoder(&self, choice: DecoderChoice) -> Result<Box<dyn Decoder + Send>> {
        choice.build_for(&self.circuit)
    }
}

/// The model of `circuit` without decomposition, its structure and rate.
fn model_at(
    circuit_at: &(dyn Fn(f64) -> Result<DetectorCircuit> + Sync),
    p: f64,
    structure: Option<&Structure>,
) -> Result<(Rate, Option<Structure>)> {
    let circuit = circuit_at(p)?;
    let dem = detector_error_model(&circuit, false)?;
    let built = match structure {
        Some(s) => {
            if !s.matches(&dem) {
                return Err(QsoError::invalid(
                    "circuit_at",
                    format!(
                        "the circuit's error model at {p} has other errors than at the start \
                         rate (each error's detectors and observables must not depend on the \
                         rate)"
                    ),
                ));
            }
            None
        }
        None => Some(Structure::new(&dem)),
    };
    Ok((Rate::new(p, circuit, &dem)?, built))
}

/// The samples of one chain after its burn-in: runs of one configuration
/// and their length.
struct ChainRun {
    runs: Vec<(Vec<u32>, usize)>,
    end: Vec<u32>,
    proposed: u64,
    accepted: u64,
    decodes: u64,
}

/// A Metropolis chain at `rate` over the configurations its decoder fails
/// on, from `start` (which must fail).
#[allow(clippy::too_many_arguments)]
fn run_chain(
    s: &Structure,
    rate: &Rate,
    decoder: &mut (dyn Decoder + Send),
    start: &[u32],
    steps: usize,
    burn_in: usize,
    rng: &mut Rng,
) -> Result<ChainRun> {
    let n = s.detectors.len();
    let mut member = vec![false; n];
    let mut set: Vec<u32> = start.to_vec();
    for &k in &set {
        member[k as usize] = true;
    }
    let mut out = ChainRun {
        runs: Vec::new(),
        end: Vec::new(),
        proposed: 0,
        accepted: 0,
        decodes: 0,
    };
    let mut changed = true;
    let mut candidate: Vec<u32> = Vec::new();
    for step in 0..burn_in + steps {
        out.proposed += 1;
        // Remove an error of the set, add one outside it, or swap one in
        // the set for a neighbour, each a third of the time, with the
        // Hastings factors of the choices (so that configurations heavier
        // than the rate's typical ones lose errors quickly).
        let log_ratio;
        candidate.clear();
        let len = set.len() as f64;
        let u = rng.uniform();
        if u < 1.0 / 3.0 {
            if set.is_empty() {
                log_ratio = f64::NEG_INFINITY;
            } else {
                let i = set[rng.below(set.len() as u64) as usize] as usize;
                log_ratio = -rate.log_odds[i] + (len / (n as f64 - len + 1.0)).ln();
                candidate.extend(set.iter().copied().filter(|&k| k as usize != i));
            }
        } else if u < 2.0 / 3.0 {
            if set.len() == n {
                log_ratio = f64::NEG_INFINITY;
            } else {
                // A uniform error outside the set, by rejection.
                let mut i = rng.below(n as u64) as usize;
                while member[i] {
                    i = rng.below(n as u64) as usize;
                }
                log_ratio = rate.log_odds[i] + ((n as f64 - len) / (len + 1.0)).ln();
                candidate.extend_from_slice(&set);
                candidate.push(i as u32);
            }
        } else if set.is_empty() {
            log_ratio = f64::NEG_INFINITY;
        } else {
            let i = set[rng.below(set.len() as u64) as usize] as usize;
            let around = &s.neighbours[i];
            if around.is_empty() {
                log_ratio = f64::NEG_INFINITY;
            } else {
                let j = around[rng.below(around.len() as u64) as usize] as usize;
                if member[j] {
                    log_ratio = f64::NEG_INFINITY;
                } else {
                    log_ratio = rate.log_odds[j] - rate.log_odds[i]
                        + (around.len() as f64 / s.neighbours[j].len() as f64).ln();
                    candidate.extend(set.iter().copied().filter(|&k| k as usize != i));
                    candidate.push(j as u32);
                }
            }
        }
        if log_ratio > f64::NEG_INFINITY && (log_ratio >= 0.0 || rng.uniform() < log_ratio.exp()) {
            candidate.sort_unstable();
            out.decodes += 1;
            if s.fails(decoder, &candidate)? {
                for &k in &set {
                    member[k as usize] = false;
                }
                std::mem::swap(&mut set, &mut candidate);
                for &k in &set {
                    member[k as usize] = true;
                }
                out.accepted += 1;
                changed = true;
            }
        }
        if step >= burn_in {
            match out.runs.last_mut() {
                Some((_, count)) if !changed => *count += 1,
                _ => out.runs.push((set.clone(), 1)),
            }
            changed = false;
        }
    }
    out.end = set;
    Ok(out)
}

/// Samples seen from another rate: `(x, count, chain)`.
type Samples = Vec<(f64, usize, usize)>;

/// The samples of chains at rate `from` seen from rate `to`: per run,
/// `ln(q_to/q_from)` (−∞ where `to`'s decoder does not fail), its count
/// and its chain.
fn cross(
    s: &Structure,
    from: &Rate,
    to: &Rate,
    choice: DecoderChoice,
    runs: &[ChainRun],
) -> Result<(Samples, u64)> {
    let parts = map_tasks(runs.len(), |c| -> Result<(Samples, u64)> {
        let mut decoder = to.decoder(choice)?;
        let mut seen: HashMap<Vec<u32>, bool> = HashMap::new();
        let mut decodes = 0;
        let mut out = Vec::with_capacity(runs[c].runs.len());
        for (set, count) in &runs[c].runs {
            let fails = match seen.get(set) {
                Some(&f) => f,
                None => {
                    decodes += 1;
                    let f = s.fails(decoder.as_mut(), set)?;
                    seen.insert(set.clone(), f);
                    f
                }
            };
            let x = if fails {
                to.log_probability(set) - from.log_probability(set)
            } else {
                f64::NEG_INFINITY
            };
            out.push((x, *count, c));
        }
        Ok((out, decodes))
    });
    let mut all = Vec::new();
    let mut decodes = 0;
    for part in parts {
        let (p, d) = part?;
        all.extend(p);
        decodes += d;
    }
    Ok((all, decodes))
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Bennett's acceptance ratio between two rates.
struct Bennett {
    /// `ln(Z₁/Z₀)`.
    log_ratio: f64,
    /// Its standard error with the chains of each rate independent.
    error: f64,
    /// `⟨σ(x − ln C)⟩₀` per chain of rate 0 over its whole, and
    /// `⟨σ(x + ln C)⟩₁` per chain of rate 1 over its whole, less one: each
    /// chain's relative deviation, for the error of a product of ratios.
    forward: Vec<f64>,
    backward: Vec<f64>,
}

/// Bennett's acceptance ratio from the samples of `chains` chains of rate
/// 0 (`x = ln(q₁/q₀)`) and of rate 1 (`x = ln(q₀/q₁)`).
fn bennett(
    forward: &[(f64, usize, usize)],
    backward: &[(f64, usize, usize)],
    chains: usize,
) -> Result<Bennett> {
    let finite = |v: &[(f64, usize, usize)]| {
        let (sum, n) = v
            .iter()
            .filter(|t| t.0.is_finite())
            .fold((0.0, 0usize), |(s, n), t| (s + t.0 * t.1 as f64, n + t.1));
        (n > 0).then(|| sum / n as f64)
    };
    let (Some(f), Some(b)) = (finite(forward), finite(backward)) else {
        return Err(QsoError::invalid(
            "splitting",
            "the chains of two consecutive rates do not overlap (no configuration fails at \
             both); use more steps or chains",
        ));
    };
    let mean = |v: &[(f64, usize, usize)], shift: f64| {
        let (s, n) = v.iter().fold((0.0, 0usize), |(s, n), t| {
            (s + sigmoid(t.0 + shift) * t.1 as f64, n + t.1)
        });
        s / n as f64
    };
    let mut log_c = 0.5 * (f - b);
    for _ in 0..1000 {
        let next = log_c + mean(forward, -log_c).ln() - mean(backward, log_c).ln();
        if !next.is_finite() {
            break;
        }
        let done = (next - log_c).abs() < 1e-13 * next.abs().max(1.0);
        log_c = next;
        if done {
            break;
        }
    }
    let a = mean(forward, -log_c);
    let bb = mean(backward, log_c);
    if !(a > 0.0 && bb > 0.0) {
        return Err(QsoError::invalid(
            "splitting",
            "the chains of two consecutive rates do not overlap; use more steps or chains",
        ));
    }
    // Each chain's mean relative to the whole.
    let per_chain = |v: &[(f64, usize, usize)], shift: f64, whole: f64| {
        let mut sums = vec![(0.0, 0usize); chains];
        for t in v {
            sums[t.2].0 += sigmoid(t.0 + shift) * t.1 as f64;
            sums[t.2].1 += t.1;
        }
        sums.into_iter()
            .map(|(s, n)| {
                if n > 0 {
                    s / n as f64 / whole - 1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<f64>>()
    };
    let forward = per_chain(forward, -log_c, a);
    let backward = per_chain(backward, log_c, bb);
    let error = (mean_variance(&forward) + mean_variance(&backward)).sqrt();
    // At the fixed point; otherwise the estimate of the last iterate.
    Ok(Bennett {
        log_ratio: log_c + a.ln() - bb.ln(),
        error,
        forward,
        backward,
    })
}

/// The variance of the mean of `x` (sample variance over its count).
fn mean_variance(x: &[f64]) -> f64 {
    let k = x.len() as f64;
    if k < 2.0 {
        return f64::INFINITY;
    }
    let m = x.iter().sum::<f64>() / k;
    x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (k - 1.0) / k
}

/// Sample `shots` configurations of `rate`'s model, decode them, and keep
/// up to `keep` of the failing ones.
fn sample_start(
    s: &Structure,
    rate: &Rate,
    choice: DecoderChoice,
    shots: usize,
    keep: usize,
    seed: u64,
) -> Result<(usize, Vec<Vec<u32>>, u64)> {
    const BLOCK: usize = 1 << 14;
    let blocks = shots.div_ceil(BLOCK);
    let root = Rng::seed_from_u64(seed);
    let parts = map_tasks(blocks, |b| -> Result<(usize, Vec<Vec<u32>>, u64)> {
        let mut rng = root.fork(b as u64);
        let mut decoder = rate.decoder(choice)?;
        let k = BLOCK.min(shots - b * BLOCK);
        // Each error's shots by geometric jumps.
        let mut sets: Vec<Vec<u32>> = vec![Vec::new(); k];
        for (i, &q) in rate.probabilities.iter().enumerate() {
            let log_miss = (-q).ln_1p();
            let mut at = 0usize;
            loop {
                let u = 1.0 - rng.uniform();
                let jump = (u.ln() / log_miss).floor();
                if jump.is_nan() || jump >= (k - at) as f64 {
                    break;
                }
                at += jump as usize;
                sets[at].push(i as u32);
                at += 1;
                if at >= k {
                    break;
                }
            }
        }
        let mut failures = 0;
        let mut kept = Vec::new();
        let mut decodes = 0;
        for set in sets {
            if set.is_empty() {
                continue;
            }
            decodes += 1;
            if s.fails(decoder.as_mut(), &set)? {
                failures += 1;
                if kept.len() < keep {
                    kept.push(set);
                }
            }
        }
        Ok((failures, kept, decodes))
    });
    let mut failures = 0;
    let mut kept = Vec::new();
    let mut decodes = 0;
    for part in parts {
        let (f, k, d) = part?;
        failures += f;
        decodes += d;
        for set in k {
            if kept.len() < keep {
                kept.push(set);
            }
        }
    }
    Ok((failures, kept, decodes))
}

/// The logical failure probability per shot of the circuits
/// `circuit_at(p)` at `target`, by splitting from `start` (see the module
/// documentation): direct sampling of the circuit's detector error model
/// (without decomposition) at `start`, then chains at rates falling by a
/// factor of at most 4, each `e^{−1/σ}` below the last for a spread `σ` of
/// the number of errors its chains sampled, down to `target`. The circuit
/// must have the same errors at every rate (only their probabilities may
/// change), and `decoder` is built for each rate's circuit as
/// [`DecoderChoice::build_for`] builds it.
pub fn splitting_failure_rate(
    circuit_at: &(dyn Fn(f64) -> Result<DetectorCircuit> + Sync),
    decoder: DecoderChoice,
    start: f64,
    target: f64,
    options: &SplittingOptions,
) -> Result<SplittingEstimate> {
    if !(target > 0.0 && target <= start && start.is_finite()) {
        return Err(QsoError::invalid(
            "rates",
            "need 0 < target ≤ start (the rates are carried downwards)",
        ));
    }
    if options.start_shots == 0 || options.chains < 2 || options.steps == 0 {
        return Err(QsoError::invalid(
            "options",
            "need start shots, at least two chains and steps",
        ));
    }
    // The decodes of the start and of one rate's chains, before anything.
    let first_decodes = options.start_shots as f64
        + options.chains as f64 * (options.steps + options.burn_in) as f64;
    if first_decodes > MAX_SPLITTING_DECODES as f64 {
        return Err(QsoError::invalid(
            "options",
            format!("about {first_decodes:.1e} decodes per rate (limit {MAX_SPLITTING_DECODES})"),
        ));
    }
    let (first, structure) = model_at(circuit_at, start, None)?;
    let s = structure.expect("the start builds the structure");
    let (failures, starts, mut decodes) = sample_start(
        &s,
        &first,
        decoder,
        options.start_shots,
        options.chains,
        options.seed,
    )?;
    if failures == 0 {
        return Err(QsoError::invalid(
            "start",
            format!(
                "no failure in {} shots at {start}; raise the start rate or the shots",
                options.start_shots
            ),
        ));
    }
    let shots = options.start_shots as f64;
    let p0 = failures as f64 / shots;
    let start_rate = LogicalErrorRate {
        shots: options.start_shots,
        errors: failures,
        rate: p0,
        standard_error: (p0 * (1.0 - p0) / shots).sqrt(),
    };
    let mut log_rate = p0.ln();
    let mut log_var = (1.0 - p0) / (shots * p0);
    let mut rates = Vec::new();
    if target == start {
        return Ok(SplittingEstimate {
            p: target,
            rate: p0,
            standard_error: start_rate.standard_error,
            log_standard_error: log_var.sqrt(),
            start: start_rate,
            rates,
            decodes,
        });
    }
    let chain_seed = Rng::seed_from_u64(options.seed ^ 0x5eed_5911_7700);
    // The chains at a rate, from their start configurations.
    let run_all = |rate: &Rate, from: &[Vec<u32>], level: usize| -> Result<Vec<ChainRun>> {
        let parts = map_tasks(options.chains, |c| -> Result<ChainRun> {
            let mut rng = chain_seed.fork((level * options.chains + c) as u64);
            let mut decoder = rate.decoder(decoder)?;
            let begin = &from[c % from.len()];
            run_chain(
                &s,
                rate,
                decoder.as_mut(),
                begin,
                options.steps,
                options.burn_in,
                &mut rng,
            )
        });
        parts.into_iter().collect()
    };
    let mut current = first;
    let mut runs = run_all(&current, &starts, 0)?;
    // The deviations of the current rate's chains in the last ratio.
    let mut previous: Vec<f64> = Vec::new();
    for level in 1..=MAX_SPLITTING_RATES {
        decodes += runs.iter().map(|r| r.decodes).sum::<u64>();
        // The spread of the weights sampled sets the next step.
        let (mut sum, mut sq, mut count) = (0.0, 0.0, 0usize);
        for r in &runs {
            for (set, n) in &r.runs {
                let w = set.len() as f64;
                sum += w * *n as f64;
                sq += w * w * *n as f64;
                count += n;
            }
        }
        let mean = sum / count as f64;
        let spread = (sq / count as f64 - mean * mean).max(0.0).sqrt();
        let step = (1.0 / spread.max(1e-300)).min(LARGEST_STEP.ln());
        let p = (current.p * (-step).exp()).max(target);
        let (next, _) = model_at(circuit_at, p, Some(&s))?;
        // Chains at the next rate start from configurations that fail
        // there too.
        let mut begins = Vec::new();
        {
            let mut decoder = next.decoder(decoder)?;
            for r in &runs {
                let mut found = None;
                for set in std::iter::once(&r.end).chain(r.runs.iter().rev().map(|t| &t.0)) {
                    decodes += 1;
                    if s.fails(decoder.as_mut(), set)? {
                        found = Some(set.clone());
                        break;
                    }
                }
                if let Some(set) = found {
                    begins.push(set);
                }
            }
        }
        if begins.is_empty() {
            return Err(QsoError::invalid(
                "splitting",
                format!(
                    "no configuration the chains at {} sampled fails at {p}; use more steps or \
                     chains",
                    current.p
                ),
            ));
        }
        let next_runs = run_all(&next, &begins, level)?;
        let (forward, d1) = cross(&s, &current, &next, decoder, &runs)?;
        let (backward, d2) = cross(&s, &next, &current, decoder, &next_runs)?;
        decodes += d1 + d2;
        if decodes > MAX_SPLITTING_DECODES {
            return Err(QsoError::invalid(
                "splitting",
                format!("more than {MAX_SPLITTING_DECODES} decodes"),
            ));
        }
        let fit = bennett(&forward, &backward, options.chains)?;
        let (log_ratio, error) = (fit.log_ratio, fit.error);
        log_rate += log_ratio;
        // The chains of the current rate entered the last ratio (backward,
        // with a minus sign) and this one (forward): their deviations add
        // per chain before the spread, so that the covariance of
        // consecutive ratios counts.
        let mixed: Vec<f64> = fit
            .forward
            .iter()
            .zip(previous.iter().chain(std::iter::repeat(&0.0)))
            .map(|(f, b)| f - b)
            .collect();
        log_var += mean_variance(&mixed);
        previous = fit.backward;
        let (proposed, accepted) = next_runs
            .iter()
            .fold((0u64, 0u64), |(p, a), r| (p + r.proposed, a + r.accepted));
        let (mut sum, mut sq, mut count) = (0.0, 0.0, 0usize);
        for r in &next_runs {
            for (set, n) in &r.runs {
                let w = set.len() as f64;
                sum += w * *n as f64;
                sq += w * w * *n as f64;
                count += n;
            }
        }
        let mean = sum / count as f64;
        rates.push(SplittingRate {
            p,
            ratio: log_ratio.exp(),
            log_ratio_error: error,
            acceptance: accepted as f64 / proposed as f64,
            mean_weight: mean,
            weight_spread: (sq / count as f64 - mean * mean).max(0.0).sqrt(),
        });
        current = next;
        runs = next_runs;
        if p <= target {
            decodes += runs.iter().map(|r| r.decodes).sum::<u64>();
            log_var += mean_variance(&previous);
            let rate = log_rate.exp();
            let log_standard_error = log_var.sqrt();
            return Ok(SplittingEstimate {
                p: target,
                rate,
                standard_error: rate * log_standard_error,
                log_standard_error,
                start: start_rate,
                rates,
                decodes,
            });
        }
    }
    Err(QsoError::invalid(
        "splitting",
        format!("more than {MAX_SPLITTING_RATES} rates; start closer to the target"),
    ))
}

/// The failure probability of a model by enumeration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnumeratedFailureRate {
    /// The failure probability of the configurations of at most
    /// `max_weight` errors.
    pub rate: f64,
    /// The probability of all configurations of more errors: the exact
    /// failure probability lies in `[rate, rate + remainder]`.
    pub remainder: f64,
    /// Per weight `k = 0..=max_weight`: the failing configurations and
    /// their probability.
    pub failing: Vec<u64>,
    pub by_weight: Vec<f64>,
    /// Configurations decoded.
    pub configurations: u64,
}

/// `C(n, k)` as a float.
fn binomial(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |c, i| c * (n - i) as f64 / (i + 1) as f64)
}

/// Every configuration of at most `max_weight` errors of the circuit's
/// detector error model (without decomposition), decoded by `decoder` as
/// [`DecoderChoice::build_for`] builds it, and the probability of the
/// configurations of more errors (from the distribution of the number of
/// errors, computed exactly). Refused beyond [`MAX_SPLITTING_DECODES`]
/// configurations.
pub fn enumerated_failure_rate(
    circuit: &DetectorCircuit,
    decoder: DecoderChoice,
    max_weight: usize,
) -> Result<EnumeratedFailureRate> {
    let dem = detector_error_model(circuit, false)?;
    let s = Structure::new(&dem);
    let rate = Rate::new(0.0, circuit.clone(), &dem)?;
    let n = s.detectors.len();
    let max_weight = max_weight.min(n);
    let total: f64 = (0..=max_weight).map(|k| binomial(n, k)).sum();
    if total > MAX_SPLITTING_DECODES as f64 {
        return Err(QsoError::invalid(
            "max_weight",
            format!(
                "{total:.1e} configurations of at most {max_weight} of {n} errors (limit \
                 {MAX_SPLITTING_DECODES})"
            ),
        ));
    }
    // The probability of more than max_weight errors: the distribution of
    // the count, its last state absorbing.
    let mut dist = vec![0.0; max_weight + 2];
    dist[0] = 1.0;
    for &q in &rate.probabilities {
        for k in (0..=max_weight + 1).rev() {
            let stay = dist[k] * (1.0 - q);
            let from = if k > 0 { dist[k - 1] * q } else { 0.0 };
            dist[k] = if k == max_weight + 1 {
                dist[k] + from
            } else {
                stay + from
            };
        }
    }
    let remainder = dist[max_weight + 1];
    let (failing, by_weight, _, configurations) = enumerate(&s, &rate, decoder, max_weight)?;
    Ok(EnumeratedFailureRate {
        rate: by_weight.iter().sum(),
        remainder,
        failing,
        by_weight,
        configurations,
    })
}

/// Per weight the failing configurations, their probability and `Σ Π p_i`,
/// and the configurations decoded.
type Enumerated = (Vec<u64>, Vec<f64>, Vec<f64>, u64);

/// Every configuration of at most `max_weight` errors decoded: per
/// weight, the failing ones, their probability and `Σ Π p_i`; and the
/// configurations decoded (the empty one included).
fn enumerate(
    s: &Structure,
    rate: &Rate,
    decoder: DecoderChoice,
    max_weight: usize,
) -> Result<Enumerated> {
    let n = s.detectors.len();
    let mut failing = vec![0u64; max_weight + 1];
    let mut by_weight = vec![0.0; max_weight + 1];
    let mut terms = vec![0.0; max_weight + 1];
    if max_weight == 0 || n == 0 {
        return Ok((failing, by_weight, terms, 1));
    }
    // Configurations by their first error, in parallel, each walked in
    // lexicographic order.
    let parts = map_tasks(n, |first| {
        let mut failing = vec![0u64; max_weight + 1];
        let mut by_weight = vec![0.0; max_weight + 1];
        let mut terms = vec![0.0; max_weight + 1];
        let mut count = 0u64;
        let mut dec = rate.decoder(decoder)?;
        let mut set = vec![first as u32];
        loop {
            count += 1;
            let k = set.len();
            if s.fails(dec.as_mut(), &set)? {
                failing[k] += 1;
                by_weight[k] += rate.log_probability(&set).exp();
                terms[k] += set
                    .iter()
                    .map(|&i| rate.probabilities[i as usize])
                    .product::<f64>();
            }
            // Next set: extend, or advance the last, or backtrack.
            if k < max_weight && (*set.last().unwrap() as usize) + 1 < n {
                let next = set.last().unwrap() + 1;
                set.push(next);
                continue;
            }
            loop {
                if set.len() == 1 {
                    return Ok::<_, QsoError>((failing, by_weight, terms, count));
                }
                let last = set.pop().unwrap();
                if (last as usize) + 1 < n {
                    set.push(last + 1);
                    break;
                }
            }
        }
    });
    let mut configurations = 1;
    for part in parts {
        let (f, b, t, c) = part?;
        for k in 0..=max_weight {
            failing[k] += f[k];
            by_weight[k] += b[k];
            terms[k] += t[k];
        }
        configurations += c;
    }
    Ok((failing, by_weight, terms, configurations))
}

/// The leading term of the failure probability.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LeadingFailures {
    /// The fewest errors of an undetectable logical error.
    pub distance: usize,
    /// The fewest errors of a failing configuration: `⌈distance/2⌉` from
    /// the logical errors, or found by enumeration (with unequal weights a
    /// decoder may fail on fewer).
    pub weight: usize,
    /// Whether every configuration of at most `weight` errors was decoded
    /// (the count is then exact); otherwise only the subsets of the
    /// logical errors of at most `2⌈d/2⌉` errors, a lower bound.
    pub exhaustive: bool,
    /// Undetectable logical errors of at most `2⌈d/2⌉` errors.
    pub logicals: usize,
    /// Failing configurations of `weight` errors found.
    pub configurations: usize,
    /// `Σ Π_{i∈E} p_i` over them: the leading term `A·p^w` of the failure
    /// probability.
    pub leading_term: f64,
    /// `Σ P(E)` over them (with `Π (1 − p_j)` of the other errors): their
    /// exact share of the failure probability, a lower bound of it.
    pub probability: f64,
    /// Sets the enumeration of logical errors visited.
    pub work: u64,
}

/// Every undetectable logical error of at most `limit` errors that cannot
/// be split into two undetectable parts, by depth-first search: each set
/// grows by an error of higher index than its first that flips its
/// lowest fired detector. `None` beyond `max_work` sets.
fn logical_errors(s: &Structure, limit: usize, max_work: u64) -> Option<(Vec<Vec<u32>>, u64)> {
    let n = s.detectors.len();
    let mut by_detector: Vec<Vec<u32>> = vec![Vec::new(); s.n_detectors];
    for (k, d) in s.detectors.iter().enumerate() {
        for &x in d {
            by_detector[x as usize].push(k as u32);
        }
    }
    let widest = s.detectors.iter().map(Vec::len).max().unwrap_or(0);
    let mut found: HashSet<Vec<u32>> = HashSet::new();
    let mut work = 0u64;
    #[allow(clippy::too_many_arguments)]
    fn grow(
        s: &Structure,
        by_detector: &[Vec<u32>],
        widest: usize,
        first: u32,
        chosen: &mut Vec<u32>,
        fired: Vec<u32>,
        observables: u64,
        limit: usize,
        found: &mut HashSet<Vec<u32>>,
        work: &mut u64,
        max_work: u64,
    ) -> bool {
        *work += 1;
        if *work > max_work {
            return false;
        }
        if fired.is_empty() {
            if observables != 0 {
                let mut set = chosen.clone();
                set.sort_unstable();
                found.insert(set);
            }
            return true;
        }
        let remaining = limit - chosen.len();
        if remaining == 0 || fired.len() > remaining * widest {
            return true;
        }
        for &k in &by_detector[fired[0] as usize] {
            if k <= first || chosen.contains(&k) {
                continue;
            }
            let mut next = Vec::with_capacity(fired.len() + widest);
            let d = &s.detectors[k as usize];
            let (mut i, mut j) = (0, 0);
            while i < fired.len() || j < d.len() {
                match (fired.get(i), d.get(j)) {
                    (Some(&a), Some(&b)) if a == b => {
                        i += 1;
                        j += 1;
                    }
                    (Some(&a), Some(&b)) if a < b => {
                        next.push(a);
                        i += 1;
                    }
                    (Some(&a), None) => {
                        next.push(a);
                        i += 1;
                    }
                    (_, Some(&b)) => {
                        next.push(b);
                        j += 1;
                    }
                    (None, None) => break,
                }
            }
            chosen.push(k);
            let ok = grow(
                s,
                by_detector,
                widest,
                first,
                chosen,
                next,
                observables ^ s.observables[k as usize],
                limit,
                found,
                work,
                max_work,
            );
            chosen.pop();
            if !ok {
                return false;
            }
        }
        true
    }
    for first in 0..n as u32 {
        let mut chosen = vec![first];
        if !grow(
            s,
            &by_detector,
            widest,
            first,
            &mut chosen,
            s.detectors[first as usize].clone(),
            s.observables[first as usize],
            limit,
            &mut found,
            &mut work,
            max_work,
        ) {
            return None;
        }
    }
    let mut out: Vec<Vec<u32>> = found.into_iter().collect();
    out.sort();
    Some((out, work))
}

/// The leading term of the failure probability of the circuit's detector
/// error model (without decomposition) under `decoder`. The distance `d`
/// comes from the undetectable logical errors, enumerated by weight (at
/// most `max_work` sets in all). When every configuration of at most
/// `⌈d/2⌉` errors is at most `max_configurations`, all of them are decoded
/// and the fewest errors that fail, their number and their probability are
/// exact (`exhaustive`). Otherwise the subsets of `⌈d/2⌉` errors of every
/// logical error of at most `2⌈d/2⌉` errors are decoded: a configuration of
/// `w` errors that fails with a correction of at most `w` errors lies in
/// such a logical error, so the count is complete only for decoders whose
/// corrections never have more errors than the configuration (with
/// unequal weights, as in circuit noise, or with BP+OSD, it is a lower
/// bound, and fewer errors may fail).
pub fn leading_failures(
    circuit: &DetectorCircuit,
    decoder: DecoderChoice,
    max_work: u64,
    max_configurations: u64,
) -> Result<LeadingFailures> {
    let dem = detector_error_model(circuit, false)?;
    let s = Structure::new(&dem);
    let rate = Rate::new(0.0, circuit.clone(), &dem)?;
    let n = s.detectors.len();
    let out_of_work = || {
        QsoError::invalid(
            "max_work",
            format!("the logical errors need more than {max_work} sets to enumerate"),
        )
    };
    // The distance by increasing weight, within one budget.
    let mut distance = None;
    let mut work = 0u64;
    for limit in 1..=n {
        let (found, w) =
            logical_errors(&s, limit, max_work.saturating_sub(work)).ok_or_else(out_of_work)?;
        work += w;
        if !found.is_empty() {
            distance = Some(limit);
            break;
        }
    }
    let Some(distance) = distance else {
        return Err(QsoError::invalid(
            "circuit",
            "no undetectable logical error: the model never fails",
        ));
    };
    let half = distance.div_ceil(2);
    let (logicals, w) =
        logical_errors(&s, 2 * half, max_work.saturating_sub(work)).ok_or_else(out_of_work)?;
    work += w;
    let all: f64 = (0..=half).map(|k| binomial(n, k)).sum();
    if all <= max_configurations as f64 {
        let (failing, by_weight, terms, _) = enumerate(&s, &rate, decoder, half)?;
        let Some(weight) = (1..=half).find(|&k| failing[k] > 0) else {
            return Err(QsoError::invalid(
                "decoder",
                format!("no configuration of at most {half} errors fails"),
            ));
        };
        return Ok(LeadingFailures {
            distance,
            weight,
            exhaustive: true,
            logicals: logicals.len(),
            configurations: failing[weight] as usize,
            leading_term: terms[weight],
            probability: by_weight[weight],
            work,
        });
    }
    let weight = half;
    let mut candidates: HashSet<Vec<u32>> = HashSet::new();
    for l in &logicals {
        if l.len() < weight {
            continue;
        }
        let mut pick: Vec<usize> = (0..weight).collect();
        loop {
            candidates.insert(pick.iter().map(|&i| l[i]).collect());
            // Next combination of `weight` of the logical's errors.
            let mut i = weight;
            while i > 0 && pick[i - 1] == l.len() - weight + i - 1 {
                i -= 1;
            }
            if i == 0 {
                break;
            }
            pick[i - 1] += 1;
            for j in i..weight {
                pick[j] = pick[j - 1] + 1;
            }
        }
    }
    let candidates: Vec<Vec<u32>> = candidates.into_iter().collect();
    let tasks = crate::parallel::threads().max(1);
    let per = candidates.len().div_ceil(tasks).max(1);
    let parts = map_tasks(candidates.len().div_ceil(per), |t| {
        let mut dec = rate.decoder(decoder)?;
        let (mut count, mut term, mut probability) = (0usize, 0.0, 0.0);
        for set in &candidates[t * per..((t + 1) * per).min(candidates.len())] {
            if s.fails(dec.as_mut(), set)? {
                count += 1;
                term += set
                    .iter()
                    .map(|&k| rate.probabilities[k as usize])
                    .product::<f64>();
                probability += rate.log_probability(set).exp();
            }
        }
        Ok::<_, QsoError>((count, term, probability))
    });
    let (mut configurations, mut leading_term, mut probability) = (0, 0.0, 0.0);
    for part in parts {
        let (c, t, q) = part?;
        configurations += c;
        leading_term += t;
        probability += q;
    }
    Ok(LeadingFailures {
        distance,
        weight,
        exhaustive: false,
        logicals: logicals.len(),
        configurations,
        leading_term,
        probability,
        work,
    })
}

#[cfg(test)]
mod tests;
