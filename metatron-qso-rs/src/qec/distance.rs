//! Circuit distance: the fewest errors of a detector error model that flip
//! a logical observable without firing a detector.
//!
//! * Graph-like errors (each flips at most two detectors) are edges of a
//!   graph with a boundary node. An undetectable set of them is a set of
//!   edges with even degree at every detector, and it flips observable
//!   `i` when it holds an odd number of edges that flip `i`. The lightest
//!   such set is a cycle, found exactly by breadth-first search on the
//!   graph doubled by parity: the shortest walk from `(s, even)` to
//!   `(s, odd)` over the start nodes `s` of the edges that flip `i`
//!   ([`graphlike_distance`], Stim's `shortest_graphlike_error`).
//! * Errors that flip more detectors make the problem that of the minimum
//!   distance of a linear code (NP-hard). [`circuit_distance`] gives
//!   bounds: an upper bound from the lightest logical error it finds (among
//!   the graph-like errors, and by BP+OSD on the checks with the parity of
//!   some observables as an extra check that must fire), and a lower bound from
//!   an exhaustive search, by increasing weight, of the sets of errors
//!   that cancel every detector, up to a weight and a work limit. When the
//!   bounds meet, the distance is exact.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::bposd::{BpOsdDecoder, BpOsdOptions, MAX_BPOSD_DETECTORS};
use super::dem::{DetectorErrorModel, ErrorMechanism};
use crate::error::{QsoError, Result};
use crate::parallel::{map_tasks, threads};
use crate::rng::Rng;

mod zimmermann;
use zimmermann::{Enumeration, Zimmermann};

/// Most BP+OSD searches per observable.
pub const MAX_BPOSD_TRIALS: usize = 1 << 16;

/// Options of [`circuit_distance`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DistanceOptions {
    /// The exhaustive search stops after proving that no logical error of
    /// this weight exists.
    pub max_weight: usize,
    /// Most sets of errors the exhaustive search visits, as the search
    /// over first errors in index order counts them (some 10⁷ a second
    /// per thread). The threads search ahead of that order, which costs
    /// time but changes neither the bounds nor the witness.
    pub max_work: u64,
    /// BP+OSD searches per observable (at most [`MAX_BPOSD_TRIALS`]):
    /// first each observable alone with equal priors, then random sets of
    /// observables with randomly perturbed priors. Skipped for models with
    /// more than `MAX_BPOSD_DETECTORS − 1` detectors.
    pub bposd_trials: usize,
    /// Seed of the perturbations.
    pub seed: u64,
}

impl Default for DistanceOptions {
    /// Weights up to 12, 2²⁴ sets, 4 BP+OSD searches per observable.
    fn default() -> Self {
        DistanceOptions {
            max_weight: 12,
            max_work: 1 << 24,
            bposd_trials: 4,
            seed: 0,
        }
    }
}

/// Bounds on the circuit distance of a model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CircuitDistance {
    /// No undetectable logical error has fewer errors (`usize::MAX`: no
    /// undetectable logical error exists).
    pub lower_bound: usize,
    /// The lightest undetectable logical error found: indices into the
    /// model's errors, ascending.
    pub witness: Option<Vec<usize>>,
    /// Observables the witness flips.
    pub observables: u64,
}

impl CircuitDistance {
    /// Errors in the witness.
    pub fn upper_bound(&self) -> Option<usize> {
        self.witness.as_ref().map(Vec::len)
    }

    /// The distance, when the bounds meet.
    pub fn distance(&self) -> Option<usize> {
        self.upper_bound().filter(|&u| u == self.lower_bound)
    }

    fn exact(witness: Vec<usize>, observables: u64) -> Self {
        CircuitDistance {
            lower_bound: witness.len(),
            witness: Some(witness),
            observables,
        }
    }

    fn none() -> Self {
        CircuitDistance {
            lower_bound: usize::MAX,
            witness: None,
            observables: 0,
        }
    }
}

/// An error that can happen, deduplicated: detectors, observables, its
/// first index in the model.
struct Error {
    detectors: Vec<usize>,
    observables: u64,
    index: usize,
}

/// The errors of positive probability that flip something, each effect
/// once (the first error with it), after checking the model.
fn errors(dem: &DetectorErrorModel) -> Result<Vec<Error>> {
    let n = dem.n_detectors;
    let mut seen: HashMap<(&[usize], u64), ()> = HashMap::new();
    let mut out = Vec::new();
    for (k, e) in dem.errors.iter().enumerate() {
        let p = e.probability;
        if !(0.0..=1.0).contains(&p) {
            return Err(QsoError::invalid(
                "dem",
                format!("error {k} has probability {p}"),
            ));
        }
        if let Some(&d) = e.detectors.iter().find(|&&d| d >= n) {
            return Err(QsoError::invalid(
                "dem",
                format!("error {k} flips detector {d} of {n}"),
            ));
        }
        if e.detectors.windows(2).any(|w| w[0] >= w[1]) {
            return Err(QsoError::invalid(
                "dem",
                format!("the detectors of error {k} must be ascending and distinct"),
            ));
        }
        if p == 0.0 || (e.detectors.is_empty() && e.observables == 0) {
            continue;
        }
        if seen
            .insert((e.detectors.as_slice(), e.observables), ())
            .is_none()
        {
            out.push(Error {
                detectors: e.detectors.clone(),
                observables: e.observables,
                index: k,
            });
        }
    }
    Ok(out)
}

/// Original indices of a set of errors, ascending, and the observables
/// they flip.
fn witness(errors: &[Error], set: &[usize]) -> (Vec<usize>, u64) {
    let mut w: Vec<usize> = set.iter().map(|&k| errors[k].index).collect();
    w.sort_unstable();
    let obs = set.iter().fold(0, |o, &k| o ^ errors[k].observables);
    (w, obs)
}

/// An error flipping observables but no detector.
fn undetectable(errors: &[Error]) -> Option<usize> {
    errors
        .iter()
        .position(|e| e.detectors.is_empty() && e.observables != 0)
}

/// The lightest undetectable logical error made of the model's graph-like
/// errors (those flipping one or two detectors; the others are ignored, as
/// by Stim's `shortest_graphlike_error`). Exact: `lower_bound` is the
/// witness's weight, or `usize::MAX` if these errors flip no observable
/// undetectably.
pub fn graphlike_distance(dem: &DetectorErrorModel) -> Result<CircuitDistance> {
    let errors = errors(dem)?;
    if let Some(k) = undetectable(&errors) {
        let (w, obs) = witness(&errors, &[k]);
        return Ok(CircuitDistance::exact(w, obs));
    }
    Ok(match shortest_cycle(dem.n_detectors, &errors) {
        Some(set) => {
            let (w, obs) = witness(&errors, &set);
            CircuitDistance::exact(w, obs)
        }
        None => CircuitDistance::none(),
    })
}

/// The shortest cycle of graph-like errors with odd parity in some
/// observable (indices into `errors`).
fn shortest_cycle(n: usize, errors: &[Error]) -> Option<Vec<usize>> {
    // Nodes 0..n are detectors, n the boundary.
    let mut adjacency: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n + 1];
    let mut ends = Vec::new();
    for (k, e) in errors.iter().enumerate() {
        let (a, b) = match e.detectors.as_slice() {
            [a] => (*a, n),
            [a, b] => (*a, *b),
            _ => continue,
        };
        adjacency[a].push((b, k));
        adjacency[b].push((a, k));
        ends.push((a, k));
    }
    let mut bits = 0u64;
    for &(_, k) in &ends {
        bits |= errors[k].observables;
    }
    let mut best: Option<Vec<usize>> = None;
    // State 2v + parity.
    let mut dist = vec![usize::MAX; 2 * (n + 1)];
    let mut parent: Vec<(usize, usize)> = vec![(0, 0); 2 * (n + 1)];
    let mut touched = Vec::new();
    let mut queue = VecDeque::new();
    for bit in (0..64).filter(|&b| bits >> b & 1 == 1) {
        let mut starts: Vec<usize> = ends
            .iter()
            .filter(|&&(_, k)| errors[k].observables >> bit & 1 == 1)
            .map(|&(a, _)| a)
            .collect();
        starts.sort_unstable();
        starts.dedup();
        for s in starts {
            let limit = best.as_ref().map_or(usize::MAX, Vec::len);
            let start = 2 * s;
            let goal = 2 * s + 1;
            dist[start] = 0;
            touched.push(start);
            queue.push_back(start);
            while let Some(x) = queue.pop_front() {
                if dist[x] + 1 >= limit {
                    break;
                }
                let (v, parity) = (x / 2, x % 2);
                for &(w, k) in &adjacency[v] {
                    let y = 2 * w + (parity ^ (errors[k].observables >> bit & 1) as usize);
                    if dist[y] != usize::MAX {
                        continue;
                    }
                    dist[y] = dist[x] + 1;
                    parent[y] = (x, k);
                    touched.push(y);
                    queue.push_back(y);
                }
                if dist[goal] != usize::MAX {
                    break;
                }
            }
            if dist[goal] != usize::MAX && dist[goal] < limit {
                let mut set = Vec::new();
                let mut y = goal;
                while y != start {
                    let (x, k) = parent[y];
                    set.push(k);
                    y = x;
                }
                // A shortest odd closed walk is a cycle; cancel repeats
                // all the same.
                set.sort_unstable();
                let mut reduced: Vec<usize> = Vec::with_capacity(set.len());
                for k in set {
                    if reduced.last() == Some(&k) {
                        reduced.pop();
                    } else {
                        reduced.push(k);
                    }
                }
                best = Some(reduced);
            }
            for &x in &touched {
                dist[x] = usize::MAX;
            }
            touched.clear();
            queue.clear();
        }
    }
    best
}

/// Keep `set` when it is lighter than the best so far.
fn consider(set: Vec<usize>, best: &mut Option<Vec<usize>>) {
    if best.as_ref().is_none_or(|b| set.len() < b.len()) {
        *best = Some(set);
    }
}

/// The lightest undetectable logical error among `cycle` (the graph-like
/// one) and those found by `trials` BP+OSD searches per observable (fewer
/// if the deadline passes).
fn upper_bound(
    n: usize,
    errors: &[Error],
    cycle: Option<Vec<usize>>,
    trials: usize,
    seed: u64,
    deadline: Option<Instant>,
) -> Result<Option<Vec<usize>>> {
    let mut best = cycle;
    let bits = errors.iter().fold(0u64, |b, e| b | e.observables);
    let singles: Vec<u64> = (0..64)
        .filter(|&b| bits >> b & 1 == 1)
        .map(|b| 1u64 << b)
        .collect();
    let mut rng = Rng::seed_from_u64(seed);
    if trials > MAX_BPOSD_TRIALS {
        return Err(QsoError::invalid(
            "bposd_trials",
            format!("at most {MAX_BPOSD_TRIALS} BP+OSD searches per observable"),
        ));
    }
    for trial in 0..trials * singles.len() {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            break;
        }
        // First every observable alone with equal priors, then random sets
        // of observables (a light logical error often flips several) with
        // perturbed priors.
        let (target, perturbed) = match singles.get(trial) {
            Some(&single) => (single, false),
            None => loop {
                let t = rng.next_u64() & bits;
                if t != 0 {
                    break (t, true);
                }
            },
        };
        if let Some(set) = bposd_logical(n, errors, target, perturbed, &mut rng)? {
            consider(set, &mut best);
        }
    }
    Ok(best)
}

/// Bounds on the circuit distance of a detector error model: the fewest
/// errors that flip some observable and no detector. Exact for graph-like
/// models; otherwise see the [module documentation](self) and
/// [`DistanceOptions`].
pub fn circuit_distance(
    dem: &DetectorErrorModel,
    options: &DistanceOptions,
) -> Result<CircuitDistance> {
    let errors = errors(dem)?;
    if let Some(k) = undetectable(&errors) {
        let (w, obs) = witness(&errors, &[k]);
        return Ok(CircuitDistance::exact(w, obs));
    }
    let n = dem.n_detectors;
    let cycle = shortest_cycle(n, &errors);
    if errors.iter().all(|e| e.detectors.len() <= 2) {
        return Ok(match cycle {
            Some(set) => {
                let (w, obs) = witness(&errors, &set);
                CircuitDistance::exact(w, obs)
            }
            None => CircuitDistance::none(),
        });
    }
    let mut best = upper_bound(n, &errors, cycle, options.bposd_trials, options.seed, None)?;
    // Exhaustive search below the best found.
    let search = Search::new(n, &errors);
    let ceiling = best
        .as_ref()
        .map_or(options.max_weight, |b| options.max_weight.min(b.len() - 1));
    let (mut lower, mut used, mut exhausted) = (2, 0u64, false);
    for w in 2..=ceiling {
        let (outcome, spent) = search.run(w, options.max_work - used, None);
        used += spent;
        match outcome {
            Outcome::Found(set) => {
                consider(set, &mut best);
                lower = w;
                break;
            }
            Outcome::NotFound { exhausted: e } => {
                lower = w + 1;
                exhausted = e;
                if e {
                    break;
                }
            }
            Outcome::OutOfWork => break,
        }
    }
    // Nothing found: the search ran out of errors to add, or linear
    // algebra decides whether anything exists.
    let proven_none = best.is_none() && (exhausted || logical_exists(n, &errors) == Some(false));
    Ok(match best {
        Some(set) => {
            let (w, obs) = witness(&errors, &set);
            CircuitDistance {
                lower_bound: lower.min(w.len()),
                witness: Some(w),
                observables: obs,
            }
        }
        None if proven_none => CircuitDistance::none(),
        None => CircuitDistance {
            lower_bound: lower,
            witness: None,
            observables: 0,
        },
    })
}

/// How [`exact_circuit_distance`] raises its lower bound.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DistanceMethod {
    /// Each step by whichever of the two below is estimated to cost less.
    #[default]
    Auto,
    /// The search over sets of errors that clear the lowest fired detector
    /// (as in [`circuit_distance`]): quick for sparse models, where every
    /// error flips few detectors and every detector few errors.
    Search,
    /// The Brouwer–Zimmermann enumeration: quick when the undetectable
    /// errors form a code of small dimension, such as classical codes.
    BrouwerZimmermann,
}

/// Options of [`exact_circuit_distance`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct ExactDistanceOptions {
    /// Seconds after which the bounds reached so far are returned
    /// (infinite: none).
    pub max_seconds: f64,
    /// How the lower bound rises (see [`DistanceMethod`]).
    pub method: DistanceMethod,
    /// BP+OSD searches per observable for a first upper bound, as in
    /// [`DistanceOptions`] (fewer if `max_seconds` passes first).
    pub bposd_trials: usize,
    /// Seed of the BP+OSD searches.
    pub seed: u64,
}

impl Default for ExactDistanceOptions {
    /// A minute, [`DistanceMethod::Auto`], 4 BP+OSD searches per
    /// observable.
    fn default() -> Self {
        ExactDistanceOptions {
            max_seconds: 60.0,
            method: DistanceMethod::Auto,
            bposd_trials: 4,
            seed: 0,
        }
    }
}

impl ExactDistanceOptions {
    pub fn with_max_seconds(mut self, seconds: f64) -> Self {
        self.max_seconds = seconds;
        self
    }

    pub fn with_method(mut self, method: DistanceMethod) -> Self {
        self.method = method;
        self
    }

    pub fn with_bposd_trials(mut self, trials: usize) -> Self {
        self.bposd_trials = trials;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Nanoseconds per set the search visits, and per word operation of the
/// Brouwer–Zimmermann enumeration (measured on one core), for choosing
/// between them.
const SEARCH_NANOS: f64 = 40.0;
const ENUMERATION_NANOS: f64 = 0.5;

/// The circuit distance of a detector error model, exactly when the time
/// allows: the fewest errors that flip some observable and no detector,
/// with the lightest such set as its witness.
///
/// Graph-like models are solved by shortest paths. Otherwise an upper
/// bound comes from the graph-like errors and BP+OSD searches, and the
/// lower bound rises step by step, by the exhaustive search of
/// [`circuit_distance`] (no logical error of the current bound's weight)
/// or by levels of the Brouwer–Zimmermann enumeration, whichever
/// [`ExactDistanceOptions::method`] picks (by default each step the one
/// estimated to cost less), until the bounds meet. After
/// [`ExactDistanceOptions::max_seconds`] the bounds reached so far are
/// returned; [`CircuitDistance::distance`] tells whether they met.
pub fn exact_circuit_distance(
    dem: &DetectorErrorModel,
    options: &ExactDistanceOptions,
) -> Result<CircuitDistance> {
    if options.max_seconds.is_nan() || options.max_seconds < 0.0 {
        return Err(QsoError::invalid(
            "max_seconds",
            format!("{} is not a time limit", options.max_seconds),
        ));
    }
    // No clock without a limit (wasm32-unknown-unknown has none).
    let deadline = (options.max_seconds.is_finite())
        .then(|| Instant::now() + Duration::from_secs_f64(options.max_seconds.min(1e9)));
    let errors = errors(dem)?;
    if let Some(k) = undetectable(&errors) {
        let (w, obs) = witness(&errors, &[k]);
        return Ok(CircuitDistance::exact(w, obs));
    }
    let n = dem.n_detectors;
    let cycle = shortest_cycle(n, &errors);
    if errors.iter().all(|e| e.detectors.len() <= 2) {
        return Ok(match cycle {
            Some(set) => {
                let (w, obs) = witness(&errors, &set);
                CircuitDistance::exact(w, obs)
            }
            None => CircuitDistance::none(),
        });
    }
    let mut best = upper_bound(
        n,
        &errors,
        cycle,
        options.bposd_trials,
        options.seed,
        deadline,
    )?;
    // Nothing flips an observable, or linear algebra rules out a logical
    // error (up to `MAX_EXISTENCE_DETECTORS` detectors).
    if best.is_none()
        && (errors.iter().all(|e| e.observables == 0) || logical_exists(n, &errors) == Some(false))
    {
        return Ok(CircuitDistance::none());
    }
    let budget = Budget::new(u64::MAX, deadline);
    let search = Search::new(n, &errors);
    // The weights of the search's last two steps and the sets each
    // visited, for the cost of its next step.
    let mut history: Vec<(usize, u64)> = Vec::new();
    let branching = {
        let entries: usize = errors.iter().map(|e| e.detectors.len()).sum();
        (entries as f64 / n.max(1) as f64).max(2.0)
    };
    let mut enumeration: Option<Enumeration> = None;
    let mut levels = 0;
    // A single error is detectable or harmless (checked above).
    let mut lower = 2;
    loop {
        let upper = best.as_ref().map_or(usize::MAX, Vec::len);
        if lower > errors.len() {
            // Every logical error is a set of distinct errors, and none is
            // lighter than `lower`: none exists.
            lower = usize::MAX;
        }
        if lower >= upper || budget.stopped() {
            break;
        }
        // The search's next step, at weight `lower`, extrapolated from its
        // last by the growth per weight (the enumeration may have raised
        // `lower` by several since).
        let search_cost = match history.as_slice() {
            [] => 0.0,
            [(w, x)] => *x as f64 * branching.powi((lower - w) as i32),
            [.., (w0, x0), (w1, x1)] => {
                let growth = (*x1 as f64 / (*x0).max(1) as f64)
                    .powf(1.0 / (w1 - w0) as f64)
                    .max(1.0);
                *x1 as f64 * growth.powi((lower - w1) as i32)
            }
        } * SEARCH_NANOS;
        // The enumeration's levels until its bound passes `lower`.
        let enumeration_cost = |z: &Zimmermann| -> f64 {
            let mut cost = 0.0;
            let mut w = levels + 1;
            loop {
                if w > z.dimension() {
                    return f64::INFINITY;
                }
                cost += z.cost(w) * ENUMERATION_NANOS;
                if z.bound(w) > lower {
                    return cost;
                }
                w += 1;
            }
        };
        let use_enumeration = match options.method {
            DistanceMethod::Search => false,
            DistanceMethod::BrouwerZimmermann => true,
            DistanceMethod::Auto => {
                if enumeration.is_none() && search_cost > 0.0 {
                    // Build the matrices only if, at their best (disjoint
                    // information sets of full rank), the levels could
                    // cost less than the search.
                    let cols = errors.len();
                    let k = cols.saturating_sub(n).max(1);
                    let optimistic = optimistic_enumeration(cols, k, lower);
                    if optimistic * ENUMERATION_NANOS < search_cost {
                        enumeration = Some(Zimmermann::build(n, &errors, &budget));
                    }
                }
                match &enumeration {
                    Some(Enumeration::Ready(z)) => enumeration_cost(z) < search_cost,
                    Some(Enumeration::NoLogicalError) => true,
                    _ => false,
                }
            }
        };
        if use_enumeration {
            let built = enumeration.get_or_insert_with(|| Zimmermann::build(n, &errors, &budget));
            let z = match built {
                Enumeration::Ready(z) => &*z,
                Enumeration::NoLogicalError => {
                    lower = usize::MAX;
                    break;
                }
                Enumeration::Stopped => break,
                Enumeration::TooLarge => {
                    return Err(QsoError::invalid(
                        "method",
                        "the detectors or the undetectable errors are too many for the \
                         Brouwer–Zimmermann enumeration",
                    ));
                }
            };
            let w = levels + 1;
            if w > z.dimension() {
                break;
            }
            let level = z.level(w, upper, &budget);
            if let Some(set) = level.found {
                consider(set, &mut best);
            }
            if !level.complete {
                break;
            }
            levels = w;
            lower = lower.max(z.bound(w));
        } else {
            let weight = lower;
            let (outcome, spent) = search.run(weight, u64::MAX, deadline);
            match outcome {
                Outcome::Found(set) => {
                    // Nothing lighter exists: this is the distance.
                    consider(set, &mut best);
                    break;
                }
                // The search ran out of errors to add: nothing exists.
                Outcome::NotFound { exhausted: true } => lower = usize::MAX,
                Outcome::NotFound { exhausted: false } => lower += 1,
                Outcome::OutOfWork => break,
            }
            history.push((weight, spent));
            if history.len() > 2 {
                history.remove(0);
            }
        }
    }
    Ok(match best {
        Some(set) => {
            let (w, obs) = witness(&errors, &set);
            CircuitDistance {
                lower_bound: lower.min(w.len()),
                witness: Some(w),
                observables: obs,
            }
        }
        None => CircuitDistance {
            lower_bound: lower,
            witness: None,
            observables: 0,
        },
    })
}

/// Word operations of the enumeration's levels until its bound passes
/// `lower`, if the code of dimension `k` on `cols` columns had disjoint
/// information sets of full rank (a lower bound on the real cost).
fn optimistic_enumeration(cols: usize, k: usize, lower: usize) -> f64 {
    let full = cols / k;
    let rest = cols % k;
    let words = cols.div_ceil(64) as f64 + 4.0;
    let mut cost = 0.0;
    for w in 1..=k {
        let sets = full + usize::from(w + 1 > k - rest);
        let combos = (0..w).fold(1.0, |acc, i| acc * (k - i) as f64 / (i + 1) as f64);
        cost += sets as f64 * combos * words;
        let bound = full * (w + 1) + (w + 1).saturating_sub(k - rest);
        if bound > lower {
            return cost;
        }
    }
    f64::INFINITY
}

/// Most detectors for which [`circuit_distance`] decides by linear algebra
/// that no undetectable logical error exists.
pub const MAX_EXISTENCE_DETECTORS: usize = 1 << 12;

/// Whether some set of errors flips an observable and no detector
/// (`None` for more than [`MAX_EXISTENCE_DETECTORS`] detectors). The error
/// columns, detectors as the low bits and observables above, go into a
/// reduced GF(2) basis whose pivots are lowest bits: a combination without
/// detector bits is a combination of the members whose pivot is an
/// observable bit.
fn logical_exists(n: usize, errors: &[Error]) -> Option<bool> {
    if n > MAX_EXISTENCE_DETECTORS {
        return None;
    }
    let len = n + 64;
    let mut basis = super::gf2::Basis::new(len, false, 0);
    for e in errors {
        let mut ones = e.detectors.clone();
        ones.extend(
            (0..64)
                .filter(|b| e.observables >> b & 1 == 1)
                .map(|b| n + b),
        );
        if let Some(k) = basis.insert(&super::gf2::from_ones(len, &ones))
            && basis.pivot(k) >= n
        {
            return Some(true);
        }
    }
    Some(false)
}

/// A light undetectable error flipping an odd number of the observables in
/// `target`, by BP+OSD on the checks with their parity as an extra check
/// that fires; with `perturbed`, the priors vary randomly by a factor of up
/// to e³ either way.
fn bposd_logical(
    n: usize,
    errors: &[Error],
    target: u64,
    perturbed: bool,
    rng: &mut Rng,
) -> Result<Option<Vec<usize>>> {
    if n + 1 > MAX_BPOSD_DETECTORS {
        return Ok(None);
    }
    let model = DetectorErrorModel {
        n_detectors: n + 1,
        n_observables: 0,
        errors: errors
            .iter()
            .map(|e| {
                let mut detectors = e.detectors.clone();
                if (e.observables & target).count_ones() % 2 == 1 {
                    detectors.push(n);
                }
                let jitter = if perturbed {
                    rng.uniform_range(-3.0, 3.0).exp()
                } else {
                    1.0
                };
                ErrorMechanism {
                    probability: 0.01 * jitter,
                    detectors,
                    observables: 0,
                }
            })
            .collect(),
    };
    let mut decoder = BpOsdDecoder::new(
        &model,
        BpOsdOptions {
            osd_order: 10,
            ..BpOsdOptions::default()
        },
    )?;
    // An error: no set of errors flips the observable undetectably.
    let Ok(outcome) = decoder.decode_detailed(&[n]) else {
        return Ok(None);
    };
    let set = outcome.errors;
    Ok(is_logical(n, errors, &set).then_some(set))
}

/// Whether a set of errors fires no detector and flips some observable.
fn is_logical(n: usize, errors: &[Error], set: &[usize]) -> bool {
    let mut fired = vec![false; n];
    let mut obs = 0;
    for &k in set {
        for &d in &errors[k].detectors {
            fired[d] ^= true;
        }
        obs ^= errors[k].observables;
    }
    obs != 0 && !fired.iter().any(|&f| f)
}

enum Outcome {
    Found(Vec<usize>),
    /// No logical error of the weight searched; `exhausted` when the
    /// weight cut no branch, so that none exists at any weight.
    NotFound {
        exhausted: bool,
    },
    OutOfWork,
}

/// What the search met from one first error.
enum Tally {
    /// No logical error: the sets visited, and whether the weight cut a
    /// branch.
    Done { work: u64, cut: bool },
    /// A logical error, at this many sets.
    Found { at: u64, set: Vec<usize> },
}

/// Limits shared by the threads of a search: units of work (sets
/// visited, or sums enumerated) and a deadline.
pub(super) struct Budget {
    work: AtomicU64,
    max_work: u64,
    deadline: Option<Instant>,
    stopped: AtomicBool,
}

impl Budget {
    pub(super) fn new(max_work: u64, deadline: Option<Instant>) -> Self {
        Budget {
            work: AtomicU64::new(0),
            max_work,
            deadline,
            stopped: AtomicBool::new(false),
        }
    }

    /// Count `n` more units: false once over the work or past the
    /// deadline, and from then on.
    pub(super) fn spend(&self, n: u64) -> bool {
        if self.stopped.load(Ordering::Relaxed) {
            return false;
        }
        let used = self.work.fetch_add(n, Ordering::Relaxed).saturating_add(n);
        if used > self.max_work || self.deadline.is_some_and(|d| Instant::now() >= d) {
            self.stopped.store(true, Ordering::Relaxed);
            return false;
        }
        true
    }

    pub(super) fn stopped(&self) -> bool {
        self.stopped.load(Ordering::Relaxed) || self.deadline.is_some_and(|d| Instant::now() >= d)
    }
}

/// Exhaustive search for undetectable logical errors of at most a given
/// weight.
///
/// Every such set of the least weight has a first (lowest-index) error;
/// starting from each error in turn, the search adds errors of higher
/// index that flip the lowest detector still fired (one of them must),
/// until no detector fires. A set whose prefix already fires no detector
/// and flips no observable is never the lightest (dropping the prefix
/// leaves a lighter logical error), so the search stops there.
///
/// The first errors are shared out between threads, which search ahead of
/// index order. The answer is that of the search over first errors in
/// index order: the set found from the lowest first error, with the sets
/// counted up to it, so neither the answer nor the count depends on the
/// threads, the work limit included (the deadline aside).
struct Search<'a> {
    errors: &'a [Error],
    detectors: Vec<Vec<u32>>,
    by_detector: Vec<Vec<u32>>,
    by_effect: HashMap<Vec<u32>, Vec<u32>>,
    widest: usize,
}

impl<'a> Search<'a> {
    fn new(n: usize, errors: &'a [Error]) -> Self {
        let detectors: Vec<Vec<u32>> = errors
            .iter()
            .map(|e| e.detectors.iter().map(|&d| d as u32).collect())
            .collect();
        let mut by_detector = vec![Vec::new(); n];
        let mut by_effect: HashMap<Vec<u32>, Vec<u32>> = HashMap::new();
        for (k, d) in detectors.iter().enumerate() {
            for &x in d {
                by_detector[x as usize].push(k as u32);
            }
            by_effect.entry(d.clone()).or_default().push(k as u32);
        }
        Search {
            errors,
            widest: detectors.iter().map(Vec::len).max().unwrap_or(0),
            detectors,
            by_detector,
            by_effect,
        }
    }

    /// Whether some undetectable logical error has at most `weight` errors
    /// within `allowance` sets visited (and the deadline), and the sets
    /// that count: those the search in index order visits up to its
    /// answer.
    fn run(&self, weight: usize, allowance: u64, deadline: Option<Instant>) -> (Outcome, u64) {
        let n = self.errors.len();
        if weight == 0 || n == 0 {
            return (Outcome::NotFound { exhausted: n == 0 }, 0);
        }
        // The threads stop soon after visiting `allowance` sets together.
        let budget = Budget::new(allowance, deadline);
        let per = n.div_ceil(16 * threads()).max(1);
        // The lowest first error from which a set was found: later first
        // errors need not be searched.
        let lowest = AtomicUsize::new(usize::MAX);
        let tallies: Vec<Vec<Tally>> = map_tasks(n.div_ceil(per), |t| {
            let mut walker = Walker::new(self, weight);
            let mut out = Vec::new();
            for first in t * per..((t + 1) * per).min(n) {
                if first > lowest.load(Ordering::Relaxed) {
                    break;
                }
                match walker.start(first as u32, weight, &budget) {
                    Some(true) => {
                        lowest.fetch_min(first, Ordering::Relaxed);
                        out.push(Tally::Found {
                            at: walker.nodes,
                            set: walker.set(),
                        });
                        break;
                    }
                    Some(false) => out.push(Tally::Done {
                        work: walker.nodes,
                        cut: walker.cut,
                    }),
                    None => break,
                }
                if !budget.spend(std::mem::take(&mut walker.pending)) {
                    break;
                }
            }
            out
        });
        // The account of the search in index order, first error by first
        // error, up to the first one the threads did not finish.
        let (mut spent, mut cut) = (0u64, false);
        for (t, tally) in tallies.iter().enumerate() {
            for entry in tally {
                match entry {
                    Tally::Found { at, set } => {
                        return if spent.saturating_add(*at) <= allowance {
                            (Outcome::Found(set.clone()), spent + at)
                        } else {
                            (Outcome::OutOfWork, allowance)
                        };
                    }
                    Tally::Done { work, cut: c } => {
                        spent = spent.saturating_add(*work);
                        cut |= c;
                        if spent > allowance {
                            return (Outcome::OutOfWork, allowance);
                        }
                    }
                }
            }
            let next = t * per + tally.len();
            if next < ((t + 1) * per).min(n) {
                return self.sequential(next, weight, allowance, deadline, spent, cut);
            }
        }
        (Outcome::NotFound { exhausted: !cut }, spent)
    }

    /// The search in index order from first error `from` on, on one
    /// thread, after `spent` sets (`cut`: whether the weight cut a branch
    /// before).
    fn sequential(
        &self,
        from: usize,
        weight: usize,
        allowance: u64,
        deadline: Option<Instant>,
        mut spent: u64,
        mut cut: bool,
    ) -> (Outcome, u64) {
        let budget = Budget::new(allowance - spent, deadline);
        let mut walker = Walker::new(self, weight);
        for first in from..self.errors.len() {
            match walker.start(first as u32, weight, &budget) {
                Some(true) => {
                    return if spent.saturating_add(walker.nodes) <= allowance {
                        (Outcome::Found(walker.set()), spent + walker.nodes)
                    } else {
                        (Outcome::OutOfWork, allowance)
                    };
                }
                Some(false) => {
                    spent = spent.saturating_add(walker.nodes);
                    cut |= walker.cut;
                    if spent > allowance {
                        return (Outcome::OutOfWork, allowance);
                    }
                }
                None => return (Outcome::OutOfWork, allowance),
            }
        }
        (Outcome::NotFound { exhausted: !cut }, spent)
    }
}

/// One thread's depth-first search, with a buffer of fired detectors per
/// depth.
struct Walker<'s, 'a> {
    search: &'s Search<'a>,
    levels: Vec<Vec<u32>>,
    chosen: Vec<u32>,
    /// Sets visited and not yet counted in the budget.
    pending: u64,
    /// Sets visited from the current first error.
    nodes: u64,
    /// Whether the weight cut a branch from the current first error.
    cut: bool,
}

impl<'s, 'a> Walker<'s, 'a> {
    fn new(search: &'s Search<'a>, weight: usize) -> Self {
        Walker {
            search,
            levels: vec![Vec::with_capacity(4 * search.widest.max(1)); weight],
            chosen: Vec::with_capacity(weight),
            pending: 0,
            nodes: 0,
            cut: false,
        }
    }

    /// The errors chosen, by index.
    fn set(&self) -> Vec<usize> {
        self.chosen.iter().map(|&k| k as usize).collect()
    }

    /// Whether error `first` extends by errors of higher index to a
    /// logical error of at most `weight` errors (`None`: out of budget).
    fn start(&mut self, first: u32, weight: usize, budget: &Budget) -> Option<bool> {
        let search = self.search;
        self.nodes = 0;
        self.cut = false;
        self.chosen.clear();
        self.chosen.push(first);
        self.levels[0].clear();
        self.levels[0].extend_from_slice(&search.detectors[first as usize]);
        let observables = search.errors[first as usize].observables;
        self.extend(0, observables, weight - 1, first, budget)
    }

    /// Whether `chosen` extends by at most `remaining` errors of index
    /// above `first` to a logical error, the detectors it fires at
    /// `levels[depth]` (`None`: out of budget).
    fn extend(
        &mut self,
        depth: usize,
        observables: u64,
        remaining: usize,
        first: u32,
        budget: &Budget,
    ) -> Option<bool> {
        let search = self.search;
        self.nodes += 1;
        self.pending += 1;
        if self.pending >= 1 << 12 {
            if !budget.spend(self.pending) {
                return None;
            }
            self.pending = 0;
        }
        let fired = self.levels[depth].len();
        if fired == 0 {
            return Some(observables != 0);
        }
        if remaining == 0 || fired > remaining * search.widest {
            self.cut = true;
            return Some(false);
        }
        let lowest = self.levels[depth][0] as usize;
        if remaining == 1 {
            // The last error must flip exactly the fired detectors.
            if let Some(ks) = search.by_effect.get(self.levels[depth].as_slice()) {
                for &k in ks {
                    if k > first
                        && !self.chosen.contains(&k)
                        && observables ^ search.errors[k as usize].observables != 0
                    {
                        self.chosen.push(k);
                        return Some(true);
                    }
                }
            }
            // Longer extensions, if any, were not tried.
            self.cut |= search.by_detector[lowest]
                .iter()
                .any(|&k| k > first && !self.chosen.contains(&k));
            return Some(false);
        }
        for &k in &search.by_detector[lowest] {
            if k <= first || self.chosen.contains(&k) {
                continue;
            }
            let (head, tail) = self.levels.split_at_mut(depth + 1);
            symmetric_difference(&head[depth], &search.detectors[k as usize], &mut tail[0]);
            self.chosen.push(k);
            let next = observables ^ search.errors[k as usize].observables;
            match self.extend(depth + 1, next, remaining - 1, first, budget) {
                Some(true) => return Some(true),
                Some(false) => {}
                None => return None,
            }
            self.chosen.pop();
        }
        Some(false)
    }
}

/// `out = a △ b` for ascending lists.
fn symmetric_difference(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
}

#[cfg(test)]
mod tests;
