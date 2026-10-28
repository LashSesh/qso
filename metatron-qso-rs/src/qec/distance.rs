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

use serde::{Deserialize, Serialize};

use super::bposd::{BpOsdDecoder, BpOsdOptions, MAX_BPOSD_DETECTORS};
use super::dem::{DetectorErrorModel, ErrorMechanism};
use crate::error::{QsoError, Result};
use crate::rng::Rng;

/// Most BP+OSD searches per observable.
pub const MAX_BPOSD_TRIALS: usize = 1 << 16;

/// Options of [`circuit_distance`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DistanceOptions {
    /// The exhaustive search stops after proving that no logical error of
    /// this weight exists.
    pub max_weight: usize,
    /// Most sets of errors the exhaustive search visits (about 10⁷ a
    /// second).
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
    let mut best = cycle;
    let consider = |set: Vec<usize>, best: &mut Option<Vec<usize>>| {
        if best.as_ref().is_none_or(|b| set.len() < b.len()) {
            *best = Some(set);
        }
    };
    let bits = errors.iter().fold(0u64, |b, e| b | e.observables);
    let singles: Vec<u64> = (0..64)
        .filter(|&b| bits >> b & 1 == 1)
        .map(|b| 1u64 << b)
        .collect();
    let mut rng = Rng::seed_from_u64(options.seed);
    if options.bposd_trials > MAX_BPOSD_TRIALS {
        return Err(QsoError::invalid(
            "bposd_trials",
            format!("at most {MAX_BPOSD_TRIALS} BP+OSD searches per observable"),
        ));
    }
    let trials = options.bposd_trials * singles.len();
    for trial in 0..trials {
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
        if let Some(set) = bposd_logical(n, &errors, target, perturbed, &mut rng)? {
            consider(set, &mut best);
        }
    }
    // Exhaustive search below the best found.
    let mut search = Search::new(n, &errors, options.max_work);
    let ceiling = best
        .as_ref()
        .map_or(options.max_weight, |b| options.max_weight.min(b.len() - 1));
    let mut lower = 2;
    for w in 2..=ceiling {
        match search.run(w) {
            Outcome::Found(set) => {
                consider(set, &mut best);
                lower = w;
                break;
            }
            Outcome::NotFound => lower = w + 1,
            Outcome::OutOfWork => break,
        }
    }
    // Nothing found: decide by linear algebra whether anything exists.
    let proven_none = best.is_none() && logical_exists(n, &errors) == Some(false);
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
        if let Some(k) = basis.insert(&super::gf2::from_ones(len, &ones)) {
            if basis.pivot(k) >= n {
                return Some(true);
            }
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
    NotFound,
    OutOfWork,
}

/// Exhaustive search for undetectable logical errors of a given weight.
///
/// Every such set of the least weight has a first (lowest-index) error;
/// starting from each error in turn, the search adds errors of higher
/// index that flip the lowest detector still fired (one of them must),
/// until no detector fires. A set whose prefix already fires no detector
/// and flips no observable is never the lightest (dropping the prefix
/// leaves a lighter logical error), so the search stops there.
struct Search<'a> {
    errors: &'a [Error],
    by_detector: Vec<Vec<usize>>,
    by_effect: HashMap<&'a [usize], Vec<usize>>,
    widest: usize,
    chosen: Vec<usize>,
    work: u64,
    max_work: u64,
}

impl<'a> Search<'a> {
    fn new(n: usize, errors: &'a [Error], max_work: u64) -> Self {
        let mut by_detector = vec![Vec::new(); n];
        let mut by_effect: HashMap<&[usize], Vec<usize>> = HashMap::new();
        for (k, e) in errors.iter().enumerate() {
            for &d in &e.detectors {
                by_detector[d].push(k);
            }
            by_effect.entry(e.detectors.as_slice()).or_default().push(k);
        }
        Search {
            errors,
            by_detector,
            by_effect,
            widest: errors.iter().map(|e| e.detectors.len()).max().unwrap_or(0),
            chosen: Vec::new(),
            work: 0,
            max_work,
        }
    }

    fn run(&mut self, weight: usize) -> Outcome {
        for first in 0..self.errors.len() {
            let e = &self.errors[first];
            self.chosen.clear();
            self.chosen.push(first);
            match self.extend(&e.detectors.clone(), e.observables, weight - 1, first) {
                Some(true) => return Outcome::Found(self.chosen.clone()),
                Some(false) => {}
                None => {
                    return Outcome::OutOfWork;
                }
            }
        }
        Outcome::NotFound
    }

    /// Whether `chosen` extends by at most `remaining` errors of index
    /// above `first` to a logical error (`None`: out of work).
    fn extend(
        &mut self,
        fired: &[usize],
        observables: u64,
        remaining: usize,
        first: usize,
    ) -> Option<bool> {
        self.work += 1;
        if self.work > self.max_work {
            return None;
        }
        if fired.is_empty() {
            return Some(observables != 0);
        }
        if remaining == 0 || fired.len() > remaining * self.widest {
            return Some(false);
        }
        if remaining == 1 {
            // The last error must flip exactly the fired detectors.
            if let Some(ks) = self.by_effect.get(fired) {
                for &k in ks {
                    if k > first
                        && !self.chosen.contains(&k)
                        && observables ^ self.errors[k].observables != 0
                    {
                        self.chosen.push(k);
                        return Some(true);
                    }
                }
            }
            return Some(false);
        }
        let lowest = fired[0];
        for i in 0..self.by_detector[lowest].len() {
            let k = self.by_detector[lowest][i];
            if k <= first || self.chosen.contains(&k) {
                continue;
            }
            let next = xor(fired, &self.errors[k].detectors);
            self.chosen.push(k);
            match self.extend(
                &next,
                observables ^ self.errors[k].observables,
                remaining - 1,
                first,
            ) {
                Some(true) => return Some(true),
                Some(false) => {}
                None => return None,
            }
            self.chosen.pop();
        }
        Some(false)
    }
}

/// Symmetric difference of two ascending lists.
fn xor(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(a.len() + b.len());
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
    out
}

#[cfg(test)]
mod tests;
