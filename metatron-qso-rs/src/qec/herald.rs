//! Heralded errors: the model a decoder conditions on the heralds that
//! fired, and a decoder that does so shot by shot.
//!
//! A heralded channel (Stim's `HERALDED_ERASE` and
//! `HERALDED_PAULI_CHANNEL_1`, see [`ExtendedOp`](super::ExtendedOp)) adds a
//! measurement record, its herald, that reads 1 exactly when the channel
//! struck. A detector whose only record is a herald, and which is the only
//! reader of that record, is a *herald detector*: it owns the herald. A
//! decoder that knows which herald detectors fired knows where the channel
//! struck: given that it fired, an erasure's X and Z parts each happen with
//! probability ½, and given that it did not, nothing happened there.
//!
//! [`heralded_error_model`] splits a circuit's model accordingly: the
//! errors of every channel whose herald no herald detector owns (by Stim's
//! approximate rule, as [`detector_error_model`](super::detector_error_model)
//! gives them), and for every herald detector the probability that it
//! fires and the parts of its channel given that it fired. A
//! [`ConditionedDecoder`] merges the parts of the fired heralds into the
//! model of each shot: union-find joins the erased edges before it grows
//! (Delfosse and Nickerson, Quantum 5, 595 (2021)), sparse matching sets
//! the weights of the shot's edges and restores them after, and BP+OSD sets
//! its priors.
//!
//! For a heralded Pauli channel the X and Z parts given the herald are
//! treated as independent, and equal parts of different channels are merged
//! as independent errors; both are exact for an erasure and approximate
//! otherwise. A channel whose X and Z parts have the same effect gives one
//! part, at its exact probability (pX + pZ)/t.
//!
//! A part can be more likely than not given its herald (a Pauli channel's X
//! part is (pX + pY)/t). The decoder then applies it before decoding and
//! decodes the rest with 1 − p, as PyMatching does with a negative weight.

use std::collections::{BTreeMap, HashMap};

use super::bposd::{BpOsdDecoder, BpOsdOptions, BpOsdOutcome, check_options, prior_of};
use super::circuit::DetectorCircuit;
use super::decoder::{MatchingGraph, UnionFindDecoder, edge_length};
use super::dem::{
    Channel, Contribute, DetectorErrorModel, ErrorMechanism, Herald, Key, MAX_MODEL_DETECTORS,
    Signature, fold, fold_flat, is_empty, model_contributions,
};
use super::repeat::RepeatedCircuit;
use super::sparse::{SparseMatchingDecoder, WEIGHT_SCALE};
use crate::error::{QsoError, Result};

/// p ⊕ q: the probability that exactly one of two independent errors of
/// probabilities p and q happens; ½ absorbs exactly (an erased error stays
/// erased whatever else happens).
pub(super) fn xor_probability(p: f64, q: f64) -> f64 {
    if p == 0.5 || q == 0.5 {
        0.5
    } else {
        p * (1.0 - q) + q * (1.0 - p)
    }
}

/// A key of the heralded model: an error of the base model, a part of the
/// channel owned by a herald detector, or the rate of that herald.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum HKey {
    Base(Signature),
    Part(usize, Signature),
    Rate(usize),
}

impl Key for HKey {
    fn lowered(&self, by: usize) -> Self {
        match self {
            HKey::Base(s) => HKey::Base(s.lowered(by)),
            HKey::Part(d, s) => HKey::Part(d - by, s.lowered(by)),
            HKey::Rate(d) => HKey::Rate(d - by),
        }
    }

    /// Never 0 for herald keys, so that the folding of a repeated block
    /// shifts them with their iteration.
    fn entries(&self) -> usize {
        match self {
            HKey::Base(s) => s.0.len(),
            HKey::Part(_, s) => 1 + s.0.len(),
            HKey::Rate(_) => 1,
        }
    }
}

/// The contributions of a channel to the heralded model: a channel whose
/// herald a herald detector owns gives that herald's rate (even 0, so that
/// the herald stays listed) and its X and Z parts given that it fired; every
/// other channel gives what it gives to [`detector_error_model`]
/// (super::detector_error_model).
fn heralded_contributions(decompose: bool) -> Box<Contribute<'static, HKey>> {
    let mut inner = model_contributions(decompose);
    let mut scratch: Vec<(Signature, f64)> = Vec::new();
    Box::new(move |channel: &Channel, basis, probabilities, out| {
        if let Some(Herald {
            owner: Some(d),
            probability,
            x,
            z,
            xz,
        }) = channel.herald
        {
            out.push((HKey::Rate(d), probability));
            if basis[0] == basis[1] {
                // X and Z have the same effect (Y has none): one part, at
                // the probability that exactly one of them happened (kept
                // at 0 for a pure Y, as two certain parts would merge).
                if (x > 0.0 || z > 0.0) && !is_empty(&basis[0]) {
                    out.push((HKey::Part(d, basis[0].clone()), xz));
                }
                return Ok(());
            }
            if x > 0.0 && !is_empty(&basis[1]) {
                out.push((HKey::Part(d, basis[1].clone()), x));
            }
            if z > 0.0 && !is_empty(&basis[0]) {
                out.push((HKey::Part(d, basis[0].clone()), z));
            }
            return Ok(());
        }
        scratch.clear();
        inner(channel, basis, probabilities, &mut scratch)?;
        out.extend(scratch.drain(..).map(|(k, p)| (HKey::Base(k), p)));
        Ok(())
    })
}

fn mechanism((detectors, observables): Signature, probability: f64) -> ErrorMechanism {
    ErrorMechanism {
        probability,
        detectors,
        observables,
    }
}

/// The heralded model from the merged contributions, in key order.
fn assemble(
    merged: BTreeMap<HKey, f64>,
    n_detectors: usize,
    n_observables: usize,
    decomposed: bool,
) -> Result<HeraldedErrorModel> {
    let mut errors = Vec::new();
    let mut herald_detectors = Vec::new();
    let mut herald_probabilities = Vec::new();
    let mut parts = Vec::new();
    for (key, p) in merged {
        match key {
            HKey::Base(s) => errors.push(mechanism(s, p)),
            HKey::Part(d, s) => parts.push((d, mechanism(s, p))),
            HKey::Rate(d) => {
                herald_detectors.push(d);
                herald_probabilities.push(p);
            }
        }
    }
    let mut outcomes = vec![Vec::new(); herald_detectors.len()];
    for (d, part) in parts {
        let h = herald_detectors.binary_search(&d).map_err(|_| {
            QsoError::invalid(
                "model",
                format!("a part of detector {d} without its herald"),
            )
        })?;
        outcomes[h].push(part);
    }
    Ok(HeraldedErrorModel {
        base: DetectorErrorModel {
            n_detectors,
            n_observables,
            errors,
        },
        herald_detectors,
        herald_probabilities,
        outcomes,
        decomposed,
    })
}

/// The heralded model of a circuit (see the module docs): its herald
/// detectors (detectors whose only record is a herald that nothing else
/// reads), and the errors of every other channel by Stim's approximate
/// rule. With `decompose`, the base model is decomposed as by
/// [`detector_error_model`](super::detector_error_model)`(circuit, true)`;
/// the parts never are (they are a channel's X and Z effects themselves).
/// Heralds that no herald detector owns (unread, read by an observable, or
/// read by a detector of several records) enter the base model by Stim's
/// rule. A circuit without herald detectors gives its full model as the
/// base.
pub fn heralded_error_model(
    circuit: &DetectorCircuit,
    decompose: bool,
) -> Result<HeraldedErrorModel> {
    let mut contribute = heralded_contributions(decompose);
    let merged = fold_flat::<HKey>(circuit, contribute.as_mut())?;
    assemble(
        merged,
        circuit.n_detectors(),
        circuit.n_observables(),
        decompose,
    )
}

/// [`heralded_error_model`] of a repeated circuit, by the folded pass.
pub(super) fn repeated_heralded_model(
    circuit: &RepeatedCircuit,
    decompose: bool,
) -> Result<HeraldedErrorModel> {
    let mut contribute = heralded_contributions(decompose);
    let merged = fold::<HKey>(circuit, contribute.as_mut())?;
    assemble(
        merged,
        circuit.n_detectors(),
        circuit.n_observables(),
        decompose,
    )
}

/// The model a decoder conditions on the heralds that fired: the errors of
/// every channel whose herald no detector of its own reads (Stim's
/// approximate rule, as [`detector_error_model`](super::detector_error_model)
/// gives them), and for each herald detector the probability that it fires
/// and the parts of its channel given that it fired.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct HeraldedErrorModel {
    base: DetectorErrorModel,
    herald_detectors: Vec<usize>,
    herald_probabilities: Vec<f64>,
    outcomes: Vec<Vec<ErrorMechanism>>,
    /// Whether the base model was decomposed into graph-like parts (built
    /// with `decompose = true`); BP+OSD refuses such a model.
    decomposed: bool,
}

fn check_probability(what: &str, p: f64) -> Result<()> {
    if !(0.0..=1.0).contains(&p) {
        return Err(QsoError::invalid(
            "model",
            format!("{what} has probability {p}, not in [0, 1]"),
        ));
    }
    Ok(())
}

impl HeraldedErrorModel {
    /// A heralded model from its parts, checked: herald detectors strictly
    /// ascending and below `base.n_detectors`; one probability (in [0, 1])
    /// and one list of parts per herald; no base error or part flips a
    /// herald detector; every error and part with a probability in
    /// [0, 1], its detectors ascending, distinct and below
    /// `base.n_detectors`, and its observables below `base.n_observables`.
    /// The parts of each herald are sorted by (detectors, observables) and
    /// equal parts merged as independent errors.
    pub fn new(
        base: DetectorErrorModel,
        herald_detectors: Vec<usize>,
        herald_probabilities: Vec<f64>,
        outcomes: Vec<Vec<ErrorMechanism>>,
    ) -> Result<Self> {
        let n = base.n_detectors;
        if n > MAX_MODEL_DETECTORS {
            return Err(QsoError::invalid(
                "model",
                format!("{n} detectors; a model holds at most {MAX_MODEL_DETECTORS}"),
            ));
        }
        if herald_detectors.len() != herald_probabilities.len()
            || herald_detectors.len() != outcomes.len()
        {
            return Err(QsoError::invalid(
                "model",
                format!(
                    "{} herald detectors, {} probabilities and {} lists of parts",
                    herald_detectors.len(),
                    herald_probabilities.len(),
                    outcomes.len()
                ),
            ));
        }
        if herald_detectors.windows(2).any(|w| w[0] >= w[1]) {
            return Err(QsoError::invalid(
                "herald_detectors",
                "herald detectors must be strictly ascending",
            ));
        }
        if let Some(&d) = herald_detectors.iter().find(|&&d| d >= n) {
            return Err(QsoError::invalid(
                "herald_detectors",
                format!("detector {d} of {n}"),
            ));
        }
        for (h, &p) in herald_probabilities.iter().enumerate() {
            check_probability(&format!("herald {h}"), p)?;
        }
        let observables_mask = if base.n_observables >= 64 {
            u64::MAX
        } else {
            (1u64 << base.n_observables) - 1
        };
        let check = |what: &str, e: &ErrorMechanism| -> Result<()> {
            check_probability(what, e.probability)?;
            if e.detectors.windows(2).any(|w| w[0] >= w[1]) {
                return Err(QsoError::invalid(
                    "model",
                    format!("the detectors of {what} must be ascending and distinct"),
                ));
            }
            if let Some(&d) = e.detectors.iter().find(|&&d| d >= n) {
                return Err(QsoError::invalid(
                    "model",
                    format!("{what} flips detector {d} of {n}"),
                ));
            }
            if let Some(&d) = e
                .detectors
                .iter()
                .find(|d| herald_detectors.binary_search(d).is_ok())
            {
                return Err(QsoError::invalid(
                    "model",
                    format!("{what} flips herald detector {d}"),
                ));
            }
            if e.observables & !observables_mask != 0 {
                return Err(QsoError::invalid(
                    "model",
                    format!(
                        "{what} flips an observable beyond the {}",
                        base.n_observables
                    ),
                ));
            }
            Ok(())
        };
        for (k, e) in base.errors.iter().enumerate() {
            check(&format!("error {k}"), e)?;
        }
        let mut merged_outcomes = Vec::with_capacity(outcomes.len());
        for (h, parts) in outcomes.into_iter().enumerate() {
            let mut merged: BTreeMap<Signature, f64> = BTreeMap::new();
            for (k, e) in parts.into_iter().enumerate() {
                check(&format!("part {k} of herald {h}"), &e)?;
                let v = merged.entry((e.detectors, e.observables)).or_insert(0.0);
                *v = xor_probability(*v, e.probability);
            }
            merged_outcomes.push(merged.into_iter().map(|(s, p)| mechanism(s, p)).collect());
        }
        Ok(HeraldedErrorModel {
            base,
            herald_detectors,
            herald_probabilities,
            outcomes: merged_outcomes,
            decomposed: false,
        })
    }

    /// Whether the base model was decomposed into graph-like parts, as
    /// [`heralded_error_model`] gives it with `decompose = true`. A model
    /// built from its parts with [`HeraldedErrorModel::new`] counts as not
    /// decomposed.
    pub fn is_decomposed(&self) -> bool {
        self.decomposed
    }

    /// Detectors of the circuit, herald detectors included.
    pub fn n_detectors(&self) -> usize {
        self.base.n_detectors
    }

    /// Logical observables of the circuit.
    pub fn n_observables(&self) -> usize {
        self.base.n_observables
    }

    /// The errors of every channel no herald detector owns; none flips a
    /// herald detector.
    pub fn base(&self) -> &DetectorErrorModel {
        &self.base
    }

    /// The herald detectors, ascending.
    pub fn herald_detectors(&self) -> &[usize] {
        &self.herald_detectors
    }

    /// The probability that herald `h` (an index into
    /// [`HeraldedErrorModel::herald_detectors`]) fires.
    ///
    /// # Panics
    ///
    /// If `h` is not below the number of herald detectors.
    pub fn herald_probability(&self, h: usize) -> f64 {
        self.herald_probabilities[h]
    }

    /// The parts of herald `h`'s channel given that it fired: its X part at
    /// (pX + pY)/t and its Z part at (pZ + pY)/t (½ each for an erasure),
    /// the herald left out, in ascending order of (detectors, observables);
    /// equal parts merged as independent errors, except that a channel
    /// whose X and Z parts have the same effect gives one part, at
    /// (pX + pZ)/t.
    ///
    /// # Panics
    ///
    /// If `h` is not below the number of herald detectors.
    pub fn outcomes(&self, h: usize) -> &[ErrorMechanism] {
        &self.outcomes[h]
    }

    /// Herald ordinals of the fired herald detectors (circuit indices),
    /// ascending, pairs cancelled.
    fn ordinals(&self, fired_heralds: &[usize]) -> Result<Vec<usize>> {
        let mut out = Vec::with_capacity(fired_heralds.len());
        for &d in fired_heralds {
            let h = self.herald_detectors.binary_search(&d).map_err(|_| {
                QsoError::invalid("heralds", format!("detector {d} is not a herald detector"))
            })?;
            out.push(h);
        }
        Ok(cancel_pairs(out))
    }

    /// The model of one shot: the base model with the parts of the fired
    /// herald detectors (circuit indices; a detector listed twice cancels)
    /// merged in as independent errors (heralds ascending, each one's parts
    /// in order), sorted by (detectors, observables).
    pub fn conditioned(&self, fired_heralds: &[usize]) -> Result<DetectorErrorModel> {
        let heralds = self.ordinals(fired_heralds)?;
        let mut merged: BTreeMap<Signature, f64> = BTreeMap::new();
        for e in &self.base.errors {
            let v = merged
                .entry((e.detectors.clone(), e.observables))
                .or_insert(0.0);
            *v = xor_probability(*v, e.probability);
        }
        for h in heralds {
            for e in &self.outcomes[h] {
                let v = merged
                    .entry((e.detectors.clone(), e.observables))
                    .or_insert(0.0);
                *v = xor_probability(*v, e.probability);
            }
        }
        Ok(DetectorErrorModel {
            n_detectors: self.base.n_detectors,
            n_observables: self.base.n_observables,
            errors: merged.into_iter().map(|(s, p)| mechanism(s, p)).collect(),
        })
    }
}

/// Sorted, each value kept if it occurs an odd number of times.
fn cancel_pairs(mut v: Vec<usize>) -> Vec<usize> {
    v.sort_unstable();
    let mut out = Vec::with_capacity(v.len());
    let mut k = 0;
    while k < v.len() {
        let mut end = k;
        while end < v.len() && v[end] == v[k] {
            end += 1;
        }
        if (end - k) % 2 == 1 {
            out.push(v[k]);
        }
        k = end;
    }
    out
}

/// How a [`ConditionedDecoder`] decodes (more may be added).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub enum HeraldedMethod {
    /// Weighted union-find ([`UnionFindDecoder`]); the erased edges are
    /// joined before growing (Delfosse and Nickerson, Quantum 5, 595
    /// (2021)). Needs graph-like errors.
    #[default]
    UnionFind,
    /// Minimum-weight perfect matching ([`SparseMatchingDecoder`]), the
    /// shot's edge weights set and restored. Needs graph-like errors.
    Matching,
    /// BP+OSD ([`BpOsdDecoder`]) with these options, its priors set shot by
    /// shot.
    BpOsd(BpOsdOptions),
}

/// The decoder a [`ConditionedDecoder`] conditions, on the ordinary
/// detectors.
#[derive(Clone, Debug)]
enum Inner {
    UnionFind(UnionFindDecoder),
    Matching(SparseMatchingDecoder),
    BpOsd(BpOsdDecoder),
}

impl Inner {
    /// Set edge or column `e` from probability `p`.
    fn set(&mut self, e: usize, p: f64) {
        match self {
            Inner::UnionFind(d) => d.set_length(e, p),
            Inner::Matching(d) => d.set_weight(e, p),
            Inner::BpOsd(d) => d.set_prior(e, p),
        }
    }
}

const NOT: u32 = u32::MAX;

/// A decoder that conditions a [`HeraldedErrorModel`] on the heralds that
/// fired, shot by shot; reusable across shots.
///
/// Its *slots* are the base model's errors, then every part of a herald
/// that no base error has (probability 0 until a herald turns it on). For
/// each shot the slots of the fired heralds' parts are merged with them as
/// independent errors (an erased part has probability ½, which absorbs
/// everything else), the inner decoder decodes the ordinary detectors (the
/// herald detectors renumbered away) with those probabilities, and the
/// slots are restored. A slot the shot makes more likely than not (p > ½,
/// such as a Pauli channel's X part (pX + pY)/t) is applied first: its
/// detectors are added to the shot's and its observables to the
/// prediction, and the inner decoder takes it with 1 − p (PyMatching's
/// rule for edges of negative weight).
#[derive(Clone, Debug)]
pub struct ConditionedDecoder {
    method: HeraldedMethod,
    n_detectors: usize,
    herald_detectors: Vec<usize>,
    /// Herald ordinal of each detector (`NOT`: ordinary).
    herald_of: Vec<u32>,
    /// Index among the ordinary detectors of each detector (`NOT`: a
    /// herald detector).
    ordinary: Vec<u32>,
    slots: Vec<ErrorMechanism>,
    base_p: Vec<f64>,
    /// The inner decoder's edge or column of each slot (`NOT`: a slot that
    /// flips no detector), and the slot of each.
    edge_of_slot: Vec<u32>,
    slot_of_edge: Vec<usize>,
    /// Herald `h`'s parts as (slot, probability):
    /// `herald_parts[herald_start[h]..herald_start[h + 1]]`.
    herald_start: Vec<usize>,
    herald_parts: Vec<(usize, f64)>,
    inner: Inner,
    // Work space kept across shots.
    current: Vec<f64>,
    marked: Vec<bool>,
    touched: Vec<usize>,
    heralds: Vec<usize>,
    fired: Vec<usize>,
    erased: Vec<usize>,
    /// The slots the shot applied before decoding (p > ½), ascending, and
    /// their observables.
    applied: Vec<usize>,
    applied_observables: u64,
}

impl ConditionedDecoder {
    /// The decoder of `circuit` by `method`, on the heralded model the
    /// method needs: decomposed for union-find and matching, without
    /// decomposition for BP+OSD (as [`DecoderChoice`](super::DecoderChoice)
    /// builds it).
    pub fn for_circuit(circuit: &DetectorCircuit, method: HeraldedMethod) -> Result<Self> {
        let decompose = !matches!(method, HeraldedMethod::BpOsd(_));
        Self::new(&heralded_error_model(circuit, decompose)?, method)
    }

    /// [`ConditionedDecoder::for_circuit`] of a repeated circuit.
    pub fn for_repeated(circuit: &RepeatedCircuit, method: HeraldedMethod) -> Result<Self> {
        let decompose = !matches!(method, HeraldedMethod::BpOsd(_));
        Self::new(&repeated_heralded_model(circuit, decompose)?, method)
    }

    /// The decoder of `model` by `method`. Union-find and matching need
    /// every base error and every part to flip at most two ordinary
    /// detectors; BP+OSD takes at most
    /// [`MAX_BPOSD_DETECTORS`](super::MAX_BPOSD_DETECTORS) ordinary
    /// detectors (herald detectors are not counted).
    ///
    /// BP+OSD decodes the model without decomposition (its parts are the
    /// circuit's errors as they happen); a decomposed model is refused for
    /// it. [`ConditionedDecoder::for_circuit`] builds the model each method
    /// needs.
    pub fn new(model: &HeraldedErrorModel, method: HeraldedMethod) -> Result<Self> {
        if let HeraldedMethod::BpOsd(options) = &method {
            check_options(options)?;
            if model.decomposed {
                return Err(QsoError::invalid(
                    "model",
                    "BP+OSD decodes the heralded model without decomposition: build it \
                     without decomposition, or build the decoder with for_circuit",
                ));
            }
        }
        let n = model.n_detectors();
        let mut herald_of = vec![NOT; n];
        for (h, &d) in model.herald_detectors.iter().enumerate() {
            herald_of[d] = h as u32;
        }
        let mut ordinary = vec![NOT; n];
        let mut m = 0u32;
        for d in 0..n {
            if herald_of[d] == NOT {
                ordinary[d] = m;
                m += 1;
            }
        }
        // Slots: the base errors, then the parts no base error has.
        let mut slots: Vec<ErrorMechanism> = model.base.errors.clone();
        let mut slot_of: HashMap<Signature, usize> = HashMap::with_capacity(slots.len());
        for (k, e) in slots.iter().enumerate() {
            slot_of
                .entry((e.detectors.clone(), e.observables))
                .or_insert(k);
        }
        let mut herald_start = Vec::with_capacity(model.outcomes.len() + 1);
        let mut herald_parts = Vec::new();
        herald_start.push(0);
        for parts in &model.outcomes {
            for e in parts {
                let key = (e.detectors.clone(), e.observables);
                let slot = match slot_of.get(&key) {
                    Some(&s) => s,
                    None => {
                        slots.push(ErrorMechanism {
                            probability: 0.0,
                            detectors: e.detectors.clone(),
                            observables: e.observables,
                        });
                        slot_of.insert(key, slots.len() - 1);
                        slots.len() - 1
                    }
                };
                herald_parts.push((slot, e.probability));
            }
            herald_start.push(herald_parts.len());
        }
        let base_p: Vec<f64> = slots.iter().map(|e| e.probability).collect();
        let graphlike = !matches!(method, HeraldedMethod::BpOsd(_));
        if graphlike {
            for &(slot, _) in &herald_parts {
                let k = slots[slot].detectors.len();
                if k > 2 {
                    return Err(QsoError::invalid(
                        "model",
                        format!(
                            "a heralded part flips {k} detectors; matching needs at most two (decode it with BP+OSD)"
                        ),
                    ));
                }
            }
        }
        // The model of the ordinary detectors: every slot that flips one.
        let mut edge_of_slot = vec![NOT; slots.len()];
        let mut slot_of_edge = Vec::new();
        let mut errors = Vec::new();
        for (slot, e) in slots.iter().enumerate() {
            if e.detectors.is_empty() {
                continue;
            }
            edge_of_slot[slot] = slot_of_edge.len() as u32;
            slot_of_edge.push(slot);
            errors.push(ErrorMechanism {
                probability: base_p[slot],
                detectors: e.detectors.iter().map(|&d| ordinary[d] as usize).collect(),
                observables: e.observables,
            });
        }
        let compact = DetectorErrorModel {
            n_detectors: m as usize,
            n_observables: model.n_observables(),
            errors,
        };
        let inner = match method {
            HeraldedMethod::UnionFind => {
                Inner::UnionFind(UnionFindDecoder::new(MatchingGraph::from_slots(&compact)?))
            }
            HeraldedMethod::Matching => {
                // The longest edge any shot can have. Merging independent
                // errors never takes a probability further from ½ than any
                // of them, and an applied slot (p > ½) is decoded with
                // 1 − p, so an edge is never longer than at min(p, 1 − p)
                // of its base error, if that is in (0, 1), and otherwise of
                // one of its parts. A slot whose every error is certain or
                // impossible has probability 0 or 1 and is never an edge
                // of finite length.
                let tame = |p: f64| p > 0.0 && p < 1.0;
                let length = |p: f64| edge_length(p.min(1.0 - p));
                let mut longest: f64 = 0.0;
                for e in compact.errors.iter().filter(|e| tame(e.probability)) {
                    longest = longest.max(length(e.probability));
                }
                for &(slot, q) in &herald_parts {
                    if !tame(base_p[slot]) && tame(q) && edge_of_slot[slot] != NOT {
                        longest = longest.max(length(q));
                    }
                }
                let scale = if longest > 0.0 {
                    WEIGHT_SCALE / longest
                } else {
                    1.0
                };
                Inner::Matching(SparseMatchingDecoder::with_scale(
                    MatchingGraph::from_slots(&compact)?,
                    scale,
                ))
            }
            HeraldedMethod::BpOsd(options) => {
                let d = BpOsdDecoder::with_columns(&compact, options)?;
                debug_assert!(d.sources().iter().enumerate().all(|(j, &k)| j == k));
                Inner::BpOsd(d)
            }
        };
        let n_slots = slots.len();
        Ok(ConditionedDecoder {
            method,
            n_detectors: n,
            herald_detectors: model.herald_detectors.clone(),
            herald_of,
            ordinary,
            slots,
            base_p,
            edge_of_slot,
            slot_of_edge,
            herald_start,
            herald_parts,
            inner,
            current: vec![0.0; n_slots],
            marked: vec![false; n_slots],
            touched: Vec::new(),
            heralds: Vec::new(),
            fired: Vec::new(),
            erased: Vec::new(),
            applied: Vec::new(),
            applied_observables: 0,
        })
    }

    /// How it decodes.
    pub fn method(&self) -> HeraldedMethod {
        self.method
    }

    /// Detectors of the circuit, herald detectors included.
    pub fn n_detectors(&self) -> usize {
        self.n_detectors
    }

    /// The herald detectors, ascending.
    pub fn herald_detectors(&self) -> &[usize] {
        &self.herald_detectors
    }

    /// The slots (edges or columns) in decoding order with their base
    /// probabilities (0 for a slot only heralds turn on), in the circuit's
    /// numbering of detectors.
    pub fn slots(&self) -> &[ErrorMechanism] {
        &self.slots
    }

    /// Checks the shot and fills `fired` (ordinary indices) and `heralds`
    /// (ordinals, pairs cancelled).
    fn prepare(&mut self, fired: &[usize], heralds: &[usize]) -> Result<()> {
        let n = self.n_detectors;
        let mut compact = std::mem::take(&mut self.fired);
        compact.clear();
        for &d in fired {
            if d >= n {
                self.fired = compact;
                return Err(QsoError::invalid("fired", format!("detector {d} of {n}")));
            }
            if self.herald_of[d] != NOT {
                self.fired = compact;
                return Err(QsoError::invalid(
                    "fired",
                    format!("detector {d} is a herald detector"),
                ));
            }
            compact.push(self.ordinary[d] as usize);
        }
        let compact = cancel_pairs(compact);
        let mut ordinals = std::mem::take(&mut self.heralds);
        ordinals.clear();
        for &d in heralds {
            if d >= n {
                self.fired = compact;
                self.heralds = ordinals;
                return Err(QsoError::invalid("heralds", format!("detector {d} of {n}")));
            }
            if self.herald_of[d] == NOT {
                self.fired = compact;
                self.heralds = ordinals;
                return Err(QsoError::invalid(
                    "heralds",
                    format!("detector {d} is not a herald detector"),
                ));
            }
            ordinals.push(self.herald_of[d] as usize);
        }
        self.heralds = cancel_pairs(ordinals);
        self.fired = compact;
        Ok(())
    }

    /// Merge the fired heralds' parts into their slots and set the inner
    /// decoder's edges or columns; the erased edges (p = ½) go to `erased`,
    /// and the slots above ½ are applied: their detectors toggle `fired`,
    /// their observables `applied_observables`, and the inner decoder
    /// takes them with 1 − p.
    fn condition(&mut self) {
        self.touched.clear();
        self.erased.clear();
        self.applied.clear();
        self.applied_observables = 0;
        for &h in &self.heralds {
            for &(slot, q) in &self.herald_parts[self.herald_start[h]..self.herald_start[h + 1]] {
                if !self.marked[slot] {
                    self.marked[slot] = true;
                    self.current[slot] = self.base_p[slot];
                    self.touched.push(slot);
                }
                self.current[slot] = xor_probability(self.current[slot], q);
            }
        }
        self.touched.sort_unstable();
        let flipped = self.fired.len();
        for &slot in &self.touched {
            let mut p = self.current[slot];
            if p > 0.5 {
                p = 1.0 - p;
                self.applied.push(slot);
                self.applied_observables ^= self.slots[slot].observables;
                let ordinary = &self.ordinary;
                self.fired.extend(
                    self.slots[slot]
                        .detectors
                        .iter()
                        .map(|&d| ordinary[d] as usize),
                );
            }
            let e = self.edge_of_slot[slot];
            if e == NOT {
                continue;
            }
            self.inner.set(e as usize, p);
            if p == 0.5 {
                self.erased.push(e as usize);
            }
        }
        if self.fired.len() > flipped {
            self.fired = cancel_pairs(std::mem::take(&mut self.fired));
        }
    }

    /// Σ −`weight`(1 − p) over the applied slots: their part of a
    /// correction's weight, with `weight` the inner decoder's weight of an
    /// error of probability 1 − p.
    fn applied_weight(&self, weight: impl Fn(f64) -> f64) -> f64 {
        self.applied
            .iter()
            .map(|&slot| -weight(1.0 - self.current[slot]))
            .sum()
    }

    /// Restore the slots a shot set, from their base probabilities.
    fn restore(&mut self) {
        for &slot in &self.touched {
            self.marked[slot] = false;
            let e = self.edge_of_slot[slot];
            if e != NOT {
                self.inner.set(e as usize, self.base_p[slot]);
            }
        }
        self.touched.clear();
        self.erased.clear();
    }

    /// Runs `decode` on the inner decoder for the conditioned shot (`empty`
    /// if no ordinary detector is left to explain once the slots above ½
    /// are applied), and restores the slots whatever it returns. The
    /// applied slots stay in `applied` until the next shot.
    fn run<T>(
        &mut self,
        fired: &[usize],
        heralds: &[usize],
        empty: T,
        decode: impl FnOnce(&mut Inner, &[usize], &[usize]) -> Result<T>,
    ) -> Result<T> {
        self.prepare(fired, heralds)?;
        self.condition();
        if self.fired.is_empty() {
            self.restore();
            return Ok(empty);
        }
        let (compact, erased) = (
            std::mem::take(&mut self.fired),
            std::mem::take(&mut self.erased),
        );
        let result = decode(&mut self.inner, &compact, &erased);
        (self.fired, self.erased) = (compact, erased);
        self.restore();
        result
    }

    /// Predicted observable flips for the fired ordinary detectors `fired`,
    /// given the fired herald detectors `heralds` (both in the circuit's
    /// numbering; a detector listed twice cancels). The errors the shot
    /// makes more likely than not are applied first (see
    /// [`ConditionedDecoder`]); a shot with no ordinary detector left to
    /// explain predicts their flips.
    pub fn decode_heralded(&mut self, fired: &[usize], heralds: &[usize]) -> Result<u64> {
        let predicted = self.run(fired, heralds, 0, |inner, fired, erased| match inner {
            Inner::UnionFind(d) => d.decode_erased(fired, erased),
            Inner::Matching(d) => d.decode(fired),
            Inner::BpOsd(d) => d.decode(fired),
        })?;
        Ok(predicted ^ self.applied_observables)
    }

    /// Matching only: the prediction and the weight of the correction,
    /// Σ ln((1 − p)/p) over its errors at the shot's probabilities (errors
    /// at p = ½ count 0, errors above ½ that the correction keeps count
    /// negatively, as in PyMatching; up to the rounding of the weights to
    /// integers).
    pub fn decode_with_weight(&mut self, fired: &[usize], heralds: &[usize]) -> Result<(u64, f64)> {
        if !matches!(self.inner, Inner::Matching(_)) {
            return Err(QsoError::invalid(
                "decoder",
                "decode_with_weight needs the matching method",
            ));
        }
        let (predicted, weight) =
            self.run(fired, heralds, (0, 0.0), |inner, fired, _| match inner {
                Inner::Matching(d) => d.decode_with_weight(fired),
                Inner::UnionFind(_) | Inner::BpOsd(_) => unreachable!("checked above"),
            })?;
        Ok((
            predicted ^ self.applied_observables,
            weight + self.applied_weight(edge_length),
        ))
    }

    /// BP+OSD only: the correction and how it was found; its `errors`
    /// index [`ConditionedDecoder::slots`], the applied ones included
    /// unless BP+OSD undid them, and its weight counts them at their
    /// shot's probabilities (negatively).
    pub fn decode_detailed(&mut self, fired: &[usize], heralds: &[usize]) -> Result<BpOsdOutcome> {
        if !matches!(self.inner, Inner::BpOsd(_)) {
            return Err(QsoError::invalid(
                "decoder",
                "decode_detailed needs the BP+OSD method",
            ));
        }
        let empty = BpOsdOutcome {
            observables: 0,
            errors: Vec::new(),
            converged: true,
            iterations: 0,
            weight: 0.0,
        };
        let mut outcome = self.run(fired, heralds, empty, |inner, fired, _| match inner {
            Inner::BpOsd(d) => d.decode_detailed(fired),
            Inner::UnionFind(_) | Inner::Matching(_) => unreachable!("checked above"),
        })?;
        for e in &mut outcome.errors {
            *e = self.slot_of_edge[*e];
        }
        // The correction is the applied slots and BP+OSD's, a slot in both
        // undone.
        outcome.errors.extend_from_slice(&self.applied);
        outcome.errors = cancel_pairs(std::mem::take(&mut outcome.errors));
        outcome.observables ^= self.applied_observables;
        outcome.weight += self.applied_weight(prior_of);
        Ok(outcome)
    }

    /// Splits every fired detector into ordinary and herald detectors.
    pub(super) fn split(&self, fired: &[usize]) -> Result<(Vec<usize>, Vec<usize>)> {
        let n = self.n_detectors;
        let (mut ordinary, mut heralds) = (Vec::new(), Vec::new());
        for &d in fired {
            if d >= n {
                return Err(QsoError::invalid(
                    "fired",
                    format!("detector {d} does not exist ({n} detectors)"),
                ));
            }
            if self.herald_of[d] == NOT {
                ordinary.push(d);
            } else {
                heralds.push(d);
            }
        }
        Ok((ordinary, heralds))
    }

    /// The inner decoder's current probability of each slot it decodes, as
    /// a model in the compact numbering (for the tests).
    #[cfg(test)]
    pub(super) fn inner_state(&self) -> Vec<f64> {
        match &self.inner {
            Inner::UnionFind(d) => d.graph().edges().iter().map(|e| e.length).collect(),
            Inner::Matching(d) => d.weight.iter().map(|&w| w as f64).collect(),
            Inner::BpOsd(d) => d.priors().to_vec(),
        }
    }
}

#[cfg(test)]
mod tests;
