//! Decoding in windows: an experiment of many rounds decoded a window of
//! rounds at a time, as hardware decodes while the experiment runs.
//!
//! The detectors fall into layers (rounds; see
//! [`DetectorCircuit::detector_layers`]). A window covers `window`
//! consecutive layers; the first `commit` of them are its commit region,
//! the rest its buffer. Windows start at layers `0, commit, 2·commit, …`
//! until one reaches the last layer, which commits everything. Each window
//! is decoded on its part of the detector error model:
//!
//! * errors whose first detector lies in the window, with the detectors
//!   beyond the window dropped (the window's future end is a boundary);
//! * errors that flip a detector before the window are left out: the
//!   windows before resolved those detectors. Where the window's decoder
//!   then finds no correction (a defect carried forward whose only way
//!   out leads back, as when the only boundary of a detector is an error
//!   at the start), a fallback decoder with those errors too, their
//!   earlier detectors dropped, decodes the window; built when first
//!   needed, it changes no prediction the window's own decoder makes.
//!
//! Of the correction, the part that touches the commit region is applied:
//! its observables are added to the prediction and its detection events
//! flipped, including those in the buffer and beyond the window, which the
//! next window then sees (the syndrome is carried forward). Every detector
//! of a commit region is resolved this way. This is the sliding window of
//! overlapping recovery (Dennis, Kitaev, Landahl and Preskill, J. Math.
//! Phys. 43, 4452 (2002)), in the commit-and-buffer form of Skoric,
//! Browne, Barnes, Gillespie and Campbell (Nat. Commun. 14, 7040 (2023)),
//! here run window after window.
//!
//! For matching ([`WindowMethod::Matching`]) the correction is the set of
//! matched pairs of detection events: a pair is committed if one of its
//! events lies in the commit region; its path flips only its two events,
//! or for a match to the boundary its event and, when the boundary is the
//! window's future end, the detectors beyond the window of the error it
//! ends on. For BP+OSD ([`WindowMethod::BpOsd`]) the correction is a set of
//! errors; those with a detector in the commit region are committed with
//! all their detectors.
//!
//! Windows of the same shape (the same errors, probabilities and
//! observables relative to their first detector, as in the bulk of a
//! memory experiment) share one decoder, so memory grows with the shapes,
//! not with the rounds, besides the model itself and an index of the
//! errors of every window. A window that covers every layer is the model
//! itself, and the decoding equals that of the whole experiment.
//! [`WindowStream`] takes the detection events layer by layer and decodes
//! each window as soon as its layers are in.

use super::bposd::{BpOsdDecoder, BpOsdOptions};
use super::decoder::MatchingGraph;
use super::dem::{DetectorErrorModel, ErrorMechanism};
use super::sparse::SparseMatchingDecoder;
use crate::error::{QsoError, Result};
use std::collections::HashMap;

/// The decoder of every window.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[non_exhaustive]
pub enum WindowMethod {
    /// Minimum-weight perfect matching ([`SparseMatchingDecoder`]); every
    /// error must flip at most two detectors (a decomposed model).
    #[default]
    Matching,
    /// BP+OSD ([`BpOsdDecoder`]) with these options.
    BpOsd(BpOsdOptions),
}

/// Sizes of the windows, in layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowOptions {
    /// Layers a window covers.
    pub window: usize,
    /// Layers of each window committed (at most `window`); the rest is
    /// the buffer.
    pub commit: usize,
}

impl WindowOptions {
    /// A window of `window` layers committing half of them (rounded up).
    pub fn new(window: usize) -> Self {
        WindowOptions {
            window,
            commit: window.div_ceil(2),
        }
    }
}

/// One window: its layers and detectors, its decoder (shared by windows
/// of the same shape) and its errors (into the model's).
#[derive(Clone, Debug)]
struct Window {
    /// Its first layer.
    start: usize,
    /// Detectors `first..end`; those below `commit` form the commit
    /// region (all of them in the last window).
    first: usize,
    end: usize,
    commit: usize,
    decoder: usize,
    errors: Vec<u32>,
    /// The decoder and errors with those that reach before the window
    /// too (their earlier detectors dropped), built when the window's own
    /// decoder fails.
    fallback: Option<(usize, Vec<u32>)>,
}

/// A correction of a window: matched pairs or errors (into the window's
/// errors).
enum Correction {
    Pairs(Vec<super::sparse::Matched>),
    Errors(Vec<usize>),
}

#[derive(Clone, Debug)]
enum Inner {
    Matching(Box<SparseMatchingDecoder>),
    BpOsd(Box<BpOsdDecoder>),
}

/// A decoder that decodes a detector error model window by window (see
/// the module docs).
#[derive(Clone, Debug)]
pub struct WindowDecoder {
    options: WindowOptions,
    method: WindowMethod,
    n_detectors: usize,
    /// First detector of every layer, and `n_detectors` at the end.
    layer_first: Vec<usize>,
    /// Errors of probability above zero that flip a detector, in the
    /// model's order.
    errors: Vec<ErrorMechanism>,
    n_observables: usize,
    /// Errors by the layer of their first detector, and the most layers
    /// an error spans.
    by_layer: Vec<Vec<u32>>,
    span: usize,
    windows: Vec<Window>,
    decoders: Vec<Inner>,
    shapes: HashMap<Shape, usize>,
}

/// The model of a window in its own detector indices, the key of its
/// shape.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Shape {
    n_detectors: usize,
    errors: Vec<(u64, Vec<usize>, u64)>,
}

impl WindowDecoder {
    /// The window decoder of `dem`, whose detector `d` lies in layer
    /// `layers[d]` (non-decreasing with `d`, starting at 0, every layer
    /// up to the last holding a detector).
    pub fn new(
        dem: &DetectorErrorModel,
        layers: &[usize],
        options: WindowOptions,
        method: WindowMethod,
    ) -> Result<Self> {
        let n = dem.n_detectors;
        if layers.len() != n {
            return Err(QsoError::invalid(
                "layers",
                format!("{} layers for {n} detectors", layers.len()),
            ));
        }
        if options.window == 0 || options.commit == 0 || options.commit > options.window {
            return Err(QsoError::invalid(
                "window",
                format!(
                    "a window of {} layers cannot commit {} (need 1 ≤ commit ≤ window)",
                    options.window, options.commit
                ),
            ));
        }
        let mut layer_first = Vec::new();
        for (d, &l) in layers.iter().enumerate() {
            match l.cmp(&layer_first.len()) {
                std::cmp::Ordering::Less => {
                    if d > 0 && l < layers[d - 1] {
                        return Err(QsoError::invalid(
                            "layers",
                            format!("layers decrease at detector {d}"),
                        ));
                    }
                }
                std::cmp::Ordering::Equal => layer_first.push(d),
                std::cmp::Ordering::Greater => {
                    return Err(QsoError::invalid(
                        "layers",
                        format!("layer {} holds no detector", layer_first.len()),
                    ));
                }
            }
        }
        layer_first.push(n);
        let n_layers = layer_first.len() - 1;
        let mut errors = Vec::with_capacity(dem.errors.len());
        for e in &dem.errors {
            if !(0.0..=1.0).contains(&e.probability) {
                return Err(QsoError::invalid(
                    "dem",
                    format!("{} is not a probability", e.probability),
                ));
            }
            if let Some(&d) = e.detectors.iter().find(|&&d| d >= n) {
                return Err(QsoError::invalid(
                    "dem",
                    format!("an error flips detector {d} of {n}"),
                ));
            }
            if matches!(method, WindowMethod::Matching) && e.detectors.len() > 2 {
                return Err(QsoError::invalid(
                    "dem",
                    format!(
                        "an error flips {} detectors; matching needs at most two (a decomposed model)",
                        e.detectors.len()
                    ),
                ));
            }
            if e.probability > 0.0 && !e.detectors.is_empty() {
                let mut sorted = e.clone();
                sorted.detectors.sort_unstable();
                errors.push(sorted);
            }
        }
        // Errors by their first layer.
        let mut by_layer: Vec<Vec<u32>> = vec![Vec::new(); n_layers];
        let mut span = 0;
        for (k, e) in errors.iter().enumerate() {
            let (lo, hi) = (layers[e.detectors[0]], layers[*e.detectors.last().unwrap()]);
            by_layer[lo].push(k as u32);
            span = span.max(hi - lo);
        }
        let mut decoder = WindowDecoder {
            options,
            method,
            n_detectors: n,
            layer_first,
            errors,
            n_observables: dem.n_observables,
            by_layer,
            span,
            windows: Vec::new(),
            decoders: Vec::new(),
            shapes: HashMap::new(),
        };
        let mut start = 0;
        loop {
            let last = start + options.window >= n_layers;
            let end_layer = if last {
                n_layers
            } else {
                start + options.window
            };
            let commit_layer = if last {
                n_layers
            } else {
                start + options.commit
            };
            let (first, end) = (decoder.layer_first[start], decoder.layer_first[end_layer]);
            let mut ids: Vec<u32> = decoder.by_layer[start..end_layer].concat();
            ids.sort_unstable();
            let index = decoder.decoder_for(first, end, &ids)?;
            decoder.windows.push(Window {
                start,
                first,
                end,
                commit: decoder.layer_first[commit_layer],
                decoder: index,
                errors: ids,
                fallback: None,
            });
            if last {
                break;
            }
            start += options.commit;
        }
        Ok(decoder)
    }

    /// The decoder of the window of detectors `first..end` with these
    /// errors (their detectors outside the window dropped), shared with
    /// every window of the same shape.
    fn decoder_for(&mut self, first: usize, end: usize, ids: &[u32]) -> Result<usize> {
        let shape = Shape {
            n_detectors: end - first,
            errors: ids
                .iter()
                .map(|&k| {
                    let e = &self.errors[k as usize];
                    let local = e
                        .detectors
                        .iter()
                        .filter(|&&d| d >= first && d < end)
                        .map(|&d| d - first)
                        .collect();
                    (e.probability.to_bits(), local, e.observables)
                })
                .collect(),
        };
        if let Some(&i) = self.shapes.get(&shape) {
            return Ok(i);
        }
        let local = DetectorErrorModel {
            n_detectors: shape.n_detectors,
            n_observables: self.n_observables,
            errors: shape
                .errors
                .iter()
                .map(|(p, d, o)| ErrorMechanism {
                    probability: f64::from_bits(*p),
                    detectors: d.clone(),
                    observables: *o,
                })
                .collect(),
        };
        self.decoders.push(match self.method {
            WindowMethod::Matching => {
                let graph = MatchingGraph::from_dem(&local)?;
                // One edge per error, in order: matches name errors by
                // their edges.
                debug_assert_eq!(graph.edges().len(), ids.len());
                Inner::Matching(Box::new(SparseMatchingDecoder::new(graph)))
            }
            WindowMethod::BpOsd(o) => Inner::BpOsd(Box::new(BpOsdDecoder::new(&local, o)?)),
        });
        self.shapes.insert(shape, self.decoders.len() - 1);
        Ok(self.decoders.len() - 1)
    }

    /// The fallback of window `w`: its errors and those that reach before
    /// it with a detector in it, the window's past end a boundary too.
    fn fallback(&mut self, w: usize) -> Result<()> {
        if self.windows[w].fallback.is_some() {
            return Ok(());
        }
        let (start, first, end) = {
            let v = &self.windows[w];
            (v.start, v.first, v.end)
        };
        let mut ids = self.windows[w].errors.clone();
        for layer in start.saturating_sub(self.span)..start {
            for &k in &self.by_layer[layer] {
                let e = &self.errors[k as usize];
                if e.detectors.iter().any(|&d| d >= first && d < end) {
                    ids.push(k);
                }
            }
        }
        ids.sort_unstable();
        let index = self.decoder_for(first, end, &ids)?;
        self.windows[w].fallback = Some((index, ids));
        Ok(())
    }

    pub fn options(&self) -> WindowOptions {
        self.options
    }

    pub fn method(&self) -> WindowMethod {
        self.method
    }

    pub fn n_detectors(&self) -> usize {
        self.n_detectors
    }

    pub fn n_layers(&self) -> usize {
        self.layer_first.len() - 1
    }

    /// Windows the experiment is decoded in.
    pub fn n_windows(&self) -> usize {
        self.windows.len()
    }

    /// Distinct decoders the windows share.
    pub fn n_decoders(&self) -> usize {
        self.decoders.len()
    }

    /// Predicted observable flips for the fired detectors (a detector
    /// listed twice cancels), window by window.
    pub fn decode(&mut self, fired: &[usize]) -> Result<u64> {
        let n = self.n_detectors;
        if let Some(&d) = fired.iter().find(|&&d| d >= n) {
            return Err(QsoError::invalid(
                "fired",
                format!("detector {d} does not exist ({n} detectors)"),
            ));
        }
        let mut state = self.new_stream();
        for &d in fired {
            state.syndrome[d] ^= true;
        }
        state.layers = self.n_layers();
        self.decode_ready(&mut state)?;
        Ok(state.observables)
    }

    /// A stream of detection events, layer by layer, borrowing the
    /// decoder.
    pub fn stream(&mut self) -> WindowStream<'_> {
        let state = self.new_stream();
        WindowStream {
            decoder: self,
            state,
        }
    }

    /// The state of a stream that holds no layer yet (see
    /// [`WindowDecoder::push_layer`]).
    pub fn new_stream(&self) -> StreamState {
        StreamState {
            syndrome: vec![false; self.n_detectors],
            layers: 0,
            next: 0,
            observables: 0,
            failed: false,
        }
    }

    /// The detection events of the stream's next layer (detector indices
    /// of the model, all in that layer; a detector listed twice cancels).
    /// Decodes the windows this completes and returns the observable flips
    /// committed so far.
    pub fn push_layer(&mut self, state: &mut StreamState, fired: &[usize]) -> Result<u64> {
        let n_layers = self.n_layers();
        if state.failed {
            return Err(QsoError::invalid(
                "stream",
                "a window of this stream failed to decode; start a new stream",
            ));
        }
        if state.syndrome.len() != self.n_detectors || state.layers >= n_layers {
            return Err(QsoError::invalid(
                "layer",
                format!("the stream holds {} of {n_layers} layers", state.layers),
            ));
        }
        let (lo, hi) = (
            self.layer_first[state.layers],
            self.layer_first[state.layers + 1],
        );
        if let Some(&d) = fired.iter().find(|&&d| d < lo || d >= hi) {
            return Err(QsoError::invalid(
                "fired",
                format!("detector {d} is not in layer {} ({lo}..{hi})", state.layers),
            ));
        }
        for &d in fired {
            state.syndrome[d] ^= true;
        }
        state.layers += 1;
        self.decode_ready(state)?;
        Ok(state.observables)
    }

    /// The prediction of a stream that holds every layer and decoded every
    /// window.
    pub fn finish(&self, state: &StreamState) -> Result<u64> {
        let n_layers = self.n_layers();
        if state.failed {
            return Err(QsoError::invalid(
                "stream",
                "a window of this stream failed to decode",
            ));
        }
        if state.layers < n_layers || state.next < self.windows.len() {
            return Err(QsoError::invalid(
                "layer",
                format!("{} of {n_layers} layers were given", state.layers),
            ));
        }
        Ok(state.observables)
    }

    /// Decode the windows whose layers are all in.
    fn decode_ready(&mut self, state: &mut StreamState) -> Result<()> {
        let mut events = Vec::new();
        while state.next < self.windows.len() {
            if self.windows[state.next].end > self.layer_first[state.layers] {
                break;
            }
            match self.decode_window(state.next, &mut state.syndrome, &mut events) {
                Ok(o) => state.observables ^= o,
                Err(e) => {
                    state.failed = true;
                    return Err(e);
                }
            }
            state.next += 1;
        }
        Ok(())
    }

    /// The correction of window `w` for its detection events, by its own
    /// decoder or the fallback's.
    fn correct(&mut self, w: usize, fallback: bool, events: &[usize]) -> Result<Correction> {
        let window = &self.windows[w];
        let index = if fallback {
            window.fallback.as_ref().expect("built").0
        } else {
            window.decoder
        };
        Ok(match &mut self.decoders[index] {
            Inner::Matching(m) => Correction::Pairs(m.decode_matches(events)?),
            Inner::BpOsd(b) => Correction::Errors(b.decode_detailed(events)?.errors),
        })
    }

    /// Decode window `w` on `syndrome` and apply its commit: returns the
    /// observables it commits. When the window's decoder finds no
    /// correction (the carried syndrome may need the window's past end as
    /// a boundary, as when errors at the start are the only boundary), the
    /// fallback with the errors that reach before the window decodes it.
    fn decode_window(
        &mut self,
        w: usize,
        syndrome: &mut [bool],
        events: &mut Vec<usize>,
    ) -> Result<u64> {
        let (first, end) = (self.windows[w].first, self.windows[w].end);
        events.clear();
        events.extend((first..end).filter(|&d| syndrome[d]).map(|d| d - first));
        if events.is_empty() {
            return Ok(0);
        }
        let (correction, fallback) = match self.correct(w, false, events) {
            Ok(c) => (c, false),
            Err(error) => {
                self.fallback(w)?;
                match self.correct(w, true, events) {
                    Ok(c) => (c, true),
                    Err(_) => return Err(error),
                }
            }
        };
        let window = &self.windows[w];
        let ids = if fallback {
            &window.fallback.as_ref().expect("built").1
        } else {
            &window.errors
        };
        let commit = window.commit - first;
        let mut observables = 0;
        match correction {
            Correction::Pairs(pairs) => {
                for pair in pairs {
                    let (u, v) = (pair.u as usize, pair.v);
                    let boundary = v == u32::MAX;
                    if u >= commit && (boundary || v as usize >= commit) {
                        continue;
                    }
                    observables ^= pair.obs;
                    syndrome[first + u] ^= true;
                    if !boundary {
                        syndrome[first + v as usize] ^= true;
                    } else {
                        // The error it reaches the boundary through: its
                        // detectors beyond the window, if any (those
                        // before it are resolved and left alone).
                        let e = &self.errors[ids[pair.via as usize] as usize];
                        for &d in e.detectors.iter().filter(|&&d| d >= end) {
                            syndrome[d] ^= true;
                        }
                    }
                }
            }
            Correction::Errors(errors) => {
                for k in errors {
                    let e = &self.errors[ids[k] as usize];
                    // Its first detector in the window.
                    let inside = e.detectors.iter().find(|&&d| d >= first);
                    if inside.is_none_or(|&d| d >= window.commit) {
                        continue;
                    }
                    observables ^= e.observables;
                    for &d in e.detectors.iter().filter(|&&d| d >= first) {
                        syndrome[d] ^= true;
                    }
                }
            }
        }
        debug_assert!(
            (first..window.commit).all(|d| !syndrome[d]),
            "the commit region of window {w} is resolved"
        );
        Ok(observables)
    }
}

impl super::Decoder for WindowDecoder {
    fn decode(&mut self, fired: &[usize]) -> Result<u64> {
        WindowDecoder::decode(self, fired)
    }
}

/// Where a stream of detection events stands: the syndrome so far
/// (with the commits applied), the layers given, the next window and the
/// observables committed.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamState {
    syndrome: Vec<bool>,
    layers: usize,
    next: usize,
    observables: u64,
    /// A window failed to decode: the stream is void.
    failed: bool,
}

impl StreamState {
    /// The observable flips committed so far.
    pub fn committed(&self) -> u64 {
        self.observables
    }

    /// Layers given so far.
    pub fn layers(&self) -> usize {
        self.layers
    }
}

/// Detection events given layer by layer to a [`WindowDecoder`], each
/// window decoded as soon as its layers are in (see
/// [`WindowDecoder::stream`]).
#[derive(Debug)]
pub struct WindowStream<'a> {
    decoder: &'a mut WindowDecoder,
    state: StreamState,
}

impl WindowStream<'_> {
    /// See [`WindowDecoder::push_layer`].
    pub fn push_layer(&mut self, fired: &[usize]) -> Result<u64> {
        self.decoder.push_layer(&mut self.state, fired)
    }

    /// The observable flips committed so far.
    pub fn committed(&self) -> u64 {
        self.state.observables
    }

    /// Layers given so far.
    pub fn layers(&self) -> usize {
        self.state.layers
    }

    /// The prediction, once every layer was given.
    pub fn finish(self) -> Result<u64> {
        self.decoder.finish(&self.state)
    }
}

#[cfg(test)]
mod tests;
