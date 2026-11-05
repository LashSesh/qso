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

/// Sizes of the windows of parallel decoding ([`WindowDecoder::parallel`]),
/// in layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParallelWindowOptions {
    /// Layers each window of the first pass commits (at least 1).
    pub commit: usize,
    /// Layers of buffer on each side of them.
    pub buffer: usize,
}

/// How a window takes its errors and applies its commit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// One after another: errors whose first detector lies in it, the
    /// future end a boundary.
    Sliding,
    /// The first pass of parallel decoding: every error with a detector in
    /// it, both ends boundaries, decoded on the detection events as
    /// sampled.
    First,
    /// The second pass: the seam between two committed regions, with the
    /// errors inside it only.
    Seam,
}

/// One window: its layers and detectors, its decoder (shared by windows
/// of the same shape) and its errors (into the model's).
#[derive(Clone, Debug)]
struct Window {
    kind: Kind,
    /// Its first layer.
    start: usize,
    /// Detectors `first..end`; those in `commit_start..commit` form the
    /// commit region (for sliding windows from `first`, all of them in the
    /// last window).
    first: usize,
    end: usize,
    commit_start: usize,
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
    parallel: Option<ParallelWindowOptions>,
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
        if options.window == 0 || options.commit == 0 || options.commit > options.window {
            return Err(QsoError::invalid(
                "window",
                format!(
                    "a window of {} layers cannot commit {} (need 1 ≤ commit ≤ window)",
                    options.window, options.commit
                ),
            ));
        }
        let mut decoder = Self::empty(dem, layers, options, None, method)?;
        let n_layers = decoder.n_layers();
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
            decoder.add_window(Kind::Sliding, start, start, commit_layer, end_layer)?;
            if last {
                break;
            }
            start += options.commit;
        }
        Ok(decoder)
    }

    /// The parallel window decoder of `dem` (layers as for
    /// [`WindowDecoder::new`]), in two passes as Skoric et al. decode
    /// (Nat. Commun. 14, 7040 (2023)). The first pass decodes windows of
    /// `buffer + commit + buffer` layers, each on the detection events as
    /// sampled with both its ends as boundaries, and commits their middle
    /// `commit` layers; between two of these lie `buffer + span` layers,
    /// `span` the most layers an error spans, so that no window of the pass
    /// changes what another commits, and they run in parallel. The second
    /// pass decodes the seams between the committed regions, with the
    /// detection events the first left, on the errors within each seam,
    /// and commits all of it; the seams too run in parallel. A stream
    /// ([`WindowDecoder::push_layer`]) decodes the same windows in order
    /// as their layers come in, with the same prediction.
    pub fn parallel(
        dem: &DetectorErrorModel,
        layers: &[usize],
        options: ParallelWindowOptions,
        method: WindowMethod,
    ) -> Result<Self> {
        let (c, b) = (options.commit, options.buffer);
        if c == 0 {
            return Err(QsoError::invalid(
                "commit",
                "a window of the first pass commits at least one layer",
            ));
        }
        let window = c.saturating_add(b.saturating_mul(2));
        let mut decoder = Self::empty(
            dem,
            layers,
            WindowOptions { window, commit: c },
            Some(options),
            method,
        )?;
        let n_layers = decoder.n_layers();
        let gap = b.saturating_add(decoder.span);
        let period = c.saturating_add(gap);
        // First-pass windows and the seams after them, in the order a
        // stream decodes them: a seam once the window after it is done.
        let mut first = Vec::new();
        let mut at = 0usize;
        while at < n_layers {
            let commit_end = at.saturating_add(c).min(n_layers);
            first.push((
                at.saturating_sub(b),
                at,
                commit_end,
                commit_end.saturating_add(b).min(n_layers),
            ));
            at = at.saturating_add(period);
        }
        for (i, &(lo, clo, chi, hi)) in first.iter().enumerate() {
            decoder.add_window(Kind::First, lo, clo, chi, hi)?;
            if i > 0 {
                let seam = (first[i - 1].2, clo);
                if seam.0 < seam.1 {
                    decoder.add_window(Kind::Seam, seam.0, seam.0, seam.1, seam.1)?;
                }
            }
        }
        let tail = first.last().map_or(0, |w| w.2);
        if tail < n_layers {
            decoder.add_window(Kind::Seam, tail, tail, n_layers, n_layers)?;
        }
        Ok(decoder)
    }

    /// A decoder of `dem` without windows yet.
    fn empty(
        dem: &DetectorErrorModel,
        layers: &[usize],
        options: WindowOptions,
        parallel: Option<ParallelWindowOptions>,
        method: WindowMethod,
    ) -> Result<Self> {
        let n = dem.n_detectors;
        if layers.len() != n {
            return Err(QsoError::invalid(
                "layers",
                format!("{} layers for {n} detectors", layers.len()),
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
        Ok(WindowDecoder {
            options,
            parallel,
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
        })
    }

    /// The errors of a window of layers `start..end_layer` of this kind
    /// (without its fallback's).
    fn window_errors(&self, kind: Kind, start: usize, end_layer: usize) -> Vec<u32> {
        let (first, end) = (self.layer_first[start], self.layer_first[end_layer]);
        let mut ids: Vec<u32> = match kind {
            Kind::Sliding => self.by_layer[start..end_layer].concat(),
            Kind::First => self.by_layer[start.saturating_sub(self.span)..end_layer]
                .iter()
                .flatten()
                .copied()
                .filter(|&k| {
                    let e = &self.errors[k as usize];
                    e.detectors.iter().any(|&d| d >= first && d < end)
                })
                .collect(),
            Kind::Seam => self.by_layer[start..end_layer]
                .iter()
                .flatten()
                .copied()
                .filter(|&k| *self.errors[k as usize].detectors.last().unwrap() < end)
                .collect(),
        };
        ids.sort_unstable();
        ids
    }

    /// Add the window of layers `start..end_layer` committing
    /// `commit_start..commit_layer`.
    fn add_window(
        &mut self,
        kind: Kind,
        start: usize,
        commit_start: usize,
        commit_layer: usize,
        end_layer: usize,
    ) -> Result<()> {
        let (first, end) = (self.layer_first[start], self.layer_first[end_layer]);
        let ids = self.window_errors(kind, start, end_layer);
        let index = self.decoder_for(first, end, &ids)?;
        self.windows.push(Window {
            kind,
            start,
            first,
            end,
            commit_start: self.layer_first[commit_start],
            commit: self.layer_first[commit_layer],
            decoder: index,
            errors: ids,
            fallback: None,
        });
        Ok(())
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
        let (kind, start, first, end) = {
            let v = &self.windows[w];
            (v.kind, v.start, v.first, v.end)
        };
        let mut ids = self.windows[w].errors.clone();
        let touches = |e: &ErrorMechanism| e.detectors.iter().any(|&d| d >= first && d < end);
        match kind {
            Kind::Sliding => {
                for layer in start.saturating_sub(self.span)..start {
                    for &k in &self.by_layer[layer] {
                        if touches(&self.errors[k as usize]) {
                            ids.push(k);
                        }
                    }
                }
            }
            // The errors that reach into the committed regions too, those
            // detectors dropped.
            Kind::Seam => {
                let end_layer = self.layer_of(end);
                for layer in start.saturating_sub(self.span)..end_layer {
                    for &k in &self.by_layer[layer] {
                        let e = &self.errors[k as usize];
                        let inside = e.detectors[0] >= first && *e.detectors.last().unwrap() < end;
                        if touches(e) && !inside {
                            ids.push(k);
                        }
                    }
                }
            }
            // A window of the first pass has every error touching it.
            Kind::First => {}
        }
        ids.sort_unstable();
        let index = self.decoder_for(first, end, &ids)?;
        self.windows[w].fallback = Some((index, ids));
        Ok(())
    }

    /// The sizes of sliding windows; for a parallel decoder the windows of
    /// its first pass (`buffer + commit + buffer` layers committing
    /// `commit`).
    pub fn options(&self) -> WindowOptions {
        self.options
    }

    /// The sizes of a parallel decoder ([`WindowDecoder::parallel`]).
    pub fn parallel_options(&self) -> Option<ParallelWindowOptions> {
        self.parallel
    }

    /// The layer whose first detector is `d` (the layer count for the end).
    fn layer_of(&self, d: usize) -> usize {
        self.layer_first.partition_point(|&f| f < d)
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
    /// listed twice cancels), window by window; for a parallel decoder the
    /// windows of each pass in parallel.
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
        if self.parallel.is_some() {
            self.decode_passes(&mut state)?;
        } else {
            self.decode_ready(&mut state)?;
        }
        Ok(state.observables)
    }

    /// Both passes of a parallel decoder, each window of a pass on its own
    /// thread's copy of the decoders; windows whose decoder fails are
    /// decoded again in order with their fallback.
    fn decode_passes(&mut self, state: &mut StreamState) -> Result<()> {
        state.original.clone_from(&state.syndrome);
        for kind in [Kind::First, Kind::Seam] {
            let ws: Vec<usize> = (0..self.windows.len())
                .filter(|&w| self.windows[w].kind == kind)
                .collect();
            let threads = crate::parallel::threads().max(1);
            let chunk = ws.len().div_ceil(threads).max(1);
            let this = &*self;
            let syndrome = &state.syndrome;
            let original = &state.original;
            let done: Vec<Vec<(usize, Result<Commit>)>> =
                crate::parallel::map_tasks(ws.len().div_ceil(chunk), |c| {
                    let mut decoders = this.decoders.clone();
                    let mut events = Vec::new();
                    ws[c * chunk..((c + 1) * chunk).min(ws.len())]
                        .iter()
                        .map(|&w| {
                            let source = if kind == Kind::First {
                                original
                            } else {
                                syndrome
                            };
                            (
                                w,
                                this.commit_of(w, false, source, &mut decoders, &mut events),
                            )
                        })
                        .collect()
                });
            let mut failed = Vec::new();
            for (w, commit) in done.into_iter().flatten() {
                match commit {
                    Ok(c) => state.apply(&c),
                    Err(_) => failed.push(w),
                }
            }
            for w in failed {
                let c = self.commit_with_fallback(w, &state.syndrome, &state.original)?;
                state.apply(&c);
            }
            for &w in &ws {
                self.check_resolved(w, &state.syndrome);
            }
        }
        state.next = self.windows.len();
        Ok(())
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
            original: if self.parallel.is_some() {
                vec![false; self.n_detectors]
            } else {
                Vec::new()
            },
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
            if let Some(o) = state.original.get_mut(d) {
                *o ^= true;
            }
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
        while state.next < self.windows.len() {
            if self.windows[state.next].end > self.layer_first[state.layers] {
                break;
            }
            let w = state.next;
            match self.commit_with_fallback(w, &state.syndrome, &state.original) {
                Ok(c) => state.apply(&c),
                Err(e) => {
                    state.failed = true;
                    return Err(e);
                }
            }
            self.check_resolved(w, &state.syndrome);
            state.next += 1;
        }
        Ok(())
    }

    /// In debug builds: the commit region of window `w` is resolved.
    fn check_resolved(&self, w: usize, syndrome: &[bool]) {
        let window = &self.windows[w];
        debug_assert!(
            (window.commit_start..window.commit).all(|d| !syndrome[d]),
            "the commit region of window {w} is resolved"
        );
    }

    /// The commit of window `w` by its own decoder or, when that finds no
    /// correction (the carried syndrome may need the window's past end as
    /// a boundary, as when errors at the start are the only boundary), by
    /// its fallback, built when first needed.
    fn commit_with_fallback(
        &mut self,
        w: usize,
        syndrome: &[bool],
        original: &[bool],
    ) -> Result<Commit> {
        let source = if self.windows[w].kind == Kind::First {
            original
        } else {
            syndrome
        };
        let mut events = Vec::new();
        let mut decoders = std::mem::take(&mut self.decoders);
        let own = self.commit_of(w, false, source, &mut decoders, &mut events);
        self.decoders = decoders;
        let Err(error) = own else { return own };
        self.fallback(w)?;
        let mut decoders = std::mem::take(&mut self.decoders);
        let fallback = self.commit_of(w, true, source, &mut decoders, &mut events);
        self.decoders = decoders;
        fallback.map_err(|_| error)
    }

    /// The commit of window `w` on the detection events of `source`, by
    /// its own decoder or (`fallback`, built) the fallback's, taken from
    /// `decoders` (the decoder's own or a thread's copy). A matched pair is
    /// committed when one of its events lies in the commit region and
    /// flips its events, or for a match to the boundary its event and the
    /// detectors outside the window of the error it ends on, where those
    /// are not resolved (after a sliding window, both sides of a window of
    /// the first pass, never for a seam); an error of BP+OSD is committed
    /// when one of its detectors lies in the commit region, with the same
    /// detectors.
    fn commit_of(
        &self,
        w: usize,
        fallback: bool,
        source: &[bool],
        decoders: &mut [Inner],
        events: &mut Vec<usize>,
    ) -> Result<Commit> {
        let window = &self.windows[w];
        let (first, end) = (window.first, window.end);
        events.clear();
        events.extend((first..end).filter(|&d| source[d]).map(|d| d - first));
        let mut c = Commit::default();
        if events.is_empty() {
            return Ok(c);
        }
        let (index, ids) = if fallback {
            let f = window.fallback.as_ref().expect("built");
            (f.0, &f.1)
        } else {
            (window.decoder, &window.errors)
        };
        let correction = match &mut decoders[index] {
            Inner::Matching(m) => Correction::Pairs(m.decode_matches(events)?),
            Inner::BpOsd(b) => Correction::Errors(b.decode_detailed(events)?.errors),
        };
        let (lo, hi) = (window.commit_start, window.commit);
        let in_commit = |d: usize| d >= lo && d < hi;
        let flips = |d: usize| {
            (d >= first && d < end)
                || match window.kind {
                    Kind::Sliding => d >= end,
                    Kind::First => true,
                    Kind::Seam => false,
                }
        };
        match correction {
            Correction::Pairs(pairs) => {
                for pair in pairs {
                    let u = first + pair.u as usize;
                    let v = (pair.v != u32::MAX).then(|| first + pair.v as usize);
                    if !in_commit(u) && !v.is_some_and(in_commit) {
                        continue;
                    }
                    c.observables ^= pair.obs;
                    c.flips.push(u);
                    match v {
                        Some(v) => c.flips.push(v),
                        None => {
                            // The error it reaches the boundary through.
                            let e = &self.errors[ids[pair.via as usize] as usize];
                            c.flips.extend(
                                e.detectors
                                    .iter()
                                    .copied()
                                    .filter(|&d| (d < first || d >= end) && flips(d)),
                            );
                        }
                    }
                }
            }
            Correction::Errors(errors) => {
                for k in errors {
                    let e = &self.errors[ids[k] as usize];
                    if !e.detectors.iter().any(|&d| in_commit(d)) {
                        continue;
                    }
                    c.observables ^= e.observables;
                    c.flips
                        .extend(e.detectors.iter().copied().filter(|&d| flips(d)));
                }
            }
        }
        Ok(c)
    }
}

/// What a window commits: observables, and the detection events it flips.
#[derive(Clone, Debug, Default)]
struct Commit {
    observables: u64,
    flips: Vec<usize>,
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
    /// The detection events as given (for the first pass of a parallel
    /// decoder; empty otherwise).
    original: Vec<bool>,
    layers: usize,
    next: usize,
    observables: u64,
    /// A window failed to decode: the stream is void.
    failed: bool,
}

impl StreamState {
    fn apply(&mut self, c: &Commit) {
        self.observables ^= c.observables;
        for &d in &c.flips {
            self.syndrome[d] ^= true;
        }
    }

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
