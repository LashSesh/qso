//! The table of strings and the passes over it.
//!
//! A table holds its strings as structure of arrays: `2·W` words per
//! string (the x words, then the z words, on the compact labels of the
//! light cone) and one coefficient.
//!
//! * An **in-place pass** (bijections only) rewrites every string of a
//!   chunk of [`CHUNK`] rows on its own, in parallel over the chunks.
//! * A **splitting pass** reads the table in batches of
//!   [`BATCH_CHUNKS`] chunks. Phase A (parallel over the chunks) drops
//!   strings below the truncation thresholds, expands every other string
//!   into its images under the level's ops and sorts the images by shard
//!   (the top six bits of a fixed hash of the key). Phase B (parallel over
//!   the 64 shards) inserts the images of every chunk, in chunk order,
//!   into the shard's open-addressing index, adding coefficients of equal
//!   strings. Every string lives in one shard and is summed in (chunk,
//!   row, image) order, chunks and batches have fixed sizes, and the
//!   shards are concatenated in shard order, so the result is
//!   bit-identical for any number of threads and in the serial build.
//!
//! At the end of a splitting pass a string whose coefficient is 0 or at
//! most `2⁻⁵⁰` of the sum of the magnitudes added into it is an exact
//! cancellation up to rounding; it is dropped and its magnitude counted in
//! the truncation bound.

use super::PropagationOptions;
use super::compile::{Level, Pass, Plan, scatter};
use super::operator::weight;
use crate::error::{QsoError, Result};
use crate::parallel::{self, CHUNK};

/// Shards of a splitting pass (the top bits of the hash pick one).
const SHARD_BITS: u32 = 6;
const SHARDS: usize = 1 << SHARD_BITS;

/// Chunks of one batch of a splitting pass.
pub(crate) const BATCH_CHUNKS: usize = 16;

/// Fewer rows (or images to merge) than this are processed on the calling
/// thread: dispatching the tasks of a level costs tens of microseconds,
/// more than a small table takes. The work and its order are the same
/// either way, so the result does not depend on it.
const PARALLEL_ROWS: usize = CHUNK;

/// [`parallel::map_tasks`] when `parallel`, else in order on this thread.
fn map_tasks<T: Send>(parallel: bool, n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    if parallel {
        parallel::map_tasks(n, f)
    } else {
        (0..n).map(f).collect()
    }
}

/// [`parallel::for_each_task_mut`] when `parallel`, else in order on this
/// thread.
fn for_each_task_mut<T: Send>(
    parallel: bool,
    items: &mut [T],
    f: impl Fn(usize, &mut T) + Sync + Send,
) {
    if parallel {
        parallel::for_each_task_mut(items, f);
    } else {
        for (i, x) in items.iter_mut().enumerate() {
            f(i, x);
        }
    }
}

/// A string whose coefficient is at most this fraction of the magnitudes
/// summed into it is a cancellation (`2⁻⁵⁰`, about 4.5 times the unit
/// round-off of a sum of a few terms).
const CANCELLATION: f64 = 1.0 / (1u64 << 50) as f64;

/// A fixed hash of a key: multiply–xorshift over the words and the
/// splitmix64 finaliser (no random state, so shards are deterministic).
#[inline]
pub(crate) fn hash(key: &[u64]) -> u64 {
    let mut h = 0u64;
    for &word in key {
        h = (h ^ word).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 29;
    }
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 31)
}

/// The result of running a plan.
#[derive(Debug)]
pub(crate) struct Outcome {
    pub(crate) value: f64,
    pub(crate) bound: f64,
    pub(crate) peak: usize,
    pub(crate) dropped: usize,
    /// The final table on compact labels.
    pub(crate) keys: Vec<u64>,
    pub(crate) coefs: Vec<f64>,
}

/// The truncation test of a string at the input of a splitting pass.
#[derive(Clone, Copy, Debug)]
struct Truncation {
    min_coefficient: f64,
    max_weight: usize,
    max_terms: usize,
    words: usize,
}

impl Truncation {
    #[inline]
    fn drops(&self, key: &[u64], c: f64) -> bool {
        c == 0.0
            || c.abs() < self.min_coefficient
            || (self.max_weight < usize::MAX && weight(key, self.words) > self.max_weight)
    }
}

fn too_many(pass: usize, held: usize) -> QsoError {
    QsoError::invalid(
        "max_terms",
        format!("pass {pass} held {held} strings; truncate or raise max_terms"),
    )
}

/// Run `plan` and read the value in the product state with Bloch vectors
/// `bloch` on the compact labels (`None`: `|0…0⟩`).
pub(crate) fn run(
    plan: &Plan,
    bloch: Option<&[[f64; 3]]>,
    options: &PropagationOptions,
) -> Result<Outcome> {
    let w = plan.words;
    let r = 2 * w;
    let cut = Truncation {
        min_coefficient: options.min_coefficient,
        max_weight: options.max_weight.unwrap_or(usize::MAX),
        max_terms: options.max_terms,
        words: w,
    };
    // Pass 0: the observable itself.
    let mut keys = Vec::with_capacity(plan.keys.len());
    let mut coefs = Vec::with_capacity(plan.coefs.len());
    let (mut partial, mut dropped) = (0.0, 0usize);
    for (key, &c) in plan.keys.chunks_exact(r).zip(&plan.coefs) {
        if cut.drops(key, c) {
            partial += c.abs();
            dropped += 1;
        } else {
            keys.extend_from_slice(key);
            coefs.push(c);
        }
    }
    if coefs.len() > options.max_terms {
        return Err(too_many(0, coefs.len()));
    }
    let mut bound = plan.gamma * partial;
    let mut peak = coefs.len();
    for (p, pass) in plan.passes.iter().enumerate() {
        let partial = if pass.in_place {
            in_place(&mut keys, &mut coefs, pass, w)
        } else {
            let out = split(&keys, &coefs, &pass.levels[0], cut, p + 1)?;
            keys = out.keys;
            coefs = out.coefs;
            dropped += out.dropped;
            peak = peak.max(out.held);
            out.bound
        };
        bound += pass.gamma * partial;
    }
    let value = readout(&keys, &coefs, w, bloch);
    // Sums of magnitudes (and products with the factors of inverted
    // layers) can overflow even when the observable times Γ does not.
    if !(value.is_finite() && bound.is_finite() && coefs.iter().all(|c| c.is_finite())) {
        return Err(QsoError::invalid(
            "observable",
            "a coefficient, the value or the truncation bound overflowed floating point; \
             scale the observable down",
        ));
    }
    Ok(Outcome {
        value,
        bound,
        peak,
        dropped,
        keys,
        coefs,
    })
}

/// One chunk of an in-place pass and its share of the bound.
type InPlaceTask<'a> = (&'a mut [u64], &'a mut [f64], f64);

/// Apply an in-place pass; returns the snapped magnitude.
fn in_place(keys: &mut [u64], coefs: &mut [f64], pass: &Pass, w: usize) -> f64 {
    let r = 2 * w;
    let mut tasks: Vec<InPlaceTask<'_>> = keys
        .chunks_mut(CHUNK * r)
        .zip(coefs.chunks_mut(CHUNK))
        .map(|(k, c)| (k, c, 0.0))
        .collect();
    let parallel = tasks.len() > 1;
    for_each_task_mut(parallel, &mut tasks, |_, (keys, coefs, bound)| {
        let mut active = Vec::new();
        for (key, c) in keys.chunks_exact_mut(r).zip(coefs.iter_mut()) {
            for level in &pass.levels {
                // Pauli–Lindblad layers act on qubits that no gate of the
                // level touches, so the key before the gates serves.
                let mut f = 1.0;
                for d in &level.diags {
                    f *= d.fidelity(key);
                }
                level.active(key, w, &mut active);
                for &(o, b) in &active {
                    let op = &level.locals[o as usize];
                    let (image, g) = op.map.images(b as usize)[0];
                    *bound += c.abs() * op.map.residual(b as usize);
                    scatter(key, w, op.qubits(), image as usize);
                    *c *= g;
                }
                *c *= f;
            }
        }
    });
    tasks.iter().map(|t| t.2).sum()
}

/// The images of one chunk of a splitting pass, ordered by shard.
struct ChunkOut {
    keys: Vec<u64>,
    coefs: Vec<f64>,
    hashes: Vec<u64>,
    /// Image indices by shard: shard `s` owns `order[start[s]..start[s + 1]]`,
    /// in emission order.
    order: Vec<u32>,
    start: Vec<usize>,
    bound: f64,
    dropped: usize,
}

/// Phase A of a splitting pass on one chunk.
fn expand_chunk(
    keys: &[u64],
    coefs: &[f64],
    level: &Level,
    cut: Truncation,
    pass: usize,
) -> Result<ChunkOut> {
    let w = cut.words;
    let r = 2 * w;
    let mut out_keys: Vec<u64> = Vec::with_capacity(2 * keys.len());
    let mut out_coefs: Vec<f64> = Vec::with_capacity(2 * coefs.len());
    let (mut bound, mut dropped) = (0.0, 0usize);
    let mut active = Vec::new();
    let (mut ak, mut ac, mut bk, mut bc): (Vec<u64>, Vec<f64>, Vec<u64>, Vec<f64>) =
        Default::default();
    for (key, &c) in keys.chunks_exact(r).zip(coefs) {
        if cut.drops(key, c) {
            bound += c.abs();
            dropped += 1;
            continue;
        }
        let mut c = c;
        for d in &level.diags {
            c *= d.fidelity(key);
        }
        level.active(key, w, &mut active);
        if active.is_empty() {
            out_keys.extend_from_slice(key);
            out_coefs.push(c);
            continue;
        }
        ak.clear();
        ac.clear();
        ak.extend_from_slice(key);
        ac.push(c);
        for &(o, b) in &active {
            let op = &level.locals[o as usize];
            let images = op.map.images(b as usize);
            let residual = op.map.residual(b as usize);
            if residual > 0.0 {
                bound += residual * ac.iter().map(|c| c.abs()).sum::<f64>();
            }
            let count = ac.len() * images.len();
            if count > cut.max_terms {
                return Err(too_many(pass, count));
            }
            bk.clear();
            bc.clear();
            for (src, &cs) in ak.chunks_exact(r).zip(&ac) {
                for &(image, f) in images {
                    let at = bk.len();
                    bk.extend_from_slice(src);
                    scatter(&mut bk[at..], w, op.qubits(), image as usize);
                    bc.push(cs * f);
                }
            }
            std::mem::swap(&mut ak, &mut bk);
            std::mem::swap(&mut ac, &mut bc);
        }
        out_keys.extend_from_slice(&ak);
        out_coefs.extend_from_slice(&ac);
        // The images of one chunk are held at once too: stop before they
        // outgrow the limit (each row alone is checked above).
        if out_coefs.len() > cut.max_terms {
            return Err(too_many(pass, out_coefs.len()));
        }
    }
    // Sort the images by shard (stable, so in emission order per shard).
    let hashes: Vec<u64> = out_keys.chunks_exact(r).map(hash).collect();
    let mut start = vec![0usize; SHARDS + 1];
    for &h in &hashes {
        start[(h >> (64 - SHARD_BITS)) as usize + 1] += 1;
    }
    for s in 0..SHARDS {
        start[s + 1] += start[s];
    }
    let mut next = start.clone();
    let mut order = vec![0u32; hashes.len()];
    for (i, &h) in hashes.iter().enumerate() {
        let s = (h >> (64 - SHARD_BITS)) as usize;
        order[next[s]] = i as u32;
        next[s] += 1;
    }
    Ok(ChunkOut {
        keys: out_keys,
        coefs: out_coefs,
        hashes,
        order,
        start,
        bound,
        dropped,
    })
}

/// One shard of the output of a splitting pass: rows and their index.
struct Shard {
    keys: Vec<u64>,
    coefs: Vec<f64>,
    /// Sum of the magnitudes added into each row.
    abs: Vec<f64>,
    /// Open addressing with linear probing: 0 is empty, else
    /// `tag << 32 | (row + 1)` with `tag` = bits 26..58 of the hash.
    slots: Vec<u64>,
    /// Slots of the index when it is allocated.
    capacity: usize,
    /// Words per row.
    r: usize,
    bound: f64,
    dropped: usize,
}

impl Shard {
    /// An empty shard whose index, allocated on the first insert, holds
    /// `rows` rows before it grows.
    fn new(r: usize, rows: usize) -> Self {
        Shard {
            keys: Vec::new(),
            coefs: Vec::new(),
            abs: Vec::new(),
            slots: Vec::new(),
            capacity: (2 * rows).next_power_of_two().max(16),
            r,
            bound: 0.0,
            dropped: 0,
        }
    }

    fn len(&self) -> usize {
        self.coefs.len()
    }

    #[inline]
    fn place(slots: &mut [u64], h: u64, row: usize) {
        let mask = slots.len() - 1;
        let mut i = h as usize & mask;
        while slots[i] != 0 {
            i = (i + 1) & mask;
        }
        slots[i] = ((h >> 26) & 0xFFFF_FFFF) << 32 | (row as u64 + 1);
    }

    /// Double the index (load at most one half).
    fn grow(&mut self) {
        let mut slots = vec![0u64; 2 * self.slots.len()];
        for (row, key) in self.keys.chunks_exact(self.r).enumerate() {
            Self::place(&mut slots, hash(key), row);
        }
        self.slots = slots;
    }

    /// Add `c` to the row of `key` (hash `h`), appending it if new.
    #[inline]
    fn insert(&mut self, key: &[u64], h: u64, c: f64) {
        if self.slots.is_empty() {
            self.slots = vec![0; self.capacity];
        }
        if 2 * (self.coefs.len() + 1) > self.slots.len() {
            self.grow();
        }
        let r = self.r;
        let mask = self.slots.len() - 1;
        let tag = (h >> 26) & 0xFFFF_FFFF;
        let mut i = h as usize & mask;
        loop {
            let slot = self.slots[i];
            if slot == 0 {
                let row = self.coefs.len();
                self.slots[i] = tag << 32 | (row as u64 + 1);
                self.keys.extend_from_slice(key);
                self.coefs.push(c);
                self.abs.push(c.abs());
                return;
            }
            if slot >> 32 == tag {
                let row = (slot & 0xFFFF_FFFF) as usize - 1;
                if &self.keys[row * r..(row + 1) * r] == key {
                    self.coefs[row] += c;
                    self.abs[row] += c.abs();
                    return;
                }
            }
            i = (i + 1) & mask;
        }
    }

    /// Drop the cancelled rows (into the bound) and free the index.
    fn close(&mut self) {
        let r = self.r;
        let mut kept = 0;
        for row in 0..self.coefs.len() {
            let c = self.coefs[row];
            // A coefficient that is not finite is never a cancellation
            // (it is reported at the end of the run).
            if c == 0.0 || (c.is_finite() && c.abs() <= CANCELLATION * self.abs[row]) {
                self.bound += c.abs();
                self.dropped += 1;
                continue;
            }
            if kept != row {
                self.keys.copy_within(row * r..(row + 1) * r, kept * r);
                self.coefs[kept] = c;
            }
            kept += 1;
        }
        self.keys.truncate(kept * r);
        self.coefs.truncate(kept);
        self.abs = Vec::new();
        self.slots = Vec::new();
    }
}

/// The table after a splitting pass.
struct SplitOut {
    keys: Vec<u64>,
    coefs: Vec<f64>,
    bound: f64,
    dropped: usize,
    /// Strings held before the cancellations were removed.
    held: usize,
}

/// Apply a splitting pass (pass number `pass`, for messages).
fn split(
    keys: &[u64],
    coefs: &[f64],
    level: &Level,
    cut: Truncation,
    pass: usize,
) -> Result<SplitOut> {
    let r = 2 * cut.words;
    let rows = coefs.len();
    let chunks = rows.div_ceil(CHUNK);
    // Indices sized for an output as large as the input (they grow by
    // doubling beyond it).
    let mut shards: Vec<Shard> = (0..SHARDS)
        .map(|_| Shard::new(r, rows.div_ceil(SHARDS)))
        .collect();
    let (mut bound, mut dropped) = (0.0, 0usize);
    let mut held = 0;
    for first in (0..chunks).step_by(BATCH_CHUNKS) {
        let count = BATCH_CHUNKS.min(chunks - first);
        let outs = map_tasks(count > 1, count, |i| {
            let c = first + i;
            let end = ((c + 1) * CHUNK).min(rows);
            expand_chunk(
                &keys[c * CHUNK * r..end * r],
                &coefs[c * CHUNK..end],
                level,
                cut,
                pass,
            )
        });
        // The first error in chunk order.
        let outs = outs.into_iter().collect::<Result<Vec<ChunkOut>>>()?;
        // The rows held and the images waiting to be merged.
        let waiting = held + outs.iter().map(|out| out.coefs.len()).sum::<usize>();
        if waiting > cut.max_terms {
            return Err(too_many(pass, waiting));
        }
        for out in &outs {
            bound += out.bound;
            dropped += out.dropped;
        }
        let images = waiting - held;
        for_each_task_mut(images >= PARALLEL_ROWS, &mut shards, |s, shard| {
            for out in &outs {
                for &i in &out.order[out.start[s]..out.start[s + 1]] {
                    let i = i as usize;
                    shard.insert(&out.keys[i * r..(i + 1) * r], out.hashes[i], out.coefs[i]);
                }
            }
        });
        held = shards.iter().map(Shard::len).sum();
        if held > cut.max_terms {
            return Err(too_many(pass, held));
        }
    }
    for_each_task_mut(held >= PARALLEL_ROWS, &mut shards, |_, shard| shard.close());
    let total: usize = shards.iter().map(Shard::len).sum();
    let mut out_keys = Vec::with_capacity(total * r);
    let mut out_coefs = Vec::with_capacity(total);
    for shard in shards {
        bound += shard.bound;
        dropped += shard.dropped;
        out_keys.extend_from_slice(&shard.keys);
        out_coefs.extend_from_slice(&shard.coefs);
    }
    Ok(SplitOut {
        keys: out_keys,
        coefs: out_coefs,
        bound,
        dropped,
        held,
    })
}

/// `Tr(ρ O)` of the table in the product state with Bloch vectors `bloch`
/// (`None`: `|0…0⟩`), summed in chunk order.
fn readout(keys: &[u64], coefs: &[f64], w: usize, bloch: Option<&[[f64; 3]]>) -> f64 {
    let r = 2 * w;
    let rows = coefs.len();
    parallel::sum_chunks_f64(rows.div_ceil(CHUNK), rows, |ch| {
        let end = ((ch + 1) * CHUNK).min(rows);
        let mut sum = 0.0;
        for row in ch * CHUNK..end {
            let key = &keys[row * r..(row + 1) * r];
            let c = coefs[row];
            match bloch {
                None => {
                    if key[..w].iter().all(|&x| x == 0) {
                        sum += c;
                    }
                }
                Some(bloch) => {
                    let mut f = c;
                    for word in 0..w {
                        let (x, z) = (key[word], key[w + word]);
                        let mut bits = x | z;
                        while bits != 0 {
                            let b = bits.trailing_zeros();
                            bits &= bits - 1;
                            let q = 64 * word + b as usize;
                            // X, Y, Z = (x, z) bits (1, 0), (1, 1), (0, 1).
                            let letter = match ((x >> b) & 1, (z >> b) & 1) {
                                (1, 0) => 0,
                                (1, 1) => 1,
                                _ => 2,
                            };
                            f *= bloch[q][letter];
                        }
                    }
                    sum += f;
                }
            }
        }
        sum
    })
}
