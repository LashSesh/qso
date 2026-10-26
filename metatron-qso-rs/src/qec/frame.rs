//! Pauli-frame sampling: 64 shots per machine word.
//!
//! Each shot carries a Pauli frame, the Pauli error accumulated so far
//! relative to a noiseless reference run. Clifford gates conjugate the
//! frame, noise multiplies it, and a measurement is flipped relative to
//! the reference when the frame has an X (or Y) on the measured qubit.
//! Bit `s` of a word belongs to shot `s`, so a gate updates 64 shots with
//! one word operation, as in Stim (Gidney, Quantum 5, 497 (2021)).
//!
//! Detectors are deterministic without noise, so a detection event is the
//! parity of the flips of its records, and no reference run is needed.
//! The Z part of the frame of a freshly reset or measured qubit is
//! randomized, which turns random measurement outcomes of the reference
//! into random flips.

use super::circuit::{DetectorCircuit, Noise, QecOp};
use crate::error::{QsoError, Result};
use crate::rng::Rng;
use crate::stabilizer::{CliffordTarget, apply_clifford};

/// Most bits of results one call returns, shots × (detectors + 64) with
/// one 64-bit observable mask per shot (1 GiB).
pub const MAX_SAMPLED_EVENTS: u64 = 1 << 33;

/// Most shots per batch, as 64-bit words.
const BATCH_WORDS: usize = 16;
/// Words of frames and records one batch may use (128 MiB).
const BATCH_MEMORY_WORDS: usize = 1 << 24;

/// Shots per batch of `circuit`: up to 1024, fewer when frames and
/// measurement records would exceed [`BATCH_MEMORY_WORDS`].
pub(crate) fn batch_shots(circuit: &DetectorCircuit) -> usize {
    let per_word = circuit.n_measurements() + 2 * circuit.n_qubits() + circuit.n_detectors();
    64 * (BATCH_MEMORY_WORDS / per_word.max(1)).clamp(1, BATCH_WORDS)
}

/// Pauli frames of 64·w shots: `x[q * w + k]` holds the X bits of qubit `q`
/// for shots `64k .. 64k + 63`.
pub(crate) struct Frames {
    pub(crate) w: usize,
    pub(crate) x: Vec<u64>,
    pub(crate) z: Vec<u64>,
}

impl Frames {
    pub(crate) fn new(n: usize, w: usize) -> Self {
        Frames {
            w,
            x: vec![0; n * w],
            z: vec![0; n * w],
        }
    }

    fn range(&self, q: usize) -> std::ops::Range<usize> {
        q * self.w..(q + 1) * self.w
    }
}

impl CliffordTarget for Frames {
    fn h(&mut self, q: usize) {
        for i in self.range(q) {
            std::mem::swap(&mut self.x[i], &mut self.z[i]);
        }
    }

    fn s(&mut self, q: usize) {
        for i in self.range(q) {
            self.z[i] ^= self.x[i];
        }
    }

    fn sdg(&mut self, q: usize) {
        self.s(q);
    }

    // Pauli gates only change signs, which frames do not track.
    fn x(&mut self, _: usize) {}

    fn y(&mut self, _: usize) {}

    fn z(&mut self, _: usize) {}

    fn cx(&mut self, c: usize, t: usize) {
        let (oc, ot) = (c * self.w, t * self.w);
        for k in 0..self.w {
            self.x[ot + k] ^= self.x[oc + k];
            self.z[oc + k] ^= self.z[ot + k];
        }
    }

    fn swap(&mut self, a: usize, b: usize) {
        let (oa, ob) = (a * self.w, b * self.w);
        for k in 0..self.w {
            self.x.swap(oa + k, ob + k);
            self.z.swap(oa + k, ob + k);
        }
    }
}

/// Call `hit(s)` for each shot `s < shots` that an event of probability
/// `p` hits, skipping geometrically between hits.
pub(crate) fn bernoulli_hits(shots: usize, p: f64, rng: &mut Rng, mut hit: impl FnMut(usize)) {
    if p <= 0.0 {
        return;
    }
    if p >= 1.0 {
        (0..shots).for_each(hit);
        return;
    }
    let log_q = (1.0 - p).ln();
    let mut s = 0usize;
    loop {
        let u = 1.0 - rng.uniform(); // in (0, 1]
        let skip = (u.ln() / log_q).floor();
        if !skip.is_finite() || skip >= (shots - s) as f64 {
            return;
        }
        s += skip as usize;
        hit(s);
        s += 1;
        if s >= shots {
            return;
        }
    }
}

/// Detection events and observable flips of many shots, bit-packed.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectorSamples {
    shots: usize,
    words: usize,
    /// `detectors[d * words + k]`: bits of shots `64k ..`.
    detectors: Vec<u64>,
    n_detectors: usize,
    /// Observable flips of each shot, bit `i` for observable `i`.
    observables: Vec<u64>,
}

impl DetectorSamples {
    pub fn shots(&self) -> usize {
        self.shots
    }

    pub fn n_detectors(&self) -> usize {
        self.n_detectors
    }

    /// Whether detector `d` fired in `shot`.
    pub fn fired(&self, d: usize, shot: usize) -> bool {
        (self.detectors[d * self.words + shot / 64] >> (shot % 64)) & 1 == 1
    }

    /// The detectors that fired in `shot`.
    pub fn fired_detectors(&self, shot: usize) -> Vec<usize> {
        (0..self.n_detectors)
            .filter(|&d| self.fired(d, shot))
            .collect()
    }

    /// The observable flips of `shot` (bit `i`: observable `i`).
    pub fn observable_flips(&self, shot: usize) -> u64 {
        self.observables[shot]
    }

    /// The fired detectors of every shot, ascending; faster than
    /// [`fired_detectors`](Self::fired_detectors) shot by shot.
    pub fn fired_by_shot(&self) -> Vec<Vec<usize>> {
        let mut out = vec![Vec::new(); self.shots];
        for d in 0..self.n_detectors {
            for (k, &word) in self.detectors[d * self.words..(d + 1) * self.words]
                .iter()
                .enumerate()
            {
                let mut bits = word;
                while bits != 0 {
                    out[64 * k + bits.trailing_zeros() as usize].push(d);
                    bits &= bits - 1;
                }
            }
        }
        out
    }

    /// How often detector `d` fired.
    pub fn count(&self, d: usize) -> usize {
        let row = &self.detectors[d * self.words..(d + 1) * self.words];
        row.iter().map(|w| w.count_ones() as usize).sum()
    }
}

/// Sample detection events and observable flips of `shots` shots.
///
/// Fails if a detector or observable is not deterministic without noise
/// (checked on 64 noiseless shots: a random parity escapes with
/// probability 2⁻⁶⁴).
pub fn sample_detectors(
    circuit: &DetectorCircuit,
    shots: usize,
    seed: u64,
) -> Result<DetectorSamples> {
    if (shots as u64).saturating_mul(circuit.n_detectors() as u64 + 64) > MAX_SAMPLED_EVENTS {
        return Err(QsoError::invalid(
            "shots",
            format!("shots × (detectors + 64) must be at most {MAX_SAMPLED_EVENTS}"),
        ));
    }
    check_deterministic(circuit)?;
    sample_with(circuit, shots, &mut Rng::seed_from_u64(seed))
}

/// [`sample_detectors`] without the checks, drawing from `rng`: samples
/// of consecutive calls whose shots are multiples of [`batch_shots`]
/// equal those of one call.
pub(crate) fn sample_with(
    circuit: &DetectorCircuit,
    shots: usize,
    rng: &mut Rng,
) -> Result<DetectorSamples> {
    let n_det = circuit.n_detectors();
    let words = shots.div_ceil(64);
    let mut out = DetectorSamples {
        shots,
        words,
        detectors: vec![0; n_det * words],
        n_detectors: n_det,
        observables: vec![0; shots],
    };
    let batch_size = batch_shots(circuit);
    let mut done = 0;
    while done < shots {
        let batch = (shots - done).min(batch_size);
        let w = batch.div_ceil(64);
        let (det, obs) = run_batch(circuit, batch, w, rng)?;
        let first_word = done / 64;
        for d in 0..n_det {
            out.detectors[d * words + first_word..d * words + first_word + w]
                .copy_from_slice(&det[d * w..(d + 1) * w]);
        }
        for (i, row) in obs.iter().enumerate() {
            for s in 0..batch {
                if (row[s / 64] >> (s % 64)) & 1 == 1 {
                    out.observables[done + s] |= 1 << i;
                }
            }
        }
        done += batch;
    }
    Ok(out)
}

pub(crate) fn check_deterministic(circuit: &DetectorCircuit) -> Result<()> {
    let quiet = circuit.without_noise();
    let mut rng = Rng::seed_from_u64(0x5eed);
    let (det, obs) = run_batch(&quiet, 64, 1, &mut rng)?;
    if let Some(d) = (0..circuit.n_detectors()).find(|&d| det[d] != 0) {
        return Err(QsoError::invalid(
            "detector",
            format!("detector {d} is not deterministic without noise"),
        ));
    }
    if let Some(i) = obs.iter().position(|row| row[0] != 0) {
        return Err(QsoError::invalid(
            "observable",
            format!("observable {i} is not deterministic without noise"),
        ));
    }
    Ok(())
}

/// Run one batch of `shots ≤ 64 w` shots; returns detection events
/// (`n_detectors × w` words) and observable flips (one row of `w` words per
/// observable). The Z frames of fresh qubits are drawn at random.
fn run_batch(
    circuit: &DetectorCircuit,
    shots: usize,
    w: usize,
    rng: &mut Rng,
) -> Result<(Vec<u64>, Vec<Vec<u64>>)> {
    let n = circuit.n_qubits();
    let mut f = Frames::new(n, w);
    let last_mask = if shots.is_multiple_of(64) {
        u64::MAX
    } else {
        (1u64 << (shots % 64)) - 1
    };
    let random_words = |rng: &mut Rng, out: &mut [u64]| {
        for (k, v) in out.iter_mut().enumerate() {
            *v = rng.next_u64();
            if k == w - 1 {
                *v &= last_mask;
            }
        }
    };
    for q in 0..n {
        random_words(rng, &mut f.z[q * w..(q + 1) * w]);
    }
    let mut records: Vec<u64> = Vec::with_capacity(circuit.n_measurements() * w);
    let mut det = Vec::with_capacity(circuit.n_detectors() * w);
    let mut obs = vec![vec![0u64; w]; circuit.n_observables()];
    let flip = |bits: &mut [u64], s: usize| bits[s / 64] ^= 1 << (s % 64);
    for op in circuit.operations() {
        match op {
            QecOp::Gate(inst) => apply_clifford(&mut f, inst, &[])?,
            QecOp::Measure { qubit, flip: p } => {
                let mut bits = f.x[qubit * w..(qubit + 1) * w].to_vec();
                bernoulli_hits(shots, *p, rng, |s| flip(&mut bits, s));
                records.extend_from_slice(&bits);
                random_words(rng, &mut f.z[qubit * w..(qubit + 1) * w]);
            }
            QecOp::Reset { qubit } => {
                f.x[qubit * w..(qubit + 1) * w].fill(0);
                random_words(rng, &mut f.z[qubit * w..(qubit + 1) * w]);
            }
            QecOp::Noise { noise, qubits } => apply_noise(&mut f, *noise, qubits, shots, rng),
            QecOp::Detector { records: recs } => {
                let mut bits = vec![0u64; w];
                for &r in recs {
                    for k in 0..w {
                        bits[k] ^= records[r * w + k];
                    }
                }
                det.extend_from_slice(&bits);
            }
            QecOp::Observable {
                index,
                records: recs,
            } => {
                for &r in recs {
                    for k in 0..w {
                        obs[*index][k] ^= records[r * w + k];
                    }
                }
            }
        }
    }
    Ok((det, obs))
}

fn apply_noise(f: &mut Frames, noise: Noise, qubits: &[usize], shots: usize, rng: &mut Rng) {
    let w = f.w;
    let toggle = |v: &mut Vec<u64>, q: usize, s: usize| v[q * w + s / 64] ^= 1 << (s % 64);
    match noise {
        Noise::XError(p) | Noise::YError(p) | Noise::ZError(p) => {
            let (dx, dz) = match noise {
                Noise::XError(_) => (true, false),
                Noise::YError(_) => (true, true),
                _ => (false, true),
            };
            for &q in qubits {
                let mut hits = Vec::new();
                bernoulli_hits(shots, p, rng, |s| hits.push(s));
                for s in hits {
                    if dx {
                        toggle(&mut f.x, q, s);
                    }
                    if dz {
                        toggle(&mut f.z, q, s);
                    }
                }
            }
        }
        Noise::Depolarize1(p) => {
            for &q in qubits {
                let mut hits = Vec::new();
                bernoulli_hits(shots, p, rng, |s| hits.push(s));
                for s in hits {
                    // 1: X, 2: Z, 3: Y.
                    let k = rng.below(3) + 1;
                    if k & 1 == 1 {
                        toggle(&mut f.x, q, s);
                    }
                    if k & 2 == 2 {
                        toggle(&mut f.z, q, s);
                    }
                }
            }
        }
        Noise::Depolarize2(p) => {
            for pair in qubits.chunks(2) {
                let mut hits = Vec::new();
                bernoulli_hits(shots, p, rng, |s| hits.push(s));
                for s in hits {
                    // Bits of k: x_a, z_a, x_b, z_b (k ≠ 0).
                    let k = rng.below(15) + 1;
                    for (bit, (q, x)) in [
                        (pair[0], true),
                        (pair[0], false),
                        (pair[1], true),
                        (pair[1], false),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        if (k >> bit) & 1 == 1 {
                            if x {
                                toggle(&mut f.x, q, s);
                            } else {
                                toggle(&mut f.z, q, s);
                            }
                        }
                    }
                }
            }
        }
    }
}
