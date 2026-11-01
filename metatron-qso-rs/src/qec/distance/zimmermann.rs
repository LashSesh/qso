//! The Brouwer–Zimmermann algorithm for the least weight of an
//! undetectable logical error (Grassl, "Searching for linear codes with
//! large minimum distance", in *Discovering Mathematics with Magma*,
//! Springer 2006).
//!
//! The undetectable errors are the code `C = ker H`, `H` the detectors
//! flipped by each error (one column per error); a logical error is a
//! word of `C` that also flips an observable. `C` has generator matrices
//! `G_j` that are systematic on pairwise disjoint sets of columns `I_j`
//! (information sets, of ranks `r_j ≤ k = dim C`). Every word `c` is the
//! sum of the rows of `G_j` picked by its ones in `I_j`, together with
//! some of the `k − r_j` rows that vanish there. Enumerating the sums of
//! at most `w` rows of every `G_j` therefore meets every word with at most
//! `w − (k − r_j)` ones in some `I_j`; a word not met has more in each,
//! and weighs at least `Σ_j max(0, w + 1 − (k − r_j))`. Enumerating level
//! by level, the lightest logical word met is the answer once that bound
//! reaches it.
//!
//! A matrix of rank `r_j < k` raises the bound only from level `k − r_j`
//! on, so it waits until then and enumerates every lower level at once:
//! the bound needs all sums of at most `w` rows of each matrix it counts
//! (a word of `c|I_j = 0` that is the sum of fewer than `k − r_j` rows
//! would otherwise be missed).

use std::sync::atomic::{AtomicUsize, Ordering};

use super::{Budget, Error};
use crate::parallel::map_tasks;

/// Most bits of all generator matrices together (256 MiB).
pub(super) const MAX_GENERATOR_BITS: usize = 1 << 31;

/// Information sets whose rank falls this far below the dimension are not
/// kept: they would raise the bound only at levels no enumeration reaches.
const MAX_DEFICIT: usize = 64;

/// A generator matrix systematic on some columns, as rows of bits, with
/// the observables each row flips.
struct Generator {
    rows: Vec<u64>,
    observables: Vec<u64>,
    /// Its information set: row `i` holds the only one of column
    /// `columns[i]`.
    #[cfg_attr(not(test), allow(dead_code))]
    columns: Vec<usize>,
    /// Size of the information set.
    rank: usize,
}

/// The generator matrices of the undetectable errors on disjoint
/// information sets.
pub(super) struct Zimmermann {
    /// Dimension of the code.
    k: usize,
    words: usize,
    matrices: Vec<Generator>,
}

/// A logical word: its weight and its columns.
type Word = (usize, Vec<usize>);

/// What one level of the enumeration met.
pub(super) struct Level {
    /// The lightest logical word below the bound given (its columns).
    pub found: Option<Vec<usize>>,
    /// Whether every sum was met (else the budget ran out).
    pub complete: bool,
}

/// `C(n, w)` as a float (for estimates).
fn binomial(n: usize, w: usize) -> f64 {
    if w > n {
        return 0.0;
    }
    (0..w).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

fn get(row: &[u64], c: usize) -> bool {
    row[c / 64] >> (c % 64) & 1 == 1
}

/// What [`Zimmermann::build`] made of the errors.
pub(super) enum Enumeration {
    /// The generator matrices, ready for the levels.
    Ready(Zimmermann),
    /// No set of errors flips an observable and no detector.
    NoLogicalError,
    /// The matrices would exceed [`MAX_GENERATOR_BITS`].
    TooLarge,
    /// The budget stopped the construction.
    Stopped,
}

impl Zimmermann {
    /// The code of the undetectable errors and its generator matrices,
    /// further ones while they fit in [`MAX_GENERATOR_BITS`] together. The
    /// budget is checked at every pivot. At the limits the detectors'
    /// matrix and the first generator matrix are held at once, each up to
    /// [`MAX_GENERATOR_BITS`]; the detectors' matrix is freed before the
    /// further ones are built.
    pub fn build(n_detectors: usize, errors: &[Error], budget: &Budget) -> Enumeration {
        let n = errors.len();
        let words = n.div_ceil(64);
        if n_detectors.saturating_mul(64 * words) > MAX_GENERATOR_BITS {
            return Enumeration::TooLarge;
        }
        // H in reduced row echelon form: one row per detector.
        let mut h = vec![0u64; n_detectors * words];
        for (c, e) in errors.iter().enumerate() {
            for &d in &e.detectors {
                h[d * words + c / 64] ^= 1 << (c % 64);
            }
        }
        let mut pivots: Vec<usize> = Vec::new();
        let mut pivot_of = vec![usize::MAX; n];
        for (c, slot) in pivot_of.iter_mut().enumerate() {
            let r = pivots.len();
            if r == n_detectors {
                break;
            }
            let Some(p) = (r..n_detectors).find(|&i| get(&h[i * words..(i + 1) * words], c)) else {
                continue;
            };
            if budget.stopped() {
                return Enumeration::Stopped;
            }
            if p != r {
                for w in 0..words {
                    h.swap(p * words + w, r * words + w);
                }
            }
            let pivot_row: Vec<u64> = h[r * words..(r + 1) * words].to_vec();
            for i in 0..n_detectors {
                if i != r && get(&h[i * words..(i + 1) * words], c) {
                    for (x, y) in h[i * words..(i + 1) * words].iter_mut().zip(&pivot_row) {
                        *x ^= y;
                    }
                }
            }
            *slot = r;
            pivots.push(c);
        }
        // The kernel: for every free column f, e_f plus the pivot columns
        // of the rows that hold f.
        let free: Vec<usize> = (0..n).filter(|&c| pivot_of[c] == usize::MAX).collect();
        let k = free.len();
        if k == 0 {
            return Enumeration::NoLogicalError;
        }
        if k.saturating_mul(64 * words) > MAX_GENERATOR_BITS {
            return Enumeration::TooLarge;
        }
        let mut row_of = vec![usize::MAX; n];
        for (t, &f) in free.iter().enumerate() {
            row_of[f] = t;
        }
        let mut rows = vec![0u64; k * words];
        let mut observables = vec![0u64; k];
        for (t, &f) in free.iter().enumerate() {
            rows[t * words + f / 64] ^= 1 << (f % 64);
            observables[t] = errors[f].observables;
        }
        for (r, &p) in pivots.iter().enumerate() {
            let row = &h[r * words..(r + 1) * words];
            for &f in &free {
                if get(row, f) {
                    let t = row_of[f];
                    rows[t * words + p / 64] ^= 1 << (p % 64);
                    observables[t] ^= errors[p].observables;
                }
            }
        }
        drop(h);
        // Every word flips the observables of the rows it sums.
        if observables.iter().all(|&o| o == 0) {
            return Enumeration::NoLogicalError;
        }
        let mut matrices = vec![Generator {
            rows,
            observables,
            columns: free,
            rank: k,
        }];
        // Further information sets among the columns not used yet, each
        // systematic on the columns it can pivot on; fewer than
        // `k − MAX_DEFICIT` columns cannot make one.
        let bits = |m: usize| (m as u64) * (k as u64) * 64 * (words as u64);
        let mut candidates = pivots;
        while !candidates.is_empty()
            && candidates.len() + MAX_DEFICIT >= k
            && bits(matrices.len() + 1) <= MAX_GENERATOR_BITS as u64
        {
            let (mut g, mut obs) = (matrices[0].rows.clone(), matrices[0].observables.clone());
            let mut used = 0;
            let mut chosen = Vec::new();
            let mut rest = Vec::new();
            for &c in &candidates {
                let Some(p) = (used..k).find(|&i| get(&g[i * words..(i + 1) * words], c)) else {
                    rest.push(c);
                    continue;
                };
                if budget.stopped() {
                    return Enumeration::Stopped;
                }
                if p != used {
                    for w in 0..words {
                        g.swap(p * words + w, used * words + w);
                    }
                    obs.swap(p, used);
                }
                let pivot_row: Vec<u64> = g[used * words..(used + 1) * words].to_vec();
                let pivot_obs = obs[used];
                for i in 0..k {
                    if i != used && get(&g[i * words..(i + 1) * words], c) {
                        for (x, y) in g[i * words..(i + 1) * words].iter_mut().zip(&pivot_row) {
                            *x ^= y;
                        }
                        obs[i] ^= pivot_obs;
                    }
                }
                chosen.push(c);
                used += 1;
            }
            if used == 0 || k - used > MAX_DEFICIT {
                break;
            }
            matrices.push(Generator {
                rows: g,
                observables: obs,
                columns: chosen,
                rank: used,
            });
            candidates = rest;
        }
        Enumeration::Ready(Zimmermann { k, words, matrices })
    }

    /// Dimension of the code of undetectable errors.
    pub fn dimension(&self) -> usize {
        self.k
    }

    /// No undetectable error lighter than this is left unmet after level
    /// `w` (`usize::MAX` once every word was met).
    pub fn bound(&self, w: usize) -> usize {
        if w >= self.k {
            return usize::MAX;
        }
        self.matrices
            .iter()
            .map(|g| (w + 1).saturating_sub(self.k - g.rank))
            .sum()
    }

    /// The numbers of rows of `g` whose sums level `w` enumerates: none
    /// before `g` raises the bound, every number up to `w` at the level
    /// where it starts to, and then `w` alone.
    fn rows_summed(&self, g: &Generator, w: usize) -> std::ops::Range<usize> {
        let first = (self.k - g.rank).max(1);
        match w.cmp(&first) {
            std::cmp::Ordering::Less => 1..1,
            std::cmp::Ordering::Equal => 1..w + 1,
            std::cmp::Ordering::Greater => w..w + 1,
        }
    }

    /// Word operations of level `w`.
    pub fn cost(&self, w: usize) -> f64 {
        let per_sum = self.words as f64 + 4.0;
        self.matrices
            .iter()
            .flat_map(|g| self.rows_summed(g, w))
            .map(|i| binomial(self.k, i) * per_sum)
            .sum()
    }

    /// Level `w` (after levels `1..w`, in order): the lightest logical
    /// word lighter than `best` errors among the sums it enumerates.
    pub fn level(&self, w: usize, best: usize, budget: &Budget) -> Level {
        let mut found: Option<Word> = None;
        for g in &self.matrices {
            for i in self.rows_summed(g, w) {
                let limit = found.as_ref().map_or(best, |f| f.0);
                let (f, done) = self.enumerate(g, i, limit, budget);
                if let Some(f) = f.filter(|f| f.0 < limit) {
                    found = Some(f);
                }
                if !done {
                    return Level {
                        found: found.map(|(_, set)| set),
                        complete: false,
                    };
                }
            }
        }
        Level {
            found: found.map(|(_, set)| set),
            complete: true,
        }
    }

    /// The lightest logical sum of `w` rows of `g` lighter than `limit`
    /// (weight, columns), the first rows in parallel, and whether all sums
    /// were met. Among equally light ones the first in enumeration order,
    /// so that the result does not depend on the threads.
    fn enumerate(
        &self,
        g: &Generator,
        w: usize,
        limit: usize,
        budget: &Budget,
    ) -> (Option<Word>, bool) {
        let (k, words) = (self.k, self.words);
        if w == 0 || w > k {
            return (None, true);
        }
        if budget.stopped() {
            return (None, false);
        }
        // The lightest weight found by any task (or the limit, exclusive):
        // others skip heavier words but keep equally light ones.
        let shared = AtomicUsize::new(limit.saturating_sub(1));
        let results: Vec<Result<Option<Word>, ()>> = map_tasks(k - w + 1, |first| {
            // A task that starts after the deadline does nothing.
            if budget.stopped() {
                return Err(());
            }
            let mut walk = Walk {
                g,
                k,
                words,
                w,
                sums: vec![0u64; (w + 1) * words],
                observables: vec![0u64; w + 1],
                picked: vec![0usize; w],
                best: None,
                pending: 0,
                shared: &shared,
            };
            walk.sums[words..2 * words]
                .copy_from_slice(&g.rows[first * words..(first + 1) * words]);
            walk.observables[1] = g.observables[first];
            walk.picked[0] = first;
            let done = walk.descend(1, first + 1, budget);
            budget.spend(walk.pending);
            if done { Ok(walk.best) } else { Err(()) }
        });
        // The lightest; among equals the first in enumeration order.
        let mut out: Option<Word> = None;
        let mut complete = true;
        for r in results {
            match r {
                Ok(Some(f)) if out.as_ref().is_none_or(|o| f.0 < o.0) => out = Some(f),
                Ok(_) => {}
                Err(()) => complete = false,
            }
        }
        (out, complete)
    }
}

/// One task's depth-first enumeration of the sums of `w` rows.
struct Walk<'a> {
    g: &'a Generator,
    k: usize,
    words: usize,
    w: usize,
    /// Partial sums by depth (`sums[d]`: the first `d` rows picked).
    sums: Vec<u64>,
    observables: Vec<u64>,
    picked: Vec<usize>,
    best: Option<Word>,
    pending: u64,
    shared: &'a AtomicUsize,
}

impl Walk<'_> {
    /// Pick rows `from..` at depth `depth`; false when the budget ran out.
    fn descend(&mut self, depth: usize, from: usize, budget: &Budget) -> bool {
        let words = self.words;
        if depth == self.w {
            return self.leaf(budget);
        }
        let last = self.k - (self.w - depth);
        for i in from..=last {
            let (head, tail) = self.sums.split_at_mut((depth + 1) * words);
            let prev = &head[depth * words..(depth + 1) * words];
            let row = &self.g.rows[i * words..(i + 1) * words];
            for ((t, p), r) in tail[..words].iter_mut().zip(prev).zip(row) {
                *t = p ^ r;
            }
            self.observables[depth + 1] = self.observables[depth] ^ self.g.observables[i];
            self.picked[depth] = i;
            if !self.descend(depth + 1, i + 1, budget) {
                return false;
            }
        }
        true
    }

    /// A full sum: kept when logical and lighter than the best so far.
    fn leaf(&mut self, budget: &Budget) -> bool {
        // Counted in word operations, so that the budget is checked about
        // every millisecond whatever the length of the words.
        self.pending += self.words as u64 + 4;
        if self.pending >= 1 << 20 {
            if !budget.spend(self.pending) {
                return false;
            }
            self.pending = 0;
        }
        let w = self.w;
        if self.observables[w] == 0 {
            return true;
        }
        let words = self.words;
        let sum = &self.sums[w * words..(w + 1) * words];
        let weight: usize = sum.iter().map(|x| x.count_ones() as usize).sum();
        let limit = self.shared.load(Ordering::Relaxed);
        if weight <= limit && self.best.as_ref().is_none_or(|b| weight < b.0) {
            self.shared.fetch_min(weight, Ordering::Relaxed);
            let mut columns = Vec::with_capacity(weight);
            for (i, &x) in sum.iter().enumerate() {
                let mut bits = x;
                while bits != 0 {
                    columns.push(64 * i + bits.trailing_zeros() as usize);
                    bits &= bits - 1;
                }
            }
            self.best = Some((weight, columns));
        }
        true
    }
}

#[cfg(test)]
pub(super) mod tests;
