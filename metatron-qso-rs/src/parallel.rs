//! Internal data-parallel helpers.
//!
//! With the `parallel` feature (default) work is distributed with rayon once
//! a problem is large enough to amortise the scheduling overhead; otherwise
//! everything runs serially. Reductions use a *fixed* chunking so that
//! floating-point results are bit-for-bit identical regardless of the number
//! of threads — reproducibility is a design goal of this crate.

use num_complex::Complex64;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Problem size from which kernels switch to multi-threaded execution.
/// Below ~2¹⁶ amplitudes a gate takes only tens of microseconds and the
/// scheduling overhead outweighs the gain (measured with the benchmark
/// report example).
#[cfg(not(miri))]
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) const PAR_THRESHOLD: usize = 1 << 16;

/// Granularity of parallel work items and of deterministic reductions.
#[cfg(not(miri))]
pub(crate) const CHUNK: usize = 1 << 12;

// Under Miri the same code paths run on registers small enough to
// interpret: 7 qubits are parallel and split into several chunks.
#[cfg(miri)]
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) const PAR_THRESHOLD: usize = 1 << 6;
#[cfg(miri)]
pub(crate) const CHUNK: usize = 1 << 3;

/// Raw pointer that may be shared between threads.
///
/// Safety contract: every user must guarantee that concurrent accesses
/// through the pointer touch pairwise disjoint elements.
#[derive(Clone, Copy)]
pub(crate) struct SyncPtr<T>(*mut T);

unsafe impl<T: Send> Send for SyncPtr<T> {}
unsafe impl<T: Send> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    pub(crate) fn new(slice: &mut [T]) -> Self {
        SyncPtr(slice.as_mut_ptr())
    }

    /// Accessor (a method, so closures capture the `Send` wrapper and not
    /// the raw field).
    #[inline(always)]
    pub(crate) fn get(self) -> *mut T {
        self.0
    }
}

/// Call `f(i)` for every `i in 0..n`, in parallel when worthwhile.
#[inline]
pub(crate) fn for_each_index<F>(n: usize, f: F)
where
    F: Fn(usize) + Sync + Send,
{
    #[cfg(feature = "parallel")]
    if n >= PAR_THRESHOLD {
        (0..n).into_par_iter().with_min_len(CHUNK).for_each(f);
        return;
    }
    for i in 0..n {
        f(i);
    }
}

/// Call `f(i)` for every `i in 0..n_items`, where the items together touch
/// `work` elements: in parallel once `work` reaches the threshold, with at
/// least `min_len` items per task.
#[inline]
pub(crate) fn for_each_item<F>(n_items: usize, work: usize, min_len: usize, f: F)
where
    F: Fn(usize) + Sync + Send,
{
    #[cfg(feature = "parallel")]
    if work >= PAR_THRESHOLD {
        (0..n_items)
            .into_par_iter()
            .with_min_len(min_len.max(1))
            .for_each(f);
        return;
    }
    let _ = (work, min_len);
    for i in 0..n_items {
        f(i);
    }
}

/// Call `f(a, b)` for every pair `(data[i], data[i + stride])` with bit
/// `stride` of `i` clear (`stride` a power of two dividing `data.len()/2`).
///
/// This is the memory layout of an uncontrolled single-qubit gate; iterating
/// over contiguous halves lets the compiler vectorise the inner loop.
#[inline]
pub(crate) fn for_each_pair<T, F>(data: &mut [T], stride: usize, f: F)
where
    T: Send,
    F: Fn(&mut T, &mut T) + Sync + Send,
{
    let block = |chunk: &mut [T]| {
        let (lo, hi) = chunk.split_at_mut(stride);
        lo.iter_mut().zip(hi).for_each(|(a, b)| f(a, b));
    };
    #[cfg(feature = "parallel")]
    if data.len() >= PAR_THRESHOLD {
        if stride >= CHUNK {
            // Few large blocks: parallelise inside each block.
            data.par_chunks_mut(2 * stride).for_each(|chunk| {
                let (lo, hi) = chunk.split_at_mut(stride);
                lo.par_chunks_mut(CHUNK)
                    .zip(hi.par_chunks_mut(CHUNK))
                    .for_each(|(l, h)| l.iter_mut().zip(h).for_each(|(a, b)| f(a, b)));
            });
        } else {
            data.par_chunks_mut(CHUNK.max(2 * stride))
                .for_each(|group| group.chunks_exact_mut(2 * stride).for_each(block));
        }
        return;
    }
    data.chunks_exact_mut(2 * stride).for_each(block);
}

/// Mutate every element of `data` with knowledge of its index.
#[inline]
pub(crate) fn for_each_mut<T, F>(data: &mut [T], f: F)
where
    T: Send,
    F: Fn(usize, &mut T) + Sync + Send,
{
    #[cfg(feature = "parallel")]
    if data.len() >= PAR_THRESHOLD {
        data.par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(c, chunk)| {
                let offset = c * CHUNK;
                for (j, x) in chunk.iter_mut().enumerate() {
                    f(offset + j, x);
                }
            });
        return;
    }
    for (i, x) in data.iter_mut().enumerate() {
        f(i, x);
    }
}

/// Deterministic sum of `f(i)` over `0..n`.
pub(crate) fn sum_f64<F>(n: usize, f: F) -> f64
where
    F: Fn(usize) -> f64 + Sync + Send,
{
    let partial = |c: usize| {
        let start = c * CHUNK;
        let end = (start + CHUNK).min(n);
        let mut acc = 0.0;
        for i in start..end {
            acc += f(i);
        }
        acc
    };
    let n_chunks = n.div_ceil(CHUNK);
    #[cfg(feature = "parallel")]
    if n >= PAR_THRESHOLD {
        let parts: Vec<f64> = (0..n_chunks).into_par_iter().map(partial).collect();
        return parts.iter().sum();
    }
    (0..n_chunks).map(partial).sum()
}

/// Deterministic complex sum of `f(i)` over `0..n`.
pub(crate) fn sum_c64<F>(n: usize, f: F) -> Complex64
where
    F: Fn(usize) -> Complex64 + Sync + Send,
{
    let partial = |c: usize| {
        let start = c * CHUNK;
        let end = (start + CHUNK).min(n);
        let mut acc = Complex64::new(0.0, 0.0);
        for i in start..end {
            acc += f(i);
        }
        acc
    };
    let n_chunks = n.div_ceil(CHUNK);
    #[cfg(feature = "parallel")]
    if n >= PAR_THRESHOLD {
        let parts: Vec<Complex64> = (0..n_chunks).into_par_iter().map(partial).collect();
        return parts.iter().sum();
    }
    (0..n_chunks).map(partial).sum()
}

/// Map `0..n` to a vector, in parallel when worthwhile. Order is preserved.
pub(crate) fn map_collect<T, F>(n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    #[cfg(feature = "parallel")]
    if n >= PAR_THRESHOLD {
        return (0..n).into_par_iter().with_min_len(CHUNK).map(f).collect();
    }
    (0..n).map(f).collect()
}

/// Map independent coarse-grained tasks (e.g. random restarts) to results,
/// in parallel regardless of their count. Order is preserved.
pub(crate) fn map_tasks<T, F>(n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        (0..n).into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..n).map(f).collect()
    }
}

/// Most partial sums [`sum_ranges`] keeps in memory at once.
const MAX_PARTS: u64 = 1 << 16;

/// Deterministic sum over `0..n` where `f(start, end)` evaluates one fixed
/// chunk `start..end` (for loops that carry state from one index to the
/// next, such as Gray-code enumerations). Chunks are combined in order.
pub(crate) fn sum_ranges<T, F>(n: u64, f: F) -> T
where
    T: Send + std::iter::Sum<T>,
    F: Fn(u64, u64) -> T + Sync + Send,
{
    sum_ranges_with(n, CHUNK as u64, MAX_PARTS, f)
}

/// `(chunks per part, number of parts)` for `n_chunks` chunks split into at
/// most `max_parts` parts.
fn partition(n_chunks: u64, max_parts: u64) -> (u64, u64) {
    let per_part = n_chunks.div_ceil(max_parts).max(1);
    (per_part, n_chunks.div_ceil(per_part))
}

/// [`sum_ranges`] with explicit sizes. At most `max_parts` partial sums are
/// formed, each over consecutive chunks (one chunk per part up to
/// `chunk · max_parts` indices); parts are evaluated in parallel and added
/// in order. The grouping depends only on `n`, never on the thread count.
fn sum_ranges_with<T, F>(n: u64, chunk: u64, max_parts: u64, f: F) -> T
where
    T: Send + std::iter::Sum<T>,
    F: Fn(u64, u64) -> T + Sync + Send,
{
    let n_chunks = n.div_ceil(chunk);
    let (per_part, n_parts) = partition(n_chunks, max_parts);
    let part = |p: u64| -> T {
        let first = p * per_part;
        (first..(first + per_part).min(n_chunks))
            .map(|c| {
                let start = c * chunk;
                f(start, (start + chunk).min(n))
            })
            .sum()
    };
    #[cfg(feature = "parallel")]
    if n >= PAR_THRESHOLD as u64 {
        let parts: Vec<T> = (0..n_parts).into_par_iter().map(part).collect();
        return parts.into_iter().sum();
    }
    (0..n_parts).map(part).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_match_serial_reference() {
        let n: usize = PAR_THRESHOLD + 123;
        // Rounding makes the sum depend on the order of additions; the
        // summand avoids transcendental functions, which Miri perturbs.
        let f = |i: usize| 1.0 / (i as f64 + 0.37);
        let serial: f64 = {
            // Same chunked order as the implementation.
            let mut total = 0.0;
            for c in 0..n.div_ceil(CHUNK) {
                let mut acc = 0.0;
                for i in c * CHUNK..((c + 1) * CHUNK).min(n) {
                    acc += f(i);
                }
                total += acc;
            }
            total
        };
        assert_eq!(sum_f64(n, f).to_bits(), serial.to_bits());
    }

    #[test]
    fn range_sums_cover_every_index_once_in_bounded_parts() {
        // Integer sums are exact, so any gap, overlap or reordering of the
        // ranges would show up as a wrong total or a wrong range list.
        for (n, chunk, max_parts) in [(0, 4, 3), (1, 4, 3), (12, 4, 3), (13, 4, 3), (1000, 7, 5)] {
            let seen = std::sync::Mutex::new(Vec::new());
            let total: u64 = sum_ranges_with(n, chunk, max_parts, |s, e| {
                assert!(s < e && e - s <= chunk && e <= n);
                seen.lock().unwrap().push((s, e));
                (s..e).sum::<u64>()
            });
            assert_eq!(total, n * n.saturating_sub(1) / 2, "n = {n}");
            let mut seen = seen.into_inner().unwrap();
            seen.sort_unstable();
            assert_eq!(seen.len() as u64, n.div_ceil(chunk));
            assert!(seen.windows(2).all(|w| w[0].1 == w[1].0));
        }
        // Beyond chunk·max_parts indices the parts grow; their number does
        // not: a 40 × 40 permanent (2³⁹ terms) keeps 2¹⁶ partial sums, not 2²⁷.
        assert_eq!(partition(1 << 27, MAX_PARTS), (1 << 11, MAX_PARTS));
        assert_eq!(partition(MAX_PARTS, MAX_PARTS), (1, MAX_PARTS));
        assert_eq!(partition(MAX_PARTS + 1, MAX_PARTS), (2, MAX_PARTS / 2 + 1));
        assert_eq!(partition(0, MAX_PARTS), (1, 0));
        for n_chunks in [1, 2, 99, 1 << 20, (1 << 40) + 3, u64::MAX / 4] {
            let (per, parts) = partition(n_chunks, MAX_PARTS);
            assert!(parts <= MAX_PARTS && per * parts >= n_chunks && per * (parts - 1) < n_chunks);
        }
    }

    #[test]
    fn for_each_mut_sees_correct_indices() {
        let mut v = vec![0usize; 2 * PAR_THRESHOLD + 7];
        for_each_mut(&mut v, |i, x| *x = i * 2);
        assert!(v.iter().enumerate().all(|(i, &x)| x == 2 * i));
    }
}
