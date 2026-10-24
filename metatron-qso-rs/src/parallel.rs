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
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) const PAR_THRESHOLD: usize = 1 << 16;

/// Granularity of parallel work items and of deterministic reductions.
const CHUNK: usize = 1 << 12;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_match_serial_reference() {
        let n: usize = (1 << 16) + 123;
        let f = |i: usize| ((i as f64) * 0.37).sin();
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
    fn for_each_mut_sees_correct_indices() {
        let mut v = vec![0usize; (1 << 15) + 7];
        for_each_mut(&mut v, |i, x| *x = i * 2);
        assert!(v.iter().enumerate().all(|(i, &x)| x == 2 * i));
    }
}
