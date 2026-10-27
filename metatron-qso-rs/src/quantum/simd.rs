//! Run primitives of the amplitude kernels, with AVX versions on x86-64
//! and NEON versions on aarch64.
//!
//! The kernels reduce every gate to a few operations on contiguous runs
//! of amplitudes: a 2×2 matrix on two runs (or on adjacent pairs, for
//! qubit 0), a phase on one run and a 4×4 matrix on four runs. Each has a
//! portable version and, on x86-64 processors with AVX (detected once at
//! run time), a version that handles two amplitudes per 256-bit register;
//! on aarch64, where NEON is part of the base instruction set, one with an
//! amplitude per 128-bit register for the complex arithmetic.
//!
//! Both versions round identically: a complex product `m·x` is formed as
//! `(m.re·x.re − m.im·x.im, m.re·x.im + m.im·x.re)` with separate
//! multiplications and additions (no fused multiply-add), and sums are
//! taken in the same order, so results agree bit for bit whichever version
//! runs. On other processors the portable versions run. Real matrices are
//! simple enough for the compiler to vectorise them on aarch64.

use std::sync::atomic::{AtomicU8, Ordering};

use num_complex::Complex64 as C64;

const UNKNOWN: u8 = 0;
const PORTABLE: u8 = 1;
const AVX: u8 = 2;
const NEON: u8 = 3;

static LEVEL: AtomicU8 = AtomicU8::new(UNKNOWN);

fn detect() -> u8 {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if std::arch::is_x86_feature_detected!("avx") {
        return AVX;
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return NEON;
    }
    PORTABLE
}

#[inline]
fn level() -> u8 {
    match LEVEL.load(Ordering::Relaxed) {
        UNKNOWN => {
            let l = detect();
            LEVEL.store(l, Ordering::Relaxed);
            l
        }
        l => l,
    }
}

/// Use the vector kernels where the processor has them (`true`, the
/// default) or the portable ones (`false`). Results are bit for bit the
/// same either way; this exists for benchmarks and tests. Returns whether
/// vector kernels are now in use.
pub fn set_simd(enabled: bool) -> bool {
    let l = if enabled { detect() } else { PORTABLE };
    LEVEL.store(l, Ordering::Relaxed);
    l != PORTABLE
}

/// The instruction set the kernels use: `"avx"`, `"neon"` or
/// `"portable"`.
pub fn simd_level() -> &'static str {
    match level() {
        AVX => "avx",
        NEON => "neon",
        _ => "portable",
    }
}

/// `a ← m₀₀·a + m₀₁·b`, `b ← m₁₀·a + m₁₁·b` elementwise over two runs of
/// equal length (`m` row-major).
#[inline]
pub(crate) fn mat2_runs(a: &mut [C64], b: &mut [C64], m: &[C64; 4]) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    if level() == NEON {
        // SAFETY: NEON is available (detected).
        unsafe { neon::mat2_runs(a, b, m) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if level() == AVX {
        // SAFETY: AVX is available (detected).
        unsafe { avx::mat2_runs(a, b, m) };
        return;
    }
    portable::mat2_runs(a, b, m);
}

/// [`mat2_runs`] for a real matrix.
#[inline]
pub(crate) fn mat2_runs_real(a: &mut [C64], b: &mut [C64], m: &[f64; 4]) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if level() == AVX {
        // SAFETY: AVX is available (detected).
        unsafe { avx::mat2_runs_real(a, b, m) };
        return;
    }
    portable::mat2_runs_real(a, b, m);
}

/// The 2×2 matrix on every pair `(x[i], x[i + stride])` with bit `stride`
/// of `i` clear (`x.len()` a multiple of `2·stride`): a gate on a low
/// qubit, applied to one chunk in one call.
#[inline]
pub(crate) fn mat2_strided(x: &mut [C64], stride: usize, m: &[C64; 4]) {
    debug_assert!(x.len().is_multiple_of(2 * stride));
    #[cfg(target_arch = "aarch64")]
    if level() == NEON {
        // SAFETY: NEON is available (detected).
        unsafe { neon::mat2_strided(x, stride, m) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if level() == AVX {
        // SAFETY: AVX is available (detected).
        unsafe { avx::mat2_strided(x, stride, m) };
        return;
    }
    portable::mat2_strided(x, stride, m);
}

/// [`mat2_strided`] for a real matrix.
#[inline]
pub(crate) fn mat2_strided_real(x: &mut [C64], stride: usize, m: &[f64; 4]) {
    debug_assert!(x.len().is_multiple_of(2 * stride));
    #[cfg(target_arch = "x86_64")]
    if level() == AVX {
        // SAFETY: AVX is available (detected).
        unsafe { avx::mat2_strided_real(x, stride, m) };
        return;
    }
    portable::mat2_strided_real(x, stride, m);
}

/// `x[i] ← x[i]·d₀` and `x[i + stride] ← x[i + stride]·d₁` for every pair
/// of [`mat2_strided`].
#[inline]
pub(crate) fn scale2_strided(x: &mut [C64], stride: usize, d0: C64, d1: C64) {
    debug_assert!(x.len().is_multiple_of(2 * stride));
    #[cfg(target_arch = "aarch64")]
    if level() == NEON {
        // SAFETY: NEON is available (detected).
        unsafe { neon::scale2_strided(x, stride, d0, d1) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if level() == AVX {
        // SAFETY: AVX is available (detected).
        unsafe { avx::scale2_strided(x, stride, d0, d1) };
        return;
    }
    portable::scale2_strided(x, stride, d0, d1);
}

/// `x[i + stride] ← x[i + stride]·z` for every pair of [`mat2_strided`],
/// leaving the lower amplitudes untouched (a phase gate on a low qubit).
#[inline]
pub(crate) fn scale_upper_strided(x: &mut [C64], stride: usize, z: C64) {
    debug_assert!(x.len().is_multiple_of(2 * stride));
    #[cfg(target_arch = "aarch64")]
    if level() == NEON {
        // SAFETY: NEON is available (detected).
        unsafe { neon::scale_upper_strided(x, stride, z) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if level() == AVX {
        // SAFETY: AVX is available (detected).
        unsafe { avx::scale_upper_strided(x, stride, z) };
        return;
    }
    portable::scale_upper_strided(x, stride, z);
}

/// `x ← x·z` over a run.
#[inline]
pub(crate) fn scale_run(x: &mut [C64], z: C64) {
    #[cfg(target_arch = "aarch64")]
    if level() == NEON {
        // SAFETY: NEON is available (detected).
        unsafe { neon::scale_run(x, z) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if level() == AVX {
        // SAFETY: AVX is available (detected).
        unsafe { avx::scale_run(x, z) };
        return;
    }
    portable::scale_run(x, z);
}

/// The 4×4 matrix `m` (row-major) on `(r[0][i], r[1][i], r[2][i],
/// r[3][i])` for every `i`; the four runs have equal length.
#[inline]
pub(crate) fn mat4_runs(r: [&mut [C64]; 4], m: &[C64; 16]) {
    debug_assert!(r.iter().all(|x| x.len() == r[0].len()));
    #[cfg(target_arch = "aarch64")]
    if level() == NEON {
        // SAFETY: NEON is available (detected).
        unsafe { neon::mat4_runs(r, m) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if level() == AVX {
        // SAFETY: AVX is available (detected).
        unsafe { avx::mat4_runs(r, m) };
        return;
    }
    portable::mat4_runs(r, m);
}

/// The reference versions: the arithmetic the kernels always used.
mod portable {
    use num_complex::Complex64 as C64;

    #[inline]
    pub(super) fn mat2_runs(a: &mut [C64], b: &mut [C64], m: &[C64; 4]) {
        for (x, y) in a.iter_mut().zip(b) {
            let (u, v) = (*x, *y);
            *x = m[0] * u + m[1] * v;
            *y = m[2] * u + m[3] * v;
        }
    }

    #[inline]
    pub(super) fn mat2_runs_real(a: &mut [C64], b: &mut [C64], m: &[f64; 4]) {
        for (x, y) in a.iter_mut().zip(b) {
            let (u, v) = (*x, *y);
            *x = u * m[0] + v * m[1];
            *y = u * m[2] + v * m[3];
        }
    }

    #[inline]
    pub(super) fn mat2_adjacent(x: &mut [C64], m: &[C64; 4]) {
        for p in x.as_chunks_mut::<2>().0 {
            let (u, v) = (p[0], p[1]);
            p[0] = m[0] * u + m[1] * v;
            p[1] = m[2] * u + m[3] * v;
        }
    }

    #[inline]
    pub(super) fn mat2_adjacent_real(x: &mut [C64], m: &[f64; 4]) {
        for p in x.as_chunks_mut::<2>().0 {
            let (u, v) = (p[0], p[1]);
            p[0] = u * m[0] + v * m[1];
            p[1] = u * m[2] + v * m[3];
        }
    }

    #[inline]
    pub(super) fn scale_run(x: &mut [C64], z: C64) {
        for a in x {
            *a *= z;
        }
    }

    #[inline]
    pub(super) fn mat2_strided(x: &mut [C64], stride: usize, m: &[C64; 4]) {
        if stride == 1 {
            return mat2_adjacent(x, m);
        }
        for block in x.chunks_exact_mut(2 * stride) {
            let (lo, hi) = block.split_at_mut(stride);
            mat2_runs(lo, hi, m);
        }
    }

    #[inline]
    pub(super) fn mat2_strided_real(x: &mut [C64], stride: usize, m: &[f64; 4]) {
        if stride == 1 {
            return mat2_adjacent_real(x, m);
        }
        for block in x.chunks_exact_mut(2 * stride) {
            let (lo, hi) = block.split_at_mut(stride);
            mat2_runs_real(lo, hi, m);
        }
    }

    #[inline]
    pub(super) fn scale_upper_strided(x: &mut [C64], stride: usize, z: C64) {
        for block in x.chunks_exact_mut(2 * stride) {
            scale_run(&mut block[stride..], z);
        }
    }

    #[inline]
    pub(super) fn scale2_strided(x: &mut [C64], stride: usize, d0: C64, d1: C64) {
        for block in x.chunks_exact_mut(2 * stride) {
            let (lo, hi) = block.split_at_mut(stride);
            scale_run(lo, d0);
            scale_run(hi, d1);
        }
    }

    #[inline]
    pub(super) fn mat4_runs(r: [&mut [C64]; 4], m: &[C64; 16]) {
        let [r0, r1, r2, r3] = r;
        for i in 0..r0.len() {
            let v = [r0[i], r1[i], r2[i], r3[i]];
            let out: [C64; 4] = std::array::from_fn(|k| {
                let row = &m[4 * k..4 * k + 4];
                row[0] * v[0] + row[1] * v[1] + row[2] * v[2] + row[3] * v[3]
            });
            r0[i] = out[0];
            r1[i] = out[1];
            r2[i] = out[2];
            r3[i] = out[3];
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx {
    //! Two complex amplitudes `[re₀, im₀, re₁, im₁]` per register.

    use std::arch::x86_64::*;

    use num_complex::Complex64 as C64;

    use super::portable;

    /// `m·x` for both lanes, with `m = (re, im)` broadcast:
    /// `(re·x.re − im·x.im, re·x.im + im·x.re)`, rounded as the scalar
    /// product.
    #[inline]
    #[target_feature(enable = "avx")]
    fn cmul(re: __m256d, im: __m256d, x: __m256d) -> __m256d {
        let p = _mm256_mul_pd(re, x);
        let q = _mm256_mul_pd(im, _mm256_permute_pd::<0b0101>(x));
        _mm256_addsub_pd(p, q)
    }

    #[inline]
    #[target_feature(enable = "avx")]
    fn split(z: C64) -> (__m256d, __m256d) {
        (_mm256_set1_pd(z.re), _mm256_set1_pd(z.im))
    }

    /// # Safety
    /// `p` is valid for reading two `C64` (four `f64`).
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn load(p: *const C64) -> __m256d {
        // SAFETY: by the caller; `C64` is `repr(C)` with two `f64`.
        unsafe { _mm256_loadu_pd(p.cast()) }
    }

    /// # Safety
    /// `p` is valid for writing two `C64`.
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn store(p: *mut C64, v: __m256d) {
        // SAFETY: by the caller.
        unsafe { _mm256_storeu_pd(p.cast(), v) }
    }

    /// # Safety
    /// AVX is available.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn mat2_runs(a: &mut [C64], b: &mut [C64], m: &[C64; 4]) {
        let n = a.len().min(b.len());
        let even = n & !1;
        let (m0r, m0i) = split(m[0]);
        let (m1r, m1i) = split(m[1]);
        let (m2r, m2i) = split(m[2]);
        let (m3r, m3i) = split(m[3]);
        let (pa, pb) = (a.as_mut_ptr(), b.as_mut_ptr());
        for i in (0..even).step_by(2) {
            // SAFETY: i + 1 < n ≤ both lengths.
            unsafe {
                let x = load(pa.add(i));
                let y = load(pb.add(i));
                let u = _mm256_add_pd(cmul(m0r, m0i, x), cmul(m1r, m1i, y));
                let v = _mm256_add_pd(cmul(m2r, m2i, x), cmul(m3r, m3i, y));
                store(pa.add(i), u);
                store(pb.add(i), v);
            }
        }
        portable::mat2_runs(&mut a[even..n], &mut b[even..n], m);
    }

    /// # Safety
    /// AVX is available.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn mat2_runs_real(a: &mut [C64], b: &mut [C64], m: &[f64; 4]) {
        let n = a.len().min(b.len());
        let even = n & !1;
        let [m0, m1, m2, m3] = m.map(|x| _mm256_set1_pd(x));
        let (pa, pb) = (a.as_mut_ptr(), b.as_mut_ptr());
        for i in (0..even).step_by(2) {
            // SAFETY: i + 1 < n ≤ both lengths.
            unsafe {
                let x = load(pa.add(i));
                let y = load(pb.add(i));
                let u = _mm256_add_pd(_mm256_mul_pd(x, m0), _mm256_mul_pd(y, m1));
                let v = _mm256_add_pd(_mm256_mul_pd(x, m2), _mm256_mul_pd(y, m3));
                store(pa.add(i), u);
                store(pb.add(i), v);
            }
        }
        portable::mat2_runs_real(&mut a[even..n], &mut b[even..n], m);
    }

    /// # Safety
    /// AVX is available.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn mat2_adjacent(x: &mut [C64], m: &[C64; 4]) {
        // Lanes hold (row 0, row 1): [m₀₀, m₁₀] times [u, u] plus
        // [m₀₁, m₁₁] times [v, v].
        let c0r = _mm256_setr_pd(m[0].re, m[0].re, m[2].re, m[2].re);
        let c0i = _mm256_setr_pd(m[0].im, m[0].im, m[2].im, m[2].im);
        let c1r = _mm256_setr_pd(m[1].re, m[1].re, m[3].re, m[3].re);
        let c1i = _mm256_setr_pd(m[1].im, m[1].im, m[3].im, m[3].im);
        let p = x.as_mut_ptr();
        for i in (0..x.len() & !1).step_by(2) {
            // SAFETY: i + 1 < x.len().
            unsafe {
                let w = load(p.add(i));
                let u = _mm256_permute2f128_pd::<0x00>(w, w);
                let v = _mm256_permute2f128_pd::<0x11>(w, w);
                store(
                    p.add(i),
                    _mm256_add_pd(cmul(c0r, c0i, u), cmul(c1r, c1i, v)),
                );
            }
        }
    }

    /// # Safety
    /// AVX is available.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn mat2_adjacent_real(x: &mut [C64], m: &[f64; 4]) {
        let c0 = _mm256_setr_pd(m[0], m[0], m[2], m[2]);
        let c1 = _mm256_setr_pd(m[1], m[1], m[3], m[3]);
        let p = x.as_mut_ptr();
        for i in (0..x.len() & !1).step_by(2) {
            // SAFETY: i + 1 < x.len().
            unsafe {
                let w = load(p.add(i));
                let u = _mm256_permute2f128_pd::<0x00>(w, w);
                let v = _mm256_permute2f128_pd::<0x11>(w, w);
                store(
                    p.add(i),
                    _mm256_add_pd(_mm256_mul_pd(u, c0), _mm256_mul_pd(v, c1)),
                );
            }
        }
    }

    /// # Safety
    /// AVX is available.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn mat2_strided(x: &mut [C64], stride: usize, m: &[C64; 4]) {
        if stride == 1 {
            // SAFETY: AVX is available (by the caller).
            return unsafe { mat2_adjacent(x, m) };
        }
        for block in x.chunks_exact_mut(2 * stride) {
            let (lo, hi) = block.split_at_mut(stride);
            // SAFETY: AVX is available (by the caller).
            unsafe { mat2_runs(lo, hi, m) };
        }
    }

    /// # Safety
    /// AVX is available.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn mat2_strided_real(x: &mut [C64], stride: usize, m: &[f64; 4]) {
        if stride == 1 {
            // SAFETY: AVX is available (by the caller).
            return unsafe { mat2_adjacent_real(x, m) };
        }
        for block in x.chunks_exact_mut(2 * stride) {
            let (lo, hi) = block.split_at_mut(stride);
            // SAFETY: AVX is available (by the caller).
            unsafe { mat2_runs_real(lo, hi, m) };
        }
    }

    /// # Safety
    /// AVX is available.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn scale_upper_strided(x: &mut [C64], stride: usize, z: C64) {
        if stride == 1 {
            // Scale both lanes, then keep the lower one as it was.
            let (zr, zi) = split(z);
            let p = x.as_mut_ptr();
            for i in (0..x.len() & !1).step_by(2) {
                // SAFETY: i + 1 < x.len().
                unsafe {
                    let w = load(p.add(i));
                    store(p.add(i), _mm256_blend_pd::<0b1100>(w, cmul(zr, zi, w)));
                }
            }
            return;
        }
        for block in x.chunks_exact_mut(2 * stride) {
            // SAFETY: AVX is available (by the caller).
            unsafe { scale_run(&mut block[stride..], z) };
        }
    }

    /// # Safety
    /// AVX is available.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn scale2_strided(x: &mut [C64], stride: usize, d0: C64, d1: C64) {
        if stride == 1 {
            // Lanes [d₀, d₁] times each adjacent pair.
            let re = _mm256_setr_pd(d0.re, d0.re, d1.re, d1.re);
            let im = _mm256_setr_pd(d0.im, d0.im, d1.im, d1.im);
            let p = x.as_mut_ptr();
            for i in (0..x.len() & !1).step_by(2) {
                // SAFETY: i + 1 < x.len().
                unsafe { store(p.add(i), cmul(re, im, load(p.add(i)))) };
            }
            return;
        }
        for block in x.chunks_exact_mut(2 * stride) {
            let (lo, hi) = block.split_at_mut(stride);
            // SAFETY: AVX is available (by the caller).
            unsafe {
                scale_run(lo, d0);
                scale_run(hi, d1);
            }
        }
    }

    /// # Safety
    /// AVX is available.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn scale_run(x: &mut [C64], z: C64) {
        let even = x.len() & !1;
        let (zr, zi) = split(z);
        let p = x.as_mut_ptr();
        for i in (0..even).step_by(2) {
            // SAFETY: i + 1 < x.len().
            unsafe { store(p.add(i), cmul(zr, zi, load(p.add(i)))) };
        }
        portable::scale_run(&mut x[even..], z);
    }

    /// # Safety
    /// AVX is available.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn mat4_runs(r: [&mut [C64]; 4], m: &[C64; 16]) {
        let n = r.iter().map(|x| x.len()).min().unwrap_or(0);
        let even = n & !1;
        let c: [(__m256d, __m256d); 16] = std::array::from_fn(|k| split(m[k]));
        let [r0, r1, r2, r3] = r;
        let p = [
            r0.as_mut_ptr(),
            r1.as_mut_ptr(),
            r2.as_mut_ptr(),
            r3.as_mut_ptr(),
        ];
        for i in (0..even).step_by(2) {
            // SAFETY: i + 1 < n ≤ every run's length.
            unsafe {
                let v = [
                    load(p[0].add(i)),
                    load(p[1].add(i)),
                    load(p[2].add(i)),
                    load(p[3].add(i)),
                ];
                for (k, &pk) in p.iter().enumerate() {
                    let row = &c[4 * k..4 * k + 4];
                    let mut s = cmul(row[0].0, row[0].1, v[0]);
                    s = _mm256_add_pd(s, cmul(row[1].0, row[1].1, v[1]));
                    s = _mm256_add_pd(s, cmul(row[2].0, row[2].1, v[2]));
                    s = _mm256_add_pd(s, cmul(row[3].0, row[3].1, v[3]));
                    store(pk.add(i), s);
                }
            }
        }
        portable::mat4_runs(
            [
                &mut r0[even..n],
                &mut r1[even..n],
                &mut r2[even..n],
                &mut r3[even..n],
            ],
            m,
        );
    }
}

/// Single-precision versions of the run primitives: AVX on x86-64 (four
/// amplitudes per register), NEON on aarch64 (two), the portable loops
/// elsewhere. Same rounding rules as the double-precision ones.
pub(crate) mod single {
    use num_complex::Complex32 as C32;

    #[cfg(target_arch = "x86_64")]
    use super::{AVX, level};
    #[cfg(target_arch = "aarch64")]
    use super::{NEON, level};

    macro_rules! dispatch {
        ($name:ident ( $($arg:ident : $ty:ty),* )) => {
            #[inline]
            pub(crate) fn $name($($arg: $ty),*) {
                #[cfg(target_arch = "aarch64")]
                if level() == NEON {
                    // SAFETY: NEON is available (detected).
                    unsafe { super::neon32::$name($($arg),*) };
                    return;
                }
                #[cfg(target_arch = "x86_64")]
                if level() == AVX {
                    // SAFETY: AVX is available (detected).
                    unsafe { super::avx32::$name($($arg),*) };
                    return;
                }
                super::super::scalar::portable::$name($($arg),*)
            }
        };
    }

    dispatch!(mat2_runs(a: &mut [C32], b: &mut [C32], m: &[C32; 4]));
    dispatch!(mat2_strided(x: &mut [C32], stride: usize, m: &[C32; 4]));
    dispatch!(mat2_strided_real(x: &mut [C32], stride: usize, m: &[f32; 4]));
    dispatch!(scale2_strided(x: &mut [C32], stride: usize, d0: C32, d1: C32));
    dispatch!(scale_upper_strided(x: &mut [C32], stride: usize, z: C32));
    dispatch!(mat2_runs_real(a: &mut [C32], b: &mut [C32], m: &[f32; 4]));
    dispatch!(scale_run(x: &mut [C32], z: C32));
    dispatch!(mat4_runs(r: [&mut [C32]; 4], m: &[C32; 16]));
}

#[cfg(target_arch = "x86_64")]
mod avx32 {
    //! Four complex amplitudes per 256-bit register.
    use std::arch::x86_64::*;

    use num_complex::Complex32 as C32;

    use super::super::scalar::portable;

    /// `m·x` for all four lanes, `m = (re, im)` broadcast.
    #[inline]
    #[target_feature(enable = "avx")]
    fn cmul(re: __m256, im: __m256, x: __m256) -> __m256 {
        let p = _mm256_mul_ps(re, x);
        let q = _mm256_mul_ps(im, _mm256_permute_ps::<0b1011_0001>(x));
        _mm256_addsub_ps(p, q)
    }

    #[inline]
    #[target_feature(enable = "avx")]
    fn split(z: C32) -> (__m256, __m256) {
        (_mm256_set1_ps(z.re), _mm256_set1_ps(z.im))
    }

    /// # Safety
    /// `p` is valid for reading four `C32`.
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn load(p: *const C32) -> __m256 {
        // SAFETY: by the caller; `C32` is `repr(C)` with two `f32`.
        unsafe { _mm256_loadu_ps(p.cast()) }
    }

    /// # Safety
    /// `p` is valid for writing four `C32`.
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn store(p: *mut C32, v: __m256) {
        // SAFETY: by the caller.
        unsafe { _mm256_storeu_ps(p.cast(), v) }
    }

    /// # Safety
    /// AVX is available.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn mat2_runs(a: &mut [C32], b: &mut [C32], m: &[C32; 4]) {
        let n = a.len().min(b.len());
        let full = n & !3;
        let (m0r, m0i) = split(m[0]);
        let (m1r, m1i) = split(m[1]);
        let (m2r, m2i) = split(m[2]);
        let (m3r, m3i) = split(m[3]);
        let (pa, pb) = (a.as_mut_ptr(), b.as_mut_ptr());
        for i in (0..full).step_by(4) {
            // SAFETY: i + 3 < n ≤ both lengths.
            unsafe {
                let x = load(pa.add(i));
                let y = load(pb.add(i));
                let u = _mm256_add_ps(cmul(m0r, m0i, x), cmul(m1r, m1i, y));
                let v = _mm256_add_ps(cmul(m2r, m2i, x), cmul(m3r, m3i, y));
                store(pa.add(i), u);
                store(pb.add(i), v);
            }
        }
        portable::mat2_runs(&mut a[full..n], &mut b[full..n], m);
    }

    /// # Safety
    /// AVX is available.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn mat2_runs_real(a: &mut [C32], b: &mut [C32], m: &[f32; 4]) {
        let n = a.len().min(b.len());
        let full = n & !3;
        let [m0, m1, m2, m3] = m.map(|x| _mm256_set1_ps(x));
        let (pa, pb) = (a.as_mut_ptr(), b.as_mut_ptr());
        for i in (0..full).step_by(4) {
            // SAFETY: i + 3 < n ≤ both lengths.
            unsafe {
                let x = load(pa.add(i));
                let y = load(pb.add(i));
                let u = _mm256_add_ps(_mm256_mul_ps(x, m0), _mm256_mul_ps(y, m1));
                let v = _mm256_add_ps(_mm256_mul_ps(x, m2), _mm256_mul_ps(y, m3));
                store(pa.add(i), u);
                store(pb.add(i), v);
            }
        }
        portable::mat2_runs_real(&mut a[full..n], &mut b[full..n], m);
    }

    /// # Safety
    /// AVX is available.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn scale_run(x: &mut [C32], z: C32) {
        let full = x.len() & !3;
        let (zr, zi) = split(z);
        let p = x.as_mut_ptr();
        for i in (0..full).step_by(4) {
            // SAFETY: i + 3 < x.len().
            unsafe { store(p.add(i), cmul(zr, zi, load(p.add(i)))) };
        }
        portable::scale_run(&mut x[full..], z);
    }

    /// Lanes `[a, b, c, d]` (complex) from four values.
    #[inline]
    #[target_feature(enable = "avx")]
    fn lanes(z: [C32; 4]) -> (__m256, __m256) {
        (
            _mm256_setr_ps(
                z[0].re, z[0].re, z[1].re, z[1].re, z[2].re, z[2].re, z[3].re, z[3].re,
            ),
            _mm256_setr_ps(
                z[0].im, z[0].im, z[1].im, z[1].im, z[2].im, z[2].im, z[3].im, z[3].im,
            ),
        )
    }

    /// Real lanes `[a, b, c, d]`, each on both parts of an amplitude.
    #[inline]
    #[target_feature(enable = "avx")]
    fn real_lanes(r: [f32; 4]) -> __m256 {
        _mm256_setr_ps(r[0], r[0], r[1], r[1], r[2], r[2], r[3], r[3])
    }

    /// The two halves of a register for the pairs of a gate with stride 1
    /// (`[u₀, v₀, u₁, v₁]` → `[u₀, u₀, u₁, u₁]`, `[v₀, v₀, v₁, v₁]`) or 2
    /// (`[u₀, u₁, v₀, v₁]` → `[u₀, u₁, u₀, u₁]`, `[v₀, v₁, v₀, v₁]`).
    #[inline]
    #[target_feature(enable = "avx")]
    fn halves(w: __m256, stride: usize) -> (__m256, __m256) {
        if stride == 1 {
            let d = _mm256_castps_pd(w);
            (
                _mm256_castpd_ps(_mm256_permute_pd::<0b0000>(d)),
                _mm256_castpd_ps(_mm256_permute_pd::<0b1111>(d)),
            )
        } else {
            (
                _mm256_permute2f128_ps::<0x00>(w, w),
                _mm256_permute2f128_ps::<0x11>(w, w),
            )
        }
    }

    /// Coefficient lanes for [`halves`]: row 0 where the pair's lower
    /// amplitude sits, row 1 where the upper one does.
    #[inline]
    fn rows<T: Copy>(lo: T, hi: T, stride: usize) -> [T; 4] {
        if stride == 1 {
            [lo, hi, lo, hi]
        } else {
            [lo, lo, hi, hi]
        }
    }

    /// # Safety
    /// AVX is available.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn mat2_strided(x: &mut [C32], stride: usize, m: &[C32; 4]) {
        if stride > 2 {
            for block in x.chunks_exact_mut(2 * stride) {
                let (lo, hi) = block.split_at_mut(stride);
                // SAFETY: AVX is available (by the caller).
                unsafe { mat2_runs(lo, hi, m) };
            }
            return;
        }
        let (c0r, c0i) = lanes(rows(m[0], m[2], stride));
        let (c1r, c1i) = lanes(rows(m[1], m[3], stride));
        let full = x.len() & !3;
        let p = x.as_mut_ptr();
        for i in (0..full).step_by(4) {
            // SAFETY: i + 3 < x.len().
            unsafe {
                let (u, v) = halves(load(p.add(i)), stride);
                store(
                    p.add(i),
                    _mm256_add_ps(cmul(c0r, c0i, u), cmul(c1r, c1i, v)),
                );
            }
        }
        portable::mat2_strided(&mut x[full..], stride, m);
    }

    /// # Safety
    /// AVX is available.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn mat2_strided_real(x: &mut [C32], stride: usize, m: &[f32; 4]) {
        if stride > 2 {
            for block in x.chunks_exact_mut(2 * stride) {
                let (lo, hi) = block.split_at_mut(stride);
                // SAFETY: AVX is available (by the caller).
                unsafe { mat2_runs_real(lo, hi, m) };
            }
            return;
        }
        let c0 = real_lanes(rows(m[0], m[2], stride));
        let c1 = real_lanes(rows(m[1], m[3], stride));
        let full = x.len() & !3;
        let p = x.as_mut_ptr();
        for i in (0..full).step_by(4) {
            // SAFETY: i + 3 < x.len().
            unsafe {
                let (u, v) = halves(load(p.add(i)), stride);
                store(
                    p.add(i),
                    _mm256_add_ps(_mm256_mul_ps(u, c0), _mm256_mul_ps(v, c1)),
                );
            }
        }
        portable::mat2_strided_real(&mut x[full..], stride, m);
    }

    /// # Safety
    /// AVX is available.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn scale2_strided(x: &mut [C32], stride: usize, d0: C32, d1: C32) {
        if stride > 2 {
            for block in x.chunks_exact_mut(2 * stride) {
                let (lo, hi) = block.split_at_mut(stride);
                // SAFETY: AVX is available (by the caller).
                unsafe {
                    scale_run(lo, d0);
                    scale_run(hi, d1);
                }
            }
            return;
        }
        let (re, im) = lanes(rows(d0, d1, stride));
        let full = x.len() & !3;
        let p = x.as_mut_ptr();
        for i in (0..full).step_by(4) {
            // SAFETY: i + 3 < x.len().
            unsafe { store(p.add(i), cmul(re, im, load(p.add(i)))) };
        }
        portable::scale2_strided(&mut x[full..], stride, d0, d1);
    }

    /// # Safety
    /// AVX is available.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn scale_upper_strided(x: &mut [C32], stride: usize, z: C32) {
        if stride > 2 {
            for block in x.chunks_exact_mut(2 * stride) {
                // SAFETY: AVX is available (by the caller).
                unsafe { scale_run(&mut block[stride..], z) };
            }
            return;
        }
        // Scale every lane, then keep the lower amplitudes as they were.
        let (zr, zi) = split(z);
        let full = x.len() & !3;
        let p = x.as_mut_ptr();
        for i in (0..full).step_by(4) {
            // SAFETY: i + 3 < x.len().
            unsafe {
                let w = load(p.add(i));
                let scaled = cmul(zr, zi, w);
                let out = if stride == 1 {
                    _mm256_blend_ps::<0b1100_1100>(w, scaled)
                } else {
                    _mm256_blend_ps::<0b1111_0000>(w, scaled)
                };
                store(p.add(i), out);
            }
        }
        portable::scale_upper_strided(&mut x[full..], stride, z);
    }

    /// # Safety
    /// AVX is available.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn mat4_runs(r: [&mut [C32]; 4], m: &[C32; 16]) {
        let n = r.iter().map(|x| x.len()).min().unwrap_or(0);
        let full = n & !3;
        let c: [(__m256, __m256); 16] = std::array::from_fn(|k| split(m[k]));
        let [r0, r1, r2, r3] = r;
        let p = [
            r0.as_mut_ptr(),
            r1.as_mut_ptr(),
            r2.as_mut_ptr(),
            r3.as_mut_ptr(),
        ];
        for i in (0..full).step_by(4) {
            // SAFETY: i + 3 < n ≤ every run's length.
            unsafe {
                let v = [
                    load(p[0].add(i)),
                    load(p[1].add(i)),
                    load(p[2].add(i)),
                    load(p[3].add(i)),
                ];
                for (k, &pk) in p.iter().enumerate() {
                    let row = &c[4 * k..4 * k + 4];
                    let mut s = cmul(row[0].0, row[0].1, v[0]);
                    s = _mm256_add_ps(s, cmul(row[1].0, row[1].1, v[1]));
                    s = _mm256_add_ps(s, cmul(row[2].0, row[2].1, v[2]));
                    s = _mm256_add_ps(s, cmul(row[3].0, row[3].1, v[3]));
                    store(pk.add(i), s);
                }
            }
        }
        portable::mat4_runs(
            [
                &mut r0[full..n],
                &mut r1[full..n],
                &mut r2[full..n],
                &mut r3[full..n],
            ],
            m,
        );
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    //! One complex amplitude `[re, im]` per register. The product `m·x`
    //! subtracts by adding the negated term, which rounds the same.

    use std::arch::aarch64::*;

    use num_complex::Complex64 as C64;

    /// `m·x` with `m = (re, im)` broadcast, and `sign = [−1, 1]`.
    #[inline]
    #[target_feature(enable = "neon")]
    fn cmul(re: float64x2_t, im: float64x2_t, sign: float64x2_t, x: float64x2_t) -> float64x2_t {
        let p = vmulq_f64(re, x);
        let q = vmulq_f64(vmulq_f64(im, vextq_f64::<1>(x, x)), sign);
        vaddq_f64(p, q)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn split(z: C64) -> (float64x2_t, float64x2_t) {
        (vdupq_n_f64(z.re), vdupq_n_f64(z.im))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn sign() -> float64x2_t {
        // SAFETY: two f64 on the stack.
        unsafe { vld1q_f64([-1.0, 1.0].as_ptr()) }
    }

    /// # Safety
    /// `p` is valid for reading one `C64`.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load(p: *const C64) -> float64x2_t {
        // SAFETY: by the caller; `C64` is `repr(C)` with two `f64`.
        unsafe { vld1q_f64(p.cast()) }
    }

    /// # Safety
    /// `p` is valid for writing one `C64`.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn store(p: *mut C64, v: float64x2_t) {
        // SAFETY: by the caller.
        unsafe { vst1q_f64(p.cast(), v) }
    }

    /// # Safety
    /// NEON is available.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn mat2_runs(a: &mut [C64], b: &mut [C64], m: &[C64; 4]) {
        let n = a.len().min(b.len());
        let s = sign();
        let (m0r, m0i) = split(m[0]);
        let (m1r, m1i) = split(m[1]);
        let (m2r, m2i) = split(m[2]);
        let (m3r, m3i) = split(m[3]);
        let (pa, pb) = (a.as_mut_ptr(), b.as_mut_ptr());
        for i in 0..n {
            // SAFETY: i < n ≤ both lengths.
            unsafe {
                let x = load(pa.add(i));
                let y = load(pb.add(i));
                let u = vaddq_f64(cmul(m0r, m0i, s, x), cmul(m1r, m1i, s, y));
                let v = vaddq_f64(cmul(m2r, m2i, s, x), cmul(m3r, m3i, s, y));
                store(pa.add(i), u);
                store(pb.add(i), v);
            }
        }
    }

    /// # Safety
    /// NEON is available.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn mat2_strided(x: &mut [C64], stride: usize, m: &[C64; 4]) {
        if stride == 1 {
            let s = sign();
            let c: [(float64x2_t, float64x2_t); 4] = std::array::from_fn(|k| split(m[k]));
            let p = x.as_mut_ptr();
            for i in (0..x.len() & !1).step_by(2) {
                // SAFETY: i + 1 < x.len().
                unsafe {
                    let u = load(p.add(i));
                    let v = load(p.add(i + 1));
                    let a = vaddq_f64(cmul(c[0].0, c[0].1, s, u), cmul(c[1].0, c[1].1, s, v));
                    let b = vaddq_f64(cmul(c[2].0, c[2].1, s, u), cmul(c[3].0, c[3].1, s, v));
                    store(p.add(i), a);
                    store(p.add(i + 1), b);
                }
            }
            return;
        }
        for block in x.chunks_exact_mut(2 * stride) {
            let (lo, hi) = block.split_at_mut(stride);
            // SAFETY: NEON is available (by the caller).
            unsafe { mat2_runs(lo, hi, m) };
        }
    }

    /// # Safety
    /// NEON is available.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn scale_run(x: &mut [C64], z: C64) {
        let s = sign();
        let (zr, zi) = split(z);
        let p = x.as_mut_ptr();
        for i in 0..x.len() {
            // SAFETY: i < x.len().
            unsafe { store(p.add(i), cmul(zr, zi, s, load(p.add(i)))) };
        }
    }

    /// # Safety
    /// NEON is available.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn scale_upper_strided(x: &mut [C64], stride: usize, z: C64) {
        for block in x.chunks_exact_mut(2 * stride) {
            // SAFETY: NEON is available (by the caller).
            unsafe { scale_run(&mut block[stride..], z) };
        }
    }

    /// # Safety
    /// NEON is available.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn scale2_strided(x: &mut [C64], stride: usize, d0: C64, d1: C64) {
        for block in x.chunks_exact_mut(2 * stride) {
            let (lo, hi) = block.split_at_mut(stride);
            // SAFETY: NEON is available (by the caller).
            unsafe {
                scale_run(lo, d0);
                scale_run(hi, d1);
            }
        }
    }

    /// # Safety
    /// NEON is available.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn mat4_runs(r: [&mut [C64]; 4], m: &[C64; 16]) {
        let n = r.iter().map(|x| x.len()).min().unwrap_or(0);
        let s = sign();
        let c: [(float64x2_t, float64x2_t); 16] = std::array::from_fn(|k| split(m[k]));
        let [r0, r1, r2, r3] = r;
        let p = [
            r0.as_mut_ptr(),
            r1.as_mut_ptr(),
            r2.as_mut_ptr(),
            r3.as_mut_ptr(),
        ];
        for i in 0..n {
            // SAFETY: i < n ≤ every run's length.
            unsafe {
                let v = [
                    load(p[0].add(i)),
                    load(p[1].add(i)),
                    load(p[2].add(i)),
                    load(p[3].add(i)),
                ];
                for (k, &pk) in p.iter().enumerate() {
                    let row = &c[4 * k..4 * k + 4];
                    let mut t = cmul(row[0].0, row[0].1, s, v[0]);
                    t = vaddq_f64(t, cmul(row[1].0, row[1].1, s, v[1]));
                    t = vaddq_f64(t, cmul(row[2].0, row[2].1, s, v[2]));
                    t = vaddq_f64(t, cmul(row[3].0, row[3].1, s, v[3]));
                    store(pk.add(i), t);
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon32 {
    //! Two complex amplitudes `[re₀, im₀, re₁, im₁]` per register. The
    //! product `m·x` subtracts by adding the negated term, which rounds
    //! the same.

    use std::arch::aarch64::*;

    use num_complex::Complex32 as C32;

    use super::super::scalar::portable;

    /// `m·x` for both lanes, `m = (re, im)` broadcast, `sign = [−1, 1, −1, 1]`.
    #[inline]
    #[target_feature(enable = "neon")]
    fn cmul(re: float32x4_t, im: float32x4_t, sign: float32x4_t, x: float32x4_t) -> float32x4_t {
        let p = vmulq_f32(re, x);
        let q = vmulq_f32(vmulq_f32(im, vrev64q_f32(x)), sign);
        vaddq_f32(p, q)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn split(z: C32) -> (float32x4_t, float32x4_t) {
        (vdupq_n_f32(z.re), vdupq_n_f32(z.im))
    }

    /// Lanes `[a, b]` (complex).
    #[inline]
    #[target_feature(enable = "neon")]
    fn pair(a: C32, b: C32) -> (float32x4_t, float32x4_t) {
        // SAFETY: four f32 on the stack.
        unsafe {
            (
                vld1q_f32([a.re, a.re, b.re, b.re].as_ptr()),
                vld1q_f32([a.im, a.im, b.im, b.im].as_ptr()),
            )
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn sign() -> float32x4_t {
        // SAFETY: four f32 on the stack.
        unsafe { vld1q_f32([-1.0, 1.0, -1.0, 1.0].as_ptr()) }
    }

    /// # Safety
    /// `p` is valid for reading two `C32`.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load(p: *const C32) -> float32x4_t {
        // SAFETY: by the caller; `C32` is `repr(C)` with two `f32`.
        unsafe { vld1q_f32(p.cast()) }
    }

    /// # Safety
    /// `p` is valid for writing two `C32`.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn store(p: *mut C32, v: float32x4_t) {
        // SAFETY: by the caller.
        unsafe { vst1q_f32(p.cast(), v) }
    }

    /// `[u, v]` → `[u, u]`, `[v, v]`.
    #[inline]
    #[target_feature(enable = "neon")]
    fn halves(w: float32x4_t) -> (float32x4_t, float32x4_t) {
        let (lo, hi) = (vget_low_f32(w), vget_high_f32(w));
        (vcombine_f32(lo, lo), vcombine_f32(hi, hi))
    }

    /// # Safety
    /// NEON is available.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn mat2_runs(a: &mut [C32], b: &mut [C32], m: &[C32; 4]) {
        let n = a.len().min(b.len());
        let full = n & !1;
        let s = sign();
        let (m0r, m0i) = split(m[0]);
        let (m1r, m1i) = split(m[1]);
        let (m2r, m2i) = split(m[2]);
        let (m3r, m3i) = split(m[3]);
        let (pa, pb) = (a.as_mut_ptr(), b.as_mut_ptr());
        for i in (0..full).step_by(2) {
            // SAFETY: i + 1 < n ≤ both lengths.
            unsafe {
                let x = load(pa.add(i));
                let y = load(pb.add(i));
                let u = vaddq_f32(cmul(m0r, m0i, s, x), cmul(m1r, m1i, s, y));
                let v = vaddq_f32(cmul(m2r, m2i, s, x), cmul(m3r, m3i, s, y));
                store(pa.add(i), u);
                store(pb.add(i), v);
            }
        }
        portable::mat2_runs(&mut a[full..n], &mut b[full..n], m);
    }

    /// # Safety
    /// NEON is available.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn mat2_runs_real(a: &mut [C32], b: &mut [C32], m: &[f32; 4]) {
        let n = a.len().min(b.len());
        let full = n & !1;
        let [m0, m1, m2, m3] = m.map(|x| vdupq_n_f32(x));
        let (pa, pb) = (a.as_mut_ptr(), b.as_mut_ptr());
        for i in (0..full).step_by(2) {
            // SAFETY: i + 1 < n ≤ both lengths.
            unsafe {
                let x = load(pa.add(i));
                let y = load(pb.add(i));
                store(pa.add(i), vaddq_f32(vmulq_f32(x, m0), vmulq_f32(y, m1)));
                store(pb.add(i), vaddq_f32(vmulq_f32(x, m2), vmulq_f32(y, m3)));
            }
        }
        portable::mat2_runs_real(&mut a[full..n], &mut b[full..n], m);
    }

    /// # Safety
    /// NEON is available.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn mat2_strided(x: &mut [C32], stride: usize, m: &[C32; 4]) {
        if stride > 1 {
            for block in x.chunks_exact_mut(2 * stride) {
                let (lo, hi) = block.split_at_mut(stride);
                // SAFETY: NEON is available (by the caller).
                unsafe { mat2_runs(lo, hi, m) };
            }
            return;
        }
        let s = sign();
        let (c0r, c0i) = pair(m[0], m[2]);
        let (c1r, c1i) = pair(m[1], m[3]);
        let p = x.as_mut_ptr();
        for i in (0..x.len() & !1).step_by(2) {
            // SAFETY: i + 1 < x.len().
            unsafe {
                let (u, v) = halves(load(p.add(i)));
                store(
                    p.add(i),
                    vaddq_f32(cmul(c0r, c0i, s, u), cmul(c1r, c1i, s, v)),
                );
            }
        }
    }

    /// # Safety
    /// NEON is available.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn mat2_strided_real(x: &mut [C32], stride: usize, m: &[f32; 4]) {
        if stride > 1 {
            for block in x.chunks_exact_mut(2 * stride) {
                let (lo, hi) = block.split_at_mut(stride);
                // SAFETY: NEON is available (by the caller).
                unsafe { mat2_runs_real(lo, hi, m) };
            }
            return;
        }
        // SAFETY: four f32 on the stack.
        let (c0, c1) = unsafe {
            (
                vld1q_f32([m[0], m[0], m[2], m[2]].as_ptr()),
                vld1q_f32([m[1], m[1], m[3], m[3]].as_ptr()),
            )
        };
        let p = x.as_mut_ptr();
        for i in (0..x.len() & !1).step_by(2) {
            // SAFETY: i + 1 < x.len().
            unsafe {
                let (u, v) = halves(load(p.add(i)));
                store(p.add(i), vaddq_f32(vmulq_f32(u, c0), vmulq_f32(v, c1)));
            }
        }
    }

    /// # Safety
    /// NEON is available.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn scale2_strided(x: &mut [C32], stride: usize, d0: C32, d1: C32) {
        if stride > 1 {
            for block in x.chunks_exact_mut(2 * stride) {
                let (lo, hi) = block.split_at_mut(stride);
                // SAFETY: NEON is available (by the caller).
                unsafe {
                    scale_run(lo, d0);
                    scale_run(hi, d1);
                }
            }
            return;
        }
        let s = sign();
        let (re, im) = pair(d0, d1);
        let p = x.as_mut_ptr();
        for i in (0..x.len() & !1).step_by(2) {
            // SAFETY: i + 1 < x.len().
            unsafe { store(p.add(i), cmul(re, im, s, load(p.add(i)))) };
        }
    }

    /// # Safety
    /// NEON is available.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn scale_upper_strided(x: &mut [C32], stride: usize, z: C32) {
        if stride > 1 {
            for block in x.chunks_exact_mut(2 * stride) {
                // SAFETY: NEON is available (by the caller).
                unsafe { scale_run(&mut block[stride..], z) };
            }
            return;
        }
        let s = sign();
        let (zr, zi) = split(z);
        let p = x.as_mut_ptr();
        for i in (0..x.len() & !1).step_by(2) {
            // SAFETY: i + 1 < x.len().
            unsafe {
                let w = load(p.add(i));
                let scaled = cmul(zr, zi, s, w);
                store(
                    p.add(i),
                    vcombine_f32(vget_low_f32(w), vget_high_f32(scaled)),
                );
            }
        }
    }

    /// # Safety
    /// NEON is available.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn scale_run(x: &mut [C32], z: C32) {
        let full = x.len() & !1;
        let s = sign();
        let (zr, zi) = split(z);
        let p = x.as_mut_ptr();
        for i in (0..full).step_by(2) {
            // SAFETY: i + 1 < x.len().
            unsafe { store(p.add(i), cmul(zr, zi, s, load(p.add(i)))) };
        }
        portable::scale_run(&mut x[full..], z);
    }

    /// # Safety
    /// NEON is available.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn mat4_runs(r: [&mut [C32]; 4], m: &[C32; 16]) {
        let n = r.iter().map(|x| x.len()).min().unwrap_or(0);
        let full = n & !1;
        let s = sign();
        let c: [(float32x4_t, float32x4_t); 16] = std::array::from_fn(|k| split(m[k]));
        let [r0, r1, r2, r3] = r;
        let p = [
            r0.as_mut_ptr(),
            r1.as_mut_ptr(),
            r2.as_mut_ptr(),
            r3.as_mut_ptr(),
        ];
        for i in (0..full).step_by(2) {
            // SAFETY: i + 1 < n ≤ every run's length.
            unsafe {
                let v = [
                    load(p[0].add(i)),
                    load(p[1].add(i)),
                    load(p[2].add(i)),
                    load(p[3].add(i)),
                ];
                for (k, &pk) in p.iter().enumerate() {
                    let row = &c[4 * k..4 * k + 4];
                    let mut acc = cmul(row[0].0, row[0].1, s, v[0]);
                    acc = vaddq_f32(acc, cmul(row[1].0, row[1].1, s, v[1]));
                    acc = vaddq_f32(acc, cmul(row[2].0, row[2].1, s, v[2]));
                    acc = vaddq_f32(acc, cmul(row[3].0, row[3].1, s, v[3]));
                    store(pk.add(i), acc);
                }
            }
        }
        portable::mat4_runs(
            [
                &mut r0[full..n],
                &mut r1[full..n],
                &mut r2[full..n],
                &mut r3[full..n],
            ],
            m,
        );
    }
}
