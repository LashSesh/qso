//! Amplitudes in double or single precision: the run primitives of the
//! gate kernels for each.
//!
//! The kernels are generic over [`Amplitude`]. Double precision
//! (`Complex64`) uses the vector primitives of [`super::simd`]; single
//! precision (`Complex32`) uses AVX versions with four amplitudes per
//! register on x86-64 and the portable loops below elsewhere.
//! Gate matrices are formed in double precision and rounded once per
//! gate.

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub};

use num_complex::{Complex32 as C32, Complex64 as C64};

use super::simd;

/// A complex amplitude: `Complex64` or `Complex32`.
pub trait Amplitude:
    Copy
    + Send
    + Sync
    + PartialEq
    + Debug
    + Default
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + AddAssign
    + MulAssign
{
    /// The real type of the parts.
    type Real: Copy + Send + Sync + PartialEq + Debug + 'static;

    const ZERO: Self;

    /// Rounded from double precision.
    fn from_c64(z: C64) -> Self;
    fn to_c64(self) -> C64;
    fn conj(self) -> Self;
    /// The real part.
    fn re(self) -> Self::Real;
    /// `self · r` for a real `r`.
    fn scale_real(self, r: Self::Real) -> Self;
    /// |z|² in double precision.
    fn norm_sqr_f64(self) -> f64;

    /// `a ← m₀₀·a + m₀₁·b`, `b ← m₁₀·a + m₁₁·b` over two runs.
    fn mat2_runs(a: &mut [Self], b: &mut [Self], m: &[Self; 4]);
    /// [`Amplitude::mat2_runs`] for a real matrix.
    fn mat2_runs_real(a: &mut [Self], b: &mut [Self], m: &[Self::Real; 4]);
    /// The 2×2 matrix on every pair `(x[i], x[i + stride])` with bit
    /// `stride` of `i` clear.
    fn mat2_strided(x: &mut [Self], stride: usize, m: &[Self; 4]);
    fn mat2_strided_real(x: &mut [Self], stride: usize, m: &[Self::Real; 4]);
    /// The lower and upper amplitude of every pair scaled by `d0`, `d1`.
    fn scale2_strided(x: &mut [Self], stride: usize, d0: Self, d1: Self);
    /// The upper amplitude of every pair scaled by `z`.
    fn scale_upper_strided(x: &mut [Self], stride: usize, z: Self);
    fn scale_run(x: &mut [Self], z: Self);
    /// The 4×4 matrix on `(r[0][i], …, r[3][i])` for every `i`.
    fn mat4_runs(r: [&mut [Self]; 4], m: &[Self; 16]);
}

impl Amplitude for C64 {
    type Real = f64;
    const ZERO: Self = C64::new(0.0, 0.0);

    #[inline]
    fn from_c64(z: C64) -> Self {
        z
    }
    #[inline]
    fn to_c64(self) -> C64 {
        self
    }
    #[inline]
    fn conj(self) -> Self {
        C64::conj(&self)
    }
    #[inline]
    fn re(self) -> f64 {
        self.re
    }
    #[inline]
    fn scale_real(self, r: f64) -> Self {
        self * r
    }
    #[inline]
    fn norm_sqr_f64(self) -> f64 {
        self.norm_sqr()
    }
    #[inline]
    fn mat2_runs(a: &mut [Self], b: &mut [Self], m: &[Self; 4]) {
        simd::mat2_runs(a, b, m)
    }
    #[inline]
    fn mat2_runs_real(a: &mut [Self], b: &mut [Self], m: &[f64; 4]) {
        simd::mat2_runs_real(a, b, m)
    }
    #[inline]
    fn mat2_strided(x: &mut [Self], stride: usize, m: &[Self; 4]) {
        simd::mat2_strided(x, stride, m)
    }
    #[inline]
    fn mat2_strided_real(x: &mut [Self], stride: usize, m: &[f64; 4]) {
        simd::mat2_strided_real(x, stride, m)
    }
    #[inline]
    fn scale2_strided(x: &mut [Self], stride: usize, d0: Self, d1: Self) {
        simd::scale2_strided(x, stride, d0, d1)
    }
    #[inline]
    fn scale_upper_strided(x: &mut [Self], stride: usize, z: Self) {
        simd::scale_upper_strided(x, stride, z)
    }
    #[inline]
    fn scale_run(x: &mut [Self], z: Self) {
        simd::scale_run(x, z)
    }
    #[inline]
    fn mat4_runs(r: [&mut [Self]; 4], m: &[Self; 16]) {
        simd::mat4_runs(r, m)
    }
}

impl Amplitude for C32 {
    type Real = f32;
    const ZERO: Self = C32::new(0.0, 0.0);

    #[inline]
    fn from_c64(z: C64) -> Self {
        C32::new(z.re as f32, z.im as f32)
    }
    #[inline]
    fn to_c64(self) -> C64 {
        C64::new(self.re as f64, self.im as f64)
    }
    #[inline]
    fn conj(self) -> Self {
        C32::conj(&self)
    }
    #[inline]
    fn re(self) -> f32 {
        self.re
    }
    #[inline]
    fn scale_real(self, r: f32) -> Self {
        self * r
    }
    #[inline]
    fn norm_sqr_f64(self) -> f64 {
        let (re, im) = (self.re as f64, self.im as f64);
        re * re + im * im
    }
    #[inline]
    fn mat2_runs(a: &mut [Self], b: &mut [Self], m: &[Self; 4]) {
        simd::single::mat2_runs(a, b, m)
    }
    #[inline]
    fn mat2_runs_real(a: &mut [Self], b: &mut [Self], m: &[f32; 4]) {
        simd::single::mat2_runs_real(a, b, m)
    }
    #[inline]
    fn mat2_strided(x: &mut [Self], stride: usize, m: &[Self; 4]) {
        simd::single::mat2_strided(x, stride, m)
    }
    #[inline]
    fn mat2_strided_real(x: &mut [Self], stride: usize, m: &[f32; 4]) {
        simd::single::mat2_strided_real(x, stride, m)
    }
    #[inline]
    fn scale2_strided(x: &mut [Self], stride: usize, d0: Self, d1: Self) {
        simd::single::scale2_strided(x, stride, d0, d1)
    }
    #[inline]
    fn scale_upper_strided(x: &mut [Self], stride: usize, z: Self) {
        simd::single::scale_upper_strided(x, stride, z)
    }
    #[inline]
    fn scale_run(x: &mut [Self], z: Self) {
        simd::single::scale_run(x, z)
    }
    #[inline]
    fn mat4_runs(r: [&mut [Self]; 4], m: &[Self; 16]) {
        simd::single::mat4_runs(r, m)
    }
}

/// The primitives for any amplitude type, with the arithmetic of the
/// double-precision portable versions.
pub(crate) mod portable {
    use super::Amplitude;

    #[inline]
    pub(crate) fn mat2_runs<A: Amplitude>(a: &mut [A], b: &mut [A], m: &[A; 4]) {
        for (x, y) in a.iter_mut().zip(b) {
            let (u, v) = (*x, *y);
            *x = m[0] * u + m[1] * v;
            *y = m[2] * u + m[3] * v;
        }
    }

    #[inline]
    pub(crate) fn mat2_runs_real<A: Amplitude>(a: &mut [A], b: &mut [A], m: &[A::Real; 4]) {
        for (x, y) in a.iter_mut().zip(b) {
            let (u, v) = (*x, *y);
            *x = u.scale_real(m[0]) + v.scale_real(m[1]);
            *y = u.scale_real(m[2]) + v.scale_real(m[3]);
        }
    }

    #[inline]
    pub(crate) fn scale_run<A: Amplitude>(x: &mut [A], z: A) {
        for a in x {
            *a *= z;
        }
    }

    #[inline]
    pub(crate) fn mat2_strided<A: Amplitude>(x: &mut [A], stride: usize, m: &[A; 4]) {
        if stride == 1 {
            for p in x.as_chunks_mut::<2>().0 {
                let (u, v) = (p[0], p[1]);
                p[0] = m[0] * u + m[1] * v;
                p[1] = m[2] * u + m[3] * v;
            }
            return;
        }
        for block in x.chunks_exact_mut(2 * stride) {
            let (lo, hi) = block.split_at_mut(stride);
            mat2_runs(lo, hi, m);
        }
    }

    #[inline]
    pub(crate) fn mat2_strided_real<A: Amplitude>(x: &mut [A], stride: usize, m: &[A::Real; 4]) {
        if stride == 1 {
            for p in x.as_chunks_mut::<2>().0 {
                let (u, v) = (p[0], p[1]);
                p[0] = u.scale_real(m[0]) + v.scale_real(m[1]);
                p[1] = u.scale_real(m[2]) + v.scale_real(m[3]);
            }
            return;
        }
        for block in x.chunks_exact_mut(2 * stride) {
            let (lo, hi) = block.split_at_mut(stride);
            mat2_runs_real(lo, hi, m);
        }
    }

    #[inline]
    pub(crate) fn scale_upper_strided<A: Amplitude>(x: &mut [A], stride: usize, z: A) {
        for block in x.chunks_exact_mut(2 * stride) {
            scale_run(&mut block[stride..], z);
        }
    }

    #[inline]
    pub(crate) fn scale2_strided<A: Amplitude>(x: &mut [A], stride: usize, d0: A, d1: A) {
        for block in x.chunks_exact_mut(2 * stride) {
            let (lo, hi) = block.split_at_mut(stride);
            scale_run(lo, d0);
            scale_run(hi, d1);
        }
    }

    #[inline]
    pub(crate) fn mat4_runs<A: Amplitude>(r: [&mut [A]; 4], m: &[A; 16]) {
        let [r0, r1, r2, r3] = r;
        for i in 0..r0.len() {
            let v = [r0[i], r1[i], r2[i], r3[i]];
            let out: [A; 4] = std::array::from_fn(|k| {
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
