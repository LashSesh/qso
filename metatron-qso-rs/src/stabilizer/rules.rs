//! The signed conjugation rules of the Clifford primitives on bit-packed
//! Pauli strings (Aaronson and Gottesman, Phys. Rev. A 70, 052328 (2004)).
//!
//! Every rule maps the X bits `x`, the Z bits `z` and the sign bits `r` of
//! up to 64 strings at once (bit `i` of each word belongs to string `i`)
//! to those of `G P G†` (Schrödinger order); `Y` sets both bits and is the
//! Hermitian `Y`. The rules use only `^`, `&`, `!` and swaps, so they act
//! on single bits as well as on whole words. The tableau
//! ([`StabilizerState`](super::StabilizerState)) applies them to the
//! columns of its generators, Pauli propagation to one string at a time.

/// Hadamard: `X ↔ Z`, `Y → −Y`.
#[inline]
pub(crate) fn h(x: &mut u64, z: &mut u64, r: &mut u64) {
    *r ^= *x & *z;
    std::mem::swap(x, z);
}

/// `S = diag(1, i)`: `X → Y`, `Y → −X`.
#[inline]
pub(crate) fn s(x: &mut u64, z: &mut u64, r: &mut u64) {
    *r ^= *x & *z;
    *z ^= *x;
}

/// `S† = diag(1, −i)`: `X → −Y`, `Y → X`.
#[inline]
pub(crate) fn sdg(x: &mut u64, z: &mut u64, r: &mut u64) {
    *r ^= *x & !*z;
    *z ^= *x;
}

/// Pauli `X`: `Y → −Y`, `Z → −Z`.
#[inline]
pub(crate) fn x(_x: &mut u64, z: &mut u64, r: &mut u64) {
    *r ^= *z;
}

/// Pauli `Y`: `X → −X`, `Z → −Z`.
#[inline]
pub(crate) fn y(x: &mut u64, z: &mut u64, r: &mut u64) {
    *r ^= *x ^ *z;
}

/// Pauli `Z`: `X → −X`, `Y → −Y`.
#[inline]
pub(crate) fn z(x: &mut u64, _z: &mut u64, r: &mut u64) {
    *r ^= *x;
}

/// CNOT with control bits `(xc, zc)` and target bits `(xt, zt)`:
/// `X_c → X_c X_t`, `Z_t → Z_c Z_t`.
#[inline]
pub(crate) fn cx(xc: &mut u64, zc: &mut u64, xt: &mut u64, zt: &mut u64, r: &mut u64) {
    *r ^= *xc & *zt & !(*xt ^ *zc);
    *xt ^= *xc;
    *zc ^= *zt;
}

/// SWAP of the qubits with bits `(xa, za)` and `(xb, zb)`.
#[inline]
pub(crate) fn swap(xa: &mut u64, za: &mut u64, xb: &mut u64, zb: &mut u64) {
    std::mem::swap(xa, xb);
    std::mem::swap(za, zb);
}
