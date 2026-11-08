//! Local maps of strings on one to four qubits: the transfer matrices of
//! gates and of the adjoints of channels, snapping of tiny entries into a
//! residual, and the cache of tables.
//!
//! A local string on the `k` qubits of an op is indexed by
//! `b = xbits | zbits << k`, bit `j` of each mask for the op's `j`-th
//! qubit. In a gate or Kraus matrix the op's `j`-th qubit is matrix bit
//! `k − 1 − j` (the first qubit most significant, as
//! [`instruction_matrix`](crate::quantum::gates::instruction_matrix) and
//! the density-matrix kernels order them).

use std::collections::BTreeMap;
use std::sync::Arc;

use num_complex::Complex64 as C64;

use crate::error::Result;
use crate::noise::Channel;

/// Entries of a table with `|T| ≤ SNAP` are dropped and their magnitude
/// kept in the residual of the input string: rotation branches at
/// multiples of π/2 (`cos(π/2) = 6.1·10⁻¹⁷`) and the rounding junk of
/// transfer matrices built from matrices.
pub(crate) const SNAP: f64 = 1e-14;

/// The images of every local string under one op, `P_b ↦ Σ f·P_{b'}`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LocalMap {
    k: usize,
    /// The images of `b` are `images[start[b]..start[b + 1]]`.
    start: Vec<u32>,
    /// `(b', f)`, ascending in `b'` for each `b`.
    images: Vec<(u16, f64)>,
    /// `Σ |f|` of the entries of each column dropped by snapping.
    residual: Vec<f64>,
    /// At most one image per string.
    permutation: bool,
    /// Every string has exactly one image and no two strings share one:
    /// the op maps distinct strings to distinct strings, so a table needs
    /// no merging after it.
    bijective: bool,
    /// The strings the op leaves exactly as they are (one image, itself,
    /// with factor 1 and nothing snapped).
    fixed: Vec<bool>,
    /// The letters every image of each string carries, the snapped ones
    /// included: bit `l` of nibble `j` for letter `l = x | z << 1` on the
    /// op's `j`-th qubit.
    reach: Vec<u16>,
}

/// The qubits (bit `j` for the op's `j`-th) where local string `b` is not
/// the identity.
#[cfg(test)]
pub(crate) fn local_support(b: usize, k: usize) -> usize {
    (b | (b >> k)) & ((1 << k) - 1)
}

/// The letter of local string `b` on each of the `k` qubits, as bit
/// `l = x | z << 1` of nibble `j` for the `j`-th qubit.
fn letters_of(b: usize, k: usize) -> u16 {
    (0..k)
        .map(|j| 1u16 << (4 * j + (((b >> j) & 1) | (((b >> (k + j)) & 1) << 1))))
        .fold(0, |a, x| a | x)
}

impl LocalMap {
    /// The map with `columns[b]` = the images `(b', f)` of string `b`
    /// (distinct `b'`), snapped: entries with `|f| ≤ SNAP` go into the
    /// residual of `b`, and the identity maps exactly to itself (the map is
    /// unital).
    pub(crate) fn from_columns(k: usize, columns: Vec<Vec<(usize, f64)>>) -> Self {
        let n = 1usize << (2 * k);
        debug_assert_eq!(columns.len(), n);
        let mut start = Vec::with_capacity(n + 1);
        let mut images: Vec<(u16, f64)> = Vec::new();
        let mut residual = vec![0.0; n];
        let mut reach = vec![0u16; n];
        let mut permutation = true;
        start.push(0);
        for (b, column) in columns.into_iter().enumerate() {
            let first = images.len();
            if b == 0 {
                images.push((0, 1.0));
                reach[0] = letters_of(0, k);
            } else {
                for (image, f) in column {
                    debug_assert!(image < n);
                    // Snapped images count: the exact map carries them on.
                    if f != 0.0 {
                        reach[b] |= letters_of(image, k);
                    }
                    if f.abs() <= SNAP {
                        residual[b] += f.abs();
                    } else {
                        images.push((image as u16, f));
                    }
                }
                images[first..].sort_by_key(|&(image, _)| image);
                debug_assert!(images[first..].windows(2).all(|w| w[0].0 < w[1].0));
            }
            permutation &= images.len() - first <= 1;
            start.push(images.len() as u32);
        }
        let mut hit = vec![false; n];
        let bijective = permutation
            && images.len() == n
            && images.iter().all(|&(image, _)| {
                let fresh = !hit[image as usize];
                hit[image as usize] = true;
                fresh
            });
        let fixed = (0..n)
            .map(|b| {
                let s = start[b] as usize..start[b + 1] as usize;
                residual[b] == 0.0 && images[s] == [(b as u16, 1.0)]
            })
            .collect();
        LocalMap {
            k,
            start,
            images,
            residual,
            permutation,
            bijective,
            fixed,
            reach,
        }
    }

    /// The letters that the images of string `b` carry, the snapped ones
    /// included: bit `l` of nibble `j` for letter `l = x | z << 1` (I, X,
    /// Z, Y) on the op's `j`-th qubit.
    #[inline]
    pub(crate) fn reach(&self, b: usize) -> u16 {
        self.reach[b]
    }

    /// The images `(b', f)` of local string `b`, ascending in `b'`.
    #[inline]
    pub(crate) fn images(&self, b: usize) -> &[(u16, f64)] {
        &self.images[self.start[b] as usize..self.start[b + 1] as usize]
    }

    /// Whether the op leaves string `b` exactly as it is.
    #[inline]
    pub(crate) fn is_fixed(&self, b: usize) -> bool {
        self.fixed[b]
    }

    /// `Σ |f|` of the entries of `b`'s column dropped by snapping.
    #[inline]
    pub(crate) fn residual(&self, b: usize) -> f64 {
        self.residual[b]
    }

    /// At most one image per string: the op permutes strings (up to
    /// factors) and needs no merging.
    #[cfg(test)]
    pub(crate) fn is_permutation(&self) -> bool {
        self.permutation
    }

    /// Every string has exactly one image, and distinct strings have
    /// distinct images: the op can be applied to a table in place.
    pub(crate) fn is_bijective(&self) -> bool {
        self.bijective
    }

    /// Whether some image has a letter on a qubit where its string has `I`
    /// (the op can grow the support of a string).
    #[cfg(test)]
    pub(crate) fn grows(&self) -> bool {
        (0..self.residual.len()).any(|b| {
            let support = local_support(b, self.k);
            self.images(b)
                .iter()
                .any(|&(image, _)| local_support(image as usize, self.k) & !support != 0)
        })
    }
}

/// The local Pauli `a` in matrix form: `P_a|j⟩ = i^{#Y}·(−1)^{|j ∧ z|}·|j ⊕ x⟩`
/// with the masks `(x, z)` in matrix bits and the factor `i^{#Y}`.
fn matrix_form(a: usize, k: usize) -> (usize, usize, C64) {
    let (mut x, mut z) = (0usize, 0usize);
    for j in 0..k {
        x |= ((a >> j) & 1) << (k - 1 - j);
        z |= ((a >> (k + j)) & 1) << (k - 1 - j);
    }
    let phase = match (x & z).count_ones() % 4 {
        0 => C64::new(1.0, 0.0),
        1 => C64::new(0.0, 1.0),
        2 => C64::new(-1.0, 0.0),
        _ => C64::new(0.0, -1.0),
    };
    (x, z, phase)
}

/// `⟨j ⊕ x| P_a |j⟩` for the matrix form `(x, z, phase)` of `P_a`.
#[inline]
fn entry(z: usize, phase: C64, j: usize) -> C64 {
    if (j & z).count_ones().is_multiple_of(2) {
        phase
    } else {
        -phase
    }
}

/// The columns `T_ab = 2^{−k} Re Tr(E(P_a) P_b)` (images `a` of every `b`,
/// ascending in `a`) of a map whose Schrödinger action on `P_a` is
/// `image(a)`, a dense row-major `2^k × 2^k` matrix.
fn columns_from_images(
    k: usize,
    mut image: impl FnMut(usize) -> Vec<C64>,
) -> Vec<Vec<(usize, f64)>> {
    let (n, d) = (1usize << (2 * k), 1usize << k);
    let forms: Vec<(usize, usize, C64)> = (0..n).map(|b| matrix_form(b, k)).collect();
    let mut columns = vec![Vec::with_capacity(n); n];
    for a in 0..n {
        let e = image(a);
        for (b, &(xb, zb, phase)) in forms.iter().enumerate() {
            // Tr(E P_b) = Σ_j E[j][j ⊕ x_b]·⟨j ⊕ x_b|P_b|j⟩.
            let mut trace = C64::new(0.0, 0.0);
            for j in 0..d {
                trace += e[j * d + (j ^ xb)] * entry(zb, phase, j);
            }
            columns[b].push((a, trace.re / d as f64));
        }
    }
    columns
}

/// The table of the unitary `u` (row-major, `2^k × 2^k`): the images of
/// `P_b` under `P ↦ U† P U`.
pub(crate) fn gate_table(u: &[C64], k: usize) -> LocalMap {
    let d = 1usize << k;
    debug_assert_eq!(u.len(), d * d);
    let columns = columns_from_images(k, |a| {
        // U P_a U†, with (U P_a)[r][j] = U[r][j ⊕ x_a]·⟨j ⊕ x_a|P_a|j⟩.
        let (xa, za, phase) = matrix_form(a, k);
        let mut up = vec![C64::new(0.0, 0.0); d * d];
        for r in 0..d {
            for j in 0..d {
                up[r * d + j] = u[r * d + (j ^ xa)] * entry(za, phase, j);
            }
        }
        let mut e = vec![C64::new(0.0, 0.0); d * d];
        for r in 0..d {
            for c in 0..d {
                let mut acc = C64::new(0.0, 0.0);
                for j in 0..d {
                    acc += up[r * d + j] * u[c * d + j].conj();
                }
                e[r * d + c] = acc;
            }
        }
        e
    });
    LocalMap::from_columns(k, columns)
}

/// The table of the adjoint of `channel`: the images of `P_b` under
/// `E†`, from the superoperator of `E`.
pub(crate) fn channel_table(channel: &Channel) -> LocalMap {
    let k = channel.n_qubits();
    let (d, dd) = (1usize << k, 1usize << (2 * k));
    let s = channel.superoperator();
    let columns = columns_from_images(k, |a| {
        // E(P_a)[r][c] = Σ_j S[(r, c), (j ⊕ x_a, j)]·⟨j ⊕ x_a|P_a|j⟩.
        let (xa, za, phase) = matrix_form(a, k);
        let mut e = vec![C64::new(0.0, 0.0); d * d];
        for (rc, out) in e.iter_mut().enumerate() {
            let row = &s[rc * dd..(rc + 1) * dd];
            let mut acc = C64::new(0.0, 0.0);
            for j in 0..d {
                acc += row[(j ^ xa) * d + j] * entry(za, phase, j);
            }
            *out = acc;
        }
        e
    });
    LocalMap::from_columns(k, columns)
}

/// Key of a gate table: kind, number of controls, number of targets, the
/// bits of the angles and of a custom matrix (re, im). It does not depend
/// on the qubits, so the cache is deterministic.
pub(crate) type GateKey = (u8, usize, usize, Vec<u64>, Vec<u64>);

/// Tables built so far, shared by every op with the same key.
#[derive(Debug, Default)]
pub(crate) struct TableCache {
    gates: BTreeMap<GateKey, Arc<LocalMap>>,
    /// Keyed by the number of qubits and the bits of the Kraus operators.
    channels: BTreeMap<Vec<u64>, Arc<LocalMap>>,
}

impl TableCache {
    /// The table of a gate with `key`, built by `build` on first use.
    pub(crate) fn gate(
        &mut self,
        key: GateKey,
        build: impl FnOnce() -> Result<LocalMap>,
    ) -> Result<Arc<LocalMap>> {
        if let Some(map) = self.gates.get(&key) {
            return Ok(Arc::clone(map));
        }
        let map = Arc::new(build()?);
        self.gates.insert(key, Arc::clone(&map));
        Ok(map)
    }

    /// The table of the adjoint of `channel`.
    pub(crate) fn channel(&mut self, channel: &Channel) -> Arc<LocalMap> {
        let mut key = vec![channel.n_qubits() as u64];
        for kraus in channel.kraus_operators() {
            key.push(kraus.len() as u64);
            key.extend(kraus.iter().flat_map(|z| [z.re.to_bits(), z.im.to_bits()]));
        }
        Arc::clone(
            self.channels
                .entry(key)
                .or_insert_with(|| Arc::new(channel_table(channel))),
        )
    }

    /// Number of distinct tables built.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.gates.len() + self.channels.len()
    }
}
