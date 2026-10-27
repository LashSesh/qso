//! Stabilizer simulation: Clifford circuits on thousands of qubits.
//!
//! A [`StabilizerState`] stores an n-qubit stabilizer state as the
//! tableau of Aaronson and Gottesman (Phys. Rev. A 70, 052328 (2004)):
//! n destabilizer and n stabilizer generators, each a Pauli string with a
//! sign, bit-packed into 64-bit words and stored by qubit. A Clifford gate
//! costs O(n/32) word operations; a measurement or an expectation value
//! O(n²) (the tableau is transposed to rows and back); sampling every qubit
//! O(n³/64) once, then O(n²/64) per shot. At 16 384 qubits a measurement
//! takes about a second and the sampler about two minutes.
//!
//! Clifford gates are H, S, S†, X, Y, Z, √X, √X†, SWAP, CX, CY and CZ, and
//! the rotations RX, RY, RZ, P, RXX, RYY, RZZ and U whose angles are
//! multiples of π/2. Global phases are not tracked: a stabilizer state is
//! defined up to one.
//!
//! [`graph_state`](StabilizerState::graph_state) builds the graph state of
//! any [`Graph`]; [`program`] runs OpenQASM programs
//! made of Clifford gates, mid-circuit measurements and conditions
//! included.
//!
//! Pauli operators are written sparsely, `"X0 Z1 Y1000"`, with an optional
//! sign `+` or `-` in front.

pub mod program;

use std::collections::BTreeMap;
use std::f64::consts::FRAC_PI_2;

use num_complex::Complex64 as C64;

use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::quantum::{GateKind, Instruction, Pauli, PauliString, PauliSum, StateVector};
use crate::rng::Rng;

/// Largest register a [`StabilizerState`] may hold. Memory grows as n²/2
/// bytes (128 MiB at the limit), measurements as n² and sampling all
/// qubits as n³.
pub const MAX_STABILIZER_QUBITS: usize = 1 << 14;

/// Most bits [`StabilizerState::sample_counts`] returns (shots × qubits).
pub const MAX_SAMPLED_BITS: usize = 1 << 26;

/// Largest register [`StabilizerState::to_statevector`] converts.
pub const MAX_STATEVECTOR_QUBITS: usize = 20;

/// Sign and Pauli string, bit-packed: bit `q` of `x`/`z` is qubit `q`
/// (`X` = x, `Z` = z, `Y` = both).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Row {
    x: Vec<u64>,
    z: Vec<u64>,
    /// −1 sign.
    neg: bool,
}

impl Row {
    fn zero(words: usize) -> Self {
        Row {
            x: vec![0; words],
            z: vec![0; words],
            neg: false,
        }
    }

    fn get(&self, q: usize) -> (bool, bool) {
        let (w, b) = (q / 64, q % 64);
        ((self.x[w] >> b) & 1 == 1, (self.z[w] >> b) & 1 == 1)
    }

    fn set(&mut self, q: usize, x: bool, z: bool) {
        let (w, b) = (q / 64, 1u64 << (q % 64));
        self.x[w] = if x { self.x[w] | b } else { self.x[w] & !b };
        self.z[w] = if z { self.z[w] | b } else { self.z[w] & !b };
    }

    /// Whether the two Pauli strings anticommute.
    fn anticommutes(&self, other: &Row) -> bool {
        let mut parity = 0u32;
        for w in 0..self.x.len() {
            parity ^= ((self.x[w] & other.z[w]) ^ (self.z[w] & other.x[w])).count_ones();
        }
        parity & 1 == 1
    }

    /// `self ← other · self`, with the sign of the product (both must
    /// commute, so the product is Hermitian).
    fn left_multiply(&mut self, other: &Row) {
        // Σ_q g(other_q, self_q): the power of i of the single-qubit
        // products (Aaronson and Gottesman's function g), word by word.
        let (mut pos, mut neg) = (0u32, 0u32);
        for w in 0..self.x.len() {
            let (x1, z1, x2, z2) = (other.x[w], other.z[w], self.x[w], self.z[w]);
            let y1 = x1 & z1;
            let xo = x1 & !z1;
            let zo = !x1 & z1;
            pos += ((y1 & z2 & !x2) | (xo & z2 & x2) | (zo & x2 & !z2)).count_ones();
            neg += ((y1 & x2 & !z2) | (xo & z2 & !x2) | (zo & x2 & z2)).count_ones();
            self.x[w] = x2 ^ x1;
            self.z[w] = z2 ^ z1;
        }
        let total = (2 * i64::from(self.neg) + 2 * i64::from(other.neg) + i64::from(pos)
            - i64::from(neg))
        .rem_euclid(4);
        debug_assert!(total % 2 == 0, "product of commuting Paulis is Hermitian");
        self.neg = total == 2;
    }
}

/// An n-qubit stabilizer state (Aaronson–Gottesman tableau).
///
/// The 2n generators (rows `0..n` destabilizers, `n..2n` stabilizers) are
/// stored by qubit: column `q` of the X part is a bit vector over the
/// rows, so a gate touches two or four contiguous columns. Measurements
/// work on rows and transpose the tableau in 64 × 64 blocks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StabilizerState {
    n: usize,
    /// Words per row (n bits).
    words: usize,
    /// Words per column (2n bits, one per row).
    rw: usize,
    /// X and Z parts, column-major: `x[q * rw + w]`.
    x: Vec<u64>,
    z: Vec<u64>,
    /// Signs, one bit per row.
    r: Vec<u64>,
}

/// Transpose a 64 × 64 bit matrix in place (`a[k]` bit `j` ↔ `a[j]` bit `k`).
fn transpose64(a: &mut [u64; 64]) {
    let masks = [
        (32, 0x0000_0000_FFFF_FFFFu64),
        (16, 0x0000_FFFF_0000_FFFF),
        (8, 0x00FF_00FF_00FF_00FF),
        (4, 0x0F0F_0F0F_0F0F_0F0F),
        (2, 0x3333_3333_3333_3333),
        (1, 0x5555_5555_5555_5555),
    ];
    for (j, m) in masks {
        for k in (0..64).filter(|k| k & j == 0) {
            let t = ((a[k] >> j) ^ a[k + j]) & m;
            a[k] ^= t << j;
            a[k + j] ^= t;
        }
    }
}

/// Largest rounding of θ (8 ulp) for which a rotation can still be
/// recognised as Clifford; larger angles (|θ| ≳ 5.6·10⁸) are refused,
/// since the multiple of π/2 they are near is no longer determined (from
/// |θ| ≈ 4·10¹⁴ on, every angle passed before this limit).
const MAX_ANGLE_ROUNDING: f64 = 1e-6;

/// How far a rotation angle may lie from a multiple of π/2 and still be
/// applied as a Clifford gate: 10⁻⁹ plus the rounding of θ.
pub fn clifford_tolerance(theta: f64) -> f64 {
    1e-9 + (8.0 * f64::EPSILON * theta.abs()).min(MAX_ANGLE_ROUNDING)
}

fn quarter_turns(theta: f64, what: &str) -> Result<u32> {
    let r = (theta / FRAC_PI_2).round();
    // An absolute tolerance on the angle, plus the rounding of θ itself:
    // huge angles are Clifford only when they are multiples of π/2 to
    // within a few units in the last place.
    let tolerance = clifford_tolerance(theta);
    if !theta.is_finite()
        || 8.0 * f64::EPSILON * theta.abs() > MAX_ANGLE_ROUNDING
        || (theta - r * FRAC_PI_2).abs() > tolerance
    {
        return Err(QsoError::invalid(
            "gate",
            format!("{what}({theta}) is not a Clifford gate (angles must be multiples of π/2)"),
        ));
    }
    Ok((r.rem_euclid(4.0)) as u32)
}

impl StabilizerState {
    /// `|0…0⟩` on `n` qubits.
    pub fn zero(n: usize) -> Result<Self> {
        if n == 0 || n > MAX_STABILIZER_QUBITS {
            return Err(QsoError::TooManyQubits {
                what: "stabilizer simulation (1 or more)",
                max: MAX_STABILIZER_QUBITS,
                requested: n,
            });
        }
        let (words, rw) = (n.div_ceil(64), (2 * n).div_ceil(64));
        let mut s = StabilizerState {
            n,
            words,
            rw,
            x: vec![0; n * rw],
            z: vec![0; n * rw],
            r: vec![0; rw],
        };
        for q in 0..n {
            s.x[q * rw + q / 64] |= 1 << (q % 64); // destabilizer X_q
            s.z[q * rw + (n + q) / 64] |= 1 << ((n + q) % 64); // stabilizer Z_q
        }
        Ok(s)
    }

    /// The 2n generators as rows.
    fn rows(&self) -> Vec<Row> {
        let (n, words, rw) = (self.n, self.words, self.rw);
        let mut rows = vec![Row::zero(words); 2 * n];
        let mut block = [0u64; 64];
        for (cols, part) in [(&self.x, 0), (&self.z, 1)] {
            for rb in 0..rw {
                for cb in 0..words {
                    for (j, b) in block.iter_mut().enumerate() {
                        let q = cb * 64 + j;
                        *b = if q < n { cols[q * rw + rb] } else { 0 };
                    }
                    transpose64(&mut block);
                    for (k, &b) in block.iter().enumerate() {
                        let i = rb * 64 + k;
                        if i < 2 * n {
                            if part == 0 {
                                rows[i].x[cb] = b;
                            } else {
                                rows[i].z[cb] = b;
                            }
                        }
                    }
                }
            }
        }
        for (i, row) in rows.iter_mut().enumerate() {
            row.neg = (self.r[i / 64] >> (i % 64)) & 1 == 1;
        }
        rows
    }

    /// Store the 2n generators `rows`.
    fn set_rows(&mut self, rows: &[Row]) {
        let (n, words, rw) = (self.n, self.words, self.rw);
        let mut block = [0u64; 64];
        for part in 0..2 {
            for rb in 0..rw {
                for cb in 0..words {
                    for (k, b) in block.iter_mut().enumerate() {
                        let i = rb * 64 + k;
                        *b = match rows.get(i) {
                            Some(row) if part == 0 => row.x[cb],
                            Some(row) => row.z[cb],
                            None => 0,
                        };
                    }
                    transpose64(&mut block);
                    let cols = if part == 0 { &mut self.x } else { &mut self.z };
                    for (j, &b) in block.iter().enumerate() {
                        let q = cb * 64 + j;
                        if q < n {
                            cols[q * rw + rb] = b;
                        }
                    }
                }
            }
        }
        self.r.iter_mut().for_each(|w| *w = 0);
        for (i, row) in rows.iter().enumerate() {
            if row.neg {
                self.r[i / 64] |= 1 << (i % 64);
            }
        }
    }

    /// The graph state `|G⟩ = Π_{(u,v)∈E} CZ_{uv} |+⟩^{⊗n}` of an
    /// unweighted graph: stabilized by `K_v = X_v Π_{w∈N(v)} Z_w`.
    pub fn graph_state(graph: &Graph) -> Result<Self> {
        if !graph.is_unweighted() {
            return Err(QsoError::invalid(
                "graph",
                "graph states are defined for unweighted graphs",
            ));
        }
        let mut s = Self::zero(graph.num_nodes())?;
        let n = s.n;
        let mut rows = vec![Row::zero(s.words); 2 * n];
        for v in 0..n {
            rows[v].set(v, false, true); // destabilizer Z_v
            let k = &mut rows[n + v];
            k.set(v, true, false);
            for &(w, _) in graph.neighbors(v) {
                k.set(w, false, true);
            }
        }
        s.set_rows(&rows);
        Ok(s)
    }

    pub fn n_qubits(&self) -> usize {
        self.n
    }

    fn check(&self, q: usize) -> Result<()> {
        if q >= self.n {
            return Err(QsoError::QubitOutOfRange {
                qubit: q,
                n_qubits: self.n,
            });
        }
        Ok(())
    }

    fn check_pair(&self, a: usize, b: usize) -> Result<()> {
        self.check(a)?;
        self.check(b)?;
        if a == b {
            return Err(QsoError::DuplicateQubit {
                gate: "two-qubit Clifford",
                qubit: a,
            });
        }
        Ok(())
    }

    // -- Primitive gates (conjugation rules of Aaronson and Gottesman) ----

    /// Apply `f(x, z, r)` to every word of column `q` and of the signs.
    fn column(&mut self, q: usize, f: impl Fn(&mut u64, &mut u64, &mut u64)) {
        let o = q * self.rw;
        for w in 0..self.rw {
            f(&mut self.x[o + w], &mut self.z[o + w], &mut self.r[w]);
        }
    }

    /// Hadamard.
    pub fn h(&mut self, q: usize) -> Result<&mut Self> {
        self.check(q)?;
        <Self as CliffordTarget>::h(self, q);
        Ok(self)
    }

    /// Phase gate `S = diag(1, i)`.
    pub fn s(&mut self, q: usize) -> Result<&mut Self> {
        self.check(q)?;
        <Self as CliffordTarget>::s(self, q);
        Ok(self)
    }

    /// `S† = diag(1, −i)`.
    pub fn sdg(&mut self, q: usize) -> Result<&mut Self> {
        self.check(q)?;
        <Self as CliffordTarget>::sdg(self, q);
        Ok(self)
    }

    pub fn x(&mut self, q: usize) -> Result<&mut Self> {
        self.check(q)?;
        <Self as CliffordTarget>::x(self, q);
        Ok(self)
    }

    pub fn y(&mut self, q: usize) -> Result<&mut Self> {
        self.check(q)?;
        <Self as CliffordTarget>::y(self, q);
        Ok(self)
    }

    pub fn z(&mut self, q: usize) -> Result<&mut Self> {
        self.check(q)?;
        <Self as CliffordTarget>::z(self, q);
        Ok(self)
    }

    /// `√X` (up to a global phase, `H S H`).
    pub fn sx(&mut self, q: usize) -> Result<&mut Self> {
        self.h(q)?.s(q)?.h(q)
    }

    /// `√X†`.
    pub fn sxdg(&mut self, q: usize) -> Result<&mut Self> {
        self.h(q)?.sdg(q)?.h(q)
    }

    /// CNOT with control `c` and target `t`.
    pub fn cx(&mut self, c: usize, t: usize) -> Result<&mut Self> {
        self.check_pair(c, t)?;
        <Self as CliffordTarget>::cx(self, c, t);
        Ok(self)
    }

    pub fn cz(&mut self, a: usize, b: usize) -> Result<&mut Self> {
        self.check_pair(a, b)?;
        self.h(b)?.cx(a, b)?.h(b)
    }

    pub fn cy(&mut self, c: usize, t: usize) -> Result<&mut Self> {
        self.check_pair(c, t)?;
        self.sdg(t)?.cx(c, t)?.s(t)
    }

    pub fn swap(&mut self, a: usize, b: usize) -> Result<&mut Self> {
        self.check_pair(a, b)?;
        <Self as CliffordTarget>::swap(self, a, b);
        Ok(self)
    }

    /// Apply a Clifford instruction (parameters bound to `theta`); an error
    /// for gates outside the Clifford group (T, rotations by other angles,
    /// most multi-controlled gates, custom unitaries).
    pub fn apply_instruction(&mut self, inst: &Instruction, theta: &[f64]) -> Result<()> {
        validate_instruction(inst, theta, self.n)?;
        apply_clifford(self, inst, theta)
    }

    // -- Measurement -------------------------------------------------------

    /// A Pauli operator as a row; qubits must be in range and distinct.
    fn pauli_row(&self, ops: &[(usize, Pauli)]) -> Result<Row> {
        let mut row = Row::zero(self.words);
        let mut seen = std::collections::BTreeSet::new();
        for &(q, p) in ops {
            self.check(q)?;
            if !seen.insert(q) {
                return Err(QsoError::invalid(
                    "pauli",
                    format!("qubit {q} appears twice"),
                ));
            }
            let (x, z) = match p {
                Pauli::I => (false, false),
                Pauli::X => (true, false),
                Pauli::Y => (true, true),
                Pauli::Z => (false, true),
            };
            row.set(q, x, z);
        }
        Ok(row)
    }

    /// ⟨P⟩ for a Pauli operator: +1, −1 or 0.
    pub fn expectation(&self, ops: &[(usize, Pauli)]) -> Result<f64> {
        let p = self.pauli_row(ops)?;
        let rows = self.rows();
        Ok(match expectation_of(&rows, self.n, &p) {
            None => 0.0,
            Some(true) => -1.0,
            Some(false) => 1.0,
        })
    }

    /// ⟨P⟩ for a Pauli operator written `"X0 Z1"` (optionally signed).
    pub fn expectation_str(&self, pauli: &str) -> Result<f64> {
        let (sign, ops) = parse_pauli(pauli)?;
        Ok(sign * self.expectation(&ops)?)
    }

    /// ⟨H⟩ for a sum of Pauli strings (on qubits below 64).
    pub fn expectation_sum(&self, h: &PauliSum) -> Result<f64> {
        let rows = self.rows();
        let mut total = 0.0;
        for (c, p) in h.terms() {
            let row = self.pauli_row(&pauli_ops(p))?;
            total += c * match expectation_of(&rows, self.n, &row) {
                None => 0.0,
                Some(true) => -1.0,
                Some(false) => 1.0,
            };
        }
        Ok(total)
    }

    /// Measure the Pauli operator `P` (projectively): `true` for the
    /// eigenvalue −1.
    pub fn measure_pauli(&mut self, ops: &[(usize, Pauli)], rng: &mut Rng) -> Result<bool> {
        let p = self.pauli_row(ops)?;
        if p.x.iter().chain(&p.z).all(|&w| w == 0) {
            return Ok(false);
        }
        let n = self.n;
        let mut rows = self.rows();
        let Some(k) = (n..2 * n).find(|&i| rows[i].anticommutes(&p)) else {
            return Ok(stabilizer_sign(&rows, n, &p));
        };
        // Random outcome: make every other row commute with P, then
        // replace the anticommuting stabilizer by ±P.
        let pivot = rows[k].clone();
        for (i, row) in rows.iter_mut().enumerate() {
            // Destabilizer k − n anticommutes with the pivot and is replaced
            // below; every other row commutes with it.
            if i != k && i != k - n && row.anticommutes(&p) {
                row.left_multiply(&pivot);
            }
        }
        let outcome = rng.next_u64() & 1 == 1;
        rows[k - n] = pivot;
        rows[k] = Row { neg: outcome, ..p };
        self.set_rows(&rows);
        Ok(outcome)
    }

    /// Measure qubit `q` in the computational basis.
    pub fn measure(&mut self, q: usize, rng: &mut Rng) -> Result<bool> {
        self.measure_pauli(&[(q, Pauli::Z)], rng)
    }

    /// Reset qubit `q` to `|0⟩` (measure, then flip a 1).
    pub fn reset(&mut self, q: usize, rng: &mut Rng) -> Result<()> {
        if self.measure(q, rng)? {
            self.x(q)?;
        }
        Ok(())
    }

    // -- Sampling and conversion ------------------------------------------

    /// The distribution of measuring every qubit: uniform over an affine
    /// subspace of {0,1}ⁿ.
    pub fn sampler(&self) -> ZSampler {
        let (n, words) = (self.n, self.words);
        // Gaussian elimination of the stabilizers on their X parts; the
        // rows left without X are signed Z strings: parity constraints.
        let mut rows: Vec<Row> = self.rows().split_off(n);
        let mut rank = 0;
        for q in 0..n {
            let Some(j) = (rank..n).find(|&j| rows[j].get(q).0) else {
                continue;
            };
            rows.swap(rank, j);
            let pivot = rows[rank].clone();
            for (m, row) in rows.iter_mut().enumerate() {
                if m != rank && row.get(q).0 {
                    row.left_multiply(&pivot);
                }
            }
            rank += 1;
        }
        // Constraints z·b = sign for rows rank..n, reduced to echelon form.
        let mut eqs: Vec<(Vec<u64>, bool)> =
            rows[rank..].iter().map(|r| (r.z.clone(), r.neg)).collect();
        let mut pivots = Vec::new();
        let mut done = 0;
        for q in 0..n {
            let bit = |v: &Vec<u64>| (v[q / 64] >> (q % 64)) & 1 == 1;
            let Some(j) = (done..eqs.len()).find(|&j| bit(&eqs[j].0)) else {
                continue;
            };
            eqs.swap(done, j);
            let (pz, pr) = eqs[done].clone();
            for (m, (z, r)) in eqs.iter_mut().enumerate() {
                if m != done && bit(z) {
                    for w in 0..words {
                        z[w] ^= pz[w];
                    }
                    *r ^= pr;
                }
            }
            pivots.push(q);
            done += 1;
        }
        let set = |v: &mut Vec<u64>, q: usize| v[q / 64] ^= 1 << (q % 64);
        let mut offset = vec![0u64; words];
        for (k, &q) in pivots.iter().enumerate() {
            if eqs[k].1 {
                set(&mut offset, q);
            }
        }
        let is_pivot: Vec<bool> = {
            let mut v = vec![false; n];
            for &q in &pivots {
                v[q] = true;
            }
            v
        };
        let mut basis = Vec::new();
        for f in (0..n).filter(|&f| !is_pivot[f]) {
            let mut v = vec![0u64; words];
            set(&mut v, f);
            for (k, &q) in pivots.iter().enumerate() {
                if (eqs[k].0[f / 64] >> (f % 64)) & 1 == 1 {
                    set(&mut v, q);
                }
            }
            basis.push(v);
        }
        ZSampler { n, offset, basis }
    }

    /// Measure every qubit `shots` times (on copies of the state):
    /// `{bitstring: count}` with qubit n−1 first. Shots × qubits must be at
    /// most [`MAX_SAMPLED_BITS`].
    pub fn sample_counts(&self, shots: usize, rng: &mut Rng) -> Result<BTreeMap<String, usize>> {
        if shots.saturating_mul(self.n) > MAX_SAMPLED_BITS {
            return Err(QsoError::invalid(
                "shots",
                format!("shots × qubits must be at most {MAX_SAMPLED_BITS}"),
            ));
        }
        let sampler = self.sampler();
        let mut counts = BTreeMap::new();
        for _ in 0..shots {
            *counts.entry(sampler.sample_string(rng)).or_default() += 1;
        }
        Ok(counts)
    }

    /// The state vector (up to a global phase) for at most
    /// [`MAX_STATEVECTOR_QUBITS`] qubits.
    pub fn to_statevector(&self) -> Result<StateVector> {
        let n = self.n;
        if n > MAX_STATEVECTOR_QUBITS {
            return Err(QsoError::TooManyQubits {
                what: "stabilizer to state-vector conversion",
                max: MAX_STATEVECTOR_QUBITS,
                requested: n,
            });
        }
        // Project a basis state in the support onto the stabilized space:
        // |ψ⟩ ∝ Π_i (1 + S_i)/2 |b⟩.
        let start = self.sampler().offset[0] as usize;
        let mut amps = vec![C64::new(0.0, 0.0); 1 << n];
        amps[start] = C64::new(1.0, 0.0);
        for s in &self.rows()[n..] {
            let (xm, zm) = (s.x[0] as usize, s.z[0] as usize);
            let mut phase = C64::new(if s.neg { -1.0 } else { 1.0 }, 0.0);
            for _ in 0..(xm & zm).count_ones() {
                phase *= C64::new(0.0, 1.0);
            }
            let mut out = amps.clone();
            for (v, a) in amps.iter().enumerate() {
                if a.norm_sqr() == 0.0 {
                    continue;
                }
                let sign = if (zm & v).count_ones() % 2 == 1 {
                    -1.0
                } else {
                    1.0
                };
                out[v ^ xm] += phase * sign * a;
            }
            amps = out;
        }
        StateVector::from_amplitudes(amps)
    }

    /// Stabilizer generators as signed Pauli strings, `"+X0 Z1"`.
    pub fn stabilizers(&self) -> Vec<String> {
        self.rows()[self.n..]
            .iter()
            .map(|r| row_string(r, self.n))
            .collect()
    }

    /// Entanglement entropy (in bits) between the qubits `part` and the
    /// rest: `rank(generators restricted to part) − |part|` (Fattal et
    /// al., arXiv:quant-ph/0406168). For graph states it equals the
    /// cut-rank of the adjacency matrix over GF(2).
    pub fn entanglement_entropy(&self, part: &[usize]) -> Result<usize> {
        let mut seen = vec![false; self.n];
        for &q in part {
            self.check(q)?;
            if std::mem::replace(&mut seen[q], true) {
                return Err(QsoError::invalid(
                    "part",
                    format!("qubit {q} appears twice"),
                ));
            }
        }
        // Each generator restricted to `part`: 2|part| bits.
        let k = part.len();
        let words = (2 * k).div_ceil(64).max(1);
        let mut m: Vec<Vec<u64>> = self.rows()[self.n..]
            .iter()
            .map(|r| {
                let mut v = vec![0u64; words];
                for (j, &q) in part.iter().enumerate() {
                    let (x, z) = r.get(q);
                    if x {
                        v[j / 64] |= 1 << (j % 64);
                    }
                    if z {
                        v[(k + j) / 64] |= 1 << ((k + j) % 64);
                    }
                }
                v
            })
            .collect();
        Ok(gf2_rank(&mut m, 2 * k) - k)
    }

    /// Whether `other` is the same state (up to a global phase): every
    /// stabilizer of `other` stabilizes `self`.
    pub fn same_state(&self, other: &StabilizerState) -> bool {
        if self.n != other.n {
            return false;
        }
        let mine = self.rows();
        other.rows()[other.n..]
            .iter()
            .all(|g| expectation_of(&mine, self.n, g) == Some(false))
    }

    /// Local complementation at node `v` of a graph state: the local
    /// Clifford `√(−iX_v) Π_{w∈N(v)} √(iZ_w)` (Hein, Eisert and Briegel,
    /// Phys. Rev. A 69, 062311 (2004)) maps `|G⟩` to `|G * v⟩`.
    pub fn local_complement(&mut self, graph: &Graph, v: usize) -> Result<()> {
        if graph.num_nodes() != self.n || v >= self.n {
            return Err(QsoError::invalid(
                "graph",
                "the graph must have one node per qubit, and v must be one of them",
            ));
        }
        // √(−iX) = RX(π/2); √(iZ) = RZ(−π/2) = S† up to phase.
        rx_quarters(self, v, 1);
        for &(w, _) in graph.neighbors(v) {
            self.sdg(w)?;
        }
        Ok(())
    }
}

/// The Clifford primitives of a simulator: every supported gate is
/// decomposed into them (up to global phases). Qubits are validated
/// before; the primitives cannot fail.
pub(crate) trait CliffordTarget {
    fn h(&mut self, q: usize);
    fn s(&mut self, q: usize);
    fn sdg(&mut self, q: usize);
    fn x(&mut self, q: usize);
    fn y(&mut self, q: usize);
    fn z(&mut self, q: usize);
    fn cx(&mut self, c: usize, t: usize);
    fn swap(&mut self, a: usize, b: usize);
}

impl CliffordTarget for StabilizerState {
    fn h(&mut self, q: usize) {
        self.column(q, |x, z, r| {
            *r ^= *x & *z;
            std::mem::swap(x, z);
        });
    }

    fn s(&mut self, q: usize) {
        self.column(q, |x, z, r| {
            *r ^= *x & *z;
            *z ^= *x;
        });
    }

    fn sdg(&mut self, q: usize) {
        self.column(q, |x, z, r| {
            *r ^= *x & !*z;
            *z ^= *x;
        });
    }

    fn x(&mut self, q: usize) {
        self.column(q, |_, z, r| *r ^= *z);
    }

    fn y(&mut self, q: usize) {
        self.column(q, |x, z, r| *r ^= *x ^ *z);
    }

    fn z(&mut self, q: usize) {
        self.column(q, |x, _, r| *r ^= *x);
    }

    fn cx(&mut self, c: usize, t: usize) {
        let (oc, ot) = (c * self.rw, t * self.rw);
        for w in 0..self.rw {
            let (xc, zc) = (self.x[oc + w], self.z[oc + w]);
            let (xt, zt) = (self.x[ot + w], self.z[ot + w]);
            self.r[w] ^= xc & zt & !(xt ^ zc);
            self.x[ot + w] = xt ^ xc;
            self.z[oc + w] = zc ^ zt;
        }
    }

    fn swap(&mut self, a: usize, b: usize) {
        let (oa, ob) = (a * self.rw, b * self.rw);
        for w in 0..self.rw {
            self.x.swap(oa + w, ob + w);
            self.z.swap(oa + w, ob + w);
        }
    }
}

fn rz_quarters<T: CliffordTarget>(t: &mut T, q: usize, k: u32) {
    match k {
        1 => t.s(q),
        2 => t.z(q),
        3 => t.sdg(q),
        _ => {}
    }
}

fn rx_quarters<T: CliffordTarget>(t: &mut T, q: usize, k: u32) {
    if k != 0 {
        t.h(q);
        rz_quarters(t, q, k);
        t.h(q);
    }
}

fn ry_quarters<T: CliffordTarget>(t: &mut T, q: usize, k: u32) {
    // RY(θ) = S RX(θ) S†.
    if k != 0 {
        t.sdg(q);
        rx_quarters(t, q, k);
        t.s(q);
    }
}

fn rzz_quarters<T: CliffordTarget>(t: &mut T, a: usize, b: usize, k: u32) {
    if k != 0 {
        t.cx(a, b);
        rz_quarters(t, b, k);
        t.cx(a, b);
    }
}

/// Apply a validated instruction through the primitives of `sim`.
pub(crate) fn apply_clifford<T: CliffordTarget>(
    sim: &mut T,
    inst: &Instruction,
    theta: &[f64],
) -> Result<()> {
    use GateKind::*;
    let a = inst.angles(theta);
    let t = &inst.targets;
    let name = inst.kind.name();
    match (inst.kind, inst.controls.as_slice()) {
        (H, []) => sim.h(t[0]),
        (X, []) => sim.x(t[0]),
        (Y, []) => sim.y(t[0]),
        (Z, []) => sim.z(t[0]),
        (S, []) => sim.s(t[0]),
        (Sdg, []) => sim.sdg(t[0]),
        (SX, []) => rx_quarters(sim, t[0], 1),
        (SXdg, []) => rx_quarters(sim, t[0], 3),
        (Swap, []) => sim.swap(t[0], t[1]),
        (X, [c]) => sim.cx(*c, t[0]),
        (Y, [c]) => {
            sim.sdg(t[0]);
            sim.cx(*c, t[0]);
            sim.s(t[0]);
        }
        (Z, [c]) => {
            sim.h(t[0]);
            sim.cx(*c, t[0]);
            sim.h(t[0]);
        }
        (RZ | P, []) => rz_quarters(sim, t[0], quarter_turns(a[0], name)?),
        (RX, []) => rx_quarters(sim, t[0], quarter_turns(a[0], name)?),
        (RY, []) => ry_quarters(sim, t[0], quarter_turns(a[0], name)?),
        (RZZ, []) => rzz_quarters(sim, t[0], t[1], quarter_turns(a[0], name)?),
        (RXX, []) => {
            let k = quarter_turns(a[0], name)?;
            sim.h(t[0]);
            sim.h(t[1]);
            rzz_quarters(sim, t[0], t[1], k);
            sim.h(t[0]);
            sim.h(t[1]);
        }
        (RYY, []) => {
            // RYY = (S⊗S) RXX (S†⊗S†).
            let k = quarter_turns(a[0], name)?;
            for &q in &t[..2] {
                sim.sdg(q);
                sim.h(q);
            }
            rzz_quarters(sim, t[0], t[1], k);
            for &q in &t[..2] {
                sim.h(q);
                sim.s(q);
            }
        }
        (U, []) => {
            // U(θ, φ, λ) = RZ(φ) RY(θ) RZ(λ) up to a global phase.
            let (kt, kp, kl) = (
                quarter_turns(a[0], "U θ")?,
                quarter_turns(a[1], "U φ")?,
                quarter_turns(a[2], "U λ")?,
            );
            rz_quarters(sim, t[0], kl);
            ry_quarters(sim, t[0], kt);
            rz_quarters(sim, t[0], kp);
        }
        // Controlled phases: CP(0) = I, CP(π) = CZ.
        (P, [c]) => match quarter_turns(a[0], "controlled P")? {
            0 => {}
            2 => {
                sim.h(t[0]);
                sim.cx(*c, t[0]);
                sim.h(t[0]);
            }
            _ => return Err(not_clifford(inst)),
        },
        _ => return Err(not_clifford(inst)),
    }
    Ok(())
}

/// Check an instruction as the state-vector simulator does: arity, qubits
/// in range and distinct, and parameters bound to finite values.
pub(crate) fn validate_instruction(inst: &Instruction, theta: &[f64], n: usize) -> Result<()> {
    inst.validate(n)?;
    for p in &inst.params {
        if let crate::quantum::Param::Symbol(s) = p
            && s.index >= theta.len()
        {
            return Err(QsoError::ParameterCount {
                expected: s.index + 1,
                actual: theta.len(),
            });
        }
    }
    if theta.iter().any(|t| !t.is_finite()) {
        return Err(QsoError::invalid("theta", "parameters must be finite"));
    }
    Ok(())
}

/// The local complement `G * v`: the neighbourhood of `v` is
/// complemented (edges between neighbours of `v` are toggled).
pub fn local_complementation(graph: &Graph, v: usize) -> Result<Graph> {
    let n = graph.num_nodes();
    if v >= n {
        return Err(QsoError::NodeOutOfRange {
            node: v,
            n_nodes: n,
        });
    }
    if !graph.is_unweighted() {
        return Err(QsoError::invalid(
            "graph",
            "local complementation is defined for unweighted graphs",
        ));
    }
    let nbrs: Vec<usize> = graph.neighbors(v).iter().map(|e| e.0).collect();
    let mut edges: std::collections::BTreeSet<(usize, usize)> = graph
        .edges()
        .iter()
        .map(|&(a, b, _)| (a.min(b), a.max(b)))
        .collect();
    for (i, &a) in nbrs.iter().enumerate() {
        for &b in &nbrs[i + 1..] {
            let e = (a.min(b), a.max(b));
            if !edges.remove(&e) {
                edges.insert(e);
            }
        }
    }
    let edges: Vec<(usize, usize)> = edges.into_iter().collect();
    Ok(Graph::from_edges(n, &edges)?.with_name(format!("{} * {v}", graph.name())))
}

/// The sign of the product of stabilizers that equals `±p`, for `p` in
/// the stabilizer group up to sign (`rows`: destabilizers, stabilizers).
fn stabilizer_sign(rows: &[Row], n: usize, p: &Row) -> bool {
    let mut acc = Row::zero(p.x.len());
    for i in 0..n {
        if rows[i].anticommutes(p) {
            acc.left_multiply(&rows[n + i]);
        }
    }
    debug_assert_eq!((&acc.x, &acc.z), (&p.x, &p.z));
    acc.neg
}

/// `None` if `p` anticommutes with a stabilizer (⟨P⟩ = 0); otherwise
/// whether `−P` (rather than `P`) is in the stabilizer group.
fn expectation_of(rows: &[Row], n: usize, p: &Row) -> Option<bool> {
    if rows[n..].iter().any(|s| s.anticommutes(p)) {
        return None;
    }
    Some(stabilizer_sign(rows, n, p) != p.neg)
}

pub(crate) fn not_clifford(inst: &Instruction) -> QsoError {
    QsoError::invalid(
        "gate",
        format!(
            "{} with {} control(s) is not a Clifford gate the stabilizer simulator supports",
            inst.kind.name(),
            inst.controls.len()
        ),
    )
}

fn row_string(r: &Row, n: usize) -> String {
    let mut out = String::from(if r.neg { "-" } else { "+" });
    let mut first = true;
    for q in 0..n {
        let p = match r.get(q) {
            (false, false) => continue,
            (true, false) => 'X',
            (true, true) => 'Y',
            (false, true) => 'Z',
        };
        if !first {
            out.push(' ');
        }
        out.push_str(&format!("{p}{q}"));
        first = false;
    }
    if first {
        out.push('I');
    }
    out
}

/// Rank over GF(2) of the rows of `m` (`bits` columns); `m` is reduced.
pub(crate) fn gf2_rank(m: &mut [Vec<u64>], bits: usize) -> usize {
    let mut rank = 0;
    for c in 0..bits {
        let (w, b) = (c / 64, 1u64 << (c % 64));
        let Some(j) = (rank..m.len()).find(|&j| m[j][w] & b != 0) else {
            continue;
        };
        m.swap(rank, j);
        let pivot = m[rank].clone();
        for (i, row) in m.iter_mut().enumerate() {
            if i != rank && row[w] & b != 0 {
                for (a, p) in row.iter_mut().zip(&pivot) {
                    *a ^= p;
                }
            }
        }
        rank += 1;
    }
    rank
}

/// The non-identity factors of a Pauli string.
fn pauli_ops(p: &PauliString) -> Vec<(usize, Pauli)> {
    (0..64)
        .map(|q| (q, p.get(q)))
        .filter(|&(_, p)| p != Pauli::I)
        .collect()
}

/// Parse `"[+|-] X0 Z1 Y12"` (also `"I"`); returns the sign and factors.
pub fn parse_pauli(text: &str) -> Result<(f64, Vec<(usize, Pauli)>)> {
    let t = text.trim();
    let (sign, rest) = match t.strip_prefix('-') {
        Some(r) => (-1.0, r),
        None => (1.0, t.strip_prefix('+').unwrap_or(t)),
    };
    let mut ops = Vec::new();
    for tok in rest.split_whitespace() {
        if tok == "I" {
            continue;
        }
        let mut chars = tok.chars();
        let p = match chars.next() {
            Some('X') => Pauli::X,
            Some('Y') => Pauli::Y,
            Some('Z') => Pauli::Z,
            _ => return Err(QsoError::Parse(format!("bad Pauli factor '{tok}'"))),
        };
        let digits = chars.as_str();
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return Err(QsoError::Parse(format!("bad Pauli factor '{tok}'")));
        }
        let q: usize = digits
            .parse()
            .map_err(|_| QsoError::Parse(format!("bad Pauli factor '{tok}'")))?;
        ops.push((q, p));
    }
    Ok((sign, ops))
}

/// Outcomes of measuring every qubit of a stabilizer state: uniform over
/// `offset ⊕ span(basis)`.
#[derive(Clone, Debug)]
pub struct ZSampler {
    n: usize,
    offset: Vec<u64>,
    basis: Vec<Vec<u64>>,
}

impl ZSampler {
    /// Number of random bits: the outcome has `2^k` equally likely values.
    pub fn random_bits(&self) -> usize {
        self.basis.len()
    }

    /// One outcome, bit `q` of word `q / 64` for qubit `q`.
    pub fn sample_words(&self, rng: &mut Rng) -> Vec<u64> {
        let mut v = self.offset.clone();
        let mut bits = 0u64;
        for (k, b) in self.basis.iter().enumerate() {
            if k % 64 == 0 {
                bits = rng.next_u64();
            }
            if (bits >> (k % 64)) & 1 == 1 {
                for (a, c) in v.iter_mut().zip(b) {
                    *a ^= c;
                }
            }
        }
        v
    }

    /// One outcome as a bit string, qubit n−1 first.
    pub fn sample_string(&self, rng: &mut Rng) -> String {
        let v = self.sample_words(rng);
        (0..self.n)
            .rev()
            .map(|q| {
                if (v[q / 64] >> (q % 64)) & 1 == 1 {
                    '1'
                } else {
                    '0'
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
